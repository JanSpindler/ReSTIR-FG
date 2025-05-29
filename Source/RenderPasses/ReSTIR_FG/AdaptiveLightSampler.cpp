#include "AdaptiveLightSampler.h"
#include <span>

AdaptiveLightSampler::AdaptiveLightSampler(ref<Device> device)
    : m_Device(device), m_LightBvh(device, {}), m_LightBvhBuilder(LightBVHBuilder::Options())
{}

void AdaptiveLightSampler::SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Reset buffers
    m_ClusterNodeIdxBuf.reset();
    m_ClusterCdfBuf.reset();
    m_ClusterStatsBuf.reset();
    m_LeafRadianceBuf.reset();
    m_NodeClusterMapBuf.reset();
    m_PhotonLeafMapBuf[0].reset();
    m_PhotonLeafMapBuf[1].reset();

    // Build tree
    const auto lightCollection = pScene->getLightCollection(pRenderContext);
    m_LightBvh = LightBVH(m_Device, lightCollection);
    if (lightCollection->getTotalLightCount() > 0)
    {
        return;
    }
    m_LightBvhBuilder.build(pRenderContext, m_LightBvh);
    FALCOR_ASSERT(m_LightBvh.isValid());

    // Node cluster map
    m_ClusterNodeMap.resize(m_MaxCutSize);
    m_ClusterStats.resize(m_MaxCutSize);
    UpdateNodeClusterMap(pRenderContext);
}

void AdaptiveLightSampler::PrepareBuffers(RenderContext* pRenderContext, const uint2 screenSize, const uint2 photonCounts)
{
    if (!m_ClusterNodeIdxBuf)
    {
        const std::vector<uint> clusterNodeIndices(m_MaxCutSize, 0);
        m_ClusterNodeIdxBuf = Buffer::create(
            m_Device, m_MaxCutSize * sizeof(uint), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, clusterNodeIndices.data()
        );
        m_ClusterNodeIdxBufCPU = Buffer::create(m_Device, m_MaxCutSize * sizeof(uint), ResourceBindFlags::None, Buffer::CpuAccess::Write);
    }

    if (!m_ClusterCdfBuf)
    {
        std::vector<float> clusterCDF(m_MaxCutSize, 0.0f);
        for (size_t i = 0; i < m_ClusterCount; ++i)
        {
            clusterCDF[i] = static_cast<float>(i + 1) / static_cast<float>(m_ClusterCount);
        }
        m_ClusterCdfBuf = Buffer::create(
            m_Device, sizeof(float) * m_MaxCutSize, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, clusterCDF.data()
        );
        m_ClusterCdfBufCPU = Buffer::create(m_Device, sizeof(float) * m_MaxCutSize, ResourceBindFlags::None, Buffer::CpuAccess::Write);
    }

    if (!m_ClusterStatsBuf)
    {
        m_ClusterStatsBuf = Buffer::createStructured(
            m_Device, sizeof(ClusterStats), m_MaxCutSize, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
        );
        m_ClusterStatsBufCPU =
            Buffer::createStructured(m_Device, sizeof(ClusterStats), m_MaxCutSize, ResourceBindFlags::None, Buffer::CpuAccess::Read);
    }

    const size_t nodeCount = std::max<size_t>(1, GetTotalNodeCount());
    if (!m_LeafRadianceBuf)
    {
        // Allocate memory for all nodes even when only using leaf nodes because of simpler indexing
        m_LeafRadianceBuf =
            Buffer::create(m_Device, sizeof(float) * nodeCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
        m_LeafRadianceBufCPU = Buffer::create(m_Device, sizeof(float) * nodeCount, ResourceBindFlags::None, Buffer::CpuAccess::Read);
    }

    if (!m_NodeClusterMapBuf)
    {
        m_NodeClusterMapBuf =
            Buffer::create(m_Device, sizeof(uint) * nodeCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
        m_NodeClusterMapBufCPU = Buffer::create(m_Device, sizeof(uint) * nodeCount, ResourceBindFlags::None, Buffer::CpuAccess::Write);
    }

    for (size_t idx = 0; idx < 2; ++idx)
    {
        if (!m_PhotonLeafMapBuf[idx])
        {
            m_PhotonLeafMapBuf[idx] = Buffer::create(
                m_Device, sizeof(uint) * photonCounts[idx], ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
            );
        }
    }
}

bool AdaptiveLightSampler::RenderUI(Gui::Widgets& widget)
{
    bool changed = false;

    if (auto group = widget.group("Adaptive Light Sampler"))
    {
        // Active
        changed |= group.checkbox("Adaptive Light Sampler", m_Active);

        // TODO: Rebuild if builder params changed?
        if (group.group("Light BVH Builder"))
        {
            m_LightBvhBuilder.renderUI(widget);
        }
        if (group.group("Light BVH"))
        {
            m_LightBvh.renderUI(widget);
        }
    }

    return changed;
}

void AdaptiveLightSampler::Run(RenderContext* pRenderContext)
{
    // If clustering changed, update leaf cluster map
    if (false)
    {
        UpdateNodeClusterMap(pRenderContext);
    }
}

void AdaptiveLightSampler::SetGeneratePhotonsVars(const ShaderVar& var) const
{
    m_LightBvh.setShaderData(var["gLightBVH"]);
    var["gLightClusterNodeIndices"] = m_ClusterNodeIdxBuf;
    var["gLightClusterCdf"] = m_ClusterCdfBuf;
    var["AdaptiveLightSampler"]["gLightClusterCount"] = m_ClusterCount;
}

void AdaptiveLightSampler::SetCollectPhotonsVars(const ShaderVar& var) const
{
    var["gClusterStats"] = m_ClusterStatsBuf;
    var["gLeafRadiance"] = m_LeafRadianceBuf;
}

void AdaptiveLightSampler::ClearClusterStatBuf(RenderContext* pRenderContext) const
{
    pRenderContext->clearUAV(m_ClusterStatsBuf->getUAV().get(), float4(0.0f));
}

void AdaptiveLightSampler::ClearLeafRadianceBuf(RenderContext* pRenderContext) const
{
    pRenderContext->clearUAV(m_LeafRadianceBuf->getUAV().get(), float4(0.0f));
}

void AdaptiveLightSampler::UpdateNodeClusterMap(RenderContext* pRenderContext)
{
    // Map cpu buffer
    std::span<uint> nodeClusterMap(
        reinterpret_cast<uint*>(m_NodeClusterMapBufCPU->map(Buffer::MapType::WriteDiscard)), GetTotalNodeCount()
    );

    // Set all indices to default invalid (0xFFFFFFFF)
    const LightBVH::NodeFunction setInvalid = [&](const LightBVH::NodeLocation& nodeLoc)
    {
        nodeClusterMap[nodeLoc.nodeIndex] = std::numeric_limits<uint>::max();
        return true;
    };
    m_LightBvh.traverseBVH(setInvalid, setInvalid, 0);

    // For each cluster set the corresponding reference of the children
    for (const uint clusterNodeIdx : m_ClusterNodeMap)
    {
        const LightBVH::NodeFunction setClusterMap = [&](const LightBVH::NodeLocation& nodeLoc)
        {
            nodeClusterMap[nodeLoc.nodeIndex] = clusterNodeIdx;
            return true;
        };
        m_LightBvh.traverseBVH(setClusterMap, setClusterMap, clusterNodeIdx);
    }

    // Copy to gpu buffer
    pRenderContext->copyBufferRegion(m_NodeClusterMapBuf.get(), 0, m_NodeClusterMapBufCPU.get(), 0, nodeClusterMap.size_bytes());

    // Unmap cpu buffer
    m_NodeClusterMapBufCPU->unmap();
}
