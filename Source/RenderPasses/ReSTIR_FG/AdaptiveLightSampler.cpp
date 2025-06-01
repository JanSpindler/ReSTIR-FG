#include "AdaptiveLightSampler.h"
#include <execution>

AdaptiveLightSampler::AdaptiveLightSampler(ref<Device> device)
    : m_Device(device), m_LightBvh(device, {}), m_LightBvhBuilder(LightBVHBuilder::Options())
{}

void AdaptiveLightSampler::SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Reset buffers
    m_ClusterNodeIdxBuf.reset();
    m_ClusterCdfBuf.reset();
    m_ClusterSampleCountBuf.reset();
    m_ClusterRadianceSqBuf.reset();
    m_LeafRadianceBuf.reset();
    m_NodeClusterMapBuf.reset();
    m_PhotonLeafMapBuf[0].reset();
    m_PhotonLeafMapBuf[1].reset();
    m_NodeImportanceBuf.reset();

    // Build tree
    const auto lightCollection = pScene->getLightCollection(pRenderContext);
    m_LightBvh = LightTree(m_Device, lightCollection);
    if (lightCollection->getTotalLightCount() == 0)
    {
        return;
    }
    m_LightBvhBuilder.build(pRenderContext, reinterpret_cast<LightBVH&>(m_LightBvh));
    FALCOR_ASSERT(m_LightBvh.isValid());

    // Init vector
    m_ClusterNodeIndices.resize(m_MaxCutSize);
    m_ClusterSampleCount.resize(m_MaxCutSize);
    m_ClusterRadiance.reserve(m_MaxCutSize);
    m_ClusterRadianceSq.resize(m_MaxCutSize);
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

    if (!m_ClusterSampleCountBuf)
    {
        m_ClusterSampleCountBuf =
            Buffer::create(m_Device, sizeof(uint) * m_MaxCutSize, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        m_ClusterSampleCountBufCPU =
            Buffer::create(m_Device, sizeof(uint) * m_MaxCutSize, ResourceBindFlags::None, Buffer::CpuAccess::Read);
    }

    if (!m_ClusterRadianceSqBuf)
    {
        m_ClusterRadianceSqBuf = Buffer::create(m_Device, sizeof(float) * m_MaxCutSize);
        m_ClusterRadianceSqBufCPU =
            Buffer::create(m_Device, sizeof(float) * m_MaxCutSize, ResourceBindFlags::None, Buffer::CpuAccess::Read);
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
        UpdateNodeClusterMap(pRenderContext); // Assumes SetScene was already executed
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

    if (!m_NodeImportanceBuf)
    {
        const std::vector<float> nodeImportance(nodeCount, 1.0f);
        m_NodeImportanceBuf = Buffer::create(
            m_Device, sizeof(float) * nodeCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nodeImportance.data()
        );
        m_NodeImportanceBufCPU = Buffer::create(m_Device, sizeof(float) * nodeCount, ResourceBindFlags::None, Buffer::CpuAccess::Write);
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
    //
    const size_t nodeCount = GetTotalNodeCount();

    // Read radiance from leaf nodes to CPU
    pRenderContext->uavBarrier(m_LeafRadianceBuf.get());
    pRenderContext->copyBufferRegion(m_LeafRadianceBufCPU.get(), 0, m_LeafRadianceBuf.get(), 0, m_LeafRadianceBuf->getSize());
    const std::span<float> leafRadiance(reinterpret_cast<float*>(m_LeafRadianceBufCPU->map(Buffer::MapType::Read)), nodeCount);

    // Update node importance on CPU
    std::span<float> nodeImportance(reinterpret_cast<float*>(m_NodeImportanceBufCPU->map(Buffer::MapType::WriteDiscard)), nodeCount);
    std::for_each(
        std::execution::par_unseq, m_ClusterNodeIndices.begin(), m_ClusterNodeIndices.begin() + m_ClusterCount,
        [&](const uint clusterNodeIdx) { m_LightBvh.UpdateNodeImportance(leafRadiance, nodeImportance, clusterNodeIdx); }
    );

    // Copy node importance to GPU
    pRenderContext->copyBufferRegion(m_NodeImportanceBuf.get(), 0, m_NodeClusterMapBufCPU.get(), 0, m_NodeImportanceBufCPU->getSize());

    // Read cluster stats from gpu
    pRenderContext->uavBarrier(m_ClusterSampleCountBuf.get());
    pRenderContext->copyBufferRegion(
        m_ClusterSampleCountBufCPU.get(), 0, m_ClusterSampleCountBuf.get(), 0, m_ClusterSampleCountBuf->getSize()
    );
    const std::span<uint> clusterSampleCounts(reinterpret_cast<uint*>(m_ClusterSampleCountBufCPU->map(Buffer::MapType::Read)), m_ClusterCount);
    pRenderContext->uavBarrier(m_ClusterRadianceSqBuf.get());
    pRenderContext->copyBufferRegion(
        m_ClusterRadianceSqBufCPU.get(), 0, m_ClusterRadianceSqBuf.get(), 0, m_ClusterRadianceSqBuf->getSize()
    );
    const std::span<float> clusterRadianceSq(
        reinterpret_cast<float*>(m_ClusterRadianceSqBufCPU->map(Buffer::MapType::Read)), m_ClusterCount
    );

    // Clustering
    const bool changeClustering = false;

    // Unmap buffers
    m_ClusterRadianceSqBufCPU->unmap();
    m_ClusterSampleCountBufCPU->unmap();
    m_NodeImportanceBufCPU->unmap();
    m_LeafRadianceBufCPU->unmap();

    // If clustering changed, update leaf cluster map
    if (changeClustering)
    {
        UpdateNodeClusterMap(pRenderContext);
    }
}

void AdaptiveLightSampler::SetGeneratePhotonsVars(const ShaderVar& var) const
{
    m_LightBvh.setShaderData(var["gLightBVH"]);
    var["gLightClusterNodeIndices"] = m_ClusterNodeIdxBuf;
    var["gLightClusterCdf"] = m_ClusterCdfBuf;
    var["gPhotonLeafMap"][0ull] = m_PhotonLeafMapBuf[0];
    var["gPhotonLeafMap"][1ull] = m_PhotonLeafMapBuf[1];
    var["gNodeImportance"] = m_NodeImportanceBuf;
    var["AdaptiveLightSampler"]["gLightClusterCount"] = m_ClusterCount;
}

void AdaptiveLightSampler::SetCollectPhotonsVars(const ShaderVar& var) const
{
    var["gClusterSampleCount"] = m_ClusterSampleCountBuf;
    var["gClusterRadianceSq"] = m_ClusterRadianceSqBuf;
    var["gLeafRadiance"] = m_LeafRadianceBuf;
    var["gNodeClusterMap"] = m_NodeClusterMapBuf;
    var["gPhotonLeafMap"][0ull] = m_PhotonLeafMapBuf[0];
    var["gPhotonLeafMap"][1ull] = m_PhotonLeafMapBuf[1];
}

void AdaptiveLightSampler::ClearClusterStatBuf(RenderContext* pRenderContext) const
{
    pRenderContext->clearUAV(m_ClusterSampleCountBuf->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(m_ClusterRadianceSqBuf->getUAV().get(), float4(0.0f));
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
    for (size_t clusterIdx = 0; clusterIdx < m_ClusterCount; ++clusterIdx)
    {
        const LightBVH::NodeFunction setClusterMap = [&](const LightBVH::NodeLocation& nodeLoc)
        {
            nodeClusterMap[nodeLoc.nodeIndex] = clusterIdx;
            return true;
        };
        m_LightBvh.traverseBVH(setClusterMap, setClusterMap, clusterIdx);
    }

    // Copy to gpu buffer
    pRenderContext->copyBufferRegion(m_NodeClusterMapBuf.get(), 0, m_NodeClusterMapBufCPU.get(), 0, nodeClusterMap.size_bytes());

    // Unmap cpu buffer
    m_NodeClusterMapBufCPU->unmap();
}

void AdaptiveLightSampler::LightTree::UpdateNodeImportance(
    const std::span<float>& leafRadiance, std::span<float>& nodeImportance, const uint nodeIdx)
{
    if (mNodes.empty())
    {
        return;
    }

    FALCOR_ASSERT(leafRadiance.size() == mNodes.size());
    FALCOR_ASSERT(nodeImportance.size() == mNodes.size());

    // Recursive lambda for post-order traversal.
    // Captures 'this' to access mNodes, and spans by reference.
    // The 'auto& self' parameter is a common pattern for recursive lambdas.
    auto calculateImportanceRecursive =
        [&](auto& self, uint32_t nodeIndex) -> float
    {
        FALCOR_ASSERT(nodeIndex < mNodes.size());
        const PackedNode& currentNode = mNodes[nodeIndex];

        if (currentNode.isLeaf())
        {
            // For leaf nodes, importance is its radiance.
            // leafRadiance and nodeImportance are indexed by nodeIndex.
            FALCOR_ASSERT(nodeIndex < leafRadiance.size());
            FALCOR_ASSERT(nodeIndex < nodeImportance.size());
            nodeImportance[nodeIndex] = leafRadiance[nodeIndex];
            return nodeImportance[nodeIndex];
        }
        else
        {
            // Internal node. Calculate importance as the sum of children's importances.
            // Left child is always at nodeIndex + 1.
            uint32_t leftChildIndex = nodeIndex + 1;
            FALCOR_ASSERT(leftChildIndex < mNodes.size()); // Should hold for a valid BVH
            float leftImportance = self(self, leftChildIndex);

            // Right child offset is relative to current node's index.
            uint32_t rightChildIndex = currentNode.getInternalNode().rightChildIdx;
            FALCOR_ASSERT(rightChildIndex < mNodes.size()); // Should hold for a valid BVH
            float rightImportance = self(self, rightChildIndex);

            FALCOR_ASSERT(nodeIndex < nodeImportance.size());
            nodeImportance[nodeIndex] = leftImportance + rightImportance;
            return nodeImportance[nodeIndex];
        }
    };

    // Start recursion from the root node (index 0).
    // If mNodes is not empty, node 0 must exist.
    calculateImportanceRecursive(calculateImportanceRecursive, nodeIdx);
}
