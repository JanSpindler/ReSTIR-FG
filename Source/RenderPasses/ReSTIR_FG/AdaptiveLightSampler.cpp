#include "AdaptiveLightSampler.h"
#include <span>

AdaptiveLightSampler::AdaptiveLightSampler(ref<Device> device)
    : m_Device(device), m_LightBvh(device, {}), m_LightBvhBuilder(LightBVHBuilder::Options())
{}

void AdaptiveLightSampler::SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    const auto lightCollection = pScene->getLightCollection(pRenderContext);
    if (lightCollection->getTotalLightCount() > 0)
    {
        m_LightBvh = LightBVH(m_Device, lightCollection);
        m_LightBvhBuilder.build(pRenderContext, m_LightBvh);
        FALCOR_ASSERT(m_LightBvh.isValid());
    }
}

void AdaptiveLightSampler::PrepareBuffers(RenderContext* pRenderContext)
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
    // TODO
}

void AdaptiveLightSampler::SetGeneratePhotonsVars(const ShaderVar& var) const
{
    if (!m_Active)
    {
        return;
    }

    m_LightBvh.setShaderData(var["gLightBVH"]);
    var["gLightClusterNodeIndices"] = m_ClusterNodeIdxBuf;
    var["gLightClusterCdf"] = m_ClusterCdfBuf;
    var["AdaptiveLightSampler"]["gLightClusterCount"] = m_ClusterCount;
}
