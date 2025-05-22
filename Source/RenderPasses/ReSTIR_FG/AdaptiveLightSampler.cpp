#include "AdaptiveLightSampler.h"
#include <span>

struct LightCluster
{
    uint nodeIdx;
    float mean;
    float variance;
};

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
    if (!m_ClusterBuf)
    {
        const LightCluster rootCluster{.nodeIdx = 0, .mean = 0.0f, .variance = 0.0f};
        const std::vector<LightCluster> clusters(m_MaxCutSize, rootCluster);
        m_ClusterBuf = Buffer::createStructured(
            m_Device, sizeof(LightCluster), m_MaxCutSize, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, clusters.data()
        );
        m_ClusterBufCPU =
            Buffer::createStructured(m_Device, sizeof(LightCluster), m_MaxCutSize, ResourceBindFlags::None, Buffer::CpuAccess::Write);
    }

    if (!m_ClusterCdfBuf)
    {
        std::vector<float> clusterCDF(m_MaxCutSize);
        for (size_t i = 0; i < m_MaxCutSize; ++i)
        {
            clusterCDF[i] = static_cast<float>(i + 1) / static_cast<float>(m_MaxCutSize);
        }
        m_ClusterCdfBuf = Buffer::create(
            m_Device, sizeof(float) * m_MaxCutSize, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, clusterCDF.data()
        );
        m_ClusterCdfBuf = Buffer::create(m_Device, sizeof(float) * m_MaxCutSize, ResourceBindFlags::None, Buffer::CpuAccess::Write);
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
