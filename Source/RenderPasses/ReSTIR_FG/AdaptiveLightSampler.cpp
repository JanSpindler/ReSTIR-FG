#include "AdaptiveLightSampler.h"

struct Cluster
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
    if (!m_ClusterBuffer)
    {
        m_ClusterBuffer = Buffer::createStructured(
            m_Device, sizeof(Cluster), m_MaxCutSize, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
    }
}

void AdaptiveLightSampler::RenderUI(Gui::Widgets& widget)
{
    if (auto group = widget.group("Adaptive Light Sampler"))
    {
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
}
