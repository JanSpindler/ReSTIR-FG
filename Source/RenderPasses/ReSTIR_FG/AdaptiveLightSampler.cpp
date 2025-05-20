#include "AdaptiveLightSampler.h"
#include <Rendering/Lights/LightBVHBuilder.h>

struct Cluster
{
    uint nodeIdx;
    float mean;
    float variance;
};

AdaptiveLightSampler::AdaptiveLightSampler(ref<Device> device) : m_Device(device), m_LightBvh(device, {}) {}

void AdaptiveLightSampler::SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    m_LightBvh = LightBVH(m_Device, pScene->getLightCollection(pRenderContext));
    LightBVHBuilder(LightBVHBuilder::Options()).build(pRenderContext, m_LightBvh);
    FALCOR_ASSERT(m_LightBvh.isValid());
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
    m_LightBvh.renderUI(widget);
}
