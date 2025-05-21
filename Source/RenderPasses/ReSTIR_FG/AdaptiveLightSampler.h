#pragma once

#include <Falcor.h>
#include <Rendering/Lights/LightBVHBuilder.h>
#include <Utils/Math/ScalarTypes.h>

using namespace Falcor;

class AdaptiveLightSampler
{
public:
    AdaptiveLightSampler(ref<Device> device);

    void SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene);
    void PrepareBuffers(RenderContext* renderContext);
    bool RenderUI(Gui::Widgets& widget);

    constexpr bool IsActive() const { return m_Active; }

private:
    ref<Device> m_Device;
    LightBVHBuilder m_LightBvhBuilder;
    LightBVH m_LightBvh;
    ref<Buffer> m_ClusterBuffer;

    bool m_Active = false;
    uint m_MaxCutSize = 32;
};
