#pragma once

#include <Falcor.h>
#include <Rendering/Lights/LightBVH.h>
#include <Utils/Math/ScalarTypes.h>

using namespace Falcor;

class AdaptiveLightSampler
{
public:
    AdaptiveLightSampler(ref<Device> device);

    void SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene);
    void PrepareBuffers(RenderContext* renderContext);
    void RenderUI(Gui::Widgets& widget);

private:
    ref<Device> m_Device;
    LightBVH m_LightBvh;

    ref<Buffer> m_ClusterBuffer;

    uint m_MaxCutSize = 32;
};
