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

    void Run(RenderContext* pRenderContext);

    constexpr bool IsActive() const { return m_Active; }

    void SetGeneratePhotonsVars(const ShaderVar& var) const;

private:
    ref<Device> m_Device;
    LightBVHBuilder m_LightBvhBuilder;
    LightBVH m_LightBvh;

    ref<Buffer> m_ClusterBuf;
    ref<Buffer> m_ClusterBufCPU;
    ref<Buffer> m_ClusterCdfBuf;
    ref<Buffer> m_ClusterCdfBufCPU;

    bool m_Active = false;
    uint m_MaxCutSize = 32;
};
