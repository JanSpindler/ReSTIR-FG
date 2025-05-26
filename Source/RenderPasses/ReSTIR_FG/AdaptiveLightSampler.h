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
    void PrepareBuffers(RenderContext* renderContext, const uint2 screenSize);
    bool RenderUI(Gui::Widgets& widget);

    void Run(RenderContext* pRenderContext);

    constexpr bool IsActive() const { return m_Active; }

    void SetGeneratePhotonsVars(const ShaderVar& var) const;
    void SetCollectPhotonsVars(const ShaderVar& var) const;
    void ClearRadianceInfoBuf(RenderContext* pRenderContext) const;

private:
    ref<Device> m_Device;
    LightBVHBuilder m_LightBvhBuilder;
    LightBVH m_LightBvh;

    ref<Buffer> m_ClusterNodeIdxBuf;
    ref<Buffer> m_ClusterNodeIdxBufCPU;
    ref<Buffer> m_ClusterCdfBuf;
    ref<Buffer> m_ClusterCdfBufCPU;
    ref<Buffer> m_RadianceInfoBuf[2];
    ref<Buffer> m_RadianceInfoBufCPU[2];

    bool m_Active = false;
    uint m_MaxCutSize = 32;
    uint m_ClusterCount = 1;
};
