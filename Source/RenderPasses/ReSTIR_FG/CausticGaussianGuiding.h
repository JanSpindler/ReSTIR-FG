#pragma once

#include <Falcor.h>
#include <RenderGraph/RenderPass.h>

using namespace Falcor;

class CausticGaussianGuiding
{
public:
    CausticGaussianGuiding(ref<Device> device) : m_Device(device) {}

    void PrepareBuffers(RenderContext* pRenderContext, const uint causticBufSize);
    bool RenderUI(Gui::Widgets& widget);
    void Run(RenderContext* pRenderContext);

    constexpr bool IsActive() const { return m_Active; }

    void ClearCausticSources(RenderContext* pRenderContext) const;
    void ClearHashGridCounter(RenderContext* pRenderContext) const;
    void SetGeneratePhotonsVars(const ShaderVar& vars);
    void SetCollectPhotonsVars(const ShaderVar& vars);
    void SetFinalShadingVars(const ShaderVar& vars);

private:
    ref<Device> m_Device;

    bool m_Active = false;
    uint m_HashGridSize = 1e5;
    float m_HashScalingFactor = 10.0f;
    float m_GaussSamplingProb = 0.5f;
    float m_GaussSigma = 1.0f / 3.0f;

    uint m_CurrentBufIdx = 0;

    ref<Buffer> m_ReservoirHashGrid[2];
    ref<Buffer> m_HashGridCounter;
    ref<Buffer> m_CausticSourceMap;
};
