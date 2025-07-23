#pragma once

#include <Falcor.h>
#include <RenderGraph/RenderPass.h>

using namespace Falcor;

class PrefixRestir
{
public:
    PrefixRestir() = default;
    PrefixRestir(ref<Device> pDevice, DefineList defines);

    void PrepareBuffers(RenderContext* pRenderContext, const uint2 screenSize);
    void SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene);
    bool RenderUI(Gui::Widgets& widget);
    void Run(RenderContext* pRenderContext, const RenderData& renderData, ref<Texture> viewDirBuf);

    DefineList GetDefines() const;

    constexpr bool IsActive() const { return m_Active; }

    void SetTraceTransmissionDeltaVars(const ShaderVar& var) const;

private:
    ref<Device> m_Device;
    ref<Scene> m_Scene;
    DefineList m_Defines;

    uint2 m_ScreenSize = uint2(0, 0);

    ref<Buffer> m_OutputReservoirs;
    ref<Buffer> m_TemporalReservoirs;
    ref<Texture> m_TemporalVBuffer;

    bool m_Active = false;
    uint m_FrameCount = 0;
    bool m_EnableTemporalReprojection = true;
    bool m_UseMaxHistory = true;
    uint m_TemporalHistoryLength = 20;

    ref<ComputePass> m_PrefixResamplingPass;

    void PrefixResampling(RenderContext* pRenderContext, const RenderData& renderData);
};
