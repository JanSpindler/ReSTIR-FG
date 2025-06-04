#pragma once

#include <Falcor.h>
#include <RenderGraph/RenderPass.h>
#include "Shader/Params.slang"

using namespace Falcor;

class PrefixRestir
{
public:
    PrefixRestir(ref<Device> pDevice) : m_Device(pDevice) {}

    void PrepareBuffers(RenderContext* pRenderContext, const uint2 screenSize);
    void SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene);
    bool RenderUI(Gui::Widgets& widget);
    void Run(RenderContext* pRenderContext, const RenderData& renderData);

    constexpr bool IsActive() const { return m_Active; }

private:
    ref<Device> m_Device;
    ref<Scene> m_Scene;

    RestirPathTracerParams mParams; // Runtime path tracer parameters.

    ref<Buffer> m_OutputReservoirs; // Output paths from the path sampling stage
    ref<Buffer> m_TemporalReservoirs;
    ref<Buffer> m_ReconnectionDataBuffer;

    ref<Texture> m_TemporalVBuffer;

    bool m_Active = false;
    bool m_EnableTemporalReprojection = true;
    bool m_NoResamplingForTemporalReuse = false;
    bool m_UseDirectLighting = false;

    ref<ComputePass> m_TemporalPathRetracePass;
    ref<ComputePass> m_TemporalReusePass; // Merges reservoirs

    void SetShaderData(const ShaderVar& var, const RenderData& renderData, bool isPathTracer, bool isPathGenerator) const;
    void PathRetracePass(RenderContext* pRenderContext, const RenderData& renderData);
    void PathReusePass(RenderContext* pRenderContext, const RenderData& renderData);
};
