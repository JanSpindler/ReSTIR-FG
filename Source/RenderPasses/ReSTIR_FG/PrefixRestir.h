#pragma once

#include <Falcor.h>
#include <RenderGraph/RenderPass.h>
#include "Shader/Params.slang"

using namespace Falcor;

class PrefixRestir
{
public:
    void SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene);
    void Run(RenderContext* pRenderContext);

private:
    ref<Scene> m_Scene;

    RestirPathTracerParams mParams; // Runtime path tracer parameters.

    ref<Buffer> m_OutputReservoirs; // Output paths from the path sampling stage
    ref<Buffer> m_TemporalReservoirs;
    ref<Buffer> m_ReconnectionDataBuffer;

    ref<Texture> m_TemporalVBuffer;

    bool m_EnableTemporalReprojection = true;
    bool m_NoResamplingForTemporalReuse = false;

    ref<ComputePass> m_TemporalPathRetracePass;
    ref<ComputePass> m_TemporalReusePass; // Merges reservoirs

    void SetShaderData(const ShaderVar& var, const RenderData& renderData, bool isPathTracer, bool isPathGenerator) const;
    void PathRetracePass(RenderContext* pRenderContext, const RenderData& renderData);
    //void PathReusePass(
    //    RenderContext* pRenderContext,
    //    uint32_t restir_i,
    //    const RenderData& renderData,
    //    bool temporalReuse = false,
    //    int spatialRoundId = 0,
    //    bool isLastRound = false
    //);
};
