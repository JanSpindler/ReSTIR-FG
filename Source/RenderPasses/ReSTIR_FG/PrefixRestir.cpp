#include "PrefixRestir.h"
#include <string>
#include <Utils/Math/Common.h>
#include <RenderGraph/RenderPassHelpers.h>

// Render pass inputs and outputs.
static const std::string kInputVBuffer = "vbuffer";
static const std::string kInputMotionVectors = "mvec"; //"motionVectors";
static const std::string kInputDirectLighting = "directLighting";

static const std::string kOutputColor = "color";
static const std::string kOutputAlbedo = "albedo";
static const std::string kOutputSpecularAlbedo = "specularAlbedo";
static const std::string kOutputIndirectAlbedo = "indirectAlbedo";
static const std::string kOutputNormal = "normal";
static const std::string kOutputReflectionPosW = "reflectionPosW";
static const std::string kOutputRayCount = "rayCount";
static const std::string kOutputPathLength = "pathLength";
static const std::string kOutputDebug = "debug";
static const std::string kOutputTime = "time";

static const std::string kPrefixPathResamplingPassFile = "RenderPasses/ReSTIR_FG/Shader/PrefixPathResampling.rt.slang";

static const uint32_t kNeighborOffsetCount = 8192;

struct PrefixPathReservoir
{
    // Jacobian is always 1 because we only perform random replay

    // Path data
    float3 F;
    uint length; // Number of vertices inlcuding primary hit.
    uint sgSeed; // State of sample generator right before generating this path

    // Resampling data
    float M;
    float weight;
};

PrefixRestir::PrefixRestir(ref<Device> pDevice, DefineList defines) : m_Device(pDevice), m_Defines(defines)
{
}

void PrefixRestir::PrepareBuffers(RenderContext* pRenderContext, const uint2 screenSize)
{
    // Compute allocation requirements for paths and output samples.
    // Note that the sample buffers are padded to whole tiles, while the max path count depends on actual frame dimension.
    // If we don't have a fixed sample count, assume the worst case.

    if (screenSize.x != m_ScreenSize.x or screenSize.y != m_ScreenSize.y)
    {
        m_FrameCount = 0;
        m_OutputReservoirs.reset();
        m_TemporalReservoirs.reset();
        m_TemporalVBuffer.reset();
    }
    m_ScreenSize = screenSize;

    const uint pixelCount = screenSize.x * screenSize.y;

    if (!m_OutputReservoirs)
    {
        m_OutputReservoirs = Buffer::createStructured(
            m_Device, sizeof(PrefixPathReservoir), pixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false
        );
    }

    if (!m_TemporalReservoirs)
    {
        m_TemporalReservoirs = Buffer::createStructured(
            m_Device, sizeof(PrefixPathReservoir), pixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false
        );
    }

    if (!m_TemporalVBuffer)
    {
        m_TemporalVBuffer = Texture::create2D(m_Device, m_ScreenSize.x, m_ScreenSize.y, m_Scene->getHitInfo().getFormat(), 1, 1);
    }
}

void PrefixRestir::SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    m_Scene = pScene;
    m_Defines.add(m_Scene->getSceneDefines());
    m_FrameCount = 0;
}

bool PrefixRestir::RenderUI(Gui::Widgets& widget)
{
    bool changed = false;

    if (auto group = widget.group("Prefix ReSTIR"))
    {
        // Active
        changed |= group.checkbox("Adaptive Light Sampler", m_Active);
    }

    return changed;
}

void PrefixRestir::Run(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (m_FrameCount > 0)
    {
    }

    pRenderContext->copyResource(m_TemporalReservoirs.get(), m_OutputReservoirs.get());
    pRenderContext->copyResource(m_TemporalVBuffer.get(), renderData[kInputVBuffer].get());

    ++m_FrameCount;
}

DefineList PrefixRestir::GetDefines() const
{
    return m_Defines;
}

void PrefixRestir::SetTraceTransmissionDeltaVars(const ShaderVar& var) const
{
    var["gOutputReservoirs"] = m_OutputReservoirs;
}

void PrefixRestir::PrefixResampling(RenderContext* pRenderContext, const RenderData& renderData)
{

}
