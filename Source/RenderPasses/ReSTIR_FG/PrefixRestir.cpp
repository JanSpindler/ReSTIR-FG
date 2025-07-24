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

struct PrefixPath
{
    float3 throughput;
    uint4 hitInfo; // PackedHitInfo
    float4 viewDir;
    float rayDist;
    float hitT;
    uint seed;
    uint length;
};

struct PrefixPathReservoir
{
    PrefixPath path;
    float weightSum;
    float confidence;
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
    // Check if all materials are instances of Falcor::StandardMaterial
    for (auto material : pScene->getMaterials())
    {
        const StandardMaterial* standardMaterial = dynamic_cast<StandardMaterial*>(material.get());
        FALCOR_ASSERT(standardMaterial);
    }

    // Store and check if frostbite brdf is used
    m_Scene = pScene;
    m_Defines.add(m_Scene->getSceneDefines());
    FALCOR_ASSERT(m_Defines["DiffuseBrdf"] == "DiffuseBrdfFrostbite");
    m_FrameCount = 0;
}

bool PrefixRestir::RenderUI(Gui::Widgets& widget)
{
    bool changed = false;

    if (auto group = widget.group("Prefix ReSTIR"))
    {
        // Active
        changed |= group.checkbox("Prefix ReSTIR", m_Active);
    }

    return changed;
}

void PrefixRestir::Run(
    RenderContext* pRenderContext,
    const RenderData& renderData,
    ref<Texture> viewDirBuf,
    const bool alphaTest,
    const float2 roughnessCutoff,
    const float diffuseCutoff,
    const bool requireDiffuseMat
)
{
    FALCOR_PROFILE(pRenderContext, "PrefixRestir");

    if (m_FrameCount > 0)
    {
        // Init shader
        if (!m_PrefixResamplingPass)
        {
            Program::Desc desc;
            desc.addShaderModules(m_Scene->getShaderModules());
            desc.addShaderLibrary("RenderPasses/ReSTIR_FG/Shader/PrefixPathResampling.cs.slang").csEntry("main").setShaderModel("6_5");
            desc.addTypeConformances(m_Scene->getTypeConformances());

            DefineList defines = m_Defines;
            defines.add("USE_ALPHA_TEST", alphaTest ? "1" : "0");
            defines.add("TRACE_TRANS_SPEC_ROUGH_CUTOFF_MIN", std::to_string(roughnessCutoff.x));
            defines.add("TRACE_TRANS_SPEC_ROUGH_CUTOFF_MAX", std::to_string(roughnessCutoff.y));
            defines.add("TRACE_TRANS_SPEC_DIFFUSEPART_CUTOFF", std::to_string(diffuseCutoff));
            m_PrefixResamplingPass = ComputePass::create(m_Device, desc, defines, true);
        }
        FALCOR_ASSERT(m_PrefixResamplingPass);

        // Defines
        m_PrefixResamplingPass->addDefine("USE_ALPHA_TEST", alphaTest ? "1" : "0");
        m_PrefixResamplingPass->addDefine("TRACE_TRANS_SPEC_ROUGH_CUTOFF_MIN", std::to_string(roughnessCutoff.x));
        m_PrefixResamplingPass->addDefine("TRACE_TRANS_SPEC_ROUGH_CUTOFF_MAX", std::to_string(roughnessCutoff.y));
        m_PrefixResamplingPass->addDefine("TRACE_TRANS_SPEC_DIFFUSEPART_CUTOFF", std::to_string(diffuseCutoff));

        // Set variables
        auto var = m_PrefixResamplingPass->getRootVar();

        var["CB"]["gEnableTemporalReprojection"] = m_EnableTemporalReprojection;
        var["CB"]["gFrameDim"] = m_ScreenSize;
        var["CB"]["gFrameCount"] = m_FrameCount;
        var["CB"]["gRequDiffParts"] = requireDiffuseMat;

        var["gVBuffer"] = renderData[kInputVBuffer]->asTexture();
        var["gVBufferPrev"] = m_TemporalVBuffer;
        var["gViewDirRayDistDI"] = viewDirBuf;
        var["gMotionVectors"] = renderData[kInputVBuffer]->asTexture();
        var["gCurrentReservoirs"] = m_OutputReservoirs;
        var["gTemporalReservoirs"] = m_TemporalReservoirs;

        // Execute
        m_PrefixResamplingPass->execute(pRenderContext, uint3(m_ScreenSize, 1));
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
