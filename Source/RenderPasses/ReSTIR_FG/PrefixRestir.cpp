#include "PrefixRestir.h"
#include <string>
#include <Utils/Math/Common.h>

// Render pass inputs and outputs.
static const std::string kInputVBuffer = "vbuffer";
static const std::string kInputMotionVectors = "motionVectors";
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

static const std::string kTemporalReusePassFile = "RenderPasses/ReSTIR_FG/Shader/TemporalReuse.cs.slang";
static const std::string kTemporalPathRetraceFile = "RenderPasses/ReSTIR_FG/Shader/TemporalPathRetrace.cs.slang";

PrefixRestir::PrefixRestir(ref<Device> pDevice, DefineList defines) : m_Device(pDevice)
{
    defines.add("SAMPLES_PER_PIXEL", "1");

    {
        Program::Desc desc;
        desc.addShaderLibrary(kTemporalPathRetraceFile).csEntry("main").setShaderModel("6_5");
        m_TemporalPathRetracePass = ComputePass::create(m_Device, desc, defines, false);
    }

    {
        Program::Desc desc;
        desc.addShaderLibrary(kTemporalReusePassFile).csEntry("main").setShaderModel("6_5");
        m_TemporalReusePass = ComputePass::create(m_Device, desc, defines, false);
    }
}

void PrefixRestir::PrepareBuffers(RenderContext* pRenderContext, const uint2 screenSize)
{
    // Compute allocation requirements for paths and output samples.
    // Note that the sample buffers are padded to whole tiles, while the max path count depends on actual frame dimension.
    // If we don't have a fixed sample count, assume the worst case.

    m_Params.frameDim = screenSize;
    if (m_Params.frameDim.x > kMaxFrameDimension || m_Params.frameDim.y > kMaxFrameDimension)
    {
        logError("Frame dimensions up to " + std::to_string(kMaxFrameDimension) + " pixels width/height are supported.");
    }
    assert(isPowerOf2(kScreenTileDim.x) && isPowerOf2(kScreenTileDim.y));
    assert(kScreenTileDim.x == (1 << kScreenTileBits.x) && kScreenTileDim.y == (1 << kScreenTileBits.y));
    m_Params.screenTiles = div_round_up(m_Params.frameDim, kScreenTileDim);

    uint32_t tileCount = m_Params.screenTiles.x * m_Params.screenTiles.y;
    const uint32_t reservoirCount = tileCount * kScreenTileDim.x * kScreenTileDim.y;
    const uint32_t screenPixelCount = m_Params.frameDim.x * m_Params.frameDim.y;
    const uint32_t sampleCount = reservoirCount; // we are effectively only using 1spp for ReSTIR

    const uint32_t baseReservoirSize = 88; // Used for ReSTIR PT
    //const uint32_t pathTreeReservoirSize = 128; // Used for bekeart style path reuse
    const uint reconnectionDataSize = 256; // Use online reconnection data size

    if (!m_ReconnectionDataBuffer)
    {
        m_ReconnectionDataBuffer = Buffer::createStructured(
            m_Device, reconnectionDataSize, reservoirCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
            Buffer::CpuAccess::None, nullptr, false
        );
    }

    if (!m_OutputReservoirs)
    {
        m_OutputReservoirs = Buffer::createStructured(
            m_Device, baseReservoirSize, reservoirCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false
        );
    }

    if (!m_TemporalReservoirs)
    {
        m_TemporalReservoirs = Buffer::createStructured(
            m_Device, baseReservoirSize, reservoirCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false
        );
    }

    if (!m_TemporalVBuffer)
    {
        m_TemporalVBuffer = Texture::create2D(m_Device, m_Params.frameDim.x, m_Params.frameDim.y, m_Scene->getHitInfo().getFormat(), 1, 1);
    }
}

void PrefixRestir::SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    m_Scene = pScene;
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
    PathRetracePass(pRenderContext, renderData);
    PathReusePass(pRenderContext, renderData);
}

void PrefixRestir::SetShaderData(const ShaderVar& var, const RenderData& renderData, bool isPathTracer, bool isPathGenerator) const
{
    // Bind runtime data.
    var["params"].setBlob(m_Params);
    var["vbuffer"] = renderData[kInputVBuffer]->asTexture();
    var["outputColor"] = renderData[kOutputColor]->asTexture();

    // TODO: Do we need this?
    //if (isPathTracer)
    //{
    //    var["isLastRound"] = !mEnableSpatialReuse && !mEnableTemporalReuse;
    //    var["useDirectLighting"] = mUseDirectLighting;
    //    var["kUseEnvLight"] = mpScene->useEnvLight();
    //    var["kUseEmissiveLights"] = mpScene->useEmissiveLights();
    //    var["kUseAnalyticLights"] = mpScene->useAnalyticLights();
    //}
    //else if (isPathGenerator)
    //{
    //    var["kUseEnvBackground"] = mpScene->useEnvBackground();
    //}

    if (auto outputDebug = var.findMember("outputDebug"); outputDebug.isValid())
    {
        outputDebug = renderData[kOutputDebug]->asTexture(); // Can be nullptr
    }
    if (auto outputTime = var.findMember("outputTime"); outputTime.isValid())
    {
        outputTime = renderData[kOutputTime]->asTexture(); // Can be nullptr
    }

    // TODO: Do we need this?
    //if (isPathTracer && mpEmissiveSampler)
    //{
    //    // TODO: Do we have to bind this every frame?
    //    bool success = mpEmissiveSampler->setShaderData(var["emissiveSampler"]);
    //    if (!success)
    //        throw std::exception("Failed to bind emissive light sampler");
    //}
}

void PrefixRestir::PathRetracePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    ref<ComputePass> pass = m_TemporalPathRetracePass;

    // Check shader assumptions.
    // We launch one thread group per screen tile, with threads linearly indexed.
    const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
    assert(kScreenTileDim.x == 16 && kScreenTileDim.y == 16); // TODO: Remove this temporary limitation when Slang bug has been fixed, see comments in shader.
    assert(kScreenTileBits.x <= 4 && kScreenTileBits.y <= 4); // Since we use 8-bit deinterleave.
    assert(pass->getThreadGroupSize().x == 16);
    assert(pass->getThreadGroupSize().y == 16 && pass->getThreadGroupSize().z == 1);

    // Additional specialization. This shouldn't change resource declarations.
    //pass->addDefine("OUTPUT_TIME", mOutputTime ? "1" : "0");
    pass->addDefine("TEMPORAL_REUSE", "1");

    // Bind resources.
    auto var = pass->getRootVar()["CB"]["gPathRetracePass"];

    // TODO: refactor arguments
    SetShaderData(var, renderData, false, false);
    var["outputReservoirs"] = m_OutputReservoirs;
    var["temporalReservoirs"] = m_TemporalReservoirs;
    var["reconnectionDataBuffer"] = m_ReconnectionDataBuffer;

    var["temporalVbuffer"] = m_TemporalVBuffer;
    var["motionVectors"] = renderData[kInputMotionVectors]->asTexture();
    var["gEnableTemporalReprojection"] = m_EnableTemporalReprojection;
    var["gNoResamplingForTemporalReuse"] = m_NoResamplingForTemporalReuse;

    // TODO: Do we need this?
    //if (!mUseMaxHistory)
    //{
    var["gTemporalHistoryLength"] = 1e30f;
    //}
    //else
    //{
    //    var["gTemporalHistoryLength"] = (float)mTemporalHistoryLength;
    //}

    // TODO: Fix (older Falcor version)
    //pass["gScene"] = m_Scene->getParameterBlock();
    //pass["gPathTracer"] = mpPathTracerBlock;

    // TODO: Do we need this?
    //mpPixelStats->prepareProgram(pass->getProgram(), pass->getRootVar());
    //mpPixelDebug->prepareProgram(pass->getProgram(), pass->getRootVar());

    {
        // Launch one thread per pixel.
        // The dimensions are padded to whole tiles to allow re-indexing the threads in the shader.
        pass->execute(pRenderContext, {m_Params.screenTiles.x * kScreenTileDim.x, m_Params.screenTiles.y * kScreenTileDim.y, 1u});
    }
}

void PrefixRestir::PathReusePass(RenderContext* pRenderContext, const RenderData& renderData)
{
    ref<ComputePass> pass = m_TemporalReusePass;

    // Check shader assumptions.
    // We launch one thread group per screen tile, with threads linearly indexed.
    const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
    assert(kScreenTileDim.x == 16 && kScreenTileDim.y == 16); // TODO: Remove this temporary limitation when Slang bug has been fixed, see
                                                              // comments in shader.
    assert(kScreenTileBits.x <= 4 && kScreenTileBits.y <= 4); // Since we use 8-bit deinterleave.
    assert(pass->getThreadGroupSize().x == 16);
    assert(pass->getThreadGroupSize().y == 16 && pass->getThreadGroupSize().z == 1);

    // Additional specialization. This shouldn't change resource declarations.
    // TODO: Do we need this?
    //pass->addDefine("OUTPUT_TIME", mOutputTime ? "1" : "0");
    pass->addDefine("TEMPORAL_REUSE", "1");
    //pass->addDefine("OUTPUT_NRD_DATA", mOutputNRDData ? "1" : "0");
    pass->addDefine("SAMPLES_PER_PIXEL", "1");

    // Bind resources.
    auto var = pass->getRootVar()["CB"]["gPathReusePass"];

    // TODO: refactor arguments
    SetShaderData(var, renderData, false, false);

    var["outputReservoirs"] = m_OutputReservoirs;
    var["temporalReservoirs"] = m_TemporalReservoirs;
    var["reconnectionDataBuffer"] = m_ReconnectionDataBuffer;

    var["temporalVbuffer"] = m_TemporalVBuffer;
    var["motionVectors"] = renderData[kInputMotionVectors]->asTexture();
    var["gEnableTemporalReprojection"] = m_EnableTemporalReprojection;
    var["gNoResamplingForTemporalReuse"] = m_NoResamplingForTemporalReuse;
    // TODO: Do we need this?
    //if (!mUseMaxHistory)
    var["gTemporalHistoryLength"] = 1e30f;
    //else
    //    var["gTemporalHistoryLength"] = (float)mTemporalHistoryLength;

    var["directLighting"] = renderData[kInputDirectLighting]->asTexture();
    var["useDirectLighting"] = m_UseDirectLighting;
    var["gIsLastRound"] = true;

    // TODO: Fix (older Falcor version)
    //pass["gScene"] = mpScene->getParameterBlock();
    //pass["gPathTracer"] = mpPathTracerBlock;

    //mpPixelStats->prepareProgram(pass->getProgram(), pass->getRootVar());
    //mpPixelDebug->prepareProgram(pass->getProgram(), pass->getRootVar());

    {
        // Launch one thread per pixel.
        // The dimensions are padded to whole tiles to allow re-indexing the threads in the shader.
        pass->execute(pRenderContext, {m_Params.screenTiles.x * kScreenTileDim.x, m_Params.screenTiles.y * kScreenTileDim.y, 1u});
    }
}
