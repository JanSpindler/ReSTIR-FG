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

static const std::string kTracePassFilename = "RenderPasses/ReSTIR_FG/Shader/TracePass.cs.slang";
static const std::string kTemporalReusePassFile = "RenderPasses/ReSTIR_FG/Shader/TemporalReuse.cs.slang";
static const std::string kTemporalPathRetraceFile = "RenderPasses/ReSTIR_FG/Shader/TemporalPathRetrace.cs.slang";

static const uint32_t kNeighborOffsetCount = 8192;

PrefixRestir::PrefixRestir(ref<Device> pDevice, DefineList defines) : m_Device(pDevice), m_Defines(defines)
{
    m_Defines.add(m_StaticParams.GetDefines());
    m_Defines.add("GBUFFER_ADJUST_SHADING_NORMALS", m_GBufferAdjustShadingNormals ? "1" : "0");
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

    static constexpr uint32_t baseReservoirSize = 88; // Used for ReSTIR PT
    //const uint32_t pathTreeReservoirSize = 128; // Used for bekeart style path reuse
    static constexpr uint reconnectionDataSize = 256; // Use online reconnection data size

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

void PrefixRestir::PreparePathTracer(const RenderData& renderData)
{
    if (!m_TracePass)
    {
        Program::Desc desc;
        desc.addShaderModules(m_Scene->getShaderModules());
        desc.addTypeConformances(m_Scene->getTypeConformances());
        desc.addShaderLibrary(kTracePassFilename).csEntry("main").setShaderModel("6_5");
        m_TracePass = ComputePass::create(m_Device, desc, m_Defines, false);
    }

    if (!m_PathTracerBlock)// || mVarsChanged)
    {
        auto reflector = m_TracePass->getProgram()->getReflector()->getParameterBlock("gPathTracer");
        m_PathTracerBlock = ParameterBlock::create(m_Device, reflector);
        assert(m_PathTracerBlock);
        //mVarsChanged = true;
    }

    // Bind resources.
    auto var = m_PathTracerBlock->getRootVar();
    SetShaderData(var, renderData, true, false);
    var["outputReservoirs"] = m_OutputReservoirs;
    //var["directLighting"] = renderData[kInputDirectLighting]->asTexture();
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
        PathRetracePass(pRenderContext, renderData);
        //PathReusePass(pRenderContext, renderData);
    }

    pRenderContext->copyResource(m_TemporalReservoirs.get(), m_OutputReservoirs.get());
    pRenderContext->copyResource(m_TemporalVBuffer.get(), renderData[kInputVBuffer].get());

    ++m_FrameCount;
}

DefineList PrefixRestir::GetDefines() const
{
    DefineList defines;
    defines.add("PREFIX_RESTIR", m_Active ? "1" : "0");
    defines.add(m_Defines);
    return defines;
}

void PrefixRestir::SetTraceTransmissionDeltaVars(const ShaderVar& var)
{
    var["gPathTracer"] = m_PathTracerBlock;
}

void PrefixRestir::SetShaderData(const ShaderVar& var, const RenderData& renderData, bool isPathTracer, bool isPathGenerator) const
{
    // Bind runtime data.
    var["params"].setBlob(m_Params);
    var["vbuffer"] = renderData[kInputVBuffer]->asTexture();
    //var["outputColor"] = renderData[kOutputColor]->asTexture();

    // TODO: Do we need this?
    if (isPathTracer)
    {
    //    var["isLastRound"] = !mEnableSpatialReuse && !mEnableTemporalReuse;
    var["useDirectLighting"] = false; // mUseDirectLighting;
    var["kUseEnvLight"] = false;//mpScene->useEnvLight();
    var["kUseEmissiveLights"] = m_Scene->useEmissiveLights();
    var["kUseAnalyticLights"] = m_Scene->useAnalyticLights();
    }
    else if (isPathGenerator)
    {
        var["kUseEnvBackground"] = false;//mpScene->useEnvBackground();
    }

    //if (auto outputDebug = var.findMember("outputDebug"); outputDebug.isValid())
    //{
    //    outputDebug = renderData[kOutputDebug]->asTexture(); // Can be nullptr
    //}
    //if (auto outputTime = var.findMember("outputTime"); outputTime.isValid())
    //{
    //    outputTime = renderData[kOutputTime]->asTexture(); // Can be nullptr
    //}

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
    if (!m_TemporalPathRetracePass)
    {
        Program::Desc desc;
        desc.addShaderModules(m_Scene->getShaderModules());
        desc.addTypeConformances(m_Scene->getTypeConformances());
        desc.addShaderLibrary(kTemporalPathRetraceFile).csEntry("main").setShaderModel("6_5");

        m_TemporalPathRetracePass = ComputePass::create(m_Device, desc, m_Defines, true);
    }
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

    if (!m_UseMaxHistory)
    {
        var["gTemporalHistoryLength"] = 1e30f;
    }
    else
    {
        var["gTemporalHistoryLength"] = static_cast<float>(m_TemporalHistoryLength);
    }

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
    if (!m_TemporalReusePass)
    {
        Program::Desc desc;
        desc.addShaderModules(m_Scene->getShaderModules());
        desc.addTypeConformances(m_Scene->getTypeConformances());
        desc.addShaderLibrary(kTemporalReusePassFile).csEntry("main").setShaderModel("6_5");
        m_TemporalReusePass = ComputePass::create(m_Device, desc, m_Defines, true);
    }
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

    if (!m_UseMaxHistory)
    {
        var["gTemporalHistoryLength"] = 1e30f;
    }
    else
    {
        var["gTemporalHistoryLength"] = static_cast<float>(m_TemporalHistoryLength);
    }

    //var["directLighting"] = renderData[kInputDirectLighting]->asTexture();
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

DefineList PrefixRestir::StaticParams::GetDefines() const
{
    DefineList defines;

    // Path tracer configuration.
    defines.add("SAMPLES_PER_PIXEL", std::to_string(samplesPerPixel));  // 0 indicates a variable sample count
    defines.add("CANDIDATE_SAMPLES", std::to_string(candidateSamples)); // 0 indicates a variable sample count
    defines.add("MAX_SURFACE_BOUNCES", std::to_string(maxSurfaceBounces));
    defines.add("MAX_DIFFUSE_BOUNCES", std::to_string(maxDiffuseBounces));
    defines.add("MAX_SPECULAR_BOUNCES", std::to_string(maxSpecularBounces));
    defines.add("MAX_TRANSMISSON_BOUNCES", std::to_string(maxTransmissionBounces));
    defines.add("ADJUST_SHADING_NORMALS", adjustShadingNormals ? "1" : "0");
    defines.add("USE_BSDF_SAMPLING", useBSDFSampling ? "1" : "0");
    defines.add("USE_NEE", useNEE ? "1" : "0");
    defines.add("USE_MIS", useMIS ? "1" : "0");
    defines.add("USE_RUSSIAN_ROULETTE", useRussianRoulette ? "1" : "0");
    defines.add("USE_ALPHA_TEST", useAlphaTest ? "1" : "0");
    defines.add("USE_LIGHTS_IN_DIELECTRIC_VOLUMES", useLightsInDielectricVolumes ? "1" : "0");
    defines.add("LIMIT_TRANSMISSION", limitTransmission ? "1" : "0");
    defines.add("MAX_TRANSMISSION_REFLECTION_DEPTH", std::to_string(maxTransmissionReflectionDepth));
    defines.add("MAX_TRANSMISSION_REFRACTION_DEPTH", std::to_string(maxTransmissionRefractionDepth));
    defines.add("DISABLE_CAUSTICS", disableCaustics ? "1" : "0");
    defines.add("DISABLE_DIRECT_ILLUMINATION", disableDirectIllumination ? "1" : "0");
    defines.add("PRIMARY_LOD_MODE", std::to_string((uint32_t)primaryLodMode));
    defines.add("USE_NRD_DEMODULATION", useNRDDemodulation ? "1" : "0");
    defines.add("COLOR_FORMAT", std::to_string((uint32_t)colorFormat));
    defines.add("MIS_HEURISTIC", std::to_string((uint32_t)misHeuristic));
    defines.add("MIS_POWER_EXPONENT", std::to_string(misPowerExponent));
    defines.add("_USE_DETERMINISTIC_BSDF", useDeterministicBSDF ? "1" : "0");
    defines.add("NEIGHBOR_OFFSET_COUNT", std::to_string(kNeighborOffsetCount));
    defines.add("SHIFT_STRATEGY", std::to_string((uint32_t)shiftStrategy));
    defines.add("PATH_SAMPLING_MODE", std::to_string((uint32_t)pathSamplingMode));

    // Sampling utilities configuration.
    //assert(owner.mpSampleGenerator);
    //defines.add(owner.mpSampleGenerator->getDefines());

    // We don't use the legacy shading code anymore (MaterialShading.slang).
    defines.add("_USE_LEGACY_SHADING_CODE", "0");

    defines.add("INTERIOR_LIST_SLOT_COUNT", std::to_string(maxNestedMaterials));

    //defines.add("GBUFFER_ADJUST_SHADING_NORMALS", owner.mGBufferAdjustShadingNormals ? "1" : "0");

    // Scene-specific configuration.
    //const auto& scene = owner.mpScene;

    // Set default (off) values for additional features.
    defines.add("OUTPUT_GUIDE_DATA", "0");
    defines.add("OUTPUT_TIME", "0");
    defines.add("OUTPUT_NRD_DATA", "0");
    defines.add("OUTPUT_NRD_ADDITIONAL_DATA", "0");

    defines.add("SPATIAL_RESTIR_MIS_KIND", std::to_string((uint32_t)spatialMisKind));
    defines.add("TEMPORAL_RESTIR_MIS_KIND", std::to_string((uint32_t)temporalMisKind));

    defines.add("TEMPORAL_UPDATE_FOR_DYNAMIC_SCENE", temporalUpdateForDynamicScene ? "1" : "0");

    defines.add("BPR", pathSamplingMode == PathSamplingMode::PathReuse ? "1" : "0");

    defines.add("SEPARATE_PATH_BSDF", separatePathBSDF ? "1" : "0");

    defines.add("RCDATA_PATH_NUM", "6");
    defines.add("RCDATA_PAD_SIZE", "1");

    return defines;
}
