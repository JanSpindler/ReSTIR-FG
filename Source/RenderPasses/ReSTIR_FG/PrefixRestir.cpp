#include "PrefixRestir.h"
#include <string>

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

void PrefixRestir::SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    m_Scene = pScene;
}

void PrefixRestir::SetShaderData(const ShaderVar& var, const RenderData& renderData, bool isPathTracer, bool isPathGenerator) const
{
    // Bind runtime data.
    var["params"].setBlob(mParams);
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

    pass["gScene"] = m_Scene->getParameterBlock();
    pass["gPathTracer"] = mpPathTracerBlock;

    // TODO: Do we need this?
    //mpPixelStats->prepareProgram(pass->getProgram(), pass->getRootVar());
    //mpPixelDebug->prepareProgram(pass->getProgram(), pass->getRootVar());

    {
        // Launch one thread per pixel.
        // The dimensions are padded to whole tiles to allow re-indexing the threads in the shader.
        pass->execute(pRenderContext, {mParams.screenTiles.x * kScreenTileDim.x, mParams.screenTiles.y * kScreenTileDim.y, 1u});
    }
}

/*void PrefixRestir::PathReusePass(
    RenderContext* pRenderContext,
    uint32_t restir_i,
    const RenderData& renderData,
    bool isTemporalReuse,
    int spatialRoundId,
    bool isLastRound
)
{
    bool isPathReuseMISWeightComputation = spatialRoundId == -1;

    PROFILE(isTemporalReuse ? "temporalReuse" : (isPathReuseMISWeightComputation ? "MISWeightComputation" : "spatialReuse"));

    ComputePass::SharedPtr pass =
        isPathReuseMISWeightComputation ? mpComputePathReuseMISWeightsPass : (isTemporalReuse ? mpTemporalReusePass : mpSpatialReusePass);

    if (isPathReuseMISWeightComputation)
    {
        spatialRoundId = 0;
        restir_i = 0;
    }

    // Check shader assumptions.
    // We launch one thread group per screen tile, with threads linearly indexed.
    const uint32_t tileSize = kScreenTileDim.x * kScreenTileDim.y;
    assert(kScreenTileDim.x == 16 && kScreenTileDim.y == 16); // TODO: Remove this temporary limitation when Slang bug has been fixed, see
                                                              // comments in shader.
    assert(kScreenTileBits.x <= 4 && kScreenTileBits.y <= 4); // Since we use 8-bit deinterleave.
    assert(pass->getThreadGroupSize().x == 16);
    assert(pass->getThreadGroupSize().y == 16 && pass->getThreadGroupSize().z == 1);

    // Additional specialization. This shouldn't change resource declarations.
    pass->addDefine("OUTPUT_TIME", mOutputTime ? "1" : "0");
    pass->addDefine("TEMPORAL_REUSE", isTemporalReuse ? "1" : "0");
    pass->addDefine("OUTPUT_NRD_DATA", mOutputNRDData ? "1" : "0");

    // Bind resources.
    auto var = pass->getRootVar()["CB"]["gPathReusePass"];

    // TODO: refactor arguments
    setShaderData(var, renderData, false, false);

    var["outputReservoirs"] = spatialRoundId % 2 == 1 ? mpTemporalReservoirs[restir_i] : mpOutputReservoirs;

    if (mStaticParams.pathSamplingMode == PathSamplingMode::PathReuse)
    {
        var["nRooksPattern"] = mNRooksPatternBuffer;
    }

    if (mStaticParams.pathSamplingMode == PathSamplingMode::PathReuse)
        var["misWeightBuffer"] = mPathReuseMISWeightBuffer;
    else if (!isPathReuseMISWeightComputation)
        var["temporalReservoirs"] = spatialRoundId % 2 == 0 ? mpTemporalReservoirs[restir_i] : mpOutputReservoirs;
    var["reconnectionDataBuffer"] = mReconnectionDataBuffer;

    var["gNumSpatialRounds"] = mNumSpatialRounds;

    if (isTemporalReuse)
    {
        var["temporalVbuffer"] = mpTemporalVBuffer;
        var["motionVectors"] = renderData[kInputMotionVectors]->asTexture();
        var["gEnableTemporalReprojection"] = mEnableTemporalReprojection;
        var["gNoResamplingForTemporalReuse"] = mNoResamplingForTemporalReuse;
        if (!mUseMaxHistory)
            var["gTemporalHistoryLength"] = 1e30f;
        else
            var["gTemporalHistoryLength"] = (float)mTemporalHistoryLength;
    }
    else
    {
        var["gSpatialReusePattern"] =
            mStaticParams.pathSamplingMode == PathSamplingMode::PathReuse ? (uint32_t)mPathReusePattern : (uint32_t)mSpatialReusePattern;

        if (!isPathReuseMISWeightComputation)
        {
            var["gNeighborCount"] = mSpatialNeighborCount;
            var["gGatherRadius"] = mSpatialReuseRadius;
            var["gSpatialRoundId"] = spatialRoundId;
            var["gSmallWindowRadius"] = mSmallWindowRestirWindowRadius;
            var["gFeatureBasedRejection"] = mFeatureBasedRejection;
            var["neighborOffsets"] = mpNeighborOffsets;
        }

        if (mOutputNRDData && !isPathReuseMISWeightComputation)
        {
            var["outputNRDDiffuseRadianceHitDist"] = renderData[kOutputNRDDiffuseRadianceHitDist]->asTexture();
            var["outputNRDSpecularRadianceHitDist"] = renderData[kOutputNRDSpecularRadianceHitDist]->asTexture();
            var["outputNRDResidualRadianceHitDist"] = renderData[kOutputNRDResidualRadianceHitDist]->asTexture();
            var["primaryHitEmission"] = renderData[kOutputNRDEmission]->asTexture();
            var["gSppId"] = restir_i;
        }
    }

    if (!isPathReuseMISWeightComputation)
    {
        var["directLighting"] = renderData[kInputDirectLighting]->asTexture();
        var["useDirectLighting"] = mUseDirectLighting;
    }
    var["gIsLastRound"] = mStaticParams.pathSamplingMode == PathSamplingMode::PathReuse || isLastRound;

    pass["gScene"] = mpScene->getParameterBlock();
    pass["gPathTracer"] = mpPathTracerBlock;

    mpPixelStats->prepareProgram(pass->getProgram(), pass->getRootVar());
    mpPixelDebug->prepareProgram(pass->getProgram(), pass->getRootVar());

    {
        // Launch one thread per pixel.
        // The dimensions are padded to whole tiles to allow re-indexing the threads in the shader.
        pass->execute(pRenderContext, {mParams.screenTiles.x * kScreenTileDim.x, mParams.screenTiles.y * kScreenTileDim.y, 1u});
    }
}*/
