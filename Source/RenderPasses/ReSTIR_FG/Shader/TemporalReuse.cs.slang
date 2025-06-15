/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
import Params;
import Shift;
import Scene.HitInfo;
import PathReservoir;
import PrefixPathTracer;
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.Ray;
import LoadShadingData;
//import Rendering.Materials.MaterialShading;

/** TODO.
*/
struct PathReusePass
{
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    // Resources
    RestirPathTracerParams params;                        ///< Runtime parameters.

    Texture2D<PackedHitInfo> vbuffer;                     ///< Fullscreen V-buffer for the primary hits.
    Texture2D<PackedHitInfo> temporalVbuffer;
    Texture2D<float2> motionVectors;

    RWTexture2D<float4> outputColor;                      ///< Output resolved color.
    RWStructuredBuffer<PathReservoir> outputReservoirs;    // reservoir for next pass
    RWStructuredBuffer<PathReservoir> temporalReservoirs;  // reservoir from previous frame
    StructuredBuffer<PixelReconnectionData> reconnectionDataBuffer;

    Texture2D<float4> directLighting;                  
    bool useDirectLighting;

    int  gNumSpatialRounds;

    bool gIsLastRound;
    bool gEnableTemporalReprojection;

    float gTemporalHistoryLength;
    bool gNoResamplingForTemporalReuse;

    bool isValidPackedHitInfo(PackedHitInfo packed)
    {
        return packed.x != 0;
    }

    bool isValidScreenRegion(int2 pixel) { return all(pixel >= 0 && pixel < params.frameDim); }

    ShadingData getPixelShadingData(int2 pixel, out PackedHitInfo PrimaryHitPacked)
    {
        ShadingData sd = {};
        Ray ray = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        PrimaryHitPacked = vbuffer[pixel];
        if (isValidPackedHitInfo(PrimaryHitPacked))
        {
            HitInfo PrimaryHit = HitInfo(PrimaryHitPacked); //PrimaryHit.unpack(PrimaryHitPacked);
            sd = loadShadingData(PrimaryHit, -ray.dir, true);
        }
        return sd;
    }

    ShadingData getPixelTemporalShadingData(int2 pixel, out PackedHitInfo PrimaryHitPacked)
    {
        ShadingData sd = {};
        Ray ray = gScene.camera.computeRayPinhole /*PrevFrame */(pixel, params.frameDim); // TODO: Support prev frame camera.
        PrimaryHitPacked = temporalVbuffer[pixel];
        if (isValidPackedHitInfo(PrimaryHitPacked))
        {
            HitInfo PrimaryHit = HitInfo(PrimaryHitPacked); //PrimaryHit.unpack(PrimaryHitPacked);
            sd = loadShadingData(PrimaryHit, -ray.dir, true);
        }
        return sd;
    }

    /*
    * Build a reservoir of paths whose suffix can be merged to a pixel's prefix.
    * This method is optimized for small windows such that we can run the entire
    * algorithm in one pass.
    */
    void ReSTIR(const uint2 pixel)
    {
        // Get sample generator.
        var sg = TinyUniformSampleGenerator(pixel, (kCandidateSamples + 1 + gNumSpatialRounds) * params.seed + kCandidateSamples);

        // Get central pixel reservoir
        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir dstReservoir = outputReservoirs[centralOffset];
        PathReservoir centralReservoir = dstReservoir;
        const float currentM = dstReservoir.M;
        const float3 initialColor = dstReservoir.F * dstReservoir.weight; // Not used?

        // Prepare reservoir for MIS
        // Use talbot mis for temporal anyways
        dstReservoir.init();

        // Get central pixel shading data
        PackedHitInfo centralPrimaryHitPacked;
        ShadingData centralPrimarySd = getPixelShadingData(pixel, centralPrimaryHitPacked);
        if (!isValidPackedHitInfo(centralPrimaryHitPacked))
        {
            return;
        }

        //
        bool chooseCurrent = true;
        static const bool doTemporalUpdateForDynamicScene = kTemporalUpdateForDynamicScene;

        // Color averaging test

        // Get prev pixel
        const float2 motionVector = motionVectors[pixel];
        int2 prevPixel = pixel;
        bool foundTemporalSurface = true;
        if (gEnableTemporalReprojection)
        {
            prevPixel = pixel + motionVector * params.frameDim + (sampleNext2D(sg) * 1.f - 0.f);
        }
        if (!isValidScreenRegion(prevPixel) || !foundTemporalSurface)
        {
            return;
        }

        // Get prev pixel reservoir
        PackedHitInfo temporalPrimaryHitPacked;
        ShadingData temporalPrimarySd = getPixelTemporalShadingData(prevPixel, temporalPrimaryHitPacked);
        if (!isValidPackedHitInfo(temporalPrimaryHitPacked))
        {
            return;
        }
        PathReservoir temporalReservoir = temporalReservoirs[params.getReservoirOffset(prevPixel)];
        temporalReservoir.M = min(gTemporalHistoryLength * currentM, temporalReservoir.M);

        // Talbot MIS used for temporal
        static const int curSampleId = -1;
        static const int prevSampleId = 0;

        float dstJacobian;
        float3 color = 0.f;
        ReconnectionData dummyRcData;
        dummyRcData.Init();

        return;
        for (int i = curSampleId; i <= prevSampleId; i++)
        {
            PathReservoir tempDstReservoir = dstReservoir;
            bool possibleToBeSelected = false;
            float p_sum = 0;
            float p_self = 0;

            if (i == curSampleId)
            {
                tempDstReservoir = outputReservoirs[centralOffset];
                dstJacobian = 1.f;
                possibleToBeSelected = tempDstReservoir.weight > 0;
            }
            else
            {
                ReconnectionData rcData;
                if (ShiftMapping(kShiftStrategy) == ShiftMapping::Hybrid && temporalReservoir.pathFlags.rcVertexLength() > 1)
                {
                    rcData = reconnectionDataBuffer[centralOffset].data[1];
                }
                else
                {
                    rcData = dummyRcData;
                }

                possibleToBeSelected = shiftAndMergeReservoir(
                    params, doTemporalUpdateForDynamicScene, dstJacobian, centralPrimaryHitPacked, centralPrimarySd, tempDstReservoir,
                    temporalPrimarySd, temporalReservoir, rcData, true, sg, false, 1.f, true); // "true" means hypothetically selected as the sample
            }

            if (possibleToBeSelected)
            {
                for (int j = curSampleId; j <= prevSampleId; ++j)
                {
                    if (j == curSampleId)
                    {
                        const float cur_p = PathReservoir::toScalar(tempDstReservoir.F) * currentM;
                        p_sum += cur_p;
                        if (i == curSampleId)
                        {
                            p_self = cur_p;
                        }
                    }
                    else
                    {
                        if (i == j)
                        {
                            p_self = PathReservoir::toScalar(temporalReservoir.F) / dstJacobian * temporalReservoir.M;
                            p_sum += p_self;
                            continue;
                        }

                        ReconnectionData rcData;
                        if (ShiftMapping(kShiftStrategy) == ShiftMapping::Hybrid && tempDstReservoir.pathFlags.rcVertexLength() > 1)
                        {
                            rcData = reconnectionDataBuffer[centralOffset].data[0];
                        }
                        else
                        {
                            rcData = dummyRcData;
                        }

                        float tneighborJacobian;
                        const float3 tneighborIntegrand = computeShiftedIntegrand(
                            params, tneighborJacobian, temporalPrimaryHitPacked, temporalPrimarySd,
                            centralPrimarySd, tempDstReservoir, rcData, true, true); //usePrev
                        const float p_ = PathReservoir::toScalar(tneighborIntegrand) * tneighborJacobian;
                        p_sum += p_ * temporalReservoir.M;
                    }
                }
            }

            const float misWeight = p_sum == 0.f ? 0.f : p_self / p_sum;
            PathReservoir neighborReservoir;
            if (i == curSampleId)
            {
                neighborReservoir = tempDstReservoir;
            }
            else
            {
                neighborReservoir = temporalReservoir;
            }
            mergeReservoirWithResamplingMIS(
                params, tempDstReservoir.F, dstJacobian, dstReservoir, tempDstReservoir, neighborReservoir, sg, false, misWeight);
        }

        if (dstReservoir.weight > 0)
        {
            dstReservoir.finalizeGRIS();
        }

        // After MIS
        if (dstReservoir.weight < 0.f || isinf(dstReservoir.weight) || isnan(dstReservoir.weight))
        {
            dstReservoir.weight = 0.f;
        }
        outputReservoirs[centralOffset] = dstReservoir;
        //color = dstReservoir.F * dstReservoir.weight;

        // Does not use direct lighting, so we can skip it.
        // Does not need to render to outputColor, so we can skip it.
    }

    /** Entry point for merge pass.
        \param[in] pixel Pixel coordinates.
    */
    void execute(const uint2 _pixel)
    {
        const uint2 pixel = _pixel;

        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);
        logSetPixel(pixel);

        ReSTIR(pixel);
    }
};

cbuffer CB
{
    PathReusePass gPathReusePass;
}


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gPathReusePass.execute(dispatchThreadId.xy);
}
