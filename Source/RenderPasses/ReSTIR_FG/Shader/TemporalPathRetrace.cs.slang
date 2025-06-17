/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
import Scene.HitInfo;
import PrefixPathReservoir;
import Scene.Scene;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.Ray;
import Rendering.Materials.MaterialShading;

#if !defined(SAMPLE_GENERATOR_TYPE) || SAMPLE_GENERATOR_TYPE != SAMPLE_GENERATOR_TINY_UNIFORM
#error "This file requires SAMPLE_GENERATOR_TYPE to be SAMPLE_GENERATOR_TINY_UNIFORM."
#endif

static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

struct PathRetracePass
{
    Texture2D<PackedHitInfo> vbuffer;                     ///< Fullscreen V-buffer for the primary hits.
    Texture2D<PackedHitInfo> temporalVbuffer;
    Texture2D<float2> motionVectors;

    RWStructuredBuffer<PathReservoir> outputReservoirs;                  
    RWStructuredBuffer<PathReservoir> temporalReservoirs;

    float temporalHistoryLength;

    bool IsValidPackedHitInfo(PackedHitInfo packed)
    {
        return packed.x != 0;
    }

    bool IsValidScreenRegion(const int2 pixel)
    {
        return all(pixel >= 0 && pixel < params.frameDim);
    }

    ShadingData GetPixelShadingData(const int2 pixel, out PackedHitInfo PrimaryHitPacked)
    {
        ShadingData sd = {};
        Ray ray = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        PrimaryHitPacked = vbuffer[pixel];
        if (isValidPackedHitInfo(PrimaryHitPacked))
        {
            HitInfo PrimaryHit; PrimaryHit.unpack(PrimaryHitPacked);
            sd = loadShadingData(PrimaryHit, -ray.dir, true);
        }
        return sd;
    }

    ShadingData GetPixelTemporalShadingData(const int2 pixel, out PackedHitInfo PrimaryHitPacked)
    {
        ShadingData sd = {};
        Ray ray = gScene.camera.computeRayPinholePrevFrame(pixel, params.frameDim);
        PrimaryHitPacked = temporalVbuffer[pixel];
        if (isValidPackedHitInfo(PrimaryHitPacked))
        {
            HitInfo PrimaryHit; PrimaryHit.unpack(PrimaryHitPacked);
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
        // TODO: How to seed efficiently?
        var sg = TinyUniformSampleGenerator(pixel, (kCandidateSamples + 1 + gNumSpatialRounds) * params.seed + kCandidateSamples);

        // Compute offset into per-sample buffers. All samples are stored consecutively at this offset.
        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = outputReservoirs[centralOffset];

        PackedHitInfo centralPrimaryHitPacked;
        ShadingData centralPrimarySd = getPixelShadingData(pixel, centralPrimaryHitPacked);
        if (!isValidPackedHitInfo(centralPrimaryHitPacked))
        {
            return;
        }

        // color averaging test

        // fetch temporal reservoir
        float2 motionVector = motionVectors[pixel];
        int2 prevPixel = pixel;

        bool foundTemporalSurface = true;

        if (gEnableTemporalReprojection)
        {
            prevPixel = pixel + motionVector * params.frameDim + (sampleNext2D(sg) * 1.f - 0.f);
        }

        if (!isValidScreenRegion(prevPixel) || !foundTemporalSurface) return;

        PackedHitInfo temporalPrimaryHitPacked;
        ShadingData temporalPrimarySd = getPixelTemporalShadingData(prevPixel, temporalPrimaryHitPacked);
        if (!isValidPackedHitInfo(temporalPrimaryHitPacked)) return;

        PathReservoir temporalReservoir = temporalReservoirs[params.getReservoirOffset(prevPixel)];

        // talbot MIS
        // compute mis weight for current pixel

        HitInfo dstRcPrevVertexHit;
        HitInfo dstRcPrevVertexHit2;
        float3 dstRcPrevVertexWo;
        float3 dstRcPrevVertexWo2;
        float3 Tp;
        float3 Tp2;

        // write to memory
        if (centralReservoir.pathFlags.rcVertexLength() > 1)
        {
            Tp = traceHybridShiftRays(params, true, temporalPrimaryHitPacked, temporalPrimarySd, centralReservoir, dstRcPrevVertexHit, dstRcPrevVertexWo);
            reconnectionDataBuffer[centralOffset].data[0] = ReconnectionData(dstRcPrevVertexHit, dstRcPrevVertexWo, Tp);
        }
        if (temporalReservoir.pathFlags.rcVertexLength() > 1)
        {
            Tp2 = traceHybridShiftRays(params, false, centralPrimaryHitPacked, centralPrimarySd, temporalReservoir, dstRcPrevVertexHit2, dstRcPrevVertexWo2);
            reconnectionDataBuffer[centralOffset].data[1] = ReconnectionData(dstRcPrevVertexHit2, dstRcPrevVertexWo2, Tp2);
        }
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
    PathRetracePass gPathRetracePass;
}


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gPathRetracePass.execute(dispatchThreadId.xy);
}
