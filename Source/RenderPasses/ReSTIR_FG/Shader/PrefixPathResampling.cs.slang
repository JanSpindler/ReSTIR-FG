#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"
#include "Utils/Sampling/SampleGeneratorType.slangh"

import Utils.Sampling.SampleGenerator;
import Utils.Math.MathHelpers;
import Scene.HitInfo;
import Scene.Intersection;
import Scene.Raytracing;
import PrefixPathReservoir;
import LoadShadingData;
import PrefixPathShift;

#if !defined(SAMPLE_GENERATOR_TYPE) || SAMPLE_GENERATOR_TYPE != SAMPLE_GENERATOR_TINY_UNIFORM
#error "This file requires SAMPLE_GENERATOR_TYPE to be SAMPLE_GENERATOR_TINY_UNIFORM."
#endif

// Constant buffer data
cbuffer CB
{
    uint2 gFrameDim;
    uint gFrameCount;
    bool gEnableTemporalReprojection;
    bool gRequDiffParts;
}

// Input/Output resources
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<PackedHitInfo> gVBufferPrev;
Texture2D<float4> gViewDirRayDistDI;
Texture2D<float2> gMotionVectors;
RWStructuredBuffer<PrefixPathReservoir> gCurrentReservoirs;
StructuredBuffer<PrefixPathReservoir> gTemporalReservoirs;

uint Index2Dto1D(uint2 idx, uint width)
{
    return idx.x + width * idx.y;
}

bool IsValidPixel(int2 pixel, uint2 frameDim)
{
    return pixel.x >= 0 && pixel.y >= 0 && 
           pixel.x < (int)frameDim.x && pixel.y < (int)frameDim.y;
}

// Temporal resampling using random replay shift
[numthreads(16, 16, 1)]
void main(uint3 launchIdx : SV_DispatchThreadID)
{
    // Out of bounds
    const uint2 pixel = launchIdx.xy;
    if (any(pixel >= gFrameDim))
    {
        return;
    }

    // Get prev pixel - FIXED motion vector calculation
    int2 prevPixel = (int2)pixel;
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    if (gEnableTemporalReprojection)
    {
        prevPixel = (int2)pixel + (int2)(gMotionVectors[pixel] * (float2)gFrameDim);
        const float2 jitter = (sampleNext2D(sg) - 0.5f) * 2.0f; // [-1, 1]
        prevPixel += (int2)jitter;
    }

    // Skip if previous pixel is out of bounds
    if (!IsValidPixel(prevPixel, gFrameDim))
    {
        return;
    }

    // Load the current reservoir
    const uint linearPixelIdx = Index2Dto1D(pixel, gFrameDim.x);
    const uint prevLinearPixelIdx = Index2Dto1D((uint2)prevPixel, gFrameDim.x);
    
    // SAFETY CHECK: Validate buffer indices
    const uint maxBufferSize = gFrameDim.x * gFrameDim.y;
    if (linearPixelIdx >= maxBufferSize || prevLinearPixelIdx >= maxBufferSize)
    {
        return;
    }
    
    PrefixPathReservoir currentReservoir = gCurrentReservoirs[linearPixelIdx];
    PrefixPathReservoir prevReservoir = gTemporalReservoirs[prevLinearPixelIdx];

    // Check for reasonable path lengths
    if (currentReservoir.path.GetLength() <= 1 || prevReservoir.path.GetLength() <= 1)
    {
        return;
    }

    // Load the hit information from the VBuffer
    const HitInfo currentHit = HitInfo(gVBuffer[pixel]);
    if (!currentHit.isValid())
    {
        return;
    }

    // Load prev hit information
    const HitInfo prevHit = HitInfo(gVBufferPrev[prevPixel]);
    if (!prevHit.isValid())
    {
        return;
    }

    // Load shading data
    const float3 camViewDir = gViewDirRayDistDI[pixel].xyz;
    const float3 prevCamViewDir = camViewDir; // TODO: Should use proper previous camera view direction
    const ShadingData currentSd = LoadShadingData(currentHit, -camViewDir, ExplicitLodTextureSampler(0.0f));
    const ShadingData prevSd = LoadShadingData(prevHit, -prevCamViewDir, ExplicitLodTextureSampler(0.0f));

    // Shift prev path into current domain
    float3 shiftedThroughput;
    float jacobian;
    bool shiftSuccess = ShiftPrefixPath(
        currentReservoir.path,
        currentSd,
        currentHit,
        prevReservoir.path,
        gScene.camera.getPosition(),
        gRequDiffParts,
        shiftedThroughput,
        jacobian
    );
}
