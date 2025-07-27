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

// Outputs
RWTexture2D<PackedHitInfo> gOutVBuffer;
RWTexture2D<float4> gOutViewDir;
RWTexture2D<float> gOutRayDist;
RWTexture2D<float4> gOutThp;
RWTexture2D<uint2> gPackedCausticSurface;

// Outputs RTXDI (ReSTIR)
RWTexture2D<PackedHitInfo> gOutVBufferDI;
RWTexture2D<float4> gOutViewDirRayDistDI;
RWTexture2D<float4> gOutThpDI;

// ReSTIR
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<PackedHitInfo> gVBufferPrev;
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

    // Load reseroirs
    const PrefixPathReservoir currentReservoir = gCurrentReservoirs[linearPixelIdx];
    const PrefixPathReservoir prevReservoir = gTemporalReservoirs[prevLinearPixelIdx];

    // Check for reasonable path lengths
    const uint currentLength = currentReservoir.path.GetLength();
    const uint prevLength = prevReservoir.path.GetLength();
    if (currentLength <= 1 || currentLength != prevLength)
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
    const float3 camViewDir = -gScene.camera.computeRayPinhole(pixel, gFrameDim).dir;
    const float3 prevCamViewDir = camViewDir; // TODO: Should use proper previous camera view direction
    const ShadingData currentSd = LoadShadingData(currentHit, -camViewDir, ExplicitLodTextureSampler(0.0f));
    const ShadingData prevSd = LoadShadingData(prevHit, -prevCamViewDir, ExplicitLodTextureSampler(0.0f));

    // Shift prev path into current domain
    const float3 camPos = gScene.camera.getPosition();
    PrefixPath prevIntoCurrentPath;
    float prevIntoCurrentJacobian;
    const bool shiftSuccessForward = ShiftPrefixPath(
        currentReservoir.path,
        currentSd,
        currentHit,
        prevReservoir.path,
        camPos,
        gRequDiffParts,
        prevIntoCurrentPath,
        prevIntoCurrentJacobian
    );

    // Shift current path into previous domain
    PrefixPath currentIntoPrevPath;
    float currentIntoPrevJacobian;
    const bool shiftSuccessBackward = ShiftPrefixPath(
        prevReservoir.path,
        prevSd,
        prevHit,
        currentReservoir.path,
        camPos,
        gRequDiffParts,
        currentIntoPrevPath,
        currentIntoPrevJacobian
    );

    // Calculate Talbot MIS weights
    float currentMisWeight = shiftSuccessBackward ?
        (currentReservoir.path.GetLuminance()) / (
            (currentReservoir.path.GetLuminance()) +
            (currentIntoPrevPath.GetLuminance() / currentIntoPrevJacobian)
        ) : 0.0f;
    float prevMisWeight = shiftSuccessForward ?
        (prevReservoir.path.GetLuminance()) / (
            (prevReservoir.path.GetLuminance()) +
            (prevIntoCurrentPath.GetLuminance() / prevIntoCurrentJacobian)
        ) : 0.0f;

    // Reservoir confidence
    const float totalConfidence = currentReservoir.confidence + prevReservoir.confidence;
    currentMisWeight *= currentReservoir.confidence / totalConfidence;
    prevMisWeight *= prevReservoir.confidence / totalConfidence;

    // RIS
    // TODO: Optimize RIS process by reducing unnecessary copies
    gCurrentReservoirs[linearPixelIdx].Reset();
    gCurrentReservoirs[linearPixelIdx].Merge(currentReservoir, currentReservoir.path, currentMisWeight, 1.0f, sg);
    gCurrentReservoirs[linearPixelIdx].Merge(prevReservoir, prevIntoCurrentPath, prevMisWeight, prevIntoCurrentJacobian, sg);

    // After RIS write data to buffers
    const PrefixPath finalPath = gCurrentReservoirs[linearPixelIdx].path;
    gOutVBuffer[pixel] = finalPath.fgHitInfo;
    gOutViewDir[pixel] = finalPath.fgViewDir;
    gOutRayDist[pixel] = finalPath.fgRayDist;
    gOutThp[pixel] = float4(finalPath.throughput, finalPath.fgHitT);

#if CAUSTIC_TEMPORAL_FILTER_ENABLED
    gPackedCausticSurface[pixel] = finalPath.causticSurface;
#endif

    gOutVBufferDI[pixel] = finalPath.diHitInfo;
    gOutViewDirRayDistDI[pixel] = finalPath.diViewDir;
    gOutThpDI[pixel] = finalPath.diThroughput;

    // TODO: Check if this is needed
    // Fill the RTXDI surface if it is enabled
//#if USE_RTXDI
//    if (hitInfoDI.isValid())
//    {
//        // Load all needed data
//        const ShadingData sd = loadShadingData(hitInfoDI, -viewDirDistDI.xyz, lod);
//        const uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
//        const let bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
//        const let bsdfProperties = bsdf.getProperties(sd);

//        gRTXDI.setSurfaceData(launchIndex, sd.computeNewRayOrigin(), bsdfProperties.guideNormal, bsdfProperties.diffuseReflectionAlbedo, bsdfProperties.specularReflectance, bsdfProperties.roughness, viewDirDistDI.w);
//    }
//    else
//    {
//        gRTXDI.setInvalidSurfaceData(launchIndex);
//    }
//#endif
}
