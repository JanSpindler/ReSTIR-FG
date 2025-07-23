#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"
#include "Utils/Sampling/SampleGeneratorType.slangh"

import Utils.Sampling.SampleGenerator;
import Utils.Math.MathHelpers;
import Scene.HitInfo;
import Scene.Intersection;
import Scene.Raytracing;
import PrefixPathReservoir;

#if !defined(SAMPLE_GENERATOR_TYPE) || SAMPLE_GENERATOR_TYPE != SAMPLE_GENERATOR_TINY_UNIFORM
#error "This file requires SAMPLE_GENERATOR_TYPE to be SAMPLE_GENERATOR_TINY_UNIFORM."
#endif

// Constant buffer data
cbuffer CB
{
    uint2 gFrameDim;
    bool gEnableTemporalReprojection;
}

// Input/Output resources
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<PackedHitInfo> gVBufferPrev;
Texture2D<float4> gViewDirRayDistDI;
Texture2D<float2> gMotionVectors;
RWStructuredBuffer<PrefixPathReservoir> gCurrentReservoirs;
StructuredBuffer<PrefixPathReservoir> gTemporalReservoirs;

uint index2Dto1D(uint2 idx, uint width)
{
    return idx.x + width * idx.y;
}

// Helper function to load shading data (simplified version)
ShadingData loadShadingData(const HitInfo hit, const float3 rayDir, const ITextureSampler lod)
{
    const TriangleHit triangleHit = hit.getTriangleHit();
    const VertexData v = gScene.getVertexData(triangleHit);
    const uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    const ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);
    return sd;
}

// Temporal resampling using random replay shift
[numthreads(16, 16, 1)]
void main(uint3 pixel : SV_DispatchThreadID)
{
    // Out of bounds
    if (pixel.x >= gFrameDim.x || pixel.y >= gFrameDim.y)
    {
        return;
    }

    // Load the current reservoir
    const uint linearPixelIdx = index2Dto1D(pixel.xy, gFrameDim.x);
    PrefixPathReservoir currentReservoir = gCurrentReservoirs[linearPixelIdx];

    // If the path is invalid or has found diffuse surface on first hit, exit
    if (currentReservoir.path.length <= 1)
    {
        return;
    }

    // Load the hit information from the VBuffer
    const HitInfo hit = HitInfo(gVBuffer[pixel.xy]);

    // Check if we have a valid hit
    if (!hit.isValid())
    {
        return;
    }

    // Load shading data
    const float3 camViewDir = gViewDirRayDistDI[pixel.xy].xyz;
    const ShadingData sd = loadShadingData(hit, -camViewDir, ExplicitLodTextureSampler(0.0f));

    // Perform temporal resampling logic here (simplified)
    

    // Store the updated reservoir back to the buffer
    //gCurrentReservoirs[pixel.x + pixel.y * gCurrentReservoirs.Length] = currentReservoir;
}
