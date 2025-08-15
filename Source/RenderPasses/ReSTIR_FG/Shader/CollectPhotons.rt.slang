#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"
#include "nvapi/nvHLSLExtns.h"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Material.ShadingUtils;
import Rendering.Lights.LightHelpers;
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Color.ColorHelpers;

import Reservoir;
import SurfaceDataFG;
import FinalGatherData;
import CausticGuiding;
import Hash;

// For syntax highlighting purposes
#ifndef MODE_FINAL_GATHER
#define MODE_FINAL_GATHER 0
#endif

// For syntax highlighting purposes
#ifndef CAUSTIC_COLLECTION_MODE
#define CAUSTIC_COLLECTION_MODE 3
#endif

// Stochastic Collect
#ifndef USE_STOCHASTIC_COLLECT
#define USE_STOCHASTIC_COLLECT 0
#endif

#ifndef STOCH_NUM_PHOTONS
#define STOCH_NUM_PHOTONS 3
#endif

cbuffer PerFrame
{
    uint gFrameCount;         // Frame count since scene was loaded.
    float gAttenuationRadius; // Radius for the distance attenuation
    float2 gPhotonRadius; // x->Global Radius, y-> Caustic Radius
    bool gCollectCaustic;   // Collect flag for caustic photons
    bool gCollectFG;        // Collect flag for final gather sample
}

// Constant settings for the temporal filter. Can be null
cbuffer TemporalFilter
{
    uint gTemporalFilterHistoryLimit;
    float gDepthThreshold;              // Depth Threshold
    float gNormalThreshold;             // Normal Threshold
    uint _pad;
}

cbuffer GaussianPhotonGuiding
{
    uint gMaxFirstHitPhotonCount;
    uint gGpgGlobalPhotonWeight;
    uint gGpgCausticPhotonWeight;
}

cbuffer AdaptiveLightSampler
{
    uint gAlsGlobalPhotonWeight;
    uint gAlsCausticPhotonWeight;
}

cbuffer CausticGaussianGuiding
{
    uint gCausticHashGridSize;
    float gCausticHashScalingFactor;
}

// Inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gView;
Texture2D<PackedHitInfo> gFinalGatherHit;
StructuredBuffer<AABB> gPhotonAABB[2];
StructuredBuffer<PhotonDataPacked> gPackedPhotonData[2];

// Caustic Filter
StructuredBuffer<PackedSurfaceDataFG> gSurface;
RWTexture2D<uint4> gCausticReservoir;
RWStructuredBuffer<CausticSample> gCausticSample;
RWTexture2D<uint4> gDirectFGReservoir;
RWStructuredBuffer<CausticSample> gDirectFGSample;

// Inputs Temporal Caustic Filter (can be null)
Texture2D<float2> gMVec;
Texture2D<uint2> gCausticSurface;
Texture2D<uint2> gCausticSurfacePrev;
Texture2D<float4> gCausticPrev;

// 3D gaussian photon guiding
Buffer<uint> gPhotonFirstHitMap[2];
RWBuffer<uint> gFirstHitCollectionCounts;

// Adaptive light sampler
RWBuffer<uint> gLeafSampleCount;
Buffer<uint> gPhotonLeafMap[2];

// Caustic gaussian guiding
StructuredBuffer<CausticSource> gCausticSources;
RWStructuredBuffer<CausticGaussianReservoir> gCausticGaussianReservoirs;
RWBuffer<uint> gCausticHashGridCounter;

// Outputs
RWStructuredBuffer<SampleGenerator> gSampleGenState;
RWTexture2D<float4> gColor;
RWStructuredBuffer<FGData> gFGSampleData;
RWTexture2D<float4> gCausticOut;

#if USE_REDUCED_RESERVOIR_FORMAT
RWTexture2D<uint2> gReservoir;
#else
RWTexture2D<uint4> gReservoir;
#endif

// Acceleration Structure
RaytracingAccelerationStructure gPhotonAS;

// Constant defines
static const bool kCollectCaustics = CAUSTIC_COLLECTION_MODE != CAUSTIC_COLLECT_MODE_NONE;
static const bool kCollectCausticsForIndirect = CAUSTIC_COLLECTION_INDIRECT;

static const bool kRejectDiffuseSurfaces = REJECT_FGSAMPLE_DIFFUSE_SURFACE; // Uses same definition as photon generation
static const bool kEmissionToCausticFilter = EMISSION_TO_CAUSTIC_FILTER;

static const uint kRayFlags = RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_SKIP_TRIANGLES;

// Optimized constants for better performance
static const float kInvPi = M_1_PI;
static const float kMinCosThresholdRcp = rcp(kMinCosTheta);
static const float kSurfaceAlignmentThreshold = 0.6f;

struct RayDataReservoir
{
    PackedSurfaceDataFG surface;
    int idx = -1;
    uint _pad;
    Reservoir r;
    SampleGenerator sg;
}

/** Optimized payload for photon collection. Collect all (~44B) and Stochastic Collect (~28B) available
*/
#if USE_STOCHASTIC_COLLECT
struct RayData
{
    uint counter;                         // Counter for photons this pixel (4B)
    uint photonIdx[STOCH_NUM_PHOTONS];   // Num Photons, variable length (12B for 3 photons)
    SampleGenerator sg;                   // Per-ray state for the sample generator (up to 16B)
  
    __init()
    {
        this.counter = 0;
        // Optimized initialization with unroll hint
        [unroll] for (uint i = 0; i < STOCH_NUM_PHOTONS; ++i)
        {
            this.photonIdx[i] = 0;
        }
    }
};
#else // USE_STOCHASTIC_COLLECT
struct RayData
{
    float3 radiance;          // Accumulated outgoing radiance from path (12B)
    uint countPhotons;        // Counts the photons (4B)
    PackedHitInfo packedHitInfo; // Hit info from vBuffer (Up to 16B)
    SampleGenerator sg;       // Per-ray state for the sample generator (up to 16B)
  
    __init()
    {
        this.radiance = float3(0);
        this.countPhotons = 0;
    }
};
#endif // USE_STOCHASTIC_COLLECT

struct SphereAttribs
{
    float radiusSq;
};

// Load in shading data from hit and direction
ShadingData loadShadingData(const HitInfo hit, const float3 rayDir, const ITextureSampler lod)
{
    const TriangleHit triangleHit = hit.getTriangleHit();
    VertexData v = gScene.getVertexData(triangleHit);
    uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);
    
    return sd;
}

CausticSurface getSurface(uint2 idx, bool prev)
{
    uint2 surfaceData = prev ? gCausticSurfacePrev[idx] : gCausticSurface[idx];
    return CausticSurface(surfaceData);  
}

// Optimized photon contribution calculation
float3 computePhotonContribution(float3 f_r, float3 flux, float NdotL, float radiusWeight)
{
    return radiusWeight * f_r * flux * rcp(NdotL);
}

[shader("miss")]
void miss(inout RayData rayData : SV_RayPayload)
{
    // Empty. Is needed for compilation
}

[shader("miss")]
void missRes(inout RayDataReservoir rayDataRes : SV_RayPayload)
{
    // Empty. Is needed for compilation
}

[shader("anyhit")]
void anyHitReservoir(
    inout RayDataReservoir rayDataRes : SV_RayPayload,
    SphereAttribs attribs : SV_IntersectionAttributes)
{
    const uint primIndex = PrimitiveIndex();
    const uint instanceIdx = InstanceIndex();

    // Early exit for invalid instances
    if (instanceIdx > 1) 
    {
        IgnoreHit();
        return;
    }

    // Get Photon data - cache for reuse
    PhotonData pd = PhotonData(gPackedPhotonData[instanceIdx][primIndex]);
    SurfaceFG surface = SurfaceFG(rayDataRes.surface, -WorldRayDirection());

    // Early exit for first hit requirement
    if (instanceIdx == 0 && !pd.isFirstHit)
    {
        IgnoreHit();
        return;
    }

    // Early exit for hemisphere check
    float NdotL = dot(pd.dir, surface.normal);
    if (NdotL < kMinCosTheta) 
    {
        IgnoreHit();
        return;
    }

    // RIS resampling - optimized with reciprocal
    float target = surface.getPdf(pd.dir, luminance(pd.flux)) * rcp(NdotL);
    float photonWeight = kInvPi * rcp(attribs.radiusSq);
    if (rayDataRes.r.updateReservoir(photonWeight, target, sampleNext1D(rayDataRes.sg)))
    {
        rayDataRes.idx = primIndex;
    }
    IgnoreHit();
}

void TrackPhotonCollection(const uint instanceIdx, const uint primitiveIdx)
{
    const uint firstHitIdx = gPhotonFirstHitMap[instanceIdx][primitiveIdx];
    if (firstHitIdx < gMaxFirstHitPhotonCount)
    {
        InterlockedAdd(gFirstHitCollectionCounts[firstHitIdx], instanceIdx == 0 ? gGpgGlobalPhotonWeight : gGpgCausticPhotonWeight);
    }
}

void TrackPhotonForAdaptiveLightSampler(const uint instanceIdx, const uint primitiveIdx)
{
    const uint leafNodeIdx = gPhotonLeafMap[instanceIdx][primitiveIdx];
    if (leafNodeIdx != 0xFFFFFFFF)
    {
        InterlockedAdd(gLeafSampleCount[leafNodeIdx], instanceIdx == 0 ? gAlsGlobalPhotonWeight : gAlsCausticPhotonWeight);
    }
}

void TrackPhotonForCausticGaussianGuiding(const uint causticPrimIdx, const float lum, inout SampleGenerator sg)
{
    // Check if caustic source is valid
    if (!gCausticSources[causticPrimIdx].valid)
    {
        return;
    }

    // Calc hash grid index
    const uint hashGridIdx = hash2(gCausticSources[causticPrimIdx].srcPos * gCausticHashScalingFactor) % gCausticHashGridSize;

    // Stop if this is not the first access to this hash grid cell
    uint hashAccessIdx = 0xFFFFFFFF; // Init to not 0
    InterlockedAdd(gCausticHashGridCounter[hashGridIdx], 1u, hashAccessIdx);
    if (hashAccessIdx != 0)
    {
        return;
    }

    // Update reservoir
    gCausticGaussianReservoirs[hashGridIdx].Update(gCausticSources[causticPrimIdx].targetPos, lum, sg);
}

// Photon Collection via anyHit
#if USE_STOCHASTIC_COLLECT

[shader("anyhit")]
void anyHit(inout RayData rayData : SV_RayPayload, SphereAttribs attribs : SV_IntersectionAttributes)
{
    uint primIndex = PrimitiveIndex();
    const uint instanceIndex = InstanceIndex();

    // Early exit for invalid instances
    if (instanceIndex > 1) 
    {
        IgnoreHit();
        return;
    }

    // Get Photon data - cache for reuse
    PhotonData pd = PhotonData(gPackedPhotonData[instanceIndex][primIndex]);

    // Change the last bit of prim index to reflect the type
    primIndex = (primIndex & 0x7FFFFFFF) | (instanceIndex << 31);

    // Early exit for first hit requirement
    if ((rayData.counter >> 24 & 1) != 0 && !pd.isFirstHit)
    {
        IgnoreHit();
        return;
    }

    // Pre-compute and cache frequently used values
    const float3 worldRayDir = WorldRayDirection();
    const float pdNormalDotRay = dot(pd.normal, worldRayDir);
    
    // Early exit for surface orientation
    if (pdNormalDotRay < kSurfaceAlignmentThreshold)
    {
        IgnoreHit();
        return;
    }
    
    // Early exit for hemisphere check
    if (dot(pd.dir, worldRayDir) <= 0)
    {
        IgnoreHit();
        return;
    }

    // Optimized reservoir sampling
    uint counter = rayData.counter & 0xFFFFFF;  // Mask out the flags
    uint idx = counter;
    counter++;
    rayData.counter++;

    // Reservoir sampling with early exit
    if (idx >= STOCH_NUM_PHOTONS)
    {
        idx = uint(sampleNext1D(rayData.sg) * counter);
    }
    
    // Insert if index is within maximum list size
    if (idx < STOCH_NUM_PHOTONS)
    {
        rayData.photonIdx[idx] = primIndex;
    }

    IgnoreHit();
}

#else // USE_STOCHASTIC_COLLECT | Collect all photons

[shader("anyhit")]
void anyHit(inout RayData rayData : SV_RayPayload, SphereAttribs attribs : SV_IntersectionAttributes)
{
    const uint instanceIdx = InstanceIndex();
    const uint primIndex = PrimitiveIndex();

    // Early exit for invalid instances
    if (instanceIdx > 1) 
    {
        IgnoreHit();
        return;
    }

    // Get Photon data - cache for reuse
    PhotonData pd = PhotonData(gPackedPhotonData[instanceIdx][primIndex]);
        
    // Early exit for first hit requirement
    if ((rayData.countPhotons >> 24 & 1) != 0 && !pd.isFirstHit)
    {
        IgnoreHit();
        return;
    }
    
    // Get hit data from payload
    const HitInfo hit = HitInfo(rayData.packedHitInfo);
    let lod = ExplicitLodTextureSampler(0.f);

    // Load shading data once
    ShadingData sd = loadShadingData(hit, WorldRayDirection(), lod);
    
    // Early exit for surface orientation
    if (dot(pd.normal, sd.faceN) < kSurfaceAlignmentThreshold)
    {
        IgnoreHit();
        return;
    }

    // Cache material instance creation
    uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;
    let bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
    let bsdfProperties = bsdf.getProperties(sd);
    
    // Pre-compute BSDF evaluation
    float3 f_r = bsdf.eval(sd, pd.dir, rayData.sg);
    float NdotL = dot(bsdfProperties.guideNormal, pd.dir);
    
    // Early exit for invalid lighting
    if (NdotL <= kMinCosTheta)
    {
        IgnoreHit();
        return;
    }
    
    // Optimized radiance calculation using reciprocal
    float radiusWeight = rcp(attribs.radiusSq);
    const float3 photonContribution = max(0.f, computePhotonContribution(f_r, pd.flux, NdotL, radiusWeight));
    rayData.radiance += photonContribution;

    // Conditional tracking to reduce overhead
#if TRACK_FIRST_HIT_PHOTONS
    TrackPhotonCollection(instanceIdx, primIndex);
#endif

#if ADAPTIVE_LIGHT_SAMPLER
    TrackPhotonForAdaptiveLightSampler(instanceIdx, primIndex);
#endif

#if CAUSTIC_GAUSSIAN_GUIDING
    if (instanceIdx == 1)
    {
        TrackPhotonForCausticGaussianGuiding(primIndex, luminance(photonContribution), rayData.sg);
    }
#endif

    IgnoreHit();
}

#endif // USE_STOCHASTIC_COLLECT

// Optimized sphere intersection test
bool hitSphere(const float3 center, const float radius, const float3 p)
{
    float3 diff = p - center;
    float distSq = dot(diff, diff);
    return distSq < radius * radius;
}

[shader("intersection")]
void intersection()
{
    const float3 origin = ObjectRayOrigin();
    const uint primIndex = PrimitiveIndex();
    const uint instanceIdx = InstanceIndex();

    // Cache AABB lookup
    AABB photonAABB = gPhotonAABB[instanceIdx][primIndex];
    float radius = (photonAABB.maxPoint.x - photonAABB.minPoint.x) * 0.5f; // Use multiplication instead of division

    // Optimized sphere intersection
    if (hitSphere(photonAABB.center(), radius, origin))
    {
        SphereAttribs attribs;
        attribs.radiusSq = radius * radius;
        ReportHit(RayTCurrent(), 0, attribs);
    }
}

// Optimized stochastic photon contribution calculation
float3 stochPhotonContribution(
    in const ShadingData sd,
    in IMaterialInstance bsdf,
    float3 N,
    inout RayData rayData,
    bool isCaustic)
{
#if USE_STOCHASTIC_COLLECT
    uint maxIdx = min(rayData.counter, STOCH_NUM_PHOTONS);
    float3 radiance = float3(0);

    // Early exit for empty collection
    if (maxIdx == 0) return radiance;

    // Cache values outside loop
    const float radiusScale = isCaustic ? rcp(gPhotonRadius.y * gPhotonRadius.y) : rcp(gPhotonRadius.x * gPhotonRadius.x);
    
    // Unroll small loops for better performance
    [unroll] for (uint i = 0; i < STOCH_NUM_PHOTONS && i < maxIdx; i++)
    {
        uint photonIndex = rayData.photonIdx[i];
        uint instanceIndex = (photonIndex >> 31) & 1;
        photonIndex = photonIndex & 0x7FFFFFFF;
        
        PhotonData pd = PhotonData(gPackedPhotonData[instanceIndex][photonIndex]);

        float3 f_r = bsdf.eval(sd, pd.dir, rayData.sg);
        float NdotL = dot(N, pd.dir);
        
        if (NdotL > kMinCosTheta)
        {
            const float3 photonContribution = max(0.f, computePhotonContribution(f_r, pd.flux, NdotL, radiusScale));
            radiance += photonContribution;

#if TRACK_FIRST_HIT_PHOTONS
            TrackPhotonCollection(instanceIndex, photonIndex);
#endif

#if ADAPTIVE_LIGHT_SAMPLER
            TrackPhotonForAdaptiveLightSampler(instanceIndex, photonIndex);
#endif

#if CAUSTIC_GAUSSIAN_GUIDING
            if (instanceIndex == 1)
            {
                TrackPhotonForCausticGaussianGuiding(photonIndex, luminance(photonContribution), rayData.sg);
            }
#endif // CAUSTIC_GAUSSIAN_GUIDING
        }
    }

    // Optimized weight calculation
    return radiance * (float(rayData.counter) * rcp(float(maxIdx)));
#else
    return float3(0);
#endif
}

void causticPhotonCollection(in ShadingData sd, inout RayData rayData, bool collectDirect, bool validSurface)
{
    const uint2 launchIndex = DispatchRaysIndex().xy;
    const uint2 launchDim = DispatchRaysDimensions().xy;
    let lod = ExplicitLodTextureSampler(0.f);

    float3 causticRadiance = float3(0);
    float3 directRadiance = float3(0);

    // Ray description
    RayDesc ray;
    ray.Origin = sd.posW;
    ray.TMin = 0.0;
    ray.TMax = 0.001;

#if USE_STOCHASTIC_COLLECT
    ray.Direction = sd.faceN; // We take the face Normal as direction to save payload space
#else
    ray.Direction = -sd.V; // We take the view dir as direction to save payload space
#endif // USE_STOCHASTIC_COLLECT

    // Reservoir Payload
#if (CAUSTIC_COLLECTION_MODE == CAUSTIC_COLLECT_MODE_RESERVOIR)
    RayDataReservoir rayDataRes;
    rayDataRes.r = Reservoir();
    rayDataRes.sg = rayData.sg;
    rayDataRes.surface = gSurface[index2Dto1D(launchIndex, launchDim.x)];
    rayDataRes.idx = -1;
    ray.Direction = -sd.V;
#endif

    //  Second Reservoir Payload for directly resampled photons
#if (CAUSTIC_COLLECTION_MODE == CAUSTIC_COLLECT_MODE_RESERVOIR) && RESERVOIR_PHOTON_DIRECT
    RayDataReservoir rayDataResDirect;
    rayDataResDirect.r = Reservoir();
    rayDataResDirect.surface = rayDataRes.surface;
    rayDataResDirect.idx = -1;
    ray.Direction = -sd.V;
#endif

    [branch] if (validSurface)
    {
        // Init material model if stochastic collect is used
#if USE_STOCHASTIC_COLLECT
        uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        let miProperties = mi.getProperties(sd);
#endif // USE_STOCHASTIC_COLLECT

        // Collect Caustic Photons as Reservoirs
#if (CAUSTIC_COLLECTION_MODE == CAUSTIC_COLLECT_MODE_RESERVOIR)
        TraceRay(
            gPhotonAS,
            kRayFlags,
            2 /* instanceInclusionMask */,
            1 /* hitIdx */,
            0 /* rayType count */,
            0 /* missIdx */,
            ray,
            rayDataRes);

        // If a sample is in the reservoir, finalize it
        if(rayDataRes.r.M > 0 && rayDataRes.idx >= 0)
        {
            rayDataRes.r.finalizeSample(1, 1.f);
        }

        rayDataRes.r.M = 1; // Set confidence weight
        rayData.sg = rayDataRes.sg;
#if USE_STOCHASTIC_COLLECT
        ray.Direction = sd.faceN;
#endif 
#else // (CAUSTIC_COLLECTION_MODE == CAUSTIC_COLLECT_MODE_RESERVOIR) | Usual collection
        TraceRay(
            gPhotonAS,
            kRayFlags,
            2 /* instanceInclusionMask */,
            0 /* hitIdx */,
            0 /* rayType count */,
            0 /* missIdx */,
            ray,
            rayData);
#if USE_STOCHASTIC_COLLECT
        causticRadiance += kInvPi * stochPhotonContribution(sd, mi, miProperties.guideNormal, rayData, true);
#else // USE_STOCHASTIC_COLLECT
        causticRadiance += kInvPi * rayData.radiance;
#endif // USE_STOCHASTIC_COLLECT
#endif // (CAUSTIC_COLLECTION_MODE == CAUSTIC_COLLECT_MODE_RESERVOIR)

        // Collect the direct light on some metal surfaces
        [branch] if (collectDirect)
        {
            // Clear Payload with optimized initialization
#if USE_STOCHASTIC_COLLECT
            rayData.counter = 1 << 24; // Combined clear and flag setting
            [unroll] for(uint i = 0; i < STOCH_NUM_PHOTONS; ++i)
            {
                rayData.photonIdx[i] = 0;
            }
#else
            rayData.radiance = float3(0);
            rayData.countPhotons = 1 << 24; // Flag that only the first hit should be collected
#endif

#if (CAUSTIC_COLLECTION_MODE == CAUSTIC_COLLECT_MODE_RESERVOIR) && RESERVOIR_PHOTON_DIRECT
            ray.Direction = -sd.V;
            rayDataResDirect.sg = rayData.sg;
            TraceRay(
                gPhotonAS,
                kRayFlags,
                1 /* instanceInclusionMask */,
                1 /* hitIdx */,
                0 /* rayType count */,
                0 /* missIdx */,
                ray,
                rayDataResDirect);
            if(rayDataResDirect.r.M > 0 && rayDataResDirect.idx >= 0)
            {
                rayDataResDirect.r.finalizeSample(1, 1.f);
            }
            rayDataResDirect.r.M = 1;
            rayData.sg = rayDataResDirect.sg;
#else // (CAUSTIC_COLLECTION_MODE == CAUSTIC_COLLECT_MODE_RESERVOIR) && RESERVOIR_PHOTON_DIRECT
            TraceRay(
                gPhotonAS,
                kRayFlags,
                1 /* instanceInclusionMask */,
                0 /* hitIdx */,
                0 /* rayType count */,
                0 /* missIdx */,
                ray,
                rayData);
#if USE_STOCHASTIC_COLLECT
            rayData.counter &= 0xFFFFFF;    //Delete flag
            directRadiance += kInvPi * stochPhotonContribution(sd, mi, miProperties.guideNormal, rayData, false);
#else // USE_STOCHASTIC_COLLECT
            rayData.countPhotons = 0; //delete flag
            directRadiance += kInvPi * rayData.radiance;
#endif // USE_STOCHASTIC_COLLECT
#endif
        } // if (collectDirect)
    } // if(validSurface)

    // Temporal Caustic Filter (non Reservoir based)
#if (CAUSTIC_COLLECTION_MODE == CAUSTIC_COLLECT_MODE_TEMPORAL)
    // Get Motion Vector and get reprojected pixel
    float2 motionOffset = gMVec[launchIndex] * float2(launchDim);
    float2 prevIdxF = motionOffset + float2(launchIndex);
    float2 rndOffset = sampleNext2D(rayData.sg) - 0.5f;
    prevIdxF += rndOffset;

    int2 tempIndex = int2(round(prevIdxF));
    causticRadiance += directRadiance;

    uint history = 1;
    if (tempIndex.x >= 0 && tempIndex.y >= 0 && tempIndex.x < launchDim.x && tempIndex.y < launchDim.y)
    {
        // Load Surfaces
        CausticSurface currentSurface = getSurface(launchIndex, false);
        CausticSurface prevSurface = getSurface(tempIndex, true);
        
        if (currentSurface.isValidNeighbor(prevSurface, gDepthThreshold, gNormalThreshold))
        {
            // Because the caustic image can be RGBA16F, we need to store the history as float.
            // As values are in a low range this should not introduce any errors
            float4 prev = gCausticPrev[tempIndex];

            // Make reflections more reactive as the reprojection is often wrong
            uint limit = currentSurface.isDelta ? gTemporalFilterHistoryLimit/2 : gTemporalFilterHistoryLimit;
            history = min(limit, uint(round(prev.w)) + history);
            prev.xyz *= max(0, (history - 1));
            causticRadiance += prev.xyz;
            causticRadiance *= rcp(max(1u, history));
        }
    }

    gCausticOut[launchIndex] = float4(causticRadiance, float(history));

    // Store Reservoir
#elif (CAUSTIC_COLLECTION_MODE == CAUSTIC_COLLECT_MODE_RESERVOIR)
    CausticSample currSample = {};
    if (rayDataRes.idx >= 0)
    {
        AABB pAABB = gPhotonAABB[1][rayDataRes.idx];
        PhotonData pd = PhotonData(gPackedPhotonData[1][rayDataRes.idx]);
        currSample.pos = pAABB.center();
        currSample.flux = pd.flux;
        currSample.dir = pd.dir;
    }

    gCausticSample[index2Dto1D(launchIndex, launchDim.x)] = currSample;
    gCausticReservoir[launchIndex] = rayDataRes.r.pack();
#if RESERVOIR_PHOTON_DIRECT
    currSample = {};
    if (rayDataResDirect.idx >= 0)
    {
        AABB pAABB = gPhotonAABB[0][rayDataResDirect.idx];
        PhotonData pd = PhotonData(gPackedPhotonData[0][rayDataResDirect.idx]);
        currSample.pos = pAABB.center();
        currSample.flux = pd.flux;
        currSample.dir = pd.dir;
    }
    gDirectFGSample[index2Dto1D(launchIndex, launchDim.x)] = currSample;
    gDirectFGReservoir[launchIndex] = rayDataResDirect.r.pack();
#else // RESERVOIR_PHOTON_DIRECT
    gCausticOut[launchIndex] = float4((directRadiance), 1.f);
#endif // RESERVOIR_PHOTON_DIRECT

#else // (CAUSTIC_COLLECTION_MODE == CAUSTIC_COLLECT_MODE_RESERVOIR)
    gCausticOut[launchIndex] = float4((causticRadiance + directRadiance), 1.f);
#endif // (CAUSTIC_COLLECTION_MODE == CAUSTIC_COLLECT_MODE_RESERVOIR)
}

// Collect photons for the final gather sample
void fgSampleCollection(float3 genPosW, inout RayData rayData, bool valid)
{
    const uint2 launchIndex = DispatchRaysIndex().xy;
    const uint2 launchDim = DispatchRaysDimensions().xy;
    let lod = ExplicitLodTextureSampler(0.f);

    PackedHitInfo packedHit = gFinalGatherHit[launchIndex];

#if USE_STOCHASTIC_COLLECT
    rayData.counter = 0;
    [unroll] for(uint i = 0; i < STOCH_NUM_PHOTONS; ++i)
    {
        rayData.photonIdx[i] = 0;
    }
#else // USE_STOCHASTIC_COLLECT
    rayData.radiance = float3(0);
    rayData.packedHitInfo = packedHit;
#endif // USE_STOCHASTIC_COLLECT

    HitInfo hit = HitInfo(packedHit);
    valid &= hit.isValid(); // Check if the final gather ray is valid
    BSDFProperties bsdfProperties = {};
    ShadingData sd = {};   

    float3 fgRadiance = float3(0);
    float jacoDeter = -1.f;
    
    [branch] if (valid)
    {
        TriangleHit triangleHit = hit.getTriangleHit();
        VertexData v = gScene.getVertexData(triangleHit);

        RayDesc ray;
        ray.TMin = 0.0;
        ray.TMax = 0.00001;
        ray.Direction = v.posW - genPosW; // Sample dir is used here to save payload space
        float distToFG = length(ray.Direction); // Distance for Jacobian determinant
        ray.Direction *= rcp(distToFG); // Normalize using reciprocal
        ray.Origin = v.posW; // Update origin
        
        uint materialID = gScene.getMaterialID(triangleHit.instanceID);
        sd = gScene.materials.prepareShadingData(v, materialID, -ray.Direction, lod);
        uint hints = (uint) MaterialInstanceHints::AdjustShadingNormal;
        let bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
        bsdfProperties = bsdf.getProperties(sd);

        // Jacobian Determinant
        float NdotV = dot(bsdfProperties.guideNormal,sd.V);
        if(NdotV > kMinCosTheta)
        {
            jacoDeter = vplAttenuation(distToFG, gAttenuationRadius) * NdotV;
        }

#if USE_STOCHASTIC_COLLECT
        // Store faceN in direction
        ray.Direction = sd.faceN;
#endif // USE_STOCHASTIC_COLLECT

        const uint instanceInclusionMask = kCollectCaustics && kCollectCausticsForIndirect ? 0xFF : 1;

        TraceRay(
            gPhotonAS,
            kRayFlags,
            instanceInclusionMask /* instanceInclusionMask */,
            0 /* hitIdx */,
            0 /* rayType count */,
            0 /* missIdx */,
            ray,
            rayData);

#if USE_STOCHASTIC_COLLECT
        fgRadiance += kInvPi * stochPhotonContribution(sd, bsdf, bsdfProperties.guideNormal, rayData, false);
#else // USE_STOCHASTIC_COLLECT
        fgRadiance += kInvPi * rayData.radiance;
#endif // USE_STOCHASTIC_COLLECT
    } // if (valid)

    // Store ReSTIR Sample
 
    // Init reservoir
    Reservoir r = Reservoir(gReservoir[launchIndex]);
    r.M = 1; // Set M to the right value

    const float Li = luminance(fgRadiance);
    r.targetFunc *= Li; // Add the missing parts of the targetFunc (Li)

    // Create the final gather sample
    FGData fgData = {};
    bool fgValid = valid;
    if (fgValid)
    {
        fgValid = !isnan(Li) && !isinf(Li);
        fgData.posW = sd.computeNewRayOrigin();
        fgData.normal = bsdfProperties.guideNormal;
        fgData.flux = fgRadiance;
    }

    // Negative jacobian determinant is an invalid sample
    fgData.jacobianDeterminant = fgValid ? jacoDeter : -1.f;
        
    if (!fgValid)
    {
        r.M = 0;
        r.weightSum = 0.f;
    }

#if USE_REDUCED_RESERVOIR_FORMAT
    gReservoir[launchIndex] = r.packReduced();
#else  //USE_REDUCED_RESERVOIR_FORMAT
    gReservoir[launchIndex] = r.pack();
#endif //USE_REDUCED_RESERVOIR_FORMAT

    gFGSampleData[index2Dto1D(launchIndex, launchDim.x)] = fgData;
}

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    // Early exit for invalid pixels
    if (any(launchIndex >= launchDim)) return;

    // Initialize payload with optimized constructor
    RayData rayData = RayData();
    
#if STORE_SAMPLE_GEN_STATE
    const uint linearIndex = index2Dto1D(launchIndex, launchDim.x);
    rayData.sg = gSampleGenState[linearIndex];
#else // STORE_SAMPLE_GEN_STATE
    rayData.sg = SampleGenerator(launchIndex, gFrameCount + 15 * 32);
#endif // STORE_SAMPLE_GEN_STATE

    // Cache packed hit info
    PackedHitInfo packedHit = gVBuffer[launchIndex];
#if !USE_STOCHASTIC_COLLECT
    rayData.packedHitInfo = packedHit;
#endif // !USE_STOCHASTIC_COLLECT

    HitInfo hit = HitInfo(packedHit);
    const bool valid = hit.isValid();

    // Early exit for invalid hits
    if (!valid && !gCollectFG) 
    {
#if STORE_SAMPLE_GEN_STATE
        gSampleGenState[linearIndex] = rayData.sg;
#endif
        return;
    }

    // Cache vertex data for reuse
    TriangleHit triangleHit = hit.getTriangleHit();
    VertexData v = gScene.getVertexData(triangleHit);
    
    // Branch prediction optimization
    [branch] if (gCollectCaustic)
    {
        float4 viewTmp = gView[launchIndex];
        float3 view = -viewTmp.xyz;
        bool collectDirect = viewTmp.w > 0.1f;

        let lod = ExplicitLodTextureSampler(0.f);
        ShadingData fgSD = {};
        if (valid)
        {
            fgSD = loadShadingData(hit, view, lod);
        }

        causticPhotonCollection(fgSD, rayData, collectDirect, valid);
    }

    [branch] if (gCollectFG)
    {
        fgSampleCollection(v.posW, rayData, valid);
    }

#if STORE_SAMPLE_GEN_STATE
    gSampleGenState[linearIndex] = rayData.sg;
#endif // STORE_SAMPLE_GEN_STATE
}
