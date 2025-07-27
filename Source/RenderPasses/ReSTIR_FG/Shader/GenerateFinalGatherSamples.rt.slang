#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Material.ShadingUtils;
import Rendering.Materials.Microfacet;
import Rendering.Materials.Fresnel;
import Rendering.Lights.LightHelpers;
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Color.ColorHelpers;

import Reservoir;
import FinalGatherData;
import SurfaceDataFG;
import Hash;

cbuffer PerFrame
{
    uint gFrameCount;       // Frame count since scene was loaded.
    float2 gCollectionRadius;
    float gHashScaleFactor; //Hash scale factor. 1/diameter
    float gAttenuationRadius;
    uint3 _pad;
}

cbuffer Constant
{
    uint gHashSize;     //Size is always 2^x
    bool gUseAlphaTest; //Alpha test
    bool gDeltaRejection;   //Accept every non delta surface as diffuse
    uint _pad2;
}  

// Inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gView;
Texture2D <float>gLinZ;

// Outputs
RWStructuredBuffer<SampleGenerator> gSampleGenState;
RWStructuredBuffer<PackedSurfaceDataFG> gSurfaceData;
RWTexture2D<PackedHitInfo> gFinalGatherHit;
RWTexture2D<uint> gPhotonCullingMask;
RWTexture2D<float4> gThp;

// Outputs
#if USE_REDUCED_RESERVOIR_FORMAT
RWTexture2D<uint2> gReservoir;
#else
RWTexture2D<uint4> gReservoir;
#endif

static const bool kUsePhotonCulling = USE_PHOTON_CULLING;
static const bool kUseCausticPhotons = USE_CAUSTIC_CULLING;

static const float kTraceRoughnessMin = MAT_ROUGHNESS_CUTOFF_MIN;
static const float kTraceRoughnessMax = MAT_ROUGHNESS_CUTOFF_MAX;
static const bool kRequireDiffuseParts = MAT_REQUIRE_DIFFUSE_PART;
static const float kTraceDiffuseCutoff = MAT_DIFFUSEPART_CUTOFF;

struct TraceData
{
    PackedHitInfo packedHitInfo; //< Reflection albedo of the current hit point
    float distance; ///<Current hit distance
    
    /** Create ray payload with default parameters.
    */
    static TraceData create()
    {
        TraceData d;
        d.packedHitInfo = { };
        d.distance = -1;
        return d;
    }
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

void cullingGetArrayOfCells(float3 position, float radius, out int3 outCells[8])
{
    float3 cell = (position.xyz * gHashScaleFactor);
    float radInScale = radius * gHashScaleFactor;
    float3 cellFloor = floor(cell);
    float3 relCellPos = abs(cell - cellFloor);

    // Get offset direction
    int3 offsetCell;
    offsetCell.x = relCellPos.x < 0.5 ? -1 : 1;
    offsetCell.y = relCellPos.y < 0.5 ? -1 : 1;
    offsetCell.z = relCellPos.z < 0.5 ? -1 : 1;

    // Fill the cell array
    outCells[0] = int3(cellFloor);
    outCells[1] = int3(floor(cell + radInScale * float3(offsetCell.x, 0, 0)));
    outCells[2] = int3(floor(cell + radInScale * float3(offsetCell.x, offsetCell.y, 0)));
    outCells[3] = int3(floor(cell + radInScale * float3(offsetCell.x, offsetCell.y, offsetCell.z)));
    outCells[4] = int3(floor(cell + radInScale * float3(offsetCell.x, 0, offsetCell.z)));
    outCells[5] = int3(floor(cell + radInScale * float3(0, offsetCell.y, 0)));
    outCells[6] = int3(floor(cell + radInScale * float3(0, offsetCell.y, offsetCell.z)));
    outCells[7] = int3(floor(cell + radInScale * float3(0, 0, offsetCell.z)));
}

void photonCulling(float3 hitPos, float radius)
{
    // Get the 7 nearest neighbors (photon can still be in it)
    int3 cells[8];
    cullingGetArrayOfCells(hitPos, radius, cells);

    // Insert them into the hash buffer
    [unroll]
    for(uint i = 0; i < 8; ++i)
    {
        uint h = hash(cells[i]) & (gHashSize - 1);    //calculateHash
        gPhotonCullingMask[ReSTIR_LinearIndexToZCurve(h)] = 1;
    }
}

[shader("miss")]
void miss(inout TraceData rayData : SV_RayPayload)
{
    rayData.distance = -1.f;
}

//Use anyhit for alpha test if activated
[shader("anyhit")]
void anyHit(
    inout TraceData rayData : SV_RayPayload,
    BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    if (gUseAlphaTest)
    {
        // Alpha test for non-opaque geometry.
        GeometryInstanceID instanceID = getGeometryInstanceID();
        VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
        uint materialID = gScene.getMaterialID(instanceID);
        if (gScene.materials.alphaTest(v, materialID, 0.f))
        {
            IgnoreHit();
        }
    }
}

[shader("closesthit")]
void closestHit(
    inout TraceData rayData : SV_RayPayload,
    BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Evaluate Falcor's material parameters at the hit point.
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    
    rayData.distance = RayTCurrent();
    rayData.packedHitInfo = HitInfo(triangleHit).getData();
}

[shader("raygeneration")]
void rayGen()
{
    // Get the launch index and dimensions
    const uint2 launchIndex = DispatchRaysIndex().xy;
    const uint2 launchDim = DispatchRaysDimensions().xy;

    // Load data and initialize variables
    // Default to invalid state
    const uint linearIdx = index2Dto1D(launchIndex, launchDim.x);
    HitInfo hitInfo = HitInfo(gVBuffer[launchIndex]);
    SurfaceFG surface = SurfaceFG();
    Reservoir reservoir = Reservoir();
    TraceData traceData = TraceData::create();

    // Get sample generator
#if STORE_SAMPLE_GEN_STATE
    SampleGenerator sg = gSampleGenState[linearIdx];
#else
    SampleGenerator sg = SampleGenerator(launchIndex, gFrameCount + 3 * 32);
#endif

    // Only trace final gather sample if previous ather hit is valid
    if (hitInfo.isValid())
    {
        const float4 viewTmp = gView[launchIndex];
        const float viewDepth = gLinZ[launchIndex];
        TraceFinalGatherSample(viewTmp, viewDepth, sg, hitInfo, surface, reservoir, traceData);
    }
    
    // Final gather data
    gFinalGatherHit[launchIndex] = traceData.packedHitInfo;
    gSurfaceData[linearIdx] = surface.pack();

    // Store sample generator state
#if STORE_SAMPLE_GEN_STATE
    gSampleGenState[linearIdx] = sg;
#endif // STORE_SAMPLE_GEN_STATE

    // Store reservoir
#if USE_REDUCED_RESERVOIR_FORMAT
    gReservoir[launchIndex] = reservoir.packReduced();
#else
    gReservoir[launchIndex] = reservoir.pack();
#endif
}

void TraceFinalGatherSample(
    const float4 viewTmp,
    const float viewDepth,
    inout SampleGenerator sg,
    inout HitInfo hitInfo,
    inout SurfaceFG surface,
    inout Reservoir reservoir,
    inout TraceData traceData)
{
    // Initialize variables
    float3 view = -viewTmp.xyz;
    const bool collectDirect = viewTmp.w == 0;
    let lod = ExplicitLodTextureSampler(0.f);

    // Trace first hit
    // Load shading data
    ShadingData sd = loadShadingData(hitInfo, view, lod);

    // Sample first BSDF
    sd.mtl.setActiveLobes((uint)LobeType::Reflection); // Use only BRDF
    static const uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
    const IMaterialInstance bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
    BSDFProperties bsdfProperties = bsdf.getProperties(sd);

    BSDFSample bsdfSample;
    bool validSample = bsdf.sample(sd, sg, bsdfSample);
    float3 thp = bsdfSample.weight;
    if (!validSample)
    {
        traceData.packedHitInfo = {};
        reservoir.weightSum = 0.f; // invalidate reservoir
        return;
    }

    // Trace ray to first potential final gather hit
    RayDesc tRay;
    tRay.Origin = sd.computeNewRayOrigin();
    tRay.TMin = 0.0001f;
    tRay.TMax = 10000.f;
    tRay.Direction = bsdfSample.wo;

    TraceRay(
            gScene.rtAccel,
            0,
            0xff /* instanceInclusionMask */,
            0 /* hitIdx */,
            rayTypeCount,
            0 /* missIdx */,
            tRay,
             traceData);
    validSample = traceData.distance > 0;
    if (!validSample)
    {
        traceData.packedHitInfo = {};
        reservoir.weightSum = 0.f; // invalidate reservoir
        return;
    }
    hitInfo = HitInfo(traceData.packedHitInfo);
    float dist = traceData.distance;

    // Continue tracing until diffuse hit if enabled
    bool firstHit = true;
#if TRACE_FG_UNTIL_DIFFUSE
    view = -bsdfSample.wo;
    TraceData currentTraceData = traceData;
    ShadingData prevSd;
    BSDFProperties prevBsdfProperties;
    BSDFSample prevBsdfSample;

    uint i = 0;
    for (; hitInfo.isValid() && i < 16; ++i)
    {
        ShadingData currentSd = loadShadingData(hitInfo, view, lod);

        // Sample BSDF at current hit
        currentSd.mtl.setActiveLobes((uint)LobeType::Reflection);
        IMaterialInstance currentBsdf = gScene.materials.getMaterialInstance(currentSd, lod, hints);
        BSDFProperties currentBsdfProperties = currentBsdf.getProperties(currentSd);

        // Check if the current hit is diffuse - if so, we're done
        if (IsDiffuseBsdf(currentBsdfProperties))
        {
            break;
        }

        // Sample the BSDF to continue tracing
        BSDFSample currentBsdfSample;
        bool currentValidSample = currentBsdf.sample(currentSd, sg, currentBsdfSample);
        if (!currentValidSample)
        {
            validSample = false;
            break;
        }

        // Update throughput and distance
        thp *= currentBsdfSample.weight;
        dist += currentTraceData.distance;

        // Setup next ray
        view = -currentBsdfSample.wo;
        tRay.Origin = currentSd.computeNewRayOrigin();
        tRay.Direction = currentBsdfSample.wo;

        // Trace next ray
        TraceRay(
            gScene.rtAccel,
            0,
            0xff /* instanceInclusionMask */,
            0 /* hitIdx */,
            rayTypeCount,
            0 /* missIdx */,
            tRay,
            currentTraceData);

        if (currentTraceData.distance <= 0)
        {
            validSample = false;
            break;
        }

        // Update for next iteration
        hitInfo = HitInfo(currentTraceData.packedHitInfo);
        prevSd = currentSd;
        prevBsdfProperties = currentBsdfProperties;
        prevBsdfSample = currentBsdfSample;
    }

    // Update
    if (validSample && i > 0)
    {
        gThp[launchIndex].xyz *= thp;
        gThp[launchIndex].w += dist;

        traceData = currentTraceData;
        sd = prevSd;
        bsdfProperties = prevBsdfProperties;
        bsdfSample = prevBsdfSample;

        firstHit = false;
    }
#endif // TRACE_FG_UNTIL_DIFFUSE

    // Fill Surface Buffer used for calculating the target PDF
    surface = SurfaceFG::create(
        sd.computeNewRayOrigin(),
        bsdfProperties.guideNormal,
        sd.V,
        viewDepth,
        bsdfProperties.roughness,
        luminance(bsdfProperties.diffuseReflectionAlbedo),
        luminance(bsdfProperties.specularReflectance));
    reservoir.targetFunc = surface.getPdf(bsdfSample.wo);
    reservoir.weightSum = 1.f / bsdfSample.pdf; // 1/selectPDF

    // Final gather sample here
    reservoir.M = 1;
    if (kUsePhotonCulling)
    {
        const float3 hitPos = tRay.Origin + tRay.Direction * traceData.distance;
        photonCulling(hitPos, gCollectionRadius.x);
        if (firstHit && (kUseCausticPhotons || collectDirect))
        {
            const float radius = gCollectionRadius.x;
            photonCulling(sd.posW, radius);
        }
    }
}

bool IsDiffuseBsdf(const in BSDFProperties bsdfProperties)
{
    const float3 diffuseProb = bsdfProperties.diffuseReflectionAlbedo / (bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.specularReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo + bsdfProperties.specularTransmissionAlbedo);
    const float meanDiffuseProb = (diffuseProb.x + diffuseProb.y + diffuseProb.z) / 3;
    const bool hasDiffuseParts = kRequireDiffuseParts ? (meanDiffuseProb > kTraceDiffuseCutoff) || bsdfProperties.roughness > kTraceRoughnessMax : true; // Check if surface is diffuse or above the roughness threshold
    const bool diffuseHit = hasDiffuseParts && bsdfProperties.roughness > kTraceRoughnessMin; // && !transmission; // Material is only diffuse if it is over the roughness min
    return diffuseHit;
}
