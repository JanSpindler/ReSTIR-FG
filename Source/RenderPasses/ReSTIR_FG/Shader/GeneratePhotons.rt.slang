#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Material.ShadingUtils;
import Rendering.Lights.LightHelpers;
import Utils.Color.ColorHelpers;
import Rendering.Materials.Microfacet;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import FinalGatherData;
import Hash;
import GaussianHelper;

cbuffer PerFrame
{
    uint gFrameCount; // Frame count since scene was loaded.
    float2 gPhotonRadius; //Radius for the vpl collection
    float gHashScaleFactor; //Hash scale factor. 1/Radius
}

cbuffer CB
{
    //
    uint gMaxRecursion; //Max photon recursion depths
    float gRejection; //Probability that a photon is stored as a light
    uint gFlags;       //Flags for the pass. See GenerationFlags enum
    uint gHashSize;     //Size is always 2^x
    int gCausticsBounces;    //Number of diffuse bounces allowed for caustic photons
    float gGenerationLampIntersectGuard;    //Guard for the first hit of photons
    float gGenerationLampIntersectGuardStoreProbability; //Percentage a photon is stored
};

cbuffer GaussianPhotonGuiding
{
    uint gGaussianCount;
    uint gAnalyticLightCount;
    uint gGeometricLightCount;
    float gCs;
    float gB;
    float gGmmMinPdf;
    float gBeta;
    uint gMaxFirstHitPhotonCount;
};

#if USE_EMISSIVE_LIGHT  //Buffer is only valid if emissive light is enabled
cbuffer Light
{
    EmissiveLightSampler gEmissiveSampler;
}
#endif

//
RWStructuredBuffer<AABB> gPhotonAABB[2];
RWStructuredBuffer<PhotonDataPacked> gPackedPhotonData[2];
RWBuffer<uint> gPhotonCounter;
Texture2D<uint> gPhotonCullingMask;

// 3D gaussian photon guiding
RWStructuredBuffer<Gaussian3D> gGaussians;
RWBuffer<uint> gFirstHitPhotonCounter;
RWStructuredBuffer<FirstHitPhotonInfo> gFirstHitPhotonInfo;
RWBuffer<uint> gPhotonFirstHitMap[2];
RWBuffer<uint> gLightFirstHitCounts;
Buffer<float> gSoftmaxWeights;

enum class GenerationFlags // : uint32_t
{
    None                = 0x00,
    AlphaTest           = 0x01,     //Enable Alpha test
    AdjShadingNormals   = 0x02,     //Adjust the shading normals
    EnableCaustics      = 0x04,     //Enable caustic photons.
    AnalyticLight       = 0x20,     //Use analytic light photons
};

static const uint2 kPhotonBufferSize = uint2(PHOTON_BUFFER_SIZE_GLOBAL, PHOTON_BUFFER_SIZE_CAUSTIC);
static const bool kUsePhotonCulling = USE_PHOTON_CULLING;
static const bool kUseCausticCulling = USE_CAUSTIC_CULLING;
static const uint kRayFlagTerminatedMiss = 1;
static const uint kRayFlagTerminatedHit = 2;

static const float kTraceRoughnessMin = MAT_ROUGHNESS_CUTOFF_MIN;
static const float kTraceRoughnessMax = MAT_ROUGHNESS_CUTOFF_MAX;
static const bool kRequireDiffuseParts = MAT_REQUIRE_DIFFUSE_PART;
static const float kTraceDiffuseCutoff = MAT_DIFFUSEPART_CUTOFF;

/** Payload for scatter ray (80B).
*/
struct RayData
{
    float3  thp;            ///< Current path throughput. This is updated at each path vertex.
    uint   packedNormal;   ///< Face normal encoded in 16 bit polar coordinates
    float3  origin;         ///< Next path segment origin.
    uint terminated; ///< Set to true when path is terminated.
    float3  direction;      ///< Next path segment direction.
    bool    diffuseHit;     ///< saves if the his is diffuse
    bool isEmissive;        ///<If the hit is emissive
    float distance;

    SampleGenerator sg;     ///< Per-ray state for the sample generator (up to 16B).

    /** Create ray payload with default parameters.
    */
    static RayData create()
    {
        RayData d;
        d.terminated = 0;
        d.packedNormal = 0;
        d.thp = float3(1);
        d.origin = float3(0);
        d.direction = float3(0);
        d.isEmissive = false;
        d.distance = 0.f;
        return d;
    }
};

struct PhotonInfo
{
    float3 flux;
    float faceNTheta;
    float3 dir;
    float faceNPhi;
};

struct PhotonLightSample
{
    float3 posW;
    float3 normal;
    float3 flux;
};

float3 fromWordlToLocal(const in float3 normal, float3 w)
{
    // reconstruct tangent frame based off worldspace normal
    // this is ok for isotropic BRDFs
    // for anisotropic BRDFs, we need a user defined tangent
    float3 bitangent = perp_stark(normal);
    float3 tangent = cross(bitangent, normal);

    return float3(dot(bitangent, w), dot(tangent, w), dot(normal, w));
}

// transforms direction from local to world space
float3 fromLocalToWorld(const in float3 normal, float3 w)
{
    // transform from local space -> random dir vector in world space
    float3 bitangent = perp_stark(normal);
    float3 tangent = cross(bitangent, normal);

    return w.x * tangent + w.y * bitangent + w.z * normal;
}

// -------------------------- 3D gaussian photon guiding --------------------------

float3 SampleGaussian(const in Gaussian3D gaussian, inout SampleGenerator sg)
{
    const float3 mu = gaussian.mean;
    const float sigma = gaussian.sigma;

    const float r1 = sampleNext1D(sg);
    const float r2 = sampleNext1D(sg);
    const float r3 = sampleNext1D(sg);

    const float theta = 2.0f * PI * r1;
    const float phi = acos(2.0f * r2 - 1.0f);
    const float radius = sigma * pow(-2.0f * log(r3), 0.5f);

    const float3 x = radius * float3(sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi));
    return x + mu;
}

float GMMProb(const float3 pos0, const float3 gmmPos, const uint lightIdx)
{
#if USE_3D_GAUSSIAN_PHOTON_GUIDING

    // Get index of first gaussian belonging to this light
    const uint startGaussIdx = lightIdx * gGaussianCount;

    // Calculate pdf by marginalizing over all gaussians
    float gmmPdf = 0.0f;
    for (uint gaussIdx = 0; gaussIdx < gGaussianCount; ++gaussIdx)
    {
        const uint totalGaussianIdx = startGaussIdx + gaussIdx;
        const Gaussian3D gaussian = gGaussians[totalGaussianIdx];
        const float gaussWeight = gSoftmaxWeights[totalGaussianIdx];
        gmmPdf += F0(gmmPos, pos0, gaussian.mean, gaussian.sigma) * gaussWeight;
    }

    // Return
    return gmmPdf;

#else

    return 0.0f;

#endif
}

float3 SampleGMM(const float3 pos0, const uint lightIdx, inout SampleGenerator sg, out float gmmPdf)
{
    // Get index of first gaussian belonging to this light
    const uint startGaussIdx = lightIdx * gGaussianCount;

    // Select a gaussian and sample position
    const float r = sampleNext1D(sg);
    float sum = 0.0f;
    float3 gmmPos;
    for (uint gaussIdx = 0; gaussIdx < gGaussianCount; ++gaussIdx)
    {
        const uint totalGaussianIdx = startGaussIdx + gaussIdx;
        const Gaussian3D gaussian = gGaussians[totalGaussianIdx];
        const float gaussWeight = gSoftmaxWeights[totalGaussianIdx];
        sum += gaussWeight;
        if (r <= sum)
        {
            gmmPos = SampleGaussian(gaussian, sg);
            break;
        }
    }

    // Calculate pdf by marginalizing over all gaussians
    gmmPdf = GMMProb(pos0, gmmPos, lightIdx);

    // Return
    return gmmPos;
}

// -----------------------------------------------------------------------

#if USE_EMISSIVE_LIGHT  // Function is only valid if the emissive sampler is set
bool createLightSampleGeometric(inout SampleGenerator sg, out PhotonLightSample pls, out uint lightIdx)
{
    // Init
    pls = {};

    // Return if not emissive light is present
    if (gScene.lightCollection.isEmpty())
    {
        lightIdx = 0;
        return false;
    }

    // Randomly pick a triangle out of the global list with uniform probability
    const uint triangleCount = gScene.lightCollection.triangleCount;

    // Safety precaution as the result of the multiplication may be rounded to triangleCount even if uLight < 1.0 when triangleCount is large.
    uint triangleIndex = min((uint)(sampleNext1D(sg) * triangleCount), triangleCount - 1);

    // Get light index from triangle index
    lightIdx = gScene.lightCollection.getLightIdx(triangleIndex);

    // Get the triangle index from the alias table
    const uint2 packed = gEmissiveSampler._emissivePower.triangleAliasTable[triangleIndex];
    const float threshold = f16tof32(packed.x >> 16u);
    const uint selectAbove = ((packed.x & 0xFFFFu) << 8u) | ((packed.y >> 24u) & 0xFFu);
    const uint selectBelow = packed.y & 0xFFFFFFu;

    // Test the threshold in the current table entry; pick one of the two options
    triangleIndex = (sampleNext1D(sg) >= threshold) ? selectAbove : selectBelow;

    // Store pdf value of sampled triangle
    const float triangleSelectionPdf = gScene.lightCollection.fluxData[triangleIndex].flux * gEmissiveSampler._emissivePower.invWeightsSum;

    // Get the triangle
    const EmissiveTriangle tri = gScene.lightCollection.getTriangle(triangleIndex);
    const float3 barycentrics = sample_triangle(sampleNext2D(sg));
    pls.posW = tri.getPosition(barycentrics);
    pls.normal = tri.normal;

    // Get Flux
    float2 uv = tri.getTexCoord(barycentrics);
    pls.flux = gScene.materials.evalEmissive(tri.materialID, uv) * tri.area;

    // If the material is double sided, randomly flip the normal
    //MaterialHeader mltHeader = gScene.materials.getMaterialHeader(tri.materialID);
    //if (mltHeader.isDoubleSided())
    //{
    //    pls.flux *= 2;
    //    if (sampleNext1D(sg) >= 0.5f)
    //        pls.normal = -tri.normal;
    //}

    // Return
    pls.posW = computeRayOrigin(pls.posW, pls.normal);
    pls.flux /= triangleSelectionPdf;
    return true;
}
#endif

float3 SampleAnalyticLightDirUniform(
    inout SampleGenerator sg,
    const in LightData lightData,
    out float invLightPdf)
{
    float3 woLocal;
    float3 rnd = sampleNext3D(sg);
    rnd.z = 0; // No smooth fade at edge of spotlight
    float spotAngle = lightData.cosOpeningAngle;

    if (lightData.openingAngle < M_2PI) // If the opening angle is smaller than 2PI it is a spotlight
    {
        spotAngle = cos(lightData.openingAngle - lightData.penumbraAngle * rnd.z);
        woLocal = sample_cone(rnd.xy, spotAngle);
    }
    else // Pointlight
    {
        woLocal = sample_sphere(rnd.xy);
    }

    invLightPdf = M_2PI * (1.f - spotAngle);
    return fromLocalToWorld(lightData.dirW, woLocal); // Used pls.normal before but seemed wrong
}

bool createLightSampleAnalytic(
    inout SampleGenerator sg,
    out PhotonLightSample pls,
    out float3 sampledDir,
    out uint lightIdx)
{
    // Init
    pls = {};
    sampledDir = float3(0);
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0)
    {
        lightIdx = 0;
        return false;
    }

    // Uniformly sample one of the analytic lights
    float rndLightIdx = float(lightCount) * sampleNext1D(sg);
    lightIdx = min(uint(floor(rndLightIdx)), lightCount-1);
    LightData lightData = gScene.getLight(lightIdx);

    // Only Point(+ Spot) lights are supported (due to RTXDI)
    if (lightData.type != uint(LightType::Point))
    {
        return false;
    }

    // Sample direction
    float invLightPdf;

    // Sample direction without photon guiding
#if !USE_3D_GAUSSIAN_PHOTON_GUIDING
    sampledDir = SampleAnalyticLightDirUniform(sg, lightData, invLightPdf);
#endif

    // Sample direction with photon guiding
#if USE_3D_GAUSSIAN_PHOTON_GUIDING
    // Sample dir from GMM
    float gmmPdf = 1.0f;
    const float3 gmmPos = SampleGMM(lightData.posW, lightIdx, sg, gmmPdf);
    sampledDir = normalize(gmmPos - lightData.posW);
    if (gmmPdf <= gGmmMinPdf)
    {
        return false;
    }
    invLightPdf = 1.0f / gmmPdf;

    // Set the flux to 0 if the light is a spotlight and the dir is outside the cone
    if (dot(sampledDir, lightData.dirW) < lightData.cosOpeningAngle)
    {
        pls.flux = float3(0.0f);
    }
#endif

    if (invLightPdf <= 0.f)
    {
        return false;
    }

    invLightPdf *= lightCount; // Take the select pdf into account
    
    // Fill the photon light struct
    pls.posW = lightData.posW;
    pls.normal = lightData.dirW;
    pls.flux = lightData.intensity * invLightPdf;
    
    return true;
}

// True if a photon should be stored, false otherwise
bool cullingTest(float3 origin)
{
    // Check if hash cell is set
    float scaleFactor = gHashScaleFactor;
    int3 cell = int3(floor(origin * scaleFactor));
    uint hashSize = (gHashSize - 1);
    uint h = hash(cell) & hashSize;

    return gPhotonCullingMask[ReSTIR_LinearIndexToZCurve(h)];
}

[shader("miss")]
void miss(inout RayData rayData : SV_RayPayload)
{
    rayData.terminated |= kRayFlagTerminatedMiss;
}

[shader("closesthit")]
void closestHit(
    inout RayData rayData : SV_RayPayload,
    BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    const float3 incomingRayDir = -WorldRayDirection();
    // Evaluate Falcor's material parameters at the hit point.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    let lod = ExplicitLodTextureSampler(0.f);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, incomingRayDir, lod);

    // Adjust shading normals if desired
    uint hints = 0;
    if (gFlags & (uint) GenerationFlags::AdjShadingNormals)
    {
        hints |= (uint) MaterialInstanceHints::AdjustShadingNormal;
    }
    
    BSDFSample bsdfSample;
    let bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
    let bsdfProperties = bsdf.getProperties(sd);
    
    bool valid = bsdf.sample(sd, rayData.sg, bsdfSample, true /*ImportanceSampling*/);

    float3 thpStart = rayData.thp;
    
    bool transmission = bsdfSample.isLobe(LobeType::Transmission);
    float3 rayOrigin = sd.computeNewRayOrigin(!transmission); // false is for transmissions
    rayData.origin = rayOrigin;
    rayData.thp *= valid ? bsdfSample.weight: float3(0);
    rayData.packedNormal = encodeNormal2x8(sd.faceN);   // Packed normal formated as 2x8 for surface rejection 
    rayData.direction = bsdfSample.wo;
    rayData.isEmissive = any(bsdfProperties.emission > 0);
    rayData.distance += RayTCurrent();

    // Check if the hit Material is diffuse
    float3 diffuseProb = bsdfProperties.diffuseReflectionAlbedo / (bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.specularReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo + bsdfProperties.specularTransmissionAlbedo);
    float meanDiffuseProb = (diffuseProb.x + diffuseProb.y + diffuseProb.z) / 3;
    bool hasDiffuseParts = kRequireDiffuseParts ? (meanDiffuseProb > kTraceDiffuseCutoff) || bsdfProperties.roughness > kTraceRoughnessMax : true; // Check if surface is diffuse or above the roughness threshold
    rayData.diffuseHit = hasDiffuseParts && bsdfProperties.roughness > kTraceRoughnessMin && !transmission; // Material is only diffuse if it is over the roughness min
    
    // if throughput is 0, return
    if (!valid || all(rayData.thp <= 0))
    {
        rayData.terminated |= kRayFlagTerminatedHit;
    }
}

// Use anyhit for alpha test if activated
[shader("anyhit")]
void anyHit(
    inout RayData rayData : SV_RayPayload,
    BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    if (gFlags & (uint) GenerationFlags::AlphaTest)
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

bool SamplePhotonDirCosineWeighted(out float3 dir, out float pdf, inout SampleGenerator sg, const float3 normal)
{
    const float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), pdf);
    if (pdf <= 0.f)
    {
        dir = float3(0.0f);
        return false;
    }
    dir = fromLocalToWorld(normal, woLocal);
    pdf = 1.0f / M_PI; // Lambert Emitter
    return true;
}

bool SamplePhotonDirGmm(
    out float3 dir,
    out float pdf,
    inout SampleGenerator sg,
    const uint lightIdx,
    const float3 pos)
{
    const float3 gmmPos = SampleGMM(pos, lightIdx, sg, pdf);
    dir = normalize(gmmPos - pos);
    if (pdf <= gGmmMinPdf)
    {
        return false;
    }
    return true;
}

bool GeneratePhotonAnalytic(
    inout SampleGenerator lightSg,
    inout PhotonLightSample pls,
    out uint lightIdx,
    inout float firstHitSamplingPdf,
    inout RayDesc ray)
{
    // Analytic light sample
    if (!createLightSampleAnalytic(lightSg, pls, ray.Direction, lightIdx))
    {
        return false;
    }
    return true;
}

#if USE_EMISSIVE_LIGHT // Function is only valid if the emissive sampler is set
bool GeneratePhotonGeometric(
    inout SampleGenerator lightSg,
    inout PhotonLightSample pls,
    out uint lightIdx,
    inout float firstHitSamplingPdf,
    inout RayDesc ray)
{
    // Sample emissive light
    if (!createLightSampleGeometric(lightSg, pls, lightIdx))
    {
        return false;
    }

    // Geometric lights come after analytic lights in gaussian buffer
    lightIdx += gAnalyticLightCount;

    // Sample direction without photon guiding
#if USE_3D_GAUSSIAN_PHOTON_GUIDING
    const float sampleGmmProb = gBeta;
#else  // USE_3D_GAUSSIAN_PHOTON_GUIDING
    const float sampleGmmProb = 0.0f;
#endif // USE_3D_GAUSSIAN_PHOTON_GUIDING

    if (sampleNext1D(lightSg) >= sampleGmmProb)
    {
        // Sample cosine weighted
        if (!SamplePhotonDirCosineWeighted(ray.Direction, firstHitSamplingPdf, lightSg, pls.normal))
        {
            return false;
        }

        // MIS weight
        const float gmmProb = GMMProb(pls.posW, pls.posW + ray.Direction, lightIdx);
        pls.flux *= firstHitSamplingPdf / (firstHitSamplingPdf + gmmProb);
    }
    else
    {
        // Sample gmm
        if (!SamplePhotonDirGmm(ray.Direction, firstHitSamplingPdf, lightSg, lightIdx, pls.posW))
        {
            return false;
        }
        pls.flux *= dot(pls.normal, ray.Direction);

        // MIS weight
        const float cosineWeightedPdf = dot(pls.normal, ray.Direction);
        pls.flux *= firstHitSamplingPdf / (firstHitSamplingPdf + cosineWeightedPdf);
    }

    pls.flux /= firstHitSamplingPdf;
    return true;
}
#endif // USE_EMISSIVE_LIGHT

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    const float invRejection = 1.f / gRejection;
    const float invInvRejection = 1.f / (1.f - gRejection);

    // Sample Generator for the light sample
    SampleGenerator lightSg = SampleGenerator(launchIndex, gFrameCount + 11 * 32);

    RayDesc ray;
    PhotonLightSample pls;
    uint lightIdx = 0;
    float firstHitSamplingPdf = 0.0f;

#if USE_EMISSIVE_LIGHT
    if (gFlags & (uint)GenerationFlags::AnalyticLight)
#endif  // USE_EMISSIVE_LIGHT
    {
        if (!GeneratePhotonAnalytic(lightSg, pls, lightIdx, firstHitSamplingPdf, ray))
        {
            return;
        }
    }
#if USE_EMISSIVE_LIGHT  
    else
    {
        if (!GeneratePhotonGeometric(lightSg, pls, lightIdx, firstHitSamplingPdf, ray))
        {
            return;
        }
    }
#endif // USE_EMISSIVE_LIGHT

    // Division through all dipatched photons
    pls.flux /= launchDim.x * launchDim.y;

    // Prepare ray payload
    RayData rayData = RayData.create();
    rayData.sg = SampleGenerator(launchIndex, gFrameCount + 1 * 32 );

    ray.Origin = pls.posW;
    ray.TMin = 0.0f;
    ray.TMax = 10000.f;
    
    // Create photon
    float3 photonPos = float3(0);
    PhotonInfo photon;
    photon.dir = float3(0);
    photon.faceNTheta = 1.f;
    photon.flux = float3(0);
    photon.faceNPhi = 1.f;
    
    uint rayFlags = 0;

    const uint maxCausticBounces = gCausticsBounces >= 0 ? gCausticsBounces : gMaxRecursion;
    uint currentCausticBounces = 0;
    bool storeAsCaustic = false;
    bool reflectedDiffuse = false;

    uint firstHitIndex = 0;

    // Recursion where the photons are shot
    for (uint i = 0; i < gMaxRecursion && !rayData.terminated; i++)
    {
        photon.flux = pls.flux * rayData.thp;
        TraceRay(
            gScene.rtAccel,
            rayFlags,
            0xff, // instanceInclusionMask
            0, // hitIdx
            rayTypeCount,
            0, // missIdx
            ray,
            rayData);

        // Stop if a light source was hit or the ray missed
        if (rayData.isEmissive || (rayData.terminated & kRayFlagTerminatedMiss > 0))
        {
            break;
        }

        // Store photon info
        photonPos = rayData.origin;
        photon.dir = ray.Direction;
        reflectedDiffuse = rayData.diffuseHit;

        // Store first hit photon even when the first hit is not stored in the photon map
        // Because resulting photons along the path might be collected
#ifdef USE_3D_GAUSSIAN_PHOTON_GUIDING
        if (i == 0)
        {
            // Get first hit photon index
            InterlockedAdd(gFirstHitPhotonCounter[0], 1u, firstHitIndex);

            // Check if buffer is full
            if (firstHitIndex < gMaxFirstHitPhotonCount)
            {
                // Track light fist hit count
                InterlockedAdd(gLightFirstHitCounts[lightIdx], 1u);

                // Store first hit photon position
                FirstHitPhotonInfo firstHitPhotonInfo;
                firstHitPhotonInfo.pos = photonPos;
                firstHitPhotonInfo.samplingPdf = firstHitSamplingPdf;
                firstHitPhotonInfo.lightIdx = lightIdx;
                gFirstHitPhotonInfo[firstHitIndex] = firstHitPhotonInfo;
            }
        }
#endif // USE_3D_GAUSSIAN_PHOTON_GUIDING

        // Gen Roulette value for rejecting
        float rndRoulette = sampleNext1D(rayData.sg);
        bool roulette = rndRoulette <= gRejection;
        bool minDistanceRoulette = true;

        // Check if photon should be rejected if it is too close to the light source
        if (rayData.distance < gGenerationLampIntersectGuard)
        {
            minDistanceRoulette = sampleNext1D(rayData.sg) < gGenerationLampIntersectGuardStoreProbability;
            photon.flux *= 1.f / gGenerationLampIntersectGuardStoreProbability;
        }

        // Store photons
        if (reflectedDiffuse && (roulette || storeAsCaustic) && minDistanceRoulette)
        {
            // Global = 0 or caustic = 1
            uint photonType = storeAsCaustic ? 1 : 0;
            bool storePhoton = true;

            // Photon Culling
            if (kUsePhotonCulling && (kUseCausticCulling || photonType == 0))
            {
                storePhoton = cullingTest(photonPos);
            }

            // Store
            if (storePhoton)
            {
                uint photonIndex = 0;
                if (photonType == 0)
                {
                    photon.flux *= invRejection;
                }

                InterlockedAdd(gPhotonCounter[photonType], 1u, photonIndex);

                // Only store photon if the buffer space allows it
                if (photonIndex <= kPhotonBufferSize[photonType])
                {
                    // Store photon
                    AABB photonAABB = AABB(photonPos - gPhotonRadius[photonType], photonPos + gPhotonRadius[photonType]);
                    gPhotonAABB[photonType][photonIndex] = photonAABB;
                    PhotonData pd = PhotonData();
                    pd.flux = photon.flux;
                    pd.dir = -photon.dir;
                    pd.packedNormal = rayData.packedNormal;
                    pd.isFirstHit = i == 0;
                
                    gPackedPhotonData[photonType][photonIndex] = pd.pack();

                    // Store mapping from photon index to first hit index
                    gPhotonFirstHitMap[photonType][photonIndex] = firstHitIndex;
                }
            }
        }

        // If next sampled ray is invalid return here
        if (rayData.terminated > 0)
        {
            break;
        }
        
        // Russian Roulette
        const float rrVal = luminance(rayData.thp);
        const float prob = max(0.f, 1.f - rrVal);
        float rnd = sampleNext1D(rayData.sg);
        if (rnd < prob)
        {
            break; // Photon is absorbed
        }
        else
        {
            rayData.thp /= (1.f - prob);
        }

        // Caustics handling
        if (gFlags & (uint) GenerationFlags::EnableCaustics)
        {
            storeAsCaustic = !reflectedDiffuse;
            if (currentCausticBounces > maxCausticBounces)
            {
                storeAsCaustic = false;
            }
            if (reflectedDiffuse)
            {
                currentCausticBounces++;
            }
        }
                        
        // Update ray for next iteration
        ray.Origin = rayData.origin;
        ray.Direction = rayData.direction;
    }
}
