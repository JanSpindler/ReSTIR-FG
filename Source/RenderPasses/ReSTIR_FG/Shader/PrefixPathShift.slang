#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import LoadShadingData;
import PrefixPathReservoir;
import Utils.Sampling.SampleGenerator;
import Scene.Scene;
import Scene.HitInfo;
import Scene.Shading;
import Rendering.Materials.TexLODTypes;
import Scene.RaytracingInline;
import Utils.Math.Ray;

#if !SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
#error "This file requires SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH) to be true."
#endif

static const float kDemodulateMinReflectance = 0.01f;
static const float kRoughnessMin = TRACE_TRANS_SPEC_ROUGH_CUTOFF_MIN;
static const float kRoughnessMax = TRACE_TRANS_SPEC_ROUGH_CUTOFF_MAX;
static const float kDiffuseCutoff = TRACE_TRANS_SPEC_DIFFUSEPART_CUTOFF;
static const bool kAlphaTest = USE_ALPHA_TEST;

bool ShiftPrefixPath(
    const in PrefixPath currentPath,
    const in ShadingData currentPrimarySd,
    const in HitInfo currenHit,
    const in PrefixPath otherPath,
    const float3 camPos,
    const bool requDiffParts,
    out PrefixPath shiftedPath,
    out float jacobian)
{
    // Validate input path length
    if (otherPath.GetLength() != currentPath.GetLength())
    {
        shiftedPath.throughput = float3(0.0f);
        jacobian = 0.0f;
        return false;
    }

    // Init shifted path
    shiftedPath.throughput = float3(1.0f);

    shiftedPath.fgRayDist = 0.0f;
    shiftedPath.fgHitT = 0.0f;
    
    shiftedPath.flags = 0u;
    shiftedPath.samplingPdf = 1.0f;
    shiftedPath.seed = otherPath.seed;

    // Tracing
    SampleGenerator sg;
    sg.rng.state = otherPath.seed;
    let lod = ExplicitLodTextureSampler(0.f);

    // FG
    HitInfo hitInfo = currenHit;
    float3 viewDir = currentPrimarySd.V;
    ShadingData sd = currentPrimarySd;

    // Caustic
#if CAUSTIC_TEMPORAL_FILTER_ENABLED
    CausticSurface causticSurface;
#endif

    // Trace path
    bool validTracing = true;
    for (uint vertexIdx = 0; vertexIdx < otherPath.GetLength(); ++vertexIdx)
    {
        if (vertexIdx != 0)
        {
            sd = LoadShadingData(hitInfo, -viewDir, lod);
        }
        else
        {
            shiftedPath.fgRayDist = distance(camPos, sd.posW);
        }

        // Get BSDF
        const uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
        const let bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
        const let bsdfProperties = bsdf.getProperties(sd);

#if CAUSTIC_TEMPORAL_FILTER_ENABLED
        if (i == 0)
        {
            causticSurface.normal = sd.faceN;
            causticSurface.viewDepth = rayDist;
        }
#endif

        // Sample BSDF
        BSDFSample bsdfSample;
        const bool validBSDF = bsdf.sample(sd, sg, bsdfSample);
        if (!validBSDF)
        {
            validTracing = false;
            break;
        }

        // Determine if surface is diffuse
        const bool isTransmission = validBSDF ? bsdfSample.isLobe(LobeType::Transmission) : false;
        const float3 diffuseProb = bsdfProperties.diffuseReflectionAlbedo /
            max(bsdfProperties.diffuseReflectionAlbedo +
                bsdfProperties.specularReflectionAlbedo +
                bsdfProperties.diffuseTransmissionAlbedo +
                bsdfProperties.specularTransmissionAlbedo, float3(1e-8f));
        const float meanDiffuseProb = (diffuseProb.x + diffuseProb.y + diffuseProb.z) / 3.0f;
        const bool hasDiffuseParts = requDiffParts ?
            (meanDiffuseProb > kDiffuseCutoff) || bsdfProperties.roughness > kRoughnessMax :
            true;
        const bool isDiffuse = hasDiffuseParts && bsdfProperties.roughness > kRoughnessMin && !isTransmission;

        // Track information for shift
        shiftedPath.IncLength();
        if (bsdfSample.isLobe(LobeType::Delta))
        {
            shiftedPath.SetDeltaEvent(true);
        }
        else
        {
            shiftedPath.samplingPdf *= bsdfSample.pdf;
        }

        // Exit if the surface is diffuse
        if (isDiffuse)
        {
            break;
        }

        // Update throughput
        shiftedPath.throughput *= bsdfSample.weight;

        // Trace next ray, because surface was not diffuse
        Ray ray;
        ray.origin = sd.computeNewRayOrigin(!isTransmission);
        ray.tMin = 0.0001f;
        ray.tMax = 10000.f;
        ray.dir = bsdfSample.wo;

        float hitT;
        SceneRayQuery<kAlphaTest> sceneRayQuery;
        const bool traceSuccess = sceneRayQuery.traceRay(ray, hitInfo, hitT, RAY_FLAG_NONE, 0xFF);
        if (!traceSuccess)
        {
            validTracing = false;
            break;
        }

        // Set is delta flag on caustic surface
#if CAUSTIC_TEMPORAL_FILTER_ENABLED
        if (i == 0)
        {
            causticSurface.isDelta = true;
        }
#endif

        // Update ray parameters for next iteration
        viewDir = -bsdfSample.wo;
        shiftedPath.fgRayDist += hitT;
        shiftedPath.fgHitT += hitT;
    }

    // Factor out the pdf from the throughput
    shiftedPath.throughput *= shiftedPath.samplingPdf;

    // Store caustic surface if enabled
#if CAUSTIC_TEMPORAL_FILTER_ENABLED
    shiftedPath.causticSurface = causticSurface.pack();
#endif

    // Check if the shifted path is actually shiftable (delta events, etc.)
    if (shiftedPath.GetLength() != currentPath.GetLength() ||
        shiftedPath.HasDeltaEvent() != currentPath.HasDeltaEvent() ||
        !validTracing ||
        shiftedPath.samplingPdf == 0.0f)
    {
        shiftedPath.throughput = float3(0.0f);
        jacobian = 0.0f;
        return false;
    }

    shiftedPath.fgHitInfo = hitInfo.getData();
    shiftedPath.fgViewDir = float4(viewDir, 0.0f);
    jacobian = otherPath.samplingPdf / shiftedPath.samplingPdf;
    return jacobian > 0.0f && !isnan(jacobian) && !isinf(jacobian);
}
