#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import LoadShadingData;
import PrefixPathReservoir;
import Utils.Sampling.SampleGenerator;
import Scene.Scene;
import Scene.HitInfo;
import Scene.Shading;
import Rendering.Materials.TexLODTypes;

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
    out float3 shiftedThroughput,
    out float jacobian)
{
    // Trace shifted path using random replay
    uint newPathLength = 0;
    uint newPathDeltaFlags = 0;
    float rayDist = 0.0f;

    SampleGenerator sg = SampleGenerator(otherPath.seed);
    let lod = ExplicitLodTextureSampler(0.f);

    HitInfo hitInfo = currenHit;
    float3 viewDir = currentPrimarySd.V;
    ShadingData sd = currentPrimarySd;

    for (uint vertexIdx = 0; vertexIdx < otherPath.length; ++vertexIdx)
    {
        if (vertexIdx != 0)
        {
            sd = LoadShadingData(hitInfo, -viewDir, lod);
        }
        else
        {
            rayDist = distance(camPos, sd.posW);
        }

        // Get BSDF
        const uint hints = (uint)MaterialInstanceHints::AdjustShadingNormal;
        const let bsdf = gScene.materials.getMaterialInstance(sd, lod, hints);
        const let bsdfProperties = bsdf.getProperties(sd);

        // Sample BSDF
        BSDFSample bsdfSample;
        const bool validBSDF = bsdf.sample(sd, sg, bsdfSample);

        // Determine if surface is diffuse
        const bool isTransmission = validBSDF ? bsdfSample.isLobe(LobeType::Transmission) : false;
        const float3 diffuseProb = bsdfProperties.diffuseReflectionAlbedo /
            (bsdfProperties.diffuseReflectionAlbedo +
            bsdfProperties.specularReflectionAlbedo +
             bsdfProperties.diffuseTransmissionAlbedo +
            bsdfProperties.specularTransmissionAlbedo);
        const float meanDiffuseProb = (diffuseProb.x + diffuseProb.y + diffuseProb.z) / 3.0f;
        const bool hasDiffuseParts = requDiffParts ?
            (meanDiffuseProb > kDiffuseCutoff) || bsdfProperties.roughness > kRoughnessMax :
            true; // Check if surface is diffuse or above the roughness threshold
        const bool isDiffuse = hasDiffuseParts && bsdfProperties.roughness > kRoughnessMin && !isTransmission; // Material is only diffuse if it is over the roughness min
    }

    // Check if the shifted path is actually shiftable (delta events, etc.)
    if (newPathLength != currentPath.length || newPathDeltaFlags != currentPath.deltaFlags)
    {
        shiftedThroughput = float3(0.0f);
        jacobian = 0.0f;
        return false;
    }

    shiftedThroughput = float3(0.0f);
    jacobian = 1.0f;
    return true;
}
