/***************************************************************************
 # Copyright (c) 2022, Daqi Lin.  All rights reserved.
 **************************************************************************/
//#include "Scene/Material/MaterialDefines.slangh"

import Scene.RaytracingInline;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
//import Rendering.Materials.StandardMaterial;
import Rendering.Materials.ClothMaterial;
import Rendering.Materials.HairMaterial;
//import Rendering.Materials.MaterialShading;
import Rendering.Materials.Microfacet;
import Rendering.Materials.InteriorListHelpers;
import Rendering.Volumes.HomogeneousVolumeSampler;
import Rendering.Volumes.PhaseFunction;
import Rendering.Utils.PixelStats;
//import RenderPasses.Shared.Denoising.NRDBuffers;
//import RenderPasses.Shared.Denoising.NRDData;
//import RenderPasses.Shared.Denoising.NRDHelpers;
//import RenderPasses.Shared.Denoising.NRDConstants;
//import NRDHelpers;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Debug.PixelDebug;
import LoadShadingData;
import PathBuilder;
import PathReservoir;

#include "Utils/Math/MathConstants.slangh"

__exported import PathState;
__exported import Params;

uint getAllowedBSDFFlags(bool isSpecularBounce)
{
    if (kSeparatePathBSDF) return isSpecularBounce ? 0xc : 0x3;
    else return -1;
}

struct PathTracer
{
    RestirPathTracerParams params;                  ///< Runtime parameters.
    EnvMapSampler envMapSampler;                    ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler;           ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.

    Texture2D<PackedHitInfo> vbuffer;               ///< Fullscreen V-buffer for the primary hits.

    //NRDBuffers outputNRD;                           ///< Output NRD data.
    //RWTexture2D<float4> outputNRDDiffuseRadianceHitDist;    ///< Output resolved diffuse color in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
    //RWTexture2D<float4> outputNRDSpecularRadianceHitDist;   ///< Output resolved specular color in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.
    //RWTexture2D<float4> outputNRDResidualRadianceHitDist;   ///< Output resolved residual color in .rgb and hit distance in .a for NRD. Only valid if kOutputNRDData == true.

    RWTexture2D<float4> outputColor;                ///< Output color buffer if kSamplesPerPixel == 1.
    RWTexture2D<float4> outputDebug;                ///< Output debug data. Valid if OUTPUT_DEBUG is nonzero.
    RWTexture2D<uint> outputTime;                   ///< Output timing data. Valid if OUTPUT_TIME is nonzero.

    Texture2D<float4> directLighting;                   ///< Output offset into per-sample buffers. Only valid when kSamplesPerPixel == 0.

    RWStructuredBuffer<PathReservoir> outputReservoirs;            ///< Output paths from the path tracing pass.
    bool isLastRound;
    bool useDirectLighting;
    int  gSppId;
    int  gNumSpatialRounds;

    /*******************************************************************
                                Static members
    *******************************************************************/

    // Render settings that depend on the scene.
    // TODO: Move into scene defines.
    //static bool kUseEnvLight = USE_ENV_LIGHT;
    //static bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
    //static bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
    bool kUseEnvLight;
    bool kUseEmissiveLights;
    bool kUseAnalyticLights;

    // Additional specialization.
    static const bool kOutputTime = OUTPUT_TIME;

    /** Types of samplable lights.
    */
    enum class LightSampleType
    {
        EnvMap, //0
        Emissive, //1
        Analytic //2
    };

    /** Describes a light sample.
    */
    struct LightSample
    {
        float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
        float   pdf;        ///< Pdf with respect to solid angle at the shading point.
        float3  origin;     ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
        float   distance;   ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
        float3  dir;        ///< Ray direction for visibility evaluation (normalized).
        uint    lightType;  ///< Light type this sample comes from (LightSampleType casted to uint).

        Ray getVisibilityRay() { return Ray(origin, dir, 0.f, distance); }
    };

    /** Describes a path vertex.
    */
    struct PathVertex
    {
        uint index;         ///< Vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
        float3 pos;         ///< Vertex position.
        float3 normal;      ///< Shading normal at the vertex (zero if not on a surface).
        float3 faceNormal;  ///< Geometry normal at the vertex (zero if not on a surface).

        /** Create a path vertex.
            \param[in] index Vertex index.
            \param[in] pos Vertex position.
            \param[in] normal Shading normal.
            \param[in] faceNormal Geometry normal.
            \return Returns a path vertex.
        */
        static PathVertex create(uint index, float3 pos, float3 normal = float3(0.f), float3 faceNormal = float3(0.f))
        {
            PathVertex v = {};
            v.index = index;
            v.pos = pos;
            v.normal = normal;
            v.faceNormal = faceNormal;
            return v;
        }

        /** Get position with offset applied in direction of the geometry normal to avoid self-intersection
            for visibility rays.
            \param[in] rayDir Direction of the visibility ray (does not need to be normalized).
            \return Returns the offseted position.
        */
        float3 getRayOrigin(float3 rayDir)
        {
            return computeRayOrigin(pos, dot(faceNormal, rayDir) >= 0 ? faceNormal : -faceNormal);
        }
    };

    /*******************************************************************
                              Member functions
    *******************************************************************/

    /** Check if the path has finished all surface bounces and needs to be terminated.
        Note: This is expected to be called after generateScatterRay(), which increments the bounce counters.
        \param[in] path Path state.
        \return Returns true if path has processed all bounces.
    */
    bool hasFinishedSurfaceBounces(const PathState path)
    {
        const uint diffuseBounces = path.getBounces(BounceType::Diffuse);
        const uint specularBounces = path.getBounces(BounceType::Specular);
        const uint transmissionBounces = path.getBounces(BounceType::Transmission);
        const uint surfaceBounces = diffuseBounces + specularBounces + transmissionBounces;
        return
            (surfaceBounces > kMaxSurfaceBounces);// ||
            //(diffuseBounces > kMaxDiffuseBounces) ||
            //(specularBounces > kMaxSpecularBounces) ||
            //(transmissionBounces > kMaxTransmissionBounces);
    }

    /** Compute the total length of a terminated path.
        \param[in] path Path state.
        \return Returns the total number of bounces a path took.
    */
    uint getTerminatedPathLength(const PathState path)
    {
        // Account for the fact that we may have counted one bounce too many (scatter ray at the last path vertex).
        uint diffuseBounces = min(kMaxDiffuseBounces, path.getBounces(BounceType::Diffuse));
        uint specularBounces = min(kMaxSpecularBounces, path.getBounces(BounceType::Specular));
        uint transmissionBounces = min(kMaxTransmissionBounces, path.getBounces(BounceType::Transmission));
        uint surfaceBounces = min(kMaxSurfaceBounces, diffuseBounces + specularBounces + transmissionBounces);
        return surfaceBounces;
    }

    /** Generate the path state for a primary hit in screen space.
        This is treated equivalent to subsequent path vertices to reduce code divergence.
        \param[in] pathID Path ID which encodes pixel and sample index.
        \param[out] path Path state for the primary hit.
    */
    void generatePath(const uint pathID, out PathState path)
    {
        path = {};
        path.setActive();
        path.id = pathID;
        path.thp = float3(1.f);
        path.prefixThp = float3(1.f);
        path.rcVertexPathTreeIrradiance = float3(0.f);

        path.LDeltaDirect = float3(0.f);
        path.enableRandomReplay = false;
        path.useHybridShift = ShiftMapping(kShiftStrategy) == ShiftMapping::Hybrid;
        path.isLastVertexClassifiedAsRough = true;
        path.isReplayForHybridShift = false;

        // Load the primary hit info from the V-buffer.
        // If invalid, the path is still active and treated as a miss.
        uint2 pixel = path.getPixel();
        const HitInfo hit = HitInfo(vbuffer[pixel]);

        if (hit.isValid())
        {
            path.setHit(hit);
            //path.setVertexIndex(1);
        }

        // TODO: Support depth-of-field using computeRayThinlens, if aperture size > 0.
        Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        path.origin = cameraRay.origin;
        path.dir = cameraRay.dir;

        // Create sample generator.
        static const uint itersPerShaderPass = PathSamplingMode(kPathSamplingMode) == PathSamplingMode::PathTracing ? kSamplesPerPixel : kCandidateSamples;

        uint maxSpp = itersPerShaderPass > 0 ? itersPerShaderPass : kMaxSamplesPerPixel;
        path.sg = SampleGenerator(pixel, (itersPerShaderPass + 1 + gNumSpatialRounds) * params.seed + path.getSampleIdx());

        if (kOutputNRDData)
        {
            path.hitDist = 0;
            path.rcPrevVertexWo = 0;
            path.sharedScatterDir = 0;
        }

        path.pathBuilder.init(path.sg);
        path.pathReservoir.init();
    }

    // generate random replay path
    void generateRandomReplayPath(out PathState path, HitInfo hit, Ray ray, uint randomSeed, int randomReplayLength, bool isLastVertexNEE, bool isRcVertexEscapedVertex)
    {
        path = {};
        path.L = 0.f;
        path.setActive();
        path.thp = float3(1.f);
        path.prefixThp = float3(1.f);
        path.rcVertexPathTreeIrradiance = float3(0.f);

        path.enableRandomReplay = true;
        path.useHybridShift = ShiftMapping(kShiftStrategy) == ShiftMapping::Hybrid;
        path.setHit(hit);

        path.origin = ray.origin;
        path.dir = ray.dir;

        path.sg = SampleGenerator(randomSeed); 

        path.randomReplayIsNEE = isLastVertexNEE && !isRcVertexEscapedVertex;
        path.randomReplayIsEscaped = !isLastVertexNEE && !isRcVertexEscapedVertex;
        path.randomReplayLength = randomReplayLength;

        path.pathReservoir.init();
    }


    /** Set up path for logging and debugging.
        \param[in] path Path state.
    */
    void setupPathLogging(const PathState path)
    {
        printSetPixel(path.getPixel());
        logSetPixel(path.getPixel());
    }

    /** Update the path throughouput
        \param[in,out] path Path state.
        \param[in] weight Vertex throughput.
    */
    void updatePathThroughput(inout PathState path, const float3 weight)
    {
        path.thp *= weight;
    }

    void invalidateAndTerminateReplayPath(inout PathState path)
    {
        path.rcPrevVertexHit = HitInfo();
        path.rcPrevVertexWo = float3(0);
        path.L = 0.f;
        path.thp = 0.f;
        path.terminate();
    }

    void setDataBeforeTermination(inout HitInfo rcPrevHit, inout float3 rcPrevHitWo)
    {
        rcPrevHit = HitInfo();
        rcPrevHitWo = float3(0);
    }

    /** Generates a new scatter ray using BSDF importance sampling.
        \param[in] sd Shading data.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const ShadingData sd, inout PathState path)
    {
        bool isCurveHit = path.hit.getType() == HitType::Curve;

        // Generate a BSDF sample.
        bool valid;
        BSDFSample result;
        let lod = ExplicitLodTextureSampler(0.f);

        // if (kUseBSDFSampling) valid = sampleBSDF(sd, path.sg, result, !kSeparatePathBSDF); // daqi: indicate using deterministic BSDF
        // else valid = sampleBSDF_Reference(sd, path.sg, result);
        let bsdf = gScene.materials.getMaterialInstance(sd, lod, 0); // TODO: If adjust shading normals add flag to hints
        let bsdfProperties = bsdf.getProperties(sd);
        valid = bsdf.sample(sd, path.sg, result, kUseBSDFSampling);

        float pdfAll = result.pdf;
        //if (kSeparatePathBSDF) result.pdf = result.pdf; // No pdfSingle in this falcor version

        // Ignore valid on purpose for now.
        if (kOutputNRDData && path.length == 0)
        {
            path.setDiffusePrimaryHit(result.isLobe(LobeType::Diffuse));
            path.setSpecularPrimaryHit(result.isLobe(LobeType::Specular));
        }

        if (valid)
        {
            // store sampled path type (diffuse or non-diffuse)

            path.dir = -sd.V;//result.wi;

            // add to path buffer
            // also multiply pdf

            path.thp *= result.weight;

            // this is pdfAll
            path.pdf = pdfAll;

            // daqi: save the Jacobian for rcVertex
            if (path.length == path.pathBuilder.rcVertexLength)
            {
                // TODO: use component PDF
                path.pathBuilder.cachedJacobian.y = result.pdf;
            }

            path.prevScatterPdf = result.pdf;

            if (result.pdf <= 0.f) // invalid
            {
                return false;
            }

#if _USE_LEGACY_SHADING_CODE
            // Legacy BRDF code sometimes returns infinite PDF values.
            // We clamp them to avoid getting NaNs from the evalMIS function.
            path.pdf = min(path.pdf, 1e30f);
#endif

            path.clearEventFlags();

            // Handle reflection events.

            if (result.isLobe(LobeType::Reflection))
            {
                // We classify specular events as diffuse if the roughness is above some threshold.
                bool isDiffuse = result.isLobe(LobeType::DiffuseReflection) || bsdfProperties.roughness > params.specularRoughnessThreshold;
                if (isDiffuse)
                {
                    path.incrementBounces(BounceType::Diffuse);
                }
                else
                {
                    path.incrementBounces(BounceType::Specular);
                    path.setSpecular();
                }
            }

            // Handle delta events.
            if (result.isLobe(LobeType::Delta))
            {
                path.setDelta();
            }

            if (result.isLobe(LobeType::Specular) || result.isLobe(LobeType::Delta))
            {
                path.setSpecularBounce();
            }

            // daqi: record delta and transmission event before/after rcVertex, this is important because such information will be used in computing the Jacobian in shift.slang
            if (path.length == path.pathBuilder.rcVertexLength)
            {
                path.pathBuilder.pathFlags.insertIsDeltaEvent(result.isLobe(LobeType::Delta), false);
                path.pathBuilder.pathFlags.insertIsTransmissionEvent(result.isLobe(LobeType::Transmission), false);
                if (kSeparatePathBSDF)
                    path.pathBuilder.pathFlags.insertIsSpecularBounce(path.isSpecularBounce(), false);
            }

            // Handle transmission events.
            if (result.isLobe(LobeType::Transmission))
            {
                path.incrementBounces(BounceType::Transmission);
                path.setTransmission();

                // Compute ray origin for next ray segment.
                path.origin = sd.computeNewRayOrigin(false);

                // Update interior list and inside volume flag.
                if (!sd.mtl.isThinSurface())
                {
                    uint nestedPriority = sd.mtl.getNestedPriority();//gScene.materials[sd.materialID].getNestedPriority();
                    path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                    path.setInsideDielectricVolume(!path.interiorList.isEmpty());
                }
            }

            // Save the shading normal. This is needed for MIS.
            path.normal = sd.frame.N;

            // Mark the path as valid only if it has a non-zero throughput.
            valid = any(path.thp > 0.f);
            // Add to the path gaph.
            // TODO: Handle specular/delta/transmission events.
            if (valid)
            {
                if (path.length <= path.pathBuilder.rcVertexLength)
                    path.recordPrefixThp();
            }
        }
        else
        {
            path.thp = 0.f;
        }

        return valid;
    }

    // TODO: support BySampledBSDF replay
    bool generateScatterRayGivenDirection(const ShadingData sd, float3 dir, inout PathState path)
    {
        bool isCurveHit = path.hit.getType() == HitType::Curve;

        // Generate a BSDF sample.
        bool valid = true;
        BSDFSample result;
        let bsdf = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0.f), 0);
        let bsdfProperties = bsdf.getProperties(sd);

        bool isDeltaEvent = path.pathBuilder.pathFlags.decodeIsDeltaEvent(path.length < 1);
        bool isSpecularBounce = path.pathBuilder.pathFlags.decodeIsSpecularBounce(path.length < 1);

        if (isDeltaEvent) valid = false; // not allowing delta event to be shifted by directional reuse

        if (valid)
        {
            float pdfAll = 0.f; // pdf with all BSDF

            {
                //result.wi = dir;

                uint allowedBSDFFlags = getAllowedBSDFFlags(isSpecularBounce);

                // TODO: Correct cosine implementation?
                result.weight = bsdf.evalPdf(sd, dir, false);//evalBSDFCosine(sd, dir, allowedBSDFFlags);

                if (kUseBSDFSampling)
                {
                    // this is single pdf
                    result.pdf = bsdf.evalPdf(sd, dir, true);//evalPdfBSDF(sd, dir, pdfAll, allowedBSDFFlags, false);
                }
                else
                {
                    result.pdf = dot(sd.frame.N, dir) / M_PI;
                }

                result.weight /= result.pdf;

                if (result.pdf <= 0.f)
                {
                    result.weight = 0.f;
                }
            }

            path.dir = dir;

            // add to path buffer
            // also multiply pdf
            path.thp *= result.weight;

            path.pdf = pdfAll;

            if (path.length == path.pathBuilder.rcVertexLength)
            {
                path.pathBuilder.cachedJacobian.y = result.pdf;
            }

            // daqi: record the scatter pdf, we will need to use that later for the cachedJacobian
            path.prevScatterPdf = result.pdf;

#if _USE_LEGACY_SHADING_CODE
            // Legacy BRDF code sometimes returns infinite PDF values.
            // We clamp them to avoid getting NaNs from the evalMIS function.
            path.pdf = min(path.pdf, 1e30f);
#endif

            path.clearEventFlags();

            // daqi: if we use direction reuse, then base and offset path reflection/transmission type is allowed to be different
            bool isTransmissionEvent = dot(sd.frame.N, dir) < 0;

            // daqi: set delta/transmission path flags. we don't care about other path flags since our resampling doesn't depend on the sampled BSDF components
            // Handle transmission events.
            if (isTransmissionEvent)
            {
                // Compute ray origin for next ray segment.
                path.origin = sd.computeNewRayOrigin(false);
                path.setTransmission();

                // Update interior list and inside volume flag.
                if (!sd.mtl.isThinSurface())
                {
                    uint nestedPriority = sd.mtl.getNestedPriority();//gScene.materials[sd.materialID].getNestedPriority();
                    path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                    path.setInsideDielectricVolume(!path.interiorList.isEmpty());
                }
            }

            if (isSpecularBounce)
            {
                path.setSpecularBounce();
            }

            // Save the shading normal. This is needed for MIS.
            path.normal = sd.frame.N;

            // Mark the path as valid only if it has a non-zero throughput.
            valid = any(path.thp > 0.f);

            // Add to the path gaph.
            // TODO: Handle specular/delta/transmission events.
            if (valid)
            {
                if (path.length <= path.pathBuilder.rcVertexLength)
                    path.recordPrefixThp();
            }
        }

        return valid;
    }



    /** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
        \param[in] n0 Number of samples taken from the first sampling strategy.
        \param[in] p0 Pdf for the first sampling strategy.
        \param[in] n1 Number of samples taken from the second sampling strategy.
        \param[in] p1 Pdf for the second sampling strategy.
        \return Weight for the contribution from the first strategy (p0).
    */
    static float evalMIS(float n0, float p0, float n1, float p1)
    {
        switch (MISHeuristic(kMISHeuristic))
        {
        case MISHeuristic::Balance:
        {
            // Balance heuristic
            float q0 = n0 * p0;
            float q1 = n1 * p1;
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerTwo:
        {
            // Power two heuristic
            float q0 = (n0 * p0) * (n0 * p0);
            float q1 = (n1 * p1) * (n1 * p1);
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerExp:
        {
            // Power exp heuristic
            float q0 = pow(n0 * p0, kMISPowerExponent);
            float q1 = pow(n1 * p1, kMISPowerExponent);
            return q0 / (q0 + q1);
        }
        default:
            return 0.f;
        }
    }

    /** Generates a light sample on the environment map.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEnvMapSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        if (!kUseEnvLight) return false;

        // Sample environment map.
        EnvMapSample lightSample;
        if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

        // Setup returned sample.
        ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
        ls.pdf = lightSample.pdf;
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        ls.distance = kRayTMax;
        ls.dir = lightSample.dir;

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the emissive geometry.
        \param[in] vertex Path vertex.
        \param[in] upperHemisphere True if only upper hemisphere (w.r.t. shading normal) should be considered.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEmissiveSample(const PathVertex vertex, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.
        if (!kUseEmissiveLights) return false;

        TriangleLightSample tls;

        // TODO: when random number replay, this can fails with light BVH that previously returns valid samples, why?
        if (!emissiveSampler.sampleLight(vertex.pos, vertex.normal, upperHemisphere, sg, tls)) return false;

        // Setup returned sample.
        ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
        ls.pdf = tls.pdf;
        // Offset shading and light position to avoid self-intersection.
        float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
        ls.origin = vertex.getRayOrigin(lightPos - vertex.pos);
        float3 toLight = lightPos - ls.origin;
        ls.distance = length(toLight);
        ls.dir = normalize(toLight);

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the analytic lights.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateAnalyticLightSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        uint lightCount = gScene.getLightCount();
        if (!kUseAnalyticLights || lightCount == 0) return false;

        // Sample analytic light source selected uniformly from the light list.
        // TODO: Sample based on estimated contributions as pdf.
        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

        // Sample local light source.
        AnalyticLightSample lightSample;
        if (!sampleLight(vertex.pos, gScene.getLight(lightIndex), sg, lightSample)) return false;

        // Setup returned sample.
        ls.pdf = lightSample.pdf / lightCount;
        ls.Li = lightSample.Li * lightCount;
        // Offset shading position to avoid self-intersection.
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        // Analytic lights do not currently have a geometric representation in the scene.
        // Do not worry about adjusting the ray length to avoid self-intersections at the light.
        ls.distance = lightSample.distance;
        ls.dir = lightSample.dir;

        return any(ls.Li > 0.f);
    }

    void analyticLightSampleSkipRandomNumber(inout SampleGenerator sg)
    {
        uint lightCount = gScene.getLightCount();
        if (!kUseAnalyticLights || lightCount == 0) return;
        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

        LightData light = gScene.getLight(lightIndex);

        switch (light.type)
        {
        case LightType::Rect:
        case LightType::Sphere:
        case LightType::Disc:
        case LightType::Distant:
            sampleNext2D(sg);
        default:
        }
    }

    /** Return the probabilities for selecting different light types.
        \param[out] p Probabilities.
    */
    void getLightTypeSelectionProbabilities(out float p[3])
    {
        // Set relative probabilities of the different sampling techniques.
        // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
        p[0] = kUseEnvLight ? 1.f : 0.f;
        p[1] = kUseEmissiveLights ? 1.f : 0.f;
        p[2] = kUseAnalyticLights ? 1.f : 0.f;

        // Normalize probabilities. Early out if zero.
        float sum = p[0] + p[1] + p[2];
        if (sum == 0.f) return;

        float invSum = 1.f / sum;
        p[0] *= invSum;
        p[1] *= invSum;
        p[2] *= invSum;
    }

    float getEnvMapSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[0]; }
    float getEmissiveSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[1]; }
    float getAnalyicSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[2]; }

    /** Select a light type for sampling.
        \param[out] lightType Selected light type.
        \param[out] pdf Probability for selected type.
        \param[in,out] sg Sample generator.
        \return Return true if selection is valid.
    */
    bool selectLightType(out uint lightType, out float pdf, inout SampleGenerator sg)
    {
        float p[3];
        getLightTypeSelectionProbabilities(p);

        float u = sampleNext1D(sg);
        pdf = 0.0f;

        //[unroll]
        for (lightType = 0; lightType < 3; ++lightType)
        {
            if (u < p[lightType])
            {
                pdf = p[lightType];
                return true;
            }
            u -= p[lightType];
        }

        return false;
    }

    /** Samples a light source in the scene.
        This function first stochastically selects a type of light source to sample,
        and then calls that the sampling function for the chosen light type.
        The upper/lower hemisphere is defined as the union of the hemispheres w.r.t. to the shading and face normals.
        \param[in] vertex Path vertex.
        \param[in] sampleUpperHemisphere True if the upper hemisphere should be sampled.
        \param[in] sampleLowerHemisphere True if the lower hemisphere should be sampled.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateLightSample(const PathVertex vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {};

        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return false;

        bool valid = false;
        if (kUseEnvLight && lightType == (uint)LightSampleType::EnvMap)
        {
            valid = generateEnvMapSample(vertex, sg, ls);
        }
        if (kUseEmissiveLights && lightType == (uint)LightSampleType::Emissive)
        {
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;

            {
                valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls);
            }
        }
        if (kUseAnalyticLights && lightType == (uint)LightSampleType::Analytic)
        {
            valid = generateAnalyticLightSample(vertex, sg, ls);
        }
        if (!valid) return false;

        // Reject samples in non-requested hemispheres.
        float cosTheta = dot(vertex.normal, ls.dir);
        // Flip the face normal to point in the same hemisphere as the shading normal.
        float3 faceNormal = sign(dot(vertex.normal, vertex.faceNormal)) * vertex.faceNormal;
        float cosThetaFace = dot(faceNormal, ls.dir);
        if (!sampleUpperHemisphere && (max(cosTheta, cosThetaFace) >= -kMinCosTheta)) return false;
        if (!sampleLowerHemisphere && (min(cosTheta, cosThetaFace) <= kMinCosTheta)) return false;

        // Account for light type selection.
        ls.lightType = lightType;
        ls.pdf *= selectionPdf;
        ls.Li /= selectionPdf;

        return true;
    }

    void skipLightSampleRandomNumbers(inout SampleGenerator sg)
    {
        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return;

        bool valid = false;
        if (kUseEnvLight && lightType == (uint)LightSampleType::EnvMap)
        {
            if (!kUseEnvLight) valid = false;
            else sampleNext2D(sg);
        }
        if (kUseEmissiveLights && lightType == (uint)LightSampleType::Emissive)
        {
            if (!kUseEmissiveLights)
            {
                valid = false;
            }
            else
            {
                // emissiveSampler.skipRandomNumber(sg);
                if (gScene.lightCollection.isEmpty())
                {
                    return;
                }
                sampleNext1D(sg);
                sampleNext1D(sg);
                sampleNext2D(sg);
            }
        }
        if (kUseAnalyticLights && lightType == (uint)LightSampleType::Analytic)
        {
            analyticLightSampleSkipRandomNumber(sg);
        }
    }

    /** Trace active path to find the next path vertex.
        \param[in,out] path The path state.
    */
    void nextVertex(inout PathState path)
    {
        // Setup ray.
        Ray ray = path.getScatterRay();

        logTraceRay(PixelStatsRayType::ClosestHit);
        HitInfo hit;
        float hitT;
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        if (sceneRayQuery.traceRay(ray, hit, hitT, RAY_FLAG_NONE, 0xff))
        {
            // Store hit info.
            path.setHit(hit);
            path.length++;
            path.sceneLength += float16_t(hitT);
        }
        else
        {
            path.clearHit();
            //path.sceneLength = float16_t(kNRDInvalidPathLength);
        }
    }

    /** Trace a visibility ray to evaluate light visibility.
        \param[in] ray Visibility ray.
        \return true if the light is visible, false if occluded.
    */
    bool traceVisibilityRay(const Ray ray)
    {
        // TODO rename to PixelStatsRayType::Visibility
        logTraceRay(PixelStatsRayType::Visibility); //logTraceRay(PixelStatsRayType::Shadow);
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);//return sceneRayQuery.traceVisibilityRay(ray.toRayDesc(), RAY_FLAG_NONE, 0xff);
    }

    /** Handle scattering in homogeneous volumes.
    \return Returns true if a scattering event took place.
*/
    bool handleHomogeneousVolumeExtinction(inout PathState path, const ShadingData sd)
    {
        // TODO: implement
        //if (!path.interiorList.isEmpty())
        //{
        //    HomogeneousVolumeSampler hvs = HomogeneousVolumeSampler::create(path.interiorList.getTopMaterialID());
        //    const float surfaceDistance = length(path.origin - sd.posW);

        //    // our implementation doesn't support volume scattering
        //    {
        //        // Handle volume extinction.
        //        updatePathThroughput(path, exp(-hvs.sigmaT * surfaceDistance));
        //    }
        //}

        return false;
    }


    /** Handle hits on dielectrics.
        \return True if this is an valid intersection, false if it is rejected.
    */
    bool handleNestedDielectrics(inout ShadingData sd, inout PathState path)
    {
        // TODO: How to handle eta in newer falcor version?

        // Compute relative index of refraction at interface.
        //sd.eta = computeRelativeIoR(path.interiorList, sd.materialID, sd.frontFacing);

        // Check for false intersections.
        uint nestedPriority = sd.mtl.getNestedPriority();//gScene.materials[sd.materialID].getNestedPriority();
        if (!path.interiorList.isTrueIntersection(nestedPriority))
        {
            if (path.rejectedHits < kMaxRejectedHits)
            {
                path.rejectedHits++;
                path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                path.origin = sd.computeNewRayOrigin(false);
                path.length--;
            }
            else
            {
                path.terminate();
            }
            return false;
        }

        // Limit specular transmission paths.
        let bsdf = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0.f), 0); // TODO: If adjust shading normals add flag to hints
        let bsdfProperties = bsdf.getProperties(sd);
        if (kLimitTransmission && any(bsdfProperties.specularTransmissionAlbedo > 0.f))
        {
            const uint transmissionBounces = path.getBounces(BounceType::Transmission);
            if (transmissionBounces >= kMaxTransmissionReflectionDepth) sd.mtl.setActiveLobes((uint)LobeType::All & ~((uint)LobeType::DeltaReflection | (uint)LobeType::SpecularReflection));
            //if (transmissionBounces >= kMaxTransmissionRefractionDepth) sd.eta = 1.f;
        }

        return true;
    }

    /** Apply russian roulette to terminate paths early.
        \param[in,out] path Path.
        \return Returns true if path was terminated.
    */
    bool terminatePathByRussianRoulette(inout PathState path)
    {
        const float rrVal = luminance(path.getCurrentThp());
        const float prob = max(0.f, 1.f - rrVal);
        if (sampleNext1D(path.sg) < prob)
        {
            path.terminate();
            return true;
        }

        path.russianRoulettePdf *= 1.f - prob;

        return false;
    }


    void generateReplayPrimaryScatterRay(inout PathState path, ShadingData sd, float3 dir = 0.f)
    {
        // Generate the next path segment or terminate.
        bool valid = true;

        if (all(dir == 0.f))
        {
            valid = generateScatterRay(sd, path);
        }
        else // generate with given direction
        {
            valid = generateScatterRayGivenDirection(sd, dir, path);
        }

        if (!valid)
        {
            path.terminate();
        }
        else
        {
            let bsdf = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0.f), 0); // TODO: If adjust shading normals add flag to hints
            let bsdfProperties = bsdf.getProperties(sd);
            path.isLastVertexClassifiedAsRough = // kSeparatePathBSDF ?
                (!path.isSpecularBounce() || bsdfProperties.roughness > params.specularRoughnessThreshold);// :
                //classifyAsRough(sd, params.specularRoughnessThreshold);
        }
    }

    /** Handle the case when a scatter ray hits the scene.
        After handling the hit, a new scatter ray is generated or the path is terminated.
        \param[in,out] path The path state.
    */
    void handleHit(inout PathState path)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Compute volume absorption
        // - Compute MIS weight if path.length > 0 and emissive hit
        // - Add emitted radiance
        // - Sample light(s) using shadow rays
        // - Sample scatter ray or terminate

        TriangleHit th = path.hit.getTriangleHit();

        bool isCurveHit = path.hit.getType() == HitType::Curve;

        // Load vertex/material attributes. This is a long latency operation.
        ShadingData sd;

        if (kPrimaryLodMode == TexLODMode::RayDiffs && path.length == 0 && path.hit.getType() == HitType::Triangle)
        {
            // Filtered lookups at primary hit
            float2 ddx, ddy;
            computeDerivativesAtPrimaryTriangleHit(path.hit.getTriangleHit(), path.getPixel(), params.frameDim, ddx, ddy);
            sd = loadShadingDataFootprint(path.hit, -path.dir, true, ddx, ddy);
        }
        else
        {
            float lod = path.length > 0 ? params.lodBias : 0.f;
            sd = loadShadingData(path.hit, -path.dir, path.length == 0, lod);
        }

        // Log
        logPathVertex();

        // Check if this is the primary hit.
        const bool isPrimaryHit = path.length == 0;

        // daqi: when doing random number replay, we will terminate the path when the length reaches the base path length and 
        // when the path types match. In this case, if the base path is a escaped path (as opposed to an NEE path), we should terminate when we 
        // find that offset path can also be an escaped path
        const bool terminateRandomReplayForEscape =
            path.enableRandomReplay &&
            path.length - 1 == path.randomReplayLength &&
            path.randomReplayIsEscaped &&
            path.randomReplayLength >= 1;

        // Terminate after scatter ray on last vertex has been processed.
        if (hasFinishedSurfaceBounces(path) || terminateRandomReplayForEscape)
        {
            path.terminate();
            return;
        }

        // daqi: save as previous path origin to compute distance between vertices
        const float3 prevPathOrigin = path.origin;

        // Compute origin for rays traced from this path vertex.
        path.origin = sd.computeNewRayOrigin();

        // Determine if BSDF supports sampling with next-event estimation.
        // The available lobes depend on the material.
        const uint lobes = sd.mtl.getActiveLobes();//getBSDFLobes(sd);

        bool is_rcVertex = false;

        let bsdf = gScene.materials.getMaterialInstance(sd, ExplicitLodTextureSampler(0.f), 0); // TODO: If adjust shading normals add flag to hints
        let bsdfProperties = bsdf.getProperties(sd);

        // TODO: Use ReSTIR_FG definition
        const bool isFarField = length(path.origin - prevPathOrigin) >= params.nearFieldDistance;
        bool isCurrentVertexClassifiedAsRough = // kSeparatePathBSDF ?
            bsdfProperties.roughness > params.specularRoughnessThreshold;//hasRoughComponent(sd, params.specularRoughnessThreshold) :
            //classifyAsRough(sd, params.specularRoughnessThreshold);
        const bool isLastVertexAcceptableForRcPrev = path.isLastVertexClassifiedAsRough;

        // daqi: if we are not doing a hybrid shift replay, we should check if current vertex satisfy the condition to be used as a rcVertex
        if (!path.isReplayForHybridShift && PathSamplingMode(kPathSamplingMode) != PathSamplingMode::PathTracing)
        {
            const bool canConnect = (
                path.length >= 1 &&
                path.length < path.pathBuilder.rcVertexLength &&
                (!(params.localStrategyType & (uint)LocalStrategy::DistanceCondition
                    && !isFarField ||
                    params.localStrategyType & (uint)LocalStrategy::RoughnessCondition &&
                    !(isCurrentVertexClassifiedAsRough && isLastVertexAcceptableForRcPrev))
                )
            );

            if (!path.useHybridShift && path.length == 1 || path.useHybridShift && canConnect)
            {
                is_rcVertex = true;
                path.pathBuilder.rcVertexHit.initFromHitInfo(path.hit);
                // daqi: save the scatter PDF as cachedJacobian
                path.pathBuilder.cachedJacobian.x = path.prevScatterPdf;
                path.pathBuilder.pathFlags.insertIsDeltaEvent(path.isDelta(), true);
                path.pathBuilder.pathFlags.insertIsTransmissionEvent(path.isTransmission(), true);
                if (kSeparatePathBSDF)
                {
                    path.pathBuilder.pathFlags.insertIsSpecularBounce(path.isSpecularBounce(), true);
                }

                // daqi: save geometry term as part of cachedJacobian
                const float3 dist = prevPathOrigin - sd.posW;
                path.pathBuilder.cachedJacobian.z = abs(dot(sd.faceN, sd.V)) / dot(dist, dist);

                // daqi: save the current path length as the rcVertex length
                if (canConnect || !path.useHybridShift)
                {
                    path.pathBuilder.rcVertexLength = path.length;
                }
            }
        }
        // path.isReplayForHybridShift = true
        else if (PathSamplingMode(kPathSamplingMode) != PathSamplingMode::PathTracing)
        {
            bool invertible = true;
            bool shouldTerminate = false;

            // daqi: check situation where invertibility is violated
            if (path.length >= 1 && path.length <= path.pathBuilder.rcVertexLength - 1)
            {
                if (kSeparatePathBSDF)
                {
                    if (params.localStrategyType & (uint)LocalStrategy::RoughnessCondition)
                    {
                        // non-invertible because current vertex will be assigned as a diffuse bounce by connecting to the reconnection vertex
                        if (path.length == path.pathBuilder.rcVertexLength - 1 && isLastVertexAcceptableForRcPrev)
                        {
                            invertible = false;
                        }

                        if (params.localStrategyType & (uint)LocalStrategy::DistanceCondition && !isFarField)
                        {
                            invertible = true;
                        }
                    }
                    else
                    {
                        if (params.localStrategyType & (uint)LocalStrategy::DistanceCondition && isFarField)
                        {
                            invertible = false;
                        }
                    }
                }
                // kSeparatePathBSDF = false
                else
                {
                    if (!(
                        params.localStrategyType & (uint)LocalStrategy::DistanceCondition &&
                        !isFarField ||
                        params.localStrategyType & (uint)LocalStrategy::RoughnessCondition &&
                        !(
                            /*supportsNEE &&*/
                            isCurrentVertexClassifiedAsRough &&
                            isLastVertexAcceptableForRcPrev)
                        )
                    )
                    {
                        invertible = false;
                    }
                }
            }

            // daqi: if next vertex is rcVertex and the shift is still invertible, then we should terminate
            if (path.length == path.pathBuilder.rcVertexLength - 1 && invertible)
            {
                // not possible to generate a diffuse bounce
                if (params.localStrategyType & (uint)LocalStrategy::RoughnessCondition &&
                    (
                        kSeparatePathBSDF &&
                        (//path.pathBuilder.pathFlags.decodeIsSpecularBounce(true) ?
                            bsdfProperties.roughness <= params.specularRoughnessThreshold// :
                            /*!hasRoughComponent(sd, 1.f)*/
                        ) ||
                        !kSeparatePathBSDF &&
                        !isCurrentVertexClassifiedAsRough
                    )
                )
                {
                    // daqi: invalidate the shift
                    invertible = false;
                }
                // daqi: near field rejection case is checked in computeShiftedIntegrandReconnection in shift.slang
                else
                {
                    shouldTerminate = true;
                }
            }

            if (invertible)
            {
                // daqi: if it is invertible and we should terminate, save the hit information to set up a reconnection
                if (shouldTerminate)
                {
                    path.rcPrevVertexHit = path.hit;
                    path.rcPrevVertexWo = sd.V;
                    path.terminate();
                }
            }
            else // daqi: if the shift is not invertible, we should mark the case, which causes the shift to terminate when it goes back to shift.slang
            {
                invalidateAndTerminateReplayPath(path);
            }

            if (!path.isActive())
            {
                return;
            }
        }

        path.setLightSampled(false, false);

        // Temporary keep path throughput from before the scatter event.
        const float3 prevThp = path.getCurrentThp();

        // daqi: save current random seed, this is required to do random number replay to construct a path
        if (path.length == 0)
        {
            path.pathReservoir.initRandomSeed = path.sg.rng.state;//path.sg.getCurrentSeed();
        }

        // Generate the next path segment or terminate.
        // TODO: Combine with ReSTIR_FG?
        bool valid = generateScatterRay(sd, path);

        // forbid current vertex being rcVertex
        if (kSeparatePathBSDF && params.localStrategyType & (uint)LocalStrategy::RoughnessCondition)
        {
            const bool seenAsConnectible = !path.isSpecularBounce() || bsdfProperties.roughness > params.specularRoughnessThreshold;

            if (path.isReplayForHybridShift && seenAsConnectible && isLastVertexAcceptableForRcPrev && path.length >= 1)
            {
                // check if near field rejection is satisfied
                if (isFarField || !(params.localStrategyType & (uint)LocalStrategy::DistanceCondition))
                {
                    // non-invertible
                    invalidateAndTerminateReplayPath(path);
                }
            }
            else if (!path.isReplayForHybridShift && path.useHybridShift && !seenAsConnectible)
            {
                if (is_rcVertex)
                {
                    is_rcVertex = false; 
                    path.pathBuilder.rcVertexLength = kMaximumPathLength; // reset
                }
            }
            if (path.useHybridShift)
            {
                isCurrentVertexClassifiedAsRough = seenAsConnectible;
            }
        }

        // cache seed for temporal validation (only possible for non early direction reuse since it uses the cachedRandomSeed slot)
        if (path.length == path.pathBuilder.rcVertexLength)
        {
            path.pathBuilder.cachedRandomSeed = path.sg.rng.state;//path.sg.getCurrentSeed();
        }

        // daqi: save the incoming direction on rcVertex
        if (is_rcVertex && valid && (int(path.length > 1) == 0 || !BPR))
        {
            path.pathBuilder.rcVertexWi[0] = path.dir;
        }

        path.isLastVertexClassifiedAsRough = isCurrentVertexClassifiedAsRough;

        if (!path.isActive())
        {
            return;
        }

        // Check if this is the last path vertex.
        const bool isLastVertex = hasFinishedSurfaceBounces(path);

        // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
        if (kUseNEE && !kUseMIS && isLastVertex && path.isLightSamplable())
        {
            valid = false;
        }

        // Terminate caustics paths.
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && path.isSpecular())
        {
            valid = false;
        }

        if (!valid)
        {
            path.terminate();
        }
    }

    /** Handle the case when a scatter ray misses the scene.
        \param[in,out] path The path state.
    */
    void handleMiss(inout PathState path)
    {
        // Upon miss:
        // - Compute MIS weight if previous path vertex sampled a light
        // - Evaluate environment map
        // - Write guiding data
        // - Terminate the path

        path.terminate();
    }

    // function to trace a path using random number replay
    float3 traceRandomReplayPath(HitInfo hit, ShadingData sd, ShadingData srcSd, Ray ray, inout PathReservoir tempPathReservoir)
    {
        PathState path;
        generateRandomReplayPath(path, hit, ray, tempPathReservoir.initRandomSeed, tempPathReservoir.pathFlags.pathLength(), tempPathReservoir.pathFlags.lastVertexNEE(), false);
        path.isReplayForHybridShift = false;

        path.origin = sd.computeNewRayOrigin();

        // set up the two scatter directions (or half vectors) for reuse

        generateReplayPrimaryScatterRay(path, sd);

        if (path.isActive()) nextVertex(path);

        float randomReplayProb = 1.f;

        while (path.isActive() && path.isHit())
        {
            // Handle surface hit 
            if (path.isActive())
            {
                if (path.isHit())
                {
                    handleHit(path);
                }
            }

            // Move to the next path vertex.
            if (path.isActive() && path.isHit()) nextVertex(path);
        }

        // Handle the miss and terminate path.
        if (path.isActive())
        {
            handleMiss(path);
        }

        // we should copy out the pathReservoir, as it contains information like rcVertex which will be needed for MIS
        tempPathReservoir = path.pathReservoir;

        return path.L;
    }

    float3 traceRandomReplayPathHybridSimple(HitInfo hit, ShadingData sd, Ray ray, ReSTIRPathFlags pathFlags,
        uint initRandomSeed, inout HitInfo rcPrevHit, inout float3 rcPrevHitWo)
    {
        PathState path;
        bool isRcVertexEscapedVertex = pathFlags.pathLength() + 1 == pathFlags.rcVertexLength();
        generateRandomReplayPath(path, hit, ray, initRandomSeed, pathFlags.pathLength(), pathFlags.lastVertexNEE(), isRcVertexEscapedVertex);
        path.useHybridShift = true;
        path.isReplayForHybridShift = true;
        path.pathBuilder.rcVertexLength = pathFlags.rcVertexLength();

        setDataBeforeTermination(rcPrevHit, rcPrevHitWo);

        path.origin = sd.computeNewRayOrigin();

        // this is also applicable for random number reuse
        path.pathBuilder.pathFlags.flags = 0;
        path.pathBuilder.pathFlags.transferSpecularBounceInformation(pathFlags);

        // set up the two scatter directions (or half vectors) for reuse

        generateReplayPrimaryScatterRay(path, sd);
        if (path.isActive()) nextVertex(path);
        else
        {
            return 0.f;
        }

        while (path.isActive() && path.isHit())
        {
            // Volume scattering is ignored
            // Handle surface hit 
            if (path.isActive())
            {
                if (path.isHit())
                {
                    handleHit(path);
                }
            }

            // Move to the next path vertex.
            if (path.isActive() && path.isHit()) nextVertex(path);
        }

        // Handle the miss and terminate path.
        if (path.isActive())
        {
            handleMiss(path);

            // non-invertible: path length doesn't match
            if (path.length != pathFlags.pathLength())
            {
                return 0.f;
            }
        }

        rcPrevHit = HitInfo();

        float3 L = 1.f;

        if (pathFlags.rcVertexLength() <= pathFlags.pathLength() || isRcVertexEscapedVertex)
        {
            rcPrevHit = path.rcPrevVertexHit;
            rcPrevHitWo = path.rcPrevVertexWo;
            L = path.getCurrentThp();
        }
        else // this branch is the case where rcVertex does not exist (or infinitely far)
        {
            L = path.L;
        }

        return L;
    }

    //void traceRandomReplayGetDenoiserData(HitInfo hit, ShadingData sd, Ray ray, ReSTIRPathFlags pathFlags,
    //    uint initRandomSeed, inout float hitDist, inout NRDPathType pathType, inout float3 sampleReflectance)
    //{
    //    PathState path;
    //    bool isRcVertexEscapedVertex = pathFlags.pathLength() + 1 == pathFlags.rcVertexLength();
    //    generateRandomReplayPath(path, hit, ray, initRandomSeed, pathFlags.pathLength(), pathFlags.lastVertexNEE(), isRcVertexEscapedVertex);
    //    path.useHybridShift = true;
    //    path.isReplayForHybridShift = true;
    //    path.pathBuilder.rcVertexLength = pathFlags.rcVertexLength();

    //    if (path.pathBuilder.rcVertexLength > 1)
    //    {
    //        path.origin = sd.computeNewRayOrigin();

    //        // this is also applicable for random number reuse
    //        path.pathBuilder.pathFlags.flags = 0;

    //        generateReplayPrimaryScatterRay(path, sd);
    //        if (path.isActive()) nextVertex(path);
    //        else
    //        {
    //            // this shouldn't happen! 
    //            return;
    //        }

    //        hitDist = float(path.sceneLength);
    //        if (path.isDiffusePrimaryHit()) pathType = NRDPathType::Diffuse;
    //        else if (path.isSpecularPrimaryHit()) pathType = NRDPathType::Specular;
    //        else pathType = NRDPathType::Residual;

    //        sampleReflectance = 1.0f;//computeSampleReflectance(kUseNRDDemodulation, path.isDiffusePrimaryHit(), path.isSpecularPrimaryHit(), sd);

    //        return;
    //    }
    //}

    void traceTemporalUpdate(ShadingData rcPrevSd, inout PathReservoir tempPathReservoir)
    {
        int rcVertexLength = tempPathReservoir.pathFlags.rcVertexLength();
        ShadingData rcVertexSd = loadShadingDataWithPrevVertexPosition(tempPathReservoir.rcVertexHit.getHitInfo(), rcPrevSd.posW, false);
        PathState path;

        //rcVertex is NEE
        if (tempPathReservoir.pathFlags.lastVertexNEE() && tempPathReservoir.pathFlags.pathLength() == rcVertexLength)
        {
            path = {};
            path.length = rcVertexLength;
            // call light sampling function
            LightSample ls;
            bool validSample = true;
            {
                uint lobes = rcVertexSd.mtl.getActiveLobes();//getBSDFLobes(rcVertexSd);
                // Setup path vertex.
                PathVertex vertex = PathVertex.create(/*path.getVertexIndex()*/ path.length + 1, rcVertexSd.posW, rcVertexSd.frame.N, rcVertexSd.faceN);

                // Determine if upper/lower hemispheres need to be sampled.
                bool sampleUpperHemisphere = ((lobes & (uint)LobeType::NonDeltaReflection) != 0);
                bool sampleLowerHemisphere = ((lobes & (uint)LobeType::NonDeltaTransmission) != 0);

                path.sg = SampleGenerator(tempPathReservoir.rcRandomSeed);
                validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, path.sg, ls);
            }

            bool visible = true;
            if (validSample)
            {
                const Ray ray = ls.getVisibilityRay();
                visible = traceVisibilityRay(ray);


                if (ls.lightType == (uint)LightSampleType::Analytic && ls.pdf == 0.f)
                {
                    ls.pdf = getAnalyicSelectionProbability(); // analytic light doesn't have a solid angle PDF.
                }
            }

            if (validSample && visible)
            {
                tempPathReservoir.rcVertexWi[0] = ls.dir;
                tempPathReservoir.rcVertexIrradiance[0] = ls.Li * ls.pdf;
                tempPathReservoir.pathFlags.insertLightType(ls.lightType);
                tempPathReservoir.lightPdf = ls.pdf;
            }
            else tempPathReservoir.rcVertexIrradiance[0] = 0.f;
        }
        else
        {
            Ray emptyRay = Ray(float3(0.f), float3(0.f), 0.f, kRayTMax);

            // should always be false in this case
            bool isRcVertexEscapedVertex = false;
            generateRandomReplayPath(path, tempPathReservoir.rcVertexHit.getHitInfo(), emptyRay, tempPathReservoir.rcRandomSeed, tempPathReservoir.pathFlags.pathLength(), tempPathReservoir.pathFlags.lastVertexNEE(), isRcVertexEscapedVertex);
            path.length = rcVertexLength;
            path.useHybridShift = false;
            path.isReplayForHybridShift = false;

            uint lobes = rcVertexSd.mtl.getActiveLobes();//getBSDFLobes(rcVertexSd);
            bool sampleUpperHemisphere = ((lobes & (uint)LobeType::NonDeltaReflection) != 0);
            if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;
            bool sampleLowerHemisphere = ((lobes & (uint)LobeType::NonDeltaTransmission) != 0);
            path.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);

            // start from path length
            float3 scatterDir = tempPathReservoir.rcVertexWi[0];
            path.pathBuilder.rcVertexLength = rcVertexLength;
            path.origin = rcVertexSd.computeNewRayOrigin(!tempPathReservoir.pathFlags.decodeIsTransmissionEvent(false));

            path.pathBuilder.pathFlags.transferDeltaInformation(tempPathReservoir.pathFlags);
            path.pathBuilder.pathFlags.transferTransmissionInformation(tempPathReservoir.pathFlags);
            path.pathBuilder.pathFlags.transferSpecularBounceInformation(tempPathReservoir.pathFlags);
            path.pathReservoir.rcVertexIrradiance[0] = 0.f;

            generateReplayPrimaryScatterRay(path, rcVertexSd, scatterDir);
            path.sg = SampleGenerator(tempPathReservoir.rcRandomSeed);
            if (path.isActive()) nextVertex(path);

            while (path.isActive() && path.isHit())
            {
                // Volume scattering is ignored
                // Handle surface hit 
                if (path.isActive())
                {
                    if (path.isHit())
                    {
                        handleHit(path);
                    }
                }

                // Move to the next path vertex.
                if (path.isActive() && path.isHit()) nextVertex(path);
            }

            // Handle the miss and terminate path.
            if (path.isActive())
            {
                handleMiss(path);
            }

            // unlike traceRandomReplayPath(), we don't directly copy the path.pathReservoir.
            // this is because we start at path.length == 2, we can also terminate early,
            // so pathReservoir won't record enough information for a complete path reservoir
            // so we need to manually copy some members we need later
            // this cannot be replaced by pathBuilder because of escapedRcVertex not setting pathBuilder.rcVertexHit (TODO: refactor)
            tempPathReservoir.rcVertexIrradiance[0] = path.pathReservoir.rcVertexIrradiance[0];
            tempPathReservoir.lightPdf = path.pathReservoir.lightPdf;
            tempPathReservoir.pathFlags.insertLightType(path.pathReservoir.pathFlags.lightType());
        }
    }

    /** Finalize the path state before writing to the output buffer.
    */
    void finalize(inout PathState path)
    {
        path.pathBuilder.finalize(path.pathReservoir);
    }

    /** Write path contribution to output buffer.
    */
    void writeOutput(PathState path, inout PathReservoir giReservoir, int sampleId)
    {
        // Log path length.
        logPathLength(getTerminatedPathLength(path));

        const uint2 pixel = path.getPixel();

        const uint reservoirIdx = params.getReservoirOffset(pixel);

        static const uint itersPerShaderPass = PathSamplingMode(kPathSamplingMode) == PathSamplingMode::PathTracing ? kSamplesPerPixel : kCandidateSamples;

        if (itersPerShaderPass == 1)
        {
            float3 L = 0.f;

            if (useDirectLighting) L += path.LDeltaDirect;

            if (PathSamplingMode(kPathSamplingMode) == PathSamplingMode::PathTracing)
                L += path.L;
            else if (PathSamplingMode(kPathSamplingMode) == PathSamplingMode::PathReuse)
            {
                path.pathReservoir.weight = 1.f;
                if (isLastRound)
                {
                    float3 color = path.pathReservoir.F + path.pathReservoir.cachedJacobian;
                    if (any(isinf(color) || isnan(color))) color = 0.f;
                    L += color;
                }
            }
            else
            {
                path.pathReservoir.finalizeRIS();

                if (isLastRound)
                {
                    // Write color directly to frame buffer.
                    float3 color = path.pathReservoir.F * path.pathReservoir.weight;

                    if (any(isinf(color) || isnan(color))) color = 0.f;

                    L += color;
                }
            }

            if (useDirectLighting && (PathSamplingMode(kPathSamplingMode) == PathSamplingMode::PathTracing || isLastRound))
            {
                L += directLighting[pixel].rgb;
            }

            if (any(isinf(L) || isnan(L))) L = 0.f;

            if (PathSamplingMode(kPathSamplingMode) != PathSamplingMode::PathTracing)
                L /= kSamplesPerPixel;

            if (PathSamplingMode(kPathSamplingMode) == PathSamplingMode::PathTracing || kSamplesPerPixel == 1 || gSppId == 0 )
                outputColor[pixel] = float4(L, 1.f);
            else
                outputColor[pixel] += float4(L, 1.f);

            if (PathSamplingMode(kPathSamplingMode) != PathSamplingMode::PathTracing)
                outputReservoirs[reservoirIdx] = path.pathReservoir;
        }
        else
        {
            if (PathSamplingMode(kPathSamplingMode) == PathSamplingMode::PathTracing)
            {
                float3 L = 0.f;
                if (sampleId == 0)
                {
                    L = path.L / kSamplesPerPixel;
                }
                else 
                {
                    L = giReservoir.F;
                    L += path.L / kSamplesPerPixel;
                }
                if (useDirectLighting && sampleId == kSamplesPerPixel - 1) L += path.LDeltaDirect;

                if (any(isinf(L) || isnan(L))) L = 0.f;

                if (sampleId == kSamplesPerPixel - 1)
                {
                    if (useDirectLighting)
                    {
                        L += directLighting[pixel].rgb;
                        if (any(isnan(L) || isinf(L))) L = 0.f;
                    }

                    outputColor[pixel] = float4(L, 1.f);
                }
                else giReservoir.F = L;
            }
            else
            {

                if (sampleId == 0)
                {
                    giReservoir = path.pathReservoir; // we are the first one
                }
                else
                {
                    if (giReservoir.mergeInSamplePixel(path.pathReservoir, path.sg))
                    {
                        giReservoir.F = path.pathReservoir.F;
                    }
                }

                if (sampleId == kCandidateSamples - 1) // last sample
                {
                    float3 L = 0.f;
                    if (useDirectLighting) L += path.LDeltaDirect;

                    giReservoir.finalizeRIS();
                    if (isLastRound)
                    {
                        float3 color = giReservoir.F * giReservoir.weight;
                        
                        if (any(isinf(color) || isnan(color))) color = 0.f;
                        L += color;
                    }

                    if (useDirectLighting && isLastRound)
                    {
                        L += directLighting[pixel].rgb;
                        if (any(isnan(L) || isinf(L))) L = 0.f;
                    }

                    L /= kSamplesPerPixel;

                    if (kSamplesPerPixel == 1 || gSppId == 0)
                        outputColor[pixel] = float4(L, 1.f);
                    else
                        outputColor[pixel] += float4(L, 1.f);

                    outputReservoirs[reservoirIdx] = giReservoir;
                }
            }
        }

        //if (kOutputNRDData && PathSamplingMode(kPathSamplingMode) == PathSamplingMode::PathTracing) 
        //{
        //    float3 diffuseRadiance = 0.f;
        //    float3 specularRadiance = 0.f;
        //    float3 residualRadiance = 0.f;
        //    float hitDist = path.hitDist;

        //    {
        //        NRDPathType pathType;
        //        if (path.isDiffusePrimaryHit()) pathType = NRDPathType::Diffuse;
        //        else if (path.isSpecularPrimaryHit()) pathType = NRDPathType::Specular;
        //        else pathType = NRDPathType::Residual;

        //        float3 radiance = path.L + path.LDeltaDirect;

        //        if (useDirectLighting)
        //        {
        //            radiance += directLighting[pixel].rgb;
        //        }

        //        float3 reflectance = path.sharedScatterDir;
        //        reflectance = max(0.04f, reflectance);
        //        float3 emission = path.rcPrevVertexWo;
        //        float3 demodulatedRadiance = max(0.f, (radiance - emission)) / reflectance;

        //        switch (pathType)
        //        {
        //        case NRDPathType::Diffuse:
        //            diffuseRadiance += demodulatedRadiance;
        //            break;
        //        case NRDPathType::Specular:
        //            specularRadiance += demodulatedRadiance;
        //            break;
        //        default:
        //            // Do not demodulate residual.
        //            residualRadiance += radiance;
        //            break;
        //        }
        //    }

        //    const uint spp = kSamplesPerPixel;
        //    const float invSpp = spp > 0 ? 1.f / spp : 1.f; // Setting invSpp to 1.0 if we have no samples to avoid NaNs below.

        //    if (sampleId == 0)
        //    {
        //        outputNRDDiffuseRadianceHitDist[pixel] = float4(invSpp * diffuseRadiance, invSpp * hitDist);
        //        outputNRDSpecularRadianceHitDist[pixel] = float4(invSpp * specularRadiance, invSpp * hitDist);
        //        outputNRDResidualRadianceHitDist[pixel] = float4(invSpp * residualRadiance, invSpp * hitDist);
        //    }
        //    else
        //    {
        //        outputNRDDiffuseRadianceHitDist[pixel] += float4(invSpp * diffuseRadiance, invSpp * hitDist);
        //        outputNRDSpecularRadianceHitDist[pixel] += float4(invSpp * specularRadiance, invSpp * hitDist);
        //        outputNRDResidualRadianceHitDist[pixel] += float4(invSpp * residualRadiance, invSpp * hitDist);
        //    }
        //}
    }
};
