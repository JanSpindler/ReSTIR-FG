#pragma once

#include <Falcor.h>
#include <RenderGraph/RenderPass.h>
#include "Shader/Params.slang"
#include <Rendering/Materials/TexLODTypes.slang>
#include <Rendering/Lights/EmissiveLightSampler.h>

using namespace Falcor;

class PrefixRestir
{
public:
    PrefixRestir() = default;
    PrefixRestir(ref<Device> pDevice, DefineList defines);

    void PrepareBuffers(RenderContext* pRenderContext, const uint2 screenSize);
    void SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene);
    bool RenderUI(Gui::Widgets& widget);
    void Run(RenderContext* pRenderContext, const RenderData& renderData);

    constexpr bool IsActive() const { return m_Active; }

private:
    struct StaticParams
    {
        // Rendering parameters
        uint32_t samplesPerPixel = 1; ///< Number of samples (paths) per pixel, unless a sample density map is used.
        uint32_t candidateSamples = 1;
        uint32_t maxSurfaceBounces = 9;   ///< Max number of surface bounces (diffuse + specular + transmission), up to kMaxPathLenth.
        uint32_t maxDiffuseBounces = -1;  ///< Max number of diffuse bounces (0 = direct only), up to kMaxBounces. This will be initialized
                                          ///< at startup.
        uint32_t maxSpecularBounces = -1; ///< Max number of specular bounces (0 = direct only), up to kMaxBounces. This will be initialized
                                          ///< at startup.
        uint32_t maxTransmissionBounces = -1; ///< Max number of transmission bounces (0 = none), up to kMaxBounces. This will be
                                              ///< initialized at startup.
        uint32_t sampleGenerator = SAMPLE_GENERATOR_TINY_UNIFORM; ///< Pseudorandom sample generator type.
        bool adjustShadingNormals = false;                        ///< Adjust shading normals on secondary hits.
        bool useBSDFSampling = true;               ///< Use BRDF importance sampling, otherwise cosine-weighted hemisphere sampling.
        bool useNEE = true;                        ///< Use next-event estimation (NEE). This enables shadow ray(s) from each path vertex.
        bool useMIS = true;                        ///< Use multiple importance sampling (MIS) when NEE is enabled.
        bool useRussianRoulette = false;           ///< Use russian roulette to terminate low throughput paths.
        bool useAlphaTest = true;                  ///< Use alpha testing on non-opaque triangles.
        uint32_t maxNestedMaterials = 2;           ///< Maximum supported number of nested materials.
        bool useLightsInDielectricVolumes = false; ///< Use lights inside of volumes (transmissive materials). We typically don't want this
                                                   ///< because lights are occluded by the interface.
        bool limitTransmission = false; ///< Limit specular transmission by handling reflection/refraction events only up to a given
                                        ///< transmission depth.
        uint32_t maxTransmissionReflectionDepth = 0; ///< Maximum transmission depth at which to sample specular reflection.
        uint32_t maxTransmissionRefractionDepth = 0; ///< Maximum transmission depth at which to sample specular refraction (after that, IoR
                                                     ///< is set to 1).
        bool disableCaustics = false;                ///< Disable sampling of caustics.
        bool disableDirectIllumination = true;       ///< Disable all direct illumination.
        TexLODMode primaryLodMode = TexLODMode::Mip0;      ///< Use filtered texture lookups at the primary hit.
        ColorFormat colorFormat = ColorFormat::LogLuvHDR;  ///< Color format used for internal per-sample color and denoiser buffers.
        MISHeuristic misHeuristic = MISHeuristic::Balance; ///< MIS heuristic.
        float misPowerExponent = 2.f; ///< MIS exponent for the power heuristic. This is only used when 'PowerExp' is chosen.
        EmissiveLightSamplerType emissiveSampler = EmissiveLightSamplerType::Power; ///< Emissive light sampler to use for NEE.

        bool useDeterministicBSDF = true; ///< Evaluate all compatible lobes at BSDF sampling time.

        ReSTIRMISKind spatialMisKind = ReSTIRMISKind::Pairwise;
        ReSTIRMISKind temporalMisKind = ReSTIRMISKind::Talbot;

        ShiftMapping shiftStrategy = ShiftMapping::Hybrid;
        bool temporalUpdateForDynamicScene = false;

        PathSamplingMode pathSamplingMode = PathSamplingMode::ReSTIR;

        bool separatePathBSDF = true;

        bool rcDataOfflineMode = false;

        // Denoising parameters
        bool useNRDDemodulation = true; ///< Global switch for NRD demodulation.

        DefineList GetDefines() const;
    };

    ref<Device> m_Device;
    ref<Scene> m_Scene;

    RestirPathTracerParams m_Params; // Runtime path tracer parameters.
    StaticParams m_StaticParams;     // Static path tracer parameters.

    ref<Buffer> m_OutputReservoirs; // Output paths from the path sampling stage
    ref<Buffer> m_TemporalReservoirs;
    ref<Buffer> m_ReconnectionDataBuffer;
    ref<Texture> m_TemporalVBuffer;

    bool m_Active = false;
    bool m_EnableTemporalReprojection = true;
    bool m_NoResamplingForTemporalReuse = false;
    bool m_UseDirectLighting = false;

    ref<ComputePass> m_TemporalPathRetracePass;
    ref<ComputePass> m_TemporalReusePass; // Merges reservoirs

    void SetShaderData(const ShaderVar& var, const RenderData& renderData, bool isPathTracer, bool isPathGenerator) const;
    void PathRetracePass(RenderContext* pRenderContext, const RenderData& renderData);
    void PathReusePass(RenderContext* pRenderContext, const RenderData& renderData);
};
