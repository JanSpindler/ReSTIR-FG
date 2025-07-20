#pragma once

#include <Falcor.h>
#include <Utils/CudaUtils.h>
#include <Utils/Math/VectorMath.h>
#include "FirstHitPhotonInfo.h"

using namespace Falcor;

class GaussianPhotonGuiding
{
public:
    enum class Optimizer : uint
    {
        SGD = 0u,
        Adam = 1u
    };

    enum class Initialization : uint
    {
        Random = 0u,
        Robust = 1u
    };

    GaussianPhotonGuiding() = default;

    GaussianPhotonGuiding(ref<Device> device, const DefineList& defines) : m_Device(device), m_Defines(defines)
    {
    }

    void SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene);
    void PrepareBuffers(const uint2 screenSize, RenderContext* renderContext, const uint2 maxPhotonCount);
    bool RenderUI(Gui::Widgets& widget);
    void ResetSceneTextures();
    void ResetPhotonFirstHitMap();

    void GenerateCausticClusters(RenderContext* renderContext);
    void TrackActualFirstHitPhotonCount(RenderContext* renderContext);
    void RobustInitialization(RenderContext* renderContext);
    void CalculateGaussianGradientCuda(RenderContext* renderContext);
    void OptimizeGaussiansPass(RenderContext* renderContext);
    void RandomReplacePass(RenderContext* renderContext);
    void CalculateSoftmaxWeightsPass(RenderContext* renderContext);
    void EndFrame(RenderContext* renderContext);

    void ClearBuffersForGeneratePhotons(RenderContext* renderContext);
    void SetGeneratePhotonsVars(const ShaderVar& var) const;

    void ClearBuffersForPhotonCollection(RenderContext* renderContext);
    void SetCollectPhotonsVars(const ShaderVar& var) const;

    void SetFinalShadingVars(const ShaderVar& var) const;

    constexpr bool IsActive() const { return m_Active; }
    constexpr bool IsOptimizing() const { return m_Optimize; }
    constexpr uint GetFrameCountAfterOptimReset() const { return m_FrameCountAfterOptimReset; }
    constexpr bool IsRobustInitialization() const { return m_Initialization == Initialization::Robust; }

private:
    // Constants
    static inline const std::string m_ShaderModel = "6_5";

    static inline const std::string m_CalculateGaussainGradiantShader = "RenderPasses/ReSTIR_FG/Shader/CalculateGaussianGradient.cs.slang";
    static inline const std::string m_OptimizeGaussiansShader = "RenderPasses/ReSTIR_FG/Shader/OptimizeGaussians.cs.slang";
    static inline const std::string m_CalculateSoftmaxWeightsShader = "RenderPasses/ReSTIR_FG/Shader/CalculateSoftmaxWeights.cs.slang";

    static inline const Gui::DropdownList m_OptimizerList{{static_cast<uint>(Optimizer::SGD), "SGD"}, {static_cast<uint>(Optimizer::Adam), "Adam"}};
    static inline const Gui::DropdownList m_InitializationList{
        {static_cast<uint>(Initialization::Random), "Random"},
        {static_cast<uint>(Initialization::Robust), "Robust"}
    };

    //
    ref<Device> m_Device;
    DefineList m_Defines;

    // General
    bool m_Active = false;
    uint m_GaussianCount = 16; // Number of gaussians per light
    uint m_MaxFirstHitPhotonCount = 100000;
    uint m_ActualFirstHitPhotonCount = 0;
    uint m_GlobalPhotonWeight = 0;
    uint m_CausticPhotonWeight = 1;

    // Sampling
    float m_MinPdf = 0.0f;
    float m_Beta = 0.8f;

    // Scene
    ref<Scene> m_Scene;
    uint m_AnalyticLightCount = 0;
    uint m_GeometricLightCount = 0;
    std::vector<uint> m_CausticGeometryInstanceIDs;
    
    // Encoding
    float m_Cb = 20.0f;
    float m_Cs = 0.65f;

    // Initialization
    Initialization m_Initialization = Initialization::Robust;
    uint m_CausticClusterCount = 8;
    uint m_GenCausticPointCount = 1000;
    std::vector<float3> m_CausticClustersPos;
    std::vector<float> m_CausticClustersPSigma;

    // Optimization
    bool m_Optimize = true;
    Optimizer m_Optimizer = Optimizer::Adam;
    float m_LearningRate = 0.01f;
    float m_Beta1 = 0.9f;
    float m_Beta2 = 0.999f;
    uint m_FrameCountAfterOptimReset = 0;

    // Random replace
    bool m_RandomReplace = false;
    uint m_RandomReplaceCount = 4;
    uint m_RandomReplaceFrequency = 256;

    // Buffers
    InteropBuffer m_GaussianBuf;
    ref<Buffer> m_GaussianBufReadCPU;
    ref<Buffer> m_GaussianBufWriteCPU;
    ref<Texture> m_GaussianTex;
    InteropBuffer m_FirstHitPhotonCountBuf;
    ref<Buffer> m_FirstHitPhotonCountBufCPU;
    InteropBuffer m_FirstHitPhotonInfoBuf;
    ref<Buffer> m_FirstHitPhotonInfoBufCPU;
    InteropBuffer m_FirstHitCollectionCountsBuf;
    ref<Buffer> m_FirstHitCollectionCountsBufCPU;
    std::array<ref<Buffer>, 2> m_PhotonFirstHitMapBufs;
    ref<Buffer> m_LightFirstHitCountsBuf;
    InteropBuffer m_GradientBuf;
    ref<Buffer> m_OptimizationBuf;
    ref<Buffer> m_OptimizationResetBuf;
    ref<Buffer> m_OptimizationResetBufWriteCPU;
    InteropBuffer m_SoftmaxBuf;

    // Passes
    ref<ComputePass> m_OptimizeGaussiansPass;
    ref<ComputePass> m_CalculateSoftmaxPass;

    // Functions
    constexpr uint GetTotalLightCount() const { return m_AnalyticLightCount + m_GeometricLightCount; }
    constexpr uint GetTotalGaussianCount() const { return GetTotalLightCount() * m_GaussianCount; }
    constexpr float GetSceneSize() const { return math::length(m_Scene->getSceneBounds().extent()); }
    constexpr uint GetTotalCausticClusterCount() const { return m_CausticClusterCount * m_CausticGeometryInstanceIDs.size(); }
    constexpr float GetPositionScaling() const { return m_Cb / GetSceneSize(); }

    void GenerateCausticPoints(
        RenderContext* pRenderContext,
        const uint geometryInstanceID,
        const std::span<PackedStaticVertexData>& vertexData,
        const std::span<uint32_t>& indexData
    );
    void HandleCausticClusterCollection(
        RenderContext* pRenderContext,
        std::vector<std::vector<float>>& pSigmaSortBuffers,
        const std::vector<FirstHitPhotonInfo>& firstHitPhotonInfos,
        const std::vector<uint>& firstHitCollectionCounts,
        const size_t firstHitPhotonCount
    );
    void GenerateFirstHitClusters(
        RenderContext* pRenderContext,
        const std::vector<FirstHitPhotonInfo>& firstHitPhotonInfos,
        const std::vector<uint>& firstHitCollectionCounts,
        const size_t firstHitPhotonCount,
        std::vector<std::vector<float3>>& lightFirstHitClusterPos
    );
};
