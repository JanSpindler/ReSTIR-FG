#pragma once

#include <Falcor.h>
#include <Utils/CudaUtils.h>
#include <Utils/Math/VectorMath.h>

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

    GaussianPhotonGuiding(ref<Device> device) : m_Device(device) {}

    void SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene);
    void PrepareBuffers(const uint2 screenSize, RenderContext* renderContext, const uint2 maxPhotonCount);
    bool RenderUI(Gui::Widgets& widget);

    void GenerateCausticClusters(RenderContext* renderContext);
    void CountCausticClustersPass(RenderContext* renderContext);
    void CalculateGaussianGradientCuda(RenderContext* renderContext);
    void OptimizeGaussiansPass(RenderContext* renderContext);
    void CalculateSoftmaxWeightsPass(RenderContext* renderContext);

private:
    // Constants
    static inline const std::string m_ShaderModel = "6_5";

    static inline const std::string m_CalculateGaussainGradiantShader = "RenderPasses/ReSTIR_FG/Shader/CalculateGaussianGradient.cs.slang";
    static inline const std::string m_OptimizeGaussiansShader = "RenderPasses/ReSTIR_FG/Shader/OptimizeGaussians.cs.slang";
    static inline const std::string m_CalculateSoftmaxWeightsShader = "RenderPasses/ReSTIR_FG/Shader/CalculateSoftmaxWeights.cs.slang";
    static inline const std::string m_CountCausticClustersShader = "RenderPasses/ReSTIR_FG/Shader/CountCausticClusters.cs.slang";

    static inline const Gui::DropdownList m_OptimizerList{{static_cast<uint>(Optimizer::SGD), "SGD"}, {static_cast<uint>(Optimizer::Adam), "Adam"}};
    static inline const Gui::DropdownList m_InitializationList{
        {static_cast<uint>(Initialization::Random), "Random"},
        {static_cast<uint>(Initialization::Robust), "Robust"}
    };

    //
    ref<Device> m_Device;

    // General
    bool m_Active = false;
    bool m_CopyToCPU = false;
    uint m_GaussianCount = 16;
    uint m_MaxFirstHitPhotonCount = 100000;
    uint m_ActualFirstHitPhotonCount = 0;

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
    uint m_GenCausticPointCount = 10000;
    std::vector<float3> m_CausticClusters;

    // Optimization
    Optimizer m_Optimizer = Optimizer::Adam;
    float m_LearningRate = 0.01f;
    float m_Beta1 = 0.9f;
    float m_Beta2 = 0.999f;
    uint m_OptimStep = 0;

    // Buffers
    InteropBuffer m_GaussianBuf;
    ref<Texture> m_GaussianTex;
    InteropBuffer m_FirstHitPhotonCountBuf;
    InteropBuffer m_FirstHitPhotonInfoBuf;
    InteropBuffer m_FirstHitCollectionCountsBuf;
    std::array<ref<Buffer>, 2> m_PhotonFirstHitMapBufs;
    ref<Buffer> m_LightFirstHitCountsBuf;
    InteropBuffer m_GradientBuf;
    ref<Buffer> m_OptimizationBuf;
    InteropBuffer m_SoftmaxBuf;
    ref<Buffer> m_CausticClusterBuf;
    ref<Buffer> m_CausticClusterCountsBuf;

    // Passes
    ref<ComputePass> m_CountCausticClustersPass;
    ref<ComputePass> m_OptimizeGaussiansPass;
    ref<ComputePass> m_CalculateSoftmaxPass;

    // Functions
    constexpr uint GetTotalLightCount() const { return m_AnalyticLightCount + m_GeometricLightCount; }
    constexpr uint GetTotalGaussianCount() const { return GetTotalLightCount() * m_GaussianCount; }
    constexpr float GetSceneSize() const { return math::length(m_Scene->getSceneBounds().extent()); }
    constexpr uint GetTotalCausticClusterCount() const { return m_CausticClusterCount * m_CausticGeometryInstanceIDs.size(); }
};
