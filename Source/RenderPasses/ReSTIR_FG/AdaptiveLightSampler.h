#pragma once

#include <Falcor.h>
#include <Rendering/Lights/LightBVHBuilder.h>
#include <Utils/Math/ScalarTypes.h>
#include <span>

using namespace Falcor;

class AdaptiveLightSampler
{
public:
    AdaptiveLightSampler(ref<Device> device);

    void SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene);
    void PrepareBuffers(RenderContext* renderContext, const uint2 screenSize, const uint2 photonCounts);
    bool RenderUI(Gui::Widgets& widget);

    void Run(RenderContext* pRenderContext);

    constexpr bool IsActive() const { return m_Active; }

    void SetGeneratePhotonsVars(const ShaderVar& var) const;
    void SetCollectPhotonsVars(const ShaderVar& var) const;
    void ClearLeafSampleCountBuf(RenderContext* pRenderContext) const;

private:
    class LightTree : public LightBVH
    {
    public:
        LightTree(ref<Device> pDevice, const ref<const LightCollection>& pLightCollection) : LightBVH(pDevice, pLightCollection) {}

        void UpdateNodeImportance(const std::span<uint>& leafSampleCount, std::span<float>& nodeImportance, const uint nodeIdx);

        bool IsLeaf(const uint nodeIdx) const;
        uint GetRightChildIdx(const uint nodeIdx) const;
    };

    // Falcor objects
    ref<Device> m_Device;
    LightBVHBuilder m_LightBvhBuilder;
    LightTree m_LightBvh;

    // Falcor buffers
    ref<Buffer> m_ClusterNodeIdxBuf; // Maps cluster to node index in light tree
    ref<Buffer> m_ClusterNodeIdxBufCPU;
    ref<Buffer> m_ClusterCdfBuf; // Cluster CDF for sampling
    ref<Buffer> m_ClusterCdfBufCPU;
    ref<Buffer> m_LeafSampleCountBuf; // Number of collected photons per leaf node
    ref<Buffer> m_LeafSampleCountBufCPU;
    ref<Buffer> m_NodeClusterMapBuf; // Maps node to cluster
    ref<Buffer> m_NodeClusterMapBufCPU;
    ref<Buffer> m_PhotonLeafMapBuf[2]; // One for global photons one for caustic photons
    ref<Buffer> m_NodeImportanceBuf; // For storing sample count per node (including child nodes)
    ref<Buffer> m_NodeImportanceBufCPU;

    // Stats
    bool m_HasLights = false;
    bool m_Active = false;
    bool m_Reset = false;
    uint m_MaxCutSize = 32;
    uint m_ClusterCount = 1;
    uint m_GlobalPhotonWeight = 0;
    uint m_CausticPhotonWeight = 1;
    uint m_SplittingThreshold = 1000;
    float m_Epsilon = 10.0f;

    // Learning rate
    uint m_TimeStep = 1;
    static constexpr float m_Beta = 4.0f;
    static constexpr float m_Omega = 6.0f / 7.0f;

    // CPU Buffer
    std::vector<uint> m_ClusterNodeIndices;
    std::vector<float> m_ClusterImportance;

    size_t GetTotalNodeCount() const { return m_LightBvh.getStats().leafNodeCount + m_LightBvh.getStats().internalNodeCount; }
    float GetAlpha() const { return 1.0f / (m_Beta * math::pow(static_cast<float>(m_TimeStep), m_Omega)); }

    void UpdateClusterNodeIndices(RenderContext* pRenderContext);
    void UpdateNodeClusterMap(RenderContext* pRenderContext);
};
