#pragma once

#include <Falcor.h>
#include <Rendering/Lights/LightBVHBuilder.h>
#include <Utils/Math/ScalarTypes.h>

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
    void ClearClusterStatBuf(RenderContext* pRenderContext) const;
    void ClearLeafRadianceBuf(RenderContext* pRenderContext) const;
    
private:
    struct ClusterStats
    {
        uint count;
        float s1;
        float s2;
    };

    ref<Device> m_Device;
    LightBVHBuilder m_LightBvhBuilder;
    LightBVH m_LightBvh;

    ref<Buffer> m_ClusterNodeIdxBuf;
    ref<Buffer> m_ClusterNodeIdxBufCPU;
    ref<Buffer> m_ClusterCdfBuf;
    ref<Buffer> m_ClusterCdfBufCPU;
    ref<Buffer> m_ClusterStatsBuf;
    ref<Buffer> m_ClusterStatsBufCPU;
    ref<Buffer> m_LeafRadianceBuf;
    ref<Buffer> m_LeafRadianceBufCPU;
    ref<Buffer> m_NodeClusterMapBuf;
    ref<Buffer> m_NodeClusterMapBufCPU;
    ref<Buffer> m_PhotonLeafMapBuf[2]; // One for global photons one for caustic photons

    bool m_Active = false;
    uint m_MaxCutSize = 32;
    uint m_ClusterCount = 1;

    std::vector<ClusterStats> m_ClusterStats;

    size_t GetTotalNodeCount() const { return m_LightBvh.getStats().leafNodeCount + m_LightBvh.getStats().internalNodeCount; }

    void UpdateNodeClusterMap(RenderContext* pRenderContext);
};
