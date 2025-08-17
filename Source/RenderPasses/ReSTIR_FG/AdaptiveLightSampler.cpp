#include "AdaptiveLightSampler.h"
#include <execution>
#include "RandomGenerator.h"

AdaptiveLightSampler::AdaptiveLightSampler(ref<Device> device)
    : m_Device(device), m_LightBvh(device, {}), m_LightBvhBuilder(LightBVHBuilder::Options())
{}

void AdaptiveLightSampler::SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Reset buffers
    m_Reset = true;

    // Build tree
    const auto lightCollection = pScene->getLightCollection(pRenderContext);
    m_LightBvh = LightTree(m_Device, lightCollection);
    m_HasLights = lightCollection->getTotalLightCount() > 0;
    if (!m_HasLights)
    {
        return;
    }
    m_LightBvhBuilder.build(pRenderContext, reinterpret_cast<LightBVH&>(m_LightBvh));
    FALCOR_ASSERT(m_LightBvh.isValid());

    // Reset alpha
    m_TimeStep = 1;
}

void AdaptiveLightSampler::PrepareBuffers(RenderContext* pRenderContext, const uint2 screenSize, const uint2 photonCounts)
{
    // Do not run if no lights are present
    if (!m_HasLights)
    {
        return;
    }

    // Check if reset is requested
    if (m_Reset)
    {
        m_TimeStep = 1;
        m_ClusterCount = 1;

        m_ClusterNodeIndices.resize(m_MaxCutSize);
        m_ClusterImportance.resize(m_MaxCutSize);

        std::fill(m_ClusterNodeIndices.begin(), m_ClusterNodeIndices.end(), 0);
        std::fill(m_ClusterImportance.begin(), m_ClusterImportance.end(), 1.0f);

        m_ClusterNodeIdxBuf.reset();
        m_ClusterCdfBuf.reset();
        m_LeafSampleCountBuf.reset();
        m_NodeClusterMapBuf.reset();
        m_PhotonLeafMapBuf[0].reset();
        m_PhotonLeafMapBuf[1].reset();
        m_NodeImportanceBuf.reset();

        m_Reset = false;
    }

    // Prepare buffers
    if (!m_ClusterNodeIdxBuf)
    {
        const std::vector<uint> clusterNodeIndices(m_MaxCutSize, 0);
        m_ClusterNodeIdxBuf = Buffer::create(
            m_Device, m_MaxCutSize * sizeof(uint), ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, clusterNodeIndices.data()
        );
        m_ClusterNodeIdxBufCPU = Buffer::create(m_Device, m_MaxCutSize * sizeof(uint), ResourceBindFlags::None, Buffer::CpuAccess::Write);
    }

    if (!m_ClusterCdfBuf)
    {
        std::vector<float> clusterCDF(m_MaxCutSize, 0.0f);
        for (size_t i = 0; i < m_ClusterCount; ++i)
        {
            clusterCDF[i] = static_cast<float>(i + 1) / static_cast<float>(m_ClusterCount);
        }
        m_ClusterCdfBuf = Buffer::create(
            m_Device, sizeof(float) * m_MaxCutSize, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, clusterCDF.data()
        );
        m_ClusterCdfBufCPU = Buffer::create(m_Device, sizeof(float) * m_MaxCutSize, ResourceBindFlags::None, Buffer::CpuAccess::Write);
    }

    const size_t nodeCount = std::max<size_t>(1, GetTotalNodeCount());
    if (!m_LeafSampleCountBuf)
    {
        // Allocate memory for all nodes even when only using leaf nodes because of simpler indexing
        m_LeafSampleCountBuf =
            Buffer::create(m_Device, sizeof(float) * nodeCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
        m_LeafSampleCountBufCPU = Buffer::create(m_Device, sizeof(float) * nodeCount, ResourceBindFlags::None, Buffer::CpuAccess::Read);
    }

    if (!m_NodeClusterMapBuf)
    {
        m_NodeClusterMapBuf =
            Buffer::create(m_Device, sizeof(uint) * nodeCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
        m_NodeClusterMapBufCPU = Buffer::create(m_Device, sizeof(uint) * nodeCount, ResourceBindFlags::None, Buffer::CpuAccess::Write);
        UpdateNodeClusterMap(pRenderContext); // Assumes SetScene was already executed
    }

    for (size_t idx = 0; idx < 2; ++idx)
    {
        if (!m_PhotonLeafMapBuf[idx])
        {
            m_PhotonLeafMapBuf[idx] = Buffer::create(
                m_Device, sizeof(uint) * photonCounts[idx], ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
            );
        }
    }

    if (!m_NodeImportanceBuf)
    {
        const std::vector<float> nodeImportance(nodeCount, 1.0f);
        m_NodeImportanceBuf = Buffer::create(
            m_Device, sizeof(float) * nodeCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nodeImportance.data()
        );
        m_NodeImportanceBufCPU = Buffer::create(m_Device, sizeof(float) * nodeCount, ResourceBindFlags::None, Buffer::CpuAccess::Write);
    }
}

bool AdaptiveLightSampler::RenderUI(Gui::Widgets& widget)
{
    bool changed = false;

    if (auto group = widget.group("Adaptive Light Sampler"))
    {
        // Active
        changed |= group.checkbox("Adaptive Light Sampler", m_Active);

        // Reset button
        if (group.button("Reset") or group.var("Max Cut Size", m_MaxCutSize, 1u, 1024u))
        {
            m_Reset = true;
            changed = true;
        }

        // Global and caustic weights
        changed |= group.var("Global photon weight", m_GlobalPhotonWeight, 0u, 1000u);
        changed |= group.var("Caustic photon weight", m_CausticPhotonWeight, 0u, 1000u);
        changed |= group.var("Splitting threshold", m_SplittingThreshold);
        changed |= group.var("Epsilon", m_Epsilon, 0.0f);

        // Adaptive light sampler stats
        group.text("Time step: " + std::to_string(m_TimeStep));
        group.text("Cluster count: " + std::to_string(m_ClusterCount));

        // TODO: Rebuild if builder params changed?
        if (group.group("Light BVH Builder"))
        {
            m_LightBvhBuilder.renderUI(widget);
        }
        if (group.group("Light BVH"))
        {
            m_LightBvh.renderUI(widget);
        }
    }

    return changed;
}

void AdaptiveLightSampler::Run(RenderContext* pRenderContext)
{
    //
    FALCOR_PROFILE(pRenderContext, "AdaptiveLightSampler CPU");
    if (!m_HasLights)
    {
        return;
    }

    //
    const size_t nodeCount = GetTotalNodeCount();

    // Read leaf sample counts from GPu
    pRenderContext->uavBarrier(m_LeafSampleCountBuf.get());
    pRenderContext->copyBufferRegion(
        m_LeafSampleCountBufCPU.get(), 0, m_LeafSampleCountBuf.get(), 0, m_LeafSampleCountBuf->getSize());
    const std::span<uint> leafSampleCounts(reinterpret_cast<uint*>(m_LeafSampleCountBufCPU->map(Buffer::MapType::Read)), nodeCount);

    // Update node importance
    std::span<float> nodeImportance(reinterpret_cast<float*>(m_NodeImportanceBufCPU->map(Buffer::MapType::WriteDiscard)), nodeCount);
    std::for_each(
        std::execution::par_unseq, m_ClusterNodeIndices.begin(), m_ClusterNodeIndices.begin() + m_ClusterCount,
        [&](const uint clusterNodeIdx) { m_LightBvh.UpdateNodeImportance(leafSampleCounts, nodeImportance, clusterNodeIdx); }
    );
    pRenderContext->copyBufferRegion(m_NodeImportanceBuf.get(), 0, m_NodeClusterMapBufCPU.get(), 0, m_NodeImportanceBufCPU->getSize());

    // Update cluster importance
    const float alpha = GetAlpha();
    for (uint clusterIdx = 0; clusterIdx < m_ClusterCount; ++clusterIdx)
    {
        const float oldImportance = m_ClusterImportance[clusterIdx];
        m_ClusterImportance[clusterIdx] = ((1.0f - alpha) * oldImportance) + (alpha * nodeImportance[m_ClusterNodeIndices[clusterIdx]]);
    }

    // Select cluster with most samples for splitting
    uint maxSampleCount = 0;
    uint splitClusterIdx = std::numeric_limits<uint>::max();
    for (uint clusterIdx = 0; clusterIdx < m_ClusterCount; ++clusterIdx)
    {
        const uint sampleCount = m_ClusterImportance[clusterIdx];
        if (sampleCount > math::max(maxSampleCount, m_SplittingThreshold))
        {
            maxSampleCount = sampleCount;
            splitClusterIdx = clusterIdx;
        }
    }

    // Split cluster
    uint newClusterCount = m_ClusterCount;
    if (m_ClusterCount < m_MaxCutSize and
        splitClusterIdx != std::numeric_limits<uint>::max() and
        !m_LightBvh.IsLeaf(m_ClusterNodeIndices[splitClusterIdx]))
    {
        const float oldClusterImportance = m_ClusterImportance[splitClusterIdx];
        const uint clusterNodeIdx = m_ClusterNodeIndices[splitClusterIdx];
        const uint leftIdx = clusterNodeIdx + 1;
        const uint rightIdx = m_LightBvh.GetRightChildIdx(clusterNodeIdx);

        m_ClusterNodeIndices[splitClusterIdx] = leftIdx;
        m_ClusterImportance[splitClusterIdx] = oldClusterImportance / 2.0f;

        m_ClusterNodeIndices[newClusterCount] = rightIdx;
        m_ClusterImportance[newClusterCount] = oldClusterImportance / 2.0f;

        ++newClusterCount;
    }

    // Update cdf
    float clusterImportanceSum = 0.0f;
    for (uint clusterIdx = 0; clusterIdx < newClusterCount; ++clusterIdx)
    {
        clusterImportanceSum += m_ClusterImportance[clusterIdx];
    }
    std::span<float> clusterCdf(reinterpret_cast<float*>(m_ClusterCdfBufCPU->map(Buffer::MapType::WriteDiscard)), newClusterCount);
    float cdf = 0.0;
    for (uint clusterIdx = 0; clusterIdx < newClusterCount; ++clusterIdx)
    {
        cdf += m_ClusterImportance[clusterIdx] / clusterImportanceSum;
        clusterCdf[clusterIdx] = cdf;
    }
    pRenderContext->copyBufferRegion(m_ClusterCdfBuf.get(), 0, m_ClusterCdfBufCPU.get(), 0, m_ClusterCdfBufCPU->getSize());

    // Unmap buffers
    m_ClusterCdfBufCPU->unmap();
    m_NodeImportanceBufCPU->unmap();
    m_LeafSampleCountBufCPU->unmap();

    // If clustering changed, update leaf cluster map
    if (newClusterCount != m_ClusterCount)
    {
        m_ClusterCount = newClusterCount;
        UpdateClusterNodeIndices(pRenderContext);
        UpdateNodeClusterMap(pRenderContext);
    }

    // Update time step
    ++m_TimeStep;
}

void AdaptiveLightSampler::SetGeneratePhotonsVars(const ShaderVar& var) const
{
    m_LightBvh.setShaderData(var["gLightBVH"]);
    var["gLightClusterNodeIndices"] = m_ClusterNodeIdxBuf;
    var["gLightClusterCdf"] = m_ClusterCdfBuf;
    var["gPhotonLeafMap"][0ull] = m_PhotonLeafMapBuf[0];
    var["gPhotonLeafMap"][1ull] = m_PhotonLeafMapBuf[1];
    var["gNodeImportance"] = m_NodeImportanceBuf;
    var["AdaptiveLightSampler"]["gLightClusterCount"] = m_ClusterCount;
    var["AdaptiveLightSampler"]["gEpsilon"] = m_Epsilon;
}

void AdaptiveLightSampler::SetCollectPhotonsVars(const ShaderVar& var) const
{
    var["gLeafSampleCount"] = m_LeafSampleCountBuf;
    var["gPhotonLeafMap"][0ull] = m_PhotonLeafMapBuf[0];
    var["gPhotonLeafMap"][1ull] = m_PhotonLeafMapBuf[1];

    var["AdaptiveLightSampler"]["gAlsGlobalPhotonWeight"] = m_GlobalPhotonWeight;
    var["AdaptiveLightSampler"]["gAlsCausticPhotonWeight"] = m_CausticPhotonWeight;
}

void AdaptiveLightSampler::ClearLeafSampleCountBuf(RenderContext* pRenderContext) const
{
    if (!m_HasLights)
    {
        return;
    }
    pRenderContext->clearUAV(m_LeafSampleCountBuf->getUAV().get(), uint4(0));
}

void AdaptiveLightSampler::UpdateClusterNodeIndices(RenderContext* pRenderContext)
{
    std::span<uint> clusterNodeIndices(reinterpret_cast<uint*>(m_ClusterNodeIdxBufCPU->map(Buffer::MapType::WriteDiscard)), m_MaxCutSize);
    std::copy(m_ClusterNodeIndices.begin(), m_ClusterNodeIndices.end(), clusterNodeIndices.begin());
    pRenderContext->copyBufferRegion(m_ClusterNodeIdxBuf.get(), 0, m_ClusterNodeIdxBufCPU.get(), 0, m_ClusterNodeIdxBufCPU->getSize());
    m_ClusterNodeIdxBufCPU->unmap();
}

void AdaptiveLightSampler::UpdateNodeClusterMap(RenderContext* pRenderContext)
{
    // Map cpu buffer
    std::span<uint> nodeClusterMap(
        reinterpret_cast<uint*>(m_NodeClusterMapBufCPU->map(Buffer::MapType::WriteDiscard)), GetTotalNodeCount()
    );

    // Set all indices to default invalid (0xFFFFFFFF)
    const LightBVH::NodeFunction setInvalid = [&](const LightBVH::NodeLocation& nodeLoc)
    {
        nodeClusterMap[nodeLoc.nodeIndex] = std::numeric_limits<uint>::max();
        return true;
    };
    m_LightBvh.traverseBVH(setInvalid, setInvalid, 0);

    // For each cluster set the corresponding reference of the children
    for (size_t clusterIdx = 0; clusterIdx < m_ClusterCount; ++clusterIdx)
    {
        const LightBVH::NodeFunction setClusterMap = [&](const LightBVH::NodeLocation& nodeLoc)
        {
            nodeClusterMap[nodeLoc.nodeIndex] = clusterIdx;
            return true;
        };
        m_LightBvh.traverseBVH(setClusterMap, setClusterMap, clusterIdx);
    }

    // Copy to gpu buffer
    pRenderContext->copyBufferRegion(m_NodeClusterMapBuf.get(), 0, m_NodeClusterMapBufCPU.get(), 0, nodeClusterMap.size_bytes());

    // Unmap cpu buffer
    m_NodeClusterMapBufCPU->unmap();
}

void AdaptiveLightSampler::LightTree::UpdateNodeImportance(
    const std::span<uint>& leafSampleCount, std::span<float>& nodeImportance, const uint nodeIdx)
{
    if (mNodes.empty())
    {
        return;
    }

    FALCOR_ASSERT(leafSampleCount.size() == mNodes.size());
    FALCOR_ASSERT(nodeImportance.size() == mNodes.size());

    // Recursive lambda for post-order traversal.
    // Captures 'this' to access mNodes, and spans by reference.
    // The 'auto& self' parameter is a common pattern for recursive lambdas.
    auto calculateImportanceRecursive =
        [&](auto& self, uint32_t nodeIndex) -> float
    {
        FALCOR_ASSERT(nodeIndex < mNodes.size());
        const PackedNode& currentNode = mNodes[nodeIndex];

        if (currentNode.isLeaf())
        {
            // For leaf nodes, importance is its radiance.
            // leafRadiance and nodeImportance are indexed by nodeIndex.
            FALCOR_ASSERT(nodeIndex < leafSampleCount.size());
            FALCOR_ASSERT(nodeIndex < nodeImportance.size());
            nodeImportance[nodeIndex] = static_cast<float>(leafSampleCount[nodeIndex]);
            return nodeImportance[nodeIndex];
        }
        else
        {
            // Internal node. Calculate importance as the sum of children's importances.
            // Left child is always at nodeIndex + 1.
            uint32_t leftChildIndex = nodeIndex + 1;
            FALCOR_ASSERT(leftChildIndex < mNodes.size()); // Should hold for a valid BVH
            float leftImportance = self(self, leftChildIndex);

            // Right child offset is relative to current node's index.
            uint32_t rightChildIndex = currentNode.getInternalNode().rightChildIdx;
            FALCOR_ASSERT(rightChildIndex < mNodes.size()); // Should hold for a valid BVH
            float rightImportance = self(self, rightChildIndex);

            FALCOR_ASSERT(nodeIndex < nodeImportance.size());
            nodeImportance[nodeIndex] = leftImportance + rightImportance;
            return nodeImportance[nodeIndex];
        }
    };

    // Start recursion from the root node (index 0).
    // If mNodes is not empty, node 0 must exist.
    calculateImportanceRecursive(calculateImportanceRecursive, nodeIdx);
}

bool AdaptiveLightSampler::LightTree::IsLeaf(const uint nodeIdx) const
{
    return mNodes[nodeIdx].isLeaf();
}

uint AdaptiveLightSampler::LightTree::GetRightChildIdx(const uint nodeIdx) const
{
    FALCOR_ASSERT(!mNodes[nodeIdx].isLeaf());
    return mNodes[nodeIdx].getInternalNode().rightChildIdx;
}
