#include "AdaptiveLightSampler.h"
#include <execution>
#include "RandomGenerator.h"

AdaptiveLightSampler::AdaptiveLightSampler(ref<Device> device)
    : m_Device(device), m_LightBvh(device, {}), m_LightBvhBuilder(LightBVHBuilder::Options())
{}

void AdaptiveLightSampler::SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Reset buffers
    m_ClusterNodeIdxBuf.reset();
    m_ClusterCdfBuf.reset();
    m_ClusterSampleCountBuf.reset();
    m_ClusterRadianceSqBuf.reset();
    m_LeafRadianceBuf.reset();
    m_NodeClusterMapBuf.reset();
    m_PhotonLeafMapBuf[0].reset();
    m_PhotonLeafMapBuf[1].reset();
    m_NodeImportanceBuf.reset();

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

    // Init vector
    m_ClusterNodeIndices.resize(m_MaxCutSize);
    m_ClusterImportance.resize(m_MaxCutSize); // Q
    m_ClusterVariance.resize(m_MaxCutSize);

    // Init cluster importance
    std::fill(m_ClusterImportance.begin(), m_ClusterImportance.end(), 1.0f);

    // Reset alpha
    m_TimeStep;
}

void AdaptiveLightSampler::PrepareBuffers(RenderContext* pRenderContext, const uint2 screenSize, const uint2 photonCounts)
{
    if (!m_HasLights)
    {
        return;
    }

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

    if (!m_ClusterSampleCountBuf)
    {
        m_ClusterSampleCountBuf =
            Buffer::create(m_Device, sizeof(uint) * m_MaxCutSize, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        m_ClusterSampleCountBufCPU =
            Buffer::create(m_Device, sizeof(uint) * m_MaxCutSize, ResourceBindFlags::None, Buffer::CpuAccess::Read);
    }

    if (!m_ClusterRadianceSqBuf)
    {
        m_ClusterRadianceSqBuf = Buffer::create(m_Device, sizeof(float) * m_MaxCutSize);
        m_ClusterRadianceSqBufCPU =
            Buffer::create(m_Device, sizeof(float) * m_MaxCutSize, ResourceBindFlags::None, Buffer::CpuAccess::Read);
    }

    const size_t nodeCount = std::max<size_t>(1, GetTotalNodeCount());
    if (!m_LeafRadianceBuf)
    {
        // Allocate memory for all nodes even when only using leaf nodes because of simpler indexing
        m_LeafRadianceBuf =
            Buffer::create(m_Device, sizeof(float) * nodeCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
        m_LeafRadianceBufCPU = Buffer::create(m_Device, sizeof(float) * nodeCount, ResourceBindFlags::None, Buffer::CpuAccess::Read);
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
        if (group.button("Reset"))
        {
            m_TimeStep = 1;
            m_ClusterCount = 1;

            std::fill(m_ClusterNodeIndices.begin(), m_ClusterNodeIndices.end(), 0);
            std::fill(m_ClusterImportance.begin(), m_ClusterImportance.end(), 1.0f);
            std::fill(m_ClusterVariance.begin(), m_ClusterVariance.end(), 0.0f);

            m_ClusterNodeIdxBuf.reset();
            m_ClusterCdfBuf.reset();
            m_ClusterSampleCountBuf.reset();
            m_ClusterRadianceSqBuf.reset();
            m_LeafRadianceBuf.reset();
            m_NodeClusterMapBuf.reset();
            m_PhotonLeafMapBuf[0].reset();
            m_PhotonLeafMapBuf[1].reset();
            m_NodeImportanceBuf.reset();

            changed = true;
        }

        // Global and caustic weights
        changed |= group.var("Global photon weight", m_GlobalPhotonWeight, 0u, 1000u);
        changed |= group.var("Caustic photon weight", m_CausticPhotonWeight, 0u, 1000u);

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

    // Read radiance from leaf nodes to CPU
    pRenderContext->uavBarrier(m_LeafRadianceBuf.get());
    pRenderContext->copyBufferRegion(m_LeafRadianceBufCPU.get(), 0, m_LeafRadianceBuf.get(), 0, m_LeafRadianceBuf->getSize());
    const std::span<float> leafRadiance(reinterpret_cast<float*>(m_LeafRadianceBufCPU->map(Buffer::MapType::Read)), nodeCount);

    // Update node importance
    std::span<float> nodeImportance(reinterpret_cast<float*>(m_NodeImportanceBufCPU->map(Buffer::MapType::WriteDiscard)), nodeCount);
    std::for_each(
        std::execution::par_unseq, m_ClusterNodeIndices.begin(), m_ClusterNodeIndices.begin() + m_ClusterCount,
        [&](const uint clusterNodeIdx) { m_LightBvh.UpdateNodeImportance(leafRadiance, nodeImportance, clusterNodeIdx); }
    );
    pRenderContext->copyBufferRegion(m_NodeImportanceBuf.get(), 0, m_NodeClusterMapBufCPU.get(), 0, m_NodeImportanceBufCPU->getSize());

    // Read cluster stats from gpu
    pRenderContext->uavBarrier(m_ClusterSampleCountBuf.get());
    pRenderContext->copyBufferRegion(
        m_ClusterSampleCountBufCPU.get(), 0, m_ClusterSampleCountBuf.get(), 0, m_ClusterSampleCountBuf->getSize()
    );
    const std::span<uint> clusterSampleCounts(reinterpret_cast<uint*>(m_ClusterSampleCountBufCPU->map(Buffer::MapType::Read)), m_ClusterCount);
    pRenderContext->uavBarrier(m_ClusterRadianceSqBuf.get());
    pRenderContext->copyBufferRegion(
        m_ClusterRadianceSqBufCPU.get(), 0, m_ClusterRadianceSqBuf.get(), 0, m_ClusterRadianceSqBuf->getSize()
    );
    const std::span<float> clusterRadianceSq(
        reinterpret_cast<float*>(m_ClusterRadianceSqBufCPU->map(Buffer::MapType::Read)), m_ClusterCount
    );

    // Update cluster importance
    const float alpha = GetAlpha();
    for (uint clusterIdx = 0; clusterIdx < m_ClusterCount; ++clusterIdx)
    {
        const float oldImportance = m_ClusterImportance[clusterIdx];
        m_ClusterImportance[clusterIdx] = ((1.0f - alpha) * oldImportance) + (alpha * nodeImportance[m_ClusterNodeIndices[clusterIdx]]);
    }

    // Clustering
    float varianceSum = 1e-6f;
    uint newClusterCount = m_ClusterCount;
    for (uint clusterIdx = 0; clusterIdx < m_ClusterCount; ++clusterIdx)
    {
        const float countF = static_cast<float>(math::max(1u, clusterSampleCounts[clusterIdx]));
        const float mean = nodeImportance[m_ClusterNodeIndices[clusterIdx]] / countF;
        const float expectedSquare = clusterRadianceSq[clusterIdx] / countF;
        const float variance = expectedSquare - (mean * mean);
        varianceSum += variance;
        m_ClusterVariance[clusterIdx] = variance;
    }
    for (uint clusterIdx = 0; clusterIdx < m_ClusterCount; ++clusterIdx)
    {
        const uint clusterNodeIdx = m_ClusterNodeIndices[clusterIdx];
        const float countF = static_cast<float>(math::max(1u, clusterSampleCounts[clusterIdx]));
        const float variance = m_ClusterVariance[clusterIdx];
        const float splitProb =
            (1.0f / (static_cast<float>(m_ClusterCount) * math::exp(-variance))) * (variance / varianceSum) * (1.0f - (1.0f / countF));
        if (newClusterCount < m_MaxCutSize and !m_LightBvh.IsLeaf(clusterNodeIdx) and RandomGenerator::Float() < splitProb)
        {
            // Calculate cluster stats
            const float oldClusterImportance = m_ClusterImportance[clusterIdx];
            const uint leftIdx = clusterNodeIdx + 1;
            const uint rightIdx = m_LightBvh.GetRightChildIdx(clusterNodeIdx);
            const float leftRadiance = nodeImportance[leftIdx];
            const float rightRadiance = nodeImportance[rightIdx];
            const float totalRadiance = leftRadiance + rightRadiance;
            const float leftA = math::pow(1.0f - alpha, countF * leftRadiance / totalRadiance);
            const float rightA = math::pow(1.0f - alpha, countF * rightRadiance / totalRadiance);

            // Left cluster at parent cluster index
            m_ClusterNodeIndices[clusterIdx] = leftIdx;
            m_ClusterImportance[clusterIdx] = (leftA * leftRadiance) + ((1.0f - leftA) * oldClusterImportance);

            // Append right cluster at the end
            m_ClusterNodeIndices[newClusterCount] = rightIdx;
            m_ClusterImportance[newClusterCount] = (rightA * rightRadiance) + ((1.0f - rightA) * oldClusterImportance);

            // Increase cluster count
            ++newClusterCount;
        }
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
    m_ClusterRadianceSqBufCPU->unmap();
    m_ClusterSampleCountBufCPU->unmap();
    m_NodeImportanceBufCPU->unmap();
    m_LeafRadianceBufCPU->unmap();

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
}

void AdaptiveLightSampler::SetCollectPhotonsVars(const ShaderVar& var) const
{
    var["gClusterSampleCount"] = m_ClusterSampleCountBuf;
    var["gClusterRadianceSq"] = m_ClusterRadianceSqBuf;
    var["gLeafRadiance"] = m_LeafRadianceBuf;
    var["gNodeClusterMap"] = m_NodeClusterMapBuf;
    var["gPhotonLeafMap"][0ull] = m_PhotonLeafMapBuf[0];
    var["gPhotonLeafMap"][1ull] = m_PhotonLeafMapBuf[1];

    var["AdaptiveLightSampler"]["gAlsGlobalPhotonWeight"] = m_GlobalPhotonWeight;
    var["AdaptiveLightSampler"]["gAlsCausticPhotonWeight"] = m_CausticPhotonWeight;
}

void AdaptiveLightSampler::ClearClusterStatBuf(RenderContext* pRenderContext) const
{
    if (!m_HasLights)
    {
        return;
    }
    pRenderContext->clearUAV(m_ClusterSampleCountBuf->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(m_ClusterRadianceSqBuf->getUAV().get(), float4(0.0f));
}

void AdaptiveLightSampler::ClearLeafRadianceBuf(RenderContext* pRenderContext) const
{
    if (!m_HasLights)
    {
        return;
    }
    pRenderContext->clearUAV(m_LeafRadianceBuf->getUAV().get(), float4(0.0f));
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
    const std::span<float>& leafRadiance, std::span<float>& nodeImportance, const uint nodeIdx)
{
    if (mNodes.empty())
    {
        return;
    }

    FALCOR_ASSERT(leafRadiance.size() == mNodes.size());
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
            FALCOR_ASSERT(nodeIndex < leafRadiance.size());
            FALCOR_ASSERT(nodeIndex < nodeImportance.size());
            nodeImportance[nodeIndex] = leafRadiance[nodeIndex];
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
