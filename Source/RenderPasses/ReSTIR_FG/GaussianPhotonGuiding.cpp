#include "GaussianPhotonGuiding.h"
#include "Gaussian3D.h"
#include "RandomGenerator.h"
#include "calc_gradient.h"
#include "dkm/dkm_parallel.hpp"
#include <execution>

static InteropBuffer CreateInteropBuffer(
    const ref<Device> pDevice,
    const size_t size,
    const Falcor::Buffer::CpuAccess cpuAccess = Falcor::Buffer::CpuAccess::None,
    const void* data = nullptr
)
{
    InteropBuffer interop;

    // Create a new DX <-> CUDA shared buffer using the Falcor API to create, then find its CUDA pointer.
    interop.buffer = Buffer::create(
        pDevice, size, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::Shared, cpuAccess,
        data
    );
    interop.devicePtr = (CUdeviceptr)getSharedDevicePtr(interop.buffer->getSharedApiHandle(), (uint32_t)interop.buffer->getSize());
    checkInvariant(interop.devicePtr != (CUdeviceptr)0, "Failed to create CUDA device ptr for buffer");

    return interop;
}

template<typename T>
static InteropBuffer CreateStructuredInteropBuffer(
    const ref<Device> pDevice,
    const size_t count,
    const Falcor::Buffer::CpuAccess cpuAccess = Falcor::Buffer::CpuAccess::None,
    const void* data = nullptr
)
{
    InteropBuffer interop;

    // Create a new DX <-> CUDA shared buffer using the Falcor API to create, then find its CUDA pointer.
    interop.buffer = Buffer::createStructured(
        pDevice, sizeof(T), count, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::Shared,
        cpuAccess, data
    );
    interop.devicePtr = (CUdeviceptr)getSharedDevicePtr(interop.buffer->getSharedApiHandle(), (uint32_t)interop.buffer->getSize());
    checkInvariant(interop.devicePtr != (CUdeviceptr)0, "Failed to create CUDA device ptr for buffer");

    return interop;
}

void GaussianPhotonGuiding::SetScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Store ref to scene
    m_Scene = pScene;

    // Defines
    m_Defines.add(m_Scene->getSceneDefines());

    // 3D gaussian photon guiding
    m_AnalyticLightCount = pScene->getLightCount();
    m_GeometricLightCount = pScene->getLightCollection(pRenderContext)->getMeshLights().size();

    // Gaussian initialization
    m_CausticGeometryInstanceIDs.clear();
    for (size_t geomInstanceIdx = 0; geomInstanceIdx < pScene->getGeometryInstanceCount(); ++geomInstanceIdx)
    {
        const uint materialId = pScene->getGeometryInstance(geomInstanceIdx).materialID;
        const ref<Material> material = pScene->getMaterial(MaterialID(materialId));
        const ref<BasicMaterial> basicMaterial = material->toBasicMaterial();

        // occlusion(R), roughness(G), metallic(B)
        const float4 specularParams = basicMaterial->getSpecularParams();
        const float roughness = specularParams.g;
        const float metallic = specularParams.b;
        const bool deltaSpecular = material->getHeader().isDeltaSpecular();

        // Check if caustic caster
        if ((metallic > 0.0f) || (roughness < 0.2f) || deltaSpecular)
        {
            const uint geometryId = pScene->getGeometryInstance(geomInstanceIdx).geometryID;
            if (pScene->getGeometryType(GlobalGeometryID(geometryId)) == GeometryType::TriangleMesh)
            {
                m_CausticGeometryInstanceIDs.push_back(geomInstanceIdx);
            }
        }
    }
}

void GaussianPhotonGuiding::PrepareBuffers(const uint2 screenSize, RenderContext* renderContext, const uint2 maxPhotonCount)
{
    const size_t lightCount = GetTotalLightCount();
    const size_t totalGaussianCount = GetTotalGaussianCount();
    if (!m_GaussianBuf.buffer)
    {
        // Init gaussians
        std::vector<Gaussian3D> gaussians(totalGaussianCount);

        // Init N random gaussians in the scene
        const float positionScaling = GetPositionScaling();
        for (size_t gaussIdx = 0; gaussIdx < totalGaussianCount; ++gaussIdx)
        {
            gaussians[gaussIdx] =
                Gaussian3D(RandomGenerator::AabbPoint(m_Scene->getSceneBounds()) * positionScaling, RandomGenerator::Float(), 0.0f);
        }

        // Create buffer
        m_GaussianBuf = CreateStructuredInteropBuffer<Gaussian3D>(m_Device, totalGaussianCount, Buffer::CpuAccess::None, gaussians.data());
        m_GaussianBufReadCPU =
            Buffer::createStructured(m_Device, sizeof(Gaussian3D), totalGaussianCount, ResourceBindFlags::None, Buffer::CpuAccess::Read);
        m_GaussianBufWriteCPU =
            Buffer::createStructured(m_Device, sizeof(Gaussian3D), totalGaussianCount, ResourceBindFlags::None, Buffer::CpuAccess::Write);

        // Reset optimization
        m_OptimizationBuf.reset();
        m_FrameCountAfterOptimReset = 0;
    }

    if (!m_FirstHitPhotonCountBuf.buffer)
    {
        m_FirstHitPhotonCountBuf = CreateInteropBuffer(m_Device, sizeof(uint));
        m_FirstHitPhotonCountBufCPU = Buffer::create(m_Device, sizeof(uint), ResourceBindFlags::None, Buffer::CpuAccess::Read);
    }

    if (!m_FirstHitPhotonInfoBuf.buffer)
    {
        m_FirstHitPhotonInfoBuf = CreateStructuredInteropBuffer<FirstHitPhotonInfo>(m_Device, m_MaxFirstHitPhotonCount);
        m_FirstHitPhotonInfoBufCPU = Buffer::createStructured(
            m_Device, sizeof(FirstHitPhotonInfo), m_MaxFirstHitPhotonCount, ResourceBindFlags::None, Buffer::CpuAccess::Read
        );
    }

    if (!m_FirstHitCollectionCountsBuf.buffer)
    {
        m_FirstHitCollectionCountsBuf = CreateInteropBuffer(m_Device, sizeof(uint) * m_MaxFirstHitPhotonCount);
        m_FirstHitCollectionCountsBufCPU =
            Buffer::create(m_Device, sizeof(uint) * m_MaxFirstHitPhotonCount, ResourceBindFlags::None, Buffer::CpuAccess::Read);
    }

    for (size_t idx = 0; idx < 2; ++idx)
    {
        if (!m_PhotonFirstHitMapBufs[idx])
        {
            m_PhotonFirstHitMapBufs[idx] = Buffer::create(
                m_Device, sizeof(uint) * maxPhotonCount[idx], ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
            );
        }
    }

    if (!m_LightFirstHitCountsBuf)
    {
        m_LightFirstHitCountsBuf =
            Buffer::create(m_Device, sizeof(uint) * lightCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    }

    if (!m_GradientBuf.buffer)
    {
        m_GradientBuf = CreateStructuredInteropBuffer<Gaussian3D>(m_Device, totalGaussianCount);
    }

    if (!m_OptimizationBuf)
    {
        const std::vector<Gaussian3DOptimizationData> gaussianOptimizationData(totalGaussianCount, Gaussian3DOptimizationData());
        m_OptimizationBuf = Buffer::createStructured(
            m_Device, sizeof(Gaussian3DOptimizationData), totalGaussianCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, gaussianOptimizationData.data()
        );

        const std::vector<uint> optimizationReset(totalGaussianCount, 0);
        m_OptimizationResetBuf = Buffer::create(
            m_Device, sizeof(uint) * totalGaussianCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, optimizationReset.data()
        );
        m_OptimizationResetBufWriteCPU =
            Buffer::create(m_Device, sizeof(uint) * totalGaussianCount, ResourceBindFlags::None, Buffer::CpuAccess::Write);

        // Reset optimization
        m_FrameCountAfterOptimReset = 0;
    }

    if (!m_SoftmaxBuf.buffer)
    {
        const std::vector<float> softmaxWeights(totalGaussianCount, 1.0f / static_cast<float>(m_GaussianCount));
        m_SoftmaxBuf = CreateInteropBuffer(m_Device, sizeof(float) * totalGaussianCount, Buffer::CpuAccess::None, softmaxWeights.data());
    }
}

bool GaussianPhotonGuiding::RenderUI(Gui::Widgets& widget)
{
    bool changed = false;

    if (auto group = widget.group("PhotonGuiding"))
    {
        changed |= group.checkbox("Use 3D Gaussian Photon Guiding", m_Active);
        group.tooltip("Use 3D Gaussian Photon Guiding for the final gather pass");

        if (group.var("Gaussians per Light", m_GaussianCount))
        {
            m_GaussianBuf.buffer.reset();
            m_GradientBuf.buffer.reset();
            m_OptimizationBuf.reset();
            m_LightFirstHitCountsBuf.reset();
            m_SoftmaxBuf.buffer.reset();

            m_GaussianCount = math::max<uint>(m_GaussianCount, 1);
            changed = true;
        }

        if (group.button("ReInit Gaussians"))
        {
            m_GaussianBuf.buffer.reset();
            m_OptimizationBuf.reset();
            m_SoftmaxBuf.buffer.reset();
            changed = true;
        }

        changed |= group.var("Minimum GMM PDF", m_MinPdf, 0.0f, 1.0f);

        changed |= group.var("Beta (MIS)", m_Beta, 0.0f, 1.0f);

        changed |= group.var("Global Photon Weight", m_GlobalPhotonWeight, 0u, 1000u);
        changed |= group.var("Caustic Photon Weight", m_CausticPhotonWeight, 0u, 1000u);

        const bool rebuildFirstPhoton = group.var("Max First Hit Photons", m_MaxFirstHitPhotonCount);
        if (m_MaxFirstHitPhotonCount > 0)
        {
            changed |= rebuildFirstPhoton;
            if (rebuildFirstPhoton)
            {
                m_FirstHitPhotonInfoBuf.buffer.reset();
                m_FirstHitCollectionCountsBuf.buffer.reset();
            }
        }

        group.text(
            "First Hit Photons: " + std::to_string(m_ActualFirstHitPhotonCount) + " / " +
            std::to_string(m_MaxFirstHitPhotonCount) + " (" +
            std::to_string(static_cast<float>(m_ActualFirstHitPhotonCount) / static_cast<float>(m_MaxFirstHitPhotonCount)) +
            ")"
        );
        
        // Optimizer
        changed |= group.checkbox("Optimize Gaussians", m_Optimize);
        const bool changedOptimizer = group.dropdown("Optimizer", m_OptimizerList, reinterpret_cast<uint&>(m_Optimizer));
        if (changedOptimizer)
        {
            m_OptimizationBuf.reset();
        }
        changed |= changedOptimizer;

        changed |= group.var("Learning Rate", m_LearningRate, 0.0f, 1.0f);

        if (m_Optimizer == Optimizer::Adam)
        {
            changed |= group.var("Adam Beta1", m_Beta1, 0.0f, 1.0f);
            changed |= group.var("Adam Beta2", m_Beta2, 0.0f, 1.0f);
        }

        // Repulsion
        group.var("Repulsive Force", m_RepulsiveForce, 0.0f);
        group.var("Repulsive Distance", m_RepulsiveDistance, 0.01f);

        // Random replace
        changed |= group.checkbox("Enable Random Replace", m_RandomReplace);
        changed |= group.var("Random Replace Count", m_RandomReplaceCount, 0u, m_GaussianCount);
        changed |= group.var("Random Replace Frequency", m_RandomReplaceFrequency, 1u, 1024u);
    }

    return changed;
}

void GaussianPhotonGuiding::ResetPhotonFirstHitMap()
{
    m_PhotonFirstHitMapBufs[0].reset();
    m_PhotonFirstHitMapBufs[1].reset();
}

void GaussianPhotonGuiding::GenerateCausticClusters(RenderContext* renderContext)
{
    // Profile
    FALCOR_PROFILE(renderContext, "GenerateCausticPoints");

    // Early exit if no caustic geometry instances
    if (m_CausticGeometryInstanceIDs.empty())
    {
        m_CausticClustersPos.clear();
        return;
    }

    // Cache frequently used values
    const size_t totalCausticClusterCount = GetTotalCausticClusterCount();
    const size_t causticGeometryCount = m_CausticGeometryInstanceIDs.size();

    // Pre-allocate output buffer
    m_CausticClustersPos.resize(totalCausticClusterCount);

    // Get vertex and index buffer references
    ref<Vao> vao = m_Scene->getMeshVao();
    ref<Buffer> vbo = vao->getVertexBuffer(0);
    ref<Buffer> ibo = vao->getIndexBuffer();

    // Validate index buffer format
    if (vao->getIndexBufferFormat() != ResourceFormat::R32Uint)
    {
        throw RuntimeError("ReSTIR_FG: Only uint32 index buffer format is supported");
    }

    // Optimize buffer copying with single allocation and async operations
    const size_t indexByteSize = ibo->getSize();
    const size_t indexCount = indexByteSize / sizeof(uint32_t);
    const uint32_t vertexCount = vbo->getElementCount();

    // Create CPU buffers with proper alignment and sizing
    ref<Buffer> iboCpu = Buffer::create(m_Device, indexByteSize, ResourceBindFlags::None, Buffer::CpuAccess::Read);
    ref<Buffer> vboCpu =
        Buffer::createStructured(m_Device, sizeof(PackedStaticVertexData), vertexCount, ResourceBindFlags::None, Buffer::CpuAccess::Read);

    // Initiate both buffer copies concurrently
    renderContext->copyBufferRegion(iboCpu.get(), 0, ibo.get(), 0, indexByteSize);
    renderContext->copyBufferRegion(vboCpu.get(), 0, vbo.get(), 0, sizeof(PackedStaticVertexData) * vertexCount);

    // Single flush for both operations
    renderContext->flush(true);

    // Map buffers and create spans
    void* indexPtr = iboCpu->map(Buffer::MapType::Read);
    void* vertexPtr = vboCpu->map(Buffer::MapType::Read);

    const std::span<uint32_t> indexData(reinterpret_cast<uint32_t*>(indexPtr), indexCount);
    const std::span<PackedStaticVertexData> vertexData(reinterpret_cast<PackedStaticVertexData*>(vertexPtr), vertexCount);

    // Use parallel execution with proper load balancing
    std::vector<size_t> indices(causticGeometryCount);
    std::iota(indices.begin(), indices.end(), 0);

    // Determine optimal parallelization strategy
    const bool useParallel = causticGeometryCount > 2 && m_GenCausticPointCount > 100;

    if (useParallel)
    {
        // Use parallel execution for multiple geometry instances
        std::for_each(
            std::execution::par_unseq, indices.begin(), indices.end(),
            [&](const size_t geomInstanceIdx) { GenerateCausticPoints(geomInstanceIdx, vertexData, indexData); }
        );
    }
    else
    {
        // Use sequential execution for better cache locality with few instances
        for (size_t geomInstanceIdx = 0; geomInstanceIdx < causticGeometryCount; ++geomInstanceIdx)
        {
            GenerateCausticPoints(geomInstanceIdx, vertexData, indexData);
        }
    }

    // Unmap buffers in proper order
    vboCpu->unmap();
    iboCpu->unmap();
}

void GaussianPhotonGuiding::TrackActualFirstHitPhotonCount(RenderContext* renderContext)
{
    // Barrier
    renderContext->uavBarrier(m_FirstHitPhotonCountBuf.buffer.get());

    // Copy the first hit photon count to a CPU buffer
    renderContext->copyBufferRegion(m_FirstHitPhotonCountBufCPU.get(), 0, m_FirstHitPhotonCountBuf.buffer.get(), 0, sizeof(uint));
    void* data = m_FirstHitPhotonCountBufCPU->map(Buffer::MapType::Read);
    std::memcpy(&m_ActualFirstHitPhotonCount, data, sizeof(uint));
    m_FirstHitPhotonCountBufCPU->unmap();
}

void GaussianPhotonGuiding::RobustInitialization(RenderContext* pRenderContext)
{
    // Profile
    FALCOR_PROFILE(pRenderContext, "RobustInitialization");

    // Early exit check - moved earlier to avoid unnecessary work
    if (m_FrameCountAfterOptimReset < 1)
    {
        return;
    }

    // Get first hit photon information from GPU
    const size_t firstHitPhotonCount = std::min<size_t>(m_MaxFirstHitPhotonCount, m_ActualFirstHitPhotonCount);

    // Early exit if no photons to process
    if (firstHitPhotonCount == 0)
    {
        return;
    }

    // Cache frequently used values
    const size_t lightCount = GetTotalLightCount();
    const size_t clusterCount = GetTotalCausticClusterCount();
    const size_t totalGaussianCount = m_GaussianCount * lightCount;
    const float positionScaling = GetPositionScaling();

    // Optimize buffer operations with single allocation and RAII management
    std::vector<FirstHitPhotonInfo> firstHitPhotonInfos;
    std::vector<uint> firstHitCollectionCounts;
    firstHitPhotonInfos.reserve(firstHitPhotonCount);
    firstHitCollectionCounts.reserve(firstHitPhotonCount);

    // Copy photon info data more efficiently
    {
        pRenderContext->uavBarrier(m_FirstHitPhotonInfoBuf.buffer.get());
        pRenderContext->copyBufferRegion(
            m_FirstHitPhotonInfoBufCPU.get(), 0, m_FirstHitPhotonInfoBuf.buffer.get(), 0, sizeof(FirstHitPhotonInfo) * firstHitPhotonCount
        );

        void* photonInfoData = m_FirstHitPhotonInfoBufCPU->map(Buffer::MapType::Read);
        firstHitPhotonInfos.resize(firstHitPhotonCount);
        std::memcpy(firstHitPhotonInfos.data(), photonInfoData, sizeof(FirstHitPhotonInfo) * firstHitPhotonCount);
        m_FirstHitPhotonInfoBufCPU->unmap();
    }

    // Copy collection counts data
    {
        pRenderContext->uavBarrier(m_FirstHitCollectionCountsBuf.buffer.get());
        pRenderContext->copyBufferRegion(
            m_FirstHitCollectionCountsBufCPU.get(), 0, m_FirstHitCollectionCountsBuf.buffer.get(), 0, sizeof(uint) * firstHitPhotonCount
        );

        void* collectionCountData = m_FirstHitCollectionCountsBufCPU->map(Buffer::MapType::Read);
        firstHitCollectionCounts.resize(firstHitPhotonCount);
        std::memcpy(firstHitCollectionCounts.data(), collectionCountData, sizeof(uint) * firstHitPhotonCount);
        m_FirstHitCollectionCountsBufCPU->unmap();
    }

    // Handle caustic cluster collection
    std::vector<std::vector<float>> pSigmaSortBuffers;
    HandleCausticClusterCollection(pRenderContext, pSigmaSortBuffers, firstHitPhotonInfos, firstHitCollectionCounts, firstHitPhotonCount);

    // Generate first hit clusters
    std::vector<std::vector<float3>> lightFirstHitClusterPos;
    GenerateFirstHitClusters(pRenderContext, firstHitPhotonInfos, firstHitCollectionCounts, firstHitPhotonCount, lightFirstHitClusterPos);

    // Pre-allocate arrays with proper sizing
    std::vector<uint> clusterIndices;
    clusterIndices.reserve(clusterCount);
    std::vector<Gaussian3D> gaussians(totalGaussianCount);

    // Optimize gaussian creation loop
    for (size_t lightIdx = 0; lightIdx < lightCount; ++lightIdx)
    {
        // Resize and initialize indices vector once per light
        clusterIndices.resize(clusterCount);
        std::iota(clusterIndices.begin(), clusterIndices.end(), 0u);

        // Sort clusters by photon count (descending)
        const size_t baseIdx = lightIdx * clusterCount;
        if (baseIdx < pSigmaSortBuffers.size())
        {
            std::sort(
                clusterIndices.begin(), clusterIndices.end(),
                [&](const uint idx1, const uint idx2)
                {
                    const size_t bufIdx1 = baseIdx + idx1;
                    const size_t bufIdx2 = baseIdx + idx2;
                    return (bufIdx1 < pSigmaSortBuffers.size() ? pSigmaSortBuffers[bufIdx1].size() : 0) >
                           (bufIdx2 < pSigmaSortBuffers.size() ? pSigmaSortBuffers[bufIdx2].size() : 0);
                }
            );
        }

        // Cache light-specific data
        const size_t lightBaseGaussianIdx = lightIdx * m_GaussianCount;
        const size_t halfGaussianCount = m_GaussianCount / 2;
        const auto& lightClusters = (lightIdx < lightFirstHitClusterPos.size()) ? lightFirstHitClusterPos[lightIdx] : std::vector<float3>{};

        // Update gaussians with optimized indexing
        for (size_t gaussianIdx = 0; gaussianIdx < m_GaussianCount; ++gaussianIdx)
        {
            Gaussian3D& gaussian = gaussians[lightBaseGaussianIdx + gaussianIdx];

            if (gaussianIdx < halfGaussianCount)
            {
                // Use caustic clusters for first half
                if (gaussianIdx < clusterCount && gaussianIdx < clusterIndices.size())
                {
                    const uint clusterIdx = clusterIndices[gaussianIdx];
                    if (clusterIdx < m_CausticClustersPos.size())
                    {
                        gaussian.mean = m_CausticClustersPos[clusterIdx] * positionScaling;
                        gaussian.pSigma = (clusterIdx < m_CausticClustersPSigma.size()) ? m_CausticClustersPSigma[clusterIdx] : 1.0f;
                    }
                    else
                    {
                        gaussian.mean = RandomGenerator::AabbPoint(m_Scene->getSceneBounds()) * positionScaling;
                        gaussian.pSigma = 1.0f;
                    }
                }
                else
                {
                    gaussian.mean = RandomGenerator::AabbPoint(m_Scene->getSceneBounds()) * positionScaling;
                    gaussian.pSigma = 1.0f;
                }
            }
            else
            {
                // Use first hit clusters for second half
                const size_t clusterIdx = gaussianIdx - halfGaussianCount;
                if (clusterIdx < lightClusters.size())
                {
                    gaussian.mean = lightClusters[clusterIdx] * positionScaling;
                    gaussian.pSigma = 1.0f;
                }
                else
                {
                    gaussian.mean = RandomGenerator::AabbPoint(m_Scene->getSceneBounds()) * positionScaling;
                    gaussian.pSigma = 1.0f;
                }
            }
            gaussian.weight = 1.0f;
        }
    }

    // Optimize GPU copy operation - create buffer with initial data to avoid separate copy
    pRenderContext->copyBufferRegion(
        m_GaussianBuf.buffer.get(), 0,
        Buffer::createStructured(
            m_Device, sizeof(Gaussian3D), totalGaussianCount, ResourceBindFlags::None, Buffer::CpuAccess::Write, gaussians.data()
        )
            .get(),
        0, sizeof(Gaussian3D) * totalGaussianCount
    );
    pRenderContext->uavBarrier(m_GaussianBuf.buffer.get());
}

void GaussianPhotonGuiding::CalculateGaussianGradientCuda(RenderContext* pRenderContext)
{
    // Profile
    FALCOR_PROFILE(pRenderContext, "CalculateGaussianGradients");

    // Clear buffers
    pRenderContext->clearUAV(m_GradientBuf.buffer->getUAV().get(), float4(0.0f));

    // Ensure all previous GPU operations are completed
    pRenderContext->flush(true);

    // Calculate gradients
    CalculateGaussianGradient(
        m_GaussianCount, m_MaxFirstHitPhotonCount, reinterpret_cast<const Gaussian3D*>(m_GaussianBuf.devicePtr),
        reinterpret_cast<const uint*>(m_FirstHitCollectionCountsBuf.devicePtr),
        reinterpret_cast<const FirstHitPhotonInfo*>(m_FirstHitPhotonInfoBuf.devicePtr),
        reinterpret_cast<const uint*>(m_FirstHitPhotonCountBuf.devicePtr), reinterpret_cast<const float*>(m_SoftmaxBuf.devicePtr),
        reinterpret_cast<Gaussian3D*>(m_GradientBuf.devicePtr), GetPositionScaling(), m_Cs
    );

    // Ensure CUDA kernel has completed before proceeding
    syncCudaDevice();

#if 0
    // Copy gaussian buffer to CPU
    pRenderContext->uavBarrier(mp3dgGaussianBuffer.buffer.get());
    const size_t gaussianCount = m3dgGaussianCount * (m3dgAnalyticLightCount + m3dgGeometricLightCount);
    pRenderContext->copyBufferRegion(
        mp3dgGaussianBufferCPU.get(),
        0,
        mp3dgGaussianBuffer.buffer.get(),
        0,
        sizeof(Gaussian3D) * gaussianCount);
    std::vector<Gaussian3D> gaussians(gaussianCount);
    void* data = mp3dgGaussianBufferCPU->map(Buffer::MapType::Read);
    std::memcpy(gaussians.data(), data, sizeof(Gaussian3D) * gaussianCount);
    mp3dgGaussianBufferCPU->unmap();

    // Copy the gradient buffer to the CPU
    pRenderContext->uavBarrier(mp3dgGradientBuffer.buffer.get());
    pRenderContext->copyBufferRegion(
        mp3dgGradientBufferCPU.get(),
        0,
        mp3dgGradientBuffer.buffer.get(),
        0,
        sizeof(Gaussian3D) * gaussianCount);
    std::vector<Gaussian3D> gradients(gaussianCount);
    data = mp3dgGradientBufferCPU->map(Buffer::MapType::Read);
    std::memcpy(gradients.data(), data, sizeof(Gaussian3D) * gaussianCount);
    mp3dgGradientBufferCPU->unmap();

    __nop();
#endif
}

void GaussianPhotonGuiding::GaussianRepulsionPass(RenderContext* renderContext)
{
    // Profile
    FALCOR_PROFILE(renderContext, "GaussianRepulsion");

    // Init shader
    if (!m_GaussianRepulsionPass)
    {
        Program::Desc desc;
        desc.addShaderModules(m_Scene->getShaderModules());
        desc.addShaderLibrary(m_GaussianRepulsionShader).csEntry("main").setShaderModel(m_ShaderModel);
        desc.addTypeConformances(m_Scene->getTypeConformances());

        DefineList defines;
        defines.add(m_Defines);

        m_GaussianRepulsionPass = ComputePass::create(m_Device, desc, defines, true);
    }
    FALCOR_ASSERT(m_GaussianRepulsionPass);

    // Set variables
    auto var = m_GaussianRepulsionPass->getRootVar();
    var["Constants"]["gGaussianCount"] = m_GaussianCount;
    var["Constants"]["gLightCount"] = GetTotalLightCount();
    var["Constants"]["gPositionScaling"] = GetPositionScaling();
    var["Constants"]["gRepulsiveForce"] = m_RepulsiveForce;
    var["Constants"]["gRepulsiveDistance"] = m_RepulsiveDistance;

    var["gGaussians"] = m_GaussianBuf.buffer;
    var["gGaussianGradients"] = m_GradientBuf.buffer;

    // Execute
    m_GaussianRepulsionPass->execute(renderContext, uint3(GetTotalGaussianCount(), 1, 1));
}

void GaussianPhotonGuiding::OptimizeGaussiansPass(RenderContext* pRenderContext)
{
    // Profile
    FALCOR_PROFILE(pRenderContext, "OptimizeGaussians");

    // Init shader
    if (!m_OptimizeGaussiansPass)
    {
        Program::Desc desc;
        desc.addShaderModules(m_Scene->getShaderModules());
        desc.addShaderLibrary(m_OptimizeGaussiansShader).csEntry("main").setShaderModel(m_ShaderModel);
        desc.addTypeConformances(m_Scene->getTypeConformances());

        DefineList defines;
        defines.add(m_Defines);
        defines.add("OPTIM_SGD", "0");
        defines.add("OPTIM_ADAM", "0");

        m_OptimizeGaussiansPass = ComputePass::create(m_Device, desc, defines, true);
    }
    FALCOR_ASSERT(m_OptimizeGaussiansPass);

    // Set variables
    const uint totalGaussianCount = GetTotalGaussianCount();
    auto var = m_OptimizeGaussiansPass->getRootVar();

    var["Constants"]["gGaussianCount"] = m_GaussianCount;
    var["Constants"]["gTotalGaussianCount"] = totalGaussianCount;
    var["Constants"]["gLearningRate"] = m_LearningRate;
    var["Constants"]["gBeta1"] = m_Beta1;
    var["Constants"]["gBeta2"] = m_Beta2;

    var["gGaussians"] = m_GaussianBuf.buffer;
    var["gGradients"] = m_GradientBuf.buffer;
    var["gLightFirstHitCounts"] = m_LightFirstHitCountsBuf;
    var["gGaussianOptimizationData"] = m_OptimizationBuf;
    var["gGaussianOptimizationReset"] = m_OptimizationResetBuf;

    // More defines
    m_OptimizeGaussiansPass->getProgram()->addDefine("OPTIM_SGD", m_Optimizer == Optimizer::SGD ? "1" : "0");
    m_OptimizeGaussiansPass->getProgram()->addDefine("OPTIM_ADAM", m_Optimizer == Optimizer::Adam ? "1" : "0");

    // Execute
    m_OptimizeGaussiansPass->execute(pRenderContext, uint3(totalGaussianCount, 1, 1));

    // Copy gaussians to CPU
#if 0
    pRenderContext->uavBarrier(m_GaussianBuf.buffer.get());
    pRenderContext->copyBufferRegion(m_GaussianBufCPU.get(), 0, m_GaussianBuf.buffer.get(), 0, sizeof(Gaussian3D) * totalGaussianCount);
    std::span<Gaussian3D> gaussians(reinterpret_cast<Gaussian3D*>(m_GaussianBufCPU->map(Buffer::MapType::Read)), totalGaussianCount);
    __nop();
    m_GaussianBufCPU->unmap();
#endif
}

void GaussianPhotonGuiding::RandomReplacePass(RenderContext* renderContext)
{
    // Profile
    FALCOR_PROFILE(renderContext, "RandomReplace");
    if (!m_RandomReplace or m_FrameCountAfterOptimReset == 0 or m_FrameCountAfterOptimReset % m_RandomReplaceFrequency != 0)
    {
        return;
    }

    // Copy gaussians to CPU
    const size_t totalGaussianCount = GetTotalGaussianCount();
    renderContext->uavBarrier(m_GaussianBuf.buffer.get());
    renderContext->copyBufferRegion(m_GaussianBufReadCPU.get(), 0, m_GaussianBuf.buffer.get(), 0, sizeof(Gaussian3D) * totalGaussianCount);
    renderContext->flush(true);
    std::span<Gaussian3D> gaussians(
        reinterpret_cast<Gaussian3D*>(m_GaussianBufWriteCPU->map(Buffer::MapType::WriteDiscard)), totalGaussianCount
    );
    const std::span<Gaussian3D> srcGaussians(
        reinterpret_cast<Gaussian3D*>(m_GaussianBufReadCPU->map(Buffer::MapType::Read)), totalGaussianCount
    );
    std::copy(srcGaussians.begin(), srcGaussians.end(), gaussians.begin());
    m_GaussianBufReadCPU->unmap();

    // Copy reset buffer to CPU
    std::span<uint> resetData(
        reinterpret_cast<uint*>(m_OptimizationResetBufWriteCPU->map(Buffer::MapType::WriteDiscard)), totalGaussianCount
    );
    std::fill(resetData.begin(), resetData.end(), 0);

    const float positionScaling = GetPositionScaling();
    for (size_t lightIdx = 0; lightIdx < GetTotalLightCount(); ++lightIdx)
    {
        const size_t lightBaseIdx = lightIdx * m_GaussianCount;

        std::vector<size_t> indices(m_GaussianCount);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(
            indices.begin(), indices.begin() + m_RandomReplaceCount, indices.end(),
            [&](uint idx1, uint idx2) { return gaussians[lightBaseIdx + idx1].weight < gaussians[lightBaseIdx + idx2].weight; }
        );

        for (size_t badGaussIdx = 0; badGaussIdx < m_RandomReplaceCount; ++badGaussIdx)
        {
            // Reset optimization
            const size_t totalGaussIdx = lightBaseIdx + indices[badGaussIdx];
            resetData[totalGaussIdx] = 1;

            // Replace with random gaussian
            gaussians[totalGaussIdx] =
                Gaussian3D(RandomGenerator::AabbPoint(m_Scene->getSceneBounds()) * positionScaling, RandomGenerator::Float(), 0.0f);
        }
    }

    // Copy reset buffer to GPU
    renderContext->copyBufferRegion(
        m_OptimizationResetBuf.get(), 0, m_OptimizationResetBufWriteCPU.get(), 0, sizeof(uint) * totalGaussianCount);
    m_OptimizationResetBufWriteCPU->unmap();

    // Copy gaussians back to GPU
    renderContext->copyBufferRegion(m_GaussianBuf.buffer.get(), 0, m_GaussianBufWriteCPU.get(), 0, sizeof(Gaussian3D) * totalGaussianCount);
    m_GaussianBufWriteCPU->unmap();
}

void GaussianPhotonGuiding::CalculateSoftmaxWeightsPass(RenderContext* pRenderContext)
{
    // Profile
    FALCOR_PROFILE(pRenderContext, "CalculateSoftmaxWeights");

    // Init shader
    if (!m_CalculateSoftmaxPass)
    {
        Program::Desc desc;
        desc.addShaderModules(m_Scene->getShaderModules());
        desc.addShaderLibrary(m_CalculateSoftmaxWeightsShader).csEntry("main").setShaderModel(m_ShaderModel);
        desc.addTypeConformances(m_Scene->getTypeConformances());

        m_CalculateSoftmaxPass = ComputePass::create(m_Device, desc, m_Defines, true);
    }
    FALCOR_ASSERT(m_CalculateSoftmaxPass);

    // Set variables
    auto var = m_CalculateSoftmaxPass->getRootVar();
    var["Constants"]["gGaussianCount"] = m_GaussianCount;
    var["Constants"]["gAnalyticLightCount"] = m_AnalyticLightCount;
    var["Constants"]["gGeometricLightCount"] = m_GeometricLightCount;

    // Buffers
    var["gGaussians"] = m_GaussianBuf.buffer;
    var["gSoftmaxWeights"] = m_SoftmaxBuf.buffer;

    // Execute
    m_CalculateSoftmaxPass->execute(pRenderContext, uint3(GetTotalLightCount(), 1, 1));

#if 0
    // Copy softmax buffer to CPU
    const uint gaussianCount = m3dgGaussianCount * (m3dgAnalyticLightCount + m3dgGeometricLightCount);
    pRenderContext->uavBarrier(mp3dgSoftmaxBuffer.buffer.get());
    pRenderContext->copyBufferRegion(
        mp3dgSoftmaxBufferCPU.get(),
        0,
        mp3dgSoftmaxBuffer.buffer.get(),
        0,
        sizeof(float) * gaussianCount);

    std::vector<float> softmaxWeights(gaussianCount);
    void* data = mp3dgSoftmaxBufferCPU->map(Buffer::MapType::Read);
    std::memcpy(softmaxWeights.data(), data, sizeof(float) * gaussianCount);
    mp3dgSoftmaxBufferCPU->unmap();

    __nop();
#endif
}

void GaussianPhotonGuiding::EndFrame(RenderContext* renderContext)
{
    // Increment counter after this is called because this happens when GaussianPhotonGuiding is used
    ++m_FrameCountAfterOptimReset;
}

void GaussianPhotonGuiding::ClearBuffersForGeneratePhotons(RenderContext* renderContext)
{
    renderContext->clearUAV(m_FirstHitPhotonCountBuf.buffer->getUAV().get(), uint4(0));
    renderContext->clearUAV(m_LightFirstHitCountsBuf->getUAV().get(), uint4(0));
}

void GaussianPhotonGuiding::SetGeneratePhotonsVars(const ShaderVar& var) const
{
    // Constants
    static const std::string nameBuf = "GaussianPhotonGuiding";
    var[nameBuf]["gGaussianCount"] = m_GaussianCount;
    var[nameBuf]["gAnalyticLightCount"] = m_AnalyticLightCount;
    var[nameBuf]["gGeometricLightCount"] = m_GeometricLightCount;
    var[nameBuf]["gGmmMinPdf"] = m_MinPdf;
    var[nameBuf]["gBeta"] = m_FrameCountAfterOptimReset <= 1 and IsRobustInitialization() ? 0.0f : m_Beta;
    var[nameBuf]["gMaxFirstHitPhotonCount"] = m_MaxFirstHitPhotonCount;
    var[nameBuf]["gPositionScaling"] = GetPositionScaling();
    var[nameBuf]["gCS"] = m_Cs;

    // Buffers
    var["gGaussians"] = m_GaussianBuf.buffer;
    var["gFirstHitPhotonCounter"] = m_FirstHitPhotonCountBuf.buffer;
    var["gFirstHitPhotonInfo"] = m_FirstHitPhotonInfoBuf.buffer;
    for (uint32_t idx = 0; idx < 2; ++idx)
    {
        var["gPhotonFirstHitMap"][idx] = m_PhotonFirstHitMapBufs[idx];
    }
    var["gLightFirstHitCounts"] = m_LightFirstHitCountsBuf;
    var["gSoftmaxWeights"] = m_SoftmaxBuf.buffer;
}

void GaussianPhotonGuiding::ClearBuffersForPhotonCollection(RenderContext* renderContext)
{
    renderContext->clearUAV(m_FirstHitCollectionCountsBuf.buffer->getUAV().get(), uint4(0));
}

void GaussianPhotonGuiding::SetCollectPhotonsVars(const ShaderVar& var) const
{
    var["GaussianPhotonGuiding"]["gMaxFirstHitPhotonCount"] = m_MaxFirstHitPhotonCount;
    var["GaussianPhotonGuiding"]["gGpgGlobalPhotonWeight"] = m_GlobalPhotonWeight;
    var["GaussianPhotonGuiding"]["gGpgCausticPhotonWeight"] = m_CausticPhotonWeight;

    for (uint32_t idx = 0; idx < 2; ++idx)
    {
        var["gPhotonFirstHitMap"][idx] = m_PhotonFirstHitMapBufs[idx];
    }
    var["gFirstHitCollectionCounts"] = m_FirstHitCollectionCountsBuf.buffer;
}

void GaussianPhotonGuiding::SetFinalShadingVars(const ShaderVar& var) const
{
    const std::string nameBuf = "GaussianPhotonGuiding";
    var[nameBuf]["gGaussianCount"] = m_GaussianCount;
    var[nameBuf]["gAnalyticLightCount"] = m_AnalyticLightCount;
    var[nameBuf]["gGeometricLightCount"] = m_GeometricLightCount;
    var[nameBuf]["gPositionScaling"] = GetPositionScaling();
    var[nameBuf]["gCS"] = m_Cs;

    var["gGaussians"] = m_GaussianBuf.buffer;
}

void GaussianPhotonGuiding::GenerateCausticPoints(
    const size_t geomInstanceIdx,
    const std::span<PackedStaticVertexData>& vertexData,
    const std::span<uint32_t>& indexData
)
{
    // Get geometry instance info with bounds checking
    if (geomInstanceIdx >= m_CausticGeometryInstanceIDs.size())
        return;

    const uint geometryInstanceID = m_CausticGeometryInstanceIDs[geomInstanceIdx];
    const uint geometryId = m_Scene->getGeometryInstance(geometryInstanceID).geometryID;

    // Get mesh info
    const MeshDesc& mesh = m_Scene->getMesh(MeshID(geometryId));
    const uint vertexOffset = mesh.vbOffset;
    const uint indexOffset = mesh.ibOffset;
    const uint triangleCount = mesh.getTriangleCount();
    const bool bit16 = mesh.use16BitIndices();

    // Early exit for invalid meshes
    if (!mesh.useVertexIndices() || triangleCount == 0)
    {
        return;
    }

    // Pre-allocate caustic points with exact size
    std::vector<std::array<float, 3>> causticPoints;
    causticPoints.reserve(m_GenCausticPointCount);

    // Cache random number generator for better performance
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::uniform_int_distribution<uint> triangleDist(0, triangleCount - 1);
    std::uniform_real_distribution<float> realDist(0.0f, 1.0f);

    // Generate caustic points with optimized random sampling
    for (size_t pointIdx = 0; pointIdx < m_GenCausticPointCount; ++pointIdx)
    {
        // Choose random triangle with better distribution
        const uint randomTriangle = triangleDist(gen);

        // Read indices with optimized bit manipulation
        uint32_t index0, index1, index2;
        if (bit16)
        {
            // Optimized 16-bit index reading
            const uint firstIndexIndex = (indexOffset * 2) + (randomTriangle * 3);
            const uint firstIndexIndex32 = firstIndexIndex / 2;

            if (firstIndexIndex32 + 1 >= indexData.size())
                continue; // Skip invalid indices

            const uint32_t data1 = indexData[firstIndexIndex32];
            const uint32_t data2 = indexData[firstIndexIndex32 + 1];

            if (firstIndexIndex % 2 == 0)
            {
                index0 = data1 & 0xFFFF;
                index1 = (data1 >> 16) & 0xFFFF;
                index2 = data2 & 0xFFFF;
            }
            else
            {
                index0 = (data1 >> 16) & 0xFFFF;
                index1 = data2 & 0xFFFF;
                index2 = (data2 >> 16) & 0xFFFF;
            }
        }
        else
        {
            // 32-bit index reading with bounds checking
            const uint firstIndexIndex = indexOffset + (randomTriangle * 3);
            if (firstIndexIndex + 2 >= indexData.size())
                continue; // Skip invalid indices

            index0 = indexData[firstIndexIndex];
            index1 = indexData[firstIndexIndex + 1];
            index2 = indexData[firstIndexIndex + 2];
        }

        // Validate vertex indices
        if (vertexOffset + index0 >= vertexData.size() || vertexOffset + index1 >= vertexData.size() ||
            vertexOffset + index2 >= vertexData.size())
        {
            continue; // Skip invalid vertices
        }

        // Read vertices
        const float3 vertex0 = vertexData[vertexOffset + index0].position;
        const float3 vertex1 = vertexData[vertexOffset + index1].position;
        const float3 vertex2 = vertexData[vertexOffset + index2].position;

        // Optimized barycentric sampling using uniform distribution
        float u = realDist(gen);
        float v = realDist(gen);

        // Ensure point is inside triangle
        if (u + v > 1.0f)
        {
            u = 1.0f - u;
            v = 1.0f - v;
        }

        const float w = 1.0f - u - v;

        // Calculate point with optimized arithmetic
        const float3 point = vertex0 * w + vertex1 * u + vertex2 * v;
        causticPoints.emplace_back(std::array<float, 3>{point.x, point.y, point.z});
    }

    // Early exit if no valid points generated
    if (causticPoints.empty())
    {
        return;
    }

    try
    {
        // Use appropriate clustering algorithm based on point count
        const size_t actualClusterCount = std::min<size_t>(m_CausticClusterCount, causticPoints.size());

        const auto [clusters, _] = (causticPoints.size() > 1000)
                                       ? dkm::kmeans_lloyd_parallel(causticPoints, dkm::clustering_parameters<float>(actualClusterCount))
                                       : dkm::kmeans_lloyd(causticPoints, dkm::clustering_parameters<float>(actualClusterCount));

        // Store clusters with bounds checking
        const size_t baseIdx = geomInstanceIdx * m_CausticClusterCount;
        for (size_t clusterIdx = 0; clusterIdx < clusters.size() && clusterIdx < m_CausticClusterCount; ++clusterIdx)
        {
            const auto& cluster = clusters[clusterIdx];
            const size_t outputIdx = baseIdx + clusterIdx;
            if (outputIdx < m_CausticClustersPos.size())
            {
                m_CausticClustersPos[outputIdx] = float3(cluster[0], cluster[1], cluster[2]);
            }
        }
    }
    catch (const std::exception&)
    {
        // Fallback: if clustering fails, use first point for all clusters
        const size_t baseIdx = geomInstanceIdx * m_CausticClusterCount;
        const auto& firstPoint = causticPoints[0];
        const float3 fallbackPos(firstPoint[0], firstPoint[1], firstPoint[2]);

        for (size_t clusterIdx = 0; clusterIdx < m_CausticClusterCount; ++clusterIdx)
        {
            const size_t outputIdx = baseIdx + clusterIdx;
            if (outputIdx < m_CausticClustersPos.size())
            {
                m_CausticClustersPos[outputIdx] = fallbackPos;
            }
        }
    }
}

void GaussianPhotonGuiding::HandleCausticClusterCollection(
    RenderContext* pRenderContext,
    std::vector<std::vector<float>>& pSigmaSortBuffers,
    const std::vector<FirstHitPhotonInfo>& firstHitPhotonInfos,
    const std::vector<uint>& firstHitCollectionCounts,
    const size_t firstHitPhotonCount
)
{
    // Profile
    FALCOR_PROFILE(pRenderContext, "HandleCausticClusterCollection");

    // Early exit if no photons to process
    if (firstHitPhotonCount == 0 || m_CausticClustersPos.empty())
    {
        const size_t lightCount = GetTotalLightCount();
        const size_t clusterCount = GetTotalCausticClusterCount();
        const size_t lightClusterCount = clusterCount * lightCount;

        pSigmaSortBuffers.assign(lightClusterCount, std::vector<float>());
        m_CausticClustersPSigma.assign(lightClusterCount, 0.0f);
        return;
    }

    // Cache frequently used values
    const size_t lightCount = GetTotalLightCount();
    const size_t clusterCount = GetTotalCausticClusterCount();
    const size_t lightClusterCount = clusterCount * lightCount;

    // Pre-allocate vectors with estimated capacity
    std::vector<size_t> firstHitPhotonClosestClusterIdx;
    std::vector<float> firstHitPhotonClosestClusterSigmaP;
    firstHitPhotonClosestClusterIdx.reserve(firstHitPhotonCount);
    firstHitPhotonClosestClusterSigmaP.reserve(firstHitPhotonCount);

    // Use parallel algorithm to find closest clusters
    firstHitPhotonClosestClusterIdx.resize(firstHitPhotonCount);
    firstHitPhotonClosestClusterSigmaP.resize(firstHitPhotonCount);

    // Create indices for parallel processing
    std::vector<size_t> indices(firstHitPhotonCount);
    std::iota(indices.begin(), indices.end(), 0);

    // Parallel computation of closest clusters with optimized distance calculation
    std::for_each(
        std::execution::par_unseq, indices.begin(), indices.end(),
        [&](const size_t firstHitPhotonIdx)
        {
            const FirstHitPhotonInfo& info = firstHitPhotonInfos[firstHitPhotonIdx];
            const float3& photonPos = info.pos;

            size_t closestClusterIdx = 0;
            float closestDistanceSq = std::numeric_limits<float>::max();

            // Use squared distance to avoid expensive sqrt operations
            for (size_t clusterIdx = 0; clusterIdx < clusterCount; ++clusterIdx)
            {
                const float3& clusterPos = m_CausticClustersPos[clusterIdx];
                const float3 diff = photonPos - clusterPos;
                const float distanceSq = dot(diff, diff);

                if (distanceSq < closestDistanceSq)
                {
                    closestDistanceSq = distanceSq;
                    closestClusterIdx = clusterIdx;
                }
            }

            // Store results
            firstHitPhotonClosestClusterIdx[firstHitPhotonIdx] = closestClusterIdx;
            firstHitPhotonClosestClusterSigmaP[firstHitPhotonIdx] = Gaussian3D::PSigmaFromDistance(std::sqrt(closestDistanceSq), m_Cs);
        }
    );

    // Pre-allocate pSigmaSortBuffers with estimated capacity
    pSigmaSortBuffers.assign(lightClusterCount, std::vector<float>());

    // First pass: calculate total capacity needed per light cluster
    std::vector<size_t> lightClusterCapacities(lightClusterCount, 0);
    for (size_t firstHitPhotonIdx = 0; firstHitPhotonIdx < firstHitPhotonCount; ++firstHitPhotonIdx)
    {
        const FirstHitPhotonInfo& info = firstHitPhotonInfos[firstHitPhotonIdx];
        const size_t lightClusterIdx = info.lightIdx * clusterCount + firstHitPhotonClosestClusterIdx[firstHitPhotonIdx];
        if (lightClusterIdx < lightClusterCount)
        {
            lightClusterCapacities[lightClusterIdx] += firstHitCollectionCounts[firstHitPhotonIdx];
        }
    }

    // Reserve capacity for each light cluster buffer
    for (size_t lightClusterIdx = 0; lightClusterIdx < lightClusterCount; ++lightClusterIdx)
    {
        pSigmaSortBuffers[lightClusterIdx].reserve(lightClusterCapacities[lightClusterIdx]);
    }

    // Second pass: populate the buffers
    for (size_t firstHitPhotonIdx = 0; firstHitPhotonIdx < firstHitPhotonCount; ++firstHitPhotonIdx)
    {
        const FirstHitPhotonInfo& info = firstHitPhotonInfos[firstHitPhotonIdx];
        const size_t lightClusterIdx = info.lightIdx * clusterCount + firstHitPhotonClosestClusterIdx[firstHitPhotonIdx];

        if (lightClusterIdx < lightClusterCount)
        {
            const float pSigma = firstHitPhotonClosestClusterSigmaP[firstHitPhotonIdx];
            const size_t collectionCount = firstHitCollectionCounts[firstHitPhotonIdx];

            // Batch insert to reduce allocation overhead
            auto& buffer = pSigmaSortBuffers[lightClusterIdx];
            buffer.insert(buffer.end(), collectionCount, pSigma);
        }
    }

    // Find the median for each light cluster using optimized approach
    m_CausticClustersPSigma.resize(lightClusterCount);

    // Parallel median calculation
    std::vector<size_t> lightClusterIndices(lightClusterCount);
    std::iota(lightClusterIndices.begin(), lightClusterIndices.end(), 0);

    std::for_each(
        std::execution::par_unseq, lightClusterIndices.begin(), lightClusterIndices.end(),
        [&](const size_t lightClusterIdx)
        {
            auto& sigmaPBuffer = pSigmaSortBuffers[lightClusterIdx];
            if (sigmaPBuffer.empty())
            {
                m_CausticClustersPSigma[lightClusterIdx] = 0.0f;
                return;
            }

            // Use nth_element for faster median finding (O(n) vs O(n log n))
            const size_t medianIndex = sigmaPBuffer.size() / 2;
            std::nth_element(sigmaPBuffer.begin(), sigmaPBuffer.begin() + medianIndex, sigmaPBuffer.end());
            m_CausticClustersPSigma[lightClusterIdx] = sigmaPBuffer[medianIndex];
        }
    );
}

void GaussianPhotonGuiding::GenerateFirstHitClusters(
    RenderContext* pRenderContext,
    const std::vector<FirstHitPhotonInfo>& firstHitPhotonInfos,
    const std::vector<uint>& firstHitCollectionCounts,
    const size_t firstHitPhotonCount,
    std::vector<std::vector<float3>>& lightFirstHitClusterPos
)
{
    // Profile
    FALCOR_PROFILE(pRenderContext, "GenerateFirstHitClusters");

    // Early exit if no photons to process
    if (firstHitPhotonCount == 0)
    {
        const size_t lightCount = GetTotalLightCount();
        lightFirstHitClusterPos.assign(lightCount, std::vector<float3>());
        return;
    }

    // Cache frequently used values
    const size_t lightCount = GetTotalLightCount();

    // Pre-allocate and estimate capacity for light hit points
    std::vector<std::vector<std::array<float, 3>>> lightFirstHitPoints(lightCount);

    // First pass: calculate total points per light for capacity estimation
    std::vector<size_t> lightPointCounts(lightCount, 0);
    for (size_t firstHitPhotonIdx = 0; firstHitPhotonIdx < firstHitPhotonCount; ++firstHitPhotonIdx)
    {
        const FirstHitPhotonInfo& info = firstHitPhotonInfos[firstHitPhotonIdx];
        if (info.lightIdx < lightCount)
        {
            lightPointCounts[info.lightIdx] += firstHitCollectionCounts[firstHitPhotonIdx];
        }
    }

    // Reserve capacity to avoid reallocations
    for (size_t lightIdx = 0; lightIdx < lightCount; ++lightIdx)
    {
        lightFirstHitPoints[lightIdx].reserve(lightPointCounts[lightIdx]);
    }

    // Second pass: populate light hit points with optimized data access
    for (size_t firstHitPhotonIdx = 0; firstHitPhotonIdx < firstHitPhotonCount; ++firstHitPhotonIdx)
    {
        const FirstHitPhotonInfo& info = firstHitPhotonInfos[firstHitPhotonIdx];
        if (info.lightIdx >= lightCount)
            continue; // Bounds check

        const uint collectionCount = firstHitCollectionCounts[firstHitPhotonIdx];
        const std::array<float, 3> point = {info.pos.x, info.pos.y, info.pos.z};

        // Batch insert to reduce allocation overhead
        auto& lightPoints = lightFirstHitPoints[info.lightIdx];
        lightPoints.insert(lightPoints.end(), collectionCount, point);
    }

    // Pre-allocate output vector
    lightFirstHitClusterPos.assign(lightCount, std::vector<float3>());

    // Process each light in parallel for clustering
    std::vector<size_t> lightIndices(lightCount);
    std::iota(lightIndices.begin(), lightIndices.end(), 0);

    // Use parallel execution for clustering when beneficial
    const bool useParallel = lightCount > 4; // Only parallelize if we have enough lights

    auto processLight = [&](size_t lightIdx)
    {
        const auto& points = lightFirstHitPoints[lightIdx];
        if (points.empty())
        {
            return; // lightFirstHitClusterPos[lightIdx] already initialized as empty
        }

        // Use std::set for unique point detection (avoids hash function requirement)
        std::set<std::array<float, 3>> uniquePointsSet;

        // Use sampling for large point sets to improve performance
        const size_t maxSampleSize = 10000;
        if (points.size() > maxSampleSize)
        {
            // Sample points to estimate uniqueness
            const size_t step = points.size() / maxSampleSize;
            for (size_t i = 0; i < points.size(); i += step)
            {
                uniquePointsSet.insert(points[i]);
                if (uniquePointsSet.size() >= maxSampleSize)
                    break;
            }
        }
        else
        {
            uniquePointsSet.insert(points.begin(), points.end());
        }

        // Determine optimal cluster count (make K configurable in the future)
        static constexpr size_t maxClusters = 8;
        const size_t clusterCount = std::min<size_t>(uniquePointsSet.size(), maxClusters);

        if (clusterCount == 0)
            return;

        try
        {
            // Use parallel k-means for larger datasets
            const auto [clusters, _] = (points.size() > 1000)
                                           ? dkm::kmeans_lloyd_parallel(points, dkm::clustering_parameters<float>(clusterCount))
                                           : dkm::kmeans_lloyd(points, dkm::clustering_parameters<float>(clusterCount));

            // Pre-allocate result vector
            lightFirstHitClusterPos[lightIdx].reserve(clusters.size());

            // Convert clusters to float3 format
            for (const auto& cluster : clusters)
            {
                lightFirstHitClusterPos[lightIdx].emplace_back(cluster[0], cluster[1], cluster[2]);
            }
        }
        catch (const std::exception&)
        {
            // Fallback: if clustering fails, use first point as single cluster
            if (!points.empty())
            {
                const auto& firstPoint = points[0];
                lightFirstHitClusterPos[lightIdx].emplace_back(firstPoint[0], firstPoint[1], firstPoint[2]);
            }
        }
    };

    // Execute clustering
    if (useParallel)
    {
        std::for_each(std::execution::par_unseq, lightIndices.begin(), lightIndices.end(), processLight);
    }
    else
    {
        std::for_each(lightIndices.begin(), lightIndices.end(), processLight);
    }
}
