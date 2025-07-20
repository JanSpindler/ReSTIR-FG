#include "GaussianPhotonGuiding.h"
#include "Gaussian3D.h"
#include "RandomGenerator.h"
#include "FirstHitPhotonInfo.h"
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

    if (!m_GaussianTex)
    {
        m_GaussianTex = Texture::create2D(
            m_Device, screenSize.x, screenSize.y, ResourceFormat::RGBA32Float, 1, 1, nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
        renderContext->clearUAV(m_GaussianTex->getUAV().get(), float4(0.0f));
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

        // Random replace
        changed |= group.checkbox("Enable Random Replace", m_RandomReplace);
        changed |= group.var("Random Replace Count", m_RandomReplaceCount, 0u, m_GaussianCount);
        changed |= group.var("Random Replace Frequency", m_RandomReplaceFrequency, 1u, 1024u);
    }

    return changed;
}

void GaussianPhotonGuiding::ResetSceneTextures()
{
    m_GaussianTex.reset();
}

void GaussianPhotonGuiding::ResetPhotonFirstHitMap()
{
    m_PhotonFirstHitMapBufs[0].reset();
    m_PhotonFirstHitMapBufs[1].reset();
}

void GaussianPhotonGuiding::GenerateCausticClusters(RenderContext* renderContext)
{
    // Profile
    logInfo("Generating caustic cluster");
    FALCOR_PROFILE(renderContext, "GenerateCausticPoints");

    // Get vertex position
    ref<Vao> vao = m_Scene->getMeshVao();
    ref<Buffer> vbo = vao->getVertexBuffer(0); // Assumes we use the static vertex buffer (index 0)
    ref<Buffer> ibo = vao->getIndexBuffer();
    if (vao->getIndexBufferFormat() != ResourceFormat::R32Uint)
    {
        throw RuntimeError("ReSTIR_FG: Only uint32 index buffer format is supported");
    }

    // Copy vertex and index buffers to CPU
    const size_t indexByteSize = ibo->getSize();
    const size_t indexCount = indexByteSize / sizeof(uint32_t);
    ref<Buffer> iboCpu = Buffer::create(m_Device, indexByteSize, ResourceBindFlags::None, Buffer::CpuAccess::Read);
    renderContext->copyBufferRegion(iboCpu.get(), 0, ibo.get(), 0, indexByteSize);
    renderContext->flush(true);
    const std::span<uint32_t> indexData(reinterpret_cast<uint32_t*>(iboCpu->map(Buffer::MapType::Read)), indexCount);

    const uint32_t vertexCount = vbo->getElementCount();
    ref<Buffer> vboCpu =
        Buffer::createStructured(m_Device, sizeof(PackedStaticVertexData), vertexCount, ResourceBindFlags::None, Buffer::CpuAccess::Read);
    renderContext->copyBufferRegion(vboCpu.get(), 0, vbo.get(), 0, sizeof(PackedStaticVertexData) * vertexCount);
    renderContext->flush(true);
    const std::span<PackedStaticVertexData> vertexData(
        reinterpret_cast<PackedStaticVertexData*>(vboCpu->map(Buffer::MapType::Read)), vertexCount
    );

    // Calculate caustic clusters
    const size_t totalCausticClusterCount = GetTotalCausticClusterCount();
    m_CausticClustersPos.resize(totalCausticClusterCount);

    std::vector<size_t> indices(m_CausticGeometryInstanceIDs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(
        std::execution::par_unseq, indices.begin(), indices.end(),
        [&](const size_t geomInstanceIdx) { GenerateCausticPoints(renderContext, geomInstanceIdx, vertexData, indexData); }
    );

    // Unmap buffers
    vbo->unmap();
    ibo->unmap();
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

void GaussianPhotonGuiding::CountCausticClustersPass(RenderContext* pRenderContext)
{
    // Profile
    FALCOR_PROFILE(pRenderContext, "CountCausticClusters");

    // Set variables
    const size_t lightCount = GetTotalLightCount();
    const size_t clusterCount = GetTotalCausticClusterCount();
    const size_t lightClusterCount = clusterCount * lightCount;
    const size_t firstHitPhotonCount = std::min<size_t>(m_MaxFirstHitPhotonCount, m_ActualFirstHitPhotonCount);

    // TODO: Count
    // Copy first hit photon info to CPU
    pRenderContext->uavBarrier(m_FirstHitPhotonInfoBuf.buffer.get());
    pRenderContext->copyBufferRegion(
        m_FirstHitPhotonInfoBufCPU.get(), 0, m_FirstHitPhotonInfoBuf.buffer.get(), 0, sizeof(FirstHitPhotonInfo) * firstHitPhotonCount
    );
    std::vector<FirstHitPhotonInfo> firstHitPhotonInfos(firstHitPhotonCount);
    std::memcpy(
        firstHitPhotonInfos.data(), m_FirstHitPhotonInfoBufCPU->map(Buffer::MapType::Read), sizeof(FirstHitPhotonInfo) * firstHitPhotonCount
    );

    // For each first hit photon track the closest caustic cluster and store the distance to it
    std::vector<size_t> firstHitPhotonClosestClusterIdx(firstHitPhotonCount, std::numeric_limits<size_t>::max());
    std::vector<size_t> firstHitPhotonClosestClusterSigmaP(firstHitPhotonCount, std::numeric_limits<size_t>::max());

    std::vector<size_t> indices(firstHitPhotonCount);
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(
        std::execution::par_unseq, indices.begin(), indices.end(),
        [&](const size_t firstHitPhotonIdx)
        {
            // Get first hit photon info
            const FirstHitPhotonInfo& info = firstHitPhotonInfos[firstHitPhotonIdx];

            // Select closest caustic cluster
            const float3& photonPos = info.pos;
            size_t closestClusterIdx = 0;
            float closestDistance = std::numeric_limits<float>::max();
            for (size_t clusterIdx = 0; clusterIdx < clusterCount; ++clusterIdx)
            {
                const float3 clusterPos = m_CausticClustersPos[clusterIdx];
                const float distance = length(photonPos - clusterPos);
                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    closestClusterIdx = clusterIdx;
                }
            }

            // Store closest cluster info
            firstHitPhotonClosestClusterIdx[firstHitPhotonIdx] = closestClusterIdx;
            firstHitPhotonClosestClusterSigmaP[firstHitPhotonIdx] = Gaussian3D::PSigmaFromDistance(closestDistance, m_Cs);
        }
    );

    // Store p sigmas in buffers
    std::vector<std::vector<float>> sigmaPSortBuffers(lightClusterCount, std::vector<float>(0));
    for (size_t firstHitPhotonIdx = 0; firstHitPhotonIdx < firstHitPhotonCount; ++firstHitPhotonIdx)
    {
        const FirstHitPhotonInfo& info = firstHitPhotonInfos[firstHitPhotonIdx];
        const size_t lightClusterIdx = info.lightIdx * clusterCount + firstHitPhotonClosestClusterIdx[firstHitPhotonIdx];
        sigmaPSortBuffers[lightClusterIdx].push_back(firstHitPhotonClosestClusterSigmaP[firstHitPhotonIdx]);
    }

    // Find the median for each light cluster
    m_CausticClustersPSigma.resize(lightClusterCount);
    for (size_t lightClusterIdx = 0; lightClusterIdx < lightClusterCount; ++lightClusterIdx)
    {
        // Get p sigma buffer
        std::vector<float>& sigmaPBuffer = sigmaPSortBuffers[lightClusterIdx];
        if (sigmaPBuffer.empty())
        {
            m_CausticClustersPSigma[lightClusterIdx] = 0.0f;
            continue;
        }

        // Sort and find median
        std::sort(sigmaPBuffer.begin(), sigmaPBuffer.end());
        const size_t medianIndex = sigmaPBuffer.size() / 2;
        m_CausticClustersPSigma[lightClusterIdx] = sigmaPBuffer[medianIndex];
    }

    // Weird logic
    if (m_FrameCountAfterOptimReset < 1)
    {
        return;
    }

    // Sort and select gaussian
    const size_t totalGaussianCount = m_GaussianCount * lightCount;

    std::vector<uint> clusterIndices(clusterCount);
    std::vector<Gaussian3D> gaussians(totalGaussianCount);

    for (size_t lightIdx = 0; lightIdx < lightCount; ++lightIdx)
    {
        // Init indices
        for (size_t clusterIdx = 0; clusterIdx < clusterCount; ++clusterIdx)
        {
            clusterIndices[clusterIdx] = clusterIdx;
        }

        // Sort
        const size_t baseIdx = lightIdx * clusterCount;
        std::sort(
            clusterIndices.begin(), clusterIndices.end(), [&](const uint idx1, const uint idx2)
            { return sigmaPSortBuffers[baseIdx + idx1].size() > sigmaPSortBuffers[baseIdx + idx2].size(); }
        );

        // Update gaussians
        const float positionScaling = GetPositionScaling();
        for (size_t gaussianIdx = 0; gaussianIdx < m_GaussianCount; ++gaussianIdx)
        {
            Gaussian3D& gaussian = gaussians[lightIdx * m_GaussianCount + gaussianIdx];
            gaussian.mean =
                gaussianIdx < clusterCount ? m_CausticClustersPos[clusterIndices[gaussianIdx]] * positionScaling : RandomGenerator::Float3();
            gaussian.pSigma = m_CausticClustersPSigma[clusterIndices[gaussianIdx]];
            gaussian.weight = 1.0f;
        }
    }

    // Copy gaussians to GPU
    ref<Buffer> gaussianBufferCPU = Buffer::createStructured(
        m_Device, sizeof(Gaussian3D), totalGaussianCount, ResourceBindFlags::None, Buffer::CpuAccess::Write, gaussians.data()
    );
    pRenderContext->copyBufferRegion(m_GaussianBuf.buffer.get(), 0, gaussianBufferCPU.get(), 0, sizeof(Gaussian3D) * totalGaussianCount);
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
    RenderContext* pRenderContext,
    const uint geomInstanceIdx,
    const std::span<PackedStaticVertexData>& vertexData,
    const std::span<uint32_t>& indexData
)
{
    // Get geometry instance info
    const uint geometryInstanceID = m_CausticGeometryInstanceIDs[geomInstanceIdx];
    const uint geometryId = m_Scene->getGeometryInstance(geometryInstanceID).geometryID;

    // Get mesh info
    const MeshDesc& mesh = m_Scene->getMesh(MeshID(geometryId));
    const uint vertexOffset = mesh.vbOffset;
    const uint indexOffset = mesh.ibOffset;
    const uint triangleCount = mesh.getTriangleCount();
    const bool bit16 = mesh.use16BitIndices();
    if (!mesh.useVertexIndices())
    {
        throw RuntimeError("ReSTIR_FG: Only indexed meshes are supported");
    }

    // Generate caustic points
    std::vector<std::array<float, 3>> causticPoints(m_GenCausticPointCount);
    for (size_t pointIdx = 0; pointIdx < m_GenCausticPointCount; ++pointIdx)
    {
        // Choose random triangle
        const uint randomTriangle = RandomGenerator::UInt() % triangleCount;

        // Read indices
        uint32_t index0 = 0, index1 = 0, index2 = 0;
        if (bit16)
        {
            const uint firstIndexIndex = (indexOffset * 2) + (randomTriangle * 3);
            const uint firstIndexIndex32 = std::lldiv(firstIndexIndex, 2).quot;
            const uint32_t data1 = indexData[firstIndexIndex32 + 0];
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
            const uint firstIndexIndex = indexOffset + (randomTriangle * 3);
            index0 = indexData[firstIndexIndex + 0];
            index1 = indexData[firstIndexIndex + 1];
            index2 = indexData[firstIndexIndex + 2];
        }

        // Read vertices
        const float3 vertex0 = vertexData[vertexOffset + index0].position;
        const float3 vertex1 = vertexData[vertexOffset + index1].position;
        const float3 vertex2 = vertexData[vertexOffset + index2].position;

        // Barycentric sampling
        float3 bary = RandomGenerator::Float3();
        bary /= bary.x + bary.y + bary.z;

        // Calculate point
        const float3 point = vertex0 * bary.x + vertex1 * bary.y + vertex2 * bary.z;
        causticPoints[pointIdx] = {point.x, point.y, point.z};
    }

    // Cluster
    const auto [clusters, _] = dkm::kmeans_lloyd(causticPoints, dkm::clustering_parameters<float>(m_CausticClusterCount));
    for (size_t clusterIdx = 0; clusterIdx < clusters.size(); ++clusterIdx)
    {
        const std::array<float, 3>&cluster = clusters[clusterIdx];
        m_CausticClustersPos[geomInstanceIdx * m_CausticClusterCount + clusterIdx] = {cluster[0], cluster[1], cluster[2]};
    }
}
