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
        const float sigma = GetSceneSize() / 30.0f;
        for (size_t gaussIdx = 0; gaussIdx < totalGaussianCount; ++gaussIdx)
        {
            gaussians[gaussIdx] = Gaussian3D(RandomGenerator::AabbPoint(m_Scene->getSceneBounds()), sigma, 0.0f);
        }

        // Create buffer
        m_GaussianBuf = CreateStructuredInteropBuffer<Gaussian3D>(m_Device, totalGaussianCount, Buffer::CpuAccess::None, gaussians.data());

        // Reset optimization
        m_OptimStep = 0;
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
        const std::vector<Gaussian3DMoments> gaussianMoments(totalGaussianCount, Gaussian3DMoments());
        m_OptimizationBuf = Buffer::createStructured(
            m_Device, sizeof(Gaussian3DMoments), totalGaussianCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, gaussianMoments.data()
        );

        // Reset optimization
        m_OptimStep = 0;
    }

    if (!m_SoftmaxBuf.buffer)
    {
        const std::vector<float> softmaxWeights(totalGaussianCount, 1.0f / static_cast<float>(m_GaussianCount));
        m_SoftmaxBuf = CreateInteropBuffer(m_Device, sizeof(float) * totalGaussianCount, Buffer::CpuAccess::None, softmaxWeights.data());
    }

    const size_t totalCausticClusterCount = GetTotalCausticClusterCount();
    if (!m_CausticClusterBuf)
    {
        m_CausticClusterBuf = Buffer::createStructured(
            m_Device, sizeof(float3), totalCausticClusterCount, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
        );
        m_CausticClusterBufCPU =
            Buffer::createStructured(m_Device, sizeof(float3), totalCausticClusterCount, ResourceBindFlags::None, Buffer::CpuAccess::Write);
    }

    if (!m_CausticClusterCountsBuf)
    {
        m_CausticClusterCountsBuf = Buffer::create(
            m_Device, sizeof(uint) * totalCausticClusterCount * lightCount,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
        );
        m_CausticClusterCountsBufCPU = Buffer::create(
            m_Device, sizeof(uint) * totalCausticClusterCount * lightCount, ResourceBindFlags::None, Buffer::CpuAccess::Read
        );
    }
}

bool GaussianPhotonGuiding::RenderUI(Gui::Widgets& widget)
{
    bool changed = false;

    if (auto group = widget.group("PhotonGuiding"))
    {
        changed |= group.checkbox("Use 3D Gaussian Photon Guiding", m_Active);
        group.tooltip("Use 3D Gaussian Photon Guiding for the final gather pass");

        changed |= group.checkbox("Copy Info to CPU", m_CopyToCPU);

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

        if (m_CopyToCPU)
        {
            group.text(
                "First Hit Photons: " + std::to_string(m_ActualFirstHitPhotonCount) + " / " +
                std::to_string(m_MaxFirstHitPhotonCount) + " (" +
                std::to_string(static_cast<float>(m_ActualFirstHitPhotonCount) / static_cast<float>(m_MaxFirstHitPhotonCount)) +
                ")"
            );
        }

        // Optimizer
        const bool changedOptimizer = group.dropdown("Optimizer", m_OptimizerList, reinterpret_cast<uint&>(m_Optimizer));
        if (changedOptimizer)
        {
            m_OptimizationBuf.reset();
        }
        changed |= changedOptimizer;

        group.text("Optimizer step: " + std::to_string(m_OptimStep));
        changed |= group.var("Learning Rate", m_LearningRate, 0.0f, 1.0f);

        if (m_Optimizer == Optimizer::Adam)
        {
            changed |= group.var("Adam Beta1", m_Beta1, 0.0f, 1.0f);
            changed |= group.var("Adam Beta2", m_Beta2, 0.0f, 1.0f);
        }
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
    m_CausticClusters.resize(totalCausticClusterCount);

    std::vector<size_t> indices(m_CausticGeometryInstanceIDs.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::for_each(
        std::execution::par_unseq, indices.begin(), indices.end(),
        [&](const size_t geomInstanceIdx) { GenerateCausticPoints(renderContext, geomInstanceIdx, vertexData, indexData); }
    );

    // Unmap buffers
    vbo->unmap();
    ibo->unmap();

    // Write to CPU buffer
    std::span<float3> cpuCausticClusters(
        reinterpret_cast<float3*>(m_CausticClusterBufCPU->map(Buffer::MapType::WriteDiscard)), m_CausticClusters.size()
    );
    std::copy(m_CausticClusters.begin(), m_CausticClusters.end(), cpuCausticClusters.begin());
    m_CausticClusterBufCPU->unmap();

    // Copy to GPU buffer
    renderContext->copyBufferRegion(
        m_CausticClusterBuf.get(), 0, m_CausticClusterBufCPU.get(), 0, sizeof(float3) * totalCausticClusterCount
    );
    renderContext->uavBarrier(m_CausticClusterBuf.get());
}

void GaussianPhotonGuiding::TrackActualFirstHitPhotonCount(RenderContext* renderContext)
{
    if (m_Active and m_CopyToCPU)
    {
        // Barrier
        renderContext->uavBarrier(m_FirstHitPhotonCountBuf.buffer.get());

        // Copy the first hit photon count to a CPU buffer
        renderContext->copyBufferRegion(m_FirstHitPhotonCountBufCPU.get(), 0, m_FirstHitPhotonCountBuf.buffer.get(), 0, sizeof(uint));
        void* data = m_FirstHitPhotonCountBufCPU->map(Buffer::MapType::Read);
        std::memcpy(&m_ActualFirstHitPhotonCount, data, sizeof(uint));
        m_FirstHitPhotonCountBufCPU->unmap();
    }
}

void GaussianPhotonGuiding::CountCausticClustersPass(RenderContext* pRenderContext, const uint frameCount)
{
    // Profile
    FALCOR_PROFILE(pRenderContext, "CountCausticClusters");

    // Init shader
    if (!m_CountCausticClustersPass)
    {
        Program::Desc desc;
        desc.addShaderModules(m_Scene->getShaderModules());
        desc.addShaderLibrary(m_CountCausticClustersShader).csEntry("main").setShaderModel(m_ShaderModel);
        desc.addTypeConformances(m_Scene->getTypeConformances());

        m_CountCausticClustersPass = ComputePass::create(m_Device, desc, m_Defines, true);
    }
    FALCOR_ASSERT(m_CountCausticClustersPass);

    // Clear
    pRenderContext->clearUAV(m_CausticClusterCountsBuf->getUAV().get(), uint4(0));

    // Set variables
    const size_t lightCount = GetTotalLightCount();
    const size_t clusterCount = GetTotalCausticClusterCount();

    auto var = m_CountCausticClustersPass->getRootVar();
    var["Constants"]["gTotalLightCount"] = lightCount;
    var["Constants"]["gCausticClusterCount"] = clusterCount;
    var["Constants"]["gMaxFirstHitPhotonCount"] = m_MaxFirstHitPhotonCount;

    // Buffers
    var["gCausticClusters"] = m_CausticClusterBuf;
    var["gCausticClusterCounters"] = m_CausticClusterCountsBuf;
    var["gFirstHitCollectionCounts"] = m_FirstHitCollectionCountsBuf.buffer;
    var["gFirstHitPhotonInfos"] = m_FirstHitPhotonInfoBuf.buffer;

    // Execute
    m_CountCausticClustersPass->execute(pRenderContext, uint3(m_MaxFirstHitPhotonCount, 1, 1));

    // Barrier
    pRenderContext->uavBarrier(m_CausticClusterCountsBuf.get());

    // Copy the caustic cluster counts to a CPU buffer
    pRenderContext->copyBufferRegion(
        m_CausticClusterCountsBufCPU.get(), 0, m_CausticClusterCountsBuf.get(), 0, sizeof(uint) * clusterCount * lightCount
    );

    // Weird logic
    if (frameCount < 1)
    {
        return;
    }

    // Access caustic cluster counters on CPU
    const std::span<uint> causticClusterCounts(
        reinterpret_cast<uint*>(m_CausticClusterCountsBufCPU->map(Buffer::MapType::Read)), clusterCount * lightCount
    );

    // Sort and select gaussian
    const float sceneSize = GetSceneSize();
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
            clusterIndices.begin(), clusterIndices.end(),
            [&](const uint idx1, const uint idx2) { return causticClusterCounts[baseIdx + idx1] > causticClusterCounts[baseIdx + idx2]; }
        );

        // Update gaussians
        for (size_t gaussianIdx = 0; gaussianIdx < m_GaussianCount; ++gaussianIdx)
        {
            Gaussian3D& gaussian = gaussians[lightIdx * m_GaussianCount + gaussianIdx];
            gaussian.mean = gaussianIdx < clusterCount ? m_CausticClusters[clusterIndices[gaussianIdx]] : RandomGenerator::Float3();
            gaussian.sigma = sceneSize / 30.0f;
            gaussian.weight = 1.0f;
        }
    }
    m_CausticClusterCountsBufCPU->unmap();

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
        reinterpret_cast<Gaussian3D*>(m_GradientBuf.devicePtr)
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

    // Increase optimizer step
    ++m_OptimStep;

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
    var["gGaussians"] = m_GaussianBuf.buffer;
    var["gGradients"] = m_GradientBuf.buffer;
    var["gLightFirstHitCounts"] = m_LightFirstHitCountsBuf;
    var["gGaussianMoments"] = m_OptimizationBuf;
    var["Constants"]["gGaussianCount"] = m_GaussianCount;
    var["Constants"]["gTotalGaussianCount"] = totalGaussianCount;
    var["Constants"]["gLearningRate"] = m_LearningRate;
    var["Constants"]["gBeta1"] = m_Beta1;
    var["Constants"]["gBeta2"] = m_Beta2;
    var["Constants"]["gOptimStep"] = static_cast<float>(m_OptimStep);

    // More defines
    m_OptimizeGaussiansPass->getProgram()->addDefine("OPTIM_SGD", m_Optimizer == Optimizer::SGD ? "1" : "0");
    m_OptimizeGaussiansPass->getProgram()->addDefine("OPTIM_ADAM", m_Optimizer == Optimizer::Adam ? "1" : "0");

    // Execute
    m_OptimizeGaussiansPass->execute(pRenderContext, uint3(totalGaussianCount, 1, 1));
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
        m_CausticClusters[geomInstanceIdx * m_CausticClusterCount + clusterIdx] = {cluster[0], cluster[1], cluster[2]};
    }
}
