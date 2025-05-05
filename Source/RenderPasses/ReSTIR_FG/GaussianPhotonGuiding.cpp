#include "GaussianPhotonGuiding.h"
#include "Gaussian3D.h"
#include "RandomGenerator.h"
#include "FirstHitPhotonInfo.h"

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
    //
    m_Scene = pScene;

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
    }

    if (!m_CausticClusterCountsBuf)
    {
        m_CausticClusterCountsBuf = Buffer::create(
            m_Device, sizeof(uint) * totalCausticClusterCount * lightCount,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
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

void GaussianPhotonGuiding::CountCausticClustersPass(RenderContext* pRenderContext)
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

        DefineList defines;
        defines.add(m_Scene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add(getMaterialDefines());

        m_CountCausticClustersPass = ComputePass::create(mpDevice, desc, defines, true);
    }
    FALCOR_ASSERT(mpCountCausticClustersPass);

    // Clear
    pRenderContext->clearUAV(mp3dgCausticClusterCountsBuffer->getUAV().get(), uint4(0));

    // Set variables
    auto var = mpCountCausticClustersPass->getRootVar();
    var["Constants"]["gTotalLightCount"] = m3dgAnalyticLightCount + m3dgGeometricLightCount;
    var["Constants"]["gCausticClusterCount"] = m3dgCausticClusterCount * m3dgCausticGeometryInstanceIDs.size();
    var["Constants"]["gMaxFirstHitPhotonCount"] = m3dgMaxFirstHitPhotonCount;

    // Buffers
    var["gCausticClusters"] = mp3dgCausticClustersBuffer;
    var["gCausticClusterCounters"] = mp3dgCausticClusterCountsBuffer;
    var["gFirstHitCollectionCounts"] = mp3dgFirstHitCollectionCountsBuffer.buffer;
    var["gFirstHitPhotonInfos"] = mp3dgFirstHitPhotonInfoBuffer.buffer;

    // Execute
    mpCountCausticClustersPass->execute(pRenderContext, uint3(m3dgMaxFirstHitPhotonCount, 1, 1));

    // Barrier
    pRenderContext->uavBarrier(mp3dgCausticClusterCountsBuffer.get());

    // Copy the caustic cluster counts to a CPU buffer
    const size_t lightCount = m3dgAnalyticLightCount + m3dgGeometricLightCount;
    const size_t clusterCount = m3dgCausticClusterCount * m3dgCausticGeometryInstanceIDs.size();
    pRenderContext->copyBufferRegion(
        mp3dgCausticClusterCountsBufferCPU.get(), 0, mp3dgCausticClusterCountsBuffer.get(), 0, sizeof(uint) * clusterCount * lightCount
    );

    // Weird logic
    if (mFrameCount < 1)
    {
        return;
    }

    // Access caustic cluster counters on CPU
    const std::span<uint> causticClusterCounts(
        reinterpret_cast<uint*>(mp3dgCausticClusterCountsBufferCPU->map(Buffer::MapType::Read)), clusterCount * lightCount
    );

    // Sort and select gaussian
    const float sceneSize = math::length(mpScene->getSceneBounds().extent());
    const size_t gaussianCount = m3dgGaussianCount * lightCount;

    std::vector<uint> clusterIndices(clusterCount);
    std::vector<Gaussian3D> gaussians(gaussianCount);

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
        for (size_t gaussianIdx = 0; gaussianIdx < m3dgGaussianCount; ++gaussianIdx)
        {
            Gaussian3D& gaussian = gaussians[lightIdx * m3dgGaussianCount + gaussianIdx];
            gaussian.mean = gaussianIdx < clusterCount ? m3dgCausticClusters[clusterIndices[gaussianIdx]] : RandomGenerator::Float3();
            gaussian.sigma = sceneSize / 30.0f;
            gaussian.weight = 1.0f;
        }
    }
    mp3dgCausticClusterCountsBufferCPU->unmap();

    // Copy gaussians to GPU
    ref<Buffer> gaussianBufferCPU = Buffer::createStructured(
        mpDevice, sizeof(Gaussian3D), gaussianCount, ResourceBindFlags::None, Buffer::CpuAccess::Write, gaussians.data()
    );
    pRenderContext->copyBufferRegion(
        mp3dgGaussianBuffer.buffer.get(), 0, gaussianBufferCPU.get(), 0, sizeof(Gaussian3D) * m3dgGaussianCount * lightCount
    );
    pRenderContext->uavBarrier(mp3dgGaussianBuffer.buffer.get());
}

void ReSTIR_FG::calculateGaussianGradientCuda(RenderContext* pRenderContext)
{
    // Profile
    FALCOR_PROFILE(pRenderContext, "CalculateGaussianGradients");

    // Clear buffers
    pRenderContext->clearUAV(mp3dgGradientBuffer.buffer->getUAV().get(), float4(0.0f));

    // Ensure all previous GPU operations are completed
    pRenderContext->flush(true);

    // Calculate gradients
    CalculateGaussianGradient(
        m3dgGaussianCount, m3dgMaxFirstHitPhotonCount, reinterpret_cast<const Gaussian3D*>(mp3dgGaussianBuffer.devicePtr),
        reinterpret_cast<const uint*>(mp3dgFirstHitCollectionCountsBuffer.devicePtr),
        reinterpret_cast<const FirstHitPhotonInfo*>(mp3dgFirstHitPhotonInfoBuffer.devicePtr),
        reinterpret_cast<const uint*>(mp3dgFirstHitPhotonCount.devicePtr), reinterpret_cast<const float*>(mp3dgSoftmaxBuffer.devicePtr),
        reinterpret_cast<Gaussian3D*>(mp3dgGradientBuffer.devicePtr)
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

void ReSTIR_FG::optimizeGaussiansPass(RenderContext* pRenderContext)
{
    // Profile
    FALCOR_PROFILE(pRenderContext, "OptimizeGaussians");

    // Increase optimizer step
    ++m3dgOptimStep;

    // Init shader
    if (!mpOptimizeGaussiansPass)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kOptimizeGaussiansShader).csEntry("main").setShaderModel(kShaderModel);
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add(getMaterialDefines());
        defines.add("OPTIM_SGD", "0");
        defines.add("OPTIM_ADAM", "0");

        mpOptimizeGaussiansPass = ComputePass::create(mpDevice, desc, defines, true);
    }
    FALCOR_ASSERT(mpOptimizeGaussiansPass);

    // Set variables
    const uint totalGaussianCount = m3dgGaussianCount * (m3dgAnalyticLightCount + m3dgGeometricLightCount);
    auto var = mpOptimizeGaussiansPass->getRootVar();
    var["gGaussians"] = mp3dgGaussianBuffer.buffer;
    var["gGradients"] = mp3dgGradientBuffer.buffer;
    var["gLightFirstHitCounts"] = mp3dgLightFirstHitCountBuffer;
    var["gGaussianMoments"] = mp3dgOptimizationBuffer;
    var["Constants"]["gGaussianCount"] = m3dgGaussianCount;
    var["Constants"]["gTotalGaussianCount"] = totalGaussianCount;
    var["Constants"]["gLearningRate"] = m3dgLearningRate;
    var["Constants"]["gBeta1"] = m3dgBeta1;
    var["Constants"]["gBeta2"] = m3dgBeta2;
    var["Constants"]["gOptimStep"] = static_cast<float>(m3dgOptimStep);

    // More defines
    mpOptimizeGaussiansPass->getProgram()->addDefine("OPTIM_SGD", m3dgOptimizer == GaussianOptimizer::SGD ? "1" : "0");
    mpOptimizeGaussiansPass->getProgram()->addDefine("OPTIM_ADAM", m3dgOptimizer == GaussianOptimizer::Adam ? "1" : "0");

    // Execute
    mpOptimizeGaussiansPass->execute(pRenderContext, uint3(totalGaussianCount, 1, 1));
}

void ReSTIR_FG::calculateSoftmaxWeightsPass(RenderContext* pRenderContext)
{
    // Profile
    FALCOR_PROFILE(pRenderContext, "CalculateSoftmaxWeights");

    // Init shader
    if (!mpCalculateSoftmaxWeightsPass)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kCalculateSoftmaxWeightsShader).csEntry("main").setShaderModel(kShaderModel);
        desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());
        defines.add(getMaterialDefines());

        mpCalculateSoftmaxWeightsPass = ComputePass::create(mpDevice, desc, defines, true);
    }
    FALCOR_ASSERT(mpCalculateSoftmaxWeightsPass);

    // Set variables
    auto var = mpCalculateSoftmaxWeightsPass->getRootVar();
    var["Constants"]["gGaussianCount"] = m3dgGaussianCount;
    var["Constants"]["gAnalyticLightCount"] = m3dgAnalyticLightCount;
    var["Constants"]["gGeometricLightCount"] = m3dgGeometricLightCount;

    // Buffers
    var["gGaussians"] = mp3dgGaussianBuffer.buffer;
    var["gSoftmaxWeights"] = mp3dgSoftmaxBuffer.buffer;

    // Execute
    mpCalculateSoftmaxWeightsPass->execute(pRenderContext, uint3(m3dgAnalyticLightCount + m3dgGeometricLightCount, 1, 1));

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
}*/
