if(NOT FALCOR_HAS_CUDA)
    return()
endif()

add_plugin(ReSTIR_FG)

target_sources(ReSTIR_FG PRIVATE
	ReSTIR_FG.cpp
    ReSTIR_FG.h
    Gaussian3D.h
    FirstHitPhotonInfo.h
    falcor_cuda_math.h
    RandomGenerator.h
    GaussianPhotonGuiding.h
    GaussianPhotonGuiding.cpp
    AdaptiveLightSampler.h
    AdaptiveLightSampler.cpp
    PrefixRestir.h
    PrefixRestir.cpp

    calc_gradient.cu
    calc_gradient.h

    dkm/dkm.hpp
    dkm/dkm_utils.hpp
    dkm/dkm_parallel.hpp

    Shader/CausticResamplingPass.cs.slang
	Shader/CollectPhotons.rt.slang
	Shader/DirectAnalytic.cs.slang
	Shader/FinalGatherData.slang
	Shader/FinalShading.cs.slang
	Shader/GenerateFinalGatherSamples.rt.slang
	Shader/GenerateGIPathSamples.rt.slang
    Shader/GeneratePhotons.rt.slang
	Shader/Hash.slang
	Shader/ResamplingPass.cs.slang
	Shader/Reservoir.slang
	Shader/SurfaceDataFG.slang
	Shader/TraceTransmissionDelta.rt.slang
    Shader/GaussianHelper.slang
    Shader/OptimizeGaussians.cs.slang
	Shader/CalculateSoftmaxWeights.cs.slang
    Shader/CountCausticClusters.cs.slang

    Shader/LoadShadingData.slang
    Shader/Params.slang
    Shader/PathReservoir.slang
    Shader/PathTracer.slang
    Shader/Shift.slang
    Shader/TemporalPathRetrace.cs.slang
    Shader/TemporalReuse.cs.slang
    Shader/StaticParams.slang
)

set_target_properties(ReSTIR_FG PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON  
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

target_copy_shaders(ReSTIR_FG RenderPasses/ReSTIR_FG)

target_source_group(ReSTIR_FG "RenderPasses")
