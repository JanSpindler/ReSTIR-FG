if(NOT FALCOR_HAS_CUDA)
    return()
endif()

add_plugin(ReSTIR_FG)

target_sources(ReSTIR_FG PRIVATE
	ReSTIR_FG.cpp
    ReSTIR_FG.h
    FalcorCUDA.cpp
    FalcorCUDA.h

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
    Shader/CalculateGaussianGradient.cs.slang
    Shader/OptimizeGaussians.cs.slang
	Shader/Printing.slang
    Shader/gpu-printing-ops.h
)

set_target_properties(ReSTIR_FG PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_copy_shaders(ReSTIR_FG RenderPasses/ReSTIR_FG)

target_source_group(ReSTIR_FG "RenderPasses")
