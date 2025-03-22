#include "calc_gradient.h"
#include <cuda_runtime.h>

__global__ void CalculateGaussianGradientKernel()
{
}

void CalculateGaussianGradient(
    const size_t maxFirstHitPhotonCount,
    const Falcor::ref<Falcor::Buffer> gaussians,
    const Falcor::ref<Falcor::Buffer> firstHitCollectionCounts,
    const Falcor::ref<Falcor::Buffer> firstHitPhotonInfo,
    const Falcor::ref<Falcor::Buffer> gradients)
{
}
