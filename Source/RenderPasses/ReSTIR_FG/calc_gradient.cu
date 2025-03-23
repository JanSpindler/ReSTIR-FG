#include "calc_gradient.h"
#include <cuda_runtime.h>

__global__ void CalculateGaussianGradientKernel(
    const uint maxFistHitPhotonCount,
    const Gaussian3D* gaussians,
    const uint* firstHitCollectionCounts,
    const FirstHitPhotonInfo* firstHitPhotonInfo,
    const uint* firstHitPhotonCount,
    Gaussian3D* gradients)
{
}

void CalculateGaussianGradient(
    const uint maxFistHitPhotonCount,
    const Gaussian3D* gaussians,
    const uint* firstHitCollectionCounts,
    const FirstHitPhotonInfo* firstHitPhotonInfo,
    const uint* firstHitPhotonCount,
    Gaussian3D* gradients)
{
}
