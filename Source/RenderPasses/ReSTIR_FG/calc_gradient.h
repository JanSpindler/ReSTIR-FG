#pragma once

#include "Gaussian3D.h"
#include "FirstHitPhotonInfo.h"

void CalculateGaussianGradient(
    const uint gaussianCount,
    const uint maxFistHitPhotonCount,
    const Gaussian3D* gaussians,
    const uint* firstHitCollectionCounts,
    const FirstHitPhotonInfo* firstHitPhotonInfo,
    const uint* firstHitPhotonCount,
    const float* softmaxWeights,
    Gaussian3D* gradients
);
