#pragma once

#include "Gaussian3D.h"
#include "FirstHitPhotonInfo.h"

void CalculateGaussianGradient(
    const uint maxFistHitPhotonCount,
    const Gaussian3D* gaussians,
    const uint* firstHitCollectionCounts,
    const FirstHitPhotonInfo* firstHitPhotonInfo,
    const uint* firstHitPhotonCount,
    Gaussian3D* gradients);
