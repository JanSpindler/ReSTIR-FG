#pragma once

#include "Falcor.h"

void CalculateGaussianGradient(
    const size_t maxFirstHitPhotonCount,
    const Falcor::ref<Falcor::Buffer> gaussians,
    const Falcor::ref<Falcor::Buffer> firstHitCollectionCounts,
    const Falcor::ref<Falcor::Buffer> firstHitPhotonInfo,
    const Falcor::ref<Falcor::Buffer> gradients);
