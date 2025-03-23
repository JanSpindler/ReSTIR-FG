#pragma once

#include "falcor_cuda_math.h"

struct Gaussian3D
{
    float3 mean;
    float sigma;
    float weight;

    constexpr struct Gaussian3D() : mean{0.f, 0.0f, 0.0f}, sigma(1.f), weight(0.f) {}

    constexpr struct Gaussian3D(const float3& mean, const float sigma, const float weight) : mean(mean), sigma(sigma), weight(weight) {}
};
