#pragma once

#include "falcor_cuda_math.h"

struct Gaussian3D
{
    float3 mean;
    float sigma;
    float weight;

    constexpr Gaussian3D() : mean{0.f, 0.0f, 0.0f}, sigma(1.f), weight(0.f) {}

    constexpr Gaussian3D(const float3& mean, const float sigma, const float weight) : mean(mean), sigma(sigma), weight(weight) {}
};

struct Gaussian3DMoments
{
    float3 meanMoment1;
    float3 meanMoment2;

    float sigmaMoment1;
    float sigmaMoment2;

    float weightMoment1;
    float weightMoment2;

    constexpr Gaussian3DMoments()
        : meanMoment1{0.f, 0.0f, 0.0f}
        , meanMoment2{0.f, 0.0f, 0.0f}
        , sigmaMoment1(0.f)
        , sigmaMoment2(0.f)
        , weightMoment1(0.f)
        , weightMoment2(0.f)
    {}
};
