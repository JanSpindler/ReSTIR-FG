#pragma once

#include "falcor_cuda_math.h"

struct Gaussian3D
{
    float3 mean;
    float pSigma;
    float weight;

    constexpr Gaussian3D() : mean{0.f, 0.0f, 0.0f}, pSigma(0.f), weight(0.f) {}

    constexpr Gaussian3D(const float3& mean, const float pSigma, const float weight) : mean(mean), pSigma(pSigma), weight(weight) {}

#ifdef __CUDACC__
    __forceinline__ __device__ float GetSigma(const float cS) const
    {
        return 0.1f + (cS / (1.0f + exp(-pSigma)));
    }

    __forceinline__ __device__ float GetSigmaDeriv(const float cS) const
    {
        const float expNegP = exp(-pSigma);    // e^(-pSigma)
        const float denom = 1.0f + expNegP;    // 1 + e^(-pSigma)
        return cS * expNegP / (denom * denom); // cS * e^(-pSigma) / (1 + e^(-pSigma))²
    }
#endif
};

struct Gaussian3DOptimizationData
{
    float3 meanMoment1;
    float3 meanMoment2;

    float pSigmaMoment1;
    float pSigmaMoment2;

    float weightMoment1;
    float weightMoment2;

    uint step;

    constexpr Gaussian3DOptimizationData()
        : meanMoment1{0.f, 0.0f, 0.0f}
        , meanMoment2{0.f, 0.0f, 0.0f}
        , pSigmaMoment1(0.f)
        , pSigmaMoment2(0.f)
        , weightMoment1(0.f)
        , weightMoment2(0.f)
        , step(0)
    {}
};
