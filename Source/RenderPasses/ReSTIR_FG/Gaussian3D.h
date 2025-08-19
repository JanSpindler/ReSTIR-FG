#pragma once

#include "falcor_cuda_math.h"

struct Gaussian3D
{
    static constexpr float MIN_SIGMA = 0.0f;

    static float DistanceFromPSigma(const float pSigma, const float cS) { return MIN_SIGMA + (cS / (1.0f + exp(-pSigma))); }

    static float PSigmaFromDistance(const float distance, const float cS)
    {
        constexpr float saturation = 10.0f;
        if (distance > DistanceFromPSigma(saturation, cS))
        {
            return saturation;
        }
        else if (distance < DistanceFromPSigma(-saturation, cS))
        {
            return -saturation;
        }
        return log((distance - MIN_SIGMA) * (1.0f + exp(-cS)) / cS); // TODO: Check if correct
    }

    float3 mean;
    float pSigma;
    float weight;

    constexpr Gaussian3D() : mean{0.f, 0.0f, 0.0f}, pSigma(0.f), weight(0.f) {}

    constexpr Gaussian3D(const float3& mean, const float pSigma, const float weight) : mean(mean), pSigma(pSigma), weight(weight) {}

#ifdef __CUDACC__
    __forceinline__ __device__ float GetSigma(const float cS) const { return MIN_SIGMA + (cS / (1.0f + __expf(-pSigma))); }

    __forceinline__ __device__ float GetSigmaDeriv(const float cS) const
    {
        const float sigma = GetSigma(cS);
        return cS * sigma * (1.0f - sigma);
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
