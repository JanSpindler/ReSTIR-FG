#include "calc_gradient.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// TODO: Optimize gradient calculation for speed
// TODO: Double check correctness

using uint = uint32_t; // For syntax highlighting

static constexpr float SQRT_8_PI3 = 15.7496099457224197f;
static constexpr float PI = 3.14159265358979323846f;

static __forceinline__ __device__ bool CheckNumeric(const float x)
{
    return !isnan(x) && !isinf(x);
}

static __forceinline__ __device__ float length(const float3& v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

static __forceinline__ __device__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __device__ float3 operator*(const float3& a, const float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

static __forceinline__ __device__ float3 operator*(const float a, const float3& b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}

static __forceinline__ __device__ float3 operator/(const float3& a, const float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

static __forceinline__ __device__ float GaussianNormTerm(const float sigma)
{
    const float denom = 2.0f * PI * sigma * sigma;
    const float denom3 = denom * denom * denom;
    return 1.0f / sqrt(denom3);
}

static __forceinline__ __device__ float EvalUnormGaussian3D(const Gaussian3D& gaussian, const float3& position, const float cS)
{
    const float distance = length(position - gaussian.mean);
    const float sigma = gaussian.GetSigma(cS);
    return exp(-0.5 * distance * distance / (sigma * sigma));
}

static __forceinline__ __device__ float3 DerivNormGaussianWrtMean(const Gaussian3D& gaussian, const float3& position, const float cS)
{
    const float sigma = gaussian.GetSigma(cS);
    const float denom = sigma * sigma;
    return GaussianNormTerm(sigma) * (position - gaussian.mean) * EvalUnormGaussian3D(gaussian, position, cS) / denom;
}

static __forceinline__ __device__ float DerivGaussianNormTermWrtSigma(const float sigma)
{
    const float sigma2 = sigma * sigma;
    const float sigma4 = sigma2 * sigma2;
    return -3.0f / (SQRT_8_PI3 * sigma4);
}

static __forceinline__ __device__ float DerivUnormGaussianWrtSigma(const Gaussian3D& gaussian, const float3& position, const float cS)
{
    const float distance = length(position - gaussian.mean);
    const float distance2 = distance * distance;
    const float sigma = gaussian.GetSigma(cS);
    const float sigma3 = sigma * sigma * sigma;
    return distance2 * EvalUnormGaussian3D(gaussian, position, cS) / sigma3;
}

static __forceinline__ __device__ float DerivNormGaussianWrtSigma(const Gaussian3D& gaussian, const float3& position, const float cS)
{
    const float sigma = gaussian.GetSigma(cS);
    return (GaussianNormTerm(sigma) * DerivUnormGaussianWrtSigma(gaussian, position, cS)) +
           (DerivGaussianNormTermWrtSigma(sigma) * EvalUnormGaussian3D(gaussian, position, cS));
}

static __forceinline__ __device__ void NumericAtomicAdd(float* dest, const float value)
{
    if (CheckNumeric(value))
    {
        atomicAdd(dest, value);
    }
}

// Warp-level reduction for float values
static __forceinline__ __device__ float WarpReduceSum(float val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Optimized version using warp-level reduction
static __forceinline__ __device__ void WarpAtomicAdd(float* dest, float value)
{
    if (!CheckNumeric(value))
    {
        value = 0.0f;
    }

    // Get the lane ID and warp size
    const int laneId = threadIdx.x & 31;

    // Reduce within the warp
    value = WarpReduceSum(value);

    // Only the first thread in the warp performs the atomic add
    if (laneId == 0 && value != 0.0f)
    {
        atomicAdd(dest, value);
    }
}

static __forceinline__ __device__ void DerivGmm(
    const Gaussian3D* gaussians,
    Gaussian3D* gradients,
    const float* softmaxWeights,
    const uint gaussianCount,
    const uint lightIdx,
    const float3& position,
    const float pdfFactor,
    const float cS
)
{
    // Get first gaussian index
    const uint firstGaussianIdx = lightIdx * gaussianCount;

    // Add to gradient
    for (uint idx = 0; idx < gaussianCount; ++idx)
    {
        // Overall
        const uint gaussianIdx = firstGaussianIdx + idx;
        const Gaussian3D& gaussian = gaussians[gaussianIdx];
        const float softmaxWeight = softmaxWeights[gaussianIdx];

        // Mean
        const float3 meanDeriv = pdfFactor * softmaxWeight * DerivNormGaussianWrtMean(gaussian, position, cS);

        // Sigma
        const float pSigmaDeriv =
            pdfFactor * softmaxWeight * DerivNormGaussianWrtSigma(gaussian, position, cS) * gaussian.GetSigmaDeriv(cS);

        // Weight
        float weightDeriv = 0.0f;
        for (uint otherIdx = 0; otherIdx < gaussianCount; ++otherIdx)
        {
            const Gaussian3D& otherGaussian = gaussians[firstGaussianIdx + otherIdx];

            const float gaussianFactor = EvalUnormGaussian3D(otherGaussian, position, cS) * GaussianNormTerm(otherGaussian.GetSigma(cS));
            const float softmaxDerivFactor =
                idx == otherIdx ? softmaxWeights[idx] * (1.0f - softmaxWeights[idx]) : -softmaxWeights[otherIdx] * softmaxWeights[idx];
            weightDeriv += softmaxDerivFactor * gaussianFactor;
        }
        weightDeriv *= pdfFactor;

        // Add gradient using warp-level reduction for better performance
#if 1
        WarpAtomicAdd(&gradients[gaussianIdx].mean.x, -meanDeriv.x);
        WarpAtomicAdd(&gradients[gaussianIdx].mean.y, -meanDeriv.y);
        WarpAtomicAdd(&gradients[gaussianIdx].mean.z, -meanDeriv.z);
        WarpAtomicAdd(&gradients[gaussianIdx].pSigma, -pSigmaDeriv);
        WarpAtomicAdd(&gradients[gaussianIdx].weight, -weightDeriv);
#else
        NumericAtomicAdd(&gradients[gaussianIdx].mean.x, -meanDeriv.x);
        NumericAtomicAdd(&gradients[gaussianIdx].mean.y, -meanDeriv.y);
        NumericAtomicAdd(&gradients[gaussianIdx].mean.z, -meanDeriv.z);
        NumericAtomicAdd(&gradients[gaussianIdx].pSigma, -pSigmaDeriv);
        NumericAtomicAdd(&gradients[gaussianIdx].weight, -weightDeriv);
#endif
    }
}

__global__ void CalculateGaussianGradientKernel(
    const uint gaussianCount,
    const uint maxFirstHitPhotonCount,
    const Gaussian3D* gaussians,
    const uint* firstHitCollectionCounts,
    const FirstHitPhotonInfo* firstHitPhotonInfo,
    const uint* firstHitPhotonCount,
    const float* softmaxWeights,
    Gaussian3D* gradients,
    const float positionScaling,
    const float cS
)
{
    // Get first hit photon index
    const uint firstHitPhotonIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (firstHitPhotonIdx >= min(maxFirstHitPhotonCount, firstHitPhotonCount[0]))
    {
        return;
    }

    // Calculate pdf factor
    const float targetPdf = static_cast<float>(firstHitCollectionCounts[firstHitPhotonIdx]);
    const float samplingPdf = firstHitPhotonInfo[firstHitPhotonIdx].samplingPdf;
    if (targetPdf <= 0.0f || samplingPdf <= 0.0f)
    {
        return;
    }
    const float pdfFactor = targetPdf / samplingPdf;
    if (!CheckNumeric(pdfFactor))
    {
        return;
    }

    // Calculate gradient wrt. parameters of gaussian
    const uint lightIdx = firstHitPhotonInfo[firstHitPhotonIdx].lightIdx;
    const float3 position = firstHitPhotonInfo[firstHitPhotonIdx].pos * positionScaling;
    DerivGmm(gaussians, gradients, softmaxWeights, gaussianCount, lightIdx, position, pdfFactor, cS);
}

void CalculateGaussianGradient(
    const uint gaussianCount,
    const uint maxFistHitPhotonCount,
    const Gaussian3D* gaussians,
    const uint* firstHitCollectionCounts,
    const FirstHitPhotonInfo* firstHitPhotonInfo,
    const uint* firstHitPhotonCount,
    const float* softmaxWeights,
    Gaussian3D* gradients,
    const float positionScaling,
    const float cS
)
{
    static constexpr size_t blockSize = 128;
    CalculateGaussianGradientKernel<<<(maxFistHitPhotonCount + blockSize - 1) / blockSize, blockSize>>>(
        gaussianCount, maxFistHitPhotonCount, gaussians, firstHitCollectionCounts, firstHitPhotonInfo, firstHitPhotonCount, softmaxWeights,
        gradients, positionScaling, cS
    );
}
