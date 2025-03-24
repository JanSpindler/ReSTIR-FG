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

static __forceinline__ __device__ bool CheckNumeric(const float3& v)
{
    return CheckNumeric(v.x) && CheckNumeric(v.y) && CheckNumeric(v.z);
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
    return sqrt(denom3);
}

static __forceinline__ __device__ float EvalUnormGaussian3D(const Gaussian3D& gaussian, const float3& position)
{
    const float distance = length(position - gaussian.mean);
    return exp(-0.5 * distance * distance / (gaussian.sigma * gaussian.sigma));
}

static __forceinline__ __device__ float3 DerivNormGaussianWrtMean(const Gaussian3D& gaussian, const float3& position)
{
    const float denom = gaussian.sigma * gaussian.sigma * GaussianNormTerm(gaussian.sigma);
    return (position - gaussian.mean) * EvalUnormGaussian3D(gaussian, position) / denom;
}

static __forceinline__ __device__ float DerivGaussianNormTermWrtSigma(const float sigma)
{
    const float sigma2 = sigma * sigma;
    const float sigma4 = sigma2 * sigma2;
    return -3.0f / (SQRT_8_PI3 * sigma4);
}

static __forceinline__ __device__ float DerivUnormGaussianWrtSigma(const Gaussian3D& gaussian, const float3& position)
{
    const float distance = length(position - gaussian.mean);
    const float distance2 = distance * distance;
    const float sigma3 = gaussian.sigma * gaussian.sigma * gaussian.sigma;
    return distance2 * EvalUnormGaussian3D(gaussian, position) / sigma3;
}

static __forceinline__ __device__ float DerivNormGaussianWrtSigma(const Gaussian3D& gaussian, const float3& position)
{
    return (GaussianNormTerm(gaussian.sigma) * DerivUnormGaussianWrtSigma(gaussian, position)) +
           (DerivGaussianNormTermWrtSigma(gaussian.sigma) * EvalUnormGaussian3D(gaussian, position));
}

static __forceinline__ __device__ void DerivGmm(
    const Gaussian3D* gaussians,
    Gaussian3D* gradients,
    const uint gaussianCount,
    const uint lightIdx,
    const float3& position,
    const float pdfFactor)
{
    // Get first gaussian index
    const uint firstGaussianIdx = lightIdx * gaussianCount;

    // Store softmax weights
    // TODO: Allocate somewhere else and calculate only once before
    static float softmaxWeights[128];
    float softmaxSum = 0.0f;
    for (uint idx = 0; idx < gaussianCount; ++idx)
    {
        const uint gaussianIdx = firstGaussianIdx + idx;
        softmaxSum += exp(gaussians[gaussianIdx].weight);
    }
    for (uint idx = 0; idx < gaussianCount; ++idx)
    {
        const uint gaussianIdx = firstGaussianIdx + idx;
        softmaxWeights[idx] = exp(gaussians[gaussianIdx].weight) / softmaxSum;
    }

    for (uint idx = 0; idx < gaussianCount; ++idx)
    {
        // Mean
        const uint gaussianIdx = firstGaussianIdx + idx;
        const Gaussian3D& gaussian = gaussians[gaussianIdx];
        const float softmaxWeight = softmaxWeights[gaussianIdx];
        const float3 meanDeriv = -1.0f * pdfFactor * softmaxWeight * DerivNormGaussianWrtMean(gaussian, position);
        if (!CheckNumeric(meanDeriv))
        {
            continue;
        }

        // Sigma
        const float sigmaDeriv = -1.0f * pdfFactor * softmaxWeight * DerivNormGaussianWrtSigma(gaussian, position);
        if (!CheckNumeric(sigmaDeriv))
        {
            continue;
        }

        // Weight
        float weightDeriv = 0.0f;
        for (uint otherIdx = 0; otherIdx < gaussianCount; ++otherIdx)
        {
            if (idx == otherIdx)
            {
                weightDeriv += softmaxWeights[idx] * (1.0f - softmaxWeights[idx]);
            }
            else
            {
                weightDeriv += -softmaxWeights[otherIdx] * softmaxWeights[idx];
            }
        }
        weightDeriv *= -1.0f * pdfFactor * EvalUnormGaussian3D(gaussian, position) * GaussianNormTerm(gaussian.sigma);
        if (!CheckNumeric(weightDeriv))
        {
            continue;
        }

        // Add to gradient
        // TODO: Optimize
        atomicAdd(reinterpret_cast<float*>(&gradients[gaussianIdx].mean.x), meanDeriv.x);
        atomicAdd(reinterpret_cast<float*>(&gradients[gaussianIdx].mean.y), meanDeriv.y);
        atomicAdd(reinterpret_cast<float*>(&gradients[gaussianIdx].mean.z), meanDeriv.z);
        atomicAdd(reinterpret_cast<float*>(&gradients[gaussianIdx].sigma), sigmaDeriv);
        atomicAdd(reinterpret_cast<float*>(&gradients[gaussianIdx].weight), weightDeriv);
    }
}

__global__ void CalculateGaussianGradientKernel(
    const uint gaussianCount,
    const uint maxFirstHitPhotonCount,
    const Gaussian3D* gaussians,
    const uint* firstHitCollectionCounts,
    const FirstHitPhotonInfo* firstHitPhotonInfo,
    const uint* firstHitPhotonCount,
    Gaussian3D* gradients)
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
    const float3& position = firstHitPhotonInfo[firstHitPhotonIdx].pos;
    DerivGmm(
        gaussians,
        gradients,
        gaussianCount,
        lightIdx,
        position,
        pdfFactor);
}

void CalculateGaussianGradient(
    const uint gaussianCount,
    const uint maxFistHitPhotonCount,
    const Gaussian3D* gaussians,
    const uint* firstHitCollectionCounts,
    const FirstHitPhotonInfo* firstHitPhotonInfo,
    const uint* firstHitPhotonCount,
    Gaussian3D* gradients)
{
    CalculateGaussianGradientKernel<<<(maxFistHitPhotonCount + 127) / 128, 128>>>(
        gaussianCount,
        maxFistHitPhotonCount,
        gaussians,
        firstHitCollectionCounts,
        firstHitPhotonInfo,
        firstHitPhotonCount,
        gradients);
}
