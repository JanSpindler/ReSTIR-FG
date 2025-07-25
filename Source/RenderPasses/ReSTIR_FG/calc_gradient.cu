#include "calc_gradient.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// TODO: Optimize gradient calculation for speed
// TODO: Double check correctness

using uint = uint32_t; // For syntax highlighting

static constexpr float PI = 3.14159265358979323846f;
static constexpr float SQRT_2PI_CUBED = 15.74960994572241974429064599; // sqrtf((2.0f * PI) * (2.0f * PI) * (2.0f * PI)); // √((2π)³)
static constexpr float INV_SQRT_2PI_CUBED = 1.0f / SQRT_2PI_CUBED;

static __forceinline__ __device__ bool CheckNumeric(const float x)
{
    return !isnan(x) && !isinf(x);
}

static __forceinline__ __device__ float length(const float3& v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

static __forceinline__ __device__ float lengthSquared(const float3& v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
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

// Fast reciprocal square root that works across all compute capabilities
static __forceinline__ __device__ float fast_rsqrtf(float x)
{
#if __CUDA_ARCH__ >= 200
    return rsqrtf(x); // Use hardware instruction if available
#else
    return 1.0f / sqrtf(x); // Fallback for older architectures
#endif
}

// Optimized Gaussian evaluation with precomputed terms
struct GaussianTerms
{
    float sigma;
    float sigmaDeriv;
    float invSigma2;
    float invSigma3;
    float normTerm;
    float unormGaussian;
};

static __forceinline__ __device__ GaussianTerms ComputeGaussianTerms(const Gaussian3D& gaussian, const float3& position, const float cS)
{
    GaussianTerms terms;

    // Compute sigma and its derivative efficiently
    const float expNegP = __expf(-gaussian.pSigma); // Fast exponential
    const float denom = 1.0f + expNegP;
    terms.sigma = Gaussian3D::MIN_SIGMA + (cS / denom);
    terms.sigmaDeriv = cS * expNegP / (denom * denom);

    // Precompute powers
    const float sigma2 = terms.sigma * terms.sigma;
    terms.invSigma2 = 1.0f / sigma2;
    terms.invSigma3 = terms.invSigma2 / terms.sigma;

    // Compute normalization term using fast math (fixed for compatibility)
    const float sigma6 = sigma2 * sigma2 * sigma2;
    terms.normTerm = fast_rsqrtf((2.0f * PI) * sigma6); // Compatible fast reciprocal sqrt

    // Compute unnormalized Gaussian
    const float distanceSquared = lengthSquared(position - gaussian.mean);
    terms.unormGaussian = __expf(-0.5f * distanceSquared * terms.invSigma2);

    return terms;
}

// Optimized DerivGmm for better register usage
static __forceinline__ __device__ void DerivGmm_Optimized(
    const Gaussian3D* __restrict__ gaussians,
    Gaussian3D* __restrict__ gradients,
    const float* __restrict__ softmaxWeights,
    const uint gaussianCount,
    const uint lightIdx,
    const float3& position,
    const float pdfFactor,
    const float cS
)
{
    const uint firstGaussianIdx = lightIdx * gaussianCount;

    // For smaller Gaussian counts, use a more cache-friendly approach
    if (gaussianCount <= 16)
    {
        // First pass: precompute all Gaussian terms
        GaussianTerms allTerms[16]; // Fixed size array for better register allocation

        for (uint idx = 0; idx < gaussianCount; ++idx)
        {
            const uint gaussianIdx = firstGaussianIdx + idx;
            const Gaussian3D& gaussian = gaussians[gaussianIdx];
            allTerms[idx] = ComputeGaussianTerms(gaussian, position, cS);
        }

        // Second pass: compute gradients using cached terms
        for (uint idx = 0; idx < gaussianCount; ++idx)
        {
            const uint gaussianIdx = firstGaussianIdx + idx;
            const Gaussian3D& gaussian = gaussians[gaussianIdx];
            const float softmaxWeight = softmaxWeights[gaussianIdx];
            const GaussianTerms& terms = allTerms[idx];

            // Mean gradient - optimized computation
            const float3 positionDiff = position - gaussian.mean;
            const float3 meanDeriv = (pdfFactor * softmaxWeight * terms.normTerm * terms.unormGaussian * terms.invSigma2) * positionDiff;

            // Sigma gradient - combine terms efficiently
            const float distanceSquared = lengthSquared(positionDiff);
            const float derivUnormGaussian = 0.5f * distanceSquared * terms.unormGaussian * terms.invSigma3; // Add missing 0.5f factor
            const float sigma4 = terms.sigma * terms.sigma * terms.sigma * terms.sigma;
            const float normTermDeriv = -3.0f * INV_SQRT_2PI_CUBED / sigma4; // Use correct normalization constant
            const float pSigmaDeriv =
                pdfFactor * softmaxWeight * (terms.normTerm * derivUnormGaussian + normTermDeriv * terms.unormGaussian) * terms.sigmaDeriv;

            // Weight gradient using cached terms
            float weightDeriv = 0.0f;
            for (uint otherIdx = 0; otherIdx < gaussianCount; ++otherIdx)
            {
                const float gaussianFactor = allTerms[otherIdx].unormGaussian * allTerms[otherIdx].normTerm;
                const float otherWeight = softmaxWeights[firstGaussianIdx + otherIdx];
                const float softmaxDerivFactor = (idx == otherIdx) ? softmaxWeight * (1.0f - softmaxWeight) : -otherWeight * softmaxWeight;
                weightDeriv += softmaxDerivFactor * gaussianFactor;
            }
            weightDeriv *= pdfFactor;

            // Accumulate gradients using atomic operations (no warp reduction since each thread handles different light sources)
            if (CheckNumeric(meanDeriv.x))
                atomicAdd(&gradients[gaussianIdx].mean.x, -meanDeriv.x);
            if (CheckNumeric(meanDeriv.y))
                atomicAdd(&gradients[gaussianIdx].mean.y, -meanDeriv.y);
            if (CheckNumeric(meanDeriv.z))
                atomicAdd(&gradients[gaussianIdx].mean.z, -meanDeriv.z);
            if (CheckNumeric(pSigmaDeriv))
                atomicAdd(&gradients[gaussianIdx].pSigma, -pSigmaDeriv);
            if (CheckNumeric(weightDeriv))
                atomicAdd(&gradients[gaussianIdx].weight, -weightDeriv);
        }
    }
    else
    {
        // For larger counts, use streaming approach to manage register pressure
        for (uint idx = 0; idx < gaussianCount; ++idx)
        {
            const uint gaussianIdx = firstGaussianIdx + idx;
            const Gaussian3D& gaussian = gaussians[gaussianIdx];
            const GaussianTerms terms = ComputeGaussianTerms(gaussian, position, cS);
            const float softmaxWeight = softmaxWeights[gaussianIdx];

            // Mean gradient
            const float3 positionDiff = position - gaussian.mean;
            const float3 meanDeriv = (pdfFactor * softmaxWeight * terms.normTerm * terms.unormGaussian * terms.invSigma2) * positionDiff;

            // Sigma gradient
            const float distanceSquared = lengthSquared(positionDiff);
            const float derivUnormGaussian = 0.5f * distanceSquared * terms.unormGaussian * terms.invSigma3; // Add missing 0.5f factor
            const float sigma4 = terms.sigma * terms.sigma * terms.sigma * terms.sigma;
            const float normTermDeriv = -3.0f * INV_SQRT_2PI_CUBED / sigma4; // Use correct normalization constant
            const float pSigmaDeriv =
                pdfFactor * softmaxWeight * (terms.normTerm * derivUnormGaussian + normTermDeriv * terms.unormGaussian) * terms.sigmaDeriv;

            // Weight gradient - compute on demand to save registers
            float weightDeriv = 0.0f;
            for (uint otherIdx = 0; otherIdx < gaussianCount; ++otherIdx)
            {
                const GaussianTerms otherTerms = ComputeGaussianTerms(gaussians[firstGaussianIdx + otherIdx], position, cS);
                const float gaussianFactor = otherTerms.unormGaussian * otherTerms.normTerm;
                const float otherWeight = softmaxWeights[firstGaussianIdx + otherIdx];
                const float softmaxDerivFactor = (idx == otherIdx) ? softmaxWeight * (1.0f - softmaxWeight) : -otherWeight * softmaxWeight;
                weightDeriv += softmaxDerivFactor * gaussianFactor;
            }
            weightDeriv *= pdfFactor;

            // Accumulate gradients using atomic operations (no warp reduction since each thread handles different light sources)
            if (CheckNumeric(meanDeriv.x))
                atomicAdd(&gradients[gaussianIdx].mean.x, -meanDeriv.x);
            if (CheckNumeric(meanDeriv.y))
                atomicAdd(&gradients[gaussianIdx].mean.y, -meanDeriv.y);
            if (CheckNumeric(meanDeriv.z))
                atomicAdd(&gradients[gaussianIdx].mean.z, -meanDeriv.z);
            if (CheckNumeric(pSigmaDeriv))
                atomicAdd(&gradients[gaussianIdx].pSigma, -pSigmaDeriv);
            if (CheckNumeric(weightDeriv))
                atomicAdd(&gradients[gaussianIdx].weight, -weightDeriv);
        }
    }
}

__global__ void CalculateGaussianGradientKernel(
    const uint gaussianCount,
    const uint maxFirstHitPhotonCount,
    const Gaussian3D* __restrict__ gaussians,
    const uint* __restrict__ firstHitCollectionCounts,
    const FirstHitPhotonInfo* __restrict__ firstHitPhotonInfo,
    const uint* __restrict__ firstHitPhotonCount,
    const float* __restrict__ softmaxWeights,
    Gaussian3D* __restrict__ gradients,
    const float positionScaling,
    const float cS
)
{
    const uint firstHitPhotonIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (firstHitPhotonIdx >= min(maxFirstHitPhotonCount, firstHitPhotonCount[0]))
    {
        return;
    }

    // Load photon data into registers
    const FirstHitPhotonInfo photon = firstHitPhotonInfo[firstHitPhotonIdx];
    const float targetPdf = static_cast<float>(firstHitCollectionCounts[firstHitPhotonIdx]);

    // Early exit for invalid samples
    if (targetPdf <= 0.0f || photon.samplingPdf <= 0.0f)
    {
        return;
    }

    const float pdfFactor = targetPdf / photon.samplingPdf;
    if (!CheckNumeric(pdfFactor))
    {
        return;
    }

    const float3 position = photon.pos * positionScaling;

    // Use the optimized function for all Gaussian counts
    DerivGmm_Optimized(gaussians, gradients, softmaxWeights, gaussianCount, photon.lightIdx, position, pdfFactor, cS);
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
    // Optimize block size based on Gaussian count and register usage
    size_t blockSize;
    if (gaussianCount <= 8)
    {
        blockSize = 256; // Higher occupancy for small Gaussian counts
    }
    else if (gaussianCount <= 16)
    {
        blockSize = 128; // Balanced approach
    }
    else
    {
        blockSize = 64; // Lower occupancy but manageable register usage for large counts
    }

    const size_t numBlocks = (maxFistHitPhotonCount + blockSize - 1) / blockSize;

    CalculateGaussianGradientKernel<<<numBlocks, blockSize>>>(
        gaussianCount, maxFistHitPhotonCount, gaussians, firstHitCollectionCounts, firstHitPhotonInfo, firstHitPhotonCount, softmaxWeights,
        gradients, positionScaling, cS
    );
}
