#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import GaussianHelper;

cbuffer Constants
{
    uint gGaussianCount;
}

StructuredBuffer<Gaussian3D> gGaussians;
Buffer<uint> gFirstHitPhotonCount;
Buffer<uint> gFirstHitCollectionCounts;
StructuredBuffer<FirstHitPhotonInfo> gFirstHitPhotonInfo;
RWBuffer<int> gGradients;

static const float SQRT_8_PI3 = 15.7496099457224197f;

// TODO: Optimize gradient calculation for speed
// TODO: Double check correctness
float3 DerivNormGaussianWrtMean(const in Gaussian3D gaussian, const float3 position)
{
    const float denom = gaussian.sigma * gaussian.sigma * GaussianNormTerm(gaussian.sigma);
    return (position - gaussian.mean) * EvalUnormGaussian3D(gaussian, position) / denom;
}

float DerivGaussianNormTermWrtSigma(const float sigma)
{
    const float sigma2 = sigma * sigma;
    const float sigma4 = sigma2 * sigma2;
    return -3.0f / (SQRT_8_PI3 * sigma4);
}

float DerivUnormGaussianWrtSigma(const in Gaussian3D gaussian, const float3 position)
{
    const float distance = length(position - gaussian.mean);
    const float distance2 = distance * distance;
    const float sigma3 = gaussian.sigma * gaussian.sigma * gaussian.sigma;
    return distance2 * EvalUnormGaussian3D(gaussian, position) / sigma3;
}

float DerivNormGaussianWrtSigma(const in Gaussian3D gaussian, const float3 position)
{
    return
        (GaussianNormTerm(gaussian.sigma) * DerivUnormGaussianWrtSigma(gaussian, position)) +
        (DerivGaussianNormTermWrtSigma(gaussian.sigma) * EvalUnormGaussian3D(gaussian, position));
}

void DerivGmm(const uint lightIdx, const float3 position, const float pdfFactor)
{
    // Get first gaussian index
    const uint firstGaussianIdx = lightIdx * gGaussianCount;

    // Store softmax weights
    // TODO: Allocate somewhere else and calculate only once before
    static float softmaxWeights[128];
    float softmaxSum = 0.0f;
    for (uint idx = 0; idx < gGaussianCount; ++idx)
    {
        const uint gaussianIdx = firstGaussianIdx + idx;
        softmaxSum += exp(gGaussians[gaussianIdx].weight);
    }
    for (uint idx = 0; idx < gGaussianCount; ++idx)
    {
        const uint gaussianIdx = firstGaussianIdx + idx;
        softmaxWeights[idx] = exp(gGaussians[gaussianIdx].weight) / softmaxSum;
    }

    for (uint idx = 0; idx < gGaussianCount; ++idx)
    {
        // Mean
        const uint gaussianIdx = firstGaussianIdx + idx;
        const Gaussian3D gaussian = gGaussians[gaussianIdx];
        const float softmaxWeight = softmaxWeights[gaussianIdx];
        const float3 meanDeriv = -1.0f * pdfFactor * softmaxWeight * DerivNormGaussianWrtMean(gaussian, position);

        // Sigma
        const float sigmaDeriv = -1.0f * pdfFactor * softmaxWeight * DerivNormGaussianWrtSigma(gaussian, position);

        // Weight
        float weightDeriv = 0.0f;
        for (uint otherIdx = 0; otherIdx < gGaussianCount; ++otherIdx)
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

        // Add to gradient
        // Need to use integer since InterlockedAdd is not possible with floats
        // TODO: Optimize
        const uint gradientIdx = gaussianIdx * 5;
        InterlockedAdd(gGradients[gradientIdx + 0], QuantizeGradient(meanDeriv.x));
        InterlockedAdd(gGradients[gradientIdx + 1], QuantizeGradient(meanDeriv.y));
        InterlockedAdd(gGradients[gradientIdx + 2], QuantizeGradient(meanDeriv.z));
        InterlockedAdd(gGradients[gradientIdx + 3], QuantizeGradient(sigmaDeriv));
        InterlockedAdd(gGradients[gradientIdx + 4], QuantizeGradient(weightDeriv));
    }
}

[numthreads(128, 1, 1)]
void main(uint2 index : SV_DispatchThreadID)
{
    // Get first hit photon index
    const uint firstHitPhotonIdx = index.x;
    if (firstHitPhotonIdx >= gFirstHitPhotonCount[0])
    {
        return;
    }

    // Calculate pdf factor
    const float targetPdf = (float)gFirstHitCollectionCounts[firstHitPhotonIdx];
    const float samplingPdf = gFirstHitPhotonInfo[firstHitPhotonIdx].samplingPdf;
    const float pdfFactor = targetPdf / samplingPdf;

    // Calculate gradient wrt. parameters of gaussian
    const uint lightIdx = gFirstHitPhotonInfo[firstHitPhotonIdx].lightIdx;
    const float3 position = gFirstHitPhotonInfo[firstHitPhotonIdx].pos;
    DerivGmm(lightIdx, position, pdfFactor);
}
