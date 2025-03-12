#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import GaussianHelper;

StructuredBuffer<Gaussian3D> gGaussians;
Buffer<uint> gFirstHitPhotonCount;
Buffer<uint> gFirstHitCollectionCounts;
StructuredBuffer<FirstHitPhotonInfo> gFirstHitPhotonInfo;
RWStructuredBuffer<Gaussian3D> gGradients;

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

// TODO: Dont forget to multiply mean and sigma gradients with weight

[numthreads(32, 1, 1)]
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
}
