#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import GaussianHelper;

cbuffer Constants
{
    uint gGaussianCount;
    uint gTotalGaussianCount;

    float gLearningRate;
    float gBeta1;
    float gBeta2;
    float gOptimStep;
}

static const float EPSILON = 1e-8f;

RWStructuredBuffer<Gaussian3D> gGaussians;
StructuredBuffer<Gaussian3D> gGradients;
Buffer<uint> gLightFirstHitCounts;
RWStructuredBuffer<Gaussian3DMoments> gGaussianMoments;

bool CheckNumeric(const float value)
{
    return isfinite(value) && !isnan(value);
}

bool CheckNumeric(const float3 value)
{
    return all(isfinite(value) && !isnan(value));
}

float Adam(const float g, inout float m, inout float v)
{
    m = gBeta1 * m + (1.0f - gBeta1) * g;
    v = gBeta2 * v + (1.0f - gBeta2) * g * g;
    const float numerator = m / (1.0f - pow(gBeta1, gOptimStep));
    const float denominator = sqrt(v / (1.0f - pow(gBeta2, gOptimStep))) + EPSILON;
    return numerator / denominator;
}

void OptimAdam(inout float param, const float gradient, inout float firstMoment, inout float secondMoment)
{
    param -= gLearningRate * Adam(gradient, firstMoment, secondMoment);
}

[numthreads(32, 1, 1)]
void main(uint2 index : SV_DispatchThreadID)
{
    // Get gaussian index
    const uint gaussianIdx = index.x;
    if (gaussianIdx >= gTotalGaussianCount)
    {
        return;
    }

    // Get light first hit count
    const uint lightIdx = (gaussianIdx - (gaussianIdx % gGaussianCount)) / gGaussianCount;
    const float lightNormFactor = 1.0f / (float)gLightFirstHitCounts[lightIdx];

    // Get the gradient
    const Gaussian3D gradient = gGradients[gaussianIdx];
    const float3 meanGradient = gradient.mean * lightNormFactor;
    const float sigmaGradient = gradient.sigma * lightNormFactor;
    const float weightGradient = gradient.weight * lightNormFactor;
    if (!CheckNumeric(meanGradient) || !CheckNumeric(sigmaGradient) || !CheckNumeric(weightGradient))
    {
        return;
    }

    // Optimizer
#if OPTIM_SGD
    gGaussians[gaussianIdx].mean -= gLearningRate * meanGradient;
    gGaussians[gaussianIdx].sigma -= gLearningRate * sigmaGradient;
    gGaussians[gaussianIdx].weight -= gLearningRate * weightGradient;
#endif
#if OPTIM_ADAM
    OptimAdam(gGaussians[gaussianIdx].mean.x, meanGradient.x, gGaussianMoments[gaussianIdx].meanMoment1.x, gGaussianMoments[gaussianIdx].meanMoment2.x);
    OptimAdam(gGaussians[gaussianIdx].mean.y, meanGradient.y, gGaussianMoments[gaussianIdx].meanMoment1.y, gGaussianMoments[gaussianIdx].meanMoment2.y);
    OptimAdam(gGaussians[gaussianIdx].mean.z, meanGradient.z, gGaussianMoments[gaussianIdx].meanMoment1.z, gGaussianMoments[gaussianIdx].meanMoment2.z);
    OptimAdam(gGaussians[gaussianIdx].sigma, sigmaGradient, gGaussianMoments[gaussianIdx].sigmaMoment1, gGaussianMoments[gaussianIdx].sigmaMoment2);
    OptimAdam(gGaussians[gaussianIdx].weight, weightGradient, gGaussianMoments[gaussianIdx].weightMoment1, gGaussianMoments[gaussianIdx].weightMoment2);
#endif

    // Safety checks
    // TODO: Handle this in a better way
    gGaussians[gaussianIdx].sigma = max(gGaussians[gaussianIdx].sigma, 0.5f);
    gGaussians[gaussianIdx].weight = clamp(gGaussians[gaussianIdx].weight, -1.0f, 1.0f);
}
