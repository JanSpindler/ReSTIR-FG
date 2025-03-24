#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import GaussianHelper;

cbuffer Constants
{
    uint gTotalGaussianCount;
}

static const float gLearningRate = 1e-4f;
static const float gBeta1 = 0.9f;
static const float gBeta2 = 0.999f;
static const float gOptimTimeStep = 2.0f;
static const float EPSILON = 1e-8f;

RWStructuredBuffer<Gaussian3D> gGaussians;
StructuredBuffer<Gaussian3D> gGradients;

float Adam(const float g, inout float m, inout float v)
{
    m = gBeta1 * m + (1.0f - gBeta1) * g;
    v = gBeta2 * v + (1.0f - gBeta2) * g * g;
    const float numerator = m / (1.0f - pow(gBeta1, gOptimTimeStep));
    const float denominator = sqrt(v / (1.0f - pow(gBeta2, gOptimTimeStep))) + EPSILON;
    return numerator / denominator;
}

bool CheckNumeric(const float value)
{
    return isfinite(value) && !isnan(value);
}

bool CheckNumeric(const float3 value)
{
    return all(isfinite(value) && !isnan(value));
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

    // Get the gradient
    const Gaussian3D gradient = gGradients[gaussianIdx];
    const float3 meanGradient = gradient.mean * gLearningRate;
    const float sigmaGradient = gradient.sigma * gLearningRate;
    const float weightGradient = gradient.weight * gLearningRate;
    if (!CheckNumeric(meanGradient) || !CheckNumeric(sigmaGradient) || !CheckNumeric(weightGradient))
    {
        return;
    }

    // SGD
    gGaussians[gaussianIdx].mean -= meanGradient;
    gGaussians[gaussianIdx].sigma -= sigmaGradient;
    gGaussians[gaussianIdx].weight -= weightGradient;

    // TODO: Remove?
    gGaussians[gaussianIdx].sigma = clamp(gGaussians[gaussianIdx].sigma, 0.5f, 2.0f);
}
