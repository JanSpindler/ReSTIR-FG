#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import GaussianHelper;

cbuffer Constants
{
    uint gTotalGaussianCount;
}

static const float gLearningRate = 0.001f;
static const float gBeta1 = 0.9f;
static const float gBeta2 = 0.999f;
static const float gOptimTimeStep = 2.0f;
static const float EPSILON = 1e-8f;

RWStructuredBuffer<Gaussian3D> gGaussians;
Buffer<int> gGradients;

float Adam(const float g, inout float m, inout float v)
{
    m = gBeta1 * m + (1.0f - gBeta1) * g;
    v = gBeta2 * v + (1.0f - gBeta2) * g * g;
    const float numerator = m / (1.0f - pow(gBeta1, gOptimTimeStep));
    const float denominator = sqrt(v / (1.0f - pow(gBeta2, gOptimTimeStep))) + EPSILON;
    return numerator / denominator;
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
    const uint gradientIdx = gaussianIdx * 5;
    const float3 meanGradient = float3(
        DequantizeGradient(gGradients[gradientIdx + 0]),
        DequantizeGradient(gGradients[gradientIdx + 1]),
        DequantizeGradient(gGradients[gradientIdx + 2]));
    const float sigmaGradient = DequantizeGradient(gGradients[gradientIdx + 3]);
    const float weightGradient = DequantizeGradient(gGradients[gradientIdx + 4]);

    // SGD
    static const float learningRate = 0.01f;
    gGaussians[gaussianIdx].mean -= learningRate * meanGradient;
    gGaussians[gaussianIdx].sigma -= learningRate * sigmaGradient;
    gGaussians[gaussianIdx].weight -= learningRate * weightGradient;

    // TODO: Remove?
    gGaussians[gaussianIdx].sigma = clamp(gGaussians[gaussianIdx].sigma, 0.5f, 2.0f);
}
