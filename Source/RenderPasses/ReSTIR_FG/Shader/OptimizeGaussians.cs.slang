#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import GaussianHelper;

cbuffer Constants
{
    uint gTotalGaussianCount;
}

RWStructuredBuffer<Gaussian3D> gGaussians;
Buffer<int> gGradients;

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
    static const float learningRate = 100.0f;
    gGaussians[gaussianIdx].mean -= learningRate * meanGradient;
    gGaussians[gaussianIdx].sigma -= learningRate * sigmaGradient;
    gGaussians[gaussianIdx].weight -= learningRate * weightGradient;
}
