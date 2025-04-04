import GaussianHelper;

cbuffer Constants
{
    uint gGaussianCount;
    uint gAnalyticLightCount;
    uint gGeometricLightCount;
}

StructuredBuffer<Gaussian3D> gGaussians;
RWBuffer<float> gSoftmaxWeights;

[numthreads(32, 1, 1)]
void main(uint2 index: SV_DispatchThreadID)
{
    // Get light index
    const uint lightIdx = index.x;
    const uint totalLightCount = gAnalyticLightCount + gGeometricLightCount;
    if (lightIdx >= totalLightCount)
    {
        return;
    }

    // Caculate softmax weight sum
    float sum = 0.0f;
    for (uint gaussianIdx = 0; gaussianIdx < gGaussianCount; ++gaussianIdx)
    {
        const uint totalGaussianIdx = lightIdx * gGaussianCount + gaussianIdx;
        const Gaussian3D gaussian = gGaussians[totalGaussianIdx];
        const float weight = gaussian.weight;
        sum += exp(weight);
    }

    // Calculate softmax weight
    for (uint gaussianIdx = 0; gaussianIdx < gGaussianCount; ++gaussianIdx)
    {
        const uint totalGaussianIdx = lightIdx * gGaussianCount + gaussianIdx;
        const Gaussian3D gaussian = gGaussians[totalGaussianIdx];
        const float weight = gaussian.weight;
        const float softmaxWeight = exp(weight) / sum;
        gSoftmaxWeights[totalGaussianIdx] = softmaxWeight;
    }
}
