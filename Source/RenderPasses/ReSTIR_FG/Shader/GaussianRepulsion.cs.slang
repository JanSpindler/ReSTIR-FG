import GaussianHelper;

cbuffer Constants
{
    uint gGaussianCount;      // Number of Gaussians per light
    uint gLightCount;         // Total number of lights
    float gPositionScaling;   // Position scaling factor
    float gRepulsiveForce;    // Repulsion force strength
    float gRepulsiveDistance; // Repulsion distance threshold
};

StructuredBuffer<Gaussian3D> gGaussians;
RWStructuredBuffer<Gaussian3D> gGaussianGradients;

bool CheckNumeric(float x)
{
    return !isnan(x) && !isinf(x);
}

float3 CalculateRepulsionForce(float3 meanCurrent, float3 meanOther)
{
    const float repulsiveDistance2 = gRepulsiveDistance * gRepulsiveDistance * gPositionScaling * gPositionScaling;
    const float repulsiveFactor = gRepulsiveForce / repulsiveDistance2;

    const float3 diffVec = meanCurrent - meanOther;
    const float diffDot = dot(diffVec, diffVec);

    // Calculate repulsion force using exponential falloff
    const float expTerm = exp(-0.5f * diffDot / repulsiveDistance2);
    return -repulsiveFactor * expTerm * diffVec;
}

[numthreads(32, 1, 1)]
void main(uint3 threadID: SV_DispatchThreadID)
{
    const uint gaussianIdx = threadID.x;
    const uint totalGaussianCount = gGaussianCount * gLightCount;

    // Early exit if thread is out of bounds
    if (gaussianIdx >= totalGaussianCount)
    {
        return;
    }

    // Calculate which light this Gaussian belongs to
    const uint lightIdx = gaussianIdx / gGaussianCount;
    const uint localGaussianIdx = gaussianIdx % gGaussianCount;

    // Get the first Gaussian index for this light
    const uint firstGaussianIdx = lightIdx * gGaussianCount;

    // Get current Gaussian
    const Gaussian3D currentGaussian = gGaussians[gaussianIdx];

    // Initialize repulsion force accumulator
    float3 totalRepulsionForce = float3(0.0f, 0.0f, 0.0f);

    // Calculate repulsion with all other Gaussians in the same light group
    for (uint otherIdx = 0; otherIdx < gGaussianCount; ++otherIdx)
    {
        // Skip self-repulsion
        if (otherIdx == localGaussianIdx)
        {
            continue;
        }

        const uint otherGaussianIdx = firstGaussianIdx + otherIdx;
        const Gaussian3D otherGaussian = gGaussians[otherGaussianIdx];

        // Calculate repulsion force between current and other Gaussian
        const float3 repulsionForce = CalculateRepulsionForce(currentGaussian.mean, otherGaussian.mean);
        totalRepulsionForce += repulsionForce;
    }

    // Add repulsion force to the gradient for the mean position
    if (CheckNumeric(totalRepulsionForce.x))
    {
        gGaussianGradients[gaussianIdx].mean.x += totalRepulsionForce.x;
    }

    if (CheckNumeric(totalRepulsionForce.y))
    {
        gGaussianGradients[gaussianIdx].mean.y += totalRepulsionForce.y;
    }

    if (CheckNumeric(totalRepulsionForce.z))
    {
        gGaussianGradients[gaussianIdx].mean.z += totalRepulsionForce.z;
    }
}
