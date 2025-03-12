#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import GaussianHelper;

StructuredBuffer<Gaussian3D> gGaussians;
Buffer<uint> gFirstHitPhotonCount;
Buffer<uint> gFirstHitCollectionCounts;
StructuredBuffer<FirstHitPhotonInfo> gFirstHitPhotonInfo;
RWStructuredBuffer<Gaussian3D> gGradients;

[numthreads(32, 1, 1)]
void main(uint2 index : SV_DispatchThreadID)
{
    // Get index
    const uint firstHitPhotonIdx = index.x;
    if (firstHitPhotonIdx >= gFirstHitPhotonCount[0])
    {
        return;
    }

    // Calculate gradient
    const float targetPdf = (float)gFirstHitCollectionCounts[firstHitPhotonIdx];
    const float samplingPdf = gFirstHitPhotonInfo[firstHitPhotonIdx].samplingPdf;
    const float pdfFactor = targetPdf / samplingPdf;
}
