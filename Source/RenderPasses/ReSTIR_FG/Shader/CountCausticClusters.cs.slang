import GaussianHelper;

cbuffer Constants
{
    uint gTotalLightCount;
    uint gCausticClusterCount;
    uint gMaxFirstHitPhotonCount;
}

StructuredBuffer<FirstHitPhotonInfo> gFirstHitPhotonInfos;
Buffer<uint> gFirstHitCollectionCounts;

StructuredBuffer<float3> gCausticClusters;
RWBuffer<uint> gCausticClusterCounters;

[numthreads(128, 1, 1)]
void main(uint3 index: SV_DispatchThreadID)
{
    // Index
    const uint firstHitPhotonIdx = index.x;
    if (firstHitPhotonIdx >= gMaxFirstHitPhotonCount)
    {
        return;
    }

    // Get first hit photon info
    const float3 position = gFirstHitPhotonInfos[firstHitPhotonIdx].pos;
    const uint lightIdx = gFirstHitPhotonInfos[firstHitPhotonIdx].lightIdx;
    const uint collectionCount = gFirstHitCollectionCounts[firstHitPhotonIdx];

    // Find closest cluster
    uint bestClusterIdx = 0;
    float bestDistance = float ::maxValue;
    for (uint clusterIdx = 0; clusterIdx < gCausticClusterCount; ++clusterIdx)
    {
        const float3 clusterPos = gCausticClusters[clusterIdx];
        const float distance = length(clusterPos - position);
        if (distance < bestDistance)
        {
            bestDistance = distance;
            bestClusterIdx = clusterIdx;
        }
    }

    // Increment counter of closest cluster
    InterlockedAdd(gCausticClusterCounters[lightIdx * gCausticClusterCount + bestClusterIdx], collectionCount);
}
