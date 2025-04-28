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
    const uint collectionCount = gFirstHitCollectionCounts[firstHitPhotonIdx];

    // For each light, find the closest cluster
    for (uint lightIdx = 0; lightIdx < gTotalLightCount; ++lightIdx)
    {
        // Find closest cluster
        uint globalBestClusterIdx = 0;
        float bestDistance = 1e10f;
        for (uint clusterIdx = 0; clusterIdx < gCausticClusterCount; ++clusterIdx)
        {
            const uint globalClusterIdx = lightIdx * gCausticClusterCount + clusterIdx;
            const float3 clusterPos = gCausticClusters[globalClusterIdx];
            const float distance = length(clusterPos - position);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                globalBestClusterIdx = globalClusterIdx;
            }
        }

        // Increment counter of closest cluster
        InterlockedAdd(gCausticClusterCounters[globalBestClusterIdx], collectionCount);
    }
}
