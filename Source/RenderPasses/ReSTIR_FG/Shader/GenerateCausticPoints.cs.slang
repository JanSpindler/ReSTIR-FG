#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.SceneTypes;
import Scene.Scene;
import Utils.Sampling.SampleGenerator;

cbuffer Constants
{
    uint gCausticPointCount;
    uint gGeometryInstanceID;
    uint gFrameCount;
}

float3 GenerateRandomMeshPoint(inout SampleGenerator sg, const in MeshDesc mesh)
{
    // Select random triangle
    const uint indexCount = mesh.indexCount;
    const uint triangleCount = indexCount / 3;
    const uint triangleIdx = sg.next() % triangleCount;

    // Get triangle indices
    const uint3 indices = gScene.getLocalIndices(mesh.ibOffset, triangleIdx, mesh.flags & uint(MeshFlags::Use16BitIndices));

    // Get vertex positions
    const float3 v0 = gScene.getVertex(mesh.vbOffset + indices.x).position;
    const float3 v1 = gScene.getVertex(mesh.vbOffset + indices.y).position;
    const float3 v2 = gScene.getVertex(mesh.vbOffset + indices.z).position;

    // Generate random barycentric coordinates
    const float2 uv = sampleNext2D(sg);
    const float3 barycentric = float3(1.0f - uv.x - uv.y, uv.x, uv.y);

    // Return
    return barycentric.x * v0 + barycentric.y * v1 + barycentric.z * v2;
}

[numthreads(128, 1, 1)]
void main(uint2 index : SV_DispatchThreadID)
{
    // Check index
    const uint pointIdx = index.x;
    if (pointIdx >= gCausticPointCount)
    {
        return;
    }

    // Init sample generator
    SampleGenerator sg = SampleGenerator(pointIdx, gFrameCount + 1 * 32);

    // Get world transform
    const GeometryInstanceData geometryInstance = gScene.geometryInstances[gGeometryInstanceID];
    const float4x4 instanceTransform = gScene.loadWorldMatrix(geometryInstance.globalMatrixID);

    // Get mesh
    const MeshDesc mesh = gScene.meshes[geometryInstance.geometryID];

    // Generate random point
    const float3 globalMeshPoint = mul(instanceTransform, float4(GenerateRandomMeshPoint(sg, mesh), 1.0f)).xyz;
}
