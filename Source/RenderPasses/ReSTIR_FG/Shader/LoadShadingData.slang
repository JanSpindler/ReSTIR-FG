import Utils.Sampling.SampleGenerator;
import Utils.Math.MathHelpers;
import Scene.HitInfo;
import Scene.Intersection;
import Scene.Raytracing;

ShadingData LoadShadingData(const HitInfo hit, const float3 rayDir, const ITextureSampler lod)
{
    const TriangleHit triangleHit = hit.getTriangleHit();
    const VertexData v = gScene.getVertexData(triangleHit);
    const uint materialID = gScene.getMaterialID(triangleHit.instanceID);
    const ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);
    return sd;
}
