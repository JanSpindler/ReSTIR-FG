import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Scene.HitInfo;

static const float MAX_CONFIDENCE = 20.0f;

struct PrefixPath
{
    float3 throughput;
    PackedHitInfo hitInfo;
    float4 viewDir;
    float rayDist;
    float hitT;
    uint seed;
    uint length;

    __init()
    {
        throughput = float3(0.0f, 0.0f, 0.0f);
        hitInfo = {};
        viewDir = float4(0.0f, 0.0f, 0.0f, 0.0f);
        rayDist = 0.0f;
        hitT = 0.0f;
        seed = 0;
        length = 0;
    }

    float GetLuminance()
    {
        return luminance(throughput);
    }
};

struct PrefixPathReservoir
{
    PrefixPath path;
    float weightSum;
    float confidence;

    __init()
    {
        path = PrefixPath();
        weightSum = 0.0f;
        confidence = 0.0f;
    }

    __init(const in PrefixPath initialPath, const float samplingPdf)
    {
        path = initialPath;
        weightSum = initialPath.GetLuminance() / samplingPdf;
        if (isnan(weightSum) || isinf(weightSum))
        {
            weightSum = 0.0f;
        }
        confidence = 1.0f;
    }

    float GetW()
    {
        const float pHat = path.GetLuminance();
        if (isnan(pHat) || isinf(pHat) || pHat <= 0.0f)
        {
            return 0.0f;
        }
        return weightSum / pHat;
    }

    [mutating] void Update(const in PrefixPath otherPath, const float samplingPdf, inout SampleGenerator sg)
    {
        float weight = otherPath.GetLuminance() / samplingPdf;
        if (isnan(weight) || isinf(weight))
        {
            weight = 0.0f;
        }

        confidence += 1.0f;
        confidence = max(confidence, MAX_CONFIDENCE);

        weightSum += weight;
        if (sampleNext1D(sg) < weight / weightSum)
        {
            path = otherPath;
        }
    }

    [mutating] void Merge(
        const in PrefixPathReservoir otherRes,
        const float3 shiftedThroughput,
        const float misWeight,
        const float jacobian,
        inout SampleGenerator sg)
    {
        float weight = misWeight * luminance(shiftedThroughput) * otherRes.GetW() * jacobian;
        if (isnan(weight) || isinf(weight))
        {
            weight = 0.0f;
        }

        confidence += otherRes.confidence;
        confidence = max(confidence, MAX_CONFIDENCE);

        weightSum += weight;
        if (sampleNext1D(sg) < (weight * otherRes.confidence) / (weightSum * confidence))
        {
            path.throughput = shiftedThroughput;
            path.hitInfo = otherRes.path.hitInfo;
            path.viewDir = otherRes.path.viewDir;
            path.rayDist = otherRes.path.rayDist;
            path.hitT = otherRes.path.hitT;
        }
    }
};
