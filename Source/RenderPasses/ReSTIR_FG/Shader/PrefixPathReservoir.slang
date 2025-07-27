import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Scene.HitInfo;
import Scene.RaytracingInline;

static const float MAX_CONFIDENCE = 20.0f;

struct PrefixPath
{
    float3 throughput;

    PackedHitInfo endHitInfo;
    float4 endViewDir;
    float endRayDist;
    float endHitT;

    uint flags;
    float samplingPdf;
    uint seed;

    __init()
    {
        throughput = float3(0.0f, 0.0f, 0.0f);

        endHitInfo = {};
        endViewDir = float4(0.0f, 0.0f, 0.0f, 0.0f);
        endRayDist = 0.0f;
        endHitT = 0.0f;

        flags = 0u;
        samplingPdf = 0.0f;
        seed = 0u;
    }

    float GetLuminance()
    {
        return luminance(throughput);
    }

    bool HasDeltaEvent()
    {
        return flags >> 31 != 0;
    }

    [mutating] void SetDeltaEvent(const bool hasDelta)
    {
        if (hasDelta)
        {
            flags |= 0x80000000u; // Set the highest bit to indicate a delta event
        }
        else
        {
            flags &= 0x7FFFFFFFu; // Clear the highest bit
        }
    }

    uint GetLength()
    {
        return flags & 0x7FFFFFFFu;
    }

    [mutating] void SetLength(const uint length)
    {
        flags = (flags & 0x80000000u) | (length & 0x7FFFFFFFu);
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

    [mutating] void Reset()
    {
        weightSum = 0.0f;
        confidence = 0.0f;
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
        confidence = min(confidence, MAX_CONFIDENCE);

        weightSum += weight;
        if (sampleNext1D(sg) < weight / weightSum)
        {
            path = otherPath;
        }
    }

    [mutating] void Merge(
        const in PrefixPathReservoir otherRes,
        const PrefixPath shiftedPath,
        const float misWeight,
        const float jacobian,
        inout SampleGenerator sg)
    {
        float weight = misWeight * shiftedPath.GetLuminance() * otherRes.GetW() * jacobian;
        if (isnan(weight) || isinf(weight))
        {
            weight = 0.0f;
        }

        confidence += otherRes.confidence;
        confidence = min(confidence, MAX_CONFIDENCE);

        weightSum += weight;
        if (sampleNext1D(sg) < (weight * otherRes.confidence) / (weightSum * confidence))
        {
            path = shiftedPath;
        }
    }
};
