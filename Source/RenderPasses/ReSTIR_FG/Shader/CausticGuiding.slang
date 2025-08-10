import Utils.Sampling.SampleGenerator;

struct CausticSource
{
    float3 srcPos;    // Position of the caustic castic diffuse surface
    float3 targetPos; // Position of the first vertex after the source in the caustic chain
};

struct CausticGaussianReservoir
{
    float3 pos; // Position of the first vertex after the source in the caustic chain
    float thp;
    float weightSum;
    float confidence;

    __init()
    {
        pos = float3(0.0);
        weightSum = 0.0;
        confidence = 0.0;
    }

    __init(const float3 initialPos, const float initialThp)
    {
        pos = initialPos;
        thp = initialThp;
        weightSum = initialThp;
        confidence = 1.0;
    }

    [mutating] void Update(const float3 newPos, float newThp, inout SampleGenerator sg)
    {
        if (isnan(newThp) || isinf(newThp))
        {
            newThp = 0.0f;
        }

        confidence += 1.0f;
        confidence = min(confidence, 50.0f);

        weightSum += newThp;
        if (sampleNext1D(sg) * weightSum < newThp)
        {
            pos = newPos;
            thp = newThp;
        }
    }

    [mutating] void Merge(const in CausticGaussianReservoir otherRes, inout SampleGenerator sg)
    {
        float weight = otherRes.thp * otherRes.confidence / (confidence + otherRes.confidence);        
        if (isnan(weight) || isinf(weight))
        {
            weight = 0.0f;
        }

        confidence += otherRes.confidence;
        confidence = min(confidence, 50.0f);
        
        weightSum += weight;
        if (sampleNext1D(sg) * weightSum < weight)
        {
            pos = otherRes.pos;
            thp = otherRes.thp;
        }
    }
};

