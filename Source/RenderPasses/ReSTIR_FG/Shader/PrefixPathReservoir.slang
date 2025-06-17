import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;

struct PrefixPathReservoir
{
    // Jacobian is always 1 because we only perform random replay

    // Path data
    float3 F;
    uint length; // Number of vertices inlcuding primary hit.
    uint sgSeed; // State of sample generator right before generating this path

    // Resampling data
    float M;
    float weight;
    
    // Empty init
    [mutating]
    __init()
    {
        F = float3(0.0f, 0.0f, 0.0f);
        length = 0;
        sgSeed = 0;
        M = 0.0f;
        weight = 0.0f;
    }

    [mutating]
    bool Add(const float3 inF, const float p, inout SampleGenerator sg)
    {
        M += 1.0f;

        const float w = luminance(inF) / p;
        if (isnan(w) || w <= 0.0f)
        {
            return false;
        }
        weight += w;

        if (sampleNext1D(sg) * weight <= w)
        {
            F = inF;
            // PHat = pHat but we are using PSS
            return true;
        }
        return false;
    }

    [mutating]
    bool Merge(
        const float3 inF,
        const float inJacobian,
        const in PrefixPathReservoir inRes,
        inout SampleGenerator sg,
        const float misWeight = 1.0f,
        const bool forceAdd = false)
    {
        M += inRes.M;

        const float w = luminance(inF) * inJacobian * inRes.M * inRes.weight * misWeight;
        if (isnan(w) || w <= 0.0f)
        {
            return false;
        }
        weight += w;

        if (forceAdd || sampleNext1D(sg) * weight <= w)
        {
            F = inF;
            length = inRes.length;
            sgSeed = inRes.sgSeed;
            return true;
        }
        return false;
    }

    [mutating]
    bool MergeWithResamplingMIS(
        const float3 inF,
        const float inJacobian,
        const in PrefixPathReservoir inRes,
        inout SampleGenerator sg,
        const float misWeight = 1.0f,
        const bool forceAdd = false)
    {
        M += inRes.M;

        const float w = luminance(inF) * inJacobian * inRes.weight * misWeight;
        if (isnan(w) || w <= 0.0f)
        {
            return false;
        }
        weight += w;

        if (forceAdd || sampleNext1D(sg) * weight <= w)
        {
            F = inF;
            length = inRes.length;
            sgSeed = inRes.sgSeed;
            return true;
        }
        return false;
    }

    [mutating]
    bool MergeInSamplePixel(const in PrefixPathReservoir inRes, inout SampleGenerator sg)
    {
        M += inRes.M;

        const float w = inRes.weight;
        if (isnan(w) || w <= 0.0f)
        {
            return false;
        }
        weight += w;

        if (sampleNext1D(sg) * weight <= w)
        {
            F = inRes.F;
            length = inRes.length;
            sgSeed = inRes.sgSeed;
            return true;
        }
        return false;
    }

    [mutating]
    void PrepareMerging()
    {
        weight *= luminance(F) * M;
    }

    [mutating]
    void FinalizeRIS()
    {
        const float pHat = luminance(F);
        if (pHat <= 0.0f || M <= 0.0f)
        {
            weight = 0.0f;
        }
        else
        {
            weight /= pHat * M;
        }
    }

    [mutating]
    void FinalizeGRIS()
    {
        const float pHat = luminance(F);
        if (pHat <= 0.0f)
        {
            weight = 0.0f;
        }
        else
        {
            weight /= pHat;
        }
    }
};
