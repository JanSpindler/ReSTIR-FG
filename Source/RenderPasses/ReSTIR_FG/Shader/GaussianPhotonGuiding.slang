static const float PI = 3.14159265359f;
static const float SQRT_2 = 1.41421356237f;

// Approximation of the error function
// Abramowitz and Stegun 7.1.26
float erf(const float x)
{
    static const float a1 = 0.254829592f;
    static const float a2 = -0.284496736f;
    static const float a3 = 1.421413741f;
    static const float a4 = -1.453152027f;
    static const float a5 = 1.061405429f;
    static const float p = 0.3275911f;
    const float t = 1.0f / (1.0f + p * x);
    const float t2 = t * t;
    const float t3 = t2 * t;
    const float t4 = t3 * t;
    const float t5 = t4 * t;
    return 1.0f - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * exp(-x * x);
}

float f0(const float3 x, const float3 x0, const float3 mu, const float sigma)
{
    const float d = length(x0 - mu);
    if (d < 1e-3f) { return 0.0f; }
    const float3 z = (mu - x0) / d;

    float3 xMinusX0 = x - x0;
    const float xMinusX0Length = length(xMinusX0);
    if (xMinusX0Length < 1e-3f) { return 0.0f; }
    xMinusX0 /= xMinusX0Length;

    const float theta = acos(dot(z, xMinusX0));

    const float sigma2 = sigma * sigma;
    const float d2 = d * d;
    const float sinTheta = sin(theta);
    const float sinTheta2 = sinTheta * sinTheta;
    const float cosTheta = cos(theta);
    const float cosTheta2 = cosTheta * cosTheta;

    const float term1 = sigma2 * exp(-0.5f * d2 / sigma2) * d * cosTheta;
    const float term2 = sqrt(PI * 0.5f) * sigma * exp(-0.5f * d2 * sinTheta2 / sigma2) * (sigma2 + (d2 * cosTheta2));
    const float term3 = 1.0f * erf((d * cosTheta) / (SQRT_2 * sigma));

    return term1 + (term2 * term3);
}

const float F0(const float3 x, const float3 x0, const float3 mu, const float sigma)
{
    const float denom = 2.0f * PI * sigma * sigma;
    const float denom3 = denom * denom * denom;
    return f0(x, x0, mu, sigma) / sqrt(denom3);
}
