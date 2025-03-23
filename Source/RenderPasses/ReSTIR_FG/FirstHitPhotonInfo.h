#pragma once

#include "falcor_cuda_math.h"

struct FirstHitPhotonInfo
{
    float3 pos;
    float samplingPdf;
    uint lightIdx;
};
