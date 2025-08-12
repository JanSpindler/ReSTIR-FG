#pragma once

#include "falcor_cuda_math.h"

#ifndef __CUDACC__
using namespace Falcor;
#endif

struct FirstHitPhotonInfo
{
    float3 pos;
    float samplingPdf;
    float gmmPdf;
    uint lightIdx;
};
