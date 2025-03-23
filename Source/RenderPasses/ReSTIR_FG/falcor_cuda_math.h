#pragma once

#ifdef __CUDACC__
using uint = uint32_t;
#else
#include <Utils/Math/VectorTypes.h>
#endif
