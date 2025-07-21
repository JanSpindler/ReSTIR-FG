#pragma once

#ifdef __CUDACC__
#include <vector_types.h>
using uint = uint32_t;
#else
#include <Utils/Math/VectorTypes.h>
#endif
