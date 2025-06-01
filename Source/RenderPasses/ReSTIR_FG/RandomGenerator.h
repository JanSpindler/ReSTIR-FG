#pragma once

#include <random>
#include "Falcor.h"

class RandomGenerator
{
public:
    static float Float()
    {
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        return dis(m_Gen);
    }

    static Falcor::float3 Float3() { return Falcor::float3(Float(), Float(), Float()); }

    static Falcor::uint UInt()
    {
        std::uniform_int_distribution<Falcor::uint> dis(0, std::numeric_limits<Falcor::uint>::max());
        return dis(m_Gen);
    }

    static Falcor::float3 AabbPoint(const AABB& aabb)
    {
        return aabb.minPoint + Float3() * aabb.extent();
    }

private:
    static inline std::random_device m_Rd;
    static inline std::mt19937 m_Gen{m_Rd()};
};
