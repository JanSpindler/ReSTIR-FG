#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

// Hash by thomas wang 
uint hash(int3 cell)
{
    // Convert to uint64
    uint64_t key = 0;
    uint64_t cells = cell.x;
    cells &= 0x1FFFFF;
    key |= cells << 42;
    cells = cell.y;
    cells &= 0x1FFFFF;
    key |= cells << 21;
    cells = cell.z;
    cells &= 0x1FFFFF;
    key |= cells;

    key = (~key) + (key << 18);
    key = key ^ (key >> 31);
    key *= 21;
    key = key ^ (key >> 11);
    key = key + (key << 6);
    uint res = uint(key) ^ uint(key >> 22);
    return res;
}

uint hash2(const int3 pos)
{
    return hash(pos);
    //return (pos.x * 1) ^ (pos.y * 2654435761) ^ (pos.z * 805459861);
}

END_NAMESPACE_FALCOR
