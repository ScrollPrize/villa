#pragma once

namespace vc::core::detail {

inline bool isPowerOfTwo(int v)
{
    return v > 0 && (v & (v - 1)) == 0;
}

inline int log2Pow2(int v)
{
    int r = 0;
    while ((v >> r) > 1) {
        ++r;
    }
    return r;
}

inline int chunkIndex(int coord, int chunkSize, bool isPow2, int shift)
{
    return isPow2 ? (coord >> shift) : (coord / chunkSize);
}

inline int localOffset(int coord, int chunkSize, bool isPow2, int mask)
{
    return isPow2 ? (coord & mask) : (coord % chunkSize);
}

} // namespace vc::core::detail
