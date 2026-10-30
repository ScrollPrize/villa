#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

namespace vc {

// Copy a rectangular intersection using element strides in Z,Y,X order.
// A small transpose tile keeps both source X runs and destination Z runs
// contiguous for chunk-C-order -> Array3D conversion, without changing either
// public layout. Also supports arbitrary positive strides and partial tiles.
// Source and destination storage must not overlap; this is not memmove.
template<class T>
void copyStrided3D(const T* src, const std::array<std::size_t, 3>& srcStride,
                   T* dst, const std::array<std::size_t, 3>& dstStride,
                   const std::array<std::size_t, 3>& size)
{
    constexpr std::size_t tile = 32;
    T scratch[tile][tile + 1];
    for (std::size_t y = 0; y < size[1]; ++y)
        for (std::size_t z = 0; z < size[0]; z += tile)
            for (std::size_t x = 0; x < size[2]; x += tile) {
                const auto nz = std::min(tile, size[0] - z);
                const auto nx = std::min(tile, size[2] - x);
                const T* in = src + z*srcStride[0] + y*srcStride[1] + x*srcStride[2];
                T* out = dst + z*dstStride[0] + y*dstStride[1] + x*dstStride[2];
                for (std::size_t a = 0; a < nz; ++a)
                    for (std::size_t b = 0; b < nx; ++b)
                        scratch[a][b] = in[a*srcStride[0] + b*srcStride[2]];
                for (std::size_t b = 0; b < nx; ++b)
                    for (std::size_t a = 0; a < nz; ++a)
                        out[a*dstStride[0] + b*dstStride[2]] = scratch[a][b];
            }
}

} // namespace vc
