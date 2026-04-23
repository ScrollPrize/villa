#pragma once

// 2× 3D pooling kernels for u8 volumes.  Each `dst` voxel is derived from
// the corresponding 2×2×2 `src` neighbourhood.  Used as configurable
// alternatives to the c3d bitstream's native LOD synthesis when the
// caller wants a specific aggregation (average / min / max) instead of
// whatever filter the codec applies.

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <vc/core/util/BinaryPyramid.hpp>  // Shape3, linearIndex

namespace vc::core::util {

inline void downsampleBoxAverage3d(const uint8_t* src,
                                   const Shape3& srcShape,
                                   uint8_t* dst,
                                   const Shape3& dstShape)
{
    for (std::size_t zz = 0; zz < dstShape[0]; ++zz) {
        for (std::size_t yy = 0; yy < dstShape[1]; ++yy) {
            for (std::size_t xx = 0; xx < dstShape[2]; ++xx) {
                unsigned sum = 0;
                unsigned n = 0;
                for (std::size_t dz = 0; dz < 2; ++dz) {
                    const std::size_t srcZ = 2 * zz + dz;
                    if (srcZ >= srcShape[0]) continue;
                    for (std::size_t dy = 0; dy < 2; ++dy) {
                        const std::size_t srcY = 2 * yy + dy;
                        if (srcY >= srcShape[1]) continue;
                        for (std::size_t dx = 0; dx < 2; ++dx) {
                            const std::size_t srcX = 2 * xx + dx;
                            if (srcX >= srcShape[2]) continue;
                            sum += src[linearIndex(srcShape, srcZ, srcY, srcX)];
                            ++n;
                        }
                    }
                }
                dst[linearIndex(dstShape, zz, yy, xx)] =
                    n ? static_cast<uint8_t>((sum + (n >> 1)) / n) : uint8_t(0);
            }
        }
    }
}

inline void downsampleMinPool3d(const uint8_t* src,
                                const Shape3& srcShape,
                                uint8_t* dst,
                                const Shape3& dstShape)
{
    for (std::size_t zz = 0; zz < dstShape[0]; ++zz) {
        for (std::size_t yy = 0; yy < dstShape[1]; ++yy) {
            for (std::size_t xx = 0; xx < dstShape[2]; ++xx) {
                uint8_t m = 255;
                bool any = false;
                for (std::size_t dz = 0; dz < 2; ++dz) {
                    const std::size_t srcZ = 2 * zz + dz;
                    if (srcZ >= srcShape[0]) continue;
                    for (std::size_t dy = 0; dy < 2; ++dy) {
                        const std::size_t srcY = 2 * yy + dy;
                        if (srcY >= srcShape[1]) continue;
                        for (std::size_t dx = 0; dx < 2; ++dx) {
                            const std::size_t srcX = 2 * xx + dx;
                            if (srcX >= srcShape[2]) continue;
                            m = std::min(m, src[linearIndex(srcShape, srcZ, srcY, srcX)]);
                            any = true;
                        }
                    }
                }
                dst[linearIndex(dstShape, zz, yy, xx)] = any ? m : uint8_t(0);
            }
        }
    }
}

inline void downsampleMaxPool3d(const uint8_t* src,
                                const Shape3& srcShape,
                                uint8_t* dst,
                                const Shape3& dstShape)
{
    for (std::size_t zz = 0; zz < dstShape[0]; ++zz) {
        for (std::size_t yy = 0; yy < dstShape[1]; ++yy) {
            for (std::size_t xx = 0; xx < dstShape[2]; ++xx) {
                uint8_t m = 0;
                for (std::size_t dz = 0; dz < 2; ++dz) {
                    const std::size_t srcZ = 2 * zz + dz;
                    if (srcZ >= srcShape[0]) continue;
                    for (std::size_t dy = 0; dy < 2; ++dy) {
                        const std::size_t srcY = 2 * yy + dy;
                        if (srcY >= srcShape[1]) continue;
                        for (std::size_t dx = 0; dx < 2; ++dx) {
                            const std::size_t srcX = 2 * xx + dx;
                            if (srcX >= srcShape[2]) continue;
                            m = std::max(m, src[linearIndex(srcShape, srcZ, srcY, srcX)]);
                        }
                    }
                }
                dst[linearIndex(dstShape, zz, yy, xx)] = m;
            }
        }
    }
}

}  // namespace vc::core::util
