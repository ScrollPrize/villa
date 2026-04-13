#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace vc::core::util {

using Shape3 = std::array<std::size_t, 3>;

inline constexpr std::size_t linearIndex(const Shape3& shape,
                                         std::size_t z,
                                         std::size_t y,
                                         std::size_t x) noexcept
{
    return (z * shape[1] + y) * shape[2] + x;
}

inline void downsampleBinaryOr(const uint8_t* src,
                               const Shape3& srcShape,
                               uint8_t* dst,
                               const Shape3& dstShape)
{
    for (std::size_t zz = 0; zz < dstShape[0]; ++zz) {
        for (std::size_t yy = 0; yy < dstShape[1]; ++yy) {
            for (std::size_t xx = 0; xx < dstShape[2]; ++xx) {
                uint8_t any = 0;
                for (std::size_t dz = 0; dz < 2; ++dz) {
                    const std::size_t srcZ = 2 * zz + dz;
                    if (srcZ >= srcShape[0]) {
                        continue;
                    }
                    for (std::size_t dy = 0; dy < 2; ++dy) {
                        const std::size_t srcY = 2 * yy + dy;
                        if (srcY >= srcShape[1]) {
                            continue;
                        }
                        for (std::size_t dx = 0; dx < 2; ++dx) {
                            const std::size_t srcX = 2 * xx + dx;
                            if (srcX >= srcShape[2]) {
                                continue;
                            }
                            any |= src[linearIndex(srcShape, srcZ, srcY, srcX)];
                        }
                    }
                }
                dst[linearIndex(dstShape, zz, yy, xx)] = any != 0 ? uint8_t(255) : uint8_t(0);
            }
        }
    }
}

inline void downsampleLabelPriority(const uint8_t* src,
                                    const Shape3& srcShape,
                                    uint8_t* dst,
                                    const Shape3& dstShape,
                                    uint8_t ignoreValue)
{
    for (std::size_t zz = 0; zz < dstShape[0]; ++zz) {
        for (std::size_t yy = 0; yy < dstShape[1]; ++yy) {
            for (std::size_t xx = 0; xx < dstShape[2]; ++xx) {
                uint8_t chosenForeground = 0;
                bool sawIgnore = false;
                for (std::size_t dz = 0; dz < 2; ++dz) {
                    const std::size_t srcZ = 2 * zz + dz;
                    if (srcZ >= srcShape[0]) {
                        continue;
                    }
                    for (std::size_t dy = 0; dy < 2; ++dy) {
                        const std::size_t srcY = 2 * yy + dy;
                        if (srcY >= srcShape[1]) {
                            continue;
                        }
                        for (std::size_t dx = 0; dx < 2; ++dx) {
                            const std::size_t srcX = 2 * xx + dx;
                            if (srcX >= srcShape[2]) {
                                continue;
                            }

                            const uint8_t value = src[linearIndex(srcShape, srcZ, srcY, srcX)];
                            if (value == 0) {
                                continue;
                            }
                            if (value == ignoreValue) {
                                sawIgnore = true;
                                continue;
                            }
                            chosenForeground = value;
                            break;
                        }
                        if (chosenForeground != 0) {
                            break;
                        }
                    }
                    if (chosenForeground != 0) {
                        break;
                    }
                }
                dst[linearIndex(dstShape, zz, yy, xx)] =
                    chosenForeground != 0 ? chosenForeground : (sawIgnore ? ignoreValue : uint8_t(0));
            }
        }
    }
}

}  // namespace vc::core::util
