#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
namespace vcr::display {
// Packed ARGB palettes. All float-to-byte conversions clamp then truncate;
// nonfinite pixels are zero and have zero coverage.
void toUint8(const float* input, std::size_t n, std::uint8_t* output, std::uint8_t* coverage = nullptr);
void windowLevelLut(std::array<std::uint32_t,256>& out, float low, float high) noexcept;
void paletteLut(std::array<std::uint32_t,256>& out, const std::array<std::uint32_t,256>& palette,
                float low, float high, bool blackZero = true) noexcept;
void overlayLut(std::array<std::uint32_t,256>& out, const std::array<std::uint32_t,256>& color,
                float low, float high, bool intensityAsAlpha, bool premultiplied) noexcept;
std::uint32_t blendArgb(std::uint32_t base, std::uint32_t overlay, float opacity) noexcept;
void applyLut(const std::uint8_t* input, std::size_t rows, std::size_t cols, std::size_t inputStride,
              const std::uint32_t* lut, std::uint32_t* output, std::size_t outputStride);
void blend(const std::uint32_t* base, const std::uint32_t* overlay, std::size_t n, float opacity, std::uint32_t* output);
#ifdef VC_READER_CUDA
namespace cuda {
// Pixel buffers and palette are device pointers; completes on the caller's stream.
void toUint8(const float* input, std::size_t n, std::uint8_t* output, std::uint8_t* coverage = nullptr);
void applyLut(const std::uint8_t* input, std::size_t rows, std::size_t cols, std::size_t inputStride,
              const std::uint32_t* lut, std::uint32_t* output, std::size_t outputStride);
void blend(const std::uint32_t* base, const std::uint32_t* overlay, std::size_t n, float opacity, std::uint32_t* output);
// LUTs are small host arrays, copied back after native generation.
void windowLevelLut(std::array<std::uint32_t,256>& out, float low, float high);
void paletteLut(std::array<std::uint32_t,256>& out, const std::array<std::uint32_t,256>& palette,
                float low, float high, bool blackZero = true);
void overlayLut(std::array<std::uint32_t,256>& out, const std::array<std::uint32_t,256>& color,
                float low, float high, bool intensityAsAlpha, bool premultiplied);
}
#endif
}
