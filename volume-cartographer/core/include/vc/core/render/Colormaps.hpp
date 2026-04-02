#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace vc {

enum class OverlayColormapKind { OpenCv, Tint, DiscreteLut };
enum class ColormapAudience { Shared, OverlayOnly };
enum class EntryScope { SharedOnly, OverlayCompatible };

struct OverlayColormapSpec {
    std::string id;
    std::string label;
    OverlayColormapKind kind;
    ColormapAudience audience;
    int opencvCode;
    cv::Vec3f tint; // R, G, B in [0,1]
    const uint32_t* discreteLut;
};

struct OverlayColormapEntry {
    std::string label;
    std::string id;
};

const std::vector<OverlayColormapSpec>& specs();
const OverlayColormapSpec& resolve(const std::string& id);

// Apply colormap and write directly into a caller-provided ARGB32 buffer.
// outBuf must point to rows*outStride uint32_t elements.
// outStride is in uint32_t units (pixels per row including padding).
void makeColors(const cv::Mat_<uint8_t>& values, const OverlayColormapSpec& spec,
                uint32_t* outBuf, int outStride);

const std::vector<OverlayColormapEntry>& entries(EntryScope scope = EntryScope::OverlayCompatible);

// Non-temporal store for write-only ARGB32 output -- bypasses cache,
// freeing cache lines for read-heavy LUT/chunk data.
inline void nt_store_u32(uint32_t* dst, uint32_t val) {
#if defined(__x86_64__)
    _mm_stream_si32(reinterpret_cast<int*>(dst), static_cast<int>(val));
#elif defined(__aarch64__) && __has_builtin(__builtin_nontemporal_store)
    __builtin_nontemporal_store(val, dst);
#else
    *dst = val;
#endif
}

// Apply a packed uint32_t[256] LUT to a grayscale image, writing ARGB32 output.
void applyPackedLut(const cv::Mat_<uint8_t>& values, const uint32_t* lut,
                    uint32_t* outBuf, int outStride);

}  // namespace vc
