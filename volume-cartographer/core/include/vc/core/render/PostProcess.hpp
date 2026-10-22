#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <cstdint>

namespace vc {

// Build a window/level LUT mapping uint8 -> ARGB32 for the fused sampling path.
// Only valid for non-stretch grayscale mode (no colormap, no stretchValues).
void buildWindowLevelLut(std::array<uint32_t, 256>& lut,
                         float windowLow, float windowHigh) noexcept;

// Same as buildWindowLevelLut but fuses in a colormap (by id).
// Empty/unknown id => grayscale (same as buildWindowLevelLut).
void buildWindowLevelColormapLut(std::array<uint32_t, 256>& lut,
                                 float windowLow, float windowHigh,
                                 const std::string& colormapId);

// Overlay LUT: premultiplied RGB with per-entry alpha in the top byte, for
// blendOverlayArgb. Default mode is opaque colour inside [windowLow,
// windowHigh] and transparent outside. intensityAsAlpha makes the windowed
// intensity the alpha (saturating above windowHigh); the colour is the
// colormap's, premultiplied or scaled by alpha per spec.premultiplied, and
// discrete (label) maps are opaque wherever the label is nonzero.
void buildOverlayLut(std::array<uint32_t, 256>& lut,
                     float windowLow, float windowHigh,
                     const std::string& colormapId,
                     bool intensityAsAlpha);

// base' = opacity * overlay.rgb + (1 - opacity * overlay.a) * base.rgb, with
// overlay from buildOverlayLut. Entries with zero alpha leave base unchanged.
inline uint32_t blendOverlayArgb(uint32_t base, uint32_t overlay, float opacity) noexcept
{
    const uint32_t oa = overlay >> 24;
    if (oa == 0) {
        return base;
    }
    const float a = std::clamp(opacity, 0.0f, 1.0f);
    const float keep = 1.0f - a * (static_cast<float>(oa) / 255.0f);
    const auto mix = [a, keep](uint32_t b, uint32_t o) -> uint32_t {
        return static_cast<uint32_t>(
            std::clamp(std::lround(keep * float(b) + a * float(o)), 0L, 255L));
    };
    const uint32_t br = (base >> 16) & 0xFFu;
    const uint32_t bg = (base >> 8) & 0xFFu;
    const uint32_t bb = base & 0xFFu;
    const uint32_t or_ = (overlay >> 16) & 0xFFu;
    const uint32_t og = (overlay >> 8) & 0xFFu;
    const uint32_t ob = overlay & 0xFFu;
    return 0xFF000000u | (mix(br, or_) << 16) | (mix(bg, og) << 8) | mix(bb, ob);
}

}  // namespace vc
