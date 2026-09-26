#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

// Per-value colour and alpha tables for compositing the volume overlay onto a
// rendered frame. Pure so the render path and its tests share one definition.
namespace vc3d::overlay_blend
{

struct Luts {
    std::array<uint32_t, 256> color{};
    std::array<float, 256> alpha{};
};

// colormapLut: the windowed colormap (buildWindowLevelColormapLut output).
// opacity: the user opacity in [0,1].
// windowLow/High: the overlay window; values outside it are never drawn, and
//   in value-weighted mode they bound the alpha ramp.
// valueWeightedAlpha: false blends every drawn value at `opacity`; true scales
//   alpha from 0 at windowLow to `opacity` at windowHigh, so a probability map
//   fades in with its confidence instead of appearing as dark smudges.
// tint: the colour of a single-colour colormap (R, G, B in [0,1]) when the
//   colormap is one. In value-weighted mode a tint's colour stays at full
//   strength and only alpha varies; otherwise the colormap LUT (which already
//   darkens the tint toward black) is used as-is.
inline void build(Luts& out,
                  const std::array<uint32_t, 256>& colormapLut,
                  float opacity,
                  float windowLow,
                  float windowHigh,
                  bool valueWeightedAlpha,
                  const std::optional<std::array<float, 3>>& tint)
{
    const float a = std::clamp(opacity, 0.0f, 1.0f);
    const float low = std::clamp(windowLow, 0.0f, 255.0f);
    const float high = std::max(std::clamp(windowHigh, 0.0f, 255.0f), low + 1.0f);
    const float span = high - low;

    uint32_t fullTint = 0;
    const bool constantColor = valueWeightedAlpha && tint.has_value();
    if (constantColor) {
        const auto channel = [](float c) {
            return static_cast<uint32_t>(std::clamp(c, 0.0f, 1.0f) * 255.0f + 0.5f);
        };
        fullTint = 0xFF000000u | (channel((*tint)[0]) << 16) |
                   (channel((*tint)[1]) << 8) | channel((*tint)[2]);
    }

    for (int v = 0; v < 256; ++v) {
        out.color[v] = constantColor ? fullTint : colormapLut[v];
        if (!valueWeightedAlpha) {
            out.alpha[v] = a;
            continue;
        }
        const float weight =
            std::clamp((static_cast<float>(v) - low) / span, 0.0f, 1.0f);
        out.alpha[v] = a * weight;
    }
}

}  // namespace vc3d::overlay_blend
