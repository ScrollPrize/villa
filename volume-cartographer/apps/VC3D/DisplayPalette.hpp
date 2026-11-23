#pragma once
#include <vc_reader/display.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <cstdint>

namespace vc {

// Same as buildWindowLevelLut but fuses in a colormap (by id).
// Empty/unknown id => grayscale (same as buildWindowLevelLut).
void buildWindowLevelColormapLut(std::array<uint32_t, 256>& lut,
                                 float windowLow, float windowHigh,
                                 const std::string& colormapId);

// Overlay LUT: premultiplied RGB with per-entry alpha in the top byte, for
// blendOverlayArgb. Default mode is opaque colour inside [windowLow,
// windowHigh] and transparent outside. intensityAsAlpha is transparent below
// the window and opaque from its top: premultiplied maps (spec.premultiplied,
// and grayscale) keep their colour with alpha = its brightest channel, so over
// black they match the opaque colormap; straight maps take the windowed
// intensity as alpha and are scaled by it.
// Discrete (label) maps ignore intensityAsAlpha and always use default mode.
void buildOverlayLut(std::array<uint32_t, 256>& lut,
                     float windowLow, float windowHigh,
                     const std::string& colormapId,
                     bool intensityAsAlpha);

} // namespace vc
