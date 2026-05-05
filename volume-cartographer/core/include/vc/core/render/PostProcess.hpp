#pragma once

#include <array>
#include <string>
#include <cstdint>

namespace vc {

// Build a window/level LUT mapping uint8 -> ARGB32 for the fused sampling path.
// Only valid for non-stretch grayscale mode (no colormap, no stretchValues).
// An optional lightFactor (0..1) scales voxel values before the window/level
// mapping, fusing directional lighting into the same LUT.
void buildWindowLevelLut(std::array<uint32_t, 256>& lut,
                         float windowLow, float windowHigh,
                         float lightFactor = 1.0f) noexcept;

// Same as buildWindowLevelLut but fuses in a colormap (by id).
// Empty/unknown id => grayscale (same as buildWindowLevelLut).
void buildWindowLevelColormapLut(std::array<uint32_t, 256>& lut,
                                 float windowLow, float windowHigh,
                                 const std::string& colormapId,
                                 float lightFactor = 1.0f);

}  // namespace vc
