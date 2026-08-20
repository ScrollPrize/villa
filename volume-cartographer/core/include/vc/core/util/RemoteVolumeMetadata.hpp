#pragma once

#include <optional>

#include "utils/Json.hpp"

namespace vc
{

// Recursively find the beamline sample-pixel size used by published volume
// metadata. Repeated copies of the same positive value are accepted; conflicting
// values are rejected so traversal order cannot silently select a voxel size.
std::optional<double> findUnambiguousSamplePixelSizeMillimeters(const utils::Json& metadata);

}  // namespace vc
