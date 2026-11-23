#pragma once

#include <cmath>
#include <cstdint>
#include <optional>

#include <opencv2/core.hpp>

#include <nlohmann/json.hpp>
#include <vc_object_store/geometry/Geometry.hpp>

namespace vc::surface
{
// Physical area for a voxel-space area, in cm^2. voxelSize is micrometers per
// voxel, as reported by Volume::voxelSize(). Returns nullopt when the voxel
// size is unusable: a volume whose metadata carries no resolution reports 0,
// and 0 cm^2 for a real surface cannot be told apart from a genuinely tiny one.
inline std::optional<double> areaCm2FromVox2(double area_vx2, double voxelSize)
{
    if (!std::isfinite(area_vx2) || !std::isfinite(voxelSize) || voxelSize <= 0.0) {
        return std::nullopt;
    }
    return area_vx2 * voxelSize * voxelSize / 1e8;
}

// Record a surface's area. area_cm2 is written only when it is derivable, so an
// unknown voxel size leaves the key absent instead of storing a bogus 0.
inline void storeAreaMeta(nlohmann::json& meta, double area_vx2, double voxelSize)
{
    meta["area_vx2"] = area_vx2;
    if (const auto cm2 = areaCm2FromVox2(area_vx2, voxelSize)) {
        meta["area_cm2"] = *cm2;
    } else if (meta.contains("area_cm2")) {
        meta.erase("area_cm2");
    }
}

} // namespace vc::surface
