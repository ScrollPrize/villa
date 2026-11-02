#pragma once

#include <cmath>
#include <cstdint>
#include <optional>

#include <opencv2/core.hpp>

#include "vc/core/util/QuadSurface.hpp"
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

namespace detail
{
template <typename Vec>
inline double surfaceArea(const cv::Mat_<Vec>& points)
{
    return vc::geometry::surfaceGridArea(
        points.rows, points.cols, [&](int64_t r, int64_t c) -> const Vec& { return points(int(r), int(c)); },
        [&](int64_t r, int64_t c) { return vc::geometry::validSurfacePoint(points(int(r), int(c))); });
}
} // namespace detail

// A grid's area in its vertices' units squared (the store's rule, geometry::surfaceGridArea).
inline double computeSurfaceAreaVox2(const cv::Mat_<cv::Vec3f>& points)
{
    return detail::surfaceArea(points);
}

inline double computeSurfaceAreaVox2(const cv::Mat_<cv::Vec3d>& points)
{
    return detail::surfaceArea(points);
}

inline double computeSurfaceAreaVox2(const QuadSurface& surface)
{
    const auto* pointsPtr = surface.rawPointsPtr();
    if (!pointsPtr || pointsPtr->empty()) {
        return 0.0;
    }
    return detail::surfaceArea(*pointsPtr);
}

} // namespace vc::surface
