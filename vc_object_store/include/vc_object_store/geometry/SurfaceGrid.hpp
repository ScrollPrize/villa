#pragma once
#include <opencv2/core.hpp>
#include <filesystem>
#include <optional>
#include <vector>
#include <cmath>
#include <span>
#include <vc_object_store/geometry/Geometry.hpp>
#include <stdexcept>
namespace vc::object_links {
// Owned, resident XYZ grid. Coordinates returned by the query layer are grid coordinates.
class SurfaceGrid {
    cv::Mat_<cv::Vec3f> points_;
    cv::Vec2f scale_;
    std::vector<uint8_t> valid_;
public:
    SurfaceGrid(const cv::Mat_<cv::Vec3f>& points, cv::Vec2f scale,
                std::span<const uint8_t> valid = {}) : points_(points.clone()),scale_(scale),valid_(valid.begin(),valid.end()) {
        if (!valid_.empty() && valid_.size()!=points.total()) throw std::invalid_argument("surface validity shape mismatch");
        if(!(scale[0]>0)||!(scale[1]>0)||!std::isfinite(scale[0])||!std::isfinite(scale[1]))
            throw std::invalid_argument("surface scale must be positive and finite");
    }
    const cv::Mat_<cv::Vec3f>& rawPoints() const {return points_;}
    const cv::Mat_<cv::Vec3f>* rawPointsPtr() const {return &points_;}
    bool valid(int row,int col) const {
        const auto& p=points_(row,col);
        return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]) &&
            (valid_.empty() ? vc::geometry::validSurfacePoint(p) : valid_[size_t(row)*points_.cols+col]!=0);
    }
    cv::Vec2f scale() const {return scale_;}
};
// tifxyz x/y/z planes; vertices failing validSurfacePoint become the (-1,-1,-1) sentinel.
SurfaceGrid readSurfaceGrid(const std::filesystem::path& directory);
// mask.tif as per-vertex validity (255 valid, 0 invalid), absent without a mask.
// A mask may be a whole multiple of the grid; a vertex is valid only when every
// mask pixel it covers is at least 255.
std::optional<std::vector<uint8_t>> readSurfaceValidity(const std::filesystem::path& directory, int rows, int cols);
}
