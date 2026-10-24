#pragma once
#include <opencv2/core.hpp>
#include <filesystem>
#include <optional>
#include <vector>
#include <cmath>
#include <stdexcept>
namespace vc::object_links {
// Owned, resident XYZ grid. Coordinates returned by the query layer are grid coordinates.
class SurfaceGrid {
    cv::Mat_<cv::Vec3f> points_;
    cv::Vec2f scale_;
public:
    SurfaceGrid(const cv::Mat_<cv::Vec3f>& points, cv::Vec2f scale) : points_(points),scale_(scale) {
        if(!(scale[0]>0)||!(scale[1]>0)||!std::isfinite(scale[0])||!std::isfinite(scale[1]))
            throw std::invalid_argument("surface scale must be positive and finite");
    }
    const cv::Mat_<cv::Vec3f>& rawPoints() const {return points_;}
    const cv::Mat_<cv::Vec3f>* rawPointsPtr() const {return &points_;}
    cv::Mat_<cv::Vec3f>* rawPointsPtr() {return &points_;}
    cv::Vec2f scale() const {return scale_;}
};
// tifxyz x/y/z planes; z <= 0 becomes the (-1,-1,-1) sentinel.
SurfaceGrid readSurfaceGrid(const std::filesystem::path& directory);
// mask.tif as per-vertex validity (255 valid, 0 invalid), absent without a mask.
// A mask may be a whole multiple of the grid; a vertex is valid only when every
// mask pixel it covers is at least 255.
std::optional<std::vector<uint8_t>> readSurfaceValidity(const std::filesystem::path& directory, int rows, int cols);
}
