#pragma once
#include <opencv2/core.hpp>
#include <filesystem>
#include <cmath>
#include <stdexcept>
namespace vc::object_links {
// Owned, resident XYZ grid. Coordinates returned by the query layer are grid coordinates.
class PatchGrid {
    cv::Mat_<cv::Vec3f> points_;
    cv::Vec2f scale_;
public:
    PatchGrid(const cv::Mat_<cv::Vec3f>& points, cv::Vec2f scale) : points_(points),scale_(scale) {
        if(!(scale[0]>0)||!(scale[1]>0)||!std::isfinite(scale[0])||!std::isfinite(scale[1]))
            throw std::invalid_argument("patch scale must be positive and finite");
    }
    const cv::Mat_<cv::Vec3f>& rawPoints() const {return points_;}
    const cv::Mat_<cv::Vec3f>* rawPointsPtr() const {return &points_;}
    cv::Mat_<cv::Vec3f>* rawPointsPtr() {return &points_;}
    cv::Vec2f scale() const {return scale_;}
};
PatchGrid readPatchGrid(const std::filesystem::path& directory);
void writePatchGrid(const std::filesystem::path& directory,const PatchGrid& grid);
}
