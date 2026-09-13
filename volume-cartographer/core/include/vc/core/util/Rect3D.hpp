#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>

struct Rect3D {
    cv::Vec3f low = {0,0,0};
    cv::Vec3f high = {0,0,0};
};

template <typename T>
[[nodiscard]] inline bool contains_point(
    const Rect3D& bounds, const cv::Vec<T, 3>& point) noexcept
{
    return point[0] >= bounds.low[0] && point[0] <= bounds.high[0] &&
           point[1] >= bounds.low[1] && point[1] <= bounds.high[1] &&
           point[2] >= bounds.low[2] && point[2] <= bounds.high[2];
}

bool intersect(const Rect3D &a, const Rect3D &b);
Rect3D expand_rect(const Rect3D &a, const cv::Vec3f &p);

// Bounding box over the valid points of a grid, where (-1,-1,-1) marks an
// invalid point. Returns false and leaves out untouched if none are valid.
bool bbox_of_valid_points(const cv::Mat_<cv::Vec3f>& points, Rect3D& out);
