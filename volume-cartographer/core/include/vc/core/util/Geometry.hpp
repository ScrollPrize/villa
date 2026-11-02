#pragma once

#include <opencv2/core/mat.hpp>

// Bilinear interpolation at fractional coordinates
cv::Vec3f at_int(const cv::Mat_<cv::Vec3f> &points, const cv::Vec2f &p);
float at_int(const cv::Mat_<float> &points, const cv::Vec2f& p);
cv::Vec3d at_int(const cv::Mat_<cv::Vec3d> &points, const cv::Vec2f& p);

// Check whether location lies in a complete valid 2x2 bilinear cell.
// Its floor-origin must satisfy 0 <= row < rows-1 and 0 <= col < cols-1.
// l is [y, x]!
bool loc_valid(const cv::Mat_<cv::Vec3f> &m, const cv::Vec2d &l);
bool loc_valid(const cv::Mat_<cv::Vec3d> &m, const cv::Vec2d &l);
bool loc_valid(const cv::Mat_<float> &m, const cv::Vec2d &l);

// Same complete-cell check with the coordinate order swapped.
// l is [x, y]!
bool loc_valid_xy(const cv::Mat_<cv::Vec3f> &m, const cv::Vec2d &l);
bool loc_valid_xy(const cv::Mat_<cv::Vec3d> &m, const cv::Vec2d &l);
bool loc_valid_xy(const cv::Mat_<float> &m, const cv::Vec2d &l);


cv::Mat_<cv::Vec3f> clean_surface_outliers(
    const cv::Mat_<cv::Vec3f>& points,
    float distance_threshold = 5.0f,
    bool print_stats = false);
