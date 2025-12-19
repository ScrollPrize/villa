#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// Compute alpha shape (concave hull) mask from a set of 2D points.
// Uses Delaunay triangulation with alpha criterion to define the boundary.
// Returns a binary mask where pixels inside the alpha shape are 255.
//
// @param points      Vector of 2D points
// @param mask_size   Size of the output mask
// @param alpha       Max edge length for triangles to be included (default: 5.0)
// @return            Binary mask (CV_8U) of the alpha shape interior
inline cv::Mat computeAlphaShapeMask(
    const std::vector<cv::Point2f>& points,
    const cv::Size& mask_size,
    double alpha = 5.0)
{
    if (points.size() < 3) {
        return cv::Mat::zeros(mask_size, CV_8U);
    }

    // Create Delaunay triangulation
    cv::Rect bounds_rect(0, 0, mask_size.width, mask_size.height);
    cv::Subdiv2D subdiv(bounds_rect);
    for (const auto& pt : points) {
        subdiv.insert(pt);
    }

    // Get all triangles
    std::vector<cv::Vec6f> triangles;
    subdiv.getTriangleList(triangles);

    // Create mask by filling triangles that pass alpha criterion
    cv::Mat mask = cv::Mat::zeros(mask_size, CV_8U);

    for (const auto& tri : triangles) {
        cv::Point2f p0(tri[0], tri[1]);
        cv::Point2f p1(tri[2], tri[3]);
        cv::Point2f p2(tri[4], tri[5]);

        // Skip triangles with vertices outside bounds
        if (!bounds_rect.contains(cv::Point(static_cast<int>(p0.x), static_cast<int>(p0.y))) ||
            !bounds_rect.contains(cv::Point(static_cast<int>(p1.x), static_cast<int>(p1.y))) ||
            !bounds_rect.contains(cv::Point(static_cast<int>(p2.x), static_cast<int>(p2.y)))) {
            continue;
        }

        // Alpha criterion: all edges must be shorter than alpha
        double d01 = cv::norm(p1 - p0);
        double d12 = cv::norm(p2 - p1);
        double d20 = cv::norm(p0 - p2);

        if (d01 <= alpha && d12 <= alpha && d20 <= alpha) {
            std::vector<cv::Point> triangle_pts = {
                cv::Point(static_cast<int>(p0.x), static_cast<int>(p0.y)),
                cv::Point(static_cast<int>(p1.x), static_cast<int>(p1.y)),
                cv::Point(static_cast<int>(p2.x), static_cast<int>(p2.y))
            };
            cv::fillConvexPoly(mask, triangle_pts, cv::Scalar(255));
        }
    }

    return mask;
}
