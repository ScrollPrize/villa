
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

#include <vc/core/util/GridStore.hpp>

#include "spiral_common.hpp"
#include "support.hpp"
#include "spiral_ceres.hpp"

ceres::CostFunction* CreateNormalConstraint(const vc::core::util::GridStore& normal_grid, float roi_radius, double weight) {
    return (new ceres::AutoDiffCostFunction<NormalConstraint, 1, 2, 2>(
        new NormalConstraint(normal_grid, roi_radius, weight)));
}

ceres::CostFunction* CreateSnappingConstraint(const vc::core::util::GridStore& normal_grid, float roi_radius, double weight, double snap_trig_th, double snap_search_range) {
    return (new ceres::AutoDiffCostFunction<SnappingConstraint, 1, 2, 2>(
        new SnappingConstraint(normal_grid, roi_radius, weight, snap_trig_th, snap_search_range)));
}

void visualize_normal_constraints(const cv::Mat& sheet_binary, const std::vector<std::pair<cv::Point, cv::Point>>& constraints, const std::string& path) {
    cv::Mat vis_img;
    cv::cvtColor(sheet_binary, vis_img, cv::COLOR_GRAY2BGR);
    cv::RNG rng(12345);
    for (const auto& constraint : constraints) {
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        cv::circle(vis_img, constraint.first, 3, color, -1);
        cv::line(vis_img, constraint.first, constraint.second, color, 1);
    }
    cv::imwrite(path, vis_img);
}

