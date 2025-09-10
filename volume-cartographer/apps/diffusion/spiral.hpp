#pragma once

#include "spiral_common.hpp"

int spiral_main(
    const cv::Mat& slice_mat,
    const std::optional<cv::Vec3f>& umbilicus_point,
    const fs::path& output_path,
    float ray_step_dist,
    const po::variables_map& vm
);