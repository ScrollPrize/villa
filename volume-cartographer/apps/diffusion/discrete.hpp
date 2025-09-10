#pragma once

#include "common.hpp"
#include <vc/ui/VCCollection.hpp>

struct discrete_options {
    std::string input_volume;
    std::string output_video;
    std::string center;
    float radius;
    int num_steps;
};

int discrete_main(
    const cv::Mat& slice_mat,
    const VCCollection& point_collection,
    const std::optional<cv::Vec3f>& umbilicus_point,
    const std::string& umbilicus_set_name,
    int iterations,
    const fs::path& output_path,
    const fs::path& conflicts_path
);
