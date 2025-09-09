#pragma once

#include "common.hpp"
#include <vc/ui/VCCollection.hpp>

struct continous_options {
    std::string input_volume;
    std::string output_video;
    std::string center;
    float radius;
    int num_steps;
};

int continous_main(
    const cv::Mat& slice_mat,
    const VCCollection& point_collection,
    const std::optional<cv::Vec3f>& umbilicus_point,
    const std::string& umbilicus_set_name,
    int iterations,
    float ray_step_dist,
    const fs::path& output_path,
    const fs::path& conflicts_path,
    const cv::Mat& mask,
    const po::variables_map& vm
);
