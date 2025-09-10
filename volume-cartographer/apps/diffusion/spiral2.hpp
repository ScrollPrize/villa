#pragma once

#include "spiral_common.hpp"
#include <vc/ui/VCCollection.hpp>

int spiral2_main(
    const cv::Mat& slice_mat,
    const VCCollection& point_collection,
    const std::optional<cv::Vec3f>& umbilicus_point,
    const std::string& umbilicus_set_name,
    const po::variables_map& vm
);
