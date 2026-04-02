#pragma once

#include "VCCollection.hpp"

#include "utils/Json.hpp"
#include <opencv2/core.hpp>

// Forward declarations
class QuadSurface;

utils::Json calc_point_winding_metrics(const VCCollection& collection, QuadSurface* surface, const cv::Mat_<float>& winding, int z_min, int z_max);
utils::Json calc_point_metrics(const VCCollection& collection, QuadSurface* surface, int z_min, int z_max);
