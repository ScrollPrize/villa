#pragma once

#include "vc/core/PointCollections.hpp"

#include "utils/Json.hpp"
#include <opencv2/core/mat.hpp>

// Forward declarations
class QuadSurface;

utils::Json calc_point_winding_metrics(const PointCollections& collection, QuadSurface* surface, const cv::Mat_<float>& winding, int z_min, int z_max);
utils::Json calc_point_metrics(const PointCollections& collection, QuadSurface* surface, int z_min, int z_max);
