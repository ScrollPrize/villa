#pragma once

#include "VCCollection.hpp"
#include "vc/core/util/Surface.hpp"

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>

nlohmann::json calc_point_winding_metrics(const VCCollection& collection, QuadSurface* surface, const cv::Mat_<float>& winding, int z_min, int z_max);
nlohmann::json calc_point_metrics(const VCCollection& collection, QuadSurface* surface, int z_min, int z_max);
