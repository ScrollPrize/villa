#pragma once

// Wrap tracking Ceres loss functions extracted from GrowSurface.cpp.
// These functions require Ceres and are part of vc_tracer library.

#include <filesystem>
#include <vector>

#include <opencv2/core.hpp>
#include "ceres/ceres.h"

#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/SurfaceHelpers.hpp"
#include "vc/core/util/SurfaceModeling.hpp"
#include "vc/core/util/Umbilicus.hpp"
#include "vc/core/util/WrapTracking.hpp"

namespace vc::wrap_tracking_losses {

// Configuration parameters (set from JSON via set_wrap_loss_params)
struct WrapLossParams {
    float tangent_ortho_w = 0.0f;
    float angle_step_w = 0.0f;
    float angle_column_w = 0.0f;
    float radial_slope_w = 0.0f;
    float inpaint_wrap_loss_scale = 0.2f;
    int angle_col_min_pts = 10;
    int angle_step_min_pts = 5;
    int radial_slope_min_pts = 10;
    float angle_column_huber_delta = 10.0f;
    float angle_step_huber_delta = 5.0f;
    float radial_slope_huber_delta = 0.0f;
    bool wrap_batch_refresh = true;
};

// Set parameters from JSON config (called once during initialization)
void set_wrap_loss_params(const WrapLossParams& params);

// Get current parameters
const WrapLossParams& get_wrap_loss_params();

// Compute median in-place (modifies input vector)
double median_in_place(std::vector<double>& values);

// Compute reference theta at seed using local samples
double compute_seed_theta_reference(
    const core::util::Umbilicus& umbilicus,
    const cv::Vec3d& seed_coord,
    const cv::Mat_<cv::Vec3d>& points,
    const cv::Mat_<uint8_t>& state,
    const cv::Vec2i& seed_loc,
    int window_radius);

// Compute wrapped angle difference in degrees
double angle_diff_deg(double a_deg, double b_deg);

// Write debug tifs for wrap tracking visualization
void write_wrap_debug_tifs(
    const cv::Mat_<uint8_t>& state,
    const cv::Rect& used_area,
    int generation,
    const std::filesystem::path& tgt_dir,
    wrap_tracking::WrapTracker* wrap_tracker);

// Clip z-index for wrap loss computation, returning clamped index
int clip_wrap_loss_z_index(
    const core::util::Umbilicus& umbilicus,
    const cv::Vec2i& p,
    double z_value,
    bool* clipped_out);

// Add wrap-aware losses for a point
// Returns the number of residuals added
int add_wrap_losses(
    const cv::Vec2i& p,
    cv::Mat_<cv::Vec3d>& points,
    const cv::Mat_<uint8_t>& state,
    ceres::Problem& problem,
    wrap_tracking::WrapTracker* wrap_tracker,
    float weight_scale = 1.0f);

}  // namespace vc::wrap_tracking_losses
