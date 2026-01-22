#include "vc/core/util/WrapTrackingLosses.hpp"

#include "vc/core/util/CostFunctions.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>

#include "vc/core/util/Tiff.hpp"

namespace vc::wrap_tracking_losses {

// Global parameters (set from JSON config)
static WrapLossParams g_params;

// Static flags for one-time loss activation logging
static bool logged_tangent_ortho_active = false;
static bool logged_angle_step_active = false;
static bool logged_angle_column_active = false;
static bool logged_radial_slope_active = false;

void set_wrap_loss_params(const WrapLossParams& params) {
    g_params = params;
}

const WrapLossParams& get_wrap_loss_params() {
    return g_params;
}

double median_in_place(std::vector<double>& values) {
    if (values.empty()) return -1.0;
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    return values[mid];
}

static double wrap_angle_diff_deg(double delta) {
    while (delta <= -180.0) delta += 360.0;
    while (delta > 180.0) delta -= 360.0;
    return delta;
}

double compute_seed_theta_reference(
    const core::util::Umbilicus& umbilicus,
    const cv::Vec3d& seed_coord,
    const cv::Mat_<cv::Vec3d>& points,
    const cv::Mat_<uint8_t>& state,
    const cv::Vec2i& seed_loc,
    int window_radius)
{
    cv::Vec3f seed_point(static_cast<float>(seed_coord[0]),
                         static_cast<float>(seed_coord[1]),
                         static_cast<float>(seed_coord[2]));
    double seed_theta = umbilicus.theta(seed_point, 0);

    if (points.empty() || state.empty())
        return seed_theta;

    const int radius = std::max(0, window_radius);
    const int y0 = std::max(0, seed_loc[0] - radius);
    const int y1 = std::min(points.rows - 1, seed_loc[0] + radius);
    const int col = seed_loc[1];
    if (col < 0 || col >= points.cols)
        return seed_theta;

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(y1 - y0 + 1));

    for (int y = y0; y <= y1; ++y) {
        if ((state(y, col) & STATE_COORD_VALID) == 0)
            continue;
        const cv::Vec3d& coord = points(y, col);
        if (!std::isfinite(coord[0]) || !std::isfinite(coord[1]) || !std::isfinite(coord[2]))
            continue;
        if (coord[0] == -1.0 && coord[1] == -1.0 && coord[2] == -1.0)
            continue;
        cv::Vec3f p(static_cast<float>(coord[0]),
                    static_cast<float>(coord[1]),
                    static_cast<float>(coord[2]));
        double theta = umbilicus.theta(p, 0);
        double delta = wrap_angle_diff_deg(theta - seed_theta);
        samples.push_back(seed_theta + delta);
    }

    if (samples.empty())
        return seed_theta;

    double ref = median_in_place(samples);
    while (ref < 0.0) ref += 360.0;
    while (ref >= 360.0) ref -= 360.0;
    return ref;
}

double angle_diff_deg(double a_deg, double b_deg) {
    double diff_rad = (a_deg - b_deg) * M_PI / 180.0;
    return std::atan2(std::sin(diff_rad), std::cos(diff_rad)) * 180.0 / M_PI;
}

void write_wrap_debug_tifs(
    const cv::Mat_<uint8_t>& state,
    const cv::Rect& used_area,
    int generation,
    const std::filesystem::path& tgt_dir,
    wrap_tracking::WrapTracker* wrap_tracker)
{
    if (!wrap_tracker) return;
    cv::Rect bounds(0, 0, state.cols, state.rows);
    cv::Rect out_area = used_area & bounds;
    if (out_area.area() <= 0) return;

    const float invalid = std::numeric_limits<float>::quiet_NaN();
    cv::Mat_<uint8_t> valid_mask(state.size(), static_cast<uint8_t>(0));
    cv::Mat_<float> theta_unwrapped(state.size(), invalid);
    cv::Mat_<float> theta_raw(state.size(), invalid);
    cv::Mat_<float> wrap_index(state.size(), invalid);
    cv::Mat_<float> expected_wrap_index(state.size(), invalid);
    cv::Mat_<float> wrap_index_residual(state.size(), invalid);
    cv::Mat_<float> expected_theta(state.size(), invalid);
    cv::Mat_<float> angle_col_residual(state.size(), invalid);
    cv::Mat_<float> dtheta_raw_step(state.size(), invalid);
    cv::Mat_<float> unwrap_trigger(state.size(), invalid);
    cv::Mat_<float> dtheta_step(state.size(), invalid);
    cv::Mat_<float> dtheta_residual(state.size(), invalid);
    cv::Mat_<float> radial_slope_step(state.size(), invalid);
    cv::Mat_<float> radial_slope_residual(state.size(), invalid);
    cv::Mat_<float> expected_radial_slope(state.size(), invalid);
    cv::Mat_<float> expected_radial_slope_local(state.size(), invalid);
    cv::Mat_<float> radial_slope_residual_local(state.size(), invalid);
    cv::Mat_<float> radius(state.size(), invalid);

    const bool angle_step_ready = (wrap_tracker->expected_dtheta_per_step() != 0.0 &&
                                   wrap_tracker->has_sufficient_dtheta_samples(g_params.angle_step_min_pts));
    const double expected_dtheta = wrap_tracker->expected_dtheta_per_step();

    std::vector<double> expected_theta_col(state.cols, invalid);
    std::vector<bool> expected_theta_ready(state.cols, false);
    std::vector<bool> angle_col_ready(state.cols, false);
    std::vector<bool> radial_slope_ready(state.cols, false);
    for (int col = out_area.x; col < out_area.br().x; ++col) {
        if (wrap_tracker->has_sufficient_column_samples(col, 1)) {
            expected_theta_col[col] = wrap_tracker->expected_theta_for_col(col);
            expected_theta_ready[col] = true;
        }
        if (wrap_tracker->has_sufficient_column_samples(col, g_params.angle_col_min_pts)) {
            angle_col_ready[col] = true;
        }
        if (wrap_tracker->has_sufficient_radial_slope_samples(col, g_params.radial_slope_min_pts)) {
            radial_slope_ready[col] = true;
        }
    }

    for (int row = out_area.y; row < out_area.br().y; ++row) {
        for (int col = out_area.x; col < out_area.br().x; ++col) {
            if ((state(row, col) & STATE_LOC_VALID) == 0) continue;
            auto cell = wrap_tracker->get_cell({row, col});
            if (!(cell.flags & 0x1)) continue;

            valid_mask(row, col) = 255;
            theta_unwrapped(row, col) = static_cast<float>(cell.theta_unwrapped_deg);
            theta_raw(row, col) = static_cast<float>(cell.theta_deg);
            wrap_index(row, col) = static_cast<float>(cell.wrap_index);
            radius(row, col) = static_cast<float>(cell.radius);

            if (expected_theta_ready[col]) {
                const double expected_col_theta = expected_theta_col[col];
                expected_theta(row, col) = static_cast<float>(expected_col_theta);
                if (angle_col_ready[col]) {
                    angle_col_residual(row, col) = static_cast<float>(
                        angle_diff_deg(cell.theta_unwrapped_deg, expected_col_theta));
                }
                double theta_offset = std::round((cell.theta_unwrapped_deg - expected_col_theta) / 360.0) * 360.0;
                double expected_theta_adj = expected_col_theta + theta_offset;
                int expected_wrap = static_cast<int>(std::floor(expected_theta_adj / 360.0));
                expected_wrap_index(row, col) = static_cast<float>(expected_wrap);
                wrap_index_residual(row, col) = static_cast<float>(cell.wrap_index - expected_wrap);
            }

            if (angle_step_ready && col > 0 && (state(row, col - 1) & STATE_LOC_VALID)) {
                auto prev_cell = wrap_tracker->get_cell({row, col - 1});
                if (prev_cell.flags & 0x1) {
                    double dtheta_raw = cell.theta_deg - prev_cell.theta_deg;
                    double dtheta = cell.theta_unwrapped_deg - prev_cell.theta_unwrapped_deg;
                    const bool trigger = (!wrap_tracker->flip_x() && dtheta_raw < -180.0) ||
                                         (wrap_tracker->flip_x() && dtheta_raw > 180.0);
                    dtheta_raw_step(row, col) = static_cast<float>(dtheta_raw);
                    unwrap_trigger(row, col) = trigger ? 1.0f : 0.0f;
                    dtheta_step(row, col) = static_cast<float>(dtheta);
                    dtheta_residual(row, col) = static_cast<float>(dtheta - expected_dtheta);
                }
            }

            if (radial_slope_ready[col] && row > 0 && (state(row - 1, col) & STATE_LOC_VALID)) {
                auto prev_cell = wrap_tracker->get_cell({row - 1, col});
                if (prev_cell.flags & 0x1) {
                    double dz = cell.z - prev_cell.z;
                    if (std::abs(dz) > 1e-6) {
                        double slope = (cell.radius - prev_cell.radius) / dz;
                        double expected_slope = wrap_tracker->expected_radial_slope_for_col(col);
                        expected_radial_slope(row, col) = static_cast<float>(expected_slope);
                        radial_slope_step(row, col) = static_cast<float>(slope);
                        radial_slope_residual(row, col) = static_cast<float>(slope - expected_slope);
                    }
                }
            }

            if (row > 0 && row + 1 < state.rows &&
                (state(row - 1, col) & STATE_LOC_VALID) && (state(row + 1, col) & STATE_LOC_VALID)) {
                auto above_cell = wrap_tracker->get_cell({row - 1, col});
                auto below_cell = wrap_tracker->get_cell({row + 1, col});
                if ((above_cell.flags & 0x1) && (below_cell.flags & 0x1) &&
                    std::isfinite(above_cell.z) && std::isfinite(below_cell.z) &&
                    std::isfinite(above_cell.radius) && std::isfinite(below_cell.radius)) {
                    double dz_above_below = below_cell.z - above_cell.z;
                    if (std::abs(dz_above_below) > 1e-6) {
                        double expected_slope_local = (below_cell.radius - above_cell.radius) / dz_above_below;
                        expected_radial_slope_local(row, col) = static_cast<float>(expected_slope_local);

                        double dz_above = cell.z - above_cell.z;
                        if (std::abs(dz_above) > 1e-6) {
                            double slope_above = (cell.radius - above_cell.radius) / dz_above;
                            radial_slope_residual_local(row, col) =
                                static_cast<float>(slope_above - expected_slope_local);
                        }
                    }
                }
            }
        }
    }

    const std::filesystem::path out_dir = tgt_dir / "wrap_debug";
    std::filesystem::create_directories(out_dir);
    const std::string prefix = "wrap_";

    writeTiff(out_dir / (prefix + "valid_mask.tif"), valid_mask(out_area));
    writeTiff(out_dir / (prefix + "theta_unwrapped.tif"), theta_unwrapped(out_area));
    writeTiff(out_dir / (prefix + "theta_raw.tif"), theta_raw(out_area));
    writeTiff(out_dir / (prefix + "wrap_index.tif"), wrap_index(out_area));
    writeTiff(out_dir / (prefix + "expected_wrap_index.tif"), expected_wrap_index(out_area));
    writeTiff(out_dir / (prefix + "wrap_index_residual.tif"), wrap_index_residual(out_area));
    writeTiff(out_dir / (prefix + "expected_theta.tif"), expected_theta(out_area));
    writeTiff(out_dir / (prefix + "angle_col_residual.tif"), angle_col_residual(out_area));
    writeTiff(out_dir / (prefix + "dtheta_raw_step.tif"), dtheta_raw_step(out_area));
    writeTiff(out_dir / (prefix + "unwrap_trigger.tif"), unwrap_trigger(out_area));
    writeTiff(out_dir / (prefix + "dtheta_step.tif"), dtheta_step(out_area));
    writeTiff(out_dir / (prefix + "dtheta_residual.tif"), dtheta_residual(out_area));
    writeTiff(out_dir / (prefix + "radial_slope_step.tif"), radial_slope_step(out_area));
    writeTiff(out_dir / (prefix + "radial_slope_residual.tif"), radial_slope_residual(out_area));
    writeTiff(out_dir / (prefix + "expected_radial_slope.tif"), expected_radial_slope(out_area));
    writeTiff(out_dir / (prefix + "expected_radial_slope_local.tif"), expected_radial_slope_local(out_area));
    writeTiff(out_dir / (prefix + "radial_slope_residual_local.tif"), radial_slope_residual_local(out_area));
    writeTiff(out_dir / (prefix + "radius.tif"), radius(out_area));
}

int clip_wrap_loss_z_index(
    const core::util::Umbilicus& umbilicus,
    const cv::Vec2i& p,
    double z_value,
    bool* clipped_out)
{
    constexpr int kMaxZClipLogs = 10;
    const int max_z = umbilicus.volume_shape()[0];
    bool clipped = false;
    bool non_finite = false;
    int z_index = 0;

    if (!std::isfinite(z_value)) {
        clipped = true;
        non_finite = true;
        z_index = 0;
    } else {
        z_index = static_cast<int>(z_value);
        if (z_index < 0 || z_index >= max_z) {
            clipped = true;
            z_index = std::clamp(z_index, 0, max_z - 1);
        }
    }

    if (clipped) {
        static std::atomic<int> warn_count{0};
        const int hit = warn_count.fetch_add(1);
        if (hit < kMaxZClipLogs) {
            std::cerr << "[WrapLoss] Clipping z at add_wrap_losses"
                      << " row=" << p[0] << " col=" << p[1]
                      << " z=" << z_value << " -> " << z_index
                      << " (max=" << max_z << ")";
            if (non_finite) {
                std::cerr << " non-finite";
            }
            std::cerr << std::endl;
            if (hit + 1 == kMaxZClipLogs) {
                std::cerr << "[WrapLoss] Further z clip warnings suppressed" << std::endl;
            }
        }
    }

    if (clipped_out) {
        *clipped_out = clipped;
    }
    return z_index;
}

int add_wrap_losses(
    const cv::Vec2i& p,
    cv::Mat_<cv::Vec3d>& points,
    const cv::Mat_<uint8_t>& state,
    ceres::Problem& problem,
    wrap_tracking::WrapTracker* wrap_tracker,
    float weight_scale)
{
    if (!wrap_tracker || weight_scale <= 0.0f) return 0;

    int count = 0;
    const int row = p[0];
    const int col = p[1];
    const bool force_unwrap = (!g_params.wrap_batch_refresh) || (weight_scale != 1.0f);
    if (force_unwrap) {
        wrap_tracker->ensure_row_unwrapped(row, state);
    }
    // Allow losses even if this cell hasn't been propagated into wrap tracking yet.

    const auto& umbilicus = wrap_tracker->umbilicus();
    const double z_value = points(p)[2];
    bool clipped = false;
    const int z_index = clip_wrap_loss_z_index(umbilicus, p, z_value, &clipped);
    cv::Vec3f center = wrap_tracker->cached_center_at(z_index);

    // TangentOrthogonalityLoss - can apply immediately, only needs neighbors
    const float tangent_w = g_params.tangent_ortho_w * weight_scale;
    if (tangent_w > 0.0f) {
        cv::Vec2i p_prev = {p[0], p[1] - 1};
        cv::Vec2i p_next = {p[0], p[1] + 1};

        if (p_prev[1] >= 0 && p_next[1] < points.cols &&
            (state(p_prev) & STATE_LOC_VALID) && (state(p_next) & STATE_LOC_VALID)) {
            if (weight_scale == 1.0f && !logged_tangent_ortho_active) {
                std::cout << "[WrapLoss] TangentOrthogonalityLoss now active (w=" << tangent_w << ")" << std::endl;
                logged_tangent_ortho_active = true;
            }
            problem.AddResidualBlock(
                TangentOrthogonalityLossAnalytic::Create(center, tangent_w),
                nullptr,
                &points(p_prev)[0],
                &points(p)[0],
                &points(p_next)[0]);
            count++;
        }
    }

    // AngleStepLoss - can apply after some data is collected
    const float angle_step_w_scaled = g_params.angle_step_w * weight_scale;
    if (angle_step_w_scaled > 0.0f && wrap_tracker->expected_dtheta_per_step() != 0 &&
        wrap_tracker->has_sufficient_dtheta_samples(g_params.angle_step_min_pts)) {
        cv::Vec2i p_prev = {p[0], p[1] - 1};
        if (p_prev[1] >= 0 && (state(p_prev) & STATE_LOC_VALID)) {
            if (weight_scale == 1.0f && !logged_angle_step_active) {
                std::cout << "[WrapLoss] AngleStepLoss now active (w=" << angle_step_w_scaled
                          << ", dtheta/step=" << wrap_tracker->expected_dtheta_per_step()
                          << "deg, huber=" << g_params.angle_step_huber_delta << "deg)" << std::endl;
                logged_angle_step_active = true;
            }
            ceres::LossFunction* loss = nullptr;
            if (g_params.angle_step_huber_delta > 0.0f) {
                loss = new ceres::HuberLoss(g_params.angle_step_huber_delta);
            }
            problem.AddResidualBlock(
                AngleStepLossAnalytic::Create(center, wrap_tracker->expected_dtheta_per_step(), angle_step_w_scaled),
                loss,
                &points(p_prev)[0],
                &points(p)[0]);
            count++;
        }
    }

    // RadialSlopeLoss - expected radial change along column (local neighbors only)
    const float radial_slope_w_scaled = g_params.radial_slope_w * weight_scale;
    if (radial_slope_w_scaled > 0.0f) {
        cv::Vec2i above = {row - 1, col};
        cv::Vec2i below = {row + 1, col};
        if (surface_helpers::point_in_bounds(state, above) && surface_helpers::point_in_bounds(state, below) &&
            (state(above) & STATE_LOC_VALID) && (state(below) & STATE_LOC_VALID)) {
            auto above_cell = wrap_tracker->get_cell(above);
            auto below_cell = wrap_tracker->get_cell(below);
            if ((above_cell.flags & 0x1) && (below_cell.flags & 0x1) &&
                std::isfinite(above_cell.z) && std::isfinite(below_cell.z) &&
                std::isfinite(above_cell.radius) && std::isfinite(below_cell.radius)) {
                const double dz = below_cell.z - above_cell.z;
                if (std::abs(dz) > 1e-6) {
                    if (weight_scale == 1.0f && !logged_radial_slope_active) {
                        std::cout << "[WrapLoss] RadialSlopeLoss now active (w=" << radial_slope_w_scaled
                                  << ", huber=" << g_params.radial_slope_huber_delta
                                  << ", local_neighbors=1)" << std::endl;
                        logged_radial_slope_active = true;
                    }
                    double expected_slope = (below_cell.radius - above_cell.radius) / dz;

                    auto add_pair = [&](const cv::Vec2i& other) {
                        if (!surface_helpers::point_in_bounds(state, other)) return;
                        if ((state(other) & STATE_LOC_VALID) == 0) return;
                        auto other_cell = wrap_tracker->get_cell(other);
                        if (!(other_cell.flags & 0x1)) return;

                        bool clipped_other = false;
                        const double other_z = points(other)[2];
                        const int other_z_index = clip_wrap_loss_z_index(umbilicus, other, other_z, &clipped_other);
                        cv::Vec3f center_other = wrap_tracker->cached_center_at(other_z_index);

                        ceres::LossFunction* loss = nullptr;
                        if (g_params.radial_slope_huber_delta > 0.0f) {
                            loss = new ceres::HuberLoss(g_params.radial_slope_huber_delta);
                        }
                        problem.AddResidualBlock(
                            RadialSlopeLossAnalytic::Create(center, center_other, expected_slope, radial_slope_w_scaled),
                            loss,
                            &points(p)[0],
                            &points(other)[0]);
                        count++;
                    };

                    add_pair(above);
                    add_pair(below);
                }
            }
        }
    }

    // Losses that require 1+ complete wraps
    if (wrap_tracker->losses_ready()) {
        // AngleColumnLoss
        const float angle_column_w_scaled = g_params.angle_column_w * weight_scale;
        if (angle_column_w_scaled > 0.0f && wrap_tracker->has_sufficient_column_samples(p[1], g_params.angle_col_min_pts)) {
            if (weight_scale == 1.0f && !logged_angle_column_active) {
                std::cout << "[WrapLoss] AngleColumnLoss now active (w=" << angle_column_w_scaled
                          << ", min_pts=" << g_params.angle_col_min_pts
                          << ", huber=" << g_params.angle_column_huber_delta << "deg)" << std::endl;
                logged_angle_column_active = true;
            }
            double expected_theta = wrap_tracker->expected_theta_for_col(p[1]);
            ceres::LossFunction* loss = nullptr;
            if (g_params.angle_column_huber_delta > 0.0f) {
                loss = new ceres::HuberLoss(g_params.angle_column_huber_delta);
            }
            problem.AddResidualBlock(
                AngleColumnLossAnalytic::Create(center, expected_theta, wrap_tracker->base_theta_offset(), angle_column_w_scaled),
                loss,
                &points(p)[0]);
            count++;
        }
    }

    return count;
}

}  // namespace vc::wrap_tracking_losses
