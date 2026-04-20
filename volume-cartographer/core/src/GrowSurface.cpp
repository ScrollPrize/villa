#include <omp.h>
#include <random>

#include "utils/Json.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/Geometry.hpp"
#include "vc/core/util/GridStore.hpp"
#include "vc/core/util/HashFunctions.hpp"
#include "vc/core/util/OMPThreadPointCollection.hpp"
#include "vc/core/util/Slicing.hpp"
#include "vc/core/util/Surface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/tracer/SurfaceModeling.hpp"
#include "vc/core/util/SurfaceArea.hpp"
#include "vc/core/util/SurfacePatchIndex.hpp"
#include "vc/core/util/Tiff.hpp"
#include "vc/core/types/ChunkedTensor.hpp"

#include "vc/core/util/LifeTime.hpp"
#include "vc/core/util/SurfaceHelpers.hpp"
#include "vc/core/util/WrapTrackingLosses.hpp"
#include "vc/core/util/Fringe.hpp"
#include "vc/core/util/WrapTracking.hpp"
#include "vc/core/util/Umbilicus.hpp"

#include <atomic>
#include <chrono>
#include <array>
#include <iomanip>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <shared_mutex>
#include <mutex>
#include <memory>
#include <cmath>
#include <filesystem>
#include <functional>
#include <queue>
#include <limits>
#include <algorithm>
#include <map>
#include <unordered_set>
#include <cmath>
#include <vector>

using namespace vc::surface_helpers;
using namespace vc::wrap_tracking_losses;

int static dbg_counter = 0;

// Config variables - defaults set from JSON parsing below (see "Loss defaults" comment)
static float local_cost_inl_th;
static float assoc_surface_th;
static float duplicate_surface_th;
static float remap_attach_surface_th;
static int point_to_max_iters;
static float straight_weight;
static float straight_weight_3D;
static float sliding_w_scale;
static float z_loc_loss_w;
static float dist_loss_2d_w;
static float dist_loss_3d_w;
static float straight_min_count;
static int inlier_base_threshold;
static bool seed_pointto_from_neighbors;

// Wrap tracking config - defaults set from JSON parsing below (see "Loss defaults" comment)
static float angle_column_w;
static float angle_step_w;
static float radial_slope_w;
static float tangent_ortho_w;
static float inpaint_wrap_loss_scale;
static int angle_col_min_pts;
static int angle_step_min_pts;
static int radial_slope_min_pts;
static float angle_column_huber_delta;
static float angle_step_huber_delta;
static float radial_slope_huber_delta;
static int wrap_stats_update_interval;
static int wrap_debug_tif_interval;
static bool wrap_batch_refresh;
static bool wrap_tracking_in_global;
static float global_wrap_loss_scale;
static int global_step_max_iters;

static vc::core::util::Umbilicus::SeamDirection parse_umbilicus_seam(const std::string& seam) {
    using SeamDirection = vc::core::util::Umbilicus::SeamDirection;
    if (seam == "+x") return SeamDirection::PositiveX;
    if (seam == "-x") return SeamDirection::NegativeX;
    if (seam == "-y") return SeamDirection::NegativeY;
    if (seam == "+y") return SeamDirection::PositiveY;
    std::cout << "warning: unknown umbilicus_seam '" << seam << "', defaulting to +y" << std::endl;
    return SeamDirection::PositiveY;
}

enum class GrowthWindingMode {
    Legacy,
    Clockwise,
    CounterClockwise,
    Bidirectional
};

static std::string to_lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

static GrowthWindingMode parse_growth_winding(const std::string& value, bool* valid_out) {
    if (valid_out) {
        *valid_out = true;
    }
    std::string dir = to_lower_ascii(value);
    dir.erase(std::remove_if(dir.begin(), dir.end(),
                             [](unsigned char ch) { return std::isspace(ch) != 0; }),
              dir.end());
    if (dir.empty() || dir == "legacy" || dir == "flip_x" || dir == "flipx")
        return GrowthWindingMode::Legacy;
    if (dir == "clockwise" || dir == "cw")
        return GrowthWindingMode::Clockwise;
    if (dir == "counterclockwise" || dir == "counter-clockwise" || dir == "ccw")
        return GrowthWindingMode::CounterClockwise;
    if (dir == "bidirectional" || dir == "both" || dir == "bi" || dir == "bidir")
        return GrowthWindingMode::Bidirectional;
    if (valid_out) {
        *valid_out = false;
    }
    return GrowthWindingMode::Legacy;
}

static double wrap_angle_diff_deg(double delta) {
    while (delta <= -180.0) delta += 360.0;
    while (delta > 180.0) delta -= 360.0;
    return delta;
}

static double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

static void add_ms_atomic(double& target, double ms) {
#pragma omp atomic update
    target += ms;
}

static void add_int_atomic(int& target, int value) {
#pragma omp atomic update
    target += value;
}

static std::atomic<uint64_t> g_point_to_bbox_rejects{0};

static uint64_t point_to_bbox_rejects_snapshot() {
    return g_point_to_bbox_rejects.load(std::memory_order_relaxed);
}

static void reset_point_to_bbox_rejects() {
    g_point_to_bbox_rejects.store(0, std::memory_order_relaxed);
}

struct ScopedMs {
    explicit ScopedMs(double& total_ms)
        : total_ms_(total_ms)
        , start_(std::chrono::steady_clock::now())
    {}

    ~ScopedMs() {
        total_ms_ += elapsed_ms(start_);
    }

    double& total_ms_;
    std::chrono::steady_clock::time_point start_;
};

struct ScopedAtomicMs {
    explicit ScopedAtomicMs(double& total_ms)
        : total_ms_(total_ms)
        , start_(std::chrono::steady_clock::now())
    {}

    ~ScopedAtomicMs() {
        add_ms_atomic(total_ms_, elapsed_ms(start_));
    }

    double& total_ms_;
    std::chrono::steady_clock::time_point start_;
};

static void print_ms(std::ostream& os, const char* label, double ms) {
    os << ' ' << label << '=' << std::fixed << std::setprecision(1) << ms << "ms";
}

static void print_count(std::ostream& os, const char* label, uint64_t count) {
    os << ' ' << label << '=' << count;
}

static QuadSurface::PointToStats point_to_stats_delta(const QuadSurface::PointToStats& end,
                                                      const QuadSurface::PointToStats& start) {
    QuadSurface::PointToStats delta;
    delta.calls = end.calls - start.calls;
    delta.initial_hits = end.initial_hits - start.initial_hits;
    delta.index_queries = end.index_queries - start.index_queries;
    delta.index_candidate_cells = end.index_candidate_cells - start.index_candidate_cells;
    delta.index_candidate_searches = end.index_candidate_searches - start.index_candidate_searches;
    delta.index_hits = end.index_hits - start.index_hits;
    delta.index_near_returns = end.index_near_returns - start.index_near_returns;
    delta.point_index_queries = end.point_index_queries - start.point_index_queries;
    delta.point_index_hits = end.point_index_hits - start.point_index_hits;
    delta.random_fallbacks = end.random_fallbacks - start.random_fallbacks;
    delta.random_iterations = end.random_iterations - start.random_iterations;
    delta.random_invalid_skips = end.random_invalid_skips - start.random_invalid_skips;
    delta.random_rescue_scans = end.random_rescue_scans - start.random_rescue_scans;
    delta.random_hits = end.random_hits - start.random_hits;
    delta.misses = end.misses - start.misses;
    return delta;
}

static void add_point_to_stats(QuadSurface::PointToStats& dst, const QuadSurface::PointToStats& src) {
    dst.calls += src.calls;
    dst.initial_hits += src.initial_hits;
    dst.index_queries += src.index_queries;
    dst.index_candidate_cells += src.index_candidate_cells;
    dst.index_candidate_searches += src.index_candidate_searches;
    dst.index_hits += src.index_hits;
    dst.index_near_returns += src.index_near_returns;
    dst.point_index_queries += src.point_index_queries;
    dst.point_index_hits += src.point_index_hits;
    dst.random_fallbacks += src.random_fallbacks;
    dst.random_iterations += src.random_iterations;
    dst.random_invalid_skips += src.random_invalid_skips;
    dst.random_rescue_scans += src.random_rescue_scans;
    dst.random_hits += src.random_hits;
    dst.misses += src.misses;
}

struct OptimizerTiming {
    double total_ms{0.0};
    double setup_ms{0.0};
    double inpaint_mask_ms{0.0};
    double thin_plate_ms{0.0};
    double inpaint_problem_ms{0.0};
    double inpaint_solve_ms{0.0};
    double inpaint_surface_setup_ms{0.0};
    double opt_problem_ms{0.0};
    double opt_solve_ms{0.0};
    double save_inp_hr_ms{0.0};
    double hr_surface_gen_ms{0.0};
    double remap_ms{0.0};
    double remap_filter_ms{0.0};
    double remap_add_iters_ms{0.0};
    double cleanup_ms{0.0};
    double point_to_ms{0.0};

    void add(const OptimizerTiming& other) {
        total_ms += other.total_ms;
        setup_ms += other.setup_ms;
        inpaint_mask_ms += other.inpaint_mask_ms;
        thin_plate_ms += other.thin_plate_ms;
        inpaint_problem_ms += other.inpaint_problem_ms;
        inpaint_solve_ms += other.inpaint_solve_ms;
        inpaint_surface_setup_ms += other.inpaint_surface_setup_ms;
        opt_problem_ms += other.opt_problem_ms;
        opt_solve_ms += other.opt_solve_ms;
        save_inp_hr_ms += other.save_inp_hr_ms;
        hr_surface_gen_ms += other.hr_surface_gen_ms;
        remap_ms += other.remap_ms;
        remap_filter_ms += other.remap_filter_ms;
        remap_add_iters_ms += other.remap_add_iters_ms;
        cleanup_ms += other.cleanup_ms;
        point_to_ms += other.point_to_ms;
    }

    void print(std::ostream& os, const char* prefix) const {
        os << prefix;
        print_ms(os, "total", total_ms);
        print_ms(os, "setup", setup_ms);
        print_ms(os, "inpaint_mask", inpaint_mask_ms);
        print_ms(os, "thin_plate", thin_plate_ms);
        print_ms(os, "inpaint_problem", inpaint_problem_ms);
        print_ms(os, "inpaint_solve", inpaint_solve_ms);
        print_ms(os, "inpaint_surface_setup", inpaint_surface_setup_ms);
        print_ms(os, "opt_problem", opt_problem_ms);
        print_ms(os, "opt_solve", opt_solve_ms);
        print_ms(os, "save_inp_hr", save_inp_hr_ms);
        print_ms(os, "hr_surface_gen", hr_surface_gen_ms);
        print_ms(os, "remap", remap_ms);
        print_ms(os, "remap_filter", remap_filter_ms);
        print_ms(os, "remap_add_iters", remap_add_iters_ms);
        print_ms(os, "cleanup", cleanup_ms);
        print_ms(os, "pointTo_sum", point_to_ms);
        os << std::endl;
    }
};

struct GrowthTimingTotals {
    int generations{0};
    int candidates_processed{0};
    int candidates_accepted{0};
    int global_opt_count{0};
    int wrap_stats_count{0};
    int umbilicus_count{0};

    double generation_total_ms{0.0};
    double point_to_ms{0.0};
    double prune_ms{0.0};
    double build_savable_ms{0.0};
    double invalidate_ms{0.0};
    double full_fringe_rebuild_ms{0.0};
    double candidate_collect_ms{0.0};
    double force_retry_ms{0.0};
    double candidate_sort_ms{0.0};
    double candidate_process_wall_ms{0.0};
    double candidate_sync_ms{0.0};
    double candidate_local_surface_ms{0.0};
    double candidate_ref_init_solve_ms{0.0};
    double candidate_inlier_eval_ms{0.0};
    double candidate_duplicate_check_ms{0.0};
    double candidate_accept_setup_ms{0.0};
    double candidate_wrap_set_ms{0.0};
    double candidate_umbilicus_sample_ms{0.0};
    double candidate_problem_build_ms{0.0};
    double candidate_overlap_attach_ms{0.0};
    double candidate_local_solve_ms{0.0};
    double candidate_post_attach_ms{0.0};
    double candidate_commit_ms{0.0};
    double mirror_ms{0.0};
    double wrap_batch_refresh_ms{0.0};
    double threshold_ms{0.0};
    double loc_count_ms{0.0};
    double debug_save_ms{0.0};
    double global_total_ms{0.0};
    double global_setup_ms{0.0};
    double global_optimize_ms{0.0};
    double global_copy_ms{0.0};
    double global_wrap_rebuild_ms{0.0};
    double global_thread_reset_ms{0.0};
    double global_area_scan_ms{0.0};
    double global_fringe_rebuild_ms{0.0};
    double progress_log_ms{0.0};
    double wrap_stats_total_ms{0.0};
    double wrap_stats_unwrap_ms{0.0};
    double wrap_stats_correct_ms{0.0};
    double wrap_stats_compute_ms{0.0};
    double wrap_debug_tif_ms{0.0};
    double umbilicus_total_ms{0.0};
    double umbilicus_sample_scan_ms{0.0};
    double umbilicus_orient_ms{0.0};
    double umbilicus_row_scan_ms{0.0};
    double umbilicus_retry_ms{0.0};
    double umbilicus_write_json_ms{0.0};
    double umbilicus_build_ms{0.0};
    double umbilicus_tracker_rebuild_ms{0.0};
    double expand_total_ms{0.0};
    double expand_resize_ms{0.0};
    double expand_shift_data_ms{0.0};
    double expand_thread_reset_ms{0.0};
    double expand_wrap_rebuild_ms{0.0};
    double expand_bounds_fringe_ms{0.0};
    double expand_area_scan_ms{0.0};
    double inliers_write_ms{0.0};
    uint64_t point_to_bbox_rejects{0};

    OptimizerTiming optimizer;
    QuadSurface::PointToStats point_to_stats;

    void reset() {
        *this = GrowthTimingTotals{};
    }

    void print(int start_generation, int end_generation) const {
        std::cout << "[Timing] gens " << start_generation << "-" << end_generation
                  << " n=" << generations;
        print_ms(std::cout, "total", generation_total_ms);
        if (generations > 0) {
            print_ms(std::cout, "avg_per_gen", generation_total_ms / generations);
        }
        print_ms(std::cout, "pointTo_sum", point_to_ms);
        std::cout << std::endl;

        std::cout << "[Timing] growth";
        print_ms(std::cout, "prune", prune_ms);
        print_ms(std::cout, "build_savable", build_savable_ms);
        print_ms(std::cout, "invalidate", invalidate_ms);
        print_ms(std::cout, "full_fringe_rebuild", full_fringe_rebuild_ms);
        print_ms(std::cout, "candidate_collect", candidate_collect_ms);
        print_ms(std::cout, "force_retry", force_retry_ms);
        print_ms(std::cout, "candidate_sort", candidate_sort_ms);
        print_ms(std::cout, "candidate_process_wall", candidate_process_wall_ms);
        print_ms(std::cout, "mirror", mirror_ms);
        print_ms(std::cout, "wrap_batch_refresh", wrap_batch_refresh_ms);
        print_ms(std::cout, "threshold", threshold_ms);
        print_ms(std::cout, "loc_count", loc_count_ms);
        print_ms(std::cout, "debug_save", debug_save_ms);
        std::cout << std::endl;

        std::cout << "[Timing] candidates processed=" << candidates_processed
                  << " accepted=" << candidates_accepted;
        print_ms(std::cout, "sync", candidate_sync_ms);
        print_ms(std::cout, "local_surfaces", candidate_local_surface_ms);
        print_ms(std::cout, "ref_init_solve", candidate_ref_init_solve_ms);
        print_ms(std::cout, "inlier_eval", candidate_inlier_eval_ms);
        print_ms(std::cout, "duplicate_check", candidate_duplicate_check_ms);
        print_ms(std::cout, "accept_setup", candidate_accept_setup_ms);
        print_ms(std::cout, "wrap_set", candidate_wrap_set_ms);
        print_ms(std::cout, "umbilicus_sample", candidate_umbilicus_sample_ms);
        print_ms(std::cout, "problem_build", candidate_problem_build_ms);
        print_ms(std::cout, "overlap_attach", candidate_overlap_attach_ms);
        print_ms(std::cout, "local_solve", candidate_local_solve_ms);
        print_ms(std::cout, "post_attach", candidate_post_attach_ms);
        print_ms(std::cout, "commit", candidate_commit_ms);
        std::cout << std::endl;

        std::cout << "[Timing] pointToStats";
        print_count(std::cout, "calls", point_to_stats.calls);
        print_count(std::cout, "bbox_rejects", point_to_bbox_rejects);
        print_count(std::cout, "initial_hits", point_to_stats.initial_hits);
        print_count(std::cout, "index_queries", point_to_stats.index_queries);
        print_count(std::cout, "index_candidate_cells", point_to_stats.index_candidate_cells);
        print_count(std::cout, "index_candidate_searches", point_to_stats.index_candidate_searches);
        print_count(std::cout, "index_hits", point_to_stats.index_hits);
        print_count(std::cout, "index_near_returns", point_to_stats.index_near_returns);
        print_count(std::cout, "point_index_queries", point_to_stats.point_index_queries);
        print_count(std::cout, "point_index_hits", point_to_stats.point_index_hits);
        print_count(std::cout, "random_fallbacks", point_to_stats.random_fallbacks);
        print_count(std::cout, "random_iterations", point_to_stats.random_iterations);
        print_count(std::cout, "random_invalid_skips", point_to_stats.random_invalid_skips);
        print_count(std::cout, "random_rescue_scans", point_to_stats.random_rescue_scans);
        print_count(std::cout, "random_hits", point_to_stats.random_hits);
        print_count(std::cout, "misses", point_to_stats.misses);
        std::cout << std::endl;

        std::cout << "[Timing] wrap stats_count=" << wrap_stats_count;
        print_ms(std::cout, "total", wrap_stats_total_ms);
        print_ms(std::cout, "unwrap", wrap_stats_unwrap_ms);
        print_ms(std::cout, "correct", wrap_stats_correct_ms);
        print_ms(std::cout, "compute", wrap_stats_compute_ms);
        print_ms(std::cout, "debug_tif", wrap_debug_tif_ms);
        std::cout << " umbilicus_count=" << umbilicus_count;
        print_ms(std::cout, "umb_total", umbilicus_total_ms);
        print_ms(std::cout, "sample_scan", umbilicus_sample_scan_ms);
        print_ms(std::cout, "orient", umbilicus_orient_ms);
        print_ms(std::cout, "row_scan", umbilicus_row_scan_ms);
        print_ms(std::cout, "retry", umbilicus_retry_ms);
        print_ms(std::cout, "write_json", umbilicus_write_json_ms);
        print_ms(std::cout, "build_umbilicus", umbilicus_build_ms);
        print_ms(std::cout, "tracker_rebuild", umbilicus_tracker_rebuild_ms);
        std::cout << std::endl;

        std::cout << "[Timing] global count=" << global_opt_count;
        print_ms(std::cout, "total", global_total_ms);
        print_ms(std::cout, "setup", global_setup_ms);
        print_ms(std::cout, "optimizer_call", global_optimize_ms);
        print_ms(std::cout, "copy_back", global_copy_ms);
        print_ms(std::cout, "wrap_rebuild", global_wrap_rebuild_ms);
        print_ms(std::cout, "thread_reset", global_thread_reset_ms);
        print_ms(std::cout, "area_scan", global_area_scan_ms);
        print_ms(std::cout, "fringe_rebuild", global_fringe_rebuild_ms);
        print_ms(std::cout, "progress_log", progress_log_ms);
        std::cout << std::endl;

        optimizer.print(std::cout, "[Timing] optimizer_sum");

        std::cout << "[Timing] expand/io";
        print_ms(std::cout, "expand_total", expand_total_ms);
        print_ms(std::cout, "resize", expand_resize_ms);
        print_ms(std::cout, "shift_data", expand_shift_data_ms);
        print_ms(std::cout, "thread_reset", expand_thread_reset_ms);
        print_ms(std::cout, "wrap_rebuild", expand_wrap_rebuild_ms);
        print_ms(std::cout, "bounds_fringe", expand_bounds_fringe_ms);
        print_ms(std::cout, "area_scan", expand_area_scan_ms);
        print_ms(std::cout, "inliers_write", inliers_write_ms);
        std::cout << std::endl;
    }
};

static bool seed_point_valid(const cv::Vec3f& p) {
    return p[0] != -1.0f && p[1] != -1.0f && p[2] != -1.0f;
}

static bool estimate_theta_sign_from_seed(const cv::Mat_<cv::Vec3f>& seed_points,
                                          const cv::Vec2i& seed_loc,
                                          const vc::core::util::Umbilicus& umbilicus,
                                          int max_offset,
                                          int* out_sign) {
    if (!out_sign)
        return false;
    if (seed_loc[0] < 0 || seed_loc[0] >= seed_points.rows ||
        seed_loc[1] < 0 || seed_loc[1] >= seed_points.cols)
        return false;

    const cv::Vec3f base = seed_points(seed_loc);
    if (!seed_point_valid(base))
        return false;

    const double theta0 = umbilicus.theta(base, 0);
    int best_sign = 0;
    double best_mag = 0.0;

    auto consider = [&](int col, int dir_sign) {
        if (col < 0 || col >= seed_points.cols)
            return;
        const cv::Vec3f neighbor = seed_points(seed_loc[0], col);
        if (!seed_point_valid(neighbor))
            return;
        const double theta1 = umbilicus.theta(neighbor, 0);
        const double delta = wrap_angle_diff_deg(theta1 - theta0);
        const double mag = std::abs(delta);
        if (mag <= 1e-3)
            return;
        int sign = (delta > 0.0) ? 1 : -1;
        if (dir_sign < 0)
            sign = -sign;
        if (mag > best_mag) {
            best_mag = mag;
            best_sign = sign;
        }
    };

    for (int offset = 1; offset <= max_offset; ++offset) {
        consider(seed_loc[1] + offset, 1);
        consider(seed_loc[1] - offset, -1);
    }

    if (best_sign == 0)
        return false;
    *out_sign = best_sign;
    return true;
}

// Types from SurfaceHelpers.hpp are now used:
// - SurfTrackerData, SurfPtrSet, Vec2iLess, SurfPtrLess
// - resId_t, resId_hash, SurfPoint_hash, SurfPoint
// - mix64, det_jitter01, det_jitter_symm
// - at_int_inv, point_in_bounds, maybe_quad_area_and_mark

static bool seed_ptr_from_neighbors(QuadSurface* surf,
                                    SurfTrackerData& data,
                                    const cv::Mat_<uint8_t>& state,
                                    const cv::Vec2i& p,
                                    const cv::Vec3f& target,
                                    cv::Vec3f* ptr_out)
{
    static const std::array<cv::Vec2i, 8> kDirs = {
        cv::Vec2i(0, -1), cv::Vec2i(0, 1), cv::Vec2i(-1, 0), cv::Vec2i(1, 0),
        cv::Vec2i(-1, -1), cv::Vec2i(-1, 1), cv::Vec2i(1, -1), cv::Vec2i(1, 1)
    };

    const auto* surf_points = surf->rawPointsPtr();
    if (!surf_points)
        return false;

    double best_dist = std::numeric_limits<double>::infinity();
    cv::Vec2d best_loc;
    bool found = false;

    for (const auto& dir : kDirs) {
        const cv::Vec2i n = p + dir;
        if (!point_in_bounds(state, n))
            continue;
        if ((state(n) & STATE_LOC_VALID) == 0)
            continue;

        cv::Vec2d loc;
        if (!data.getLoc(surf, n, &loc))
            continue;
        if (!loc_valid(*surf_points, loc))
            continue;

        const cv::Vec3f coord = at_int_inv(*surf_points,
                                           {static_cast<float>(loc[0]), static_cast<float>(loc[1])});
        if (coord[0] == -1)
            continue;

        const double dist = cv::norm(cv::Vec3d(coord) - cv::Vec3d(target));
        if (dist < best_dist) {
            best_dist = dist;
            best_loc = loc;
            found = true;
        }
    }

    if (!found)
        return false;

    const cv::Vec3f center = surf->center();
    const cv::Vec2f scale = surf->scale();
    *ptr_out = cv::Vec3f(static_cast<float>(best_loc[1] - center[0] * scale[0]),
                         static_cast<float>(best_loc[0] - center[1] * scale[1]),
                         0.0f);
    return true;
}

static float pointTo_seeded_neighbor(QuadSurface* surf,
                                     SurfTrackerData& data,
                                     const cv::Mat_<uint8_t>& state,
                                     const cv::Vec2i& p,
                                     const cv::Vec3f& target,
                                     float th,
                                     int max_iters,
                                     SurfacePatchIndex* surface_patch_index,
                                     cv::Vec3f* ptr_out)
{
    auto bbox_excludes_threshold = [](const Rect3D& bbox, const cv::Vec3f& point, float threshold) {
        if (bbox.low[0] == -1.0f || bbox.high[0] == -1.0f)
            return false;

        double dist_sq = 0.0;
        for (int d = 0; d < 3; ++d) {
            if (!std::isfinite(bbox.low[d]) || !std::isfinite(bbox.high[d]) || !std::isfinite(point[d]))
                return false;
            if (point[d] < bbox.low[d]) {
                const double delta = static_cast<double>(bbox.low[d]) - static_cast<double>(point[d]);
                dist_sq += delta * delta;
            } else if (point[d] > bbox.high[d]) {
                const double delta = static_cast<double>(point[d]) - static_cast<double>(bbox.high[d]);
                dist_sq += delta * delta;
            }
        }

        const double threshold_sq = static_cast<double>(threshold) * static_cast<double>(threshold);
        return dist_sq > threshold_sq;
    };

    if (surf && bbox_excludes_threshold(surf->bbox(), target, th)) {
        g_point_to_bbox_rejects.fetch_add(1, std::memory_order_relaxed);
        if (ptr_out)
            *ptr_out = surf->pointer();
        return std::nextafter(th, std::numeric_limits<float>::infinity());
    }

    cv::Vec3f ptr;
    if (!seed_pointto_from_neighbors ||
        !seed_ptr_from_neighbors(surf, data, state, p, target, &ptr)) {
        ptr = surf->pointer();
    }

    const float res = surf->pointTo(ptr, target, th, max_iters, surface_patch_index);
    if (ptr_out)
        *ptr_out = ptr;
    return res;
}

static bool indexed_overlap_probe(QuadSurface& a, QuadSurface& b, SurfacePatchIndex* surface_patch_index, int max_iters)
{
    if (!intersect(a.bbox(), b.bbox()))
        return false;

    cv::Mat_<cv::Vec3f> points = a.rawPoints();
    if (points.cols <= 0 || points.rows <= 0)
        return false;

    const int probes = std::max(10, max_iters / 10);
    uint64_t seed = 1469598103934665603ULL;
    auto mix = [&](const std::string& s) {
        for (unsigned char ch : s) {
            seed ^= ch;
            seed *= 1099511628211ULL;
        }
    };
    mix(a.id);
    mix(b.id);
    std::mt19937 rng(static_cast<uint32_t>(seed ^ (seed >> 32)));
    std::uniform_int_distribution<int> rx(0, points.cols - 1);
    std::uniform_int_distribution<int> ry(0, points.rows - 1);

    for (int r = 0; r < probes; ++r) {
        const int x = rx(rng);
        const int y = ry(rng);
        cv::Vec3f loc = points(y, x);
        if (loc[0] == -1)
            continue;

        cv::Vec3f ptr(0, 0, 0);
        if (b.pointTo(ptr, loc, 2.0f, max_iters, surface_patch_index) <= 2.0f)
            return true;
    }

    return false;
}

static void copy(const SurfTrackerData &src, SurfTrackerData &tgt, const cv::Rect &roi_)
{
    cv::Rect roi(roi_.y,roi_.x,roi_.height,roi_.width);

    {
        auto it = tgt._data.begin();
        while (it != tgt._data.end()) {
            if (roi.contains(cv::Point(it->first.second)))
                it = tgt._data.erase(it);
            else
                ++it;
        }
    }

    {
        auto it = tgt._surfs.begin();
        while (it != tgt._surfs.end()) {
            if (roi.contains(cv::Point(it->first)))
                it = tgt._surfs.erase(it);
            else
                ++it;
        }
    }

    for(auto &it : src._data)
        if (roi.contains(cv::Point(it.first.second)))
            tgt._data[it.first] = it.second;
    for(auto &it : src._surfs)
        if (roi.contains(cv::Point(it.first)))
            tgt._surfs[it.first] = it.second;

    // tgt.seed_loc = src.seed_loc;
    // tgt.seed_coord = src.seed_coord;
}

//try flattening the current surface mapping assuming direct 3d distances
//this is basically just a reparametrization
static void optimize_surface_mapping(SurfTrackerData &data, cv::Mat_<uint8_t> &state, cv::Mat_<cv::Vec3d> &points, cv::Mat_<uint16_t> &generations, cv::Rect used_area,
    cv::Rect static_bounds, float step, float src_step, const cv::Vec2i &seed, int closing_r,
    const std::unordered_map<QuadSurface*, SurfPtrSet>& overlapping_map,
    SurfacePatchIndex* surface_patch_index = nullptr,
    int window_inpaint_max_iters = 100,
    int window_opt_max_iters = 1000,
    bool save_inp_hr = true,
    const std::filesystem::path& tgt_dir = std::filesystem::path(),
    std::string* last_inp_hr_uuid = nullptr,
    bool hr_gen_parallel = true,
    bool remap_parallel = false,
    bool use_cuda_sparse = true,
    vc::wrap_tracking::WrapTracker* wrap_tracker = nullptr,
    bool remap_use_inpaint = false,
    bool debug_diagnostics = false,
    bool skip_inpaint = false,
    OptimizerTiming* timing = nullptr)
{
    const auto optimizer_start = std::chrono::steady_clock::now();
    std::cout << "optimizer: optimizing surface " << state.size() << " " << used_area <<  " " << static_bounds << std::endl;

    const int step_int = surftrack_round_step(step);
    auto count_valid = [&](const cv::Mat_<uint8_t>& st, const cv::Rect& roi, uint8_t mask) {
        cv::Rect clamp_roi = roi & cv::Rect(0, 0, st.cols, st.rows);
        int count = 0;
        for (int j = clamp_roi.y; j < clamp_roi.br().y; ++j)
            for (int i = clamp_roi.x; i < clamp_roi.br().x; ++i)
                if (st(j, i) & mask)
                    count++;
        return count;
    };
    if (debug_diagnostics) {
        int valid_in = count_valid(state, used_area, STATE_LOC_VALID);
        std::cout << "[GrowSurface][diag] optimizer input valid=" << valid_in
                  << " used_area=" << used_area
                  << " static_bounds=" << static_bounds << std::endl;
    }

    cv::Mat_<cv::Vec3d> points_new;
    cv::Mat_<cv::Vec3f> points_f;
    QuadSurface* sm = nullptr;
    std::shared_mutex mutex;
    double pointTo_total_ms = 0.0;
    double timing_dummy_ms = 0.0;
    auto optimizer_timer_target = [&](double OptimizerTiming::*member) -> double& {
        return timing ? timing->*member : timing_dummy_ms;
    };
    SurfTrackerData data_new;
    cv::Rect used_area_hr;
    ceres::Problem problem_inpaint;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    {
        ScopedMs timer(optimizer_timer_target(&OptimizerTiming::setup_ms));
        points_new = points.clone();
        points.convertTo(points_f, CV_32FC3);
        sm = new QuadSurface(points_f, {1,1});

        data_new._data = data._data;

        used_area = cv::Rect(used_area.x-2,used_area.y-2,used_area.size().width+4,used_area.size().height+4);
        // Clamp expanded used_area to valid grid bounds to avoid OOB in subsequent loops
        {
            cv::Rect grid_bounds(0, 0, state.cols, state.rows);
            used_area = used_area & grid_bounds;
        }
        used_area_hr = {used_area.x*step_int, used_area.y*step_int, used_area.width*step_int, used_area.height*step_int};
#ifdef VC_USE_CUDA_SPARSE
        if (use_cuda_sparse && ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::CUDA_SPARSE)) {
            options.linear_solver_type = ceres::SPARSE_SCHUR;
            options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
        } else {
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        }
#else
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
#endif
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = std::max(1, window_inpaint_max_iters);
        options.num_threads = omp_get_max_threads();
        options.use_nonmonotonic_steps = false;

        for(int j=used_area.y;j<used_area.br().y;j++)
            for(int i=used_area.x;i<used_area.br().x;i++)
                if (state(j,i) & STATE_LOC_VALID) {
                    data_new.surfs({j,i}).insert(sm);
                    data_new.loc(sm, {j,i}) = {static_cast<double>(j),static_cast<double>(i)};
                }
    }

    cv::Mat_<uint8_t> new_state = state.clone();

    //generate closed version of state
    cv::Mat m = cv::getStructuringElement(cv::MORPH_RECT, {3,3});

    uint8_t STATE_VALID = STATE_LOC_VALID | STATE_COORD_VALID;

    int res_count = 0;
    cv::Mat_<uint8_t> inpaint_mask(state.size(), static_cast<uint8_t>(0));
    int inpainted_count = 0;
    if (!skip_inpaint) {
        {
            ScopedMs timer(optimizer_timer_target(&OptimizerTiming::inpaint_mask_ms));
            // Build inpaint mask by closing valid regions and marking newly included cells.
            for(int r=0;r<closing_r+2;r++) {
                cv::Mat_<uint8_t> masked;
                bitwise_and(state, STATE_VALID, masked);
                cv::dilate(masked, masked, m, {-1,-1}, r);
                cv::erode(masked, masked, m, {-1,-1}, std::min(r,closing_r));
                // cv::imwrite("masked.tif", masked);

                for(int j=used_area.y;j<used_area.br().y;j++)
                    for(int i=used_area.x;i<used_area.br().x;i++)
                        if ((masked(j,i) & STATE_VALID) && ((new_state(j,i) & STATE_VALID) == 0)) {
                            new_state(j, i) = STATE_COORD_VALID;
                            if (points_new(j, i)[0] == -1)
                                points_new(j, i) = {0,0,0};
                            inpaint_mask(j, i) = 255;
                            inpainted_count++;
                        }
            }
        }

        if (inpainted_count > 0) {
            const int warmup_iters = std::max(10, closing_r);
            const int thin_plate_iters = std::max(50, closing_r * 10);
            std::cout << "optimizer: thin-plate inpaint " << inpainted_count
                      << " cells (warmup " << warmup_iters
                      << ", iters " << thin_plate_iters << ")" << std::endl;
            {
                ScopedMs timer(optimizer_timer_target(&OptimizerTiming::thin_plate_ms));
                thin_plate_inpaint(points_new, new_state, inpaint_mask, used_area, warmup_iters, thin_plate_iters);
            }

            {
                ScopedMs timer(optimizer_timer_target(&OptimizerTiming::inpaint_problem_ms));
                for(int j=used_area.y;j<used_area.br().y;j++)
                    for(int i=used_area.x;i<used_area.br().x;i++)
                        if (inpaint_mask(j, i)) {
                            res_count += surftrack_add_global(sm, {j,i}, data_new, problem_inpaint, new_state, points_new, step*src_step, LOSS_3D_INDIRECT | OPTIMIZE_ALL);
                        }
            }
        }
    }

    {
        ScopedMs timer(optimizer_timer_target(&OptimizerTiming::inpaint_problem_ms));
        for(int j=used_area.y;j<used_area.br().y;j++)
            for(int i=used_area.x;i<used_area.br().x;i++)
                if (state(j,i) & STATE_LOC_VALID)
                    if (problem_inpaint.HasParameterBlock(&points_new(j,i)[0]))
                        problem_inpaint.SetParameterBlockConstant(&points_new(j,i)[0]);
    }

    {
        ScopedMs timer(optimizer_timer_target(&OptimizerTiming::inpaint_solve_ms));
        ceres::Solve(options, &problem_inpaint, &summary);
    }
    std::cout << summary.BriefReport() << std::endl;

    cv::Mat_<cv::Vec3d> points_inpainted;
    cv::Mat_<cv::Vec3f> points_inpainted_f;
    QuadSurface* sm_inp = nullptr;
    SurfTrackerData data_inp;
    {
        ScopedMs timer(optimizer_timer_target(&OptimizerTiming::inpaint_surface_setup_ms));
        points_inpainted = points_new.clone();

        //TODO we could directly use higher res here?
        points_inpainted.convertTo(points_inpainted_f, CV_32FC3);
        sm_inp = new QuadSurface(points_inpainted_f, {1,1});

        data_inp._data = data_new._data;

        for(int j=used_area.y;j<used_area.br().y;j++)
            for(int i=used_area.x;i<used_area.br().x;i++)
                if (new_state(j,i) & STATE_VALID) {
                    data_inp.surfs({j,i}).insert(sm_inp);
                    data_inp.loc(sm_inp, {j,i}) = {static_cast<double>(j),static_cast<double>(i)};
                }
    }

    std::unique_ptr<vc::wrap_tracking::WrapTracker> global_wrap_tracker;
    if (wrap_tracker && wrap_tracking_in_global && global_wrap_loss_scale > 0.0f) {
        ScopedMs timer(optimizer_timer_target(&OptimizerTiming::opt_problem_ms));
        try {
            const auto& umbilicus = wrap_tracker->umbilicus();
            global_wrap_tracker = std::make_unique<vc::wrap_tracking::WrapTracker>(
                umbilicus, points_new.rows, points_new.cols, wrap_tracker->flip_x());

            cv::Vec3d seed_coord = points_new(seed);
            if (!std::isfinite(seed_coord[0]) ||
                !std::isfinite(seed_coord[1]) ||
                !std::isfinite(seed_coord[2]) ||
                (seed_coord[0] == -1.0 && seed_coord[1] == -1.0 && seed_coord[2] == -1.0)) {
                seed_coord = data.seed_coord;
            }
            global_wrap_tracker->initialize_from_seed(seed_coord, seed[1]);

            int repopulated = 0;
            for (int r = 0; r < points_new.rows; ++r) {
                for (int c = 0; c < points_new.cols; ++c) {
                    if (new_state(r, c) & STATE_LOC_VALID) {
                        global_wrap_tracker->set_cell({r, c}, points_new(r, c));
                        repopulated++;
                    }
                }
            }
            for (int row = 0; row < points_new.rows; ++row) {
                global_wrap_tracker->unwrap_row(row, new_state);
            }
            for (int row = 0; row < points_new.rows; ++row) {
                global_wrap_tracker->correct_wrap_from_neighbors(row, new_state);
            }
            global_wrap_tracker->compute_statistics(new_state);

            std::cout << "optimizer: global wrap losses enabled"
                      << " scale=" << global_wrap_loss_scale
                      << " cells=" << repopulated << std::endl;
        } catch (const std::exception& ex) {
            std::cerr << "optimizer: disabling global wrap losses: " << ex.what() << std::endl;
            global_wrap_tracker.reset();
        }
    }

    ceres::Problem problem;

    std::cout << "optimizer: using " << used_area.tl() << used_area.br() << std::endl;

    int fix_points = 0;
    int global_wrap_res_count = 0;
    {
        ScopedMs timer(optimizer_timer_target(&OptimizerTiming::opt_problem_ms));
        for(int j=used_area.y;j<used_area.br().y;j++)
            for(int i=used_area.x;i<used_area.br().x;i++) {
                if (!(new_state(j,i) & STATE_LOC_VALID)) continue;  // skip cells without valid locs
                res_count += surftrack_add_global(sm_inp, {j,i}, data_inp, problem, new_state, points_new, step*src_step, LOSS_3D_INDIRECT | SURF_LOSS | OPTIMIZE_ALL);
                if (global_wrap_tracker) {
                    const int added = add_wrap_losses({j, i}, points_new, new_state, problem,
                                                      global_wrap_tracker.get(), global_wrap_loss_scale);
                    res_count += added;
                    global_wrap_res_count += added;
                }
                fix_points++;
                if (problem.HasParameterBlock(&data_inp.loc(sm_inp, {j,i})[0]))
                    problem.AddResidualBlock(LinChkDistLoss::Create(data_inp.loc(sm_inp, {j,i}), 1.0), nullptr, &data_inp.loc(sm_inp, {j,i})[0]);
            }
    }

    std::cout << "optimizer: num fix points " << fix_points << std::endl;
    if (global_wrap_tracker) {
        std::cout << "optimizer: global wrap residuals " << global_wrap_res_count << std::endl;
    }

    data_inp.seed_loc = seed;
    data_inp.seed_coord = points_new(seed);

    int fix_points_z = 0;
    {
        ScopedMs timer(optimizer_timer_target(&OptimizerTiming::opt_problem_ms));
        for(int j=used_area.y;j<used_area.br().y;j++)
            for(int i=used_area.x;i<used_area.br().x;i++) {
                if (!(new_state(j,i) & STATE_LOC_VALID)) continue;  // skip cells without valid locs
                fix_points_z++;
                if (problem.HasParameterBlock(&data_inp.loc(sm_inp, {j,i})[0]))
                    problem.AddResidualBlock(ZLocationLoss<cv::Vec3d>::Create(points_new, data_inp.seed_coord[2] - (j-data.seed_loc[0])*step*src_step, z_loc_loss_w), new ceres::HuberLoss(1.0), &data_inp.loc(sm_inp, {j,i})[0]);
            }
    }

    std::cout << "optimizer: optimizing " << res_count << " residuals, seed " << seed << std::endl;

    {
        ScopedMs timer(optimizer_timer_target(&OptimizerTiming::opt_problem_ms));
        for(int j=used_area.y;j<used_area.br().y;j++)
            for(int i=used_area.x;i<used_area.br().x;i++)
                if (static_bounds.contains(cv::Point(i,j))) {
                    if (problem.HasParameterBlock(&data_inp.loc(sm_inp, {j,i})[0]))
                        problem.SetParameterBlockConstant(&data_inp.loc(sm_inp, {j,i})[0]);
                    if (problem.HasParameterBlock(&points_new(j, i)[0]))
                        problem.SetParameterBlockConstant(&points_new(j, i)[0]);
                }
    }

    options.max_num_iterations = std::max(1, window_opt_max_iters);
    options.use_nonmonotonic_steps = false;
    options.use_inner_iterations = false;
    {
        ScopedMs timer(optimizer_timer_target(&OptimizerTiming::opt_solve_ms));
        ceres::Solve(options, &problem, &summary);
    }
    std::cout << summary.FullReport() << std::endl;
    std::cout << "optimizer: rms " << sqrt(summary.final_cost/summary.num_residual_blocks) << " count " << summary.num_residual_blocks << std::endl;

    if (save_inp_hr) {
        ScopedMs timer(optimizer_timer_target(&OptimizerTiming::save_inp_hr_ms));
        cv::Mat_<cv::Vec3f> points_hr_inp =
            surftrack_genpoints_hr(data, new_state, points_inpainted, used_area, step, src_step,
                                   /*inpaint=*/true,
                                   /*parallel=*/hr_gen_parallel);
        try {
            auto dbg_surf = new QuadSurface(points_hr_inp(used_area_hr), {1/src_step,1/src_step});
            auto gen_channel = surftrack_generation_channel(generations, used_area, step);
            if (!gen_channel.empty())
                dbg_surf->setChannel("generations", gen_channel);
            // Delete previous inp_hr surface we wrote this session before saving new one
            if (last_inp_hr_uuid && !last_inp_hr_uuid->empty()) {
                std::error_code ec;
                std::filesystem::remove_all(tgt_dir / *last_inp_hr_uuid, ec);
            }
            std::string uuid = Z_DBG_GEN_PREFIX+get_surface_time_str()+"_inp_hr";
            dbg_surf->save(tgt_dir / uuid, uuid);
            if (last_inp_hr_uuid) {
                *last_inp_hr_uuid = uuid;
            }
            delete dbg_surf;
        } catch (cv::Exception&) {
            // We did not find a valid region of interest to expand to
            std::cout << "optimizer: no valid region of interest found" << std::endl;
        }
    }

    cv::Mat_<cv::Vec3f> points_hr;
    {
        ScopedMs timer(optimizer_timer_target(&OptimizerTiming::hr_surface_gen_ms));
        points_hr = surftrack_genpoints_hr(data, new_state, points_inpainted, used_area, step, src_step,
                                           /*inpaint=*/remap_use_inpaint,
                                           /*parallel=*/hr_gen_parallel);
    }
    if (debug_diagnostics) {
        int valid_new_state = count_valid(new_state, used_area, STATE_LOC_VALID | STATE_COORD_VALID);
        std::cout << "[GrowSurface][diag] remap inputs new_state_valid=" << valid_new_state
                  << " inpainted=" << inpainted_count
                  << " points_hr=" << points_hr.cols << "x" << points_hr.rows
                  << " step=" << step
                  << " src_step=" << src_step
                  << " inpaint_hr=" << remap_use_inpaint << std::endl;
    }
    SurfTrackerData data_out;
    cv::Mat_<cv::Vec3d> points_out(points.size(), {-1,-1,-1});
    cv::Mat_<uint8_t> state_out(state.size(), 0);

    std::cout << "remap: start used_area=" << used_area << " parallel=" << (remap_parallel?1:0) << std::endl;
    std::atomic<int> remap_newstate_valid{0};
    std::atomic<int> remap_loc_valid{0};
    std::atomic<int> remap_loc_oob{0};
    std::atomic<int> remap_loc_invalid{0};
    {
    ScopedMs timer(optimizer_timer_target(&OptimizerTiming::remap_ms));
#pragma omp parallel for if(remap_parallel)
    for(int j=used_area.y;j<used_area.br().y;j++)
        for(int i=used_area.x;i<used_area.br().x;i++)
            if (static_bounds.contains(cv::Point(i,j))) {
                points_out(j, i) = points(j, i);
                state_out(j, i) = state(j, i);
                //FIXME copy surfs and locs
                mutex.lock();
                data_out.surfs({j,i}) = data.surfsC({j,i});
                for(auto &s : data_out.surfs({j,i}))
                    data_out.loc(s, {j,i}) = data.loc(s, {j,i});
                mutex.unlock();
            }
            else if (new_state(j,i) & STATE_VALID) {
                remap_newstate_valid.fetch_add(1, std::memory_order_relaxed);
                cv::Vec2d l = data_inp.loc(sm_inp ,{j,i});
                int y = static_cast<int>(l[0]);
                int x = static_cast<int>(l[1]);
                l *= step;
                const bool in_bounds = (l[0] >= 0.0 && l[1] >= 0.0 &&
                                        l[0] < points_hr.rows - 1 && l[1] < points_hr.cols - 1);
                if (!in_bounds)
                    remap_loc_oob.fetch_add(1, std::memory_order_relaxed);
                if (loc_valid(points_hr, l)) {
                    remap_loc_valid.fetch_add(1, std::memory_order_relaxed);
                    // Clamp HR interpolation location to ensure yi+1/xi+1 are in-bounds
                    l[0] = std::max(0.0, std::min<double>(l[0], points_hr.rows - 2 - 1e-6));
                    l[1] = std::max(0.0, std::min<double>(l[1], points_hr.cols - 2 - 1e-6));
                    // Clamp LR indices to ensure neighbor access (y+1,x+1) stays in-bounds
                    y = std::max(0, std::min(y, state.rows - 2));
                    x = std::max(0, std::min(x, state.cols - 2));

                    points_out(j, i) = interp_lin_2d(points_hr, l);
                    state_out(j, i) = STATE_LOC_VALID | STATE_COORD_VALID;

                    SurfPtrSet surfs;
                    const auto& s00 = data.surfsC({y, x});
                    const auto& s01 = data.surfsC({y, x + 1});
                    const auto& s10 = data.surfsC({y + 1, x});
                    const auto& s11 = data.surfsC({y + 1, x + 1});
                    surfs.insert(s00.begin(), s00.end());
                    surfs.insert(s01.begin(), s01.end());
                    surfs.insert(s10.begin(), s10.end());
                    surfs.insert(s11.begin(), s11.end());

                    for (auto& s : surfs) {
                        const float thr = remap_attach_surface_th;
                        // use the same threshold inside pointTo
                        auto _t0 = std::chrono::steady_clock::now();
                        cv::Vec3f ptr;
                        float res = pointTo_seeded_neighbor(s, data, state, {j, i}, points_out(j, i),
                                                            thr, point_to_max_iters, surface_patch_index, &ptr);
                        double _elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t0).count();
                        #pragma omp atomic
                        pointTo_total_ms += _elapsed;
                        if (timing) {
                            add_ms_atomic(timing->point_to_ms, _elapsed);
                        }
                        if (res <= thr) {
                            cv::Vec3f loc = s->loc_raw(ptr);
                            mutex.lock();
                            data_out.surfs({j,i}).insert(s);
                            data_out.loc(s, {j,i}) = {loc[1], loc[0]};
                            mutex.unlock();
                        }
                    }

                    // No fallback attachment; if nothing attaches, orphan cleanup below will drop the point.
                } else {
                    remap_loc_invalid.fetch_add(1, std::memory_order_relaxed);
                }
            }
    }
    std::cout << "remap: done" << std::endl;
    if (debug_diagnostics) {
        std::cout << "[GrowSurface][diag] remap loc_valid=" << remap_loc_valid.load(std::memory_order_relaxed)
                  << " loc_invalid=" << remap_loc_invalid.load(std::memory_order_relaxed)
                  << " loc_oob=" << remap_loc_oob.load(std::memory_order_relaxed)
                  << " new_state_valid=" << remap_newstate_valid.load(std::memory_order_relaxed)
                  << std::endl;
    }

    {
        ScopedMs timer(optimizer_timer_target(&OptimizerTiming::remap_filter_ms));
        //now filter by consistency
        for(int j=used_area.y;j<used_area.br().y-1;j++)
            for(int i=used_area.x;i<used_area.br().x-1;i++)
                if (!static_bounds.contains(cv::Point(i,j)) && state_out(j,i) & STATE_VALID) {
                    SurfPtrSet surf_src = data_out.surfs({j,i});
                    for (auto s : surf_src) {
                        int count;
                        float cost = local_cost(s, {j,i}, data_out, state_out, points_out, step, src_step, &count);
                        if (cost >= local_cost_inl_th /*|| count < 1*/) {
                            data_out.erase(s, {j,i});
                            data_out.eraseSurf(s, {j,i});
                        }
                    }
                }
    }

    const int cols = state_out.cols;
    const int j_start = used_area.y;
    const int i_start = used_area.x;
    const int j_end = used_area.br().y - 1;
    const int i_end = used_area.br().x - 1;

    const int frontier_cap = std::max(0, (j_end - j_start) * (i_end - i_start));
    std::vector<int> frontier;
    frontier.reserve(static_cast<size_t>(frontier_cap));
    for (int j = j_start; j < j_end; ++j)
        for (int i = i_start; i < i_end; ++i)
            if (!static_bounds.contains(cv::Point(i, j)) && (state_out(j, i) & STATE_LOC_VALID))
                frontier.push_back(j * cols + i);

    int added = 1;
    {
    ScopedMs timer(optimizer_timer_target(&OptimizerTiming::remap_add_iters_ms));
    for (int r = 0; r < 30 && added && !frontier.empty(); ++r) {
        ALifeTime timer("optimizer: add iteration\n");

        added = 0;
        std::vector<int> next_frontier;
        next_frontier.reserve(frontier.size());

#pragma omp parallel
        {
            std::vector<int> local_next;
            local_next.reserve(256);
            int local_added = 0;

#pragma omp for schedule(dynamic)
            for (size_t idx = 0; idx < frontier.size(); ++idx) {
                const int pos = frontier[idx];
                const int j = pos / cols;
                const int i = pos - j * cols;

                if (!static_bounds.contains(cv::Point(i,j)) && state_out(j,i) & STATE_LOC_VALID) {
                    mutex.lock_shared();
                    SurfPtrSet surf_cands = data_out.surfs({j,i});
                    for(auto s : data_out.surfs({j,i})) {
                        auto it = overlapping_map.find(s);
                        if (it != overlapping_map.end())
                            surf_cands.insert(it->second.begin(), it->second.end());
                    }
                    mutex.unlock_shared();

                    for(auto test_surf : surf_cands) {
                        mutex.lock_shared();
                        if (data_out.has(test_surf, {j,i})) {
                            mutex.unlock_shared();
                            continue;
                        }
                        mutex.unlock_shared();

                        cv::Vec3f ptr;
                        bool has_seed = false;
                        if (seed_pointto_from_neighbors) {
                            std::shared_lock<std::shared_mutex> lock(mutex);
                            has_seed = seed_ptr_from_neighbors(test_surf, data_out, state_out, {j, i},
                                                               points_out(j, i), &ptr);
                        }
                        if (!has_seed)
                            ptr = test_surf->pointer();
                        auto _t0 = std::chrono::steady_clock::now();
                        float _res = test_surf->pointTo(ptr, points_out(j, i), remap_attach_surface_th, point_to_max_iters, surface_patch_index);
                        double _elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t0).count();
                        #pragma omp atomic
                        pointTo_total_ms += _elapsed;
                        if (timing) {
                            add_ms_atomic(timing->point_to_ms, _elapsed);
                        }
                        if (_res > remap_attach_surface_th)
                            continue;

                        cv::Vec3f loc_3d = test_surf->loc_raw(ptr);
                        int count = 0;
                        int straight_count = 0;
                        float cost;
                        cost = local_cost_snapshot(test_surf, {j,i}, data_out, state_out, points_out,
                                                   step, src_step, loc_3d, mutex, &count, &straight_count);

                        if (cost > local_cost_inl_th)
                            continue;

                        mutex.lock();
                        data_out.surfs({j,i}).insert(test_surf);
                        data_out.loc(test_surf, {j,i}) = {loc_3d[1], loc_3d[0]};
                        mutex.unlock();

                        local_added++;
                        for(int y=j-2;y<=j+2;y++)
                            for(int x=i-2;x<=i+2;x++)
                                if (y >= j_start && y < j_end && x >= i_start && x < i_end)
                                    local_next.push_back(y * cols + x);
                    }
                }
            }

#pragma omp atomic
            added += local_added;

#pragma omp critical
            next_frontier.insert(next_frontier.end(), local_next.begin(), local_next.end());
        }

        if (!next_frontier.empty()) {
            std::sort(next_frontier.begin(), next_frontier.end());
            next_frontier.erase(std::unique(next_frontier.begin(), next_frontier.end()), next_frontier.end());
        }
        frontier.swap(next_frontier);

        std::cout << "optimizer: added " << added << std::endl;
    }
    }

    if (debug_diagnostics) {
        int valid_pre_clean = 0;
        int attached_pre_clean = 0;
        for (int j = used_area.y; j < used_area.br().y; ++j)
            for (int i = used_area.x; i < used_area.br().x; ++i)
                if (state_out(j, i) & STATE_LOC_VALID) {
                    valid_pre_clean++;
                    if (!data_out.surfsC({j, i}).empty())
                        attached_pre_clean++;
                }
        std::cout << "[GrowSurface][diag] remap pre-clean valid=" << valid_pre_clean
                  << " attached=" << attached_pre_clean << std::endl;
    }

    {
    ScopedMs timer(optimizer_timer_target(&OptimizerTiming::cleanup_ms));
    //reset unsupported points
#pragma omp parallel for
    for(int j=used_area.y;j<used_area.br().y-1;j++)
        for(int i=used_area.x;i<used_area.br().x-1;i++)
            if (!static_bounds.contains(cv::Point(i,j))) {
                if (state_out(j,i) & STATE_LOC_VALID) {
                    if (data_out.surfs({j,i}).empty()) {
                        state_out(j,i) = 0;
                        points_out(j, i) = {-1,-1,-1};
                    }
                }
                else {
                    state_out(j,i) = 0;
                    points_out(j, i) = {-1,-1,-1};
                }
            }

    if (debug_diagnostics) {
        int valid_post_clean = count_valid(state_out, used_area, STATE_LOC_VALID);
        std::cout << "[GrowSurface][diag] remap post-clean valid=" << valid_post_clean << std::endl;
    }

    points = points_out;
    state = state_out;
    data = data_out;
    data.seed_loc = seed;
    data.seed_coord = points(seed);

    for (int j = used_area.y; j < used_area.br().y; ++j)
        for (int i = used_area.x; i < used_area.br().x; ++i)
            if ((state(j,i) & STATE_LOC_VALID) == 0)
                generations(j,i) = 0;

    // Cleanup temporary QuadSurface objects to avoid memory leak
    delete sm;
    delete sm_inp;
    }

    if (timing) {
        timing->total_ms += elapsed_ms(optimizer_start);
    }
    dbg_counter++;
}

QuadSurface *grow_surf_from_surfs(QuadSurface *seed, const std::vector<QuadSurface*> &surfs_v, const utils::Json &params, float voxelsize)
{
    // Timing accumulator for pointTo calls (thread-safe via omp atomic)
    double pointTo_total_ms = 0.0;

    bool flip_x_param = params.value("flip_x", 0);
    std::string wrap_growth_direction_raw = params.value("wrap_growth_direction", std::string(""));
    bool wrap_growth_direction_valid = true;
    GrowthWindingMode growth_mode = parse_growth_winding(wrap_growth_direction_raw, &wrap_growth_direction_valid);
    bool wrap_direction_override = (growth_mode != GrowthWindingMode::Legacy);
    bool bidirectional_growth = (growth_mode == GrowthWindingMode::Bidirectional);
    bool mirror_pending = (!wrap_direction_override) ? flip_x_param : false;
    bool mirror_done = false;
    int mirror_ready_generation = 1;
    bool wrap_flip_x = flip_x_param;
    bool wrap_flip_x_after_mirror = wrap_flip_x;
    bool wrap_direction_applied = false;
    int global_steps_per_window = params.value("global_steps_per_window", 0);
    bool debug_diagnostics = params.value("debug_diagnostics", false);
    bool remap_use_inpaint = params.value("remap_use_inpaint", false);
    int window_inpaint_max_iters = params.value("window_inpaint_max_iters", 100);
    if (window_inpaint_max_iters < 1)
        window_inpaint_max_iters = 1;
    bool skip_inpaint = params.value("skip_inpaint", false);
    int window_opt_max_iters = params.value("window_opt_max_iters", 1000);
    if (window_opt_max_iters < 1)
        window_opt_max_iters = 1;
    int timing_report_interval = params.value("timing_report_interval", 500);
    if (timing_report_interval < 0)
        timing_report_interval = 0;


    std::cout << "global_steps_per_window: " << global_steps_per_window << std::endl;
    std::cout << "window_inpaint_max_iters: " << window_inpaint_max_iters << std::endl;
    std::cout << "window_opt_max_iters: " << window_opt_max_iters << std::endl;
    std::cout << "global_step_max_iters: " << global_step_max_iters << std::endl;
    std::cout << "timing_report_interval: " << timing_report_interval << std::endl;
    std::cout << "flip_x: " << flip_x_param << std::endl;
    if (!wrap_growth_direction_raw.empty() && !wrap_growth_direction_valid) {
        std::cout << "warning: unknown wrap_growth_direction '" << wrap_growth_direction_raw
                  << "', using legacy flip_x" << std::endl;
    }
    std::filesystem::path tgt_dir = params["tgt_dir"].get_string();

    // Track surfaces we wrote this session so we can delete them when writing new ones
    std::string last_hr_surface_uuid;
    std::string last_inp_hr_surface_uuid;

    std::unordered_map<std::string,QuadSurface*> surfs;
    const bool src_step_set = params.contains("src_step");
    float src_step = params.value("src_step", 20.0f);
    float step = params.value("step", 10.0f);
    int max_width = params.value("max_width", 80000);

    // =========================================================================
    // Loss defaults - ALL loss weight defaults are defined here
    // =========================================================================
    // Surface tracking loss weights
    local_cost_inl_th       = params.value("local_cost_inl_th",       0.2f);
    straight_weight         = params.value("straight_weight",         0.7f);
    straight_weight_3D      = params.value("straight_weight_3D",      4.0f);
    z_loc_loss_w            = params.value("z_loc_loss_w",            0.1f);
    dist_loss_2d_w          = params.value("dist_loss_2d_w",          1.0f);
    dist_loss_3d_w          = params.value("dist_loss_3d_w",          2.0f);
    straight_min_count      = params.value("straight_min_count",      1.0f);
    sliding_w_scale         = params.value("sliding_w_scale",         1.0f);
    inlier_base_threshold   = params.value("inlier_base_threshold",   20);

    // Wrap tracking loss weights (0 = disabled)
    tangent_ortho_w         = params.value("tangent_ortho_w",         0.0f);
    angle_step_w            = params.value("angle_step_w",            0.0f);
    angle_column_w          = params.value("angle_column_w",          0.0f);
    radial_slope_w          = params.value("radial_slope_w",          0.0f);
    inpaint_wrap_loss_scale = params.value("inpaint_wrap_loss_scale", 0.2f);

    // Wrap tracking min sample counts
    angle_col_min_pts       = params.value("angle_col_min_pts",       10);
    angle_step_min_pts      = params.value("angle_step_min_pts",      5);
    radial_slope_min_pts    = params.value("radial_slope_min_pts",    10);

    // Wrap tracking Huber deltas (0 = no robustification)
    angle_column_huber_delta = params.value("angle_column_huber_delta", 10.0f);
    angle_step_huber_delta   = params.value("angle_step_huber_delta",   5.0f);
    radial_slope_huber_delta = params.value("radial_slope_huber_delta", 0.0f);
    // =========================================================================

    // Clamp negative values
    if (inpaint_wrap_loss_scale < 0.0f) inpaint_wrap_loss_scale = 0.0f;
    if (angle_column_huber_delta < 0.0f) angle_column_huber_delta = 0.0f;
    if (angle_step_huber_delta < 0.0f) angle_step_huber_delta = 0.0f;
    if (radial_slope_huber_delta < 0.0f) radial_slope_huber_delta = 0.0f;

    // Surface association thresholds
    assoc_surface_th = params.value("same_surface_th", 2.0f);
    duplicate_surface_th = params.value("duplicate_surface_th", assoc_surface_th);
    remap_attach_surface_th = params.value("remap_attach_surface_th", assoc_surface_th);

    // PointTo settings
    point_to_max_iters = params.value("point_to_max_iters", 10);
    int point_to_seed_max_iters = params.value("point_to_seed_max_iters", 1000);
    seed_pointto_from_neighbors = params.value("seed_pointto_from_neighbors", false);

    // Surface patch index settings
    bool use_surface_patch_index = params.value("use_surface_patch_index", false);
    int surface_patch_stride = params.value("surface_patch_stride", 1);
    float surface_patch_bbox_pad = params.value("surface_patch_bbox_pad", 0.0f);
    bool surface_patch_cache = params.value("surface_patch_cache", true);
    std::filesystem::path surface_patch_cache_dir = tgt_dir / ".surface_patch_index_cache";
    if (params.contains("surface_patch_cache_dir") &&
        params["surface_patch_cache_dir"].is_string()) {
        surface_patch_cache_dir = params["surface_patch_cache_dir"].get_string();
    }
    int max_local_surfs = params.value("max_local_surfs", 0);
    if (max_local_surfs < 0) max_local_surfs = 0;

    // Wrap tracking runtime settings
    wrap_stats_update_interval = params.value("wrap_stats_update_interval", 50);
    if (wrap_stats_update_interval < 1) wrap_stats_update_interval = 50;
    wrap_debug_tif_interval = params.value("wrap_debug_tif_interval", 500);
    if (wrap_debug_tif_interval < 1) wrap_debug_tif_interval = 0;
    wrap_batch_refresh = params.value("wrap_batch_refresh", true);
    wrap_tracking_in_global = params.value("wrap_tracking_in_global", false);
    global_wrap_loss_scale = params.value("global_wrap_loss_scale", inpaint_wrap_loss_scale);
    if (global_wrap_loss_scale < 0.0f) global_wrap_loss_scale = 0.0f;
    global_step_max_iters = params.value("global_step_max_iters", 200);
    if (global_step_max_iters < 1) global_step_max_iters = 1;

    // Initialize loss params for extracted helper functions
    {
        LossParams lp;
        lp.local_cost_inl_th = local_cost_inl_th;
        lp.straight_weight = straight_weight;
        lp.straight_weight_3D = straight_weight_3D;
        lp.z_loc_loss_w = z_loc_loss_w;
        lp.dist_loss_2d_w = dist_loss_2d_w;
        lp.dist_loss_3d_w = dist_loss_3d_w;
        lp.straight_min_count = straight_min_count;
        set_loss_params(lp);
    }
    {
        WrapLossParams wp;
        wp.tangent_ortho_w = tangent_ortho_w;
        wp.angle_step_w = angle_step_w;
        wp.angle_column_w = angle_column_w;
        wp.radial_slope_w = radial_slope_w;
        wp.inpaint_wrap_loss_scale = inpaint_wrap_loss_scale;
        wp.angle_col_min_pts = angle_col_min_pts;
        wp.angle_step_min_pts = angle_step_min_pts;
        wp.radial_slope_min_pts = radial_slope_min_pts;
        wp.angle_column_huber_delta = angle_column_huber_delta;
        wp.angle_step_huber_delta = angle_step_huber_delta;
        wp.radial_slope_huber_delta = radial_slope_huber_delta;
        wp.wrap_batch_refresh = wrap_batch_refresh;
        set_wrap_loss_params(wp);
    }

    std::string umbilicus_seam = params.value("umbilicus_seam", "+y");

    const int step_int = surftrack_round_step(step);
    uint64_t deterministic_seed = uint64_t(params.value("deterministic_seed", 5489));
    double deterministic_jitter_px = params.value("deterministic_jitter_px", 0.15);
    std::string candidate_ordering = params.value("candidate_ordering", "row_col");
    bool candidate_min_dist_set = params.contains("candidate_min_dist");
    int candidate_min_dist = params.value("candidate_min_dist", 0);
    if (candidate_min_dist < 0)
        candidate_min_dist = 0;
    int neighbor_connectivity = params.value("neighbor_connectivity", 4);
    int force_retry_min_neighbors = params.value("force_retry_min_neighbors", 3);
    int force_retry_max = params.value("force_retry_max", 3);
    float fringe_savable_dist = params.value("fringe_savable_dist", 25.0f);
    std::string fringe_savable_metric = to_lower_ascii(params.value("fringe_savable_metric", "l2"));
    int fringe_max_attempts = params.value("fringe_max_attempts", 0);
    int max_height = params.value("max_height", 15000);
    int misconnect_prune_interval = params.value("misconnect_prune_interval", 1000);
    int misconnect_prune_kernel = params.value("misconnect_prune_kernel", 3);
    std::string misconnect_prune_mode = to_lower_ascii(params.value("misconnect_prune_mode", "erosion"));
    float misconnect_bottleneck_radius = params.value("misconnect_bottleneck_radius", -1.0f);
    int misconnect_prune_source_band = params.value("misconnect_prune_source_band", 1);
    int fringe_invalidate_interval = params.value("fringe_invalidate_interval", 0);
    int fringe_full_rebuild_interval = params.value("fringe_full_rebuild_interval", 0);
    bool fringe_full_boundary = params.value("fringe_full_boundary", false);
    bool resume_mode = params.value("resume", false);
    int resume_max_local_surfs = params.value("resume_max_local_surfs", 12);
    if (resume_max_local_surfs < 0)
        resume_max_local_surfs = 0;
    int min_test_surface_neighbor_count =
        params.value("min_test_surface_neighbor_count", resume_mode ? 2 : 1);
    if (min_test_surface_neighbor_count < 1)
        min_test_surface_neighbor_count = 1;
    if (force_retry_min_neighbors < 1)
        force_retry_min_neighbors = 0;
    if (force_retry_max < 1)
        force_retry_max = 0;
    if (force_retry_max > 255)
        force_retry_max = 255;
    if (fringe_savable_dist < 0.0f)
        fringe_savable_dist = 0.0f;
    if (misconnect_prune_interval < 1)
        misconnect_prune_interval = 0;
    if (misconnect_prune_kernel < 1)
        misconnect_prune_kernel = 1;
    if ((misconnect_prune_kernel % 2) == 0)
        misconnect_prune_kernel += 1;
    if (misconnect_prune_mode != "erosion" && misconnect_prune_mode != "bottleneck") {
        std::cout << "warning: unknown misconnect_prune_mode '" << misconnect_prune_mode
                  << "', using erosion" << std::endl;
        misconnect_prune_mode = "erosion";
    }
    if (misconnect_bottleneck_radius < 0.0f)
        misconnect_bottleneck_radius = static_cast<float>((misconnect_prune_kernel - 1) / 2);
    if (misconnect_prune_source_band < 1)
        misconnect_prune_source_band = 1;
    if (fringe_savable_metric == "manhattan" || fringe_savable_metric == "taxicab")
        fringe_savable_metric = "l1";
    if (fringe_savable_metric != "l1" && fringe_savable_metric != "l2") {
        std::cout << "warning: unknown fringe_savable_metric '" << fringe_savable_metric
                  << "', using l2" << std::endl;
        fringe_savable_metric = "l2";
    }
    if (fringe_max_attempts < 0)
        fringe_max_attempts = 0;
    if (fringe_max_attempts > std::numeric_limits<uint16_t>::max())
        fringe_max_attempts = std::numeric_limits<uint16_t>::max();
    if (max_height < 0)
        max_height = 0;
    if (fringe_invalidate_interval < 1)
        fringe_invalidate_interval = 0;
    if (fringe_full_rebuild_interval < 1)
        fringe_full_rebuild_interval = 0;
    if (neighbor_connectivity != 4 && neighbor_connectivity != 8) {
        std::cout << "warning: neighbor_connectivity must be 4 or 8; using 4" << std::endl;
        neighbor_connectivity = 4;
    }
    bool use_spread_out_ordering = false;
    if (candidate_ordering == "legacy") {
        use_spread_out_ordering = true;
        if (!candidate_min_dist_set)
            candidate_min_dist = 9;
    } else if (candidate_ordering == "spread_out" || candidate_ordering == "spread-out") {
        use_spread_out_ordering = true;
        if (!candidate_min_dist_set)
            candidate_min_dist = 9;
        std::cout << "warning: candidate_ordering '" << candidate_ordering
                  << "' is deprecated; use 'legacy' to match SurfaceHelpers" << std::endl;
        candidate_ordering = "legacy";
    } else if (candidate_ordering != "row_col") {
        std::cout << "warning: unknown candidate_ordering '" << candidate_ordering
                  << "', using row_col" << std::endl;
        candidate_ordering = "row_col";
    }

    // Optional hard z-range constraint: [z_min, z_max]
    bool enforce_z_range = false;
    double z_min = 0.0, z_max = 0.0;
    if (params.contains("z_range")) {
        try {
            if (params["z_range"].is_array() && params["z_range"].size() == 2) {
                z_min = params["z_range"][0].get_double();
                z_max = params["z_range"][1].get_double();
                if (z_min > z_max)
                    std::swap(z_min, z_max);
                enforce_z_range = true;
            }
        } catch (...) {
            // Ignore malformed z_range silently; fall back to no constraint
            enforce_z_range = false;
        }
    } else if (params.contains("z_min") && params.contains("z_max")) {
        try {
            z_min = params["z_min"].get_double();
            z_max = params["z_max"].get_double();
            if (z_min > z_max)
                std::swap(z_min, z_max);
            enforce_z_range = true;
        } catch (...) {
            enforce_z_range = false;
        }
    }

    std::cout << "  local_cost_inl_th: " << local_cost_inl_th << std::endl;
    std::cout << "  assoc_surface_th: " << assoc_surface_th << std::endl;
    std::cout << "  duplicate_surface_th: " << duplicate_surface_th << std::endl;
    std::cout << "  remap_attach_surface_th: " << remap_attach_surface_th << std::endl;
    std::cout << "  point_to_max_iters: " << point_to_max_iters << std::endl;
    std::cout << "  point_to_seed_max_iters: " << point_to_seed_max_iters << std::endl;
    std::cout << "  use_surface_patch_index: " << use_surface_patch_index << std::endl;
    std::cout << "  surface_patch_stride: " << surface_patch_stride << std::endl;
    std::cout << "  surface_patch_bbox_pad: " << surface_patch_bbox_pad << std::endl;
    std::cout << "  surface_patch_cache: " << (surface_patch_cache ? "true" : "false") << std::endl;
    if (surface_patch_cache)
        std::cout << "  surface_patch_cache_dir: " << surface_patch_cache_dir << std::endl;
    std::cout << "  straight_weight: " << straight_weight << std::endl;
    std::cout << "  straight_weight_3D: " << straight_weight_3D << std::endl;
    std::cout << "  straight_min_count: " << straight_min_count << std::endl;
    std::cout << "  inlier_base_threshold: " << inlier_base_threshold << std::endl;
    std::cout << "  sliding_w_scale: " << sliding_w_scale << std::endl;
    std::cout << "  z_loc_loss_w: " << z_loc_loss_w << std::endl;
    std::cout << "  dist_loss_2d_w: " << dist_loss_2d_w << std::endl;
    std::cout << "  dist_loss_3d_w: " << dist_loss_3d_w << std::endl;
    std::cout << "  deterministic_seed: " << deterministic_seed << std::endl;
    std::cout << "  deterministic_jitter_px: " << deterministic_jitter_px << std::endl;
    std::cout << "  candidate_ordering: " << candidate_ordering << std::endl;
    std::cout << "  candidate_min_dist: " << candidate_min_dist << std::endl;
    std::cout << "  neighbor_connectivity: " << neighbor_connectivity << std::endl;
    std::cout << "  fringe_full_boundary: " << (fringe_full_boundary ? "true" : "false") << std::endl;
    std::cout << "  fringe_savable_dist: " << fringe_savable_dist << std::endl;
    std::cout << "  fringe_savable_metric: " << fringe_savable_metric << std::endl;
    std::cout << "  fringe_max_attempts: " << fringe_max_attempts << std::endl;
    std::cout << "  force_retry_min_neighbors: " << force_retry_min_neighbors << std::endl;
    std::cout << "  force_retry_max: " << force_retry_max << std::endl;
    std::cout << "  max_height: " << max_height << std::endl;
    std::cout << "  misconnect_prune_interval: " << misconnect_prune_interval << std::endl;
    std::cout << "  misconnect_prune_kernel: " << misconnect_prune_kernel << std::endl;
    std::cout << "  misconnect_prune_mode: " << misconnect_prune_mode << std::endl;
    std::cout << "  misconnect_bottleneck_radius: " << misconnect_bottleneck_radius << std::endl;
    std::cout << "  misconnect_prune_source_band: " << misconnect_prune_source_band << std::endl;
    std::cout << "  fringe_invalidate_interval: " << fringe_invalidate_interval << std::endl;
    std::cout << "  fringe_full_rebuild_interval: " << fringe_full_rebuild_interval << std::endl;
    std::cout << "  max_local_surfs: " << max_local_surfs << std::endl;
    std::cout << "  resume_max_local_surfs: " << resume_max_local_surfs << std::endl;
    std::cout << "  min_test_surface_neighbor_count: " << min_test_surface_neighbor_count << std::endl;
    std::cout << "  seed_pointto_from_neighbors: " << (seed_pointto_from_neighbors ? 1 : 0) << std::endl;
    std::cout << "  remap_use_inpaint: " << (remap_use_inpaint ? "true" : "false") << std::endl;
    std::cout << "  wrap_tracking_in_global: " << (wrap_tracking_in_global ? "true" : "false") << std::endl;
    std::cout << "  global_wrap_loss_scale: " << global_wrap_loss_scale << std::endl;
    if (enforce_z_range)
        std::cout << "  z_range: [" << z_min << ", " << z_max << "]" << std::endl;

    std::cout << "total surface count: " << surfs_v.size() << std::endl;

    for(auto &sm : surfs_v) {
        if (!sm->meta.contains("tags") || !sm->meta.at("tags").contains("defective")) {
            surfs[sm->id] = sm;
        }
    }

    // Build overlapping map: for each surface, collect pointers to its overlapping surfaces
    std::unordered_map<QuadSurface*, SurfPtrSet> overlapping_map;
    for(auto &sm : surfs_v)
        for(const auto& name : sm->overlappingIds())
            if (surfs.contains(name))
                overlapping_map[sm].insert(surfs[name]);

    const bool need_resume_overlap_index = resume_mode && seed->overlappingIds().empty();
    SurfacePatchIndex patch_index;
    SurfacePatchIndex* patch_index_ptr = nullptr;
    if (use_surface_patch_index || need_resume_overlap_index) {
        std::vector<SurfacePatchIndex::SurfacePtr> patch_surfaces;
        if (need_resume_overlap_index) {
            patch_surfaces.reserve(surfs_v.size());
            for (auto* sm : surfs_v) {
                if (sm)
                    patch_surfaces.emplace_back(SurfacePatchIndex::SurfacePtr(sm, [](QuadSurface*) {}));
            }
        } else {
            patch_surfaces.reserve(surfs.size());
            for (const auto& it : surfs)
                patch_surfaces.emplace_back(SurfacePatchIndex::SurfacePtr(it.second, [](QuadSurface*) {}));
        }
        patch_index.setSamplingStride(surface_patch_stride);
        bool cache_hit = false;
        std::filesystem::path cache_path;
        std::string cache_key;
        if (surface_patch_cache) {
            cache_key = SurfacePatchIndex::cacheKeyForSurfaces(
                patch_surfaces, surface_patch_stride, surface_patch_bbox_pad);
            cache_path = surface_patch_cache_dir / ("surface_patch_index_" + cache_key + ".bin");
            const auto cache_load_start = std::chrono::steady_clock::now();
            cache_hit = patch_index.loadCache(cache_path, patch_surfaces, cache_key);
            const double cache_load_elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - cache_load_start).count();
            std::cout << "SurfacePatchIndex cache "
                      << (cache_hit ? "hit" : "miss")
                      << " key=" << cache_key
                      << " time=" << cache_load_elapsed << "s"
                      << " path=" << cache_path << std::endl;
        }
        if (!cache_hit) {
            const auto rebuild_start = std::chrono::steady_clock::now();
            patch_index.rebuild(patch_surfaces, surface_patch_bbox_pad);
            const double rebuild_elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - rebuild_start).count();
            std::cout << "SurfacePatchIndex rebuild time=" << rebuild_elapsed << "s" << std::endl;
            if (surface_patch_cache && !cache_path.empty()) {
                const auto cache_save_start = std::chrono::steady_clock::now();
                const bool saved = patch_index.saveCache(cache_path, cache_key);
                const double cache_save_elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - cache_save_start).count();
                std::cout << "SurfacePatchIndex cache save "
                          << (saved ? "ok" : "failed")
                          << " time=" << cache_save_elapsed << "s"
                          << " path=" << cache_path << std::endl;
            }
        }
        patch_index_ptr = &patch_index;
        std::cout << "SurfacePatchIndex built for " << patch_surfaces.size() << " surfaces"
                  << (need_resume_overlap_index ? " (resume overlap discovery)" : "")
                  << " patches=" << patch_index.patchCount()
                  << (cache_hit ? " (cache)" : "") << std::endl;
    }

    // In resume mode, seed may not have overlapping.json - compute overlaps dynamically
    if (resume_mode && seed->overlappingIds().empty()) {
        std::cout << "Computing overlaps for resumed seed surface using indexed parallel checks..." << std::endl;
        std::vector<QuadSurface*> overlap_hits;
        overlap_hits.reserve(surfs_v.size());
        std::atomic<int> checked{0};
        const int total = static_cast<int>(surfs_v.size());
        auto overlap_start_time = std::chrono::steady_clock::now();

#pragma omp parallel
        {
            std::vector<QuadSurface*> local_hits;
#pragma omp for schedule(dynamic)
            for (int idx = 0; idx < static_cast<int>(surfs_v.size()); ++idx) {
                QuadSurface* sm = surfs_v[idx];
                if (!sm || sm == seed)
                    continue;

                if (indexed_overlap_probe(*seed, *sm, patch_index_ptr, 150) ||
                    indexed_overlap_probe(*sm, *seed, patch_index_ptr, 150)) {
                    local_hits.push_back(sm);
                }

                const int done = checked.fetch_add(1, std::memory_order_relaxed) + 1;
                if (done == total || done % 25 == 0) {
#pragma omp critical
                    {
                        const double elapsed = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - overlap_start_time).count();
                        const double rate = elapsed > 0.0 ? static_cast<double>(done) / elapsed : 0.0;
                        printf("  resume overlap checks: %d/%d (%.1f/s)\n", done, total, rate);
                    }
                }
            }
#pragma omp critical
            overlap_hits.insert(overlap_hits.end(), local_hits.begin(), local_hits.end());
        }
        for (auto* sm : overlap_hits)
            overlapping_map[seed].insert(sm);
        std::cout << "Found " << overlapping_map[seed].size() << " overlapping surfaces for seed" << std::endl;
    } else {
        // Add seed's existing overlapping IDs if available
        for (const auto& name : seed->overlappingIds())
            if (surfs.contains(name))
                overlapping_map[seed].insert(surfs[name]);
    }

    std::cout << "total surface count (after defective filter): " << surfs.size() << std::endl;
    std::cout << "seed " << seed << " name " << seed->id << " seed overlapping: "
              << overlapping_map[seed].size() << "/" << seed->overlappingIds().size() << std::endl;

    cv::Mat_<cv::Vec3f> seed_points = seed->rawPoints();
    std::string seed_surface_id = seed->id;
    std::string seed_surface_name = seed->id;
    if (resume_mode) {
        if (seed->meta.contains("seed_surface_id") && seed->meta["seed_surface_id"].is_string())
            seed_surface_id = seed->meta["seed_surface_id"].get_string();
        if (seed->meta.contains("seed_surface_name") && seed->meta["seed_surface_name"].is_string())
            seed_surface_name = seed->meta["seed_surface_name"].get_string();
    }

    int stop_gen = 100000;
    int closing_r = 20; //FIXME dont forget to reset!

    // Get sliding window scale from params (set earlier from JSON)

    //1k ~ 1cm, scaled by sliding_w_scale parameter
    int sliding_w = static_cast<int>(1000/src_step/step*2 * sliding_w_scale);
    int w = 2000/src_step/step*2+10+2*closing_r;
    int h = 15000/src_step/step*2+10+2*closing_r;
    if (resume_mode) {
        const int resume_rows = seed_points.rows;
        const int resume_cols = seed_points.cols;
        if (resume_rows > 0 && resume_cols > 0) {
            w = std::max(1, (resume_cols + step_int - 1) / step_int);
            h = std::max(1, (resume_rows + step_int - 1) / step_int);
        }
    }
    cv::Size size = {w,h};
    cv::Rect bounds(0,0,w-1,h-1);
    cv::Rect active_bounds(closing_r+5,closing_r+5,
                           std::max(0, w-closing_r-10),
                           std::max(0, h-closing_r-10));
    cv::Rect static_bounds(0,0,0,h);

    int x0 = w/2;
    int y0 = h/2;

    std::cout << "starting with size " << size << " seed " << cv::Vec2i(y0,x0) << std::endl;

    std::vector<cv::Vec2i> neighs = {{1,0},{0,1},{-1,0},{0,-1}};
    if (neighbor_connectivity == 8) {
        neighs.insert(neighs.end(), {{1,1},{1,-1},{-1,1},{-1,-1}});
    }

    Fringe fringe;
    {
        Fringe::Config fringe_config;
        fringe_config.max_attempts = fringe_max_attempts;
        fringe_config.full_boundary = fringe_full_boundary;
        fringe_config.neighbor_connectivity = neighbor_connectivity;
        fringe.init(size, fringe_config);
    }

    cv::Mat_<uint8_t> state(size,0);
    cv::Mat_<uint16_t> generations(size, static_cast<uint16_t>(0));
    cv::Mat_<uint16_t> inliers_sum_dbg(size,0);
    cv::Mat_<cv::Vec3d> points(size,{-1,-1,-1});
    cv::Mat_<uint8_t> umbilicus_sampled(size, static_cast<uint8_t>(0));
    cv::Mat_<uint8_t> force_retry_count(size, static_cast<uint8_t>(0));
    cv::Mat_<uint8_t> force_retry_mark(size, static_cast<uint8_t>(0));
    std::vector<cv::Vec2i> force_retry;

    cv::Rect used_area(x0,y0,2,2);
    cv::Rect used_area_hr = {used_area.x*step_int, used_area.y*step_int, used_area.width*step_int, used_area.height*step_int};

    SurfTrackerData data;

    // Per-quad "already counted" mask for area accumulation (rows-1) x (cols-1)
    cv::Mat_<uint8_t> quad_done(cv::Size(std::max(0, w-1), std::max(0, h-1)), 0);
    // Live accumulated area in voxel^2 (progress logging). Updated O(1) per accepted vertex.
    double area_accum_vox2 = 0.0;
    auto init_area_scan = [&](void) {
        area_accum_vox2 = 0.0;
        quad_done = cv::Mat_<uint8_t>(cv::Size(std::max(0, points.cols-1), std::max(0, points.rows-1)), 0);
        for (int j = 0; j < state.rows - 1; ++j)
            for (int i = 0; i < state.cols - 1; ++i)
                area_accum_vox2 += maybe_quad_area_and_mark(j, i, state, points, quad_done);
    };

    cv::Rect savable_roi(0, 0, 0, 0);
    cv::Mat_<uint8_t> savable_mask;
    cv::Mat_<int> savable_dist_l1;
    auto build_savable_mask = [&]() {
        if (fringe_savable_dist <= 0.0f) {
            savable_roi = used_area & cv::Rect(0, 0, state.cols, state.rows);
            savable_mask.release();
            return;
        }
        const int pad = static_cast<int>(std::ceil(fringe_savable_dist)) + 2;
        cv::Rect roi = used_area;
        roi.x = std::max(0, roi.x - pad);
        roi.y = std::max(0, roi.y - pad);
        roi.width = std::min(state.cols - roi.x, roi.width + 2 * pad);
        roi.height = std::min(state.rows - roi.y, roi.height + 2 * pad);
        savable_roi = roi;
        if (roi.width <= 0 || roi.height <= 0) {
            savable_mask.release();
            return;
        }

        if (fringe_savable_metric == "l1") {
            const int max_dist = static_cast<int>(std::ceil(fringe_savable_dist));
            const int inf = max_dist + 1;
            savable_dist_l1.create(roi.height, roi.width);

            for (int y = 0; y < roi.height; ++y) {
                const uint8_t* state_row = state.ptr<uint8_t>(y + roi.y);
                int* dist_row = savable_dist_l1.ptr<int>(y);
                const int* prev_row = (y > 0) ? savable_dist_l1.ptr<int>(y - 1) : nullptr;
                for (int x = 0; x < roi.width; ++x) {
                    if (state_row[x + roi.x] & STATE_LOC_VALID) {
                        dist_row[x] = 0;
                        continue;
                    }
                    int best = inf;
                    if (prev_row)
                        best = std::min(best, prev_row[x] + 1);
                    if (x > 0)
                        best = std::min(best, dist_row[x - 1] + 1);
                    dist_row[x] = best;
                }
            }

            savable_mask = cv::Mat_<uint8_t>(roi.height, roi.width, static_cast<uint8_t>(0));
            for (int y = roi.height - 1; y >= 0; --y) {
                int* dist_row = savable_dist_l1.ptr<int>(y);
                const int* next_row = (y + 1 < roi.height) ? savable_dist_l1.ptr<int>(y + 1) : nullptr;
                for (int x = roi.width - 1; x >= 0; --x) {
                    int best = dist_row[x];
                    if (next_row)
                        best = std::min(best, next_row[x] + 1);
                    if (x + 1 < roi.width)
                        best = std::min(best, dist_row[x + 1] + 1);
                    dist_row[x] = best;
                    if (best <= max_dist)
                        savable_mask(y, x) = static_cast<uint8_t>(255);
                }
            }
            return;
        }

        cv::Mat invalid_mask(roi.height, roi.width, CV_8U, cv::Scalar(255));
        for (int y = 0; y < roi.height; ++y) {
            uint8_t* row = invalid_mask.ptr<uint8_t>(y);
            for (int x = 0; x < roi.width; ++x) {
                if (state(y + roi.y, x + roi.x) & STATE_LOC_VALID)
                    row[x] = 0;
            }
        }

        cv::Mat dist;
        cv::distanceTransform(invalid_mask, dist, cv::DIST_L2, 3);
        savable_mask = dist <= fringe_savable_dist;
    };

    auto is_savable = [&](const cv::Vec2i& p) -> bool {
        if (!point_in_bounds(state, p))
            return false;
        if (fringe_savable_dist <= 0.0f)
            return (state(p) & STATE_LOC_VALID) != 0;
        if (!savable_roi.contains(cv::Point(p[1], p[0])))
            return false;
        return savable_mask(p[0] - savable_roi.y, p[1] - savable_roi.x) != 0;
    };

    cv::Vec2i seed_loc = {seed_points.rows/2, seed_points.cols/2};
    bool resume_prefilled = false;
    int resume_valid_count = 0;

    if (resume_mode) {
        const int hr_rows = seed_points.rows;
        const int hr_cols = seed_points.cols;
        cv::Mat_<uint16_t> resume_generations = seed->channel("generations");
        bool has_generations = !resume_generations.empty();
        int min_x = w;
        int min_y = h;
        int max_x = -1;
        int max_y = -1;
        cv::Vec2i center = {h / 2, w / 2};
        double best_dist = std::numeric_limits<double>::max();

        for (int y = 0; y < h; ++y) {
            int hr_y = y * step_int;
            if (hr_y >= hr_rows)
                break;
            for (int x = 0; x < w; ++x) {
                int hr_x = x * step_int;
                if (hr_x >= hr_cols)
                    break;
                const cv::Vec3f& coord = seed_points(hr_y, hr_x);
                if (!seed_point_valid(coord))
                    continue;
                state(y, x) = STATE_LOC_VALID | STATE_COORD_VALID;
                points(y, x) = cv::Vec3d(coord[0], coord[1], coord[2]);
                data.surfs({y, x}).insert(seed);
                data.loc(seed, {y, x}) = {static_cast<double>(hr_y), static_cast<double>(hr_x)};
                if (has_generations) {
                    uint16_t gen = resume_generations(hr_y, hr_x);
                    generations(y, x) = (gen > 0) ? gen : static_cast<uint16_t>(1);
                } else {
                    generations(y, x) = 1;
                }
                resume_valid_count++;
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);
                double dy = static_cast<double>(y - center[0]);
                double dx = static_cast<double>(x - center[1]);
                double dist = dy * dy + dx * dx;
                if (dist < best_dist) {
                    best_dist = dist;
                    y0 = y;
                    x0 = x;
                }
            }
        }

        if (resume_valid_count > 0) {
            used_area = cv::Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
            used_area_hr = {used_area.x*step_int, used_area.y*step_int, used_area.width*step_int, used_area.height*step_int};
            cv::Vec2d seed_loc_d = data.loc(seed, {y0, x0});
            seed_loc = {static_cast<int>(std::lround(seed_loc_d[0])), static_cast<int>(std::lround(seed_loc_d[1]))};
            data.seed_coord = points(y0, x0);
            data.seed_loc = cv::Point2i(y0, x0);
            resume_prefilled = true;
            if (global_steps_per_window > 0) {
                const int overlap = 5;
                const int window_w = std::min(w, sliding_w + 2*closing_r + 10 + overlap);
                const int left_x = std::max(0, used_area.x);
                const int clamped_w = std::max(0, w - left_x);
                const int width = std::min(window_w, clamped_w);
                active_bounds = {left_x, closing_r + 5, width, std::max(0, h - closing_r - 10)};
            }
        } else {
            std::cout << "warning: resume surface contained no valid points; falling back to seed search" << std::endl;
        }
    }

    if (!resume_prefilled) {
        // Deterministic seed search around center using PRNG seeded by param
        {
            std::mt19937_64 rng(deterministic_seed);
            std::uniform_int_distribution<int> ry(0, seed_points.rows - 1);
            std::uniform_int_distribution<int> rx(0, seed_points.cols - 1);
            int tries = 0;
            while (seed_points(seed_loc)[0] == -1 ||
                   (enforce_z_range && (seed_points(seed_loc)[2] < z_min || seed_points(seed_loc)[2] > z_max))) {
                seed_loc = {ry(rng), rx(rng)};
                if (++tries > 10000) break;
            }
            if (tries > 0) {
                std::cout << "deterministic seed search tries: " << tries << " got " << seed_loc << std::endl;
            }
        }

        data.loc(seed,{y0,x0}) = {static_cast<double>(seed_loc[0]), static_cast<double>(seed_loc[1])};
        data.surfs({y0,x0}).insert(seed);
        points(y0,x0) = data.lookup_int(seed,{y0,x0});

        data.seed_coord = points(y0,x0);
        data.seed_loc = cv::Point2i(y0,x0);

        std::cout << "seed coord " << data.seed_coord << " at " << data.seed_loc << std::endl;
        if (enforce_z_range && (data.seed_coord[2] < z_min || data.seed_coord[2] > z_max))
            std::cout << "warning: seed z " << data.seed_coord[2] << " is outside z_range; growth will be restricted to [" << z_min << ", " << z_max << "]" << std::endl;

        state(y0,x0) = STATE_LOC_VALID | STATE_COORD_VALID;
        generations(y0,x0) = 1;
        fringe.insert(cv::Vec2i(y0,x0));

        // Initial area is zero (need 2x2 block to form first quad)
        init_area_scan(); // harmless here; leaves accumulator at 0
    } else {
        std::cout << "resume seed coord " << data.seed_coord << " at " << data.seed_loc << std::endl;
        if (enforce_z_range && (data.seed_coord[2] < z_min || data.seed_coord[2] > z_max))
            std::cout << "warning: seed z " << data.seed_coord[2] << " is outside z_range; growth will be restricted to [" << z_min << ", " << z_max << "]" << std::endl;
    }

    // Set up fringe context for rebuild operations
    fringe.setContext({&state, &used_area, &active_bounds, is_savable});

    if (resume_prefilled) {
        init_area_scan();
        build_savable_mask();
        fringe.rebuildBoundary();
    }

    auto seed_theta_reference = [&](const vc::core::util::Umbilicus& umbilicus) {
        constexpr int kSeedThetaWindow = 2;
        return compute_seed_theta_reference(umbilicus, data.seed_coord, points, state,
                                            {data.seed_loc[0], data.seed_loc[1]},
                                            kSeedThetaWindow);
    };

    // Initialize wrap tracking (if umbilicus provided or any wrap loss enabled)
    std::unique_ptr<vc::wrap_tracking::WrapTracker> wrap_tracker;
    std::unique_ptr<vc::core::util::Umbilicus> umbilicus_ptr;
    std::unique_ptr<vc::wrap_tracking::UmbilicusEstimator> umbilicus_estimator;
    bool wrap_tracking_enabled = (angle_column_w > 0 || angle_step_w > 0 ||
                                   radial_slope_w > 0 || tangent_ortho_w > 0);

    // Get volume_shape from params (required for umbilicus)
    cv::Vec3i volume_shape(0, 0, 0);
    if (params.contains("volume_shape") && params["volume_shape"].is_array() && params["volume_shape"].size() == 3) {
        volume_shape[0] = params["volume_shape"][0].get_int();  // Z
        volume_shape[1] = params["volume_shape"][1].get_int();  // Y
        volume_shape[2] = params["volume_shape"][2].get_int();  // X
    }

    auto apply_umbilicus_seam = [&](vc::core::util::Umbilicus& umbilicus) {
        umbilicus.set_seam(parse_umbilicus_seam(umbilicus_seam));
    };

    auto apply_wrap_growth_direction = [&](const vc::core::util::Umbilicus& umbilicus) {
        if (!wrap_direction_override || wrap_direction_applied)
            return;

        if (growth_mode == GrowthWindingMode::Bidirectional) {
            bidirectional_growth = true;
            mirror_pending = false;
        }

        int theta_sign = 0;
        if (!estimate_theta_sign_from_seed(seed_points, seed_loc, umbilicus, 10, &theta_sign)) {
            if (growth_mode == GrowthWindingMode::Clockwise ||
                growth_mode == GrowthWindingMode::CounterClockwise) {
                std::cout << "warning: wrap_growth_direction '" << wrap_growth_direction_raw
                          << "' could not infer theta direction from seed; using legacy flip_x"
                          << std::endl;
                mirror_pending = flip_x_param;
                wrap_flip_x = flip_x_param;
                wrap_flip_x_after_mirror = wrap_flip_x;
            } else {
                std::cout << "warning: wrap_growth_direction '" << wrap_growth_direction_raw
                          << "' could not infer theta direction from seed; using legacy flip_x for wrap tracking"
                          << std::endl;
            }
            wrap_direction_applied = true;
            return;
        }

        const bool right_is_ccw = (theta_sign > 0);
        wrap_flip_x = !right_is_ccw;
        if (growth_mode == GrowthWindingMode::Clockwise) {
            mirror_pending = right_is_ccw && !mirror_done;
        } else if (growth_mode == GrowthWindingMode::CounterClockwise) {
            mirror_pending = (!right_is_ccw) && !mirror_done;
        } else if (growth_mode == GrowthWindingMode::Bidirectional) {
            mirror_pending = false;
            bidirectional_growth = true;
        }

        if (mirror_done) {
            wrap_flip_x_after_mirror = !wrap_flip_x;
            wrap_flip_x = wrap_flip_x_after_mirror;
            mirror_pending = false;
        } else if (mirror_pending) {
            wrap_flip_x_after_mirror = !wrap_flip_x;
        } else {
            wrap_flip_x_after_mirror = wrap_flip_x;
        }

        wrap_direction_applied = true;
        std::cout << "[GrowSurface] wrap_growth_direction=" << wrap_growth_direction_raw
                  << " right_is_ccw=" << (right_is_ccw ? 1 : 0)
                  << " wrap_flip_x=" << wrap_flip_x
                  << " mirror_pending=" << (mirror_pending ? 1 : 0);
        if (bidirectional_growth)
            std::cout << " bidirectional=1";
        std::cout << std::endl;
    };

    if (params.contains("umbilicus_path")) {
        if (volume_shape[0] <= 0) {
            std::cerr << "[WrapTracking] Error: volume_shape required in params when using umbilicus_path" << std::endl;
        } else {
            try {
                std::string umbilicus_path = params["umbilicus_path"].get_string();
                umbilicus_ptr = std::make_unique<vc::core::util::Umbilicus>(
                    vc::core::util::Umbilicus::FromFile(umbilicus_path, volume_shape));
                apply_umbilicus_seam(*umbilicus_ptr);
                apply_wrap_growth_direction(*umbilicus_ptr);
                wrap_tracker = std::make_unique<vc::wrap_tracking::WrapTracker>(
                    *umbilicus_ptr, h, w, wrap_flip_x);
                double seed_theta_ref = seed_theta_reference(*umbilicus_ptr);
                wrap_tracker->initialize_from_seed(data.seed_coord, data.seed_loc[1], seed_theta_ref);
                std::cout << "[WrapTracking] Initialized from umbilicus file: " << umbilicus_path << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[WrapTracking] Failed to load umbilicus: " << e.what() << std::endl;
                wrap_tracker.reset();
                umbilicus_ptr.reset();
            }
        }
    } else if (wrap_tracking_enabled && volume_shape[0] > 0) {
        // No umbilicus file provided but wrap losses enabled - use dynamic estimation
        // Try to load existing estimated umbilicus if present
        std::filesystem::path paths_dir = tgt_dir.parent_path();
        std::filesystem::path json_path = paths_dir / "estimated_umbilicus.json";
        umbilicus_estimator = std::make_unique<vc::wrap_tracking::UmbilicusEstimator>(
            points.rows, json_path.string());

        // If points were loaded from JSON, build initial umbilicus
        if (umbilicus_estimator->has_any_points()) {
            try {
                auto built_umbilicus = umbilicus_estimator->build_umbilicus(volume_shape);
                apply_umbilicus_seam(built_umbilicus);
                umbilicus_ptr = std::make_unique<vc::core::util::Umbilicus>(std::move(built_umbilicus));
                apply_wrap_growth_direction(*umbilicus_ptr);
                wrap_tracker = std::make_unique<vc::wrap_tracking::WrapTracker>(
                    *umbilicus_ptr, h, w, wrap_flip_x);
                double seed_theta_ref = seed_theta_reference(*umbilicus_ptr);
                wrap_tracker->initialize_from_seed(data.seed_coord, data.seed_loc[1], seed_theta_ref);
                std::cout << "[WrapTracking] Initialized from loaded estimated_umbilicus.json ("
                          << umbilicus_estimator->center_count() << " centers)" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[WrapTracking] Failed to build umbilicus from loaded JSON: " << e.what() << std::endl;
                wrap_tracker.reset();
                umbilicus_ptr.reset();
            }
        } else {
            std::cout << "[WrapTracking] Will estimate umbilicus from surface normals (per-row)" << std::endl;
        }
    }

    if (wrap_direction_override && !wrap_direction_applied && !umbilicus_ptr) {
        if (umbilicus_estimator) {
            std::cout << "[GrowSurface] wrap_growth_direction '" << wrap_growth_direction_raw
                      << "' deferred until umbilicus is available" << std::endl;
        } else if (!bidirectional_growth) {
            mirror_pending = flip_x_param;
            wrap_flip_x = flip_x_param;
            wrap_flip_x_after_mirror = wrap_flip_x;
            wrap_direction_applied = true;
            std::cout << "warning: wrap_growth_direction '" << wrap_growth_direction_raw
                      << "' requested without umbilicus; using legacy flip_x" << std::endl;
        }
    }

    int seed_init_max_iters = point_to_seed_max_iters;
    if (resume_prefilled && seed_init_max_iters > 10) {
        seed_init_max_iters = 10;
        std::cout << "resume init: clamping point_to_seed_max_iters to "
                  << seed_init_max_iters << std::endl;
    }

    // Insert initial surfs per location (parallel)
    {
        std::shared_mutex data_mutex;
        std::vector<cv::Vec2i> fringe_vec(fringe.begin(), fringe.end());
        const int total = static_cast<int>(fringe_vec.size());
        std::atomic<int> done_count{0};
        auto start_time = std::chrono::steady_clock::now();

        std::cout << "Processing " << total << " fringe points against "
                  << overlapping_map[seed].size() << " overlapping surfaces..." << std::endl;

        #pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < total; ++idx) {
            const cv::Vec2i p = fringe_vec[idx];
            const cv::Vec3f coord = points(p);

            // Thread-local collection of results for this point
            std::vector<std::pair<QuadSurface*, cv::Vec2d>> found_surfs;
            found_surfs.reserve(overlapping_map[seed].size());

            for (auto s : overlapping_map[seed]) {
                auto _t0 = std::chrono::steady_clock::now();
                cv::Vec3f ptr;
                float _res = pointTo_seeded_neighbor(s, data, state, p, coord,
                                                     assoc_surface_th, seed_init_max_iters,
                                                     patch_index_ptr, &ptr);
                double _elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - _t0).count();
                #pragma omp atomic
                pointTo_total_ms += _elapsed;

                if (_res <= assoc_surface_th) {
                    cv::Vec3f loc = s->loc_raw(ptr);
                    found_surfs.emplace_back(s, cv::Vec2d(loc[1], loc[0]));
                }
            }

            // Write results under lock
            {
                std::unique_lock<std::shared_mutex> lock(data_mutex);
                data.surfs(p).insert(seed);
                for (const auto& [s, loc] : found_surfs) {
                    data.surfs(p).insert(s);
                    data.loc(s, p) = loc;
                }
            }

            // Progress reporting
            int completed = done_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (completed % 100 == 0 || completed == total) {
                #pragma omp critical
                {
                    auto now = std::chrono::steady_clock::now();
                    double elapsed_s = std::chrono::duration<double>(now - start_time).count();
                    double rate = completed / elapsed_s;
                    int remaining = total - completed;
                    double eta_s = (rate > 0) ? remaining / rate : 0;
                    std::cout << "\r[fringe init] " << completed << "/" << total
                              << " (" << std::fixed << std::setprecision(1)
                              << (100.0 * completed / total) << "%) ETA: "
                              << std::setprecision(1) << eta_s << "s" << std::flush;
                }
            }
        }
        std::cout << std::endl;

        // Print summary after parallel section
        for (const auto& p : fringe_vec) {
            std::cout << "fringe point " << p << " surfcount " << data.surfs(p).size()
                      << " init " << data.loc(seed, p) << data.lookup_int(seed, p) << std::endl;
        }
    }

    std::cout << "starting from " << x0 << " " << y0 << std::endl;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = global_step_max_iters;

    int final_opts = global_steps_per_window;

    int loc_valid_count = 0;
    int succ = 0;
    int curr_best_inl_th = inlier_base_threshold;
    int last_succ_parametrization = 0;
    int global_opt_count = 0;
    if (resume_prefilled) {
        succ = resume_valid_count;
        last_succ_parametrization = resume_valid_count;
    }

    std::vector<SurfTrackerData> data_ths(omp_get_max_threads());
    std::vector<std::vector<cv::Vec2i>> added_points_threads(omp_get_max_threads());
    for(int i=0;i<omp_get_max_threads();i++)
        data_ths[i] = data;

    // Track which rows need unwrapping after wrap tracking updates.
    std::vector<std::atomic<uint8_t>> wrap_rows_touched(h);
    for (auto& flag : wrap_rows_touched) {
        flag.store(0, std::memory_order_relaxed);
    }

    cv::Point bottleneck_seed_global(-1, -1);
    bool bottleneck_seed_locked = false;

    auto reset_after_prune = [&](bool rebuild_fringe) {
        for(int i=0;i<omp_get_max_threads();i++) {
            data_ths[i] = data;
            added_points_threads[i].clear();
        }
        for (const auto& p : force_retry)
            force_retry_mark(p) = 0;
        force_retry.clear();
        init_area_scan();
        build_savable_mask();
        if (rebuild_fringe) {
            fringe.resetAllAttempts();
            fringe.clearBorderFlags();
            if (fringe_full_boundary) {
                fringe.rebuildBoundary();
            } else {
                fringe.rebuildIncrementalRect(used_area, 2);
            }
        }
    };

    auto prune_misconnects = [&](int generation, bool use_wrap_batch_refresh) -> bool {
        if (misconnect_prune_interval <= 0)
            return false;
        if (generation <= 0 || (generation % misconnect_prune_interval) != 0)
            return false;

        std::cout << "[GrowSurface] misconnect prune tick(" << misconnect_prune_mode
                  << "): gen " << generation << std::endl;

        cv::Rect grid_bounds(0, 0, state.cols, state.rows);
        cv::Rect prune_area = used_area & grid_bounds;
        if (prune_area.width <= 0 || prune_area.height <= 0) {
            std::cout << "[GrowSurface] misconnect prune skip: empty prune_area" << std::endl;
            return false;
        }

        cv::Mat_<uint8_t> mask(prune_area.height, prune_area.width, static_cast<uint8_t>(0));
        int valid_cells = 0;
        for (int y = 0; y < prune_area.height; ++y) {
            const int gy = prune_area.y + y;
            uint8_t* row = mask.ptr<uint8_t>(y);
            for (int x = 0; x < prune_area.width; ++x) {
                const int gx = prune_area.x + x;
                if (state(gy, gx) & STATE_LOC_VALID) {
                    row[x] = static_cast<uint8_t>(255);
                    valid_cells++;
                }
            }
        }
        if (valid_cells == 0) {
            std::cout << "[GrowSurface] misconnect prune skip: no valid cells" << std::endl;
            return false;
        }

        cv::Mat labels_unused;
        int components_before = cv::connectedComponents(mask, labels_unused, neighbor_connectivity, CV_32S);
        std::cout << "[GrowSurface] misconnect prune: valid_cells=" << valid_cells
                  << " components_before=" << (components_before - 1) << std::endl;

        cv::Mat keep_mask;
        int components_after = 0;
        if (misconnect_prune_mode == "bottleneck") {
            cv::Mat dist;
            cv::distanceTransform(mask, dist, cv::DIST_L2, 3);

            cv::Mat_<float> best(prune_area.height, prune_area.width, 0.0f);
            struct BottleneckNode {
                float score;
                int idx;
            };
            struct BottleneckCompare {
                bool operator()(const BottleneckNode& a, const BottleneckNode& b) const {
                    return a.score < b.score;
                }
            };

            std::priority_queue<BottleneckNode, std::vector<BottleneckNode>, BottleneckCompare> queue;
            auto seed_neighborhood_valid = [&](int local_y, int local_x, int radius) -> bool {
                if (local_y - radius < 0 || local_y + radius >= prune_area.height ||
                    local_x - radius < 0 || local_x + radius >= prune_area.width) {
                    return false;
                }
                for (int dy = -radius; dy <= radius; ++dy) {
                    const uint8_t* mask_row = mask.ptr<uint8_t>(local_y + dy);
                    for (int dx = -radius; dx <= radius; ++dx) {
                        if (!mask_row[local_x + dx])
                            return false;
                    }
                }
                return true;
            };

            const int target_local_x = prune_area.width / 3;
            const int target_local_y = prune_area.height / 2;
            auto find_seed_near = [&](int radius, int* out_y, int* out_x) -> bool {
                if (prune_area.width <= radius * 2 || prune_area.height <= radius * 2)
                    return false;
                const int y_min_bound = radius;
                const int y_max_bound = prune_area.height - 1 - radius;
                const int x_min_bound = radius;
                const int x_max_bound = prune_area.width - 1 - radius;
                const int max_r = std::max(prune_area.width, prune_area.height);
                for (int r = 0; r <= max_r; ++r) {
                    const int y_min = std::max(y_min_bound, target_local_y - r);
                    const int y_max = std::min(y_max_bound, target_local_y + r);
                    const int x_min = std::max(x_min_bound, target_local_x - r);
                    const int x_max = std::min(x_max_bound, target_local_x + r);
                    if (y_min > y_max || x_min > x_max)
                        continue;
                    for (int y = y_min; y <= y_max; ++y) {
                        const bool edge_y = (y == y_min || y == y_max);
                        for (int x = x_min; x <= x_max; ++x) {
                            if (!edge_y && x != x_min && x != x_max)
                                continue;
                            if (seed_neighborhood_valid(y, x, radius)) {
                                *out_y = y;
                                *out_x = x;
                                return true;
                            }
                        }
                    }
                }
                return false;
            };

            int seed_local_y = -1;
            int seed_local_x = -1;
            bool ring2_ok = false;
            bool ring1_ok = false;
            bool using_locked = false;
            if (bottleneck_seed_locked) {
                seed_local_x = bottleneck_seed_global.x - prune_area.x;
                seed_local_y = bottleneck_seed_global.y - prune_area.y;
                if (seed_local_y < 0 || seed_local_y >= prune_area.height ||
                    seed_local_x < 0 || seed_local_x >= prune_area.width) {
                    std::cout << "[GrowSurface] misconnect prune skip: locked seed outside prune_area"
                              << " seed=(" << bottleneck_seed_global.x << "," << bottleneck_seed_global.y << ")"
                              << " prune_area=(" << prune_area.width << "x" << prune_area.height << ")"
                              << std::endl;
                    return false;
                }
                ring2_ok = seed_neighborhood_valid(seed_local_y, seed_local_x, 2);
                ring1_ok = ring2_ok ? true : seed_neighborhood_valid(seed_local_y, seed_local_x, 1);
                using_locked = true;
            } else {
                ring2_ok = find_seed_near(2, &seed_local_y, &seed_local_x);
                if (ring2_ok) {
                    bottleneck_seed_global = cv::Point(seed_local_x + prune_area.x,
                                                       seed_local_y + prune_area.y);
                    bottleneck_seed_locked = true;
                    ring1_ok = true;
                } else {
                    ring1_ok = find_seed_near(1, &seed_local_y, &seed_local_x);
                }
            }
            if (!ring1_ok) {
                std::cout << "[GrowSurface] misconnect prune skip: no seed near target"
                          << " target=(" << target_local_x << "," << target_local_y << ")"
                          << " prune_area=(" << prune_area.width << "x" << prune_area.height << ")"
                          << " locked=" << (using_locked ? 1 : 0)
                          << std::endl;
                return false;
            }
            const int seed_global_x = seed_local_x + prune_area.x;
            const int seed_global_y = seed_local_y + prune_area.y;
            const float score = dist.at<float>(seed_local_y, seed_local_x);
            best(seed_local_y, seed_local_x) = score;
            queue.push({score, seed_local_y * prune_area.width + seed_local_x});
            std::cout << "[GrowSurface] misconnect prune(bottleneck): sources=1"
                      << " ring2_ok=" << ring2_ok
                      << " locked=" << (bottleneck_seed_locked ? 1 : 0)
                      << " seed=(" << seed_global_x << "," << seed_global_y << ")"
                      << " target=(" << target_local_x << "," << target_local_y << ")"
                      << " radius=" << misconnect_bottleneck_radius << std::endl;

            static const std::array<cv::Vec2i, 4> kDirs4 = {
                cv::Vec2i(1, 0), cv::Vec2i(0, 1), cv::Vec2i(-1, 0), cv::Vec2i(0, -1)
            };
            static const std::array<cv::Vec2i, 8> kDirs8 = {
                cv::Vec2i(1, 0), cv::Vec2i(0, 1), cv::Vec2i(-1, 0), cv::Vec2i(0, -1),
                cv::Vec2i(1, 1), cv::Vec2i(-1, 1), cv::Vec2i(-1, -1), cv::Vec2i(1, -1)
            };
            const cv::Vec2i* dirs = nullptr;
            int dir_count = 0;
            if (neighbor_connectivity == 8) {
                dirs = kDirs8.data();
                dir_count = static_cast<int>(kDirs8.size());
            } else {
                dirs = kDirs4.data();
                dir_count = static_cast<int>(kDirs4.size());
            }
            while (!queue.empty()) {
                const BottleneckNode cur = queue.top();
                queue.pop();
                const int cy = cur.idx / prune_area.width;
                const int cx = cur.idx % prune_area.width;
                if (cur.score < best(cy, cx))
                    continue;
                for (int i = 0; i < dir_count; ++i) {
                    const cv::Vec2i& d = dirs[i];
                    const int ny = cy + d[0];
                    const int nx = cx + d[1];
                    if (ny < 0 || ny >= prune_area.height || nx < 0 || nx >= prune_area.width)
                        continue;
                    if (!mask(ny, nx))
                        continue;
                    const float cand = std::min(cur.score, dist.at<float>(ny, nx));
                    if (cand > best(ny, nx)) {
                        best(ny, nx) = cand;
                        queue.push({cand, ny * prune_area.width + nx});
                    }
                }
            }

            keep_mask = cv::Mat(prune_area.height, prune_area.width, CV_8U, cv::Scalar(0));
            int keep_count = 0;
            for (int y = 0; y < prune_area.height; ++y) {
                uint8_t* keep_row = keep_mask.ptr<uint8_t>(y);
                for (int x = 0; x < prune_area.width; ++x) {
                    if (best(y, x) > misconnect_bottleneck_radius) {
                        keep_row[x] = static_cast<uint8_t>(255);
                        keep_count++;
                    }
                }
            }
            std::cout << "[GrowSurface] misconnect prune(bottleneck): keep_count=" << keep_count << std::endl;
            if (keep_count == 0) {
                std::cout << "[GrowSurface] misconnect prune skip: keep_count=0" << std::endl;
                return false;
            }
            cv::bitwise_and(keep_mask, mask, keep_mask);

            cv::Mat labels_bottleneck;
            components_after = cv::connectedComponents(
                keep_mask, labels_bottleneck, neighbor_connectivity, CV_32S);
            std::cout << "[GrowSurface] misconnect prune(bottleneck): components_after="
                      << (components_after - 1) << std::endl;
        } else {
            // --- Erosion-based approach ---
            // Create structuring element (kernel for erosion/dilation)
            const int erode_iters = (misconnect_prune_kernel + 1) / 2;  // e.g., kernel=3 -> 2 iterations
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

            // Erode the mask to break thin bridges
            cv::Mat eroded;
            cv::erode(mask, eroded, kernel, cv::Point(-1,-1), erode_iters);

            // Compute connected components on eroded mask
            cv::Mat labels;
            cv::Mat stats;
            cv::Mat centroids;
            components_after = cv::connectedComponentsWithStats(
                eroded, labels, stats, centroids, neighbor_connectivity, CV_32S);

            // Early exit if erosion didn't create multiple components
            if (components_after <= 2) {
                std::cout << "[GrowSurface] misconnect prune skip: erosion no split" << std::endl;
                return false;
            }

            // Find largest component in eroded mask
            int largest_label = 0;
            int largest_area = 0;
            for (int label = 1; label < components_after; ++label) {
                const int area = stats.at<int>(label, cv::CC_STAT_AREA);
                if (area > largest_area) {
                    largest_area = area;
                    largest_label = label;
                }
            }
            if (largest_label == 0 || largest_area == 0)
                return false;

            // Create mask of largest eroded component
            cv::Mat largest_eroded;
            cv::compare(labels, largest_label, largest_eroded, cv::CMP_EQ);

            // Dilate back to recover original extent
            cv::Mat dilated_largest;
            cv::dilate(largest_eroded, dilated_largest, kernel, cv::Point(-1,-1), erode_iters);

            // Keep only original valid cells that overlap with dilated largest component
            cv::bitwise_and(dilated_largest, mask, keep_mask);
        }

        int removed = 0;
        int kept = 0;
        int min_x = prune_area.x + prune_area.width;
        int min_y = prune_area.y + prune_area.height;
        int max_x = -1;
        int max_y = -1;
        std::vector<uint8_t> rows_touched;
        if (wrap_tracker) {
            rows_touched.assign(h, static_cast<uint8_t>(0));
        }

        for (int y = 0; y < prune_area.height; ++y) {
            const int gy = prune_area.y + y;
            const uint8_t* keep_row = keep_mask.ptr<uint8_t>(y);
            for (int x = 0; x < prune_area.width; ++x) {
                const int gx = prune_area.x + x;
                if ((state(gy, gx) & STATE_LOC_VALID) == 0)
                    continue;
                if (keep_row[x] == 0) {
                    auto &surfs_here = data.surfs({gy, gx});
                    for (auto s : surfs_here)
                        data.erase(s, {gy, gx});
                    surfs_here.clear();
                    state(gy, gx) = 0;
                    points(gy, gx) = {-1,-1,-1};
                    generations(gy, gx) = 0;
                    inliers_sum_dbg(gy, gx) = 0;
                    force_retry_count(gy, gx) = 0;
                    force_retry_mark(gy, gx) = 0;
                    umbilicus_sampled(gy, gx) = 0;
                    removed++;
                    if (!rows_touched.empty())
                        rows_touched[gy] = 1;
                } else {
                    kept++;
                    min_x = std::min(min_x, gx);
                    min_y = std::min(min_y, gy);
                    max_x = std::max(max_x, gx);
                    max_y = std::max(max_y, gy);
                }
            }
        }

        if (removed == 0)
            return false;

        used_area = cv::Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
        used_area_hr = {used_area.x*step_int, used_area.y*step_int,
                        used_area.width*step_int, used_area.height*step_int};

        if (wrap_tracker && !rows_touched.empty()) {
            if (use_wrap_batch_refresh) {
                for (int row = 0; row < h; ++row) {
                    if (rows_touched[row])
                        wrap_rows_touched[row].store(1, std::memory_order_relaxed);
                }
            } else {
                for (int row = 0; row < h; ++row) {
                    if (!rows_touched[row])
                        continue;
                    wrap_tracker->unwrap_row(row, state);
                    wrap_tracker->correct_wrap_from_neighbors(row, state);
                }
            }
        }

        std::cout << "[GrowSurface] misconnect prune(" << misconnect_prune_mode << "): gen " << generation
                  << " comps " << (components_before - 1) << " -> " << (components_after - 1)
                  << ", removed " << removed << " cells, kept " << kept << std::endl;
        return true;
    };

    auto invalidate_fringe_cells = [&](bool use_wrap_batch_refresh, bool* cleared_all) -> bool {
        if (cleared_all)
            *cleared_all = false;
        if (fringe_invalidate_interval <= 0)
            return false;

        cv::Rect grid_bounds(0, 0, state.cols, state.rows);
        cv::Rect scan_area = used_area & grid_bounds;
        if (scan_area.width <= 0 || scan_area.height <= 0)
            return false;

        static const std::array<cv::Vec2i, 4> kDirs = {
            cv::Vec2i(1, 0), cv::Vec2i(0, 1), cv::Vec2i(-1, 0), cv::Vec2i(0, -1)
        };

        std::vector<cv::Vec2i> fringe_cells;
        fringe_cells.reserve(static_cast<size_t>(scan_area.width * scan_area.height / 8 + 1));

        for (int y = scan_area.y; y < scan_area.br().y; ++y) {
            for (int x = scan_area.x; x < scan_area.br().x; ++x) {
                if ((state(y, x) & STATE_LOC_VALID) == 0)
                    continue;
                bool has_invalid_neighbor = false;
                for (const auto& d : kDirs) {
                    cv::Vec2i pn = {y + d[0], x + d[1]};
                    if (!point_in_bounds(state, pn))
                        continue;
                    const uint8_t nstate = state(pn);
                    if ((nstate & STATE_LOC_VALID) == 0 || (nstate & STATE_PROCESSING) != 0) {
                        has_invalid_neighbor = true;
                        break;
                    }
                }
                if (has_invalid_neighbor)
                    fringe_cells.push_back({y, x});
            }
        }

        if (fringe_cells.empty())
            return false;

        std::vector<uint8_t> rows_touched;
        if (wrap_tracker) {
            rows_touched.assign(h, static_cast<uint8_t>(0));
        }

        for (const auto& p : fringe_cells) {
            if ((state(p) & STATE_LOC_VALID) == 0)
                continue;
            auto& surfs_here = data.surfs(p);
            for (auto s : surfs_here)
                data.erase(s, p);
            surfs_here.clear();
            state(p) = 0;
            points(p) = {-1,-1,-1};
            generations(p) = 0;
            inliers_sum_dbg(p) = 0;
            force_retry_count(p) = 0;
            force_retry_mark(p) = 0;
            umbilicus_sampled(p) = 0;
            if (!rows_touched.empty())
                rows_touched[p[0]] = 1;
        }

        int kept = 0;
        int min_x = scan_area.x + scan_area.width;
        int min_y = scan_area.y + scan_area.height;
        int max_x = -1;
        int max_y = -1;
        for (int y = scan_area.y; y < scan_area.br().y; ++y) {
            for (int x = scan_area.x; x < scan_area.br().x; ++x) {
                if ((state(y, x) & STATE_LOC_VALID) == 0)
                    continue;
                kept++;
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);
            }
        }

        if (kept == 0) {
            used_area = cv::Rect(0, 0, 0, 0);
            used_area_hr = {0, 0, 0, 0};
            if (cleared_all)
                *cleared_all = true;
        } else {
            used_area = cv::Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
            used_area_hr = {used_area.x*step_int, used_area.y*step_int,
                            used_area.width*step_int, used_area.height*step_int};
        }

        if (wrap_tracker && !rows_touched.empty()) {
            if (use_wrap_batch_refresh) {
                for (int row = 0; row < h; ++row) {
                    if (rows_touched[row])
                        wrap_rows_touched[row].store(1, std::memory_order_relaxed);
                }
            } else {
                for (int row = 0; row < h; ++row) {
                    if (!rows_touched[row])
                        continue;
                    wrap_tracker->unwrap_row(row, state);
                    wrap_tracker->correct_wrap_from_neighbors(row, state);
                }
            }
        }

        std::cout << "[GrowSurface] fringe invalidation: removed "
                  << fringe_cells.size() << " cells" << std::endl;
        return true;
    };

    bool skip_inpaint_next_opt = skip_inpaint;
    GrowthTimingTotals timing_totals;
    int timing_report_start_generation = 0;
    QuadSurface::resetPointToStats();
    reset_point_to_bbox_rejects();
    for(int generation=0;generation<stop_gen;generation++) {
        const auto generation_timing_start = std::chrono::steady_clock::now();
        const double point_to_generation_start_ms = pointTo_total_ms;
        const QuadSurface::PointToStats point_to_generation_start_stats = QuadSurface::pointToStatsSnapshot();
        const uint64_t point_to_bbox_generation_start = point_to_bbox_rejects_snapshot();
        bool generation_timing_finished = false;
        auto finish_generation_timing = [&](bool force_report) {
            if (generation_timing_finished)
                return;
            generation_timing_finished = true;
            timing_totals.generations++;
            timing_totals.generation_total_ms += elapsed_ms(generation_timing_start);
            timing_totals.point_to_ms += pointTo_total_ms - point_to_generation_start_ms;
            add_point_to_stats(timing_totals.point_to_stats,
                               point_to_stats_delta(QuadSurface::pointToStatsSnapshot(),
                                                    point_to_generation_start_stats));
            timing_totals.point_to_bbox_rejects +=
                point_to_bbox_rejects_snapshot() - point_to_bbox_generation_start;
            const bool interval_report = timing_report_interval > 0 &&
                                         generation > 0 &&
                                         (generation % timing_report_interval) == 0;
            if ((interval_report || force_report) && timing_totals.generations > 0) {
                timing_totals.print(timing_report_start_generation, generation);
                timing_totals.reset();
                timing_report_start_generation = generation + 1;
            }
        };
        bool wrap_stats_refreshed_this_generation = false;
        const bool use_wrap_batch_refresh = (wrap_tracker && wrap_batch_refresh);
        bool fringe_rebuilt = false;
        bool misconnect_pruned = false;
        {
            ScopedMs timer(timing_totals.prune_ms);
            misconnect_pruned = prune_misconnects(generation, use_wrap_batch_refresh);
        }
        if (misconnect_pruned) {
            reset_after_prune(true);
            skip_inpaint_next_opt = true;
            fringe_rebuilt = true;
        } else {
            ScopedMs timer(timing_totals.build_savable_ms);
            build_savable_mask();
        }
        bool cleared_all = false;
        if (fringe_invalidate_interval > 0 && generation > 0 &&
            (generation % fringe_invalidate_interval) == 0) {
            bool invalidated = false;
            {
                ScopedMs timer(timing_totals.invalidate_ms);
                invalidated = invalidate_fringe_cells(use_wrap_batch_refresh, &cleared_all);
            }
            if (invalidated && !cleared_all) {
                reset_after_prune(true);
                skip_inpaint_next_opt = true;
                fringe_rebuilt = true;
            }
            if (cleared_all) {
                fringe.clear();
                finish_generation_timing(true);
                break;
            }
        }
        if (!fringe_rebuilt && fringe_full_rebuild_interval > 0 && generation > 0 &&
            (generation % fringe_full_rebuild_interval) == 0) {
            ScopedMs timer(timing_totals.full_fringe_rebuild_ms);
            fringe.clearBorderFlags();
            build_savable_mask();
            fringe.rebuildBoundary();
        }
        std::set<cv::Vec2i, Vec2iLess> cands;
        std::set<cv::Vec2i, Vec2iLess> fringe_requeue;
        bool pruned_any = false;
        auto valid_neighbor_count = [&](const cv::Vec2i& p) {
            int count = 0;
            for (const auto& n : neighs) {
                cv::Vec2i pn = p + n;
                if (point_in_bounds(state, pn) && (state(pn) & STATE_LOC_VALID))
                    count++;
            }
            return count;
        };
        {
            ScopedMs timer(timing_totals.candidate_collect_ms);
            if (generation == 0 && !resume_prefilled) {
                cands.insert(cv::Vec2i(y0-1,x0));
            }
            else
                for(const auto& p : fringe)
                {
                    if ((state(p) & STATE_LOC_VALID) == 0)
                        continue;

                    fringe.incrementAttempts(p);
                    if (fringe.shouldPrune(p)) {
                        int neighbor_count = valid_neighbor_count(p);
                        if (neighbor_count >= 3) {
                            fringe.resetAttempts(p);
                        } else {
                            if (state(p) & STATE_LOC_VALID) {
                                auto &surfs_here = data.surfs(p);
                                for (auto s : surfs_here)
                                    data.erase(s, p);
                                surfs_here.clear();
                            }
                            state(p) = 0;
                            points(p) = {-1,-1,-1};
                            generations(p) = 0;
                            inliers_sum_dbg(p) = 0;
                            force_retry_count(p) = 0;
                            force_retry_mark(p) = 0;
                            umbilicus_sampled(p) = 0;
                            fringe.resetAttempts(p);
                            pruned_any = true;
                            for (const auto& n : neighs) {
                                cv::Vec2i pn = p + n;
                                if (point_in_bounds(state, pn) && (state(pn) & STATE_LOC_VALID))
                                    fringe_requeue.insert(pn);
                            }
                            continue;
                        }
                    }

                    for(const auto& n : neighs) {
                        cv::Vec2i pn = p+n;
                        if (!point_in_bounds(state, pn)) {
                            fringe.checkBorderContact(pn, size);
                            continue;
                        }
                        if (is_savable(pn) &&
                            (state(pn) & STATE_PROCESSING) == 0 &&
                            (state(pn) & STATE_LOC_VALID) == 0) {
                            state(pn) |= STATE_PROCESSING;
                            cands.insert(pn);
                        }
                    }
                }
            fringe.clear();
            if (!fringe_requeue.empty()) {
                for (const auto& p : fringe_requeue)
                    fringe.insert(p);
            }
            if (pruned_any) {
                reset_after_prune(false);
            }
        }

        if (!force_retry.empty()) {
            ScopedMs timer(timing_totals.force_retry_ms);
            for (const auto& p : force_retry) {
                force_retry_mark(p) = 0;
                if (state(p) & STATE_LOC_VALID)
                    continue;
                if (state(p) & STATE_PROCESSING)
                    continue;
                if (!is_savable(p))
                    continue;
                state(p) |= STATE_PROCESSING;
                cands.insert(p);
            }
            force_retry.clear();
        }

        if (generation % 100 == 0) {
            std::cout << "go with cands " << cands.size() << " inl_th " << curr_best_inl_th << std::endl;
        }

        // Deterministic, sorted vector of candidates
        std::vector<cv::Vec2i> cands_vec;
        {
            ScopedMs timer(timing_totals.candidate_sort_ms);
            cands_vec.assign(cands.begin(), cands.end());
        }

        std::shared_mutex mutex;
        int best_inliers_gen = 0;
        const int r = 1;

        auto queue_force_retry = [&](const cv::Vec2i& p, int neighbor_count) {
            if (force_retry_max == 0 || force_retry_min_neighbors == 0)
                return;
            if (neighbor_count < force_retry_min_neighbors)
                return;
            if (!is_savable(p))
                return;
            std::lock_guard<std::shared_mutex> lock(mutex);
            if (force_retry_mark(p))
                return;
            if (force_retry_count(p) >= force_retry_max)
                return;
            force_retry_mark(p) = 1;
            force_retry_count(p) = static_cast<uint8_t>(force_retry_count(p) + 1);
            force_retry.push_back(p);
        };

        auto process_candidate = [&](const cv::Vec2i& p) {

            if (state(p) & STATE_LOC_VALID)
                return;
            add_int_atomic(timing_totals.candidates_processed, 1);

            if (points(p)[0] != -1)
                throw std::runtime_error("oops points(p)[0]");

            SurfPtrSet local_surfs;
            local_surfs.insert(seed);
            std::unordered_map<QuadSurface*, int> local_counts;
            local_counts[seed] = 1;

            SurfTrackerData &data_th = data_ths[omp_get_thread_num()];
            {
                ScopedAtomicMs timer(timing_totals.candidate_sync_ms);
                mutex.lock_shared();
                int misses = 0;
                for(const auto& added : added_points_threads[omp_get_thread_num()]) {
                    data_th.surfs(added) = data.surfs(added);
                    for (auto &s : data.surfsC(added)) {
                        if (!data.has(s, added)) {
                            // Inconsistent: surface present without a stored mapping.
                            // Drop from thread-local set to avoid stale references.
                            data_th.surfs(added).erase(s);
                            misses++;
                            continue;
                        }
                        data_th.loc(s, added) = data.loc(s, added);
                    }
                }
                if (misses) {
                    std::cout << "grow: cleaned " << misses << " stale surface refs in thread-local cache" << std::endl;
                }
                mutex.unlock_shared();
                mutex.lock();
                added_points_threads[omp_get_thread_num()].resize(0);
                mutex.unlock();
            }

            const auto local_surface_start = std::chrono::steady_clock::now();
            // Build local_surfs from valid neighbors
            for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,h-1);oy++)
                for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,w-1);ox++)
                    if (state(oy,ox) & STATE_LOC_VALID) {
                        auto p_surfs = data_th.surfsC({oy,ox});
                        for (auto s : p_surfs) {
                            local_surfs.insert(s);
                            local_counts[s] += 1;
                        }
            }

            std::vector<QuadSurface*> local_surfs_vec(local_surfs.begin(), local_surfs.end());
            int effective_max_local_surfs = max_local_surfs;
            if (resume_prefilled && effective_max_local_surfs == 0)
                effective_max_local_surfs = resume_max_local_surfs;
            if (effective_max_local_surfs > 0 &&
                local_surfs_vec.size() > static_cast<size_t>(effective_max_local_surfs)) {
                // Prefer frequently-seen local surfaces when trimming.
                std::sort(local_surfs_vec.begin(), local_surfs_vec.end(),
                          [&](QuadSurface* a, QuadSurface* b) {
                              const auto a_it = local_counts.find(a);
                              const auto b_it = local_counts.find(b);
                              const int a_count = (a_it == local_counts.end()) ? 0 : a_it->second;
                              const int b_count = (b_it == local_counts.end()) ? 0 : b_it->second;
                              if (a_count != b_count)
                                  return a_count > b_count;
                              return a->id < b->id;
                          });
                local_surfs_vec.resize(effective_max_local_surfs);
            }

            SurfPtrSet local_surfs_filtered(local_surfs_vec.begin(), local_surfs_vec.end());
            std::vector<QuadSurface*> test_surfs;
            test_surfs.reserve(local_surfs_vec.size());
            for (auto* s : local_surfs_vec) {
                const auto count_it = local_counts.find(s);
                const int count = (count_it == local_counts.end()) ? 0 : count_it->second;
                if (s == seed || count >= min_test_surface_neighbor_count)
                    test_surfs.push_back(s);
            }

            std::vector<QuadSurface*> ref_surfs = local_surfs_vec;
            std::sort(ref_surfs.begin(), ref_surfs.end(),
                      [&](QuadSurface* a, QuadSurface* b) {
                          return a->id < b->id;
                      });
            add_ms_atomic(timing_totals.candidate_local_surface_ms, elapsed_ms(local_surface_start));

            cv::Vec3d best_coord = {-1,-1,-1};
            int best_inliers = -1;
            QuadSurface *best_surf = nullptr;
            cv::Vec2d best_loc = {-1,-1};
            bool best_ref_seed = false;
            for(auto ref_surf : ref_surfs) {
                int ref_count = 0;
                cv::Vec2d avg = {0,0};
                bool ref_seed = false;
                for(int oy=std::max(p[0]-r,0);oy<=std::min(p[0]+r,h-1);oy++)
                    for(int ox=std::max(p[1]-r,0);ox<=std::min(p[1]+r,w-1);ox++)
                        if ((state(oy,ox) & STATE_LOC_VALID) && data_th.valid_int(ref_surf,{oy,ox})) {
                            ref_count++;
                            avg += data_th.loc(ref_surf,{oy,ox});
                            if (data_th.seed_loc == cv::Vec2i(oy,ox))
                                ref_seed = true;
                        }

                if (ref_count < 2 && !ref_seed)
                    continue;

                avg /= ref_count;

                // Deterministic symmetric jitter (tie-breaker) in [-jitter_px, +jitter_px),
                // salted by a stable key (surface name), not pointer value.
                uint64_t surf_key = mix64(uint64_t(std::hash<std::string>{}(ref_surf->id)));
                uint64_t salt = deterministic_seed ^ surf_key;
                double j0 = det_jitter_symm(p[0], p[1], salt) * deterministic_jitter_px;
                double j1 = det_jitter_symm(p[0], p[1], salt ^ 0x9e3779b97f4a7c15ULL) * deterministic_jitter_px;
                data_th.loc(ref_surf,p) = avg + cv::Vec2d(j0, j1);

                bool fail = false;
                cv::Vec3d coord;
                cv::Vec2d ref_loc;
                {
                    ScopedAtomicMs timer(timing_totals.candidate_ref_init_solve_ms);
                    ceres::Problem problem;

                    state(p) = STATE_LOC_VALID | STATE_COORD_VALID;

                    int straight_count_init = 0;
                    int count_init = surftrack_add_local(ref_surf, p, data_th, problem, state, points, step, src_step, LOSS_ZLOC | LOSS_3D_INDIRECT, &straight_count_init);
                    if (count_init == 0) {
                        state(p) = 0;
                        points(p) = {-1,-1,-1};
                        generations(p) = 0;
                        data_th.erase(ref_surf, p);
                        continue;
                    }
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    float cost_init = sqrt(summary.final_cost/count_init);
                    (void)cost_init;

                    ref_loc = data_th.loc(ref_surf,p);

                    if (!data_th.valid_int(ref_surf,p))
                        fail = true;

                    if (!fail) {
                        coord = data_th.lookup_int(ref_surf,p);
                        if (coord[0] == -1)
                            fail = true;
                    }
                }


                if (fail) {
                    state(p) = 0;
                    points(p) = {-1,-1,-1};
                    generations(p) = 0;
                    data_th.erase(ref_surf, p);
                    continue;
                }

                state(p) = 0;
                points(p) = {-1,-1,-1};
                generations(p) = 0;

                int inliers_sum = 0;
                int inliers_count = 0;

                {
                    ScopedAtomicMs timer(timing_totals.candidate_inlier_eval_ms);
                    for(auto test_surf : test_surfs) {
                        //FIXME this does not check geometry, only if its also on the surfaces (which might be good enough...)
                        auto _t0 = std::chrono::steady_clock::now();
                        cv::Vec3f ptr;
                        float _res = pointTo_seeded_neighbor(test_surf, data_th, state, p, coord,
                                                             assoc_surface_th, point_to_max_iters, patch_index_ptr, &ptr);
                        double _elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t0).count();
                        #pragma omp atomic
                        pointTo_total_ms += _elapsed;
                        if (_res <= assoc_surface_th) {
                            cv::Vec3f loc = test_surf->loc_raw(ptr);
                            int count = 0;
                            int straight_count = 0;
                            state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                            data_th.loc(test_surf, p) = {loc[1], loc[0]};
                            float cost = local_cost(test_surf, p, data_th, state, points, step, src_step, &count, &straight_count);
                            state(p) = 0;
                            points(p) = {-1,-1,-1};
                            generations(p) = 0;
                            data_th.erase(test_surf, p);
                            if (cost < local_cost_inl_th && (ref_seed || (count >= 2 && straight_count >= straight_min_count))) {
                                inliers_sum += count;
                                inliers_count++;
                            }
                        }
                    }
                }
                if ((inliers_count >= 2 || ref_seed) && inliers_sum > best_inliers) {
                    if (enforce_z_range && (coord[2] < z_min || coord[2] > z_max)) {
                        // Do not consider candidates outside the allowed z-range
                        data_th.erase(ref_surf, p);
                        continue;
                    }
                    best_inliers = inliers_sum;
                    best_coord = coord;
                    best_surf = ref_surf;
                    best_loc = ref_loc;
                    best_ref_seed = ref_seed;
                }
                data_th.erase(ref_surf, p);
            }

            if (points(p)[0] != -1)
                throw std::runtime_error("oops points(p)[0]");

            // Guard against duplicating existing 3D coords.
            {
            ScopedAtomicMs timer(timing_totals.candidate_duplicate_check_ms);
            if (best_inliers >= curr_best_inl_th || best_ref_seed)
            {
                if (enforce_z_range && (best_coord[2] < z_min || best_coord[2] > z_max)) {
                    // Final guard: reject best candidate outside z-range
                    best_inliers = -1;
                    best_ref_seed = false;
                } else if (used_area.width >=4 && used_area.height >= 4) {
                    cv::Vec2f tmp_loc_;
                    cv::Rect used_th = used_area & cv::Rect(0, 0, points.cols, points.rows);
                    float dist = pointTo(tmp_loc_, points(used_th), best_coord, duplicate_surface_th, 1000, 1.0/(step*src_step));
                    tmp_loc_ += cv::Vec2f(used_th.x,used_th.y);
                    if (dist <= duplicate_surface_th) {
                        int state_sum = state(tmp_loc_[1],tmp_loc_[0]) + state(tmp_loc_[1]+1,tmp_loc_[0]) + state(tmp_loc_[1],tmp_loc_[0]+1) + state(tmp_loc_[1]+1,tmp_loc_[0]+1);
                        best_inliers = -1;
                        best_ref_seed = false;
                        if (!state_sum)
                            throw std::runtime_error("this should not have any location?!");
                    }
                }
            }
            }

            if (best_inliers >= curr_best_inl_th || best_ref_seed) {
                add_int_atomic(timing_totals.candidates_accepted, 1);
                if (best_coord[0] == -1)
                    throw std::runtime_error("oops best_cord[0]");

                {
                    ScopedAtomicMs timer(timing_totals.candidate_accept_setup_ms);
                    data_th.surfs(p).insert(best_surf);
                    data_th.loc(best_surf, p) = best_loc;
                    state(p) = STATE_LOC_VALID | STATE_COORD_VALID;
                    points(p) = best_coord;
                }

                if (wrap_tracker && !use_wrap_batch_refresh) {
                    ScopedAtomicMs timer(timing_totals.candidate_wrap_set_ms);
                    wrap_tracker->set_cell(p, best_coord);
                }

                // Collect samples for dynamic umbilicus estimation (per-row)
                if (umbilicus_estimator && !umbilicus_estimator->has_center(p[0])) {
                    ScopedAtomicMs timer(timing_totals.candidate_umbilicus_sample_ms);
                    if (!umbilicus_sampled(p)) {
                        cv::Vec3f normal = grid_normal(
                            points, cv::Vec3f(static_cast<float>(p[1]), static_cast<float>(p[0]), 0.0f));
                        if (std::isfinite(normal[0]) && std::isfinite(normal[1]) && std::isfinite(normal[2]) &&
                            (normal[0] != 0 || normal[1] != 0 || normal[2] != 0)) {
                            umbilicus_estimator->add_sample(p[0], best_coord, normal);
                            umbilicus_sampled(p) = 1;
                        }
                    }
                }

                force_retry_count(p) = 0;
                force_retry_mark(p) = 0;
                inliers_sum_dbg(p) = best_inliers;
                generations(p) = static_cast<uint16_t>(std::min<int>(std::numeric_limits<uint16_t>::max(), generation + 1));

                ceres::Problem problem;
                {
                    ScopedAtomicMs timer(timing_totals.candidate_problem_build_ms);
                    surftrack_add_local(best_surf, p, data_th, problem, state, points, step, src_step, SURF_LOSS | LOSS_ZLOC | LOSS_3D_INDIRECT);

                    // Add wrap-aware losses if enabled
                    if (wrap_tracker) {
                        add_wrap_losses(p, points, state, problem, wrap_tracker.get());
                    }
                }

                SurfPtrSet more_local_surfs;

                {
                    ScopedAtomicMs timer(timing_totals.candidate_overlap_attach_ms);
                    for(auto test_surf : test_surfs) {
                        for(auto s : overlapping_map[test_surf])
                            if (!local_surfs_filtered.contains(s) && s != best_surf)
                                more_local_surfs.insert(s);

                        if (test_surf == best_surf)
                            continue;

                        auto _t0 = std::chrono::steady_clock::now();
                        cv::Vec3f ptr;
                        float _res = pointTo_seeded_neighbor(test_surf, data_th, state, p, best_coord,
                                                             assoc_surface_th, point_to_max_iters, patch_index_ptr, &ptr);
                        double _elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t0).count();
                        #pragma omp atomic
                        pointTo_total_ms += _elapsed;
                        if (_res <= assoc_surface_th) {
                            cv::Vec3f loc = test_surf->loc_raw(ptr);
                            cv::Vec3f coord = SurfTrackerData::lookup_int_loc(test_surf, {loc[1], loc[0]});
                            if (coord[0] == -1) {
                                continue;
                            }
                            int count = 0;
                            float cost = local_cost_destructive(test_surf, p, data_th, state, points, step, src_step, loc, &count);
                            if (cost < local_cost_inl_th) {
                                data_th.loc(test_surf, p) = {loc[1], loc[0]};
                                data_th.surfs(p).insert(test_surf);
                                surftrack_add_local(test_surf, p, data_th, problem, state, points, step, src_step, SURF_LOSS | LOSS_ZLOC | LOSS_3D_INDIRECT);
                            }
                            else
                                data_th.erase(test_surf, p);
                        }
                    }
                }

                ceres::Solver::Summary summary;

                {
                    ScopedAtomicMs timer(timing_totals.candidate_local_solve_ms);
                    ceres::Solve(options, &problem, &summary);
                }

                if (wrap_tracker && use_wrap_batch_refresh) {
                    ScopedAtomicMs timer(timing_totals.candidate_wrap_set_ms);
                    wrap_tracker->set_cell(p, points(p));
                    wrap_rows_touched[p[0]].store(1, std::memory_order_relaxed);
                }

                //TODO only add/test if we have 2 neighs which both find locations
                {
                    ScopedAtomicMs timer(timing_totals.candidate_post_attach_ms);
                    for(auto test_surf : test_surfs) {
                        auto _t0 = std::chrono::steady_clock::now();
                        cv::Vec3f ptr;
                        float res = pointTo_seeded_neighbor(test_surf, data_th, state, p, best_coord,
                                                            assoc_surface_th, point_to_max_iters, patch_index_ptr, &ptr);
                        double _elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _t0).count();
                        #pragma omp atomic
                        pointTo_total_ms += _elapsed;
                        if (res <= assoc_surface_th) {
                            cv::Vec3f loc = test_surf->loc_raw(ptr);
                            cv::Vec3f coord = SurfTrackerData::lookup_int_loc(test_surf, {loc[1], loc[0]});
                            if (coord[0] == -1) {
                                continue;
                            }
                            int count = 0;
                            float cost = local_cost_destructive(test_surf, p, data_th, state, points, step, src_step, loc, &count);
                            if (cost < local_cost_inl_th) {
                                data_th.loc(test_surf, p) = {loc[1], loc[0]};
                                data_th.surfs(p).insert(test_surf);
                            };
                        };
                    }
                }

                {
                ScopedAtomicMs timer(timing_totals.candidate_commit_ms);
                mutex.lock();
                succ++;

                // Rebuild global set only from surfaces that have a valid loc in thread-local
                SurfPtrSet accepted;
                for (auto &s : data_th.surfs(p))
                    if (data_th.has(s, p))
                        accepted.insert(s);

                data.surfs(p).clear();
                for (auto &s : accepted) {
                    data.surfs(p).insert(s);
                    data.loc(s, p) = data_th.loc(s, p);
                }
                for(int t=0;t<omp_get_max_threads();t++)
                    added_points_threads[t].push_back(p);

                if (!used_area.contains(cv::Point(p[1],p[0]))) {
                    used_area = used_area | cv::Rect(p[1],p[0],1,1);
                    used_area_hr = {used_area.x*step_int, used_area.y*step_int, used_area.width*step_int, used_area.height*step_int};
                }
                fringe.insert(p);
                fringe.resetAttempts(p);
                for (const auto& n : neighs) {
                    cv::Vec2i pn = p + n;
                    if (point_in_bounds(state, pn) && (state(pn) & STATE_LOC_VALID))
                        fringe.resetAttempts(pn);
                }
                // O(1) geometric area update: up to four quads become complete with a new vertex.
                area_accum_vox2 += maybe_quad_area_and_mark(p[0]-1, p[1]-1, state, points, quad_done);
                area_accum_vox2 += maybe_quad_area_and_mark(p[0]-1, p[1]  , state, points, quad_done);
                area_accum_vox2 += maybe_quad_area_and_mark(p[0]  , p[1]-1, state, points, quad_done);
                area_accum_vox2 += maybe_quad_area_and_mark(p[0]  , p[1]  , state, points, quad_done);
                mutex.unlock();
                }
            }
            else if (best_inliers == -1) {
                //just try again some other time
                state(p) = 0;
                points(p) = {-1,-1,-1};
                generations(p) = 0;
                queue_force_retry(p, valid_neighbor_count(p));
            }
            else {
                state(p) = 0;
                points(p) = {-1,-1,-1};
                generations(p) = 0;
                queue_force_retry(p, valid_neighbor_count(p));
#pragma omp critical
                best_inliers_gen = std::max(best_inliers_gen, best_inliers);
            }
        };

        auto process_cands_row_col = [&]() {
            // Column-wise processing: grow left-to-right by column (x).
            std::map<int, std::vector<cv::Vec2i>> by_col;
            for (const auto& p : cands_vec)
                by_col[p[1]].push_back(p);

            if (bidirectional_growth) {
                std::vector<int> col_order;
                col_order.reserve(by_col.size());
                for (const auto& kv : by_col)
                    col_order.push_back(kv.first);
                std::sort(col_order.begin(), col_order.end(),
                          [&](int a, int b) {
                              const int da = std::abs(a - x0);
                              const int db = std::abs(b - x0);
                              if (da != db)
                                  return da < db;
                              return a < b;
                          });
                for (int col : col_order) {
                    auto& col_points = by_col[col];
#pragma omp parallel for schedule(static)
                    for (int idx = 0; idx < static_cast<int>(col_points.size()); ++idx)
                        process_candidate(col_points[idx]);
                }
                return;
            }

            for (auto &kv : by_col) {
#pragma omp parallel for schedule(static)
                for (int idx = 0; idx < static_cast<int>(kv.second.size()); ++idx)
                    process_candidate(kv.second[idx]);
            }
        };

        auto process_cands_spread = [&]() {
            OmpThreadPointCol cands_threadcol(candidate_min_dist, cands_vec);
#pragma omp parallel
            {
                while (true) {
                    cv::Vec2i p = cands_threadcol.next();
                    if (p[0] == -1)
                        break;
                    process_candidate(p);
                }
            }
        };

        {
            ScopedMs timer(timing_totals.candidate_process_wall_ms);
            if (use_spread_out_ordering)
                process_cands_spread();
            else
                process_cands_row_col();
        }

        if (mirror_pending && !mirror_done && generation >= mirror_ready_generation) {
            ScopedMs timer(timing_totals.mirror_ms);
            data.flip_x(x0);

            for(int i=0;i<omp_get_max_threads();i++) {
                data_ths[i] = data;
                added_points_threads[i].clear();
            }

            cv::Mat_<uint8_t> state_orig = state.clone();
            cv::Mat_<cv::Vec3d> points_orig = points.clone();
            cv::Mat_<uint16_t> generations_orig = generations.clone();
            state.setTo(0);
            points.setTo(cv::Vec3d(-1,-1,-1));
            generations.setTo(0);
            force_retry_count.setTo(0);
            force_retry_mark.setTo(0);
            umbilicus_sampled.setTo(0);
            fringe.resetAllAttempts();
            force_retry.clear();
            cv::Rect new_used_area = used_area;
            int mirror_clipped_count = 0;
            // Clamp loop bounds to original matrix dimensions
            const int j_max = std::min(used_area.br().y + 1, state_orig.rows - 1);
            const int i_max = std::min(used_area.br().x + 1, state_orig.cols - 1);
            for(int j=used_area.y; j<=j_max; j++)
                for(int i=used_area.x; i<=i_max; i++)
                    if (state_orig(j, i)) {
                        int nx = x0 + x0 - i;
                        int ny = j;
                        // Skip points that would be mirrored out of bounds
                        if (nx < 0 || nx >= state.cols || ny < 0 || ny >= state.rows) {
                            mirror_clipped_count++;
                            continue;
                        }
                        state(ny, nx) = state_orig(j, i);
                        points(ny, nx) = points_orig(j, i);
                        generations(ny, nx) = generations_orig(j, i);
                        new_used_area = new_used_area | cv::Rect(nx, ny, 1, 1);
                    }

            if (mirror_clipped_count > 0) {
                std::cout << "[GrowSurface] mirror: " << mirror_clipped_count
                          << " points clipped due to out-of-bounds coordinates" << std::endl;
            }

            used_area = new_used_area;
            // Clamp used_area to valid grid bounds
            {
                cv::Rect grid_bounds(0, 0, state.cols, state.rows);
                used_area = used_area & grid_bounds;
            }
            used_area_hr = {used_area.x * step_int, used_area.y * step_int,
                            used_area.width * step_int, used_area.height * step_int};

            if (fringe_full_boundary) {
                build_savable_mask();
                fringe.rebuildBoundary();
            } else {
                fringe.rebuildIncrementalRect(used_area, 2);
            }
            // Geometry changed globally; rebuild area mask & accumulator
            init_area_scan();
            if (wrap_tracker && wrap_flip_x_after_mirror != wrap_flip_x) {
                wrap_flip_x = wrap_flip_x_after_mirror;
                const auto& umbilicus = wrap_tracker->umbilicus();
                auto new_tracker = std::make_unique<vc::wrap_tracking::WrapTracker>(
                    umbilicus, points.rows, points.cols, wrap_flip_x);
                double seed_theta_ref = seed_theta_reference(umbilicus);
                new_tracker->initialize_from_seed(data.seed_coord, data.seed_loc[1], seed_theta_ref);
                int repopulated = 0;
                for (int r = 0; r < points.rows; ++r) {
                    for (int c = 0; c < points.cols; ++c) {
                        if (state(r, c) & STATE_COORD_VALID) {
                            new_tracker->set_cell({r, c}, points(r, c));
                            repopulated++;
                        }
                    }
                }
                for (int row = 0; row < points.rows; ++row) {
                    new_tracker->unwrap_row(row, state);
                }
                new_tracker->compute_statistics(state);
                wrap_stats_refreshed_this_generation = true;
                std::cout << "[WrapTracker] Rebuilt after mirror (repopulated "
                          << repopulated << " cells)" << std::endl;
                wrap_tracker = std::move(new_tracker);
            }
            mirror_done = true;
            mirror_pending = false;
            if (use_wrap_batch_refresh) {
                for (int row = 0; row < h; ++row) {
                    wrap_rows_touched[row].store(1, std::memory_order_relaxed);
                }
            }
        }

        if (use_wrap_batch_refresh) {
            ScopedMs timer(timing_totals.wrap_batch_refresh_ms);
            for (int row = 0; row < h; ++row) {
                if (!wrap_rows_touched[row].load(std::memory_order_relaxed)) {
                    continue;
                }
                wrap_tracker->unwrap_row(row, state);
                // Correct wrap errors using neighbors (which may be from previous generations)
                wrap_tracker->correct_wrap_from_neighbors(row, state);
                wrap_rows_touched[row].store(0, std::memory_order_relaxed);
            }
        }

        {
        ScopedMs timer(timing_totals.threshold_ms);
        int inl_lower_bound_reg = params.value("consensus_default_th", 10);
        int inl_lower_bound_b = params.value("consensus_limit_th", 2);
        int inl_lower_bound = inl_lower_bound_reg;

        if (!fringe.atAnyBorder() &&
            curr_best_inl_th <= inl_lower_bound)
            inl_lower_bound = inl_lower_bound_b;

        if (fringe.empty() && curr_best_inl_th > inl_lower_bound) {
            curr_best_inl_th = std::max(inl_lower_bound, curr_best_inl_th - 1);
            fringe.rebuildIncremental(2);
        } else
            curr_best_inl_th = inlier_base_threshold;
        }

        {
        ScopedMs timer(timing_totals.loc_count_ms);
        loc_valid_count = 0;
        for(int j=used_area.y;j<used_area.br().y-1;j++)
            for(int i=used_area.x;i<used_area.br().x-1;i++)
                if (state(j,i) & STATE_LOC_VALID)
                    loc_valid_count++;
        }

        bool update_mapping = (succ >= 1000 && (loc_valid_count-last_succ_parametrization) >= std::max(100.0, 0.3*last_succ_parametrization));
        if (fringe.empty() && final_opts) {
            final_opts--;
            update_mapping = true;
        }

        if (!global_steps_per_window)
            update_mapping = false;

        bool save_generation_debug = (generation % 250 == 0 || update_mapping /*|| generation < 10*/);
        if (resume_prefilled && generation == 0 && !update_mapping &&
            !params.value("save_resume_gen0_debug", false)) {
            save_generation_debug = false;
            std::cout << "resume: skipping generation 0 HR debug surface save" << std::endl;
        }

        if (save_generation_debug) {
            ScopedMs timer(timing_totals.debug_save_ms);
            {
                cv::Mat_<cv::Vec3f> points_hr =
                    surftrack_genpoints_hr(data, state, points, used_area, step, src_step,
                                           /*inpaint=*/false,
                                           /*parallel=*/params.value("hr_gen_parallel", false));
                auto dbg_surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});
                dbg_surf->meta = utils::Json::object();
                dbg_surf->meta["vc_grow_seg_from_segments_params"] = utils::Json::parse(params.dump());

                auto gen_channel = surftrack_generation_channel(generations, used_area, step);
                if (!gen_channel.empty())
                    dbg_surf->setChannel("generations", gen_channel);

                dbg_surf->meta["max_gen"] = surftrack_max_generation(generations, used_area);

                // Use exact geometric area accumulated so far
                const double area_exact_vx2 = area_accum_vox2;
                const double area_exact_cm2 = area_exact_vx2 * double(voxelsize) * double(voxelsize) / 1e8;
                dbg_surf->meta["area_vx2"] = area_exact_vx2;
                dbg_surf->meta["area_cm2"] = area_exact_cm2;
                dbg_surf->meta["seed_surface_name"] = seed_surface_name;
                dbg_surf->meta["seed_surface_id"] = seed_surface_id;
                // Delete previous HR surface we wrote this session before saving new one
                if (!last_hr_surface_uuid.empty()) {
                    std::error_code ec;
                    std::filesystem::remove_all(tgt_dir / last_hr_surface_uuid, ec);
                }
                std::string uuid = Z_DBG_GEN_PREFIX+get_surface_time_str();
                dbg_surf->save(tgt_dir / uuid, uuid);
                last_hr_surface_uuid = uuid;
                delete dbg_surf;
            }
        }

        //lets just see what happens
        if (update_mapping) {
            ScopedMs global_timer(timing_totals.global_total_ms);
            timing_totals.global_opt_count++;
            global_opt_count++;
            bool save_inp_hr = (global_opt_count % 5 == 0);
            dbg_counter = generation;
            SurfTrackerData opt_data;
            cv::Mat_<uint8_t> opt_state;
            cv::Mat_<cv::Vec3d> opt_points;
            cv::Mat_<uint16_t> opt_generations;
            cv::Rect active;
            {
                ScopedMs timer(timing_totals.global_setup_ms);
                opt_data = data;
                opt_state = state.clone();
                opt_points = points.clone();
                opt_generations = generations.clone();
                active = active_bounds & used_area;
            }

            OptimizerTiming optimizer_timing;
            {
                ScopedMs timer(timing_totals.global_optimize_ms);
                optimize_surface_mapping(opt_data, opt_state, opt_points, opt_generations, active, static_bounds, step, src_step,
                                         {y0,x0}, closing_r, overlapping_map, patch_index_ptr,
                                         window_inpaint_max_iters, window_opt_max_iters,
                                         save_inp_hr, tgt_dir, &last_inp_hr_surface_uuid,
                                         /*hr_gen_parallel=*/params.value("hr_gen_parallel", false),
                                         /*remap_parallel=*/params.value("remap_parallel", false),
                                         /*use_cuda_sparse=*/params.value("use_cuda_sparse", true),
                                         wrap_tracker.get(),
                                         remap_use_inpaint,
                                         debug_diagnostics,
                                         skip_inpaint_next_opt,
                                         &optimizer_timing);
            }
            timing_totals.optimizer.add(optimizer_timing);
            optimizer_timing.print(std::cout, "[Timing] optimizer_call");
            skip_inpaint_next_opt = skip_inpaint;
            if (active.area() > 0) {
                {
                    ScopedMs timer(timing_totals.global_copy_ms);
                    copy(opt_data, data, active);
                    opt_points(active).copyTo(points(active));
                    opt_state(active).copyTo(state(active));
                    opt_generations(active).copyTo(generations(active));
                    for (int j = active.y; j < active.br().y; ++j)
                        for (int i = active.x; i < active.br().x; ++i) {
                            force_retry_count(j, i) = 0;
                            force_retry_mark(j, i) = 0;
                        }
                }

                if (wrap_tracker) {
                    ScopedMs timer(timing_totals.global_wrap_rebuild_ms);
                    const auto& umbilicus = wrap_tracker->umbilicus();
                    const bool flip_x_local = wrap_tracker->flip_x();
                    auto new_tracker = std::make_unique<vc::wrap_tracking::WrapTracker>(
                        umbilicus, points.rows, points.cols, flip_x_local);
                    double seed_theta_ref = seed_theta_reference(umbilicus);
                    new_tracker->initialize_from_seed(data.seed_coord, data.seed_loc[1], seed_theta_ref);
                    int repopulated = 0;
                    for (int r = 0; r < points.rows; ++r) {
                        for (int c = 0; c < points.cols; ++c) {
                            if (state(r, c) & STATE_COORD_VALID) {
                                new_tracker->set_cell({r, c}, points(r, c));
                                repopulated++;
                            }
                        }
                    }
                    for (int row = 0; row < points.rows; ++row) {
                        new_tracker->unwrap_row(row, state);
                    }
                    // Correct wrap errors using row-to-row consistency
                    for (int row = 0; row < points.rows; ++row) {
                        new_tracker->correct_wrap_from_neighbors(row, state);
                    }
                    new_tracker->compute_statistics(state);
                    wrap_stats_refreshed_this_generation = true;
                    std::cout << "[WrapTracker] Rebuilt after mapping update (repopulated "
                              << repopulated << " cells)" << std::endl;
                    wrap_tracker = std::move(new_tracker);
                }

                {
                    ScopedMs timer(timing_totals.global_thread_reset_ms);
                    for(int i=0;i<omp_get_max_threads();i++) {
                        data_ths[i] = data;
                        added_points_threads[i].resize(0);
                    }
                    force_retry.clear();
                }
                // After remap, geometry/states may have changed -> recompute area mask/accumulator
                {
                    ScopedMs timer(timing_totals.global_area_scan_ms);
                    init_area_scan();
                }
            }

            last_succ_parametrization = loc_valid_count;
            //recalc fringe after surface optimization (which often shrinks the surf)
            curr_best_inl_th = inlier_base_threshold;
            {
                ScopedMs timer(timing_totals.global_fringe_rebuild_ms);
                if (fringe_full_boundary) {
                    build_savable_mask();
                    fringe.rebuildBoundary();
                } else {
                    fringe.rebuildIncremental(2);
                }
            }
        }

        const double current_area_vx2 = area_accum_vox2;
        const double current_area_cm2 = current_area_vx2 * double(voxelsize) * double(voxelsize) / 1e8;
        if (generation % 100 == 0) {
            ScopedMs timer(timing_totals.progress_log_ms);
            printf("gen %d processing %lu fringe cands (total done %d fringe: %lu) area %.0f vx^2 (%f cm^2) best th: %d\n",
                   generation, static_cast<unsigned long>(cands.size()), succ, static_cast<unsigned long>(fringe.size()),
                   current_area_vx2, current_area_cm2, best_inliers_gen);
        }

        // Update wrap tracking statistics periodically
        if (wrap_tracker && generation > 0 && generation % wrap_stats_update_interval == 0) {
            ScopedMs wrap_total_timer(timing_totals.wrap_stats_total_ms);
            timing_totals.wrap_stats_count++;
            if (!wrap_stats_refreshed_this_generation) {
                std::cout << "[WrapTracker] Updating stats at gen " << generation << std::endl;
                // Unwrap all rows with valid cells
                {
                    ScopedMs timer(timing_totals.wrap_stats_unwrap_ms);
                    for (int row = 0; row < h; ++row) {
                        wrap_tracker->unwrap_row(row, state);
                    }
                }
                // Correct wrap errors using row-to-row consistency
                {
                    ScopedMs timer(timing_totals.wrap_stats_correct_ms);
                    for (int row = 0; row < h; ++row) {
                        wrap_tracker->correct_wrap_from_neighbors(row, state);
                    }
                }
                {
                    ScopedMs timer(timing_totals.wrap_stats_compute_ms);
                    wrap_tracker->compute_statistics(state);
                }
                wrap_stats_refreshed_this_generation = true;
            }
            if (wrap_debug_tif_interval > 0 && generation % wrap_debug_tif_interval == 0) {
                ScopedMs timer(timing_totals.wrap_debug_tif_ms);
                try {
                    write_wrap_debug_tifs(state, used_area, generation, tgt_dir, wrap_tracker.get());
                } catch (const std::exception& ex) {
                    std::cerr << "[WrapTracker] Debug TIFF write failed: " << ex.what() << std::endl;
                }
            }
        }

        // Dynamic umbilicus estimation: update wrap_tracker as new rows wrap around
        if (umbilicus_estimator && generation > 0 && generation % wrap_stats_update_interval == 0) {
            ScopedMs umbilicus_total_timer(timing_totals.umbilicus_total_ms);
            timing_totals.umbilicus_count++;
            int sampled_attempts = 0;
            int sampled_added = 0;
            int sampled_invalid = 0;
            cv::Rect sample_area = used_area & cv::Rect(0, 0, points.cols, points.rows);
            {
                ScopedMs timer(timing_totals.umbilicus_sample_scan_ms);
                for (int row = sample_area.y; row < sample_area.br().y; ++row) {
                    if (umbilicus_estimator->has_center(row)) {
                        continue;
                    }
                    for (int col = sample_area.x; col < sample_area.br().x; ++col) {
                        if (!(state(row, col) & STATE_COORD_VALID)) {
                            continue;
                        }
                        if (umbilicus_sampled(row, col)) {
                            continue;
                        }
                        sampled_attempts++;
                        cv::Vec3f normal = grid_normal(
                            points, cv::Vec3f(static_cast<float>(col), static_cast<float>(row), 0.0f));
                        if (std::isfinite(normal[0]) && std::isfinite(normal[1]) && std::isfinite(normal[2]) &&
                            (normal[0] != 0 || normal[1] != 0 || normal[2] != 0)) {
                            umbilicus_estimator->add_sample(row, points(row, col), normal);
                            umbilicus_sampled(row, col) = 1;
                            sampled_added++;
                        } else {
                            sampled_invalid++;
                        }
                    }
                }
            }
            // Orient normals toward known umbilicus when available; otherwise use
            // the sample centroid while bootstrapping the first centers.
            {
                ScopedMs timer(timing_totals.umbilicus_orient_ms);
                if (umbilicus_ptr) {
                    umbilicus_estimator->orient_normals_to_umbilicus(*umbilicus_ptr);
                } else {
                    umbilicus_estimator->orient_normals_to_centroid();
                }
            }
            // Scan rows that have wrapped around and estimate centers
            int centers_estimated = 0;
            int rows_wrapped = 0;
            int rows_with_samples = 0;
            int rows_with_min_samples = 0;
            int max_samples = 0;
            const int min_wrap_samples = 8;
            {
                ScopedMs timer(timing_totals.umbilicus_row_scan_ms);
                for (int row = 0; row < points.rows; ++row) {
                    int sample_count = umbilicus_estimator->sample_count(row);
                    if (sample_count > 0) {
                        rows_with_samples++;
                        if (sample_count >= min_wrap_samples) {
                            rows_with_min_samples++;
                        }
                        if (sample_count > max_samples) {
                            max_samples = sample_count;
                        }
                    }
                    if (umbilicus_estimator->has_wrapped(row)) {
                        rows_wrapped++;
                        if (!umbilicus_estimator->has_center(row)) {
                            cv::Vec2d center;
                            double mean_z;
                            if (umbilicus_estimator->estimate_center(row, &center, &mean_z)) {
                                centers_estimated++;
                            }
                        }
                    }
                }
            }
            // Retry pending rows every 50 generations
            int centers_from_retry = 0;
            if (generation % 50 == 0 && generation > 0) {
                ScopedMs timer(timing_totals.umbilicus_retry_ms);
                centers_from_retry = umbilicus_estimator->retry_pending_rows();
            }

            const int centers_total = umbilicus_estimator->center_count();
            std::cout << "[UmbilicusEstimator] gen " << generation
                      << " samples_attempted=" << sampled_attempts
                      << " samples_added=" << sampled_added
                      << " samples_invalid=" << sampled_invalid
                      << ": rows_with_samples=" << rows_with_samples
                      << " rows_with_min_samples=" << rows_with_min_samples
                      << " max_samples=" << max_samples
                      << " rows_wrapped=" << rows_wrapped
                      << " centers_estimated=" << centers_estimated
                      << " centers_from_retry=" << centers_from_retry
                      << " centers_total=" << centers_total << std::endl;

            // Write JSON if we have any accepted centers
            // Write to paths/ directory (one level up from tgt_dir which is paths/segment/)
            if ((centers_estimated > 0 || centers_from_retry > 0) && centers_total > 0) {
                ScopedMs timer(timing_totals.umbilicus_write_json_ms);
                std::filesystem::path paths_dir = tgt_dir.parent_path();
                std::filesystem::path json_path = paths_dir / "estimated_umbilicus.json";
                umbilicus_estimator->write_json(json_path.string());
            }

            // Once we have new rows estimated, rebuild the umbilicus
            if (centers_estimated > 0 || centers_from_retry > 0) {
                try {
                    std::unique_ptr<vc::core::util::Umbilicus> new_umbilicus;
                    {
                        ScopedMs timer(timing_totals.umbilicus_build_ms);
                        auto built_umbilicus = umbilicus_estimator->build_umbilicus(volume_shape);
                        apply_umbilicus_seam(built_umbilicus);
                        new_umbilicus = std::make_unique<vc::core::util::Umbilicus>(std::move(built_umbilicus));
                        apply_wrap_growth_direction(*new_umbilicus);
                    }
                    {
                        ScopedMs timer(timing_totals.umbilicus_tracker_rebuild_ms);
                        auto new_tracker = std::make_unique<vc::wrap_tracking::WrapTracker>(
                            *new_umbilicus, points.rows, points.cols, wrap_flip_x);
                        double seed_theta_ref = seed_theta_reference(*new_umbilicus);
                        new_tracker->initialize_from_seed(data.seed_coord, data.seed_loc[1], seed_theta_ref);
                        // Populate existing cells
                        for (int r = 0; r < points.rows; ++r) {
                            for (int c = 0; c < points.cols; ++c) {
                                if (state(r, c) & STATE_COORD_VALID) {
                                    new_tracker->set_cell({r, c}, points(r, c));
                                }
                            }
                        }
                        const bool had_wrap_tracker = static_cast<bool>(wrap_tracker);
                        wrap_tracker = std::move(new_tracker);
                        umbilicus_ptr = std::move(new_umbilicus);
                        std::cout << "[WrapTracker] " << (had_wrap_tracker ? "Updated" : "Created")
                                  << " from estimated umbilicus (" << centers_total
                                  << " rows) at generation " << generation << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[WrapTracker] Failed to build umbilicus: " << e.what() << std::endl;
                }
            }
        }

        //continue expansion
        bool should_expand = fringe.empty() ||
                             (fringe_full_boundary &&
                              fringe.atAnyBorder() &&
                              cands.empty());
        const int max_width_cells = (max_width > 0) ? static_cast<int>(max_width / step) : w;
        const int max_height_cells = (max_height > 0) ? static_cast<int>(max_height / step) : h;
        int expand_right = 0;
        int expand_top = 0;
        int expand_bottom = 0;
        if (should_expand) {
            if (w < max_width_cells)
                expand_right = std::min(sliding_w, max_width_cells - w);
            if (h < max_height_cells) {
                const int available_h = max_height_cells - h;
                const int expand_step = std::min(sliding_w, available_h);
                if (fringe.atTopBorder() && fringe.atBottomBorder()) {
                    const int top_step = std::min(expand_step, available_h / 2);
                    expand_top = top_step;
                    expand_bottom = std::min(expand_step, available_h - top_step);
                } else if (fringe.atTopBorder()) {
                    expand_top = expand_step;
                } else if (fringe.atBottomBorder()) {
                    expand_bottom = expand_step;
                }
            }
        }
        if (should_expand && (expand_right > 0 || expand_top > 0 || expand_bottom > 0)) {
            ScopedMs expand_total_timer(timing_totals.expand_total_ms);
            fringe.clearBorderFlags();
            std::cout << "expanding by (right=" << expand_right
                      << ", top=" << expand_top
                      << ", bottom=" << expand_bottom << ")" << std::endl;

            std::cout << size << bounds << used_area << active_bounds << (used_area & active_bounds) << static_bounds << std::endl;
            final_opts = global_steps_per_window;

            const int old_w = w;
            const int old_h = h;
            const int y_offset = expand_top;
            {
                ScopedMs timer(timing_totals.expand_resize_ms);
                w += expand_right;
                h += (expand_top + expand_bottom);
                size = {w,h};
                bounds = {0,0,w-1,h-1};

                cv::Rect copy_roi(0, y_offset, old_w, old_h);

                cv::Mat_<cv::Vec3d> old_points = points;
                points = cv::Mat_<cv::Vec3d>(size, {-1,-1,-1});
                old_points.copyTo(points(copy_roi));

                cv::Mat_<uint8_t> old_state = state;
                state = cv::Mat_<uint8_t>(size, 0);
                old_state.copyTo(state(copy_roi));

                cv::Mat_<uint16_t> old_generations = generations;
                generations = cv::Mat_<uint16_t>(size, static_cast<uint16_t>(0));
                old_generations.copyTo(generations(copy_roi));

                cv::Mat_<uint16_t> old_inliers_sum_dbg = inliers_sum_dbg;
                inliers_sum_dbg = cv::Mat_<uint16_t>(size, 0);
                old_inliers_sum_dbg.copyTo(inliers_sum_dbg(copy_roi));

                cv::Mat_<uint8_t> old_umbilicus_sampled = umbilicus_sampled;
                umbilicus_sampled = cv::Mat_<uint8_t>(size, static_cast<uint8_t>(0));
                old_umbilicus_sampled.copyTo(umbilicus_sampled(copy_roi));

                cv::Mat_<uint8_t> old_force_retry_count = force_retry_count;
                force_retry_count = cv::Mat_<uint8_t>(size, static_cast<uint8_t>(0));
                old_force_retry_count.copyTo(force_retry_count(copy_roi));

                cv::Mat_<uint8_t> old_force_retry_mark = force_retry_mark;
                force_retry_mark = cv::Mat_<uint8_t>(size, static_cast<uint8_t>(0));
                old_force_retry_mark.copyTo(force_retry_mark(copy_roi));

                fringe.resize(size, copy_roi);

                wrap_rows_touched = std::vector<std::atomic<uint8_t>>(h);
                for (auto& flag : wrap_rows_touched)
                    flag.store(0, std::memory_order_relaxed);

                if ((expand_top + expand_bottom) > 0 && umbilicus_estimator) {
                    umbilicus_estimator = std::make_unique<vc::wrap_tracking::UmbilicusEstimator>(
                        points.rows);
                    umbilicus_sampled.setTo(0);
                    std::cout << "[UmbilicusEstimator] Reset after vertical resize" << std::endl;
                }
            }

            if (y_offset != 0) {
                ScopedMs timer(timing_totals.expand_shift_data_ms);
                used_area.y += y_offset;
                used_area_hr = {used_area.x*step_int, used_area.y*step_int, used_area.width*step_int, used_area.height*step_int};
                y0 += y_offset;
                auto shift_data_y = [&](SurfTrackerData& d, int dy) {
                    if (dy == 0)
                        return;
                    SurfTrackerData old = d;
                    d._data.clear();
                    d._res_blocks.clear();
                    d._surfs.clear();
                    for (auto& it : old._data) {
                        cv::Vec2i loc = it.first.second;
                        loc[0] += dy;
                        d._data[{it.first.first, loc}] = it.second;
                    }
                    for (auto& it : old._surfs) {
                        cv::Vec2i loc = it.first;
                        loc[0] += dy;
                        d._surfs[loc] = it.second;
                    }
                    d.seed_loc[0] += dy;
                };
                shift_data_y(data, y_offset);
                for (auto& p : force_retry)
                    p[0] += y_offset;
            }

            {
                ScopedMs timer(timing_totals.expand_thread_reset_ms);
                for(int i=0;i<omp_get_max_threads();i++) {
                    data_ths[i] = data;
                    added_points_threads[i].clear();
                }
            }

            if (wrap_tracker) {
                ScopedMs timer(timing_totals.expand_wrap_rebuild_ms);
                const auto& umbilicus = wrap_tracker->umbilicus();
                bool flip_x_local = wrap_tracker->flip_x();
                auto new_tracker = std::make_unique<vc::wrap_tracking::WrapTracker>(
                    umbilicus, points.rows, points.cols, flip_x_local);
                double seed_theta_ref = seed_theta_reference(umbilicus);
                new_tracker->initialize_from_seed(data.seed_coord, data.seed_loc[1], seed_theta_ref);
                int repopulated = 0;
                for (int r = 0; r < points.rows; ++r) {
                    for (int c = 0; c < points.cols; ++c) {
                        if (state(r, c) & STATE_COORD_VALID) {
                            new_tracker->set_cell({r, c}, points(r, c));
                            repopulated++;
                        }
                    }
                }
                std::cout << "[WrapTracker] Resized to " << points.rows << "x" << points.cols
                          << " (repopulated " << repopulated << " cells)" << std::endl;
                wrap_tracker = std::move(new_tracker);
            }

            {
                ScopedMs timer(timing_totals.expand_bounds_fringe_ms);
                if (expand_right > 0) {
                    int overlap = 5;
                    active_bounds = {w-sliding_w-2*closing_r-10-overlap,
                                     closing_r+5,
                                     sliding_w+2*closing_r+10+overlap,
                                     h-closing_r-10};
                    static_bounds = {0,0,w-sliding_w-2*closing_r-10,h};
                } else {
                    active_bounds.y = closing_r+5;
                    active_bounds.height = h - closing_r - 10;
                    static_bounds.height = h;
                }

                cv::Rect active = active_bounds & used_area;

                std::cout << size << bounds << used_area << active_bounds << (used_area & active_bounds) << static_bounds << std::endl;
                curr_best_inl_th = inlier_base_threshold;
                if (fringe_full_boundary) {
                    build_savable_mask();
                    fringe.rebuildBoundary();
                } else {
                    fringe.rebuildIncremental(2);
                }
            }
            // Grid grew; rebuild area mask/accumulator in new shape
            {
                ScopedMs timer(timing_totals.expand_area_scan_ms);
                init_area_scan();
            }
        }

        if (generation % 1000 == 0) {
            ScopedMs timer(timing_totals.inliers_write_ms);
            try {
                writeTiff(tgt_dir / "inliers_sum.tif", inliers_sum_dbg(used_area));
            } catch (const std::exception& ex) {
                std::cerr << "warning: failed to write inliers_sum.tif: " << ex.what() << std::endl;
            }
        }

        if (fringe.empty()) {
            if (debug_diagnostics) {
                int valid_cells = 0;
                for (int j = used_area.y; j < used_area.br().y; ++j)
                    for (int i = used_area.x; i < used_area.br().x; ++i)
                        if (state(j, i) & STATE_LOC_VALID)
                            valid_cells++;
                std::cout << "[GrowSurface][diag] fringe empty at gen " << generation
                          << " valid=" << valid_cells
                          << " cands=" << cands_vec.size()
                          << " force_retry=" << force_retry.size()
                          << " used_area=" << used_area << std::endl;
            }
            finish_generation_timing(true);
            break;
        }
        finish_generation_timing(false);
    }
    if (timing_totals.generations > 0) {
        timing_totals.print(timing_report_start_generation, std::max(0, stop_gen - 1));
    }

    // Final exact surface area from final optimized geometry.
    double area_final_vox2 = 0.0;
    for (int j = 0; j < state.rows - 1; ++j)
        for (int i = 0; i < state.cols - 1; ++i)
            if ( (state(j,   i  ) & STATE_LOC_VALID) &&
                 (state(j,   i+1) & STATE_LOC_VALID) &&
                 (state(j+1, i  ) & STATE_LOC_VALID) &&
                 (state(j+1, i+1) & STATE_LOC_VALID) )
            {
                area_final_vox2 += vc::surface::quadAreaVox2(points(j,   i  ),
                                                            points(j,   i+1),
                                                            points(j+1, i  ),
                                                            points(j+1, i+1));
            }
    const double area_final_cm2 = area_final_vox2 * double(voxelsize) * double(voxelsize) / 1e8;
    std::cout << "area exact: " << area_final_vox2 << " vx^2 (" << area_final_cm2 << " cm^2)" << std::endl;

    cv::Mat_<cv::Vec3f> points_hr =
        surftrack_genpoints_hr(data, state, points, used_area, step, src_step,
                               /*inpaint=*/false);

    auto surf = new QuadSurface(points_hr(used_area_hr), {1/src_step,1/src_step});

    auto gen_channel = surftrack_generation_channel(generations, used_area, step);
    if (!gen_channel.empty())
        surf->setChannel("generations", gen_channel);

    surf->meta = utils::Json::object();
    surf->meta["max_gen"] = surftrack_max_generation(generations, used_area);
    surf->meta["area_vx2"] = area_final_vox2;
    surf->meta["area_cm2"] = area_final_cm2;
    surf->meta["seed_surface_name"] = seed_surface_name;
    surf->meta["seed_surface_id"] = seed_surface_id;

    std::cout << "pointTo total time: " << pointTo_total_ms << " ms" << std::endl;

    return surf;
}
