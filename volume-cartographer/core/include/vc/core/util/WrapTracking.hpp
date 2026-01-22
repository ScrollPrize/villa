#pragma once

#include <opencv2/core.hpp>
#include "vc/core/util/Umbilicus.hpp"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace vc::wrap_tracking {

// Per-cell spiral state
struct SpiralCellState {
    double theta_deg{0.0};            // Raw angle in [0, 360) from Umbilicus::theta
    double theta_unwrapped_deg{0.0};  // Monotonically increasing (or decreasing if flip_x)
    int wrap_index{0};                // floor(theta_unwrapped / 360)
    double wrap_frac{0.0};            // theta_unwrapped / 360 (fractional wraps)
    double radius{0.0};               // Distance to umbilicus center at current z
    double z{0.0};                    // Z used for radius/theta (clipped if needed)
    uint8_t flags{0};                 // 0x1=initialized
};

class WrapTracker {
public:
    WrapTracker(const core::util::Umbilicus& umbilicus, int rows, int cols, bool flip_x);

    // Initialization - call once with seed to establish theta reference
    void initialize_from_seed(const cv::Vec3d& seed_coord, int seed_col);
    void initialize_from_seed(const cv::Vec3d& seed_coord, int seed_col, double reference_theta_deg);

    // Core operations
    void set_cell(const cv::Vec2i& p, const cv::Vec3d& coord);
    void unwrap_row(int row, const cv::Mat_<uint8_t>& state);
    void ensure_row_unwrapped(int row, const cv::Mat_<uint8_t>& state);
    void correct_wrap_from_neighbors(int row, const cv::Mat_<uint8_t>& state);
    void compute_statistics(const cv::Mat_<uint8_t>& state);

    // Queries
    SpiralCellState get_cell(const cv::Vec2i& p) const;
    double expected_theta_for_col(int col) const;
    double expected_dtheta_per_step() const;
    double expected_radial_slope_for_col(int col) const;
    bool has_sufficient_column_samples(int col, int min_pts) const;
    bool has_sufficient_dtheta_samples(int min_pts) const;
    bool has_sufficient_radial_slope_samples(int col, int min_pts) const;

    // Loss activation checks
    bool losses_ready() const;

    // Accessors
    const core::util::Umbilicus& umbilicus() const { return _umbilicus; }
    bool flip_x() const { return _flip_x; }
    int rows() const { return _rows; }
    int cols() const { return _cols; }
    double base_theta_offset() const { return _base_theta_offset; }

    // Logging control (set generation to gate verbose prints to every 1000 gens)
    void set_log_generation(int gen) { _log_generation = gen; }

    // Cached umbilicus center lookup (avoids repeated spline evaluation)
    cv::Vec3f cached_center_at(int z_index) const;

private:
    const core::util::Umbilicus& _umbilicus;
    std::vector<SpiralCellState> _cells;  // Row-major storage: _cells[row * _cols + col]
    int _rows;
    int _cols;
    int _seed_col{0};  // Reference column for unwrapping (stable anchor)
    bool _flip_x;

    // Statistics
    std::vector<double> _expected_theta_per_col;          // Per-column median/fit
    std::vector<int> _column_sample_counts;               // Number of valid samples per column
    double _global_dtheta_per_step{0.0};                  // Global median step
    int _dtheta_sample_count{0};                          // Number of dtheta samples used
    bool _has_expected_theta{false};
    std::vector<double> _expected_radial_slope_per_col;   // Per-column median dr/dz
    std::vector<int> _radial_slope_sample_counts;         // Number of slope samples per column
    std::vector<uint8_t> _row_dirty;

    // Theta offset (set from seed)
    double _base_theta_offset{0.0};   // Offset for theta normalization

    // Logging control: -1 = always log, else log when gen % 1000 == 0
    int _log_generation{-1};

    // Umbilicus center cache (avoids repeated spline evaluation)
    mutable std::vector<cv::Vec3f> _center_cache;
    mutable std::vector<bool> _center_cache_valid;

    // Helpers
    double normalize_theta(double raw_theta) const;  // Apply base offset
    void update_cell_indices(SpiralCellState& cell);  // Compute wrap_index, wrap_frac
};

// Umbilicus estimator for when no umbilicus file is provided
// Estimates center from surface normal convergence using RANSAC
// Samples are binned by surface row (each row spirals around, sampling all angles)
class UmbilicusEstimator {
public:
    explicit UmbilicusEstimator(int num_rows, const std::string& json_path = "");

    // Add sample with row index
    void add_sample(int row, const cv::Vec3d& point, const cv::Vec3f& normal);

    // Estimate umbilicus center for a row (call when row has wrapped around)
    bool estimate_center(int row, cv::Vec2d* center_out, double* mean_z_out);

    // Check if row has wrapped around (samples in all 4 quadrants)
    bool has_wrapped(int row) const;
    bool has_center(int row) const;
    int sample_count(int row) const;
    int center_count() const;

    // Build full Umbilicus object from estimates
    core::util::Umbilicus build_umbilicus(const cv::Vec3i& volume_shape) const;

    // Re-estimate rows that haven't met threshold yet (if more samples available)
    int retry_pending_rows();

    // Orient all sample normals to point toward the mesh centroid
    // Should be called before estimation to ensure consistent normal direction
    void orient_normals_to_centroid();

    // Get score for a row (-1 if not estimated)
    double get_score(int row) const;

    // Check if there are any loaded/estimated points (for initial umbilicus build)
    bool has_any_points() const;

    // Export to JSON file atomically (write to .tmp then rename)
    bool write_json(const std::string& path) const;

    // Logging control (set generation to gate verbose prints to every 1000 gens)
    void set_log_generation(int gen) { _log_generation = gen; }

private:
    // Per-row sample storage
    struct Sample {
        cv::Vec2d point_xy;   // XY position
        cv::Vec2f normal_xy;  // XY normal component (normalized)
        double z;             // Z coordinate of sample
    };
    struct SampleStats {
        cv::Vec2d median_xy;
        double median_radius;
    };
    std::vector<std::vector<Sample>> _samples_per_row;
    std::vector<cv::Vec2d> _estimated_centers;  // Per-row estimated centers
    std::vector<double> _estimated_z;           // Mean Z for each row's estimate
    std::vector<bool> _center_valid;
    std::vector<double> _estimated_scores;      // Score for each accepted row
    std::vector<double> _last_attempted_score;  // Last score attempted (even if rejected)
    std::vector<int> _last_attempted_sample_count;  // Sample count at last attempt
    std::vector<bool> _retry_exhausted;         // True if re-estimate didn't improve
    std::vector<int> _retry_fail_count;         // Number of consecutive non-improving retries
    static constexpr double kMinScoreThreshold{0.75};
    static constexpr double kHighScoreThreshold{0.75};
    static constexpr double kLowScoreZTolerance{200.0};  // For anchor proximity check
    static constexpr double kSameZTolerance{50.0};       // Points within 50 voxels are same Z (grid spacing)

    // Unified control points list (loaded + estimated, managed by Z)
    struct ControlPoint {
        cv::Vec2d xy;
        double z;
        double score;
    };
    std::vector<ControlPoint> _control_points;

    // Logging control: -1 = always log, else log when gen % 1000 == 0
    int _log_generation{-1};

    // Load existing estimates from JSON
    bool load_json(const std::string& path);

    // Add or update control point (replaces if within kSameZTolerance and score is better)
    void add_or_update_control_point(const cv::Vec2d& xy, double z, double score);

    // Filter control points by score for robust interpolation
    std::vector<cv::Vec3f> filter_control_points_by_score() const;

    SampleStats compute_sample_stats(const std::vector<Sample>& samples) const;
    double center_offset_penalty(const SampleStats& stats, const cv::Vec2d& center) const;
    cv::Vec2d ransac_find_center(const std::vector<Sample>& samples, const SampleStats& stats) const;
    cv::Vec2d refine_center(const std::vector<Sample>& samples,
                            const SampleStats& stats,
                            cv::Vec2d initial) const;
    double score_center(const std::vector<Sample>& samples,
                        const SampleStats& stats,
                        const cv::Vec2d& center) const;
    std::vector<Sample> filter_samples_for_estimate(const std::vector<Sample>& samples) const;
    bool center_passes_sanity(const std::vector<Sample>& samples,
                              const SampleStats& stats,
                              const cv::Vec2d& center) const;
    // Gather samples from target row and neighbors (row-1, row, row+1) for more robust estimation
    std::vector<Sample> gather_samples_for_row(int row) const;
};

} // namespace vc::wrap_tracking
