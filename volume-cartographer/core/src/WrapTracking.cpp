#include "vc/core/util/WrapTracking.hpp"

// State bit flags (subset of SurfaceModeling.hpp, avoiding Ceres dependency)
#define STATE_LOC_VALID 1

#include <algorithm>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>

namespace vc::wrap_tracking {

namespace {
inline bool cell_is_valid(const SpiralCellState& cell,
                          const cv::Mat_<uint8_t>& state,
                          int row,
                          int col) {
    return (cell.flags & 0x1) && (state(row, col) & STATE_LOC_VALID);
}

constexpr int kMaxZClipLogs = 10;
constexpr int kUmbilicusMinFilterSamples = 12;
constexpr int kUmbilicusMinFilteredSamples = 8;
constexpr double kUmbilicusMadScale = 8.0;
constexpr double kUmbilicusMinMad = 1e-3;
constexpr double kUmbilicusMaxCenterOffsetScale = 4.0;
constexpr double kWrapReverseToleranceDeg = 30.0;
constexpr double kWrapReversePenaltyDeg = 720.0;

double median_of(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }

    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    double med = values[mid];
    if (values.size() % 2 == 0) {
        const auto lower_max = std::max_element(values.begin(), values.begin() + mid);
        med = 0.5 * (med + *lower_max);
    }
    return med;
}

double choose_unwrapped_theta(double theta_deg,
                              double prev_unwrapped,
                              int prev_wrap,
                              double expected_step,
                              int direction_sign) {
    const double target = prev_unwrapped + expected_step;
    double best_theta = theta_deg + 360.0 * prev_wrap;
    double best_score = std::numeric_limits<double>::infinity();

    for (int dk = -1; dk <= 1; ++dk) {
        int wrap = prev_wrap + dk;
        double candidate = theta_deg + 360.0 * wrap;
        double score = std::abs(candidate - target);

        if (direction_sign > 0) {
            if (candidate < prev_unwrapped - kWrapReverseToleranceDeg) {
                score += kWrapReversePenaltyDeg;
            }
        } else if (direction_sign < 0) {
            if (candidate > prev_unwrapped + kWrapReverseToleranceDeg) {
                score += kWrapReversePenaltyDeg;
            }
        }

        if (score < best_score) {
            best_score = score;
            best_theta = candidate;
        }
    }

    return best_theta;
}

int clip_z_index(const cv::Vec2i& p, double z_value, int max_z, const char* context, bool* clipped_out) {
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
            std::cerr << "[WrapTracker] Clipping z at " << context
                      << " row=" << p[0] << " col=" << p[1]
                      << " z=" << z_value << " -> " << z_index
                      << " (max=" << max_z << ")";
            if (non_finite) {
                std::cerr << " non-finite";
            }
            std::cerr << std::endl;
            if (hit + 1 == kMaxZClipLogs) {
                std::cerr << "[WrapTracker] Further z clip warnings suppressed" << std::endl;
            }
        }
    }

    if (clipped_out) {
        *clipped_out = clipped;
    }
    return z_index;
}
}  // namespace

// ============================================================================
// WrapTracker Implementation
// ============================================================================

WrapTracker::WrapTracker(const core::util::Umbilicus& umbilicus, int rows, int cols, bool flip_x)
    : _umbilicus(umbilicus)
    , _cells(rows * cols)  // Row-major storage
    , _rows(rows)
    , _cols(cols)
    , _flip_x(flip_x)
    , _expected_theta_per_col(cols, 0.0)
    , _column_sample_counts(cols, 0)
    , _expected_radial_slope_per_col(cols, 0.0)
    , _radial_slope_sample_counts(cols, 0)
    , _row_dirty(rows, 0)
    , _center_cache(umbilicus.volume_shape()[0])
    , _center_cache_valid(umbilicus.volume_shape()[0], false)
{
    // Cells are default-initialized by vector constructor
}

cv::Vec3f WrapTracker::cached_center_at(int z_index) const {
    if (z_index < 0 || z_index >= static_cast<int>(_center_cache.size())) {
        // Fallback for out-of-bounds (shouldn't happen with proper clipping)
        return _umbilicus.center_at(std::clamp(z_index, 0, static_cast<int>(_center_cache.size()) - 1));
    }
    if (!_center_cache_valid[z_index]) {
        _center_cache[z_index] = _umbilicus.center_at(z_index);
        _center_cache_valid[z_index] = true;
    }
    return _center_cache[z_index];
}

void WrapTracker::initialize_from_seed(const cv::Vec3d& seed_coord, int seed_col) {
    initialize_from_seed(seed_coord, seed_col, std::numeric_limits<double>::quiet_NaN());
}

void WrapTracker::initialize_from_seed(const cv::Vec3d& seed_coord, int seed_col, double reference_theta_deg) {
    // Store seed column as reference for unwrapping
    _seed_col = seed_col;

    // Get raw theta at seed position (respects umbilicus seam)
    cv::Vec3f seed_point(static_cast<float>(seed_coord[0]),
                         static_cast<float>(seed_coord[1]),
                         static_cast<float>(seed_coord[2]));
    double raw_theta = _umbilicus.theta(seed_point, 0);
    double theta_ref = std::isfinite(reference_theta_deg) ? reference_theta_deg : raw_theta;

    // Align reference theta to 0 for stability (median around seed if provided)
    _base_theta_offset = -theta_ref;

    if (_log_generation < 0 || _log_generation % 1000 == 0) {
        std::cout << "[WrapTracker] Initialized from seed at theta=" << raw_theta;
        if (std::isfinite(reference_theta_deg))
            std::cout << ", ref_theta=" << theta_ref;
        std::cout << ", base_offset=" << _base_theta_offset << std::endl;
    }
}

double WrapTracker::normalize_theta(double raw_theta) const {
    double normalized = raw_theta + _base_theta_offset;
    // Keep in [0, 360) for single-wrap calculations
    while (normalized < 0) normalized += 360.0;
    while (normalized >= 360.0) normalized -= 360.0;
    return normalized;
}

void WrapTracker::set_cell(const cv::Vec2i& p, const cv::Vec3d& coord) {
    if (p[0] < 0 || p[0] >= _rows || p[1] < 0 || p[1] >= _cols) {
        return;
    }

    // Validate all coordinates are finite - skip cells with NaN/Inf
    if (!std::isfinite(coord[0]) || !std::isfinite(coord[1]) || !std::isfinite(coord[2])) {
        return;
    }

    const double z_value = coord[2];
    const int max_z = _umbilicus.volume_shape()[0];
    bool clipped = false;
    const int z_index = clip_z_index(p, z_value, max_z, "set_cell", &clipped);
    const double z_theta = clipped ? static_cast<double>(z_index) : z_value;

    SpiralCellState& cell = _cells[p[0] * _cols + p[1]];
    cv::Vec3f center = _umbilicus.center_at(z_index);
    double dx = coord[0] - center[0];
    double dy = coord[1] - center[1];

    double radius = std::sqrt(dx * dx + dy * dy);
    cv::Vec3f point(static_cast<float>(coord[0]),
                    static_cast<float>(coord[1]),
                    static_cast<float>(z_theta));
    double raw_theta = _umbilicus.theta(point, 0);

    cell.theta_deg = normalize_theta(raw_theta);
    cell.radius = radius;
    cell.z = z_theta;
    cell.flags = 0x1;  // Mark as initialized
    _row_dirty[p[0]] = 1;

    // theta_unwrapped, wrap_index, wrap_frac are computed in unwrap_row
}

void WrapTracker::update_cell_indices(SpiralCellState& cell) {
    cell.wrap_index = static_cast<int>(std::floor(cell.theta_unwrapped_deg / 360.0));
    cell.wrap_frac = cell.theta_unwrapped_deg / 360.0;
}

void WrapTracker::unwrap_row(int row, const cv::Mat_<uint8_t>& state) {
    if (row < 0 || row >= _rows) return;

    // Use seed column as the reference for all rows (stable anchor)
    // Fall back to nearest valid cell only if seed_col cell isn't valid
    int ref_col = _seed_col;
    if (ref_col < 0 || ref_col >= _cols ||
        !cell_is_valid(_cells[row * _cols + ref_col], state, row, ref_col)) {
        // Seed col not valid in this row, find nearest valid cell
        ref_col = -1;
        int best_dist = std::numeric_limits<int>::max();
        for (int col = 0; col < _cols; ++col) {
            if (!cell_is_valid(_cells[row * _cols + col], state, row, col)) continue;
            int dist = std::abs(col - _seed_col);
            if (dist < best_dist) {
                best_dist = dist;
                ref_col = col;
            }
        }
    }
    if (ref_col < 0) return;  // No valid cells in row

    // Reference cell starts at cumulative = 0 (theta_unwrapped = theta_deg)
    SpiralCellState& ref_cell = _cells[row * _cols + ref_col];
    ref_cell.theta_unwrapped_deg = ref_cell.theta_deg;
    update_cell_indices(ref_cell);

    const double expected_step =
        (_dtheta_sample_count > 0) ? _global_dtheta_per_step : 0.0;
    const int sign_right = _flip_x ? -1 : 1;
    const int sign_left = -sign_right;

    // Unwrap LEFTWARD from reference (cols < ref_col)
    double prev_unwrapped = ref_cell.theta_unwrapped_deg;
    for (int col = ref_col - 1; col >= 0; --col) {
        SpiralCellState& cell = _cells[row * _cols + col];
        if (!cell_is_valid(cell, state, row, col)) continue;

        int prev_wrap = static_cast<int>(std::floor(prev_unwrapped / 360.0));
        cell.theta_unwrapped_deg = choose_unwrapped_theta(
            cell.theta_deg, prev_unwrapped, prev_wrap, -expected_step, sign_left);
        update_cell_indices(cell);
        prev_unwrapped = cell.theta_unwrapped_deg;
    }

    // Unwrap RIGHTWARD from reference (cols > ref_col)
    prev_unwrapped = ref_cell.theta_unwrapped_deg;
    for (int col = ref_col + 1; col < _cols; ++col) {
        SpiralCellState& cell = _cells[row * _cols + col];
        if (!cell_is_valid(cell, state, row, col)) continue;

        int prev_wrap = static_cast<int>(std::floor(prev_unwrapped / 360.0));
        cell.theta_unwrapped_deg = choose_unwrapped_theta(
            cell.theta_deg, prev_unwrapped, prev_wrap, expected_step, sign_right);
        update_cell_indices(cell);
        prev_unwrapped = cell.theta_unwrapped_deg;
    }

    if (_has_expected_theta) {
        std::vector<double> row_offsets;
        row_offsets.reserve(_cols);
        for (int col = 0; col < _cols; ++col) {
            SpiralCellState& cell = _cells[row * _cols + col];
            if (!cell_is_valid(cell, state, row, col)) continue;
            if (_column_sample_counts[col] <= 0) continue;
            double diff = _expected_theta_per_col[col] - cell.theta_unwrapped_deg;
            double offset = std::round(diff / 360.0) * 360.0;
            row_offsets.push_back(offset);
        }

        if (!row_offsets.empty()) {
            size_t mid = row_offsets.size() / 2;
            std::nth_element(row_offsets.begin(), row_offsets.begin() + mid, row_offsets.end());
            double row_offset = row_offsets[mid];
            if (row_offset != 0.0) {
                for (int col = 0; col < _cols; ++col) {
                    SpiralCellState& cell = _cells[row * _cols + col];
                    if (!cell_is_valid(cell, state, row, col)) continue;
                    cell.theta_unwrapped_deg += row_offset;
                    update_cell_indices(cell);
                }
            }
        }
    }

    _row_dirty[row] = 0;
}

void WrapTracker::ensure_row_unwrapped(int row, const cv::Mat_<uint8_t>& state) {
    if (row < 0 || row >= _rows) return;
    if (!_row_dirty[row]) return;
    unwrap_row(row, state);
}

void WrapTracker::correct_wrap_from_neighbors(int row, const cv::Mat_<uint8_t>& state) {
    if (row < 0 || row >= _rows) return;

    for (int col = 0; col < _cols; ++col) {
        SpiralCellState& cell = _cells[row * _cols + col];
        if (!cell_is_valid(cell, state, row, col)) continue;
        if (!std::isfinite(cell.theta_unwrapped_deg)) continue;

        // Find a valid neighbor in adjacent row (prefer row-1, fallback to row+1)
        double neighbor_theta = std::numeric_limits<double>::quiet_NaN();
        for (int dr : {-1, 1}) {
            int nr = row + dr;
            if (nr < 0 || nr >= _rows) continue;
            const SpiralCellState& neighbor = _cells[nr * _cols + col];
            if (!cell_is_valid(neighbor, state, nr, col)) continue;
            if (!std::isfinite(neighbor.theta_unwrapped_deg)) continue;
            neighbor_theta = neighbor.theta_unwrapped_deg;
            break;
        }

        if (!std::isfinite(neighbor_theta)) continue;

        // Correct only whole-wrap errors - preserve natural tilt
        double diff = neighbor_theta - cell.theta_unwrapped_deg;
        double wrap_correction = std::round(diff / 360.0) * 360.0;

        if (std::abs(wrap_correction) >= 360.0) {
            cell.theta_unwrapped_deg += wrap_correction;
            update_cell_indices(cell);
        }
    }
}

void WrapTracker::compute_statistics(const cv::Mat_<uint8_t>& state) {
    // Clear old statistics
    std::fill(_column_sample_counts.begin(), _column_sample_counts.end(), 0);

    // Compute expected theta per column
    std::vector<std::vector<double>> col_theta_samples(_cols);

    for (int row = 0; row < _rows; ++row) {
        for (int col = 0; col < _cols; ++col) {
            const SpiralCellState& cell = _cells[row * _cols + col];
            if (cell_is_valid(cell, state, row, col)) {
                // Defense: skip cells with non-finite theta or radius
                if (!std::isfinite(cell.theta_unwrapped_deg) || !std::isfinite(cell.radius)) {
                    continue;
                }
                col_theta_samples[col].push_back(cell.theta_unwrapped_deg);
            }
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (int col = 0; col < _cols; ++col) {
        auto& samples = col_theta_samples[col];
        _column_sample_counts[col] = static_cast<int>(samples.size());

        if (samples.empty()) {
            _expected_theta_per_col[col] = 0.0;
        } else {
            size_t mid = samples.size() / 2;
            std::nth_element(samples.begin(), samples.begin() + mid, samples.end());
            _expected_theta_per_col[col] = samples[mid];
        }
    }

    // Compute expected radial slope (dr/dz) per column
    std::vector<std::vector<double>> col_slope_samples(_cols);

    for (int row = 0; row + 1 < _rows; ++row) {
        for (int col = 0; col < _cols; ++col) {
            const SpiralCellState& cell_a = _cells[row * _cols + col];
            const SpiralCellState& cell_b = _cells[(row + 1) * _cols + col];
            if (!cell_is_valid(cell_a, state, row, col) ||
                !cell_is_valid(cell_b, state, row + 1, col)) {
                continue;
            }
            if (!std::isfinite(cell_a.radius) || !std::isfinite(cell_b.radius) ||
                !std::isfinite(cell_a.z) || !std::isfinite(cell_b.z)) {
                continue;
            }
            double dz = cell_b.z - cell_a.z;
            if (std::abs(dz) < 1e-6) {
                continue;
            }
            double slope = (cell_b.radius - cell_a.radius) / dz;
            if (!std::isfinite(slope)) {
                continue;
            }
            col_slope_samples[col].push_back(slope);
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (int col = 0; col < _cols; ++col) {
        auto& samples = col_slope_samples[col];
        _radial_slope_sample_counts[col] = static_cast<int>(samples.size());
        if (samples.empty()) {
            _expected_radial_slope_per_col[col] = 0.0;
        } else {
            size_t mid = samples.size() / 2;
            std::nth_element(samples.begin(), samples.begin() + mid, samples.end());
            _expected_radial_slope_per_col[col] = samples[mid];
        }
    }

    // Compute global dtheta per step
    std::vector<double> dtheta_samples;

    for (int row = 0; row < _rows; ++row) {
        double prev_theta = -1e9;
        bool prev_valid = false;

        for (int col = 0; col < _cols; ++col) {
            const SpiralCellState& cell = _cells[row * _cols + col];
            if (!cell_is_valid(cell, state, row, col)) {
                prev_valid = false;
                continue;
            }
            // Defense: skip cells with non-finite theta
            if (!std::isfinite(cell.theta_unwrapped_deg)) {
                prev_valid = false;
                continue;
            }

            if (prev_valid) {
                double dtheta = cell.theta_unwrapped_deg - prev_theta;
                // Only include reasonable steps (not huge jumps from gaps)
                if (std::abs(dtheta) < 90.0) {
                    dtheta_samples.push_back(dtheta);
                }
            }

            prev_theta = cell.theta_unwrapped_deg;
            prev_valid = true;
        }
    }

    if (!dtheta_samples.empty()) {
        size_t mid = dtheta_samples.size() / 2;
        std::nth_element(dtheta_samples.begin(), dtheta_samples.begin() + mid, dtheta_samples.end());
        _global_dtheta_per_step = dtheta_samples[mid];
        _dtheta_sample_count = static_cast<int>(dtheta_samples.size());
    } else {
        _global_dtheta_per_step = 0.0;
        _dtheta_sample_count = 0;
    }

    // Print statistics summary
    int total_cells = 0;
    double min_theta = 1e9, max_theta = -1e9;
    int min_wrap = std::numeric_limits<int>::max();
    int max_wrap = std::numeric_limits<int>::min();
    for (int i = 0; i < _rows * _cols; ++i) {
        int row = i / _cols;
        int col = i % _cols;
        if (cell_is_valid(_cells[i], state, row, col)) {
            total_cells++;
            min_theta = std::min(min_theta, _cells[i].theta_unwrapped_deg);
            max_theta = std::max(max_theta, _cells[i].theta_unwrapped_deg);
            min_wrap = std::min(min_wrap, _cells[i].wrap_index);
            max_wrap = std::max(max_wrap, _cells[i].wrap_index);
        }
    }
    int cols_with_data = 0;
    for (int c = 0; c < _cols; ++c) {
        if (_column_sample_counts[c] > 0) cols_with_data++;
    }
    bool has_expected = false;
    for (int col = 0; col < _cols; ++col) {
        if (_column_sample_counts[col] > 0) {
            has_expected = true;
            break;
        }
    }
    _has_expected_theta = has_expected;
    double angular_span = (max_theta - min_theta) / 360.0;
    if (_log_generation < 0 || _log_generation % 1000 == 0) {
        std::cout << "[WrapTracker] Stats: " << total_cells << " cells, "
                  << "wraps " << min_wrap << "-" << max_wrap << ", "
                  << "span=" << std::fixed << std::setprecision(2) << angular_span << ", "
                  << "dtheta/step=" << _global_dtheta_per_step << "deg, "
                  << cols_with_data << "/" << _cols << " cols with data" << std::endl;
    }
}

SpiralCellState WrapTracker::get_cell(const cv::Vec2i& p) const {
    if (p[0] < 0 || p[0] >= _rows || p[1] < 0 || p[1] >= _cols) {
        return SpiralCellState{};
    }
    return _cells[p[0] * _cols + p[1]];
}

double WrapTracker::expected_theta_for_col(int col) const {
    if (col < 0 || col >= static_cast<int>(_expected_theta_per_col.size())) {
        return 0.0;
    }
    return _expected_theta_per_col[col];
}

double WrapTracker::expected_dtheta_per_step() const {
    return _global_dtheta_per_step;
}

double WrapTracker::expected_radial_slope_for_col(int col) const {
    if (col < 0 || col >= static_cast<int>(_expected_radial_slope_per_col.size())) {
        return 0.0;
    }
    return _expected_radial_slope_per_col[col];
}

bool WrapTracker::has_sufficient_column_samples(int col, int min_pts) const {
    if (col < 0 || col >= static_cast<int>(_column_sample_counts.size())) {
        return false;
    }
    return _column_sample_counts[col] >= min_pts;
}

bool WrapTracker::has_sufficient_dtheta_samples(int min_pts) const {
    return _dtheta_sample_count >= min_pts;
}

bool WrapTracker::has_sufficient_radial_slope_samples(int col, int min_pts) const {
    if (col < 0 || col >= static_cast<int>(_radial_slope_sample_counts.size())) {
        return false;
    }
    return _radial_slope_sample_counts[col] >= min_pts;
}

bool WrapTracker::losses_ready() const {
    // Check if we have sufficient statistics for wrap-based losses
    return _has_expected_theta && _dtheta_sample_count > 0;
}

// ============================================================================
// UmbilicusEstimator Implementation
// ============================================================================

UmbilicusEstimator::UmbilicusEstimator(int num_rows, const std::string& json_path)
    : _samples_per_row(num_rows)
    , _estimated_centers(num_rows)
    , _estimated_z(num_rows, 0.0)
    , _center_valid(num_rows, false)
    , _estimated_scores(num_rows, -1.0)
    , _last_attempted_score(num_rows, -1.0)
    , _last_attempted_sample_count(num_rows, 0)
    , _retry_exhausted(num_rows, false)
    , _retry_fail_count(num_rows, 0)
{
    if (!json_path.empty()) {
        load_json(json_path);
    }
}

void UmbilicusEstimator::add_sample(int row, const cv::Vec3d& point, const cv::Vec3f& normal) {
    if (row < 0 || row >= static_cast<int>(_samples_per_row.size())) return;
    if (_center_valid[row]) return;

    if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) {
        return;
    }

    if (!std::isfinite(normal[0]) || !std::isfinite(normal[1]) || !std::isfinite(normal[2])) {
        return;
    }

    // Normalize XY normal component
    cv::Vec2f normal_xy(normal[0], normal[1]);
    float len = std::sqrt(normal_xy[0] * normal_xy[0] + normal_xy[1] * normal_xy[1]);
    if (len < 1e-6f) return;  // Skip if normal is mostly vertical

    normal_xy /= len;

    Sample sample;
    sample.point_xy = cv::Vec2d(point[0], point[1]);
    sample.normal_xy = normal_xy;
    sample.z = point[2];

    _samples_per_row[row].push_back(sample);

}

UmbilicusEstimator::SampleStats UmbilicusEstimator::compute_sample_stats(
    const std::vector<Sample>& samples) const {
    SampleStats stats;
    stats.median_xy = cv::Vec2d(0.0, 0.0);
    stats.median_radius = 0.0;
    if (samples.empty()) {
        return stats;
    }

    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(samples.size());
    ys.reserve(samples.size());
    for (const auto& s : samples) {
        xs.push_back(s.point_xy[0]);
        ys.push_back(s.point_xy[1]);
    }

    const double med_x = median_of(xs);
    const double med_y = median_of(ys);
    stats.median_xy = cv::Vec2d(med_x, med_y);

    std::vector<double> distances;
    distances.reserve(samples.size());
    for (const auto& s : samples) {
        distances.push_back(cv::norm(s.point_xy - stats.median_xy));
    }
    stats.median_radius = median_of(distances);
    return stats;
}

double UmbilicusEstimator::center_offset_penalty(const SampleStats& stats,
                                                 const cv::Vec2d& center) const {
    if (stats.median_radius < kUmbilicusMinMad) {
        return 1.0;
    }
    const double offset = cv::norm(center - stats.median_xy);
    const double ratio = offset / std::max(stats.median_radius, kUmbilicusMinMad);
    return 1.0 / (1.0 + ratio * ratio);
}

double UmbilicusEstimator::score_center(const std::vector<Sample>& samples,
                                        const SampleStats& stats,
                                        const cv::Vec2d& center) const {
    double toward_sum = 0.0;  // Normals pointing toward center (correct)
    double away_sum = 0.0;    // Normals pointing away from center
    double wsum = 0.0;

    for (const auto& s : samples) {
        cv::Vec2d to_center = center - s.point_xy;
        double dist = cv::norm(to_center);
        if (dist < 1e-6) continue;

        to_center /= dist;  // Normalize

        // Positive = normal points toward center (correct for scroll)
        // Negative = normal points away from center
        double cos_angle = to_center[0] * s.normal_xy[0] + to_center[1] * s.normal_xy[1];

        double weight = 1.0 / std::max(100.0, dist);

        if (cos_angle > 0) {
            toward_sum += cos_angle * weight;
        } else {
            away_sum += (-cos_angle) * weight;
        }
        wsum += weight;
    }

    if (wsum < 1e-9) return 0.0;

    double total = toward_sum + away_sum;
    if (total < 1e-9) return 0.0;

    // Score = (fraction pointing toward) × (alignment strength)
    // True center: all normals point toward → dominance ≈ 1.0
    // Wrong center: mixed directions → dominance ≈ 0.5
    double dominance = toward_sum / total;
    double alignment = total / wsum;

    const double base_score = dominance * alignment;
    return base_score * center_offset_penalty(stats, center);
}

std::vector<UmbilicusEstimator::Sample> UmbilicusEstimator::filter_samples_for_estimate(
    const std::vector<Sample>& samples) const {
    if (samples.size() < static_cast<size_t>(kUmbilicusMinFilterSamples)) {
        return {};
    }

    std::vector<double> xs;
    std::vector<double> ys;
    xs.reserve(samples.size());
    ys.reserve(samples.size());
    for (const auto& s : samples) {
        xs.push_back(s.point_xy[0]);
        ys.push_back(s.point_xy[1]);
    }

    const double med_x = median_of(xs);
    const double med_y = median_of(ys);

    std::vector<double> dev_x;
    std::vector<double> dev_y;
    dev_x.reserve(samples.size());
    dev_y.reserve(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        dev_x.push_back(std::abs(xs[i] - med_x));
        dev_y.push_back(std::abs(ys[i] - med_y));
    }

    const double mad_x = median_of(dev_x);
    const double mad_y = median_of(dev_y);
    if (mad_x < kUmbilicusMinMad && mad_y < kUmbilicusMinMad) {
        return {};
    }

    const double thresh_x = std::max(mad_x, kUmbilicusMinMad) * kUmbilicusMadScale;
    const double thresh_y = std::max(mad_y, kUmbilicusMinMad) * kUmbilicusMadScale;

    std::vector<Sample> filtered;
    filtered.reserve(samples.size());
    for (const auto& s : samples) {
        if (std::abs(s.point_xy[0] - med_x) <= thresh_x &&
            std::abs(s.point_xy[1] - med_y) <= thresh_y) {
            filtered.push_back(s);
        }
    }

    if (filtered.size() < static_cast<size_t>(kUmbilicusMinFilteredSamples) ||
        filtered.size() == samples.size()) {
        return {};
    }

    return filtered;
}

bool UmbilicusEstimator::center_passes_sanity(const std::vector<Sample>& samples,
                                              const SampleStats& stats,
                                              const cv::Vec2d& center) const {
    if (!std::isfinite(center[0]) || !std::isfinite(center[1])) {
        return false;
    }
    if (samples.size() < static_cast<size_t>(kUmbilicusMinFilteredSamples)) {
        return true;
    }
    if (stats.median_radius < kUmbilicusMinMad) {
        return true;
    }

    const double center_offset = cv::norm(center - stats.median_xy);
    return center_offset <= kUmbilicusMaxCenterOffsetScale * stats.median_radius;
}

cv::Vec2d UmbilicusEstimator::ransac_find_center(const std::vector<Sample>& samples,
                                                 const SampleStats& stats) const {
    const int num_iterations = 500;
    const int samples_per_iter = std::min(500, static_cast<int>(samples.size()));

    cv::Vec2d best_center;
    double best_score = -std::numeric_limits<double>::infinity();

    // Find bounding box of samples
    cv::Vec2d min_pt = samples[0].point_xy, max_pt = samples[0].point_xy;
    for (const auto& s : samples) {
        min_pt[0] = std::min(min_pt[0], s.point_xy[0]);
        min_pt[1] = std::min(min_pt[1], s.point_xy[1]);
        max_pt[0] = std::max(max_pt[0], s.point_xy[0]);
        max_pt[1] = std::max(max_pt[1], s.point_xy[1]);
    }

    std::mt19937 rng(42);  // Deterministic
    std::uniform_real_distribution<double> dist_x(min_pt[0], max_pt[0]);
    std::uniform_real_distribution<double> dist_y(min_pt[1], max_pt[1]);
    std::uniform_int_distribution<int> sample_index(0, static_cast<int>(samples.size() - 1));

    for (int iter = 0; iter < num_iterations; iter++) {
        cv::Vec2d candidate(dist_x(rng), dist_y(rng));

        double score = 0.0;
        for (int i = 0; i < samples_per_iter; i++) {
            const auto& s = samples[sample_index(rng)];

            // Vector from sample point to candidate center
            cv::Vec2d to_center = candidate - s.point_xy;
            double dist = cv::norm(to_center);
            if (dist < 1e-6) continue;

            to_center /= dist;  // Normalize

            // Dot product: how well does normal point toward/away from center?
            double cos_angle = to_center[0] * s.normal_xy[0] + to_center[1] * s.normal_xy[1];

            // Weight by proximity (closer points matter more)
            double weight = 1.0 / std::max(100.0, dist);
            score += cos_angle * cos_angle * weight;
        }

        score *= center_offset_penalty(stats, candidate);
        if (score > best_score) {
            best_score = score;
            best_center = candidate;
        }
    }

    return refine_center(samples, stats, best_center);
}

cv::Vec2d UmbilicusEstimator::refine_center(const std::vector<Sample>& samples,
                                            const SampleStats& stats,
                                            cv::Vec2d initial) const {
    // Hill-climbing refinement (similar to normalgridtools.cpp)
    cv::Vec2d center = initial;
    double step_size = 512.0;
    double current_score = score_center(samples, stats, center);

    while (step_size >= 1.0) {
        bool improved = false;
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (dx == 0 && dy == 0) continue;
                cv::Vec2d candidate = center + cv::Vec2d(dx * step_size, dy * step_size);

                double score = score_center(samples, stats, candidate);

                if (score > current_score) {
                    center = candidate;
                    current_score = score;
                    improved = true;
                }
            }
        }
        if (!improved) step_size /= 2.0;
    }

    return center;
}

bool UmbilicusEstimator::estimate_center(int row, cv::Vec2d* center_out, double* mean_z_out) {
    if (row < 0 || row >= static_cast<int>(_samples_per_row.size())) {
        return false;
    }
    if (_center_valid[row]) {
        return false;
    }

    // Gather samples from this row and neighbors for more robust estimation
    const std::vector<Sample> samples = gather_samples_for_row(row);
    if (samples.empty()) {
        return false;
    }

    auto filtered_samples = filter_samples_for_estimate(samples);
    const auto& fit_samples = filtered_samples.empty() ? samples : filtered_samples;
    const SampleStats stats = compute_sample_stats(fit_samples);

    cv::Vec2d center = ransac_find_center(fit_samples, stats);
    double score = score_center(fit_samples, stats, center);

    // Compute mean Z for this row's samples
    double sum_z = 0.0;
    for (const auto& s : fit_samples) {
        sum_z += s.z;
    }
    double mean_z = sum_z / static_cast<double>(fit_samples.size());

    // Track attempt for retry logic
    _last_attempted_sample_count[row] = static_cast<int>(samples.size());

    bool sanity_ok = center_passes_sanity(fit_samples, stats, center);
    _last_attempted_score[row] = sanity_ok ? score : -1.0;

    if (!sanity_ok) {
        return false;
    }

    // Check score threshold
    if (score < kMinScoreThreshold) {
        return false;
    }

    _estimated_centers[row] = center;
    _estimated_z[row] = mean_z;
    _center_valid[row] = true;
    _estimated_scores[row] = score;

    // Update unified control points (will add new or replace if better score)
    add_or_update_control_point(center, mean_z, score);

    if (center_out) {
        *center_out = center;
    }
    if (mean_z_out) {
        *mean_z_out = mean_z;
    }

    return true;
}

bool UmbilicusEstimator::has_wrapped(int row) const {
    if (row < 0 || row >= static_cast<int>(_samples_per_row.size())) {
        return false;
    }
    const auto& samples = _samples_per_row[row];
    if (samples.size() < 8) {
        return false;
    }

    cv::Vec2d centroid(0, 0);
    for (const auto& s : samples) {
        centroid += s.point_xy;
    }
    centroid /= static_cast<double>(samples.size());

    uint8_t mask = 0;
    for (const auto& s : samples) {
        cv::Vec2d delta = s.point_xy - centroid;
        int quadrant = 0;
        if (delta[0] >= 0 && delta[1] >= 0) quadrant = 0;       // +X +Y
        else if (delta[0] < 0 && delta[1] >= 0) quadrant = 1;   // -X +Y
        else if (delta[0] < 0 && delta[1] < 0) quadrant = 2;    // -X -Y
        else quadrant = 3;                                      // +X -Y
        mask |= static_cast<uint8_t>(1 << quadrant);
        if (mask == 0x0F) {
            return true;
        }
    }

    return false;
}

bool UmbilicusEstimator::has_center(int row) const {
    if (row < 0 || row >= static_cast<int>(_center_valid.size())) {
        return false;
    }
    return _center_valid[row];
}

int UmbilicusEstimator::sample_count(int row) const {
    if (row < 0 || row >= static_cast<int>(_samples_per_row.size())) {
        return 0;
    }
    return static_cast<int>(_samples_per_row[row].size());
}

int UmbilicusEstimator::center_count() const {
    return static_cast<int>(std::count(_center_valid.begin(), _center_valid.end(), true));
}

core::util::Umbilicus UmbilicusEstimator::build_umbilicus(const cv::Vec3i& volume_shape) const {
    // Use score-aware filtering (prefers high-score anchors, nudges low-scored toward them)
    std::vector<cv::Vec3f> points = filter_control_points_by_score();

    if (points.empty()) {
        throw std::runtime_error("No valid umbilicus control points");
    }

    if (_log_generation < 0 || _log_generation % 1000 == 0) {
        std::cout << "[UmbilicusEstimator] Building umbilicus from " << points.size()
                  << " filtered control points (of " << _control_points.size() << " total)" << std::endl;
    }

    // Points are already sorted by Z from filter_control_points_by_score()
    return core::util::Umbilicus::FromPoints(std::move(points), volume_shape);
}

void UmbilicusEstimator::orient_normals_to_centroid() {
    // Compute centroid of all sampled surface points
    cv::Vec2d centroid(0, 0);
    int total_samples = 0;

    for (const auto& row_samples : _samples_per_row) {
        for (const auto& s : row_samples) {
            centroid += s.point_xy;
            total_samples++;
        }
    }

    if (total_samples < 10) return;
    centroid /= static_cast<double>(total_samples);

    // Count normals pointing toward vs away from centroid
    int toward_count = 0;
    int away_count = 0;

    for (const auto& row_samples : _samples_per_row) {
        for (const auto& s : row_samples) {
            cv::Vec2d to_centroid = centroid - s.point_xy;
            double dist = cv::norm(to_centroid);
            if (dist < 1e-6) continue;
            to_centroid /= dist;

            // Positive dot = normal points toward centroid
            double dot = to_centroid[0] * s.normal_xy[0] + to_centroid[1] * s.normal_xy[1];
            if (dot > 0) toward_count++;
            else away_count++;
        }
    }

    // Normals should point TOWARD the centroid (toward umbilicus)
    // If majority point away, flip all normals
    if (away_count > toward_count) {
        std::cout << "[UmbilicusEstimator] Flipping normals: " << away_count
                  << " away vs " << toward_count << " toward centroid" << std::endl;
        for (auto& row_samples : _samples_per_row) {
            for (auto& s : row_samples) {
                s.normal_xy = -s.normal_xy;
            }
        }
    }
}

int UmbilicusEstimator::retry_pending_rows() {
    int accepted_count = 0;

    for (int row = 0; row < static_cast<int>(_samples_per_row.size()); ++row) {
        // Skip if already accepted or exhausted
        if (_center_valid[row] || _retry_exhausted[row]) {
            continue;
        }

        // Skip if hasn't wrapped yet
        if (!has_wrapped(row)) {
            continue;
        }

        // Gather samples from this row and neighbors for more robust estimation
        const std::vector<Sample> samples = gather_samples_for_row(row);
        if (samples.empty()) {
            continue;
        }

        // Skip if no new samples since last attempt
        int current_sample_count = static_cast<int>(samples.size());
        if (current_sample_count <= _last_attempted_sample_count[row]) {
            continue;
        }

        // Re-estimate
        auto filtered_samples = filter_samples_for_estimate(samples);
        const auto& fit_samples = filtered_samples.empty() ? samples : filtered_samples;

        const SampleStats stats = compute_sample_stats(fit_samples);
        cv::Vec2d center = ransac_find_center(fit_samples, stats);
        double score = score_center(fit_samples, stats, center);

        double prev_score = _last_attempted_score[row];
        _last_attempted_sample_count[row] = current_sample_count;

        bool sanity_ok = center_passes_sanity(fit_samples, stats, center);
        _last_attempted_score[row] = sanity_ok ? score : -1.0;

        if (!sanity_ok) {
            continue;
        }

        // Check if improved
        if (score <= prev_score) {
            // No improvement - increment fail count
            _retry_fail_count[row]++;
            continue;
        }

        // Check threshold
        if (score < kMinScoreThreshold) {
            continue;
        }

        // Accept!
        double sum_z = 0.0;
        for (const auto& s : fit_samples) {
            sum_z += s.z;
        }
        double mean_z = sum_z / static_cast<double>(fit_samples.size());

        _estimated_centers[row] = center;
        _estimated_z[row] = mean_z;
        _center_valid[row] = true;
        _estimated_scores[row] = score;
        _retry_fail_count[row] = 0;  // Reset on success
        accepted_count++;

        // Update unified control points (will add new or replace if better score)
        add_or_update_control_point(center, mean_z, score);
    }

    return accepted_count;
}

double UmbilicusEstimator::get_score(int row) const {
    if (row < 0 || row >= static_cast<int>(_estimated_scores.size())) {
        return -1.0;
    }
    return _estimated_scores[row];
}

bool UmbilicusEstimator::has_any_points() const {
    return !_control_points.empty();
}

bool UmbilicusEstimator::load_json(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        // File doesn't exist - not an error, just nothing to load
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();

    // Simple JSON parsing for control_points array
    // Format: {"control_points": [{"x": ..., "y": ..., "z": ..., "score": ...}, ...]}
    size_t arr_start = content.find("\"control_points\"");
    if (arr_start == std::string::npos) {
        std::cerr << "[UmbilicusEstimator] No control_points in " << path << std::endl;
        return false;
    }

    arr_start = content.find('[', arr_start);
    if (arr_start == std::string::npos) {
        return false;
    }

    size_t arr_end = content.find(']', arr_start);
    if (arr_end == std::string::npos) {
        return false;
    }

    std::string arr_content = content.substr(arr_start + 1, arr_end - arr_start - 1);

    // Parse each object in the array into temporary storage
    std::vector<ControlPoint> raw_points;
    size_t pos = 0;
    while (pos < arr_content.size()) {
        size_t obj_start = arr_content.find('{', pos);
        if (obj_start == std::string::npos) break;

        size_t obj_end = arr_content.find('}', obj_start);
        if (obj_end == std::string::npos) break;

        std::string obj = arr_content.substr(obj_start, obj_end - obj_start + 1);

        // Extract x, y, z, score using simple parsing
        auto extract_value = [&obj](const std::string& key) -> double {
            size_t key_pos = obj.find("\"" + key + "\"");
            if (key_pos == std::string::npos) return std::numeric_limits<double>::quiet_NaN();
            size_t colon = obj.find(':', key_pos);
            if (colon == std::string::npos) return std::numeric_limits<double>::quiet_NaN();
            size_t val_start = colon + 1;
            while (val_start < obj.size() && (obj[val_start] == ' ' || obj[val_start] == '\t')) {
                val_start++;
            }
            return std::stod(obj.substr(val_start));
        };

        double x = extract_value("x");
        double y = extract_value("y");
        double z = extract_value("z");
        double score = extract_value("score");

        if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && std::isfinite(score)) {
            raw_points.push_back({cv::Vec2d(x, y), z, score});
        }

        pos = obj_end + 1;
    }

    if (raw_points.empty()) {
        return false;
    }

    // Group points by quantized Z (round to nearest grid spacing)
    // This handles legacy data that may have points at arbitrary Z values
    std::map<int, std::vector<ControlPoint>> z_groups;
    for (const auto& pt : raw_points) {
        int quant_z = static_cast<int>(std::round(pt.z / kSameZTolerance) * kSameZTolerance);
        z_groups[quant_z].push_back(pt);
    }

    // Merge each group: weighted average XY by score, keep max score
    int merged_count = 0;
    for (const auto& [quant_z, group] : z_groups) {
        if (group.size() == 1) {
            _control_points.push_back({group[0].xy, static_cast<double>(quant_z), group[0].score});
        } else {
            // Weighted average by score
            double total_weight = 0.0;
            cv::Vec2d weighted_xy(0.0, 0.0);
            double max_score = 0.0;
            for (const auto& pt : group) {
                weighted_xy += pt.xy * pt.score;
                total_weight += pt.score;
                max_score = std::max(max_score, pt.score);
            }
            if (total_weight > 0.0) {
                weighted_xy /= total_weight;
            }
            _control_points.push_back({weighted_xy, static_cast<double>(quant_z), max_score});
        }
        merged_count++;
    }

    std::cout << "[UmbilicusEstimator] Loaded " << raw_points.size()
              << " control points from " << path;
    if (merged_count != static_cast<int>(raw_points.size())) {
        std::cout << " (merged to " << merged_count << " grid points)";
    }
    std::cout << std::endl;

    return merged_count > 0;
}

void UmbilicusEstimator::add_or_update_control_point(const cv::Vec2d& xy, double z, double score) {
    // Quantize Z to 25-voxel grid
    double grid_z = std::round(z / kSameZTolerance) * kSameZTolerance;

    // Find existing point at this grid location
    for (auto& pt : _control_points) {
        if (std::abs(pt.z - grid_z) < 1.0) {  // Exact grid match (within rounding tolerance)
            if (score > pt.score) {
                if (_log_generation < 0 || _log_generation % 1000 == 0) {
                    std::cout << "[UmbilicusEstimator] Updating control point at z="
                              << std::fixed << std::setprecision(0) << pt.z
                              << " score " << std::setprecision(4) << pt.score
                              << " -> " << score << std::endl;
                }
                pt.xy = xy;
                pt.z = grid_z;
                pt.score = score;
            }
            return;  // Found match, done
        }
    }
    // No point at this grid location -> add new
    _control_points.push_back({xy, grid_z, score});
}

std::vector<cv::Vec3f> UmbilicusEstimator::filter_control_points_by_score() const {
    if (_control_points.empty()) {
        return {};
    }

    // Make a working copy sorted by Z
    std::vector<ControlPoint> sorted = _control_points;
    std::sort(sorted.begin(), sorted.end(),
              [](const ControlPoint& a, const ControlPoint& b) { return a.z < b.z; });

    // Separate into anchors (high score) and low-scored
    std::vector<const ControlPoint*> anchors;
    std::vector<const ControlPoint*> low_scored;

    for (const auto& pt : sorted) {
        if (pt.score >= kHighScoreThreshold) {
            anchors.push_back(&pt);
        } else {
            low_scored.push_back(&pt);
        }
    }

    // If no anchors, use all points as-is
    if (anchors.empty()) {
        std::vector<cv::Vec3f> result;
        result.reserve(sorted.size());
        for (const auto& pt : sorted) {
            result.push_back(cv::Vec3f(
                static_cast<float>(pt.xy[0]),
                static_cast<float>(pt.xy[1]),
                static_cast<float>(pt.z)));
        }
        return result;
    }

    // Build output starting with anchors
    std::vector<cv::Vec3f> result;
    result.reserve(sorted.size());
    for (const auto* anchor : anchors) {
        result.push_back(cv::Vec3f(
            static_cast<float>(anchor->xy[0]),
            static_cast<float>(anchor->xy[1]),
            static_cast<float>(anchor->z)));
    }

    // For each low-scored point, check if we need it
    for (const auto* low_pt : low_scored) {
        // Find nearest anchor by Z distance
        double nearest_dist = std::numeric_limits<double>::max();
        const ControlPoint* nearest_anchor = nullptr;

        for (const auto* anchor : anchors) {
            double dist = std::abs(anchor->z - low_pt->z);
            if (dist < nearest_dist) {
                nearest_dist = dist;
                nearest_anchor = anchor;
            }
        }

        // If no anchor within tolerance, include this low-scored point (nudged)
        if (nearest_dist > kLowScoreZTolerance) {
            // Nudge toward nearest anchor proportional to score
            // blend = 0 at kMinScoreThreshold, blend = 1 at kHighScoreThreshold
            double blend = (low_pt->score - kMinScoreThreshold) /
                           (kHighScoreThreshold - kMinScoreThreshold);
            blend = std::clamp(blend, 0.0, 1.0);

            cv::Vec2d nudged_xy = low_pt->xy * blend;
            if (nearest_anchor) {
                nudged_xy += nearest_anchor->xy * (1.0 - blend);
            } else {
                nudged_xy = low_pt->xy;  // No anchor to nudge toward
            }

            result.push_back(cv::Vec3f(
                static_cast<float>(nudged_xy[0]),
                static_cast<float>(nudged_xy[1]),
                static_cast<float>(low_pt->z)));
        }
        // else: skip - good anchor coverage exists
    }

    // Sort result by Z
    std::sort(result.begin(), result.end(),
              [](const cv::Vec3f& a, const cv::Vec3f& b) { return a[2] < b[2]; });

    return result;
}

std::vector<UmbilicusEstimator::Sample> UmbilicusEstimator::gather_samples_for_row(int row) const {
    std::vector<Sample> combined;
    const int num_rows = static_cast<int>(_samples_per_row.size());

    // Gather from row-1, row, row+1 (when they exist)
    for (int r = std::max(0, row - 1); r <= std::min(num_rows - 1, row + 1); ++r) {
        const auto& row_samples = _samples_per_row[r];
        combined.insert(combined.end(), row_samples.begin(), row_samples.end());
    }

    return combined;
}

bool UmbilicusEstimator::write_json(const std::string& path) const {
    // Build JSON manually (no external JSON library dependency)
    std::ostringstream json;
    json << std::fixed;

    json << "{\n";
    json << "  \"control_points\": [\n";

    // Sort control points by Z for consistent output
    std::vector<ControlPoint> sorted = _control_points;
    std::sort(sorted.begin(), sorted.end(),
              [](const ControlPoint& a, const ControlPoint& b) { return a.z < b.z; });

    bool first = true;
    for (const auto& pt : sorted) {
        if (!first) json << ",\n";
        first = false;

        json << "    {\"x\": " << static_cast<int>(std::round(pt.xy[0]))
             << ", \"y\": " << static_cast<int>(std::round(pt.xy[1]))
             << ", \"z\": " << static_cast<int>(std::round(pt.z))
             << ", \"score\": " << std::setprecision(4) << pt.score << "}";
    }

    json << "\n  ],\n";
    json << "  \"metadata\": {\n";
    json << "    \"z_grid_spacing\": " << static_cast<int>(kSameZTolerance) << ",\n";
    json << "    \"min_score_threshold\": " << std::setprecision(2) << kMinScoreThreshold << ",\n";
    json << "    \"high_score_threshold\": " << kHighScoreThreshold << ",\n";
    json << "    \"total_points\": " << _control_points.size() << ",\n";

    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    gmtime_r(&time_t_now, &tm_buf);
    char time_str[32];
    std::strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);

    json << "    \"timestamp\": \"" << time_str << "\"\n";
    json << "  }\n";
    json << "}\n";

    // Write to temp file then rename (atomic on POSIX)
    std::string tmp_path = path + ".tmp";
    std::ofstream out(tmp_path);
    if (!out) {
        std::cerr << "[UmbilicusEstimator] Failed to open " << tmp_path << " for writing" << std::endl;
        return false;
    }

    out << json.str();
    out.close();

    if (!out) {
        std::cerr << "[UmbilicusEstimator] Failed to write to " << tmp_path << std::endl;
        return false;
    }

    // Atomic rename
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::cerr << "[UmbilicusEstimator] Failed to rename " << tmp_path << " to " << path << std::endl;
        return false;
    }

    if (_log_generation < 0 || _log_generation % 1000 == 0) {
        std::cout << "[UmbilicusEstimator] Wrote umbilicus to " << path << std::endl;
    }
    return true;
}

} // namespace vc::wrap_tracking
