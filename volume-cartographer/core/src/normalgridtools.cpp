#include "vc/core/util/GridStore.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <optional>
#include <cstdint>
#include <iostream>
#include <opencv2/imgproc.hpp>

#include "vc/core/util/normalgridtools.hpp"

namespace vc::core::util {

cv::Vec2f align_and_extract_umbilicus(const GridStore& grid_store, std::optional<std::uint32_t> seed,
                                      float uncrush) {
    auto segments = grid_store.get_all();
    if (segments.empty()) {
        return cv::Vec2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    }

    // 1. Define RANSAC parameters
    const int num_ransac_iterations = 1000;
    const int num_samples_per_iteration = 10000;

    // 2. Determine grid extents for sampling umbilicus candidates
    cv::Size grid_size = grid_store.size();
    float min_x = 0;
    float max_x = grid_size.width;
    float min_y = 0;
    float max_y = grid_size.height;

    // 3. Create a global list of all line segments
    std::vector<std::pair<cv::Point, cv::Point>> all_line_segments;
    for (const auto& path : segments) {
        if (path->size() < 2) continue;
        for (size_t i = 0; i < path->size() - 1; ++i) {
            all_line_segments.push_back(std::make_pair((*path)[i], (*path)[i+1]));
        }
    }

    if (all_line_segments.empty()) {
        return cv::Vec2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
    }

    // 3b. Optional un-crush. The score below looks for the point the sheet normals point at,
    // which is the centre of round windings but not of crushed ones: the normals of an ellipse
    // do not point at its centre. With uncrush = beta > 0 the section is first stretched back
    // towards round, by the anisotropy of its normals and not of its extent. On a crushed section
    // the normals crowd across its flat sides, so their axial orientation tensor
    // T = sum w n n^T is long in that direction, and x' = W (x - mu) with
    // W = diag((t_i / t_mean)^(beta / 2)) V^T stretches the section along it. The normals are
    // carried by the inverse transpose of W, the search runs in that frame over the bounds of the
    // transformed midpoints, and its result is mapped back with W^-1. uncrush = 0 is the
    // estimator unchanged, draw for draw.
    //
    // T counts segments, so uneven sampling would look like crushing: a round half section with
    // ten times more segments in one quadrant reads 3.2:1. Each segment is therefore weighted by
    // one over the number of segments in its cell of a 32 x 32 grid over the section, so that
    // every part of the section counts once however densely it was sampled; that section then
    // reads 1.02:1, and a 2.5:1 ellipse 2.6:1. A section
    // that spans less than half a turn leaves some normal directions empty, and its T is long
    // even when its windings are round, so the step is skipped, and the estimate is the one with
    // uncrush = 0, when the emptiest of 12 direction bins holds under 1 % of an even share.
    // The step is also skipped when T is nearly round (l1 / l2 under 1.1): the stretch would be
    // a few per cent, but the new search frame alone (recentred, rotated, new bounds) gives
    // RANSAC other candidates and can move the estimate to a neighbouring maximum.
    double w00 = 1.0, w01 = 0.0, w10 = 0.0, w11 = 1.0;   // W, row major
    double mu_x = 0.0, mu_y = 0.0;
    bool whiten = false;
    if (uncrush > 0.0f) {
        const double n = (double)all_line_segments.size();
        double lo_x = std::numeric_limits<double>::max(), lo_y = lo_x;
        double hi_x = std::numeric_limits<double>::lowest(), hi_y = hi_x;
        for (const auto& s : all_line_segments) {
            const double qx = 0.5 * (s.first.x + s.second.x), qy = 0.5 * (s.first.y + s.second.y);
            mu_x += qx;
            mu_y += qy;
            lo_x = std::min(lo_x, qx); hi_x = std::max(hi_x, qx);
            lo_y = std::min(lo_y, qy); hi_y = std::max(hi_y, qy);
        }
        mu_x /= n;
        mu_y /= n;
        constexpr int kCells = 32, kBins = 12;
        constexpr double kPi = 3.14159265358979323846;
        const double cell = std::max(1.0, std::max(hi_x - lo_x, hi_y - lo_y) / kCells);
        auto cell_of = [&](const std::pair<cv::Point, cv::Point>& s) {
            const int cx = std::min(kCells, (int)((0.5 * (s.first.x + s.second.x) - lo_x) / cell));
            const int cy = std::min(kCells, (int)((0.5 * (s.first.y + s.second.y) - lo_y) / cell));
            return static_cast<size_t>(cy * (kCells + 1) + cx);
        };
        std::vector<int> count((kCells + 1) * (kCells + 1), 0);
        for (const auto& s : all_line_segments) ++count[cell_of(s)];
        double txx = 0.0, txy = 0.0, tyy = 0.0, bins[kBins] = {};
        for (const auto& s : all_line_segments) {
            const double tx = s.second.x - s.first.x, ty = s.second.y - s.first.y;
            const double len = std::hypot(tx, ty);
            if (len <= 0.0) continue;
            const double nx = -ty / len, ny = tx / len;
            const double w = 1.0 / count[cell_of(s)];
            txx += w * nx * nx;
            txy += w * nx * ny;
            tyy += w * ny * ny;
            double dir = std::atan2(ny, nx);   // the normal's axial direction, in [0, pi)
            if (dir < 0.0) dir += kPi;
            bins[std::min(kBins - 1, (int)(dir / (kPi / kBins)))] += w;
        }
        double total = 0.0;
        for (const double b : bins) total += b;
        const bool spans_half_turn = total > 0.0 && *std::min_element(bins, bins + kBins) * kBins >= 0.01 * total;
        // Eigen decomposition of the symmetric 2 x 2 tensor, in closed form. Only the ratios of
        // the eigenvalues are used, so it is not divided by the number of segments.
        const double half_trace = 0.5 * (txx + tyy);
        const double r = std::hypot(0.5 * (txx - tyy), txy);
        const double l1 = half_trace + r, l2 = half_trace - r;
        constexpr double kRound = 1.1;
        if (spans_half_turn && l2 > 0.0 && l1 >= kRound * l2) {
            const double t = 0.5 * std::atan2(2.0 * txy, txx - tyy);   // where the normals crowd
            const double f1 = std::pow(l1 / half_trace, 0.5 * uncrush);
            const double f2 = std::pow(l2 / half_trace, 0.5 * uncrush);
            w00 = f1 * std::cos(t);  w01 = f1 * std::sin(t);
            w10 = -f2 * std::sin(t); w11 = f2 * std::cos(t);
            whiten = true;
            min_x = min_y = std::numeric_limits<float>::max();
            max_x = max_y = std::numeric_limits<float>::lowest();
            for (const auto& s : all_line_segments) {
                const double dx = 0.5 * (s.first.x + s.second.x) - mu_x;
                const double dy = 0.5 * (s.first.y + s.second.y) - mu_y;
                const float wx = (float)(w00 * dx + w01 * dy), wy = (float)(w10 * dx + w11 * dy);
                min_x = std::min(min_x, wx); max_x = std::max(max_x, wx);
                min_y = std::min(min_y, wy); max_y = std::max(max_y, wy);
            }
        }
    }
    const double det = w00 * w11 - w01 * w10;

    // 4. Pre-sample points and calculate normals for consistent scoring
    std::vector<cv::Point2f> sample_points;
    std::vector<cv::Vec2f> sample_normals;
    sample_points.reserve(num_samples_per_iteration);
    sample_normals.reserve(num_samples_per_iteration);

    // The generator. With no seed it is the one static generator seeded once from the hardware,
    // as published, so two calls in one process draw different samples. With a seed it is a
    // fresh generator seeded with that value, so that a run can be repeated exactly.
    static std::mt19937 shared_gen(std::random_device{}());
    std::mt19937 seeded_gen;
    if (seed) seeded_gen.seed(*seed);
    std::mt19937& gen = seed ? seeded_gen : shared_gen;
    std::uniform_int_distribution<> segment_dist(0, all_line_segments.size() - 1);

    for (int j = 0; j < num_samples_per_iteration; ++j) {
        const auto& line_segment = all_line_segments[segment_dist(gen)];
        const auto& p1 = line_segment.first;
        const auto& p2 = line_segment.second;

        sample_points.emplace_back((float)(p1.x + p2.x) / 2.0f, (float)(p1.y + p2.y) / 2.0f);

        cv::Vec2f tangent((float)(p2.x - p1.x), (float)(p2.y - p1.y));
        cv::normalize(tangent, tangent);
        sample_normals.emplace_back(-tangent[1], tangent[0]); // Perpendicular

        if (whiten) {
            const double dx = sample_points.back().x - mu_x, dy = sample_points.back().y - mu_y;
            sample_points.back() = cv::Point2f((float)(w00 * dx + w01 * dy), (float)(w10 * dx + w11 * dy));
            const cv::Vec2f nrm = sample_normals.back();   // a normal goes through W^-T
            cv::Vec2f wn((float)((w11 * nrm[0] - w10 * nrm[1]) / det),
                         (float)((-w01 * nrm[0] + w00 * nrm[1]) / det));
            cv::normalize(wn, wn);
            sample_normals.back() = wn;
        }
    }

    // 4. RANSAC loop
    cv::Vec2f best_umbilicus(0, 0);
    double best_score = -1.0;

    std::uniform_real_distribution<float> x_dist(min_x, max_x);
    std::uniform_real_distribution<float> y_dist(min_y, max_y);

    for (int i = 0; i < num_ransac_iterations; ++i) {
        // a. Sample a candidate umbilicus point
        cv::Vec2f candidate_umbilicus(x_dist(gen), y_dist(gen));

        // b. Score the candidate
        double current_score = 0.0;
        for (size_t j = 0; j < sample_points.size(); ++j) {
            const auto& point = sample_points[j];
            const auto& normal = sample_normals[j];

            cv::Vec2f umbilicus_to_segment = cv::Vec2f(point) - candidate_umbilicus;
            cv::normalize(umbilicus_to_segment, umbilicus_to_segment);

            double cos_angle = umbilicus_to_segment.dot(normal);
            current_score += cos_angle * cos_angle;
        }

        // c. Update best estimate
        if (current_score > best_score) {
            best_score = current_score;
            best_umbilicus = candidate_umbilicus;
        }
    }

    // 5. Refine the best estimate using a hill-climbing direct search.
    auto score_candidate = [&](const cv::Vec2f& candidate) {
        double score = 0.0;
        for (size_t j = 0; j < sample_points.size(); ++j) {
            const auto& point = sample_points[j];
            const auto& normal = sample_normals[j];

            cv::Vec2f umbilicus_to_segment = cv::Vec2f(point) - candidate;
            float dist = cv::norm(umbilicus_to_segment);
            if (dist < 1e-6) continue;

            umbilicus_to_segment /= dist; // Manual normalization

            double cos_angle = umbilicus_to_segment.dot(normal);
            float weight = 1.0f / std::max(100.0f, dist);
            score += (cos_angle * cos_angle) * weight;
        }
        // Weighted sum, not weighted mean. The sum weights the alignment of each normal by the
        // proximity of its segment, so it favours candidates near many segments as well as
        // candidates the normals point at; it vanishes far from the section and so has a maximum
        // at finite distance, which the mean does not on partial predictions. That maximum is a
        // property of the score and not a guarantee about the search below it: nothing bounds the
        // number of steps the climb takes or keeps it in the grid, and that it settled inside the
        // grid on 576 of 576 measured slices is observed and not enforced. On synthetic
        // sections with a known centre the mean is exact and the sum is biased towards the denser
        // side by 0.4 to 2.4 mm (evidence/synthetic-density.csv of the validation paper). On
        // fifteen real scrolls with a manual reference the sum's median error is 0.4 to 7.1 mm
        // against 0.4 to 133 mm for the mean, with no failure above 10 mm against seven scrolls,
        // and the failure persists with the walk clipped to the grid: it is the objective and
        // not the search.
        return score;
    };

    double refined_best_score = score_candidate(best_umbilicus);
    float step_size = 1024.0f;

    while (step_size >= 1.0f) {
        bool moved = false;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                if (dx == 0 && dy == 0) continue;

                cv::Vec2f candidate = best_umbilicus + cv::Vec2f(dx * step_size, dy * step_size);
                double candidate_score = score_candidate(candidate);

                if (candidate_score > refined_best_score) {
                    refined_best_score = candidate_score;
                    best_umbilicus = candidate;
                    moved = true;
                }
            }
        }

        if (!moved) {
            step_size /= 2.0f;
        }
    }

    if (whiten) {   // back from the un-crushed frame, x = W^-1 x' + mu
        const double ux = best_umbilicus[0], uy = best_umbilicus[1];
        best_umbilicus = cv::Vec2f((float)((w11 * ux - w01 * uy) / det + mu_x),
                                   (float)((-w10 * ux + w00 * uy) / det + mu_y));
    }

    std::cout << "Refined umbilicus estimate: " << best_umbilicus << " with score " << refined_best_score << std::endl;

    return best_umbilicus;
}

namespace {

double tricube(double u)
{
    u = std::min(std::abs(u), 1.0);
    const double t = 1.0 - u * u * u;
    return t * t * t;
}

// Weighted least squares line through (z, x) and (z, y), read at z0, over the estimates (sorted by
// z) that lie within half_width of z0. False when fewer than 3 of them carry weight. A window
// whose estimates all share one z gives their weighted mean.
bool local_linear(const std::vector<cv::Vec3d>& e, const std::vector<double>& w, double z0,
                  double half_width, cv::Vec2d& out)
{
    const auto byZ = [](const cv::Vec3d& a, double z) { return a[0] < z; };
    const size_t lo = std::lower_bound(e.begin(), e.end(), z0 - half_width, byZ) - e.begin();
    double sw = 0.0, swz = 0.0, swx = 0.0, swy = 0.0;
    int count = 0;
    size_t hi = lo;
    for (; hi < e.size() && e[hi][0] <= z0 + half_width; ++hi) {
        const double k = tricube((e[hi][0] - z0) / half_width) * w[hi];
        if (k > 0.0) {
            ++count;
        }
        sw += k;
        swz += k * e[hi][0];
        swx += k * e[hi][1];
        swy += k * e[hi][2];
    }
    if (count < 3 || sw <= 0.0) {
        return false;
    }
    const double zm = swz / sw, xm = swx / sw, ym = swy / sw;
    double szz = 0.0, szx = 0.0, szy = 0.0;
    for (size_t i = lo; i < hi; ++i) {
        const double k = tricube((e[i][0] - z0) / half_width) * w[i];
        const double dz = e[i][0] - zm;
        szz += k * dz * dz;
        szx += k * dz * (e[i][1] - xm);
        szy += k * dz * (e[i][2] - ym);
    }
    if (szz <= 1e-12 * std::max(1.0, sw)) {
        out = cv::Vec2d(xm, ym);
    } else {
        out = cv::Vec2d(xm + szx / szz * (z0 - zm), ym + szy / szz * (z0 - zm));
    }
    return true;
}

double median_of(std::vector<double> v)
{
    const size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    const double hi = v[n / 2];
    if (n % 2 == 1) {
        return hi;
    }
    return 0.5 * (hi + *std::max_element(v.begin(), v.begin() + n / 2));
}

} // namespace

std::vector<cv::Vec2d> fit_umbilicus_along_z(const std::vector<cv::Vec3d>& estimates,
                                             const std::vector<double>& query_z,
                                             double half_width,
                                             int robustness_iterations)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<cv::Vec2d> result(query_z.size(), cv::Vec2d(nan, nan));
    if (estimates.empty() || !(half_width > 0.0)) {
        return result;
    }

    std::vector<cv::Vec3d> sorted = estimates;
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const cv::Vec3d& a, const cv::Vec3d& b) { return a[0] < b[0]; });

    std::vector<double> robust(sorted.size(), 1.0);
    for (int it = 0; it < robustness_iterations; ++it) {
        std::vector<double> residual(sorted.size(), 0.0);
        for (size_t i = 0; i < sorted.size(); ++i) {
            cv::Vec2d fit;
            if (local_linear(sorted, robust, sorted[i][0], half_width, fit)) {
                residual[i] = std::hypot(sorted[i][1] - fit[0], sorted[i][2] - fit[1]);
            }
        }
        const double s = median_of(residual);
        if (!(s > 0.0)) {
            break;
        }
        for (size_t i = 0; i < sorted.size(); ++i) {
            const double u = std::min(residual[i] / (6.0 * s), 1.0);
            robust[i] = (1.0 - u * u) * (1.0 - u * u);
        }
    }

    for (size_t q = 0; q < query_z.size(); ++q) {
        if (local_linear(sorted, robust, query_z[q], half_width, result[q])) {
            continue;
        }
        std::vector<double> xs, ys;
        for (const auto& e : sorted) {
            if (e[0] == query_z[q]) {
                xs.push_back(e[1]);
                ys.push_back(e[2]);
            }
        }
        if (!xs.empty()) {
            result[q] = cv::Vec2d(median_of(xs), median_of(ys));
        }
    }
    return result;
}

void populate_normal_grid(const vcr::thinning::Traces& traces, GridStore& normal_grid, double spiral_step) {
    const float target_dist_sq = static_cast<float>(spiral_step * spiral_step);

    for (const auto& trace : traces) {
        if (trace.size() < 2) continue;

        std::vector<cv::Point> resampled_path;
        resampled_path.emplace_back(trace[0].x, trace[0].y);
        cv::Point last_point(trace[0].x, trace[0].y);

        for (size_t i = 1; i < trace.size(); ++i) {
            cv::Point current_point(trace[i].x, trace[i].y);
            auto v = current_point - last_point;
            double dist_sq = v.ddot(v);
            if (dist_sq >= target_dist_sq) {
                resampled_path.push_back(current_point);
                last_point = current_point;
            }
        }

        if (resampled_path.size() >= 2) {
            normal_grid.add(resampled_path);
        }
    }
}


}
