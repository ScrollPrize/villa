#include "vc/core/util/QuadSurface.hpp"
#include "vc/core/util/ArgParse.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void usage(const char* prog)
{
    std::cerr
        << "usage: " << prog << " --input <src tifxyz dir> [--output <dir>] [--start-winding 0]\n"
        << "  Cuts a rolled tifxyz surface into separate winding tifxyz folders.\n"
        << "  Outputs are named wNN_ddmmyyhhmm under <output dir>, or beside the\n"
        << "  source segment when <output dir> is omitted. Boundary columns are\n"
        << "  duplicated so adjacent windings touch but do not share quads.\n";
}

static std::string timestamp_ddmmyyhhmm()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    std::ostringstream out;
    out << std::put_time(&tm, "%d%m%y%H%M");
    return out.str();
}

static bool valid_point(const cv::Vec3f& p)
{
    return p[0] != -1.f && p[2] > 0.f;
}

static bool valid_col_range(const cv::Mat_<cv::Vec3f>& points, int& out_c0, int& out_c1);

struct GridPoint {
    int row = -1;
    int col = -1;
};

struct CutSelection {
    std::vector<int> cuts;
    std::string method;
    GridPoint seed;
    std::vector<int> hit_cols;
    GridPoint fallback_start;
    GridPoint fallback_end;
};

static bool middle_valid_row(const cv::Mat_<cv::Vec3f>& points, int col, int& out_row)
{
    std::vector<int> rows;
    rows.reserve(static_cast<size_t>(points.rows));
    for (int r = 0; r < points.rows; ++r) {
        if (valid_point(points(r, col))) {
            rows.push_back(r);
        }
    }

    if (rows.empty()) return false;
    out_row = rows[rows.size() / 2];
    return true;
}

static double point_distance(const cv::Vec3f& a, const cv::Vec3f& b)
{
    const double dx = double(a[0]) - double(b[0]);
    const double dy = double(a[1]) - double(b[1]);
    const double dz = double(a[2]) - double(b[2]);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

static std::vector<GridPoint> shortest_valid_path(
    const cv::Mat_<cv::Vec3f>& points,
    GridPoint start,
    GridPoint end)
{
    struct QueueItem {
        double dist = 0.0;
        int idx = -1;
    };
    struct QueueItemGreater {
        bool operator()(const QueueItem& a, const QueueItem& b) const
        {
            return a.dist > b.dist;
        }
    };

    const int rows = points.rows;
    const int cols = points.cols;
    const auto idx_of = [cols](int r, int c) { return r * cols + c; };
    const int start_idx = idx_of(start.row, start.col);
    const int end_idx = idx_of(end.row, end.col);
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);

    std::vector<double> dist(total, std::numeric_limits<double>::infinity());
    std::vector<int> prev(total, -1);
    std::priority_queue<QueueItem, std::vector<QueueItem>, QueueItemGreater> queue;

    dist[static_cast<size_t>(start_idx)] = 0.0;
    queue.push({0.0, start_idx});

    constexpr int dr[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    constexpr int dc[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

    while (!queue.empty()) {
        const QueueItem cur = queue.top();
        queue.pop();

        if (cur.dist != dist[static_cast<size_t>(cur.idx)]) continue;
        if (cur.idx == end_idx) break;

        const int r = cur.idx / cols;
        const int c = cur.idx % cols;
        const cv::Vec3f& p = points(r, c);
        for (int n = 0; n < 8; ++n) {
            const int nr = r + dr[n];
            const int nc = c + dc[n];
            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
            const cv::Vec3f& q = points(nr, nc);
            if (!valid_point(q)) continue;

            const int next_idx = idx_of(nr, nc);
            const double next_dist = cur.dist + point_distance(p, q);
            if (next_dist < dist[static_cast<size_t>(next_idx)]) {
                dist[static_cast<size_t>(next_idx)] = next_dist;
                prev[static_cast<size_t>(next_idx)] = cur.idx;
                queue.push({next_dist, next_idx});
            }
        }
    }

    if (!std::isfinite(dist[static_cast<size_t>(end_idx)])) {
        throw std::runtime_error("no valid shortest path from first valid column to last valid column");
    }

    std::vector<GridPoint> path;
    for (int idx = end_idx; idx >= 0; idx = prev[static_cast<size_t>(idx)]) {
        path.push_back({idx / cols, idx % cols});
        if (idx == start_idx) break;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

static std::vector<int> choose_shortest_path_cut_columns(
    const cv::Mat_<cv::Vec3f>& points,
    const cv::Vec2d& center,
    GridPoint& out_start,
    GridPoint& out_end)
{
    int valid_c0 = 0;
    int valid_c1 = points.cols - 1;
    if (!valid_col_range(points, valid_c0, valid_c1)) {
        throw std::runtime_error("no valid points in tifxyz");
    }

    if (!middle_valid_row(points, valid_c0, out_start.row) ||
        !middle_valid_row(points, valid_c1, out_end.row)) {
        throw std::runtime_error("failed to choose middle rows in first and last valid columns");
    }
    out_start.col = valid_c0;
    out_end.col = valid_c1;

    const std::vector<GridPoint> path = shortest_valid_path(points, out_start, out_end);
    if (path.size() < 3) return {};

    constexpr double pi = 3.1415926535897932384626433832795;
    constexpr double two_pi = 2.0 * pi;
    std::vector<double> theta;
    theta.reserve(path.size());
    for (const GridPoint& gp : path) {
        const cv::Vec3f& p = points(gp.row, gp.col);
        theta.push_back(std::atan2(double(p[1]) - center[1], double(p[0]) - center[0]));
    }

    for (size_t i = 1; i < theta.size(); ++i) {
        double delta = theta[i] - theta[i - 1];
        if (delta > pi) {
            theta[i] -= two_pi * std::ceil((delta - pi) / two_pi);
        } else if (delta < -pi) {
            theta[i] += two_pi * std::ceil((-pi - delta) / two_pi);
        }
    }

    const double total_turns = (theta.back() - theta.front()) / two_pi;
    const double sign = total_turns >= 0.0 ? 1.0 : -1.0;
    const int crossing_count = static_cast<int>(std::floor(std::abs(total_turns)));

    std::vector<int> cuts;
    cuts.reserve(static_cast<size_t>(crossing_count));
    for (int k = 1; k <= crossing_count; ++k) {
        const double target = theta.front() + sign * two_pi * double(k);
        for (size_t i = 1; i < path.size(); ++i) {
            const double prev = theta[i - 1];
            const double next = theta[i];
            const bool crosses = sign > 0.0
                ? prev <= target && target <= next
                : next <= target && target <= prev;
            if (!crosses || next == prev) continue;

            const double t = (target - prev) / (next - prev);
            const double col = double(path[i - 1].col) * (1.0 - t) + double(path[i].col) * t;
            cuts.push_back(std::clamp(static_cast<int>(std::lround(col)), 0, points.cols - 1));
            break;
        }
    }

    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    return cuts;
}

static int valid_count_in_col(const cv::Mat_<cv::Vec3f>& points, int col)
{
    int count = 0;
    for (int r = 0; r < points.rows; ++r) {
        if (valid_point(points(r, col))) ++count;
    }
    return count;
}

static bool nearest_valid_in_direction(
    const cv::Mat_<cv::Vec3f>& points,
    int row,
    int col,
    int dr,
    int dc,
    cv::Vec3d& out)
{
    constexpr int max_step = 30;
    for (int step = 1; step <= max_step; ++step) {
        const int r = row + dr * step;
        const int c = col + dc * step;
        if (r < 0 || r >= points.rows || c < 0 || c >= points.cols) continue;
        const cv::Vec3f& p = points(r, c);
        if (!valid_point(p)) continue;
        out = cv::Vec3d(double(p[0]), double(p[1]), double(p[2]));
        return true;
    }
    return false;
}

static bool normal_at_grid_point(
    const cv::Mat_<cv::Vec3f>& points,
    int row,
    int col,
    cv::Vec3d& out_normal)
{
    cv::Vec3d left;
    cv::Vec3d right;
    cv::Vec3d up;
    cv::Vec3d down;
    if (!nearest_valid_in_direction(points, row, col, 0, -1, left) ||
        !nearest_valid_in_direction(points, row, col, 0, 1, right) ||
        !nearest_valid_in_direction(points, row, col, -1, 0, up) ||
        !nearest_valid_in_direction(points, row, col, 1, 0, down)) {
        return false;
    }

    out_normal = (right - left).cross(down - up);
    const double len = cv::norm(out_normal);
    if (len == 0.0 || !std::isfinite(len)) return false;
    out_normal *= 1.0 / len;
    return true;
}

struct RayHit {
    int col = -1;
    int row = -1;
    double dist = std::numeric_limits<double>::infinity();
};

static std::vector<RayHit> nearest_columns_to_normal_ray(
    const cv::Mat_<cv::Vec3f>& points,
    GridPoint seed,
    const cv::Vec3d& normal,
    int valid_c0,
    int valid_c1)
{
    const cv::Vec3f& seed_f = points(seed.row, seed.col);
    const cv::Vec3d origin{double(seed_f[0]), double(seed_f[1]), double(seed_f[2])};
    constexpr double max_dist_to_ray = 18.0;

    std::vector<RayHit> raw_hits;
    raw_hits.reserve(static_cast<size_t>(valid_c1 - valid_c0 + 1));
    for (int c = valid_c0; c <= valid_c1; ++c) {
        RayHit best;
        best.col = c;
        for (int r = 0; r < points.rows; ++r) {
            const cv::Vec3f& p_f = points(r, c);
            if (!valid_point(p_f)) continue;

            const cv::Vec3d p{double(p_f[0]), double(p_f[1]), double(p_f[2])};
            const cv::Vec3d rel = p - origin;
            const cv::Vec3d closest = rel - normal * rel.dot(normal);
            const double dist = cv::norm(closest);
            if (dist < best.dist) {
                best.row = r;
                best.dist = dist;
            }
        }
        if (best.row >= 0 && best.dist <= max_dist_to_ray) {
            raw_hits.push_back(best);
        }
    }

    std::vector<RayHit> hits;
    for (const RayHit& hit : raw_hits) {
        if (!hits.empty() && hit.col <= hits.back().col + 2) {
            if (hit.dist < hits.back().dist) {
                hits.back() = hit;
            }
        } else {
            hits.push_back(hit);
        }
    }
    return hits;
}

static std::vector<int> best_consistent_hit_run(
    const std::vector<RayHit>& hits,
    int seed_col,
    int valid_c0,
    int valid_c1)
{
    if (hits.size() < 4) return {};

    const int span = std::max(1, valid_c1 - valid_c0);
    const int min_gap = std::max(20, span / 9);
    const int max_gap = std::max(min_gap + 1, span / 3);

    std::vector<int> best;
    bool best_contains_seed = false;
    double best_gap_error = std::numeric_limits<double>::infinity();

    for (size_t start = 0; start < hits.size(); ++start) {
        std::vector<int> run;
        run.push_back(hits[start].col);
        for (size_t i = start + 1; i < hits.size(); ++i) {
            const int gap = hits[i].col - run.back();
            if (gap < min_gap) continue;
            if (gap > max_gap) break;
            run.push_back(hits[i].col);
        }

        if (run.size() < 4) continue;

        const bool contains_seed = std::any_of(run.begin(), run.end(), [&](int col) {
            return std::abs(col - seed_col) <= 2;
        });

        double mean_gap = 0.0;
        for (size_t i = 1; i < run.size(); ++i) {
            mean_gap += double(run[i] - run[i - 1]);
        }
        mean_gap /= double(run.size() - 1);

        double gap_error = 0.0;
        for (size_t i = 1; i < run.size(); ++i) {
            const double diff = double(run[i] - run[i - 1]) - mean_gap;
            gap_error += diff * diff;
        }
        gap_error /= double(run.size() - 1);

        const bool better =
            (contains_seed != best_contains_seed && contains_seed) ||
            (contains_seed == best_contains_seed && run.size() > best.size()) ||
            (contains_seed == best_contains_seed && run.size() == best.size() && gap_error < best_gap_error);
        if (better) {
            best = std::move(run);
            best_contains_seed = contains_seed;
            best_gap_error = gap_error;
        }
    }

    return best;
}

static CutSelection choose_normal_ray_cut_columns(
    const cv::Mat_<cv::Vec3f>& points,
    const cv::Vec2d& center)
{
    int valid_c0 = 0;
    int valid_c1 = points.cols - 1;
    if (!valid_col_range(points, valid_c0, valid_c1)) {
        throw std::runtime_error("no valid points in tifxyz");
    }

    int seed_col = valid_c0;
    int best_count = -1;
    for (int c = valid_c0; c <= valid_c1; ++c) {
        const int count = valid_count_in_col(points, c);
        if (count > best_count) {
            best_count = count;
            seed_col = c;
        }
    }

    std::vector<int> seed_rows;
    seed_rows.reserve(static_cast<size_t>(best_count));
    for (int r = 0; r < points.rows; ++r) {
        if (valid_point(points(r, seed_col))) seed_rows.push_back(r);
    }

    std::vector<double> fractions = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    std::vector<int> best_run;
    GridPoint best_seed;
    for (double fraction : fractions) {
        if (seed_rows.empty()) break;
        const size_t idx = std::min(
            seed_rows.size() - 1,
            static_cast<size_t>(std::lround(fraction * double(seed_rows.size() - 1))));
        const GridPoint seed{seed_rows[idx], seed_col};

        cv::Vec3d normal;
        if (!normal_at_grid_point(points, seed.row, seed.col, normal)) continue;

        const std::vector<RayHit> hits =
            nearest_columns_to_normal_ray(points, seed, normal, valid_c0, valid_c1);
        std::vector<int> run = best_consistent_hit_run(hits, seed.col, valid_c0, valid_c1);
        if (run.size() > best_run.size()) {
            best_run = std::move(run);
            best_seed = seed;
        }
    }

    CutSelection selection;
    if (best_run.size() >= 4) {
        selection.seed = best_seed;
        selection.hit_cols = best_run;
        selection.cuts.assign(best_run.begin() + 1, best_run.end() - 1);
        selection.method = "normal-ray radial bands";
        return selection;
    }

    selection.cuts = choose_shortest_path_cut_columns(
        points,
        center,
        selection.fallback_start,
        selection.fallback_end);
    selection.method = "shortest-path revolutions fallback";
    return selection;
}

static bool has_valid_point(const cv::Mat_<cv::Vec3f>& points)
{
    for (int r = 0; r < points.rows; ++r) {
        for (int c = 0; c < points.cols; ++c) {
            if (valid_point(points(r, c))) return true;
        }
    }
    return false;
}

struct WindingSlice {
    int c0 = 0;
    int c1 = 0;
    double radius = std::numeric_limits<double>::infinity();
};

static double mean_slice_radius(
    const cv::Mat_<cv::Vec3f>& points,
    const cv::Vec2d& center,
    int c0,
    int c1)
{
    double sum = 0.0;
    int count = 0;
    c0 = std::clamp(c0, 0, points.cols - 1);
    c1 = std::clamp(c1, 0, points.cols - 1);

    for (int r = 0; r < points.rows; ++r) {
        for (int c = c0; c <= c1; ++c) {
            const cv::Vec3f& p = points(r, c);
            if (!valid_point(p)) continue;

            const double dx = double(p[0]) - center[0];
            const double dy = double(p[1]) - center[1];
            sum += std::sqrt(dx * dx + dy * dy);
            ++count;
        }
    }

    if (count == 0) {
        return std::numeric_limits<double>::infinity();
    }
    return sum / double(count);
}

static bool valid_col_range(const cv::Mat_<cv::Vec3f>& points, int& out_c0, int& out_c1)
{
    std::vector<int> valid_counts(static_cast<size_t>(points.cols), 0);
    for (int c = 0; c < points.cols; ++c) {
        for (int r = 0; r < points.rows; ++r) {
            if (valid_point(points(r, c))) {
                ++valid_counts[static_cast<size_t>(c)];
            }
        }
    }

    int best_c0 = -1;
    int best_c1 = -1;
    int cur_c0 = -1;
    for (int c = 0; c < points.cols; ++c) {
        if (valid_counts[static_cast<size_t>(c)] > 0) {
            if (cur_c0 < 0) cur_c0 = c;
        } else if (cur_c0 >= 0) {
            const int cur_c1 = c - 1;
            if (best_c0 < 0 || cur_c1 - cur_c0 > best_c1 - best_c0) {
                best_c0 = cur_c0;
                best_c1 = cur_c1;
            }
            cur_c0 = -1;
        }
    }
    if (cur_c0 >= 0) {
        const int cur_c1 = points.cols - 1;
        if (best_c0 < 0 || cur_c1 - cur_c0 > best_c1 - best_c0) {
            best_c0 = cur_c0;
            best_c1 = cur_c1;
        }
    }

    if (best_c0 < 0) return false;
    out_c0 = best_c0;
    out_c1 = best_c1;
    return true;
}

int main(int argc, char* argv[])
{
    vc::cli::ArgParser parser;
    parser.add_option("input", {"i"}, true, "Input tifxyz directory");
    parser.add_option("output", {"o"}, false, "Output directory");
    parser.add_option("start-winding", {"s"}, false, "First winding number to use in output names");
    parser.add_flag("help", {"h"}, "Show this help text");

    std::string parse_error;
    const vc::cli::ParsedArgs args = parser.parse(argc, argv, &parse_error);
    if (args.has("help")) {
        usage(argv[0]);
        std::cerr << parser.help_text("options:");
        return EXIT_SUCCESS;
    }

    if (!parse_error.empty()) {
        std::cerr << "error: " << parse_error << "\n";
        usage(argv[0]);
        std::cerr << parser.help_text("options:");
        return EXIT_FAILURE;
    }

    if (!args.positionals.empty()) {
        std::cerr << "error: positional arguments are not supported; use named options\n";
        usage(argv[0]);
        std::cerr << parser.help_text("options:");
        return EXIT_FAILURE;
    }

    const fs::path src = fs::absolute(fs::path(args.value("input"))).lexically_normal();
    fs::path out_root = src.parent_path();
    if (args.has("output")) {
        out_root = fs::absolute(fs::path(args.value("output"))).lexically_normal();
    }

    int start_winding = 0;
    if (args.has("start-winding")) {
        try {
            start_winding = std::stoi(args.value("start-winding"));
        } catch (const std::exception&) {
            std::cerr << "error: --start-winding must be an integer\n";
            return EXIT_FAILURE;
        }
        if (start_winding < 0) {
            std::cerr << "error: --start-winding must be nonnegative\n";
            return EXIT_FAILURE;
        }
    }

    std::unique_ptr<QuadSurface> surf;
    try {
        surf = load_quad_from_tifxyz(src);
    } catch (const std::exception& e) {
        std::cerr << "error loading " << src << ": " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    cv::Mat_<cv::Vec3f>* points = surf->rawPointsPtr();
    if (!points || points->empty()) {
        std::cerr << "error: empty tifxyz point grid\n";
        return EXIT_FAILURE;
    }

    const Rect3D bbox = surf->bbox();
    const cv::Vec2d center(
        (double(bbox.low[0]) + double(bbox.high[0])) * 0.5,
        (double(bbox.low[1]) + double(bbox.high[1])) * 0.5);

    CutSelection cut_selection;
    try {
        cut_selection = choose_normal_ray_cut_columns(*points, center);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::vector<int> cuts = cut_selection.cuts;
    if (cuts.empty()) {
        std::cerr << "error: no cut columns selected\n";
        return EXIT_FAILURE;
    }

    fs::create_directories(out_root);
    const std::string suffix = timestamp_ddmmyyhhmm();
    const cv::Vec2f scale = surf->scale();

    int valid_c0 = 0;
    int valid_c1 = points->cols - 1;
    if (!valid_col_range(*points, valid_c0, valid_c1)) {
        std::cerr << "error: no valid points in tifxyz\n";
        return EXIT_FAILURE;
    }

    cuts.erase(
        std::remove_if(cuts.begin(), cuts.end(), [&](int cut) {
            return cut <= valid_c0 || cut >= valid_c1;
        }),
        cuts.end());
    if (cuts.empty()) {
        std::cerr << "error: no cut columns inside main valid column run "
                  << valid_c0 << ".." << valid_c1 << "\n";
        return EXIT_FAILURE;
    }

    std::vector<WindingSlice> slices;
    slices.reserve(cuts.size() + 1);

    std::vector<int> starts;
    starts.reserve(cuts.size() + 1);
    std::vector<int> ends;
    ends.reserve(cuts.size() + 1);
    starts.push_back(valid_c0);
    for (int cut : cuts) {
        starts.push_back(std::clamp(cut, valid_c0, valid_c1));
        ends.push_back(std::clamp(cut, valid_c0, valid_c1));
    }
    ends.push_back(valid_c1);

    for (size_t i = 0; i < starts.size(); ++i) {
        const int c0 = starts[i];
        const int c1 = ends[i];
        if (c1 < c0) continue;
        slices.push_back({
            c0,
            c1,
            mean_slice_radius(*points, center, c0, c1)
        });
    }

    const bool number_right_to_left =
        !slices.empty() && slices.back().radius < slices.front().radius;

    if (cut_selection.method == "normal-ray radial bands") {
        std::cout << "normal ray seed: (" << cut_selection.seed.col << ","
                  << cut_selection.seed.row << ")\n";
        std::cout << "normal ray hit columns:";
        for (int col : cut_selection.hit_cols) std::cout << " " << col;
        std::cout << "\n";
    } else {
        std::cout << "shortest path: (" << cut_selection.fallback_start.col << ","
                  << cut_selection.fallback_start.row << ") -> ("
                  << cut_selection.fallback_end.col << ","
                  << cut_selection.fallback_end.row << ")\n";
    }
    std::cout << "cut method: " << cut_selection.method << "\n";
    std::cout << "cut columns:";
    for (int cut : cuts) std::cout << " " << cut;
    std::cout << "\n";
    std::cout << "output order: "
              << (number_right_to_left ? "right-to-left" : "left-to-right")
              << " (smaller radius side first)\n";

    int written = 0;
    for (size_t i = 0; i < slices.size(); ++i) {
        const WindingSlice& slice = number_right_to_left
            ? slices[slices.size() - 1 - i]
            : slices[i];
        const int c0 = slice.c0;
        const int c1 = slice.c1;
        const cv::Rect rect(c0, 0, c1 - c0 + 1, points->rows);
        cv::Mat_<cv::Vec3f> crop = (*points)(rect).clone();
        if (!has_valid_point(crop)) {
            std::cerr << "warning: skipping empty winding at cols " << c0 << ".." << c1 << "\n";
            continue;
        }

        std::ostringstream name;
        name << "w" << std::setw(2) << std::setfill('0') << (start_winding + written) << "_" << suffix;
        const fs::path out_path = out_root / name.str();

        try {
            QuadSurface out(crop, scale);
            out.save(out_path.string(), name.str(), false);
        } catch (const std::exception& e) {
            std::cerr << "error writing " << out_path << ": " << e.what() << "\n";
            return EXIT_FAILURE;
        }

        std::cout << name.str() << ": cols " << c0 << ".." << c1
                  << " mean_radius " << slice.radius
                  << " -> " << out_path << "\n";
        ++written;
    }

    std::cout << "wrote " << written << " winding tifxyz directories\n";
    return EXIT_SUCCESS;
}
