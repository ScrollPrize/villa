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
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct RowCrossings {
    int row = -1;
    std::vector<double> cols;
};

static void usage(const char* prog)
{
    std::cerr
        << "usage: " << prog << " --input <src tifxyz dir> [--output <dir>] [--band-height 8] [--start-winding 0]\n"
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

static bool longest_valid_segment(const cv::Mat_<cv::Vec3f>& points, int row, int& out_c0, int& out_c1)
{
    int best_c0 = -1;
    int best_c1 = -1;
    int cur_c0 = -1;

    for (int c = 0; c < points.cols; ++c) {
        if (valid_point(points(row, c))) {
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

static RowCrossings detect_row_crossings(
    const cv::Mat_<cv::Vec3f>& points,
    int row,
    double center_x,
    double center_y)
{
    RowCrossings out;
    out.row = row;

    int c0 = -1;
    int c1 = -1;
    if (!longest_valid_segment(points, row, c0, c1) || c1 - c0 < 2) {
        return out;
    }

    std::vector<double> theta;
    theta.reserve(static_cast<size_t>(c1 - c0 + 1));
    for (int c = c0; c <= c1; ++c) {
        const cv::Vec3f& p = points(row, c);
        theta.push_back(std::atan2(double(p[1]) - center_y, double(p[0]) - center_x));
    }

    constexpr double pi = 3.1415926535897932384626433832795;
    constexpr double two_pi = 2.0 * pi;
    for (size_t i = 1; i < theta.size(); ++i) {
        double delta = theta[i] - theta[i - 1];
        if (delta > pi) {
            theta[i] -= two_pi * std::ceil((delta - pi) / two_pi);
        } else if (delta < -pi) {
            theta[i] += two_pi * std::ceil((-pi - delta) / two_pi);
        }
    }

    const double sign = (theta.back() >= theta.front()) ? 1.0 : -1.0;
    std::vector<double> turns(theta.size());
    for (size_t i = 0; i < theta.size(); ++i) {
        turns[i] = sign * (theta[i] - theta.front()) / two_pi;
    }

    const int crossing_count = static_cast<int>(std::floor(turns.back()));
    for (int k = 1; k <= crossing_count; ++k) {
        const auto it = std::lower_bound(turns.begin(), turns.end(), double(k));
        if (it == turns.begin() || it == turns.end()) continue;

        const size_t idx = static_cast<size_t>(it - turns.begin());
        const double prev = turns[idx - 1];
        const double next = turns[idx];
        if (next == prev) continue;

        const double t = (double(k) - prev) / (next - prev);
        const double col = double(c0 + int(idx) - 1) * (1.0 - t) + double(c0 + int(idx)) * t;
        out.cols.push_back(col);
    }

    return out;
}

static std::vector<int> choose_cut_columns(
    const cv::Mat_<cv::Vec3f>& points,
    const cv::Vec2d& center,
    int band_height,
    int& out_band_start,
    int& out_band_end)
{
    std::vector<RowCrossings> rows;
    rows.reserve(static_cast<size_t>(points.rows));
    int max_crossings = 0;
    for (int r = 0; r < points.rows; ++r) {
        RowCrossings crossings = detect_row_crossings(points, r, center[0], center[1]);
        max_crossings = std::max<int>(max_crossings, crossings.cols.size());
        rows.push_back(std::move(crossings));
    }

    if (max_crossings <= 0) {
        throw std::runtime_error("no winding crossings detected");
    }

    int best_start = -1;
    int best_end = -1;
    int cur_start = -1;
    for (int r = 0; r < points.rows; ++r) {
        if (static_cast<int>(rows[r].cols.size()) == max_crossings) {
            if (cur_start < 0) cur_start = r;
        } else if (cur_start >= 0) {
            const int cur_end = r - 1;
            if (best_start < 0 || cur_end - cur_start > best_end - best_start) {
                best_start = cur_start;
                best_end = cur_end;
            }
            cur_start = -1;
        }
    }
    if (cur_start >= 0) {
        const int cur_end = points.rows - 1;
        if (best_start < 0 || cur_end - cur_start > best_end - best_start) {
            best_start = cur_start;
            best_end = cur_end;
        }
    }

    if (best_start < 0) {
        throw std::runtime_error("failed to locate a full winding row band");
    }

    const int full_run_height = best_end - best_start + 1;
    if (band_height > full_run_height) {
        std::cerr << "warning: requested band height " << band_height
                  << " exceeds full-width row run height " << full_run_height
                  << "; clamping to " << full_run_height << "\n";
        band_height = full_run_height;
    }
    band_height = std::max(1, band_height);
    const int mid = (best_start + best_end) / 2;
    out_band_start = std::clamp(mid - band_height / 2, best_start, best_end - band_height + 1);
    out_band_end = out_band_start + band_height - 1;

    std::vector<int> cuts;
    cuts.reserve(static_cast<size_t>(max_crossings));
    for (int k = 0; k < max_crossings; ++k) {
        std::vector<double> vals;
        vals.reserve(static_cast<size_t>(band_height));
        for (int r = out_band_start; r <= out_band_end; ++r) {
            vals.push_back(rows[r].cols[static_cast<size_t>(k)]);
        }
        std::sort(vals.begin(), vals.end());
        const double med = vals[vals.size() / 2];
        const int cut = static_cast<int>(std::lround(med));
        if (cuts.empty() || cut > cuts.back()) {
            cuts.push_back(cut);
        }
    }

    return cuts;
}

static std::vector<int> choose_centerline_cut_columns(
    const cv::Mat_<cv::Vec3f>& points,
    const cv::Vec2d& center)
{
    std::vector<int> cols;
    std::vector<double> theta;
    cols.reserve(static_cast<size_t>(points.cols));
    theta.reserve(static_cast<size_t>(points.cols));

    for (int c = 0; c < points.cols; ++c) {
        std::vector<int> rows;
        rows.reserve(static_cast<size_t>(points.rows));
        for (int r = 0; r < points.rows; ++r) {
            if (valid_point(points(r, c))) {
                rows.push_back(r);
            }
        }
        if (rows.size() < 3) {
            continue;
        }

        const int r = rows[rows.size() / 2];
        const cv::Vec3f& p = points(r, c);
        cols.push_back(c);
        theta.push_back(std::atan2(double(p[1]) - center[1], double(p[0]) - center[0]));
    }

    if (cols.size() < 3) {
        return {};
    }

    constexpr double pi = 3.1415926535897932384626433832795;
    constexpr double two_pi = 2.0 * pi;
    for (size_t i = 1; i < theta.size(); ++i) {
        double delta = theta[i] - theta[i - 1];
        if (delta > pi) {
            theta[i] -= two_pi * std::ceil((delta - pi) / two_pi);
        } else if (delta < -pi) {
            theta[i] += two_pi * std::ceil((-pi - delta) / two_pi);
        }
    }

    const double sign = (theta.back() >= theta.front()) ? 1.0 : -1.0;
    std::vector<double> turns(theta.size());
    for (size_t i = 0; i < theta.size(); ++i) {
        turns[i] = sign * (theta[i] - theta.front()) / two_pi;
    }

    std::vector<int> cuts;
    const int crossing_count = static_cast<int>(std::floor(turns.back()));
    for (int k = 1; k <= crossing_count; ++k) {
        const auto it = std::lower_bound(turns.begin(), turns.end(), double(k));
        if (it == turns.begin() || it == turns.end()) continue;

        const size_t idx = static_cast<size_t>(it - turns.begin());
        const double prev = turns[idx - 1];
        const double next = turns[idx];
        if (next == prev) continue;

        const double t = (double(k) - prev) / (next - prev);
        const double col = double(cols[idx - 1]) * (1.0 - t) + double(cols[idx]) * t;
        const int cut = std::clamp(static_cast<int>(std::lround(col)), 0, points.cols - 1);
        if (cuts.empty() || cut > cuts.back()) {
            cuts.push_back(cut);
        }
    }

    return cuts;
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

static double mean_band_radius(
    const cv::Mat_<cv::Vec3f>& points,
    const cv::Vec2d& center,
    int row0,
    int row1,
    int c0,
    int c1)
{
    double sum = 0.0;
    int count = 0;
    row0 = std::clamp(row0, 0, points.rows - 1);
    row1 = std::clamp(row1, 0, points.rows - 1);
    c0 = std::clamp(c0, 0, points.cols - 1);
    c1 = std::clamp(c1, 0, points.cols - 1);

    for (int r = row0; r <= row1; ++r) {
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
    out_c0 = points.cols;
    out_c1 = -1;
    for (int c = 0; c < points.cols; ++c) {
        bool any = false;
        for (int r = 0; r < points.rows; ++r) {
            if (valid_point(points(r, c))) {
                any = true;
                break;
            }
        }
        if (any) {
            out_c0 = std::min(out_c0, c);
            out_c1 = c;
        }
    }
    return out_c1 >= out_c0;
}

int main(int argc, char* argv[])
{
    vc::cli::ArgParser parser;
    parser.add_option("input", {"i"}, true, "Input tifxyz directory");
    parser.add_option("output", {"o"}, false, "Output directory");
    parser.add_option("band-height", {"b"}, false, "Row band height used to estimate cut columns");
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

    int band_height = 8;
    if (args.has("band-height")) {
        try {
            band_height = std::stoi(args.value("band-height"));
        } catch (const std::exception&) {
            std::cerr << "error: --band-height must be an integer\n";
            return EXIT_FAILURE;
        }
        if (band_height <= 0) {
            std::cerr << "error: --band-height must be positive\n";
            return EXIT_FAILURE;
        }
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

    int band_start = -1;
    int band_end = -1;
    std::vector<int> cuts;
    try {
        cuts = choose_cut_columns(*points, center, band_height, band_start, band_end);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::string cut_method = "row-band";
    const std::vector<int> centerline_cuts = choose_centerline_cut_columns(*points, center);
    if (centerline_cuts.size() > cuts.size()) {
        cuts = centerline_cuts;
        cut_method = "centerline";
    }

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
            cut_method == "centerline"
                ? mean_slice_radius(*points, center, c0, c1)
                : mean_band_radius(*points, center, band_start, band_end, c0, c1)
        });
    }

    const bool number_right_to_left =
        !slices.empty() && slices.back().radius < slices.front().radius;

    std::cout << "selected row band " << band_start << ".." << band_end << "\n";
    std::cout << "cut method: " << cut_method << "\n";
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
                  << " mean_band_radius " << slice.radius
                  << " -> " << out_path << "\n";
        ++written;
    }

    std::cout << "wrote " << written << " winding tifxyz directories\n";
    return EXIT_SUCCESS;
}
