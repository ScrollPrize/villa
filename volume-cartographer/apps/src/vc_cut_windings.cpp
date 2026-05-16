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
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const cv::Vec3f kInvalidPoint{-1.0f, -1.0f, -1.0f};

static void usage(const char* prog)
{
    std::cerr
        << "usage: " << prog << " --input <src tifxyz dir> [--output <dir>] [--start-winding 0]\n"
        << "  Cuts a rolled tifxyz surface into separate winding tifxyz folders.\n"
        << "  Outputs are named wNN_ddmmyyhhmm under <output dir>, or beside the\n"
        << "  source segment when <output dir> is omitted. Cut columns partition\n"
        << "  the source so each input column is written to exactly one output.\n";
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
static bool row_valid_col_range(const cv::Mat_<cv::Vec3f>& points, int row, int& out_c0, int& out_c1);
static bool row_valid_col_range_in_bounds(
    const cv::Mat_<cv::Vec3f>& points,
    int row,
    int c0,
    int c1,
    int& out_c0,
    int& out_c1);

struct CutSample {
    int row = -1;
    double col = 0.0;
};

struct CutCurve {
    int cut = 0;
    std::vector<CutSample> samples;
};

struct RowRevolutionCuts {
    std::vector<int> cuts;
    std::vector<CutCurve> curves;
    int rows_used = 0;
};

static void finalize_cut_curves(
    std::vector<CutCurve>& curves,
    int valid_c0,
    int valid_c1)
{
    for (CutCurve& curve : curves) {
        std::sort(curve.samples.begin(), curve.samples.end(), [](const CutSample& a, const CutSample& b) {
            if (a.row != b.row) return a.row < b.row;
            return a.col < b.col;
        });
        std::vector<double> cols;
        cols.reserve(curve.samples.size());
        for (const CutSample& sample : curve.samples) cols.push_back(sample.col);
        std::sort(cols.begin(), cols.end());
        const double med = cols[cols.size() / 2];
        curve.cut = std::clamp(static_cast<int>(std::lround(med)), valid_c0, valid_c1);
    }
}

static RowRevolutionCuts choose_row_revolution_cut_curves(
    const cv::Mat_<cv::Vec3f>& points,
    const cv::Vec2d& center,
    int valid_c0,
    int valid_c1)
{
    constexpr double pi = 3.1415926535897932384626433832795;
    constexpr double two_pi = 2.0 * pi;

    RowRevolutionCuts result;
    for (int r = 0; r < points.rows; ++r) {
        int row_c0 = 0;
        int row_c1 = points.cols - 1;
        if (!row_valid_col_range(points, r, row_c0, row_c1) || row_c1 - row_c0 < 2) {
            continue;
        }
        ++result.rows_used;

        std::vector<int> cols;
        std::vector<double> theta;
        cols.reserve(static_cast<size_t>(row_c1 - row_c0 + 1));
        theta.reserve(static_cast<size_t>(row_c1 - row_c0 + 1));
        for (int c = row_c0; c <= row_c1; ++c) {
            const cv::Vec3f& p = points(r, c);
            if (!valid_point(p)) continue;
            cols.push_back(c);
            theta.push_back(std::atan2(double(p[1]) - center[1], double(p[0]) - center[0]));
        }
        if (cols.size() < 3) continue;

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
        if (crossing_count <= 0) continue;

        if (result.curves.size() < static_cast<size_t>(crossing_count)) {
            result.curves.resize(static_cast<size_t>(crossing_count));
        }
        for (int k = 1; k <= crossing_count; ++k) {
            const double target = theta.front() + sign * two_pi * double(k);
            for (size_t i = 1; i < theta.size(); ++i) {
                const double prev = theta[i - 1];
                const double next = theta[i];
                const bool crosses = sign > 0.0
                    ? prev <= target && target <= next
                    : next <= target && target <= prev;
                if (!crosses || next == prev) continue;

                const double t = (target - prev) / (next - prev);
                const double col = double(cols[i - 1]) * (1.0 - t) + double(cols[i]) * t;
                if (col > valid_c0 && col < valid_c1) {
                    result.curves[static_cast<size_t>(k - 1)].samples.push_back({r, col});
                }
                break;
            }
        }
    }

    const int min_samples = std::max(3, result.rows_used / 20);
    result.curves.erase(
        std::remove_if(result.curves.begin(), result.curves.end(), [&](const CutCurve& curve) {
            return static_cast<int>(curve.samples.size()) < min_samples;
        }),
        result.curves.end());
    finalize_cut_curves(result.curves, valid_c0, valid_c1);
    std::sort(result.curves.begin(), result.curves.end(), [](const CutCurve& a, const CutCurve& b) {
        return a.cut < b.cut;
    });

    result.cuts.reserve(result.curves.size());
    for (const CutCurve& curve : result.curves) {
        result.cuts.push_back(curve.cut);
    }
    if (result.cuts.empty()) {
        throw std::runtime_error("no row revolution crossings detected");
    }
    return result;
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
    int left_curve = -1;
    int right_curve = -1;
    double radius = std::numeric_limits<double>::infinity();
};

static bool curve_col_at_row(const CutCurve& curve, int row, double& out_col)
{
    if (curve.samples.empty()) return false;
    if (row <= curve.samples.front().row) {
        out_col = curve.samples.front().col;
        return true;
    }
    if (row >= curve.samples.back().row) {
        out_col = curve.samples.back().col;
        return true;
    }

    auto it = std::lower_bound(
        curve.samples.begin(),
        curve.samples.end(),
        row,
        [](const CutSample& sample, int target_row) {
            return sample.row < target_row;
        });
    if (it == curve.samples.end()) return false;
    if (it->row == row) {
        out_col = it->col;
        return true;
    }
    if (it == curve.samples.begin()) return false;

    const CutSample& hi = *it;
    const CutSample& lo = *(it - 1);
    if (hi.row == lo.row) {
        out_col = lo.col;
        return true;
    }

    const double t = double(row - lo.row) / double(hi.row - lo.row);
    out_col = lo.col * (1.0 - t) + hi.col * t;
    return true;
}

static std::pair<int, int> slice_row_bounds(
    const WindingSlice& slice,
    const std::vector<CutCurve>& curves,
    int row,
    int valid_c0,
    int valid_c1)
{
    int c0 = slice.c0;
    int c1 = slice.c1;
    if (slice.left_curve >= 0) {
        double col = 0.0;
        if (curve_col_at_row(curves[static_cast<size_t>(slice.left_curve)], row, col)) {
            c0 = static_cast<int>(std::ceil(col));
        }
    }
    if (slice.right_curve >= 0) {
        double col = 0.0;
        if (curve_col_at_row(curves[static_cast<size_t>(slice.right_curve)], row, col)) {
            c1 = static_cast<int>(std::ceil(col)) - 1;
        }
    }
    return {std::clamp(c0, valid_c0, valid_c1), std::clamp(c1, valid_c0, valid_c1)};
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

static bool row_valid_col_range(const cv::Mat_<cv::Vec3f>& points, int row, int& out_c0, int& out_c1)
{
    out_c0 = points.cols;
    out_c1 = -1;
    for (int c = 0; c < points.cols; ++c) {
        if (valid_point(points(row, c))) {
            out_c0 = std::min(out_c0, c);
            out_c1 = c;
        }
    }
    return out_c1 >= out_c0;
}

static bool row_valid_col_range_in_bounds(
    const cv::Mat_<cv::Vec3f>& points,
    int row,
    int c0,
    int c1,
    int& out_c0,
    int& out_c1)
{
    c0 = std::clamp(c0, 0, points.cols - 1);
    c1 = std::clamp(c1, 0, points.cols - 1);
    if (c1 < c0) return false;

    out_c0 = points.cols;
    out_c1 = -1;
    for (int c = c0; c <= c1; ++c) {
        if (valid_point(points(row, c))) {
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

    std::shared_ptr<QuadSurface> surf;
    try {
        surf = std::shared_ptr<QuadSurface>(load_quad_from_tifxyz(src));
    } catch (const std::exception& e) {
        std::cerr << "error loading " << src << ": " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    cv::Mat_<cv::Vec3f>* points = surf->rawPointsPtr();
    if (!points || points->empty()) {
        std::cerr << "error: empty tifxyz point grid\n";
        return EXIT_FAILURE;
    }

    const cv::Vec2d center(
        (double(surf->bbox().low[0]) + double(surf->bbox().high[0])) * 0.5,
        (double(surf->bbox().low[1]) + double(surf->bbox().high[1])) * 0.5);

    int valid_c0 = 0;
    int valid_c1 = points->cols - 1;
    if (!valid_col_range(*points, valid_c0, valid_c1)) {
        std::cerr << "error: no valid points in tifxyz\n";
        return EXIT_FAILURE;
    }

    RowRevolutionCuts row_cuts;
    try {
        row_cuts = choose_row_revolution_cut_curves(*points, center, valid_c0, valid_c1);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
    std::vector<int> cuts = row_cuts.cuts;

    fs::create_directories(out_root);
    const std::string suffix = timestamp_ddmmyyhhmm();
    const cv::Vec2f scale = surf->scale();

    std::vector<WindingSlice> slices;
    slices.reserve(cuts.size() + 1);

    auto add_slice = [&](int c0, int c1, int left_curve, int right_curve) {
        if (c1 < c0) return;
        slices.push_back({
            c0,
            c1,
            left_curve,
            right_curve,
            mean_slice_radius(*points, center, c0, c1)
        });
    };

    add_slice(valid_c0, cuts.front() - 1, -1, 0);
    for (size_t i = 0; i + 1 < cuts.size(); ++i) {
        add_slice(cuts[i], cuts[i + 1] - 1, static_cast<int>(i), static_cast<int>(i + 1));
    }
    add_slice(cuts.back(), valid_c1, static_cast<int>(cuts.size() - 1), -1);

    const bool number_right_to_left =
        !slices.empty() && slices.back().radius < slices.front().radius;

    std::cout << "cut method: row theta revolutions\n";
    std::cout << "rows used: " << row_cuts.rows_used << "\n";
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

        std::vector<std::pair<int, int>> row_bounds(static_cast<size_t>(points->rows));
        int crop_c0 = valid_c1;
        int crop_c1 = valid_c0;
        int crop_r0 = points->rows;
        int crop_r1 = -1;
        for (int r = 0; r < points->rows; ++r) {
            auto [row_c0, row_c1] = slice_row_bounds(
                slice, row_cuts.curves, r, valid_c0, valid_c1);
            int valid_row_c0 = 0;
            int valid_row_c1 = points->cols - 1;
            if (!row_valid_col_range(*points, r, valid_row_c0, valid_row_c1)) {
                row_bounds[static_cast<size_t>(r)] = {1, 0};
                continue;
            }
            row_c0 = std::max({row_c0, slice.c0, valid_row_c0});
            row_c1 = std::min({row_c1, slice.c1, valid_row_c1});
            if (row_c1 < row_c0) {
                const int fallback_c0 = std::max(slice.c0, valid_row_c0);
                const int fallback_c1 = std::min(slice.c1, valid_row_c1);
                row_c0 = fallback_c0;
                row_c1 = fallback_c1;
            }

            int actual_row_c0 = 0;
            int actual_row_c1 = points->cols - 1;
            if (!row_valid_col_range_in_bounds(
                    *points, r, row_c0, row_c1, actual_row_c0, actual_row_c1)) {
                row_bounds[static_cast<size_t>(r)] = {1, 0};
                continue;
            }
            row_c0 = actual_row_c0;
            row_c1 = actual_row_c1;

            row_bounds[static_cast<size_t>(r)] = {row_c0, row_c1};
            if (row_c1 >= row_c0) {
                crop_c0 = std::min(crop_c0, row_c0);
                crop_c1 = std::max(crop_c1, row_c1);
                crop_r0 = std::min(crop_r0, r);
                crop_r1 = std::max(crop_r1, r);
            }
        }

        if (crop_c1 < crop_c0 || crop_r1 < crop_r0) {
            std::cerr << "warning: skipping empty winding at cols " << slice.c0 << ".." << slice.c1 << "\n";
            continue;
        }

        const cv::Rect rect(crop_c0, crop_r0, crop_c1 - crop_c0 + 1, crop_r1 - crop_r0 + 1);
        cv::Mat_<cv::Vec3f> crop = (*points)(rect).clone();
        for (int r = 0; r < crop.rows; ++r) {
            const int src_r = crop_r0 + r;
            const auto [row_c0, row_c1] = row_bounds[static_cast<size_t>(src_r)];
            for (int c = 0; c < crop.cols; ++c) {
                const int src_c = crop_c0 + c;
                if (src_c < row_c0 || src_c > row_c1) {
                    crop(r, c) = kInvalidPoint;
                }
            }
        }

        if (!has_valid_point(crop)) {
            std::cerr << "warning: skipping empty winding at cols " << slice.c0 << ".." << slice.c1 << "\n";
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

        std::cout << name.str() << ": cols " << crop_c0 << ".." << crop_c1
                  << " mean_radius " << slice.radius
                  << " -> " << out_path << "\n";
        ++written;
    }

    std::cout << "wrote " << written << " winding tifxyz directories\n";
    return EXIT_SUCCESS;
}
