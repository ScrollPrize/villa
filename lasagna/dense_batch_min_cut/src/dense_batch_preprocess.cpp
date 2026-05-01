#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <tiffio.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double kFixedThreshold = 127.0;
constexpr float kMinComponentRidgeRadius = 2.0f;
constexpr double kMinBoundaryAngleDegrees = 120.0;
using Clock = std::chrono::steady_clock;

struct TimingMark {
    Clock::time_point elapsed;
    std::clock_t cpu = 0;
};

struct StageTiming {
    std::string name;
    double elapsed_ms = 0.0;
    double cpu_ms = 0.0;
};

TimingMark start_timing() {
    return {Clock::now(), std::clock()};
}

StageTiming finish_timing(const std::string& name, const TimingMark& start) {
    const auto elapsed_end = Clock::now();
    const std::clock_t cpu_end = std::clock();
    return {name,
            std::chrono::duration<double, std::milli>(elapsed_end -
                                                       start.elapsed)
                .count(),
            1000.0 * static_cast<double>(cpu_end - start.cpu) /
                static_cast<double>(CLOCKS_PER_SEC)};
}

void print_stage_timings(const std::vector<StageTiming>& timings) {
    constexpr int kStageWidth = 44;
    constexpr int kNumericWidth = 14;
    double total_elapsed_ms = 0.0;
    for (const StageTiming& timing : timings) {
        if (timing.name == "total") {
            total_elapsed_ms = timing.elapsed_ms;
            break;
        }
    }
    if (total_elapsed_ms <= 0.0 && !timings.empty()) {
        total_elapsed_ms = timings.back().elapsed_ms;
    }

    std::cout << "Timings:\n"
              << "  " << std::left << std::setw(kStageWidth) << "stage"
              << std::right << std::setw(kNumericWidth) << "runtime_%"
              << std::setw(kNumericWidth) << "elapsed_ms"
              << std::setw(kNumericWidth) << "cpu_ms"
              << std::setw(kNumericWidth) << "cpu/elapsed"
              << "\n";
    std::cout << "  "
              << std::string(kStageWidth + 4 * kNumericWidth, '-')
              << "\n";
    for (const StageTiming& timing : timings) {
        const double runtime_percent =
            total_elapsed_ms > 0.0
                ? (100.0 * timing.elapsed_ms / total_elapsed_ms)
                : 0.0;
        const double utilization =
            timing.elapsed_ms > 0.0 ? timing.cpu_ms / timing.elapsed_ms : 0.0;
        std::cout << "  " << std::left << std::setw(kStageWidth)
                  << timing.name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(kNumericWidth) << runtime_percent
                  << std::setw(kNumericWidth) << timing.elapsed_ms
                  << std::setw(kNumericWidth) << timing.cpu_ms
                  << std::setw(kNumericWidth) << utilization
                  << "\n";
    }
}

struct Args {
    fs::path input;
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " -i <image>\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if ((key == "--input" || key == "-i") && i + 1 < argc) {
            args.input = argv[++i];
        } else if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + key);
        }
    }

    if (args.input.empty()) {
        throw std::runtime_error("--input/-i is required");
    }
    return args;
}

cv::Mat load_grayscale(const fs::path& path) {
    cv::Mat src = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (src.empty()) {
        throw std::runtime_error("failed to read input image: " + path.string());
    }

    if (src.channels() > 1) {
        cv::Mat gray;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    return src;
}

cv::Mat to_u8_for_threshold(const cv::Mat& src) {
    if (src.depth() == CV_8U) {
        return src;
    }

    cv::Mat src_float;
    src.convertTo(src_float, CV_32F);

    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(src_float, &min_value, &max_value);
    if (max_value <= min_value) {
        return cv::Mat::zeros(src.size(), CV_8U);
    }

    cv::Mat normalized;
    src_float.convertTo(normalized, CV_8U, 255.0 / (max_value - min_value),
                        -min_value * 255.0 / (max_value - min_value));
    return normalized;
}

cv::Mat binarize_fixed_threshold(const cv::Mat& gray) {
    cv::Mat u8 = to_u8_for_threshold(gray);
    cv::Mat binary;
    cv::threshold(u8, binary, kFixedThreshold, 255, cv::THRESH_BINARY_INV);
    return binary;
}

cv::Mat normalized_dt_u16(const cv::Mat& dt) {
    double max_value = 0.0;
    cv::minMaxLoc(dt, nullptr, &max_value);
    cv::Mat out;
    if (max_value <= 0.0) {
        out = cv::Mat::zeros(dt.size(), CV_16U);
    } else {
        dt.convertTo(out, CV_16U, 65535.0 / max_value);
    }
    return out;
}

int transition_count(const std::uint8_t p[8]) {
    int count = 0;
    for (int i = 0; i < 8; ++i) {
        if (p[i] == 0 && p[(i + 1) % 8] != 0) {
            ++count;
        }
    }
    return count;
}

int nonzero_count(const std::uint8_t p[8]) {
    return static_cast<int>(std::count_if(p, p + 8, [](std::uint8_t v) {
        return v != 0;
    }));
}

void thinning_iteration(cv::Mat& img, int iter) {
    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8U);

    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            if (img.at<std::uint8_t>(y, x) == 0) {
                continue;
            }

            const std::uint8_t p[8] = {
                img.at<std::uint8_t>(y - 1, x),
                img.at<std::uint8_t>(y - 1, x + 1),
                img.at<std::uint8_t>(y, x + 1),
                img.at<std::uint8_t>(y + 1, x + 1),
                img.at<std::uint8_t>(y + 1, x),
                img.at<std::uint8_t>(y + 1, x - 1),
                img.at<std::uint8_t>(y, x - 1),
                img.at<std::uint8_t>(y - 1, x - 1),
            };

            const int nz = nonzero_count(p);
            const int transitions = transition_count(p);
            if (nz < 2 || nz > 6 || transitions != 1) {
                continue;
            }

            const bool remove =
                iter == 0
                    ? (p[0] * p[2] * p[4] == 0 && p[2] * p[4] * p[6] == 0)
                    : (p[0] * p[2] * p[6] == 0 && p[0] * p[4] * p[6] == 0);
            if (remove) {
                marker.at<std::uint8_t>(y, x) = 255;
            }
        }
    }

    img.setTo(0, marker);
}

cv::Mat zhang_suen_thinning(const cv::Mat& src) {
    CV_Assert(src.type() == CV_8U);

    cv::Mat img;
    cv::threshold(src, img, 0, 1, cv::THRESH_BINARY);

    cv::Mat prev = cv::Mat::zeros(img.size(), CV_8U);
    cv::Mat diff;
    do {
        thinning_iteration(img, 0);
        thinning_iteration(img, 1);
        cv::absdiff(img, prev, diff);
        img.copyTo(prev);
    } while (cv::countNonZero(diff) > 0);

    img *= 255;
    return img;
}

cv::Mat optimized_thinning(const cv::Mat& src) {
    CV_Assert(src.type() == CV_8U);

    cv::Mat binary;
    cv::threshold(src, binary, 0, 255, cv::THRESH_BINARY);
    cv::Mat out;
    cv::ximgproc::thinning(binary, out, cv::ximgproc::THINNING_ZHANGSUEN);
    return out;
}

struct PixelByDistance {
    int x = 0;
    int y = 0;
    float distance = 0.0f;
};

struct PixelPriorityGreater {
    bool operator()(const PixelByDistance& a, const PixelByDistance& b) const {
        if (a.distance != b.distance) {
            return a.distance > b.distance;
        }
        if (a.y != b.y) {
            return a.y > b.y;
        }
        return a.x > b.x;
    }
};

int count_local_components(const std::array<std::uint8_t, 9>& values,
                           bool foreground, bool eight_connected) {
    constexpr std::array<std::pair<int, int>, 8> kDirs8 = {
        {{-1, -1}, {0, -1}, {1, -1}, {-1, 0},
         {1, 0},   {-1, 1}, {0, 1},  {1, 1}}};
    constexpr std::array<std::pair<int, int>, 4> kDirs4 = {
        {{0, -1}, {-1, 0}, {1, 0}, {0, 1}}};

    std::array<bool, 9> seen{};
    int components = 0;
    for (int start = 0; start < 9; ++start) {
        const bool matches = foreground ? values[start] != 0 : values[start] == 0;
        if (!matches || seen[start]) {
            continue;
        }

        ++components;
        std::array<int, 9> stack{};
        int stack_size = 0;
        stack[stack_size++] = start;
        seen[start] = true;

        while (stack_size > 0) {
            const int idx = stack[--stack_size];
            const int cx = idx % 3;
            const int cy = idx / 3;
            for (int i = 0; i < (eight_connected ? 8 : 4); ++i) {
                const auto dir = eight_connected ? kDirs8[i] : kDirs4[i];
                const int nx = cx + dir.first;
                const int ny = cy + dir.second;
                if (nx < 0 || nx >= 3 || ny < 0 || ny >= 3) {
                    continue;
                }
                const int next_idx = ny * 3 + nx;
                const bool next_matches =
                    foreground ? values[next_idx] != 0 : values[next_idx] == 0;
                if (next_matches && !seen[next_idx]) {
                    seen[next_idx] = true;
                    stack[stack_size++] = next_idx;
                }
            }
        }
    }
    return components;
}

std::array<std::uint8_t, 9> mask_to_values(int mask) {
    std::array<std::uint8_t, 9> values{};
    for (int i = 0; i < 9; ++i) {
        values[i] = (mask & (1 << i)) != 0 ? 1 : 0;
    }
    return values;
}

std::array<std::uint8_t, 512> make_neighbor_count_lut() {
    std::array<std::uint8_t, 512> lut{};
    for (int mask = 0; mask < 512; ++mask) {
        int count = 0;
        for (int i = 0; i < 9; ++i) {
            if (i != 4 && (mask & (1 << i)) != 0) {
                ++count;
            }
        }
        lut[mask] = static_cast<std::uint8_t>(count);
    }
    return lut;
}

std::array<std::uint8_t, 512> make_simple_point_lut() {
    std::array<std::uint8_t, 512> lut{};
    for (int mask = 0; mask < 512; ++mask) {
        if ((mask & (1 << 4)) == 0) {
            continue;
        }

        std::array<std::uint8_t, 9> before = mask_to_values(mask);
        std::array<std::uint8_t, 9> after = before;
        after[4] = 0;

        const int fg_before = count_local_components(before, true, true);
        const int fg_after = count_local_components(after, true, true);
        const int bg_before = count_local_components(before, false, false);
        const int bg_after = count_local_components(after, false, false);

        lut[mask] = (fg_before == fg_after && bg_before == bg_after) ? 1 : 0;
    }
    return lut;
}

int neighborhood_mask(const cv::Mat& img, int x, int y) {
    int mask = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int idx = (dy + 1) * 3 + (dx + 1);
            if (img.at<std::uint8_t>(y + dy, x + dx) != 0) {
                mask |= 1 << idx;
            }
        }
    }
    return mask;
}

int count_foreground_neighbors(int mask) {
    static const std::array<std::uint8_t, 512> lut = make_neighbor_count_lut();
    return lut[mask];
}

bool is_simple_point_after_removal(int mask) {
    static const std::array<std::uint8_t, 512> lut = make_simple_point_lut();
    return lut[mask] != 0;
}

cv::Mat distance_ordered_thinning(const cv::Mat& binary, const cv::Mat& dt) {
    CV_Assert(binary.type() == CV_8U);
    CV_Assert(dt.type() == CV_32F);

    cv::Mat img;
    cv::threshold(binary, img, 0, 255, cv::THRESH_BINARY);

    std::priority_queue<PixelByDistance, std::vector<PixelByDistance>,
                        PixelPriorityGreater>
        active;
    cv::Mat in_queue = cv::Mat::zeros(img.size(), CV_8U);
    const auto push_candidate = [&](int x, int y) {
        if (x <= 0 || x >= img.cols - 1 || y <= 0 || y >= img.rows - 1) {
            return;
        }
        if (img.at<std::uint8_t>(y, x) == 0 ||
            in_queue.at<std::uint8_t>(y, x) != 0) {
            return;
        }
        const int mask = neighborhood_mask(img, x, y);
        if (count_foreground_neighbors(mask) <= 1) {
            return;
        }
        active.push({x, y, dt.at<float>(y, x)});
        in_queue.at<std::uint8_t>(y, x) = 1;
    };

    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            push_candidate(x, y);
        }
    }

    while (!active.empty()) {
        const PixelByDistance pixel = active.top();
        active.pop();
        in_queue.at<std::uint8_t>(pixel.y, pixel.x) = 0;

        if (img.at<std::uint8_t>(pixel.y, pixel.x) == 0) {
            continue;
        }
        const int mask = neighborhood_mask(img, pixel.x, pixel.y);
        if (count_foreground_neighbors(mask) <= 1) {
            continue;
        }
        if (!is_simple_point_after_removal(mask)) {
            continue;
        }

        img.at<std::uint8_t>(pixel.y, pixel.x) = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                const int nx = pixel.x + dx;
                const int ny = pixel.y + dy;
                push_candidate(nx, ny);
            }
        }
    }

    return img;
}

cv::Mat distance_ordered_thinning_full_pass_reference(const cv::Mat& binary,
                                                      const cv::Mat& dt) {
    CV_Assert(binary.type() == CV_8U);
    CV_Assert(dt.type() == CV_32F);

    cv::Mat img;
    cv::threshold(binary, img, 0, 255, cv::THRESH_BINARY);

    std::vector<PixelByDistance> pixels;
    pixels.reserve(static_cast<std::size_t>(cv::countNonZero(img)));
    for (int y = 1; y < img.rows - 1; ++y) {
        for (int x = 1; x < img.cols - 1; ++x) {
            if (img.at<std::uint8_t>(y, x) != 0) {
                pixels.push_back({x, y, dt.at<float>(y, x)});
            }
        }
    }

    std::sort(pixels.begin(), pixels.end(), [](const auto& a, const auto& b) {
        return PixelPriorityGreater{}(b, a);
    });

    bool changed = false;
    do {
        changed = false;
        for (const PixelByDistance& pixel : pixels) {
            if (img.at<std::uint8_t>(pixel.y, pixel.x) == 0) {
                continue;
            }
            const int mask = neighborhood_mask(img, pixel.x, pixel.y);
            if (count_foreground_neighbors(mask) <= 1) {
                continue;
            }
            if (!is_simple_point_after_removal(mask)) {
                continue;
            }
            img.at<std::uint8_t>(pixel.y, pixel.x) = 0;
            changed = true;
        }
    } while (changed);

    return img;
}

bool has_label(const std::vector<int>& values, int label) {
    return std::find(values.begin(), values.end(), label) != values.end();
}

double max_boundary_angle_degrees(const std::vector<int>& labels,
                                  const std::vector<cv::Point>& source_points,
                                  cv::Point pixel) {
    double min_dot = 1.0;
    bool have_pair = false;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        const cv::Point a = source_points[labels[i]];
        if (a.x < 0) {
            continue;
        }
        const double ax = static_cast<double>(a.x - pixel.x);
        const double ay = static_cast<double>(a.y - pixel.y);
        const double alen = std::sqrt(ax * ax + ay * ay);
        if (alen == 0.0) {
            continue;
        }
        for (std::size_t j = i + 1; j < labels.size(); ++j) {
            const cv::Point b = source_points[labels[j]];
            if (b.x < 0) {
                continue;
            }
            const double bx = static_cast<double>(b.x - pixel.x);
            const double by = static_cast<double>(b.y - pixel.y);
            const double blen = std::sqrt(bx * bx + by * by);
            if (blen == 0.0) {
                continue;
            }
            const double dot = (ax * bx + ay * by) / (alen * blen);
            min_dot = std::min(min_dot, std::max(-1.0, std::min(1.0, dot)));
            have_pair = true;
        }
    }

    if (!have_pair) {
        return 0.0;
    }
    return std::acos(min_dot) * 180.0 / CV_PI;
}

cv::Mat voronoi_label_ridges(const cv::Mat& binary,
                             const cv::Mat* foreground_labels = nullptr) {
    CV_Assert(binary.type() == CV_8U);
    CV_Assert(foreground_labels == nullptr || foreground_labels->type() == CV_32S);

    cv::Mat dt_approx;
    cv::Mat labels;
    cv::distanceTransform(binary, dt_approx, labels, cv::DIST_L2, cv::DIST_MASK_5,
                          cv::DIST_LABEL_PIXEL);

    cv::Mat ridges = cv::Mat::zeros(binary.size(), CV_8U);
    for (int y = 1; y < binary.rows - 1; ++y) {
        for (int x = 1; x < binary.cols - 1; ++x) {
            if (binary.at<std::uint8_t>(y, x) == 0) {
                continue;
            }
            const int foreground_label =
                foreground_labels == nullptr ? 0 : foreground_labels->at<int>(y, x);

            const int center_label = labels.at<int>(y, x);
            bool touches_multiple_sites = false;
            for (int dy = -1; dy <= 1 && !touches_multiple_sites; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    if (binary.at<std::uint8_t>(y + dy, x + dx) == 0) {
                        continue;
                    }
                    if (foreground_labels != nullptr &&
                        foreground_labels->at<int>(y + dy, x + dx) !=
                            foreground_label) {
                        continue;
                    }
                    if (labels.at<int>(y + dy, x + dx) != center_label) {
                        touches_multiple_sites = true;
                        break;
                    }
                }
            }

            if (touches_multiple_sites) {
                ridges.at<std::uint8_t>(y, x) = 255;
            }
        }
    }

    return ridges;
}

std::vector<cv::Point> label_source_points(const cv::Mat& zero_source_mask,
                                           const cv::Mat& labels) {
    int max_label = 0;
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            max_label = std::max(max_label, labels.at<int>(y, x));
        }
    }

    std::vector<cv::Point> source_points(static_cast<std::size_t>(max_label + 1),
                                         cv::Point(-1, -1));
    for (int y = 0; y < zero_source_mask.rows; ++y) {
        for (int x = 0; x < zero_source_mask.cols; ++x) {
            if (zero_source_mask.at<std::uint8_t>(y, x) != 0) {
                continue;
            }
            const int label = labels.at<int>(y, x);
            if (label > 0 && source_points[label].x < 0) {
                source_points[label] = cv::Point(x, y);
            }
        }
    }
    return source_points;
}

cv::Mat per_component_voronoi_ridges(const cv::Mat& binary,
                                     bool require_angular_separation) {
    CV_Assert(binary.type() == CV_8U);

    cv::Mat component_labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int num_components = cv::connectedComponentsWithStats(
        binary, component_labels, stats, centroids, 8, CV_32S);

    cv::Mat ridges = cv::Mat::zeros(binary.size(), CV_8U);
    for (int component = 1; component < num_components; ++component) {
        const int area = stats.at<int>(component, cv::CC_STAT_AREA);
        if (area < 8) {
            continue;
        }

        const int left = stats.at<int>(component, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(component, cv::CC_STAT_TOP);
        const int width = stats.at<int>(component, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(component, cv::CC_STAT_HEIGHT);
        const int x0 = std::max(0, left - 1);
        const int y0 = std::max(0, top - 1);
        const int x1 = std::min(binary.cols, left + width + 1);
        const int y1 = std::min(binary.rows, top + height + 1);
        const cv::Rect roi(x0, y0, x1 - x0, y1 - y0);

        cv::Mat component_mask = cv::Mat::zeros(roi.size(), CV_8U);
        for (int y = 0; y < roi.height; ++y) {
            for (int x = 0; x < roi.width; ++x) {
                if (component_labels.at<int>(roi.y + y, roi.x + x) == component) {
                    component_mask.at<std::uint8_t>(y, x) = 255;
                }
            }
        }

        cv::Mat component_dt;
        cv::Mat nearest_labels;
        cv::distanceTransform(component_mask, component_dt, nearest_labels,
                              cv::DIST_L2, cv::DIST_MASK_5,
                              cv::DIST_LABEL_PIXEL);
        const std::vector<cv::Point> source_points =
            label_source_points(component_mask, nearest_labels);

        for (int y = 1; y < roi.height - 1; ++y) {
            for (int x = 1; x < roi.width - 1; ++x) {
                if (component_mask.at<std::uint8_t>(y, x) == 0 ||
                    component_dt.at<float>(y, x) < kMinComponentRidgeRadius) {
                    continue;
                }

                std::vector<int> local_labels;
                const int radius = require_angular_separation ? 2 : 1;
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        const int nx = x + dx;
                        const int ny = y + dy;
                        if (nx < 0 || nx >= roi.width || ny < 0 ||
                            ny >= roi.height ||
                            component_mask.at<std::uint8_t>(ny, nx) == 0) {
                            continue;
                        }
                        const int label = nearest_labels.at<int>(ny, nx);
                        if (label > 0 && !has_label(local_labels, label)) {
                            local_labels.push_back(label);
                        }
                    }
                }

                if (local_labels.size() < 2) {
                    continue;
                }
                if (require_angular_separation &&
                    max_boundary_angle_degrees(local_labels, source_points,
                                               cv::Point(x, y)) <
                        kMinBoundaryAngleDegrees) {
                    continue;
                }

                ridges.at<std::uint8_t>(roi.y + y, roi.x + x) = 255;
            }
        }
    }

    return ridges;
}

int ridge_degree(const cv::Mat& img, int x, int y) {
    int degree = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            if (img.at<std::uint8_t>(y + dy, x + dx) != 0) {
                ++degree;
            }
        }
    }
    return degree;
}

cv::Mat prune_to_2core(const cv::Mat& src) {
    CV_Assert(src.type() == CV_8U);

    cv::Mat img;
    cv::threshold(src, img, 0, 255, cv::THRESH_BINARY);

    bool changed = false;
    do {
        changed = false;
        cv::Mat remove = cv::Mat::zeros(img.size(), CV_8U);
        for (int y = 1; y < img.rows - 1; ++y) {
            for (int x = 1; x < img.cols - 1; ++x) {
                if (img.at<std::uint8_t>(y, x) != 0 &&
                    ridge_degree(img, x, y) <= 1) {
                    remove.at<std::uint8_t>(y, x) = 255;
                    changed = true;
                }
            }
        }
        img.setTo(0, remove);
    } while (changed);

    return img;
}

cv::Mat biconnected_cycle_pixels(const cv::Mat& ridge_mask) {
    CV_Assert(ridge_mask.type() == CV_8U);

    return prune_to_2core(zhang_suen_thinning(ridge_mask));
}

cv::Mat binary_contour_loops(const cv::Mat& binary) {
    CV_Assert(binary.type() == CV_8U);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary.clone(), contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_NONE);

    cv::Mat out = cv::Mat::zeros(binary.size(), CV_8U);
    for (std::size_t i = 0; i < contours.size(); ++i) {
        if (hierarchy.empty() || hierarchy[i][3] < 0) {
            continue;
        }
        if (std::abs(cv::contourArea(contours[i])) < 8.0) {
            continue;
        }
        cv::drawContours(out, contours, static_cast<int>(i), cv::Scalar(255), 1,
                         cv::LINE_8, hierarchy);
    }
    return out;
}

cv::Mat prune_short_low_dt_spurs(const cv::Mat& skeleton, const cv::Mat& dt) {
    CV_Assert(skeleton.type() == CV_8U);
    CV_Assert(dt.type() == CV_32F);

    constexpr int kMaxPrunedSpurLength = 16;
    constexpr float kMinKeptSpurMaxDistance = 8.0f;
    const std::array<cv::Point, 8> kDirs = {
        {{-1, -1}, {0, -1}, {1, -1}, {-1, 0},
         {1, 0},   {-1, 1}, {0, 1},  {1, 1}}};

    cv::Mat img;
    cv::threshold(skeleton, img, 0, 255, cv::THRESH_BINARY);

    bool changed = false;
    do {
        changed = false;
        cv::Mat remove = cv::Mat::zeros(img.size(), CV_8U);

        for (int y = 1; y < img.rows - 1; ++y) {
            for (int x = 1; x < img.cols - 1; ++x) {
                if (img.at<std::uint8_t>(y, x) == 0 ||
                    ridge_degree(img, x, y) != 1) {
                    continue;
                }

                std::vector<cv::Point> branch;
                branch.push_back(cv::Point(x, y));
                float max_dt = dt.at<float>(y, x);
                cv::Point prev(-1, -1);
                cv::Point cur(x, y);

                while (static_cast<int>(branch.size()) <=
                       kMaxPrunedSpurLength + 1) {
                    std::vector<cv::Point> neighbors;
                    for (const cv::Point dir : kDirs) {
                        const cv::Point next = cur + dir;
                        if (next.x <= 0 || next.x >= img.cols - 1 ||
                            next.y <= 0 || next.y >= img.rows - 1 ||
                            next == prev ||
                            img.at<std::uint8_t>(next.y, next.x) == 0) {
                            continue;
                        }
                        neighbors.push_back(next);
                    }

                    if (neighbors.empty()) {
                        break;
                    }
                    if (neighbors.size() > 1) {
                        break;
                    }

                    prev = cur;
                    cur = neighbors.front();
                    const int degree = ridge_degree(img, cur.x, cur.y);
                    max_dt = std::max(max_dt, dt.at<float>(cur.y, cur.x));

                    if (degree >= 3) {
                        break;
                    }

                    branch.push_back(cur);
                    if (degree <= 1) {
                        break;
                    }
                }

                if (static_cast<int>(branch.size()) <= kMaxPrunedSpurLength &&
                    max_dt < kMinKeptSpurMaxDistance) {
                    for (const cv::Point pixel : branch) {
                        remove.at<std::uint8_t>(pixel.y, pixel.x) = 255;
                    }
                    changed = true;
                }
            }
        }

        img.setTo(0, remove);
    } while (changed);

    return img;
}

cv::Mat source_pixel_label_ridges(const cv::Mat& white_domain,
                                  const cv::Mat& source_pixel_labels) {
    CV_Assert(white_domain.type() == CV_8U);
    CV_Assert(source_pixel_labels.type() == CV_32S);

    cv::Mat ridges = cv::Mat::zeros(white_domain.size(), CV_8U);
    cv::parallel_for_(cv::Range(1, white_domain.rows - 1),
                      [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            for (int x = 1; x < white_domain.cols - 1; ++x) {
                if (white_domain.at<std::uint8_t>(y, x) == 0) {
                    continue;
                }
                const int center = source_pixel_labels.at<int>(y, x);
                if (center <= 0) {
                    continue;
                }

                bool touches_other = false;
                for (int dy = -1; dy <= 1 && !touches_other; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        if (white_domain.at<std::uint8_t>(y + dy, x + dx) ==
                            0) {
                            continue;
                        }
                        const int other =
                            source_pixel_labels.at<int>(y + dy, x + dx);
                        if (other > 0 && other != center) {
                            touches_other = true;
                            break;
                        }
                    }
                }

                if (touches_other) {
                    ridges.at<std::uint8_t>(y, x) = 255;
                }
            }
        }
    });

    return ridges;
}

cv::Mat connect_clean_skeleton_with_source_ridges(const cv::Mat& clean_skeleton,
                                                  const cv::Mat& source_skeleton,
                                                  const cv::Mat& dt) {
    CV_Assert(clean_skeleton.type() == CV_8U);
    CV_Assert(source_skeleton.type() == CV_8U);
    CV_Assert(dt.type() == CV_32F);

    cv::Mat clean;
    cv::threshold(clean_skeleton, clean, 0, 255, cv::THRESH_BINARY);

    cv::Mat clean_labels;
    const int num_clean_components =
        cv::connectedComponents(clean, clean_labels, 8, CV_32S);
    if (num_clean_components <= 2) {
        return clean.clone();
    }

    cv::Mat connector_candidates;
    cv::threshold(source_skeleton, connector_candidates, 0, 255,
                  cv::THRESH_BINARY);
    connector_candidates.setTo(0, clean);

    cv::Mat candidate_labels;
    cv::Mat candidate_stats;
    cv::Mat candidate_centroids;
    const int num_candidates = cv::connectedComponentsWithStats(
        connector_candidates, candidate_labels, candidate_stats,
        candidate_centroids, 8, CV_32S);
    if (num_candidates <= 1) {
        return clean.clone();
    }

    struct CandidateWorkspace {
        cv::Rect bounds;
        std::vector<cv::Point> pixels;
        cv::Mat local_index;
    };

    std::vector<CandidateWorkspace> candidate_workspaces(
        static_cast<std::size_t>(num_candidates));
    for (int label = 1; label < num_candidates; ++label) {
        const int left = candidate_stats.at<int>(label, cv::CC_STAT_LEFT);
        const int top = candidate_stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = candidate_stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = candidate_stats.at<int>(label, cv::CC_STAT_HEIGHT);
        CandidateWorkspace& workspace = candidate_workspaces[label];
        workspace.bounds = cv::Rect(left, top, width, height);
        workspace.pixels.reserve(
            static_cast<std::size_t>(
                candidate_stats.at<int>(label, cv::CC_STAT_AREA)));
        workspace.local_index =
            cv::Mat(height, width, CV_32S, cv::Scalar(-1));
    }

    for (int y = 0; y < candidate_labels.rows; ++y) {
        for (int x = 0; x < candidate_labels.cols; ++x) {
            const int label = candidate_labels.at<int>(y, x);
            if (label <= 0) {
                continue;
            }
            CandidateWorkspace& workspace = candidate_workspaces[label];
            const int local =
                static_cast<int>(workspace.pixels.size());
            workspace.pixels.push_back(cv::Point(x, y));
            workspace.local_index.at<int>(y - workspace.bounds.y,
                                          x - workspace.bounds.x) = local;
        }
    }

    cv::Mat clean_source_mask(clean.size(), CV_8U, cv::Scalar(255));
    clean_source_mask.setTo(0, clean);

    cv::Mat distance_to_clean;
    cv::Mat nearest_clean_pixel;
    cv::distanceTransform(clean_source_mask, distance_to_clean,
                          nearest_clean_pixel, cv::DIST_L2, cv::DIST_MASK_5,
                          cv::DIST_LABEL_PIXEL);
    const std::vector<cv::Point> clean_source_points =
        label_source_points(clean_source_mask, nearest_clean_pixel);

    struct Attachment {
        int component = 0;
        cv::Point pixel{-1, -1};
        cv::Point clean_pixel{-1, -1};
    };

    struct CandidateEdge {
        int label = 0;
        int a = 0;
        int b = 0;
        float bottleneck_dt = 0.0f;
        int length = 0;
        cv::Point clean_a{-1, -1};
        cv::Point clean_b{-1, -1};
        std::vector<cv::Point> path;
    };

    struct DisjointSet {
        std::vector<int> parent;
        explicit DisjointSet(int n) : parent(static_cast<std::size_t>(n)) {
            std::iota(parent.begin(), parent.end(), 0);
        }
        int find(int x) {
            while (parent[x] != x) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }
        bool unite(int a, int b) {
            a = find(a);
            b = find(b);
            if (a == b) {
                return false;
            }
            parent[b] = a;
            return true;
        }
    };

    constexpr float kMaxAttachDistance = 3.0f;
    const std::array<cv::Point, 8> kDirs = {
        {{-1, -1}, {0, -1}, {1, -1}, {-1, 0},
         {1, 0},   {-1, 1}, {0, 1},  {1, 1}}};
    std::vector<std::vector<Attachment>> attachments(
        static_cast<std::size_t>(num_candidates));

    for (int y = 0; y < candidate_labels.rows; ++y) {
        for (int x = 0; x < candidate_labels.cols; ++x) {
            const int candidate_label = candidate_labels.at<int>(y, x);
            if (candidate_label <= 0) {
                continue;
            }

            if (distance_to_clean.at<float>(y, x) > kMaxAttachDistance) {
                continue;
            }

            const int source_label = nearest_clean_pixel.at<int>(y, x);
            if (source_label <= 0 ||
                source_label >= static_cast<int>(clean_source_points.size())) {
                continue;
            }
            const cv::Point source = clean_source_points[source_label];
            if (source.x < 0) {
                continue;
            }

            const int clean_label = clean_labels.at<int>(source.y, source.x);
            if (clean_label <= 0) {
                continue;
            }

            attachments[candidate_label].push_back(
                {clean_label, cv::Point(x, y), source});
        }
    }

    std::vector<std::vector<CandidateEdge>> edges_by_candidate(
        static_cast<std::size_t>(num_candidates));
    cv::parallel_for_(cv::Range(1, num_candidates), [&](const cv::Range& range) {
        for (int label = range.start; label < range.end; ++label) {
            struct State {
                int idx = 0;
                int owner = 0;
                float bottleneck = 0.0f;
                int length = 0;
            };
            struct StateLess {
                bool operator()(const State& a, const State& b) const {
                    if (a.bottleneck != b.bottleneck) {
                        return a.bottleneck < b.bottleneck;
                    }
                    if (a.length != b.length) {
                        return a.length > b.length;
                    }
                    return a.owner > b.owner;
                }
            };

            const CandidateWorkspace& workspace = candidate_workspaces[label];
            const int total = static_cast<int>(workspace.pixels.size());
            if (total == 0) {
                continue;
            }

            std::vector<float> best(static_cast<std::size_t>(total), -1.0f);
            std::vector<int> best_length(static_cast<std::size_t>(total),
                                         std::numeric_limits<int>::max());
            std::vector<int> parent(static_cast<std::size_t>(total), -1);
            std::vector<int> owner(static_cast<std::size_t>(total), 0);
            std::vector<cv::Point> owner_clean(
                static_cast<std::size_t>(total), cv::Point(-1, -1));
            std::priority_queue<State, std::vector<State>, StateLess> queue;

            for (const Attachment& attachment : attachments[label]) {
                const int idx = workspace.local_index.at<int>(
                    attachment.pixel.y - workspace.bounds.y,
                    attachment.pixel.x - workspace.bounds.x);
                if (idx < 0) {
                    continue;
                }
                const float start_dt =
                    dt.at<float>(attachment.pixel.y, attachment.pixel.x);
                const bool better =
                    start_dt > best[idx] ||
                    (start_dt == best[idx] &&
                     (owner[idx] == 0 || attachment.component < owner[idx]));
                if (!better) {
                    continue;
                }
                best[idx] = start_dt;
                best_length[idx] = 1;
                parent[idx] = idx;
                owner[idx] = attachment.component;
                owner_clean[idx] = attachment.clean_pixel;
                queue.push({idx, attachment.component, start_dt, 1});
            }

            std::vector<CandidateEdge> local_edges;

            const auto path_to_root = [&](int idx) {
                std::vector<cv::Point> path;
                while (idx >= 0) {
                    path.push_back(workspace.pixels[idx]);
                    if (parent[idx] == idx) {
                        break;
                    }
                    idx = parent[idx];
                }
                return path;
            };

            while (!queue.empty()) {
                const State state = queue.top();
                queue.pop();
                if (state.owner != owner[state.idx] ||
                    state.bottleneck < best[state.idx] ||
                    (state.bottleneck == best[state.idx] &&
                     state.length > best_length[state.idx])) {
                    continue;
                }

                const cv::Point pixel = workspace.pixels[state.idx];
                for (const cv::Point dir : kDirs) {
                    const cv::Point next_pixel = pixel + dir;
                    if (!workspace.bounds.contains(next_pixel)) {
                        continue;
                    }
                    const int next_idx = workspace.local_index.at<int>(
                        next_pixel.y - workspace.bounds.y,
                        next_pixel.x - workspace.bounds.x);
                    if (next_idx < 0) {
                        continue;
                    }

                    if (owner[next_idx] > 0 &&
                        owner[next_idx] != state.owner) {
                        CandidateEdge edge;
                        edge.label = label;
                        edge.a = state.owner;
                        edge.b = owner[next_idx];
                        edge.clean_a = owner_clean[state.idx];
                        edge.clean_b = owner_clean[next_idx];
                        edge.bottleneck_dt =
                            std::min(state.bottleneck, best[next_idx]);

                        std::vector<cv::Point> b_path =
                            path_to_root(next_idx);
                        std::reverse(b_path.begin(), b_path.end());
                        edge.path = std::move(b_path);
                        std::vector<cv::Point> a_path =
                            path_to_root(state.idx);
                        edge.path.insert(edge.path.end(), a_path.begin(),
                                         a_path.end());
                        edge.length = static_cast<int>(edge.path.size());
                        local_edges.push_back(std::move(edge));
                        continue;
                    }

                    const float next_bottleneck =
                        std::min(state.bottleneck,
                                 dt.at<float>(next_pixel.y, next_pixel.x));
                    const int next_length = state.length + 1;
                    if (owner[next_idx] == 0 ||
                        next_bottleneck > best[next_idx] ||
                        (owner[next_idx] == state.owner &&
                         next_bottleneck == best[next_idx] &&
                         next_length < best_length[next_idx])) {
                        best[next_idx] = next_bottleneck;
                        best_length[next_idx] = next_length;
                        parent[next_idx] = state.idx;
                        owner[next_idx] = state.owner;
                        owner_clean[next_idx] = owner_clean[state.idx];
                        queue.push({next_idx, state.owner, next_bottleneck,
                                    next_length});
                    }
                }
            }

            std::sort(local_edges.begin(), local_edges.end(),
                      [](const CandidateEdge& a, const CandidateEdge& b) {
                const int a0 = std::min(a.a, a.b);
                const int a1 = std::max(a.a, a.b);
                const int b0 = std::min(b.a, b.b);
                const int b1 = std::max(b.a, b.b);
                if (a0 != b0) {
                    return a0 < b0;
                }
                if (a1 != b1) {
                    return a1 < b1;
                }
                if (a.bottleneck_dt != b.bottleneck_dt) {
                    return a.bottleneck_dt > b.bottleneck_dt;
                }
                return a.length < b.length;
            });

            std::vector<CandidateEdge> deduped_edges;
            for (CandidateEdge& edge : local_edges) {
                if (!deduped_edges.empty()) {
                    const CandidateEdge& prev = deduped_edges.back();
                    if (std::min(prev.a, prev.b) == std::min(edge.a, edge.b) &&
                        std::max(prev.a, prev.b) == std::max(edge.a, edge.b)) {
                        continue;
                    }
                }
                deduped_edges.push_back(std::move(edge));
            }
            edges_by_candidate[label] = std::move(deduped_edges);
        }
    });

    std::vector<CandidateEdge> edges;
    for (int label = 1; label < num_candidates; ++label) {
        for (CandidateEdge& edge : edges_by_candidate[label]) {
            edges.push_back(std::move(edge));
        }
    }

    std::sort(edges.begin(), edges.end(), [](const CandidateEdge& a,
                                             const CandidateEdge& b) {
        if (a.bottleneck_dt != b.bottleneck_dt) {
            return a.bottleneck_dt > b.bottleneck_dt;
        }
        if (a.length != b.length) {
            return a.length < b.length;
        }
        return a.label < b.label;
    });

    cv::Mat out = clean.clone();
    DisjointSet sets(num_clean_components);
    for (const CandidateEdge& edge : edges) {
        if (!sets.unite(edge.a, edge.b)) {
            continue;
        }
        for (const cv::Point pixel : edge.path) {
            out.at<std::uint8_t>(pixel.y, pixel.x) = 255;
        }
        if (!edge.path.empty()) {
            if (edge.clean_a.x >= 0) {
                cv::line(out, edge.clean_a, edge.path.back(), cv::Scalar(255),
                         1, cv::LINE_8);
            }
            if (edge.clean_b.x >= 0) {
                cv::line(out, edge.clean_b, edge.path.front(), cv::Scalar(255),
                         1, cv::LINE_8);
            }
        }
    }

    return out;
}

struct ComponentVoronoiResult {
    cv::Mat labels_u16;
    cv::Mat boundaries;
    cv::Mat boundary_skeleton;
    cv::Mat boundary_skeleton_pruned;
    cv::Mat source_pixel_ridges;
    cv::Mat source_pixel_ridge_skeleton;
    cv::Mat boundary_skeleton_hybrid;
    cv::Mat cell_loops;
    cv::Mat cell_loops_connected;
    cv::Mat rings;
    std::vector<StageTiming> timings;
};

struct GraphEdge {
    int a = 0;
    int b = 0;
    float capacity = 0.0f;
    std::vector<cv::Point> pixels;
};

struct SkeletonGraph {
    std::vector<cv::Point2f> nodes;
    std::vector<GraphEdge> edges;
};

std::vector<cv::Point> skeleton_neighbors(const cv::Mat& skeleton,
                                          const cv::Point pixel) {
    std::vector<cv::Point> neighbors;
    neighbors.reserve(8);
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            const int x = pixel.x + dx;
            const int y = pixel.y + dy;
            if (x < 0 || x >= skeleton.cols || y < 0 || y >= skeleton.rows) {
                continue;
            }
            if (skeleton.at<std::uint8_t>(y, x) != 0) {
                neighbors.push_back(cv::Point(x, y));
            }
        }
    }
    return neighbors;
}

int skeleton_topological_branch_count(const cv::Mat& skeleton,
                                      const cv::Point pixel) {
    const auto get = [&](int y, int x) -> std::uint8_t {
        if (x < 0 || x >= skeleton.cols || y < 0 || y >= skeleton.rows) {
            return 0;
        }
        return skeleton.at<std::uint8_t>(y, x);
    };
    const std::uint8_t p[8] = {
        get(pixel.y - 1, pixel.x),     get(pixel.y - 1, pixel.x + 1),
        get(pixel.y, pixel.x + 1),     get(pixel.y + 1, pixel.x + 1),
        get(pixel.y + 1, pixel.x),     get(pixel.y + 1, pixel.x - 1),
        get(pixel.y, pixel.x - 1),     get(pixel.y - 1, pixel.x - 1),
    };
    return transition_count(p);
}

void add_unique_label(std::vector<int>& labels, int label) {
    if (label > 0 && std::find(labels.begin(), labels.end(), label) ==
                         labels.end()) {
        labels.push_back(label);
    }
}

SkeletonGraph extract_skeleton_graph(const cv::Mat& skeleton, const cv::Mat& dt) {
    CV_Assert(skeleton.type() == CV_8U);
    CV_Assert(dt.type() == CV_32F);

    cv::Mat skel;
    cv::threshold(skeleton, skel, 0, 255, cv::THRESH_BINARY);

    cv::Mat node_seed = cv::Mat::zeros(skel.size(), CV_8U);
    for (int y = 0; y < skel.rows; ++y) {
        for (int x = 0; x < skel.cols; ++x) {
            if (skel.at<std::uint8_t>(y, x) == 0) {
                continue;
            }
            const cv::Point pixel(x, y);
            const int degree =
                static_cast<int>(skeleton_neighbors(skel, pixel).size());
            const int branches = skeleton_topological_branch_count(skel, pixel);
            if (degree <= 1 || branches >= 3) {
                node_seed.at<std::uint8_t>(y, x) = 255;
            }
        }
    }

    cv::Mat node_labels;
    int num_nodes = cv::connectedComponents(node_seed, node_labels, 8, CV_32S);
    std::vector<cv::Point2d> node_sums(static_cast<std::size_t>(num_nodes));
    std::vector<int> node_counts(static_cast<std::size_t>(num_nodes), 0);
    std::vector<std::vector<cv::Point>> node_pixels(
        static_cast<std::size_t>(num_nodes));
    for (int y = 0; y < node_labels.rows; ++y) {
        for (int x = 0; x < node_labels.cols; ++x) {
            const int label = node_labels.at<int>(y, x);
            if (label <= 0) {
                continue;
            }
            node_sums[label] += cv::Point2d(x, y);
            ++node_counts[label];
            node_pixels[label].push_back(cv::Point(x, y));
        }
    }

    cv::Mat skeleton_components;
    const int num_skeleton_components =
        cv::connectedComponents(skel, skeleton_components, 8, CV_32S);
    std::vector<char> component_has_node(
        static_cast<std::size_t>(num_skeleton_components), 0);
    std::vector<cv::Point> component_first_pixel(
        static_cast<std::size_t>(num_skeleton_components), cv::Point(-1, -1));
    for (int y = 0; y < skel.rows; ++y) {
        for (int x = 0; x < skel.cols; ++x) {
            if (skel.at<std::uint8_t>(y, x) == 0) {
                continue;
            }
            const int component = skeleton_components.at<int>(y, x);
            if (component_first_pixel[component].x < 0) {
                component_first_pixel[component] = cv::Point(x, y);
            }
            if (node_labels.at<int>(y, x) > 0) {
                component_has_node[component] = 1;
            }
        }
    }

    for (int component = 1; component < num_skeleton_components; ++component) {
        if (component_has_node[component] != 0 ||
            component_first_pixel[component].x < 0) {
            continue;
        }
        const int label = num_nodes++;
        const cv::Point pixel = component_first_pixel[component];
        node_labels.at<int>(pixel.y, pixel.x) = label;
        node_sums.push_back(cv::Point2d(pixel.x, pixel.y));
        node_counts.push_back(1);
        node_pixels.push_back({pixel});
    }

    SkeletonGraph graph;
    graph.nodes.resize(static_cast<std::size_t>(num_nodes));
    for (int label = 1; label < num_nodes; ++label) {
        graph.nodes[label] =
            cv::Point2f(static_cast<float>(node_sums[label].x / node_counts[label]),
                        static_cast<float>(node_sums[label].y / node_counts[label]));
    }

    cv::Mat node_mask;
    cv::compare(node_labels, 0, node_mask, cv::CMP_GT);

    cv::Mat edge_mask = skel.clone();
    edge_mask.setTo(0, node_mask);

    cv::Mat edge_labels;
    const int num_edge_components =
        cv::connectedComponents(edge_mask, edge_labels, 8, CV_32S);
    std::vector<GraphEdge> edge_components(
        static_cast<std::size_t>(num_edge_components));
    std::vector<std::vector<int>> touched_nodes(
        static_cast<std::size_t>(num_edge_components));

    for (int y = 0; y < edge_labels.rows; ++y) {
        for (int x = 0; x < edge_labels.cols; ++x) {
            const int edge_label = edge_labels.at<int>(y, x);
            if (edge_label <= 0) {
                continue;
            }

            GraphEdge& edge = edge_components[edge_label];
            edge.pixels.push_back(cv::Point(x, y));
            if (edge.pixels.size() == 1) {
                edge.capacity = dt.at<float>(y, x);
            } else {
                edge.capacity = std::min(edge.capacity, dt.at<float>(y, x));
            }

            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const int nx = x + dx;
                    const int ny = y + dy;
                    if (nx < 0 || nx >= node_labels.cols || ny < 0 ||
                        ny >= node_labels.rows) {
                        continue;
                    }
                    const int node_label = node_labels.at<int>(ny, nx);
                    if (node_label > 0) {
                        add_unique_label(touched_nodes[edge_label], node_label);
                        edge.capacity =
                            std::min(edge.capacity, dt.at<float>(ny, nx));
                    }
                }
            }
        }
    }

    for (int edge_label = 1; edge_label < num_edge_components; ++edge_label) {
        GraphEdge& edge = edge_components[edge_label];
        if (edge.pixels.empty() || touched_nodes[edge_label].empty()) {
            continue;
        }

        std::sort(touched_nodes[edge_label].begin(),
                  touched_nodes[edge_label].end());
        edge.a = touched_nodes[edge_label].front();
        edge.b = touched_nodes[edge_label].size() > 1
                     ? touched_nodes[edge_label][1]
                     : touched_nodes[edge_label].front();
        for (const int node : {edge.a, edge.b}) {
            if (node <= 0 ||
                node >= static_cast<int>(node_pixels.size())) {
                continue;
            }
            for (const cv::Point pixel : node_pixels[node]) {
                edge.pixels.push_back(pixel);
                edge.capacity =
                    std::min(edge.capacity, dt.at<float>(pixel.y, pixel.x));
            }
        }
        graph.edges.push_back(std::move(edge));
    }

    cv::Mat covered = cv::Mat::zeros(skel.size(), CV_8U);
    for (const GraphEdge& edge : graph.edges) {
        for (const cv::Point pixel : edge.pixels) {
            covered.at<std::uint8_t>(pixel.y, pixel.x) = 255;
        }
    }
    cv::Mat uncovered;
    cv::bitwise_and(skel, ~covered, uncovered);
    cv::Mat uncovered_labels;
    const int num_uncovered =
        cv::connectedComponents(uncovered, uncovered_labels, 8, CV_32S);
    std::vector<GraphEdge> uncovered_edges(
        static_cast<std::size_t>(num_uncovered));
    for (int y = 0; y < uncovered.rows; ++y) {
        for (int x = 0; x < uncovered.cols; ++x) {
            const int label = uncovered_labels.at<int>(y, x);
            if (label <= 0) {
                continue;
            }
            GraphEdge& edge = uncovered_edges[label];
            edge.a = node_labels.at<int>(y, x);
            edge.b = edge.a;
            edge.pixels.push_back(cv::Point(x, y));
            if (edge.pixels.size() == 1) {
                edge.capacity = dt.at<float>(y, x);
            } else {
                edge.capacity = std::min(edge.capacity, dt.at<float>(y, x));
            }
        }
    }
    for (int label = 1; label < num_uncovered; ++label) {
        if (!uncovered_edges[label].pixels.empty()) {
            graph.edges.push_back(std::move(uncovered_edges[label]));
        }
    }

    return graph;
}

cv::Scalar deterministic_edge_color(int edge_index) {
    std::uint32_t value =
        0x9E3779B9u * static_cast<std::uint32_t>(edge_index + 1);
    int b = 96 + static_cast<int>(value & 0x7Fu);
    int g = 96 + static_cast<int>((value >> 8U) & 0x7Fu);
    int r = 96 + static_cast<int>((value >> 16U) & 0x7Fu);
    switch ((value >> 24U) % 3U) {
        case 0:
            b = 255;
            break;
        case 1:
            g = 255;
            break;
        default:
            r = 255;
            break;
    }
    return cv::Scalar(b, g, r);
}

void draw_graph_edge(cv::Mat& out, const GraphEdge& edge,
                     const cv::Scalar color) {
    for (const cv::Point pixel : edge.pixels) {
        if (out.channels() == 1) {
            out.at<std::uint8_t>(pixel.y, pixel.x) =
                static_cast<std::uint8_t>(color[0]);
        } else {
            out.at<cv::Vec3b>(pixel.y, pixel.x) =
                cv::Vec3b(static_cast<std::uint8_t>(color[0]),
                          static_cast<std::uint8_t>(color[1]),
                          static_cast<std::uint8_t>(color[2]));
        }
    }
}

cv::Mat render_graph_random_colors(const SkeletonGraph& graph, cv::Size size) {
    cv::Mat out(size, CV_8UC3, cv::Scalar(0, 0, 0));
    for (std::size_t i = 0; i < graph.edges.size(); ++i) {
        draw_graph_edge(out, graph.edges[i],
                        deterministic_edge_color(static_cast<int>(i)));
    }
    for (std::size_t label = 1; label < graph.nodes.size(); ++label) {
        cv::circle(out, graph.nodes[label], 3, cv::Scalar(255, 255, 255),
                   cv::FILLED, cv::LINE_8);
        cv::circle(out, graph.nodes[label], 4, cv::Scalar(0, 0, 0), 1,
                   cv::LINE_8);
    }
    return out;
}

cv::Mat render_graph_capacity(const SkeletonGraph& graph, cv::Size size) {
    float max_capacity = 0.0f;
    for (const GraphEdge& edge : graph.edges) {
        max_capacity = std::max(max_capacity, edge.capacity);
    }

    cv::Mat out(size, CV_8UC1, cv::Scalar(0));
    for (const GraphEdge& edge : graph.edges) {
        const int value =
            max_capacity > 0.0f
                ? static_cast<int>(std::lround(
                      std::clamp(edge.capacity / max_capacity, 0.0f, 1.0f) *
                      255.0f))
                : 0;
        draw_graph_edge(out, edge, cv::Scalar(value));
    }
    for (std::size_t label = 1; label < graph.nodes.size(); ++label) {
        cv::circle(out, graph.nodes[label], 3, cv::Scalar(255), cv::FILLED,
                   cv::LINE_8);
    }
    return out;
}

cv::Mat to_bgr_layer(const cv::Mat& image) {
    if (image.type() == CV_8UC3) {
        return image;
    }
    cv::Mat u8;
    if (image.depth() == CV_8U) {
        u8 = image.channels() == 1 ? image : cv::Mat();
    } else {
        double max_value = 0.0;
        cv::minMaxLoc(image, nullptr, &max_value);
        if (max_value > 0.0) {
            image.convertTo(u8, CV_8U, 255.0 / max_value);
        } else {
            u8 = cv::Mat::zeros(image.size(), CV_8U);
        }
    }
    cv::Mat bgr;
    cv::cvtColor(u8, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

cv::Mat labels_to_u16(const cv::Mat& labels, int max_label) {
    CV_Assert(labels.type() == CV_32S);

    cv::Mat out(labels.size(), CV_16U, cv::Scalar(0));
    if (max_label <= 0) {
        return out;
    }

    const double scale = max_label > 65535 ? 65535.0 / max_label : 1.0;
    cv::parallel_for_(cv::Range(0, labels.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            for (int x = 0; x < labels.cols; ++x) {
                const int label = labels.at<int>(y, x);
                if (label > 0) {
                    out.at<std::uint16_t>(y, x) =
                        static_cast<std::uint16_t>(std::lround(label * scale));
                }
            }
        }
    });
    return out;
}

ComponentVoronoiResult component_voronoi(const cv::Mat& binary) {
    CV_Assert(binary.type() == CV_8U);

    std::vector<StageTiming> timings;

    cv::Mat component_labels;
    int num_components = 0;
    {
        const TimingMark timing = start_timing();
        num_components = cv::connectedComponents(binary, component_labels, 8,
                                                 CV_32S);
        timings.push_back(
            finish_timing("component.connected_components", timing));
    }

    cv::Mat source_mask;
    {
        const TimingMark timing = start_timing();
        source_mask = cv::Mat(binary.size(), CV_8U, cv::Scalar(255));
        source_mask.setTo(0, binary);
        timings.push_back(finish_timing("component.source_mask", timing));
    }

    cv::Mat dt_to_component;
    cv::Mat source_pixel_labels;
    {
        const TimingMark timing = start_timing();
        cv::distanceTransform(source_mask, dt_to_component, source_pixel_labels,
                              cv::DIST_L2, cv::DIST_MASK_5,
                              cv::DIST_LABEL_PIXEL);
        timings.push_back(finish_timing("component.labeled_dt", timing));
    }

    std::vector<cv::Point> source_points;
    {
        const TimingMark timing = start_timing();
        source_points = label_source_points(source_mask, source_pixel_labels);
        timings.push_back(
            finish_timing("component.label_source_points", timing));
    }

    cv::Mat nearest_component(binary.size(), CV_32S, cv::Scalar(0));
    {
        const TimingMark timing = start_timing();
        cv::parallel_for_(cv::Range(0, binary.rows),
                          [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                for (int x = 0; x < binary.cols; ++x) {
                    if (binary.at<std::uint8_t>(y, x) != 0) {
                        nearest_component.at<int>(y, x) =
                            component_labels.at<int>(y, x);
                        continue;
                    }

                    const int source_label = source_pixel_labels.at<int>(y, x);
                    if (source_label <= 0 ||
                        source_label >=
                            static_cast<int>(source_points.size())) {
                        continue;
                    }
                    const cv::Point source = source_points[source_label];
                    if (source.x >= 0) {
                        nearest_component.at<int>(y, x) =
                            component_labels.at<int>(source.y, source.x);
                    }
                }
            }
        });
        timings.push_back(finish_timing("component.nearest_component", timing));
    }

    cv::Mat boundaries = cv::Mat::zeros(binary.size(), CV_8U);
    {
        const TimingMark timing = start_timing();
        cv::parallel_for_(cv::Range(1, binary.rows - 1),
                          [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                for (int x = 1; x < binary.cols - 1; ++x) {
                    if (binary.at<std::uint8_t>(y, x) != 0) {
                        continue;
                    }
                    const int center = nearest_component.at<int>(y, x);
                    if (center <= 0) {
                        continue;
                    }
                    bool touches_other = false;
                    for (int dy = -1; dy <= 1 && !touches_other; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx == 0 && dy == 0) {
                                continue;
                            }
                            const int other =
                                nearest_component.at<int>(y + dy, x + dx);
                            if (other > 0 && other != center) {
                                touches_other = true;
                                break;
                            }
                        }
                    }
                    if (touches_other) {
                        boundaries.at<std::uint8_t>(y, x) = 255;
                    }
                }
            }
        });
        timings.push_back(finish_timing("component.boundaries", timing));
    }

    cv::Mat boundary_skeleton;
    {
        const TimingMark timing = start_timing();
        boundary_skeleton = optimized_thinning(boundaries);
        timings.push_back(
            finish_timing("component.boundary_skeleton", timing));
    }

    cv::Mat boundary_skeleton_pruned;
    {
        const TimingMark timing = start_timing();
        boundary_skeleton_pruned =
            prune_short_low_dt_spurs(boundary_skeleton, dt_to_component);
        timings.push_back(finish_timing("component.prune_spurs", timing));
    }

    cv::Mat source_pixel_ridges;
    {
        const TimingMark timing = start_timing();
        source_pixel_ridges =
            source_pixel_label_ridges(source_mask, source_pixel_labels);
        timings.push_back(
            finish_timing("component.source_pixel_ridges", timing));
    }

    cv::Mat source_pixel_ridge_skeleton;
    {
        const TimingMark timing = start_timing();
        source_pixel_ridge_skeleton =
            cv::Mat::zeros(source_pixel_ridges.size(), CV_8U);
        timings.push_back(
            finish_timing("component.source_ridge_skeleton_skipped", timing));
    }

    cv::Mat cell_loops = cv::Mat::zeros(binary.size(), CV_8U);
    cv::Mat rings = cv::Mat::zeros(binary.size(), CV_8U);
    {
        const TimingMark timing = start_timing();
        timings.push_back(
            finish_timing("component.cell_ring_contours_skipped", timing));
    }

    cv::Mat connected_skeleton;
    {
        const TimingMark timing = start_timing();
        connected_skeleton = connect_clean_skeleton_with_source_ridges(
            boundary_skeleton_pruned, source_pixel_ridges, dt_to_component);
        timings.push_back(finish_timing("component.connect_ridges", timing));
    }
    cv::Mat cell_loops_connected;
    connected_skeleton.copyTo(cell_loops_connected);

    cv::Mat labels_u16;
    {
        const TimingMark timing = start_timing();
        labels_u16 = labels_to_u16(nearest_component, num_components - 1);
        timings.push_back(finish_timing("component.labels_to_u16", timing));
    }

    return {labels_u16,
            boundaries,
            boundary_skeleton,
            boundary_skeleton_pruned,
            source_pixel_ridges,
            source_pixel_ridge_skeleton,
            connected_skeleton,
            cell_loops,
            cell_loops_connected,
            rings,
            timings};
}

void write_image(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("failed to write image: " + path.string());
    }
}

struct NamedLayer {
    std::string name;
    cv::Mat image;
};

void write_named_layered_tiff(const fs::path& path,
                              const std::vector<NamedLayer>& layers) {
    TIFF* tiff = TIFFOpen(path.string().c_str(), "w");
    if (tiff == nullptr) {
        throw std::runtime_error("failed to open layered TIFF: " +
                                 path.string());
    }

    for (std::size_t layer_index = 0; layer_index < layers.size();
         ++layer_index) {
        const NamedLayer& layer = layers[layer_index];
        CV_Assert(layer.image.depth() == CV_8U);
        CV_Assert(layer.image.channels() == 1 || layer.image.channels() == 3);

        cv::Mat image;
        if (layer.image.isContinuous()) {
            image = layer.image;
        } else {
            image = layer.image.clone();
        }

        const int channels = image.channels();
        TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH,
                     static_cast<std::uint32_t>(image.cols));
        TIFFSetField(tiff, TIFFTAG_IMAGELENGTH,
                     static_cast<std::uint32_t>(image.rows));
        TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL,
                     static_cast<std::uint16_t>(channels));
        TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, static_cast<std::uint16_t>(8));
        TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
        TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC,
                     channels == 1 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB);
        TIFFSetField(tiff, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
        TIFFSetField(tiff, TIFFTAG_PAGENUMBER,
                     static_cast<std::uint16_t>(layer_index),
                     static_cast<std::uint16_t>(layers.size()));
        TIFFSetField(tiff, TIFFTAG_PAGENAME, layer.name.c_str());

        std::vector<std::uint8_t> rgb_row(
            static_cast<std::size_t>(image.cols * channels));
        for (int y = 0; y < image.rows; ++y) {
            const std::uint8_t* row = image.ptr<std::uint8_t>(y);
            void* row_to_write = const_cast<std::uint8_t*>(row);
            if (channels == 3) {
                for (int x = 0; x < image.cols; ++x) {
                    rgb_row[static_cast<std::size_t>(x * 3 + 0)] =
                        row[x * 3 + 2];
                    rgb_row[static_cast<std::size_t>(x * 3 + 1)] =
                        row[x * 3 + 1];
                    rgb_row[static_cast<std::size_t>(x * 3 + 2)] =
                        row[x * 3 + 0];
                }
                row_to_write = rgb_row.data();
            }

            if (TIFFWriteScanline(tiff, row_to_write,
                                  static_cast<std::uint32_t>(y), 0) < 0) {
                TIFFClose(tiff);
                throw std::runtime_error("failed to write layered TIFF row: " +
                                         path.string());
            }
        }

        if (TIFFWriteDirectory(tiff) != 1) {
            TIFFClose(tiff);
            throw std::runtime_error("failed to write layered TIFF directory: " +
                                     path.string());
        }
    }

    TIFFClose(tiff);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const TimingMark total_start = start_timing();
        std::vector<StageTiming> timings;

        const Args args = parse_args(argc, argv);
        const fs::path workdir = fs::current_path();

        cv::Mat gray;
        {
            const TimingMark timing = start_timing();
            gray = load_grayscale(args.input);
            timings.push_back(finish_timing("input_load", timing));
        }

        cv::Mat binary;
        {
            const TimingMark timing = start_timing();
            binary = binarize_fixed_threshold(gray);
            timings.push_back(finish_timing("binarize", timing));
        }

        cv::Mat white_domain(binary.size(), CV_8U, cv::Scalar(255));
        white_domain.setTo(0, binary);

        cv::Mat dt;
        {
            const TimingMark timing = start_timing();
            cv::distanceTransform(white_domain, dt, cv::DIST_L2,
                                  cv::DIST_MASK_PRECISE);
            timings.push_back(finish_timing("precise_dt", timing));
        }

        ComponentVoronoiResult component_voronoi_result;
        {
            const TimingMark timing = start_timing();
            component_voronoi_result = component_voronoi(binary);
            timings.push_back(finish_timing("component_voronoi", timing));
            timings.insert(timings.end(),
                           component_voronoi_result.timings.begin(),
                           component_voronoi_result.timings.end());
        }

        SkeletonGraph graph;
        {
            const TimingMark timing = start_timing();
            graph = extract_skeleton_graph(
                component_voronoi_result.cell_loops_connected, dt);
            timings.push_back(finish_timing("graph_extract", timing));
        }

        cv::Mat graph_random_colors;
        {
            const TimingMark timing = start_timing();
            graph_random_colors = render_graph_random_colors(graph, binary.size());
            timings.push_back(finish_timing("graph_random_color_render", timing));
        }

        cv::Mat graph_capacity;
        {
            const TimingMark timing = start_timing();
            graph_capacity = render_graph_capacity(graph, binary.size());
            timings.push_back(finish_timing("graph_capacity_render", timing));
        }

        cv::Mat contour_loops;
        {
            const TimingMark timing = start_timing();
            contour_loops = binary_contour_loops(binary);
            timings.push_back(finish_timing("binary_contour_loops", timing));
        }

        cv::Mat dt_u16;
        {
            const TimingMark timing = start_timing();
            dt_u16 = normalized_dt_u16(dt);
            timings.push_back(finish_timing("dt_normalize", timing));
        }

        const std::string stem = args.input.stem().string();
        {
            const TimingMark timing = start_timing();
            write_image(workdir / (stem + "_binary.tif"), binary);
            write_image(workdir / (stem + "_dt.tif"), dt_u16);
            write_image(workdir / (stem + "_component_voronoi_labels.tif"),
                        component_voronoi_result.labels_u16);
            write_image(workdir / (stem + "_component_voronoi_boundaries.tif"),
                        component_voronoi_result.boundaries);
            write_image(workdir /
                            (stem + "_component_voronoi_boundary_skeleton.tif"),
                        component_voronoi_result.boundary_skeleton);
            write_image(
                workdir /
                    (stem + "_component_voronoi_boundary_skeleton_pruned.tif"),
                component_voronoi_result.boundary_skeleton_pruned);
            write_image(workdir /
                            (stem + "_source_pixel_voronoi_ridges.tif"),
                        component_voronoi_result.source_pixel_ridges);
            write_image(workdir /
                            (stem + "_source_pixel_voronoi_ridge_skeleton.tif"),
                        component_voronoi_result.source_pixel_ridge_skeleton);
            write_image(
                workdir /
                    (stem + "_component_voronoi_boundary_skeleton_hybrid.tif"),
                component_voronoi_result.boundary_skeleton_hybrid);
            write_image(workdir / (stem + "_component_voronoi_cell_loops.tif"),
                        component_voronoi_result.cell_loops);
            write_image(
                workdir /
                    (stem + "_component_voronoi_cell_loops_connected.tif"),
                component_voronoi_result.cell_loops_connected);
            write_image(workdir / (stem + "_component_voronoi_rings.tif"),
                        component_voronoi_result.rings);
            write_image(workdir / (stem + "_binary_contour_loops.tif"),
                        contour_loops);
            write_image(workdir / (stem + "_graph_random_edges.tif"),
                        graph_random_colors);
            write_image(workdir / (stem + "_graph_capacity.tif"),
                        graph_capacity);
            timings.push_back(finish_timing("write_regular_outputs", timing));
        }

        const std::vector<NamedLayer> layered_tiff = {
            {"dt", to_bgr_layer(dt_u16)},
            {"loops", to_bgr_layer(component_voronoi_result.boundary_skeleton_pruned)},
            {"loops_connected",
             to_bgr_layer(component_voronoi_result.cell_loops_connected)},
            {"graph_random_edges", graph_random_colors},
            {"graph_capacity", to_bgr_layer(graph_capacity)},
        };
        {
            const TimingMark timing = start_timing();
            write_named_layered_tiff(workdir / (stem + "_layers.tif"),
                                     layered_tiff);
            timings.push_back(finish_timing("write_layered_tiff", timing));
        }
        timings.push_back(finish_timing("total", total_start));

        print_stage_timings(timings);
        std::cout << "Graph:\n"
                  << "  graph_nodes: " << (graph.nodes.size() > 0
                                               ? graph.nodes.size() - 1
                                               : 0)
                  << "\n"
                  << "  graph_edges: " << graph.edges.size() << "\n";

        std::cout << "Wrote:\n"
                  << "  " << (workdir / (stem + "_binary.tif")) << "\n"
                  << "  " << (workdir / (stem + "_dt.tif")) << "\n"
                  << "  "
                  << (workdir / (stem + "_component_voronoi_labels.tif"))
                  << "\n"
                  << "  "
                  << (workdir / (stem + "_component_voronoi_boundaries.tif"))
                  << "\n"
                  << "  "
                  << (workdir /
                      (stem + "_component_voronoi_boundary_skeleton.tif"))
                  << "\n"
                  << "  "
                  << (workdir /
                      (stem +
                       "_component_voronoi_boundary_skeleton_pruned.tif"))
                  << "\n"
                  << "  "
                  << (workdir / (stem + "_source_pixel_voronoi_ridges.tif"))
                  << "\n"
                  << "  "
                  << (workdir /
                      (stem + "_source_pixel_voronoi_ridge_skeleton.tif"))
                  << "\n"
                  << "  "
                  << (workdir /
                      (stem +
                       "_component_voronoi_boundary_skeleton_hybrid.tif"))
                  << "\n"
                  << "  "
                  << (workdir / (stem + "_component_voronoi_cell_loops.tif"))
                  << "\n"
                  << "  "
                  << (workdir /
                      (stem + "_component_voronoi_cell_loops_connected.tif"))
                  << "\n"
                  << "  "
                  << (workdir / (stem + "_component_voronoi_rings.tif"))
                  << "\n"
                  << "  " << (workdir / (stem + "_binary_contour_loops.tif"))
                  << "\n"
                  << "  " << (workdir / (stem + "_graph_random_edges.tif"))
                  << "\n"
                  << "  " << (workdir / (stem + "_graph_capacity.tif"))
                  << "\n"
                  << "  " << (workdir / (stem + "_layers.tif"))
                  << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
