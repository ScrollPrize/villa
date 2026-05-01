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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
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
    bool has_source = false;
    cv::Point source{-1, -1};
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " -i <image> [--source x,y]\n";
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if ((key == "--input" || key == "-i") && i + 1 < argc) {
            args.input = argv[++i];
        } else if (key == "--source" && i + 1 < argc) {
            const std::string value(argv[++i]);
            const std::size_t comma = value.find(',');
            if (comma == std::string::npos) {
                throw std::runtime_error("--source must be formatted as x,y");
            }
            args.source.x = std::stoi(value.substr(0, comma));
            args.source.y = std::stoi(value.substr(comma + 1));
            args.has_source = true;
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
    cv::Mat node_mask;
    cv::Mat edge_mask;
    int skeleton_pixels = 0;
    int node_pixels = 0;
    int edge_path_pixels = 0;
    int unique_edge_pixels = 0;
    int missing_pixels = 0;
};

struct GraphComponentStats {
    int id = 0;
    int nodes = 0;
    int edges = 0;
    int self_loop_edges = 0;
    int one_endpoint_edges = 0;
    int zero_endpoint_edges = 0;
};

struct GraphConnectivityStats {
    std::vector<GraphComponentStats> components;
    int valid_nodes = 0;
    int valid_edges = 0;
    int self_loop_edges = 0;
    int one_endpoint_edges = 0;
    int zero_endpoint_edges = 0;
    int skeleton_pixels = 0;
    int node_pixels = 0;
    int edge_path_pixels = 0;
    int unique_edge_pixels = 0;
    int missing_pixels = 0;
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

std::vector<cv::Point> sorted_skeleton_neighbors(const cv::Mat& skeleton,
                                                 const cv::Point pixel) {
    std::vector<cv::Point> neighbors = skeleton_neighbors(skeleton, pixel);
    std::sort(neighbors.begin(), neighbors.end(),
              [](const cv::Point& a, const cv::Point& b) {
        if (a.y != b.y) {
            return a.y < b.y;
        }
        return a.x < b.x;
    });
    return neighbors;
}

SkeletonGraph extract_skeleton_graph(const cv::Mat& skeleton, const cv::Mat& dt) {
    CV_Assert(skeleton.type() == CV_8U);
    CV_Assert(dt.type() == CV_32F);

    cv::Mat skel;
    cv::threshold(skeleton, skel, 0, 255, cv::THRESH_BINARY);

    SkeletonGraph graph;
    graph.nodes.push_back(cv::Point2f(-1.0f, -1.0f));
    graph.skeleton_pixels = cv::countNonZero(skel);

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
    const int initial_node_count =
        cv::connectedComponents(node_seed, node_labels, 8, CV_32S);
    std::vector<cv::Point2d> node_sums(
        static_cast<std::size_t>(initial_node_count));
    std::vector<int> node_counts(
        static_cast<std::size_t>(initial_node_count), 0);
    std::vector<std::vector<cv::Point>> node_pixels(
        static_cast<std::size_t>(initial_node_count));
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
    graph.nodes.resize(static_cast<std::size_t>(initial_node_count));
    for (int label = 1; label < initial_node_count; ++label) {
        graph.nodes[label] = cv::Point2f(
            static_cast<float>(node_sums[label].x / node_counts[label]),
            static_cast<float>(node_sums[label].y / node_counts[label]));
    }

    cv::compare(node_labels, 0, graph.node_mask, cv::CMP_GT);

    const auto is_skeleton = [&](const cv::Point pixel) {
        return pixel.x >= 0 && pixel.x < skel.cols && pixel.y >= 0 &&
               pixel.y < skel.rows &&
               skel.at<std::uint8_t>(pixel.y, pixel.x) != 0;
    };
    const auto is_node = [&](const cv::Point pixel) {
        return is_skeleton(pixel) &&
               node_labels.at<int>(pixel.y, pixel.x) > 0;
    };
    const auto is_edge_pixel = [&](const cv::Point pixel) {
        return is_skeleton(pixel) &&
               node_labels.at<int>(pixel.y, pixel.x) == 0;
    };
    const auto four_connected = [](const cv::Point a, const cv::Point b) {
        return std::abs(a.x - b.x) + std::abs(a.y - b.y) == 1;
    };
    const auto make_node = [&](const cv::Point pixel) {
        int& label = node_labels.at<int>(pixel.y, pixel.x);
        if (label > 0) {
            return label;
        }
        label = static_cast<int>(graph.nodes.size());
        graph.nodes.push_back(cv::Point2f(static_cast<float>(pixel.x),
                                          static_cast<float>(pixel.y)));
        node_pixels.push_back({pixel});
        graph.node_mask.at<std::uint8_t>(pixel.y, pixel.x) = 255;
        return label;
    };
    const auto adjacent_nodes = [&](const cv::Point pixel) {
        std::vector<int> nodes;
        for (const cv::Point neighbor : sorted_skeleton_neighbors(skel, pixel)) {
            add_unique_label(nodes, node_labels.at<int>(neighbor.y, neighbor.x));
        }
        std::sort(nodes.begin(), nodes.end());
        return nodes;
    };
    const auto choose_next = [&](const cv::Point current,
                                 const std::vector<cv::Point>& candidates) {
        std::vector<cv::Point> four_candidates;
        for (const cv::Point candidate : candidates) {
            if (four_connected(current, candidate)) {
                four_candidates.push_back(candidate);
            }
        }
        if (four_candidates.size() == 1) {
            return four_candidates.front();
        }
        if (candidates.size() == 1) {
            return candidates.front();
        }
        return cv::Point(-1, -1);
    };

    cv::Mat visited_edges = cv::Mat::zeros(skel.size(), CV_8U);
    const auto trace_edge = [&](int start_node, const cv::Point start_pixel,
                                const cv::Point first_pixel) {
        GraphEdge edge;
        edge.a = start_node;
        edge.b = 0;
        edge.capacity = std::numeric_limits<float>::max();

        cv::Point previous = start_pixel;
        cv::Point current = first_pixel;
        while (is_skeleton(current)) {
            if (is_node(current)) {
                edge.b = node_labels.at<int>(current.y, current.x);
                break;
            }

            if (visited_edges.at<std::uint8_t>(current.y, current.x) != 0) {
                break;
            }

            std::vector<int> nodes = adjacent_nodes(current);
            nodes.erase(std::remove(nodes.begin(), nodes.end(), start_node),
                        nodes.end());
            if (!nodes.empty() && !edge.pixels.empty()) {
                edge.b = nodes.front();
                break;
            }

            std::vector<cv::Point> candidates;
            for (const cv::Point neighbor :
                 sorted_skeleton_neighbors(skel, current)) {
                if (neighbor == previous || is_node(neighbor) ||
                    !is_edge_pixel(neighbor) ||
                    visited_edges.at<std::uint8_t>(neighbor.y, neighbor.x) !=
                        0) {
                    continue;
                }
                candidates.push_back(neighbor);
            }

            const cv::Point next = choose_next(current, candidates);
            if (next.x < 0 && !candidates.empty()) {
                edge.b = make_node(current);
                break;
            }

            visited_edges.at<std::uint8_t>(current.y, current.x) = 255;
            edge.pixels.push_back(current);
            edge.capacity =
                std::min(edge.capacity, dt.at<float>(current.y, current.x));

            if (next.x < 0) {
                nodes = adjacent_nodes(current);
                if (edge.pixels.size() > 1 &&
                    std::find(nodes.begin(), nodes.end(), start_node) !=
                        nodes.end()) {
                    edge.b = start_node;
                } else {
                    edge.b = make_node(current);
                    edge.pixels.pop_back();
                    visited_edges.at<std::uint8_t>(current.y, current.x) = 0;
                }
                break;
            }

            previous = current;
            current = next;
        }

        if (edge.b > 0 && !edge.pixels.empty()) {
            if (edge.capacity == std::numeric_limits<float>::max()) {
                edge.capacity = 0.0f;
            }
            graph.edges.push_back(std::move(edge));
        }
    };

    for (std::size_t node = 1; node < graph.nodes.size(); ++node) {
        for (const cv::Point node_pixel : node_pixels[node]) {
            for (const cv::Point neighbor :
                 sorted_skeleton_neighbors(skel, node_pixel)) {
                if (is_edge_pixel(neighbor) &&
                    visited_edges.at<std::uint8_t>(neighbor.y, neighbor.x) ==
                        0) {
                    trace_edge(static_cast<int>(node), node_pixel, neighbor);
                }
            }
        }
    }

    for (int y = 0; y < skel.rows; ++y) {
        for (int x = 0; x < skel.cols; ++x) {
            const cv::Point start(x, y);
            if (!is_edge_pixel(start) ||
                visited_edges.at<std::uint8_t>(y, x) != 0) {
                continue;
            }
            const int node = make_node(start);
            visited_edges.at<std::uint8_t>(y, x) = 255;
            std::vector<cv::Point> candidates;
            for (const cv::Point neighbor :
                 sorted_skeleton_neighbors(skel, start)) {
                if (is_edge_pixel(neighbor) &&
                    visited_edges.at<std::uint8_t>(neighbor.y, neighbor.x) ==
                        0) {
                    candidates.push_back(neighbor);
                }
            }
            if (candidates.empty()) {
                continue;
            }
            const cv::Point first = choose_next(start, candidates);
            trace_edge(node, start,
                       first.x >= 0 ? first : candidates.front());
        }
    }

    graph.node_pixels = cv::countNonZero(graph.node_mask);
    graph.edge_mask = cv::Mat::zeros(skel.size(), CV_8U);
    graph.edge_path_pixels = 0;
    for (const GraphEdge& edge : graph.edges) {
        graph.edge_path_pixels += static_cast<int>(edge.pixels.size());
        for (const cv::Point pixel : edge.pixels) {
            if (graph.node_mask.at<std::uint8_t>(pixel.y, pixel.x) == 0) {
                graph.edge_mask.at<std::uint8_t>(pixel.y, pixel.x) = 255;
            }
        }
    }
    graph.unique_edge_pixels = cv::countNonZero(graph.edge_mask);
    cv::Mat covered;
    cv::bitwise_or(graph.node_mask, graph.edge_mask, covered);
    cv::Mat missing;
    cv::bitwise_and(skel, ~covered, missing);
    graph.missing_pixels = cv::countNonZero(missing);

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

cv::Mat render_graph_edges_random_colors(const SkeletonGraph& graph,
                                         cv::Size size) {
    cv::Mat out(size, CV_8UC3, cv::Scalar(0, 0, 0));
    for (std::size_t i = 0; i < graph.edges.size(); ++i) {
        draw_graph_edge(out, graph.edges[i],
                        deterministic_edge_color(static_cast<int>(i)));
    }
    return out;
}

cv::Mat render_graph_nodes(const SkeletonGraph& graph, cv::Size size) {
    cv::Mat out(size, CV_8UC1, cv::Scalar(0));
    for (std::size_t label = 1; label < graph.nodes.size(); ++label) {
        cv::circle(out, graph.nodes[label], 3, cv::Scalar(255), cv::FILLED,
                   cv::LINE_8);
    }
    return out;
}

GraphConnectivityStats graph_connectivity_stats(const SkeletonGraph& graph) {
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
        void unite(int a, int b) {
            a = find(a);
            b = find(b);
            if (a != b) {
                parent[b] = a;
            }
        }
    };

    const int node_count = static_cast<int>(graph.nodes.size());
    DisjointSet sets(node_count);
    const auto valid_node = [&](int node) {
        return node > 0 && node < node_count;
    };

    GraphConnectivityStats stats;
    stats.valid_nodes = std::max(0, node_count - 1);
    stats.valid_edges = static_cast<int>(graph.edges.size());
    stats.skeleton_pixels = graph.skeleton_pixels;
    stats.node_pixels = graph.node_pixels;
    stats.edge_path_pixels = graph.edge_path_pixels;
    stats.unique_edge_pixels = graph.unique_edge_pixels;
    stats.missing_pixels = graph.missing_pixels;

    for (const GraphEdge& edge : graph.edges) {
        const bool valid_a = valid_node(edge.a);
        const bool valid_b = valid_node(edge.b);
        if (valid_a && valid_b && edge.a != edge.b) {
            sets.unite(edge.a, edge.b);
        }
    }

    std::vector<int> root_to_component(static_cast<std::size_t>(node_count), -1);
    for (int node = 1; node < node_count; ++node) {
        const int root = sets.find(node);
        if (root_to_component[root] < 0) {
            root_to_component[root] =
                static_cast<int>(stats.components.size());
            GraphComponentStats component;
            component.id = root_to_component[root];
            stats.components.push_back(component);
        }
        ++stats.components[root_to_component[root]].nodes;
    }

    auto add_edge_to_component = [&](int node, const GraphEdge& edge) {
        const int root = sets.find(node);
        if (root_to_component[root] < 0) {
            root_to_component[root] =
                static_cast<int>(stats.components.size());
            GraphComponentStats component;
            component.id = root_to_component[root];
            stats.components.push_back(component);
        }
        GraphComponentStats& component =
            stats.components[root_to_component[root]];
        ++component.edges;
        if (edge.a == edge.b) {
            ++component.self_loop_edges;
        }
    };

    for (const GraphEdge& edge : graph.edges) {
        const bool valid_a = valid_node(edge.a);
        const bool valid_b = valid_node(edge.b);
        if (valid_a && valid_b) {
            add_edge_to_component(edge.a, edge);
            if (edge.a == edge.b) {
                ++stats.self_loop_edges;
            }
        } else if (valid_a || valid_b) {
            add_edge_to_component(valid_a ? edge.a : edge.b, edge);
            ++stats.one_endpoint_edges;
            const int root = sets.find(valid_a ? edge.a : edge.b);
            ++stats.components[root_to_component[root]].one_endpoint_edges;
        } else {
            ++stats.zero_endpoint_edges;
        }
    }

    std::sort(stats.components.begin(), stats.components.end(),
              [](const GraphComponentStats& a,
                 const GraphComponentStats& b) {
        if (a.nodes != b.nodes) {
            return a.nodes > b.nodes;
        }
        return a.edges > b.edges;
    });
    for (int i = 0; i < static_cast<int>(stats.components.size()); ++i) {
        stats.components[i].id = i;
    }
    return stats;
}

void write_graph_connectivity_report(const fs::path& path,
                                     const GraphConnectivityStats& stats) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write graph report: " +
                                 path.string());
    }
    out << "graph_components: " << stats.components.size() << "\n";
    out << "valid_nodes: " << stats.valid_nodes << "\n";
    out << "valid_edges: " << stats.valid_edges << "\n";
    out << "self_loop_edges: " << stats.self_loop_edges << "\n";
    out << "one_endpoint_edges: " << stats.one_endpoint_edges << "\n";
    out << "zero_endpoint_edges: " << stats.zero_endpoint_edges << "\n";
    out << "skeleton_pixels: " << stats.skeleton_pixels << "\n";
    out << "node_pixels: " << stats.node_pixels << "\n";
    out << "edge_path_pixels: " << stats.edge_path_pixels << "\n";
    out << "unique_edge_pixels: " << stats.unique_edge_pixels << "\n";
    out << "missing_pixels: " << stats.missing_pixels << "\n";
    out << "\n";
    out << "component,nodes,edges,self_loop_edges,one_endpoint_edges,zero_endpoint_edges\n";
    for (const GraphComponentStats& component : stats.components) {
        out << component.id << "," << component.nodes << ","
            << component.edges << "," << component.self_loop_edges << ","
            << component.one_endpoint_edges << ","
            << component.zero_endpoint_edges << "\n";
    }
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

struct DinicEdge {
    int to = 0;
    int rev = 0;
    double cap = 0.0;
};

class Dinic {
   public:
    explicit Dinic(int n) : graph_(static_cast<std::size_t>(n)) {}

    void add_edge(int from, int to, double cap) {
        DinicEdge forward{to, static_cast<int>(graph_[to].size()), cap};
        DinicEdge reverse{from, static_cast<int>(graph_[from].size()), 0.0};
        graph_[from].push_back(forward);
        graph_[to].push_back(reverse);
    }

    double max_flow(int source, int sink) {
        double flow = 0.0;
        while (build_levels(source, sink)) {
            next_.assign(graph_.size(), 0);
            while (true) {
                const double pushed = push(source, sink, kFlowInf);
                if (pushed <= kFlowEpsilon) {
                    break;
                }
                flow += pushed;
            }
        }
        return flow;
    }

   private:
    static constexpr double kFlowInf = 1.0e18;
    static constexpr double kFlowEpsilon = 1.0e-9;

    bool build_levels(int source, int sink) {
        level_.assign(graph_.size(), -1);
        std::queue<int> queue;
        level_[source] = 0;
        queue.push(source);
        while (!queue.empty()) {
            const int v = queue.front();
            queue.pop();
            for (const DinicEdge& edge : graph_[v]) {
                if (edge.cap > kFlowEpsilon && level_[edge.to] < 0) {
                    level_[edge.to] = level_[v] + 1;
                    queue.push(edge.to);
                }
            }
        }
        return level_[sink] >= 0;
    }

    double push(int v, int sink, double flow) {
        if (v == sink) {
            return flow;
        }
        for (int& i = next_[v]; i < static_cast<int>(graph_[v].size()); ++i) {
            DinicEdge& edge = graph_[v][i];
            if (edge.cap <= kFlowEpsilon || level_[edge.to] != level_[v] + 1) {
                continue;
            }
            const double pushed =
                push(edge.to, sink, std::min(flow, edge.cap));
            if (pushed <= kFlowEpsilon) {
                continue;
            }
            edge.cap -= pushed;
            graph_[edge.to][edge.rev].cap += pushed;
            return pushed;
        }
        return 0.0;
    }

    std::vector<std::vector<DinicEdge>> graph_;
    std::vector<int> level_;
    std::vector<int> next_;
};

struct DenseFlowResult {
    cv::Mat dense_flow;
    cv::Mat dense_flow_u16;
    cv::Mat graph_edge_flow;
    cv::Mat graph_source_edges;
    int source_edges = 0;
    int seeded_nodes = 0;
    float finite_edge_flow_min = 0.0f;
    float finite_edge_flow_max = 0.0f;
    int finite_edge_flows = 0;
    std::vector<StageTiming> timings;
};

std::vector<cv::Point> unique_edge_pixels(const std::vector<cv::Point>& pixels) {
    std::vector<cv::Point> unique = pixels;
    std::sort(unique.begin(), unique.end(), [](const cv::Point& a,
                                               const cv::Point& b) {
        if (a.y != b.y) {
            return a.y < b.y;
        }
        return a.x < b.x;
    });
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    return unique;
}

std::vector<cv::Point> order_edge_pixels(const GraphEdge& edge,
                                         const SkeletonGraph& graph) {
    std::vector<cv::Point> pixels = unique_edge_pixels(edge.pixels);
    if (pixels.size() <= 2) {
        return pixels;
    }

    cv::Rect bounds = cv::boundingRect(pixels);
    cv::Mat local_index(bounds.height, bounds.width, CV_32S, cv::Scalar(-1));
    for (int i = 0; i < static_cast<int>(pixels.size()); ++i) {
        local_index.at<int>(pixels[i].y - bounds.y, pixels[i].x - bounds.x) = i;
    }

    std::vector<int> degree(pixels.size(), 0);
    const std::array<cv::Point, 8> kDirs = {
        {{-1, -1}, {0, -1}, {1, -1}, {-1, 0},
         {1, 0},   {-1, 1}, {0, 1},  {1, 1}}};
    for (int i = 0; i < static_cast<int>(pixels.size()); ++i) {
        for (const cv::Point dir : kDirs) {
            const cv::Point next = pixels[i] + dir;
            if (!bounds.contains(next)) {
                continue;
            }
            if (local_index.at<int>(next.y - bounds.y, next.x - bounds.x) >= 0) {
                ++degree[i];
            }
        }
    }

    const auto distance_to_node = [&](int pixel_index, int node) {
        if (node <= 0 || node >= static_cast<int>(graph.nodes.size())) {
            return std::numeric_limits<double>::max();
        }
        const cv::Point2f center = graph.nodes[node];
        const double dx = static_cast<double>(pixels[pixel_index].x) - center.x;
        const double dy = static_cast<double>(pixels[pixel_index].y) - center.y;
        return dx * dx + dy * dy;
    };

    int start = 0;
    int end = -1;
    if (edge.a != edge.b) {
        double best_start = std::numeric_limits<double>::max();
        double best_end = std::numeric_limits<double>::max();
        for (int i = 0; i < static_cast<int>(pixels.size()); ++i) {
            const double da = distance_to_node(i, edge.a);
            const double db = distance_to_node(i, edge.b);
            if (da < best_start) {
                best_start = da;
                start = i;
            }
            if (db < best_end) {
                best_end = db;
                end = i;
            }
        }
    } else {
        for (int i = 0; i < static_cast<int>(degree.size()); ++i) {
            if (degree[i] <= 1) {
                start = i;
                break;
            }
        }
    }

    std::vector<char> visited(pixels.size(), 0);
    std::vector<cv::Point> ordered;
    ordered.reserve(pixels.size());
    int current = start;
    int previous = -1;
    while (current >= 0 && !visited[current]) {
        visited[current] = 1;
        ordered.push_back(pixels[current]);

        int next_index = -1;
        double best_score = std::numeric_limits<double>::max();
        for (const cv::Point dir : kDirs) {
            const cv::Point next = pixels[current] + dir;
            if (!bounds.contains(next)) {
                continue;
            }
            const int candidate =
                local_index.at<int>(next.y - bounds.y, next.x - bounds.x);
            if (candidate < 0 || visited[candidate] || candidate == previous) {
                continue;
            }
            double score = 0.0;
            if (end >= 0) {
                const double dx = pixels[candidate].x - pixels[end].x;
                const double dy = pixels[candidate].y - pixels[end].y;
                score = dx * dx + dy * dy;
            } else {
                score = degree[candidate];
            }
            if (score < best_score) {
                best_score = score;
                next_index = candidate;
            }
        }
        previous = current;
        current = next_index;
    }

    if (ordered.size() == pixels.size()) {
        return ordered;
    }
    for (int i = 0; i < static_cast<int>(pixels.size()); ++i) {
        if (!visited[i]) {
            ordered.push_back(pixels[i]);
        }
    }
    return ordered;
}

DenseFlowResult compute_dense_source_flow(const cv::Mat& white_domain,
                                          const cv::Mat& dt,
                                          const SkeletonGraph& graph,
                                          const cv::Point source) {
    CV_Assert(white_domain.type() == CV_8U);
    CV_Assert(dt.type() == CV_32F);

    if (source.x < 0 || source.x >= white_domain.cols || source.y < 0 ||
        source.y >= white_domain.rows) {
        throw std::runtime_error("--source is outside the image");
    }
    if (white_domain.at<std::uint8_t>(source.y, source.x) == 0) {
        throw std::runtime_error("--source must be inside the white distance domain");
    }

    std::vector<StageTiming> timings;
    constexpr float kDenseFlowInf = 1.0e9f;
    constexpr double kGraphFlowInf = 1.0e12;
    const int node_count = static_cast<int>(graph.nodes.size());

    cv::Mat graph_edge_mask(white_domain.size(), CV_8U, cv::Scalar(0));
    cv::Mat graph_edge_index(white_domain.size(), CV_32S, cv::Scalar(-1));
    cv::Mat graph_pixel_flow(white_domain.size(), CV_32F, cv::Scalar(0));
    for (int edge_index = 0; edge_index < static_cast<int>(graph.edges.size());
         ++edge_index) {
        const GraphEdge& edge = graph.edges[edge_index];
        for (const cv::Point pixel : edge.pixels) {
            if (pixel.x >= 0 && pixel.x < graph_edge_mask.cols &&
                pixel.y >= 0 && pixel.y < graph_edge_mask.rows) {
                graph_edge_mask.at<std::uint8_t>(pixel.y, pixel.x) = 255;
                graph_edge_index.at<int>(pixel.y, pixel.x) = edge_index;
            }
        }
    }

    std::vector<char> seeded_nodes(static_cast<std::size_t>(node_count), 0);
    std::vector<char> source_edges(graph.edges.size(), 0);
    {
        const TimingMark timing = start_timing();
        cv::Mat domain_without_graph = white_domain.clone();
        domain_without_graph.setTo(0, graph_edge_mask);

        cv::Mat region_labels;
        cv::connectedComponents(domain_without_graph, region_labels, 4, CV_32S);
        const int source_region = region_labels.at<int>(source.y, source.x);
        const std::array<cv::Point, 4> kCardinalDirs = {
            {{0, -1}, {-1, 0}, {1, 0}, {0, 1}}};

        for (int edge_index = 0; edge_index < static_cast<int>(graph.edges.size());
             ++edge_index) {
            const GraphEdge& edge = graph.edges[edge_index];
            bool touches_source = false;
            for (const cv::Point pixel : edge.pixels) {
                if (pixel == source) {
                    touches_source = true;
                    break;
                }
                for (const cv::Point dir : kCardinalDirs) {
                    const int x = pixel.x + dir.x;
                    const int y = pixel.y + dir.y;
                    if (x < 0 || x >= region_labels.cols || y < 0 ||
                        y >= region_labels.rows) {
                        continue;
                    }
                    if (region_labels.at<int>(y, x) == source_region &&
                        source_region > 0) {
                        touches_source = true;
                        break;
                    }
                }
                if (touches_source) {
                    break;
                }
            }
            if (!touches_source) {
                continue;
            }
            source_edges[edge_index] = 1;
        }

        cv::Mat source_edge_mask(white_domain.size(), CV_8U, cv::Scalar(0));
        for (int edge_index = 0;
             edge_index < static_cast<int>(graph.edges.size()); ++edge_index) {
            if (source_edges[edge_index] == 0) {
                continue;
            }
            for (const cv::Point pixel : graph.edges[edge_index].pixels) {
                source_edge_mask.at<std::uint8_t>(pixel.y, pixel.x) = 255;
            }
        }

        cv::Mat dilated_source_edges;
        cv::dilate(source_edge_mask, dilated_source_edges, cv::Mat());
        for (int y = 0; y < dilated_source_edges.rows; ++y) {
            for (int x = 0; x < dilated_source_edges.cols; ++x) {
                if (dilated_source_edges.at<std::uint8_t>(y, x) == 0 ||
                    graph_edge_mask.at<std::uint8_t>(y, x) == 0) {
                    continue;
                }
                const int edge_index = graph_edge_index.at<int>(y, x);
                if (edge_index >= 0) {
                    source_edges[edge_index] = 1;
                }
            }
        }

        for (int edge_index = 0;
             edge_index < static_cast<int>(graph.edges.size()); ++edge_index) {
            if (source_edges[edge_index] == 0) {
                continue;
            }
            const GraphEdge& edge = graph.edges[edge_index];
            if (edge.a > 0 && edge.a < node_count) {
                seeded_nodes[edge.a] = 1;
            }
            if (edge.b > 0 && edge.b < node_count) {
                seeded_nodes[edge.b] = 1;
            }
        }
        timings.push_back(finish_timing("source_region_detect", timing));
    }

    int source_edge_count = 0;
    for (char value : source_edges) {
        if (value != 0) {
            ++source_edge_count;
        }
    }
    int seeded_node_count = 0;
    for (int node = 1; node < node_count; ++node) {
        if (seeded_nodes[node] != 0) {
            ++seeded_node_count;
        }
    }

    const auto max_flow_to_node = [&](int target, int removed_edge) {
        if (target <= 0 || target >= node_count) {
            return 0.0;
        }
        if (seeded_nodes[target] != 0) {
            return kGraphFlowInf;
        }

        Dinic flow(node_count);
        constexpr int kSuperSource = 0;
        for (int node = 1; node < node_count; ++node) {
            if (seeded_nodes[node] != 0) {
                flow.add_edge(kSuperSource, node, kGraphFlowInf);
            }
        }
        for (int edge_index = 0; edge_index < static_cast<int>(graph.edges.size());
             ++edge_index) {
            if (edge_index == removed_edge) {
                continue;
            }
            const GraphEdge& edge = graph.edges[edge_index];
            if (edge.a <= 0 || edge.b <= 0 || edge.a >= node_count ||
                edge.b >= node_count || edge.a == edge.b) {
                continue;
            }
            const double capacity =
                source_edges[edge_index] != 0
                    ? kGraphFlowInf
                    : std::max(0.0f, edge.capacity);
            flow.add_edge(edge.a, edge.b, capacity);
            flow.add_edge(edge.b, edge.a, capacity);
        }
        return flow.max_flow(kSuperSource, target);
    };

    std::vector<double> node_flow(static_cast<std::size_t>(node_count), 0.0);
    std::vector<float> edge_flow(graph.edges.size(), 0.0f);
    {
        const TimingMark timing = start_timing();
        for (int node = 1; node < node_count; ++node) {
            node_flow[node] = max_flow_to_node(node, -1);
        }
        for (int edge_index = 0; edge_index < static_cast<int>(graph.edges.size());
             ++edge_index) {
            const GraphEdge& edge = graph.edges[edge_index];
            double value = 0.0;
            if (source_edges[edge_index] != 0) {
                value = kGraphFlowInf;
            } else {
                if (edge.a > 0 && edge.a < node_count) {
                    value = std::max(value, node_flow[edge.a]);
                }
                if (edge.b > 0 && edge.b < node_count) {
                    value = std::max(value, node_flow[edge.b]);
                }
            }
            edge_flow[edge_index] =
                value >= kGraphFlowInf * 0.5
                    ? kDenseFlowInf
                    : static_cast<float>(std::max(0.0, value));
        }
        timings.push_back(finish_timing("graph_node_maxflow", timing));
    }

    {
        const TimingMark timing = start_timing();
        for (int edge_index = 0; edge_index < static_cast<int>(graph.edges.size());
             ++edge_index) {
            const GraphEdge& edge = graph.edges[edge_index];
            std::vector<cv::Point> ordered = order_edge_pixels(edge, graph);
            if (ordered.empty()) {
                continue;
            }

            double side_a = source_edges[edge_index] != 0
                                ? kGraphFlowInf
                                : max_flow_to_node(edge.a, edge_index);
            double side_b = 0.0;
            if (edge.b != edge.a) {
                side_b = source_edges[edge_index] != 0
                             ? kGraphFlowInf
                             : max_flow_to_node(edge.b, edge_index);
            }

            std::vector<float> left(ordered.size(), 0.0f);
            std::vector<float> right(ordered.size(), 0.0f);
            float min_dt = static_cast<float>(
                std::min(side_a, static_cast<double>(kDenseFlowInf)));
            for (std::size_t i = 0; i < ordered.size(); ++i) {
                min_dt = std::min(min_dt, dt.at<float>(ordered[i].y, ordered[i].x));
                left[i] = min_dt;
            }
            min_dt = static_cast<float>(
                std::min(side_b, static_cast<double>(kDenseFlowInf)));
            for (std::size_t i = ordered.size(); i-- > 0;) {
                min_dt = std::min(min_dt, dt.at<float>(ordered[i].y, ordered[i].x));
                right[i] = min_dt;
            }

            for (std::size_t i = 0; i < ordered.size(); ++i) {
                const cv::Point pixel = ordered[i];
                const float value =
                    std::min(kDenseFlowInf, left[i] + right[i]);
                graph_pixel_flow.at<float>(pixel.y, pixel.x) =
                    std::max(graph_pixel_flow.at<float>(pixel.y, pixel.x),
                             value);
            }
        }
        timings.push_back(finish_timing("graph_edge_point_flow", timing));
    }

    cv::Mat dense_flow(white_domain.size(), CV_32F, cv::Scalar(0));
    {
        const TimingMark timing = start_timing();
        cv::Mat graph_source_mask(white_domain.size(), CV_8U, cv::Scalar(255));
        graph_source_mask.setTo(0, graph_edge_mask);
        cv::Mat nearest_dt;
        cv::Mat nearest_graph_pixel;
        cv::distanceTransform(graph_source_mask, nearest_dt, nearest_graph_pixel,
                              cv::DIST_L2, cv::DIST_MASK_5,
                              cv::DIST_LABEL_PIXEL);
        const std::vector<cv::Point> graph_source_points =
            label_source_points(graph_source_mask, nearest_graph_pixel);

        cv::parallel_for_(cv::Range(0, white_domain.rows),
                          [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                for (int x = 0; x < white_domain.cols; ++x) {
                    if (white_domain.at<std::uint8_t>(y, x) == 0) {
                        continue;
                    }
                    const int label = nearest_graph_pixel.at<int>(y, x);
                    if (label <= 0 ||
                        label >= static_cast<int>(graph_source_points.size())) {
                        continue;
                    }
                    const cv::Point graph_pixel = graph_source_points[label];
                    if (graph_pixel.x < 0) {
                        continue;
                    }
                    const float edge_value =
                        graph_pixel_flow.at<float>(graph_pixel.y,
                                                   graph_pixel.x);
                    dense_flow.at<float>(y, x) =
                        std::min(dt.at<float>(y, x), edge_value);
                }
            }
        });
        timings.push_back(finish_timing("dense_flow_nearest_edge", timing));
    }

    cv::Mat dense_flow_u16 = normalized_dt_u16(dense_flow);
    cv::Mat graph_edge_flow(white_domain.size(), CV_8U, cv::Scalar(0));
    cv::Mat graph_source_edges(white_domain.size(), CV_8U, cv::Scalar(0));
    double max_edge_flow = 0.0;
    float finite_edge_flow_min = std::numeric_limits<float>::max();
    float finite_edge_flow_max = 0.0f;
    int finite_edge_flow_count = 0;
    for (float value : edge_flow) {
        if (value < kDenseFlowInf * 0.5f) {
            finite_edge_flow_min = std::min(finite_edge_flow_min, value);
            finite_edge_flow_max = std::max(finite_edge_flow_max, value);
            ++finite_edge_flow_count;
            max_edge_flow = std::max(max_edge_flow, static_cast<double>(value));
        }
    }
    if (finite_edge_flow_count == 0) {
        finite_edge_flow_min = 0.0f;
    }
    if (max_edge_flow <= 0.0) {
        max_edge_flow = 1.0;
    }
    for (int edge_index = 0; edge_index < static_cast<int>(graph.edges.size());
         ++edge_index) {
        const int value = edge_flow[edge_index] >= kDenseFlowInf * 0.5f
                              ? 255
                              : static_cast<int>(std::lround(
                                    std::clamp(edge_flow[edge_index] /
                                                   static_cast<float>(max_edge_flow),
                                               0.0f, 1.0f) *
                                    255.0f));
        draw_graph_edge(graph_edge_flow, graph.edges[edge_index],
                        cv::Scalar(value));
        if (source_edges[edge_index] != 0) {
            draw_graph_edge(graph_source_edges, graph.edges[edge_index],
                            cv::Scalar(255));
        }
    }

    return {dense_flow,
            dense_flow_u16,
            graph_edge_flow,
            graph_source_edges,
            source_edge_count,
            seeded_node_count,
            finite_edge_flow_min,
            finite_edge_flow_max,
            finite_edge_flow_count,
            timings};
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
    cell_loops_connected = optimized_thinning(connected_skeleton);

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

        GraphConnectivityStats graph_stats;
        {
            const TimingMark timing = start_timing();
            graph_stats = graph_connectivity_stats(graph);
            timings.push_back(finish_timing("graph_connectivity_stats", timing));
        }

        cv::Mat graph_random_colors;
        {
            const TimingMark timing = start_timing();
            graph_random_colors = render_graph_random_colors(graph, binary.size());
            timings.push_back(finish_timing("graph_random_color_render", timing));
        }

        cv::Mat graph_edges_random_colors;
        {
            const TimingMark timing = start_timing();
            graph_edges_random_colors =
                render_graph_edges_random_colors(graph, binary.size());
            timings.push_back(
                finish_timing("graph_edge_only_random_color_render", timing));
        }

        cv::Mat graph_nodes;
        {
            const TimingMark timing = start_timing();
            graph_nodes = render_graph_nodes(graph, binary.size());
            timings.push_back(finish_timing("graph_node_render", timing));
        }

        cv::Mat graph_capacity;
        {
            const TimingMark timing = start_timing();
            graph_capacity = render_graph_capacity(graph, binary.size());
            timings.push_back(finish_timing("graph_capacity_render", timing));
        }

        DenseFlowResult dense_flow_result;
        bool has_dense_flow = false;
        if (args.has_source) {
            dense_flow_result =
                compute_dense_source_flow(white_domain, dt, graph, args.source);
            timings.insert(timings.end(), dense_flow_result.timings.begin(),
                           dense_flow_result.timings.end());
            has_dense_flow = true;
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
            write_image(workdir / (stem + "_graph_edges_random.tif"),
                        graph_edges_random_colors);
            write_image(workdir / (stem + "_graph_nodes.tif"),
                        graph_nodes);
            write_image(workdir / (stem + "_graph_capacity.tif"),
                        graph_capacity);
            write_graph_connectivity_report(
                workdir / (stem + "_graph_components.txt"), graph_stats);
            timings.push_back(finish_timing("write_regular_outputs", timing));
        }

        if (has_dense_flow) {
            const TimingMark timing = start_timing();
            write_image(workdir / (stem + "_dense_flow.tif"),
                        dense_flow_result.dense_flow);
            write_image(workdir / (stem + "_dense_flow_u16.tif"),
                        dense_flow_result.dense_flow_u16);
            write_image(workdir / (stem + "_graph_edge_flow.tif"),
                        dense_flow_result.graph_edge_flow);
            write_image(workdir / (stem + "_graph_source_edges.tif"),
                        dense_flow_result.graph_source_edges);
            timings.push_back(finish_timing("dense_flow_write", timing));
        }

        std::vector<NamedLayer> layered_tiff = {
            {"binary_threshold", to_bgr_layer(binary)},
            {"dt", to_bgr_layer(dt_u16)},
            {"loops", to_bgr_layer(component_voronoi_result.boundary_skeleton_pruned)},
            {"loops_connected",
             to_bgr_layer(component_voronoi_result.cell_loops_connected)},
            {"graph_random_edges", graph_random_colors},
            {"graph_edges_random", graph_edges_random_colors},
            {"graph_nodes", to_bgr_layer(graph_nodes)},
            {"graph_capacity", to_bgr_layer(graph_capacity)},
        };
        if (has_dense_flow) {
            layered_tiff.push_back(
                {"dense_flow", to_bgr_layer(dense_flow_result.dense_flow_u16)});
            layered_tiff.push_back(
                {"graph_edge_flow",
                 to_bgr_layer(dense_flow_result.graph_edge_flow)});
            layered_tiff.push_back(
                {"graph_source_edges",
                 to_bgr_layer(dense_flow_result.graph_source_edges)});
        }
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
        std::cout << "Graph connectivity:\n"
                  << "  graph_components: "
                  << graph_stats.components.size() << "\n"
                  << "  self_loop_edges: "
                  << graph_stats.self_loop_edges << "\n"
                  << "  one_endpoint_edges: "
                  << graph_stats.one_endpoint_edges << "\n"
                  << "  zero_endpoint_edges: "
                  << graph_stats.zero_endpoint_edges << "\n"
                  << "  skeleton_pixels: "
                  << graph_stats.skeleton_pixels << "\n"
                  << "  node_pixels: " << graph_stats.node_pixels << "\n"
                  << "  unique_edge_pixels: "
                  << graph_stats.unique_edge_pixels << "\n"
                  << "  edge_path_pixels: "
                  << graph_stats.edge_path_pixels << "\n"
                  << "  missing_pixels: "
                  << graph_stats.missing_pixels << "\n";
        const int components_to_print =
            std::min<int>(10, graph_stats.components.size());
        for (int i = 0; i < components_to_print; ++i) {
            const GraphComponentStats& component = graph_stats.components[i];
            std::cout << "  component_" << component.id
                      << ": nodes=" << component.nodes
                      << " edges=" << component.edges
                      << " self_loops=" << component.self_loop_edges
                      << " one_endpoint=" << component.one_endpoint_edges
                      << "\n";
        }
        if (has_dense_flow) {
            std::cout << "Dense flow:\n"
                      << "  source_edges: "
                      << dense_flow_result.source_edges << " / "
                      << graph.edges.size() << "\n"
                      << "  seeded_nodes: "
                      << dense_flow_result.seeded_nodes << " / "
                      << (graph.nodes.size() > 0 ? graph.nodes.size() - 1 : 0)
                      << "\n"
                      << "  finite_edge_flows: "
                      << dense_flow_result.finite_edge_flows << "\n"
                      << "  finite_edge_flow_min: "
                      << dense_flow_result.finite_edge_flow_min << "\n"
                      << "  finite_edge_flow_max: "
                      << dense_flow_result.finite_edge_flow_max << "\n";
        }

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
                  << "  " << (workdir / (stem + "_graph_edges_random.tif"))
                  << "\n"
                  << "  " << (workdir / (stem + "_graph_nodes.tif"))
                  << "\n"
                  << "  " << (workdir / (stem + "_graph_capacity.tif"))
                  << "\n"
                  << "  " << (workdir / (stem + "_graph_components.txt"))
                  << "\n"
                  << "  " << (workdir / (stem + "_layers.tif")) << "\n";
        if (has_dense_flow) {
            std::cout << "  " << (workdir / (stem + "_dense_flow.tif"))
                      << "\n"
                      << "  " << (workdir / (stem + "_dense_flow_u16.tif"))
                      << "\n"
                      << "  " << (workdir / (stem + "_graph_edge_flow.tif"))
                      << "\n"
                      << "  " << (workdir / (stem + "_graph_source_edges.tif"))
                      << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
