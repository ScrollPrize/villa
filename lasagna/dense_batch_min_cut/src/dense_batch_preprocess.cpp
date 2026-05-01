#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
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

struct ComponentVoronoiResult {
    cv::Mat labels_u16;
    cv::Mat boundaries;
    cv::Mat boundary_skeleton;
    cv::Mat boundary_skeleton_pruned;
    cv::Mat cell_loops;
    cv::Mat cell_loops_connected;
    cv::Mat rings;
};

cv::Mat labels_to_u16(const cv::Mat& labels, int max_label) {
    CV_Assert(labels.type() == CV_32S);

    cv::Mat out(labels.size(), CV_16U, cv::Scalar(0));
    if (max_label <= 0) {
        return out;
    }

    const double scale = max_label > 65535 ? 65535.0 / max_label : 1.0;
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            const int label = labels.at<int>(y, x);
            if (label > 0) {
                out.at<std::uint16_t>(y, x) =
                    static_cast<std::uint16_t>(std::lround(label * scale));
            }
        }
    }
    return out;
}

ComponentVoronoiResult component_voronoi(const cv::Mat& binary) {
    CV_Assert(binary.type() == CV_8U);

    cv::Mat component_labels;
    const int num_components = cv::connectedComponents(binary, component_labels, 8,
                                                       CV_32S);

    cv::Mat source_mask(binary.size(), CV_8U, cv::Scalar(255));
    source_mask.setTo(0, binary);

    cv::Mat dt_to_component;
    cv::Mat source_pixel_labels;
    cv::distanceTransform(source_mask, dt_to_component, source_pixel_labels,
                          cv::DIST_L2, cv::DIST_MASK_5, cv::DIST_LABEL_PIXEL);

    const std::vector<cv::Point> source_points =
        label_source_points(source_mask, source_pixel_labels);

    cv::Mat nearest_component(binary.size(), CV_32S, cv::Scalar(0));
    for (int y = 0; y < binary.rows; ++y) {
        for (int x = 0; x < binary.cols; ++x) {
            if (binary.at<std::uint8_t>(y, x) != 0) {
                nearest_component.at<int>(y, x) = component_labels.at<int>(y, x);
                continue;
            }

            const int source_label = source_pixel_labels.at<int>(y, x);
            if (source_label <= 0 ||
                source_label >= static_cast<int>(source_points.size())) {
                continue;
            }
            const cv::Point source = source_points[source_label];
            if (source.x >= 0) {
                nearest_component.at<int>(y, x) =
                    component_labels.at<int>(source.y, source.x);
            }
        }
    }

    cv::Mat boundaries = cv::Mat::zeros(binary.size(), CV_8U);
    for (int y = 1; y < binary.rows - 1; ++y) {
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
                    const int other = nearest_component.at<int>(y + dy, x + dx);
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

    const cv::Mat boundary_skeleton = zhang_suen_thinning(boundaries);
    const cv::Mat boundary_skeleton_pruned =
        prune_short_low_dt_spurs(boundary_skeleton, dt_to_component);

    cv::Mat cell_loops = cv::Mat::zeros(binary.size(), CV_8U);
    cv::Mat rings = cv::Mat::zeros(binary.size(), CV_8U);
    for (int component = 1; component < num_components; ++component) {
        cv::Mat cell_mask;
        cv::compare(nearest_component, component, cell_mask, cv::CMP_EQ);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(cell_mask, contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_NONE);
        for (std::size_t i = 0; i < contours.size(); ++i) {
            if (std::abs(cv::contourArea(contours[i])) < 8.0) {
                continue;
            }
            cv::drawContours(cell_loops, contours, static_cast<int>(i),
                             cv::Scalar(255), 1);
        }

        cv::Mat ring_mask = cell_mask.clone();
        ring_mask.setTo(0, component_labels == component);

        std::vector<std::vector<cv::Point>> ring_contours;
        std::vector<cv::Vec4i> ring_hierarchy;
        cv::findContours(ring_mask, ring_contours, ring_hierarchy, cv::RETR_TREE,
                         cv::CHAIN_APPROX_NONE);
        for (std::size_t i = 0; i < ring_contours.size(); ++i) {
            if (std::abs(cv::contourArea(ring_contours[i])) < 8.0) {
                continue;
            }

            const cv::Rect bounds = cv::boundingRect(ring_contours[i]);
            const bool touches_image_border =
                bounds.x <= 0 || bounds.y <= 0 ||
                bounds.x + bounds.width >= binary.cols ||
                bounds.y + bounds.height >= binary.rows;
            if (touches_image_border && ring_hierarchy[i][3] < 0) {
                continue;
            }

            cv::drawContours(rings, ring_contours, static_cast<int>(i),
                             cv::Scalar(255), 1);
        }
    }

    cv::Mat cell_loops_connected;
    cv::bitwise_or(cell_loops, boundary_skeleton_pruned, cell_loops_connected);

    return {labels_to_u16(nearest_component, num_components - 1), boundaries,
            boundary_skeleton, boundary_skeleton_pruned, cell_loops,
            cell_loops_connected, rings};
}

void write_image(const fs::path& path, const cv::Mat& image) {
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("failed to write image: " + path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const fs::path workdir = fs::current_path();

        const cv::Mat gray = load_grayscale(args.input);
        const cv::Mat binary = binarize_fixed_threshold(gray);

        cv::Mat white_domain(binary.size(), CV_8U, cv::Scalar(255));
        white_domain.setTo(0, binary);

        cv::Mat dt;
        cv::distanceTransform(white_domain, dt, cv::DIST_L2,
                              cv::DIST_MASK_PRECISE);

        const auto component_voronoi_start = Clock::now();
        const ComponentVoronoiResult component_voronoi_result =
            component_voronoi(binary);
        const auto component_voronoi_end = Clock::now();

        const auto contour_loops_start = Clock::now();
        const cv::Mat contour_loops = binary_contour_loops(binary);
        const auto contour_loops_end = Clock::now();

        const cv::Mat dt_u16 = normalized_dt_u16(dt);

        const std::string stem = args.input.stem().string();
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
        write_image(workdir / (stem + "_component_voronoi_cell_loops.tif"),
                    component_voronoi_result.cell_loops);
        write_image(workdir /
                        (stem + "_component_voronoi_cell_loops_connected.tif"),
                    component_voronoi_result.cell_loops_connected);
        write_image(workdir / (stem + "_component_voronoi_rings.tif"),
                    component_voronoi_result.rings);
        write_image(workdir / (stem + "_binary_contour_loops.tif"),
                    contour_loops);

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
                  << "\n";
        std::cout << "Timings:\n"
                  << "  component_voronoi_ms: "
                  << std::chrono::duration<double, std::milli>(
                         component_voronoi_end - component_voronoi_start)
                         .count()
                  << "\n"
                  << "  binary_contour_loops_ms: "
                  << std::chrono::duration<double, std::milli>(
                         contour_loops_end - contour_loops_start)
                         .count()
                  << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
