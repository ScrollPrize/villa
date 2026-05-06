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
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double kFixedThreshold = 110.0;
constexpr float kMinComponentRidgeRadius = 2.0f;
constexpr double kMinBoundaryAngleDegrees = 120.0;
constexpr float kCapacityScale = 2.0f;
constexpr bool kEnableLegacyVoronoiTreeDensification = false;
constexpr bool kEnableLegacyDenseDtAscent = false;
constexpr bool kWriteReferenceDenseBacktrackNn = false;
constexpr bool kEnablePixelDenseBacktrackFlood = false;
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

struct CoutSilencer {
    std::streambuf* old = nullptr;
    std::ostringstream sink;
    explicit CoutSilencer(bool active) {
        if (active) {
            old = std::cout.rdbuf(sink.rdbuf());
        }
    }
    ~CoutSilencer() {
        if (old != nullptr) {
            std::cout.rdbuf(old);
        }
    }
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

float capacity_from_dt(float value) {
    return kCapacityScale * value;
}

void print_stage_timings(const std::vector<StageTiming>& timings) {
    constexpr int kStageWidth = 44;
    constexpr int kNumericWidth = 14;
    constexpr int kSpeedupWidth = 20;
    std::vector<StageTiming> aggregated;
    for (const StageTiming& timing : timings) {
        auto it = std::find_if(aggregated.begin(), aggregated.end(),
                               [&](const StageTiming& other) {
                                   return other.name == timing.name;
                               });
        if (it == aggregated.end()) {
            aggregated.push_back(timing);
        } else {
            it->elapsed_ms += timing.elapsed_ms;
            it->cpu_ms += timing.cpu_ms;
        }
    }
    auto is_total = [](const StageTiming& timing) {
        return timing.name == "total";
    };
    auto is_io_stage = [](const StageTiming& timing) {
        return timing.name == "input_load" ||
               timing.name == "write_regular_outputs" ||
               timing.name == "dense_flow_write" ||
               timing.name == "write_layered_tiff" ||
               timing.name == "carrier_debug_render" ||
               timing.name == "carrier_debug_render_skipped" ||
               timing.name == "dense_grid.debug_paths" ||
               timing.name == "dense_grid.debug_paths_skipped" ||
               timing.name == "dense_backtrack_debug_render" ||
               timing.name == "dense_backtrack_stall_diagnostics" ||
               timing.name == "tree_path_debug_render";
    };
    auto child_parent_name = [](const std::string& name) -> std::string {
        if (name.rfind("component.connect_ridges.", 0) == 0) {
            return "component.connect_ridges";
        }
        if (name.rfind("component.", 0) == 0) {
            return "component_voronoi";
        }
        if (name.rfind("dense_grid.", 0) == 0) {
            return "dense_backtrack_grid_carrier";
        }
        return {};
    };
    auto display_stage_name = [&](const std::string& name) -> std::string {
        if (name.rfind("component.connect_ridges.", 0) == 0) {
            return "        " +
                   name.substr(std::string("component.connect_ridges.").size());
        }
        if (name.rfind("component.", 0) == 0) {
            return "    " + name.substr(std::string("component.").size());
        }
        if (name.rfind("dense_grid.", 0) == 0) {
            return "    " + name.substr(std::string("dense_grid.").size());
        }
        return name;
    };

    double total_elapsed_ms = 0.0;
    double total_cpu_ms = 0.0;
    double compute_elapsed_ms = 0.0;
    double compute_cpu_ms = 0.0;
    double io_elapsed_ms = 0.0;
    double io_cpu_ms = 0.0;
    for (const StageTiming& timing : aggregated) {
        if (is_total(timing)) {
            total_elapsed_ms = timing.elapsed_ms;
            total_cpu_ms = timing.cpu_ms;
        } else if (is_io_stage(timing)) {
            io_elapsed_ms += timing.elapsed_ms;
            io_cpu_ms += timing.cpu_ms;
        } else {
            compute_elapsed_ms += timing.elapsed_ms;
            compute_cpu_ms += timing.cpu_ms;
        }
    }
    if (total_elapsed_ms <= 0.0 && !timings.empty()) {
        total_elapsed_ms = timings.back().elapsed_ms;
    }
    if (total_elapsed_ms > 0.0) {
        compute_elapsed_ms = std::max(0.0, total_elapsed_ms - io_elapsed_ms);
    }
    if (total_cpu_ms > 0.0) {
        compute_cpu_ms = std::max(0.0, total_cpu_ms - io_cpu_ms);
    }
    if (compute_elapsed_ms <= 0.0) {
        compute_elapsed_ms = total_elapsed_ms;
    }
    if (io_elapsed_ms <= 0.0) {
        io_elapsed_ms = 1.0;
    }

    auto print_summary = [&](const std::string& name, const double elapsed_ms,
                             const double cpu_ms) {
        const double utilization =
            elapsed_ms > 0.0 ? cpu_ms / elapsed_ms : 0.0;
        std::cout << "  " << std::left << std::setw(kStageWidth) << name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(kNumericWidth) << 100.0
                  << std::setw(kNumericWidth) << elapsed_ms
                  << std::setw(kNumericWidth) << cpu_ms
                  << std::setw(kNumericWidth) << utilization
                  << std::setw(kSpeedupWidth) << 1.0
                  << "\n";
    };

    auto print_table = [&](const std::string& title, const double denom_ms,
                           const double summary_cpu_ms, const bool want_io) {
        std::cout << title << ":\n"
                  << "  " << std::left << std::setw(kStageWidth) << "stage"
                  << std::right << std::setw(kNumericWidth) << "runtime_%"
                  << std::setw(kNumericWidth) << "elapsed_ms"
                  << std::setw(kNumericWidth) << "cpu_ms"
                  << std::setw(kNumericWidth) << "cpu/elapsed"
                  << std::setw(kSpeedupWidth) << "perfect_par_gain_%"
                  << "\n";
        std::cout << "  "
                  << std::string(kStageWidth + 4 * kNumericWidth +
                                     kSpeedupWidth,
                                 '-')
                  << "\n";
        std::vector<std::uint8_t> printed(aggregated.size(), 0);
        const auto stage_in_table = [&](const std::string& name) {
            return std::find_if(aggregated.begin(), aggregated.end(),
                                [&](const StageTiming& timing) {
                                    return timing.name == name &&
                                           !is_total(timing) &&
                                           is_io_stage(timing) == want_io;
                                }) != aggregated.end();
        };
        const auto print_row = [&](const StageTiming& timing,
                                   const std::string& name) {
            const double runtime_fraction =
                denom_ms > 0.0 ? timing.elapsed_ms / denom_ms : 0.0;
            const double runtime_percent =
                100.0 * runtime_fraction;
            const double utilization =
                timing.elapsed_ms > 0.0 ? timing.cpu_ms / timing.elapsed_ms
                                        : 0.0;
            const double perfect_parallel_speedup =
                runtime_fraction < 1.0
                    ? 1.0 / std::max(1.0e-12, 1.0 - runtime_fraction)
                    : std::numeric_limits<double>::infinity();
            const double perfect_parallel_gain_percent =
                100.0 * (perfect_parallel_speedup - 1.0);
            std::cout << "  " << std::left << std::setw(kStageWidth)
                      << name
                      << std::right << std::fixed << std::setprecision(2)
                      << std::setw(kNumericWidth) << runtime_percent
                      << std::setw(kNumericWidth) << timing.elapsed_ms
                      << std::setw(kNumericWidth) << timing.cpu_ms
                      << std::setw(kNumericWidth) << utilization
                      << std::setw(kSpeedupWidth)
                      << perfect_parallel_gain_percent
                      << "\n";
        };
        for (std::size_t i = 0; i < aggregated.size(); ++i) {
            const StageTiming& timing = aggregated[i];
            if (printed[i] || is_total(timing) ||
                is_io_stage(timing) != want_io) {
                continue;
            }
            const std::string parent = child_parent_name(timing.name);
            if (!parent.empty() && stage_in_table(parent)) {
                continue;
            }
            print_row(timing, display_stage_name(timing.name));
            printed[i] = 1;

            for (std::size_t child_i = 0; child_i < aggregated.size();
                 ++child_i) {
                if (printed[child_i] ||
                    is_io_stage(aggregated[child_i]) != want_io ||
                    child_parent_name(aggregated[child_i].name) !=
                        timing.name) {
                    continue;
                }
                print_row(aggregated[child_i],
                          display_stage_name(aggregated[child_i].name));
                printed[child_i] = 1;

                for (std::size_t grandchild_i = 0;
                     grandchild_i < aggregated.size(); ++grandchild_i) {
                    if (printed[grandchild_i] ||
                        is_io_stage(aggregated[grandchild_i]) != want_io ||
                        child_parent_name(aggregated[grandchild_i].name) !=
                            aggregated[child_i].name) {
                        continue;
                    }
                    print_row(aggregated[grandchild_i],
                              display_stage_name(
                                  aggregated[grandchild_i].name));
                    printed[grandchild_i] = 1;
                }
            }
        }
        print_summary(want_io ? "io_debug_total" : "compute_total",
                      denom_ms, summary_cpu_ms);
    };

    std::cout << "Timings:\n"
              << "  opencv_threads=" << cv::getNumThreads()
              << " hardware_threads="
              << std::thread::hardware_concurrency() << "\n";
    print_table("  compute", compute_elapsed_ms, compute_cpu_ms, false);
    print_table("  io_debug", io_elapsed_ms, io_cpu_ms, true);
    for (const StageTiming& timing : aggregated) {
        if (is_total(timing)) {
            const double utilization =
                timing.elapsed_ms > 0.0 ? timing.cpu_ms / timing.elapsed_ms
                                        : 0.0;
            std::cout << "  total_elapsed_ms=" << std::fixed
                      << std::setprecision(2) << timing.elapsed_ms
                      << " total_cpu_ms=" << timing.cpu_ms
                      << " total_cpu/elapsed=" << utilization << "\n";
            break;
        }
    }
}

struct Args {
    fs::path input;
    bool has_source = false;
    cv::Point source{-1, -1};
    int grid_step = 50;
    float backtrack_distance = 10.0f;
    float flow_weight_one = 100.0f;
    float flow_weight_zero = 20.0f;
    int compute_repeats = 1;
    bool write_outputs = true;
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " -i <image> [--source x,y] [--grid-step pixels]"
              << " [--backtrack-distance pixels]"
              << " [--flow-weight-one value] [--flow-weight-zero value]"
              << " [--compute-repeats n] [--no-write]\n";
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
        } else if (key == "--grid-step" && i + 1 < argc) {
            args.grid_step = std::max(1, std::stoi(argv[++i]));
        } else if (key == "--backtrack-distance" && i + 1 < argc) {
            args.backtrack_distance =
                std::max(0.0f, std::stof(argv[++i]));
        } else if (key == "--flow-weight-one" && i + 1 < argc) {
            args.flow_weight_one = std::stof(argv[++i]);
        } else if (key == "--flow-weight-zero" && i + 1 < argc) {
            args.flow_weight_zero = std::stof(argv[++i]);
        } else if (key == "--compute-repeats" && i + 1 < argc) {
            args.compute_repeats = std::max(1, std::stoi(argv[++i]));
        } else if (key == "--no-write") {
            args.write_outputs = false;
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
    cv::Mat median;
    cv::medianBlur(u8, median, 3);
    cv::Mat white_domain;
    cv::threshold(median, white_domain, kFixedThreshold, 255, cv::THRESH_BINARY);

    cv::Mat binary;
    cv::bitwise_not(white_domain, binary);
    return binary;
}

cv::Mat keep_source_white_component(const cv::Mat& binary,
                                    const cv::Point source) {
    CV_Assert(binary.type() == CV_8U);
    if (source.x < 0 || source.x >= binary.cols || source.y < 0 ||
        source.y >= binary.rows) {
        throw std::runtime_error("--source is outside the image");
    }

    cv::Mat white_domain(binary.size(), CV_8U, cv::Scalar(255));
    white_domain.setTo(0, binary);
    if (white_domain.at<std::uint8_t>(source.y, source.x) == 0) {
        throw std::runtime_error("--source must be inside the white distance domain");
    }

    cv::Mat labels;
    const int component_count =
        cv::connectedComponents(white_domain, labels, 8, CV_32S);
    const int source_label = labels.at<int>(source.y, source.x);
    cv::Mat filtered(binary.size(), CV_8U, cv::Scalar(255));
    if (source_label <= 0 || source_label >= component_count) {
        return filtered;
    }
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            if (labels.at<int>(y, x) == source_label) {
                filtered.at<std::uint8_t>(y, x) = 0;
            }
        }
    }
    return filtered;
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
    double max_label_value = 0.0;
    cv::minMaxLoc(labels, nullptr, &max_label_value);
    const int max_label = static_cast<int>(max_label_value);

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
    cv::parallel_for_(cv::Range(0, white_domain.rows),
                      [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            for (int x = 0; x < white_domain.cols; ++x) {
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
                        const int nx = x + dx;
                        const int ny = y + dy;
                        if (nx < 0 || nx >= white_domain.cols || ny < 0 ||
                            ny >= white_domain.rows ||
                            white_domain.at<std::uint8_t>(ny, nx) == 0) {
                            continue;
                        }
                        const int other =
                            source_pixel_labels.at<int>(ny, nx);
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

cv::Mat connect_clean_skeleton_with_source_ridges(
    const cv::Mat& clean_skeleton,
    const cv::Mat& source_skeleton,
    const cv::Mat& dt,
    std::vector<StageTiming>* timings = nullptr) {
    CV_Assert(clean_skeleton.type() == CV_8U);
    CV_Assert(source_skeleton.type() == CV_8U);
    CV_Assert(dt.type() == CV_32F);

    cv::Mat clean;
    cv::threshold(clean_skeleton, clean, 0, 255, cv::THRESH_BINARY);

    cv::Mat clean_labels;
    int num_clean_components = 0;
    {
        const TimingMark timing = start_timing();
        num_clean_components =
            cv::connectedComponents(clean, clean_labels, 8, CV_32S);
        if (timings != nullptr) {
            timings->push_back(finish_timing(
                "component.connect_ridges.clean_components", timing));
        }
    }
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
    int num_candidates = 0;
    {
        const TimingMark timing = start_timing();
        num_candidates = cv::connectedComponentsWithStats(
            connector_candidates, candidate_labels, candidate_stats,
            candidate_centroids, 8, CV_32S);
        if (timings != nullptr) {
            timings->push_back(finish_timing(
                "component.connect_ridges.candidate_components", timing));
        }
    }
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
    {
        const TimingMark timing = start_timing();
        for (int label = 1; label < num_candidates; ++label) {
            const int left = candidate_stats.at<int>(label, cv::CC_STAT_LEFT);
            const int top = candidate_stats.at<int>(label, cv::CC_STAT_TOP);
            const int width = candidate_stats.at<int>(label, cv::CC_STAT_WIDTH);
            const int height =
                candidate_stats.at<int>(label, cv::CC_STAT_HEIGHT);
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
                const int local = static_cast<int>(workspace.pixels.size());
                workspace.pixels.push_back(cv::Point(x, y));
                workspace.local_index.at<int>(y - workspace.bounds.y,
                                              x - workspace.bounds.x) = local;
            }
        }
        if (timings != nullptr) {
            timings->push_back(finish_timing(
                "component.connect_ridges.workspace_build", timing));
        }
    }

    cv::Mat clean_source_mask(clean.size(), CV_8U, cv::Scalar(255));
    clean_source_mask.setTo(0, clean);

    cv::Mat distance_to_clean;
    cv::Mat nearest_clean_pixel;
    std::vector<cv::Point> clean_source_points;
    {
        const TimingMark timing = start_timing();
        cv::distanceTransform(clean_source_mask, distance_to_clean,
                              nearest_clean_pixel, cv::DIST_L2,
                              cv::DIST_MASK_5, cv::DIST_LABEL_PIXEL);
        clean_source_points =
            label_source_points(clean_source_mask, nearest_clean_pixel);
        if (timings != nullptr) {
            timings->push_back(finish_timing(
                "component.connect_ridges.clean_distance_transform", timing));
        }
    }

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
    {
        const TimingMark timing = start_timing();
        std::vector<std::vector<Attachment>> row_attachments(
            static_cast<std::size_t>(candidate_labels.rows));
        cv::parallel_for_(cv::Range(0, candidate_labels.rows),
                          [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                std::vector<Attachment>& row = row_attachments[y];
                for (int x = 0; x < candidate_labels.cols; ++x) {
                    const int candidate_label =
                        candidate_labels.at<int>(y, x);
                    if (candidate_label <= 0) {
                        continue;
                    }

                    if (distance_to_clean.at<float>(y, x) >
                        kMaxAttachDistance) {
                        continue;
                    }

                    const int source_label =
                        nearest_clean_pixel.at<int>(y, x);
                    if (source_label <= 0 ||
                        source_label >=
                            static_cast<int>(clean_source_points.size())) {
                        continue;
                    }
                    const cv::Point source =
                        clean_source_points[source_label];
                    if (source.x < 0) {
                        continue;
                    }

                    const int clean_label =
                        clean_labels.at<int>(source.y, source.x);
                    if (clean_label <= 0) {
                        continue;
                    }

                    row.push_back(
                        {clean_label, cv::Point(x, y), source});
                }
            }
        });
        for (const std::vector<Attachment>& row : row_attachments) {
            for (const Attachment& attachment : row) {
                const int label =
                    candidate_labels.at<int>(attachment.pixel.y,
                                             attachment.pixel.x);
                attachments[label].push_back(attachment);
            }
        }
        if (timings != nullptr) {
            timings->push_back(finish_timing(
                "component.connect_ridges.attachment_collect", timing));
        }
    }

    std::vector<std::vector<CandidateEdge>> edges_by_candidate(
        static_cast<std::size_t>(num_candidates));
    {
        const TimingMark timing = start_timing();
        cv::parallel_for_(cv::Range(1, num_candidates),
                          [&](const cv::Range& range) {
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

            struct LocalEdgeCandidate {
                int label = 0;
                int a = 0;
                int b = 0;
                float bottleneck_dt = 0.0f;
                int length = 0;
                int a_idx = -1;
                int b_idx = -1;
                cv::Point clean_a{-1, -1};
                cv::Point clean_b{-1, -1};
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

            std::vector<LocalEdgeCandidate> local_candidates;
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
                        LocalEdgeCandidate edge;
                        edge.label = label;
                        edge.a = state.owner;
                        edge.b = owner[next_idx];
                        edge.clean_a = owner_clean[state.idx];
                        edge.clean_b = owner_clean[next_idx];
                        edge.bottleneck_dt =
                            std::min(state.bottleneck, best[next_idx]);
                        edge.length =
                            best_length[next_idx] + best_length[state.idx];
                        edge.a_idx = state.idx;
                        edge.b_idx = next_idx;
                        local_candidates.push_back(edge);
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

            std::sort(local_candidates.begin(), local_candidates.end(),
                      [](const LocalEdgeCandidate& a,
                         const LocalEdgeCandidate& b) {
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

            std::vector<LocalEdgeCandidate> deduped_candidates;
            for (LocalEdgeCandidate& edge : local_candidates) {
                if (!deduped_candidates.empty()) {
                    const LocalEdgeCandidate& prev =
                        deduped_candidates.back();
                    if (std::min(prev.a, prev.b) ==
                            std::min(edge.a, edge.b) &&
                        std::max(prev.a, prev.b) ==
                            std::max(edge.a, edge.b)) {
                        continue;
                    }
                }
                deduped_candidates.push_back(std::move(edge));
            }

            std::vector<CandidateEdge> deduped_edges;
            deduped_edges.reserve(deduped_candidates.size());
            for (const LocalEdgeCandidate& candidate : deduped_candidates) {
                CandidateEdge edge;
                edge.label = candidate.label;
                edge.a = candidate.a;
                edge.b = candidate.b;
                edge.bottleneck_dt = candidate.bottleneck_dt;
                edge.length = candidate.length;
                edge.clean_a = candidate.clean_a;
                edge.clean_b = candidate.clean_b;

                std::vector<cv::Point> b_path =
                    path_to_root(candidate.b_idx);
                std::reverse(b_path.begin(), b_path.end());
                edge.path = std::move(b_path);
                std::vector<cv::Point> a_path =
                    path_to_root(candidate.a_idx);
                edge.path.insert(edge.path.end(), a_path.begin(),
                                 a_path.end());
                edge.length = static_cast<int>(edge.path.size());
                deduped_edges.push_back(std::move(edge));
            }
            edges_by_candidate[label] = std::move(deduped_edges);
        }
    });
        if (timings != nullptr) {
            timings->push_back(finish_timing(
                "component.connect_ridges.candidate_search", timing));
        }
    }

    std::vector<CandidateEdge> edges;
    {
        const TimingMark timing = start_timing();
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
        if (timings != nullptr) {
            timings->push_back(
                finish_timing("component.connect_ridges.edge_sort", timing));
        }
    }

    cv::Mat out = clean.clone();
    {
        const TimingMark timing = start_timing();
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
                    cv::line(out, edge.clean_a, edge.path.back(),
                             cv::Scalar(255), 1, cv::LINE_8);
                }
                if (edge.clean_b.x >= 0) {
                    cv::line(out, edge.clean_b, edge.path.front(),
                             cv::Scalar(255), 1, cv::LINE_8);
                }
            }
        }
        if (timings != nullptr) {
            timings->push_back(
                finish_timing("component.connect_ridges.mst_emit", timing));
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
                std::min(edge.capacity,
                         capacity_from_dt(dt.at<float>(current.y, current.x)));

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
            if (out.depth() == CV_32F) {
                out.at<float>(pixel.y, pixel.x) =
                    static_cast<float>(color[0]);
            } else {
                out.at<std::uint8_t>(pixel.y, pixel.x) =
                    static_cast<std::uint8_t>(color[0]);
            }
        } else {
            if (out.depth() == CV_32F) {
                out.at<cv::Vec3f>(pixel.y, pixel.x) =
                    cv::Vec3f(static_cast<float>(color[0]),
                              static_cast<float>(color[1]),
                              static_cast<float>(color[2]));
            } else {
                out.at<cv::Vec3b>(pixel.y, pixel.x) =
                    cv::Vec3b(static_cast<std::uint8_t>(color[0]),
                              static_cast<std::uint8_t>(color[1]),
                              static_cast<std::uint8_t>(color[2]));
            }
        }
    }
}

std::string format_scalar_value(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(0) << value;
    return out.str();
}

void draw_debug_label(cv::Mat& image, const cv::Point anchor,
                      const std::string& text, const cv::Scalar color) {
    if (text.empty()) {
        return;
    }
    cv::Point pos(anchor.x + 3, anchor.y - 3);
    pos.x = std::clamp(pos.x, 0, std::max(0, image.cols - 1));
    pos.y = std::clamp(pos.y, 8, std::max(8, image.rows - 1));
    cv::putText(image, text, pos + cv::Point(1, 1),
                cv::FONT_HERSHEY_PLAIN, 0.55, cv::Scalar(0, 0, 0), 1,
                cv::LINE_AA);
    cv::putText(image, text, pos, cv::FONT_HERSHEY_PLAIN, 0.55, color, 1,
                cv::LINE_AA);
}

void draw_large_debug_label(cv::Mat& image, const cv::Point anchor,
                            const std::string& text,
                            const cv::Scalar color) {
    if (text.empty()) {
        return;
    }
    cv::Point pos(anchor.x + 4, anchor.y - 4);
    pos.x = std::clamp(pos.x, 0, std::max(0, image.cols - 1));
    pos.y = std::clamp(pos.y, 12, std::max(12, image.rows - 1));
    cv::putText(image, text, pos + cv::Point(1, 1),
                cv::FONT_HERSHEY_PLAIN, 0.85, cv::Scalar(0, 0, 0), 2,
                cv::LINE_AA);
    cv::putText(image, text, pos, cv::FONT_HERSHEY_PLAIN, 0.85, color, 1,
                cv::LINE_AA);
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

cv::Mat render_graph_capacity(const SkeletonGraph& graph, cv::Size size,
                              std::uint8_t background) {
    cv::Mat out(size, CV_32FC3,
                cv::Scalar(background, background, background));
    for (const GraphEdge& edge : graph.edges) {
        draw_graph_edge(out, edge,
                        cv::Scalar(edge.capacity, edge.capacity,
                                   edge.capacity));
    }
    for (std::size_t label = 1; label < graph.nodes.size(); ++label) {
        cv::circle(out, graph.nodes[label], 3, cv::Scalar(180, 180, 180),
                   cv::FILLED, cv::LINE_8);
    }
    for (const GraphEdge& edge : graph.edges) {
        if (edge.pixels.empty()) {
            continue;
        }
        draw_debug_label(out, edge.pixels[edge.pixels.size() / 2],
                         format_scalar_value(edge.capacity),
                         cv::Scalar(255, 255, 255));
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
    cv::Mat voronoi_tree_flow;
    cv::Mat voronoi_tree_flow_gray_bg;
    cv::Mat tree_dense_nn_flow;
    cv::Mat dense_backtrack_nn_flow;
    cv::Mat smooth_grid_flow;
    cv::Mat tree_dense_flow;
    cv::Mat tree_path_debug;
    cv::Mat carrier_debug;
    cv::Mat tree_flow_attn;
    cv::Mat flow_attn;
    cv::Mat graph_edge_flow;
    cv::Mat graph_edge_flow_gray_bg;
    cv::Mat edge_flow_px;
    cv::Mat edge_flow_px_gray_bg;
    cv::Mat graph_source_edges;
    cv::Mat flow_gate_weight;
    cv::Point source_seed_pixel{-1, -1};
    float source_seed_capacity = 0.0f;
    int source_edges = 0;
    int seeded_nodes = 0;
    float finite_edge_flow_min = 0.0f;
    float finite_edge_flow_max = 0.0f;
    int finite_edge_flows = 0;
    std::vector<StageTiming> timings;
};

std::vector<cv::Point> unique_edge_pixels(const std::vector<cv::Point>& pixels) {
    std::vector<cv::Point> unique;
    unique.reserve(pixels.size());
    for (const cv::Point pixel : pixels) {
        if (std::find(unique.begin(), unique.end(), pixel) == unique.end()) {
            unique.push_back(pixel);
        }
    }
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

    const auto four_connected = [&](int a, int b) {
        return std::abs(pixels[a].x - pixels[b].x) +
                   std::abs(pixels[a].y - pixels[b].y) ==
               1;
    };

    const auto choose_next = [&](int current,
                                 const std::vector<int>& candidates) {
        if (candidates.empty()) {
            return -1;
        }
        std::vector<int> four_candidates;
        for (int candidate : candidates) {
            if (four_connected(current, candidate)) {
                four_candidates.push_back(candidate);
            }
        }
        const std::vector<int>& choices =
            four_candidates.empty() ? candidates : four_candidates;

        int best = choices.front();
        double best_score = std::numeric_limits<double>::max();
        for (int candidate : choices) {
            double score = 0.0;
            if (end >= 0) {
                const double dx = pixels[candidate].x - pixels[end].x;
                const double dy = pixels[candidate].y - pixels[end].y;
                score = dx * dx + dy * dy;
            } else {
                score = degree[candidate];
            }
            if (score < best_score ||
                (score == best_score &&
                 (pixels[candidate].y < pixels[best].y ||
                  (pixels[candidate].y == pixels[best].y &&
                   pixels[candidate].x < pixels[best].x)))) {
                best_score = score;
                best = candidate;
            }
        }
        return best;
    };

    std::vector<char> visited(pixels.size(), 0);
    std::vector<cv::Point> ordered;
    ordered.reserve(pixels.size());
    int current = start;
    while (current >= 0 && !visited[current]) {
        visited[current] = 1;
        ordered.push_back(pixels[current]);

        std::vector<int> candidates;
        for (const cv::Point dir : kDirs) {
            const cv::Point next = pixels[current] + dir;
            if (!bounds.contains(next)) {
                continue;
            }
            const int candidate =
                local_index.at<int>(next.y - bounds.y, next.x - bounds.x);
            if (candidate >= 0 && !visited[candidate]) {
                candidates.push_back(candidate);
            }
        }
        current = choose_next(current, candidates);
    }

    if (ordered.size() == pixels.size()) {
        return ordered;
    }

    while (ordered.size() < pixels.size()) {
        int restart = -1;
        double best_restart = std::numeric_limits<double>::max();
        for (int i = 0; i < static_cast<int>(pixels.size()); ++i) {
            if (visited[i]) {
                continue;
            }
            double score = 0.0;
            if (!ordered.empty()) {
                const cv::Point last = ordered.back();
                const double dx = pixels[i].x - last.x;
                const double dy = pixels[i].y - last.y;
                score = dx * dx + dy * dy;
            }
            if (restart < 0 || score < best_restart) {
                best_restart = score;
                restart = i;
            }
        }
        if (restart < 0) {
            break;
        }
        current = restart;
        while (current >= 0 && !visited[current]) {
            visited[current] = 1;
            ordered.push_back(pixels[current]);
            std::vector<int> candidates;
            for (const cv::Point dir : kDirs) {
                const cv::Point next = pixels[current] + dir;
                if (!bounds.contains(next)) {
                    continue;
                }
                const int candidate =
                    local_index.at<int>(next.y - bounds.y, next.x - bounds.x);
                if (candidate >= 0 && !visited[candidate]) {
                    candidates.push_back(candidate);
                }
            }
            current = choose_next(current, candidates);
        }
    }
    return ordered;
}

struct DenseBacktrackResult {
    cv::Mat nn_flow;
    cv::Mat smooth_grid_flow;
    cv::Mat flow;
    cv::Mat debug_paths;
    cv::Mat carrier_debug;
    int seeded_pixels = 0;
    int reached_pixels = 0;
    int unreached_white_pixels = 0;
    std::vector<StageTiming> timings;
};

cv::Mat bilinear_from_regular_grid_samples(const cv::Mat& source,
                                           int grid_step);

DenseBacktrackResult compute_dense_backtrack_flow(const cv::Mat& white_domain,
                                                  const cv::Mat& dt,
                                                  const cv::Mat& graph_pixel_flow,
                                                  const cv::Mat& graph_node_flow,
                                                  const cv::Mat& source_edge_mask,
                                                  const cv::Mat& graph_node_mask,
                                                  const SkeletonGraph& graph,
                                                  int grid_step,
                                                  float backtrack_distance,
                                                  bool enable_debug_outputs) {
    CV_Assert(white_domain.type() == CV_8U);
    CV_Assert(dt.type() == CV_32F);
    CV_Assert(graph_pixel_flow.type() == CV_32F);
    CV_Assert(graph_node_flow.type() == CV_32F);
    CV_Assert(source_edge_mask.type() == CV_8U);
    CV_Assert(graph_node_mask.type() == CV_8U);

    DenseBacktrackResult result;
    result.nn_flow = cv::Mat(white_domain.size(), CV_32F, cv::Scalar(0));
    result.smooth_grid_flow = cv::Mat(white_domain.size(), CV_32F, cv::Scalar(0));
    result.flow = cv::Mat(white_domain.size(), CV_32F, cv::Scalar(0));
    if (enable_debug_outputs) {
        result.debug_paths =
            cv::Mat(white_domain.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    }

    constexpr float kFlowEpsilon = 1.0e-4f;
    const float kBacktrackRadius = std::max(0.0f, backtrack_distance);
    const int rows = white_domain.rows;
    const int cols = white_domain.cols;
    const int pixel_count = rows * cols;
    const auto linear_index = [cols](const cv::Point pixel) {
        return pixel.y * cols + pixel.x;
    };
    const auto point_from_index = [cols](int index) {
        return cv::Point(index % cols, index / cols);
    };
    const auto step_distance = [](const cv::Point a, const cv::Point b) {
        const int dx = std::abs(a.x - b.x);
        const int dy = std::abs(a.y - b.y);
        return dx + dy == 2 ? static_cast<float>(std::sqrt(2.0f)) : 1.0f;
    };
    const auto in_white_domain = [&](const cv::Point pixel) {
        return pixel.x >= 0 && pixel.x < cols && pixel.y >= 0 &&
               pixel.y < rows &&
               white_domain.at<std::uint8_t>(pixel.y, pixel.x) != 0;
    };

    std::vector<float> routed_flow(static_cast<std::size_t>(pixel_count), 0.0f);
    std::vector<float> routed_distance(
        static_cast<std::size_t>(pixel_count),
        std::numeric_limits<float>::infinity());
    std::vector<int> next_pixel(static_cast<std::size_t>(pixel_count), -1);
    std::vector<float> next_distance(static_cast<std::size_t>(pixel_count),
                                     0.0f);
    std::vector<int> scalar_next_pixel(static_cast<std::size_t>(pixel_count), -1);
    std::vector<float> scalar_next_distance(
        static_cast<std::size_t>(pixel_count), 0.0f);
    std::vector<int> graph_next_pixel(static_cast<std::size_t>(pixel_count), -1);
    std::vector<float> graph_next_distance(static_cast<std::size_t>(pixel_count),
                                           0.0f);
    std::vector<float> graph_seed_flow(static_cast<std::size_t>(pixel_count),
                                       0.0f);
    std::vector<char> graph_route_pixel(static_cast<std::size_t>(pixel_count),
                                        0);

    struct QueueItem {
        float flow = 0.0f;
        float distance = 0.0f;
        int index = -1;
        bool operator<(const QueueItem& other) const {
            if (flow != other.flow) {
                return flow < other.flow;
            }
            return distance > other.distance;
        }
    };
    std::priority_queue<QueueItem> queue;

    {
        const TimingMark timing = start_timing();
        std::vector<float> graph_route_flow(
            static_cast<std::size_t>(pixel_count), 0.0f);
        std::vector<float> graph_route_distance(
            static_cast<std::size_t>(pixel_count),
            std::numeric_limits<float>::infinity());
        struct GraphRouteItem {
            float distance = 0.0f;
            int index = -1;
            bool operator<(const GraphRouteItem& other) const {
                return distance > other.distance;
            }
        };
        std::priority_queue<GraphRouteItem> graph_queue;
        int graph_edge_pixels = 0;
        int graph_node_pixels = 0;
        int graph_root_pixels = 0;
        const auto graph_route_capacity = [&](const cv::Point pixel) {
            const float node_flow =
                graph_node_flow.at<float>(pixel.y, pixel.x);
            if (node_flow > 0.0f) {
                return node_flow;
            }
            const float edge_flow =
                graph_pixel_flow.at<float>(pixel.y, pixel.x);
            if (edge_flow > 0.0f) {
                return capacity_from_dt(dt.at<float>(pixel.y, pixel.x));
            }
            return 0.0f;
        };
        const auto is_graph_route_pixel = [&](const cv::Point pixel) {
            return graph_pixel_flow.at<float>(pixel.y, pixel.x) > 0.0f ||
                   graph_node_flow.at<float>(pixel.y, pixel.x) > 0.0f;
        };
        const auto graph_scalar_seed = [&](const cv::Point pixel) {
            if (graph_node_mask.at<std::uint8_t>(pixel.y, pixel.x) == 0) {
                return 0.0f;
            }
            return graph_node_flow.at<float>(pixel.y, pixel.x);
        };

        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (graph_pixel_flow.at<float>(y, x) > 0.0f) {
                    ++graph_edge_pixels;
                }
                if (graph_node_flow.at<float>(y, x) > 0.0f) {
                    ++graph_node_pixels;
                }
                if (!is_graph_route_pixel(cv::Point(x, y))) {
                    continue;
                }
                if (source_edge_mask.at<std::uint8_t>(y, x) == 0) {
                    continue;
                }
                const int index = y * cols + x;
                graph_route_flow[index] =
                    std::max(graph_route_capacity(cv::Point(x, y)),
                             capacity_from_dt(dt.at<float>(y, x)));
                graph_route_distance[index] = 0.0f;
                graph_queue.push({0.0f, index});
                ++graph_root_pixels;
            }
        }

        const std::array<cv::Point, 8> kDirs = {
            cv::Point(-1, -1), cv::Point(0, -1), cv::Point(1, -1),
            cv::Point(-1, 0),                    cv::Point(1, 0),
            cv::Point(-1, 1),  cv::Point(0, 1),  cv::Point(1, 1)};
        while (!graph_queue.empty()) {
            const GraphRouteItem item = graph_queue.top();
            graph_queue.pop();
            if (item.distance >
                graph_route_distance[item.index] + kFlowEpsilon) {
                continue;
            }
            const cv::Point pixel = point_from_index(item.index);
            for (const cv::Point dir : kDirs) {
                const cv::Point next = pixel + dir;
                if (next.x < 0 || next.x >= cols || next.y < 0 ||
                    next.y >= rows || !is_graph_route_pixel(next)) {
                    continue;
                }
                const int next_index = linear_index(next);
                const float candidate_distance =
                    item.distance + step_distance(next, pixel);
                if (candidate_distance + kFlowEpsilon >=
                    graph_route_distance[next_index]) {
                    continue;
                }
                graph_route_flow[next_index] = graph_route_capacity(next);
                graph_route_distance[next_index] = candidate_distance;
                graph_next_pixel[next_index] = item.index;
                graph_next_distance[next_index] = step_distance(next, pixel);
                graph_queue.push({candidate_distance, next_index});
            }
        }

        int graph_routed_pixels = 0;
        int graph_routed_route_pixels = 0;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                const int index = y * cols + x;
                if (graph_route_flow[index] > 0.0f) {
                    ++graph_routed_route_pixels;
                    graph_route_pixel[index] = 1;
                    graph_seed_flow[index] = graph_scalar_seed(cv::Point(x, y));
                }
                if (graph_pixel_flow.at<float>(y, x) <= 0.0f) {
                    continue;
                }
                if (graph_route_flow[index] > 0.0f) {
                    ++graph_routed_pixels;
                }
            }
        }
        std::cout << "Dense graph backtrack:\n"
                  << "  graph_edge_pixels: " << graph_edge_pixels << "\n"
                  << "  graph_node_pixels: " << graph_node_pixels << "\n"
                  << "  graph_root_pixels: " << graph_root_pixels << "\n"
                  << "  graph_routed_pixels: " << graph_routed_pixels << "\n"
                  << "  graph_routed_route_pixels: "
                  << graph_routed_route_pixels << "\n"
                  << "  graph_unrouted_pixels: "
                  << (graph_edge_pixels - graph_routed_pixels) << "\n";
        result.timings.push_back(
            finish_timing("dense_backtrack_graph_route", timing));
    }

    {
        const TimingMark timing = start_timing();
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (white_domain.at<std::uint8_t>(y, x) == 0) {
                    continue;
                }
                const float seed = graph_seed_flow[y * cols + x];
                if (seed <= 0.0f) {
                    continue;
                }
                const float flow =
                    std::min(seed, capacity_from_dt(dt.at<float>(y, x)));
                if (flow <= 0.0f) {
                    continue;
                }
                const int index = y * cols + x;
                routed_flow[index] = flow;
                routed_distance[index] = 0.0f;
                next_pixel[index] = graph_next_pixel[index];
                next_distance[index] = graph_next_distance[index];
                queue.push({flow, 0.0f, index});
                ++result.seeded_pixels;
            }
        }
        result.timings.push_back(finish_timing("dense_backtrack_seed", timing));
    }

    if (kEnablePixelDenseBacktrackFlood) {
        const TimingMark timing = start_timing();
        const std::array<cv::Point, 8> kDirs = {
            cv::Point(-1, -1), cv::Point(0, -1), cv::Point(1, -1),
            cv::Point(-1, 0),                    cv::Point(1, 0),
            cv::Point(-1, 1),  cv::Point(0, 1),  cv::Point(1, 1)};
        while (!queue.empty()) {
            const QueueItem item = queue.top();
            queue.pop();
            if (item.flow + kFlowEpsilon < routed_flow[item.index] ||
                (std::abs(item.flow - routed_flow[item.index]) <=
                     kFlowEpsilon &&
                 item.distance >
                     routed_distance[item.index] + kFlowEpsilon)) {
                continue;
            }
            const cv::Point pixel = point_from_index(item.index);
            for (const cv::Point dir : kDirs) {
                const cv::Point next = pixel + dir;
                if (!in_white_domain(next)) {
                    continue;
                }
                const int next_index = linear_index(next);
                const float candidate =
                    std::min(item.flow,
                             capacity_from_dt(dt.at<float>(next.y, next.x)));
                const float candidate_distance =
                    item.distance + step_distance(next, pixel);
                const bool better_flow =
                    candidate > routed_flow[next_index] + kFlowEpsilon;
                const bool same_flow_shorter =
                    std::abs(candidate - routed_flow[next_index]) <=
                        kFlowEpsilon &&
                    candidate_distance + kFlowEpsilon <
                        routed_distance[next_index];
                if (!better_flow && !same_flow_shorter) {
                    continue;
                }
                routed_flow[next_index] = candidate;
                routed_distance[next_index] = candidate_distance;
                next_pixel[next_index] = item.index;
                next_distance[next_index] = step_distance(next, pixel);
                queue.push({candidate, candidate_distance, next_index});
            }
        }
        result.timings.push_back(
            finish_timing("dense_backtrack_flood", timing));
    } else {
        result.timings.push_back({"dense_backtrack_flood_skipped", 0.0, 0.0});
    }

    {
        const TimingMark dense_grid_outer_timing = start_timing();
        const std::size_t dense_grid_timing_start = result.timings.size();
        const int kGridStep = std::max(1, grid_step);

        struct CarrierNode {
            cv::Point pixel;
            float capacity = 0.0f;
            float source_flow = 0.0f;
            bool grid = false;
            bool fixed = false;
        };
        struct CarrierNeighbor {
            int node = -1;
            float distance = 0.0f;
        };

        const auto line_in_white = [&](const cv::Point a, const cv::Point b) {
            cv::LineIterator it(white_domain, a, b, 8);
            for (int i = 0; i < it.count; ++i, ++it) {
                const cv::Point pixel = it.pos();
                if (!in_white_domain(pixel)) {
                    return false;
                }
                if (kEnablePixelDenseBacktrackFlood &&
                    routed_flow[linear_index(pixel)] <= 0.0f) {
                    return false;
                }
            }
            return true;
        };
        const auto add_axis = [kGridStep](const int limit) {
            std::vector<int> axis;
            for (int value = 0; value < limit; value += kGridStep) {
                axis.push_back(value);
            }
            if (axis.empty() || axis.back() != limit - 1) {
                axis.push_back(limit - 1);
            }
            return axis;
        };

        const TimingMark axis_setup_timing = start_timing();
        const std::vector<int> grid_xs = add_axis(cols);
        const std::vector<int> grid_ys = add_axis(rows);
        const int grid_cols = static_cast<int>(grid_xs.size());
        const int grid_rows = static_cast<int>(grid_ys.size());
        std::vector<int> grid_node_ids(
            static_cast<std::size_t>(grid_cols * grid_rows), -1);
        std::vector<CarrierNode> carriers;
        carriers.reserve(static_cast<std::size_t>(grid_cols * grid_rows + 4096));
        result.timings.push_back(
            finish_timing("dense_grid.axis_setup", axis_setup_timing));

        const auto add_carrier = [&](const cv::Point pixel,
                                     const float capacity,
                                     const float source_flow,
                                     const bool grid,
                                     const bool fixed) {
            if (!in_white_domain(pixel) || capacity <= 0.0f) {
                return -1;
            }
            const int id = static_cast<int>(carriers.size());
            carriers.push_back({pixel, capacity, source_flow, grid, fixed});
            return id;
        };

        const auto graph_seed_near_grid_point = [&](const cv::Point pixel) {
            float seed = 0.0f;
            const int radius = std::max(1, kGridStep);
            const float radius_sq = static_cast<float>(radius * radius);
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx * dx + dy * dy > radius_sq) {
                        continue;
                    }
                    const cv::Point candidate(pixel.x + dx, pixel.y + dy);
                    if (!in_white_domain(candidate)) {
                        continue;
                    }
                    const float value = routed_flow[linear_index(candidate)];
                    if (value <= seed) {
                        continue;
                    }
                    if (!line_in_white(pixel, candidate)) {
                        continue;
                    }
                    seed = value;
                }
            }
            return seed;
        };

        {
            const TimingMark grid_carrier_seed_timing = start_timing();
            struct GridCarrierCandidate {
                float capacity = 0.0f;
                float source_flow = 0.0f;
            };
            std::vector<GridCarrierCandidate> grid_candidates(
                static_cast<std::size_t>(grid_cols * grid_rows));
            cv::parallel_for_(cv::Range(0, grid_rows),
                              [&](const cv::Range& range) {
                for (int gy = range.start; gy < range.end; ++gy) {
                    for (int gx = 0; gx < grid_cols; ++gx) {
                        const cv::Point pixel(grid_xs[gx], grid_ys[gy]);
                        const std::size_t index =
                            static_cast<std::size_t>(gy * grid_cols + gx);
                        if (!in_white_domain(pixel)) {
                            continue;
                        }
                        const float grid_capacity =
                            capacity_from_dt(dt.at<float>(pixel.y, pixel.x));
                        if (grid_capacity <= 0.0f) {
                            continue;
                        }
                        const float source_flow = std::min(
                            graph_seed_near_grid_point(pixel), grid_capacity);
                        grid_candidates[index] = {grid_capacity, source_flow};
                    }
                }
            });
            for (int gy = 0; gy < grid_rows; ++gy) {
                for (int gx = 0; gx < grid_cols; ++gx) {
                    const cv::Point pixel(grid_xs[gx], grid_ys[gy]);
                    const GridCarrierCandidate candidate =
                        grid_candidates[static_cast<std::size_t>(
                            gy * grid_cols + gx)];
                    const int id = add_carrier(pixel, candidate.capacity,
                                               candidate.source_flow,
                                               true,
                                               candidate.source_flow > 0.0f);
                    grid_node_ids[static_cast<std::size_t>(gy * grid_cols + gx)] =
                        id;
                }
            }
            result.timings.push_back(finish_timing(
                "dense_grid.grid_carrier_seed", grid_carrier_seed_timing));
        }

        int graph_carriers = 0;
        int graph_node_carriers = 0;
        int graph_edge_carriers = 0;

        std::vector<std::vector<CarrierNeighbor>> carrier_edges(
            carriers.size());
        const auto append_carrier_edge = [&](const int a, const int b) {
            if (a < 0 || b < 0 || a == b) {
                return;
            }
            const cv::Point pa = carriers[a].pixel;
            const cv::Point pb = carriers[b].pixel;
            const float distance = step_distance(pa, pb) *
                                   cv::norm(cv::Point2f(
                                       static_cast<float>(pa.x - pb.x),
                                       static_cast<float>(pa.y - pb.y))) /
                                   std::max(1.0f, step_distance(pa, pb));
            const auto exists = [&](const int from, const int to) {
                for (const CarrierNeighbor neighbor : carrier_edges[from]) {
                    if (neighbor.node == to) {
                        return true;
                    }
                }
                return false;
            };
            if (!exists(a, b)) {
                carrier_edges[a].push_back({b, distance});
            }
            if (!exists(b, a)) {
                carrier_edges[b].push_back({a, distance});
            }
        };
        const auto add_carrier_edge = [&](const int a, const int b) {
            if (a < 0 || b < 0 || a == b) {
                return;
            }
            if (!line_in_white(carriers[a].pixel, carriers[b].pixel)) {
                return;
            }
            append_carrier_edge(a, b);
        };

        {
            const TimingMark grid_edges_timing = start_timing();
            const std::array<cv::Point, 4> kForwardGridDirs = {
                cv::Point(1, -1), cv::Point(1, 0), cv::Point(1, 1),
                cv::Point(0, 1)};
            std::vector<std::vector<std::pair<int, int>>> row_edge_pairs(
                static_cast<std::size_t>(grid_rows));
            cv::parallel_for_(cv::Range(0, grid_rows),
                              [&](const cv::Range& range) {
                for (int gy = range.start; gy < range.end; ++gy) {
                    std::vector<std::pair<int, int>>& pairs =
                        row_edge_pairs[static_cast<std::size_t>(gy)];
                    pairs.reserve(static_cast<std::size_t>(grid_cols * 4));
                    for (int gx = 0; gx < grid_cols; ++gx) {
                        const int a = grid_node_ids[static_cast<std::size_t>(
                            gy * grid_cols + gx)];
                        if (a < 0) {
                            continue;
                        }
                        for (const cv::Point dir : kForwardGridDirs) {
                            const int nx = gx + dir.x;
                            const int ny = gy + dir.y;
                            if (nx < 0 || nx >= grid_cols || ny < 0 ||
                                ny >= grid_rows) {
                                continue;
                            }
                            const int b =
                                grid_node_ids[static_cast<std::size_t>(
                                    ny * grid_cols + nx)];
                            if (b < 0) {
                                continue;
                            }
                            if (line_in_white(carriers[a].pixel,
                                              carriers[b].pixel)) {
                                pairs.push_back({a, b});
                            }
                        }
                    }
                }
            });
            for (const std::vector<std::pair<int, int>>& pairs :
                 row_edge_pairs) {
                for (const auto [a, b] : pairs) {
                    append_carrier_edge(a, b);
                }
            }
            result.timings.push_back(
                finish_timing("dense_grid.grid_edges", grid_edges_timing));
        }

        int grid_carriers = 0;
        for (const CarrierNode& carrier : carriers) {
            if (carrier.grid) {
                ++grid_carriers;
            }
        }
        {
            const TimingMark graph_edges_timing = start_timing();
            for (int id = 0; id < static_cast<int>(carriers.size()); ++id) {
                if (carriers[id].grid) {
                    continue;
                }
                std::vector<std::pair<float, int>> nearest;
                for (int candidate = 0;
                     candidate < static_cast<int>(carriers.size());
                     ++candidate) {
                    if (!carriers[candidate].grid) {
                        continue;
                    }
                    const float distance = static_cast<float>(
                        cv::norm(carriers[id].pixel - carriers[candidate].pixel));
                    if (distance > kGridStep * 1.75f) {
                        continue;
                    }
                    nearest.push_back({distance, candidate});
                }
                std::sort(nearest.begin(), nearest.end());
                const int limit = std::min(4, static_cast<int>(nearest.size()));
                for (int i = 0; i < limit; ++i) {
                    add_carrier_edge(id, nearest[i].second);
                }
            }
            result.timings.push_back(
                finish_timing("dense_grid.graph_edges", graph_edges_timing));
        }

        std::vector<float> carrier_value(carriers.size(), 0.0f);
        struct CarrierFlowItem {
            float flow = 0.0f;
            int node = -1;
            bool operator<(const CarrierFlowItem& other) const {
                if (flow != other.flow) {
                    return flow < other.flow;
                }
                return node > other.node;
            }
        };
        std::priority_queue<CarrierFlowItem> carrier_flow_queue;
        int carrier_flow_seeds = 0;
        {
            const TimingMark carrier_flow_timing = start_timing();
            for (int node = 0; node < static_cast<int>(carriers.size());
                 ++node) {
                if (!carriers[node].fixed) {
                    continue;
                }
                carrier_value[node] = carriers[node].source_flow;
                carrier_flow_queue.push({carrier_value[node], node});
                ++carrier_flow_seeds;
            }
            while (!carrier_flow_queue.empty()) {
                const CarrierFlowItem item = carrier_flow_queue.top();
                carrier_flow_queue.pop();
                if (item.flow + kFlowEpsilon < carrier_value[item.node]) {
                    continue;
                }
                for (const CarrierNeighbor neighbor : carrier_edges[item.node]) {
                    const float candidate =
                        std::min(item.flow, carriers[neighbor.node].capacity);
                    if (candidate <=
                        carrier_value[neighbor.node] + kFlowEpsilon) {
                        continue;
                    }
                    carrier_value[neighbor.node] = candidate;
                    carrier_flow_queue.push({candidate, neighbor.node});
                }
            }
            result.timings.push_back(
                finish_timing("dense_grid.carrier_flow", carrier_flow_timing));
        }
        int carrier_flow_reached = 0;
        for (const float value : carrier_value) {
            if (value > 0.0f) {
                ++carrier_flow_reached;
            }
        }

        constexpr float kCarrierRouteBucketPx = 10.0f;
        const int kCarrierRouteBuckets =
            static_cast<int>(kBacktrackRadius / kCarrierRouteBucketPx) + 1;
        const int carrier_count = static_cast<int>(carriers.size());
        std::vector<float> carrier_route_dp(
            static_cast<std::size_t>(carrier_count) * kCarrierRouteBuckets,
            0.0f);
        std::vector<float> carrier_route_distance_dp(
            static_cast<std::size_t>(carrier_count) * kCarrierRouteBuckets,
            0.0f);
        std::vector<int> carrier_route_next(
            static_cast<std::size_t>(carrier_count) * kCarrierRouteBuckets,
            -1);
        const auto route_index = [&](const int node, const int bucket) {
            return static_cast<std::size_t>(node) * kCarrierRouteBuckets +
                   static_cast<std::size_t>(bucket);
        };
        const auto route_flow = [&](const int node, const int bucket) -> float& {
            return carrier_route_dp[route_index(node, bucket)];
        };
        const auto route_distance =
            [&](const int node, const int bucket) -> float& {
            return carrier_route_distance_dp[route_index(node, bucket)];
        };
        const auto route_next = [&](const int node, const int bucket) -> int& {
            return carrier_route_next[route_index(node, bucket)];
        };
        {
            const TimingMark route_dp_timing = start_timing();
            for (int node = 0; node < carrier_count; ++node) {
                route_flow(node, 0) = carrier_value[node];
                route_distance(node, 0) = 0.0f;
            }
            for (int bucket = 1; bucket < kCarrierRouteBuckets; ++bucket) {
                const float budget = bucket * kCarrierRouteBucketPx;
                for (int node = 0; node < carrier_count; ++node) {
                    float best_flow = carrier_value[node];
                    float best_distance = 0.0f;
                    int best_next = -1;
                    for (const CarrierNeighbor neighbor : carrier_edges[node]) {
                        float candidate_flow = carrier_value[neighbor.node];
                        float candidate_distance =
                            std::min(neighbor.distance, budget);
                        if (neighbor.distance >= budget - kFlowEpsilon) {
                            const float t = neighbor.distance > 0.0f
                                                ? std::clamp(
                                                      budget / neighbor.distance,
                                                      0.0f, 1.0f)
                                                : 1.0f;
                            candidate_flow =
                                carrier_value[node] * (1.0f - t) +
                                carrier_value[neighbor.node] * t;
                        } else {
                            const int next_bucket = std::clamp(
                                static_cast<int>(
                                    std::floor((budget - neighbor.distance) /
                                               kCarrierRouteBucketPx)),
                                0, bucket - 1);
                            candidate_flow =
                                route_flow(neighbor.node, next_bucket);
                            candidate_distance =
                                neighbor.distance +
                                route_distance(neighbor.node, next_bucket);
                        }
                        const bool better_flow =
                            candidate_flow > best_flow + kFlowEpsilon;
                        const bool same_flow_longer =
                            std::abs(candidate_flow - best_flow) <=
                                kFlowEpsilon &&
                            candidate_distance > best_distance + kFlowEpsilon;
                        const bool same_flow_same_distance_stable =
                            std::abs(candidate_flow - best_flow) <=
                                kFlowEpsilon &&
                            std::abs(candidate_distance - best_distance) <=
                                kFlowEpsilon &&
                            (best_next < 0 || neighbor.node < best_next);
                        if (better_flow || same_flow_longer ||
                            same_flow_same_distance_stable) {
                            best_flow = candidate_flow;
                            best_distance = candidate_distance;
                            best_next = neighbor.node;
                        }
                    }
                    route_flow(node, bucket) = best_flow;
                    route_distance(node, bucket) = best_distance;
                    route_next(node, bucket) = best_next;
                }
            }
            result.timings.push_back(
                finish_timing("dense_grid.route_dp", route_dp_timing));
        }
        const int route_final_bucket = kCarrierRouteBuckets - 1;

        const auto print_carrier_debug = [&](const cv::Point query) {
            int best_node = -1;
            double best_distance = std::numeric_limits<double>::max();
            for (int node = 0; node < static_cast<int>(carriers.size());
                 ++node) {
                const double distance = cv::norm(query - carriers[node].pixel);
                if (distance < best_distance) {
                    best_distance = distance;
                    best_node = node;
                }
            }
            std::cout << "Carrier debug query=(" << query.x << "," << query.y
                      << ")\n";
            if (query.x >= 0 && query.x < cols && query.y >= 0 &&
                query.y < rows) {
                const int query_index = linear_index(query);
                std::cout << "  query_white="
                          << static_cast<int>(
                                 white_domain.at<std::uint8_t>(query.y,
                                                               query.x))
                          << " query_routed_flow="
                          << routed_flow[query_index]
                          << " query_graph_px="
                          << graph_pixel_flow.at<float>(query.y, query.x)
                          << " query_graph_node="
                          << graph_node_flow.at<float>(query.y, query.x)
                          << " query_dt=" << dt.at<float>(query.y, query.x)
                          << "\n";
            }
            if (best_node < 0) {
                std::cout << "  no carrier nodes\n";
                return;
            }
            const CarrierNode& carrier = carriers[best_node];
            std::cout << "  nearest_carrier=" << best_node
                      << " pixel=(" << carrier.pixel.x << ","
                      << carrier.pixel.y << ")"
                      << " distance=" << best_distance
                      << " is_grid=" << carrier.grid
                      << " capacity=" << carrier.capacity
                      << " source_flow=" << carrier.source_flow
                      << " smooth_flow=" << carrier_value[best_node]
                      << " route300="
                      << route_flow(best_node, route_final_bucket)
                      << " route_dist="
                      << route_distance(best_node, route_final_bucket)
                      << " route_next="
                      << route_next(best_node, route_final_bucket)
                      << " degree=" << carrier_edges[best_node].size()
                      << "\n";
            std::vector<CarrierNeighbor> neighbors = carrier_edges[best_node];
            std::sort(neighbors.begin(), neighbors.end(),
                      [&](const CarrierNeighbor a, const CarrierNeighbor b) {
                          return carriers[a.node].capacity >
                                 carriers[b.node].capacity;
                      });
            const int limit =
                std::min(12, static_cast<int>(neighbors.size()));
            for (int i = 0; i < limit; ++i) {
                const CarrierNeighbor neighbor = neighbors[i];
                const CarrierNode& next = carriers[neighbor.node];
                std::cout << "    neighbor_" << i
                          << " id=" << neighbor.node
                          << " pixel=(" << next.pixel.x << ","
                          << next.pixel.y << ")"
                          << " is_grid=" << next.grid
                          << " dist=" << neighbor.distance
                          << " capacity=" << next.capacity
                          << " source_flow=" << next.source_flow
                          << " smooth_flow="
                          << carrier_value[neighbor.node]
                          << " route300="
                          << route_flow(neighbor.node, route_final_bucket)
                          << " route_dist="
                          << route_distance(neighbor.node, route_final_bucket)
                          << " route_next="
                          << route_next(neighbor.node, route_final_bucket)
                          << " uphill="
                          << (next.capacity > carrier.capacity + kFlowEpsilon)
                          << "\n";
            }
        };
        const auto grid_node_id = [&](const int gx, const int gy) {
            if (gx < 0 || gx >= grid_cols || gy < 0 || gy >= grid_rows) {
                return -1;
            }
            return grid_node_ids[static_cast<std::size_t>(gy * grid_cols + gx)];
        };
        const auto lower_axis_index = [](const std::vector<int>& axis,
                                         const int value) {
            const auto upper = std::upper_bound(axis.begin(), axis.end(), value);
            if (upper == axis.begin()) {
                return 0;
            }
            return static_cast<int>(std::distance(axis.begin(), upper)) - 1;
        };

        if (enable_debug_outputs) {
            const TimingMark debug_paths_timing = start_timing();
            const std::array<cv::Point, 2> kDebugGridPoints = {
                cv::Point(236, 180), cv::Point(240, 184)};
            const std::array<cv::Scalar, 2> kDebugColors = {
                cv::Scalar(255.0, 0.0, 0.0),
                cv::Scalar(0.0, 255.0, 0.0)};
            for (int query_id = 0;
                 query_id < static_cast<int>(kDebugGridPoints.size());
                 ++query_id) {
                const cv::Point query = kDebugGridPoints[query_id];
                std::cout << "Grid carrier route debug point=(" << query.x
                          << "," << query.y << ")\n";
                const auto gx_it =
                    std::find(grid_xs.begin(), grid_xs.end(), query.x);
                const auto gy_it =
                    std::find(grid_ys.begin(), grid_ys.end(), query.y);
                if (gx_it == grid_xs.end() || gy_it == grid_ys.end()) {
                    std::cout << "  skipped: not an exact grid point\n";
                    continue;
                }
                const int gx =
                    static_cast<int>(std::distance(grid_xs.begin(), gx_it));
                const int gy =
                    static_cast<int>(std::distance(grid_ys.begin(), gy_it));
                int current = grid_node_id(gx, gy);
                if (current < 0) {
                    std::cout << "  skipped: no grid carrier\n";
                    continue;
                }
                int bucket = route_final_bucket;
                int steps = 0;
                std::cout << "  node=" << current
                          << " capacity=" << carriers[current].capacity
                          << " source_flow=" << carriers[current].source_flow
                          << " smooth_flow=" << carrier_value[current]
                          << " route_flow="
                          << route_flow(current, route_final_bucket)
                          << " route_dist="
                          << route_distance(current, route_final_bucket)
                          << " route_next="
                          << route_next(current, route_final_bucket) << "\n";
                while (current >= 0 && bucket > 0 &&
                       steps < static_cast<int>(carriers.size())) {
                    const int next = route_next(current, bucket);
                    if (next < 0) {
                        break;
                    }
                    float edge_distance = 0.0f;
                    for (const CarrierNeighbor neighbor :
                         carrier_edges[current]) {
                        if (neighbor.node == next) {
                            edge_distance = neighbor.distance;
                            break;
                        }
                    }
                    cv::line(result.debug_paths, carriers[current].pixel,
                             carriers[next].pixel, kDebugColors[query_id], 1,
                             cv::LINE_8);
                    const int bucket_delta = std::max(
                        1, static_cast<int>(
                               std::ceil(edge_distance /
                                         kCarrierRouteBucketPx)));
                    current = next;
                    bucket = std::max(0, bucket - bucket_delta);
                    ++steps;
                }
            }
            result.timings.push_back(
                finish_timing("dense_grid.debug_paths", debug_paths_timing));
        } else {
            result.timings.push_back({"dense_grid.debug_paths_skipped", 0.0,
                                      0.0});
        }

        if constexpr (false) {
            const std::array<cv::Point, 2> kDebugQueries = {
                cv::Point(242, 186), cv::Point(241, 185)};
            const std::array<std::array<cv::Scalar, 4>, 2> kDebugColors = {{
                {cv::Scalar(255.0, 0.0, 0.0),
                 cv::Scalar(0.0, 255.0, 0.0),
                 cv::Scalar(0.0, 0.0, 255.0),
                 cv::Scalar(255.0, 255.0, 0.0)},
                {cv::Scalar(255.0, 0.0, 255.0),
                 cv::Scalar(0.0, 255.0, 255.0),
                 cv::Scalar(255.0, 128.0, 0.0),
                 cv::Scalar(128.0, 255.0, 0.0)},
            }};
            const auto dark_color = [](const cv::Scalar color) {
                return cv::Scalar(color[0] * 0.35, color[1] * 0.35,
                                  color[2] * 0.35);
            };
            for (int query_id = 0;
                 query_id < static_cast<int>(kDebugQueries.size());
                 ++query_id) {
                const cv::Point query = kDebugQueries[query_id];
                std::cout << "Grid route debug query=(" << query.x << ","
                          << query.y << ")\n";
                if (query.x < 0 || query.x >= cols || query.y < 0 ||
                    query.y >= rows ||
                    white_domain.at<std::uint8_t>(query.y, query.x) == 0) {
                    std::cout << "  skipped: outside image or not in white "
                                 "domain\n";
                    continue;
                }
                const int query_index = linear_index(query);
                const int gx0 = lower_axis_index(grid_xs, query.x);
                const int gy0 = lower_axis_index(grid_ys, query.y);
                const int gx1 = std::min(gx0 + 1, grid_cols - 1);
                const int gy1 = std::min(gy0 + 1, grid_rows - 1);
                const float x0 = static_cast<float>(grid_xs[gx0]);
                const float x1 = static_cast<float>(grid_xs[gx1]);
                const float y0 = static_cast<float>(grid_ys[gy0]);
                const float y1 = static_cast<float>(grid_ys[gy1]);
                const float tx =
                    x1 > x0 ? (static_cast<float>(query.x) - x0) / (x1 - x0)
                            : 0.0f;
                const float ty =
                    y1 > y0 ? (static_cast<float>(query.y) - y0) / (y1 - y0)
                            : 0.0f;
                const std::array<std::tuple<int, int, double>, 4> corners = {
                    std::make_tuple(gx0, gy0, (1.0 - tx) * (1.0 - ty)),
                    std::make_tuple(gx1, gy0, tx * (1.0 - ty)),
                    std::make_tuple(gx0, gy1, (1.0 - tx) * ty),
                    std::make_tuple(gx1, gy1, tx * ty)};
                double weighted_sum = 0.0;
                double weight_sum = 0.0;
                double smooth_weighted_sum = 0.0;
                double smooth_weight_sum = 0.0;
                for (int i = 0; i < static_cast<int>(corners.size()); ++i) {
                    const auto [gx, gy, bilinear_weight] = corners[i];
                    const int node = grid_node_id(gx, gy);
                    if (bilinear_weight <= 0.0 || node < 0) {
                        continue;
                    }
                    if (!line_in_white(query, carriers[node].pixel)) {
                        std::cout << "  corner_" << i
                                  << " skipped: not line-visible\n";
                        continue;
                    }
                    weighted_sum += route_flow(node, route_final_bucket) *
                                    bilinear_weight;
                    weight_sum += bilinear_weight;
                    smooth_weighted_sum += carrier_value[node] *
                                           bilinear_weight;
                    smooth_weight_sum += bilinear_weight;
                    const cv::Scalar bright = kDebugColors[query_id][i];
                    const cv::Scalar dark = dark_color(bright);
                    const cv::Point corner_pixel = carriers[node].pixel;
                    cv::line(result.debug_paths, query, corner_pixel, dark, 1,
                             cv::LINE_8);
                    int current = node;
                    int bucket = route_final_bucket;
                    int steps = 0;
                    while (current >= 0 && bucket > 0 &&
                           steps < static_cast<int>(carriers.size())) {
                        const int next = route_next(current, bucket);
                        if (next < 0) {
                            break;
                        }
                        float edge_distance = 0.0f;
                        for (const CarrierNeighbor neighbor :
                             carrier_edges[current]) {
                            if (neighbor.node == next) {
                                edge_distance = neighbor.distance;
                                break;
                            }
                        }
                        cv::line(result.debug_paths, carriers[current].pixel,
                                 carriers[next].pixel, bright, 1,
                                 cv::LINE_8);
                        const int bucket_delta = std::max(
                            1, static_cast<int>(
                                   std::ceil(edge_distance /
                                             kCarrierRouteBucketPx)));
                        current = next;
                        bucket = std::max(0, bucket - bucket_delta);
                        ++steps;
                    }
                    std::cout << "  corner_" << i << " node=" << node
                              << " pixel=(" << corner_pixel.x << ","
                              << corner_pixel.y << ")"
                              << " weight=" << bilinear_weight
                              << " capacity=" << carriers[node].capacity
                              << " source_flow=" << carriers[node].source_flow
                              << " smooth_flow=" << carrier_value[node]
                              << " route300="
                              << route_flow(node, route_final_bucket)
                              << " route_dist="
                              << route_distance(node, route_final_bucket)
                              << " route_next="
                              << route_next(node, route_final_bucket) << "\n";
                }
                std::cout << "  query_routed_flow="
                          << routed_flow[query_index]
                          << " query_grid_value="
                          << (weight_sum > 0.0 ? weighted_sum / weight_sum
                                                : 0.0)
                          << " query_smooth_grid_value="
                          << (smooth_weight_sum > 0.0
                                  ? smooth_weighted_sum / smooth_weight_sum
                                  : 0.0)
                          << " weight_sum=" << weight_sum << "\n";
            }
        }

        {
            const TimingMark dense_render_timing = start_timing();
            cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
                for (int y = range.start; y < range.end; ++y) {
                    for (int x = 0; x < cols; ++x) {
                        const int index = y * cols + x;
                        if (white_domain.at<std::uint8_t>(y, x) == 0) {
                            continue;
                        }
                        result.nn_flow.at<float>(y, x) = routed_flow[index];
                        const cv::Point pixel(x, y);
                        const int gx0 = lower_axis_index(grid_xs, x);
                        const int gy0 = lower_axis_index(grid_ys, y);
                        const int gx1 = std::min(gx0 + 1, grid_cols - 1);
                        const int gy1 = std::min(gy0 + 1, grid_rows - 1);
                        const float x0 = static_cast<float>(grid_xs[gx0]);
                        const float x1 = static_cast<float>(grid_xs[gx1]);
                        const float y0 = static_cast<float>(grid_ys[gy0]);
                        const float y1 = static_cast<float>(grid_ys[gy1]);
                        const float tx =
                            x1 > x0
                                ? (static_cast<float>(x) - x0) / (x1 - x0)
                                : 0.0f;
                        const float ty =
                            y1 > y0
                                ? (static_cast<float>(y) - y0) / (y1 - y0)
                                : 0.0f;
                        const std::array<std::tuple<int, int, double>, 4>
                            corners = {
                                std::make_tuple(gx0, gy0,
                                                (1.0 - tx) * (1.0 - ty)),
                                std::make_tuple(gx1, gy0, tx * (1.0 - ty)),
                                std::make_tuple(gx0, gy1, (1.0 - tx) * ty),
                                std::make_tuple(gx1, gy1, tx * ty)};
                        double weighted_sum = 0.0;
                        double weight_sum = 0.0;
                        double smooth_weighted_sum = 0.0;
                        double smooth_weight_sum = 0.0;
                        for (const auto [gx, gy, bilinear_weight] : corners) {
                            if (bilinear_weight <= 0.0) {
                                continue;
                            }
                            const int node = grid_node_id(gx, gy);
                            if (node < 0) {
                                continue;
                            }
                            if (!line_in_white(pixel, carriers[node].pixel)) {
                                continue;
                            }
                            weighted_sum +=
                                route_flow(node, route_final_bucket) *
                                bilinear_weight;
                            weight_sum += bilinear_weight;
                            smooth_weighted_sum += carrier_value[node] *
                                                   bilinear_weight;
                            smooth_weight_sum += bilinear_weight;
                        }
                        result.flow.at<float>(y, x) =
                            weight_sum > 0.0
                                ? static_cast<float>(weighted_sum / weight_sum)
                                : routed_flow[index];
                        result.smooth_grid_flow.at<float>(y, x) =
                            smooth_weight_sum > 0.0
                                ? static_cast<float>(smooth_weighted_sum /
                                                     smooth_weight_sum)
                                : routed_flow[index];
                    }
                }
            });
            result.timings.push_back(
                finish_timing("dense_grid.dense_render", dense_render_timing));
        }

        if (enable_debug_outputs) {
            const TimingMark carrier_debug_timing = start_timing();
            result.carrier_debug =
                cv::Mat(white_domain.size(), CV_32F, cv::Scalar(0.0f));
            for (int y = 0; y < rows; ++y) {
                float* out_row = result.carrier_debug.ptr<float>(y);
                for (int x = 0; x < cols; ++x) {
                    if (white_domain.at<std::uint8_t>(y, x) == 0) {
                        continue;
                    }
                    const cv::Point pixel(x, y);
                    const int gx0 = lower_axis_index(grid_xs, x);
                    const int gy0 = lower_axis_index(grid_ys, y);
                    const int gx1 = std::min(gx0 + 1, grid_cols - 1);
                    const int gy1 = std::min(gy0 + 1, grid_rows - 1);
                    const float x0 = static_cast<float>(grid_xs[gx0]);
                    const float x1 = static_cast<float>(grid_xs[gx1]);
                    const float y0 = static_cast<float>(grid_ys[gy0]);
                    const float y1 = static_cast<float>(grid_ys[gy1]);
                    const float tx =
                        x1 > x0 ? (static_cast<float>(x) - x0) / (x1 - x0)
                                : 0.0f;
                    const float ty =
                        y1 > y0 ? (static_cast<float>(y) - y0) / (y1 - y0)
                                : 0.0f;
                    const std::array<std::tuple<int, int, double>, 4>
                        corners = {
                            std::make_tuple(gx0, gy0,
                                            (1.0 - tx) * (1.0 - ty)),
                            std::make_tuple(gx1, gy0, tx * (1.0 - ty)),
                            std::make_tuple(gx0, gy1, (1.0 - tx) * ty),
                            std::make_tuple(gx1, gy1, tx * ty)};
                    double weighted_sum = 0.0;
                    double weight_sum = 0.0;
                    for (const auto [gx, gy, bilinear_weight] : corners) {
                        if (bilinear_weight <= 0.0) {
                            continue;
                        }
                        const int node = grid_node_id(gx, gy);
                        if (node < 0) {
                            continue;
                        }
                        if (!line_in_white(pixel, carriers[node].pixel)) {
                            continue;
                        }
                        weighted_sum += carrier_value[node] * bilinear_weight;
                        weight_sum += bilinear_weight;
                    }
                    out_row[x] = weight_sum > 0.0
                                     ? static_cast<float>(weighted_sum /
                                                          weight_sum)
                                     : routed_flow[linear_index(pixel)];
                }
            }
            for (int id = 0; id < static_cast<int>(carriers.size()); ++id) {
                const CarrierNode& carrier = carriers[id];
                if (carrier.grid) {
                    continue;
                }
                const float value = carrier_value[id];
                if (value <= 0.0f) {
                    continue;
                }
                cv::circle(result.carrier_debug, carrier.pixel, 2,
                           cv::Scalar(value), cv::FILLED, cv::LINE_8);
            }
            result.timings.push_back(
                finish_timing("carrier_debug_render", carrier_debug_timing));
        } else {
            result.timings.push_back({"carrier_debug_render_skipped", 0.0,
                                      0.0});
        }

        int carrier_edges_count = 0;
        for (const std::vector<CarrierNeighbor>& edges : carrier_edges) {
            carrier_edges_count += static_cast<int>(edges.size());
        }
        int improved_carriers = 0;
        float min_route_distance = std::numeric_limits<float>::infinity();
        float max_route_distance = 0.0f;
        for (int node = 0; node < static_cast<int>(carriers.size()); ++node) {
            if (route_next(node, route_final_bucket) >= 0) {
                ++improved_carriers;
                min_route_distance =
                    std::min(min_route_distance,
                             route_distance(node, route_final_bucket));
                max_route_distance =
                    std::max(max_route_distance,
                             route_distance(node, route_final_bucket));
            }
        }
        if (improved_carriers == 0) {
            min_route_distance = 0.0f;
        }
        std::cout << "Dense grid carrier backtrack:\n"
                  << "  grid_step: " << kGridStep << "\n"
                  << "  grid_carriers: " << grid_carriers << "\n"
                  << "  graph_carriers: " << graph_carriers << "\n"
                  << "  graph_node_carriers: " << graph_node_carriers << "\n"
                  << "  graph_edge_carriers: " << graph_edge_carriers << "\n"
                  << "  carrier_edges: " << (carrier_edges_count / 2) << "\n"
                  << "  carrier_flow_seeds: " << carrier_flow_seeds << "\n"
                  << "  carrier_flow_reached: " << carrier_flow_reached
                  << "\n"
                  << "  route_bucket_px: " << kCarrierRouteBucketPx << "\n"
                  << "  route_buckets: " << kCarrierRouteBuckets << "\n"
                  << "  improved_carriers: " << improved_carriers << "\n"
                  << "  min_route_distance: " << min_route_distance << "\n"
                  << "  max_route_distance: " << max_route_distance << "\n";
        StageTiming dense_grid_total =
            finish_timing("dense_backtrack_grid_carrier",
                          dense_grid_outer_timing);
        for (std::size_t i = dense_grid_timing_start;
             i < result.timings.size(); ++i) {
            const std::string& name = result.timings[i].name;
            if (name == "dense_grid.debug_paths" ||
                name == "carrier_debug_render") {
                dense_grid_total.elapsed_ms =
                    std::max(0.0, dense_grid_total.elapsed_ms -
                                      result.timings[i].elapsed_ms);
                dense_grid_total.cpu_ms =
                    std::max(0.0,
                             dense_grid_total.cpu_ms - result.timings[i].cpu_ms);
            }
        }
        result.timings.push_back(dense_grid_total);
    }

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (white_domain.at<std::uint8_t>(y, x) == 0) {
                continue;
            }
            const int index = y * cols + x;
            const float reached_value = kEnablePixelDenseBacktrackFlood
                                            ? routed_flow[index]
                                            : result.flow.at<float>(y, x);
            if (reached_value > 0.0f) {
                ++result.reached_pixels;
            } else {
                ++result.unreached_white_pixels;
            }
        }
    }

    return result;

    {
        const TimingMark timing = start_timing();
        const std::array<cv::Point, 8> kDirs = {
            cv::Point(-1, -1), cv::Point(0, -1), cv::Point(1, -1),
            cv::Point(-1, 0),                    cv::Point(1, 0),
            cv::Point(-1, 1),  cv::Point(0, 1),  cv::Point(1, 1)};
        struct GraphScoreItem {
            float score = 0.0f;
            float distance = 0.0f;
            int index = -1;
            int source = -1;
            bool operator<(const GraphScoreItem& other) const {
                if (score != other.score) {
                    return score < other.score;
                }
                return distance > other.distance;
            }
        };
        std::vector<float> best_score(static_cast<std::size_t>(pixel_count),
                                      -std::numeric_limits<float>::infinity());
        std::vector<float> best_distance(
            static_cast<std::size_t>(pixel_count),
            std::numeric_limits<float>::infinity());
        std::vector<float> nearest_graph_distance(
            static_cast<std::size_t>(pixel_count),
            std::numeric_limits<float>::infinity());
        std::vector<int> graph_source_id(static_cast<std::size_t>(pixel_count),
                                         -1);
        std::vector<int> graph_sources;
        graph_sources.reserve(static_cast<std::size_t>(pixel_count / 32));

        for (int index = 0; index < pixel_count; ++index) {
            if (graph_route_pixel[index] == 0 || routed_flow[index] <= 0.0f) {
                continue;
            }
            graph_source_id[index] = static_cast<int>(graph_sources.size());
            graph_sources.push_back(index);
        }

        struct GraphDistanceItem {
            float distance = 0.0f;
            int index = -1;
            bool operator<(const GraphDistanceItem& other) const {
                return distance > other.distance;
            }
        };
        std::priority_queue<GraphDistanceItem> distance_queue;
        for (const int index : graph_sources) {
            nearest_graph_distance[index] = 0.0f;
            distance_queue.push({0.0f, index});
        }
        while (!distance_queue.empty()) {
            const GraphDistanceItem item = distance_queue.top();
            distance_queue.pop();
            if (item.distance >
                nearest_graph_distance[item.index] + kFlowEpsilon) {
                continue;
            }
            const cv::Point pixel = point_from_index(item.index);
            for (const cv::Point dir : kDirs) {
                const cv::Point next = pixel + dir;
                if (!in_white_domain(next)) {
                    continue;
                }
                const int next_index = linear_index(next);
                if (routed_flow[next_index] <= 0.0f) {
                    continue;
                }
                const float candidate =
                    item.distance + step_distance(pixel, next);
                if (candidate + kFlowEpsilon >=
                    nearest_graph_distance[next_index]) {
                    continue;
                }
                nearest_graph_distance[next_index] = candidate;
                distance_queue.push({candidate, next_index});
            }
        }

        constexpr int kScoreBucketSize = 4;
        const int kScoreBucketCount =
            static_cast<int>(kBacktrackRadius) / kScoreBucketSize + 1;
        constexpr float kMaxScoreDistanceMargin = 32.0f;
        std::vector<float> graph_source_score_cache(
            graph_sources.size() * static_cast<std::size_t>(kScoreBucketCount),
            std::numeric_limits<float>::quiet_NaN());
        const auto graph_endpoint_score = [&](const int source,
                                              const float remaining) {
            const int bucket = std::clamp(
                cvRound(std::max(0.0f, remaining) /
                        static_cast<float>(kScoreBucketSize)),
                0, kScoreBucketCount - 1);
            float& cached =
                graph_source_score_cache
                    [static_cast<std::size_t>(source) * kScoreBucketCount +
                     bucket];
            if (!std::isnan(cached)) {
                return cached;
            }
            float budget = static_cast<float>(bucket * kScoreBucketSize);
            int current_index = graph_sources[source];
            float endpoint_flow = routed_flow[current_index];
            while (budget > kFlowEpsilon && current_index >= 0) {
                const int next_index = graph_next_pixel[current_index];
                const float step = graph_next_distance[current_index];
                if (next_index < 0 || step <= 0.0f) {
                    break;
                }
                const float segment = std::min(step, budget);
                const float t = step > 0.0f ? segment / step : 0.0f;
                endpoint_flow = routed_flow[current_index] * (1.0f - t) +
                                routed_flow[next_index] * t;
                if (step >= budget - kFlowEpsilon) {
                    break;
                }
                budget -= step;
                current_index = next_index;
            }
            cached = endpoint_flow;
            return cached;
        };

        std::priority_queue<GraphScoreItem> score_queue;
        for (int source = 0; source < static_cast<int>(graph_sources.size());
             ++source) {
            const int index = graph_sources[source];
            const float score = graph_endpoint_score(source, kBacktrackRadius);
            best_score[index] = score;
            best_distance[index] = 0.0f;
            score_queue.push({score, 0.0f, index, source});
        }

        int attraction_reached = 0;
        int attraction_updates = 0;
        while (!score_queue.empty()) {
            const GraphScoreItem item = score_queue.top();
            score_queue.pop();
            if (item.score + kFlowEpsilon < best_score[item.index] ||
                (std::abs(item.score - best_score[item.index]) <=
                     kFlowEpsilon &&
                 item.distance > best_distance[item.index] + kFlowEpsilon)) {
                continue;
            }
            ++attraction_reached;
            const cv::Point pixel = point_from_index(item.index);
            for (const cv::Point dir : kDirs) {
                const cv::Point next = pixel + dir;
                if (!in_white_domain(next)) {
                    continue;
                }
                const int next_index = linear_index(next);
                if (routed_flow[next_index] <= 0.0f) {
                    continue;
                }
                const float step = step_distance(pixel, next);
                const float candidate_distance = item.distance + step;
                if (candidate_distance >
                    nearest_graph_distance[next_index] +
                        kMaxScoreDistanceMargin + kFlowEpsilon) {
                    continue;
                }
                const float remaining =
                    std::max(0.0f, kBacktrackRadius - candidate_distance);
                const float candidate_score =
                    graph_endpoint_score(item.source, remaining);
                const bool better_score =
                    candidate_score > best_score[next_index] + kFlowEpsilon;
                const bool same_score_shorter =
                    std::abs(candidate_score - best_score[next_index]) <=
                        kFlowEpsilon &&
                    candidate_distance + kFlowEpsilon <
                        best_distance[next_index];
                if (!better_score && !same_score_shorter) {
                    continue;
                }
                best_score[next_index] = candidate_score;
                best_distance[next_index] = candidate_distance;
                scalar_next_pixel[next_index] = item.index;
                scalar_next_distance[next_index] = step;
                ++attraction_updates;
                score_queue.push(
                    {candidate_score, candidate_distance, next_index,
                     item.source});
            }
        }

        std::cout << "Dense graph max-score attraction:\n"
                  << "  attraction_seeds: " << graph_sources.size() << "\n"
                  << "  distance_margin: " << kMaxScoreDistanceMargin << "\n"
                  << "  attraction_reached: " << attraction_reached << "\n"
                  << "  attraction_updates: " << attraction_updates << "\n";
        result.timings.push_back(
            finish_timing("dense_backtrack_parent_max_score", timing));
    }

    int jump_levels = 1;
    while ((1 << jump_levels) <= static_cast<int>(kBacktrackRadius) + 2) {
        ++jump_levels;
    }
    std::vector<std::vector<int>> jump_to(
        jump_levels, std::vector<int>(static_cast<std::size_t>(pixel_count), -1));
    std::vector<std::vector<float>> jump_dist(
        jump_levels, std::vector<float>(static_cast<std::size_t>(pixel_count),
                                        0.0f));
    {
        const TimingMark timing = start_timing();
        for (int index = 0; index < pixel_count; ++index) {
            jump_to[0][index] = next_pixel[index];
            jump_dist[0][index] = next_distance[index];
        }
        for (int level = 1; level < jump_levels; ++level) {
            for (int index = 0; index < pixel_count; ++index) {
                const int mid = jump_to[level - 1][index];
                if (mid < 0) {
                    continue;
                }
                const int end = jump_to[level - 1][mid];
                if (end < 0) {
                    continue;
                }
                jump_to[level][index] = end;
                jump_dist[level][index] =
                    jump_dist[level - 1][index] + jump_dist[level - 1][mid];
            }
        }
        result.timings.push_back(finish_timing("dense_backtrack_jump", timing));
    }

    {
        const TimingMark timing = start_timing();
        cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                for (int x = 0; x < cols; ++x) {
                    if (white_domain.at<std::uint8_t>(y, x) == 0) {
                        continue;
                    }
                    int current_index = y * cols + x;
                    if (routed_flow[current_index] <= 0.0f) {
                        continue;
                    }
                    result.nn_flow.at<float>(y, x) =
                        routed_flow[current_index];
                    float remaining = kBacktrackRadius;
                    float endpoint_flow = routed_flow[current_index];
                    bool graph_mode = false;
                    while (remaining > kFlowEpsilon && current_index >= 0) {
                        if (graph_route_pixel[current_index] != 0) {
                            graph_mode = true;
                        }
                        const int next_index =
                            graph_mode ? graph_next_pixel[current_index]
                                       : scalar_next_pixel[current_index];
                        const float step =
                            graph_mode ? graph_next_distance[current_index]
                                       : scalar_next_distance[current_index];
                        if (next_index < 0 || step <= 0.0f) {
                            break;
                        }
                        const float segment = std::min(step, remaining);
                        const float t = step > 0.0f ? segment / step : 0.0f;
                        endpoint_flow =
                            routed_flow[current_index] * (1.0f - t) +
                            routed_flow[next_index] * t;
                        if (step >= remaining - kFlowEpsilon) {
                            break;
                        }
                        remaining -= step;
                        current_index = next_index;
                    }
                    result.flow.at<float>(y, x) = endpoint_flow;
                }
            }
        });
        result.timings.push_back(
            finish_timing("dense_backtrack_resolve", timing));
    }

    {
        const TimingMark timing = start_timing();
        const std::array<cv::Point, 2> kDebugPoints = {
            cv::Point(980, 749), cv::Point(980, 752)};
        const std::array<cv::Scalar, 2> kColors = {
            cv::Scalar(255.0, 0.0, 0.0), cv::Scalar(0.0, 255.0, 0.0)};
        const std::array<cv::Scalar, 2> kDirectColors = {
            cv::Scalar(0.0, 128.0, 255.0), cv::Scalar(255.0, 255.0, 0.0)};
        const std::array<cv::Point, 8> kDirs = {
            cv::Point(-1, -1), cv::Point(0, -1), cv::Point(1, -1),
            cv::Point(-1, 0),                    cv::Point(1, 0),
            cv::Point(-1, 1),  cv::Point(0, 1),  cv::Point(1, 1)};
        struct DirectSearchItem {
            float distance = 0.0f;
            int index = -1;
            bool operator<(const DirectSearchItem& other) const {
                return distance > other.distance;
            }
        };
        for (std::size_t point_index = 0; point_index < kDebugPoints.size();
             ++point_index) {
            const cv::Point query = kDebugPoints[point_index];
            std::cout << "Dense backtrack debug point " << point_index
                      << " query=(" << query.x << "," << query.y << ")\n";
            if (!in_white_domain(query)) {
                std::cout << "  skipped: outside image or not in white domain\n";
                continue;
            }
            int current_index = linear_index(query);
            const int first_next = graph_route_pixel[current_index] != 0
                                       ? graph_next_pixel[current_index]
                                       : scalar_next_pixel[current_index];
            std::cout << "  start_flow=" << routed_flow[current_index]
                      << " next=" << first_next << "\n";
            float remaining = kBacktrackRadius;
            int steps = 0;
            bool graph_mode = false;
            while (remaining > kFlowEpsilon && current_index >= 0) {
                const cv::Point current = point_from_index(current_index);
                if (graph_route_pixel[current_index] != 0) {
                    graph_mode = true;
                }
                const int next_index =
                    graph_mode ? graph_next_pixel[current_index]
                               : scalar_next_pixel[current_index];
                const float step =
                    graph_mode ? graph_next_distance[current_index]
                               : scalar_next_distance[current_index];
                if (next_index < 0 || step <= 0.0f) {
                    break;
                }
                const cv::Point next = point_from_index(next_index);
                if (steps < 40) {
                    std::cout << "    step " << steps << ": current=("
                              << current.x << "," << current.y
                              << ") flow=" << routed_flow[current_index]
                              << " mode="
                              << (graph_mode ? "graph" : "scalar")
                              << " next=(" << next.x << "," << next.y
                              << ") next_flow=" << routed_flow[next_index]
                              << " step_len=" << step
                              << " remaining=" << remaining << "\n";
                }
                cv::line(result.debug_paths, current, next, kColors[point_index],
                         1, cv::LINE_8);
                if (step >= remaining - kFlowEpsilon) {
                    break;
                }
                remaining -= step;
                current_index = next_index;
                ++steps;
            }
            std::cout << "  finished: steps=" << steps
                      << " remaining=" << remaining
                      << " final_index=" << current_index << "\n";

            const int query_index = linear_index(query);
            std::vector<float> direct_distance(
                static_cast<std::size_t>(pixel_count),
                std::numeric_limits<float>::infinity());
            std::vector<int> direct_parent(
                static_cast<std::size_t>(pixel_count), -1);
            std::priority_queue<DirectSearchItem> direct_queue;
            direct_distance[query_index] = 0.0f;
            direct_queue.push({0.0f, query_index});
            int direct_goal = -1;
            while (!direct_queue.empty()) {
                const DirectSearchItem item = direct_queue.top();
                direct_queue.pop();
                if (item.distance >
                    direct_distance[item.index] + kFlowEpsilon) {
                    continue;
                }
                if (item.index != query_index &&
                    graph_route_pixel[item.index] != 0) {
                    direct_goal = item.index;
                    break;
                }
                const cv::Point current = point_from_index(item.index);
                for (const cv::Point dir : kDirs) {
                    const cv::Point next = current + dir;
                    if (!in_white_domain(next)) {
                        continue;
                    }
                    const int next_index = linear_index(next);
                    if (routed_flow[next_index] <= 0.0f) {
                        continue;
                    }
                    const float candidate =
                        item.distance + step_distance(current, next);
                    if (candidate + kFlowEpsilon >=
                        direct_distance[next_index]) {
                        continue;
                    }
                    direct_distance[next_index] = candidate;
                    direct_parent[next_index] = item.index;
                    direct_queue.push({candidate, next_index});
                }
            }
            if (direct_goal < 0) {
                std::cout << "  direct_search: no routed graph pixel found\n";
            } else {
                std::vector<cv::Point> direct_path;
                for (int index = direct_goal; index >= 0;
                     index = direct_parent[index]) {
                    direct_path.push_back(point_from_index(index));
                    if (index == query_index) {
                        break;
                    }
                }
                std::reverse(direct_path.begin(), direct_path.end());
                for (std::size_t i = 1; i < direct_path.size(); ++i) {
                    cv::line(result.debug_paths, direct_path[i - 1],
                             direct_path[i], kDirectColors[point_index], 1,
                             cv::LINE_8);
                }
                const cv::Point goal = point_from_index(direct_goal);
                std::cout << "  direct_search: goal=(" << goal.x << ","
                          << goal.y << ")"
                          << " distance=" << direct_distance[direct_goal]
                          << " goal_flow=" << routed_flow[direct_goal]
                          << " path_pixels=" << direct_path.size() << "\n";
            }
        }
        result.timings.push_back(
            finish_timing("dense_backtrack_debug_render", timing));
    }

    {
        const TimingMark timing = start_timing();
        constexpr int kMaxPrintedFailures = 8;
        constexpr int kMaxPrintedSteps = 24;
        int scalar_stalls = 0;
        int graph_stalls = 0;
        int completed_walks = 0;
        int printed_failures = 0;

        const auto print_failure_path = [&](const cv::Point query,
                                            const bool scalar_failure) {
            std::cout << "  failure_sample_" << printed_failures
                      << ": query=(" << query.x << "," << query.y << ")"
                      << " phase=" << (scalar_failure ? "scalar" : "graph")
                      << " start_flow="
                      << routed_flow[linear_index(query)] << "\n";
            int current_index = linear_index(query);
            float remaining = kBacktrackRadius;
            bool graph_mode = false;
            for (int step_index = 0;
                 step_index < kMaxPrintedSteps && remaining > kFlowEpsilon &&
                 current_index >= 0;
                 ++step_index) {
                const cv::Point current = point_from_index(current_index);
                if (graph_route_pixel[current_index] != 0) {
                    graph_mode = true;
                }
                const int next_index =
                    graph_mode ? graph_next_pixel[current_index]
                               : scalar_next_pixel[current_index];
                const float step =
                    graph_mode ? graph_next_distance[current_index]
                               : scalar_next_distance[current_index];
                std::cout << "    step " << step_index << ": current=("
                          << current.x << "," << current.y << ")"
                          << " flow=" << routed_flow[current_index]
                          << " mode=" << (graph_mode ? "graph" : "scalar")
                          << " next=" << next_index
                          << " step_len=" << step
                          << " remaining=" << remaining << "\n";
                if (next_index < 0 || step <= 0.0f) {
                    break;
                }
                if (step >= remaining - kFlowEpsilon) {
                    break;
                }
                remaining -= step;
                current_index = next_index;
            }
            ++printed_failures;
        };

        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (white_domain.at<std::uint8_t>(y, x) == 0) {
                    continue;
                }
                int current_index = y * cols + x;
                if (routed_flow[current_index] <= 0.0f) {
                    continue;
                }
                float remaining = kBacktrackRadius;
                bool graph_mode = false;
                bool failed = false;
                bool scalar_failure = false;
                while (remaining > kFlowEpsilon && current_index >= 0) {
                    const cv::Point current = point_from_index(current_index);
                    if (graph_route_pixel[current_index] != 0) {
                        graph_mode = true;
                    }
                    const int next_index =
                        graph_mode ? graph_next_pixel[current_index]
                                   : scalar_next_pixel[current_index];
                    const float step =
                        graph_mode ? graph_next_distance[current_index]
                                   : scalar_next_distance[current_index];
                    if (next_index < 0 || step <= 0.0f) {
                        failed = true;
                        scalar_failure = !graph_mode;
                        break;
                    }
                    if (step >= remaining - kFlowEpsilon) {
                        break;
                    }
                    remaining -= step;
                    current_index = next_index;
                }
                if (failed) {
                    if (scalar_failure) {
                        ++scalar_stalls;
                    } else {
                        ++graph_stalls;
                    }
                    if (printed_failures < kMaxPrintedFailures) {
                        print_failure_path(cv::Point(x, y), scalar_failure);
                    }
                } else {
                    ++completed_walks;
                }
            }
        }
        std::cout << "Dense backtrack stalls:\n"
                  << "  completed_walks: " << completed_walks << "\n"
                  << "  scalar_stalls: " << scalar_stalls << "\n"
                  << "  graph_stalls: " << graph_stalls << "\n"
                  << "  printed_failure_samples: " << printed_failures
                  << "\n";
        result.timings.push_back(
            finish_timing("dense_backtrack_stall_diagnostics", timing));
    }

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (white_domain.at<std::uint8_t>(y, x) == 0) {
                continue;
            }
            const int index = y * cols + x;
            if (routed_flow[index] > 0.0f) {
                ++result.reached_pixels;
            } else {
                ++result.unreached_white_pixels;
            }
        }
    }

    return result;
}

DenseFlowResult compute_dense_source_flow(const cv::Mat& white_domain,
                                          const cv::Mat& dt,
                                          const SkeletonGraph& input_graph,
                                          const cv::Mat& source_pixel_ridges,
                                          const cv::Point source,
                                          int grid_step = 50,
                                          float backtrack_distance = 10.0f,
                                          bool enable_debug_outputs = true) {
    CV_Assert(white_domain.type() == CV_8U);
    CV_Assert(dt.type() == CV_32F);
    CV_Assert(source_pixel_ridges.type() == CV_8U);
    CV_Assert(source_pixel_ridges.size() == white_domain.size());

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
    SkeletonGraph graph = input_graph;

    cv::Mat graph_edge_mask(white_domain.size(), CV_8U, cv::Scalar(0));
    cv::Mat graph_edge_index(white_domain.size(), CV_32S, cv::Scalar(-1));
    cv::Mat graph_pixel_flow(white_domain.size(), CV_32F, cv::Scalar(0));
    const auto rebuild_graph_edge_maps = [&]() {
        graph_edge_mask.setTo(0);
        graph_edge_index.setTo(-1);
        for (int edge_index = 0;
             edge_index < static_cast<int>(graph.edges.size()); ++edge_index) {
            const GraphEdge& edge = graph.edges[edge_index];
            for (const cv::Point pixel : edge.pixels) {
                if (pixel.x >= 0 && pixel.x < graph_edge_mask.cols &&
                    pixel.y >= 0 && pixel.y < graph_edge_mask.rows) {
                    graph_edge_mask.at<std::uint8_t>(pixel.y, pixel.x) = 255;
                    graph_edge_index.at<int>(pixel.y, pixel.x) = edge_index;
                }
            }
        }
    };
    rebuild_graph_edge_maps();

    int source_seed_node = -1;
    cv::Point source_seed_pixel(-1, -1);
    float source_seed_capacity = 0.0f;
    {
        const TimingMark timing = start_timing();
        const std::array<cv::Point, 8> kDirs = {
            {{-1, -1}, {0, -1}, {1, -1}, {-1, 0},
             {1, 0},   {-1, 1}, {0, 1},  {1, 1}}};

        const auto valid_domain = [&](const cv::Point pixel) {
            return pixel.x >= 0 && pixel.x < white_domain.cols &&
                   pixel.y >= 0 && pixel.y < white_domain.rows &&
                   white_domain.at<std::uint8_t>(pixel.y, pixel.x) != 0;
        };

        int source_edge = graph_edge_index.at<int>(source.y, source.x);
        cv::Point current = source;
        cv::Mat visited = cv::Mat::zeros(white_domain.size(), CV_8U);
        constexpr float kDtEpsilon = 1.0e-4f;
        while (source_edge < 0 && valid_domain(current) &&
               visited.at<std::uint8_t>(current.y, current.x) == 0) {
            visited.at<std::uint8_t>(current.y, current.x) = 255;

            cv::Point best_edge_pixel(-1, -1);
            float best_edge_dt = -std::numeric_limits<float>::infinity();
            for (const cv::Point dir : kDirs) {
                const cv::Point neighbor = current + dir;
                if (neighbor.x < 0 || neighbor.x >= graph_edge_index.cols ||
                    neighbor.y < 0 || neighbor.y >= graph_edge_index.rows) {
                    continue;
                }
                const int edge_index =
                    graph_edge_index.at<int>(neighbor.y, neighbor.x);
                if (edge_index < 0) {
                    continue;
                }
                const float neighbor_dt = dt.at<float>(neighbor.y, neighbor.x);
                if (neighbor_dt > best_edge_dt ||
                    (neighbor_dt == best_edge_dt &&
                     (neighbor.y < best_edge_pixel.y ||
                      (neighbor.y == best_edge_pixel.y &&
                       neighbor.x < best_edge_pixel.x)))) {
                    best_edge_dt = neighbor_dt;
                    best_edge_pixel = neighbor;
                    source_edge = edge_index;
                }
            }
            if (source_edge >= 0) {
                break;
            }

            const float current_dt = dt.at<float>(current.y, current.x);
            cv::Point next(-1, -1);
            float best_dt = current_dt;
            for (const cv::Point dir : kDirs) {
                const cv::Point neighbor = current + dir;
                if (!valid_domain(neighbor) ||
                    graph.node_mask.at<std::uint8_t>(neighbor.y, neighbor.x) !=
                        0 ||
                    visited.at<std::uint8_t>(neighbor.y, neighbor.x) != 0) {
                    continue;
                }
                const float neighbor_dt = dt.at<float>(neighbor.y, neighbor.x);
                if (neighbor_dt > best_dt + kDtEpsilon ||
                    (neighbor_dt > best_dt - kDtEpsilon &&
                     next.x >= 0 && neighbor_dt == best_dt &&
                     (neighbor.y < next.y ||
                      (neighbor.y == next.y && neighbor.x < next.x)))) {
                    best_dt = neighbor_dt;
                    next = neighbor;
                }
            }
            if (next.x < 0 || best_dt <= current_dt + kDtEpsilon) {
                break;
            }
            current = next;
        }

        if (source_edge < 0) {
            cv::Mat graph_source_mask(white_domain.size(), CV_8U,
                                      cv::Scalar(255));
            graph_source_mask.setTo(0, graph_edge_mask);
            cv::Mat nearest_dt;
            cv::Mat nearest_graph_pixel;
            cv::distanceTransform(graph_source_mask, nearest_dt,
                                  nearest_graph_pixel, cv::DIST_L2,
                                  cv::DIST_MASK_5, cv::DIST_LABEL_PIXEL);
            const std::vector<cv::Point> graph_source_points =
                label_source_points(graph_source_mask, nearest_graph_pixel);
            const int label = nearest_graph_pixel.at<int>(source.y, source.x);
            if (label > 0 &&
                label < static_cast<int>(graph_source_points.size())) {
                const cv::Point graph_pixel = graph_source_points[label];
                if (graph_pixel.x >= 0) {
                    source_edge =
                        graph_edge_index.at<int>(graph_pixel.y, graph_pixel.x);
                }
            }
        }

        if (source_edge >= 0 &&
            source_edge < static_cast<int>(graph.edges.size())) {
            GraphEdge edge = graph.edges[source_edge];
            std::vector<cv::Point> ordered = order_edge_pixels(edge, graph);
            if (ordered.empty()) {
                ordered = edge.pixels;
            }
            if (!ordered.empty()) {
                int split_index = 0;
                double best_distance = std::numeric_limits<double>::max();
                for (int i = 0; i < static_cast<int>(ordered.size()); ++i) {
                    const double dx = ordered[i].x - source.x;
                    const double dy = ordered[i].y - source.y;
                    const double distance = dx * dx + dy * dy;
                    if (distance < best_distance) {
                        best_distance = distance;
                        split_index = i;
                    }
                }
                if (ordered.size() > 2) {
                    split_index =
                        std::clamp(split_index, 1,
                                   static_cast<int>(ordered.size()) - 2);
                }

                const cv::Point split_pixel = ordered[split_index];
                source_seed_pixel = split_pixel;
                source_seed_node = static_cast<int>(graph.nodes.size());
                graph.nodes.push_back(cv::Point2f(
                    static_cast<float>(split_pixel.x),
                    static_cast<float>(split_pixel.y)));
                if (graph.node_mask.empty()) {
                    graph.node_mask =
                        cv::Mat::zeros(white_domain.size(), CV_8U);
                }
                cv::circle(graph.node_mask, split_pixel, 2, cv::Scalar(255),
                           cv::FILLED, cv::LINE_8);
                source_seed_capacity =
                    capacity_from_dt(dt.at<float>(split_pixel.y, split_pixel.x));

                const auto edge_capacity_from_pixels =
                    [&](const std::vector<cv::Point>& pixels) {
                        float capacity = std::numeric_limits<float>::max();
                        for (const cv::Point pixel : pixels) {
                            capacity = std::min(
                                capacity,
                                capacity_from_dt(dt.at<float>(pixel.y,
                                                              pixel.x)));
                        }
                        return capacity == std::numeric_limits<float>::max()
                                   ? 0.0f
                                   : capacity;
                    };

                std::vector<cv::Point> first(
                    ordered.begin(), ordered.begin() + split_index + 1);
                std::vector<cv::Point> second(
                    ordered.begin() + split_index, ordered.end());
                graph.edges[source_edge] =
                    GraphEdge{edge.a, source_seed_node,
                              edge_capacity_from_pixels(first), first};
                graph.edges.push_back(
                    GraphEdge{source_seed_node, edge.b,
                              edge_capacity_from_pixels(second), second});
                rebuild_graph_edge_maps();
            }
        }
        std::cout << "Source graph split:\n"
                  << "  source_seed_node: " << source_seed_node << "\n"
                  << "  source_seed_pixel: (" << source_seed_pixel.x
                  << "," << source_seed_pixel.y << ")\n"
                  << "  source_seed_capacity: " << source_seed_capacity << "\n";
        timings.push_back(finish_timing("source_region_detect", timing));
    }

    const int node_count = static_cast<int>(graph.nodes.size());
    std::vector<char> seeded_nodes(static_cast<std::size_t>(node_count), 0);
    std::vector<double> seed_node_capacity(static_cast<std::size_t>(node_count),
                                           0.0);
    if (source_seed_node > 0 && source_seed_node < node_count &&
        source_seed_capacity > 0.0f) {
        seeded_nodes[source_seed_node] = 1;
        seed_node_capacity[source_seed_node] = source_seed_capacity;
    }
    std::vector<char> source_edges(graph.edges.size(), 0);
    for (int edge_index = 0; edge_index < static_cast<int>(graph.edges.size());
         ++edge_index) {
        const GraphEdge& edge = graph.edges[edge_index];
        if (edge.a == source_seed_node || edge.b == source_seed_node) {
            source_edges[edge_index] = 1;
        }
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
            return seed_node_capacity[target];
        }

        Dinic flow(node_count);
        constexpr int kSuperSource = 0;
        for (int node = 1; node < node_count; ++node) {
            if (seeded_nodes[node] != 0) {
                flow.add_edge(kSuperSource, node,
                              std::max(0.0, seed_node_capacity[node]));
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
            const double capacity = std::max(0.0f, edge.capacity);
            flow.add_edge(edge.a, edge.b, capacity);
            flow.add_edge(edge.b, edge.a, capacity);
        }
        return flow.max_flow(kSuperSource, target);
    };

    std::vector<double> node_flow(static_cast<std::size_t>(node_count), 0.0);
    std::vector<float> edge_flow(graph.edges.size(), 0.0f);
    {
        const TimingMark timing = start_timing();
        cv::parallel_for_(cv::Range(1, node_count), [&](const cv::Range& range) {
            for (int node = range.start; node < range.end; ++node) {
                node_flow[node] = max_flow_to_node(node, -1);
            }
        });
        cv::parallel_for_(
            cv::Range(0, static_cast<int>(graph.edges.size())),
            [&](const cv::Range& range) {
                for (int edge_index = range.start; edge_index < range.end;
                     ++edge_index) {
                    const GraphEdge& edge = graph.edges[edge_index];
                    double value = 0.0;
                    if (edge.a > 0 && edge.a < node_count) {
                        value = std::max(value, node_flow[edge.a]);
                    }
                    if (edge.b > 0 && edge.b < node_count) {
                        value = std::max(value, node_flow[edge.b]);
                    }
                    edge_flow[edge_index] =
                        value >= kGraphFlowInf * 0.5
                            ? kDenseFlowInf
                            : static_cast<float>(std::max(0.0, value));
                }
            }
        );
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

            std::vector<float> cap_a(ordered.size(), 0.0f);
            std::vector<float> cap_b(ordered.size(), 0.0f);
            std::vector<double> dist_a(ordered.size(), 0.0);
            std::vector<double> dist_b(ordered.size(), 0.0);
            const float node_a_flow =
                edge.a > 0 && edge.a < node_count
                    ? static_cast<float>(std::min(
                          static_cast<double>(kDenseFlowInf),
                          std::max(0.0, node_flow[edge.a])))
                    : 0.0f;
            const float node_b_flow =
                edge.b > 0 && edge.b < node_count
                    ? static_cast<float>(std::min(
                          static_cast<double>(kDenseFlowInf),
                          std::max(0.0, node_flow[edge.b])))
                    : node_a_flow;

            float min_capacity = node_a_flow;
            for (std::size_t i = 0; i < ordered.size(); ++i) {
                if (i > 0) {
                    const int dx = std::abs(ordered[i].x - ordered[i - 1].x);
                    const int dy = std::abs(ordered[i].y - ordered[i - 1].y);
                    dist_a[i] = dist_a[i - 1] +
                                (dx + dy == 2 ? std::sqrt(2.0) : 1.0);
                }
                min_capacity = std::min(
                    min_capacity,
                    capacity_from_dt(dt.at<float>(ordered[i].y, ordered[i].x)));
                cap_a[i] = min_capacity;
            }
            min_capacity = node_b_flow;
            for (std::size_t i = ordered.size(); i-- > 0;) {
                if (i + 1 < ordered.size()) {
                    const int dx = std::abs(ordered[i].x - ordered[i + 1].x);
                    const int dy = std::abs(ordered[i].y - ordered[i + 1].y);
                    dist_b[i] = dist_b[i + 1] +
                                (dx + dy == 2 ? std::sqrt(2.0) : 1.0);
                }
                min_capacity = std::min(
                    min_capacity,
                    capacity_from_dt(dt.at<float>(ordered[i].y, ordered[i].x)));
                cap_b[i] = min_capacity;
            }

            double edge_sum = 0.0;
            for (std::size_t i = 0; i < ordered.size(); ++i) {
                const double total_dist = dist_a[i] + dist_b[i];
                const double weight_a =
                    total_dist > 0.0 ? dist_b[i] / total_dist : 0.5;
                const double weight_b =
                    total_dist > 0.0 ? dist_a[i] / total_dist : 0.5;
                const float value = static_cast<float>(std::min(
                    static_cast<double>(kDenseFlowInf),
                    weight_a * cap_a[i] + weight_b * cap_b[i]));
                const cv::Point pixel = ordered[i];
                graph_pixel_flow.at<float>(pixel.y, pixel.x) =
                    std::max(graph_pixel_flow.at<float>(pixel.y, pixel.x),
                             value);
                edge_sum += value;
            }
            if (!ordered.empty()) {
                edge_flow[edge_index] = static_cast<float>(
                    std::min(static_cast<double>(kDenseFlowInf),
                             edge_sum / static_cast<double>(ordered.size())));
            }
        }
        timings.push_back(finish_timing("graph_edge_point_flow", timing));
    }

    cv::Mat tree_pixel_flow(white_domain.size(), CV_32F, cv::Scalar(0));
    cv::Mat tree_parent(white_domain.size(), CV_32SC2, cv::Scalar(-1, -1));
    cv::Mat source_edge_mask(white_domain.size(), CV_8U, cv::Scalar(0));
    cv::Mat graph_node_flow(white_domain.size(), CV_32F, cv::Scalar(0));
    cv::Mat graph_flow_dilated;
    {
        const TimingMark timing = start_timing();
        for (int edge_index = 0;
             edge_index < static_cast<int>(graph.edges.size()); ++edge_index) {
            if (source_edges[edge_index] == 0) {
                continue;
            }
            for (const cv::Point pixel : graph.edges[edge_index].pixels) {
                if (pixel.x >= 0 && pixel.x < source_edge_mask.cols &&
                    pixel.y >= 0 && pixel.y < source_edge_mask.rows) {
                    source_edge_mask.at<std::uint8_t>(pixel.y, pixel.x) = 255;
                }
            }
        }
        if (source_seed_node > 0 && source_seed_node < node_count) {
            const cv::Point seed_pixel(cvRound(graph.nodes[source_seed_node].x),
                                       cvRound(graph.nodes[source_seed_node].y));
            if (seed_pixel.x >= 0 && seed_pixel.x < source_edge_mask.cols &&
                seed_pixel.y >= 0 && seed_pixel.y < source_edge_mask.rows) {
                cv::circle(source_edge_mask, seed_pixel, 2, cv::Scalar(255),
                           cv::FILLED, cv::LINE_8);
            }
        }
        timings.push_back(finish_timing("source_edge_mask", timing));
    }

    // Disabled reference path: this Voronoi-tree densification produced the
    // older tree/NN debug images, but current pred_tree_dense_flow comes from
    // compute_dense_backtrack_flow below. Keep this code for future expansion.
    if (kEnableLegacyVoronoiTreeDensification) {
        const TimingMark timing = start_timing();
        cv::Mat tree_mask = cv::Mat::zeros(white_domain.size(), CV_8U);
        for (int y = 0; y < white_domain.rows; ++y) {
            for (int x = 0; x < white_domain.cols; ++x) {
                if (white_domain.at<std::uint8_t>(y, x) != 0 &&
                    source_pixel_ridges.at<std::uint8_t>(y, x) != 0) {
                    tree_mask.at<std::uint8_t>(y, x) = 255;
                }
            }
        }

        cv::Mat graph_flow_mask = graph_pixel_flow > 0.0f;
        cv::dilate(graph_flow_mask, graph_flow_dilated, cv::Mat());
        struct QueueItem {
            float flow = 0.0f;
            cv::Point pixel;
            bool operator<(const QueueItem& other) const {
                return flow < other.flow;
            }
        };
        std::priority_queue<QueueItem> queue;
        constexpr float kFlowEpsilon = 1.0e-4f;
        for (int y = 0; y < tree_mask.rows; ++y) {
            for (int x = 0; x < tree_mask.cols; ++x) {
                if (tree_mask.at<std::uint8_t>(y, x) == 0 ||
                    graph_flow_dilated.at<std::uint8_t>(y, x) == 0) {
                    continue;
                }
                float best_flow = 0.0f;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = x + dx;
                        const int ny = y + dy;
                        if (nx < 0 || nx >= graph_pixel_flow.cols || ny < 0 ||
                            ny >= graph_pixel_flow.rows) {
                            continue;
                        }
                        best_flow =
                            std::max(best_flow,
                                     graph_pixel_flow.at<float>(ny, nx));
                    }
                }
                if (best_flow <= 0.0f) {
                    continue;
                }
                const float rooted_flow =
                    std::min(best_flow, capacity_from_dt(dt.at<float>(y, x)));
                if (rooted_flow > tree_pixel_flow.at<float>(y, x) +
                                      kFlowEpsilon) {
                    tree_pixel_flow.at<float>(y, x) = rooted_flow;
                    tree_parent.at<cv::Vec2i>(y, x) = cv::Vec2i(x, y);
                    queue.push({rooted_flow, cv::Point(x, y)});
                }
            }
        }
        timings.push_back(finish_timing("voronoi_tree_seed", timing));

        const TimingMark propagate_timing = start_timing();
        const std::array<cv::Point, 8> kDirs = {
            cv::Point(-1, -1), cv::Point(0, -1), cv::Point(1, -1),
            cv::Point(-1, 0),                    cv::Point(1, 0),
            cv::Point(-1, 1),  cv::Point(0, 1),  cv::Point(1, 1)};
        while (!queue.empty()) {
            const QueueItem item = queue.top();
            queue.pop();
            if (item.flow + kFlowEpsilon <
                tree_pixel_flow.at<float>(item.pixel.y, item.pixel.x)) {
                continue;
            }
            for (const cv::Point dir : kDirs) {
                const cv::Point next = item.pixel + dir;
                if (next.x < 0 || next.x >= tree_mask.cols || next.y < 0 ||
                    next.y >= tree_mask.rows ||
                    tree_mask.at<std::uint8_t>(next.y, next.x) == 0) {
                    continue;
                }
                const float candidate =
                    std::min(item.flow,
                             capacity_from_dt(dt.at<float>(next.y, next.x)));
                if (candidate <=
                    tree_pixel_flow.at<float>(next.y, next.x) +
                        kFlowEpsilon) {
                    continue;
                }
                tree_pixel_flow.at<float>(next.y, next.x) = candidate;
                tree_parent.at<cv::Vec2i>(next.y, next.x) =
                    cv::Vec2i(item.pixel.x, item.pixel.y);
                queue.push({candidate, next});
            }
        }
        timings.push_back(
            finish_timing("voronoi_tree_propagate", propagate_timing));
    }
    if (kEnableLegacyVoronoiTreeDensification) {
        const TimingMark timing = start_timing();
        constexpr int kTreeFlowSmoothIterations = 2;
        for (int iteration = 0; iteration < kTreeFlowSmoothIterations;
             ++iteration) {
            cv::Mat smoothed = tree_pixel_flow.clone();
            for (int y = 0; y < tree_pixel_flow.rows; ++y) {
                for (int x = 0; x < tree_pixel_flow.cols; ++x) {
                    if (tree_pixel_flow.at<float>(y, x) <= 0.0f) {
                        continue;
                    }
                    double sum = 0.0;
                    int count = 0;
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int nx = x + dx;
                            const int ny = y + dy;
                            if (nx < 0 || nx >= tree_pixel_flow.cols ||
                                ny < 0 || ny >= tree_pixel_flow.rows) {
                                continue;
                            }
                            const float value =
                                tree_pixel_flow.at<float>(ny, nx);
                            if (value <= 0.0f) {
                                continue;
                            }
                            sum += value;
                            ++count;
                        }
                    }
                    if (count > 0) {
                        smoothed.at<float>(y, x) =
                            static_cast<float>(sum / count);
                    }
                }
            }
            tree_pixel_flow = smoothed;
        }
        timings.push_back(finish_timing("voronoi_tree_flow_smooth", timing));
    }

    {
        const TimingMark timing = start_timing();
        for (int node = 1; node < node_count; ++node) {
            const cv::Point anchor(cvRound(graph.nodes[node].x),
                                   cvRound(graph.nodes[node].y));
            if (anchor.x < 0 || anchor.x >= graph_node_flow.cols ||
                anchor.y < 0 || anchor.y >= graph_node_flow.rows ||
                white_domain.at<std::uint8_t>(anchor.y, anchor.x) == 0) {
                continue;
            }
            float value = 0.0f;
            if (node_flow[node] >= kGraphFlowInf * 0.5) {
                value = capacity_from_dt(dt.at<float>(anchor.y, anchor.x));
            } else {
                value = static_cast<float>(std::min(
                    static_cast<double>(kDenseFlowInf),
                    std::max(0.0, node_flow[node])));
            }
            if (value <= 0.0f) {
                continue;
            }
            cv::circle(graph_node_flow, anchor, 2, cv::Scalar(value),
                       cv::FILLED, cv::LINE_8);
        }
        timings.push_back(finish_timing("graph_node_flow_seed", timing));
    }

    cv::Mat tree_dense_nn_flow(white_domain.size(), CV_32F, cv::Scalar(0));
    cv::Mat dense_backtrack_nn_flow(white_domain.size(), CV_32F, cv::Scalar(0));
    cv::Mat smooth_grid_flow(white_domain.size(), CV_32F, cv::Scalar(0));
    cv::Mat tree_dense_flow(white_domain.size(), CV_32F, cv::Scalar(0));
    cv::Mat tree_path_debug(white_domain.size(), CV_32FC3,
                            cv::Scalar(0, 0, 0));
    cv::Mat carrier_debug(white_domain.size(), CV_32FC3,
                          cv::Scalar(0, 0, 0));
    if (kEnableLegacyVoronoiTreeDensification) {
        const TimingMark timing = start_timing();
        cv::Mat tree_source_mask(white_domain.size(), CV_8U, cv::Scalar(255));
        for (int y = 0; y < tree_pixel_flow.rows; ++y) {
            for (int x = 0; x < tree_pixel_flow.cols; ++x) {
                if (tree_pixel_flow.at<float>(y, x) > 0.0f) {
                    tree_source_mask.at<std::uint8_t>(y, x) = 0;
                }
            }
        }

        if (cv::countNonZero(tree_source_mask == 0) > 0) {
            cv::Mat nearest_tree_distance;
            cv::Mat nearest_tree_label;
            cv::distanceTransform(tree_source_mask, nearest_tree_distance,
                                  nearest_tree_label, cv::DIST_L2,
                                  cv::DIST_MASK_5, cv::DIST_LABEL_PIXEL);
            const std::vector<cv::Point> tree_source_points =
                label_source_points(tree_source_mask, nearest_tree_label);
            timings.push_back(finish_timing("voronoi_tree_nearest", timing));

            const TimingMark dense_timing = start_timing();
            constexpr float kTreeRadius = 300.0f;
            constexpr float kFlowEpsilon = 1.0e-4f;
            const auto step_distance = [](const cv::Point a,
                                          const cv::Point b) {
                const int dx = std::abs(a.x - b.x);
                const int dy = std::abs(a.y - b.y);
                return dx + dy == 2 ? static_cast<float>(std::sqrt(2.0))
                                    : 1.0f;
            };
            const std::array<cv::Point, 8> kTreeDirs = {
                cv::Point(-1, -1), cv::Point(0, -1), cv::Point(1, -1),
                cv::Point(-1, 0),                    cv::Point(1, 0),
                cv::Point(-1, 1),  cv::Point(0, 1),  cv::Point(1, 1)};
            cv::Mat tree_index(tree_pixel_flow.size(), CV_32S,
                               cv::Scalar(-1));
            std::vector<cv::Point> tree_pixels;
            tree_pixels.reserve(static_cast<std::size_t>(
                cv::countNonZero(tree_pixel_flow > 0.0f)));
            for (int ty = 0; ty < tree_pixel_flow.rows; ++ty) {
                for (int tx = 0; tx < tree_pixel_flow.cols; ++tx) {
                    if (tree_pixel_flow.at<float>(ty, tx) <= 0.0f) {
                        continue;
                    }
                    tree_index.at<int>(ty, tx) =
                        static_cast<int>(tree_pixels.size());
                    tree_pixels.emplace_back(tx, ty);
                }
            }
            const auto tree_neighbor_indices = [&](const int index) {
                std::vector<int> neighbors;
                neighbors.reserve(8);
                const cv::Point pixel = tree_pixels[index];
                for (const cv::Point dir : kTreeDirs) {
                    const cv::Point next = pixel + dir;
                    if (next.x < 0 || next.x >= tree_pixel_flow.cols ||
                        next.y < 0 || next.y >= tree_pixel_flow.rows) {
                        continue;
                    }
                    const int next_index =
                        tree_index.at<int>(next.y, next.x);
                    if (next_index >= 0) {
                        neighbors.push_back(next_index);
                    }
                }
                return neighbors;
            };
            std::vector<int> next_tree(tree_pixels.size(), -1);
            std::vector<float> next_tree_distance(tree_pixels.size(), 0.0f);
            {
                const TimingMark root_timing = start_timing();
                struct RootQueueItem {
                    float flow = 0.0f;
                    float distance = 0.0f;
                    int index = -1;
                    bool operator<(const RootQueueItem& other) const {
                        if (flow != other.flow) {
                            return flow < other.flow;
                        }
                        return distance > other.distance;
                    }
                };
                std::vector<float> best_route_flow(
                    tree_pixels.size(),
                    -std::numeric_limits<float>::infinity());
                std::vector<float> best_route_distance(
                    tree_pixels.size(),
                    std::numeric_limits<float>::infinity());
                std::priority_queue<RootQueueItem> root_queue;
                for (int index = 0; index < static_cast<int>(tree_pixels.size());
                     ++index) {
                    const cv::Point pixel = tree_pixels[index];
                    if (source_edge_mask.at<std::uint8_t>(pixel.y,
                                                          pixel.x) == 0) {
                        continue;
                    }
                    const float flow =
                        tree_pixel_flow.at<float>(pixel.y, pixel.x);
                    best_route_flow[index] = flow;
                    best_route_distance[index] = 0.0f;
                    root_queue.push({flow, 0.0f, index});
                }

                while (!root_queue.empty()) {
                    const RootQueueItem item = root_queue.top();
                    root_queue.pop();
                    if (item.flow + kFlowEpsilon <
                            best_route_flow[item.index] ||
                        (std::abs(item.flow - best_route_flow[item.index]) <=
                             kFlowEpsilon &&
                         item.distance >
                             best_route_distance[item.index] +
                                 kFlowEpsilon)) {
                        continue;
                    }

                    const cv::Point pixel = tree_pixels[item.index];
                    const float current_flow =
                        tree_pixel_flow.at<float>(pixel.y, pixel.x);
                    for (const int next_index :
                         tree_neighbor_indices(item.index)) {
                        const cv::Point next = tree_pixels[next_index];
                        const float next_flow =
                            tree_pixel_flow.at<float>(next.y, next.x);
                        constexpr float kFlowTolerance = 12.0f;
                        if (next_flow > current_flow + kFlowTolerance) {
                            continue;
                        }
                        const float candidate_flow =
                            std::min(item.flow, next_flow);
                        const float candidate_distance =
                            item.distance + step_distance(pixel, next);
                        const bool better_flow =
                            candidate_flow >
                            best_route_flow[next_index] + kFlowEpsilon;
                        const bool same_flow_shorter =
                            std::abs(candidate_flow -
                                     best_route_flow[next_index]) <=
                                kFlowEpsilon &&
                            candidate_distance + kFlowEpsilon <
                                best_route_distance[next_index];
                        if (!better_flow && !same_flow_shorter) {
                            continue;
                        }
                        best_route_flow[next_index] = candidate_flow;
                        best_route_distance[next_index] = candidate_distance;
                        next_tree[next_index] = item.index;
                        next_tree_distance[next_index] =
                            step_distance(next, pixel);
                        root_queue.push({candidate_flow, candidate_distance,
                                         next_index});
                    }
                }
                int untouched_tree_pixels = 0;
                int routed_tree_pixels = 0;
                int route_roots = 0;
                int reached_without_parent = 0;
                for (int index = 0; index < static_cast<int>(tree_pixels.size());
                     ++index) {
                    if (best_route_flow[index] ==
                        -std::numeric_limits<float>::infinity()) {
                        ++untouched_tree_pixels;
                        continue;
                    }
                    ++routed_tree_pixels;
                    if (best_route_distance[index] <= kFlowEpsilon) {
                        ++route_roots;
                    } else if (next_tree[index] < 0) {
                        ++reached_without_parent;
                    }
                }
                std::cout << "Voronoi tree route:\n"
                          << "  tree_pixels: " << tree_pixels.size() << "\n"
                          << "  routed_tree_pixels: " << routed_tree_pixels
                          << "\n"
                          << "  route_roots: " << route_roots << "\n"
                          << "  untouched_tree_pixels: "
                          << untouched_tree_pixels << "\n"
                          << "  reached_without_parent: "
                          << reached_without_parent << "\n";
                timings.push_back(
                    finish_timing("voronoi_tree_flow_route", root_timing));
            }

            int jump_levels = 1;
            while ((1 << jump_levels) <= static_cast<int>(kTreeRadius) + 2) {
                ++jump_levels;
            }
            std::vector<std::vector<int>> jump_to(
                jump_levels, std::vector<int>(tree_pixels.size(), -1));
            std::vector<std::vector<float>> jump_dist(
                jump_levels, std::vector<float>(tree_pixels.size(), 0.0f));
            for (int index = 0; index < static_cast<int>(tree_pixels.size());
                 ++index) {
                jump_to[0][index] = next_tree[index];
                jump_dist[0][index] = next_tree_distance[index];
            }
            for (int level = 1; level < jump_levels; ++level) {
                for (int index = 0;
                     index < static_cast<int>(tree_pixels.size()); ++index) {
                    const int mid = jump_to[level - 1][index];
                    if (mid < 0) {
                        continue;
                    }
                    const int end = jump_to[level - 1][mid];
                    if (end < 0) {
                        continue;
                    }
                    jump_to[level][index] = end;
                    jump_dist[level][index] =
                        jump_dist[level - 1][index] +
                        jump_dist[level - 1][mid];
                }
            }
            {
                const TimingMark debug_timing = start_timing();
                const std::array<cv::Point, 2> kDebugPoints = {
                    cv::Point(849, 816), cv::Point(906, 869)};
                const std::array<cv::Scalar, 2> kBrightColors = {
                    cv::Scalar(255.0, 0.0, 0.0),
                    cv::Scalar(0.0, 255.0, 0.0)};
                const auto dark_color = [](const cv::Scalar color) {
                    return cv::Scalar(color[0] * 0.35, color[1] * 0.35,
                                      color[2] * 0.35);
                };
                for (std::size_t point_index = 0;
                     point_index < kDebugPoints.size(); ++point_index) {
                    const cv::Point query = kDebugPoints[point_index];
                    std::cout << "Tree path debug point " << point_index
                              << " query=(" << query.x << "," << query.y
                              << ")\n";
                    if (query.x < 0 || query.x >= white_domain.cols ||
                        query.y < 0 || query.y >= white_domain.rows ||
                        white_domain.at<std::uint8_t>(query.y, query.x) == 0) {
                        std::cout << "  skipped: outside image or not in "
                                     "white domain\n";
                        continue;
                    }
                    const int label =
                        nearest_tree_label.at<int>(query.y, query.x);
                    if (label <= 0 ||
                        label >=
                            static_cast<int>(tree_source_points.size())) {
                        std::cout << "  skipped: invalid nearest label "
                                  << label << "\n";
                        continue;
                    }
                    const cv::Point tree_pixel = tree_source_points[label];
                    if (tree_pixel.x < 0) {
                        std::cout << "  skipped: invalid tree source point\n";
                        continue;
                    }
                    const float nearest_distance =
                        nearest_tree_distance.at<float>(query.y, query.x);
                    const float source_flow =
                        tree_pixel_flow.at<float>(tree_pixel.y,
                                                  tree_pixel.x);
                    std::cout << "  nearest_label=" << label
                              << " tree=(" << tree_pixel.x << ","
                              << tree_pixel.y << ")"
                              << " nn_dist=" << nearest_distance
                              << " tree_flow=" << source_flow << "\n";
                    const cv::Scalar backtrack_color =
                        kBrightColors[point_index];
                    const cv::Scalar nn_color = dark_color(backtrack_color);
                    cv::line(tree_path_debug, query, tree_pixel, nn_color, 1,
                             cv::LINE_8);

                    float remaining =
                        std::max(0.0f, kTreeRadius -
                                           nearest_tree_distance.at<float>(
                                               query.y, query.x));
                    int current_index =
                        tree_index.at<int>(tree_pixel.y, tree_pixel.x);
                    if (current_index < 0) {
                        std::cout << "  skipped: nearest tree pixel has no "
                                     "tree_index\n";
                        continue;
                    }
                    std::cout << "  start_index=" << current_index
                              << " remaining=" << remaining
                              << " next_tree=" << next_tree[current_index]
                              << "\n";
                    int debug_steps = 0;
                    float consumed = 0.0f;
                    while (remaining > kFlowEpsilon &&
                           next_tree[current_index] >= 0) {
                        const int next_index = next_tree[current_index];
                        const cv::Point current = tree_pixels[current_index];
                        const cv::Point next = tree_pixels[next_index];
                        const float step = step_distance(current, next);
                        if (debug_steps < 40) {
                            std::cout
                                << "    step " << debug_steps
                                << ": current=(" << current.x << ","
                                << current.y << ")"
                                << " flow="
                                << tree_pixel_flow.at<float>(current.y,
                                                             current.x)
                                << " next=(" << next.x << "," << next.y
                                << ")"
                                << " next_flow="
                                << tree_pixel_flow.at<float>(next.y, next.x)
                                << " step_len=" << step
                                << " remaining=" << remaining
                                << " next_next=" << next_tree[next_index]
                                << "\n";
                        }
                        cv::line(tree_path_debug, current, next,
                                 backtrack_color, 1, cv::LINE_8);
                        if (step >= remaining - kFlowEpsilon) {
                            consumed += remaining;
                            break;
                        }
                        remaining -= step;
                        consumed += step;
                        current_index = next_index;
                        ++debug_steps;
                    }
                    std::cout << "  finished: steps=" << debug_steps
                              << " consumed=" << consumed
                              << " remaining=" << remaining
                              << " final_index=" << current_index
                              << " final_next="
                              << (current_index >= 0
                                      ? next_tree[current_index]
                                      : -1)
                              << "\n";
                }
                timings.push_back(
                    finish_timing("tree_path_debug_render", debug_timing));
            }
            cv::parallel_for_(cv::Range(0, white_domain.rows),
                              [&](const cv::Range& range) {
                for (int y = range.start; y < range.end; ++y) {
                    for (int x = 0; x < white_domain.cols; ++x) {
                    if (white_domain.at<std::uint8_t>(y, x) == 0) {
                        continue;
                    }
                    const int label = nearest_tree_label.at<int>(y, x);
                    if (label <= 0 ||
                        label >= static_cast<int>(tree_source_points.size())) {
                        continue;
                    }
                    const cv::Point tree_pixel = tree_source_points[label];
                    if (tree_pixel.x < 0) {
                        continue;
                    }
                    const float tree_flow =
                        tree_pixel_flow.at<float>(tree_pixel.y,
                                                    tree_pixel.x);
                    if (tree_flow <= 0.0f) {
                        continue;
                    }
                    tree_dense_nn_flow.at<float>(y, x) = tree_flow;

                    float remaining =
                        std::max(0.0f, kTreeRadius -
                                           nearest_tree_distance.at<float>(y, x));
                    int current_index =
                        tree_index.at<int>(tree_pixel.y, tree_pixel.x);
                    if (current_index < 0) {
                        continue;
                    }
                    float endpoint_flow = tree_flow;
                    for (int level = jump_levels - 1; level >= 0; --level) {
                        if (current_index < 0 ||
                            jump_to[level][current_index] < 0 ||
                            jump_dist[level][current_index] <= 0.0f ||
                            jump_dist[level][current_index] >
                                remaining + kFlowEpsilon) {
                            continue;
                        }
                        remaining -= jump_dist[level][current_index];
                        current_index = jump_to[level][current_index];
                        const cv::Point current = tree_pixels[current_index];
                        endpoint_flow =
                            tree_pixel_flow.at<float>(current.y, current.x);
                    }
                    if (remaining > kFlowEpsilon && current_index >= 0 &&
                        next_tree[current_index] >= 0) {
                        const cv::Point current = tree_pixels[current_index];
                        const cv::Point next =
                            tree_pixels[next_tree[current_index]];
                        const float step = step_distance(current, next);
                        const float segment = std::min(step, remaining);
                        const float next_flow =
                            tree_pixel_flow.at<float>(next.y, next.x);
                        const float t = step > 0.0f ? segment / step : 0.0f;
                        endpoint_flow =
                            endpoint_flow * (1.0f - t) + next_flow * t;
                    }

                    tree_dense_flow.at<float>(y, x) = endpoint_flow;
                }
                }
            });
            timings.push_back(finish_timing("tree_dense_flow", dense_timing));
        } else {
            timings.push_back(finish_timing("voronoi_tree_nearest", timing));
            timings.push_back({"tree_dense_flow", 0.0, 0.0});
        }
    }

    {
        const DenseBacktrackResult dense_backtrack =
            compute_dense_backtrack_flow(white_domain, dt, graph_pixel_flow,
                                         graph_node_flow, source_edge_mask,
                                         graph.node_mask, graph, grid_step,
                                         backtrack_distance,
                                         enable_debug_outputs);
        // Reference dense flood: useful for later experiments, but its TIFF is
        // disabled by default so timing focuses on the current tree dense flow.
        dense_backtrack_nn_flow = dense_backtrack.nn_flow;
        smooth_grid_flow = bilinear_from_regular_grid_samples(
            dense_backtrack.smooth_grid_flow, grid_step);
        tree_dense_flow =
            bilinear_from_regular_grid_samples(dense_backtrack.flow, grid_step);
        tree_path_debug = dense_backtrack.debug_paths;
        carrier_debug = dense_backtrack.carrier_debug;
        timings.insert(timings.end(), dense_backtrack.timings.begin(),
                       dense_backtrack.timings.end());
        std::cout << "Dense backtrack:\n"
                  << "  seeded_pixels: " << dense_backtrack.seeded_pixels
                  << "\n"
                  << "  reached_pixels: " << dense_backtrack.reached_pixels
                  << "\n"
                  << "  unreached_white_pixels: "
                  << dense_backtrack.unreached_white_pixels << "\n";
    }

    cv::Mat dense_flow(white_domain.size(), CV_32F, cv::Scalar(0));
    if (kEnableLegacyDenseDtAscent) {
        const TimingMark timing = start_timing();
        dense_flow.setTo(-1.0f, white_domain);
        cv::Mat cached_flow(white_domain.size(), CV_32F, cv::Scalar(-1.0f));
        cv::Mat cached_flow_pixel(white_domain.size(), CV_32SC2,
                                  cv::Scalar(-1, -1));
        for (int y = 0; y < graph_pixel_flow.rows; ++y) {
            for (int x = 0; x < graph_pixel_flow.cols; ++x) {
                const float value = graph_pixel_flow.at<float>(y, x);
                if (value > 0.0f) {
                    cached_flow.at<float>(y, x) = value;
                    cached_flow_pixel.at<cv::Vec2i>(y, x) = cv::Vec2i(x, y);
                    dense_flow.at<float>(y, x) = value;
                }
            }
        }

        const std::array<cv::Point, 8> kDirs = {
            cv::Point(-1, -1), cv::Point(0, -1), cv::Point(1, -1),
            cv::Point(-1, 0),                    cv::Point(1, 0),
            cv::Point(-1, 1),  cv::Point(0, 1),  cv::Point(1, 1)};
        constexpr float kDtStepEps = 1e-4f;
        const auto in_white_domain = [&](const cv::Point pixel) {
            return pixel.x >= 0 && pixel.x < white_domain.cols &&
                   pixel.y >= 0 && pixel.y < white_domain.rows &&
                   white_domain.at<std::uint8_t>(pixel.y, pixel.x) != 0;
        };
        const auto next_ascent_pixel = [&](const cv::Point pixel) {
            const float current_dt = dt.at<float>(pixel.y, pixel.x);
            cv::Point best(-1, -1);
            float best_dt = current_dt;
            for (const cv::Point dir : kDirs) {
                const cv::Point next = pixel + dir;
                if (!in_white_domain(next)) {
                    continue;
                }
                const float next_dt = dt.at<float>(next.y, next.x);
                if (next_dt <= current_dt + kDtStepEps) {
                    continue;
                }
                if (best.x < 0 || next_dt > best_dt + kDtStepEps ||
                    (std::abs(next_dt - best_dt) <= kDtStepEps &&
                     (std::abs(dir.x) + std::abs(dir.y) == 1))) {
                    best = next;
                    best_dt = next_dt;
                }
            }
            return best;
        };
        const auto attenuation_value = [&](float flow_value,
                                           const cv::Point query_pixel) {
            const float query_dt = dt.at<float>(query_pixel.y, query_pixel.x);
            if (query_dt <= kDtStepEps) {
                return flow_value;
            }
            return flow_value / query_dt;
        };
        const auto nearby_graph_flow = [&](const cv::Point pixel, int radius,
                                           cv::Point& flow_pixel) {
            float best = 0.0f;
            flow_pixel = cv::Point(-1, -1);
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const cv::Point next(pixel.x + dx, pixel.y + dy);
                    if (next.x < 0 || next.x >= graph_pixel_flow.cols ||
                        next.y < 0 || next.y >= graph_pixel_flow.rows) {
                        continue;
                    }
                    const float value =
                        graph_pixel_flow.at<float>(next.y, next.x);
                    if (value > best) {
                        best = value;
                        flow_pixel = next;
                    }
                }
            }
            return best;
        };
        const auto nearby_higher_dt_pixel = [&](const cv::Point pixel,
                                                int radius) {
            const float current_dt = dt.at<float>(pixel.y, pixel.x);
            cv::Point best(-1, -1);
            float best_dt = current_dt;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const cv::Point next(pixel.x + dx, pixel.y + dy);
                    if (!in_white_domain(next)) {
                        continue;
                    }
                    const float next_dt = dt.at<float>(next.y, next.x);
                    if (next_dt > best_dt + kDtStepEps) {
                        best = next;
                        best_dt = next_dt;
                    }
                }
            }
            return best;
        };

        std::vector<cv::Point> path;
        path.reserve(256);
        for (int y = 0; y < white_domain.rows; ++y) {
            for (int x = 0; x < white_domain.cols; ++x) {
                if (white_domain.at<std::uint8_t>(y, x) == 0 ||
                    dense_flow.at<float>(y, x) >= 0.0f) {
                    continue;
                }

                path.clear();
                cv::Point current(x, y);
                float result_flow = 0.0f;
                cv::Point result_pixel(-1, -1);
                while (in_white_domain(current)) {
                    const float cached =
                        cached_flow.at<float>(current.y, current.x);
                    if (cached >= 0.0f) {
                        result_flow = cached;
                        const cv::Vec2i cached_pixel =
                            cached_flow_pixel.at<cv::Vec2i>(current.y,
                                                             current.x);
                        result_pixel =
                            cv::Point(cached_pixel[0], cached_pixel[1]);
                        break;
                    }
                    path.push_back(current);
                    const cv::Point next = next_ascent_pixel(current);
                    if (next.x < 0) {
                        result_flow =
                            nearby_graph_flow(current, 2, result_pixel);
                        if (result_flow > 0.0f) {
                            break;
                        }
                        const cv::Point jump = nearby_higher_dt_pixel(current, 2);
                        if (jump.x >= 0) {
                            current = jump;
                            continue;
                        }
                        break;
                    }
                    current = next;
                }

                for (const cv::Point pixel : path) {
                    if (result_flow > 0.0f && result_pixel.x >= 0) {
                        cached_flow.at<float>(pixel.y, pixel.x) = result_flow;
                        cached_flow_pixel.at<cv::Vec2i>(pixel.y, pixel.x) =
                            cv::Vec2i(result_pixel.x, result_pixel.y);
                        dense_flow.at<float>(pixel.y, pixel.x) =
                            attenuation_value(result_flow, pixel);
                    } else {
                        dense_flow.at<float>(pixel.y, pixel.x) =
                            ((pixel.x + pixel.y) & 1) != 0 ? 127.0f : 0.0f;
                    }
                }
            }
        }
        dense_flow.setTo(0.0f, dense_flow < 0.0f);
        timings.push_back(finish_timing("dense_flow_dt_ascent", timing));
    }

    cv::Mat graph_edge_flow(white_domain.size(), CV_32FC3,
                            cv::Scalar(0, 0, 0));
    cv::Mat graph_edge_flow_gray_bg(white_domain.size(), CV_32FC3,
                                    cv::Scalar(127, 127, 127));
    cv::Mat edge_flow_px(white_domain.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat edge_flow_px_gray_bg(white_domain.size(), CV_32FC3,
                                 cv::Scalar(127, 127, 127));
    cv::Mat flow_attn(white_domain.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat voronoi_tree_flow(white_domain.size(), CV_32FC3,
                              cv::Scalar(0, 0, 0));
    cv::Mat voronoi_tree_flow_gray_bg(white_domain.size(), CV_32FC3,
                                      cv::Scalar(127, 127, 127));
    cv::Mat tree_flow_attn(white_domain.size(), CV_32FC3,
                           cv::Scalar(0, 0, 0));
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
    const auto flow_text = [&](double value) {
        if (value >= static_cast<double>(kDenseFlowInf) * 0.5) {
            return std::string("inf");
        }
        return format_scalar_value(value);
    };
    for (int y = 0; y < graph_pixel_flow.rows; ++y) {
        for (int x = 0; x < graph_pixel_flow.cols; ++x) {
            const float value = graph_pixel_flow.at<float>(y, x);
            if (value <= 0.0f) {
                continue;
            }
            edge_flow_px.at<cv::Vec3f>(y, x) =
                cv::Vec3f(value, value, value);
            edge_flow_px_gray_bg.at<cv::Vec3f>(y, x) =
                cv::Vec3f(value, value, value);
            const float local_dt = dt.at<float>(y, x);
            const float attn = local_dt > 1e-4f ? value / local_dt : value;
            flow_attn.at<cv::Vec3f>(y, x) = cv::Vec3f(attn, attn, attn);
        }
    }
    if (kEnableLegacyVoronoiTreeDensification) {
        const TimingMark timing = start_timing();
        int tree_label_stride = 0;
        for (int y = 0; y < tree_pixel_flow.rows; ++y) {
            for (int x = 0; x < tree_pixel_flow.cols; ++x) {
                const float value = tree_pixel_flow.at<float>(y, x);
                if (value <= 0.0f) {
                    continue;
                }
                voronoi_tree_flow.at<cv::Vec3f>(y, x) =
                    cv::Vec3f(value, value, value);
                voronoi_tree_flow_gray_bg.at<cv::Vec3f>(y, x) =
                    cv::Vec3f(value, value, value);
                const float local_dt = dt.at<float>(y, x);
                const float attn = local_dt > 1e-4f ? value / local_dt : value;
                tree_flow_attn.at<cv::Vec3f>(y, x) =
                    cv::Vec3f(attn, attn, attn);
                if ((tree_label_stride++ % 1800) == 0) {
                    const cv::Point anchor(x, y);
                    draw_debug_label(voronoi_tree_flow, anchor,
                                     flow_text(value),
                                     cv::Scalar(255, 255, 255));
                    draw_debug_label(voronoi_tree_flow_gray_bg, anchor,
                                     flow_text(value),
                                     cv::Scalar(255, 255, 255));
                    draw_debug_label(tree_flow_attn, anchor, flow_text(attn),
                                     cv::Scalar(255, 255, 255));
                }
            }
        }
        timings.push_back(finish_timing("tree_flow_render", timing));
    }
    for (int edge_index = 0; edge_index < static_cast<int>(graph.edges.size());
         ++edge_index) {
        const float raw_value = edge_flow[edge_index];
        draw_graph_edge(graph_edge_flow, graph.edges[edge_index],
                        cv::Scalar(raw_value, raw_value, raw_value));
        draw_graph_edge(graph_edge_flow_gray_bg, graph.edges[edge_index],
                        cv::Scalar(raw_value, raw_value, raw_value));
        if (source_edges[edge_index] != 0) {
            draw_graph_edge(graph_source_edges, graph.edges[edge_index],
                            cv::Scalar(255));
        }
    }

    for (int edge_index = 0; edge_index < static_cast<int>(graph.edges.size());
         ++edge_index) {
        const GraphEdge& edge = graph.edges[edge_index];
        const std::vector<cv::Point> ordered = order_edge_pixels(edge, graph);
        if (ordered.empty()) {
            continue;
        }
        const cv::Point anchor = ordered[ordered.size() / 2];
        const std::string edge_text = flow_text(edge_flow[edge_index]);
        const float px_value = graph_pixel_flow.at<float>(anchor.y, anchor.x);
        const std::string px_text = flow_text(px_value);
        const float anchor_dt = dt.at<float>(anchor.y, anchor.x);
        const float attn_value =
            anchor_dt > 1e-4f ? px_value / anchor_dt : px_value;
        const std::string attn_text = flow_text(attn_value);
        draw_debug_label(graph_edge_flow_gray_bg, anchor, edge_text,
                         cv::Scalar(255, 255, 255));
        draw_debug_label(graph_edge_flow, anchor, edge_text,
                         cv::Scalar(255, 255, 255));
        draw_debug_label(edge_flow_px, anchor, px_text,
                         cv::Scalar(255, 255, 255));
        draw_debug_label(edge_flow_px_gray_bg, anchor, px_text,
                         cv::Scalar(255, 255, 255));
        draw_debug_label(flow_attn, anchor, attn_text,
                         cv::Scalar(255, 255, 255));
    }

    for (int node = 1; node < node_count; ++node) {
        const cv::Point anchor(cvRound(graph.nodes[node].x),
                               cvRound(graph.nodes[node].y));
        const std::string text = flow_text(node_flow[node]);
        const float node_dt = dt.at<float>(anchor.y, anchor.x);
        const double node_attn =
            node_dt > 1e-4f ? node_flow[node] / node_dt : node_flow[node];
        const std::string attn_text = flow_text(node_attn);
        draw_debug_label(graph_edge_flow, anchor, text,
                         cv::Scalar(180, 180, 180));
        draw_debug_label(edge_flow_px, anchor, text,
                         cv::Scalar(180, 180, 180));
        draw_debug_label(edge_flow_px_gray_bg, anchor, text,
                         cv::Scalar(180, 180, 180));
        draw_debug_label(flow_attn, anchor, attn_text,
                         cv::Scalar(180, 180, 180));
    }
    std::vector<cv::Point> graph_edge_flow_gray_bg_label_anchors;
    graph_edge_flow_gray_bg_label_anchors.reserve(graph.nodes.size());
    const auto gray_bg_label_is_clear = [&](const cv::Point anchor) {
        constexpr int kMinLabelDistanceSq = 12 * 12;
        for (const cv::Point existing : graph_edge_flow_gray_bg_label_anchors) {
            const int dx = anchor.x - existing.x;
            const int dy = anchor.y - existing.y;
            if (dx * dx + dy * dy < kMinLabelDistanceSq) {
                return false;
            }
        }
        return true;
    };
    for (int node = 1; node < node_count; ++node) {
        const cv::Point anchor(cvRound(graph.nodes[node].x),
                               cvRound(graph.nodes[node].y));
        if (!gray_bg_label_is_clear(anchor)) {
            continue;
        }
        draw_large_debug_label(graph_edge_flow_gray_bg, anchor,
                               flow_text(node_flow[node]),
                               cv::Scalar(180, 180, 180));
        graph_edge_flow_gray_bg_label_anchors.push_back(anchor);
    }

    return {dense_flow,
            voronoi_tree_flow,
            voronoi_tree_flow_gray_bg,
            tree_dense_nn_flow,
            dense_backtrack_nn_flow,
            smooth_grid_flow,
            tree_dense_flow,
            tree_path_debug,
            carrier_debug,
            tree_flow_attn,
            flow_attn,
            graph_edge_flow,
            graph_edge_flow_gray_bg,
            edge_flow_px,
            edge_flow_px_gray_bg,
            graph_source_edges,
            cv::Mat(),
            source_seed_pixel,
            source_seed_capacity,
            source_edge_count,
            seeded_node_count,
            finite_edge_flow_min,
            finite_edge_flow_max,
            finite_edge_flow_count,
            timings};
}

cv::Mat to_float_layer(const cv::Mat& image) {
    CV_Assert(image.channels() == 1 || image.channels() == 3);
    cv::Mat float_image;
    image.convertTo(float_image, CV_MAKETYPE(CV_32F, image.channels()));
    if (float_image.channels() == 3) {
        return float_image;
    }

    cv::Mat bgr;
    cv::cvtColor(float_image, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

std::vector<int> regular_grid_axis(const int limit, const int step) {
    std::vector<int> axis;
    const int safe_step = std::max(1, step);
    for (int value = 0; value < limit; value += safe_step) {
        axis.push_back(value);
    }
    if (axis.empty() || axis.back() != limit - 1) {
        axis.push_back(limit - 1);
    }
    return axis;
}

cv::Mat bilinear_from_regular_grid_samples(const cv::Mat& source,
                                           const int grid_step) {
    CV_Assert(source.type() == CV_32F);
    const int rows = source.rows;
    const int cols = source.cols;
    const std::vector<int> grid_xs = regular_grid_axis(cols, grid_step);
    const std::vector<int> grid_ys = regular_grid_axis(rows, grid_step);
    cv::Mat out(source.size(), CV_32F, cv::Scalar(0));

    const auto lower_axis_index = [](const std::vector<int>& axis,
                                     const int value) {
        const auto upper = std::upper_bound(axis.begin(), axis.end(), value);
        if (upper == axis.begin()) {
            return 0;
        }
        return static_cast<int>(std::distance(axis.begin(), upper)) - 1;
    };

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const int gy0 = lower_axis_index(grid_ys, y);
            const int gy1 = std::min(gy0 + 1,
                                     static_cast<int>(grid_ys.size()) - 1);
            const float y0 = static_cast<float>(grid_ys[gy0]);
            const float y1 = static_cast<float>(grid_ys[gy1]);
            const float ty =
                y1 > y0 ? (static_cast<float>(y) - y0) / (y1 - y0) : 0.0f;
            float* dst = out.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                const int gx0 = lower_axis_index(grid_xs, x);
                const int gx1 = std::min(
                    gx0 + 1, static_cast<int>(grid_xs.size()) - 1);
                const float x0 = static_cast<float>(grid_xs[gx0]);
                const float x1 = static_cast<float>(grid_xs[gx1]);
                const float tx =
                    x1 > x0 ? (static_cast<float>(x) - x0) / (x1 - x0)
                            : 0.0f;
                const float v00 =
                    source.at<float>(grid_ys[gy0], grid_xs[gx0]);
                const float v10 =
                    source.at<float>(grid_ys[gy0], grid_xs[gx1]);
                const float v01 =
                    source.at<float>(grid_ys[gy1], grid_xs[gx0]);
                const float v11 =
                    source.at<float>(grid_ys[gy1], grid_xs[gx1]);
                const float v0 = v00 * (1.0f - tx) + v10 * tx;
                const float v1 = v01 * (1.0f - tx) + v11 * tx;
                dst[x] = v0 * (1.0f - ty) + v1 * ty;
            }
        }
    });

    return out;
}

cv::Mat compute_flow_gate_weight_image(const cv::Mat& flow,
                                       const cv::Mat& dt,
                                       const cv::Point gate_source,
                                       const float gate_source_capacity,
                                       const int grid_step,
                                       const float configured_one,
                                       const float configured_zero) {
    CV_Assert(flow.type() == CV_32F);
    CV_Assert(dt.type() == CV_32F);
    CV_Assert(flow.size() == dt.size());
    const int rows = flow.rows;
    const int cols = flow.cols;
    cv::Mat weight(flow.size(), CV_32F, cv::Scalar(0));
    if (gate_source.x < 0 || gate_source.x >= cols || gate_source.y < 0 ||
        gate_source.y >= rows) {
        return weight;
    }

    const float safe_one = std::max(1.0e-6f, configured_one);
    const float seed_dt = dt.at<float>(gate_source.y, gate_source.x);
    const float seed_capacity =
        gate_source_capacity > 0.0f ? gate_source_capacity
                                    : capacity_from_dt(seed_dt);
    float effective_one = safe_one;
    float effective_zero = configured_zero;
    if (seed_capacity > 0.0f && seed_capacity < safe_one) {
        const float scale = seed_capacity / safe_one;
        effective_one = seed_capacity;
        effective_zero = configured_zero * scale;
    }
    if (effective_one <= effective_zero) {
        effective_zero = std::max(0.0f, effective_one - 1.0f);
    }
    const float denom = std::max(1.0e-6f, effective_one - effective_zero);
    const float seed_radius_px =
        2.0f * static_cast<float>(std::max(1, grid_step));
    const float seed_radius_sq = seed_radius_px * seed_radius_px;

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            float* out_row = weight.ptr<float>(y);
            const float* flow_row = flow.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                float value = (flow_row[x] - effective_zero) / denom;
                value = std::max(0.0f, std::min(1.0f, value));
                const float dx = static_cast<float>(x - gate_source.x);
                const float dy = static_cast<float>(y - gate_source.y);
                if (dx * dx + dy * dy <= seed_radius_sq) {
                    value = 1.0f;
                }
                out_row[x] = value;
            }
        }
    });

    std::cout << "Flow gate weight:\n"
              << "  gate_source: (" << gate_source.x << "," << gate_source.y
              << ")\n"
              << "  gate_source_dt: " << seed_dt << "\n"
              << "  gate_source_capacity: " << seed_capacity << "\n"
              << "  configured_zero: " << configured_zero << "\n"
              << "  configured_one: " << safe_one << "\n"
              << "  effective_zero: " << effective_zero << "\n"
              << "  effective_one: " << effective_one << "\n";
    return weight;
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
        cv::parallel_for_(cv::Range(0, binary.rows),
                          [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                for (int x = 0; x < binary.cols; ++x) {
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
                            const int nx = x + dx;
                            const int ny = y + dy;
                            if (nx < 0 || nx >= binary.cols || ny < 0 ||
                                ny >= binary.rows) {
                                continue;
                            }
                            const int other =
                                nearest_component.at<int>(ny, nx);
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
            boundary_skeleton_pruned, source_pixel_ridges, dt_to_component,
            &timings);
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

cv::Mat add_valid_frame_to_skeleton(const cv::Mat& skeleton, const cv::Mat& dt) {
    CV_Assert(skeleton.type() == CV_8U);
    CV_Assert(dt.type() == CV_32F);
    CV_Assert(skeleton.size() == dt.size());

    cv::Mat out;
    cv::threshold(skeleton, out, 0, 255, cv::THRESH_BINARY);
    const cv::Mat original = out.clone();
    cv::Mat added_frame = cv::Mat::zeros(out.size(), CV_8U);
    const auto on_frame = [&](const cv::Point pixel) {
        return pixel.x == 0 || pixel.y == 0 || pixel.x == out.cols - 1 ||
               pixel.y == out.rows - 1;
    };
    const auto add_if_valid = [&](const cv::Point pixel) {
        if (dt.at<float>(pixel.y, pixel.x) > 0.0f) {
            if (original.at<std::uint8_t>(pixel.y, pixel.x) == 0) {
                added_frame.at<std::uint8_t>(pixel.y, pixel.x) = 255;
            }
            out.at<std::uint8_t>(pixel.y, pixel.x) = 255;
        }
    };

    for (int x = 0; x < out.cols; ++x) {
        add_if_valid(cv::Point(x, 0));
        if (out.rows > 1) {
            add_if_valid(cv::Point(x, out.rows - 1));
        }
    }
    for (int y = 1; y < out.rows - 1; ++y) {
        add_if_valid(cv::Point(0, y));
        if (out.cols > 1) {
            add_if_valid(cv::Point(out.cols - 1, y));
        }
    }

    std::vector<cv::Point> frame_pixels;
    frame_pixels.reserve(static_cast<std::size_t>(2 * out.cols + 2 * out.rows));
    for (int x = 0; x < out.cols; ++x) {
        frame_pixels.push_back(cv::Point(x, 0));
    }
    for (int y = 1; y < out.rows; ++y) {
        frame_pixels.push_back(cv::Point(out.cols - 1, y));
    }
    if (out.rows > 1) {
        for (int x = out.cols - 2; x >= 0; --x) {
            frame_pixels.push_back(cv::Point(x, out.rows - 1));
        }
    }
    if (out.cols > 1) {
        for (int y = out.rows - 2; y >= 1; --y) {
            frame_pixels.push_back(cv::Point(0, y));
        }
    }

    const auto has_non_frame_neighbor = [&](const cv::Point pixel) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                const int x = pixel.x + dx;
                const int y = pixel.y + dy;
                if (x < 0 || x >= out.cols || y < 0 || y >= out.rows) {
                    continue;
                }
                const cv::Point neighbor(x, y);
                if (!on_frame(neighbor) &&
                    out.at<std::uint8_t>(y, x) != 0) {
                    return true;
                }
            }
        }
        return false;
    };

    const int frame_count = static_cast<int>(frame_pixels.size());
    std::vector<char> removed(static_cast<std::size_t>(frame_count), 0);
    const auto cleanup_from_invalid = [&](int invalid_index, int step) {
        int index = (invalid_index + step + frame_count) % frame_count;
        while (removed[index] == 0) {
            const cv::Point pixel = frame_pixels[index];
            if (added_frame.at<std::uint8_t>(pixel.y, pixel.x) == 0) {
                break;
            }
            if (has_non_frame_neighbor(pixel)) {
                break;
            }
            out.at<std::uint8_t>(pixel.y, pixel.x) = 0;
            added_frame.at<std::uint8_t>(pixel.y, pixel.x) = 0;
            removed[index] = 1;
            index = (index + step + frame_count) % frame_count;
        }
    };

    for (int i = 0; i < frame_count; ++i) {
        const cv::Point pixel = frame_pixels[i];
        if (dt.at<float>(pixel.y, pixel.x) > 0.0f) {
            continue;
        }
        cleanup_from_invalid(i, 1);
        cleanup_from_invalid(i, -1);
    }
    return out;
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
        CV_Assert(layer.image.depth() == CV_32F);
        CV_Assert(layer.image.channels() == 3);

        cv::Mat image;
        cv::cvtColor(layer.image, image, cv::COLOR_BGR2RGB);
        if (!image.isContinuous()) {
            image = image.clone();
        }

        const int channels = image.channels();
        TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH,
                     static_cast<std::uint32_t>(image.cols));
        TIFFSetField(tiff, TIFFTAG_IMAGELENGTH,
                     static_cast<std::uint32_t>(image.rows));
        TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL,
                     static_cast<std::uint16_t>(channels));
        TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE,
                     static_cast<std::uint16_t>(32));
        TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
        TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
        TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        TIFFSetField(tiff, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
        TIFFSetField(tiff, TIFFTAG_PAGENUMBER,
                     static_cast<std::uint16_t>(layer_index),
                     static_cast<std::uint16_t>(layers.size()));
        TIFFSetField(tiff, TIFFTAG_PAGENAME, layer.name.c_str());

        for (int y = 0; y < image.rows; ++y) {
            const float* row = image.ptr<float>(y);
            void* row_to_write = const_cast<float*>(row);

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

extern "C" int dense_batch_flow_grid_u8(const unsigned char* image,
                                        int width,
                                        int height,
                                        int source_x,
                                        int source_y,
                                        const float* query_xy,
                                        int query_count,
                                        float* query_flow,
                                        float* dense_flow,
                                        float* smooth_grid_flow,
                                        float* graph_edge_flow_rgb,
                                        int grid_step,
                                        float backtrack_distance,
                                        int* resolved_source_x,
                                        int* resolved_source_y,
                                        float* resolved_source_capacity,
                                        char* error_message,
                                        int error_message_size,
                                        int verbose) {
    const auto set_error = [&](const std::string& message) {
        if (error_message != nullptr && error_message_size > 0) {
            const std::size_t n = std::min<std::size_t>(
                static_cast<std::size_t>(error_message_size - 1),
                message.size());
            std::copy(message.begin(), message.begin() + n, error_message);
            error_message[n] = '\0';
        }
    };

    try {
        if (image == nullptr) {
            throw std::runtime_error("image pointer is null");
        }
        if (width <= 0 || height <= 0) {
            throw std::runtime_error("image dimensions must be positive");
        }
        if (query_count < 0) {
            throw std::runtime_error("query_count must be non-negative");
        }
        if (query_count > 0 &&
            (query_xy == nullptr || query_flow == nullptr)) {
            throw std::runtime_error(
                "query_xy and query_flow are required when query_count > 0");
        }
        struct CoutSilencer {
            std::streambuf* old = nullptr;
            std::ostringstream sink;
            explicit CoutSilencer(bool active) {
                if (active) {
                    old = std::cout.rdbuf(sink.rdbuf());
                }
            }
            ~CoutSilencer() {
                if (old != nullptr) {
                    std::cout.rdbuf(old);
                }
            }
        } silence(verbose == 0);

        std::vector<StageTiming> timings;
        const TimingMark total_start = start_timing();

        cv::Mat gray(height, width, CV_8U,
                     const_cast<unsigned char*>(image));
        gray = gray.clone();

        cv::Mat binary;
        {
            const TimingMark timing = start_timing();
            binary = binarize_fixed_threshold(gray);
            timings.push_back(finish_timing("binarize", timing));
        }

        const cv::Point source(source_x, source_y);
        {
            const TimingMark timing = start_timing();
            binary = keep_source_white_component(binary, source);
            timings.push_back(
                finish_timing("source_connected_component", timing));
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

        {
            const TimingMark timing = start_timing();
            component_voronoi_result.cell_loops_connected =
                add_valid_frame_to_skeleton(
                    component_voronoi_result.cell_loops_connected, dt);
            timings.push_back(finish_timing("add_valid_frame", timing));
        }

        SkeletonGraph graph;
        {
            const TimingMark timing = start_timing();
            graph = extract_skeleton_graph(
                component_voronoi_result.cell_loops_connected, dt);
            timings.push_back(finish_timing("graph_extract", timing));
        }

        const bool enable_debug_outputs =
            smooth_grid_flow != nullptr || graph_edge_flow_rgb != nullptr;
        DenseFlowResult dense_flow_result =
            compute_dense_source_flow(
                white_domain, dt, graph,
                component_voronoi_result.source_pixel_ridges, source,
                grid_step, backtrack_distance, enable_debug_outputs);
        if (resolved_source_x != nullptr) {
            *resolved_source_x = dense_flow_result.source_seed_pixel.x;
        }
        if (resolved_source_y != nullptr) {
            *resolved_source_y = dense_flow_result.source_seed_pixel.y;
        }
        if (resolved_source_capacity != nullptr) {
            *resolved_source_capacity = dense_flow_result.source_seed_capacity;
        }
        timings.insert(timings.end(), dense_flow_result.timings.begin(),
                       dense_flow_result.timings.end());

        if (dense_flow != nullptr) {
            for (int y = 0; y < height; ++y) {
                const float* src_row =
                    dense_flow_result.tree_dense_flow.ptr<float>(y);
                std::copy(src_row, src_row + width,
                          dense_flow + static_cast<std::size_t>(y) * width);
            }
        }
        if (smooth_grid_flow != nullptr) {
            for (int y = 0; y < height; ++y) {
                const float* src_row =
                    dense_flow_result.smooth_grid_flow.ptr<float>(y);
                std::copy(src_row, src_row + width,
                          smooth_grid_flow +
                              static_cast<std::size_t>(y) * width);
            }
        }
        if (graph_edge_flow_rgb != nullptr) {
            cv::Mat graph_rgb;
            cv::cvtColor(dense_flow_result.graph_edge_flow_gray_bg, graph_rgb,
                         cv::COLOR_BGR2RGB);
            for (int y = 0; y < height; ++y) {
                const cv::Vec3f* src_row = graph_rgb.ptr<cv::Vec3f>(y);
                float* dst_row =
                    graph_edge_flow_rgb +
                    static_cast<std::size_t>(y) * width * 3;
                for (int x = 0; x < width; ++x) {
                    dst_row[3 * x + 0] = src_row[x][0];
                    dst_row[3 * x + 1] = src_row[x][1];
                    dst_row[3 * x + 2] = src_row[x][2];
                }
            }
        }

        const auto sample_flow = [&](const float x, const float y) {
            if (x < 0.0f || y < 0.0f || x > static_cast<float>(width - 1) ||
                y > static_cast<float>(height - 1)) {
                return 0.0f;
            }
            const int x0 = std::clamp(static_cast<int>(std::floor(x)), 0,
                                      width - 1);
            const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0,
                                      height - 1);
            const int x1 = std::min(x0 + 1, width - 1);
            const int y1 = std::min(y0 + 1, height - 1);
            const float fx = x - static_cast<float>(x0);
            const float fy = y - static_cast<float>(y0);
            const float v00 =
                dense_flow_result.tree_dense_flow.at<float>(y0, x0);
            const float v10 =
                dense_flow_result.tree_dense_flow.at<float>(y0, x1);
            const float v01 =
                dense_flow_result.tree_dense_flow.at<float>(y1, x0);
            const float v11 =
                dense_flow_result.tree_dense_flow.at<float>(y1, x1);
            const float v0 = v00 * (1.0f - fx) + v10 * fx;
            const float v1 = v01 * (1.0f - fx) + v11 * fx;
            return v0 * (1.0f - fy) + v1 * fy;
        };

        for (int i = 0; i < query_count; ++i) {
            query_flow[i] = sample_flow(query_xy[2 * i],
                                        query_xy[2 * i + 1]);
        }

        timings.push_back(finish_timing("total", total_start));
        if (verbose != 0) {
            print_stage_timings(timings);
        }
        if (error_message != nullptr && error_message_size > 0) {
            error_message[0] = '\0';
        }
        return 0;
    } catch (const std::exception& ex) {
        set_error(ex.what());
        return 1;
    } catch (...) {
        set_error("unknown dense_batch_flow_grid_u8 error");
        return 1;
    }
}

#ifndef DENSE_BATCH_MIN_CUT_NO_MAIN
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
        cv::Mat white_domain;
        cv::Mat dt;
        ComponentVoronoiResult component_voronoi_result;
        SkeletonGraph graph;
        GraphConnectivityStats graph_stats;
        cv::Mat graph_random_colors;
        cv::Mat graph_edges_random_colors;
        cv::Mat graph_nodes;
        cv::Mat graph_capacity;
        cv::Mat graph_capacity_gray_bg;
        DenseFlowResult dense_flow_result;
        bool has_dense_flow = false;
        cv::Mat contour_loops;
        for (int repeat = 0; repeat < args.compute_repeats; ++repeat) {
            CoutSilencer silence_repeated_details(
                args.compute_repeats > 1 && repeat + 1 < args.compute_repeats);
            {
                const TimingMark timing = start_timing();
                binary = binarize_fixed_threshold(gray);
                timings.push_back(finish_timing("binarize", timing));
            }
            if (args.has_source) {
                const TimingMark timing = start_timing();
                binary = keep_source_white_component(binary, args.source);
                timings.push_back(
                    finish_timing("source_connected_component", timing));
            }

            white_domain = cv::Mat(binary.size(), CV_8U, cv::Scalar(255));
            white_domain.setTo(0, binary);

            {
                const TimingMark timing = start_timing();
                cv::distanceTransform(white_domain, dt, cv::DIST_L2,
                                      cv::DIST_MASK_PRECISE);
                timings.push_back(finish_timing("precise_dt", timing));
            }

            {
                const TimingMark timing = start_timing();
                component_voronoi_result = component_voronoi(binary);
                timings.push_back(finish_timing("component_voronoi", timing));
                timings.insert(timings.end(),
                               component_voronoi_result.timings.begin(),
                               component_voronoi_result.timings.end());
            }

            {
                const TimingMark timing = start_timing();
                component_voronoi_result.cell_loops_connected =
                    add_valid_frame_to_skeleton(
                        component_voronoi_result.cell_loops_connected, dt);
                timings.push_back(finish_timing("add_valid_frame", timing));
            }

            {
                const TimingMark timing = start_timing();
                graph = extract_skeleton_graph(
                    component_voronoi_result.cell_loops_connected, dt);
                timings.push_back(finish_timing("graph_extract", timing));
            }

            {
                const TimingMark timing = start_timing();
                graph_stats = graph_connectivity_stats(graph);
                timings.push_back(
                    finish_timing("graph_connectivity_stats", timing));
            }

            {
                const TimingMark timing = start_timing();
                graph_random_colors =
                    render_graph_random_colors(graph, binary.size());
                timings.push_back(
                    finish_timing("graph_random_color_render", timing));
            }

            {
                const TimingMark timing = start_timing();
                graph_edges_random_colors =
                    render_graph_edges_random_colors(graph, binary.size());
                timings.push_back(
                    finish_timing("graph_edge_only_random_color_render",
                                  timing));
            }

            {
                const TimingMark timing = start_timing();
                graph_nodes = render_graph_nodes(graph, binary.size());
                timings.push_back(finish_timing("graph_node_render", timing));
            }

            {
                const TimingMark timing = start_timing();
                graph_capacity =
                    render_graph_capacity(graph, binary.size(), 0);
                graph_capacity_gray_bg =
                    render_graph_capacity(graph, binary.size(), 127);
                timings.push_back(
                    finish_timing("graph_capacity_render", timing));
            }

            has_dense_flow = false;
            if (args.has_source) {
                dense_flow_result =
                    compute_dense_source_flow(
                        white_domain, dt, graph,
                        component_voronoi_result.source_pixel_ridges,
                        args.source, args.grid_step,
                        args.backtrack_distance, args.write_outputs);
                {
                    const TimingMark timing = start_timing();
                    dense_flow_result.flow_gate_weight =
                        compute_flow_gate_weight_image(
                            dense_flow_result.tree_dense_flow, dt,
                            dense_flow_result.source_seed_pixel,
                            dense_flow_result.source_seed_capacity,
                            args.grid_step, args.flow_weight_one,
                            args.flow_weight_zero);
                    timings.push_back(
                        finish_timing("flow_gate_weight", timing));
                }
                timings.insert(timings.end(), dense_flow_result.timings.begin(),
                               dense_flow_result.timings.end());
                has_dense_flow = true;
            }

            {
                const TimingMark timing = start_timing();
                contour_loops = binary_contour_loops(binary);
                timings.push_back(finish_timing("binary_contour_loops",
                                                timing));
            }
        }

        const std::string stem = args.input.stem().string();
        if (args.write_outputs) {
        {
            const TimingMark timing = start_timing();
            write_image(workdir / (stem + "_binary.tif"), binary);
            write_image(workdir / (stem + "_dt.tif"), dt);
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
            write_image(workdir / (stem + "_graph_capacity_gray_bg.tif"),
                        graph_capacity_gray_bg);
            write_graph_connectivity_report(
                workdir / (stem + "_graph_components.txt"), graph_stats);
            timings.push_back(finish_timing("write_regular_outputs", timing));
        }

        if (has_dense_flow) {
            const TimingMark timing = start_timing();
            if (kEnableLegacyDenseDtAscent) {
                write_image(workdir / (stem + "_dense_flow.tif"),
                            dense_flow_result.dense_flow);
            }
            if (kEnableLegacyVoronoiTreeDensification) {
                write_image(workdir / (stem + "_voronoi_tree_flow.tif"),
                            dense_flow_result.voronoi_tree_flow);
                write_image(workdir /
                                (stem + "_voronoi_tree_flow_gray_bg.tif"),
                            dense_flow_result.voronoi_tree_flow_gray_bg);
                write_image(workdir / (stem + "_tree_dense_nn_flow.tif"),
                            dense_flow_result.tree_dense_nn_flow);
                write_image(workdir / (stem + "_tree_flow_attn.tif"),
                            dense_flow_result.tree_flow_attn);
            }
            if (kWriteReferenceDenseBacktrackNn) {
                write_image(workdir / (stem + "_dense_backtrack_nn_flow.tif"),
                            dense_flow_result.dense_backtrack_nn_flow);
            }
            write_image(workdir / (stem + "_smooth_grid_flow.tif"),
                        dense_flow_result.smooth_grid_flow);
            write_image(workdir / (stem + "_tree_dense_flow.tif"),
                        dense_flow_result.tree_dense_flow);
            write_image(workdir / (stem + "_tree_paths_debug.tif"),
                        dense_flow_result.tree_path_debug);
            write_image(workdir / (stem + "_carrier_debug.tif"),
                        dense_flow_result.carrier_debug);
            write_image(workdir / (stem + "_graph_edge_flow.tif"),
                        dense_flow_result.graph_edge_flow);
            write_image(workdir / (stem + "_graph_edge_flow_gray_bg.tif"),
                        dense_flow_result.graph_edge_flow_gray_bg);
            write_image(workdir / (stem + "_edge_flow_px.tif"),
                        dense_flow_result.edge_flow_px);
            write_image(workdir / (stem + "_edge_flow_px_gray_bg.tif"),
                        dense_flow_result.edge_flow_px_gray_bg);
            write_image(workdir / (stem + "_graph_source_edges.tif"),
                        dense_flow_result.graph_source_edges);
            write_image(workdir / (stem + "_flow_gate_weight.tif"),
                        dense_flow_result.flow_gate_weight);
            timings.push_back(finish_timing("dense_flow_write", timing));
        }

        std::vector<NamedLayer> layered_tiff = {
            {"binary_threshold", to_float_layer(binary)},
            {"dt", to_float_layer(dt)},
            {"loops",
             to_float_layer(component_voronoi_result.boundary_skeleton_pruned)},
            {"loops_connected",
             to_float_layer(component_voronoi_result.cell_loops_connected)},
            {"graph_random_edges", to_float_layer(graph_random_colors)},
            {"graph_edges_random", to_float_layer(graph_edges_random_colors)},
            {"graph_nodes", to_float_layer(graph_nodes)},
            {"graph_capacity", to_float_layer(graph_capacity)},
            {"graph_capacity_gray_bg", to_float_layer(graph_capacity_gray_bg)},
        };
        if (has_dense_flow) {
            if (kEnableLegacyDenseDtAscent) {
                layered_tiff.push_back(
                    {"dense_flow",
                     to_float_layer(dense_flow_result.dense_flow)});
            }
            if (kEnableLegacyVoronoiTreeDensification) {
                layered_tiff.push_back(
                    {"voronoi_tree_flow",
                     to_float_layer(dense_flow_result.voronoi_tree_flow)});
                layered_tiff.push_back(
                    {"voronoi_tree_flow_gray_bg",
                     to_float_layer(
                         dense_flow_result.voronoi_tree_flow_gray_bg)});
                layered_tiff.push_back(
                    {"tree_dense_nn_flow",
                     to_float_layer(dense_flow_result.tree_dense_nn_flow)});
                layered_tiff.push_back(
                    {"tree_flow_attn",
                     to_float_layer(dense_flow_result.tree_flow_attn)});
            }
            if (kWriteReferenceDenseBacktrackNn) {
                layered_tiff.push_back(
                    {"dense_backtrack_nn_flow",
                     to_float_layer(dense_flow_result.dense_backtrack_nn_flow)});
            }
            layered_tiff.push_back(
                {"smooth_grid_flow",
                 to_float_layer(dense_flow_result.smooth_grid_flow)});
            layered_tiff.push_back(
                {"tree_dense_flow",
                 to_float_layer(dense_flow_result.tree_dense_flow)});
            layered_tiff.push_back(
                {"carrier_debug",
                 to_float_layer(dense_flow_result.carrier_debug)});
            layered_tiff.push_back(
                {"graph_edge_flow",
                 to_float_layer(dense_flow_result.graph_edge_flow)});
            layered_tiff.push_back(
                {"graph_edge_flow_gray_bg",
                 to_float_layer(dense_flow_result.graph_edge_flow_gray_bg)});
            layered_tiff.push_back(
                {"edge_flow_px",
                 to_float_layer(dense_flow_result.edge_flow_px)});
            layered_tiff.push_back(
                {"edge_flow_px_gray_bg",
                 to_float_layer(dense_flow_result.edge_flow_px_gray_bg)});
            layered_tiff.push_back(
                {"graph_source_edges",
                 to_float_layer(dense_flow_result.graph_source_edges)});
            layered_tiff.push_back(
                {"flow_gate_weight",
                 to_float_layer(dense_flow_result.flow_gate_weight)});
        }
        {
            const TimingMark timing = start_timing();
            write_named_layered_tiff(workdir / (stem + "_layers.tif"),
                                     layered_tiff);
            timings.push_back(finish_timing("write_layered_tiff", timing));
        }
        }
        timings.push_back(finish_timing("total", total_start));

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

        print_stage_timings(timings);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
#endif
