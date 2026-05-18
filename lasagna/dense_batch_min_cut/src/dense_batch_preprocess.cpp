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
#include <memory>
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
constexpr double kIslandRemovalScoreThreshold = 0.5;
constexpr float kFlowGateRegionMergeRatio = 0.75f;
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

struct RimPosition {
    int contour = -1;
    float arc = 0.0f;
    float total = 0.0f;
};

struct DenseBatchScratch {
    cv::Mat rim_contour_input;
    std::vector<RimPosition> rim_lookup;

    std::vector<float> dense_routed_flow;
    std::vector<int> dense_graph_next_pixel;
    std::vector<float> dense_graph_next_distance;
    std::vector<float> dense_graph_seed_flow;
    std::vector<char> dense_graph_route_pixel;
    std::vector<char> dense_graph_root_seen;
    std::vector<char> dense_graph_edge_route_seen;
    std::vector<int> dense_seeded_by_row;
    std::vector<int> dense_grid_node_ids;
    std::vector<float> dense_carrier_value;
    std::vector<float> dense_carrier_route_dp;
    std::vector<float> dense_carrier_route_distance_dp;
    std::vector<int> dense_carrier_route_next;
};

DenseBatchScratch& dense_batch_scratch() {
    thread_local DenseBatchScratch scratch;
    return scratch;
}

template <typename T>
void resize_fill(std::vector<T>& values, const std::size_t size,
                 const T& value) {
    values.resize(size);
    std::fill(values.begin(), values.end(), value);
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
               timing.name == "tree_path_debug_render";
    };
    auto child_parent_name = [](const std::string& name) -> std::string {
        if (name.rfind("component.connect_ridges.", 0) == 0) {
            return "component.connect_ridges";
        }
        if (name.rfind("source_rim.rim_distance_ridges.", 0) == 0) {
            return "source_rim.rim_distance_ridges";
        }
        if (name.rfind("source_rim.", 0) == 0) {
            return "source_rim_skeleton";
        }
        if (name.rfind("component.", 0) == 0) {
            return "legacy_component";
        }
        if (name.rfind("dense_grid.", 0) == 0) {
            return "dense_backtrack_grid_carrier";
        }
        if (name.rfind("dense_graph_route.", 0) == 0) {
            return "dense_backtrack_graph_route";
        }
        return {};
    };
    auto display_stage_name = [&](const std::string& name) -> std::string {
        if (name.rfind("component.connect_ridges.", 0) == 0) {
            return "        " +
                   name.substr(std::string("component.connect_ridges.").size());
        }
        if (name.rfind("source_rim.rim_distance_ridges.", 0) == 0) {
            return "        " +
                   name.substr(
                       std::string("source_rim.rim_distance_ridges.").size());
        }
        if (name.rfind("source_rim.", 0) == 0) {
            return "    " + name.substr(std::string("source_rim.").size());
        }
        if (name.rfind("component.", 0) == 0) {
            return "    " + name.substr(std::string("component.").size());
        }
        if (name.rfind("dense_grid.", 0) == 0) {
            return "    " + name.substr(std::string("dense_grid.").size());
        }
        if (name.rfind("dense_graph_route.", 0) == 0) {
            return "    " +
                   name.substr(std::string("dense_graph_route.").size());
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
    float local_boost = 1.0f;
    float flow_weight_one = 100.0f;
    float flow_weight_zero = 20.0f;
    int compute_repeats = 1;
    bool write_outputs = true;
};

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " -i <image> [--source x,y] [--grid-step pixels]"
              << " [--backtrack-distance pixels]"
              << " [--local-boost value]"
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
        } else if (key == "--local-boost" && i + 1 < argc) {
            args.local_boost =
                std::clamp(std::stof(argv[++i]), 0.0f, 1.0f);
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

cv::Mat keep_source_white_components(const cv::Mat& binary,
                                     const std::vector<cv::Point>& sources) {
    CV_Assert(binary.type() == CV_8U);
    if (sources.empty()) {
        throw std::runtime_error("--source is required");
    }
    const cv::Point primary_source = sources.front();
    if (primary_source.x < 0 || primary_source.x >= binary.cols ||
        primary_source.y < 0 || primary_source.y >= binary.rows) {
        throw std::runtime_error("--source is outside the image");
    }

    cv::Mat white_domain(binary.size(), CV_8U, cv::Scalar(255));
    white_domain.setTo(0, binary);
    if (white_domain.at<std::uint8_t>(primary_source.y,
                                      primary_source.x) == 0) {
        throw std::runtime_error("--source must be inside the white distance domain");
    }

    cv::Mat labels;
    const int component_count =
        cv::connectedComponents(white_domain, labels, 8, CV_32S);
    std::vector<char> keep(static_cast<std::size_t>(component_count), 0);
    for (const cv::Point source : sources) {
        if (source.x < 0 || source.x >= binary.cols || source.y < 0 ||
            source.y >= binary.rows ||
            white_domain.at<std::uint8_t>(source.y, source.x) == 0) {
            continue;
        }
        const int label = labels.at<int>(source.y, source.x);
        if (label > 0 && label < component_count) {
            keep[static_cast<std::size_t>(label)] = 1;
        }
    }
    cv::Mat filtered(binary.size(), CV_8U, cv::Scalar(255));
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            const int label = labels.at<int>(y, x);
            if (label > 0 && label < component_count &&
                keep[static_cast<std::size_t>(label)] != 0) {
                filtered.at<std::uint8_t>(y, x) = 0;
            }
        }
    }
    return filtered;
}

cv::Mat keep_source_white_component(const cv::Mat& binary,
                                    const cv::Point source) {
    return keep_source_white_components(binary, std::vector<cv::Point>{source});
}

cv::Mat keep_source_components_white_domain(
        const cv::Mat& white_domain,
        const std::vector<cv::Point>& sources,
        const std::string& context) {
    CV_Assert(white_domain.type() == CV_8U);
    if (sources.empty()) {
        throw std::runtime_error(context + " source is required");
    }
    const cv::Point primary_source = sources.front();
    if (primary_source.x < 0 || primary_source.x >= white_domain.cols ||
        primary_source.y < 0 || primary_source.y >= white_domain.rows) {
        throw std::runtime_error(context + " source is outside the image");
    }
    if (white_domain.at<std::uint8_t>(primary_source.y,
                                      primary_source.x) == 0) {
        throw std::runtime_error(context +
                                 " source is outside the expanded rim domain");
    }

    cv::Mat labels;
    const int component_count =
        cv::connectedComponents(white_domain, labels, 8, CV_32S);
    std::vector<char> keep(static_cast<std::size_t>(component_count), 0);
    for (const cv::Point source : sources) {
        if (source.x < 0 || source.x >= white_domain.cols ||
            source.y < 0 || source.y >= white_domain.rows ||
            white_domain.at<std::uint8_t>(source.y, source.x) == 0) {
            continue;
        }
        const int label = labels.at<int>(source.y, source.x);
        if (label > 0 && label < component_count) {
            keep[static_cast<std::size_t>(label)] = 1;
        }
    }
    cv::Mat filtered = cv::Mat::zeros(white_domain.size(), CV_8U);
    for (int y = 0; y < labels.rows; ++y) {
        const int* label_row = labels.ptr<int>(y);
        std::uint8_t* out_row = filtered.ptr<std::uint8_t>(y);
        for (int x = 0; x < labels.cols; ++x) {
            const int label = label_row[x];
            if (label > 0 && label < component_count &&
                keep[static_cast<std::size_t>(label)] != 0) {
                out_row[x] = 255;
            }
        }
    }
    return filtered;
}

cv::Mat keep_source_component_white_domain(const cv::Mat& white_domain,
                                           const cv::Point source,
                                           const std::string& context) {
    return keep_source_components_white_domain(
        white_domain, std::vector<cv::Point>{source}, context);
}

cv::Mat make_padded_expanded_rim_label_domain(const cv::Mat& white_domain,
                                              const std::vector<cv::Point>&
                                                  sources) {
    CV_Assert(white_domain.type() == CV_8U);

    cv::Mat expanded_source;
    cv::bitwise_not(white_domain, expanded_source);
    if (!expanded_source.empty()) {
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(expanded_source, expanded_source, kernel);
    }

    cv::Mat expanded_white;
    cv::bitwise_not(expanded_source, expanded_white);
    if (!sources.empty() && (sources.front().x >= 0 || sources.front().y >= 0)) {
        expanded_white = keep_source_components_white_domain(
            expanded_white, sources, "rim-label");
    }

    cv::Mat domain(expanded_white.rows + 2, expanded_white.cols + 2, CV_8U,
                   cv::Scalar(0));
    if (!expanded_white.empty()) {
        expanded_white.copyTo(
            domain(cv::Rect(1, 1, expanded_white.cols, expanded_white.rows)));
    }

    return domain;
}

cv::Mat make_padded_expanded_rim_label_domain(const cv::Mat& white_domain,
                                              const cv::Point source) {
    return make_padded_expanded_rim_label_domain(
        white_domain, std::vector<cv::Point>{source});
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
    cv::ximgproc::thinning(binary, out, cv::ximgproc::THINNING_GUOHALL);
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

    const int cols = zero_source_mask.cols;
    std::vector<cv::Point> source_points(static_cast<std::size_t>(max_label + 1),
                                         cv::Point(-1, -1));

    // All callers build labels with DIST_LABEL_PIXEL, where each zero source
    // pixel owns one unique label. That makes the per-label writes independent.
    cv::parallel_for_(cv::Range(0, zero_source_mask.rows),
                      [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const std::uint8_t* mask_row =
                zero_source_mask.ptr<std::uint8_t>(y);
            const int* label_row = labels.ptr<int>(y);
            for (int x = 0; x < cols; ++x) {
                if (mask_row[x] != 0) {
                    continue;
                }
                const int label = label_row[x];
                if (label > 0) {
                    source_points[static_cast<std::size_t>(label)] =
                        cv::Point(x, y);
                }
            }
        }
    });
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

std::string rim_assignment_error(
    const cv::Mat& white_domain, const cv::Mat& source_pixel_labels,
    const std::vector<cv::Point>& source_points,
    const std::vector<RimPosition>& rim_lookup) {
    CV_Assert(white_domain.type() == CV_8U);
    CV_Assert(source_pixel_labels.type() == CV_32S);
    CV_Assert(white_domain.size() == source_pixel_labels.size());

    struct ComponentRimUse {
        int pixels = 0;
        int missing = 0;
        cv::Point sample{-1, -1};
        std::vector<int> contours;
    };

    const int rows = white_domain.rows;
    const int cols = white_domain.cols;
    if (rows == 0 || cols == 0) {
        return {};
    }

    cv::Mat white_components;
    const int component_count =
        cv::connectedComponents(white_domain, white_components, 8, CV_32S);
    std::vector<ComponentRimUse> rim_uses(
        static_cast<std::size_t>(component_count));

    const auto add_contour = [](ComponentRimUse& use, const int contour) {
        if (std::find(use.contours.begin(), use.contours.end(), contour) ==
            use.contours.end()) {
            use.contours.push_back(contour);
        }
    };
    const auto in_bounds = [rows, cols](const cv::Point pixel) {
        return pixel.x >= 0 && pixel.x < cols && pixel.y >= 0 &&
               pixel.y < rows;
    };

    for (int y = 0; y < rows; ++y) {
        const std::uint8_t* domain_row = white_domain.ptr<std::uint8_t>(y);
        const int* component_row = white_components.ptr<int>(y);
        const int* label_row = source_pixel_labels.ptr<int>(y);
        for (int x = 0; x < cols; ++x) {
            if (domain_row[x] == 0) {
                continue;
            }
            const int component = component_row[x];
            if (component <= 0 || component >= component_count) {
                continue;
            }

            ComponentRimUse& use =
                rim_uses[static_cast<std::size_t>(component)];
            ++use.pixels;
            if (use.sample.x < 0) {
                use.sample = cv::Point(x, y);
            }

            const int label = label_row[x];
            if (label <= 0 ||
                label >= static_cast<int>(source_points.size())) {
                ++use.missing;
                continue;
            }
            const cv::Point source = source_points[static_cast<std::size_t>(
                label)];
            if (!in_bounds(source)) {
                ++use.missing;
                continue;
            }

            const std::size_t source_index =
                static_cast<std::size_t>(source.y * cols + source.x);
            if (source_index >= rim_lookup.size() ||
                rim_lookup[source_index].contour < 0) {
                ++use.missing;
                continue;
            }
            add_contour(use, rim_lookup[source_index].contour);
        }
    }

    int components_with_missing = 0;
    int missing_assignments = 0;
    int multi_rim_components = 0;
    int max_contours_per_component = 0;
    for (int component = 1; component < component_count; ++component) {
        ComponentRimUse& use = rim_uses[static_cast<std::size_t>(component)];
        std::sort(use.contours.begin(), use.contours.end());
        missing_assignments += use.missing;
        max_contours_per_component = std::max(
            max_contours_per_component, static_cast<int>(use.contours.size()));
        if (use.pixels > 0 && use.contours.size() > 1) {
            ++multi_rim_components;
        }
        if (use.pixels > 0 && use.missing > 0) {
            ++components_with_missing;
        }
    }
    if (missing_assignments == 0) {
        return {};
    }

    std::ostringstream out;
    out << "source rim labeling has unassigned nearest-rim source pixels:"
        << " every source pixel selected by the rim distance transform must"
        << " map to a rim contour"
        << " white_components=" << std::max(0, component_count - 1)
        << " components_with_missing=" << components_with_missing
        << " missing_assignments=" << missing_assignments
        << " multi_rim_components=" << multi_rim_components
        << " max_rim_contours_per_component="
        << max_contours_per_component;

    int reported = 0;
    constexpr int kMaxReportedComponents = 5;
    constexpr int kMaxReportedContours = 8;
    for (int component = 1;
         component < component_count && reported < kMaxReportedComponents;
         ++component) {
        const ComponentRimUse& use =
            rim_uses[static_cast<std::size_t>(component)];
        if (use.pixels == 0 || use.missing == 0) {
            continue;
        }
        out << " component_" << component << "_pixels=" << use.pixels
            << " component_" << component << "_rim_contours="
            << use.contours.size() << " component_" << component
            << "_missing=" << use.missing << " component_" << component
            << "_sample=(" << use.sample.x << "," << use.sample.y << ")"
            << " component_" << component << "_contours=";
        const int contours_to_report =
            std::min<int>(kMaxReportedContours, use.contours.size());
        for (int i = 0; i < contours_to_report; ++i) {
            if (i > 0) {
                out << ",";
            }
            out << use.contours[static_cast<std::size_t>(i)];
        }
        if (contours_to_report < static_cast<int>(use.contours.size())) {
            out << ",...";
        }
        ++reported;
    }
    return out.str();
}

cv::Mat source_rim_distance_label_ridges(
    const cv::Mat& white_domain, const cv::Mat& source_pixel_labels,
    const std::vector<cv::Point>& source_points,
    const float min_rim_distance_px,
    cv::Mat* rim_distance_debug_out = nullptr,
    cv::Mat* rim_arc_debug_out = nullptr,
    std::vector<StageTiming>* timings = nullptr,
    std::string* rim_connectivity_error_out = nullptr) {
    CV_Assert(white_domain.type() == CV_8U);
    CV_Assert(source_pixel_labels.type() == CV_32S);

    DenseBatchScratch& scratch = dense_batch_scratch();
    const int rows = white_domain.rows;
    const int cols = white_domain.cols;
    constexpr bool kDebugSourceRimPixel = false;
    const cv::Point debug_pixel(129, 218);
    const auto record_timing = [&](const std::string& name,
                                   const TimingMark& timing) {
        if (timings != nullptr) {
            timings->push_back(finish_timing(
                "source_rim.rim_distance_ridges." + name, timing));
        }
    };

    std::vector<std::vector<cv::Point>> contours;
    {
        const TimingMark timing = start_timing();
        white_domain.copyTo(scratch.rim_contour_input);
        cv::findContours(scratch.rim_contour_input, contours, cv::RETR_LIST,
                         cv::CHAIN_APPROX_NONE);
        record_timing("white_boundary_contours", timing);
    }

    std::vector<RimPosition>& rim_lookup = scratch.rim_lookup;
    {
        const TimingMark timing = start_timing();
        resize_fill(rim_lookup, static_cast<std::size_t>(rows * cols),
                    RimPosition{});
        record_timing("rim_lookup_alloc", timing);
    }
    const auto linear_index = [cols](const cv::Point pixel) {
        return static_cast<std::size_t>(pixel.y * cols + pixel.x);
    };
    const auto in_bounds = [rows, cols](const cv::Point pixel) {
        return pixel.x >= 0 && pixel.x < cols && pixel.y >= 0 &&
               pixel.y < rows;
    };
    const auto segment_length = [](const cv::Point a, const cv::Point b) {
        const float dx = static_cast<float>(a.x - b.x);
        const float dy = static_cast<float>(a.y - b.y);
        return std::sqrt(dx * dx + dy * dy);
    };
    const auto assign_rim_position = [&](const cv::Point pixel,
                                         const RimPosition position) {
        if (!in_bounds(pixel)) {
            return;
        }
        RimPosition& current = rim_lookup[linear_index(pixel)];
        if (current.contour < 0) {
            current = position;
        }
    };

    {
        const TimingMark timing = start_timing();
        for (int contour_id = 0; contour_id < static_cast<int>(contours.size());
             ++contour_id) {
            const std::vector<cv::Point>& contour = contours[contour_id];
            if (contour.empty()) {
                continue;
            }

            std::vector<float> arc(contour.size(), 0.0f);
            for (std::size_t i = 1; i < contour.size(); ++i) {
                arc[i] =
                    arc[i - 1] + segment_length(contour[i], contour[i - 1]);
            }
            const float total_length = std::max(
                1.0f,
                arc.back() + segment_length(contour.front(), contour.back()));

            for (std::size_t i = 0; i < contour.size(); ++i) {
                assign_rim_position(
                    contour[i],
                    {contour_id, arc[i], total_length});
            }
            for (std::size_t i = 0; i < contour.size(); ++i) {
                const RimPosition position{contour_id, arc[i], total_length};
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const cv::Point pixel =
                            contour[i] + cv::Point(dx, dy);
                        if (!in_bounds(pixel) ||
                            white_domain.at<std::uint8_t>(pixel.y,
                                                           pixel.x) != 0) {
                            continue;
                        }
                        assign_rim_position(pixel, position);
                    }
                }
            }
        }
        record_timing("rim_lookup", timing);
    }

    const auto rim_position = [&](const cv::Point pixel) {
        if (!in_bounds(pixel)) {
            return RimPosition{};
        }
        return rim_lookup[linear_index(pixel)];
    };

    if (rim_connectivity_error_out != nullptr) {
        const TimingMark timing = start_timing();
        *rim_connectivity_error_out = rim_assignment_error(
            white_domain, source_pixel_labels, source_points, rim_lookup);
        record_timing("rim_assignment_check", timing);
    }

    if (rim_arc_debug_out != nullptr) {
        const TimingMark timing = start_timing();
        cv::Mat rim_arc_debug = cv::Mat::zeros(white_domain.size(), CV_8U);
        cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                std::uint8_t* out_row = rim_arc_debug.ptr<std::uint8_t>(y);
                for (int x = 0; x < cols; ++x) {
                    const RimPosition position =
                        rim_lookup[static_cast<std::size_t>(y * cols + x)];
                    if (position.contour < 0 || position.total <= 0.0f) {
                        continue;
                    }
                    const float t = std::max(
                        0.0f, std::min(1.0f, position.arc / position.total));
                    out_row[x] = static_cast<std::uint8_t>(
                        1 + static_cast<int>(std::lround(254.0f * t)));
                }
            }
        });
        *rim_arc_debug_out = rim_arc_debug;
        record_timing("rim_arc_debug", timing);
    }

    cv::Mat ridges;
    cv::Mat rim_distance_debug;
    {
        const TimingMark timing = start_timing();
        ridges = cv::Mat::zeros(white_domain.size(), CV_8U);
        rim_distance_debug = cv::Mat::zeros(white_domain.size(), CV_8U);
        cv::parallel_for_(cv::Range(0, white_domain.rows),
                          [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            for (int x = 0; x < white_domain.cols; ++x) {
                const bool debug_this_pixel =
                    kDebugSourceRimPixel && x == debug_pixel.x &&
                    y == debug_pixel.y;
                std::unique_ptr<std::ostringstream> debug;
                if constexpr (kDebugSourceRimPixel) {
                    if (debug_this_pixel) {
                        debug = std::make_unique<std::ostringstream>();
                        *debug << "source_rim_ridges debug pixel=("
                               << x << "," << y << ")\n";
                    }
                }
                if (white_domain.at<std::uint8_t>(y, x) == 0) {
                    if (debug != nullptr) {
                        *debug << "  skipped: outside white domain\n";
                        std::cout << debug->str();
                    }
                    continue;
                }
                const int center = source_pixel_labels.at<int>(y, x);
                if (center <= 0 ||
                    center >= static_cast<int>(source_points.size())) {
                    if (debug != nullptr) {
                        *debug << "  skipped: invalid center_label=" << center
                               << " source_points="
                               << source_points.size() << "\n";
                        std::cout << debug->str();
                    }
                    continue;
                }
                const cv::Point center_source = source_points[center];
                if (center_source.x < 0) {
                    if (debug != nullptr) {
                        *debug << "  skipped: missing center_source for label="
                               << center << "\n";
                        std::cout << debug->str();
                    }
                    continue;
                }
                if (debug != nullptr) {
                    *debug << "  center_label=" << center
                           << " center_source=(" << center_source.x << ","
                           << center_source.y << ")";
                }
                const RimPosition center_rim = rim_position(center_source);
                if (debug != nullptr) {
                    *debug << " center_contour=" << center_rim.contour
                           << " center_arc=" << center_rim.arc
                           << " center_total=" << center_rim.total << "\n";
                }

                float max_neighbor_rim_distance = 0.0f;
                for (int dy = -1; dy <= 1; ++dy) {
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
                        if (other <= 0 || other == center ||
                            other >= static_cast<int>(source_points.size())) {
                            continue;
                        }
                        const cv::Point other_source = source_points[other];
                        if (other_source.x < 0) {
                            continue;
                        }
                        const RimPosition other_rim =
                            rim_position(other_source);
                        float rim_distance = -1.0f;
                        const char* reason = "missing_rim";
                        if (center_rim.contour >= 0 &&
                            other_rim.contour >= 0) {
                            if (center_rim.contour != other_rim.contour) {
                                // Separate source rings are intentionally
                                // treated as infinitely far apart along the
                                // rim, so their Voronoi boundary is kept.
                                rim_distance = 255.0f;
                                reason = "different_contour";
                            } else {
                                const float arc_delta =
                                    std::abs(center_rim.arc - other_rim.arc);
                                rim_distance =
                                    std::min(arc_delta,
                                             center_rim.total - arc_delta);
                                if (rim_distance > min_rim_distance_px) {
                                    reason = "rim_gt_threshold";
                                } else {
                                    reason = "rim_le_threshold";
                                }
                            }
                            if (rim_distance > 0.0f) {
                                max_neighbor_rim_distance = std::max(
                                    max_neighbor_rim_distance, rim_distance);
                            }
                        }

                        if (debug != nullptr) {
                            *debug << "  neighbor=(" << nx << "," << ny
                                   << ") other_label=" << other
                                   << " other_source=(" << other_source.x
                                   << "," << other_source.y << ")"
                                   << " center_contour="
                                   << center_rim.contour
                                   << " center_arc=" << center_rim.arc
                                   << " center_total=" << center_rim.total
                                   << " other_contour=" << other_rim.contour
                                   << " other_arc=" << other_rim.arc
                                   << " other_total=" << other_rim.total
                                   << " rim_dist=" << rim_distance
                                   << " threshold=" << min_rim_distance_px
                                   << " reason=" << reason
                                   << " keep="
                                   << (rim_distance > min_rim_distance_px ? 1
                                                                          : 0)
                                   << "\n";
                        }
                    }
                }

                const bool keep =
                    max_neighbor_rim_distance > min_rim_distance_px;
                if (keep) {
                    ridges.at<std::uint8_t>(y, x) = 255;
                }
                rim_distance_debug.at<std::uint8_t>(y, x) =
                    static_cast<std::uint8_t>(
                        std::min(255.0f, max_neighbor_rim_distance));
                if (debug != nullptr) {
                    *debug << "  max_neighbor_rim_dist="
                           << max_neighbor_rim_distance
                           << " final_keep=" << (keep ? 1 : 0) << "\n";
                    std::cout << debug->str();
                }
            }
        }
        });
        record_timing("pixel_scan", timing);
    }

    if (rim_distance_debug_out != nullptr) {
        *rim_distance_debug_out = rim_distance_debug;
    }
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
        struct CandidateSearchTiming {
            double probe_ms = 0.0;
            double sparse_graph_build_ms = 0.0;
            double sparse_search_ms = 0.0;
            double sparse_materialize_ms = 0.0;
        };
        std::vector<CandidateSearchTiming> candidate_search_timings(
            static_cast<std::size_t>(num_candidates));
        const auto elapsed_ms_since = [](const Clock::time_point& start) {
            return std::chrono::duration<double, std::milli>(Clock::now() -
                                                             start)
                .count();
        };
        cv::parallel_for_(cv::Range(1, num_candidates),
                          [&](const cv::Range& range) {
            for (int label = range.start; label < range.end; ++label) {
            struct State {
                int node = 0;
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
                    if (a.owner != b.owner) {
                        return a.owner > b.owner;
                    }
                    return a.node > b.node;
                }
            };

            struct LocalEdgeCandidate {
                int label = 0;
                int a = 0;
                int b = 0;
                float bottleneck_dt = 0.0f;
                int length = 0;
                int a_node = -1;
                int b_node = -1;
                int crossing_edge = -1;
                cv::Point clean_a{-1, -1};
                cv::Point clean_b{-1, -1};
            };

            struct GraphEdge {
                int u = -1;
                int v = -1;
                float min_dt = 0.0f;
                int length = 0;
                std::vector<cv::Point> path;
            };

            const CandidateWorkspace& workspace = candidate_workspaces[label];
            const int total = static_cast<int>(workspace.pixels.size());
            if (total == 0) {
                continue;
            }

            const Clock::time_point probe_start = Clock::now();
            std::vector<std::array<int, 8>> pixel_neighbors(
                static_cast<std::size_t>(total));
            std::vector<std::uint8_t> pixel_degree(
                static_cast<std::size_t>(total), 0);
            for (int idx = 0; idx < total; ++idx) {
                pixel_neighbors[idx].fill(-1);
                const cv::Point pixel = workspace.pixels[idx];
                int degree = 0;
                for (const cv::Point dir : kDirs) {
                    const cv::Point next_pixel = pixel + dir;
                    if (!workspace.bounds.contains(next_pixel)) {
                        continue;
                    }
                    const int next_idx = workspace.local_index.at<int>(
                        next_pixel.y - workspace.bounds.y,
                        next_pixel.x - workspace.bounds.x);
                    if (next_idx >= 0) {
                        pixel_neighbors[idx][degree++] = next_idx;
                    }
                }
                pixel_degree[idx] = static_cast<std::uint8_t>(degree);
            }

            std::vector<std::uint8_t> is_attachment(
                static_cast<std::size_t>(total), 0);
            for (const Attachment& attachment : attachments[label]) {
                const int idx = workspace.local_index.at<int>(
                    attachment.pixel.y - workspace.bounds.y,
                    attachment.pixel.x - workspace.bounds.x);
                if (idx < 0) {
                    continue;
                }
                is_attachment[idx] = 1;
            }

            std::vector<int> node_for_pixel(static_cast<std::size_t>(total),
                                            -1);
            std::vector<int> node_pixels;
            node_pixels.reserve(static_cast<std::size_t>(total / 4 + 1));
            for (int idx = 0; idx < total; ++idx) {
                if (pixel_degree[idx] != 2 || is_attachment[idx] != 0) {
                    node_for_pixel[idx] =
                        static_cast<int>(node_pixels.size());
                    node_pixels.push_back(idx);
                }
            }
            candidate_search_timings[label].probe_ms =
                elapsed_ms_since(probe_start);
            if (node_pixels.empty()) {
                continue;
            }

            const Clock::time_point sparse_graph_build_start = Clock::now();
            std::vector<GraphEdge> graph_edges;
            std::vector<std::vector<int>> adjacency(node_pixels.size());
            std::unordered_set<std::uint64_t> visited_starts;
            const auto directed_key = [](int a, int b) {
                return (static_cast<std::uint64_t>(
                            static_cast<std::uint32_t>(a))
                        << 32) |
                       static_cast<std::uint32_t>(b);
            };
            const auto add_graph_edge = [&](GraphEdge edge) {
                if (edge.u < 0 || edge.v < 0 || edge.u == edge.v ||
                    edge.path.size() < 2) {
                    return;
                }
                edge.length = static_cast<int>(edge.path.size());
                const int edge_id = static_cast<int>(graph_edges.size());
                adjacency[static_cast<std::size_t>(edge.u)].push_back(edge_id);
                adjacency[static_cast<std::size_t>(edge.v)].push_back(edge_id);
                graph_edges.push_back(std::move(edge));
            };

            for (int start_node = 0;
                 start_node < static_cast<int>(node_pixels.size());
                 ++start_node) {
                const int start_idx = node_pixels[start_node];
                const cv::Point start_pixel = workspace.pixels[start_idx];
                for (int neighbor_slot = 0; neighbor_slot < 8;
                     ++neighbor_slot) {
                    const int first_idx =
                        pixel_neighbors[start_idx][neighbor_slot];
                    if (first_idx < 0) {
                        break;
                    }
                    const std::uint64_t start_key =
                        directed_key(start_idx, first_idx);
                    if (visited_starts.find(start_key) !=
                        visited_starts.end()) {
                        continue;
                    }
                    visited_starts.insert(start_key);

                    GraphEdge edge;
                    edge.u = start_node;
                    edge.min_dt =
                        dt.at<float>(start_pixel.y, start_pixel.x);
                    edge.path.push_back(start_pixel);

                    int prev_idx = start_idx;
                    int cur_idx = first_idx;
                    bool valid = true;
                    while (true) {
                        const cv::Point cur_pixel =
                            workspace.pixels[cur_idx];
                        edge.path.push_back(cur_pixel);
                        edge.min_dt = std::min(
                            edge.min_dt,
                            dt.at<float>(cur_pixel.y, cur_pixel.x));

                        const int cur_node = node_for_pixel[cur_idx];
                        if (cur_node >= 0) {
                            edge.v = cur_node;
                            visited_starts.insert(
                                directed_key(cur_idx, prev_idx));
                            break;
                        }

                        int next_idx = -1;
                        for (int slot = 0; slot < 8; ++slot) {
                            const int candidate =
                                pixel_neighbors[cur_idx][slot];
                            if (candidate < 0) {
                                break;
                            }
                            if (candidate != prev_idx) {
                                next_idx = candidate;
                                break;
                            }
                        }
                        if (next_idx < 0) {
                            valid = false;
                            break;
                        }
                        prev_idx = cur_idx;
                        cur_idx = next_idx;
                    }

                    if (valid) {
                        add_graph_edge(std::move(edge));
                    }
                }
            }
            if (graph_edges.empty()) {
                continue;
            }
            candidate_search_timings[label].sparse_graph_build_ms =
                elapsed_ms_since(sparse_graph_build_start);

            const Clock::time_point sparse_search_start = Clock::now();
            const int node_count = static_cast<int>(node_pixels.size());
            std::vector<float> best(static_cast<std::size_t>(node_count),
                                    -1.0f);
            std::vector<int> best_length(static_cast<std::size_t>(node_count),
                                         std::numeric_limits<int>::max());
            std::vector<int> parent_node(static_cast<std::size_t>(node_count),
                                         -1);
            std::vector<int> parent_edge(static_cast<std::size_t>(node_count),
                                         -1);
            std::vector<int> owner(static_cast<std::size_t>(node_count), 0);
            std::vector<cv::Point> owner_clean(
                static_cast<std::size_t>(node_count), cv::Point(-1, -1));
            std::priority_queue<State, std::vector<State>, StateLess> queue;

            for (const Attachment& attachment : attachments[label]) {
                const int idx = workspace.local_index.at<int>(
                    attachment.pixel.y - workspace.bounds.y,
                    attachment.pixel.x - workspace.bounds.x);
                if (idx < 0) {
                    continue;
                }
                const int node = node_for_pixel[idx];
                if (node < 0) {
                    continue;
                }
                const float start_dt =
                    dt.at<float>(attachment.pixel.y, attachment.pixel.x);
                const bool better =
                    start_dt > best[node] ||
                    (start_dt == best[node] &&
                     (owner[node] == 0 ||
                      attachment.component < owner[node]));
                if (!better) {
                    continue;
                }
                best[node] = start_dt;
                best_length[node] = 1;
                parent_node[node] = node;
                parent_edge[node] = -1;
                owner[node] = attachment.component;
                owner_clean[node] = attachment.clean_pixel;
                queue.push({node, attachment.component, start_dt, 1});
            }

            if (queue.empty()) {
                continue;
            }

            const auto edge_path_between = [&](int edge_id, int from,
                                               int to) {
                const GraphEdge& graph_edge = graph_edges[edge_id];
                std::vector<cv::Point> path = graph_edge.path;
                if (graph_edge.u == from && graph_edge.v == to) {
                    return path;
                }
                std::reverse(path.begin(), path.end());
                return path;
            };

            const auto path_to_root = [&](int node) {
                std::vector<cv::Point> path;
                if (node < 0) {
                    return path;
                }

                path.push_back(workspace.pixels[node_pixels[node]]);
                while (parent_node[node] >= 0 &&
                       parent_node[node] != node) {
                    const int next_node = parent_node[node];
                    const int edge_id = parent_edge[node];
                    std::vector<cv::Point> segment =
                        edge_path_between(edge_id, node, next_node);
                    path.insert(path.end(), segment.begin() + 1,
                                segment.end());
                    node = next_node;
                }
                return path;
            };

            const auto append_path_tail =
                [](std::vector<cv::Point>& dst,
                   const std::vector<cv::Point>& src) {
                    if (src.empty()) {
                        return;
                    }
                    if (dst.empty()) {
                        dst.insert(dst.end(), src.begin(), src.end());
                    } else {
                        dst.insert(dst.end(), src.begin() + 1, src.end());
                    }
                };

            std::vector<LocalEdgeCandidate> local_candidates;
            while (!queue.empty()) {
                const State state = queue.top();
                queue.pop();
                if (state.owner != owner[state.node] ||
                    state.bottleneck < best[state.node] ||
                    (state.bottleneck == best[state.node] &&
                     state.length > best_length[state.node])) {
                    continue;
                }

                for (const int edge_id : adjacency[state.node]) {
                    const GraphEdge& graph_edge = graph_edges[edge_id];
                    const int next_node =
                        graph_edge.u == state.node ? graph_edge.v
                                                   : graph_edge.u;

                    if (owner[next_node] > 0 &&
                        owner[next_node] != state.owner) {
                        LocalEdgeCandidate edge;
                        edge.label = label;
                        edge.a = state.owner;
                        edge.b = owner[next_node];
                        edge.clean_a = owner_clean[state.node];
                        edge.clean_b = owner_clean[next_node];
                        edge.bottleneck_dt =
                            std::min({state.bottleneck, best[next_node],
                                      graph_edge.min_dt});
                        edge.length = best_length[state.node] +
                                      best_length[next_node] +
                                      graph_edge.length - 2;
                        edge.a_node = state.node;
                        edge.b_node = next_node;
                        edge.crossing_edge = edge_id;
                        local_candidates.push_back(edge);
                        break;
                    }

                    const float next_bottleneck =
                        std::min(state.bottleneck, graph_edge.min_dt);
                    const int next_length =
                        state.length + graph_edge.length - 1;
                    if (owner[next_node] == 0 ||
                        next_bottleneck > best[next_node] ||
                        (owner[next_node] == state.owner &&
                         next_bottleneck == best[next_node] &&
                         next_length < best_length[next_node])) {
                        best[next_node] = next_bottleneck;
                        best_length[next_node] = next_length;
                        parent_node[next_node] = state.node;
                        parent_edge[next_node] = edge_id;
                        owner[next_node] = state.owner;
                        owner_clean[next_node] = owner_clean[state.node];
                        queue.push({next_node, state.owner, next_bottleneck,
                                    next_length});
                    }
                }
            }
            candidate_search_timings[label].sparse_search_ms =
                elapsed_ms_since(sparse_search_start);

            const Clock::time_point sparse_materialize_start = Clock::now();
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
                    path_to_root(candidate.b_node);
                std::reverse(b_path.begin(), b_path.end());
                edge.path = std::move(b_path);
                std::vector<cv::Point> crossing_path = edge_path_between(
                    candidate.crossing_edge, candidate.b_node,
                    candidate.a_node);
                append_path_tail(edge.path, crossing_path);
                std::vector<cv::Point> a_path =
                    path_to_root(candidate.a_node);
                append_path_tail(edge.path, a_path);
                edge.length = static_cast<int>(edge.path.size());
                deduped_edges.push_back(std::move(edge));
            }
            edges_by_candidate[label] = std::move(deduped_edges);
            candidate_search_timings[label].sparse_materialize_ms =
                elapsed_ms_since(sparse_materialize_start);
        }
    });
        if (timings != nullptr) {
            CandidateSearchTiming total_search_timing;
            for (const CandidateSearchTiming& candidate_timing :
                 candidate_search_timings) {
                total_search_timing.probe_ms += candidate_timing.probe_ms;
                total_search_timing.sparse_graph_build_ms +=
                    candidate_timing.sparse_graph_build_ms;
                total_search_timing.sparse_search_ms +=
                    candidate_timing.sparse_search_ms;
                total_search_timing.sparse_materialize_ms +=
                    candidate_timing.sparse_materialize_ms;
            }
            timings->push_back(
                {"component.connect_ridges.candidate_probe_work",
                 total_search_timing.probe_ms,
                 total_search_timing.probe_ms});
            timings->push_back(
                {"component.connect_ridges.sparse_graph_build_work",
                 total_search_timing.sparse_graph_build_ms,
                 total_search_timing.sparse_graph_build_ms});
            timings->push_back(
                {"component.connect_ridges.sparse_search_work",
                 total_search_timing.sparse_search_ms,
                 total_search_timing.sparse_search_ms});
            timings->push_back(
                {"component.connect_ridges.sparse_materialize_work",
                 total_search_timing.sparse_materialize_ms,
                 total_search_timing.sparse_materialize_ms});
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

struct SourceRimSkeletonResult {
    cv::Mat source_rim_ridges;
    cv::Mat source_rim_distance;
    cv::Mat source_rim_arc;
    cv::Mat source_rim_arc_skeleton;
    cv::Mat loops_connected;
    std::vector<StageTiming> timings;
    std::string rim_connectivity_error;
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
    std::vector<std::vector<cv::Point>> node_pixel_groups;
    std::vector<std::vector<cv::Point>> pruned_graph_component_pixel_groups;
    cv::Mat node_mask;
    cv::Mat edge_mask;
    int skeleton_pixels = 0;
    int node_pixels = 0;
    int edge_path_pixels = 0;
    int unique_edge_pixels = 0;
    int missing_pixels = 0;
    int adjacent_component_repairs = 0;
    int adjacent_component_contacts = 0;
    int pruned_graph_components = 0;
    int pruned_graph_nodes = 0;
    int pruned_graph_edges = 0;
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
    int adjacent_component_repairs = 0;
    int adjacent_component_contacts = 0;
    cv::Point adjacent_component_a{-1, -1};
    cv::Point adjacent_component_b{-1, -1};
    int adjacent_component_a_id = -1;
    int adjacent_component_b_id = -1;
    int pruned_graph_components = 0;
    int pruned_graph_nodes = 0;
    int pruned_graph_edges = 0;
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

    const std::array<cv::Point, 8> kSortedDirs = {
        {{-1, -1}, {0, -1}, {1, -1}, {-1, 0},
         {1, 0},   {-1, 1}, {0, 1},  {1, 1}}};
    constexpr std::array<std::uint8_t, 8> kTopologyBits = {
        {1U << 1U, 1U << 2U, 1U << 4U, 1U << 7U,
         1U << 6U, 1U << 5U, 1U << 3U, 1U << 0U}};
    const int rows = skel.rows;
    const int cols = skel.cols;
    const auto linear_index = [cols](const cv::Point pixel) {
        return static_cast<std::size_t>(pixel.y * cols + pixel.x);
    };
    const auto in_bounds = [rows, cols](const cv::Point pixel) {
        return pixel.x >= 0 && pixel.x < cols && pixel.y >= 0 &&
               pixel.y < rows;
    };
    std::vector<std::uint8_t> neighbor_masks(
        static_cast<std::size_t>(rows * cols), 0);
    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const std::uint8_t* prev_row =
                y > 0 ? skel.ptr<std::uint8_t>(y - 1) : nullptr;
            const std::uint8_t* skel_row = skel.ptr<std::uint8_t>(y);
            const std::uint8_t* next_row =
                y + 1 < rows ? skel.ptr<std::uint8_t>(y + 1) : nullptr;
            for (int x = 0; x < cols; ++x) {
                if (skel_row[x] == 0) {
                    continue;
                }
                std::uint8_t mask = 0;
                if (prev_row != nullptr) {
                    if (x > 0 && prev_row[x - 1] != 0) {
                        mask |= static_cast<std::uint8_t>(1U << 0U);
                    }
                    if (prev_row[x] != 0) {
                        mask |= static_cast<std::uint8_t>(1U << 1U);
                    }
                    if (x + 1 < cols && prev_row[x + 1] != 0) {
                        mask |= static_cast<std::uint8_t>(1U << 2U);
                    }
                }
                if (x > 0 && skel_row[x - 1] != 0) {
                    mask |= static_cast<std::uint8_t>(1U << 3U);
                }
                if (x + 1 < cols && skel_row[x + 1] != 0) {
                    mask |= static_cast<std::uint8_t>(1U << 4U);
                }
                if (next_row != nullptr) {
                    if (x > 0 && next_row[x - 1] != 0) {
                        mask |= static_cast<std::uint8_t>(1U << 5U);
                    }
                    if (next_row[x] != 0) {
                        mask |= static_cast<std::uint8_t>(1U << 6U);
                    }
                    if (x + 1 < cols && next_row[x + 1] != 0) {
                        mask |= static_cast<std::uint8_t>(1U << 7U);
                    }
                }
                neighbor_masks[static_cast<std::size_t>(y * cols + x)] = mask;
            }
        }
    });
    const auto neighbor_mask_at = [&](const cv::Point pixel) {
        return neighbor_masks[linear_index(pixel)];
    };
    const auto neighbor_count = [](std::uint8_t mask) {
        int count = 0;
        while (mask != 0) {
            count += static_cast<int>(mask & 1U);
            mask >>= 1U;
        }
        return count;
    };
    const auto branch_count_from_mask = [&](const std::uint8_t mask) {
        int count = 0;
        for (int i = 0; i < static_cast<int>(kTopologyBits.size()); ++i) {
            const bool current = (mask & kTopologyBits[i]) != 0;
            const bool next =
                (mask & kTopologyBits[(i + 1) % kTopologyBits.size()]) != 0;
            if (!current && next) {
                ++count;
            }
        }
        return count;
    };
    const auto for_sorted_neighbors = [&](const cv::Point pixel,
                                          const auto& fn) {
        const std::uint8_t mask = neighbor_mask_at(pixel);
        for (int i = 0; i < static_cast<int>(kSortedDirs.size()); ++i) {
            if ((mask & (1U << i)) == 0) {
                continue;
            }
            fn(pixel + kSortedDirs[i]);
        }
    };

    cv::Mat node_seed = cv::Mat::zeros(skel.size(), CV_8U);
    for (int y = 0; y < rows; ++y) {
        const std::uint8_t* skel_row = skel.ptr<std::uint8_t>(y);
        std::uint8_t* node_row = node_seed.ptr<std::uint8_t>(y);
        for (int x = 0; x < cols; ++x) {
            if (skel_row[x] == 0) {
                continue;
            }
            const cv::Point pixel(x, y);
            const std::uint8_t mask = neighbor_mask_at(pixel);
            const int degree = neighbor_count(mask);
            const int branches = branch_count_from_mask(mask);
            if (degree <= 1 || branches >= 3) {
                node_row[x] = 255;
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
    for (int y = 0; y < rows; ++y) {
        const int* labels = node_labels.ptr<int>(y);
        for (int x = 0; x < cols; ++x) {
            const int label = labels[x];
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
        return in_bounds(pixel) &&
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
    struct LabelList {
        std::array<int, 8> values{};
        int count = 0;

        void add_unique(int label) {
            if (label <= 0) {
                return;
            }
            for (int i = 0; i < count; ++i) {
                if (values[i] == label) {
                    return;
                }
            }
            values[count++] = label;
        }

        void sort_values() {
            std::sort(values.begin(), values.begin() + count);
        }

        void erase(int label) {
            int write = 0;
            for (int read = 0; read < count; ++read) {
                if (values[read] != label) {
                    values[write++] = values[read];
                }
            }
            count = write;
        }

        bool contains(int label) const {
            for (int i = 0; i < count; ++i) {
                if (values[i] == label) {
                    return true;
                }
            }
            return false;
        }

        bool empty() const {
            return count == 0;
        }

        int front() const {
            return values[0];
        }
    };
    const auto adjacent_nodes = [&](const cv::Point pixel) {
        LabelList nodes;
        for_sorted_neighbors(pixel, [&](const cv::Point neighbor) {
            nodes.add_unique(node_labels.at<int>(neighbor.y, neighbor.x));
        });
        nodes.sort_values();
        return nodes;
    };
    const auto choose_next = [&](const cv::Point current,
                                 const std::array<cv::Point, 8>& candidates,
                                 const int candidate_count) {
        cv::Point four_candidate(-1, -1);
        int four_count = 0;
        for (int i = 0; i < candidate_count; ++i) {
            const cv::Point candidate = candidates[i];
            if (four_connected(current, candidate)) {
                four_candidate = candidate;
                ++four_count;
            }
        }
        if (four_count == 1) {
            return four_candidate;
        }
        if (candidate_count == 1) {
            return candidates[0];
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

	    const auto append_edge_pixel = [&](const cv::Point pixel) {
	        visited_edges.at<std::uint8_t>(pixel.y, pixel.x) = 255;
	        edge.pixels.push_back(pixel);
	        edge.capacity =
	            std::min(edge.capacity,
	                     capacity_from_dt(dt.at<float>(pixel.y, pixel.x)));
	    };

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

	        LabelList nodes = adjacent_nodes(current);
	        nodes.erase(start_node);
	        if (!nodes.empty()) {
	            append_edge_pixel(current);
	            edge.b = nodes.front();
	            break;
	        }

            std::array<cv::Point, 8> candidates;
            int candidate_count = 0;
            for_sorted_neighbors(current, [&](const cv::Point neighbor) {
                if (neighbor == previous || is_node(neighbor) ||
                    !is_edge_pixel(neighbor) ||
                    visited_edges.at<std::uint8_t>(neighbor.y, neighbor.x) !=
                        0) {
                    return;
                }
                candidates[candidate_count++] = neighbor;
            });

	        const cv::Point next =
	            choose_next(current, candidates, candidate_count);
	        if (next.x < 0 && candidate_count > 0) {
	            append_edge_pixel(current);
	            edge.b = make_node(current);
	            break;
	        }

	        append_edge_pixel(current);

	        if (next.x < 0) {
	            nodes = adjacent_nodes(current);
	            if (edge.pixels.size() > 1 &&
	                nodes.contains(start_node)) {
	                edge.b = start_node;
	            } else {
	                edge.b = make_node(current);
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
            for_sorted_neighbors(node_pixel, [&](const cv::Point neighbor) {
                if (is_edge_pixel(neighbor) &&
                    visited_edges.at<std::uint8_t>(neighbor.y, neighbor.x) ==
                        0) {
                    trace_edge(static_cast<int>(node), node_pixel, neighbor);
                }
            });
        }
    }

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const cv::Point start(x, y);
            if (!is_edge_pixel(start) ||
                visited_edges.at<std::uint8_t>(y, x) != 0) {
                continue;
            }
            const int node = make_node(start);
            visited_edges.at<std::uint8_t>(y, x) = 255;
            std::array<cv::Point, 8> candidates;
            int candidate_count = 0;
            for_sorted_neighbors(start, [&](const cv::Point neighbor) {
                if (is_edge_pixel(neighbor) &&
                    visited_edges.at<std::uint8_t>(neighbor.y, neighbor.x) ==
                        0) {
                    candidates[candidate_count++] = neighbor;
                }
            });
            if (candidate_count == 0) {
                continue;
            }
            const cv::Point first = choose_next(start, candidates,
                                               candidate_count);
            trace_edge(node, start,
                       first.x >= 0 ? first : candidates[0]);
        }
    }

    struct RasterComponentContact {
        bool found = false;
        cv::Point a{-1, -1};
        cv::Point b{-1, -1};
        int component_a = -1;
        int component_b = -1;
        int contacts = 0;
    };

    const auto edge_capacity_from_pixels =
        [&](const std::vector<cv::Point>& pixels) {
            float capacity = std::numeric_limits<float>::max();
            for (const cv::Point pixel : pixels) {
                if (!in_bounds(pixel)) {
                    continue;
                }
                capacity = std::min(
                    capacity, capacity_from_dt(dt.at<float>(pixel.y, pixel.x)));
            }
            return capacity == std::numeric_limits<float>::max() ? 0.0f
                                                                 : capacity;
        };

    const auto node_component_ids = [&]() {
        struct LocalDisjointSet {
            std::vector<int> parent;
            explicit LocalDisjointSet(int n)
                : parent(static_cast<std::size_t>(n)) {
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
        LocalDisjointSet sets(node_count);
        const auto valid_node = [&](int node) {
            return node > 0 && node < node_count;
        };
        for (const GraphEdge& edge : graph.edges) {
            if (valid_node(edge.a) && valid_node(edge.b) &&
                edge.a != edge.b) {
                sets.unite(edge.a, edge.b);
            }
        }

        std::vector<int> root_to_component(
            static_cast<std::size_t>(node_count), -1);
        std::vector<int> node_component(
            static_cast<std::size_t>(node_count), -1);
        int next_component = 0;
        for (int node = 1; node < node_count; ++node) {
            const int root = sets.find(node);
            int& component =
                root_to_component[static_cast<std::size_t>(root)];
            if (component < 0) {
                component = next_component++;
            }
            node_component[static_cast<std::size_t>(node)] = component;
        }
        return node_component;
    };

    const auto find_raster_component_contact = [&]() {
        RasterComponentContact contact;
        const std::vector<int> node_component = node_component_ids();
        const int node_count = static_cast<int>(node_component.size());
        const auto component_for_node = [&](int node) {
            if (node <= 0 || node >= node_count) {
                return -1;
            }
            return node_component[static_cast<std::size_t>(node)];
        };

        cv::Mat pixel_component(skel.size(), CV_32S, cv::Scalar(-1));
        const auto set_pixel_component = [&](const cv::Point pixel,
                                             int component) {
            if (!in_bounds(pixel) || component < 0) {
                return;
            }
            int& dst = pixel_component.at<int>(pixel.y, pixel.x);
            if (dst < 0) {
                dst = component;
            }
        };

        const std::size_t node_group_count =
            std::min(node_pixels.size(), node_component.size());
        for (std::size_t node = 1; node < node_group_count; ++node) {
            const int component = node_component[node];
            for (const cv::Point pixel : node_pixels[node]) {
                set_pixel_component(pixel, component);
            }
        }
        for (const GraphEdge& edge : graph.edges) {
            int component = component_for_node(edge.a);
            if (component < 0) {
                component = component_for_node(edge.b);
            }
            for (const cv::Point pixel : edge.pixels) {
                set_pixel_component(pixel, component);
            }
        }

        const std::array<cv::Point, 4> forward_dirs = {
            {{1, 0}, {-1, 1}, {0, 1}, {1, 1}}};
        for (int y = 0; y < rows; ++y) {
            const int* component_row = pixel_component.ptr<int>(y);
            for (int x = 0; x < cols; ++x) {
                const int component = component_row[x];
                if (component < 0) {
                    continue;
                }
                const cv::Point pixel(x, y);
                for (const cv::Point dir : forward_dirs) {
                    const cv::Point neighbor = pixel + dir;
                    if (!in_bounds(neighbor)) {
                        continue;
                    }
                    const int other =
                        pixel_component.at<int>(neighbor.y, neighbor.x);
                    if (other < 0 || other == component) {
                        continue;
                    }
                    ++contact.contacts;
                    if (!contact.found) {
                        contact.found = true;
                        contact.a = pixel;
                        contact.b = neighbor;
                        contact.component_a = component;
                        contact.component_b = other;
                    }
                }
            }
        }
        return contact;
    };

    const auto split_edge_at_pixel = [&](int edge_index,
                                         const cv::Point pixel) {
        if (edge_index < 0 ||
            edge_index >= static_cast<int>(graph.edges.size())) {
            return make_node(pixel);
        }

        const GraphEdge edge = graph.edges[static_cast<std::size_t>(
            edge_index)];
        const auto it =
            std::find(edge.pixels.begin(), edge.pixels.end(), pixel);
        if (it == edge.pixels.end()) {
            return make_node(pixel);
        }

        const int node = make_node(pixel);
        const int split_index =
            static_cast<int>(std::distance(edge.pixels.begin(), it));
        std::vector<cv::Point> first(edge.pixels.begin(),
                                     edge.pixels.begin() + split_index + 1);
        std::vector<cv::Point> second(edge.pixels.begin() + split_index,
                                      edge.pixels.end());

        graph.edges[static_cast<std::size_t>(edge_index)] =
            GraphEdge{edge.a, node, edge_capacity_from_pixels(first), first};
        graph.edges.push_back(
            GraphEdge{node, edge.b, edge_capacity_from_pixels(second), second});
        return node;
    };

    const auto ensure_graph_node_at_pixel = [&](const cv::Point pixel) {
        if (!in_bounds(pixel)) {
            return 0;
        }
        const int existing = node_labels.at<int>(pixel.y, pixel.x);
        if (existing > 0) {
            return existing;
        }
        for (int edge_index = 0;
             edge_index < static_cast<int>(graph.edges.size()); ++edge_index) {
            const std::vector<cv::Point>& pixels =
                graph.edges[static_cast<std::size_t>(edge_index)].pixels;
            if (std::find(pixels.begin(), pixels.end(), pixel) !=
                pixels.end()) {
                return split_edge_at_pixel(edge_index, pixel);
            }
        }
        return make_node(pixel);
    };

    int adjacent_component_repairs = 0;
    const int max_adjacent_component_repairs =
        std::max(1, graph.skeleton_pixels * 4);
    while (true) {
        const RasterComponentContact contact =
            find_raster_component_contact();
        if (!contact.found) {
            graph.adjacent_component_contacts = contact.contacts;
            break;
        }
        if (adjacent_component_repairs >= max_adjacent_component_repairs) {
            graph.adjacent_component_contacts = contact.contacts;
            break;
        }

        const int node_a = ensure_graph_node_at_pixel(contact.a);
        const int node_b = ensure_graph_node_at_pixel(contact.b);
        if (node_a > 0 && node_b > 0 && node_a != node_b) {
            std::vector<cv::Point> connector_pixels{contact.a, contact.b};
            graph.edges.push_back(
                GraphEdge{node_a, node_b,
                          edge_capacity_from_pixels(connector_pixels),
                          connector_pixels});
            ++adjacent_component_repairs;
        } else {
            break;
        }
    }
    graph.adjacent_component_repairs = adjacent_component_repairs;

    const auto prune_to_largest_graph_component = [&]() {
        const std::vector<int> node_component = node_component_ids();
        int component_count = 0;
        for (int node = 1; node < static_cast<int>(node_component.size());
             ++node) {
            component_count =
                std::max(component_count, node_component[node] + 1);
        }
        if (component_count <= 1) {
            return;
        }

        std::vector<int> component_nodes(
            static_cast<std::size_t>(component_count), 0);
        std::vector<int> component_edges(
            static_cast<std::size_t>(component_count), 0);
        for (int node = 1; node < static_cast<int>(node_component.size());
             ++node) {
            const int component = node_component[node];
            if (component >= 0) {
                ++component_nodes[static_cast<std::size_t>(component)];
            }
        }
        for (const GraphEdge& edge : graph.edges) {
            if (edge.a <= 0 ||
                edge.a >= static_cast<int>(node_component.size())) {
                continue;
            }
            const int component = node_component[edge.a];
            if (component >= 0) {
                ++component_edges[static_cast<std::size_t>(component)];
            }
        }

        int keep_component = 0;
        for (int component = 1; component < component_count; ++component) {
            if (component_nodes[component] > component_nodes[keep_component] ||
                (component_nodes[component] == component_nodes[keep_component] &&
                 component_edges[component] >
                     component_edges[keep_component])) {
                keep_component = component;
            }
        }

        graph.pruned_graph_component_pixel_groups.clear();
        std::vector<int> pruned_group_for_component(
            static_cast<std::size_t>(component_count), -1);
        const auto pruned_group = [&](int component) -> std::vector<cv::Point>& {
            int& group =
                pruned_group_for_component[static_cast<std::size_t>(component)];
            if (group < 0) {
                group = static_cast<int>(
                    graph.pruned_graph_component_pixel_groups.size());
                graph.pruned_graph_component_pixel_groups.emplace_back();
            }
            return graph.pruned_graph_component_pixel_groups[
                static_cast<std::size_t>(group)];
        };
        for (int node = 1; node < static_cast<int>(node_component.size());
             ++node) {
            const int component = node_component[node];
            if (component < 0 || component == keep_component ||
                node >= static_cast<int>(node_pixels.size())) {
                continue;
            }
            std::vector<cv::Point>& pixels = pruned_group(component);
            pixels.insert(pixels.end(),
                          node_pixels[static_cast<std::size_t>(node)].begin(),
                          node_pixels[static_cast<std::size_t>(node)].end());
        }
        for (const GraphEdge& edge : graph.edges) {
            if (edge.a <= 0 ||
                edge.a >= static_cast<int>(node_component.size())) {
                continue;
            }
            const int component = node_component[edge.a];
            if (component < 0 || component == keep_component) {
                continue;
            }
            std::vector<cv::Point>& pixels = pruned_group(component);
            pixels.insert(pixels.end(), edge.pixels.begin(), edge.pixels.end());
        }

        std::vector<int> node_remap(node_component.size(), -1);
        node_remap[0] = 0;
        std::vector<cv::Point2f> kept_nodes;
        std::vector<std::vector<cv::Point>> kept_node_pixels;
        kept_nodes.push_back(graph.nodes.front());
        kept_node_pixels.emplace_back();
        for (int node = 1; node < static_cast<int>(node_component.size());
             ++node) {
            if (node_component[node] != keep_component) {
                continue;
            }
            node_remap[node] = static_cast<int>(kept_nodes.size());
            kept_nodes.push_back(graph.nodes[static_cast<std::size_t>(node)]);
            if (node < static_cast<int>(node_pixels.size())) {
                kept_node_pixels.push_back(
                    node_pixels[static_cast<std::size_t>(node)]);
            } else {
                kept_node_pixels.emplace_back();
            }
        }

        std::vector<GraphEdge> kept_edges;
        kept_edges.reserve(graph.edges.size());
        for (GraphEdge edge : graph.edges) {
            if (edge.a <= 0 || edge.b <= 0 ||
                edge.a >= static_cast<int>(node_remap.size()) ||
                edge.b >= static_cast<int>(node_remap.size())) {
                continue;
            }
            const int a = node_remap[static_cast<std::size_t>(edge.a)];
            const int b = node_remap[static_cast<std::size_t>(edge.b)];
            if (a <= 0 || b <= 0) {
                continue;
            }
            edge.a = a;
            edge.b = b;
            kept_edges.push_back(std::move(edge));
        }

        graph.pruned_graph_components = component_count - 1;
        graph.pruned_graph_nodes =
            static_cast<int>(graph.nodes.size()) -
            static_cast<int>(kept_nodes.size());
        graph.pruned_graph_edges =
            static_cast<int>(graph.edges.size()) -
            static_cast<int>(kept_edges.size());
        graph.nodes = std::move(kept_nodes);
        graph.edges = std::move(kept_edges);
        node_pixels = std::move(kept_node_pixels);
    };

    prune_to_largest_graph_component();
    graph.adjacent_component_contacts =
        find_raster_component_contact().contacts;

    graph.node_mask = cv::Mat::zeros(skel.size(), CV_8U);
    for (std::size_t node = 1; node < node_pixels.size(); ++node) {
        for (const cv::Point pixel : node_pixels[node]) {
            if (in_bounds(pixel)) {
                graph.node_mask.at<std::uint8_t>(pixel.y, pixel.x) = 255;
            }
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
    graph.node_pixel_groups = std::move(node_pixels);

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

std::vector<int> graph_node_component_ids(const SkeletonGraph& graph) {
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

    for (const GraphEdge& edge : graph.edges) {
        if (valid_node(edge.a) && valid_node(edge.b) && edge.a != edge.b) {
            sets.unite(edge.a, edge.b);
        }
    }

    std::vector<int> root_to_component(static_cast<std::size_t>(node_count), -1);
    std::vector<int> node_component(static_cast<std::size_t>(node_count), -1);
    int next_component = 0;
    for (int node = 1; node < node_count; ++node) {
        const int root = sets.find(node);
        int& component = root_to_component[static_cast<std::size_t>(root)];
        if (component < 0) {
            component = next_component++;
        }
        node_component[static_cast<std::size_t>(node)] = component;
    }
    return node_component;
}

struct GraphRasterAdjacencyStats {
    int contacts = 0;
    cv::Point a{-1, -1};
    cv::Point b{-1, -1};
    int component_a = -1;
    int component_b = -1;
};

GraphRasterAdjacencyStats graph_raster_adjacency_stats(
    const SkeletonGraph& graph) {
    GraphRasterAdjacencyStats stats;
    if (graph.node_mask.empty()) {
        return stats;
    }

    const cv::Size size = graph.node_mask.size();
    const std::vector<int> node_component = graph_node_component_ids(graph);
    const int node_count = static_cast<int>(node_component.size());
    const auto in_bounds = [size](const cv::Point pixel) {
        return pixel.x >= 0 && pixel.x < size.width && pixel.y >= 0 &&
               pixel.y < size.height;
    };
    const auto component_for_node = [&](int node) {
        if (node <= 0 || node >= node_count) {
            return -1;
        }
        return node_component[static_cast<std::size_t>(node)];
    };

    cv::Mat pixel_component(size, CV_32S, cv::Scalar(-1));
    const auto set_pixel_component = [&](const cv::Point pixel,
                                         int component) {
        if (!in_bounds(pixel) || component < 0) {
            return;
        }
        int& dst = pixel_component.at<int>(pixel.y, pixel.x);
        if (dst < 0) {
            dst = component;
        }
    };

    const std::size_t node_group_count =
        std::min(graph.node_pixel_groups.size(), node_component.size());
    for (std::size_t node = 1; node < node_group_count; ++node) {
        const int component = node_component[node];
        for (const cv::Point pixel : graph.node_pixel_groups[node]) {
            set_pixel_component(pixel, component);
        }
    }
    for (const GraphEdge& edge : graph.edges) {
        int component = component_for_node(edge.a);
        if (component < 0) {
            component = component_for_node(edge.b);
        }
        for (const cv::Point pixel : edge.pixels) {
            set_pixel_component(pixel, component);
        }
    }

    const std::array<cv::Point, 4> forward_dirs = {
        {{1, 0}, {-1, 1}, {0, 1}, {1, 1}}};
    for (int y = 0; y < size.height; ++y) {
        const int* component_row = pixel_component.ptr<int>(y);
        for (int x = 0; x < size.width; ++x) {
            const int component = component_row[x];
            if (component < 0) {
                continue;
            }
            const cv::Point pixel(x, y);
            for (const cv::Point dir : forward_dirs) {
                const cv::Point neighbor = pixel + dir;
                if (!in_bounds(neighbor)) {
                    continue;
                }
                const int other =
                    pixel_component.at<int>(neighbor.y, neighbor.x);
                if (other < 0 || other == component) {
                    continue;
                }
                ++stats.contacts;
                if (stats.a.x < 0) {
                    stats.a = pixel;
                    stats.b = neighbor;
                    stats.component_a = component;
                    stats.component_b = other;
                }
            }
        }
    }
    return stats;
}

cv::Mat render_graph_component_colors(const SkeletonGraph& graph, cv::Size size) {
    cv::Mat out(size, CV_8UC3, cv::Scalar(0, 0, 0));
    const std::vector<int> node_component = graph_node_component_ids(graph);
    const int node_count = static_cast<int>(node_component.size());
    const auto component_for_node = [&](int node) {
        if (node <= 0 || node >= node_count) {
            return -1;
        }
        return node_component[static_cast<std::size_t>(node)];
    };

    for (const GraphEdge& edge : graph.edges) {
        int component = component_for_node(edge.a);
        if (component < 0) {
            component = component_for_node(edge.b);
        }
        if (component < 0) {
            continue;
        }
        draw_graph_edge(out, edge, deterministic_edge_color(component));
    }

    const std::size_t node_group_count =
        std::min(graph.node_pixel_groups.size(), node_component.size());
    for (std::size_t node = 1; node < node_group_count; ++node) {
        const int component = node_component[node];
        if (component < 0) {
            continue;
        }
        const cv::Scalar color = deterministic_edge_color(component);
        const cv::Vec3b pixel_color(
            static_cast<std::uint8_t>(color[0]),
            static_cast<std::uint8_t>(color[1]),
            static_cast<std::uint8_t>(color[2]));
        for (const cv::Point pixel : graph.node_pixel_groups[node]) {
            if (pixel.x < 0 || pixel.x >= out.cols || pixel.y < 0 ||
                pixel.y >= out.rows) {
                continue;
            }
            out.at<cv::Vec3b>(pixel.y, pixel.x) = pixel_color;
        }
    }

    for (std::size_t group = 0;
         group < graph.pruned_graph_component_pixel_groups.size(); ++group) {
        const cv::Scalar color =
            deterministic_edge_color(1000 + static_cast<int>(group));
        const cv::Vec3b pixel_color(
            static_cast<std::uint8_t>(color[0]),
            static_cast<std::uint8_t>(color[1]),
            static_cast<std::uint8_t>(color[2]));
        for (const cv::Point pixel :
             graph.pruned_graph_component_pixel_groups[group]) {
            if (pixel.x < 0 || pixel.x >= out.cols || pixel.y < 0 ||
                pixel.y >= out.rows) {
                continue;
            }
            out.at<cv::Vec3b>(pixel.y, pixel.x) = pixel_color;
        }
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
    stats.adjacent_component_repairs = graph.adjacent_component_repairs;
    const GraphRasterAdjacencyStats raster_adjacency =
        graph_raster_adjacency_stats(graph);
    stats.adjacent_component_contacts = raster_adjacency.contacts;
    stats.adjacent_component_a = raster_adjacency.a;
    stats.adjacent_component_b = raster_adjacency.b;
    stats.adjacent_component_a_id = raster_adjacency.component_a;
    stats.adjacent_component_b_id = raster_adjacency.component_b;
    stats.pruned_graph_components = graph.pruned_graph_components;
    stats.pruned_graph_nodes = graph.pruned_graph_nodes;
    stats.pruned_graph_edges = graph.pruned_graph_edges;

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

std::string graph_connectivity_error(const GraphConnectivityStats& stats) {
    if (stats.adjacent_component_contacts > 0) {
        std::ostringstream out;
        out << "graph extraction left raster-adjacent disconnected components:"
            << " contacts=" << stats.adjacent_component_contacts
            << " first=(" << stats.adjacent_component_a.x << ","
            << stats.adjacent_component_a.y << ")-("
            << stats.adjacent_component_b.x << ","
            << stats.adjacent_component_b.y << ")"
            << " components=" << stats.adjacent_component_a_id << ","
            << stats.adjacent_component_b_id
            << " repairs=" << stats.adjacent_component_repairs;
        return out.str();
    }
    if (stats.valid_nodes > 0 && stats.components.size() == 1) {
        return {};
    }

    std::ostringstream out;
    out << "extracted graph is disconnected: components="
        << stats.components.size() << " valid_nodes=" << stats.valid_nodes
        << " valid_edges=" << stats.valid_edges
        << " skeleton_pixels=" << stats.skeleton_pixels;
    const int components_to_report =
        std::min<int>(5, stats.components.size());
    for (int i = 0; i < components_to_report; ++i) {
        const GraphComponentStats& component = stats.components[i];
        out << " component_" << component.id << "_nodes=" << component.nodes
            << " component_" << component.id << "_edges=" << component.edges;
    }
    return out.str();
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
    out << "adjacent_component_repairs: "
        << stats.adjacent_component_repairs << "\n";
    out << "adjacent_component_contacts: "
        << stats.adjacent_component_contacts << "\n";
    out << "pruned_graph_components: "
        << stats.pruned_graph_components << "\n";
    out << "pruned_graph_nodes: " << stats.pruned_graph_nodes << "\n";
    out << "pruned_graph_edges: " << stats.pruned_graph_edges << "\n";
    if (stats.adjacent_component_contacts > 0) {
        out << "adjacent_component_first: " << stats.adjacent_component_a.x
            << "," << stats.adjacent_component_a.y << " "
            << stats.adjacent_component_b.x << ","
            << stats.adjacent_component_b.y << " components="
            << stats.adjacent_component_a_id << ","
            << stats.adjacent_component_b_id << "\n";
    }
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

float max_graph_edge_capacity(const SkeletonGraph& graph) {
    float max_capacity = 0.0f;
    for (const GraphEdge& edge : graph.edges) {
        if (std::isfinite(edge.capacity)) {
            max_capacity = std::max(max_capacity, edge.capacity);
        }
    }
    return max_capacity;
}

cv::Mat render_graph_capacity_normalized_u8(const SkeletonGraph& graph,
                                            cv::Size size) {
    cv::Mat out(size, CV_8UC3, cv::Scalar(0, 0, 0));
    const float max_capacity = max_graph_edge_capacity(graph);
    const float denom = max_capacity > 0.0f ? max_capacity : 1.0f;
    for (const GraphEdge& edge : graph.edges) {
        const float normalized =
            std::clamp(edge.capacity / denom, 0.0f, 1.0f);
        const auto value = static_cast<std::uint8_t>(
            std::lround(static_cast<double>(normalized) * 255.0));
        draw_graph_edge(out, edge, cv::Scalar(value, value, value));
    }
    return out;
}

cv::Mat render_graph_capacity_normalized_float(const SkeletonGraph& graph,
                                               cv::Size size) {
    cv::Mat out(size, CV_32FC3, cv::Scalar(0, 0, 0));
    const float max_capacity = max_graph_edge_capacity(graph);
    const float denom = max_capacity > 0.0f ? max_capacity : 1.0f;
    for (const GraphEdge& edge : graph.edges) {
        const float normalized =
            std::clamp(edge.capacity / denom, 0.0f, 1.0f);
        draw_graph_edge(out, edge,
                        cv::Scalar(normalized, normalized, normalized));
    }
    return out;
}

struct GraphEdgeMaps {
    cv::Mat edge_mask;
    cv::Mat edge_index;
};

GraphEdgeMaps build_graph_edge_maps(const SkeletonGraph& graph,
                                    cv::Size size) {
    GraphEdgeMaps maps;
    maps.edge_mask = cv::Mat(size, CV_8U, cv::Scalar(0));
    maps.edge_index = cv::Mat(size, CV_32S, cv::Scalar(-1));
    for (int edge_index = 0; edge_index < static_cast<int>(graph.edges.size());
         ++edge_index) {
        const GraphEdge& edge = graph.edges[edge_index];
        for (const cv::Point pixel : edge.pixels) {
            if (pixel.x < 0 || pixel.x >= maps.edge_mask.cols || pixel.y < 0 ||
                pixel.y >= maps.edge_mask.rows) {
                continue;
            }
            maps.edge_mask.at<std::uint8_t>(pixel.y, pixel.x) = 255;
            maps.edge_index.at<int>(pixel.y, pixel.x) = edge_index;
        }
    }
    return maps;
}

struct IslandObstacleDebugIsland {
    int label = 0;
    double score = 1.0;
    std::vector<int> loop_edges;
    std::vector<cv::Point> pixels;
};

struct IslandObstacleDebugResult {
    cv::Mat label_factor;
    int islands = 0;
    int loop_sampled_islands = 0;
    std::vector<std::string> score_logs;
    std::vector<IslandObstacleDebugIsland> scored_islands;
};

IslandObstacleDebugResult render_island_obstacle_factors(
    const cv::Mat& white_domain,
    const SkeletonGraph& graph,
    const cv::Mat& graph_edge_index) {
    CV_Assert(white_domain.type() == CV_8U);
    CV_Assert(graph_edge_index.empty() ||
              graph_edge_index.type() == CV_32S);

    IslandObstacleDebugResult result;
    result.label_factor =
        cv::Mat(white_domain.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    if (white_domain.empty()) {
        return result;
    }

    cv::Mat obstacle_mask;
    cv::compare(white_domain, 0, obstacle_mask, cv::CMP_EQ);

    cv::Mat obstacle_labels;
    cv::Mat obstacle_stats;
    cv::Mat obstacle_centroids;
    const int obstacle_count = cv::connectedComponentsWithStats(
        obstacle_mask, obstacle_labels, obstacle_stats, obstacle_centroids, 8,
        CV_32S);
    if (obstacle_count <= 1) {
        return result;
    }

    cv::Mat graph_mask = cv::Mat::zeros(white_domain.size(), CV_8U);
    if (!graph.edge_mask.empty()) {
        cv::bitwise_or(graph_mask, graph.edge_mask, graph_mask);
    }
    if (!graph.node_mask.empty()) {
        cv::bitwise_or(graph_mask, graph.node_mask, graph_mask);
    }

    const auto factor_text = [](int label, double factor) {
        std::ostringstream out;
        out << label << ":" << std::fixed << std::setprecision(2) << factor;
        return out.str();
    };
    const auto in_image = [&](const cv::Point pixel) {
        return pixel.x >= 0 && pixel.x < white_domain.cols && pixel.y >= 0 &&
               pixel.y < white_domain.rows;
    };
    constexpr float kDtEpsilon = 1.0e-4f;

    for (int label = 1; label < obstacle_count; ++label) {
        const int area = obstacle_stats.at<int>(label, cv::CC_STAT_AREA);
        if (area <= 0) {
            continue;
        }
        const int left = obstacle_stats.at<int>(label, cv::CC_STAT_LEFT);
        const int top = obstacle_stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = obstacle_stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = obstacle_stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const bool touches_border =
            left <= 0 || top <= 0 || left + width >= white_domain.cols ||
            top + height >= white_domain.rows;
        if (touches_border) {
            continue;
        }

        ++result.islands;
        const int longest = std::max(width, height);
        const int padding = std::clamp(longest + 16, 16, 128);
        const int roi_left = std::max(0, left - padding);
        const int roi_top = std::max(0, top - padding);
        const int roi_right = std::min(white_domain.cols, left + width + padding);
        const int roi_bottom =
            std::min(white_domain.rows, top + height + padding);
        const cv::Rect roi(roi_left, roi_top, roi_right - roi_left,
                           roi_bottom - roi_top);
        if (roi.empty()) {
            continue;
        }

        cv::Mat patch_domain = white_domain(roi).clone();
        cv::Mat nearest_source_label;
        cv::Mat nearest_source_dt;
        cv::distanceTransform(patch_domain, nearest_source_dt,
                              nearest_source_label,
                              cv::DIST_L2, cv::DIST_MASK_5,
                              cv::DIST_LABEL_PIXEL);
        const std::vector<cv::Point> source_points =
            label_source_points(patch_domain, nearest_source_label);
        std::vector<char> source_is_island(source_points.size(), 0);
        for (int source_label = 1;
             source_label < static_cast<int>(source_points.size());
             ++source_label) {
            const cv::Point local = source_points[source_label];
            const cv::Point global(local.x + roi.x, local.y + roi.y);
            if (!in_image(global)) {
                continue;
            }
            source_is_island[static_cast<std::size_t>(source_label)] =
                obstacle_labels.at<int>(global.y, global.x) == label ? 1 : 0;
        }

        std::vector<cv::Point> island_pixels;
        island_pixels.reserve(static_cast<std::size_t>(area));
        cv::Mat island_source_domain(roi.size(), CV_8U, cv::Scalar(255));
        for (int y = top; y < top + height; ++y) {
            for (int x = left; x < left + width; ++x) {
                if (obstacle_labels.at<int>(y, x) != label) {
                    continue;
                }
                const cv::Point global(x, y);
                const cv::Point local(x - roi.x, y - roi.y);
                island_source_domain.at<std::uint8_t>(local.y, local.x) = 0;
                island_pixels.push_back(global);
            }
        }

        std::vector<cv::Point> loop_pixels;
        for (int y = 0; y < roi.height; ++y) {
            for (int x = 0; x < roi.width; ++x) {
                const cv::Point global(x + roi.x, y + roi.y);
                if (graph_mask.at<std::uint8_t>(global.y, global.x) == 0) {
                    continue;
                }
                bool touches_island_voronoi = false;
                for (int dy = -1; dy <= 1 && !touches_island_voronoi; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = x + dx;
                        const int ny = y + dy;
                        if (nx < 0 || nx >= roi.width || ny < 0 ||
                            ny >= roi.height) {
                            continue;
                        }
                        const int nearest =
                            nearest_source_label.at<int>(ny, nx);
                        if (nearest > 0 &&
                            nearest <
                                static_cast<int>(source_is_island.size()) &&
                            source_is_island
                                    [static_cast<std::size_t>(nearest)] != 0) {
                            touches_island_voronoi = true;
                            break;
                        }
                    }
                }
                if (!touches_island_voronoi) {
                    continue;
                }
                loop_pixels.push_back(global);
            }
        }

        cv::Point representative_pixel = island_pixels.empty()
                                             ? cv::Point(left, top)
                                             : island_pixels.front();
        float representative_loop_distance = 0.0f;
        cv::Mat loop_source_domain(roi.size(), CV_8U, cv::Scalar(255));
        for (const cv::Point pixel : loop_pixels) {
            loop_source_domain.at<std::uint8_t>(pixel.y - roi.y,
                                                pixel.x - roi.x) = 0;
        }
        if (!loop_pixels.empty() && !island_pixels.empty()) {
            cv::Mat loop_dt;
            cv::distanceTransform(loop_source_domain, loop_dt, cv::DIST_L2,
                                  cv::DIST_MASK_5);
            float best_loop_distance = -1.0f;
            for (const cv::Point pixel : island_pixels) {
                const cv::Point local(pixel.x - roi.x, pixel.y - roi.y);
                const float value = loop_dt.at<float>(local.y, local.x);
                if (value > best_loop_distance ||
                    (std::abs(value - best_loop_distance) <= kDtEpsilon &&
                     (pixel.y < representative_pixel.y ||
                      (pixel.y == representative_pixel.y &&
                       pixel.x < representative_pixel.x)))) {
                    best_loop_distance = value;
                    representative_pixel = pixel;
                }
            }
            representative_loop_distance = std::max(best_loop_distance, 0.0f);
        }

        cv::Mat actual_island_dt;
        cv::distanceTransform(island_source_domain, actual_island_dt,
                              cv::DIST_L2, cv::DIST_MASK_5);

        cv::Mat point_source_domain(roi.size(), CV_8U, cv::Scalar(255));
        const cv::Point representative_local(representative_pixel.x - roi.x,
                                             representative_pixel.y - roi.y);
        if (representative_local.x >= 0 &&
            representative_local.x < point_source_domain.cols &&
            representative_local.y >= 0 &&
            representative_local.y < point_source_domain.rows) {
            point_source_domain.at<std::uint8_t>(representative_local.y,
                                                 representative_local.x) = 0;
        }
        cv::Mat point_dt;
        cv::distanceTransform(point_source_domain, point_dt, cv::DIST_L2,
                              cv::DIST_MASK_5);

        double score = 1.0;
        bool has_loop_score = false;
        int scored_loop_pixels = 0;
        double ratio_sum = 0.0;
        double ratio_min = 1.0;
        double ratio_max = 0.0;
        cv::Point worst_loop_pixel(-1, -1);
        float worst_island_distance = 0.0f;
        float worst_single_point_distance = 0.0f;
        for (const cv::Point pixel : loop_pixels) {
            const cv::Point local(pixel.x - roi.x, pixel.y - roi.y);
            const float single_point_distance =
                point_dt.at<float>(local.y, local.x);
            if (single_point_distance <= kDtEpsilon) {
                continue;
            }
            const float island_distance =
                actual_island_dt.at<float>(local.y, local.x);
            const double point_score = std::clamp(
                static_cast<double>(island_distance) /
                    static_cast<double>(single_point_distance),
                0.0, 1.0);
            ++scored_loop_pixels;
            ratio_sum += point_score;
            ratio_min = std::min(ratio_min, point_score);
            ratio_max = std::max(ratio_max, point_score);
            if (point_score <= score) {
                worst_loop_pixel = pixel;
                worst_island_distance = island_distance;
                worst_single_point_distance = single_point_distance;
            }
            score = std::min(score, point_score);
            has_loop_score = true;
        }
        if (has_loop_score) {
            ++result.loop_sampled_islands;
        }
        std::vector<int> loop_edge_indices;
        if (!graph_edge_index.empty()) {
            std::vector<char> seen_edges(graph.edges.size(), 0);
            const auto add_loop_edge = [&](const cv::Point pixel) {
                if (pixel.x < 0 || pixel.x >= graph_edge_index.cols ||
                    pixel.y < 0 || pixel.y >= graph_edge_index.rows) {
                    return;
                }
                const int edge_index =
                    graph_edge_index.at<int>(pixel.y, pixel.x);
                if (edge_index < 0 ||
                    edge_index >= static_cast<int>(graph.edges.size()) ||
                    seen_edges[static_cast<std::size_t>(edge_index)] != 0) {
                    return;
                }
                seen_edges[static_cast<std::size_t>(edge_index)] = 1;
                loop_edge_indices.push_back(edge_index);
            };
            for (const cv::Point pixel : loop_pixels) {
                add_loop_edge(pixel);
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        add_loop_edge(cv::Point(pixel.x + dx, pixel.y + dy));
                    }
                }
            }
        }
        const std::size_t loop_edge_count = loop_edge_indices.size();
        if (has_loop_score) {
            result.scored_islands.push_back(
                {label, score, std::move(loop_edge_indices), island_pixels});
        }
        {
            std::ostringstream log;
            log << "  island " << label << ":"
                << " score=" << std::fixed << std::setprecision(4) << score
                << " area=" << area << " bbox=" << left << "," << top
                << "," << width << "x" << height
                << " loop_pixels=" << loop_pixels.size()
                << " scored_loop_pixels=" << scored_loop_pixels
                << " loop_edges=" << loop_edge_count
                << " rep=" << representative_pixel.x << ","
                << representative_pixel.y
                << " rep_loop_dt=" << std::setprecision(3)
                << representative_loop_distance;
            if (has_loop_score) {
                const double ratio_mean =
                    scored_loop_pixels > 0
                        ? ratio_sum / static_cast<double>(scored_loop_pixels)
                        : 0.0;
                log << " worst_loop=" << worst_loop_pixel.x << ","
                    << worst_loop_pixel.y
                    << " island_dt=" << worst_island_distance
                    << " point_dt=" << worst_single_point_distance
                    << " ratio_min=" << std::setprecision(4) << ratio_min
                    << " ratio_mean=" << ratio_mean
                    << " ratio_max=" << ratio_max;
            } else {
                log << " no_loop_score";
            }
            result.score_logs.push_back(log.str());
        }

        const cv::Scalar base_color = deterministic_edge_color(label);
        const double scale = 0.35 + 0.65 * score;
        const cv::Vec3b island_color(
            static_cast<std::uint8_t>(
                std::clamp(base_color[0] * scale, 0.0, 255.0)),
            static_cast<std::uint8_t>(
                std::clamp(base_color[1] * scale, 0.0, 255.0)),
            static_cast<std::uint8_t>(
                std::clamp(base_color[2] * scale, 0.0, 255.0)));

        for (const cv::Point pixel : island_pixels) {
            result.label_factor.at<cv::Vec3b>(pixel.y, pixel.x) = island_color;
        }
        if (in_image(representative_pixel)) {
            cv::circle(result.label_factor, representative_pixel, 1,
                       cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_8);
        }

        draw_debug_label(result.label_factor, representative_pixel,
                         factor_text(label, score), cv::Scalar(255, 255, 255));
    }

    return result;
}

struct IslandFlowPropagationResult {
    std::vector<float> propagated_edge_flow;
    std::vector<float> bonus_edge_flow;
    std::vector<float> edge_passability;
    int linked_islands = 0;
    int transition_links = 0;
    int boosted_edges = 0;
};

IslandFlowPropagationResult propagate_flow_across_island_links(
    const SkeletonGraph& graph,
    const std::vector<float>& edge_flow,
    const IslandObstacleDebugResult& island_debug) {
    CV_Assert(edge_flow.size() == graph.edges.size());

    struct Transition {
        int to = -1;
        float passability = 0.0f;
    };
    struct QueueItem {
        float flow = 0.0f;
        int edge = -1;
        bool operator<(const QueueItem& other) const {
            return flow < other.flow;
        }
    };

    IslandFlowPropagationResult result;
    result.propagated_edge_flow = edge_flow;
    result.bonus_edge_flow.assign(edge_flow.size(), 0.0f);
    result.edge_passability.assign(edge_flow.size(), 0.0f);

    std::vector<std::vector<Transition>> transitions(edge_flow.size());
    for (const IslandObstacleDebugIsland& island :
         island_debug.scored_islands) {
        const float passability =
            std::clamp(static_cast<float>(island.score), 0.0f, 1.0f);
        if (passability <= 1.0e-4f || island.loop_edges.size() < 2) {
            continue;
        }
        ++result.linked_islands;
        for (const int edge_index : island.loop_edges) {
            if (edge_index < 0 ||
                edge_index >= static_cast<int>(edge_flow.size())) {
                continue;
            }
            float& current =
                result.edge_passability[static_cast<std::size_t>(edge_index)];
            current = std::max(current, passability);
        }
        for (std::size_t i = 0; i < island.loop_edges.size(); ++i) {
            const int a = island.loop_edges[i];
            if (a < 0 || a >= static_cast<int>(edge_flow.size())) {
                continue;
            }
            for (std::size_t j = i + 1; j < island.loop_edges.size(); ++j) {
                const int b = island.loop_edges[j];
                if (b < 0 || b >= static_cast<int>(edge_flow.size()) ||
                    a == b) {
                    continue;
                }
                transitions[static_cast<std::size_t>(a)].push_back(
                    {b, passability});
                transitions[static_cast<std::size_t>(b)].push_back(
                    {a, passability});
                result.transition_links += 2;
            }
        }
    }

    std::priority_queue<QueueItem> queue;
    for (int edge_index = 0; edge_index < static_cast<int>(edge_flow.size());
         ++edge_index) {
        const float value = edge_flow[static_cast<std::size_t>(edge_index)];
        if (std::isfinite(value) && value > 0.0f) {
            queue.push({value, edge_index});
        }
    }

    constexpr float kFlowEpsilon = 1.0e-4f;
    while (!queue.empty()) {
        const QueueItem item = queue.top();
        queue.pop();
        float& current =
            result.propagated_edge_flow[static_cast<std::size_t>(item.edge)];
        if (item.flow + kFlowEpsilon < current) {
            continue;
        }
        for (const Transition transition :
             transitions[static_cast<std::size_t>(item.edge)]) {
            const float candidate = item.flow * transition.passability;
            float& target = result.propagated_edge_flow
                [static_cast<std::size_t>(transition.to)];
            if (candidate <= target + kFlowEpsilon) {
                continue;
            }
            target = candidate;
            queue.push({candidate, transition.to});
        }
    }

    for (int edge_index = 0; edge_index < static_cast<int>(edge_flow.size());
         ++edge_index) {
        const float raw = edge_flow[static_cast<std::size_t>(edge_index)];
        const float propagated =
            result.propagated_edge_flow[static_cast<std::size_t>(edge_index)];
        const float bonus = std::max(0.0f, propagated - raw);
        result.bonus_edge_flow[static_cast<std::size_t>(edge_index)] = bonus;
        if (bonus > kFlowEpsilon) {
            ++result.boosted_edges;
        }
    }

    return result;
}

cv::Mat render_graph_edge_values(const SkeletonGraph& graph,
                                 cv::Size size,
                                 const std::vector<float>& values,
                                 float scale = 1.0f) {
    CV_Assert(values.size() == graph.edges.size());
    cv::Mat out(size, CV_32FC3, cv::Scalar(0, 0, 0));
    for (int edge_index = 0; edge_index < static_cast<int>(graph.edges.size());
         ++edge_index) {
        const float value =
            values[static_cast<std::size_t>(edge_index)] * scale;
        if (value <= 0.0f) {
            continue;
        }
        draw_graph_edge(out, graph.edges[edge_index],
                        cv::Scalar(value, value, value));
    }
    return out;
}

struct IslandRemovalResult {
    cv::Mat white_domain;
    cv::Mat removed_mask;
    int removed_islands = 0;
    int removed_pixels = 0;
};

IslandRemovalResult remove_high_score_islands(
    const cv::Mat& white_domain,
    const IslandObstacleDebugResult& island_debug) {
    CV_Assert(white_domain.type() == CV_8U);

    IslandRemovalResult result;
    result.white_domain = white_domain.clone();
    result.removed_mask = cv::Mat::zeros(white_domain.size(), CV_8U);
    for (const IslandObstacleDebugIsland& island :
         island_debug.scored_islands) {
        if (island.score <= kIslandRemovalScoreThreshold) {
            continue;
        }
        ++result.removed_islands;
        for (const cv::Point pixel : island.pixels) {
            if (pixel.x < 0 || pixel.x >= result.white_domain.cols ||
                pixel.y < 0 || pixel.y >= result.white_domain.rows) {
                continue;
            }
            if (result.white_domain.at<std::uint8_t>(pixel.y, pixel.x) != 0) {
                continue;
            }
            result.white_domain.at<std::uint8_t>(pixel.y, pixel.x) = 255;
            result.removed_mask.at<std::uint8_t>(pixel.y, pixel.x) = 255;
            ++result.removed_pixels;
        }
    }
    return result;
}

cv::Mat nearest_graph_flow_projection(const cv::Mat& white_domain,
                                      const cv::Mat& graph_mask,
                                      const cv::Mat& graph_pixel_flow) {
    CV_Assert(white_domain.type() == CV_8U);
    CV_Assert(graph_mask.type() == CV_8U);
    CV_Assert(graph_pixel_flow.type() == CV_32F);
    CV_Assert(white_domain.size() == graph_mask.size());
    CV_Assert(white_domain.size() == graph_pixel_flow.size());

    cv::Mat out(white_domain.size(), CV_32F, cv::Scalar(0));
    if (cv::countNonZero(graph_mask) == 0) {
        return out;
    }

    cv::Mat graph_source_mask(white_domain.size(), CV_8U, cv::Scalar(255));
    graph_source_mask.setTo(0, graph_mask);

    cv::Mat nearest_graph_distance;
    cv::Mat nearest_graph_label;
    cv::distanceTransform(graph_source_mask, nearest_graph_distance,
                          nearest_graph_label, cv::DIST_L2, cv::DIST_MASK_5,
                          cv::DIST_LABEL_PIXEL);
    const std::vector<cv::Point> graph_source_points =
        label_source_points(graph_source_mask, nearest_graph_label);

    cv::parallel_for_(cv::Range(0, white_domain.rows),
                      [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const std::uint8_t* domain_row = white_domain.ptr<std::uint8_t>(y);
            const int* label_row = nearest_graph_label.ptr<int>(y);
            float* out_row = out.ptr<float>(y);
            for (int x = 0; x < white_domain.cols; ++x) {
                if (domain_row[x] == 0) {
                    continue;
                }
                const int label = label_row[x];
                if (label <= 0 ||
                    label >= static_cast<int>(graph_source_points.size())) {
                    continue;
                }
                const cv::Point graph_pixel =
                    graph_source_points[static_cast<std::size_t>(label)];
                if (graph_pixel.x < 0) {
                    continue;
                }
                out_row[x] =
                    graph_pixel_flow.at<float>(graph_pixel.y, graph_pixel.x);
            }
        }
    });
    return out;
}

cv::Mat greedy_increasing_flow_ascent(const cv::Mat& white_domain,
                                      const cv::Mat& source_flow) {
    CV_Assert(white_domain.type() == CV_8U);
    CV_Assert(source_flow.type() == CV_32F);
    CV_Assert(white_domain.size() == source_flow.size());

    cv::Mat out(white_domain.size(), CV_32F, cv::Scalar(0));
    cv::Mat cached(white_domain.size(), CV_32F, cv::Scalar(-1));
    constexpr float kFlowEpsilon = 1.0e-4f;
    const std::array<cv::Point, 8> kDirs = {
        cv::Point(-1, -1), cv::Point(0, -1), cv::Point(1, -1),
        cv::Point(-1, 0),                    cv::Point(1, 0),
        cv::Point(-1, 1),  cv::Point(0, 1),  cv::Point(1, 1)};

    const auto in_white_domain = [&](const cv::Point pixel) {
        return pixel.x >= 0 && pixel.x < white_domain.cols && pixel.y >= 0 &&
               pixel.y < white_domain.rows &&
               white_domain.at<std::uint8_t>(pixel.y, pixel.x) != 0;
    };

    std::vector<cv::Point> path;
    path.reserve(256);
    for (int y = 0; y < white_domain.rows; ++y) {
        for (int x = 0; x < white_domain.cols; ++x) {
            const cv::Point start(x, y);
            if (!in_white_domain(start)) {
                continue;
            }
            if (cached.at<float>(y, x) >= 0.0f) {
                out.at<float>(y, x) = cached.at<float>(y, x);
                continue;
            }

            path.clear();
            cv::Point current = start;
            float terminal_flow = 0.0f;
            while (in_white_domain(current)) {
                const float cached_flow =
                    cached.at<float>(current.y, current.x);
                if (cached_flow >= 0.0f) {
                    terminal_flow = cached_flow;
                    break;
                }
                path.push_back(current);

                const float current_flow =
                    source_flow.at<float>(current.y, current.x);
                cv::Point next(-1, -1);
                float best_flow = current_flow;
                for (const cv::Point dir : kDirs) {
                    const cv::Point neighbor = current + dir;
                    if (!in_white_domain(neighbor)) {
                        continue;
                    }
                    const float neighbor_flow =
                        source_flow.at<float>(neighbor.y, neighbor.x);
                    if (neighbor_flow <= best_flow + kFlowEpsilon) {
                        continue;
                    }
                    best_flow = neighbor_flow;
                    next = neighbor;
                }
                if (next.x < 0) {
                    terminal_flow = std::max(0.0f, current_flow);
                    break;
                }
                current = next;
            }

            for (const cv::Point pixel : path) {
                cached.at<float>(pixel.y, pixel.x) = terminal_flow;
                out.at<float>(pixel.y, pixel.x) = terminal_flow;
            }
        }
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
    cv::Mat tree_dense_flow_no_backtrack;
    cv::Mat nearest_graph_flow;
    cv::Mat tree_dense_flow_greedy_ascent;
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
    cv::Mat island_obstacle_factor;
    cv::Mat island_flow_passability;
    cv::Mat island_propagated_edge_flow;
    cv::Mat island_bonus_edge_flow;
    cv::Mat island_tree_dense_flow_no_backtrack;
    cv::Mat island_tree_dense_flow_greedy_ascent;
    cv::Mat flow_gate_regions;
    cv::Mat flow_gate_weight;
    cv::Point source_seed_pixel{-1, -1};
    float source_seed_capacity = 0.0f;
    int source_edges = 0;
    int seeded_nodes = 0;
    int accepted_sources = 0;
    float finite_edge_flow_min = 0.0f;
    float finite_edge_flow_max = 0.0f;
    int finite_edge_flows = 0;
    std::vector<StageTiming> timings;
};

struct SourceEdgeStart {
    cv::Point source{-1, -1};
    int source_index = -1;
    int edge = -1;
    cv::Point seed_pixel{-1, -1};
    float seed_capacity = 0.0f;
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

struct FlowGateRegionBuildResult {
    cv::Mat labels;
    int region_count = 0;
    int source_groups = 0;
    int merged_source_pairs = 0;
    int graph_seed_pixels = 0;
    int unlabeled_white_pixels = 0;
};

cv::Mat bilinear_from_regular_grid_samples(const cv::Mat& source,
                                           int grid_step);

FlowGateRegionBuildResult build_flow_gate_region_labels(
        const cv::Mat& white_domain,
        const cv::Mat& dt,
        const SkeletonGraph& graph,
        const std::vector<SourceEdgeStart>& source_starts);

DenseBacktrackResult compute_dense_backtrack_flow(const cv::Mat& white_domain,
                                                  const cv::Mat& dt,
                                                  const cv::Mat& graph_pixel_flow,
                                                  const cv::Mat& graph_node_flow,
                                                  const SkeletonGraph& graph,
                                                  const std::vector<double>& node_flow,
                                                  const std::vector<float>& edge_flow,
                                                  int source_seed_node,
                                                  const std::vector<char>& source_edges,
                                                  int grid_step,
                                                  float backtrack_distance,
                                                  bool enable_debug_outputs) {
    CV_Assert(white_domain.type() == CV_8U);
    CV_Assert(dt.type() == CV_32F);
    CV_Assert(graph_pixel_flow.type() == CV_32F);
    CV_Assert(graph_node_flow.type() == CV_32F);
    CV_Assert(node_flow.size() == graph.nodes.size());
    CV_Assert(edge_flow.size() == graph.edges.size());

    DenseBatchScratch& scratch = dense_batch_scratch();
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

    std::vector<float>& routed_flow = scratch.dense_routed_flow;
    std::vector<int>& graph_next_pixel = scratch.dense_graph_next_pixel;
    std::vector<float>& graph_next_distance =
        scratch.dense_graph_next_distance;
    std::vector<float>& graph_seed_flow = scratch.dense_graph_seed_flow;
    std::vector<char>& graph_route_pixel = scratch.dense_graph_route_pixel;
    resize_fill(routed_flow, static_cast<std::size_t>(pixel_count), 0.0f);
    resize_fill(graph_next_pixel, static_cast<std::size_t>(pixel_count), -1);
    resize_fill(graph_next_distance, static_cast<std::size_t>(pixel_count),
                0.0f);
    resize_fill(graph_seed_flow, static_cast<std::size_t>(pixel_count), 0.0f);
    resize_fill(graph_route_pixel, static_cast<std::size_t>(pixel_count),
                static_cast<char>(0));

    {
        const TimingMark timing = start_timing();
        const int node_count = static_cast<int>(graph.nodes.size());
        const auto valid_node = [&](const int node) {
            return node > 0 && node < node_count;
        };
        const auto in_bounds = [&](const cv::Point pixel) {
            return pixel.x >= 0 && pixel.x < cols && pixel.y >= 0 &&
                   pixel.y < rows;
        };

        struct RouteNeighbor {
            int node = -1;
            int edge = -1;
            double distance = 0.0;
        };

        std::vector<std::vector<RouteNeighbor>> route_graph(
            static_cast<std::size_t>(node_count));
        std::vector<std::vector<float>> edge_prefix(
            static_cast<std::size_t>(graph.edges.size()));
        std::vector<float> edge_length(
            static_cast<std::size_t>(graph.edges.size()), 0.0f);
        {
            const TimingMark sub_timing = start_timing();
            for (int edge_index = 0;
                 edge_index < static_cast<int>(graph.edges.size());
                 ++edge_index) {
                const GraphEdge& edge = graph.edges[edge_index];
                std::vector<float>& prefix =
                    edge_prefix[static_cast<std::size_t>(edge_index)];
                prefix.resize(edge.pixels.size(), 0.0f);
                for (std::size_t i = 1; i < edge.pixels.size(); ++i) {
                    prefix[i] = prefix[i - 1] +
                                step_distance(edge.pixels[i - 1],
                                              edge.pixels[i]);
                }
                edge_length[static_cast<std::size_t>(edge_index)] =
                    prefix.empty() ? 0.0f : prefix.back();

                if (!valid_node(edge.a) || !valid_node(edge.b) ||
                    edge.a == edge.b) {
                    continue;
                }
                const double distance =
                    std::max<double>(edge_length[edge_index], 1.0);
                route_graph[edge.a].push_back(
                    {edge.b, edge_index, distance});
                route_graph[edge.b].push_back(
                    {edge.a, edge_index, distance});
            }
            result.timings.push_back(
                finish_timing("dense_graph_route.adjacency", sub_timing));
        }

        std::vector<double> node_route_distance(
            static_cast<std::size_t>(node_count),
            std::numeric_limits<double>::infinity());
        std::vector<int> node_route_prev_edge(
            static_cast<std::size_t>(node_count), -1);
        {
            const TimingMark sub_timing = start_timing();
            struct NodeRouteItem {
                double distance = 0.0;
                int node = -1;
                bool operator<(const NodeRouteItem& other) const {
                    return distance > other.distance;
                }
            };
            std::priority_queue<NodeRouteItem> node_queue;
            std::vector<char> root_nodes(static_cast<std::size_t>(node_count),
                                         0);
            const auto add_root_node = [&](const int node) {
                if (!valid_node(node) ||
                    root_nodes[static_cast<std::size_t>(node)] != 0) {
                    return;
                }
                root_nodes[static_cast<std::size_t>(node)] = 1;
                node_route_distance[static_cast<std::size_t>(node)] = 0.0;
                node_queue.push({0.0, node});
            };
            if (source_edges.size() == graph.edges.size()) {
                for (int edge_index = 0;
                     edge_index < static_cast<int>(graph.edges.size());
                     ++edge_index) {
                    if (source_edges[static_cast<std::size_t>(edge_index)] ==
                        0) {
                        continue;
                    }
                    add_root_node(graph.edges[edge_index].a);
                    add_root_node(graph.edges[edge_index].b);
                }
            }
            if (node_queue.empty()) {
                add_root_node(source_seed_node);
            }

            while (!node_queue.empty()) {
                const NodeRouteItem item = node_queue.top();
                node_queue.pop();
                if (item.distance >
                    node_route_distance[item.node] + kFlowEpsilon) {
                    continue;
                }
                for (const RouteNeighbor neighbor : route_graph[item.node]) {
                    const double candidate =
                        item.distance + neighbor.distance;
                    if (candidate + kFlowEpsilon >=
                        node_route_distance[neighbor.node]) {
                        continue;
                    }
                    node_route_distance[neighbor.node] = candidate;
                    node_route_prev_edge[neighbor.node] = neighbor.edge;
                    node_queue.push({candidate, neighbor.node});
                }
            }
            result.timings.push_back(
                finish_timing("dense_graph_route.dijkstra", sub_timing));
        }

        int graph_root_pixels = 0;
        int graph_routed_pixels = 0;
        int graph_routed_route_pixels = 0;
        {
            const TimingMark sub_timing = start_timing();
            std::vector<char>& root_seen = scratch.dense_graph_root_seen;
            std::vector<char>& edge_route_seen =
                scratch.dense_graph_edge_route_seen;
            resize_fill(root_seen, static_cast<std::size_t>(pixel_count),
                        static_cast<char>(0));
            resize_fill(edge_route_seen, static_cast<std::size_t>(pixel_count),
                        static_cast<char>(0));
            const auto mark_root_pixel = [&](const cv::Point pixel) {
                if (!in_bounds(pixel)) {
                    return;
                }
                const int index = linear_index(pixel);
                if (root_seen[static_cast<std::size_t>(index)] == 0) {
                    root_seen[static_cast<std::size_t>(index)] = 1;
                    ++graph_root_pixels;
                }
            };
            if (source_edges.size() == graph.edges.size()) {
                for (int edge_index = 0;
                     edge_index < static_cast<int>(graph.edges.size());
                     ++edge_index) {
                    if (source_edges[static_cast<std::size_t>(edge_index)] ==
                        0) {
                        continue;
                    }
                    for (const cv::Point pixel : graph.edges[edge_index].pixels) {
                        mark_root_pixel(pixel);
                    }
                }
            }

            const auto node_anchor = [&](const int node) {
                if (!valid_node(node)) {
                    return cv::Point(-1, -1);
                }
                return cv::Point(cvRound(graph.nodes[node].x),
                                 cvRound(graph.nodes[node].y));
            };
            const auto for_node_pixels = [&](const int node,
                                             const auto& callback) {
                if (node > 0 &&
                    node < static_cast<int>(graph.node_pixel_groups.size()) &&
                    !graph.node_pixel_groups[static_cast<std::size_t>(node)].empty()) {
                    for (const cv::Point pixel :
                         graph.node_pixel_groups[static_cast<std::size_t>(node)]) {
                        callback(pixel);
                    }
                    return;
                }
                const cv::Point anchor = node_anchor(node);
                if (anchor.x >= 0) {
                    callback(anchor);
                }
            };
            for_node_pixels(source_seed_node, mark_root_pixel);

            constexpr double kRouteGraphFlowInf = 1.0e12;
            constexpr double kRouteDenseFlowInf = 1.0e9;
            const auto node_value = [&](const int node) {
                if (!valid_node(node) ||
                    node >= static_cast<int>(node_flow.size())) {
                    return 0.0f;
                }
                const cv::Point anchor = node_anchor(node);
                if (node_flow[static_cast<std::size_t>(node)] >=
                        kRouteGraphFlowInf * 0.5 &&
                    in_bounds(anchor)) {
                    return capacity_from_dt(dt.at<float>(anchor.y, anchor.x));
                }
                return static_cast<float>(std::min(
                    kRouteDenseFlowInf,
                    std::max(0.0,
                             node_flow[static_cast<std::size_t>(node)])));
            };
            const auto closest_node_pixel = [&](const int node,
                                                const cv::Point target) {
                cv::Point best = node_anchor(node);
                double best_distance = std::numeric_limits<double>::max();
                for_node_pixels(node, [&](const cv::Point pixel) {
                    const double dx = pixel.x - target.x;
                    const double dy = pixel.y - target.y;
                    const double distance = dx * dx + dy * dy;
                    if (distance < best_distance) {
                        best_distance = distance;
                        best = pixel;
                    }
                });
                return best;
            };
            const auto write_next_pixel = [&](const cv::Point pixel,
                                              const cv::Point next) {
                if (!in_bounds(pixel) || !in_bounds(next) || next == pixel) {
                    return;
                }
                const int index = linear_index(pixel);
                graph_next_pixel[static_cast<std::size_t>(index)] =
                    linear_index(next);
                graph_next_distance[static_cast<std::size_t>(index)] =
                    step_distance(pixel, next);
            };
            const auto edge_endpoint_pixel = [&](const GraphEdge& edge,
                                                 const int node) {
                if (edge.pixels.empty()) {
                    return node_anchor(node);
                }
                if (edge.a == node) {
                    return edge.pixels.front();
                }
                if (edge.b == node) {
                    return edge.pixels.back();
                }
                return closest_node_pixel(node, edge.pixels.front());
            };
            const auto node_next_pixel = [&](const int node) {
                if (!valid_node(node)) {
                    return cv::Point(-1, -1);
                }
                const int edge_index =
                    node_route_prev_edge[static_cast<std::size_t>(node)];
                if (edge_index < 0 ||
                    edge_index >= static_cast<int>(graph.edges.size())) {
                    return cv::Point(-1, -1);
                }
                return edge_endpoint_pixel(graph.edges[edge_index], node);
            };

            for (int edge_index = 0;
                 edge_index < static_cast<int>(graph.edges.size());
                 ++edge_index) {
                const GraphEdge& edge = graph.edges[edge_index];
                if (edge.pixels.empty()) {
                    continue;
                }
                const std::vector<float>& prefix =
                    edge_prefix[static_cast<std::size_t>(edge_index)];
                const float length =
                    edge_length[static_cast<std::size_t>(edge_index)];
                const bool route_a =
                    valid_node(edge.a) &&
                    std::isfinite(
                        node_route_distance[static_cast<std::size_t>(edge.a)]);
                const bool route_b =
                    valid_node(edge.b) &&
                    std::isfinite(
                        node_route_distance[static_cast<std::size_t>(edge.b)]);
                if (!route_a && !route_b) {
                    continue;
                }

                for (std::size_t i = 0; i < edge.pixels.size(); ++i) {
                    const cv::Point pixel = edge.pixels[i];
                    if (!in_bounds(pixel)) {
                        continue;
                    }
                    const double distance_to_a =
                        route_a
                            ? node_route_distance
                                      [static_cast<std::size_t>(edge.a)] +
                                  (prefix.empty() ? 0.0 : prefix[i])
                            : std::numeric_limits<double>::infinity();
                    const double distance_to_b =
                        route_b
                            ? node_route_distance
                                      [static_cast<std::size_t>(edge.b)] +
                                  static_cast<double>(
                                      length -
                                      (prefix.empty() ? 0.0f : prefix[i]))
                            : std::numeric_limits<double>::infinity();
                    const bool toward_a = distance_to_a <= distance_to_b;

                    const int index = linear_index(pixel);
                    if (edge_route_seen[static_cast<std::size_t>(index)] == 0) {
                        edge_route_seen[static_cast<std::size_t>(index)] = 1;
                        ++graph_routed_pixels;
                    }
                    if (graph_route_pixel[static_cast<std::size_t>(index)] ==
                        0) {
                        graph_route_pixel[static_cast<std::size_t>(index)] = 1;
                        ++graph_routed_route_pixels;
                    }
                    if (toward_a) {
                        const cv::Point next =
                            i > 0 ? edge.pixels[i - 1]
                                  : closest_node_pixel(edge.a, pixel);
                        write_next_pixel(pixel, next);
                    } else {
                        const cv::Point next =
                            i + 1 < edge.pixels.size()
                                ? edge.pixels[i + 1]
                                : closest_node_pixel(edge.b, pixel);
                        write_next_pixel(pixel, next);
                    }
                }
            }

            for (int node = 1; node < node_count; ++node) {
                if (!std::isfinite(
                        node_route_distance[static_cast<std::size_t>(node)])) {
                    continue;
                }
                const float value = node_value(node);
                const cv::Point next = node_next_pixel(node);
                for_node_pixels(node, [&](const cv::Point pixel) {
                    if (!in_bounds(pixel)) {
                        return;
                    }
                    const int index = linear_index(pixel);
                    if (graph_route_pixel[static_cast<std::size_t>(index)] ==
                        0) {
                        graph_route_pixel[static_cast<std::size_t>(index)] = 1;
                        ++graph_routed_route_pixels;
                    }
                    if (value > 0.0f) {
                        graph_seed_flow[static_cast<std::size_t>(index)] =
                            value;
                    }
                    write_next_pixel(pixel, next);
                });
            }
            result.timings.push_back(
                finish_timing("dense_graph_route.pixel_project", sub_timing));
        }

        int graph_edge_pixels = graph.unique_edge_pixels;
        int graph_node_pixels = 0;
        {
            const TimingMark sub_timing = start_timing();
            graph_node_pixels = cv::countNonZero(graph_node_flow > 0.0f);
            result.timings.push_back(
                finish_timing("dense_graph_route.stats", sub_timing));
        }
        if (enable_debug_outputs) {
            std::cout << "Dense graph backtrack:\n"
                      << "  graph_edge_pixels: " << graph_edge_pixels << "\n"
                      << "  graph_node_pixels: " << graph_node_pixels << "\n"
                      << "  graph_root_pixels: " << graph_root_pixels << "\n"
                      << "  graph_routed_pixels: " << graph_routed_pixels
                      << "\n"
                      << "  graph_routed_route_pixels: "
                      << graph_routed_route_pixels << "\n"
                      << "  graph_unrouted_pixels: "
                      << (graph_edge_pixels - graph_routed_pixels) << "\n";
        }
        result.timings.push_back(
            finish_timing("dense_backtrack_graph_route", timing));
    }

    {
        const TimingMark timing = start_timing();
        std::vector<int>& seeded_by_row = scratch.dense_seeded_by_row;
        resize_fill(seeded_by_row, static_cast<std::size_t>(rows), 0);
        cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                int row_seeded = 0;
                const std::uint8_t* white_row =
                    white_domain.ptr<std::uint8_t>(y);
                const float* dt_row = dt.ptr<float>(y);
                for (int x = 0; x < cols; ++x) {
                    if (white_row[x] == 0) {
                        continue;
                    }
                    const int index = y * cols + x;
                    const float seed = std::max(
                        graph_seed_flow[static_cast<std::size_t>(index)],
                        graph_pixel_flow.at<float>(y, x));
                    if (seed <= 0.0f) {
                        continue;
                    }
                    const float flow =
                        std::min(seed, capacity_from_dt(dt_row[x]));
                    if (flow <= 0.0f) {
                        continue;
                    }
                    routed_flow[static_cast<std::size_t>(index)] = flow;
                    ++row_seeded;
                }
                seeded_by_row[static_cast<std::size_t>(y)] = row_seeded;
            }
        });
        result.seeded_pixels +=
            std::accumulate(seeded_by_row.begin(), seeded_by_row.end(), 0);
        result.timings.push_back(finish_timing("dense_backtrack_seed", timing));
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
        std::vector<int>& grid_node_ids = scratch.dense_grid_node_ids;
        resize_fill(grid_node_ids,
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

        std::vector<float>& carrier_value = scratch.dense_carrier_value;
        resize_fill(carrier_value, carriers.size(), 0.0f);
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
        const std::size_t carrier_route_size =
            static_cast<std::size_t>(carrier_count) * kCarrierRouteBuckets;
        std::vector<float>& carrier_route_dp =
            scratch.dense_carrier_route_dp;
        std::vector<float>& carrier_route_distance_dp =
            scratch.dense_carrier_route_distance_dp;
        std::vector<int>& carrier_route_next =
            scratch.dense_carrier_route_next;
        resize_fill(carrier_route_dp, carrier_route_size, 0.0f);
        resize_fill(carrier_route_distance_dp, carrier_route_size, 0.0f);
        resize_fill(carrier_route_next, carrier_route_size, -1);
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
                cv::parallel_for_(cv::Range(0, carrier_count),
                                  [&](const cv::Range& range) {
                    for (int node = range.start; node < range.end; ++node) {
                        float best_flow = carrier_value[node];
                        float best_distance = 0.0f;
                        int best_next = -1;
                        for (const CarrierNeighbor neighbor :
                             carrier_edges[node]) {
                            float candidate_flow =
                                carrier_value[neighbor.node];
                            float candidate_distance =
                                std::min(neighbor.distance, budget);
                            if (neighbor.distance >= budget - kFlowEpsilon) {
                                const float t =
                                    neighbor.distance > 0.0f
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
                                        std::floor(
                                            (budget - neighbor.distance) /
                                            kCarrierRouteBucketPx)),
                                    0, bucket - 1);
                                candidate_flow =
                                    route_flow(neighbor.node, next_bucket);
                                candidate_distance =
                                    neighbor.distance +
                                    route_distance(neighbor.node,
                                                   next_bucket);
                            }
                            const bool better_flow =
                                candidate_flow > best_flow + kFlowEpsilon;
                            const bool same_flow_longer =
                                std::abs(candidate_flow - best_flow) <=
                                    kFlowEpsilon &&
                                candidate_distance >
                                    best_distance + kFlowEpsilon;
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
                });
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
        if (enable_debug_outputs) {
            std::cout << "Dense grid carrier backtrack:\n"
                      << "  grid_step: " << kGridStep << "\n"
                      << "  grid_carriers: " << grid_carriers << "\n"
                      << "  graph_carriers: " << graph_carriers << "\n"
                      << "  graph_node_carriers: " << graph_node_carriers
                      << "\n"
                      << "  graph_edge_carriers: " << graph_edge_carriers
                      << "\n"
                      << "  carrier_edges: " << (carrier_edges_count / 2)
                      << "\n"
                      << "  carrier_flow_seeds: " << carrier_flow_seeds
                      << "\n"
                      << "  carrier_flow_reached: " << carrier_flow_reached
                      << "\n"
                      << "  route_bucket_px: " << kCarrierRouteBucketPx
                      << "\n"
                      << "  route_buckets: " << kCarrierRouteBuckets << "\n"
                      << "  improved_carriers: " << improved_carriers << "\n"
                      << "  min_route_distance: " << min_route_distance
                      << "\n"
                      << "  max_route_distance: " << max_route_distance
                      << "\n";
        }
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
            const float reached_value = result.flow.at<float>(y, x);
            if (reached_value > 0.0f) {
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
                                          const cv::Mat& tree_candidates,
                                          const std::vector<cv::Point>& sources,
                                          int grid_step = 50,
                                          float backtrack_distance = 10.0f,
                                          bool enable_debug_outputs = true) {
    CV_Assert(white_domain.type() == CV_8U);
    CV_Assert(dt.type() == CV_32F);
    CV_Assert(tree_candidates.type() == CV_8U);
    CV_Assert(tree_candidates.size() == white_domain.size());

    if (sources.empty()) {
        throw std::runtime_error("--source is required");
    }
    const cv::Point primary_source = sources.front();
    if (primary_source.x < 0 || primary_source.x >= white_domain.cols ||
        primary_source.y < 0 || primary_source.y >= white_domain.rows) {
        throw std::runtime_error("--source is outside the image");
    }
    if (white_domain.at<std::uint8_t>(primary_source.y, primary_source.x) == 0) {
        throw std::runtime_error("--source must be inside the white distance domain");
    }

    std::vector<StageTiming> timings;
    constexpr float kDenseFlowInf = 1.0e9f;
    constexpr double kGraphFlowInf = 1.0e12;
    SkeletonGraph graph = input_graph;

    cv::Mat graph_edge_mask;
    cv::Mat graph_edge_index;
    cv::Mat graph_pixel_flow(white_domain.size(), CV_32F, cv::Scalar(0));
    {
        const GraphEdgeMaps edge_maps =
            build_graph_edge_maps(graph, white_domain.size());
        graph_edge_mask = edge_maps.edge_mask;
        graph_edge_index = edge_maps.edge_index;
    }

    cv::Mat island_obstacle_factor(white_domain.size(), CV_8UC3,
                                   cv::Scalar(0, 0, 0));
    IslandObstacleDebugResult island_debug;
    if (enable_debug_outputs) {
        const TimingMark timing = start_timing();
        island_debug = render_island_obstacle_factors(
            white_domain, graph, graph_edge_index);
        island_obstacle_factor = island_debug.label_factor;
        std::cout << "Island obstacle factors:\n"
                  << "  islands: " << island_debug.islands << "\n"
                  << "  loop_sampled_islands: "
                  << island_debug.loop_sampled_islands << "\n";
        for (const std::string& log : island_debug.score_logs) {
            std::cout << log << "\n";
        }
        timings.push_back(finish_timing("island_obstacle_factor", timing));
    }

    int source_seed_node = -1;
    cv::Point source_seed_pixel(-1, -1);
    float source_seed_capacity = 0.0f;
    int source_initial_edge = -1;
    std::vector<SourceEdgeStart> source_starts;
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

        cv::Mat nearest_graph_pixel;
        std::vector<cv::Point> graph_source_points;
        const auto ensure_nearest_graph_pixel = [&]() {
            if (!nearest_graph_pixel.empty()) {
                return;
            }
            cv::Mat graph_source_mask(white_domain.size(), CV_8U,
                                      cv::Scalar(255));
            graph_source_mask.setTo(0, graph_edge_mask);
            cv::Mat nearest_dt;
            cv::distanceTransform(graph_source_mask, nearest_dt,
                                  nearest_graph_pixel, cv::DIST_L2,
                                  cv::DIST_MASK_5, cv::DIST_LABEL_PIXEL);
            graph_source_points =
                label_source_points(graph_source_mask, nearest_graph_pixel);
        };

        constexpr float kDtEpsilon = 1.0e-4f;
        const auto detect_source_start =
            [&](const cv::Point source, const int source_index) {
                SourceEdgeStart result;
                result.source = source;
                result.source_index = source_index;
                if (!valid_domain(source)) {
                    return result;
                }

                int source_edge = graph_edge_index.at<int>(source.y, source.x);
                cv::Point current = source;
                cv::Mat visited = cv::Mat::zeros(white_domain.size(), CV_8U);
                while (source_edge < 0 && valid_domain(current) &&
                       visited.at<std::uint8_t>(current.y, current.x) == 0) {
                    visited.at<std::uint8_t>(current.y, current.x) = 255;

                    cv::Point best_edge_pixel(-1, -1);
                    float best_edge_dt =
                        -std::numeric_limits<float>::infinity();
                    for (const cv::Point dir : kDirs) {
                        const cv::Point neighbor = current + dir;
                        if (neighbor.x < 0 ||
                            neighbor.x >= graph_edge_index.cols ||
                            neighbor.y < 0 ||
                            neighbor.y >= graph_edge_index.rows) {
                            continue;
                        }
                        const int edge_index =
                            graph_edge_index.at<int>(neighbor.y, neighbor.x);
                        if (edge_index < 0) {
                            continue;
                        }
                        const float neighbor_dt =
                            dt.at<float>(neighbor.y, neighbor.x);
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

                    const float current_dt =
                        dt.at<float>(current.y, current.x);
                    cv::Point next(-1, -1);
                    float best_dt = current_dt;
                    for (const cv::Point dir : kDirs) {
                        const cv::Point neighbor = current + dir;
                        if (!valid_domain(neighbor) ||
                            graph.node_mask.at<std::uint8_t>(
                                neighbor.y, neighbor.x) != 0 ||
                            visited.at<std::uint8_t>(neighbor.y,
                                                     neighbor.x) != 0) {
                            continue;
                        }
                        const float neighbor_dt =
                            dt.at<float>(neighbor.y, neighbor.x);
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
                    ensure_nearest_graph_pixel();
                    const int label =
                        nearest_graph_pixel.at<int>(source.y, source.x);
                    if (label > 0 &&
                        label < static_cast<int>(graph_source_points.size())) {
                        const cv::Point graph_pixel =
                            graph_source_points[label];
                        if (graph_pixel.x >= 0) {
                            source_edge = graph_edge_index.at<int>(
                                graph_pixel.y, graph_pixel.x);
                        }
                    }
                }

                if (source_edge >= 0 &&
                    source_edge < static_cast<int>(graph.edges.size())) {
                    result.edge = source_edge;
                    const GraphEdge& edge = graph.edges[source_edge];
                    std::vector<cv::Point> ordered =
                        order_edge_pixels(edge, graph);
                    if (ordered.empty()) {
                        ordered = edge.pixels;
                    }
                    if (!ordered.empty()) {
                        int closest_index = 0;
                        double best_distance =
                            std::numeric_limits<double>::max();
                        for (int i = 0;
                             i < static_cast<int>(ordered.size()); ++i) {
                            const double dx = ordered[i].x - source.x;
                            const double dy = ordered[i].y - source.y;
                            const double distance = dx * dx + dy * dy;
                            if (distance < best_distance) {
                                best_distance = distance;
                                closest_index = i;
                            }
                        }
                        result.seed_pixel = ordered[closest_index];
                        result.seed_capacity = std::max(0.0f, edge.capacity);
                    }
                }
                return result;
            };

        for (int i = 0; i < static_cast<int>(sources.size()); ++i) {
            SourceEdgeStart start = detect_source_start(sources[i], i);
            if (start.edge < 0) {
                continue;
            }
            source_starts.push_back(start);
        }
        if (!source_starts.empty()) {
            source_initial_edge = source_starts.front().edge;
            source_seed_pixel = source_starts.front().seed_pixel;
            source_seed_capacity = source_starts.front().seed_capacity;
        }
        std::cout << "Source graph edges:\n"
                  << "  requested_sources: " << sources.size() << "\n"
                  << "  accepted_sources: " << source_starts.size() << "\n"
                  << "  first_source_edge: " << source_initial_edge << "\n"
                  << "  source_seed_pixel: (" << source_seed_pixel.x
                  << "," << source_seed_pixel.y << ")\n"
                  << "  source_seed_capacity: " << source_seed_capacity << "\n";
        timings.push_back(finish_timing("source_edge_detect", timing));
    }

    const int node_count = static_cast<int>(graph.nodes.size());
    std::vector<char> seeded_nodes(static_cast<std::size_t>(node_count), 0);
    std::vector<double> seed_node_capacity(static_cast<std::size_t>(node_count),
                                           0.0);
    std::vector<char> source_edges(graph.edges.size(), 0);
    std::vector<float> source_edge_capacity(graph.edges.size(), 0.0f);
    int source_first_edge = -1;
    int source_final_edge = -1;
    float source_final_edge_capacity = 0.0f;
    {
        const TimingMark timing = start_timing();
        const auto valid_node = [&](const int node) {
            return node > 0 && node < node_count;
        };
        const auto valid_edge = [&](const int edge_index) {
            return edge_index >= 0 &&
                   edge_index < static_cast<int>(graph.edges.size());
        };
        const auto edge_other_node = [&](const int edge_index,
                                         const int node) {
            if (!valid_edge(edge_index)) {
                return -1;
            }
            const GraphEdge& edge = graph.edges[edge_index];
            if (edge.a == node) {
                return edge.b;
            }
            if (edge.b == node) {
                return edge.a;
            }
            return -1;
        };

        std::vector<std::vector<int>> incident_edges(
            static_cast<std::size_t>(node_count));
        for (int edge_index = 0;
             edge_index < static_cast<int>(graph.edges.size()); ++edge_index) {
            const GraphEdge& edge = graph.edges[edge_index];
            if (valid_node(edge.a)) {
                incident_edges[static_cast<std::size_t>(edge.a)].push_back(
                    edge_index);
            }
            if (valid_node(edge.b) && edge.b != edge.a) {
                incident_edges[static_cast<std::size_t>(edge.b)].push_back(
                    edge_index);
            }
        }

        constexpr float kCapacityEpsilon = 1.0e-4f;
        source_first_edge = !source_starts.empty() &&
                                    valid_edge(source_starts.front().edge)
                                ? source_starts.front().edge
                                : -1;

        std::vector<int> traversed_edges;
        int terminal_node = -1;
        struct SourceAscentStep {
            int source_index = -1;
            int edge = -1;
            int node = -1;
            int next_edge = -1;
        };
        std::vector<SourceAscentStep> ascent_steps;
        struct SourceAscentNext {
            int edge = -1;
            int node = -1;
        };
        const auto select_next_edge = [&](const int edge_index,
                                          const int entry_node,
                                          const std::vector<char>&
                                              visited_edges) {
            SourceAscentNext result;
            if (!valid_edge(edge_index)) {
                return result;
            }
            const GraphEdge& edge = graph.edges[edge_index];
            std::array<int, 2> candidate_nodes = {edge.a, edge.b};
            if (valid_node(entry_node)) {
                candidate_nodes = {edge_other_node(edge_index, entry_node), -1};
            }
            float best_capacity = edge.capacity;
            for (const int node : candidate_nodes) {
                if (!valid_node(node)) {
                    continue;
                }
                for (const int candidate :
                     incident_edges[static_cast<std::size_t>(node)]) {
                    if (candidate == edge_index ||
                        visited_edges[static_cast<std::size_t>(candidate)] !=
                            0) {
                        continue;
                    }
                    const float capacity = graph.edges[candidate].capacity;
                    if (capacity <= best_capacity + kCapacityEpsilon) {
                        continue;
                    }
                    if (result.edge < 0 ||
                        capacity > best_capacity + kCapacityEpsilon ||
                        (std::abs(capacity - best_capacity) <=
                             kCapacityEpsilon &&
                         candidate < result.edge)) {
                        result.edge = candidate;
                        result.node = node;
                        best_capacity = capacity;
                    }
                }
            }
            return result;
        };

        for (const SourceEdgeStart& start : source_starts) {
            int current_edge = valid_edge(start.edge) ? start.edge : -1;
            int current_entry_node = -1;
            std::vector<char> visited_edges(graph.edges.size(), 0);
            while (valid_edge(current_edge) &&
                   visited_edges[static_cast<std::size_t>(current_edge)] ==
                       0) {
                visited_edges[static_cast<std::size_t>(current_edge)] = 1;
                traversed_edges.push_back(current_edge);
                source_edges[static_cast<std::size_t>(current_edge)] = 1;
                source_edge_capacity
                    [static_cast<std::size_t>(current_edge)] = std::max(
                        source_edge_capacity
                            [static_cast<std::size_t>(current_edge)],
                        std::max(0.0f, graph.edges[current_edge].capacity));
                source_final_edge = current_edge;
                source_final_edge_capacity = source_edge_capacity
                    [static_cast<std::size_t>(current_edge)];

                const SourceAscentNext next =
                    select_next_edge(current_edge, current_entry_node,
                                     visited_edges);
                terminal_node = next.node;
                if (next.edge < 0) {
                    ascent_steps.push_back(
                        {start.source_index, current_edge, next.node, -1});
                    break;
                }
                ascent_steps.push_back(
                    {start.source_index, current_edge, next.node, next.edge});
                current_entry_node = next.node;
                current_edge = next.edge;
            }
        }

        if (traversed_edges.empty() && valid_edge(source_first_edge)) {
            const GraphEdge& edge = graph.edges[source_first_edge];
            traversed_edges.push_back(source_first_edge);
            source_edges[static_cast<std::size_t>(source_first_edge)] = 1;
            source_edge_capacity[static_cast<std::size_t>(source_first_edge)] =
                std::max(0.0f, edge.capacity);
            source_final_edge = source_first_edge;
            source_final_edge_capacity =
                source_edge_capacity[static_cast<std::size_t>(
                    source_first_edge)];
        }

        float max_source_capacity = 0.0f;
        int max_source_edge = -1;
        for (int edge_index = 0;
             edge_index < static_cast<int>(graph.edges.size()); ++edge_index) {
            if (source_edges[static_cast<std::size_t>(edge_index)] == 0) {
                continue;
            }
            const GraphEdge& edge = graph.edges[edge_index];
            const float capacity = std::max(
                source_edge_capacity[static_cast<std::size_t>(edge_index)],
                std::max(0.0f, edge.capacity));
            source_edge_capacity[static_cast<std::size_t>(edge_index)] =
                capacity;
            if (capacity > max_source_capacity + kCapacityEpsilon ||
                (std::abs(capacity - max_source_capacity) <=
                     kCapacityEpsilon &&
                 (max_source_edge < 0 || edge_index < max_source_edge))) {
                max_source_capacity = capacity;
                max_source_edge = edge_index;
            }
            if (valid_node(edge.a)) {
                seeded_nodes[static_cast<std::size_t>(edge.a)] = 1;
                seed_node_capacity[static_cast<std::size_t>(edge.a)] =
                    std::max(seed_node_capacity
                                 [static_cast<std::size_t>(edge.a)],
                             static_cast<double>(capacity));
            }
            if (valid_node(edge.b)) {
                seeded_nodes[static_cast<std::size_t>(edge.b)] = 1;
                seed_node_capacity[static_cast<std::size_t>(edge.b)] =
                    std::max(seed_node_capacity
                                 [static_cast<std::size_t>(edge.b)],
                             static_cast<double>(capacity));
            }
        }

        if (max_source_capacity > 0.0f) {
            source_seed_capacity = max_source_capacity;
        }
        if (valid_edge(max_source_edge)) {
            source_final_edge = max_source_edge;
            source_final_edge_capacity = max_source_capacity;
        }
        if (valid_edge(source_final_edge) &&
            !graph.edges[source_final_edge].pixels.empty()) {
            const GraphEdge& final_edge = graph.edges[source_final_edge];
            cv::Point best_pixel = final_edge.pixels.front();
            float best_dt = dt.at<float>(best_pixel.y, best_pixel.x);
            for (const cv::Point pixel : final_edge.pixels) {
                const float pixel_dt = dt.at<float>(pixel.y, pixel.x);
                if (pixel_dt > best_dt + kCapacityEpsilon ||
                    (std::abs(pixel_dt - best_dt) <= kCapacityEpsilon &&
                     (pixel.y < best_pixel.y ||
                      (pixel.y == best_pixel.y && pixel.x < best_pixel.x)))) {
                    best_dt = pixel_dt;
                    best_pixel = pixel;
                }
            }
            source_seed_pixel = best_pixel;
        }
        std::cout << "Source edge ascent:\n"
                  << "  first_edge: " << source_first_edge << "\n"
                  << "  final_edge: " << source_final_edge << "\n"
                  << "  final_edge_capacity: " << source_final_edge_capacity
                  << "\n"
                  << "  traversed_edges: " << traversed_edges.size() << "\n";
        const auto print_edge = [&](const std::string& prefix,
                                    const int edge_index) {
            if (!valid_edge(edge_index)) {
                std::cout << prefix << edge_index << " invalid\n";
                return;
            }
            const GraphEdge& edge = graph.edges[edge_index];
            std::cout << prefix << edge_index << " cap=" << edge.capacity
                      << " a=" << edge.a << " b=" << edge.b
                      << " px=" << edge.pixels.size() << "\n";
        };
        const auto print_node_candidates = [&](const std::string& prefix,
                                               const int node,
                                               const int from_edge) {
            std::cout << prefix << " node=" << node;
            if (!valid_node(node)) {
                std::cout << " invalid\n";
                return;
            }
            std::cout << " degree="
                      << incident_edges[static_cast<std::size_t>(node)].size()
                      << " from_edge=" << from_edge << "\n";
            for (const int candidate :
                 incident_edges[static_cast<std::size_t>(node)]) {
                const GraphEdge& edge = graph.edges[candidate];
                std::cout << "    cand edge=" << candidate
                          << " cap=" << edge.capacity << " a=" << edge.a
                          << " b=" << edge.b
                          << " source="
                          << static_cast<int>(
                                 source_edges[static_cast<std::size_t>(
                                     candidate)])
                          << " uphill_from_current="
                          << (valid_edge(from_edge)
                                  ? edge.capacity >
                                        graph.edges[from_edge].capacity +
                                            kCapacityEpsilon
                                  : false)
                          << "\n";
            }
        };
        for (int step = 0; step < static_cast<int>(ascent_steps.size());
             ++step) {
            const SourceAscentStep& ascent_step = ascent_steps[step];
            std::cout << "  step_" << step << ": source="
                      << ascent_step.source_index << " edge="
                      << ascent_step.edge
                      << " node=" << ascent_step.node
                      << " next_edge=" << ascent_step.next_edge << "\n";
            print_node_candidates("    considered", ascent_step.node,
                                  ascent_step.edge);
        }
        print_edge("  first_edge_detail: ", source_first_edge);
        print_edge("  final_edge_detail: ", source_final_edge);
        if (valid_edge(source_final_edge)) {
            const GraphEdge& final_edge = graph.edges[source_final_edge];
            print_node_candidates("  final_endpoint_a", final_edge.a,
                                  source_final_edge);
            if (final_edge.b != final_edge.a) {
                print_node_candidates("  final_endpoint_b", final_edge.b,
                                      source_final_edge);
            }
        }
        std::cout << "  terminal_node: " << terminal_node << "\n";
        timings.push_back(finish_timing("source_edge_ascent", timing));
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

        int edge_source_count = 0;
        for (int edge_index = 0;
             edge_index < static_cast<int>(graph.edges.size()); ++edge_index) {
            if (edge_index == removed_edge ||
                source_edges[static_cast<std::size_t>(edge_index)] == 0 ||
                source_edge_capacity[static_cast<std::size_t>(edge_index)] <=
                    0.0f) {
                continue;
            }
            const GraphEdge& edge = graph.edges[edge_index];
            if ((edge.a > 0 && edge.a < node_count) ||
                (edge.b > 0 && edge.b < node_count)) {
                ++edge_source_count;
            }
        }

        Dinic flow(node_count + edge_source_count);
        constexpr int kSuperSource = 0;
        if (edge_source_count > 0) {
            int edge_source_node = node_count;
            for (int edge_index = 0;
                 edge_index < static_cast<int>(graph.edges.size());
                 ++edge_index) {
                if (edge_index == removed_edge ||
                    source_edges[static_cast<std::size_t>(edge_index)] == 0) {
                    continue;
                }
                const float capacity =
                    source_edge_capacity[static_cast<std::size_t>(edge_index)];
                if (capacity <= 0.0f) {
                    continue;
                }
                const GraphEdge& edge = graph.edges[edge_index];
                if ((edge.a <= 0 || edge.a >= node_count) &&
                    (edge.b <= 0 || edge.b >= node_count)) {
                    continue;
                }
                flow.add_edge(kSuperSource, edge_source_node, capacity);
                if (edge.a > 0 && edge.a < node_count) {
                    flow.add_edge(edge_source_node, edge.a, kGraphFlowInf);
                }
                if (edge.b > 0 && edge.b < node_count) {
                    flow.add_edge(edge_source_node, edge.b, kGraphFlowInf);
                }
                ++edge_source_node;
            }
        } else {
            for (int node = 1; node < node_count; ++node) {
                if (seeded_nodes[node] != 0) {
                    flow.add_edge(kSuperSource, node,
                                  std::max(0.0, seed_node_capacity[node]));
                }
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
        struct PixelFlowUpdate {
            int linear_index = 0;
            float value = 0.0f;
        };
        std::vector<std::vector<PixelFlowUpdate>> pixel_updates(
            graph.edges.size());
        cv::parallel_for_(
            cv::Range(0, static_cast<int>(graph.edges.size())),
            [&](const cv::Range& range) {
                for (int edge_index = range.start; edge_index < range.end;
                     ++edge_index) {
                    const GraphEdge& edge = graph.edges[edge_index];
                    std::vector<cv::Point> ordered =
                        order_edge_pixels(edge, graph);
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
                            const int dx =
                                std::abs(ordered[i].x - ordered[i - 1].x);
                            const int dy =
                                std::abs(ordered[i].y - ordered[i - 1].y);
                            dist_a[i] =
                                dist_a[i - 1] +
                                (dx + dy == 2 ? std::sqrt(2.0) : 1.0);
                        }
                        min_capacity = std::min(
                            min_capacity,
                            capacity_from_dt(
                                dt.at<float>(ordered[i].y, ordered[i].x)));
                        cap_a[i] = min_capacity;
                    }
                    min_capacity = node_b_flow;
                    for (std::size_t i = ordered.size(); i-- > 0;) {
                        if (i + 1 < ordered.size()) {
                            const int dx =
                                std::abs(ordered[i].x - ordered[i + 1].x);
                            const int dy =
                                std::abs(ordered[i].y - ordered[i + 1].y);
                            dist_b[i] =
                                dist_b[i + 1] +
                                (dx + dy == 2 ? std::sqrt(2.0) : 1.0);
                        }
                        min_capacity = std::min(
                            min_capacity,
                            capacity_from_dt(
                                dt.at<float>(ordered[i].y, ordered[i].x)));
                        cap_b[i] = min_capacity;
                    }

                    double edge_sum = 0.0;
                    std::vector<PixelFlowUpdate>& updates =
                        pixel_updates[static_cast<std::size_t>(edge_index)];
                    updates.reserve(ordered.size());
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
                        updates.push_back({pixel.y * graph_pixel_flow.cols +
                                               pixel.x,
                                           value});
                        edge_sum += value;
                    }
                    if (!ordered.empty()) {
                        edge_flow[static_cast<std::size_t>(edge_index)] =
                            static_cast<float>(std::min(
                                static_cast<double>(kDenseFlowInf),
                                edge_sum /
                                    static_cast<double>(ordered.size())));
                    }
                }
            });

        for (const std::vector<PixelFlowUpdate>& updates : pixel_updates) {
            for (const PixelFlowUpdate update : updates) {
                float& value = graph_pixel_flow.at<float>(
                    update.linear_index / graph_pixel_flow.cols,
                    update.linear_index % graph_pixel_flow.cols);
                value = std::max(value, update.value);
            }
        }
        timings.push_back(finish_timing("graph_edge_point_flow", timing));
    }

    cv::Mat island_flow_passability(white_domain.size(), CV_32FC3,
                                    cv::Scalar(0, 0, 0));
    cv::Mat island_propagated_edge_flow(white_domain.size(), CV_32FC3,
                                        cv::Scalar(0, 0, 0));
    cv::Mat island_bonus_edge_flow(white_domain.size(), CV_32FC3,
                                   cv::Scalar(0, 0, 0));
    IslandFlowPropagationResult island_flow;
    if (enable_debug_outputs) {
        const TimingMark timing = start_timing();
        island_flow =
            propagate_flow_across_island_links(graph, edge_flow, island_debug);
        island_flow_passability =
            render_graph_edge_values(graph, white_domain.size(),
                                     island_flow.edge_passability, 255.0f);
        island_propagated_edge_flow =
            render_graph_edge_values(graph, white_domain.size(),
                                     island_flow.propagated_edge_flow);
        island_bonus_edge_flow =
            render_graph_edge_values(graph, white_domain.size(),
                                     island_flow.bonus_edge_flow);
        std::cout << "Island flow propagation:\n"
                  << "  linked_islands: " << island_flow.linked_islands << "\n"
                  << "  transition_links: " << island_flow.transition_links
                  << "\n"
                  << "  boosted_edges: " << island_flow.boosted_edges << "\n";
        timings.push_back(
            finish_timing("island_flow_propagation", timing));
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
                    tree_candidates.at<std::uint8_t>(y, x) != 0) {
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
    cv::Mat tree_dense_flow_no_backtrack(white_domain.size(), CV_32F,
                                         cv::Scalar(0));
    cv::Mat nearest_graph_flow(white_domain.size(), CV_32F, cv::Scalar(0));
    cv::Mat tree_dense_flow_greedy_ascent(white_domain.size(), CV_32F,
                                          cv::Scalar(0));
    cv::Mat island_tree_dense_flow_no_backtrack(white_domain.size(), CV_32F,
                                                cv::Scalar(0));
    cv::Mat island_tree_dense_flow_greedy_ascent(white_domain.size(), CV_32F,
                                                 cv::Scalar(0));
    cv::Mat tree_dense_flow(white_domain.size(), CV_32F, cv::Scalar(0));
    cv::Mat tree_path_debug(white_domain.size(), CV_32FC3,
                            cv::Scalar(0, 0, 0));
    cv::Mat carrier_debug(white_domain.size(), CV_32FC3,
                          cv::Scalar(0, 0, 0));
    {
        const TimingMark timing = start_timing();
        nearest_graph_flow = nearest_graph_flow_projection(
            white_domain, graph_edge_mask, graph_pixel_flow);
        timings.push_back(finish_timing("nearest_graph_flow", timing));
    }
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
                                         graph_node_flow, graph, node_flow,
                                         edge_flow, source_seed_node,
                                         source_edges, grid_step,
                                         backtrack_distance,
                                         enable_debug_outputs);
        // Reference dense flood: useful for later experiments, but its TIFF is
        // disabled by default so timing focuses on the current tree dense flow.
        dense_backtrack_nn_flow = dense_backtrack.nn_flow;
        smooth_grid_flow = bilinear_from_regular_grid_samples(
            dense_backtrack.smooth_grid_flow, grid_step);
        tree_dense_flow_no_backtrack = smooth_grid_flow;
        {
            const TimingMark timing = start_timing();
            tree_dense_flow_greedy_ascent =
                greedy_increasing_flow_ascent(white_domain,
                                              tree_dense_flow_no_backtrack);
            timings.push_back(finish_timing(
                "tree_dense_flow_greedy_ascent", timing));
        }
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

    cv::Mat flow_gate_regions;
    {
        const TimingMark timing = start_timing();
        const FlowGateRegionBuildResult regions =
            build_flow_gate_region_labels(white_domain, dt, graph,
                                          source_starts);
        flow_gate_regions = regions.labels;
        std::cout << "Flow gate normalization regions:\n"
                  << "  regions: " << regions.region_count << "\n"
                  << "  source_groups: " << regions.source_groups << "\n"
                  << "  merged_source_pairs: "
                  << regions.merged_source_pairs << "\n"
                  << "  graph_seed_pixels: " << regions.graph_seed_pixels
                  << "\n"
                  << "  unlabeled_white_pixels: "
                  << regions.unlabeled_white_pixels << "\n";
        timings.push_back(
            finish_timing("flow_gate_region_labels", timing));
    }

    if (enable_debug_outputs &&
        island_flow.propagated_edge_flow.size() == graph.edges.size()) {
        const TimingMark timing = start_timing();
        std::vector<double> island_node_flow(
            static_cast<std::size_t>(node_count), 0.0);
        for (int node = 1; node < node_count; ++node) {
            const double value = node_flow[static_cast<std::size_t>(node)];
            if (std::isfinite(value) && value > 0.0) {
                island_node_flow[static_cast<std::size_t>(node)] = value;
            }
        }
        cv::Mat island_graph_pixel_flow(white_domain.size(), CV_32F,
                                        cv::Scalar(0));
        for (int edge_index = 0;
             edge_index < static_cast<int>(graph.edges.size()); ++edge_index) {
            const GraphEdge& edge = graph.edges[edge_index];
            const float value = island_flow.propagated_edge_flow
                [static_cast<std::size_t>(edge_index)];
            if (!std::isfinite(value) || value <= 0.0f) {
                continue;
            }
            if (edge.a > 0 && edge.a < node_count) {
                double& node_value =
                    island_node_flow[static_cast<std::size_t>(edge.a)];
                node_value = std::max(node_value, static_cast<double>(value));
            }
            if (edge.b > 0 && edge.b < node_count) {
                double& node_value =
                    island_node_flow[static_cast<std::size_t>(edge.b)];
                node_value = std::max(node_value, static_cast<double>(value));
            }
            for (const cv::Point pixel : edge.pixels) {
                if (pixel.x < 0 || pixel.x >= island_graph_pixel_flow.cols ||
                    pixel.y < 0 || pixel.y >= island_graph_pixel_flow.rows) {
                    continue;
                }
                float& pixel_value =
                    island_graph_pixel_flow.at<float>(pixel.y, pixel.x);
                pixel_value = std::max(pixel_value, value);
            }
        }

        cv::Mat island_graph_node_flow(white_domain.size(), CV_32F,
                                       cv::Scalar(0));
        for (int node = 1; node < node_count; ++node) {
            const cv::Point anchor(cvRound(graph.nodes[node].x),
                                   cvRound(graph.nodes[node].y));
            if (anchor.x < 0 || anchor.x >= island_graph_node_flow.cols ||
                anchor.y < 0 || anchor.y >= island_graph_node_flow.rows ||
                white_domain.at<std::uint8_t>(anchor.y, anchor.x) == 0) {
                continue;
            }
            const float value = static_cast<float>(std::min(
                static_cast<double>(kDenseFlowInf),
                std::max(0.0,
                         island_node_flow[static_cast<std::size_t>(node)])));
            if (value <= 0.0f) {
                continue;
            }
            cv::circle(island_graph_node_flow, anchor, 2, cv::Scalar(value),
                       cv::FILLED, cv::LINE_8);
        }

        const DenseBacktrackResult island_dense_backtrack =
            compute_dense_backtrack_flow(white_domain, dt,
                                         island_graph_pixel_flow,
                                         island_graph_node_flow, graph,
                                         island_node_flow,
                                         island_flow.propagated_edge_flow,
                                         source_seed_node, source_edges,
                                         grid_step, backtrack_distance, false);
        island_tree_dense_flow_no_backtrack =
            bilinear_from_regular_grid_samples(
                island_dense_backtrack.smooth_grid_flow, grid_step);
        island_tree_dense_flow_greedy_ascent =
            greedy_increasing_flow_ascent(
                white_domain, island_tree_dense_flow_no_backtrack);
        std::cout << "Island propagated dense flow:\n"
                  << "  seeded_pixels: "
                  << island_dense_backtrack.seeded_pixels << "\n"
                  << "  reached_pixels: "
                  << island_dense_backtrack.reached_pixels << "\n"
                  << "  unreached_white_pixels: "
                  << island_dense_backtrack.unreached_white_pixels << "\n";
        timings.push_back(
            finish_timing("island_propagated_dense_flow", timing));
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
            tree_dense_flow_no_backtrack,
            nearest_graph_flow,
            tree_dense_flow_greedy_ascent,
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
            island_obstacle_factor,
            island_flow_passability,
            island_propagated_edge_flow,
            island_bonus_edge_flow,
            island_tree_dense_flow_no_backtrack,
            island_tree_dense_flow_greedy_ascent,
            flow_gate_regions,
            cv::Mat(),
            source_seed_pixel,
            source_seed_capacity,
            source_edge_count,
            seeded_node_count,
            static_cast<int>(source_starts.size()),
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

FlowGateRegionBuildResult build_flow_gate_region_labels(
        const cv::Mat& white_domain,
        const cv::Mat& dt,
        const SkeletonGraph& graph,
        const std::vector<SourceEdgeStart>& source_starts) {
    CV_Assert(white_domain.type() == CV_8U);
    CV_Assert(dt.type() == CV_32F);
    CV_Assert(white_domain.size() == dt.size());

    FlowGateRegionBuildResult result;
    result.labels = cv::Mat::zeros(white_domain.size(), CV_32S);
    if (source_starts.empty()) {
        cv::Mat cc_labels;
        const int cc_count =
            cv::connectedComponents(white_domain, cc_labels, 8, CV_32S);
        result.labels = cc_labels;
        result.region_count = std::max(0, cc_count - 1);
        return result;
    }

    const int node_count = static_cast<int>(graph.nodes.size());
    const auto valid_node = [&](const int node) {
        return node > 0 && node < node_count;
    };
    const auto valid_edge = [&](const int edge_index) {
        return edge_index >= 0 &&
               edge_index < static_cast<int>(graph.edges.size());
    };
    const auto edge_other_node = [&](const int edge_index, const int node) {
        if (!valid_edge(edge_index)) {
            return -1;
        }
        const GraphEdge& edge = graph.edges[edge_index];
        if (edge.a == node) {
            return edge.b;
        }
        if (edge.b == node) {
            return edge.a;
        }
        return -1;
    };

    std::vector<std::vector<int>> incident_edges(
        static_cast<std::size_t>(node_count));
    std::vector<double> edge_lengths(graph.edges.size(), 1.0);
    for (int edge_index = 0;
         edge_index < static_cast<int>(graph.edges.size()); ++edge_index) {
        const GraphEdge& edge = graph.edges[edge_index];
        if (valid_node(edge.a)) {
            incident_edges[static_cast<std::size_t>(edge.a)].push_back(
                edge_index);
        }
        if (valid_node(edge.b) && edge.b != edge.a) {
            incident_edges[static_cast<std::size_t>(edge.b)].push_back(
                edge_index);
        }
        const std::vector<cv::Point> ordered = order_edge_pixels(edge, graph);
        double length = 1.0;
        for (std::size_t i = 1; i < ordered.size(); ++i) {
            const int dx = std::abs(ordered[i].x - ordered[i - 1].x);
            const int dy = std::abs(ordered[i].y - ordered[i - 1].y);
            length += (dx + dy == 2 ? std::sqrt(2.0) : 1.0);
        }
        edge_lengths[static_cast<std::size_t>(edge_index)] = length;
    }

    std::vector<float> best_score(static_cast<std::size_t>(node_count),
                                  -std::numeric_limits<float>::infinity());
    std::vector<double> best_distance(static_cast<std::size_t>(node_count),
                                      std::numeric_limits<double>::infinity());
    std::vector<int> node_label(static_cast<std::size_t>(node_count), 0);
    std::vector<int> edge_source_label(graph.edges.size(), 0);
    const int source_count = static_cast<int>(source_starts.size());
    struct SourceEndpointSeed {
        int node = -1;
        float capacity = 0.0f;
        double distance = 0.0;
    };
    std::vector<std::vector<SourceEndpointSeed>> source_endpoint_seeds(
        static_cast<std::size_t>(source_count + 1));
    std::vector<float> source_value(static_cast<std::size_t>(source_count + 1),
                                    0.0f);

    const auto add_endpoint_seed =
        [&](std::vector<SourceEndpointSeed>& seeds, const int node,
            const float capacity, const double distance) {
            if (!valid_node(node)) {
                return;
            }
            const float safe_capacity = std::max(0.0f, capacity);
            for (SourceEndpointSeed& seed : seeds) {
                if (seed.node != node) {
                    continue;
                }
                if (safe_capacity > seed.capacity + 1.0e-4f ||
                    (std::abs(safe_capacity - seed.capacity) <= 1.0e-4f &&
                     distance < seed.distance)) {
                    seed.capacity = safe_capacity;
                    seed.distance = distance;
                }
                return;
            }
            seeds.push_back({node, safe_capacity, distance});
        };

    for (int i = 0; i < source_count; ++i) {
        const int source_label = i + 1;
        const SourceEdgeStart& start = source_starts[i];
        if (!valid_edge(start.edge)) {
            continue;
        }
        const GraphEdge& edge = graph.edges[start.edge];
        std::vector<cv::Point> ordered = order_edge_pixels(edge, graph);
        if (ordered.empty()) {
            ordered = edge.pixels;
        }
        int seed_index = 0;
        if (!ordered.empty()) {
            double best_seed_distance = std::numeric_limits<double>::max();
            const cv::Point seed_pixel =
                start.seed_pixel.x >= 0 ? start.seed_pixel : start.source;
            for (int j = 0; j < static_cast<int>(ordered.size()); ++j) {
                const cv::Point pixel =
                    ordered[static_cast<std::size_t>(j)];
                const double dx = pixel.x - seed_pixel.x;
                const double dy = pixel.y - seed_pixel.y;
                const double distance = dx * dx + dy * dy;
                if (distance < best_seed_distance) {
                    best_seed_distance = distance;
                    seed_index = j;
                }
            }
        }
        auto segment_seed = [&](const int node, const int begin,
                                const int end) {
            if (!valid_node(node) || ordered.empty()) {
                return;
            }
            const int step = begin <= end ? 1 : -1;
            float capacity = std::numeric_limits<float>::infinity();
            double distance = 0.0;
            int previous = begin;
            for (int j = begin;; j += step) {
                const cv::Point pixel = ordered[static_cast<std::size_t>(j)];
                capacity = std::min(
                    capacity,
                    capacity_from_dt(dt.at<float>(pixel.y, pixel.x)));
                if (j != begin) {
                    const cv::Point prev =
                        ordered[static_cast<std::size_t>(previous)];
                    const int dx = std::abs(pixel.x - prev.x);
                    const int dy = std::abs(pixel.y - prev.y);
                    distance += (dx + dy == 2 ? std::sqrt(2.0) : 1.0);
                }
                if (j == end) {
                    break;
                }
                previous = j;
            }
            if (!std::isfinite(capacity)) {
                capacity = std::max(start.seed_capacity,
                                    std::max(0.0f, edge.capacity));
            }
            add_endpoint_seed(
                source_endpoint_seeds[static_cast<std::size_t>(source_label)],
                node, capacity, distance);
        };
        segment_seed(edge.a, seed_index, 0);
        segment_seed(edge.b, seed_index,
                     std::max(0, static_cast<int>(ordered.size()) - 1));
        if (ordered.empty()) {
            const float score = std::max(start.seed_capacity,
                                         std::max(0.0f, edge.capacity));
            add_endpoint_seed(
                source_endpoint_seeds[static_cast<std::size_t>(source_label)],
                edge.a, score, 0.0);
            add_endpoint_seed(
                source_endpoint_seeds[static_cast<std::size_t>(source_label)],
                edge.b, score, 0.0);
        }
        for (const SourceEndpointSeed& seed :
             source_endpoint_seeds[static_cast<std::size_t>(source_label)]) {
            source_value[static_cast<std::size_t>(source_label)] =
                std::max(source_value[static_cast<std::size_t>(source_label)],
                         seed.capacity);
        }
        if (source_value[static_cast<std::size_t>(source_label)] <= 0.0f) {
            source_value[static_cast<std::size_t>(source_label)] = std::max(
                start.seed_capacity, std::max(0.0f, edge.capacity));
        }
    }

    struct SourceDsu {
        std::vector<int> parent;
        explicit SourceDsu(const int n) : parent(static_cast<std::size_t>(n)) {
            std::iota(parent.begin(), parent.end(), 0);
        }
        int find(const int x) {
            if (parent[static_cast<std::size_t>(x)] == x) {
                return x;
            }
            parent[static_cast<std::size_t>(x)] =
                find(parent[static_cast<std::size_t>(x)]);
            return parent[static_cast<std::size_t>(x)];
        }
        void unite(const int a, const int b) {
            int ra = find(a);
            int rb = find(b);
            if (ra == rb) {
                return;
            }
            if (ra > rb) {
                std::swap(ra, rb);
            }
            parent[static_cast<std::size_t>(rb)] = ra;
        }
    };
    SourceDsu source_dsu(source_count + 1);

    struct SourceMergeQueueItem {
        float score = 0.0f;
        int node = -1;
    };
    struct SourceMergeQueueCompare {
        bool operator()(const SourceMergeQueueItem& a,
                        const SourceMergeQueueItem& b) const {
            if (std::abs(a.score - b.score) > 1.0e-4f) {
                return a.score < b.score;
            }
            return a.node > b.node;
        }
    };
    for (int source_a = 1; source_a <= source_count; ++source_a) {
        if (source_endpoint_seeds[static_cast<std::size_t>(source_a)].empty()) {
            continue;
        }
        std::vector<float> widest(static_cast<std::size_t>(node_count),
                                  -std::numeric_limits<float>::infinity());
        std::priority_queue<SourceMergeQueueItem,
                            std::vector<SourceMergeQueueItem>,
                            SourceMergeQueueCompare>
            merge_queue;
        const auto seed_widest = [&](const int node, const float score) {
            if (!valid_node(node) || score <= 0.0f ||
                score <= widest[static_cast<std::size_t>(node)] + 1.0e-4f) {
                return;
            }
            widest[static_cast<std::size_t>(node)] = score;
            merge_queue.push({score, node});
        };
        for (const SourceEndpointSeed& seed :
             source_endpoint_seeds[static_cast<std::size_t>(source_a)]) {
            seed_widest(seed.node, seed.capacity);
        }
        while (!merge_queue.empty()) {
            const SourceMergeQueueItem item = merge_queue.top();
            merge_queue.pop();
            if (!valid_node(item.node) ||
                std::abs(widest[static_cast<std::size_t>(item.node)] -
                         item.score) > 1.0e-4f) {
                continue;
            }
            for (const int edge_index :
                 incident_edges[static_cast<std::size_t>(item.node)]) {
                const int next_node = edge_other_node(edge_index, item.node);
                if (!valid_node(next_node)) {
                    continue;
                }
                const float score = std::min(
                    item.score,
                    std::max(0.0f, graph.edges[edge_index].capacity));
                seed_widest(next_node, score);
            }
        }
        for (int source_b = source_a + 1; source_b <= source_count;
             ++source_b) {
            float bottleneck = 0.0f;
            for (const SourceEndpointSeed& seed :
                 source_endpoint_seeds[static_cast<std::size_t>(source_b)]) {
                if (!valid_node(seed.node)) {
                    continue;
                }
                const float reached =
                    widest[static_cast<std::size_t>(seed.node)];
                if (reached <= 0.0f) {
                    continue;
                }
                bottleneck =
                    std::max(bottleneck, std::min(reached, seed.capacity));
            }
            const float required = kFlowGateRegionMergeRatio *
                                   std::max(source_value
                                                [static_cast<std::size_t>(
                                                    source_a)],
                                            source_value
                                                [static_cast<std::size_t>(
                                                    source_b)]);
            if (required > 1.0e-4f && bottleneck + 1.0e-4f >= required) {
                if (source_dsu.find(source_a) != source_dsu.find(source_b)) {
                    ++result.merged_source_pairs;
                }
                source_dsu.unite(source_a, source_b);
            }
        }
    }
    std::vector<int> root_region_label(
        static_cast<std::size_t>(source_count + 1), 0);
    std::vector<int> source_region_label(
        static_cast<std::size_t>(source_count + 1), 0);
    for (int i = 1; i <= source_count; ++i) {
        const int root = source_dsu.find(i);
        int& label = root_region_label[static_cast<std::size_t>(root)];
        if (label == 0) {
            label = ++result.source_groups;
        }
        source_region_label[static_cast<std::size_t>(i)] = label;
    }

    struct QueueItem {
        float score = 0.0f;
        double distance = 0.0;
        int label = 0;
        int node = 0;
    };
    struct QueueCompare {
        bool operator()(const QueueItem& a, const QueueItem& b) const {
            constexpr float kScoreEps = 1.0e-4f;
            constexpr double kDistEps = 1.0e-6;
            if (std::abs(a.score - b.score) > kScoreEps) {
                return a.score < b.score;
            }
            if (std::abs(a.distance - b.distance) > kDistEps) {
                return a.distance > b.distance;
            }
            return a.label > b.label;
        }
    };
    std::priority_queue<QueueItem, std::vector<QueueItem>, QueueCompare> queue;

    const auto better_node_label = [&](const int node, const int label,
                                       const float score,
                                       const double distance) {
        constexpr float kScoreEps = 1.0e-4f;
        constexpr double kDistEps = 1.0e-6;
        if (score > best_score[static_cast<std::size_t>(node)] + kScoreEps) {
            return true;
        }
        if (std::abs(score -
                     best_score[static_cast<std::size_t>(node)]) <=
            kScoreEps) {
            if (distance <
                best_distance[static_cast<std::size_t>(node)] - kDistEps) {
                return true;
            }
            if (std::abs(distance -
                         best_distance[static_cast<std::size_t>(node)]) <=
                    kDistEps &&
                (node_label[static_cast<std::size_t>(node)] == 0 ||
                 label < node_label[static_cast<std::size_t>(node)])) {
                return true;
            }
        }
        return false;
    };
    const auto seed_node = [&](const int node, const int label,
                               const float score, const double distance) {
        if (!valid_node(node) || label <= 0) {
            return;
        }
        const float safe_score = std::max(0.0f, score);
        if (!better_node_label(node, label, safe_score, distance)) {
            return;
        }
        best_score[static_cast<std::size_t>(node)] = safe_score;
        best_distance[static_cast<std::size_t>(node)] = distance;
        node_label[static_cast<std::size_t>(node)] = label;
        queue.push({safe_score, distance, label, node});
    };

    for (int i = 0; i < source_count; ++i) {
        const SourceEdgeStart& start = source_starts[i];
        if (!valid_edge(start.edge)) {
            continue;
        }
        const int source_label = i + 1;
        const int region_label =
            source_region_label[static_cast<std::size_t>(source_label)];
        if (region_label <= 0) {
            continue;
        }
        edge_source_label[static_cast<std::size_t>(start.edge)] =
            region_label;
        for (const SourceEndpointSeed& seed :
             source_endpoint_seeds[static_cast<std::size_t>(source_label)]) {
            seed_node(seed.node, region_label, seed.capacity, seed.distance);
        }
    }

    while (!queue.empty()) {
        const QueueItem item = queue.top();
        queue.pop();
        if (!valid_node(item.node)) {
            continue;
        }
        const int node_index = item.node;
        if (node_label[static_cast<std::size_t>(node_index)] != item.label ||
            std::abs(best_score[static_cast<std::size_t>(node_index)] -
                     item.score) > 1.0e-4f ||
            std::abs(best_distance[static_cast<std::size_t>(node_index)] -
                     item.distance) > 1.0e-6) {
            continue;
        }
        for (const int edge_index :
             incident_edges[static_cast<std::size_t>(node_index)]) {
            const int next_node = edge_other_node(edge_index, node_index);
            if (!valid_node(next_node)) {
                continue;
            }
            const GraphEdge& edge = graph.edges[edge_index];
            const float score =
                std::min(item.score, std::max(0.0f, edge.capacity));
            const double distance =
                item.distance +
                edge_lengths[static_cast<std::size_t>(edge_index)];
            seed_node(next_node, item.label, score, distance);
        }
    }

    cv::Mat graph_labels = cv::Mat::zeros(white_domain.size(), CV_32S);
    const auto set_graph_label = [&](const cv::Point pixel, const int label) {
        if (label <= 0 || pixel.x < 0 || pixel.x >= graph_labels.cols ||
            pixel.y < 0 || pixel.y >= graph_labels.rows ||
            white_domain.at<std::uint8_t>(pixel.y, pixel.x) == 0) {
            return;
        }
        int& current = graph_labels.at<int>(pixel.y, pixel.x);
        if (current == 0 || label < current) {
            current = label;
        }
    };

    for (int node = 1; node < node_count; ++node) {
        const int label = node_label[static_cast<std::size_t>(node)];
        if (label <= 0) {
            continue;
        }
        if (node < static_cast<int>(graph.node_pixel_groups.size())) {
            for (const cv::Point pixel :
                 graph.node_pixel_groups[static_cast<std::size_t>(node)]) {
                set_graph_label(pixel, label);
            }
        }
        const cv::Point anchor(cvRound(graph.nodes[node].x),
                               cvRound(graph.nodes[node].y));
        set_graph_label(anchor, label);
    }

    for (int edge_index = 0;
         edge_index < static_cast<int>(graph.edges.size()); ++edge_index) {
        const GraphEdge& edge = graph.edges[edge_index];
        int label_a =
            valid_node(edge.a) ? node_label[static_cast<std::size_t>(edge.a)]
                               : 0;
        int label_b =
            valid_node(edge.b) ? node_label[static_cast<std::size_t>(edge.b)]
                               : 0;
        const int source_label =
            edge_source_label[static_cast<std::size_t>(edge_index)];
        if (label_a == 0 && label_b == 0 && source_label > 0) {
            label_a = source_label;
            label_b = source_label;
        } else if (label_a == 0) {
            label_a = label_b > 0 ? label_b : source_label;
        } else if (label_b == 0) {
            label_b = label_a > 0 ? label_a : source_label;
        }
        std::vector<cv::Point> ordered = order_edge_pixels(edge, graph);
        if (ordered.empty()) {
            ordered = edge.pixels;
        }
        if (ordered.empty()) {
            continue;
        }
        if (label_a == label_b || label_a <= 0 || label_b <= 0) {
            const int label = label_a > 0 ? label_a : label_b;
            for (const cv::Point pixel : ordered) {
                set_graph_label(pixel, label);
            }
            continue;
        }

        if (edge.a != edge.b && ordered.size() > 1) {
            const cv::Point first = ordered.front();
            const cv::Point last = ordered.back();
            const cv::Point2f a = graph.nodes[edge.a];
            const cv::Point2f b = graph.nodes[edge.b];
            const double first_a = (first.x - a.x) * (first.x - a.x) +
                                   (first.y - a.y) * (first.y - a.y);
            const double first_b = (first.x - b.x) * (first.x - b.x) +
                                   (first.y - b.y) * (first.y - b.y);
            const double last_a = (last.x - a.x) * (last.x - a.x) +
                                  (last.y - a.y) * (last.y - a.y);
            const double last_b = (last.x - b.x) * (last.x - b.x) +
                                  (last.y - b.y) * (last.y - b.y);
            if (first_b + last_a < first_a + last_b) {
                std::reverse(ordered.begin(), ordered.end());
            }
        }

        int cut_index = static_cast<int>(ordered.size() / 2);
        float best_capacity = std::numeric_limits<float>::infinity();
        const double middle = static_cast<double>(ordered.size() - 1) * 0.5;
        for (int i = 0; i < static_cast<int>(ordered.size()); ++i) {
            const cv::Point pixel = ordered[static_cast<std::size_t>(i)];
            if (pixel.x < 0 || pixel.x >= dt.cols || pixel.y < 0 ||
                pixel.y >= dt.rows) {
                continue;
            }
            const float capacity = capacity_from_dt(
                dt.at<float>(pixel.y, pixel.x));
            const double middle_distance = std::abs(static_cast<double>(i) -
                                                    middle);
            const double best_middle_distance =
                std::abs(static_cast<double>(cut_index) - middle);
            if (capacity < best_capacity - 1.0e-4f ||
                (std::abs(capacity - best_capacity) <= 1.0e-4f &&
                 middle_distance < best_middle_distance)) {
                best_capacity = capacity;
                cut_index = i;
            }
        }
        for (int i = 0; i < static_cast<int>(ordered.size()); ++i) {
            set_graph_label(ordered[static_cast<std::size_t>(i)],
                            i <= cut_index ? label_a : label_b);
        }
    }

    struct PixelQueueItem {
        float distance = 0.0f;
        int label = 0;
        cv::Point pixel{-1, -1};
    };
    struct PixelQueueCompare {
        bool operator()(const PixelQueueItem& a,
                        const PixelQueueItem& b) const {
            if (std::abs(a.distance - b.distance) > 1.0e-6f) {
                return a.distance > b.distance;
            }
            return a.label > b.label;
        }
    };
    std::priority_queue<PixelQueueItem, std::vector<PixelQueueItem>,
                        PixelQueueCompare>
        pixel_queue;
    cv::Mat label_distance(white_domain.size(), CV_32F,
                           cv::Scalar(std::numeric_limits<float>::infinity()));
    const auto seed_label_pixel = [&](const cv::Point pixel, const int label,
                                      const float distance) {
        if (label <= 0 || pixel.x < 0 || pixel.x >= white_domain.cols ||
            pixel.y < 0 || pixel.y >= white_domain.rows ||
            white_domain.at<std::uint8_t>(pixel.y, pixel.x) == 0) {
            return;
        }
        float& current_distance =
            label_distance.at<float>(pixel.y, pixel.x);
        int& current_label = result.labels.at<int>(pixel.y, pixel.x);
        if (distance < current_distance - 1.0e-6f ||
            (std::abs(distance - current_distance) <= 1.0e-6f &&
             (current_label == 0 || label < current_label))) {
            current_distance = distance;
            current_label = label;
            pixel_queue.push({distance, label, pixel});
        }
    };
    for (int y = 0; y < graph_labels.rows; ++y) {
        for (int x = 0; x < graph_labels.cols; ++x) {
            const int label = graph_labels.at<int>(y, x);
            if (label <= 0) {
                continue;
            }
            ++result.graph_seed_pixels;
            seed_label_pixel(cv::Point(x, y), label, 0.0f);
        }
    }

    const std::array<cv::Point, 8> kDirs = {
        {{-1, -1}, {0, -1}, {1, -1}, {-1, 0},
         {1, 0},   {-1, 1}, {0, 1},  {1, 1}}};
    while (!pixel_queue.empty()) {
        const PixelQueueItem item = pixel_queue.top();
        pixel_queue.pop();
        if (item.pixel.x < 0 || item.pixel.x >= white_domain.cols ||
            item.pixel.y < 0 || item.pixel.y >= white_domain.rows) {
            continue;
        }
        if (result.labels.at<int>(item.pixel.y, item.pixel.x) != item.label ||
            std::abs(label_distance.at<float>(item.pixel.y, item.pixel.x) -
                     item.distance) > 1.0e-6f) {
            continue;
        }
        for (const cv::Point dir : kDirs) {
            const cv::Point next = item.pixel + dir;
            if (next.x < 0 || next.x >= white_domain.cols || next.y < 0 ||
                next.y >= white_domain.rows ||
                white_domain.at<std::uint8_t>(next.y, next.x) == 0) {
                continue;
            }
            const float step = (std::abs(dir.x) + std::abs(dir.y) == 2)
                                   ? static_cast<float>(std::sqrt(2.0))
                                   : 1.0f;
            seed_label_pixel(next, item.label, item.distance + step);
        }
    }

    double max_label_value = 0.0;
    cv::minMaxLoc(result.labels, nullptr, &max_label_value);
    result.region_count = static_cast<int>(max_label_value);

    cv::Mat cc_labels;
    const int cc_count =
        cv::connectedComponents(white_domain, cc_labels, 8, CV_32S);
    std::vector<int> component_label(static_cast<std::size_t>(cc_count), 0);
    for (int y = 0; y < result.labels.rows; ++y) {
        for (int x = 0; x < result.labels.cols; ++x) {
            const int label = result.labels.at<int>(y, x);
            if (label <= 0) {
                continue;
            }
            const int component = cc_labels.at<int>(y, x);
            if (component > 0 && component < cc_count &&
                component_label[static_cast<std::size_t>(component)] == 0) {
                component_label[static_cast<std::size_t>(component)] = label;
            }
        }
    }
    for (int y = 0; y < result.labels.rows; ++y) {
        for (int x = 0; x < result.labels.cols; ++x) {
            if (white_domain.at<std::uint8_t>(y, x) == 0 ||
                result.labels.at<int>(y, x) > 0) {
                continue;
            }
            ++result.unlabeled_white_pixels;
            const int component = cc_labels.at<int>(y, x);
            if (component <= 0 || component >= cc_count) {
                continue;
            }
            int& label = component_label[static_cast<std::size_t>(component)];
            if (label == 0) {
                label = ++result.region_count;
            }
            result.labels.at<int>(y, x) = label;
        }
    }

    return result;
}

struct FlowGateNormalizationStats {
    int region_count = 0;
    double global_max_flow = 0.0;
};

cv::Mat normalize_flow_by_regions(
        const cv::Mat& flow,
        const cv::Mat& region_labels,
        FlowGateNormalizationStats* stats = nullptr) {
    CV_Assert(flow.type() == CV_32F);
    CV_Assert(region_labels.empty() ||
              (region_labels.type() == CV_32S &&
               region_labels.size() == flow.size()));

    cv::Mat nonnegative_flow = flow.clone();
    nonnegative_flow.setTo(0.0f, nonnegative_flow < 0.0f);

    double global_max_flow = 0.0;
    cv::minMaxLoc(nonnegative_flow, nullptr, &global_max_flow);
    double max_region_label_value = 0.0;
    if (!region_labels.empty()) {
        cv::minMaxLoc(region_labels, nullptr, &max_region_label_value);
    }
    const int region_count = static_cast<int>(max_region_label_value);

    std::vector<float> region_max(static_cast<std::size_t>(region_count + 1),
                                  0.0f);
    if (region_count > 0) {
        for (int y = 0; y < flow.rows; ++y) {
            const float* flow_row = nonnegative_flow.ptr<float>(y);
            const int* label_row = region_labels.ptr<int>(y);
            for (int x = 0; x < flow.cols; ++x) {
                const int label = label_row[x];
                if (label <= 0 || label > region_count) {
                    continue;
                }
                region_max[static_cast<std::size_t>(label)] =
                    std::max(region_max[static_cast<std::size_t>(label)],
                             flow_row[x]);
            }
        }
    }

    const float fallback_denominator =
        global_max_flow > 1.0e-6 ? static_cast<float>(global_max_flow) : 0.0f;
    cv::Mat normalized_flow(flow.size(), CV_32F, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, flow.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const float* flow_row = nonnegative_flow.ptr<float>(y);
            const int* label_row =
                region_count > 0 ? region_labels.ptr<int>(y) : nullptr;
            float* out_row = normalized_flow.ptr<float>(y);
            for (int x = 0; x < flow.cols; ++x) {
                if (flow_row[x] <= 0.0f) {
                    continue;
                }
                const int label = label_row != nullptr ? label_row[x] : 0;
                const float denominator =
                    label > 0 && label <= region_count
                        ? region_max[static_cast<std::size_t>(label)]
                        : fallback_denominator;
                if (denominator <= 1.0e-6f) {
                    continue;
                }
                out_row[x] = std::clamp(flow_row[x] / denominator, 0.0f, 1.0f);
            }
        }
    });

    if (stats != nullptr) {
        stats->region_count = region_count;
        stats->global_max_flow = global_max_flow;
    }
    return normalized_flow;
}

cv::Mat compute_flow_gate_weight_image(const cv::Mat& flow,
                                       const float backtrack_distance,
                                       const float local_boost,
                                       const cv::Mat& region_labels =
                                           cv::Mat()) {
    CV_Assert(flow.type() == CV_32F);
    CV_Assert(region_labels.empty() ||
              (region_labels.type() == CV_32S &&
               region_labels.size() == flow.size()));
    const int rows = flow.rows;
    const int cols = flow.cols;
    cv::Mat weight(flow.size(), CV_32F, cv::Scalar(0));

    FlowGateNormalizationStats norm_stats;
    cv::Mat normalized_flow =
        normalize_flow_by_regions(flow, region_labels, &norm_stats);
    const int region_count = norm_stats.region_count;
    const float local_blend = std::clamp(local_boost, 0.0f, 1.0f);

    const int local_radius =
        std::max(0, static_cast<int>(std::ceil(backtrack_distance)));
    cv::Mat local_max;
    if (local_radius > 0) {
        const int kernel_size = local_radius * 2 + 1;
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
        if (region_count > 0) {
            local_max = cv::Mat::zeros(flow.size(), CV_32F);
            for (int label = 1; label <= region_count; ++label) {
                const cv::Mat label_mask = region_labels == label;
                cv::Mat label_flow = cv::Mat::zeros(flow.size(), CV_32F);
                normalized_flow.copyTo(label_flow, label_mask);
                cv::Mat label_local_max;
                cv::dilate(label_flow, label_local_max, kernel);
                label_local_max.copyTo(local_max, label_mask);
            }
        } else {
            cv::dilate(normalized_flow, local_max, kernel);
        }
    } else {
        local_max = normalized_flow;
    }

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            float* out_row = weight.ptr<float>(y);
            const float* norm_row = normalized_flow.ptr<float>(y);
            const float* max_row = local_max.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                const float normalized_gate = norm_row[x];
                if (normalized_gate <= 0.0f) {
                    continue;
                }
                const float local_denominator = max_row[x];
                const float local_gate =
                    local_denominator > 1.0e-6f
                        ? std::clamp(normalized_gate / local_denominator,
                                     0.0f, 1.0f)
                        : 0.0f;
                out_row[x] = local_blend * local_gate +
                             (1.0f - local_blend) * normalized_gate;
            }
        }
    });

    std::cout << "Flow gate weight:\n"
              << "  local_max_radius: " << local_radius << "\n"
              << "  backtrack_distance: " << backtrack_distance << "\n"
              << "  local_boost: " << local_blend << "\n"
              << "  normalization_regions: " << region_count << "\n"
              << "  global_max_flow: " << norm_stats.global_max_flow << "\n";
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

SourceRimSkeletonResult source_rim_skeleton(
    const cv::Mat& white_domain,
    const cv::Mat& source_pixel_labels,
    const std::vector<cv::Point>& source_points) {
    CV_Assert(white_domain.type() == CV_8U);
    CV_Assert(source_pixel_labels.type() == CV_32S);
    CV_Assert(white_domain.size() == source_pixel_labels.size());

    std::vector<StageTiming> timings;

    cv::Mat source_rim_ridges;
    cv::Mat source_rim_distance;
    cv::Mat source_rim_arc;
    std::string rim_connectivity_error;
    {
        const TimingMark timing = start_timing();
        constexpr float kSourceRimRidgeThresholdPx = 12.0f;
        source_rim_ridges =
            source_rim_distance_label_ridges(
                white_domain, source_pixel_labels, source_points,
                kSourceRimRidgeThresholdPx,
                &source_rim_distance, &source_rim_arc, &timings,
                &rim_connectivity_error);
        timings.push_back(
            finish_timing("source_rim.rim_distance_ridges", timing));
    }

    cv::Mat source_rim_arc_skeleton;
    {
        const TimingMark timing = start_timing();
        source_rim_arc_skeleton = optimized_thinning(source_rim_arc);
        timings.push_back(finish_timing("source_rim.rim_arc_thinning", timing));
    }

    cv::Mat loops_connected;
    {
        const TimingMark timing = start_timing();
        loops_connected = optimized_thinning(source_rim_ridges);
        timings.push_back(finish_timing("source_rim.thinning", timing));
    }

    return {source_rim_ridges,
            source_rim_distance,
            source_rim_arc,
            source_rim_arc_skeleton,
            loops_connected,
            timings,
            rim_connectivity_error};
}

SourceRimSkeletonResult crop_padded_source_rim_result(
    const SourceRimSkeletonResult& padded, const cv::Size original_size) {
    const cv::Rect roi(1, 1, original_size.width, original_size.height);
    return {padded.source_rim_ridges(roi).clone(),
            padded.source_rim_distance(roi).clone(),
            padded.source_rim_arc(roi).clone(),
            padded.source_rim_arc_skeleton(roi).clone(),
            padded.loops_connected(roi).clone(),
            padded.timings,
            padded.rim_connectivity_error};
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

struct FlowGraphPipelineResult {
    cv::Mat white_domain;
    cv::Mat binary;
    cv::Mat dt;
    cv::Mat source_pixel_labels;
    SourceRimSkeletonResult source_rim_result;
    SkeletonGraph graph;
    GraphConnectivityStats graph_stats;
    std::string rim_error;
    std::string graph_error;
};

FlowGraphPipelineResult build_flow_graph_pipeline(
    const cv::Mat& input_white_domain,
    const std::vector<cv::Point>& sources,
    std::vector<StageTiming>& timings,
    const std::string& timing_prefix = "") {
    CV_Assert(input_white_domain.type() == CV_8U);

    FlowGraphPipelineResult result;
    result.white_domain = input_white_domain.clone();
    cv::bitwise_not(result.white_domain, result.binary);

    {
        const TimingMark timing = start_timing();
        cv::distanceTransform(result.white_domain, result.dt,
                              result.source_pixel_labels, cv::DIST_L2,
                              cv::DIST_MASK_5, cv::DIST_LABEL_PIXEL);
        timings.push_back(
            finish_timing(timing_prefix + "labeled_dt", timing));
    }

    cv::Mat rim_label_domain;
    cv::Mat rim_dt;
    cv::Mat rim_source_pixel_labels;
    {
        const TimingMark timing = start_timing();
        rim_label_domain =
            make_padded_expanded_rim_label_domain(result.white_domain, sources);
        cv::distanceTransform(rim_label_domain, rim_dt,
                              rim_source_pixel_labels, cv::DIST_L2,
                              cv::DIST_MASK_5, cv::DIST_LABEL_PIXEL);
        timings.push_back(
            finish_timing(timing_prefix + "rim_labeled_dt", timing));
    }

    std::vector<cv::Point> rim_source_points;
    {
        const TimingMark timing = start_timing();
        rim_source_points =
            label_source_points(rim_label_domain, rim_source_pixel_labels);
        timings.push_back(
            finish_timing(timing_prefix + "rim_label_source_points", timing));
    }

    {
        const TimingMark timing = start_timing();
        result.source_rim_result = source_rim_skeleton(
            rim_label_domain, rim_source_pixel_labels, rim_source_points);
        result.source_rim_result = crop_padded_source_rim_result(
            result.source_rim_result, result.white_domain.size());
        timings.push_back(
            finish_timing(timing_prefix + "source_rim_skeleton", timing));
        timings.insert(timings.end(),
                       result.source_rim_result.timings.begin(),
                       result.source_rim_result.timings.end());
        result.rim_error = result.source_rim_result.rim_connectivity_error;
    }

    {
        const TimingMark timing = start_timing();
        result.source_rim_result.loops_connected =
            add_valid_frame_to_skeleton(
                result.source_rim_result.loops_connected, result.dt);
        timings.push_back(
            finish_timing(timing_prefix + "add_valid_frame", timing));
    }

    {
        const TimingMark timing = start_timing();
        result.graph = extract_skeleton_graph(
            result.source_rim_result.loops_connected, result.dt);
        timings.push_back(
            finish_timing(timing_prefix + "graph_extract", timing));
    }

    {
        const TimingMark timing = start_timing();
        result.graph_stats = graph_connectivity_stats(result.graph);
        result.graph_error = graph_connectivity_error(result.graph_stats);
        timings.push_back(
            finish_timing(timing_prefix + "graph_connectivity_stats", timing));
    }

    return result;
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
                                        const int* extra_source_xy,
                                        int extra_source_count,
                                        const float* query_xy,
                                        int query_count,
                                        float* query_flow,
                                        float* dense_flow,
                                        float* smooth_grid_flow,
                                        float* gate_basis_flow,
                                        float* graph_edge_flow_rgb,
                                        float* island_obstacle_factor_rgb,
                                        float* island_removed_mask,
                                        float* island_flow_passability_rgb,
                                        float* island_propagated_edge_flow_rgb,
                                        float* island_bonus_edge_flow_rgb,
                                        float* island_tree_dense_no_backtrack,
                                        float* island_tree_dense_greedy_ascent,
                                        float* source_edge_mask,
                                        float* source_component_mask,
                                        int grid_step,
                                        float backtrack_distance,
                                        float local_boost,
                                        int* resolved_source_x,
                                        int* resolved_source_y,
                                        float* resolved_source_capacity,
                                        int* resolved_accepted_sources,
                                        int* resolved_source_edges,
                                        int* resolved_seeded_nodes,
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
        if (extra_source_count < 0) {
            throw std::runtime_error("extra_source_count must be non-negative");
        }
        if (extra_source_count > 0 && extra_source_xy == nullptr) {
            throw std::runtime_error(
                "extra_source_xy is required when extra_source_count > 0");
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
        std::vector<cv::Point> flow_sources;
        flow_sources.reserve(static_cast<std::size_t>(extra_source_count) + 1);
        flow_sources.push_back(source);
        for (int i = 0; i < extra_source_count; ++i) {
            flow_sources.push_back(
                cv::Point(extra_source_xy[2 * i], extra_source_xy[2 * i + 1]));
        }
        {
            const TimingMark timing = start_timing();
            binary = keep_source_white_components(binary, flow_sources);
            timings.push_back(
                finish_timing("source_connected_component", timing));
        }

        cv::Mat white_domain(binary.size(), CV_8U, cv::Scalar(255));
        white_domain.setTo(0, binary);

        const bool enable_debug_outputs =
            smooth_grid_flow != nullptr || gate_basis_flow != nullptr ||
            graph_edge_flow_rgb != nullptr ||
            island_obstacle_factor_rgb != nullptr ||
            island_removed_mask != nullptr ||
            island_flow_passability_rgb != nullptr ||
            island_propagated_edge_flow_rgb != nullptr ||
            island_bonus_edge_flow_rgb != nullptr ||
            island_tree_dense_no_backtrack != nullptr ||
            island_tree_dense_greedy_ascent != nullptr ||
            source_edge_mask != nullptr ||
            source_component_mask != nullptr;

        const auto allow_multi_source_graph_error =
            [&](const std::string& graph_error) {
                return flow_sources.size() > 1 &&
                       graph_error.rfind("extracted graph is disconnected", 0) ==
                           0;
            };
        const auto clear_allowed_graph_error =
            [&](FlowGraphPipelineResult& result) {
                if (allow_multi_source_graph_error(result.graph_error)) {
                    std::cout << "Multi-source flow graph:\n"
                              << "  allowing disconnected source components: "
                              << result.graph_error << "\n";
                    result.graph_error.clear();
                }
            };

        FlowGraphPipelineResult pipeline =
            build_flow_graph_pipeline(white_domain, flow_sources, timings);
        clear_allowed_graph_error(pipeline);
        if (!pipeline.rim_error.empty()) {
            throw std::runtime_error(pipeline.rim_error);
        }

        IslandRemovalResult island_removal;
        {
            const TimingMark timing = start_timing();
            const GraphEdgeMaps edge_maps =
                build_graph_edge_maps(pipeline.graph,
                                      pipeline.white_domain.size());
            const IslandObstacleDebugResult island_debug =
                render_island_obstacle_factors(
                    pipeline.white_domain, pipeline.graph,
                    edge_maps.edge_index);
            island_removal =
                remove_high_score_islands(pipeline.white_domain, island_debug);
            std::cout << "Island removal filter:\n"
                      << "  threshold: " << kIslandRemovalScoreThreshold
                      << "\n"
                      << "  scored_islands: "
                      << island_debug.scored_islands.size() << "\n"
                      << "  removed_islands: "
                      << island_removal.removed_islands << "\n"
                      << "  removed_pixels: "
                      << island_removal.removed_pixels << "\n";
            timings.push_back(finish_timing("island_removal_filter", timing));
        }

        if (island_removal.removed_pixels > 0) {
            pipeline = build_flow_graph_pipeline(
                island_removal.white_domain, flow_sources, timings,
                "filtered_");
            clear_allowed_graph_error(pipeline);
            if (!pipeline.rim_error.empty()) {
                throw std::runtime_error(pipeline.rim_error);
            }
        }
        if (!pipeline.graph_error.empty()) {
            throw std::runtime_error(pipeline.graph_error);
        }

        white_domain = pipeline.white_domain;
        binary = pipeline.binary;
        cv::Mat& dt = pipeline.dt;
        SourceRimSkeletonResult& source_rim_result =
            pipeline.source_rim_result;
        SkeletonGraph& graph = pipeline.graph;

        DenseFlowResult dense_flow_result =
            compute_dense_source_flow(
                white_domain, dt, graph,
                source_rim_result.source_rim_ridges, flow_sources,
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
        if (resolved_accepted_sources != nullptr) {
            *resolved_accepted_sources = dense_flow_result.accepted_sources;
        }
        if (resolved_source_edges != nullptr) {
            *resolved_source_edges = dense_flow_result.source_edges;
        }
        if (resolved_seeded_nodes != nullptr) {
            *resolved_seeded_nodes = dense_flow_result.seeded_nodes;
        }
        timings.insert(timings.end(), dense_flow_result.timings.begin(),
                       dense_flow_result.timings.end());

        const cv::Mat flow_gate_weight = compute_flow_gate_weight_image(
            dense_flow_result.tree_dense_flow_greedy_ascent,
            backtrack_distance, local_boost,
            dense_flow_result.flow_gate_regions);
        const cv::Mat flow_gate_basis = normalize_flow_by_regions(
            dense_flow_result.tree_dense_flow_greedy_ascent,
            dense_flow_result.flow_gate_regions);

        if (dense_flow != nullptr) {
            for (int y = 0; y < height; ++y) {
                const float* src_row = flow_gate_weight.ptr<float>(y);
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
        if (gate_basis_flow != nullptr) {
            for (int y = 0; y < height; ++y) {
                const float* src_row = flow_gate_basis.ptr<float>(y);
                std::copy(src_row, src_row + width,
                          gate_basis_flow +
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
        if (island_obstacle_factor_rgb != nullptr) {
            cv::Mat island_rgb;
            cv::cvtColor(dense_flow_result.island_obstacle_factor, island_rgb,
                         cv::COLOR_BGR2RGB);
            island_rgb.convertTo(island_rgb, CV_32FC3);
            for (int y = 0; y < height; ++y) {
                const cv::Vec3f* src_row = island_rgb.ptr<cv::Vec3f>(y);
                float* dst_row =
                    island_obstacle_factor_rgb +
                    static_cast<std::size_t>(y) * width * 3;
                for (int x = 0; x < width; ++x) {
                    dst_row[3 * x + 0] = src_row[x][0];
                    dst_row[3 * x + 1] = src_row[x][1];
                    dst_row[3 * x + 2] = src_row[x][2];
                }
            }
        }
        if (island_removed_mask != nullptr) {
            cv::Mat removed_float;
            island_removal.removed_mask.convertTo(removed_float, CV_32F);
            for (int y = 0; y < height; ++y) {
                const float* src_row = removed_float.ptr<float>(y);
                std::copy(src_row, src_row + width,
                          island_removed_mask +
                              static_cast<std::size_t>(y) * width);
            }
        }
        const auto copy_bgr_float_image = [&](const cv::Mat& bgr_image,
                                              float* dst) {
            if (dst == nullptr) {
                return;
            }
            cv::Mat rgb_image;
            cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);
            rgb_image.convertTo(rgb_image, CV_32FC3);
            for (int y = 0; y < height; ++y) {
                const cv::Vec3f* src_row = rgb_image.ptr<cv::Vec3f>(y);
                float* dst_row =
                    dst + static_cast<std::size_t>(y) * width * 3;
                for (int x = 0; x < width; ++x) {
                    dst_row[3 * x + 0] = src_row[x][0];
                    dst_row[3 * x + 1] = src_row[x][1];
                    dst_row[3 * x + 2] = src_row[x][2];
                }
            }
        };
        copy_bgr_float_image(dense_flow_result.island_flow_passability,
                             island_flow_passability_rgb);
        copy_bgr_float_image(dense_flow_result.island_propagated_edge_flow,
                             island_propagated_edge_flow_rgb);
        copy_bgr_float_image(dense_flow_result.island_bonus_edge_flow,
                             island_bonus_edge_flow_rgb);
        const auto copy_float_image = [&](const cv::Mat& image, float* dst) {
            if (dst == nullptr) {
                return;
            }
            CV_Assert(image.type() == CV_32F);
            for (int y = 0; y < height; ++y) {
                const float* src_row = image.ptr<float>(y);
                std::copy(src_row, src_row + width,
                          dst + static_cast<std::size_t>(y) * width);
            }
        };
        copy_float_image(dense_flow_result.island_tree_dense_flow_no_backtrack,
                         island_tree_dense_no_backtrack);
        copy_float_image(dense_flow_result.island_tree_dense_flow_greedy_ascent,
                         island_tree_dense_greedy_ascent);
        if (source_edge_mask != nullptr) {
            CV_Assert(dense_flow_result.graph_source_edges.type() == CV_8U);
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* src_row =
                    dense_flow_result.graph_source_edges.ptr<std::uint8_t>(y);
                float* dst_row =
                    source_edge_mask + static_cast<std::size_t>(y) * width;
                for (int x = 0; x < width; ++x) {
                    dst_row[x] = src_row[x] != 0 ? 1.0f : 0.0f;
                }
            }
        }
        if (source_component_mask != nullptr) {
            CV_Assert(white_domain.type() == CV_8U);
            for (int y = 0; y < height; ++y) {
                const std::uint8_t* src_row =
                    white_domain.ptr<std::uint8_t>(y);
                float* dst_row =
                    source_component_mask +
                    static_cast<std::size_t>(y) * width;
                for (int x = 0; x < width; ++x) {
                    dst_row[x] = src_row[x] != 0 ? 255.0f : 0.0f;
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
            const float v00 = flow_gate_weight.at<float>(y0, x0);
            const float v10 = flow_gate_weight.at<float>(y0, x1);
            const float v01 = flow_gate_weight.at<float>(y1, x0);
            const float v11 = flow_gate_weight.at<float>(y1, x1);
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
        cv::Mat source_pixel_labels;
        SourceRimSkeletonResult source_rim_result;
        SkeletonGraph graph;
        GraphConnectivityStats graph_stats;
        cv::Mat graph_random_colors;
        cv::Mat graph_edges_random_colors;
        cv::Mat graph_component_colors;
        cv::Mat graph_nodes;
        cv::Mat graph_capacity;
        cv::Mat graph_capacity_gray_bg;
        cv::Mat graph_capacity_normalized;
        cv::Mat graph_capacity_normalized_layer;
        cv::Mat island_removed_mask;
        DenseFlowResult dense_flow_result;
        bool has_dense_flow = false;
        std::string rim_error;
        std::string graph_error;
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
                cv::distanceTransform(white_domain, dt, source_pixel_labels,
                                      cv::DIST_L2, cv::DIST_MASK_5,
                                      cv::DIST_LABEL_PIXEL);
                timings.push_back(finish_timing("labeled_dt", timing));
            }

            cv::Mat rim_label_domain;
            cv::Mat rim_dt;
            cv::Mat rim_source_pixel_labels;
            {
                const TimingMark timing = start_timing();
                rim_label_domain =
                    make_padded_expanded_rim_label_domain(
                        white_domain,
                        args.has_source ? args.source : cv::Point(-1, -1));
                cv::distanceTransform(rim_label_domain, rim_dt,
                                      rim_source_pixel_labels, cv::DIST_L2,
                                      cv::DIST_MASK_5, cv::DIST_LABEL_PIXEL);
                timings.push_back(finish_timing("rim_labeled_dt", timing));
            }

            std::vector<cv::Point> rim_source_points;
            {
                const TimingMark timing = start_timing();
                rim_source_points = label_source_points(
                    rim_label_domain, rim_source_pixel_labels);
                timings.push_back(
                    finish_timing("rim_label_source_points", timing));
            }

            {
                const TimingMark timing = start_timing();
                source_rim_result = source_rim_skeleton(
                    rim_label_domain, rim_source_pixel_labels,
                    rim_source_points);
                source_rim_result = crop_padded_source_rim_result(
                    source_rim_result, white_domain.size());
                timings.push_back(finish_timing("source_rim_skeleton", timing));
                timings.insert(timings.end(),
                               source_rim_result.timings.begin(),
                               source_rim_result.timings.end());
                rim_error = source_rim_result.rim_connectivity_error;
            }

            {
                const TimingMark timing = start_timing();
                source_rim_result.loops_connected =
                    add_valid_frame_to_skeleton(
                        source_rim_result.loops_connected, dt);
                timings.push_back(finish_timing("add_valid_frame", timing));
            }

            {
                const TimingMark timing = start_timing();
                graph = extract_skeleton_graph(
                    source_rim_result.loops_connected, dt);
                timings.push_back(finish_timing("graph_extract", timing));
            }

            {
                const TimingMark timing = start_timing();
                graph_stats = graph_connectivity_stats(graph);
                graph_error = graph_connectivity_error(graph_stats);
                timings.push_back(
                    finish_timing("graph_connectivity_stats", timing));
            }

            island_removed_mask =
                cv::Mat::zeros(white_domain.size(), CV_8U);
            if (args.has_source && rim_error.empty()) {
                const TimingMark timing = start_timing();
                const GraphEdgeMaps edge_maps =
                    build_graph_edge_maps(graph, white_domain.size());
                const IslandObstacleDebugResult island_debug =
                    render_island_obstacle_factors(
                        white_domain, graph, edge_maps.edge_index);
                const IslandRemovalResult island_removal =
                    remove_high_score_islands(white_domain, island_debug);
                island_removed_mask = island_removal.removed_mask;
                std::cout << "Island removal filter:\n"
                          << "  threshold: "
                          << kIslandRemovalScoreThreshold << "\n"
                          << "  scored_islands: "
                          << island_debug.scored_islands.size() << "\n"
                          << "  removed_islands: "
                          << island_removal.removed_islands << "\n"
                          << "  removed_pixels: "
                          << island_removal.removed_pixels << "\n";
                timings.push_back(
                    finish_timing("island_removal_filter", timing));

                if (island_removal.removed_pixels > 0) {
                    FlowGraphPipelineResult filtered =
                        build_flow_graph_pipeline(
                            island_removal.white_domain,
                            std::vector<cv::Point>{args.source}, timings,
                            "filtered_");
                    white_domain = filtered.white_domain;
                    binary = filtered.binary;
                    dt = filtered.dt;
                    source_pixel_labels = filtered.source_pixel_labels;
                    source_rim_result = filtered.source_rim_result;
                    graph = filtered.graph;
                    graph_stats = filtered.graph_stats;
                    rim_error = filtered.rim_error;
                    graph_error = filtered.graph_error;
                }
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
                graph_component_colors =
                    render_graph_component_colors(graph, binary.size());
                timings.push_back(
                    finish_timing("graph_component_color_render", timing));
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
                graph_capacity_normalized =
                    render_graph_capacity_normalized_u8(graph, binary.size());
                graph_capacity_normalized_layer =
                    render_graph_capacity_normalized_float(graph,
                                                           binary.size());
                timings.push_back(
                    finish_timing("graph_capacity_render", timing));
            }

            has_dense_flow = false;
            if (args.has_source && rim_error.empty() && graph_error.empty()) {
                dense_flow_result =
                    compute_dense_source_flow(
                        white_domain, dt, graph,
                        source_rim_result.source_rim_ridges,
                        std::vector<cv::Point>{args.source}, args.grid_step,
                        args.backtrack_distance, args.write_outputs);
                {
                    const TimingMark timing = start_timing();
                    dense_flow_result.flow_gate_weight =
                        compute_flow_gate_weight_image(
                            dense_flow_result.tree_dense_flow_greedy_ascent,
                            args.backtrack_distance, args.local_boost,
                            dense_flow_result.flow_gate_regions);
                    timings.push_back(
                        finish_timing("flow_gate_weight", timing));
                }
                timings.insert(timings.end(), dense_flow_result.timings.begin(),
                               dense_flow_result.timings.end());
                has_dense_flow = true;
            }

        }

        const std::string stem = args.input.stem().string();
        if (args.write_outputs) {
        {
            const TimingMark timing = start_timing();
            write_image(workdir / (stem + "_binary.tif"), binary);
            write_image(workdir / (stem + "_dt.tif"), dt);
            write_image(workdir / (stem + "_island_removed_mask.tif"),
                        island_removed_mask);
            write_image(workdir / (stem + "_source_components.tif"),
                        white_domain);
            write_image(workdir / (stem + "_source_rim_ridges.tif"),
                        source_rim_result.source_rim_ridges);
            write_image(workdir / (stem + "_source_rim_distance.tif"),
                        source_rim_result.source_rim_distance);
            write_image(workdir / (stem + "_source_rim_arc.tif"),
                        source_rim_result.source_rim_arc);
            write_image(workdir / (stem + "_source_rim_arc_skeleton.tif"),
                        source_rim_result.source_rim_arc_skeleton);
            write_image(workdir / (stem + "_source_rim_skeleton.tif"),
                        source_rim_result.loops_connected);
            write_image(workdir / (stem + "_graph_random_edges.tif"),
                        graph_random_colors);
            write_image(workdir / (stem + "_graph_edges_random.tif"),
                        graph_edges_random_colors);
            write_image(workdir / (stem + "_graph_components_random.tif"),
                        graph_component_colors);
            write_image(workdir / (stem + "_graph_nodes.tif"),
                        graph_nodes);
            write_image(workdir / (stem + "_graph_capacity.tif"),
                        graph_capacity);
            write_image(workdir / (stem + "_graph_capacity_gray_bg.tif"),
                        graph_capacity_gray_bg);
            write_image(workdir / (stem + "_graph_capacity_normalized.tif"),
                        graph_capacity_normalized);
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
            write_image(workdir /
                            (stem + "_tree_dense_flow_no_backtrack.tif"),
                        dense_flow_result.tree_dense_flow_no_backtrack);
            write_image(workdir / (stem + "_nearest_graph_flow.tif"),
                        dense_flow_result.nearest_graph_flow);
            write_image(workdir /
                            (stem + "_tree_dense_flow_greedy_ascent.tif"),
                        dense_flow_result.tree_dense_flow_greedy_ascent);
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
            if (!dense_flow_result.flow_gate_regions.empty()) {
                double max_region_label = 0.0;
                cv::minMaxLoc(dense_flow_result.flow_gate_regions, nullptr,
                              &max_region_label);
                write_image(workdir / (stem + "_flow_gate_regions.tif"),
                            labels_to_u16(
                                dense_flow_result.flow_gate_regions,
                                static_cast<int>(max_region_label)));
            }
            write_image(workdir / (stem + "_island_obstacle_factor.tif"),
                        dense_flow_result.island_obstacle_factor);
            write_image(workdir / (stem + "_island_flow_passability.tif"),
                        dense_flow_result.island_flow_passability);
            write_image(workdir /
                            (stem + "_island_propagated_edge_flow.tif"),
                        dense_flow_result.island_propagated_edge_flow);
            write_image(workdir / (stem + "_island_bonus_edge_flow.tif"),
                        dense_flow_result.island_bonus_edge_flow);
            write_image(workdir /
                            (stem +
                             "_island_tree_dense_flow_no_backtrack.tif"),
                        dense_flow_result
                            .island_tree_dense_flow_no_backtrack);
            write_image(workdir /
                            (stem +
                             "_island_tree_dense_flow_greedy_ascent.tif"),
                        dense_flow_result
                            .island_tree_dense_flow_greedy_ascent);
            write_image(workdir / (stem + "_flow_gate_weight.tif"),
                        dense_flow_result.flow_gate_weight);
            timings.push_back(finish_timing("dense_flow_write", timing));
        }

        std::vector<NamedLayer> layered_tiff = {
            {"binary_threshold", to_float_layer(binary)},
            {"dt", to_float_layer(dt)},
            {"island_removed_mask", to_float_layer(island_removed_mask)},
            {"source_components", to_float_layer(white_domain)},
            {"loops_connected",
             to_float_layer(source_rim_result.loops_connected)},
            {"source_rim_ridges",
             to_float_layer(source_rim_result.source_rim_ridges)},
            {"source_rim_distance",
             to_float_layer(source_rim_result.source_rim_distance)},
            {"source_rim_arc",
             to_float_layer(source_rim_result.source_rim_arc)},
            {"source_rim_arc_skeleton",
             to_float_layer(source_rim_result.source_rim_arc_skeleton)},
            {"graph_random_edges", to_float_layer(graph_random_colors)},
            {"graph_edges_random", to_float_layer(graph_edges_random_colors)},
            {"graph_components_random",
             to_float_layer(graph_component_colors)},
            {"graph_nodes", to_float_layer(graph_nodes)},
            {"graph_capacity", to_float_layer(graph_capacity)},
            {"graph_capacity_gray_bg", to_float_layer(graph_capacity_gray_bg)},
            {"graph_capacity_normalized", graph_capacity_normalized_layer},
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
                {"tree_dense_flow_no_backtrack",
                 to_float_layer(
                     dense_flow_result.tree_dense_flow_no_backtrack)});
            layered_tiff.push_back(
                {"nearest_graph_flow",
                 to_float_layer(dense_flow_result.nearest_graph_flow)});
            layered_tiff.push_back(
                {"tree_dense_flow_greedy_ascent",
                 to_float_layer(
                     dense_flow_result.tree_dense_flow_greedy_ascent)});
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
            if (!dense_flow_result.flow_gate_regions.empty()) {
                cv::Mat flow_gate_regions_float;
                dense_flow_result.flow_gate_regions.convertTo(
                    flow_gate_regions_float, CV_32F);
                layered_tiff.push_back(
                    {"flow_gate_regions",
                     to_float_layer(flow_gate_regions_float)});
            }
            layered_tiff.push_back(
                {"island_obstacle_factor",
                 to_float_layer(dense_flow_result.island_obstacle_factor)});
            layered_tiff.push_back(
                {"island_flow_passability",
                 to_float_layer(dense_flow_result.island_flow_passability)});
            layered_tiff.push_back(
                {"island_propagated_edge_flow",
                 to_float_layer(
                     dense_flow_result.island_propagated_edge_flow)});
            layered_tiff.push_back(
                {"island_bonus_edge_flow",
                 to_float_layer(dense_flow_result.island_bonus_edge_flow)});
            layered_tiff.push_back(
                {"island_tree_dense_flow_no_backtrack",
                 to_float_layer(
                     dense_flow_result
                         .island_tree_dense_flow_no_backtrack)});
            layered_tiff.push_back(
                {"island_tree_dense_flow_greedy_ascent",
                 to_float_layer(
                     dense_flow_result
                         .island_tree_dense_flow_greedy_ascent)});
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
        std::cout << "Rim connectivity:\n"
                  << "  rim_assignments: "
                  << (rim_error.empty() ? "ok" : "failed") << "\n";
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
                  << graph_stats.missing_pixels << "\n"
                  << "  adjacent_component_repairs: "
                  << graph_stats.adjacent_component_repairs << "\n"
                  << "  adjacent_component_contacts: "
                  << graph_stats.adjacent_component_contacts << "\n"
                  << "  pruned_graph_components: "
                  << graph_stats.pruned_graph_components << "\n"
                  << "  pruned_graph_nodes: "
                  << graph_stats.pruned_graph_nodes << "\n"
                  << "  pruned_graph_edges: "
                  << graph_stats.pruned_graph_edges << "\n";
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
        if (!rim_error.empty() || !graph_error.empty()) {
            std::cout << "error: "
                      << (!rim_error.empty() ? rim_error : graph_error)
                      << "\n";
            return 2;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
#endif
