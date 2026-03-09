#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include "vc/core/util/Thinning.hpp"

namespace fs = std::filesystem;
namespace po = boost::program_options;

int main(int argc, char** argv)
{
    po::options_description desc("vc_thinning_bench options");
    desc.add_options()
        ("help,h", "Show help")
        ("input", po::value<std::string>(), "Input binary image path")
        ("mode", po::value<std::string>()->default_value("traces-only"), "Benchmark mode: full or traces-only")
        ("iterations", po::value<int>()->default_value(20), "Iterations")
        ("metrics-json", po::value<std::string>(), "Write metrics json");

    po::positional_options_description pos;
    pos.add("input", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
        if (vm.count("help") || !vm.count("input")) {
            std::cout << desc << "\n";
            return 0;
        }
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << "\n\n" << desc << "\n";
        return 1;
    }

    const fs::path input_path = vm["input"].as<std::string>();
    const std::string mode = vm["mode"].as<std::string>();
    const int iterations = vm["iterations"].as<int>();
    if (iterations <= 0) {
        std::cerr << "--iterations must be > 0\n";
        return 1;
    }
    if (mode != "full" && mode != "traces-only") {
        std::cerr << "--mode must be full or traces-only\n";
        return 1;
    }

    cv::Mat input = cv::imread(input_path.string(), cv::IMREAD_GRAYSCALE);
    if (input.empty()) {
        std::cerr << "failed to read input image: " << input_path << "\n";
        return 1;
    }

    ThinningStats aggregate;
    std::vector<std::vector<cv::Point>> traces;
    cv::Mat output;

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (mode == "full") {
            customThinning(input, output, &traces, &aggregate);
        } else {
            customThinningTraceOnly(input, traces, &aggregate);
        }
    }
    const double total_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();

    nlohmann::json metrics;
    metrics["input"] = input_path.string();
    metrics["mode"] = mode;
    metrics["iterations"] = iterations;
    metrics["total_seconds"] = total_seconds;
    metrics["avg_seconds"] = total_seconds / static_cast<double>(iterations);
    metrics["distance_transform_seconds"] = aggregate.distanceTransformSeconds;
    metrics["seed_detection_seconds"] = aggregate.seedDetectionSeconds;
    metrics["trace_paths_seconds"] = aggregate.tracePathsSeconds;
    metrics["avg_distance_transform_seconds"] = aggregate.distanceTransformSeconds / static_cast<double>(iterations);
    metrics["avg_seed_detection_seconds"] = aggregate.seedDetectionSeconds / static_cast<double>(iterations);
    metrics["avg_trace_paths_seconds"] = aggregate.tracePathsSeconds / static_cast<double>(iterations);
    metrics["seed_count_total"] = aggregate.seedCount;
    metrics["trace_count_total"] = aggregate.traceCount;
    metrics["trace_steps_total"] = aggregate.traceSteps;
    metrics["candidate_evaluations_total"] = aggregate.candidateEvaluations;
    metrics["trace_count_last"] = traces.size();
    metrics["nonzero_output_last"] = output.empty() ? 0 : cv::countNonZero(output);

    if (vm.count("metrics-json")) {
        std::ofstream out(vm["metrics-json"].as<std::string>());
        out << metrics.dump(2) << "\n";
    }

    std::cout << metrics.dump(2) << "\n";
    return 0;
}
