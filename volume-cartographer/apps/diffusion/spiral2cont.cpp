#include "spiral2cont.hpp"
#include "support.hpp"
#include "spiral_ceres.hpp"
#include "spiral.hpp"
#include "spiral2.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>
#include <filesystem>
#include <chrono>
#include <unordered_set>
#include <iomanip>
#include <random>
#include <atomic>
#include <omp.h>
#include <fstream>

#include <boost/graph/adjacency_list.hpp>
#include <ceres/ceres.h>
#include <opencv2/ximgproc.hpp>
#include <nlohmann/json.hpp>

#include "vc/core/util/GridStore.hpp"
#include "vc/ui/VCCollection.hpp"

void optimize_spiral_trace(
    std::vector<SpiralPoint>& spiral_trace,
    const vc::core::util::GridStore& normal_grid,
    double spiral_step,
    cv::VideoWriter& video_writer,
    cv::Mat& vis_frame
);

std::pair<SkeletonGraph, cv::Mat> generate_skeleton_graph(const cv::Mat& binary_slice, const po::variables_map& vm);
void populate_normal_grid(const SkeletonGraph& graph, vc::core::util::GridStore& normal_grid, double spiral_step);
void visualize_spiral(cv::Mat& vis, const std::vector<SpiralPoint>& spiral, const cv::Scalar& color, const cv::Scalar& endpoint_color, bool draw_endpoints);
int spiral2cont_main(
    const cv::Mat& slice_mat,
    const std::optional<cv::Vec3f>& umbilicus_point,
    const fs::path& output_path,
    const po::variables_map& vm
) {
    if (!vm.count("json-in") || !vm.count("collection-name") || !vm.count("start-winding") || !vm.count("end-winding")) {
        std::cerr << "Error: For spiral2cont mode, --json-in, --collection-name, --start-winding, and --end-winding are required." << std::endl;
        return 1;
    }

    std::string json_in_path = vm["json-in"].as<std::string>();
    std::string collection_name = vm["collection-name"].as<std::string>();
    double start_winding = vm["start-winding"].as<double>();
    double end_winding = vm["end-winding"].as<double>();

    std::ifstream i(json_in_path);
    nlohmann::json json_input;
    i >> json_input;

    nlohmann::json json_collection;
    bool collection_found = false;
    for (const auto& col : json_input) {
        if (col["name"] == collection_name) {
            json_collection = col;
            collection_found = true;
            break;
        }
    }

    if (!collection_found) {
        std::cerr << "Error: Collection '" << collection_name << "' not found in JSON file." << std::endl;
        return 1;
    }

    std::vector<std::vector<SpiralPoint>> all_spiral_wraps = json_collection["spirals"].get<std::vector<std::vector<SpiralPoint>>>();

    std::vector<SpiralPoint> target_spiral;
    for (const auto& spiral : all_spiral_wraps) {
        if (spiral.empty() || spiral.back().winding < start_winding || spiral.front().winding > end_winding) {
            continue;
        }

        bool started = false;
        for (const auto& p : spiral) {
            if (p.winding >= start_winding) {
                started = true;
            }
            if (started) {
                target_spiral.push_back(p);
            }
            if (p.winding >= end_winding) {
                break;
            }
        }

        if (!target_spiral.empty()) {
            break;
        }
    }

    if (target_spiral.empty()) {
        std::cerr << "Error: Could not find the specified spiral wrap." << std::endl;
        return 1;
    }
    else
        std::cout << "tgt spiral size: " << target_spiral.size() << std::endl;

    double spiral_step = vm["spiral-step"].as<double>();
    cv::Point umbilicus_p(
        static_cast<int>(std::round((*umbilicus_point)[0])),
                        static_cast<int>(std::round((*umbilicus_point)[1]))
    );

    cv::Mat binary_slice = slice_mat > 0;
    auto [skeleton_graph, skeleton_id_img] = generate_skeleton_graph(binary_slice, vm);
    vc::core::util::GridStore normal_grid(
        cv::Rect(0, 0, slice_mat.cols, slice_mat.rows), 32
    );
    populate_normal_grid(skeleton_graph, normal_grid, spiral_step);

    cv::Mat vis_frame = cv::Mat::zeros(slice_mat.size(), CV_8UC3);
    cv::VideoWriter video_writer; // Dummy writer, not used for optimization visualization in this mode yet

    optimize_spiral_trace(target_spiral, normal_grid, spiral_step, video_writer, vis_frame);

    cv::Mat final_vis = cv::Mat::zeros(slice_mat.size(), CV_8UC3);
    visualize_spiral(final_vis, target_spiral, cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), false);

    cv::Mat slice_bgr;
    cv::cvtColor(slice_mat, slice_bgr, cv::COLOR_GRAY2BGR);
    cv::addWeighted(slice_bgr, 0.5, final_vis, 0.5, 0.0, final_vis);

    cv::imwrite(output_path.string(), final_vis);
    std::cout << "Saved optimized spiral to " << output_path << std::endl;
    return 0;
}
