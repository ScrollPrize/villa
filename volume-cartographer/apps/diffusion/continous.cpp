#include "continous.hpp"
#include "support.hpp"
#include "discrete.hpp"

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

#include <vc/ui/VCCollection.hpp>

const float sheet_step_weight = 2.0f;

struct SheetConstraintRay {
    cv::Point2f dir;
    std::vector<std::pair<cv::Point, cv::Point>> constraints;
};

void calculate_sheet_distance_constraints(
    const cv::Mat& slice_mat,
    const cv::Point& umbilicus,
    float ray_step_dist,
    std::vector<SheetConstraintRay>& constraint_rays,
    std::unordered_map<cv::Point, cv::Point, PointHash>& constraints
);

void visualize_sheet_distance_constraints(
    const cv::Mat& slice_mat,
    const std::unordered_map<cv::Point, cv::Point, PointHash>& constraints,
    const fs::path& vis_path
) {
    cv::Mat viz;
    cv::cvtColor(slice_mat, viz, cv::COLOR_GRAY2BGR);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 255);

    for (const auto& [p_outer, p_inner] : constraints) {
        cv::Scalar color(distrib(gen), distrib(gen), distrib(gen));
        cv::line(viz, p_inner, p_outer, color, 1);
        cv::circle(viz, p_outer, 3, color, -1);
    }

    if (cv::imwrite(vis_path.string(), viz)) {
        std::cout << "Saved sheet constraint visualization to " << vis_path << std::endl;
    } else {
        std::cerr << "Error: Failed to write sheet constraint visualization to " << vis_path << std::endl;
    }
}


int continous_main(
    const cv::Mat& slice_mat,
    const VCCollection& point_collection,
    const std::optional<cv::Vec3f>& umbilicus_point,
    const std::string& umbilicus_set_name,
    int iterations,
    float ray_step_dist,
    const fs::path& output_path,
    const fs::path& conflicts_path,
    const cv::Mat& mask,
    const po::variables_map& vm
) {
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << std::endl;
    cv::Mat winding(slice_mat.size(), CV_32F, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    if (!mask.empty()) {
        for (int y = 0; y < slice_mat.rows; ++y) {
            for (int x = 0; x < slice_mat.cols; ++x) {
                if (mask.at<uint8_t>(y, x) > 0) {
                    winding.at<float>(y, x) = 0.0f;
                }
            }
        }
    } else {
        winding = cv::Scalar(0.0f);
    }

    int umb_slice_x = static_cast<int>(std::round((*umbilicus_point)[0]));
    int umb_slice_y = static_cast<int>(std::round((*umbilicus_point)[1]));
    cv::Point umbilicus_p_int(umb_slice_x, umb_slice_y);
    cv::Point2f umbilicus_p_float(umb_slice_x, umb_slice_y);

    const float radial_gradient = 0.02f;

    auto start_diffusion = std::chrono::high_resolution_clock::now();

    std::unordered_map<cv::Point, cv::Point, PointHash> constraints;
    std::vector<SheetConstraintRay> constraint_rays;
    if (ray_step_dist > 0) {
        calculate_sheet_distance_constraints(slice_mat, umbilicus_p_int, ray_step_dist, constraint_rays, constraints);
        if (vm.count("debug")) {
            visualize_sheet_distance_constraints(slice_mat, constraints, "sheet_constr_vis.tif");
        }
    }

    std::unordered_map<cv::Point, cv::Point, PointHash> inner_constraints;
    std::unordered_map<cv::Point, cv::Point, PointHash> outer_constraints;
    for (const auto& [outer, inner] : constraints) {
        inner_constraints[inner] = outer;
        outer_constraints[outer] = inner;
    }

    cv::Mat processed_mask(slice_mat.size(), CV_8U, cv::Scalar(0));
    std::vector<cv::Point> active_pixels;
    active_pixels.push_back(umbilicus_p_int);
    processed_mask.at<uint8_t>(umbilicus_p_int) = 1;

    size_t head = 0;
    auto last_report_time = std::chrono::high_resolution_clock::now();
    std::cout << "Starting wavefront propagation..." << std::endl;

    while(head < active_pixels.size()) {
        cv::Point p = active_pixels[head++];

        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - last_report_time;
        if (elapsed.count() >= 1.0) {
            float progress = 100.0f * static_cast<float>(head) / active_pixels.size();
            std::cout << "  Propagation progress: " << std::fixed << std::setprecision(2) << progress << "% (" << head << " / " << active_pixels.size() << " pixels processed).\r" << std::flush;
            last_report_time = now;
        }

        int dx[] = {0, 0, 1, -1};
        int dy[] = {1, -1, 0, 0};
        for (int j = 0; j < 4; ++j) {
            cv::Point n(p.x + dx[j], p.y + dy[j]);

            if (n.x < 0 || n.x >= winding.cols || n.y < 0 || n.y >= winding.rows || processed_mask.at<uint8_t>(n) || std::isnan(winding.at<float>(n))) {
                continue;
            }

            float total_constraint_value = 0;
            float total_weight = 0;

            // Neighborhood constraints
            for (int k = 0; k < 4; ++k) {
                cv::Point processed_neighbor(n.x + dx[k], n.y + dy[k]);
                if (processed_neighbor.x >= 0 && processed_neighbor.x < winding.cols &&
                    processed_neighbor.y >= 0 && processed_neighbor.y < winding.rows &&
                    processed_mask.at<uint8_t>(processed_neighbor)) {

                    float neighbor_winding = winding.at<float>(processed_neighbor);
                if (std::isnan(neighbor_winding)) continue;

                if (processed_neighbor.y >= umb_slice_y && n.y == processed_neighbor.y) {
                    if (processed_neighbor.x < umb_slice_x && n.x >= umb_slice_x) neighbor_winding--;
                    else if (processed_neighbor.x >= umb_slice_x && n.x < umb_slice_x) neighbor_winding++;
                }

                cv::Point2f neighbor_to_umb = cv::Point2f(processed_neighbor.x, processed_neighbor.y) - umbilicus_p_float;
                double norm_neighbor_to_umb = cv::norm(neighbor_to_umb);
                cv::Point2f u_radial = (norm_neighbor_to_umb > 1e-6) ? neighbor_to_umb / norm_neighbor_to_umb : cv::Point2f(0, 0);

                cv::Point2f neighbor_to_n_vec = cv::Point2f(n.x - processed_neighbor.x, n.y - processed_neighbor.y);

                float radial_dist = neighbor_to_n_vec.dot(u_radial);

                float expected_winding = neighbor_winding + radial_dist * radial_gradient;

                total_constraint_value += expected_winding;
                total_weight += 1.0f;
                    }
            }

            // Sheet distance constraints
            // Check if 'n' is an inner point of a constraint
            if (inner_constraints.count(n)) {
                cv::Point outer_p = inner_constraints.at(n);
                if (processed_mask.at<uint8_t>(outer_p) && !std::isnan(winding.at<float>(outer_p))) {
                    float outer_winding = winding.at<float>(outer_p);
                    total_constraint_value += (outer_winding - 1.0f) * sheet_step_weight;
                    total_weight += sheet_step_weight;
                }
            }
            // Check if 'n' is an outer point of a constraint
            if (outer_constraints.count(n)) {
                cv::Point inner_p = outer_constraints.at(n);
                if (processed_mask.at<uint8_t>(inner_p) && !std::isnan(winding.at<float>(inner_p))) {
                    float inner_winding = winding.at<float>(inner_p);
                    total_constraint_value += (inner_winding + 1.0f) * sheet_step_weight;
                    total_weight += sheet_step_weight;
                }
            }

            if (total_weight > 0) {
                float new_winding = total_constraint_value / total_weight;
                winding.at<float>(n) = new_winding;
                processed_mask.at<uint8_t>(n) = 1;
                active_pixels.push_back(n);
            }
        }
    }

    auto end_wavefront = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> wavefront_time = end_wavefront - start_diffusion;
    std::cout << "Wavefront propagation finished after " << wavefront_time.count() << " s." << std::endl;

    float damping_factor = vm["dampening"].as<float>();

    if (iterations > 0) {
        std::cout << "Starting iterative diffusion for " << iterations << " iterations with damping factor " << damping_factor << "..." << std::endl;
        auto start_iterative = std::chrono::high_resolution_clock::now();

        cv::Mat winding_prev;
        std::atomic<int> processed_rows(0);
        auto last_iter_report_time = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            winding_prev = winding.clone();
            processed_rows = 0;

            #pragma omp parallel for
            for (int y = 0; y < winding.rows; ++y) {
                for (int x = 0; x < winding.cols; ++x) {
                    if (std::isnan(winding_prev.at<float>(y, x))) continue;

                    float total_constraint_value = 0;
                    float total_weight = 0;
                    cv::Point n(x, y);

                    // Neighborhood constraints
                    int dx[] = {0, 0, 1, -1};
                    int dy[] = {1, -1, 0, 0};
                    for (int k = 0; k < 4; ++k) {
                        cv::Point neighbor(n.x + dx[k], n.y + dy[k]);
                        if (neighbor.x >= 0 && neighbor.x < winding.cols && neighbor.y >= 0 && neighbor.y < winding.rows) {
                            float neighbor_winding = winding_prev.at<float>(neighbor);
                            if (std::isnan(neighbor_winding)) continue;

                            if (neighbor.y >= umb_slice_y && n.y == neighbor.y) {
                                if (neighbor.x < umb_slice_x && n.x >= umb_slice_x) neighbor_winding--;
                                else if (neighbor.x >= umb_slice_x && n.x < umb_slice_x) neighbor_winding++;
                            }

                            total_constraint_value += neighbor_winding;
                            total_weight += 1.0f;
                        }
                    }

                    // Sheet distance constraints
                    if (inner_constraints.count(n)) {
                        cv::Point outer_p = inner_constraints.at(n);
                        if (!std::isnan(winding_prev.at<float>(outer_p))) {
                            total_constraint_value += (winding_prev.at<float>(outer_p) - 1.0f) * sheet_step_weight;
                            total_weight += sheet_step_weight;
                        }
                    }
                    if (outer_constraints.count(n)) {
                        cv::Point inner_p = outer_constraints.at(n);
                        if (!std::isnan(winding_prev.at<float>(inner_p))) {
                            total_constraint_value += (winding_prev.at<float>(inner_p) + 1.0f) * sheet_step_weight;
                            total_weight += sheet_step_weight;
                        }
                    }

                    if (total_weight > 0) {
                        float new_winding = total_constraint_value / total_weight;
                        winding.at<float>(n) =  damping_factor * winding_prev.at<float>(n) + (1.0f - damping_factor) * new_winding;
                    }
                }

                #pragma omp critical
                processed_rows++;
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = now - last_iter_report_time;
                if (elapsed.count() >= 1.0) {
                    #pragma omp critical
                    {
                        now = std::chrono::high_resolution_clock::now();
                        elapsed = now - last_iter_report_time;
                        if (elapsed.count() >= 1.0) {
                            float progress = 100.0f * static_cast<float>(processed_rows) / winding.rows;
                            std::cout << "  Iteration " << i + 1 << "/" << iterations
                            << ", processing " << std::fixed << std::setprecision(2) << progress << "%\r" << std::flush;
                            last_iter_report_time = now;
                        }
                    }
                }
            }
            std::cout << "  Iteration " << i + 1 << "/" << iterations << " complete.                                  " << std::endl;
        }

        auto end_iterative = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> iterative_time = end_iterative - start_iterative;
        std::cout << "Iterative diffusion finished after " << iterative_time.count() << " s." << std::endl;
    }

    // Visualization
    cv::imwrite("wind_float.tif", winding);
    std::cout << "Saved floating point winding data to wind_float.tif" << std::endl;

    cv::Mat viz;
    cv::Mat is_nan_mask(winding.size(), CV_8U);
    for(int y = 0; y < winding.rows; ++y) {
        for(int x = 0; x < winding.cols; ++x) {
            float val = winding.at<float>(y, x);
            is_nan_mask.at<uint8_t>(y, x) = (std::isnan(val) || std::isinf(val)) ? 0 : 255;
        }
    }
    cv::normalize(winding, viz, 0, 255, cv::NORM_MINMAX, CV_8U, is_nan_mask);
    cv::applyColorMap(viz, viz, cv::COLORMAP_JET);

    if (!cv::imwrite(output_path.string(), viz)) {
        std::cerr << "Error: Failed to write output image to " << output_path << std::endl;
    } else {
        std::cout << "Saved visualization to " << output_path << std::endl;
    }

    cv::Mat spiral_viz(winding.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < winding.rows - 1; ++y) {
        for (int x = 0; x < winding.cols - 1; ++x) {
            float w = winding.at<float>(y, x);

            int dx[] = {1, 0, 1, 1};
            int dy[] = {0, 1, 1, -1};

            for (int i = 0; i < 4; ++i) {
                int nx = x + dx[i];
                int ny = y + dy[i];

                if (nx >= 0 && nx < winding.cols && ny >= 0 && ny < winding.rows) {
                    float nw = winding.at<float>(ny, nx);

                    if (std::floor(w) != std::floor(nw)) {
                        spiral_viz.at<uint8_t>(y, x) = 255;
                        spiral_viz.at<uint8_t>(ny, nx) = 255;
                    }
                }
            }
        }
    }
    if (vm.count("debug")) {
        cv::imwrite("spiral.tif", spiral_viz);
    }
    return 0;
}
