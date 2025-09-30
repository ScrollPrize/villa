#include "spiral.hpp"
#include "support.hpp"
#include "spiral_ceres.hpp"

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

#include <boost/graph/adjacency_list.hpp>
#include <ceres/ceres.h>
#include <opencv2/ximgproc.hpp>

#include "vc/core/util/xtensor_include.hpp"
#include XTENSORINCLUDE(containers, xarray.hpp)
#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/common.hxx"
#include "z5/multiarray/xtensor_access.hxx"

#include "vc/core/util/GridStore.hpp"
#include "vc/ui/VCCollection.hpp"


using SkeletonGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, SkeletonVertex, SkeletonEdge>;

void generate_optimized_spiral(
    std::vector<SpiralPoint>& all_points,
    std::vector<std::vector<SpiralPoint>>& points_per_revolution,
    double starting_diameter,
    double end_diameter,
    double spiral_step,
    int revolutions,
    const cv::Point& umbilicus,
    const vc::core::util::GridStore& normal_grid,
    const cv::Size& slice_size,
    bool debug
);

void visualize_spiral(
    const std::vector<SpiralPoint>& all_points,
    const cv::Size& slice_size,
    const fs::path& output_path,
    const cv::Scalar& color,
    const std::vector<SheetConstraintRay>& constraint_rays,
    bool draw_neighbors,
    bool draw_indices
);

std::pair<SkeletonGraph, cv::Mat> generate_skeleton_graph(const cv::Mat& binary_slice, const po::variables_map& vm);
void populate_normal_grid(const SkeletonGraph& graph, vc::core::util::GridStore& normal_grid, double spiral_step);
void visualize_normal_grid(const vc::core::util::GridStore& normal_grid, const cv::Size& size, const std::string& path);

void generate_simple_spiral(
    std::vector<std::vector<SpiralPoint>>& points_per_revolution,
    double starting_diameter,
    double end_diameter,
    double spiral_step,
    int revolutions,
    const cv::Point& umbilicus
) {
    std::cout << "Generating simple spiral..." << std::endl;
    double start_radius = starting_diameter / 2.0;
    double end_radius = end_diameter / 2.0;

    if (spiral_step <= 0) {
        std::cerr << "Error: spiral-step must be positive." << std::endl;
        return;
    }

    for (int i = 0; i < revolutions; ++i) {
        double radius_low = start_radius + (end_radius - start_radius) * (double)i / revolutions;
        double radius_high = start_radius + (end_radius - start_radius) * (double)(i + 1) / revolutions;
        double avg_radius = (radius_low + radius_high) / 2.0;
        int n_points = static_cast<int>(2 * CV_PI * avg_radius / spiral_step);

        for (int j = 0; j < n_points; ++j) {
            double angle = 2 * CV_PI * (double)j / n_points;
            double radius = radius_low + (radius_high - radius_low) * (double)j / n_points;
            SpiralPoint sp;
            sp.pos[0] = umbilicus.x + radius * std::cos(angle);
            sp.pos[1] = umbilicus.y + radius * std::sin(angle);
            sp.pos[2] = 0;
            sp.winding = (double)i + (double)j / n_points;
            sp.dist_low = radius - (start_radius + (end_radius - start_radius) * (double)(i - 1) / revolutions);
            sp.dist_high = (start_radius + (end_radius - start_radius) * (double)(i + 1) / revolutions) - radius;
            points_per_revolution[i].push_back(sp);
        }
    }
}

void run_spiral_generation(
    const cv::Mat& slice_mat,
    double starting_diameter,
    double end_diameter,
    double spiral_step,
    int revolutions,
    const cv::Point& umbilicus,
    const cv::Size& slice_size,
    const fs::path& output_path,
    float ray_step_dist,
    const po::variables_map& vm
) {
    std::vector<SpiralPoint> all_points;
    std::vector<std::vector<SpiralPoint>> points_per_revolution(revolutions);

    // --- Sheet Distance Constraints ---
    std::vector<SheetConstraintRay> constraint_rays;
    // if (ray_step_dist > 0) {
    //     std::unordered_map<cv::Point, cv::Point, PointHash> constraints;
    //     calculate_sheet_distance_constraints(slice_mat, umbilicus, ray_step_dist, constraint_rays, constraints);
    //     if (vm.count("debug")) {
    //         visualize_sheet_distance_constraints(slice_mat, constraints, "sheet_constr_vis_spiral.tif");
    //     }
    // }

    cv::Mat binary_slice = slice_mat > 0;
    auto [skeleton_graph, skeleton_id_img] = generate_skeleton_graph(binary_slice, vm);
    vc::core::util::GridStore normal_grid(
        cv::Rect(0, 0, slice_mat.cols, slice_mat.rows), 32
    );
    populate_normal_grid(skeleton_graph, normal_grid, spiral_step);

    if (vm.count("debug")) {
        cv::Mat skeleton_img;
        cv::ximgproc::thinning(binary_slice, skeleton_img, cv::ximgproc::THINNING_GUOHALL);
        cv::imwrite("skeleton.tif", skeleton_img);

        cv::Mat skeleton_ids_32f;
        skeleton_id_img.convertTo(skeleton_ids_32f, CV_32F);
        cv::imwrite("skeleton_ids.tif", skeleton_ids_32f);

        visualize_normal_grid(normal_grid, slice_mat.size(), "normal_constraints_vis.tif");
    }

    if (vm["no-optimized-spiral"].as<bool>()) {
        generate_simple_spiral(points_per_revolution, starting_diameter, end_diameter, spiral_step, revolutions, umbilicus);
    } else {
        generate_optimized_spiral(all_points, points_per_revolution, starting_diameter, end_diameter, spiral_step, revolutions, umbilicus, normal_grid, slice_size, vm.count("debug"));
    }

    // Flatten the points and find neighbors
    if (vm["no-optimized-spiral"].as<bool>()) {
        all_points.clear();
    }
    std::vector<int> revolution_start_indices(revolutions + 1, 0);
    for(int i = 0; i < revolutions; ++i) {
        revolution_start_indices[i+1] = revolution_start_indices[i] + points_per_revolution[i].size();
        for(const auto& sp : points_per_revolution[i]) {
            all_points.push_back(sp);
        }
    }

    std::cout << "Generated " << all_points.size() << " spiral points. Finding neighbors..." << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < revolutions; ++i) {
        for (size_t j = 0; j < points_per_revolution[i].size(); ++j) {
            int current_point_idx = revolution_start_indices[i] + j;
            const auto& current_point = all_points[current_point_idx];

            // Find neighbors in revolution below
            if (i > 0) {
                std::vector<std::pair<double, int>> dists;
                std::vector<std::pair<double, int>> winding_dists;
                for (size_t k = 0; k < all_points.size(); ++k) {
                    if (k == current_point_idx) continue;
                    double winding_diff = std::abs(all_points[k].winding - (current_point.winding - 1.0));
                    winding_dists.emplace_back(winding_diff, k);
                }
                std::partial_sort(winding_dists.begin(), winding_dists.begin() + 3, winding_dists.end());
                for(int k = 0; k < 3; ++k) {
                    dists.emplace_back(cv::norm(current_point.pos - all_points[winding_dists[k].second].pos), winding_dists[k].second);
                }
                std::partial_sort(dists.begin(), dists.begin() + std::min(3, (int)dists.size()), dists.end());
                for(int k = 0; k < std::min(3, (int)dists.size()); ++k) {
                    all_points[current_point_idx].neighbors_low.push_back({dists[k].second, dists[k].first / current_point.dist_low});
                }

                if (all_points[current_point_idx].neighbors_low.size() >= 2) {
                    std::sort(all_points[current_point_idx].neighbors_low.begin(), all_points[current_point_idx].neighbors_low.end());
                    const auto& p_self = all_points[current_point_idx].pos;
                    const auto& p_n1 = all_points[all_points[current_point_idx].neighbors_low[0].first].pos;
                    const auto& p_n2 = all_points[all_points[current_point_idx].neighbors_low[1].first].pos;
                    cv::Vec3d v = p_n2 - p_n1;
                    cv::Vec3d w = p_self - p_n1;
                    double c1 = w.dot(v);
                    double c2 = v.dot(v);
                    all_points[current_point_idx].fraction_low = (c2 > 1e-9) ? (c1 / c2) : 0.5;
                }
            }

            // Find neighbors in revolution above
            if (i < revolutions - 1) {
                std::vector<std::pair<double, int>> dists;
                std::vector<std::pair<double, int>> winding_dists;
                for (size_t k = 0; k < all_points.size(); ++k) {
                    if (k == current_point_idx) continue;
                    double winding_diff = std::abs(all_points[k].winding - (current_point.winding + 1.0));
                    winding_dists.emplace_back(winding_diff, k);
                }
                std::partial_sort(winding_dists.begin(), winding_dists.begin() + 3, winding_dists.end());
                for(int k = 0; k < 3; ++k) {
                    dists.emplace_back(cv::norm(current_point.pos - all_points[winding_dists[k].second].pos), winding_dists[k].second);
                }
                std::partial_sort(dists.begin(), dists.begin() + std::min(3, (int)dists.size()), dists.end());
                for(int k = 0; k < std::min(3, (int)dists.size()); ++k) {
                    all_points[current_point_idx].neighbors_high.push_back({dists[k].second, dists[k].first / current_point.dist_high});
                }

                if (all_points[current_point_idx].neighbors_high.size() >= 2) {
                    std::sort(all_points[current_point_idx].neighbors_high.begin(), all_points[current_point_idx].neighbors_high.end());
                    const auto& p_self = all_points[current_point_idx].pos;
                    const auto& p_n1 = all_points[all_points[current_point_idx].neighbors_high[0].first].pos;
                    const auto& p_n2 = all_points[all_points[current_point_idx].neighbors_high[1].first].pos;
                    cv::Vec3d v = p_n2 - p_n1;
                    cv::Vec3d w = p_self - p_n1;
                    double c1 = w.dot(v);
                    double c2 = v.dot(v);
                    all_points[current_point_idx].fraction_high = (c2 > 1e-9) ? (c1 / c2) : 0.5;
                }
            }
        }
    }


    // Visualize before optimization
    visualize_spiral(all_points, slice_size, "spiral_init.tif", cv::Scalar(0, 0, 255), constraint_rays, false, true); // Red for before

    // --- Sheet Distance Constraints ---
    // std::vector<SheetConstraintRay> constraint_rays;
    // if (ray_step_dist > 0) {
    //     std::unordered_map<cv::Point, cv::Point, PointHash> constraints;
    //     calculate_sheet_distance_constraints(slice_mat, umbilicus, ray_step_dist, constraint_rays, constraints);
    //     if (vm.count("debug")) {
    //         visualize_sheet_distance_constraints(slice_mat, constraints, "sheet_constr_vis_spiral.tif");
    //     }
    // }

    // --- Ceres Optimization ---
    std::vector<double> positions(all_points.size() * 2);
    std::vector<double> dists_low(all_points.size());
    std::vector<double> dists_high(all_points.size());
    for(size_t i = 0; i < all_points.size(); ++i) {
        positions[i*2 + 0] = all_points[i].pos[0];
        positions[i*2 + 1] = all_points[i].pos[1];
        dists_low[i] = all_points[i].dist_low;
        dists_high[i] = all_points[i].dist_high;
    }

    // if (vm.count("debug")) {
    //     cv::Mat influence_before = visualize_sheet_constraint_influence(all_points, slice_size);
    //     cv::imwrite("influence_before.tif", influence_before);
    // }

    ceres::Problem problem;
    for (size_t i = 0; i < all_points.size(); ++i) {
        problem.AddParameterBlock(&positions[i*2], 2);
        problem.AddParameterBlock(&dists_low[i], 1);
        problem.AddParameterBlock(&dists_high[i], 1);
    }

    for (size_t i = 0; i < all_points.size(); ++i) {
        if (all_points[i].neighbors_low.size() == 3) {
            const auto& n = all_points[i].neighbors_low;
            problem.AddResidualBlock(PointToLineDistanceConstraint::Create(1.0), nullptr, &positions[i*2], &positions[n[0].first*2], &positions[n[1].first*2], &dists_low[i]);
            problem.AddResidualBlock(PointToLineDistanceConstraint::Create(1.0), nullptr, &positions[i*2], &positions[n[0].first*2], &positions[n[2].first*2], &dists_low[i]);
            problem.AddResidualBlock(PointToLineDistanceConstraint::Create(1.0), nullptr, &positions[i*2], &positions[n[1].first*2], &positions[n[2].first*2], &dists_low[i]);
        }
        if (all_points[i].neighbors_high.size() == 3) {
            const auto& n = all_points[i].neighbors_high;
            problem.AddResidualBlock(PointToLineDistanceConstraint::Create(1.0), nullptr, &positions[i*2], &positions[n[0].first*2], &positions[n[1].first*2], &dists_high[i]);
            problem.AddResidualBlock(PointToLineDistanceConstraint::Create(1.0), nullptr, &positions[i*2], &positions[n[0].first*2], &positions[n[2].first*2], &dists_high[i]);
            problem.AddResidualBlock(PointToLineDistanceConstraint::Create(1.0), nullptr, &positions[i*2], &positions[n[1].first*2], &positions[n[2].first*2], &dists_high[i]);
        }
    }

    // Fractional constraints to prevent lateral drift
    for (size_t i = 0; i < all_points.size(); ++i) {
        if (all_points[i].neighbors_low.size() >= 2) {
            const auto& n = all_points[i].neighbors_low;
            ceres::CostFunction* cost_function = FractionalConstraint::Create(all_points[i].fraction_low, 2.0);
            problem.AddResidualBlock(cost_function, nullptr, &positions[i*2], &positions[n[0].first*2], &positions[n[1].first*2]);
        }
        if (all_points[i].neighbors_high.size() >= 2) {
            const auto& n = all_points[i].neighbors_high;
            ceres::CostFunction* cost_function = FractionalConstraint::Create(all_points[i].fraction_high, 2.0);
            problem.AddResidualBlock(cost_function, nullptr, &positions[i*2], &positions[n[0].first*2], &positions[n[1].first*2]);
        }
    }

    // Spacing smoothness constraints
    for (size_t i = 0; i < all_points.size() - 2; ++i) {
        if (all_points[i].winding > all_points[i+1].winding || all_points[i+1].winding > all_points[i+2].winding) continue;
        ceres::CostFunction* cost_function = SpacingSmoothnessConstraint::Create(20.0);
        problem.AddResidualBlock(cost_function, nullptr, &positions[i*2], &positions[(i+1)*2], &positions[(i+2)*2]);
    }

    // Max angle constraints
    // for (size_t i = 0; i < all_points.size() - 2; ++i) {
    //     if (all_points[i].winding > all_points[i+1].winding || all_points[i+1].winding > all_points[i+2].winding) continue;
    //     ceres::CostFunction* cost_function = MaxAngleConstraint::Create(60.0, 200.0);
    //     problem.AddResidualBlock(cost_function, nullptr, &positions[i*2], &positions[(i+1)*2], &positions[(i+2)*2]);
    // }

    // Spacing constraints
    const double spacing_weight = 10.0;
    for (size_t i = 0; i < all_points.size() - 1; ++i) {
        if (all_points[i].winding > all_points[i+1].winding) continue;
        ceres::CostFunction* cost_function = SpacingConstraint::Create(spiral_step, spacing_weight);
        problem.AddResidualBlock(cost_function, nullptr, &positions[i*2], &positions[(i+1)*2]);
    }

    // Smoothness constraints
    // for (size_t i = 0; i < all_points.size() - 1; ++i) {
    //     if (all_points[i].winding > all_points[i+1].winding) continue; // Don't connect between revolutions
    //     ceres::CostFunction* cost_function_low = SmoothnessConstraint::Create(0.01);
    //     problem.AddResidualBlock(cost_function_low, nullptr, &dists_low[i], &dists_low[i+1]);
    //     ceres::CostFunction* cost_function_high = SmoothnessConstraint::Create(0.01);
    //     problem.AddResidualBlock(cost_function_high, nullptr, &dists_high[i], &dists_high[i+1]);
    // }

    // Add min distance constraints
    const double min_sheet_dist = 5.0;
    for (size_t i = 0; i < all_points.size(); ++i) {
        problem.AddResidualBlock(MinDistanceConstraint::Create(min_sheet_dist, 1.0), nullptr, &dists_low[i]);
        problem.AddResidualBlock(MinDistanceConstraint::Create(-min_sheet_dist, 1.0), nullptr, &dists_high[i]);
    }

    // Set parameter blocks constant
    // for (int i = 0; i < revolution_start_indices[2]; ++i) { // First two revolutions
    //     dists_low[i] = 10.0;
    //     dists_high[i] = 10.0;
    //     problem.SetParameterBlockConstant(&dists_low[i]);
    //     problem.SetParameterBlockConstant(&dists_high[i]);
    // }
    // for (size_t i = revolution_start_indices[revolutions - 2]; i < all_points.size(); ++i) { // Last two revolutions
    //     dists_low[i] = 40.0;
    //     dists_high[i] = 40.0;
    //     problem.SetParameterBlockConstant(&dists_low[i]);
    //     problem.SetParameterBlockConstant(&dists_high[i]);
    // }

    for(size_t i = 0; i < all_points.size(); ++i)
        problem.SetParameterBlockConstant(&positions[i*2]);
    // problem.SetParameterBlockConstant(&positions[0]);

    problem.SetParameterBlockConstant(&positions[positions.size()-2]);

    // Solve
    ceres::Solver::Options options;
    options.max_num_iterations = 10000;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    // #ifdef VC_USE_CUDA_SPARSE
    // options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    // #endif
    options.num_threads = omp_get_max_threads();
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    // Update all_points with optimized values
    for(size_t i = 0; i < all_points.size(); ++i) {
        all_points[i].pos[0] = positions[i*2 + 0];
        all_points[i].pos[1] = positions[i*2 + 1];
        all_points[i].dist_low = dists_low[i];
        all_points[i].dist_high = dists_high[i];
    }

    // Visualize before optimization
    visualize_spiral(all_points, slice_size, "spiral_before.tif", cv::Scalar(0, 0, 255), constraint_rays, false, false); // Red for before

    for(size_t i = 0; i < all_points.size(); ++i)
        problem.SetParameterBlockVariable(&positions[i*2]);


    // for (int i = 0; i < revolution_start_indices[1]; ++i) { // keep first revolution fixed
    //     problem.SetParameterBlockConstant(&positions[i*2]);
    // }
    for (int i = 0; i < 3; ++i)
        problem.SetParameterBlockConstant(&positions[i*2]);

    // problem.SetParameterBlockConstant(&positions[0]);
    // problem.SetParameterBlockConstant(&positions[positions.size()-2]);

    const double spacing_dist_weight = 2.0;
    for (size_t i = 0; i < all_points.size(); ++i) {
        problem.AddResidualBlock(SpacingDistConstraint::Create(dists_low[i], spacing_dist_weight), new ceres::HuberLoss(1.0), &dists_low[i]);
        problem.AddResidualBlock(SpacingDistConstraint::Create(dists_high[i], spacing_dist_weight), new ceres::HuberLoss(1.0), &dists_high[i]);
    }

    // Add sheet distance constraints
    // std::vector<PointInfluences> all_influences(all_points.size());
    // if (ray_step_dist > 0) {
    //     for (size_t i = 0; i < all_points.size(); ++i) {
    //         ceres::CostFunction* cost_function = SheetDistanceConstraint::Create(&all_influences[i]);
    //         problem.AddResidualBlock(cost_function, nullptr, &dists_low[i], &dists_high[i]);
    //     }
    // }

    // SheetConstraintCallback callback(&all_points, &positions, &constraint_rays, &all_influences, &umbilicus);
    // options.callbacks.push_back(&callback);
    // options.update_state_every_iteration = true;

    const float normal_roi_radius = 64.0f;
    const double normal_weight = 400.0;
    for (size_t i = 0; i < all_points.size() - 1; ++i) {
        if (all_points[i].winding > all_points[i+1].winding) continue;
        ceres::CostFunction* cost_function = CreateNormalConstraint(normal_grid, normal_roi_radius, normal_weight);
        problem.AddResidualBlock(cost_function, nullptr, &positions[i*2], &positions[(i+1)*2]);
    }

    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    // Update all_points with optimized values
    for(size_t i = 0; i < all_points.size(); ++i) {
        all_points[i].pos[0] = positions[i*2 + 0];
        all_points[i].pos[1] = positions[i*2 + 1];
        all_points[i].dist_low = dists_low[i];
        all_points[i].dist_high = dists_high[i];
    }

    // Visualization after optimization
    visualize_spiral(all_points, slice_size, output_path, cv::Scalar(0, 255, 0), constraint_rays, false, false); // Green for after
    // if (vm.count("debug")) {
    //     cv::Mat influence_after = visualize_sheet_constraint_influence(all_points, slice_size);
    //     cv::imwrite("influence_after.tif", influence_after);
    // }

    // if (vm.count("debug")) {
    //     std::cout << "\n--- Post-Optimization Debug Output ---" << std::endl;
    //     std::cout << "PointIdx, Winding, ParamLow, CalcLow, ParamHigh, CalcHigh" << std::endl;
    //
    //     PointToLineDistanceConstraint dist_functor(1.0);
    //
    //     for (size_t i = 0; i < all_points.size(); ++i) {
    //         const auto& p = all_points[i];
    //         double calc_dist_low = -1.0;
    //         double calc_dist_high = -1.0;
    //
    //         double p_self_arr[2] = {p.pos[0], p.pos[1]};
    //
    //         if (p.neighbors_low.size() >= 2) {
    //             const auto& n1 = all_points[p.neighbors_low[0].first];
    //             const auto& n2 = all_points[p.neighbors_low[1].first];
    //             double p_n1_arr[2] = {n1.pos[0], n1.pos[1]};
    //             double p_n2_arr[2] = {n2.pos[0], n2.pos[1]};
    //             double dist_param = p.dist_low;
    //             double residual;
    //             dist_functor(p_self_arr, p_n1_arr, p_n2_arr, &dist_param, &residual);
    //             calc_dist_low = residual + dist_param;
    //         }
    //
    //         if (p.neighbors_high.size() >= 2) {
    //             const auto& n1 = all_points[p.neighbors_high[0].first];
    //             const auto& n2 = all_points[p.neighbors_high[1].first];
    //             double p_n1_arr[2] = {n1.pos[0], n1.pos[1]};
    //             double p_n2_arr[2] = {n2.pos[0], n2.pos[1]};
    //             double dist_param = p.dist_high;
    //             double residual;
    //             dist_functor(p_self_arr, p_n1_arr, p_n2_arr, &dist_param, &residual);
    //             calc_dist_high = residual + dist_param;
    //         }
    //
    //         std::cout << std::fixed << std::setprecision(2)
    //                   << i << ", "
    //                   << p.winding << ", "
    //                   << p.dist_low << ", " << calc_dist_low << ", "
    //                   << p.dist_high << ", " << calc_dist_high
    //                   << std::endl;
    //     }
    // }
}


void generate_optimized_spiral(
    std::vector<SpiralPoint>& all_points,
    std::vector<std::vector<SpiralPoint>>& points_per_revolution,
    double starting_diameter,
    double end_diameter,
    double spiral_step,
    int revolutions,
    const cv::Point& umbilicus,
    const vc::core::util::GridStore& normal_grid,
    const cv::Size& slice_size,
    bool debug
) {
    std::cout << "Generating optimized spiral..." << std::endl;

    cv::Mat steps_vis;
    if (debug) {
        steps_vis = cv::Mat::zeros(slice_size, CV_8UC3);
    }

    double start_radius = starting_diameter / 2.0;
    double end_radius = end_diameter / 2.0;

    if (spiral_step <= 0) {
        std::cerr << "Error: spiral-step must be positive." << std::endl;
        return;
    }

    all_points.clear();

    SpiralPoint first_point;
    first_point.pos = cv::Vec3d(1880, 2070, 0);
    first_point.winding = 0;
    all_points.push_back(first_point);

    double angle = 0;

    for (int i = 1; ; ++i) {
        const auto& prev_point_const = all_points.back();
        double prev_pos[2] = {prev_point_const.pos[0], prev_point_const.pos[1]};

        // Initial guess based on the last optimized point
        cv::Point2d prev_p(prev_pos[0], prev_pos[1]);
        cv::Point2d umbilicus_p(umbilicus.x, umbilicus.y);
        cv::Point2d vec_to_prev = prev_p - umbilicus_p;
        double current_radius = cv::norm(vec_to_prev);

        cv::Point2d tangent_dir(-vec_to_prev.y, vec_to_prev.x);
        if (cv::norm(tangent_dir) > 1e-6) {
            tangent_dir /= cv::norm(tangent_dir);
        } else {
            tangent_dir = cv::Point2d(1, 0);
        }

        cv::Point2d new_pos_guess = prev_p + tangent_dir * spiral_step;

        double new_pos[2] = {new_pos_guess.x, new_pos_guess.y};

        if (debug) {
            cv::line(steps_vis, cv::Point(prev_pos[0], prev_pos[1]), cv::Point(new_pos[0], new_pos[1]), cv::Scalar(128, 0, 0), 1); // Blue for guess
            cv::circle(steps_vis, cv::Point(new_pos[0], new_pos[1]), 2, cv::Scalar(0, 255, 0), -1); // Green for guess point
        }

        ceres::Problem problem;
        problem.AddParameterBlock(prev_pos, 2);
        problem.AddParameterBlock(new_pos, 2);
        problem.SetParameterBlockConstant(prev_pos);

        // Spacing constraint
        ceres::CostFunction* spacing_cost = SpacingConstraint::Create(spiral_step, 100.0);
        problem.AddResidualBlock(spacing_cost, nullptr, prev_pos, new_pos);

        // Normal constraint
        const float normal_roi_radius = 64.0f;
        const double normal_weight = 400.0;
        ceres::CostFunction* normal_cost = CreateNormalConstraint(normal_grid, normal_roi_radius, normal_weight);
        problem.AddResidualBlock(normal_cost, nullptr, prev_pos, new_pos);

        // ceres::CostFunction* snapping_cost = CreateSnappingConstraint(normal_grid, normal_roi_radius, snapping_w, snap_trigger_th, snap_search_range);
        // problem.AddResidualBlock(snapping_cost, nullptr, prev_pos, new_pos);

        // Smoothness constraint
        if (all_points.size() > 1) {
            const auto& prev_prev_point = all_points[all_points.size() - 2];
            double prev_prev_pos[2] = {prev_prev_point.pos[0], prev_prev_point.pos[1]};
            problem.AddParameterBlock(prev_prev_pos, 2);
            problem.SetParameterBlockConstant(prev_prev_pos);
            ceres::CostFunction* smoothness_cost = SpacingSmoothnessConstraint::Create(200.0);
            problem.AddResidualBlock(smoothness_cost, nullptr, prev_prev_pos, prev_pos, new_pos);

            // ceres::CostFunction* angle_cost = MaxAngleConstraint::Create(45.0, 100.0);
            // problem.AddResidualBlock(angle_cost, nullptr, prev_prev_pos, prev_pos, new_pos);
        }

        ceres::Solver::Options options;
        options.max_num_iterations = 1000;
        options.linear_solver_type = ceres::DENSE_QR;
        options.logging_type = ceres::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (debug) {
            cv::line(steps_vis, cv::Point(prev_pos[0], prev_pos[1]), cv::Point(new_pos[0], new_pos[1]), cv::Scalar(0, 0, 255), 1); // Red for optimized
        }

        SpiralPoint new_point;
        new_point.pos = cv::Vec3d(new_pos[0], new_pos[1], 0);

        cv::Point2d vec_from_umbilicus(new_pos[0] - umbilicus.x, new_pos[1] - umbilicus.y);
        double current_angle = std::atan2(vec_from_umbilicus.y, vec_from_umbilicus.x);
        if (all_points.size() > 1) {
            const auto& prev_point = all_points.back();
            cv::Point2d prev_vec_from_umbilicus(prev_point.pos[0] - umbilicus.x, prev_point.pos[1] - umbilicus.y);
            double prev_angle = std::atan2(prev_vec_from_umbilicus.y, prev_vec_from_umbilicus.x);
            double angle_diff = current_angle - prev_angle;
            if (angle_diff > CV_PI) angle_diff -= 2 * CV_PI;
            if (angle_diff < -CV_PI) angle_diff += 2 * CV_PI;
            new_point.winding = prev_point.winding + angle_diff / (2 * CV_PI);
        } else {
            new_point.winding = current_angle / (2*CV_PI);
        }

        // std::cout << "  Point " << i << ": cost=" << summary.final_cost << ", winding=" << new_point.winding << std::endl;

        int current_rev = static_cast<int>(new_point.winding);
        if (current_rev >= 0 && current_rev < revolutions) {
            points_per_revolution[current_rev].push_back(new_point);
        }
        all_points.push_back(new_point);

        if (new_point.winding >= revolutions) {
            break;
        }
    }

    if (debug) {
        cv::imwrite("spiral_generation_steps.tif", steps_vis);
        std::cout << "Saved spiral generation steps visualization to spiral_generation_steps.tif" << std::endl;
    }
}


int spiral_main(
    const cv::Mat& slice_mat,
    const std::optional<cv::Vec3f>& umbilicus_point,
    const fs::path& output_path,
    float ray_step_dist,
    const po::variables_map& vm
) {
    if (!vm.count("starting-diameter") || !vm.count("end-diameter") || !vm.count("spiral-step")) {
        std::cerr << "Error: For spiral mode, --starting-diameter, --end-diameter, and --spiral-step are required." << std::endl;
        return 1;
    }
    double starting_diameter = vm["starting-diameter"].as<double>();
    double end_diameter = vm["end-diameter"].as<double>();
    double spiral_step = vm["spiral-step"].as<double>();
    int revolutions = vm["revolutions"].as<int>();
    cv::Point umbilicus_p(
        static_cast<int>(std::round((*umbilicus_point)[0])),
        static_cast<int>(std::round((*umbilicus_point)[1]))
    );
    run_spiral_generation(slice_mat, starting_diameter, end_diameter, spiral_step, revolutions, umbilicus_p, slice_mat.size(), output_path, ray_step_dist, vm);
    return 0;
}
