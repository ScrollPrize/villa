#include "spiral2.hpp"
#include "support.hpp"
#include "spiral_ceres.hpp"
#include "spiral.hpp"

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

class VideoCallback : public ceres::IterationCallback {
public:
    VideoCallback(const std::vector<double>* positions, cv::Mat* frame, cv::VideoWriter* writer)
        : positions_(positions), frame_(frame), writer_(writer) {}

    virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
        if (writer_ && writer_->isOpened()) {
            // Redraw the entire trace for this frame
            frame_->setTo(cv::Scalar(0,0,0)); // Clear frame
            for (size_t i = 0; i < positions_->size() / 2 - 1; ++i) {
                cv::Point p1((*positions_)[i*2], (*positions_)[i*2+1]);
                cv::Point p2((*positions_)[(i+1)*2], (*positions_)[(i+1)*2+1]);
                cv::line(*frame_, p1, p2, cv::Scalar(0, 0, 255), 1);
            }
            writer_->write(*frame_);
        }
        return ceres::SOLVER_CONTINUE;
    }

private:
    const std::vector<double>* positions_;
    cv::Mat* frame_;
    cv::VideoWriter* writer_;
};

void generate_bidir_spiral(
    std::vector<SpiralPoint>& all_points,
    std::vector<std::vector<SpiralPoint>>& points_per_revolution,
    const SpiralPoint& start_point,
    const std::string& direction,
    double spiral_step,
    int revolutions,
    const cv::Point& umbilicus,
    const vc::core::util::GridStore& normal_grid,
    const cv::Size& slice_size,
    bool debug,
    cv::VideoWriter& video_writer,
    cv::Mat& vis_frame
);

void visualize_normal_grid(const vc::core::util::GridStore& normal_grid, const cv::Size& size, const std::string& path);
void visualize_spiral(cv::Mat& vis, const std::vector<SpiralPoint>& spiral, const cv::Scalar& color, const cv::Scalar& endpoint_color, bool draw_endpoints);

// JSON serialization for SpiralPoint
void visualize_annotations(
    const cv::Size& slice_size,
    const VCCollection::Collection& collection,
    const ColPoint& start_point_fw,
    const ColPoint& start_point_bw,
    const std::string& path
) {
    cv::Mat viz = cv::Mat::zeros(slice_size, CV_8UC3);
    cv::RNG rng(12345);
    cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

    for (const auto& [id, point] : collection.points) {
        cv::Point p(point.p[0], point.p[1]);
        cv::circle(viz, p, 3, color, -1);
        std::stringstream ss;
        ss << collection.name << ":" << std::fixed << std::setprecision(2) << point.winding_annotation;
        cv::putText(viz, ss.str(), p + cv::Point(5, 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    // Highlight start points
    cv::Point p_fw(start_point_fw.p[0], start_point_fw.p[1]);
    cv::circle(viz, p_fw, 8, cv::Scalar(0, 255, 0), 2); // Green circle for forward start

    cv::Point p_bw(start_point_bw.p[0], start_point_bw.p[1]);
    cv::circle(viz, p_bw, 8, cv::Scalar(0, 0, 255), 2); // Blue circle for backward start

    if (!cv::imwrite(path, viz)) {
        std::cerr << "Error: Failed to write annotation visualization to " << path << std::endl;
    } else {
        std::cout << "Saved annotation visualization to " << path << std::endl;
    }
}

const float snap_trigger_th = 2.0;
const float snap_search_range = 8.0;
const float  snapping_w = 50.0;


void optimize_spiral_trace(
    std::vector<SpiralPoint>& spiral_trace,
    const vc::core::util::GridStore& normal_grid,
    double spiral_step,
    cv::VideoWriter& video_writer,
    cv::Mat& vis_frame
) {
    if (spiral_trace.size() < 3) {
        std::cout << "Trace is too short to optimize, skipping." << std::endl;
        return;
    }

    std::cout << "Optimizing spiral trace with " << spiral_trace.size() << " points..." << std::endl;

    std::vector<double> positions(spiral_trace.size() * 2);
    for(size_t i = 0; i < spiral_trace.size(); ++i) {
        positions[i*2 + 0] = spiral_trace[i].pos[0];
        positions[i*2 + 1] = spiral_trace[i].pos[1];
    }

    ceres::Problem problem;
    for (size_t i = 0; i < spiral_trace.size(); ++i) {
        problem.AddParameterBlock(&positions[i*2], 2);
    }

    // Fix the first and last points
    problem.SetParameterBlockConstant(&positions[0]);
    problem.SetParameterBlockConstant(&positions[(spiral_trace.size() - 1) * 2]);

    // Add Normal Constraints
    const float normal_roi_radius = 64.0f;
    const double normal_weight = 400.0;
    for (size_t i = 0; i < spiral_trace.size() - 1; ++i) {
        ceres::CostFunction* normal_cost = CreateNormalConstraint(normal_grid, normal_roi_radius, normal_weight);
        problem.AddResidualBlock(normal_cost, nullptr, &positions[i*2], &positions[(i+1)*2]);

        ceres::CostFunction* snapping_fw = CreateSnappingConstraint(normal_grid, normal_roi_radius, snapping_w, snap_trigger_th, snap_search_range);
        problem.AddResidualBlock(snapping_fw, nullptr, &positions[i*2], &positions[(i+1)*2]);

        ceres::CostFunction* snapping_bw = CreateSnappingConstraint(normal_grid, normal_roi_radius, snapping_w, snap_trigger_th, snap_search_range);
        problem.AddResidualBlock(snapping_bw, nullptr, &positions[(i+1)*2], &positions[i*2]);
    }

    // Add Spacing Smoothness Constraints
    for (size_t i = 0; i < spiral_trace.size() - 2; ++i) {
        ceres::CostFunction* cost_function = SpacingSmoothnessConstraint::Create(100.0);
        problem.AddResidualBlock(cost_function, nullptr, &positions[i*2], &positions[(i+1)*2], &positions[(i+2)*2]);
    }

    // Add Even Spacing Constraints
    for (size_t i = 0; i < spiral_trace.size() - 2; ++i) {
        ceres::CostFunction* cost_function = EvenSpacingConstraint::Create(5.0);
        problem.AddResidualBlock(cost_function, nullptr, &positions[i*2], &positions[(i+1)*2], &positions[(i+2)*2]);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 1000;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = omp_get_max_threads();
    options.minimizer_progress_to_stdout = true;

    VideoCallback callback(&positions, &vis_frame, &video_writer);
    if (video_writer.isOpened()) {
        options.callbacks.push_back(&callback);
        options.update_state_every_iteration = true;
    }

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    // Update spiral_trace with optimized values
    for(size_t i = 0; i < spiral_trace.size(); ++i) {
        spiral_trace[i].pos[0] = positions[i*2 + 0];
        spiral_trace[i].pos[1] = positions[i*2 + 1];
    }
}

std::vector<SpiralPoint> generate_single_spiral_wrap(
    const cv::Mat& slice_mat,
    const SpiralPoint& start_point_fw,
    const SpiralPoint& start_point_bw,
    double spiral_step,
    const cv::Point& umbilicus_p,
    const vc::core::util::GridStore& normal_grid,
    const cv::Size& slice_size,
    bool debug,
    cv::VideoWriter& video_writer,
    cv::Mat& vis_frame
) {
    std::vector<SpiralPoint> all_points_fw, all_points_bw;
    std::vector<std::vector<SpiralPoint>> points_per_revolution_fw(1), points_per_revolution_bw(1);

    generate_bidir_spiral(all_points_fw, points_per_revolution_fw, start_point_fw, "fw", spiral_step, 1, umbilicus_p, normal_grid, slice_size, debug, video_writer, vis_frame);
    generate_bidir_spiral(all_points_bw, points_per_revolution_bw, start_point_bw, "bw", spiral_step, 1, umbilicus_p, normal_grid, slice_size, debug, video_writer, vis_frame);

    std::vector<SpiralPoint> spiral_interp;
    if (!all_points_fw.empty() && !all_points_bw.empty()) {

        double min_winding_fw = all_points_fw.front().winding;
        double max_winding_fw = all_points_fw.back().winding;
        double winding_range = max_winding_fw - min_winding_fw;

        std::vector<SpiralPoint> sorted_bw = all_points_bw;
        std::sort(sorted_bw.begin(), sorted_bw.end(), [](const auto& a, const auto& b){
            return a.winding < b.winding;
        });

        for (const auto& p_fw : all_points_fw) {
            double winding_fw = p_fw.winding;

            auto it = std::lower_bound(sorted_bw.begin(), sorted_bw.end(), winding_fw,
                                       [](const SpiralPoint& p, double w) {
                                           return p.winding < w;
                                       });

            if (it != sorted_bw.begin() && it != sorted_bw.end()) {
                const auto& p_bw2 = *it;
                const auto& p_bw1 = *(it - 1);

                double t = (winding_fw - p_bw1.winding) / (p_bw2.winding - p_bw1.winding);
                cv::Vec3d pos_bw = p_bw1.pos * (1.0 - t) + p_bw2.pos * t;

                double blend_factor = (winding_fw - min_winding_fw) / winding_range;

                SpiralPoint p_interp;
                p_interp.pos = p_fw.pos * (1.0 - blend_factor) + pos_bw * blend_factor;
                p_interp.winding = winding_fw;
                spiral_interp.push_back(p_interp);
            }
        }
        std::vector<SpiralPoint> spiral_avg = spiral_interp;
        optimize_spiral_trace(spiral_interp, normal_grid, spiral_step, video_writer, vis_frame);

        if (debug) {
            cv::Mat fit_vis = cv::Mat::zeros(slice_size, CV_8UC3);
            visualize_spiral(fit_vis, all_points_fw, cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), false);
            visualize_spiral(fit_vis, all_points_bw, cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0), false);
            visualize_spiral(fit_vis, spiral_avg, cv::Scalar(0, 180, 180), cv::Scalar(0, 255, 255), false);
            visualize_spiral(fit_vis, spiral_interp, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), false);

            cv::Mat slice_bgr;
            cv::cvtColor(slice_mat, slice_bgr, cv::COLOR_GRAY2BGR);
            slice_bgr *= 0.5;
            cv::max(slice_bgr, fit_vis, fit_vis);

            cv::imwrite("debug_bidir_fit_avg.tif", fit_vis);
        }
    }
    return spiral_interp;
}

int spiral2_main(
    const cv::Mat& slice_mat,
    const VCCollection& point_collection,
    const std::optional<cv::Vec3f>& umbilicus_point,
    const std::string& umbilicus_set_name,
    const po::variables_map& vm
) {
    cv::VideoWriter video_writer;
    if (vm.count("video-out")) {
        std::string video_out_path = vm["video-out"].as<std::string>();
        if (!video_out_path.empty()) {
            video_writer.open(video_out_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, cv::Size(slice_mat.cols, slice_mat.rows));
            if (!video_writer.isOpened()) {
                std::cerr << "Error: Could not open video writer for path: " << video_out_path << std::endl;
            }
        }
    }

    if (!vm.count("spiral-step")) {
        std::cerr << "Error: For spiral2 mode, --spiral-step is required." << std::endl;
        return 1;
    }
    double spiral_step = vm["spiral-step"].as<double>();
    cv::Point umbilicus_p(
        static_cast<int>(std::round((*umbilicus_point)[0])),
        static_cast<int>(std::round((*umbilicus_point)[1]))
    );

    cv::Mat binary_slice = slice_mat > 0;

    cv::setNumThreads(0);

    auto start_skeletonization = std::chrono::high_resolution_clock::now();
    std::clock_t start_cpu_skeletonization = std::clock();
    auto [skeleton_graph, skeleton_id_img] = generate_skeleton_graph(binary_slice, vm);
    std::clock_t end_cpu_skeletonization = std::clock();
    auto end_skeletonization = std::chrono::high_resolution_clock::now();
    double real_time_skeletonization = std::chrono::duration<double>(end_skeletonization - start_skeletonization).count();
    double cpu_time_skeletonization = static_cast<double>(end_cpu_skeletonization - start_cpu_skeletonization) / CLOCKS_PER_SEC;
    std::cout << "Skeletonization took " << real_time_skeletonization << " s (real), "
              << cpu_time_skeletonization << " s (user), "
              << (cpu_time_skeletonization / real_time_skeletonization * 100.0) << "%" << std::endl;

    auto start_grid_construction = std::chrono::high_resolution_clock::now();
    std::clock_t start_cpu_grid_construction = std::clock();
    vc::core::util::GridStore normal_grid_src(
        cv::Rect(0, 0, slice_mat.cols, slice_mat.rows), 32
    );
    populate_normal_grid(skeleton_graph, normal_grid_src, spiral_step);
    std::clock_t end_cpu_grid_construction = std::clock();
    auto end_grid_construction = std::chrono::high_resolution_clock::now();
    double real_time_grid_construction = std::chrono::duration<double>(end_grid_construction - start_grid_construction).count();
    double cpu_time_grid_construction = static_cast<double>(end_cpu_grid_construction - start_cpu_grid_construction) / CLOCKS_PER_SEC;
    std::cout << "Normal grid construction took " << real_time_grid_construction << " s (real), "
              << cpu_time_grid_construction << " s (user), "
              << (cpu_time_grid_construction / real_time_grid_construction * 100.0) << "%" << std::endl;

    size_t grid_memory = normal_grid_src.get_memory_usage();
    std::cout << "Normal grid memory usage: " << grid_memory / (1024.0 * 1024.0) << " MB" << std::endl;

    //so we test it here ...
    normal_grid_src.save("test_gridstore.bin");
    vc::core::util::GridStore normal_grid("test_gridstore.bin");

    if (vm.count("debug")) {
        visualize_normal_grid(normal_grid, slice_mat.size(), "normal_constraints_vis_spiral2.tif");
    }

    const auto& collections = point_collection.getAllCollections();
    nlohmann::json json_output;

    for (const auto& [id, collection] : collections) {
        if (collection.name == umbilicus_set_name) continue;
        if (collection.name != "col3") continue;

        nlohmann::json json_collection;
        json_collection["name"] = collection.name;
        std::vector<std::vector<SpiralPoint>> all_spiral_wraps;

        cv::Mat composite_vis = cv::Mat::zeros(slice_mat.size(), CV_8UC3);
        std::map<float, const ColPoint*> sorted_points;
        for (const auto& [point_id, point] : collection.points) {
            if (!std::isnan(point.winding_annotation)) {
                sorted_points[point.winding_annotation] = &point;
            }
        }

        for (auto it = sorted_points.begin(); it != sorted_points.end(); ++it) {
            auto next_it = std::next(it);
            if (next_it != sorted_points.end()) {
                if (std::abs(next_it->first - it->first - 1.0f) < 1e-6) {
                    if (it->first != 15.0) continue;

                    const ColPoint* p_data_fw = it->second;
                    const ColPoint* p_data_bw = next_it->second;

                    SpiralPoint start_point_fw;
                    start_point_fw.pos = cv::Vec3d(p_data_fw->p[0], p_data_fw->p[1], 0);
                    start_point_fw.winding = p_data_fw->winding_annotation;

                    SpiralPoint start_point_bw;
                    start_point_bw.pos = cv::Vec3d(p_data_bw->p[0], p_data_bw->p[1], 0);
                    start_point_bw.winding = p_data_bw->winding_annotation;

                    std::cout << "Generating spiral wrap for " << collection.name
                                << " from winding " << start_point_fw.winding
                                << " to " << start_point_bw.winding << std::endl;

                    std::vector<SpiralPoint> spiral_wrap = generate_single_spiral_wrap(
                        slice_mat, start_point_fw, start_point_bw, spiral_step, umbilicus_p,
                        normal_grid, slice_mat.size(), vm.count("debug"),
                        video_writer, composite_vis
                    );

                    if (!spiral_wrap.empty()) {
                        all_spiral_wraps.push_back(spiral_wrap);
                        cv::RNG rng(it->first);
                        cv::Scalar color(rng.uniform(50, 256), rng.uniform(50, 256), rng.uniform(50, 256));
                        visualize_spiral(composite_vis, spiral_wrap, color, color, false);

                        const auto& start_p = spiral_wrap.front();
                        cv::Point p(start_p.pos[0], start_p.pos[1]);
                        std::stringstream ss;
                        ss << std::fixed << std::setprecision(2) << start_p.winding;
                        cv::putText(composite_vis, ss.str(), p + cv::Point(5, 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv::LINE_AA);

                        std::string intermediate_path = "bidir_spirals_" + collection.name + "_intermediate.tif";
                        cv::imwrite(intermediate_path, composite_vis);
                    }
                }
            }
        }
        std::string final_path = "bidir_spirals_" + collection.name + ".tif";
        cv::Mat slice_bgr;
        cv::cvtColor(slice_mat, slice_bgr, cv::COLOR_GRAY2BGR);
        slice_bgr *= 0.5;
        cv::max(slice_bgr, composite_vis, composite_vis);
        cv::imwrite(final_path, composite_vis);
        std::cout << "Saved final spiral visualization for " << collection.name << " to " << final_path << std::endl;

        json_collection["spirals"] = all_spiral_wraps;
        json_output.push_back(json_collection);
    }

    if (vm.count("json-out")) {
        std::string json_out_path = vm["json-out"].as<std::string>();
        std::ofstream o(json_out_path);
        o << std::setw(4) << json_output << std::endl;
        std::cout << "Saved spiral data to " << json_out_path << std::endl;
    }
    return 0;
}


void generate_bidir_spiral(
    std::vector<SpiralPoint>& all_points,
    std::vector<std::vector<SpiralPoint>>& points_per_revolution,
    const SpiralPoint& start_point,
    const std::string& direction,
    double spiral_step,
    int revolutions,
    const cv::Point& umbilicus,
    const vc::core::util::GridStore& normal_grid,
    const cv::Size& slice_size,
    bool debug,
    cv::VideoWriter& video_writer,
    cv::Mat& vis_frame
) {
    std::cout << "Generating bidirectional spiral (direction: " << direction << ")..." << std::endl;

    cv::Mat steps_vis;
    if (debug) {
        steps_vis = cv::Mat::zeros(slice_size, CV_8UC3);
    }

    if (spiral_step <= 0) {
        std::cerr << "Error: spiral-step must be positive." << std::endl;
        return;
    }

    all_points.clear();
    all_points.push_back(start_point);

    double angle_mult = (direction == "fw") ? 1.0 : -1.0;

    for (int i = 1; ; ++i) {
        const auto& prev_point_const = all_points.back();
        double prev_pos[2] = {prev_point_const.pos[0], prev_point_const.pos[1]};

        cv::Point2d prev_p(prev_pos[0], prev_pos[1]);
        cv::Point2d umbilicus_p(umbilicus.x, umbilicus.y);
        cv::Point2d vec_to_prev = prev_p - umbilicus_p;

        cv::Point2d tangent_dir(-vec_to_prev.y, vec_to_prev.x);
        if (cv::norm(tangent_dir) > 1e-6) {
            tangent_dir /= cv::norm(tangent_dir);
        } else {
            tangent_dir = cv::Point2d(1, 0);
        }

        cv::Point2d new_pos_guess = prev_p + tangent_dir * spiral_step * angle_mult;

        double new_pos[2] = {new_pos_guess.x, new_pos_guess.y};

        if (video_writer.isOpened()) {
            cv::Point p1(prev_pos[0], prev_pos[1]);
            cv::Point p2(new_pos[0], new_pos[1]);
            cv::line(vis_frame, p1, p2, (direction == "fw" ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0)), 1);
            video_writer.write(vis_frame);
        }

        ceres::Problem problem;
        problem.AddParameterBlock(prev_pos, 2);
        problem.AddParameterBlock(new_pos, 2);
        problem.SetParameterBlockConstant(prev_pos);

        ceres::CostFunction* spacing_cost = SpacingConstraint::Create(spiral_step, 5.0);
        problem.AddResidualBlock(spacing_cost, nullptr, prev_pos, new_pos);

        const float normal_roi_radius = 64.0f;
        const double normal_weight = 400.0;
        ceres::CostFunction* normal_cost = CreateNormalConstraint(normal_grid, normal_roi_radius, normal_weight);
        problem.AddResidualBlock(normal_cost, nullptr, prev_pos, new_pos);

        ceres::CostFunction* snapping_cost = CreateSnappingConstraint(normal_grid, normal_roi_radius, snapping_w, snap_trigger_th, snap_search_range);
        problem.AddResidualBlock(snapping_cost, nullptr, new_pos, prev_pos);

        double prev_prev_pos[2];

        if (all_points.size() > 1) {
            const auto& prev_prev_point = all_points[all_points.size() - 2];
            prev_prev_pos[0] = prev_prev_point.pos[0];
            prev_prev_pos[1] = prev_prev_point.pos[1];
            problem.AddParameterBlock(prev_prev_pos, 2);
            problem.SetParameterBlockConstant(prev_prev_pos);
            ceres::CostFunction* smoothness_cost = SpacingSmoothnessConstraint::Create(100.0);
            problem.AddResidualBlock(smoothness_cost, nullptr, prev_prev_pos, prev_pos, new_pos);

            // ceres::CostFunction* angle_cost = MaxAngleConstraint::Create(45.0, 100.0);
            // problem.AddResidualBlock(angle_cost, nullptr, prev_prev_pos, prev_pos, new_pos);
        }

        ceres::Solver::Options options;
        options.max_num_iterations = 50;
        options.linear_solver_type = ceres::DENSE_QR;
        options.logging_type = ceres::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (debug) {
            cv::line(steps_vis, cv::Point(prev_pos[0], prev_pos[1]), cv::Point(new_pos[0], new_pos[1]), cv::Scalar(0, 0, 255), 1);
        }

        SpiralPoint new_point;
        new_point.pos = cv::Vec3d(new_pos[0], new_pos[1], 0);

        cv::Point2d vec_from_umbilicus(new_pos[0] - umbilicus.x, new_pos[1] - umbilicus.y);
        double current_angle = std::atan2(vec_from_umbilicus.y, vec_from_umbilicus.x);

        const auto& prev_point = all_points.back();
        cv::Point2d prev_vec_from_umbilicus(prev_point.pos[0] - umbilicus.x, prev_point.pos[1] - umbilicus.y);
        double prev_angle = std::atan2(prev_vec_from_umbilicus.y, prev_vec_from_umbilicus.x);
        double angle_diff = current_angle - prev_angle;
        if (angle_diff > CV_PI) angle_diff -= 2 * CV_PI;
        if (angle_diff < -CV_PI) angle_diff += 2 * CV_PI;
        new_point.winding = prev_point.winding + angle_diff / (2 * CV_PI);

        // std::cout << "  Point " << i << ": cost=" << summary.final_cost << ", winding=" << new_point.winding << std::endl;

        int current_rev = static_cast<int>(std::abs(new_point.winding - start_point.winding));
        if (current_rev >= 0 && current_rev < revolutions) {
            points_per_revolution[current_rev].push_back(new_point);
        }
        all_points.push_back(new_point);

        if (current_rev >= revolutions || all_points.size() >= 10000) {
            break;
        }
    }

    // if (debug) {
    //     std::string vis_path = "spiral_generation_steps_" + direction + ".tif";
    //     cv::imwrite(vis_path, steps_vis);
    //     std::cout << "Saved spiral generation steps visualization to " << vis_path << std::endl;
    // }
}
