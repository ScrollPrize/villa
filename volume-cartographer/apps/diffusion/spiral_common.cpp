
#include "spiral_common.hpp"
#include "spiral_ceres.hpp"

void visualize_spiral(
    const std::vector<SpiralPoint>& all_points,
    const cv::Size& slice_size,
    const fs::path& output_path,
    const cv::Scalar& point_color,
    const std::vector<SheetConstraintRay>& constraint_rays,
    bool draw_influence,
    bool draw_winding_text
) {
    cv::Mat viz = cv::Mat::zeros(slice_size, CV_8UC3);
    visualize_spiral(viz, all_points, cv::Scalar(255, 255, 255), point_color, draw_winding_text);

    if (cv::imwrite(output_path.string(), viz)) {
        std::cout << "Saved spiral visualization to " << output_path << std::endl;
    } else {
        std::cerr << "Error: Failed to write spiral visualization to " << output_path << std::endl;
    }
}


#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


bool find_intersections(
    const cv::Point2f& ray_origin,
    const cv::Point2f& ray_dir,
    const std::vector<SpiralPoint>& all_points,
    std::vector<SpiralIntersection>& intersections
) {
    intersections.clear();
    for (size_t i = 0; i < all_points.size() - 1; ++i) {
        if (all_points[i].winding > all_points[i+1].winding) continue;

        const auto& p1 = all_points[i].pos;
        const auto& p2 = all_points[i+1].pos;

        cv::Point2f v1(p1[0] - ray_origin.x, p1[1] - ray_origin.y);
        cv::Point2f v2(p2[0] - ray_origin.x, p2[1] - ray_origin.y);
        cv::Point2f v3(-ray_dir.y, ray_dir.x); // Perpendicular to ray direction

        double dot1 = v1.dot(v3);
        double dot2 = v2.dot(v3);

        if (dot1 * dot2 < 0) { // If signs are different, an intersection occurs
            double t = dot1 / (dot1 - dot2);
            cv::Point2d intersection_pt(p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]));

            // Check if intersection is along the ray direction
            cv::Point2f vec_to_intersect(intersection_pt.x - ray_origin.x, intersection_pt.y - ray_origin.y);
            if (vec_to_intersect.dot(ray_dir) > 0) {
                intersections.push_back({
                    (int)i,
                                        (int)i + 1,
                                        t,
                                        intersection_pt,
                                        all_points[i].winding + t * (all_points[i+1].winding - all_points[i].winding)
                });
            }
        }
    }

    std::sort(intersections.begin(), intersections.end(), [](const auto& a, const auto& b){
        return a.winding < b.winding;
    });

    return !intersections.empty();
}

SheetConstraintCallback::SheetConstraintCallback(
    std::vector<SpiralPoint>* all_points,
    const std::vector<double>* positions,
    const std::vector<SheetConstraintRay>* constraint_rays,
    std::vector<PointInfluences>* all_influences,
    const cv::Point* umbilicus
) : all_points_(all_points),
positions_(positions),
constraint_rays_(constraint_rays),
all_influences_(all_influences),
umbilicus_(umbilicus) {}

ceres::CallbackReturnType SheetConstraintCallback::operator()(const ceres::IterationSummary& summary) {
    // Update all_points with current parameter values from the solver
    for(size_t i = 0; i < all_points_->size(); ++i) {
        (*all_points_)[i].pos[0] = (*positions_)[i*2 + 0];
        (*all_points_)[i].pos[1] = (*positions_)[i*2 + 1];
    }

    // Clear previous influences
    for(auto& influences : *all_influences_) {
        influences.low_influences.clear();
        influences.high_influences.clear();
    }

    // Recalculate intersections and distribute influences
    std::vector<bool> constraint_applied(constraint_rays_->size(), false);
    int ray_idx = 0;
    for (const auto& ray : *constraint_rays_) {
        std::vector<SpiralIntersection> intersections;
        find_intersections(cv::Point2f(umbilicus_->x, umbilicus_->y), ray.dir, *all_points_, intersections);

        if (intersections.size() < 2) {
            ray_idx++;
            continue;
        }

        int constraint_in_ray_idx = 0;
        for (const auto& constraint : ray.constraints) {
            double target_dist = cv::norm(constraint.first - constraint.second);
            cv::Point2f p_inner_float(constraint.second.x, constraint.second.y);

            int best_idx = -1;
            for(size_t i = 0; i < intersections.size() - 1; ++i) {
                cv::Point2f p1(intersections[i].intersection_point.x, intersections[i].intersection_point.y);
                cv::Point2f p2(intersections[i+1].intersection_point.x, intersections[i+1].intersection_point.y);
                if (cv::norm(p1 - p_inner_float) + cv::norm(p2 - p_inner_float) < cv::norm(p1-p2) + 1e-6) {
                    best_idx = i;
                    break;
                }
            }

            if (best_idx != -1) {
                constraint_applied[ray_idx + constraint_in_ray_idx] = true;
                const auto& intersection_low = intersections[best_idx];
                const auto& intersection_high = intersections[best_idx + 1];

                cv::Point2f p_outer_float(constraint.first.x, constraint.first.y);
                cv::Point2f vec_constraint = p_outer_float - p_inner_float;
                double constraint_len_sq = vec_constraint.dot(vec_constraint);

                cv::Point2f vec_inner_to_low(intersection_low.intersection_point.x - p_inner_float.x, intersection_low.intersection_point.y - p_inner_float.y);
                double t = vec_inner_to_low.dot(vec_constraint) / (constraint_len_sq + 1e-9);

                double w_low = 0.0, w_high = 0.0;
                if (t > 0 && t < 1) {
                    w_low = 1.0 - t;
                    w_high = t;
                } else if (t <= 0) {
                    w_low = 1.0;
                } else {
                    w_high = 1.0;
                }

                if (w_high > 0) {
                    (*all_influences_)[intersection_low.point_idx1].high_influences.push_back({target_dist, w_high * (1.0 - intersection_low.t)});
                    (*all_influences_)[intersection_low.point_idx2].high_influences.push_back({target_dist, w_high * intersection_low.t});
                }
                if (w_low > 0) {
                    (*all_influences_)[intersection_high.point_idx1].low_influences.push_back({target_dist, w_low * (1.0 - intersection_high.t)});
                    (*all_influences_)[intersection_high.point_idx2].low_influences.push_back({target_dist, w_low * intersection_high.t});
                }
            }
            constraint_in_ray_idx++;
        }
        ray_idx += ray.constraints.size();
    }

    // Fallback logic
    ray_idx = 0;
    for (const auto& ray : *constraint_rays_) {
        for (const auto& constraint : ray.constraints) {
            if (!constraint_applied[ray_idx]) {
                double target_dist = cv::norm(constraint.first - constraint.second);
                cv::Point2f p_inner_float(constraint.second.x, constraint.second.y);

                int closest_point_idx = -1;
                double min_dist = std::numeric_limits<double>::max();
                for(size_t i = 0; i < all_points_->size(); ++i) {
                    double d = cv::norm(cv::Point2f((*all_points_)[i].pos[0], (*all_points_)[i].pos[1]) - p_inner_float);
                    if (d < min_dist) {
                        min_dist = d;
                        closest_point_idx = i;
                    }
                }

                if (closest_point_idx != -1) {
                    (*all_influences_)[closest_point_idx].low_influences.push_back({target_dist, 1.0});
                    (*all_influences_)[closest_point_idx].high_influences.push_back({target_dist, 1.0});
                }
            }
            ray_idx++;
        }
    }

    return ceres::SOLVER_CONTINUE;
}

cv::Mat visualize_sheet_constraint_influence(
    const std::vector<SpiralPoint>& all_points,
    const cv::Size& slice_size,
    const std::vector<PointInfluences>& all_influences
) {
    cv::Mat viz = cv::Mat::zeros(slice_size, CV_8UC3);
    // This function will be updated later to visualize the new constraint logic.
    return viz;
}



void visualize_spiral(
    cv::Mat& viz,
    const std::vector<SpiralPoint>& all_points,
    const cv::Scalar& line_color,
    const cv::Scalar& point_color,
    bool draw_winding_text
) {
    if (all_points.empty()) return;

    // Draw spiral edges
    for (size_t i = 0; i < all_points.size() - 1; ++i) {
        // A simple check to handle forward and backward spirals without complex sorting
        if (std::abs(all_points[i+1].winding - all_points[i].winding) < 0.5) {
            cv::Point p1(all_points[i].pos[0], all_points[i].pos[1]);
            cv::Point p2(all_points[i+1].pos[0], all_points[i+1].pos[1]);
            cv::line(viz, p1, p2, line_color, 1, cv::LINE_AA);
        }
    }

    // Draw the points
    for (size_t i = 0; i < all_points.size(); ++i) {
        cv::Point p(all_points[i].pos[0], all_points[i].pos[1]);
        cv::circle(viz, p, 3, point_color, -1, cv::LINE_AA);
        if (draw_winding_text) {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << all_points[i].winding;
            cv::putText(viz, ss.str(), p + cv::Point(5, 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        }
    }
}
