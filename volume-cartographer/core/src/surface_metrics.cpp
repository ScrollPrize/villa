#include "vc/ui/surface_metrics.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>

// Helper to get point-to-line-segment squared distance
static float dist_point_segment_sq(const cv::Vec3f& p, const cv::Vec3f& a, const cv::Vec3f& b) {
    cv::Vec3f ab = b - a;
    cv::Vec3f ap = p - a;
    float t = ap.dot(ab) / ab.dot(ab);
    t = std::max(0.0f, std::min(1.0f, t));
    cv::Vec3f closest_point = a + t * ab;
    cv::Vec3f d = p - closest_point;
    return d.dot(d);
}

// Direct search method to minimize point-line distance from a starting location
static float find_intersection_direct(QuadSurface* surface, cv::Vec2f& loc, const cv::Vec3f& p1, const cv::Vec3f& p2, float init_step, float min_step, const cv::Vec3f& center_in_points)
{
    cv::Vec3f ptr_loc = cv::Vec3f(loc[0], loc[1], 0) - center_in_points;
    if (!surface->valid(ptr_loc)) {
        return -1.0f;
    }

    cv::Mat_<cv::Vec3f> points = surface->rawPoints();
    cv::Rect bounds = {0, 0, points.cols - 1, points.rows - 1};

    bool changed = true;
    cv::Vec3f surface_point = surface->coord(ptr_loc);
    float best_dist_sq = dist_point_segment_sq(surface_point, p1, p2);
    float current_dist_sq;

    std::vector<cv::Vec2f> search = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    float step = init_step;

    while (true) {
        changed = false;

        for (auto& off : search) {
            cv::Vec2f cand_loc_2f = loc + off * step;

            if (!bounds.contains(cv::Point(cand_loc_2f[0], cand_loc_2f[1])))
                continue;

            cv::Vec3f cand_ptr_loc = cv::Vec3f(cand_loc_2f[0], cand_loc_2f[1], 0) - center_in_points;
            if (!surface->valid(cand_ptr_loc))
                continue;

            surface_point = surface->coord(cand_ptr_loc);
            current_dist_sq = dist_point_segment_sq(surface_point, p1, p2);

            if (current_dist_sq < best_dist_sq) {
                changed = true;
                best_dist_sq = current_dist_sq;
                loc = cand_loc_2f;
            }
        }

        if (changed)
            continue;

        step *= 0.5;
        if (step < min_step)
            break;
    }

    return sqrt(best_dist_sq);
}

static cv::Vec2f find_closest_intersection(QuadSurface* surface, const cv::Vec3f& p1, const cv::Vec3f& p2, const cv::Vec3f& proximity_point, float& line_dist, float& prox_dist)
{
    cv::Vec2f best_loc = {-1, -1};
    line_dist = -1.0f;
    prox_dist = -1.0f;

    cv::Size s_size = surface->size();
    cv::Vec2f scale = surface->scale();

    cv::Vec3f zero_ptr(0, 0, 0);
    cv::Vec3f center_in_points = surface->loc_raw(zero_ptr);

    srand(time(NULL));

    for (int i = 0; i < 1000; ++i) { // 1000 random trials
        cv::Vec2f nominal_loc = {
            (float)(rand() % s_size.width),
            (float)(rand() % s_size.height)
        };

        cv::Vec2f cand_loc_abs = { nominal_loc[0] * scale[0], nominal_loc[1] * scale[1] };

        cv::Vec3f ptr_loc = cv::Vec3f(cand_loc_abs[0], cand_loc_abs[1], 0) - center_in_points;
        if (!surface->valid(ptr_loc)) {
            continue;
        }

        float dist = find_intersection_direct(surface, cand_loc_abs, p1, p2, 16.0f, 0.0001f, center_in_points);

        if (dist < 0 || dist >= 0.01) {
            continue;
        }

        cv::Vec3f res_ptr = cv::Vec3f(cand_loc_abs[0], cand_loc_abs[1], 0) - center_in_points;
        cv::Vec3f intersection_3d = surface->coord(res_ptr);
        float current_prox_dist = cv::norm(intersection_3d - proximity_point);

        if (prox_dist < 0 || current_prox_dist < prox_dist) {
            prox_dist = current_prox_dist;
            line_dist = dist;
            best_loc = cand_loc_abs;
        }
    }

    return best_loc;
}


nlohmann::json calc_point_metrics(const VCCollection& collection, QuadSurface* surface, int z_min, int z_max)
{
    nlohmann::json results;
    int total_points_for_in_surface_metric = 0;
    int valid_in_surface_points = 0;

    for (const auto& pair : collection.getAllCollections()) {
        const auto& coll = pair.second;

        std::vector<ColPoint> points;
        for (const auto& p_pair : coll.points) {
            if (p_pair.second.p[2] >= z_min && p_pair.second.p[2] <= z_max) {
                points.push_back(p_pair.second);
            }
        }

        if (points.empty()) {
            continue;
        }

        std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) {
            return a.id < b.id;
        });

        for (size_t i = 0; i < points.size() - 1; ++i) {
            const auto& p1 = points[i];
            const auto& p2 = points[i+1];

            if (std::isnan(p1.winding_annotation) || std::isnan(p2.winding_annotation)) {
                continue;
            }

            if (round(p1.winding_annotation) != round(p2.winding_annotation)) {
                continue;
            }

            total_points_for_in_surface_metric++;

            cv::Vec3f diff = p1.p - p2.p;
            float dist_3d = cv::norm(diff);

            cv::Vec3f ptr1 = surface->pointer();
            surface->pointTo(ptr1, p1.p, 5.0);
            cv::Vec2f loc1(ptr1[0], ptr1[1]);

            cv::Vec3f ptr2 = surface->pointer();
            surface->pointTo(ptr2, p2.p, 5.0);
            cv::Vec2f loc2(ptr2[0], ptr2[1]);

            cv::Vec2f scale = surface->scale();
            cv::Vec2f diff_loc = (loc1 - loc2);
            diff_loc[0] /= scale[0];
            diff_loc[1] /= scale[1];
            float dist_surface = cv::norm(diff_loc);

            if (dist_3d > 1e-6 && dist_surface > 1e-6) {
                float ratio = std::max(dist_surface, dist_3d) / std::min(dist_surface, dist_3d);
                if (ratio <= 1.5) {
                    valid_in_surface_points++;
                }
            }
        }
    }

    if (total_points_for_in_surface_metric > 0) {
        results["in_surface_frac_valid"] = (float)valid_in_surface_points / total_points_for_in_surface_metric;
    }

    return results;
}

nlohmann::json calc_point_winding_metrics(const VCCollection& collection, QuadSurface* surface, const cv::Mat_<float>& winding, int z_min, int z_max)
{
    nlohmann::json results;

    int total_invalid_intersections = 0;
    int total_correct_winding = 0;
    int total_correct_winding_inv = 0;
    int total_comparisons = 0;
    int total_segments = 0;

    for (const auto& pair : collection.getAllCollections()) {
        const auto& coll = pair.second;

        std::vector<ColPoint> points_with_winding;
        for (const auto& p_pair : coll.points) {
            if (!std::isnan(p_pair.second.winding_annotation)) {
                if (p_pair.second.p[2] >= z_min && p_pair.second.p[2] <= z_max) {
                    points_with_winding.push_back(p_pair.second);
                }
            }
        }

        if (points_with_winding.size() < 2) {
            continue;
        }

        std::sort(points_with_winding.begin(), points_with_winding.end(), [](const auto& a, const auto& b) {
            return a.winding_annotation < b.winding_annotation;
        });

        std::vector<float> intersection_windings;
        for (size_t i = 0; i < points_with_winding.size() - 1; ++i) {
            total_segments++;
            const auto& p1_info = points_with_winding[i];
            const auto& p2_info = points_with_winding[i+1];

            float line_dist = -1.0f, prox_dist = -1.0f;
            cv::Vec2f loc = find_closest_intersection(surface, p1_info.p, p2_info.p, p1_info.p, line_dist, prox_dist);

            if (loc[0] < 0) {
                intersection_windings.push_back(NAN);
                total_invalid_intersections++;
            } else {
                float intersection_winding = winding(loc[1], loc[0]);
                intersection_windings.push_back(intersection_winding);
            }
        }

        if (points_with_winding.size() >= 3) {
            for (size_t i = 0; i < intersection_windings.size() - 1; ++i) {
                total_comparisons++;
                float prev_intersection = intersection_windings[i];
                float curr_intersection = intersection_windings[i+1];

                if (std::isnan(prev_intersection) || std::isnan(curr_intersection)) {
                    continue;
                }

                float annotated_winding_diff = points_with_winding[i+1].winding_annotation - points_with_winding[i].winding_annotation;
                float sampled_winding_diff = curr_intersection - prev_intersection;

                if (std::abs(sampled_winding_diff - annotated_winding_diff) < 0.1) {
                    total_correct_winding++;
                } else if (std::abs(-sampled_winding_diff - annotated_winding_diff) < 0.1) {
                    total_correct_winding_inv++;
                }
            }
        }
    }

    if (total_segments > 0) {
        results["surface_missing_fraction"] = (float)total_invalid_intersections / total_segments;
    }

    if (total_comparisons > 0) {
        results["winding_valid_fraction"] = (float)std::max(total_correct_winding, total_correct_winding_inv) / total_comparisons;
    }

    return results;
}
