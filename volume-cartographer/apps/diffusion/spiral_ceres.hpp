#pragma once

#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

#include <vc/core/util/GridStore.hpp>
#include "spiral_common.hpp"

void visualize_normal_constraints(const cv::Mat& sheet_binary, const std::vector<std::pair<cv::Point, cv::Point>>& constraints, const std::string& path);

ceres::CostFunction* CreateNormalConstraint(const vc::core::util::GridStore& normal_grid, float roi_radius, double weight);
ceres::CostFunction* CreateSnappingConstraint(const vc::core::util::GridStore& normal_grid, float roi_radius, double weight, double snap_trig_th, double snap_search_range);


// Ceres cost functor for the distance constraint
struct PointToLineDistanceConstraint {
    PointToLineDistanceConstraint(double weight) : weight_(weight) {}

    template <typename T>
    bool operator()(const T* const p_self, const T* const p_n1, const T* const p_n2, const T* const dist, T* residual) const {
        T v_x = p_n2[0] - p_n1[0];
        T v_y = p_n2[1] - p_n1[1];
        T w_x = p_self[0] - p_n1[0];
        T w_y = p_self[1] - p_n1[1];

        T c1 = w_x * v_x + w_y * v_y;
        T c2 = v_x * v_x + v_y * v_y;

        // Check if the projection is within the segment
        if (c1 < T(0.0) || c1 > c2) {
            residual[0] = T(0.0); // No constraint if outside
            return true;
        }

        T b = c1 / c2;
        T p_proj_x = p_n1[0] + b * v_x;
        T p_proj_y = p_n1[1] + b * v_y;

        T diff_x = p_self[0] - p_proj_x;
        T diff_y = p_self[1] - p_proj_y;
        T true_distance = ceres::sqrt(diff_x * diff_x + diff_y * diff_y);

        T sign = (w_x * v_y - w_y * v_x > T(0.0)) ? T(1.0) : T(-1.0);

        residual[0] = T(weight_) * (sign * true_distance - (*dist));
        return true;
    }

    static ceres::CostFunction* Create(double weight) {
        return (new ceres::AutoDiffCostFunction<PointToLineDistanceConstraint, 1, 2, 2, 2, 1>(
            new PointToLineDistanceConstraint(weight)));
    }
private:
    const double weight_;
};

struct SpacingConstraint {
    SpacingConstraint(double target_distance, double weight)
    : target_distance_(target_distance), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const p1, const T* const p2, T* residual) const {
        T dx = p1[0] - p2[0];
        T dy = p1[1] - p2[1];
        T dist_norm = ceres::sqrt(dx * dx + dy * dy)/T(target_distance_);
        //FIXME
        if (dist_norm < T(1.0))
            residual[0] = T(weight_) * (dist_norm-T(1.0))/dist_norm;
        else
            residual[0] = T(weight_) * (dist_norm-T(1.0));

        // residual[0] = T(weight_) * (ceres::sqrt(dx * dx + dy * dy)/T(target_distance_) - T(1.0));
        return true;
    }

    static ceres::CostFunction* Create(double target_distance, double weight) {
        return (new ceres::AutoDiffCostFunction<SpacingConstraint, 1, 2, 2>(
            new SpacingConstraint(target_distance, weight)));
    }

private:
    double target_distance_;
    double weight_;
};

struct SpacingSmoothnessConstraint {
    SpacingSmoothnessConstraint(double weight) : weight_(weight) {}

    template <typename T>
    bool operator()(const T* const p1, const T* const p2, const T* const p3, T* residual) const {
        T dx_13 = p3[0] - p1[0];
        T dy_13 = p3[1] - p1[1];
        T dist_13_sq = dx_13 * dx_13 + dy_13 * dy_13;
        T dist_13 = ceres::sqrt(dist_13_sq + T(1e-9));

        T avg_p1_p3_x = (p1[0] + p3[0]) * T(0.5);
        T avg_p1_p3_y = (p1[1] + p3[1]) * T(0.5);

        residual[0] = T(weight_) * (p2[0] - avg_p1_p3_x) / dist_13;
        residual[1] = T(weight_) * (p2[1] - avg_p1_p3_y) / dist_13;

        return true;
    }

    static ceres::CostFunction* Create(double weight) {
        return (new ceres::AutoDiffCostFunction<SpacingSmoothnessConstraint, 2, 2, 2, 2>(
            new SpacingSmoothnessConstraint(weight)));
    }

private:
    const double weight_;
};

struct SpacingDistConstraint {
    SpacingDistConstraint(double initial_dist, double weight) : initial_dist_(initial_dist), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const dist, T* residual) const {
        residual[0] = T(weight_) * (*dist - T(initial_dist_));
        return true;
    }

    static ceres::CostFunction* Create(double initial_dist, double weight) {
        return (new ceres::AutoDiffCostFunction<SpacingDistConstraint, 1, 1>(
            new SpacingDistConstraint(initial_dist, weight)));
    }

private:
    const double initial_dist_;
    const double weight_;
};

struct MinDistanceConstraint {
    MinDistanceConstraint(double min_distance, double weight) : min_distance_(min_distance), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const dist, T* residual) const {
        if (*dist < T(min_distance_)) {
            residual[0] = T(weight_) * (T(min_distance_) - *dist);
        } else {
            residual[0] = T(0.0);
        }
        return true;
    }

    static ceres::CostFunction* Create(double min_distance, double weight) {
        return (new ceres::AutoDiffCostFunction<MinDistanceConstraint, 1, 1>(
            new MinDistanceConstraint(min_distance, weight)));
    }

private:
    const double min_distance_;
    const double weight_;
};

// Ceres cost functor for the smoothness constraint
struct SmoothnessConstraint {
    SmoothnessConstraint(double weight) : weight_(weight) {}

    template <typename T>
    bool operator()(const T* const dist1, const T* const dist2, T* residual) const {
        residual[0] = (*dist1 - *dist2) * T(weight_);
        return true;
    }

    static ceres::CostFunction* Create(double weight) {
        return (new ceres::AutoDiffCostFunction<SmoothnessConstraint, 1, 1, 1>(
            new SmoothnessConstraint(weight)));
    }

private:
    const double weight_;
};



struct FractionalConstraint {
    FractionalConstraint(double initial_fraction, double weight)
    : initial_fraction_(initial_fraction), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const p_self, const T* const p_n1, const T* const p_n2, T* residual) const {
        T v_x = p_n2[0] - p_n1[0];
        T v_y = p_n2[1] - p_n1[1];
        T w_x = p_self[0] - p_n1[0];
        T w_y = p_self[1] - p_n1[1];

        T c1 = w_x * v_x + w_y * v_y;
        T c2 = v_x * v_x + v_y * v_y;

        T current_fraction = c1 / (c2 + T(1e-9));

        residual[0] = T(weight_) * (current_fraction - T(initial_fraction_));
        return true;
    }

    static ceres::CostFunction* Create(double initial_fraction, double weight) {
        return (new ceres::AutoDiffCostFunction<FractionalConstraint, 1, 2, 2, 2>(
            new FractionalConstraint(initial_fraction, weight)));
    }

private:
    const double initial_fraction_;
    const double weight_;
};

struct MaxAngleConstraint {
    MaxAngleConstraint(double max_angle_deg, double weight)
    : max_angle_rad_(max_angle_deg * CV_PI / 180.0), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const p1, const T* const p2, const T* const p3, T* residual) const {
        T v1_x = p2[0] - p1[0];
        T v1_y = p2[1] - p1[1];
        T v2_x = p3[0] - p2[0];
        T v2_y = p3[1] - p2[1];

        T dot_product = v1_x * v2_x + v1_y * v2_y;
        T len1_sq = v1_x * v1_x + v1_y * v1_y;
        T len2_sq = v2_x * v2_x + v2_y * v2_y;

        T cos_angle = dot_product / (ceres::sqrt(len1_sq * len2_sq) + T(1e-9));

        // Clamp cos_angle to avoid domain errors with acos
        cos_angle = std::max(T(-1.0), std::min(T(1.0), cos_angle));
        T angle = ceres::acos(cos_angle);

        if (angle > T(max_angle_rad_)) {
            T angle_diff = angle - T(max_angle_rad_);
            residual[0] = T(weight_) * angle_diff * angle_diff;
        } else {
            residual[0] = T(0.0);
        }

        return true;
    }

    static ceres::CostFunction* Create(double max_angle_deg, double weight) {
        return (new ceres::AutoDiffCostFunction<MaxAngleConstraint, 1, 2, 2, 2>(
            new MaxAngleConstraint(max_angle_deg, weight)));
    }

private:
    const double max_angle_rad_;
    const double weight_;
};

struct EvenSpacingConstraint {
    EvenSpacingConstraint(double weight) : weight_(weight) {}

    template <typename T>
    bool operator()(const T* const p1, const T* const p2, const T* const p3, T* residual) const {
        T dx1 = p2[0] - p1[0];
        T dy1 = p2[1] - p1[1];
        T len1 = ceres::sqrt(dx1 * dx1 + dy1 * dy1);

        T dx2 = p3[0] - p2[0];
        T dy2 = p3[1] - p2[1];
        T len2 = ceres::sqrt(dx2 * dx2 + dy2 * dy2);

        residual[0] = T(weight_) * (len1 - len2);
        return true;
    }

    static ceres::CostFunction* Create(double weight) {
        return (new ceres::AutoDiffCostFunction<EvenSpacingConstraint, 1, 2, 2, 2>(
            new EvenSpacingConstraint(weight)));
    }

private:
    const double weight_;
};

struct ConstraintInfluence {
    double target_distance;
    double weight;
};

struct PointInfluences {
    std::vector<ConstraintInfluence> low_influences;
    std::vector<ConstraintInfluence> high_influences;
};

struct SheetDistanceConstraint {
    SheetDistanceConstraint(const PointInfluences* influences) : influences_(influences) {}

    template <typename T>
    bool operator()(const T* const dist_low, const T* const dist_high, T* residuals) const {
        if (!influences_) {
            residuals[0] = T(0.0);
            residuals[1] = T(0.0);
            return true;
        }

        T total_weighted_error_low = T(0.0);
        T total_weight_low = T(0.0);
        for (const auto& influence : influences_->low_influences) {
            total_weighted_error_low += T(influence.weight) * (*dist_low - T(influence.target_distance));
            total_weight_low += T(influence.weight);
        }

        T total_weighted_error_high = T(0.0);
        T total_weight_high = T(0.0);
        for (const auto& influence : influences_->high_influences) {
            total_weighted_error_high += T(influence.weight) * (*dist_high - T(influence.target_distance));
            total_weight_high += T(influence.weight);
        }

        residuals[0] = (total_weight_low > T(1e-9)) ? (total_weighted_error_low / total_weight_low) : T(0.0);
        residuals[1] = (total_weight_high > T(1e-9)) ? (total_weighted_error_high / total_weight_high) : T(0.0);

        return true;
    }

    static ceres::CostFunction* Create(const PointInfluences* influences) {
        return new ceres::AutoDiffCostFunction<SheetDistanceConstraint, 2, 1, 1>(
            new SheetDistanceConstraint(influences)
        );
    }

private:
    const PointInfluences* influences_;
};


//apply a snapping constraint from p2 to p1 (if p2 is within snap_trig_th of a linesegment and p1 is within search_range of a neighboring segment we try to pull it towards that
struct SnappingConstraint {
    static double val(const double& v) { return v; }
    template <typename JetT>
    static double val(const JetT& v) { return v.a; }

    const vc::core::util::GridStore& normal_grid;
    const float roi_radius;
    const double weight;
    const double snap_trig_th;
    const double snap_search_range;

    SnappingConstraint(const vc::core::util::GridStore& normal_grid, float roi_radius, double weight, double snap_trig_th, double snap_search_range)
    : normal_grid(normal_grid), roi_radius(roi_radius), weight(weight), snap_trig_th(snap_trig_th), snap_search_range(snap_search_range) {}

    static float point_line_dist_sq(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b) {
        cv::Point2f ab = b - a;
        cv::Point2f ap = p - a;
        float ab_len_sq = ab.dot(ab);
        if (ab_len_sq < 1e-9) {
            return ap.dot(ap);
        }
        float t = ap.dot(ab) / ab_len_sq;
        t = std::max(0.0f, std::min(1.0f, t));
        cv::Point2f projection = a + t * ab;
        return (p - projection).dot(p - projection);
    }

    template <typename T>
    static T point_line_dist_sq_differentiable(const T* p, const cv::Point2f& a, const cv::Point2f& b) {
        T ab_x = T(b.x - a.x);
        T ab_y = T(b.y - a.y);
        T ap_x = p[0] - T(a.x);
        T ap_y = p[1] - T(a.y);

        T ab_len_sq = ab_x * ab_x + ab_y * ab_y;
        if (ab_len_sq < T(1e-9)) {
            return ap_x * ap_x + ap_y * ap_y;
        }
        T t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq;

        // Clamping t using conditionals that are safe for Jets
        if (t < T(0.0)) t = T(0.0);
        if (t > T(1.0)) t = T(1.0);

        T proj_x = T(a.x) + t * ab_x;
        T proj_y = T(a.y) + t * ab_y;

        T dx = p[0] - proj_x;
        T dy = p[1] - proj_y;
        return dx * dx + dy * dy;
    }

    template <typename T>
    bool operator()(const T* const p1, const T* const p2, T* residual) const {
        cv::Point2f p1_cv(val(p1[0]), val(p1[1]));
        cv::Point2f p2_cv(val(p2[0]), val(p2[1]));
        cv::Point2f midpoint_cv = (p1_cv + p2_cv) * 0.5f;

        std::vector<std::shared_ptr<std::vector<cv::Point>>> nearby_paths = normal_grid.get(midpoint_cv, roi_radius);
        residual[0] = T(0.0);

        if (nearby_paths.empty()) {
            return true;
        }

        float closest_dist_norm = std::numeric_limits<float>::max();
        int best_path_idx = -1;
        int best_seg_idx = -1;
        bool best_is_next = false;

        for (int path_idx = 0; path_idx < nearby_paths.size(); ++path_idx) {
            const auto& path = *nearby_paths[path_idx];
            if (path.size() < 2) continue;

            for (int i = 0; i < path.size() - 1; ++i) {
                float d2_sq = point_line_dist_sq(p2_cv, path[i], path[i+1]);
                if (d2_sq >= snap_trig_th * snap_trig_th) continue;

                if (i < path.size() - 2) { // Check next segment
                    float d1_sq = point_line_dist_sq(p1_cv, path[i+1], path[i+2]);
                    if (d1_sq < snap_search_range * snap_search_range) {
                        float dist_norm = 0.5f * (sqrt(d1_sq)/snap_search_range + sqrt(d2_sq)/snap_trig_th);
                        if (dist_norm < closest_dist_norm) {
                            closest_dist_norm = dist_norm;
                            best_path_idx = path_idx;
                            best_seg_idx = i;
                            best_is_next = true;
                        }
                    }
                }
                if (i > 0) { // Check prev segment
                    float d1_sq = point_line_dist_sq(p1_cv, path[i-1], path[i]);
                     if (d1_sq < snap_search_range * snap_search_range) {
                        float dist_norm = 0.5f * (sqrt(d1_sq)/snap_search_range + sqrt(d2_sq)/snap_trig_th);
                        if (dist_norm < closest_dist_norm) {
                            closest_dist_norm = dist_norm;
                            best_path_idx = path_idx;
                            best_seg_idx = i;
                            best_is_next = false;
                        }
                    }
                }
            }
        }

        if (best_path_idx != -1) {
            const auto& best_path = *nearby_paths[best_path_idx];
            const auto& seg2_p1 = best_path[best_seg_idx];
            const auto& seg2_p2 = best_path[best_seg_idx + 1];

            cv::Point2f seg1_p1, seg1_p2;
            if (best_is_next) {
                seg1_p1 = best_path[best_seg_idx + 1];
                seg1_p2 = best_path[best_seg_idx + 2];
            } else {
                seg1_p1 = best_path[best_seg_idx - 1];
                seg1_p2 = best_path[best_seg_idx];
            }

            T d1_sq = point_line_dist_sq_differentiable(p1, seg1_p1, seg1_p2);
            T d2_sq = point_line_dist_sq_differentiable(p2, seg2_p1, seg2_p2);

            T d1_norm = ceres::sqrt(d1_sq) / T(snap_search_range);
            T d2_norm = ceres::sqrt(d2_sq) / T(snap_trig_th);

            residual[0] = T(weight) * (d1_norm * (T(1.0) - d2_norm) + d2_norm);
        } else {
            residual[0] = T(weight);
        }

        return true;
    }
};

struct NormalConstraint {
    static double val(const double& v) { return v; }
    template <typename JetT>
    static double val(const JetT& v) { return v.a; }

    const vc::core::util::GridStore& normal_grid;
    const float roi_radius;
    const double weight;

    NormalConstraint(const vc::core::util::GridStore& normal_grid, float roi_radius, double weight)
    : normal_grid(normal_grid), roi_radius(roi_radius), weight(weight) {}

    // Function to calculate the squared distance between two line segments
    static float seg_dist_sq(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4) {
        auto dot = [](cv::Point2f a, cv::Point2f b) { return a.x * b.x + a.y * b.y; };
        auto dist_sq = [&](cv::Point2f p) { return p.x * p.x + p.y * p.y; };

        cv::Point2f u = p2 - p1;
        cv::Point2f v = p4 - p3;
        cv::Point2f w = p1 - p3;

        float a = dot(u, u);
        float b = dot(u, v);
        float c = dot(v, v);
        float d = dot(u, w);
        float e = dot(v, w);
        float D = a * c - b * b;
        float sc, sN, sD = D;
        float tc, tN, tD = D;

        if (D < 1e-7) {
            sN = 0.0;
            sD = 1.0;
            tN = e;
            tD = c;
        } else {
            sN = (b * e - c * d);
            tN = (a * e - b * d);
            if (sN < 0.0) {
                sN = 0.0;
                tN = e;
                tD = c;
            } else if (sN > sD) {
                sN = sD;
                tN = e + b;
                tD = c;
            }
        }

        if (tN < 0.0) {
            tN = 0.0;
            if (-d < 0.0) sN = 0.0;
            else if (-d > a) sN = sD;
            else {
                sN = -d;
                sD = a;
            }
        } else if (tN > tD) {
            tN = tD;
            if ((-d + b) < 0.0) sN = 0.0;
            else if ((-d + b) > a) sN = sD;
            else {
                sN = (-d + b);
                sD = a;
            }
        }

        sc = (std::abs(sN) < 1e-7 ? 0.0 : sN / sD);
        tc = (std::abs(tN) < 1e-7 ? 0.0 : tN / tD);

        cv::Point2f dP = w + (sc * u) - (tc * v);
        return dist_sq(dP);
    }

    template <typename T>
    bool operator()(const T* const p1, const T* const p2, T* residual) const {
        T edge_vec_x = p2[0] - p1[0];
        T edge_vec_y = p2[1] - p1[1];

        T edge_len_sq = edge_vec_x * edge_vec_x + edge_vec_y * edge_vec_y;
        if (edge_len_sq < T(1e-12)) {
            residual[0] = T(0.0);
            return true;
        }
        T edge_len = ceres::sqrt(edge_len_sq);

        T edge_normal_x = edge_vec_y / edge_len;
        T edge_normal_y = -edge_vec_x / edge_len;

        cv::Point2f midpoint_cv(val(p1[0] + edge_vec_x * 0.5), val(p1[1] + edge_vec_y * 0.5));
        std::vector<std::shared_ptr<std::vector<cv::Point>>> nearby_paths = normal_grid.get(midpoint_cv, roi_radius);

        residual[0] = T(0.0);
        if (nearby_paths.empty()) {
            return true;
        }

        T total_weighted_dot_product = T(0.0);
        T total_weight = T(0.0);

        for (const auto& path_ptr : nearby_paths) {
            const auto& path = *path_ptr;
            if (path.size() < 2) continue;

            for (size_t i = 0; i < path.size() - 1; ++i) {
                cv::Point2f p_a = path[i];
                cv::Point2f p_b = path[i+1];

                cv::Point2f p1_cv(val(p1[0]), val(p1[1]));
                cv::Point2f p2_cv(val(p2[0]), val(p2[1]));

                float dist_sq = NormalConstraint::seg_dist_sq(p1_cv, p2_cv, p_a, p_b);
                dist_sq = std::max(0.1f, dist_sq);

                T weight_n = T(1.0 / ceres::sqrt(dist_sq));

                cv::Point2f tangent = p_b - p_a;
                float length = cv::norm(tangent);
                if (length > 0) {
                    tangent /= length;
                }
                cv::Point2f normal(-tangent.y, tangent.x);

                T dot_product = ceres::abs(edge_normal_x * T(normal.x) + edge_normal_y * T(normal.y));

                total_weighted_dot_product += weight_n * dot_product;
                total_weight += weight_n;
            }
        }

        if (total_weight > T(1e-9)) {
            T avg_dot_product = total_weighted_dot_product / total_weight;
            residual[0] = T(weight) * (T(1.0) - avg_dot_product);
        } else {
            residual[0] = T(0.0);
        }

        return true;
    }
};



class SheetConstraintCallback : public ceres::IterationCallback {
public:
    SheetConstraintCallback(
        std::vector<SpiralPoint>* all_points,
        const std::vector<double>* positions,
        const std::vector<SheetConstraintRay>* constraint_rays,
        std::vector<PointInfluences>* all_influences,
        const cv::Point* umbilicus
    );

    virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary);

private:
    std::vector<SpiralPoint>* all_points_;
    const std::vector<double>* positions_;
    const std::vector<SheetConstraintRay>* constraint_rays_;
    std::vector<PointInfluences>* all_influences_;
    const cv::Point* umbilicus_;
};
