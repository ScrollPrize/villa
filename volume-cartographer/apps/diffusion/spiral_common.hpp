#pragma once

#include "common.hpp"
#include <nlohmann/json.hpp>

struct SpiralIntersection;

struct SheetConstraintRay {
    cv::Point2f dir;
    std::vector<std::pair<cv::Point, cv::Point>> constraints;
};

struct SpiralPoint {
    cv::Vec3d pos;
    double winding;
    double dist_low;
    double dist_high;
    std::vector<std::pair<int, double>> neighbors_low;
    std::vector<std::pair<int, double>> neighbors_high;
    double fraction_low;
    double fraction_high;
};

bool find_intersections(
    const cv::Point2f& ray_origin,
    const cv::Point2f& ray_dir,
    const std::vector<SpiralPoint>& all_points,
    std::vector<SpiralIntersection>& intersections
);

void visualize_spiral(
    cv::Mat& viz,
    const std::vector<SpiralPoint>& all_points,
    const cv::Scalar& line_color,
    const cv::Scalar& point_color,
    bool draw_winding_text
);

void visualize_spiral(
    const std::vector<SpiralPoint>& all_points,
    const cv::Size& slice_size,
    const fs::path& output_path,
    const cv::Scalar& point_color,
    const std::vector<SheetConstraintRay>& constraint_rays,
    bool draw_influence,
    bool draw_winding_text
);

inline void to_json(nlohmann::json& j, const SpiralPoint& p) {
    j = nlohmann::json{
        {"pos", {p.pos[0], p.pos[1], p.pos[2]}},
        {"winding", p.winding}
    };
}

inline void from_json(const nlohmann::json& j, SpiralPoint& p) {
    std::vector<double> pos_vec;
    j.at("pos").get_to(pos_vec);
    if (pos_vec.size() == 3) {
        p.pos = cv::Vec3d(pos_vec[0], pos_vec[1], pos_vec[2]);
    }
    j.at("winding").get_to(p.winding);
}

struct SpiralIntersection {
    int point_idx1;
    int point_idx2;
    double t;
    cv::Point2d intersection_point;
    double winding;
};
