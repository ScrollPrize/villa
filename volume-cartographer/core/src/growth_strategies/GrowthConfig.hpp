#pragma once

#include <opencv2/core.hpp>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <iosfwd>
#include <vector>

struct GrowthConfig {
    bool has_growth_directions = false;
    bool grow_down = false;
    bool grow_right = true;
    bool grow_up = false;
    bool grow_left = false;
    bool disable_grid_expansion = false;
    bool candidate_priority_existing_depth = false;
    bool rollout_growth = false;

    int requested_neighbor_count = 4;
    int max_no_growth_expansions = 5;
    int candidate_support_depth_radius = 0;
    int rollout_width = 8;
    int rollout_depth = 2;
    int rollout_max_children = 4;
    int rollout_max_commits_per_generation = 1;
    int rollout_min_separation = 3;
    int64_t rollout_area_weight = 100;
    int64_t rollout_inlier_weight = 1;
    int64_t rollout_connection_weight = 10;
    int64_t rollout_base_connection_weight = 0;
    int64_t rollout_internal_connection_weight = 0;

    std::vector<cv::Vec2i> legacy_4_neighs;
    std::vector<cv::Vec2i> all_8_neighs;
    std::vector<cv::Vec2i> neighs;

    void log(std::ostream& out, int stop_gen) const;
};

GrowthConfig parse_growth_config(const nlohmann::json& params, bool bidirectional, bool flip_x);
