#pragma once

#include <opencv2/core.hpp>

#include <nlohmann/json.hpp>

#include <iosfwd>
#include <vector>

struct GrowthConfig {
    bool radial_growth = false;
    bool has_growth_directions = false;
    bool grow_down = false;
    bool grow_right = true;
    bool grow_up = false;
    bool grow_left = false;
    bool disable_grid_expansion = false;
    bool candidate_priority_existing_depth = false;

    int requested_neighbor_count = 4;
    int max_no_growth_expansions = 5;
    int candidate_support_depth_radius = 0;
    int radial_prune_interval = 0;
    int radial_prune_attempts = 3;
    int radial_prune_max_points = 32;
    int radial_prune_max_neighbors = 2;

    std::vector<cv::Vec2i> legacy_4_neighs;
    std::vector<cv::Vec2i> all_8_neighs;
    std::vector<cv::Vec2i> neighs;

    void log(std::ostream& out, int stop_gen) const;
};

GrowthConfig parse_growth_config(const nlohmann::json& params, bool bidirectional, bool flip_x);
