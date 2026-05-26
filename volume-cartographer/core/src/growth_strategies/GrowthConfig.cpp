#include "GrowthConfig.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

namespace {

struct DirectionFlags {
    bool down = false;
    bool right = false;
    bool up = false;
    bool left = false;
    bool all_neighbors = false;
};

void append_unique_neigh(std::vector<cv::Vec2i>& neighs, const cv::Vec2i& value)
{
    for (const auto& existing : neighs) {
        if (existing == value) {
            return;
        }
    }
    neighs.push_back(value);
}

int64_t nonnegative_int64_param(const nlohmann::json& params,
                                const char* key,
                                int64_t default_value)
{
    return std::max<int64_t>(0, params.value(key, default_value));
}

DirectionFlags parse_direction_flags(const nlohmann::json& directions)
{
    DirectionFlags flags;
    if (!directions.is_array()) {
        flags.down = flags.right = flags.up = flags.left = true;
        flags.all_neighbors = true;
        return flags;
    }

    for (const auto& dir : directions) {
        if (!dir.is_string()) {
            continue;
        }
        const std::string value = dir.get<std::string>();
        if (value == "all") {
            flags.down = flags.right = flags.up = flags.left = true;
            flags.all_neighbors = true;
            break;
        }
        if (value == "down") {
            flags.down = true;
        } else if (value == "right") {
            flags.right = true;
        } else if (value == "up") {
            flags.up = true;
        } else if (value == "left") {
            flags.left = true;
        }
    }

    if (!flags.down && !flags.right && !flags.up && !flags.left) {
        flags.down = flags.right = flags.up = flags.left = true;
        flags.all_neighbors = true;
    }
    return flags;
}

void apply_growth_direction_flags(GrowthConfig& config, const DirectionFlags& flags)
{
    config.grow_down = flags.down;
    config.grow_right = flags.right;
    config.grow_up = flags.up;
    config.grow_left = flags.left;

    config.neighs.clear();
    if (flags.all_neighbors) {
        config.neighs = config.all_8_neighs;
        return;
    }
    if (flags.down) {
        append_unique_neigh(config.neighs, {1, 0});
    }
    if (flags.right) {
        append_unique_neigh(config.neighs, {0, 1});
    }
    if (flags.up) {
        append_unique_neigh(config.neighs, {-1, 0});
    }
    if (flags.left) {
        append_unique_neigh(config.neighs, {0, -1});
    }
}

void apply_expansion_direction_flags(GrowthConfig& config, const DirectionFlags& flags)
{
    config.expand_down = flags.down;
    config.expand_right = flags.right;
    config.expand_up = flags.up;
    config.expand_left = flags.left;
}

} // namespace

GrowthConfig parse_growth_config(const nlohmann::json& params, bool bidirectional, bool flip_x)
{
    GrowthConfig config;

    const std::string growth_mode = params.value("growth_mode", std::string("point"));
    config.has_growth_directions = params.contains("growth_directions") && params["growth_directions"].is_array();
    config.grow_left = bidirectional && !config.has_growth_directions;
    config.disable_grid_expansion = params.value("disable_grid_expansion", params.value("fill_growth", false));

    config.legacy_4_neighs = {
        { 1,  0},
        { 0,  1},
        {-1,  0},
        { 0, -1},
    };
    config.all_8_neighs = {
        { 1,  0},
        { 1,  1},
        { 0,  1},
        {-1,  1},
        {-1,  0},
        {-1, -1},
        { 0, -1},
        { 1, -1},
    };

    config.requested_neighbor_count = params.value("growth_neighbor_count", 4);
    if (config.requested_neighbor_count != 4 && config.requested_neighbor_count != 8) {
        std::cerr << "warning: growth_neighbor_count must be 4 or 8; defaulting to 4" << std::endl;
        config.requested_neighbor_count = 4;
    }
    config.max_no_growth_expansions = params.value("max_no_growth_expansions", 5);
    config.neighs = config.requested_neighbor_count == 8 ? config.all_8_neighs : config.legacy_4_neighs;
    if (growth_mode != "point") {
        std::cerr << "warning: unknown growth_mode '" << growth_mode
                  << "'; defaulting to point growth" << std::endl;
    }

    const std::string candidate_priority = params.value("candidate_priority", std::string("valid_neighbors"));
    config.candidate_priority_existing_depth = candidate_priority == "existing_depth";
    if (candidate_priority != "valid_neighbors" && candidate_priority != "existing_depth") {
        std::cerr << "warning: unknown candidate_priority '" << candidate_priority
                  << "'; defaulting to valid_neighbors" << std::endl;
    }
    config.candidate_support_depth_radius =
        config.candidate_priority_existing_depth ? std::max(1, params.value("candidate_support_depth_radius", 8)) : 0;
    config.rollout_growth = params.value("rollout_growth", false);
    config.rollout_width = std::max(1, params.value("rollout_width", 8));
    config.rollout_depth = std::max(1, params.value("rollout_depth", 2));
    config.rollout_max_children = std::max(1, params.value("rollout_max_children", 4));
    config.rollout_max_commits_per_generation =
        std::max(1, params.value("rollout_max_commits_per_generation", 1));
    config.rollout_min_separation =
        std::max(0, params.value("rollout_min_separation", config.rollout_depth + 1));
    config.rollout_area_weight = nonnegative_int64_param(params, "rollout_area_weight", 100);
    config.rollout_inlier_weight = nonnegative_int64_param(params, "rollout_inlier_weight", 1);
    config.rollout_connection_weight = nonnegative_int64_param(params, "rollout_connection_weight", 10);
    config.rollout_base_connection_weight = nonnegative_int64_param(params, "rollout_base_connection_weight", 0);
    config.rollout_internal_connection_weight = nonnegative_int64_param(params, "rollout_internal_connection_weight", 0);

    if (config.has_growth_directions) {
        apply_growth_direction_flags(config, parse_direction_flags(params["growth_directions"]));
    }

    config.expand_down = config.grow_down;
    config.expand_right = config.grow_right;
    config.expand_up = config.grow_up;
    config.expand_left = config.grow_left;
    if (params.contains("expansion_directions")) {
        apply_expansion_direction_flags(config, parse_direction_flags(params["expansion_directions"]));
    }
    if (flip_x) {
        std::swap(config.grow_left, config.grow_right);
        std::swap(config.expand_left, config.expand_right);
        for (auto& neigh : config.neighs) {
            neigh[1] = -neigh[1];
        }
    }
    return config;
}

void GrowthConfig::log(std::ostream& out, int stop_gen) const
{
    out << "growth directions:"
        << " down=" << grow_down
        << " right=" << grow_right
        << " up=" << grow_up
        << " left=" << grow_left
        << " expansion down=" << expand_down
        << " right=" << expand_right
        << " up=" << expand_up
        << " left=" << expand_left
        << " neighbor_count=" << neighs.size()
        << " max_no_growth_expansions=" << max_no_growth_expansions
        << " expand_grid=" << !disable_grid_expansion
        << " growth_mode=point"
        << " candidate_priority=" << (candidate_priority_existing_depth ? "existing_depth" : "valid_neighbors")
        << " support_depth_radius=" << candidate_support_depth_radius
        << " steps=" << stop_gen << std::endl;
    if (rollout_growth) {
        out << "rollout growth:"
            << " width=" << rollout_width
            << " depth=" << rollout_depth
            << " max_children=" << rollout_max_children
            << " max_commits_per_generation=" << rollout_max_commits_per_generation
            << " min_separation=" << rollout_min_separation
            << " area_weight=" << rollout_area_weight
            << " inlier_weight=" << rollout_inlier_weight
            << " connection_weight=" << rollout_connection_weight
            << " base_connection_weight=" << rollout_base_connection_weight
            << " internal_connection_weight=" << rollout_internal_connection_weight << std::endl;
    }
}
