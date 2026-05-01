#include "GrowthConfig.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>

namespace {

void append_unique_neigh(std::vector<cv::Vec2i>& neighs, const cv::Vec2i& value)
{
    for (const auto& existing : neighs) {
        if (existing == value) {
            return;
        }
    }
    neighs.push_back(value);
}

} // namespace

GrowthConfig parse_growth_config(const nlohmann::json& params, bool bidirectional, bool flip_x)
{
    GrowthConfig config;

    const std::string growth_mode = params.value("growth_mode", std::string("point"));
    config.radial_growth = growth_mode == "radial";
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
    if (growth_mode != "point" &&
        growth_mode != "radial") {
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
    config.radial_prune_interval = std::max(0, params.value("frontier_prune_interval", 0));
    config.radial_prune_attempts = std::max(1, params.value("frontier_prune_attempts", 3));
    config.radial_prune_max_points = std::max(1, params.value("frontier_prune_max_points", 32));
    config.radial_prune_max_neighbors = std::max(0, params.value("frontier_prune_max_neighbors", 2));

    if (config.has_growth_directions) {
        config.grow_down = config.grow_right = config.grow_up = config.grow_left = false;
        config.neighs.clear();
        for (const auto& dir : params["growth_directions"]) {
            if (!dir.is_string()) {
                continue;
            }
            const std::string value = dir.get<std::string>();
            if (value == "all") {
                config.grow_down = config.grow_right = config.grow_up = config.grow_left = true;
                config.neighs = config.all_8_neighs;
                break;
            }
            if (value == "down") {
                config.grow_down = true;
                append_unique_neigh(config.neighs, {1, 0});
            }
            else if (value == "right") {
                config.grow_right = true;
                append_unique_neigh(config.neighs, {0, 1});
            }
            else if (value == "up") {
                config.grow_up = true;
                append_unique_neigh(config.neighs, {-1, 0});
            }
            else if (value == "left") {
                config.grow_left = true;
                append_unique_neigh(config.neighs, {0, -1});
            }
        }
        if (!config.grow_down && !config.grow_right && !config.grow_up && !config.grow_left) {
            config.grow_down = config.grow_right = config.grow_up = config.grow_left = true;
            config.neighs = config.all_8_neighs;
        }
    }
    if (flip_x) {
        std::swap(config.grow_left, config.grow_right);
        for (auto& neigh : config.neighs) {
            neigh[1] = -neigh[1];
        }
    }
    if (config.radial_growth && config.has_growth_directions) {
        std::vector<cv::Vec2i> radial_neighs;
        auto add_radial_neigh = [&](bool enabled, const cv::Vec2i& value) {
            if (enabled) {
                append_unique_neigh(radial_neighs, value);
            }
        };

        add_radial_neigh(config.grow_down, {1, 0});
        add_radial_neigh(config.grow_right, {0, 1});
        add_radial_neigh(config.grow_up, {-1, 0});
        add_radial_neigh(config.grow_left, {0, -1});

        if (config.requested_neighbor_count == 8) {
            add_radial_neigh(config.grow_down && config.grow_right, {1, 1});
            add_radial_neigh(config.grow_down && config.grow_left, {1, -1});
            add_radial_neigh(config.grow_up && config.grow_right, {-1, 1});
            add_radial_neigh(config.grow_up && config.grow_left, {-1, -1});
        }

        if (!radial_neighs.empty()) {
            config.neighs = std::move(radial_neighs);
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
        << " neighbor_count=" << neighs.size()
        << " max_no_growth_expansions=" << max_no_growth_expansions
        << " expand_grid=" << !disable_grid_expansion
        << " growth_mode=" << (radial_growth ? "radial" : "point")
        << " candidate_priority=" << (candidate_priority_existing_depth ? "existing_depth" : "valid_neighbors")
        << " support_depth_radius=" << candidate_support_depth_radius
        << " steps=" << stop_gen << std::endl;
    if (radial_growth) {
        out << "radial growth:"
            << " prune_interval=" << radial_prune_interval
            << " prune_attempts=" << radial_prune_attempts
            << " prune_max_points=" << radial_prune_max_points
            << " prune_max_neighbors=" << radial_prune_max_neighbors << std::endl;
    }
}
