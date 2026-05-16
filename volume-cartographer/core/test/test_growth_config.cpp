// Coverage for core/src/growth_strategies/GrowthConfig.cpp: parses a JSON
// blob into a strongly-typed GrowthConfig and clamps invalid values.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/growth_strategies/GrowthConfig.hpp"

#include <nlohmann/json.hpp>

#include <sstream>

using nlohmann::json;

TEST_CASE("parse_growth_config: empty object yields sane defaults")
{
    json p = json::object();
    GrowthConfig cfg = parse_growth_config(p, /*bidirectional=*/false, /*flip_x=*/false);
    CHECK_FALSE(cfg.has_growth_directions);
    CHECK(cfg.requested_neighbor_count == 4);
    CHECK(cfg.neighs.size() == 4);
    CHECK(cfg.max_no_growth_expansions == 5);
    CHECK_FALSE(cfg.rollout_growth);
}

TEST_CASE("parse_growth_config: bidirectional enables grow_left when no directions")
{
    json p = json::object();
    auto cfg = parse_growth_config(p, /*bidirectional=*/true, /*flip_x=*/false);
    CHECK(cfg.grow_left);
}

TEST_CASE("parse_growth_config: null params would throw — caller must pass object")
{
    json p; // default-constructed null
    CHECK_THROWS(parse_growth_config(p, false, false));
}

TEST_CASE("parse_growth_config: growth_neighbor_count=8 picks all_8_neighs")
{
    json p; p["growth_neighbor_count"] = 8;
    auto cfg = parse_growth_config(p, false, false);
    CHECK(cfg.requested_neighbor_count == 8);
    CHECK(cfg.neighs.size() == 8);
}

TEST_CASE("parse_growth_config: invalid neighbor count clamped to 4 with warning")
{
    json p; p["growth_neighbor_count"] = 7;
    auto cfg = parse_growth_config(p, false, false);
    CHECK(cfg.requested_neighbor_count == 4);
    CHECK(cfg.neighs.size() == 4);
}

TEST_CASE("parse_growth_config: candidate_priority existing_depth sets support radius")
{
    json p;
    p["candidate_priority"] = "existing_depth";
    p["candidate_support_depth_radius"] = 12;
    auto cfg = parse_growth_config(p, false, false);
    CHECK(cfg.candidate_priority_existing_depth);
    CHECK(cfg.candidate_support_depth_radius == 12);
}

TEST_CASE("parse_growth_config: unknown candidate_priority emits warning, defaults")
{
    json p;
    p["candidate_priority"] = "garbage";
    auto cfg = parse_growth_config(p, false, false);
    CHECK_FALSE(cfg.candidate_priority_existing_depth);
    CHECK(cfg.candidate_support_depth_radius == 0);
}

TEST_CASE("parse_growth_config: unknown growth_mode warns and defaults")
{
    json p; p["growth_mode"] = "non_point";
    auto cfg = parse_growth_config(p, false, false);
    // Should not crash; defaults apply.
    CHECK(cfg.requested_neighbor_count == 4);
}

TEST_CASE("parse_growth_config: rollout block parses with min clamps")
{
    json p;
    p["rollout_growth"] = true;
    p["rollout_width"] = 0;       // clamped up to 1
    p["rollout_depth"] = 0;       // clamped up to 1
    p["rollout_max_children"] = 0; // clamped up to 1
    p["rollout_max_commits_per_generation"] = 0;
    p["rollout_min_separation"] = -5;
    p["rollout_area_weight"] = -10;
    p["rollout_inlier_weight"] = -1;
    p["rollout_connection_weight"] = -100;
    auto cfg = parse_growth_config(p, false, false);
    CHECK(cfg.rollout_growth);
    CHECK(cfg.rollout_width == 1);
    CHECK(cfg.rollout_depth == 1);
    CHECK(cfg.rollout_max_children == 1);
    CHECK(cfg.rollout_max_commits_per_generation == 1);
    CHECK(cfg.rollout_min_separation == 0);
    CHECK(cfg.rollout_area_weight == 0);
    CHECK(cfg.rollout_inlier_weight == 0);
    CHECK(cfg.rollout_connection_weight == 0);
}

TEST_CASE("parse_growth_config: growth_directions=[all] picks 8-neigh")
{
    json p;
    p["growth_directions"] = {"all"};
    auto cfg = parse_growth_config(p, false, false);
    CHECK(cfg.has_growth_directions);
    CHECK(cfg.grow_down);
    CHECK(cfg.grow_up);
    CHECK(cfg.grow_left);
    CHECK(cfg.grow_right);
    CHECK(cfg.neighs.size() == 8);
}

TEST_CASE("parse_growth_config: growth_directions enumerates explicit dirs")
{
    json p;
    p["growth_directions"] = {"down", "right"};
    auto cfg = parse_growth_config(p, false, false);
    CHECK(cfg.has_growth_directions);
    CHECK(cfg.grow_down);
    CHECK(cfg.grow_right);
    CHECK_FALSE(cfg.grow_up);
    CHECK_FALSE(cfg.grow_left);
    CHECK(cfg.neighs.size() == 2);
}

TEST_CASE("parse_growth_config: growth_directions=[] (empty) falls back to all-8")
{
    json p;
    p["growth_directions"] = json::array();
    auto cfg = parse_growth_config(p, false, false);
    CHECK(cfg.has_growth_directions);
    CHECK(cfg.grow_down);
    CHECK(cfg.grow_up);
    CHECK(cfg.grow_left);
    CHECK(cfg.grow_right);
    CHECK(cfg.neighs.size() == 8);
}

TEST_CASE("parse_growth_config: flip_x swaps grow_left/right and mirrors neighs")
{
    json p; p["growth_directions"] = {"right"};
    auto a = parse_growth_config(p, false, false);
    auto b = parse_growth_config(p, false, true);
    // a.grow_right == true; b should swap so a's right becomes b's left
    CHECK(a.grow_right);
    CHECK_FALSE(a.grow_left);
    CHECK(b.grow_left);
    CHECK_FALSE(b.grow_right);
    // Neighbor x components inverted
    REQUIRE_FALSE(b.neighs.empty());
    CHECK(b.neighs[0][1] == -a.neighs[0][1]);
}

TEST_CASE("GrowthConfig::log emits expected substrings")
{
    json p;
    p["rollout_growth"] = true;
    auto cfg = parse_growth_config(p, false, false);
    std::ostringstream oss;
    cfg.log(oss, /*stop_gen=*/42);
    auto s = oss.str();
    CHECK(s.find("growth directions:") != std::string::npos);
    CHECK(s.find("steps=42") != std::string::npos);
    CHECK(s.find("rollout growth:") != std::string::npos);
}

TEST_CASE("GrowthConfig::log without rollout skips rollout block")
{
    json p = json::object();
    auto cfg = parse_growth_config(p, false, false);
    std::ostringstream oss;
    cfg.log(oss, 0);
    auto s = oss.str();
    CHECK(s.find("rollout growth:") == std::string::npos);
}

TEST_CASE("parse_growth_config: disable_grid_expansion follows fill_growth alias")
{
    json p; p["fill_growth"] = true;
    auto cfg = parse_growth_config(p, false, false);
    CHECK(cfg.disable_grid_expansion);
    // Explicit key wins over alias
    json p2; p2["fill_growth"] = true; p2["disable_grid_expansion"] = false;
    auto cfg2 = parse_growth_config(p2, false, false);
    CHECK_FALSE(cfg2.disable_grid_expansion);
}
