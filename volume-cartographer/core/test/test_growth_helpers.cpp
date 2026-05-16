// Coverage for growth_strategies/CandidateOrdering.cpp and ComponentPruning.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/growth_strategies/CandidateOrdering.hpp"
#include "../src/growth_strategies/ComponentPruning.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace {

GrowthConfig defaultConfig()
{
    GrowthConfig c;
    c.candidate_priority_existing_depth = false;
    return c;
}

} // namespace

// -------- CandidateOrdering --------

TEST_CASE("order_growth_candidates: empty input returns empty")
{
    auto cfg = defaultConfig();
    auto out = order_growth_candidates({}, cfg,
        [](const cv::Vec2i&) { return 0; },
        [](const cv::Vec2i&) { return 0; });
    CHECK(out.candidates.empty());
    CHECK(out.points.empty());
}

TEST_CASE("order_growth_candidates: sorts by valid_neighbor_count desc")
{
    auto cfg = defaultConfig();
    std::vector<cv::Vec2i> in = {{0, 0}, {1, 0}, {2, 0}};
    auto out = order_growth_candidates(in, cfg,
        // Map: (0,0)→1, (1,0)→3, (2,0)→2
        [](const cv::Vec2i& p) {
            if (p == cv::Vec2i{0, 0}) return 1;
            if (p == cv::Vec2i{1, 0}) return 3;
            return 2;
        },
        [](const cv::Vec2i&) { return 0; });
    REQUIRE(out.points.size() == 3);
    CHECK(out.points[0] == cv::Vec2i{1, 0}); // count=3
    CHECK(out.points[1] == cv::Vec2i{2, 0}); // count=2
    CHECK(out.points[2] == cv::Vec2i{0, 0}); // count=1
}

TEST_CASE("order_growth_candidates: tie broken by (x,y) ascending")
{
    auto cfg = defaultConfig();
    std::vector<cv::Vec2i> in = {{5, 5}, {1, 1}, {3, 2}};
    auto out = order_growth_candidates(in, cfg,
        [](const cv::Vec2i&) { return 1; }, // all tied
        [](const cv::Vec2i&) { return 0; });
    REQUIRE(out.points.size() == 3);
    CHECK(out.points[0] == cv::Vec2i{1, 1});
    CHECK(out.points[1] == cv::Vec2i{3, 2});
    CHECK(out.points[2] == cv::Vec2i{5, 5});
}

TEST_CASE("order_growth_candidates: existing_depth priority promotes deeper support")
{
    auto cfg = defaultConfig();
    cfg.candidate_priority_existing_depth = true;
    std::vector<cv::Vec2i> in = {{0, 0}, {1, 0}};
    auto out = order_growth_candidates(in, cfg,
        // Equal neighbor counts so the depth tiebreaker matters.
        [](const cv::Vec2i&) { return 5; },
        [](const cv::Vec2i& p) { return p == cv::Vec2i{1, 0} ? 9 : 1; });
    REQUIRE(out.points.size() == 2);
    CHECK(out.points[0] == cv::Vec2i{1, 0}); // depth=9 wins
}

// -------- ComponentPruning --------

TEST_CASE("find_largest_component_prune_points: empty active rect returns empty")
{
    auto r = find_largest_component_prune_points(cv::Rect(0, 0, 0, 0),
                                                 cv::Vec2i{0, 0},
                                                 [](const cv::Vec2i&) { return true; },
                                                 [](const cv::Vec2i&) { return false; });
    CHECK(r.points_to_prune.empty());
    CHECK(r.kept_component_size == 0);
    CHECK(r.component_count == 0);
}

TEST_CASE("find_largest_component_prune_points: single connected component, no pruning")
{
    cv::Rect active(0, 0, 5, 5);
    auto valid = [](const cv::Vec2i& p) { return p[0] >= 0 && p[1] >= 0 && p[0] < 5 && p[1] < 5; };
    auto preserve = [](const cv::Vec2i&) { return false; };
    auto r = find_largest_component_prune_points(active, cv::Vec2i{0, 0}, valid, preserve);
    CHECK(r.component_count == 1);
    CHECK(r.points_to_prune.empty());
}

TEST_CASE("find_largest_component_prune_points: two disjoint components — smaller pruned")
{
    cv::Rect active(0, 0, 10, 4);
    // Block A: rows 0..3, cols 0..2  (12 points)
    // Block B: rows 0..1, cols 5..6  ( 4 points)  — distance >=2, no 8-neigh contact
    auto valid = [](const cv::Vec2i& p) {
        // p is (y, x) per the impl
        int y = p[0], x = p[1];
        if (y < 0 || x < 0) return false;
        bool inA = (y >= 0 && y < 4 && x >= 0 && x <= 2);
        bool inB = (y >= 0 && y < 2 && x >= 5 && x <= 6);
        return inA || inB;
    };
    auto preserve = [](const cv::Vec2i&) { return false; };
    auto r = find_largest_component_prune_points(active, cv::Vec2i{0, 0}, valid, preserve);
    CHECK(r.component_count == 2);
    CHECK(r.kept_component_size == 12);
    CHECK(r.points_to_prune.size() == 4);
}

TEST_CASE("find_largest_component_prune_points: preserve_point keeps a point in pruned component")
{
    cv::Rect active(0, 0, 10, 4);
    auto valid = [](const cv::Vec2i& p) {
        int y = p[0], x = p[1];
        if (y < 0 || x < 0) return false;
        bool inA = (y >= 0 && y < 4 && x >= 0 && x <= 2);
        bool inB = (y >= 0 && y < 2 && x >= 5 && x <= 6);
        return inA || inB;
    };
    // Preserve all points in component B.
    auto preserve = [](const cv::Vec2i& p) {
        return p[1] >= 5; // x>=5 → in B
    };
    auto r = find_largest_component_prune_points(active, cv::Vec2i{0, 0}, valid, preserve);
    CHECK(r.component_count == 2);
    CHECK(r.points_to_prune.empty());
}

TEST_CASE("find_largest_component_prune_points: equal-size components prefer seed-containing")
{
    cv::Rect active(0, 0, 10, 4);
    // Two 4-point blocks
    auto valid = [](const cv::Vec2i& p) {
        int y = p[0], x = p[1];
        bool inA = (y >= 0 && y < 2 && x >= 0 && x <= 1);
        bool inB = (y >= 0 && y < 2 && x >= 5 && x <= 6);
        return inA || inB;
    };
    auto preserve = [](const cv::Vec2i&) { return false; };
    // Seed in block B
    auto r = find_largest_component_prune_points(active, cv::Vec2i{0, 5}, valid, preserve);
    CHECK(r.component_count == 2);
    CHECK(r.kept_component_size == 4);
    // A (4 points) pruned because the seed is in B; preserved=0 means all of A pruned.
    CHECK(r.points_to_prune.size() == 4);
}
