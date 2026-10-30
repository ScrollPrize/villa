// Coverage for growth_strategies/CandidateOrdering.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/growth_strategies/CandidateOrdering.hpp"

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
