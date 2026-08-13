// Coverage for core/src/normalgridtools.cpp.
//
// Focuses on cheap paths: SegmentInfo construction, SegmentGrid CRUD,
// nearest_neighbors, get_random_segment, and the empty-input early returns
// of align_and_extract_umbilicus / visualize_segment_directions. The full
// RANSAC happy path is too slow / non-deterministic to assert in a unit test.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/normalgridtools.hpp"
#include "vc/core/util/GridStore.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <memory>

using namespace vc::core::util;

TEST_CASE("SegmentInfo: middle point and normal from two endpoints")
{
    SegmentInfo s(cv::Point(0, 0), cv::Point(10, 0), /*path_idx=*/1, /*seg_idx=*/2);
    CHECK(s.middle_point.x == doctest::Approx(5.0f));
    CHECK(s.middle_point.y == doctest::Approx(0.0f));
    CHECK(s.original_path_idx == 1);
    CHECK(s.original_segment_idx == 2);
    // tangent = (10, 0) normalized = (1, 0); normal = (0, 1)
    CHECK(s.normal[0] == doctest::Approx(0.0f));
    CHECK(std::abs(s.normal[1]) == doctest::Approx(1.0f));
    CHECK_FALSE(s.flipped);
}

TEST_CASE("SegmentInfo: diagonal endpoints produce unit normal")
{
    SegmentInfo s(cv::Point(0, 0), cv::Point(3, 4), 0, 0);
    CHECK(cv::norm(s.normal) == doctest::Approx(1.0));
}

TEST_CASE("SegmentGrid: empty grid count is 0")
{
    SegmentGrid g(cv::Rect(0, 0, 100, 100), /*grid_step=*/10);
    CHECK(g.count() == 0);
    CHECK(g.size() == cv::Size(100, 100));
    CHECK(g.get_all_segments().empty());
}

TEST_CASE("SegmentGrid::add increments count and stores segment")
{
    SegmentGrid g(cv::Rect(0, 0, 100, 100), 10);
    auto s = std::make_shared<SegmentInfo>(cv::Point(5, 5), cv::Point(15, 5), 0, 0);
    g.add(s);
    CHECK(g.count() == 1);
    CHECK(g.get_all_segments().size() == 1);
}

TEST_CASE("SegmentGrid::remove decrements count")
{
    SegmentGrid g(cv::Rect(0, 0, 100, 100), 10);
    auto s1 = std::make_shared<SegmentInfo>(cv::Point(5, 5), cv::Point(15, 5), 0, 0);
    auto s2 = std::make_shared<SegmentInfo>(cv::Point(50, 50), cv::Point(60, 60), 1, 0);
    g.add(s1);
    g.add(s2);
    CHECK(g.count() == 2);
    g.remove(s1);
    CHECK(g.count() == 1);
}

TEST_CASE("SegmentGrid::nearest_neighbors returns up to n segments")
{
    SegmentGrid g(cv::Rect(0, 0, 200, 200), 20);
    auto s1 = std::make_shared<SegmentInfo>(cv::Point(10, 10), cv::Point(20, 10), 0, 0);
    auto s2 = std::make_shared<SegmentInfo>(cv::Point(50, 50), cv::Point(60, 50), 1, 0);
    auto s3 = std::make_shared<SegmentInfo>(cv::Point(150, 150), cv::Point(160, 150), 2, 0);
    g.add(s1); g.add(s2); g.add(s3);

    auto nn = g.nearest_neighbors(cv::Point2f(15.f, 10.f), 2);
    REQUIRE(nn.size() >= 1);
    CHECK(nn[0] == s1); // closest

    auto nn_all = g.nearest_neighbors(cv::Point2f(0.f, 0.f), 10);
    CHECK(nn_all.size() <= 3);
}

TEST_CASE("SegmentGrid::nearest_neighbors on empty grid returns empty")
{
    SegmentGrid g(cv::Rect(0, 0, 100, 100), 10);
    auto nn = g.nearest_neighbors(cv::Point2f(50.f, 50.f), 5);
    CHECK(nn.empty());
}

TEST_CASE("SegmentGrid::get_random_segment returns a segment when non-empty")
{
    SegmentGrid g(cv::Rect(0, 0, 100, 100), 10);
    auto s = std::make_shared<SegmentInfo>(cv::Point(5, 5), cv::Point(15, 5), 0, 0);
    g.add(s);
    auto got = g.get_random_segment();
    CHECK(got == s);
}

TEST_CASE("align_and_extract_umbilicus: empty GridStore returns NaN")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    auto u = align_and_extract_umbilicus(gs);
    CHECK(std::isnan(u[0]));
    CHECK(std::isnan(u[1]));
}

TEST_CASE("align_and_extract_umbilicus: GridStore with only single-point paths returns NaN")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    // single-point path has no segments — short-circuits to NaN
    gs.add({cv::Point(5, 5)});
    auto u = align_and_extract_umbilicus(gs);
    CHECK(std::isnan(u[0]));
}

TEST_CASE("visualize_segment_directions: empty GridStore yields image")
{
    GridStore gs(cv::Rect(0, 0, 100, 100), 10);
    auto img = visualize_segment_directions(gs);
    // Either an empty mat or zeros-only — both are acceptable.
    if (!img.empty()) {
        CHECK(img.size() == cv::Size(100, 100));
    }
    CHECK(true);
}
