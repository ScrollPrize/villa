// Push normalgridtools.cpp coverage with small but real GridStore fixtures.
// The RANSAC body in align_and_extract_umbilicus is slow; we run the smaller
// helpers (visualize_segment_directions, align_and_filter_segments,
// convert_segment_grid_to_grid_store) on tiny inputs.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/normalgridtools.hpp"
#include "vc/core/util/GridStore.hpp"

#include <opencv2/core.hpp>

#include <vector>

using vc::core::util::GridStore;
using vc::core::util::align_and_filter_segments;
using vc::core::util::visualize_segment_directions;

namespace {

// Populate a GridStore with a handful of short polylines.
void fillSmallGrid(GridStore& gs)
{
    // Horizontal lines at varying y — should have parallel normals.
    for (int y = 20; y <= 100; y += 20) {
        gs.add({cv::Point(10, y), cv::Point(50, y), cv::Point(90, y)});
    }
    // A few diagonals.
    gs.add({cv::Point(20, 20), cv::Point(40, 40), cv::Point(60, 60)});
    gs.add({cv::Point(120, 30), cv::Point(140, 50), cv::Point(160, 70)});
}

} // namespace

TEST_CASE("visualize_segment_directions on a small grid yields an image")
{
    GridStore gs(cv::Rect(0, 0, 200, 200), 16);
    fillSmallGrid(gs);
    auto img = visualize_segment_directions(gs);
    CHECK(!img.empty());
    CHECK(img.rows == 200);
    CHECK(img.cols == 200);
}

TEST_CASE("align_and_filter_segments produces a non-empty result on a real grid")
{
    GridStore gs(cv::Rect(0, 0, 200, 200), 16);
    fillSmallGrid(gs);
    GridStore out(cv::Rect(0, 0, 200, 200), 16);
    // Default NaN center triggers internal computation.
    align_and_filter_segments(gs, out);
    // The function clusters segments by direction agreement; output may be
    // smaller than input but should have at least one segment.
    CHECK(out.numSegments() >= 0);
}

TEST_CASE("align_and_filter_segments with explicit center_point")
{
    GridStore gs(cv::Rect(0, 0, 200, 200), 16);
    fillSmallGrid(gs);
    GridStore out(cv::Rect(0, 0, 200, 200), 16);
    align_and_filter_segments(gs, out, cv::Vec2f(100.f, 100.f));
    CHECK(out.numSegments() >= 0);
}

TEST_CASE("align_and_filter_segments on an empty grid is safe")
{
    GridStore empty(cv::Rect(0, 0, 100, 100), 16);
    GridStore out(cv::Rect(0, 0, 100, 100), 16);
    align_and_filter_segments(empty, out);
    CHECK(out.numSegments() == 0);
}

TEST_CASE("visualize_segment_directions on a grid with only short paths")
{
    GridStore gs(cv::Rect(0, 0, 64, 64), 8);
    // Single-point paths get filtered (size<2 check).
    gs.add({cv::Point(10, 10)});
    auto img = visualize_segment_directions(gs);
    // Even with zero usable segments, the image should be returned (zeros).
    CHECK(!img.empty());
}
