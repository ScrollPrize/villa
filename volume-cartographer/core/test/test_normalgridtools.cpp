// Coverage for core/src/normalgridtools.cpp.
//
// Focuses on cheap paths: SegmentInfo construction, SegmentGrid CRUD,
// nearest_neighbors, get_random_segment, and the empty-input early returns
// of align_and_extract_umbilicus / visualize_segment_directions. The full
// estimate is exercised at the end on synthetic sections whose centre is
// known by construction, with the generator seed fixed so that it repeats.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/normalgridtools.hpp"
#include "vc/core/util/GridStore.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

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

// ---------------------------------------------------------------------------------------------
// The estimate on synthetic sections with a centre known by construction.
//
// The sections are the ones the density table of the validation paper is built on: a square of
// 8000 by 8000 grid units, concentric circles of radius 300 to 3500 in steps of 80, consecutive
// points about 12 units apart, centre (4000, 4000). One grid unit is one voxel at 9.362 um. A
// fraction below 1 keeps that fraction of every circle (0.5 is a half circle, open on one side);
// a thinning ratio above 1 keeps one segment in that many of those with x below the centre and
// every segment above it, so the dense side is +x.
//
// What these tests guard, and what they do not. On these sections the published objective, the
// weighted mean, is exact (0.4 units), and the weighted sum used since the correction is biased
// towards the side that holds more segments: 46 units (0.43 mm) on the uniform circle, 259 units
// (2.42 mm) on the half circle and 244 units (2.29 mm) at a 1:10 thinning, medians over 20 seeds
// with p90 of 65, 341 and 289 units. So a synthetic section cannot tell the sum from the mean;
// what it can do is bound the sum's known bias so that it cannot grow unnoticed, and check that
// the walk settles inside the grid. The tolerances are 1.5 times those p90 values, rounded up,
// so that no seed fails them by statistics alone. The defect the correction repairs shows only on
// real predictions, where the mean has no interior maximum and the walk leaves the volume.

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kUnitMm = 0.009362;   // one grid unit, in millimetres
constexpr int kSide = 8000;
constexpr double kCentreX = 4000.0, kCentreY = 4000.0;

void addSyntheticSection(GridStore& gs, double frac, int thin_ratio, std::uint32_t thin_seed)
{
    std::mt19937 rng(thin_seed);
    std::uniform_int_distribution<int> keep(1, std::max(1, thin_ratio));
    for (double r = 300.0; r <= 3500.0; r += 80.0) {
        const int n = std::max(8, (int)(2.0 * kPi * r / 12.0));
        const int m = std::max(3, (int)(n * frac));
        std::vector<cv::Point> pts;
        pts.reserve(m);
        for (int i = 0; i < m; ++i) {
            // a full circle stops one step short of 2 pi, a partial one reaches its end point
            const double th = frac >= 1.0 ? 2.0 * kPi * i / m : 2.0 * kPi * frac * i / (m - 1);
            pts.emplace_back(cvRound(kCentreX + r * std::cos(th)), cvRound(kCentreY + r * std::sin(th)));
        }
        for (int i = 0; i + 1 < m; ++i) {
            const double mid_x = 0.5 * (pts[i].x + pts[i + 1].x);
            if (thin_ratio > 1 && mid_x < kCentreX && keep(rng) != 1) continue;
            gs.add({pts[i], pts[i + 1]});
        }
    }
}

double errorMm(const cv::Vec2f& u)
{
    return std::hypot(u[0] - kCentreX, u[1] - kCentreY) * kUnitMm;
}

bool insideGrid(const cv::Vec2f& u)
{
    return u[0] >= 0.0f && u[0] <= kSide && u[1] >= 0.0f && u[1] <= kSide;
}

} // namespace

TEST_CASE("align_and_extract_umbilicus: a seed makes the estimate repeatable")
{
    GridStore gs(cv::Rect(0, 0, kSide, kSide), 64);
    addSyntheticSection(gs, 1.0, 1, 0);
    auto a = align_and_extract_umbilicus(gs, 7u);
    auto b = align_and_extract_umbilicus(gs, 7u);
    CHECK(a[0] == b[0]);
    CHECK(a[1] == b[1]);
}

TEST_CASE("align_and_extract_umbilicus: concentric circles, within 1.0 mm of the known centre")
{
    GridStore gs(cv::Rect(0, 0, kSide, kSide), 64);
    addSyntheticSection(gs, 1.0, 1, 0);
    auto u = align_and_extract_umbilicus(gs, 1u);
    REQUIRE(insideGrid(u));
    CHECK(errorMm(u) < 1.0);   // 20 seed median 0.43 mm, p90 0.61 mm
}

TEST_CASE("align_and_extract_umbilicus: half circle, the bias towards the open side stays under 5 mm")
{
    GridStore gs(cv::Rect(0, 0, kSide, kSide), 64);
    addSyntheticSection(gs, 0.5, 1, 0);
    auto u = align_and_extract_umbilicus(gs, 1u);
    REQUIRE(insideGrid(u));
    CHECK(errorMm(u) < 5.0);   // 20 seed median 2.42 mm, p90 3.20 mm
}

TEST_CASE("align_and_extract_umbilicus: circles thinned 1:10 on one side, the bias stays under 4.5 mm")
{
    GridStore gs(cv::Rect(0, 0, kSide, kSide), 64);
    addSyntheticSection(gs, 1.0, 10, 3u);
    auto u = align_and_extract_umbilicus(gs, 1u);
    REQUIRE(insideGrid(u));
    CHECK(errorMm(u) < 4.5);   // 20 seed median 2.29 mm, p90 2.71 mm
    CHECK(u[0] > kCentreX);    // and it leans towards the dense side, +x
}
