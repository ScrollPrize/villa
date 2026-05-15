#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <set>
#include <string>
#include <vector>

namespace {

cv::Mat_<cv::Vec3f> makePlanarGrid(int rows, int cols, float z = 50.f)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(static_cast<float>(c), static_cast<float>(r), z);
    return m;
}

cv::Mat_<cv::Vec3f> makeSparseGrid(int rows, int cols, int patchH, int patchW)
{
    cv::Mat_<cv::Vec3f> m(rows, cols, cv::Vec3f(-1.f, -1.f, -1.f));
    const int r0 = (rows - patchH) / 2;
    const int c0 = (cols - patchW) / 2;
    for (int r = r0; r < r0 + patchH; ++r)
        for (int c = c0; c < c0 + patchW; ++c)
            m(r, c) = cv::Vec3f(static_cast<float>(c), static_cast<float>(r), 50.f);
    return m;
}

} // namespace

TEST_CASE("Rect3D default-construction is zero box")
{
    Rect3D r;
    CHECK(r.low == cv::Vec3f(0, 0, 0));
    CHECK(r.high == cv::Vec3f(0, 0, 0));
}

TEST_CASE("expand_rect grows box to include a new point")
{
    Rect3D a;
    a.low = a.high = cv::Vec3f(1, 2, 3);
    auto r = expand_rect(a, cv::Vec3f(0, 5, 2));
    CHECK(r.low == cv::Vec3f(0, 2, 2));
    CHECK(r.high == cv::Vec3f(1, 5, 3));
}

TEST_CASE("expand_rect with point inside is identity")
{
    Rect3D a;
    a.low = cv::Vec3f(0, 0, 0);
    a.high = cv::Vec3f(10, 10, 10);
    auto r = expand_rect(a, cv::Vec3f(5, 5, 5));
    CHECK(r.low == a.low);
    CHECK(r.high == a.high);
}

TEST_CASE("intersect on overlapping boxes is true")
{
    Rect3D a, b;
    a.low = cv::Vec3f(0, 0, 0);
    a.high = cv::Vec3f(10, 10, 10);
    b.low = cv::Vec3f(5, 5, 5);
    b.high = cv::Vec3f(15, 15, 15);
    CHECK(intersect(a, b));
    CHECK(intersect(b, a));
}

TEST_CASE("intersect on touching boxes is true")
{
    Rect3D a, b;
    a.low = cv::Vec3f(0, 0, 0);
    a.high = cv::Vec3f(10, 10, 10);
    b.low = cv::Vec3f(10, 10, 10);
    b.high = cv::Vec3f(20, 20, 20);
    CHECK(intersect(a, b));
}

TEST_CASE("intersect on disjoint boxes is false")
{
    Rect3D a, b;
    a.low = cv::Vec3f(0, 0, 0);
    a.high = cv::Vec3f(1, 1, 1);
    b.low = cv::Vec3f(2, 0, 0);
    b.high = cv::Vec3f(3, 1, 1);
    CHECK_FALSE(intersect(a, b));
    CHECK_FALSE(intersect(b, a));
}

TEST_CASE("QuadSurface ctor from Mat retains scale and is loaded")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.5f, 2.0f));
    CHECK(qs.isLoaded());
    CHECK(qs.scale()[0] == doctest::Approx(1.5f));
    CHECK(qs.scale()[1] == doctest::Approx(2.0f));
    CHECK(qs.canUnload() == false); // no path
}

TEST_CASE("size() returns scaled grid size")
{
    auto pts = makePlanarGrid(8, 10);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    auto sz = qs.size();
    CHECK(sz.width > 0);
    CHECK(sz.height > 0);
}

TEST_CASE("rawPoints / rawPointsPtr round-trip the input")
{
    auto pts = makePlanarGrid(4, 5);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    auto* p = qs.rawPointsPtr();
    REQUIRE(p);
    CHECK(p->rows == 4);
    CHECK(p->cols == 5);
    CHECK((*p)(0, 0) == cv::Vec3f(0, 0, 50.f));
}

TEST_CASE("isPointValid / isQuadValid on a dense planar grid")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    CHECK(qs.isPointValid(0, 0));
    CHECK(qs.isPointValid(3, 3));
    CHECK(qs.isQuadValid(0, 0));
    CHECK(qs.isQuadValid(2, 2));
}

TEST_CASE("isPointValid / isQuadValid honor -1 sentinels")
{
    auto pts = makePlanarGrid(4, 4);
    pts(1, 1) = cv::Vec3f(-1.f, -1.f, -1.f);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    CHECK_FALSE(qs.isPointValid(1, 1));
    CHECK_FALSE(qs.isQuadValid(0, 0)); // quad (0,0)-(1,1) touches sentinel
    CHECK(qs.isPointValid(0, 0));
    CHECK(qs.isQuadValid(2, 2));
}

TEST_CASE("countValidPoints / countValidQuads on dense grid")
{
    auto pts = makePlanarGrid(3, 3);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    CHECK(qs.countValidPoints() == 9);
    CHECK(qs.countValidQuads() == 4); // (rows-1)*(cols-1)
}

TEST_CASE("countValidPoints / countValidQuads on sparse grid")
{
    auto pts = makeSparseGrid(20, 20, 5, 5);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    CHECK(qs.countValidPoints() == 25);
    CHECK(qs.countValidQuads() == 16); // 4x4 quads in a 5x5 patch
}

TEST_CASE("validPoints range yields only non-sentinel cells")
{
    auto pts = makeSparseGrid(10, 10, 3, 3);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    int count = 0;
    for (auto&& [r, c, p] : qs.validPoints()) {
        CHECK(p[0] != -1.f);
        ++count;
    }
    CHECK(count == 9);
}

TEST_CASE("validQuads range yields only fully-valid 2x2 cells")
{
    auto pts = makeSparseGrid(10, 10, 3, 3);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    int count = 0;
    for (auto&& [r, c, p00, p01, p10, p11] : qs.validQuads()) {
        CHECK(p00[0] != -1.f);
        CHECK(p01[0] != -1.f);
        CHECK(p10[0] != -1.f);
        CHECK(p11[0] != -1.f);
        ++count;
    }
    CHECK(count == 4); // 2x2 quads in a 3x3 patch
}

TEST_CASE("validMask matches sentinel layout")
{
    auto pts = makeSparseGrid(6, 6, 2, 2);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    auto mask = qs.validMask();
    REQUIRE(mask.rows == 6);
    REQUIRE(mask.cols == 6);
    int valid = 0;
    for (int r = 0; r < mask.rows; ++r)
        for (int c = 0; c < mask.cols; ++c)
            if (mask(r, c)) ++valid;
    CHECK(valid == 4);
}

TEST_CASE("bbox covers valid points and ignores sentinels")
{
    auto pts = makeSparseGrid(20, 20, 4, 4);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    auto bb = qs.bbox();
    // valid patch is centered: r0 = 8, c0 = 8, extent 4x4. World coords:
    // x in [c0, c0+3] = [8,11], y in [r0, r0+3] = [8,11], z = 50.
    CHECK(bb.low[2] == doctest::Approx(50.0f));
    CHECK(bb.high[2] == doctest::Approx(50.0f));
    CHECK(bb.low[0] <= 8.0f);
    CHECK(bb.high[0] >= 11.0f);
}

TEST_CASE("valid/coord/normal on an in-memory grid")
{
    auto pts = makePlanarGrid(10, 10);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Vec3f ptr(0, 0, 0);
    CHECK(qs.valid(ptr));
    auto c = qs.coord(ptr);
    CHECK(std::isfinite(c[0]));
    auto n = qs.normal(ptr);
    CHECK(std::isfinite(n[0]));
}

TEST_CASE("move adds offset to ptr")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Vec3f ptr(0, 0, 0);
    qs.move(ptr, cv::Vec3f(1, 2, 0));
    CHECK(ptr[0] == doctest::Approx(1.0f));
    CHECK(ptr[1] == doctest::Approx(2.0f));
}

TEST_CASE("flipU reverses rows (vertical flip)")
{
    // makePlanarGrid: m(r,c) = (c, r, z). After flipU (rows reversed):
    // new (0, c) == old (rows-1, c) == (c, rows-1, z).
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    qs.flipU();
    const auto at00 = (*qs.rawPointsPtr())(0, 0);
    CHECK(at00[0] == doctest::Approx(0.0f));
    CHECK(at00[1] == doctest::Approx(3.0f));
}

TEST_CASE("flipV reverses columns (horizontal flip)")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    qs.flipV();
    const auto at00 = (*qs.rawPointsPtr())(0, 0);
    CHECK(at00[0] == doctest::Approx(3.0f));
    CHECK(at00[1] == doctest::Approx(0.0f));
}

TEST_CASE("rotate runs without crashing on dense grid")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    qs.rotate(15.0f);
    CHECK(qs.rawPointsPtr() != nullptr);
}

TEST_CASE("overlappingIds get/set/add/remove")
{
    auto pts = makePlanarGrid(2, 2);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    CHECK(qs.overlappingIds().empty());
    qs.setOverlappingIds({"a", "b"});
    CHECK(qs.overlappingIds().size() == 2);
    qs.addOverlappingId("c");
    CHECK(qs.overlappingIds().count("c") == 1);
    qs.removeOverlappingId("a");
    CHECK(qs.overlappingIds().count("a") == 0);
}

TEST_CASE("dpi getter / setter")
{
    auto pts = makePlanarGrid(2, 2);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    qs.setDpi(300.0f);
    CHECK(qs.dpi() == doctest::Approx(300.0f));
}

TEST_CASE("setChannel + channel + channelNames")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    cv::Mat ch(4, 4, CV_8UC1, cv::Scalar(123));
    qs.setChannel("test", ch);
    auto names = qs.channelNames();
    CHECK(!names.empty());
    bool found = false;
    for (const auto& n : names) if (n == "test") found = true;
    CHECK(found);
    cv::Mat got = qs.channel("test", SURF_CHANNEL_NORESIZE);
    CHECK(got.rows == 4);
    CHECK(got.cols == 4);
}

TEST_CASE("invalidateCache / invalidateMask are no-throw")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    qs.invalidateCache();
    qs.invalidateMask();
    CHECK(true);
}

TEST_CASE("canUnload is false without a path")
{
    auto pts = makePlanarGrid(2, 2);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    CHECK_FALSE(qs.canUnload());
}
