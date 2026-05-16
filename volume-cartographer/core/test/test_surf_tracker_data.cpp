// Coverage for core/src/SurfTrackerData.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "../src/SurfTrackerData.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <stdexcept>

namespace {

cv::Mat_<cv::Vec3f> makePlanarGrid(int rows, int cols, float z = 0.f)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(static_cast<float>(c), static_cast<float>(r), z);
    return m;
}

} // namespace

// ----- resId_t -----

TEST_CASE("resId_t: default constructor")
{
    resId_t id;
    CHECK(id._type == 0);
    CHECK(id._sm == nullptr);
}

TEST_CASE("resId_t: constructor from (type, sm, p)")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1, 1));
    resId_t id(2, &qs, cv::Vec2i(3, 4));
    CHECK(id._type == 2);
    CHECK(id._sm == &qs);
    CHECK(id._p == cv::Vec2i(3, 4));
}

TEST_CASE("resId_t: pair-constructor picks smaller cv::Vec2i lexicographically")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1, 1));
    SUBCASE("same row: picks smaller col") {
        resId_t id(1, &qs, cv::Vec2i(5, 7), cv::Vec2i(5, 2));
        CHECK(id._p == cv::Vec2i(5, 2));
    }
    SUBCASE("different rows: picks smaller row") {
        resId_t id(1, &qs, cv::Vec2i(3, 9), cv::Vec2i(5, 1));
        CHECK(id._p == cv::Vec2i(3, 9));
    }
    SUBCASE("a < b, same row, in-order args") {
        resId_t id(1, &qs, cv::Vec2i(5, 2), cv::Vec2i(5, 7));
        CHECK(id._p == cv::Vec2i(5, 2));
    }
}

TEST_CASE("resId_t::operator== compares all three fields")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs1(pts, cv::Vec2f(1, 1));
    QuadSurface qs2(pts, cv::Vec2f(1, 1));
    resId_t a(1, &qs1, cv::Vec2i(0, 0));
    resId_t b(1, &qs1, cv::Vec2i(0, 0));
    resId_t c(2, &qs1, cv::Vec2i(0, 0));   // different type
    resId_t d(1, &qs2, cv::Vec2i(0, 0));   // different sm
    resId_t e(1, &qs1, cv::Vec2i(1, 0));   // different point
    CHECK(a == b);
    CHECK_FALSE(a == c);
    CHECK_FALSE(a == d);
    CHECK_FALSE(a == e);
}

TEST_CASE("resId_hash + SurfPoint_hash produce deterministic non-zero hashes")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1, 1));
    resId_t id(3, &qs, cv::Vec2i(5, 6));
    auto h1 = resId_hash{}(id);
    auto h2 = resId_hash{}(id);
    CHECK(h1 == h2);

    SurfPoint sp(&qs, cv::Vec2i(1, 2));
    auto sh1 = SurfPoint_hash{}(sp);
    auto sh2 = SurfPoint_hash{}(sp);
    CHECK(sh1 == sh2);
}

// ----- SurfTrackerData -----

TEST_CASE("loc inserts on first access; has reflects state")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1, 1));
    SurfTrackerData d;
    CHECK_FALSE(d.has(&qs, cv::Vec2i(0, 0)));
    auto& l = d.loc(&qs, cv::Vec2i(0, 0)); // default-inserts
    l = cv::Vec2d(2.0, 3.0);
    CHECK(d.has(&qs, cv::Vec2i(0, 0)));
    CHECK(d.loc(&qs, cv::Vec2i(0, 0))[0] == doctest::Approx(2.0));
}

TEST_CASE("erase removes a key")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1, 1));
    SurfTrackerData d;
    d.loc(&qs, cv::Vec2i(0, 0)) = cv::Vec2d(1, 1);
    d.erase(&qs, cv::Vec2i(0, 0));
    CHECK_FALSE(d.has(&qs, cv::Vec2i(0, 0)));
}

TEST_CASE("surfs / eraseSurf / surfsC behaviors")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs1(pts, cv::Vec2f(1, 1));
    QuadSurface qs2(pts, cv::Vec2f(1, 1));
    SurfTrackerData d;

    // surfsC returns an empty static reference for missing locs
    const auto& empty = d.surfsC(cv::Vec2i(5, 5));
    CHECK(empty.empty());

    auto& set00 = d.surfs(cv::Vec2i(0, 0));
    set00.insert(&qs1);
    set00.insert(&qs2);
    CHECK(d.surfsC(cv::Vec2i(0, 0)).size() == 2);

    d.eraseSurf(&qs1, cv::Vec2i(0, 0));
    CHECK(d.surfsC(cv::Vec2i(0, 0)).size() == 1);
}

TEST_CASE("hasResId / resId default-insert behavior")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1, 1));
    SurfTrackerData d;
    resId_t id(1, &qs, cv::Vec2i(0, 0));
    CHECK_FALSE(d.hasResId(id));
    (void)d.resId(id); // default-inserts a value
    CHECK(d.hasResId(id));
}

TEST_CASE("lookup_int throws when key missing")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1, 1));
    SurfTrackerData d;
    CHECK_THROWS_AS(d.lookup_int(&qs, cv::Vec2i(0, 0)), std::runtime_error);
}

TEST_CASE("lookup_int returns sentinel when loc[0]==-1")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1, 1));
    SurfTrackerData d;
    d.loc(&qs, cv::Vec2i(0, 0)) = cv::Vec2d(-1, -1);
    auto v = d.lookup_int(&qs, cv::Vec2i(0, 0));
    CHECK(v == cv::Vec3d(-1, -1, -1));
}

TEST_CASE("lookup_int returns -1 sentinel for out-of-bounds loc")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1, 1));
    SurfTrackerData d;
    d.loc(&qs, cv::Vec2i(0, 0)) = cv::Vec2d(1000.0, 1000.0); // outside rawPoints
    auto v = d.lookup_int(&qs, cv::Vec2i(0, 0));
    CHECK(v == cv::Vec3d(-1, -1, -1));
}

TEST_CASE("lookup_int returns interpolated value for in-bounds loc")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1, 1));
    SurfTrackerData d;
    // loc is (y, x) in the impl — point at row=2, col=3 → grid value (3, 2, 0).
    d.loc(&qs, cv::Vec2i(0, 0)) = cv::Vec2d(2.0, 3.0);
    auto v = d.lookup_int(&qs, cv::Vec2i(0, 0));
    CHECK(v[0] == doctest::Approx(3.0));
    CHECK(v[1] == doctest::Approx(2.0));
}

TEST_CASE("valid_int: missing key → false; sentinel → false; OOB → false; valid → true")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1, 1));
    SurfTrackerData d;
    CHECK_FALSE(d.valid_int(&qs, cv::Vec2i(0, 0)));
    d.loc(&qs, cv::Vec2i(0, 0)) = cv::Vec2d(-1, -1);
    CHECK_FALSE(d.valid_int(&qs, cv::Vec2i(0, 0)));
    d.loc(&qs, cv::Vec2i(0, 0)) = cv::Vec2d(1000, 1000);
    CHECK_FALSE(d.valid_int(&qs, cv::Vec2i(0, 0)));
    d.loc(&qs, cv::Vec2i(0, 0)) = cv::Vec2d(2.0, 3.0);
    CHECK(d.valid_int(&qs, cv::Vec2i(0, 0)));
}

TEST_CASE("valid_int: sentinel within the 2x2 neighborhood -> false")
{
    auto pts = makePlanarGrid(8, 8);
    pts(2, 3) = cv::Vec3f(-1.f, -1.f, -1.f);
    QuadSurface qs(pts, cv::Vec2f(1, 1));
    SurfTrackerData d;
    d.loc(&qs, cv::Vec2i(0, 0)) = cv::Vec2d(2.0, 3.0);
    CHECK_FALSE(d.valid_int(&qs, cv::Vec2i(0, 0)));
}

TEST_CASE("lookup_int_loc static: sentinel and OOB short-circuits")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1, 1));
    CHECK(SurfTrackerData::lookup_int_loc(&qs, cv::Vec2f(-1, 0)) == cv::Vec3d(-1, -1, -1));
    CHECK(SurfTrackerData::lookup_int_loc(&qs, cv::Vec2f(1e6, 1e6)) == cv::Vec3d(-1, -1, -1));
    auto v = SurfTrackerData::lookup_int_loc(&qs, cv::Vec2f(2, 3));
    CHECK(v[0] != -1);
}

TEST_CASE("flip_x mirrors x coords around x0")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1, 1));
    SurfTrackerData d;
    d.loc(&qs, cv::Vec2i(0, 2)) = cv::Vec2d(1.0, 2.0);
    d.surfs(cv::Vec2i(0, 2)).insert(&qs);
    d.flip_x(/*x0=*/5);
    // Before: x=2 → after: 5+5-2 = 8
    CHECK(d.has(&qs, cv::Vec2i(0, 8)));
    CHECK(d.surfsC(cv::Vec2i(0, 8)).count(&qs) == 1);
}

TEST_CASE("translate(0,0) is a no-op; non-zero shifts keys")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1, 1));
    SurfTrackerData d;
    d.loc(&qs, cv::Vec2i(0, 0)) = cv::Vec2d(1, 1);
    d.surfs(cv::Vec2i(0, 0)).insert(&qs);
    d.seed_loc = cv::Vec2i(7, 8);

    d.translate(cv::Vec2i(0, 0));
    CHECK(d.has(&qs, cv::Vec2i(0, 0)));
    CHECK(d.seed_loc == cv::Vec2i(7, 8));

    d.translate(cv::Vec2i(3, -2));
    CHECK_FALSE(d.has(&qs, cv::Vec2i(0, 0)));
    CHECK(d.has(&qs, cv::Vec2i(3, -2)));
    CHECK(d.surfsC(cv::Vec2i(3, -2)).count(&qs) == 1);
    CHECK(d.seed_loc == cv::Vec2i(10, 6));
}
