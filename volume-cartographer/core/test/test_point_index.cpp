#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <vc_object_store/geometry/VertexIndex.hpp>

#include <opencv2/core.hpp>

#include <cmath>
#include <utility>

namespace {
// A 1x2 grid at (0, 0, 5) and (10, 0, 5); ids are row * cols + col.
vc::geometry::VertexIndex twoPoints()
{
    cv::Mat_<cv::Vec3f> m(1, 2);
    m(0, 0) = cv::Vec3f(0, 0, 5);
    m(0, 1) = cv::Vec3f(10, 0, 5);
    vc::geometry::VertexIndex idx;
    idx.rebuild(m);
    return idx;
}
}

TEST_CASE("default-constructed index is empty")
{
    vc::geometry::VertexIndex idx;
    CHECK(idx.empty());
    CHECK_FALSE(idx.nearest(cv::Vec3f(0, 0, 0)).has_value());
}

TEST_CASE("buildFromMat indexes valid points and ignores -1 sentinels")
{
    cv::Mat_<cv::Vec3f> m(2, 3, cv::Vec3f(-1.f, -1.f, -1.f));
    m(0, 0) = cv::Vec3f(1, 1, 1);
    m(1, 2) = cv::Vec3f(2, 2, 2);
    vc::geometry::VertexIndex idx;
    idx.rebuild(m);
    CHECK_FALSE(idx.empty());
    auto r = idx.nearest(cv::Vec3f(2, 2, 2.4f));
    REQUIRE(r.has_value());
    CHECK(r->index == 5);
    CHECK(r->position == cv::Vec3f(2, 2, 2));
    CHECK(r->distanceSq == doctest::Approx(0.16f));
}

TEST_CASE("buildFromMat replaces the previous contents")
{
    vc::geometry::VertexIndex idx = twoPoints();
    idx.rebuild(cv::Mat_<cv::Vec3f>());
    CHECK(idx.empty());
}

TEST_CASE("nearest with non-finite query returns nullopt")
{
    const vc::geometry::VertexIndex idx = twoPoints();
    CHECK_FALSE(idx.nearest(cv::Vec3f(std::nanf(""), 0, 5)).has_value());
}

TEST_CASE("nearest with maxDistance filter")
{
    const vc::geometry::VertexIndex idx = twoPoints();
    CHECK_FALSE(idx.nearest(cv::Vec3f(100, 0, 5), 50.0f).has_value());
    auto r = idx.nearest(cv::Vec3f(100, 0, 5), 200.0f);
    REQUIRE(r.has_value());
    CHECK(r->index == 1);
}

TEST_CASE("move construction / assignment transfers ownership")
{
    vc::geometry::VertexIndex a = twoPoints();
    vc::geometry::VertexIndex b(std::move(a));
    CHECK_FALSE(b.empty());
    vc::geometry::VertexIndex c;
    c = std::move(b);
    REQUIRE(c.nearest(cv::Vec3f(9, 0, 5)).has_value());
    CHECK(c.nearest(cv::Vec3f(9, 0, 5))->index == 1);
}
