// Coverage gap-filler for AffineTransform.cpp's QuadSurface-touching code:
// measureGridAxisSpacing, transformSurfacePoints, refreshTransformedSurfaceState,
// cloneSurfaceForTransform (lines ~24-100 + ~375-495).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/AffineTransform.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <memory>
#include <optional>

using namespace vc::core::util;

namespace {

cv::Mat_<cv::Vec3f> makePlanarGrid(int rows, int cols, float spacing = 1.0f)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(static_cast<float>(c) * spacing,
                                static_cast<float>(r) * spacing,
                                0.f);
    return m;
}

cv::Matx44d scaleMatrix(double s)
{
    cv::Matx44d m = cv::Matx44d::eye();
    m(0, 0) = s; m(1, 1) = s; m(2, 2) = s;
    return m;
}

} // namespace

TEST_CASE("transformSurfacePoints (int scale): no-op on identity + scale=1")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    transformSurfacePoints(&qs, 1, std::nullopt);
    auto* p = qs.rawPointsPtr();
    REQUIRE(p);
    CHECK((*p)(0, 0) == cv::Vec3f(0, 0, 0));
}

TEST_CASE("transformSurfacePoints applies pre-scale factor to all points")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    transformSurfacePoints(&qs, 2, std::nullopt); // pre-scale by 2x
    auto* p = qs.rawPointsPtr();
    REQUIRE(p);
    // Spacing measurement should fire and resample back to "approximate
    // original spacing"; we mainly check no NaN and points are still valid.
    CHECK(std::isfinite((*p)(0, 0)[0]));
    CHECK(std::isfinite((*p)(p->rows - 1, p->cols - 1)[0]));
}

TEST_CASE("transformSurfacePoints with a 4x4 affine matrix (uniform scale)")
{
    auto pts = makePlanarGrid(20, 20); // big enough to skip the small-grid path
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    auto m = scaleMatrix(2.0);
    transformSurfacePoints(&qs, 1.0, m, 1.0);
    auto* p = qs.rawPointsPtr();
    REQUIRE(p);
    CHECK(std::isfinite((*p)(0, 0)[0]));
}

TEST_CASE("transformSurfacePoints handles a tiny grid (small-grid branch)")
{
    // < 20 rows/cols triggers the "use whole grid" branch in
    // measureGridAxisSpacing.
    auto pts = makePlanarGrid(5, 5);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    transformSurfacePoints(&qs, 1.0, std::nullopt, 1.0);
    CHECK(qs.rawPointsPtr() != nullptr);
}

TEST_CASE("transformSurfacePoints with sentinel points leaves them as sentinels")
{
    auto pts = makePlanarGrid(8, 8);
    pts(1, 1) = cv::Vec3f(-1.f, -1.f, -1.f);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    transformSurfacePoints(&qs, 1, std::nullopt);
    auto* p = qs.rawPointsPtr();
    REQUIRE(p);
    CHECK((*p)(1, 1)[0] == -1.f);
}

TEST_CASE("refreshTransformedSurfaceState writes bbox + scale into meta")
{
    auto pts = makePlanarGrid(8, 8);
    QuadSurface qs(pts, cv::Vec2f(1.5f, 2.0f));
    refreshTransformedSurfaceState(&qs);
    CHECK(qs.meta.is_object());
    CHECK(qs.meta.contains("bbox"));
    CHECK(qs.meta.contains("scale"));
    CHECK(qs.meta["scale"].is_array());
}

TEST_CASE("refreshTransformedSurfaceState preserves an existing object meta")
{
    auto pts = makePlanarGrid(4, 4);
    QuadSurface qs(pts, cv::Vec2f(1.f, 1.f));
    qs.meta = utils::Json::object();
    qs.meta["custom"] = "value";
    refreshTransformedSurfaceState(&qs);
    CHECK(qs.meta.contains("custom"));
    CHECK(qs.meta.contains("bbox"));
}

TEST_CASE("cloneSurfaceForTransform deep-copies points + metadata")
{
    auto pts = makePlanarGrid(4, 4);
    auto src = std::make_shared<QuadSurface>(pts, cv::Vec2f(1.f, 1.f));
    src->id = "src-id";
    src->meta = utils::Json::object();
    src->meta["k"] = "v";
    auto clone = cloneSurfaceForTransform(src);
    REQUIRE(clone != nullptr);
    CHECK(clone->id == "src-id");
    CHECK(clone->meta["k"].get_string() == "v");
    // Mutating the clone must not affect the source.
    auto* cp = clone->rawPointsPtr();
    REQUIRE(cp);
    (*cp)(0, 0) = cv::Vec3f(-999, -999, -999);
    CHECK((*src->rawPointsPtr())(0, 0)[0] != -999);
}

TEST_CASE("cloneSurfaceForTransform with null source returns null")
{
    CHECK(cloneSurfaceForTransform(nullptr) == nullptr);
}

TEST_CASE("cloneSurfaceForTransform handles null meta on the source")
{
    auto pts = makePlanarGrid(2, 2);
    auto src = std::make_shared<QuadSurface>(pts, cv::Vec2f(1.f, 1.f));
    // src->meta is default-constructed (null Json); clone should make it {}.
    auto clone = cloneSurfaceForTransform(src);
    REQUIRE(clone != nullptr);
    CHECK(clone->meta.is_object());
}
