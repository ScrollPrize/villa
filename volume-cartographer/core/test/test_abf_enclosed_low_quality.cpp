// ABF++ flattening: the triangle quality filter must not punch holes inside
// the mesh. A cluster of long-edge triangles fully enclosed by good ones
// (here a 2x2 block of vertices pushed 150 voxels off a curved strip) used to
// be dropped, leaving its vertices without UVs and an interior hole whose
// angles ABF++ does not close.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"
#include "vc/flattening/ABFFlattening.hpp"

#include <opencv2/core.hpp>

#include <cmath>

namespace {

constexpr int kRows = 40;
constexpr int kCols = 120;
constexpr float kStep = 20.f;     // voxels between grid vertices
constexpr float kRadius = 1200.f; // cylinder radius, voxels

// Part of a cylinder (a developable, curved strip) sampled every kStep voxels.
cv::Mat_<cv::Vec3f> curvedStrip()
{
    cv::Mat_<cv::Vec3f> p(kRows, kCols);
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
            const float t = c * kStep / kRadius;
            p(r, c) = cv::Vec3f(3000.f + kRadius * std::cos(t), 3000.f + kRadius * std::sin(t), 5000.f + r * kStep);
        }
    }
    return p;
}

// Push a 2x2 block of vertices 150 voxels along the outward normal: every
// triangle touching the block gets an edge longer than the filter's maximum.
void addSpike(cv::Mat_<cv::Vec3f>& p, int r0, int c0)
{
    for (int r = r0; r < r0 + 2; ++r) {
        for (int c = c0; c < c0 + 2; ++c) {
            const float t = c * kStep / kRadius;
            p(r, c) += 150.f * cv::Vec3f(std::cos(t), std::sin(t), 0.f);
        }
    }
}

int verticesWithoutUV(const cv::Mat_<cv::Vec3f>& points, const cv::Mat_<cv::Vec2f>& uv)
{
    int missing = 0;
    for (int r = 0; r < points.rows; ++r) {
        for (int c = 0; c < points.cols; ++c) {
            const cv::Vec2f& t = uv(r, c);
            const bool flattened = std::isfinite(t[0]) && std::isfinite(t[1]) && !(t[0] == -1.f && t[1] == -1.f);
            if (points(r, c)[0] != -1.f && !flattened) {
                ++missing;
            }
        }
    }
    return missing;
}

} // namespace

TEST_CASE("abfFlatten: a clean curved strip is flattened completely")
{
    cv::Mat_<cv::Vec3f> points = curvedStrip();
    QuadSurface surface(points, cv::Vec2f(1.f / kStep, 1.f / kStep));

    const cv::Mat_<cv::Vec2f> uv = vc::abfFlatten(surface);
    REQUIRE_FALSE(uv.empty());
    REQUIRE(uv.rows == kRows);
    REQUIRE(uv.cols == kCols);
    CHECK(verticesWithoutUV(points, uv) == 0);
}

TEST_CASE("abfFlatten: low-quality triangles enclosed by the mesh are kept, not cut out")
{
    cv::Mat_<cv::Vec3f> points = curvedStrip();
    addSpike(points, kRows / 2, kCols / 2);
    QuadSurface surface(points, cv::Vec2f(1.f / kStep, 1.f / kStep));

    const cv::Mat_<cv::Vec2f> uv = vc::abfFlatten(surface);
    REQUIRE_FALSE(uv.empty());
    REQUIRE(uv.rows == kRows);
    REQUIRE(uv.cols == kCols);
    // Before the fix the four spike vertices lost all their triangles and came
    // back without UVs (an interior hole in the parameterization).
    CHECK(verticesWithoutUV(points, uv) == 0);
}
