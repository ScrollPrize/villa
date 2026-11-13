// Coverage for inpaintSurfaceHoles (InpaintSurface.hpp) as vc_tifxyz2obj
// --inpaint calls it: default unit and iterations. The surface is a tilted
// plane, so the right answer for every hole cell is known exactly: the plane
// point of that cell. Holes wider than two cells used to come back with cells
// stacked on their neighbours (the one pass mean fill starts a cell on top of
// its only known neighbour, and the losses have no gradient at coincidence).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/InpaintSurface.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <functional>

namespace {

constexpr int kN = 60;

cv::Vec3d planePoint(int r, int c, double step)
{
    // Orthonormal in-plane axes of the plane with normal (1, 2, 3).
    const cv::Vec3d n = cv::normalize(cv::Vec3d(1, 2, 3));
    const cv::Vec3d u = cv::normalize(n.cross(cv::Vec3d(0, 0, 1)));
    const cv::Vec3d v = n.cross(u);
    return cv::Vec3d(3000, 3000, 3000) + (r * u + c * v) * step;
}

struct Result {
    int filled = 0;
    int holeCells = 0;
    double maxErr = 0.0;
    double minSpacing = 1e300;
};

Result fillPlaneHole(double step, const std::function<bool(int, int)>& inHole)
{
    cv::Mat_<cv::Vec3f> pts(kN, kN);
    Result res;
    for (int r = 0; r < kN; ++r)
        for (int c = 0; c < kN; ++c) {
            if (inHole(r, c)) { pts(r, c) = cv::Vec3f(-1, -1, -1); ++res.holeCells; }
            else pts(r, c) = cv::Vec3f(planePoint(r, c, step));
        }
    res.filled = vc::core::util::inpaintSurfaceHoles(pts);
    for (int r = 0; r < kN; ++r)
        for (int c = 0; c < kN; ++c) {
            if (!inHole(r, c)) continue;
            const cv::Vec3d p(pts(r, c));
            res.maxErr = std::max(res.maxErr, cv::norm(p - planePoint(r, c, step)));
            if (c + 1 < kN) res.minSpacing = std::min(res.minSpacing, cv::norm(p - cv::Vec3d(pts(r, c + 1))));
            if (r + 1 < kN) res.minSpacing = std::min(res.minSpacing, cv::norm(p - cv::Vec3d(pts(r + 1, c))));
        }
    return res;
}

bool square8(int r, int c) { return r >= 26 && r < 34 && c >= 26 && c < 34; }
bool discR10(int r, int c) { return (r - 30) * (r - 30) + (c - 30) * (c - 30) <= 100; }
bool square3(int r, int c) { return std::abs(r - 30) <= 1 && std::abs(c - 30) <= 1; }

void checkOnPlane(double step, const std::function<bool(int, int)>& hole)
{
    const Result res = fillPlaneHole(step, hole);
    MESSAGE("step " << step << " hole cells " << res.holeCells << " max error " << res.maxErr
         << " min spacing " << res.minSpacing);
    CHECK(res.filled == res.holeCells);
    // Every hole cell back at its own plane point, within 0.05 voxel.
    CHECK(res.maxErr <= 0.05);
    // No hole cell sits on its neighbour.
    CHECK(res.minSpacing >= 0.5 * step);
}

} // namespace

TEST_CASE("inpaintSurfaceHoles: 3x3 hole in a tilted plane")
{
    checkOnPlane(1.0, square3);
    checkOnPlane(4.0, square3);
}

TEST_CASE("inpaintSurfaceHoles: 8x8 hole in a tilted plane, step 1")
{
    checkOnPlane(1.0, square8);
}

TEST_CASE("inpaintSurfaceHoles: 8x8 hole in a tilted plane, step 4")
{
    checkOnPlane(4.0, square8);
}

TEST_CASE("inpaintSurfaceHoles: disc hole of radius 10 in a tilted plane, step 1")
{
    checkOnPlane(1.0, discR10);
}

TEST_CASE("inpaintSurfaceHoles: disc hole of radius 10 in a tilted plane, step 4")
{
    checkOnPlane(4.0, discR10);
}
