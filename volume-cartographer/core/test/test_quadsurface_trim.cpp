// Tests for QuadSurface::trimToValidBbox aggressive-trim guard.
//
// Scenario being guarded against: on-disk x/y/z.tif somehow have most
// cells set to (-1,-1,-1) outside a small valid patch (corruption on
// save, torn write, or upstream preview leaving the grid sparse). The
// pre-guard trim would crop the entire surface down to that patch and
// the next saveOverwrite would discard the rest of the mesh forever.
// The guard refuses the crop in that case so the on-disk x/y/z.tif
// stays intact.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

namespace {

// Build a points grid where everything is invalid except a centered
// patch of size patchH x patchW.
cv::Mat_<cv::Vec3f> makeSparseGrid(int rows, int cols, int patchH, int patchW)
{
    cv::Mat_<cv::Vec3f> pts(rows, cols, cv::Vec3f(-1.f, -1.f, -1.f));
    const int r0 = (rows - patchH) / 2;
    const int c0 = (cols - patchW) / 2;
    for (int r = r0; r < r0 + patchH; ++r) {
        for (int c = c0; c < c0 + patchW; ++c) {
            pts(r, c) = cv::Vec3f(static_cast<float>(c),
                                  static_cast<float>(r),
                                  10.f);
        }
    }
    return pts;
}

}  // namespace

TEST_CASE("trimToValidBbox: refuses to crop large surface to tiny patch")
{
    // 100x100 surface (10000 cells > 2500 size floor) with a 5x5 patch.
    // keepFraction = 25 / 10000 = 0.25% — well under default 40%. Guard
    // must refuse and leave _points untouched.
    cv::Mat_<cv::Vec3f> pts = makeSparseGrid(100, 100, 5, 5);
    QuadSurface surf(pts, cv::Vec2f(1.0f, 1.0f));

    const cv::Size before = surf.rawPointsPtr()->size();
    REQUIRE(before == cv::Size(100, 100));

    bool trimmed = surf.trimToValidBbox();

    CHECK_FALSE(trimmed);
    CHECK(surf.rawPointsPtr()->size() == before);
}

TEST_CASE("trimToValidBbox: allows reasonable 64% trim")
{
    // 1000x1000 surface; 800x800 valid patch in the middle.
    // keepFraction = 640000 / 1000000 = 64% — above 40%. Guard allows.
    cv::Mat_<cv::Vec3f> pts = makeSparseGrid(1000, 1000, 800, 800);
    QuadSurface surf(pts, cv::Vec2f(1.0f, 1.0f));

    bool trimmed = surf.trimToValidBbox();

    CHECK(trimmed);
    CHECK(surf.rawPointsPtr()->size() == cv::Size(800, 800));
}

TEST_CASE("trimToValidBbox: refuses 5% kept on large surface")
{
    // 1000x1000 surface; 220x220 patch -> keepFraction ~4.84% < 40%.
    cv::Mat_<cv::Vec3f> pts = makeSparseGrid(1000, 1000, 220, 220);
    QuadSurface surf(pts, cv::Vec2f(1.0f, 1.0f));

    const cv::Size before = surf.rawPointsPtr()->size();
    bool trimmed = surf.trimToValidBbox();

    CHECK_FALSE(trimmed);
    CHECK(surf.rawPointsPtr()->size() == before);
}

TEST_CASE("trimToValidBbox: tiny surfaces under size floor are exempt")
{
    // 40x40 surface (1600 cells <= 2500 floor) with a 5x5 patch. Even
    // though keepFraction = 25/1600 = 1.6%, the grid is below the floor
    // and aggressive trimming is normal here. Guard allows.
    cv::Mat_<cv::Vec3f> pts = makeSparseGrid(40, 40, 5, 5);
    QuadSurface surf(pts, cv::Vec2f(1.0f, 1.0f));

    bool trimmed = surf.trimToValidBbox();

    CHECK(trimmed);
    CHECK(surf.rawPointsPtr()->size() == cv::Size(5, 5));
}

TEST_CASE("trimToValidBbox: no-op when bbox matches grid")
{
    // Fully valid grid — bbH==rows and bbW==cols, so the existing early
    // check returns false (not the guard).
    cv::Mat_<cv::Vec3f> pts = makeSparseGrid(64, 64, 64, 64);
    QuadSurface surf(pts, cv::Vec2f(1.0f, 1.0f));

    bool trimmed = surf.trimToValidBbox();

    CHECK_FALSE(trimmed);
    CHECK(surf.rawPointsPtr()->size() == cv::Size(64, 64));
}

TEST_CASE("trimToValidBbox: empty surface returns false safely")
{
    QuadSurface surf;

    bool trimmed = surf.trimToValidBbox();

    CHECK_FALSE(trimmed);
}
