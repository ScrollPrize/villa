// Coverage for core/src/ABFFlattening.cpp — abfFlatten on surfaces whose
// correct parameterization is known analytically.
//
// A cylinder is developable (zero Gaussian curvature), so it admits an exactly
// isometric flattening: every edge length is preserved, up to one global scale.
// That gives a ground truth that follows from the geometry rather than from a
// recorded output, so the bound below does not need updating when the solver
// changes.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/flattening/ABFFlattening.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <vector>

namespace {

// Patch of a cylinder of radius R: columns sweep the angle, rows the axis.
cv::Mat_<cv::Vec3f> cylinderPatch(int rows, int cols, double radius,
                                  double halfAngle, double height)
{
    cv::Mat_<cv::Vec3f> pts(rows, cols);
    for (int r = 0; r < rows; ++r) {
        const double z = height * (double(r) / (rows - 1));
        for (int c = 0; c < cols; ++c) {
            const double a = -halfAngle + 2.0 * halfAngle * (double(c) / (cols - 1));
            pts(r, c) = cv::Vec3f(float(500.0 + radius * std::cos(a)),
                                  float(500.0 + radius * std::sin(a)),
                                  float(100.0 + z));
        }
    }
    return pts;
}

double dist3(const cv::Vec3f& a, const cv::Vec3f& b)
{
    return std::sqrt(double(a[0] - b[0]) * (a[0] - b[0]) +
                     double(a[1] - b[1]) * (a[1] - b[1]) +
                     double(a[2] - b[2]) * (a[2] - b[2]));
}

} // namespace

TEST_CASE("abfFlatten: a developable patch flattens isometrically")
{
    const int kRows = 40, kCols = 40;
    const double kRadius = 100.0, kHalfAngle = 0.6, kHeight = 120.0;

    auto pts = cylinderPatch(kRows, kCols, kRadius, kHalfAngle, kHeight);
    QuadSurface surf(pts, cv::Vec2f(1.0f, 1.0f));

    vc::ABFConfig cfg;
    cfg.maxIterations = 20;
    auto uv = vc::abfFlatten(surf, cfg);

    REQUIRE_FALSE(uv.empty());
    REQUIRE(uv.rows == kRows);
    REQUIRE(uv.cols == kCols);

    // Collect the ratio of flattened edge length to 3D edge length. On a
    // developable surface every ratio must equal the same global scale.
    std::vector<double> ratios;
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
            for (int k = 0; k < 2; ++k) {
                const int r2 = r + (k == 0 ? 0 : 1);
                const int c2 = c + (k == 0 ? 1 : 0);
                if (r2 >= kRows || c2 >= kCols) continue;
                const double d3 = dist3(pts(r, c), pts(r2, c2));
                const cv::Vec2f& u0 = uv(r, c);
                const cv::Vec2f& u1 = uv(r2, c2);
                if (!std::isfinite(u0[0]) || !std::isfinite(u1[0])) continue;
                const double duv = std::hypot(double(u1[0] - u0[0]),
                                              double(u1[1] - u0[1]));
                if (d3 > 1e-9 && duv > 1e-9) ratios.push_back(duv / d3);
            }
        }
    }

    REQUIRE(ratios.size() > 1000);

    double mean = 0.0;
    for (double v : ratios) mean += v;
    mean /= double(ratios.size());
    REQUIRE(mean > 0.0);

    double worst = 0.0;
    for (double v : ratios) worst = std::max(worst, std::abs(v / mean - 1.0));

    // An isometric map has a single scale factor everywhere. Measured worst
    // deviation here is ~2e-6, so 1e-3 keeps roughly three orders of magnitude
    // of headroom for solver and platform variation while still failing on any
    // meaningful loss of isometry on a developable input.
    CHECK(worst < 1e-3);
}
