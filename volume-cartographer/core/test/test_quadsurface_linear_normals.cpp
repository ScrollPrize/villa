// LinearBlendedNormals, cylinder R 40: an offset of k planes errs by |k| times the normal error, so k = 11 covers |k| <= 11.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "vc/core/util/QuadSurface.hpp"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>

TEST_CASE("blended normals keep a plane offset on a cylinder's analytic normal")
{
    cv::Mat_<cv::Vec3f> pts(81, 81), p, n;
    for (int r = 0; r < 81; ++r)
        for (int c = 0; c < 81; ++c) {
            const double th = M_PI / 2 - 1.0 + c / 40.0;
            pts(r, c) = cv::Vec3f(float(64 + 40 * std::cos(th)), float(24 + r), float(64 + 40 * std::sin(th)));
        }
    QuadSurface s(pts, cv::Vec2f(1.f, 1.f));
    s.setGenInterpolation(GenInterpolation::LinearBlendedNormals);
    REQUIRE(s.center()[0] - 32.f == doctest::Approx(8.5));  // pixel i samples vertex i + 8.5, the mid-cell
    s.gen(&p, &n, cv::Size(64, 64), cv::Vec3f(0, 0, 0), 1.0f, cv::Vec3f(-32.f, -32.f, 0.f));
    double worst = 0.0;
    for (int j = 0; j < 64; ++j)
        for (int i = 0; i < 64; ++i) {
            const cv::Vec3d q(p(j, i)), nt = cv::normalize(cv::Vec3d(64.0 - q[0], 0.0, 64.0 - q[2]));  // inward
            worst = std::max(worst, 11.0 * cv::norm(cv::normalize(cv::Vec3d(n(j, i))) - nt));
        }
    CHECK_MESSAGE(worst < 1e-3, "11 planes off the surface: worst position error " << worst << " voxels");
}
