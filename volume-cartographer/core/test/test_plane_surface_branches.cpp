// Coverage gap-filler for PlaneSurface.cpp: exercises every branch in the
// anonymous-namespace vx/vy/basis helpers in PlaneSurface.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/PlaneSurface.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <vector>

namespace {

bool finiteVec(const cv::Vec3f& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

cv::Mat_<cv::Vec3f> makePlanarGrid(int rows, int cols, float z = 1.0f)
{
    cv::Mat_<cv::Vec3f> m(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = cv::Vec3f(static_cast<float>(c), static_cast<float>(r), z);
    return m;
}

} // namespace

// ------------------- vx_from_orig_norm via setNormal branches -------------------

TEST_CASE("PlaneSurface with normal=(1,0,0): n[1]==0 && n[2]==0 branch")
{
    PlaneSurface p(cv::Vec3f(0, 0, 0), cv::Vec3f(1, 0, 0));
    // The plane normal becomes ±(1,0,0); basis must still be unit-length
    // and orthogonal to the normal even though the vx helper returned zero.
    auto vx = p.basisX();
    auto vy = p.basisY();
    CHECK(finiteVec(vx));
    CHECK(finiteVec(vy));
    CHECK(std::abs(cv::norm(vx) - 1.0) < 1e-3);
    CHECK(std::abs(cv::norm(vy) - 1.0) < 1e-3);
    auto n = p.normal(cv::Vec3f(0, 0, 0));
    CHECK(std::abs(vx.dot(n)) < 1e-3f);
    CHECK(std::abs(vy.dot(n)) < 1e-3f);
}

TEST_CASE("PlaneSurface with normal=(1,1,0): n[2]==0 branch")
{
    PlaneSurface p(cv::Vec3f(0, 0, 0), cv::Vec3f(1, 1, 0));
    auto vx = p.basisX();
    auto vy = p.basisY();
    auto n = p.normal(cv::Vec3f(0, 0, 0));
    CHECK(finiteVec(vx));
    CHECK(finiteVec(vy));
    CHECK(std::abs(vx.dot(n)) < 1e-3f);
    CHECK(std::abs(vy.dot(n)) < 1e-3f);
}

TEST_CASE("PlaneSurface with normal=(1,0,1): n[1]==0 branch")
{
    PlaneSurface p(cv::Vec3f(0, 0, 0), cv::Vec3f(1, 0, 1));
    auto vx = p.basisX();
    auto vy = p.basisY();
    auto n = p.normal(cv::Vec3f(0, 0, 0));
    CHECK(finiteVec(vx));
    CHECK(finiteVec(vy));
    CHECK(std::abs(vx.dot(n)) < 1e-3f);
    CHECK(std::abs(vy.dot(n)) < 1e-3f);
}

TEST_CASE("PlaneSurface with normal=(1,1,1): general path")
{
    PlaneSurface p(cv::Vec3f(0, 0, 0), cv::Vec3f(1, 1, 1));
    auto vx = p.basisX();
    auto vy = p.basisY();
    auto n = p.normal(cv::Vec3f(0, 0, 0));
    CHECK(finiteVec(vx));
    CHECK(finiteVec(vy));
    CHECK(std::abs(vx.dot(n)) < 1e-3f);
    CHECK(std::abs(vy.dot(n)) < 1e-3f);
}

TEST_CASE("PlaneSurface with normal=(0,1,0): vy degenerate branch")
{
    // Triggers vy_from_orig_norm's swapped-input n[1]==0 && n[2]==0 case.
    PlaneSurface p(cv::Vec3f(0, 0, 0), cv::Vec3f(0, 1, 0));
    auto vx = p.basisX();
    auto vy = p.basisY();
    auto n = p.normal(cv::Vec3f(0, 0, 0));
    CHECK(finiteVec(vx));
    CHECK(finiteVec(vy));
    CHECK(std::abs(vx.dot(n)) < 1e-3f);
    CHECK(std::abs(vy.dot(n)) < 1e-3f);
}
