// Coverage gap-filler for PlaneSurface.cpp: exercises every branch in the
// anonymous-namespace vx/vy/basis helpers (lines ~26-83 in PlaneSurface.cpp)
// plus the free function min_loc (lines ~387-453).

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

cv::Mat_<cv::Vec3f> makePlanarGrid(int rows, int cols, float z = 0.0f)
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

// ------------------- min_loc free function -------------------

TEST_CASE("min_loc: invalid starting location returns -1 sentinel")
{
    auto m = makePlanarGrid(8, 8);
    m(2, 2) = cv::Vec3f(-1.f, -1.f, -1.f);
    cv::Vec2f loc(2.0f, 2.0f); // points to invalid cell (loc is [x,y])
    cv::Vec3f out;
    std::vector<cv::Vec3f> tgts = {cv::Vec3f(0, 0, 0)};
    std::vector<float> tds = {0.f};
    float r = min_loc(m, loc, out, tgts, tds, nullptr);
    CHECK(r == doctest::Approx(-1.0f));
    CHECK(out == cv::Vec3f(-1.f, -1.f, -1.f));
}

TEST_CASE("min_loc: zero-target on flat grid stays put")
{
    auto m = makePlanarGrid(16, 16);
    // Target is the start point itself; min_loc should stabilize there.
    cv::Vec2f loc(5.f, 5.f);
    cv::Vec3f start = cv::Vec3f(5.f, 5.f, 0.f);
    std::vector<cv::Vec3f> tgts = {start};
    std::vector<float> tds = {0.f};
    cv::Vec3f out;
    float best = min_loc(m, loc, out, tgts, tds, nullptr, /*init_step=*/2.0f, /*min_step=*/0.5f);
    CHECK(best >= 0.0f);
    CHECK(out[2] == doctest::Approx(0.0f));
}

TEST_CASE("min_loc: with plane constraint penalty")
{
    auto m = makePlanarGrid(16, 16, /*z=*/0.0f);
    PlaneSurface plane(cv::Vec3f(0, 0, 0), cv::Vec3f(0, 0, 1));
    cv::Vec2f loc(8.f, 8.f);
    cv::Vec3f out;
    std::vector<cv::Vec3f> tgts = {cv::Vec3f(8.f, 8.f, 0.f)};
    std::vector<float> tds = {0.f};
    float best = min_loc(m, loc, out, tgts, tds, &plane, 2.0f, 0.5f);
    CHECK(best >= 0.0f);
    CHECK(std::isfinite(best));
}

TEST_CASE("min_loc: target far away — search exits when step < min_step")
{
    auto m = makePlanarGrid(32, 32);
    cv::Vec2f loc(16.f, 16.f);
    cv::Vec3f out;
    // Target outside grid: search clamps and converges or hits min_step.
    std::vector<cv::Vec3f> tgts = {cv::Vec3f(1000.f, 1000.f, 0.f)};
    std::vector<float> tds = {0.f};
    float best = min_loc(m, loc, out, tgts, tds, nullptr, 8.0f, 0.25f);
    CHECK(best >= 0.0f);
}
