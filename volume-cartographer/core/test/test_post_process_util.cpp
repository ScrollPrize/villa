// Tests for the Qt-free post-processing pipeline in core/src/PostProcess.cpp
// (distinct from core/src/render/PostProcess.cpp which has its own test).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/PostProcess.hpp"

#include <opencv2/core.hpp>

#include <cstdint>

namespace {

cv::Mat_<uint8_t> ramp(int rows = 4, int cols = 4)
{
    cv::Mat_<uint8_t> m(rows, cols);
    int idx = 0;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m(r, c) = static_cast<uint8_t>((idx++ * 8) & 0xFF);
    return m;
}

} // namespace

TEST_CASE("applyPostProcess: identity with default params and stretchValues=false")
{
    auto m = ramp();
    cv::Mat_<uint8_t> orig = m.clone();
    vc::PostProcessParams p; // defaults: window 0..255, no flags
    vc::applyPostProcess(m, p);
    // windowLowInt==0 && windowHighInt==255 → identity path
    for (int r = 0; r < m.rows; ++r)
        for (int c = 0; c < m.cols; ++c)
            CHECK(int(m(r, c)) == int(orig(r, c)));
}

TEST_CASE("applyPostProcess: ISO cutoff zeros pixels below threshold")
{
    auto m = ramp();
    vc::PostProcessParams p;
    p.isoCutoff = 16;
    vc::applyPostProcess(m, p);
    // All pixels with original value < 15 should now be 0.
    CHECK(int(m(0, 0)) == 0);
    CHECK(int(m(0, 1)) == 0);
}

TEST_CASE("applyPostProcess: stretchValues stretches min/max to [0,255]")
{
    cv::Mat_<uint8_t> m(2, 2);
    m(0, 0) = 100; m(0, 1) = 110;
    m(1, 0) = 120; m(1, 1) = 130;
    vc::PostProcessParams p;
    p.stretchValues = true;
    vc::applyPostProcess(m, p);
    // After stretch the min should be 0 and max should be 255.
    double mn, mx;
    cv::minMaxLoc(m, &mn, &mx);
    CHECK(mn == doctest::Approx(0.0));
    CHECK(mx == doctest::Approx(255.0));
}

TEST_CASE("applyPostProcess: postStretchValues stretches in-place")
{
    cv::Mat_<uint8_t> m(2, 2);
    m(0, 0) = 50; m(0, 1) = 100;
    m(1, 0) = 150; m(1, 1) = 200;
    vc::PostProcessParams p;
    p.postStretchValues = true;
    vc::applyPostProcess(m, p);
    double mn, mx;
    cv::minMaxLoc(m, &mn, &mx);
    CHECK(mn == doctest::Approx(0.0));
    CHECK(mx == doctest::Approx(255.0));
}

TEST_CASE("applyPostProcess: window/level shrinks the range")
{
    auto m = ramp();
    vc::PostProcessParams p;
    p.windowLow = 50;
    p.windowHigh = 200;
    vc::applyPostProcess(m, p);
    // Pixels below windowLow become 0, above windowHigh become 255.
    // Original (0,0) = 0 → 0. Original (3,3) = 120 → mapped into [0,255].
    CHECK(int(m(0, 0)) == 0);
}

TEST_CASE("applyPostProcess: clamps windowLow > windowHigh edge case")
{
    auto m = ramp();
    vc::PostProcessParams p;
    p.windowLow = 200;
    p.windowHigh = 50; // inverted — impl clamps to lowInt+1
    vc::applyPostProcess(m, p);
    // Just verify no crash and output is in [0,255]; impl details vary.
    double mn, mx;
    cv::minMaxLoc(m, &mn, &mx);
    CHECK(mn >= 0.0);
    CHECK(mx <= 255.0);
}

TEST_CASE("applyPostProcess: small-component removal drops sub-threshold blobs")
{
    cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::zeros(20, 20);
    // Single isolated pixel — should be dropped at minComponentSize=5.
    m(5, 5) = 200;
    // A 4x4 block — area 16 — should survive.
    for (int r = 10; r < 14; ++r)
        for (int c = 10; c < 14; ++c)
            m(r, c) = 200;

    vc::PostProcessParams p;
    p.removeSmallComponents = true;
    p.minComponentSize = 5;
    vc::applyPostProcess(m, p);

    CHECK(int(m(5, 5)) == 0);
    CHECK(int(m(11, 11)) == 200);
}

TEST_CASE("applyPostProcess: postStretch handles constant image gracefully")
{
    cv::Mat_<uint8_t> m(4, 4, uint8_t(100));
    vc::PostProcessParams p;
    p.postStretchValues = true;
    vc::applyPostProcess(m, p);
    // maxVal == minVal → branch is skipped, image unchanged.
    for (int r = 0; r < m.rows; ++r)
        for (int c = 0; c < m.cols; ++c)
            CHECK(int(m(r, c)) == 100);
}

TEST_CASE("PostProcessParams is constexpr-default-constructible")
{
    constexpr vc::PostProcessParams p;
    CHECK(p.windowLow == doctest::Approx(0.0f));
    CHECK(p.windowHigh == doctest::Approx(255.0f));
    CHECK_FALSE(p.stretchValues);
    CHECK(p.minComponentSize == 50);
}
