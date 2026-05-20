// Coverage for core/src/Thinning.cpp — customThinning entrypoints and the
// pruneThinningTraces helper. Inputs are tiny synthetic binary masks.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Thinning.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace {

cv::Mat horizontalBar(int rows = 16, int cols = 32, int barRow = 8, int barWidth = 3)
{
    cv::Mat m = cv::Mat::zeros(rows, cols, CV_8UC1);
    for (int r = barRow; r < barRow + barWidth; ++r)
        for (int c = 2; c < cols - 2; ++c)
            m.at<uint8_t>(r, c) = 255;
    return m;
}

cv::Mat allZeros(int rows = 8, int cols = 8)
{
    return cv::Mat::zeros(rows, cols, CV_8UC1);
}

} // namespace

TEST_CASE("customThinning on all-zero input produces empty output")
{
    auto input = allZeros();
    cv::Mat out;
    std::vector<std::vector<cv::Point>> traces;
    customThinning(input, out, &traces);
    CHECK(out.size() == input.size());
    CHECK(out.type() == CV_8UC1);
    CHECK(traces.empty());
}

TEST_CASE("customThinning on a horizontal bar produces at least one trace")
{
    auto input = horizontalBar();
    cv::Mat out;
    std::vector<std::vector<cv::Point>> traces;
    customThinning(input, out, &traces);
    CHECK_FALSE(traces.empty());
    // Output image must be same size as input and have some non-zero pixels.
    CHECK(out.size() == input.size());
    CHECK(cv::countNonZero(out) > 0);
}

TEST_CASE("customThinning with nullptr trace vector still produces output image")
{
    auto input = horizontalBar();
    cv::Mat out;
    customThinning(input, out, nullptr);
    CHECK(cv::countNonZero(out) > 0);
}

TEST_CASE("customThinning with stats: counters move on a real input")
{
    auto input = horizontalBar();
    cv::Mat out;
    std::vector<std::vector<cv::Point>> traces;
    ThinningStats stats;
    customThinning(input, out, &traces, &stats);
    CHECK(stats.seedCount >= 1);
    CHECK(stats.distanceTransformSeconds >= 0.0);
    CHECK(stats.tracePathsSeconds >= 0.0);
}

TEST_CASE("customThinningTraceOnly: traces are populated, no image output")
{
    auto input = horizontalBar();
    std::vector<std::vector<cv::Point>> traces;
    customThinningTraceOnly(input, traces);
    CHECK_FALSE(traces.empty());
}

TEST_CASE("customThinningTraceOnly with external scratch reuses buffers")
{
    auto input = horizontalBar();
    ThinningScratch scratch;
    std::vector<std::vector<cv::Point>> traces;
    ThinningStats stats;
    customThinningTraceOnly(input, traces, &stats, scratch);
    customThinningTraceOnly(input, traces, &stats, scratch);
    CHECK_FALSE(traces.empty());
    // Stats accumulated across both calls.
    CHECK(stats.seedCount >= 2);
}

TEST_CASE("customThinning with pruning drops short traces")
{
    auto input = horizontalBar();
    cv::Mat out;
    std::vector<std::vector<cv::Point>> traces;
    ThinningStats stats;
    ThinningPruneParams prune;
    prune.minLengthPx = 1000.0; // huge — should prune everything
    customThinning(input, out, &traces, &stats, prune);
    CHECK(traces.empty());
    CHECK(stats.tracesPruned >= 1);
}

TEST_CASE("pruneThinningTraces standalone: drops <2-point traces")
{
    std::vector<std::vector<cv::Point>> traces = {
        {{0, 0}},                 // 1 point — drop
        {{0, 0}, {10, 0}},        // length 10 — keep at minLengthPx=5
        {{0, 0}, {1, 0}},         // length 1 — drop at 5
    };
    ThinningStats stats;
    ThinningPruneParams p;
    p.minLengthPx = 5.0;
    pruneThinningTraces(traces, p, &stats);
    REQUIRE(traces.size() == 1);
    CHECK(traces[0].size() == 2);
    CHECK(traces[0][1] == cv::Point(10, 0));
    CHECK(stats.tracesPruned == 2);
    CHECK(stats.tracesKept == 1);
}

TEST_CASE("pruneThinningTraces with minLengthPx=0 keeps everything except <2-pt")
{
    std::vector<std::vector<cv::Point>> traces = {
        {{0, 0}, {1, 1}},
        {{0, 0}},
    };
    ThinningPruneParams p; // minLengthPx default 0
    p.minLengthPx = 0.0;
    pruneThinningTraces(traces, p, nullptr);
    // <2-pt drop is unconditional.
    REQUIRE(traces.size() == 1);
}

TEST_CASE("ThinningStats::accumulate sums all fields")
{
    ThinningStats a;
    a.seedCount = 3;
    a.traceSteps = 100;
    a.tracesKept = 2;
    ThinningStats b;
    b.seedCount = 4;
    b.traceSteps = 50;
    b.tracesPruned = 1;
    a.accumulate(b);
    CHECK(a.seedCount == 7);
    CHECK(a.traceSteps == 150);
    CHECK(a.tracesKept == 2);
    CHECK(a.tracesPruned == 1);
}
