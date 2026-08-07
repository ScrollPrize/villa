#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "LineAnnotationFiberSegments.hpp"
#include "vc/fiber_tracer/FiberJson.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <string>
#include <vector>

using vc3d::line_annotation::FiberOptimizationMode;
using vc3d::line_annotation::FiberTraceSegmentMetadata;
using vc3d::line_annotation::FiberTraceState;
using vc3d::line_annotation::LineControlPoint;
using vc3d::line_annotation::SegmentInterpolationMode;
using vc3d::line_annotation::StoredControlPoint;
using vc3d::line_annotation::applyTraceReviewTags;
using vc3d::line_annotation::deriveTraceState;
using vc3d::line_annotation::hasAcceptedTraceSpan;
using vc3d::line_annotation::kTraceNeedsReviewTag;

namespace {

FiberTraceSegmentMetadata segmentWithMode(SegmentInterpolationMode mode)
{
    FiberTraceSegmentMetadata metadata;
    metadata.interpMode = mode;
    return metadata;
}

std::vector<StoredControlPoint> storedControls(
    const std::vector<SegmentInterpolationMode>& modes)
{
    std::vector<StoredControlPoint> controls(modes.size() + 1);
    for (size_t i = 0; i < modes.size(); ++i) {
        controls[i].segmentToNext = segmentWithMode(modes[i]);
    }
    return controls;
}

}  // namespace

TEST_CASE("hasAcceptedTraceSpan detects trace producers on both control types")
{
    CHECK_FALSE(hasAcceptedTraceSpan(std::vector<StoredControlPoint>{}));
    CHECK_FALSE(hasAcceptedTraceSpan(
        storedControls({SegmentInterpolationMode::Lasagna,
                        SegmentInterpolationMode::Cspline})));
    CHECK(hasAcceptedTraceSpan(
        storedControls({SegmentInterpolationMode::Lasagna,
                        SegmentInterpolationMode::Trace})));

    std::vector<LineControlPoint> sessionControls(2);
    CHECK_FALSE(hasAcceptedTraceSpan(sessionControls));
    sessionControls.front().segmentToNext =
        segmentWithMode(SegmentInterpolationMode::Trace);
    CHECK(hasAcceptedTraceSpan(sessionControls));
}

TEST_CASE("deriveTraceState maps producer and mode to the panel state")
{
    const auto lasagnaOnly =
        storedControls({SegmentInterpolationMode::Lasagna,
                        SegmentInterpolationMode::Cspline});
    const auto withTrace =
        storedControls({SegmentInterpolationMode::Trace,
                        SegmentInterpolationMode::Lasagna});

    CHECK(deriveTraceState(FiberOptimizationMode::Lasagna, lasagnaOnly) ==
          FiberTraceState::Legacy);
    CHECK(deriveTraceState(FiberOptimizationMode::NativeFiberTrace3d,
                           lasagnaOnly) == FiberTraceState::Legacy);
    CHECK(deriveTraceState(FiberOptimizationMode::NativeFiberTrace3d,
                           withTrace) == FiberTraceState::Predictions);
    CHECK(deriveTraceState(FiberOptimizationMode::Lasagna, withTrace) ==
          FiberTraceState::Mixed);
    CHECK(deriveTraceState(FiberOptimizationMode::NativeFiberTrace3d, {}) ==
          FiberTraceState::Legacy);
}

TEST_CASE("applyTraceReviewTags transitions")
{
    SUBCASE("traced geometry demands review")
    {
        std::vector<std::string> tags{"approved"};
        applyTraceReviewTags(tags, true);
        CHECK(tags == std::vector<std::string>{"approved",
                                               kTraceNeedsReviewTag});
    }
    SUBCASE("un-traced geometry leaves the workflow")
    {
        std::vector<std::string> tags{kTraceNeedsReviewTag, "zebra"};
        applyTraceReviewTags(tags, false);
        CHECK(tags == std::vector<std::string>{"zebra"});
    }
    SUBCASE("idempotent and keeps sorted order")
    {
        std::vector<std::string> tags{"alpha", "zebra"};
        applyTraceReviewTags(tags, true);
        applyTraceReviewTags(tags, true);
        CHECK(tags == std::vector<std::string>{"alpha",
                                               kTraceNeedsReviewTag,
                                               "zebra"});
    }
    SUBCASE("no-op on untagged legacy fibers")
    {
        std::vector<std::string> tags{"alpha"};
        applyTraceReviewTags(tags, false);
        CHECK(tags == std::vector<std::string>{"alpha"});
    }
}

// Golden fixture pinning the default lasagna segment descriptor — the
// exact document fiberSaveSnapshotToJson emits for a v1-upgraded span.
// External tooling (fiber_merge validators, offline upgraders) relies on
// these bytes staying stable.
TEST_CASE("default lasagna segment metadata matches the migration fixture")
{
    const auto expected = nlohmann::json::parse(R"({
        "optimizer": "native_fiber_trace3d",
        "metadata_version": 3,
        "tracer_version": 2,
        "interp_goal": "global",
        "interp_mode": "lasagna",
        "metric": null,
        "msg": "lasagna",
        "normal_manifest": "",
        "fiber_manifest": "",
        "trace_to_base_scale": 1.0,
        "meeting_error_base_voxels": null,
        "meeting_error_ratio": null,
        "meeting_source": "",
        "failure_code": "",
        "failure_detail": "",
        "lasagna_failure_code": "",
        "lasagna_failure_detail": "",
        "config": {
            "step_voxels": 4.0,
            "cone_angle_degrees": 25.0,
            "cone_angle_step_degrees": 5.0,
            "cone_grid_size": 25,
            "beam_width": 8,
            "beam_prune_distance_voxels": 1.0,
            "beam_lookahead_steps": 2,
            "smoothness_weight": 2.0,
            "smoothness_normal_weight": 0.1,
            "smoothness_tangent_weight": 10.0,
            "smoothness_free_angle_degrees": 0.0,
            "cumulative_smoothness_steps": 4,
            "cumulative_smoothness_tangent_weight": 2.0,
            "initial_free_angle_degrees": 0.0,
            "max_step_factor": 3.0,
            "meeting_accept_max_error_ratio": 0.1,
            "endpoint_accept_threshold_base_voxels": 20.0
        }
    })");

    FiberTraceSegmentMetadata metadata;
    metadata.message = "lasagna";
    const nlohmann::json actual =
        vc3d::line_annotation::fiberTraceSegmentMetadataToJson(metadata);
    CHECK(actual == expected);

    // nlohmann's operator== treats 25 and 25.0 as equal; the strict
    // validators do not, so pin the integer-typed config values.
    for (const char* key : {"cone_grid_size", "beam_width",
                            "beam_lookahead_steps",
                            "cumulative_smoothness_steps"}) {
        CAPTURE(key);
        CHECK(actual.at("config").at(key).is_number_integer());
    }
    CHECK_NOTHROW(vc::fiber_tracer::detail::validateSegmentMetadata(actual));
}

TEST_CASE("computeFiberSplitPlan slices at exact ordered line indices")
{
    using vc3d::line_annotation::computeFiberSplitPlan;
    using vc3d::line_annotation::remappedSplitControlPointIndex;

    // 9 line points on the x axis; 5 controls at line indices 0, 2, 4, 6, 8.
    std::vector<cv::Vec3d> linePoints;
    for (int i = 0; i < 9; ++i) {
        linePoints.emplace_back(static_cast<double>(i), 0.0, 0.0);
    }
    std::vector<cv::Vec3d> controls{linePoints[0], linePoints[2], linePoints[4],
                                    linePoints[6], linePoints[8]};

    const auto plan = computeFiberSplitPlan(controls, linePoints, 1);
    REQUIRE(plan.has_value());
    CHECK(plan->splitAfterControlIndex == 1);
    CHECK(plan->prefixControlCount == 2);
    CHECK(plan->prefixLineCount == 3);   // line points [0..2]
    CHECK(plan->suffixControlBegin == 2);
    CHECK(plan->suffixLineBegin == 4);   // line index of control 2

    const auto prefixHome = remappedSplitControlPointIndex(*plan, 1);
    REQUIRE(prefixHome.has_value());
    CHECK_FALSE(prefixHome->first);
    CHECK(prefixHome->second == 1);
    const auto suffixFirst = remappedSplitControlPointIndex(*plan, 2);
    REQUIRE(suffixFirst.has_value());
    CHECK(suffixFirst->first);
    CHECK(suffixFirst->second == 0);
    const auto suffixLast = remappedSplitControlPointIndex(*plan, 4);
    REQUIRE(suffixLast.has_value());
    CHECK(suffixLast->first);
    CHECK(suffixLast->second == 2);
    CHECK_FALSE(remappedSplitControlPointIndex(*plan, -1).has_value());
}

TEST_CASE("computeFiberSplitPlan enforces two control points per half")
{
    using vc3d::line_annotation::computeFiberSplitPlan;

    std::vector<cv::Vec3d> linePoints;
    for (int i = 0; i < 9; ++i) {
        linePoints.emplace_back(static_cast<double>(i), 0.0, 0.0);
    }
    const std::vector<cv::Vec3d> controls{linePoints[0], linePoints[2],
                                          linePoints[4], linePoints[6],
                                          linePoints[8]};

    CHECK_FALSE(computeFiberSplitPlan(controls, linePoints, 0).has_value());
    CHECK(computeFiberSplitPlan(controls, linePoints, 2).has_value());
    CHECK_FALSE(computeFiberSplitPlan(controls, linePoints, 3).has_value());
    CHECK_FALSE(computeFiberSplitPlan(controls, linePoints, 17).has_value());

    // A 4-control fiber has exactly one legal cut.
    const std::vector<cv::Vec3d> fourControls{linePoints[0], linePoints[2],
                                              linePoints[4], linePoints[6]};
    CHECK(computeFiberSplitPlan(fourControls, linePoints, 1).has_value());
    CHECK_FALSE(computeFiberSplitPlan(fourControls, linePoints, 2).has_value());
}

TEST_CASE("computeFiberSplitPlan applies the loader's exact-membership scan")
{
    using vc3d::line_annotation::computeFiberSplitPlan;

    std::vector<cv::Vec3d> linePoints;
    for (int i = 0; i < 9; ++i) {
        linePoints.emplace_back(static_cast<double>(i), 0.0, 0.0);
    }
    std::vector<cv::Vec3d> controls{linePoints[0], linePoints[2], linePoints[4],
                                    linePoints[6]};

    // A control nudged beyond the 1e-8 tolerance no longer matches its line
    // point; within the tolerance it still does.
    controls[1][1] = 1.0e-7;
    CHECK_FALSE(computeFiberSplitPlan(controls, linePoints, 1).has_value());
    controls[1][1] = 1.0e-9;
    CHECK(computeFiberSplitPlan(controls, linePoints, 1).has_value());

    // Coincident line points: the forward scan must keep advancing, so two
    // controls sharing one position match distinct line indices.
    std::vector<cv::Vec3d> repeatedLine{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                                        {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
                                        {3.0, 0.0, 0.0}, {4.0, 0.0, 0.0}};
    const std::vector<cv::Vec3d> repeatedControls{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                                                  {1.0, 0.0, 0.0}, {3.0, 0.0, 0.0}};
    const auto repeatedPlan = computeFiberSplitPlan(repeatedControls, repeatedLine, 1);
    REQUIRE(repeatedPlan.has_value());
    CHECK(repeatedPlan->prefixLineCount == 2);  // first "1" belongs to the prefix
    CHECK(repeatedPlan->suffixLineBegin == 2);  // second "1" starts the suffix
}

TEST_CASE("reversedStoredControlPoints shifts span descriptors with their spans")
{
    using vc3d::line_annotation::reversedStoredControlPoints;

    // 4 CPs, 3 spans with distinct interp modes: T, L, C.
    std::vector<StoredControlPoint> controls;
    for (int i = 0; i < 4; ++i) {
        controls.emplace_back(cv::Vec3d(static_cast<double>(i), 0.0, 0.0));
    }
    controls[0].segmentToNext = segmentWithMode(SegmentInterpolationMode::Trace);
    controls[1].segmentToNext = segmentWithMode(SegmentInterpolationMode::Lasagna);
    controls[2].segmentToNext = segmentWithMode(SegmentInterpolationMode::Cspline);

    const auto reversed = reversedStoredControlPoints(controls);
    REQUIRE(reversed.size() == 4);
    // Positions reversed.
    CHECK(static_cast<cv::Vec3d>(reversed[0]) == cv::Vec3d(3.0, 0.0, 0.0));
    CHECK(static_cast<cv::Vec3d>(reversed[3]) == cv::Vec3d(0.0, 0.0, 0.0));
    // Reversed span j carries original span (n-2-j)'s descriptor.
    REQUIRE(reversed[0].segmentToNext.has_value());
    CHECK(reversed[0].segmentToNext->interpMode == SegmentInterpolationMode::Cspline);
    REQUIRE(reversed[1].segmentToNext.has_value());
    CHECK(reversed[1].segmentToNext->interpMode == SegmentInterpolationMode::Lasagna);
    REQUIRE(reversed[2].segmentToNext.has_value());
    CHECK(reversed[2].segmentToNext->interpMode == SegmentInterpolationMode::Trace);
    // The new final CP carries no descriptor (v3 contract).
    CHECK_FALSE(reversed[3].segmentToNext.has_value());

    CHECK(reversedStoredControlPoints({}).empty());
    std::vector<StoredControlPoint> single{
        StoredControlPoint{cv::Vec3d(1.0, 2.0, 3.0)}};
    const auto reversedSingle = reversedStoredControlPoints(single);
    REQUIRE(reversedSingle.size() == 1);
    CHECK_FALSE(reversedSingle[0].segmentToNext.has_value());
}

TEST_CASE("computeFiberMergeGeometry slices tails and synthesizes the join span")
{
    using vc3d::line_annotation::computeFiberMergeGeometry;
    using vc3d::line_annotation::SegmentInterpolationGoal;
    using vc3d::line_annotation::validateStoredControlPoints;

    // Side a: CPs at x = 1, 3 on a line 0..5 (tails at 0, 4, 5).
    std::vector<StoredControlPoint> aControls{
        StoredControlPoint{cv::Vec3d(1.0, 0.0, 0.0)},
        StoredControlPoint{cv::Vec3d(3.0, 0.0, 0.0)}};
    aControls[0].segmentToNext = segmentWithMode(SegmentInterpolationMode::Trace);
    std::vector<cv::Vec3d> aLine;
    for (int i = 0; i <= 5; ++i) {
        aLine.emplace_back(static_cast<double>(i), 0.0, 0.0);
    }
    // Side b: CPs at y = 11, 13 on a line y = 9..14 (tails at 9, 10, 14).
    std::vector<StoredControlPoint> bControls{
        StoredControlPoint{cv::Vec3d(0.0, 11.0, 0.0)},
        StoredControlPoint{cv::Vec3d(0.0, 13.0, 0.0)}};
    bControls[0].segmentToNext = segmentWithMode(SegmentInterpolationMode::Cspline);
    std::vector<cv::Vec3d> bLine;
    for (int i = 9; i <= 14; ++i) {
        bLine.emplace_back(0.0, static_cast<double>(i), 0.0);
    }

    const auto merged = computeFiberMergeGeometry(aControls, aLine, bControls, bLine);
    REQUIRE(merged.has_value());
    REQUIRE(merged->controlPoints.size() == 4);
    CHECK(merged->joinControlIndex == 1);
    // a tail (x=4,5) and b tail (y=9,10) dropped: line runs x=0..3 then y=11..14.
    REQUIRE(merged->linePoints.size() == 8);
    CHECK(merged->linePoints[3] == cv::Vec3d(3.0, 0.0, 0.0));
    CHECK(merged->linePoints[4] == cv::Vec3d(0.0, 11.0, 0.0));
    // Join descriptor synthesized on a's former final CP; b's descriptors ride.
    REQUIRE(merged->controlPoints[1].segmentToNext.has_value());
    CHECK(merged->controlPoints[1].segmentToNext->interpGoal ==
          SegmentInterpolationGoal::Global);
    CHECK(merged->controlPoints[1].segmentToNext->interpMode ==
          SegmentInterpolationMode::Lasagna);
    REQUIRE(merged->controlPoints[2].segmentToNext.has_value());
    CHECK(merged->controlPoints[2].segmentToNext->interpMode ==
          SegmentInterpolationMode::Cspline);
    CHECK_NOTHROW(validateStoredControlPoints(merged->controlPoints));

    // A control off its line breaks the ordered scan.
    auto badControls = aControls;
    badControls[1] = StoredControlPoint{cv::Vec3d(3.0, 1.0e-6, 0.0)};
    CHECK_FALSE(
        computeFiberMergeGeometry(badControls, aLine, bControls, bLine).has_value());
    // Fewer than 2 CPs per side is not mergeable.
    CHECK_FALSE(computeFiberMergeGeometry({aControls[0]}, aLine, bControls, bLine)
                    .has_value());
}

TEST_CASE("reverse-then-merge covers the endpoint orientation recipe")
{
    using vc3d::line_annotation::computeFiberMergeGeometry;
    using vc3d::line_annotation::reversedStoredControlPoints;

    // Fiber whose merge endpoint is its FIRST CP: reverse it so it ends at
    // the join, then merge; the reversed side must still pass the scan.
    std::vector<StoredControlPoint> controls{
        StoredControlPoint{cv::Vec3d(5.0, 0.0, 0.0)},   // merge endpoint
        StoredControlPoint{cv::Vec3d(7.0, 0.0, 0.0)}};
    controls[0].segmentToNext = segmentWithMode(SegmentInterpolationMode::Trace);
    std::vector<cv::Vec3d> line{{4.0, 0.0, 0.0}, {5.0, 0.0, 0.0},
                                {6.0, 0.0, 0.0}, {7.0, 0.0, 0.0},
                                {8.0, 0.0, 0.0}};

    auto reversedControls = reversedStoredControlPoints(controls);
    auto reversedLine = line;
    std::reverse(reversedLine.begin(), reversedLine.end());

    std::vector<StoredControlPoint> bControls{
        StoredControlPoint{cv::Vec3d(3.0, 0.0, 0.0)},
        StoredControlPoint{cv::Vec3d(1.0, 0.0, 0.0)}};
    std::vector<cv::Vec3d> bLine{{3.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};

    const auto merged =
        computeFiberMergeGeometry(reversedControls, reversedLine, bControls, bLine);
    REQUIRE(merged.has_value());
    REQUIRE(merged->controlPoints.size() == 4);
    // Reversed side runs 7 -> 5; join meets b running 3 -> 1.
    CHECK(static_cast<cv::Vec3d>(merged->controlPoints[0]) == cv::Vec3d(7.0, 0.0, 0.0));
    CHECK(static_cast<cv::Vec3d>(merged->controlPoints[1]) == cv::Vec3d(5.0, 0.0, 0.0));
    CHECK(static_cast<cv::Vec3d>(merged->controlPoints[2]) == cv::Vec3d(3.0, 0.0, 0.0));
    // The reversed side's span descriptor traveled with its span, and the
    // join descriptor replaced it on the new boundary CP.
    REQUIRE(merged->controlPoints[0].segmentToNext.has_value());
    CHECK(merged->controlPoints[0].segmentToNext->interpMode ==
          SegmentInterpolationMode::Trace);
    REQUIRE(merged->controlPoints[1].segmentToNext.has_value());
    CHECK(merged->controlPoints[1].segmentToNext->interpMode ==
          SegmentInterpolationMode::Lasagna);
    // Tails dropped on both sides of the join: 5.0 then 3.0 consecutive.
    CHECK(merged->linePoints[merged->linePoints.size() - 4] ==
          cv::Vec3d(5.0, 0.0, 0.0));
    CHECK(merged->linePoints[merged->linePoints.size() - 3] ==
          cv::Vec3d(3.0, 0.0, 0.0));
}
