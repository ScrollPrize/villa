#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "LineAnnotationFiberSegments.hpp"
#include "vc/fiber_tracer/FiberJson.hpp"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

using vc3d::line_annotation::FiberOptimizationMode;
using vc3d::line_annotation::FiberTraceSegmentMetadata;
using vc3d::line_annotation::FiberTraceState;
using vc3d::line_annotation::LineControlPoint;
using vc3d::line_annotation::SegmentInterpolationMode;
using vc3d::line_annotation::StoredControlPoint;
using vc3d::line_annotation::applyTraceReviewTagExclusivity;
using vc3d::line_annotation::applyTraceReviewTags;
using vc3d::line_annotation::deriveTraceState;
using vc3d::line_annotation::hasAcceptedTraceSpan;
using vc3d::line_annotation::kTraceNeedsReviewTag;
using vc3d::line_annotation::kTraceVerifiedTag;

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
    SUBCASE("traced geometry demands review and clears verified")
    {
        std::vector<std::string> tags{"approved", kTraceVerifiedTag};
        applyTraceReviewTags(tags, true);
        CHECK(tags == std::vector<std::string>{"approved",
                                               kTraceNeedsReviewTag});
    }
    SUBCASE("un-traced geometry leaves the workflow")
    {
        std::vector<std::string> tags{kTraceNeedsReviewTag,
                                      kTraceVerifiedTag, "zebra"};
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

TEST_CASE("applyTraceReviewTagExclusivity keeps the pair mutually exclusive")
{
    std::vector<std::string> tags{kTraceNeedsReviewTag, kTraceVerifiedTag};
    applyTraceReviewTagExclusivity(tags, kTraceVerifiedTag);
    CHECK(tags == std::vector<std::string>{kTraceVerifiedTag});

    tags = {kTraceNeedsReviewTag, kTraceVerifiedTag};
    applyTraceReviewTagExclusivity(tags, kTraceNeedsReviewTag);
    CHECK(tags == std::vector<std::string>{kTraceNeedsReviewTag});

    tags = {kTraceNeedsReviewTag, "other"};
    applyTraceReviewTagExclusivity(tags, "other");
    CHECK(tags == std::vector<std::string>{kTraceNeedsReviewTag, "other"});
}

// Golden fixture shared with scripts/fiber_migrate_v1_to_v3.py: the Python
// LASAGNA_SEGMENT constant must serialize the exact document VC3D writes
// for a v1-upgraded span. Keep this literal textually identical to the
// Python constant (and to what fiberSaveSnapshotToJson emits).
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
