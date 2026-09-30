#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberReferenceReplayBenchmark.hpp"
#include "vc/fiber_tracer/FiberAnchorCoverage.hpp"
#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/fiber_tracer/FiberletPipeline.hpp"
#include "vc/fiber_tracer/FiberJson.hpp"
#include "vc/lasagna/LineOptimizer.hpp"

#include <cmath>
#include <limits>

namespace
{

class ConstantNormalSampler final : public vc::lasagna::NormalSampler
{
public:
    explicit ConstantNormalSampler(cv::Vec3d normal, bool valid = true)
        : normal_(normal), valid_(valid)
    {
    }

    vc::lasagna::NormalSample sampleNormal(
        const cv::Vec3d&) const override
    {
        return {normal_, valid_, valid_ ? "" : "missing"};
    }

private:
    cv::Vec3d normal_;
    bool valid_ = true;
};

}  // namespace

TEST_CASE("reference replay interval tolerates accumulated roundoff but not missing coverage")
{
    using namespace vc::fiber_tracer;
    FiberReferenceReplayCase replayCase;
    // Actual training case 2293: a 19,394-vertex reverse Paris4 run.
    replayCase.referenceLengthBaseVoxels = 154719.34786287573;
    replayCase.pointsBaseXYZ.resize(19394);
    const auto check = [&](auto replay) {
        replay.referenceEndArcBase = 154719.3478628727;
        replay.completedReferenceArcBase = replay.referenceEndArcBase;
        auto outcome = measureFiberReferenceReplayOutcome(replayCase, replay);
        CHECK(outcome.evaluationComplete);
        CHECK(outcome.evaluatedThroughBaseVoxels == replayCase.referenceLengthBaseVoxels);
        const std::vector<FiberReferenceReplayOutcome> outcomes{outcome};
        CHECK_NOTHROW(summarizeFiberReferenceReplay(1, outcomes, 2.4));
        replay.completedReferenceArcBase -= 0.01;
        CHECK_FALSE(measureFiberReferenceReplayOutcome(replayCase, replay).evaluationComplete);
        replay.referenceEndArcBase -= 0.01;
        CHECK_THROWS_AS(measureFiberReferenceReplayOutcome(replayCase, replay), std::invalid_argument);
        replay.referenceEndArcBase = std::numeric_limits<double>::quiet_NaN();
        CHECK_THROWS_AS(measureFiberReferenceReplayOutcome(replayCase, replay), std::invalid_argument);
    };
    check(FiberReplayTraceResult{});
    check(FiberletGraphReplayResult{});
}

TEST_CASE("precomputed replay emits fixed offset geometry and terminal samples")
{
    using namespace vc::fiber_tracer;
    const ConstantNormalSampler normals({0,0,1});
    const std::vector<cv::Vec3d> reference{{0,0,0},{82,0,0}};
    std::vector<PolylineArcGeometry> candidates{makePolylineArcGeometry({{0,2,0},{82,2,0}})};
    const auto replay = tracePrecomputedFiberReplay(candidates, reference, normals);
    CHECK(replay.failures.empty());
    REQUIRE(replay.segments.size() == 1);
    const auto& points = replay.segments.front().tracePointsBase;
    REQUIRE(points.size() >= 3);
    CHECK(points[1] == cv::Vec3d(0,2,0)); // Actual offset acquisition, not snapped GT.
    CHECK(points[2] == cv::Vec3d(8,2,0));
    CHECK(points.back() == cv::Vec3d(82,2,0)); // Non-step-multiple endpoint.
    auto reverse = reference;
    std::reverse(reverse.begin(), reverse.end());
    const auto backwards = tracePrecomputedFiberReplay(candidates, reverse, normals);
    CHECK(backwards.failures.empty());
    CHECK(backwards.segments.front().tracePointsBase.back() == cv::Vec3d(0,2,0));

    candidates.clear();
    const auto missing = tracePrecomputedFiberReplay(candidates, reference, normals);
    REQUIRE_FALSE(missing.failures.empty());
    CHECK(missing.failures.front().reason == "no_aligned_candidate");
    CHECK(missing.completedReferenceArcBase == doctest::Approx(82));
    candidates.push_back(makePolylineArcGeometry({{0,-10,0},{0,10,0}}));
    const auto wrong = tracePrecomputedFiberReplay(candidates, reference, normals);
    CHECK(wrong.failures.front().reason == "no_aligned_candidate");
    candidates[0] = makePolylineArcGeometry({{0,0,0},{10,0,0}});
    const auto shortLine = tracePrecomputedFiberReplay(candidates, reference, normals);
    REQUIRE_FALSE(shortLine.failures.empty());
    CHECK(shortLine.failures.front().reason == "candidate_end");
    CHECK(shortLine.segments.front().tracePointsBase.back() == cv::Vec3d(10,0,0));
    CHECK(shortLine.completedReferenceArcBase == doctest::Approx(82));
    candidates[0] = makePolylineArcGeometry({{0,0,21},{82,0,21}});
    const auto outside = tracePrecomputedFiberReplay(candidates, reference, normals);
    CHECK(outside.failures.front().reason == "no_aligned_candidate");
}

TEST_CASE("native collection parser validates scale without applying historical transforms")
{
    using namespace vc::fiber_tracer;
    const nlohmann::json entry = {
        {"type", "vc3d_fiber"}, {"version", 1}, {"hv_classification", "H"},
        {"line_points", {{100,200,300},{110,200,300}}},
        {"control_points", {{100,200,300},{110,200,300}}}};
    nlohmann::json root = {{"type", "vc3d_fiber_collection"}, {"version", 1}, {"scale", 1},
        {"point_collections", nlohmann::json::array({entry})}};
    const auto parsed = parseNativeVc3dFiberCollection(root, "fixture");
    REQUIRE(parsed.size() == 1);
    CHECK(parsed[0].linePoints.front() == cv::Vec3d(100,200,300));
    root["scale"] = 4;
    CHECK_THROWS_AS(parseNativeVc3dFiberCollection(root, "fixture"), std::invalid_argument);
    root["scale"] = 1;
    root["vc_open_data_source_coordinate_level"] = 2;
    CHECK_THROWS_AS(parseNativeVc3dFiberCollection(root, "fixture"), std::invalid_argument);
}

TEST_CASE("precomputed acquisition freezes the initially closest candidate without future GT reranking")
{
    using namespace vc::fiber_tracer;
    const ConstantNormalSampler normals({0,0,1});
    const std::vector<cv::Vec3d> reference{{0,0,0},{100,0,0}};
    const std::vector<PolylineArcGeometry> candidates{
        makePolylineArcGeometry({{0,1,0},{16,1,0},{32,1,40},{100,1,40}}),
        makePolylineArcGeometry({{0,2,0},{100,2,0}})};
    const auto replay = tracePrecomputedFiberReplay(candidates, reference, normals);
    REQUIRE_FALSE(replay.failures.empty());
    CHECK(replay.failures.front().reason == "distance_above_threshold");
    REQUIRE_FALSE(replay.segments.empty());
    const auto& first = replay.segments.front();
    REQUIRE(first.tracePointsBase.size() > 2);
    CHECK(first.tracePointsBase[1] == cv::Vec3d(0,1,0));
    CHECK(first.tracePointsBase.back()[1] == doctest::Approx(1));
    CHECK(first.tracePointsBase.back()[2] > 20);
    // A future-aware selector would take candidate 1 immediately and report
    // zero failures. A mid-segment switch would conceal candidate 0's failure.

    const std::vector<PolylineArcGeometry> ties{
        makePolylineArcGeometry({{0,1,0},{100,1,0}}),
        makePolylineArcGeometry({{0,-1,0},{100,-1,0}})};
    const auto tied = tracePrecomputedFiberReplay(ties, reference, normals);
    REQUIRE_FALSE(tied.segments.empty());
    CHECK(tied.segments.front().tracePointsBase[1] == cv::Vec3d(0,1,0));
    CHECK(tied.failures.empty());
}

TEST_CASE("anchor coverage uses ellipse flat caps and arclength gaps")
{
    const ConstantNormalSampler normals({0,0,1});
    std::vector<cv::Vec3d> line{{0,0,0},{128,0,0}};
    const std::vector<cv::Vec3d> anchors{
        {16,0,40}, {48,80,0}, {80,0,40.01}, {112,80.01,0},
        {-1,0,0}, {129,0,0}, {48,0,0}, {128,0,0}};
    auto result = vc::fiber_tracer::measureFiberAnchorCoverage(line,anchors,normals,1,40,2,32);
    CHECK(result["accepted_anchors"].size()==4);
    CHECK(result["end_cap_rejections"]==2);
    CHECK(result["unique_arcs_base"]==nlohmann::json({16,48,128}));
    CHECK(result["interior_gaps_base"]==nlohmann::json({32,80}));
    CHECK(result["leading_gap_base"]==16);
    CHECK(result["trailing_gap_base"]==0);
    CHECK(result["empty_arc_bins"]==1);
    std::reverse(line.begin(),line.end());
    const auto reverse=vc::fiber_tracer::measureFiberAnchorCoverage(line,anchors,normals,1,40,2,32);
    CHECK(reverse["interior_gaps_base"]==nlohmann::json({80,32}));
    CHECK(reverse["leading_gap_base"]==0);
    CHECK(reverse["trailing_gap_base"]==16);
}

TEST_CASE("weighted anchor fit is single sign invariant and moves toward presence")
{
    using namespace vc::fiber_tracer;
    FiberAnchorConfig config;
    config.fitMethod="weighted";
    config.peakGradientWeight=0;
    config.weightedFitIterations=0;
    std::vector<FiberAnchorObservation> samples;
    for(int z=0;z<4;++z) for(int y=0;y<4;++y) for(int x=0;x<4;++x)
        samples.push_back({{float(x),float(y),float(z)},{1,0,0},x==3?1.0F:0.1F,true});
    const auto initial=fitFiberCellAnchors({0,0,0},{0,0,0},{4,4,4},samples,config);
    REQUIRE(initial.retainedAnchorCount==1);
    CHECK(initial.components[0].anchor.positionPredictionXYZ[0]==doctest::Approx(1.5));
    config.weightedFitIterations=8;
    const auto refined=fitFiberCellAnchors({0,0,0},{0,0,0},{4,4,4},samples,config);
    REQUIRE(refined.retainedAnchorCount==1);
    const auto position=refined.components[0].anchor.positionPredictionXYZ;
    CHECK(position[0]>2.0F);
    for(int d=0;d<3;++d) {CHECK(position[d]>=0);CHECK(position[d]<=3);}
    for(auto& sample:samples) sample.direction=-sample.direction;
    const auto flipped=fitFiberCellAnchors({0,0,0},{0,0,0},{4,4,4},samples,config);
    CHECK(flipped.components[0].anchor.positionPredictionXYZ==position);
    CHECK(flipped.components[0].anchor.axisXYZ==refined.components[0].anchor.axisXYZ);
    for(auto& sample:samples) sample.valid=false;
    CHECK(fitFiberCellAnchors({0,0,0},{0,0,0},{4,4,4},samples,config).retainedAnchorCount==0);
}

TEST_CASE("weighted fit cache identities separate while legacy identity ignores disabled settings")
{
    using namespace vc::fiber_tracer;
    FiberAnchorConfig a;
    FiberletPathConfig p;
    FiberPredictionGridInfo grid{{32,32,32},8};
    auto identity=[&] {return makeFiberletDatasetMetadata(FiberletDatasetKind::Anchors,grid,a,p,128,nlohmann::json::object()).datasetFingerprint;};
    auto legacy=identity();
    a.weightedFitIterations=8;
    CHECK(identity()==legacy);
    a.fitMethod="weighted";
    auto weighted=identity();
    CHECK(weighted!=legacy);
    a.weightedFitIterations=4;
    CHECK(identity()!=weighted);
}

TEST_CASE("canonical generated cache preserves scientific settings across runtime roots")
{
    using namespace vc::fiber_tracer;
    FiberAnchorConfig anchors;
    FiberletPathConfig paths;
    anchors.parallelThreads = 32;
    paths.parallelThreads = 32;
    paths.longitudinalStepPredictionVoxels = 1.0F;
    FiberPredictionGridInfo grid{{32,32,32},8};
    const auto sources = nlohmann::json{{"model", "test"}};
    const auto anchorMetadata = makeFiberletDatasetMetadata(
        FiberletDatasetKind::Anchors, grid, anchors, paths, 128, sources);
    const auto pathMetadata = makeFiberletDatasetMetadata(
        FiberletDatasetKind::Fiberlets, grid, anchors, paths, 128, sources);
    FiberletChunkCacheOptions cache;
    cache.service.fetchConcurrency.workerCapacity = 8;
    auto sampler = [](const auto&, int, auto&) {};
    const auto normals = std::make_shared<ConstantNormalSampler>(cv::Vec3d{0,0,1});
    const auto first = makeCanonicalFiberletOnDemandConfig(
        "/first/anchors", "/first/paths", anchorMetadata, pathMetadata,
        grid, anchors, paths, sampler, normals, cache);
    const auto second = makeCanonicalFiberletOnDemandConfig(
        "/second/anchors", "/second/paths", anchorMetadata, pathMetadata,
        grid, anchors, paths, sampler, normals, cache);
    CHECK(first.fiberletMetadata.datasetFingerprint == second.fiberletMetadata.datasetFingerprint);
    CHECK(first.pathConfig.longitudinalStepPredictionVoxels == 1.0F);
    CHECK(first.anchorConfig.parallelThreads == 1);
    CHECK(first.pathConfig.parallelThreads == 1);
    CHECK(first.fiberletCacheOptions.service.fetchConcurrency.workerCapacity == 8);
    CHECK(first.anchorCellPredicate({999,999,999}));
    CHECK(first.pointPredicate({999,999,999}));
    auto changedPaths = paths;
    changedPaths.longitudinalStepPredictionVoxels = 2.0F;
    CHECK(makeFiberletDatasetMetadata(FiberletDatasetKind::Fiberlets, grid,
        anchors, changedPaths, 128, sources).datasetFingerprint != pathMetadata.datasetFingerprint);
}

TEST_CASE("anchor direction error ignores sign")
{
    const ConstantNormalSampler normals({0,0,1});
    const std::vector<cv::Vec3d> points{{0,0,0},{96,0,0}},anchors{{32,0,0},{64,0,0}},
        axes{{-1,0,0},{0,1,0}};
    const auto r=vc::fiber_tracer::measureFiberAnchorCoverage(points,anchors,normals,1,40,2,32,axes);
    CHECK(r["accepted_anchors"][0]["direction_error_degrees"].get<double>()==doctest::Approx(0));
    CHECK(r["accepted_anchors"][1]["direction_error_degrees"].get<double>()==doctest::Approx(90));
}

TEST_CASE("anchor coverage preserves empty singleton and invalid normal cases")
{
    const ConstantNormalSampler missing({0,0,1},false);
    const std::vector<cv::Vec3d> line{{0,0,0},{65,0,0}};
    auto empty=vc::fiber_tracer::measureFiberAnchorCoverage(line,{},missing,1,40,2,32);
    CHECK(empty["no_anchor_length_base"]==65);
    CHECK(empty["interior_gaps_base"].empty());
    CHECK(empty["leading_gap_base"].is_null());
    CHECK(empty["empty_arc_bin_runs"]==nlohmann::json({3}));
    CHECK(empty["empty_arc_bin_length_base"]==65);
    const std::vector<cv::Vec3d> anchors{{32,60,0},{65,0,0}};
    auto single=vc::fiber_tracer::measureFiberAnchorCoverage(line,anchors,missing,1,40,2,32);
    CHECK(single["invalid_normal_tests"]==2);
    CHECK(single["accepted_anchors"].size()==1);
    CHECK(single["interior_gaps_base"].empty());
    CHECK(single["empty_arc_bin_runs"]==nlohmann::json({2}));
    CHECK(single["empty_arc_bin_length_base"]==64);
}

TEST_CASE("anchor indexed projection agrees with shared projection on bends")
{
    const ConstantNormalSampler normals({0,0,1});
    const std::vector<cv::Vec3d> line{{0,0,0},{100,0,0},{100,100,0},{200,100,0}};
    const auto geometry=vc::fiber_tracer::makePolylineArcGeometry(line);
    std::vector<cv::Vec3d> anchors;
    for(int x=10;x<200;x+=13) for(int y=0;y<100;y+=17) anchors.emplace_back(x,y,0);
    auto result=vc::fiber_tracer::measureFiberAnchorCoverage(line,anchors,normals,1,40,2,32);
    for(const auto& a:result["accepted_anchors"]) {
        auto expected=vc::fiber_tracer::projectPointToPolylineArc(geometry,
            anchors[a["anchor_index"].get<size_t>()],0,geometry.length());
        CHECK(a["reference_arc_base"].get<double>()==doctest::Approx(expected.arc));
    }
}

TEST_CASE("reference replay preparation retains crop re-entry runs in both directions")
{
    const std::vector<std::vector<cv::Vec3d>> fibers{{{-1, 1, 1}, {1, 1, 1}, {3, 1, 1}, {1, 1, 1}, {-1, 1, 1}}};
    const auto cases = vc::fiber_tracer::makeFiberReferenceReplayCases(fibers, {0, 0, 0}, {2, 2, 2});

    REQUIRE(cases.size() == 4);
    CHECK(cases[0].runIndex == 0);
    CHECK_FALSE(cases[0].reverse);
    CHECK(cases[1].reverse);
    CHECK(cases[2].runIndex == 1);
    CHECK(cases[0].pointsBaseXYZ.front()[0] == doctest::Approx(0.0));
    CHECK(cases[0].pointsBaseXYZ.back()[0] == doctest::Approx(2.0));
    CHECK(cases[1].pointsBaseXYZ.front() == cases[0].pointsBaseXYZ.back());
}

TEST_CASE("reference replay outcome retains every failure and credits only seeded spans")
{
    vc::fiber_tracer::FiberReferenceReplayCase replayCase;
    replayCase.sourceIndex = 2;
    replayCase.runIndex = 3;
    replayCase.referenceLengthBaseVoxels = 10.0;
    vc::fiber_tracer::FiberletGraphReplayResult replay;
    replay.referenceBeginArcBase = 0.0;
    replay.referenceEndArcBase = 10.0;
    replay.completedReferenceArcBase = 10.0;
    replay.segments.resize(3);
    replay.segments[0].seedKey = vc::fiber_tracer::FiberletStorageKey{};
    replay.segments[0].endReferenceArcBase = 4.25;
    replay.segments[0].matches.push_back({0, 0.0, 0.0, {}, 0.0, 1.0, {}});
    replay.segments[1].startReferenceArcBase = 4.25;
    replay.segments[1].endReferenceArcBase = 6.0;
    replay.segments[1].terminationReason = "missing_seed_gap";
    replay.segments[2].seedKey = vc::fiber_tracer::FiberletStorageKey{};
    replay.segments[2].startReferenceArcBase = 6.0;
    replay.segments[2].endReferenceArcBase = 10.0;
    // The seed search begins at resetArc, before the explicit missing-seed gap.
    // Only [seedArc,endArc] is traced, not [resetArc,endArc].
    replay.segments[2].matches.push_back({0, 6.0, 6.0, {}, 4.25, 6.0, {}});
    replay.failures.push_back({0, 0, "distance_above_threshold", 4.25, 0.425, {4.25, 0, 0}});
    replay.failures.push_back({1, 1, "missing_seed_gap", 4.25, 0.425, {4.25, 0, 0}});

    const auto forward = vc::fiber_tracer::measureFiberReferenceReplayOutcome(replayCase, replay);
    CHECK(forward.tracedLengthBaseVoxels == doctest::Approx(8.25));
    CHECK(forward.evaluationComplete);
    CHECK_FALSE(forward.failureFree);
    REQUIRE(forward.failures.size() == 2);
    CHECK(forward.failures[0].tracedSpanBaseVoxels == doctest::Approx(4.25));
    CHECK(forward.failures[1].tracedSpanBaseVoxels == doctest::Approx(0.0));
    CHECK(forward.failures[0].sourceReferenceArcBaseVoxels == doctest::Approx(4.25));
    replayCase.reverse = true;
    const auto reverse = vc::fiber_tracer::measureFiberReferenceReplayOutcome(replayCase, replay);
    CHECK(reverse.reverse);
    CHECK(reverse.failures[0].sourceReferenceArcBaseVoxels == doctest::Approx(5.75));
}

TEST_CASE("reference replay summary includes successes and explicit voxel scale")
{
    std::vector<vc::fiber_tracer::FiberReferenceReplayOutcome> outcomes(2);
    outcomes[0].referenceLengthBaseVoxels = 100.0;
    outcomes[0].tracedLengthBaseVoxels = 100.0;
    outcomes[0].evaluatedThroughBaseVoxels = 100.0;
    outcomes[0].seededSpans = 1;
    outcomes[0].evaluationComplete = true;
    outcomes[0].failureFree = true;
    outcomes[1].reverse = true;
    outcomes[1].referenceLengthBaseVoxels = 100.0;
    outcomes[1].tracedLengthBaseVoxels = 75.0;
    outcomes[1].evaluatedThroughBaseVoxels = 100.0;
    outcomes[1].seededSpans = 2;
    outcomes[1].evaluationComplete = true;
    outcomes[1].failures.push_back({0, 0, "route_state_limit", 25.0, 0.25, 75.0, 0.75, 25.0});

    const auto summary = vc::fiber_tracer::summarizeFiberReferenceReplay(1, outcomes, 5.0);
    CHECK(summary.directedCases == 2);
    CHECK(summary.evaluatedCases == 2);
    CHECK(summary.failureFreeCases == 1);
    CHECK(summary.totalFailures == 1);
    CHECK(summary.meanCreditedLengthMillimeters == doctest::Approx(0.4375));
    CHECK(summary.meanSeededSpanLengthMillimeters == doctest::Approx(175.0 / 3.0 * 0.005));
    CHECK(summary.meanFailedSpanLengthMillimeters == doctest::Approx(0.125));
    CHECK(summary.lengthWeightedSuccessPercent == doctest::Approx(87.5));
    CHECK(summary.failuresPerDirectedMillimeter == doctest::Approx(1.0));
    CHECK(summary.meanSegmentLengthBaseVoxels == doctest::Approx(100.0));
    CHECK(summary.meanSegmentLengthMillimeters == doctest::Approx(0.5));
    CHECK(summary.meanSegmentLengthPercent == doctest::Approx(50.0));
    REQUIRE(summary.failureReasons.size() == 1);
    CHECK(summary.failureReasons.front().first == "route_state_limit");
}

TEST_CASE("reference replay mean segment length handles zero and multiple failures")
{
    vc::fiber_tracer::FiberReferenceReplayOutcome outcome;
    outcome.referenceLengthBaseVoxels = 200.0;
    outcome.tracedLengthBaseVoxels = 200.0;
    outcome.evaluatedThroughBaseVoxels = 200.0;
    outcome.evaluationComplete = true;
    outcome.failureFree = true;

    const std::array zeroFailureOutcomes{outcome};
    const auto zero = vc::fiber_tracer::summarizeFiberReferenceReplay(1, zeroFailureOutcomes, 2.4);
    CHECK(zero.meanSegmentLengthBaseVoxels == doctest::Approx(200.0));
    CHECK(zero.meanSegmentLengthMillimeters == doctest::Approx(0.48));
    CHECK(zero.meanSegmentLengthPercent == doctest::Approx(100.0));

    outcome.failureFree = false;
    outcome.failures.resize(4);
    for (std::size_t index = 0; index < outcome.failures.size(); ++index) {
        outcome.failures[index].index = index;
        outcome.failures[index].reason = "distance_above_threshold";
    }
    const std::array fourFailureOutcomes{outcome};
    const auto four = vc::fiber_tracer::summarizeFiberReferenceReplay(1, fourFailureOutcomes, 2.4);
    CHECK(four.meanSegmentLengthBaseVoxels == doctest::Approx(40.0));
    CHECK(four.meanSegmentLengthMillimeters == doctest::Approx(0.096));
    CHECK(four.meanSegmentLengthPercent == doctest::Approx(20.0));

    outcome.evaluationComplete = false;
    outcome.evaluatedThroughBaseVoxels = 100.0;
    const std::array incompleteOutcomes{outcome};
    CHECK_THROWS_AS(vc::fiber_tracer::summarizeFiberReferenceReplay(1, incompleteOutcomes, 2.4), std::invalid_argument);
}

TEST_CASE("reference replay corpus score uses failures plus one regardless of direction count")
{
    vc::fiber_tracer::FiberReferenceReplayOutcome outcome;
    outcome.referenceLengthBaseVoxels = 10000.0;
    outcome.tracedLengthBaseVoxels = 9000.0;
    outcome.evaluatedThroughBaseVoxels = 10000.0;
    outcome.evaluationComplete = true;
    outcome.failures.resize(1);
    outcome.failures[0].reason = "distance_above_threshold";
    outcome.failures[0].directionalReferenceArcBaseVoxels = 3000.0;
    const std::array oneRestart{outcome};
    const auto single = vc::fiber_tracer::summarizeFiberReferenceReplay(1, oneRestart, 1.0);
    CHECK(single.meanSegmentLengthMillimeters == doctest::Approx(5.0));

    auto longer = outcome;
    longer.sourceIndex = 1;
    longer.reverse = true;
    longer.referenceLengthBaseVoxels = 20000.0;
    longer.tracedLengthBaseVoxels = 20000.0;
    longer.evaluatedThroughBaseVoxels = 20000.0;
    longer.failureFree = true;
    longer.failures.clear();
    const std::array unequalDirections{outcome, longer};
    const auto pooled = vc::fiber_tracer::summarizeFiberReferenceReplay(2, unequalDirections, 1.0);
    CHECK(pooled.directedCases == 2);
    CHECK(pooled.totalFailures == 1);
    CHECK(pooled.meanSegmentLengthMillimeters == doctest::Approx(15.0));
    CHECK(pooled.meanSegmentLengthPercent == doctest::Approx(50.0));

    outcome.failureFree = true;
    outcome.failures.clear();
    const std::array noFailures{outcome, longer};
    const auto successful = vc::fiber_tracer::summarizeFiberReferenceReplay(2, noFailures, 1.0);
    CHECK(successful.meanSegmentLengthMillimeters == doctest::Approx(30.0));
    CHECK(successful.meanSegmentLengthPercent == doctest::Approx(100.0));
    const nlohmann::json config{{"error_threshold_base_voxels", 2.0}};
    const auto json = vc::fiber_tracer::fiberReferenceReplayBenchmarkJson(
        successful, noFailures, "fiberlet", config, {}, {0, 0, 0}, {1, 1, 1});
    CHECK(json.at("summary").at("reliability_segments") == 1);
}

TEST_CASE("reference replay empty summary has no reliability partitions")
{
    const std::vector<vc::fiber_tracer::FiberReferenceReplayOutcome> outcomes;
    const auto summary = vc::fiber_tracer::summarizeFiberReferenceReplay(2, outcomes, 2.4);
    CHECK(summary.directedCases == 0);
    CHECK(summary.sourcesWithoutRuns == 2);
    CHECK(summary.meanSegmentLengthBaseVoxels == 0.0);
    CHECK(summary.meanSegmentLengthMillimeters == 0.0);
    CHECK(summary.meanSegmentLengthPercent == 0.0);
    const nlohmann::json config{{"error_threshold_base_voxels", 2.0}};
    const auto json = vc::fiber_tracer::fiberReferenceReplayBenchmarkJson(
        summary, outcomes, "fiberlet", config, {}, {0, 0, 0}, {1, 1, 1});
    CHECK(json.at("summary").at("reliability_segments") == 0);
}

TEST_CASE("reference replay JSON version five preserves failure diagnostics")
{
    vc::fiber_tracer::FiberReferenceReplayOutcome outcome;
    outcome.referenceLengthBaseVoxels = 10.0;
    outcome.tracedLengthBaseVoxels = 8.0;
    outcome.evaluatedThroughBaseVoxels = 10.0;
    outcome.seededSpans = 2;
    outcome.evaluationComplete = true;
    vc::fiber_tracer::FiberReferenceReplayFailure failure;
    failure.reason = "distance_above_threshold";
    failure.directionalReferenceArcBaseVoxels = 3.0;
    failure.directionalReferenceArcFraction = 0.3;
    failure.sourceReferenceArcBaseVoxels = 7.0;
    failure.sourceReferenceArcFraction = 0.7;
    failure.referencePointBaseXYZ = {1.0, 2.0, 3.0};
    failure.evaluatorPointBaseXYZ = cv::Vec3d{4.0, 5.0, 6.0};
    failure.thresholdMeasurement = vc::fiber_tracer::FiberReplayThresholdMeasurement{5.0, 3.0, 4.0, std::sqrt(10.0), std::sqrt(10.0) / 2.0, true};
    outcome.failures.push_back(failure);
    const std::array outcomes{outcome};
    const auto summary = vc::fiber_tracer::summarizeFiberReferenceReplay(1, outcomes, 2.4);
    const nlohmann::json commonConfig = {
        {"error_threshold_base_voxels", 2.0},
        {"match_refine_steps", 1.0},
    };
    const auto json = vc::fiber_tracer::fiberReferenceReplayBenchmarkJson(
        summary, outcomes, "fiberlet", commonConfig,
        {{"beam_width", 16}}, {0, 0, 0}, {10, 10, 10});

    CHECK(json.at("version") == 6);
    CHECK(json.at("metric_definition") == "corpus-failures-plus-one-v1");
    CHECK(json.at("tracer") == "fiberlet");
    CHECK(json.at("backend_config").at("beam_width") == 16);
    CHECK(json.at("summary").at("reliability_segments") == 2);
    CHECK(json.at("summary").at("mean_segment_length_mm") == doctest::Approx(0.012));
    CHECK_FALSE(json.at("summary").contains("mean_distance_per_failure_base_voxels"));
    CHECK_FALSE(json.at("summary").contains("mean_distance_per_failure_mm"));
    CHECK_FALSE(json.at("summary").contains("mean_distance_per_failure_percent"));
    CHECK_FALSE(json.at("summary").contains("zero_failure_convention_applied"));
    const auto& failureJson = json.at("cases").at(0).at("failures").at(0);
    CHECK(failureJson.at("source_reference_arc_base_voxels") == doctest::Approx(7.0));
    CHECK(failureJson.at("threshold_measurement").at("normal_error_base_voxels") == doctest::Approx(3.0));
}

TEST_CASE("direct replay outcomes use the common benchmark accounting")
{
    vc::fiber_tracer::FiberReferenceReplayCase replayCase;
    replayCase.referenceLengthBaseVoxels = 10.0;
    vc::fiber_tracer::FiberReplayTraceResult replay;
    replay.referenceEndArcBase = 10.0;
    replay.completedReferenceArcBase = 10.0;
    replay.segments.resize(1);
    replay.segments[0].endReferenceArcBase = 10.0;
    replay.segments[0].matches.push_back(
        {0, 0.0, 0.0, {}, 0.0, 1.0, {}});

    const auto outcome =
        vc::fiber_tracer::measureFiberReferenceReplayOutcome(
            replayCase, replay);
    CHECK(outcome.evaluationComplete);
    CHECK(outcome.failureFree);
    CHECK(outcome.seededSpans == 1);
    CHECK(outcome.tracedLengthBaseVoxels == doctest::Approx(10.0));
}

TEST_CASE("Lasagna replay transports a reference tangent on a planar normal field")
{
    ConstantNormalSampler normals({0.0, 0.0, 1.0});
    vc::fiber_tracer::LasagnaReplayTraceRequest request;
    request.referencePointsBase = {{0, 0, 0}, {40, 0, 0}};
    request.stepBaseVoxels = 4.0;
    request.errorThresholdBaseVoxels = 1.0;
    const auto replay = vc::fiber_tracer::traceLasagnaReplay(normals, request);

    CHECK(replay.completedReferenceArcBase == doctest::Approx(40.0));
    CHECK(replay.failures.empty());
    REQUIRE(replay.segments.size() == 1);
    CHECK(replay.segments.front().tracePointsBase.back()[0] ==
          doctest::Approx(40.0));
}

TEST_CASE("Lasagna replay retains direction across invalid normal samples")
{
    ConstantNormalSampler normals({0.0, 0.0, 0.0}, false);
    vc::fiber_tracer::LasagnaReplayTraceRequest request;
    request.referencePointsBase = {{0, 0, 0}, {20, 0, 0}};
    request.stepBaseVoxels = 4.0;
    request.errorThresholdBaseVoxels = 1.0;
    const auto replay = vc::fiber_tracer::traceLasagnaReplay(normals, request);

    CHECK(replay.completedReferenceArcBase == doctest::Approx(20.0));
    CHECK(replay.failures.empty());
}

TEST_CASE("Lasagna tangent transport resolves antipodal normal encodings")
{
    const cv::Vec3d direction{1.0, 0.0, 0.0};
    const auto same = vc::lasagna::transportDirectionToNormalPlane(
        direction, {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0});
    CHECK(same[0] == doctest::Approx(1.0));
    CHECK(same[1] == doctest::Approx(0.0));
    CHECK(same[2] == doctest::Approx(0.0));
}
