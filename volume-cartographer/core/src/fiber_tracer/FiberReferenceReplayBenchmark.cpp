#include "vc/fiber_tracer/FiberReferenceReplayBenchmark.hpp"

#include "vc/fiber_tracer/FiberReplayMetric.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-9;

double polylineLength(const std::vector<cv::Vec3d>& points)
{
    return makePolylineArcGeometry(points).length();
}

nlohmann::json pointJson(const cv::Vec3d& point)
{
    return nlohmann::json::array({point[0], point[1], point[2]});
}

nlohmann::json optionalPointJson(const std::optional<cv::Vec3d>& point)
{
    return point.has_value() ? pointJson(*point) : nlohmann::json(nullptr);
}

}  // namespace

std::vector<FiberReferenceReplayCase> makeFiberReferenceReplayCases(
    std::span<const std::vector<cv::Vec3d>> sourceFibersBaseXYZ, const cv::Vec3d& minimumBaseXYZ, const cv::Vec3d& maximumBaseXYZ)
{
    std::vector<FiberReferenceReplayCase> result;
    for (std::size_t source = 0; source < sourceFibersBaseXYZ.size(); ++source) {
        auto runs = clipPolylineToHalfOpenBox(sourceFibersBaseXYZ[source], minimumBaseXYZ, maximumBaseXYZ);
        for (std::size_t run = 0; run < runs.size(); ++run) {
            const double length = polylineLength(runs[run]);
            if (!(length > kEpsilon))
                continue;
            FiberReferenceReplayCase forward;
            forward.sourceIndex = source;
            forward.runIndex = run;
            forward.pointsBaseXYZ = runs[run];
            forward.referenceLengthBaseVoxels = length;
            result.push_back(forward);

            FiberReferenceReplayCase reverse = forward;
            reverse.reverse = true;
            std::reverse(reverse.pointsBaseXYZ.begin(), reverse.pointsBaseXYZ.end());
            result.push_back(std::move(reverse));
        }
    }
    return result;
}

FiberReferenceReplayOutcome measureFiberReferenceReplayOutcome(const FiberReferenceReplayCase& replayCase, const FiberletGraphReplayResult& replay)
{
    if (!(replayCase.referenceLengthBaseVoxels > 0.0) || !std::isfinite(replayCase.referenceLengthBaseVoxels)) {
        throw std::invalid_argument("reference replay case length must be positive and finite");
    }
    if (std::abs(replay.referenceBeginArcBase) > kEpsilon || std::abs(replay.referenceEndArcBase - replayCase.referenceLengthBaseVoxels) > kEpsilon) {
        throw std::invalid_argument("reference replay result interval does not match its case");
    }

    FiberReferenceReplayOutcome outcome;
    outcome.sourceIndex = replayCase.sourceIndex;
    outcome.runIndex = replayCase.runIndex;
    outcome.reverse = replayCase.reverse;
    outcome.referenceLengthBaseVoxels = replayCase.referenceLengthBaseVoxels;
    outcome.evaluatedThroughBaseVoxels =
        std::clamp(replay.completedReferenceArcBase - replay.referenceBeginArcBase, 0.0, replayCase.referenceLengthBaseVoxels);
    outcome.evaluationComplete = outcome.evaluatedThroughBaseVoxels >= replayCase.referenceLengthBaseVoxels - kEpsilon;
    outcome.failureFree = replay.failures.empty();

    std::vector<std::pair<double, double>> tracedIntervals;
    tracedIntervals.reserve(replay.segments.size());
    for (const auto& segment : replay.segments) {
        if (!segment.seedKey.has_value() || segment.matches.empty())
            continue;
        const double begin =
            std::clamp(segment.matches.front().searchBeginArcBase - replay.referenceBeginArcBase, 0.0, replayCase.referenceLengthBaseVoxels);
        const double end = std::clamp(segment.endReferenceArcBase - replay.referenceBeginArcBase, 0.0, replayCase.referenceLengthBaseVoxels);
        if (end > begin + kEpsilon)
            tracedIntervals.emplace_back(begin, end);
    }
    std::sort(tracedIntervals.begin(), tracedIntervals.end());
    outcome.seededSpans = tracedIntervals.size();
    for (std::size_t index = 0; index < tracedIntervals.size();) {
        double begin = tracedIntervals[index].first;
        double end = tracedIntervals[index].second;
        ++index;
        while (index < tracedIntervals.size() && tracedIntervals[index].first <= end + kEpsilon) {
            end = std::max(end, tracedIntervals[index].second);
            ++index;
        }
        outcome.tracedLengthBaseVoxels += end - begin;
    }

    outcome.failures.reserve(replay.failures.size());
    double previousDirectionalArc = -1.0;
    for (const auto& failure : replay.failures) {
        if (failure.index != outcome.failures.size() || failure.segmentIndex >= replay.segments.size())
            throw std::invalid_argument("reference replay failures are not in stable segment order");
        const auto& segment = replay.segments[failure.segmentIndex];
        const double begin = !segment.matches.empty() ? segment.matches.front().searchBeginArcBase : failure.referenceArcBase;
        const double directionalArc = std::clamp(failure.referenceArcBase - replay.referenceBeginArcBase, 0.0, replayCase.referenceLengthBaseVoxels);
        if (directionalArc + kEpsilon < previousDirectionalArc)
            throw std::invalid_argument("reference replay failures are not in directional arc order");
        previousDirectionalArc = directionalArc;
        const double sourceArc = replayCase.reverse ? replayCase.referenceLengthBaseVoxels - directionalArc : directionalArc;
        outcome.failures.push_back({
            failure.index,
            failure.segmentIndex,
            failure.reason,
            directionalArc,
            directionalArc / replayCase.referenceLengthBaseVoxels,
            sourceArc,
            sourceArc / replayCase.referenceLengthBaseVoxels,
            std::clamp(failure.referenceArcBase - begin, 0.0, replayCase.referenceLengthBaseVoxels),
            failure.referencePointBase,
            failure.evaluatorPointBase,
            failure.thresholdMeasurement,
        });
    }
    return outcome;
}

FiberReferenceReplaySummary summarizeFiberReferenceReplay(std::size_t selectedSources, std::span<const FiberReferenceReplayOutcome> outcomes, double baseVoxelSizeUm)
{
    if (!(baseVoxelSizeUm > 0.0) || !std::isfinite(baseVoxelSizeUm)) {
        throw std::invalid_argument("reference replay base voxel size must be positive and finite");
    }
    FiberReferenceReplaySummary summary;
    summary.selectedSources = selectedSources;
    summary.directedCases = outcomes.size();
    summary.baseVoxelSizeUm = baseVoxelSizeUm;
    std::map<std::pair<std::size_t, std::size_t>, double> runs;
    std::map<std::string, std::size_t> reasons;
    double failedSpanLength = 0.0;
    for (const auto& outcome : outcomes) {
        if (!(outcome.referenceLengthBaseVoxels > 0.0) || !std::isfinite(outcome.referenceLengthBaseVoxels) ||
            !(outcome.tracedLengthBaseVoxels >= 0.0) || !std::isfinite(outcome.tracedLengthBaseVoxels) ||
            outcome.tracedLengthBaseVoxels > outcome.referenceLengthBaseVoxels + kEpsilon) {
            throw std::invalid_argument("reference replay outcome length is invalid");
        }
        if (!outcome.evaluationComplete || outcome.evaluatedThroughBaseVoxels < outcome.referenceLengthBaseVoxels - kEpsilon)
            throw std::invalid_argument("reference replay benchmark requires complete directional evaluation");
        const auto key = std::pair{outcome.sourceIndex, outcome.runIndex};
        const auto [found, inserted] = runs.emplace(key, outcome.referenceLengthBaseVoxels);
        if (!inserted && std::abs(found->second - outcome.referenceLengthBaseVoxels) > kEpsilon) {
            throw std::invalid_argument("reference replay run has inconsistent directed lengths");
        }
        summary.directedReferenceLengthBaseVoxels += outcome.referenceLengthBaseVoxels;
        summary.tracedLengthBaseVoxels += outcome.tracedLengthBaseVoxels;
        summary.seededSpans += outcome.seededSpans;
        if (outcome.evaluationComplete) {
            ++summary.evaluatedCases;
        } else {
            ++summary.incompleteCases;
        }
        if (outcome.failureFree) {
            ++summary.failureFreeCases;
        } else {
            ++summary.casesWithFailures;
        }
        summary.totalFailures += outcome.failures.size();
        for (const auto& failure : outcome.failures) {
            failedSpanLength += failure.tracedSpanBaseVoxels;
            ++reasons[failure.reason];
        }
    }
    summary.inCropRuns = runs.size();
    for (const auto& [key, length] : runs) {
        (void)key;
        summary.undirectedReferenceLengthBaseVoxels += length;
    }
    std::vector<bool> sourceHasRun(selectedSources, false);
    for (const auto& [key, length] : runs) {
        (void)length;
        if (key.first >= selectedSources)
            throw std::invalid_argument("reference replay outcome source index is invalid");
        sourceHasRun[key.first] = true;
    }
    summary.sourcesWithoutRuns = static_cast<std::size_t>(std::count(sourceHasRun.begin(), sourceHasRun.end(), false));
    if (!outcomes.empty()) {
        summary.meanCreditedLengthBaseVoxels = summary.tracedLengthBaseVoxels / static_cast<double>(outcomes.size());
        summary.failureFreeCasesPercent = 100.0 * static_cast<double>(summary.failureFreeCases) / static_cast<double>(outcomes.size());
    }
    if (summary.seededSpans > 0)
        summary.meanSeededSpanLengthBaseVoxels = summary.tracedLengthBaseVoxels / static_cast<double>(summary.seededSpans);
    if (summary.totalFailures > 0)
        summary.meanFailedSpanLengthBaseVoxels = failedSpanLength / static_cast<double>(summary.totalFailures);
    if (summary.directedReferenceLengthBaseVoxels > 0.0) {
        summary.lengthWeightedSuccessPercent = 100.0 * summary.tracedLengthBaseVoxels / summary.directedReferenceLengthBaseVoxels;
    }
    const double baseToMillimeters = baseVoxelSizeUm / 1000.0;
    summary.tracedLengthMillimeters = summary.tracedLengthBaseVoxels * baseToMillimeters;
    summary.meanCreditedLengthMillimeters = summary.meanCreditedLengthBaseVoxels * baseToMillimeters;
    summary.meanSeededSpanLengthMillimeters = summary.meanSeededSpanLengthBaseVoxels * baseToMillimeters;
    summary.meanFailedSpanLengthMillimeters = summary.meanFailedSpanLengthBaseVoxels * baseToMillimeters;
    if (summary.directedReferenceLengthBaseVoxels > 0.0)
        summary.failuresPerDirectedMillimeter =
            static_cast<double>(summary.totalFailures) / (summary.directedReferenceLengthBaseVoxels * baseToMillimeters);
    if (summary.directedReferenceLengthBaseVoxels > 0.0) {
        const double divisor = static_cast<double>(std::max<std::size_t>(summary.totalFailures, 1));
        summary.meanDistancePerFailureBaseVoxels = summary.directedReferenceLengthBaseVoxels / divisor;
        summary.meanDistancePerFailureMillimeters = summary.meanDistancePerFailureBaseVoxels * baseToMillimeters;
        summary.meanDistancePerFailurePercent = 100.0 * summary.meanDistancePerFailureBaseVoxels / summary.directedReferenceLengthBaseVoxels;
    }
    summary.failureReasons.assign(reasons.begin(), reasons.end());
    return summary;
}

nlohmann::json fiberReferenceReplayBenchmarkJson(
    const FiberReferenceReplaySummary& summary,
    std::span<const FiberReferenceReplayOutcome> outcomes,
    const FiberletGraphReplayConfig& replayConfig,
    double seedWindowBaseVoxels,
    const cv::Vec3d& minimumBaseXYZ,
    const cv::Vec3d& maximumBaseXYZ)
{
    nlohmann::json cases = nlohmann::json::array();
    for (const auto& outcome : outcomes) {
        nlohmann::json failures = nlohmann::json::array();
        for (const auto& failure : outcome.failures) {
            failures.push_back({
                {"index", failure.index},
                {"segment_index", failure.segmentIndex},
                {"reason", failure.reason},
                {"directional_reference_arc_base_voxels", failure.directionalReferenceArcBaseVoxels},
                {"directional_reference_arc_fraction", failure.directionalReferenceArcFraction},
                {"source_reference_arc_base_voxels", failure.sourceReferenceArcBaseVoxels},
                {"source_reference_arc_fraction", failure.sourceReferenceArcFraction},
                {"traced_span_base_voxels", failure.tracedSpanBaseVoxels},
                {"reference_point_base_xyz", pointJson(failure.referencePointBaseXYZ)},
                {"evaluator_point_base_xyz", optionalPointJson(failure.evaluatorPointBaseXYZ)},
                {"threshold_measurement", fiberReplayOptionalThresholdMeasurementJson(failure.thresholdMeasurement, replayConfig.errorThresholdBaseVoxels)},
            });
        }
        cases.push_back({
            {"source_index", outcome.sourceIndex},
            {"run_index", outcome.runIndex},
            {"direction", outcome.reverse ? "reverse" : "forward"},
            {"reference_length_base_voxels", outcome.referenceLengthBaseVoxels},
            {"traced_length_base_voxels", outcome.tracedLengthBaseVoxels},
            {"evaluated_through_base_voxels", outcome.evaluatedThroughBaseVoxels},
            {"seeded_spans", outcome.seededSpans},
            {"evaluation_complete", outcome.evaluationComplete},
            {"failure_free", outcome.failureFree},
            {"failures", std::move(failures)},
        });
    }
    nlohmann::json reasons = nlohmann::json::object();
    for (const auto& [reason, count] : summary.failureReasons)
        reasons[reason] = count;
    return {
        {"format", "vc_fiber_reference_replay_benchmark"},
        {"version", 2},
        {"coordinates",
         {
             {"order", "XYZ"},
             {"space", "base_volume"},
             {"crop_half_open", true},
             {"minimum_base_xyz", pointJson(minimumBaseXYZ)},
             {"maximum_base_xyz", pointJson(maximumBaseXYZ)},
         }},
        {"threshold", fiberReplayThresholdDescriptorJson(replayConfig.errorThresholdBaseVoxels)},
        {"config",
         {
             {"beam_width", replayConfig.beamWidth},
             {"beam_step_distance_base_voxels", replayConfig.beamStepDistanceBaseVoxels},
             {"lookahead_distance_base_voxels", replayConfig.lookaheadDistanceBaseVoxels},
             {"maximum_generated_states_per_iteration", replayConfig.maximumGeneratedStatesPerIteration},
             {"match_refine_steps", replayConfig.matchRefineSteps},
             {"require_initial_seed_in_first_window", replayConfig.requireInitialSeedInFirstWindow},
             {"stop_at_first_failure", replayConfig.stopAtFirstFailure},
             {"minimum_reset_advance_base_voxels", replayConfig.minimumResetAdvanceBaseVoxels},
             {"seed_window_base_voxels", seedWindowBaseVoxels},
             {"post_failure_seed_scan", "successive seed windows through remaining reference"},
         }},
        {"summary",
         {
             {"selected_sources", summary.selectedSources},
             {"sources_without_runs", summary.sourcesWithoutRuns},
             {"in_crop_runs", summary.inCropRuns},
             {"directed_cases", summary.directedCases},
             {"evaluated_cases", summary.evaluatedCases},
             {"incomplete_cases", summary.incompleteCases},
             {"failure_free_cases", summary.failureFreeCases},
             {"cases_with_failures", summary.casesWithFailures},
             {"total_failures", summary.totalFailures},
             {"seeded_spans", summary.seededSpans},
             {"undirected_reference_length_base_voxels", summary.undirectedReferenceLengthBaseVoxels},
             {"directed_reference_length_base_voxels", summary.directedReferenceLengthBaseVoxels},
             {"traced_length_base_voxels", summary.tracedLengthBaseVoxels},
             {"mean_credited_length_base_voxels", summary.meanCreditedLengthBaseVoxels},
             {"mean_seeded_span_length_base_voxels", summary.meanSeededSpanLengthBaseVoxels},
             {"mean_failed_span_length_base_voxels", summary.meanFailedSpanLengthBaseVoxels},
             {"base_voxel_size_um", summary.baseVoxelSizeUm},
             {"traced_length_mm", summary.tracedLengthMillimeters},
             {"mean_credited_length_mm", summary.meanCreditedLengthMillimeters},
             {"mean_seeded_span_length_mm", summary.meanSeededSpanLengthMillimeters},
             {"mean_failed_span_length_mm", summary.meanFailedSpanLengthMillimeters},
             {"length_weighted_success_percent", summary.lengthWeightedSuccessPercent},
             {"failure_free_cases_percent", summary.failureFreeCasesPercent},
             {"failures_per_directed_mm", summary.failuresPerDirectedMillimeter},
             {"mean_distance_per_failure_base_voxels", summary.meanDistancePerFailureBaseVoxels},
             {"mean_distance_per_failure_mm", summary.meanDistancePerFailureMillimeters},
             {"mean_distance_per_failure_percent", summary.meanDistancePerFailurePercent},
             {"zero_failure_convention_applied", summary.totalFailures == 0},
             {"failure_reasons", std::move(reasons)},
         }},
        {"cases", std::move(cases)},
    };
}

}  // namespace vc::fiber_tracer
