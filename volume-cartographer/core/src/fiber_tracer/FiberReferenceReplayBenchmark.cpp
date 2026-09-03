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
    outcome.completed = replay.failures.empty();
    if (outcome.completed) {
        outcome.tracedLengthBaseVoxels = replayCase.referenceLengthBaseVoxels;
    } else {
        const auto& failure = replay.failures.front();
        outcome.tracedLengthBaseVoxels = std::clamp(failure.referenceArcBase - replay.referenceBeginArcBase, 0.0, replayCase.referenceLengthBaseVoxels);
        outcome.failureReason = failure.reason;
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
    double failedLength = 0.0;
    for (const auto& outcome : outcomes) {
        if (!(outcome.referenceLengthBaseVoxels > 0.0) || !std::isfinite(outcome.referenceLengthBaseVoxels) ||
            !(outcome.tracedLengthBaseVoxels >= 0.0) || !std::isfinite(outcome.tracedLengthBaseVoxels) ||
            outcome.tracedLengthBaseVoxels > outcome.referenceLengthBaseVoxels + kEpsilon) {
            throw std::invalid_argument("reference replay outcome length is invalid");
        }
        const auto key = std::pair{outcome.sourceIndex, outcome.runIndex};
        const auto [found, inserted] = runs.emplace(key, outcome.referenceLengthBaseVoxels);
        if (!inserted && std::abs(found->second - outcome.referenceLengthBaseVoxels) > kEpsilon) {
            throw std::invalid_argument("reference replay run has inconsistent directed lengths");
        }
        summary.directedReferenceLengthBaseVoxels += outcome.referenceLengthBaseVoxels;
        summary.tracedLengthBaseVoxels += outcome.tracedLengthBaseVoxels;
        if (outcome.completed) {
            ++summary.completedCases;
        } else {
            ++summary.failedCases;
            failedLength += outcome.tracedLengthBaseVoxels;
            ++reasons[outcome.failureReason];
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
        summary.completedCasesPercent = 100.0 * static_cast<double>(summary.completedCases) / static_cast<double>(outcomes.size());
    }
    if (summary.failedCases > 0) {
        summary.meanFailureLengthBaseVoxels = failedLength / static_cast<double>(summary.failedCases);
    }
    if (summary.directedReferenceLengthBaseVoxels > 0.0) {
        summary.lengthWeightedSuccessPercent = 100.0 * summary.tracedLengthBaseVoxels / summary.directedReferenceLengthBaseVoxels;
    }
    const double baseToMillimeters = baseVoxelSizeUm / 1000.0;
    summary.tracedLengthMillimeters = summary.tracedLengthBaseVoxels * baseToMillimeters;
    summary.meanCreditedLengthMillimeters = summary.meanCreditedLengthBaseVoxels * baseToMillimeters;
    summary.meanFailureLengthMillimeters = summary.meanFailureLengthBaseVoxels * baseToMillimeters;
    summary.failureReasons.assign(reasons.begin(), reasons.end());
    return summary;
}

nlohmann::json fiberReferenceReplayBenchmarkJson(
    const FiberReferenceReplaySummary& summary,
    std::span<const FiberReferenceReplayOutcome> outcomes,
    const FiberletGraphReplayConfig& replayConfig,
    const cv::Vec3d& minimumBaseXYZ,
    const cv::Vec3d& maximumBaseXYZ)
{
    nlohmann::json cases = nlohmann::json::array();
    for (const auto& outcome : outcomes) {
        cases.push_back({
            {"source_index", outcome.sourceIndex},
            {"run_index", outcome.runIndex},
            {"direction", outcome.reverse ? "reverse" : "forward"},
            {"reference_length_base_voxels", outcome.referenceLengthBaseVoxels},
            {"traced_length_base_voxels", outcome.tracedLengthBaseVoxels},
            {"completed", outcome.completed},
            {"failure_reason", outcome.completed ? nlohmann::json(nullptr) : nlohmann::json(outcome.failureReason)},
        });
    }
    nlohmann::json reasons = nlohmann::json::object();
    for (const auto& [reason, count] : summary.failureReasons)
        reasons[reason] = count;
    return {
        {"format", "vc_fiber_reference_replay_benchmark"},
        {"version", 1},
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
         }},
        {"summary",
         {
             {"selected_sources", summary.selectedSources},
             {"sources_without_runs", summary.sourcesWithoutRuns},
             {"in_crop_runs", summary.inCropRuns},
             {"directed_cases", summary.directedCases},
             {"completed_cases", summary.completedCases},
             {"failed_cases", summary.failedCases},
             {"undirected_reference_length_base_voxels", summary.undirectedReferenceLengthBaseVoxels},
             {"directed_reference_length_base_voxels", summary.directedReferenceLengthBaseVoxels},
             {"traced_length_base_voxels", summary.tracedLengthBaseVoxels},
             {"mean_credited_length_base_voxels", summary.meanCreditedLengthBaseVoxels},
             {"mean_failure_length_base_voxels", summary.meanFailureLengthBaseVoxels},
             {"base_voxel_size_um", summary.baseVoxelSizeUm},
             {"traced_length_mm", summary.tracedLengthMillimeters},
             {"mean_credited_length_mm", summary.meanCreditedLengthMillimeters},
             {"mean_failure_length_mm", summary.meanFailureLengthMillimeters},
             {"length_weighted_success_percent", summary.lengthWeightedSuccessPercent},
             {"completed_cases_percent", summary.completedCasesPercent},
             {"failure_reasons", std::move(reasons)},
         }},
        {"cases", std::move(cases)},
    };
}

}  // namespace vc::fiber_tracer
