#include "vc/fiber_tracer/FiberTraceConsensus.hpp"

#include "vc/core/io/PolylineObj.hpp"
#include "vc/fiber_tracer/FiberTraceSeed.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

namespace vc::fiber_tracer
{
namespace
{

std::size_t labelIndex(FiberTraceConsensusLabel label)
{
    return static_cast<std::size_t>(label);
}

bool isActive(FiberTraceConsensusLabel label)
{
    return label == FiberTraceConsensusLabel::H || label == FiberTraceConsensusLabel::V;
}

struct Evidence {
    std::size_t neighbor = 0;
    std::size_t pieceA = 0;
    std::size_t pieceB = 0;
    std::size_t constraintIndex = 0;
};

bool isSnapshotMilestone(std::size_t addedCount)
{
    return addedCount > 0 && ((addedCount <= 100 && addedCount % 10 == 0) || (addedCount > 100 && addedCount % 100 == 0));
}

FiberTraceConsensusObjPaths outputPaths(const std::filesystem::path& outputBase, const std::string& suffix)
{
    auto base = outputBase;
    if (base.has_extension())
        base.replace_extension();
    const auto directory = base.parent_path();
    const std::string stem = base.filename().string();
    if (stem.empty())
        throw std::invalid_argument("Consensus OBJ output basename is empty");
    return {
        directory / (stem + suffix + "_h.obj"),
        directory / (stem + suffix + "_v.obj"),
        directory / (stem + suffix + "_broken.obj"),
    };
}

std::array<std::size_t, 3> writeLabels(
    const std::vector<FiberletCropTraceLine>& traces,
    const std::vector<FiberTraceConsensusLabel>& labels,
    const FiberTraceConsensusObjPaths& paths,
    const std::string& commentSuffix)
{
    std::vector<vc::core::io::NamedPolyline> h;
    std::vector<vc::core::io::NamedPolyline> v;
    std::vector<vc::core::io::NamedPolyline> broken;
    for (std::size_t trace = 0; trace < traces.size(); ++trace) {
        auto* output = labels[trace] == FiberTraceConsensusLabel::H
            ? &h
            : labels[trace] == FiberTraceConsensusLabel::V
                ? &v
                : labels[trace] == FiberTraceConsensusLabel::Broken
                    ? &broken
                    : nullptr;
        if (output != nullptr) {
            output->push_back({
                "trace_" + std::to_string(trace),
                traces[trace].pointsBaseXYZ,
            });
        }
    }
    if (!paths.h.parent_path().empty())
        std::filesystem::create_directories(paths.h.parent_path());
    vc::core::io::writePolylinesObj(h, paths.h, "VC3D consensus H fibers " + commentSuffix);
    vc::core::io::writePolylinesObj(v, paths.v, "VC3D consensus V fibers " + commentSuffix);
    vc::core::io::writePolylinesObj(
        broken, paths.broken, "VC3D consensus broken fibers " + commentSuffix);
    return {h.size(), v.size(), broken.size()};
}

}  // namespace

FiberTraceConsensusReport growFiberTraceConsensus(
    const std::vector<FiberletCropTraceLine>& traces, const FiberTraceConstraintReport& constraints, const FiberTraceConsensusConfig& config)
{
    if (!std::isfinite(config.brokenCostPerConstraint) || config.brokenCostPerConstraint < 0.0) {
        throw std::invalid_argument("Consensus broken cost per constraint must be finite and nonnegative");
    }
    const auto geometryReport = measureFiberTraceSeedGeometry(
        traces,
        config.cropMinimumBaseXYZ,
        config.cropMaximumBaseXYZ);
    const auto& geometry = geometryReport.traces;

    FiberTraceConsensusReport report;
    report.labels.assign(traces.size(), FiberTraceConsensusLabel::Unassigned);
    std::size_t remaining = 0;
    for (std::size_t trace = 0; trace < traces.size(); ++trace) {
        if (geometry[trace].valid) {
            ++remaining;
        } else {
            report.labels[trace] = FiberTraceConsensusLabel::Broken;
            ++report.degenerateTraces;
        }
    }
    if (remaining == 0) {
        throw std::invalid_argument(
            "Consensus has no primary seed longer than half the nominal crop size");
    }

    std::vector<std::vector<Evidence>> adjacency(traces.size());
    std::vector<std::size_t> componentByTrace(traces.size(), std::numeric_limits<std::size_t>::max());
    for (std::size_t index = 0; index < constraints.constraints.size(); ++index) {
        const auto& constraint = constraints.constraints[index];
        if (constraint.pieceA >= constraints.pieces.size() || constraint.pieceB >= constraints.pieces.size()) {
            throw std::invalid_argument("Consensus constraint references an invalid piece");
        }
        const std::size_t traceA = constraints.pieces[constraint.pieceA].traceIndex;
        const std::size_t traceB = constraints.pieces[constraint.pieceB].traceIndex;
        if (traceA >= traces.size() || traceB >= traces.size())
            throw std::invalid_argument("Consensus piece references an invalid trace");
        if (!std::isfinite(constraint.closestDistanceBaseVoxels) || constraint.closestDistanceBaseVoxels < 0.0 ||
            !std::isfinite(constraint.parallelScore) || constraint.parallelScore < 0.0 || constraint.parallelScore > 1.0) {
            throw std::invalid_argument("Consensus constraint has invalid evidence");
        }
        if (traceA == traceB)
            continue;
        adjacency[traceA].push_back({traceB, constraint.pieceA, constraint.pieceB, index});
        adjacency[traceB].push_back({traceA, constraint.pieceA, constraint.pieceB, index});
        ++report.retainedCrossTraceConstraints;
    }
    for (auto& evidence : adjacency) {
        std::sort(evidence.begin(), evidence.end(), [](const Evidence& left, const Evidence& right) {
            return std::tie(left.neighbor, left.pieceA, left.pieceB, left.constraintIndex) <
                   std::tie(right.neighbor, right.pieceA, right.pieceB, right.constraintIndex);
        });
    }

    const auto seed = [&](bool requirePrimaryLength) -> std::size_t {
        std::vector<unsigned char> eligible(traces.size(), 0);
        for (std::size_t trace = 0; trace < traces.size(); ++trace) {
            if (report.labels[trace] != FiberTraceConsensusLabel::Unassigned)
                continue;
            eligible[trace] = 1;
        }
        return selectCentralStraightFiberTrace(
                   geometryReport, eligible, requirePrimaryLength)
            .value_or(traces.size());
    };

    struct Candidate {
        std::size_t trace = 0;
        std::size_t count = 0;
        double meanDistance = 0.0;
        double score = 0.0;
        double hCost = 0.0;
        double vCost = 0.0;
        std::size_t componentIndex = std::numeric_limits<std::size_t>::max();
    };
    const auto candidate = [&](std::size_t trace) -> std::optional<Candidate> {
        Candidate result;
        result.trace = trace;
        double distanceSum = 0.0;
        for (const auto& evidence : adjacency[trace]) {
            const auto neighborLabel = report.labels[evidence.neighbor];
            if (!isActive(neighborLabel))
                continue;
            const auto& constraint = constraints.constraints[evidence.constraintIndex];
            ++result.count;
            result.componentIndex = std::min(result.componentIndex, componentByTrace[evidence.neighbor]);
            distanceSum += constraint.closestDistanceBaseVoxels;
            const bool neighborH = neighborLabel == FiberTraceConsensusLabel::H;
            result.hCost += neighborH ? 1.0 - constraint.parallelScore : constraint.parallelScore;
            result.vCost += neighborH ? constraint.parallelScore : 1.0 - constraint.parallelScore;
        }
        if (result.count == 0)
            return std::nullopt;
        result.meanDistance = distanceSum / static_cast<double>(result.count);
        result.score = result.meanDistance == 0.0 ? std::numeric_limits<double>::infinity() : static_cast<double>(result.count) / result.meanDistance;
        return result;
    };

    while (remaining > 0) {
        std::optional<Candidate> best;
        for (std::size_t trace = 0; trace < traces.size(); ++trace) {
            if (report.labels[trace] != FiberTraceConsensusLabel::Unassigned)
                continue;
            const auto current = candidate(trace);
            if (!current)
                continue;
            if (!best || current->score > best->score ||
                (current->score == best->score &&
                 (current->count > best->count ||
                  (current->count == best->count && (current->meanDistance < best->meanDistance ||
                                                     (current->meanDistance == best->meanDistance && current->trace < best->trace)))))) {
                best = current;
            }
        }

        FiberTraceConsensusStep step;
        if (!best) {
            const bool primary = report.components == 0;
            step.traceIndex = seed(primary);
            if (step.traceIndex == traces.size()) {
                if (primary) {
                    throw std::invalid_argument(
                        "Consensus has no primary seed longer than half the nominal crop size");
                }
                throw std::logic_error("Consensus has no selectable trace");
            }
            step.componentSeed = true;
            step.componentIndex = report.components++;
            step.label = FiberTraceConsensusLabel::H;
            step.seedStraightness = geometry[step.traceIndex].straightness;
            step.seedCenterDistanceBaseVoxels =
                geometry[step.traceIndex].centerDistanceBaseVoxels;
            step.seedArcLengthBaseVoxels =
                geometry[step.traceIndex].arcLengthBaseVoxels;
        } else {
            step.traceIndex = best->trace;
            step.componentIndex = best->componentIndex;
            step.evidenceCount = best->count;
            step.meanDistanceBaseVoxels = best->meanDistance;
            step.connectivityScore = best->score;
            step.hCost = best->hCost;
            step.vCost = best->vCost;
            step.brokenCost = config.brokenCostPerConstraint * static_cast<double>(best->count);
            step.label = FiberTraceConsensusLabel::H;
            step.selectedCost = step.hCost;
            if (step.vCost < step.selectedCost) {
                step.label = FiberTraceConsensusLabel::V;
                step.selectedCost = step.vCost;
            }
            if (step.brokenCost < step.selectedCost) {
                step.label = FiberTraceConsensusLabel::Broken;
                step.selectedCost = step.brokenCost;
            }
            if (step.label == FiberTraceConsensusLabel::Broken)
                report.brokenCost += step.selectedCost;
            else
                report.orientationCost += step.selectedCost;
        }
        report.labels[step.traceIndex] = step.label;
        componentByTrace[step.traceIndex] = step.componentIndex;
        --remaining;
        step.addedCount = report.steps.size() + 1;
        report.steps.push_back(step);
        if (isSnapshotMilestone(step.addedCount))
            report.snapshotAddedCounts.push_back(step.addedCount);
    }

    for (const auto label : report.labels)
        ++report.labelCounts[labelIndex(label)];
    report.objective = report.orientationCost + report.brokenCost;
    return report;
}

FiberTraceConsensusObjReport writeFiberTraceConsensusObjs(
    const std::vector<FiberletCropTraceLine>& traces, const FiberTraceConsensusReport& consensus, const std::filesystem::path& outputBase)
{
    if (consensus.labels.size() != traces.size())
        throw std::invalid_argument("Consensus label count does not match traces");
    FiberTraceConsensusObjReport result;
    result.finalPaths = outputPaths(outputBase, "");
    auto finalLabels = consensus.labels;
    std::vector<bool> assigned(traces.size(), false);
    for (const auto& step : consensus.steps) {
        if (step.traceIndex >= traces.size() || assigned[step.traceIndex])
            throw std::invalid_argument("Consensus step trace index is invalid or repeated");
        assigned[step.traceIndex] = true;
    }
    for (std::size_t trace = 0; trace < traces.size(); ++trace) {
        if (finalLabels[trace] == FiberTraceConsensusLabel::Broken &&
            !assigned[trace]) {
            finalLabels[trace] = FiberTraceConsensusLabel::Unassigned;
        }
    }
    const auto finalCounts = writeLabels(
        traces, finalLabels, result.finalPaths, "final");
    result.hCount = finalCounts[0];
    result.vCount = finalCounts[1];
    result.brokenCount = finalCounts[2];

    std::vector<FiberTraceConsensusLabel> labels(traces.size(), FiberTraceConsensusLabel::Unassigned);
    std::size_t stepIndex = 0;
    for (const std::size_t milestone : consensus.snapshotAddedCounts) {
        if (milestone == 0 || milestone > consensus.steps.size() || milestone < stepIndex) {
            throw std::invalid_argument("Consensus snapshot milestone is invalid");
        }
        while (stepIndex < milestone) {
            const auto& step = consensus.steps[stepIndex++];
            labels[step.traceIndex] = step.label;
        }
        FiberTraceConsensusSnapshotObjReport snapshot;
        snapshot.addedCount = milestone;
        snapshot.paths = outputPaths(outputBase, "_step_" + std::to_string(milestone));
        const auto counts = writeLabels(
            traces,
            labels,
            snapshot.paths,
            "after " + std::to_string(milestone) + " assignments");
        snapshot.hCount = counts[0];
        snapshot.vCount = counts[1];
        snapshot.brokenCount = counts[2];
        result.snapshots.push_back(std::move(snapshot));
    }
    return result;
}

}  // namespace vc::fiber_tracer
