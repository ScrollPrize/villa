#include "vc/fiber_tracer/FiberTraceWindingOrderedCuts.hpp"

#include <ceres/ceres.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kPhase = 0.5;

double phase(FiberTraceFixedOrientation orientation)
{
    switch (orientation) {
    case FiberTraceFixedOrientation::Horizontal:
        return 0.0;
    case FiberTraceFixedOrientation::Vertical:
        return kPhase;
    case FiberTraceFixedOrientation::Mixed:
        return 0.0;
    }
    throw std::invalid_argument("Ordered-cut orientation is invalid");
}

double checkedSqrt(double value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(
            std::string{name} + " must be finite and nonnegative");
    }
    return std::sqrt(value);
}

struct SignResidual {
    double phaseDelta = 0.0;
    double sign = 1.0;
    double margin = 0.5;
    double signCoefficient = 1.0;
    double targetCoefficient = 1.0;

    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residuals) const
    {
        const T signedDistance = T(sign) *
            (b[0] - a[0] + T(phaseDelta));
        const T slack = T(margin) - signedDistance;
        residuals[0] = slack > T(0.0)
            ? T(signCoefficient) * slack
            : T(0.0);
        residuals[1] = T(targetCoefficient) *
            (signedDistance - T(margin));
        return true;
    }
};

struct ContinuationResidual {
    double coefficient = 1.0;

    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const
    {
        residual[0] = T(coefficient) * (b[0] - a[0]);
        return true;
    }
};

class ProgressCallback final : public ceres::IterationCallback {
public:
    ProgressCallback(
        std::size_t maximumIterations,
        FiberTraceWindingOrderedCutsProgressCallback callback,
        std::chrono::steady_clock::time_point started)
        : maximumIterations_{maximumIterations},
          callback_{std::move(callback)},
          started_{started}
    {
    }

    ceres::CallbackReturnType operator()(
        const ceres::IterationSummary& summary) override
    {
        if (callback_) {
            callback_({
                FiberTraceWindingOrderedCutsProgressPhase::Iterating,
                static_cast<std::size_t>(summary.iteration + 1),
                maximumIterations_,
                summary.cost,
                summary.gradient_max_norm,
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started_).count(),
            });
        }
        return ceres::SOLVER_CONTINUE;
    }

private:
    std::size_t maximumIterations_;
    FiberTraceWindingOrderedCutsProgressCallback callback_;
    std::chrono::steady_clock::time_point started_;
};

struct DisjointSet {
    explicit DisjointSet(std::size_t size) : parent(size), rank(size, 0)
    {
        std::iota(parent.begin(), parent.end(), std::size_t{0});
    }

    std::size_t find(std::size_t value)
    {
        if (parent[value] != value)
            parent[value] = find(parent[value]);
        return parent[value];
    }

    void unite(std::size_t a, std::size_t b)
    {
        a = find(a);
        b = find(b);
        if (a == b)
            return;
        if (rank[a] < rank[b])
            std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b])
            ++rank[a];
    }

    std::vector<std::size_t> parent;
    std::vector<unsigned char> rank;
};

FiberTraceWindingOrderedCutsCostSummary evaluateCosts(
    const FiberTraceWindingOrderedOffsetReport& report,
    const FiberTraceWindingOrderedCutsConfig& config)
{
    FiberTraceWindingOrderedCutsCostSummary result;
    for (const auto& factor : report.signFactors) {
        const double delta =
            report.offsetByPiece[factor.pieceB] +
                phase(report.orientationByPiece[factor.pieceB]) -
            report.offsetByPiece[factor.pieceA] -
                phase(report.orientationByPiece[factor.pieceA]);
        const double signedDistance = factor.sign * delta;
        const double slack = std::max(0.0, factor.margin - signedDistance);
        result.signMargin += config.signMarginWeight * factor.confidence *
            slack * slack;
        const double target = signedDistance - factor.margin;
        result.targetDistance += factor.confidence * target * target;
    }
    for (const auto [a, b] : report.continuationEdges) {
        const double delta =
            report.offsetByPiece[b] - report.offsetByPiece[a];
        result.continuation += config.continuationWeight * delta * delta;
    }
    return result;
}

bool infringed(
    const FiberTraceWindingOrderedSignFactor& factor,
    std::span<const int> windingByRun,
    std::span<const std::size_t> runByPiece,
    std::span<const FiberTraceFixedOrientation> orientations)
{
    const double delta =
        static_cast<double>(windingByRun[runByPiece[factor.pieceB]]) +
            phase(orientations[factor.pieceB]) -
        static_cast<double>(windingByRun[runByPiece[factor.pieceA]]) -
            phase(orientations[factor.pieceA]);
    return factor.sign * delta <= 0.0;
}

bool continuouslyInfringed(
    const FiberTraceWindingOrderedSignFactor& factor,
    const FiberTraceWindingOrderedOffsetReport& ordering)
{
    const double delta =
        ordering.offsetByPiece[factor.pieceB] +
            phase(ordering.orientationByPiece[factor.pieceB]) -
        ordering.offsetByPiece[factor.pieceA] -
            phase(ordering.orientationByPiece[factor.pieceA]);
    return factor.sign * delta <= 0.0;
}

int compareFractions(
    std::size_t leftNumerator,
    std::size_t leftDenominator,
    std::size_t rightNumerator,
    std::size_t rightDenominator)
{
    if (leftDenominator == 0 || rightDenominator == 0)
        throw std::invalid_argument("Cannot compare a fraction with zero denominator");
    bool reverse = false;
    while (true) {
        const std::size_t leftQuotient = leftNumerator / leftDenominator;
        const std::size_t rightQuotient = rightNumerator / rightDenominator;
        if (leftQuotient != rightQuotient) {
            const int comparison = leftQuotient < rightQuotient ? -1 : 1;
            return reverse ? -comparison : comparison;
        }
        const std::size_t leftRemainder = leftNumerator % leftDenominator;
        const std::size_t rightRemainder = rightNumerator % rightDenominator;
        if (leftRemainder == 0 || rightRemainder == 0) {
            if (leftRemainder == 0 && rightRemainder == 0)
                return 0;
            const int comparison = leftRemainder == 0 ? -1 : 1;
            return reverse ? -comparison : comparison;
        }
        leftNumerator = leftDenominator;
        leftDenominator = leftRemainder;
        rightNumerator = rightDenominator;
        rightDenominator = rightRemainder;
        reverse = !reverse;
    }
}

FiberTraceWindingOrderedCutStep makeStep(
    const FiberTraceWindingOrderedOffsetReport& ordering,
    std::span<const int> windingByRun,
    std::span<const std::size_t> runByPiece,
    std::size_t splits,
    std::optional<double> threshold)
{
    FiberTraceWindingOrderedCutStep result;
    result.splits = splits;
    result.signFactors = ordering.signFactors.size();
    result.threshold = threshold;
    result.windingByPiece.assign(ordering.offsetByPiece.size(), 0);
    int minimum = std::numeric_limits<int>::max();
    int maximum = std::numeric_limits<int>::lowest();
    for (std::size_t piece = 0; piece < result.windingByPiece.size(); ++piece) {
        if (ordering.activeByPiece[piece] == 0)
            continue;
        const int winding = windingByRun[runByPiece[piece]];
        result.windingByPiece[piece] = winding;
        minimum = std::min(minimum, winding);
        maximum = std::max(maximum, winding);
    }
    if (minimum <= maximum) {
        for (std::size_t piece = 0; piece < result.windingByPiece.size(); ++piece) {
            if (ordering.activeByPiece[piece] != 0)
                result.windingByPiece[piece] -= minimum;
        }
        result.windings = static_cast<std::size_t>(maximum - minimum + 1);
    } else {
        result.windings = 0;
    }
    for (const auto& factor : ordering.signFactors) {
        result.signInfringements += infringed(
            factor,
            windingByRun,
            runByPiece,
            ordering.orientationByPiece)
            ? 1
            : 0;
    }
    for (const auto [a, b] : ordering.continuationEdges) {
        result.continuationCuts +=
            result.windingByPiece[a] != result.windingByPiece[b] ? 1 : 0;
    }
    return result;
}

}  // namespace

FiberTraceWindingOrderedViolationSummary
summarizeFiberTraceWindingOrderedViolations(
    const FiberTraceWindingOrderedOffsetReport& ordering,
    const FiberTraceConstraintReport& constraints,
    std::span<const unsigned char> includedTraces)
{
    const std::size_t pieceCount = constraints.pieces.size();
    if (ordering.offsetByPiece.size() != pieceCount ||
        ordering.orientationByPiece.size() != pieceCount ||
        ordering.activeByPiece.size() != pieceCount) {
        throw std::invalid_argument(
            "Ordered violation inputs do not match represented pieces");
    }
    std::size_t traceCount = constraints.inputTraces;
    for (const auto& piece : constraints.pieces)
        traceCount = std::max(traceCount, piece.traceIndex + 1);
    if (!includedTraces.empty() && includedTraces.size() != traceCount) {
        throw std::invalid_argument(
            "Ordered violation trace mask has the wrong size");
    }
    const auto included = [&](const std::size_t trace) {
        return includedTraces.empty() || includedTraces[trace] != 0;
    };

    FiberTraceWindingOrderedViolationSummary result;
    result.traces.resize(traceCount);
    for (std::size_t trace = 0; trace < traceCount; ++trace)
        result.traces[trace].traceIndex = trace;
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        const std::size_t trace = constraints.pieces[piece].traceIndex;
        if (ordering.activeByPiece[piece] != 0 && included(trace))
            ++result.traces[trace].pieces;
    }
    for (const auto& factor : ordering.signFactors) {
        if (factor.pieceA >= pieceCount || factor.pieceB >= pieceCount) {
            throw std::invalid_argument(
                "Ordered sign factor references an invalid piece");
        }
        const std::size_t traceA =
            constraints.pieces[factor.pieceA].traceIndex;
        const std::size_t traceB =
            constraints.pieces[factor.pieceB].traceIndex;
        if (!included(traceA) || !included(traceB))
            continue;
        const bool violation = continuouslyInfringed(factor, ordering);
        ++result.factors;
        result.infringements += violation ? 1 : 0;
        ++result.traces[traceA].incidentFactors;
        result.traces[traceA].violatedFactors += violation ? 1 : 0;
        if (traceB != traceA) {
            ++result.traces[traceB].incidentFactors;
            result.traces[traceB].violatedFactors += violation ? 1 : 0;
        }
    }
    for (const auto& trace : result.traces) {
        if (trace.incidentFactors == 0 || trace.violatedFactors == 0)
            continue;
        if (!result.worstTrace) {
            result.worstTrace = trace.traceIndex;
            continue;
        }
        const auto& incumbent = result.traces[*result.worstTrace];
        const int percentage = compareFractions(
            trace.violatedFactors,
            trace.incidentFactors,
            incumbent.violatedFactors,
            incumbent.incidentFactors);
        if (percentage > 0 ||
            (percentage == 0 &&
             (trace.violatedFactors > incumbent.violatedFactors ||
              (trace.violatedFactors == incumbent.violatedFactors &&
               trace.traceIndex < incumbent.traceIndex)))) {
            result.worstTrace = trace.traceIndex;
        }
    }
    return result;
}

FiberTraceWindingOrderedOffsetReport fitFiberTraceWindingOrderedOffsets(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    std::span<const FiberTraceFixedOrientation> fixedOrientations,
    const FiberTraceWindingOrderedCutsConfig& config,
    std::span<const FiberTraceWindingOrderedCutsFixedOffset> fixedOffsets,
    const FiberTraceWindingOrderedCutsProgressCallback& progress,
    std::span<const unsigned char> activePieces)
{
    const auto started = std::chrono::steady_clock::now();
    const std::size_t pieceCount = constraints.pieces.size();
    if (pieceCount == 0 || fixedOrientations.size() != pieceCount ||
        topology.pieceLines.size() != pieceCount ||
        (!fixedOffsets.empty() && fixedOffsets.size() != pieceCount) ||
        (!activePieces.empty() && activePieces.size() != pieceCount)) {
        throw std::invalid_argument(
            "Ordered-cut winding inputs do not match represented pieces");
    }
    if (!std::isfinite(config.signMarginWeight) ||
        config.signMarginWeight < 0.0 ||
        !std::isfinite(config.continuationWeight) ||
        config.continuationWeight < 0.0 ||
        !std::isfinite(config.measurementScale) ||
        !(config.measurementScale > 0.0) || config.maximumIterations == 0 ||
        config.parallelWorkers == 0) {
        throw std::invalid_argument("Ordered-cut winding config is invalid");
    }
    if (progress) {
        progress({
            FiberTraceWindingOrderedCutsProgressPhase::Preparing,
            0,
            config.maximumIterations,
            0.0,
            0.0,
            0.0,
        });
    }

    const auto prepared = prepareFiberTraceWindingModel(
        constraints,
        topology,
        config,
        fixedOrientations,
        true,
        config.measurementScale);
    FiberTraceWindingOrderedOffsetReport report;
    report.offsetByPiece.assign(pieceCount, 0.0);
    report.orientationByPiece.assign(
        fixedOrientations.begin(), fixedOrientations.end());
    report.activeByPiece.resize(pieceCount, 0);
    report.factorDiagnostics = prepared.diagnostics;
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        report.activeByPiece[piece] =
            fixedOrientations[piece] != FiberTraceFixedOrientation::Mixed &&
                (activePieces.empty() || activePieces[piece] != 0)
            ? 1
            : 0;
        if (!fixedOffsets.empty() && fixedOffsets[piece].fixed) {
            if (!std::isfinite(fixedOffsets[piece].offset)) {
                throw std::invalid_argument(
                    "Ordered-cut fixed offset must be finite");
            }
            report.offsetByPiece[piece] = fixedOffsets[piece].offset;
        }
    }

    for (const auto& edge : prepared.edges) {
        if (report.activeByPiece[edge.a] == 0 ||
            report.activeByPiece[edge.b] == 0) {
            continue;
        }
        if (edge.containsHardContinuity)
            report.continuationEdges.emplace_back(edge.a, edge.b);
        for (const auto& measurement : edge.measurements) {
            if (measurement.continuity ||
                !(measurement.selectedConfidence > 0.0)) {
                continue;
            }
            const bool parallel = measurement.parallelDominant;
            const bool signPresent = parallel
                ? measurement.parallelSignPresent
                : measurement.perpendicularSignPresent;
            const auto target = parallel
                ? measurement.parallelSignedDelta
                : measurement.perpendicularSignedDelta;
            if (!signPresent || !target || *target == 0.0)
                continue;
            report.signFactors.push_back({
                measurement.constraintIndex,
                edge.a,
                edge.b,
                std::copysign(1.0, *target),
                parallel ? 1.0 : 0.5,
                measurement.selectedConfidence,
                parallel,
            });
        }
    }

    std::vector<std::vector<std::size_t>> adjacency(pieceCount);
    for (const auto& factor : report.signFactors) {
        adjacency[factor.pieceA].push_back(factor.pieceB);
        adjacency[factor.pieceB].push_back(factor.pieceA);
    }
    for (const auto [a, b] : report.continuationEdges) {
        adjacency[a].push_back(b);
        adjacency[b].push_back(a);
    }
    report.componentByPiece.assign(pieceCount, 0);
    std::vector<unsigned char> visited(pieceCount, 0);
    for (std::size_t startPiece = 0; startPiece < pieceCount; ++startPiece) {
        if (report.activeByPiece[startPiece] == 0 || visited[startPiece] != 0)
            continue;
        const std::size_t component = report.gaugePieces.size();
        std::vector<std::size_t> componentPieces;
        std::queue<std::size_t> pending;
        pending.push(startPiece);
        visited[startPiece] = 1;
        while (!pending.empty()) {
            const std::size_t piece = pending.front();
            pending.pop();
            componentPieces.push_back(piece);
            report.componentByPiece[piece] = component;
            for (const std::size_t neighbor : adjacency[piece]) {
                if (visited[neighbor] == 0) {
                    visited[neighbor] = 1;
                    pending.push(neighbor);
                }
            }
        }
        const auto gauge = *std::min_element(
            componentPieces.begin(),
            componentPieces.end(),
            [&](std::size_t left, std::size_t right) {
                const double leftDistance =
                    topology.pieceCenterDistanceBaseVoxels[left];
                const double rightDistance =
                    topology.pieceCenterDistanceBaseVoxels[right];
                return leftDistance != rightDistance
                    ? leftDistance < rightDistance
                    : left < right;
            });
        report.gaugePieces.push_back(gauge);
    }

    report.initialCosts = evaluateCosts(report, config);
    if (std::none_of(
            report.activeByPiece.begin(),
            report.activeByPiece.end(),
            [](const unsigned char active) { return active != 0; })) {
        report.finalCosts = report.initialCosts;
        report.solutionUsable = true;
        report.status = "EMPTY";
        report.briefReport = "No active ordered-winding pieces";
        report.solveSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        if (progress) {
            progress({
                FiberTraceWindingOrderedCutsProgressPhase::Complete,
                0,
                config.maximumIterations,
                0.0,
                0.0,
                report.solveSeconds,
            });
        }
        return report;
    }
    ceres::Problem problem;
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        if (report.activeByPiece[piece] == 0)
            continue;
        problem.AddParameterBlock(&report.offsetByPiece[piece], 1);
    }
    for (const auto& factor : report.signFactors) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SignResidual, 2, 1, 1>(
                new SignResidual{
                    phase(report.orientationByPiece[factor.pieceB]) -
                        phase(report.orientationByPiece[factor.pieceA]),
                    factor.sign,
                    factor.margin,
                    checkedSqrt(
                        config.signMarginWeight * factor.confidence,
                        "Ordered-cut sign coefficient"),
                    checkedSqrt(
                        factor.confidence,
                        "Ordered-cut target coefficient"),
                }),
            nullptr,
            &report.offsetByPiece[factor.pieceA],
            &report.offsetByPiece[factor.pieceB]);
    }
    const double continuationCoefficient = checkedSqrt(
        config.continuationWeight,
        "Ordered-cut continuation coefficient");
    for (const auto [a, b] : report.continuationEdges) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ContinuationResidual, 1, 1, 1>(
                new ContinuationResidual{continuationCoefficient}),
            nullptr,
            &report.offsetByPiece[a],
            &report.offsetByPiece[b]);
    }

    std::vector<unsigned char> componentFixed(report.gaugePieces.size(), 0);
    if (!fixedOffsets.empty()) {
        for (std::size_t piece = 0; piece < pieceCount; ++piece) {
            if (report.activeByPiece[piece] == 0 ||
                !fixedOffsets[piece].fixed) {
                continue;
            }
            problem.SetParameterBlockConstant(&report.offsetByPiece[piece]);
            componentFixed.at(report.componentByPiece[piece]) = 1;
        }
    }
    for (std::size_t component = 0;
         component < report.gaugePieces.size(); ++component) {
        if (componentFixed[component] != 0)
            continue;
        const std::size_t gauge = report.gaugePieces[component];
        report.offsetByPiece[gauge] = 0.0;
        problem.SetParameterBlockConstant(&report.offsetByPiece[gauge]);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = static_cast<int>(std::min<std::size_t>(
        config.maximumIterations,
        static_cast<std::size_t>(std::numeric_limits<int>::max())));
    options.num_threads = static_cast<int>(std::min<std::size_t>(
        config.parallelWorkers,
        static_cast<std::size_t>(std::numeric_limits<int>::max())));
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.logging_type = ceres::SILENT;
    options.update_state_every_iteration = true;
    ProgressCallback callback{config.maximumIterations, progress, started};
    if (progress)
        options.callbacks.push_back(&callback);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    report.iterations = summary.iterations.size();
    report.effectiveWorkers = config.parallelWorkers;
    report.initialCost = summary.initial_cost;
    report.finalCost = summary.final_cost;
    report.solutionUsable = summary.IsSolutionUsable();
    report.status = ceres::TerminationTypeToString(summary.termination_type);
    report.briefReport = summary.BriefReport();
    report.solveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    report.finalCosts = evaluateCosts(report, config);
    if (!report.solutionUsable) {
        throw std::runtime_error(
            "Ordered-cut continuous solve is not usable: " +
            report.briefReport);
    }
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        if (report.activeByPiece[piece] != 0 &&
            !std::isfinite(report.offsetByPiece[piece])) {
            throw std::runtime_error(
                "Ordered-cut continuous solve produced a nonfinite offset");
        }
    }
    if (progress) {
        progress({
            FiberTraceWindingOrderedCutsProgressPhase::Complete,
            report.iterations,
            config.maximumIterations,
            report.finalCost,
            0.0,
            report.solveSeconds,
        });
    }
    return report;
}

FiberTraceWindingOrderedCutsReport solveFiberTraceWindingOrderedCuts(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    std::span<const FiberTraceFixedOrientation> fixedOrientations,
    const FiberTraceWindingOrderedCutsConfig& config,
    const FiberTraceWindingOrderedCutsProgressCallback& progress,
    const FiberTraceWindingOrderedRemovalCallback& removalProgress)
{
    FiberTraceWindingOrderedCutsReport report;
    const std::size_t pieceCount = constraints.pieces.size();
    std::vector<unsigned char> activePieces(pieceCount, 1);
    report.ordering = fitFiberTraceWindingOrderedOffsets(
        constraints,
        topology,
        fixedOrientations,
        config,
        {},
        progress,
        activePieces);
    if (config.removeOffendingFibers) {
        std::size_t traceCount = constraints.inputTraces;
        for (const auto& piece : constraints.pieces)
            traceCount = std::max(traceCount, piece.traceIndex + 1);
        std::vector<unsigned char> activeTraces(traceCount, 0);
        for (std::size_t piece = 0; piece < pieceCount; ++piece) {
            if (report.ordering.activeByPiece[piece] != 0) {
                activeTraces[constraints.pieces[piece].traceIndex] = 1;
            }
        }
        while (true) {
            const auto old = summarizeFiberTraceWindingOrderedViolations(
                report.ordering, constraints);
            if (old.infringements == 0)
                break;
            if (!old.worstTrace) {
                throw std::logic_error(
                    "Ordered winding infringements have no offending trace");
            }
            const std::size_t removedTrace = *old.worstTrace;
            const auto offender = old.traces.at(removedTrace);
            activeTraces.at(removedTrace) = 0;
            for (std::size_t piece = 0; piece < pieceCount; ++piece) {
                if (constraints.pieces[piece].traceIndex == removedTrace)
                    activePieces[piece] = 0;
            }
            const auto survivingBefore =
                summarizeFiberTraceWindingOrderedViolations(
                    report.ordering, constraints, activeTraces);
            auto next = fitFiberTraceWindingOrderedOffsets(
                constraints,
                topology,
                fixedOrientations,
                config,
                {},
                {},
                activePieces);
            const auto survivingAfter =
                summarizeFiberTraceWindingOrderedViolations(next, constraints);
            FiberTraceWindingOrderedRemovalStep step;
            step.iteration = report.removals.size() + 1;
            step.removedTrace = removedTrace;
            step.removedPieces = offender.pieces;
            step.incidentFactors = offender.incidentFactors;
            step.violatedFactors = offender.violatedFactors;
            step.oldFactors = old.factors;
            step.oldInfringements = old.infringements;
            step.survivingFactors = survivingBefore.factors;
            step.survivingBeforeInfringements =
                survivingBefore.infringements;
            step.survivingAfterInfringements =
                survivingAfter.infringements;
            step.remainingTraces = static_cast<std::size_t>(std::count(
                activeTraces.begin(),
                activeTraces.end(),
                static_cast<unsigned char>(1)));
            step.solveSeconds = next.solveSeconds;
            report.removals.push_back(step);
            if (removalProgress)
                removalProgress(report.removals.back());
            report.ordering = std::move(next);
            if (report.removals.size() > traceCount) {
                throw std::logic_error(
                    "Ordered offender removal exceeded represented traces");
            }
        }
    }

    DisjointSet continuation(pieceCount);
    for (const auto [a, b] : report.ordering.continuationEdges)
        continuation.unite(a, b);
    std::map<std::size_t, std::size_t> runByRoot;
    std::vector<std::size_t> runByPiece(pieceCount, 0);
    std::vector<std::vector<std::size_t>> piecesByRun;
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        if (report.ordering.activeByPiece[piece] == 0)
            continue;
        const std::size_t root = continuation.find(piece);
        const auto [found, inserted] = runByRoot.try_emplace(
            root, piecesByRun.size());
        if (inserted)
            piecesByRun.emplace_back();
        runByPiece[piece] = found->second;
        piecesByRun[found->second].push_back(piece);
    }

    struct OrderedRun {
        std::size_t run = 0;
        std::size_t minimumPiece = 0;
        double mean = 0.0;
    };
    std::vector<OrderedRun> orderedRuns;
    orderedRuns.reserve(piecesByRun.size());
    for (std::size_t run = 0; run < piecesByRun.size(); ++run) {
        const double sum = std::accumulate(
            piecesByRun[run].begin(),
            piecesByRun[run].end(),
            0.0,
            [&](double value, std::size_t piece) {
                return value + report.ordering.offsetByPiece[piece];
            });
        orderedRuns.push_back({
            run,
            *std::min_element(
                piecesByRun[run].begin(), piecesByRun[run].end()),
            sum / static_cast<double>(piecesByRun[run].size()),
        });
    }
    std::sort(
        orderedRuns.begin(), orderedRuns.end(),
        [](const OrderedRun& left, const OrderedRun& right) {
            if (left.mean != right.mean)
                return left.mean < right.mean;
            return left.minimumPiece < right.minimumPiece;
        });

    struct RunGroup {
        double mean = 0.0;
        std::vector<std::size_t> runs;
    };
    std::vector<RunGroup> groups;
    for (const auto& ordered : orderedRuns) {
        if (groups.empty() || groups.back().mean != ordered.mean)
            groups.push_back({ordered.mean, {}});
        groups.back().runs.push_back(ordered.run);
    }
    std::vector<std::vector<std::size_t>> factorsByRun(piecesByRun.size());
    for (std::size_t factor = 0;
         factor < report.ordering.signFactors.size(); ++factor) {
        const auto& current = report.ordering.signFactors[factor];
        const std::size_t runA = runByPiece[current.pieceA];
        const std::size_t runB = runByPiece[current.pieceB];
        factorsByRun[runA].push_back(factor);
        if (runA != runB)
            factorsByRun[runB].push_back(factor);
    }

    std::vector<int> windingByRun(piecesByRun.size(), 0);
    report.steps.push_back(makeStep(
        report.ordering, windingByRun, runByPiece, 0, std::nullopt));
    if (groups.size() < 2)
        return report;
    std::vector<unsigned char> usedBoundary(groups.size() - 1, 0);
    std::vector<std::size_t> factorMark(
        report.ordering.signFactors.size(), 0);
    std::size_t generation = 0;
    while (config.maximumSplits == 0 ||
           report.steps.back().splits < config.maximumSplits) {
        std::vector<int> candidate = windingByRun;
        for (int& value : candidate)
            ++value;
        std::size_t candidateInfringements = 0;
        for (const auto& factor : report.ordering.signFactors) {
            candidateInfringements += infringed(
                factor,
                candidate,
                runByPiece,
                report.ordering.orientationByPiece)
                ? 1
                : 0;
        }
        std::optional<std::size_t> bestBoundary;
        std::size_t bestInfringements = report.steps.back().signInfringements;
        for (std::size_t group = 0; group + 1 < groups.size(); ++group) {
            ++generation;
            std::vector<std::size_t> touched;
            for (const std::size_t run : groups[group].runs) {
                for (const std::size_t factor : factorsByRun[run]) {
                    if (factorMark[factor] == generation)
                        continue;
                    factorMark[factor] = generation;
                    touched.push_back(factor);
                }
            }
            for (const std::size_t factor : touched) {
                candidateInfringements -= infringed(
                    report.ordering.signFactors[factor],
                    candidate,
                    runByPiece,
                    report.ordering.orientationByPiece)
                    ? 1
                    : 0;
            }
            for (const std::size_t run : groups[group].runs)
                candidate[run] = windingByRun[run];
            for (const std::size_t factor : touched) {
                candidateInfringements += infringed(
                    report.ordering.signFactors[factor],
                    candidate,
                    runByPiece,
                    report.ordering.orientationByPiece)
                    ? 1
                    : 0;
            }
            if (usedBoundary[group] == 0 &&
                candidateInfringements < bestInfringements) {
                bestBoundary = group;
                bestInfringements = candidateInfringements;
            }
        }
        if (!bestBoundary)
            break;
        usedBoundary[*bestBoundary] = 1;
        for (std::size_t group = *bestBoundary + 1;
             group < groups.size(); ++group) {
            for (const std::size_t run : groups[group].runs)
                ++windingByRun[run];
        }
        const double threshold = groups[*bestBoundary].mean +
            0.5 * (groups[*bestBoundary + 1].mean -
                   groups[*bestBoundary].mean);
        report.steps.push_back(makeStep(
            report.ordering,
            windingByRun,
            runByPiece,
            report.steps.back().splits + 1,
            threshold));
        if (report.steps.back().signInfringements != bestInfringements) {
            throw std::logic_error(
                "Ordered-cut incremental scan produced an inconsistent objective");
        }
    }
    return report;
}

FiberTraceInterleavedWindingReport makeFiberTraceOrderedCutsWindingReport(
    const FiberTraceWindingOrderedCutsReport& ordered,
    const FiberTraceWindingOrderedCutsConfig& config,
    std::size_t stepIndex)
{
    if (stepIndex >= ordered.steps.size()) {
        throw std::out_of_range("Ordered-cut winding step is out of range");
    }
    const auto& ordering = ordered.ordering;
    const auto& step = ordered.steps[stepIndex];
    const std::size_t pieces = ordering.offsetByPiece.size();
    if (step.windingByPiece.size() != pieces)
        throw std::invalid_argument("Ordered-cut winding step has wrong size");

    FiberTraceInterleavedWindingReport result;
    result.solver = FiberTraceWindingSolver::OrderedCuts;
    result.orientationMode = FiberTraceWindingOrientationMode::FixedPrepass;
    result.calibrationMode = FiberTraceWindingCalibrationMode::Fixed;
    result.phaseMagnitude = kPhase;
    result.measurementScale = config.measurementScale;
    result.variables = pieces;
    result.factors = ordering.signFactors.size();
    result.connectedComponents = ordering.gaugePieces.size();
    result.gaugePieces = ordering.gaugePieces;
    result.factorDiagnostics = ordering.factorDiagnostics;
    result.continuousSolveSeconds = ordering.solveSeconds;
    result.messageIterations = ordering.iterations;
    result.effectiveWorkers = ordering.effectiveWorkers;
    result.messageConverged = ordering.solutionUsable;
    result.status = ordering.status;
    result.decodedEnergy = static_cast<double>(step.signInfringements);
    result.decodedDataEnergy = result.decodedEnergy;
    result.windingValid = ordering.activeByPiece;
    result.continuousWinding.resize(pieces);
    result.mapWinding = step.windingByPiece;
    result.posteriorMeanWinding.resize(pieces);
    result.mapProbability.assign(pieces, 1.0);
    result.entropy.assign(pieces, 0.0);
    result.candidateMinimum = step.windingByPiece;
    result.candidateMaximum = step.windingByPiece;
    result.componentByPiece = ordering.componentByPiece;
    result.integerGaugeByPiece = ordering.componentByPiece;
    result.classAProbability.assign(pieces, 0.0);
    result.mixedProbability.assign(pieces, 0.0);
    result.classBProbability.assign(pieces, 0.0);
    result.posteriorMeanLatentCoordinate.resize(pieces);
    result.mapLatentCoordinate.resize(pieces);
    result.mapOrientationByPiece = ordering.orientationByPiece;
    result.fixedOrientationByPiece = ordering.orientationByPiece;
    result.incidentSignedConstraints.assign(pieces, 0);
    result.incidentSkippedConstraints.assign(pieces, 0);
    result.componentPhaseSign.assign(ordering.gaugePieces.size(), 1);
    result.componentPositivePhaseSignProbability.assign(
        ordering.gaugePieces.size(), 1.0);
    for (std::size_t piece = 0; piece < pieces; ++piece) {
        const bool active = ordering.activeByPiece[piece] != 0;
        const double piecePhase = phase(ordering.orientationByPiece[piece]);
        result.continuousWinding[piece] = ordering.offsetByPiece[piece];
        result.posteriorMeanWinding[piece] = ordering.offsetByPiece[piece];
        result.posteriorMeanLatentCoordinate[piece] =
            ordering.offsetByPiece[piece] + piecePhase;
        result.mapLatentCoordinate[piece] = active
            ? static_cast<double>(step.windingByPiece[piece]) + piecePhase
            : 0.0;
        if (!active) {
            result.mixedProbability[piece] = 1.0;
        } else if (ordering.orientationByPiece[piece] ==
                   FiberTraceFixedOrientation::Horizontal) {
            result.classAProbability[piece] = 1.0;
        } else {
            result.classBProbability[piece] = 1.0;
        }
    }
    for (const auto& factor : ordering.signFactors) {
        ++result.incidentSignedConstraints[factor.pieceA];
        ++result.incidentSignedConstraints[factor.pieceB];
    }
    return result;
}

}  // namespace vc::fiber_tracer
