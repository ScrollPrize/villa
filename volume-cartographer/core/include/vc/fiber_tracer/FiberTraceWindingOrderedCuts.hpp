#pragma once

#include "vc/fiber_tracer/FiberTraceWindingBeliefPropagation.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace vc::fiber_tracer
{

struct FiberTraceWindingOrderedCutsConfig
    : FiberTraceWindingBeliefPropagationConfig {
    double signMarginWeight = 16.0;
    double continuationWeight = 16.0;
    double measurementScale = 1.0;
    std::size_t maximumIterations = 100;
    std::size_t maximumSplits = 0;
    bool removeOffendingFibers = false;
};

struct FiberTraceWindingOrderedCutsFixedOffset {
    bool fixed = false;
    double offset = 0.0;
};

enum class FiberTraceWindingOrderedCutsProgressPhase : unsigned char {
    Preparing,
    Iterating,
    Complete,
};

struct FiberTraceWindingOrderedCutsProgress {
    FiberTraceWindingOrderedCutsProgressPhase phase =
        FiberTraceWindingOrderedCutsProgressPhase::Preparing;
    std::size_t iteration = 0;
    std::size_t maximumIterations = 0;
    double cost = 0.0;
    double gradientMaximumNorm = 0.0;
    double elapsedSeconds = 0.0;
};

using FiberTraceWindingOrderedCutsProgressCallback = std::function<void(
    const FiberTraceWindingOrderedCutsProgress&)>;

struct FiberTraceWindingOrderedSignFactor {
    std::size_t constraintIndex = 0;
    std::size_t pieceA = 0;
    std::size_t pieceB = 0;
    double sign = 1.0;
    double margin = 0.5;
    double confidence = 1.0;
    bool parallel = false;
};

struct FiberTraceWindingOrderedCutsCostSummary {
    double signMargin = 0.0;
    double targetDistance = 0.0;
    double continuation = 0.0;

    [[nodiscard]] double total() const noexcept
    {
        return signMargin + targetDistance + continuation;
    }
};

struct FiberTraceWindingOrderedOffsetReport {
    std::vector<double> offsetByPiece;
    std::vector<FiberTraceFixedOrientation> orientationByPiece;
    std::vector<unsigned char> activeByPiece;
    std::vector<std::size_t> componentByPiece;
    std::vector<std::size_t> gaugePieces;
    std::vector<FiberTraceWindingOrderedSignFactor> signFactors;
    std::vector<std::pair<std::size_t, std::size_t>> continuationEdges;
    std::vector<FiberTraceWindingFactorDiagnostic> factorDiagnostics;
    FiberTraceWindingOrderedCutsCostSummary initialCosts;
    FiberTraceWindingOrderedCutsCostSummary finalCosts;
    std::size_t iterations = 0;
    std::size_t effectiveWorkers = 1;
    double initialCost = 0.0;
    double finalCost = 0.0;
    double solveSeconds = 0.0;
    bool solutionUsable = false;
    std::string status;
    std::string briefReport;
};

struct FiberTraceWindingOrderedTraceViolation {
    std::size_t traceIndex = 0;
    std::size_t pieces = 0;
    std::size_t incidentFactors = 0;
    std::size_t violatedFactors = 0;
};

struct FiberTraceWindingOrderedViolationSummary {
    std::size_t factors = 0;
    std::size_t infringements = 0;
    std::vector<FiberTraceWindingOrderedTraceViolation> traces;
    std::optional<std::size_t> worstTrace;
};

struct FiberTraceWindingOrderedRemovalStep {
    std::size_t iteration = 0;
    std::size_t removedTrace = 0;
    std::size_t removedPieces = 0;
    std::size_t incidentFactors = 0;
    std::size_t violatedFactors = 0;
    std::size_t oldFactors = 0;
    std::size_t oldInfringements = 0;
    std::size_t survivingFactors = 0;
    std::size_t survivingBeforeInfringements = 0;
    std::size_t survivingAfterInfringements = 0;
    std::size_t remainingTraces = 0;
    double solveSeconds = 0.0;
};

using FiberTraceWindingOrderedRemovalCallback = std::function<void(
    const FiberTraceWindingOrderedRemovalStep&)>;

struct FiberTraceWindingOrderedCutStep {
    std::size_t splits = 0;
    std::size_t windings = 1;
    std::size_t signFactors = 0;
    std::size_t signInfringements = 0;
    std::size_t continuationCuts = 0;
    std::optional<double> threshold;
    std::vector<int> windingByPiece;
};

struct FiberTraceWindingOrderedCutsReport {
    FiberTraceWindingOrderedOffsetReport ordering;
    std::vector<FiberTraceWindingOrderedRemovalStep> removals;
    std::vector<FiberTraceWindingOrderedCutStep> steps;
};

[[nodiscard]] FiberTraceWindingOrderedViolationSummary
summarizeFiberTraceWindingOrderedViolations(
    const FiberTraceWindingOrderedOffsetReport& ordering,
    const FiberTraceConstraintReport& constraints,
    std::span<const unsigned char> includedTraces = {});

[[nodiscard]] FiberTraceWindingOrderedOffsetReport
fitFiberTraceWindingOrderedOffsets(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    std::span<const FiberTraceFixedOrientation> fixedOrientations,
    const FiberTraceWindingOrderedCutsConfig& config,
    std::span<const FiberTraceWindingOrderedCutsFixedOffset> fixedOffsets = {},
    const FiberTraceWindingOrderedCutsProgressCallback& progress = {},
    std::span<const unsigned char> activePieces = {});

[[nodiscard]] FiberTraceWindingOrderedCutsReport
solveFiberTraceWindingOrderedCuts(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    std::span<const FiberTraceFixedOrientation> fixedOrientations,
    const FiberTraceWindingOrderedCutsConfig& config,
    const FiberTraceWindingOrderedCutsProgressCallback& progress = {},
    const FiberTraceWindingOrderedRemovalCallback& removalProgress = {});

[[nodiscard]] FiberTraceInterleavedWindingReport
makeFiberTraceOrderedCutsWindingReport(
    const FiberTraceWindingOrderedCutsReport& ordered,
    const FiberTraceWindingOrderedCutsConfig& config,
    std::size_t stepIndex);

}  // namespace vc::fiber_tracer
