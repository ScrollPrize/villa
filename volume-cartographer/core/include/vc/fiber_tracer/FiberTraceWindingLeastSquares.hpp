#pragma once

#include "vc/fiber_tracer/FiberTraceWindingBeliefPropagation.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace vc::fiber_tracer
{

struct FiberTraceWindingLeastSquaresState {
    double horizontalness = 0.5;
    double activity = 1.0;
    double winding = 0.0;
};

struct FiberTraceWindingLeastSquaresFixedState {
    bool fixed = false;
    FiberTraceWindingLeastSquaresState state;
};

struct FiberTraceWindingLeastSquaresConfig
    : FiberTraceWindingBeliefPropagationConfig {
    double defectCost = 1.0;
    double pieceBreakCost = 0.0;
    double orientationExtremenessCost = 1.0;
    double hardConstraintCost = 10'000.0;
    double signMargin = 0.01;
    double measurementScale = 1.0;
    std::size_t maximumIterations = 100;
};

enum class FiberTraceWindingLeastSquaresProgressPhase : unsigned char {
    Preparing,
    Iterating,
    Complete,
};

struct FiberTraceWindingLeastSquaresProgress {
    FiberTraceWindingLeastSquaresProgressPhase phase =
        FiberTraceWindingLeastSquaresProgressPhase::Preparing;
    std::size_t iteration = 0;
    std::size_t maximumIterations = 0;
    double cost = 0.0;
    double gradientMaximumNorm = 0.0;
    double elapsedSeconds = 0.0;
};

using FiberTraceWindingLeastSquaresProgressCallback = std::function<void(
    const FiberTraceWindingLeastSquaresProgress&)>;

struct FiberTraceWindingLeastSquaresCostSummary {
    double orientation = 0.0;
    double windingMagnitude = 0.0;
    double windingSign = 0.0;
    double defect = 0.0;
    double orientationExtremeness = 0.0;
    double continuation = 0.0;
    double pieceBreak = 0.0;

    [[nodiscard]] double total() const noexcept;
};

struct FiberTraceWindingLeastSquaresReport {
    std::vector<FiberTraceWindingLeastSquaresState> states;
    std::vector<std::size_t> componentByPiece;
    std::vector<std::size_t> gaugePieces;
    std::vector<std::size_t> incidentEffectiveConstraints;
    std::vector<FiberTraceWindingFactorDiagnostic> factorDiagnostics;
    FiberTraceWindingLeastSquaresCostSummary initialCosts;
    FiberTraceWindingLeastSquaresCostSummary finalCosts;
    std::size_t factors = 0;
    std::size_t iterations = 0;
    std::size_t effectiveWorkers = 1;
    double initialCost = 0.0;
    double finalCost = 0.0;
    double solveSeconds = 0.0;
    bool solutionUsable = false;
    std::string status;
    std::string briefReport;
};

[[nodiscard]] FiberTraceWindingLeastSquaresReport
solveFiberTraceWindingLeastSquares(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceWindingLeastSquaresConfig& config,
    std::span<const FiberTraceWindingLeastSquaresState> initialStates = {},
    std::span<const FiberTraceWindingLeastSquaresFixedState> fixedStates = {},
    const FiberTraceWindingLeastSquaresProgressCallback& progress = {});

[[nodiscard]] FiberTraceInterleavedWindingReport
makeFiberTraceInterleavedWindingReport(
    const FiberTraceWindingLeastSquaresReport& leastSquares,
    const FiberTraceWindingLeastSquaresConfig& config);

}  // namespace vc::fiber_tracer
