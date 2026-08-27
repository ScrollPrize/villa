#pragma once

#include "vc/fiber_tracer/FiberTraceBeliefPropagation.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace vc::fiber_tracer
{

struct FiberTraceWindingBeliefPropagationConfig {
    double temperature = 0.25;
    double messageDamping = 0.5;
    double messageResidualTolerance = 1.0e-8;
    double boundaryProbabilityThreshold = 0.01;
    std::size_t maximumMessageIterations = 500;
    std::size_t maximumTotalCandidateStates = 4'000'000;
    std::size_t parallelWorkers = 1;
};

struct FiberTraceInterleavedWindingConfig : FiberTraceWindingBeliefPropagationConfig {
    double minimumMeasurementScale = 0.5;
    double maximumMeasurementScale = 2.0;
    double calibrationTolerance = 1.0e-4;
    std::size_t maximumCalibrationIterations = 8;
};

enum class FiberTraceInterleavedWindingProgressPhase {
    Preparing,
    MessagePassing,
    Calibration,
    InitializationComplete,
    Complete,
};

struct FiberTraceInterleavedWindingProgress {
    FiberTraceInterleavedWindingProgressPhase phase =
        FiberTraceInterleavedWindingProgressPhase::Preparing;
    std::size_t initialization = 0;
    std::size_t initializationCount = 0;
    std::size_t calibrationIteration = 0;
    std::size_t maximumCalibrationIterations = 0;
    std::size_t adaptiveSupportRound = 0;
    std::size_t messageIteration = 0;
    std::size_t maximumMessageIterations = 0;
    std::size_t accumulatedMessageIterations = 0;
    std::size_t candidateStates = 0;
    double messageResidual = 0.0;
    double phaseMagnitude = 0.0;
    double measurementScale = 1.0;
    double elapsedSeconds = 0.0;
};

using FiberTraceInterleavedWindingProgressCallback =
    std::function<void(const FiberTraceInterleavedWindingProgress&)>;

struct FiberTraceWindingFactorDiagnostic {
    std::size_t constraintIndex = 0;
    std::size_t pieceA = 0;
    std::size_t pieceB = 0;
    std::size_t canonicalNodeA = 0;
    std::size_t canonicalNodeB = 0;
    double parallelScore = 0.0;
    double perpendicularScore = 0.0;
    std::optional<double> originalSignedDelta;
    std::optional<double> canonicalSignedDelta;
    std::optional<std::size_t> normalComponent;
    bool selfEdge = false;
};

struct FiberTraceWindingBeliefPropagationReport {
    std::vector<double> continuousWinding;
    std::vector<int> mapWinding;
    std::vector<double> posteriorMeanWinding;
    std::vector<double> mapProbability;
    std::vector<double> entropy;
    std::vector<int> candidateMinimum;
    std::vector<int> candidateMaximum;
    std::vector<std::size_t> componentByPiece;
    std::vector<std::size_t> gaugePieces;
    std::vector<std::size_t> incidentSignedConstraints;
    std::vector<std::size_t> incidentSkippedConstraints;
    std::vector<FiberTraceWindingFactorDiagnostic> factorDiagnostics;
    std::size_t variables = 0;
    std::size_t factors = 0;
    std::size_t connectedComponents = 0;
    std::size_t expansionRounds = 0;
    std::size_t messageIterations = 0;
    std::size_t totalCandidateStates = 0;
    std::size_t effectiveWorkers = 1;
    double continuousRootMeanSquareResidual = 0.0;
    double temperature = 0.0;
    double messageResidual = 0.0;
    double continuousSolveSeconds = 0.0;
    double discreteSolveSeconds = 0.0;
    bool messageConverged = false;
    std::string status;
};

struct FiberTraceInterleavedWindingReport : FiberTraceWindingBeliefPropagationReport {
    std::vector<double> classAProbability;
    std::vector<double> mixedProbability;
    std::vector<double> classBProbability;
    std::vector<double> posteriorMeanLatentCoordinate;
    std::vector<int> componentPhaseSign;
    double phaseMagnitude = 0.0;
    double measurementScale = 1.0;
    double decodedEnergy = 0.0;
    std::size_t calibrationIterations = 0;
    std::size_t selectedInitialization = 0;
    std::size_t rankDeficientUpdates = 0;
    bool calibrationConverged = false;
};

[[nodiscard]] FiberTraceWindingBeliefPropagationReport
solveFiberTraceWindingBeliefPropagation(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceWindingBeliefPropagationConfig& config = {});

[[nodiscard]] FiberTraceInterleavedWindingReport
solveFiberTraceInterleavedWindingBeliefPropagation(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceBeliefPropagationReport& orientationBeliefs,
    const FiberTraceInterleavedWindingConfig& config = {},
    const FiberTraceInterleavedWindingProgressCallback& progress = {});

}  // namespace vc::fiber_tracer
