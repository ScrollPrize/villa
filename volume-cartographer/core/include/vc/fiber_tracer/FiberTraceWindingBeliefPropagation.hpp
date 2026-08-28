#pragma once

#include "vc/fiber_tracer/FiberTraceBeliefPropagation.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace vc::fiber_tracer
{

enum class FiberTraceWindingSolver : unsigned char {
    JointGrid,
    Alternating,
};

[[nodiscard]] const char* fiberTraceWindingSolverName(
    FiberTraceWindingSolver solver) noexcept;

enum class FiberTraceWindingOrientationMode : unsigned char {
    Joint,
    FixedPrepass,
};

[[nodiscard]] const char* fiberTraceWindingOrientationModeName(
    FiberTraceWindingOrientationMode mode) noexcept;

enum class FiberTraceFixedOrientation : unsigned char {
    Horizontal,
    Mixed,
    Vertical,
};

[[nodiscard]] const char* fiberTraceFixedOrientationName(
    FiberTraceFixedOrientation orientation) noexcept;

[[nodiscard]] std::vector<FiberTraceFixedOrientation>
fixedFiberTraceOrientations(
    const FiberTraceBeliefPropagationReport& orientationBeliefs);

enum class FiberTraceWindingCalibrationMode : unsigned char {
    Adaptive,
    Fixed,
};

[[nodiscard]] const char* fiberTraceWindingCalibrationModeName(
    FiberTraceWindingCalibrationMode mode) noexcept;

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
    double mixedUnaryCost = 0.5;
    double orientationTemperature = 0.25;
    double minimumMeasurementScale = 0.5;
    double maximumMeasurementScale = 2.0;
    double calibrationTolerance = 1.0e-4;
    std::size_t maximumCalibrationIterations = 8;
};

struct FiberTraceJointGridWindingConfig : FiberTraceWindingBeliefPropagationConfig {
    double mixedUnaryCost = 0.5;
    double orientationTemperature = 0.25;
    std::optional<double> fixedPhaseMagnitude;
    std::optional<double> fixedMeasurementScale;
    double logGainStep = 0.09531017980432486;
    double calibrationBoundaryProbabilityThreshold = 0.25;
    double calibrationDiscardProbabilityThreshold = 0.001;
    double calibrationPosteriorTolerance = 1.0e-6;
    std::size_t initialGainCells = 5;
    std::size_t phaseCells = 6;
    std::size_t maximumGainCells = 17;
    std::size_t maximumGridShifts = 32;
    std::size_t stableIterations = 3;
};

enum class FiberTraceJointGridProgressPhase {
    Preparing,
    MessagePassing,
    SupportChanged,
    Complete,
};

struct FiberTraceJointGridProgress {
    FiberTraceJointGridProgressPhase phase =
        FiberTraceJointGridProgressPhase::Preparing;
    FiberTraceWindingCalibrationMode calibrationMode =
        FiberTraceWindingCalibrationMode::Adaptive;
    std::size_t messageIteration = 0;
    std::size_t maximumMessageIterations = 0;
    std::size_t candidateStates = 0;
    std::size_t gainCells = 0;
    std::size_t phaseCells = 0;
    std::size_t gridShifts = 0;
    double messageResidual = 0.0;
    double calibrationPosteriorResidual = 0.0;
    double phaseMap = 0.0;
    double phaseMean = 0.0;
    double scaleMap = 1.0;
    double scaleMean = 1.0;
    double lowerGainBoundaryProbability = 0.0;
    double upperGainBoundaryProbability = 0.0;
    double minimumGain = 1.0;
    double maximumGain = 1.0;
    double elapsedSeconds = 0.0;
};

using FiberTraceJointGridProgressCallback =
    std::function<void(const FiberTraceJointGridProgress&)>;

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
    double windingWeightMultiplier = 1.0;
    double effectiveParallelWindingWeight = 0.0;
    double effectivePerpendicularWindingWeight = 0.0;
    std::optional<double> originalSignedDelta;
    std::optional<double> canonicalSignedDelta;
    std::optional<double> effectiveSignedDelta;
    std::optional<std::size_t> normalComponent;
    bool selfEdge = false;
};

struct FiberTraceWindingBeliefPropagationReport {
    std::vector<unsigned char> windingValid;
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
    std::vector<double> componentPositivePhaseSignProbability;
    std::vector<FiberTraceFixedOrientation> fixedOrientationByPiece;
    FiberTraceWindingSolver solver = FiberTraceWindingSolver::Alternating;
    FiberTraceWindingOrientationMode orientationMode =
        FiberTraceWindingOrientationMode::Joint;
    FiberTraceWindingCalibrationMode calibrationMode =
        FiberTraceWindingCalibrationMode::Adaptive;
    double phaseMagnitude = 0.0;
    double measurementScale = 1.0;
    double defectUnaryCost = 0.5;
    double calibrationPhaseMean = 0.0;
    double calibrationScaleMean = 1.0;
    double calibrationEntropy = 0.0;
    double lowerGainBoundaryProbability = 0.0;
    double upperGainBoundaryProbability = 0.0;
    double minimumCalibrationGain = 1.0;
    double maximumCalibrationGain = 1.0;
    double decodedEnergy = 0.0;
    std::size_t calibrationGridCells = 0;
    std::size_t calibrationGridShifts = 0;
    std::size_t calibrationIterations = 0;
    std::size_t hardSignProjectedDefects = 0;
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
    const FiberTraceInterleavedWindingProgressCallback& progress = {},
    std::span<const FiberTraceFixedOrientation> fixedOrientations = {});

[[nodiscard]] FiberTraceInterleavedWindingReport
solveFiberTraceJointGridWindingBeliefPropagation(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceJointGridWindingConfig& config = {},
    const FiberTraceJointGridProgressCallback& progress = {},
    std::span<const FiberTraceFixedOrientation> fixedOrientations = {});

}  // namespace vc::fiber_tracer
