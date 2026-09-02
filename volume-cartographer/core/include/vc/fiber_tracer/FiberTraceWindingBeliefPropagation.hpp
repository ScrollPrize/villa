#pragma once

#include "vc/fiber_tracer/FiberTraceBeliefPropagation.hpp"

#include <array>

#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace vc::fiber_tracer
{

[[nodiscard]] double quantizedHalfWindingTarget(double value);
[[nodiscard]] double quantizedIntegerWindingTarget(double value);

struct FiberTraceCanonicalConstraintCounts {
    std::size_t correct = 0;
    std::size_t falseCount = 0;
    std::size_t total = 0;

    void add(double canonicalStep, double groundTruthStep) noexcept;
};

struct FiberTraceReferenceConstraintDiagnosticRow {
    std::size_t constraintIndex = 0;
    std::size_t ownerSource = 0;
    std::size_t targetSource = 0;
    bool perpendicularDominant = false;
    bool signedMeasurement = false;
    double rawStep = 0.0;
    double calibratedStep = 0.0;
    double canonicalStep = 0.0;
    double groundTruthStep = 0.0;
};

struct FiberTraceReferenceConstraintDiagnosticReport {
    std::vector<FiberTraceReferenceConstraintDiagnosticRow> rows;
    FiberTraceCanonicalConstraintCounts counts;
};

enum class FiberTraceWindingSolver : unsigned char {
    JointGrid,
    Alternating,
    Ceres,
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

struct FiberTraceFixedWindingState {
    bool fixed = false;
    FiberTraceFixedOrientation orientation =
        FiberTraceFixedOrientation::Horizontal;
    int winding = 0;
    int componentPhaseSign = 1;
};

struct FiberTraceJointGridInitialState {
    bool active = false;
    FiberTraceFixedOrientation orientation =
        FiberTraceFixedOrientation::Horizontal;
    int winding = 0;
    int componentPhaseSign = 1;
};

enum class FiberTraceJointGridInitializationMode : unsigned char {
    SupportOnly,
    ConditionedMessages,
};

struct FiberTraceFinalStateCounts {
    std::size_t pieces = 0;
    std::size_t horizontal = 0;
    std::size_t vertical = 0;
    std::size_t defect = 0;

    [[nodiscard]] std::size_t active() const noexcept
    {
        return horizontal + vertical;
    }
};

struct FiberTraceFinalStateCohortSummary {
    FiberTraceFinalStateCounts selected;
    FiberTraceFinalStateCounts other;
    FiberTraceFinalStateCounts total;
};

[[nodiscard]] FiberTraceFinalStateCohortSummary
summarizeFiberTraceFinalStates(
    std::span<const FiberTraceFixedOrientation> orientations,
    std::span<const unsigned char> windingValid,
    std::span<const unsigned char> selectedCohort);

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

inline constexpr std::array<double, 5>
    kDefaultFiberTraceWindingClassWeights{0.0, 0.0, 0.5, 4.0, 1.0};
inline constexpr std::array<double, 2>
    kDefaultFiberTraceWindingSignWeights{1.0, 0.5};

enum class FiberTraceWindingDecisionConfidence : unsigned char {
    Legacy,
    Linear,
    Cosine,
};

enum class FiberTraceWindingNormalConfidence : unsigned char {
    None,
    Linear,
    Cosine,
};

[[nodiscard]] const char* fiberTraceWindingDecisionConfidenceName(
    FiberTraceWindingDecisionConfidence mode) noexcept;

[[nodiscard]] const char* fiberTraceWindingNormalConfidenceName(
    FiberTraceWindingNormalConfidence mode) noexcept;

struct FiberTraceWindingBeliefPropagationConfig {
    double temperature = 0.25;
    double messageDamping = 0.5;
    double messageResidualTolerance = 1.0e-8;
    double boundaryProbabilityThreshold = 0.01;
    std::size_t maximumMessageIterations = 500;
    std::size_t maximumTotalCandidateStates = 4'000'000;
    std::size_t parallelWorkers = 1;
    std::optional<double> parallelWindingDistanceCutoff;
    bool enforcePerpendicularWindingSign = true;
    bool enforceParallelWindingSign = true;
    bool enforceHardSplitContinuity = true;
    std::optional<double> hardSignMinimumNormalAlignment =
        0.8660254037844386;
    FiberTraceWindingDecisionConfidence decisionConfidence =
        FiberTraceWindingDecisionConfidence::Cosine;
    FiberTraceWindingNormalConfidence normalConfidence =
        FiberTraceWindingNormalConfidence::Linear;
    std::optional<double> finiteSignInfringementCost = 44.0;
    double perpendicularNextWeight =
        kDefaultFiberTraceWindingClassWeights[0];
    double perpendicularFarWeight =
        kDefaultFiberTraceWindingClassWeights[1];
    double parallelSameWeight =
        kDefaultFiberTraceWindingClassWeights[2];
    double parallelOneWeight =
        kDefaultFiberTraceWindingClassWeights[3];
    double parallelFarWeight =
        kDefaultFiberTraceWindingClassWeights[4];
    double perpendicularSignWeight =
        kDefaultFiberTraceWindingSignWeights[0];
    double parallelSignWeight =
        kDefaultFiberTraceWindingSignWeights[1];
};

struct FiberTraceWindingComponentSelection {
    std::vector<std::size_t> retainedPieceIndices;
    std::size_t components = 0;
    std::size_t retainedPieces = 0;
    std::size_t removedPieces = 0;
};

[[nodiscard]] FiberTraceWindingComponentSelection selectLargestFiberTraceWindingComponent(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceWindingBeliefPropagationConfig& config,
    std::span<const FiberTraceFixedOrientation> fixedOrientations = {},
    bool quantizeComponentTargets = true,
    std::optional<std::size_t> preferredPiece = std::nullopt,
    double measurementScale = 1.0);

struct FiberTraceInterleavedWindingConfig : FiberTraceWindingBeliefPropagationConfig {
    double mixedUnaryCost = 100.0;
    double pieceBreakCost = 0.0;
    double orientationTemperature = 0.25;
    double minimumMeasurementScale = 0.5;
    double maximumMeasurementScale = 2.0;
    double calibrationTolerance = 1.0e-4;
    std::size_t maximumCalibrationIterations = 8;
};

struct FiberTraceJointGridWindingConfig : FiberTraceWindingBeliefPropagationConfig {
    double mixedUnaryCost = 100.0;
    double pieceBreakCost = 0.0;
    double continuousCoordinateCost = 0.0;
    double localSpanCost = 0.0;
    double hardSignSlackCost = 0.0;
    std::optional<int> minimumActiveWinding;
    std::optional<int> maximumActiveWinding;
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
    double parallelWindingWeightMultiplier = 1.0;
    double perpendicularWindingWeightMultiplier = 1.0;
    double effectiveParallelWindingWeight = 0.0;
    double effectivePerpendicularWindingWeight = 0.0;
    double decisionConfidenceMultiplier = 1.0;
    double normalConfidenceMultiplier = 1.0;
    double effectiveParallelSignPenalty = 0.0;
    double effectivePerpendicularSignPenalty = 0.0;
    double parallelSignWeightMultiplier = 1.0;
    double perpendicularSignWeightMultiplier = 1.0;
    std::optional<double> perpendicularNormalAlignment;
    std::optional<double> parallelNormalAlignment;
    std::optional<double> originalSignedDelta;
    std::optional<double> canonicalSignedDelta;
    double effectiveParallelWindingDistance = 0.0;
    std::optional<double> effectivePerpendicularSignedDelta;
    std::optional<std::size_t> normalComponent;
    bool parallelWindingRetained = false;
    bool parallelMagnitudePresent = false;
    bool perpendicularMagnitudePresent = false;
    bool parallelSignPresent = false;
    bool perpendicularSignPresent = false;
    bool selfEdge = false;
    std::optional<double> originalSignedParallelDelta;
    std::optional<double> canonicalSignedParallelDelta;
    std::optional<double> effectiveSignedParallelDelta;
    bool hardPerpendicularSign = false;
    bool hardParallelSign = false;
    bool perpendicularSignPromotedByAlignment = false;
    bool parallelSignPromotedByAlignment = false;
};

// Shared, fully materialized winding model. Solvers must consume this model
// rather than reconstructing scale, confidence, incidence, or gauge semantics
// from diagnostic output.
struct FiberTracePreparedWindingMeasurement {
    std::size_t constraintIndex = 0;
    std::size_t a = 0;
    std::size_t b = 0;
    double parallel = 0.0;
    double perpendicular = 0.0;
    double parallelConfidence = 0.0;
    double perpendicularConfidence = 0.0;
    double parallelMultiplier = 1.0;
    double perpendicularMultiplier = 1.0;
    double parallelDistance = 0.0;
    std::optional<double> parallelSignedDelta;
    std::optional<double> perpendicularSignedDelta;
    std::optional<std::size_t> normalComponent;
    bool hardParallelSign = false;
    bool hardPerpendicularSign = false;
    double parallelSignPenalty = 0.0;
    double perpendicularSignPenalty = 0.0;
    double rawParallelDistance = 0.0;
    std::optional<double> rawParallelSignedDelta;
    std::optional<double> rawPerpendicularSignedDelta;
    std::optional<double> selectedNormalAlignment;
    double selectedConfidence = 1.0;
    bool continuity = false;
    bool quantizeTargets = true;
    bool parallelDominant = false;
    bool perpendicularDominant = false;
    bool parallelMagnitudePresent = false;
    bool perpendicularMagnitudePresent = false;
    bool parallelSignPresent = false;
    bool perpendicularSignPresent = false;
};

struct FiberTracePreparedWindingEdge {
    std::size_t a = 0;
    std::size_t b = 0;
    std::vector<FiberTracePreparedWindingMeasurement> measurements;
    bool containsHardContinuity = false;
};

struct FiberTracePreparedWindingModel {
    std::vector<std::size_t> pieceToNode;
    std::vector<std::vector<std::size_t>> piecesByNode;
    std::vector<FiberTracePreparedWindingEdge> edges;
    std::vector<std::vector<std::size_t>> adjacency;
    std::vector<std::size_t> incidentMeasurements;
    std::vector<std::size_t> incidentWindingMeasurements;
    std::vector<std::size_t> componentByNode;
    std::vector<std::size_t> integerGaugeByNode;
    std::vector<std::size_t> gaugeNodeByComponent;
    std::vector<std::size_t> gaugePieceByComponent;
    std::vector<std::size_t> integerGaugeNodes;
    std::vector<FiberTraceWindingFactorDiagnostic> diagnostics;
    FiberTraceWindingBeliefPropagationConfig config;
    double preparedMeasurementScale = 1.0;
};

[[nodiscard]] FiberTracePreparedWindingModel prepareFiberTraceWindingModel(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceWindingBeliefPropagationConfig& config = {},
    std::span<const FiberTraceFixedOrientation> fixedOrientations = {},
    bool quantizeComponentTargets = true,
    double measurementScale = 1.0);

[[nodiscard]] std::vector<FiberTraceWindingFactorDiagnostic>
diagnoseFiberTraceWindingFactors(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceWindingBeliefPropagationConfig& config = {},
    std::span<const FiberTraceFixedOrientation> fixedOrientations = {},
    bool quantizeComponentTargets = true,
    double measurementScale = 1.0);

enum class FiberTraceConstraintEvidenceClass : unsigned char {
    Continuity,
    PerpendicularMagnitude,
    PerpendicularSign,
    ParallelSameMagnitude,
    ParallelOtherMagnitude,
    ParallelSign,
    Count,
};

[[nodiscard]] const char* fiberTraceConstraintEvidenceClassName(
    FiberTraceConstraintEvidenceClass evidenceClass) noexcept;

struct FiberTraceConstraintEvidenceCounts {
    std::size_t incidences = 0;
    std::size_t activeIncidences = 0;
    std::size_t defectIncidences = 0;
    std::size_t hardSignIncidences = 0;
    std::size_t activeHardSignIncidences = 0;
    std::size_t defectHardSignIncidences = 0;
    double effectiveWeight = 0.0;
    double activeEffectiveWeight = 0.0;
    double defectEffectiveWeight = 0.0;
};

struct FiberTraceConstraintEvidenceCohort {
    FiberTraceFinalStateCounts states;
    std::array<
        FiberTraceConstraintEvidenceCounts,
        static_cast<std::size_t>(FiberTraceConstraintEvidenceClass::Count)>
        classes;
    FiberTraceConstraintEvidenceCounts total;
};

struct FiberTraceConstraintEvidenceSummary {
    FiberTraceConstraintEvidenceCohort selected;
    FiberTraceConstraintEvidenceCohort other;
    FiberTraceConstraintEvidenceCohort total;
};

enum class FiberTraceConstraintAgreementClass : unsigned char {
    Continuity,
    PerpendicularOrientation,
    PerpendicularMagnitudeNext,
    PerpendicularMagnitudeFar,
    PerpendicularSign,
    ParallelOrientation,
    ParallelMagnitudeSame,
    ParallelMagnitudeOne,
    ParallelMagnitudeFar,
    ParallelSign,
    Count,
};

[[nodiscard]] const char* fiberTraceConstraintAgreementClassName(
    FiberTraceConstraintAgreementClass agreementClass) noexcept;

struct FiberTraceConstraintAgreementCounts {
    std::size_t prepared = 0;
    std::size_t evaluated = 0;
    std::size_t defectNeutralized = 0;
    std::size_t infringed = 0;
};

struct FiberTraceConstraintAgreementSummary {
    std::array<
        FiberTraceConstraintAgreementCounts,
        static_cast<std::size_t>(FiberTraceConstraintAgreementClass::Count)>
        classes;
    FiberTraceConstraintAgreementCounts total;
};

[[nodiscard]] FiberTraceConstraintEvidenceSummary
summarizeFiberTraceConstraintEvidence(
    const FiberTraceConstraintReport& constraints,
    std::span<const FiberTraceWindingFactorDiagnostic> diagnostics,
    std::span<const FiberTraceFixedOrientation> orientations,
    std::span<const unsigned char> windingValid,
    std::span<const unsigned char> selectedCohort);

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
    std::vector<std::size_t> integerGaugeByPiece;
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
    std::vector<double> mapLatentCoordinate;
    std::vector<FiberTraceFixedOrientation> mapOrientationByPiece;
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
    double defectUnaryCost = 1.0;
    double pieceBreakCost = 0.0;
    double calibrationPhaseMean = 0.0;
    double calibrationScaleMean = 1.0;
    double calibrationEntropy = 0.0;
    double lowerGainBoundaryProbability = 0.0;
    double upperGainBoundaryProbability = 0.0;
    double minimumCalibrationGain = 1.0;
    double maximumCalibrationGain = 1.0;
    double decodedEnergy = 0.0;
    double decodedDataEnergy = 0.0;
    double continuousCoordinateEnergy = 0.0;
    double localSpanEnergy = 0.0;
    double hardSignSlackEnergy = 0.0;
    double initialStateDecodedEnergy =
        std::numeric_limits<double>::quiet_NaN();
    std::size_t calibrationGridCells = 0;
    std::size_t calibrationGridShifts = 0;
    std::size_t calibrationIterations = 0;
    std::size_t hardSignProjectedDefects = 0;
    std::size_t initialDefectGaugeComponents = 0;
    std::size_t selectedInitialization = 0;
    std::size_t rankDeficientUpdates = 0;
    bool initializedFromState = false;
    bool conditionedMessageInitialization = false;
    bool calibrationConverged = false;
};

[[nodiscard]] FiberTraceConstraintAgreementSummary
summarizeFiberTraceConstraintAgreement(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceInterleavedWindingReport& winding);

enum class FiberTraceReferenceConstraintClass : unsigned char {
    Perpendicular,
    ParallelSameWinding,
    ParallelOtherWinding,
};

enum class FiberTraceReferenceConstraintGroup : unsigned char {
    PerpendicularNext,
    PerpendicularFar,
    ParallelSame,
    ParallelOne,
    ParallelTwoPlus,
    Count,
};

[[nodiscard]] const char* fiberTraceReferenceConstraintGroupName(
    FiberTraceReferenceConstraintGroup group) noexcept;

struct FiberTraceReferenceWindingObservation {
    FiberTraceReferenceConstraintClass constraintClass = FiberTraceReferenceConstraintClass::Perpendicular;
    std::size_t integerGauge = 0;
    double virtualReferenceWinding = 0.0;
    std::array<double, 2> inferredReferenceWindings{0.0, 0.0};
    std::size_t inferredReferenceWindingCount = 0;
    std::size_t referenceSource = 0;
    double canonicalWindingDistance = 0.0;
    double rawCoefficient = 1.0;
    double admittedCoefficient = 1.0;
    double coordinateResidualScale = 1.0;
    bool exactWindingFactor = false;
    double bpLatentCoordinate = 0.0;
    double referenceDeltaSign = 1.0;
    double rawParallelCoefficient = 0.0;
    double admittedParallelCoefficient = 0.0;
    double perpendicularCoefficient = 0.0;
    double perpendicularSignWeightMultiplier = 1.0;
    double parallelSignWeightMultiplier = 1.0;
    double decisionConfidenceMultiplier = 1.0;
    double normalConfidenceMultiplier = 1.0;
    double parallelSignPenalty = 0.0;
    double perpendicularSignPenalty = 0.0;
    double parallelDistance = 0.0;
    std::optional<double> signedParallelTarget;
    std::optional<double> signedPerpendicularTarget;
    bool hardParallelSign = false;
    bool hardPerpendicularSign = false;
    bool parallelMagnitudePresent = false;
    bool perpendicularMagnitudePresent = false;
    bool parallelSignPresent = false;
    bool perpendicularSignPresent = false;
    FiberTraceFixedOrientation bpEndpointOrientation =
        FiberTraceFixedOrientation::Mixed;
    std::size_t bpOrientationComponent = 0;
    bool bpEndpointActive = false;
    std::size_t bpPiece = 0;
    std::size_t constraintIndex = 0;
};

struct FiberTraceReferenceBenchmarkCounts {
    std::size_t right = 0;
    std::size_t wrong = 0;
    std::size_t total = 0;
};

enum class FiberTraceReferenceBenchmarkClass : unsigned char {
    PerpendicularMagnitude,
    PerpendicularSign,
    ParallelSameMagnitude,
    ParallelOtherMagnitude,
    ParallelSign,
    Count,
};

enum class FiberTraceReferenceFactorClass : unsigned char {
    PerpendicularMagnitudeNext,
    PerpendicularMagnitudeFar,
    PerpendicularSign,
    ParallelMagnitudeSame,
    ParallelMagnitudeOne,
    ParallelMagnitudeFar,
    ParallelSign,
    Count,
};

[[nodiscard]] const char* fiberTraceReferenceFactorClassName(
    FiberTraceReferenceFactorClass factorClass) noexcept;

struct FiberTraceReferenceClampedFactorConflict {
    std::size_t referenceSource = 0;
    std::size_t bpPiece = 0;
    std::size_t constraintIndex = 0;
    FiberTraceReferenceFactorClass factorClass =
        FiberTraceReferenceFactorClass::PerpendicularMagnitudeNext;
    bool hardViolation = false;
    double predictedDelta = 0.0;
    double targetDelta = 0.0;
    double residual = 0.0;
    double effectiveWeight = 0.0;
    double weightedLoss = 0.0;
};

struct FiberTraceReferenceConstraintFactorConflict {
    std::size_t constraintIndex = 0;
    std::size_t sourceA = 0;
    std::size_t sourceB = 0;
    FiberTraceReferenceFactorClass factorClass =
        FiberTraceReferenceFactorClass::PerpendicularMagnitudeNext;
    bool hardViolation = false;
    double predictedDelta = 0.0;
    double targetDelta = 0.0;
    double residual = 0.0;
    double effectiveWeight = 0.0;
    double weightedLoss = 0.0;
};

[[nodiscard]] const char* fiberTraceReferenceBenchmarkClassName(
    FiberTraceReferenceBenchmarkClass benchmarkClass) noexcept;

struct FiberTraceReferenceGaugeCalibration {
    std::size_t integerGauge = 0;
    double offset = 0.0;
    std::size_t estimateVotes = 0;
    std::size_t exactMatches = 0;
};

struct FiberTraceReferenceRawWindingEstimate {
    std::size_t referenceSource = 0;
    std::size_t integerGauge = 0;
    double winding = 0.0;
    std::size_t observations = 0;
    double admittedCoefficient = 0.0;
};

struct FiberTraceReferenceSourceBenchmark {
    std::array<
        FiberTraceReferenceBenchmarkCounts,
        static_cast<std::size_t>(FiberTraceReferenceBenchmarkClass::Count)>
        classes;
    FiberTraceReferenceBenchmarkCounts sum;
    std::optional<double> rawEstimatedWinding;
    std::optional<double> estimatedWinding;
    std::optional<std::size_t> estimatedIntegerGauge;
    std::optional<std::size_t> estimatedOrientationComponent;
    std::size_t estimatedWindingSupport = 0;
    std::size_t estimatedWindingObservations = 0;
    std::optional<bool> estimatedParityMatches;
};

struct FiberTraceReferenceWindingBenchmark {
    double tolerance = 0.5;
    int globalSign = 1;
    std::vector<FiberTraceReferenceGaugeCalibration> gauges;
    std::array<
        FiberTraceReferenceBenchmarkCounts,
        static_cast<std::size_t>(FiberTraceReferenceBenchmarkClass::Count)>
        classes;
    std::vector<FiberTraceReferenceSourceBenchmark> references;
    FiberTraceReferenceBenchmarkCounts sum;
};

enum class FiberTraceReferenceOrientationRelation : unsigned char {
    Perpendicular,
    Parallel,
    Count,
};

struct FiberTraceReferenceOrientationCounts {
    std::size_t right = 0;
    std::size_t wrong = 0;

    [[nodiscard]] std::size_t total() const noexcept
    {
        return right + wrong;
    }
};

struct FiberTraceReferenceOrientationSourceBenchmark {
    std::array<
        FiberTraceReferenceOrientationCounts,
        static_cast<std::size_t>(
            FiberTraceReferenceOrientationRelation::Count)>
        relations;
    FiberTraceReferenceOrientationCounts sum;
};

struct FiberTraceReferenceOrientationComponentCalibration {
    std::size_t component = 0;
    bool evenReferenceIsHorizontal = true;
    std::size_t evenHorizontalRight = 0;
    std::size_t evenVerticalRight = 0;
};

struct FiberTraceReferenceOrientationBenchmark {
    std::vector<FiberTraceReferenceOrientationComponentCalibration>
        components;
    std::array<
        FiberTraceReferenceOrientationCounts,
        static_cast<std::size_t>(
            FiberTraceReferenceOrientationRelation::Count)>
        relations;
    std::vector<FiberTraceReferenceOrientationSourceBenchmark> references;
    FiberTraceReferenceOrientationCounts sum;
    std::size_t excludedInactive = 0;
};

[[nodiscard]] std::optional<bool> fiberTraceReferenceEstimateParityMatches(
    std::size_t referenceSource,
    std::optional<double> estimatedWinding);

[[nodiscard]] std::optional<int> fiberTraceReferenceOutputWinding(
    std::size_t referenceSource,
    const FiberTraceReferenceSourceBenchmark& reference,
    const FiberTraceReferenceOrientationBenchmark& orientation,
    std::span<const int> componentPhaseSigns,
    double phaseMagnitude,
    int outputOffset);

[[nodiscard]] FiberTraceReferenceOrientationBenchmark
benchmarkFiberTraceReferenceOrientations(
    std::span<const FiberTraceReferenceWindingObservation> observations);

[[nodiscard]] FiberTraceReferenceConstraintDiagnosticReport
makeFiberTraceReferenceConstraintDiagnosticReport(
    const FiberTraceConstraintReport& constraints,
    const std::vector<std::size_t>& sourceIdsByTrace,
    const FiberTraceReferenceWindingBenchmark& calibration);

struct FiberTraceReferenceScaleFit {
    std::size_t observations = 0;
    std::size_t admittedObservations = 0;
    std::size_t informativeObservations = 0;
    double effectiveWeight = 0.0;
    double reciprocalScaleWeight = 0.0;
    double unitScaleLoss = 0.0;
    std::optional<double> fittedScale;
    double fittedLoss = 0.0;
    bool atLowerBound = false;
    bool atUpperBound = false;
};

struct FiberTraceReferenceScaleCalibrationReport {
    double minimumScale = 0.5;
    double maximumScale = 2.0;
    FiberTraceReferenceScaleFit rawPerpendicular;
    FiberTraceReferenceScaleFit canonicalPerpendicular;
    FiberTraceReferenceScaleFit rawParallel;
    FiberTraceReferenceScaleFit canonicalParallel;
    FiberTraceReferenceScaleFit rawAll;
    FiberTraceReferenceScaleFit canonicalAll;
    std::array<
        FiberTraceReferenceScaleFit,
        static_cast<std::size_t>(FiberTraceReferenceConstraintGroup::Count)>
        rawGroups;
    std::array<
        FiberTraceReferenceScaleFit,
        static_cast<std::size_t>(FiberTraceReferenceConstraintGroup::Count)>
        canonicalGroups;
};

struct FiberTraceReferencePhaseFit {
    int windingDirection = 1;
    bool evenReferenceIsHorizontal = true;
    std::size_t totalRows = 0;
    std::size_t identifyingRows = 0;
    std::size_t usedRows = 0;
    std::size_t perpendicularSameParityRows = 0;
    std::size_t parallelSameParityRows = 0;
    std::size_t parallelOppositeParityRows = 0;
    double effectiveWeight = 0.0;
    double lossAtZero = 0.0;
    double lossAtHalf = 0.0;
    std::optional<double> fittedPhase;
    double fittedLoss = 0.0;
    std::size_t fittedSignDisagreements = 0;
};

struct FiberTraceReferencePhaseCalibrationReport {
    double measurementScale = 1.0;
    std::array<FiberTraceReferencePhaseFit, 4> gauges;
    std::optional<std::size_t> selectedGauge;
};

struct FiberTraceReferenceStepStatistics {
    std::size_t observations = 0;
    double minimum = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double maximum = 0.0;
};

struct FiberTraceReferenceStepStatisticsReport {
    // [perpendicular=0/parallel=1][owner parity][target parity][distance band]
    std::array<std::array<std::array<std::array<
        FiberTraceReferenceStepStatistics, 3>, 2>, 2>, 2> groups;
};

[[nodiscard]] FiberTraceReferenceScaleCalibrationReport
calibrateFiberTraceReferenceConstraintScales(
    const FiberTraceReferenceConstraintDiagnosticReport& reference,
    std::span<const FiberTraceWindingFactorDiagnostic> factors,
    double minimumScale = 0.5,
    double maximumScale = 2.0);

[[nodiscard]] FiberTraceReferencePhaseCalibrationReport
calibrateFiberTraceReferenceConstraintPhase(
    const FiberTraceReferenceConstraintDiagnosticReport& reference,
    double measurementScale);

[[nodiscard]] FiberTraceReferenceStepStatisticsReport
summarizeFiberTraceReferenceConstraintSteps(
    const FiberTraceReferenceConstraintDiagnosticReport& reference);

struct FiberTraceReferenceConstraintGroupDiagnostic {
    std::size_t observations = 0;
    double rawCoefficient = 0.0;
    double admittedCoefficient = 0.0;
    std::size_t truthHardViolations = 0;
    double truthLoss = 0.0;
    std::optional<double> preferredWinding;
    std::size_t preferredHardViolations = 0;
    double preferredLoss = 0.0;
};

struct FiberTraceReferenceSourceConstraintGroups {
    std::array<
        FiberTraceReferenceConstraintGroupDiagnostic,
        static_cast<std::size_t>(FiberTraceReferenceConstraintGroup::Count)>
        groups;
    FiberTraceReferenceConstraintGroupDiagnostic all;
};

[[nodiscard]] FiberTraceReferenceWindingObservation makeFiberTraceReferenceWindingObservation(
    const FiberTraceConstraint& constraint, bool referenceIsEndpointA, double virtualReferenceWinding, std::size_t bpPiece, const FiberTraceInterleavedWindingReport& winding,
    const FiberTraceWindingBeliefPropagationConfig& config = {});

[[nodiscard]] FiberTraceReferenceWindingBenchmark calibrateFiberTraceReferenceWindings(
    std::span<const FiberTraceReferenceWindingObservation> observations, double tolerance = 0.5);

[[nodiscard]] std::vector<FiberTraceReferenceClampedFactorConflict>
diagnoseFiberTraceReferenceClampedConflicts(
    std::span<const FiberTraceReferenceWindingObservation> observations,
    const FiberTraceReferenceWindingBenchmark& calibration);

[[nodiscard]] std::vector<FiberTraceReferenceConstraintFactorConflict>
diagnoseFiberTraceReferenceConstraintConflicts(
    const FiberTraceConstraintReport& constraints,
    std::span<const std::size_t> sourceIdsByTrace,
    std::span<const FiberTraceWindingFactorDiagnostic> factors,
    int globalSign);

[[nodiscard]] std::vector<FiberTraceReferenceRawWindingEstimate>
inferFiberTraceReferenceRawWindings(
    std::span<const FiberTraceReferenceWindingObservation> observations);

[[nodiscard]] std::vector<FiberTraceReferenceSourceConstraintGroups>
summarizeFiberTraceReferenceConstraintGroups(
    std::span<const FiberTraceReferenceWindingObservation> observations,
    const FiberTraceReferenceWindingBenchmark& calibration);

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
    std::span<const FiberTraceFixedOrientation> fixedOrientations = {},
    std::span<const FiberTraceFixedWindingState> fixedStates = {},
    std::span<const FiberTraceJointGridInitialState> initialStates = {},
    FiberTraceJointGridInitializationMode initializationMode =
        FiberTraceJointGridInitializationMode::ConditionedMessages);

}  // namespace vc::fiber_tracer
