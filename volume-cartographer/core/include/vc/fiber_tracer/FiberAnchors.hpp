#pragma once

#include "vc/fiber_tracer/FiberTrace.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <filesystem>
#include <optional>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>
#include <nlohmann/json.hpp>

namespace vc::fiber_tracer
{

struct FiberAnchorObservation {
    cv::Vec3f positionPredictionXYZ{0.0F, 0.0F, 0.0F};
    cv::Vec3f direction{0.0F, 0.0F, 0.0F};
    float presence = 0.0F;
    bool valid = false;
    cv::Vec3f presenceGradientPredictionXYZ{0.0F, 0.0F, 0.0F};
    bool presenceGradientValid = false;
};

struct FiberAnchorConfig {
    int cellSizePredictionVoxels = 4;
    float gaussianSigmaPredictionVoxels = 2.0F;
    float peakSigmaPredictionVoxels = 1.5F;
    float peakAxialSigmaPredictionVoxels = 6.0F;
    float peakGridStepPredictionVoxels = 0.5F;
    float peakGradientWeight = 1.0F;
    float peakGradientReliabilityScale = 0.05F;
    float gaussianCutoffSigmas = 3.0F;
    float localWindowRadiusPredictionVoxels = 4.0F;
    float axialSupportHalfWidthPredictionVoxels = 6.0F;
    float positionConvergenceTolerancePredictionVoxels = 1.0e-4F;
    float nmsMaximumAngleDegrees = 10.0F;
    float nmsTransverseRadiusPredictionVoxels = 2.0F;
    float nmsLongitudinalRadiusPredictionVoxels = 1.0F;
    float observationPresenceFloor = 0.05F;
    float minimumAlignedSupport = 0.05F;
    float robustMaximumTrimMassFraction = 0.20F;
    float robustMadMultiplier = 3.0F;
    float robustMinimumAngleDegrees = 5.0F;
    float mergeMaximumAngleDegrees = 10.0F;
    float mergeMaximumAbsoluteObjectiveLoss = 0.01F;
    float mergeMaximumRelativeObjectiveLoss = 0.05F;
    size_t maximumSeedCount = 8;
    int maximumIterations = 1;
    bool verifySpatialObjective = false;
    float convergenceTolerance = 1.0e-6F;
    size_t maximumConcurrentSampleBytes =
        2ULL * 1024ULL * 1024ULL * 1024ULL;
    int parallelThreads = 1;
};

struct FiberAnchorQuadraticPeakOffset {
    float firstGridSteps = 0.0F;
    float secondGridSteps = 0.0F;
};

// Samples are indexed [first + 1][second + 1] for offsets in {-1, 0, 1}.
[[nodiscard]] std::optional<FiberAnchorQuadraticPeakOffset>
fitFiberAnchorQuadraticPeak(
    const std::array<std::array<float, 3>, 3>& response);

struct FiberAnchorCrop {
    std::array<size_t, 3> originXYZ{0, 0, 0};
    std::array<size_t, 3> sizeXYZ{0, 0, 0};
};

struct FiberAnchor {
    std::array<size_t, 3> cellZYX{0, 0, 0};
    cv::Vec3f positionPredictionXYZ{0.0F, 0.0F, 0.0F};
    cv::Vec3f axisXYZ{1.0F, 0.0F, 0.0F};
    float alignedSupport = 0.0F;
    float directionalCoherence = 0.0F;
    float refinementScore = 0.0F;
    size_t refinementIterations = 0;
};

struct FiberAnchorDiagnosticMetrics {
    std::optional<size_t> assignedObservationCount;
    std::optional<float> objectiveContribution;
    std::optional<float> alignedSupport;
    std::optional<float> directionalCoherence;
    std::optional<float> refinementScore;
    std::optional<size_t> refinementIterations;
};

struct FiberAnchorDiagnosticSuppressor {
    std::array<size_t, 3> cellZYX{0, 0, 0};
    size_t candidateId = 0;
    bool externalContext = false;
    float alignedSupport = 0.0F;
    float directionalCoherence = 0.0F;
};

struct FiberAnchorDiagnosticTransition {
    std::string outcome;
    std::optional<std::string> reason;
    std::optional<size_t> successorId;
    std::optional<float> testedValue;
    std::optional<float> threshold;
    std::optional<FiberAnchorDiagnosticSuppressor> suppressor;
};

struct FiberAnchorDiagnosticRecord {
    std::array<size_t, 3> cellZYX{0, 0, 0};
    size_t candidateId = 0;
    std::vector<size_t> parentIds;
    std::optional<FiberAnchor> anchor;
    // Transient benchmark provenance. Deliberately omitted from artifacts.
    std::optional<cv::Vec3f> discretePeakPositionPredictionXYZ;
    std::optional<cv::Vec3f> separablePeakPositionPredictionXYZ;
    std::optional<cv::Vec3f> jointPeakPositionPredictionXYZ;
    FiberAnchorDiagnosticMetrics metrics;
    FiberAnchorDiagnosticTransition transition;
};

enum class FiberAnchorDiagnosticStage : size_t {
    Initialized = 0,
    Refined,
    Support,
    Selection,
    Nms,
    Count,
};

inline constexpr size_t kFiberAnchorDiagnosticStageCount =
    static_cast<size_t>(FiberAnchorDiagnosticStage::Count);

[[nodiscard]] const char* fiberAnchorDiagnosticStageName(
    FiberAnchorDiagnosticStage stage);

struct FiberAnchorComponent {
    FiberAnchor anchor;
    std::optional<cv::Vec3f> discretePeakPositionPredictionXYZ;
    std::optional<cv::Vec3f> separablePeakPositionPredictionXYZ;
    std::optional<cv::Vec3f> jointPeakPositionPredictionXYZ;
    bool retained = false;
    std::string rejectionReason;
    size_t assignedObservationCount = 0;
    size_t diagnosticId = std::numeric_limits<size_t>::max();
    std::vector<size_t> diagnosticParentIds;
    bool retainedAfterSupport = false;
    bool retainedAfterSelection = false;
    std::optional<float> selectionTestedValue;
    std::optional<float> selectionThreshold;
    std::optional<FiberAnchorDiagnosticSuppressor> nmsSuppressor;
    bool removedDuringRobustRefinement = false;
};

struct FiberAnchorMergeEvaluation {
    float angleDegrees = 0.0F;
    float jointObjective = 0.0F;
    float splitObjective = 0.0F;
    float objectiveLoss = 0.0F;
    float allowedObjectiveLoss = 0.0F;
    bool merged = false;
};

struct FiberCellAnchorResult {
    std::array<size_t, 3> cellZYX{0, 0, 0};
    std::array<FiberAnchorComponent, 2> components;
    std::optional<FiberAnchorMergeEvaluation> mergeEvaluation;
    float objective = 0.0F;
    size_t retainedAnchorCount = 0;
    std::array<FiberAnchorDiagnosticRecord, 2> initializedDiagnostics;
};

struct FiberAnchorExtractionDiagnostics {
    size_t totalCells = 0;
    size_t zeroAnchorCells = 0;
    size_t oneAnchorCells = 0;
    size_t twoAnchorCells = 0;
    size_t emptyComponents = 0;
    size_t degenerateComponents = 0;
    size_t belowSupportComponents = 0;
    size_t mergedComponentPairs = 0;
    size_t nmsSuppressedComponents = 0;
    size_t outsideSelectionComponents = 0;
};

struct FiberAnchorResidualSample {
    float residual = 0.0F;
    float mass = 0.0F;
};

struct FiberAnchorRobustCutoff {
    float cutoffResidual = 1.0F;
    float totalMass = 0.0F;
    float candidateTrimmedMass = 0.0F;
    float trimmedMass = 0.0F;
    float retainedMass = 0.0F;
    bool detectedOutliers = false;
};

[[nodiscard]] FiberAnchorRobustCutoff selectFiberAnchorRobustCutoff(
    const std::vector<FiberAnchorResidualSample>& samples,
    float maximumTrimMassFraction,
    float madMultiplier,
    float minimumAngleDegrees);

[[nodiscard]] std::vector<float> fiberAnchorSpatialBacktrackingFractions(
    float maximumDisplacementPredictionVoxels,
    float targetStepPredictionVoxels,
    int maximumHalvings = 8);

struct FiberAnchorFitProfile {
    size_t invocations = 0;
    size_t nonemptyCells = 0;
    size_t weightedObservations = 0;
    size_t ownedDiscoveryObservationVisits = 0;
    size_t ownedInitializationObservationVisits = 0;
    size_t avoidedOwnedSupportObservationVisits = 0;
    size_t seeds = 0;
    size_t seedGenerationObservationVisits = 0;
    size_t seedPairs = 0;
    size_t seedPairIterations = 0;
    size_t seedAssignmentObservationVisits = 0;
    size_t seedTensorObservationVisits = 0;
    size_t seedObjectiveObservationVisits = 0;
    size_t initializationObservationVisits = 0;
    size_t localRefinementAttempts = 0;
    size_t localRefinementAcceptedSteps = 0;
    size_t backtrackingEvaluations = 0;
    size_t directCentroidAcceptances = 0;
    size_t robustComponentsWithoutOutliers = 0;
    size_t robustTrimmedComponents = 0;
    size_t robustRemovedNonuniqueComponents = 0;
    size_t robustHardLimitHits = 0;
    size_t spatialCandidatesTested = 0;
    std::array<size_t, 9> spatialCandidatesTestedByDepth{};
    std::array<size_t, 9> spatialCandidatesAcceptedByDepth{};
    float robustCandidateTrimmedMass = 0.0F;
    float robustTrimmedMass = 0.0F;
    float robustRetainedMass = 0.0F;
    size_t localTensorObservationVisits = 0;
    size_t robustAxisProposalCalls = 0;
    size_t robustAxisLogicalObservationVisits = 0;
    size_t robustAxisEligibleObservationVisits = 0;
    size_t robustAxisIndexedObservationVisits = 0;
    size_t robustAxisCutoffObservationVisits = 0;
    size_t robustMembershipProposalCalls = 0;
    size_t robustMembershipLogicalObservationVisits = 0;
    size_t robustMembershipEligibleObservationVisits = 0;
    size_t robustMembershipIndexedObservationVisits = 0;
    size_t robustMembershipCutoffObservationVisits = 0;
    size_t robustProposalBufferInitializations = 0;
    size_t robustProposalInitializedBytes = 0;
    size_t robustEvaluationCopiedBytes = 0;
    size_t robustPreparedObservationRecords = 0;
    size_t robustPreparedObservationRecordBytes = 0;
    size_t localCentroidObservationVisits = 0;
    size_t localCentroidIndexedObservationVisits = 0;
    size_t refinedEvaluationObservationVisits = 0;
    size_t peakComponents = 0;
    size_t peakPreparationObservationVisits = 0;
    size_t peakPreparedResponseObservations = 0;
    size_t peakPreparedEvidenceObservations = 0;
    size_t peakResponseObservationRecordBytes = 0;
    size_t peakEvidenceObservationRecordBytes = 0;
    size_t peakMaximumObservationStorageBytes = 0;
    size_t peakGridResponseRequests = 0;
    size_t peakComputedGridResponses = 0;
    size_t peakAcceptanceResponses = 0;
    size_t peakResponseObservationVisits = 0;
    size_t peakResponseRadialAcceptances = 0;
    size_t peakResponseEvidenceObservationVisits = 0;
    size_t finalEvaluationObservationVisits = 0;
    double setupWorkSeconds = 0.0;
    double seedGenerationWorkSeconds = 0.0;
    double seedPairRefinementWorkSeconds = 0.0;
    double initializationWorkSeconds = 0.0;
    double localRefinementWorkSeconds = 0.0;
    double localTensorProposalWorkSeconds = 0.0;
    double robustAxisProposalWorkSeconds = 0.0;
    double robustMembershipProposalWorkSeconds = 0.0;
    double robustObservationPreparationWorkSeconds = 0.0;
    double localCentroidProposalWorkSeconds = 0.0;
    double localStateEvaluationWorkSeconds = 0.0;
    double peakSearchWorkSeconds = 0.0;
    double finalEvaluationWorkSeconds = 0.0;
};

struct FiberAnchorExtractionProfile {
    size_t selectedCells = 0;
    size_t contextCells = 0;
    size_t workCells = 0;
    size_t tiles = 0;
    size_t samplingPartitions = 0;
    size_t workers = 0;
    size_t predictionSamplerCalls = 0;
    size_t sharedSamplingBatches = 0;
    size_t maximumSamplingBatchVoxels = 0;
    size_t submittedPredictionVoxels = 0;
    size_t uniqueTilePredictionVoxels = 0;
    size_t reusedPredictionVoxels = 0;
    size_t sharedObservationVoxels = 0;
    size_t maximumSharedSampleBytes = 0;
    size_t maximumAccountedLiveBytes = 0;
    size_t candidateObservations = 0;
    size_t retainedObservations = 0;
    size_t supportStencilCells = 0;
    size_t clippedSupportCells = 0;
    size_t gradientAttempts = 0;
    size_t validGradients = 0;
    size_t gradientComputations = 0;
    size_t validGradientComputations = 0;
    size_t retainPredicateCalls = 0;
    size_t fitIterations = 0;
    double setupSeconds = 0.0;
    double tilePlanningSeconds = 0.0;
    double cellProcessingSeconds = 0.0;
    double cellProcessingCpuSeconds = 0.0;
    double sharedSamplingSeconds = 0.0;
    double sharedSamplingCpuSeconds = 0.0;
    double coordinateConstructionWorkSeconds = 0.0;
    double predictionSamplingWorkSeconds = 0.0;
    double sharedObservationConstructionWorkSeconds = 0.0;
    double tileObservationIndexWorkSeconds = 0.0;
    double gradientConstructionWorkSeconds = 0.0;
    double observationConstructionWorkSeconds = 0.0;
    double fittingWorkSeconds = 0.0;
    double partitionP50Seconds = 0.0;
    double partitionP95Seconds = 0.0;
    double partitionMaximumSeconds = 0.0;
    double tilePreparationP50Seconds = 0.0;
    double tilePreparationP95Seconds = 0.0;
    double tilePreparationMaximumSeconds = 0.0;
    double cellProcessingP50Seconds = 0.0;
    double cellProcessingP95Seconds = 0.0;
    double cellProcessingMaximumSeconds = 0.0;
    double selectionSeconds = 0.0;
    double initialDiagnosticsSeconds = 0.0;
    double duplicateSuppressionSeconds = 0.0;
    double finalizationSeconds = 0.0;
    double elapsedCpuSeconds = 0.0;
    FiberAnchorFitProfile fit;
};

struct FiberAnchorExtractionReport {
    FiberPredictionGridInfo grid;
    FiberAnchorConfig config;
    FiberAnchorCrop selectedCrop;
    std::array<size_t, 3> selectedCellBeginZYX{0, 0, 0};
    std::array<size_t, 3> selectedCellEndZYX{0, 0, 0};
    std::vector<std::array<size_t, 3>> selectedCellsZYX;
    FiberAnchorExtractionDiagnostics diagnostics;
    FiberAnchorExtractionProfile profile;
    std::vector<FiberCellAnchorResult> nonEmptyCells;
    std::array<std::vector<FiberAnchorDiagnosticRecord>,
               kFiberAnchorDiagnosticStageCount> diagnosticStages;
    double elapsedSeconds = 0.0;
};

struct FiberAnchorDistanceStatistics {
    size_t count = 0;
    std::optional<float> minimum;
    std::optional<float> mean;
    std::optional<float> median;
    std::optional<float> percentile95;
    std::optional<float> maximum;
};

struct FiberAnchorBenchmarkThreshold {
    float thresholdBaseVoxels = 0.0F;
    size_t anchorHits = 0;
    std::optional<float> anchorHitRate;
    size_t cellHits = 0;
    float cellHitRate = 0.0F;
};

struct FiberAnchorBenchmarkStageReport {
    size_t referenceCells = 0;
    size_t cellsWithRefinedAnchors = 0;
    size_t refinedAnchors = 0;
    FiberAnchorDistanceStatistics anchorDistancesBaseVoxels;
    std::vector<FiberAnchorBenchmarkThreshold> thresholds;
};

struct FiberAnchorBenchmarkReport {
    FiberAnchorBenchmarkStageReport discrete;
    FiberAnchorBenchmarkStageReport separable1d;
    FiberAnchorBenchmarkStageReport joint2d;
};

struct FiberAnchorArtifactInfo {
    std::string sourceLocator;
    std::string manifestContentHash;
    float glyphLengthBaseVoxels = 16.0F;
    std::optional<double> baseVoxelSizeUm;
};

struct FiberAnchorProgress {
    std::string phase;
    size_t completed = 0;
    size_t total = 0;
    double elapsedSeconds = 0.0;
};

using FiberStoredPredictionBatchSampler =
    std::function<void(const std::vector<std::array<size_t, 3>>& indicesZYX, int parallelThreads, std::vector<FiberStoredPredictionSample>& samples)>;

struct FiberAnchorRetainEvaluation {
    bool retained = true;
    std::optional<float> testedValue;
    std::optional<float> threshold;
};

using FiberAnchorRetainPredicate =
    std::function<FiberAnchorRetainEvaluation(const FiberAnchor& anchor)>;
using FiberAnchorProgressCallback =
    std::function<void(const FiberAnchorProgress& progress)>;

void validateFiberAnchorConfig(const FiberAnchorConfig& config);

[[nodiscard]] FiberAnchorCrop fiberAnchorCropFromBaseVoxels(const FiberAnchorCrop& baseCrop, double predictionToBaseScale);

[[nodiscard]] FiberCellAnchorResult fitFiberCellAnchors(
    const std::array<size_t, 3>& cellZYX,
    const std::array<size_t, 3>& cellBeginZYX,
    const std::array<size_t, 3>& cellEndZYX,
    const std::vector<FiberAnchorObservation>& observations,
    const FiberAnchorConfig& config,
    FiberAnchorFitProfile* profile = nullptr);

[[nodiscard]] FiberAnchorExtractionReport extractFiberAnchors(
    const FiberPredictionGridInfo& grid,
    const FiberAnchorConfig& config,
    const FiberStoredPredictionBatchSampler& sampler,
    std::optional<FiberAnchorCrop> crop = std::nullopt,
    const FiberAnchorProgressCallback& progressCallback = {});

[[nodiscard]] FiberAnchorExtractionReport extractFiberAnchorsForCells(
    const FiberPredictionGridInfo& grid,
    const FiberAnchorConfig& config,
    const FiberStoredPredictionBatchSampler& sampler,
    std::vector<std::array<size_t, 3>> cellsZYX,
    const FiberAnchorRetainPredicate& retainPredicate = {},
    const FiberAnchorProgressCallback& progressCallback = {});

[[nodiscard]] FiberAnchorExtractionReport extractRefinedFiberAnchorsForCells(
    const FiberPredictionGridInfo& grid,
    const FiberAnchorConfig& config,
    const FiberStoredPredictionBatchSampler& sampler,
    std::vector<std::array<size_t, 3>> cellsZYX,
    const FiberAnchorProgressCallback& progressCallback = {});

[[nodiscard]] std::vector<std::array<size_t, 3>>
fiberAnchorCellsNearPolyline(
    const std::vector<cv::Vec3d>& referenceLineBase,
    double radiusBaseVoxels,
    const FiberPredictionGridInfo& grid,
    int anchorCellSizePredictionVoxels);

[[nodiscard]] FiberAnchorBenchmarkReport benchmarkRefinedFiberAnchors(
    const FiberAnchorExtractionReport& anchors,
    const std::vector<cv::Vec3d>& referenceLineBase,
    const std::vector<float>& thresholdsBaseVoxels);

void suppressFiberAnchorDuplicates(
    std::vector<FiberCellAnchorResult>& cells,
    const FiberAnchorConfig& config);

[[nodiscard]] nlohmann::json fiberAnchorReportJson(const FiberAnchorExtractionReport& report, const FiberAnchorArtifactInfo& artifact);

[[nodiscard]] nlohmann::json fiberAnchorDiagnosticStageJson(
    const FiberAnchorExtractionReport& report,
    const FiberAnchorArtifactInfo& artifact,
    FiberAnchorDiagnosticStage stage);

[[nodiscard]] std::string fiberAnchorReportObj(const FiberAnchorExtractionReport& report, const FiberAnchorArtifactInfo& artifact);

[[nodiscard]] std::string fiberAnchorCellReportObj(
    const FiberAnchorExtractionReport& report);

void writeFiberAnchorArtifacts(const std::filesystem::path& outputDirectory, const FiberAnchorExtractionReport& report, const FiberAnchorArtifactInfo& artifact);

}  // namespace vc::fiber_tracer
