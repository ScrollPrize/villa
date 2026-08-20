#pragma once

#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

struct LoadedFiberAnchorArtifact {
    FiberAnchorExtractionReport report;
    FiberAnchorArtifactInfo artifact;
};

struct FiberletAnchorId {
    std::array<size_t, 3> cellZYX{0, 0, 0};
    size_t componentIndex = 0;

    auto operator<=>(const FiberletAnchorId&) const = default;
};

struct FiberletPathConfig {
    int cellRadius = 4;
    float neighborhoodMarginCells = 0.5F;
    float longitudinalStepPredictionVoxels = 2.0F;
    float transverseStepPredictionVoxels = 0.5F;
    float maximumEndpointAngleDegrees = 45.0F;
    float maximumPredictionDeviationDegrees = 25.0F;
    float corridorRadiusPredictionVoxels = 0.0F;
    float invalidPredictionCostPerVoxel = 4.0F;
    float smoothnessWeight = 2.0F;
    float smoothnessNormalWeight = 0.1F;
    float smoothnessTangentWeight = 10.0F;
    float smoothnessFreeAngleDegrees = 0.0F;
    int samplingBatchCoordinates = 65536;
    int parallelThreads = 1;
};

struct FiberletPathCost {
    float invalidPrediction = 0.0F;
    float alignment = 0.0F;
    float isotropicSmoothness = 0.0F;
    float tangentSmoothness = 0.0F;
    float normalSmoothness = 0.0F;

    [[nodiscard]] float total() const noexcept;
    FiberletPathCost& operator+=(const FiberletPathCost& other) noexcept;
};

struct FiberletPredictionSample {
    cv::Vec3f direction{0.0F, 0.0F, 0.0F};
    float presence = 0.0F;
    bool valid = false;
    bool presenceValid = false;
};

struct FiberletCandidateResult {
    FiberletAnchorId start;
    FiberletAnchorId target;
    cv::Vec3f startPositionPredictionXYZ{0.0F, 0.0F, 0.0F};
    cv::Vec3f targetPositionPredictionXYZ{0.0F, 0.0F, 0.0F};
    cv::Vec3f startAxisXYZ{1.0F, 0.0F, 0.0F};
    cv::Vec3f targetAxisXYZ{1.0F, 0.0F, 0.0F};
    FiberletPredictionSample startPrediction;
    FiberletPredictionSample targetPrediction;
    cv::Vec3f startNormalXYZ{0.0F, 0.0F, 0.0F};
    cv::Vec3f targetNormalXYZ{0.0F, 0.0F, 0.0F};
    bool startNormalValid = false;
    bool targetNormalValid = false;
    bool searched = false;
    bool scoreValid = false;
    bool success = false;
    std::string reason;
    FiberletPathCost cost;
    // Exact selected transverse lattice coordinate for every interior DP
    // layer. This is the persistence geometry; expanded XYZ is an adapter.
    std::vector<std::array<std::int16_t, 2>> routeLatticeUV;
    std::vector<cv::Vec3f> pointsPredictionXYZ;
};

struct FiberletScoreStatistics {
    size_t count = 0;
    std::optional<float> minimum;
    std::optional<float> mean;
    std::optional<float> maximum;
};

struct FiberletPathStatistics {
    size_t anchors = 0;
    size_t candidates = 0;
    size_t preDpRejected = 0;
    size_t dpSearched = 0;
    size_t searchedUnscored = 0;
    size_t scored = 0;
    size_t accepted = 0;
    size_t unscored = 0;
    FiberletScoreStatistics allScores;
    FiberletScoreStatistics acceptedScores;
    FiberletScoreStatistics acceptedLossDensities;
};

struct FiberletPathVisualMetric {
    size_t candidateIndex = 0;
    float pathLengthPredictionVoxels = 0.0F;
    float totalLoss = 0.0F;
    float lossPerPredictionVoxel = 0.0F;
    float relativeQuality = 0.0F;
};

struct FiberletPathVisualReport {
    std::vector<FiberletPathVisualMetric> paths;
    std::optional<float> minimumLossPerPredictionVoxel;
    std::optional<float> maximumLossPerPredictionVoxel;
};

struct FiberPresenceSlicePixel {
    std::array<size_t, 3> indexZYX{0, 0, 0};
    float presence = 0.0F;
};

struct FiberPresenceSlice {
    std::string name;
    std::array<size_t, 2> varyingAxesXYZ{0, 1};
    size_t fixedAxisXYZ = 2;
    size_t fixedIndex = 0;
    size_t width = 0;
    size_t height = 0;
    std::vector<FiberPresenceSlicePixel> pixels;
};

struct FiberPresenceSliceReport {
    FiberAnchorCrop cropPredictionXYZ;
    std::vector<FiberPresenceSlice> planes;

    [[nodiscard]] size_t pixelCount() const noexcept;
};

using FiberStoredPresenceBatchSampler =
    std::function<void(const std::vector<std::array<size_t, 3>>& indicesZYX, int parallelThreads, std::vector<FiberStoredPresenceSample>& samples)>;

struct FiberletPathDiagnostics {
    size_t occupiedAnchors = 0;
    size_t neighborhoodOffsets = 0;
    size_t neighborhoodTargetsOutOfGrid = 0;
    size_t generatedPairs = 0;
    size_t zeroLengthPairs = 0;
    size_t axisRejectedPairs = 0;
    size_t searchedPairs = 0;
    size_t successfulPaths = 0;
    size_t noPathPairs = 0;
};

struct FiberletPathReport {
    FiberPredictionGridInfo grid;
    int anchorCellSizePredictionVoxels = 0;
    FiberletPathConfig config;
    FiberletPathDiagnostics diagnostics;
    std::vector<FiberletCandidateResult> candidates;
    size_t sampledVoxels = 0;
    size_t peakCoordinateBatchVoxels = 0;
    size_t samplingCoordinateBatches = 0;
    size_t predictionSamplingCalls = 0;
    size_t normalSamplingCalls = 0;
    size_t evaluatedDpNodes = 0;
    size_t preparedCandidates = 0;
    size_t preparedGeometryBytes = 0;
    size_t peakSearchTransientBytes = 0;
    size_t estimatedPeakOwnedBytes = 0;
    size_t candidateGenerationWorkers = 0;
    size_t candidateWorkers = 0;
    size_t candidatePointPredicateCalls = 0;
    size_t latticeNodePositions = 0;
    size_t corridorSegmentTests = 0;
    size_t corridorAcceptedNodes = 0;
    size_t nodePointPredicateCalls = 0;
    size_t retainedSearchNodes = 0;
    size_t interpolationCornerInsertions = 0;
    size_t cornerWorkerUniqueVoxels = 0;
    size_t cornerWorkerPages = 0;
    size_t cornerPageDirectoryProbes = 0;
    size_t cornerSamePageHits = 0;
    size_t cornerCachedPageHits = 0;
    size_t cornerMergedPages = 0;
    size_t interpolatedScoringPoints = 0;
    size_t endpointScoringInterpolations = 0;
    size_t lazyNodeScoringRequests = 0;
    size_t lazyNodeScoringMaterializations = 0;
    size_t lazyNodeScoringCacheHits = 0;
    size_t scoringPageCount = 0;
    size_t scoringPageSlots = 0;
    size_t scoringPageDirectoryProbes = 0;
    size_t interpolationProfiledPoints = 0;
    size_t interpolationProfiledCorners = 0;
    size_t interpolationProfiledPredictionIdentical = 0;
    size_t interpolationProfiledNormalIdentical = 0;
    size_t interpolationProfiledPredictionPrincipalSolves = 0;
    size_t interpolationProfiledNormalPrincipalSolves = 0;
    size_t interpolationPredictionClosedFormResolutions = 0;
    size_t interpolationNormalClosedFormResolutions = 0;
    size_t interpolationPredictionIterativeFallbacks = 0;
    size_t interpolationNormalIterativeFallbacks = 0;
    size_t dpNodeIndexEntries = 0;
    size_t dpNodeIndexSlots = 0;
    size_t dpPreparedNodes = 0;
    size_t dpMaximumPreparedNodeBytes = 0;
    size_t dpMaximumLazyCacheIndexBytes = 0;
    size_t dpMaximumDirectIndexBytes = 0;
    size_t dpMaximumStateBytes = 0;
    size_t dpSharedScoringBytes = 0;
    size_t dpReachedNodes = 0;
    size_t dpGeneratedEdges = 0;
    size_t dpValidEdges = 0;
    size_t dpReusedEdges = 0;
    size_t dpTransitionLookups = 0;
    size_t dpReachedStateVisits = 0;
    size_t dpRelaxations = 0;
    double candidateGenerationSeconds = 0.0;
    double candidateGenerationCpuSeconds = 0.0;
    double preparationSeconds = 0.0;
    double preparationCpuSeconds = 0.0;
    double preparationGeometryWorkSeconds = 0.0;
    double preparationNodeEnumerationWorkSeconds = 0.0;
    double preparationCornerCollectionWorkSeconds = 0.0;
    double cornerMergeSeconds = 0.0;
    double cornerMergeCpuSeconds = 0.0;
    double predictionSamplingSeconds = 0.0;
    double predictionSamplingCpuSeconds = 0.0;
    double normalSamplingSeconds = 0.0;
    double normalSamplingCpuSeconds = 0.0;
    double samplingMaterializationSeconds = 0.0;
    double samplingMaterializationCpuSeconds = 0.0;
    double scoringIndexSeconds = 0.0;
    double scoringIndexCpuSeconds = 0.0;
    double scoringPreparationSeconds = 0.0;
    double scoringPreparationCpuSeconds = 0.0;
    double interpolationMaterializationSeconds = 0.0;
    double interpolationMaterializationCpuSeconds = 0.0;
    double interpolationProfiledLookupSeconds = 0.0;
    double interpolationProfiledPredictionCornerSeconds = 0.0;
    double interpolationProfiledNormalCornerSeconds = 0.0;
    double interpolationProfiledPredictionResolveSeconds = 0.0;
    double interpolationProfiledNormalResolveSeconds = 0.0;
    double searchSeconds = 0.0;
    double searchCpuSeconds = 0.0;
    double searchNodeIndexWorkSeconds = 0.0;
    double searchNodePreparationWorkSeconds = 0.0;
    double searchDpWorkSeconds = 0.0;
    double elapsedSeconds = 0.0;
    double elapsedCpuSeconds = 0.0;
};

struct FiberletArtifactInfo {
    std::string fiberManifestLocator;
    std::string fiberManifestContentHash;
    std::string normalManifestLocator;
    std::string normalManifestContentHash;
    std::string anchorArtifactLocator;
    std::string anchorArtifactContentHash;
    std::optional<double> baseVoxelSizeUm;
};

struct FiberletPathProgress {
    std::string phase;
    size_t completed = 0;
    size_t total = 0;
    double elapsedSeconds = 0.0;
};

using FiberletPathProgressCallback = std::function<void(const FiberletPathProgress& progress)>;
using FiberletPointPredicate = std::function<bool(const cv::Vec3f& pointPredictionXYZ)>;
using FiberletCandidatePredicate = std::function<bool(
    const FiberletAnchorId& first,
    const FiberletAnchorId& second)>;
using FiberletSourcePredicate =
    std::function<bool(const FiberletAnchorId& source)>;

[[nodiscard]] std::vector<cv::Vec3f> reconstructFiberletRoutePoints(
    const cv::Vec3f& startPositionPredictionXYZ,
    const cv::Vec3f& startAxisXYZ,
    const cv::Vec3f& targetPositionPredictionXYZ,
    const cv::Vec3f& targetAxisXYZ,
    std::span<const std::array<std::int16_t, 2>> interiorLatticeUV,
    const FiberletPathConfig& config);

void validateFiberletPathConfig(const FiberletPathConfig& config);

[[nodiscard]] LoadedFiberAnchorArtifact loadFiberAnchorArtifact(const std::filesystem::path& path);

[[nodiscard]] std::vector<std::array<int, 3>> fiberletCellNeighborhoodOffsets(int radius, float margin);

[[nodiscard]] FiberletPathReport traceFiberletPaths(
    const LoadedFiberAnchorArtifact& anchors,
    const FiberPredictionGridInfo& grid,
    const FiberletPathConfig& config,
    const FiberStoredPredictionBatchSampler& predictionSampler,
    const vc::lasagna::NormalSampler& normalSampler,
    const FiberletPathProgressCallback& progressCallback = {},
    const FiberletPointPredicate& pointPredicate = {},
    const FiberletCandidatePredicate& candidatePredicate = {},
    const FiberletSourcePredicate& sourcePredicate = {});

[[nodiscard]] FiberletPathStatistics fiberletPathStatistics(const FiberletPathReport& report);

[[nodiscard]] FiberletPathVisualReport fiberletPathVisualMetrics(const FiberletPathReport& report);

[[nodiscard]] FiberAnchorCrop fiberAnchorCellCoverageCrop(const LoadedFiberAnchorArtifact& anchors);

[[nodiscard]] FiberPresenceSliceReport sampleFiberPresenceSlices(
    const FiberAnchorCrop& cropPredictionXYZ, const FiberPredictionGridInfo& grid, const FiberStoredPresenceBatchSampler& presenceSampler, int parallelThreads);

[[nodiscard]] nlohmann::json fiberletPathReportJson(const FiberletPathReport& report, const FiberletArtifactInfo& artifact);

[[nodiscard]] std::string fiberletPathReportObj(const FiberletPathReport& report);

void writeFiberletPathArtifacts(const std::filesystem::path& outputDirectory, const FiberletPathReport& report, const FiberletArtifactInfo& artifact);

void writeFiberPresenceSliceArtifacts(const std::filesystem::path& outputDirectory, const FiberPresenceSliceReport& report, const FiberPredictionGridInfo& grid);

void removeFiberPresenceSliceArtifacts(const std::filesystem::path& outputDirectory);

#ifdef VC_TESTING
namespace testing
{

struct FiberletCorridorContainmentDebug {
    bool inside = false;
    size_t segmentTests = 0;
};

[[nodiscard]] FiberletCorridorContainmentDebug debugFiberletCorridorContains(
    const cv::Vec3f& point,
    const std::vector<cv::Vec3f>& reference,
    float radius,
    std::optional<size_t> adjacentSegment = std::nullopt);

[[nodiscard]] std::vector<std::array<int64_t, 3>>
debugFinalizeFiberletCornerSets(
    const std::vector<std::vector<std::array<int64_t, 3>>>& cornerSets);

}  // namespace testing
#endif

}  // namespace vc::fiber_tracer
