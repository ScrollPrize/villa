#pragma once

#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <array>
#include <compare>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
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
    double neighborhoodMarginCells = 0.5;
    double longitudinalStepPredictionVoxels = 2.0;
    double transverseStepPredictionVoxels = 0.5;
    double maximumEndpointAngleDegrees = 45.0;
    double maximumPredictionDeviationDegrees = 25.0;
    double corridorRadiusPredictionVoxels = 0.0;
    double invalidPredictionCostPerVoxel = 4.0;
    double smoothnessWeight = 2.0;
    double smoothnessNormalWeight = 0.1;
    double smoothnessTangentWeight = 10.0;
    double smoothnessFreeAngleDegrees = 0.0;
    int parallelThreads = 1;
};

struct FiberletPathCost {
    double invalidPrediction = 0.0;
    double alignment = 0.0;
    double isotropicSmoothness = 0.0;
    double tangentSmoothness = 0.0;
    double normalSmoothness = 0.0;

    [[nodiscard]] double total() const noexcept;
    FiberletPathCost& operator+=(const FiberletPathCost& other) noexcept;
};

struct FiberletCandidateResult {
    FiberletAnchorId start;
    FiberletAnchorId target;
    cv::Vec3d startPositionPredictionXYZ{0.0, 0.0, 0.0};
    cv::Vec3d targetPositionPredictionXYZ{0.0, 0.0, 0.0};
    cv::Vec3d startAxisXYZ{1.0, 0.0, 0.0};
    cv::Vec3d targetAxisXYZ{1.0, 0.0, 0.0};
    FiberStoredPredictionSample startPrediction;
    FiberStoredPredictionSample targetPrediction;
    cv::Vec3d startNormalXYZ{0.0, 0.0, 0.0};
    cv::Vec3d targetNormalXYZ{0.0, 0.0, 0.0};
    bool startNormalValid = false;
    bool targetNormalValid = false;
    bool searched = false;
    bool scoreValid = false;
    bool success = false;
    std::string reason;
    FiberletPathCost cost;
    std::vector<cv::Vec3d> pointsPredictionXYZ;
};

struct FiberletScoreStatistics {
    size_t count = 0;
    std::optional<double> minimum;
    std::optional<double> mean;
    std::optional<double> maximum;
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
    double pathLengthPredictionVoxels = 0.0;
    double totalLoss = 0.0;
    double lossPerPredictionVoxel = 0.0;
    double relativeQuality = 0.0;
};

struct FiberletPathVisualReport {
    std::vector<FiberletPathVisualMetric> paths;
    std::optional<double> minimumLossPerPredictionVoxel;
    std::optional<double> maximumLossPerPredictionVoxel;
};

struct FiberPresenceSlicePixel {
    std::array<size_t, 3> indexZYX{0, 0, 0};
    double presence = 0.0;
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
    size_t preloadedVoxels = 0;
    size_t evaluatedDpNodes = 0;
    size_t estimatedPreloadBytes = 0;
    size_t candidateWorkers = 0;
    double candidateGenerationSeconds = 0.0;
    double preloadSeconds = 0.0;
    double searchSeconds = 0.0;
    double elapsedSeconds = 0.0;
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
    size_t completed = 0;
    size_t total = 0;
    double elapsedSeconds = 0.0;
};

using FiberletPathProgressCallback = std::function<void(const FiberletPathProgress& progress)>;
using FiberletPointPredicate = std::function<bool(const cv::Vec3d& pointPredictionXYZ)>;

void validateFiberletPathConfig(const FiberletPathConfig& config);

[[nodiscard]] LoadedFiberAnchorArtifact loadFiberAnchorArtifact(const std::filesystem::path& path);

[[nodiscard]] std::vector<std::array<int, 3>> fiberletCellNeighborhoodOffsets(int radius, double margin);

[[nodiscard]] FiberletPathReport traceFiberletPaths(
    const LoadedFiberAnchorArtifact& anchors,
    const FiberPredictionGridInfo& grid,
    const FiberletPathConfig& config,
    const FiberStoredPredictionBatchSampler& predictionSampler,
    const vc::lasagna::NormalSampler& normalSampler,
    const FiberletPathProgressCallback& progressCallback = {},
    const FiberletPointPredicate& pointPredicate = {});

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

}  // namespace vc::fiber_tracer
