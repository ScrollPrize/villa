#pragma once

#include "vc/fiber_tracer/FiberAnchors.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <array>
#include <compare>
#include <cstddef>
#include <filesystem>
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
    double shellHalfWidthCells = 0.5;
    double maximumEndpointAngleDegrees = 45.0;
    double corridorRadiusPredictionVoxels = 0.0;
    double presenceWeight = 1.0;
    double directionWeight = 1.0;
    double invalidPredictionCostPerVoxel = 4.0;
    double smoothnessWeight = 2.0;
    double smoothnessNormalWeight = 0.1;
    double smoothnessTangentWeight = 10.0;
    double smoothnessFreeAngleDegrees = 45.0;
    int parallelThreads = 1;
};

struct FiberletPathCost {
    double invalidPrediction = 0.0;
    double presence = 0.0;
    double direction = 0.0;
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
};

struct FiberletPathDiagnostics {
    size_t occupiedAnchors = 0;
    size_t shellOffsets = 0;
    size_t shellTargetsOutOfGrid = 0;
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

void validateFiberletPathConfig(const FiberletPathConfig& config);

[[nodiscard]] LoadedFiberAnchorArtifact loadFiberAnchorArtifact(const std::filesystem::path& path);

[[nodiscard]] std::vector<std::array<int, 3>> fiberletCellShellOffsets(int radius, double halfWidth);

[[nodiscard]] FiberletPathReport traceFiberletPaths(
    const LoadedFiberAnchorArtifact& anchors,
    const FiberPredictionGridInfo& grid,
    const FiberletPathConfig& config,
    const FiberStoredPredictionBatchSampler& predictionSampler,
    const vc::lasagna::NormalSampler& normalSampler);

[[nodiscard]] FiberletPathStatistics fiberletPathStatistics(const FiberletPathReport& report);

[[nodiscard]] nlohmann::json fiberletPathReportJson(const FiberletPathReport& report, const FiberletArtifactInfo& artifact);

[[nodiscard]] std::string fiberletPathReportObj(const FiberletPathReport& report);

void writeFiberletPathArtifacts(const std::filesystem::path& outputDirectory, const FiberletPathReport& report, const FiberletArtifactInfo& artifact);

}  // namespace vc::fiber_tracer
