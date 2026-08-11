#pragma once

#include "vc/fiber_tracer/FiberTrace.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>
#include <nlohmann/json.hpp>

namespace vc::fiber_tracer
{

struct FiberAnchorObservation {
    cv::Vec3d positionPredictionXYZ{0.0, 0.0, 0.0};
    cv::Vec3d direction{0.0, 0.0, 0.0};
    double presence = 0.0;
    bool valid = false;
};

struct FiberAnchorConfig {
    int cellSizePredictionVoxels = 4;
    double gaussianSigmaPredictionVoxels = 2.0;
    double gaussianCutoffSigmas = 3.0;
    double localWindowRadiusPredictionVoxels = 4.0;
    double axialSupportHalfWidthPredictionVoxels = 6.0;
    double positionConvergenceTolerancePredictionVoxels = 1.0e-4;
    double nmsMaximumAngleDegrees = 10.0;
    double nmsLongitudinalRadiusPredictionVoxels = 2.0;
    double observationPresenceFloor = 0.05;
    double minimumAlignedSupport = 0.05;
    double mergeMaximumAngleDegrees = 10.0;
    double mergeMaximumAbsoluteObjectiveLoss = 0.01;
    double mergeMaximumRelativeObjectiveLoss = 0.05;
    size_t maximumSeedCount = 8;
    int maximumIterations = 64;
    double convergenceTolerance = 1.0e-12;
    size_t processingBlockCellSide = 4;
    size_t maximumSampleBlockBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    int parallelThreads = 1;
};

struct FiberAnchorCrop {
    std::array<size_t, 3> originXYZ{0, 0, 0};
    std::array<size_t, 3> sizeXYZ{0, 0, 0};
};

struct FiberAnchor {
    std::array<size_t, 3> cellZYX{0, 0, 0};
    cv::Vec3d positionPredictionXYZ{0.0, 0.0, 0.0};
    cv::Vec3d axisXYZ{1.0, 0.0, 0.0};
    double alignedSupport = 0.0;
    double directionalCoherence = 0.0;
    double refinementScore = 0.0;
    size_t refinementIterations = 0;
};

struct FiberAnchorComponent {
    FiberAnchor anchor;
    bool retained = false;
    std::string rejectionReason;
    size_t assignedObservationCount = 0;
};

struct FiberAnchorMergeEvaluation {
    double angleDegrees = 0.0;
    double jointObjective = 0.0;
    double splitObjective = 0.0;
    double objectiveLoss = 0.0;
    double allowedObjectiveLoss = 0.0;
    bool merged = false;
};

struct FiberCellAnchorResult {
    std::array<size_t, 3> cellZYX{0, 0, 0};
    std::array<FiberAnchorComponent, 2> components;
    std::optional<FiberAnchorMergeEvaluation> mergeEvaluation;
    double objective = 0.0;
    size_t retainedAnchorCount = 0;
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

struct FiberAnchorExtractionReport {
    FiberPredictionGridInfo grid;
    FiberAnchorConfig config;
    FiberAnchorCrop selectedCrop;
    std::array<size_t, 3> selectedCellBeginZYX{0, 0, 0};
    std::array<size_t, 3> selectedCellEndZYX{0, 0, 0};
    std::vector<std::array<size_t, 3>> selectedCellsZYX;
    FiberAnchorExtractionDiagnostics diagnostics;
    std::vector<FiberCellAnchorResult> nonEmptyCells;
    double elapsedSeconds = 0.0;
};

struct FiberAnchorArtifactInfo {
    std::string sourceLocator;
    std::string manifestContentHash;
    double glyphLengthBaseVoxels = 16.0;
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

using FiberAnchorRetainPredicate = std::function<bool(const FiberAnchor& anchor)>;
using FiberAnchorProgressCallback =
    std::function<void(const FiberAnchorProgress& progress)>;

void validateFiberAnchorConfig(const FiberAnchorConfig& config);

[[nodiscard]] FiberAnchorCrop fiberAnchorCropFromBaseVoxels(const FiberAnchorCrop& baseCrop, double predictionToBaseScale);

[[nodiscard]] FiberCellAnchorResult fitFiberCellAnchors(
    const std::array<size_t, 3>& cellZYX,
    const std::array<size_t, 3>& cellBeginZYX,
    const std::array<size_t, 3>& cellEndZYX,
    const std::vector<FiberAnchorObservation>& observations,
    const FiberAnchorConfig& config);

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

void suppressFiberAnchorDuplicates(
    std::vector<FiberCellAnchorResult>& cells,
    const FiberAnchorConfig& config);

[[nodiscard]] nlohmann::json fiberAnchorReportJson(const FiberAnchorExtractionReport& report, const FiberAnchorArtifactInfo& artifact);

[[nodiscard]] std::string fiberAnchorReportObj(const FiberAnchorExtractionReport& report, const FiberAnchorArtifactInfo& artifact);

[[nodiscard]] std::string fiberAnchorCellReportObj(
    const FiberAnchorExtractionReport& report);

void writeFiberAnchorArtifacts(const std::filesystem::path& outputDirectory, const FiberAnchorExtractionReport& report, const FiberAnchorArtifactInfo& artifact);

}  // namespace vc::fiber_tracer
