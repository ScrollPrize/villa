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

namespace vc::fiber_tracer {

struct FiberAnchorObservation {
    cv::Vec3d positionPredictionXYZ{0.0, 0.0, 0.0};
    cv::Vec3d direction{0.0, 0.0, 0.0};
    double presence = 0.0;
    bool valid = false;
};

struct FiberAnchorConfig {
    int cellSizePredictionVoxels = 4;
    double gaussianSigmaPredictionVoxels = 2.0;
    double observationPresenceFloor = 0.05;
    double minimumAlignedSupport = 0.05;
    size_t maximumSeedCount = 8;
    int maximumIterations = 64;
    double convergenceTolerance = 1.0e-12;
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
};

struct FiberAnchorComponent {
    FiberAnchor anchor;
    bool retained = false;
    std::string rejectionReason;
    size_t assignedObservationCount = 0;
};

struct FiberCellAnchorResult {
    std::array<size_t, 3> cellZYX{0, 0, 0};
    std::array<FiberAnchorComponent, 2> components;
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
};

struct FiberAnchorExtractionReport {
    FiberPredictionGridInfo grid;
    FiberAnchorConfig config;
    FiberAnchorCrop selectedCrop;
    std::array<size_t, 3> selectedCellBeginZYX{0, 0, 0};
    std::array<size_t, 3> selectedCellEndZYX{0, 0, 0};
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

using FiberStoredPredictionBatchSampler = std::function<void(
    const std::vector<std::array<size_t, 3>>& indicesZYX,
    int parallelThreads,
    std::vector<FiberStoredPredictionSample>& samples)>;

void validateFiberAnchorConfig(const FiberAnchorConfig& config);

[[nodiscard]] FiberAnchorCrop fiberAnchorCropFromBaseVoxels(
    const FiberAnchorCrop& baseCrop,
    double predictionToBaseScale);

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
    std::optional<FiberAnchorCrop> crop = std::nullopt);

[[nodiscard]] nlohmann::json fiberAnchorReportJson(
    const FiberAnchorExtractionReport& report,
    const FiberAnchorArtifactInfo& artifact);

[[nodiscard]] std::string fiberAnchorReportObj(
    const FiberAnchorExtractionReport& report,
    const FiberAnchorArtifactInfo& artifact);

void writeFiberAnchorArtifacts(
    const std::filesystem::path& outputDirectory,
    const FiberAnchorExtractionReport& report,
    const FiberAnchorArtifactInfo& artifact);

} // namespace vc::fiber_tracer
