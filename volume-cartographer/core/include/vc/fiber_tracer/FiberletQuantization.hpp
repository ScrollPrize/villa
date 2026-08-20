#pragma once

#include "vc/fiber_tracer/FiberGraph.hpp"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace vc::fiber_tracer
{

struct FiberletQuantizationScenario {
    std::string name;
    int positionQuantumBaseVoxels = 0;
    bool compactAxes = false;
    int costBits = 0;
};

struct FiberletQuantizationSummary {
    size_t count = 0;
    double minimum = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double maximum = 0.0;
};

struct FiberletQuantizationProgress {
    std::string phase;
    std::string scenario;
    size_t completed = 0;
    size_t total = 0;
    double elapsedSeconds = 0.0;
};

using FiberletQuantizationProgressCallback =
    std::function<void(const FiberletQuantizationProgress& progress)>;

struct FiberletQuantizationScenarioReport {
    FiberletQuantizationScenario scenario;
    bool valid = false;
    std::string reason;
    int anchorPositionBits = 0;
    int anchorDeltaBits = 0;
    size_t anchors = 0;
    size_t coincidentPositionGroups = 0;
    size_t maximumVariants = 0;
    size_t graphNodes = 0;
    size_t graphEdges = 0;
    size_t graphTransitions = 0;
    size_t baselineSuccessfulFiberlets = 0;
    size_t scenarioSuccessfulFiberlets = 0;
    size_t commonSuccessfulFiberlets = 0;
    size_t addedSuccessfulFiberlets = 0;
    size_t removedSuccessfulFiberlets = 0;
    size_t addedTransitions = 0;
    size_t removedTransitions = 0;
    FiberletQuantizationSummary positionErrorBaseVoxels;
    FiberletQuantizationSummary axisErrorDegrees;
    FiberletQuantizationSummary pathPointErrorBaseVoxels;
    FiberletQuantizationSummary pathLengthErrorPredictionVoxels;
    FiberletQuantizationSummary costAbsoluteError;
    FiberletQuantizationSummary costRelativeError;
    FiberletQuantizationSummary joinAngleErrorDegrees;
    FiberletQuantizationSummary joinCostAbsoluteError;
    uint64_t costOrderingInversions = 0;
    uint64_t costOrderingPairs = 0;
    uint64_t chunkCostOrderingInversions = 0;
    uint64_t chunkCostOrderingPairs = 0;
    size_t costTopK = 0;
    size_t costTopKAgreement = 0;
    size_t baselineReplayFailures = 0;
    size_t replayFailures = 0;
    int64_t replayFailureDelta = 0;
    double replayCompletedFraction = 0.0;
    size_t replaySelectedEdges = 0;
    bool lineDistanceAvailable = false;
    size_t lineDistanceSamples = 0;
    size_t lineDistanceInvalidNormalSamples = 0;
    double maximumLineDistanceBaseVoxels = 0.0;
    double maximumLineNormalDistanceBaseVoxels = 0.0;
    double maximumLineTangentialDistanceBaseVoxels = 0.0;
    size_t baselineReferenceInvalidNormalSamples = 0;
    FiberletQuantizationSummary baselineReferenceDistanceBaseVoxels;
    FiberletQuantizationSummary baselineReferenceNormalDistanceBaseVoxels;
    FiberletQuantizationSummary baselineReferenceTangentialDistanceBaseVoxels;
    size_t scenarioReferenceInvalidNormalSamples = 0;
    FiberletQuantizationSummary scenarioReferenceDistanceBaseVoxels;
    FiberletQuantizationSummary scenarioReferenceNormalDistanceBaseVoxels;
    FiberletQuantizationSummary scenarioReferenceTangentialDistanceBaseVoxels;
    std::string geometryReferenceScenario;
    FiberletQuantizationSummary geometryCostAbsoluteError;
    FiberletQuantizationSummary geometryCostRelativeError;
    uint64_t geometryCostOrderingInversions = 0;
    uint64_t geometryCostOrderingPairs = 0;
    size_t geometryCostTopK = 0;
    size_t geometryCostTopKAgreement = 0;
    double geometryDpWallSeconds = 0.0;
    double wallSeconds = 0.0;
};

using FiberletQuantizedPathExtractor =
    std::function<FiberletPathReport(const LoadedFiberAnchorArtifact& anchors)>;

[[nodiscard]] std::vector<FiberletQuantizationScenario> standardFiberletQuantizationScenarios();

[[nodiscard]] std::vector<FiberletQuantizationScenarioReport> benchmarkFiberletQuantization(
    const LoadedFiberAnchorArtifact& baselineAnchors,
    const FiberletPathReport& baselinePaths,
    const std::vector<cv::Vec3d>& referencePointsBaseXYZ,
    const vc::lasagna::NormalSampler& replayNormalSampler,
    double normalWorkingToBaseScale,
    const FiberletGraphReplayConfig& replayConfig,
    const FiberletQuantizedPathExtractor& pathExtractor,
    int storageChunkSideBaseVoxels = 512,
    std::optional<std::string> selectedScenario = std::nullopt,
    const FiberletQuantizationProgressCallback& progress = {});

}  // namespace vc::fiber_tracer
