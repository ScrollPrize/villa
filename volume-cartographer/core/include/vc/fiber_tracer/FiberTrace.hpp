#pragma once

#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer {

struct FiberTraceConfig {
    double stepVoxels = 4.0;
    double coneAngleDegrees = 25.0;
    double coneAngleStepDegrees = 5.0;
    int coneGridSize = 25;
    int beamWidth = 8;
    double beamPruneDistanceVoxels = 1.0;
    int beamLookaheadSteps = 2;
    int parallelThreads = 0;
    double smoothnessWeight = 2.0;
    double smoothnessNormalWeight = 0.1;
    double smoothnessTangentWeight = 10.0;
    double smoothnessFreeAngleDegrees = 0.0;
    int cumulativeSmoothnessSteps = 4;
    double cumulativeSmoothnessTangentWeight = 2.0;
    double initialFreeAngleDegrees = 0.0;
    double maxStepFactor = 3.0;
    double fusionGapFactor = 2.0;
    double endpointAcceptThresholdUm = 50.0;
    double voxelSizeUm = 0.0;
};

struct FiberTraceProgress {
    int step = 0;
    int maxSteps = 0;
    double targetPlaneProgress = 0.0;
    std::string phase;
    std::string reason;
};

struct FiberPredictionSampleOption {
    cv::Vec3d direction{0.0, 0.0, 0.0};
    double presence = 0.0;
    bool valid = false;
};

struct FiberPredictionSample {
    std::vector<FiberPredictionSampleOption> options;
};

struct FiberPredictionTraceScales {
    double traceToBaseScale = 1.0;
    double predictionToBaseScale = 1.0;
    double predictionSpacingInTraceVoxels = 1.0;
};

struct FiberInput {
    std::filesystem::path path;
    std::vector<cv::Vec3d> linePointsXyzBase;
    std::vector<cv::Vec3d> controlPointsXyzBase;
    std::vector<size_t> controlPointLineIndices;
};

class FiberPredictionSource {
public:
    virtual ~FiberPredictionSource() = default;
    [[nodiscard]] virtual bool supportsConcurrentSampling() const noexcept
    {
        return false;
    }
    [[nodiscard]] virtual vc::lasagna::NormalPrefetchReport prefetchSamples(
        const std::vector<cv::Vec3d>& /*volumePoints*/) const
    {
        return {};
    }
    virtual void sampleBatch(
        const std::vector<cv::Vec3d>& volumePoints,
        const std::vector<cv::Vec3d>& referenceDirections,
        int parallelThreads,
        std::vector<FiberPredictionSample>& samples) const
    {
        (void)parallelThreads;
        if (volumePoints.size() != referenceDirections.size()) {
            throw std::invalid_argument(
                "fiber prediction batch points and reference directions size mismatch");
        }
        (void)prefetchSamples(volumePoints);
        samples.clear();
        samples.reserve(volumePoints.size());
        for (size_t index = 0; index < volumePoints.size(); ++index) {
            samples.push_back(sample(volumePoints[index], referenceDirections[index]));
        }
    }
    [[nodiscard]] virtual FiberPredictionSample sample(
        const cv::Vec3d& volumePoint,
        const cv::Vec3d& referenceDirection) const = 0;
};

class FiberPredictionField final : public FiberPredictionSource {
public:
    explicit FiberPredictionField(
        const vc::lasagna::LasagnaDataset& dataset,
        size_t maxCachedBytes = 512ULL * 1024ULL * 1024ULL);
    ~FiberPredictionField() override;

    [[nodiscard]] bool supportsConcurrentSampling() const noexcept override { return true; }
    [[nodiscard]] vc::lasagna::NormalPrefetchReport prefetchSamples(
        const std::vector<cv::Vec3d>& volumePoints) const override;
    void sampleBatch(
        const std::vector<cv::Vec3d>& volumePoints,
        const std::vector<cv::Vec3d>& referenceDirections,
        int parallelThreads,
        std::vector<FiberPredictionSample>& samples) const override;
    [[nodiscard]] FiberPredictionSample sample(
        const cv::Vec3d& volumePoint,
        const cv::Vec3d& referenceDirection) const override;
    [[nodiscard]] size_t optionCount() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

struct FiberTraceSegmentRequest {
    std::vector<cv::Vec3d> referenceLine;
    size_t startIndex = 0;
    size_t targetIndex = 0;
    std::optional<cv::Vec3d> targetPlaneNormal;
    FiberTraceConfig config;
};

struct FiberTraceOneWayRequest {
    cv::Vec3d startPoint{0.0, 0.0, 0.0};
    cv::Vec3d targetPoint{0.0, 0.0, 0.0};
    cv::Vec3d initialDirection{1.0, 0.0, 0.0};
    cv::Vec3d targetPlaneNormal{1.0, 0.0, 0.0};
    double budgetSpanVoxels = 0.0;
    FiberTraceConfig config;
};

struct FiberTraceOneWayResult {
    std::vector<cv::Vec3d> points;
    bool reachedTargetPlane = false;
    std::string reason;
    int steps = 0;
};

struct FiberTraceSegmentResult {
    FiberTraceOneWayResult forward;
    FiberTraceOneWayResult reverse;
    std::vector<cv::Vec3d> fusedLine;
    double forwardEndpointErrorVoxels = 0.0;
    double reverseEndpointErrorVoxels = 0.0;
    double maxEndpointErrorVoxels = 0.0;
    double maxEndpointErrorUm = 0.0;
    bool accepted = false;
    std::string reason;
};

struct FiberTraceWholeFiberSegmentResult {
    size_t startControlPointIndex = 0;
    size_t targetControlPointIndex = 0;
    FiberTraceOneWayResult trace;
    bool success = false;
    bool restart = false;
    std::string reason;
    double inPlaneErrorVoxels = 0.0;
    double referenceArcDistanceVoxels = 0.0;
};

struct FiberTraceWholeFiberResult {
    std::vector<FiberTraceWholeFiberSegmentResult> segments;
    std::vector<cv::Vec3d> stitchedTrace;
    int restartCount = 0;
    int segmentCount = 0;
    double restartsPerKvx = 0.0;
    double referenceLengthVoxels = 0.0;
    std::optional<double> referenceLengthMeters;
    std::optional<double> restartsPerMeter;
};

struct FiberTraceWholeFiberMetricRequest {
    FiberInput fiber;
    double workingToBaseScale = 1.0;
    double errorThresholdVoxels = 10.0;
    std::optional<double> voxelSizeUm;
    FiberTraceConfig config;
};

struct FiberTraceWholeFiberProgress {
    int completedSegments = 0;
    int segmentCount = 0;
    int currentSegment = 0;
    int restartCount = 0;
    double restartsPerKvx = 0.0;
    std::optional<double> restartsPerMeter;
    std::optional<double> referenceLengthMeters;
    std::string status;
    FiberTraceProgress traceProgress;
    bool hasTraceProgress = false;
};

using FiberTraceProgressCallback = std::function<void(const FiberTraceProgress&)>;
using FiberTraceWholeFiberProgressCallback =
    std::function<void(const FiberTraceWholeFiberProgress&)>;

#ifdef VC_TESTING
namespace testing {

struct BeamDebugState {
    double loss = 0.0;
    int depth = 0;
    double tracedLength = 0.0;
    cv::Vec3d point{0.0, 0.0, 0.0};
    bool reached = false;
};

[[nodiscard]] std::vector<size_t> debugPruneBeamStateIndices(
    const std::vector<BeamDebugState>& states,
    int beamWidth,
    double pruneDistanceVoxels);

[[nodiscard]] std::optional<size_t> debugBestReachedBeamStateIndex(
    const std::vector<BeamDebugState>& states);

[[nodiscard]] int debugTraceWorkerCount(
    bool predictionConcurrent,
    bool normalConcurrent,
    bool hasNormalSampler,
    int parallelThreads,
    size_t taskCount);

} // namespace testing
#endif

[[nodiscard]] FiberInput loadFiberJson(const std::filesystem::path& path);

[[nodiscard]] FiberPredictionTraceScales resolveFiberPredictionTraceScales(
    const vc::lasagna::LasagnaDatasetManifest& manifest,
    int inferenceScaledownPower = 2);

[[nodiscard]] double inferFiberPredictionWorkingToBaseScale(
    const vc::lasagna::LasagnaDatasetManifest& manifest,
    int inferenceScaledownPower = 2);

[[nodiscard]] FiberTraceOneWayResult traceFiberOneWay(
    const FiberPredictionSource& predictions,
    const FiberTraceOneWayRequest& request,
    const vc::lasagna::NormalSampler* normalSampler = nullptr,
    const FiberTraceProgressCallback& progress = {});

[[nodiscard]] FiberTraceSegmentResult traceFiberSegment(
    const FiberPredictionSource& predictions,
    const FiberTraceSegmentRequest& request,
    const vc::lasagna::NormalSampler* normalSampler = nullptr,
    const FiberTraceProgressCallback& progress = {});

[[nodiscard]] FiberTraceWholeFiberResult traceWholeFiberMetric(
    const FiberPredictionSource& predictions,
    const FiberTraceWholeFiberMetricRequest& request,
    const vc::lasagna::NormalSampler* normalSampler = nullptr,
    const FiberTraceWholeFiberProgressCallback& progress = {});

[[nodiscard]] cv::Vec3d referenceTangentToward(
    const std::vector<cv::Vec3d>& line,
    size_t startIndex,
    size_t targetIndex);

} // namespace vc::fiber_tracer
