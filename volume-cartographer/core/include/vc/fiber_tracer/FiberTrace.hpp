#pragma once

#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer {

struct FiberTraceConfig {
    double stepVoxels = 4.0;
    double coneAngleDegrees = 25.0;
    double coneAngleStepDegrees = 5.0;
    int beamWidth = 8;
    int beamLookaheadSteps = 1;
    double smoothnessWeight = 2.0;
    double smoothnessNormalWeight = 0.1;
    double smoothnessTangentWeight = 10.0;
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

class FiberPredictionSource {
public:
    virtual ~FiberPredictionSource() = default;
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

using FiberTraceProgressCallback = std::function<void(const FiberTraceProgress&)>;

[[nodiscard]] FiberTraceSegmentResult traceFiberSegment(
    const FiberPredictionSource& predictions,
    const FiberTraceSegmentRequest& request,
    const vc::lasagna::NormalSampler* normalSampler = nullptr,
    const FiberTraceProgressCallback& progress = {});

[[nodiscard]] cv::Vec3d referenceTangentToward(
    const std::vector<cv::Vec3d>& line,
    size_t startIndex,
    size_t targetIndex);

} // namespace vc::fiber_tracer
