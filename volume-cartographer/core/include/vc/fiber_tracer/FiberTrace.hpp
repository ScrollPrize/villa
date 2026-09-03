#pragma once

#include "vc/fiber_tracer/FiberReplayMetric.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include "vc/lasagna/LineModel.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

struct FiberTraceProfile {
    size_t oneWayCalls = 0;
    size_t generations = 0;
    size_t candidateTasks = 0;
    size_t lookaheadFinalFrontiers = 0;
    size_t lookaheadTotalParents = 0;
    size_t lookaheadRequiredParents = 0;
    size_t lookaheadEvaluatedParents = 0;
    size_t lookaheadTotalChildCandidates = 0;
    size_t lookaheadRequiredChildCandidates = 0;
    size_t lookaheadEvaluatedChildCandidates = 0;
    std::vector<size_t> lookaheadParentCounts;
    std::vector<size_t> lookaheadRequiredParentCounts;
    size_t candidateDepth1Batches = 0;
    size_t candidateDepth2Batches = 0;
    size_t candidateDepth1Points = 0;
    size_t candidateDepth2Points = 0;
    std::vector<size_t> candidateDepth1BatchSizes;
    std::vector<size_t> candidateDepth2BatchSizes;
    uint64_t cornerPointCount = 0;
    uint64_t cornerUniqueVoxelCubes = 0;
    uint64_t cornerWorkerTasks = 0;
    uint64_t cornerMaxCandidatesPerCube = 0;
    std::array<uint64_t, 65> cornerCubeOccupancyHistogram{};
    uint64_t depthDependencyShared = 0;
    uint64_t depthDependencyUnion = 0;
    uint64_t stepDependencyShared = 0;
    uint64_t stepDependencyUnion = 0;
    std::vector<uint64_t> localityCurrentDepth1Dependencies;
    std::vector<uint64_t> localityPreviousStepDependencies;
    double startSampleSeconds = 0.0;
    double taskBuildSeconds = 0.0;
    double predictionBatchSeconds = 0.0;
    double predictionPrepareSeconds = 0.0;
    double predictionPrefetchSeconds = 0.0;
    double predictionAssignSeconds = 0.0;
    double predictionMaterializeSeconds = 0.0;
    double predictionCornerSeconds = 0.0;
    double predictionCornerPrepareSeconds = 0.0;
    double predictionCornerLayoutSeconds = 0.0;
    double predictionCornerPinSeconds = 0.0;
    double predictionCornerGatherSeconds = 0.0;
    uint64_t predictionCornerLayoutChunkRuns = 0;
    uint64_t predictionCornerBoundaryPoints = 0;
    uint64_t predictionCornerDependencies = 0;
    double predictionDecodeSeconds = 0.0;
    double normalDecodeSeconds = 0.0;
    double normalBatchSeconds = 0.0;
    double normalPrefetchSeconds = 0.0;
    double normalMaterializeSeconds = 0.0;
    double candidateScoreSeconds = 0.0;
    double frontierSeconds = 0.0;
    double pruneSeconds = 0.0;
    double lookaheadDecisionSeconds = 0.0;
    double lookaheadParentOrderSeconds = 0.0;
    double lookaheadFrontierStorageSeconds = 0.0;
    size_t lookaheadFrontierAllocatedSlots = 0;
    size_t lookaheadFrontierEvaluatedSlots = 0;
};

struct FiberTraceConfig {
    double stepVoxels = 4.0;
    double coneAngleDegrees = 25.0;
    double coneAngleStepDegrees = 5.0;
    int coneGridSize = 25;
    int beamWidth = 8;
    double beamPruneDistanceVoxels = 1.0;
    int beamLookaheadSteps = 2;
    bool lazyLookahead = true;
    size_t lookaheadParentCap = 32;
    size_t lookaheadRetryParentCap = 0;
    int parallelThreads = 0;
    double smoothnessWeight = 2.0;
    double smoothnessNormalWeight = 0.1;
    double smoothnessTangentWeight = 10.0;
    double smoothnessFreeAngleDegrees = 0.0;
    int cumulativeSmoothnessSteps = 4;
    double cumulativeSmoothnessTangentWeight = 2.0;
    double initialFreeAngleDegrees = 0.0;
    double maxStepFactor = 3.0;
    double meetingAcceptMaxErrorRatio = 0.1;
    double endpointAcceptThresholdBaseVoxels = 20.0;
    double traceToBaseScale = 1.0;
    std::optional<double> baseVoxelSizeUm;
    FiberTraceProfile* profile = nullptr;
};

struct FiberTraceProgress {
    int step = 0;
    int maxSteps = 0;
    double targetPlaneProgress = 0.0;
    std::string phase;
    std::string reason;
    std::optional<size_t> replaySegmentIndex;
    std::optional<double> referenceArcBase;
    std::optional<double> referenceArcFraction;
};

struct FiberPredictionSampleOption {
    cv::Vec3f direction{0.0f, 0.0f, 0.0f};
    float presence = 0.0f;
    bool valid = false;
};

class FiberPredictionSampleOptions
{
public:
    class const_iterator
    {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = FiberPredictionSampleOption;
        using difference_type = std::ptrdiff_t;
        using pointer = const FiberPredictionSampleOption*;
        using reference = const FiberPredictionSampleOption&;

        const_iterator() = default;
        const_iterator(const FiberPredictionSampleOptions* owner, size_t index) : owner_(owner), index_(index) {}

        reference operator*() const { return (*owner_)[index_]; }
        pointer operator->() const { return &(*owner_)[index_]; }
        const_iterator& operator++()
        {
            ++index_;
            return *this;
        }
        const_iterator operator++(int)
        {
            const_iterator copy = *this;
            ++(*this);
            return copy;
        }
        [[nodiscard]] bool operator==(const const_iterator& other) const { return owner_ == other.owner_ && index_ == other.index_; }
        [[nodiscard]] bool operator!=(const const_iterator& other) const { return !(*this == other); }

    private:
        const FiberPredictionSampleOptions* owner_ = nullptr;
        size_t index_ = 0;
    };

    void clear()
    {
        size_ = 0;
        overflow_.clear();
    }

    void reserve(size_t count)
    {
        if (count > inline_.size()) {
            overflow_.reserve(count);
        }
    }

    void push_back(const FiberPredictionSampleOption& option)
    {
        if (size_ < inline_.size() && overflow_.empty()) {
            inline_[size_++] = option;
            return;
        }
        if (overflow_.empty()) {
            overflow_.assign(inline_.begin(), inline_.end());
        }
        overflow_.push_back(option);
        ++size_;
    }

    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] const FiberPredictionSampleOption& operator[](size_t index) const
    {
        return overflow_.empty() ? inline_[index] : overflow_[index];
    }

    [[nodiscard]] const_iterator begin() const { return {this, 0}; }
    [[nodiscard]] const_iterator end() const { return {this, size_}; }

private:
    std::array<FiberPredictionSampleOption, 4> inline_{};
    std::vector<FiberPredictionSampleOption> overflow_;
    size_t size_ = 0;
};

struct FiberPredictionSample {
    FiberPredictionSampleOptions options;
};

struct FiberPredictionTraceScales {
    double traceToBaseScale = 1.0;
    double predictionToBaseScale = 1.0;
    double predictionSpacingInTraceVoxels = 1.0;
};

struct FiberPredictionGridInfo {
    std::array<size_t, 3> shapeZYX{0, 0, 0};
    double predictionToBaseScale = 1.0;
};

struct FiberStoredPredictionSample {
    cv::Vec3d direction{0.0, 0.0, 0.0};
    double presence = 0.0;
    bool valid = false;
    bool presenceValid = false;
};

struct FiberStoredPresenceSample {
    double presence = 0.0;
    bool valid = false;
};

struct FiberPresenceChunkScanReport {
    std::array<size_t, 3> shapeZYX{0, 0, 0};
    std::array<size_t, 3> chunksZYX{0, 0, 0};
    std::array<size_t, 3> chunkGridShapeZYX{0, 0, 0};
    size_t missingChunks = 0;
    size_t emptyChunks = 0;
    std::vector<std::array<size_t, 3>> nonemptyChunksZYX;
};

enum class FiberPredictionFieldBindingMode {
    TraceOptions,
    CanonicalStoredGrid,
};

struct FiberTraceCoordinateAdapter {
    explicit FiberTraceCoordinateAdapter(double traceToBaseScale);

    [[nodiscard]] cv::Vec3d baseToTrace(const cv::Vec3d& point) const;
    [[nodiscard]] cv::Vec3d traceToBase(const cv::Vec3d& point) const;
    [[nodiscard]] std::vector<cv::Vec3d> baseToTrace(const std::vector<cv::Vec3d>& points) const;
    [[nodiscard]] std::vector<cv::Vec3d> traceToBase(const std::vector<cv::Vec3d>& points) const;
    [[nodiscard]] std::vector<cv::Vec3d> traceSegmentToBase(const std::vector<cv::Vec3d>& points, const cv::Vec3d& exactStartBase, const cv::Vec3d& exactTargetBase) const;
    [[nodiscard]] double baseDistanceToTrace(double distanceBaseVoxels) const;
    [[nodiscard]] double traceDistanceToBase(double distanceTraceVoxels) const;

    double traceToBaseScale = 1.0;
};

struct FiberInput {
    std::filesystem::path path;
    std::vector<cv::Vec3d> linePointsXyzBase;
    std::vector<cv::Vec3d> controlPointsXyzBase;
    std::vector<size_t> controlPointLineIndices;
};

class FiberPredictionSource
{
public:
    virtual ~FiberPredictionSource() = default;
    [[nodiscard]] virtual bool supportsConcurrentSampling() const noexcept { return false; }
    [[nodiscard]] virtual vc::lasagna::NormalPrefetchReport prefetchSamples(const std::vector<cv::Vec3d>& /*volumePoints*/) const
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
            throw std::invalid_argument("fiber prediction batch points and reference directions size mismatch");
        }
        (void)prefetchSamples(volumePoints);
        samples.clear();
        samples.reserve(volumePoints.size());
        for (size_t index = 0; index < volumePoints.size(); ++index) {
            samples.push_back(sample(volumePoints[index], referenceDirections[index]));
        }
    }
    [[nodiscard]] virtual FiberPredictionSample sample(const cv::Vec3d& volumePoint, const cv::Vec3d& referenceDirection) const = 0;
};

class FiberPredictionField final : public FiberPredictionSource
{
public:
    explicit FiberPredictionField(
        const vc::lasagna::LasagnaDataset& dataset,
        size_t maxCachedBytes = 512ULL * 1024ULL * 1024ULL,
        FiberPredictionFieldBindingMode bindingMode = FiberPredictionFieldBindingMode::TraceOptions);
    ~FiberPredictionField() override;

    [[nodiscard]] bool supportsConcurrentSampling() const noexcept override { return true; }
    [[nodiscard]] vc::lasagna::NormalPrefetchReport prefetchSamples(const std::vector<cv::Vec3d>& volumePoints) const override;
    void sampleBatch(
        const std::vector<cv::Vec3d>& volumePoints,
        const std::vector<cv::Vec3d>& referenceDirections,
        int parallelThreads,
        std::vector<FiberPredictionSample>& samples) const override;
    void sampleBatch(
        const std::vector<cv::Vec3d>& volumePoints,
        const std::vector<cv::Vec3d>& referenceDirections,
        int parallelThreads,
        std::vector<FiberPredictionSample>& samples,
        FiberTraceProfile* profile) const;
    void sampleBatch(
        const std::vector<cv::Vec3f>& volumePoints,
        const std::vector<cv::Vec3f>& referenceDirections,
        int parallelThreads,
        std::vector<FiberPredictionSample>& samples,
        FiberTraceProfile* profile) const;
    [[nodiscard]] bool sampleCornerBatchWithNormals(
        const vc::lasagna::LasagnaNormalSampler& normalSampler,
        const std::vector<cv::Vec3f>& volumePoints,
        int parallelThreads,
        vc::lasagna::LasagnaCornerBatch* cornerScratch,
        FiberTraceProfile* profile) const;
    [[nodiscard]] bool visitCornerBatchWithNormals(
        const vc::lasagna::LasagnaNormalSampler& normalSampler,
        const std::vector<cv::Vec3f>& volumePoints,
        int parallelThreads,
        void* visitorContext,
        vc::lasagna::LasagnaCornerPointVisitor visitor,
        int lookaheadDepth,
        FiberTraceProfile* profile) const;
    [[nodiscard]] FiberPredictionSample sample(const cv::Vec3d& volumePoint, const cv::Vec3d& referenceDirection) const override;
    [[nodiscard]] FiberPredictionGridInfo storedGridInfo() const;
    [[nodiscard]] FiberPresenceChunkScanReport scanStoredPresenceChunks(int parallelThreads) const;
    void sampleStoredGridBatch(const std::vector<std::array<size_t, 3>>& indicesZYX, int parallelThreads, std::vector<FiberStoredPredictionSample>& samples) const;
    void sampleStoredPresenceBatch(const std::vector<std::array<size_t, 3>>& indicesZYX, int parallelThreads, std::vector<FiberStoredPresenceSample>& samples) const;
    [[nodiscard]] size_t optionCount() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

struct FiberTraceSegmentRequest {
    std::vector<cv::Vec3d> referenceLine;
    size_t startIndex = 0;
    size_t targetIndex = 0;
    FiberTraceConfig config;
};

struct FiberTraceTargetPlane {
    std::string name;
    cv::Vec3d point{0.0, 0.0, 0.0};
    cv::Vec3d normal{1.0, 0.0, 0.0};
};

struct FiberTraceTargetPlaneCrossing {
    std::string name;
    cv::Vec3d point{0.0, 0.0, 0.0};
    double inPlaneErrorVoxels = 0.0;
};

struct FiberTraceOneWayRequest {
    cv::Vec3d startPoint{0.0, 0.0, 0.0};
    cv::Vec3d targetPoint{0.0, 0.0, 0.0};
    cv::Vec3d initialDirection{1.0, 0.0, 0.0};
    std::vector<FiberTraceTargetPlane> targetPlanes;
    std::optional<double> targetPlaneAcceptThresholdVoxels;
    bool snapTraceToSelectedCrossing = true;
    double budgetSpanVoxels = 0.0;
    FiberTraceConfig config;
};

struct FiberTraceOneWayResult {
    std::vector<cv::Vec3d> points;
    bool reachedTargetPlane = false;
    bool reachedTraceLength = false;
    std::string reason;
    int steps = 0;
    std::vector<FiberTraceTargetPlaneCrossing> targetPlaneCrossings;
    std::string selectedTargetPlaneName;
    std::optional<cv::Vec3d> selectedTargetPlaneCrossing;
    double selectedTargetPlaneErrorVoxels = std::numeric_limits<double>::infinity();
};

struct FiberTraceSegmentResult {
    FiberTraceOneWayResult forward;
    FiberTraceOneWayResult reverse;
    std::vector<cv::Vec3d> fusedLine;
    double forwardEndpointErrorTraceVoxels = 0.0;
    double reverseEndpointErrorTraceVoxels = 0.0;
    double maxEndpointErrorTraceVoxels = 0.0;
    double maxEndpointErrorBaseVoxels = 0.0;
    std::optional<double> maxEndpointErrorUm;
    double meetingErrorTraceVoxels = std::numeric_limits<double>::infinity();
    double meetingErrorBaseVoxels = std::numeric_limits<double>::infinity();
    double meetingErrorRatio = std::numeric_limits<double>::infinity();
    double meetingTraceLengthTraceVoxels = 0.0;
    std::optional<double> meetingErrorUm;
    std::string meetingSource;
    bool accepted = false;
    std::string reason;
    std::string detail;
};

struct FiberTraceWholeFiberSegmentResult {
    size_t startControlPointIndex = 0;
    size_t targetControlPointIndex = 0;
    FiberTraceOneWayResult trace;
    bool success = false;
    bool restart = false;
    std::string reason;
    double inPlaneErrorTraceVoxels = 0.0;
    double inPlaneErrorBaseVoxels = 0.0;
    double referenceArcDistanceVoxels = 0.0;
};

struct FiberTraceWholeFiberResult {
    std::vector<FiberTraceWholeFiberSegmentResult> segments;
    std::vector<cv::Vec3d> stitchedTrace;
    int restartCount = 0;
    int lookaheadRetryCount = 0;
    int lookaheadRetryRecoveredCount = 0;
    int segmentCount = 0;
    double restartsPerKvx = 0.0;
    double referenceLengthVoxels = 0.0;
    std::optional<double> referenceLengthMeters;
    std::optional<double> restartsPerMeter;
};

struct FiberTraceWholeFiberMetricRequest {
    FiberInput fiber;
    double workingToBaseScale = 1.0;
    double errorThresholdBaseVoxels = 20.0;
    std::optional<double> voxelSizeUm;
    FiberTraceConfig config;
};

struct FiberReplayMatch {
    size_t tracePointIndex = 0;
    double predictedReferenceArcBase = 0.0;
    double matchedReferenceArcBase = 0.0;
    cv::Vec3d matchedReferencePointBase{0.0, 0.0, 0.0};
    double searchBeginArcBase = 0.0;
    double searchEndArcBase = 0.0;
    FiberReplayThresholdMeasurement thresholdMeasurement;
};

struct FiberReplayTraceRequest {
    FiberInput fiber;
    size_t startControlPointIndex = 0;
    std::optional<double> referenceBeginArcBase;
    double traceToBaseScale = 1.0;
    double errorThresholdBaseVoxels = 20.0;
    double matchRefineSteps = 1.0;
    std::optional<double> referenceEndArcBase;
    FiberTraceConfig config;
};

struct LasagnaReplayTraceRequest {
    std::vector<cv::Vec3d> referencePointsBase;
    double stepBaseVoxels = 16.0;
    double errorThresholdBaseVoxels = 20.0;
    double matchRefineSteps = 1.0;
    double maxStepFactor = 3.0;
};

struct FiberReplayFailure {
    size_t index = 0;
    size_t segmentIndex = 0;
    std::string reason;
    double referenceArcBase = 0.0;
    double referenceArcFraction = 0.0;
    cv::Vec3d referencePointBase{0.0, 0.0, 0.0};
    std::optional<cv::Vec3d> evaluatorPointBase;
    std::optional<size_t> segmentPointIndex;
    std::optional<size_t> candidateIndex;
    std::optional<size_t> arcIndex;
    std::optional<size_t> candidatePathPointIndex;
    std::optional<FiberReplayThresholdMeasurement> thresholdMeasurement;
};

struct FiberReplayTraceSegment {
    double startReferenceArcBase = 0.0;
    double endReferenceArcBase = 0.0;
    std::string terminationReason;
    std::vector<cv::Vec3d> tracePointsBase;
    std::vector<double> cumulativeLosses;
    std::vector<FiberReplayMatch> matches;
};

struct FiberReplayTraceResult {
    double referenceBeginArcBase = 0.0;
    double referenceEndArcBase = 0.0;
    double completedReferenceArcBase = 0.0;
    std::vector<FiberReplayTraceSegment> segments;
    std::vector<FiberReplayFailure> failures;
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
using FiberReplayFailureCallback = std::function<void(const FiberReplayFailure&)>;
using FiberTraceWholeFiberProgressCallback = std::function<void(const FiberTraceWholeFiberProgress&)>;

#ifdef VC_TESTING
namespace testing
{

struct BeamDebugState {
    double loss = 0.0;
    int depth = 0;
    double tracedLength = 0.0;
    cv::Vec3d point{0.0, 0.0, 0.0};
    bool reached = false;
};

struct CandidateScoreDebug {
    double loss = 0.0;
    cv::Vec3d selectedDirection{0.0, 0.0, 0.0};
    double selectedPresence = 0.0;
    bool valid = false;
};

[[nodiscard]] std::vector<size_t> debugPruneBeamStateIndices(const std::vector<BeamDebugState>& states, int beamWidth, double pruneDistanceVoxels);

[[nodiscard]] std::optional<size_t> debugBestReachedBeamStateIndex(const std::vector<BeamDebugState>& states);

[[nodiscard]] int debugTraceWorkerCount(bool predictionConcurrent, bool normalConcurrent, bool hasNormalSampler, int parallelThreads, size_t taskCount);

[[nodiscard]] CandidateScoreDebug debugCandidateLossFromCorners(
    const vc::lasagna::LasagnaCornerBatch& corners,
    size_t optionCount,
    size_t pointIndex,
    const cv::Vec3d& previousStepDirection,
    const cv::Vec3d& currentSampleDirection,
    const cv::Vec3d& historyDirection,
    const cv::Vec3d& candidateDirection,
    const FiberTraceConfig& config);

[[nodiscard]] CandidateScoreDebug debugCandidateLossFromSample(
    const FiberPredictionSample& sample,
    const cv::Vec3d& previousStepDirection,
    const cv::Vec3d& currentSampleDirection,
    const cv::Vec3d& historyDirection,
    const cv::Vec3d& candidateDirection,
    const FiberTraceConfig& config);

[[nodiscard]] size_t debugExactLookaheadRequiredParentCount(const std::vector<double>& parentLowerBounds, std::optional<double> resultThreshold, bool finalBeamSetComplete);

[[nodiscard]] std::vector<size_t> debugOrderedIndexPrefix(const std::vector<double>& losses, size_t limit);

[[nodiscard]] bool debugShouldRetryLookahead(bool lazyLookahead, size_t parentCap, size_t retryParentCap, bool segmentSuccess);

[[nodiscard]] FiberTraceSegmentResult debugFuseTraceSegment(const std::vector<cv::Vec3d>& forward, const std::vector<cv::Vec3d>& reverse, const FiberTraceConfig& config);

}  // namespace testing
#endif

[[nodiscard]] FiberInput loadFiberJson(const std::filesystem::path& path);

[[nodiscard]] FiberPredictionTraceScales resolveFiberPredictionTraceScales(const vc::lasagna::LasagnaDatasetManifest& manifest, int inferenceScaledownPower = 2);

[[nodiscard]] double inferFiberPredictionWorkingToBaseScale(const vc::lasagna::LasagnaDatasetManifest& manifest, int inferenceScaledownPower = 2);

[[nodiscard]] FiberTraceOneWayResult traceFiberOneWay(
    const FiberPredictionSource& predictions,
    const FiberTraceOneWayRequest& request,
    const vc::lasagna::NormalSampler* normalSampler = nullptr,
    const FiberTraceProgressCallback& progress = {});

[[nodiscard]] FiberTraceOneWayResult traceFiberExtrapolation(
    const FiberPredictionSource& predictions,
    const cv::Vec3d& startPoint,
    const cv::Vec3d& outwardDirection,
    double distanceVoxels,
    const FiberTraceConfig& config,
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

[[nodiscard]] FiberReplayTraceResult traceFiberReplay(
    const FiberPredictionSource& predictions,
    const FiberReplayTraceRequest& request,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const FiberTraceProgressCallback& progress = {},
    const FiberReplayFailureCallback& failure = {});

// Reference-tangent-initialized normal-transport control. Invalid normal
// samples retain the previous transported direction.
[[nodiscard]] FiberReplayTraceResult traceLasagnaReplay(
    const vc::lasagna::NormalSampler& normalSampler,
    const LasagnaReplayTraceRequest& request,
    const FiberTraceProgressCallback& progress = {},
    const FiberReplayFailureCallback& failure = {});

[[nodiscard]] cv::Vec3d referenceTangentToward(const std::vector<cv::Vec3d>& line, size_t startIndex, size_t targetIndex);

}  // namespace vc::fiber_tracer
