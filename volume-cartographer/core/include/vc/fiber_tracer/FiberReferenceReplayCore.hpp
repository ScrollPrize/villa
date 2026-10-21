#pragma once

#include "vc/fiber_tracer/FiberTrace.hpp"
#include "vc/fiber_tracer/FiberReplayMetric.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"

namespace vc::fiber_tracer {

struct FiberReplayMatch {
    size_t tracePointIndex = 0;
    double predictedReferenceArcBase = 0.0;
    double matchedReferenceArcBase = 0.0;
    cv::Vec3d matchedReferencePointBase{0.0, 0.0, 0.0};
    double searchBeginArcBase = 0.0;
    double searchEndArcBase = 0.0;
    FiberReplayThresholdMeasurement thresholdMeasurement;
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
    std::optional<FiberReplayThresholdMeasurement> thresholdMeasurement;
    // Optional graph provenance; absent for direct/chunk tracing.
    std::optional<size_t> candidateIndex;
    std::optional<size_t> candidatePathPointIndex;
    std::optional<size_t> arcIndex;
};

struct FiberReplayTraceSegment {
    double startReferenceArcBase = 0.0;
    double endReferenceArcBase = 0.0;
    std::string terminationReason;
    std::vector<cv::Vec3d> tracePointsBase;
    std::vector<double> cumulativeLosses;
    std::vector<FiberReplayMatch> matches;
    // Terminal GT-distance failure rejected by the optional reverse check:
    // x,y,z,reference arc before rollback; never an input to reverse acceptance.
    std::optional<std::array<double, 4>> filteredDistanceFailure;
};

struct FiberReplayTraceResult {
    // [rejected-window begin, detection arc, next seed/end arc], base voxels.
    std::vector<std::array<double, 3>> filteredStops;
    size_t reverseChecks = 0;
    double tangentialThresholdFactor = 4.0;
    double referenceBeginArcBase = 0.0;
    double referenceEndArcBase = 0.0;
    double completedReferenceArcBase = 0.0;
    // Extra reference arc skipped by seed search, excluding mandatory reset step.
    double seedSearchSkippedBase = 0.0;
    std::vector<FiberReplayTraceSegment> segments;
    std::vector<FiberReplayFailure> failures;
};

using FiberReplayFailureCallback = std::function<void(const FiberReplayFailure&)>;
// Committed point in base XYZ, cumulative loss, step index -> continue?
using ReplayPointObserver = std::function<bool(const cv::Vec3d&, double, int)>;
// Backend must respect maximum steps and stop when the observer returns false.
// Callbacks are synchronous and must not be retained after runSegment returns.
using ReplaySegmentRunner = std::function<std::string(
    const PolylineArcSample&, double remainingBase, int maximumSteps,
    const ReplayPointObserver&, const FiberTraceProgressCallback&)>;

struct ReplayConsistencyPolicy {
    // Current and previous committed base points; false filters this point.
    std::function<bool(const cv::Vec3d&, const cv::Vec3d&)> acceptPoint;
    double intervalBase = 0;
    bool toStart = false;
    bool allEndpoints = false;
    bool seedAllStarts = false;
    // Check actual forward geometry, not GT. Can use verifyFiberReverseSpan.
    std::function<bool(const std::vector<cv::Vec3d>&)> verify;
    // Prediction-derived seed with corresponding reference arc; null ends search.
    std::function<std::optional<PolylineArcSample>(double)> nextSeed;
};

// Shared evaluation protocol: reference matching, anisotropic acceptance,
// excursion grace, failure records, filtering and deterministic forward resets.
// Geometry must be built with makePolylineArcGeometry. All distances are base
// voxels except normal sampler coordinates, converted by its explicit scale.
[[nodiscard]] FiberReplayTraceResult traceReferenceReplayCore(
    const PolylineArcGeometry& reference,
    double startArcBase, double endArcBase, double nominalStepBase,
    double maxStepFactor, double errorThresholdBaseVoxels,
    double excursionGraceBaseVoxels, double tangentialThresholdFactor,
    double matchRefineSteps, const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale, const ReplaySegmentRunner& runSegment,
    const FiberTraceProgressCallback& progress = {},
    const FiberReplayFailureCallback& failureCallback = {},
    const ReplayConsistencyPolicy& consistency = {});

} // namespace vc::fiber_tracer
