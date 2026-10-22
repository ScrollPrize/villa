#pragma once

#include "vc/fiber_tracer/FiberChunkTrace.hpp"
#include "vc/fiber_tracer/FiberCaseQueue.hpp"

namespace vc::fiber_tracer {

struct FiberEvaluationCase {
    std::vector<cv::Vec3d> referenceBase;
    double beginArcBase = 0;
    std::optional<double> endArcBase;
};

struct FiberEvaluationStatus {
    size_t total = 0, completed = 0;
    bool started = false, finished = false, cancelled = false;
    std::string error;
};

// Shared policy for serial and batch chunk replay. Sampling sources must outlive
// the returned synchronous callback; configuration is captured by value.
[[nodiscard]] ReplayConsistencyPolicy fiberChunkReplayConsistency(
    const FiberPredictionSource&, const vc::lasagna::NormalSampler&,
    const FiberTraceConfig&, const FiberChunkReplayConfig&, double intervalBase);

// One invocation per control object. Poll/cancel from another thread; no Python
// callbacks on suspended native stacks. Cancellation also wakes publication and
// shared chunk waiters. It cannot interrupt arbitrary blocking sampler I/O.
class FiberEvaluationControl {
public:
    void cancel(const std::string& reason = "fiber evaluation cancelled");
    [[nodiscard]] FiberEvaluationStatus status() const;
private:
    friend class FiberEvaluationRun;
    mutable std::mutex mutex_;
    FiberEvaluationStatus status_;
    FiberCaseQueue* queue_ = nullptr; // protected by mutex_, borrowed during run
};

// Ordered, all-or-error results. All admitted jobs are joined before throwing.
// References/configs are copied before work starts. Dataset and cache lifetimes
// must cover the call; they must not be mutated concurrently. Live publication
// can be attached to the prediction field before calling this function.
[[nodiscard]] std::vector<FiberReplayTraceResult> evaluateFiberChunkCases(
    FiberChunkTraceCache&, const FiberPredictionSource&, const FiberTraceConfig&,
    double traceToBaseScale, const vc::lasagna::NormalSampler& tracingNormals,
    const vc::lasagna::NormalSampler& evaluationNormals, double normalWorkingToBaseScale,
    const std::vector<FiberEvaluationCase>&, const FiberChunkReplayConfig&,
    FiberEvaluationControl&, size_t workers = 4, size_t contextsPerWorker = 8,
    double reverseIntervalBase = 0);

// Existing training-benchmark protocol: committed-point presence filtering,
// prediction-aligned restart seeds and separately configured reverse radii.
// Uses the same beam/replay implementation, without changing the ordinary
// callable evaluator's candidate filtering or reference-point restart policy.
[[nodiscard]] FiberReplayTraceResult replayFiberTrainingCase(
    const FiberPredictionSource&, const FiberTraceConfig&, double traceToBaseScale,
    const vc::lasagna::NormalSampler& tracingNormals,
    const vc::lasagna::NormalSampler& evaluationNormals, double normalWorkingToBaseScale,
    const FiberEvaluationCase&, const FiberChunkReplayConfig&,
    const FiberTraceProgressCallback& = {});

// Direct beam backend, without building/caching chunk paths. Same reference
// metric and scheduler; starts/restarts at the selected reference point.
[[nodiscard]] FiberReplayTraceResult replayFiberBeamCase(
    const FiberPredictionSource&, const FiberTraceConfig&, double traceToBaseScale,
    const vc::lasagna::NormalSampler& tracingNormals,
    const vc::lasagna::NormalSampler& evaluationNormals, double normalWorkingToBaseScale,
    const FiberEvaluationCase&, const FiberChunkReplayConfig&,
    double reverseIntervalBase = 0, const FiberTraceProgressCallback& = {});

[[nodiscard]] std::vector<FiberReplayTraceResult> evaluateFiberBeamCases(
    const FiberPredictionSource&, const FiberTraceConfig&, double traceToBaseScale,
    const vc::lasagna::NormalSampler& tracingNormals,
    const vc::lasagna::NormalSampler& evaluationNormals, double normalWorkingToBaseScale,
    const std::vector<FiberEvaluationCase>&, const FiberChunkReplayConfig&,
    FiberEvaluationControl&, size_t workers = 4, size_t contextsPerWorker = 8,
    double reverseIntervalBase = 0);

} // namespace vc::fiber_tracer
