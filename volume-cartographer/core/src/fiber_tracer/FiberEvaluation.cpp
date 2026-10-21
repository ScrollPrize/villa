#include "vc/fiber_tracer/FiberEvaluation.hpp"
#include "vc/fiber_tracer/FiberReverseConsistency.hpp"
#include <atomic>
#include <cmath>

namespace vc::fiber_tracer {

ReplayConsistencyPolicy fiberChunkReplayConsistency(const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler& normals, const FiberTraceConfig& trace,
    const FiberChunkReplayConfig& config, double interval)
{
    if (!std::isfinite(interval) || interval < 0)
        throw std::invalid_argument("reverse interval must be finite and nonnegative");
    ReplayConsistencyPolicy policy;
    policy.intervalBase = interval;
    if (interval > 0) policy.verify = [&predictions, &normals, trace, config](const auto& points) {
        const auto path = makePolylineArcGeometry(points);
        return verifyFiberReverseSpan(predictions, normals, trace, path, path.length(),
            config.errorThresholdBase, config.errorThresholdBase * config.tangentialThresholdFactor) <= 1e-12;
    };
    return policy;
}

void FiberEvaluationControl::cancel(const std::string& reason)
{
    std::lock_guard lock(mutex_);
    if (status_.finished || status_.cancelled) return;
    status_.cancelled = true;
    status_.error = reason.empty() ? "fiber evaluation cancelled" : reason;
    if (queue_) queue_->cancel(status_.error);
}

FiberEvaluationStatus FiberEvaluationControl::status() const
{
    std::lock_guard lock(mutex_);
    return status_;
}

class FiberEvaluationRun {
public:
    FiberEvaluationRun(FiberEvaluationControl& control, size_t total) : control_(control) {
        std::lock_guard lock(control_.mutex_);
        if (control_.status_.started) throw std::invalid_argument("evaluation control already used");
        control_.status_.started = true;
        control_.status_.total = total;
    }
    ~FiberEvaluationRun() {
        std::lock_guard lock(control_.mutex_);
        control_.queue_ = nullptr;
        control_.status_.finished = true;
    }
    void attach(FiberCaseQueue* queue) {
        std::lock_guard lock(control_.mutex_);
        control_.queue_ = queue;
        if (queue && control_.status_.cancelled) queue->cancel(control_.status_.error);
    }
    void check() const {
        const auto s = control_.status();
        if (s.cancelled) throw std::runtime_error(s.error);
    }
    void completed() {
        std::lock_guard lock(control_.mutex_);
        ++control_.status_.completed;
    }
    void finish() {
        std::lock_guard lock(control_.mutex_);
        if (control_.status_.cancelled) throw std::runtime_error(control_.status_.error);
        control_.status_.finished = true;
    }
    void failed(const std::string& message) {
        std::lock_guard lock(control_.mutex_);
        if (control_.status_.error.empty()) control_.status_.error = message;
    }
private:
    FiberEvaluationControl& control_;
};

FiberReplayTraceResult replayFiberBeamCase(
    const FiberPredictionSource& predictions, const FiberTraceConfig& trace, double scale,
    const vc::lasagna::NormalSampler& tracingNormals,
    const vc::lasagna::NormalSampler& evaluationNormals, double normalScale,
    const FiberEvaluationCase& c, const FiberChunkReplayConfig& config,
    double reverseInterval, const FiberTraceProgressCallback& progress)
{
    const FiberTraceCoordinateAdapter coordinates(scale);
    validateFiberTraceConfig(trace);
    const auto reference = makePolylineArcGeometry(c.referenceBase);
    const double end = c.endArcBase.value_or(reference.length());
    auto policy = fiberChunkReplayConsistency(predictions, tracingNormals, trace, config, reverseInterval);
    policy.nextSeed = [&](double arc) -> std::optional<PolylineArcSample> {
        if (arc >= end) return std::nullopt;
        return samplePolylineArc(reference, arc);
    };
    double widthMean = trace.widthReferenceMean;
    if (reverseInterval > 0) policy.verify = [&](const auto& points) {
        auto reverse = trace;
        reverse.traceToBaseScale = scale;
        if (reverse.widthRelativeTolerance > 0) reverse.widthReferenceMean = widthMean;
        const auto path = makePolylineArcGeometry(points);
        return verifyFiberReverseSpan(predictions, tracingNormals, reverse, path, path.length(),
            config.errorThresholdBase, config.errorThresholdBase * config.tangentialThresholdFactor) <= 1e-12;
    };
    const ReplaySegmentRunner run = [&](const PolylineArcSample& seed, double remaining,
        int maximumSteps, const ReplayPointObserver& observe, const FiberTraceProgressCallback& report) {
        FiberTraceOneWayRequest request;
        request.startPoint = coordinates.baseToTrace(seed.point);
        request.targetPoint = coordinates.baseToTrace(samplePolylineArc(reference, end).point);
        request.initialDirection = seed.tangent;
        request.budgetSpanVoxels = remaining / scale;
        request.config = trace;
        request.config.traceToBaseScale = scale;
        widthMean = trace.widthReferenceMean;
        return traceFiberReplaySegment(predictions, request, maximumSteps, tracingNormals, report,
            [&](const auto& point, double loss, int step) {
                return observe(coordinates.traceToBase(point), loss, step);
            }, widthMean);
    };
    return traceReferenceReplayCore(reference, c.beginArcBase, end, trace.stepVoxels * scale,
        trace.maxStepFactor, config.errorThresholdBase, config.excursionGraceBase,
        config.tangentialThresholdFactor, config.matchRefineSteps, evaluationNormals,
        normalScale, run, progress, {}, policy);
}

static std::vector<FiberReplayTraceResult> evaluateCases(
    FiberChunkTraceCache* cache, const FiberPredictionSource& predictions,
    const FiberTraceConfig& traceInput, double scale,
    const vc::lasagna::NormalSampler& tracingNormals,
    const vc::lasagna::NormalSampler& evaluationNormals, double normalScale,
    const std::vector<FiberEvaluationCase>& input, const FiberChunkReplayConfig& replayInput,
    FiberEvaluationControl& control, size_t workers, size_t contexts, double reverseInterval)
{
    FiberEvaluationRun run(control, input.size());
    try {
        run.check();
        if (!workers || !contexts) throw std::invalid_argument("evaluation worker/context budgets must be positive");
        if (traceInput.parallelThreads != 1)
            throw std::invalid_argument("evaluation requires parallel_threads=1; the case queue owns parallelism");
        if (!predictions.supportsConcurrentSampling() || !tracingNormals.supportsConcurrentSampling() ||
            !evaluationNormals.supportsConcurrentSampling())
            throw std::invalid_argument("evaluation requires concurrent sampling sources");
        if (traceInput.profile)
            throw std::invalid_argument("batch evaluation cannot share a mutable trace profile");
        const auto trace = traceInput;
        const auto config = replayInput;
        const auto cases = input;
        const auto policy = fiberChunkReplayConsistency(predictions, tracingNormals, trace, config, reverseInterval);
        std::vector<PolylineArcGeometry> references;
        for (const auto& c : cases) {
            auto line = makePolylineArcGeometry(c.referenceBase);
            const auto end = c.endArcBase.value_or(line.length());
            if (!std::isfinite(c.beginArcBase) || !std::isfinite(end) ||
                c.beginArcBase < 0 || end <= c.beginArcBase || end > line.length())
                throw std::invalid_argument("evaluation case has invalid reference arc interval");
            references.push_back(std::move(line));
        }
        std::vector<FiberReplayTraceResult> results(cases.size());
        std::atomic<size_t> next{0};
        // Keep the queue alive until detached from cancel(), including on error.
        auto queue = std::make_unique<FiberCaseQueue>(workers, cases.size(), [&] {
            run.check();
            const size_t i = next.fetch_add(1);
            if (!cache) results[i] = replayFiberBeamCase(predictions, trace, scale, tracingNormals,
                evaluationNormals, normalScale, cases[i], config, reverseInterval,
                [&](const auto&) { run.check(); });
            else results[i] = traceFiberChunkReferenceReplay(*cache, predictions, trace, scale, tracingNormals,
                references[i], cases[i].beginArcBase, cases[i].endArcBase.value_or(references[i].length()),
                evaluationNormals, normalScale, config, FiberCaseQueue::parallelFor,
                [](const auto& ready) { FiberCaseQueue::await(ready); },
                [&](const auto&) { run.check(); }, {}, policy);
            run.check();
            run.completed();
        }, contexts);
        run.attach(queue.get());
        try { queue->join(); }
        catch (...) { run.attach(nullptr); throw; }
        run.attach(nullptr);
        run.finish();
        return results;
    } catch (const std::exception& e) {
        run.failed(e.what());
        throw;
    } catch (...) {
        run.failed("unknown evaluation error");
        throw;
    }
}
std::vector<FiberReplayTraceResult> evaluateFiberChunkCases(
    FiberChunkTraceCache& cache, const FiberPredictionSource& predictions,
    const FiberTraceConfig& trace, double scale, const vc::lasagna::NormalSampler& normals,
    const vc::lasagna::NormalSampler& evaluationNormals, double normalScale,
    const std::vector<FiberEvaluationCase>& cases, const FiberChunkReplayConfig& config,
    FiberEvaluationControl& control, size_t workers, size_t contexts, double reverseInterval)
{
    return evaluateCases(&cache, predictions, trace, scale, normals, evaluationNormals,
        normalScale, cases, config, control, workers, contexts, reverseInterval);
}

std::vector<FiberReplayTraceResult> evaluateFiberBeamCases(
    const FiberPredictionSource& predictions, const FiberTraceConfig& trace, double scale,
    const vc::lasagna::NormalSampler& normals, const vc::lasagna::NormalSampler& evaluationNormals,
    double normalScale, const std::vector<FiberEvaluationCase>& cases,
    const FiberChunkReplayConfig& config, FiberEvaluationControl& control,
    size_t workers, size_t contexts, double reverseInterval)
{
    return evaluateCases(nullptr, predictions, trace, scale, normals, evaluationNormals,
        normalScale, cases, config, control, workers, contexts, reverseInterval);
}
} // namespace vc::fiber_tracer
