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

std::optional<PolylineArcSample> findFiberReplayRestartSeed(
    const FiberPredictionSource& predictions, const FiberTraceConfig& trace, double scale,
    const vc::lasagna::NormalSampler& normalSampler, double normalWorkingToBaseScale,
    const PolylineArcGeometry& reference, double fromArc, double endArcBase,
    double nominalStepBase)
{
    const double grid = scale;
    const double minDot = std::cos(25.0 * std::acos(-1.0) / 180.0);
    for (double arc = fromArc; arc < endArcBase; arc += nominalStepBase) {
        auto ref = samplePolylineArc(reference, arc);
        const auto tangent = ref.tangent / cv::norm(ref.tangent);
        const auto normal = normalSampler.sampleNormal(ref.point / normalWorkingToBaseScale);
        cv::Vec3d u = normal.valid ? normal.normal : cv::Vec3d(0, 0, 1);
        u -= tangent * u.dot(tangent);
        if (cv::norm(u) < 1e-6) {
            u = cv::Vec3d(0, 1, 0) - tangent * tangent[1];
            if (cv::norm(u) < 1e-6) u = cv::Vec3d(1, 0, 0) - tangent * tangent[0];
        }
        u /= cv::norm(u);
        const cv::Vec3d v = tangent.cross(u);
        const auto sample = [&](const cv::Vec3d& point) {
            FiberPredictionSampleOption best;
            for (const auto& option : predictions.sample(point / grid, tangent).options)
                if (option.valid && option.presence > best.presence) best = option;
            return best;
        };
        std::optional<PolylineArcSample> chosen;
        double bestDistance = std::numeric_limits<double>::infinity();
        for (int i = -int(std::ceil(10 / grid)); i <= int(std::ceil(10 / grid)); ++i)
            for (int j = -int(std::ceil(20 / grid)); j <= int(std::ceil(20 / grid)); ++j) {
                const cv::Vec3d point = ref.point + grid * (i * u + j * v);
                const auto distance = measureFiberReplayThreshold(point, ref.point,
                    normalSampler, normalWorkingToBaseScale, 10, 2);
                if (fiberReplayThresholdExceeded(distance, 10)) continue;
                const auto value = sample(point);
                cv::Vec3d dir(value.direction);
                if (!value.valid || value.presence <= 0 || value.presence < trace.replayPresenceCutoff || cv::norm(dir) <= 1.0e-12) continue;
                dir /= cv::norm(dir);
                const double dot = dir.dot(tangent);
                if (std::abs(dot) < minDot) continue;
                bool peak = true, strict = false;
                for (int a = -1; a <= 1; ++a) for (int b = -1; b <= 1; ++b) {
                    if (!a && !b) continue;
                    const double other = sample(point + grid * (a * u + b * v)).presence;
                    if (other > value.presence) peak = false;
                    if (other < value.presence) strict = true;
                }
                if (peak && strict && distance.thresholdErrorRatio < bestDistance) {
                    bestDistance = distance.thresholdErrorRatio;
                    chosen = ref;
                    chosen->point = point;
                    chosen->tangent = dot < 0 ? -dir : dir;
                }
            }
        if (chosen) return chosen;
    }
    return std::nullopt;
}

static FiberReplayTraceResult replayBeamCase(
    const FiberPredictionSource& predictions, const FiberTraceConfig& trace, double scale,
    const vc::lasagna::NormalSampler& tracingNormals,
    const vc::lasagna::NormalSampler& evaluationNormals, double normalScale,
    const FiberEvaluationCase& c, const FiberChunkReplayConfig& config,
    double reverseInterval, const FiberTraceProgressCallback& progress, bool trainingProtocol)
{
    const FiberTraceCoordinateAdapter coordinates(scale);
    validateFiberTraceConfig(trace);
    const auto reference = makePolylineArcGeometry(c.referenceBase);
    const double end = c.endArcBase.value_or(reference.length());
    const double interval = !trainingProtocol || reverseInterval > 0 ? reverseInterval : trace.replayReverseIntervalBase;
    auto policy = fiberChunkReplayConsistency(predictions, tracingNormals, trace, config, interval);
    policy.toStart = trainingProtocol && trace.replayReverseToStart;
    policy.allEndpoints = trainingProtocol && trace.replayReverseAllEndpoints;
    policy.nextSeed = [&](double arc) -> std::optional<PolylineArcSample> {
        if (arc >= end) return std::nullopt;
        return samplePolylineArc(reference, arc);
    };
    double widthMean = trace.widthReferenceMean;
    float committedModulatedPresence = std::numeric_limits<float>::quiet_NaN();
    if (trainingProtocol && trace.replayPresenceCutoff > 0) {
        policy.acceptPoint = [&](const cv::Vec3d& point, const cv::Vec3d& previous) {
            if (trace.replayPresenceModulated) {
                if (!std::isfinite(committedModulatedPresence))
                    throw std::logic_error("missing committed modulated presence");
                return committedModulatedPresence >= trace.replayPresenceCutoff;
            }
            const auto direction = point - previous;
            float presence = 0;
            for (const auto& option : predictions.sample(coordinates.baseToTrace(point), direction).options)
                if (option.valid && std::isfinite(option.presence)) presence = std::max(presence, option.presence);
            return presence >= trace.replayPresenceCutoff;
        };
    }
    if (trainingProtocol && (interval > 0 || policy.acceptPoint)) {
        policy.nextSeed = [&](double arc) {
            return findFiberReplayRestartSeed(predictions, trace, scale, evaluationNormals,
                normalScale, reference, arc, end, trace.stepVoxels * scale);
        };
    }
    if (interval > 0) policy.verify = [&](const auto& points) {
        auto reverse = trace;
        reverse.traceToBaseScale = scale;
        if (reverse.widthRelativeTolerance > 0) reverse.widthReferenceMean = widthMean;
        const auto path = makePolylineArcGeometry(points);
        return verifyFiberReverseSpan(predictions, tracingNormals, reverse, path, path.length(),
            trainingProtocol ? trace.replayReverseNormalRadiusBase : config.errorThresholdBase,
            trainingProtocol ? trace.replayReverseTangentialRadiusBase : config.errorThresholdBase * config.tangentialThresholdFactor,
            trainingProtocol ? trace.replayReverseHistorySteps : 0) <= 1e-12;
    };
    const ReplaySegmentRunner run = [&](const PolylineArcSample& seed, double remaining,
        int maximumSteps, const ReplayPointObserver& observe, const FiberTraceProgressCallback& report) {
        FiberTraceOneWayRequest request;
        request.startPoint = coordinates.baseToTrace(seed.point);
        request.targetPoint = coordinates.baseToTrace(samplePolylineArc(reference, end).point);
        request.initialDirection = seed.tangent;
        request.budgetSpanVoxels = remaining / scale;
        request.config = trace;
        if (trainingProtocol) request.config.replayPresenceCutoff = 0;
        request.config.traceToBaseScale = scale;
        widthMean = trace.widthReferenceMean;
        return traceFiberReplaySegment(predictions, request, maximumSteps, tracingNormals, report,
            [&](const auto& point, double loss, int step) {
                return observe(coordinates.traceToBase(point), loss, step);
            }, widthMean, &committedModulatedPresence);
    };
    return traceReferenceReplayCore(reference, c.beginArcBase, end, trace.stepVoxels * scale,
        trace.maxStepFactor, config.errorThresholdBase, config.excursionGraceBase,
        config.tangentialThresholdFactor, config.matchRefineSteps, evaluationNormals,
        normalScale, run, progress, {}, policy);
}

FiberReplayTraceResult replayFiberBeamCase(
    const FiberPredictionSource& predictions, const FiberTraceConfig& trace, double scale,
    const vc::lasagna::NormalSampler& tracingNormals,
    const vc::lasagna::NormalSampler& evaluationNormals, double normalScale,
    const FiberEvaluationCase& c, const FiberChunkReplayConfig& config,
    double reverseInterval, const FiberTraceProgressCallback& progress)
{
    return replayBeamCase(predictions, trace, scale, tracingNormals, evaluationNormals,
        normalScale, c, config, reverseInterval, progress, false);
}

FiberReplayTraceResult replayFiberTrainingCase(
    const FiberPredictionSource& predictions, const FiberTraceConfig& trace, double scale,
    const vc::lasagna::NormalSampler& tracingNormals,
    const vc::lasagna::NormalSampler& evaluationNormals, double normalScale,
    const FiberEvaluationCase& c, const FiberChunkReplayConfig& config,
    const FiberTraceProgressCallback& progress)
{
    return replayBeamCase(predictions, trace, scale, tracingNormals, evaluationNormals,
        normalScale, c, config, 0, progress, true);
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
