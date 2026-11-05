#include "vc/fiber_tracer/FiberReverseConsistency.hpp"
#include "vc/fiber_tracer/FiberReplayMetric.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <chrono>

namespace vc::fiber_tracer {
namespace {
constexpr double kEpsilon = 1.0e-12;
void validateReverseSpan(const PolylineArcGeometry& geometry, double startArc,
    double scale, double normalRadius, double tangentRadius)
{
    if (!std::isfinite(normalRadius) || normalRadius <= 0 ||
        !std::isfinite(tangentRadius) || tangentRadius <= 0 ||
        !std::isfinite(tangentRadius / normalRadius))
        throw std::invalid_argument("reverse radii must be finite and positive");
    if (!std::isfinite(scale) || scale <= 0 || !std::isfinite(startArc) ||
        startArc < 0 || startArc > geometry.length())
        throw std::invalid_argument("invalid reverse start arc or coordinate scale");
    (void)samplePolylineArc(geometry, startArc);
}
}

void FiberReverseCheckSchedule::update(const std::vector<cv::Vec3d>& points)
{
    if(points.empty() || observed_ >= points.size())
        throw std::logic_error("reverse schedule requires append-only committed points");
    for(size_t i=observed_+1;i<points.size();++i)
        unchecked_+=cv::norm(points[i]-points[i-1]);
    observed_=points.size()-1;
}
void FiberReverseCheckSchedule::accept(const std::vector<cv::Vec3d>& points,size_t target)
{
    checked_=target;unchecked_=0;
    for(size_t i=target+1;i<points.size();++i)unchecked_+=cv::norm(points[i]-points[i-1]);
}
bool FiberReverseCheckSchedule::check(const std::vector<cv::Vec3d>& points,bool terminal,const Verify& verify)
{
    if(interval_<=0)return true;
    update(points);
    const size_t last=points.size()-1;
    if(last<=checked_)return true;
    if(!pending_ && !terminal && unchecked_>=interval_)pending_=last;
    if(pending_ && (terminal || last-*pending_>=delay_)) {
        if(!verify(*pending_,std::min(checked_,*pending_),false))return false;
        accept(points,*pending_);pending_.reset();
    }
    if(terminal && last>checked_) {
        if(!verify(last,checked_,true))return false;
        accept(points,last);
    }
    return true;
}
bool FiberReverseCheckSchedule::stopTail(const std::vector<cv::Vec3d>& points,const Verify& verify)
{
    if(interval_<=0)return true;
    const size_t last=points.size()-1;
    const size_t target=last>delay_ ? last-delay_ : 0;
    historyOnly_=target==0;
    if(target && !verify(target,std::min(checked_,target),false))return false;
    accept(points,last);pending_.reset();observed_=last;
    return true;
}

FiberReverseCheckpointResult verifyFiberReverseCheckpoint(
    const FiberPredictionSource& predictions,const vc::lasagna::NormalSampler& normals,
    const FiberTraceConfig& config,const std::vector<cv::Vec3d>& points,
    size_t checkpoint,size_t checked,double widthMean)
{
    const auto begin=std::chrono::steady_clock::now();
    auto reverse=config;
    if(reverse.widthRelativeTolerance>0)reverse.widthReferenceMean=widthMean;
    const auto path=makePolylineArcGeometry(points);
    ReverseSpanOptions options;
    options.initialDirections=fiberReverseFutureHistory(path,checkpoint,config.replayReverseDelaySteps);
    options.verifiedThroughArc=path.vertexArcs.at(checked);
    options.minimumTravel=config.replayReverseConvergenceMinBase;
    options.convergenceDistance=config.replayReverseConvergenceDistanceBase;
    options.convergenceAngleDegrees=config.replayReverseConvergenceAngleDegrees;
    options.convergenceSteps=config.replayReverseConvergenceSteps;
    const auto check=verifyFiberReverseSpanDetailed(predictions,normals,reverse,path,
        path.vertexArcs.at(checkpoint),config.replayReverseNormalRadiusBase,
        config.replayReverseTangentialRadiusBase,0,options);
    return {check.passed,check.travelled,check.converged,options.initialDirections.size(),
        std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()};
}

ReverseSpanVerifier::ReverseSpanVerifier(const PolylineArcGeometry& geometry,
    double startArc, const cv::Vec3d& startPoint,
    const vc::lasagna::NormalSampler& normals, double scale,
    double normalRadius, double tangentRadius, const ReverseSpanOptions& options)
    : geometry_(geometry), normals_(normals), scale_(scale),
      normalRadius_(normalRadius), tangentRadius_(tangentRadius),
      arc_(startArc), previous_(startPoint), options_(options)
{
    validateReverseSpan(geometry, startArc, scale, normalRadius, tangentRadius);
    for (double value : startPoint.val)
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite reverse start point");
    if (!std::isfinite(options.minimumTravel) || options.minimumTravel < 0 ||
        !std::isfinite(options.verifiedThroughArc) || options.verifiedThroughArc < 0 || options.verifiedThroughArc > startArc ||
        !std::isfinite(options.convergenceDistance) || options.convergenceDistance <= 0 ||
        !std::isfinite(options.convergenceAngleDegrees) || options.convergenceAngleDegrees <= 0 || options.convergenceAngleDegrees >= 90 ||
        options.convergenceSteps < 1 || options.convergenceSteps > 64)
        throw std::invalid_argument("invalid reverse convergence options");
}

std::vector<cv::Vec3d> fiberReverseFutureHistory(const PolylineArcGeometry& geometry,
    size_t checkpoint, size_t count)
{
    if (checkpoint >= geometry.points.size()) throw std::invalid_argument("invalid reverse checkpoint");
    const size_t end = checkpoint + std::min(count, geometry.points.size()-1-checkpoint);
    std::vector<cv::Vec3d> directions;
    for (size_t i=end; i>checkpoint; --i) {
        const auto delta=geometry.points[i-1]-geometry.points[i];
        const double length=cv::norm(delta);
        if (length > kEpsilon) directions.push_back(delta/length);
    }
    return directions;
}

bool ReverseSpanVerifier::observe(const cv::Vec3d& point)
{
    for (double value : point.val)
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite reverse point");
    const auto delta = point - previous_;
    const double advance = cv::norm(delta);
    if (!(advance > kEpsilon)) return arc_ > kEpsilon;
    previous_ = point;
    travelled_ += advance;
    const auto projection = projectPointToPolylineArc(geometry_, point,
        std::max(0.0, arc_ - 2.0 * advance), arc_);
    const auto measurement = measureFiberReplayThreshold(point, projection.point,
        normals_, scale_, normalRadius_, tangentRadius_ / normalRadius_);
    if (fiberReplayThresholdExceeded(measurement, normalRadius_) ||
        !(projection.arc < arc_ - kEpsilon)) return false;
    arc_ = projection.arc;
    if (options_.minimumTravel > 0 && travelled_ >= options_.minimumTravel &&
        options_.verifiedThroughArc > 0 && arc_ <= options_.verifiedThroughArc) {
        const auto tangent=samplePolylineArc(geometry_, arc_).tangent;
        const double cosine = -(delta/advance).dot(tangent);
        const bool close = projection.distance <= options_.convergenceDistance &&
            cosine >= std::cos(options_.convergenceAngleDegrees * std::acos(-1.0)/180.0);
        closeSteps_ = close ? closeSteps_+1 : 0;
        converged_ = closeSteps_ >= options_.convergenceSteps;
    } else closeSteps_=0;
    return arc_ > kEpsilon && !converged_;
}

ReverseSpanResult verifyFiberReverseSpanDetailed(const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler& normals, const FiberTraceConfig& config,
    const PolylineArcGeometry& geometry, double startArc,
    double normalRadiusBase, double tangentialRadiusBase, int historySteps,
    const ReverseSpanOptions& options)
{
    validateReverseSpan(geometry, startArc, config.traceToBaseScale,
        normalRadiusBase, tangentialRadiusBase);
    if (historySteps < 0 || historySteps > 64)
        throw std::invalid_argument("reverse history steps must be in [0,64]");
    if (historySteps > 0 && !options.initialDirections.empty())
        throw std::invalid_argument("cannot combine legacy and future reverse history");
    auto start = samplePolylineArc(geometry, startArc);
    std::vector<cv::Vec3d> prior;
    if (historySteps > 0) {
        const auto it = std::lower_bound(geometry.vertexArcs.begin(), geometry.vertexArcs.end(), startArc);
        const size_t end = std::min(static_cast<size_t>(it - geometry.vertexArcs.begin()), geometry.points.size() - 1);
        start.point = geometry.points[end];
        start.arc = geometry.vertexArcs[end];
        const size_t begin = end - std::min(end, static_cast<size_t>(historySteps));
        for (size_t i = begin; i < end; ++i) {
            const auto direction = geometry.points[i] - geometry.points[i + 1];
            const double length = std::sqrt(direction.dot(direction));
            if (length > kEpsilon) prior.push_back(direction / length);
        }
        if (!prior.empty()) start.tangent = -prior.back();
    }
    const double verifiedArc = historySteps > 0 ? start.arc : startArc;
    if (!options.initialDirections.empty()) {
        prior=options.initialDirections;
        start.tangent=-prior.back();
    }
    const double scale = config.traceToBaseScale;
    ReverseSpanVerifier verifier(geometry, verifiedArc, start.point, normals,
        scale, normalRadiusBase, tangentialRadiusBase, options);
    if (verifiedArc == 0) return verifier.result();
    // This check measures geometric reversibility, not a second forward
    // presence stop. Preserve the fixed width reference and all scoring terms.
    auto reverseConfig = config;
    reverseConfig.replayPresenceCutoff = 0;
    if (!options.initialDirections.empty()) reverseConfig.preserveInitialDirection = true;
    try {
        (void)traceFiberExtrapolation(predictions, start.point / scale,
            -start.tangent, std::max(3.0 * verifiedArc / scale,
                2.0 * config.stepVoxels), reverseConfig, &normals, {}, false,
            [&](const cv::Vec3d& tracePoint, double, int) {
                return verifier.observe(tracePoint * scale);
            }, prior);
    } catch (const std::invalid_argument& error) {
        // A prediction hole at the seed fails the check. Never hide invalid
        // configuration, transport failures or other sampling errors.
        if (std::string_view(error.what()) !=
            "fiber trace start point has no valid prediction direction")
            throw;
    }
    return verifier.result();
}

double verifyFiberReverseSpan(const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler& normals, const FiberTraceConfig& config,
    const PolylineArcGeometry& geometry, double startArc,
    double normalRadiusBase, double tangentialRadiusBase, int historySteps)
{
    return verifyFiberReverseSpanDetailed(predictions,normals,config,geometry,startArc,
        normalRadiusBase,tangentialRadiusBase,historySteps,{}).remainingArc;
}
} // namespace vc::fiber_tracer
