#include "vc/fiber_tracer/FiberReferenceReplayCore.hpp"

#include "vc/fiber_tracer/FiberReplayMetric.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace vc::fiber_tracer
{
namespace
{
constexpr double kEpsilon = 1.0e-12;
}  // namespace

ReverseSpanVerifier::ReverseSpanVerifier(const PolylineArcGeometry& geometry,
    double startArc, const cv::Vec3d& startPoint,
    const vc::lasagna::NormalSampler& normals, double scale,
    double normalRadius, double tangentRadius)
    : geometry_(geometry), normals_(normals), scale_(scale),
      normalRadius_(normalRadius), tangentRadius_(tangentRadius),
      arc_(startArc), previous_(startPoint)
{
    if (!std::isfinite(normalRadius) || !(normalRadius > 0) ||
        !std::isfinite(tangentRadius) || !(tangentRadius > 0))
        throw std::invalid_argument("reverse radii must be finite and positive");
}

bool ReverseSpanVerifier::observe(const cv::Vec3d& point)
{
    const double advance = cv::norm(point - previous_);
    if (!(advance > kEpsilon)) return true;
    previous_ = point;
    const auto projection = projectPointToPolylineArc(geometry_, point,
        std::max(0.0, arc_ - 2.0 * advance), arc_);
    const auto measurement = measureFiberReplayThreshold(point, projection.point,
        normals_, scale_, normalRadius_, tangentRadius_ / normalRadius_);
    if (fiberReplayThresholdExceeded(measurement, normalRadius_) ||
        !(projection.arc < arc_ - kEpsilon)) return false;
    arc_ = projection.arc;
    return arc_ > kEpsilon;
}

FiberReplayTraceResult traceReferenceReplayCore(
    const PolylineArcGeometry& reference,
    double startArcBase,
    double endArcBase,
    double nominalStepBase,
    double maxStepFactor,
    double errorThresholdBaseVoxels,
    double excursionGraceBaseVoxels,
    double tangentialThresholdFactor,
    double matchRefineSteps,
    const vc::lasagna::NormalSampler& normalSampler,
    double normalWorkingToBaseScale,
    const ReplaySegmentRunner& runSegment,
    const FiberTraceProgressCallback& progress,
    const FiberReplayFailureCallback& failureCallback,
    const ReplayConsistencyPolicy& consistency)
{
    if (!std::isfinite(consistency.intervalBase) || consistency.intervalBase < 0 ||
        (consistency.intervalBase > 0 && (!consistency.verify || !consistency.nextSeed)) ||
        (consistency.acceptPoint && !consistency.nextSeed))
        throw std::invalid_argument("invalid replay consistency policy");
    if (!std::isfinite(excursionGraceBaseVoxels) || excursionGraceBaseVoxels < 0)
        throw std::invalid_argument("replay excursion grace must be finite and nonnegative");
    const double remainingArcBase = endArcBase - startArcBase;
    FiberReplayTraceResult result;
    result.tangentialThresholdFactor = tangentialThresholdFactor;
    result.referenceBeginArcBase = startArcBase;
    result.referenceEndArcBase = endArcBase;
    result.completedReferenceArcBase = startArcBase;
    const size_t maximumSegments =
        static_cast<size_t>(std::ceil(remainingArcBase / nominalStepBase)) + 1;

    double segmentStartArc = startArcBase;
    std::optional<PolylineArcSample> predictedSeed;
    for (size_t iteration = 0; iteration < maximumSegments &&
         segmentStartArc < endArcBase - kEpsilon; ++iteration) {
        FiberReplayTraceSegment segment;
        segment.startReferenceArcBase = segmentStartArc;
        segment.endReferenceArcBase = segmentStartArc;
        const auto start = predictedSeed.value_or(samplePolylineArc(reference, segmentStartArc));
        predictedSeed.reset();
        segment.tracePointsBase.push_back(start.point);
        segment.cumulativeLosses.push_back(0.0);

        double previousArcBase = segmentStartArc;
        bool referenceExhausted = false;
        bool distanceFailed = false;
        bool filtered = false;
        bool presenceFiltered = false;
        size_t checkedPoint = 0;
        size_t checkedMatches = 0;
        double checkedArc = segmentStartArc;
        double uncheckedTravel = 0;
        const auto checkSpan = [&]() {
            if (consistency.intervalBase <= 0 || segment.tracePointsBase.size() <= checkedPoint + 1)
                return true;
            ++result.reverseChecks;
            const std::vector<cv::Vec3d> span(segment.tracePointsBase.begin() + (consistency.toStart ? 0 : checkedPoint),
                                             segment.tracePointsBase.end());
            if (!consistency.verify(span)) { filtered = true; return false; }
            checkedPoint = segment.tracePointsBase.size() - 1;
            checkedMatches = segment.matches.size();
            checkedArc = previousArcBase;
            uncheckedTravel = 0;
            return true;
        };
        FiberReplayExcursion excursion{excursionGraceBaseVoxels, std::nullopt};
        const ReplayPointObserver observe =
            [&](const cv::Vec3d& pointBase, double cumulativeLoss, int) {
                if (previousArcBase >= endArcBase - kEpsilon) {
                    referenceExhausted = true;
                    return false;
                }
                const auto forwardMatch = matchForwardPolylinePoint(
                    reference, pointBase, previousArcBase, nominalStepBase,
                    matchRefineSteps, endArcBase);
                if (!(forwardMatch.searchEndArc >
                      previousArcBase + kEpsilon)) {
                    referenceExhausted = true;
                    return false;
                }
                const auto& match = forwardMatch.projection;
                const auto thresholdMeasurement = measureFiberReplayThreshold(
                    pointBase, match.point, normalSampler,
                    normalWorkingToBaseScale, errorThresholdBaseVoxels, tangentialThresholdFactor);
                const double stepTravel = cv::norm(pointBase - segment.tracePointsBase.back());
                segment.tracePointsBase.push_back(pointBase);
                segment.cumulativeLosses.push_back(cumulativeLoss);
                segment.matches.push_back({
                    segment.tracePointsBase.size() - 1,
                    forwardMatch.predictedArc,
                    match.arc,
                    match.point,
                    previousArcBase,
                    forwardMatch.searchEndArc,
                    thresholdMeasurement,
                });
                previousArcBase = match.arc;
                segment.endReferenceArcBase = match.arc;
                distanceFailed = excursion.observe(fiberReplayThresholdExceeded(
                    thresholdMeasurement, errorThresholdBaseVoxels), stepTravel);
                if (previousArcBase >= endArcBase - kEpsilon)
                    referenceExhausted = true;
                if (referenceExhausted && excursion.outsideTravel) distanceFailed = true;
                uncheckedTravel += stepTravel;
                if (consistency.acceptPoint && !consistency.acceptPoint(pointBase,
                        segment.tracePointsBase[segment.tracePointsBase.size()-2])) {
                    filtered = true;
                    presenceFiltered = true;
                    return false;
                }
                if (consistency.acceptPoint && consistency.intervalBase <= 0) {
                    checkedPoint = segment.tracePointsBase.size()-1;
                    checkedMatches = segment.matches.size();
                    checkedArc = previousArcBase;
                }
                // Default policy preserves earlier GT failures. The opt-in terminal
                // check below audits any failure subsequently rejected by reverse tracing.
                if (!distanceFailed && consistency.intervalBase > 0 &&
                    (uncheckedTravel >= consistency.intervalBase || referenceExhausted))
                    if (!checkSpan()) return false;
                return !distanceFailed && !referenceExhausted;
            };

        FiberTraceProgressCallback segmentProgress;
        if (progress) {
            const size_t segmentIndex = result.segments.size();
            segmentProgress = [&, segmentIndex](const FiberTraceProgress& event) {
                auto replayEvent = event;
                replayEvent.replaySegmentIndex = segmentIndex;
                replayEvent.referenceArcBase = previousArcBase;
                replayEvent.referenceArcFraction = std::clamp(
                    (previousArcBase - startArcBase) / remainingArcBase,
                    0.0, 1.0);
                progress(replayEvent);
            };
        }
        const int maximumSteps = std::max(
            1,
            static_cast<int>(std::ceil(
                maxStepFactor * (endArcBase - segmentStartArc) /
                nominalStepBase)) + 1);
        const std::string terminationReason = runSegment(
            start, endArcBase - segmentStartArc, maximumSteps, observe,
            segmentProgress);

        if (excursion.outsideTravel) distanceFailed = true;

        // Optional terminal check includes ordinary GT-failure endpoints.
        // Its acceptance still depends only on the saved forward geometry.
        if (!filtered && (!distanceFailed || consistency.allEndpoints)) (void)checkSpan();
        if (filtered) {
            const double detectedArc = previousArcBase;
            if (distanceFailed) {
                const auto& p = segment.tracePointsBase.back();
                segment.filteredDistanceFailure = {p[0], p[1], p[2], detectedArc};
            }
            segment.tracePointsBase.resize(checkedPoint + 1);
            segment.cumulativeLosses.resize(checkedPoint + 1);
            segment.matches.resize(checkedMatches);
            segment.endReferenceArcBase = checkedArc;
            segment.terminationReason = presenceFiltered ? "presence_filtered" : "reverse_consistency_filtered";
            result.segments.push_back(std::move(segment));
            const double searchArc = std::max(detectedArc, segmentStartArc) + nominalStepBase;
            if (searchArc < endArcBase) predictedSeed = consistency.nextSeed(searchArc);
            const double nextArc = predictedSeed ? predictedSeed->arc : endArcBase;
            if (!(nextArc > segmentStartArc) || nextArc > endArcBase)
                throw std::logic_error("filtered replay reseed did not advance within reference");
            result.filteredStops.push_back({checkedArc, detectedArc, nextArc});
            segmentStartArc = nextArc;
            result.completedReferenceArcBase = nextArc;
            continue;
        }

        if (referenceExhausted && !distanceFailed) {
            segment.endReferenceArcBase = endArcBase;
            segment.terminationReason = "reference_end";
            result.segments.push_back(std::move(segment));
            result.completedReferenceArcBase = endArcBase;
            break;
        }

        FiberReplayFailure replayFailure;
        replayFailure.index = result.failures.size();
        replayFailure.segmentIndex = result.segments.size();
        replayFailure.reason = distanceFailed
            ? "distance_above_threshold"
            : terminationReason;
        replayFailure.referenceArcBase = previousArcBase;
        replayFailure.referenceArcFraction = std::clamp(
            (previousArcBase - startArcBase) / remainingArcBase, 0.0, 1.0);
        replayFailure.referencePointBase =
            samplePolylineArc(reference, previousArcBase).point;
        if (distanceFailed) {
            const auto& match = segment.matches.back();
            replayFailure.evaluatorPointBase = segment.tracePointsBase.back();
            replayFailure.segmentPointIndex = segment.tracePointsBase.size() - 1;
            replayFailure.thresholdMeasurement = match.thresholdMeasurement;
        }
        segment.terminationReason = replayFailure.reason;
        result.segments.push_back(std::move(segment));
        result.failures.push_back(replayFailure);
        if (failureCallback)
            failureCallback(result.failures.back());

        const double resetArc = std::min(
            endArcBase,
            std::max(previousArcBase, segmentStartArc + nominalStepBase));
        if (!(resetArc > segmentStartArc + kEpsilon))
            throw std::logic_error("fiber replay reset did not advance");
        segmentStartArc = resetArc;
        result.completedReferenceArcBase = resetArc;
    }
    if (result.completedReferenceArcBase < endArcBase - kEpsilon) {
        throw std::logic_error(
            "fiber replay exceeded its deterministic reset bound");
    }
    return result;
}

}  // namespace vc::fiber_tracer
