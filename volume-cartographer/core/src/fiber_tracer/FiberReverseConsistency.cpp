#include "vc/fiber_tracer/FiberReverseConsistency.hpp"
#include "vc/fiber_tracer/FiberReplayMetric.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>

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

ReverseSpanVerifier::ReverseSpanVerifier(const PolylineArcGeometry& geometry,
    double startArc, const cv::Vec3d& startPoint,
    const vc::lasagna::NormalSampler& normals, double scale,
    double normalRadius, double tangentRadius)
    : geometry_(geometry), normals_(normals), scale_(scale),
      normalRadius_(normalRadius), tangentRadius_(tangentRadius),
      arc_(startArc), previous_(startPoint)
{
    validateReverseSpan(geometry, startArc, scale, normalRadius, tangentRadius);
    for (double value : startPoint.val)
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite reverse start point");
}

bool ReverseSpanVerifier::observe(const cv::Vec3d& point)
{
    for (double value : point.val)
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite reverse point");
    const double advance = cv::norm(point - previous_);
    if (!(advance > kEpsilon)) return arc_ > kEpsilon;
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

double verifyFiberReverseSpan(const FiberPredictionSource& predictions,
    const vc::lasagna::NormalSampler& normals, const FiberTraceConfig& config,
    const PolylineArcGeometry& geometry, double startArc,
    double normalRadiusBase, double tangentialRadiusBase, int historySteps)
{
    validateReverseSpan(geometry, startArc, config.traceToBaseScale,
        normalRadiusBase, tangentialRadiusBase);
    if (historySteps < 0 || historySteps > 64)
        throw std::invalid_argument("reverse history steps must be in [0,64]");
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
    const double scale = config.traceToBaseScale;
    ReverseSpanVerifier verifier(geometry, verifiedArc, start.point, normals,
        scale, normalRadiusBase, tangentialRadiusBase);
    if (verifiedArc == 0) return 0;
    try {
        (void)traceFiberExtrapolation(predictions, start.point / scale,
            -start.tangent, std::max(3.0 * verifiedArc / scale,
                2.0 * config.stepVoxels), config, &normals, {}, false,
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
    return verifier.remainingArc();
}
} // namespace vc::fiber_tracer
