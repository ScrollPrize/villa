#include "SpiralPointChain.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vc3d::spiral {
namespace {

constexpr float kMinimumSpanLength = 1e-4f;
constexpr float kMaximumTessellationChord = 10.0f;
constexpr float kMaximumChordError = 0.25f;
constexpr int kMinimumSubdivisionDepth = 4;
constexpr int kMaximumSubdivisionDepth = 12;
constexpr qreal kIntersectionEpsilon = 1e-7;

float volumeDistance(const cv::Vec3f& left, const cv::Vec3f& right)
{
    const cv::Vec3f delta = right - left;
    return std::sqrt(delta.dot(delta));
}

qreal surfaceDistance(const QPointF& left, const QPointF& right)
{
    return std::hypot(right.x() - left.x(), right.y() - left.y());
}

QPointF interpolatePoint(const QPointF& left, const QPointF& right, qreal fraction)
{
    return left * (1.0 - fraction) + right * fraction;
}

QPointF catmullRomPoint(
    const QPointF& p0,
    const QPointF& p1,
    const QPointF& p2,
    const QPointF& p3,
    qreal fraction)
{
    auto nextParameter = [](qreal parameter, const QPointF& left, const QPointF& right) {
        return parameter + std::sqrt(std::max(surfaceDistance(left, right), qreal{1e-12}));
    };
    const qreal t0 = 0.0;
    const qreal t1 = nextParameter(t0, p0, p1);
    const qreal t2 = nextParameter(t1, p1, p2);
    const qreal t3 = nextParameter(t2, p2, p3);
    const qreal t = t1 + std::clamp(fraction, qreal{0.0}, qreal{1.0}) * (t2 - t1);

    const QPointF a1 = interpolatePoint(p0, p1, (t - t0) / (t1 - t0));
    const QPointF a2 = interpolatePoint(p1, p2, (t - t1) / (t2 - t1));
    const QPointF a3 = interpolatePoint(p2, p3, (t - t2) / (t3 - t2));
    const QPointF b1 = interpolatePoint(a1, a2, (t - t0) / (t2 - t0));
    const QPointF b2 = interpolatePoint(a2, a3, (t - t1) / (t3 - t1));
    return interpolatePoint(b1, b2, (t - t1) / (t2 - t1));
}

struct CurveNode {
    qreal parameter = 0.0;
    QPointF surface;
    cv::Vec3f volume;
};

struct SpanCurve {
    QPointF p0;
    QPointF p1;
    QPointF p2;
    QPointF p3;
    const SurfacePointSampler* sampler = nullptr;

    QPointF surfaceAt(qreal parameter) const
    {
        if (parameter <= 0.0) return p1;
        if (parameter >= 1.0) return p2;
        return catmullRomPoint(p0, p1, p2, p3, parameter);
    }

    std::optional<cv::Vec3f> volumeAt(qreal parameter) const
    {
        return (*sampler)(surfaceAt(parameter));
    }
};

bool appendAdaptiveNodes(
    const SpanCurve& curve,
    const CurveNode& left,
    const CurveNode& right,
    int depth,
    std::vector<CurveNode>& output)
{
    const qreal middleParameter = (left.parameter + right.parameter) * 0.5;
    const QPointF middleSurface = curve.surfaceAt(middleParameter);
    const auto middleVolume = (*curve.sampler)(middleSurface);
    if (!middleVolume) return false;

    const cv::Vec3f chordMiddle = (left.volume + right.volume) * 0.5f;
    const float chordError = volumeDistance(chordMiddle, *middleVolume);
    const float chordLength = volumeDistance(left.volume, right.volume);
    const bool subdivide = depth < kMinimumSubdivisionDepth
        || chordLength > kMaximumTessellationChord
        || chordError > kMaximumChordError;
    if (subdivide && depth >= kMaximumSubdivisionDepth) return false;
    if (subdivide) {
        const CurveNode middle{middleParameter, middleSurface, *middleVolume};
        if (!appendAdaptiveNodes(curve, left, middle, depth + 1, output))
            return false;
        return appendAdaptiveNodes(curve, middle, right, depth + 1, output);
    }
    output.push_back(right);
    return true;
}

qreal orientation(const QPointF& a, const QPointF& b, const QPointF& c)
{
    return (b.x() - a.x()) * (c.y() - a.y())
        - (b.y() - a.y()) * (c.x() - a.x());
}

bool pointOnSegment(const QPointF& point, const QPointF& start, const QPointF& end)
{
    if (std::abs(orientation(start, end, point)) > kIntersectionEpsilon) return false;
    return point.x() >= std::min(start.x(), end.x()) - kIntersectionEpsilon
        && point.x() <= std::max(start.x(), end.x()) + kIntersectionEpsilon
        && point.y() >= std::min(start.y(), end.y()) - kIntersectionEpsilon
        && point.y() <= std::max(start.y(), end.y()) + kIntersectionEpsilon;
}

bool segmentsIntersect(
    const QPointF& a,
    const QPointF& b,
    const QPointF& c,
    const QPointF& d)
{
    const qreal o1 = orientation(a, b, c);
    const qreal o2 = orientation(a, b, d);
    const qreal o3 = orientation(c, d, a);
    const qreal o4 = orientation(c, d, b);
    const bool proper = ((o1 > kIntersectionEpsilon && o2 < -kIntersectionEpsilon)
                         || (o1 < -kIntersectionEpsilon && o2 > kIntersectionEpsilon))
        && ((o3 > kIntersectionEpsilon && o4 < -kIntersectionEpsilon)
            || (o3 < -kIntersectionEpsilon && o4 > kIntersectionEpsilon));
    return proper || pointOnSegment(c, a, b) || pointOnSegment(d, a, b)
        || pointOnSegment(a, c, d) || pointOnSegment(b, c, d);
}

bool selfIntersects(const std::vector<QPointF>& points)
{
    if (points.size() < 4) return false;
    for (std::size_t first = 0; first + 1 < points.size(); ++first) {
        for (std::size_t second = first + 2; second + 1 < points.size(); ++second) {
            if (segmentsIntersect(points[first], points[first + 1],
                                  points[second], points[second + 1]))
                return true;
        }
    }
    return false;
}

} // namespace

PointChainBuildResult buildPointChain(
    const std::vector<PointChainAnchor>& anchors,
    const SurfacePointSampler& sampleSurface,
    float spacing)
{
    PointChainBuildResult result;
    if (anchors.empty()) return result;
    if (!sampleSurface || !std::isfinite(spacing) || spacing <= 0.0f) {
        result.error = PointChainBuildError::InvalidSurface;
        return result;
    }
    result.samples.push_back({anchors.front().surface, anchors.front().volume, true, 0});
    if (anchors.size() == 1) return result;

    std::vector<QPointF> tessellatedCurve;
    tessellatedCurve.push_back(anchors.front().surface);
    for (std::size_t span = 0; span + 1 < anchors.size(); ++span) {
        const auto& start = anchors[span];
        const auto& end = anchors[span + 1];
        if (volumeDistance(start.volume, end.volume) < kMinimumSpanLength
            || surfaceDistance(start.surface, end.surface) < kIntersectionEpsilon) {
            result.samples.clear();
            result.error = PointChainBuildError::DegenerateSpan;
            return result;
        }

        const QPointF p0 = span > 0
            ? anchors[span - 1].surface
            : start.surface * 2.0 - end.surface;
        const QPointF p3 = span + 2 < anchors.size()
            ? anchors[span + 2].surface
            : end.surface * 2.0 - start.surface;
        const SpanCurve curve{p0, start.surface, end.surface, p3, &sampleSurface};
        std::vector<CurveNode> nodes;
        nodes.push_back({0.0, start.surface, start.volume});
        const CurveNode last{1.0, end.surface, end.volume};
        if (!appendAdaptiveNodes(curve, nodes.front(), last, 0, nodes)) {
            result.samples.clear();
            result.error = PointChainBuildError::InvalidSurface;
            return result;
        }

        std::vector<float> cumulative(nodes.size(), 0.0f);
        for (std::size_t index = 1; index < nodes.size(); ++index) {
            const float distance = volumeDistance(nodes[index - 1].volume, nodes[index].volume);
            if (!std::isfinite(distance)) {
                result.samples.clear();
                result.error = PointChainBuildError::InvalidSurface;
                return result;
            }
            cumulative[index] = cumulative[index - 1] + distance;
            tessellatedCurve.push_back(nodes[index].surface);
        }
        const float spanLength = cumulative.back();
        if (spanLength < kMinimumSpanLength) {
            result.samples.clear();
            result.error = PointChainBuildError::DegenerateSpan;
            return result;
        }

        std::size_t bracket = 1;
        for (float target = spacing; target < spanLength - kMinimumSpanLength;
             target += spacing) {
            while (bracket + 1 < cumulative.size() && cumulative[bracket] < target)
                ++bracket;
            const float lowerDistance = cumulative[bracket - 1];
            const float interval = cumulative[bracket] - lowerDistance;
            const qreal fraction = interval > kMinimumSpanLength
                ? std::clamp((target - lowerDistance) / interval, 0.0f, 1.0f)
                : 0.0f;
            const qreal parameter = nodes[bracket - 1].parameter
                + (nodes[bracket].parameter - nodes[bracket - 1].parameter) * fraction;
            const QPointF surface = curve.surfaceAt(parameter);
            const auto volume = sampleSurface(surface);
            if (!volume) {
                result.samples.clear();
                result.error = PointChainBuildError::InvalidSurface;
                return result;
            }
            result.samples.push_back({surface, *volume, false, span});
        }
        result.samples.push_back({end.surface, end.volume, true, span + 1});
    }

    if (selfIntersects(tessellatedCurve)) {
        result.samples.clear();
        result.error = PointChainBuildError::SelfIntersection;
    }
    return result;
}

AnchorEraseDecision classifyAnchorErase(const std::vector<bool>& touchedAnchors)
{
    AnchorEraseDecision decision;
    if (touchedAnchors.empty()
        || std::none_of(touchedAnchors.begin(), touchedAnchors.end(), [](bool touched) {
               return touched;
           }))
        return decision;

    while (decision.removePrefix < touchedAnchors.size()
           && touchedAnchors[decision.removePrefix])
        ++decision.removePrefix;
    while (decision.removeSuffix + decision.removePrefix < touchedAnchors.size()
           && touchedAnchors[touchedAnchors.size() - 1 - decision.removeSuffix])
        ++decision.removeSuffix;

    for (std::size_t index = decision.removePrefix;
         index + decision.removeSuffix < touchedAnchors.size(); ++index) {
        if (touchedAnchors[index]) {
            decision.action = AnchorEraseAction::DeleteChain;
            return decision;
        }
    }
    if (touchedAnchors.size() - decision.removePrefix - decision.removeSuffix < 2) {
        decision.action = AnchorEraseAction::DeleteChain;
        return decision;
    }
    decision.action = AnchorEraseAction::Trim;
    return decision;
}

} // namespace vc3d::spiral
