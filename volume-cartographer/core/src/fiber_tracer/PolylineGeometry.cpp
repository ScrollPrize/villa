#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-12;

double vectorLength(const cv::Vec3d& value)
{
    return std::sqrt(value.dot(value));
}

bool finitePoint(const cv::Vec3d& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) &&
        std::isfinite(point[2]);
}

size_t segmentAtArc(const PolylineArcGeometry& geometry, double arc)
{
    const auto upper = std::upper_bound(
        geometry.vertexArcs.begin(), geometry.vertexArcs.end(), arc);
    size_t segment = upper == geometry.vertexArcs.begin()
        ? 0
        : static_cast<size_t>(upper - geometry.vertexArcs.begin() - 1);
    if (segment + 1 >= geometry.points.size())
        segment = geometry.points.size() - 2;
    while (segment + 1 < geometry.points.size() &&
           geometry.vertexArcs[segment + 1] <= geometry.vertexArcs[segment] + kEpsilon) {
        ++segment;
    }
    if (segment + 1 >= geometry.points.size()) {
        segment = geometry.points.size() - 2;
        while (segment > 0 &&
               geometry.vertexArcs[segment + 1] <= geometry.vertexArcs[segment] + kEpsilon) {
            --segment;
        }
    }
    return segment;
}

template <typename Visitor>
void visitClippedPolylineArcSegments(
    const PolylineArcGeometry& geometry,
    double beginArc,
    double endArc,
    Visitor&& visitor)
{
    if (!std::isfinite(beginArc) || !std::isfinite(endArc) || beginArc > endArc)
        throw std::invalid_argument("polyline interval is invalid");
    beginArc = std::clamp(beginArc, 0.0, geometry.length());
    endArc = std::clamp(endArc, 0.0, geometry.length());
    bool found = false;
    for (size_t segment = 0; segment + 1 < geometry.points.size(); ++segment) {
        const double sourceBegin = geometry.vertexArcs[segment];
        const double sourceEnd = geometry.vertexArcs[segment + 1];
        if (!(sourceEnd > sourceBegin + kEpsilon) || sourceEnd < beginArc ||
            sourceBegin > endArc) {
            continue;
        }
        const double clippedBegin = std::max(beginArc, sourceBegin);
        const double clippedEnd = std::min(endArc, sourceEnd);
        const cv::Vec3d delta = geometry.points[segment + 1] - geometry.points[segment];
        const double edgeLength = sourceEnd - sourceBegin;
        visitor(PolylineArcSegment{
            geometry.points[segment] +
                delta * ((clippedBegin - sourceBegin) / edgeLength),
            geometry.points[segment] +
                delta * ((clippedEnd - sourceBegin) / edgeLength),
            clippedBegin,
            clippedEnd,
            segment,
        });
        found = true;
    }
    if (!found) {
        const auto sample = samplePolylineArc(geometry, beginArc);
        visitor(PolylineArcSegment{
            sample.point,
            sample.point,
            sample.arc,
            sample.arc,
            sample.segmentIndex,
        });
    }
}

} // namespace

double segmentAabbDistanceSquared(
    const cv::Vec3d& start,
    const cv::Vec3d& end,
    const cv::Vec3d& low,
    const cv::Vec3d& high)
{
    constexpr float geometryEpsilon = 1.0e-6F;
    const cv::Vec3d delta = end - start;
    std::vector<double> breaks{0.0F, 1.0F};
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(delta[axis]) <= geometryEpsilon)
            continue;
        for (const double bound : {low[axis], high[axis]}) {
            const double t = (bound - start[axis]) / delta[axis];
            if (t > 0.0F && t < 1.0F)
                breaks.push_back(t);
        }
    }
    std::sort(breaks.begin(), breaks.end());
    breaks.erase(std::unique(breaks.begin(), breaks.end()), breaks.end());
    double best = std::numeric_limits<double>::infinity();
    const auto evaluate = [&](double t) {
        const cv::Vec3d point = start + delta * t;
        double squared = 0.0F;
        for (int axis = 0; axis < 3; ++axis) {
            const double outside = point[axis] < low[axis]
                ? low[axis] - point[axis]
                : point[axis] > high[axis]
                    ? point[axis] - high[axis]
                    : 0.0F;
            squared += outside * outside;
        }
        best = std::min(best, squared);
    };
    for (size_t interval = 0; interval + 1 < breaks.size(); ++interval) {
        const double begin = breaks[interval];
        const double finish = breaks[interval + 1];
        evaluate(begin);
        evaluate(finish);
        const double middle = 0.5F * (begin + finish);
        double quadratic = 0.0F;
        double linear = 0.0F;
        for (int axis = 0; axis < 3; ++axis) {
            const double point = start[axis] + delta[axis] * middle;
            double offset = 0.0F;
            if (point < low[axis])
                offset = start[axis] - low[axis];
            else if (point > high[axis])
                offset = start[axis] - high[axis];
            else
                continue;
            quadratic += delta[axis] * delta[axis];
            linear += delta[axis] * offset;
        }
        if (quadratic > geometryEpsilon)
            evaluate(std::clamp(-linear / quadratic, begin, finish));
    }
    return best;
}

double PolylineArcGeometry::length() const noexcept
{
    return vertexArcs.empty() ? 0.0 : vertexArcs.back();
}

PolylineArcGeometry makePolylineArcGeometry(const std::vector<cv::Vec3d>& points)
{
    if (points.size() < 2)
        throw std::invalid_argument("polyline requires at least two points");
    PolylineArcGeometry geometry;
    geometry.points = points;
    geometry.vertexArcs.resize(points.size(), 0.0);
    bool hasEdge = false;
    for (size_t index = 0; index < points.size(); ++index) {
        if (!finitePoint(points[index]))
            throw std::invalid_argument("polyline contains a non-finite point");
        if (index == 0)
            continue;
        const double edgeLength = vectorLength(points[index] - points[index - 1]);
        if (!std::isfinite(edgeLength))
            throw std::invalid_argument("polyline contains a non-finite edge");
        geometry.vertexArcs[index] = geometry.vertexArcs[index - 1] + edgeLength;
        hasEdge = hasEdge || edgeLength > kEpsilon;
    }
    if (!hasEdge)
        throw std::invalid_argument("polyline has no non-degenerate edge");
    return geometry;
}

ForwardPolylineArcInterval selectForwardPolylineArcInterval(
    const PolylineArcGeometry& geometry,
    size_t beginVertexIndex,
    std::optional<double> maximumLength)
{
    if (geometry.points.size() < 2 ||
        geometry.vertexArcs.size() != geometry.points.size() ||
        beginVertexIndex >= geometry.vertexArcs.size()) {
        throw std::invalid_argument("forward polyline interval start is invalid");
    }
    if (maximumLength.has_value() &&
        (!std::isfinite(*maximumLength) || !(*maximumLength > 0.0))) {
        throw std::invalid_argument(
            "forward polyline interval maximum length must be positive");
    }
    const double beginArc = geometry.vertexArcs[beginVertexIndex];
    const double endArc = maximumLength.has_value()
        ? std::min(geometry.length(), beginArc + *maximumLength)
        : geometry.length();
    if (!(endArc > beginArc))
        throw std::invalid_argument("forward polyline interval has no forward extent");
    return {beginArc, endArc};
}

PolylineArcSample samplePolylineArc(
    const PolylineArcGeometry& geometry,
    double inputArc)
{
    if (geometry.points.size() < 2 ||
        geometry.vertexArcs.size() != geometry.points.size()) {
        throw std::invalid_argument("polyline geometry is incomplete");
    }
    if (!std::isfinite(inputArc))
        throw std::invalid_argument("polyline sample arclength must be finite");
    const double arc = std::clamp(inputArc, 0.0, geometry.length());
    const size_t segment = segmentAtArc(geometry, arc);
    const double begin = geometry.vertexArcs[segment];
    const double end = geometry.vertexArcs[segment + 1];
    const cv::Vec3d delta = geometry.points[segment + 1] - geometry.points[segment];
    const double edgeLength = end - begin;
    if (!(edgeLength > kEpsilon))
        throw std::logic_error("polyline arclength selected a degenerate edge");
    const double fraction = std::clamp((arc - begin) / edgeLength, 0.0, 1.0);
    return {
        geometry.points[segment] + delta * fraction,
        delta / edgeLength,
        arc,
        segment,
    };
}

std::vector<cv::Vec3d> slicePolylineArc(
    const PolylineArcGeometry& geometry,
    double beginArc,
    double endArc)
{
    if (!std::isfinite(beginArc) || !std::isfinite(endArc) || beginArc > endArc)
        throw std::invalid_argument("polyline slice interval is invalid");
    beginArc = std::clamp(beginArc, 0.0, geometry.length());
    endArc = std::clamp(endArc, 0.0, geometry.length());
    std::vector<cv::Vec3d> result{samplePolylineArc(geometry, beginArc).point};
    for (size_t index = 1; index + 1 < geometry.points.size(); ++index) {
        if (geometry.vertexArcs[index] > beginArc + kEpsilon &&
            geometry.vertexArcs[index] < endArc - kEpsilon) {
            result.push_back(geometry.points[index]);
        }
    }
    const cv::Vec3d end = samplePolylineArc(geometry, endArc).point;
    if (vectorLength(end - result.back()) > kEpsilon)
        result.push_back(end);
    return result;
}

std::vector<PolylineArcSegment> clippedPolylineArcSegments(
    const PolylineArcGeometry& geometry,
    double beginArc,
    double endArc)
{
    std::vector<PolylineArcSegment> segments;
    segments.reserve(geometry.points.size());
    visitClippedPolylineArcSegments(
        geometry, beginArc, endArc,
        [&](const PolylineArcSegment& segment) { segments.push_back(segment); });
    return segments;
}

PolylineArcProjection projectPointToPolylineArc(
    const PolylineArcGeometry& geometry,
    const cv::Vec3d& point,
    double beginArc,
    double endArc)
{
    if (!finitePoint(point) || !std::isfinite(beginArc) || !std::isfinite(endArc) ||
        beginArc > endArc) {
        throw std::invalid_argument("polyline projection input is invalid");
    }
    PolylineArcProjection best;
    best.distance = std::numeric_limits<double>::infinity();
    bool found = false;
    visitClippedPolylineArcSegments(
        geometry, beginArc, endArc, [&](const PolylineArcSegment& segment) {
        const cv::Vec3d clippedDelta = segment.finish - segment.start;
        const double denominator = clippedDelta.dot(clippedDelta);
        const double fraction = denominator > kEpsilon
            ? std::clamp((point - segment.start).dot(clippedDelta) / denominator, 0.0, 1.0)
            : 0.0;
        const cv::Vec3d projected = segment.start + clippedDelta * fraction;
        const double arc = segment.beginArc +
            (segment.endArc - segment.beginArc) * fraction;
        const double distance = vectorLength(point - projected);
        if (!found || distance < best.distance - kEpsilon ||
            (std::abs(distance - best.distance) <= kEpsilon &&
             (arc < best.arc - kEpsilon ||
              (std::abs(arc - best.arc) <= kEpsilon &&
               segment.segmentIndex < best.segmentIndex)))) {
            best = {projected, arc, distance, segment.segmentIndex};
            found = true;
        }
    });
    return best;
}

double distanceToPolylineArc(
    const PolylineArcGeometry& geometry,
    const cv::Vec3d& point,
    double beginArc,
    double endArc)
{
    return projectPointToPolylineArc(geometry, point, beginArc, endArc).distance;
}

ForwardPolylineMatch matchForwardPolylinePoint(
    const PolylineArcGeometry& geometry,
    const cv::Vec3d& point,
    double previousArc,
    double expectedAdvance,
    double refineAdvanceFactor,
    std::optional<double> maximumArc)
{
    if (!std::isfinite(previousArc) || !(expectedAdvance > 0.0) ||
        !std::isfinite(expectedAdvance) || !(refineAdvanceFactor >= 0.0) ||
        !std::isfinite(refineAdvanceFactor) ||
        (maximumArc.has_value() && !std::isfinite(*maximumArc))) {
        throw std::invalid_argument("forward polyline match input is invalid");
    }
    const double limit = maximumArc.has_value()
        ? std::clamp(*maximumArc, 0.0, geometry.length())
        : geometry.length();
    const double begin = std::clamp(previousArc, 0.0, limit);
    const double predicted = std::min(
        limit, begin + expectedAdvance);
    const double end = std::min(
        limit, predicted + refineAdvanceFactor * expectedAdvance);
    return {
        predicted,
        begin,
        end,
        projectPointToPolylineArc(geometry, point, begin, end),
    };
}

} // namespace vc::fiber_tracer
