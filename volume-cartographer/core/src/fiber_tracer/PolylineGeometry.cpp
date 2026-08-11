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

} // namespace

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
    beginArc = std::clamp(beginArc, 0.0, geometry.length());
    endArc = std::clamp(endArc, 0.0, geometry.length());
    PolylineArcProjection best;
    best.distance = std::numeric_limits<double>::infinity();
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
        const cv::Vec3d clippedStart = geometry.points[segment] +
            delta * ((clippedBegin - sourceBegin) / edgeLength);
        const cv::Vec3d clippedFinish = geometry.points[segment] +
            delta * ((clippedEnd - sourceBegin) / edgeLength);
        const cv::Vec3d clippedDelta = clippedFinish - clippedStart;
        const double denominator = clippedDelta.dot(clippedDelta);
        const double fraction = denominator > kEpsilon
            ? std::clamp((point - clippedStart).dot(clippedDelta) / denominator, 0.0, 1.0)
            : 0.0;
        const cv::Vec3d projected = clippedStart + clippedDelta * fraction;
        const double arc = clippedBegin + (clippedEnd - clippedBegin) * fraction;
        const double distance = vectorLength(point - projected);
        if (!found || distance < best.distance - kEpsilon ||
            (std::abs(distance - best.distance) <= kEpsilon &&
             (arc < best.arc - kEpsilon ||
              (std::abs(arc - best.arc) <= kEpsilon && segment < best.segmentIndex)))) {
            best = {projected, arc, distance, segment};
            found = true;
        }
    }
    if (!found) {
        const auto sample = samplePolylineArc(geometry, beginArc);
        return {sample.point, sample.arc, vectorLength(point - sample.point), sample.segmentIndex};
    }
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

} // namespace vc::fiber_tracer
