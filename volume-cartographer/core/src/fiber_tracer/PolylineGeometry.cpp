#include "vc/fiber_tracer/PolylineGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
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

void validateBox(const cv::Vec3d& minimum, const cv::Vec3d& maximum)
{
    if (!finitePoint(minimum) || !finitePoint(maximum))
        throw std::invalid_argument("polyline clipping box must be finite");
    for (int axis = 0; axis < 3; ++axis) {
        if (!(maximum[axis] > minimum[axis])) {
            throw std::invalid_argument(
                "polyline clipping box must have positive extent");
        }
    }
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
    // Arc lengths are monotone. Include edges ending exactly at beginArc and
    // beginning exactly at endArc, as the original full scan did (including
    // duplicate-vertex runs). All visited edges keep their original order/math.
    const auto firstEnd=std::lower_bound(
        geometry.vertexArcs.begin()+1,geometry.vertexArcs.end(),beginArc);
    const size_t firstSegment=static_cast<size_t>(firstEnd-geometry.vertexArcs.begin()-1);
    for (size_t segment = firstSegment; segment + 1 < geometry.points.size(); ++segment) {
        const double sourceBegin = geometry.vertexArcs[segment];
        const double sourceEnd = geometry.vertexArcs[segment + 1];
        if(sourceBegin>endArc)break;
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

PolylineGridPartition partitionPolylineArcGrid(
    const PolylineArcGeometry& geometry,double coreSize)
{
    if(!(coreSize>0.) || !std::isfinite(coreSize))
        throw std::invalid_argument("Spatial core size must be positive and finite");
    if(geometry.points.size()!=geometry.vertexArcs.size())
        throw std::invalid_argument("Polyline arc array size differs");
    PolylineGridPartition result;
    const auto cellIndex=[&](double x) {
        const double value=std::floor(x/coreSize);
        // Use exclusive +/-2^63 bounds without an overflowing double->int cast.
        if(!std::isfinite(value) || value< -0x1p63 || value>=0x1p63)
            throw std::overflow_error("Spatial cell index exceeds int64");
        return static_cast<std::int64_t>(value);
    };
    const auto append=[&](double begin,double end,const auto& cell) {
        if(!(end>begin))return;
        if(!result.pieces.empty() && result.pieces.back().cell==cell &&
           result.boundaries.back().arc==begin) {
            result.boundaries.back().arc=end;
            return;
        }
        const auto first=result.boundaries.size();
        if(first==0)result.boundaries.push_back({begin,{0.,0.,0.}});
        else if(result.boundaries.back().arc!=begin)
            throw std::logic_error("Spatial fiber partition has an arc gap");
        const auto start=result.boundaries.size()-1;
        result.boundaries.push_back({end,{0.,0.,0.}});
        result.pieces.push_back({cell,start,result.boundaries.size()-1});
    };
    // Only crossed grid planes are visited, not a segment's 3D bounding box.
    // Geometry itself remains untouched; all coordinates use its shared sampler.
    std::vector<double> cuts;
    for(std::size_t segment=0;segment+1<geometry.points.size();++segment) {
        const auto& a=geometry.points[segment];const auto& b=geometry.points[segment+1];
        if(!finitePoint(a)||!finitePoint(b))throw std::invalid_argument("Nonfinite fiber point");
        const double firstArc=geometry.vertexArcs[segment],lastArc=geometry.vertexArcs[segment+1];
        if(!(lastArc>firstArc))continue;
        cuts.assign({0.,1.});
        for(int axis=0;axis<3;++axis) {
            const double delta=b[axis]-a[axis];
            if(delta==0.)continue;
            const auto low=cellIndex(std::min(a[axis],b[axis]));
            const auto high=cellIndex(std::max(a[axis],b[axis]));
            for(auto cell=low;cell<high;) {
                ++cell;
                const double t=(static_cast<double>(cell)*coreSize-a[axis])/delta;
                if(t>0. && t<1.)cuts.push_back(t);
            }
        }
        std::sort(cuts.begin(),cuts.end());cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
        for(std::size_t j=1;j<cuts.size();++j) {
            const double start=cuts[j-1],finish=cuts[j];
            const auto middle=a+(b-a)*std::midpoint(start,finish);
            std::array<std::int64_t,3> cell;
            for(int axis=0;axis<3;++axis)cell[axis]=cellIndex(middle[axis]);
            const double begin=start==0.?firstArc:firstArc+(lastArc-firstArc)*start;
            const double end=finish==1.?lastArc:firstArc+(lastArc-firstArc)*finish;
            append(begin,end,cell);
        }
    }
    for(auto& boundary:result.boundaries)
        boundary.point=samplePolylineArc(geometry,boundary.arc).point;
    return result;
}

bool pointInHalfOpenBox(
    const cv::Vec3d& point,
    const cv::Vec3d& minimum,
    const cv::Vec3d& maximum)
{
    validateBox(minimum, maximum);
    if (!finitePoint(point))
        throw std::invalid_argument("polyline clipping point must be finite");
    for (int axis = 0; axis < 3; ++axis) {
        if (!(point[axis] >= minimum[axis] && point[axis] < maximum[axis]))
            return false;
    }
    return true;
}

std::optional<BoxClippedLineSegment> clipLineSegmentToHalfOpenBox(
    const cv::Vec3d& start,
    const cv::Vec3d& finish,
    const cv::Vec3d& minimum,
    const cv::Vec3d& maximum)
{
    validateBox(minimum, maximum);
    if (!finitePoint(start) || !finitePoint(finish)) {
        throw std::invalid_argument(
            "polyline clipping segment endpoints must be finite");
    }
    const cv::Vec3d delta = finish - start;
    double begin = 0.0;
    double end = 1.0;
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(delta[axis]) <= kEpsilon) {
            if (!(start[axis] >= minimum[axis] &&
                  start[axis] < maximum[axis])) {
                return std::nullopt;
            }
            continue;
        }
        double near = (minimum[axis] - start[axis]) / delta[axis];
        double far = (maximum[axis] - start[axis]) / delta[axis];
        if (near > far)
            std::swap(near, far);
        begin = std::max(begin, near);
        end = std::min(end, far);
        if (!(end > begin + kEpsilon))
            return std::nullopt;
    }
    return BoxClippedLineSegment{
        start + delta * begin,
        start + delta * end,
        begin,
        end,
    };
}

std::vector<std::vector<cv::Vec3d>> clipPolylineToHalfOpenBox(
    const std::vector<cv::Vec3d>& points,
    const cv::Vec3d& minimum,
    const cv::Vec3d& maximum,
    std::vector<std::pair<double, double>>* sourceArcs)
{
    validateBox(minimum, maximum);
    for (const auto& point : points) {
        if (!finitePoint(point))
            throw std::invalid_argument("polyline clipping input must be finite");
    }
    std::vector<std::vector<cv::Vec3d>> runs;
    if (sourceArcs) sourceArcs->clear();
    double arc=0, beginArc=0, endArc=0;
    std::vector<cv::Vec3d> current;
    const auto finishCurrent = [&] {
        if (current.size() >= 2) {
            runs.push_back(std::move(current));
            if (sourceArcs) sourceArcs->emplace_back(beginArc,endArc);
        }
        current.clear();
    };
    for (std::size_t index = 1; index < points.size(); ++index) {
        const double length=vectorLength(points[index]-points[index-1]);
        const double startArc=arc;arc+=length;
        const auto clipped = clipLineSegmentToHalfOpenBox(
            points[index - 1], points[index], minimum, maximum);
        if (!clipped) {
            finishCurrent();
            continue;
        }
        if (current.empty() ||
            vectorLength(current.back() - clipped->start) > kEpsilon) {
            finishCurrent();
            current.push_back(clipped->start);
            beginArc=startArc+length*clipped->beginFraction;
        }
        if (vectorLength(current.back() - clipped->finish) > kEpsilon)
            current.push_back(clipped->finish);
        endArc=startArc+length*clipped->endFraction;
        if (!pointInHalfOpenBox(points[index], minimum, maximum))
            finishCurrent();
    }
    finishCurrent();
    return runs;
}

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

PolylineArcProjection projectPointToPolylineSegment(
    const PolylineArcSegment& segment, const cv::Vec3d& point)
{
    const cv::Vec3d clippedDelta = segment.finish - segment.start;
    const double denominator = clippedDelta.dot(clippedDelta);
    const double fraction = denominator > kEpsilon
        ? std::clamp((point - segment.start).dot(clippedDelta) / denominator, 0.0, 1.0)
        : 0.0;
    const cv::Vec3d projected = segment.start + clippedDelta * fraction;
    const double arc = segment.beginArc +
        (segment.endArc - segment.beginArc) * fraction;
    return {projected, arc, vectorLength(point - projected), segment.segmentIndex};
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
        const auto projection = projectPointToPolylineSegment(segment, point);
        const auto& projected = projection.point;
        const double arc = projection.arc;
        const double distance = projection.distance;
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
