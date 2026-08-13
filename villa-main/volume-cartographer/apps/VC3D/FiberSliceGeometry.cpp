#include "FiberSliceGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>

namespace vc3d::fiber_slice {
namespace {

constexpr double kEpsilon = 1.0e-10;

double norm(const cv::Vec3d& value)
{
    return std::sqrt(value.dot(value));
}

cv::Vec3d stableOrientedNormal(cv::Vec3d normal)
{
    normal = normalizedOrZero(normal);
    if (norm(normal) <= kEpsilon) {
        return {0.0, 0.0, 1.0};
    }
    int axis = 0;
    if (std::abs(normal[1]) > std::abs(normal[axis])) {
        axis = 1;
    }
    if (std::abs(normal[2]) > std::abs(normal[axis])) {
        axis = 2;
    }
    if (normal[axis] < 0.0) {
        normal *= -1.0;
    }
    return normal;
}

cv::Vec3d fallbackInPlaneAxis(const cv::Vec3d& normal)
{
    const cv::Vec3d axes[] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    cv::Vec3d best{1.0, 0.0, 0.0};
    double bestProjectedNorm = -1.0;
    for (const auto& axis : axes) {
        const cv::Vec3d projected = axis - normal * axis.dot(normal);
        const double projectedNorm = norm(projected);
        if (projectedNorm > bestProjectedNorm) {
            bestProjectedNorm = projectedNorm;
            best = projected;
        }
    }
    return normalizedOrZero(best);
}

} // namespace

bool isFinitePoint(const cv::Vec3d& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

cv::Vec3d normalizedOrZero(const cv::Vec3d& value)
{
    if (!isFinitePoint(value)) {
        return {0.0, 0.0, 0.0};
    }
    const double length = norm(value);
    if (length <= kEpsilon) {
        return {0.0, 0.0, 0.0};
    }
    return value * (1.0 / length);
}

cv::Vec3d projectPointToPlane(const cv::Vec3d& point, const Plane& plane)
{
    const cv::Vec3d normal = normalizedOrZero(plane.normal);
    return point - normal * ((point - plane.origin).dot(normal));
}

double signedDistanceToPlane(const cv::Vec3d& point, const Plane& plane)
{
    const cv::Vec3d normal = normalizedOrZero(plane.normal);
    return (point - plane.origin).dot(normal);
}

size_t nearestLinePointIndex(const std::vector<cv::Vec3d>& linePoints,
                             const cv::Vec3d& controlPoint)
{
    size_t bestIndex = 0;
    double bestDistance2 = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < linePoints.size(); ++i) {
        if (!isFinitePoint(linePoints[i])) {
            continue;
        }
        const cv::Vec3d delta = linePoints[i] - controlPoint;
        const double distance2 = delta.dot(delta);
        if (distance2 < bestDistance2) {
            bestDistance2 = distance2;
            bestIndex = i;
        }
    }
    return bestIndex;
}

ControlSpanSelection selectControlSpan(const std::vector<cv::Vec3d>& linePoints,
                                       const std::vector<cv::Vec3d>& controlPoints)
{
    ControlSpanSelection span;
    if (controlPoints.size() < 2) {
        span.error = "At least two control points are required.";
        return span;
    }
    if (linePoints.empty()) {
        span.error = "The selected fiber has no line points.";
        return span;
    }

    size_t first = linePoints.size() - 1;
    size_t last = 0;
    bool anyControl = false;
    for (const cv::Vec3d& control : controlPoints) {
        if (!isFinitePoint(control)) {
            continue;
        }
        const size_t index = nearestLinePointIndex(linePoints, control);
        first = std::min(first, index);
        last = std::max(last, index);
        anyControl = true;
    }
    if (!anyControl) {
        span.error = "Control points contain no finite coordinates.";
        return span;
    }

    span.firstLineIndex = first;
    span.lastLineIndex = last;
    cv::Vec3d centroid{0.0, 0.0, 0.0};
    for (size_t i = first; i <= last && i < linePoints.size(); ++i) {
        if (!isFinitePoint(linePoints[i])) {
            continue;
        }
        span.samples.push_back(linePoints[i]);
        centroid += linePoints[i];
    }
    if (span.samples.size() < 3) {
        span.error = "At least three finite line points are required between the first and last control point.";
        return span;
    }
    span.centroid = centroid * (1.0 / static_cast<double>(span.samples.size()));
    span.valid = true;
    return span;
}

PlaneFit fitLeastSquaresPlane(const ControlSpanSelection& span,
                              const std::vector<cv::Vec3d>& linePoints)
{
    PlaneFit fit;
    if (!span.valid || span.samples.size() < 3) {
        fit.error = span.error.empty()
            ? "At least three finite line points are required to fit a plane."
            : span.error;
        return fit;
    }

    cv::Mat data(static_cast<int>(span.samples.size()), 3, CV_64F);
    for (int row = 0; row < data.rows; ++row) {
        const cv::Vec3d& point = span.samples[static_cast<size_t>(row)];
        data.at<double>(row, 0) = point[0];
        data.at<double>(row, 1) = point[1];
        data.at<double>(row, 2) = point[2];
    }

    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    if (pca.eigenvectors.rows < 3 || pca.eigenvectors.cols < 3) {
        fit.error = "Could not compute a stable plane fit.";
        return fit;
    }

    cv::Vec3d normal{
        pca.eigenvectors.at<double>(2, 0),
        pca.eigenvectors.at<double>(2, 1),
        pca.eigenvectors.at<double>(2, 2),
    };
    normal = stableOrientedNormal(normal);
    if (norm(normal) <= kEpsilon) {
        fit.error = "Could not compute a finite plane normal.";
        return fit;
    }

    cv::Vec3d direction{0.0, 0.0, 0.0};
    if (span.lastLineIndex < linePoints.size() && span.firstLineIndex < linePoints.size()) {
        direction = linePoints[span.lastLineIndex] - linePoints[span.firstLineIndex];
    }
    cv::Vec3d upHint = direction - normal * direction.dot(normal);
    upHint = normalizedOrZero(upHint);
    if (norm(upHint) <= kEpsilon) {
        upHint = fallbackInPlaneAxis(normal);
    }

    fit.origin = span.centroid;
    fit.normal = normal;
    fit.upHint = upHint;
    fit.valid = true;
    return fit;
}

double viewportMinVoxelSpan(double visibleWidthVx, double visibleHeightVx)
{
    if (!std::isfinite(visibleWidthVx) || !std::isfinite(visibleHeightVx) ||
        visibleWidthVx <= 0.0 || visibleHeightVx <= 0.0) {
        return 1.0;
    }
    return std::max(1.0, std::min(visibleWidthVx, visibleHeightVx));
}

double distanceScaledSize(double distanceToPlane,
                          double minVisibleViewportSpanVx,
                          double fullSize,
                          double minSize)
{
    const double span = viewportMinVoxelSpan(minVisibleViewportSpanVx, minVisibleViewportSpanVx);
    const double fullDistance = 0.01 * span;
    const double minDistance = 0.10 * span;
    if (distanceToPlane <= fullDistance) {
        return fullSize;
    }
    if (distanceToPlane >= minDistance || minDistance <= fullDistance) {
        return minSize;
    }
    const double t = (distanceToPlane - fullDistance) / (minDistance - fullDistance);
    return fullSize + (minSize - fullSize) * std::clamp(t, 0.0, 1.0);
}

ArclengthSample samplePolylineAtArclength(const std::vector<cv::Vec3d>& linePoints,
                                          double arclength)
{
    ArclengthSample sample;
    if (linePoints.size() < 2 || !std::isfinite(arclength)) {
        return sample;
    }

    double accumulated = 0.0;
    std::optional<cv::Vec3d> previous;
    size_t previousIndex = 0;
    for (size_t currentIndex = 0; currentIndex < linePoints.size(); ++currentIndex) {
        const cv::Vec3d& current = linePoints[currentIndex];
        if (!isFinitePoint(current)) {
            previous.reset();
            continue;
        }
        if (!previous) {
            previous = current;
            previousIndex = currentIndex;
            continue;
        }

        const cv::Vec3d segment = current - *previous;
        const double segmentLength = norm(segment);
        if (segmentLength <= kEpsilon) {
            previous = current;
            continue;
        }

        if (arclength <= accumulated + segmentLength) {
            const double t = std::clamp((arclength - accumulated) / segmentLength, 0.0, 1.0);
            sample.valid = true;
            sample.point = *previous + segment * t;
            sample.tangent = segment * (1.0 / segmentLength);
            sample.arclength = accumulated + segmentLength * t;
            sample.linePosition = static_cast<double>(previousIndex) +
                                  (static_cast<double>(currentIndex - previousIndex) * t);
            return sample;
        }

        accumulated += segmentLength;
        previous = current;
        previousIndex = currentIndex;
    }

    if (previous && accumulated > kEpsilon) {
        for (size_t i = linePoints.size(); i-- > 0;) {
            if (!isFinitePoint(linePoints[i])) {
                continue;
            }
            for (size_t j = i; j-- > 0;) {
                if (!isFinitePoint(linePoints[j])) {
                    break;
                }
                const cv::Vec3d segment = linePoints[i] - linePoints[j];
                const double segmentLength = norm(segment);
                if (segmentLength <= kEpsilon) {
                    continue;
                }
                sample.valid = true;
                sample.point = linePoints[i];
                sample.tangent = segment * (1.0 / segmentLength);
                sample.arclength = accumulated;
                sample.linePosition = static_cast<double>(i);
                return sample;
            }
            break;
        }
    }
    return sample;
}

double linePositionAtArclength(const std::vector<cv::Vec3d>& linePoints,
                               double arclength)
{
    const ArclengthSample sample = samplePolylineAtArclength(linePoints, arclength);
    return sample.valid ? sample.linePosition : std::numeric_limits<double>::quiet_NaN();
}

ControlTripletSelection selectControlTriplet(const std::vector<cv::Vec3d>& linePoints,
                                             const std::vector<cv::Vec3d>& controlPoints,
                                             double currentLinePosition,
                                             const cv::Vec3d& currentPoint)
{
    ControlTripletSelection selection;
    if (linePoints.empty() || !std::isfinite(currentLinePosition) || !isFinitePoint(currentPoint)) {
        return selection;
    }

    const double maxLinePosition = static_cast<double>(linePoints.size() - 1);
    selection.currentLinePosition = std::clamp(currentLinePosition, 0.0, maxLinePosition);
    selection.currentPoint = currentPoint;

    struct Candidate {
        double position = 0.0;
        cv::Vec3d point{0.0, 0.0, 0.0};
    };
    std::vector<Candidate> candidates;
    candidates.reserve(controlPoints.size() + 2);
    if (isFinitePoint(linePoints.front())) {
        candidates.push_back({0.0, linePoints.front()});
    }
    if (linePoints.size() > 1 && isFinitePoint(linePoints.back())) {
        candidates.push_back({maxLinePosition, linePoints.back()});
    }
    for (const cv::Vec3d& control : controlPoints) {
        if (!isFinitePoint(control)) {
            continue;
        }
        candidates.push_back({static_cast<double>(nearestLinePointIndex(linePoints, control)),
                              control});
    }
    if (candidates.empty()) {
        return selection;
    }

    const Candidate* previous = nullptr;
    const Candidate* next = nullptr;
    const Candidate* nearest = nullptr;
    double bestPreviousDistance = std::numeric_limits<double>::infinity();
    double bestNextDistance = std::numeric_limits<double>::infinity();
    double bestNearestDistance = std::numeric_limits<double>::infinity();
    for (const Candidate& candidate : candidates) {
        const double signedDistance = candidate.position - selection.currentLinePosition;
        const double absDistance = std::abs(signedDistance);
        if (absDistance < bestNearestDistance) {
            bestNearestDistance = absDistance;
            nearest = &candidate;
        }
        if (signedDistance < -1.0e-9 && -signedDistance < bestPreviousDistance) {
            bestPreviousDistance = -signedDistance;
            previous = &candidate;
        } else if (signedDistance > 1.0e-9 && signedDistance < bestNextDistance) {
            bestNextDistance = signedDistance;
            next = &candidate;
        }
    }
    if (!previous) {
        previous = nearest;
    }
    if (!next) {
        next = nearest;
    }
    if (!previous || !next) {
        return selection;
    }

    selection.previousLinePosition = previous->position;
    selection.previousPoint = previous->point;
    selection.nextLinePosition = next->position;
    selection.nextPoint = next->point;
    selection.valid = true;
    return selection;
}

PlaneFit planeFromNormalAndTangent(const cv::Vec3d& origin,
                                   const cv::Vec3d& normal,
                                   const cv::Vec3d& tangent)
{
    PlaneFit fit;
    if (!isFinitePoint(origin)) {
        fit.error = "Plane origin is not finite.";
        return fit;
    }

    cv::Vec3d planeNormal = stableOrientedNormal(normal);
    if (norm(planeNormal) <= kEpsilon) {
        fit.error = "Plane normal is not finite.";
        return fit;
    }

    cv::Vec3d upHint = tangent - planeNormal * tangent.dot(planeNormal);
    upHint = normalizedOrZero(upHint);
    if (norm(upHint) <= kEpsilon) {
        upHint = fallbackInPlaneAxis(planeNormal);
    }

    fit.valid = true;
    fit.origin = origin;
    fit.normal = planeNormal;
    fit.upHint = upHint;
    return fit;
}

PlaneFit planeFromDirections(const cv::Vec3d& origin,
                             const cv::Vec3d& firstDirection,
                             const cv::Vec3d& secondDirection)
{
    PlaneFit fit;
    if (!isFinitePoint(origin)) {
        fit.error = "Plane origin is not finite.";
        return fit;
    }

    cv::Vec3d first = normalizedOrZero(firstDirection);
    if (norm(first) <= kEpsilon) {
        fit.error = "Plane direction is not finite.";
        return fit;
    }

    cv::Vec3d second = secondDirection - first * secondDirection.dot(first);
    second = normalizedOrZero(second);
    if (norm(second) <= kEpsilon) {
        second = fallbackInPlaneAxis(first);
    }

    cv::Vec3d normal{
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    };
    normal = stableOrientedNormal(normal);
    if (norm(normal) <= kEpsilon) {
        fit.error = "Could not compute a finite plane normal.";
        return fit;
    }

    fit.valid = true;
    fit.origin = origin;
    fit.normal = normal;
    fit.upHint = first;
    return fit;
}

double connectorNormalizedThickness(double distanceToSliceVx,
                                    double maxDistanceVx,
                                    double fullSize,
                                    double minSize)
{
    if (!std::isfinite(distanceToSliceVx) || distanceToSliceVx <= 0.0) {
        return fullSize;
    }
    const double span = std::max(std::abs(maxDistanceVx), kEpsilon);
    const double t = std::clamp(distanceToSliceVx / span, 0.0, 1.0);
    return fullSize + (minSize - fullSize) * t;
}

double focusedIntersectionMarkerThreshold(double minVisibleViewportSpanVx,
                                          double viewportFraction)
{
    if (!std::isfinite(minVisibleViewportSpanVx) ||
        !std::isfinite(viewportFraction) ||
        viewportFraction < 0.0) {
        return 0.0;
    }
    return viewportFraction * viewportMinVoxelSpan(minVisibleViewportSpanVx,
                                                   minVisibleViewportSpanVx);
}

bool focusedIntersectionMarkerVisible(double distanceToPlane,
                                      double minVisibleViewportSpanVx,
                                      double viewportFraction)
{
    if (!std::isfinite(distanceToPlane)) {
        return false;
    }
    return std::abs(distanceToPlane) <=
           focusedIntersectionMarkerThreshold(minVisibleViewportSpanVx, viewportFraction);
}

std::optional<SegmentPlaneIntersection> segmentPlaneIntersection(const cv::Vec3d& p0,
                                                                 const cv::Vec3d& p1,
                                                                 const Plane& plane)
{
    if (!isFinitePoint(p0) || !isFinitePoint(p1)) {
        return std::nullopt;
    }
    const cv::Vec3d tangent = p1 - p0;
    const double tangentLength = norm(tangent);
    if (tangentLength <= kEpsilon) {
        return std::nullopt;
    }

    const double d0 = signedDistanceToPlane(p0, plane);
    const double d1 = signedDistanceToPlane(p1, plane);
    if (!std::isfinite(d0) || !std::isfinite(d1)) {
        return std::nullopt;
    }
    if (std::abs(d0) <= kEpsilon && std::abs(d1) <= kEpsilon) {
        return std::nullopt;
    }
    if ((d0 > kEpsilon && d1 > kEpsilon) || (d0 < -kEpsilon && d1 < -kEpsilon)) {
        return std::nullopt;
    }

    const double denom = d0 - d1;
    if (std::abs(denom) <= kEpsilon) {
        return std::nullopt;
    }
    const double t = std::clamp(d0 / denom, 0.0, 1.0);
    const cv::Vec3d normal = normalizedOrZero(plane.normal);
    const double dot = std::clamp(std::abs((tangent * (1.0 / tangentLength)).dot(normal)), 0.0, 1.0);
    const double angleDegrees = std::acos(dot) * 180.0 / std::acos(-1.0);

    return SegmentPlaneIntersection{
        p0 + tangent * t,
        tangent * (1.0 / tangentLength),
        angleDegrees,
    };
}

double intersectionOpacityForAngle(double angleDegrees)
{
    if (!std::isfinite(angleDegrees)) {
        return 0.0;
    }
    if (angleDegrees <= 45.0) {
        return 1.0;
    }
    if (angleDegrees >= 90.0) {
        return 0.0;
    }
    return std::clamp((90.0 - angleDegrees) / 45.0, 0.0, 1.0);
}

EllipseStyle ellipseStyleForAngle(double angleDegrees, double baseRadius)
{
    const double t = std::clamp((angleDegrees - 45.0) / 45.0, 0.0, 1.0);
    return EllipseStyle{
        std::max(0.25, baseRadius * (1.0 + 3.0 * t)),
        std::max(0.25, baseRadius * (1.0 - 0.65 * t)),
        intersectionOpacityForAngle(angleDegrees),
    };
}

} // namespace vc3d::fiber_slice
