#include "FiberLineGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>

namespace vc3d::fiber_line {
namespace {

constexpr double kEpsilon = 1.0e-10;

} // namespace

bool isFinitePoint(const cv::Vec3d& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
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

double polylineLengthRange(const std::vector<cv::Vec3d>& linePoints,
                           size_t firstIndex,
                           size_t lastIndex)
{
    if (linePoints.size() < 2 || firstIndex >= linePoints.size() ||
        lastIndex >= linePoints.size() || lastIndex <= firstIndex) {
        return 0.0;
    }
    double length = 0.0;
    for (size_t i = firstIndex + 1; i <= lastIndex; ++i) {
        if (!isFinitePoint(linePoints[i - 1]) || !isFinitePoint(linePoints[i])) {
            continue;
        }
        const cv::Vec3d delta = linePoints[i] - linePoints[i - 1];
        const double step = std::sqrt(delta.dot(delta));
        if (std::isfinite(step)) {
            length += step;
        }
    }
    return length;
}

double annotatedLineLengthVx(const std::vector<cv::Vec3d>& linePoints,
                             const std::vector<cv::Vec3d>& controlPoints)
{
    if (linePoints.size() < 2) {
        return 0.0;
    }
    size_t first = linePoints.size();
    size_t last = 0;
    size_t finiteControls = 0;
    for (const cv::Vec3d& control : controlPoints) {
        if (!isFinitePoint(control)) {
            continue;
        }
        ++finiteControls;
        const size_t index = nearestLinePointIndex(linePoints, control);
        first = std::min(first, index);
        last = std::max(last, index);
    }
    if (finiteControls < 2) {
        return polylineLengthRange(linePoints, 0, linePoints.size() - 1);
    }
    return polylineLengthRange(linePoints, first, last);
}

namespace {

template <typename Point>
std::vector<double> buildCumulativePolylineArclengths(
    const std::vector<Point>& linePoints)
{
    if (linePoints.empty()) {
        return {};
    }
    std::vector<double> cumulative(linePoints.size(), 0.0);
    for (size_t index = 1; index < linePoints.size(); ++index) {
        const auto& previous = linePoints[index - 1];
        const auto& current = linePoints[index];
        const bool finite = std::isfinite(previous[0]) &&
                            std::isfinite(previous[1]) &&
                            std::isfinite(previous[2]) &&
                            std::isfinite(current[0]) &&
                            std::isfinite(current[1]) &&
                            std::isfinite(current[2]);
        if (!finite) {
            return {};
        }
        const double dx = static_cast<double>(current[0]) -
                          static_cast<double>(previous[0]);
        const double dy = static_cast<double>(current[1]) -
                          static_cast<double>(previous[1]);
        const double dz = static_cast<double>(current[2]) -
                          static_cast<double>(previous[2]);
        const double step = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (!std::isfinite(step)) {
            return {};
        }
        cumulative[index] = cumulative[index - 1] + step;
    }
    return cumulative;
}

}  // namespace

std::vector<double> cumulativePolylineArclengths(
    const std::vector<cv::Vec3d>& linePoints)
{
    return buildCumulativePolylineArclengths(linePoints);
}

std::vector<double> cumulativePolylineArclengths(
    const std::vector<cv::Vec3f>& linePoints)
{
    return buildCumulativePolylineArclengths(linePoints);
}

double arclengthAtLinePosition(const std::vector<double>& cumulativeArclengths,
                               double linePosition)
{
    if (cumulativeArclengths.empty() || !std::isfinite(linePosition)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    linePosition = std::clamp(
        linePosition, 0.0, static_cast<double>(cumulativeArclengths.size() - 1));
    const size_t lower = static_cast<size_t>(std::floor(linePosition));
    const size_t upper = std::min(lower + 1, cumulativeArclengths.size() - 1);
    const double t = linePosition - static_cast<double>(lower);
    return cumulativeArclengths[lower] * (1.0 - t) +
           cumulativeArclengths[upper] * t;
}

double linePositionAtArclength(const std::vector<double>& cumulativeArclengths,
                               double arclength)
{
    if (cumulativeArclengths.empty() || !std::isfinite(arclength)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    arclength = std::clamp(arclength, 0.0, cumulativeArclengths.back());
    const auto upper = std::lower_bound(cumulativeArclengths.begin(),
                                        cumulativeArclengths.end(), arclength);
    if (upper == cumulativeArclengths.end()) {
        return static_cast<double>(cumulativeArclengths.size() - 1);
    }
    const size_t upperIndex = static_cast<size_t>(
        std::distance(cumulativeArclengths.begin(), upper));
    if (upperIndex == 0 || std::abs(*upper - arclength) <= kEpsilon) {
        return static_cast<double>(upperIndex);
    }
    size_t lowerIndex = upperIndex - 1;
    while (lowerIndex > 0 &&
           cumulativeArclengths[upperIndex] - cumulativeArclengths[lowerIndex] <=
               kEpsilon) {
        --lowerIndex;
    }
    const double span = cumulativeArclengths[upperIndex] -
                        cumulativeArclengths[lowerIndex];
    if (span <= kEpsilon) {
        return static_cast<double>(lowerIndex);
    }
    return static_cast<double>(lowerIndex) +
           (arclength - cumulativeArclengths[lowerIndex]) / span *
               static_cast<double>(upperIndex - lowerIndex);
}

std::vector<size_t> linePositionIndicesWithinArclengthRadius(
    const std::vector<double>& cumulativeArclengths,
    double linePosition,
    const std::vector<double>& candidateLinePositions,
    double radiusBaseVoxels)
{
    std::vector<size_t> matches;
    if (!std::isfinite(radiusBaseVoxels) || radiusBaseVoxels < 0.0) {
        return matches;
    }
    const double targetArclength = arclengthAtLinePosition(
        cumulativeArclengths, linePosition);
    if (!std::isfinite(targetArclength)) {
        return matches;
    }
    constexpr double kDistanceTolerance = 1.0e-6;
    for (size_t index = 0; index < candidateLinePositions.size(); ++index) {
        const double candidateArclength = arclengthAtLinePosition(
            cumulativeArclengths, candidateLinePositions[index]);
        if (std::isfinite(candidateArclength) &&
            std::abs(candidateArclength - targetArclength) <=
                radiusBaseVoxels + kDistanceTolerance) {
            matches.push_back(index);
        }
    }
    return matches;
}

bool linePositionWithinControlExtrapolationDistance(
    const std::vector<double>& cumulativeArclengths,
    double linePosition,
    const std::vector<double>& controlLinePositions,
    double maxDistanceBaseVoxels)
{
    if (!std::isfinite(linePosition)) {
        return false;
    }
    if (!std::isfinite(maxDistanceBaseVoxels) ||
        maxDistanceBaseVoxels <= 0.0) {
        return true;
    }

    std::optional<double> firstControl;
    std::optional<double> lastControl;
    for (const double position : controlLinePositions) {
        if (!std::isfinite(position)) {
            continue;
        }
        firstControl = firstControl ? std::min(*firstControl, position) : position;
        lastControl = lastControl ? std::max(*lastControl, position) : position;
    }
    if (!firstControl || !lastControl) {
        return false;
    }
    if (linePosition >= *firstControl && linePosition <= *lastControl) {
        return true;
    }

    const double boundaryPosition = linePosition < *firstControl
        ? *firstControl
        : *lastControl;
    const double lineArclength = arclengthAtLinePosition(
        cumulativeArclengths, linePosition);
    const double boundaryArclength = arclengthAtLinePosition(
        cumulativeArclengths, boundaryPosition);
    constexpr double kDistanceTolerance = 1.0e-6;
    return std::isfinite(lineArclength) && std::isfinite(boundaryArclength) &&
           std::abs(lineArclength - boundaryArclength) <=
               maxDistanceBaseVoxels + kDistanceTolerance;
}

std::optional<double> controlExtrapolationBoundaryLinePosition(
    const std::vector<double>& cumulativeArclengths,
    const std::vector<double>& controlLinePositions,
    int direction,
    double lineEndPosition,
    double maxDistanceBaseVoxels)
{
    if (direction == 0 || !std::isfinite(lineEndPosition) ||
        cumulativeArclengths.empty()) {
        return std::nullopt;
    }

    std::optional<double> outerControl;
    for (const double position : controlLinePositions) {
        if (!std::isfinite(position)) {
            continue;
        }
        if (!outerControl ||
            (direction > 0 ? position > *outerControl : position < *outerControl)) {
            outerControl = position;
        }
    }
    if (!outerControl ||
        (direction > 0 ? lineEndPosition <= *outerControl
                       : lineEndPosition >= *outerControl)) {
        return std::nullopt;
    }

    const double outerArclength = arclengthAtLinePosition(
        cumulativeArclengths, *outerControl);
    const double endArclength = arclengthAtLinePosition(
        cumulativeArclengths, lineEndPosition);
    if (!std::isfinite(outerArclength) || !std::isfinite(endArclength)) {
        return std::nullopt;
    }

    double targetArclength = endArclength;
    if (std::isfinite(maxDistanceBaseVoxels) &&
        maxDistanceBaseVoxels > 0.0) {
        targetArclength = direction > 0
            ? std::min(endArclength,
                       outerArclength + maxDistanceBaseVoxels)
            : std::max(endArclength,
                       outerArclength - maxDistanceBaseVoxels);
    }
    const double targetPosition = linePositionAtArclength(
        cumulativeArclengths, targetArclength);
    if (!std::isfinite(targetPosition) ||
        (direction > 0 ? targetPosition <= *outerControl
                       : targetPosition >= *outerControl)) {
        return std::nullopt;
    }
    return targetPosition;
}

} // namespace vc3d::fiber_line
