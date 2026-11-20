#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace vc3d::fiber_line {

bool isFinitePoint(const cv::Vec3d& point);

size_t nearestLinePointIndex(const std::vector<cv::Vec3d>& linePoints,
                             const cv::Vec3d& controlPoint);
// Arc length of the line from linePoints[firstIndex] to linePoints[lastIndex]
// (voxels), skipping non-finite points; 0 unless lastIndex > firstIndex.
double polylineLengthRange(const std::vector<cv::Vec3d>& linePoints,
                           size_t firstIndex,
                           size_t lastIndex);
// The annotated line of a fiber as line indices [begin, end): a manual
// fiber's first to last control. The open tails the optimizer extrapolates
// past them are not fiber and every consumer ignores them; with fewer than
// two control points (automatic traces) the whole line. The one rule is
// vc::geometry::annotatedLineRange; this only adapts the point type.
struct LineIndexRange {
    size_t begin = 0, end = 0;
};
LineIndexRange annotatedLineRange(const std::vector<cv::Vec3d>& linePoints,
                                  const std::vector<cv::Vec3d>& controlPoints);
// The length of a fiber as annotated (annotatedLineRange), in voxels.
double annotatedLineLengthVx(const std::vector<cv::Vec3d>& linePoints,
                             const std::vector<cv::Vec3d>& controlPoints);
std::vector<double> cumulativePolylineArclengths(
    const std::vector<cv::Vec3d>& linePoints);
std::vector<double> cumulativePolylineArclengths(
    const std::vector<cv::Vec3f>& linePoints);
double arclengthAtLinePosition(const std::vector<double>& cumulativeArclengths,
                               double linePosition);
double linePositionAtArclength(const std::vector<double>& cumulativeArclengths,
                               double arclength);
std::vector<size_t> linePositionIndicesWithinArclengthRadius(
    const std::vector<double>& cumulativeArclengths,
    double linePosition,
    const std::vector<double>& candidateLinePositions,
    double radiusBaseVoxels);
// Interior positions are unrestricted. Positions outside the outer controls
// are limited by optimized-polyline arclength in base-volume voxels.
bool linePositionWithinControlExtrapolationDistance(
    const std::vector<double>& cumulativeArclengths,
    double linePosition,
    const std::vector<double>& controlLinePositions,
    double maxDistanceBaseVoxels);
std::optional<double> controlExtrapolationBoundaryLinePosition(
    const std::vector<double>& cumulativeArclengths,
    const std::vector<double>& controlLinePositions,
    int direction,
    double lineEndPosition,
    double maxDistanceBaseVoxels);

} // namespace vc3d::fiber_line
