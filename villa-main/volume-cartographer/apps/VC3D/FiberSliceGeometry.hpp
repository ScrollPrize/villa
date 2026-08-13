#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace vc3d::fiber_slice {

struct FiberInput {
    uint64_t id = 0;
    std::vector<cv::Vec3d> controlPoints;
    std::vector<cv::Vec3d> linePoints;
};

struct ControlSpanSelection {
    bool valid = false;
    size_t firstLineIndex = 0;
    size_t lastLineIndex = 0;
    std::vector<cv::Vec3d> samples;
    cv::Vec3d centroid{0.0, 0.0, 0.0};
    std::string error;
};

struct PlaneFit {
    bool valid = false;
    cv::Vec3d origin{0.0, 0.0, 0.0};
    cv::Vec3d normal{0.0, 0.0, 1.0};
    cv::Vec3d upHint{0.0, 1.0, 0.0};
    std::string error;
};

struct Plane {
    cv::Vec3d origin{0.0, 0.0, 0.0};
    cv::Vec3d normal{0.0, 0.0, 1.0};
};

struct ArclengthSample {
    bool valid = false;
    cv::Vec3d point{0.0, 0.0, 0.0};
    cv::Vec3d tangent{1.0, 0.0, 0.0};
    double arclength = 0.0;
    double linePosition = 0.0;
};

struct ControlTripletSelection {
    bool valid = false;
    double previousLinePosition = 0.0;
    double currentLinePosition = 0.0;
    double nextLinePosition = 0.0;
    cv::Vec3d previousPoint{0.0, 0.0, 0.0};
    cv::Vec3d currentPoint{0.0, 0.0, 0.0};
    cv::Vec3d nextPoint{0.0, 0.0, 0.0};
};

struct SegmentPlaneIntersection {
    cv::Vec3d point{0.0, 0.0, 0.0};
    cv::Vec3d tangent{0.0, 0.0, 0.0};
    double angleDegrees = 0.0;
};

struct EllipseStyle {
    double majorRadius = 3.0;
    double minorRadius = 3.0;
    double opacity = 1.0;
};

bool isFinitePoint(const cv::Vec3d& point);
cv::Vec3d normalizedOrZero(const cv::Vec3d& value);
cv::Vec3d projectPointToPlane(const cv::Vec3d& point, const Plane& plane);
double signedDistanceToPlane(const cv::Vec3d& point, const Plane& plane);

size_t nearestLinePointIndex(const std::vector<cv::Vec3d>& linePoints,
                             const cv::Vec3d& controlPoint);
ControlSpanSelection selectControlSpan(const std::vector<cv::Vec3d>& linePoints,
                                       const std::vector<cv::Vec3d>& controlPoints);
PlaneFit fitLeastSquaresPlane(const ControlSpanSelection& span,
                              const std::vector<cv::Vec3d>& linePoints);

double viewportMinVoxelSpan(double visibleWidthVx, double visibleHeightVx);
double distanceScaledSize(double distanceToPlane,
                          double minVisibleViewportSpanVx,
                          double fullSize,
                          double minSize);
ArclengthSample samplePolylineAtArclength(const std::vector<cv::Vec3d>& linePoints,
                                          double arclength);
double linePositionAtArclength(const std::vector<cv::Vec3d>& linePoints,
                               double arclength);
ControlTripletSelection selectControlTriplet(const std::vector<cv::Vec3d>& linePoints,
                                             const std::vector<cv::Vec3d>& controlPoints,
                                             double currentLinePosition,
                                             const cv::Vec3d& currentPoint);
PlaneFit planeFromNormalAndTangent(const cv::Vec3d& origin,
                                   const cv::Vec3d& normal,
                                   const cv::Vec3d& tangent);
PlaneFit planeFromDirections(const cv::Vec3d& origin,
                             const cv::Vec3d& firstDirection,
                             const cv::Vec3d& secondDirection);
double connectorNormalizedThickness(double distanceToSliceVx,
                                    double maxDistanceVx,
                                    double fullSize,
                                    double minSize);
double focusedIntersectionMarkerThreshold(double minVisibleViewportSpanVx,
                                          double viewportFraction = 0.05);
bool focusedIntersectionMarkerVisible(double distanceToPlane,
                                      double minVisibleViewportSpanVx,
                                      double viewportFraction = 0.05);

std::optional<SegmentPlaneIntersection> segmentPlaneIntersection(const cv::Vec3d& p0,
                                                                 const cv::Vec3d& p1,
                                                                 const Plane& plane);
double intersectionOpacityForAngle(double angleDegrees);
EllipseStyle ellipseStyleForAngle(double angleDegrees, double baseRadius);

} // namespace vc3d::fiber_slice
