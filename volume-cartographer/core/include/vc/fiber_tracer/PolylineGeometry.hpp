#pragma once

#include <cstddef>
#include <vector>

#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

struct PolylineArcGeometry {
    std::vector<cv::Vec3d> points;
    std::vector<double> vertexArcs;

    [[nodiscard]] double length() const noexcept;
};

struct PolylineArcSample {
    cv::Vec3d point{0.0, 0.0, 0.0};
    cv::Vec3d tangent{0.0, 0.0, 0.0};
    double arc = 0.0;
    size_t segmentIndex = 0;
};

struct PolylineArcProjection {
    cv::Vec3d point{0.0, 0.0, 0.0};
    double arc = 0.0;
    double distance = 0.0;
    size_t segmentIndex = 0;
};

struct ForwardPolylineMatch {
    double predictedArc = 0.0;
    double searchBeginArc = 0.0;
    double searchEndArc = 0.0;
    PolylineArcProjection projection;
};

[[nodiscard]] PolylineArcGeometry makePolylineArcGeometry(
    const std::vector<cv::Vec3d>& points);

[[nodiscard]] PolylineArcSample samplePolylineArc(
    const PolylineArcGeometry& geometry,
    double arc);

[[nodiscard]] std::vector<cv::Vec3d> slicePolylineArc(
    const PolylineArcGeometry& geometry,
    double beginArc,
    double endArc);

[[nodiscard]] PolylineArcProjection projectPointToPolylineArc(
    const PolylineArcGeometry& geometry,
    const cv::Vec3d& point,
    double beginArc,
    double endArc);

[[nodiscard]] double distanceToPolylineArc(
    const PolylineArcGeometry& geometry,
    const cv::Vec3d& point,
    double beginArc,
    double endArc);

[[nodiscard]] ForwardPolylineMatch matchForwardPolylinePoint(
    const PolylineArcGeometry& geometry,
    const cv::Vec3d& point,
    double previousArc,
    double expectedAdvance,
    double refineAdvanceFactor);

} // namespace vc::fiber_tracer
