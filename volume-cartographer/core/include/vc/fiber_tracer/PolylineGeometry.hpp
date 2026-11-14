#pragma once

#include <cstddef>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>
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

struct PolylineArcSegment {
    cv::Vec3d start{0.0, 0.0, 0.0};
    cv::Vec3d finish{0.0, 0.0, 0.0};
    double beginArc = 0.0;
    double endArc = 0.0;
    size_t segmentIndex = 0;
};

struct ForwardPolylineMatch {
    double predictedArc = 0.0;
    double searchBeginArc = 0.0;
    double searchEndArc = 0.0;
    PolylineArcProjection projection;
};

struct BoxClippedLineSegment {
    cv::Vec3d start{0.0, 0.0, 0.0};
    cv::Vec3d finish{0.0, 0.0, 0.0};
    double beginFraction = 0.0;
    double endFraction = 1.0;
};

// A spatial piece refers to the ORIGINAL curve's arc coordinates; its ends
// are not new scientific fiber endpoints. Adjacent pieces share one boundary
// record, including its exact arc position and sampled coordinate.
struct PolylineGridBoundary {
    double arc = 0.;
    cv::Vec3d point{0.,0.,0.};
};
struct PolylineGridPiece {
    std::array<std::int64_t,3> cell{}; // XYZ, half-open cells
    std::size_t beginBoundary = 0, endBoundary = 0;
};
struct PolylineGridPartition {
    std::vector<PolylineGridBoundary> boundaries;
    std::vector<PolylineGridPiece> pieces; // source arc order, including reentries
};

[[nodiscard]] PolylineGridPartition partitionPolylineArcGrid(
    const PolylineArcGeometry& geometry, double coreSizeBaseVoxels);

[[nodiscard]] bool pointInHalfOpenBox(
    const cv::Vec3d& point,
    const cv::Vec3d& minimum,
    const cv::Vec3d& maximum);

[[nodiscard]] std::optional<BoxClippedLineSegment>
clipLineSegmentToHalfOpenBox(
    const cv::Vec3d& start,
    const cv::Vec3d& finish,
    const cv::Vec3d& minimum,
    const cv::Vec3d& maximum);

[[nodiscard]] std::vector<std::vector<cv::Vec3d>>
clipPolylineToHalfOpenBox(
    const std::vector<cv::Vec3d>& points,
    const cv::Vec3d& minimum,
    const cv::Vec3d& maximum,
    std::vector<std::pair<double, double>>* sourceArcs = nullptr);

[[nodiscard]] double segmentAabbDistanceSquared(
    const cv::Vec3d& start,
    const cv::Vec3d& end,
    const cv::Vec3d& low,
    const cv::Vec3d& high);

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

[[nodiscard]] PolylineArcProjection projectPointToPolylineSegment(
    const PolylineArcSegment& segment, const cv::Vec3d& point);

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
    double refineAdvanceFactor,
    std::optional<double> maximumArc = std::nullopt);

} // namespace vc::fiber_tracer
