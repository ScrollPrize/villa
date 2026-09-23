#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>
#include <opencv2/core/types.hpp>

namespace vc::fiber_tracer
{

enum class FiberCrossSectionKind {
    Line,
    Polygon,
};

struct FiberCrossSectionAnnotation {
    static constexpr int SchemaVersion = 1;

    std::string id;
    FiberCrossSectionKind kind = FiberCrossSectionKind::Line;
    std::vector<cv::Vec3d> pointsXyz;
    cv::Vec3d planeOriginXyz{0.0, 0.0, 0.0};
    cv::Vec3d planeNormalXyz{0.0, 0.0, 1.0};
    cv::Vec3d planeUpXyz{0.0, 1.0, 0.0};
    cv::Vec3d recordedLinePositionXyz{0.0, 0.0, 0.0};
    double recordedArclengthBaseVoxels = 0.0;
    std::uint64_t geometryGeneration = 1;
    bool detached = false;
};

[[nodiscard]] std::string fiberCrossSectionKindToString(FiberCrossSectionKind kind);
[[nodiscard]] FiberCrossSectionKind fiberCrossSectionKindFromString(const std::string& value);

// IDs are canonical RFC-4122 text without braces: 8-4-4-4-12 hexadecimal
// characters. Geometry is expressed in the owning fiber's XYZ coordinate
// domain. Polygon closure is implicit; pointsXyz never repeats the first point.
void validateFiberCrossSectionAnnotation(
    const FiberCrossSectionAnnotation& annotation, const std::string& context = "fiber cross-section annotation");
void validateFiberCrossSectionAnnotations(
    const std::vector<FiberCrossSectionAnnotation>& annotations, const std::string& context = "fiber cross-sections");

[[nodiscard]] FiberCrossSectionAnnotation fiberCrossSectionAnnotationFromJson(
    const nlohmann::json& value, const std::string& context = "fiber cross-section annotation");
[[nodiscard]] nlohmann::json fiberCrossSectionAnnotationToJson(const FiberCrossSectionAnnotation& annotation);

// Missing cross_sections means an empty collection. A present field is parsed
// strictly and duplicate annotation IDs are rejected.
[[nodiscard]] std::vector<FiberCrossSectionAnnotation> fiberCrossSectionAnnotationsFromJson(const nlohmann::json& fiberRoot, const std::string& context);
[[nodiscard]] nlohmann::json fiberCrossSectionAnnotationsToJson(const std::vector<FiberCrossSectionAnnotation>& annotations);

// Uniform coordinate-domain conversion. Directions remain unit vectors.
void scaleFiberCrossSectionAnnotation(FiberCrossSectionAnnotation& annotation, double scale);

// Distance to the nearest point on the polyline, including segment interiors.
// Empty input has infinite distance and a one-point input uses point distance.
[[nodiscard]] double fiberCrossSectionDistanceToPolyline(const std::vector<cv::Vec3d>& line, const cv::Vec3d& point);

}  // namespace vc::fiber_tracer
