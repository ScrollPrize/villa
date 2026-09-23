#include "vc/fiber_tracer/FiberCrossSection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kDirectionTolerance = 1.0e-5;
constexpr double kGeometryToleranceBaseVoxels = 1.0e-4;

bool finitePoint(const cv::Vec3d& point)
{
    return std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]);
}

double norm(const cv::Vec3d& value)
{
    return std::sqrt(value.dot(value));
}

void rejectUnknownKeys(const nlohmann::json& value, const std::unordered_set<std::string>& allowed, const std::string& context)
{
    if (!value.is_object()) {
        throw std::runtime_error(context + " must be an object");
    }
    for (const auto& [key, item] : value.items()) {
        (void)item;
        if (!allowed.contains(key)) {
            throw std::runtime_error(context + " contains unknown field: " + key);
        }
    }
}

cv::Vec3d pointFromJson(const nlohmann::json& value, const std::string& context)
{
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(context + " must be [x, y, z]");
    }
    cv::Vec3d point;
    for (int axis = 0; axis < 3; ++axis) {
        if (!value.at(static_cast<size_t>(axis)).is_number()) {
            throw std::runtime_error(context + " must contain three numbers");
        }
        point[axis] = value.at(static_cast<size_t>(axis)).get<double>();
    }
    if (!finitePoint(point)) {
        throw std::runtime_error(context + " contains non-finite coordinates");
    }
    return point;
}

nlohmann::json pointToJson(const cv::Vec3d& point)
{
    return nlohmann::json::array({point[0], point[1], point[2]});
}

bool canonicalUuid(const std::string& value)
{
    if (value.size() != 36) {
        return false;
    }
    for (size_t index = 0; index < value.size(); ++index) {
        if (index == 8 || index == 13 || index == 18 || index == 23) {
            if (value[index] != '-') {
                return false;
            }
            continue;
        }
        const unsigned char ch = static_cast<unsigned char>(value[index]);
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) {
            return false;
        }
    }
    return true;
}

struct Point2 {
    double x = 0.0;
    double y = 0.0;
};

double orient(const Point2& a, const Point2& b, const Point2& c)
{
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool within(double value, double a, double b)
{
    return value >= std::min(a, b) - kGeometryToleranceBaseVoxels && value <= std::max(a, b) + kGeometryToleranceBaseVoxels;
}

bool onSegment(const Point2& a, const Point2& b, const Point2& point)
{
    return std::abs(orient(a, b, point)) <= kGeometryToleranceBaseVoxels && within(point.x, a.x, b.x) && within(point.y, a.y, b.y);
}

int orientationSign(double value)
{
    if (value > kGeometryToleranceBaseVoxels)
        return 1;
    if (value < -kGeometryToleranceBaseVoxels)
        return -1;
    return 0;
}

bool segmentsIntersect(const Point2& a, const Point2& b, const Point2& c, const Point2& d)
{
    const int abc = orientationSign(orient(a, b, c));
    const int abd = orientationSign(orient(a, b, d));
    const int cda = orientationSign(orient(c, d, a));
    const int cdb = orientationSign(orient(c, d, b));
    if (abc != abd && cda != cdb) {
        return true;
    }
    return (abc == 0 && onSegment(a, b, c)) || (abd == 0 && onSegment(a, b, d)) || (cda == 0 && onSegment(c, d, a)) ||
           (cdb == 0 && onSegment(c, d, b));
}

void validatePolygon(const FiberCrossSectionAnnotation& annotation, const std::string& context)
{
    const cv::Vec3d planeRight = annotation.planeUpXyz.cross(annotation.planeNormalXyz);
    std::vector<Point2> points;
    points.reserve(annotation.pointsXyz.size());
    for (const auto& point : annotation.pointsXyz) {
        const cv::Vec3d local = point - annotation.planeOriginXyz;
        points.push_back({local.dot(planeRight), local.dot(annotation.planeUpXyz)});
    }

    for (size_t first = 0; first < points.size(); ++first) {
        for (size_t second = first + 1; second < points.size(); ++second) {
            const double dx = points[first].x - points[second].x;
            const double dy = points[first].y - points[second].y;
            if (std::hypot(dx, dy) <= kGeometryToleranceBaseVoxels) {
                throw std::runtime_error(context + " polygon contains duplicate vertices");
            }
        }
    }

    double twiceArea = 0.0;
    for (size_t index = 0; index < points.size(); ++index) {
        const Point2& current = points[index];
        const Point2& next = points[(index + 1) % points.size()];
        twiceArea += current.x * next.y - next.x * current.y;
    }
    if (std::abs(twiceArea) <= kGeometryToleranceBaseVoxels * kGeometryToleranceBaseVoxels) {
        throw std::runtime_error(context + " polygon has zero area");
    }

    for (size_t first = 0; first < points.size(); ++first) {
        const size_t firstNext = (first + 1) % points.size();
        for (size_t second = first + 1; second < points.size(); ++second) {
            const size_t secondNext = (second + 1) % points.size();
            if (first == second || firstNext == second || secondNext == first) {
                continue;
            }
            if (segmentsIntersect(points[first], points[firstNext], points[second], points[secondNext])) {
                throw std::runtime_error(context + " polygon self-intersects");
            }
        }
    }
}

}  // namespace

std::string fiberCrossSectionKindToString(FiberCrossSectionKind kind)
{
    switch (kind) {
        case FiberCrossSectionKind::Line:
            return "line";
        case FiberCrossSectionKind::Polygon:
            return "polygon";
    }
    throw std::runtime_error("unsupported fiber cross-section kind");
}

FiberCrossSectionKind fiberCrossSectionKindFromString(const std::string& value)
{
    if (value == "line")
        return FiberCrossSectionKind::Line;
    if (value == "polygon")
        return FiberCrossSectionKind::Polygon;
    throw std::runtime_error("unsupported fiber cross-section kind: " + value);
}

void validateFiberCrossSectionAnnotation(const FiberCrossSectionAnnotation& annotation, const std::string& context)
{
    if (!canonicalUuid(annotation.id)) {
        throw std::runtime_error(context + " id must be a canonical lowercase UUID");
    }
    if (!finitePoint(annotation.planeOriginXyz) || !finitePoint(annotation.planeNormalXyz) || !finitePoint(annotation.planeUpXyz) ||
        !finitePoint(annotation.recordedLinePositionXyz)) {
        throw std::runtime_error(context + " contains a non-finite frame or line position");
    }
    if (!std::isfinite(annotation.recordedArclengthBaseVoxels) || annotation.recordedArclengthBaseVoxels < 0.0) {
        throw std::runtime_error(context + " recorded arclength must be finite and non-negative");
    }
    if (annotation.geometryGeneration == 0) {
        throw std::runtime_error(context + " geometry generation must be positive");
    }
    const double normalLength = norm(annotation.planeNormalXyz);
    const double upLength = norm(annotation.planeUpXyz);
    if (std::abs(normalLength - 1.0) > kDirectionTolerance || std::abs(upLength - 1.0) > kDirectionTolerance ||
        std::abs(annotation.planeNormalXyz.dot(annotation.planeUpXyz)) > kDirectionTolerance) {
        throw std::runtime_error(context + " plane normal and up must be orthonormal");
    }
    const size_t expected = annotation.kind == FiberCrossSectionKind::Line ? 2 : 3;
    if ((annotation.kind == FiberCrossSectionKind::Line && annotation.pointsXyz.size() != expected) ||
        (annotation.kind == FiberCrossSectionKind::Polygon && annotation.pointsXyz.size() < expected)) {
        throw std::runtime_error(
            context + (annotation.kind == FiberCrossSectionKind::Line ? " line must contain exactly two points"
                                                                      : " polygon must contain at least three points"));
    }
    for (const auto& point : annotation.pointsXyz) {
        if (!finitePoint(point)) {
            throw std::runtime_error(context + " contains a non-finite geometry point");
        }
        const double planeDistance = std::abs((point - annotation.planeOriginXyz).dot(annotation.planeNormalXyz));
        if (planeDistance > kGeometryToleranceBaseVoxels) {
            throw std::runtime_error(context + " geometry point is outside its recorded plane");
        }
    }
    if (annotation.kind == FiberCrossSectionKind::Line) {
        if (norm(annotation.pointsXyz[1] - annotation.pointsXyz[0]) <= kGeometryToleranceBaseVoxels) {
            throw std::runtime_error(context + " line endpoints must be distinct");
        }
    } else {
        validatePolygon(annotation, context);
    }
}

void validateFiberCrossSectionAnnotations(const std::vector<FiberCrossSectionAnnotation>& annotations, const std::string& context)
{
    std::unordered_set<std::string> ids;
    ids.reserve(annotations.size());
    for (size_t index = 0; index < annotations.size(); ++index) {
        const std::string itemContext = context + "[" + std::to_string(index) + "]";
        validateFiberCrossSectionAnnotation(annotations[index], itemContext);
        if (!ids.insert(annotations[index].id).second) {
            throw std::runtime_error(context + " contains duplicate annotation id: " + annotations[index].id);
        }
    }
}

FiberCrossSectionAnnotation fiberCrossSectionAnnotationFromJson(const nlohmann::json& value, const std::string& context)
{
    rejectUnknownKeys(
        value,
        {"version",
         "id",
         "kind",
         "points_xyz",
         "plane_origin_xyz",
         "plane_normal_xyz",
         "plane_up_xyz",
         "recorded_line_position_xyz",
         "recorded_arclength_base_voxels",
         "geometry_generation",
         "detached"},
        context);
    const std::array<const char*, 11> required{
        "version",
        "id",
        "kind",
        "points_xyz",
        "plane_origin_xyz",
        "plane_normal_xyz",
        "plane_up_xyz",
        "recorded_line_position_xyz",
        "recorded_arclength_base_voxels",
        "geometry_generation",
        "detached"};
    for (const char* field : required) {
        if (!value.contains(field)) {
            throw std::runtime_error(context + " is missing field: " + field);
        }
    }
    if (!value.at("version").is_number_integer() || value.at("version").get<int>() != FiberCrossSectionAnnotation::SchemaVersion) {
        throw std::runtime_error(context + " has unsupported version");
    }
    if (!value.at("id").is_string() || !value.at("kind").is_string()) {
        throw std::runtime_error(context + " id and kind must be strings");
    }
    if (!value.at("points_xyz").is_array()) {
        throw std::runtime_error(context + " points_xyz must be an array");
    }
    if (!value.at("recorded_arclength_base_voxels").is_number() || !value.at("geometry_generation").is_number_unsigned() ||
        !value.at("detached").is_boolean()) {
        throw std::runtime_error(context + " has invalid provenance fields");
    }

    FiberCrossSectionAnnotation annotation;
    annotation.id = value.at("id").get<std::string>();
    annotation.kind = fiberCrossSectionKindFromString(value.at("kind").get<std::string>());
    annotation.pointsXyz.reserve(value.at("points_xyz").size());
    for (size_t index = 0; index < value.at("points_xyz").size(); ++index) {
        annotation.pointsXyz.push_back(pointFromJson(value.at("points_xyz").at(index), context + ".points_xyz[" + std::to_string(index) + "]"));
    }
    annotation.planeOriginXyz = pointFromJson(value.at("plane_origin_xyz"), context + ".plane_origin_xyz");
    annotation.planeNormalXyz = pointFromJson(value.at("plane_normal_xyz"), context + ".plane_normal_xyz");
    annotation.planeUpXyz = pointFromJson(value.at("plane_up_xyz"), context + ".plane_up_xyz");
    annotation.recordedLinePositionXyz = pointFromJson(value.at("recorded_line_position_xyz"), context + ".recorded_line_position_xyz");
    annotation.recordedArclengthBaseVoxels = value.at("recorded_arclength_base_voxels").get<double>();
    annotation.geometryGeneration = value.at("geometry_generation").get<std::uint64_t>();
    annotation.detached = value.at("detached").get<bool>();
    validateFiberCrossSectionAnnotation(annotation, context);
    return annotation;
}

nlohmann::json fiberCrossSectionAnnotationToJson(const FiberCrossSectionAnnotation& annotation)
{
    validateFiberCrossSectionAnnotation(annotation);
    nlohmann::json points = nlohmann::json::array();
    for (const auto& point : annotation.pointsXyz) {
        points.push_back(pointToJson(point));
    }
    return {
        {"version", FiberCrossSectionAnnotation::SchemaVersion},
        {"id", annotation.id},
        {"kind", fiberCrossSectionKindToString(annotation.kind)},
        {"points_xyz", std::move(points)},
        {"plane_origin_xyz", pointToJson(annotation.planeOriginXyz)},
        {"plane_normal_xyz", pointToJson(annotation.planeNormalXyz)},
        {"plane_up_xyz", pointToJson(annotation.planeUpXyz)},
        {"recorded_line_position_xyz", pointToJson(annotation.recordedLinePositionXyz)},
        {"recorded_arclength_base_voxels", annotation.recordedArclengthBaseVoxels},
        {"geometry_generation", annotation.geometryGeneration},
        {"detached", annotation.detached},
    };
}

std::vector<FiberCrossSectionAnnotation> fiberCrossSectionAnnotationsFromJson(const nlohmann::json& fiberRoot, const std::string& context)
{
    if (!fiberRoot.contains("cross_sections")) {
        return {};
    }
    const auto& values = fiberRoot.at("cross_sections");
    if (!values.is_array()) {
        throw std::runtime_error(context + " cross_sections must be an array");
    }
    std::vector<FiberCrossSectionAnnotation> annotations;
    annotations.reserve(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        annotations.push_back(fiberCrossSectionAnnotationFromJson(values.at(index), context + " cross_sections[" + std::to_string(index) + "]"));
    }
    validateFiberCrossSectionAnnotations(annotations, context + " cross_sections");
    return annotations;
}

nlohmann::json fiberCrossSectionAnnotationsToJson(const std::vector<FiberCrossSectionAnnotation>& annotations)
{
    validateFiberCrossSectionAnnotations(annotations);
    nlohmann::json result = nlohmann::json::array();
    for (const auto& annotation : annotations) {
        result.push_back(fiberCrossSectionAnnotationToJson(annotation));
    }
    return result;
}

void scaleFiberCrossSectionAnnotation(FiberCrossSectionAnnotation& annotation, double scale)
{
    if (!std::isfinite(scale) || scale <= 0.0) {
        throw std::invalid_argument("fiber cross-section scale must be finite and positive");
    }
    if (scale == 1.0) {
        return;
    }
    for (auto& point : annotation.pointsXyz) {
        point *= scale;
    }
    annotation.planeOriginXyz *= scale;
    annotation.recordedLinePositionXyz *= scale;
    annotation.recordedArclengthBaseVoxels *= scale;
    validateFiberCrossSectionAnnotation(annotation);
}

double fiberCrossSectionDistanceToPolyline(const std::vector<cv::Vec3d>& line, const cv::Vec3d& point)
{
    if (line.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    if (line.size() == 1) {
        return cv::norm(line.front() - point);
    }

    double distance = std::numeric_limits<double>::infinity();
    for (size_t index = 1; index < line.size(); ++index) {
        const cv::Vec3d start = line[index - 1];
        const cv::Vec3d delta = line[index] - start;
        const double lengthSquared = delta.dot(delta);
        const double t = lengthSquared > 0.0 ? std::clamp((point - start).dot(delta) / lengthSquared, 0.0, 1.0) : 0.0;
        distance = std::min(distance, cv::norm(start + t * delta - point));
    }
    return distance;
}

}  // namespace vc::fiber_tracer
