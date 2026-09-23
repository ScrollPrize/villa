#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberCrossSection.hpp"
#include "vc/fiber_tracer/FiberJson.hpp"

#include <nlohmann/json.hpp>

#include <cmath>

namespace
{

using vc::fiber_tracer::FiberCrossSectionAnnotation;
using vc::fiber_tracer::FiberCrossSectionKind;

FiberCrossSectionAnnotation lineAnnotation()
{
    FiberCrossSectionAnnotation annotation;
    annotation.id = "01234567-89ab-4cde-8fab-0123456789ab";
    annotation.kind = FiberCrossSectionKind::Line;
    annotation.pointsXyz = {
        cv::Vec3d{8.0, 19.0, 30.0},
        cv::Vec3d{12.0, 19.0, 30.0},
    };
    annotation.planeOriginXyz = {10.0, 20.0, 30.0};
    annotation.planeNormalXyz = {0.0, 0.0, 1.0};
    annotation.planeUpXyz = {0.0, 1.0, 0.0};
    annotation.recordedLinePositionXyz = {10.0, 20.0, 30.0};
    annotation.recordedArclengthBaseVoxels = 42.5;
    annotation.geometryGeneration = 7;
    return annotation;
}

FiberCrossSectionAnnotation polygonAnnotation()
{
    FiberCrossSectionAnnotation annotation = lineAnnotation();
    annotation.id = "fedcba98-7654-4321-8abc-fedcba987654";
    annotation.kind = FiberCrossSectionKind::Polygon;
    annotation.pointsXyz = {
        cv::Vec3d{8.0, 18.0, 30.0},
        cv::Vec3d{12.0, 18.0, 30.0},
        cv::Vec3d{12.0, 22.0, 30.0},
        cv::Vec3d{8.0, 22.0, 30.0},
    };
    annotation.detached = true;
    return annotation;
}

nlohmann::json minimalVersion3Fiber()
{
    return {
        {"type", "vc3d_fiber"},
        {"version", 3},
        {"optimization_mode", "lasagna"},
        {"line_points", nlohmann::json::array()},
        {"control_points", nlohmann::json::array()},
    };
}

void checkPoint(const cv::Vec3d& actual, const cv::Vec3d& expected)
{
    for (int axis = 0; axis < 3; ++axis) {
        CHECK(actual[axis] == doctest::Approx(expected[axis]));
    }
}

}  // namespace

TEST_CASE("fiber cross-section annotations round trip through their strict schema")
{
    const auto line = lineAnnotation();
    const auto polygon = polygonAnnotation();
    const auto lineJson = vc::fiber_tracer::fiberCrossSectionAnnotationToJson(line);
    const auto polygonJson = vc::fiber_tracer::fiberCrossSectionAnnotationToJson(polygon);

    CHECK(lineJson.at("kind") == "line");
    CHECK(lineJson.at("points_xyz").size() == 2);
    CHECK(polygonJson.at("kind") == "polygon");
    CHECK(polygonJson.at("detached") == true);
    CHECK(polygonJson.at("points_xyz").size() == 4);

    const auto parsedLine = vc::fiber_tracer::fiberCrossSectionAnnotationFromJson(lineJson, "test line");
    const auto parsedPolygon = vc::fiber_tracer::fiberCrossSectionAnnotationFromJson(polygonJson, "test polygon");
    CHECK(parsedLine.id == line.id);
    CHECK(parsedLine.kind == FiberCrossSectionKind::Line);
    checkPoint(parsedLine.pointsXyz[1], line.pointsXyz[1]);
    CHECK(parsedLine.recordedArclengthBaseVoxels == doctest::Approx(42.5));
    CHECK(parsedLine.geometryGeneration == 7);
    CHECK(parsedPolygon.id == polygon.id);
    CHECK(parsedPolygon.kind == FiberCrossSectionKind::Polygon);
    CHECK(parsedPolygon.detached);
}

TEST_CASE("shared fiber JSON parsing exposes optional cross-section annotations")
{
    nlohmann::json root = minimalVersion3Fiber();
    const auto absent = vc::fiber_tracer::parseVc3dFiberJson(root, "fiber");
    CHECK(absent.crossSections.empty());

    root["cross_sections"] = vc::fiber_tracer::fiberCrossSectionAnnotationsToJson({lineAnnotation(), polygonAnnotation()});
    const auto parsed = vc::fiber_tracer::parseVc3dFiberJson(root, "fiber");
    REQUIRE(parsed.crossSections.size() == 2);
    CHECK(parsed.crossSections[0].id == lineAnnotation().id);
    CHECK(parsed.crossSections[1].id == polygonAnnotation().id);

    auto legacy = root;
    legacy["version"] = 1;
    legacy.erase("optimization_mode");
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::parseVc3dFiberJson(legacy, "fiber"), doctest::Contains("require vc3d_fiber version 3"), std::runtime_error);
}

TEST_CASE("cross-section coordinate scaling preserves its frame directions")
{
    auto annotation = lineAnnotation();
    vc::fiber_tracer::scaleFiberCrossSectionAnnotation(annotation, 0.5);
    checkPoint(annotation.pointsXyz[0], cv::Vec3d{4.0, 9.5, 15.0});
    checkPoint(annotation.planeOriginXyz, cv::Vec3d{5.0, 10.0, 15.0});
    checkPoint(annotation.recordedLinePositionXyz, cv::Vec3d{5.0, 10.0, 15.0});
    checkPoint(annotation.planeNormalXyz, cv::Vec3d{0.0, 0.0, 1.0});
    checkPoint(annotation.planeUpXyz, cv::Vec3d{0.0, 1.0, 0.0});
    CHECK(annotation.recordedArclengthBaseVoxels == doctest::Approx(21.25));
    CHECK_THROWS_AS(vc::fiber_tracer::scaleFiberCrossSectionAnnotation(annotation, 0.0), std::invalid_argument);
}

TEST_CASE("cross-section validation rejects malformed identity frame and geometry")
{
    auto invalid = lineAnnotation();
    invalid.id = "NOT-A-UUID";
    CHECK_THROWS_AS(vc::fiber_tracer::validateFiberCrossSectionAnnotation(invalid), std::runtime_error);

    invalid = lineAnnotation();
    invalid.planeUpXyz = invalid.planeNormalXyz;
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::validateFiberCrossSectionAnnotation(invalid), doctest::Contains("orthonormal"), std::runtime_error);

    invalid = lineAnnotation();
    invalid.pointsXyz[1][2] += 1.0;
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::validateFiberCrossSectionAnnotation(invalid), doctest::Contains("outside its recorded plane"), std::runtime_error);

    invalid = lineAnnotation();
    invalid.pointsXyz[1] = invalid.pointsXyz[0];
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::validateFiberCrossSectionAnnotation(invalid), doctest::Contains("distinct"), std::runtime_error);

    invalid = polygonAnnotation();
    invalid.pointsXyz = {
        cv::Vec3d{8.0, 18.0, 30.0},
        cv::Vec3d{12.0, 22.0, 30.0},
        cv::Vec3d{8.0, 22.0, 30.0},
        cv::Vec3d{12.0, 18.0, 30.0},
    };
    CHECK_THROWS_AS(vc::fiber_tracer::validateFiberCrossSectionAnnotation(invalid), std::runtime_error);

    auto json = vc::fiber_tracer::fiberCrossSectionAnnotationToJson(lineAnnotation());
    json["unknown"] = 1;
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::fiberCrossSectionAnnotationFromJson(json), doctest::Contains("unknown field"), std::runtime_error);
}

TEST_CASE("fiber cross-section IDs are unique within a fiber")
{
    const auto annotation = lineAnnotation();
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::validateFiberCrossSectionAnnotations({annotation, annotation}), doctest::Contains("duplicate annotation id"), std::runtime_error);

    nlohmann::json root = minimalVersion3Fiber();
    const auto item = vc::fiber_tracer::fiberCrossSectionAnnotationToJson(annotation);
    root["cross_sections"] = nlohmann::json::array({item, item});
    CHECK_THROWS_WITH_AS(vc::fiber_tracer::parseVc3dFiberJson(root, "fiber"), doctest::Contains("duplicate annotation id"), std::runtime_error);
}

TEST_CASE("split ownership distance uses polyline segments rather than vertices")
{
    const std::vector<cv::Vec3d> line{
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 0.0},
        {10.0, 10.0, 0.0},
    };
    CHECK(vc::fiber_tracer::fiberCrossSectionDistanceToPolyline(line, {5.0, 2.0, 0.0}) == doctest::Approx(2.0));
    CHECK(vc::fiber_tracer::fiberCrossSectionDistanceToPolyline(line, {13.0, 5.0, 0.0}) == doctest::Approx(3.0));
    CHECK(std::isinf(vc::fiber_tracer::fiberCrossSectionDistanceToPolyline({}, {0.0, 0.0, 0.0})));
}
