#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "CState.hpp"
#include "FiberSliceGeometry.hpp"
#include "LineAnnotationFiberClassification.hpp"
#include "LineAnnotationFiberNaming.hpp"
#include "LineAnnotationShiftScroll.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace {

vc::lasagna::NormalSample normal()
{
    return {{0.0, 0.0, 1.0}, true, {}};
}

vc::lasagna::LineModel lineModel()
{
    vc::lasagna::LineModel line;
    line.points = {
        {{0.0, 0.0, 0.0}, normal(), true},
        {{10.0, 0.0, 0.0}, normal(), true},
        {{20.0, 0.0, 0.0}, normal(), true},
    };
    line.segmentSamples = {
        {{{0.0, {0.0, 0.0, 0.0}, normal()},
          {1.0, {10.0, 0.0, 0.0}, normal()}}},
        {{{0.0, {10.0, 0.0, 0.0}, normal()},
          {1.0, {20.0, 0.0, 0.0}, normal()}}},
    };
    return line;
}

} // namespace

TEST_CASE("line annotation generated runtime surfaces register and clean up")
{
    CState state(64 * 1024 * 1024);
    state.setSurface("line_annotation_slice_1", state.surface("xy plane"));

    const auto views = vc::lasagna::buildLineViewSurfaces(lineModel());
    std::vector<std::string> generatedNames{"line-surface", "line-side-slice"};

    state.setSurface("line-surface", views.lineSurface);
    state.setSurface("line-side-slice", views.lineSideSlice);
    for (size_t i = 0; i < views.lineZSlices.size(); ++i) {
        const std::string name = "line-z-slice-" + std::to_string(i);
        state.setSurface(name, views.lineZSlices[i]);
        generatedNames.push_back(name);
    }

    CHECK(state.surface("line_annotation_slice_1") != nullptr);
    for (const auto& name : generatedNames) {
        CHECK(state.surface(name) != nullptr);
    }

    state.setSurface("line_annotation_slice_1", nullptr);
    for (const auto& name : generatedNames) {
        state.setSurface(name, nullptr);
    }

    CHECK(state.surface("line_annotation_slice_1") == nullptr);
    for (const auto& name : generatedNames) {
        CHECK(state.surface(name) == nullptr);
    }
}

TEST_CASE("line annotation shift scroll uses viewer slice step size")
{
    CHECK(vc3d::line_annotation::shiftScrollLineStepSize(0) == 1);
    CHECK(vc3d::line_annotation::shiftedLinePosition(40.0, 2, 5, 101) == 50.0);
    CHECK(vc3d::line_annotation::shiftedLinePosition(40.0, -3, 4, 101) == 28.0);
    CHECK(vc3d::line_annotation::shiftedLinePosition(98.0, 2, 5, 101) == 100.0);
    CHECK(vc3d::line_annotation::shiftedLinePosition(2.0, -2, 5, 101) == 0.0);
}

TEST_CASE("line annotation bottom shift scroll preserves generated slice spacing")
{
    const double shiftedCenter = vc3d::line_annotation::shiftedLinePosition(50.0, 2, 3, 101);
    CHECK(shiftedCenter == 56.0);

    CHECK(vc3d::line_annotation::bottomCrossSliceLinePosition(shiftedCenter, 0, 7, 101) == 26.0);
    CHECK(vc3d::line_annotation::bottomCrossSliceLinePosition(shiftedCenter, 1, 7, 101) == 36.0);
    CHECK(vc3d::line_annotation::bottomCrossSliceLinePosition(shiftedCenter, 2, 7, 101) == 46.0);
    CHECK(vc3d::line_annotation::bottomCrossSliceLinePosition(shiftedCenter, 3, 7, 101) == 56.0);
    CHECK(vc3d::line_annotation::bottomCrossSliceLinePosition(shiftedCenter, 4, 7, 101) == 66.0);
    CHECK(vc3d::line_annotation::bottomCrossSliceLinePosition(shiftedCenter, 5, 7, 101) == 76.0);
    CHECK(vc3d::line_annotation::bottomCrossSliceLinePosition(shiftedCenter, 6, 7, 101) == 86.0);

    CHECK(vc3d::line_annotation::shiftedLinePosition(98.0, 2, 3, 101) == 100.0);
    CHECK(vc3d::line_annotation::bottomCrossSliceLinePosition(100.0, 6, 7, 101) == 100.0);
}

TEST_CASE("line annotation bottom cross slice spacing scales exponentially")
{
    CHECK(vc3d::line_annotation::adjustedBottomCrossSliceLineStep(10.0, 1, 101) ==
          doctest::Approx(15.0));
    CHECK(vc3d::line_annotation::adjustedBottomCrossSliceLineStep(10.0, -1, 101) ==
          doctest::Approx(10.0 / 1.5));
    CHECK(vc3d::line_annotation::adjustedBottomCrossSliceLineStep(10.0, 2, 101) ==
          doctest::Approx(22.5));

    const double spacing = 15.0;
    CHECK(vc3d::line_annotation::bottomCrossSliceLinePosition(50.0, 0, 7, 101, spacing) == 5.0);
    CHECK(vc3d::line_annotation::bottomCrossSliceLinePosition(50.0, 3, 7, 101, spacing) == 50.0);
    CHECK(vc3d::line_annotation::bottomCrossSliceLinePosition(50.0, 6, 7, 101, spacing) == 95.0);
}

TEST_CASE("line annotation fixed current slice snaps only within quarter line position")
{
    const std::vector<double> controlPositions{12.0, 20.0, 40.0};

    CHECK(vc3d::line_annotation::snappedControlPointLinePosition(19.75, controlPositions) ==
          doctest::Approx(20.0));
    CHECK(vc3d::line_annotation::snappedControlPointLinePosition(20.25, controlPositions) ==
          doctest::Approx(20.0));
    CHECK(vc3d::line_annotation::snappedControlPointLinePosition(20.2501, controlPositions) ==
          doctest::Approx(20.2501));
    CHECK(vc3d::line_annotation::snappedControlPointLinePosition(19.7499, controlPositions) ==
          doctest::Approx(19.7499));
}

TEST_CASE("line annotation fiber naming uses username timestamp and sequence")
{
    CHECK(vc3d::line_annotation::normalizedFiberUsername("") == "anon");
    CHECK(vc3d::line_annotation::normalizedFiberUsername("  alice  ") == "alice");
    CHECK(vc3d::line_annotation::normalizedFiberUsername("A User/Name") == "A_User_Name");
    CHECK(vc3d::line_annotation::fiberFileName("alice", "20260605T123456789", 42) ==
          "alice_20260605T123456789_000042.json");
}

TEST_CASE("line annotation fiber h/v classification scores endpoint z distance")
{
    using vc3d::line_annotation::FiberHvTag;
    using vc3d::line_annotation::classifyFiberHv;

    const auto horizontal = classifyFiberHv({
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 0.0},
    });
    CHECK(horizontal.valid);
    CHECK(horizontal.zDistance == doctest::Approx(0.0));
    CHECK(horizontal.fiberLength == doctest::Approx(10.0));
    CHECK(horizontal.horizontalScore == doctest::Approx(1.0));
    CHECK(horizontal.verticalScore == doctest::Approx(0.0));
    CHECK(horizontal.automaticTag == FiberHvTag::H);
    CHECK(horizontal.automaticCertainty == doctest::Approx(1.0));

    const auto vertical = classifyFiberHv({
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 5.0},
        {0.0, 0.0, 10.0},
    });
    CHECK(vertical.valid);
    CHECK(vertical.zDistance == doctest::Approx(10.0));
    CHECK(vertical.fiberLength == doctest::Approx(10.0));
    CHECK(vertical.horizontalScore == doctest::Approx(0.0));
    CHECK(vertical.verticalScore == doctest::Approx(1.0));
    CHECK(vertical.automaticTag == FiberHvTag::V);
    CHECK(vertical.automaticCertainty == doctest::Approx(1.0));

    const auto boundary = classifyFiberHv({
        {0.0, 0.0, 0.0},
        {std::sqrt(100.0 - 5.0 * 5.0), 0.0, 5.0},
    });
    CHECK(boundary.valid);
    CHECK(boundary.zDistance == doctest::Approx(5.0));
    CHECK(boundary.fiberLength == doctest::Approx(10.0));
    CHECK(boundary.verticalScore == doctest::Approx(0.5));
    CHECK(boundary.automaticTag == FiberHvTag::V);
    CHECK(boundary.automaticCertainty == doctest::Approx(0.0));

    const auto quarterVertical = classifyFiberHv({
        {0.0, 0.0, 0.0},
        {std::sqrt(800.0 * 800.0 - 200.0 * 200.0), 0.0, 200.0},
    });
    CHECK(quarterVertical.valid);
    CHECK(quarterVertical.zDistance == doctest::Approx(200.0));
    CHECK(quarterVertical.fiberLength == doctest::Approx(800.0));
    CHECK(quarterVertical.verticalScore == doctest::Approx(0.25));
    CHECK(quarterVertical.horizontalScore == doctest::Approx(0.75));
    CHECK(quarterVertical.automaticTag == FiberHvTag::H);
    CHECK(quarterVertical.automaticCertainty == doctest::Approx(0.5));

    const auto invalid = classifyFiberHv({{0.0, 0.0, 0.0}});
    CHECK_FALSE(invalid.valid);
    CHECK(invalid.automaticTag == FiberHvTag::Unknown);
}

TEST_CASE("fiber slice control span uses nearest control line indices")
{
    using namespace vc3d::fiber_slice;
    const std::vector<cv::Vec3d> linePoints{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {3.0, 0.0, 0.0},
        {4.0, 0.0, 0.0},
    };
    const std::vector<cv::Vec3d> controls{
        {3.1, 0.0, 0.0},
        {1.1, 0.0, 0.0},
    };

    const auto span = selectControlSpan(linePoints, controls);
    CHECK(span.valid);
    CHECK(span.firstLineIndex == 1);
    CHECK(span.lastLineIndex == 3);
    CHECK(span.samples.size() == 3);
    CHECK(span.centroid[0] == doctest::Approx(2.0));
}

TEST_CASE("fiber slice rejects insufficient controls or fit samples")
{
    using namespace vc3d::fiber_slice;
    CHECK_FALSE(selectControlSpan({{0.0, 0.0, 0.0},
                                   {1.0, 0.0, 0.0},
                                   {2.0, 0.0, 0.0}},
                                  {{0.0, 0.0, 0.0}}).valid);
    CHECK_FALSE(selectControlSpan({{0.0, 0.0, 0.0},
                                   {1.0, 0.0, 0.0}},
                                  {{0.0, 0.0, 0.0},
                                   {1.0, 0.0, 0.0}}).valid);
}

TEST_CASE("fiber slice plane fit recovers synthetic plane")
{
    using namespace vc3d::fiber_slice;
    std::vector<cv::Vec3d> linePoints;
    for (int i = -3; i <= 3; ++i) {
        const double x = static_cast<double>(i);
        const double y = static_cast<double>(i * i - 2);
        const double z = 2.0 * x + 3.0 * y + 5.0;
        linePoints.push_back({x, y, z});
    }
    const auto span = selectControlSpan(linePoints, {linePoints.front(), linePoints.back()});
    const auto fit = fitLeastSquaresPlane(span, linePoints);
    CHECK(fit.valid);

    const cv::Vec3d expected = normalizedOrZero({-2.0, -3.0, 1.0});
    CHECK(std::abs(fit.normal.dot(expected)) == doctest::Approx(1.0).epsilon(1e-6));
    for (const auto& point : linePoints) {
        CHECK(std::abs(signedDistanceToPlane(point, {fit.origin, fit.normal})) <= 1.0e-6);
    }
}

TEST_CASE("fiber slice distance scaling clamps at viewport thresholds")
{
    using namespace vc3d::fiber_slice;
    CHECK(distanceScaledSize(0.5, 100.0, 10.0, 2.0) == doctest::Approx(10.0));
    CHECK(distanceScaledSize(1.0, 100.0, 10.0, 2.0) == doctest::Approx(10.0));
    CHECK(distanceScaledSize(10.0, 100.0, 10.0, 2.0) == doctest::Approx(2.0));
    CHECK(distanceScaledSize(20.0, 100.0, 10.0, 2.0) == doctest::Approx(2.0));
    CHECK(distanceScaledSize(5.5, 100.0, 10.0, 2.0) == doctest::Approx(6.0));
}

TEST_CASE("fiber slice segment-plane intersection handles crossings")
{
    using namespace vc3d::fiber_slice;
    const Plane plane{{0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}};
    const auto crossing = segmentPlaneIntersection({0.0, 0.0, -1.0},
                                                   {2.0, 0.0, 1.0},
                                                   plane);
    REQUIRE(crossing.has_value());
    CHECK(crossing->point[0] == doctest::Approx(1.0));
    CHECK(crossing->point[1] == doctest::Approx(0.0));
    CHECK(crossing->point[2] == doctest::Approx(0.0));

    CHECK_FALSE(segmentPlaneIntersection({0.0, 0.0, 1.0},
                                         {2.0, 0.0, 1.0},
                                         plane).has_value());
    CHECK_FALSE(segmentPlaneIntersection({0.0, 0.0, 1.0},
                                         {2.0, 0.0, 2.0},
                                         plane).has_value());
    CHECK_FALSE(segmentPlaneIntersection({0.0, 0.0, 0.0},
                                         {0.0, 0.0, 0.0},
                                         plane).has_value());
}

TEST_CASE("fiber slice intersection opacity fades from 45 to 90 degrees")
{
    using namespace vc3d::fiber_slice;
    CHECK(intersectionOpacityForAngle(20.0) == doctest::Approx(1.0));
    CHECK(intersectionOpacityForAngle(45.0) == doctest::Approx(1.0));
    CHECK(intersectionOpacityForAngle(67.5) == doctest::Approx(0.5));
    CHECK(intersectionOpacityForAngle(90.0) == doctest::Approx(0.0));
    CHECK(intersectionOpacityForAngle(100.0) == doctest::Approx(0.0));

    const auto round = ellipseStyleForAngle(45.0, 3.0);
    const auto flat = ellipseStyleForAngle(89.0, 3.0);
    CHECK(flat.majorRadius > round.majorRadius);
    CHECK(flat.minorRadius < round.minorRadius);
}
