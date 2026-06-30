#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "CState.hpp"
#include "FiberSliceGeometry.hpp"
#include "LineAnnotationFiberClassification.hpp"
#include "LineAnnotationFiberNaming.hpp"
#include "LineAnnotationGeneratedViews.hpp"
#include "LineAnnotationShiftScroll.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"

#include <cmath>
#include <limits>
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

TEST_CASE("line annotation straight shift scroll moves cut origin along plane normal")
{
    const cv::Vec3f origin{1.0f, 2.0f, 3.0f};
    const cv::Vec3f normal{0.0f, 0.0f, 2.0f};
    const double linePosition = 40.0;

    const cv::Vec3f shifted =
        vc3d::line_annotation::shiftedPlaneOriginAlongNormal(origin, normal, 3, 4);

    CHECK(shifted[0] == doctest::Approx(1.0f));
    CHECK(shifted[1] == doctest::Approx(2.0f));
    CHECK(shifted[2] == doctest::Approx(15.0f));
    CHECK(linePosition == doctest::Approx(40.0));
    CHECK(cv::norm(normal - cv::Vec3f{0.0f, 0.0f, 2.0f}) == doctest::Approx(0.0f));
}

TEST_CASE("line annotation straight shift scroll clamps invalid step size but not line position")
{
    const cv::Vec3f origin{10.0f, 0.0f, 0.0f};
    const cv::Vec3f normal{1.0f, 0.0f, 0.0f};
    const double linePosition = 8.0;

    const cv::Vec3f shifted =
        vc3d::line_annotation::shiftedPlaneOriginAlongNormal(origin, normal, -2, 0);

    CHECK(shifted[0] == doctest::Approx(8.0f));
    CHECK(shifted[1] == doctest::Approx(0.0f));
    CHECK(shifted[2] == doctest::Approx(0.0f));
    CHECK(linePosition == doctest::Approx(8.0));
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

TEST_CASE("line annotation current cut manual rotations preserve orthonormal frame")
{
    using vc3d::line_annotation::GeneratedCutRotationAxis;
    constexpr float pi = 3.14159265358979323846f;

    cv::Matx33f rotation = cv::Matx33f::eye();
    rotation = vc3d::line_annotation::accumulatedGeneratedCutRotation(
        rotation,
        GeneratedCutRotationAxis::Horizontal,
        pi / 12.0f);
    rotation = vc3d::line_annotation::accumulatedGeneratedCutRotation(
        rotation,
        GeneratedCutRotationAxis::Vertical,
        -pi / 18.0f);
    rotation = vc3d::line_annotation::accumulatedGeneratedCutRotation(
        rotation,
        GeneratedCutRotationAxis::Horizontal,
        pi / 20.0f);

    const auto frame = vc3d::line_annotation::generatedCutFrameWithManualRotation(
        {1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        rotation);
    CHECK(vc3d::line_annotation::generatedCutFrameIsOrthonormal(frame));
}

TEST_CASE("line annotation current cut manual rotation reapplies at new line position")
{
    using vc3d::line_annotation::GeneratedCutRotationAxis;
    constexpr float pi = 3.14159265358979323846f;

    const cv::Matx33f rotation = vc3d::line_annotation::accumulatedGeneratedCutRotation(
        cv::Matx33f::eye(),
        GeneratedCutRotationAxis::Vertical,
        pi / 6.0f);

    const auto first = vc3d::line_annotation::generatedCutFrameWithManualRotation(
        {1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        rotation);
    const auto second = vc3d::line_annotation::generatedCutFrameWithManualRotation(
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        rotation);

    CHECK(vc3d::line_annotation::generatedCutFrameIsOrthonormal(first));
    CHECK(vc3d::line_annotation::generatedCutFrameIsOrthonormal(second));
    CHECK(std::abs(first.normal.dot({1.0f, 0.0f, 0.0f})) < 0.99f);
    CHECK(std::abs(second.normal.dot({0.0f, 1.0f, 0.0f})) < 0.99f);
}

TEST_CASE("line annotation reset navigation state restores initial generated view values")
{
    auto state = vc3d::line_annotation::resetGeneratedLineViewNavigationState(12.0, 15.0, 7.5);

    CHECK(state.currentLinePosition == doctest::Approx(12.0));
    CHECK(state.bottomCenterPosition == doctest::Approx(15.0));
    CHECK(state.bottomSliceLineStep == doctest::Approx(7.5));
    CHECK_FALSE(state.currentCutManualRotationActive);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const float expected = row == col ? 1.0f : 0.0f;
            CHECK(state.currentCutManualRotation(row, col) == doctest::Approx(expected));
        }
    }
}

TEST_CASE("line annotation control point navigation chooses nearest previous and next")
{
    const std::vector<double> positions{40.0, 12.0, 28.0, 20.0};

    const auto previous = vc3d::line_annotation::previousGeneratedControlPointLinePosition(
        25.0,
        positions);
    const auto next = vc3d::line_annotation::nextGeneratedControlPointLinePosition(
        25.0,
        positions);

    REQUIRE(previous.has_value());
    REQUIRE(next.has_value());
    CHECK(*previous == doctest::Approx(20.0));
    CHECK(*next == doctest::Approx(28.0));
}

TEST_CASE("line annotation control point navigation boundaries do not wrap")
{
    const std::vector<double> positions{12.0, 20.0, 40.0};

    CHECK_FALSE(vc3d::line_annotation::previousGeneratedControlPointLinePosition(
        12.0,
        positions).has_value());
    CHECK_FALSE(vc3d::line_annotation::nextGeneratedControlPointLinePosition(
        40.0,
        positions).has_value());
    CHECK_FALSE(vc3d::line_annotation::previousGeneratedControlPointLinePosition(
        std::numeric_limits<double>::quiet_NaN(),
        positions).has_value());
    CHECK_FALSE(vc3d::line_annotation::nextGeneratedControlPointLinePosition(
        std::numeric_limits<double>::quiet_NaN(),
        positions).has_value());
}

TEST_CASE("line annotation closest control point chooses nearest valid position")
{
    const std::vector<double> positions{12.0, 20.0, 40.0};

    const auto closest = vc3d::line_annotation::closestGeneratedControlPointLinePosition(
        26.0,
        positions);

    REQUIRE(closest.has_value());
    CHECK(*closest == doctest::Approx(20.0));
}

TEST_CASE("line annotation control point navigation ignores invalid positions")
{
    using vc3d::line_annotation::GeneratedOverlay;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<GeneratedOverlay::ControlPointMarker> controls{
        {{0.0f, 0.0f, 0.0f}, 30.0, false},
        {{0.0f, 0.0f, 0.0f}, nan, false},
        {{0.0f, 0.0f, 0.0f}, 10.0, true},
        {{0.0f, 0.0f, 0.0f}, std::numeric_limits<double>::infinity(), false},
    };

    const auto positions = vc3d::line_annotation::finiteGeneratedControlPointLinePositions(controls);
    REQUIRE(positions.size() == 2);
    CHECK(positions[0] == doctest::Approx(10.0));
    CHECK(positions[1] == doctest::Approx(30.0));

    const auto previous = vc3d::line_annotation::previousGeneratedControlPointLinePosition(
        20.0,
        positions);
    const auto next = vc3d::line_annotation::nextGeneratedControlPointLinePosition(
        20.0,
        positions);
    const auto closest = vc3d::line_annotation::closestGeneratedControlPointLinePosition(
        26.0,
        positions);

    REQUIRE(previous.has_value());
    REQUIRE(next.has_value());
    REQUIRE(closest.has_value());
    CHECK(*previous == doctest::Approx(10.0));
    CHECK(*next == doctest::Approx(30.0));
    CHECK(*closest == doctest::Approx(30.0));
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

TEST_CASE("line annotation max control distance uses previous flattened control")
{
    const std::vector<double> controlPositions{10.0, 100.0};

    CHECK(vc3d::line_annotation::generatedControlPointPlacementWithinPreviousDistance(
        250.0,
        controlPositions,
        0.0));
    CHECK(vc3d::line_annotation::generatedControlPointPlacementWithinPreviousDistance(
        70.0,
        controlPositions,
        80.0));
    CHECK_FALSE(vc3d::line_annotation::generatedControlPointPlacementWithinPreviousDistance(
        95.0,
        controlPositions,
        80.0));
    CHECK(vc3d::line_annotation::generatedControlPointPlacementWithinPreviousDistance(
        100.25,
        controlPositions,
        80.0));
    CHECK(vc3d::line_annotation::generatedControlPointPlacementWithinPreviousDistance(
        5.0,
        controlPositions,
        80.0));
    CHECK_FALSE(vc3d::line_annotation::generatedControlPointPlacementWithinPreviousDistance(
        5.0,
        std::vector<double>{100.0},
        80.0));
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

TEST_CASE("line annotation intersection display h side uses manual tags before scores")
{
    using vc3d::line_annotation::FiberHvClassification;
    using vc3d::line_annotation::firstFiberDisplaysAsH;

    FiberHvClassification first;
    first.horizontalScore = 0.25;
    first.verticalScore = 0.75;
    FiberHvClassification second;
    second.horizontalScore = 0.75;
    second.verticalScore = 0.25;

    CHECK_FALSE(firstFiberDisplaysAsH(first, "", second, ""));
    CHECK(firstFiberDisplaysAsH(first, "H", second, ""));
    CHECK(firstFiberDisplaysAsH(first, "", second, "V"));
    CHECK_FALSE(firstFiberDisplaysAsH(first, "V", second, ""));
    CHECK_FALSE(firstFiberDisplaysAsH(first, "H", second, "H"));

    second.horizontalScore = first.horizontalScore;
    second.verticalScore = first.verticalScore;
    CHECK(firstFiberDisplaysAsH(first, "", second, "", true));
    CHECK_FALSE(firstFiberDisplaysAsH(first, "", second, "", false));
}

TEST_CASE("line annotation generated strip overlay includes controls and current marker")
{
    vc3d::line_annotation::GeneratedViews views;
    views.linePoints = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {2.0f, 0.0f, 0.0f},
    };
    views.seedLineIndex = 1;
    views.controlPoints = {
        {{0.0f, 0.0f, 0.0f}, 0.0, false},
        {{2.0f, 0.0f, 0.0f}, 2.0, true},
    };

    const auto overlay =
        vc3d::line_annotation::makeGeneratedStripOverlay(views, 1.0, {0.0, 1.0, 2.0});
    CHECK(overlay.useSurfaceCenterLine);
    CHECK(overlay.currentLinePosition == doctest::Approx(1.0));
    CHECK(overlay.controlPoints.size() == 2);
    CHECK(overlay.markerLinePositions.size() == 3);
    CHECK(overlay.seedLineIndex == -1);
}

TEST_CASE("line annotation generated overlays include pred-snap connector endpoints")
{
    vc3d::line_annotation::GeneratedViews views;
    views.linePoints = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {2.0f, 0.0f, 0.0f},
    };
    views.controlPoints = {
        {{1.0f, 0.0f, 0.0f}, 1.0, false},
    };
    views.predSnapPoints = {
        {{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 1.0f}, 1.0, 0, true},
    };

    const auto overlay = vc3d::line_annotation::makeGeneratedStripOverlay(views, 1.0, {});

    REQUIRE(overlay.predSnapPoints.size() == 1);
    CHECK(overlay.predSnapPoints[0].controlIndex == 0);
    CHECK(overlay.predSnapPoints[0].controlPoint[0] == doctest::Approx(1.0f));
    CHECK(overlay.predSnapPoints[0].snapPoint[2] == doctest::Approx(1.0f));
}

TEST_CASE("line annotation nearest generated control point supports shift pred-snap updates")
{
    using vc3d::line_annotation::GeneratedOverlay;
    const std::vector<GeneratedOverlay::ControlPointMarker> controls{
        {{0.0f, 0.0f, 0.0f}, 0.0, false},
        {{5.0f, 0.0f, 0.0f}, 5.0, false},
    };

    const auto nearest = vc3d::line_annotation::nearestGeneratedControlPointIndex(
        controls,
        {4.5f, 0.0f, 0.0f});

    REQUIRE(nearest.has_value());
    CHECK(*nearest == 1);
}

TEST_CASE("line annotation generated strip static and dynamic overlays split ownership")
{
    vc3d::line_annotation::GeneratedViews views;
    views.linePoints = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {2.0f, 0.0f, 0.0f},
    };
    views.seedLineIndex = 1;
    views.controlPoints = {
        {{0.0f, 0.0f, 0.0f}, 0.0, false},
        {{2.0f, 0.0f, 0.0f}, 2.0, true},
    };

    const auto staticOverlay = vc3d::line_annotation::makeGeneratedStaticStripOverlay(views);
    CHECK(staticOverlay.useSurfaceCenterLine);
    CHECK(staticOverlay.linePoints.size() == 3);
    CHECK(staticOverlay.controlPoints.size() == 2);
    CHECK(staticOverlay.markerLinePositions.empty());
    CHECK_FALSE(std::isfinite(staticOverlay.currentLinePosition));

    const auto dynamicOverlay =
        vc3d::line_annotation::makeGeneratedDynamicStripOverlay(views, 1.0, {0.0, 2.0});
    CHECK(dynamicOverlay.useSurfaceCenterLine);
    CHECK(dynamicOverlay.linePoints.empty());
    CHECK(dynamicOverlay.controlPoints.empty());
    CHECK(dynamicOverlay.markerLinePositions.size() == 2);
    CHECK(dynamicOverlay.currentLinePosition == doctest::Approx(1.0));
}

TEST_CASE("line annotation span alignment metric uses control midpoint")
{
    using vc3d::line_annotation::GeneratedOverlay;
    const std::vector<GeneratedOverlay::ControlPointMarker> controls{
        {{0.0f, 0.0f, 0.0f}, 4.0, false},
        {{0.0f, 0.0f, 0.0f}, 10.0, false},
        {{0.0f, 0.0f, 0.0f}, 20.0, true},
    };

    auto metric = vc3d::line_annotation::makeGeneratedSpanAlignmentMetric(
        2,
        0,
        2,
        controls);
    metric.available = true;
    metric.maxErrorDegrees = 52.0;

    CHECK(metric.spanIndex == 2);
    CHECK(metric.firstControlIndex == 0);
    CHECK(metric.secondControlIndex == 2);
    CHECK(metric.firstControlLinePosition == doctest::Approx(4.0));
    CHECK(metric.secondControlLinePosition == doctest::Approx(20.0));
    CHECK(metric.maxErrorDegrees == doctest::Approx(52.0));
    const auto center =
        vc3d::line_annotation::generatedSpanAlignmentMetricCenterLinePosition(metric);
    REQUIRE(center.has_value());
    CHECK(*center == doctest::Approx(12.0));

    const auto invalid = vc3d::line_annotation::makeGeneratedSpanAlignmentMetric(
        0,
        0,
        99,
        controls);
    CHECK_FALSE(vc3d::line_annotation::generatedSpanAlignmentMetricCenterLinePosition(
        invalid).has_value());
}

TEST_CASE("line annotation generated line tail detection uses control span")
{
    using vc3d::line_annotation::GeneratedOverlay;
    using vc3d::line_annotation::generatedControlLinePositionRange;
    using vc3d::line_annotation::generatedLineSegmentIsTail;

    const std::vector<GeneratedOverlay::ControlPointMarker> controls{
        {{0.0f, 0.0f, 0.0f}, 4.0, false},
        {{0.0f, 0.0f, 0.0f}, 1.0, true},
    };
    const auto range = generatedControlLinePositionRange(controls);
    REQUIRE(range.has_value());
    CHECK(range->first == doctest::Approx(1.0));
    CHECK(range->second == doctest::Approx(4.0));
    CHECK(generatedLineSegmentIsTail(0.0, 1.0, range));
    CHECK_FALSE(generatedLineSegmentIsTail(1.0, 2.0, range));
    CHECK_FALSE(generatedLineSegmentIsTail(3.0, 4.0, range));
    CHECK(generatedLineSegmentIsTail(4.0, 5.0, range));
}

TEST_CASE("line annotation control point index returns sorted line position window")
{
    using vc3d::line_annotation::GeneratedOverlay;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<GeneratedOverlay::ControlPointMarker> controls{
        {{0.0f, 0.0f, 0.0f}, 12.0, false},
        {{0.0f, 0.0f, 0.0f}, nan, false},
        {{0.0f, 0.0f, 0.0f}, 5.0, true},
        {{0.0f, 0.0f, 0.0f}, 9.0, false},
        {{0.0f, 0.0f, 0.0f}, std::numeric_limits<double>::infinity(), false},
    };

    const auto index =
        vc3d::line_annotation::buildGeneratedControlPointLinePositionIndex(controls);
    REQUIRE(index.sortedControlIndices.size() == 3);
    CHECK(index.sortedControlIndices[0] == 2);
    CHECK(index.sortedControlIndices[1] == 3);
    CHECK(index.sortedControlIndices[2] == 0);

    const auto candidates =
        vc3d::line_annotation::generatedControlPointCandidateIndicesInLinePositionWindow(
            controls,
            index,
            10.0,
            2.0);
    REQUIRE(candidates.size() == 2);
    CHECK(candidates[0] == 3);
    CHECK(candidates[1] == 0);
}

TEST_CASE("line annotation line position radius uses local spacing and minimum")
{
    const std::vector<cv::Vec3f> points{
        {0.0f, 0.0f, 0.0f},
        {10.0f, 0.0f, 0.0f},
        {20.0f, 0.0f, 0.0f},
    };

    CHECK(vc3d::line_annotation::generatedLinePositionRadiusForVolumeThreshold(
              points,
              0.5,
              2.0f) == doctest::Approx(0.5));
    CHECK(vc3d::line_annotation::generatedLinePositionRadiusForVolumeThreshold(
              points,
              0.5,
              15.0f) == doctest::Approx(1.5));
}

TEST_CASE("line annotation generated cross slice filters controls by viewport threshold")
{
    vc3d::line_annotation::GeneratedViews views;
    views.linePoints = {
        {0.0f, 0.0f, 0.0f},
        {10.0f, 0.0f, 0.0f},
    };
    views.controlPoints = {
        {{0.0f, 0.0f, 0.0f}, 0.0, false},
        {{0.0f, 0.0f, 4.9f}, 0.5, false},
        {{0.0f, 0.0f, 5.1f}, 1.0, true},
    };

    const auto overlay = vc3d::line_annotation::makeGeneratedCrossSliceOverlay(
        views,
        0.5,
        true,
        5.0f,
        [](const cv::Vec3f& point) {
            return point[2];
        });
    CHECK(overlay.emphasizedPointMarker);
    CHECK(overlay.pointMarker[0] == doctest::Approx(5.0f));
    CHECK(overlay.controlPoints.size() == 2);
    CHECK(overlay.controlPoints[0].linePosition == doctest::Approx(0.0));
    CHECK(overlay.controlPoints[1].linePosition == doctest::Approx(0.5));
}

TEST_CASE("line annotation generated cross slice marker uses explicit focus point")
{
    vc3d::line_annotation::GeneratedViews views;
    views.linePoints = {
        {0.0f, 0.0f, 0.0f},
        {10.0f, 0.0f, 0.0f},
    };
    views.focusPoint = {2.0f, 3.0f, 4.0f};

    const auto focused = vc3d::line_annotation::makeGeneratedCrossSliceOverlay(
        views,
        0.5,
        true,
        std::nullopt,
        {});
    CHECK(focused.pointMarker[0] == doctest::Approx(2.0f));
    CHECK(focused.pointMarker[1] == doctest::Approx(3.0f));
    CHECK(focused.pointMarker[2] == doctest::Approx(4.0f));

    const auto plain = vc3d::line_annotation::makeGeneratedCrossSliceOverlay(
        views,
        0.5,
        false,
        std::nullopt,
        {});
    CHECK(plain.pointMarker[0] == doctest::Approx(5.0f));
}

TEST_CASE("line annotation generated cross slice filters indexed candidates then plane distance")
{
    vc3d::line_annotation::GeneratedViews views;
    views.linePoints = {
        {0.0f, 0.0f, 0.0f},
        {10.0f, 0.0f, 0.0f},
        {20.0f, 0.0f, 0.0f},
    };
    views.controlPoints = {
        {{0.0f, 0.0f, 0.0f}, 0.0, false},
        {{10.0f, 0.0f, 4.9f}, 1.0, false},
        {{10.0f, 0.0f, 5.1f}, 1.25, false},
        {{20.0f, 0.0f, 0.0f}, 2.0, true},
    };
    const auto index =
        vc3d::line_annotation::buildGeneratedControlPointLinePositionIndex(views.controlPoints);

    const auto overlay = vc3d::line_annotation::makeGeneratedCrossSliceOverlay(
        views,
        1.0,
        true,
        5.0f,
        [](const cv::Vec3f& point) {
            return point[2];
        },
        &index,
        0.5);

    REQUIRE(overlay.controlPoints.size() == 1);
    CHECK(overlay.controlPoints[0].linePosition == doctest::Approx(1.0));
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

TEST_CASE("fiber slice focused marker uses five percent viewport threshold")
{
    using namespace vc3d::fiber_slice;
    CHECK(focusedIntersectionMarkerThreshold(100.0) == doctest::Approx(5.0));
    CHECK(focusedIntersectionMarkerVisible(5.0, 100.0));
    CHECK(focusedIntersectionMarkerVisible(-5.0, 100.0));
    CHECK_FALSE(focusedIntersectionMarkerVisible(5.001, 100.0));
    CHECK_FALSE(focusedIntersectionMarkerVisible(std::numeric_limits<double>::infinity(), 100.0));
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
