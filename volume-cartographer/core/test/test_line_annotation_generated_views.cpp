#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "CState.hpp"
#include "FiberSliceGeometry.hpp"
#include "LineAnnotationFiberClassification.hpp"
#include "LineAnnotationFiberNaming.hpp"
#include "LineAnnotationFiberSaveJob.hpp"
#include "LineAnnotationFiberSegments.hpp"
#include "vc/lasagna/LineSpline.hpp"
#include "LineAnnotationGeneratedViews.hpp"
#include "LineAnnotationShiftScroll.hpp"
#include "vc/fiber_tracer/FiberJson.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

class FiberModeNormalSampler final : public vc::lasagna::NormalSampler {
public:
    explicit FiberModeNormalSampler(cv::Vec3d normal = {0.0, 0.0, 1.0})
        : normal_(normal)
    {
    }

    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override
    {
        return {normal_, true, {}};
    }

private:
    cv::Vec3d normal_;
};

class FiberModePrediction final : public vc::fiber_tracer::FiberPredictionSource {
public:
    explicit FiberModePrediction(double invalidX =
                                     std::numeric_limits<double>::infinity())
        : invalidX_(invalidX)
    {
    }

    vc::fiber_tracer::FiberPredictionSample sample(
        const cv::Vec3d& point,
        const cv::Vec3d& referenceDirection) const override
    {
        vc::fiber_tracer::FiberPredictionSample sample;
        if (std::abs(point[0] - invalidX_) < 1.0e-6) {
            sample.options.push_back({});
            return sample;
        }
        const float sign = referenceDirection[0] < 0.0 ? -1.0f : 1.0f;
        sample.options.push_back({{sign, 0.0f, 0.0f}, 1.0f, true});
        return sample;
    }

private:
    double invalidX_;
};

class AlwaysInvalidFiberModePrediction final
    : public vc::fiber_tracer::FiberPredictionSource {
public:
    vc::fiber_tracer::FiberPredictionSample sample(
        const cv::Vec3d&,
        const cv::Vec3d&) const override
    {
        vc::fiber_tracer::FiberPredictionSample sample;
        sample.options.push_back({});
        return sample;
    }
};

class FirstStepInvalidFiberModePrediction final
    : public vc::fiber_tracer::FiberPredictionSource {
public:
    vc::fiber_tracer::FiberPredictionSample sample(
        const cv::Vec3d& point,
        const cv::Vec3d& referenceDirection) const override
    {
        vc::fiber_tracer::FiberPredictionSample sample;
        if (std::abs(point[0]) >= 4.0 - 1.0e-6) {
            sample.options.push_back({});
            return sample;
        }
        const float sign = referenceDirection[0] < 0.0 ? -1.0f : 1.0f;
        sample.options.push_back({{sign, 0.0f, 0.0f}, 1.0f, true});
        return sample;
    }
};

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

std::filesystem::path makeTempSaveDir(const std::string& testName)
{
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir = std::filesystem::temp_directory_path() /
        ("vc3d_fiber_save_" + testName + "_" + std::to_string(stamp));
    std::filesystem::create_directories(dir);
    return dir;
}

void writeText(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream out(path);
    out << text;
}

std::string readText(const std::filesystem::path& path)
{
    std::ifstream in(path);
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

std::vector<std::filesystem::path> recoveryFilesIn(const std::filesystem::path& dir)
{
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().string().find(".recovery.") != std::string::npos) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
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

TEST_CASE("line annotation max control distance uses nearest flattened control")
{
    const std::vector<double> controlPositions{10.0, 100.0};

    CHECK(vc3d::line_annotation::generatedControlPointPlacementWithinAnyDistance(
        250.0,
        controlPositions,
        0.0));
    CHECK(vc3d::line_annotation::generatedControlPointPlacementWithinAnyDistance(
        70.0,
        controlPositions,
        80.0));
    CHECK(vc3d::line_annotation::generatedControlPointPlacementWithinAnyDistance(
        95.0,
        controlPositions,
        80.0));
    CHECK_FALSE(vc3d::line_annotation::generatedControlPointPlacementWithinAnyDistance(
        55.0,
        controlPositions,
        40.0));
    CHECK(vc3d::line_annotation::generatedControlPointPlacementWithinAnyDistance(
        100.25,
        controlPositions,
        80.0));
    CHECK(vc3d::line_annotation::generatedControlPointPlacementWithinAnyDistance(
        5.0,
        controlPositions,
        80.0));
    CHECK_FALSE(vc3d::line_annotation::generatedControlPointPlacementWithinAnyDistance(
        5.0,
        std::vector<double>{100.0},
        80.0));

    CHECK(vc3d::line_annotation::generatedLinePositionWithinAnyControlDistance(
        95.0,
        controlPositions,
        80.0));
    CHECK_FALSE(vc3d::line_annotation::generatedLinePositionWithinAnyControlDistance(
        55.0,
        controlPositions,
        40.0));
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

TEST_CASE("line annotation stored single point fiber seed accepts seed-only geometry")
{
    const cv::Vec3d seed{1.0, 2.0, 3.0};

    const auto controlAndLine =
        vc3d::line_annotation::storedSinglePointFiberSeed({seed}, {seed});
    REQUIRE(controlAndLine.has_value());
    CHECK((*controlAndLine)[0] == doctest::Approx(1.0));
    CHECK((*controlAndLine)[1] == doctest::Approx(2.0));
    CHECK((*controlAndLine)[2] == doctest::Approx(3.0));

    const auto controlOnly =
        vc3d::line_annotation::storedSinglePointFiberSeed({seed}, {});
    REQUIRE(controlOnly.has_value());
    CHECK((*controlOnly)[2] == doctest::Approx(3.0));

    const auto lineOnly =
        vc3d::line_annotation::storedSinglePointFiberSeed({}, {seed});
    REQUIRE(lineOnly.has_value());
    CHECK((*lineOnly)[0] == doctest::Approx(1.0));
}

TEST_CASE("line annotation stored single point fiber seed rejects real or conflicting fibers")
{
    const cv::Vec3d seed{1.0, 2.0, 3.0};

    CHECK_FALSE(vc3d::line_annotation::storedSinglePointFiberSeed(
        {seed},
        {seed, {2.0, 2.0, 3.0}}).has_value());
    CHECK_FALSE(vc3d::line_annotation::storedSinglePointFiberSeed(
        {seed, {1.5, 2.0, 3.0}},
        {seed}).has_value());
    CHECK_FALSE(vc3d::line_annotation::storedSinglePointFiberSeed(
        {{9.0, 2.0, 3.0}},
        {seed}).has_value());
}

TEST_CASE("line annotation branch metadata json field set is a compatibility boundary")
{
    // This locks the documented branch metadata field names into the test suite.
    // Do not update this set unless the user explicitly asks for an on-disk
    // format change and the docs/migration path are updated with it.
    const std::vector<std::string> stableFields{
        "control_point_index",
        "branch_fiber_id",
        "branch_control_point_index",
        "control_point_direction",
        "branch_control_point_direction",
        "control_point_position",
        "branch_control_point_position",
        "branch_file",
    };

    CHECK(stableFields.size() == 8);
    CHECK(stableFields.front() == "control_point_index");
    CHECK(stableFields.back() == "branch_file");
}

TEST_CASE("line annotation single fiber save does not create recovery backup")
{
    const auto dir = makeTempSaveDir("single");
    const auto path = dir / "fiber_a.json";
    writeText(path, "{\"old\":true}\n");

    vc3d::line_annotation::FiberSavePayload payload;
    payload.fiberId = 1;
    payload.generation = 2;
    payload.path = path;
    payload.json = nlohmann::json{{"new", true}};

    const auto result = vc3d::line_annotation::runFiberSaveJob(10, {payload});

    CHECK(result.ok);
    CHECK(result.recoveryFiles.empty());
    CHECK(recoveryFilesIn(dir).empty());
    CHECK(readText(path).find("\"new\": true") != std::string::npos);
    std::filesystem::remove_all(dir);
}

TEST_CASE("line annotation successful multi fiber save deletes recovery backups")
{
    const auto dir = makeTempSaveDir("multi_success");
    const auto first = dir / "fiber_a.json";
    const auto second = dir / "fiber_b.json";
    writeText(first, "{\"old\":\"a\"}\n");
    writeText(second, "{\"old\":\"b\"}\n");

    std::vector<vc3d::line_annotation::FiberSavePayload> payloads{
        {1, 2, first, nlohmann::json{{"new", "a"}}},
        {2, 3, second, nlohmann::json{{"new", "b"}}},
    };

    const auto result = vc3d::line_annotation::runFiberSaveJob(11, std::move(payloads));

    CHECK(result.ok);
    CHECK(result.recoveryFiles.empty());
    CHECK(recoveryFilesIn(dir).empty());
    CHECK(readText(first).find("\"new\": \"a\"") != std::string::npos);
    CHECK(readText(second).find("\"new\": \"b\"") != std::string::npos);
    std::filesystem::remove_all(dir);
}

TEST_CASE("line annotation failed multi fiber save keeps recovery backups")
{
    const auto dir = makeTempSaveDir("multi_failure");
    const auto first = dir / "fiber_a.json";
    const auto second = dir / "fiber_b.json";
    writeText(first, "{\"old\":\"a\"}\n");
    writeText(second, "{\"old\":\"b\"}\n");

    setenv("VC3D_FIBER_SAVE_FAIL_AFTER_FIRST_REPLACE", "1", 1);
    std::vector<vc3d::line_annotation::FiberSavePayload> payloads{
        {1, 2, first, nlohmann::json{{"new", "a"}}},
        {2, 3, second, nlohmann::json{{"new", "b"}}},
    };

    const auto result = vc3d::line_annotation::runFiberSaveJob(12, std::move(payloads));
    unsetenv("VC3D_FIBER_SAVE_FAIL_AFTER_FIRST_REPLACE");

    CHECK_FALSE(result.ok);
    CHECK(result.error.find("Injected failure") != std::string::npos);
    REQUIRE(result.recoveryFiles.size() == 2);
    for (const auto& recovery : result.recoveryFiles) {
        CHECK(std::filesystem::exists(recovery));
        CHECK(recovery.filename().string().find(".recovery.") != std::string::npos);
    }
    CHECK(recoveryFilesIn(dir).size() == 2);
    std::filesystem::remove_all(dir);
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
    views.controlPoints[0].hasBranches = true;
    views.controlPoints[0].branchIds = {7};
    views.controlPoints[0].branchLinks = {{7, 3}};
    views.branchLinks = {
        {7,
         {0.0f, 0.0f, 0.0f},
         {1.0f, 0.0f, 0.0f},
         {0.0f, 0.0f, 1.0f},
         {0.0f, 0.0f, 1.0f},
         {0.0f, 0.0f, 1.0f},
         false},
    };

    const auto overlay =
        vc3d::line_annotation::makeGeneratedStripOverlay(views, 1.0, {0.0, 1.0, 2.0});
    CHECK(overlay.useSurfaceCenterLine);
    CHECK(overlay.currentLinePosition == doctest::Approx(1.0));
    CHECK(overlay.controlPoints.size() == 2);
    CHECK(overlay.branchLinks.empty());
    REQUIRE(overlay.controlPoints[0].branchLinks.size() == 1);
    REQUIRE(overlay.controlPoints[0].branchIds.size() == 1);
    CHECK(overlay.controlPoints[0].hasBranches);
    CHECK(overlay.controlPoints[0].branchIds[0] == 7);
    CHECK(overlay.controlPoints[0].branchLinks[0].fiberId == 7);
    CHECK(overlay.controlPoints[0].branchLinks[0].controlPointIndex == 3);
    CHECK(overlay.markerLinePositions.size() == 3);
    CHECK(overlay.seedLineIndex == -1);
}

TEST_CASE("branch overlay replacement retains supplied native span diagnostics")
{
    vc3d::line_annotation::GeneratedViews views;
    views.spanAlignmentMetrics.push_back({});
    views.fiberIntersections.push_back({});
    vc3d::line_annotation::GeneratedSpanAlignmentMetric metric;
    metric.kind = vc3d::line_annotation::GeneratedSpanAlignmentMetric::Kind::NativeMeetingError;
    metric.meetingErrorBaseVoxels = 2.5;

    vc3d::line_annotation::replaceGeneratedBranchOverlayData(
        views,
        {{{1.0f, 2.0f, 3.0f}, 4.0, false}},
        {{{1.0f, 0.0f, 0.0f}}},
        {},
        {metric});

    REQUIRE(views.controlPoints.size() == 1);
    CHECK(views.controlPoints.front().linePosition == doctest::Approx(4.0));
    REQUIRE(views.spanAlignmentMetrics.size() == 1);
    CHECK(views.spanAlignmentMetrics.front().kind ==
          vc3d::line_annotation::GeneratedSpanAlignmentMetric::Kind::NativeMeetingError);
    CHECK(views.spanAlignmentMetrics.front().meetingErrorBaseVoxels ==
          doctest::Approx(2.5));
    CHECK(views.fiberIntersections.empty());
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
    views.controlPoints[0].hasBranches = true;
    views.controlPoints[0].branchIds = {7};
    views.controlPoints[0].branchLinks = {{7, 3}};
    views.branchLinks = {
        {7,
         {0.0f, 0.0f, 0.0f},
         {1.0f, 0.0f, 0.0f},
         {0.0f, 0.0f, 1.0f},
         {0.0f, 0.0f, 1.0f},
         {0.0f, 0.0f, 1.0f},
         false},
    };

    const auto staticOverlay = vc3d::line_annotation::makeGeneratedStaticStripOverlay(views);
    CHECK(staticOverlay.useSurfaceCenterLine);
    CHECK(staticOverlay.linePoints.size() == 3);
    CHECK(staticOverlay.controlPoints.size() == 2);
    CHECK(staticOverlay.branchLinks.empty());
    REQUIRE(staticOverlay.controlPoints[0].branchLinks.size() == 1);
    REQUIRE(staticOverlay.controlPoints[0].branchIds.size() == 1);
    CHECK(staticOverlay.controlPoints[0].hasBranches);
    CHECK(staticOverlay.controlPoints[0].branchIds[0] == 7);
    CHECK(staticOverlay.controlPoints[0].branchLinks[0].fiberId == 7);
    CHECK(staticOverlay.controlPoints[0].branchLinks[0].controlPointIndex == 3);
    CHECK(staticOverlay.markerLinePositions.empty());
    CHECK_FALSE(std::isfinite(staticOverlay.currentLinePosition));

    const auto dynamicOverlay =
        vc3d::line_annotation::makeGeneratedDynamicStripOverlay(views, 1.0, {0.0, 2.0});
    CHECK(dynamicOverlay.useSurfaceCenterLine);
    CHECK(dynamicOverlay.linePoints.empty());
    CHECK(dynamicOverlay.controlPoints.empty());
    CHECK(dynamicOverlay.branchLinks.empty());
    CHECK(dynamicOverlay.markerLinePositions.size() == 2);
    CHECK(dynamicOverlay.currentLinePosition == doctest::Approx(1.0));
}

TEST_CASE("line annotation cross-slice overlay does not carry side-strip intersection markers")
{
    vc3d::line_annotation::GeneratedViews views;
    views.linePoints = {
        {0.0f, 0.0f, 0.0f},
        {10.0f, 0.0f, 0.0f},
    };
    views.fiberIntersections = {
        {{2.0f, 1.0f, 0.0f}, 42, 3, 12.0, 1.0},
    };

    const auto overlay = vc3d::line_annotation::makeGeneratedCrossSliceOverlay(
        views,
        0.0,
        false,
        std::nullopt,
        {});

    CHECK(overlay.fiberIntersections.empty());
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

TEST_CASE("fiber segment metadata round trips with its owning control point")
{
    vc3d::line_annotation::StoredControlPoint control{{1.0, 2.0, 3.0}};
    vc3d::line_annotation::FiberTraceSegmentMetadata metadata;
    metadata.normalManifestLocation = "s3://bucket/normals.lasagna.json";
    metadata.fiberManifestLocation = "s3://bucket/fibers.lasagna.json";
    metadata.traceToBaseScale = 4.0;
    metadata.config.traceToBaseScale = 4.0;
    metadata.interpMode = vc3d::line_annotation::SegmentInterpolationMode::Trace;
    metadata.meetingErrorBaseVoxels = 2.5;
    metadata.meetingErrorRatio = 1.25;
    metadata.meetingSource = "forward_moving_plane";
    metadata.metric = 2.5;
    metadata.message = "trace";
    control.segmentToNext = metadata;

    const auto json = vc3d::line_annotation::storedControlPointToJson(control);
    const auto parsed = vc3d::line_annotation::storedControlPointFromJson(json, 3);
    REQUIRE(parsed.segmentToNext.has_value());
    CHECK(parsed[0] == doctest::Approx(1.0));
    CHECK(parsed.segmentToNext->normalManifestLocation ==
          "s3://bucket/normals.lasagna.json");
    CHECK(parsed.segmentToNext->traceToBaseScale == doctest::Approx(4.0));
    REQUIRE(parsed.segmentToNext->meetingErrorBaseVoxels.has_value());
    REQUIRE(parsed.segmentToNext->meetingErrorRatio.has_value());
    CHECK(*parsed.segmentToNext->meetingErrorBaseVoxels == doctest::Approx(2.5));
    CHECK(*parsed.segmentToNext->meetingErrorRatio == doctest::Approx(1.25));
    CHECK(parsed.segmentToNext->meetingSource == "forward_moving_plane");
    CHECK(vc3d::line_annotation::isAcceptedNativeTrace(
        parsed.segmentToNext));

    auto obsoleteMetadataJson = json;
    obsoleteMetadataJson["segment_to_next"]["metadata_version"] = 2;
    CHECK_THROWS_AS(
        vc3d::line_annotation::storedControlPointFromJson(obsoleteMetadataJson, 3),
        std::runtime_error);
    CHECK_THROWS_AS(
        vc3d::line_annotation::storedControlPointFromJson(json, 2),
        std::runtime_error);

    metadata.outcome = vc3d::line_annotation::FiberTraceSegmentMetadata::Outcome::LasagnaFallback;
    metadata.interpMode = vc3d::line_annotation::SegmentInterpolationMode::Lasagna;
    metadata.metric.reset();
    metadata.failureCode = "no_trace_plane_intersection";
    metadata.failureDetail = "forward=max_step_factor reverse=no_valid_candidates";
    control.segmentToNext = metadata;
    const auto fallbackJson = vc3d::line_annotation::storedControlPointToJson(control);
    CHECK(fallbackJson["segment_to_next"]["meeting_error_base_voxels"].is_null());
    CHECK(fallbackJson["segment_to_next"]["meeting_error_ratio"].is_null());
    CHECK(fallbackJson["segment_to_next"]["meeting_source"] == "");

    const nlohmann::json sharedReaderRoot = {
        {"control_points",
         nlohmann::json::array({
             fallbackJson,
             nlohmann::json{{"position", {4.0, 5.0, 6.0}}},
         })}};
    CHECK_NOTHROW(vc::fiber_tracer::vc3dFiberPointArrayFromJson(
        sharedReaderRoot, "control_points", 3, "test fiber"));
    CHECK_THROWS_AS(vc::fiber_tracer::vc3dFiberPointArrayFromJson(
        sharedReaderRoot, "control_points", 2, "test fiber"),
        std::runtime_error);

    nlohmann::json strictRoot = sharedReaderRoot;
    strictRoot["type"] = "vc3d_fiber";
    strictRoot["version"] = 3;
    strictRoot["optimization_mode"] = "native_fiber_trace3d";
    strictRoot["line_points"] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    CHECK_NOTHROW(vc::fiber_tracer::parseVc3dFiberJson(strictRoot, "test fiber"));
    auto missingMode = strictRoot;
    missingMode.erase("optimization_mode");
    CHECK_THROWS_AS(
        vc::fiber_tracer::parseVc3dFiberJson(missingMode, "test fiber"),
        std::runtime_error);
    auto missingDescriptor = strictRoot;
    missingDescriptor["control_points"][0].erase("segment_to_next");
    CHECK_THROWS_AS(
        vc::fiber_tracer::parseVc3dFiberJson(missingDescriptor, "test fiber"),
        std::runtime_error);
    const auto regeneratedLasagna =
        vc::fiber_tracer::makeLasagnaSegmentMetadataJson(
            "trace",
            "s3://bucket/new-normals.lasagna.json",
            4.0,
            fallbackJson.at("segment_to_next").at("config"),
            3.5);
    CHECK(regeneratedLasagna.at("interp_goal") == "trace");
    CHECK(regeneratedLasagna.at("interp_mode") == "lasagna");
    CHECK(regeneratedLasagna.at("metric").get<double>() == doctest::Approx(3.5));
    CHECK(regeneratedLasagna.at("fiber_manifest") == "");
    CHECK(regeneratedLasagna.at("meeting_error_base_voxels").is_null());

    vc::fiber_tracer::FiberTraceSegmentResult rejectedResult;
    rejectedResult.meetingErrorBaseVoxels = 250.0;
    rejectedResult.meetingErrorRatio = 2.5;
    rejectedResult.meetingSource = "discarded_native_meeting";
    rejectedResult.reason = "meeting_error_threshold";
    const auto rejectedMetadata =
        vc3d::line_annotation::fiberTraceSegmentMetadataForResult(
            metadata.normalManifestLocation,
            metadata.fiberManifestLocation,
            metadata.traceToBaseScale,
            metadata.config,
            rejectedResult);
    CHECK_FALSE(rejectedMetadata.meetingErrorBaseVoxels.has_value());
    CHECK_FALSE(rejectedMetadata.meetingErrorRatio.has_value());
    CHECK(rejectedMetadata.meetingSource.empty());
    CHECK(rejectedMetadata.failureCode == "meeting_error_threshold");

    std::vector<vc3d::line_annotation::StoredControlPoint> invalid{parsed};
    CHECK_THROWS_AS(vc3d::line_annotation::validateStoredControlPoints(invalid),
                    std::runtime_error);
}

TEST_CASE("fiber optimization mode has stable persisted values")
{
    using vc3d::line_annotation::FiberOptimizationMode;
    CHECK(vc3d::line_annotation::kDefaultNewFiberOptimizationMode ==
          FiberOptimizationMode::NativeFiberTrace3d);
    CHECK(vc3d::line_annotation::fiberOptimizationModeToString(
              FiberOptimizationMode::Lasagna) == "lasagna");
    CHECK(vc3d::line_annotation::fiberOptimizationModeToString(
              FiberOptimizationMode::NativeFiberTrace3d) ==
          "native_fiber_trace3d");
    CHECK(vc3d::line_annotation::fiberOptimizationModeFromString("lasagna") ==
          FiberOptimizationMode::Lasagna);
    CHECK(vc3d::line_annotation::fiberOptimizationModeFromString(
              "native_fiber_trace3d") ==
          FiberOptimizationMode::NativeFiberTrace3d);
    CHECK_THROWS_AS(
        vc3d::line_annotation::fiberOptimizationModeFromString("unknown"),
        std::runtime_error);
}

TEST_CASE("native seed tracing requires native mode and configured inference")
{
    using vc3d::line_annotation::FiberOptimizationMode;
    using vc3d::line_annotation::shouldRunNativeSeedTrace;

    CHECK(shouldRunNativeSeedTrace(
        FiberOptimizationMode::NativeFiberTrace3d, true, 0));
    CHECK(shouldRunNativeSeedTrace(
        FiberOptimizationMode::NativeFiberTrace3d, false, 1));
    CHECK_FALSE(shouldRunNativeSeedTrace(
        FiberOptimizationMode::NativeFiberTrace3d, false, 0));
    CHECK_FALSE(shouldRunNativeSeedTrace(
        FiberOptimizationMode::NativeFiberTrace3d, false, 2));
    CHECK_FALSE(shouldRunNativeSeedTrace(FiberOptimizationMode::Lasagna, true, 1));
}

TEST_CASE("segment interpolation resolution applies short fallback only to global goals")
{
    using namespace vc3d::line_annotation;
    CHECK(resolveSegmentInterpolationMode(
              SegmentInterpolationGoal::Global,
              FiberOptimizationMode::NativeFiberTrace3d,
              99.999) == SegmentInterpolationMode::Cspline);
    CHECK(resolveSegmentInterpolationMode(
              SegmentInterpolationGoal::Global,
              FiberOptimizationMode::NativeFiberTrace3d,
              100.0) == SegmentInterpolationMode::Trace);
    CHECK(resolveSegmentInterpolationMode(
              SegmentInterpolationGoal::Trace,
              FiberOptimizationMode::Lasagna,
              1.0) == SegmentInterpolationMode::Trace);
    CHECK(resolveSegmentInterpolationMode(
              SegmentInterpolationGoal::Lasagna,
              FiberOptimizationMode::NativeFiberTrace3d,
              1.0) == SegmentInterpolationMode::Lasagna);
}

TEST_CASE("fiber mode falls back only the failed native span")
{
    FiberModeNormalSampler normals;
    FiberModePrediction predictions(32.0);
    vc3d::line_annotation::FiberModeOptimizationRequest request;
    request.controlPoints = {
        {2.0, {0.0, 0.0, 0.0}, true, 2},
        {6.0, {16.0, 0.0, 0.0}, false, 6},
        {10.0, {32.0, 0.0, 0.0}, false, 10},
    };
    request.controlPoints[0].segmentToNext.emplace();
    request.controlPoints[0].segmentToNext->interpGoal =
        vc3d::line_annotation::SegmentInterpolationGoal::Trace;
    request.controlPoints[1].segmentToNext.emplace();
    request.controlPoints[1].segmentToNext->interpGoal =
        vc3d::line_annotation::SegmentInterpolationGoal::Trace;
    for (int x = -8; x <= 40; x += 4) {
        request.linePointsBase.push_back(
            {static_cast<double>(x), 0.0, 0.0});
    }
    request.predictions = &predictions;
    request.baseNormalSampler = &normals;
    request.traceNormalSampler = &normals;
    request.normalManifestLocation = "normal.lasagna.json";
    request.fiberManifestLocation = "fiber.lasagna.json";
    request.extrapolationDistanceBaseVoxels = 8.0;
    request.retraceAll = true;
    request.traceConfig.stepVoxels = 4.0;
    request.traceConfig.coneAngleDegrees = 0.0;
    request.traceConfig.beamWidth = 1;
    request.traceConfig.maxStepFactor = 2.0;
    request.traceConfig.smoothnessWeight = 0.0;
    request.traceConfig.smoothnessNormalWeight = 0.0;
    request.traceConfig.smoothnessTangentWeight = 0.0;
    request.traceConfig.cumulativeSmoothnessTangentWeight = 0.0;
    request.lasagnaConfig.segmentsPerSide = 2;
    request.lasagnaConfig.segmentLength = 4.0;
    request.lasagnaConfig.maxIterations = 20;
    request.lasagnaConfig.printSolverProgress = false;

    const auto result =
        vc3d::line_annotation::optimizeFiberWithNativeFallback(
            std::move(request));

    REQUIRE(result.controlPoints.size() == 3);
    CHECK(result.nativeSegments == 1);
    CHECK(result.lasagnaFallbackSegments == 1);
    CHECK(result.controlPoints[0].segmentToNext.has_value());
    CHECK(vc3d::line_annotation::isAcceptedNativeTrace(
        result.controlPoints[0].segmentToNext));
    CHECK(result.controlPoints[0].segmentToNext->normalManifestLocation ==
          "normal.lasagna.json");
    CHECK(result.controlPoints[0].segmentToNext->fiberManifestLocation ==
          "fiber.lasagna.json");
    REQUIRE(result.controlPoints[1].segmentToNext.has_value());
    CHECK_FALSE(vc3d::line_annotation::isAcceptedNativeTrace(
        result.controlPoints[1].segmentToNext));
    CHECK(result.controlPoints[1].segmentToNext->failureCode ==
          "trace_exception");
    CHECK(result.controlPoints[1].segmentToNext->normalManifestLocation ==
          "normal.lasagna.json");
    CHECK(result.controlPoints[1].segmentToNext->fiberManifestLocation ==
          "fiber.lasagna.json");
    CHECK(result.nativeExtrapolations +
              result.lasagnaFallbackExtrapolations == 2);
    CHECK(result.lasagnaFallbackExtrapolations >= 1);
    const auto& points = result.optimization.line.points;
    REQUIRE(points.size() >= 3);
    size_t sharedIndex = 0;
    double sharedDistance = std::numeric_limits<double>::infinity();
    for (size_t index = 0; index < points.size(); ++index) {
        const double distance = cv::norm(
            points[index].position - cv::Vec3d{16.0, 0.0, 0.0});
        if (distance < sharedDistance) {
            sharedDistance = distance;
            sharedIndex = index;
        }
    }
    REQUIRE(sharedDistance < 1.0e-9);
    REQUIRE(sharedIndex + 1 < points.size());
    const cv::Vec3d fallbackDirection = vc3d::fiber_slice::normalizedOrZero(
        points[sharedIndex + 1].position - points[sharedIndex].position);
    CHECK(fallbackDirection.dot(cv::Vec3d{1.0, 0.0, 0.0}) ==
          doctest::Approx(1.0).epsilon(1.0e-10));
}

TEST_CASE("fiber mode records only manifest identities used by direct interpolation")
{
    FiberModeNormalSampler normals;
    const auto optimize = [&](vc3d::line_annotation::SegmentInterpolationGoal goal) {
        vc3d::line_annotation::FiberModeOptimizationRequest request;
        request.controlPoints = {
            {2.0, {0.0, 0.0, 0.0}, true, 2},
            {6.0, {16.0, 0.0, 0.0}, false, 6},
        };
        request.controlPoints.front().segmentToNext.emplace();
        request.controlPoints.front().segmentToNext->interpGoal = goal;
        request.controlPoints.front().segmentToNext->normalManifestLocation = "stale-normal";
        request.controlPoints.front().segmentToNext->fiberManifestLocation = "stale-fiber";
        for (int x = -8; x <= 24; x += 4)
            request.linePointsBase.push_back({static_cast<double>(x), 0.0, 0.0});
        request.baseNormalSampler = &normals;
        request.normalManifestLocation = "normal.lasagna.json";
        request.fiberManifestLocation = "fiber.lasagna.json";
        request.globalMode = vc3d::line_annotation::FiberOptimizationMode::Lasagna;
        request.lasagnaConfig.segmentsPerSide = 2;
        request.lasagnaConfig.segmentLength = 4.0;
        request.lasagnaConfig.maxIterations = 20;
        request.lasagnaConfig.printSolverProgress = false;
        return vc3d::line_annotation::optimizeFiberWithNativeFallback(
            std::move(request));
    };

    const auto lasagna = optimize(
        vc3d::line_annotation::SegmentInterpolationGoal::Lasagna);
    REQUIRE(lasagna.controlPoints.front().segmentToNext.has_value());
    CHECK(lasagna.controlPoints.front().segmentToNext->normalManifestLocation ==
          "normal.lasagna.json");
    CHECK(lasagna.controlPoints.front().segmentToNext->fiberManifestLocation.empty());

    const auto spline = optimize(
        vc3d::line_annotation::SegmentInterpolationGoal::Cspline);
    REQUIRE(spline.controlPoints.front().segmentToNext.has_value());
    CHECK(spline.controlPoints.front().segmentToNext->normalManifestLocation.empty());
    CHECK(spline.controlPoints.front().segmentToNext->fiberManifestLocation.empty());
}

TEST_CASE("fiber mode truncates extrapolation at an invalid prediction edge")
{
    FiberModeNormalSampler normals;
    FiberModePrediction predictions(8.0);
    vc3d::line_annotation::FiberModeOptimizationRequest request;
    request.controlPoints = {
        {2.0, {0.0, 0.0, 0.0}, true, 2},
    };
    for (int x = -8; x <= 8; x += 4) {
        request.linePointsBase.push_back(
            {static_cast<double>(x), 0.0, 0.0});
    }
    request.predictions = &predictions;
    request.baseNormalSampler = &normals;
    request.traceNormalSampler = &normals;
    request.extrapolationDistanceBaseVoxels = 8.0;
    request.traceConfig.stepVoxels = 4.0;
    request.traceConfig.coneAngleDegrees = 0.0;
    request.traceConfig.beamWidth = 1;
    request.traceConfig.maxStepFactor = 2.0;
    request.traceConfig.smoothnessWeight = 0.0;
    request.traceConfig.smoothnessNormalWeight = 0.0;
    request.traceConfig.smoothnessTangentWeight = 0.0;
    request.traceConfig.cumulativeSmoothnessTangentWeight = 0.0;
    request.lasagnaConfig.segmentsPerSide = 2;
    request.lasagnaConfig.segmentLength = 4.0;
    request.lasagnaConfig.maxIterations = 20;
    request.lasagnaConfig.printSolverProgress = false;

    const auto result =
        vc3d::line_annotation::optimizeFiberWithNativeFallback(
            std::move(request));

    REQUIRE(result.controlPoints.size() == 1);
    CHECK(result.nativeSegments == 0);
    CHECK(result.lasagnaFallbackSegments == 0);
    CHECK(result.nativeExtrapolations == 2);
    CHECK(result.lasagnaFallbackExtrapolations == 0);
    REQUIRE(result.optimization.line.points.size() >= 3);
    const auto& points = result.optimization.line.points;
    CHECK(points.front().position[0] == doctest::Approx(-8.0));
    CHECK(points.back().position[0] == doctest::Approx(4.0));
}

TEST_CASE("fiber mode reports no-progress extrapolation fallback reasons")
{
    FiberModeNormalSampler normals;
    FirstStepInvalidFiberModePrediction predictions;
    std::vector<vc3d::line_annotation::FiberExtrapolationFallbackDiagnostic>
        extrapolationFallbacks;
    vc3d::line_annotation::FiberModeOptimizationRequest request;
    request.controlPoints = {
        {2.0, {0.0, 0.0, 0.0}, true, 2},
    };
    for (int x = -8; x <= 8; x += 4) {
        request.linePointsBase.push_back(
            {static_cast<double>(x), 0.0, 0.0});
    }
    request.predictions = &predictions;
    request.baseNormalSampler = &normals;
    request.traceNormalSampler = &normals;
    request.extrapolationDistanceBaseVoxels = 8.0;
    request.extrapolationFallbackCallback =
        [&extrapolationFallbacks](const auto& diagnostic) {
            extrapolationFallbacks.push_back(diagnostic);
        };
    request.traceConfig.stepVoxels = 4.0;
    request.traceConfig.coneAngleDegrees = 0.0;
    request.traceConfig.beamWidth = 1;
    request.traceConfig.maxStepFactor = 100.0;
    request.traceConfig.smoothnessNormalWeight = 0.0;
    request.traceConfig.smoothnessTangentWeight = 0.0;
    request.traceConfig.cumulativeSmoothnessTangentWeight = 0.0;
    request.lasagnaConfig.segmentsPerSide = 2;
    request.lasagnaConfig.segmentLength = 4.0;
    request.lasagnaConfig.maxIterations = 20;
    request.lasagnaConfig.printSolverProgress = false;

    const auto result =
        vc3d::line_annotation::optimizeFiberWithNativeFallback(
            std::move(request));

    CHECK(result.nativeExtrapolations == 0);
    CHECK(result.lasagnaFallbackExtrapolations == 2);
    REQUIRE(extrapolationFallbacks.size() == 2);
    CHECK(extrapolationFallbacks[0].side ==
          vc3d::line_annotation::FiberExtrapolationFallbackDiagnostic::Side::Left);
    CHECK(extrapolationFallbacks[1].side ==
          vc3d::line_annotation::FiberExtrapolationFallbackDiagnostic::Side::Right);
    for (const auto& diagnostic : extrapolationFallbacks) {
        CHECK(diagnostic.reason == "no_valid_candidates");
        CHECK(diagnostic.tracePointCount == 1);
        CHECK_FALSE(diagnostic.fromException);
    }
}

TEST_CASE("fiber mode retries stored trace results from their goal")
{
    FiberModeNormalSampler baseNormals({1.0, 0.0, 0.0});
    FiberModeNormalSampler traceNormals;
    AlwaysInvalidFiberModePrediction predictions;
    std::vector<vc3d::line_annotation::FiberExtrapolationFallbackDiagnostic>
        extrapolationFallbacks;
    vc3d::line_annotation::FiberModeOptimizationRequest request;
    request.controlPoints = {
        {2.0, {0.0, 0.0, 0.0}, true, 2},
        {6.0, {16.0, 0.0, 0.0}, false, 6},
        {10.0, {32.0, 0.0, 0.0}, false, 10},
    };
    request.controlPoints[0].segmentToNext.emplace();
    request.controlPoints[0].segmentToNext->interpGoal =
        vc3d::line_annotation::SegmentInterpolationGoal::Trace;
    request.controlPoints[1].segmentToNext.emplace();
    request.controlPoints[1].segmentToNext->interpGoal =
        vc3d::line_annotation::SegmentInterpolationGoal::Trace;
    request.controlPoints[1].segmentToNext->interpMode =
        vc3d::line_annotation::SegmentInterpolationMode::Trace;
    request.linePointsBase = {
        {-8.0, 0.0, 0.0},
        {-4.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {4.0, 0.0, 0.0},
        {8.0, 0.0, 0.0},
        {12.0, 0.0, 0.0},
        {16.0, 0.0, 0.0},
        {20.0, 4.0, 0.0},
        {24.0, 5.0, 0.0},
        {28.0, 3.0, 0.0},
        {32.0, 0.0, 0.0},
        {36.0, 0.0, 0.0},
    };
    request.predictions = &predictions;
    request.baseNormalSampler = &baseNormals;
    request.traceNormalSampler = &traceNormals;
    request.extrapolationDistanceBaseVoxels = 8.0;
    request.extrapolationFallbackCallback =
        [&extrapolationFallbacks](const auto& diagnostic) {
            extrapolationFallbacks.push_back(diagnostic);
        };
    request.traceConfig.stepVoxels = 4.0;
    request.traceConfig.coneAngleDegrees = 0.0;
    request.traceConfig.beamWidth = 1;
    request.traceConfig.maxStepFactor = 2.0;
    request.lasagnaConfig.segmentsPerSide = 2;
    request.lasagnaConfig.segmentLength = 4.0;
    request.lasagnaConfig.maxIterations = 20;
    request.lasagnaConfig.printSolverProgress = false;

    const auto result =
        vc3d::line_annotation::optimizeFiberWithNativeFallback(
            std::move(request));

    REQUIRE(result.nativeSegments == 0);
    REQUIRE(result.lasagnaFallbackSegments == 2);
    REQUIRE(result.lasagnaFallbackExtrapolations == 2);
    REQUIRE(extrapolationFallbacks.size() == 2);
    for (size_t index = 0; index < 2; ++index) {
        REQUIRE(result.controlPoints[index].segmentToNext.has_value());
        CHECK(result.controlPoints[index].segmentToNext->interpGoal ==
              vc3d::line_annotation::SegmentInterpolationGoal::Trace);
        CHECK(result.controlPoints[index].segmentToNext->interpMode !=
              vc3d::line_annotation::SegmentInterpolationMode::Trace);
        CHECK_FALSE(result.controlPoints[index].segmentToNext->failureCode.empty());
    }
}

TEST_CASE("fiber segment goals survive CP edits and copy on split")
{
    using vc3d::line_annotation::LineControlPoint;
    vc3d::line_annotation::FiberTraceSegmentMetadata metadata;
    metadata.normalManifestLocation = "normal";
    metadata.fiberManifestLocation = "fiber";
    std::vector<LineControlPoint> controls{
        {0.0, {0.0, 0.0, 0.0}, false, 0},
        {20.0, {20.0, 0.0, 0.0}, false, 20},
        {10.0, {10.0, 0.0, 0.0}, false, 10},
    };
    controls[0].segmentToNext = metadata;
    controls[2].segmentToNext = metadata;

    vc3d::line_annotation::invalidateSegmentsAdjacentToControl(controls, 1);
    CHECK(controls[0].segmentToNext.has_value());
    CHECK(controls[2].segmentToNext.has_value());
    CHECK_FALSE(controls[1].segmentToNext.has_value());

    controls[0].segmentToNext = metadata;
    controls.push_back({5.0, {5.0, 0.0, 0.0}, false, 5});
    vc3d::line_annotation::invalidateSegmentSplitByInsertedControl(controls, 3);
    CHECK(controls[0].segmentToNext.has_value());
    REQUIRE(controls[3].segmentToNext.has_value());
    CHECK(controls[3].segmentToNext->interpGoal ==
          controls[0].segmentToNext->interpGoal);
}

TEST_CASE("line spline jointly interpolates ordered controls")
{
    vc::lasagna::LineSplineRequest request;
    request.controlPoints = {
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 0.0},
        {25.0, 5.0, 0.0},
    };
    request.sampleSpacing = 2.0;
    const auto result = vc::lasagna::interpolateLineControlPoints(request);
    REQUIRE(result.controlPointIndices.size() == 3);
    CHECK(result.points.front() == request.controlPoints.front());
    CHECK(result.points[static_cast<size_t>(result.controlPointIndices[1])] ==
          request.controlPoints[1]);
    CHECK(result.points.back() == request.controlPoints.back());
    for (const auto& point : result.points) {
        CHECK(std::isfinite(point[0]));
        CHECK(std::isfinite(point[1]));
        CHECK(std::isfinite(point[2]));
    }
}

TEST_CASE("two-point line spline is exactly straight and honors spacing")
{
    vc::lasagna::LineSplineRequest request;
    request.controlPoints = {{1.0, 2.0, 3.0}, {11.0, 2.0, 3.0}};
    request.sampleSpacing = 2.0;
    const auto result = vc::lasagna::interpolateLineControlPoints(request);
    REQUIRE(result.points.size() == 6);
    for (const auto& point : result.points) {
        CHECK(point[1] == doctest::Approx(2.0));
        CHECK(point[2] == doctest::Approx(3.0));
    }
}

TEST_CASE("line spline honors hard endpoint directions")
{
    vc::lasagna::LineSplineRequest request;
    request.controlPoints = {{0.0, 0.0, 0.0}, {10.0, 10.0, 0.0}};
    request.leftDirection = cv::Vec3d{1.0, 0.0, 0.0};
    request.rightDirection = cv::Vec3d{0.0, 1.0, 0.0};
    request.sampleSpacing = 0.25;
    const auto result = vc::lasagna::interpolateLineControlPoints(request);
    REQUIRE(result.points.size() > 3);
    const cv::Vec3d first = result.points[1] - result.points[0];
    const cv::Vec3d last = result.points.back() - result.points[result.points.size() - 2];
    CHECK(first.dot(cv::Vec3d{1.0, 0.0, 0.0}) / cv::norm(first) > 0.99);
    CHECK(last.dot(cv::Vec3d{0.0, 1.0, 0.0}) / cv::norm(last) > 0.99);
}
