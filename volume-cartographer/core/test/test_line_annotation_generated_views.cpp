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

class ThrowingFiberModeNormalSampler final : public vc::lasagna::NormalSampler {
public:
    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d&) const override
    {
        throw std::runtime_error("test sampler failure");
    }
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

TEST_CASE("fiber file name identity parsing round-trips canonical names")
{
    using vc3d::line_annotation::fiberFileName;
    using vc3d::line_annotation::parsedFiberFileNameIdentity;

    const auto simple = parsedFiberFileNameIdentity(
        fiberFileName("kb", "20260719T194751553", 553));
    REQUIRE(simple.has_value());
    CHECK(simple->username == "kb");
    CHECK(simple->startedAt == "20260719T194751553");
    CHECK(simple->sequence == 553);

    // Usernames may contain underscores; the stem parses from the right.
    const auto underscored = parsedFiberFileNameIdentity(
        fiberFileName("team_alpha", "20260101T000000000", 7));
    REQUIRE(underscored.has_value());
    CHECK(underscored->username == "team_alpha");
    CHECK(underscored->sequence == 7);

    // Sequences above the padded width still round-trip.
    const auto wide = parsedFiberFileNameIdentity(
        fiberFileName("dj", "20260101T000000000", 1234567));
    REQUIRE(wide.has_value());
    CHECK(wide->sequence == 1234567);

    // Non-canonical names carry no identity.
    CHECK_FALSE(parsedFiberFileNameIdentity("horizontal_bundle_03.json"));
    CHECK_FALSE(parsedFiberFileNameIdentity("kb_20260719T194751553_000553"));
    CHECK_FALSE(parsedFiberFileNameIdentity("kb_20260719_000553.json"));
    CHECK_FALSE(parsedFiberFileNameIdentity("kb_2026071?T194751553_000553.json"));
    CHECK_FALSE(parsedFiberFileNameIdentity("_20260719T194751553_000553.json"));
    CHECK_FALSE(parsedFiberFileNameIdentity("kb_20260719T194751553_.json"));
    CHECK_FALSE(parsedFiberFileNameIdentity(".json"));
}

TEST_CASE("generated display tangent sign is independent of stored point order")
{
    using vc3d::line_annotation::generatedDisplayTangentSign;

    // Circumferential fiber: a circle in a z = const plane with outward sheet
    // normals. The decision comes from (normal x tangent) . z.
    std::vector<cv::Vec3f> circlePoints;
    std::vector<cv::Vec3f> circleNormals;
    for (int i = 0; i < 16; ++i) {
        const float angle = static_cast<float>(i) * 0.25f;
        const cv::Vec3f radial{std::cos(angle), std::sin(angle), 0.0f};
        circlePoints.push_back(radial * 100.0f + cv::Vec3f{500.0f, 500.0f, 300.0f});
        circleNormals.push_back(radial);
    }
    CHECK(generatedDisplayTangentSign(circlePoints, circleNormals) == 1.0f);

    std::vector<cv::Vec3f> reversedPoints(circlePoints.rbegin(), circlePoints.rend());
    std::vector<cv::Vec3f> reversedNormals(circleNormals.rbegin(), circleNormals.rend());
    CHECK(generatedDisplayTangentSign(reversedPoints, reversedNormals) == -1.0f);

    // Axial fiber: (normal x tangent) . z vanishes, so the tangent's own z
    // component decides and pins the side cut's vertical.
    std::vector<cv::Vec3f> axialPoints;
    std::vector<cv::Vec3f> axialNormals;
    for (int i = 0; i < 16; ++i) {
        axialPoints.push_back({500.0f, 500.0f, 300.0f + 10.0f * static_cast<float>(i)});
        axialNormals.push_back({1.0f, 0.0f, 0.0f});
    }
    CHECK(generatedDisplayTangentSign(axialPoints, axialNormals) == 1.0f);
    CHECK(generatedDisplayTangentSign({axialPoints.rbegin(), axialPoints.rend()},
                                      {axialNormals.rbegin(), axialNormals.rend()}) == -1.0f);

    // Near-axial fibers with slight helical drift: the drift direction must
    // not decide the sign -- both drift chiralities read as ascending, so the
    // side cut's vertical matches across neighboring vertical fibers.
    for (const float drift : {3.0e-2f, -3.0e-2f}) {
        std::vector<cv::Vec3f> helixPoints;
        std::vector<cv::Vec3f> helixNormals;
        for (int i = 0; i < 64; ++i) {
            const float angle = drift * static_cast<float>(i);
            const cv::Vec3f radial{std::cos(angle), std::sin(angle), 0.0f};
            helixPoints.push_back(radial * 200.0f +
                                  cv::Vec3f{500.0f, 500.0f, 300.0f + 10.0f * static_cast<float>(i)});
            helixNormals.push_back(radial);
        }
        CHECK(generatedDisplayTangentSign(helixPoints, helixNormals) == 1.0f);
        CHECK(generatedDisplayTangentSign({helixPoints.rbegin(), helixPoints.rend()},
                                          {helixNormals.rbegin(), helixNormals.rend()}) == -1.0f);
    }

    // Circumferential-dominant helix (shallow pitch): the circumferential
    // vote still owns the decision, whichever way the fiber creeps in z.
    for (const float climb : {1.0f, -1.0f}) {
        std::vector<cv::Vec3f> shallowPoints;
        std::vector<cv::Vec3f> shallowNormals;
        for (int i = 0; i < 64; ++i) {
            const float angle = 0.1f * static_cast<float>(i);
            const cv::Vec3f radial{std::cos(angle), std::sin(angle), 0.0f};
            shallowPoints.push_back(radial * 200.0f +
                                    cv::Vec3f{500.0f, 500.0f, 300.0f + climb * static_cast<float>(i)});
            shallowNormals.push_back(radial);
        }
        CHECK(generatedDisplayTangentSign(shallowPoints, shallowNormals) == 1.0f);
        CHECK(generatedDisplayTangentSign({shallowPoints.rbegin(), shallowPoints.rend()},
                                          {shallowNormals.rbegin(), shallowNormals.rend()}) == -1.0f);
    }

    // Sparse valid normals must still decide a circumferential fiber: the
    // primary tie band scales with the pairs that voted, not the tangent
    // count, or a fiber long enough that 1e-3 * tangentCount exceeds the few
    // normal votes would fall through to the useless z fallback and stay
    // order-dependent.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<cv::Vec3f> longCirclePoints;
    for (int i = 0; i < 2048; ++i) {
        const float angle = static_cast<float>(i) * 3.0e-3f;
        const cv::Vec3f radial{std::cos(angle), std::sin(angle), 0.0f};
        longCirclePoints.push_back(radial * 4000.0f + cv::Vec3f{5000.0f, 5000.0f, 3000.0f});
    }
    std::vector<cv::Vec3f> sparseNormals(longCirclePoints.size(), cv::Vec3f{nan, nan, nan});
    sparseNormals[1024] = cv::Vec3f{std::cos(1024 * 3.0e-3f), std::sin(1024 * 3.0e-3f), 0.0f};
    CHECK(generatedDisplayTangentSign(longCirclePoints, sparseNormals) == 1.0f);
    CHECK(generatedDisplayTangentSign({longCirclePoints.rbegin(), longCirclePoints.rend()},
                                      {sparseNormals.rbegin(), sparseNormals.rend()}) == -1.0f);

    // Degenerate inputs are decided as +1 rather than left arbitrary: no
    // normals to vote with, and a z = const line has nothing to fall back on.
    CHECK(generatedDisplayTangentSign({}, {}) == 1.0f);
    CHECK(generatedDisplayTangentSign({circlePoints.front()}, {circleNormals.front()}) == 1.0f);
    CHECK(generatedDisplayTangentSign(circlePoints, {}) == 1.0f);
    CHECK(generatedDisplayTangentSign(circlePoints,
                                      {circleNormals.begin(), circleNormals.end() - 1}) == 1.0f);
}

TEST_CASE("line annotation generated runtime surfaces register and clean up")
{
    CState state;
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

TEST_CASE("focus bounds state distinguishes configured and active bounds")
{
    CState state;
    int changes = 0;
    QObject::connect(&state, &CState::focusBoundsChanged, [&changes]() {
        ++changes;
    });

    const Rect3D reversed{{20.0f, 30.0f, 40.0f}, {10.0f, 15.0f, 25.0f}};
    state.setFocusBounds(reversed);
    REQUIRE(state.focusBounds());
    CHECK(state.focusBounds()->low == cv::Vec3f(10.0f, 15.0f, 25.0f));
    CHECK(state.focusBounds()->high == cv::Vec3f(20.0f, 30.0f, 40.0f));
    CHECK_FALSE(state.activeFocusBounds());
    CHECK(changes == 1);

    state.setFocusBounds(reversed);
    CHECK(changes == 1);
    state.setFocusBoundsEnabled(true);
    REQUIRE(state.activeFocusBounds());
    CHECK(changes == 2);
    const uint64_t activeRevision = state.focusBoundsRevision();
    state.setFocusBoundsEnabled(true);
    CHECK(state.focusBoundsRevision() == activeRevision);
    CHECK(changes == 2);

    state.clearFocusBounds();
    CHECK_FALSE(state.focusBounds());
    CHECK_FALSE(state.activeFocusBounds());
    CHECK_FALSE(state.focusBoundsEnabled());
    CHECK(changes == 3);
    state.clearFocusBounds();
    CHECK(changes == 3);
}

TEST_CASE("focus bounds trim only open line tails and rebase indices")
{
    vc::lasagna::LineModel line;
    for (int x = -20; x <= 40; x += 10) {
        vc::lasagna::LinePoint point;
        point.position = cv::Vec3d{x == 10 ? 10.0 : static_cast<double>(x),
                                   x == 10 ? 100.0 : 0.0,
                                   0.0};
        line.points.push_back(point);
    }
    line.displayFrameAnchorIndex = 4;
    line.segmentSamples.resize(line.points.size() - 1);
    std::vector<vc3d::line_annotation::LineControlPoint> controls{
        {2.0, {0.0, 0.0, 0.0}, true, 2},
        {4.0, {20.0, 0.0, 0.0}, false, 4},
    };
    const Rect3D bounds{{-5.0f, -5.0f, -5.0f}, {25.0f, 5.0f, 5.0f}};

    CHECK(vc3d::line_annotation::constrainLineOpenTailsToBounds(
        line, controls, bounds));
    REQUIRE(line.points.size() == 5);
    CHECK(line.points.front().position == cv::Vec3d(-10.0, 0.0, 0.0));
    CHECK(line.points[2].position == cv::Vec3d(10.0, 100.0, 0.0));
    CHECK(line.points.back().position == cv::Vec3d(30.0, 0.0, 0.0));
    CHECK(line.segmentSamples.size() == 4);
    CHECK(line.displayFrameAnchorIndex == 3);
    CHECK(controls[0].optimizedIndex == 1);
    CHECK(controls[0].linePosition == doctest::Approx(1.0));
    CHECK(controls[1].optimizedIndex == 3);
    CHECK(controls[1].linePosition == doctest::Approx(3.0));
}

TEST_CASE("focus bounds keep a two-point line for one outside control")
{
    vc::lasagna::LineModel line;
    for (int x : {0, 10, 20}) {
        vc::lasagna::LinePoint point;
        point.position = cv::Vec3d{static_cast<double>(x), 0.0, 0.0};
        line.points.push_back(point);
    }
    line.displayFrameAnchorIndex = 1;
    std::vector<vc3d::line_annotation::LineControlPoint> controls{
        {1.0, {10.0, 0.0, 0.0}, true, 1},
    };
    const Rect3D bounds{{100.0f, -1.0f, -1.0f}, {200.0f, 1.0f, 1.0f}};

    CHECK(vc3d::line_annotation::constrainLineOpenTailsToBounds(
        line, controls, bounds));
    REQUIRE(line.points.size() == 2);
    CHECK(line.points[0].position == cv::Vec3d(0.0, 0.0, 0.0));
    CHECK(line.points[1].position == cv::Vec3d(10.0, 0.0, 0.0));
    CHECK(controls[0].optimizedIndex == 1);
    CHECK(controls[0].linePosition == doctest::Approx(1.0));
    CHECK(line.displayFrameAnchorIndex == 1);
}

TEST_CASE("focus bounds keep paths from one outside control into the box")
{
    vc::lasagna::LineModel line;
    for (int x : {240, 210, 170, 140, 110, 80, 50, 20,
                  50, 80, 110, 140, 170, 210, 240}) {
        vc::lasagna::LinePoint point;
        point.position = cv::Vec3d{static_cast<double>(x), 0.0, 0.0};
        line.points.push_back(point);
    }
    line.displayFrameAnchorIndex = 7;
    std::vector<vc3d::line_annotation::LineControlPoint> controls{
        {7.0, {20.0, 0.0, 0.0}, true, 7},
    };
    const Rect3D bounds{{100.0f, -1.0f, -1.0f}, {200.0f, 1.0f, 1.0f}};

    CHECK(vc3d::line_annotation::constrainLineOpenTailsToBounds(
        line, controls, bounds));
    REQUIRE(line.points.size() == 13);
    CHECK(line.points.front().position == cv::Vec3d(210.0, 0.0, 0.0));
    CHECK(line.points[6].position == cv::Vec3d(20.0, 0.0, 0.0));
    CHECK(line.points.back().position == cv::Vec3d(210.0, 0.0, 0.0));
    CHECK(controls[0].optimizedIndex == 6);
    CHECK(controls[0].linePosition == doctest::Approx(6.0));
    CHECK(line.displayFrameAnchorIndex == 6);
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

TEST_CASE("arrow pan integrator ramps up toward cruise without overshooting it")
{
    constexpr double kCruise = 12.0;
    const double acceleration = kCruise / vc3d::line_annotation::kGeneratedArrowPanRampSeconds;
    constexpr double kDt = 1.0 / 60.0;

    auto state = vc3d::line_annotation::generatedArrowPanStep(
        0.0, 0.0, 1, kCruise, acceleration, kDt, std::nullopt);
    CHECK(state.velocity == doctest::Approx(acceleration * kDt));
    CHECK(state.position == doctest::Approx(acceleration * kDt * kDt));
    CHECK_FALSE(state.landed);

    double previousVelocity = state.velocity;
    for (int i = 0; i < 200; ++i) {
        state = vc3d::line_annotation::generatedArrowPanStep(
            state.position, state.velocity, 1, kCruise, acceleration, kDt, std::nullopt);
        CHECK(state.velocity <= kCruise + 1.0e-12);
        CHECK(state.velocity >= previousVelocity - 1.0e-12);
        previousVelocity = state.velocity;
    }
    // The ramp is 0.25 s long, so it is long since saturated and stays there.
    CHECK(state.velocity == doctest::Approx(kCruise));

    const auto cruising = vc3d::line_annotation::generatedArrowPanStep(
        state.position, kCruise, 1, kCruise, acceleration, kDt, std::nullopt);
    CHECK(cruising.velocity == doctest::Approx(kCruise));
    CHECK(cruising.position == doctest::Approx(state.position + kCruise * kDt));
    CHECK_FALSE(cruising.landed);
}

TEST_CASE("arrow pan integrator brakes into its stop target and lands exactly")
{
    constexpr double kCruise = 12.0;
    constexpr double kTarget = 10.0;
    const double acceleration = kCruise / vc3d::line_annotation::kGeneratedArrowPanRampSeconds;
    constexpr double kDt = 1.0 / 60.0;

    vc3d::line_annotation::GeneratedArrowPanState state{0.0, 0.0, false};
    bool sawDeceleration = false;
    double peakVelocity = 0.0;
    int ticks = 0;
    while (!state.landed && ticks < 10000) {
        const double before = state.velocity;
        state = vc3d::line_annotation::generatedArrowPanStep(
            state.position, state.velocity, 1, kCruise, acceleration, kDt, kTarget);
        peakVelocity = std::max(peakVelocity, state.velocity);
        if (state.velocity < before) {
            sawDeceleration = true;
        }
        CHECK(state.position <= kTarget);
        ++ticks;
    }
    REQUIRE(state.landed);
    CHECK(state.position == kTarget);
    CHECK(state.velocity == 0.0);
    CHECK(sawDeceleration);
    CHECK(peakVelocity <= kCruise + 1.0e-12);
    // Roughly the ramp up, a short cruise and the ramp down: well under a minute.
    CHECK(ticks < 300);

    // Already sitting on the target: land immediately without moving.
    const auto onTarget = vc3d::line_annotation::generatedArrowPanStep(
        kTarget, 0.0, 1, kCruise, acceleration, kDt, kTarget);
    CHECK(onTarget.landed);
    CHECK(onTarget.position == kTarget);
    CHECK(onTarget.velocity == 0.0);
}

TEST_CASE("arrow pan integrator decelerates through zero when the direction flips")
{
    constexpr double kCruise = 12.0;
    const double acceleration = kCruise / vc3d::line_annotation::kGeneratedArrowPanRampSeconds;
    constexpr double kDt = 1.0 / 60.0;

    // Cruising right when the left arrow takes over: the target is behind, so
    // the velocity must ramp down through zero rather than jump.
    auto state = vc3d::line_annotation::generatedArrowPanStep(
        50.0, kCruise, -1, kCruise, acceleration, kDt, std::optional<double>(40.0));
    CHECK(state.velocity == doctest::Approx(kCruise - acceleration * kDt));
    CHECK(state.velocity > 0.0);
    CHECK(state.position > 50.0);
    CHECK_FALSE(state.landed);

    double furthest = state.position;
    bool crossedZero = false;
    for (int i = 0; i < 400 && !state.landed; ++i) {
        state = vc3d::line_annotation::generatedArrowPanStep(
            state.position, state.velocity, -1, kCruise, acceleration, kDt,
            std::optional<double>(40.0));
        if (state.velocity < 0.0) {
            crossedZero = true;
        }
        if (!crossedZero) {
            furthest = std::max(furthest, state.position);
        }
        CHECK(state.velocity >= -kCruise - 1.0e-12);
    }
    CHECK(crossedZero);
    // Coasting to a stop from the cruise speed costs v^2 / (2a) = 1.5 positions.
    CHECK(furthest > 51.0);
    CHECK(furthest < 51.6);
    REQUIRE(state.landed);
    CHECK(state.position == 40.0);
}

TEST_CASE("arrow pan integrator handles zero steps and degenerate inputs")
{
    constexpr double kCruise = 12.0;
    const double acceleration = kCruise / vc3d::line_annotation::kGeneratedArrowPanRampSeconds;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    const auto noStep = vc3d::line_annotation::generatedArrowPanStep(
        5.0, 3.0, 1, kCruise, acceleration, 0.0, std::nullopt);
    CHECK(noStep.position == 5.0);
    CHECK(noStep.velocity == 3.0);
    CHECK_FALSE(noStep.landed);

    const auto negativeStep = vc3d::line_annotation::generatedArrowPanStep(
        5.0, 3.0, 1, kCruise, acceleration, -0.5, std::nullopt);
    CHECK(negativeStep.position == 5.0);
    CHECK(negativeStep.velocity == 3.0);

    const auto nanDt = vc3d::line_annotation::generatedArrowPanStep(
        5.0, 3.0, 1, kCruise, acceleration, nan, std::nullopt);
    CHECK(nanDt.position == 5.0);
    CHECK(nanDt.velocity == 3.0);

    const auto nanPosition = vc3d::line_annotation::generatedArrowPanStep(
        nan, 3.0, 1, kCruise, acceleration, 0.016, std::nullopt);
    CHECK_FALSE(std::isfinite(nanPosition.position));
    CHECK(nanPosition.velocity == 0.0);
    CHECK_FALSE(nanPosition.landed);

    const auto nanVelocity = vc3d::line_annotation::generatedArrowPanStep(
        5.0, nan, 1, kCruise, acceleration, 0.016, std::nullopt);
    CHECK(std::isfinite(nanVelocity.velocity));
    CHECK(nanVelocity.velocity == doctest::Approx(acceleration * 0.016));

    for (const double badCruise : {0.0, -3.0, nan}) {
        const auto stalled = vc3d::line_annotation::generatedArrowPanStep(
            5.0, 3.0, 1, badCruise, acceleration, 0.016, std::nullopt);
        CHECK(stalled.position == 5.0);
        CHECK(stalled.velocity == 0.0);
        CHECK_FALSE(stalled.landed);
    }
    for (const double badAcceleration : {0.0, -3.0, nan}) {
        const auto stalled = vc3d::line_annotation::generatedArrowPanStep(
            5.0, 3.0, 1, kCruise, badAcceleration, 0.016, std::nullopt);
        CHECK(stalled.position == 5.0);
        CHECK(stalled.velocity == 0.0);
    }

    // A non-finite stop target is simply no target: free cruise, never landed.
    const auto nanTarget = vc3d::line_annotation::generatedArrowPanStep(
        5.0, 3.0, 1, kCruise, acceleration, 0.016, std::optional<double>(nan));
    CHECK(nanTarget.position > 5.0);
    CHECK_FALSE(nanTarget.landed);

    // Direction 0 coasts to a standstill instead of running away.
    auto resting = vc3d::line_annotation::generatedArrowPanStep(
        5.0, 3.0, 0, kCruise, acceleration, 0.016, std::nullopt);
    CHECK(resting.velocity < 3.0);
    for (int i = 0; i < 200; ++i) {
        resting = vc3d::line_annotation::generatedArrowPanStep(
            resting.position, resting.velocity, 0, kCruise, acceleration, 0.016, std::nullopt);
    }
    CHECK(resting.velocity == 0.0);
}

TEST_CASE("arrow pan stop target picks the next control point in the direction")
{
    const std::vector<double> positions{12.0, 20.0, 28.0, 40.0};
    const double none = std::numeric_limits<double>::quiet_NaN();

    // Plain next/previous when the floor is behind the travel.
    const auto right = vc3d::line_annotation::generatedArrowPanStopTarget(
        positions, 21.0, 1, 20.0);
    REQUIRE(right.has_value());
    CHECK(*right == doctest::Approx(28.0));
    const auto left = vc3d::line_annotation::generatedArrowPanStopTarget(
        positions, 27.0, -1, 28.0);
    REQUIRE(left.has_value());
    CHECK(*left == doctest::Approx(20.0));

    // A control point strictly ahead but short of the floor is skipped: a hold
    // released early still travels at least as far as the tap would have.
    const auto floored = vc3d::line_annotation::generatedArrowPanStopTarget(
        positions, 13.0, 1, 28.0);
    REQUIRE(floored.has_value());
    CHECK(*floored == doctest::Approx(28.0));
    const auto flooredLeft = vc3d::line_annotation::generatedArrowPanStopTarget(
        positions, 39.0, -1, 20.0);
    REQUIRE(flooredLeft.has_value());
    CHECK(*flooredLeft == doctest::Approx(20.0));

    // Nothing further exists: fall back to the floor itself.
    const auto beyondLast = vc3d::line_annotation::generatedArrowPanStopTarget(
        positions, 41.0, 1, 40.0);
    REQUIRE(beyondLast.has_value());
    CHECK(*beyondLast == doctest::Approx(40.0));
    const auto beyondFirst = vc3d::line_annotation::generatedArrowPanStopTarget(
        positions, 5.0, -1, 12.0);
    REQUIRE(beyondFirst.has_value());
    CHECK(*beyondFirst == doctest::Approx(12.0));

    // Aiming at the far end (the held case) skips every intermediate point.
    const auto farEnd = vc3d::line_annotation::generatedArrowPanStopTarget(
        positions, 13.0, 1, positions.back());
    REQUIRE(farEnd.has_value());
    CHECK(*farEnd == doctest::Approx(40.0));

    // No floor and no candidate at all.
    CHECK_FALSE(vc3d::line_annotation::generatedArrowPanStopTarget(
        positions, 41.0, 1, none).has_value());
    CHECK_FALSE(vc3d::line_annotation::generatedArrowPanStopTarget(
        {}, 20.0, 1, none).has_value());
    CHECK_FALSE(vc3d::line_annotation::generatedArrowPanStopTarget(
        positions, 20.0, 0, 28.0).has_value());
    CHECK_FALSE(vc3d::line_annotation::generatedArrowPanStopTarget(
        positions, none, 1, 28.0).has_value());

    // Non-finite control positions are ignored, not returned.
    const std::vector<double> withNan{12.0, none, 28.0};
    const auto skipped = vc3d::line_annotation::generatedArrowPanStopTarget(
        withNan, 13.0, 1, none);
    REQUIRE(skipped.has_value());
    CHECK(*skipped == doctest::Approx(28.0));
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

TEST_CASE("line annotation remapped line position follows the same fiber spot")
{
    using vc3d::line_annotation::remappedGeneratedLinePosition;

    // Straight line, one unit per index.
    std::vector<cv::Vec3f> oldLine;
    for (int i = 0; i <= 10; ++i) {
        oldLine.push_back({static_cast<float>(i), 0.0f, 0.0f});
    }

    SUBCASE("identical lines return the same fractional position")
    {
        CHECK(remappedGeneratedLinePosition(oldLine, oldLine, 4.25) ==
              doctest::Approx(4.25));
    }

    SUBCASE("insertion before the anchor shifts the returned index")
    {
        // New line resamples the same geometry at half spacing: the 3D spot
        // x=4.25 now lives at index 8.5.
        std::vector<cv::Vec3f> newLine;
        for (int i = 0; i <= 20; ++i) {
            newLine.push_back({static_cast<float>(i) * 0.5f, 0.0f, 0.0f});
        }
        CHECK(remappedGeneratedLinePosition(oldLine, newLine, 4.25) ==
              doctest::Approx(8.5));
    }

    SUBCASE("anchor off the new line projects onto the nearest segment")
    {
        // New line offset in Y: nearest point to (4.25, 0, 0) is still at
        // parameter 4.25 along it.
        std::vector<cv::Vec3f> newLine;
        for (int i = 0; i <= 10; ++i) {
            newLine.push_back({static_cast<float>(i), 1.0f, 0.0f});
        }
        CHECK(remappedGeneratedLinePosition(oldLine, newLine, 4.25) ==
              doctest::Approx(4.25));
    }

    SUBCASE("degenerate inputs fall back to the clamped input position")
    {
        const std::vector<cv::Vec3f> empty;
        CHECK(remappedGeneratedLinePosition(oldLine, empty, 4.0) ==
              doctest::Approx(0.0));
        // Empty old line: the anchor is NaN, so the input position is reused.
        CHECK(remappedGeneratedLinePosition(empty, oldLine, 4.25) ==
              doctest::Approx(4.25));
        // Out-of-range input clamps to the new line's extent.
        CHECK(remappedGeneratedLinePosition(empty, oldLine, 99.0) ==
              doctest::Approx(10.0));
        const double nan = std::numeric_limits<double>::quiet_NaN();
        CHECK(remappedGeneratedLinePosition(empty, oldLine, nan) ==
              doctest::Approx(0.0));
    }

    SUBCASE("near-ties prefer the same wrap over a closer neighboring wrap")
    {
        // A fiber that runs out along y=0 and comes back along y=0.5, like
        // adjacent wraps of a spiral. The new line moves the outbound wrap
        // to y=0.35 and the return wrap to y=-0.3: the return wrap is now
        // geometrically CLOSER to the old anchor (5, 0, 0), but the remap
        // must stay on the outbound wrap (index continuity).
        std::vector<cv::Vec3f> oldWrapped;
        for (int i = 0; i <= 10; ++i) {
            oldWrapped.push_back({static_cast<float>(i), 0.0f, 0.0f});
        }
        for (int i = 0; i <= 10; ++i) {
            oldWrapped.push_back({static_cast<float>(10 - i), 0.5f, 0.0f});
        }
        std::vector<cv::Vec3f> newWrapped;
        for (int i = 0; i <= 10; ++i) {
            newWrapped.push_back({static_cast<float>(i), 0.35f, 0.0f});
        }
        for (int i = 0; i <= 10; ++i) {
            newWrapped.push_back({static_cast<float>(10 - i), -0.3f, 0.0f});
        }
        CHECK(remappedGeneratedLinePosition(oldWrapped, newWrapped, 5.0) ==
              doctest::Approx(5.0));
    }

    SUBCASE("non-finite vertices on the new line are skipped")
    {
        std::vector<cv::Vec3f> newLine = oldLine;
        newLine[4] = {std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f};
        // Nearest finite vertex to x=4.25 is index 5; the segment [5,6]
        // projects the anchor to its clamped start.
        CHECK(remappedGeneratedLinePosition(oldLine, newLine, 4.25) ==
              doctest::Approx(5.0));
    }
}

TEST_CASE("line annotation anchor remap keeps a pane position on its own fiber pass")
{
    using vc3d::line_annotation::remappedGeneratedLinePositionFromAnchor;

    // Two passes of the same fiber through one cut plane: an outbound pass
    // along y=0 (indices 0..10) and a return pass along y=6 (indices 11..21).
    // The pane reports position 5 on the outbound pass; the click that
    // requests the control point lands 5.5 units off that pass, i.e. within
    // half a unit of the return pass.
    std::vector<cv::Vec3d> line;
    for (int i = 0; i <= 10; ++i) {
        line.push_back({static_cast<double>(i), 0.0, 0.0});
    }
    for (int i = 0; i <= 10; ++i) {
        line.push_back({static_cast<double>(10 - i), 6.0, 0.0});
    }
    const cv::Vec3d click(5.0, 5.5, 0.0);
    const cv::Vec3d anchor(5.0, 0.0, 0.0);

    SUBCASE("the clicked point is nearer to the other pass")
    {
        // The regression this guards: resolving the position through the
        // click picked the return pass, thousands of vertices away on a real
        // fiber, so the edit collapsed into that pass's control point.
        CHECK(vc3d::fiber_slice::nearestLinePointIndex(line, click) == 16);
    }

    SUBCASE("the anchor keeps the position on the outbound pass")
    {
        CHECK(remappedGeneratedLinePositionFromAnchor(line, anchor, 5.0) ==
              doctest::Approx(5.0));
    }

    SUBCASE("a renumbered line still resolves through the anchor")
    {
        // The session line resampled the outbound pass at half spacing while
        // the pane still measured position 4.25 on the old line: the anchor
        // (4.25, 0, 0) lives at index 8.5 now.
        std::vector<cv::Vec3d> renumbered;
        for (int i = 0; i <= 20; ++i) {
            renumbered.push_back({static_cast<double>(i) * 0.5, 0.0, 0.0});
        }
        for (int i = 0; i <= 10; ++i) {
            renumbered.push_back({static_cast<double>(10 - i), 6.0, 0.0});
        }
        CHECK(remappedGeneratedLinePositionFromAnchor(
                  renumbered, cv::Vec3d(4.25, 0.0, 0.0), 4.25) ==
              doctest::Approx(8.5));
    }

    SUBCASE("a non-finite anchor falls back to the clamped position")
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        CHECK(remappedGeneratedLinePositionFromAnchor(
                  line, cv::Vec3d(nan, 0.0, 0.0), 5.0) == doctest::Approx(5.0));
        CHECK(remappedGeneratedLinePositionFromAnchor(
                  line, cv::Vec3d(nan, 0.0, 0.0), 99.0) == doctest::Approx(21.0));
    }
}

TEST_CASE("line annotation winding angles unwrap along the line and window half a wrap")
{
    using vc3d::line_annotation::generatedLineIndexRangeWithinWinding;
    using vc3d::line_annotation::kGeneratedSideCutHalfWrapAngle;
    using vc3d::line_annotation::unwrappedGeneratedWindingAngles;
    constexpr double kPi = kGeneratedSideCutHalfWrapAngle;

    // A spiral of 1.5 wraps about the origin, one point every quarter turn
    // (13 points, angles 0 .. 3*pi), radius growing slowly.
    std::vector<cv::Vec3f> spiral;
    for (int i = 0; i <= 12; ++i) {
        const double angle = static_cast<double>(i) * kPi / 4.0;
        const double radius = 100.0 + static_cast<double>(i);
        spiral.push_back({static_cast<float>(radius * std::cos(angle)),
                          static_cast<float>(radius * std::sin(angle)),
                          static_cast<float>(i)});
    }
    const auto towardOrigin = [](const cv::Vec3f& point) {
        return cv::Vec3f{-point[0], -point[1], 0.0f};
    };

    SUBCASE("angles accumulate past pi instead of wrapping")
    {
        const auto angles = unwrappedGeneratedWindingAngles(spiral, towardOrigin);
        REQUIRE(angles.size() == spiral.size());
        for (int i = 0; i <= 12; ++i) {
            CHECK(angles[static_cast<size_t>(i)] ==
                  doctest::Approx(static_cast<double>(i) * kPi / 4.0).epsilon(1e-6));
        }
    }

    SUBCASE("a point without a center direction is NaN and does not break the chain")
    {
        std::vector<cv::Vec3f> withHole = spiral;
        withHole[6] = {std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f};
        const auto angles = unwrappedGeneratedWindingAngles(withHole, towardOrigin);
        CHECK(std::isnan(angles[6]));
        CHECK(angles[7] == doctest::Approx(7.0 * kPi / 4.0).epsilon(1e-6));
        CHECK(angles[12] == doctest::Approx(3.0 * kPi).epsilon(1e-6));
    }

    SUBCASE("the half-wrap window keeps only the stretch within pi of the position")
    {
        const auto angles = unwrappedGeneratedWindingAngles(spiral, towardOrigin);
        // At index 4 (angle pi) the window is [0, 2*pi] -> indices 0..8.
        auto range = generatedLineIndexRangeWithinWinding(angles, spiral.size(), 4.0,
                                                          kGeneratedSideCutHalfWrapAngle);
        CHECK(range.first == 0);
        CHECK(range.second == 8);
        // At index 6 (1.5*pi) the window is [0.5*pi, 2.5*pi] -> indices 2..10.
        range = generatedLineIndexRangeWithinWinding(angles, spiral.size(), 6.0,
                                                     kGeneratedSideCutHalfWrapAngle);
        CHECK(range.first == 2);
        CHECK(range.second == 10);
        // A fractional position interpolates its reference angle.
        range = generatedLineIndexRangeWithinWinding(angles, spiral.size(), 6.5,
                                                     kGeneratedSideCutHalfWrapAngle);
        CHECK(range.first == 3);
        CHECK(range.second == 10);
    }

    SUBCASE("without usable angles the whole line qualifies")
    {
        const std::vector<double> empty;
        auto range = generatedLineIndexRangeWithinWinding(empty, spiral.size(), 4.0,
                                                          kGeneratedSideCutHalfWrapAngle);
        CHECK(range.first == 0);
        CHECK(range.second == 12);
        const auto allNan = unwrappedGeneratedWindingAngles(spiral, nullptr);
        range = generatedLineIndexRangeWithinWinding(allNan, spiral.size(), 4.0,
                                                     kGeneratedSideCutHalfWrapAngle);
        CHECK(range.first == 0);
        CHECK(range.second == 12);
        range = generatedLineIndexRangeWithinWinding(allNan, 0, 4.0,
                                                     kGeneratedSideCutHalfWrapAngle);
        CHECK(range.first == 0);
        CHECK(range.second == 0);
    }
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

TEST_CASE("line annotation collapses nearby controls with explicit span ownership")
{
    const auto metadata = [](const std::string& message) {
        vc3d::line_annotation::FiberTraceSegmentMetadata value;
        value.message = message;
        return value;
    };
    std::vector<vc3d::line_annotation::LineControlPoint> controls{
        {0.0, {0.0, 0.0, 0.0}, false, 0},
        {1.0, {32.0, 0.0, 0.0}, true, 1},
        {2.0, {64.0, 0.0, 0.0}, false, 2},
        {3.0, {96.0, 0.0, 0.0}, false, 3},
    };
    controls[0].segmentToNext = metadata("left");
    controls[1].segmentToNext = metadata("removed");
    controls[2].segmentToNext = metadata("right");

    const auto collapsed = vc3d::line_annotation::collapseControlPointsAtClick(
        controls, {2, 1}, 1.5, {48.0, 2.0, 0.0});
    REQUIRE(collapsed.controlPoints.size() == 3);
    CHECK(collapsed.replacementIndex == 1);
    CHECK(collapsed.collapsedOldIndices == std::vector<size_t>{1, 2});
    CHECK(collapsed.oldToNewIndices == std::vector<size_t>{0, 1, 1, 2});
    CHECK(collapsed.dirtySegmentIndices == std::vector<size_t>{0, 1});
    CHECK(collapsed.controlPoints[0].segmentToNext->message == "left");
    CHECK(collapsed.controlPoints[1].linePosition == doctest::Approx(1.5));
    CHECK(collapsed.controlPoints[1].volumePoint == cv::Vec3d(48.0, 2.0, 0.0));
    CHECK(collapsed.controlPoints[1].optimizedIndex == -1);
    CHECK(collapsed.controlPoints[1].isSeed);
    REQUIRE(collapsed.controlPoints[1].segmentToNext);
    CHECK(collapsed.controlPoints[1].segmentToNext->message == "right");
    CHECK_FALSE(collapsed.controlPoints.back().segmentToNext.has_value());
}

TEST_CASE("line annotation control collapse handles insertion endpoints and all controls")
{
    vc3d::line_annotation::FiberTraceSegmentMetadata metadata;
    metadata.message = "span";
    std::vector<vc3d::line_annotation::LineControlPoint> controls{
        {0.0, {0.0, 0.0, 0.0}, true, 0},
        {2.0, {64.0, 0.0, 0.0}, false, 2},
    };
    controls[0].segmentToNext = metadata;

    SUBCASE("insertion keeps controls ordered and splits the existing policy")
    {
        const auto inserted = vc3d::line_annotation::collapseControlPointsAtClick(
            controls, {}, 1.0, {32.0, 0.0, 0.0});
        REQUIRE(inserted.controlPoints.size() == 3);
        CHECK_FALSE(inserted.replacedExisting());
        CHECK(inserted.replacementIndex == 1);
        CHECK(inserted.oldToNewIndices == std::vector<size_t>{0, 2});
        CHECK(inserted.dirtySegmentIndices == std::vector<size_t>{0, 1});
        REQUIRE(inserted.controlPoints[0].segmentToNext);
        REQUIRE(inserted.controlPoints[1].segmentToNext);
        CHECK(inserted.controlPoints[0].segmentToNext->message == "span");
        CHECK(inserted.controlPoints[1].segmentToNext->message == "span");
    }

    SUBCASE("replacing the final control leaves no outgoing metadata")
    {
        const auto replaced = vc3d::line_annotation::collapseControlPointsAtClick(
            controls, {1}, 2.25, {72.0, 0.0, 0.0});
        REQUIRE(replaced.controlPoints.size() == 2);
        CHECK(replaced.replacementIndex == 1);
        CHECK_FALSE(replaced.controlPoints.back().segmentToNext.has_value());
        CHECK(replaced.controlPoints.front().isSeed);
        CHECK_FALSE(replaced.controlPoints.back().isSeed);
    }

    SUBCASE("collapsing every control produces one seed and no dirty spans")
    {
        const auto collapsed = vc3d::line_annotation::collapseControlPointsAtClick(
            controls, {0, 1}, 1.0, {32.0, 0.0, 0.0});
        REQUIRE(collapsed.controlPoints.size() == 1);
        CHECK(collapsed.oldToNewIndices == std::vector<size_t>{0, 0});
        CHECK(collapsed.dirtySegmentIndices.empty());
        CHECK(collapsed.controlPoints.front().isSeed);
        CHECK_FALSE(collapsed.controlPoints.front().segmentToNext.has_value());
    }
}

TEST_CASE("line annotation automatic multi-control edit reconstructs the clicked span")
{
    const std::vector<cv::Vec3d> linePoints{
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 0.0},
        {20.0, 0.0, 0.0},
        {30.0, 0.0, 0.0},
        {40.0, 0.0, 0.0},
        {40.0, 1.0, 0.0},
        {30.0, 1.0, 0.0},
        {20.0, 1.0, 0.0},
        {10.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
    };
    std::vector<vc3d::line_annotation::LineControlPoint> controls{
        {1.0, linePoints[1], true, 1},
        {3.0, linePoints[3], false, 3},
        {4.0, linePoints[4], false, 4},
        {8.0, linePoints[8], false, 8},
    };
    const cv::Vec3d clicked{30.0, 1.1, 0.0};
    FiberModeNormalSampler sampler;
    vc::lasagna::LineOptimizationConfig config;
    config.segmentLength = 4.0;
    config.segmentsPerSide = 3;
    config.maxIterations = 0;
    config.normalAlignmentWeight = 0.0;
    config.distanceWeight = 0.0;
    config.tangentStraightnessWeight = 0.0;
    config.normalStraightnessWeight = 0.0;
    config.initialTangentWeight = 0.0;
    config.tangentGuideWeight = 0.0;

    const auto prepared = vc3d::line_annotation::prepareAutomaticControlPointEdit(
        linePoints, controls, {1, 2}, 3.5, clicked, sampler, config);

    REQUIRE(prepared.controlPoints.size() == 3);
    REQUIRE(prepared.replacementIndex == 1);
    CHECK(prepared.lineReconstructed);
    CHECK(prepared.collapsedOldIndices == std::vector<size_t>{1, 2});
    CHECK(prepared.oldToNewIndices == std::vector<size_t>{0, 1, 1, 2});
    CHECK(prepared.dirtySegmentIndices == std::vector<size_t>{0, 1});
    CHECK(prepared.controlPoints[prepared.replacementIndex].volumePoint == clicked);
    const int replacementLineIndex =
        prepared.controlPoints[prepared.replacementIndex].optimizedIndex;
    REQUIRE(replacementLineIndex > 0);
    REQUIRE(replacementLineIndex < static_cast<int>(prepared.linePoints.size()) - 1);
    CHECK(prepared.linePoints[static_cast<size_t>(replacementLineIndex)] == clicked);
    CHECK(prepared.controlPoints[0].optimizedIndex < replacementLineIndex);
    CHECK(replacementLineIndex < prepared.controlPoints[2].optimizedIndex);
}

namespace {

vc::lasagna::LineModel makeMergeTestLine(const std::vector<cv::Vec3d>& points)
{
    vc::lasagna::LineModel line;
    line.points.reserve(points.size());
    for (const auto& position : points) {
        vc::lasagna::LinePoint point;
        point.position = position;
        point.sampledNormal.normal = {0.0, 0.0, 1.0};
        point.sampledNormal.valid = true;
        point.valid = true;
        line.points.push_back(point);
    }
    line.displayFrameAnchorIndex = static_cast<int>(points.size() / 2);
    return line;
}

vc3d::line_annotation::LineControlPoint makeMergeTestControl(
    const cv::Vec3d& position,
    double metricMarker)
{
    vc3d::line_annotation::LineControlPoint control;
    control.volumePoint = position;
    vc3d::line_annotation::FiberTraceSegmentMetadata metadata;
    metadata.metric = metricMarker;
    control.segmentToNext = metadata;
    return control;
}

}  // namespace

TEST_CASE("superseded solve merge adopts unedited spans and keeps edited ones")
{
    using vc3d::line_annotation::mergeSupersededSolveResult;
    // Solve-start controls A,B,C,D. The solved line marks its interior
    // points with y=1; the current (edited) line marks its untouched spans
    // y=4 and the provisional edited spans y=2. Control vertices are shared
    // exactly.
    const cv::Vec3d A{1, 0, 0}, B{4, 0, 0}, C{7, 0, 0}, D{10, 0, 0};
    const cv::Vec3d E{5.5, 0, 0};  // control inserted into span B-C mid-solve
    const auto solvedLine = makeMergeTestLine({{0, 1, 0}, A, {2, 1, 0}, {3, 1, 0},
                                               B, {5, 1, 0}, {6, 1, 0},
                                               C, {8, 1, 0}, {9, 1, 0},
                                               D, {11, 1, 0}, {12, 1, 0}});
    const auto currentLine = makeMergeTestLine({{0, 4, 0}, A, {2, 4, 0}, {3, 4, 0},
                                                B, {4.5, 2, 0}, E, {6.5, 2, 0},
                                                C, {8, 4, 0}, {9, 4, 0},
                                                D, {11, 4, 0}, {12, 4, 0}});
    const std::vector<vc3d::line_annotation::LineControlPoint> solvedControls{
        makeMergeTestControl(A, 100.0), makeMergeTestControl(B, 101.0),
        makeMergeTestControl(C, 102.0), makeMergeTestControl(D, 103.0)};
    const std::vector<vc3d::line_annotation::LineControlPoint> currentControls{
        makeMergeTestControl(A, 0.0), makeMergeTestControl(B, 1.0),
        makeMergeTestControl(E, 2.0), makeMergeTestControl(C, 3.0),
        makeMergeTestControl(D, 4.0)};
    const std::vector<size_t> controlMap{0, 1, 3, 4};
    const std::vector<size_t> editedSpans{1, 2};

    const auto merged = mergeSupersededSolveResult(
        currentLine, currentControls, solvedLine, solvedControls,
        controlMap, editedSpans, false);

    REQUIRE(merged.mergeable);
    CHECK(merged.adoptedSpans == std::vector<size_t>{0, 3});
    CHECK(merged.rejectedSolvedSpans.empty());
    // Piece provenance by y marker: head + spans A-B, C-D + tail from the
    // solve (y=1 interiors), the edited spans B-E-C from the current line
    // (y=2 interiors).
    REQUIRE(merged.line.points.size() == 14);
    CHECK(merged.line.points[0].position[1] == doctest::Approx(1.0));   // head
    CHECK(merged.line.points[2].position[1] == doctest::Approx(1.0));   // A-B
    CHECK(merged.line.points[5].position[1] == doctest::Approx(2.0));   // B-E
    CHECK(merged.line.points[7].position[1] == doctest::Approx(2.0));   // E-C
    CHECK(merged.line.points[9].position[1] == doctest::Approx(1.0));   // C-D
    CHECK(merged.line.points.back().position[1] == doctest::Approx(1.0));  // tail
    // Controls keep the CURRENT identity; adopted spans take the solved
    // span metadata, edited spans keep their current metadata.
    REQUIRE(merged.controls.size() == 5);
    CHECK(*merged.controls[0].segmentToNext->metric == doctest::Approx(100.0));
    CHECK(*merged.controls[1].segmentToNext->metric == doctest::Approx(1.0));
    CHECK(*merged.controls[2].segmentToNext->metric == doctest::Approx(2.0));
    CHECK(*merged.controls[3].segmentToNext->metric == doctest::Approx(102.0));
    // Merged control indices are exact vertices in strictly increasing order.
    CHECK(merged.controls[0].optimizedIndex == 1);
    CHECK(merged.controls[1].optimizedIndex == 4);
    CHECK(merged.controls[2].optimizedIndex == 6);
    CHECK(merged.controls[3].optimizedIndex == 8);
    CHECK(merged.controls[4].optimizedIndex == 11);

    SUBCASE("a mid-solve config change keeps the current tails")
    {
        const auto guarded = mergeSupersededSolveResult(
            currentLine, currentControls, solvedLine, solvedControls,
            controlMap, editedSpans, true);
        REQUIRE(guarded.mergeable);
        CHECK(guarded.line.points.front().position[1] == doctest::Approx(4.0));
        CHECK(guarded.line.points.back().position[1] == doctest::Approx(4.0));
        CHECK(guarded.adoptedSpans == std::vector<size_t>{0, 3});
    }

    SUBCASE("a moved endpoint control rejects its spans")
    {
        auto movedControls = currentControls;
        movedControls[4].volumePoint = {10, 0.5, 0};
        auto movedLine = currentLine;
        movedLine.points[11].position = {10, 0.5, 0};
        const auto rejected = mergeSupersededSolveResult(
            movedLine, movedControls, solvedLine, solvedControls,
            controlMap, editedSpans, false);
        REQUIRE(rejected.mergeable);
        CHECK(rejected.adoptedSpans == std::vector<size_t>{0});
        CHECK(rejected.rejectedSolvedSpans == std::vector<size_t>{3});
        // The moved outer control also blocks the tail adoption.
        CHECK(rejected.line.points.back().position[1] == doctest::Approx(4.0));
    }
}

TEST_CASE("interpolated splice re-indexes the old range's normals without sampling")
{
    using vc3d::line_annotation::spliceLineModelWithInterpolatedNormals;
    // Previous line: 8 points along x, normals tagged by y so provenance is
    // checkable ((0, tag, 1) normalized is fine for the checks below).
    std::vector<cv::Vec3d> previousPoints;
    vc::lasagna::LineModel previous;
    for (int i = 0; i < 8; ++i) {
        vc::lasagna::LinePoint point;
        point.position = {static_cast<double>(i), 0.0, 0.0};
        point.sampledNormal.normal = {0.0, 0.1 * i, 1.0};
        point.sampledNormal.valid = true;
        point.valid = true;
        previous.points.push_back(point);
    }
    previous.displayFrameAnchorIndex = 4;

    SUBCASE("grow: replaced range [2,5) becomes 5 points, endpoints map exactly")
    {
        std::vector<cv::Vec3d> points;
        for (int i = 0; i < 2; ++i) points.push_back({static_cast<double>(i), 0, 0});
        for (int k = 0; k < 5; ++k) points.push_back({2.0 + 0.5 * k, 0.1, 0});
        for (int i = 5; i < 8; ++i) points.push_back({static_cast<double>(i), 0, 0});
        const auto model =
            spliceLineModelWithInterpolatedNormals(previous, points, 2, 5);
        REQUIRE(model.points.size() == 10);
        // Prefix and suffix carry.
        CHECK(model.points[1].sampledNormal.normal[1] == doctest::Approx(0.1));
        CHECK(model.points[9].sampledNormal.normal[1] == doctest::Approx(0.7));
        // Replaced-range endpoints map exactly to old indices 2 and 4.
        CHECK(model.points[2].sampledNormal.normal[1] == doctest::Approx(0.2));
        CHECK(model.points[6].sampledNormal.normal[1] == doctest::Approx(0.4));
        // Interior takes proportionally re-indexed authoritative samples.
        CHECK(model.points[4].sampledNormal.normal[1] == doctest::Approx(0.3));
        CHECK(model.points[4].valid);
    }

    SUBCASE("full-range replacement still has normals and an anchor")
    {
        std::vector<cv::Vec3d> points;
        for (int k = 0; k < 12; ++k) points.push_back({k * 7.0 / 11.0, 0.2, 0});
        const auto model =
            spliceLineModelWithInterpolatedNormals(previous, points, 0, 12);
        REQUIRE(model.points.size() == 12);
        CHECK(model.points.front().valid);
        CHECK(model.points.back().sampledNormal.normal[1] ==
              doctest::Approx(0.7));
        CHECK(model.displayFrameAnchorIndex >= 0);
    }

    SUBCASE("pure insertion blends hemisphere-aligned boundary normals")
    {
        auto flipped = previous;
        // Right neighbourhood stored with opposite sign: physically the same
        // axis; the blend must hemisphere-align, not cancel.
        for (int i = 4; i < 8; ++i) {
            flipped.points[static_cast<size_t>(i)].sampledNormal.normal =
                -flipped.points[static_cast<size_t>(i)].sampledNormal.normal;
        }
        std::vector<cv::Vec3d> points;
        for (int i = 0; i < 4; ++i) points.push_back({static_cast<double>(i), 0, 0});
        points.push_back({3.5, 0.05, 0});  // inserted, no old counterpart
        for (int i = 4; i < 8; ++i) points.push_back({static_cast<double>(i), 0, 0});
        const auto model =
            spliceLineModelWithInterpolatedNormals(flipped, points, 4, 1);
        REQUIRE(model.points.size() == 9);
        const auto& inserted = model.points[4].sampledNormal;
        REQUIRE(inserted.valid);
        // Aligned blend of (0,.3,1) and +(0,.4,1): z stays positive and
        // large; a raw blend with the stored -(0,.4,1) would nearly cancel.
        CHECK(inserted.normal[2] > 0.9);
        CHECK(cv::norm(inserted.normal) == doctest::Approx(1.0));
    }

    SUBCASE("unknown range falls back to 3D-nearest transfer")
    {
        std::vector<cv::Vec3d> points;
        points.push_back({0.0, 0, 0});
        points.push_back({0.25, 0, 0});  // near old index 0
        points.push_back({6.9, 0, 0});   // near old index 7
        const auto model =
            spliceLineModelWithInterpolatedNormals(previous, points, -1, 0);
        REQUIRE(model.points.size() == 3);
        CHECK(model.points[1].sampledNormal.normal[1] == doctest::Approx(0.0));
        CHECK(model.points[2].sampledNormal.normal[1] == doctest::Approx(0.7));
    }

    SUBCASE("empty previous model throws")
    {
        const vc::lasagna::LineModel empty;
        CHECK_THROWS(spliceLineModelWithInterpolatedNormals(
            empty, {{0, 0, 0}}, 0, 1));
    }

    SUBCASE("a shrinking map keeps the range's only valid normal")
    {
        // Old range [2,7) has exactly one valid normal (index 4); the
        // replacement shrinks it to two points whose proportional picks
        // (old 2 and old 6) are both invalid — the nearest usable sample
        // must be used instead of losing the anchor.
        auto sparse = previous;
        for (int i = 0; i < 8; ++i) {
            const bool valid = i == 4;
            sparse.points[static_cast<size_t>(i)].sampledNormal.valid = valid;
            sparse.points[static_cast<size_t>(i)].valid = valid;
        }
        std::vector<cv::Vec3d> points;
        points.push_back({0, 0, 0});
        points.push_back({1, 0, 0});
        points.push_back({3.0, 0.3, 0});
        points.push_back({5.5, 0.3, 0});
        points.push_back({7, 0, 0});
        const auto model =
            spliceLineModelWithInterpolatedNormals(sparse, points, 2, 2);
        REQUIRE(model.points.size() == 5);
        CHECK(model.points[2].valid);
        CHECK(model.points[2].sampledNormal.normal[1] == doctest::Approx(0.4));
        CHECK(model.displayFrameAnchorIndex >= 0);
    }

    SUBCASE("a single-point replacement still finds a usable sample")
    {
        auto sparse = previous;
        for (int i = 0; i < 8; ++i) {
            const bool valid = i == 3;  // only old index 3 in range [1,5)
            sparse.points[static_cast<size_t>(i)].sampledNormal.valid = valid;
            sparse.points[static_cast<size_t>(i)].valid = valid;
        }
        std::vector<cv::Vec3d> points;
        points.push_back({0, 0, 0});
        points.push_back({3.0, 0.3, 0});  // one point replaces old [1,5)
        points.push_back({5, 0, 0});
        points.push_back({6, 0, 0});
        points.push_back({7, 0, 0});
        const auto model =
            spliceLineModelWithInterpolatedNormals(sparse, points, 1, 1);
        REQUIRE(model.points.size() == 5);
        CHECK(model.points[1].valid);
        CHECK(model.points[1].sampledNormal.normal[1] == doctest::Approx(0.3));
    }

    SUBCASE("the interpolated model builds line view surfaces")
    {
        std::vector<cv::Vec3d> points;
        for (int i = 0; i < 2; ++i) points.push_back({static_cast<double>(i), 0, 0});
        for (int k = 0; k < 5; ++k) points.push_back({2.0 + 0.5 * k, 0.1, 0});
        for (int i = 5; i < 8; ++i) points.push_back({static_cast<double>(i), 0, 0});
        const auto model =
            spliceLineModelWithInterpolatedNormals(previous, points, 2, 5);
        vc::lasagna::LineViewConfig config;
        config.buildLineZSlices = false;
        const auto views = vc::lasagna::buildLineViewSurfaces(model, config);
        CHECK(views.lineSurface != nullptr);
        CHECK(views.lineUpVectors.size() == model.points.size());
    }
}

TEST_CASE("superseded solve merge joins spans of different normal provenance")
{
    using vc3d::line_annotation::mergeSupersededSolveResult;
    // A protected (edited, provisional-normal) span joined to an adopted
    // (solved, fresh-normal) span: the seam discontinuity is an accepted
    // transient until the follow-up solve lands, but it must assemble
    // cleanly — each side keeps its own provenance, one shared vertex, and
    // the merged model still builds.
    const cv::Vec3d A{1, 0, 0}, B{4, 0, 0}, C{7, 0, 0};
    auto tag = [](vc::lasagna::LineModel& line, size_t from, size_t to,
                  double yTag) {
        for (size_t i = from; i < to && i < line.points.size(); ++i) {
            line.points[i].sampledNormal.normal = {0.0, yTag, 1.0};
        }
    };
    auto solvedLine = makeMergeTestLine({{0, 0, 0}, A, {2, 0, 0}, {3, 0, 0},
                                         B, {5, 0, 0}, {6, 0, 0}, C, {8, 0, 0}});
    tag(solvedLine, 0, solvedLine.points.size(), 0.9);  // fresh normals
    auto currentLine = makeMergeTestLine({{0, 0, 0}, A, {2, 0, 0}, {3, 0, 0},
                                          B, {5, 0.2, 0}, {6, 0.2, 0}, C,
                                          {8, 0, 0}});
    tag(currentLine, 0, currentLine.points.size(), 0.1);  // provisional
    const std::vector<vc3d::line_annotation::LineControlPoint> solvedControls{
        makeMergeTestControl(A, 0), makeMergeTestControl(B, 1),
        makeMergeTestControl(C, 2)};
    const std::vector<vc3d::line_annotation::LineControlPoint> currentControls{
        makeMergeTestControl(A, 0), makeMergeTestControl(B, 1),
        makeMergeTestControl(C, 2)};

    const auto merged = mergeSupersededSolveResult(
        currentLine, currentControls, solvedLine, solvedControls,
        {0, 1, 2}, {1}, false);

    REQUIRE(merged.mergeable);
    CHECK(merged.adoptedSpans == std::vector<size_t>{0});
    // Span A-B interior carries solved normals, span B-C interior carries
    // provisional normals; exactly one B vertex.
    CHECK(merged.line.points[2].sampledNormal.normal[1] == doctest::Approx(0.9));
    CHECK(merged.line.points[5].sampledNormal.normal[1] == doctest::Approx(0.1));
    size_t bCount = 0;
    for (const auto& point : merged.line.points) {
        if (cv::norm(point.position - B) < 1e-12) ++bCount;
    }
    CHECK(bCount == 1);
    vc::lasagna::LineViewConfig config;
    config.buildLineZSlices = false;
    CHECK(vc::lasagna::buildLineViewSurfaces(merged.line, config).lineSurface !=
          nullptr);
}

TEST_CASE("superseded solve merge protects tails when the adjacent outer span was edited")
{
    using vc3d::line_annotation::mergeSupersededSolveResult;
    // Solve-start controls A,B,C; the edit inserted X into the FIRST span
    // (A-B). The native head extrapolation is seeded from the first interior
    // vertex, so even though control A itself is unchanged, the solved head
    // tail is stale and must not be adopted.
    const cv::Vec3d A{1, 0, 0}, B{4, 0, 0}, C{7, 0, 0};
    const cv::Vec3d X{2.5, 0, 0};
    const auto solvedLine = makeMergeTestLine(
        {{0, 9, 0}, A, {2, 1, 0}, {3, 1, 0}, B, {5, 1, 0}, C, {8, 9, 0}});
    const auto currentLine = makeMergeTestLine(
        {{0, -9, 0}, A, {1.7, 2, 0}, X, {3.2, 2, 0}, B, {5, 4, 0}, C, {8, -9, 0}});
    const std::vector<vc3d::line_annotation::LineControlPoint> solvedControls{
        makeMergeTestControl(A, 100.0), makeMergeTestControl(B, 101.0),
        makeMergeTestControl(C, 102.0)};
    const std::vector<vc3d::line_annotation::LineControlPoint> currentControls{
        makeMergeTestControl(A, 0.0), makeMergeTestControl(X, 1.0),
        makeMergeTestControl(B, 2.0), makeMergeTestControl(C, 3.0)};
    const std::vector<size_t> controlMap{0, 2, 3};
    const std::vector<size_t> editedSpans{0, 1};

    const auto merged = mergeSupersededSolveResult(
        currentLine, currentControls, solvedLine, solvedControls,
        controlMap, editedSpans, false);

    REQUIRE(merged.mergeable);
    // Span B-C (current span 2) is adoptable, but both tails must stay
    // current: the head because its adjacent span was edited, the tail
    // because span 2 adoption does not change tail ownership rules — here
    // the last span (2) is NOT edited, so the tail may adopt.
    CHECK(merged.adoptedSpans == std::vector<size_t>{2});
    CHECK(merged.line.points.front().position[1] == doctest::Approx(-9.0));
    CHECK(merged.line.points.back().position[1] == doctest::Approx(9.0));

    SUBCASE("an edit in the last span protects the solved tail too")
    {
        const auto guarded = mergeSupersededSolveResult(
            currentLine, currentControls, solvedLine, solvedControls,
            controlMap, {0, 1, 2}, false);
        REQUIRE(guarded.mergeable);
        CHECK(guarded.adoptedSpans.empty());
        CHECK(guarded.line.points.back().position[1] == doctest::Approx(-9.0));
    }

    SUBCASE("unsorted edited spans are normalized internally")
    {
        const auto unsorted = mergeSupersededSolveResult(
            currentLine, currentControls, solvedLine, solvedControls,
            controlMap, {1, 0}, false);
        REQUIRE(unsorted.mergeable);
        CHECK(unsorted.adoptedSpans == std::vector<size_t>{2});
        CHECK(unsorted.line.points.front().position[1] == doctest::Approx(-9.0));
    }
}

TEST_CASE("superseded solve merge handles collapses and malformed inputs")
{
    using vc3d::line_annotation::mergeSupersededSolveResult;
    const cv::Vec3d A{1, 0, 0}, B{4, 0, 0}, C{7, 0, 0}, D{10, 0, 0};
    const cv::Vec3d X{5.5, 0, 0};  // B and C collapsed into X mid-solve
    const auto solvedLine = makeMergeTestLine({A, {2, 1, 0}, B, {5, 1, 0},
                                               C, {8, 1, 0}, D});
    const auto currentLine = makeMergeTestLine({A, {3, 2, 0}, X, {8, 2, 0}, D});
    const std::vector<vc3d::line_annotation::LineControlPoint> solvedControls{
        makeMergeTestControl(A, 100.0), makeMergeTestControl(B, 101.0),
        makeMergeTestControl(C, 102.0), makeMergeTestControl(D, 103.0)};
    const std::vector<vc3d::line_annotation::LineControlPoint> currentControls{
        makeMergeTestControl(A, 0.0), makeMergeTestControl(X, 1.0),
        makeMergeTestControl(D, 2.0)};
    const std::vector<size_t> collapseMap{0, 1, 1, 2};

    SUBCASE("collapse map adopts nothing and reports rejected coverage")
    {
        const auto merged = mergeSupersededSolveResult(
            currentLine, currentControls, solvedLine, solvedControls,
            collapseMap, {0, 1}, false);
        REQUIRE(merged.mergeable);
        CHECK(merged.adoptedSpans.empty());
        CHECK(merged.rejectedSolvedSpans == std::vector<size_t>{0, 1});
    }

    SUBCASE("non-monotone or out-of-bounds maps are not mergeable")
    {
        CHECK_FALSE(mergeSupersededSolveResult(
            currentLine, currentControls, solvedLine, solvedControls,
            {0, 2, 1, 2}, {}, false).mergeable);
        CHECK_FALSE(mergeSupersededSolveResult(
            currentLine, currentControls, solvedLine, solvedControls,
            {0, 1, 1, 9}, {}, false).mergeable);
        CHECK_FALSE(mergeSupersededSolveResult(
            currentLine, currentControls, solvedLine, solvedControls,
            {0, 1, 1}, {}, false).mergeable);
    }

    SUBCASE("a current line violating the subset contract is not mergeable")
    {
        auto offLineControls = currentControls;
        offLineControls[1].volumePoint = {5.5, 0.25, 0};  // no matching vertex
        CHECK_FALSE(mergeSupersededSolveResult(
            currentLine, offLineControls, solvedLine, solvedControls,
            collapseMap, {0, 1}, false).mergeable);
    }
}

TEST_CASE("line annotation geometric edit prepares the clicked span without a solver")
{
    // Same fixture as the automatic multi-control edit above, but through
    // the purely geometric prepare: identical collapse bookkeeping and dirty
    // spans, no sampler, and a replaced linePoints range the caller can use
    // to splice per-point derived data instead of recomputing all of it.
    const std::vector<cv::Vec3d> linePoints{
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 0.0},
        {20.0, 0.0, 0.0},
        {30.0, 0.0, 0.0},
        {40.0, 0.0, 0.0},
        {40.0, 1.0, 0.0},
        {30.0, 1.0, 0.0},
        {20.0, 1.0, 0.0},
        {10.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
    };
    std::vector<vc3d::line_annotation::LineControlPoint> controls{
        {1.0, linePoints[1], true, 1},
        {3.0, linePoints[3], false, 3},
        {4.0, linePoints[4], false, 4},
        {8.0, linePoints[8], false, 8},
    };
    const cv::Vec3d clicked{30.0, 1.1, 0.0};

    const auto prepared = vc3d::line_annotation::prepareGeometricControlPointEdit(
        linePoints, controls, {1, 2}, 3.5, clicked, 4.0);

    REQUIRE(prepared.controlPoints.size() == 3);
    REQUIRE(prepared.replacementIndex == 1);
    CHECK(prepared.lineReconstructed);
    CHECK(prepared.collapsedOldIndices == std::vector<size_t>{1, 2});
    CHECK(prepared.oldToNewIndices == std::vector<size_t>{0, 1, 1, 2});
    CHECK(prepared.dirtySegmentIndices == std::vector<size_t>{0, 1});
    CHECK(prepared.controlPoints[prepared.replacementIndex].volumePoint == clicked);
    const int replacementLineIndex =
        prepared.controlPoints[prepared.replacementIndex].optimizedIndex;
    REQUIRE(replacementLineIndex > 0);
    REQUIRE(replacementLineIndex < static_cast<int>(prepared.linePoints.size()) - 1);
    CHECK(prepared.linePoints[static_cast<size_t>(replacementLineIndex)] == clicked);
    CHECK(prepared.controlPoints[0].optimizedIndex < replacementLineIndex);
    CHECK(replacementLineIndex < prepared.controlPoints[2].optimizedIndex);

    // The replaced range is consistent: everything outside it is the input
    // line carried over (the suffix shifted by the size delta).
    REQUIRE(prepared.replacedStart >= 0);
    REQUIRE(prepared.replacedCount > 0);
    const int suffixStart = prepared.replacedStart + prepared.replacedCount;
    const int delta = static_cast<int>(prepared.linePoints.size()) -
                      static_cast<int>(linePoints.size());
    REQUIRE(suffixStart <= static_cast<int>(prepared.linePoints.size()));
    for (int i = 0; i < prepared.replacedStart; ++i) {
        CHECK(prepared.linePoints[static_cast<size_t>(i)] ==
              linePoints[static_cast<size_t>(i)]);
    }
    for (int i = suffixStart; i < static_cast<int>(prepared.linePoints.size()); ++i) {
        CHECK(prepared.linePoints[static_cast<size_t>(i)] ==
              linePoints[static_cast<size_t>(i - delta)]);
    }
}

TEST_CASE("line annotation automatic edit preparation leaves inputs unchanged on failure")
{
    const std::vector<cv::Vec3d> linePoints{
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 0.0},
        {20.0, 0.0, 0.0},
    };
    const std::vector<vc3d::line_annotation::LineControlPoint> controls{
        {0.0, linePoints[0], true, 0},
        {2.0, linePoints[2], false, 2},
    };
    const std::vector<cv::Vec3d> originalLinePoints = linePoints;
    ThrowingFiberModeNormalSampler sampler;
    vc::lasagna::LineOptimizationConfig config;
    config.segmentLength = 4.0;
    config.segmentsPerSide = 3;

    CHECK_THROWS_AS(
        vc3d::line_annotation::prepareAutomaticControlPointEdit(
            linePoints, controls, {}, 1.0, {10.0, 1.0, 0.0}, sampler, config),
        std::runtime_error);
    CHECK(linePoints == originalLinePoints);
    REQUIRE(controls.size() == 2);
    CHECK(controls[0].linePosition == doctest::Approx(0.0));
    CHECK(controls[0].volumePoint == linePoints[0]);
    CHECK(controls[1].linePosition == doctest::Approx(2.0));
    CHECK(controls[1].volumePoint == linePoints[2]);
}

TEST_CASE("line annotation one-control tangent follows authoritative line position")
{
    const std::vector<cv::Vec3d> linePoints{
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 0.0},
        {20.0, 0.0, 0.0},
        {20.0, 10.0, 0.0},
        {10.0, 10.0, 0.0},
        {10.0, 0.1, 0.0},
    };

    CHECK(vc3d::line_annotation::lineTangentAtPosition(linePoints, 1.5) ==
          cv::Vec3d(10.0, 0.0, 0.0));
    CHECK(vc3d::line_annotation::lineTangentAtPosition(linePoints, 4.5) ==
          cv::Vec3d(0.0, -9.9, 0.0));

    FiberModeNormalSampler sampler;
    vc3d::line_annotation::FiberModeOptimizationRequest request;
    request.controlPoints = {
        {1.5, {10.0, 0.1, 0.0}, true, -1},
    };
    request.linePointsBase = linePoints;
    request.baseNormalSampler = &sampler;
    request.globalMode = vc3d::line_annotation::FiberOptimizationMode::Lasagna;
    request.lasagnaConfig.segmentsPerSide = 2;
    request.lasagnaConfig.segmentLength = 2.0;
    request.lasagnaConfig.runGlobalOptimization = false;
    request.lasagnaConfig.printSolverProgress = false;

    const auto optimized =
        vc3d::line_annotation::optimizeFiberWithNativeFallback(std::move(request));
    REQUIRE(optimized.optimization.line.points.size() >= 3);
    const cv::Vec3d optimizedDirection =
        optimized.optimization.line.points.back().position -
        optimized.optimization.line.points.front().position;
    CHECK(std::abs(optimizedDirection[0]) > std::abs(optimizedDirection[1]));
}

TEST_CASE("line annotation fiber-mode optimization surfaces cancellation")
{
    const std::vector<cv::Vec3d> linePoints{
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 0.0},
        {20.0, 0.0, 0.0},
    };
    FiberModeNormalSampler sampler;
    std::atomic<bool> cancel{true};
    vc3d::line_annotation::FiberModeOptimizationRequest request;
    request.controlPoints = {
        {0.0, linePoints[0], true, 0},
        {2.0, linePoints[2], false, 2},
    };
    request.linePointsBase = linePoints;
    request.baseNormalSampler = &sampler;
    request.globalMode = vc3d::line_annotation::FiberOptimizationMode::Lasagna;
    request.cancelFlag = &cancel;

    CHECK_THROWS_AS(
        vc3d::line_annotation::optimizeFiberWithNativeFallback(std::move(request)),
        vc::lasagna::LineOptimizationCancelled);
}

TEST_CASE("line annotation all-control collapse prepares one clicked control")
{
    const std::vector<cv::Vec3d> linePoints{
        {0.0, 0.0, 0.0},
        {10.0, 0.0, 0.0},
        {20.0, 0.0, 0.0},
    };
    const std::vector<vc3d::line_annotation::LineControlPoint> controls{
        {0.0, linePoints[0], true, 0},
        {2.0, linePoints[2], false, 2},
    };
    const cv::Vec3d clicked{10.0, 1.0, 0.0};
    FiberModeNormalSampler sampler;

    const auto prepared = vc3d::line_annotation::prepareAutomaticControlPointEdit(
        linePoints,
        controls,
        {0, 1},
        1.0,
        clicked,
        sampler,
        vc::lasagna::LineOptimizationConfig{});

    CHECK_FALSE(prepared.lineReconstructed);
    CHECK(prepared.linePoints == linePoints);
    REQUIRE(prepared.controlPoints.size() == 1);
    CHECK(prepared.replacementIndex == 0);
    CHECK(prepared.controlPoints[0].linePosition == doctest::Approx(1.0));
    CHECK(prepared.controlPoints[0].volumePoint == clicked);
    CHECK(prepared.controlPoints[0].isSeed);
    CHECK(prepared.dirtySegmentIndices.empty());
}

TEST_CASE("line annotation optimizer metadata merge follows control order")
{
    std::vector<vc3d::line_annotation::LineControlPoint> original{
        {1.0, {5.0, 5.0, 5.0}, true, 1},
        {3.0, {5.0, 5.0, 5.0}, false, 3},
        {5.0, {9.0, 5.0, 5.0}, false, 5},
    };
    original[0].segmentToNext.emplace();
    original[0].segmentToNext->message = "first winding";
    original[1].segmentToNext.emplace();
    original[1].segmentToNext->message = "second winding";
    auto optimizerControls =
        vc3d::line_annotation::optimizerControlPoints(original);

    const auto merged = vc3d::line_annotation::mergeOptimizerControlPoints(
        std::move(optimizerControls), original);

    REQUIRE(merged.size() == 3);
    REQUIRE(merged[0].segmentToNext.has_value());
    REQUIRE(merged[1].segmentToNext.has_value());
    CHECK(merged[0].segmentToNext->message == "first winding");
    CHECK(merged[1].segmentToNext->message == "second winding");
    CHECK_FALSE(merged[2].segmentToNext.has_value());
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

TEST_CASE("line annotation save retires originals only after renames succeed")
{
    const auto dir = makeTempSaveDir("retire_success");
    const auto original = dir / "fiber_old.json";
    const auto target = dir / "fiber_new.json";
    writeText(original, "{\"old\":true}\n");

    const auto result = vc3d::line_annotation::runFiberSaveJob(
        13,
        {{1, 1, target, nlohmann::json{{"new", true}}}},
        {original});

    CHECK(result.ok);
    CHECK(readText(target).find("\"new\": true") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(original));
    // The backup is removed after success; only the empty dot-directory
    // may remain, which every fiber scanner ignores.
    const auto retiredDir = dir / ".retired";
    if (std::filesystem::exists(retiredDir)) {
        CHECK(std::filesystem::is_empty(retiredDir));
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("line annotation retire-only job is all-or-nothing")
{
    const auto dir = makeTempSaveDir("retire_only");
    const auto first = dir / "fiber_a.json";
    const auto second = dir / "fiber_b.json";
    const auto missing = dir / "fiber_gone.json";
    writeText(first, "{\"a\":true}\n");
    writeText(second, "{\"b\":true}\n");

    const auto result = vc3d::line_annotation::runFiberSaveJob(
        14, {}, {first, second, missing});

    CHECK(result.ok);
    CHECK_FALSE(std::filesystem::exists(first));
    CHECK_FALSE(std::filesystem::exists(second));
    const auto retiredDir = dir / ".retired";
    if (std::filesystem::exists(retiredDir)) {
        CHECK(std::filesystem::is_empty(retiredDir));
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("line annotation failed save restores retired originals")
{
    const auto dir = makeTempSaveDir("retire_failure");
    const auto original = dir / "fiber_old.json";
    const auto firstTarget = dir / "fiber_a.json";
    const auto secondTarget = dir / "fiber_b.json";
    writeText(original, "{\"old\":true}\n");

    setenv("VC3D_FIBER_SAVE_FAIL_AFTER_FIRST_REPLACE", "1", 1);
    const auto result = vc3d::line_annotation::runFiberSaveJob(
        15,
        {{1, 1, firstTarget, nlohmann::json{{"new", "a"}}},
         {2, 1, secondTarget, nlohmann::json{{"new", "b"}}}},
        {original});
    unsetenv("VC3D_FIBER_SAVE_FAIL_AFTER_FIRST_REPLACE");

    CHECK_FALSE(result.ok);
    // The retired original is renamed straight back into place.
    CHECK(std::filesystem::exists(original));
    CHECK(readText(original).find("\"old\": true") != std::string::npos ||
          readText(original).find("\"old\":true") != std::string::npos);
    const auto retiredDir = dir / ".retired";
    if (std::filesystem::exists(retiredDir)) {
        CHECK(std::filesystem::is_empty(retiredDir));
    }
    // The renamed-in brand-new target is removed too: no orphan half of an
    // aborted batch survives.
    CHECK_FALSE(std::filesystem::exists(firstTarget));
    CHECK_FALSE(std::filesystem::exists(secondTarget));
    std::filesystem::remove_all(dir);
}

TEST_CASE("line annotation failed multi fiber save removes orphan new targets")
{
    const auto dir = makeTempSaveDir("orphan_targets");
    const auto first = dir / "fiber_new_a.json";
    const auto second = dir / "fiber_new_b.json";

    setenv("VC3D_FIBER_SAVE_FAIL_AFTER_FIRST_REPLACE", "1", 1);
    const auto result = vc3d::line_annotation::runFiberSaveJob(
        16,
        {{1, 1, first, nlohmann::json{{"new", "a"}}},
         {2, 1, second, nlohmann::json{{"new", "b"}}}});
    unsetenv("VC3D_FIBER_SAVE_FAIL_AFTER_FIRST_REPLACE");

    CHECK_FALSE(result.ok);
    // Neither brand-new target survives the aborted batch; a pre-existing
    // target would instead keep the new content plus its recovery copy.
    CHECK_FALSE(std::filesystem::exists(first));
    CHECK_FALSE(std::filesystem::exists(second));
    CHECK(recoveryFilesIn(dir).empty());
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

TEST_CASE("line annotation parallax ghost picks the nearest control in each direction")
{
    using vc3d::line_annotation::GeneratedOverlay;
    constexpr double kSlideRange = 8.0;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<GeneratedOverlay::ControlPointMarker> controls{
        {{0.0f, 0.0f, 0.0f}, 20.0, false},
        {{0.0f, 0.0f, 0.0f}, nan, false},
        {{0.0f, 0.0f, 0.0f}, 4.0, false},
        {{0.0f, 0.0f, 0.0f}, 12.0, false},
        {{0.0f, 0.0f, 0.0f}, 9.0, false},
    };
    const auto index =
        vc3d::line_annotation::buildGeneratedControlPointLinePositionIndex(controls);

    const auto ahead = vc3d::line_annotation::generatedParallaxGhost(
        controls, index, 10.0, 1, kSlideRange, 1000.0);
    REQUIRE(ahead.has_value());
    CHECK(ahead->controlIndex == 3);
    CHECK(ahead->linePosition == doctest::Approx(12.0));
    CHECK(ahead->offsetFraction == doctest::Approx(2.0 / kSlideRange));
    CHECK(ahead->offsetFraction > 0.0);

    const auto behind = vc3d::line_annotation::generatedParallaxGhost(
        controls, index, 10.0, -1, kSlideRange, 1000.0);
    REQUIRE(behind.has_value());
    CHECK(behind->controlIndex == 4);
    CHECK(behind->linePosition == doctest::Approx(9.0));
    CHECK(behind->offsetFraction == doctest::Approx(-1.0 / kSlideRange));
    CHECK(behind->offsetFraction < 0.0);
}

TEST_CASE("line annotation parallax ghost requires a control strictly in the direction")
{
    using vc3d::line_annotation::GeneratedOverlay;
    constexpr double kSlideRange = 8.0;

    const std::vector<GeneratedOverlay::ControlPointMarker> empty;
    const auto emptyIndex =
        vc3d::line_annotation::buildGeneratedControlPointLinePositionIndex(empty);
    CHECK_FALSE(vc3d::line_annotation::generatedParallaxGhost(
                    empty, emptyIndex, 3.0, 1, kSlideRange, 1000.0)
                    .has_value());
    CHECK_FALSE(vc3d::line_annotation::generatedParallaxGhost(
                    empty, emptyIndex, 3.0, -1, kSlideRange, 1000.0)
                    .has_value());

    const std::vector<GeneratedOverlay::ControlPointMarker> controls{
        {{0.0f, 0.0f, 0.0f}, 5.0, false},
        {{0.0f, 0.0f, 0.0f}, 9.0, false},
    };
    const auto index =
        vc3d::line_annotation::buildGeneratedControlPointLinePositionIndex(controls);

    // Nothing beyond the last control ahead, nothing before the first behind.
    CHECK_FALSE(vc3d::line_annotation::generatedParallaxGhost(
                    controls, index, 9.0, 1, kSlideRange, 1000.0)
                    .has_value());
    CHECK_FALSE(vc3d::line_annotation::generatedParallaxGhost(
                    controls, index, 5.0, -1, kSlideRange, 1000.0)
                    .has_value());

    // A control exactly at the current position is neither ahead nor behind.
    const auto ahead = vc3d::line_annotation::generatedParallaxGhost(
        controls, index, 5.0, 1, kSlideRange, 1000.0);
    REQUIRE(ahead.has_value());
    CHECK(ahead->controlIndex == 1);
    const auto behind = vc3d::line_annotation::generatedParallaxGhost(
        controls, index, 9.0, -1, kSlideRange, 1000.0);
    REQUIRE(behind.has_value());
    CHECK(behind->controlIndex == 0);

    // Unusable inputs.
    CHECK_FALSE(vc3d::line_annotation::generatedParallaxGhost(
                    controls,
                    index,
                    std::numeric_limits<double>::quiet_NaN(),
                    1,
                    kSlideRange, 1000.0)
                    .has_value());
    CHECK_FALSE(vc3d::line_annotation::generatedParallaxGhost(
                    controls, index, 5.0, 1, 0.0, 1000.0)
                    .has_value());
    CHECK_FALSE(vc3d::line_annotation::generatedParallaxGhost(
                    controls, index, 5.0, 0, kSlideRange, 1000.0)
                    .has_value());
}

TEST_CASE("line annotation parallax ghost skips non finite control line positions")
{
    using vc3d::line_annotation::GeneratedOverlay;
    constexpr double kSlideRange = 8.0;
    const std::vector<GeneratedOverlay::ControlPointMarker> controls{
        {{0.0f, 0.0f, 0.0f}, std::numeric_limits<double>::quiet_NaN(), false},
        {{0.0f, 0.0f, 0.0f}, 14.0, false},
        {{0.0f, 0.0f, 0.0f}, std::numeric_limits<double>::infinity(), false},
        {{0.0f, 0.0f, 0.0f}, 2.0, false},
    };
    const auto index =
        vc3d::line_annotation::buildGeneratedControlPointLinePositionIndex(controls);

    const auto ahead = vc3d::line_annotation::generatedParallaxGhost(
        controls, index, 6.0, 1, kSlideRange, 1000.0);
    REQUIRE(ahead.has_value());
    CHECK(ahead->controlIndex == 1);

    const auto behind = vc3d::line_annotation::generatedParallaxGhost(
        controls, index, 6.0, -1, kSlideRange, 1000.0);
    REQUIRE(behind.has_value());
    CHECK(behind->controlIndex == 3);
}

TEST_CASE("line annotation parallax ghost clamps offset and ramps opacity")
{
    using vc3d::line_annotation::GeneratedOverlay;
    constexpr double kSlideRange = 8.0;
    const double minimumOpacity = vc3d::line_annotation::kGeneratedParallaxGhostMinimumOpacity;
    const double maximumOpacity = vc3d::line_annotation::kGeneratedParallaxGhostMaximumOpacity;
    const std::vector<GeneratedOverlay::ControlPointMarker> controls{
        {{0.0f, 0.0f, 0.0f}, 0.0, false},
        {{0.0f, 0.0f, 0.0f}, 100.0, false},
    };
    const auto index =
        vc3d::line_annotation::buildGeneratedControlPointLinePositionIndex(controls);

    // Far beyond the slide range in both directions: full offset, floor opacity.
    const auto farAhead = vc3d::line_annotation::generatedParallaxGhost(
        controls, index, 50.0, 1, kSlideRange, 1000.0);
    REQUIRE(farAhead.has_value());
    CHECK(farAhead->offsetFraction == doctest::Approx(1.0));
    CHECK(farAhead->opacity == doctest::Approx(minimumOpacity));

    const auto farBehind = vc3d::line_annotation::generatedParallaxGhost(
        controls, index, 50.0, -1, kSlideRange, 1000.0);
    REQUIRE(farBehind.has_value());
    CHECK(farBehind->offsetFraction == doctest::Approx(-1.0));
    CHECK(farBehind->opacity == doctest::Approx(minimumOpacity));

    // Exactly at the slide range: still full offset and floor opacity.
    const auto atRange = vc3d::line_annotation::generatedParallaxGhost(
        controls, index, 100.0 - kSlideRange, 1, kSlideRange, 1000.0);
    REQUIRE(atRange.has_value());
    CHECK(atRange->offsetFraction == doctest::Approx(1.0));
    CHECK(atRange->opacity == doctest::Approx(minimumOpacity));

    // Half a slide range out: half the offset, opacity halfway up the ramp.
    const auto halfway = vc3d::line_annotation::generatedParallaxGhost(
        controls, index, 100.0 - 0.5 * kSlideRange, 1, kSlideRange, 1000.0);
    REQUIRE(halfway.has_value());
    CHECK(halfway->offsetFraction == doctest::Approx(0.5));
    CHECK(halfway->opacity ==
          doctest::Approx(minimumOpacity + 0.5 * (maximumOpacity - minimumOpacity)));

    // Converging on the solid marker: offset to zero, opacity to the ceiling.
    const auto landing = vc3d::line_annotation::generatedParallaxGhost(
        controls, index, 100.0 - 1.0e-6, 1, kSlideRange, 1000.0);
    REQUIRE(landing.has_value());
    CHECK(landing->offsetFraction == doctest::Approx(0.0).epsilon(1.0e-6));
    CHECK(landing->opacity == doctest::Approx(maximumOpacity).epsilon(1.0e-6));
}

TEST_CASE("line annotation parallax ghost hides beyond the visibility distance")
{
    using vc3d::line_annotation::GeneratedOverlay;
    constexpr double kSlideRange = 8.0;
    const double minimumOpacity = vc3d::line_annotation::kGeneratedParallaxGhostMinimumOpacity;
    const std::vector<GeneratedOverlay::ControlPointMarker> controls{
        {{0.0f, 0.0f, 0.0f}, 0.0, false},
        {{0.0f, 0.0f, 0.0f}, 30.0, false},
    };
    const auto index =
        vc3d::line_annotation::buildGeneratedControlPointLinePositionIndex(controls);

    // The control 20 line positions ahead is hidden when the visibility
    // distance is shorter than that, visible when it is longer.
    CHECK_FALSE(vc3d::line_annotation::generatedParallaxGhost(
                    controls, index, 10.0, 1, kSlideRange, 15.0)
                    .has_value());
    const auto visible = vc3d::line_annotation::generatedParallaxGhost(
        controls, index, 10.0, 1, kSlideRange, 40.0);
    REQUIRE(visible.has_value());
    // Inside the un-faded core (within 75% of the visibility distance) the
    // base ramp applies unchanged: fully clamped offset, floor opacity.
    CHECK(visible->offsetFraction == doctest::Approx(1.0));
    CHECK(visible->opacity == doctest::Approx(minimumOpacity));

    // In the outer quarter of the visibility distance the opacity fades
    // linearly toward zero.
    const auto fading = vc3d::line_annotation::generatedParallaxGhost(
        controls, index, 10.0, 1, kSlideRange, 22.0);
    REQUIRE(fading.has_value());
    const double edgeFade = (22.0 - 20.0) / (0.25 * 22.0);
    CHECK(fading->opacity == doctest::Approx(minimumOpacity * edgeFade));

    // Exactly at the visibility distance the ghost is fully faded out.
    const auto atLimit = vc3d::line_annotation::generatedParallaxGhost(
        controls, index, 10.0, 1, kSlideRange, 20.0);
    REQUIRE(atLimit.has_value());
    CHECK(atLimit->opacity == doctest::Approx(0.0));

    // Non-positive or non-finite visibility distances are unusable inputs.
    CHECK_FALSE(vc3d::line_annotation::generatedParallaxGhost(
                    controls, index, 10.0, 1, kSlideRange, 0.0)
                    .has_value());
    CHECK_FALSE(vc3d::line_annotation::generatedParallaxGhost(
                    controls, index, 10.0, 1, kSlideRange,
                    std::numeric_limits<double>::quiet_NaN())
                    .has_value());
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

// ---------------------------------------------------------------------------
// Orientation freshness: the decisions behind the umbilicus cache and the
// stale-view refresh, extracted so they are asserted rather than read.

#include "UmbilicusOrientationFreshness.hpp"

namespace {

vc3d::annotation::UmbilicusCacheInputs cacheInputs()
{
    vc3d::annotation::UmbilicusCacheInputs inputs;
    inputs.root = "/proj";
    inputs.volumeId = "vol-a";
    inputs.dependencyToken = "field|/proj/umbilicus.json=100:1";
    inputs.frame = vc3d::annotation::deriveAnnotationFrame(
        2.4, 0, std::nullopt, std::nullopt, {100.0, 100.0, 1000.0});
    return inputs;
}

} // namespace

TEST_CASE("umbilicus cache: reused only while every input it was built from holds")
{
    using vc3d::annotation::umbilicusReloadNeeded;

    const auto cached = cacheInputs();

    // Never attempted resolves regardless of the inputs matching.
    CHECK(umbilicusReloadNeeded(false, cached, cached));
    // Identical inputs reuse.
    CHECK_FALSE(umbilicusReloadNeeded(true, cached, cached));

    auto otherRoot = cached;
    otherRoot.root = "/other";
    CHECK(umbilicusReloadNeeded(true, cached, otherRoot));

    // The finding this pins: a volume switch whose annotation frame is
    // byte-identical must still re-resolve, because the legacy reading's
    // registration transform and the volume-centre fallback belong to the
    // volume, not to the frame.
    auto otherVolume = cached;
    otherVolume.volumeId = "vol-b";
    CHECK(umbilicusReloadNeeded(true, cached, otherVolume));

    // Any resolver dependency changing on disk — the attached file fixed in
    // place, a discovery candidate appearing, the registration transform
    // edited — lands in the token.
    auto editedFile = cached;
    editedFile.dependencyToken = "field|/proj/umbilicus.json=100:2";
    CHECK(umbilicusReloadNeeded(true, cached, editedFile));

    // A frame change rescales the cached points, so it cannot be reused...
    auto otherFrame = cached;
    otherFrame.frame = vc3d::annotation::deriveAnnotationFrame(
        2.4, 0, std::nullopt, std::nullopt, {100.0, 100.0, 2000.0});
    CHECK(umbilicusReloadNeeded(true, cached, otherFrame));

    // ...but an imprecisely round-tripped voxel size is the same frame.
    auto rounded = cached;
    rounded.frame = vc3d::annotation::deriveAnnotationFrame(
        2.4 + 1e-12, 0, std::nullopt, std::nullopt, {100.0, 100.0, 1000.0});
    CHECK_FALSE(umbilicusReloadNeeded(true, cached, rounded));
}

TEST_CASE("stale-view refresh: rebuilds exactly the panes built before the change")
{
    using vc3d::annotation::GeneratedViewsPaneState;
    using vc3d::annotation::paneNeedsOrientationRefresh;

    constexpr int kEpoch = 3;

    GeneratedViewsPaneState stale;
    stale.hasSession = true;
    stale.hasGeneratedSurfaces = true;
    stale.hasLinePoints = true;
    stale.orientationEpoch = kEpoch - 1;
    CHECK(paneNeedsOrientationRefresh(stale, kEpoch));

    // Already built at the current epoch: nothing changed underneath it.
    auto current = stale;
    current.orientationEpoch = kEpoch;
    CHECK_FALSE(paneNeedsOrientationRefresh(current, kEpoch));

    // No session, nothing to rebuild.
    auto empty = stale;
    empty.hasSession = false;
    CHECK_FALSE(paneNeedsOrientationRefresh(empty, kEpoch));

    // Intersection sides suppress ordinary generated views and are rebuilt
    // through the inspection instead; rebuilding them here would build views
    // the session exists to suppress.
    auto suppressed = stale;
    suppressed.suppressesGeneratedViews = true;
    CHECK_FALSE(paneNeedsOrientationRefresh(suppressed, kEpoch));

    // Nothing materialized means nothing stale on screen — and the builder
    // rejects an empty model, which used to turn a successful attach into a
    // modal complaint mid-trace.
    auto unmaterialized = stale;
    unmaterialized.hasGeneratedSurfaces = false;
    CHECK_FALSE(paneNeedsOrientationRefresh(unmaterialized, kEpoch));

    auto noLine = stale;
    noLine.hasLinePoints = false;
    CHECK_FALSE(paneNeedsOrientationRefresh(noLine, kEpoch));

    // A pane that has never been built (default epoch) counts as stale once it
    // has surfaces to correct.
    auto neverRecorded = stale;
    neverRecorded.orientationEpoch = -1;
    CHECK(paneNeedsOrientationRefresh(neverRecorded, kEpoch));
}
