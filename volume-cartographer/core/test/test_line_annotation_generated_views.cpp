#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "CState.hpp"
#include "LineAnnotationFiberNaming.hpp"
#include "LineAnnotationShiftScroll.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "vc/lasagna/LineViewBuilder.hpp"

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

TEST_CASE("line annotation fiber naming uses username timestamp and sequence")
{
    CHECK(vc3d::line_annotation::normalizedFiberUsername("") == "anon");
    CHECK(vc3d::line_annotation::normalizedFiberUsername("  alice  ") == "alice");
    CHECK(vc3d::line_annotation::normalizedFiberUsername("A User/Name") == "A_User_Name");
    CHECK(vc3d::line_annotation::fiberFileName("alice", "20260605T123456789", 42) ==
          "alice_20260605T123456789_000042.json");
}
