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
#include "vc/fiber_tracer/FiberDisplay.hpp"
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

TEST_CASE("Strip context target: outer quarters of a span are its control points, the middle half the span")
{
    using namespace vc3d::line_annotation;
    using Kind = GeneratedStripContextTarget::Kind;
    const auto cp = [](size_t rank) { return GeneratedStripContextTarget{Kind::ControlPoint, rank}; };
    const auto span = [](size_t rank) { return GeneratedStripContextTarget{Kind::Span, rank}; };

    // Three controls at strip grid columns 100, 200, 400: spans of length 100 and 200.
    const std::vector<double> xs{100.0, 200.0, 400.0};
    CHECK(generatedStripContextTarget(xs, 100.0) == cp(0));
    CHECK(generatedStripContextTarget(xs, 124.0) == cp(0));
    CHECK(generatedStripContextTarget(xs, 126.0) == span(0));
    CHECK(generatedStripContextTarget(xs, 150.0) == span(0));
    CHECK(generatedStripContextTarget(xs, 174.0) == span(0));
    CHECK(generatedStripContextTarget(xs, 176.0) == cp(1));
    CHECK(generatedStripContextTarget(xs, 200.0) == cp(1));
    // The longer span: its quarters are 50 wide.
    CHECK(generatedStripContextTarget(xs, 249.0) == cp(1));
    CHECK(generatedStripContextTarget(xs, 251.0) == span(1));
    CHECK(generatedStripContextTarget(xs, 349.0) == span(1));
    CHECK(generatedStripContextTarget(xs, 351.0) == cp(2));
    CHECK(generatedStripContextTarget(xs, 400.0) == cp(2));
    // Exactly on the quarter line the span wins.
    CHECK(generatedStripContextTarget(xs, 125.0) == span(0));
    CHECK(generatedStripContextTarget(xs, 175.0) == span(0));

    // Beyond the ends: the end control, however far.
    CHECK(generatedStripContextTarget(xs, 0.0) == cp(0));
    CHECK(generatedStripContextTarget(xs, 99.0) == cp(0));
    CHECK(generatedStripContextTarget(xs, 401.0) == cp(2));
    CHECK(generatedStripContextTarget(xs, 5000.0) == cp(2));

    // A single control is always the target; nothing yields nothing.
    CHECK(generatedStripContextTarget({250.0}, 10.0) == cp(0));
    CHECK_FALSE(generatedStripContextTarget({}, 150.0));
    CHECK_FALSE(generatedStripContextTarget(xs, NAN));

    // Two controls on one strip column (duplicate points map to one column):
    // the zero-length span claims nothing, the span beside them keeps its
    // middle half, and on the shared column and in the quarter zone next to
    // it the first of them wins.
    const std::vector<double> coincident{100.0, 100.0, 300.0};
    CHECK(generatedStripContextTarget(coincident, 100.0) == cp(0));
    CHECK(generatedStripContextTarget(coincident, 120.0) == cp(0));
    CHECK(generatedStripContextTarget(coincident, 150.0) == span(1));
    CHECK(generatedStripContextTarget(coincident, 200.0) == span(1));
    CHECK(generatedStripContextTarget(coincident, 290.0) == cp(2));
    CHECK(generatedStripContextTarget({50.0, 100.0, 100.0, 300.0}, 100.0) == cp(1));
    CHECK(generatedStripContextTarget({50.0, 100.0, 100.0, 300.0}, 200.0) == span(2));
    CHECK(generatedStripContextTarget({100.0, 300.0, 300.0}, 300.0) == cp(1));
    CHECK(generatedStripContextTarget({100.0, 300.0, 300.0}, 900.0) == cp(1));
    CHECK(generatedStripContextTarget({100.0, 100.0, 100.0}, 100.0) == cp(0));
    CHECK(generatedStripContextTarget({100.0, 100.0, 100.0}, 7.0) == cp(0));
}

TEST_CASE("Strip context index: controls in line order with nondecreasing centre-line columns")
{
    using namespace vc3d::line_annotation;
    std::vector<GeneratedOverlay::ControlPointMarker> controls(5);
    // Out of line order on purpose; one without a control index, one off the line.
    controls[0].linePosition = 30.0; controls[0].controlIndex = 0;
    controls[1].linePosition = 10.0; controls[1].controlIndex = 1;
    controls[2].linePosition = 20.0; controls[2].controlIndex = std::numeric_limits<size_t>::max();
    controls[3].linePosition = 99.0; controls[3].controlIndex = 3;   // beyond the line
    controls[4].linePosition = 20.0; controls[4].controlIndex = 4;
    vc::lasagna::LineStripPositionMap noMap;
    const auto index = buildGeneratedStripContextIndex(controls, 40, noMap);
    REQUIRE(index.controlIndices.size() == 3);
    CHECK(index.controlIndices == std::vector<size_t>{1, 4, 0});
    CHECK(index.gridColumns == std::vector<double>{10.0, 20.0, 30.0});
    CHECK_FALSE(index.empty());
    CHECK(buildGeneratedStripContextIndex({}, 40, noMap).empty());
}

TEST_CASE("Overview anchors: settled layout is kept while the geometry is in flight")
{
    using namespace vc3d::line_annotation;
    auto control = [](float id, double linePosition, double arc = NAN, double total = NAN) {
        GeneratedOverlay::ControlPointMarker m;
        m.point = {id, 0.0f, 0.0f};
        m.linePosition = linePosition;
        m.arcLength = arc;
        m.lineArcLength = total;
        return m;
    };
    // A straight line along x: `count` points with `step` spacing.
    auto line = [](size_t count, float step) {
        std::vector<cv::Vec3f> points;
        for (size_t i = 0; i < count; ++i) {
            points.push_back({static_cast<float>(i) * step, 0.0f, 0.0f});
        }
        return points;
    };

    // Settled: 101 unit samples (arc length 100), controls at 0, 50, 100.
    const auto settledLine = line(101, 1.0f);
    const std::vector<GeneratedOverlay::ControlPointMarker> settledControls{
        control(1.0f, 0.0), control(2.0f, 50.0), control(3.0f, 100.0)};
    const auto settled = generatedOverviewSettledLayout(settledControls, settledLine);
    REQUIRE(settled.anchors.size() == 3);
    CHECK(settled.totalArcLength == doctest::Approx(100.0));
    CHECK(settled.anchors[1].arcLength == doctest::Approx(50.0));
    CHECK(settled.anchors[0].fraction == doctest::Approx(0.0));
    CHECK(settled.anchors[1].fraction == doctest::Approx(0.5));
    CHECK(settled.anchors[2].fraction == doctest::Approx(1.0));
    // Identity mapping when settled, both ways, by arc length.
    CHECK(generatedOverviewFraction(settled.anchors, 25.0, 100.0) == doctest::Approx(0.25));
    CHECK(generatedOverviewArcLength(settled.anchors, 0.25, 100.0) == doctest::Approx(25.0));

    // In flight, all controls in the displayed line's arc space: the old ones
    // keep their fractions, a new one at arc 20 of the 0..50 span lands at 0.2.
    const std::vector<GeneratedOverlay::ControlPointMarker> liveControls{
        control(1.0f, 0.0, 0.0, 100.0), control(9.0f, 20.0, 20.0, 100.0),
        control(2.0f, 50.0, 50.0, 100.0), control(3.0f, 100.0, 100.0, 100.0)};
    const auto frozen = generatedOverviewFrozenAnchors(settled.anchors, liveControls, settledLine);
    REQUIRE(frozen.size() == 4);
    CHECK(frozen[0].fraction == doctest::Approx(0.0));
    CHECK(frozen[1].fraction == doctest::Approx(0.2));
    CHECK(frozen[2].fraction == doctest::Approx(0.5));
    CHECK(frozen[3].fraction == doctest::Approx(1.0));
    // Arc lengths come from the controls when present, else from the line.
    const std::vector<GeneratedOverlay::ControlPointMarker> noArcs{
        control(1.0f, 0.0), control(9.0f, 20.0), control(2.0f, 50.0), control(3.0f, 100.0)};
    CHECK(generatedOverviewFrozenAnchors(settled.anchors, noArcs, settledLine)[1].fraction ==
          doctest::Approx(0.2));

    // Known fractions include controls placed earlier in the same flight:
    // they keep theirs even when a later publish moves their arc length.
    std::vector<GeneratedOverviewAnchor> known = settled.anchors;
    known.push_back(frozen[1]);  // the new control at 0.2
    const std::vector<GeneratedOverlay::ControlPointMarker> later{
        control(1.0f, 0.0, 0.0, 110.0), control(9.0f, 30.0, 30.0, 110.0),
        control(2.0f, 60.0, 60.0, 110.0), control(3.0f, 110.0, 110.0, 110.0)};
    const auto kept = generatedOverviewFrozenAnchors(known, later, settledLine);
    CHECK(kept[1].fraction == doctest::Approx(0.2));

    // Extending at the end: a settled line with controls at 0 and 50 of 100
    // (a 50-long tail). A point at arc 60 (20% into the tail) lands at
    // 0.5 + 0.2 * 0.5 = 0.6, a point before the first control toward the start.
    const auto settledTwo = generatedOverviewSettledLayout(
        {control(1.0f, 0.0), control(2.0f, 50.0)}, settledLine);
    const std::vector<GeneratedOverlay::ControlPointMarker> extended{
        control(1.0f, 0.0, 0.0, 100.0), control(2.0f, 50.0, 50.0, 100.0), control(8.0f, 60.0, 60.0, 100.0)};
    const auto frozenExtended = generatedOverviewFrozenAnchors(settledTwo.anchors, extended, settledLine);
    REQUIRE(frozenExtended.size() == 3);
    CHECK(frozenExtended[2].fraction == doctest::Approx(0.6));
    const std::vector<GeneratedOverlay::ControlPointMarker> prepended{
        control(7.0f, 5.0, 5.0, 100.0), control(2.0f, 50.0, 50.0, 100.0)};
    const auto frozenPrepended = generatedOverviewFrozenAnchors(settledTwo.anchors, prepended, settledLine);
    CHECK(frozenPrepended[0].fraction == doctest::Approx(0.05));
    for (size_t i = 1; i < frozenExtended.size(); ++i) {
        CHECK(frozenExtended[i].fraction >= frozenExtended[i - 1].fraction);
    }

    // Replacing a control in place: the old anchor B (arc 60, fraction .5)
    // is gone with its point; the replacement at the same displayed arc
    // takes its first fraction from the marker's mapping (A/B/C still there),
    // .5, not from interpolation between the surviving dots (.6).
    {
        std::vector<GeneratedOverviewAnchor> mapping = {
            {0, {1.0f, 0.0f, 0.0f}, 0.0, 0.0, 0.0},
            {0, {2.0f, 0.0f, 0.0f}, 60.0, 60.0, 0.5},
            {0, {3.0f, 0.0f, 0.0f}, 100.0, 100.0, 1.0}};
        const std::vector<GeneratedOverlay::ControlPointMarker> replaced{
            control(1.0f, 0.0, 0.0, 100.0), control(21.0f, 60.0, 60.0, 100.0), control(3.0f, 100.0, 100.0, 100.0)};
        const auto withMapping = generatedOverviewFrozenAnchors(mapping, replaced, settledLine, &mapping);
        CHECK(withMapping[1].fraction == doctest::Approx(0.5));
        const auto withoutMapping = generatedOverviewFrozenAnchors(mapping, replaced, settledLine);
        CHECK(withoutMapping[1].fraction == doctest::Approx(0.6));
    }

    // Two controls at (almost) one point of a line that returns on itself,
    // at arcs 40 and 160 of 200: identities tell them apart whatever their
    // arcs do between publishes (here an edit upstream moved both by +110,
    // which by arc alone would swap them); each keeps its own fraction.
    {
        std::vector<GeneratedOverviewAnchor> twin = {
            {11, {40.0f, 0.0f, 0.0f}, 40.0, 40.0, 0.2},
            {12, {40.0f, 0.0005f, 0.0f}, 160.0, 160.0, 0.8}};
        auto first = control(0.0f, 150.0, 150.0, 310.0);
        first.point = {40.0f, 0.0f, 0.0f};
        first.identity = 11;
        auto second = control(0.0f, 270.0, 270.0, 310.0);
        second.point = {40.0f, 0.0005f, 0.0f};
        second.identity = 12;
        const auto twinFrozen = generatedOverviewFrozenAnchors(twin, {first, second}, settledLine);
        CHECK(twinFrozen[0].fraction == doctest::Approx(0.2));
        CHECK(twinFrozen[1].fraction == doctest::Approx(0.8));
        // Without identities the point stands in: first unused match, in order.
        first.identity = 0;
        second.identity = 0;
        std::vector<GeneratedOverviewAnchor> twinNoId = twin;
        twinNoId[0].identity = 0;
        twinNoId[1].identity = 0;
        const auto byPoint = generatedOverviewFrozenAnchors(twinNoId, {first, second}, settledLine);
        CHECK(byPoint[0].fraction == doctest::Approx(0.2));
        CHECK(byPoint[1].fraction == doctest::Approx(0.8));
    }
    // Unfit anchors (non-finite) are skipped by the mappings, not dereferenced.
    {
        std::vector<GeneratedOverviewAnchor> unfit = {{0, {0, 0, 0}, 0.0, NAN, 0.3}};
        CHECK(generatedOverviewFraction(unfit, 50.0, 100.0) == doctest::Approx(0.5));
        CHECK(generatedOverviewArcLength(unfit, 0.5, 100.0) == doctest::Approx(50.0));
        CHECK(generatedOverviewFraction(unfit, 50.0, NAN) == 0.0);
        const std::vector<cv::Vec3f> huge{{-3.0e38f, 0, 0}, {3.0e38f, 0, 0}};
        const auto cumulativeHuge = generatedCumulativeArcLength(huge);
        CHECK(std::isfinite(cumulativeHuge[1]));
    }
    // Malformed (unsorted, or NaN-masked) cumulative arcs are refused, not searched.
    CHECK(generatedLinePositionAtArcLength({0.0, 20.0, 10.0, 30.0}, 15.0) == 0.0);
    CHECK(generatedLinePositionAtArcLength({0.0, 20.0, NAN, 10.0, 30.0}, 15.0) == 0.0);
    // Duplicate arc lengths (repeated points) invert to the FIRST position.
    CHECK(generatedLinePositionAtArcLength({0.0, 10.0, 10.0, 20.0}, 10.0) == doctest::Approx(1.0));
    CHECK(generatedLinePositionAtArcLength({0.0, 10.0, 10.0}, 10.0) == doctest::Approx(1.0));
    CHECK(generatedLinePositionAtArcLength({0.0, 10.0, 10.0, 20.0}, 15.0) == doctest::Approx(2.5));

    // Inconsistent metadata is clamped into the layout's invariants:
    // decreasing arcs become nondecreasing, a zero total gives zero arcs.
    {
        const auto decreasing = generatedOverviewSettledLayout(
            {control(1.0f, 0.0, 80.0, 100.0), control(2.0f, 10.0, 20.0, 100.0)}, settledLine);
        CHECK(decreasing.anchors[1].arcLength >= decreasing.anchors[0].arcLength);
        CHECK(decreasing.anchors[1].fraction >= decreasing.anchors[0].fraction);
        const auto zeroTotal = generatedOverviewSettledLayout({control(1.0f, 0.0, 10.0, 0.0)}, settledLine);
        CHECK(zeroTotal.anchors[0].arcLength == doctest::Approx(0.0));
        CHECK(generatedOverviewArcLength(zeroTotal.anchors, 0.5, zeroTotal.totalArcLength) == doctest::Approx(0.0));
        const auto negative = generatedOverviewSettledLayout({control(1.0f, 0.0, 10.0, -1.0)}, settledLine);
        CHECK(negative.totalArcLength == doctest::Approx(0.0));
        const auto frozenZero = generatedOverviewFrozenAnchors({}, {control(1.0f, 0.0, 10.0, 0.0)}, settledLine);
        CHECK(frozenZero[0].arcLength == doctest::Approx(0.0));
    }
    // An anonymous anchor at the same point never shadows an identified control's own anchor.
    {
        std::vector<GeneratedOverviewAnchor> mixed = {{0, {5.0f, 0.0f, 0.0f}, 10.0, 10.0, 0.1},
                                                      {12, {5.0f, 0.0f, 0.0f}, 80.0, 80.0, 0.8}};
        auto identified = control(0.0f, 80.0, 80.0, 100.0);
        identified.point = {5.0f, 0.0f, 0.0f};
        identified.identity = 12;
        CHECK(generatedOverviewFrozenAnchors(mixed, {identified}, settledLine)[0].fraction == doctest::Approx(0.8));
    }

    // Nothing known: arc-length fractions.
    const auto none = generatedOverviewFrozenAnchors({}, {control(40.0f, 50.0, 50.0, 100.0)}, settledLine);
    REQUIRE(none.size() == 1);
    CHECK(none[0].fraction == doctest::Approx(0.5));

    // Mapping by arc length agrees with the dots on a NON-uniformly sampled
    // line: samples at arc 0, 40, 80, 100 (positions 0..3), controls at the
    // ends; position 2 (arc 80) is at 0.8 both as a dot and as the marker.
    const std::vector<cv::Vec3f> uneven{{0, 0, 0}, {40, 0, 0}, {80, 0, 0}, {100, 0, 0}};
    const auto unevenLayout = generatedOverviewSettledLayout(
        {control(1.0f, 0.0), control(3.0f, 3.0)}, uneven);
    const auto cumulative = generatedCumulativeArcLength(uneven);
    CHECK(generatedOverviewFraction(unevenLayout.anchors, generatedArcLengthAt(cumulative, 2.0), 100.0) ==
          doctest::Approx(0.8));
    const auto unevenDots = generatedOverviewFrozenAnchors(
        unevenLayout.anchors, {control(1.0f, 0.0, 0.0, 100.0), control(9.0f, 2.0, 80.0, 100.0),
                               control(3.0f, 3.0, 100.0, 100.0)}, uneven);
    CHECK(unevenDots[1].fraction == doctest::Approx(0.8));
    // ... and back: 0.8 of the bar is arc 80 is position 2.
    CHECK(generatedLinePositionAtArcLength(
              cumulative, generatedOverviewArcLength(unevenLayout.anchors, 0.8, 100.0)) ==
          doctest::Approx(2.0));
    CHECK(generatedLinePositionAtArcLength(cumulative, 60.0) == doctest::Approx(1.5));

    // No anchors: plain arc-length fraction both ways; tails stretch.
    CHECK(generatedOverviewFraction({}, 25.0, 100.0) == doctest::Approx(0.25));
    CHECK(generatedOverviewArcLength({}, 0.25, 100.0) == doctest::Approx(25.0));
    CHECK(generatedOverviewFraction(frozen, 100.5, 100.0) == doctest::Approx(1.0));
    CHECK(generatedOverviewArcLength(frozen, 1.0, 100.0) == doctest::Approx(100.0));
    CHECK(generatedOverviewFraction(frozen, NAN, 100.0) == 0.0);

    // Arc length helpers: non-finite points add nothing, fractional positions interpolate.
    std::vector<cv::Vec3f> broken = line(4, 2.0f);
    broken[2] = {NAN, NAN, NAN};
    CHECK(generatedCumulativeArcLength(broken) == std::vector<double>{0.0, 2.0, 2.0, 2.0});
    CHECK(generatedArcLengthAt(generatedCumulativeArcLength(line(3, 2.0f)), 0.5) == doctest::Approx(1.0));
}

TEST_CASE("Display-space arc lengths: published controls are re-expressed on the displayed line")
{
    using namespace vc3d::line_annotation;
    auto pendingAt = [](cv::Vec3f point, double arc, double position, uint64_t revision) {
        GeneratedPendingPlacement p;
        p.point = point;
        p.anchor = point;
        p.token = 1;
        p.arcLength = arc;
        p.linePosition = position;
        p.lineRevision = revision;
        return p;
    };
    auto control = [](float id, double linePosition, double arc, double total, uint64_t revision = 7) {
        GeneratedOverlay::ControlPointMarker m;
        m.point = {id, 0.0f, 0.0f};
        m.identity = static_cast<uint64_t>(id);
        m.linePosition = linePosition;
        m.arcLength = arc;
        m.lineArcLength = total;
        m.lineRevision = revision;
        return m;
    };
    // Displayed line (revision 6): 100 long, controls at 0, 60, 100 (identities 1, 2, 3).
    GeneratedOverviewLayout displayed;
    displayed.totalArcLength = 100.0;
    displayed.anchors = {{1, {1.0f, 0.0f, 0.0f}, 0.0, 0.0, 0.0},
                         {2, {2.0f, 0.0f, 0.0f}, 60.0, 60.0, 0.6},
                         {3, {3.0f, 0.0f, 0.0f}, 100.0, 100.0, 1.0}};
    // The user placed a point at displayed arc 45. The controller's splice
    // through it (revision 7) is shorter: its arcs read 40 and 50 where the
    // display has 45 and 60; the line's total became 90.
    std::vector<GeneratedPendingPlacement> pending{pendingAt({9.0f, 0.0f, 0.0f}, 45.0, 45.0, 6)};
    std::vector<GeneratedPendingPlacement> resolved;
    const std::vector<GeneratedOverlay::ControlPointMarker> published{
        control(1.0f, 0.0, 0.0, 90.0), control(9.0f, 4.0, 40.0, 90.0),
        control(2.0f, 5.0, 50.0, 90.0), control(3.0f, 15.0, 90.0, 90.0)};
    const auto display = generatedDisplaySpaceControlArcLengths(displayed, published, pending, resolved, 6);
    REQUIRE(display.size() == 4);
    CHECK(display[0].arcLength == doctest::Approx(0.0));
    CHECK(display[1].arcLength == doctest::Approx(45.0));   // the recorded click, exactly
    CHECK(display[2].arcLength == doctest::Approx(60.0));   // the displayed control, unmoved
    CHECK(display[3].arcLength == doctest::Approx(100.0));
    for (const auto& c : display) {
        CHECK(c.lineArcLength == doctest::Approx(100.0));
        CHECK(c.lineRevision == 6);                          // reads as the displayed line's now
    }
    CHECK(pending.empty());                                   // consumed ...
    REQUIRE(resolved.size() == 1);                            // ... into the resolved list
    CHECK(resolved[0].arcLength == doctest::Approx(45.0));
    // An identical later publish (a link-state refresh, say) draws it in the
    // same spot, from the resolved list, with nothing left to consume.
    const auto again = generatedDisplaySpaceControlArcLengths(displayed, published, pending, resolved, 6);
    CHECK(again[1].arcLength == doctest::Approx(45.0));
    CHECK(resolved.size() == 1);

    // A placement recorded for another displayed revision is ignored; the
    // control is then interpolated by the ratio of its live arcs: 40/50 of
    // 0..60 = 48, and that estimate is kept for later publishes too.
    std::vector<GeneratedPendingPlacement> stale{pendingAt({9.0f, 0.0f, 0.0f}, 45.0, 45.0, 5)};
    std::vector<GeneratedPendingPlacement> estimates;
    const auto interpolated = generatedDisplaySpaceControlArcLengths(displayed, published, stale, estimates, 6);
    CHECK(interpolated[1].arcLength == doctest::Approx(48.0));
    CHECK(stale.size() == 1);
    REQUIRE(estimates.size() == 1);
    CHECK(estimates[0].arcLength == doctest::Approx(48.0));
    const std::vector<GeneratedOverlay::ControlPointMarker> moved{
        control(1.0f, 0.0, 0.0, 95.0), control(9.0f, 4.0, 44.0, 95.0),
        control(2.0f, 5.0, 55.0, 95.0), control(3.0f, 15.0, 95.0, 95.0)};
    CHECK(generatedDisplaySpaceControlArcLengths(displayed, moved, stale, estimates, 6)[1].arcLength ==
          doctest::Approx(48.0));

    // Two new controls close together (0.4 apart) with two placements: each
    // takes its nearest, one placement per control.
    std::vector<GeneratedPendingPlacement> two{pendingAt({9.0f, 0.0f, 0.0f}, 30.0, 30.0, 6), pendingAt({9.4f, 0.0f, 0.0f}, 80.0, 80.0, 6)};
    std::vector<GeneratedPendingPlacement> twoResolved;
    const std::vector<GeneratedOverlay::ControlPointMarker> pair{
        control(1.0f, 0.0, 0.0, 90.0), control(9.0f, 3.0, 25.0, 90.0),
        control(2.0f, 5.0, 50.0, 90.0), control(9.4f, 8.0, 75.0, 90.0), control(3.0f, 15.0, 90.0, 90.0)};
    const auto both = generatedDisplaySpaceControlArcLengths(displayed, pair, two, twoResolved, 6);
    CHECK(both[1].arcLength == doctest::Approx(30.0));
    CHECK(both[3].arcLength == doctest::Approx(80.0));
    CHECK(two.empty());
    CHECK(twoResolved.size() == 2);
    // Publishing only the second control: it takes ITS placement (the
    // nearest point), not the first one recorded along the line.
    {
        std::vector<GeneratedPendingPlacement> twoAgain{
            pendingAt({9.0f, 0.0f, 0.0f}, 30.0, 30.0, 6), pendingAt({9.4f, 0.0f, 0.0f}, 80.0, 80.0, 6)};
        std::vector<GeneratedPendingPlacement> scratchAgain;
        const auto onlySecond = generatedDisplaySpaceControlArcLengths(
            displayed, {control(1.0f, 0.0, 0.0, 90.0), control(2.0f, 5.0, 50.0, 90.0),
                        control(9.4f, 8.0, 75.0, 90.0), control(3.0f, 15.0, 90.0, 90.0)},
            twoAgain, scratchAgain, 6);
        CHECK(onlySecond[2].arcLength == doctest::Approx(80.0));
        CHECK(twoAgain.size() == 1);
    }
    // Two new controls at ONE point (within tolerance) of a returning line,
    // placements recorded at arcs 40 and 160: paired in line order with the
    // placements in arc order, one each; a repeated publish keeps both.
    {
        std::vector<GeneratedPendingPlacement> twinPending{
            pendingAt({50.0f, 0.0f, 0.0f}, 40.0, 40.0, 6), pendingAt({50.0f, 0.0005f, 0.0f}, 160.0, 160.0, 6)};
        std::vector<GeneratedPendingPlacement> twinResolved;
        GeneratedOverviewLayout longDisplayed;
        longDisplayed.totalArcLength = 200.0;
        longDisplayed.anchors = {{1, {1.0f, 0.0f, 0.0f}, 0.0, 0.0, 0.0},
                                 {2, {2.0f, 0.0f, 0.0f}, 100.0, 100.0, 0.5},
                                 {3, {3.0f, 0.0f, 0.0f}, 200.0, 200.0, 1.0}};
        auto twinA = control(31.0f, 2.0, 35.0, 190.0);
        twinA.point = {50.0f, 0.0f, 0.0f};
        auto twinB = control(32.0f, 8.0, 150.0, 190.0);
        twinB.point = {50.0f, 0.0005f, 0.0f};
        const std::vector<GeneratedOverlay::ControlPointMarker> twins{
            control(1.0f, 0.0, 0.0, 190.0), twinA, control(2.0f, 5.0, 95.0, 190.0), twinB,
            control(3.0f, 15.0, 190.0, 190.0)};
        const auto firstPublish = generatedDisplaySpaceControlArcLengths(longDisplayed, twins, twinPending, twinResolved, 6);
        CHECK(firstPublish[1].arcLength == doctest::Approx(40.0));
        CHECK(firstPublish[3].arcLength == doctest::Approx(160.0));
        CHECK(twinPending.empty());
        const auto secondPublish = generatedDisplaySpaceControlArcLengths(longDisplayed, twins, twinPending, twinResolved, 6);
        CHECK(secondPublish[1].arcLength == doctest::Approx(40.0));
        CHECK(secondPublish[3].arcLength == doctest::Approx(160.0));
        CHECK(twinResolved.size() == 2);
        CHECK(twinResolved[0].identity == 31);
        CHECK(twinResolved[1].identity == 32);
        // Without identities (identity 0) the resolved fallback by point is
        // still one entry per control on the repeated publish.
        auto twinA0 = twinA; twinA0.identity = 0;
        auto twinB0 = twinB; twinB0.identity = 0;
        std::vector<GeneratedPendingPlacement> noIdPending{
            pendingAt({50.0f, 0.0f, 0.0f}, 40.0, 40.0, 6), pendingAt({50.0f, 0.0005f, 0.0f}, 160.0, 160.0, 6)};
        std::vector<GeneratedPendingPlacement> noIdResolved;
        const std::vector<GeneratedOverlay::ControlPointMarker> twins0{
            control(1.0f, 0.0, 0.0, 190.0), twinA0, control(2.0f, 5.0, 95.0, 190.0), twinB0,
            control(3.0f, 15.0, 190.0, 190.0)};
        (void)generatedDisplaySpaceControlArcLengths(longDisplayed, twins0, noIdPending, noIdResolved, 6);
        const auto repeat0 = generatedDisplaySpaceControlArcLengths(longDisplayed, twins0, noIdPending, noIdResolved, 6);
        CHECK(repeat0[1].arcLength == doctest::Approx(40.0));
        CHECK(repeat0[3].arcLength == doctest::Approx(160.0));
        // An anonymous resolved entry at the same point never shadows an
        // identified control's own resolved entry (and its provenance).
        std::vector<GeneratedPendingPlacement> mixedResolved(2);
        mixedResolved[0].point = {50.0f, 0.0f, 0.0f};
        mixedResolved[0].arcLength = 10.0;
        mixedResolved[0].lineRevision = 6;
        mixedResolved[1].point = {50.0f, 0.0f, 0.0f};
        mixedResolved[1].arcLength = 80.0;
        mixedResolved[1].lineRevision = 6;
        mixedResolved[1].identity = 31;
        mixedResolved[1].fromPlacement = true;
        std::vector<GeneratedPendingPlacement> noneLeft;
        const auto mixed = generatedDisplaySpaceControlArcLengths(
            longDisplayed, {control(1.0f, 0.0, 0.0, 190.0), twinA, control(3.0f, 15.0, 190.0, 190.0)},
            noneLeft, mixedResolved, 6);
        CHECK(mixed[1].arcLength == doctest::Approx(80.0));
        CHECK(mixed[1].onLine);
    }

    // Past the displayed end: clamped to it; before the start: toward 0.
    std::vector<GeneratedPendingPlacement> noPending, scratch;
    const auto ext = generatedDisplaySpaceControlArcLengths(
        displayed, {control(3.0f, 0.0, 100.0, 140.0), control(8.0f, 1.0, 110.0, 140.0)}, noPending, scratch, 6);
    CHECK(ext[1].arcLength == doctest::Approx(100.0));
    const auto pre = generatedDisplaySpaceControlArcLengths(
        displayed, {control(7.0f, 0.0, 10.0, 100.0), control(1.0f, 1.0, 20.0, 100.0),
                    control(2.0f, 2.0, 80.0, 100.0)}, noPending, scratch, 6);
    CHECK(pre[0].arcLength == doctest::Approx(0.0));
    CHECK(pre[1].arcLength == doctest::Approx(0.0));
    CHECK(pre[2].arcLength == doctest::Approx(60.0));

    // No displayed layout: controls are returned unchanged.
    const auto same = generatedDisplaySpaceControlArcLengths({}, published, noPending, scratch, 6);
    CHECK(same[1].arcLength == doctest::Approx(40.0));
    CHECK(same[1].lineRevision == 7);

    // A control this dialog placed (matched to its pending placement) is
    // drawn on the displayed centre line at its placed-at arc, even when the
    // click was across the strip (the solve pulls the line through it); a
    // provisional control from elsewhere is judged against the displayed line.
    std::vector<cv::Vec3f> displayedLine;
    for (int i = 0; i <= 100; ++i) {
        displayedLine.push_back({static_cast<float>(i), 0.0f, 0.0f});
    }
    auto offLine = control(9.0f, 4.0, 40.0, 90.0);
    offLine.point = {45.0f, 10.0f, 0.0f};
    offLine.onLine = true;
    auto onLine = control(19.0f, 6.0, 55.0, 90.0);
    onLine.point = {70.0f, 0.0f, 0.0f};
    onLine.onLine = true;
    std::vector<GeneratedPendingPlacement> twoMore{pendingAt({45.0f, 10.0f, 0.0f}, 45.0, 45.0, 6), pendingAt({70.0f, 0.0f, 0.0f}, 70.0, 70.0, 6)};
    std::vector<GeneratedPendingPlacement> scratch2;
    const auto judged = generatedDisplaySpaceControlArcLengths(
        displayed, {control(1.0f, 0.0, 0.0, 90.0), offLine, control(2.0f, 5.0, 50.0, 90.0), onLine,
                    control(3.0f, 15.0, 90.0, 90.0)},
        twoMore, scratch2, 6, displayedLine);
    CHECK(judged[1].arcLength == doctest::Approx(45.0));
    CHECK(judged[1].onLine);
    CHECK(judged[3].arcLength == doctest::Approx(70.0));
    CHECK(judged[3].onLine);
    // ... and still on a repeated publish, from the resolved entry's provenance.
    const auto judgedAgain = generatedDisplaySpaceControlArcLengths(
        displayed, {control(1.0f, 0.0, 0.0, 90.0), offLine, control(2.0f, 5.0, 50.0, 90.0), onLine,
                    control(3.0f, 15.0, 90.0, 90.0)},
        twoMore, scratch2, 6, displayedLine);
    CHECK(judgedAgain[1].onLine);
    CHECK(judgedAgain[1].arcLength == doctest::Approx(45.0));
    // Provisional controls also learn their position on the displayed line,
    // which the span pieces (gap, damaged) are drawn between.
    CHECK(judged[1].displayedLinePosition == doctest::Approx(45.0));
    CHECK(judged[3].displayedLinePosition == doctest::Approx(70.0));
    {
        auto owner = judged[1];
        owner.hasGapToNext = true;
        const auto ranges = generatedGapLineRanges({judged[0], owner, judged[2]});
        REQUIRE(ranges.size() == 1);
        CHECK(ranges[0].first == doctest::Approx(45.0));   // displayed, not the provisional index 4
        CHECK(ranges[0].second == doctest::Approx(60.0));
    }
    // A provisional control from elsewhere (no placement of ours) that sits
    // across the strip is NOT on the displayed line; one on it is.
    {
        auto acrossStrip = control(29.0f, 7.0, 60.0, 90.0);
        acrossStrip.point = {80.0f, 3.0f, 0.0f};
        acrossStrip.onLine = true;
        auto onIt = control(28.0f, 6.0, 55.0, 90.0);
        onIt.point = {65.0f, 0.0f, 0.0f};  // live 55 between live 50/90 -> displayed 60 + 0.125 * 40 = 65
        onIt.onLine = false;
        std::vector<GeneratedPendingPlacement> none2;
        std::vector<GeneratedPendingPlacement> scratch3;
        const auto judged2 = generatedDisplaySpaceControlArcLengths(
            displayed, {control(1.0f, 0.0, 0.0, 90.0), control(2.0f, 5.0, 50.0, 90.0), onIt, acrossStrip,
                        control(3.0f, 15.0, 90.0, 90.0)},
            none2, scratch3, 6, displayedLine);
        CHECK(judged2[2].onLine);
        CHECK_FALSE(judged2[3].onLine);
    }
    CHECK(judged[2].onLine == false);  // untouched: a displayed control keeps the producer's flag (false by default here)
}

TEST_CASE("Strip grid column for an arc length follows the map's per-column arc lengths")
{
    using namespace vc3d::line_annotation;
    vc::lasagna::LineStripPositionMap map;
    map.originalArclengths = {0.0, 10.0, 20.0, 30.0};
    map.stripGridArclengths = {0.0, 5.0, 10.0, 20.0, 30.0};  // uneven columns
    map.totalArclength = 30.0;
    map.stripGridSpacingBaseVoxels = 7.5;
    map.stripGridColumnCount = 5;
    REQUIRE(map.valid());
    CHECK(generatedStripGridColumnForArcLength(map, 0.0) == doctest::Approx(0.0));
    CHECK(generatedStripGridColumnForArcLength(map, 2.5) == doctest::Approx(0.5));
    CHECK(generatedStripGridColumnForArcLength(map, 10.0) == doctest::Approx(2.0));
    CHECK(generatedStripGridColumnForArcLength(map, 15.0) == doctest::Approx(2.5));
    CHECK(generatedStripGridColumnForArcLength(map, 30.0) == doctest::Approx(4.0));
    CHECK(generatedStripGridColumnForArcLength(map, -3.0) == doctest::Approx(0.0));
    CHECK(generatedStripGridColumnForArcLength(map, 99.0) == doctest::Approx(4.0));
    CHECK(std::isnan(generatedStripGridColumnForArcLength(map, NAN)));
    CHECK(std::isnan(generatedStripGridColumnForArcLength(vc::lasagna::LineStripPositionMap{}, 1.0)));
    // A malformed map (non-finite entry) is answered within bounds, never read past.
    vc::lasagna::LineStripPositionMap broken = map;
    broken.stripGridArclengths = {0.0, NAN};
    broken.stripGridColumnCount = 2;
    CHECK(std::isnan(generatedStripGridColumnForArcLength(broken, 5.0)));
    broken.stripGridArclengths = {0.0, 20.0, 10.0, 30.0};
    broken.stripGridColumnCount = 4;
    CHECK(std::isnan(generatedStripGridColumnForArcLength(broken, 15.0)));
    broken.stripGridArclengths = {0.0, 20.0, NAN, 10.0, 30.0};
    broken.stripGridColumnCount = 5;
    CHECK(std::isnan(generatedStripGridColumnForArcLength(broken, 15.0)));
    broken.stripGridArclengths = {0.0, std::numeric_limits<double>::infinity(), 30.0};
    broken.stripGridColumnCount = 3;
    CHECK(std::isnan(generatedStripGridColumnForArcLength(broken, 15.0)));
}

TEST_CASE("Strip context index: provisional controls are kept by their displayed position")
{
    using namespace vc3d::line_annotation;
    // Three displayed samples (positions 0..2); the controls' own indices
    // {0, 5, 10} belong to the controller's longer line, their displayed
    // positions {0, 1, 2} to the strip. All three stay in the index.
    std::vector<GeneratedOverlay::ControlPointMarker> controls(3);
    for (size_t i = 0; i < 3; ++i) {
        controls[i].controlIndex = i;
        controls[i].linePosition = 5.0 * static_cast<double>(i);
        controls[i].displayedLinePosition = static_cast<double>(i);
    }
    vc::lasagna::LineStripPositionMap noMap;
    const auto index = buildGeneratedStripContextIndex(controls, 3, noMap);
    REQUIRE(index.controlIndices.size() == 3);
    CHECK(index.gridColumns == std::vector<double>{0.0, 1.0, 2.0});
}

TEST_CASE("Clearing CP corrections leaves other controls and span metadata intact")
{
    using namespace vc3d::line_annotation;
    std::vector<LineControlPoint> controls(3);
    for (auto& cp : controls) {
        cp.direction = cv::Vec3d(1,0,0);
        cp.displayNormal = cv::Vec3d(0,1,0);
        cp.displayNormalSource = "manual";
        cp.segmentToNext.emplace();
        cp.segmentToNext->interpGoal = SegmentInterpolationGoal::Trace;
    }
    clearControlPointCorrections(controls[1]);
    CHECK_FALSE(controls[1].direction);
    CHECK_FALSE(controls[1].displayNormal);
    CHECK(controls[1].displayNormalSource == "unknown");
    CHECK(controls[0].direction.has_value());
    CHECK(controls[2].displayNormal.has_value());
    CHECK(controls[1].segmentToNext->interpGoal == SegmentInterpolationGoal::Trace);
    for (auto& cp : controls) clearControlPointCorrections(cp);
    for (const auto& cp : controls) {
        CHECK_FALSE(cp.direction);
        CHECK_FALSE(cp.displayNormal);
        CHECK(cp.segmentToNext->interpGoal == SegmentInterpolationGoal::Trace);
    }
}

TEST_CASE("Direction handles use a valid local frame even on two-column strips")
{
    cv::Mat_<cv::Vec3f> points(7,2);
    for (int y=0;y<7;++y)
        for (int x=0;x<2;++x) points(y,x)={float(x*10),float(y-3),0};
    QuadSurface surface(points,{1,1});
    const auto frame=vc3d::line_annotation::generatedStripFrame(
        &surface,surface.gridToSurface({0.5,3}));
    REQUIRE(frame);
    CHECK(cv::norm(frame->along-cv::Vec3d(1,0,0))<1e-6);
    CHECK(cv::norm(frame->across-cv::Vec3d(0,1,0))<1e-6);
    CHECK(cv::norm(frame->normal-cv::Vec3d(0,0,1))<1e-6);
    CHECK_FALSE(vc3d::line_annotation::generatedStripFrame(nullptr,{0,0}));
}

TEST_CASE("Arclength Hermite preserves samples and has continuous analytic tangents")
{
    const std::vector<cv::Vec3d> p{{0,0,0},{10,0,0},{10,20,0},{20,30,0}};
    for (size_t i=0;i<p.size();++i)
        CHECK(cv::norm(vc::geometry::sampleLine(p,double(i)).value-p[i]) < 1e-12);
    const auto mid=vc::geometry::sampleLine(p,0.5);
    // Python _arc_derivatives: d0=(1,0,0), d1=(1/3,2/3,0).
    CHECK(mid.value[0] == doctest::Approx(35.0/6));
    CHECK(mid.value[1] == doctest::Approx(-5.0/6));
    CHECK(cv::norm(vc::geometry::sampleLine(p,1-1e-7).derivative-
                   vc::geometry::sampleLine(p,1+1e-7).derivative) < 1e-6);
    const std::vector<cv::Vec3d> duplicates{{0,0,0},{10,0,0},{10,0,0},{20,0,0}};
    CHECK(vc::geometry::sampleLine(duplicates,1).derivative[0] == doctest::Approx(1));
}

TEST_CASE("CP displacement field clamps outer derivatives without changing control displacements")
{
    const std::vector<double> arcs{0,10,30};
    const std::vector<cv::Vec3d> values{{0,0,0},{0,5,0},{0,0,0}};
    const std::vector<bool> flat{true,false,true};
    for (size_t i=0;i<arcs.size();++i)
        CHECK(cv::norm(vc::geometry::sampleField(arcs,values,arcs[i],flat)-values[i]) < 1e-12);
    CHECK(cv::norm(vc::geometry::sampleField(arcs,values,1e-4,flat))/1e-4 < 1e-4);
    CHECK(cv::norm(vc::geometry::sampleField(arcs,values,30-1e-4,flat))/1e-4 < 1e-4);
    CHECK(vc::geometry::sampleField(arcs,values,-1,flat) == cv::Vec3d(0,0,0));
    CHECK(vc::geometry::sampleField(arcs,values,31,flat) == cv::Vec3d(0,0,0));
}

TEST_CASE("Display normal provenance survives storage replacement and reversal")
{
    using namespace vc3d::line_annotation;
    const nlohmann::json input{{"position",{1,2,3}}, {"display_normal",{0,1,0}},
                               {"display_normal_source","interpolated"}};
    const auto stored=storedControlPointFromJson(input,3);
    CHECK(stored.displayNormalSource == "interpolated");
    CHECK(storedControlPointToJson(stored).at("display_normal_source") == "interpolated");
    CHECK(reversedStoredControlPoints({stored}).front().displayNormalSource == "interpolated");
    LineControlPoint cp(0,{1,2,3},true,0);
    cp.displayNormal=stored.displayNormal;
    cp.displayNormalSource=stored.displayNormalSource;
    const auto replacement=collapseControlPointsAtClick({cp},{0},0,{2,2,3});
    CHECK(replacement.controlPoints.front().displayNormalSource == "interpolated");
    auto legacy=input; legacy.erase("display_normal_source");
    CHECK(storedControlPointFromJson(legacy,3).displayNormalSource == "unknown");
    auto manual=input; manual["display_normal_source"]="manual";
    CHECK(storedControlPointFromJson(manual,3).displayNormalSource == "manual");
    manual.erase("display_normal");
    CHECK_THROWS(storedControlPointFromJson(manual,3));
}

TEST_CASE("Cross section center drag translates without changing width or orientation")
{
    const auto result = vc::fiber_tracer::dragFiberWidth(
        {10, 20, 30}, {0, 1, 0}, {1, 0, 0}, {0, 0, 1}, 8, 0, {3, -2, 7});
    REQUIRE(result);
    CHECK(cv::norm(result->center - cv::Vec3d(13, 18, 30)) < 1e-9);
    CHECK(cv::norm(result->normal - cv::Vec3d(0, 1, 0)) < 1e-9);
    CHECK(result->width == 8);
    CHECK(vc::fiber_tracer::dragFiberWidth(
        {0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 1}, 0, 0, {1, 2, 0}).has_value());
}

TEST_CASE("Cross section edge drags preserve width and put the selected edge at the target")
{
    using namespace vc::fiber_tracer;
    for (const int handle : {-1, 1}) {
        for (const double scale : {1.0, 0.25}) {
            // Non-axis-aligned cross plane, like a rotated annotation view.
            const auto plane = *displayUnit({1, 2, 3});
            const auto up = *projectDisplayNormal({0, 1, 0}, plane);
            const auto axis = up.cross(plane);
            const cv::Vec3d center = cv::Vec3d(17, 29, 31) * scale;
            const double width = 8 * scale;
            const cv::Vec3d delta = (axis * 3 + up * 4) * scale;
            const auto result = dragFiberWidth(center, up, axis, plane, width, handle, delta);
            REQUIRE(result);
            const auto originalOpposite = center - axis * (handle * width / 2);
            const auto target = center + axis * (handle * width / 2) + delta;
            CHECK(cv::norm(result->edgeAxis * handle - *displayUnit(target - originalOpposite)) < 1e-9);
            CHECK(cv::norm(result->center - (target - result->edgeAxis * (handle * width / 2))) < 1e-9);
            CHECK(cv::norm((result->center + result->edgeAxis * (handle * result->width / 2)) -
                           (center + axis * (handle * width / 2) + delta)) < 1e-9);
            CHECK(std::abs(result->normal.dot(result->edgeAxis)) < 1e-9);
            CHECK(cv::norm(result->normal.cross(plane) - result->edgeAxis) < 1e-9);
            CHECK(result->width == width);
            // Both tolerance pairs follow the dragged frame without resizing
            // the nominal width, including fibers with a custom gap.
            for (const double gap : {0.2, 0.35}) {
                const auto offsets = fiberWidthEdgeOffsets(result->width, gap);
                for (int side : {-1, 1}) {
                    const size_t first = side < 0 ? 0 : 2;
                    const auto innerOuterMidpoint = result->center + result->edgeAxis *
                        ((offsets[first] + offsets[first + 1]) / 2);
                    CHECK(cv::norm(innerOuterMidpoint -
                        (result->center + result->edgeAxis * (side * width / 2))) < 1e-9);
                    CHECK(offsets[first + 1] - offsets[first] == doctest::Approx(gap * width));
                }
            }
        }
    }
    CHECK_FALSE(dragFiberWidth({0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 1}, 8, 1, {-8, 0, 0}));
    CHECK_FALSE(dragFiberWidth({0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {0, 0, 1}, 0, 1, {1, 0, 0}));
}

TEST_CASE("Successive cross section drags build on the preview in the unchanged plane")
{
    using namespace vc::fiber_tracer;
    const cv::Vec3d plane(0, 0, 1);
    const auto first = dragFiberWidth({0, 0, 0}, {0, 1, 0}, {1, 0, 0}, plane, 8, 1, {0, 4, 0});
    REQUIRE(first);
    // A center stroke must retain the orientation from the preceding edge stroke.
    const auto second = dragFiberWidth(first->center, first->normal, first->edgeAxis,
        plane, first->width, 0, {3, -2, 0});
    REQUIRE(second);
    CHECK(cv::norm(second->normal - first->normal) < 1e-9);
    CHECK(cv::norm(second->center - (first->center + cv::Vec3d(3, -2, 0))) < 1e-9);
    const auto third = dragFiberWidth(second->center, second->normal, second->edgeAxis,
        plane, second->width, -1, {-2, 1, 0});
    REQUIRE(third);
    CHECK(cv::norm(third->center - third->edgeAxis * 4 -
        (second->center - second->edgeAxis * 4 + cv::Vec3d(-2, 1, 0))) < 1e-9);
    CHECK(third->width == 8);
    CHECK(cv::norm(third->normal.cross(plane) - third->edgeAxis) < 1e-9);
}

TEST_CASE("New control points inherit the existing interpolated display correction")
{
    using namespace vc::fiber_tracer;
    using namespace vc3d::line_annotation;
    std::vector<cv::Vec3f> points, normals;
    for (int i = 0; i <= 30; ++i) {
        points.emplace_back(float(i * i), 0, 0);
        normals.emplace_back(0, 0, 1);
    }
    const std::vector<double> positions{0, 10, 20, 30};
    const std::vector<double> arcs{0, 100, 400, 900};
    const std::vector<std::optional<cv::Vec3d>> manual{
        cv::Vec3d(0, -1, 0), std::nullopt, std::nullopt, std::nullopt};
    const auto field = fiberDisplayField(points, normals, positions, manual);
    for (double position : {5.0, 5.5}) {
        const int i = int(position);
        const double arc = i * i + (position - i) * ((i + 1) * (i + 1) - i * i);
        const auto inherited = inheritedFiberDisplayNormal(field, arcs, arc, position);
        REQUIRE(inherited);
        CHECK(cv::norm(*inherited - *displayUnit(displayVectorAt(field.normals, position))) < 1e-9);
        std::vector<LineControlPoint> controls(positions.size());
        for (size_t k = 0; k < positions.size(); ++k) {
            controls[k].linePosition = positions[k];
            controls[k].volumePoint = displayVectorAt(points, positions[k]);
            controls[k].displayNormal = manual[k];
        }
        auto inserted = collapseControlPointsAtClick(controls, {}, position, displayVectorAt(points, position));
        inserted.controlPoints[inserted.replacementIndex].displayNormal = inherited;
        auto& created = inserted.controlPoints[inserted.replacementIndex];
        created.displayNormalSource = "interpolated";
        created.direction = editControlDirection({1,0,0}, {0,0,1}, {1,0.2,0});
        REQUIRE(created.direction);
        StoredControlPoint stored(created.volumePoint);
        stored.displayNormal = created.displayNormal;
        stored.displayNormalSource = created.displayNormalSource;
        stored.direction = created.direction;
        const auto restored = storedControlPointFromJson(storedControlPointToJson(stored), 3);
        REQUIRE(restored.direction);
        REQUIRE(restored.displayNormal);
        CHECK(cv::norm(*restored.direction - *created.direction) < 1e-9);
        CHECK(cv::norm(*restored.displayNormal - *inherited) < 1e-9);
        CHECK(restored.displayNormalSource == "interpolated");
        std::vector<double> newPositions;
        std::vector<std::optional<cv::Vec3d>> newNormals;
        for (const auto& cp : inserted.controlPoints) {
            newPositions.push_back(cp.linePosition);
            newNormals.push_back(cp.displayNormal);
        }
        const auto updated = fiberDisplayField(points, normals, newPositions, newNormals);
        CHECK(updated.controlOffsets[inserted.replacementIndex] == doctest::Approx(
            *displayNormalOffset({0, 0, 1}, *inherited, {1, 0, 0})));
        CHECK(std::abs(updated.controlOffsets[inserted.replacementIndex]) > 0.1);
    }
    // Uncorrected regions should remain unset and follow future Lasagna updates.
    CHECK_FALSE(inheritedFiberDisplayNormal(field, arcs, 225, 15));
    const auto baselineOnly = fiberDisplayField(points, normals, positions,
        std::vector<std::optional<cv::Vec3d>>(positions.size()));
    CHECK_FALSE(inheritedFiberDisplayNormal(baselineOnly, arcs, 25, 5));
}

TEST_CASE("Fiber display normals use the cut tangent and preserve unset controls")
{
    using namespace vc::fiber_tracer;
    std::vector<cv::Vec3f> points, normals;
    for (int i = 0; i <= 20; ++i) {
        points.emplace_back(float(i), 0, 0);
        normals.emplace_back(0, 0, 1);
    }
    const auto original = normals;
    const auto field = fiberDisplayField(points, normals, {0, 10, 20},
        {std::nullopt, cv::Vec3d(0, -1, 0), std::nullopt});
    CHECK(field.resetControls.empty());
    CHECK(field.controlOffsets[1] == doctest::Approx(std::acos(-1.0) / 2));
    CHECK(cv::norm(field.normals[0] - normals[0]) < 1e-6);
    CHECK(cv::norm(field.normals[20] - normals[20]) < 1e-6);
    CHECK(cv::norm(field.normals[10] - cv::Vec3f(0, -1, 0)) < 1e-6);
    CHECK(normals == original);
    const auto reset = fiberDisplayField(points, normals, {10}, {cv::Vec3d(1, 0, 0)});
    REQUIRE(reset.resetControls.size() == 1);
    CHECK(reset.resetControls[0] == 0);
    points[12][1] = 5;
    const auto tangent = displayTangentAt(points, 8.0);
    CHECK(cv::norm(tangent - cv::Vec3d(1, 0, 0)) < 1e-6); // Regular central chord.
    const auto moved = fiberDisplayField(points, normals, {8}, {cv::Vec3d(0, -1, 0)});
    CHECK(cv::norm(moved.controlTangents[0] - *displayUnit(tangent)) < 1e-6);
    const double pi = std::acos(-1.0);
    CHECK(interpolateDisplayOffset({0, 10}, {170 * pi / 180, -170 * pi / 180}, 5)
        == doctest::Approx(pi));
    auto scaled = points;
    for (auto& point : scaled) point *= 4;
    const auto scaledField = fiberDisplayField(scaled, normals, {8}, {cv::Vec3d(0, -1, 0)});
    CHECK(scaledField.controlOffsets[0] == doctest::Approx(moved.controlOffsets[0]));
}

TEST_CASE("Fiber display metadata validates and survives CP roundtrip and reversal")
{
    using namespace vc::fiber_tracer;
    CHECK(fiberWidthFromJson(nlohmann::json::object()) == 0);
    CHECK(fiberWidthFromJson({{"width", 12.5}}) == 12.5);
    CHECK(fiberWidthGapFromJson(nlohmann::json::object()) == doctest::Approx(0.2));
    CHECK(fiberWidthGapFromJson({{"width_gap_fraction", 0.35}}) == doctest::Approx(0.35));
    CHECK(fiberWidthGapFromJson({{"width_gap_fraction", 0}}) == 0);
    CHECK_THROWS(fiberWidthGapFromJson({{"width_gap_fraction", -0.1}}));
    CHECK_THROWS(fiberWidthGapFromJson({{"width_gap_fraction", 1.1}}));
    CHECK_THROWS(fiberWidthGapFromJson({{"width_gap_fraction", "20%"}}));
    CHECK_THROWS(fiberWidthFromJson({{"width", -1}}));
    CHECK_THROWS(displayNormalFromJson({{"display_normal", {0, 0, 0}}}));
    vc3d::line_annotation::StoredControlPoint cp;
    cp.displayNormal = cv::Vec3d(0, 1, 0);
    const auto json = vc3d::line_annotation::storedControlPointToJson(cp);
    const auto restored = vc3d::line_annotation::storedControlPointFromJson(json, 3);
    REQUIRE(restored.displayNormal.has_value());
    CHECK(cv::norm(*restored.displayNormal - *cp.displayNormal) < 1e-6);
    const auto reversed = vc3d::line_annotation::reversedStoredControlPoints({cp, cp});
    REQUIRE(reversed[0].displayNormal.has_value());
    CHECK(cv::norm(*reversed[0].displayNormal - *cp.displayNormal) < 1e-6);
    vc3d::line_annotation::LineControlPoint live;
    live.displayNormal = cp.displayNormal;
    const auto optimized = vc3d::line_annotation::mergeOptimizerControlPoints(
        vc3d::line_annotation::optimizerControlPoints({live}), {live});
    REQUIRE(optimized[0].displayNormal.has_value());
    CHECK(cv::norm(*optimized[0].displayNormal - *cp.displayNormal) < 1e-6);
    const auto collapsed = vc3d::line_annotation::collapseControlPointsAtClick(
        {live}, {0}, 0.0, cv::Vec3d(1, 2, 3));
    REQUIRE(collapsed.controlPoints[0].displayNormal.has_value());
    CHECK(cv::norm(*collapsed.controlPoints[0].displayNormal - *cp.displayNormal) < 1e-6);
}

TEST_CASE("Fiber width tolerance guides bracket the full width by twenty percent")
{
    const auto offsets = vc::fiber_tracer::fiberWidthEdgeOffsets(40);
    CHECK(offsets[0] == doctest::Approx(-24));
    CHECK(offsets[1] == doctest::Approx(-16));
    CHECK(offsets[2] == doctest::Approx(16));
    CHECK(offsets[3] == doctest::Approx(24));
    const auto scaled = vc::fiber_tracer::fiberWidthEdgeOffsets(10);
    const auto unset = vc::fiber_tracer::fiberWidthEdgeOffsets(0);
    const auto custom = vc::fiber_tracer::fiberWidthEdgeOffsets(40, 0.5);
    CHECK(custom[0] == doctest::Approx(-30));
    CHECK(custom[1] == doctest::Approx(-10));
    CHECK(custom[2] == doctest::Approx(10));
    CHECK(custom[3] == doctest::Approx(30));
    for (size_t i = 0; i < offsets.size(); ++i) {
        CHECK(scaled[i] * 4 == doctest::Approx(offsets[i]));
        CHECK(unset[i] == 0);
    }
}

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

TEST_CASE("Display normal override rotates both ribbons without changing model normals")
{
    const auto model = lineModel();
    vc::lasagna::LineViewConfig config;
    const auto baseline = vc::lasagna::buildLineViewSurfaces(model, config);
    config.orientedPointNormals.assign(model.points.size(), cv::Vec3f(0, 0, 1));
    config.displayPointNormals.assign(model.points.size(), cv::Vec3f(0, 1, 0));
    const auto corrected = vc::lasagna::buildLineViewSurfaces(model, config);
    const auto* top = corrected.lineSurface->rawPointsPtr();
    const auto* side = corrected.lineSideSlice->rawPointsPtr();
    REQUIRE(top != nullptr);
    REQUIRE(side != nullptr);
    const auto topAcross = (*top)(top->rows - 1, 0) - (*top)(0, 0);
    const auto sideAcross = (*side)(side->rows - 1, 0) - (*side)(0, 0);
    CHECK(std::abs(topAcross[2]) > 1);
    CHECK(std::abs(topAcross[1]) < 1e-5);
    CHECK(std::abs(sideAcross[1]) > 1);
    CHECK(std::abs(sideAcross[2]) < 1e-5);
    CHECK(cv::norm(corrected.lineUpVectors.front() - cv::Vec3f(0, 1, 0)) < 1e-5);
    const auto after = vc::lasagna::buildLineViewSurfaces(model);
    CHECK(cv::norm(*baseline.lineSurface->rawPointsPtr(), *after.lineSurface->rawPointsPtr()) == 0);
}

TEST_CASE("CP direction editing preserves the unedited component and round trips")
{
    using namespace vc::fiber_tracer;
    const auto axis=*displayUnit({1,0,0.5});
    const auto edited=editControlDirection(axis,{0,0,1},{1,1,99});
    REQUIRE(edited);
    CHECK((*edited)[2] == doctest::Approx(axis[2]));
    CHECK((*edited)[0] == doctest::Approx((*edited)[1]));
    CHECK(cv::norm(*edited) == doctest::Approx(1));
    const auto reversed=editControlDirection(-axis,{0,0,1},{-1,-1,0});
    REQUIRE(reversed);
    CHECK(cv::norm(*reversed+*edited)<1e-9);
    const auto oppositeDrag=editControlDirection(axis,{0,0,1},{-1,-1,0},cv::Vec3d(1,0,0));
    REQUIRE(oppositeDrag);
    CHECK(cv::norm(*oppositeDrag-*edited)<1e-9);
    CHECK_FALSE(editControlDirection(axis,{0,0,1},{0,0,1}));
    vc3d::line_annotation::StoredControlPoint cp;
    cp.direction=*edited;
    auto json=vc3d::line_annotation::storedControlPointToJson(cp);
    const auto loaded=vc3d::line_annotation::storedControlPointFromJson(json,3);
    REQUIRE(loaded.direction);
    CHECK(cv::norm(*loaded.direction-*edited)<1e-9);
    const auto reversedControls=vc3d::line_annotation::reversedStoredControlPoints({cp});
    REQUIRE(reversedControls.front().direction);
    CHECK(cv::norm(*reversedControls.front().direction+*edited)<1e-9);
    json["direction"]={0,0,0};
    CHECK_THROWS(vc3d::line_annotation::storedControlPointFromJson(json,3));
}

TEST_CASE("Spline uses signed annotated interior tangents")
{
    vc::lasagna::LineSplineRequest request;
    request.controlPoints={{0,0,0},{10,0,0},{20,0,0}};
    request.sampleSpacing=0.01;
    request.controlDirections={std::nullopt,cv::Vec3d(1,0.5,0),std::nullopt};
    const auto result=vc::lasagna::interpolateLineControlPoints(request);
    const int k=result.controlPointIndices[1];
    const auto tangent=*vc::fiber_tracer::displayUnit(result.points[k+1]-result.points[k-1]);
    CHECK(tangent.dot(*vc::fiber_tracer::displayUnit({1,0.5,0}))>0.999);
    request.controlDirections[1]=cv::Vec3d(-1,-0.5,0);
    std::reverse(request.controlPoints.begin(),request.controlPoints.end());
    const auto reversed=vc::lasagna::interpolateLineControlPoints(request);
    const int j=reversed.controlPointIndices[1];
    CHECK(vc::fiber_tracer::displayUnit(reversed.points[j+1]-reversed.points[j-1])->dot(-tangent)>0.999);
}

TEST_CASE("Fiber mode forwards CP axes into spline runs and Lasagna constraints")
{
    FiberModeNormalSampler sampler;
    for (const auto goal : {vc3d::line_annotation::SegmentInterpolationGoal::Cspline,
                           vc3d::line_annotation::SegmentInterpolationGoal::Lasagna}) {
        vc3d::line_annotation::FiberModeOptimizationRequest request;
        request.baseNormalSampler=&sampler;
        request.globalMode=vc3d::line_annotation::FiberOptimizationMode::Lasagna;
        for (int i=0;i<=20;++i) request.linePointsBase.push_back({double(i),0,0});
        request.controlPoints={{0,{0,0,0},true,0},{10,{10,0,0},false,10},{20,{20,0,0},false,20}};
        request.controlPoints[1].direction=cv::Vec3d(1,0.2,0);
        for (size_t i=0;i<2;++i) {
            request.controlPoints[i].segmentToNext.emplace();
            request.controlPoints[i].segmentToNext->interpGoal=goal;
        }
        request.lasagnaConfig.segmentLength=0.5;
        request.lasagnaConfig.maxIterations=20;
        request.lasagnaConfig.printSolverProgress=false;
        request.extrapolationDistanceBaseVoxels=0;
        request.retainOpenTails=false;
        const auto result=vc3d::line_annotation::optimizeFiberWithNativeFallback(request);
        REQUIRE(result.controlPoints[1].direction);
        const int k=result.controlPoints[1].optimizedIndex;
        REQUIRE(k>0);
        REQUIRE(k+1<int(result.optimization.line.points.size()));
        const auto direction=*vc::fiber_tracer::displayUnit(
            result.optimization.line.points[k+1].position-result.optimization.line.points[k-1].position);
        CHECK(direction.dot(*vc::fiber_tracer::displayUnit({1,0.2,0}))>0.98);
    }
}

TEST_CASE("Display normal axes choose the short rotation for either saved sign")
{
    using namespace vc::fiber_tracer;
    const double pi = std::acos(-1.0);
    const cv::Vec3d tangent{1,0,0}, baseline{0,0,1};
    const auto almostReversed = rotateDisplayNormal(baseline, tangent, 179*pi/180);
    REQUIRE(displayNormalOffset(baseline, almostReversed, tangent));
    CHECK(*displayNormalOffset(baseline, almostReversed, tangent) ==
          doctest::Approx(-pi/180));
    CHECK(*displayNormalOffset(baseline, -almostReversed, tangent) ==
          doctest::Approx(-pi/180));
    CHECK(interpolateDisplayOffset({0,1}, {85*pi/180,-85*pi/180}, 0.5) ==
          doctest::Approx(pi/2));
    std::vector<cv::Vec3f> points, normals;
    for (int i=0; i<=20; ++i) {
        points.emplace_back(i,0,0);
        normals.emplace_back(0,0,1);
    }
    const auto a = fiberDisplayField(points,normals,{0,10,20},
        {baseline,almostReversed,baseline});
    const auto b = fiberDisplayField(points,normals,{0,10,20},
        {-baseline,-almostReversed,-baseline});
    for (size_t i=0; i<points.size(); ++i) {
        CHECK(cv::norm(a.normals[i]-b.normals[i]) < 1e-5);
        CHECK(a.normals[i].dot(normals[i]) > 0.99);
    }
    auto model = lineModel();
    vc::lasagna::LineViewConfig config;
    config.orientedPointNormals.assign(3, cv::Vec3f(baseline));
    config.displayPointNormals = {cv::Vec3f(baseline),cv::Vec3f(almostReversed),cv::Vec3f(-baseline)};
    const auto views = vc::lasagna::buildLineViewSurfaces(model,config);
    for (auto surface : {views.lineSurface,views.lineSideSlice}) {
        const auto& grid = *surface->rawPointsPtr();
        for (int i=1; i<grid.cols; ++i)
            CHECK((grid(6,i)-grid(0,i)).dot(grid(6,i-1)-grid(0,i-1)) > 0);
    }
}

TEST_CASE("Corrected normals use exactly the ordinary construction pipeline")
{
    auto model = lineModel();
    model.points[1].position = {1,0.2,0};
    model.points[2].position = {2,0,0};
    vc::lasagna::LineViewConfig config;
    config.controlPointLinePositions = {0,1,2};
    config.displayPointNormals = {{0,0,1},{0,0.4f,0.916515f},{0,0,-1}};
    const auto corrected = vc::lasagna::buildLineViewSurfaces(model,config);
    auto injected = model;
    for (size_t i=0; i<model.points.size(); ++i)
        injected.points[i].sampledNormal = {cv::Vec3d(config.displayPointNormals[i]),true,{}};
    config.orientedPointNormals = config.displayPointNormals;
    config.displayPointNormals.clear();
    const auto ordinary = vc::lasagna::buildLineViewSurfaces(injected,config);
    CHECK(cv::norm(*ordinary.lineSurface->rawPointsPtr(),
                   *corrected.lineSurface->rawPointsPtr()) == 0);
    CHECK(cv::norm(*ordinary.lineSideSlice->rawPointsPtr(),
                   *corrected.lineSideSlice->rawPointsPtr()) == 0);
    CHECK(ordinary.lineUpVectors == corrected.lineUpVectors);
}

TEST_CASE("Equal CP axes do not inherit a half turn from the sampled baseline")
{
    using namespace vc::fiber_tracer;
    std::vector<cv::Vec3f> points, normals;
    const cv::Vec3d up{0,0,1}, tangent{1,0,0};
    for (int i=0; i<=20; ++i) {
        points.emplace_back(i*0.1f,0,0);
        normals.emplace_back(rotateDisplayNormal(up,tangent,i*170.0/20*std::acos(-1.0)/180));
    }
    const auto field = fiberDisplayField(points,normals,{0,20},{up,-up});
    for (const auto& n : field.normals) CHECK(std::abs(n.dot(cv::Vec3f(up))) > 0.99999);
    const auto baseline = fiberDisplayField(points,normals,{0,20},{std::nullopt,std::nullopt});
    for (size_t i=0; i<points.size(); ++i)
        CHECK(std::abs(baseline.normals[i].dot(normals[i])) > 0.99999);
}

TEST_CASE("Zero display correction preserves smoothed ribbon geometry")
{
    auto model = lineModel();
    vc::lasagna::LineViewConfig config;
    config.orientedPointNormals = {{0,0,1}, {0,0.6f,0.8f}, {0,0,1}};
    for (size_t i=0; i<model.points.size(); ++i)
        model.points[i].sampledNormal.normal = cv::Vec3d(config.orientedPointNormals[i]);
    const auto baseline = vc::lasagna::buildLineViewSurfaces(model, config);
    config.displayPointNormals = config.orientedPointNormals;
    const auto corrected = vc::lasagna::buildLineViewSurfaces(model, config);
    CHECK(cv::norm(*baseline.lineSurface->rawPointsPtr(),
                   *corrected.lineSurface->rawPointsPtr()) < 1e-5);
    CHECK(cv::norm(*baseline.lineSideSlice->rawPointsPtr(),
                   *corrected.lineSideSlice->rawPointsPtr()) < 1e-5);
}

TEST_CASE("Ribbons retain ordinary QuadSurface rendering and picking after origin shifts")
{
    auto model=lineModel();
    model.points[2].position={10,20,0};
    vc::lasagna::LineViewConfig config;
    config.targetSpacingBaseVoxels=50; // Deliberately coarse support grid.
    config.controlPointLinePositions={0,1,2};
    config.displayPointNormals.assign(3,cv::Vec3f(0,0,1));
    const auto views=vc::lasagna::buildLineViewSurfaces(model,config);
    const std::vector<cv::Vec3f> p{{0,0,0},{10,0,0},{10,20,0}};
    for (const auto& surface : {views.lineSurface,views.lineSideSlice}) {
        CHECK(typeid(*surface) == typeid(QuadSurface));
        surface->shiftSurfaceOrigin({123,-17});
        const double col=views.stripPositionMap.originalPositionToStripGridColumn(0.5);
        const auto uv=surface->gridToSurface({col,3});
        const auto sample=surface->sampleAtSurface(uv);
        REQUIRE(sample.valid());
        const auto expected=cv::Vec3f(5,0,0);
        CHECK(cv::norm(sample.volume-expected)<1e-5);
        for (float scale : {0.5f,2.0f}) {
            cv::Mat_<cv::Vec3f> coords,normals;
            surface->gen(&coords,&normals,{1,1},{0,0,0},scale,
                          {float(uv[0]*scale),float(uv[1]*scale),0});
            CHECK(cv::norm(coords(0,0)-sample.volume)<1e-4);
            CHECK(cv::norm(normals(0,0))==doctest::Approx(1));
        }
        cv::Vec3f ptr{float(uv[0]*surface->scale()[0]),float(uv[1]*surface->scale()[1]),0};
        CHECK(cv::norm(surface->coord(ptr)-sample.volume)<1e-4);
        CHECK(surface->pointTo(ptr,expected,0.01f)<0.001f);
        CHECK(cv::norm(surface->coord(ptr)-expected)<0.001f);
        CHECK_FALSE(surface->sampleAtSurface(surface->gridToSurface({-1,3})).valid());
        // Off-center depth sampling must use the same geometry-derived normals
        // as an ordinary surface, not a centerline frame extended across rows.
        QuadSurface reference(*surface->rawPointsPtr(), surface->scale());
        reference.shiftSurfaceOrigin({123,-17});
        const auto edgeUV = surface->gridToSurface({col, 4.5});
        cv::Mat_<cv::Vec3f> actualCoords, actualNormals, expectedCoords, expectedNormals;
        const cv::Vec3f offset{float(edgeUV[0]), float(edgeUV[1]), 3.0f};
        surface->gen(&actualCoords, &actualNormals, {1,1}, {0,0,0}, 1, offset);
        reference.gen(&expectedCoords, &expectedNormals, {1,1}, {0,0,0}, 1, offset);
        CHECK(cv::norm(actualCoords, expectedCoords) < 1e-5);
        CHECK(cv::norm(actualNormals, expectedNormals) < 1e-5);
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

TEST_CASE("line annotation shift scroll by arclength moves one strip column per notch")
{
    using vc3d::line_annotation::kShiftScrollLineStepBaseVoxels;
    using vc3d::line_annotation::shiftedLinePositionByArclength;
    CHECK(kShiftScrollLineStepBaseVoxels == doctest::Approx(8.0));

    // Mixed density: 4 vx vertices for positions 0..10 (arclength 0..40),
    // then 32 vx vertices for positions 11..15 (arclength 72..200).
    std::vector<double> arclengths;
    for (int i = 0; i <= 10; ++i) {
        arclengths.push_back(4.0 * i);
    }
    for (int i = 1; i <= 5; ++i) {
        arclengths.push_back(40.0 + 32.0 * i);
    }
    REQUIRE(arclengths.size() == 16);

    // One notch = 8 vx: two vertices in the dense region, a quarter vertex in
    // the sparse one.
    CHECK(shiftedLinePositionByArclength(2.0, 1, 1, arclengths) == doctest::Approx(4.0));
    CHECK(shiftedLinePositionByArclength(12.0, 1, 1, arclengths) == doctest::Approx(12.25));
    CHECK(shiftedLinePositionByArclength(12.0, -1, 1, arclengths) == doctest::Approx(11.75));
    // Step size scales the notch; crossing the density boundary stays exact.
    CHECK(shiftedLinePositionByArclength(8.0, 1, 5, arclengths) == doctest::Approx(11.0));
    // Clamped at both ends.
    CHECK(shiftedLinePositionByArclength(15.0, 3, 1, arclengths) == doctest::Approx(15.0));
    CHECK(shiftedLinePositionByArclength(0.5, -1, 1, arclengths) == doctest::Approx(0.0));
    // Without a usable map the index step applies.
    const std::vector<double> none;
    CHECK(shiftedLinePositionByArclength(3.0, 2, 1, none) == doctest::Approx(3.0));
}

TEST_CASE("line annotation arclength snap uses a quarter strip column")
{
    using vc3d::line_annotation::kControlPointSnapArclengthBaseVoxels;
    using vc3d::line_annotation::snappedControlPointLinePositionByArclength;
    CHECK(kControlPointSnapArclengthBaseVoxels == doctest::Approx(2.0));

    const std::vector<double> controlPositions{12.0, 20.0, 40.0};
    std::vector<double> dense;  // 4 vx per position: 2 vx = half a position
    for (int i = 0; i <= 50; ++i) {
        dense.push_back(4.0 * i);
    }
    CHECK(snappedControlPointLinePositionByArclength(19.5, controlPositions, dense) ==
          doctest::Approx(20.0));
    CHECK(snappedControlPointLinePositionByArclength(20.5, controlPositions, dense) ==
          doctest::Approx(20.0));
    CHECK(snappedControlPointLinePositionByArclength(19.49, controlPositions, dense) ==
          doctest::Approx(19.49));

    std::vector<double> sparse;  // 32 vx per position: 2 vx = 1/16 position
    for (int i = 0; i <= 50; ++i) {
        sparse.push_back(32.0 * i);
    }
    CHECK(snappedControlPointLinePositionByArclength(20.0625, controlPositions, sparse) ==
          doctest::Approx(20.0));
    CHECK(snappedControlPointLinePositionByArclength(20.07, controlPositions, sparse) ==
          doctest::Approx(20.07));

    // Without a usable map the quarter-position rule applies.
    const std::vector<double> none;
    CHECK(snappedControlPointLinePositionByArclength(19.75, controlPositions, none) ==
          doctest::Approx(20.0));
    CHECK(snappedControlPointLinePositionByArclength(19.7, controlPositions, none) ==
          doctest::Approx(19.7));
}

TEST_CASE("line annotation overlay group key is the registration key")
{
    // Registration (applyGeneratedOverlay) and the pan-time translation must
    // address the viewer group by the same string; both go through this helper.
    CHECK(vc3d::line_annotation::generatedOverlayGroupKey("line_surface_x") ==
          "line_annotation_overlay_line_surface_x");
}

TEST_CASE("line annotation static overlay pan translation shifts at fixed zoom only")
{
    using vc3d::line_annotation::GeneratedOverlayCameraBaseline;
    using vc3d::line_annotation::generatedOverlayPanTranslation;

    GeneratedOverlayCameraBaseline baseline;
    baseline.referenceScene = QPointF(100.0, 50.0);
    baseline.scale = 2.0;

    SUBCASE("a camera pan is a common scene delta")
    {
        const auto delta = generatedOverlayPanTranslation(baseline, QPointF(90.0, 50.0), 2.0);
        REQUIRE(delta.has_value());
        CHECK(delta->x() == doctest::Approx(-10.0));
        CHECK(delta->y() == doctest::Approx(0.0));
    }

    SUBCASE("an unchanged camera yields a zero delta")
    {
        const auto delta = generatedOverlayPanTranslation(baseline, QPointF(100.0, 50.0), 2.0);
        REQUIRE(delta.has_value());
        CHECK(delta->isNull());
    }

    SUBCASE("a zoom change requires a rebuild")
    {
        CHECK_FALSE(generatedOverlayPanTranslation(baseline, QPointF(90.0, 50.0), 2.5).has_value());
    }

    SUBCASE("an unknown baseline requires a rebuild")
    {
        const GeneratedOverlayCameraBaseline unknown;
        CHECK_FALSE(generatedOverlayPanTranslation(unknown, QPointF(90.0, 50.0), 2.0).has_value());
        CHECK_FALSE(generatedOverlayPanTranslation(
                        baseline, QPointF(std::numeric_limits<double>::quiet_NaN(), 0.0), 2.0)
                        .has_value());
    }
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

TEST_CASE("line annotation plane origin shift along normal is signed and normal-length independent")
{
    const cv::Vec3f origin{1.0f, 2.0f, 3.0f};
    const cv::Vec3f normal{0.0f, -4.0f, 0.0f};

    const cv::Vec3f forward =
        vc3d::line_annotation::planeOriginShiftedAlongNormal(origin, normal, 8.0);
    CHECK(forward[0] == doctest::Approx(1.0f));
    CHECK(forward[1] == doctest::Approx(-6.0f));
    CHECK(forward[2] == doctest::Approx(3.0f));

    const cv::Vec3f backward =
        vc3d::line_annotation::planeOriginShiftedAlongNormal(origin, normal, -8.0);
    CHECK(backward[1] == doctest::Approx(10.0f));

    const cv::Vec3f degenerate = vc3d::line_annotation::planeOriginShiftedAlongNormal(
        origin, cv::Vec3f{0.0f, 0.0f, 0.0f}, 8.0);
    CHECK(cv::norm(degenerate - origin) == doctest::Approx(0.0f));
    const cv::Vec3f nanDistance = vc3d::line_annotation::planeOriginShiftedAlongNormal(
        origin, normal, std::numeric_limits<double>::quiet_NaN());
    CHECK(cv::norm(nanDistance - origin) == doctest::Approx(0.0f));
}

TEST_CASE("line annotation straight-ahead direction follows the raw tangent and locks per gesture")
{
    using vc3d::line_annotation::straightAheadDirection;
    const cv::Vec3f ahead{0.0f, 0.0f, 1.0f};

    // Fresh gesture: the sign is whichever half-space of the normal faces
    // increasing line position, so a display-sign-flipped (or rotated past a
    // quarter turn) normal translates against itself.
    CHECK(straightAheadDirection(cv::Vec3f{0.0f, 0.0f, 1.0f}, ahead, false, 1.0) == 1.0);
    CHECK(straightAheadDirection(cv::Vec3f{0.0f, 0.0f, -1.0f}, ahead, false, 1.0) == -1.0);
    CHECK(straightAheadDirection(cv::Vec3f{0.1f, 0.0f, -0.05f}, ahead, false, 1.0) == -1.0);
    // Exactly orthogonal resolves to forward.
    CHECK(straightAheadDirection(cv::Vec3f{1.0f, 0.0f, 0.0f}, ahead, false, -1.0) == 1.0);

    // Mid-gesture: the tangent at the marker has swung past the plane normal
    // (the model curving away), yet the plane keeps the sign it started with,
    // so two +1 notches and one +2 notch travel the same way.
    CHECK(straightAheadDirection(cv::Vec3f{0.0f, 0.0f, -1.0f}, ahead, true, 1.0) == 1.0);
    CHECK(straightAheadDirection(cv::Vec3f{0.0f, 0.0f, 1.0f}, ahead, true, -1.0) == -1.0);
    // A stale lock value that is not a sign is ignored.
    CHECK(straightAheadDirection(cv::Vec3f{0.0f, 0.0f, -1.0f}, ahead, true, 0.0) == -1.0);
}

TEST_CASE("line annotation straight-ahead distance follows the marker's actual arclength advance")
{
    using vc3d::line_annotation::kShiftScrollLineStepBaseVoxels;
    using vc3d::line_annotation::shiftedLinePositionByArclength;
    using vc3d::line_annotation::straightAheadDistanceForShiftScroll;

    // Same mixed-density map as the along-line test: 4 vx vertices for
    // positions 0..10, 32 vx vertices for positions 11..15 (total 200 vx).
    std::vector<double> arclengths;
    for (int i = 0; i <= 10; ++i) {
        arclengths.push_back(4.0 * i);
    }
    for (int i = 1; i <= 5; ++i) {
        arclengths.push_back(40.0 + 32.0 * i);
    }
    REQUIRE(arclengths.size() == 16);

    // One notch inside the line: the plane travels exactly the marker's 8 vx,
    // in both densities and in both directions.
    double from = 2.0;
    double to = shiftedLinePositionByArclength(from, 1, 1, arclengths);
    CHECK(straightAheadDistanceForShiftScroll(from, to, arclengths) ==
          doctest::Approx(kShiftScrollLineStepBaseVoxels));
    from = 12.0;
    to = shiftedLinePositionByArclength(from, -1, 1, arclengths);
    CHECK(straightAheadDistanceForShiftScroll(from, to, arclengths) ==
          doctest::Approx(-kShiftScrollLineStepBaseVoxels));
    // Slice step size scales the notch.
    from = 2.0;
    to = shiftedLinePositionByArclength(from, 1, 3, arclengths);
    CHECK(straightAheadDistanceForShiftScroll(from, to, arclengths) ==
          doctest::Approx(3.0 * kShiftScrollLineStepBaseVoxels));

    // A notch that clamps at the line end moves the plane only as far as the
    // marker got (196 -> 200 is 4 vx, not 8), and a notch AT the end moves nothing.
    from = vc3d::fiber_slice::linePositionAtArclength(arclengths, 196.0);
    to = shiftedLinePositionByArclength(from, 1, 1, arclengths);
    CHECK(to == doctest::Approx(15.0));
    CHECK(straightAheadDistanceForShiftScroll(from, to, arclengths) == doctest::Approx(4.0));
    to = shiftedLinePositionByArclength(15.0, 1, 1, arclengths);
    CHECK(straightAheadDistanceForShiftScroll(15.0, to, arclengths) == doctest::Approx(0.0));

    // Without a usable map (the dialog passes an empty one) the marker does not
    // move and neither does the plane.
    const std::vector<double> noMap;
    CHECK(shiftedLinePositionByArclength(2.0, 2, 1, noMap) == doctest::Approx(2.0));
    CHECK(straightAheadDistanceForShiftScroll(2.0, 2.0, noMap) == doctest::Approx(0.0));
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

TEST_CASE("line annotation carried line position stays with its controls across a landing")
{
    using vc3d::line_annotation::carriedGeneratedLinePosition;
    using vc3d::line_annotation::GeneratedLineCarryControl;
    using vc3d::line_annotation::remappedGeneratedLinePosition;

    // Two wraps of one fiber: an outbound pass along y=0 (indices 0..100,
    // x = index) and a return pass along y=5 (indices 101..201, x = 100 -
    // (index - 101)), five voxels apart like adjacent windings. Controls A
    // (index 20) and B (index 120, the last one); everything past B is the
    // extrapolated tail.
    const auto wraps = [](int tailEnd, float outboundY) {
        std::vector<cv::Vec3f> line;
        for (int i = 0; i <= 100; ++i) {
            line.push_back({static_cast<float>(i), outboundY, 0.0f});
        }
        for (int i = 101; i <= tailEnd; ++i) {
            line.push_back({static_cast<float>(100 - (i - 101)), 5.0f, 0.0f});
        }
        return line;
    };
    const std::vector<cv::Vec3f> oldLine = wraps(201, 0.0f);
    const std::vector<GeneratedLineCarryControl> oldControls{{1, 20.0}, {2, 120.0}};

    SUBCASE("a re-traced tail that stops short clamps the position to its end")
    {
        // The landing's trace from B found no candidates past index 130: the
        // old spot 60 samples into the tail (x=21, y=5) is 50 voxels from the
        // new tail's end but only 5 from the outbound pass underneath it.
        const std::vector<cv::Vec3f> newLine = wraps(130, 0.0f);
        const std::vector<GeneratedLineCarryControl> newControls{{1, 20.0}, {2, 120.0}};
        // The regression: nearest-vertex matching lands on the other wrap
        // (its index tiebreak only chooses among that wrap's vertices).
        CHECK(remappedGeneratedLinePosition(oldLine, newLine, 180.0) < 100.0);
        CHECK(carriedGeneratedLinePosition(oldLine, oldControls, newLine, newControls, 180.0) ==
              doctest::Approx(130.0));
    }

    SUBCASE("a tail re-traced further keeps the distance past the last control")
    {
        const std::vector<cv::Vec3f> newLine = wraps(260, 0.0f);
        const std::vector<GeneratedLineCarryControl> newControls{{1, 20.0}, {2, 120.0}};
        CHECK(carriedGeneratedLinePosition(oldLine, oldControls, newLine, newControls, 180.0) ==
              doctest::Approx(180.0));
    }

    SUBCASE("a control placed in the tail does not move a position beyond it")
    {
        // The placement that extends the fiber: C lands at index 150 and the
        // tail is re-traced past it; the position stays 60 past B.
        const std::vector<cv::Vec3f> newLine = wraps(260, 0.0f);
        const std::vector<GeneratedLineCarryControl> newControls{
            {1, 20.0}, {2, 120.0}, {3, 150.0}};
        CHECK(carriedGeneratedLinePosition(oldLine, oldControls, newLine, newControls, 180.0) ==
              doctest::Approx(180.0));
    }

    SUBCASE("a renumbered tail carries by arc length, not by sample count")
    {
        // The new line samples the tail at half spacing from B on: 60
        // voxels past B is now 120 samples past it.
        std::vector<cv::Vec3f> newLine(oldLine.begin(), oldLine.begin() + 121);
        for (int k = 1; k <= 160; ++k) {
            newLine.push_back({static_cast<float>(81.0 - 0.5 * k), 5.0f, 0.0f});
        }
        const std::vector<GeneratedLineCarryControl> newControls{{1, 20.0}, {2, 120.0}};
        CHECK(carriedGeneratedLinePosition(oldLine, oldControls, newLine, newControls, 180.0) ==
              doctest::Approx(240.0));
    }

    SUBCASE("a kept tail resolves the exact spot even after the span before it changed length")
    {
        // Manual mode, or a tail the landing kept: the old tail bulged to
        // y=8 between indices 125 and 145 (between B and the spot, which
        // made that stretch longer) and the re-trace straightened it; past
        // the bulge the tail is the same geometry. Counting arc length from
        // B would overshoot by the bulge's extra length; the old spot is
        // right there on the new line.
        std::vector<cv::Vec3f> bulged = oldLine;
        for (int i = 125; i <= 145; ++i) {
            bulged[static_cast<size_t>(i)][1] =
                5.0f + 3.0f * (1.0f - std::abs(static_cast<float>(i - 135)) / 10.0f);
        }
        const std::vector<GeneratedLineCarryControl> newControls{{1, 20.0}, {2, 120.0}};
        const std::vector<double> bulgedArcs =
            vc3d::line_annotation::generatedCumulativeArcLength(bulged);
        const double arcCarried = 120.0 + (bulgedArcs[180] - bulgedArcs[120]);
        REQUIRE(arcCarried > 180.5);
        CHECK(carriedGeneratedLinePosition(bulged, oldControls, oldLine, newControls, 180.0) ==
              doctest::Approx(180.0));
        // Beyond tolerance the arc-length carry decides: the same tail
        // shifted two voxels sideways.
        std::vector<cv::Vec3f> shifted = oldLine;
        for (size_t i = 121; i < shifted.size(); ++i) {
            shifted[i][1] = 7.0f;
        }
        const std::vector<double> shiftedArcs =
            vc3d::line_annotation::generatedCumulativeArcLength(shifted);
        const double expectedShifted = vc3d::line_annotation::generatedLinePositionAtArcLength(
            shiftedArcs, shiftedArcs[120] + (bulgedArcs[180] - bulgedArcs[120]));
        CHECK(carriedGeneratedLinePosition(bulged, oldControls, shifted, newControls, 180.0) ==
              doctest::Approx(expectedShifted));
        // With the spot gone from the new tail, the arc-length carry takes
        // over (the new tail is sampled at half spacing: twice the samples).
        std::vector<cv::Vec3f> resampledTail(oldLine.begin(), oldLine.begin() + 121);
        for (int k = 1; k <= 160; ++k) {
            resampledTail.push_back({static_cast<float>(81.0 - 0.5 * k), 5.0f, 0.0f});
        }
        // Resampled at half spacing the old spot IS still on the new tail;
        // shift the tail sideways so it is not.
        for (size_t i = 121; i < resampledTail.size(); ++i) {
            resampledTail[i][1] = 7.0f;
        }
        // The old arc from B to the spot (60 voxels plus the bulge) past B:
        // the first step reaches the shifted tail (sqrt(0.5^2 + 2^2)), the
        // rest are half-voxel samples.
        const double expected =
            121.0 + ((bulgedArcs[180] - bulgedArcs[120]) - std::sqrt(0.25 + 4.0)) / 0.5;
        CHECK(carriedGeneratedLinePosition(bulged, oldControls, resampledTail, newControls, 180.0) ==
              doctest::Approx(expected).epsilon(1.0e-4));
    }

    SUBCASE("inside a span the search never leaves the span")
    {
        // The span A..B was re-solved twelve voxels off its old path, which
        // puts the old spot (60, 0) nearer to the return pass (five voxels)
        // than to its own span, beyond the plain remap's tiebreak band. Plain
        // matching jumps wraps; the carry stays between A and B and lands on
        // the nearest point of the span.
        const std::vector<cv::Vec3f> newLine = wraps(201, -12.0f);
        const std::vector<GeneratedLineCarryControl> newControls{{1, 20.0}, {2, 120.0}};
        CHECK(remappedGeneratedLinePosition(oldLine, newLine, 60.0) > 100.0);
        CHECK(carriedGeneratedLinePosition(oldLine, oldControls, newLine, newControls, 60.0) ==
              doctest::Approx(60.0));
    }

    SUBCASE("a tail passing the old spot twice takes the passage nearest the expectation")
    {
        // The re-traced tail from B runs out along y=5, turns, and comes
        // back along y=5.6 (within tolerance of the old spot (21, 5) on
        // both passages) before leaving sideways. The arc-length
        // expectation is 60 past B, on the outbound passage: that passage
        // wins, not the geometrically nearest one.
        std::vector<cv::Vec3f> newLine(oldLine.begin(), oldLine.begin() + 121);
        for (int k = 1; k <= 75; ++k) {  // outbound: x 80..6 at y=5
            newLine.push_back({static_cast<float>(81 - k), 5.0f, 0.0f});
        }
        for (int k = 1; k <= 75; ++k) {  // return: x 7..81 at y=5.6
            newLine.push_back({static_cast<float>(6 + k), 5.6f, 0.0f});
        }
        const std::vector<GeneratedLineCarryControl> newControls{{1, 20.0}, {2, 120.0}};
        const double carried =
            carriedGeneratedLinePosition(oldLine, oldControls, newLine, newControls, 180.0);
        CHECK(carried == doctest::Approx(180.0));
        // With the outbound passage moved just outside tolerance (1.1
        // away) only the return passage still runs through the spot: it is
        // taken, at index 210 (x=21 on the return), although the arc-length
        // expectation (180) and the geometrically nearest candidates are on
        // the outbound passage. A nearest-with-tiebreak search followed by
        // a tolerance check settles on the outbound passage and, failing
        // the tolerance, falls back to the arc carry instead.
        std::vector<cv::Vec3f> outboundAway = newLine;
        for (size_t i = 121; i <= 195; ++i) {
            outboundAway[i][1] = 6.1f;
        }
        REQUIRE(outboundAway[210] == cv::Vec3f(21.0f, 5.6f, 0.0f));
        const double onReturn =
            carriedGeneratedLinePosition(oldLine, oldControls, outboundAway, newControls, 180.0);
        CHECK(onReturn == doctest::Approx(210.0));
        CHECK(onReturn != doctest::Approx(180.0));
    }

    SUBCASE("a sharp turn around the old spot is two passages")
    {
        // The tail from B heads toward the old spot (21, 5), overshoots it
        // by two voxels, turns sharply and comes back past it: both legs
        // run within tolerance of the spot, the turning vertex does not.
        // The leg nearer the arc-length expectation (60 past B: the first)
        // is taken at its closest point, not the second leg's.
        std::vector<cv::Vec3f> newLine(oldLine.begin(), oldLine.begin() + 121);
        for (int k = 1; k <= 62; ++k) {  // first leg: x 80..19 at y=5
            newLine.push_back({static_cast<float>(81 - k), 5.0f, 0.0f});
        }
        newLine.push_back({19.0f, 7.0f, 0.0f});          // the turn, 2.8 away
        for (int k = 1; k <= 62; ++k) {  // second leg: x 20..81 at y=5.8
            newLine.push_back({static_cast<float>(19 + k), 5.8f, 0.0f});
        }
        const std::vector<GeneratedLineCarryControl> newControls{{1, 20.0}, {2, 120.0}};
        CHECK(carriedGeneratedLinePosition(oldLine, oldControls, newLine, newControls, 180.0) ==
              doctest::Approx(180.0));
    }

    SUBCASE("a reversed line carries the tail on the other side of its control")
    {
        // The stored line was reversed between publishes: A and B swap
        // order, and a position 10 past B (old 130) is 10 BEFORE B on the
        // new line. Straight line, unit spacing, so arc and index agree.
        std::vector<cv::Vec3f> straight;
        for (int i = 0; i <= 200; ++i) {
            straight.push_back({static_cast<float>(i), 0.0f, 0.0f});
        }
        std::vector<cv::Vec3f> reversedLine(straight.rbegin(), straight.rend());
        const std::vector<GeneratedLineCarryControl> forward{{1, 20.0}, {2, 120.0}};
        const std::vector<GeneratedLineCarryControl> backward{{1, 180.0}, {2, 80.0}};
        // The exact spot (130, 0) is at reversed index 70.
        CHECK(carriedGeneratedLinePosition(straight, forward, reversedLine, backward, 130.0) ==
              doctest::Approx(70.0));
        // With the spot gone (the reversed tail shifted sideways) the arc
        // carry still goes 10 before B: index 70.
        std::vector<cv::Vec3f> reversedShifted = reversedLine;
        for (size_t i = 0; i < 80; ++i) {
            reversedShifted[i][1] = 3.0f;
        }
        const std::vector<double> shiftedArcs =
            vc3d::line_annotation::generatedCumulativeArcLength(reversedShifted);
        const double tenBeforeB = vc3d::line_annotation::generatedLinePositionAtArcLength(
            shiftedArcs, shiftedArcs[80] - 10.0);
        REQUIRE(tenBeforeB < 80.0);
        CHECK(carriedGeneratedLinePosition(straight, forward, reversedShifted, backward, 130.0) ==
              doctest::Approx(tenBeforeB));
        // Before A on the old line (old 10): 10 past A on the reversed one.
        CHECK(carriedGeneratedLinePosition(straight, forward, reversedLine, backward, 10.0) ==
              doctest::Approx(190.0));
        // Inside a span the bounds already follow the order: old 70 is
        // reversed index 130.
        CHECK(carriedGeneratedLinePosition(straight, forward, reversedLine, backward, 70.0) ==
              doctest::Approx(130.0));
        // With a single matched control the order is assumed kept (a
        // reversal comes from a merge and matches every control): the
        // geometry at that control is deliberately not consulted, so a
        // sharp bend from a re-solve, a repeated point or a non-finite
        // neighbour at the control cannot turn a kept line into a
        // "reversed" one.
        const std::vector<GeneratedLineCarryControl> onlyB{{2, 120.0}};
        CHECK(carriedGeneratedLinePosition(straight, onlyB, straight, onlyB, 130.0) ==
              doctest::Approx(130.0));
        std::vector<cv::Vec3f> bent = straight;
        for (int i = 121; i <= 125; ++i) {  // the line doubles back right after B
            bent[static_cast<size_t>(i)] = {static_cast<float>(240 - i), 0.0f, 0.0f};
        }
        std::vector<cv::Vec3f> bentShifted = bent;
        for (size_t i = 121; i < bentShifted.size(); ++i) {
            bentShifted[i][1] = 3.0f;
        }
        // Past B on the bent line; the arc carry goes 10 past B along it.
        const std::vector<double> bentArcs =
            vc3d::line_annotation::generatedCumulativeArcLength(bentShifted);
        CHECK(carriedGeneratedLinePosition(straight, onlyB, bentShifted, onlyB, 130.0) ==
              doctest::Approx(vc3d::line_annotation::generatedLinePositionAtArcLength(
                  bentArcs, bentArcs[120] + 10.0)));
        std::vector<cv::Vec3f> repeated = straight;
        repeated[121] = repeated[120];
        CHECK(carriedGeneratedLinePosition(straight, onlyB, repeated, onlyB, 130.0) > 120.0);
        std::vector<cv::Vec3f> holed = straight;
        holed[121] = {std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f};
        CHECK(carriedGeneratedLinePosition(straight, onlyB, holed, onlyB, 130.0) > 120.0);
    }

    SUBCASE("inside a sparse span the continuity tiebreak keeps the pass")
    {
        // One long span A(20)..B(190) holds both passes. The re-solve moved
        // the outbound pass six voxels, the return pass is five away from
        // the old spot (60, 0): within the remap's tiebreak band, the
        // candidate nearest the spot's expected place in the span wins.
        const std::vector<GeneratedLineCarryControl> sparseControls{{1, 20.0}, {2, 190.0}};
        const std::vector<cv::Vec3f> newLine = wraps(201, -6.0f);
        CHECK(carriedGeneratedLinePosition(oldLine, sparseControls, newLine, sparseControls, 60.0) ==
              doctest::Approx(60.0));
    }

    SUBCASE("a control placed since the displayed line anchors the cursor past it")
    {
        // The displayed line's controls are A and B; C was placed at index
        // 150 and published without a line, so it is only known on the
        // displayed line through its resolved placement. The cursor is 30
        // voxels past C. The landing lengthens span B..C (a bulge) and
        // re-traces the tail sideways (y=7), so the old spot is gone.
        std::vector<cv::Vec3f> newLine(oldLine.begin(), oldLine.begin() + 151);
        for (int i = 122; i <= 148; ++i) {
            newLine[static_cast<size_t>(i)][1] =
                5.0f + 0.5f * (14.0f - std::abs(static_cast<float>(i - 135)));
        }
        for (int k = 1; k <= 50; ++k) {
            newLine.push_back({static_cast<float>(51 - k), 7.0f, 0.0f});
        }
        const std::vector<GeneratedLineCarryControl> newControls{
            {1, 20.0}, {2, 120.0}, {3, 150.0}};
        const std::vector<double> newArcs =
            vc3d::line_annotation::generatedCumulativeArcLength(newLine);
        const double pastC = vc3d::line_annotation::generatedLinePositionAtArcLength(
            newArcs, vc3d::line_annotation::generatedArcLengthAt(newArcs, 150.0) + 30.0);
        const double pastB = vc3d::line_annotation::generatedLinePositionAtArcLength(
            newArcs, vc3d::line_annotation::generatedArcLengthAt(newArcs, 120.0) + 60.0);
        REQUIRE(pastC > pastB + 2.0);
        // With C among the displayed line's anchors: 30 past C.
        const std::vector<GeneratedLineCarryControl> withPlacement{
            {1, 20.0}, {2, 120.0}, {3, 150.0}};
        CHECK(carriedGeneratedLinePosition(oldLine, withPlacement, newLine, newControls, 180.0) ==
              doctest::Approx(pastC));
        // Without it the cursor would be measured from B and land before
        // the spot it had past C.
        CHECK(carriedGeneratedLinePosition(oldLine, oldControls, newLine, newControls, 180.0) ==
              doctest::Approx(pastB));
        // The dialog adds the placement from its resolved entries, by
        // identity and only for the displayed line's revision.
        std::vector<GeneratedLineCarryControl> anchors = oldControls;
        vc3d::line_annotation::GeneratedPendingPlacement placedC;
        placedC.identity = 3;
        placedC.linePosition = 150.0;
        placedC.lineRevision = 7;
        vc3d::line_annotation::GeneratedPendingPlacement otherRevision = placedC;
        otherRevision.identity = 4;
        otherRevision.lineRevision = 8;
        vc3d::line_annotation::GeneratedPendingPlacement knownB;
        knownB.identity = 2;
        knownB.linePosition = 999.0;
        knownB.lineRevision = 7;
        vc3d::line_annotation::appendGeneratedLineCarryControls(
            anchors,
            std::vector<vc3d::line_annotation::GeneratedPendingPlacement>{
                placedC, otherRevision, knownB},
            7);
        REQUIRE(anchors.size() == 3);
        CHECK(anchors[1].linePosition == doctest::Approx(120.0));
        CHECK(anchors[2].identity == 3);
        CHECK(carriedGeneratedLinePosition(oldLine, anchors, newLine, newControls, 180.0) ==
              doctest::Approx(pastC));
    }

    SUBCASE("an unchanged span resolves the exact spot after renumbering")
    {
        // The head grew by 30 samples: every index shifts, the spot does not.
        std::vector<cv::Vec3f> newLine;
        for (int i = 0; i < 30; ++i) {
            newLine.push_back({static_cast<float>(i) - 30.0f, 0.0f, 0.0f});
        }
        newLine.insert(newLine.end(), oldLine.begin(), oldLine.end());
        const std::vector<GeneratedLineCarryControl> newControls{{1, 50.0}, {2, 150.0}};
        CHECK(carriedGeneratedLinePosition(oldLine, oldControls, newLine, newControls, 60.25) ==
              doctest::Approx(90.25));
        // Before the first control: the same distance ahead of A.
        CHECK(carriedGeneratedLinePosition(oldLine, oldControls, newLine, newControls, 10.0) ==
              doctest::Approx(40.0));
    }

    SUBCASE("a deleted bracketing control falls back to the next matched one")
    {
        // Old controls A, B, C; the position sits between B and C and C is
        // deleted by the edit: B anchors it as a tail position.
        const std::vector<GeneratedLineCarryControl> threeControls{
            {1, 20.0}, {2, 120.0}, {3, 160.0}};
        const std::vector<GeneratedLineCarryControl> newControls{{1, 20.0}, {2, 120.0}};
        CHECK(carriedGeneratedLinePosition(oldLine, threeControls, oldLine, newControls, 140.0) ==
              doctest::Approx(140.0));
    }

    SUBCASE("without matchable controls the plain remap decides")
    {
        const std::vector<cv::Vec3f> newLine = wraps(130, 0.0f);
        const std::vector<GeneratedLineCarryControl> anonymous{{0, 20.0}, {0, 120.0}};
        CHECK(carriedGeneratedLinePosition(oldLine, anonymous, newLine, anonymous, 180.0) ==
              doctest::Approx(remappedGeneratedLinePosition(oldLine, newLine, 180.0)));
        // An identity published twice is ambiguous and anchors nothing,
        // on either side.
        const std::vector<GeneratedLineCarryControl> twice{{2, 20.0}, {2, 120.0}};
        CHECK(carriedGeneratedLinePosition(oldLine, oldControls, newLine, twice, 180.0) ==
              doctest::Approx(remappedGeneratedLinePosition(oldLine, newLine, 180.0)));
        // Old side: {1 at 20, 1 at 120} against {1 at 120} on an identical
        // line would otherwise bracket position 60 with one control on both
        // sides and collapse it onto that control.
        const std::vector<GeneratedLineCarryControl> oldTwice{{1, 20.0}, {1, 120.0}};
        const std::vector<GeneratedLineCarryControl> oneNew{{1, 120.0}};
        CHECK(carriedGeneratedLinePosition(oldLine, oldTwice, oldLine, oneNew, 60.0) ==
              doctest::Approx(60.0));
        // A duplicate with a non-finite position still counts as one.
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const std::vector<GeneratedLineCarryControl> oneOld{{1, 20.0}};
        const std::vector<GeneratedLineCarryControl> newWithNan{{1, 120.0}, {1, nan}};
        CHECK(carriedGeneratedLinePosition(oldLine, oneOld, oldLine, newWithNan, 60.0) ==
              doctest::Approx(60.0));
        const std::vector<GeneratedLineCarryControl> oldWithInf{
            {1, 20.0}, {1, std::numeric_limits<double>::infinity()}};
        CHECK(carriedGeneratedLinePosition(oldLine, oldWithInf, oldLine, oneOld, 60.0) ==
              doctest::Approx(60.0));
    }

    SUBCASE("degenerate inputs")
    {
        const std::vector<cv::Vec3f> empty;
        const std::vector<GeneratedLineCarryControl> newControls{{1, 20.0}, {2, 120.0}};
        CHECK(carriedGeneratedLinePosition(oldLine, oldControls, empty, newControls, 180.0) ==
              doctest::Approx(0.0));
        CHECK(carriedGeneratedLinePosition(empty, oldControls, oldLine, newControls, 180.0) ==
              doctest::Approx(180.0));
        const double nan = std::numeric_limits<double>::quiet_NaN();
        CHECK(carriedGeneratedLinePosition(oldLine, oldControls, oldLine, newControls, nan) ==
              doctest::Approx(0.0));
        // Past the old line's end: clamped onto it first, then carried.
        CHECK(carriedGeneratedLinePosition(oldLine, oldControls, oldLine, newControls, 999.0) ==
              doctest::Approx(201.0));
    }
}

TEST_CASE("line annotation bounded remap measures the distance of the position it returns")
{
    using vc3d::line_annotation::remappedGeneratedLinePositionWithin;
    using vc3d::line_annotation::remappedGeneratedLinePositionFromAnchor;
    const std::vector<cv::Vec3f> line{{0.0f, 0.0f, 0.0f}, {10.0f, 0.0f, 0.0f}, {20.0f, 0.0f, 0.0f}};
    const cv::Vec3f anchor(0.0f, 0.0f, 0.0f);

    SUBCASE("a fractional lower bound is the nearest reachable position, at its own distance")
    {
        const auto within = remappedGeneratedLinePositionWithin(line, anchor, 0.0, 0.5, 2.0);
        REQUIRE(within);
        CHECK(within->position == doctest::Approx(0.5));
        CHECK(within->distanceSq == doctest::Approx(25.0));
    }

    SUBCASE("a fractional upper bound clamps the segment projection")
    {
        const auto within = remappedGeneratedLinePositionWithin(line, cv::Vec3f(20.0f, 0.0f, 0.0f), 0.0, 0.0, 1.25);
        REQUIRE(within);
        CHECK(within->position == doctest::Approx(1.25));
        CHECK(within->distanceSq == doctest::Approx(7.5 * 7.5));
    }

    SUBCASE("the whole-line remap is unchanged by the bounds")
    {
        CHECK(remappedGeneratedLinePositionFromAnchor(line, cv::Vec3f(12.5f, 1.0f, 0.0f), 0.0) ==
              doctest::Approx(1.25));
        const auto within = remappedGeneratedLinePositionWithin(line, cv::Vec3f(12.5f, 1.0f, 0.0f), 0.0, 0.0, 2.0);
        REQUIRE(within);
        CHECK(within->position == doctest::Approx(1.25));
        CHECK(within->distanceSq == doctest::Approx(1.0));
    }

    SUBCASE("the spot search splits passages at a vertex outside tolerance")
    {
        using vc3d::line_annotation::generatedLinePositionNearSpot;
        // A V around the anchor: both legs pass within tolerance, the tip
        // is 2 away. The passage nearer the expectation (the second leg)
        // is answered at its own closest point.
        const std::vector<cv::Vec3f> vee{{-10.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {-10.0f, 1.0f, 0.0f}};
        const auto second = generatedLinePositionNearSpot(vee, cv::Vec3f(0.0f, 0.0f, 0.0f), 1.17, 0.0, 2.0, 1.0);
        REQUIRE(second);
        CHECK(*second > 1.1);
        CHECK(*second < 1.2);
        const auto first = generatedLinePositionNearSpot(vee, cv::Vec3f(0.0f, 0.0f, 0.0f), 0.8, 0.0, 2.0, 1.0);
        REQUIRE(first);
        CHECK(*first == doctest::Approx(10.0 / 12.0));
        // Nothing within tolerance: nullopt.
        CHECK_FALSE(generatedLinePositionNearSpot(vee, cv::Vec3f(0.0f, 5.0f, 0.0f), 1.0, 0.0, 2.0, 1.0));
        // A single position.
        CHECK(generatedLinePositionNearSpot(vee, cv::Vec3f(2.0f, 0.5f, 0.0f), 1.0, 1.0, 1.0, 1.0) ==
              doctest::Approx(1.0));
    }

    SUBCASE("non-finite vertices in the bounds are skipped, none at all gives nothing")
    {
        std::vector<cv::Vec3f> holed = line;
        holed[1] = {std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f};
        const auto within = remappedGeneratedLinePositionWithin(holed, anchor, 0.0, 0.0, 2.0);
        REQUIRE(within);
        CHECK(within->position == doctest::Approx(0.0));
        CHECK_FALSE(remappedGeneratedLinePositionWithin(holed, anchor, 0.0, 0.5, 1.5));
    }
}

TEST_CASE("line annotation placement resolved by the rebase anchors the carry on the displayed line")
{
    using namespace vc3d::line_annotation;
    // Displayed line (revision 6): the two-wrap fiber of the carry tests,
    // controls A (identity 1, index 20) and B (identity 2, index 120).
    std::vector<cv::Vec3f> displayedLine;
    for (int i = 0; i <= 100; ++i) {
        displayedLine.push_back({static_cast<float>(i), 0.0f, 0.0f});
    }
    for (int i = 101; i <= 201; ++i) {
        displayedLine.push_back({static_cast<float>(100 - (i - 101)), 5.0f, 0.0f});
    }
    const auto marker = [](uint64_t identity, cv::Vec3f point, double linePosition, double arc,
                           double total, uint64_t revision) {
        GeneratedOverlay::ControlPointMarker m;
        m.identity = identity;
        m.point = point;
        m.linePosition = linePosition;
        m.arcLength = arc;
        m.lineArcLength = total;
        m.lineRevision = revision;
        return m;
    };
    const std::vector<GeneratedOverlay::ControlPointMarker> displayedControls{
        marker(1, {20.0f, 0.0f, 0.0f}, 20.0, 20.0, 201.0, 6),
        marker(2, {81.0f, 5.0f, 0.0f}, 120.0, 120.0, 201.0, 6)};
    const GeneratedOverviewLayout displayed = generatedOverviewSettledLayout(displayedControls, displayedLine);
    REQUIRE_FALSE(displayed.empty());

    // The user placed C at displayed index 150 (x=51 on the return pass);
    // the controller's splice (revision 7) numbers it 4 among coarse
    // provisional samples, so its published linePosition is NOT a displayed
    // position.
    GeneratedPendingPlacement placement;
    placement.point = {51.0f, 5.0f, 0.0f};
    placement.anchor = placement.point;
    placement.token = 11;
    placement.arcLength = 150.0;
    placement.linePosition = 150.0;
    placement.lineRevision = 6;
    std::vector<GeneratedPendingPlacement> pending{placement};
    std::vector<GeneratedPendingPlacement> resolved;
    const std::vector<GeneratedOverlay::ControlPointMarker> published{
        marker(1, {20.0f, 0.0f, 0.0f}, 1.0, 20.0, 160.0, 7),
        marker(2, {81.0f, 5.0f, 0.0f}, 3.0, 120.0, 160.0, 7),
        marker(3, {51.0f, 5.0f, 0.0f}, 4.0, 150.0, 160.0, 7)};
    (void)generatedDisplaySpaceControlArcLengths(displayed, published, pending, resolved, 6, displayedLine);
    REQUIRE(resolved.size() == 1);
    CHECK(resolved[0].identity == 3);
    CHECK(resolved[0].lineRevision == 6);
    CHECK(resolved[0].linePosition == doctest::Approx(150.0));  // displayed, not 4
    CHECK(pending.empty());

    // The landing (revision 8): span B..C re-solved with a bulge (longer),
    // the tail past C re-traced sideways (the old spot is gone). The cursor
    // was 30 voxels past C on the displayed line (index 180).
    std::vector<cv::Vec3f> landed(displayedLine.begin(), displayedLine.begin() + 151);
    for (int i = 122; i <= 148; ++i) {
        landed[static_cast<size_t>(i)][1] = 5.0f + 0.5f * (14.0f - std::abs(static_cast<float>(i - 135)));
    }
    for (int k = 1; k <= 50; ++k) {
        landed.push_back({static_cast<float>(51 - k), 7.0f, 0.0f});
    }
    const std::vector<GeneratedLineCarryControl> landedControls{{1, 20.0}, {2, 120.0}, {3, 150.0}};
    const std::vector<double> landedArcs = generatedCumulativeArcLength(landed);
    const double pastC = generatedLinePositionAtArcLength(landedArcs, landedArcs[150] + 30.0);
    const double pastB = generatedLinePositionAtArcLength(landedArcs, landedArcs[120] + 60.0);
    REQUIRE(pastC > pastB + 2.0);

    // The old-side anchors: the displayed line's controls plus the resolved
    // placement, for the displayed revision only.
    std::vector<GeneratedLineCarryControl> anchors = generatedLineCarryControls(displayedControls);
    appendGeneratedLineCarryControls(anchors, resolved, 6);
    REQUIRE(anchors.size() == 3);
    CHECK(carriedGeneratedLinePosition(displayedLine, anchors, landed, landedControls, 180.0) ==
          doctest::Approx(pastC));
    // For another revision the entry is rejected and the cursor would be
    // measured from B.
    std::vector<GeneratedLineCarryControl> wrongRevision = generatedLineCarryControls(displayedControls);
    appendGeneratedLineCarryControls(wrongRevision, resolved, 5);
    CHECK(wrongRevision.size() == 2);
    CHECK(carriedGeneratedLinePosition(displayedLine, wrongRevision, landed, landedControls, 180.0) ==
          doctest::Approx(pastB));
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

TEST_CASE("segment interpolation mode uses regime-specific minimum spans")
{
    using vc3d::line_annotation::FiberOptimizationMode;
    using vc3d::line_annotation::SegmentInterpolationGoal;
    using vc3d::line_annotation::SegmentInterpolationMode;
    using vc3d::line_annotation::resolveSegmentInterpolationMode;
    using vc3d::line_annotation::segmentInterpolationCutoffs;

    vc::fiber_tracer::FiberTraceConfig traceConfig;  // stepVoxels 4
    vc::lasagna::LineOptimizationConfig lasagnaConfig;
    lasagnaConfig.segmentLength = 31.5;
    const auto cutoffs = segmentInterpolationCutoffs(traceConfig, lasagnaConfig, 1.0);
    CHECK(cutoffs.traceMinimumSpanBaseVoxels == doctest::Approx(48.0));
    CHECK(cutoffs.lasagnaMinimumSpanBaseVoxels == doctest::Approx(63.0));

    SUBCASE("trace-to-base scale converts the tracer step into base voxels")
    {
        const auto scaled = segmentInterpolationCutoffs(traceConfig, lasagnaConfig, 2.0);
        CHECK(scaled.traceMinimumSpanBaseVoxels == doctest::Approx(96.0));
        CHECK(scaled.lasagnaMinimumSpanBaseVoxels == doctest::Approx(63.0));
    }

    SUBCASE("global goal in trace mode uses the trace minimum")
    {
        const auto mode = FiberOptimizationMode::NativeFiberTrace3d;
        CHECK(resolveSegmentInterpolationMode(SegmentInterpolationGoal::Global, mode, 47.9, cutoffs) ==
              SegmentInterpolationMode::Cspline);
        CHECK(resolveSegmentInterpolationMode(SegmentInterpolationGoal::Global, mode, 48.0, cutoffs) ==
              SegmentInterpolationMode::Trace);
        CHECK(resolveSegmentInterpolationMode(SegmentInterpolationGoal::Global, mode, 100.0, cutoffs) ==
              SegmentInterpolationMode::Trace);
    }

    SUBCASE("global goal in lasagna mode uses the lasagna minimum")
    {
        const auto mode = FiberOptimizationMode::Lasagna;
        CHECK(resolveSegmentInterpolationMode(SegmentInterpolationGoal::Global, mode, 62.9, cutoffs) ==
              SegmentInterpolationMode::Cspline);
        CHECK(resolveSegmentInterpolationMode(SegmentInterpolationGoal::Global, mode, 63.0, cutoffs) ==
              SegmentInterpolationMode::Lasagna);
    }

    SUBCASE("explicit goals ignore the cutoffs")
    {
        const auto mode = FiberOptimizationMode::NativeFiberTrace3d;
        CHECK(resolveSegmentInterpolationMode(SegmentInterpolationGoal::Trace, mode, 10.0, cutoffs) ==
              SegmentInterpolationMode::Trace);
        CHECK(resolveSegmentInterpolationMode(SegmentInterpolationGoal::Lasagna, mode, 10.0, cutoffs) ==
              SegmentInterpolationMode::Lasagna);
        CHECK(resolveSegmentInterpolationMode(SegmentInterpolationGoal::Cspline, mode, 1000.0, cutoffs) ==
              SegmentInterpolationMode::Cspline);
    }

    SUBCASE("degenerate configs are rejected")
    {
        vc::fiber_tracer::FiberTraceConfig zeroStep = traceConfig;
        zeroStep.stepVoxels = 0.0;
        CHECK_THROWS_AS(segmentInterpolationCutoffs(zeroStep, lasagnaConfig, 1.0),
                        std::invalid_argument);
        vc::lasagna::LineOptimizationConfig zeroSegment = lasagnaConfig;
        zeroSegment.segmentLength = 0.0;
        CHECK_THROWS_AS(segmentInterpolationCutoffs(traceConfig, zeroSegment, 1.0),
                        std::invalid_argument);
        CHECK_THROWS_AS(resolveSegmentInterpolationMode(SegmentInterpolationGoal::Global,
                                                        FiberOptimizationMode::NativeFiberTrace3d,
                                                        -1.0,
                                                        cutoffs),
                        std::invalid_argument);
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

TEST_CASE("line annotation failed multi fiber save restores overwritten targets")
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
    // The overwritten first target is restored from its recovery copy, the
    // second was never replaced, and no artifact of the undo survives.
    CHECK(result.recoveryFiles.empty());
    CHECK(readText(first) == "{\"old\":\"a\"}\n");
    CHECK(readText(second) == "{\"old\":\"b\"}\n");
    CHECK(recoveryFilesIn(dir).empty());
    std::filesystem::remove_all(dir);
}

TEST_CASE("line annotation failed save restores an overwritten peer after a later payload fails")
{
    const auto dir = makeTempSaveDir("peer_restore");
    const auto merged = dir / "fiber_merged.json";
    const auto peerA = dir / "fiber_peer_a.json";
    const auto peerB = dir / "fiber_peer_b.json";
    const auto original = dir / "fiber_original.json";
    writeText(peerA, "{\"peer\":\"a\"}\n");
    writeText(peerB, "{\"peer\":\"b\"}\n");
    writeText(original, "{\"original\":true}\n");

    // Payload order: new fiber, peer A (overwritten), peer B; fail right after
    // peer A landed, with the original already retired.
    setenv("VC3D_FIBER_SAVE_FAIL_STAGE", "replace:1", 1);
    const auto result = vc3d::line_annotation::runFiberSaveJob(
        17,
        {{1, 1, merged, nlohmann::json{{"merged", true}}},
         {2, 5, peerA, nlohmann::json{{"peer", "a-redirected"}}},
         {3, 5, peerB, nlohmann::json{{"peer", "b-redirected"}}}},
        {original});
    unsetenv("VC3D_FIBER_SAVE_FAIL_STAGE");

    CHECK_FALSE(result.ok);
    CHECK(result.recoveryFiles.empty());
    CHECK_FALSE(std::filesystem::exists(merged));
    CHECK(readText(peerA) == "{\"peer\":\"a\"}\n");
    CHECK(readText(peerB) == "{\"peer\":\"b\"}\n");
    CHECK(readText(original) == "{\"original\":true}\n");
    CHECK(recoveryFilesIn(dir).empty());
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        CHECK(entry.path().filename().string().find(".tmp.") == std::string::npos);
    }
    const auto retiredDir = dir / ".retired";
    if (std::filesystem::exists(retiredDir)) {
        CHECK(std::filesystem::is_empty(retiredDir));
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("line annotation write-stage failure leaves no temp file and no change")
{
    const auto dir = makeTempSaveDir("write_failure");
    const auto first = dir / "fiber_a.json";
    const auto second = dir / "fiber_b.json";
    writeText(first, "{\"old\":\"a\"}\n");

    setenv("VC3D_FIBER_SAVE_FAIL_STAGE", "write:1", 1);
    const auto result = vc3d::line_annotation::runFiberSaveJob(
        18,
        {{1, 1, first, nlohmann::json{{"new", "a"}}},
         {2, 1, second, nlohmann::json{{"new", "b"}}}});
    unsetenv("VC3D_FIBER_SAVE_FAIL_STAGE");

    CHECK_FALSE(result.ok);
    CHECK(result.recoveryFiles.empty());
    CHECK(readText(first) == "{\"old\":\"a\"}\n");
    CHECK_FALSE(std::filesystem::exists(second));
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        CHECK(entry.path().filename().string().find(".tmp.") == std::string::npos);
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("line annotation retire-stage failure restores the earlier retirement")
{
    const auto dir = makeTempSaveDir("retire_stage");
    const auto first = dir / "fiber_a.json";
    const auto second = dir / "fiber_b.json";
    const auto target = dir / "fiber_new.json";
    writeText(first, "{\"a\":true}\n");
    writeText(second, "{\"b\":true}\n");

    setenv("VC3D_FIBER_SAVE_FAIL_STAGE", "retire:1", 1);
    const auto result = vc3d::line_annotation::runFiberSaveJob(
        19, {{1, 1, target, nlohmann::json{{"new", true}}}}, {first, second});
    unsetenv("VC3D_FIBER_SAVE_FAIL_STAGE");

    CHECK_FALSE(result.ok);
    CHECK(result.recoveryFiles.empty());
    CHECK(readText(first) == "{\"a\":true}\n");
    CHECK(readText(second) == "{\"b\":true}\n");
    CHECK_FALSE(std::filesystem::exists(target));
    const auto retiredDir = dir / ".retired";
    if (std::filesystem::exists(retiredDir)) {
        CHECK(std::filesystem::is_empty(retiredDir));
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("line annotation failed restore is reported as recovery required")
{
    const auto dir = makeTempSaveDir("restore_failure");
    const auto first = dir / "fiber_a.json";
    const auto second = dir / "fiber_b.json";
    writeText(first, "{\"old\":\"a\"}\n");
    writeText(second, "{\"old\":\"b\"}\n");

    // Both replaced, then the restore of the first is made to fail.
    setenv("VC3D_FIBER_SAVE_FAIL_STAGE", "restore:0", 1);
    const auto result = vc3d::line_annotation::runFiberSaveJob(
        20,
        {{1, 1, first, nlohmann::json{{"new", "a"}}},
         {2, 1, second, nlohmann::json{{"new", "b"}}}});
    unsetenv("VC3D_FIBER_SAVE_FAIL_STAGE");

    CHECK_FALSE(result.ok);
    CHECK(result.error.find("could not restore") != std::string::npos);
    // The second target was restored; the first keeps the new content and
    // its recovery copy is reported so the caller can ask for recovery.
    CHECK(readText(second) == "{\"old\":\"b\"}\n");
    CHECK(readText(first).find("\"new\": \"a\"") != std::string::npos);
    REQUIRE(result.recoveryFiles.size() == 1);
    CHECK(std::filesystem::exists(result.recoveryFiles.front()));
    CHECK(readText(result.recoveryFiles.front()) == "{\"old\":\"a\"}\n");
    CHECK(recoveryFilesIn(dir).size() == 1);
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
    // Neither brand-new target survives the aborted batch (a pre-existing
    // target is restored from its recovery copy instead).
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

TEST_CASE("fiber mode sends a failed short global-goal trace straight to cspline")
{
    // Trace minimum 12 * 4 = 48 vx, Lasagna minimum 2 * 40 = 80 vx. Two
    // 64 vx Global-goal spans: both long enough to trace; the second one's
    // trace throws (invalid prediction at x = 96) and, being shorter than the
    // Lasagna minimum, goes to cspline instead of the Lasagna fallback.
    FiberModeNormalSampler normals;
    FiberModePrediction predictions(96.0);
    vc3d::line_annotation::FiberModeOptimizationRequest request;
    request.controlPoints = {
        {2.0, {0.0, 0.0, 0.0}, true, 2},
        {18.0, {64.0, 0.0, 0.0}, false, 18},
        {34.0, {128.0, 0.0, 0.0}, false, 34},
    };
    request.controlPoints[0].segmentToNext.emplace();
    request.controlPoints[0].segmentToNext->interpGoal =
        vc3d::line_annotation::SegmentInterpolationGoal::Global;
    request.controlPoints[1].segmentToNext.emplace();
    request.controlPoints[1].segmentToNext->interpGoal =
        vc3d::line_annotation::SegmentInterpolationGoal::Global;
    for (int x = -8; x <= 136; x += 4) {
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
    request.globalMode = vc3d::line_annotation::FiberOptimizationMode::NativeFiberTrace3d;
    request.traceConfig.stepVoxels = 4.0;
    request.traceConfig.coneAngleDegrees = 0.0;
    request.traceConfig.beamWidth = 1;
    request.traceConfig.maxStepFactor = 2.0;
    request.traceConfig.smoothnessWeight = 0.0;
    request.traceConfig.smoothnessNormalWeight = 0.0;
    request.traceConfig.smoothnessTangentWeight = 0.0;
    request.traceConfig.cumulativeSmoothnessTangentWeight = 0.0;
    request.lasagnaConfig.segmentsPerSide = 2;
    request.lasagnaConfig.segmentLength = 40.0;
    request.lasagnaConfig.maxIterations = 20;
    request.lasagnaConfig.printSolverProgress = false;

    const auto result =
        vc3d::line_annotation::optimizeFiberWithNativeFallback(
            std::move(request));

    REQUIRE(result.controlPoints.size() == 3);
    CHECK(result.nativeSegments == 1);
    CHECK(result.lasagnaFallbackSegments == 0);
    CHECK(result.csplineFallbackSegments == 1);
    REQUIRE(result.controlPoints[0].segmentToNext.has_value());
    CHECK(vc3d::line_annotation::isAcceptedNativeTrace(
        result.controlPoints[0].segmentToNext));
    REQUIRE(result.controlPoints[1].segmentToNext.has_value());
    const auto& gated = *result.controlPoints[1].segmentToNext;
    CHECK(gated.interpMode == vc3d::line_annotation::SegmentInterpolationMode::Cspline);
    CHECK(gated.interpGoal == vc3d::line_annotation::SegmentInterpolationGoal::Global);
    // The trace failure stays on the span; the message records the gate.
    CHECK_FALSE(vc3d::line_annotation::isAcceptedNativeTrace(
        result.controlPoints[1].segmentToNext));
    CHECK_FALSE(gated.failureCode.empty());
    CHECK(gated.message.find("short span, trace -> cspline") != std::string::npos);
    CHECK(result.optimization.report.message.find("cspline_fallback_segments=1") !=
          std::string::npos);
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

TEST_CASE("fiber mode can return only the inclusive control span")
{
    FiberModeNormalSampler normals;
    vc3d::line_annotation::FiberModeOptimizationRequest request;
    request.controlPoints = {
        {2.0, {0.0, 0.0, 0.0}, true, 2},
        {6.0, {16.0, 0.0, 0.0}, false, 6},
    };
    request.controlPoints.front().segmentToNext.emplace();
    request.controlPoints.front().segmentToNext->interpGoal =
        vc3d::line_annotation::SegmentInterpolationGoal::Cspline;
    for (int x = -8; x <= 24; x += 4) {
        request.linePointsBase.push_back({static_cast<double>(x), 0.0, 0.0});
    }
    request.baseNormalSampler = &normals;
    request.globalMode =
        vc3d::line_annotation::FiberOptimizationMode::NativeFiberTrace3d;
    request.retainOpenTails = false;
    request.extrapolationDistanceBaseVoxels = 8.0;
    request.lasagnaConfig.segmentsPerSide = 2;
    request.lasagnaConfig.segmentLength = 4.0;
    request.lasagnaConfig.maxIterations = 20;
    request.lasagnaConfig.printSolverProgress = false;

    const auto result = vc3d::line_annotation::optimizeFiberWithNativeFallback(
        std::move(request));

    REQUIRE(result.optimization.line.points.size() >= 2);
    CHECK(cv::norm(result.optimization.line.points.front().position -
                   cv::Vec3d{0.0, 0.0, 0.0}) < 1.0e-12);
    CHECK(cv::norm(result.optimization.line.points.back().position -
                   cv::Vec3d{16.0, 0.0, 0.0}) < 1.0e-12);
    CHECK(result.nativeExtrapolations == 0);
    CHECK(result.lasagnaFallbackExtrapolations == 0);
    REQUIRE(result.controlPoints.size() == 2);
    CHECK(result.controlPoints.front().optimizedIndex == 0);
    CHECK(result.controlPoints.back().optimizedIndex ==
          static_cast<int>(result.optimization.line.points.size()) - 1);
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

TEST_CASE("controlled span keeps only the line between the outer control points")
{
    using vc3d::line_annotation::linePointsBetweenOuterControlPoints;
    const std::vector<cv::Vec3d> line{
        {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {4.0, 0.0, 0.0}};

    SUBCASE("interior controls drop both extrapolated tails")
    {
        const auto span = linePointsBetweenOuterControlPoints(line, {line[1], line[3]});
        REQUIRE(span.size() == 3);
        CHECK(span.front() == line[1]);
        CHECK(span.back() == line[3]);
    }
    SUBCASE("one control is a single point, so nothing of the fiber is drawn")
    {
        const auto span = linePointsBetweenOuterControlPoints(line, {line[2]});
        REQUIRE(span.size() == 1);
        CHECK(span.front() == line[2]);
    }
    SUBCASE("a control off the line has no controlled span")
    {
        CHECK(linePointsBetweenOuterControlPoints(line, {line[1], {2.5, 0.0, 0.0}}).empty());
    }
    SUBCASE("no controls has no controlled span")
    {
        CHECK(linePointsBetweenOuterControlPoints(line, {}).empty());
    }
    SUBCASE("a line revisiting a coordinate is mapped in loader order, not by nearest point")
    {
        const cv::Vec3d a{0.0, 0.0, 0.0};
        const cv::Vec3d b{2.0, 0.0, 0.0};
        const cv::Vec3d c{2.0, 2.0, 0.0};
        const cv::Vec3d d{0.0, -2.0, 0.0};
        const std::vector<cv::Vec3d> loop{a, b, c, a, d};
        const auto span = linePointsBetweenOuterControlPoints(loop, {b, a});
        // Controls B then A: the second A (index 3), so the span is B, C, A.
        REQUIRE(span.size() == 3);
        CHECK(span[0] == b);
        CHECK(span[1] == c);
        CHECK(span[2] == a);
    }
}

TEST_CASE("Generated markers map every position into the downsampled viewer")
{
    using namespace vc3d::line_annotation;
    for (const double scale : {1.0, 0.25}) {
        GeneratedOverlay::BranchLinkMarker branch;
        branch.localControlPoint = {120, 240, 360};
        branch.linkedControlPoint = {160, 280, 400};
        branch.planePoint = branch.linkedControlPoint;
        branch.localDirection = {1, 0, 0};
        branch.linkedDirection = {0, 1, 0};
        branch.linkedFiberId = 42;
        scaleGeneratedMarkerForVolume(branch, scale);
        CHECK(branch.localControlPoint == cv::Vec3f(120, 240, 360) * scale);
        CHECK(branch.linkedControlPoint == cv::Vec3f(160, 280, 400) * scale);
        CHECK(branch.planePoint == branch.linkedControlPoint);
        CHECK(branch.localDirection == cv::Vec3f(1, 0, 0));
        CHECK(branch.linkedDirection == cv::Vec3f(0, 1, 0));
        CHECK(branch.linkedFiberId == 42);

        GeneratedOverlay::PredSnapMarker snap;
        snap.controlPoint = {120, 240, 360};
        snap.snapPoint = {124, 248, 372};
        snap.linePosition = 2.5;
        snap.controlIndex = 3;
        snap.manual = true;
        scaleGeneratedMarkerForVolume(snap, scale);
        CHECK(snap.controlPoint == cv::Vec3f(120, 240, 360) * scale);
        CHECK(snap.snapPoint == cv::Vec3f(124, 248, 372) * scale);
        CHECK(snap.linePosition == 2.5);
        CHECK(snap.controlIndex == 3);
        CHECK(snap.manual);
    }
}

TEST_CASE("Winding queries use the center frame at every volume level")
{
    using namespace vc3d::line_annotation;
    std::vector<cv::Vec3f> basePoints;
    // Off-origin, z-dependent center exposes both XY and Z frame mismatches.
    const auto towardCenter = [](const cv::Vec3f& p) -> cv::Vec3f {
        return {1000.0f + p[2] - p[0], 2000.0f - p[2] - p[1], 0};
    };
    for (int i = 0; i < 40; ++i) {
        const float z = 100.0f + i * 4.0f;
        const float angle = i * 0.3f;
        basePoints.push_back({1000.0f + z + 100.0f * std::cos(angle),
                              2000.0f - z + 100.0f * std::sin(angle), z});
    }
    const auto expected = unwrappedGeneratedWindingAngles(basePoints, towardCenter);
    auto viewerPoints = basePoints;
    for (auto& point : viewerPoints) {
        point *= 0.25f;
    }
    const auto actual = unwrappedGeneratedWindingAngles(viewerPoints, towardCenter, 4.0f);
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        CHECK(actual[i] == doctest::Approx(expected[i]));
    }
    CHECK(generatedLineIndexRangeWithinWinding(actual, actual.size(), 20.0,
                                               kGeneratedSideCutHalfWrapAngle) ==
          generatedLineIndexRangeWithinWinding(expected, expected.size(), 20.0,
                                               kGeneratedSideCutHalfWrapAngle));
}

TEST_CASE("control point tags round trip and travel with the point")
{
    using namespace vc3d::line_annotation;

    // Untagged points serialize exactly as before: no "tags" key at all.
    StoredControlPoint untagged{{1.0, 2.0, 3.0}};
    CHECK(!storedControlPointToJson(untagged).contains("tags"));

    StoredControlPoint tagged{{1.0, 2.0, 3.0}};
    CHECK(setControlPointTag(tagged.tags, kKollesisTerminationTag, true));
    CHECK(!setControlPointTag(tagged.tags, kKollesisTerminationTag, true));
    CHECK(hasControlPointTag(tagged.tags, kKollesisTerminationTag));
    const auto json = storedControlPointToJson(tagged);
    REQUIRE(json.contains("tags"));
    CHECK(json["tags"] == nlohmann::json::array({"kollesis_termination"}));
    const auto parsed = storedControlPointFromJson(json, 3);
    CHECK(parsed.tags == tagged.tags);
    CHECK(setControlPointTag(tagged.tags, kKollesisTerminationTag, false));
    CHECK(tagged.tags.empty());
    CHECK(!setControlPointTag(tagged.tags, kKollesisTerminationTag, false));

    // Normalised on read: trimmed, blanks dropped, unique, sorted.
    const nlohmann::json messy{{"position", {1.0, 2.0, 3.0}},
                               {"tags", {" b ", "a", "", "a"}}};
    CHECK(storedControlPointFromJson(messy, 3).tags ==
          std::vector<std::string>{"a", "b"});
    const nlohmann::json notArray{{"position", {1.0, 2.0, 3.0}},
                                  {"tags", "kollesis_termination"}};
    CHECK_THROWS_AS(storedControlPointFromJson(notArray, 3), std::runtime_error);
    const nlohmann::json notStrings{{"position", {1.0, 2.0, 3.0}}, {"tags", {1}}};
    CHECK_THROWS_AS(storedControlPointFromJson(notStrings, 3), std::runtime_error);
    // Any other per-point field is still rejected.
    const nlohmann::json unknown{{"position", {1.0, 2.0, 3.0}},
                                 {"kollesis_termination", true}};
    CHECK_THROWS_AS(storedControlPointFromJson(unknown, 3), std::runtime_error);

    // The shared core reader (atlas, tracer CLI, inspect tools) accepts the
    // field and rejects the same malformed shapes.
    const nlohmann::json root = {{"control_points", nlohmann::json::array({json})}};
    CHECK_NOTHROW(vc::fiber_tracer::vc3dFiberPointArrayFromJson(
        root, "control_points", 3, "test fiber"));
    const nlohmann::json badTags = {{"control_points", nlohmann::json::array({notStrings})}};
    CHECK_THROWS_AS(vc::fiber_tracer::vc3dFiberPointArrayFromJson(
                        badTags, "control_points", 3, "test fiber"),
                    std::runtime_error);
    const nlohmann::json badField = {{"control_points", nlohmann::json::array({unknown})}};
    CHECK_THROWS_AS(vc::fiber_tracer::vc3dFiberPointArrayFromJson(
                        badField, "control_points", 3, "test fiber"),
                    std::runtime_error);

    // Edits: the click that collapses tagged points keeps the union of their
    // tags, the optimizer round trip and a reverse carry them unchanged.
    std::vector<LineControlPoint> controls{
        LineControlPoint{0.0, cv::Vec3d(0.0, 0.0, 0.0), true, 0},
        LineControlPoint{5.0, cv::Vec3d(5.0, 0.0, 0.0), false, 5},
        LineControlPoint{10.0, cv::Vec3d(10.0, 0.0, 0.0), false, 10}};
    controls[1].tags = {"kollesis_termination"};
    controls[2].tags = {"other"};
    const auto collapse =
        collapseControlPointsAtClick(controls, {1, 2}, 7.0, cv::Vec3d(7.0, 0.0, 0.0));
    REQUIRE(collapse.replacementIndex < collapse.controlPoints.size());
    CHECK(collapse.controlPoints[collapse.replacementIndex].tags ==
          std::vector<std::string>{"kollesis_termination", "other"});
    CHECK(collapse.controlPoints[0].tags.empty());
    const auto fresh =
        collapseControlPointsAtClick(controls, {}, 7.0, cv::Vec3d(7.0, 0.0, 0.0));
    REQUIRE(fresh.replacementIndex < fresh.controlPoints.size());
    CHECK(fresh.controlPoints[fresh.replacementIndex].tags.empty());

    const auto merged = mergeOptimizerControlPoints(optimizerControlPoints(controls), controls);
    REQUIRE(merged.size() == controls.size());
    CHECK(merged[1].tags == controls[1].tags);
    CHECK(merged[2].tags == controls[2].tags);

    std::vector<StoredControlPoint> stored{StoredControlPoint{cv::Vec3d(0.0, 0.0, 0.0)},
                                           StoredControlPoint{cv::Vec3d(5.0, 0.0, 0.0)},
                                           StoredControlPoint{cv::Vec3d(10.0, 0.0, 0.0)}};
    stored[2].tags = {"kollesis_termination"};
    const auto reversed = reversedStoredControlPoints(stored);
    REQUIRE(reversed.size() == 3);
    CHECK(reversed[0].tags == stored[2].tags);
    CHECK(reversed[2].tags.empty());
}

TEST_CASE("kollesis terminations sit on fiber ends and block placement beyond them")
{
    using namespace vc3d::line_annotation;
    using Marker = GeneratedOverlay::ControlPointMarker;
    const auto marker = [](size_t index, double linePosition, bool tagged) {
        Marker m;
        m.controlIndex = index;
        m.linePosition = linePosition;
        m.isKollesisTermination = tagged;
        return m;
    };

    // Session order need not be line order: index 2 is the first point.
    std::vector<Marker> markers{marker(0, 10.0, false), marker(1, 25.0, false),
                                marker(2, 3.0, false), marker(3, 40.0, false)};
    CHECK(generatedControlPointIsEndpoint(markers, 2));
    CHECK(generatedControlPointIsEndpoint(markers, 3));
    CHECK(!generatedControlPointIsEndpoint(markers, 0));
    CHECK(!generatedControlPointIsEndpoint(markers, 1));
    CHECK(!generatedControlPointIsEndpoint(markers, 9));
    CHECK(generatedControlPointIsEndpoint({marker(0, 5.0, false)}, 0));
    CHECK(!generatedControlPointIsEndpoint({}, 0));
    // A marker without a line position neither is an end nor hides one.
    markers.push_back(marker(4, std::numeric_limits<double>::quiet_NaN(), true));
    CHECK(!generatedControlPointIsEndpoint(markers, 4));
    CHECK(generatedControlPointIsEndpoint(markers, 3));

    // Nothing tagged: nothing blocked.
    CHECK(!generatedLinePositionBeyondKollesisTermination(markers, -5.0));
    CHECK(!generatedLinePositionBeyondKollesisTermination(markers, 100.0));

    // Tagged last point: only positions strictly past it are blocked.
    markers[3].isKollesisTermination = true;
    CHECK(generatedLinePositionBeyondKollesisTermination(markers, 40.5));
    CHECK(generatedLinePositionBeyondKollesisTermination(markers, 100.0));
    CHECK(!generatedLinePositionBeyondKollesisTermination(markers, 40.0));
    CHECK(!generatedLinePositionBeyondKollesisTermination(markers, 30.0));
    CHECK(!generatedLinePositionBeyondKollesisTermination(markers, -5.0));
    CHECK(!generatedLinePositionBeyondKollesisTermination(
        markers, std::numeric_limits<double>::quiet_NaN()));

    // Tagged first point: the other side.
    markers[3].isKollesisTermination = false;
    markers[2].isKollesisTermination = true;
    CHECK(generatedLinePositionBeyondKollesisTermination(markers, 2.9));
    CHECK(!generatedLinePositionBeyondKollesisTermination(markers, 3.0));
    CHECK(!generatedLinePositionBeyondKollesisTermination(markers, 100.0));

    // An interior tag (from an edited file) blocks nothing.
    markers[2].isKollesisTermination = false;
    markers[1].isKollesisTermination = true;
    CHECK(!generatedLinePositionBeyondKollesisTermination(markers, -5.0));
    CHECK(!generatedLinePositionBeyondKollesisTermination(markers, 100.0));

    // The rule the controller applies to session controls directly.
    const std::vector<double> positions{10.0, 25.0, 3.0, 40.0};
    CHECK(generatedLinePositionBeyondTaggedEnd(positions, {false, false, false, true}, 41.0));
    CHECK(!generatedLinePositionBeyondTaggedEnd(positions, {false, false, false, true}, 40.0));
    CHECK(generatedLinePositionBeyondTaggedEnd(positions, {false, false, true, false}, 2.0));
    CHECK(!generatedLinePositionBeyondTaggedEnd(positions, {false, true, false, false}, 100.0));
    // Mismatched vectors mean no tags rather than a misaligned read.
    CHECK(!generatedLinePositionBeyondTaggedEnd(positions, {true}, 100.0));
    CHECK(!generatedLinePositionBeyondTaggedEnd({}, {}, 5.0));
}

TEST_CASE("break tags make gap spans between consecutive tagged points")
{
    using namespace vc3d::line_annotation;

    // The tag string loads through both readers like any other tag, and a
    // point with both tags is a conflict the toggles and the collapse refuse.
    StoredControlPoint stored{{1.0, 2.0, 3.0}};
    CHECK(setControlPointTag(stored.tags, kBreakTag, true));
    const auto json = storedControlPointToJson(stored);
    REQUIRE(json.contains("tags"));
    CHECK(json["tags"] == nlohmann::json::array({"break"}));
    CHECK(storedControlPointFromJson(json, 3).tags == std::vector<std::string>{"break"});
    const nlohmann::json root = {{"control_points", nlohmann::json::array({json})}};
    CHECK_NOTHROW(vc::fiber_tracer::vc3dFiberPointArrayFromJson(
        root, "control_points", 3, "test fiber"));
    CHECK(!controlPointTagsConflict(stored.tags));
    CHECK(controlPointTagsConflict(
        mergedControlPointTags(stored.tags, {kKollesisTerminationTag})));

    // Gap spans over control lists: both endpoints tagged, by index.
    std::vector<LineControlPoint> controls{
        LineControlPoint{0.0, cv::Vec3d(0.0, 0.0, 0.0), true, 0},
        LineControlPoint{10.0, cv::Vec3d(10.0, 0.0, 0.0), false, 10},
        LineControlPoint{20.0, cv::Vec3d(20.0, 0.0, 0.0), false, 20},
        LineControlPoint{30.0, cv::Vec3d(30.0, 0.0, 0.0), false, 30},
        LineControlPoint{40.0, cv::Vec3d(40.0, 0.0, 0.0), false, 40}};
    using Spans = std::vector<std::pair<size_t, size_t>>;
    CHECK(gapSpansForControls(controls).empty());
    controls[1].tags = {kBreakTag};
    // A lone break point makes no gap.
    CHECK(gapSpansForControls(controls).empty());
    controls[2].tags = {kBreakTag};
    CHECK(gapSpansForControls(controls) == Spans{{1, 2}});
    // Three in a row: two consecutive gap spans.
    controls[3].tags = {kBreakTag};
    CHECK(gapSpansForControls(controls) == Spans{{1, 2}, {2, 3}});
    // A break at either fiber end alone changes nothing.
    controls[4].tags = {kBreakTag};
    controls[3].tags.clear();
    CHECK(gapSpansForControls(controls) == Spans{{1, 2}});
    // Neighbours are taken in LINE-POSITION order, not vector order, and the
    // pair is named lower-position first: a session whose controls were
    // reopened out of order still gates and owns the same span the overlays
    // draw.
    {
        std::vector<LineControlPoint> shuffled{
            LineControlPoint{0.0, cv::Vec3d(0.0, 0.0, 0.0), true, 0},
            LineControlPoint{20.0, cv::Vec3d(20.0, 0.0, 0.0), false, 20},
            LineControlPoint{10.0, cv::Vec3d(10.0, 0.0, 0.0), false, 10},
            LineControlPoint{40.0, cv::Vec3d(40.0, 0.0, 0.0), false, 40}};
        shuffled[1].tags = {kBreakTag};
        shuffled[2].tags = {kBreakTag};
        CHECK(gapSpansForControls(shuffled) == Spans{{2, 1}});
        // A break at 40 is not a neighbour of the one at 20 in either order.
        shuffled[3].tags = {kBreakTag};
        shuffled[2].tags.clear();
        CHECK(gapSpansForControls(shuffled) == Spans{{1, 3}});
        // NaN positions take no part.
        shuffled[1].linePosition = std::numeric_limits<double>::quiet_NaN();
        CHECK(gapSpansForControls(shuffled).empty());
    }

    // Overlay markers read the gap from the SPAN tag (hasGapToNext), which
    // syncGapSpanTags puts in step with the break tags; the ranges run from
    // the owner to the next control in line-position order. Then the
    // placement gate: strictly inside a gap is blocked, the endpoints and
    // everything else stay open.
    {
        const auto sync = syncGapSpanTags(controls);
        CHECK(sync.formed == std::vector<size_t>{1});
        CHECK(sync.dissolved.empty());
        CHECK(spanIsGap(controls[1].segmentToNext));
        CHECK(!spanIsGap(controls[0].segmentToNext));
        CHECK(!spanIsGap(controls[2].segmentToNext));
        // Idempotent.
        CHECK(!syncGapSpanTags(controls).changed());
    }
    std::vector<GeneratedOverlay::ControlPointMarker> markers;
    for (size_t i = 0; i < controls.size(); ++i) {
        GeneratedOverlay::ControlPointMarker m;
        m.controlIndex = i;
        m.linePosition = controls[i].linePosition;
        m.isBreak = hasControlPointTag(controls[i].tags, kBreakTag);
        m.hasGapToNext = spanIsGap(controls[i].segmentToNext);
        markers.push_back(m);
    }
    const auto ranges = generatedGapLineRanges(markers);
    REQUIRE(ranges.size() == 1);
    CHECK(ranges[0].first == 10.0);
    CHECK(ranges[0].second == 20.0);
    CHECK(generatedLinePositionInsideGap(ranges, 15.0));
    CHECK(generatedLinePositionInsideGap(ranges, 10.5));
    CHECK(!generatedLinePositionInsideGap(ranges, 10.0));
    CHECK(!generatedLinePositionInsideGap(ranges, 20.0));
    CHECK(!generatedLinePositionInsideGap(ranges, 25.0));
    CHECK(!generatedLinePositionInsideGap(ranges, std::numeric_limits<double>::quiet_NaN()));
    // Dense segments: every one between the two controls is in the gap, the
    // neighbours outside are not.
    CHECK(generatedLineSegmentInGap(10.0, 11.0, ranges));
    CHECK(generatedLineSegmentInGap(19.0, 20.0, ranges));
    CHECK(!generatedLineSegmentInGap(9.0, 10.0, ranges));
    CHECK(!generatedLineSegmentInGap(20.0, 21.0, ranges));

    // Two rings without the span tag are NOT a gap for the overlays: the
    // drawn line reports the span metadata, so a file whose span tag is
    // missing shows rings and no amber line until the load heal fixes it.
    markers[1].hasGapToNext = false;
    CHECK(generatedGapLineRanges(markers).empty());
    markers[1].hasGapToNext = true;
    // The ranges come from the full list: filtering the markers (as the
    // cross-slice overlay does by plane distance) must not move a range's
    // end to the next VISIBLE control. Here 20 is hidden, so a filtered
    // caller would span 10..40.
    std::vector<GeneratedOverlay::ControlPointMarker> filtered{markers[1], markers[4]};
    CHECK(generatedGapLineRanges(filtered)[0].second == 40.0);
    CHECK(generatedGapLineRanges(markers)[0].second == 20.0);
    // Reversed order in the vector does not matter: sorted by line position.
    std::reverse(markers.begin(), markers.end());
    const auto reversedRanges = generatedGapLineRanges(markers);
    REQUIRE(reversedRanges.size() == 1);
    CHECK(reversedRanges[0].first == 10.0);
    CHECK(reversedRanges[0].second == 20.0);

    // Dissolving: removing a break clears the span tag, and the goal policy
    // for structural edits returns a cspline span to global (any other goal
    // is left alone).
    controls[1].segmentToNext->interpGoal = SegmentInterpolationGoal::Cspline;
    controls[2].tags.clear();
    const auto dissolved = applyGapSpanPolicy(controls);
    CHECK(dissolved.dissolved == std::vector<size_t>{1});
    CHECK(!spanIsGap(controls[1].segmentToNext));
    CHECK(controls[1].segmentToNext->interpGoal == SegmentInterpolationGoal::Global);
    controls[2].tags = {kBreakTag};
    controls[1].segmentToNext->interpGoal = SegmentInterpolationGoal::Trace;
    const auto formed = applyGapSpanPolicy(controls);
    CHECK(formed.formed == std::vector<size_t>{1});
    CHECK(controls[1].segmentToNext->interpGoal == SegmentInterpolationGoal::Cspline);
    controls[2].tags.clear();
    controls[1].segmentToNext->interpGoal = SegmentInterpolationGoal::Lasagna;
    applyGapSpanPolicy(controls);
    CHECK(controls[1].segmentToNext->interpGoal == SegmentInterpolationGoal::Lasagna);
}

TEST_CASE("version 4 carries span tags; version 3 spans may not")
{
    using namespace vc3d::line_annotation;
    CHECK(kFiberFormatVersion == 4);

    StoredControlPoint owner{{0.0, 0.0, 0.0}};
    owner.segmentToNext.emplace();
    owner.segmentToNext->interpMode = SegmentInterpolationMode::Lasagna;
    owner.segmentToNext->message = "lasagna";
    // No tags: the span serializes exactly as in version 3 (no "tags" key).
    CHECK(!storedControlPointToJson(owner)["segment_to_next"].contains("tags"));
    CHECK(setControlPointTag(owner.segmentToNext->tags, kGapSpanTag, true));
    const auto json = storedControlPointToJson(owner);
    REQUIRE(json["segment_to_next"].contains("tags"));
    CHECK(json["segment_to_next"]["tags"] == nlohmann::json::array({"gap"}));
    // Round trip under version 4; rejected as an unknown field under 3.
    const auto parsed = storedControlPointFromJson(json, 4);
    REQUIRE(parsed.segmentToNext);
    CHECK(parsed.segmentToNext->tags == std::vector<std::string>{"gap"});
    CHECK(spanIsGap(parsed.segmentToNext));
    CHECK_THROWS_AS(storedControlPointFromJson(json, 3), std::runtime_error);
    // Malformed shapes.
    nlohmann::json bad = json;
    bad["segment_to_next"]["tags"] = "gap";
    CHECK_THROWS_AS(storedControlPointFromJson(bad, 4), std::runtime_error);
    bad["segment_to_next"]["tags"] = nlohmann::json::array({1});
    CHECK_THROWS_AS(storedControlPointFromJson(bad, 4), std::runtime_error);

    // The shared core reader: same rules, per version.
    StoredControlPoint last{{10.0, 0.0, 0.0}};
    const nlohmann::json root = {
        {"control_points", nlohmann::json::array({json, storedControlPointToJson(last)})}};
    CHECK_NOTHROW(vc::fiber_tracer::vc3dFiberPointArrayFromJson(
        root, "control_points", 4, "test fiber"));
    CHECK_THROWS_AS(vc::fiber_tracer::vc3dFiberPointArrayFromJson(
                        root, "control_points", 3, "test fiber"),
                    std::runtime_error);
    const nlohmann::json badRoot = {
        {"control_points", nlohmann::json::array({bad, storedControlPointToJson(last)})}};
    CHECK_THROWS_AS(vc::fiber_tracer::vc3dFiberPointArrayFromJson(
                        badRoot, "control_points", 4, "test fiber"),
                    std::runtime_error);

    // Load heal over stored controls: two consecutive breaks without the
    // span tag get it (and a stale tag goes away); nothing else changes.
    std::vector<StoredControlPoint> stored{StoredControlPoint{cv::Vec3d(0.0, 0.0, 0.0)},
                                           StoredControlPoint{cv::Vec3d(5.0, 0.0, 0.0)},
                                           StoredControlPoint{cv::Vec3d(10.0, 0.0, 0.0)},
                                           StoredControlPoint{cv::Vec3d(15.0, 0.0, 0.0)}};
    for (size_t i = 0; i + 1 < stored.size(); ++i) {
        stored[i].segmentToNext.emplace();
        stored[i].segmentToNext->interpMode = SegmentInterpolationMode::Lasagna;
        stored[i].segmentToNext->message = "lasagna";
    }
    CHECK(!syncGapSpanTags(stored));
    stored[1].tags = {kBreakTag};
    stored[2].tags = {kBreakTag};
    setControlPointTag(stored[0].segmentToNext->tags, kGapSpanTag, true);  // stale
    CHECK(syncGapSpanTags(stored));
    CHECK(spanIsGap(stored[1].segmentToNext));
    CHECK(!spanIsGap(stored[0].segmentToNext));
    CHECK(!spanIsGap(stored[2].segmentToNext));
    CHECK(!syncGapSpanTags(stored));
    // A reverse keeps the gap on the same span (descriptors shift with it).
    const auto reversed = reversedStoredControlPoints(stored);
    REQUIRE(reversed.size() == 4);
    CHECK(spanIsGap(reversed[1].segmentToNext));
    CHECK(!syncGapSpanTags(const_cast<std::vector<StoredControlPoint>&>(reversed)));
}

TEST_CASE("damaged span tag: display only, never on a gap span, alternating ranges")
{
    using namespace vc3d::line_annotation;

    std::vector<LineControlPoint> controls{
        LineControlPoint{0.0, cv::Vec3d(0.0, 0.0, 0.0), true, 0},
        LineControlPoint{10.0, cv::Vec3d(10.0, 0.0, 0.0), false, 10},
        LineControlPoint{20.0, cv::Vec3d(20.0, 0.0, 0.0), false, 20},
        LineControlPoint{30.0, cv::Vec3d(30.0, 0.0, 0.0), false, 30}};
    // setSpanTag never creates a descriptor (a control without one owns no
    // span: the final control, or a peer's control whose span is gone).
    CHECK(!controls[1].segmentToNext);
    CHECK(!setSpanTag(controls[1].segmentToNext, kDamagedSpanTag, true));
    CHECK(!controls[1].segmentToNext);
    controls[1].segmentToNext.emplace();
    controls[1].segmentToNext->interpMode = SegmentInterpolationMode::Lasagna;
    controls[1].segmentToNext->message = "lasagna";
    CHECK(setSpanTag(controls[1].segmentToNext, kDamagedSpanTag, true));
    CHECK(spanIsDamaged(controls[1].segmentToNext));
    CHECK(!setSpanTag(controls[1].segmentToNext, kDamagedSpanTag, true));
    // Clearing a tag on a span without a descriptor is a no-op.
    CHECK(!setSpanTag(controls[2].segmentToNext, kDamagedSpanTag, false));
    CHECK(!controls[2].segmentToNext);
    // Nothing else about the span changed: no goal, no point tags.
    CHECK(controls[1].segmentToNext->interpGoal == SegmentInterpolationGoal::Global);
    CHECK(controls[1].tags.empty());

    // A gap wins: when the span becomes a gap the damaged tag goes away.
    controls[1].tags = {kBreakTag};
    controls[2].tags = {kBreakTag};
    const auto sync = syncGapSpanTags(controls);
    CHECK(sync.formed == std::vector<size_t>{1});
    CHECK(spanIsGap(controls[1].segmentToNext));
    CHECK(!spanIsDamaged(controls[1].segmentToNext));
    // Same over stored controls.
    std::vector<StoredControlPoint> stored{StoredControlPoint{cv::Vec3d(0.0, 0.0, 0.0)},
                                           StoredControlPoint{cv::Vec3d(5.0, 0.0, 0.0)},
                                           StoredControlPoint{cv::Vec3d(10.0, 0.0, 0.0)}};
    for (size_t i = 0; i + 1 < stored.size(); ++i) {
        stored[i].segmentToNext.emplace();
        stored[i].segmentToNext->interpMode = SegmentInterpolationMode::Lasagna;
        stored[i].segmentToNext->message = "lasagna";
    }
    setControlPointTag(stored[0].segmentToNext->tags, kDamagedSpanTag, true);
    stored[0].tags = {kBreakTag};
    stored[1].tags = {kBreakTag};
    CHECK(syncGapSpanTags(stored));
    CHECK(spanIsGap(stored[0].segmentToNext));
    CHECK(!spanIsDamaged(stored[0].segmentToNext));
    // Round trip under version 4.
    const auto json = storedControlPointToJson(stored[0]);
    CHECK(json["segment_to_next"]["tags"] == nlohmann::json::array({"gap"}));
    StoredControlPoint damagedOnly{{0.0, 0.0, 0.0}};
    damagedOnly.segmentToNext.emplace();
    damagedOnly.segmentToNext->interpMode = SegmentInterpolationMode::Lasagna;
    damagedOnly.segmentToNext->message = "lasagna";
    setSpanTag(damagedOnly.segmentToNext, kDamagedSpanTag, true);
    const auto parsed = storedControlPointFromJson(storedControlPointToJson(damagedOnly), 4);
    CHECK(spanIsDamaged(parsed.segmentToNext));

    // Overlay ranges: damaged spans by the span flag, never where the span is
    // also a gap; the placement gate ignores damaged spans.
    std::vector<GeneratedOverlay::ControlPointMarker> markers(4);
    for (size_t i = 0; i < 4; ++i) {
        markers[i].controlIndex = i;
        markers[i].linePosition = 10.0 * static_cast<double>(i);
    }
    markers[0].hasDamagedToNext = true;
    markers[1].hasDamagedToNext = true;
    markers[1].hasGapToNext = true;
    const auto damaged = generatedDamagedLineRanges(markers);
    REQUIRE(damaged.size() == 1);
    CHECK(damaged[0].first == 0.0);
    CHECK(damaged[0].second == 10.0);
    const auto gaps = generatedGapLineRanges(markers);
    REQUIRE(gaps.size() == 1);
    CHECK(gaps[0].first == 10.0);
    CHECK(!generatedLinePositionInsideGap(gaps, 5.0));
    CHECK(generatedLinePositionInsideGap(gaps, 15.0));

    // A break is refused at or next to a kollesis termination: the
    // line-order neighbour test the menus use.
    markers[3].isKollesisTermination = true;
    CHECK(generatedLineOrderNeighbourIsKollesisTermination(markers, 2));
    CHECK(!generatedLineOrderNeighbourIsKollesisTermination(markers, 1));
    CHECK(!generatedLineOrderNeighbourIsKollesisTermination(markers, 0));
    // Order is by line position, not vector order.
    std::swap(markers[2], markers[3]);
    CHECK(generatedLineOrderNeighbourIsKollesisTermination(markers, 2));
    CHECK(!generatedLineOrderNeighbourIsKollesisTermination(markers, 1));
}

TEST_CASE("stored gap policy: a merge join between two breaks becomes a cspline gap")
{
    using namespace vc3d::line_annotation;
    std::vector<StoredControlPoint> stored{StoredControlPoint{cv::Vec3d(0.0, 0.0, 0.0)},
                                           StoredControlPoint{cv::Vec3d(5.0, 0.0, 0.0)},
                                           StoredControlPoint{cv::Vec3d(10.0, 0.0, 0.0)},
                                           StoredControlPoint{cv::Vec3d(15.0, 0.0, 0.0)}};
    for (size_t i = 0; i + 1 < stored.size(); ++i) {
        stored[i].segmentToNext.emplace();
        stored[i].segmentToNext->interpMode = SegmentInterpolationMode::Lasagna;
        stored[i].segmentToNext->message = "lasagna";
    }
    CHECK(!applyGapSpanPolicy(stored));
    stored[1].tags = {kBreakTag};
    stored[2].tags = {kBreakTag};
    setControlPointTag(stored[1].segmentToNext->tags, kDamagedSpanTag, true);
    CHECK(applyGapSpanPolicy(stored));
    CHECK(spanIsGap(stored[1].segmentToNext));
    CHECK(!spanIsDamaged(stored[1].segmentToNext));
    CHECK(stored[1].segmentToNext->interpGoal == SegmentInterpolationGoal::Cspline);
    CHECK(stored[0].segmentToNext->interpGoal == SegmentInterpolationGoal::Global);
    // Dissolving returns a still-cspline span to global; an explicit other
    // goal is left alone.
    stored[2].tags.clear();
    CHECK(applyGapSpanPolicy(stored));
    CHECK(!spanIsGap(stored[1].segmentToNext));
    CHECK(stored[1].segmentToNext->interpGoal == SegmentInterpolationGoal::Global);
    stored[2].tags = {kBreakTag};
    stored[1].segmentToNext->interpGoal = SegmentInterpolationGoal::Trace;
    applyGapSpanPolicy(stored);
    stored[2].tags.clear();
    stored[1].segmentToNext->interpGoal = SegmentInterpolationGoal::Lasagna;
    applyGapSpanPolicy(stored);
    CHECK(stored[1].segmentToNext->interpGoal == SegmentInterpolationGoal::Lasagna);
}

TEST_CASE("the JSON-level gap normalisation agrees with the typed sync")
{
    using namespace vc3d::line_annotation;
    // Typed side: two breaks around span 1, a stale gap on span 0, damaged on
    // span 1 (which the gap must clear) and an unrelated tag on span 2.
    std::vector<StoredControlPoint> stored{StoredControlPoint{cv::Vec3d(0.0, 0.0, 0.0)},
                                           StoredControlPoint{cv::Vec3d(5.0, 0.0, 0.0)},
                                           StoredControlPoint{cv::Vec3d(10.0, 0.0, 0.0)},
                                           StoredControlPoint{cv::Vec3d(15.0, 0.0, 0.0)}};
    for (size_t i = 0; i + 1 < stored.size(); ++i) {
        stored[i].segmentToNext.emplace();
        stored[i].segmentToNext->interpMode = SegmentInterpolationMode::Lasagna;
        stored[i].segmentToNext->message = "lasagna";
    }
    stored[1].tags = {kBreakTag};
    stored[2].tags = {kBreakTag};
    setControlPointTag(stored[0].segmentToNext->tags, kGapSpanTag, true);
    setControlPointTag(stored[1].segmentToNext->tags, kDamagedSpanTag, true);
    setControlPointTag(stored[2].segmentToNext->tags, "other", true);
    nlohmann::json controls = nlohmann::json::array();
    for (const auto& control : stored) {
        controls.push_back(storedControlPointToJson(control));
    }
    vc::fiber_tracer::normalizeGapSpanTagsJson(controls);
    syncGapSpanTags(stored);
    for (size_t i = 0; i < stored.size(); ++i) {
        CHECK(controls[i] == storedControlPointToJson(stored[i]));
    }
    CHECK(!controls[0]["segment_to_next"].contains("tags"));
    CHECK(controls[1]["segment_to_next"]["tags"] == nlohmann::json::array({"gap"}));
    CHECK(controls[2]["segment_to_next"]["tags"] == nlohmann::json::array({"other"}));
}

TEST_CASE("Overview adoption gate: a publish alone never settles, the report does, requests hold")
{
    using namespace vc3d::line_annotation;
    GeneratedOverviewGateState gate;
    // First publish of a fiber: nothing to keep, adopt.
    CHECK(generatedOverviewAdopts(gate));
    gate.layoutEmpty = false;
    // A placement publishes its spliced controls before the solve is queued:
    // the publish marks the geometry unconfirmed, so no adoption on it ...
    gate.geometryUnconfirmed = true;
    CHECK_FALSE(generatedOverviewAdopts(gate));
    // ... nor on the controller's report that a solve is pending / running.
    gate.geometryUnconfirmed = false;
    gate.solvePending = true;
    CHECK_FALSE(generatedOverviewAdopts(gate));
    gate.solvePending = false;
    gate.solveRunning = true;
    CHECK_FALSE(generatedOverviewAdopts(gate));
    // The landing publishes (unconfirmed again) ...
    gate.geometryUnconfirmed = true;
    CHECK_FALSE(generatedOverviewAdopts(gate));
    // ... and the report of an idle controller settles it.
    gate.solveRunning = false;
    gate.geometryUnconfirmed = false;
    CHECK(generatedOverviewAdopts(gate));
    // A landing epilogue still owing detached pending work reports pending.
    gate.solvePending = true;
    CHECK_FALSE(generatedOverviewAdopts(gate));
    // In manual mode queued edits never dispatch: the spliced line is adopted.
    gate.autoReoptimize = false;
    CHECK(generatedOverviewAdopts(gate));
    gate.autoReoptimize = true;
    gate.solvePending = false;
    // Controls re-expressed on the displayed line, or of another line, are
    // never adopted, however idle the controller.
    gate.controlsRebased = true;
    CHECK_FALSE(generatedOverviewAdopts(gate));
    gate.controlsRebased = false;
    gate.controlsIndexDisplayedLine = false;
    CHECK_FALSE(generatedOverviewAdopts(gate));
    gate.controlsIndexDisplayedLine = true;
    // A placement request of this dialog still out (its confirmation dialog
    // open while another solve landed and reported idle) holds the layout
    // it was made against; retiring the request releases it.
    gate.placementOutstanding = true;
    CHECK_FALSE(generatedOverviewAdopts(gate));
    gate.placementOutstanding = false;
    CHECK(generatedOverviewAdopts(gate));
    // Rollback of a failed view installation restores the previous state's
    // flags: the gate is a pure function of them, so the restored state
    // decides the same way it did before.
    GeneratedOverviewGateState before = gate;
    before.geometryUnconfirmed = true;
    GeneratedOverviewGateState restored = before;
    CHECK(generatedOverviewAdopts(before) == generatedOverviewAdopts(restored));
}

TEST_CASE("Strip control column: revision-gated arc lengths agree with the builder's map")
{
    using namespace vc3d::line_annotation;
    // The real strip map of the three-point line (x = 0, 10, 20), controls
    // retained as supports at every point.
    vc::lasagna::LineViewConfig config;
    config.controlPointLinePositions = {0.0, 1.0, 2.0};
    auto views = vc::lasagna::buildLineViewSurfaces(lineModel(), config);
    auto& map = views.stripPositionMap;
    REQUIRE(map.valid());
    map.lineRevision = 5;
    auto control = [](double linePosition, double arc, uint64_t revision) {
        GeneratedOverlay::ControlPointMarker m;
        m.linePosition = linePosition;
        m.arcLength = arc;
        m.lineArcLength = 20.0;
        m.lineRevision = revision;
        m.onLine = true;
        return m;
    };
    // Matching revision: the arc-length column is the position's column, at
    // the supports, at the endpoints and between them.
    for (const double position : {0.0, 0.5, 1.0, 1.5, 2.0}) {
        const double arc = position * 10.0;
        CHECK(generatedStripControlGridColumn(control(position, arc, 5), map) ==
              doctest::Approx(map.originalPositionToStripGridColumn(position)).epsilon(1e-6));
    }
    // A control whose arc and position disagree shows which path is taken:
    // matching revision -> the arc; mismatched or unknown -> the position.
    const auto disagreeing = [&](uint64_t revision) { return control(0.0, 20.0, revision); };
    CHECK(generatedStripControlGridColumn(disagreeing(5), map) ==
          doctest::Approx(map.originalPositionToStripGridColumn(2.0)));
    CHECK(generatedStripControlGridColumn(disagreeing(6), map) ==
          doctest::Approx(map.originalPositionToStripGridColumn(0.0)));
    CHECK(generatedStripControlGridColumn(disagreeing(0), map) ==
          doctest::Approx(map.originalPositionToStripGridColumn(0.0)));
    // A map without a stamped revision (the dialog-less inspection panes)
    // never takes the arc path.
    auto unstamped = map;
    unstamped.lineRevision = 0;
    CHECK(generatedStripControlGridColumn(disagreeing(5), unstamped) ==
          doctest::Approx(unstamped.originalPositionToStripGridColumn(0.0)));
    // A provisional control re-expressed on the displayed line takes its
    // displayed position for the fallback, not its own index.
    auto provisional = control(7.0, 10.0, 6);
    provisional.displayedLinePosition = 1.0;
    CHECK(generatedStripControlGridColumn(provisional, map) ==
          doctest::Approx(map.originalPositionToStripGridColumn(1.0)));
    // Display scaling: the map of the line scaled by 0.5 is in display
    // units, and so are the arc lengths the controller publishes.
    auto scaled = lineModel();
    for (auto& point : scaled.points) {
        point.position *= 0.5;
    }
    for (auto& segment : scaled.segmentSamples) {
        for (auto& sample : segment.samples) {
            sample.position *= 0.5;
        }
    }
    auto scaledViews = vc::lasagna::buildLineViewSurfaces(scaled, config);
    auto& scaledMap = scaledViews.stripPositionMap;
    REQUIRE(scaledMap.valid());
    scaledMap.lineRevision = 5;
    for (const double position : {0.0, 1.0, 1.5, 2.0}) {
        CHECK(generatedStripControlGridColumn(control(position, position * 5.0, 5), scaledMap) ==
              doctest::Approx(scaledMap.originalPositionToStripGridColumn(position)).epsilon(1e-6));
    }
}
