// The approval dock's ranking (FiberMapScaffoldRanking.hpp): the accused
// stretches of a solve's violated constraints, merged per stretch, and the
// three orders the dock offers.

#include "FiberMapScaffoldRanking.hpp"

#include <QtTest/QtTest>

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace vc3d::fiber_map;
using vc::fiber_graph::ScaffoldStoredPiece;
using vc::fiber_graph::ScaffoldStoredResidual;
using vc::fiber_graph::ScaffoldStoredWinding;

namespace
{

ScaffoldStoredPiece piece(const std::string& fiber, double from, double to, double score,
                          std::int64_t controlFrom = 0, std::int64_t controlTo = 1)
{
    ScaffoldStoredPiece entry;
    entry.fiberUuid = fiber;
    entry.arcFrom = from;
    entry.arcTo = to;
    entry.score = score;
    entry.controlFrom = controlFrom;
    entry.controlTo = controlTo;
    return entry;
}

ScaffoldStoredResidual residual(std::int64_t index, double cost, std::optional<std::int64_t> hops,
                                std::vector<ScaffoldStoredPiece> pieces)
{
    ScaffoldStoredResidual entry;
    entry.index = index;
    entry.weight = 1.0;
    entry.residual = cost;
    entry.cycleHops = hops;
    if (hops) entry.redundancy = 1;
    entry.pieces = std::move(pieces);
    return entry;
}

vc::fiber_graph::ScaffoldStoredWinding winding(std::int64_t turns, std::int64_t component,
                                              std::optional<std::int64_t> normalSign = std::nullopt)
{
    vc::fiber_graph::ScaffoldStoredWinding entry;
    entry.winding = turns;
    entry.component = component;
    entry.normalSign = normalSign;
    return entry;
}

ScaffoldApprovalRow row(double cost, std::int64_t cycle, double length)
{
    return ScaffoldApprovalRow{cost, cycle, length};
}

} // namespace

class FiberMapScaffoldRankingTest : public QObject
{
    Q_OBJECT

private slots:
    void cyclesAreNumberedByTheStretchesTheyRunAlong()
    {
        const std::vector<ScaffoldStoredResidual> residuals{
            // Two violated constraints of one loop: the same stretches, so
            // the same number.
            residual(0, 2.0, 3, {piece("a", 0.0, 10.0, 0.5), piece("b", 5.0, 15.0, 0.5)}),
            residual(1, 1.5, 3, {piece("b", 5.0, 15.0, 0.5), piece("a", 0.0, 10.0, 0.5)}),
            // Another loop, sharing one stretch with the first.
            residual(2, 1.0, 4, {piece("a", 0.0, 10.0, 0.4), piece("c", 1.0, 2.0, 0.6)}),
            // No cycle localized this one.
            residual(3, 0.5, std::nullopt, {})};
        const auto numbers = scaffoldCycleNumbers(residuals);
        QCOMPARE(numbers.size(), std::size_t{4});
        QCOMPARE(numbers[0], std::int64_t{1});
        QCOMPARE(numbers[1], std::int64_t{1});
        QCOMPARE(numbers[2], std::int64_t{2});
        QCOMPARE(numbers[3], std::int64_t{0});
        // A stretch lists every loop that runs along it, ascending.
        const auto ranked = rankScaffoldPieces(residuals);
        const auto of = [&](const std::string& fiber) {
            for (const auto& entry : ranked)
                if (entry.fiberUuid == fiber) return entry.cycles;
            return std::vector<std::int64_t>{};
        };
        QCOMPARE(of("a"), (std::vector<std::int64_t>{1, 2}));
        QCOMPARE(of("b"), (std::vector<std::int64_t>{1}));
        QCOMPARE(of("c"), (std::vector<std::int64_t>{2}));
    }

    void aStretchAccusedTwiceSumsBothShares()
    {
        const std::vector<ScaffoldStoredResidual> residuals{
            residual(0, 1.0, 3, {piece("a", 0.0, 10.0, 0.4), piece("b", 5.0, 15.0, 0.6)}),
            residual(1, 1.0, 5, {piece("a", 0.0, 10.0, 0.25), piece("c", 1.0, 2.0, 0.75)})};
        const auto ranked = rankScaffoldPieces(residuals);
        QCOMPARE(ranked.size(), std::size_t{3});
        // Highest score first: c (0.75), a (0.4 + 0.25), b (0.6)... a sums to
        // 0.65, so the order is c, a, b.
        QCOMPARE(ranked[0].fiberUuid, std::string{"c"});
        QCOMPARE(ranked[1].fiberUuid, std::string{"a"});
        QCOMPARE(ranked[2].fiberUuid, std::string{"b"});
        QCOMPARE(ranked[1].constraints, std::int64_t{2});
        QVERIFY(qFuzzyCompare(ranked[1].score, 0.65));
        // The shortest cycle of the constraints that accuse it.
        QCOMPARE(ranked[1].cycleHops, std::int64_t{3});
        QCOMPARE(ranked[0].cycleHops, std::int64_t{5});
    }

    void stretchesOfOneFiberStayApartWhenTheirArcsDiffer()
    {
        const std::vector<ScaffoldStoredResidual> residuals{
            residual(0, 1.0, 3, {piece("a", 0.0, 10.0, 0.5), piece("a", 20.0, 30.0, 0.5)})};
        const auto ranked = rankScaffoldPieces(residuals);
        QCOMPARE(ranked.size(), std::size_t{2});
        QCOMPARE(ranked[0].constraints, std::int64_t{1});
    }

    void aViolationWithoutACycleKeepsNoHops()
    {
        const auto ranked = rankScaffoldPieces({residual(0, 1.0, std::nullopt, {piece("a", 0.0, 1.0, 1.0)})});
        QCOMPARE(ranked.size(), std::size_t{1});
        QCOMPARE(ranked[0].cycleHops, std::int64_t{-1});
    }

    void cycleOrderPrefersTheShortestLoopThenTheShorterStretch()
    {
        const auto order = ScaffoldApprovalOrder::Cycle;
        QVERIFY(scaffoldApprovalRowBefore(row(0.1, 3, 500.0), row(9.0, 4, 10.0), order));
        QVERIFY(scaffoldApprovalRowBefore(row(0.1, 3, 10.0), row(0.1, 3, 11.0), order));
        // A row no cycle localizes sorts last, whatever it costs.
        QVERIFY(scaffoldApprovalRowBefore(row(0.01, 9, 1.0), row(100.0, -1, 1.0), order));
        QVERIFY(!scaffoldApprovalRowBefore(row(100.0, -1, 1.0), row(0.01, 9, 1.0), order));
    }

    void costOrderPrefersTheMostExpensiveThenTheShortestCycle()
    {
        const auto order = ScaffoldApprovalOrder::Cost;
        QVERIFY(scaffoldApprovalRowBefore(row(2.0, 9, 900.0), row(1.0, 3, 1.0), order));
        QVERIFY(scaffoldApprovalRowBefore(row(1.0, 3, 900.0), row(1.0, 4, 1.0), order));
        QVERIFY(scaffoldApprovalRowBefore(row(1.0, 3, 10.0), row(1.0, 3, 20.0), order));
    }

    void jointOrderIsCostPerCycleConstraint()
    {
        const auto order = ScaffoldApprovalOrder::Joint;
        // 1.0 over 3 hops beats 2.0 over 8.
        QVERIFY(scaffoldApprovalRowBefore(row(1.0, 3, 1.0), row(2.0, 8, 1.0), order));
        QVERIFY(!scaffoldApprovalRowBefore(row(2.0, 8, 1.0), row(1.0, 3, 1.0), order));
        // Equal per-hop cost falls back to the shorter cycle.
        QVERIFY(scaffoldApprovalRowBefore(row(1.0, 2, 1.0), row(2.0, 4, 1.0), order));
    }

    // Every fiber's unapproved geometry is cut into stretches, accused or
    // not: the fewest stretches within the length limit, on control points.
    void unaccusedFiberIsCutIntoTheFewestStretches()
    {
        std::map<std::string, ScaffoldApprovalSpans> fibers;
        fibers["f"].controlArcs = {0., 800., 1500., 1900., 3000., 3900.};
        fibers["f"].approved[0] = {0, 0, 0, 0, 0};
        auto segments = scaffoldApprovalSegments({}, fibers);
        std::sort(segments.begin(), segments.end(), [](const auto& x, const auto& y) { return x.arcFrom < y.arcFrom; });
        // 3900 voxels with a 2000 voxel limit: two stretches, both on control points.
        QCOMPARE(segments.size(), std::size_t(2));
        QCOMPARE(segments[0].arcFrom, 0.);
        QCOMPARE(segments[0].arcTo, segments[1].arcFrom);
        QCOMPARE(segments[1].arcTo, 3900.);
        for (const auto& segment : segments) {
            QVERIFY(segment.arcTo - segment.arcFrom <= 2000.);
            QCOMPARE(segment.controlFrom, std::round(segment.controlFrom));
            QCOMPARE(segment.controlTo, std::round(segment.controlTo));
            QCOMPARE(segment.cost, 0.);
            QCOMPARE(segment.constraints, std::int64_t(0));
            QCOMPARE(segment.cycleHops, std::int64_t(-1));
        }
        // A single span longer than the limit is a stretch of its own.
        fibers["f"].controlArcs = {0., 5000., 5400.};
        fibers["f"].approved[0] = {0, 0};
        segments = scaffoldApprovalSegments({}, fibers);
        QCOMPARE(segments.size(), std::size_t(2));
    }

    // Among the cuttings with the fewest stretches, the cut that splits the
    // least accused weight is taken.
    void cutsKeepAccusedStretchesWhole()
    {
        std::map<std::string, ScaffoldApprovalSpans> fibers;
        fibers["f"].controlArcs = {0., 1000., 2000., 3000.};
        fibers["f"].approved[0] = {0, 0, 0};
        // Two stretches either way (cut at 1000 or at 2000). The accusation
        // runs across control point 1, so the cut goes to control point 2.
        const std::vector<ScaffoldStoredResidual> residuals{residual(0, 4.0, 2, {piece("f", 600., 1400., 0.)})};
        auto segments = scaffoldApprovalSegments(residuals, fibers);
        std::sort(segments.begin(), segments.end(), [](const auto& x, const auto& y) { return x.arcFrom < y.arcFrom; });
        QCOMPARE(segments.size(), std::size_t(2));
        QCOMPARE(segments[0].arcTo, 2000.);
        QVERIFY(std::abs(segments[0].cost - 4.0) < 1e-9);
        QCOMPARE(segments[0].constraints, std::int64_t(1));
        QCOMPARE(segments[0].cycleHops, std::int64_t(2));
        QVERIFY(std::abs(segments[0].density - 4.0 / 2000.) < 1e-12);
        QCOMPARE(segments[1].cost, 0.);
        // And across control point 2 it goes to control point 1.
        const std::vector<ScaffoldStoredResidual> other{residual(0, 4.0, 2, {piece("f", 1600., 2400., 0.)})};
        segments = scaffoldApprovalSegments(other, fibers);
        std::sort(segments.begin(), segments.end(), [](const auto& x, const auto& y) { return x.arcFrom < y.arcFrom; });
        QCOMPARE(segments[0].arcTo, 1000.);
        QVERIFY(std::abs(segments[1].cost - 4.0) < 1e-9);
    }

    // An accusation is shared by length between the stretches it lies in; one
    // of no length belongs to the stretch holding that point and is never a
    // row of its own.
    void accusationsOnlyAddCost()
    {
        std::map<std::string, ScaffoldApprovalSpans> fibers;
        fibers["f"].controlArcs = {0., 1500., 3000., 4500.};
        fibers["f"].approved[0] = {0, 0, 0};
        const std::vector<ScaffoldStoredResidual> residuals{
            residual(0, 2.0, 2, {piece("f", 1000., 2000., 0.)}),  // half in each of the first two stretches
            residual(1, 5.0, 2, {piece("f", 3600., 3600., 0.)}),  // a point in the third
        };
        auto segments = scaffoldApprovalSegments(residuals, fibers);
        std::sort(segments.begin(), segments.end(), [](const auto& x, const auto& y) { return x.arcFrom < y.arcFrom; });
        QCOMPARE(segments.size(), std::size_t(3));
        QVERIFY(std::abs(segments[0].cost - 1.0) < 1e-9);
        QVERIFY(std::abs(segments[1].cost - 1.0) < 1e-9);
        QVERIFY(std::abs(segments[2].cost - 5.0) < 1e-9);
        for (const auto& segment : segments) QVERIFY(segment.arcTo > segment.arcFrom);
    }

    // Approved geometry is not offered again: only what is unapproved,
    // reaching a little back into the piece it continues.
    void approvedGeometryIsSubtracted()
    {
        const std::vector<ScaffoldStoredResidual> residuals{residual(0, 2.0, 2, {piece("f", 500., 1500., 0.)})};
        std::map<std::string, ScaffoldApprovalSpans> fibers;
        fibers["f"].controlArcs = {0., 1000., 2000., 3000.};
        fibers["f"].approved[0] = {1, 0, 0};
        const auto segments = scaffoldApprovalSegments(residuals, fibers);
        QCOMPARE(segments.size(), std::size_t(1));
        QCOMPARE(segments[0].arcFrom, 950.);  // 50 voxels back into the approved piece
        QCOMPARE(segments[0].arcTo, 3000.);
        QVERIFY(segments[0].extendsApproved);
        QVERIFY(std::abs(segments[0].controlFrom - 0.95) < 1e-12);
        QCOMPARE(segments[0].controlTo, 3.);
        // Only the accusation's unapproved half counts.
        QVERIFY(std::abs(segments[0].cost - 1.0) < 1e-9);
        // A fiber approved throughout asks for nothing.
        fibers["f"].approved[0] = {1, 1, 1};
        QVERIFY(scaffoldApprovalSegments(residuals, fibers).empty());
    }

    // What a run accuses on fiber the annotation no longer has stays a row,
    // without control positions.
    void accusationsOutsideTheControlsAreUnplaced()
    {
        std::map<std::string, ScaffoldApprovalSpans> fibers;
        fibers["f"].controlArcs = {0., 1000.};
        fibers["f"].approved[0] = {0};
        const std::vector<ScaffoldStoredResidual> residuals{
            residual(0, 2.0, 2, {piece("f", 800., 1200., 0.)}),
            residual(1, 1.0, 2, {piece("g", 10., 20., 0.)}),
        };
        const auto segments = scaffoldApprovalSegments(residuals, fibers);
        QCOMPARE(segments.size(), std::size_t(3));
        std::size_t unplaced = 0;
        for (const auto& segment : segments)
            if (segment.controlFrom < 0) {
                ++unplaced;
                QVERIFY(segment.cost > 0.);
            }
        QCOMPARE(unplaced, std::size_t(2));  // 1000-1200 of f, and all of g
    }

    void densityOrderPrefersTheHighestCostPerVoxel()
    {
        ScaffoldApprovalRow dense{1.0, 2, 100.};
        dense.density = 0.01;
        ScaffoldApprovalRow heavy{5.0, 2, 5000.};
        heavy.density = 0.001;
        QVERIFY(scaffoldApprovalRowBefore(dense, heavy, ScaffoldApprovalOrder::Density));
        QVERIFY(scaffoldApprovalRowBefore(heavy, dense, ScaffoldApprovalOrder::Cost));
    }

    void aGaugeShiftIsNotADisagreement()
    {
        // The same relative placement, one run gauged two turns higher: no
        // fiber moved against its neighbours.
        std::map<std::string, ScaffoldStoredWinding> first{
            {"a", winding(0, 0)}, {"b", winding(1, 0)}, {"c", winding(2, 0)}};
        std::map<std::string, ScaffoldStoredWinding> second{
            {"a", winding(2, 0)}, {"b", winding(3, 0)}, {"c", winding(4, 0)}};
        const auto diff = scaffoldRunDiff(first, second);
        QCOMPARE(diff.shared, 3);
        QCOMPARE(diff.winding, 0);
        QCOMPARE(diff.groups, 1);
        QCOMPARE(diff.onlyFirst, 0);
        QCOMPARE(diff.onlySecond, 0);
    }

    void oneFiberMovingAgainstItsComponentIsCounted()
    {
        std::map<std::string, ScaffoldStoredWinding> first{
            {"a", winding(0, 0)}, {"b", winding(1, 0)}, {"c", winding(2, 0)}};
        std::map<std::string, ScaffoldStoredWinding> second{
            {"a", winding(2, 0)}, {"b", winding(3, 0)}, {"c", winding(7, 0)}};
        const auto diff = scaffoldRunDiff(first, second);
        QCOMPARE(diff.shared, 3);
        QCOMPARE(diff.winding, 1);
    }

    void componentsAreComparedApartAndFibersOnlyOneRunPlacedAreNamed()
    {
        // Two components here, each with its own gauge there, plus a fiber
        // the second run did not solve and one only it did.
        std::map<std::string, ScaffoldStoredWinding> first{
            {"a", winding(0, 0)}, {"b", winding(1, 0)}, {"c", winding(0, 1)}, {"d", winding(1, 1)},
            {"gone", winding(5, 1)}};
        std::map<std::string, ScaffoldStoredWinding> second{
            {"a", winding(3, 0)}, {"b", winding(4, 0)}, {"c", winding(9, 7)}, {"d", winding(11, 7)},
            {"new", winding(0, 7)}};
        const auto diff = scaffoldRunDiff(first, second);
        QCOMPARE(diff.shared, 4);
        QCOMPARE(diff.groups, 2);
        QCOMPARE(diff.onlyFirst, 1);
        QCOMPARE(diff.onlySecond, 1);
        // Component 0 shifted by 3 as a whole, so neither of its fibers
        // moved. The other group's two fibers shifted by 9 and 10; the tie
        // goes to the lower offset, so the other one is the fiber counted.
        QCOMPARE(diff.winding, 1);
    }

    void normalSignsAreComparedWhereBothRunsStateThem()
    {
        std::map<std::string, ScaffoldStoredWinding> first{
            {"a", winding(0, 0, 1)}, {"b", winding(1, 0, -1)}, {"c", winding(2, 0)}};
        std::map<std::string, ScaffoldStoredWinding> second{
            {"a", winding(0, 0, 1)}, {"b", winding(1, 0, 1)}, {"c", winding(2, 0, 1)}};
        const auto diff = scaffoldRunDiff(first, second);
        QCOMPARE(diff.signStated, 2);
        QCOMPARE(diff.sign, 1);
    }
};

QTEST_APPLESS_MAIN(FiberMapScaffoldRankingTest)
#include "test_fiber_map_scaffold_ranking.moc"
