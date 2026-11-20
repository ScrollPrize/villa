// The approval dock's ranking (FiberMapScaffoldRanking.hpp): the accused
// stretches of a solve's violated constraints, merged per stretch, and the
// three orders the dock offers.

#include "FiberMapScaffoldRanking.hpp"

#include <QtTest/QtTest>

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

    // Cycles that leave a fiber at different crossings accuse nearly the
    // same piece of it: the dock offers that piece once.
    void overlappingStretchesBecomeOneSegment()
    {
        // The densest stretch is taken first; the others reach at most 122
        // voxels past it and are absorbed.
        const std::vector<ScaffoldStoredResidual> residuals{
            residual(0, 4.0, 2, {piece("f", 2426.6, 3682.0, 0.)}),
            residual(1, 1.0, 2, {piece("f", 2426.6, 3726.0, 0.)}),
            residual(2, 1.0, 2, {piece("f", 2426.6, 3743.3, 0.)}),
            residual(3, 1.0, 2, {piece("f", 2426.6, 3804.0, 0.)}),
        };
        std::map<std::string, ScaffoldApprovalSpans> fibers;
        fibers["f"].controlArcs = {0., 1000., 2000., 6000.};
        fibers["f"].approved[0] = {0, 0, 0};
        const auto segments = scaffoldApprovalSegments(residuals, fibers);
        QCOMPARE(segments.size(), std::size_t(1));
        const auto& segment = segments.front();
        // The start snaps outward to control point 2 (427 voxels away); the
        // end has no control point within reach and stays, as a fraction.
        QCOMPARE(segment.arcFrom, 2000.);
        QCOMPARE(segment.arcTo, 3804.);
        QCOMPARE(segment.controlFrom, 2.);
        QVERIFY(std::abs(segment.controlTo - (2. + 1804. / 4000.)) < 1e-12);
        QCOMPARE(segment.constraints, std::int64_t(4));
        QVERIFY(std::abs(segment.cost - 7.0) < 1e-9); // every share, summed
        QVERIFY(std::abs(segment.density - 4.0 / (3682.0 - 2426.6)) < 1e-12); // the densest stretch's
        QVERIFY(!segment.extendsApproved);
        QCOMPARE(segment.cycles.size(), std::size_t(4));
    }

    // Absorption is bounded by the segment's original extent, so it cannot
    // chain along the fiber.
    void absorptionDoesNotChain()
    {
        const std::vector<ScaffoldStoredResidual> residuals{
            residual(0, 9.0, 2, {piece("f", 0., 1000., 0.)}),
            residual(1, 0.3, 2, {piece("f", 1000., 1400., 0.)}),
            residual(2, 0.2, 2, {piece("f", 1400., 1800., 0.)}),
            residual(3, 0.1, 2, {piece("f", 1800., 2200., 0.)}),
        };
        auto segments = scaffoldApprovalSegments(residuals, {});
        std::sort(segments.begin(), segments.end(), [](const auto& x, const auto& y) { return x.arcFrom < y.arcFrom; });
        QCOMPARE(segments.size(), std::size_t(2));
        // 1000-1400 lies within 500 voxels of the first segment's original
        // end; 1400-1800 does not, although the grown segment now reaches it.
        QCOMPARE(segments[0].arcFrom, 0.);
        QCOMPARE(segments[0].arcTo, 1400.);
        // The next one starts over, and absorbs what is within its own reach.
        QCOMPARE(segments[1].arcFrom, 1400.);
        QCOMPARE(segments[1].arcTo, 2200.);
        // Without the fiber's controls nothing is snapped or positioned.
        QCOMPARE(segments[0].controlFrom, -1.);
    }

    // Approved geometry is not offered again: only what is new, reaching a
    // little back into the piece it continues.
    void approvedGeometryIsSubtracted()
    {
        const std::vector<ScaffoldStoredResidual> residuals{residual(0, 2.0, 2, {piece("f", 500., 1600., 0.)})};
        std::map<std::string, ScaffoldApprovalSpans> fibers;
        fibers["f"].controlArcs = {0., 1000., 2000., 3000.};
        fibers["f"].approved[0] = {1, 0, 0};
        const auto segments = scaffoldApprovalSegments(residuals, fibers);
        QCOMPARE(segments.size(), std::size_t(1));
        QCOMPARE(segments[0].arcFrom, 950.);  // 50 voxels back into the approved piece
        QCOMPARE(segments[0].arcTo, 2000.);   // outward to control point 2
        QVERIFY(segments[0].extendsApproved);
        QVERIFY(std::abs(segments[0].controlFrom - 0.95) < 1e-12);
        QCOMPARE(segments[0].controlTo, 2.);
        // A stretch that lies wholly on approved geometry asks for nothing.
        const std::vector<ScaffoldStoredResidual> covered{residual(0, 2.0, 2, {piece("f", 100., 900., 0.)})};
        QVERIFY(scaffoldApprovalSegments(covered, fibers).empty());
    }

    // Ends never move inward, and never across a neighbouring segment.
    void snappingIsOutwardAndStopsAtANeighbour()
    {
        const std::vector<ScaffoldStoredResidual> residuals{
            residual(0, 9.0, 2, {piece("f", 1100., 1200., 0.)}),
            residual(1, 1.0, 2, {piece("f", 3000., 3100., 0.)}),
        };
        std::map<std::string, ScaffoldApprovalSpans> fibers;
        fibers["f"].controlArcs = {0., 1000., 1900., 3500., 9000.};
        fibers["f"].approved[0] = {0, 0, 0, 0};
        auto segments = scaffoldApprovalSegments(residuals, fibers);
        std::sort(segments.begin(), segments.end(), [](const auto& x, const auto& y) { return x.arcFrom < y.arcFrom; });
        QCOMPARE(segments.size(), std::size_t(2));
        QCOMPARE(segments[0].arcFrom, 1000.);
        QCOMPARE(segments[0].arcTo, 1900.);   // 700 voxels out, within reach
        // The control point below 3000 is 1900, 1100 voxels away: out of
        // reach, so the start stays a fraction rather than moving inward.
        QCOMPARE(segments[1].arcFrom, 3000.);
        QCOMPARE(segments[1].arcTo, 3500.);
        QVERIFY(segments[1].controlFrom > 2. && segments[1].controlFrom < 3.);
    }

    // Where two segments meet is an accident of the accusations: the shared
    // boundary moves to a control point within reach, and nothing is lost.
    void aSharedBoundaryMovesToAControlPoint()
    {
        const std::vector<ScaffoldStoredResidual> residuals{
            residual(0, 9.0, 2, {piece("f", 1000., 2100., 0.)}),
            residual(1, 1.0, 2, {piece("f", 2100., 4000., 0.)}),
        };
        std::map<std::string, ScaffoldApprovalSpans> fibers;
        fibers["f"].controlArcs = {0., 1000., 2000., 4000.};
        fibers["f"].approved[0] = {0, 0, 0};
        auto segments = scaffoldApprovalSegments(residuals, fibers);
        std::sort(segments.begin(), segments.end(), [](const auto& x, const auto& y) { return x.arcFrom < y.arcFrom; });
        QCOMPARE(segments.size(), std::size_t(2));
        QCOMPARE(segments[0].arcFrom, 1000.);
        QCOMPARE(segments[0].arcTo, 2000.);   // the boundary at 2100 moved to control point 2
        QCOMPARE(segments[1].arcFrom, 2000.);
        QCOMPARE(segments[1].arcTo, 4000.);
        QCOMPARE(segments[0].controlTo, 2.);
    }

    // A very dense accusation a few voxels long is taken first and would be
    // a row of its own; it joins the segment it touches instead.
    void aSliverJoinsItsNeighbour()
    {
        const std::vector<ScaffoldStoredResidual> residuals{
            residual(0, 5.0, 2, {piece("f", 3000., 3040., 0.)}),
            residual(1, 1.0, 2, {piece("f", 1500., 3000., 0.)}),
        };
        const auto segments = scaffoldApprovalSegments(residuals, {});
        QCOMPARE(segments.size(), std::size_t(1));
        QCOMPARE(segments[0].arcFrom, 1500.);
        QCOMPARE(segments[0].arcTo, 3040.);
        QVERIFY(std::abs(segments[0].cost - 6.0) < 1e-9);
        QVERIFY(std::abs(segments[0].density - 5.0 / 40.0) < 1e-12);
        QCOMPARE(segments[0].constraints, std::int64_t(2));
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
