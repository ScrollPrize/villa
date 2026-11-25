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

    // A stretch runs from constraint to constraint, as the cycle states it:
    // its ends are not moved onto control points, and a fiber nothing
    // accuses asks for nothing.
    void stretchesRunFromConstraintToConstraint()
    {
        std::map<std::string, ScaffoldApprovalSpans> fibers;
        fibers["f"].controlArcs = {0., 1000., 2000., 3000.};
        fibers["g"].controlArcs = {0., 1000.};
        const std::vector<ScaffoldStoredResidual> residuals{residual(0, 4.0, 2, {piece("f", 600., 1400., 0.)})};
        const auto segments = scaffoldApprovalSegments(residuals, fibers);
        QCOMPARE(segments.size(), std::size_t(1));
        QCOMPARE(segments[0].fiberUuid, std::string("f"));
        QCOMPARE(segments[0].arcFrom, 600.);
        QCOMPARE(segments[0].arcTo, 1400.);
        QVERIFY(std::abs(segments[0].controlFrom - 0.6) < 1e-12);
        QVERIFY(std::abs(segments[0].controlTo - 1.4) < 1e-12);
        QVERIFY(std::abs(segments[0].cost - 4.0) < 1e-9);
        QVERIFY(std::abs(segments[0].density - 4.0 / 800.) < 1e-12);
        QCOMPARE(segments[0].constraints, std::int64_t(1));
        QCOMPARE(segments[0].cycleHops, std::int64_t(2));
    }

    // Stretches of one fiber that overlap or lie close together are one
    // stretch covering their constraints; far apart, or too long together,
    // they stay apart.
    void closeStretchesAreJoined()
    {
        std::map<std::string, ScaffoldApprovalSpans> fibers;
        fibers["f"].controlArcs = {0., 9000.};
        const std::vector<ScaffoldStoredResidual> residuals{
            residual(0, 2.0, 2, {piece("f", 100., 500., 0.)}),
            residual(1, 3.0, 3, {piece("f", 400., 700., 0.)}),    // overlaps the first
            residual(2, 1.0, 2, {piece("f", 900., 900., 0.)}),    // a point 200 voxels behind
            residual(3, 5.0, 2, {piece("f", 1300., 1500., 0.)}),  // 400 voxels behind: its own
            residual(4, 7.0, 2, {piece("f", 1700., 3400., 0.)}),  // close, but 2100 voxels together
        };
        auto segments = scaffoldApprovalSegments(residuals, fibers);
        std::sort(segments.begin(), segments.end(), [](const auto& x, const auto& y) { return x.arcFrom < y.arcFrom; });
        QCOMPARE(segments.size(), std::size_t(3));
        QCOMPARE(segments[0].arcFrom, 100.);
        QCOMPARE(segments[0].arcTo, 900.);
        QVERIFY(std::abs(segments[0].cost - 6.0) < 1e-9);
        QCOMPARE(segments[0].constraints, std::int64_t(3));
        QCOMPARE(segments[0].cycleHops, std::int64_t(2));
        QCOMPARE(segments[1].arcFrom, 1300.);
        QCOMPARE(segments[1].arcTo, 1500.);
        QCOMPARE(segments[2].arcFrom, 1700.);
        QCOMPARE(segments[2].arcTo, 3400.);
    }

    // A cycle's cost is split over its stretches by 1 / length, among the
    // ones whose geometry is not approved: an approved stretch cannot be the
    // error, and with one stretch left all of the cost is there.
    void approvedStretchesTakeNoShare()
    {
        const std::vector<ScaffoldStoredResidual> residuals{
            residual(0, 3.0, 3, {piece("f", 500., 1500., 0.), piece("g", 100., 600., 0.)})};
        std::map<std::string, ScaffoldApprovalSpans> fibers;
        fibers["f"].controlArcs = {0., 3000.};
        fibers["g"].controlArcs = {0., 3000.};
        auto segments = scaffoldApprovalSegments(residuals, fibers);
        QCOMPARE(segments.size(), std::size_t(2));
        // 1/1000 against 1/500: one third and two thirds.
        QCOMPARE(segments[0].fiberUuid, std::string("g"));
        QVERIFY(std::abs(segments[0].cost - 2.0) < 1e-9);
        QVERIFY(std::abs(segments[1].cost - 1.0) < 1e-9);
        // An approval that does not reach both constraints changes nothing.
        fibers["g"].approvedArcs[0] = {{50., 400.}};
        segments = scaffoldApprovalSegments(residuals, fibers);
        QCOMPARE(segments.size(), std::size_t(2));
        fibers["g"].approvedArcs[0] = {{50., 700.}};
        segments = scaffoldApprovalSegments(residuals, fibers);
        QCOMPARE(segments.size(), std::size_t(1));
        QCOMPARE(segments[0].fiberUuid, std::string("f"));
        QVERIFY(std::abs(segments[0].cost - 3.0) < 1e-9);
        fibers["f"].approvedArcs[0] = {{0., 3000.}};
        QVERIFY(scaffoldApprovalSegments(residuals, fibers).empty());
    }

    // What a run accuses on fiber the annotation no longer has stays a row,
    // without control positions.
    void accusationsOutsideTheControlsAreUnplaced()
    {
        std::map<std::string, ScaffoldApprovalSpans> fibers;
        fibers["f"].controlArcs = {0., 1000.};
        const std::vector<ScaffoldStoredResidual> residuals{
            residual(0, 2.0, 2, {piece("f", 800., 1200., 0.)}),
            residual(1, 1.0, 2, {piece("g", 10., 20., 0.)}),
            residual(2, 1.0, 2, {piece("f", 100., 200., 0.)}),
        };
        const auto segments = scaffoldApprovalSegments(residuals, fibers);
        QCOMPARE(segments.size(), std::size_t(3));
        std::size_t unplaced = 0;
        for (const auto& segment : segments)
            if (segment.controlFrom < 0) ++unplaced;
        QCOMPARE(unplaced, std::size_t(2));  // 800-1200 of f, and all of g
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
