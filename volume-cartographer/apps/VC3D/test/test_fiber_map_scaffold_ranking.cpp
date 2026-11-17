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
