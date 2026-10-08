// The Fiber Map's kollesis seams (FiberMapKollesis.hpp): terminations
// grouped by a gap in windings, each side's bound the vertical through its
// mean, the seam centred between them, sheet-length statistics over the
// gaps, and predicted seams inside multi-sheet gaps and past the last seam.

#include "FiberMapKollesis.hpp"

#include <QtTest/QtTest>

#include <cmath>
#include <limits>
#include <vector>

using namespace vc3d::fiber_map::kollesis;

namespace
{

constexpr double kWinding = 1000.0;  // map x per winding

Termination at(uint64_t fiber, double xMap, double z, bool left, double sceneScale = 1.0)
{
    Termination t;
    t.fiberId = fiber;
    t.controlIndex = left ? 0 : 5;
    t.xMapVx = xMap;
    t.xSceneVx = xMap * sceneScale;
    t.zVx = z;
    t.left = left;
    return t;
}

Params params(double gapWindings = 0.75, int ahead = 0)
{
    Params p;
    p.gapWindings = gapWindings;
    p.windingWidthVx = kWinding;
    p.predictInterior = true;
    p.extrapolateCount = ahead;
    return p;
}

bool near(double a, double b, double tolerance = 1e-9)
{
    return std::abs(a - b) <= tolerance;
}

}  // namespace

class FiberMapKollesisTest : public QObject
{
    Q_OBJECT

private slots:
    void emptyInputGivesNoSeams()
    {
        const Model model = buildModel({}, params(0.75, 3));
        QVERIFY(model.seams.empty());
        QVERIFY(model.sheetLengthVx.empty());
        QVERIFY(!model.statistics.valid());
        QVERIFY(model.predicted.empty());
    }

    void aBoundIsTheVerticalThroughTheMean()
    {
        const Bound one = fitBound({at(1, 500.0, 300.0, true)});
        QCOMPARE(one.count, 1);
        QCOMPARE(one.xVx, 500.0);
        // Different heights change nothing: the bound is vertical.
        const Bound two = fitBound({at(1, 400.0, 0.0, true), at(2, 600.0, 1000.0, true)});
        QCOMPARE(two.count, 2);
        QCOMPARE(two.xVx, 500.0);
    }

    void aSingleTerminationIsALineNotABand()
    {
        const Model model = buildModel({at(1, 2500.0, 400.0, false)}, params());
        QCOMPARE(model.seams.size(), std::size_t{1});
        const Seam& seam = model.seams.front();
        QVERIFY(!seam.left);
        QVERIFY(seam.right);
        QVERIFY(!seam.hasBand());
        QCOMPARE(seam.xVx(), 2500.0);
        QCOMPARE(seam.members.size(), std::size_t{1});
    }

    void aLeftRightPairBoundsABandWithTheSeamCentred()
    {
        // The newer sheet starts at 2400 (left terminations), the older one
        // ends at 2600 (right terminations): a 200 vx overlap.
        const Model model = buildModel(
            {at(1, 2600.0, 200.0, false), at(2, 2400.0, 800.0, true), at(3, 2650.0, 500.0, false)},
            params());
        QCOMPARE(model.seams.size(), std::size_t{1});
        const Seam& seam = model.seams.front();
        QVERIFY(seam.hasBand());
        QCOMPARE(seam.left->xVx, 2400.0);
        QCOMPARE(seam.left->count, 1);
        QCOMPARE(seam.right->xVx, 2625.0);
        QCOMPARE(seam.right->count, 2);
        QCOMPARE(seam.xVx(), 2512.5);
        // Members come back ascending in map x.
        QCOMPARE(seam.members.front().fiberId, uint64_t{2});
        QCOMPARE(seam.members.back().fiberId, uint64_t{3});
    }

    void terminationsFartherThanTheGapStartANewSeam()
    {
        const Model model = buildModel(
            {at(1, 2600.0, 100.0, false), at(2, 2400.0, 900.0, true), at(3, 7500.0, 500.0, true),
             at(4, 12000.0, 500.0, false), at(5, 12100.0, 900.0, true)},
            params());
        QCOMPARE(model.seams.size(), std::size_t{3});
        QVERIFY(model.seams[0].hasBand());
        QVERIFY(!model.seams[1].hasBand());
        QVERIFY(model.seams[2].hasBand());
        // 2400/2600 split by a tighter gap: 0.1 windings is 100 vx.
        const Model tight = buildModel(
            {at(1, 2600.0, 100.0, false), at(2, 2400.0, 900.0, true)}, params(0.1));
        QCOMPARE(tight.seams.size(), std::size_t{2});
    }

    void seamsComeAscendingWithGapsBetween()
    {
        const Model model = buildModel(
            {at(4, 12000.0, 500.0, false), at(1, 2600.0, 100.0, false),
             at(2, 2400.0, 900.0, true), at(3, 7500.0, 500.0, true)},
            params());
        QCOMPARE(model.seams.size(), std::size_t{3});
        QCOMPARE(model.sheetLengthVx.size(), std::size_t{2});
        QCOMPARE(model.seams[0].xVx(), 2500.0);
        QCOMPARE(model.seams[1].xVx(), 7500.0);
        QCOMPARE(model.seams[2].xVx(), 12000.0);
        QCOMPARE(model.sheetLengthVx[0], 5000.0);
        QCOMPARE(model.sheetLengthVx[1], 4500.0);
    }

    void groupingIsInMapXButSeamsAreInSceneX()
    {
        // Scene x is twice the map x (an outer winding drawn wider): the
        // 2400/2600 pair still groups on the map's 200 vx, while the band
        // and the gaps read in scene units.
        const Model model = buildModel(
            {at(1, 2600.0, 100.0, false, 2.0), at(2, 2400.0, 900.0, true, 2.0),
             at(3, 7500.0, 500.0, true, 2.0)},
            params());
        QCOMPARE(model.seams.size(), std::size_t{2});
        QCOMPARE(model.seams[0].left->xVx, 4800.0);
        QCOMPARE(model.seams[0].right->xVx, 5200.0);
        QCOMPARE(model.sheetLengthVx[0], 10000.0);
    }

    void statisticsReadEqualGapsAsOneSheetEach()
    {
        const SheetStatistics stats = estimateSheetStatistics({5000.0, 5100.0, 4900.0});
        QVERIFY(stats.valid());
        QCOMPARE(stats.sheetCounts, (std::vector<int>{1, 1, 1}));
        QVERIFY(near(stats.unitLengthVx, 5000.0));
        // Sample deviation of 5000, 5100, 4900.
        QVERIFY(near(stats.spreadVx, 100.0));
    }

    void statisticsReadADoubleGapAsTwoSheets()
    {
        const SheetStatistics stats = estimateSheetStatistics({5000.0, 10200.0, 4900.0});
        QCOMPARE(stats.sheetCounts, (std::vector<int>{1, 2, 1}));
        // (5000 + 10200 + 4900) / 4 sheets.
        QVERIFY(near(stats.unitLengthVx, 5025.0));
        // Per-sheet lengths 5000, 5100, 4900.
        QVERIFY(near(stats.spreadVx, 100.0));
    }

    void statisticsSeedResistsOneSpuriousShortGap()
    {
        // One 300 vx gap (a seam split in two by the grouping) among real
        // sheets: the seed is a low percentile rather than the minimum, so
        // the real gaps still read as one sheet each.
        const SheetStatistics stats =
            estimateSheetStatistics({5000.0, 300.0, 5100.0, 4900.0, 5050.0});
        QCOMPARE(stats.sheetCounts, (std::vector<int>{1, 1, 1, 1, 1}));
    }

    void statisticsSettleSoCountsAgreeWithTheUnit()
    {
        // Three fixed rounds left counts [1,2,2,23] with a unit that rounds
        // the last gap to 24: the refinement runs to a fixed point instead.
        const SheetStatistics stats = estimateSheetStatistics({2000.0, 3000.0, 3000.0, 42000.0});
        QVERIFY(stats.valid());
        QCOMPARE(stats.sheetCounts, (std::vector<int>{1, 2, 2, 24}));
        QVERIFY(near(stats.unitLengthVx, 50000.0 / 29.0));
        const std::vector<double> gaps{2000.0, 3000.0, 3000.0, 42000.0};
        for (std::size_t i = 0; i < gaps.size(); ++i) {
            QCOMPARE(static_cast<int>(std::lround(gaps[i] / stats.unitLengthVx)),
                     stats.sheetCounts[i]);
        }
    }

    void aSpuriousTinyGapMakesTheEstimateUnusable()
    {
        // A near-zero gap as the seed would read the real gap as billions
        // of sheets: no statistics, no predictions, no huge allocation.
        const SheetStatistics stats = estimateSheetStatistics({1e-6, 5000.0});
        QVERIFY(!stats.valid());
        QCOMPARE(stats.sheetCounts, (std::vector<int>{1, 1}));
        Params p = params(0.75, 3);
        p.windingWidthVx = 0.0;
        const Model model = buildModel(
            {at(1, 0.0, 0.0, true), at(2, 0.000001, 0.0, true), at(3, 5000.000001, 0.0, true)}, p);
        QCOMPARE(model.seams.size(), std::size_t{3});
        QVERIFY(!model.statistics.valid());
        QVERIFY(model.predicted.empty());
    }

    void singleGapHasNoSpread()
    {
        const SheetStatistics stats = estimateSheetStatistics({5000.0});
        QVERIFY(stats.valid());
        QCOMPARE(stats.unitLengthVx, 5000.0);
        QCOMPARE(stats.spreadVx, 0.0);
    }

    void nonPositiveGapsGiveNoStatistics()
    {
        QVERIFY(!estimateSheetStatistics({}).valid());
        QVERIFY(!estimateSheetStatistics({0.0, -5.0}).valid());
    }

    void interiorPredictionsFillAMultiSheetGapEvenly()
    {
        // Seams at 0, 5000, 15200 (two sheets), 20100.
        const Model model = buildModel(
            {at(1, 0.0, 0.0, true), at(2, 5000.0, 0.0, true), at(3, 15200.0, 0.0, true),
             at(4, 20100.0, 0.0, true)},
            params());
        QCOMPARE(model.statistics.sheetCounts, (std::vector<int>{1, 2, 1}));
        QCOMPARE(model.predicted.size(), std::size_t{1});
        const PredictedSeam& seam = model.predicted.front();
        QVERIFY(!seam.extrapolated);
        QCOMPARE(seam.gapIndex, 1);
        QCOMPARE(seam.step, 1);
        // Halfway across the measured gap, not one unit from its start.
        QCOMPARE(seam.xVx, 10100.0);
        QVERIFY(near(seam.spreadVx, model.statistics.spreadVx));
    }

    void extrapolationStepsOneUnitPastTheLastSeamWithGrowingSpread()
    {
        const Model model = buildModel(
            {at(1, 0.0, 0.0, true), at(2, 5000.0, 0.0, true), at(3, 10200.0, 0.0, true)},
            params(0.75, 3));
        QCOMPARE(model.predicted.size(), std::size_t{3});
        const double unit = model.statistics.unitLengthVx;
        QVERIFY(near(unit, 5100.0));
        const double spread = model.statistics.spreadVx;
        QVERIFY(spread > 0.0);
        for (std::size_t i = 0; i < model.predicted.size(); ++i) {
            const PredictedSeam& seam = model.predicted[i];
            QVERIFY(seam.extrapolated);
            QCOMPARE(seam.gapIndex, -1);
            QCOMPARE(seam.step, static_cast<int>(i) + 1);
            QVERIFY(near(seam.xVx, 10200.0 + unit * static_cast<double>(i + 1)));
            QVERIFY(near(seam.spreadVx, spread * std::sqrt(static_cast<double>(i + 1))));
        }
    }

    void predictionsNeedTwoSeamsAndAreAscending()
    {
        // One seam: no gap, no unit, nothing predicted however many asked.
        const Model single = buildModel({at(1, 2500.0, 0.0, true)}, params(0.75, 5));
        QVERIFY(single.predicted.empty());
        // Interior then extrapolated, ascending in x.
        const Model both = buildModel(
            {at(1, 0.0, 0.0, true), at(2, 5000.0, 0.0, true), at(3, 15000.0, 0.0, true)},
            params(0.75, 2));
        QCOMPARE(both.predicted.size(), std::size_t{3});
        for (std::size_t i = 1; i < both.predicted.size(); ++i) {
            QVERIFY(both.predicted[i - 1].xVx < both.predicted[i].xVx);
        }
        QVERIFY(!both.predicted[0].extrapolated);
        QVERIFY(both.predicted[1].extrapolated);
    }

    void interiorPredictionsCanBeSwitchedOff()
    {
        Params p = params(0.75, 1);
        p.predictInterior = false;
        const Model model = buildModel(
            {at(1, 0.0, 0.0, true), at(2, 5000.0, 0.0, true), at(3, 15000.0, 0.0, true)}, p);
        QCOMPARE(model.predicted.size(), std::size_t{1});
        QVERIFY(model.predicted.front().extrapolated);
    }

    void withoutAWindingWidthEveryDistinctXIsItsOwnSeam()
    {
        Params p = params();
        p.windingWidthVx = 0.0;
        const Model model = buildModel(
            {at(1, 2600.0, 100.0, false), at(2, 2400.0, 900.0, true)}, p);
        QCOMPARE(model.seams.size(), std::size_t{2});
    }

    void nonFiniteTerminationsAreDropped()
    {
        Termination bad = at(9, 2500.0, 100.0, true);
        bad.xSceneVx = std::numeric_limits<double>::quiet_NaN();
        const Model model = buildModel({bad, at(1, 7500.0, 500.0, false)}, params());
        QCOMPARE(model.seams.size(), std::size_t{1});
        QCOMPARE(model.seams.front().members.front().fiberId, uint64_t{1});
    }
};

QTEST_APPLESS_MAIN(FiberMapKollesisTest)
#include "test_fiber_map_kollesis.moc"
