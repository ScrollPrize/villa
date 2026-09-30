// Coverage for FiberMapRulerMath.hpp: the 1-2-5 ladder the Fiber Map's rulers
// pick tick steps from, the unit a physical ruler labels in, and the label
// formatting. Pure arithmetic, so no widget is involved.

#include <QtTest/QtTest>

#include "FiberMapRulerMath.hpp"

#include <vector>

using namespace vc3d::fiber_map::ruler;

class TestFiberMapRulerMath : public QObject
{
    Q_OBJECT

private slots:
    void ladderReturnsTheSmallestStepNotBelowTheMinimum()
    {
        QCOMPARE(niceStepAtLeast(1.0), 1.0);
        QCOMPARE(niceStepAtLeast(1.1), 2.0);
        QCOMPARE(niceStepAtLeast(2.0), 2.0);
        QCOMPARE(niceStepAtLeast(2.1), 5.0);
        QCOMPARE(niceStepAtLeast(5.0), 5.0);
        QCOMPARE(niceStepAtLeast(5.1), 10.0);
        QCOMPARE(niceStepAtLeast(10.0), 10.0);
        QCOMPARE(niceStepAtLeast(0.3), 0.5);
        QCOMPARE(niceStepAtLeast(0.05), 0.05);
        QCOMPARE(niceStepAtLeast(730.0), 1000.0);
        QCOMPARE(niceStepAtLeast(1000.0), 1000.0);
        QCOMPARE(niceStepAtLeast(123456.0), 200000.0);
    }

    void ladderIsDefensiveAboutBadInput()
    {
        QCOMPARE(niceStepAtLeast(0.0), 1.0);
        QCOMPARE(niceStepAtLeast(-3.0), 1.0);
        QCOMPARE(niceStepAtLeast(std::numeric_limits<double>::quiet_NaN()), 1.0);
        QCOMPARE(niceStepAtLeast(std::numeric_limits<double>::infinity()), 1.0);
    }

    void integerLadderNeverGoesBelowOne()
    {
        QCOMPARE(niceIntegerStepAtLeast(0.01), 1);
        QCOMPARE(niceIntegerStepAtLeast(0.9), 1);
        QCOMPARE(niceIntegerStepAtLeast(1.5), 2);
        QCOMPARE(niceIntegerStepAtLeast(3.0), 5);
        QCOMPARE(niceIntegerStepAtLeast(7.0), 10);
        QCOMPARE(niceIntegerStepAtLeast(11.0), 20);
        // Saturates instead of overflowing the int.
        QCOMPARE(niceIntegerStepAtLeast(1e30), 1000000000);
        QCOMPARE(niceIntegerStepAtLeast(std::numeric_limits<double>::infinity()), 1);
    }

    void unitFollowsTheTickStep()
    {
        QCOMPARE(lengthUnitForStepUm(20.0), LengthUnit::Micrometre);
        QCOMPARE(lengthUnitForStepUm(100.0), LengthUnit::Millimetre);
        QCOMPARE(lengthUnitForStepUm(5000.0), LengthUnit::Millimetre);
        QCOMPARE(lengthUnitForStepUm(10000.0), LengthUnit::Centimetre);
        QCOMPARE(lengthUnitForStepUm(50000.0), LengthUnit::Centimetre);
        QCOMPARE(lengthUnitForStepUm(100000.0), LengthUnit::Metre);
        QCOMPARE(lengthUnitForStepUm(2000000.0), LengthUnit::Metre);
        // A cap holds the unit down however coarse the step gets.
        QCOMPARE(lengthUnitForStepUm(100000.0, LengthUnit::Centimetre), LengthUnit::Centimetre);
        QCOMPARE(lengthUnitForStepUm(5000000.0, LengthUnit::Centimetre), LengthUnit::Centimetre);
        QCOMPARE(lengthUnitForStepUm(5000.0, LengthUnit::Centimetre), LengthUnit::Millimetre);
        QCOMPARE(lengthUnitUm(LengthUnit::Millimetre), 1000.0);
        QCOMPARE(lengthUnitUm(LengthUnit::Metre), 1000000.0);
        QCOMPARE(lengthUnitSuffix(LengthUnit::Centimetre), QStringLiteral("cm"));
    }

    void lengthLabelsCarryOnlyTheDecimalsTheyNeed()
    {
        QCOMPARE(formatLength(0.0, LengthUnit::Centimetre), QStringLiteral("0"));
        QCOMPARE(formatLength(200000.0, LengthUnit::Centimetre), QStringLiteral("20"));
        QCOMPARE(formatLength(12500.0, LengthUnit::Millimetre), QStringLiteral("12.5"));
        QCOMPARE(formatLength(1250000.0, LengthUnit::Metre), QStringLiteral("1.25"));
        QCOMPARE(formatLength(-50000.0, LengthUnit::Centimetre), QStringLiteral("-5"));
        // Floating-point residue from a step multiplied out does not leak
        // into the label.
        QCOMPARE(formatLength(0.1 * 3.0 * 10000.0, LengthUnit::Centimetre),
                 QStringLiteral("0.3"));
        // Below the third decimal the value is zero, not "-0".
        QCOMPARE(formatLength(-0.1, LengthUnit::Metre), QStringLiteral("0"));
    }

    void narrowestNeighbourGapPrefersPairsOnScreen()
    {
        const std::vector<double> xs = {0.0, 10.0, 25.0, 45.0, 70.0};
        const auto self = [](double x) { return x; };
        // The pairs touching [30, 60] are (25, 45) and (45, 70).
        QCOMPARE(narrowestNeighbourGap(xs.begin(), xs.end(), self, 30.0, 60.0, 99.0), 20.0);
        // A window inside one pair still sees that pair.
        QCOMPARE(narrowestNeighbourGap(xs.begin(), xs.end(), self, 50.0, 60.0, 99.0), 25.0);
        // Nothing on screen: the narrowest anywhere.
        QCOMPARE(narrowestNeighbourGap(xs.begin(), xs.end(), self, 100.0, 200.0, 99.0), 10.0);
        // Fewer than two marks: the fallback.
        QCOMPARE(narrowestNeighbourGap(xs.begin(), xs.begin() + 1, self, 0.0, 1.0, 99.0),
                 99.0);
        QCOMPARE(narrowestNeighbourGap(xs.begin(), xs.begin(), self, 0.0, 1.0, 99.0), 99.0);
        // Duplicate positions are not a gap.
        const std::vector<double> flat = {3.0, 3.0, 3.0};
        QCOMPARE(narrowestNeighbourGap(flat.begin(), flat.end(), self, 0.0, 9.0, 7.0), 7.0);
    }

    void distanceTicksStopAtTheFloor()
    {
        // The reviewer's case: step 1000 over [-3926.99, 3000] keeps one spare
        // step below the range, which would put a labelled -4000 on the
        // continuation below the floor; the floor filter drops it and the
        // minor -3500 stays.
        const double floor = -3926.990817;
        const std::vector<DistanceTick> ticks =
            distanceTickCandidates(floor, 3000.0, 1000.0, floor, 2000);
        QVERIFY(!ticks.empty());
        QCOMPARE(ticks.front().distance, -3500.0);
        QVERIFY(!ticks.front().major);
        // The spare step above the range carries its half step too.
        QCOMPARE(ticks.back().distance, 4500.0);
        QVERIFY(!ticks.back().major);
        for (std::size_t i = 1; i < ticks.size(); ++i) {
            QVERIFY(ticks[i].distance > ticks[i - 1].distance);
            QVERIFY(ticks[i].distance >= floor);
        }
        // A tick exactly on the floor is kept; one just below is not.
        const std::vector<DistanceTick> onFloor =
            distanceTickCandidates(-2000.0, 0.0, 1000.0, -2000.0, 2000);
        QCOMPARE(onFloor.front().distance, -2000.0);
        const std::vector<DistanceTick> justBelow =
            distanceTickCandidates(-1999.0, 0.0, 1000.0, -1999.0, 2000);
        QCOMPARE(justBelow.front().distance, -1500.0);
        // Without a floor the spares survive on both sides.
        const std::vector<DistanceTick> floorless = distanceTickCandidates(
            -2000.0, 0.0, 1000.0, -std::numeric_limits<double>::infinity(), 2000);
        QCOMPARE(floorless.front().distance, -3000.0);
        QCOMPARE(floorless.back().distance, 1500.0);
        QCOMPARE(floorless.size(), std::size_t(10));
        // A range the step cannot cover within the cap yields nothing.
        QVERIFY(distanceTickCandidates(0.0, 1e9, 1.0, 0.0, 2000).empty());
        QVERIFY(distanceTickCandidates(0.0, std::numeric_limits<double>::infinity(), 1.0, 0.0,
                                       2000)
                    .empty());
    }

    void voxelLabelsAbbreviateThousands()
    {
        QCOMPARE(formatVoxels(0.0), QStringLiteral("0"));
        QCOMPARE(formatVoxels(0.2), QStringLiteral("0"));
        QCOMPARE(formatVoxels(500.0), QStringLiteral("500"));
        QCOMPARE(formatVoxels(999.0), QStringLiteral("999"));
        QCOMPARE(formatVoxels(1000.0), QStringLiteral("1k"));
        QCOMPARE(formatVoxels(2500.0), QStringLiteral("2.5k"));
        QCOMPARE(formatVoxels(20000.0), QStringLiteral("20k"));
        QCOMPARE(formatVoxels(-5000.0), QStringLiteral("-5k"));
        QCOMPARE(formatVoxels(1234567.0), QStringLiteral("1234.57k"));
    }
};

QTEST_APPLESS_MAIN(TestFiberMapRulerMath)

#include "test_fiber_map_ruler_math.moc"
