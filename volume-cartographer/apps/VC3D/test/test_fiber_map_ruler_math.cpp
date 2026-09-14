// Coverage for FiberMapRulerMath.hpp: the 1-2-5 ladder the Fiber Map's rulers
// pick tick steps from, the unit a physical ruler labels in, and the label
// formatting. Pure arithmetic, so no widget is involved.

#include <QtTest/QtTest>

#include "FiberMapRulerMath.hpp"

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
