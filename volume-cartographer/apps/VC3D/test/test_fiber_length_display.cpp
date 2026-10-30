// The fiber-length display helpers behind the Fibers dock's len column and
// total, the Fiber Map dock's len column and totals, and the status bar's
// fiber-length readout: centimetres only with a usable voxel size, voxels
// otherwise, and the column header naming whichever unit the cells are in.

#include "FiberLengthDisplay.hpp"

#include <QtTest/QtTest>

#include <cmath>
#include <limits>
#include <optional>

using namespace vc3d::fiber_length;

class FiberLengthDisplayTest : public QObject
{
    Q_OBJECT

private slots:
    void physicalScaleNeedsAPositiveFiniteVoxelSize()
    {
        QVERIFY(hasPhysicalScale(2.4));
        QVERIFY(!hasPhysicalScale(std::nullopt));
        QVERIFY(!hasPhysicalScale(0.0));
        QVERIFY(!hasPhysicalScale(-7.91));
        QVERIFY(!hasPhysicalScale(std::numeric_limits<double>::quiet_NaN()));
        QVERIFY(!hasPhysicalScale(std::numeric_limits<double>::infinity()));
    }

    void unitFollowsTheScale()
    {
        QCOMPARE(unitLabel(2.4), QStringLiteral("cm"));
        QCOMPARE(unitLabel(std::nullopt), QStringLiteral("vx"));
        QCOMPARE(unitLabel(0.0), QStringLiteral("vx"));
    }

    void displayValueConvertsOnlyWithAScale()
    {
        // 10000 µm per cm: 5000 voxels of 2.4 µm are 1.2 cm.
        QCOMPARE(displayValue(5000.0, 2.4), 1.2);
        QCOMPARE(displayValue(5000.0, std::nullopt), 5000.0);
        // A rebased store's 7.91 µm frame.
        QVERIFY(std::abs(displayValue(1000.0, 7.91) - 0.791) < 1e-12);
    }

    void cellValuesUseTwoDecimalsInCmAndOneInVx()
    {
        QCOMPARE(formatValue(5000.0, 2.4), QStringLiteral("1.20"));
        QCOMPARE(formatValue(412.34, std::nullopt), QStringLiteral("412.3"));
        // A short span still prints as a cm figure, never flips unit.
        QCOMPARE(formatValue(200.0, 2.4), QStringLiteral("0.05"));
        // Whole voxels on request (the map's distances).
        QCOMPARE(formatValue(1234.6, std::nullopt, 0), QStringLiteral("1235"));
        QCOMPARE(formatValue(std::numeric_limits<double>::quiet_NaN(), 2.4),
                 QStringLiteral("-"));
    }

    void labelsCarryTheUnit()
    {
        QCOMPARE(formatLength(5000.0, 2.4), QStringLiteral("1.20 cm"));
        QCOMPARE(formatLength(412.34, std::nullopt), QStringLiteral("412.3 vx"));
        QCOMPARE(formatLength(1234.6, std::nullopt, 0), QStringLiteral("1235 vx"));
        QCOMPARE(formatLength(0.0, 2.4), QStringLiteral("0.00 cm"));
    }

    void headersNameTheUnitOfTheirCells()
    {
        QCOMPARE(columnHeader(QStringLiteral("len"), 2.4), QStringLiteral("len (cm)"));
        QCOMPARE(columnHeader(QStringLiteral("len"), std::nullopt), QStringLiteral("len (vx)"));
        QCOMPARE(columnHeader(QStringLiteral("Len"), 0.0), QStringLiteral("Len (vx)"));
    }
};

QTEST_APPLESS_MAIN(FiberLengthDisplayTest)
#include "test_fiber_length_display.moc"
