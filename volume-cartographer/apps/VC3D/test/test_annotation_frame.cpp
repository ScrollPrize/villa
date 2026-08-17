#include <QtTest>

#include <array>
#include <optional>

#include "AnnotationFrame.hpp"

using vc3d::annotation::AnnotationFrame;
using vc3d::annotation::deriveAnnotationFrame;

namespace
{
    // A scan whose source frame is 2.4 um; every case below is some store of it.
    constexpr double kSourceUm = 2.4;
    const std::array<double, 3> kSourceDims{20000.0, 20000.0, 68000.0};

    std::array<double, 3> dimsAtLevel(int level)
    {
        const double divisor = std::pow(2.0, static_cast<double>(level));
        return {kSourceDims[0] / divisor,
                kSourceDims[1] / divisor,
                kSourceDims[2] / divisor};
    }
} // namespace

class TestAnnotationFrame : public QObject
{
    Q_OBJECT

private slots:
    // A store sitting at its own level 0 with no tags is already the annotated
    // frame, so nothing is scaled.
    void plainStoreIsItsOwnFrame()
    {
        const auto frame =
            deriveAnnotationFrame(kSourceUm, 0, std::nullopt, kSourceDims);

        QVERIFY(frame.voxelSizeUm.has_value());
        QCOMPARE(*frame.voxelSizeUm, kSourceUm);
        QCOMPARE(frame.factor, 1.0);
        QCOMPARE(frame.extentXyz[2], kSourceDims[2]);
    }

    // The regression this exists for: an untagged store rebased to level 2
    // reports level-2 dimensions alongside a voxel size already multiplied by 4.
    // Reading the resolution back to level 0 while leaving the counts alone
    // described a scroll a quarter of its real size.
    void rebasedUntaggedStoreLiftsDimensions()
    {
        const auto frame =
            deriveAnnotationFrame(kSourceUm * 4.0, 2, std::nullopt, dimsAtLevel(2));

        QVERIFY(frame.voxelSizeUm.has_value());
        QCOMPARE(*frame.voxelSizeUm, kSourceUm);
        QCOMPARE(frame.factor, 4.0);
        QCOMPARE(frame.extentXyz[0], kSourceDims[0]);
        QCOMPARE(frame.extentXyz[1], kSourceDims[1]);
        QCOMPARE(frame.extentXyz[2], kSourceDims[2]);
    }

    // A tagged store: the stamp states the source resolution outright, and the
    // dimensions follow from the ratio to the store's own.
    void stampedResolutionDecidesTheFrame()
    {
        const auto frame =
            deriveAnnotationFrame(kSourceUm * 4.0, 0, kSourceUm, dimsAtLevel(2));

        QVERIFY(frame.voxelSizeUm.has_value());
        QCOMPARE(*frame.voxelSizeUm, kSourceUm);
        QCOMPARE(frame.factor, 4.0);
        QCOMPARE(frame.extentXyz[2], kSourceDims[2]);
    }

    // The stamp outranks the pyramid position, so a store that is both rebased
    // and tagged still lands on one self-consistent grid — the case where
    // composing the two levels arithmetically would disagree with itself.
    void stampedResolutionOutranksRebase()
    {
        const auto frame =
            deriveAnnotationFrame(kSourceUm * 4.0, 2, kSourceUm, dimsAtLevel(2));

        QVERIFY(frame.voxelSizeUm.has_value());
        QCOMPARE(*frame.voxelSizeUm, kSourceUm);
        QCOMPARE(frame.factor, 4.0);
        QCOMPARE(frame.extentXyz[2], kSourceDims[2]);

        // Whatever the factor, resolution times count is the same physical
        // extent as the source scan: that is what makes the grid consistent.
        QCOMPARE(*frame.voxelSizeUm * frame.extentXyz[2],
                 kSourceUm * kSourceDims[2]);
    }

    // A stamp that is not a power-of-two downsample of the store is still
    // honoured; nothing here assumes a pyramid.
    void nonPowerOfTwoStampIsHonoured()
    {
        const auto frame =
            deriveAnnotationFrame(7.5, 0, 2.5, {100.0, 200.0, 300.0});

        QVERIFY(frame.voxelSizeUm.has_value());
        QCOMPARE(*frame.voxelSizeUm, 2.5);
        QCOMPARE(frame.factor, 3.0);
        QCOMPARE(frame.extentXyz[0], 300.0);
        QCOMPARE(frame.extentXyz[2], 900.0);
    }

    // Two stores of one scan at different levels must agree on the frame, which
    // is what lets a level switch leave derived geometry alone.
    void differentLevelsOfOneScanAgree()
    {
        const auto atLevel0 =
            deriveAnnotationFrame(kSourceUm, 0, std::nullopt, kSourceDims);
        const auto atLevel2 =
            deriveAnnotationFrame(kSourceUm * 4.0, 2, std::nullopt, dimsAtLevel(2));

        QCOMPARE(*atLevel0.voxelSizeUm, *atLevel2.voxelSizeUm);
        for (int axis = 0; axis < 3; ++axis) {
            QCOMPARE(atLevel0.extentXyz[axis], atLevel2.extentXyz[axis]);
        }
    }

    // Unusable inputs report "unknown" rather than a plausible-looking guess.
    void unusableInputsStayUnknown()
    {
        const auto noVoxel =
            deriveAnnotationFrame(0.0, 0, std::nullopt, kSourceDims);
        QVERIFY(!noVoxel.voxelSizeUm.has_value());
        QCOMPARE(noVoxel.factor, 1.0);

        const auto noDims =
            deriveAnnotationFrame(kSourceUm * 4.0, 2, std::nullopt, {0.0, 0.0, 0.0});
        QVERIFY(noDims.voxelSizeUm.has_value());
        QCOMPARE(noDims.extentXyz[0], 0.0);
        QCOMPARE(noDims.extentXyz[2], 0.0);

        // A nonsense stamp falls through to the store's own reading rather than
        // being taken at face value.
        const auto badStamp =
            deriveAnnotationFrame(kSourceUm * 4.0, 2, -1.0, dimsAtLevel(2));
        QVERIFY(badStamp.voxelSizeUm.has_value());
        QCOMPARE(*badStamp.voxelSizeUm, kSourceUm);
        QCOMPARE(badStamp.factor, 4.0);
    }
};

QTEST_APPLESS_MAIN(TestAnnotationFrame)
#include "test_annotation_frame.moc"
