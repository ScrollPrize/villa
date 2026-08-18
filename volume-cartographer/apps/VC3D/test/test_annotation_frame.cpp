#include <QtTest>

#include <array>
#include <optional>

#include "AnnotationFrame.hpp"

#include "vc/core/util/ScrollUmbilicus.hpp"

using vc3d::annotation::AnnotationFrame;
using vc3d::annotation::deriveAnnotationFrame;
using vc3d::annotation::OrientationKey;
using vc3d::annotation::sameAnnotationFrame;
using vc3d::annotation::sameAnnotationGrid;
using vc3d::annotation::sameOrientationKey;
using vc3d::annotation::UmbilicusOrientationMode;
using vc::core::util::UmbilicusScaleSource;

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
    // PHerc0139_ds2.volpkg: two volumes of one scan, byte-identical voxel counts,
    // recorded voxel sizes 2.5% apart. Measured from their meta.json, not
    // constructed, because the whole question this pair settles is what real
    // metadata does.
    const std::array<double, 3> kPHerc0139Dims{6628.0, 6628.0, 19239.0};
    constexpr double kPHerc0139RawUm = 9.596;
    constexpr double kPHerc0139SurfUm = 9.362;

    AnnotationFrame frameAt(double voxelUm)
    {
        return deriveAnnotationFrame(voxelUm, 0, std::nullopt, kPHerc0139Dims);
    }

    AnnotationFrame otherGrid()
    {
        return deriveAnnotationFrame(kPHerc0139RawUm, 0, std::nullopt,
                                     {8174.0, 8174.0, 18946.0});
    }

    OrientationKey appliedKey(double voxelUm, UmbilicusScaleSource source)
    {
        OrientationKey key;
        key.frame = frameAt(voxelUm);
        key.rawVolumeShapeXyz = {6628, 6628, 19239};
        key.mode = UmbilicusOrientationMode::Applied;
        key.scaleSource = source;
        return key;
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

    // Used as a cache key: geometry scaled into one frame is only reusable in a
    // frame that compares equal, so this decides when a cached umbilicus is kept.
    void frameComparisonDecidesReuse()
    {
        using vc3d::annotation::sameAnnotationFrame;

        const auto atLevel0 =
            deriveAnnotationFrame(kSourceUm, 0, std::nullopt, kSourceDims);
        const auto atLevel2 =
            deriveAnnotationFrame(kSourceUm * 4.0, 2, std::nullopt, dimsAtLevel(2));
        QVERIFY(sameAnnotationFrame(atLevel0, atLevel2));

        // A different scan: same voxel size, different extent.
        const auto otherScan = deriveAnnotationFrame(
            kSourceUm, 0, std::nullopt, {20000.0, 20000.0, 40000.0});
        QVERIFY(!sameAnnotationFrame(atLevel0, otherScan));

        // Same extent, different resolution.
        const auto otherResolution =
            deriveAnnotationFrame(7.0, 0, std::nullopt, kSourceDims);
        QVERIFY(!sameAnnotationFrame(atLevel0, otherResolution));

        // A voxel size that round-trips imprecisely is still the same grid.
        const auto rounded = deriveAnnotationFrame(
            kSourceUm + 1e-12, 0, std::nullopt, kSourceDims);
        QVERIFY(sameAnnotationFrame(atLevel0, rounded));

        // Unknown is not equal to known, so a frame that stops being derivable
        // invalidates rather than silently matching.
        const auto unknown =
            deriveAnnotationFrame(0.0, 0, std::nullopt, kSourceDims);
        QVERIFY(!sameAnnotationFrame(atLevel0, unknown));
        QVERIFY(sameAnnotationFrame(unknown, unknown));
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
    // The measurement that decides how destructive a voxel-size-only change may
    // be. Both volumes index the same voxels; only the micrometre label differs.
    void sameGridDifferentVoxelSize()
    {
        const auto raw =
            deriveAnnotationFrame(kPHerc0139RawUm, 0, std::nullopt, kPHerc0139Dims);
        const auto surf =
            deriveAnnotationFrame(kPHerc0139SurfUm, 0, std::nullopt, kPHerc0139Dims);

        QCOMPARE(*raw.voxelSizeUm, kPHerc0139RawUm);
        QCOMPARE(*surf.voxelSizeUm, kPHerc0139SurfUm);
        QCOMPARE(raw.extentXyz, surf.extentXyz);

        // Different frames, same grid: geometry in voxels still means the same
        // thing, so this must not be read as "the map is meaningless here".
        QVERIFY(!sameAnnotationFrame(raw, surf));
        QVERIFY(sameAnnotationGrid(raw, surf));
    }

    void differentCountsAreADifferentGrid()
    {
        const auto here = deriveAnnotationFrame(kSourceUm, 0, std::nullopt, kSourceDims);
        const auto elsewhere =
            deriveAnnotationFrame(kSourceUm, 0, std::nullopt, {20000.0, 20000.0, 40000.0});
        QVERIFY(!sameAnnotationGrid(here, elsewhere));
        QVERIFY(!sameAnnotationFrame(here, elsewhere));

        // An unknown voxel size never silently matches a known one, on either
        // predicate's terms. Not even the grid: without the store's own voxel size
        // there is no factor, so its raw counts are not counts in the annotated
        // frame and the extent is left at zero rather than guessed.
        const auto unknown = deriveAnnotationFrame(0.0, 0, std::nullopt, kSourceDims);
        QVERIFY(!sameAnnotationFrame(here, unknown));
        QVERIFY(!sameAnnotationGrid(here, unknown));
    }

    // Which scale path the umbilicus took is what decides whether the 2.5% voxel
    // size difference above reaches the geometry at all.
    void scalePathsAcrossTheSameGrid()
    {
        vc::core::util::UmbilicusFileInfo stamped;
        stamped.controlPoints = {{100.0f, 100.0f, 200.0f},
                                 {120.0f, 120.0f, 19000.0f}};
        stamped.volumeWidth = 6628;
        stamped.volumeHeight = 6628;
        stamped.volumeSlices = 19239;
        stamped.voxelsizeUm = kPHerc0139RawUm;

        const auto viaDimsRaw = vc::core::util::deriveUmbilicusScale(
            stamped, kPHerc0139Dims, kPHerc0139RawUm);
        const auto viaDimsSurf = vc::core::util::deriveUmbilicusScale(
            stamped, kPHerc0139Dims, kPHerc0139SurfUm);
        QVERIFY(viaDimsRaw.has_value());
        QVERIFY(viaDimsSurf.has_value());
        // Dimensions win over voxel size, and the dimensions are identical, so
        // the factor is too: switching between these volumes moves nothing.
        QCOMPARE(viaDimsRaw->factor, 1.0);
        QCOMPARE(viaDimsSurf->factor, 1.0);

        vc::core::util::UmbilicusFileInfo voxelOnly;
        voxelOnly.controlPoints = stamped.controlPoints;
        voxelOnly.voxelsizeUm = kPHerc0139RawUm;
        const auto viaVoxelRaw = vc::core::util::deriveUmbilicusScale(
            voxelOnly, kPHerc0139Dims, kPHerc0139RawUm);
        const auto viaVoxelSurf = vc::core::util::deriveUmbilicusScale(
            voxelOnly, kPHerc0139Dims, kPHerc0139SurfUm);
        QVERIFY(viaVoxelRaw.has_value());
        QVERIFY(viaVoxelSurf.has_value());
        QCOMPARE(viaVoxelRaw->factor, 1.0);
        // 9.596 / 9.362: here the label does reach the geometry.
        QVERIFY(std::abs(viaVoxelSurf->factor - 1.024995) < 1e-5);
    }

    // The orientation key is what the views were built from, so it must rebuild
    // exactly when the orientation would come out different — and it must reach
    // that answer without re-resolving the umbilicus, since a comparison that read
    // the cached factor would compare the old value against itself.
    void orientationKeyFollowsWhatReachedTheGeometry()
    {
        const auto dims = UmbilicusScaleSource::StampedDimensions;
        const auto voxel = UmbilicusScaleSource::StampedVoxelSize;
        const auto inferred = UmbilicusScaleSource::InferredFromGrid;

        // Identical inputs.
        QVERIFY(sameOrientationKey(appliedKey(9.596, dims), appliedKey(9.596, dims)));

        // Same voxel counts, 2.5% different µm figure. Through dimensions or
        // inference the figure never reached the geometry, so nothing to rebuild.
        QVERIFY(sameOrientationKey(appliedKey(9.596, dims), appliedKey(9.362, dims)));
        QVERIFY(sameOrientationKey(appliedKey(9.596, inferred),
                                   appliedKey(9.362, inferred)));

        // Through a stamped voxel size it did: 9.596/9.362 is a real scale change.
        QVERIFY(!sameOrientationKey(appliedKey(9.596, voxel), appliedKey(9.362, voxel)));

        // Different voxel counts are a different grid regardless.
        auto otherCounts = appliedKey(9.596, dims);
        otherCounts.frame = otherGrid();
        QVERIFY(!sameOrientationKey(appliedKey(9.596, dims), otherCounts));

        // The volume's own shape is in the key because the volume-centre fallback
        // and the legacy reading use it directly.
        auto otherShape = appliedKey(9.596, dims);
        otherShape.rawVolumeShapeXyz = {3314, 3314, 9620};
        QVERIFY(!sameOrientationKey(appliedKey(9.596, dims), otherShape));

        // As is how the umbilicus was read at all.
        auto legacy = appliedKey(9.596, dims);
        legacy.mode = UmbilicusOrientationMode::Legacy;
        QVERIFY(!sameOrientationKey(appliedKey(9.596, dims), legacy));

        // And the registration transform a legacy reading went through.
        auto viaTransform = legacy;
        viaTransform.transformPath = "/vol/transform.json";
        viaTransform.transformSize = 512;
        viaTransform.transformWriteTime = 1000;
        QVERIFY(!sameOrientationKey(legacy, viaTransform));

        // Rewriting the matrix leaves the byte length alone, so size is not
        // identity on its own.
        auto rewritten = viaTransform;
        rewritten.transformWriteTime = 2000;
        QVERIFY(!sameOrientationKey(viaTransform, rewritten));
    }

    // An unknown voxel size means an unknown factor, so the volume's raw counts are
    // not counts in the annotated frame and must not be presented as though they
    // were.
    void anUnknownFactorYieldsNoExtent()
    {
        const auto noVoxelSize =
            deriveAnnotationFrame(0.0, 0, std::nullopt, kSourceDims);
        QVERIFY(!noVoxelSize.voxelSizeUm.has_value());
        QCOMPARE(noVoxelSize.extentXyz[0], 0.0);
        QCOMPARE(noVoxelSize.extentXyz[2], 0.0);

        // A stamped resolution alone does not rescue it: without the store's own
        // voxel size there is no ratio to carry the counts by.
        const auto stampedOnly = deriveAnnotationFrame(0.0, 0, 2.4, kSourceDims);
        QCOMPARE(stampedOnly.extentXyz[2], 0.0);
    }
};

QTEST_APPLESS_MAIN(TestAnnotationFrame)
#include "test_annotation_frame.moc"
