#include <QtTest>

#include <array>
#include <optional>

#include "AnnotationFrame.hpp"

#include "vc/core/util/ScrollUmbilicus.hpp"

using vc3d::annotation::AnnotationFrame;
using vc3d::annotation::deriveAnnotationFrame;
using vc3d::annotation::sameAnnotationFrame;
using vc3d::annotation::sameAnnotationGrid;

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
} // namespace

class TestAnnotationFrame : public QObject
{
    Q_OBJECT

private slots:
    // A store sitting at its own level 0 with no tags is already the annotated
    // frame, so nothing is scaled.
    void plainStoreIsItsOwnFrame()
    {
        const auto frame = deriveAnnotationFrame(
            kSourceUm, 0, std::nullopt, std::nullopt, kSourceDims);

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
        const auto frame = deriveAnnotationFrame(
            kSourceUm * 4.0, 2, std::nullopt, std::nullopt, dimsAtLevel(2));

        QVERIFY(frame.voxelSizeUm.has_value());
        QCOMPARE(*frame.voxelSizeUm, kSourceUm);
        QCOMPARE(frame.factor, 4.0);
        QCOMPARE(frame.extentXyz[0], kSourceDims[0]);
        QCOMPARE(frame.extentXyz[1], kSourceDims[1]);
        QCOMPARE(frame.extentXyz[2], kSourceDims[2]);
    }

    // The rebase factor is exact — a power of two read off the store's own
    // pyramid position, not a quotient of two stored doubles.
    void rebasedFactorIsExactWithoutAnyResolution()
    {
        const auto frame = deriveAnnotationFrame(
            0.0, 2, std::nullopt, std::nullopt, dimsAtLevel(2));

        // No voxel size anywhere, but the pyramid position alone still lifts
        // the counts: the factor is a statement about grids, not micrometres.
        QVERIFY(!frame.voxelSizeUm.has_value());
        QCOMPARE(frame.factor, 4.0);
        QCOMPARE(frame.extentXyz[2], kSourceDims[2]);
    }

    // A tagged store: the tag's coordinate factor is an exact integer statement
    // and decides the extents; the stamped resolution stays what it is,
    // physical metadata.
    void exactFactorDecidesExtents()
    {
        const auto frame = deriveAnnotationFrame(
            kSourceUm * 4.0, 0, 4.0, kSourceUm, dimsAtLevel(2));

        QVERIFY(frame.voxelSizeUm.has_value());
        QCOMPARE(*frame.voxelSizeUm, kSourceUm);
        QCOMPARE(frame.factor, 4.0);
        QCOMPARE(frame.extentXyz[2], kSourceDims[2]);
    }

    // The finding this redesign answers: a store whose recorded voxel size is
    // rounded (9.596 um against a 2.4 um source gives a ratio of 3.99833...)
    // must still derive its extents through the exact factor, or downstream
    // exact integer-rescale tests refuse a correctly stamped umbilicus over a
    // fractional grid the derivation itself invented.
    void exactFactorOutranksRoundedResolutionRatio()
    {
        const auto exact = deriveAnnotationFrame(
            kPHerc0139RawUm, 0, 4.0, kSourceUm, kPHerc0139Dims);
        QCOMPARE(exact.factor, 4.0);
        QCOMPARE(exact.extentXyz[0], kPHerc0139Dims[0] * 4.0);
        QCOMPARE(exact.extentXyz[2], kPHerc0139Dims[2] * 4.0);
        QCOMPARE(*exact.voxelSizeUm, kSourceUm);

        // Without the exact factor the ratio really is fractional — the case
        // only discriminates because 9.596 / 2.4 is not 4.
        const auto viaRatio = deriveAnnotationFrame(
            kPHerc0139RawUm, 0, std::nullopt, kSourceUm, kPHerc0139Dims);
        QVERIFY(std::abs(viaRatio.factor - 4.0) > 1e-4);
    }

    // A tag whose store carries no physical resolution at all still yields the
    // exact factor; the voxel size is then the store's own carried through it.
    void exactFactorSurvivesMissingResolution()
    {
        const auto frame = deriveAnnotationFrame(
            kPHerc0139RawUm, 0, 4.0, std::nullopt, kPHerc0139Dims);
        QCOMPARE(frame.factor, 4.0);
        QCOMPARE(frame.extentXyz[2], kPHerc0139Dims[2] * 4.0);
        QVERIFY(frame.voxelSizeUm.has_value());
        QCOMPARE(*frame.voxelSizeUm, kPHerc0139RawUm / 4.0);
    }

    // End to end through the shipped path: PHercParis4's real values. The
    // stamped 8174x8174x18946 grid must take Apply at exactly x4 against a
    // level-2 store of the 32693x32693x75784 scan — and through the rounded
    // resolution ratio it would not, which is the conflict between this PR's
    // own earlier fixes that the redesign removes.
    void stampedUmbilicusAppliesThroughExactFactor()
    {
        vc::core::util::UmbilicusFileInfo info;
        info.controlPoints = {{4000.0f, 4000.0f, 500.0f},
                              {4100.0f, 4100.0f, 18000.0f}};
        info.volumeWidth = 8174;
        info.volumeHeight = 8174;
        info.volumeSlices = 18946;
        info.voxelsizeUm = 9.6;

        // The real store: level-2 counts (32693/4 = 8173.25, rounded up), a
        // rounded 9.596 um recorded resolution, an exact x4 coordinate tag.
        const std::array<double, 3> storeDims{8174.0, 8174.0, 18946.0};
        const auto frame =
            deriveAnnotationFrame(9.596, 0, 4.0, 2.4, storeDims);

        const auto scale = vc::core::util::deriveUmbilicusScale(
            info, frame.extentXyz, frame.voxelSizeUm);
        QVERIFY(scale.has_value());
        QCOMPARE(scale->factor, 4.0);
        QCOMPARE(static_cast<int>(scale->source),
                 static_cast<int>(
                     vc::core::util::UmbilicusScaleSource::StampedDimensions));

        const auto action = vc::core::util::decideUmbilicusLoadAction(
            scale, vc::core::util::umbilicusFrameClaim(info),
            frame.extentXyz[2] > 0.0);
        QCOMPARE(static_cast<int>(action),
                 static_cast<int>(vc::core::util::UmbilicusLoadAction::Apply));

        // The fractional extents a resolution ratio would have produced are
        // refused by the same exact test — proof the redesign is load-bearing.
        const auto viaRatio =
            deriveAnnotationFrame(9.596, 0, std::nullopt, 2.4, storeDims);
        const auto refused = vc::core::util::deriveUmbilicusScale(
            info, viaRatio.extentXyz, viaRatio.voxelSizeUm);
        QVERIFY(!refused.has_value() ||
                refused->source !=
                    vc::core::util::UmbilicusScaleSource::StampedDimensions);
    }

    // A stamp that is not a power-of-two downsample of the store is still
    // honoured through the resolution ratio when nothing exact was said;
    // nothing here assumes a pyramid.
    void nonPowerOfTwoStampIsHonoured()
    {
        const auto frame = deriveAnnotationFrame(
            7.5, 0, std::nullopt, 2.5, {100.0, 200.0, 300.0});

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
        const auto atLevel0 = deriveAnnotationFrame(
            kSourceUm, 0, std::nullopt, std::nullopt, kSourceDims);
        const auto atLevel2 = deriveAnnotationFrame(
            kSourceUm * 4.0, 2, std::nullopt, std::nullopt, dimsAtLevel(2));

        QCOMPARE(*atLevel0.voxelSizeUm, *atLevel2.voxelSizeUm);
        for (int axis = 0; axis < 3; ++axis) {
            QCOMPARE(atLevel0.extentXyz[axis], atLevel2.extentXyz[axis]);
        }
    }

    // Used as a cache key: geometry scaled into one frame is only reusable in a
    // frame that compares equal, so this decides when a cached umbilicus is kept.
    void frameComparisonDecidesReuse()
    {
        const auto atLevel0 = deriveAnnotationFrame(
            kSourceUm, 0, std::nullopt, std::nullopt, kSourceDims);
        const auto atLevel2 = deriveAnnotationFrame(
            kSourceUm * 4.0, 2, std::nullopt, std::nullopt, dimsAtLevel(2));
        QVERIFY(sameAnnotationFrame(atLevel0, atLevel2));

        // A different scan: same voxel size, different extent.
        const auto otherScan = deriveAnnotationFrame(
            kSourceUm, 0, std::nullopt, std::nullopt,
            {20000.0, 20000.0, 40000.0});
        QVERIFY(!sameAnnotationFrame(atLevel0, otherScan));

        // Same extent, different resolution.
        const auto otherResolution = deriveAnnotationFrame(
            7.0, 0, std::nullopt, std::nullopt, kSourceDims);
        QVERIFY(!sameAnnotationFrame(atLevel0, otherResolution));

        // A voxel size that round-trips imprecisely is still the same grid.
        const auto rounded = deriveAnnotationFrame(
            kSourceUm + 1e-12, 0, std::nullopt, std::nullopt, kSourceDims);
        QVERIFY(sameAnnotationFrame(atLevel0, rounded));

        // An unknown voxel size never matches a known one on the frame's
        // terms: the physical label differs even though the grid is the same.
        const auto unknownVoxel = deriveAnnotationFrame(
            0.0, 0, std::nullopt, std::nullopt, kSourceDims);
        QVERIFY(!sameAnnotationFrame(atLevel0, unknownVoxel));
        QVERIFY(sameAnnotationFrame(unknownVoxel, unknownVoxel));
    }

    // What each kind of missing input costs, and no more. An untagged level-0
    // store with an unknown voxel size still indexes its own grid — the factor
    // is 1 by construction, not by guess — while a store claiming a foreign
    // frame it cannot be related to yields no counts at all.
    void missingInputsCostExactlyWhatTheyRemove()
    {
        // Unknown voxel size, nothing claiming another frame: the grid is the
        // store's own; only the physical label is unknown.
        const auto noVoxel = deriveAnnotationFrame(
            0.0, 0, std::nullopt, std::nullopt, kSourceDims);
        QVERIFY(!noVoxel.voxelSizeUm.has_value());
        QCOMPARE(noVoxel.factor, 1.0);
        QCOMPARE(noVoxel.extentXyz[2], kSourceDims[2]);
        const auto known = deriveAnnotationFrame(
            kSourceUm, 0, std::nullopt, std::nullopt, kSourceDims);
        QVERIFY(sameAnnotationGrid(known, noVoxel));

        // Unusable dimensions: a factor but nothing to carry with it.
        const auto noDims = deriveAnnotationFrame(
            kSourceUm * 4.0, 2, std::nullopt, std::nullopt, {0.0, 0.0, 0.0});
        QVERIFY(noDims.voxelSizeUm.has_value());
        QCOMPARE(noDims.extentXyz[0], 0.0);
        QCOMPARE(noDims.extentXyz[2], 0.0);

        // A nonsense stamp falls through to the store's own reading rather than
        // being taken at face value.
        const auto badStamp = deriveAnnotationFrame(
            kSourceUm * 4.0, 2, std::nullopt, -1.0, dimsAtLevel(2));
        QVERIFY(badStamp.voxelSizeUm.has_value());
        QCOMPARE(*badStamp.voxelSizeUm, kSourceUm);
        QCOMPARE(badStamp.factor, 4.0);

        // A stamped resolution with no exact factor and no usable store voxel
        // size claims a frame nothing can relate this store's counts to: the
        // factor is unknown and the counts stay zero rather than guessed. The
        // stamp outranks the pyramid position, so the rebase does not answer
        // for it either.
        const auto stampedOnly = deriveAnnotationFrame(
            0.0, 2, std::nullopt, kSourceUm, dimsAtLevel(2));
        QVERIFY(stampedOnly.voxelSizeUm.has_value());
        QCOMPARE(stampedOnly.extentXyz[2], 0.0);
    }

    // The measurement that decides how destructive a voxel-size-only change may
    // be. Both volumes index the same voxels; only the micrometre label differs.
    void sameGridDifferentVoxelSize()
    {
        const auto raw = deriveAnnotationFrame(
            kPHerc0139RawUm, 0, std::nullopt, std::nullopt, kPHerc0139Dims);
        const auto surf = deriveAnnotationFrame(
            kPHerc0139SurfUm, 0, std::nullopt, std::nullopt, kPHerc0139Dims);

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
        const auto here = deriveAnnotationFrame(
            kSourceUm, 0, std::nullopt, std::nullopt, kSourceDims);
        const auto elsewhere = deriveAnnotationFrame(
            kSourceUm, 0, std::nullopt, std::nullopt,
            {20000.0, 20000.0, 40000.0});
        QVERIFY(!sameAnnotationGrid(here, elsewhere));
        QVERIFY(!sameAnnotationFrame(here, elsewhere));
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
};

QTEST_APPLESS_MAIN(TestAnnotationFrame)
#include "test_annotation_frame.moc"
