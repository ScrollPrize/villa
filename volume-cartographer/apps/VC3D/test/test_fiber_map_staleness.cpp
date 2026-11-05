// Coverage for apps/VC3D/FiberMapStaleness.hpp — the decision that says whether a
// built Fiber Map layout is still current, out of date, or meaningless.
//
// This is the layer four rounds of review findings landed in, every one of them a
// dependency recorded when the layout was built and then not re-examined at the
// moment it mattered. The decision is a pure function of two dependency sets
// precisely so that each arm can be asserted here rather than read. Staleness is
// derived from the comparison wherever a dependency can express it; the latched
// reason exists only for the holder's invariant-violation defenses.

#include <QtTest>

#include "FiberMapStaleness.hpp"

using vc3d::fiber_map::FiberMapDependencies;
using vc3d::fiber_map::StaleVerdict;
using vc3d::fiber_map::staleVerdictFor;

namespace
{
    FiberMapDependencies baseline()
    {
        FiberMapDependencies deps;
        deps.fiberGeneration = 7;
        deps.packageGeneration = 3;
        deps.umbilicusGeneration = 2;
        deps.umbilicusFingerprint = QStringLiteral("umb|1024:99");
        deps.annotationVolume = QStringLiteral("20260102150214");
        deps.voxelSizeUm = 9.596;
        deps.catalogOrientation = QStringLiteral("z-top-to-bottom 1, left-handed 0");
        return deps;
    }

    StaleVerdict verdict(const FiberMapDependencies& built,
                         const FiberMapDependencies& current,
                         bool layoutBuilt = true,
                         bool latched = false)
    {
        return staleVerdictFor(built, current, layoutBuilt,
                               latched ? QStringLiteral("already stale")
                                       : QString());
    }
} // namespace

class TestFiberMapStaleness : public QObject
{
    Q_OBJECT

private slots:
    void unchangedDependenciesAreFresh()
    {
        const auto deps = baseline();
        const auto result = verdict(deps, deps);
        QCOMPARE(result.action, StaleVerdict::Action::Fresh);
    }

    // Before a first build there is nothing to compare against. Comparing anyway
    // would differ on every check — the recorded dependencies are default-constructed while
    // the current one is populated — and clear the layout indefinitely.
    void nothingBuiltIsNeverStale()
    {
        FiberMapDependencies built;   // all defaults
        const auto result = verdict(built, baseline(), /*layoutBuilt=*/false);
        QCOMPARE(result.action, StaleVerdict::Action::Fresh);
    }

    // ...but an *empty* layout is a built layout. A map built before an umbilicus
    // was attached must notice the attachment rather than reporting "none found"
    // forever, which is the state that started this whole thread.
    void anEmptyLayoutStillGoesStale()
    {
        auto current = baseline();
        current.umbilicusGeneration += 1;
        const auto result = verdict(baseline(), current, /*layoutBuilt=*/true);
        QCOMPARE(result.action, StaleVerdict::Action::MarkStale);
    }

    void aDifferentPackageClears()
    {
        auto current = baseline();
        current.packageGeneration += 1;
        const auto result = verdict(baseline(), current);
        QCOMPARE(result.action, StaleVerdict::Action::ClearLayout);
    }

    // Checked before the volume, because two projects can share one and the fibers
    // are gone either way.
    void aDifferentPackageOutranksAMatchingVolume()
    {
        auto current = baseline();
        current.packageGeneration += 1;
        current.fiberGeneration += 1;
        const auto result = verdict(baseline(), current);
        QCOMPARE(result.action, StaleVerdict::Action::ClearLayout);
    }

    // Another annotation volume is usually the user LOOKING at another volume
    // (the annotation volume follows the current volume), which reverts when
    // they switch back - so it marks stale like every other derived dependency
    // instead of destroying a layout that becomes correct again on return. It
    // outranks the auto-updatable causes, so a fiber save while looking at
    // the other volume does not rebuild the map there.
    void anotherAnnotationVolumeMarksStaleAndReverts()
    {
        auto current = baseline();
        current.annotationVolume = QStringLiteral("20250728140407");
        const auto result = verdict(baseline(), current);
        QCOMPARE(result.action, StaleVerdict::Action::MarkStale);
        QCOMPARE(result.cause, StaleVerdict::Cause::Volume);
        current.fiberGeneration += 1;
        QCOMPARE(verdict(baseline(), current).cause, StaleVerdict::Cause::Volume);
        // ...and switching back reads as current again with no rebuild.
        current = baseline();
        QCOMPARE(verdict(baseline(), current).action, StaleVerdict::Action::Fresh);
    }

    // The causes drive the auto-update decision: a rebuild genuinely fixes
    // fiber and umbilicus staleness, and only those.
    void causesSeparateAutoUpdatableStaleness()
    {
        auto fibers = baseline();
        fibers.fiberGeneration += 1;
        QCOMPARE(verdict(baseline(), fibers).cause, StaleVerdict::Cause::Fibers);

        auto umbilicus = baseline();
        umbilicus.umbilicusFingerprint = QStringLiteral("umb|2|2");
        QCOMPARE(verdict(baseline(), umbilicus).cause,
                 StaleVerdict::Cause::Umbilicus);

        auto voxel = baseline();
        voxel.voxelSizeUm = 9.362;
        QCOMPARE(verdict(baseline(), voxel).cause, StaleVerdict::Cause::VoxelSize);

        const auto deps = baseline();
        QCOMPARE(verdict(deps, deps, true, /*latched=*/true).cause,
                 StaleVerdict::Cause::Latched);
    }

    // Same volume, 2.5% different micrometre label (PHerc0139_ds2's raw and surf
    // pair). Not meaningless, but not current either: the layout's smoothing sigma, resample
    // step, label pads, panel tick and minimum gap are all physical quantities
    // converted through that figure, so a rebuild would place things differently. An
    // earlier version of this treated the case as a relabel and reported the layout
    // current, which was wrong.
    void voxelSizeOnlyChangeMarksStaleRatherThanClearing()
    {
        auto current = baseline();
        current.voxelSizeUm = 9.362;
        QCOMPARE(verdict(baseline(), current).action, StaleVerdict::Action::MarkStale);
        // Round-trip noise is not a change.
        current.voxelSizeUm = 9.596 * (1 + 1e-9);
        QCOMPARE(verdict(baseline(), current).action, StaleVerdict::Action::Fresh);
    }

    void changedFibersMarkStale()
    {
        auto current = baseline();
        current.fiberGeneration += 1;
        const auto result = verdict(baseline(), current);
        QCOMPARE(result.action, StaleVerdict::Action::MarkStale);
    }

    void anAttachedUmbilicusMarksStale()
    {
        auto current = baseline();
        current.umbilicusGeneration += 1;
        QCOMPARE(verdict(baseline(), current).action, StaleVerdict::Action::MarkStale);
    }

    // The fingerprint covers the file changing underneath VC3D, which no counter
    // reports.
    void aRewrittenUmbilicusFileMarksStale()
    {
        auto current = baseline();
        current.umbilicusFingerprint = QStringLiteral("umb|2048:101");
        QCOMPARE(verdict(baseline(), current).action, StaleVerdict::Action::MarkStale);
    }

    // The catalog's orientation of the volume is a dependency like the
    // umbilicus: changed, the winding sense may read differently; the same
    // again is fresh. A volume the catalog does not publish has none.
    void aChangedCatalogOrientationMarksStale()
    {
        auto current = baseline();
        current.catalogOrientation = QStringLiteral("z-top-to-bottom 0, left-handed 0");
        const StaleVerdict changed = verdict(baseline(), current);
        QCOMPARE(changed.action, StaleVerdict::Action::MarkStale);
        QCOMPARE(changed.cause, StaleVerdict::Cause::Catalog);
        current.catalogOrientation = baseline().catalogOrientation;
        QCOMPARE(verdict(baseline(), current).action, StaleVerdict::Action::Fresh);
        auto untagged = baseline();
        untagged.catalogOrientation.clear();
        QCOMPARE(verdict(untagged, untagged).action, StaleVerdict::Action::Fresh);
    }

    // A latched reason stays, and keeps its original wording rather than being
    // relabelled by whichever check happens to run next. Latching is reserved
    // for staleness the holder asserted directly — the invariant-violation
    // defenses — which nothing in the dependency sets can prove wrong.
    void aLatchedReasonKeepsItsWording()
    {
        const auto deps = baseline();
        const auto result = verdict(deps, deps, true, /*latched=*/true);
        QCOMPARE(result.action, StaleVerdict::Action::MarkStale);
        QCOMPARE(result.reason, QStringLiteral("already stale"));
    }

    // The unknown->known voxel-size transition: gaining one later is the same
    // volume under a new physical label — out of date, never meaningless.
    void gainingAVoxelSizeMarksStaleWithoutClearing()
    {
        auto built = baseline();
        built.voxelSizeUm.reset();
        const auto result = verdict(built, baseline());
        QCOMPARE(result.action, StaleVerdict::Action::MarkStale);
        QVERIFY(result.reason.contains(QStringLiteral("voxel size")));

        // And symmetrically: losing it is a relabel of the same volume too.
        auto current = baseline();
        current.voxelSizeUm.reset();
        QCOMPARE(verdict(baseline(), current).action, StaleVerdict::Action::MarkStale);
    }

    // A clear outranks an existing stale mark: a different package has
    // different fibers entirely.
    void clearingOutranksAnExistingStaleMark()
    {
        auto current = baseline();
        current.packageGeneration += 1;
        const auto result = verdict(baseline(), current, true, /*latched=*/true);
        QCOMPARE(result.action, StaleVerdict::Action::ClearLayout);
    }

    // A volume mismatch still outranks the latch for display: the volume
    // reason shows while it holds, and the latch resurfaces when it reverts.
    void volumeMismatchDisplaysOverALatch()
    {
        auto current = baseline();
        current.annotationVolume = QStringLiteral("20250728140407");
        const auto result = verdict(baseline(), current, true, /*latched=*/true);
        QCOMPARE(result.action, StaleVerdict::Action::MarkStale);
        QCOMPARE(result.cause, StaleVerdict::Cause::Volume);
    }
};

QTEST_APPLESS_MAIN(TestFiberMapStaleness)
#include "test_fiber_map_staleness.moc"
