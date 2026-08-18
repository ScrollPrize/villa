// Coverage for apps/VC3D/FiberMapStaleness.hpp — the decision that says whether a
// built Fiber Map layout is still current, out of date, or meaningless.
//
// This is the layer four rounds of review findings landed in, every one of them a
// dependency recorded when the layout was built and then not re-examined at the
// moment it mattered. The decision is a pure function of two dependency sets
// precisely so that each arm can be asserted here rather than read.

#include <QtTest>

#include "FiberMapStaleness.hpp"

using vc3d::annotation::AnnotationFrame;
using vc3d::fiber_map::FiberMapDependencies;
using vc3d::fiber_map::StaleVerdict;
using vc3d::fiber_map::staleVerdictFor;

namespace
{
    // PHerc0139_ds2.volpkg: two volumes of one scan, byte-identical voxel counts,
    // recorded voxel sizes 2.5% apart. Measured from their meta.json.
    AnnotationFrame frameAt(double voxelUm)
    {
        AnnotationFrame frame;
        frame.voxelSizeUm = voxelUm;
        frame.factor = 1.0;
        frame.extentXyz = {6628.0, 6628.0, 19239.0};
        return frame;
    }

    AnnotationFrame otherGrid()
    {
        AnnotationFrame frame;
        frame.voxelSizeUm = 9.596;
        frame.factor = 1.0;
        frame.extentXyz = {8174.0, 8174.0, 18946.0};
        return frame;
    }

    FiberMapDependencies baseline()
    {
        FiberMapDependencies deps;
        deps.fiberGeneration = 7;
        deps.packageGeneration = 3;
        deps.umbilicusGeneration = 2;
        deps.umbilicusFingerprint = QStringLiteral("umb|1024:99");
        deps.frame = frameAt(9.596);
        return deps;
    }

    StaleVerdict verdict(const FiberMapDependencies& built,
                         const FiberMapDependencies& current,
                         bool layoutBuilt = true,
                         bool alreadyStale = false)
    {
        return staleVerdictFor(built, current, layoutBuilt, alreadyStale,
                              QStringLiteral("already stale"));
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
    // would differ on every check — the recorded frame is default-constructed while
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

    // Checked before the grid, because two projects can share one and the fibers
    // are gone either way.
    void aDifferentPackageOutranksAMatchingGrid()
    {
        auto current = baseline();
        current.packageGeneration += 1;
        current.fiberGeneration += 1;
        const auto result = verdict(baseline(), current);
        QCOMPARE(result.action, StaleVerdict::Action::ClearLayout);
    }

    void differentVoxelCountsClear()
    {
        auto current = baseline();
        current.frame = otherGrid();
        const auto result = verdict(baseline(), current);
        QCOMPARE(result.action, StaleVerdict::Action::ClearLayout);
    }

    // Same voxels, 2.5% different micrometre label. Not meaningless -- the counts
    // are identical -- but not current either: the layout's smoothing sigma, resample
    // step, label pads, panel tick and minimum gap are all physical quantities
    // converted through that figure, so a rebuild would place things differently. An
    // earlier version of this treated the case as a relabel and reported the layout
    // current, which was wrong.
    void voxelSizeOnlyChangeMarksStaleRatherThanClearing()
    {
        auto current = baseline();
        current.frame = frameAt(9.362);
        QCOMPARE(verdict(baseline(), current).action, StaleVerdict::Action::MarkStale);
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

    // Already stale stays stale, and keeps its original reason rather than being
    // relabelled by whichever check happens to run next.
    void alreadyStaleKeepsItsReason()
    {
        const auto deps = baseline();
        const auto result = verdict(deps, deps, true, /*alreadyStale=*/true);
        QCOMPARE(result.action, StaleVerdict::Action::MarkStale);
        QCOMPARE(result.reason, QStringLiteral("already stale"));
    }

    // A clear outranks an existing stale mark: the geometry is not merely out of
    // date, it is about another grid.
    void clearingOutranksAnExistingStaleMark()
    {
        auto current = baseline();
        current.frame = otherGrid();
        const auto result = verdict(baseline(), current, true, /*alreadyStale=*/true);
        QCOMPARE(result.action, StaleVerdict::Action::ClearLayout);
    }
};

QTEST_APPLESS_MAIN(TestFiberMapStaleness)
#include "test_fiber_map_staleness.moc"
