#pragma once

#include <QString>

#include <cstdint>
#include <optional>

#include "vc/core/util/ScrollUmbilicus.hpp"

#include "AnnotationFrame.hpp"

namespace vc3d::fiber_map {

// Everything a Fiber Map layout is derived from, gathered in one value so that
// "what it was built from" and "what a rebuild would use now" are the same shape
// and can simply be compared.
//
// Counters rather than signals: a workspace that may never be opened must cost
// annotation work nothing, so the controller bumps integers and this is read at
// the moments that matter.
struct FiberMapDependencies {
    uint64_t fiberGeneration = 0;
    uint64_t packageGeneration = 0;
    uint64_t umbilicusGeneration = 0;
    // Cheap metadata token over the files the umbilicus resolver depends on,
    // covering a file changing underneath VC3D, which no counter reports.
    QString umbilicusFingerprint;
    vc3d::annotation::AnnotationFrame frame;
    // The layout-shaping settings the build consumed. Dependencies like any
    // other: a map rebuilt under different settings is a different picture,
    // and comparing them — rather than latching a flag when a spinbox moves —
    // is what lets a setting changed and then changed back read as current
    // again.
    int maxNetworks = 0;
    int minFibers = 0;
};

// What a comparison concluded, separately from acting on it.
struct StaleVerdict {
    enum class Action {
        Fresh,
        MarkStale,
        ClearLayout,
    };
    Action action = Action::Fresh;
    // Names what actually changed, for the status line.
    QString reason;
};

// The whole decision, as a function of the two dependency sets.
//
// Three outcomes rather than two, because the changes are not equally severe:
//
//  - A different package, or a different *grid*, leaves the layout meaningless:
//    geometry unrolled over one set of voxels says nothing about another set.
//    Clearing is right.
//  - Changed fibers, a changed umbilicus, changed settings, or a changed voxel
//    size leave it out of date. The banner says so and the map refuses to act,
//    but the geometry survives for the rebuild. The voxel size belongs here
//    whether or not the umbilicus scale came from it: the layout's smoothing
//    sigma, resample step, label pads, panel tick and minimum gap are all
//    physical quantities converted through that figure, so a rebuild would
//    place things differently even where the umbilicus itself would not move.
//
// `latchedReason`, when non-empty, is staleness the holder asserted directly
// rather than derived from these sets — the defense paths that fire when an
// invariant looks violated (a fiber file name that no longer resolves under an
// unchanged generation). Nothing in the sets can prove it wrong, so it survives
// until the layout it describes is rebuilt or cleared. The clearing arms above
// it still outrank it, and a higher-priority derived reason may be *displayed*
// over it while its cause holds — so the holder must keep the latched wording
// in its own field, never in whatever the status line currently says.
// Everything derivable deliberately does NOT latch, which is what lets a
// dependency that changes back — a setting reverted, one scan's volume
// switched away and back — read as current again.
inline StaleVerdict staleVerdictFor(const FiberMapDependencies& built,
                                    const FiberMapDependencies& current,
                                    bool layoutBuilt,
                                    const QString& latchedReason)
{
    StaleVerdict verdict;
    // Nothing built yet is not the same as a layout with no networks: an empty
    // result is still a result, derived from dependencies that go out of date.
    // Before a first build there is nothing to compare against, and comparing
    // anyway would clear on every check.
    if (!layoutBuilt) {
        return verdict;
    }

    // Before the grid, because two projects can share one.
    if (current.packageGeneration != built.packageGeneration) {
        verdict.action = StaleVerdict::Action::ClearLayout;
        verdict.reason = QObject::tr("project changed — press Rebuild layout");
        return verdict;
    }

    if (!vc3d::annotation::sameAnnotationGrid(current.frame, built.frame)) {
        verdict.action = StaleVerdict::Action::ClearLayout;
        verdict.reason = QObject::tr("coordinate frame changed — press Rebuild layout");
        return verdict;
    }

    // Same voxels, different physical scale. Two stores of one scan can disagree
    // here -- PHerc0139_ds2's raw and surf pair are byte-identical in voxel counts
    // and 2.5% apart in recorded voxel size. Not meaningless, so this does not
    // clear; out of date, because every physical tuning parameter the layout used
    // was converted through that figure.
    if (!vc3d::annotation::sameAnnotationFrame(current.frame, built.frame)) {
        verdict.action = StaleVerdict::Action::MarkStale;
        verdict.reason = QObject::tr("voxel size changed — press Rebuild layout");
        return verdict;
    }

    if (!latchedReason.isEmpty()) {
        verdict.action = StaleVerdict::Action::MarkStale;
        verdict.reason = latchedReason;
        return verdict;
    }

    if (current.fiberGeneration != built.fiberGeneration) {
        verdict.action = StaleVerdict::Action::MarkStale;
        verdict.reason = QObject::tr("Fibers changed — press Rebuild layout");
        return verdict;
    }

    // The counter covers what VC3D itself did; the fingerprint covers the file
    // being added, replaced or rewritten underneath it.
    if (current.umbilicusGeneration != built.umbilicusGeneration ||
        current.umbilicusFingerprint != built.umbilicusFingerprint) {
        verdict.action = StaleVerdict::Action::MarkStale;
        verdict.reason = QObject::tr("Umbilicus changed — press Rebuild layout");
        return verdict;
    }

    if (current.maxNetworks != built.maxNetworks ||
        current.minFibers != built.minFibers) {
        verdict.action = StaleVerdict::Action::MarkStale;
        verdict.reason = QObject::tr("Settings changed — press Rebuild layout");
        return verdict;
    }

    return verdict;
}

} // namespace vc3d::fiber_map
