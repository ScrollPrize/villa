#pragma once

// Edge *detection* for the unified metadata graph: sibling layer above
// vc_graph (the store). Detectors read objects from the graph DB, load
// their external geometry, compute candidate connections outside any
// write transaction, and commit them through commitCandidates(), which
// stamps detector provenance into each edge's meta and leaves every
// found edge approval_state='pending' until reviewed.
//
// vc_graph itself stays a metadata-only store; everything geometric
// lives here (and in vc_core). Detector entry points are placeholders
// until their geometry passes land; the commit plumbing is final.

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "vc/core/graph/GraphDb.hpp"

namespace vc::graph_links {

// Detector identities recorded in edge meta ("detector.name"). A
// detector may replace only its own still-pending edges on re-runs.
inline constexpr const char* kPatchOverlapDetector = "patch-overlap";
inline constexpr const char* kFiberPatchDetector = "fiber-patch";
inline constexpr const char* kFiberFiberDetector = "fiber-fiber";
inline constexpr const char* kTrackFiberDetector = "track-fiber";
inline constexpr const char* kTrackTrackDetector = "track-track";
inline constexpr const char* kTrackPatchDetector = "track-patch";

// One proposed connection. `kind` is the edge kind metadata to store
// (detectors default it to "same-surface"); `score` and `meta` are
// recorded under meta.detector / meta respectively.
struct LinkCandidate {
    std::string aUuid, bUuid;
    std::string kind = "same-surface";
    graph::Anchor aAnchor, bAnchor;
    std::optional<double> score;               // detector confidence/quality, if meaningful
    std::string meta = "{}";                   // extra JSON object merged into edge meta
    std::vector<std::string> tags;
};

struct DetectorReport {
    int64_t objectsConsidered = 0;   // objects the pass examined
    int64_t candidatesFound = 0;     // candidates produced (before dedupe)
    int64_t edgesCreated = 0;
    int64_t edgesSkippedExisting = 0;
    int64_t staleRemoved = 0;        // own pending edges replaced on re-run
    std::vector<std::string> problems;
};

// Optional stale-edge scope for a partial detector run. Both endpoints
// must be in objectUuids; empty sets deliberately remove nothing.
struct StaleEdgeScope {
    std::vector<std::string> edgeKinds;
    std::vector<std::string> objectUuids;
};

// Commits candidates as edges in one write transaction:
// - each new edge gets meta.detector = {name, version, score?} merged
//   over candidate.meta, and starts life 'pending' (vc_graph default);
// - candidates already present (same kind, endpoints, anchor indexes)
//   are skipped, whatever their approval state or provenance;
// - when replaceStalePending is true, this detector's own *pending*
//   edges (matched by meta.detector.name) that are not among the
//   candidates are removed first. The scan covers the edge kinds
//   present in this batch (a detector re-run emits its kinds even
//   when individual pairs change; an empty batch removes nothing).
//   Approved, rejected, manually created, and other detectors'
//   edges are never touched.
// - with staleScope, stale removal uses its explicit kinds and requires
//   both endpoints in its object set, including for an empty batch.
DetectorReport commitCandidates(graph::GraphDb& db, const std::string& detectorName, int detectorVersion,
                                const std::vector<LinkCandidate>& candidates,
                                bool replaceStalePending = true,
                                const StaleEdgeScope* staleScope = nullptr);

// Same operation in a caller-owned transaction. The caller commits/rolls back;
// permits atomic registration of new objects together with their links.
DetectorReport commitCandidates(graph::GraphDb& db, graph::WriteTxn& txn,
                                const std::string& detectorName, int detectorVersion,
                                const std::vector<LinkCandidate>& candidates,
                                bool replaceStalePending = true,
                                const StaleEdgeScope* staleScope = nullptr);

// --- Detectors ---
// Patch overlap, fiber->patch and track->track are implemented; the rest are placeholders that
// count the objects they would consider, write nothing, and report
// "not implemented". Signatures and parameter structs are final
// shapes so callers can be wired now.

// Parameters mirror the patch-merging tool this detector is based on
// (spiral-fitting's patch_merger, merging-patches branch). Distances
// are in base-volume voxels unless noted.
struct PatchOverlapParams {
    // 3D distance within which two surface points are the same sheet.
    // Used for correspondence search and (with uvInlierTolerance) the
    // dense 3D agreement radius.
    double tolerance = 2.0;
    // Spacing (voxels) between sampled grid points per patch.
    double sampleSpacing = 4.0;
    // One correspondence is kept per seed-UV cell of this size.
    double thinningSpacing = 2.0;
    std::size_t maxCorrespondences = 4096;
    // UV residual (voxels) tolerated by the 2D pose RANSAC/refit.
    double uvInlierTolerance = 3.0;
    std::size_t minInliers = 32;
    // Minimum extents (max-min projection onto the inliers' principal
    // axes in seed UV space) for the overlap to count as a 2D region
    // rather than a sliver: major = length, minor = "thickness".
    double minMajorSpread = 20.0;
    double minMinorSpread = 5.0;
    double maxRefitRms = 2.0;
    // Fraction of dense overlap samples that must agree in 3D within
    // tolerance + uvInlierTolerance under the fitted pose.
    double minAgreementRatio = 0.6;
    // Minimum of the two one-sided overlap fractions (share of each
    // patch's samples agreeing with the other). 0 disables.
    double minOverlapFraction = 0.0;
    double ransacConfidence = 0.999;
    std::size_t ransacMaxHypotheses = 512;
    bool allowReflection = true;
    std::string edgeKind = "same-surface";
    bool replaceStalePending = true;
};
// Patch <-> patch: surfaces physically connected along the same sheet.
// Loads every tifxyz segment directory under patchesDir, registers
// patch objects by their meta.json UUIDs where missing, and for each
// pair: harvests 3D-proximity correspondences, fits a 2D rigid
// (optionally reflected) pose between parameterizations by two-point
// RANSAC gated on 3D tangent-frame agreement, refits with Huber
// weights, requires the inlier spread gates and dense 3D agreement,
// and commits accepted pairs as pending edges with fit provenance
// (inliers, RMS, spreads, agreement) in edge meta.
DetectorReport linkPatchOverlaps(graph::GraphDb& db, const std::filesystem::path& patchesDir,
                                 const PatchOverlapParams& params = {},
                                 const std::atomic_bool* cancel = nullptr);

// Parameters for the fiber->patch auto-linker, ported from
// vc::constraints::AutoLink. The historical review results apply to
// the original ribbon, not this reduced-width variant. Distances are in
// volume voxels. The strip is deliberately reduced relative to the
// original 7-row ribbon: stripRowsEachSide rows at stripSpacing on
// each side of the centerline (default: one row at 20 vx each side).
struct FiberPatchParams {
    // Isotropic coordinate multipliers: source XYZ * scale = working XYZ.
    // All detector distances are working voxels. Working XYZ multiplied by
    // workingToBaseScale / ctToBaseScale addresses selected CT voxels.
    // ctToBaseScale includes the selected level; no extra pyramid scaling.
    double fiberScale = 1.0;
    double patchScale = 1.0;
    double workingToBaseScale = 1.0;
    double ctToBaseScale = 1.0;
    int ctLevel = 0;
    // Optional Lasagna reference source for an otherwise identical strip/link
    // pipeline. Empty selects CT. Lasagna coordinates use workingToBaseScale.
    std::string normalsManifest;
    // Discovery: fiber dense-line point -> patch surface distance.
    double proximity = 1.5;
    std::size_t minContacts = 2;
    // Fiber strip (normal-field ribbon around the fiber).
    double stripSpacing = 20.0;   // column spacing and row spacing (vx)
    int stripRowsEachSide = 1;    // 1 = centerline +/- one row (3 rows total)
    double stripMarchStep = 5.0;  // marching step across the sheet (vx)
    // Anchors and growth (AutoLinkSettings).
    double contactGap = 40.0;        // arclength gap splitting contact runs
    double anchorContact = 75.0;     // min merged contact run length (vx)
    double sameSheet = 3.0;          // vertex-surface distance counted as same sheet
    double adjacentLow = 4.5, adjacentHigh = 16.0;  // adjacent-sheet distance band
    double sameNormal = 0.7, adjacentNormal = 0.85; // |cos| normal agreement
    double growSame = 0.1;           // min same-sheet vertex fraction vs linked set
    double growRatio = 2.0;          // same fraction must exceed ratio * adjacent
    std::array<double, 4> stripWeights{1.0, 0.7, 0.4, 0.15};  // trust by row from centerline
    double stripGateRatio = 1.5;     // anchor: strip same > ratio * adjacent + margin
    double stripGateMargin = 0.0;
    double stripMargin = 20.0;       // bbox margin for strip/patch queries
    int maxRounds = 8;
    std::string edgeKind = "same-surface";
    bool replaceStalePending = true;
};
// Fiber -> patch auto-linking, a port of vc::constraints::autoLinkFiber
// (removed constraint-graph workspace; see ConstraintGraph.md in git
// history). Loads vc3d_fiber JSON files from fibersDir and tifxyz
// segment dirs from patchesDir, registers fiber/patch objects by UUID
// where missing, and per fiber: discovers candidate patches by dense
// line-point proximity, builds the (reduced) normal-field strip from
// CT intensities at ctVolume, links long-contact patches
// that agree with the strip as anchors, then grows the linked set by
// same-sheet overlap until stable. Linked pairs become pending edges
// (fiber anchor at the control point nearest the median contact) with
// the linker measurements as provenance. Source geometry is scaled only
// in memory; anchor positions and registered object geometry remain in
// source coordinates. Provenance records the coordinate multipliers.
// Cancellation returns a problem without committing edges. Failed fibers
// and unreadable patches are excluded from stale-edge removal, as are
// objects outside these input directories.
DetectorReport linkFibersToPatches(graph::GraphDb& db, const std::filesystem::path& fibersDir,
                                   const std::filesystem::path& patchesDir,
                                   const std::filesystem::path& ctVolume,
                                   const FiberPatchParams& params = {},
                                   const std::atomic_bool* cancel = nullptr);

struct FiberFiberParams {
    double maxDistanceVoxels = 3.0;  // control/line point proximity for candidate pairs
    std::string edgeKind = "same-surface";
    bool replaceStalePending = true;
};
// Fiber <-> fiber: two fibers lie on the same surface (crossing or
// otherwise co-located), anchored at the nearest control points.
DetectorReport linkFibersToFibers(graph::GraphDb& db, const FiberFiberParams& params = {},
                                  const std::atomic_bool* cancel = nullptr);

struct TrackFiberParams {
    double maxDistanceVoxels = 3.0;
    double minCoverageFraction = 0.5;
    std::string edgeKind = "same-surface";
    bool replaceStalePending = true;
};
// Track -> fiber: an external (Spiral) track follows the same fiber.
DetectorReport linkTracksToFibers(graph::GraphDb& db, const TrackFiberParams& params = {},
                                  const std::atomic_bool* cancel = nullptr);

struct TrackTrackParams {
    // DBM logical path (also accepts .dbm.db) or packed .vctracks directory.
    // Nonempty: discover/register tracks and compute links, atomically.
    // Empty: link already registered tracks as before.
    std::filesystem::path tracksPath;
    std::vector<uint64_t> sourceIds; // optional subset when tracksPath is supplied
    double minAngleDegrees = 30.0;
    double tangentRadiusVoxels = 12.0; // Euclidean radius, not arclength
    int clusterIndexRadius = 4;
    int workers = 1;
    std::string datasetRef;          // empty: all registered datasets
    std::vector<std::string> trackUuids; // empty: all registered tracks
    std::string edgeKind = "crossing";
    bool replaceStalePending = true;
};
// Track <-> track: opposite-family tracks sharing voxels, mirroring
// Spiral's exact-voxel crossing relation. tracksPath discovers the adjacent
// .vctracks geometry for a DBM and registers the input tracks automatically.
// It never reads or writes crossings.npz. Without tracksPath, reads directories
// named by track.datasetRef (relative to the graph DB, or absolute). externalKey
// is the decimal source_id, not the row index. Only registered/selected tracks
// within the same dataset are paired; unknown families are ignored. Point
// indices are raw local indices; anchor positions/directions are XYZ in dataset
// voxel units. Selects one crossing per pair by clustered median and greatest
// endpoint clearance. On load failure or cancellation, no edges are changed.
// Stale removal is limited to selected endpoints and the requested edge kind.
DetectorReport linkTracksToTracks(graph::GraphDb& db, const TrackTrackParams& params = {},
                                  const std::atomic_bool* cancel = nullptr);

struct TrackPatchParams {
    double maxDistanceVoxels = 3.0;
    double minCoverageFraction = 0.5;
    std::string edgeKind = "same-surface";
    bool replaceStalePending = true;
};
// Track -> patch: the track lies on the patch's surface.
DetectorReport linkTracksToPatches(graph::GraphDb& db, const TrackPatchParams& params = {},
                                   const std::atomic_bool* cancel = nullptr);

}  // namespace vc::graph_links
