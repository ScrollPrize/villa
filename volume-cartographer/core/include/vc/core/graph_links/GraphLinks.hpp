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
DetectorReport commitCandidates(graph::GraphDb& db, const std::string& detectorName, int detectorVersion,
                                const std::vector<LinkCandidate>& candidates,
                                bool replaceStalePending = true);

// --- Detectors ---
// linkPatchOverlaps is implemented; the rest are placeholders that
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

struct FiberPatchParams {
    double maxDistanceVoxels = 3.0;    // fiber line point -> surface distance
    double minCoverageFraction = 0.5;  // fraction of the fiber within tolerance
    std::string edgeKind = "same-surface";
    bool replaceStalePending = true;
};
// Fiber -> patch: the fiber lies on the patch's surface.
DetectorReport linkFibersToPatches(graph::GraphDb& db, const FiberPatchParams& params = {},
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
    double maxDistanceVoxels = 2.0;
    std::string edgeKind = "crossing";
    bool replaceStalePending = true;
};
// Track <-> track: opposite-family tracks sharing voxels, mirroring
// the crossing relation Spiral materialises today as a packed CSR
// sparse array (.crossings sidecars) next to its track datasets.
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
