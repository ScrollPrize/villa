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

// --- Detectors (placeholders: geometry passes not yet implemented) ---
// Each currently counts the objects it would consider, writes nothing,
// and reports "not implemented" in problems. Signatures and parameter
// structs are the intended final shapes so callers can be wired now.

struct PatchOverlapParams {
    double maxDistanceVoxels = 2.0;    // surface-to-surface tolerance
    int64_t minMatchedPoints = 16;     // evidence threshold for a pair
    std::string edgeKind = "same-surface";
    bool replaceStalePending = true;
};
// Patch <-> patch: surfaces physically connected along the same sheet.
// Intended implementation: SurfacePatchIndex over tifxyz grids, as in
// vc_seg_add_overlap, retargeted from overlapping.json to graph edges.
DetectorReport linkPatchOverlaps(graph::GraphDb& db, const PatchOverlapParams& params = {},
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
