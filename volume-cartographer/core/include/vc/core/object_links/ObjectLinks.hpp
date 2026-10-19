#pragma once

// Relationship discovery over immutable geometry in ObjectStore. Detectors
// select stored objects, compute outside write transactions, and save pending
// edges in the same store. Source files are only used by ingestion, never here.
// An empty UUID selection means all objects of the relevant kind. Unknown or
// wrong-kind UUIDs are errors. Inputs changed during discovery abort publication.

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "vc/core/objects/ObjectStore.hpp"
#include "vc/core/util/Progress.hpp"

namespace vc::object_links {

// Detector identities recorded in edge meta ("detector.name"). A
// detector may replace only its own still-pending edges on re-runs.
inline constexpr const char* kPatchOverlapDetector = "patch-overlap";
inline constexpr const char* kFiberPatchDetector = "fiber-patch";
inline constexpr const char* kFiberFiberDetector = "fiber-fiber";
inline constexpr const char* kTrackFiberDetector = "track-fiber";
inline constexpr const char* kTrackTrackDetector = "track-track";
inline constexpr const char* kTrackPatchDetector = "track-patch";
inline constexpr const char* kPointCollectionPatchDetector = "point-collection-patch";

// One proposed connection. `kind` is the edge kind metadata to store
// (detectors default it to "same-surface"); `score` and `meta` are
// recorded under meta.detector / meta respectively.
struct LinkCandidate {
    std::string aUuid, bUuid;
    std::string kind = "same-surface";
    objects::Anchor aAnchor, bAnchor;
    std::optional<double> score;               // detector confidence/quality, if meaningful
    std::string meta = "{}";                   // extra JSON object merged into edge meta
    std::vector<std::string> tags;
};

struct DetectorReport {
    int64_t objectsConsidered = 0;   // objects the pass examined
    int64_t candidatesFound = 0;     // candidates produced (before dedupe)
    int64_t edgesCreated = 0;
    int64_t edgesUpdated = 0;
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
//   over candidate.meta, and starts life 'pending' (vc_objects default);
// - candidates already present (same kind, endpoints, anchor indexes)
//   refresh this detector's pending edges when replacement is enabled,
//   preserving their UUIDs and graph annotations; unchanged edges are skipped;
// - when replaceStalePending is true, this detector's own *pending*
//   edges (matched by meta.detector.name) that are not among the
//   candidates are removed first. The scan covers the edge kinds
//   present in this batch (a detector re-run emits its kinds even
//   when individual pairs change; an empty batch removes nothing).
//   Approved, rejected, manually created, and other detectors'
//   edges are never touched by detector reconciliation.
// - with staleScope, stale removal uses its explicit kinds and requires
//   both endpoints in its object set, including for an empty batch.
// - a candidate with a changed nonempty meta.anchor_index_space replaces
//   this detector's matching pending edge when stale replacement is enabled.
DetectorReport commitCandidates(objects::ObjectStore& db, const std::string& detectorName, int detectorVersion,
                                const std::vector<LinkCandidate>& candidates,
                                bool replaceStalePending = true,
                                const StaleEdgeScope* staleScope = nullptr,
                                const util::ProgressCallback& progress = {});

// Same operation in a caller-owned transaction. The caller commits/rolls back;
// permits atomic registration of new objects together with their links.
DetectorReport commitCandidates(objects::ObjectStore& db, objects::WriteTxn& txn,
                                const std::string& detectorName, int detectorVersion,
                                const std::vector<LinkCandidate>& candidates,
                                bool replaceStalePending = true,
                                const StaleEdgeScope* staleScope = nullptr,
                                const util::ProgressCallback& progress = {});

// --- Detectors ---
struct PointCollectionPatchParams {
    std::vector<std::string> collectionUuids, patchUuids;
    double tolerance = 2.5;       // working voxels, matching fit_spiral's default
    double pointScale = 1.0;      // source XYZ -> working XYZ
    double patchScale = 1.0;
    int windowPoints = 1;         // even values use the next odd centered window
    int windowMinPoints = 1;      // clipped at collection ends
    // Override the ingestion-time role stored in geometry attributes.
    std::optional<bool> absoluteWinding;
    std::string edgeKind = "point-patch";
    bool replaceStalePending = true;
    util::ProgressCallback progress;
};

// Stored point collections -> stored patches. Anchors retain point IDs and
// source XYZ. General collections prefer largest valid-quad area, then distance;
// between_patches__A__B collections prefer the nearest within that named pair.
DetectorReport linkPointCollectionsToPatches(objects::ObjectStore& db,
    const PointCollectionPatchParams& params = {}, const std::atomic_bool* cancel = nullptr);

// PCL->patch, patch overlap, fiber->patch, track->track and track->patch are implemented; the rest are placeholders that
// count the objects they would consider, write nothing, and report
// "not implemented". Signatures and parameter structs are final
// shapes so callers can be wired now.

// Parameters mirror the patch-merging tool this detector is based on
// (spiral-fitting's patch_merger, merging-patches branch). Distances
// are in base-volume voxels unless noted.
struct PatchOverlapParams {
    std::vector<std::string> patchUuids;
    int workers = 0; // 0: auto, at most 8; positive: requested search concurrency
    util::ProgressCallback progress; // optional; no terminal output from the library
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
// Stored patches on the same sheet, using 3D correspondences, robust UV fitting
// and dense overlap agreement. Accepted pairs become pending edges.
DetectorReport linkPatchOverlaps(objects::ObjectStore& db,
    const PatchOverlapParams& params = {}, const std::atomic_bool* cancel = nullptr);

// Shared parameters for fiber/track->patch auto-linking, ported from
// vc::constraints::AutoLink. The historical review results apply to
// the original ribbon, not this reduced-width variant. Distances are in
// volume voxels. The strip is deliberately reduced relative to the
// original 7-row ribbon: stripRowsEachSide rows at stripSpacing on
// each side of the centerline (default: one row at 20 vx each side).
struct LinePatchParams {
    std::vector<std::string> patchUuids;
    util::ProgressCallback progress; // optional; no terminal output from the library
    // Isotropic coordinate multipliers: source XYZ * scale = working XYZ.
    // All detector distances are working voxels. Working XYZ multiplied by
    // workingToBaseScale / ctToBaseScale addresses selected CT voxels.
    // ctToBaseScale includes the selected level; no extra pyramid scaling.
    double patchScale = 1.0;
    double workingToBaseScale = 1.0;
    double ctToBaseScale = 1.0;
    int ctLevel = 0;
    // Optional Lasagna reference source for an otherwise identical strip/link
    // pipeline. Empty selects CT. Lasagna coordinates use workingToBaseScale.
    std::string normalsManifest;
    // Discovery: dense line point -> patch surface distance.
    double proximity = 1.5;
    std::size_t minContacts = 2;
    // Normal-field ribbon around the line.
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
// The shared rules apply to both fiber and track centerlines.
struct FiberPatchParams : LinePatchParams {
    std::vector<std::string> fiberUuids;
    // Optional disposable strip cache; empty keeps all work in memory.
    std::filesystem::path cacheDirectory;
    double fiberScale = 1.0;
    // When cacheDirectory is set, strips use <uuid>.cbor entries.
    // Geometry, strip parameters, normal method, source location and source
    // metadata invalidate the cache. CT/normal voxel data is assumed immutable;
    // set this after editing it in place without changing its metadata.
    bool rebuildStrips = false;
};

// Stored fibers -> stored patches. CT or Lasagna supplies the normal field.
// Only line points between projected endpoint controls participate. Anchors use
// raw line indices and source XYZ. Failed inputs are excluded from stale removal.
DetectorReport linkFibersToPatches(objects::ObjectStore& db,
    const std::filesystem::path& ctVolume, const FiberPatchParams& params = {},
    const std::atomic_bool* cancel = nullptr);

struct FiberFiberParams {
    double maxDistanceVoxels = 3.0;  // control/line point proximity for candidate pairs
    std::string edgeKind = "same-surface";
    bool replaceStalePending = true;
};
// Fiber <-> fiber: two fibers lie on the same surface (crossing or
// otherwise co-located), anchored at the nearest control points.
DetectorReport linkFibersToFibers(objects::ObjectStore& db, const FiberFiberParams& params = {},
                                  const std::atomic_bool* cancel = nullptr);

struct TrackFiberParams {
    double maxDistanceVoxels = 3.0;
    double minCoverageFraction = 0.5;
    std::string edgeKind = "same-surface";
    bool replaceStalePending = true;
};
// Track -> fiber: an external (Spiral) track follows the same fiber.
DetectorReport linkTracksToFibers(objects::ObjectStore& db, const TrackFiberParams& params = {},
                                  const std::atomic_bool* cancel = nullptr);

struct TrackTrackParams {
    util::ProgressCallback progress; // optional; no terminal output from the library
    double minAngleDegrees = 30.0;
    double tangentRadiusVoxels = 12.0; // Euclidean radius, not arclength
    int clusterIndexRadius = 4;
    int workers = 1;
    std::string datasetRef;          // empty: all registered datasets
    std::vector<std::string> trackUuids; // empty: all registered tracks
    std::string edgeKind = "crossing";
    bool replaceStalePending = true;
};
// Opposite-family stored tracks sharing voxels within the same dataset.
// datasetRef is an identity/grouping label, never a path to open. Raw local point
// indices and source XYZ anchor each crossing. Failure/cancellation writes nothing.
DetectorReport linkTracksToTracks(objects::ObjectStore& db, const TrackTrackParams& params = {},
                                  const std::atomic_bool* cancel = nullptr);

struct TrackPatchParams : LinePatchParams {
    double trackScale = 1.0; // source track XYZ -> working XYZ
    std::vector<std::string> trackUuids; // optional existing graph-object subset
};
// Uses the same discovery, CT strip, contact gates and growth as fiber->patch.
// Reads stored tracks and patches; anchors use raw indices and source XYZ.
DetectorReport linkTracksToPatches(objects::ObjectStore& db,
    const std::filesystem::path& ctVolume, const TrackPatchParams& params = {},
    const std::atomic_bool* cancel = nullptr);

}  // namespace vc::object_links
