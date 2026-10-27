#pragma once
#include "vc_object_store/objects/VolumeTransforms.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace vc::objects {
// Oriented slab: local coordinates are projections onto u, v and normal about origin.
struct RegionSlice {
    std::array<double, 3> origin{0, 0, 0};
    std::array<double, 3> u{1, 0, 0}, v{0, 1, 0}, normal{0, 0, 1};
    std::array<double, 6> localBounds{-1, -1, -1, 1, 1, 1};
};
struct RegionOptions {
    // Object kind to intersect. Polyline kinds: "fiber" (nonempty line_points,
    // otherwise control_points) and "track" (points). "surface" returns the
    // polylines where its valid triangles cross the slab's mid-plane, clipped to
    // the u/v bounds; family and minLength do not apply to surfaces.
    std::string kind = "fiber";
    uint64_t maximumObjects = 0; // zero means unlimited; selected is then ignored
    std::optional<std::string> selected;
    std::optional<std::string> family; // null = all; otherwise exact annotation
    double minLength = 0; // source annotation, not recomputed arclength
    int64_t minPoints = 2; // checked against current sampled/fallback geometry
    uint64_t outputByteBudget = 8 * 1024 * 1024;
    uint64_t processingByteBudget = 64 * 1024 * 1024;
    uint64_t candidateWorkBudget = 1000000;
    uint64_t sqlStepBudget = 10000000;
    // Reject a larger indivisible payload/manifest/decompression operation.
    uint64_t maxUnitBytes = 1024 * 1024;
    // Applied before selected priority and maximumObjects. nullopt is unrestricted;
    // an explicitly empty set matches nothing. UUIDs are canonicalized by the store.
    std::optional<std::vector<std::string>> uuids;
    // Exact coordinate declaration, using the same JSON filter as ObjectSelection.
    std::optional<std::string> coordinates;
    // Volume target, exclusive with coordinates: the slice, minLength and the
    // returned run points are in this volume's voxels, across every frame
    // declared in it (see ObjectSelection::volume). maximumObjects, selected
    // priority and UUID order apply across the frames.
    std::optional<std::string> volume;
    // With a volume target: voxels of that volume at this scale (a pyramid
    // level's voxels per level-0 voxel, e.g. 0.5 for level 1).
    double targetScale = 1;
    std::shared_ptr<const VolumeTransforms> transforms;
    // Internal frame traversal: source coordinates to the query coordinate frame.
    std::optional<VolumeAffine> toTarget;
};
enum class RegionStop { Exhausted, ObjectLimit, OutputBudget, ProcessingBudget, CandidateBudget, SqlBudget, Cancelled };
struct RegionStats {
    uint64_t outputBytes = 0, processingBytes = 0;
    uint64_t payloadBytes = 0, verifiedBytes = 0, metadataBytes = 0;
    uint64_t reconstructedBytes = 0, cacheBytes = 0;
    uint64_t candidateWork = 0, candidateVisits = 0, segmentTests = 0, sqlSteps = 0;
};
struct VisibleRun {
    std::string objectUuid, revisionUuid, arrayName, family;
    int64_t firstSegment = 0;
    std::vector<std::array<double, 3>> points;
};
struct RegionResult {
    std::vector<VisibleRun> runs;
    // Surfaces that reach the region by their record bounds but whose geometry
    // is not local yet (catalog placeholders), in UUID order. A catalog
    // attachment fetches them; they appear in runs once their geometry is local.
    std::vector<std::string> remote;
    RegionStats stats;
    uint64_t intersectingObjects = 0;
    RegionStop stop = RegionStop::Exhausted;
    bool complete() const { return stop == RegionStop::Exhausted; }
};
const char* regionStopName(RegionStop);
// The nearest point of every surface within tolerance of a point: one hit per
// surface, ordered by distance then UUID. Surfaces are their valid triangles,
// split along the same diagonal as surface regions. With a volume target the
// point, tolerance and results are in that volume's voxels across its frames.
struct SurfaceProximityQuery {
    std::array<double, 3> point{0, 0, 0};
    double tolerance = 1; // inclusive
    // nullopt is every surface; an explicitly empty set matches nothing.
    std::optional<std::vector<std::string>> uuids;
    std::optional<std::string> coordinates, volume; // exclusive, as in RegionOptions
    double targetScale = 1; // as in RegionOptions
    std::shared_ptr<const VolumeTransforms> transforms;
    // Internal frame traversal: source coordinates to the query coordinate frame.
    std::optional<VolumeAffine> toTarget;
};
struct SurfaceProximityHit {
    std::string objectUuid, revisionUuid;
    std::array<double, 2> grid{0, 0};      // column, row on the points grid (fractional)
    std::array<double, 3> point{0, 0, 0};  // the closest surface point
    double distance = 0;
};
// Optional cancellation predicate runs on the transaction thread, must not
// re-enter the store, and may inspect counters (also during SQL). No callback
// exception crosses SQLite's C boundary. An atomic flag works equally well.
using RegionCancel = std::function<bool(const RegionStats&)>;
}
