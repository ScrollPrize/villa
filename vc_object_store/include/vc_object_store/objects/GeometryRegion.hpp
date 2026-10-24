#pragma once
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
    RegionStats stats;
    uint64_t intersectingObjects = 0;
    RegionStop stop = RegionStop::Exhausted;
    bool complete() const { return stop == RegionStop::Exhausted; }
};
const char* regionStopName(RegionStop);
// Optional cancellation predicate runs on the transaction thread, must not
// re-enter the store, and may inspect counters (also during SQL). No callback
// exception crosses SQLite's C boundary. An atomic flag works equally well.
using RegionCancel = std::function<bool(const RegionStats&)>;
}
