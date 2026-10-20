#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace vc::objects {
struct FiberSlice {
    std::array<double, 3> origin{0, 0, 0};
    std::array<double, 3> u{1, 0, 0}, v{0, 1, 0}, normal{0, 0, 1};
    std::array<double, 6> localBounds{-1, -1, -1, 1, 1, 1};
};
struct FiberRegionOptions {
    uint64_t maximumFibers = 0; // zero means unlimited; selected is then ignored
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
};
enum class FiberRegionStop { Exhausted, FiberLimit, OutputBudget, ProcessingBudget, CandidateBudget, SqlBudget, Cancelled };
struct FiberRegionStats {
    uint64_t outputBytes = 0, processingBytes = 0;
    uint64_t payloadBytes = 0, verifiedBytes = 0, metadataBytes = 0;
    uint64_t reconstructedBytes = 0, cacheBytes = 0;
    uint64_t candidateWork = 0, candidateVisits = 0, treeVisits = 0, segmentTests = 0, sqlSteps = 0;
};
struct FiberVisibleRun {
    std::string objectUuid, revisionUuid, arrayName, family;
    int64_t firstSegment = 0;
    std::vector<std::array<double, 3>> points;
};
struct FiberRegionResult {
    std::vector<FiberVisibleRun> runs;
    FiberRegionStats stats;
    uint64_t intersectingFibers = 0;
    FiberRegionStop stop = FiberRegionStop::Exhausted;
    bool complete() const { return stop == FiberRegionStop::Exhausted; }
};
const char* fiberRegionStopName(FiberRegionStop);
// Optional cancellation predicate runs on the transaction thread, must not
// re-enter the store, and may inspect counters (also during SQL). No callback
// exception crosses SQLite's C boundary. An atomic flag works equally well.
using FiberRegionCancel = std::function<bool(const FiberRegionStats&)>;
}
