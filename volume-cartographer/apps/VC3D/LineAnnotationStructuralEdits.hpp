#pragma once

// Pure planning of the structural fiber edits (merge two fibers into one,
// split one into two) on stored records, free of the controller and of Qt,
// so the outcome that gets written to disk can be unit-tested against the
// loader's own link checks.
//
// The controller's job is reduced to: pick the participants, run the
// planner, redirect the peers' refs with redirectBranchRefs, run the batch
// through canonicalizeFiberSaveSnapshots + validateFiberSaveSnapshots and
// checkStructuralEditGraph, and commit it with ONE runFiberSaveJob (new
// files + redirected peers as payloads, originals as retirements).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "LineAnnotationFiberLinkValidation.hpp"
#include "LineAnnotationFiberLinks.hpp"
#include "LineAnnotationFiberSegments.hpp"
#include "LineAnnotationStoredFiber.hpp"

namespace vc3d::line_annotation {

// Set on a fiber whose line was assembled from pieces (merge join span,
// split halves without their extrapolation tails) and awaits a re-fit; the
// literal is duplicated in scripts/fiber_merge.py (REOPTIMIZE_TAG).
inline constexpr const char* kNeedsReoptimizationTag = "needs_reoptimization";

// Identity of a fiber the edit creates: allocated by the controller (runtime
// id, canonical file name from username/startedAt/sequence, primary source).
struct NewFiberIdentity {
    uint64_t id = 0;
    std::string username;
    std::string startedAt;
    uint64_t sequence = 0;
    std::string fileName;
    std::filesystem::path sourceRoot;
};

struct MergePlan {
    StoredFiber merged;
    std::size_t joinControlIndex = 0;
    bool reverseClicked = false;
    bool reverseCandidate = false;
    std::size_t clickedCount = 0;
    std::size_t farCount = 0;

    // Stored index on the clicked / far fiber -> stored index on the merged
    // fiber (the orientation remaps the merge applied).
    [[nodiscard]] int clickedRemap(int index) const;
    [[nodiscard]] int farRemap(int index) const;
};

// Plans the merge of `far` onto `clicked` at the given stored control-point
// indices (both must be endpoints). Refuses, with the user-facing reason in
// `*error`, when: an index is not an endpoint; the fibers are linked
// anywhere other than exactly between the two endpoints; that endpoint link
// is an adjacent-winding link; the coordinate domains differ; a join control
// point carries a kollesis termination; the stored control points do not
// resolve on the stored lines. The merged fiber's own link directions are
// NOT recomputed here: the batch canonicalization does that from the final
// line, for the merged fiber and for every redirected peer alike.
[[nodiscard]] std::optional<MergePlan> planFiberMerge(
    const StoredFiber& clicked,
    int clickedIndex,
    const StoredFiber& far,
    int candidateIndex,
    FiberOptimizationMode mode,
    const NewFiberIdentity& identity,
    std::string* error);

struct SplitPlanResult {
    StoredFiber prefix;
    StoredFiber suffix;
    FiberSplitPlan plan;
};

// Plans the split of `parent` after stored control point `splitAfter`.
// Both halves inherit tags, width, mode, coordinate domain and per-span
// metadata; branch refs are partitioned by control-point range; with
// `linkHalves` the boundary gets a pending reciprocal link.
[[nodiscard]] std::optional<SplitPlanResult> planFiberSplit(
    const StoredFiber& parent,
    std::size_t splitAfter,
    const NewFiberIdentity& prefixIdentity,
    const NewFiberIdentity& suffixIdentity,
    bool linkHalves,
    std::string* error);

// Where a ref that pointed at an original now points.
struct RedirectTarget {
    uint64_t id = 0;
    std::string fileName;
    int controlPointIndex = -1;
};

// One original consumed by the edit: how refs to it are redirected. `map`
// takes the original's STORED control-point index and returns the new
// target, or nullopt when the index has no counterpart (the ref is left
// alone and reported through `error` by redirectBranchRefs).
struct BranchRedirectSource {
    uint64_t id = 0;
    std::string fileName;
    // The original's stored record: refs coming from a live session may hold
    // the original's SESSION index (syncBranchEndpointPositions stores the
    // partner pane's index), so the target endpoint is re-resolved on this
    // record by position before `map` is applied.
    const StoredFiber* stored = nullptr;
    std::function<std::optional<RedirectTarget>(int storedIndex)> map;
};

// Rewrites, in place, every ref in `refs` that points at one of `sources`
// (by the shared identity rule) onto its new target: only the target-side
// fields change (branchFiberId, branchFileName, branchControlPointIndex);
// the owner's own index and both positions are untouched (the target
// position is the same control point, moved to another file). Returns the
// number of refs redirected; a ref that cannot be resolved on the original's
// stored control points, or that maps to nothing, sets `*error` and returns
// nullopt without modifying anything.
[[nodiscard]] std::optional<std::size_t> redirectBranchRefs(
    std::vector<FiberBranchRef>& refs,
    const std::vector<BranchRedirectSource>& sources,
    std::string* error);

struct StructuralGraphCheck {
    // Issues that involve the edit: the owning fiber is in the batch, or the
    // entry's target is in the batch or is a retired original. Any of these
    // must abort the edit before the disk commit.
    std::vector<BranchLinkValidationIssue> blocking;
    // Issues between untouched records: pre-existing defects the loader
    // prompt owns; logged, never blocking.
    std::vector<BranchLinkValidationIssue> unrelated;
};

// Runs the loader's own issue collector over the graph the next load will
// see (untouched records + batch records, originals removed) and splits the
// result. `batchKeys` / `retiredKeys` are the source-qualified keys the
// collector's sourceKey produces for the batch files and the retired
// originals; `sourceKey(fiber, fileName)` must be the same callback the
// collector uses.
template<class Collect, class SourceKey>
StructuralGraphCheck checkStructuralEditGraph(const std::vector<StoredFiber>& graph,
                                              const std::unordered_set<std::string>& batchKeys,
                                              const std::unordered_set<std::string>& retiredKeys,
                                              Collect collect,
                                              SourceKey sourceKey)
{
    StructuralGraphCheck check;
    for (const auto& issue : collect(graph)) {
        if (issue.fiberIndex >= graph.size()) {
            continue;
        }
        const StoredFiber& owner = graph[issue.fiberIndex];
        bool blocking = batchKeys.count(sourceKey(owner, owner.fileName)) != 0;
        if (!blocking && issue.branchIndex < owner.branches.size()) {
            const std::string targetKey =
                sourceKey(owner, owner.branches[issue.branchIndex].branchFileName);
            blocking = batchKeys.count(targetKey) != 0 || retiredKeys.count(targetKey) != 0;
        }
        (blocking ? check.blocking : check.unrelated).push_back(issue);
    }
    return check;
}

// Exact geometry fingerprint the loader's source dedupe uses to recognise
// the same fiber stored under two names (raw double bits, format-independent).
// A structural edit must not produce a fiber with the fingerprint of a loaded
// one: dedupe runs before link validation on load and would hide the new
// fiber together with its links.
[[nodiscard]] std::string fiberGeometryKey(const std::vector<StoredControlPoint>& controls,
                                           const std::vector<cv::Vec3d>& linePoints);

}  // namespace vc3d::line_annotation
