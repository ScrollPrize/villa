#pragma once

// Load-time validation of fiber branch links, free of the controller so it
// can be unit-tested: the issue collector (what the "Broken branch links"
// prompt lists) and the fixed-point neutralizer that drops offending ENTRIES
// until the set is consistent. Neither removes a fiber: a link problem is a
// property of two entries, and removing a fiber only manufactures "missing
// linked fiber" on its peers (the cascade that erased 451 of 473 PHerc0139
// fibers on 2026-09-30).
//
// Fiber is any type with `fileName`, `sourceRoot`, `controlPoints`,
// `linePoints` and `branches` (a vector of FiberBranchRef) members.

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "LineAnnotationFiberLinks.hpp"

namespace vc3d::line_annotation {

struct BranchLinkValidationIssue {
    std::size_t fiberIndex = 0;
    std::size_t branchIndex = 0;
    std::string reason;
};

// `sourceKey(fiber, fileName)` keys a fiber file within its source root,
// `linkKey(fiber, branchFileName)` resolves a branch target to such a key
// (the controller routes dedupe aliases through it), `reciprocal(fiber,
// branch, candidate)` is the reciprocity predicate (reciprocalBranchRefMatches
// with the shared position/direction tolerances).
template<class Fiber, class SourceKey, class LinkKey, class Reciprocal>
std::vector<BranchLinkValidationIssue> collectFiberBranchIssues(
    const std::vector<Fiber>& fibers,
    SourceKey sourceKey,
    LinkKey linkKey,
    Reciprocal reciprocal)
{
    std::vector<BranchLinkValidationIssue> issues;
    std::unordered_map<std::string, std::size_t> indexByFileName;
    indexByFileName.reserve(fibers.size());
    for (std::size_t i = 0; i < fibers.size(); ++i) {
        if (!fibers[i].fileName.empty()) {
            indexByFileName[sourceKey(fibers[i], fibers[i].fileName)] = i;
        }
    }

    for (std::size_t fiberIndex = 0; fiberIndex < fibers.size(); ++fiberIndex) {
        const Fiber& fiber = fibers[fiberIndex];
        for (std::size_t branchIndex = 0; branchIndex < fiber.branches.size(); ++branchIndex) {
            const FiberBranchRef& branch = fiber.branches[branchIndex];
            auto addIssue = [&](const std::string& reason) {
                issues.push_back({fiberIndex, branchIndex, reason});
            };

            if (branch.controlPointIndex < 0 ||
                static_cast<std::size_t>(branch.controlPointIndex) >= fiber.controlPoints.size()) {
                addIssue("local CP index out of range");
                continue;
            }
            if (!pointsApproximatelyEqual(
                    fiber.controlPoints[static_cast<std::size_t>(branch.controlPointIndex)],
                    branch.controlPointPosition)) {
                addIssue("local CP position mismatch");
                continue;
            }
            if (!finiteDirection(branch.controlPointDirection) ||
                !finiteDirection(branch.branchControlPointDirection)) {
                addIssue("invalid branch directions");
                continue;
            }
            if (fiber.linePoints.size() >= 2) {
                const cv::Vec3d expectedLocal =
                    endpointTangentFromLinePoints(fiber.linePoints,
                                                  branch.controlPointPosition);
                if (!branchDirectionsCompatible(branch.controlPointDirection,
                                                expectedLocal)) {
                    addIssue("branch endpoint direction mismatch");
                    continue;
                }
            }
            if (branch.branchFileName.empty()) {
                addIssue("missing branch_file");
                continue;
            }
            const auto targetIndex = indexByFileName.find(
                linkKey(fiber, branch.branchFileName));
            if (targetIndex == indexByFileName.end()) {
                addIssue("missing linked fiber");
                continue;
            }
            const Fiber& target = fibers[targetIndex->second];
            if (branch.branchControlPointIndex < 0 ||
                static_cast<std::size_t>(branch.branchControlPointIndex) >=
                    target.controlPoints.size()) {
                addIssue("linked CP index out of range");
                continue;
            }
            if (!pointsApproximatelyEqual(
                    target.controlPoints[static_cast<std::size_t>(branch.branchControlPointIndex)],
                    branch.branchControlPointPosition)) {
                addIssue("linked CP position mismatch");
                continue;
            }
            if (target.linePoints.size() >= 2) {
                const cv::Vec3d expectedLinked =
                    endpointTangentFromLinePoints(target.linePoints,
                                                  branch.branchControlPointPosition);
                if (!branchDirectionsCompatible(branch.branchControlPointDirection,
                                                expectedLinked)) {
                    addIssue("branch endpoint direction mismatch");
                    continue;
                }
            }

            const auto found = std::find_if(
                target.branches.begin(),
                target.branches.end(),
                [&fiber, &branch, &reciprocal](const FiberBranchRef& candidate) {
                    return reciprocal(fiber, branch, candidate);
                });
            if (found == target.branches.end()) {
                addIssue("missing reciprocal branch");
            }
        }
    }
    return issues;
}

struct BranchNeutralizationResult {
    std::size_t removedEntries = 0;
    // Indices into the fiber vector whose `branches` lost at least one entry.
    std::vector<std::size_t> changedFibers;
};

// Drops every entry named in `issues`, then re-collects with `collect(fibers)`
// and repeats until the set reports nothing or nothing more can be removed
// (an entry whose partner was just dropped is dropped on the next pass, so no
// one-way link survives). Fibers are never removed. Pure on `fibers` apart
// from the entry removal: persistence flags are the caller's business.
template<class Fiber, class Collect>
BranchNeutralizationResult neutralizeFiberBranchIssues(
    std::vector<Fiber>& fibers,
    std::vector<BranchLinkValidationIssue> issues,
    Collect collect)
{
    BranchNeutralizationResult result;
    std::unordered_set<std::size_t> changed;
    auto removeIssues = [&](const std::vector<BranchLinkValidationIssue>& toRemove) {
        std::unordered_map<std::size_t, std::vector<std::size_t>> branchIndicesByFiber;
        for (const auto& issue : toRemove) {
            if (issue.fiberIndex >= fibers.size() ||
                issue.branchIndex >= fibers[issue.fiberIndex].branches.size()) {
                continue;
            }
            branchIndicesByFiber[issue.fiberIndex].push_back(issue.branchIndex);
        }
        bool removedAny = false;
        for (auto& [fiberIndex, branchIndices] : branchIndicesByFiber) {
            auto& fiber = fibers[fiberIndex];
            std::sort(branchIndices.begin(), branchIndices.end());
            branchIndices.erase(std::unique(branchIndices.begin(), branchIndices.end()),
                                branchIndices.end());
            for (auto it = branchIndices.rbegin(); it != branchIndices.rend(); ++it) {
                if (*it >= fiber.branches.size()) {
                    continue;
                }
                fiber.branches.erase(fiber.branches.begin() +
                                     static_cast<std::ptrdiff_t>(*it));
                ++result.removedEntries;
                changed.insert(fiberIndex);
                removedAny = true;
            }
        }
        return removedAny;
    };

    (void)removeIssues(issues);
    for (;;) {
        const auto remaining = collect(fibers);
        if (remaining.empty() || !removeIssues(remaining)) {
            break;
        }
    }
    result.changedFibers.assign(changed.begin(), changed.end());
    std::sort(result.changedFibers.begin(), result.changedFibers.end());
    return result;
}

}  // namespace vc3d::line_annotation
