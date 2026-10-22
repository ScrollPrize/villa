#pragma once

// Strict structural link validation. No load-time repair or neutralization.

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

} // namespace vc3d::line_annotation
