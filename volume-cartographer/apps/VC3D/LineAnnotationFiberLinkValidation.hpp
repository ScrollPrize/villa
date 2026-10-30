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

template<class Fiber, class Reciprocal>
std::vector<BranchLinkValidationIssue> collectFiberBranchIssues(
    const std::vector<Fiber>& fibers, Reciprocal reciprocal)
{
    std::vector<BranchLinkValidationIssue> issues;
    std::unordered_map<uint64_t, std::size_t> indexById;
    for (std::size_t i = 0; i < fibers.size(); ++i) indexById.emplace(fibers[i].id, i);

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
            const auto targetIndex = indexById.find(branch.branchFiberId);
            if (targetIndex == indexById.end()) {
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
