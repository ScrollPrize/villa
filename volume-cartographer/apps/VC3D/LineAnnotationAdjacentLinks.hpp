#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vc3d::line_annotation {

template<class Fiber, class Branch, class PointsEqual, class DirectionsCompatible>
bool reciprocalBranchRefMatches(const Fiber& fiber, const Branch& branch,
                                const Branch& candidate, PointsEqual pointsEqual,
                                DirectionsCompatible directionsCompatible)
{
    return candidate.adjacent == branch.adjacent &&
           candidate.branchFileName == fiber.fileName &&
           candidate.controlPointIndex == branch.branchControlPointIndex &&
           candidate.branchControlPointIndex == branch.controlPointIndex &&
           pointsEqual(candidate.controlPointPosition, branch.branchControlPointPosition) &&
           pointsEqual(candidate.branchControlPointPosition, branch.controlPointPosition) &&
           directionsCompatible(candidate.controlPointDirection,
                                branch.branchControlPointDirection) &&
           directionsCompatible(candidate.branchControlPointDirection,
                                branch.controlPointDirection);
}

// Stage the missing reciprocals before appending them: a self-link must not
// invalidate the vector being traversed, and several peers may all restore
// links into the same legacy file. Presence means presence at load time.
template<class Fiber, class LinkKey, class ReciprocalMatches>
void restoreMissingAdjacentBranchRefs(std::vector<Fiber>& fibers,
                                     LinkKey linkKey,
                                     ReciprocalMatches reciprocalMatches)
{
    using Branch = typename decltype(Fiber::branches)::value_type;
    std::unordered_map<std::string, std::size_t> indexByFileName;
    for (std::size_t i = 0; i < fibers.size(); ++i) {
        if (!fibers[i].fileName.empty()) {
            indexByFileName[(fibers[i].sourceRoot / fibers[i].fileName)
                               .lexically_normal().string()] = i;
        }
    }
    std::vector<std::vector<Branch>> restored(fibers.size());
    for (const auto& fiber : fibers) {
        for (const auto& branch : fiber.branches) {
            if (!branch.adjacent || branch.branchFileName.empty()) {
                continue;
            }
            const auto target = indexByFileName.find(linkKey(fiber, branch.branchFileName));
            if (target == indexByFileName.end()) {
                continue;
            }
            const auto& other = fibers[target->second];
            if (other.adjacentBranchesPresent) {
                continue;
            }
            auto matches = [&](const Branch& candidate) {
                return reciprocalMatches(fiber, branch, candidate);
            };
            auto& additions = restored[target->second];
            if (std::any_of(other.branches.begin(), other.branches.end(), matches) ||
                std::any_of(additions.begin(), additions.end(), matches)) {
                continue;
            }
            Branch reciprocal = branch;
            reciprocal.branchFiberId = fiber.id;
            reciprocal.branchFileName = fiber.fileName;
            std::swap(reciprocal.controlPointIndex, reciprocal.branchControlPointIndex);
            std::swap(reciprocal.controlPointPosition, reciprocal.branchControlPointPosition);
            std::swap(reciprocal.controlPointDirection, reciprocal.branchControlPointDirection);
            additions.push_back(std::move(reciprocal));
        }
    }
    for (std::size_t i = 0; i < fibers.size(); ++i) {
        if (!restored[i].empty()) {
            auto& fiber = fibers[i];
            fiber.branches.insert(fiber.branches.end(), restored[i].begin(), restored[i].end());
            fiber.needsSave = true;
            fiber.adjacentHealed = true;
        }
    }
}

} // namespace vc3d::line_annotation
