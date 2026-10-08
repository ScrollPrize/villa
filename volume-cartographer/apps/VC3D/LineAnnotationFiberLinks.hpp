#pragma once

// Branch-link predicates and the save-time link canonicalization/validation,
// moved out of LineAnnotationController.cpp so the structural-edit planners
// and the loader's link validation can share ONE definition of "these two
// link entries agree" and be unit-tested without the controller.
//
// Numeric contract (mirrored by scripts/fiber_merge.py's loader port; do not
// change without changing both): positions agree within 1e-6, directions
// agree when |dot| is within 1e-5 of 1, the tangent at a control point is the
// forward difference at its nearest line point.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core/types.hpp>

#include "FiberSliceGeometry.hpp"
#include "LineAnnotationFiberDeletion.hpp"
#include "LineAnnotationFiberSegments.hpp"
#include "LineAnnotationStoredFiber.hpp"

namespace vc3d::line_annotation {

inline constexpr double kLinkEpsilon = 1.0e-12;

inline bool finitePoint(const cv::Vec3d& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

inline bool finiteDirection(const cv::Vec3d& v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]) &&
           std::sqrt(v.dot(v)) > kLinkEpsilon;
}

inline cv::Vec3d normalizedOrZero(const cv::Vec3d& v)
{
    if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) {
        return {0.0, 0.0, 0.0};
    }
    const double n = std::sqrt(v.dot(v));
    if (n <= kLinkEpsilon) {
        return {0.0, 0.0, 0.0};
    }
    return v * (1.0 / n);
}

inline bool pointsApproximatelyEqual(const cv::Vec3d& a,
                                     const cv::Vec3d& b,
                                     double tolerance = 1.0e-6)
{
    if (!finitePoint(a) || !finitePoint(b)) {
        return false;
    }
    const cv::Vec3d delta = a - b;
    return delta.dot(delta) <= tolerance * tolerance;
}

inline std::optional<int> storedControlPointIndexByPosition(
    const std::vector<StoredControlPoint>& controlPoints,
    const cv::Vec3d& point)
{
    if (!finitePoint(point)) {
        return std::nullopt;
    }
    for (std::size_t i = 0; i < controlPoints.size(); ++i) {
        if (pointsApproximatelyEqual(controlPoints[i], point)) {
            return static_cast<int>(i);
        }
    }
    return std::nullopt;
}

inline std::optional<int> matchingStoredControlPointIndex(
    const std::vector<StoredControlPoint>& controlPoints,
    int fallbackIndex,
    const cv::Vec3d& point)
{
    if (auto index = storedControlPointIndexByPosition(controlPoints, point)) {
        return index;
    }
    if (fallbackIndex < 0 ||
        static_cast<std::size_t>(fallbackIndex) >= controlPoints.size() ||
        !pointsApproximatelyEqual(controlPoints[static_cast<std::size_t>(fallbackIndex)],
                                  point)) {
        return std::nullopt;
    }
    return fallbackIndex;
}

inline std::string fiberErrorName(const std::string& fileName)
{
    const std::string baseName = std::filesystem::path(fileName).filename().string();
    return baseName.empty() ? std::string{"<unknown>"} : baseName;
}

inline bool branchDirectionsCompatible(const cv::Vec3d& a,
                                       const cv::Vec3d& b,
                                       double tolerance = 1.0e-5)
{
    const cv::Vec3d na = normalizedOrZero(a);
    const cv::Vec3d nb = normalizedOrZero(b);
    if (!finiteDirection(na) || !finiteDirection(nb)) {
        return false;
    }
    return std::abs(std::abs(na.dot(nb)) - 1.0) <= tolerance;
}

// Runtime ids are unique across every registered source and stable across
// reloads (FiberRuntimeIds). Once both sides have one, do not let an equal
// filename in another source create a false match. Filename matching remains
// the legacy/on-load fallback. The rule is sameFiberIdentity, shared with the
// deletion helpers and the branch synchronizers.
inline bool branchReferencesFiber(const FiberBranchRef& branch,
                                  uint64_t fiberId,
                                  const std::string& fileName)
{
    return sameFiberIdentity(branch.branchFiberId, branch.branchFileName, fiberId, fileName);
}

// Forward difference at the line position; the loader, the save-time
// canonicalizer and link creation all derive link directions through this.
inline cv::Vec3d tangentAtLinePosition(const std::vector<cv::Vec3d>& points,
                                       double linePosition)
{
    if (points.size() < 2 || !std::isfinite(linePosition)) {
        return {1.0, 0.0, 0.0};
    }
    linePosition = std::clamp(linePosition, 0.0, static_cast<double>(points.size() - 1));
    int lower = static_cast<int>(std::floor(linePosition));
    int upper = std::min<int>(lower + 1, static_cast<int>(points.size()) - 1);
    if (lower == upper && lower > 0) {
        --lower;
    }
    cv::Vec3d tangent = points[static_cast<std::size_t>(upper)] -
                        points[static_cast<std::size_t>(lower)];
    tangent = normalizedOrZero(tangent);
    return finiteDirection(tangent) ? tangent : cv::Vec3d{1.0, 0.0, 0.0};
}

inline cv::Vec3d endpointTangentFromLinePoints(const std::vector<cv::Vec3d>& linePoints,
                                               const cv::Vec3d& controlPoint,
                                               const cv::Vec3d& fallback = {0.0, 0.0, 0.0})
{
    if (linePoints.size() >= 2 && finitePoint(controlPoint)) {
        const std::size_t index = vc3d::fiber_slice::nearestLinePointIndex(linePoints, controlPoint);
        const cv::Vec3d tangent = tangentAtLinePosition(linePoints, static_cast<double>(index));
        if (finiteDirection(tangent)) {
            return normalizedOrZero(tangent);
        }
    }
    return finiteDirection(fallback) ? normalizedOrZero(fallback) : cv::Vec3d{0.0, 0.0, 0.0};
}

// Save-time canonicalization of a batch of snapshots (any type with a
// `StoredFiber fiber` member): local link endpoints are re-resolved on the
// fiber's own control points and their directions recomputed from its line;
// for targets INSIDE the batch the linked endpoint is re-resolved and its
// direction recomputed from the target's line, and the reciprocal entry is
// rewritten from the same record. Targets outside the batch keep their
// stored linked-side fields.
template<class Snapshot>
void canonicalizeFiberSaveSnapshots(std::vector<Snapshot>& snapshots)
{
    auto findSnapshotForBranch = [&snapshots](const FiberBranchRef& branch) -> Snapshot* {
        for (auto& snapshot : snapshots) {
            if (branchReferencesFiber(branch, snapshot.fiber.id, snapshot.fiber.fileName)) {
                return &snapshot;
            }
        }
        return nullptr;
    };

    for (auto& snapshot : snapshots) {
        for (auto& branch : snapshot.fiber.branches) {
            if (auto localIndex = matchingStoredControlPointIndex(
                    snapshot.fiber.controlPoints,
                    branch.controlPointIndex,
                    branch.controlPointPosition)) {
                branch.controlPointIndex = *localIndex;
                branch.controlPointPosition =
                    snapshot.fiber.controlPoints[static_cast<std::size_t>(*localIndex)];
            }
            branch.controlPointDirection =
                endpointTangentFromLinePoints(snapshot.fiber.linePoints,
                                              branch.controlPointPosition,
                                              branch.controlPointDirection);
            if (finiteDirection(branch.branchControlPointDirection)) {
                branch.branchControlPointDirection =
                    normalizedOrZero(branch.branchControlPointDirection);
            }

            auto* target = findSnapshotForBranch(branch);
            if (!target) {
                continue;
            }
            branch.branchFiberId = target->fiber.id;
            branch.branchFileName = target->fiber.fileName;
            if (auto targetIndex = matchingStoredControlPointIndex(
                    target->fiber.controlPoints,
                    branch.branchControlPointIndex,
                    branch.branchControlPointPosition)) {
                branch.branchControlPointIndex = *targetIndex;
                branch.branchControlPointPosition =
                    target->fiber.controlPoints[static_cast<std::size_t>(*targetIndex)];
            }
            branch.branchControlPointDirection =
                endpointTangentFromLinePoints(target->fiber.linePoints,
                                              branch.branchControlPointPosition,
                                              branch.branchControlPointDirection);
        }
    }

    for (auto& snapshot : snapshots) {
        for (auto& branch : snapshot.fiber.branches) {
            if (branch.controlPointIndex < 0 ||
                branch.branchControlPointIndex < 0 ||
                static_cast<std::size_t>(branch.controlPointIndex) >=
                    snapshot.fiber.controlPoints.size()) {
                continue;
            }
            auto* target = findSnapshotForBranch(branch);
            if (!target ||
                static_cast<std::size_t>(branch.branchControlPointIndex) >=
                    target->fiber.controlPoints.size()) {
                continue;
            }
            auto reciprocal = std::find_if(
                target->fiber.branches.begin(),
                target->fiber.branches.end(),
                [&snapshot, &branch](const FiberBranchRef& candidate) {
                    return candidate.adjacent == branch.adjacent &&
                           branchReferencesFiber(candidate,
                                                 snapshot.fiber.id,
                                                 snapshot.fiber.fileName) &&
                           (candidate.controlPointIndex ==
                                branch.branchControlPointIndex ||
                            pointsApproximatelyEqual(candidate.controlPointPosition,
                                                     branch.branchControlPointPosition)) &&
                           (candidate.branchControlPointIndex ==
                                branch.controlPointIndex ||
                            pointsApproximatelyEqual(candidate.branchControlPointPosition,
                                                     branch.controlPointPosition));
                });
            if (reciprocal == target->fiber.branches.end()) {
                continue;
            }

            branch.branchFiberId = target->fiber.id;
            branch.branchFileName = target->fiber.fileName;
            branch.controlPointPosition =
                snapshot.fiber.controlPoints[static_cast<std::size_t>(branch.controlPointIndex)];
            branch.branchControlPointPosition =
                target->fiber.controlPoints[static_cast<std::size_t>(branch.branchControlPointIndex)];

            reciprocal->controlPointIndex = branch.branchControlPointIndex;
            reciprocal->branchFiberId = snapshot.fiber.id;
            reciprocal->branchControlPointIndex = branch.controlPointIndex;
            reciprocal->branchFileName = snapshot.fiber.fileName;
            reciprocal->controlPointPosition = branch.branchControlPointPosition;
            reciprocal->branchControlPointPosition = branch.controlPointPosition;
            reciprocal->controlPointDirection = branch.branchControlPointDirection;
            reciprocal->branchControlPointDirection = branch.controlPointDirection;
            reciprocal->pending = branch.pending;
        }
    }
}

// Throws std::runtime_error (message prefixed with the offending file name)
// when a snapshot could not be loaded back: control points must be an exact
// ordered subset of line_points, and links whose target is in the batch must
// resolve on both sides and have a reciprocal entry that agrees on indices,
// positions and directions.
template<class Snapshot>
void validateFiberSaveSnapshots(const std::vector<Snapshot>& snapshots)
{
    // Geometry guard for every snapshot, linked or not: the v3 loader (and
    // split/merge planning) requires control points to be an exact ordered
    // subset of line_points. A violation here means a session was serialized
    // before its geometry was finalized by a solve; writing it would produce
    // a fiber that fails to load. Refuse the save instead.
    for (const auto& snapshot : snapshots) {
        if (snapshot.fiber.controlPoints.empty()) {
            continue;
        }
        if (snapshot.fiber.linePoints.empty()) {
            // Control points with no line at all cannot satisfy the subset
            // contract either; the loader would reject the file.
            throw std::runtime_error(
                fiberErrorName(snapshot.fiber.fileName) +
                ": control points present but line_points is empty; the "
                "fiber was not finalized before saving");
        }
        if (!orderedControlPointLineIndices(
                storedControlPointPositions(snapshot.fiber.controlPoints),
                snapshot.fiber.linePoints)) {
            throw std::runtime_error(
                fiberErrorName(snapshot.fiber.fileName) +
                ": control points are not an ordered exact subset of "
                "line_points; the fiber was not finalized before saving");
        }
    }

    if (snapshots.size() < 2) {
        return;
    }

    auto findSnapshotForBranch = [&snapshots](const FiberBranchRef& branch) -> const Snapshot* {
        for (const auto& snapshot : snapshots) {
            if (branchReferencesFiber(branch, snapshot.fiber.id, snapshot.fiber.fileName)) {
                return &snapshot;
            }
        }
        return nullptr;
    };
    auto fail = [](const Snapshot& snapshot, const std::string& reason) {
        throw std::runtime_error(fiberErrorName(snapshot.fiber.fileName) + ": " + reason);
    };

    for (const auto& snapshot : snapshots) {
        for (const auto& branch : snapshot.fiber.branches) {
            const auto localIndex = matchingStoredControlPointIndex(
                snapshot.fiber.controlPoints,
                branch.controlPointIndex,
                branch.controlPointPosition);
            if (!localIndex) {
                fail(snapshot, "local CP position mismatch");
            }
            if (branch.branchFileName.empty()) {
                fail(snapshot, "missing branch_file");
            }

            const Snapshot* target = findSnapshotForBranch(branch);
            if (!target) {
                continue;
            }
            const auto targetIndex = matchingStoredControlPointIndex(
                target->fiber.controlPoints,
                branch.branchControlPointIndex,
                branch.branchControlPointPosition);
            if (!targetIndex) {
                fail(snapshot, "linked CP position mismatch");
            }

            const auto reciprocal = std::find_if(
                target->fiber.branches.begin(),
                target->fiber.branches.end(),
                [&snapshot, &branch](const FiberBranchRef& candidate) {
                    return candidate.adjacent == branch.adjacent &&
                           branchReferencesFiber(candidate,
                                                 snapshot.fiber.id,
                                                 snapshot.fiber.fileName) &&
                           candidate.controlPointIndex == branch.branchControlPointIndex &&
                           candidate.branchControlPointIndex == branch.controlPointIndex &&
                           pointsApproximatelyEqual(candidate.controlPointPosition,
                                                    branch.branchControlPointPosition) &&
                           pointsApproximatelyEqual(candidate.branchControlPointPosition,
                                                    branch.controlPointPosition) &&
                           branchDirectionsCompatible(candidate.controlPointDirection,
                                                      branch.branchControlPointDirection) &&
                           branchDirectionsCompatible(candidate.branchControlPointDirection,
                                                      branch.controlPointDirection);
                });
            if (reciprocal == target->fiber.branches.end()) {
                fail(snapshot, "missing reciprocal branch");
            }
        }
    }
}

}  // namespace vc3d::line_annotation
