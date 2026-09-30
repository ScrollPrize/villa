#include "LineAnnotationStructuralEdits.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace vc3d::line_annotation {

namespace {

void addUniqueSorted(std::vector<std::string>& values, const std::string& value)
{
    if (value.empty()) {
        return;
    }
    if (std::find(values.begin(), values.end(), value) != values.end()) {
        return;
    }
    values.push_back(value);
    std::sort(values.begin(), values.end());
}

bool fail(std::string* error, const char* message)
{
    if (error) {
        *error = message;
    }
    return false;
}

void applyIdentity(StoredFiber& fiber, const NewFiberIdentity& identity)
{
    fiber.id = identity.id;
    fiber.username = identity.username;
    fiber.startedAt = identity.startedAt;
    fiber.sequence = identity.sequence;
    fiber.fileName = identity.fileName;
    fiber.sourceRoot = identity.sourceRoot;
    fiber.generation = 1;
}

}  // namespace

int MergePlan::clickedRemap(int index) const
{
    return reverseClicked ? static_cast<int>(clickedCount) - 1 - index : index;
}

int MergePlan::farRemap(int index) const
{
    const int within = reverseCandidate ? static_cast<int>(farCount) - 1 - index : index;
    return static_cast<int>(clickedCount) + within;
}

std::optional<MergePlan> planFiberMerge(const StoredFiber& clicked,
                                        int clickedIndex,
                                        const StoredFiber& far,
                                        int candidateIndex,
                                        FiberOptimizationMode mode,
                                        const NewFiberIdentity& identity,
                                        std::string* error)
{
    const std::size_t clickedCount = clicked.controlPoints.size();
    const std::size_t farCount = far.controlPoints.size();
    if (clickedIndex < 0 || static_cast<std::size_t>(clickedIndex) >= clickedCount ||
        candidateIndex < 0 || static_cast<std::size_t>(candidateIndex) >= farCount) {
        fail(error, "Could not resolve the merge control points.");
        return std::nullopt;
    }
    const bool clickedIsEndpoint =
        clickedIndex == 0 || static_cast<std::size_t>(clickedIndex) + 1 == clickedCount;
    const bool candidateIsEndpoint =
        candidateIndex == 0 || static_cast<std::size_t>(candidateIndex) + 1 == farCount;
    if (!clickedIsEndpoint || !candidateIsEndpoint) {
        fail(error, "Merge joins fiber endpoints; pick the first or last control point on both fibers.");
        return std::nullopt;
    }

    // The only consumable mutual link is one between exactly the two merge
    // endpoints; anything else connecting these fibers blocks the merge, and
    // an adjacent-winding link between the endpoints cannot become one fiber.
    const auto isPairLink = [&](const FiberBranchRef& branch, uint64_t otherId,
                                const std::string& otherFile, int localIdx, int otherIdx) {
        return branchReferencesFiber(branch, otherId, otherFile) &&
               branch.controlPointIndex == localIdx &&
               branch.branchControlPointIndex == otherIdx;
    };
    for (const auto& branch : clicked.branches) {
        if (!branchReferencesFiber(branch, far.id, far.fileName)) {
            continue;
        }
        if (!isPairLink(branch, far.id, far.fileName, clickedIndex, candidateIndex)) {
            fail(error, "These fibers are linked elsewhere; unlink first.");
            return std::nullopt;
        }
        if (branch.adjacent) {
            fail(error, "Cannot merge: the endpoints are linked as adjacent windings; "
                        "an adjacent link cannot become one fiber.");
            return std::nullopt;
        }
    }
    for (const auto& branch : far.branches) {
        if (!branchReferencesFiber(branch, clicked.id, clicked.fileName)) {
            continue;
        }
        if (!isPairLink(branch, clicked.id, clicked.fileName, candidateIndex, clickedIndex)) {
            fail(error, "These fibers are linked elsewhere; unlink first.");
            return std::nullopt;
        }
        if (branch.adjacent) {
            fail(error, "Cannot merge: the endpoints are linked as adjacent windings; "
                        "an adjacent link cannot become one fiber.");
            return std::nullopt;
        }
    }
    if (clicked.coordinateBaseShapeZYX != far.coordinateBaseShapeZYX) {
        fail(error, "Cannot merge: the fibers are stored in different coordinate domains.");
        return std::nullopt;
    }

    MergePlan plan;
    plan.clickedCount = clickedCount;
    plan.farCount = farCount;
    // Orient both sides so the clicked fiber ends at the join and the
    // candidate fiber starts at it; the merged line reads clicked-first.
    plan.reverseClicked = clickedIndex == 0;
    plan.reverseCandidate = static_cast<std::size_t>(candidateIndex) + 1 == farCount;
    std::vector<StoredControlPoint> clickedControls =
        plan.reverseClicked ? reversedStoredControlPoints(clicked.controlPoints)
                            : clicked.controlPoints;
    std::vector<cv::Vec3d> clickedLine = clicked.linePoints;
    if (plan.reverseClicked) {
        std::reverse(clickedLine.begin(), clickedLine.end());
    }
    std::vector<StoredControlPoint> farControls =
        plan.reverseCandidate ? reversedStoredControlPoints(far.controlPoints)
                              : far.controlPoints;
    std::vector<cv::Vec3d> farLine = far.linePoints;
    if (plan.reverseCandidate) {
        std::reverse(farLine.begin(), farLine.end());
    }
    // The join makes both join-side ends interior, and a kollesis
    // termination is a claim that the fiber ends there. Refuse rather than
    // silently drop the claim; the user untags first if the merge is right.
    if ((!clickedControls.empty() &&
         hasControlPointTag(clickedControls.back().tags, kKollesisTerminationTag)) ||
        (!farControls.empty() &&
         hasControlPointTag(farControls.front().tags, kKollesisTerminationTag))) {
        fail(error, "Cannot merge: a join control point is tagged as a kollesis "
                    "termination. Remove the tag first.");
        return std::nullopt;
    }
    auto geometry = computeFiberMergeGeometry(clickedControls, clickedLine, farControls, farLine);
    if (!geometry) {
        fail(error, "Cannot merge: the stored control points do not resolve on the stored lines.");
        return std::nullopt;
    }

    StoredFiber& merged = plan.merged;
    applyIdentity(merged, identity);
    // A merged fiber has one width: prefer the clicked fiber's annotation.
    merged.width = clicked.width > 0 ? clicked.width : far.width;
    merged.widthGapFraction = clicked.width > 0 ? clicked.widthGapFraction : far.widthGapFraction;
    merged.controlPoints = std::move(geometry->controlPoints);
    merged.linePoints = std::move(geometry->linePoints);
    plan.joinControlIndex = geometry->joinControlIndex;
    // The join can put two break points next to each other: the live copy
    // carries the gap tag and its cubic-spline goal like any other
    // structural edit.
    applyGapSpanPolicy(merged.controlPoints);
    merged.optimizationMode = mode;
    merged.manualHvTag =
        clicked.manualHvTag == far.manualHvTag ? clicked.manualHvTag : std::string{};
    for (const auto& tag : clicked.tags) {
        addUniqueSorted(merged.tags, tag);
    }
    for (const auto& tag : far.tags) {
        addUniqueSorted(merged.tags, tag);
    }
    // The join span has no real geometry until the merged line is re-fit.
    addUniqueSorted(merged.tags, std::string{kNeedsReoptimizationTag});
    merged.hvClassification =
        classifyFiberHv(storedControlPointPositions(merged.controlPoints));
    merged.coordinateBaseShapeZYX = clicked.coordinateBaseShapeZYX;
    // Third-party links only: the pair link between the merge endpoints is
    // consumed, and other mutual links were rejected above. Directions are
    // left as stored; the batch canonicalization recomputes them from the
    // merged line (the incident: a copied direction on a reversed line).
    for (const auto& branch : clicked.branches) {
        if (branchReferencesFiber(branch, far.id, far.fileName) || branch.controlPointIndex < 0) {
            continue;
        }
        FiberBranchRef moved = branch;
        moved.controlPointIndex = plan.clickedRemap(branch.controlPointIndex);
        merged.branches.push_back(std::move(moved));
    }
    for (const auto& branch : far.branches) {
        if (branchReferencesFiber(branch, clicked.id, clicked.fileName) ||
            branch.controlPointIndex < 0) {
            continue;
        }
        FiberBranchRef moved = branch;
        moved.controlPointIndex = plan.farRemap(branch.controlPointIndex);
        merged.branches.push_back(std::move(moved));
    }
    return plan;
}

std::optional<SplitPlanResult> planFiberSplit(const StoredFiber& parent,
                                              std::size_t splitAfter,
                                              const NewFiberIdentity& prefixIdentity,
                                              const NewFiberIdentity& suffixIdentity,
                                              bool linkHalves,
                                              std::string* error)
{
    const auto plan = computeFiberSplitPlan(
        storedControlPointPositions(parent.controlPoints), parent.linePoints, splitAfter);
    if (!plan) {
        fail(error, "Cannot split here: each half must keep at least 2 control "
                    "points on the stored line.");
        return std::nullopt;
    }

    SplitPlanResult result;
    result.plan = *plan;
    const auto makeHalf = [&](StoredFiber& half, const NewFiberIdentity& identity) {
        applyIdentity(half, identity);
        half.manualHvTag = parent.manualHvTag;
        half.tags = parent.tags;
        half.width = parent.width;
        half.widthGapFraction = parent.widthGapFraction;
        // The parent's extrapolated tails don't survive the cut: each half
        // ends exactly on its split CP until its line is re-fit.
        addUniqueSorted(half.tags, std::string{kNeedsReoptimizationTag});
        half.optimizationMode = parent.optimizationMode;
        half.coordinateBaseShapeZYX = parent.coordinateBaseShapeZYX;
    };
    StoredFiber& prefix = result.prefix;
    StoredFiber& suffix = result.suffix;
    makeHalf(prefix, prefixIdentity);
    makeHalf(suffix, suffixIdentity);

    prefix.controlPoints.assign(parent.controlPoints.begin(),
                                parent.controlPoints.begin() +
                                    static_cast<std::ptrdiff_t>(plan->prefixControlCount));
    // The removed span's descriptor lives on the new final CP; clearing it
    // is the span deletion and restores the v3 final-CP contract.
    prefix.controlPoints.back().segmentToNext.reset();
    prefix.linePoints.assign(parent.linePoints.begin(),
                             parent.linePoints.begin() +
                                 static_cast<std::ptrdiff_t>(plan->prefixLineCount));
    suffix.controlPoints.assign(parent.controlPoints.begin() +
                                    static_cast<std::ptrdiff_t>(plan->suffixControlBegin),
                                parent.controlPoints.end());
    suffix.linePoints.assign(parent.linePoints.begin() +
                                 static_cast<std::ptrdiff_t>(plan->suffixLineBegin),
                             parent.linePoints.end());
    applyGapSpanPolicy(prefix.controlPoints);
    applyGapSpanPolicy(suffix.controlPoints);

    for (const auto& branch : parent.branches) {
        const auto remapped = remappedSplitControlPointIndex(*plan, branch.controlPointIndex);
        if (!remapped) {
            continue;
        }
        FiberBranchRef moved = branch;
        moved.controlPointIndex = remapped->second;
        (remapped->first ? suffix : prefix).branches.push_back(std::move(moved));
    }
    prefix.hvClassification = classifyFiberHv(storedControlPointPositions(prefix.controlPoints));
    suffix.hvClassification = classifyFiberHv(storedControlPointPositions(suffix.controlPoints));

    if (linkHalves) {
        // "Split and link, same winding": record the boundary connection as
        // a reciprocal branch link. It lands pending like any manual link.
        const cv::Vec3d prefixPoint = prefix.controlPoints.back();
        const cv::Vec3d suffixPoint = suffix.controlPoints.front();
        const cv::Vec3d fallbackDirection = normalizedOrZero(suffixPoint - prefixPoint);

        FiberBranchRef prefixRef;
        prefixRef.pending = true;
        prefixRef.controlPointIndex = static_cast<int>(prefix.controlPoints.size()) - 1;
        prefixRef.branchFiberId = suffix.id;
        prefixRef.branchControlPointIndex = 0;
        prefixRef.branchFileName = suffix.fileName;
        prefixRef.controlPointDirection =
            endpointTangentFromLinePoints(prefix.linePoints, prefixPoint, fallbackDirection);
        prefixRef.branchControlPointDirection =
            endpointTangentFromLinePoints(suffix.linePoints, suffixPoint, -fallbackDirection);
        prefixRef.controlPointPosition = prefixPoint;
        prefixRef.branchControlPointPosition = suffixPoint;

        FiberBranchRef suffixRef;
        suffixRef.pending = true;
        suffixRef.controlPointIndex = 0;
        suffixRef.branchFiberId = prefix.id;
        suffixRef.branchControlPointIndex = prefixRef.controlPointIndex;
        suffixRef.branchFileName = prefix.fileName;
        suffixRef.controlPointDirection = prefixRef.branchControlPointDirection;
        suffixRef.branchControlPointDirection = prefixRef.controlPointDirection;
        suffixRef.controlPointPosition = suffixPoint;
        suffixRef.branchControlPointPosition = prefixPoint;

        prefix.branches.push_back(std::move(prefixRef));
        suffix.branches.push_back(std::move(suffixRef));
    }
    return result;
}

std::optional<std::size_t> redirectBranchRefs(std::vector<FiberBranchRef>& refs,
                                              const std::vector<BranchRedirectSource>& sources,
                                              std::string* error)
{
    // Resolve everything first so a failure modifies nothing.
    std::vector<std::pair<std::size_t, RedirectTarget>> resolved;
    for (std::size_t i = 0; i < refs.size(); ++i) {
        const FiberBranchRef& ref = refs[i];
        for (const auto& source : sources) {
            if (!branchReferencesFiber(ref, source.id, source.fileName)) {
                continue;
            }
            if (!source.stored || !source.map) {
                fail(error, "internal: redirect source without a stored record");
                return std::nullopt;
            }
            // A live session's ref can hold the partner pane's SESSION index
            // for this control point; the position is the primary key.
            const auto storedIndex = matchingStoredControlPointIndex(
                source.stored->controlPoints, ref.branchControlPointIndex,
                ref.branchControlPointPosition);
            if (!storedIndex) {
                if (error) {
                    *error = "a link to " + fiberErrorName(source.fileName) +
                             " does not resolve on its stored control points";
                }
                return std::nullopt;
            }
            const auto target = source.map(*storedIndex);
            if (!target) {
                if (error) {
                    *error = "a link to " + fiberErrorName(source.fileName) +
                             " points at a control point the edit does not keep";
                }
                return std::nullopt;
            }
            resolved.emplace_back(i, *target);
            break;
        }
    }
    for (const auto& [index, target] : resolved) {
        FiberBranchRef& ref = refs[index];
        ref.branchFiberId = target.id;
        ref.branchFileName = target.fileName;
        ref.branchControlPointIndex = target.controlPointIndex;
    }
    return resolved.size();
}

std::vector<std::string> referencedFiberFileNames(const nlohmann::json& root)
{
    std::vector<std::string> names;
    if (!root.is_object()) {
        return names;
    }
    for (const char* kind : {"branches", "adjacent_branches"}) {
        const auto it = root.find(kind);
        if (it == root.end() || !it->is_array()) {
            continue;
        }
        for (const auto& entry : *it) {
            if (!entry.is_object()) {
                continue;
            }
            const auto file = entry.find("branch_file");
            if (file == entry.end() || !file->is_string()) {
                continue;
            }
            const std::string name =
                std::filesystem::path(file->get<std::string>()).filename().string();
            if (!name.empty()) {
                names.push_back(name);
            }
        }
    }
    return names;
}

std::string fiberGeometryKey(const std::vector<StoredControlPoint>& controls,
                             const std::vector<cv::Vec3d>& linePoints)
{
    if (controls.empty() && linePoints.empty()) return {};
    std::string key;
    key.reserve(sizeof(uint64_t) + (controls.size() + linePoints.size()) * 3 * sizeof(uint64_t));
    auto appendPoint = [&key](const cv::Vec3d& point) {
        for (int axis = 0; axis < 3; ++axis) {
            uint64_t bits = 0;
            std::memcpy(&bits, &point[axis], sizeof(bits));
            key.append(reinterpret_cast<const char*>(&bits), sizeof(bits));
        }
    };
    const uint64_t controlCount = controls.size();
    key.append(reinterpret_cast<const char*>(&controlCount), sizeof(controlCount));
    for (const auto& control : controls) appendPoint(control);
    for (const auto& point : linePoints) appendPoint(point);
    return key;
}

}  // namespace vc3d::line_annotation
