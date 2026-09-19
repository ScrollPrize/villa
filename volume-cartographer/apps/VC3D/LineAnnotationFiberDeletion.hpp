#pragma once

// Identity bookkeeping for deleting stored fibers across a wait.
//
// LineAnnotationController::deleteFibers has to drain queued saves before it
// removes files, and the drain runs a nested event loop that still delivers
// input: the package can change, a fiber can be deleted, renamed or replaced
// meanwhile, and the list can be reloaded. The identity that survives the
// wait is the file: its source root plus its file name (a file name alone is
// not unique across registered sources), and the identity of the package is
// its generation counter plus its fibers directory. These helpers hold that
// reasoning in one place, free of Qt and of the controller, so the capture /
// wait / resolve sequence can be exercised in a unit test with a wait that
// changes the list.
//
// Fiber is any type with `uint64_t id`, `std::string fileName` and
// `std::filesystem::path sourceRoot` members.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace vc3d::line_annotation
{

// Whether two (runtime id, file name) pairs name the same stored fiber: the
// rule shared by branchReferencesFiber, the branch synchronizers and the
// deletion helpers. Runtime ids are unique across every registered source
// and stable across reloads (FiberRuntimeIds), so once both sides carry one
// the ids decide, and an equal file name in another source is not a match.
// The file name is the fallback when either side has no id (legacy or
// not-yet-loaded references).
inline bool sameFiberIdentity(uint64_t idA, const std::string& fileNameA,
                              uint64_t idB, const std::string& fileNameB)
{
    if (idA != 0 && idB != 0) {
        return idA == idB;
    }
    return !fileNameA.empty() && fileNameA == fileNameB;
}

struct FiberDeleteTarget {
    uint64_t requestedId = 0;
    std::string fileName;
    std::filesystem::path sourceRoot;

    [[nodiscard]] bool namesFile(const std::string& otherFileName,
                                 const std::filesystem::path& otherSourceRoot) const
    {
        return fileName == otherFileName && sourceRoot == otherSourceRoot;
    }
};

struct FiberDeletePackageIdentity {
    uint64_t packageGeneration = 0;
    std::string fibersDir;
};

struct FiberDeleteCapture {
    // Requested fibers that were loaded and named at capture time.
    std::vector<FiberDeleteTarget> targets;
    // Requested ids that named no loaded fiber.
    std::vector<uint64_t> notLoaded;
    // Requested ids that named a loaded fiber without a file name; such a
    // fiber has no identity that survives a reload, so it is never deleted
    // through this path.
    std::vector<uint64_t> unnamed;
};

struct FiberDeleteResolution {
    // The package identity moved during the wait: nothing may be deleted.
    bool aborted = false;
    std::string abortReason;
    // Current runtime ids of the targets still loaded, matched by source root
    // and file name. Sorted and unique, for the caller's binary searches.
    std::vector<uint64_t> resolvedIds;
    // Targets whose file no longer names a loaded fiber.
    std::vector<FiberDeleteTarget> missing;
    // Targets whose file names more than one loaded fiber; deleting either
    // would be a guess, so neither is.
    std::vector<FiberDeleteTarget> ambiguous;
    // From the capture, carried through for the caller's reporting.
    std::vector<FiberDeleteTarget> targets;
    std::vector<uint64_t> notLoaded;
    std::vector<uint64_t> unnamed;
};

// A fiber the delete removed (or found already absent).
struct FiberDeleted {
    uint64_t id = 0;
    std::string fileName;
    std::filesystem::path sourceRoot;
};

// What a delete did, for callers that must report per requested fiber: a
// requested id can stop naming a loaded fiber during the wait, so the outcome
// speaks in the files captured before it. `requested` holds the capture
// (requested id -> file) and `deleted` the files that were removed; `aborted`
// says the package identity moved and nothing was removed.
struct FiberDeleteOutcome {
    bool aborted = false;
    std::string error;
    std::vector<FiberDeleteTarget> requested;
    std::vector<FiberDeleted> deleted;
    // Current runtime ids of the deleted fibers, as emitted to observers.
    std::vector<uint64_t> deletedIds;

    [[nodiscard]] bool deletedRequested(uint64_t requestedId) const
    {
        for (const FiberDeleteTarget& target : requested) {
            if (target.requestedId != requestedId) {
                continue;
            }
            return std::any_of(deleted.begin(), deleted.end(), [&target](const FiberDeleted& entry) {
                return target.namesFile(entry.fileName, entry.sourceRoot);
            });
        }
        return false;
    }
};

// Before the wait: the file each requested id would be deleted at.
template <class Fiber>
FiberDeleteCapture captureFiberDeleteTargets(const std::vector<uint64_t>& requestedIds,
                                             const std::vector<Fiber>& fibers)
{
    FiberDeleteCapture capture;
    for (const uint64_t requestedId : requestedIds) {
        const auto it = std::find_if(fibers.begin(), fibers.end(), [requestedId](const Fiber& fiber) {
            return fiber.id == requestedId;
        });
        if (it == fibers.end()) {
            capture.notLoaded.push_back(requestedId);
            continue;
        }
        if (it->fileName.empty()) {
            capture.unnamed.push_back(requestedId);
            continue;
        }
        capture.targets.push_back(FiberDeleteTarget{requestedId, it->fileName, it->sourceRoot});
    }
    return capture;
}

// After the wait: refuse if the package identity moved, otherwise match each
// captured file against the current list. The requested ids are never
// consulted here; the files are the identity.
template <class Fiber>
FiberDeleteResolution resolveFiberDeleteTargets(const FiberDeleteCapture& capture,
                                                const std::vector<Fiber>& fibersNow,
                                                const FiberDeletePackageIdentity& before,
                                                const FiberDeletePackageIdentity& after)
{
    FiberDeleteResolution resolution;
    resolution.targets = capture.targets;
    resolution.notLoaded = capture.notLoaded;
    resolution.unnamed = capture.unnamed;
    // An empty fibers directory is a valid configuration (fibers loaded from
    // registered external sources only; each target carries its own source
    // root), so only a CHANGE of directory or package aborts.
    if (before.packageGeneration != after.packageGeneration) {
        resolution.aborted = true;
        resolution.abortReason = "the project changed";
        return resolution;
    }
    if (before.fibersDir != after.fibersDir) {
        resolution.aborted = true;
        resolution.abortReason = "the fibers directory changed";
        return resolution;
    }
    for (const FiberDeleteTarget& target : capture.targets) {
        uint64_t matchedId = 0;
        int matches = 0;
        for (const Fiber& fiber : fibersNow) {
            if (target.namesFile(fiber.fileName, fiber.sourceRoot)) {
                ++matches;
                matchedId = fiber.id;
            }
        }
        if (matches == 0) {
            resolution.missing.push_back(target);
        } else if (matches > 1) {
            resolution.ambiguous.push_back(target);
        } else {
            resolution.resolvedIds.push_back(matchedId);
        }
    }
    std::sort(resolution.resolvedIds.begin(), resolution.resolvedIds.end());
    resolution.resolvedIds.erase(
        std::unique(resolution.resolvedIds.begin(), resolution.resolvedIds.end()),
        resolution.resolvedIds.end());
    return resolution;
}

// The whole sequence: capture from the current list, wait, resolve against
// the list as it is after the wait. `fibersNow()` returns the current list
// (by reference; the wait may replace its contents), `identityNow()` the
// current package identity, and `wait()` is the drain. A capture with no
// targets skips the wait: there is nothing a drain would protect.
template <class FibersNow, class IdentityNow, class Wait>
FiberDeleteResolution resolveFiberDeletionAcrossWait(const std::vector<uint64_t>& requestedIds,
                                                     FibersNow fibersNow,
                                                     IdentityNow identityNow,
                                                     Wait wait)
{
    const FiberDeleteCapture capture = captureFiberDeleteTargets(requestedIds, fibersNow());
    if (capture.targets.empty()) {
        FiberDeleteResolution resolution;
        resolution.targets = capture.targets;
        resolution.notLoaded = capture.notLoaded;
        resolution.unnamed = capture.unnamed;
        return resolution;
    }
    const FiberDeletePackageIdentity before = identityNow();
    wait();
    const FiberDeletePackageIdentity after = identityNow();
    return resolveFiberDeleteTargets(capture, fibersNow(), before, after);
}

// Whether an open annotation session belongs to one of the fibers just
// deleted, for suppressing its save: sameFiberIdentity against each deleted
// fiber.
inline bool sessionBelongsToDeletedFiber(uint64_t sessionFiberId,
                                         const std::string& sessionFileName,
                                         const std::vector<FiberDeleted>& deleted)
{
    return std::any_of(deleted.begin(), deleted.end(), [&](const FiberDeleted& entry) {
        return sameFiberIdentity(sessionFiberId, sessionFileName, entry.id, entry.fileName);
    });
}

// Whether a branch (cross-fiber link) reference points at a fiber just
// deleted; the same rule.
inline bool branchRefersToDeletedFiber(uint64_t branchFiberId,
                                       const std::string& branchFileName,
                                       uint64_t deletedFiberId,
                                       const std::string& deletedFileName)
{
    return sameFiberIdentity(branchFiberId, branchFileName, deletedFiberId, deletedFileName);
}

} // namespace vc3d::line_annotation
