#pragma once

// Identity bookkeeping for deleting stored fibers across a wait.
//
// LineAnnotationController::deleteFibers has to drain queued saves before it
// removes files, and the drain runs a nested event loop that still delivers
// input: the package can change, a fiber can be deleted, renamed or replaced
// meanwhile (and, before runtime ids were made stable across reloads, a reload
// renumbered every fiber - the regression the tests below still replay). The
// identity that survives the wait is the file name, and the identity of the
// package is its generation counter plus its fibers directory. These helpers
// hold that reasoning in one place, free of Qt and of the controller, so the
// capture / wait / resolve sequence can be exercised in a unit test with a
// wait that changes the list.
//
// Fiber is any type with `uint64_t id` and `std::string fileName` members.

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "LineAnnotationFiberIdentity.hpp"

namespace vc3d::line_annotation
{

struct FiberDeleteTarget {
    uint64_t requestedId = 0;
    std::string fileName;
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
    // Current runtime ids of the targets still loaded, matched by file name.
    // Sorted and unique, for the caller's binary searches.
    std::vector<uint64_t> resolvedIds;
    // Targets whose file name no longer names a loaded fiber.
    std::vector<FiberDeleteTarget> missing;
    // Targets whose file name names more than one loaded fiber; deleting
    // either would be a guess, so neither is.
    std::vector<FiberDeleteTarget> ambiguous;
    // From the capture, carried through for the caller's reporting.
    std::vector<FiberDeleteTarget> targets;
    std::vector<uint64_t> notLoaded;
    std::vector<uint64_t> unnamed;
};

// What a delete did, for callers that must report per requested fiber: a
// requested id can stop naming a loaded fiber during the wait, so the outcome
// speaks in the file names captured before it. `requested` holds the
// capture (requested id -> file name) and `deletedFileNames` the files that
// were removed (or found already absent); `aborted` says the package identity moved and
// nothing was removed.
struct FiberDeleteOutcome {
    bool aborted = false;
    std::string error;
    std::vector<FiberDeleteTarget> requested;
    std::vector<std::string> deletedFileNames;
    // Current runtime ids of the deleted fibers, as emitted to observers.
    std::vector<uint64_t> deletedIds;

    [[nodiscard]] bool deletedRequested(uint64_t requestedId) const
    {
        for (const FiberDeleteTarget& target : requested) {
            if (target.requestedId == requestedId) {
                return std::find(deletedFileNames.begin(), deletedFileNames.end(),
                                 target.fileName) != deletedFileNames.end();
            }
        }
        return false;
    }
};

// Before the wait: the file name each requested id would be deleted at.
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
        capture.targets.push_back(FiberDeleteTarget{requestedId, it->fileName});
    }
    return capture;
}

// After the wait: refuse if the package identity moved, otherwise match each
// captured file name against the current list. The requested ids are never
// consulted here; the names are the identity.
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
    if (before.fibersDir.empty() || after.fibersDir.empty()) {
        resolution.aborted = true;
        resolution.abortReason = "the package has no fibers directory";
        return resolution;
    }
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
            if (fiber.fileName == target.fileName) {
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
// deleted, for suppressing its save. A session that knows its file name is
// matched by that name only (the file name is the identity; an id can be
// stale); an id match counts only for a session with no name, which has no
// other identity.
inline bool sessionBelongsToDeletedFiber(uint64_t sessionFiberId,
                                         const std::string& sessionFileName,
                                         const std::vector<uint64_t>& deletedIdsSorted,
                                         const std::vector<std::string>& deletedFileNames)
{
    if (!sessionFileName.empty()) {
        return std::find(deletedFileNames.begin(), deletedFileNames.end(), sessionFileName) !=
               deletedFileNames.end();
    }
    return std::binary_search(deletedIdsSorted.begin(), deletedIdsSorted.end(), sessionFiberId);
}

// Whether a branch (cross-fiber link) reference points at a fiber just
// deleted. Branch refs carry both the runtime id and the file name of the
// fiber they point to; the id can be stale after a reload for the same
// reason as a session's, so when both sides have a file name the names
// decide, and the id counts only when a name is missing on either side.
inline bool branchRefersToDeletedFiber(uint64_t branchFiberId,
                                       const std::string& branchFileName,
                                       uint64_t deletedFiberId,
                                       const std::string& deletedFileName)
{
    return sameFiberIdentity(branchFiberId, branchFileName, deletedFiberId, deletedFileName);
}

} // namespace vc3d::line_annotation
