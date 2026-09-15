#pragma once

// Runtime identity of stored fibers and open annotation sessions.
//
// A fiber's stable identity is its file name. Its runtime id is a per-package
// number used as a key everywhere in the controller and by everything that
// listens to it (the Fibers docks, the fiber map, atlas search, the agent
// bridge). Two rules keep the two consistent:
//
//  1. Within one package, a fiber keeps its runtime id across reloads of the
//     fiber list (import, repair reloads, the vpkg-ready load), and ids are
//     never recycled: a new file gets a fresh id above every id ever handed out
//     in the package, a deleted fiber's id stays retired. Only a package switch
//     restarts the numbering. See assignStableRuntimeIds.
//  2. Where a record carries both an id and a file name, the names decide when
//     both are known and the id counts only when a name is missing on either
//     side. See sameFiberIdentity. This covers the window inside a reload and
//     any id captured before one.
//
// Free of Qt and of the controller so the decisions can be unit-tested.

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vc3d::line_annotation
{

// Whether two (id, file name) pairs name the same stored fiber.
inline bool sameFiberIdentity(uint64_t idA, const std::string& fileNameA,
                              uint64_t idB, const std::string& fileNameB)
{
    if (!fileNameA.empty() && !fileNameB.empty()) {
        return fileNameA == fileNameB;
    }
    return idA != 0 && idA == idB;
}

// The id space of one package's reload: which file names already hold an id,
// and the next id never handed out (the high-water mark).
struct RuntimeFiberIdSpace {
    std::unordered_map<std::string, uint64_t> idByFileName;
    uint64_t nextId = 1;
};

// Assigns runtime ids to freshly (re)loaded fibers. Fiber is any type with
// `uint64_t id` and `std::string fileName` members. A fiber whose file name
// the space already knows keeps that id; every other fiber - a new file, an
// empty file name, a duplicate of a name already assigned in this pass -
// gets a fresh id from the mark. `reservedIds` are ids that are live outside
// the stored list (open sessions); the mark is lifted above them so a new
// file cannot collide with one. Ids are never recycled: the mark only rises.
// The space is updated in place with the assignments made.
template <class Fiber>
void assignStableRuntimeIds(std::vector<Fiber>& fibers,
                            RuntimeFiberIdSpace& space,
                            const std::vector<uint64_t>& reservedIds)
{
    for (const uint64_t reserved : reservedIds) {
        space.nextId = std::max(space.nextId, reserved + 1);
    }
    for (const auto& [name, id] : space.idByFileName) {
        space.nextId = std::max(space.nextId, id + 1);
    }
    std::unordered_map<std::string, uint64_t> assignedThisPass;
    assignedThisPass.reserve(fibers.size());
    for (Fiber& fiber : fibers) {
        const bool named = !fiber.fileName.empty();
        if (named) {
            if (const auto known = space.idByFileName.find(fiber.fileName);
                known != space.idByFileName.end() &&
                assignedThisPass.find(fiber.fileName) == assignedThisPass.end()) {
                fiber.id = known->second;
                assignedThisPass.emplace(fiber.fileName, fiber.id);
                continue;
            }
        }
        fiber.id = space.nextId++;
        if (named && assignedThisPass.find(fiber.fileName) == assignedThisPass.end()) {
            assignedThisPass.emplace(fiber.fileName, fiber.id);
            space.idByFileName[fiber.fileName] = fiber.id;
        }
    }
}

// A fresh id for a fiber created outside a reload (a new session, a merge, a
// split): above everything the space has handed out, above every live id the
// caller knows of, and at least `minimumId` (callers that want a child above
// its parent). Advances the mark past the id returned, so it is never handed
// out again - which is also why the caller must not derive further ids from
// the result by arithmetic; it asks again instead.
inline uint64_t allocateRuntimeFiberId(RuntimeFiberIdSpace& space,
                                       const std::vector<uint64_t>& liveIds,
                                       uint64_t minimumId = 0)
{
    for (const uint64_t live : liveIds) {
        space.nextId = std::max(space.nextId, live + 1);
    }
    space.nextId = std::max(space.nextId, minimumId);
    return space.nextId++;
}

} // namespace vc3d::line_annotation
