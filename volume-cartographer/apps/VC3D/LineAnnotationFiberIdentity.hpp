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
//     in the package. A file name that is merely absent from one load (a parse
//     failure, a strict pass) keeps its binding and returns with its id; a file
//     name whose fiber was deleted through the app has its binding retired, so
//     a later file under that name is a new fiber with a new id. Only a package
//     switch restarts the numbering. Identity here is the package-local file
//     name, not the physical file: a file replaced on disk under an unchanged
//     name outside the app is, by this rule, the same fiber. See
//     assignStableRuntimeIds and the bind / retire / rebind helpers.
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

// A fiber created by the app (a saved session, a merge, a split, a linked
// seed) establishes its identity when its first save is accepted or has
// succeeded: its file name is bound to the id it was allocated, so the next
// reload keeps that id. Binding
// a name already bound to another id moves it (the name now denotes this
// fiber); the mark never drops below a bound id.
inline void bindRuntimeFiberIdentity(RuntimeFiberIdSpace& space,
                                     const std::string& fileName,
                                     uint64_t id)
{
    if (fileName.empty() || id == 0) {
        return;
    }
    space.idByFileName[fileName] = id;
    space.nextId = std::max(space.nextId, id + 1);
}

// A fiber deleted through the app: the name's binding is dropped (the id
// itself stays retired, since the mark never falls), so a later file under
// the same name is a new fiber with a fresh id rather than the old one
// resurrected for holders that still remember it.
inline void retireRuntimeFiberIdentity(RuntimeFiberIdSpace& space, const std::string& fileName)
{
    space.idByFileName.erase(fileName);
}

// A fiber renamed through the app keeps its id under the new name; the old
// name is unbound, so a later file under it is a different fiber.
inline void rebindRuntimeFiberIdentity(RuntimeFiberIdSpace& space,
                                       const std::string& oldFileName,
                                       const std::string& newFileName,
                                       uint64_t id)
{
    space.idByFileName.erase(oldFileName);
    bindRuntimeFiberIdentity(space, newFileName, id);
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
