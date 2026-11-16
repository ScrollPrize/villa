#pragma once
// A selected-graph read (LocalStoreReader::selectedInputs, relationshipsAt)
// asks the same questions many times over: every selected object is resolved
// once to read it, again to find its scope, again inside members(); every
// edge's recorded binding is decoded once per selected endpoint, and inside
// each binding both endpoint revisions are resolved in full only to name their
// numbered references, which the selection already holds. Within one call the
// store is one snapshot, so this per-call memo hands back the earlier answer.
// It changes no result: every entry is what the lookup it replaces would return.
#include "vc_object_store/objects/ObjectStore.hpp"
#include <sqlite3.h>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vc::objects::detail {
struct SelectedReadMemo {
    // revision uuid -> the numbered reference of that selected revision
    std::unordered_map<std::string, ObjectVersion> referencesByRevision;
    // revision uuid -> the source that holds it; numbered reference -> its source
    std::unordered_map<std::string, sqlite3*> revisionSources;
    std::map<ObjectVersion, sqlite3*> versionSources;
    // (scope, edge uuid) -> the edge's recorded binding, decoded once
    std::map<std::pair<sqlite3*, std::string>, std::optional<RelationshipRevision>> bindings;
    // (scope, edge uuid) -> edgeSource(scope, edge uuid)
    std::map<std::pair<sqlite3*, std::string>, sqlite3*> edgeSources;
    // (scope, query source) -> object uuid -> its incident live edges, in edge
    // uuid order: the rows the per-object incidence query returns for it
    std::map<sqlite3*, std::unordered_map<std::string, std::vector<std::string>>> incidence;
    // scopes whose incident edges' bindings were decoded in one batch already
    std::set<sqlite3*> prefilled;
};
inline thread_local SelectedReadMemo* activeSelectedRead = nullptr;
// Opens a memo for the enclosing read unless one is open already (then the
// enclosing read's memo keeps serving, since it is the same snapshot).
struct SelectedReadScope {
    SelectedReadMemo memo;
    const bool owner;
    SelectedReadScope() : owner(activeSelectedRead == nullptr) { if (owner) activeSelectedRead = &memo; }
    ~SelectedReadScope() { if (owner) activeSelectedRead = nullptr; }
    SelectedReadScope(const SelectedReadScope&) = delete;
    SelectedReadScope& operator=(const SelectedReadScope&) = delete;
    SelectedReadMemo& active() const { return *activeSelectedRead; }
};
}
