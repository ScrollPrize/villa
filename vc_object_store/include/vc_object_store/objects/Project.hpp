#pragma once
// A store as a project: besides its objects, the data a project uses that the
// store does not hold (volumes, normal grids, Lasagna datasets), as resource
// records, and each author's own view of it.
//
// - kProjectKind: at most one; its name is the project's name, its meta free
//   for the application (e.g. the Open Data sample it was made from). Any
//   store is a project; one without the record is unnamed.
// - Resource kinds: one record per (kind, location, owner), storageKind
//   kDatasetStorage, storageRef the location (URL or local path), tags and meta
//   for the application. A shared resource (no owner) is everyone's; an owned
//   one (meta.owner, e.g. a path on its author's disk) is its owner's alone.
//   Its UUID is named by kind, location and owner, so replicas adding the same
//   resource agree on it. Removing one retires it.
// - kProjectViewKind: one per author (name), its meta the author's view state.
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace vc::objects {

inline constexpr const char* kProjectKind = "project";
inline constexpr const char* kProjectViewKind = "project_view";
inline constexpr const char* kDatasetStorage = "dataset";
inline constexpr const char* kVolumeResource = "volume";
inline constexpr const char* kNormalGridResource = "normal_grid";
inline constexpr const char* kLasagnaDatasetResource = "lasagna_dataset";
bool isProjectResourceKind(std::string_view kind);

struct ProjectResource {
    std::string uuid, kind;
    std::string location;
    std::string owner;  // empty: shared
    std::vector<std::string> tags;
    nlohmann::json meta = nlohmann::json::object();
};

// Create a store at `path` (which must not exist) holding a project named
// `name` with meta `meta`.
void createProject(const std::filesystem::path& path, const std::string& name, const nlohmann::json& meta,
                   const OpenOptions& options);
// A standalone copy of an attached project at `destination` (which must not
// exist), named `name`: every object, relationship and its history, under a
// store identity of its own (a copy of a hosted replica is not one), with the
// project's artifact directory.
void copyProject(ObjectStoreSession&, const AttachmentIdentity&, const std::filesystem::path& destination,
                 const std::string& name, const OpenOptions& options);
// The store's project record; nullopt for a store without one.
std::optional<ObjectRecord> projectRecord(const ObjectStore&, const Txn&);
// Name the project, creating its record when the store has none.
void setProjectName(ObjectStoreSession&, const AttachmentIdentity&, const std::string& name);

// The live resources of `kind` `viewer` sees (the shared ones and those it
// owns; every one when nullopt), in the order they were added.
std::vector<ProjectResource> projectResources(const ObjectStore&, const Txn&, const std::string& kind,
                                              const std::optional<std::string>& viewer = std::nullopt);
// The resource's UUID, the same in every store.
std::string projectResourceUuid(const std::string& kind, const std::string& location, const std::string& owner = {});
// The record of a resource, as addProjectResource writes it (upsertObject).
ObjectRecord projectResourceRecord(const std::string& kind, const std::string& location,
                                   const std::vector<std::string>& tags = {},
                                   const nlohmann::json& meta = nlohmann::json::object(),
                                   const std::string& owner = {});
// Add a resource (or bring back a removed one) with these tags and meta, shared
// or `owner`'s; false when it is already there, unchanged.
bool addProjectResource(ObjectStoreSession&, const AttachmentIdentity&, const std::string& kind,
                        const std::string& location, const std::vector<std::string>& tags = {},
                        const nlohmann::json& meta = nlohmann::json::object(), const std::string& owner = {});
// Merge `meta`'s keys into the resource's meta; no change, no revision.
void updateProjectResourceMeta(ObjectStoreSession&, const AttachmentIdentity&, const std::string& uuid,
                               const nlohmann::json& meta);
// Retire a resource; false when it is not there.
bool removeProjectResource(ObjectStoreSession&, const AttachmentIdentity&, const std::string& uuid);

// `author`'s view of the project (an empty object without one).
nlohmann::json projectView(const ObjectStore&, const Txn&, const std::string& author);
// Replace `author`'s view; no change, no revision.
void setProjectView(ObjectStoreSession&, const AttachmentIdentity&, const std::string& author,
                    const nlohmann::json& view);

} // namespace vc::objects
