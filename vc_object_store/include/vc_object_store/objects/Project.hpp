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
//   Its UUID is named by kind, location and owner, so clients adding the same
//   resource agree on it. Removing one retires it.
// - kProjectViewKind: one per author (name), its meta the author's view state.
#include "vc_object_store/objects/StoreReader.hpp"
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
// A standalone copy of a locally attached project at `destination` (which
// must not exist), named `name`: every object, relationship and its history,
// with its own writable store identity and the project's artifacts. Immutable
// dependencies retain their identities in the copied closure; sharded data
// stays sealed behind the writable root. Move the .stores/.artifacts sidecars
// with the root file. Operational retry receipts are not copied into the new
// writable store identity.
void copyProject(ObjectStoreSession&, const AttachmentIdentity&, const std::filesystem::path& destination,
                 const std::string& name, const OpenOptions& options);
// The store's project record; nullopt for a store without one.
std::optional<ObjectRecord> projectRecord(const StoreReader&);
// Name the project, creating its record when the store has none.
void setProjectName(ObjectStoreSession&, const AttachmentIdentity&, const std::string& name);

// The live resources of `kind` `viewer` sees (the shared ones and those it
// owns; every one when nullopt), in creation order. Legacy whole-second
// stamps sort after precise stamps within their second; ties use UUID.
std::vector<ProjectResource> projectResources(const StoreReader&, const std::string& kind,
                                              const std::optional<std::string>& viewer = std::nullopt);
// Deterministic UUID for a new resource. An existing legacy resource keeps
// its stored UUID; use the listed or returned UUID for generic object APIs.
std::string projectResourceUuid(const std::string& kind, const std::string& location, const std::string& owner = {});
// Candidate record for a new resource, before resolving any legacy identity.
ObjectRecord projectResourceRecord(const std::string& kind, const std::string& location,
                                   const std::vector<std::string>& tags = {},
                                   const nlohmann::json& meta = nlohmann::json::object(),
                                   const std::string& owner = {});
// Preserve an existing legacy UUID only when it names this exact resource.
ObjectRecord resolveProjectResourceIdentity(const StoreReader&, ObjectRecord);
// Direct-store counterpart of the checked session helper, with the same
// identity and retirement rules. Returns the actual stored UUID.
std::string addProjectResource(ObjectStore&, WriteTxn&, ObjectRecord);
// Add a resource (or bring back a removed one) with these tags and meta, shared
// or `owner`'s; false when it is already there, unchanged.
bool addProjectResource(ObjectStoreSession&, const AttachmentIdentity&, const std::string& kind,
                        const std::string& location, const std::vector<std::string>& tags = {},
                        const nlohmann::json& meta = nlohmann::json::object(), const std::string& owner = {});
// Merge metadata without changing the resource's owner. The deterministic
// new-resource UUID also resolves to a matching legacy record here.
void updateProjectResourceMeta(ObjectStoreSession&, const AttachmentIdentity&, const std::string& uuid,
                               const nlohmann::json& meta);
// Retire a resource; false when it is not there.
bool removeProjectResource(ObjectStoreSession&, const AttachmentIdentity&, const std::string& uuid);

// `author`'s view of the project (an empty object without one).
nlohmann::json projectView(const StoreReader&, const std::string& author);
// Replace `author`'s view; no change, no revision.
void setProjectView(ObjectStoreSession&, const AttachmentIdentity&, const std::string& author,
                    const nlohmann::json& view);

} // namespace vc::objects
