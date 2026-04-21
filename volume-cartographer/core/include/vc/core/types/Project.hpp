#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "utils/Json.hpp"

namespace vc {

// Type of data that a DataSource exposes.
enum class DataSourceType {
    VolumesDir,       // directory whose children are each a zarr volume
    ZarrVolume,       // single zarr volume (meta.json + zarr levels)
    SegmentsDir,      // directory whose children are each a segment
    Segment,          // single segment
    NormalGrid,       // single normal grid
    NormalDirVolume,  // directory-backed normal/dir volume
    SyncDir           // bidirectional mirror between a local dir and remote URL
};

enum class SyncDirection {
    Pull,             // remote -> local (default)
    Push,             // local -> remote
    Both              // two-way mirror
};

std::string sync_direction_to_string(SyncDirection d);
SyncDirection sync_direction_from_string(const std::string& s);

enum class LocationKind {
    Local,            // filesystem path (absolute, or relative to project dir)
    Remote            // URL (http://, https://, s3://, ...)
};

std::string data_source_type_to_string(DataSourceType t);
DataSourceType data_source_type_from_string(const std::string& s);
std::string location_kind_to_string(LocationKind k);
LocationKind location_kind_from_string(const std::string& s);

struct DataSource {
    std::string id;
    DataSourceType type = DataSourceType::SegmentsDir;
    std::string location;
    LocationKind location_kind = LocationKind::Local;
    // For directory-typed sources: descend into subdirs on load.
    bool recursive = true;
    // For directory-typed sources: watch for changes (inotify etc.).
    bool track_changes = false;
    // When false, the loader will skip this source but keep it listed.
    bool enabled = true;
    // Optional explicit child IDs (used when recursive is false, or to pin
    // a curated subset). Child IDs refer to other DataSource entries.
    std::vector<std::string> children;
    // Free-form labels for filtering / grouping. Spec calls this out as a
    // "generic tagging" capability that applies to any source, not just
    // segments.
    std::vector<std::string> tags;

    // SyncDir-only: the remote URL paired with this.location (local path),
    // and the direction of synchronisation.
    std::string sync_remote;
    SyncDirection sync_direction = SyncDirection::Pull;

    // True if this source was pulled in via a LinkedProject reference.
    // Transient: not serialized to JSON (otherwise the sources would
    // duplicate when a linked-through project was re-saved).
    bool imported = false;

    [[nodiscard]] bool is_remote() const noexcept
    {
        return location_kind == LocationKind::Remote;
    }
};

// A Group bundles a set of DataSource ids under a user-facing label,
// letting the UI show sources in collapsible sections.
struct Group {
    std::string id;
    std::string name;
    std::vector<std::string> source_ids;
};

// Reference to another Project file. On load, the referenced project's
// data sources are merged into the current view (id-prefixed to avoid
// collisions). Lets one project "track" another — e.g. a vis project
// that imports segments from a segmentation project.
struct LinkedProject {
    std::string path;     // absolute or project-relative path to the JSON
    bool read_only = true;
    std::string id_prefix; // prefix applied to imported sources' ids
};

// Classify a location string (e.g. "s3://...", "https://...", "/abs/path")
// as remote or local by scheme sniffing. Returns Remote iff the string has
// a recognised URL scheme.
LocationKind infer_location_kind(const std::string& location);

// Project is the new top-level VC3D working-set descriptor. It replaces the
// rigid on-disk volpkg layout with a JSON document listing flexible
// DataSource entries. A Project can still be backed by a legacy volpkg
// directory (see from_volpkg()).
class Project {
public:
    Project() = default;

    std::string name;
    int version = 1;
    std::string description;

    std::vector<DataSource> data_sources;
    std::vector<Group> groups;
    std::vector<LinkedProject> linked_projects;

    // Records where this project came from so that legacy volpkg behaviour
    // (writing new segments into <root>/paths/, etc.) keeps working.
    struct Origin {
        std::string kind;                 // "volpkg" | "project"
        std::filesystem::path root;       // root dir of the legacy layout
        // Full verbatim config.json from the legacy volpkg (minus name/version
        // which we promote to top level). Preserved so round-tripping through
        // a project does not drop voxel size, scroll id, material, etc.
        utils::Json legacy_config;
    };
    std::optional<Origin> origin;

    // The active segments_dir source (legacy "current segmentation directory").
    // Empty = first segments_dir source.
    std::string active_segments_source_id;

    // Where newly created segments land (id of a segments_dir source).
    std::string output_segments_source_id;

    // ---- Serialization ----
    [[nodiscard]] utils::Json to_json() const;
    static Project from_json(const utils::Json& j);

    void save_to_file(const std::filesystem::path& path) const;
    static Project load_from_file(const std::filesystem::path& path);

    // ---- Volpkg compatibility ----

    // True if `dir` looks like a volpkg root (has config.json with name+version).
    static bool looks_like_volpkg(const std::filesystem::path& dir);

    // Build a Project that mirrors an existing volpkg directory.
    static Project from_volpkg(const std::filesystem::path& volpkg_root);

    // True if this project's layout is compatible with being written back
    // onto a volpkg directory (i.e. origin points at one).
    [[nodiscard]] bool is_volpkg_compatible() const;

    // ---- Lookups ----
    [[nodiscard]] const DataSource* find_source(const std::string& id) const;
    DataSource* find_source(const std::string& id);

    [[nodiscard]] std::vector<const DataSource*>
        sources_of_type(DataSourceType t) const;

    // Filter helpers — return pointers into data_sources, so callers
    // don't have to hand-loop. Tags match any (OR) semantics.
    [[nodiscard]] std::vector<const DataSource*>
        sources_with_tag(const std::string& tag) const;
    [[nodiscard]] std::vector<const DataSource*>
        sources_in_group(const std::string& group_id) const;

    // Recursively follow linked_projects and append their data_sources
    // into this project as "imported" entries (marked so they're skipped
    // on save). Id collisions are avoided by prefixing with the linked
    // project's id_prefix. Safe against cycles.
    void merge_linked_projects();

    // Resolve a DataSource location to a concrete filesystem path.
    // Throws if the source is Remote. Relative paths resolve against
    // origin.root (when set) or the directory of path().
    [[nodiscard]] std::filesystem::path
        resolve_local(const DataSource& ds) const;

    // Return the URL for a remote DataSource. Throws if the source is Local.
    [[nodiscard]] std::string remote_url(const DataSource& ds) const;

    // Return a string usable by loaders that accept "either a local path or
    // a URL": the absolute local path for Local sources, the URL for Remote.
    [[nodiscard]] std::string resolve_location(const DataSource& ds) const;

    // Resolve the on-disk path for a segments_dir by id. If no source with
    // that id exists but this project has a volpkg origin, falls back to
    // <volpkg root>/<id_or_name> so legacy layouts keep working.
    // Throws if nothing can be resolved.
    [[nodiscard]] std::filesystem::path
        resolve_segments_dir(const std::string& id_or_name) const;

    // Active (currently selected) segments_dir. First segments_dir source,
    // or active_segments_source_id if set. Falls back to <volpkg>/paths.
    [[nodiscard]] std::filesystem::path
        resolve_active_segments_dir() const;

    // Where new segments should land. output_segments_source_id if set,
    // else the active segments dir.
    [[nodiscard]] std::filesystem::path
        resolve_output_segments_dir() const;

    // First volumes_dir source, else <volpkg>/volumes.
    [[nodiscard]] std::filesystem::path
        resolve_volumes_dir() const;

    // All segments_dir source paths (for file watchers etc.). For
    // volpkg-derived projects with no explicit sources, returns the
    // conventional "paths"/"traces"/"export" dirs that exist on disk.
    [[nodiscard]] std::vector<std::filesystem::path>
        all_segments_dirs() const;

    // Resolve the path for a sibling file like "seed.json", "expand.json",
    // "trace_params.json", "remote_volumes.json". For volpkg-derived
    // projects this is <origin.root>/<filename>; for a saved pure project
    // it's next to the project JSON.
    [[nodiscard]] std::filesystem::path
        support_file_path(const std::string& filename) const;

    // Path on disk of the currently loaded project JSON (empty if unsaved).
    [[nodiscard]] std::filesystem::path path() const { return path_; }
    void set_path(const std::filesystem::path& p) { path_ = p; }

private:
    std::filesystem::path path_;
};

} // namespace vc
