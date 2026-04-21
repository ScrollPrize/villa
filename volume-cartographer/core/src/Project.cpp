#include "vc/core/types/Project.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>

#include "vc/core/util/LoadJson.hpp"
#include "vc/core/util/Logging.hpp"

namespace vc {

namespace {

constexpr const char* kSchema = "vc3d-project";
constexpr int kSchemaVersion = 1;

// Currently-known segmentation subdir names inside a legacy volpkg.
const std::vector<std::string>& kLegacySegmentsDirs()
{
    static const std::vector<std::string> v = {"paths", "traces", "export"};
    return v;
}

} // namespace

std::string data_source_type_to_string(DataSourceType t)
{
    switch (t) {
        case DataSourceType::VolumesDir:      return "volumes_dir";
        case DataSourceType::ZarrVolume:      return "zarr_volume";
        case DataSourceType::SegmentsDir:     return "segments_dir";
        case DataSourceType::Segment:         return "segment";
        case DataSourceType::NormalGrid:      return "normal_grid";
        case DataSourceType::NormalDirVolume: return "normal_dir_volume";
        case DataSourceType::SyncDir:         return "sync_dir";
    }
    throw std::runtime_error("unknown DataSourceType");
}

DataSourceType data_source_type_from_string(const std::string& s)
{
    if (s == "volumes_dir")       return DataSourceType::VolumesDir;
    if (s == "zarr_volume")       return DataSourceType::ZarrVolume;
    if (s == "segments_dir")      return DataSourceType::SegmentsDir;
    if (s == "segment")           return DataSourceType::Segment;
    if (s == "normal_grid")       return DataSourceType::NormalGrid;
    if (s == "normal_dir_volume") return DataSourceType::NormalDirVolume;
    if (s == "sync_dir")          return DataSourceType::SyncDir;
    throw std::runtime_error("unknown data source type: " + s);
}

std::string sync_direction_to_string(SyncDirection d)
{
    switch (d) {
        case SyncDirection::Pull: return "pull";
        case SyncDirection::Push: return "push";
        case SyncDirection::Both: return "both";
    }
    throw std::runtime_error("unknown SyncDirection");
}

SyncDirection sync_direction_from_string(const std::string& s)
{
    if (s == "pull") return SyncDirection::Pull;
    if (s == "push") return SyncDirection::Push;
    if (s == "both") return SyncDirection::Both;
    throw std::runtime_error("unknown sync direction: " + s);
}

LocationKind infer_location_kind(const std::string& location)
{
    static constexpr const char* kRemoteSchemes[] = {
        "http://", "https://", "s3://"
    };
    for (const char* scheme : kRemoteSchemes) {
        const std::string s(scheme);
        if (location.size() >= s.size()
            && location.compare(0, s.size(), s) == 0)
        {
            return LocationKind::Remote;
        }
    }
    return LocationKind::Local;
}

std::string location_kind_to_string(LocationKind k)
{
    switch (k) {
        case LocationKind::Local:  return "local";
        case LocationKind::Remote: return "remote";
    }
    throw std::runtime_error("unknown LocationKind");
}

LocationKind location_kind_from_string(const std::string& s)
{
    if (s == "local")  return LocationKind::Local;
    if (s == "remote") return LocationKind::Remote;
    throw std::runtime_error("unknown location kind: " + s);
}

utils::Json Project::to_json() const
{
    auto root = utils::Json::object();
    root["schema"] = kSchema;
    root["schema_version"] = kSchemaVersion;
    root["name"] = name;
    root["version"] = version;
    if (!description.empty()) {
        root["description"] = description;
    }

    auto sources = utils::Json::array();
    for (const auto& ds : data_sources) {
        if (ds.imported) continue;  // linked-in, don't serialise
        auto s = utils::Json::object();
        s["id"] = ds.id;
        s["type"] = data_source_type_to_string(ds.type);
        s["location"] = ds.location;
        s["location_kind"] = location_kind_to_string(ds.location_kind);
        s["recursive"] = ds.recursive;
        s["track_changes"] = ds.track_changes;
        if (!ds.enabled) s["enabled"] = false;
        if (!ds.children.empty()) {
            auto arr = utils::Json::array();
            for (const auto& c : ds.children) arr.push_back(c);
            s["children"] = arr;
        }
        if (!ds.tags.empty()) {
            auto arr = utils::Json::array();
            for (const auto& t : ds.tags) arr.push_back(t);
            s["tags"] = arr;
        }
        if (ds.type == DataSourceType::SyncDir) {
            s["sync_remote"] = ds.sync_remote;
            s["sync_direction"] = sync_direction_to_string(ds.sync_direction);
        }
        sources.push_back(s);
    }
    root["data_sources"] = sources;

    if (!groups.empty()) {
        auto g = utils::Json::array();
        for (const auto& gr : groups) {
            auto o = utils::Json::object();
            o["id"] = gr.id;
            o["name"] = gr.name;
            auto arr = utils::Json::array();
            for (const auto& s : gr.source_ids) arr.push_back(s);
            o["source_ids"] = arr;
            g.push_back(o);
        }
        root["groups"] = g;
    }

    if (!linked_projects.empty()) {
        auto lp = utils::Json::array();
        for (const auto& link : linked_projects) {
            auto o = utils::Json::object();
            o["path"] = link.path;
            o["read_only"] = link.read_only;
            if (!link.id_prefix.empty()) o["id_prefix"] = link.id_prefix;
            lp.push_back(o);
        }
        root["linked_projects"] = lp;
    }

    if (origin) {
        auto o = utils::Json::object();
        o["kind"] = origin->kind;
        o["root"] = origin->root.string();
        if (!origin->legacy_config.is_null()) {
            o["legacy_config"] = origin->legacy_config;
        }
        root["origin"] = o;
    }

    if (!active_segments_source_id.empty()) {
        root["active_segments_source_id"] = active_segments_source_id;
    }
    if (!output_segments_source_id.empty()) {
        root["output_segments_source_id"] = output_segments_source_id;
    }

    return root;
}

Project Project::from_json(const utils::Json& j)
{
    Project p;
    if (j.contains("name"))        p.name = j["name"].get_string();
    if (j.contains("version"))     p.version = j["version"].get_int();
    if (j.contains("description")) p.description = j["description"].get_string();

    if (j.contains("data_sources")) {
        const auto& sources = j["data_sources"];
        for (std::size_t i = 0; i < sources.size(); ++i) {
            const auto& s = sources[i];
            DataSource ds;
            ds.id = s["id"].get_string();
            ds.type = data_source_type_from_string(s["type"].get_string());
            ds.location = s["location"].get_string();
            ds.location_kind = location_kind_from_string(
                s.value("location_kind", std::string("local")));
            ds.recursive = s.value("recursive", true);
            ds.track_changes = s.value("track_changes", false);
            ds.enabled = s.value("enabled", true);
            if (s.contains("children")) {
                ds.children = s["children"].get_string_array();
            }
            if (s.contains("tags")) {
                ds.tags = s["tags"].get_string_array();
            }
            if (s.contains("sync_remote")) {
                ds.sync_remote = s["sync_remote"].get_string();
            }
            if (s.contains("sync_direction")) {
                ds.sync_direction = sync_direction_from_string(
                    s["sync_direction"].get_string());
            }
            p.data_sources.push_back(std::move(ds));
        }
    }

    if (j.contains("groups")) {
        const auto& gs = j["groups"];
        for (std::size_t i = 0; i < gs.size(); ++i) {
            const auto& g = gs[i];
            Group gr;
            gr.id = g["id"].get_string();
            gr.name = g.value("name", gr.id);
            if (g.contains("source_ids")) {
                gr.source_ids = g["source_ids"].get_string_array();
            }
            p.groups.push_back(std::move(gr));
        }
    }

    if (j.contains("linked_projects")) {
        const auto& ls = j["linked_projects"];
        for (std::size_t i = 0; i < ls.size(); ++i) {
            const auto& l = ls[i];
            LinkedProject lp;
            lp.path = l["path"].get_string();
            lp.read_only = l.value("read_only", true);
            lp.id_prefix = l.value("id_prefix", std::string());
            p.linked_projects.push_back(std::move(lp));
        }
    }

    if (j.contains("origin")) {
        const auto& o = j["origin"];
        Origin origin;
        origin.kind = o.value("kind", std::string("project"));
        origin.root = o.value("root", std::string());
        if (o.contains("legacy_config")) {
            origin.legacy_config = o["legacy_config"];
        }
        p.origin = origin;
    }

    p.active_segments_source_id =
        j.value("active_segments_source_id", std::string());
    p.output_segments_source_id =
        j.value("output_segments_source_id", std::string());

    return p;
}

void Project::save_to_file(const std::filesystem::path& path) const
{
    auto j = to_json();
    auto tmp = path;
    tmp += ".tmp";
    {
        std::ofstream o(tmp);
        if (!o) {
            throw std::runtime_error("failed to open for write: " + tmp.string());
        }
        o << j.dump(2) << '\n';
    }
    std::filesystem::rename(tmp, path);
}

Project Project::load_from_file(const std::filesystem::path& path)
{
    auto j = vc::json::load_json_file(path);
    auto p = Project::from_json(j);
    p.set_path(path);
    // Inline linked projects so callers get the complete set of data
    // sources. Imported entries are marked so they won't be re-serialised
    // if the user saves this project back out.
    p.merge_linked_projects();
    return p;
}

bool Project::looks_like_volpkg(const std::filesystem::path& dir)
{
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
        return false;
    }
    const auto cfg = dir / "config.json";
    if (!std::filesystem::exists(cfg, ec)) {
        return false;
    }
    try {
        auto j = vc::json::load_json_file(cfg);
        return j.contains("name") && j.contains("version");
    } catch (...) {
        return false;
    }
}

Project Project::from_volpkg(const std::filesystem::path& volpkg_root)
{
    Project p;
    const auto cfg = volpkg_root / "config.json";
    auto j = vc::json::load_json_file(cfg);
    vc::json::require_fields(j, {"name", "version"}, cfg.string());

    p.name = j["name"].get_string();
    p.version = j["version"].get_int();

    Origin origin;
    origin.kind = "volpkg";
    origin.root = volpkg_root;
    // Stash everything except name/version so fields like voxel size,
    // scroll id, material, etc. survive a project round-trip.
    auto legacy = j;
    legacy.erase("name");
    legacy.erase("version");
    if (!legacy.empty()) {
        origin.legacy_config = legacy;
    }
    p.origin = origin;

    // Volumes directory - always expose if it exists.
    {
        const auto vdir = volpkg_root / "volumes";
        if (std::filesystem::exists(vdir)) {
            DataSource ds;
            ds.id = "volumes";
            ds.type = DataSourceType::VolumesDir;
            ds.location = "volumes";
            ds.location_kind = LocationKind::Local;
            ds.recursive = true;
            ds.track_changes = true;
            p.data_sources.push_back(std::move(ds));
        }
    }

    // Segment directories - add each that exists.
    for (const auto& name : kLegacySegmentsDirs()) {
        const auto d = volpkg_root / name;
        if (!std::filesystem::exists(d) || !std::filesystem::is_directory(d)) {
            continue;
        }
        DataSource ds;
        ds.id = name;
        ds.type = DataSourceType::SegmentsDir;
        ds.location = name;
        ds.location_kind = LocationKind::Local;
        ds.recursive = true;
        ds.track_changes = true;
        p.data_sources.push_back(std::move(ds));
    }

    // Default active/output segments source: first segments_dir.
    for (const auto& ds : p.data_sources) {
        if (ds.type == DataSourceType::SegmentsDir) {
            p.active_segments_source_id = ds.id;
            p.output_segments_source_id = ds.id;
            break;
        }
    }

    return p;
}

bool Project::is_volpkg_compatible() const
{
    return origin.has_value() && origin->kind == "volpkg"
        && !origin->root.empty();
}

const DataSource* Project::find_source(const std::string& id) const
{
    for (const auto& ds : data_sources) {
        if (ds.id == id) return &ds;
    }
    return nullptr;
}

DataSource* Project::find_source(const std::string& id)
{
    for (auto& ds : data_sources) {
        if (ds.id == id) return &ds;
    }
    return nullptr;
}

std::vector<const DataSource*>
Project::sources_of_type(DataSourceType t) const
{
    std::vector<const DataSource*> out;
    for (const auto& ds : data_sources) {
        if (ds.type == t) out.push_back(&ds);
    }
    return out;
}

std::vector<const DataSource*>
Project::sources_with_tag(const std::string& tag) const
{
    std::vector<const DataSource*> out;
    for (const auto& ds : data_sources) {
        for (const auto& t : ds.tags) {
            if (t == tag) { out.push_back(&ds); break; }
        }
    }
    return out;
}

void Project::merge_linked_projects()
{
    if (linked_projects.empty()) return;

    auto base_dir = [&]() -> std::filesystem::path {
        if (is_volpkg_compatible()) return origin->root;
        if (!path_.empty()) return path_.parent_path();
        return std::filesystem::current_path();
    };

    // BFS through the link graph, avoiding cycles via canonicalised paths.
    std::set<std::string> seen;
    std::vector<std::pair<LinkedProject, std::filesystem::path>> todo;
    for (const auto& lp : linked_projects) {
        std::filesystem::path p = lp.path;
        if (!p.is_absolute()) p = base_dir() / p;
        todo.emplace_back(lp, p);
    }

    while (!todo.empty()) {
        auto [link, linkPath] = todo.back();
        todo.pop_back();

        std::error_code ec;
        auto canon = std::filesystem::weakly_canonical(linkPath, ec).string();
        if (ec || canon.empty()) canon = linkPath.string();
        if (seen.count(canon)) continue;
        seen.insert(canon);

        Project other;
        try { other = Project::load_from_file(linkPath); }
        catch (const std::exception&) { continue; }

        for (auto ds : other.data_sources) {
            if (ds.imported) continue;   // don't chain imports
            if (!link.id_prefix.empty()) ds.id = link.id_prefix + ds.id;
            ds.imported = true;
            data_sources.push_back(std::move(ds));
        }
        // Chase the link graph one hop further.
        for (const auto& sub : other.linked_projects) {
            std::filesystem::path p = sub.path;
            if (!p.is_absolute()) p = linkPath.parent_path() / sub.path;
            todo.emplace_back(sub, p);
        }
    }
}

std::vector<const DataSource*>
Project::sources_in_group(const std::string& group_id) const
{
    std::vector<const DataSource*> out;
    for (const auto& g : groups) {
        if (g.id != group_id) continue;
        for (const auto& sid : g.source_ids) {
            if (const auto* ds = find_source(sid)) out.push_back(ds);
        }
        break;
    }
    return out;
}

std::filesystem::path Project::resolve_local(const DataSource& ds) const
{
    if (ds.location_kind != LocationKind::Local) {
        throw std::runtime_error(
            "resolve_local called on remote source: " + ds.id);
    }

    std::filesystem::path loc{ds.location};
    if (loc.is_absolute()) {
        return loc;
    }

    // Prefer volpkg/project origin root for relative paths.
    if (origin && !origin->root.empty()) {
        return origin->root / loc;
    }
    if (!path_.empty()) {
        return path_.parent_path() / loc;
    }
    return std::filesystem::absolute(loc);
}

std::string Project::remote_url(const DataSource& ds) const
{
    if (ds.location_kind != LocationKind::Remote) {
        throw std::runtime_error(
            "remote_url called on local source: " + ds.id);
    }
    return ds.location;
}

std::string Project::resolve_location(const DataSource& ds) const
{
    if (ds.location_kind == LocationKind::Remote) {
        return ds.location;
    }
    return resolve_local(ds).string();
}

std::filesystem::path
Project::resolve_segments_dir(const std::string& id_or_name) const
{
    if (const auto* ds = find_source(id_or_name);
        ds && ds->type == DataSourceType::SegmentsDir
           && ds->location_kind == LocationKind::Local)
    {
        return resolve_local(*ds);
    }
    if (is_volpkg_compatible()) {
        return origin->root / id_or_name;
    }
    throw std::runtime_error(
        "cannot resolve segments dir '" + id_or_name + "'");
}

std::filesystem::path Project::resolve_active_segments_dir() const
{
    std::string id = active_segments_source_id;
    if (id.empty()) {
        for (const auto& ds : data_sources) {
            if (ds.type == DataSourceType::SegmentsDir) {
                id = ds.id;
                break;
            }
        }
    }
    if (!id.empty()) {
        return resolve_segments_dir(id);
    }
    if (is_volpkg_compatible()) {
        return origin->root / "paths";
    }
    throw std::runtime_error("no active segments dir in project");
}

std::filesystem::path Project::resolve_output_segments_dir() const
{
    if (!output_segments_source_id.empty()) {
        return resolve_segments_dir(output_segments_source_id);
    }
    return resolve_active_segments_dir();
}

std::filesystem::path Project::resolve_volumes_dir() const
{
    for (const auto& ds : data_sources) {
        if (ds.type == DataSourceType::VolumesDir
            && ds.location_kind == LocationKind::Local)
        {
            return resolve_local(ds);
        }
    }
    if (is_volpkg_compatible()) {
        return origin->root / "volumes";
    }
    throw std::runtime_error("no volumes dir in project");
}

std::filesystem::path
Project::support_file_path(const std::string& filename) const
{
    if (is_volpkg_compatible()) {
        return origin->root / filename;
    }
    if (!path_.empty()) {
        return path_.parent_path() / filename;
    }
    throw std::runtime_error(
        "cannot resolve support file '" + filename + "': project is unsaved");
}

std::vector<std::filesystem::path> Project::all_segments_dirs() const
{
    std::vector<std::filesystem::path> out;
    for (const auto& ds : data_sources) {
        if (ds.type == DataSourceType::SegmentsDir
            && ds.location_kind == LocationKind::Local)
        {
            out.push_back(resolve_local(ds));
        }
    }
    if (out.empty() && is_volpkg_compatible()) {
        for (const char* name : {"paths", "traces", "export"}) {
            auto p = origin->root / name;
            if (std::filesystem::exists(p)
                && std::filesystem::is_directory(p))
            {
                out.push_back(std::move(p));
            }
        }
    }
    return out;
}

} // namespace vc
