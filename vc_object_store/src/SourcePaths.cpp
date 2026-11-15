#include "internal/RecordSql.hpp"
#include "internal/RecordResolver.hpp"
#include "vc_object_store/objects/PackedStore.hpp"

namespace vc::objects {
using namespace detail;
void ObjectStore::validateExportDestination(const Txn& txn, const std::filesystem::path& destination) const
{
    namespace fs = std::filesystem;
    const auto target = fs::weakly_canonical(fs::absolute(destination));
    const auto ancestor = [](fs::path child, const fs::path& parent) {
        for (; !child.empty(); child = child.parent_path()) {
            if (child == parent) return true;
            if (child == child.root_path()) break;
        }
        return false;
    };
    for (auto* source : querySources(static_cast<sqlite3*>(connection(txn)))) {
        fs::path file = sqlite3_db_filename(source, "main");
        if (isPackedDatabasePath(file)) file = packedContainerPath(file);
        file = fs::weakly_canonical(file);
        if (ancestor(file, target)) throw std::runtime_error("export destination contains a backing object database");
        for (const auto* suffix : {"", "-wal", "-shm", "-journal", "-owner", ".artifacts"}) {
            const fs::path protectedPath = fs::weakly_canonical(file.string() + suffix);
            if (target == protectedPath || (fs::exists(target) && fs::exists(protectedPath) && fs::equivalent(target, protectedPath)) ||
                (std::string_view(suffix) == ".artifacts" && (ancestor(target, protectedPath) || ancestor(protectedPath, target))))
                throw std::runtime_error("cannot export over backing database files or artifacts");
        }
    }
}

namespace {
// Where a store's relative source locators start: the directory of its file.
std::filesystem::path storeDirectory(sqlite3* source)
{
    namespace fs = std::filesystem;
    fs::path ownerPath = sqlite3_db_filename(source, "main");
    if (isPackedDatabasePath(ownerPath)) ownerPath = packedContainerPath(ownerPath);
    return fs::absolute(ownerPath).parent_path();
}
// A track's dataset: its dataset_ref below the root its owning store resolves
// source locators from.
std::filesystem::path trackDatasetPath(const std::filesystem::path& root, const std::string& datasetRef)
{
    return (std::filesystem::absolute(root).lexically_normal() / datasetRef).lexically_normal();
}
}

std::string trackDatasetIdentity(const std::filesystem::path& resolved)
{
    return std::filesystem::weakly_canonical(resolved).generic_string();
}

std::map<std::string, std::string> ObjectStore::trackDatasets(const Txn& txn, const std::vector<ObjectRevision>& revisions) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    std::map<std::string, std::filesystem::path> roots;  // by owning store
    bool listed = false;
    // One filesystem resolution per (owning store, dataset_ref), not per track.
    std::map<std::pair<std::string, std::string>, std::string> resolved;
    std::map<std::string, std::string> result;
    for (const auto& revision : revisions) {
        const auto& record = revision.record;
        if (record.kind != "track") continue;
        if (!record.track || record.track->datasetRef.empty())
            throw std::runtime_error("track has no dataset_ref: " + record.uuid);
        const std::pair key{revision.storeUuid, record.track->datasetRef};
        auto found = resolved.find(key);
        if (found == resolved.end()) {
            if (!listed) {
                for (auto* source : querySources(db)) roots.emplace(storeUuidText(source), storeDirectory(source));
                listed = true;
            }
            const auto root = roots.find(revision.storeUuid);
            if (root == roots.end())
                throw std::runtime_error("track's owning store is not open: " + record.uuid);
            found = resolved.emplace(key, trackDatasetIdentity(trackDatasetPath(root->second, key.second))).first;
        }
        result[record.uuid] = found->second;
    }
    return result;
}

std::optional<std::filesystem::path> ObjectStore::resolveStoragePath(
    const Txn& txn, const std::string& objectUuid,
    const std::optional<std::filesystem::path>& datasetRoot) const
{
    namespace fs = std::filesystem;
    const auto resolved = resolveObject(static_cast<sqlite3*>(connection(txn)), objectUuid);
    if (!resolved) return std::nullopt;
    const auto* object = &resolved->record;
    const auto root = fs::absolute(datasetRoot.value_or(storeDirectory(resolved->source))).lexically_normal();
    if (object->track && !object->track->datasetRef.empty())
        return trackDatasetPath(root, object->track->datasetRef);
    if ((object->storageKind != "file" && object->storageKind != "sqlite") ||
        (object->kind != "surface" && object->kind != "fiber" && object->kind != "point_collection"))
        return std::nullopt;

    const bool surface = object->kind == "surface";
    const auto uuid = parseUuid(object->uuid);
    const auto matches = [&](const fs::path& candidate) {
        const auto metadata = surface ? candidate / "meta.json" : candidate;
        if (!surface && metadata.extension() != ".json")
            return false;
        std::ifstream input(metadata);
        if (!input)
            return false;
        try {
            const auto json = nlohmann::json::parse(input);
            if (object->kind == "point_collection") {
                if (!json.is_object() || json.value("vc_pointcollections_json_version", "") != "1" ||
                    !json.contains("collections") || !json.at("collections").is_object()) return false;
                for (const auto& collection : json.at("collections")) {
                    const auto sourceUuid = collection.value("uuid", "");
                    if (util::isUuid(sourceUuid) && parseUuid(sourceUuid) == uuid) return true;
                }
                return false;
            }
            if (!json.is_object() || !json.contains("uuid") || !json["uuid"].is_string())
                return false;
            if (!surface && !json.contains("control_points") && !json.contains("line_points"))
                return false;
            const auto sourceUuid = json["uuid"].get<std::string>();
            return util::isUuid(sourceUuid) && parseUuid(sourceUuid) == uuid;
        } catch (const nlohmann::json::exception&) {
            return false;
        }
    };

    if (!object->storageRef.empty() && object->storageRef.find("://") == std::string::npos) {
        const auto hint = (root / object->storageRef).lexically_normal();
        if (matches(hint))
            return hint;
    }
    if (!fs::exists(root))
        return std::nullopt;
    // The caller may supply the surface directory itself as the search root.
    if (surface && matches(root))
        return root;

    std::optional<fs::path> found;
    for (fs::recursive_directory_iterator it(root), end; it != end; ++it) {
        const auto& entry = *it;
        fs::path candidate;
        if (entry.is_directory()) {
            const auto name = entry.path().filename().string();
            if (name.front() == '.' || entry.path().extension() == ".zarr") {
                it.disable_recursion_pending();
                continue;
            }
            if (!fs::is_regular_file(entry.path() / "meta.json"))
                continue;
            // Surface geometry can be large; only its metadata is relevant.
            it.disable_recursion_pending();
            if (!surface)
                continue;
            candidate = entry.path();
        } else {
            if (surface || entry.path().filename() == "meta.json" || !entry.is_regular_file())
                continue;
            candidate = entry.path();
        }
        if (!matches(candidate))
            continue;
        if (found) {
            // Multiple symlinks/hard links to the same object are harmless.
            if (fs::equivalent(*found, candidate))
                continue;
            throw std::runtime_error("ambiguous storage for UUID " + object->uuid + ": " +
                                     found->string() + " and " + candidate.string());
        }
        found = candidate;
    }
    return found;
}

} // namespace vc::objects
