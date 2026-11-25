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
// A track's dataset: its dataset_ref below the root its owning store resolves
// source locators from.
std::filesystem::path trackDatasetPath(const std::filesystem::path& root, const std::string& datasetRef)
{
    return (std::filesystem::absolute(root).lexically_normal() / datasetRef).lexically_normal();
}
}

std::filesystem::path detail::storeDirectory(sqlite3* source)
{
    namespace fs = std::filesystem;
    fs::path ownerPath = sqlite3_db_filename(source, "main");
    if (isPackedDatabasePath(ownerPath)) ownerPath = packedContainerPath(ownerPath);
    return fs::absolute(ownerPath).parent_path();
}

std::string trackDatasetIdentity(const std::filesystem::path& resolved)
{
    return std::filesystem::weakly_canonical(resolved).generic_string();
}

void detail::TrackDatasetResolver::append(const std::optional<ResolvedObjectVersion>& resolved)
{
    int32_t index = -1;
    if (resolved && resolved->revision.record.kind == "track" && resolved->revision.record.track
        && !resolved->revision.record.track->datasetRef.empty()) {
        const std::pair key{resolved->source, resolved->revision.record.track->datasetRef};
        auto found = resolved_.find(key);
        if (found == resolved_.end()) {
            const auto identity = trackDatasetIdentity(trackDatasetPath(storeDirectory(key.first), key.second));
            auto known = identities_.find(identity);
            if (known == identities_.end()) {
                known = identities_.emplace(identity, int32_t(out_.datasets.size())).first;
                out_.datasets.push_back(identity);
            }
            found = resolved_.emplace(key, known->second).first;
        }
        index = found->second;
    }
    out_.objects.push_back(index);
}

void detail::TrackDatasetResolver::finish()
{
    if (out_.datasets.empty()) out_.objects.clear();
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
