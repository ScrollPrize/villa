#include "internal/RecordSql.hpp"

namespace vc::objects {
using namespace detail;
std::optional<std::filesystem::path> ObjectStore::resolveStoragePath(
    const Txn& txn, const std::string& objectUuid,
    const std::optional<std::filesystem::path>& datasetRoot) const
{
    namespace fs = std::filesystem;
    const auto object = objectByUuid(txn, objectUuid);
    if (!object || (object->storageKind != "file" && object->storageKind != "sqlite") ||
        (object->kind != "patch" && object->kind != "fiber" && object->kind != "point_collection"))
        return std::nullopt;

    const bool patch = object->kind == "patch";
    const auto uuid = parseUuid(object->uuid);
    const auto matches = [&](const fs::path& candidate) {
        const auto metadata = patch ? candidate / "meta.json" : candidate;
        if (!patch && metadata.extension() != ".json")
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
            if (!patch && !json.contains("control_points") && !json.contains("line_points"))
                return false;
            const auto sourceUuid = json["uuid"].get<std::string>();
            return util::isUuid(sourceUuid) && parseUuid(sourceUuid) == uuid;
        } catch (const nlohmann::json::exception&) {
            return false;
        }
    };

    const auto root = fs::absolute(datasetRoot.value_or(fs::absolute(impl_->path).parent_path())).lexically_normal();
    if (!object->storageRef.empty() && object->storageRef.find("://") == std::string::npos) {
        const auto hint = (root / object->storageRef).lexically_normal();
        if (matches(hint))
            return hint;
    }
    if (!fs::exists(root))
        return std::nullopt;
    // The caller may supply the patch directory itself as the search root.
    if (patch && matches(root))
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
            // Patch geometry can be large; only its metadata is relevant.
            it.disable_recursion_pending();
            if (!patch)
                continue;
            candidate = entry.path();
        } else {
            if (patch || entry.path().filename() == "meta.json" || !entry.is_regular_file())
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
