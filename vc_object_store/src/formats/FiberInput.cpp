#include "formats/GeometryInput.hpp"
#include "vc_object_store/objects/FiberJson.hpp"
#include "vc_object_store/Uuid.hpp"
#include <map>
#include <set>
namespace vc::objects::formats {
namespace {
// A fiber file's key is its stem, which is also how version-3 branches name their targets.
nlohmann::json fiberIdentity(nlohmann::json root, const std::filesystem::path& path,
    const std::optional<std::string>& uuidNamespace)
{
    if (!uuidNamespace && !root.contains("uuid"))
        root["uuid"] = util::nameUuid("vc3d_fiber/" + path.filename().string());
    else
        deriveSourceUuid(root, "fiber", path.stem().string(), uuidNamespace);
    return root;
}
Prepared fiberDocument(const nlohmann::json& root, const std::filesystem::path& path,
    const std::optional<std::string>& uuidNamespace = std::nullopt)
{
    auto result = decodeFiberJson(fiberIdentity(root, path, uuidNamespace).dump());
    if (result.object.name.empty()) result.object.name = path.stem().string();
    result.object.storageKind = "file";
    result.object.storageRef = std::filesystem::absolute(path).lexically_normal().string();
    return result;
}
}
Prepared fiber(const std::filesystem::path& path) { return fiberDocument(readJson(path), path); }

std::vector<Prepared> fibers(const std::filesystem::path& path, const std::vector<std::string>& available,
    const SourceIngestOptions& options)
{
    const auto& uuidNamespace = options.uuidNamespace;
    namespace fs = std::filesystem;
    using Json = nlohmann::json;
    std::set<std::string> existing, incoming;
    for (const auto& uuid : available) existing.insert(util::canonicalUuid(uuid));
    std::vector<Prepared> batch;
    const auto add = [&](const Json& document, const fs::path& source) {
        auto decoded = fiberDocument(document, source, uuidNamespace);
        if (!incoming.insert(decoded.object.uuid).second) throw std::runtime_error("duplicate fiber UUID in import");
        batch.push_back(std::move(decoded));
    };
    const auto read = [&](const fs::path& source) {
        const auto root = readJson(source);
        if (root.is_object() && root.value("type", "") == "vc3d_fiber") add(root, source);
        else {
            if (!root.is_object() || root.value("type", "") != "vc3d_fiber_collection" || root.value("version", 0) != 1 ||
                !root.at("point_collections").is_array()) throw std::runtime_error("expected fiber JSON or fiber collection");
            for (const auto& document : root.at("point_collections"))
                add(document, source.parent_path() / (util::canonicalUuid(document.at("uuid").get<std::string>()) + ".json"));
        }
    };
    if (fs::is_directory(path)) {
        std::vector<fs::path> files;
        for (const auto& entry : fs::directory_iterator(path))
            if (entry.is_regular_file() && entry.path().extension() == ".json") files.push_back(entry.path());
        std::sort(files.begin(), files.end());
        for (const auto& file : files) {
            const auto root = readJson(file);
            if (root.is_object() && root.value("type", "") == "vc3d_fiber") add(root, file);
        }
    } else read(path);
    if (batch.empty()) throw std::runtime_error("no fibers were found");
    // Versions 1 and 3 name branch targets only by file; they resolve through the peer file's UUID.
    // Reciprocal declarations are reconciled by the relationship importer.
    std::map<fs::path, std::string> fileUuids;
    for (size_t i = 0; i < batch.size(); ++i) {
        const auto source = fs::path(batch[i].object.storageRef);
        // Copy before appending peers, which can reallocate the batch.
        const auto metadata = batch[i].source;
        const bool byFile = metadata.value("version", 0) < 4;
        auto dropped = Json::array();
        for (const auto* key : {"branches", "adjacent_branches"}) {
            auto kept = Json::array();
            for (auto branch : metadata.value(key, Json::array())) {
                const auto peerPath = [&] {
                    return source.parent_path() / fs::path(branch.at("branch_file").get<std::string>()).filename();
                };
                // Only a declaration that must be resolved from an absent file can dangle.
                const bool known = !byFile && branch.contains("target_uuid") &&
                    (existing.contains(util::canonicalUuid(branch.at("target_uuid").get<std::string>())) ||
                     incoming.contains(util::canonicalUuid(branch.at("target_uuid").get<std::string>())));
                if (!known && branch.contains("branch_file") && !fs::exists(peerPath())) {
                    if (!options.dropDanglingBranches)
                        throw std::runtime_error("branch target file is absent: " + peerPath().string());
                    auto record = branch;
                    record["declaration"] = key;
                    dropped.push_back(std::move(record));
                    continue;
                }
                if (byFile) {
                    const auto peer = peerPath();
                    if (!fileUuids.contains(peer))
                        fileUuids[peer] = util::canonicalUuid(
                            fiberIdentity(readJson(peer), peer, uuidNamespace).at("uuid").get<std::string>());
                    branch["target_uuid"] = fileUuids.at(peer);
                }
                const auto target = util::canonicalUuid(branch.at("target_uuid").get<std::string>());
                kept.push_back(branch);
                if (existing.contains(target) || incoming.contains(target)) continue;
                const auto peer = readJson(peerPath());
                if (util::canonicalUuid(fiberIdentity(peer, peerPath(), uuidNamespace).at("uuid").get<std::string>()) != target)
                    throw std::runtime_error("branch target UUID does not match its artifact");
                add(peer, peerPath());
            }
            if ((byFile || !dropped.empty()) && metadata.contains(key)) batch[i].source[key] = std::move(kept);
        }
        if (!dropped.empty()) batch[i].source["dropped_branches"] = std::move(dropped);
    }
    return batch;
}
}
