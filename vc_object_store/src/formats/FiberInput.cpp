#include "formats/GeometryInput.hpp"
#include "vc_object_store/objects/FiberJson.hpp"
#include "vc_object_store/Uuid.hpp"
#include <set>
namespace vc::objects::formats {
namespace {
Prepared fiberDocument(const nlohmann::json& root, const std::filesystem::path& path)
{
    auto result = decodeFiberJson(root.dump());
    if (result.object.name.empty()) result.object.name = path.stem().string();
    result.object.storageKind = "file";
    result.object.storageRef = std::filesystem::absolute(path).lexically_normal().string();
    return result;
}
}
Prepared fiber(const std::filesystem::path& path) { return fiberDocument(readJson(path), path); }

std::vector<Prepared> fibers(const std::filesystem::path& path, const std::vector<std::string>& available)
{
    namespace fs = std::filesystem;
    using Json = nlohmann::json;
    std::set<std::string> existing, incoming;
    for (const auto& uuid : available) existing.insert(util::canonicalUuid(uuid));
    std::vector<Prepared> batch;
    const auto add = [&](const Json& document, const fs::path& source) {
        auto decoded = fiberDocument(document, source);
        if (!incoming.insert(decoded.object.uuid).second) throw std::runtime_error("duplicate fiber UUID in import");
        batch.push_back(std::move(decoded));
    };
    const auto read = [&](const fs::path& source) {
        const auto root = readJson(source);
        if (root.is_object() && root.value("type", "") == "vc3d_fiber") add(root, source);
        else {
            if (!root.is_object() || root.value("type", "") != "vc3d_fiber_collection" || root.value("version", 0) != 1 ||
                !root.at("point_collections").is_array()) throw std::runtime_error("expected current fiber JSON or fiber collection");
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
    for (size_t i = 0; i < batch.size(); ++i) {
        const auto source = fs::path(batch[i].object.storageRef);
        // Copy before appending peers, which can reallocate the batch.
        const auto metadata = batch[i].source;
        for (const auto* key : {"branches", "adjacent_branches"})
            for (const auto& branch : metadata.value(key, Json::array())) {
                const auto target = util::canonicalUuid(branch.at("target_uuid").get<std::string>());
                if (existing.contains(target) || incoming.contains(target)) continue;
                const auto peerPath = source.parent_path() / fs::path(branch.at("branch_file").get<std::string>()).filename();
                const auto peer = readJson(peerPath);
                if (util::canonicalUuid(peer.at("uuid").get<std::string>()) != target)
                    throw std::runtime_error("branch target UUID does not match its artifact");
                add(peer, peerPath);
            }
    }
    return batch;
}
}
