#include "formats/GeometryInput.hpp"
#include "vc_object_store/objects/FiberJson.hpp"
#include "vc_object_store/Uuid.hpp"
#include <cmath>
#include <map>
#include <set>
#include <tuple>
namespace vc::objects::formats {
namespace {
// A fiber file without a UUID (as VC3D wrote before fibers had identities) is
// named from its file name, so every import of that file, and every branch
// that names it by file, resolves to the same fiber.
std::string fileUuid(const nlohmann::json& root, const std::filesystem::path& path)
{
    if (root.contains("uuid")) return util::canonicalUuid(root.at("uuid").get<std::string>());
    return util::nameUuid("vc3d_fiber/" + path.filename().string());
}

Prepared fiberDocument(nlohmann::json root, const std::filesystem::path& path)
{
    root["uuid"] = fileUuid(root, path);
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
    // Versions 1 and 3 name branch targets only by file, and declare each link from both ends with
    // separately computed directions. Targets resolve through the peer file's UUID; the second
    // declaration of a link must match the first and is dropped.
    std::map<fs::path, std::string> fileUuids;
    std::map<std::tuple<std::string, int64_t, std::string, int64_t>, Json> fileLinks;
    const auto sameDirection = [](const Json& left, const Json& right) {
        if (left.is_null() || right.is_null()) return left.is_null() && right.is_null();
        for (size_t axis = 0; axis < 3; ++axis)
            if (std::abs(left.at(axis).get<double>() - right.at(axis).get<double>()) > 1e-9) return false;
        return true;
    };
    for (size_t i = 0; i < batch.size(); ++i) {
        const auto source = fs::path(batch[i].object.storageRef);
        const auto self = batch[i].object.uuid;
        // Copy before appending peers, which can reallocate the batch.
        const auto metadata = batch[i].source;
        const bool byFile = metadata.value("version", 0) < 4;
        for (const auto* key : {"branches", "adjacent_branches"}) {
            auto kept = Json::array();
            for (auto branch : metadata.value(key, Json::array())) {
                const auto peerPath = [&] {
                    return source.parent_path() / fs::path(branch.at("branch_file").get<std::string>()).filename();
                };
                if (byFile) {
                    const auto peer = peerPath();
                    if (!fileUuids.contains(peer))
                        fileUuids[peer] = fileUuid(readJson(peer), peer);
                    branch["target_uuid"] = fileUuids.at(peer);
                }
                const auto target = util::canonicalUuid(branch.at("target_uuid").get<std::string>());
                if (byFile) {
                    const auto own = branch.at("control_point_index").get<int64_t>();
                    const auto other = branch.at("branch_control_point_index").get<int64_t>();
                    if (const auto first = fileLinks.find({target, other, self, own}); first != fileLinks.end()) {
                        if (!sameDirection(first->second.value("control_point_direction", Json()), branch.value("branch_control_point_direction", Json())) ||
                            !sameDirection(first->second.value("branch_control_point_direction", Json()), branch.value("control_point_direction", Json())))
                            throw std::runtime_error("conflicting reciprocal fiber branch declarations: " + self + " <-> " + target);
                        continue;
                    }
                    fileLinks.emplace(std::tuple{self, own, target, other}, branch);
                }
                kept.push_back(branch);
                if (existing.contains(target) || incoming.contains(target)) continue;
                const auto peer = readJson(peerPath());
                if (fileUuid(peer, peerPath()) != target)
                    throw std::runtime_error("branch target UUID does not match its artifact");
                add(peer, peerPath());
            }
            if (byFile && metadata.contains(key)) batch[i].source[key] = std::move(kept);
        }
    }
    return batch;
}
}
