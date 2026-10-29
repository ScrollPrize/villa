#include "vc/core/types/ProjectDocument.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include <vc_object_store/Uuid.hpp>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <stdexcept>

namespace vc::project {
namespace {
bool remote(const std::string& location) { return location.find("://") != std::string::npos; }
void checkExtension(const std::filesystem::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
    if (extension != ".vcproj")
        throw std::invalid_argument("Open a .vcproj project file. To use a .vcobj store, create a project and attach it.");
}
bool hasWritePermission(const std::filesystem::path& path) {
    std::error_code error;
    const auto permissions = std::filesystem::status(path, error).permissions();
    using P = std::filesystem::perms;
    return !error && (permissions & (P::owner_write | P::group_write | P::others_write)) != P::none;
}
}
const std::vector<std::string>& ProjectDocument::kinds() {
    static const std::vector<std::string> value{"surface", "fiber", "point_collection", "umbilicus", "volume_transform"};
    return value;
}
void ProjectDocument::validate(const nlohmann::json& data) {
    if (!data.is_object() || data.value("format", std::string{}) != "vcproj" || data.at("version") != 1)
        throw std::invalid_argument("Unsupported VC3D project format or version");
    (void)data.at("name").get<std::string>();
    const auto& stores = data.at("stores");
    if (!stores.is_array() || stores.empty()) throw std::invalid_argument("A project must reference at least one store");
    std::set<std::string> ids, paths;
    for (const auto& store : stores) {
        const auto id = store.at("id").get<std::string>();
        const auto location = store.at("location").get<std::string>();
        if (id.empty() || location.empty() || remote(location) || !ids.insert(id).second || !paths.insert(location).second)
            throw std::invalid_argument("Invalid or duplicate project store reference");
        (void)store.value("read_only", false);
    }
    const auto& routes = data.at("selections");
    if (!routes.is_object()) throw std::invalid_argument("Project selections must be an object");
    for (const auto& kind : kinds()) {
        const auto& id = routes.at(kind);
        if (!id.is_null() && (!id.is_string() || !ids.contains(id.get<std::string>())))
            throw std::invalid_argument("Selection references an unknown store: " + kind);
    }
    for (const auto* key : {"volumes", "normal_grids", "lasagna_datasets"}) {
        if (!data.at(key).is_array()) throw std::invalid_argument(std::string(key) + " must be an array");
        for (const auto& entry : data.at(key)) {
            if (entry.at("location").get<std::string>().empty()) throw std::invalid_argument("Empty resource location");
            (void)entry.at("tags").get<std::vector<std::string>>();
        }
    }
    if (!data.at("view").is_object()) throw std::invalid_argument("Project view must be an object");
    if (data.contains("open_data")) {
        (void)data.at("open_data").at("sample").get<std::string>();
        if (!ids.contains(data.at("open_data").at("store").get<std::string>()))
            throw std::invalid_argument("Open Data references an unknown store");
    }
}
ProjectDocument ProjectDocument::load(const std::filesystem::path& path) {
    checkExtension(path);
    ProjectDocument result;
    result.path_ = std::filesystem::canonical(path);
    std::ifstream input(result.path_);
    if (!input) throw std::runtime_error("Cannot open project " + path.string());
    input >> result.data_;
    validate(result.data_);
    return result;
}
ProjectDocument ProjectDocument::create(const std::filesystem::path& path, const std::string& name,
                                        const std::filesystem::path& store, bool readOnly) {
    checkExtension(path);
    if (std::filesystem::exists(path)) throw std::runtime_error(path.string() + " already exists");
    ProjectDocument result;
    result.path_ = std::filesystem::absolute(path).lexically_normal();
    const auto id = vc::objects::util::newUuid();
    nlohmann::json selections = nlohmann::json::object();
    for (const auto& kind : kinds()) selections[kind] = id;
    std::filesystem::create_directories(result.path_.parent_path());
    result.replace({{"format", "vcproj"}, {"version", 1}, {"name", name},
        {"stores", {{{"id", id}, {"location", result.location(store)}, {"read_only", readOnly}}}},
        {"selections", selections}, {"volumes", nlohmann::json::array()},
        {"normal_grids", nlohmann::json::array()}, {"lasagna_datasets", nlohmann::json::array()},
        {"view", nlohmann::json::object()}});
    return result;
}
bool ProjectDocument::writable() const {
    return (!std::filesystem::exists(path_) || hasWritePermission(path_)) && hasWritePermission(path_.parent_path());
}
void ProjectDocument::replace(nlohmann::json data) {
    validate(data);
    if (data == data_) return;
    if (!writable()) throw std::runtime_error("Project is read-only: " + path_.string());
    vc::core::util::atomicWriteString(path_, data.dump(2) + "\n");
    data_ = std::move(data);
}
std::filesystem::path ProjectDocument::resolve(const std::string& location) const {
    const std::filesystem::path path(location);
    return (path.is_absolute() ? path : path_.parent_path() / path).lexically_normal();
}
std::string ProjectDocument::location(const std::filesystem::path& path) const {
    const auto absolute = std::filesystem::absolute(path).lexically_normal();
    const auto relative = absolute.lexically_relative(path_.parent_path());
    return (relative.empty() ? absolute : relative).generic_string();
}
ProjectDocument ProjectDocument::saveAs(const std::filesystem::path& destination) const {
    checkExtension(destination);
    if (std::filesystem::exists(destination)) throw std::runtime_error(destination.string() + " already exists");
    ProjectDocument copy;
    copy.path_ = std::filesystem::absolute(destination).lexically_normal();
    auto data = data_;
    const auto rebase = [&](const std::string& location) {
        return remote(location) ? location : copy.location(resolve(location));
    };
    for (const auto* key : {"stores", "volumes", "normal_grids", "lasagna_datasets"})
        for (auto& entry : data[key]) {
            entry["location"] = rebase(entry.at("location"));
            if (entry.contains("tags")) for (auto& tag : entry["tags"]) {
                const auto value = tag.get<std::string>();
                for (const std::string prefix : {"vc-lasagna-derived:", "vc-lasagna-manifest:"})
                    if (value.starts_with(prefix)) tag = prefix + rebase(value.substr(prefix.size()));
            }
        }
    for (const auto* key : {"selected_lasagna_dataset", "selected_fiber_inference_dataset"})
        if (data["view"].contains(key) && !data["view"][key].get<std::string>().empty())
            data["view"][key] = rebase(data["view"][key]);
    std::filesystem::create_directories(copy.path_.parent_path());
    copy.replace(std::move(data));
    return copy;
}
}
