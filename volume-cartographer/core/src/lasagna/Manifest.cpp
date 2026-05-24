#include "vc/lasagna/Manifest.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace vc::lasagna {
namespace {

struct PathCandidate {
    std::string key;
    std::string value;
    NormalSourceKind kind = NormalSourceKind::None;
};

std::filesystem::path manifestBaseDir(const std::filesystem::path& manifestPath)
{
    if (manifestPath.empty()) {
        return std::filesystem::current_path();
    }
    const auto parent = manifestPath.parent_path();
    if (parent.empty()) {
        return std::filesystem::current_path();
    }
    return std::filesystem::absolute(parent).lexically_normal();
}

std::filesystem::path resolvePath(
    const std::filesystem::path& baseDirectory,
    const std::string& rawPath)
{
    std::filesystem::path path(rawPath);
    if (path.is_relative()) {
        path = baseDirectory / path;
    }
    return std::filesystem::absolute(path).lexically_normal();
}

std::optional<std::string> stringAtPath(
    const nlohmann::json& root,
    std::initializer_list<const char*> keys)
{
    const nlohmann::json* current = &root;
    for (const char* key : keys) {
        if (!current->is_object() || !current->contains(key)) {
            return std::nullopt;
        }
        current = &current->at(key);
    }

    if (current->is_string()) {
        return current->get<std::string>();
    }
    if (current->is_object()) {
        for (const char* key : {"path", "location", "src", "source", "url"}) {
            if (current->contains(key) && current->at(key).is_string()) {
                return current->at(key).get<std::string>();
            }
        }
    }
    return std::nullopt;
}

void addPathCandidate(
    std::vector<PathCandidate>& candidates,
    const nlohmann::json& root,
    std::initializer_list<const char*> keys,
    NormalSourceKind kind)
{
    if (auto value = stringAtPath(root, keys)) {
        std::string key;
        for (const char* part : keys) {
            if (!key.empty()) {
                key += ".";
            }
            key += part;
        }
        candidates.push_back({std::move(key), std::move(*value), kind});
    }
}

NormalSourceKind kindFromObjectOrDefault(
    const nlohmann::json& root,
    std::initializer_list<const char*> keys,
    NormalSourceKind fallback)
{
    const nlohmann::json* current = &root;
    for (const char* key : keys) {
        if (!current->is_object() || !current->contains(key)) {
            return fallback;
        }
        current = &current->at(key);
    }
    if (!current->is_object()) {
        return fallback;
    }
    for (const char* typeKey : {"type", "kind", "format"}) {
        if (!current->contains(typeKey) || !current->at(typeKey).is_string()) {
            continue;
        }
        const std::string value = current->at(typeKey).get<std::string>();
        if (value == "normal_grid" || value == "normal_grids" || value == "grid") {
            return NormalSourceKind::NormalGrid;
        }
        if (value == "zarr" || value == "dense_zarr" || value == "normals_zarr") {
            return NormalSourceKind::DenseZarr;
        }
    }
    return fallback;
}

std::optional<PathCandidate> findVolumePath(const nlohmann::json& root)
{
    std::vector<PathCandidate> candidates;
    addPathCandidate(candidates, root, {"volume_path"}, NormalSourceKind::None);
    addPathCandidate(candidates, root, {"volume"}, NormalSourceKind::None);
    addPathCandidate(candidates, root, {"zarr_path"}, NormalSourceKind::None);
    addPathCandidate(candidates, root, {"data_path"}, NormalSourceKind::None);
    addPathCandidate(candidates, root, {"input_path"}, NormalSourceKind::None);
    addPathCandidate(candidates, root, {"paths", "volume"}, NormalSourceKind::None);
    addPathCandidate(candidates, root, {"paths", "data"}, NormalSourceKind::None);
    addPathCandidate(candidates, root, {"data", "volume"}, NormalSourceKind::None);
    if (candidates.empty()) {
        return std::nullopt;
    }
    return candidates.front();
}

std::optional<PathCandidate> findNormalPath(const nlohmann::json& root)
{
    std::vector<PathCandidate> candidates;
    addPathCandidate(candidates, root, {"normal_grid_path"}, NormalSourceKind::NormalGrid);
    addPathCandidate(candidates, root, {"normal_grid"}, NormalSourceKind::NormalGrid);
    addPathCandidate(candidates, root, {"normal_grids"}, NormalSourceKind::NormalGrid);
    addPathCandidate(candidates, root, {"paths", "normal_grid"}, NormalSourceKind::NormalGrid);
    addPathCandidate(candidates, root, {"paths", "normal_grids"}, NormalSourceKind::NormalGrid);
    addPathCandidate(candidates, root, {"normal_path"}, NormalSourceKind::DenseZarr);
    addPathCandidate(candidates, root, {"normals_path"}, NormalSourceKind::DenseZarr);
    addPathCandidate(candidates, root, {"normal_zarr"}, NormalSourceKind::DenseZarr);
    addPathCandidate(candidates, root, {"normals_zarr"}, NormalSourceKind::DenseZarr);
    addPathCandidate(candidates, root, {"gt_normals"}, NormalSourceKind::DenseZarr);
    addPathCandidate(candidates, root, {"normals"}, NormalSourceKind::DenseZarr);
    addPathCandidate(candidates, root, {"normal_dataset"}, NormalSourceKind::DenseZarr);
    addPathCandidate(candidates, root, {"paths", "normals"}, NormalSourceKind::DenseZarr);
    addPathCandidate(candidates, root, {"data", "normals"}, NormalSourceKind::DenseZarr);

    if (candidates.empty()) {
        return std::nullopt;
    }

    PathCandidate candidate = candidates.front();
    if (candidate.key == "normal_grid" || candidate.key == "normal_grids" ||
        candidate.key == "normals" || candidate.key == "gt_normals" ||
        candidate.key == "normal_dataset") {
        candidate.kind = kindFromObjectOrDefault(root, {candidate.key.c_str()}, candidate.kind);
    }
    return candidate;
}

} // namespace

bool LasagnaDatasetManifest::hasNormalSource() const noexcept
{
    return normalPath.has_value() && normalSourceKind != NormalSourceKind::None;
}

LasagnaDatasetManifest LasagnaDatasetManifest::parseFile(const std::filesystem::path& manifestPath)
{
    std::ifstream input(manifestPath);
    if (!input) {
        throw std::runtime_error("Failed to open Lasagna manifest: " + manifestPath.string());
    }

    nlohmann::json root;
    input >> root;
    LasagnaDatasetManifest manifest = parseText(root.dump(), manifestPath);
    manifest.raw = std::move(root);
    return manifest;
}

LasagnaDatasetManifest LasagnaDatasetManifest::parseText(
    std::string_view jsonText,
    const std::filesystem::path& manifestPath)
{
    nlohmann::json root = nlohmann::json::parse(jsonText);
    if (!root.is_object()) {
        throw std::runtime_error("Lasagna manifest root must be a JSON object");
    }

    LasagnaDatasetManifest manifest;
    manifest.manifestPath = manifestPath.empty()
        ? std::filesystem::path{}
        : std::filesystem::absolute(manifestPath).lexically_normal();
    manifest.baseDirectory = manifestBaseDir(manifestPath);
    manifest.raw = root;

    if (auto volume = findVolumePath(root)) {
        manifest.volumePath = resolvePath(manifest.baseDirectory, volume->value);
    }

    if (auto normal = findNormalPath(root)) {
        manifest.normalPath = resolvePath(manifest.baseDirectory, normal->value);
        manifest.normalSourceKind = normal->kind;
        manifest.normalSourceKey = normal->key;
    }

    return manifest;
}

std::string toString(NormalSourceKind kind)
{
    switch (kind) {
    case NormalSourceKind::None:
        return "none";
    case NormalSourceKind::NormalGrid:
        return "normal_grid";
    case NormalSourceKind::DenseZarr:
        return "dense_zarr";
    }
    return "unknown";
}

} // namespace vc::lasagna
