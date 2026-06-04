#include "vc/lasagna/Manifest.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace vc::lasagna {
namespace {

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

std::vector<std::string> parseChannels(const nlohmann::json& value)
{
    if (!value.is_array()) {
        throw std::runtime_error("Lasagna channel group 'channels' must be an array");
    }

    std::vector<std::string> channels;
    channels.reserve(value.size());
    for (const auto& channel : value) {
        if (!channel.is_string()) {
            throw std::runtime_error("Lasagna channel names must be strings");
        }
        channels.push_back(channel.get<std::string>());
    }
    return channels;
}

std::vector<LasagnaChannelGroup> parseGroups(
    const nlohmann::json& root,
    const std::filesystem::path& baseDirectory)
{
    if (!root.contains("groups")) {
        return {};
    }
    const auto& groupsJson = root.at("groups");
    if (!groupsJson.is_object()) {
        throw std::runtime_error("Lasagna manifest 'groups' must be a JSON object");
    }

    std::vector<LasagnaChannelGroup> groups;
    groups.reserve(groupsJson.size());
    for (auto it = groupsJson.begin(); it != groupsJson.end(); ++it) {
        if (!it.value().is_object()) {
            throw std::runtime_error("Lasagna channel group '" + it.key() + "' must be an object");
        }
        const auto& groupJson = it.value();
        if (!groupJson.contains("zarr") || !groupJson.at("zarr").is_string()) {
            throw std::runtime_error("Lasagna channel group '" + it.key() + "' is missing string field 'zarr'");
        }
        if (!groupJson.contains("channels")) {
            throw std::runtime_error("Lasagna channel group '" + it.key() + "' is missing field 'channels'");
        }

        LasagnaChannelGroup group;
        group.name = it.key();
        group.zarrPath = resolvePath(baseDirectory, groupJson.at("zarr").get<std::string>());
        group.scaledown = groupJson.value("scaledown", 0);
        if (group.scaledown < 0 || group.scaledown > 30) {
            throw std::runtime_error("Lasagna channel group '" + it.key() + "' has invalid scaledown");
        }
        group.channels = parseChannels(groupJson.at("channels"));
        groups.push_back(std::move(group));
    }
    return groups;
}

} // namespace

int LasagnaChannelGroup::scaleFactor() const noexcept
{
    return 1 << scaledown;
}

bool LasagnaChannelGroup::hasChannel(std::string_view channel) const noexcept
{
    return channelIndex(channel).has_value();
}

std::optional<size_t> LasagnaChannelGroup::channelIndex(std::string_view channel) const noexcept
{
    const auto it = std::find(channels.begin(), channels.end(), channel);
    if (it == channels.end()) {
        return std::nullopt;
    }
    return static_cast<size_t>(std::distance(channels.begin(), it));
}

bool LasagnaDatasetManifest::hasNormalSource() const noexcept
{
    return groupForChannel("grad_mag") != nullptr &&
           groupForChannel("nx") != nullptr &&
           groupForChannel("ny") != nullptr;
}

const LasagnaChannelGroup* LasagnaDatasetManifest::groupForChannel(std::string_view channel) const noexcept
{
    for (const auto& group : groups) {
        if (group.hasChannel(channel)) {
            return &group;
        }
    }
    return nullptr;
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
    manifest.version = root.value("version", 0);
    manifest.sourceToBase = root.value("source_to_base", 1.0);
    if (!(manifest.sourceToBase > 0.0)) {
        throw std::runtime_error("Lasagna manifest source_to_base must be positive");
    }
    if (root.contains("init_shell_dir")) {
        if (!root.at("init_shell_dir").is_string()) {
            throw std::runtime_error("Lasagna manifest init_shell_dir must be a string");
        }
        manifest.initShellDir = resolvePath(manifest.baseDirectory,
                                            root.at("init_shell_dir").get<std::string>());
    }
    manifest.groups = parseGroups(root, manifest.baseDirectory);

    if (manifest.hasNormalSource()) {
        manifest.normalPath = manifest.manifestPath;
        manifest.normalSourceKind = NormalSourceKind::DenseZarr;
        manifest.normalSourceKey = "groups.grad_mag_nx_ny";
    }

    return manifest;
}

std::string toString(NormalSourceKind kind)
{
    switch (kind) {
    case NormalSourceKind::None:
        return "none";
    case NormalSourceKind::DenseZarr:
        return "dense_zarr";
    }
    return "unknown";
}

} // namespace vc::lasagna
