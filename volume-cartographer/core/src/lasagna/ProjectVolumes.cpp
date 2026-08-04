#include "vc/lasagna/ProjectVolumes.hpp"

#include "vc/core/types/Volume.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace vc::lasagna
{
namespace
{

void addUniqueTag(std::vector<std::string>& tags, std::string tag)
{
    if (tag.empty())
        return;
    if (std::find(tags.begin(), tags.end(), tag) == tags.end())
        tags.push_back(std::move(tag));
}

bool isDigits(std::string_view value)
{
    return !value.empty() &&
        std::all_of(value.begin(), value.end(), [](unsigned char c) {
            return std::isdigit(c) != 0;
        });
}

bool isLocalZarrRoot(const std::filesystem::path& path)
{
    return path.extension() == ".zarr" ||
           std::filesystem::exists(path / ".zarray") ||
           std::filesystem::exists(path / ".zgroup") ||
           std::filesystem::exists(path / ".zattrs") ||
           std::filesystem::exists(path / "zarr.json");
}

std::filesystem::path localAttachmentRoot(const LasagnaChannelGroup& group)
{
    std::filesystem::path path = group.zarrPath.empty()
        ? std::filesystem::path(lasagnaGroupSourceLocation(group))
        : group.zarrPath;
    path = std::filesystem::absolute(path).lexically_normal();
    if (isDigits(path.filename().string())) {
        const auto parent = path.parent_path();
        if (!parent.empty() && isLocalZarrRoot(parent))
            return parent;
    }
    return path;
}

std::string remoteAttachmentRoot(std::string location)
{
    const auto fragment = location.find('#');
    const auto query = location.find('?');
    const auto suffixStart = std::min(
        fragment == std::string::npos ? location.size() : fragment,
        query == std::string::npos ? location.size() : query);
    std::string path = location.substr(0, suffixStart);
    const std::string suffix = location.substr(suffixStart);

    while (!path.empty() && path.back() == '/')
        path.pop_back();
    const auto slash = path.rfind('/');
    if (slash == std::string::npos)
        return location;

    const auto leaf = path.substr(slash + 1);
    const auto parent = path.substr(0, slash);
    if (isDigits(leaf) && parent.size() >= 5 &&
        parent.compare(parent.size() - 5, 5, ".zarr") == 0) {
        return parent + suffix;
    }
    return location;
}

std::string attachmentLocationForGroup(const LasagnaChannelGroup& group)
{
    if (group.isRemote())
        return remoteAttachmentRoot(lasagnaGroupSourceLocation(group));
    return localAttachmentRoot(group).string();
}

std::shared_ptr<Volume> openRegularVolumeForGroup(
    const LasagnaChannelGroup& group)
{
    const std::string location = attachmentLocationForGroup(group);
    if (group.isRemote()) {
        return Volume::NewFromUrl(location, group.remoteCacheRoot,
                                  group.remoteAuth);
    }
    return Volume::New(std::filesystem::path(location));
}

}  // namespace

std::vector<PreparedLasagnaProjectVolume> prepareLasagnaProjectVolumes(
    const LasagnaDataset& dataset,
    std::string manifestLocation)
{
    const auto& manifest = dataset.manifest();
    if (manifestLocation.empty()) {
        manifestLocation = manifest.manifestLocation.empty()
            ? manifest.manifestPath.string()
            : manifest.manifestLocation;
    }

    std::vector<PreparedLasagnaProjectVolume> prepared;
    for (const auto& group : manifest.groups) {
        const std::string location = attachmentLocationForGroup(group);
        auto existing = std::find_if(
            prepared.begin(), prepared.end(), [&](const auto& candidate) {
                return candidate.location == location;
            });

        const std::string manifestTag =
            std::string(kLasagnaVolumeManifestTagPrefix) + manifestLocation;
        const std::string groupTag =
            std::string(kLasagnaVolumeGroupTagPrefix) + group.name;

        if (existing != prepared.end()) {
            addUniqueTag(existing->tags, manifestTag);
            addUniqueTag(existing->tags, groupTag);
            continue;
        }

        std::vector<std::string> tags;
        addUniqueTag(tags, manifestTag);
        addUniqueTag(tags, groupTag);
        prepared.push_back({
            location,
            std::move(tags),
            openRegularVolumeForGroup(group),
        });
    }
    return prepared;
}

}  // namespace vc::lasagna
