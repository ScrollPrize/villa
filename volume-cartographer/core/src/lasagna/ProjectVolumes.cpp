#include "vc/lasagna/ProjectVolumes.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include <vc_reader/zarr.hpp>

#include <algorithm>
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

// The volume a group attaches as: its array's group when the array is a numbered
// pyramid level, otherwise the array itself.
std::string attachmentLocationForGroup(const LasagnaChannelGroup& group)
{
    const std::string location = group.isRemote() || group.zarrPath.empty()
        ? lasagnaGroupSourceLocation(group)
        : std::filesystem::absolute(group.zarrPath).lexically_normal().string();
    return vcr::zarr::levelGroupLocation(location).value_or(location);
}

std::shared_ptr<Volume> openRegularVolumeForGroup(
    const LasagnaChannelGroup& group)
{
    const std::string location = attachmentLocationForGroup(group);
    if (group.isRemote()) {
        const auto auth = group.discoverAwsCredentials
            ? group.remoteAuth
            : vcr::Auth{};
        return Volume::NewFromUrl(
            location, auth, {}, group.discoverAwsCredentials);
    }
    return Volume::New(std::filesystem::path(location));
}

}  // namespace

std::vector<std::string> lasagnaProjectVolumeTags(
    const LasagnaChannelGroup& group,
    std::string_view manifestLocation)
{
    std::vector<std::string> tags;
    addUniqueTag(
        tags, std::string(kLasagnaVolumeManifestTagPrefix) +
                  std::string(manifestLocation));
    addUniqueTag(tags, std::string(kLasagnaVolumeGroupTagPrefix) + group.name);
    if (group.isRemote() && !group.discoverAwsCredentials) {
        addUniqueTag(
            tags, std::string(vc::project::kAnonymousRemoteAuthTag));
    }
    return tags;
}

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

        auto tags = lasagnaProjectVolumeTags(group, manifestLocation);

        if (existing != prepared.end()) {
            for (auto& tag : tags)
                addUniqueTag(existing->tags, std::move(tag));
            continue;
        }

        prepared.push_back({
            location,
            std::move(tags),
            openRegularVolumeForGroup(group),
        });
    }
    return prepared;
}

}  // namespace vc::lasagna
