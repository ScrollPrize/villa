#include "vc/lasagna/ProjectVolumes.hpp"

#include <map>
#include <mutex>

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include <vc_reader/zarr.hpp>

#include <algorithm>
#include <cmath>
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
    if (group.isRemote())
        return Volume::NewFromUrl(location, {}, group.discoverAwsCredentials);
    return Volume::New(std::filesystem::path(location));
}

// The power-of-two factor mapping `from` into `to`, or nullopt when the two
// shapes do not identify exactly one level: the helper throws, and here a pair
// that cannot be related is simply not a pair.
std::optional<double> dyadicFactor(const std::array<std::size_t, 3>& from,
                                   const std::array<std::size_t, 3>& to)
{
    try {
        return dyadicCoordinateScaleBetweenShapes(from, to);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

}  // namespace

namespace
{
// Base shapes per manifest location: a manifest is immutable, so each is read
// once however often the project's volumes are walked.
std::optional<std::array<std::size_t, 3>> baseShapeOfTags(const std::vector<std::string>& tags,
                                                          const std::filesystem::path& remoteCacheRoot)
{
    static std::mutex mutex;
    static std::map<std::string, std::optional<std::array<std::size_t, 3>>> cache;
    const std::string prefix(kLasagnaVolumeManifestTagPrefix);
    for (const auto& tag : tags) {
        if (tag.rfind(prefix, 0) != 0) continue;
        const auto location = tag.substr(prefix.size());
        if (location.empty()) continue;
        const std::lock_guard<std::mutex> lock(mutex);
        if (const auto cached = cache.find(location); cached != cache.end()) return cached->second;
        std::optional<std::array<std::size_t, 3>> shape;
        try {
            LasagnaDatasetOpenOptions options;
            options.remoteCacheRoot = remoteCacheRoot;
            shape = LasagnaDataset::openLocation(location, options).manifest().baseShapeZYX;
        } catch (const std::exception&) {
            shape = std::nullopt;  // unreadable manifest: no base, no edge
        }
        cache.emplace(location, shape);
        return shape;
    }
    return std::nullopt;
}
}  // namespace

std::vector<vcr::catalog::Registration> lasagnaProjectRegistrations(const VolumePkg& pkg,
                                                                   const std::filesystem::path& remoteCacheRoot)
{
    std::vector<LasagnaRegistrationVolume> volumes;
    for (const auto& id : pkg.volumeIDs()) {
        std::shared_ptr<Volume> volume;
        try {
            volume = pkg.volume(id);
        } catch (const std::exception&) {
            continue;
        }
        if (!volume) continue;
        const auto voxels = vc::project::annotationVolume(&pkg, *volume);
        const auto shape = volume->shape();
        if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0) continue;
        LasagnaRegistrationVolume entry;
        entry.coordinateId = voxels.volume;
        entry.coordinateScale = voxels.scale;
        entry.shapeZYX = {std::size_t(shape[0]), std::size_t(shape[1]), std::size_t(shape[2])};
        entry.lasagnaBaseShapeZYX = baseShapeOfTags(pkg.volumeTags(id), remoteCacheRoot);
        volumes.push_back(std::move(entry));
    }
    return lasagnaBaseShapeRegistrations(volumes);
}

std::vector<vcr::catalog::Registration> lasagnaBaseShapeRegistrations(
    std::span<const LasagnaRegistrationVolume> volumes)
{
    std::vector<vcr::catalog::Registration> out;
    for (const auto& group : volumes) {
        if (!group.lasagnaBaseShapeZYX || group.coordinateId.empty() || !(group.coordinateScale > 0))
            continue;
        for (const auto& candidate : volumes) {
            if (&candidate == &group || candidate.coordinateId.empty() ||
                candidate.coordinateId == group.coordinateId || !(candidate.coordinateScale > 0))
                continue;
            // One edge per pair: two groups of a dataset would otherwise state
            // the same relation twice, once in each direction.
            if (candidate.lasagnaBaseShapeZYX && !(group.coordinateId < candidate.coordinateId)) continue;
            // The candidate belongs to this dataset when its shape is a level
            // of the base shape the manifest declares. Every such volume is
            // related, not just one: a project holds the base volume and any
            // number of derived levels, and they are all the same frame.
            if (!dyadicFactor(candidate.shapeZYX, *group.lasagnaBaseShapeZYX)) continue;
            const auto working = dyadicFactor(group.shapeZYX, candidate.shapeZYX);
            if (!working) continue;
            // toVolume composes scale(targetScale) * matrix * scale(1/sourceScale),
            // so the level-0 matrix that leaves `working` between the two
            // working frames is the factor carried by both volumes' scales.
            const double matrix = *working * group.coordinateScale / candidate.coordinateScale;
            if (!std::isfinite(matrix) || matrix <= 0) continue;
            out.push_back({"lasagna-base-shape:" + group.coordinateId + ":" + candidate.coordinateId,
                           std::to_string(matrix), group.coordinateId, candidate.coordinateId,
                           vcr::catalog::Affine::scale(matrix), "Lasagna manifest base shape"});
        }
    }
    return out;
}

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
        // Dataset-wide capability: the normal heads may be in separate groups.
        if (manifest.groupForChannel("sheet_nx") && manifest.groupForChannel("sheet_ny"))
            addUniqueTag(tags, std::string(vc::project::kPredictedFiberNormalsTag));

        if (existing != prepared.end()) {
            if (existing->volume->channelCount() && existing->channels != group.channels)
                throw std::runtime_error("conflicting display channel names for " + location);
            for (auto& tag : tags)
                addUniqueTag(existing->tags, std::move(tag));
            continue;
        }

        auto volume = openRegularVolumeForGroup(group);
        const auto channels = volume->channelCount() ? group.channels : std::vector<std::string>{};
        if (volume->channelCount() && *volume->channelCount() != static_cast<std::int64_t>(channels.size()))
            throw std::runtime_error("manifest channel count does not match " + location);
        prepared.push_back({
            location,
            std::move(tags),
            std::move(volume),
            channels,
        });
    }
    return prepared;
}

}  // namespace vc::lasagna
