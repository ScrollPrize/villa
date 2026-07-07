#include "OpenDataSampleProject.hpp"

#include "OpenDataNormalGrids.hpp"
#include "OpenDataSegmentCache.hpp"

#include "vc/core/types/VolumePkg.hpp"

#include <algorithm>
#include <cctype>
#include <exception>
#include <filesystem>
#include <map>
#include <system_error>

namespace vc3d::opendata {
namespace {

std::string lowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string trimTrailingSlashes(std::string value)
{
    while (!value.empty() && value.back() == '/') {
        value.pop_back();
    }
    return value;
}

bool containsInsensitive(const std::string& haystack, const std::string& needle)
{
    return lowerCopy(haystack).find(lowerCopy(needle)) != std::string::npos;
}

std::string artifactUrl(const OpenDataArtifact& artifact)
{
    return trimTrailingSlashes(artifact.resolvedUrl.empty()
                                  ? artifact.sourcePath
                                  : artifact.resolvedUrl);
}

bool isSupportedRemoteZarr(const OpenDataVolume& volume,
                           const OpenDataArtifact& artifact,
                           const std::string& url)
{
    if (url.empty()) {
        return false;
    }
    const std::string loweredUrl = lowerCopy(url);
    if (loweredUrl.rfind("http://", 0) != 0 &&
        loweredUrl.rfind("https://", 0) != 0 &&
        loweredUrl.rfind("s3://", 0) != 0) {
        return false;
    }
    if (loweredUrl.size() >= 5 &&
        loweredUrl.substr(loweredUrl.size() - 5) == ".zarr") {
        return true;
    }
    if (containsInsensitive(artifact.type, "zarr")) {
        return true;
    }
    return artifact.type.empty() && containsInsensitive(volume.dataFormat, "zarr");
}

bool jsonStringEqualsInsensitive(const nlohmann::json& obj,
                                 const char* key,
                                 const std::string& expected)
{
    if (!obj.is_object()) {
        return false;
    }
    const auto it = obj.find(key);
    if (it == obj.end() || !it->is_string()) {
        return false;
    }
    return lowerCopy(it->get<std::string>()) == expected;
}

std::vector<std::string> volumeTags(const OpenDataSample& sample,
                                    const OpenDataVolume& volume,
                                    const OpenDataArtifact& artifact)
{
    std::vector<std::string> tags;
    auto addUnique = [&tags](std::string tag) {
        if (!tag.empty() &&
            std::find(tags.begin(), tags.end(), tag) == tags.end()) {
            tags.push_back(std::move(tag));
        }
    };

    if (jsonStringEqualsInsensitive(artifact.properties, "representation", "normal3d") ||
        jsonStringEqualsInsensitive(volume.properties, "representation", "normal3d") ||
        containsInsensitive(artifact.type, "normal3d") ||
        containsInsensitive(volume.suffix, "normal3d")) {
        addUnique("normal3d");
    }
    if (containsInsensitive(artifact.type, "prediction") ||
        containsInsensitive(artifact.type, "pred")) {
        addUnique("prediction");
    }
    if (containsInsensitive(artifact.type, "surface") &&
        containsInsensitive(artifact.type, "prediction")) {
        addUnique("surface-prediction");
    }
    if (containsInsensitive(artifact.type, "ink")) {
        addUnique("ink-detection");
    }
    if (containsInsensitive(artifact.type, "ink") &&
        containsInsensitive(artifact.type, "3d")) {
        addUnique("ink-detection-3d");
    }
    if (!volume.id.empty()) {
        addUnique("vc-open-data-volume-id:" + volume.id);
    }
    if (!sample.id.empty()) {
        addUnique(std::string(kOpenDataSampleIdTagPrefix) + sample.id);
    }
    if (const auto normalGridsUrl = normalGridsArtifactUrl(volume); !normalGridsUrl.empty()) {
        addUnique(std::string(kOpenDataNormalGridsTagPrefix) + normalGridsUrl);
    }
    if (volume.pixelSizeUm && *volume.pixelSizeUm > 0.0) {
        addUnique("vc-open-data-voxel-size-um:" + std::to_string(*volume.pixelSizeUm));
    }

    return tags;
}

bool hasVolumeEntry(const VolumePkg& pkg, const std::string& location)
{
    const auto& entries = pkg.volumeEntries();
    return std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
        return entry.location == location;
    });
}

std::string volumeLabel(const OpenDataVolume& volume)
{
    return volume.id.empty() ? std::string("<unnamed volume>") : volume.id;
}

std::string volumeArtifactLabel(const OpenDataVolume& volume,
                                const OpenDataArtifact& artifact)
{
    auto label = volumeLabel(volume);
    if (!artifact.type.empty()) {
        label += " (" + artifact.type + ")";
    }
    return label;
}

std::string safePathComponent(std::string value)
{
    for (char& c : value) {
        const auto uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '-' && c != '_' && c != '.') {
            c = '_';
        }
    }
    while (!value.empty() && (value.front() == '.' || value.front() == '_')) {
        value.erase(value.begin());
    }
    return value.empty() ? std::string("unnamed") : value;
}

std::string segmentSourcePreferredVolumeId(
    const OpenDataSample& sample,
    const std::vector<std::string>& supportedVolumeIds)
{
    if (supportedVolumeIds.empty() || sample.segments.empty()) {
        return {};
    }

    std::map<std::string, std::size_t> segmentCountsByVolumeId;
    for (const auto& segment : sample.segments) {
        if (!segment.originalVolumeId.empty()) {
            ++segmentCountsByVolumeId[segment.originalVolumeId];
        }
    }

    std::string preferredId;
    std::size_t preferredCount = 0;
    for (const auto& volumeId : supportedVolumeIds) {
        const auto it = segmentCountsByVolumeId.find(volumeId);
        const std::size_t count =
            it == segmentCountsByVolumeId.end() ? 0 : it->second;
        if (count > preferredCount) {
            preferredId = volumeId;
            preferredCount = count;
        }
    }

    return preferredCount == 0 ? std::string{} : preferredId;
}

std::filesystem::path sampleProjectCachePath(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample)
{
    return remoteCacheRoot / "open_data" / "projects" /
           (safePathComponent(sample.id.empty() ? "sample" : sample.id) + ".volpkg.json");
}

bool isNonEmptyFile(const std::filesystem::path& path)
{
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec) &&
           std::filesystem::file_size(path, ec) > 0 &&
           !ec;
}

int openDataSegmentEntryCount(const VolumePkg& pkg,
                              const std::filesystem::path& remoteCacheRoot,
                              const OpenDataSample& sample)
{
    if (remoteCacheRoot.empty() || sample.tifxyzSegmentCount() == 0) {
        return 0;
    }

    const auto sampleSegmentsRoot = remoteCacheRoot / "open_data" / "segments" /
                                    safePathComponent(sample.id.empty() ? "sample" : sample.id);
    const auto sampleSegmentsRootString = sampleSegmentsRoot.string();
    const auto sampleSegmentsPrefix =
        sampleSegmentsRootString + std::string(1, std::filesystem::path::preferred_separator);
    int count = 0;
    for (const auto& entry : pkg.segmentEntries()) {
        std::error_code ec;
        if (!std::filesystem::is_directory(entry.location, ec) || ec) {
            continue;
        }
        const bool openDataSegmentEntry =
            std::find(entry.tags.begin(), entry.tags.end(), "open-data") != entry.tags.end() &&
            std::any_of(entry.tags.begin(), entry.tags.end(), [](const std::string& tag) {
                return tag.rfind("vc-open-data-source-volume-id:", 0) == 0 ||
                       tag.rfind("vc-open-data-target-volume-id:", 0) == 0;
            });
        if (openDataSegmentEntry &&
            entry.location.rfind(sampleSegmentsPrefix, 0) == 0 &&
            entry.location.find(std::string(1, std::filesystem::path::preferred_separator),
                                sampleSegmentsPrefix.size()) == std::string::npos) {
            ++count;
        }
    }
    return count;
}

} // namespace

std::shared_ptr<VolumePkg> createOpenDataSampleProject(
    const OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    OpenDataSampleProjectResult* resultOut,
    const OpenDataSampleProgressCallback& progressCallback)
{
    auto result = OpenDataSampleProjectResult{};
    std::shared_ptr<VolumePkg> pkg;
    bool loadedCachedProject = false;
    const auto cachedProjectPath = remoteCacheRoot.empty()
        ? std::filesystem::path{}
        : sampleProjectCachePath(remoteCacheRoot, sample);

    if (!cachedProjectPath.empty() && isNonEmptyFile(cachedProjectPath)) {
        try {
            vc::project::LoadOptions opts;
            opts.remoteCacheRoot = remoteCacheRoot;
            pkg = VolumePkg::load(
                cachedProjectPath,
                opts);
            loadedCachedProject = true;
            result.messages.push_back("Loaded cached sample project: " +
                                      cachedProjectPath.string());
        } catch (const std::exception& e) {
            result.messages.push_back("Ignored cached sample project " +
                                      cachedProjectPath.string() + ": " + e.what());
        } catch (...) {
            result.messages.push_back("Ignored cached sample project " +
                                      cachedProjectPath.string() + ": unknown error.");
        }
    }

    if (!pkg) {
        pkg = VolumePkg::newEmpty();
    }
    pkg->setName(sample.id.empty() ? "Open Data Sample" : sample.id);
    if (!remoteCacheRoot.empty()) {
        pkg->setRemoteCacheRoot(remoteCacheRoot);
    }

    if (!remoteCacheRoot.empty()) {
        const int attachedNormalGrids =
            attachOpenDataNormalGrids(*pkg, sample, remoteCacheRoot);
        if (attachedNormalGrids > 0) {
            result.messages.push_back("Attached " + std::to_string(attachedNormalGrids) +
                                      " streaming normal grid store(s).");
        }
    }

    auto attachResult = attachOpenDataSampleVolumes(*pkg, sample);
    result.supportedVolumes = attachResult.supportedVolumes;
    result.attachedVolumeEntries = attachResult.attachedVolumeEntries;
    result.skippedVolumes = attachResult.skippedVolumes;
    result.failedVolumes = attachResult.failedVolumes;
    result.preferredVolumeId = attachResult.preferredVolumeId;
    result.messages.insert(result.messages.end(),
                           attachResult.messages.begin(),
                           attachResult.messages.end());
    if (loadedCachedProject &&
        openDataSegmentEntryCount(*pkg, remoteCacheRoot, sample) > 0) {
        const auto cacheAttachResult =
            attachExistingOpenDataSegmentCaches(*pkg, sample, remoteCacheRoot);
        result.supportedTifxyzSegments += cacheAttachResult.supportedTifxyzSegments;
        result.cachedTifxyzSegments += cacheAttachResult.cachedTifxyzSegments;
        result.attachedSegmentEntries += cacheAttachResult.attachedSegmentEntries;
        result.skippedTifxyzSegments += cacheAttachResult.skippedTifxyzSegments;
        result.failedTifxyzSegments += cacheAttachResult.failedTifxyzSegments;
        result.transformedTifxyzSegments += cacheAttachResult.transformedTifxyzSegments;
        result.failedTransformedTifxyzSegments += cacheAttachResult.failedTransformedTifxyzSegments;
        result.messages.insert(result.messages.end(),
                               cacheAttachResult.messages.begin(),
                               cacheAttachResult.messages.end());
        result.messages.push_back("Reused cached sample project segment entries.");
    } else {
        attachOpenDataSampleSegments(*pkg, sample, remoteCacheRoot, result, progressCallback);
    }

    if (!cachedProjectPath.empty()) {
        try {
            pkg->save(cachedProjectPath);
        } catch (const std::exception& e) {
            result.messages.push_back("Failed to save cached sample project " +
                                      cachedProjectPath.string() + ": " + e.what());
        } catch (...) {
            result.messages.push_back("Failed to save cached sample project " +
                                      cachedProjectPath.string() + ": unknown error.");
        }
    }

    if (resultOut) {
        *resultOut = std::move(result);
    }
    return pkg;
}

OpenDataSampleProjectResult attachOpenDataSampleVolumes(
    VolumePkg& pkg,
    const OpenDataSample& sample)
{
    OpenDataSampleProjectResult result;
    std::vector<std::string> supportedVolumeIds;

    for (const auto& volume : sample.volumes) {
        if (volume.artifacts.empty()) {
            ++result.skippedVolumes;
            result.messages.push_back("Skipped " + volumeLabel(volume) + ": no volume artifact.");
            continue;
        }

        bool foundSupportedArtifact = false;
        for (const auto& artifact : volume.artifacts) {
            const std::string url = artifactUrl(artifact);
            if (!isSupportedRemoteZarr(volume, artifact, url)) {
                continue;
            }

            if (!foundSupportedArtifact && !volume.id.empty()) {
                supportedVolumeIds.push_back(volume.id);
            }
            foundSupportedArtifact = true;
            ++result.supportedVolumes;
            if (result.preferredVolumeId.empty()) {
                result.preferredVolumeId = volume.id;
            }

            const auto label = volumeArtifactLabel(volume, artifact);
            const auto tags = volumeTags(sample, volume, artifact);
            if (hasVolumeEntry(pkg, url)) {
                pkg.mergeVolumeEntryTags(url, tags);
                ++result.skippedVolumes;
                result.messages.push_back("Skipped " + label + ": already attached.");
                continue;
            }

            try {
                if (pkg.addVolumeEntry(url, tags)) {
                    ++result.attachedVolumeEntries;
                } else {
                    ++result.failedVolumes;
                    result.messages.push_back("Failed to attach " + label + ".");
                }
            } catch (const std::exception& e) {
                ++result.failedVolumes;
                result.messages.push_back("Failed to attach " + label + ": " + e.what());
            } catch (...) {
                ++result.failedVolumes;
                result.messages.push_back("Failed to attach " + label + ": unknown error.");
            }
        }

        if (!foundSupportedArtifact) {
            ++result.skippedVolumes;
            result.messages.push_back("Skipped " + volumeLabel(volume) + ": unsupported volume artifact.");
        }
    }

    if (const auto segmentPreferredId =
            segmentSourcePreferredVolumeId(sample, supportedVolumeIds);
        !segmentPreferredId.empty()) {
        result.preferredVolumeId = segmentPreferredId;
    }

    return result;
}

void attachOpenDataSampleSegments(
    VolumePkg& pkg,
    const OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    OpenDataSampleProjectResult& result,
    const OpenDataSampleProgressCallback& progressCallback)
{
    const auto cacheResult = reconcileOpenDataSampleSegments(
        pkg,
        sample,
        remoteCacheRoot,
        progressCallback);
    result.supportedTifxyzSegments += cacheResult.supportedTifxyzSegments;
    result.cachedTifxyzSegments += cacheResult.cachedTifxyzSegments;
    result.attachedSegmentEntries += cacheResult.attachedSegmentEntries;
    result.skippedTifxyzSegments += cacheResult.skippedTifxyzSegments;
    result.failedTifxyzSegments += cacheResult.failedTifxyzSegments;
    result.transformedTifxyzSegments += cacheResult.transformedTifxyzSegments;
    result.failedTransformedTifxyzSegments += cacheResult.failedTransformedTifxyzSegments;
    result.messages.insert(result.messages.end(),
                           cacheResult.messages.begin(),
                           cacheResult.messages.end());
}


} // namespace vc3d::opendata
