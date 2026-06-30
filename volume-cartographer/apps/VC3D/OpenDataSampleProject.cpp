#include "OpenDataSampleProject.hpp"

#include "vc/core/types/VolumePkg.hpp"

#include <algorithm>
#include <cctype>
#include <exception>

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
    return containsInsensitive(artifact.type, "zarr") ||
           containsInsensitive(volume.dataFormat, "zarr");
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

std::vector<std::string> volumeTags(const OpenDataVolume& volume,
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

} // namespace

std::shared_ptr<VolumePkg> createOpenDataSampleProject(
    const OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    OpenDataSampleProjectResult* resultOut)
{
    auto pkg = VolumePkg::newEmpty();
    pkg->setName(sample.id.empty() ? "Open Data Sample" : sample.id);
    if (!remoteCacheRoot.empty()) {
        pkg->setRemoteCacheRoot(remoteCacheRoot);
    }

    auto result = attachOpenDataSampleVolumes(*pkg, sample);
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

    for (const auto& volume : sample.volumes) {
        const auto* artifact = preferredVolumeArtifact(volume);
        if (!artifact) {
            ++result.skippedVolumes;
            result.messages.push_back("Skipped " + volumeLabel(volume) + ": no volume artifact.");
            continue;
        }

        const std::string url = artifactUrl(*artifact);
        if (!isSupportedRemoteZarr(volume, *artifact, url)) {
            ++result.skippedVolumes;
            result.messages.push_back("Skipped " + volumeLabel(volume) + ": unsupported volume artifact.");
            continue;
        }

        ++result.supportedVolumes;
        if (result.preferredVolumeId.empty()) {
            result.preferredVolumeId = volume.id;
        }

        if (hasVolumeEntry(pkg, url)) {
            ++result.skippedVolumes;
            result.messages.push_back("Skipped " + volumeLabel(volume) + ": already attached.");
            continue;
        }

        try {
            if (pkg.addVolumeEntry(url, volumeTags(volume, *artifact))) {
                ++result.attachedVolumeEntries;
            } else {
                ++result.failedVolumes;
                result.messages.push_back("Failed to attach " + volumeLabel(volume) + ".");
            }
        } catch (const std::exception& e) {
            ++result.failedVolumes;
            result.messages.push_back("Failed to attach " + volumeLabel(volume) + ": " + e.what());
        } catch (...) {
            ++result.failedVolumes;
            result.messages.push_back("Failed to attach " + volumeLabel(volume) + ": unknown error.");
        }
    }

    return result;
}

} // namespace vc3d::opendata
