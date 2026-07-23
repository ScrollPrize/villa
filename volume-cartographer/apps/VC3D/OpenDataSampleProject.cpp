#include "OpenDataSampleProject.hpp"

#include "OpenDataNormalGrids.hpp"
#include "OpenDataLasagna.hpp"
#include "OpenDataSegmentCache.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/util/RemoteUrl.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
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
    std::string networkUrl;
    try {
        networkUrl = vc::parseRemoteVolumeSpec(url).sourceUrl;
    } catch (...) {
        return false;
    }
    const std::string loweredUrl = lowerCopy(networkUrl);
    const auto query = loweredUrl.find('?');
    const std::string pathOnly = loweredUrl.substr(0, query);
    if (loweredUrl.rfind("http://", 0) != 0 &&
        loweredUrl.rfind("https://", 0) != 0 &&
        loweredUrl.rfind("s3://", 0) != 0) {
        return false;
    }
    if (pathOnly.size() >= 5 &&
        pathOnly.substr(pathOnly.size() - 5) == ".zarr") {
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
                                    const OpenDataArtifact& artifact,
                                    bool preferredNativeSource,
                                    std::optional<int> coordinateLevel = std::nullopt,
                                    std::optional<double> effectiveVoxelSize = std::nullopt,
                                    std::string sourcePath = {})
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
    const auto lasagna = lasagnaArtifacts(sample.id, volume);
    if (lasagna.size() == 1) {
        addUnique(std::string(kOpenDataLasagnaArtifactTagPrefix) +
                  lasagna.front().artifactUrl);
    } else if (lasagna.size() > 1) {
        addUnique(std::string(kOpenDataLasagnaArtifactTagPrefix) + "ambiguous");
    }
    if (preferredNativeSource && !coordinateLevel)
        coordinateLevel = 0;
    if (preferredNativeSource)
        addUnique("vc-open-data-preferred-source");
    if (coordinateLevel) {
        addUnique("vc-open-data-source-coordinate-level:" +
                  std::to_string(*coordinateLevel));
        addUnique("vc-open-data-coordinate-space:" + sample.id + "/" +
                  volume.id + "@L" + std::to_string(*coordinateLevel));
        if (!sourcePath.empty())
            addUnique("vc-open-data-source-path:" + sourcePath);
        if (volume.pixelSizeUm && std::isfinite(*volume.pixelSizeUm) &&
            *volume.pixelSizeUm > 0.0) {
            addUnique("vc-open-data-source-original-resolution:" +
                      std::to_string(*volume.pixelSizeUm));
        }
    }
    if (effectiveVoxelSize && *effectiveVoxelSize > 0.0) {
        addUnique("vc-open-data-voxel-size-um:" + std::to_string(*effectiveVoxelSize));
    } else if (!containsInsensitive(artifact.type, "prediction") &&
               volume.pixelSizeUm && *volume.pixelSizeUm > 0.0) {
        addUnique("vc-open-data-voxel-size-um:" + std::to_string(*volume.pixelSizeUm));
    }

    return tags;
}

bool isSupportedSurfacePrediction(const OpenDataArtifact& artifact)
{
    return lowerCopy(artifact.type) == "surface-prediction-zarr";
}

bool isSupportedInk3dPrediction(const OpenDataArtifact& artifact)
{
    const auto type = lowerCopy(artifact.type);
    return type == "ink-detection-3d-zarr" ||
           type == "ink_detection_3d_zarr";
}

bool isSupportedCoordinatePrediction(const OpenDataArtifact& artifact)
{
    return isSupportedSurfacePrediction(artifact) ||
           isSupportedInk3dPrediction(artifact);
}

std::optional<int> predictionSourceCoordinateLevel(const OpenDataArtifact& artifact)
{
    if (!isSupportedCoordinatePrediction(artifact))
        return std::nullopt;

    if (artifact.sourceCoordinateLevel)
        return artifact.sourceCoordinateLevel;

    // The first published 3D-ink stores predate parameters.level. They are
    // native-resolution predictions, but only trust that legacy convention
    // after preflightPredictionAndSource verifies the prediction descriptor
    // against physical source /0.
    if (isSupportedInk3dPrediction(artifact) && !artifact.levelParameterPresent)
        return 0;

    return std::nullopt;
}

std::string coordinateSpaceId(const OpenDataSample& sample,
                              const OpenDataVolume& volume,
                              int level)
{
    return sample.id + "/" + volume.id + "@L" + std::to_string(level);
}

std::vector<std::string> coordinateTags(const OpenDataSample& sample,
                                        const OpenDataVolume& volume,
                                        int level,
                                        double voxelSize,
                                        const std::string& sourcePath)
{
    return {
        "vc-open-data-source-coordinate-level:" + std::to_string(level),
        "vc-open-data-coordinate-space:" + coordinateSpaceId(sample, volume, level),
        "vc-open-data-voxel-size-um:" + std::to_string(voxelSize),
        "vc-open-data-source-path:" + sourcePath,
        "vc-open-data-source-original-resolution:" +
            std::to_string(*volume.pixelSizeUm),
    };
}

const std::vector<std::string>& coordinateSingletonPrefixes()
{
    static const std::vector<std::string> prefixes{
        "vc-open-data-voxel-size-um:",
        "vc-open-data-source-coordinate-level:",
        "vc-open-data-coordinate-space:",
        "vc-open-data-source-path:",
        "vc-open-data-source-original-resolution:",
        "vc-open-data-name:",
        "vc-open-data-coordinate-level-unknown",
    };
    return prefixes;
}

void preflightPredictionAndSource(const std::string& predictionUrl,
                                  const std::string& sourceUrl,
                                  int sourceLevel)
{
    auto prediction = vc::render::openHttpZarrPyramid(
        predictionUrl, vc::HttpAuth{}, 0); // explicit zero requests strict validation
    auto source = vc::render::openHttpZarrPyramid(
        sourceUrl, vc::HttpAuth{}, sourceLevel);
    if (prediction.shapes.empty() || source.shapes.empty())
        throw std::runtime_error("prediction/source descriptor is empty");
    if (prediction.shapes[0] != source.shapes[0])
        throw std::runtime_error("prediction /0 shape does not match source physical /" +
                                 std::to_string(sourceLevel));
    if (prediction.dtype != source.dtype)
        throw std::runtime_error("prediction/source dtype mismatch");
    if (!prediction.physicalLevelZeroTransformIsIdentity)
        throw std::runtime_error("prediction /0 coordinate scale must be identity");
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

} // namespace

std::shared_ptr<VolumePkg> createOpenDataSampleProject(
    const OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    OpenDataSampleProjectResult* resultOut,
    const OpenDataSampleProgressCallback& progressCallback,
    const OpenDataResourceSelection* selection)
{
    auto result = OpenDataSampleProjectResult{};
    std::shared_ptr<VolumePkg> pkg;
    const auto cachedProjectPath = remoteCacheRoot.empty()
        ? std::filesystem::path{}
        : sampleProjectCachePath(remoteCacheRoot, sample);

    if (!cachedProjectPath.empty() && isNonEmptyFile(cachedProjectPath)) {
        try {
            vc::project::LoadOptions opts;
            opts.remoteCacheRoot = remoteCacheRoot;
            opts.deferResolution = true;
            pkg = VolumePkg::load(
                cachedProjectPath,
                opts);
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
        vc::project::LoadOptions opts;
        opts.remoteCacheRoot = remoteCacheRoot;
        opts.deferResolution = true;
        pkg = VolumePkg::newEmpty(opts);
    }
    pkg->setName(sample.id.empty() ? "Open Data Sample" : sample.id);
    if (!remoteCacheRoot.empty()) {
        pkg->setRemoteCacheRoot(remoteCacheRoot);
    }

    auto attachResult = attachOpenDataSampleVolumes(*pkg, sample, selection);
    result.supportedVolumes = attachResult.supportedVolumes;
    result.attachedVolumeEntries = attachResult.attachedVolumeEntries;
    result.skippedVolumes = attachResult.skippedVolumes;
    result.failedVolumes = attachResult.failedVolumes;
    result.preferredVolumeId = attachResult.preferredVolumeId;
    result.messages.insert(result.messages.end(),
                           attachResult.messages.begin(),
                           attachResult.messages.end());
    if (!remoteCacheRoot.empty()) {
        result.attachedNormalGrids =
            attachOpenDataNormalGrids(*pkg, sample, remoteCacheRoot, selection);
        if (result.attachedNormalGrids > 0) {
            result.messages.push_back("Attached " +
                                      std::to_string(result.attachedNormalGrids) +
                                      " streaming normal grid store(s).");
        }
        result.attachedLasagnaDatasets = attachOpenDataLasagna(
            *pkg, sample, remoteCacheRoot, &result.messages, selection);
        if (result.attachedLasagnaDatasets > 0) {
            result.messages.push_back(
                "Attached " + std::to_string(result.attachedLasagnaDatasets) +
                " manifest-backed Lasagna dataset(s).");
        }
    }
    // Reconcile against the current manifest on every open. This is metadata-
    // only for lazy segments and also removes stale layout entries from older
    // cached projects.
    attachOpenDataSampleSegments(*pkg, sample, remoteCacheRoot, result,
                                 progressCallback);
    // Catalog loads remain unresolved until volume tags, normal-grid paths,
    // and every segment representation/cache entry have been reconciled.
    if (pkg->entryResolutionDeferred()) {
        if (progressCallback) {
            OpenDataSampleDownloadProgress progress;
            progress.totalSegments = static_cast<int>(sample.volumes.size());
            progress.status = "resolving-volumes";
            try { progressCallback(progress); } catch (...) {}
        }
        pkg->resolveDeferredEntries();
        if (progressCallback) {
            OpenDataSampleDownloadProgress progress;
            progress.totalSegments = static_cast<int>(sample.volumes.size());
            progress.completedSegments = progress.totalSegments;
            progress.status = "project-ready";
            try { progressCallback(progress); } catch (...) {}
        }
        const std::string sampleTag =
            std::string(kOpenDataSampleIdTagPrefix) + sample.id;
        for (const auto& entry : pkg->volumeEntries()) {
            if (std::find(entry.tags.begin(), entry.tags.end(), sampleTag) ==
                    entry.tags.end() ||
                pkg->hasLoadedVolumeEntry(entry.location)) {
                continue;
            }
            ++result.failedVolumes;
            result.messages.push_back(
                "Catalog volume failed remote open/base-level validation: " +
                entry.location + ".");
        }
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
    const OpenDataSample& sample,
    const OpenDataResourceSelection* selection)
{
    OpenDataSampleProjectResult result;
    std::vector<std::string> supportedVolumeIds;

    struct PredictionCandidate {
        const OpenDataVolume* volume = nullptr;
        const OpenDataArtifact* prediction = nullptr;
        std::string predictionUrl;
        std::string sourceUrl;
        int sourceCoordinateLevel = 0;
        bool inferredSourceCoordinateLevel = false;
    };
    std::vector<PredictionCandidate> predictions;
    std::vector<std::string> attachedLocations;

    for (std::size_t volumeIndex = 0; volumeIndex < sample.volumes.size();
         ++volumeIndex) {
        const auto& volume = sample.volumes[volumeIndex];
        // Skip an excluded volume and all of its artifacts and counters.
        if (selection && !selection->allowsVolume(volume.id)) {
            continue;
        }
        if (volume.artifacts.empty()) {
            ++result.skippedVolumes;
            result.messages.push_back("Skipped " + volumeLabel(volume) + ": no volume artifact.");
            continue;
        }

        bool foundSupportedArtifact = false;
        const auto* preferredSource = preferredVolumeArtifact(volume);
        std::string preferredSourceUrl;
        if (preferredSource) {
            preferredSourceUrl = artifactUrl(*preferredSource);
            if (isSupportedCoordinatePrediction(*preferredSource) ||
                !isSupportedRemoteZarr(volume, *preferredSource, preferredSourceUrl)) {
                preferredSource = nullptr;
                preferredSourceUrl.clear();
            }
        }
        for (std::size_t artifactIndex = 0;
             artifactIndex < volume.artifacts.size(); ++artifactIndex) {
            const auto& artifact = volume.artifacts[artifactIndex];
            const std::string url = artifactUrl(artifact);
            if (!isSupportedRemoteZarr(volume, artifact, url)) {
                continue;
            }

            // Source volumes use only the volume filter; predictions must also
            // pass the representation and kind filters.
            if (selection) {
                if (const auto reprKind =
                        classifyDerivedRepresentation(artifact)) {
                    if (!selection->allowsRepresentation(
                            volumeIndex, artifactIndex, *reprKind, volume.id)) {
                        continue;
                    }
                }
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
            const bool preferredNativeSource = preferredSource == &artifact;
            auto tags = volumeTags(
                sample, volume, artifact, preferredNativeSource,
                preferredNativeSource ? std::optional<int>{0} : std::nullopt,
                preferredNativeSource ? volume.pixelSizeUm : std::nullopt,
                preferredNativeSource ? preferredSourceUrl : std::string{});
            if (isSupportedCoordinatePrediction(artifact) &&
                !artifact.sourceCoordinateLevel) {
                tags.push_back(
                    artifact.levelParameterPresent
                        ? "vc-open-data-coordinate-level-unknown:invalid-parameters.level"
                        : "vc-open-data-coordinate-level-unknown:missing-parameters.level");
            }
            if (hasVolumeEntry(pkg, url)) {
                pkg.reconcileVolumeEntryTags(
                    url, tags, coordinateSingletonPrefixes());
                ++result.skippedVolumes;
                result.messages.push_back("Skipped " + label + ": already attached.");
                if (!pkg.entryResolutionDeferred() &&
                    !pkg.hasLoadedVolumeEntry(url)) {
                    ++result.failedVolumes;
                    result.messages.push_back(
                        "Attached entry remains unavailable after remote validation: " +
                        label + ".");
                }
            } else {
                try {
                    if (pkg.addVolumeEntry(url, tags)) {
                        ++result.attachedVolumeEntries;
                        if (!pkg.entryResolutionDeferred() &&
                            !pkg.hasLoadedVolumeEntry(url)) {
                            ++result.failedVolumes;
                            result.messages.push_back(
                                "Persisted " + label +
                                " but remote open/base-level validation failed.");
                        }
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
            if (hasVolumeEntry(pkg, url))
                attachedLocations.push_back(url);

            if (const auto level = predictionSourceCoordinateLevel(artifact)) {
                predictions.push_back(PredictionCandidate{
                    &volume,
                    &artifact,
                    url,
                    preferredSourceUrl,
                    *level,
                    !artifact.sourceCoordinateLevel.has_value()});
            } else if (isSupportedCoordinatePrediction(artifact)) {
                result.messages.push_back(
                    "Kept " + label +
                    " coordinate-unspecified: parameters.level is missing or invalid.");
            }
        }

        if (!foundSupportedArtifact) {
            ++result.skippedVolumes;
            result.messages.push_back("Skipped " + volumeLabel(volume) + ": unsupported volume artifact.");
        }
    }

    std::vector<std::string> attachedVirtualLocators;
    for (const auto& candidate : predictions) {
        const auto& volume = *candidate.volume;
        const auto& prediction = *candidate.prediction;
        const auto predictionLabel = volumeArtifactLabel(volume, prediction);
        const int level = candidate.sourceCoordinateLevel;
        if (candidate.sourceUrl.empty() ||
            std::find(attachedLocations.begin(), attachedLocations.end(),
                      candidate.sourceUrl) == attachedLocations.end()) {
            ++result.failedVolumes;
            result.messages.push_back(
                "Skipped virtual source for " + predictionLabel +
                ": preferred source Zarr is unavailable or was not attached.");
            continue;
        }
        if (!volume.pixelSizeUm || !std::isfinite(*volume.pixelSizeUm) ||
            *volume.pixelSizeUm <= 0.0) {
            ++result.failedVolumes;
            result.messages.push_back(
                "Skipped coordinate pairing for " + predictionLabel +
                ": source volume has no positive voxel size.");
            continue;
        }

        try {
            preflightPredictionAndSource(
                candidate.predictionUrl, candidate.sourceUrl, level);
        } catch (const std::exception& e) {
            ++result.failedVolumes;
            result.messages.push_back(
                "Skipped coordinate pairing for " + predictionLabel +
                " against source L" + std::to_string(level) + ": " + e.what());
            continue;
        }

        const double effectiveVoxelSize =
            *volume.pixelSizeUm * static_cast<double>(std::uint64_t{1} << level);
        auto predictionTags = volumeTags(
            sample, volume, prediction, false, level, effectiveVoxelSize,
            candidate.sourceUrl);
        pkg.reconcileVolumeEntryTags(
            candidate.predictionUrl,
            predictionTags,
            coordinateSingletonPrefixes());
        if (candidate.inferredSourceCoordinateLevel) {
            result.messages.push_back(
                "Paired legacy " + predictionLabel + " with source L" +
                std::to_string(level) + " after descriptor validation.");
        }

        if (level == 0)
            continue;

        const auto virtualLocator = vc::parseRemoteVolumeSpec(
            candidate.sourceUrl + "#vc-base-scale=" + std::to_string(level)).portableLocator;
        if (std::find(attachedVirtualLocators.begin(), attachedVirtualLocators.end(),
                      virtualLocator) != attachedVirtualLocators.end()) {
            continue;
        }
        attachedVirtualLocators.push_back(virtualLocator);

        auto tags = coordinateTags(
            sample, volume, level, effectiveVoxelSize, candidate.sourceUrl);
        tags.push_back(std::string(kOpenDataSampleIdTagPrefix) + sample.id);
        tags.push_back("vc-open-data-volume-id:" + volume.id);
        const auto lasagna = lasagnaArtifacts(sample.id, volume);
        if (lasagna.size() == 1) {
            tags.push_back(std::string(kOpenDataLasagnaArtifactTagPrefix) +
                           lasagna.front().artifactUrl);
        } else if (lasagna.size() > 1) {
            tags.push_back(
                std::string(kOpenDataLasagnaArtifactTagPrefix) + "ambiguous");
        }
        tags.push_back("vc-open-data-virtual-source");
        const std::string sourceName = volume.suffix.empty() ? volumeLabel(volume) : volume.suffix;
        tags.push_back("vc-open-data-name:" + sourceName + " [source L" +
                       std::to_string(level) + ", " +
                       std::to_string(effectiveVoxelSize) + " um]");

        if (hasVolumeEntry(pkg, virtualLocator)) {
            pkg.reconcileVolumeEntryTags(
                virtualLocator, tags, coordinateSingletonPrefixes());
            ++result.skippedVolumes;
        } else if (pkg.addVolumeEntry(virtualLocator, tags)) {
            ++result.attachedVolumeEntries;
            result.messages.push_back(
                "Attached rebased source view " + virtualLocator + ".");
            if (!pkg.entryResolutionDeferred() &&
                !pkg.hasLoadedVolumeEntry(virtualLocator)) {
                ++result.failedVolumes;
                result.messages.push_back(
                    "Persisted rebased source view but remote open/base-level validation failed: " +
                    virtualLocator + ".");
            }
        } else {
            ++result.failedVolumes;
            result.messages.push_back(
                "Failed to persist rebased source view " + virtualLocator + ".");
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
