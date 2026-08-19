#include "OpenDataLasagna.hpp"

#include "OpenDataNormalGrids.hpp"
#include "OpenDataSegmentCacheIO.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/RemoteFileCache.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/ProjectVolumes.hpp"
#include "utils/zarr.hpp"

#include <QUrl>
#include <QXmlStreamReader>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>

namespace vc3d::opendata {
namespace {

std::string lowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string trimSlashes(std::string value)
{
    while (!value.empty() && value.back() == '/') value.pop_back();
    return value;
}

std::string safeComponent(std::string value)
{
    for (char& c : value) {
        const auto uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '-' && c != '_' && c != '.') c = '_';
    }
    while (!value.empty() && (value.front() == '.' || value.front() == '_'))
        value.erase(value.begin());
    return value.empty() ? "unnamed" : value;
}

std::string identityHash(std::string_view value)
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char c : value) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}

std::string tagValue(const std::vector<std::string>& tags, std::string_view prefix)
{
    for (const auto& tag : tags)
        if (tag.rfind(prefix, 0) == 0) return tag.substr(prefix.size());
    return {};
}

bool hasTag(const std::vector<std::string>& tags, std::string_view value)
{
    return std::any_of(tags.begin(), tags.end(), [&](const std::string& tag) {
        return tag == value;
    });
}

bool splitPrefixUrl(const std::string& rawUrl,
                    std::string& origin,
                    std::string& prefix)
{
    const auto url = trimSlashes(rawUrl);
    const auto scheme = url.find("://");
    if (scheme == std::string::npos) return false;
    const auto path = url.find('/', scheme + 3);
    if (path == std::string::npos) return false;
    origin = url.substr(0, path);
    prefix = url.substr(path + 1) + "/";
    return !prefix.empty();
}

std::string discoverManifestKey(const OpenDataLasagnaInfo& info)
{
    std::string origin;
    std::string prefix;
    if (!splitPrefixUrl(info.artifactUrl, origin, prefix))
        throw std::runtime_error("Lasagna artifact URL is not a listable prefix: " +
                                 info.artifactUrl);
    const QString listUrl = QString::fromStdString(origin) +
        QStringLiteral("/?list-type=2&max-keys=100&delimiter=%2F&prefix=") +
        QString::fromUtf8(QUrl::toPercentEncoding(QString::fromStdString(prefix)));
    const auto xmlText = vc::httpGetString(listUrl.toStdString());
    QXmlStreamReader xml(QString::fromStdString(xmlText));
    std::vector<std::string> keys;
    while (!xml.atEnd()) {
        xml.readNext();
        if (xml.isStartElement() && xml.name() == QStringLiteral("Key")) {
            const auto key = xml.readElementText().toStdString();
            if (key.rfind(prefix, 0) != 0) continue;
            const auto relative = key.substr(prefix.size());
            if (!relative.empty() && relative.find('/') == std::string::npos &&
                relative.ends_with(".lasagna.json")) {
                keys.push_back(key);
            }
        }
    }
    if (xml.hasError())
        throw std::runtime_error("Failed to parse Lasagna artifact listing: " +
                                 xml.errorString().toStdString());
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    if (keys.size() != 1) {
        throw std::runtime_error(
            "Lasagna artifact must contain exactly one root .lasagna.json; found " +
            std::to_string(keys.size()));
    }
    return keys.front();
}

void validateGroupDescriptor(const vc::lasagna::LasagnaDatasetManifest& manifest,
                             const vc::lasagna::LasagnaChannelGroup& group)
{
    const auto array = vc::lasagna::openLasagnaChannelArray(manifest, group, 1);
    const auto& meta = array.metadata();
    if (meta.dtype != utils::ZarrDtype::uint8)
        throw std::runtime_error("Lasagna channel group '" + group.name +
                                 "' must be uint8");
    if (group.channels.size() != 1) {
        throw std::runtime_error("Lasagna channel group '" + group.name +
                                 "' must describe exactly one 3D channel");
    }
    if (meta.shape.size() != 3 || meta.chunks.size() != 3) {
        throw std::runtime_error("Lasagna channel group '" + group.name +
                                 "' must be a 3D (Z,Y,X) Zarr");
    }
    if (!manifest.baseShapeZYX) return;
    const double spacing = static_cast<double>(group.scaleFactor()) *
                           manifest.sourceToBase;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto expected = static_cast<std::size_t>(std::ceil(
            static_cast<double>((*manifest.baseShapeZYX)[axis]) / spacing));
        const auto actual = meta.shape[axis];
        const auto padding = meta.chunks[axis];
        if (actual < expected || actual > expected + padding) {
            throw std::runtime_error("Lasagna channel group '" + group.name +
                                     "' shape is incompatible with base_shape_zyx");
        }
    }
}

bool compatibleBaseShape(
    const std::array<std::size_t, 3>& catalogShape,
    const std::array<std::size_t, 3>& manifestShape)
{
    for (std::size_t axis = 0; axis < catalogShape.size(); ++axis) {
        // Published volume shapes and Lasagna base shapes use extent and
        // inclusive-coordinate conventions, respectively. They can therefore
        // differ by one voxel while describing the same coordinate domain.
        const auto smaller = std::min(catalogShape[axis], manifestShape[axis]);
        const auto larger = std::max(catalogShape[axis], manifestShape[axis]);
        if (larger - smaller > 1) return false;
    }
    return true;
}

void validatePrepared(const OpenDataLasagnaInfo& info,
                      const std::filesystem::path& manifestPath)
{
    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    const auto& manifest = dataset.manifest();
    (void)validateOpenDataLasagnaManifest(info, manifest);
    std::set<std::string> scheduled;
    std::vector<std::future<void>> validations;
    for (const auto& group : manifest.groups) {
        if (scheduled.insert(group.relativeZarrKey).second) {
            const auto* groupPtr = &group;
            validations.push_back(std::async(std::launch::async, [&manifest, groupPtr]() {
                validateGroupDescriptor(manifest, *groupPtr);
            }));
        }
    }
    for (auto& validation : validations) {
        validation.get();
    }
}

std::vector<std::string> entryTags(const OpenDataLasagnaInfo& info)
{
    std::vector<std::string> tags{
        "open-data",
        std::string(vc::project::kAnonymousRemoteAuthTag),
        std::string(kOpenDataLasagnaEntryTag),
        std::string(kOpenDataSampleIdTagPrefix) + info.sampleId,
        "vc-open-data-volume-id:" + info.volumeId,
        "vc-open-data-source-coordinate-level:" +
            std::to_string(info.sourceCoordinateLevel),
        "vc-open-data-coordinate-space:" + info.sampleId + "/" + info.volumeId +
            "@L" + std::to_string(info.sourceCoordinateLevel),
        std::string(kOpenDataLasagnaArtifactTagPrefix) + info.artifactUrl,
    };
    if (!info.modelId.empty())
        tags.push_back(std::string(kOpenDataLasagnaModelTagPrefix) + info.modelId);
    return tags;
}

int dyadicLevelForShapes(const std::array<std::size_t, 3>& baseShape,
                         const std::array<int, 3>& workingShape)
{
    std::vector<int> matches;
    for (int level = 0; level <= 5; ++level) {
        const std::size_t scale = std::size_t{1} << level;
        bool compatible = true;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (workingShape[axis] <= 0) {
                compatible = false;
                break;
            }
            const std::size_t actual = static_cast<std::size_t>(workingShape[axis]);
            const std::size_t ceilShape = (baseShape[axis] + scale - 1) / scale;
            const std::size_t floorShape = std::max<std::size_t>(1, baseShape[axis] / scale);
            if (actual != ceilShape && actual != floorShape) {
                compatible = false;
                break;
            }
        }
        if (compatible) matches.push_back(level);
    }
    if (matches.size() != 1)
        throw std::runtime_error(
            "Active volume shape does not identify exactly one supported Lasagna "
            "coordinate scale (L0-L5)");
    return matches.front();
}

std::optional<ResolvedOpenDataLasagna> resolveForTags(
    const VolumePkg& pkg,
    const std::vector<std::string>& volumeTags,
    const std::optional<std::array<int, 3>>& workingShape)
{
    const auto sampleId = tagValue(volumeTags, kOpenDataSampleIdTagPrefix);
    const auto volumeId = tagValue(volumeTags, "vc-open-data-volume-id:");
    const auto levelText = tagValue(volumeTags, "vc-open-data-source-coordinate-level:");
    if (!sampleId.empty() && !volumeId.empty()) {
        std::optional<int> activeLevel;
        if (!levelText.empty()) {
            try {
                std::size_t consumed = 0;
                const int level = std::stoi(levelText, &consumed);
                if (consumed != levelText.size() || level < 0 || level > 5)
                    throw std::runtime_error("invalid");
                activeLevel = level;
            } catch (...) {
                throw std::runtime_error(
                    "Active catalog volume has an invalid coordinate level");
            }
        }
        const auto lasagnaEntries = pkg.lasagnaDatasetEntries();
        std::vector<const vc::project::Entry*> parentMatches;
        for (const auto& entry : lasagnaEntries) {
            if (!hasTag(entry.tags, kOpenDataLasagnaEntryTag)) continue;
            if (tagValue(entry.tags, kOpenDataSampleIdTagPrefix) == sampleId &&
                tagValue(entry.tags, "vc-open-data-volume-id:") == volumeId)
                parentMatches.push_back(&entry);
        }

        if (!parentMatches.empty()) {
            const auto shapeManifestPath = vc::project::resolveLocalPath(
                parentMatches.front()->location, pkg.path().parent_path());
            if (workingShape) {
                const auto dataset =
                    vc::lasagna::LasagnaDataset::open(shapeManifestPath);
                if (!dataset.manifest().baseShapeZYX)
                    throw std::runtime_error(
                        "Lasagna manifest has no base_shape_zyx for coordinate pairing");
                const int shapeLevel = dyadicLevelForShapes(
                    *dataset.manifest().baseShapeZYX, *workingShape);
                if (activeLevel && *activeLevel != shapeLevel)
                    throw std::runtime_error(
                        "Active volume shape disagrees with its catalog coordinate level");
                activeLevel = shapeLevel;
            } else if (!activeLevel) {
                throw std::runtime_error(
                    "Catalog coordinate tags have no level and no volume shape was supplied");
            }

            std::vector<const vc::project::Entry*> compatibleMatches;
            for (const auto* entry : parentMatches) {
                const auto artifactLevelText = tagValue(
                    entry->tags, "vc-open-data-source-coordinate-level:");
                int artifactLevel = -1;
                try {
                    std::size_t consumed = 0;
                    artifactLevel = std::stoi(
                        artifactLevelText, &consumed);
                    if (consumed != artifactLevelText.size() ||
                        artifactLevel < 0 || artifactLevel > 5)
                        throw std::runtime_error("invalid");
                } catch (...) {
                    throw std::runtime_error(
                        "Cached Lasagna entry has an invalid coordinate level");
                }
                if (*activeLevel == 0 || *activeLevel == artifactLevel)
                    compatibleMatches.push_back(entry);
            }
            if (compatibleMatches.size() > 1)
                throw std::runtime_error(
                    "Multiple Lasagna datasets match the active catalog coordinate space");
            if (compatibleMatches.empty())
                throw std::runtime_error(
                    "No Lasagna dataset is published for the active catalog coordinate level");

            const auto* match = compatibleMatches.front();
            const auto path = vc::project::resolveLocalPath(
                match->location, pkg.path().parent_path());
            const auto artifactUrl =
                tagValue(match->tags, kOpenDataLasagnaArtifactTagPrefix);
            return ResolvedOpenDataLasagna{
                path,
                lasagnaSourceManifestLocation(*match),
                static_cast<double>(std::uint64_t{1} << *activeLevel),
                sampleId + "/" + volumeId + "@L" +
                    std::to_string(*activeLevel),
                artifactUrl,
                true};
        }
        if (!tagValue(volumeTags, kOpenDataLasagnaArtifactTagPrefix).empty()) {
            throw std::runtime_error(
                "The manifest declares Lasagna for the active catalog volume, "
                "but its validated local cache entry is unavailable");
        }
    }

    const auto manualLocation = pkg.selectedLasagnaDataset();
    if (manualLocation.empty()) return std::nullopt;
    vc::lasagna::LasagnaDatasetOpenOptions options;
    options.remoteCacheRoot = pkg.remoteCacheRootOrEmpty();
    const auto resolvedLocation = vc::project::isLocationRemote(manualLocation)
        ? manualLocation
        : vc::project::resolveLocalPath(
              manualLocation, pkg.path().parent_path()).string();
    const auto dataset = vc::lasagna::LasagnaDataset::openLocation(
        resolvedLocation, options);
    return ResolvedOpenDataLasagna{
        dataset.manifest().manifestPath, manualLocation, 1.0, {}, {}, false};
}

} // namespace

OpenDataLasagnaDatasetKind validateOpenDataLasagnaManifest(
    const OpenDataLasagnaInfo& info,
    const vc::lasagna::LasagnaDatasetManifest& manifest)
{
    if (!info.levelWasExplicit ||
        info.sourceCoordinateLevel < 0 ||
        info.sourceCoordinateLevel > 5) {
        throw std::runtime_error(
            "Lasagna artifact has missing or invalid parameters.level");
    }
    if (info.lasagnaVersionPresent && !info.lasagnaVersion) {
        throw std::runtime_error(
            "Lasagna artifact has invalid creation_info.lasagna_version");
    }
    if (info.sourceToBasePresent && !info.sourceToBase) {
        throw std::runtime_error(
            "Lasagna artifact has invalid creation_info.source_to_base");
    }
    if (manifest.version != 2) {
        throw std::runtime_error("Unsupported open-data Lasagna manifest version " +
                                 std::to_string(manifest.version));
    }
    if (info.lasagnaVersion && manifest.version != *info.lasagnaVersion) {
        throw std::runtime_error("Outer and inner Lasagna versions disagree");
    }
    if (info.sourceToBase &&
        std::abs(manifest.sourceToBase - *info.sourceToBase) > 1.0e-12) {
        throw std::runtime_error("Outer and inner source_to_base values disagree");
    }
    if (info.baseShapeZYX &&
        (!manifest.baseShapeZYX ||
         !compatibleBaseShape(*info.baseShapeZYX, *manifest.baseShapeZYX))) {
        throw std::runtime_error(
            "Lasagna base_shape_zyx does not match its parent volume");
    }

    if (!manifest.fiberPredictionPrefixes().empty()) {
        return OpenDataLasagnaDatasetKind::FiberInference;
    }
    if (manifest.hasNormalSource()) {
        return OpenDataLasagnaDatasetKind::Normal;
    }
    throw std::runtime_error(
        "Open-data Lasagna artifact contains neither normal Lasagna "
        "(grad_mag/nx/ny) nor fiber inference (presence/nx/ny) channels");
}

std::string lasagnaSourceManifestLocation(const vc::project::Entry& entry)
{
    const auto artifactUrl =
        tagValue(entry.tags, kOpenDataLasagnaArtifactTagPrefix);
    if (artifactUrl.empty())
        return entry.location;
    return joinOpenDataUrl(
        artifactUrl, std::filesystem::path(entry.location).filename().string());
}

std::vector<OpenDataLasagnaInfo> lasagnaArtifacts(
    const std::string& sampleId,
    const OpenDataVolume& volume)
{
    std::vector<OpenDataLasagnaInfo> result;
    for (std::size_t artifactIndex = 0;
         artifactIndex < volume.artifacts.size(); ++artifactIndex) {
        const auto& artifact = volume.artifacts[artifactIndex];
        if (lowerCopy(artifact.type) != kLasagnaArtifactType) continue;
        const auto url = trimSlashes(
            artifact.resolvedUrl.empty() ? artifact.sourcePath : artifact.resolvedUrl);
        if (url.empty()) continue;
        OpenDataLasagnaInfo info;
        info.sampleId = sampleId;
        info.volumeId = volume.id;
        info.artifactIndex = artifactIndex;
        info.artifactUrl = url;
        info.modelId = artifact.modelId.value_or(std::string{});
        if (!artifact.levelParameterPresent || !artifact.sourceCoordinateLevel)
            continue;
        info.sourceCoordinateLevel = *artifact.sourceCoordinateLevel;
        info.levelWasExplicit = true;
        info.lasagnaVersionPresent = artifact.lasagnaVersionPresent;
        info.lasagnaVersion = artifact.lasagnaVersion;
        info.sourceToBasePresent = artifact.sourceToBasePresent;
        info.sourceToBase = artifact.sourceToBase;
        info.baseShapeZYX = volume.shapeZYX;
        const auto duplicate = std::find_if(result.begin(), result.end(), [&](const auto& item) {
            return item.artifactUrl == info.artifactUrl &&
                   item.modelId == info.modelId &&
                   item.sourceCoordinateLevel == info.sourceCoordinateLevel;
        });
        if (duplicate == result.end()) result.push_back(std::move(info));
    }
    return result;
}

std::filesystem::path lasagnaCacheDir(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataLasagnaInfo& info)
{
    std::ostringstream identity;
    identity << info.artifactUrl << '\n' << info.modelId << '\n'
             << info.lasagnaVersion.value_or(-1) << '\n'
             << std::setprecision(17) << info.sourceToBase.value_or(-1.0);
    if (info.baseShapeZYX) {
        for (const auto extent : *info.baseShapeZYX) identity << '\n' << extent;
    }
    return remoteCacheRoot / "open_data" / "lasagna" /
           safeComponent(info.sampleId) / safeComponent(info.volumeId) /
           identityHash(identity.str());
}

std::filesystem::path prepareOpenDataLasagna(
    const OpenDataLasagnaInfo& info,
    const std::filesystem::path& remoteCacheRoot,
    std::string* errorOut)
{
    try {
        if (remoteCacheRoot.empty())
            throw std::runtime_error("No remote cache directory configured");
        const auto cacheRoot = std::filesystem::absolute(remoteCacheRoot).lexically_normal();
        const auto finalDir = lasagnaCacheDir(cacheRoot, info);
        const auto markerPath = finalDir / vc::lasagna::kLasagnaRemoteMarker;
        if (std::filesystem::is_regular_file(markerPath)) {
            auto marker = nlohmann::json::parse(std::ifstream(markerPath));
            if (marker.value("artifact_url", std::string{}) == info.artifactUrl &&
                marker.value("sample_id", std::string{}) == info.sampleId &&
                marker.value("volume_id", std::string{}) == info.volumeId &&
                marker.value("model_id", std::string{}) == info.modelId) {
                const auto manifest = finalDir /
                    marker.value("manifest_file", std::string{});
                if (std::filesystem::is_regular_file(manifest)) {
                    bool markerChanged = false;
                    if (!marker.value("anonymous", false)) {
                        marker["anonymous"] = true;
                        markerChanged = true;
                    }
                    if (marker.value("source_coordinate_level", -1) !=
                        info.sourceCoordinateLevel) {
                        marker["source_coordinate_level"] =
                            info.sourceCoordinateLevel;
                        markerChanged = true;
                    }
                    if (markerChanged) {
                        detail::writeStringAtomic(markerPath, marker.dump(2));
                    }
                    validatePrepared(info, manifest);
                    return manifest;
                }
            }
        }

        const std::string key = discoverManifestKey(info);
        std::string origin;
        std::string prefix;
        if (!splitPrefixUrl(info.artifactUrl, origin, prefix))
            throw std::runtime_error("Invalid Lasagna artifact URL");
        const auto relativeName = key.substr(prefix.size());
        const auto manifestUrl = joinOpenDataUrl(origin, key);
        vc::core::util::RemoteFileCacheOptions cacheOptions;
        cacheOptions.cacheRoot = cacheRoot;
        cacheOptions.destination =
            (finalDir / relativeName).lexically_relative(cacheRoot);
        cacheOptions.discoverAwsCredentials = false;
        const auto cached = vc::core::util::cacheRemoteFile(
            manifestUrl, cacheOptions);
        nlohmann::json marker{
            {"version", 1},
            {"anonymous", true},
            {"artifact_url", info.artifactUrl},
            {"sample_id", info.sampleId},
            {"volume_id", info.volumeId},
            {"model_id", info.modelId},
            {"source_coordinate_level", info.sourceCoordinateLevel},
            {"manifest_file", relativeName},
        };
        if (info.lasagnaVersion)
            marker["lasagna_version"] = *info.lasagnaVersion;
        if (info.sourceToBase)
            marker["source_to_base"] = *info.sourceToBase;
        if (info.baseShapeZYX)
            marker["base_shape_zyx"] = *info.baseShapeZYX;
        detail::writeStringAtomic(
            finalDir / vc::lasagna::kLasagnaRemoteMarker, marker.dump(2));
        try {
            validatePrepared(info, cached.path);
        } catch (...) {
            std::error_code ec;
            std::filesystem::remove(markerPath, ec);
            throw;
        }
        return cached.path;
    } catch (const std::exception& e) {
        if (errorOut) *errorOut = e.what();
        return {};
    }
}

int attachOpenDataLasagna(VolumePkg& pkg,
                          const OpenDataSample& sample,
                          const std::filesystem::path& remoteCacheRoot,
                          std::vector<std::string>* messages,
                          const OpenDataResourceSelection* selection)
{
    int attached = 0;
    std::set<std::string> expected;
    for (std::size_t volumeIndex = 0; volumeIndex < sample.volumes.size();
         ++volumeIndex) {
        const auto& volume = sample.volumes[volumeIndex];
        if (selection && !selection->allowsVolume(volume.id)) {
            continue;
        }
        const auto infos = lasagnaArtifacts(sample.id, volume);
        const bool hasLasagnaArtifact = std::any_of(
            volume.artifacts.begin(), volume.artifacts.end(), [](const auto& artifact) {
                return lowerCopy(artifact.type) == kLasagnaArtifactType;
            });
        if (infos.empty() && hasLasagnaArtifact) {
            if (messages) messages->push_back(
                "Skipped Lasagna for " + volume.id +
                ": parameters.level is missing or invalid.");
            continue;
        }
        if (infos.empty()) continue;

        for (const auto& info : infos) {
            if (selection &&
                !selection->allowsRepresentation(
                    volumeIndex, info.artifactIndex,
                    OpenDataRepresentationKind::Lasagna, volume.id)) {
                continue;
            }
            std::string error;
            const auto manifest =
                prepareOpenDataLasagna(info, remoteCacheRoot, &error);
            if (manifest.empty()) {
                if (messages) {
                    messages->push_back(
                        "Failed to prepare Lasagna for " + volume.id +
                        " at coordinate level L" +
                        std::to_string(info.sourceCoordinateLevel) + ": " + error);
                }
                continue;
            }
            expected.insert(manifest.string());
            const auto tags = entryTags(info);
            try {
                const auto dataset = vc::lasagna::LasagnaDataset::open(manifest);
                const auto kind = validateOpenDataLasagnaManifest(
                    info, dataset.manifest());
                const bool fiberInference =
                    kind == OpenDataLasagnaDatasetKind::FiberInference;
                std::vector<VolumePkg::PreparedVolumeAttachment> volumes;
                for (auto& prepared :
                     vc::lasagna::prepareLasagnaProjectVolumes(
                         dataset, manifest.string())) {
                    volumes.push_back({
                        std::move(prepared.location), std::move(prepared.tags),
                        std::move(prepared.volume)});
                }
                const auto result = pkg.attachPreparedLasagnaDataset(
                    manifest.string(), tags, fiberInference, volumes,
                    remoteCacheRoot,
                    true, true,
                    {std::string(kOpenDataLasagnaArtifactTagPrefix),
                     std::string(kOpenDataLasagnaModelTagPrefix),
                     "vc-open-data-source-coordinate-level:",
                     "vc-open-data-coordinate-space:"});
                if (result == VolumePkg::AttachLasagnaResult::VolumeIdConflict)
                    throw std::runtime_error(
                        "a Lasagna volume id conflicts with the project");
            } catch (const std::exception& error) {
                if (messages) {
                    messages->push_back(
                        "Failed to attach Lasagna volumes for " + volume.id +
                        ": " + error.what());
                }
                continue;
            }
            ++attached;
        }
    }

    std::vector<std::string> stale;
    for (const auto& entry : pkg.allLasagnaDatasetEntries()) {
        if (!hasTag(entry.tags, kOpenDataLasagnaEntryTag) ||
            tagValue(entry.tags, kOpenDataSampleIdTagPrefix) != sample.id)
            continue;
        if (expected.count(entry.location) == 0) stale.push_back(entry.location);
    }
    for (const auto& location : stale) pkg.removeEntry(location);
    return attached;
}

std::optional<ResolvedOpenDataLasagna> resolveLasagnaForVolume(
    const VolumePkg& pkg,
    const std::string& loadedVolumeId)
{
    const auto volume = pkg.volume(loadedVolumeId);
    if (!volume)
        throw std::runtime_error("Active volume is not loaded for Lasagna shape pairing");
    return resolveForTags(pkg, pkg.volumeTags(loadedVolumeId), volume->shape());
}

std::optional<ResolvedOpenDataLasagna> resolveLasagnaForCoordinateTags(
    const VolumePkg& pkg,
    const std::vector<std::string>& volumeTags)
{
    return resolveForTags(pkg, volumeTags, std::nullopt);
}

std::optional<ResolvedOpenDataLasagna> resolveLasagnaForCoordinateShape(
    const VolumePkg& pkg,
    const std::vector<std::string>& volumeTags,
    const std::array<int, 3>& workingShapeZYX)
{
    return resolveForTags(pkg, volumeTags, workingShapeZYX);
}

} // namespace vc3d::opendata
