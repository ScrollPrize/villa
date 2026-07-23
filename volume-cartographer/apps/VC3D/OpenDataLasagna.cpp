#include "OpenDataLasagna.hpp"

#include "OpenDataNormalGrids.hpp"

#include "vc/core/types/Volume.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/HttpFetch.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "utils/zarr.hpp"

#include <QUrl>
#include <QXmlStreamReader>

#include <algorithm>
#include <atomic>
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

void writeText(const std::filesystem::path& path, std::string_view text)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Failed to create " + path.string());
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.close();
    if (!out) throw std::runtime_error("Failed to write " + path.string());
}

void validateGroupDescriptor(const vc::lasagna::LasagnaDatasetManifest& manifest,
                             const vc::lasagna::LasagnaChannelGroup& group)
{
    const auto array = vc::lasagna::openLasagnaChannelArray(manifest, group, 1);
    const auto& meta = array.metadata();
    if (meta.dtype != utils::ZarrDtype::uint8)
        throw std::runtime_error("Lasagna channel group '" + group.name +
                                 "' must be uint8");
    const std::size_t spatialOffset = meta.shape.size() == 4 ? 1 : 0;
    if ((meta.shape.size() != 3 && meta.shape.size() != 4) ||
        meta.chunks.size() != meta.shape.size()) {
        throw std::runtime_error("Lasagna channel group '" + group.name +
                                 "' must be a 3D or channel-first 4D Zarr");
    }
    if (meta.shape.size() == 4 && group.channels.size() > meta.shape[0])
        throw std::runtime_error("Lasagna channel group '" + group.name +
                                 "' has more channels than its Zarr");
    if (!manifest.baseShapeZYX) return;
    const double spacing = static_cast<double>(group.scaleFactor()) *
                           manifest.sourceToBase;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto expected = static_cast<std::size_t>(std::ceil(
            static_cast<double>((*manifest.baseShapeZYX)[axis]) / spacing));
        const auto actual = meta.shape[spatialOffset + axis];
        const auto padding = meta.chunks[spatialOffset + axis];
        if (actual < expected || actual > expected + padding) {
            throw std::runtime_error("Lasagna channel group '" + group.name +
                                     "' shape is incompatible with base_shape_zyx");
        }
    }
}

void validatePrepared(const OpenDataLasagnaInfo& info,
                      const std::filesystem::path& manifestPath)
{
    if (!info.lasagnaVersionPresent || !info.lasagnaVersion)
        throw std::runtime_error(
            "Lasagna artifact has missing or invalid creation_info.lasagna_version");
    if (!info.sourceToBasePresent || !info.sourceToBase)
        throw std::runtime_error(
            "Lasagna artifact has missing or invalid creation_info.source_to_base");
    const auto dataset = vc::lasagna::LasagnaDataset::open(manifestPath);
    const auto& manifest = dataset.manifest();
    if (manifest.version != 2)
        throw std::runtime_error("Unsupported open-data Lasagna manifest version " +
                                 std::to_string(manifest.version));
    if (info.lasagnaVersion && manifest.version != *info.lasagnaVersion)
        throw std::runtime_error("Outer and inner Lasagna versions disagree");
    if (info.sourceToBase &&
        std::abs(manifest.sourceToBase - *info.sourceToBase) > 1.0e-12)
        throw std::runtime_error("Outer and inner source_to_base values disagree");
    if (info.baseShapeZYX) {
        if (!manifest.baseShapeZYX || *manifest.baseShapeZYX != *info.baseShapeZYX)
            throw std::runtime_error("Lasagna base_shape_zyx does not match its parent volume");
    }
    for (const auto* channel : {"grad_mag", "nx", "ny"}) {
        if (!manifest.groupForChannel(channel))
            throw std::runtime_error("Open-data Lasagna artifact is missing required channel '" +
                                     std::string(channel) + "'");
    }
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
        std::string(kOpenDataLasagnaEntryTag),
        std::string(kOpenDataSampleIdTagPrefix) + info.sampleId,
        "vc-open-data-volume-id:" + info.volumeId,
        "vc-open-data-source-coordinate-level:0",
        "vc-open-data-coordinate-space:" + info.sampleId + "/" + info.volumeId + "@L0",
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
        std::optional<int> declaredLevel;
        if (!levelText.empty()) {
            std::size_t consumed = 0;
            const int level = std::stoi(levelText, &consumed);
            if (consumed != levelText.size() || level < 0 || level > 5)
                throw std::runtime_error(
                    "Active catalog volume has an invalid coordinate level");
            declaredLevel = level;
        }
        std::vector<const vc::project::Entry*> matches;
        for (const auto& entry : pkg.lasagnaDatasetEntries()) {
            if (!hasTag(entry.tags, kOpenDataLasagnaEntryTag)) continue;
            if (tagValue(entry.tags, kOpenDataSampleIdTagPrefix) == sampleId &&
                tagValue(entry.tags, "vc-open-data-volume-id:") == volumeId)
                matches.push_back(&entry);
        }
        if (matches.size() > 1)
            throw std::runtime_error(
                "Multiple Lasagna datasets match the active catalog volume");
        if (matches.size() == 1) {
            const auto path = vc::project::resolveLocalPath(
                matches.front()->location, pkg.path().parent_path());
            int level = declaredLevel.value_or(0);
            if (workingShape) {
                const auto dataset = vc::lasagna::LasagnaDataset::open(path);
                if (!dataset.manifest().baseShapeZYX)
                    throw std::runtime_error(
                        "Lasagna manifest has no base_shape_zyx for coordinate pairing");
                level = dyadicLevelForShapes(
                    *dataset.manifest().baseShapeZYX, *workingShape);
                if (declaredLevel && *declaredLevel != level)
                    throw std::runtime_error(
                        "Active volume shape disagrees with its catalog coordinate level");
            } else if (!declaredLevel) {
                throw std::runtime_error(
                    "Catalog coordinate tags have no level and no volume shape was supplied");
            }
            return ResolvedOpenDataLasagna{
                path,
                static_cast<double>(std::uint64_t{1} << level),
                sampleId + "/" + volumeId + "@L" + std::to_string(level),
                tagValue(matches.front()->tags, kOpenDataLasagnaArtifactTagPrefix),
                true};
        }
        if (!tagValue(volumeTags, kOpenDataLasagnaArtifactTagPrefix).empty()) {
            throw std::runtime_error(
                "The manifest declares Lasagna for the active catalog volume, "
                "but its validated local cache entry is unavailable");
        }
    }

    const auto manual = pkg.selectedLasagnaDatasetPath();
    if (manual.empty()) return std::nullopt;
    return ResolvedOpenDataLasagna{manual, 1.0, {}, {}, false};
}

} // namespace

std::vector<OpenDataLasagnaInfo> lasagnaArtifacts(
    const std::string& sampleId,
    const OpenDataVolume& volume)
{
    std::vector<OpenDataLasagnaInfo> result;
    for (const auto& artifact : volume.artifacts) {
        if (lowerCopy(artifact.type) != kLasagnaArtifactType) continue;
        const auto url = trimSlashes(
            artifact.resolvedUrl.empty() ? artifact.sourcePath : artifact.resolvedUrl);
        if (url.empty()) continue;
        OpenDataLasagnaInfo info;
        info.sampleId = sampleId;
        info.volumeId = volume.id;
        info.artifactUrl = url;
        info.modelId = artifact.modelId.value_or(std::string{});
        info.lasagnaVersionPresent = artifact.lasagnaVersionPresent;
        info.lasagnaVersion = artifact.lasagnaVersion;
        info.sourceToBasePresent = artifact.sourceToBasePresent;
        info.sourceToBase = artifact.sourceToBase;
        info.baseShapeZYX = volume.shapeZYX;
        const auto duplicate = std::find_if(result.begin(), result.end(), [&](const auto& item) {
            return item.artifactUrl == info.artifactUrl && item.modelId == info.modelId;
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
    std::filesystem::path tempDir;
    try {
        if (remoteCacheRoot.empty())
            throw std::runtime_error("No remote cache directory configured");
        const auto finalDir = lasagnaCacheDir(remoteCacheRoot, info);
        const auto markerPath = finalDir / vc::lasagna::kLasagnaRemoteMarker;
        if (std::filesystem::is_regular_file(markerPath)) {
            const auto marker = nlohmann::json::parse(std::ifstream(markerPath));
            if (marker.value("artifact_url", std::string{}) == info.artifactUrl &&
                marker.value("sample_id", std::string{}) == info.sampleId &&
                marker.value("volume_id", std::string{}) == info.volumeId &&
                marker.value("model_id", std::string{}) == info.modelId) {
                const auto manifest = finalDir /
                    marker.value("manifest_file", std::string{});
                if (std::filesystem::is_regular_file(manifest)) {
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
        const auto body = vc::httpGetString(joinOpenDataUrl(origin, key));
        if (body.empty()) throw std::runtime_error("Lasagna manifest download was empty");

        static std::atomic<std::uint64_t> serial{0};
        tempDir = std::filesystem::path(
            finalDir.string() + ".tmp-" + std::to_string(serial.fetch_add(1)));
        std::error_code ec;
        std::filesystem::remove_all(tempDir, ec);
        std::filesystem::create_directories(tempDir);
        const auto tempManifest = tempDir / relativeName;
        writeText(tempManifest, body);
        nlohmann::json marker{
            {"version", 1},
            {"artifact_url", info.artifactUrl},
            {"sample_id", info.sampleId},
            {"volume_id", info.volumeId},
            {"model_id", info.modelId},
            {"manifest_file", relativeName},
        };
        if (info.lasagnaVersion)
            marker["lasagna_version"] = *info.lasagnaVersion;
        if (info.sourceToBase)
            marker["source_to_base"] = *info.sourceToBase;
        if (info.baseShapeZYX)
            marker["base_shape_zyx"] = *info.baseShapeZYX;
        writeText(tempDir / vc::lasagna::kLasagnaRemoteMarker, marker.dump(2));
        validatePrepared(info, tempManifest);

        std::filesystem::create_directories(finalDir.parent_path());
        std::filesystem::remove_all(finalDir, ec);
        std::filesystem::rename(tempDir, finalDir);
        return finalDir / relativeName;
    } catch (const std::exception& e) {
        if (!tempDir.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(tempDir, ec);
        }
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
        if (infos.size() > 1) {
            if (messages) messages->push_back(
                "Skipped Lasagna for " + volume.id +
                ": multiple manifest artifacts are ambiguous.");
            continue;
        }
        if (infos.empty()) continue;
        if (selection) {
            // Map the representation back to its artifact index before applying
            // the representation and kind filters.
            std::optional<std::size_t> artifactIndex;
            for (std::size_t ai = 0; ai < volume.artifacts.size(); ++ai) {
                const auto& art = volume.artifacts[ai];
                if (lowerCopy(art.type) != kLasagnaArtifactType) continue;
                const auto url = trimSlashes(
                    art.resolvedUrl.empty() ? art.sourcePath : art.resolvedUrl);
                if (url == infos.front().artifactUrl) {
                    artifactIndex = ai;
                    break;
                }
            }
            if (!artifactIndex ||
                !selection->allowsRepresentation(
                    volumeIndex, *artifactIndex,
                    OpenDataRepresentationKind::Lasagna, volume.id)) {
                continue;
            }
        }
        std::string error;
        const auto manifest = prepareOpenDataLasagna(infos.front(), remoteCacheRoot, &error);
        if (manifest.empty()) {
            if (messages) messages->push_back(
                "Failed to prepare Lasagna for " + volume.id + ": " + error);
            continue;
        }
        expected.insert(manifest.string());
        const auto tags = entryTags(infos.front());
        if (!pkg.addLasagnaDatasetEntry(manifest.string(), tags)) {
            pkg.reconcileLasagnaDatasetEntryTags(
                manifest.string(), tags,
                {std::string(kOpenDataLasagnaArtifactTagPrefix),
                 std::string(kOpenDataLasagnaModelTagPrefix),
                 "vc-open-data-source-coordinate-level:",
                 "vc-open-data-coordinate-space:"});
        }
        ++attached;
    }

    std::vector<std::string> stale;
    for (const auto& entry : pkg.lasagnaDatasetEntries()) {
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
