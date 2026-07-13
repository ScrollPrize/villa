#include "OpenDataSegmentCache.hpp"
#include "OpenDataSegmentCacheIO.hpp"

#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/AffineTransform.hpp"
#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <exception>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>

namespace vc3d::opendata {
namespace {

using detail::cacheOptionalFile;
using detail::isNonEmptyFile;
using detail::readTextFile;
using detail::writeBytesAtomic;
using detail::writeCachedTifxyzBand;
using detail::writeStringAtomic;

constexpr const char* kMaterializationKey = "materialization";
constexpr const char* kLazyPlaceholderMetaKey =
    "vc_open_data_lazy_placeholder";

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

std::string artifactUrl(const OpenDataArtifact& artifact)
{
    return trimTrailingSlashes(artifact.resolvedUrl.empty()
                                  ? artifact.sourcePath
                                  : artifact.resolvedUrl);
}

std::string urlPathWithoutQuery(std::string value)
{
    if (const auto pos = value.find('#'); pos != std::string::npos) {
        value.erase(pos);
    }
    if (const auto pos = value.find('?'); pos != std::string::npos) {
        value.erase(pos);
    }
    while (!value.empty() && value.back() == '/') {
        value.pop_back();
    }
    return value;
}

std::string imageExtensionForUrl(const std::string& url)
{
    const std::string path = lowerCopy(urlPathWithoutQuery(url));
    for (const auto* ext : {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}) {
        if (path.size() >= std::strlen(ext) &&
            path.compare(path.size() - std::strlen(ext), std::strlen(ext), ext) == 0) {
            return ext;
        }
    }
    return {};
}

bool isJpegExtension(const std::string& ext)
{
    return ext == ".jpg" || ext == ".jpeg";
}

bool isSupportedInkImageArtifact(const OpenDataArtifact& artifact)
{
    const std::string type = lowerCopy(artifact.type);
    if (type.find("ink") == std::string::npos) {
        return false;
    }
    if (type.find("zarr") != std::string::npos || type.find("3d") != std::string::npos) {
        return false;
    }
    return !imageExtensionForUrl(artifactUrl(artifact)).empty();
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

std::string segmentLabel(const OpenDataSegment& segment)
{
    return segment.id.empty() ? std::string("<unnamed segment>") : segment.id;
}

std::string segmentStableId(const OpenDataSegment& segment)
{
    if (!segment.longId.empty()) {
        return segment.longId;
    }
    if (!segment.id.empty()) {
        return segment.id;
    }
    return safePathComponent(segment.suffix);
}

std::string sourceVolumeIdForSegment(const OpenDataSegment& segment);

bool hasSegmentEntry(const VolumePkg& pkg, const std::string& location)
{
    const auto& entries = pkg.segmentEntries();
    return std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
        return entry.location == location;
    });
}

std::optional<double> jsonNumberLike(const nlohmann::json& obj, const char* key)
{
    if (!obj.is_object()) {
        return std::nullopt;
    }
    const auto it = obj.find(key);
    if (it == obj.end()) {
        return std::nullopt;
    }
    if (it->is_number()) {
        return it->get<double>();
    }
    if (it->is_string()) {
        try {
            return std::stod(it->get<std::string>());
        } catch (...) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

double originalVolumeDownscale(const OpenDataSegment& segment)
{
    for (const auto* key : {"original_volume_downscale",
                            "originalVolumeDownscale",
                            "volume_downscale",
                            "volumeDownscale"}) {
        if (const auto value = jsonNumberLike(segment.properties, key);
            value.has_value() && std::isfinite(*value) && *value > 0.0) {
            return *value;
        }
    }
    return 1.0;
}

std::string derivedVolumeId(const OpenDataSegment& segment)
{
    if (segment.raw.is_object()) {
        const auto creation = segment.raw.find("creation");
        if (creation != segment.raw.end() && creation->is_object()) {
            const auto derived = creation->find("derived_from");
            if (derived != creation->end() && derived->is_object()) {
                const auto type = derived->find("type");
                const auto id = derived->find("id");
                if (type != derived->end() && type->is_string() &&
                    id != derived->end() && id->is_string() &&
                    lowerCopy(type->get<std::string>()) == "volume") {
                    return id->get<std::string>();
                }
            }
        }
    }
    return segment.originalVolumeId;
}

std::string sourceVolumeIdForSegment(const OpenDataSegment& segment)
{
    const auto derived = derivedVolumeId(segment);
    return derived.empty() ? segment.originalVolumeId : derived;
}

void applyOpenDataMetadata(nlohmann::json& meta,
                           const std::string& baseUrl,
                           const OpenDataSegment& segment)
{
    meta["vc_open_data_tifxyz_url"] = baseUrl;
    meta["vc_open_data_segment_id"] = segment.id;
    meta["vc_open_data_segment_long_id"] = segment.longId;
    meta["vc_open_data_original_volume_id"] = segment.originalVolumeId;
    if (const auto derivedId = derivedVolumeId(segment); !derivedId.empty()) {
        meta["vc_open_data_derived_volume_id"] = derivedId;
    }
    meta["vc_open_data_original_volume_downscale"] = originalVolumeDownscale(segment);
}

bool coordinatesAlreadyScaled(const nlohmann::json& meta, double expectedFactor)
{
    const auto factor = jsonNumberLike(
        meta, "vc_open_data_coordinates_scaled_to_original_volume");
    if (!factor)
        return false;
    if (!std::isfinite(*factor) ||
        std::abs(*factor - expectedFactor) >
            1.0e-9 * std::max({1.0, std::abs(*factor), std::abs(expectedFactor)})) {
        throw std::runtime_error(
            "legacy catalog segment coordinate-scale marker does not match manifest downscale");
    }
    return true;
}

std::optional<cv::Matx44d> parseOpenDataMatrix(const nlohmann::json& matrixJson)
{
    if (!matrixJson.is_array() || (matrixJson.size() != 3 && matrixJson.size() != 4)) {
        return std::nullopt;
    }

    cv::Matx44d matrix = cv::Matx44d::eye();
    for (int row = 0; row < static_cast<int>(matrixJson.size()); ++row) {
        const auto& rowJson = matrixJson.at(static_cast<std::size_t>(row));
        if (!rowJson.is_array() || rowJson.size() != 4) {
            return std::nullopt;
        }
        for (int col = 0; col < 4; ++col) {
            if (!rowJson.at(static_cast<std::size_t>(col)).is_number()) {
                return std::nullopt;
            }
            matrix(row, col) = rowJson.at(static_cast<std::size_t>(col)).get<double>();
        }
    }
    if (matrixJson.size() == 4 &&
        (std::abs(matrix(3, 0)) > 1e-12 ||
         std::abs(matrix(3, 1)) > 1e-12 ||
         std::abs(matrix(3, 2)) > 1e-12 ||
         std::abs(matrix(3, 3) - 1.0) > 1e-12)) {
        return std::nullopt;
    }
    return matrix;
}

std::optional<cv::Matx44d> findVolumeTransformMatrix(const OpenDataSample& sample,
                                                     const std::string& sourceVolumeId,
                                                     const std::string& targetVolumeId)
{
    if (sourceVolumeId.empty() || targetVolumeId.empty() || sourceVolumeId == targetVolumeId) {
        return std::nullopt;
    }

    const nlohmann::json* volumeTransforms = nullptr;
    auto maybeSetTransforms = [&](const nlohmann::json& owner) {
        if (volumeTransforms || !owner.is_object()) {
            return;
        }
        const auto it = owner.find("volume_transforms");
        if (it != owner.end()) {
            volumeTransforms = &*it;
        }
    };

    maybeSetTransforms(sample.properties);
    if (sample.properties.is_object()) {
        if (const auto nested = sample.properties.find("properties"); nested != sample.properties.end()) {
            maybeSetTransforms(*nested);
        }
    }
    if (sample.raw.is_object()) {
        if (const auto sampleIt = sample.raw.find("sample"); sampleIt != sample.raw.end()) {
            maybeSetTransforms(*sampleIt);
            if (sampleIt->is_object()) {
                if (const auto propsIt = sampleIt->find("properties"); propsIt != sampleIt->end()) {
                    maybeSetTransforms(*propsIt);
                }
            }
        }
        if (const auto propsIt = sample.raw.find("properties"); propsIt != sample.raw.end()) {
            maybeSetTransforms(*propsIt);
        }
    }

    if (!volumeTransforms) {
        return std::nullopt;
    }

    auto inspectTransformList = [&](const nlohmann::json& transforms) -> std::optional<cv::Matx44d> {
        if (!transforms.is_array()) {
            return std::nullopt;
        }
        for (const auto& transform : transforms) {
            if (!transform.is_object()) {
                continue;
            }
            const auto toIt = transform.find("to_volume_id");
            if (toIt == transform.end() || !toIt->is_string() ||
                toIt->get<std::string>() != targetVolumeId) {
                continue;
            }
            const auto matrixIt = transform.find("matrix");
            if (matrixIt == transform.end()) {
                continue;
            }
            if (auto matrix = parseOpenDataMatrix(*matrixIt)) {
                return matrix;
            }
        }
        return std::nullopt;
    };

    if (volumeTransforms->is_array()) {
        for (const auto& fromEntry : *volumeTransforms) {
            if (!fromEntry.is_object()) {
                continue;
            }
            const auto fromIt = fromEntry.find("from_volume_id");
            if (fromIt == fromEntry.end() || !fromIt->is_string() ||
                fromIt->get<std::string>() != sourceVolumeId) {
                continue;
            }
            const auto transformsIt = fromEntry.find("transforms");
            if (transformsIt != fromEntry.end()) {
                if (auto matrix = inspectTransformList(*transformsIt)) {
                    return matrix;
                }
            }
        }
        return std::nullopt;
    }

    if (volumeTransforms->is_object()) {
        const auto fromIt = volumeTransforms->find(sourceVolumeId);
        if (fromIt == volumeTransforms->end()) {
            return std::nullopt;
        }
        if (fromIt->is_object()) {
            const auto transformsIt = fromIt->find("transforms");
            if (transformsIt != fromIt->end()) {
                if (auto matrix = inspectTransformList(*transformsIt)) {
                    return matrix;
                }
            }
            const auto targetIt = fromIt->find(targetVolumeId);
            if (targetIt != fromIt->end()) {
                if (targetIt->is_object()) {
                    if (const auto matrixIt = targetIt->find("matrix"); matrixIt != targetIt->end()) {
                        return parseOpenDataMatrix(*matrixIt);
                    }
                }
                return parseOpenDataMatrix(*targetIt);
            }
        }
    }

    return std::nullopt;
}

nlohmann::json matrixToJson(const cv::Matx44d& matrix)
{
    nlohmann::json out = nlohmann::json::array();
    for (int row = 0; row < 3; ++row) {
        nlohmann::json rowJson = nlohmann::json::array();
        for (int col = 0; col < 4; ++col) {
            rowJson.push_back(matrix(row, col));
        }
        out.push_back(std::move(rowJson));
    }
    return out;
}

utils::Json matrixToUtilsJson(const cv::Matx44d& matrix)
{
    auto out = utils::Json::array();
    for (int row = 0; row < 3; ++row) {
        auto rowJson = utils::Json::array();
        for (int col = 0; col < 4; ++col) {
            rowJson.push_back(matrix(row, col));
        }
        out.push_back(std::move(rowJson));
    }
    return out;
}

void applyOriginalVolumeDownscale(const std::filesystem::path& segmentDir,
                                  const OpenDataSegment& segment,
                                  const std::string& representationId)
{
    const double downscale = originalVolumeDownscale(segment);
    const auto metaPath = segmentDir / "meta.json";
    auto meta = nlohmann::json::parse(readTextFile(metaPath));
    if (!std::isfinite(downscale) || downscale <= 0.0 || downscale == 1.0 ||
        coordinatesAlreadyScaled(meta, downscale)) {
        return;
    }

    auto surface = load_quad_from_tifxyz(segmentDir.string());
    if (!surface) {
        throw std::runtime_error("failed to load cached catalog tifxyz segment: " +
                                 segmentDir.string());
    }

    vc::core::util::transformSurfacePoints(surface.get(), downscale, std::nullopt, 1.0);
    vc::core::util::refreshTransformedSurfaceState(surface.get());
    surface->meta["vc_open_data_coordinates_scaled_to_original_volume"] = downscale;
    surface->save(segmentDir.string(), representationId, true);
}

bool cachedMetadataNeedsNormalization(const std::filesystem::path& metaPath,
                                      const OpenDataSegment& segment,
                                      bool applyDownscale,
                                      const std::string& representationId)
{
    try {
        const auto meta = nlohmann::json::parse(readTextFile(metaPath));
        if (!meta.is_object()) {
            return true;
        }
        auto stringField = [&](const char* key) -> std::string {
            const auto it = meta.find(key);
            return it != meta.end() && it->is_string() ? it->get<std::string>() : std::string{};
        };
        if (stringField("uuid") != representationId ||
            stringField("type") != "seg" ||
            stringField("format") != "tifxyz" ||
            stringField("vc_open_data_segment_id") != segment.id ||
            stringField("vc_open_data_segment_long_id") != segment.longId ||
            stringField("vc_open_data_source_path").empty() ||
            !meta.contains("vc_open_data_source_coordinate_scale_factor") ||
            !meta.contains("vc_open_data_source_original_resolution")) {
            return true;
        }
        return applyDownscale &&
               originalVolumeDownscale(segment) != 1.0 &&
               !coordinatesAlreadyScaled(meta, originalVolumeDownscale(segment));
    } catch (...) {
        return true;
    }
}

const OpenDataVolume* findSampleVolume(const OpenDataSample& sample,
                                       const std::string& volumeId)
{
    const auto it = std::find_if(
        sample.volumes.begin(), sample.volumes.end(), [&](const auto& volume) {
            return volume.id == volumeId;
        });
    return it == sample.volumes.end() ? nullptr : &*it;
}

void applySourceCoordinateMetadata(nlohmann::json& meta,
                                   const OpenDataSample& sample,
                                   const std::string& volumeId,
                                   int coordinateLevel)
{
    const auto* volume = findSampleVolume(sample, volumeId);
    if (!volume || !volume->pixelSizeUm ||
        !std::isfinite(*volume->pixelSizeUm) || *volume->pixelSizeUm <= 0.0) {
        return;
    }
    const auto* source = preferredVolumeArtifact(*volume);
    if (!source)
        return;
    const std::string sourcePath = artifactUrl(*source);
    if (sourcePath.empty())
        return;
    meta["vc_open_data_source_path"] = sourcePath;
    meta["vc_open_data_source_coordinate_scale_factor"] =
        std::uint64_t{1} << coordinateLevel;
    meta["vc_open_data_source_original_resolution"] = *volume->pixelSizeUm;
}

void applySourceCoordinateMetadata(utils::Json& meta,
                                   const OpenDataSample& sample,
                                   const std::string& volumeId,
                                   int coordinateLevel)
{
    const auto* volume = findSampleVolume(sample, volumeId);
    if (!volume || !volume->pixelSizeUm ||
        !std::isfinite(*volume->pixelSizeUm) || *volume->pixelSizeUm <= 0.0) {
        return;
    }
    const auto* source = preferredVolumeArtifact(*volume);
    if (!source)
        return;
    const std::string sourcePath = artifactUrl(*source);
    if (sourcePath.empty())
        return;
    meta["vc_open_data_source_path"] = sourcePath;
    meta["vc_open_data_source_coordinate_scale_factor"] =
        std::uint64_t{1} << coordinateLevel;
    meta["vc_open_data_source_original_resolution"] = *volume->pixelSizeUm;
}

void normalizeCachedMetadata(const std::string& baseUrl,
                             const OpenDataSample& sample,
                             const OpenDataSegment& segment,
                             const std::filesystem::path& target,
                             bool applyDownscale,
                             const std::string& representationId,
                             const std::string& representation,
                             const std::string& coordinateSpace,
                             int coordinateLevel,
                             const std::string& coordinateVolumeId)
{
    auto meta = nlohmann::json::parse(readTextFile(target));
    if (!meta.is_object()) {
        throw std::runtime_error("meta.json is not an object");
    }
    if (!meta.contains("type") || !meta["type"].is_string() || meta["type"].get<std::string>().empty()) {
        meta["type"] = "seg";
    }
    meta["uuid"] = representationId;
    if (!meta.contains("name") || !meta["name"].is_string() || meta["name"].get<std::string>().empty()) {
        meta["name"] = segment.suffix.empty() ? segmentLabel(segment) : segment.suffix;
    }
    if (!meta.contains("format") || !meta["format"].is_string() || meta["format"].get<std::string>().empty()) {
        meta["format"] = "tifxyz";
    }
    applyOpenDataMetadata(meta, baseUrl, segment);
    meta["vc_open_data_catalog_segment_lineage_id"] = segmentStableId(segment);
    meta["vc_open_data_representation"] = representation;
    meta["vc_open_data_source_coordinate_level"] = coordinateLevel;
    if (!coordinateSpace.empty())
        meta["vc_open_data_coordinate_space"] = coordinateSpace;
    applySourceCoordinateMetadata(
        meta, sample, coordinateVolumeId, coordinateLevel);
    writeStringAtomic(target, meta.dump(2));
    if (applyDownscale) {
        applyOriginalVolumeDownscale(target.parent_path(), segment, representationId);
    }
}

std::optional<nlohmann::json> readCatalogOrigin(const std::filesystem::path& dir);

std::string inkDetectionLabel(const OpenDataSegment& segment,
                              const OpenDataArtifact& artifact,
                              std::size_t index)
{
    std::string label = segment.suffix.empty() ? segmentLabel(segment) : segment.suffix;
    if (!artifact.type.empty()) {
        label += " - " + artifact.type;
    } else {
        label += " - ink detection";
    }
    if (index > 0) {
        label += " " + std::to_string(index + 1);
    }
    return label;
}

std::vector<nlohmann::json> readInkDetectionRecords(const std::filesystem::path& segmentDir)
{
    auto readArrayFile = [&](const std::filesystem::path& path) -> std::vector<nlohmann::json> {
        if (!std::filesystem::is_regular_file(path)) {
            return {};
        }
        try {
            auto root = nlohmann::json::parse(readTextFile(path));
            if (!root.is_array()) {
                return {};
            }
            std::vector<nlohmann::json> out;
            out.reserve(root.size());
            for (const auto& item : root) {
                if (item.is_object()) {
                    out.push_back(item);
                }
            }
            return out;
        } catch (...) {
            return {};
        }
    };

    auto records = readArrayFile(segmentDir / "ink-detections.json");
    if (!records.empty()) {
        return records;
    }
    if (auto origin = readCatalogOrigin(segmentDir)) {
        const auto it = origin->find("ink_detections");
        if (it != origin->end() && it->is_array()) {
            std::vector<nlohmann::json> out;
            out.reserve(it->size());
            for (const auto& item : *it) {
                if (item.is_object()) {
                    out.push_back(item);
                }
            }
            return out;
        }
    }
    return {};
}

nlohmann::json inkDetectionMaterializationRecipe(
    const OpenDataSample& sample,
    const OpenDataSegment& segment)
{
    std::vector<const OpenDataArtifact*> supportedArtifacts;
    bool hasJpegArtifact = false;
    for (const auto& artifact : segment.artifacts) {
        if (!isSupportedInkImageArtifact(artifact)) {
            continue;
        }
        const std::string ext = imageExtensionForUrl(artifactUrl(artifact));
        hasJpegArtifact = hasJpegArtifact || isJpegExtension(ext);
        supportedArtifacts.push_back(&artifact);
    }

    nlohmann::json recipe = nlohmann::json::array();
    std::size_t supportedIndex = 0;
    for (const auto* artifactPtr : supportedArtifacts) {
        const auto& artifact = *artifactPtr;
        const std::string url = artifactUrl(artifact);
        const std::string ext = imageExtensionForUrl(url);
        if (url.empty() || ext.empty() ||
            (hasJpegArtifact && !isJpegExtension(ext))) {
            continue;
        }

        std::string baseName = artifact.type.empty() ? "ink_detection"
                                                     : artifact.type;
        if (supportedIndex > 0) {
            baseName += "_" + std::to_string(supportedIndex + 1);
        }
        const auto relative = std::filesystem::path("ink-detections") /
            (safePathComponent(baseName) + ext);
        recipe.push_back({
            {"download_url", url},
            {"label", inkDetectionLabel(segment, artifact, supportedIndex)},
            {"sample_id", sample.id},
            {"segment_id", segment.id},
            {"segment_long_id", segment.longId},
            {"artifact_type", artifact.type},
            {"original_source_uri", artifact.sourcePath},
            {"resolved_http_url", artifact.resolvedUrl.empty()
                                      ? artifact.sourcePath
                                      : artifact.resolvedUrl},
            {"local_file", relative.generic_string()}
        });
        ++supportedIndex;
    }
    return recipe;
}

std::string nowUtcIso()
{
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    char buf[32] = {};
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

nlohmann::json catalogOriginJson(const OpenDataSample& sample,
                                 const OpenDataSegment& segment,
                                 const OpenDataArtifact& artifact,
                                 const std::vector<std::string>& downloadedFiles)
{
    nlohmann::json j;
    j["manifest_url"] = std::string(kDefaultManifestUrl);
    j["sample_id"] = sample.id;
    j["segment_id"] = segment.id;
    j["segment_long_id"] = segment.longId;
    j["artifact_type"] = artifact.type;
    j["original_source_uri"] = artifact.sourcePath;
    j["resolved_http_url"] = artifact.resolvedUrl;
    j["downloaded_at_utc"] = nowUtcIso();
    j["downloaded_files"] = downloadedFiles;
    j["source_file_metadata"] = nlohmann::json::array();
    j["cache_state"] = cacheStateName(OpenDataSegmentCacheState::Current);
    return j;
}

bool requiredFilesPresent(const std::filesystem::path& dir)
{
    return isNonEmptyFile(dir / "meta.json") &&
           isNonEmptyFile(dir / "x.tif") &&
           isNonEmptyFile(dir / "y.tif") &&
           isNonEmptyFile(dir / "z.tif");
}

bool originMatches(const nlohmann::json& origin,
                   const OpenDataSample& sample,
                   const OpenDataSegment& segment,
                   const OpenDataArtifact& artifact)
{
    if (!origin.is_object()) {
        return false;
    }
    auto stringField = [&](const char* key) -> std::string {
        const auto it = origin.find(key);
        return it != origin.end() && it->is_string() ? it->get<std::string>() : std::string{};
    };
    const std::string expectedUrl = artifact.resolvedUrl.empty() ? artifact.sourcePath : artifact.resolvedUrl;
    const std::string actualUrl = stringField("resolved_http_url").empty()
        ? stringField("original_source_uri")
        : stringField("resolved_http_url");
    return stringField("sample_id") == sample.id &&
           stringField("segment_id") == segment.id &&
           (actualUrl.empty() || expectedUrl.empty() || actualUrl == expectedUrl);
}

bool transformedOriginMatches(const nlohmann::json& origin,
                              const OpenDataSample& sample,
                              const OpenDataSegment& segment,
                              const std::string& sourceVolumeId,
                              const std::string& targetVolumeId,
                              const cv::Matx44d& matrix)
{
    if (!origin.is_object()) {
        return false;
    }
    auto stringField = [&](const char* key) -> std::string {
        const auto it = origin.find(key);
        return it != origin.end() && it->is_string() ? it->get<std::string>() : std::string{};
    };
    if (stringField("sample_id") != sample.id ||
        stringField("segment_id") != segment.id ||
        stringField("source_volume_id") != sourceVolumeId ||
        stringField("target_volume_id") != targetVolumeId) {
        return false;
    }
    const auto matrixIt = origin.find("matrix");
    const auto cachedMatrix = matrixIt == origin.end() ? std::optional<cv::Matx44d>{}
                                                       : parseOpenDataMatrix(*matrixIt);
    if (!cachedMatrix) {
        return false;
    }
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            if (std::abs((*cachedMatrix)(row, col) - matrix(row, col)) > 1e-12) {
                return false;
            }
        }
    }
    return true;
}

std::optional<nlohmann::json> readCatalogOrigin(const std::filesystem::path& dir)
{
    const auto path = dir / "catalog-origin.json";
    if (!std::filesystem::is_regular_file(path)) {
        return std::nullopt;
    }
    try {
        return nlohmann::json::parse(readTextFile(path));
    } catch (...) {
        return std::nullopt;
    }
}

bool originStateAllowsFastOpen(const nlohmann::json& origin)
{
    const auto it = origin.find("cache_state");
    if (it == origin.end()) {
        return true;
    }
    if (!it->is_string()) {
        return false;
    }
    const auto state = lowerCopy(it->get<std::string>());
    return state.empty() || state == cacheStateName(OpenDataSegmentCacheState::Current);
}

void writeCatalogOriginState(const std::filesystem::path& dir,
                             OpenDataSegmentCacheState state)
{
    auto origin = readCatalogOrigin(dir).value_or(nlohmann::json::object());
    origin["cache_state"] = cacheStateName(state);
    writeStringAtomic(dir / "catalog-origin.json", origin.dump(2));
}

std::filesystem::path makeTempSegmentDir(const std::filesystem::path& finalDir)
{
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    return finalDir.parent_path() /
           (finalDir.filename().string() + ".tmp-" + std::to_string(tick));
}

void publishSegmentDirectory(const std::filesystem::path& tempDir,
                             const std::filesystem::path& finalDir)
{
    std::error_code ec;
    std::filesystem::create_directories(finalDir.parent_path(), ec);
    if (ec) {
        throw std::runtime_error("failed to create cache parent: " + ec.message());
    }

    const auto backupDir = finalDir.parent_path() /
        (finalDir.filename().string() + ".previous");
    std::filesystem::remove_all(backupDir, ec);
    ec.clear();
    if (std::filesystem::exists(finalDir, ec)) {
        std::filesystem::rename(finalDir, backupDir, ec);
        if (ec) {
            throw std::runtime_error("failed to move existing cache aside: " + ec.message());
        }
    }

    std::filesystem::rename(tempDir, finalDir, ec);
    if (ec) {
        std::error_code restoreEc;
        if (std::filesystem::exists(backupDir, restoreEc) &&
            !std::filesystem::exists(finalDir, restoreEc)) {
            std::filesystem::rename(backupDir, finalDir, restoreEc);
        }
        throw std::runtime_error("failed to publish segment cache: " + ec.message());
    }

    std::filesystem::remove_all(backupDir, ec);
}

nlohmann::json placeholderMetadata(
    const OpenDataSample& sample,
    const OpenDataSegment& segment,
    const std::string& representationId,
    const std::string& representation,
    const std::string& coordinateVolumeId,
    const std::string& coordinateSpace,
    const std::string& sourceUrl = {})
{
    nlohmann::json meta = nlohmann::json::object();
    if (segment.raw.is_object()) {
        const auto creation = segment.raw.find("creation");
        if (creation != segment.raw.end() && creation->is_object()) {
            const auto metadata = creation->find("metadata");
            if (metadata != creation->end() && metadata->is_object()) {
                meta = *metadata;
            }
        }
    }
    if (segment.properties.is_object()) {
        const auto coverage = segment.properties.find("volume_coverage");
        if (coverage != segment.properties.end() && coverage->is_object()) {
            const auto target = coverage->find(coordinateVolumeId);
            if (target != coverage->end() && target->is_object()) {
                const auto bbox = target->find("bbox_transformed");
                if (bbox != target->end() && bbox->is_array()) {
                    meta["bbox"] = *bbox;
                }
            }
        }
    }
    meta["type"] = "seg";
    meta["uuid"] = representationId;
    meta["name"] = segment.suffix.empty() ? segmentLabel(segment) : segment.suffix;
    meta["format"] = "tifxyz";
    if (!meta.contains("scale")) {
        meta["scale"] = nlohmann::json::array({1.0, 1.0});
    }
    applyOpenDataMetadata(meta, sourceUrl, segment);
    meta["vc_open_data_catalog_segment_lineage_id"] = segmentStableId(segment);
    meta["vc_open_data_representation"] = representation;
    meta["vc_open_data_source_coordinate_level"] = 0;
    meta["vc_open_data_coordinate_space"] = coordinateSpace;
    meta[kLazyPlaceholderMetaKey] = true;
    applySourceCoordinateMetadata(meta, sample, coordinateVolumeId, 0);
    return meta;
}

using PlaceholderPreparer = std::function<bool(
    const std::filesystem::path&, std::string*)>;

bool refreshMaterializedSegment(
    const std::filesystem::path& segmentDir,
    const PlaceholderPreparer& preparePlaceholder,
    std::string* errorOut);

bool prepareRepresentationPlaceholder(
    const OpenDataSample& sample,
    const OpenDataSegment& segment,
    const OpenDataSegmentRepresentation& representation,
    const std::filesystem::path& segmentDir,
    bool forceRefresh,
    std::string* errorOut)
{
    if (!representation.artifact) {
        if (errorOut) *errorOut = "representation has no artifact";
        return false;
    }
    const bool applyDownscale =
        representation.kind ==
        OpenDataSegmentRepresentationKind::DerivedNative;
    const char* representationName =
        representation.kind == OpenDataSegmentRepresentationKind::DerivedNative
            ? "derived-native"
            : "published-transformed";
    if (forceRefresh) {
        return refreshMaterializedSegment(
            segmentDir,
            [&](const std::filesystem::path& stagingDir,
                std::string* stagingError) {
                return prepareRepresentationPlaceholder(
                    sample, segment, representation, stagingDir, false,
                    stagingError);
            },
            errorOut);
    }
    if (requiredFilesPresent(segmentDir)) {
        if (cachedMetadataNeedsNormalization(
                segmentDir / "meta.json", segment, applyDownscale,
                representation.representationId)) {
            try {
                normalizeCachedMetadata(
                    artifactUrl(*representation.artifact), sample, segment,
                    segmentDir / "meta.json", applyDownscale,
                    representation.representationId,
                    applyDownscale ? "derived-native"
                                   : "published-transformed",
                    representation.coordinateSpace,
                    representation.sourceCoordinateLevel,
                    representation.coordinateVolumeId);
            } catch (const std::exception& e) {
                if (errorOut) *errorOut = e.what();
                return false;
            }
        }
        return true;
    }
    const std::string url = artifactUrl(*representation.artifact);
    if (url.empty()) {
        if (errorOut) *errorOut = "representation has no public URL";
        return false;
    }

    const auto tempDir = makeTempSegmentDir(segmentDir);
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
    std::filesystem::create_directories(tempDir, ec);
    if (ec) {
        if (errorOut) *errorOut = ec.message();
        return false;
    }
    try {
        writeStringAtomic(
            tempDir / "meta.json",
            placeholderMetadata(sample, segment,
                                representation.representationId,
                                representationName,
                                representation.coordinateVolumeId,
                                representation.coordinateSpace,
                                url).dump(2));
        auto origin = catalogOriginJson(sample, segment,
                                        *representation.artifact,
                                        {"meta.json"});
        origin["cache_state"] = "remote";
        origin[kMaterializationKey] = {
            {"kind", "download"},
            {"url", url},
            {"representation_id", representation.representationId},
            {"apply_downscale",
             representation.kind ==
                 OpenDataSegmentRepresentationKind::DerivedNative},
            {"original_volume_downscale", originalVolumeDownscale(segment)}
        };
        origin[kMaterializationKey]["ink_detections"] =
            inkDetectionMaterializationRecipe(sample, segment);
        writeStringAtomic(tempDir / "catalog-origin.json", origin.dump(2));
        publishSegmentDirectory(tempDir, segmentDir);
        return true;
    } catch (const std::exception& e) {
        std::filesystem::remove_all(tempDir, ec);
        if (errorOut) *errorOut = e.what();
        return false;
    }
}

bool prepareGeneratedPlaceholder(
    const OpenDataSample& sample,
    const OpenDataSegment& segment,
    const std::filesystem::path& sourceSegmentDir,
    const std::filesystem::path& targetSegmentDir,
    const std::string& targetVolumeId,
    bool forceRefresh,
    std::string* errorOut)
{
    if (forceRefresh) {
        return refreshMaterializedSegment(
            targetSegmentDir,
            [&](const std::filesystem::path& stagingDir,
                std::string* stagingError) {
                return prepareGeneratedPlaceholder(
                    sample, segment, sourceSegmentDir, stagingDir,
                    targetVolumeId, false, stagingError);
            },
            errorOut);
    }
    if (requiredFilesPresent(targetSegmentDir)) {
        return true;
    }
    const std::string sourceVolumeId = sourceVolumeIdForSegment(segment);
    const auto matrix = findVolumeTransformMatrix(
        sample, sourceVolumeId, targetVolumeId);
    if (!matrix) {
        if (errorOut) *errorOut = "missing volume transform";
        return false;
    }

    const std::string representationId =
        segmentStableId(segment) + "-generated-" +
        safePathComponent(targetVolumeId) + "-L0";
    const auto tempDir = makeTempSegmentDir(targetSegmentDir);
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
    std::filesystem::create_directories(tempDir, ec);
    if (ec) {
        if (errorOut) *errorOut = ec.message();
        return false;
    }
    try {
        writeStringAtomic(
            tempDir / "meta.json",
            placeholderMetadata(
                sample, segment, representationId,
                "generated-native-transform", targetVolumeId,
                sample.id + "/" + targetVolumeId + "@L0").dump(2));
        nlohmann::json origin;
        origin["manifest_url"] = std::string(kDefaultManifestUrl);
        origin["sample_id"] = sample.id;
        origin["segment_id"] = segment.id;
        origin["segment_long_id"] = segment.longId;
        origin["source_volume_id"] = sourceVolumeId;
        origin["target_volume_id"] = targetVolumeId;
        origin["matrix"] = matrixToJson(*matrix);
        origin["cache_state"] = "remote";
        origin[kMaterializationKey] = {
            {"kind", "generate"},
            {"source_segment_dir", sourceSegmentDir.string()},
            {"representation_id", representationId},
            {"source_volume_id", sourceVolumeId},
            {"target_volume_id", targetVolumeId},
            {"coordinate_space", sample.id + "/" + targetVolumeId + "@L0"},
            {"matrix", matrixToJson(*matrix)}
        };
        writeStringAtomic(tempDir / "catalog-origin.json", origin.dump(2));
        publishSegmentDirectory(tempDir, targetSegmentDir);
        return true;
    } catch (const std::exception& e) {
        std::filesystem::remove_all(tempDir, ec);
        if (errorOut) *errorOut = e.what();
        return false;
    }
}

std::shared_ptr<std::mutex> materializationMutexFor(
    const std::filesystem::path& segmentDir)
{
    static std::mutex mutexesMutex;
    static std::unordered_map<std::string, std::shared_ptr<std::mutex>> mutexes;
    const auto key = std::filesystem::absolute(segmentDir).lexically_normal().string();
    std::lock_guard<std::mutex> lock(mutexesMutex);
    auto& mutex = mutexes[key];
    if (!mutex) mutex = std::make_shared<std::mutex>();
    return mutex;
}

OpenDataSegmentMaterializationResult materializeOne(
    const std::filesystem::path& segmentDir)
{
    OpenDataSegmentMaterializationResult result;
    const auto pathMutex = materializationMutexFor(segmentDir);
    std::lock_guard<std::mutex> pathLock(*pathMutex);
    if (requiredFilesPresent(segmentDir)) {
        result.success = true;
        result.alreadyMaterialized = true;
        return result;
    }

    const auto origin = readCatalogOrigin(segmentDir);
    if (!origin || !origin->is_object()) {
        result.message = "segment has no open-data materialization recipe";
        result.failedSegments = 1;
        return result;
    }
    const auto recipeIt = origin->find(kMaterializationKey);
    if (recipeIt == origin->end() || !recipeIt->is_object()) {
        result.message = "segment has no open-data materialization recipe";
        result.failedSegments = 1;
        return result;
    }
    const auto& recipe = *recipeIt;
    const std::string kind = recipe.value("kind", std::string{});
    const auto tempDir = makeTempSegmentDir(segmentDir);
    std::error_code ec;
    std::filesystem::remove_all(tempDir, ec);
    std::filesystem::create_directories(tempDir, ec);
    if (ec) {
        result.message = ec.message();
        result.failedSegments = 1;
        return result;
    }

    try {
        auto meta = nlohmann::json::parse(readTextFile(segmentDir / "meta.json"));
        meta.erase(kLazyPlaceholderMetaKey);
        nlohmann::json materializedInkDetections = nlohmann::json::array();

        if (kind == "download") {
            const std::string url = recipe.value("url", std::string{});
            if (url.empty()) {
                throw std::runtime_error("download recipe has no URL");
            }
            writeStringAtomic(tempDir / "meta.json", meta.dump(2));
            writeCachedTifxyzBand(url, "x.tif", tempDir / "x.tif");
            writeCachedTifxyzBand(url, "y.tif", tempDir / "y.tif");
            writeCachedTifxyzBand(url, "z.tif", tempDir / "z.tif");
            (void)cacheOptionalFile(url, "mask.tif", tempDir / "mask.tif");
            (void)cacheOptionalFile(url, "overlapping.json",
                                    tempDir / "overlapping.json");
            if (const auto inkIt = recipe.find("ink_detections");
                inkIt != recipe.end() && inkIt->is_array()) {
                for (const auto& item : *inkIt) {
                    if (!item.is_object()) {
                        continue;
                    }
                    const std::string imageUrl =
                        item.value("download_url", std::string{});
                    const std::string localFile =
                        item.value("local_file", std::string{});
                    if (imageUrl.empty() || localFile.empty()) {
                        continue;
                    }
                    try {
                        auto bytes = vc::httpGetBytes(imageUrl);
                        if (bytes.empty()) {
                            continue;
                        }
                        writeBytesAtomic(tempDir / localFile, bytes);
                        auto record = item;
                        record.erase("download_url");
                        materializedInkDetections.push_back(std::move(record));
                    } catch (...) {
                    }
                }
            }
            if (recipe.value("apply_downscale", false)) {
                OpenDataSegment segment;
                segment.properties["original_volume_downscale"] =
                    recipe.value("original_volume_downscale", 1.0);
                applyOriginalVolumeDownscale(
                    tempDir, segment,
                    recipe.value("representation_id", std::string{}));
            }
        } else if (kind == "generate") {
            const auto sourceDir = std::filesystem::path(
                recipe.value("source_segment_dir", std::string{}));
            if (sourceDir.empty() ||
                std::filesystem::absolute(sourceDir).lexically_normal() ==
                    std::filesystem::absolute(segmentDir).lexically_normal()) {
                throw std::runtime_error("invalid generated-segment source");
            }
            const auto sourceResult = materializeOne(sourceDir);
            if (!sourceResult.success) {
                throw std::runtime_error(
                    "failed to materialize canonical source: " +
                    sourceResult.message);
            }
            const auto matrix = parseOpenDataMatrix(recipe.at("matrix"));
            if (!matrix) {
                throw std::runtime_error("invalid generated-segment matrix");
            }
            auto surface = load_quad_from_tifxyz(sourceDir.string());
            if (!surface) {
                throw std::runtime_error("failed to load canonical source");
            }
            vc::core::util::transformSurfacePoints(
                surface.get(), 1.0, *matrix, 1.0);
            vc::core::util::refreshTransformedSurfaceState(surface.get());
            surface->meta.update(utils::Json::parse(meta.dump()));
            surface->meta.erase(kLazyPlaceholderMetaKey);
            surface->meta["uuid"] =
                recipe.value("representation_id", std::string{});
            surface->meta["vc_open_data_transform_source_volume_id"] =
                recipe.value("source_volume_id", std::string{});
            surface->meta["vc_open_data_transform_target_volume_id"] =
                recipe.value("target_volume_id", std::string{});
            surface->meta["vc_open_data_volume_transform_matrix"] =
                matrixToUtilsJson(*matrix);
            surface->meta["vc_open_data_representation"] =
                "generated-native-transform";
            surface->meta["vc_open_data_source_coordinate_level"] = 0;
            surface->meta["vc_open_data_coordinate_space"] =
                recipe.value("coordinate_space", std::string{});
            surface->save(
                tempDir.string(),
                recipe.value("representation_id", std::string{}), true);
        } else {
            throw std::runtime_error("unknown materialization recipe kind");
        }

        if (!requiredFilesPresent(tempDir)) {
            throw std::runtime_error(
                "materialized segment is missing required tifxyz files");
        }
        auto updatedOrigin = *origin;
        updatedOrigin["cache_state"] =
            cacheStateName(OpenDataSegmentCacheState::Current);
        updatedOrigin["materialized_at_utc"] = nowUtcIso();
        if (!materializedInkDetections.empty()) {
            writeStringAtomic(tempDir / "ink-detections.json",
                              materializedInkDetections.dump(2));
            updatedOrigin["ink_detections"] = materializedInkDetections;
        }
        writeStringAtomic(tempDir / "catalog-origin.json",
                          updatedOrigin.dump(2));
        publishSegmentDirectory(tempDir, segmentDir);
        result.success = true;
        result.materializedSegments = 1;
        return result;
    } catch (const std::exception& e) {
        std::filesystem::remove_all(tempDir, ec);
        result.message = e.what();
        result.failedSegments = 1;
        return result;
    }
}

bool refreshMaterializedSegment(
    const std::filesystem::path& segmentDir,
    const PlaceholderPreparer& preparePlaceholder,
    std::string* errorOut)
{
    const auto stagingDir = makeTempSegmentDir(segmentDir);
    std::error_code error;
    std::filesystem::remove_all(stagingDir, error);

    if (!preparePlaceholder(stagingDir, errorOut)) {
        std::filesystem::remove_all(stagingDir, error);
        return false;
    }

    const auto materialized = materializeOne(stagingDir);
    if (!materialized.success) {
        std::filesystem::remove_all(stagingDir, error);
        if (errorOut) {
            *errorOut = materialized.message;
        }
        return false;
    }

    try {
        publishSegmentDirectory(stagingDir, segmentDir);
        return true;
    } catch (const std::exception& exception) {
        std::filesystem::remove_all(stagingDir, error);
        if (errorOut) {
            *errorOut = exception.what();
        }
        return false;
    }
}

void markOrphanedEntries(const std::filesystem::path& segmentsRoot,
                         const std::set<std::string>& expectedDirNames)
{
    std::error_code ec;
    if (!std::filesystem::is_directory(segmentsRoot, ec)) {
        return;
    }
    for (const auto& entry : std::filesystem::directory_iterator(segmentsRoot, ec)) {
        if (ec || !entry.is_directory()) {
            continue;
        }
        const auto name = entry.path().filename().string();
        if (expectedDirNames.find(name) != expectedDirNames.end()) {
            continue;
        }
        if (isOpenDataCatalogSegmentDirectory(entry.path())) {
            try {
                writeCatalogOriginState(entry.path(), OpenDataSegmentCacheState::Orphaned);
            } catch (...) {
            }
        }
    }
}

void reportProgress(const OpenDataSampleProgressCallback& callback,
                    const OpenDataSampleDownloadProgress& progress) noexcept
{
    if (!callback) {
        return;
    }
    try {
        callback(progress);
    } catch (...) {
    }
}

} // namespace

bool isOpenDataSegmentPlaceholder(const std::filesystem::path& segmentDir)
{
    if (requiredFilesPresent(segmentDir)) {
        return false;
    }
    const auto origin = readCatalogOrigin(segmentDir);
    return origin && origin->is_object() &&
           origin->contains(kMaterializationKey) &&
           (*origin)[kMaterializationKey].is_object();
}

OpenDataSegmentMaterializationResult materializeOpenDataSegment(
    const std::filesystem::path& segmentDir)
{
    return materializeOne(segmentDir);
}

OpenDataSegmentMaterializationResult materializeOpenDataSegmentFolder(
    const std::filesystem::path& segmentsRoot,
    const OpenDataSegmentMaterializationProgress& progressCallback)
{
    OpenDataSegmentMaterializationResult result;
    std::vector<std::filesystem::path> pending;
    if (isOpenDataSegmentPlaceholder(segmentsRoot)) {
        pending.push_back(segmentsRoot);
    } else {
        std::error_code ec;
        std::filesystem::recursive_directory_iterator it(segmentsRoot, ec);
        const std::filesystem::recursive_directory_iterator end;
        while (!ec && it != end) {
            if (it->is_directory() &&
                isOpenDataSegmentPlaceholder(it->path())) {
                pending.push_back(it->path());
                it.disable_recursion_pending();
            }
            it.increment(ec);
        }
        if (ec) {
            result.message = ec.message();
            result.failedSegments = 1;
            return result;
        }
    }
    std::sort(pending.begin(), pending.end());
    if (pending.empty()) {
        result.success = true;
        result.alreadyMaterialized = true;
        return result;
    }

    std::atomic_size_t next{0};
    std::atomic_int completed{0};
    std::mutex resultMutex;
    std::mutex callbackMutex;
    const int total = static_cast<int>(pending.size());
    const auto hardware = std::thread::hardware_concurrency();
    const std::size_t workerCount = std::min<std::size_t>(
        pending.size(), std::max<std::size_t>(
                            1, std::min<std::size_t>(
                                   hardware == 0 ? 4 : hardware, 8)));
    auto report = [&](const std::filesystem::path& path,
                      const std::string& status) {
        if (!progressCallback) return;
        std::lock_guard<std::mutex> lock(callbackMutex);
        progressCallback(completed.load(std::memory_order_relaxed),
                         total, path, status);
    };
    auto worker = [&]() {
        for (;;) {
            const auto index = next.fetch_add(1, std::memory_order_relaxed);
            if (index >= pending.size()) return;
            const auto& path = pending[index];
            report(path, "starting");
            const auto one = materializeOne(path);
            {
                std::lock_guard<std::mutex> lock(resultMutex);
                result.materializedSegments += one.materializedSegments;
                result.failedSegments += one.failedSegments;
                if (!one.success && !one.message.empty()) {
                    if (!result.message.empty()) result.message += '\n';
                    result.message += path.filename().string() + ": " +
                                      one.message;
                }
            }
            completed.fetch_add(1, std::memory_order_relaxed);
            report(path, one.success ? "done" : "failed");
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (std::size_t i = 0; i < workerCount; ++i) {
        workers.emplace_back(worker);
    }
    for (auto& workerThread : workers) workerThread.join();
    result.success = result.failedSegments == 0;
    return result;
}

std::vector<OpenDataSegmentRepresentation>
classifyOpenDataSegmentRepresentations(const OpenDataSample& sample,
                                       const OpenDataSegment& segment)
{
    std::vector<OpenDataSegmentRepresentation> result;
    std::set<std::string> volumeIds;
    for (const auto& volume : sample.volumes) {
        if (!volume.id.empty())
            volumeIds.insert(volume.id);
    }

    const auto hashUrl = [](std::string_view value) {
        std::uint64_t hash = 14695981039346656037ULL;
        for (const unsigned char c : value) {
            hash ^= c;
            hash *= 1099511628211ULL;
        }
        std::ostringstream out;
        out << std::hex << std::setfill('0') << std::setw(16) << hash;
        return out.str();
    };
    const auto levelForDownscale = [&]() -> std::optional<int> {
        const double downscale = originalVolumeDownscale(segment);
        for (int level = 0; level <= 5; ++level) {
            if (std::abs(downscale - std::ldexp(1.0, level)) <=
                1.0e-9 * std::max(1.0, std::abs(downscale))) {
                return level;
            }
        }
        return std::nullopt;
    }();

    const std::string sourceId = sourceVolumeIdForSegment(segment);
    if (sourceId.empty() || volumeIds.find(sourceId) == volumeIds.end()) {
        return result;
    }

    const OpenDataArtifact* canonicalAuthored = nullptr;
    for (const auto* preferredType :
         {"tifxyz-flattened", "tifxyz", "tifxyz-normalized"}) {
        const auto it = std::find_if(
            segment.artifacts.begin(), segment.artifacts.end(),
            [&](const OpenDataArtifact& artifact) {
                return lowerCopy(artifact.type) == preferredType &&
                       !artifactUrl(artifact).empty();
            });
        if (it != segment.artifacts.end()) {
            canonicalAuthored = &*it;
            break;
        }
    }

    std::map<std::string, const OpenDataArtifact*> publishedByTarget;
    for (const auto& artifact : segment.artifacts) {
        const auto type = lowerCopy(artifact.type);
        if (type != "tifxyz-transformed" || artifactUrl(artifact).empty() ||
            !artifact.targetVolumeId || artifact.targetVolumeId->empty() ||
            volumeIds.find(*artifact.targetVolumeId) == volumeIds.end()) {
            continue;
        }
        publishedByTarget.emplace(*artifact.targetVolumeId, &artifact);
    }

    const std::string lineage = segmentStableId(segment);
    const auto makePublished = [&](const std::string& targetId,
                                   const OpenDataArtifact& artifact,
                                   bool canonicalSource) {
        OpenDataSegmentRepresentation published;
        published.artifact = &artifact;
        published.kind = OpenDataSegmentRepresentationKind::PublishedTransformed;
        published.canonicalSource = canonicalSource;
        published.coordinateVolumeId = targetId;
        published.sourceCoordinateLevel = 0;
        published.coordinateSpace = sample.id + "/" + targetId + "@L0";
        published.representationId = lineage + "-published-" +
            safePathComponent(targetId) + "-L0-" + hashUrl(artifactUrl(artifact));
        return published;
    };

    if (const auto publishedSource = publishedByTarget.find(sourceId);
        publishedSource != publishedByTarget.end()) {
        result.push_back(makePublished(sourceId, *publishedSource->second, true));
    } else if (canonicalAuthored && levelForDownscale) {
        OpenDataSegmentRepresentation canonical;
        canonical.artifact = canonicalAuthored;
        canonical.kind = OpenDataSegmentRepresentationKind::DerivedNative;
        canonical.canonicalSource = true;
        canonical.coordinateVolumeId = sourceId;
        canonical.sourceCoordinateLevel = 0;
        canonical.coordinateSpace = sample.id + "/" + sourceId + "@L0";
        canonical.representationId = lineage + "-derived-native-L0-" +
            hashUrl(artifactUrl(*canonicalAuthored));
        result.push_back(std::move(canonical));
    }

    for (const auto& [targetId, artifact] : publishedByTarget) {
        if (targetId == sourceId) {
            continue;
        }
        result.push_back(makePublished(targetId, *artifact, false));
    }
    return result;
}

std::filesystem::path openDataSegmentRepresentationCacheRoot(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegmentRepresentation& representation)
{
    auto root = openDataSegmentCacheRoot(
        remoteCacheRoot, sample, representation.coordinateVolumeId);
    switch (representation.kind) {
        case OpenDataSegmentRepresentationKind::Authored:
            return root / ("authored-L" +
                           std::to_string(representation.sourceCoordinateLevel));
        case OpenDataSegmentRepresentationKind::DerivedNative:
            return root;
        case OpenDataSegmentRepresentationKind::PublishedTransformed:
            return root / "published-L0";
    }
    return root / "unknown";
}

std::filesystem::path openDataSegmentRepresentationCacheDirectory(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment& segment,
    const OpenDataSegmentRepresentation& representation)
{
    if (representation.kind == OpenDataSegmentRepresentationKind::DerivedNative) {
        return openDataSegmentRepresentationCacheRoot(
                   remoteCacheRoot, sample, representation) /
               safePathComponent(segment.id);
    }
    return openDataSegmentRepresentationCacheRoot(remoteCacheRoot, sample, representation) /
           safePathComponent(representation.representationId);
}

std::filesystem::path openDataCanonicalSegmentCacheDirectory(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment& segment)
{
    const auto representations =
        classifyOpenDataSegmentRepresentations(sample, segment);
    const auto canonical = std::find_if(
        representations.begin(), representations.end(),
        [](const auto& representation) { return representation.canonicalSource; });
    if (canonical == representations.end()) {
        return {};
    }
    return openDataSegmentRepresentationCacheDirectory(
        remoteCacheRoot, sample, segment, *canonical);
}

const char* cacheStateName(OpenDataSegmentCacheState state) noexcept
{
    switch (state) {
        case OpenDataSegmentCacheState::Missing: return "missing";
        case OpenDataSegmentCacheState::Current: return "current";
        case OpenDataSegmentCacheState::Incomplete: return "incomplete";
        case OpenDataSegmentCacheState::Stale: return "stale";
        case OpenDataSegmentCacheState::Orphaned: return "orphaned";
    }
    return "unknown";
}

std::filesystem::path openDataSegmentCacheRoot(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const std::string& sourceVolumeId)
{
    const auto sampleComponent = safePathComponent(sample.id.empty() ? "sample" : sample.id);
    const auto sourceComponent = safePathComponent(sourceVolumeId.empty() ? "volume" : sourceVolumeId);
    return remoteCacheRoot / "open_data" / "segments" /
           sampleComponent / sourceComponent;
}

std::filesystem::path openDataSegmentCacheDirectory(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment& segment)
{
    return openDataSegmentCacheRoot(remoteCacheRoot, sample, sourceVolumeIdForSegment(segment)) /
           safePathComponent(segment.id);
}

std::filesystem::path openDataTransformedSegmentCacheRoot(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const std::string& sourceVolumeId,
    const std::string& targetVolumeId)
{
    return openDataSegmentCacheRoot(remoteCacheRoot, sample, targetVolumeId) /
           ("generated-from-" +
            safePathComponent(sourceVolumeId.empty() ? "volume" : sourceVolumeId) +
            "-L0");
}

std::filesystem::path openDataTransformedSegmentCacheDirectory(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment& segment,
    const std::string& targetVolumeId)
{
    return openDataTransformedSegmentCacheRoot(
               remoteCacheRoot, sample, sourceVolumeIdForSegment(segment), targetVolumeId) /
           safePathComponent(segment.id);
}

std::filesystem::path openDataEditableSegmentRoot(
    const std::filesystem::path& remoteCacheRoot,
    const std::string& sampleId)
{
    return remoteCacheRoot / "open_data" / "editable" /
           safePathComponent(sampleId.empty() ? "sample" : sampleId);
}

std::filesystem::path openDataPatchesRoot(
    const std::filesystem::path& remoteCacheRoot,
    const std::string& sampleId)
{
    return remoteCacheRoot / "open_data" / "segments" /
           safePathComponent(sampleId.empty() ? "sample" : sampleId) / "patches";
}

std::size_t manualOpenDataSegmentCount(
    const std::filesystem::path& remoteCacheRoot,
    const std::string& sampleId)
{
    const auto sampleRoot = remoteCacheRoot / "open_data" / "segments" /
                            safePathComponent(sampleId.empty() ? "sample" : sampleId);
    std::error_code ec;
    std::filesystem::recursive_directory_iterator entries(
        sampleRoot,
        std::filesystem::directory_options::skip_permission_denied,
        ec);
    if (ec) {
        return 0;
    }

    std::size_t count = 0;
    const std::filesystem::recursive_directory_iterator end;
    while (entries != end) {
        const auto& entry = *entries;
        std::error_code entryEc;
        if (entry.is_directory(entryEc) && !entryEc) {
            const auto name = entry.path().filename().string();
            if (name == "backups" || (!name.empty() && name.front() == '.')) {
                entries.disable_recursion_pending();
            } else {
                const bool isSegment =
                    std::filesystem::is_regular_file(entry.path() / "meta.json", entryEc);
                entryEc.clear();
                const bool isCatalogSegment = std::filesystem::is_regular_file(
                    entry.path() / "catalog-origin.json", entryEc);
                if (isSegment) {
                    if (!isCatalogSegment) {
                        ++count;
                    }
                    entries.disable_recursion_pending();
                }
            }
        }
        entries.increment(ec);
        if (ec) {
            break;
        }
    }
    return count;
}

OpenDataSegmentCacheState cacheStateForSegment(
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSample& sample,
    const OpenDataSegment& segment)
{
    const auto representations =
        classifyOpenDataSegmentRepresentations(sample, segment);
    const auto canonical = std::find_if(
        representations.begin(), representations.end(),
        [](const auto& representation) { return representation.canonicalSource; });
    if (canonical == representations.end() || !canonical->artifact) {
        return OpenDataSegmentCacheState::Orphaned;
    }
    const auto dir = openDataSegmentRepresentationCacheDirectory(
        remoteCacheRoot, sample, segment, *canonical);
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) {
        return OpenDataSegmentCacheState::Missing;
    }
    if (isOpenDataSegmentPlaceholder(dir)) {
        return OpenDataSegmentCacheState::Missing;
    }
    if (!requiredFilesPresent(dir)) {
        return OpenDataSegmentCacheState::Incomplete;
    }

    const auto origin = readCatalogOrigin(dir);
    if (!origin) {
        return OpenDataSegmentCacheState::Current;
    }
    if (!originMatches(*origin, sample, segment, *canonical->artifact)) {
        return OpenDataSegmentCacheState::Stale;
    }
    const auto it = origin->find("cache_state");
    if (it != origin->end() && it->is_string()) {
        const auto state = lowerCopy(it->get<std::string>());
        if (state == "orphaned") return OpenDataSegmentCacheState::Orphaned;
        if (state == "stale") return OpenDataSegmentCacheState::Stale;
        if (state == "incomplete") return OpenDataSegmentCacheState::Incomplete;
    }
    return OpenDataSegmentCacheState::Current;
}

OpenDataSegmentCacheReconcileResult attachExistingOpenDataSegmentCaches(
    VolumePkg& pkg,
    const OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot)
{
    OpenDataSegmentCacheReconcileResult result;
    if (sample.tifxyzSegmentCount() == 0) {
        return result;
    }

    result.supportedTifxyzSegments = static_cast<int>(sample.tifxyzSegmentCount());
    if (remoteCacheRoot.empty()) {
        result.skippedTifxyzSegments = result.supportedTifxyzSegments;
        result.messages.push_back("Skipped cached tifxyz segment discovery: no remote cache directory configured.");
        return result;
    }

    std::vector<const OpenDataSegment*> tifxyzSegments;
    tifxyzSegments.reserve(sample.segments.size());
    for (const auto& segment : sample.segments) {
        if (segment.hasTifxyz()) {
            tifxyzSegments.push_back(&segment);
        }
    }

    auto attachEntry = [&](const std::string& location,
                           std::vector<std::string> tags,
                           const std::string& label,
                           const std::string& cacheDescription,
                           int& failedCount) {
        const std::string failurePrefix =
            "Failed to attach " + cacheDescription + " for " + label;
        try {
            if (pkg.addSegmentsEntry(location, tags)) {
                ++result.attachedSegmentEntries;
            } else if (hasSegmentEntry(pkg, location)) {
                pkg.reconcileSegmentsEntryTags(
                    location, tags,
                    {"vc-open-data-source-coordinate-level:",
                     "vc-open-data-coordinate-space:",
                     "vc-open-data-source-volume-id:",
                     "vc-open-data-target-volume-id:",
                     "vc-open-data-segment-representation:"});
            } else {
                ++failedCount;
                result.messages.push_back(failurePrefix + ".");
            }
        } catch (const std::exception& e) {
            ++failedCount;
            result.messages.push_back(failurePrefix + ": " + e.what());
        } catch (...) {
            ++failedCount;
            result.messages.push_back(failurePrefix + ": unknown error.");
        }
    };

    std::map<std::filesystem::path, OpenDataSegmentRepresentation>
        cachedRepresentationRoots;
    for (const auto* segment : tifxyzSegments) {
        for (const auto& representation :
             classifyOpenDataSegmentRepresentations(sample, *segment)) {
            const auto dir = openDataSegmentRepresentationCacheDirectory(
                remoteCacheRoot, sample, *segment, representation);
            const bool materialized = requiredFilesPresent(dir);
            if (!materialized && !isOpenDataSegmentPlaceholder(dir))
                continue;
            if (representation.canonicalSource && materialized) {
                ++result.cachedTifxyzSegments;
            }
            cachedRepresentationRoots.emplace(
                openDataSegmentRepresentationCacheRoot(
                    remoteCacheRoot, sample, representation),
                representation);
        }
    }
    for (const auto& [root, representation] : cachedRepresentationRoots) {
        std::string kindTag;
        switch (representation.kind) {
            case OpenDataSegmentRepresentationKind::Authored:
                kindTag = "authored";
                break;
            case OpenDataSegmentRepresentationKind::DerivedNative:
                kindTag = "derived-native";
                break;
            case OpenDataSegmentRepresentationKind::PublishedTransformed:
                kindTag = "published-transformed";
                break;
        }
        std::vector<std::string> tags{
            "open-data",
            "immutable",
            "vc-open-data-segment-representation:" + kindTag,
            "vc-open-data-source-coordinate-level:" +
                std::to_string(representation.sourceCoordinateLevel),
            "vc-open-data-coordinate-space:" + representation.coordinateSpace,
        };
        if (representation.kind == OpenDataSegmentRepresentationKind::PublishedTransformed)
            tags.push_back("vc-open-data-target-volume-id:" +
                           representation.coordinateVolumeId);
        else
            tags.push_back("vc-open-data-source-volume-id:" +
                           representation.coordinateVolumeId);
        if (representation.canonicalSource) {
            tags.push_back("vc-open-data-canonical-source");
            if (representation.kind ==
                OpenDataSegmentRepresentationKind::PublishedTransformed) {
                tags.push_back("vc-open-data-source-volume-id:" +
                               representation.coordinateVolumeId);
            }
        }
        attachEntry(root.string(), std::move(tags), representation.coordinateSpace,
                    "coordinate-specific tifxyz representation",
                    result.failedTifxyzSegments);
    }

    std::set<std::string> targetVolumeIds;
    for (const auto& volume : sample.volumes) {
        if (!volume.id.empty()) {
            targetVolumeIds.insert(volume.id);
        }
    }

    std::map<std::pair<std::string, std::string>, int> transformedByRoute;
    for (const auto& targetVolumeId : targetVolumeIds) {
        for (const auto* segment : tifxyzSegments) {
            const auto sourceVolumeId = sourceVolumeIdForSegment(*segment);
            if (sourceVolumeId.empty() || sourceVolumeId == targetVolumeId) {
                continue;
            }
            const auto representations =
                classifyOpenDataSegmentRepresentations(sample, *segment);
            const bool hasPublishedTarget = std::any_of(
                representations.begin(), representations.end(),
                [&](const auto& representation) {
                    return representation.kind ==
                               OpenDataSegmentRepresentationKind::PublishedTransformed &&
                           representation.coordinateVolumeId == targetVolumeId;
                });
            if (hasPublishedTarget) {
                continue;
            }
            const auto matrix = findVolumeTransformMatrix(sample, sourceVolumeId, targetVolumeId);
            if (!matrix) {
                continue;
            }
            const auto transformedDir = openDataTransformedSegmentCacheDirectory(
                remoteCacheRoot, sample, *segment, targetVolumeId);
            const bool materialized = requiredFilesPresent(transformedDir);
            if (!materialized &&
                !isOpenDataSegmentPlaceholder(transformedDir)) {
                continue;
            }
            const auto origin = readCatalogOrigin(transformedDir);
            if (materialized && origin && !transformedOriginMatches(
                              *origin, sample, *segment, sourceVolumeId, targetVolumeId, *matrix)) {
                continue;
            }
            if (materialized) ++result.transformedTifxyzSegments;
            ++transformedByRoute[{sourceVolumeId, targetVolumeId}];
        }
    }

    for (const auto& [route, transformedCount] : transformedByRoute) {
        if (transformedCount <= 0) {
            continue;
        }
        const auto& sourceVolumeId = route.first;
        const auto& targetVolumeId = route.second;
        attachEntry(openDataTransformedSegmentCacheRoot(
                        remoteCacheRoot, sample, sourceVolumeId, targetVolumeId).string(),
                    {"open-data",
                     "immutable",
                     "open-data-transformed",
                     "vc-open-data-segment-representation:generated-native-transform",
                     "vc-open-data-source-volume-id:" + sourceVolumeId,
                     "vc-open-data-target-volume-id:" + targetVolumeId,
                     "vc-open-data-source-coordinate-level:0",
                     "vc-open-data-coordinate-space:" + sample.id + "/" +
                         targetVolumeId + "@L0"},
                    sourceVolumeId + " to " + targetVolumeId,
                    "transformed tifxyz segment directory",
                    result.failedTransformedTifxyzSegments);
    }

    return result;
}

bool isOpenDataCatalogSegmentDirectory(const std::filesystem::path& segmentDir)
{
    return std::filesystem::is_regular_file(segmentDir / "catalog-origin.json");
}

std::vector<OpenDataInkDetectionEntry> cachedInkDetectionsForSegmentDirectory(
    const std::filesystem::path& segmentDir)
{
    std::vector<OpenDataInkDetectionEntry> out;
    if (!std::filesystem::is_directory(segmentDir)) {
        return out;
    }

    for (const auto& record : readInkDetectionRecords(segmentDir)) {
        auto stringField = [&](const char* key) -> std::string {
            const auto it = record.find(key);
            return it != record.end() && it->is_string() ? it->get<std::string>() : std::string{};
        };
        std::string localFile = stringField("local_file");
        if (localFile.empty()) {
            continue;
        }
        std::filesystem::path localPath = std::filesystem::path(localFile);
        if (localPath.is_relative()) {
            localPath = segmentDir / localPath;
        }
        if (!isNonEmptyFile(localPath)) {
            continue;
        }
        OpenDataInkDetectionEntry entry;
        entry.label = stringField("label");
        entry.sampleId = stringField("sample_id");
        entry.segmentId = stringField("segment_id");
        entry.segmentLongId = stringField("segment_long_id");
        entry.artifactType = stringField("artifact_type");
        entry.sourceUrl = stringField("resolved_http_url");
        if (entry.sourceUrl.empty()) {
            entry.sourceUrl = stringField("original_source_uri");
        }
        entry.localPath = std::move(localPath);
        if (entry.label.empty()) {
            entry.label = entry.segmentId.empty() ? entry.localPath.filename().string()
                                                  : entry.segmentId;
        }
        out.push_back(std::move(entry));
    }
    return out;
}

std::filesystem::path defaultEditableCopyPathForCatalogSegment(
    const std::filesystem::path& catalogSegmentDir,
    const std::filesystem::path& activeSegmentsRoot)
{
    const auto currentSegmentsRoot = activeSegmentsRoot.empty()
        ? catalogSegmentDir.parent_path()
        : activeSegmentsRoot;
    auto editableRootName = currentSegmentsRoot.filename().string();
    if (editableRootName.empty()) {
        editableRootName = "segments";
    }
    editableRootName += "_editable";

    auto baseName = catalogSegmentDir.filename().string();
    if (baseName.empty()) {
        baseName = "segment";
    }
    return currentSegmentsRoot.parent_path() / editableRootName /
           baseName;
}

void copyCatalogSegmentToEditableDirectory(
    const std::filesystem::path& catalogSegmentDir,
    const std::filesystem::path& editableSegmentDir)
{
    if (!isOpenDataCatalogSegmentDirectory(catalogSegmentDir) ||
        !requiredFilesPresent(catalogSegmentDir)) {
        throw std::runtime_error("catalog segment is not a complete open-data tifxyz directory");
    }
    std::error_code ec;
    if (std::filesystem::exists(editableSegmentDir, ec)) {
        if (!std::filesystem::is_directory(editableSegmentDir, ec)) {
            throw std::runtime_error("editable destination exists and is not a directory");
        }
        if (!std::filesystem::is_empty(editableSegmentDir, ec) &&
            !requiredFilesPresent(editableSegmentDir)) {
            throw std::runtime_error("editable destination is not empty and is not a tifxyz segment");
        }
    } else {
        std::filesystem::create_directories(editableSegmentDir);
    }

    if (std::filesystem::is_empty(editableSegmentDir, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(catalogSegmentDir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            if (entry.path().filename() == "catalog-origin.json") {
                continue;
            }
            std::filesystem::copy_file(
                entry.path(),
                editableSegmentDir / entry.path().filename(),
                std::filesystem::copy_options::overwrite_existing);
        }
    }
    if (!requiredFilesPresent(editableSegmentDir)) {
        throw std::runtime_error("editable destination is missing required tifxyz files");
    }
}

OpenDataSegmentCacheReconcileResult reconcileOpenDataSampleSegments(
    VolumePkg& pkg,
    const OpenDataSample& sample,
    const std::filesystem::path& remoteCacheRoot,
    const OpenDataSampleProgressCallback& progressCallback,
    bool forceRefresh)
{
    OpenDataSegmentCacheReconcileResult result;
    if (sample.tifxyzSegmentCount() == 0) {
        return result;
    }

    result.supportedTifxyzSegments = static_cast<int>(sample.tifxyzSegmentCount());

    if (remoteCacheRoot.empty()) {
        result.skippedTifxyzSegments += result.supportedTifxyzSegments;
        result.messages.push_back("Skipped tifxyz segments: no remote cache directory configured.");
        return result;
    }

    struct RepresentationTask {
        const OpenDataSegment* segment = nullptr;
        OpenDataSegmentRepresentation representation;
    };

    std::vector<const OpenDataSegment*> tifxyzSegments;
    std::vector<RepresentationTask> representationTasks;
    tifxyzSegments.reserve(sample.segments.size());
    std::map<std::string, std::set<std::string>> expectedDirNamesBySource;
    for (const auto& volume : sample.volumes) {
        if (!volume.id.empty()) {
            expectedDirNamesBySource.try_emplace(volume.id);
        }
    }
    for (const auto& segment : sample.segments) {
        if (!segment.hasTifxyz()) {
            continue;
        }
        tifxyzSegments.push_back(&segment);
        auto representations =
            classifyOpenDataSegmentRepresentations(sample, segment);
        if (representations.empty()) {
            ++result.skippedTifxyzSegments;
            result.messages.push_back(
                "Skipped " + segmentLabel(segment) +
                ": no canonical manifest source or valid published target.");
            continue;
        }
        for (auto& representation : representations) {
            if (representation.canonicalSource &&
                representation.kind ==
                    OpenDataSegmentRepresentationKind::DerivedNative) {
                expectedDirNamesBySource[representation.coordinateVolumeId].insert(
                    safePathComponent(segment.id));
            }
            representationTasks.push_back({&segment, std::move(representation)});
        }
    }
    if (representationTasks.empty()) {
        return result;
    }

    std::vector<char> representationSucceeded(representationTasks.size(), 0);
    int completedRepresentations = 0;
    for (std::size_t i = 0; i < representationTasks.size(); ++i) {
        const auto& task = representationTasks[i];
        const auto& representation = task.representation;
        const auto segmentDir = openDataSegmentRepresentationCacheDirectory(
            remoteCacheRoot, sample, *task.segment, representation);
        std::string error;
        const bool prepared = prepareRepresentationPlaceholder(
            sample, *task.segment, representation, segmentDir,
            forceRefresh, &error);
        if (prepared) {
            representationSucceeded[i] = 1;
            if (representation.canonicalSource &&
                requiredFilesPresent(segmentDir)) {
                ++result.cachedTifxyzSegments;
            }
        } else {
            ++result.failedTifxyzSegments;
            result.messages.push_back(
                "Failed to prepare " + segmentLabel(*task.segment) +
                " for " + representation.coordinateSpace + ": " + error);
        }
        ++completedRepresentations;
        OpenDataSampleDownloadProgress progress;
        progress.totalSegments =
            static_cast<int>(representationTasks.size());
        progress.completedSegments = completedRepresentations;
        progress.failedSegments = result.failedTifxyzSegments;
        progress.segmentId = segmentLabel(*task.segment);
        progress.status = prepared ? "placeholder-ready" : "placeholder-failed";
        reportProgress(progressCallback, progress);
    }
    for (const auto& [sourceVolumeId, expectedDirNames] : expectedDirNamesBySource) {
        markOrphanedEntries(
            openDataSegmentCacheRoot(remoteCacheRoot, sample, sourceVolumeId),
            expectedDirNames);
    }
    std::map<std::filesystem::path, OpenDataSegmentRepresentation> preparedRoots;
    int preparedRepresentations = 0;
    for (std::size_t i = 0; i < representationTasks.size(); ++i) {
        if (!representationSucceeded[i]) {
            continue;
        }
        const auto& representation = representationTasks[i].representation;
        ++preparedRepresentations;
        preparedRoots.emplace(
            openDataSegmentRepresentationCacheRoot(
                remoteCacheRoot, sample, representation),
            representation);
    }
    std::set<std::string> aggregateVolumeIds;
    for (const auto& volume : sample.volumes) {
        if (!volume.id.empty()) aggregateVolumeIds.insert(volume.id);
    }
    for (const auto& [root, representation] : preparedRoots) {
        (void)root;
        if (!representation.coordinateVolumeId.empty()) {
            aggregateVolumeIds.insert(representation.coordinateVolumeId);
        }
    }
    std::set<std::string> expectedSegmentEntryLocations;
    for (const auto& volumeId : aggregateVolumeIds) {
        const auto root = openDataSegmentCacheRoot(
            remoteCacheRoot, sample, volumeId);
        std::error_code ec;
        std::filesystem::create_directories(root, ec);
        if (ec) {
            result.messages.push_back(
                "Failed to prepare segment volume folder " + volumeId +
                ": " + ec.message());
            continue;
        }
        const auto location = root.string();
        expectedSegmentEntryLocations.insert(location);
        const std::vector<std::string> tags{
            "open-data",
            "immutable",
            "vc-open-data-segment-aggregate",
            "vc-open-data-target-volume-id:" + volumeId,
            "vc-open-data-source-coordinate-level:0",
            "vc-open-data-coordinate-space:" + sample.id + "/" +
                volumeId + "@L0",
        };
        if (pkg.addSegmentsEntry(location, tags)) {
            ++result.attachedSegmentEntries;
        } else if (hasSegmentEntry(pkg, location)) {
            pkg.reconcileSegmentsEntryTags(
                location, tags,
                {"vc-open-data-source-coordinate-level:",
                 "vc-open-data-coordinate-space:",
                 "vc-open-data-source-volume-id:",
                 "vc-open-data-target-volume-id:",
                 "vc-open-data-segment-representation:"});
        }
    }

    if (preparedRepresentations <= 0) {
        return result;
    }

    std::set<std::string> targetVolumeIds;
    for (const auto& volume : sample.volumes) {
        if (!volume.id.empty()) {
            targetVolumeIds.insert(volume.id);
        }
    }

    struct TransformTask {
        const OpenDataSegment* segment = nullptr;
        std::string targetVolumeId;
    };
    std::vector<TransformTask> transformTasks;
    for (const auto& targetVolumeId : targetVolumeIds) {
        for (const auto* segment : tifxyzSegments) {
            const auto sourceVolumeId = sourceVolumeIdForSegment(*segment);
            if (sourceVolumeId.empty() || sourceVolumeId == targetVolumeId) {
                continue;
            }
            const auto representations =
                classifyOpenDataSegmentRepresentations(sample, *segment);
            const bool hasPublishedTarget = std::any_of(
                representations.begin(), representations.end(),
                [&](const auto& representation) {
                    return representation.kind ==
                               OpenDataSegmentRepresentationKind::PublishedTransformed &&
                           representation.coordinateVolumeId == targetVolumeId;
                });
            if (hasPublishedTarget ||
                !findVolumeTransformMatrix(sample, sourceVolumeId, targetVolumeId)) {
                continue;
            }
            transformTasks.push_back({segment, targetVolumeId});
        }
    }

    int completedTransforms = 0;
    for (const auto& task : transformTasks) {
        const auto sourceSegmentDir = openDataCanonicalSegmentCacheDirectory(
            remoteCacheRoot, sample, *task.segment);
        const auto targetSegmentDir = openDataTransformedSegmentCacheDirectory(
            remoteCacheRoot, sample, *task.segment, task.targetVolumeId);
        std::string error;
        if (prepareGeneratedPlaceholder(
                sample, *task.segment, sourceSegmentDir, targetSegmentDir,
                task.targetVolumeId, forceRefresh, &error)) {
            if (requiredFilesPresent(targetSegmentDir)) {
                ++result.transformedTifxyzSegments;
            }
        } else {
            ++result.failedTransformedTifxyzSegments;
            result.messages.push_back(
                "Failed to prepare transform " +
                segmentLabel(*task.segment) + " to " +
                task.targetVolumeId + ": " + error);
        }
        ++completedTransforms;
        OpenDataSampleDownloadProgress progress;
        progress.totalSegments = static_cast<int>(
            representationTasks.size() + transformTasks.size());
        progress.completedSegments = completedRepresentations +
                                     completedTransforms;
        progress.failedSegments = result.failedTifxyzSegments +
                                  result.failedTransformedTifxyzSegments;
        progress.segmentId = segmentLabel(*task.segment);
        progress.fileName = task.targetVolumeId;
        progress.status = error.empty() ? "placeholder-transform-ready"
                                        : "placeholder-transform-failed";
        reportProgress(progressCallback, progress);
    }

    const auto sampleSegmentsRoot = remoteCacheRoot / "open_data" / "segments" /
        safePathComponent(sample.id.empty() ? "sample" : sample.id);
    const auto sampleSegmentsPrefix = sampleSegmentsRoot.string() +
        std::string(1, std::filesystem::path::preferred_separator);
    const auto existingEntries = pkg.segmentEntries();
    for (const auto& entry : existingEntries) {
        const bool immutable =
            std::find(entry.tags.begin(), entry.tags.end(), "immutable") !=
            entry.tags.end();
        if (!immutable ||
            entry.location.rfind(sampleSegmentsPrefix, 0) != 0 ||
            expectedSegmentEntryLocations.contains(entry.location)) {
            continue;
        }
        if (pkg.removeEntry(entry.location)) {
            result.messages.push_back(
                "Removed stale open-data segment root: " + entry.location);
        }
    }

    OpenDataSampleDownloadProgress finishedProgress;
    finishedProgress.totalSegments = static_cast<int>(
        representationTasks.size() + transformTasks.size());
    finishedProgress.completedSegments = completedRepresentations +
                                         completedTransforms;
    finishedProgress.failedSegments = result.failedTifxyzSegments +
                                      result.failedTransformedTifxyzSegments;
    finishedProgress.status = "placeholders-finished";
    reportProgress(progressCallback, finishedProgress);

    return result;
}

} // namespace vc3d::opendata
