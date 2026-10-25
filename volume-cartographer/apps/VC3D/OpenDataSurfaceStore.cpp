#include "OpenDataSurfaceStore.hpp"

#include "vc/core/util/AffineTransform.hpp"
#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/GeometryIngest.hpp>
#include <vc_object_store/objects/ObjectStore.hpp>

#include <QSaveFile>
#include <QString>
#include <QUuid>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string_view>

namespace vc3d::opendata {
namespace {

namespace fs = std::filesystem;
using vc::objects::ObjectStore;

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

std::string artifactUrl(const vcr::catalog::OpenDataArtifact& artifact)
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

bool isSupportedInkImageArtifact(const vcr::catalog::OpenDataArtifact& artifact)
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

std::string segmentLabel(const vcr::catalog::OpenDataSegment& segment)
{
    return segment.id.empty() ? std::string("<unnamed segment>") : segment.id;
}

std::string segmentStableId(const vcr::catalog::OpenDataSegment& segment)
{
    if (!segment.longId.empty()) {
        return segment.longId;
    }
    if (!segment.id.empty()) {
        return segment.id;
    }
    return safePathComponent(segment.suffix);
}

std::string surfaceTimestampFromCatalogDate(std::string_view date)
{
    const auto isDigit = [&](std::size_t index) {
        return index < date.size() &&
               std::isdigit(static_cast<unsigned char>(date[index]));
    };
    if ((date.size() == 14 || date.size() == 17) &&
        std::all_of(date.begin(), date.end(), [](unsigned char c) {
            return std::isdigit(c);
        })) {
        return std::string(date.substr(0, 14));
    }
    if (date.size() < 20 || date[4] != '-' || date[7] != '-' ||
        date[10] != 'T' || date[13] != ':' || date[16] != ':' ||
        !isDigit(0) || !isDigit(1) || !isDigit(2) || !isDigit(3) ||
        !isDigit(5) || !isDigit(6) || !isDigit(8) || !isDigit(9) ||
        !isDigit(11) || !isDigit(12) || !isDigit(14) || !isDigit(15) ||
        !isDigit(17) || !isDigit(18)) {
        return std::string(date);
    }

    if (date[19] == 'Z') {
        if (date.size() != 20) {
            return std::string(date);
        }
    } else if (date[19] == '.') {
        constexpr std::size_t fraction = 20;
        auto suffix = fraction;
        while (suffix < date.size() &&
               std::isdigit(static_cast<unsigned char>(date[suffix]))) {
            ++suffix;
        }
        if (suffix == fraction || suffix + 1 != date.size() ||
            date[suffix] != 'Z') {
            return std::string(date);
        }
    } else {
        return std::string(date);
    }

    std::string timestamp;
    timestamp.reserve(14);
    for (const auto [begin, count] : {
             std::pair<std::size_t, std::size_t>{0, 4},
             {5, 2}, {8, 2}, {11, 2}, {14, 2}, {17, 2}}) {
        timestamp.append(date.substr(begin, count));
    }

    return timestamp;
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

double originalVolumeDownscale(const vcr::catalog::OpenDataSegment& segment)
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

std::string derivedVolumeId(const vcr::catalog::OpenDataSegment& segment)
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

std::string sourceVolumeIdForSegment(const vcr::catalog::OpenDataSegment& segment)
{
    const auto derived = derivedVolumeId(segment);
    return derived.empty() ? segment.originalVolumeId : derived;
}

bool applyCatalogCreationMetadata(nlohmann::json& meta,
                                  const vcr::catalog::OpenDataSegment& segment)
{
    if (segment.createdAt.empty()) {
        return false;
    }
    const auto timestamp = surfaceTimestampFromCatalogDate(segment.createdAt);
    const bool changed =
        meta.value("date_last_modified", std::string{}) != timestamp ||
        meta.value("vc_open_data_creation_date", std::string{}) !=
            segment.createdAt;
    meta["date_last_modified"] = timestamp;
    meta["vc_open_data_creation_date"] = segment.createdAt;
    return changed;
}

void applyOpenDataMetadata(nlohmann::json& meta,
                           const std::string& baseUrl,
                           const vcr::catalog::OpenDataSegment& segment)
{
    meta["vc_open_data_tifxyz_url"] = baseUrl;
    meta["vc_open_data_segment_id"] = segment.id;
    meta["vc_open_data_segment_long_id"] = segment.longId;
    meta["vc_open_data_original_volume_id"] = segment.originalVolumeId;
    if (const auto derivedId = derivedVolumeId(segment); !derivedId.empty()) {
        meta["vc_open_data_derived_volume_id"] = derivedId;
    }
    meta["vc_open_data_original_volume_downscale"] = originalVolumeDownscale(segment);
    applyCatalogCreationMetadata(meta, segment);
}

const vcr::catalog::OpenDataVolume* findSampleVolume(const vcr::catalog::OpenDataSample& sample,
                                       const std::string& volumeId)
{
    const auto it = std::find_if(
        sample.volumes.begin(), sample.volumes.end(), [&](const auto& volume) {
            return volume.id == volumeId;
        });
    return it == sample.volumes.end() ? nullptr : &*it;
}

void applySourceCoordinateMetadata(nlohmann::json& meta,
                                   const vcr::catalog::OpenDataSample& sample,
                                   const std::string& volumeId,
                                   int coordinateLevel)
{
    const auto* volume = findSampleVolume(sample, volumeId);
    if (!volume || !volume->pixelSizeUm ||
        !std::isfinite(*volume->pixelSizeUm) || *volume->pixelSizeUm <= 0.0) {
        return;
    }
    const auto* source = vcr::catalog::preferredVolumeArtifact(*volume);
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

std::string inkDetectionLabel(const vcr::catalog::OpenDataSegment& segment,
                              const vcr::catalog::OpenDataArtifact& artifact,
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

nlohmann::json inkDetectionMaterializationRecipe(
    const vcr::catalog::OpenDataSample& sample,
    const vcr::catalog::OpenDataSegment& segment)
{
    std::vector<const vcr::catalog::OpenDataArtifact*> supportedArtifacts;
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

// The record meta of a representation: the catalog's creation metadata and
// the vc_open_data_* identity keys (coordinate space, source, lineage).
nlohmann::json surfaceRecordMeta(
    const vcr::catalog::OpenDataSample& sample,
    const vcr::catalog::OpenDataSegment& segment,
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
    for (const char* key : {"uuid", "name", "type", "format", "scale", "bbox", "coordinates", "provenance"}) {
        meta.erase(key);
    }
    applyOpenDataMetadata(meta, sourceUrl, segment);
    meta["vc_open_data_catalog_segment_lineage_id"] = segmentStableId(segment);
    meta["vc_open_data_representation"] = representation;
    meta["vc_open_data_source_coordinate_level"] = 0;
    meta["vc_open_data_coordinate_space"] = coordinateSpace;
    applySourceCoordinateMetadata(meta, sample, coordinateVolumeId, 0);
    return meta;
}

std::string hashUrl(std::string_view value)
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

const char* representationName(OpenDataSegmentRepresentationKind kind)
{
    switch (kind) {
    case OpenDataSegmentRepresentationKind::DerivedNative: return "derived-native";
    case OpenDataSegmentRepresentationKind::PublishedTransformed: return "published-transformed";
    }
    return "";
}

vc::objects::OpenOptions importerOptions()
{
    vc::objects::OpenOptions options;
    options.author = "vc3d-open-data";
    return options;
}

bool selected(const vcr::catalog::OpenDataSegment& segment, const std::optional<std::vector<std::string>>& segmentIds)
{
    return !segmentIds || std::find(segmentIds->begin(), segmentIds->end(), segment.id) != segmentIds->end();
}

// A surface record for a representation: named like the segment, bound to its
// catalog volume at L0, with what materializing it needs in meta.open_data.
vc::objects::ObjectRecord representationRecord(const vcr::catalog::OpenDataSample& sample,
                                               const vcr::catalog::OpenDataSegment& segment,
                                               const OpenDataSegmentRepresentation& representation)
{
    const std::string url = representation.artifact ? artifactUrl(*representation.artifact) : std::string{};
    auto meta = surfaceRecordMeta(sample, segment, representationName(representation.kind),
                                  representation.coordinateVolumeId, representation.coordinateSpace, url);
    // Authored coordinates are the source volume's downscaled voxels.
    const double storedPerVoxel = representation.kind == OpenDataSegmentRepresentationKind::DerivedNative
        ? 1.0 / originalVolumeDownscale(segment)
        : 1.0;
    meta["coordinates"] = vc::objects::geometryCoordinates(
        {{"volume_uuid", representation.coordinateVolumeId}, {"scale_factor", storedPerVoxel}});
    nlohmann::json openData{
        {"sample_id", sample.id},
        {"segment_id", segment.id},
        {"segment_long_id", segment.longId},
        {"lineage", segmentStableId(segment)},
        {"representation_id", representation.representationId},
        {"representation", representationName(representation.kind)},
        {"ink_detections", inkDetectionMaterializationRecipe(sample, segment)},
    };
    if (!url.empty()) openData["source_url"] = url;
    meta["open_data"] = std::move(openData);

    vc::objects::ObjectRecord record;
    record.uuid = surfaceUuidForRepresentation(representation.representationId);
    record.kind = "surface";
    record.name = segment.suffix.empty() ? segmentLabel(segment) : segment.suffix;
    record.meta = meta.dump();
    if (!url.empty()) {
        // Registered: the store fetches the tifxyz when it is first opened.
        record.storageKind = "url";
        record.storageRef = url;
    }
    return record;
}

} // namespace

std::vector<OpenDataSegmentRepresentation>
classifyOpenDataSegmentRepresentations(const vcr::catalog::OpenDataSample& sample,
                                       const vcr::catalog::OpenDataSegment& segment)
{
    std::vector<OpenDataSegmentRepresentation> result;
    std::set<std::string> volumeIds;
    for (const auto& volume : sample.volumes) {
        if (!volume.id.empty())
            volumeIds.insert(volume.id);
    }

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

    const vcr::catalog::OpenDataArtifact* canonicalAuthored = nullptr;
    for (const auto* preferredType :
         {"tifxyz-flattened", "tifxyz", "tifxyz-normalized"}) {
        const auto it = std::find_if(
            segment.artifacts.begin(), segment.artifacts.end(),
            [&](const vcr::catalog::OpenDataArtifact& artifact) {
                return lowerCopy(artifact.type) == preferredType &&
                       !artifactUrl(artifact).empty();
            });
        if (it != segment.artifacts.end()) {
            canonicalAuthored = &*it;
            break;
        }
    }

    std::map<std::string, const vcr::catalog::OpenDataArtifact*> publishedByTarget;
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
                                   const vcr::catalog::OpenDataArtifact& artifact,
                                   bool canonicalSource) {
        OpenDataSegmentRepresentation published;
        published.artifact = &artifact;
        published.kind = OpenDataSegmentRepresentationKind::PublishedTransformed;
        published.canonicalSource = canonicalSource;
        published.coordinateVolumeId = targetId;
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

    if (result.empty() || !result.front().canonicalSource) {
        return result;
    }
    // Registered destinations are views of the canonical object, not copied geometry.
    return result;
}

std::filesystem::path surfaceStorePath(const std::filesystem::path& remoteCacheRoot, const std::string& sampleId)
{
    return remoteCacheRoot / "open_data" / "stores" / (safePathComponent(sampleId) + ".vcobj");
}

std::string surfaceUuidForRepresentation(const std::string& representationId)
{
    static const QUuid kNamespace(QStringLiteral("3f0c2b8e-6a8d-4d5f-9a3e-7c1b2d4e5f60"));
    return QUuid::createUuidV5(kNamespace, QString::fromStdString(representationId))
        .toString(QUuid::WithoutBraces)
        .toStdString();
}

std::optional<CatalogVolume> catalogVolumeFromTags(const std::vector<std::string>& tags)
{
    constexpr std::string_view samplePrefix = "vc-open-data-sample-id:";
    constexpr std::string_view volumePrefix = "vc-open-data-volume-id:";
    constexpr std::string_view spacePrefix = "vc-open-data-coordinate-space:";
    CatalogVolume volume;
    std::optional<int> level;
    for (const auto& tag : tags) {
        if (tag.rfind(samplePrefix, 0) == 0) {
            volume.sampleId = tag.substr(samplePrefix.size());
        } else if (tag.rfind(volumePrefix, 0) == 0) {
            volume.volumeId = tag.substr(volumePrefix.size());
        } else if (tag.rfind(spacePrefix, 0) == 0) {
            // <sample>/<volume>@L<level>
            const auto space = tag.substr(spacePrefix.size());
            const auto at = space.rfind("@L");
            if (at != std::string::npos) {
                try {
                    std::size_t consumed = 0;
                    const auto digits = space.substr(at + 2);
                    const int parsed = std::stoi(digits, &consumed);
                    if (consumed == digits.size() && parsed >= 0 && parsed <= 5) level = parsed;
                } catch (...) {
                }
            }
        }
    }
    if (volume.sampleId.empty() || volume.volumeId.empty() || !level) {
        return std::nullopt;
    }
    volume.level = *level;
    return volume;
}

SurfaceImportResult importSampleSurfaces(const std::filesystem::path& remoteCacheRoot,
                                         const vcr::catalog::OpenDataSample& sample,
                                         const std::optional<std::vector<std::string>>& segmentIds)
{
    SurfaceImportResult result;
    if (remoteCacheRoot.empty()) {
        result.messages.push_back("Skipped Open Data surfaces: no remote cache directory configured.");
        return result;
    }
    std::vector<vc::objects::ObjectRecord> records;
    for (const auto& segment : sample.segments) {
        if (!selected(segment, segmentIds)) continue;
        for (const auto& representation : classifyOpenDataSegmentRepresentations(sample, segment)) {
            records.push_back(representationRecord(sample, segment, representation));
        }
    }
    result.representations = static_cast<int>(records.size());
    if (records.empty()) return result;

    const auto path = surfaceStorePath(remoteCacheRoot, sample.id);
    fs::create_directories(path.parent_path());
    if (!fs::exists(path)) {
        auto created = ObjectStore::create(path, importerOptions());
    }
    auto db = ObjectStore::open(path, importerOptions());
    vc::objects::RevisionSelection expected;
    std::vector<vc::objects::ObjectRecord> changed;
    {
        auto read = db.read();
        for (auto& record : records) {
            const auto existing = db.objectByUuid(read, record.uuid);
            if (!existing) {
                expected[record.uuid] = std::nullopt;
                ++result.added;
                changed.push_back(std::move(record));
                continue;
            }
            if (existing->name == record.name && nlohmann::json::parse(existing->meta) == nlohmann::json::parse(record.meta)) {
                continue;
            }
            // Fetched geometry stays; the record follows the manifest.
            record.storageKind = existing->storageKind;
            record.storageRef = existing->storageRef;
            record.tags = existing->tags;
            expected[record.uuid] = db.currentObjectRevision(read, record.uuid)->uuid;
            ++result.updated;
            changed.push_back(std::move(record));
        }
    }
    if (!changed.empty()) {
        auto txn = db.write(expected);
        for (const auto& record : changed) db.upsertObject(txn, record);
        txn.commit();
    }
    return result;
}

SurfaceState surfaceStateForSegment(const std::filesystem::path& remoteCacheRoot,
                                    const vcr::catalog::OpenDataSample& sample,
                                    const vcr::catalog::OpenDataSegment& segment)
{
    const auto representations = classifyOpenDataSegmentRepresentations(sample, segment);
    const auto path = surfaceStorePath(remoteCacheRoot, sample.id);
    if (representations.empty() || !fs::exists(path)) return SurfaceState::Missing;
    const auto uuid = surfaceUuidForRepresentation(representations.front().representationId);
    vc::objects::OpenOptions options;
    options.readOnly = true;
    auto db = ObjectStore::open(path, options);
    auto read = db.read();
    if (!db.objectByUuid(read, uuid)) return SurfaceState::Missing;
    return db.currentGeometry(read, uuid) ? SurfaceState::Stored : SurfaceState::Remote;
}

const char* surfaceStateName(SurfaceState state) noexcept
{
    switch (state) {
    case SurfaceState::Missing: return "missing";
    case SurfaceState::Remote: return "remote";
    case SurfaceState::Stored: return "stored";
    }
    return "";
}

bool surfaceHasGeometry(const std::filesystem::path& store, const std::string& uuid)
{
    vc::objects::OpenOptions options;
    options.readOnly = true;
    auto db = ObjectStore::open(store, options);
    auto read = db.read();
    return db.currentGeometry(read, uuid).has_value();
}

void materializeSurface(const std::filesystem::path& store, const std::string& uuid)
{
    if (surfaceHasGeometry(store, uuid)) return;
    auto db = ObjectStore::open(store, importerOptions());
    const auto report = vc::objects::ingestRegisteredGeometry(db, std::nullopt, {uuid});
    if (!report.problems.empty()) throw std::runtime_error("Fetching Open Data surface failed: " + report.problems.front());
}

std::vector<InkDetection> inkDetections(const nlohmann::json& surfaceMeta,
                                        const std::filesystem::path& artifactDirectory)
{
    std::vector<InkDetection> detections;
    const auto records = surfaceMeta.value("open_data", nlohmann::json::object())
                             .value("ink_detections", nlohmann::json::array());
    for (const auto& record : records) {
        if (!record.is_object()) continue;
        InkDetection detection;
        detection.label = record.value("label", std::string{});
        detection.artifactType = record.value("artifact_type", std::string{});
        detection.url = record.value("download_url", std::string{});
        detection.file = artifactDirectory / record.value("local_file", std::string{});
        if (!detection.url.empty() && detection.file != artifactDirectory) detections.push_back(std::move(detection));
    }
    return detections;
}

void fetchInkDetection(const InkDetection& detection)
{
    if (fs::is_regular_file(detection.file)) return;
    const auto bytes = vc::httpGetBytes(detection.url);
    fs::create_directories(detection.file.parent_path());
    QSaveFile output(QString::fromStdString(detection.file.string()));
    if (!output.open(QIODevice::WriteOnly) ||
        output.write(reinterpret_cast<const char*>(bytes.data()), qint64(bytes.size())) != qint64(bytes.size()) ||
        !output.commit()) {
        throw std::runtime_error("Cannot write ink detection " + detection.file.string());
    }
}

} // namespace vc3d::opendata
