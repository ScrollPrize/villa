#include "OpenDataSurfaceStore.hpp"

#include "vc/core/util/HttpFetch.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/GeometryIngest.hpp>
#include <vc_object_store/objects/ObjectStore.hpp>
#include <vc_object_store/objects/VolumeTransforms.hpp>

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

void applySourceCoordinateMetadata(nlohmann::json& meta, const vcr::catalog::OpenDataSample& sample,
                                   const std::string& volumeId)
{
    const auto* volume = sample.findVolume(volumeId);
    if (!volume || !volume->pixelSizeUm || !std::isfinite(*volume->pixelSizeUm) || *volume->pixelSizeUm <= 0.0) {
        return;
    }
    const auto* source = vcr::catalog::preferredVolumeArtifact(*volume);
    if (!source || source->url().empty())
        return;
    meta["vc_open_data_source_path"] = source->url();
    meta["vc_open_data_source_coordinate_scale_factor"] = 1;
    meta["vc_open_data_source_original_resolution"] = *volume->pixelSizeUm;
}

// The record meta of a surface: the catalog's creation metadata and the
// vc_open_data_* identity keys (coordinate space, source, lineage).
nlohmann::json surfaceRecordMeta(const vcr::catalog::OpenDataSample& sample,
                                 const vcr::catalog::OpenDataSegment& segment,
                                 const vcr::catalog::OpenDataSurface& surface)
{
    nlohmann::json meta = nlohmann::json::object();
    if (const auto creation = segment.raw.find("creation"); creation != segment.raw.end() && creation->is_object()) {
        if (const auto metadata = creation->find("metadata"); metadata != creation->end() && metadata->is_object()) {
            meta = *metadata;
        }
    }
    for (const char* key : {"uuid", "name", "type", "format", "scale", "bbox", "coordinates", "provenance"}) {
        meta.erase(key);
    }
    meta["vc_open_data_tifxyz_url"] = surface.artifact->url();
    meta["vc_open_data_segment_id"] = segment.id;
    meta["vc_open_data_segment_long_id"] = segment.longId;
    meta["vc_open_data_original_volume_id"] = segment.originalVolumeId;
    meta["vc_open_data_derived_volume_id"] = segment.sourceVolumeId();
    meta["vc_open_data_original_volume_downscale"] = segment.originalVolumeDownscale;
    if (!segment.createdAt.empty()) {
        meta["date_last_modified"] = vcr::catalog::catalogTimestamp(segment.createdAt);
        meta["vc_open_data_creation_date"] = segment.createdAt;
    }
    meta["vc_open_data_catalog_segment_lineage_id"] = segment.stableId();
    meta["vc_open_data_representation"] = surface.authored ? "derived-native" : "published-transformed";
    meta["vc_open_data_source_coordinate_level"] = 0;
    meta["vc_open_data_coordinate_space"] = sample.id + "/" + surface.volumeId + "@L0";
    applySourceCoordinateMetadata(meta, sample, surface.volumeId);
    return meta;
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

// A surface record: named like the segment, bound to its catalog volume at L0,
// with what materializing it needs in meta.open_data.
vc::objects::ObjectRecord surfaceRecord(const vcr::catalog::OpenDataSample& sample,
                                        const vcr::catalog::OpenDataSegment& segment,
                                        const vcr::catalog::OpenDataSurface& surface)
{
    const std::string url = surface.artifact->url();
    auto meta = surfaceRecordMeta(sample, segment, surface);
    meta["coordinates"] = vc::objects::geometryCoordinates(
        {{"volume_uuid", surface.volumeId}, {"scale_factor", surface.unitsPerVoxel}});
    nlohmann::json images = nlohmann::json::array();
    for (const auto& image : vcr::catalog::inkDetectionImages(segment)) {
        images.push_back({{"download_url", image.url},
                          {"label", image.label},
                          {"artifact_type", image.artifact->type},
                          {"local_file", (fs::path("ink-detections") / image.fileName).generic_string()}});
    }
    meta["open_data"] = {
        {"sample_id", sample.id},
        {"segment_id", segment.id},
        {"segment_long_id", segment.longId},
        {"lineage", segment.stableId()},
        {"representation_id", surface.id},
        {"representation", surface.authored ? "derived-native" : "published-transformed"},
        {"ink_detections", std::move(images)},
        {"source_url", url},
    };

    vc::objects::ObjectRecord record;
    record.uuid = surfaceUuidForRepresentation(surface.id);
    record.kind = "surface";
    record.name = segment.suffix.empty() ? (segment.id.empty() ? std::string("<unnamed segment>") : segment.id)
                                         : segment.suffix;
    record.meta = meta.dump();
    // Registered: the store fetches the tifxyz when it is first opened.
    record.storageKind = "url";
    record.storageRef = url;
    return record;
}

} // namespace

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

bool registerVolumeLevel(vc::objects::ObjectStoreSession& session, const vc::objects::AttachmentIdentity& attachment,
                         const std::string& volume, const std::string& shown, int level)
{
    if (volume == shown || !session.writable(attachment)) return false;
    bool related = false;
    session.read(attachment, [&](const auto& db, const auto& txn, const auto&) {
        related = db.volumeTransforms(txn).resolve(shown, volume).status != vcr::catalog::TransformStatus::Unavailable;
    });
    if (related) return false;
    static const QUuid kNamespace(QStringLiteral("8d7f3c1a-2b4e-4f6a-9c5d-1e0b7a6f3d28"));
    const auto uuid = QUuid::createUuidV5(kNamespace, QString::fromStdString(shown + "\n" + volume))
                          .toString(QUuid::WithoutBraces).toStdString();
    session.act(attachment, {{uuid, std::nullopt}}, [&](auto& db, auto& txn) {
        db.registerVolumeTransform(txn, shown, volume, vc::objects::VolumeAffine::scale(std::ldexp(1.0, -level)),
                                   "pyramid level " + std::to_string(level), uuid);
    });
    return true;
}

std::optional<vc::objects::VolumeAffine> volumeToVolume(vc::objects::VolumeTransforms transforms,
                                                        const std::string& from, const std::optional<CatalogVolume>& fromShown,
                                                        const std::string& to, const std::optional<CatalogVolume>& toShown)
{
    for (const auto& [id, shown] : {std::pair{from, fromShown}, std::pair{to, toShown}}) {
        if (shown && shown->volumeId != id &&
            transforms.resolve(shown->volumeId, id).status == vcr::catalog::TransformStatus::Unavailable)
            transforms.graph.add({"level:" + id, "", shown->volumeId, id,
                                  vc::objects::VolumeAffine::scale(std::ldexp(1.0, -shown->level)), "pyramid level"});
    }
    const auto resolution = transforms.resolve(from, to);
    if (resolution.status != vcr::catalog::TransformStatus::Resolved) return std::nullopt;
    return resolution.resolved().forward;
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
        for (const auto& surface : sample.surfaces(segment)) {
            records.push_back(surfaceRecord(sample, segment, surface));
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
    const auto surfaces = sample.surfaces(segment);
    const auto path = surfaceStorePath(remoteCacheRoot, sample.id);
    if (surfaces.empty() || !fs::exists(path)) return SurfaceState::Missing;
    const auto uuid = surfaceUuidForRepresentation(surfaces.front().id);
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
