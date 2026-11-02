#include "catalog/Catalog.hpp"
#include "formats/GeometryInput.hpp"
#include "formats/RemoteSource.hpp"
#include "geometry/ArrayLayout.hpp"
#include <vc_reader/native/http.hpp>
#include <fstream>
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/Uuid.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace vc::objects {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;

// Provenance of a catalog surface: how the catalog says it was made.
Json provenance(const vcr::catalog::OpenDataSegment& segment, const vcr::catalog::OpenDataSurface& surface)
{
    const auto& creation = surface.authored || surface.artifact->creation.process.empty() ? segment.creation
                                                                                          : surface.artifact->creation;
    Json sources = Json::array();
    if (creation.derivedFrom) sources.push_back({{"type", creation.derivedFrom->type}, {"id", creation.derivedFrom->id}});
    return {{"tool", creation.process.empty() ? std::string("open-data") : creation.process},
            {"date", creation.date.empty() ? segment.creation.date : creation.date},
            {"parameters", Json::object()},
            {"sources", std::move(sources)}};
}

ObjectRecord placeholder(const vcr::catalog::OpenDataSample& sample, const vcr::catalog::OpenDataSegment& segment,
                         const vcr::catalog::OpenDataSurface& surface)
{
    ObjectRecord record;
    record.uuid = catalogSurfaceUuid(surface);
    record.kind = "surface";
    record.name = !segment.suffix.empty() ? segment.suffix : !segment.id.empty() ? segment.id : "<unnamed segment>";
    record.storageKind = kCatalogStorage;
    Json files = Json::array();
    for (const auto& image : vcr::catalog::inkDetectionImages(segment))
        files.push_back({{"name", image.fileName}, {"label", image.label}, {"url", image.url}});
    Json meta = {
        {"coordinates", geometryCoordinates({{"volume_uuid", surface.volumeId}, {"scale_factor", surface.unitsPerVoxel}})},
        {"provenance", provenance(segment, surface)},
        {"catalog", {{"sample", sample.id}, {"segment", segment.id}, {"artifact", surface.artifact->type},
                     {"volume", surface.volumeId}, {"url", surface.artifact->url()}, {"files", std::move(files)}}},
    };
    record.meta = meta.dump();
    SurfaceDetail detail;
    if (surface.authored && surface.artifact->type == "tifxyz" && segment.originalTifxyz)
        detail.scale = segment.originalTifxyz->scale;
    // Bounds in stored units, from where the catalog says the segment lies.
    if (const auto* coverage = segment.coverage(surface.volumeId); coverage && coverage->overlapRatio > 0) {
        const auto& b = coverage->bbox;
        const double s = surface.unitsPerVoxel;
        detail.bbox = std::array<double, 6>{b.low[0] * s, b.low[1] * s, b.low[2] * s, b.high[0] * s, b.high[1] * s, b.high[2] * s};
    }
    record.surface = detail;
    return record;
}

// A surface authored in a downscaled level has its coordinates in that level's voxels, and
// a level's voxel is centred on the block it reduces: voxel v of a level at factor f lies at
// f * v + (f - 1) / 2 in the volume, while the store maps stored coordinates by scale alone.
// Shift the valid vertices so that scale alone lands them where they were drawn; invalid
// vertices keep their stored values.
void centreOnReducedBlocks(GeometryData& geometry, double unitsPerVoxel)
{
    if (!(unitsPerVoxel > 0) || unitsPerVoxel >= 1) return;
    const float shift = float((1.0 - unitsPerVoxel) / 2.0);
    const auto points = std::find_if(geometry.arrays.begin(), geometry.arrays.end(),
                                     [](const GeometryArray& array) { return array.name == "points"; });
    if (points == geometry.arrays.end()) return;
    const auto valid = std::find_if(geometry.arrays.begin(), geometry.arrays.end(),
                                    [](const GeometryArray& array) { return array.name == detail::surfaceValidChannel; });
    const std::size_t count = points->bytes.size() / (3 * sizeof(float));
    for (std::size_t i = 0; i < count; ++i) {
        float xyz[3];
        std::memcpy(xyz, points->bytes.data() + i * sizeof(xyz), sizeof(xyz));
        const bool sentinel = xyz[0] == -1.f && xyz[1] == -1.f && xyz[2] == -1.f;
        if (sentinel || !std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) || !std::isfinite(xyz[2]) ||
            (valid != geometry.arrays.end() && !valid->bytes[i]))
            continue;
        for (auto& component : xyz) component += shift;
        std::memcpy(points->bytes.data() + i * sizeof(xyz), xyz, sizeof(xyz));
    }
}

// Fetches record their geometry head separately from user revisions. Old stores
// without this marker are conservatively preserved once geometry has a parent.
bool locallyEdited(const ObjectRecord& record, const GeometryRevision& geometry)
{
    const auto meta = Json::parse(record.meta);
    const auto baseline = meta.find("catalog_geometry_revision");
    return baseline != meta.end() ? *baseline != geometry.uuid : geometry.parentUuid.has_value();
}

bool sameCatalogRecord(const ObjectRecord& a, const ObjectRecord& b)
{
    const auto bounds = [](const ObjectRecord& r) { return r.surface ? r.surface->bbox : std::nullopt; };
    auto left = Json::parse(a.meta), right = Json::parse(b.meta);
    for (const auto* key : {"catalog_geometry_revision", "catalog_write"}) {
        left.erase(key);
        right.erase(key);
    }
    return a.name == b.name && left == right && bounds(a) == bounds(b);
}
} // namespace

std::string catalogSurfaceUuid(const vcr::catalog::OpenDataSurface& surface)
{
    // Fixed namespace for catalog surfaces; never change it.
    return util::nameUuid("3f0c2b8e-6a8d-4d5f-9a3e-7c1b2d4e5f60", surface.id);
}

std::optional<CatalogReference> catalogReference(const ObjectRecord& record)
{
    const auto meta = Json::parse(record.meta.empty() ? std::string("{}") : record.meta);
    const auto catalog = meta.find("catalog");
    if (catalog == meta.end() || !catalog->is_object()) return std::nullopt;
    CatalogReference reference;
    reference.sample = catalog->value("sample", "");
    reference.segment = catalog->value("segment", "");
    reference.artifact = catalog->value("artifact", "");
    reference.volume = catalog->value("volume", "");
    reference.url = catalog->value("url", "");
    for (const auto& file : catalog->value("files", Json::array()))
        reference.files.push_back({file.value("name", ""), file.value("label", ""), file.value("url", "")});
    return reference;
}

namespace detail {

CatalogStoreReport updateCatalogRecords(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                        const vcr::catalog::OpenDataSample& sample, bool refresh,
                                        const std::optional<std::vector<std::string>>& segmentIds,
                                        std::vector<std::string>& stale)
{
    CatalogStoreReport report;
    report.path = session.path(attachment);
    std::vector<ObjectRecord> records;
    for (const auto& segment : sample.segments) {
        if (segmentIds && std::find(segmentIds->begin(), segmentIds->end(), segment.id) == segmentIds->end()) continue;
        for (const auto& surface : sample.surfaces(segment)) records.push_back(placeholder(sample, segment, surface));
    }
    report.surfaces = int(records.size());
    RevisionSelection expected;
    std::vector<ObjectRecord> changed;
    session.read(attachment, [&](const auto& db, const auto& read, const auto&) {
        for (auto& record : records) {
            const auto existing = db.objectByUuid(read, record.uuid);
            if (!existing) {
                expected[record.uuid] = std::nullopt;
                ++report.added;
                changed.push_back(std::move(record));
                continue;
            }
            if (!refresh) continue;
            const auto geometry = db.currentGeometry(read, record.uuid);
            if (geometry && locallyEdited(*existing, *geometry)) {
                ++report.preserved;
                continue;
            }
            if (sameCatalogRecord(*existing, record)) continue;
            if (geometry) {
                auto meta = Json::parse(record.meta);
                meta["catalog_geometry_revision"] = geometry->uuid;
                record.meta = meta.dump();
                // The fetched grid stays; a changed published tifxyz is fetched again.
                if (existing->surface) {
                    record.surface->gridWidth = existing->surface->gridWidth;
                    record.surface->gridHeight = existing->surface->gridHeight;
                    record.surface->scale = existing->surface->scale;
                }
                if (Json::parse(geometry->attributes).value("source_url", "") != catalogReference(record)->url) {
                    ++report.stale;
                    stale.push_back(record.uuid);
                }
            }
            record.tags = existing->tags;
            expected[record.uuid] = db.currentObjectRevision(read, record.uuid)->uuid;
            ++report.updated;
            changed.push_back(std::move(record));
        }
    });
    if (!changed.empty())
        session.fetchWrite(attachment, expected, [&](ObjectStore& db, WriteTxn& txn) {
            for (auto record : changed) {
                auto meta = Json::parse(record.meta);
                meta["catalog_write"] = util::newUuid();
                record.meta = meta.dump();
                db.upsertObject(txn, record);
            }
        });
    return report;
}

void downloadFile(const std::string& url, const fs::path& path)
{
    vcr::HttpClient client(vcr::HttpOptions{});
    std::vector<std::uint8_t> body;
    if (!client.get(url, body)) throw std::runtime_error("cannot fetch " + url);
    fs::create_directories(path.parent_path());
    const auto partial = path.string() + ".partial-" + util::newUuid();
    {
        std::ofstream output(partial, std::ios::binary);
        output.write(reinterpret_cast<const char*>(body.data()), std::streamsize(body.size()));
        if (!output) throw std::runtime_error("cannot write " + path.string());
    }
    fs::rename(partial, path);
}

} // namespace detail

void fetchCatalogSurface(ObjectStore& db, const std::string& uuid, const ObjectStoreSession::FetchWrite& write)
{
    std::optional<ObjectRecord> record;
    std::optional<std::string> geometry, revision;
    {
        auto read = db.read();
        record = db.objectByUuid(read, uuid);
        if (const auto current = db.currentGeometry(read, uuid)) {
            // A local edit may have landed after refresh queued this fetch.
            if (record && locallyEdited(*record, *current)) return;
            geometry = current->uuid;
        }
        if (const auto current = db.currentObjectRevision(read, uuid)) revision = current->uuid;
    }
    const auto reference = record ? catalogReference(*record) : std::nullopt;
    if (!reference || reference->url.empty()) throw std::runtime_error("not a catalog surface: " + uuid);
    const auto staging = fs::temp_directory_path() / ("vc-catalog-fetch-" + util::newUuid());
    fs::create_directory(staging);
    formats::Prepared prepared;
    try {
        formats::fetchRemoteSurface(reference->url, staging);
        prepared = formats::surfaceGeometry(staging);
    } catch (...) { std::error_code ignored; fs::remove_all(staging, ignored); throw; }
    std::error_code ignored;
    fs::remove_all(staging, ignored);
    const auto coordinates = Json::parse(record->meta.empty() ? std::string("{}") : record->meta).value("coordinates", Json::object());
    if (const auto scale = coordinates.find("scale_factor"); scale != coordinates.end() && scale->is_number())
        centreOnReducedBlocks(prepared.geometry, scale->get<double>());
    // Only the geometry is taken; the published meta.json is not the record.
    auto attributes = Json::parse(prepared.geometry.attributes);
    attributes["source_url"] = reference->url;
    prepared.geometry.attributes = attributes.dump();
    auto updated = *record;
    if (!updated.surface) updated.surface = SurfaceDetail{};
    updated.surface->gridWidth = prepared.object.surface->gridWidth;
    updated.surface->gridHeight = prepared.object.surface->gridHeight;
    updated.surface->scale = prepared.object.surface->scale;
    // Expected at the revision read: a concurrent change (another fetch, a
    // refresh) fails this write, and the next read fetches again.
    write({{uuid, revision}}, [&](ObjectStore& store, WriteTxn& txn) {
        store.upsertObject(txn, updated);
        const auto fetched = store.putGeometry(txn, uuid, prepared.geometry, geometry);
        store.setMeta(txn, uuid, "catalog_geometry_revision", Json(fetched).dump());
        store.setMeta(txn, uuid, "catalog_write", Json(util::newUuid()).dump());
    });
}

CatalogStoreReport addCatalogSample(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                    const vcr::catalog::OpenDataSample& sample,
                                    const std::optional<std::vector<std::string>>& segmentIds)
{
    std::vector<std::string> stale;
    return detail::updateCatalogRecords(session, attachment, sample, false, segmentIds, stale);
}

void restoreCatalogVersion(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                           const std::string& objectUuid)
{
    const auto uuid = util::canonicalUuid(objectUuid);
    std::optional<std::string> revision;
    session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
        const auto record = store.objectByUuid(txn, uuid);
        if (!record || !catalogReference(*record)) throw std::runtime_error("not a catalog surface: " + uuid);
        const auto meta = Json::parse(record->meta);
        if (!meta.contains("catalog_geometry_revision"))
            throw std::runtime_error("the catalog version of " + uuid + " was never fetched");
        // The catalog's last write (a fetch or refresh) minted this token; the
        // user's later revisions carry it forward.
        const auto token = meta.value("catalog_write", std::string());
        for (const auto& version : store.objectHistory(txn, uuid))
            if (!revision && Json::parse(version.record.meta).value("catalog_write", std::string()) == token)
                revision = version.uuid;
    });
    if (!revision) throw std::runtime_error("the catalog version of " + uuid + " is not in its history");
    session.editCurrent(attachment, {uuid}, [&](auto& store, auto& txn, const auto&) { store.restoreObject(txn, uuid, *revision); });
}

CatalogStoreReport refreshCatalog(ObjectStoreSession& session, const AttachmentIdentity& attachment,
                                  const vcr::catalog::OpenDataSample& sample,
                                  const std::optional<std::vector<std::string>>& segmentIds)
{
    std::vector<std::string> stale;
    auto report = detail::updateCatalogRecords(session, attachment, sample, true, segmentIds, stale);
    session.refetch(attachment, stale);
    return report;
}

fs::path publishedFile(ObjectStoreSession& session, const AttachmentIdentity& attachment, const std::string& objectUuid,
                       const std::string& name)
{
    std::optional<CatalogReference> reference;
    session.read(attachment, [&](const auto& store, const auto& txn, const auto&) {
        if (const auto record = store.objectByUuid(txn, objectUuid)) reference = catalogReference(*record);
    });
    if (!reference) throw std::runtime_error("not a catalog surface: " + objectUuid);
    const auto file = std::find_if(reference->files.begin(), reference->files.end(),
                                   [&](const auto& f) { return f.name == name; });
    if (file == reference->files.end()) throw std::runtime_error("no published file " + name + " for " + objectUuid);
    const auto target = session.artifactDirectory(attachment, objectUuid) / file->name;
    if (!fs::is_regular_file(target)) detail::downloadFile(file->url, target);
    return target;
}

} // namespace vc::objects
