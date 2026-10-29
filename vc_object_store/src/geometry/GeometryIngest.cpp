#include "geometry/GeometryIngestBatch.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/GeometryIngest.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/PackedTrackStore.hpp"
#include "vc_object_store/Uuid.hpp"
#include "formats/GeometryInput.hpp"
#include "formats/TrackInput.hpp"
#include "formats/RemoteSource.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <map>
#include <set>

namespace vc::objects {
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
using formats::Prepared;
using formats::fiber;
using formats::surface;
using formats::collections;
using formats::track;
using formats::readJson;
// One committed write transaction; a session serializes it with its actions.
using Commit = ObjectStoreSession::Commit;
struct Expected { std::optional<int64_t> generation; };
Expected expectedAt(ObjectStore& db, const Txn& read, const std::string& id)
{
    Expected e;
    if (const auto object = db.objectByUuid(read, id)) e.generation = object->generation;
    return e;
}
Commit directCommit(ObjectStore& db)
{
    return [&db](const auto& write) { auto txn = db.write(); write(db, txn); txn.commit(); };
}
void ingestBatch(const Commit& commit, std::vector<Prepared> batch,
    const std::map<std::string, Expected>& expectations, GeometryIngestReport& report,
    const std::string& coordinateContext = "{}", bool registeredInitial = false)
{
    const auto count = batch.size();
    int64_t edges = 0;
    commit([&](ObjectStore& db, WriteTxn& write) {
        for (const auto& p : batch) {
            const auto existing = db.objectByUuid(write, p.object.uuid);
            const auto& e = expectations.at(p.object.uuid);
            if ((existing ? std::optional(existing->generation) : std::nullopt) != e.generation)
                throw std::runtime_error("object changed while reading source; retry ingestion");
            if (existing && existing->kind != p.object.kind) throw std::runtime_error("source UUID belongs to another object kind");
        }
        edges = detail::ingestGeometryBatch(db, write, std::move(batch), coordinateContext, registeredInitial);
    });
    report.ingested += count; report.edgesImported += edges;
}
void ingestOne(const Commit& commit, Prepared p, const Expected& expected, GeometryIngestReport& report,
    const std::string& coordinateContext)
{
    std::map<std::string, Expected> expectations{{p.object.uuid, expected}};
    std::vector<Prepared> batch;
    batch.push_back(std::move(p));
    ingestBatch(commit, std::move(batch), expectations, report, coordinateContext);
}
// A tifxyz surface directory itself, or a folder of them (each with its meta.json).
std::vector<fs::path> surfaceDirectories(const fs::path& source)
{
    if (!fs::is_directory(source) || fs::exists(source / "meta.json")) return {source};
    std::vector<fs::path> result;
    for (const auto& entry : fs::directory_iterator(source))
        if (entry.is_directory() && fs::exists(entry.path() / "meta.json")) result.push_back(entry.path());
    if (result.empty()) throw std::runtime_error("no tifxyz surface (a directory with meta.json) found");
    std::sort(result.begin(), result.end());
    return result;
}
// Adds the import's tags to those the source declares.
void addTags(Prepared& p, const std::vector<std::string>& tags)
{
    for (const auto& tag : tags)
        if (std::find(p.object.tags.begin(), p.object.tags.end(), tag) == p.object.tags.end()) p.object.tags.push_back(tag);
}
void problem(GeometryIngestReport& report, const std::string& label, const std::exception& error)
{
    report.problems.push_back(label + ": " + error.what());
}
}

std::vector<GeometrySource> prepareGeometrySource(const std::string& kind, const fs::path& source,
    const std::vector<std::string>& availableUuids)
{
    if (kind == "fiber") return formats::fibers(source, availableUuids);
    if (kind == "surface") return {surface(source)};
    if (kind == "point_collection") return collections(source);
    if (kind == "umbilicus") return {formats::umbilicus(source)};
    throw std::invalid_argument("unsupported batch geometry kind: " + kind);
}

GeometryIngestReport ingestRegisteredGeometry(ObjectStore& db, const std::optional<fs::path>& datasetRoot,
    const std::vector<std::string>& objectUuids, const util::ProgressCallback& progress)
{
    GeometryIngestReport report;
    std::vector<ObjectRecord> objects;
    std::map<std::string, Expected> expectations;
    {
        auto read = db.read();
        if (objectUuids.empty()) {
            for (const char* kind : {"surface", "fiber", "point_collection", "track"}) {
                auto batch = db.objectsByKind(read, kind); objects.insert(objects.end(), batch.begin(), batch.end());
            }
        } else {
            std::set<std::string> seen;
            for (const auto& id : objectUuids) {
                const auto object = db.objectByUuid(read, id);
                if (!object) report.problems.push_back(id + ": object not found");
                else if (seen.insert(object->uuid).second) objects.push_back(*object);
            }
        }
        for (const auto& object : objects) expectations[object.uuid] = expectedAt(db, read, object.uuid);
    }
    std::map<fs::path, std::vector<Prepared>> pclCache;
    std::map<fs::path, std::unique_ptr<PackedTrackStore>> trackCache;
    std::vector<Prepared> prepared;
    for (size_t i = 0; i < objects.size(); ++i) {
        const auto& object = objects[i];
        util::reportProgress(progress, "ingesting geometry", i, objects.size(), object.uuid);
        try {
            const auto e = expectations.at(object.uuid);
            Prepared p;
            if (object.kind == "track") {
                if (!object.track) throw std::runtime_error("track detail missing");
                fs::path path(object.track->datasetRef);
                if (path.is_relative()) path = datasetRoot.value_or(db.path().parent_path()) / path;
                const auto packed = resolvePackedTrackStore(path);
                auto& store = trackCache[packed];
                if (!store) store = std::make_unique<PackedTrackStore>(packed);
                const auto id = detail::trackSourceId(object.track->externalKey);
                const auto found = std::lower_bound(store->sourceIds().begin(), store->sourceIds().end(), id);
                if (found == store->sourceIds().end() || *found != id) throw std::runtime_error("track source ID missing");
                p = track(*store, found - store->sourceIds().begin(), object);
                if (fs::exists(packed / "metadata.json")) p.source = readJson(packed / "metadata.json");
            } else if (object.storageKind == "url") {
                // Remote tifxyz: fetched once, decoded here, then held by the store.
                if (object.kind != "surface" || !formats::isRemoteSource(object.storageRef))
                    throw std::runtime_error("remote sources are tifxyz surfaces at http(s):// or s3:// URLs");
                const auto staging = fs::temp_directory_path() / ("vc-object-fetch-" + util::newUuid());
                fs::create_directory(staging);
                try {
                    formats::fetchRemoteSurface(object.storageRef, staging);
                    p = surface(staging);
                } catch (...) { std::error_code ignored; fs::remove_all(staging, ignored); throw; }
                std::error_code ignored; fs::remove_all(staging, ignored);
                // The registration, not the remote meta.json, is the identity.
                p.object.uuid = object.uuid;
            } else {
                std::optional<fs::path> path;
                { auto read = db.read(); path = db.resolveStoragePath(read, object.uuid, datasetRoot); }
                if (!path) throw std::runtime_error("source not found by UUID");
                if (object.kind == "surface") p = surface(*path);
                else if (object.kind == "fiber") p = fiber(*path);
                else if (object.kind == "point_collection") {
                    if (!pclCache.contains(*path)) pclCache[*path] = collections(*path);
                    auto& collections = pclCache.at(*path);
                    const auto found = std::find_if(collections.begin(), collections.end(), [&](const auto& c) { return c.object.uuid == object.uuid; });
                    if (found == collections.end()) throw std::runtime_error("collection disappeared");
                    p = std::move(*found);
                } else throw std::runtime_error("unsupported object kind");
                if (p.object.uuid != object.uuid) throw std::runtime_error("source UUID changed during ingestion");
            }
            expectations[p.object.uuid] = e; prepared.push_back(std::move(p));
        } catch (const std::exception& error) { problem(report, object.uuid, error); }
    }
    if (report.problems.empty()) {
        try { ingestBatch(directCommit(db), std::move(prepared), expectations, report, "{}", true); }
        catch (const std::exception& error) { problem(report, "batch", error); }
    }
    util::reportProgress(progress, "ingesting geometry", objects.size(), objects.size());
    return report;
}

namespace {
// Stops before the next surface or track once `cancelled` is set.
GeometryIngestReport ingestSource(ObjectStore& db, const Commit& commit, const std::string& kind,
    const fs::path& source, const util::ProgressCallback& progress, const std::string& coordinateContext,
    const std::vector<std::string>& tags, const std::atomic_bool& cancelled)
{
    GeometryIngestReport report;
    try {
        geometryCoordinates(Json::parse(coordinateContext));
        if (kind == "track") {
            const auto packed = resolvePackedTrackStore(source);
            PackedTrackStore store(packed);
            std::map<std::string, ObjectRecord> existing;
            std::map<std::string, Expected> expectations;
            std::set<std::string> refs;
            {
                auto read = db.read();
                for (const auto& object : db.objectsByKind(read, "track", std::nullopt, true)) {
                    if (!object.track) continue;
                    fs::path path(object.track->datasetRef);
                    if (path.is_relative()) path = db.path().parent_path() / path;
                    std::error_code error;
                    if (fs::weakly_canonical(path, error) == packed && !error) {
                        refs.insert(object.track->datasetRef); existing.emplace(object.track->externalKey, object);
                        expectations[object.uuid] = expectedAt(db, read, object.uuid);
                    }
                }
            }
            if (refs.size() > 1) throw std::runtime_error("track dataset registered under multiple aliases");
            const auto ref = refs.empty() ? packed.string() : *refs.begin();
            for (size_t row = 0; row < store.sourceIds().size() && !cancelled; ++row) {
                const auto key = std::to_string(store.sourceIds()[row]);
                util::reportProgress(progress, "ingesting tracks", row, store.sourceIds().size(), key);
                try {
                    ObjectRecord object;
                    if (existing.contains(key)) object = existing.at(key);
                    else {
                        object.uuid = util::newUuid(); object.kind = "track"; object.name = key;
                        object.storageKind = "dataset"; object.storageRef = ref;
                        object.track = TrackDetail{}; object.track->datasetRef = ref; object.track->externalKey = key;
                    }
                    const auto e = expectations.contains(object.uuid) ? expectations.at(object.uuid) : Expected{};
                    auto prepared = track(store, row, object);
                    if (fs::exists(packed / "metadata.json")) prepared.source = readJson(packed / "metadata.json");
                    addTags(prepared, tags);
                    ingestOne(commit, std::move(prepared), e, report, coordinateContext);
                } catch (const std::exception& error) { problem(report, key, error); }
            }
        } else if (kind == "surface") {
            // Surfaces have no native relationships among themselves: each commits on its own.
            const auto directories = surfaceDirectories(source);
            for (size_t i = 0; i < directories.size() && !cancelled; ++i) {
                const auto& directory = directories[i];
                util::reportProgress(progress, "ingesting surfaces", i, directories.size(), directory.filename().string());
                try {
                    Expected expected;
                    auto prepared = surface(directory);
                    addTags(prepared, tags);
                    { auto read = db.read(); expected = expectedAt(db, read, prepared.object.uuid); }
                    ingestOne(commit, std::move(prepared), expected, report, coordinateContext);
                } catch (const std::exception& error) { problem(report, directory.string(), error); }
            }
            util::reportProgress(progress, "ingesting surfaces", directories.size(), directories.size());
        } else {
            std::vector<Prepared> batch;
            std::map<std::string, Expected> expectations;
            {
                // Pin registration expectations while reading files. Ingestion
                // rechecks duplicate identities under the writer lock.
                auto read = db.read();
                std::vector<std::string> available;
                for (const auto& object : db.objectsByKind(read, kind)) available.push_back(object.uuid);
                batch = prepareGeometrySource(kind, source, available);
                for (auto& p : batch) { expectations[p.object.uuid] = expectedAt(db, read, p.object.uuid); addTags(p, tags); }
            }
            ingestBatch(commit, std::move(batch), expectations, report, coordinateContext);
        }
    } catch (const std::exception& error) { problem(report, source.string(), error); }
    return report;
}
}

GeometryIngestReport ingestGeometrySource(ObjectStore& db, const std::string& kind, const fs::path& source,
    const util::ProgressCallback& progress, const std::string& coordinateContext, const std::vector<std::string>& tags)
{
    static const std::atomic_bool never{false};
    return ingestSource(db, directCommit(db), kind, source, progress, coordinateContext, tags, never);
}

GeometryIngestReport ingestGeometrySource(ObjectStoreSession& session, const AttachmentIdentity& attachment,
    const std::string& kind, const fs::path& source, const util::ProgressCallback& progress,
    const std::string& coordinateContext, const std::vector<std::string>& tags)
{
    GeometryIngestReport report;
    session.writeJob(attachment, [&](ObjectStore& db, const Commit& commit, const std::atomic_bool& cancelled) {
        report = ingestSource(db, commit, kind, source, progress, coordinateContext, tags, cancelled);
    });
    return report;
}
std::string importSurfaceChannel(ObjectStore& db, WriteTxn& txn, const std::string& objectUuid,
    const std::string& name, const fs::path& tiff)
{
    const auto current = db.currentGeometry(txn, objectUuid);
    if (!current || current->kind != "surface") throw std::invalid_argument("channel import requires a stored surface: " + objectUuid);
    return db.updateSurfaceChannels(txn, objectUuid, {formats::surfaceChannel(name, tiff)}, {}, current->uuid);
}
} // namespace vc::objects
