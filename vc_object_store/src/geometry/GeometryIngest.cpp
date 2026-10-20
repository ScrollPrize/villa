#include "imports/SourceRelationships.hpp"
#include "internal/Database.hpp"
#include "internal/GeometryCoordinates.hpp"
#include "vc_object_store/objects/GeometryIngest.hpp"
#include "vc_object_store/PackedTrackStore.hpp"
#include "vc_object_store/Uuid.hpp"
#include "formats/GeometryInput.hpp"
#include "formats/TrackInput.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <span>

namespace vc::objects {
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
using formats::Prepared;
using formats::fiber;
using formats::patch;
using formats::collections;
using formats::track;
using formats::readJson;
struct Expected { std::optional<std::string> revision; std::optional<int64_t> generation; };
Expected expectedAt(ObjectStore& db, const Txn& read, const std::string& id)
{
    Expected e;
    if (const auto object = db.objectByUuid(read, id)) e.generation = object->generation;
    if (const auto geometry = db.currentGeometry(read, id)) e.revision = geometry->uuid;
    return e;
}
void normalizeMetadata(Prepared& p, const std::string& coordinateContext)
{
    auto source = p.source;
    const auto coordinates = detail::mergeGeometryCoordinates(
        detail::sourceGeometryCoordinates(source), Json::parse(coordinateContext));
    detail::eraseSourceGeometryCoordinates(source);
    for (const auto* field : {"uuid", "name", "type", "version", "branches", "adjacent_branches", "windings_linked"}) source.erase(field);
    if (source.contains("tags") && source["tags"].is_array()) source.erase("tags");
    if (p.object.kind == "patch") for (const auto* field : {"scale", "bbox", "tiff_dimensions", "format"}) source.erase(field);
    if (p.object.kind == "fiber") for (const auto* field : {"width", "width_gap_fraction", "optimization_mode"}) source.erase(field);
    if (p.object.kind == "fiber" && source.contains("hv_classification")) {
        source["hv_classification"].erase("manual_tag");
        if (!p.object.fiber->manualFamily) source["hv_classification"].erase("automatic_tag");
        if (source["hv_classification"].empty()) source.erase("hv_classification");
    }
    if (p.object.kind == "point_collection" && source.contains("point_attributes"))
        for (auto& point : source["point_attributes"]) if (point.is_object()) point.erase("links");
    p.object.meta = Json{{"source", source}, {"coordinates", coordinates}}.dump();
}
void publishBatch(ObjectStore& db, std::vector<Prepared> batch,
    const std::map<std::string, Expected>& expectations, GeometryIngestReport& report,
    const std::string& coordinateContext = "{}", bool registeredInitial = false)
{
    auto write = db.write();
    auto* connection=ObjectStoreImplAccess::conn(write);
    std::set<std::string> incoming;
    // Check the entire batch under the writer lock before mutating any object.
    for(const auto& p:batch) {
        if(!incoming.insert(p.object.uuid).second)
            throw std::runtime_error("duplicate object UUID in import: "+p.object.uuid);
        const auto existing=db.objectByUuid(write,p.object.uuid);
        detail::Statement history(connection,"SELECT 1 FROM geometry_revisions WHERE object_uuid=? LIMIT 1");
        history.bindBlob(1,detail::parseUuid(p.object.uuid));
        detail::Statement prior(connection,"SELECT 1 FROM object_revisions WHERE object_uuid=? LIMIT 1"); prior.bind(1,p.object.uuid);
        if(registeredInitial ? (!existing || existing->retired || history.row()) : (existing.has_value() || prior.row()))
            throw std::runtime_error("object already exists or was previously imported; re-import is unsupported: "+p.object.uuid);
    }
    int64_t ingested = 0;
    std::vector<detail::SourceRecord> sources;
    for (auto& p : batch) {
        const auto& e = expectations.at(p.object.uuid);
        const auto existing = db.objectByUuid(write, p.object.uuid);
        if ((existing ? std::optional(existing->generation) : std::nullopt) != e.generation)
            throw std::runtime_error("object changed while reading source; retry ingestion");
        if (existing && existing->kind != p.object.kind) throw std::runtime_error("source UUID belongs to another object kind");
        normalizeMetadata(p, coordinateContext);
        if (!existing) db.upsertObject(write, p.object);
        db.putGeometry(write, p.object.uuid, p.geometry, e.revision);
        // Publication no longer needs these arrays; release them before the
        // next object and the topology import allocate their workspaces.
        p.geometry.arrays.clear();
        sources.push_back({p.object, std::move(p.source)});
        auto record = *db.objectByUuid(write, p.object.uuid);
        auto metadata = Json::parse(record.meta);
        // First geometry loading seeds source fields, preserving unrelated
        // placeholder metadata and rejecting conflicting coordinate declarations.
        const auto incoming = Json::parse(p.object.meta);
        metadata["source"] = incoming["source"];
        metadata["coordinates"] = detail::mergeGeometryCoordinates(
            metadata.value("coordinates", Json::object()), incoming["coordinates"]);
        record.meta = metadata.dump();
        if (p.object.fiber) record.fiber = p.object.fiber;
        if (p.object.patch) record.patch = p.object.patch;
        if (p.object.track) record.track = p.object.track;
        db.upsertObject(write, record);
        ++ingested;
    }
    const auto edges = detail::importSourceRelationships(db, write, sources);
    write.commit();
    report.ingested += ingested; report.edgesImported += edges;
}
void publish(ObjectStore& db, Prepared p, const Expected& expected, GeometryIngestReport& report,
    const std::string& coordinateContext)
{
    std::map<std::string, Expected> expectations{{p.object.uuid, expected}};
    std::vector<Prepared> batch;
    batch.push_back(std::move(p));
    publishBatch(db, std::move(batch), expectations, report, coordinateContext);
}
void problem(GeometryIngestReport& report, const std::string& label, const std::exception& error)
{
    report.problems.push_back(label + ": " + error.what());
}
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
            for (const char* kind : {"patch", "fiber", "point_collection", "track"}) {
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
            } else {
                std::optional<fs::path> path;
                { auto read = db.read(); path = db.resolveStoragePath(read, object.uuid, datasetRoot); }
                if (!path) throw std::runtime_error("source not found by UUID");
                if (object.kind == "patch") p = patch(*path);
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
        try { publishBatch(db, std::move(prepared), expectations, report, "{}", true); }
        catch (const std::exception& error) { problem(report, "batch", error); }
    }
    util::reportProgress(progress, "ingesting geometry", objects.size(), objects.size());
    return report;
}

GeometryIngestReport ingestGeometrySource(ObjectStore& db, const std::string& kind, const fs::path& source,
    const util::ProgressCallback& progress, const std::string& coordinateContext)
{
    GeometryIngestReport report;
    try {
        detail::geometryCoordinates(Json::parse(coordinateContext));
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
            for (size_t row = 0; row < store.sourceIds().size(); ++row) {
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
                    publish(db, track(store, row, object), e, report, coordinateContext);
                } catch (const std::exception& error) { problem(report, key, error); }
            }
        } else {
            std::vector<Prepared> batch;
            std::map<std::string, Expected> expectations;
            {
                // Pin registration expectations while reading files. Publication
                // rechecks duplicate identities under the writer lock.
                auto read = db.read();
                if (kind == "patch") batch.push_back(patch(source));
                else if (kind == "fiber") {
                    if (fs::is_directory(source)) {
                        std::vector<fs::path> files;
                        for (const auto& entry : fs::directory_iterator(source))
                            if (entry.is_regular_file() && entry.path().extension() == ".json") files.push_back(entry.path());
                        std::sort(files.begin(), files.end());
                        for (const auto& file : files) if (readJson(file).value("type", "") == "vc3d_fiber") batch.push_back(fiber(file));
                    } else batch.push_back(fiber(source));
                }
                else if (kind == "point_collection") batch = collections(source);
                else throw std::runtime_error("unsupported geometry kind: " + kind);
                for (const auto& p : batch) expectations[p.object.uuid] = expectedAt(db, read, p.object.uuid);
            }
            publishBatch(db, std::move(batch), expectations, report, coordinateContext);
        }
    } catch (const std::exception& error) { problem(report, source.string(), error); }
    return report;
}
} // namespace vc::objects
