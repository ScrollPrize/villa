#include "geometry/GeometryIngestBatch.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"
#include "vc_object_store/objects/GeometryIngest.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/PackedTrackStore.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include <cstdio>
#include <fstream>
#include <limits>
#include "vc_object_store/Uuid.hpp"
#include "formats/GeometryInput.hpp"
#include "formats/TrackInput.hpp"
#include "formats/RemoteSource.hpp"
#include "internal/Publication.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <functional>
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
// One committed write transaction (placeholder ingestion).
using Commit = std::function<void(const std::function<void(ObjectStore&, WriteTxn&)>&)>;
struct Expected { std::optional<int64_t> version; };
Expected expectedAt(ObjectStore& db, const Txn& read, const std::string& id)
{
    Expected e;
    if (const auto object = db.objectByUuid(read, id)) e.version = object->version;
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
    std::vector<std::string> created;
    commit([&](ObjectStore& db, WriteTxn& write) {
        created.clear();
        for (const auto& p : batch) {
            const auto existing = db.objectByUuid(write, p.object.uuid);
            const auto& e = expectations.at(p.object.uuid);
            if ((existing ? existing->version : std::nullopt) != e.version)
                throw GeometrySourceRejected("object changed while reading source; retry ingestion");
            if (existing && existing->kind != p.object.kind) throw GeometrySourceRejected("source UUID belongs to another object kind");
        }
        for (const auto& p : batch) created.push_back(detail::sourceUuid(p.object.uuid));
        edges = detail::ingestGeometryBatch(db, write, std::move(batch), coordinateContext, registeredInitial);
    });
    report.ingested += count; report.edgesImported += edges;
    report.uuids.insert(report.uuids.end(), created.begin(), created.end());
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
// Describe the SourceIngestOptions repairs recorded in a prepared batch's metadata.
std::vector<std::string> repairNotices(const std::vector<Prepared>& batch, const fs::path& source)
{
    std::vector<std::string> result;
    size_t renumbered = 0;
    for (const auto& p : batch) {
        if (p.source.value("point_ids_renumbered", false)) ++renumbered;
        if (!p.source.contains("dropped_branches")) continue;
        std::string targets;
        for (const auto& branch : p.source.at("dropped_branches"))
            targets += (targets.empty() ? "" : ", ") + branch.at("branch_file").get<std::string>();
        result.push_back("fiber " + p.object.uuid + " (" + p.object.name + "): dropped " +
            std::to_string(p.source.at("dropped_branches").size()) + " branch declarations to absent files: " + targets);
    }
    if (renumbered)
        result.push_back(source.string() + ": renumbered point IDs of " + std::to_string(renumbered) +
            " collections sharing IDs; file IDs kept as source_point_id");
    return result;
}
}

std::vector<GeometrySource> prepareGeometrySource(const std::string& kind, const fs::path& source,
    const std::vector<std::string>& availableUuids, const SourceIngestOptions& options)
{
    if (options.uuidNamespace) util::canonicalUuid(*options.uuidNamespace);
    if (kind == "fiber") return formats::fibers(source, availableUuids, options);
    if (kind == "surface") return {surface(source, options.uuidNamespace)};
    if (kind == "point_collection") return collections(source, options);
    if (kind == "umbilicus") return {formats::umbilicus(source)};
    throw std::invalid_argument("unsupported batch geometry kind: " + kind);
}

GeometryIngestReport ingestRegisteredGeometry(ObjectStore& db, const std::optional<fs::path>& datasetRoot,
    const std::vector<std::string>& objectUuids, const util::ProgressCallback& progress,
    const std::string& coordinateContext)
{
    geometryCoordinates(Json::parse(coordinateContext));
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
                std::optional<fs::path> path;
                { auto read = db.read(); path = db.resolveStoragePath(read, object.uuid, datasetRoot); }
                if (!path) throw std::runtime_error("track source is unavailable");
                const auto packed = resolvePackedTrackStore(*path);
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
                p = formats::readRemoteSurfaceGeometry(object.storageRef);
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
        try { ingestBatch(directCommit(db), std::move(prepared), expectations, report, coordinateContext, true); }
        catch (const std::exception& error) { problem(report, "batch", error); }
    }
    util::reportProgress(progress, "ingesting geometry", objects.size(), objects.size());
    return report;
}

namespace {
// Where ingestion reads and commits: one checked request per commit.
struct IngestTarget {
    std::function<void(const std::function<void(const StoreReader&)>&)> read;
    std::function<TransactionRequest()> newRequest;
    std::function<TransactionReceipt(const TransactionRequest&)> submit;
    // Stops before the next surface or track, or before a batch commits.
    std::function<bool()> cancelled;
};

// Ends ingestion at a failure that is not the source's (stopOnInfrastructureError).
struct Stopped {};
// Reports the failure of the step `label`. A rejected source is a problem of
// that source. Any other failure is the store's: a problem as well, unless the
// options stop ingestion at it.
void failed(GeometryIngestReport& report, const SourceIngestOptions& options, const std::string& label,
    const std::exception& error)
{
    if (const auto* exists = dynamic_cast<const GeometryIdentitiesExist*>(&error))
        report.existing.insert(report.existing.end(), exists->uuids.begin(), exists->uuids.end());
    if (dynamic_cast<const GeometrySourceRejected*>(&error) || !options.stopOnInfrastructureError) {
        problem(report, label, error);
        return;
    }
    report.fatalError = label + ": " + error.what();
    throw Stopped{};
}
// A step that only reads and decodes source files or the caller's request:
// whatever fails there rejects the source.
template <class Step> auto fromSource(Step&& step)
{
    try { return step(); }
    catch (const GeometrySourceRejected&) { throw; }
    catch (const std::exception& error) { throw GeometrySourceRejected(error.what()); }
}

void commitBatch(const IngestTarget& target, std::vector<Prepared> batch, GeometryIngestReport& report,
    const std::string& coordinateContext)
{
    const auto count = int64_t(batch.size());
    std::vector<std::string> created;
    for (const auto& p : batch) created.push_back(detail::sourceUuid(p.object.uuid));
    auto request = target.newRequest();
    target.read([&](const StoreReader& reader) {
        // Identities the store already has (including retired and deleted
        // objects) reject the whole batch, listing every clash.
        std::vector<std::string> clashes;
        for (const auto& uuid : created)
            if (reader.currentObject(uuid)) clashes.push_back(uuid);
        if (!clashes.empty()) throw GeometryIdentitiesExist(std::move(clashes));
        addIngestOperation(request, reader, std::move(batch), coordinateContext);
    });
    TransactionReceipt receipt;
    // The request expects every new object absent and every existing endpoint
    // at the version read: a conflict is the store changing under the source.
    try { receipt = target.submit(request); }
    catch (const TransactionConflict& conflict) { throw GeometrySourceRejected(conflict.what()); }
    // The new objects, then the relationships their sources declare.
    report.ingested += count; report.edgesImported += int64_t(receipt.created.size()) - count;
    report.uuids.insert(report.uuids.end(), created.begin(), created.end());
}

void ingestSource(GeometryIngestReport& report, const IngestTarget& target, const std::string& kind,
    const fs::path& source, const util::ProgressCallback& progress, const std::string& coordinateContext,
    const std::vector<std::string>& tags, const SourceIngestOptions& options)
try {
    auto one = [&](Prepared p) {
        std::vector<Prepared> batch;
        batch.push_back(std::move(p));
        commitBatch(target, std::move(batch), report, coordinateContext);
    };
    try {
        fromSource([&] {
            geometryCoordinates(Json::parse(coordinateContext));
            if (options.uuidNamespace) util::canonicalUuid(*options.uuidNamespace);
            return 0;
        });
        if (kind == "track") {
            const auto packed = fromSource([&] { return resolvePackedTrackStore(source); });
            const auto store = fromSource([&] { return std::make_unique<PackedTrackStore>(packed); });
            std::map<std::string, ObjectRecord> existing;
            std::set<std::string> refs;
            target.read([&](const StoreReader& reader) {
                ObjectSelection tracks; tracks.kind = "track"; tracks.includeRetired = true;
                for (const auto& object : reader.selectObjects(tracks)) {
                    if (!object.track) continue;
                    const auto path = reader.resolveStoragePath(object.uuid);
                    if (!path) continue;
                    std::error_code error;
                    if (fs::weakly_canonical(*path, error) == packed && !error) {
                        refs.insert(path->string()); existing.emplace(object.track->externalKey, object);
                    }
                }
            });
            if (refs.size() > 1) throw GeometrySourceRejected("track dataset registered under multiple aliases");
            const auto ref = refs.empty() ? packed.string() : *refs.begin();
            size_t row = 0;
            for (; row < store->sourceIds().size() && !target.cancelled(); ++row) {
                const auto key = std::to_string(store->sourceIds()[row]);
                util::reportProgress(progress, "ingesting tracks", row, store->sourceIds().size(), key);
                try {
                    ObjectRecord object;
                    if (existing.contains(key)) object = existing.at(key);
                    else {
                        object.uuid = util::newUuid(); object.kind = "track"; object.name = key;
                        object.storageKind = "dataset"; object.storageRef = ref;
                        object.track = TrackDetail{}; object.track->datasetRef = ref; object.track->externalKey = key;
                    }
                    auto prepared = fromSource([&] {
                        auto p = track(*store, row, object);
                        if (fs::exists(packed / "metadata.json")) p.source = readJson(packed / "metadata.json");
                        return p;
                    });
                    addTags(prepared, tags);
                    one(std::move(prepared));
                } catch (const std::exception& error) { failed(report, options, key, error); }
            }
            report.cancelled = row < store->sourceIds().size();
        } else if (kind == "surface") {
            // Surfaces have no native relationships among themselves: each commits on its own.
            const auto directories = fromSource([&] { return surfaceDirectories(source); });
            size_t i = 0;
            for (; i < directories.size() && !target.cancelled(); ++i) {
                const auto& directory = directories[i];
                util::reportProgress(progress, "ingesting surfaces", i, directories.size(), directory.filename().string());
                try {
                    auto prepared = fromSource([&] { return surface(directory, options.uuidNamespace); });
                    addTags(prepared, tags);
                    one(std::move(prepared));
                } catch (const std::exception& error) { failed(report, options, directory.string(), error); }
            }
            report.cancelled = i < directories.size();
            util::reportProgress(progress, "ingesting surfaces", directories.size(), directories.size());
        } else {
            std::vector<std::string> available;
            target.read([&](const StoreReader& reader) {
                ObjectSelection objects; objects.kind = kind;
                for (const auto& object : reader.selectObjects(objects, std::vector<std::string>{"uuid"}))
                    available.push_back(object.uuid);
            });
            auto batch = fromSource([&] { return prepareGeometrySource(kind, source, available, options); });
            for (auto& p : batch) addTags(p, tags);
            const auto notices = repairNotices(batch, source);
            // The batch commits whole or not at all.
            if (target.cancelled()) { report.cancelled = true; return; }
            commitBatch(target, std::move(batch), report, coordinateContext);
            report.notices.insert(report.notices.end(), notices.begin(), notices.end());
        }
    } catch (const std::exception& error) { failed(report, options, source.string(), error); }
} catch (const Stopped&) {}
}

GeometryIngestReport ingestGeometrySource(ObjectStore& db, const std::string& kind, const fs::path& source,
    const util::ProgressCallback& progress, const std::string& coordinateContext, const std::vector<std::string>& tags,
    const SourceIngestOptions& options)
{
    IngestTarget target;
    target.read = [&](const auto& fn) { auto txn = db.read(); fn(LocalStoreReader(db, txn)); };
    target.newRequest = [&] {
        TransactionRequest request;
        StoreStatus current;
        { auto txn = db.read(); current = db.status(txn); }
        request.requestUuid = util::newUuid(); request.storeUuid = current.storeUuid;
        request.generation = current.generation; request.author = db.author();
        return request;
    };
    target.submit = [&](const TransactionRequest& request) { return db.submit(request); };
    target.cancelled = [] { return false; };
    GeometryIngestReport report;
    ingestSource(report, target, kind, source, progress, coordinateContext, tags, options);
    return report;
}

GeometryIngestReport ingestGeometrySource(ObjectStoreSession& session, const AttachmentIdentity& attachment,
    const std::string& kind, const fs::path& source, const util::ProgressCallback& progress,
    const std::string& coordinateContext, const std::vector<std::string>& tags, const SourceIngestOptions& options,
    const std::atomic_bool* cancel)
{
    GeometryIngestReport report;
    if (!session.writable(attachment)) {
        const std::string error = "object store: ingestion needs a writable attachment";
        if (!options.stopOnInfrastructureError) throw std::invalid_argument(error);
        report.fatalError = error;
        return report;
    }
    IngestTarget target;
    target.read = [&](const auto& fn) {
        session.readStore(attachment, [&](const StoreReader& reader, const auto&) { fn(reader); });
    };
    target.newRequest = [&] { return session.newRequest(attachment); };
    target.submit = [&](const TransactionRequest& request) { return session.submit(attachment, request); };
    target.cancelled = [&] { return !session.current(attachment) || (cancel && cancel->load()); };
    ingestSource(report, target, kind, source, progress, coordinateContext, tags, options);
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

namespace vc::objects {
namespace {
GeometryIngestReport bulkIngest(const fs::path& source, const fs::path& destination, const std::string& author,
    const std::string& coordinateContext, const std::vector<std::string>& tags, std::size_t shards, InitialImportResult& sealed)
{
    GeometryIngestReport report;
    if (shards == 0) shards = 1;
    if (fs::exists(destination)) throw std::runtime_error("bulk ingest: destination exists: " + destination.string());
    geometryCoordinates(nlohmann::json::parse(coordinateContext));
    const auto packed = resolvePackedTrackStore(source);
    PackedTrackStore store(packed);
    const size_t rows = store.sourceIds().size();
    nlohmann::json sourceMeta = nlohmann::json::object();
    if (fs::exists(packed / "metadata.json")) sourceMeta = readJson(packed / "metadata.json");
    const std::string ref = packed.string();
    // Fresh identities, routed to their shard (the initial-import API requires uuidShardIndex routing).
    std::vector<std::string> uuids(rows);
    std::vector<std::vector<uint32_t>> byShard(shards);
    for (size_t i = 0; i < rows; ++i) { uuids[i] = util::newUuid(); byShard[uuidShardIndex(uuids[i], shards)].push_back(uint32_t(i)); }
    InitialImportOptions options;
    options.store.author = author;
    options.shards.assign(shards, InitialShardSpec{});
    const int64_t per = std::numeric_limits<int64_t>::max() / int64_t(shards);
    for (size_t k = 0; k < shards; ++k) options.shards[k].objectIds = {int64_t(k) * per + 1, int64_t(k + 1) * per};
    std::atomic<int64_t> done{0};
    sealed = importInitialCollection(destination, options, [&](size_t shardIndex, InitialShardImport& shard) {
        std::vector<InitialObjectRecord> batch;
        auto flush = [&] { if (batch.empty()) return; shard.appendObjects(batch); done += int64_t(batch.size()); batch.clear(); };
        for (const uint32_t row : byShard[shardIndex]) {
            ObjectRecord object;
            object.uuid = uuids[row]; object.kind = "track"; object.name = std::to_string(store.sourceIds()[row]);
            object.storageKind = "dataset"; object.storageRef = ref;
            object.track = TrackDetail{}; object.track->datasetRef = ref; object.track->externalKey = object.name;
            auto p = track(store, row, object);
            p.source = sourceMeta;
            addTags(p, tags);
            detail::normalizeIngestMetadata(p, coordinateContext);
            InitialObjectRecord record; record.record = std::move(p.object); record.geometry = std::move(p.geometry);
            batch.push_back(std::move(record));
            if (batch.size() >= 2000) { flush(); shard.checkCancelled(); }
        }
        flush();
    });
    report.ingested = done.load();
    report.uuids = std::move(uuids);        // the objects created, as the per-track path reports them
    return report;
}
}

GeometryIngestReport ingestTracksBulk(const fs::path& source, const fs::path& destination, const std::string& author,
    const std::string& coordinateContext, const std::vector<std::string>& tags, std::size_t shards)
{
    InitialImportResult sealed;
    return bulkIngest(source, destination, author, coordinateContext, tags, shards, sealed);
}
}

namespace vc::objects {
namespace {
using Stage = BulkOverlayError::Stage;
nlohmann::json bulkMarkerRead(const fs::path& marker)
{
    std::ifstream in(marker);
    if (!in) return nullptr;
    try { return nlohmann::json::parse(in); } catch (const std::exception&) { return nullptr; }
}
// Written only after the sealed tracks are published; it is what lets a retry tell "ours, finished" from "somebody else's".
// Published WITHOUT replacing: a hard link fails if the marker path exists (a marker created meanwhile by another writer is preserved); on a
// filesystem without hard links an exclusive create ("wx") is used instead, whose half-written result a retry reads as foreign and refuses.
void bulkMarkerWrite(const fs::path& marker, const nlohmann::json& body)
{
    const auto temporary = fs::path(marker.string() + "." + util::newUuid() + ".tmp");
    const std::string text = body.dump(2) + "\n";
    {
        std::ofstream out(temporary, std::ios::binary);
        out << text;
        out.flush();
        if (!out) { std::error_code ignored; fs::remove(temporary, ignored); throw std::runtime_error("cannot write " + temporary.string()); }
    }
    std::error_code error;
    fs::create_hard_link(temporary, marker, error);
    if (error && error != std::errc::file_exists) {
        if (std::FILE* file = std::fopen(marker.string().c_str(), "wx")) {
            const bool ok = std::fwrite(text.data(), 1, text.size(), file) == text.size();
            if (std::fclose(file) == 0 && ok) error.clear();
            else { std::error_code ignored; fs::remove(marker, ignored); }
        }
    }
    std::error_code ignored; fs::remove(temporary, ignored);
    if (error) throw std::runtime_error("cannot publish " + marker.string() + " (an existing marker is never replaced): " + error.message());
}
// Cheap identity of the packed source: every file's relative path and size, its modification time, and the bytes of the small files (header and
// metadata). A changed source at the same path therefore does not match a marker written for the old one (a copied source reads as changed:
// the retry then asks for the marker to be removed rather than silently reusing tracks that may differ).
std::string bulkSourceFingerprint(const fs::path& packed)
{
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void* data, std::size_t n) { const auto* b = static_cast<const unsigned char*>(data); for (std::size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; } };
    std::vector<fs::path> files;
    std::error_code error;
    if (fs::is_directory(packed, error)) { for (fs::recursive_directory_iterator it(packed, error), end; !error && it != end; it.increment(error)) if (it->is_regular_file(error)) files.push_back(it->path()); }
    else files.push_back(packed);
    std::sort(files.begin(), files.end());
    for (const auto& file : files) {
        const auto name = fs::is_directory(packed, error) ? fs::relative(file, packed, error).generic_string() : file.filename().string();
        const std::uint64_t size = fs::file_size(file, error), stamp = std::uint64_t(fs::last_write_time(file, error).time_since_epoch().count());
        mix(name.data(), name.size()); mix(&size, sizeof size); mix(&stamp, sizeof stamp);
        if (size <= (1u << 20)) { std::ifstream in(file, std::ios::binary); std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()); mix(bytes.data(), bytes.size()); }
    }
    char text[32]; std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(h));
    return text;
}
// The content-affecting options, normalised: coordinate frame as parsed JSON (key order irrelevant), tags sorted and deduplicated, author.
nlohmann::json bulkOptions(const std::string& coordinateContext, const std::vector<std::string>& tags, const std::string& author)
{
    std::vector<std::string> sorted = tags; std::sort(sorted.begin(), sorted.end()); sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    return {{"coordinate_context", nlohmann::json::parse(coordinateContext)}, {"tags", sorted}, {"author", author}};
}
// Identity of the COMPLETE sealed publication (every member of the base collection: store and lineage UUIDs, generation, exact commit sequence), sorted.
// Recorded in the marker when the sealed tracks are published and compared on every retry, so a replacement store (a different valid base, one shard of a
// four-shard collection, an emptier collection) is refused before an overlay is attached to it.
nlohmann::json bulkMembersIdentity(const std::vector<CollectionMember>& members, std::size_t first = 0)
{
    std::vector<nlohmann::json> rows;
    for (std::size_t i = first; i < members.size(); ++i)
        rows.push_back({{"database_uuid", members[i].databaseUuid}, {"store_uuid", members[i].storeUuid}, {"generation", members[i].generation}, {"commit_sequence", members[i].commitSequence}});
    std::sort(rows.begin(), rows.end(), [](const nlohmann::json& a, const nlohmann::json& b) { return a.dump() < b.dump(); });
    return nlohmann::json(rows);
}
nlohmann::json bulkBaseIdentity(const fs::path& base)
{
    OpenOptions readOnly; readOnly.readOnly = true;
    auto root = ObjectCollection::open(base, readOnly);
    std::int64_t objects = 0;
    { auto txn = root.read(); objects = root.objectCount(txn, "track"); }          // the number of tracks the publication really holds: a count, not a scan that decodes every record (review of PR 2588)
    return {{"members", bulkMembersIdentity(root.members())}, {"objects", objects}};
}
bool bulkMarkerMatches(const nlohmann::json& marker, const std::string& packed, const std::string& fingerprint, const nlohmann::json& options, std::size_t shards, std::size_t rows)
{
    return marker.is_object() && marker.value("kind", std::string()) == "vc_object_ingest_bulk" && marker.value("source", std::string()) == packed
        && marker.value("fingerprint", std::string()) == fingerprint && marker.contains("options") && marker.at("options") == options
        && marker.value("shards", std::size_t(0)) == shards && marker.value("tracks", std::size_t(0)) == rows;
}
}

BulkOverlayReport ingestTracksBulkOverlay(const fs::path& source, const fs::path& overlayPath, const std::string& author,
    const std::string& coordinateContext, const std::vector<std::string>& tags, std::size_t shards)
{
    if (shards == 0) shards = 1;
    BulkOverlayReport out;
    out.tracksPath = fs::path(overlayPath.string() + ".tracks");
    out.marker = fs::path(overlayPath.string() + ".tracks.bulk.json");
    std::string packedText, fingerprint; nlohmann::json options; std::size_t rows = 0;
    try {
        const auto packed = resolvePackedTrackStore(source);
        packedText = packed.string(); rows = PackedTrackStore(packed).sourceIds().size();
        geometryCoordinates(nlohmann::json::parse(coordinateContext));
        fingerprint = bulkSourceFingerprint(packed); options = bulkOptions(coordinateContext, tags, author);
    } catch (const std::exception& error) {
        throw BulkOverlayError(Stage::Preflight, std::string("bulk ingest: ") + error.what());
    }
    const auto recorded = bulkMarkerRead(out.marker);
    const bool ours = bulkMarkerMatches(recorded, packedText, fingerprint, options, shards, rows);
    std::error_code markerProbe;
    if (!ours && (fs::exists(out.marker, markerProbe) || fs::is_symlink(out.marker, markerProbe)))
        throw BulkOverlayError(Stage::Preflight, "bulk ingest: " + out.marker.string() + " exists and does not match this command (another writer's file, a different source, "
            "source contents, coordinate frame, tags, author or shard count, or a damaged record); it is left untouched. Remove it (and the sealed tracks it describes) to start over.");
    const auto base = shards > 1 ? out.tracksPath / "collection.vcobj" : out.tracksPath;
    OpenOptions rootOptions; rootOptions.author = author;
    auto overlayError = [&](const std::exception& error) {
        return BulkOverlayError(Stage::Overlay, std::string("bulk ingest: the overlay ") + overlayPath.string() + " could not be created: " + error.what()
            + ". The sealed tracks are kept at " + out.tracksPath.string() + " (recorded in " + out.marker.string() + "); no file of another writer was touched. "
            "Remove whatever the message names, then run the same command again: it reuses the sealed tracks and only creates the overlay.");
    };

    if (fs::exists(overlayPath) || fs::is_symlink(overlayPath)) {
        if (!ours) throw BulkOverlayError(Stage::Preflight, "bulk ingest: the overlay path exists and this command did not make it: " + overlayPath.string());
        // Read-only: opening verifies every dependency pin; then the overlay must depend on EXACTLY the sealed publication recorded when this command made it
        // (every shard, same UUIDs and commit sequences), and the base that sits there now must still be that publication.
        try {
            const auto& wanted = recorded.at("base_members");
            OpenOptions readOnly; readOnly.readOnly = true;
            if (bulkBaseIdentity(base) != wanted) throw std::runtime_error("the sealed tracks at " + out.tracksPath.string() + " are no longer the publication this command made (a different or partial store sits there)");
            auto root = ObjectCollection::open(overlayPath, readOnly);
            const auto members = root.members();
            if (members.size() < 2 || bulkMembersIdentity(members, 1) != wanted.at("members")) throw std::runtime_error("it does not depend on the complete sealed tracks recorded for this command (" + out.tracksPath.string() + ")");
            auto writableRoot = ObjectCollection::open(overlayPath, rootOptions);
            (void)writableRoot;
        } catch (const std::exception& error) { throw overlayError(error); }
        out.overlayAlreadyComplete = true; out.reusedSealedTracks = true; out.ingest.ingested = int64_t(rows);
        return out;
    }
    const bool tracksPresent = fs::exists(out.tracksPath) || fs::is_symlink(out.tracksPath);
    nlohmann::json recordedBase;
    if (ours && !tracksPresent)
        throw BulkOverlayError(Stage::Preflight, "bulk ingest: " + out.marker.string() + " records sealed tracks that are missing now (" + out.tracksPath.string()
            + "); nothing was re-ingested and no file was touched. Remove the marker to start over, or restore the tracks.");
    if (tracksPresent) {
        if (!ours) throw BulkOverlayError(Stage::Preflight, "bulk ingest: the tracks destination exists and this command did not make it (no matching "
            + out.marker.string() + "); it is left untouched: " + out.tracksPath.string());
        try {
            if (!recorded.contains("base_members") || bulkBaseIdentity(base) != recorded.at("base_members"))
                throw std::runtime_error("the sealed tracks at " + out.tracksPath.string() + " are not the complete publication this command made (a different, partial or replaced store); "
                    "they and the marker are left untouched, no overlay was created");
        } catch (const std::exception& error) { throw BulkOverlayError(Stage::Preflight, std::string("bulk ingest: ") + error.what()); }
        recordedBase = recorded.at("base_members");
        out.reusedSealedTracks = true; out.ingest.ingested = int64_t(rows);
    } else {
        InitialImportResult sealed;
        try { out.ingest = bulkIngest(source, out.tracksPath, author, coordinateContext, tags, shards, sealed); }
        catch (const std::exception& error) { throw BulkOverlayError(Stage::SealedTracks, std::string("bulk ingest: ") + error.what()); }
        try {
            recordedBase = bulkBaseIdentity(base);
            const auto& members = recordedBase.at("members");
            if (recordedBase.at("objects") != out.ingest.ingested
                || std::any_of(members.begin(), members.end(), [&](const nlohmann::json& member) { return member.at("store_uuid") != sealed.storeUuid; }))
                throw std::runtime_error("the store at " + out.tracksPath.string() + " is not the one this command sealed");
            bulkMarkerWrite(out.marker, {{"base_members", recordedBase}, {"kind", "vc_object_ingest_bulk"}, {"source", packedText}, {"fingerprint", fingerprint}, {"options", options}, {"shards", shards}, {"tracks", rows}, {"author", author},
                                         {"tracks_path", out.tracksPath.string()}});
        } catch (const std::exception& error) {
            throw BulkOverlayError(Stage::Marker, std::string("bulk ingest: the sealed tracks were published at ") + out.tracksPath.string() + " but could not be recorded ("
                + error.what() + "); they are kept. Create " + overlayPath.string() + " over them with ObjectCollection::create, or remove them and run again.");
        }
    }
    // The overlay is verified under a private name, so a failure never leaves one at overlayPath over different tracks.
    try {
        detail::StagedDatabase staged(overlayPath, false);
        {
            auto root = ObjectCollection::create(staged.path, {base}, rootOptions);
            const auto members = root.members();
            if (members.size() < 2 || bulkMembersIdentity(members, 1) != recordedBase.at("members"))
                throw std::runtime_error("the sealed tracks at " + out.tracksPath.string() + " changed while the overlay was attached");
        }
        staged.publish();
        auto root = ObjectCollection::open(overlayPath, rootOptions); (void)root;
    } catch (const std::exception& error) { throw overlayError(error); }
    return out;
}
}
