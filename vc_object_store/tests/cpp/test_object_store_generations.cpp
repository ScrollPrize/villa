#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/Hub.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/Database.hpp"
#include "sync/Apply.hpp"
#include "sync/GeometryTransfer.hpp"
#include <bit>

using namespace vc::objects;

namespace {
int64_t generation(ObjectStore& store)
{
    auto txn = store.read();
    return store.provisioning(txn).generation;
}

GeometryArrayEdit moveX(double value)
{
    GeometryArrayEdit edit{"line_points", 0, std::vector<uint8_t>(8)};
    const auto bits = std::bit_cast<uint64_t>(value);
    for (int i = 0; i < 8; ++i) edit.bytes[i] = uint8_t(bits >> (i * 8));
    return edit;
}

ObjectRecord fiber(const std::string& uuid)
{
    ObjectRecord record;
    record.uuid = uuid; record.kind = "fiber"; record.fiber = FiberDetail{};
    return record;
}

void nativeChanges(HistoryMode history)
{
    TemporaryDirectory dir;
    OpenOptions options; options.author = "native"; options.history = history;
    auto store = ObjectStore::create(dir.path / "native.vcobj", options);
    const auto a = util::newUuid(), b = util::newUuid();
    std::vector<std::string> revisions;
    std::string edge;
    {
        auto txn = store.write();
        store.upsertObject(txn, fiber(a));
        store.upsertObject(txn, fiber(b));
        revisions.push_back(store.putGeometry(txn, a, vc::geometry::encode(
            vc::geometry::Fiber{{}, {{0,0,0}, {10,0,0}}, {}})));
        revisions.push_back(store.editGeometry(txn, a, {moveX(1)}, revisions.back()));
        revisions.push_back(store.editGeometry(txn, a, {moveX(2)}, revisions.back()));
        EdgeRecord relation; relation.kind = "perpendicular"; relation.aUuid = a; relation.bUuid = b;
        relation.aAnchor.points = {{"line_points", {1}}}; relation.bAnchor.wholeGeometry = true;
        edge = store.addEdge(txn, relation);
        store.setMeta(txn, b, "value", "7");
        store.setApproval(txn, a, "approved");
        store.setApproval(txn, edge, "approved");
        txn.commit();
    }
    {
        auto txn = store.read();
        require(store.provisioning(txn).generation == 2, "multiple operations commit one global generation");
        require(store.objectByUuid(txn, a)->generation == 2 && store.objectByUuid(txn, b)->generation == 2,
                "every changed object uses committed generation");
        require(store.edgeByUuid(txn, edge)->generation == 2, "edge uses committed generation");
        require(store.objectByUuid(txn, a)->approvedGeneration == 2 && store.edgeByUuid(txn, edge)->approvedGeneration == 2,
                "approval covers the local publication generation");
        require(store.currentGeometry(txn, a)->objectGeneration == 2 && store.objectByUuid(txn, a)->dataGeneration == 2,
                "geometry and current data generation agree");
        if (history == HistoryMode::On) {
            const auto geometries = store.geometryHistory(txn, a);
            require(geometries.size() == 3, "retain multiple geometry edits within one transaction");
            for (size_t i = 0; i < revisions.size(); ++i) {
                require(geometries[i].uuid == revisions[i] && geometries[i].objectGeneration == 2,
                        "same-generation geometry ancestry remains chronological");
                require(!store.readGeometryArray(txn, revisions[i], "line_points").empty(), "read intermediate geometry");
            }
            for (const auto& event : store.journalSince(txn, 0)) require(event.generation == 2, "journal global stamp");
            for (const auto& version : store.objectHistory(txn, a)) require(version.generation == 2, "history global stamp");
        }
    }
    { auto txn = store.write(); store.setMeta(txn, b, "value", "7"); txn.commit(); }
    require(generation(store) == 2, "same metadata no-op preserves generation");
    { auto txn = store.write(); store.setMeta(txn, b, "value", "8"); }
    require(generation(store) == 2, "rollback preserves generation");
    {
        auto txn = store.write();
        // Failure rolls back the API savepoint, including any allocated ID.
        EdgeRecord relation; relation.kind = "bad"; relation.aUuid = a; relation.bUuid = util::newUuid();
        mustFail([&] { store.addEdge(txn, relation); });
        txn.commit();
    }
    require(generation(store) == 2, "failed operation and empty commit preserve generation");
    {
        auto txn = store.write();
        mustFail([&] { detail::execute(ObjectStoreImplAccess::conn(txn),
            "INSERT INTO geometry_chunks VALUES(?,NULL)",std::string(64,'0')); });
        txn.commit();
    }
    require(generation(store) == 2, "plain stores cannot silently acquire missing/external chunk payloads");
    {
        auto txn = store.write(); store.setApproval(txn, b, "approved"); txn.commit();
    }
    {
        auto txn = store.read();
        require(store.objectByUuid(txn, b)->generation == 3 && store.objectByUuid(txn, b)->approvedGeneration == 3,
                "approval alone advances object publication");
        require(store.objectByUuid(txn, a)->generation == 2, "unchanged object retains original generation");
    }
    int64_t cursor = 0;
    if (history == HistoryMode::On) { auto txn = store.read(); cursor = store.journalPosition(txn); }
    {
        auto txn = store.write(); store.removeEdge(txn, edge); store.deleteObject(txn, b); txn.commit();
    }
    require(generation(store) == 4, "deletion advances generation");
    if (history == HistoryMode::On) {
        auto txn = store.read();
        const auto events = store.journalSince(txn, cursor);
        require(events.size() == 2, "edge and object deletions recorded");
        for (const auto& event : events) require(event.generation == 4, "deletions use committing generation, not zero");
        const auto versions = store.objectHistory(txn, b);
        require(versions.back().deleted && versions.back().generation == 4, "deleted history version global generation");
    }
}

void metadataImport(HistoryMode history)
{
    TemporaryDirectory dir;
    OpenOptions options; options.author = "source"; options.history = history;
    auto source = ObjectStore::create(dir.path / "source.vcobj", options);
    const auto a = util::newUuid(), b = util::newUuid();
    std::string edge;
    {
        auto txn = source.write(); source.upsertObject(txn, fiber(a));
        auto external = fiber(b); external.storageKind = "file"; external.storageRef = "external.fiber";
        external.dataGeneration = 123;
        source.upsertObject(txn, external);
        EdgeRecord relation; relation.kind = "perpendicular"; relation.aUuid = a; relation.bUuid = b;
        edge = source.addEdge(txn, relation); txn.commit();
    }
    for (int i = 0; i < 4; ++i) {
        auto txn = source.write(); source.setMeta(txn, a, "revision", std::to_string(i)); txn.commit();
    }
    { auto txn = source.write(); source.setApproval(txn, a, "approved"); source.setApproval(txn, edge, "approved"); txn.commit(); }
    std::string snapshot;
    { auto txn = source.read(); snapshot = source.exportJson(txn); }
    options.author = "destination";
    auto target = ObjectStore::create(dir.path / "target.vcobj", options);
    { auto txn = target.write(); target.importJson(txn, snapshot); txn.commit(); }
    {
        auto txn = target.read();
        const auto record = target.objectByUuid(txn, a);
        require(generation(source) == 7 && target.provisioning(txn).generation == 2, "source and destination counters intentionally differ");
        require(record->generation == 2 && record->approvedGeneration == 2 && record->createdBy == "source",
                "import stamps local generation while retaining authorship and approval provenance");
        require(target.edgeByUuid(txn, edge)->generation == 2 && target.edgeByUuid(txn, edge)->approvedGeneration == 2,
                "imported relationship approval uses destination generation");
        require(target.objectByUuid(txn, b)->dataGeneration == 123, "external data version remains content, not a local counter");
    }
    { auto txn = target.write(); target.importJson(txn, snapshot); txn.commit(); }
    require(generation(target) == 2, "reimport remains idempotent despite differing publication generations");
    {
        auto changed = nlohmann::json::parse(snapshot);
        for (auto& record : changed["objects"]) if (record["uuid"] == b) record["data_generation"] = 124;
        auto txn = target.write(); mustFail([&] { target.importJson(txn, changed.dump()); }); txn.commit();
    }
    require(generation(target) == 2, "external data version mismatch remains a conflict");
    { auto txn = target.write(); target.setMeta(txn, a, "revision", "99"); txn.commit(); }
    { auto txn = target.write(); mustFail([&] { target.importJson(txn, snapshot); }); txn.commit(); }
    require(generation(target) == 3, "real content conflict remains an error and changes nothing");
}

void transferredGeometry()
{
    TemporaryDirectory dir;
    OpenOptions options; options.author = "source";
    auto source = ObjectStore::create(dir.path / "source.vcobj", options);
    auto target = ObjectStore::create(dir.path / "target.vcobj", options);
    const auto a = util::newUuid();
    std::string head;
    {
        auto txn = source.write(); source.upsertObject(txn, fiber(a));
        head = source.putGeometry(txn, a, vc::geometry::encode(vc::geometry::Fiber{{}, {{0,0,0}, {10,0,0}}, {}}));
        txn.commit();
    }
    for (int i = 0; i < 3; ++i) {
        auto txn = source.write(); head = source.editGeometry(txn, a, {moveX(i + 1)}, head); txn.commit();
    }
    {
        auto txn = source.write(); source.setApproval(txn, a, "approved"); txn.commit();
    }
    const auto transfer = [&] {
        auto from = source.read(); auto to = target.write();
        const auto record = detail::syncRecord(source, from, a);
        nlohmann::json before = nlohmann::json::object(), merged = {{a, record}};
        const auto current = detail::syncRecord(target, to, a);
        if (!current.is_null()) before[a] = current;
        detail::applyObjects(target, to, before, merged);
        detail::transferGeometry(ObjectStoreImplAccess::conn(from), ObjectStoreImplAccess::conn(to), a, 0);
        detail::applyHeadsAndEdges(target, to, before, merged);
        require(detail::syncRecord(target, to, a) == record, "content identity excludes database-local counters");
        to.commit();
    };
    transfer();
    {
        auto txn = target.read();
        require(target.objectByUuid(txn, a)->generation == 2 && target.objectByUuid(txn, a)->dataGeneration == 2,
                "transferred records and current geometry publish locally");
        for (const auto& revision : target.geometryHistory(txn, a)) {
            require(revision.objectGeneration == 2, "transferred historical geometry installed in local generation");
            require(!target.readGeometryArray(txn, revision.uuid, "line_points").empty(), "transferred ancestry remains readable");
        }
    }
    transfer();
    require(generation(target) == 2, "repeat transfer is a no-op despite source geometry generation differences");
}

void hostedPolicy()
{
    TemporaryDirectory dir;
    OpenOptions options; options.author = "curator";
    const auto directPath = dir.path / "direct.vcobj", sealedPath = dir.path / "sealed.vcobj";
    {
        options.history = HistoryMode::Off;
        auto direct = ObjectStore::create(directPath, options);
        direct.checkpoint();
    }
    mustFail([&] { initHub(directPath, "curator"); });
    {
        auto direct = ObjectStore::open(directPath, options);
        require(!hubStatus(direct) && generation(direct) == 1, "history-off hub setup rejected without logical changes");
    }
    options.history = HistoryMode::On;
    {
        auto sealed = ObjectStore::create(sealedPath, options);
        auto txn = sealed.write(); sealed.seal(txn); txn.commit(); sealed.checkpoint();
    }
    mustFail([&] { initHub(sealedPath, "curator"); });
    const auto hubPath = dir.path / "hub.vcobj", replicaPath = dir.path / "replica.vcobj";
    { auto initial = ObjectStore::create(hubPath, options); initial.checkpoint(); }
    initHub(hubPath, "curator");
    DatabaseIdentity hubIdentity;
    {
        auto shared = options; shared.sharedFile = true;
        auto hub = ObjectStore::open(hubPath, shared);
        auto txn = hub.read();
        require(hub.provisioning(txn).generation == 2, "hub provisioning publishes a generation");
        hubIdentity = hub.databaseIdentity(txn);
        bool found = false;
        for (const auto& table : hub.databaseIdentity(txn).tables) found |= table.name == "hub_leases";
        require(found, "hub provisioning registers added tables");
    }
    cloneHub(hubPath, replicaPath, options);
    int64_t beforeDetach;
    {
        auto replica = ObjectStore::open(replicaPath, options); beforeDetach = generation(replica);
        auto txn = replica.read();
        const auto identity = replica.databaseIdentity(txn);
        require(identity.uuid != hubIdentity.uuid, "replica gets its own physical database UUID");
        for (const auto& table : identity.tables) for (const auto& original : hubIdentity.tables)
            require(table.uuid != original.uuid, "copied table instances get independent UUIDs");
    }
    detachHub(replicaPath, "curator");
    {
        auto detached = ObjectStore::open(replicaPath, options);
        auto txn = detached.read();
        require(!hubStatus(detached) && detached.provisioning(txn).generation == beforeDetach + 1,
                "hub detachment publishes a generation");
        for (const auto& table : detached.databaseIdentity(txn).tables)
            require(!table.name.starts_with("hub_"), "hub detachment removes directory entries");
    }
    {
        auto extended = ObjectStore::create(dir.path / "extension.vcobj", options);
        auto txn = extended.write();
        detail::exec(ObjectStoreImplAccess::conn(txn),
            "CREATE TABLE custom_reference(table_uuid BLOB REFERENCES table_directory(uuid));");
        txn.commit();
        const auto before = generation(extended);
        mustFail([&] { extended.forkReplica(); });
        require(generation(extended) == before, "unknown physical identity references refuse fork atomically");
    }
}
}

int main()
{
    nativeChanges(HistoryMode::On);
    nativeChanges(HistoryMode::Off);
    metadataImport(HistoryMode::On);
    metadataImport(HistoryMode::Off);
    transferredGeometry();
    hostedPolicy();
}
