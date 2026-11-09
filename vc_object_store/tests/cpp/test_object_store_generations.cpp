#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/Database.hpp"
#include "internal/Digest.hpp"
#include <bit>

using namespace vc::objects;

namespace {
int64_t commitSequence(ObjectStore& store)
{
    auto txn = store.read();
    return store.provisioning(txn).commitSequence;
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
    {
        auto txn=store.read();auto* sql=ObjectStoreImplAccess::conn(txn);
        detail::Statement removed(sql,"SELECT count(*) FROM sqlite_schema WHERE name IN ('object_lifecycle','retired_objects')");
        require(removed.row()&&removed.i64(0)==0,"lifecycle has no separate table or index");
        detail::Statement columns(sql,"SELECT count(*) FROM pragma_table_info('objects') WHERE name IN ('state','lifecycle_epoch')");
        require(columns.row()&&columns.i64(0)==2,"lifecycle state is directly on objects");
        detail::Statement oldColumns(sql,"SELECT count(*) FROM pragma_table_info('objects') WHERE name IN ('retired','deleted')");
        require(oldColumns.row()&&oldColumns.i64(0)==0,"one state replaces independent lifecycle flags");
    }
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
        require(store.provisioning(txn).commitSequence == 1, "multiple operations commit one global generation");
        require(store.objectByUuid(txn, a)->version == 1 && store.objectByUuid(txn, b)->version == 1,
                "new objects start at version one despite repeated edits");
        require(store.edgeByUuid(txn, edge)->version == 1, "new edge starts at version one");
        require(store.objectByUuid(txn, a)->approvedVersion == 1 && store.edgeByUuid(txn, edge)->approvedVersion == 1,
                "approval covers the current record version");
        require(store.currentGeometry(txn, a)->objectVersion == 1,
                "geometry descriptor belongs to the object version");
        if (history == HistoryMode::On) {
            const auto geometries = store.geometryHistory(txn, a);
            require(geometries.size() == 3, "retain multiple geometry edits within one transaction");
            for (size_t i = 0; i < revisions.size(); ++i) {
                require(geometries[i].uuid == revisions[i] && geometries[i].objectVersion == 1,
                        "same-version geometry ancestry remains chronological");
                require(!store.readGeometryArray(txn, revisions[i], "line_points").empty(), "read intermediate geometry");
            }
            for (const auto& event : store.journalSince(txn, 0)) require(event.commitSequence == 1, "journal global stamp");
            for (const auto& version : store.objectHistory(txn, a))
                require(version.version == 1 && version.commitSequence == 1 &&
                        version.generation == store.status(txn).generation, "history shares record version and store identity");
        }
    }
    { auto txn = store.write(); store.setMeta(txn, b, "value", "7"); txn.commit(); }
    require(commitSequence(store) == 1, "same metadata no-op preserves generation");
    { auto txn = store.write(); store.setMeta(txn, b, "value", "8"); }
    require(commitSequence(store) == 1, "rollback preserves generation");
    {
        auto txn = store.write();
        // Failure rolls back the API savepoint, including any allocated ID.
        EdgeRecord relation; relation.kind = "bad"; relation.aUuid = a; relation.bUuid = util::newUuid();
        mustFail([&] { store.addEdge(txn, relation); });
        txn.commit();
    }
    require(commitSequence(store) == 1, "failed operation and empty commit preserve generation");
    {
        auto txn = store.write();
        mustFail([&] { detail::execute(ObjectStoreImplAccess::conn(txn),
            "INSERT INTO geometry_chunks VALUES(?,NULL)",detail::parseDigest(std::string(64,'0'))); });
        txn.commit();
    }
    require(commitSequence(store) == 1, "plain stores cannot silently acquire missing/external chunk payloads");
    {
        auto txn = store.write(); store.setApproval(txn, b, "approved"); txn.commit();
    }
    {
        auto txn = store.read();
        require(store.objectByUuid(txn, b)->version == 2 && store.objectByUuid(txn, b)->approvedVersion == 2,
                "approval alone advances object publication");
        require(store.objectByUuid(txn, a)->version == 1, "unchanged object retains original version");
    }
    int64_t cursor = 0;
    if (history == HistoryMode::On) { auto txn = store.read(); cursor = store.journalPosition(txn); }
    {
        auto txn = store.write(); store.removeEdge(txn, edge); store.deleteObject(txn, b); txn.commit();
    }
    require(commitSequence(store) == 3, "deletion advances generation");
    {
        auto txn=store.read();auto* sql=ObjectStoreImplAccess::conn(txn);
        require(!store.objectByUuid(txn,b)&&!store.edgeByUuid(txn,edge),
                "deleted objects and relations are absent from current UUID lookups");
        require(store.objectCount(txn,"fiber",true)==1&&store.edgesOf(txn,a,true).empty(),
                "include-retired queries do not expose deleted records");
        detail::Statement object(sql,"SELECT state,version FROM objects WHERE uuid=?",detail::parseUuid(b));
        require(object.row()&&object.i64(0)==2&&object.i64(1)==3,"object deletion is an ordinary stamped tombstone");
        detail::Statement relation(sql,"SELECT state,version,a_uuid,b_uuid FROM edges WHERE uuid=?",detail::parseUuid(edge));
        require(relation.row()&&relation.i64(0)==2&&relation.i64(1)==2,"edge deletion is an ordinary stamped tombstone");
        const auto first=detail::formatUuid(relation.blob16(2)),second=detail::formatUuid(relation.blob16(3));
        require((first==a&&second==b)||(first==b&&second==a),"tombstones retain original endpoint UUIDs");
    }
    if (history == HistoryMode::On) {
        auto txn = store.read();
        const auto events = store.journalSince(txn, cursor);
        require(events.size() == 2, "edge and object deletions recorded");
        for (const auto& event : events) require(event.commitSequence == 3, "deletions use committing generation, not zero");
        const auto versions = store.objectHistory(txn, b);
        require(versions.back().deleted && versions.back().version == 3 && versions.back().commitSequence == 3, "deleted history version global generation");
    }
    {
        auto txn=store.write();
        for(const auto& uuid:{b,edge}){
            mustFail([&]{store.setMeta(txn,uuid,"value","9");});
            mustFail([&]{store.removeMeta(txn,uuid,"absent");});
            mustFail([&]{store.setApproval(txn,uuid,"approved");});
        }
        mustFail([&]{store.putGeometry(txn,b,vc::geometry::encode(vc::geometry::Fiber{{},{{0,0,0},{1,0,0}},{}}));});
        txn.commit();
    }
    require(commitSequence(store)==3,"rejected tombstone edits do not publish a generation");
    {
        auto txn=store.write();store.upsertObject(txn,fiber(b));txn.commit();
    }
    {
        auto txn=store.read();
        require(store.objectByUuid(txn,b).has_value()&&store.objectCount(txn,"fiber",true)==2,
                "explicit object upsert revives a tombstone at the same UUID");
        require(!store.edgeByUuid(txn,edge),"object revival does not revive deleted relationships");
        require(store.objectByUuid(txn,b)->version == 4, "revival advances beyond the deletion version");
    }
}

void deletedTrackIdentity(HistoryMode history)
{
    TemporaryDirectory dir;
    OpenOptions options;options.author="deleted-track";options.history=history;
    auto store=ObjectStore::create(dir.path/"tracks.vcobj",options);
    ObjectRecord original;original.uuid=util::newUuid();original.kind="track";
    original.track.emplace();original.track->datasetRef="source";original.track->externalKey="track-1";
    {auto txn=store.write();store.upsertObject(txn,original);txn.commit();}
    {auto txn=store.write();store.deleteObject(txn,original.uuid);txn.commit();}
    auto replacement=original;replacement.uuid=util::newUuid();
    {auto txn=store.write();store.upsertObject(txn,replacement);txn.commit();}
    auto txn=store.read();
    require(!store.objectByUuid(txn,original.uuid)&&store.objectByUuid(txn,replacement.uuid).has_value(),
            "deleting a track releases its current source identity without deleting its UUID tombstone");
    require(store.objectCount(txn,"track",true)==1,"only the replacement track is current");
    if(history==HistoryMode::On){
        const auto versions=store.objectHistory(txn,original.uuid);
        require(versions.size()>=2&&versions.front().record.track&&
                versions.front().record.track->externalKey=="track-1"&&versions.back().deleted,
                "track deletion preserves historical source identity");
    }
}

void deletedGeometryRevival(HistoryMode history)
{
    TemporaryDirectory dir;
    OpenOptions options;options.author="deleted-geometry";options.history=history;
    auto store=ObjectStore::create(dir.path/"geometry.vcobj",options);
    const auto uuid=util::newUuid();std::string geometry;
    {auto txn=store.write();store.upsertObject(txn,fiber(uuid));
        geometry=store.putGeometry(txn,uuid,vc::geometry::encode(vc::geometry::Fiber{{},{{0,0,0},{1,0,0}},{}}));txn.commit();}
    VersionSelection selected;
    {auto txn=store.read();selected[uuid]=store.currentVersion(txn,uuid);}
    {auto txn=store.write(selected);store.retireObject(txn,uuid);txn.commit();}
    {auto txn=store.write();store.deleteObject(txn,uuid);txn.commit();}
    {auto txn=store.write();store.upsertObject(txn,fiber(uuid));txn.commit();}
    auto txn=store.read();const auto revived=store.objectByUuid(txn,uuid);
    require(revived&&!revived->retired&&revived->lifecycleEpoch.empty(),
            "explicit revival starts active rather than retaining a deleted lifecycle");
    require(!store.currentGeometry(txn,uuid),"explicit revival does not resurrect deleted current geometry");
    if(history==HistoryMode::On)
        require(store.readGeometryArray(txn,geometry,"line_points").size()==48,
                "deletion and revival retain archived historical geometry");
}

void deletedEdgeRevival(HistoryMode history)
{
    TemporaryDirectory dir;OpenOptions options;options.author="edge-revival";options.history=history;
    auto store=ObjectStore::create(dir.path/"edges.vcobj",options);
    const auto a=util::newUuid(),b=util::newUuid();
    EdgeRecord edge;edge.uuid=util::newUuid();edge.kind="perpendicular";edge.aUuid=a;edge.bUuid=b;
    edge.tags={"old"};edge.meta=R"({"version":1})";
    {auto txn=store.write();store.upsertObject(txn,fiber(a));store.upsertObject(txn,fiber(b));
        store.addEdge(txn,edge);txn.commit();}
    std::vector<ObjectVersion> original;
    if(history==HistoryMode::On){auto txn=store.read();const auto state=store.status(txn);
        for(const auto& uuid:{a,b})original.push_back({state.storeUuid,state.generation,uuid,*store.currentVersion(txn,uuid)});}
    {auto txn=store.write();store.removeEdge(txn,edge.uuid);txn.commit();}
    {auto txn=store.read();require(!store.edgeByUuid(txn,edge.uuid),"removed edge remains absent until explicit add");}
    edge.tags={"new"};edge.meta=R"({"version":2})";
    {auto txn=store.write();require(store.addEdge(txn,edge)==edge.uuid,"explicit recreation keeps edge UUID");txn.commit();}
    {auto txn=store.read();const auto current=store.edgeByUuid(txn,edge.uuid);
        require(current&&current->tags==edge.tags&&current->meta==edge.meta&&current->version==3,
                "explicit recreation replaces the deletion marker and old current tags");
        require(store.edgesOf(txn,a).size()==1&&store.edgesOf(txn,b).size()==1,"recreated edge is visible at both UUID endpoints");
        if(history==HistoryMode::On){
            const auto historical=store.relationshipsAt(txn,original);
            require(historical.size()==1&&historical.front().relationship.record.meta==R"({"version":1})",
                    "recreation retains original historical relationship");
        }}
    {auto txn=store.write();mustFail([&]{store.addEdge(txn,edge);});txn.commit();}
    require(commitSequence(store)==3,"explicit insertion still rejects a duplicate live edge UUID");
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
        require(commitSequence(source) == 6 && target.provisioning(txn).commitSequence == 1, "source and destination counters intentionally differ");
        require(record->version == 1 && record->approvedVersion == 1 && record->createdBy == "source",
                "import starts a local record version while retaining authorship and approval provenance");
        require(target.edgeByUuid(txn, edge)->version == 1 && target.edgeByUuid(txn, edge)->approvedVersion == 1,
                "imported relationship approval uses destination generation");
        require(target.objectByUuid(txn, b)->storageRef == "external.fiber", "external geometry reference remains content");
    }
    { auto txn = target.write(); target.importJson(txn, snapshot); txn.commit(); }
    require(commitSequence(target) == 1, "reimport remains idempotent despite differing publication generations");
    {
        auto changed = nlohmann::json::parse(snapshot);
        for (auto& record : changed["objects"]) if (record["uuid"] == b) record["storage_ref"] = "changed.fiber";
        auto txn = target.write(); mustFail([&] { target.importJson(txn, changed.dump()); }); txn.commit();
    }
    require(commitSequence(target) == 1, "external geometry reference mismatch remains a conflict");
    { auto txn = target.write(); target.setMeta(txn, a, "revision", "99"); txn.commit(); }
    { auto txn = target.write(); mustFail([&] { target.importJson(txn, snapshot); }); txn.commit(); }
    require(commitSequence(target) == 2, "real content conflict remains an error and changes nothing");
}


}

int main()
{
    nativeChanges(HistoryMode::On);
    nativeChanges(HistoryMode::Off);
    deletedTrackIdentity(HistoryMode::On);
    deletedTrackIdentity(HistoryMode::Off);
    deletedGeometryRevival(HistoryMode::On);
    deletedGeometryRevival(HistoryMode::Off);
    deletedEdgeRevival(HistoryMode::On);
    deletedEdgeRevival(HistoryMode::Off);
    metadataImport(HistoryMode::On);
    metadataImport(HistoryMode::Off);
}
