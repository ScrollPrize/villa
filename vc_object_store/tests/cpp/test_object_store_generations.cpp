#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/Database.hpp"
#include "internal/Digest.hpp"
#include "internal/RecordResolver.hpp"
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

void transactionAuthors(HistoryMode history)
{
    TemporaryDirectory dir;
    OpenOptions options; options.author = "store-owner"; options.history = history;
    auto store = ObjectStore::create(dir.path / "authors.vcobj", options);
    const auto a = util::newUuid(), b = util::newUuid();
    {
        auto txn = store.write();
        store.upsertObject(txn, fiber(a)); store.upsertObject(txn, fiber(b));
        txn.commit();
    }
    EdgeRecord edge; edge.uuid = util::newUuid(); edge.kind = "perpendicular";
    edge.aUuid = a; edge.bUuid = b;
    edge.aAnchor.wholeGeometry = edge.bAnchor.wholeGeometry = true;
    TransactionRequest request;
    {
        auto txn = store.read(); const auto status = store.status(txn);
        request.requestUuid = util::newUuid(); request.storeUuid = status.storeUuid;
        request.generation = status.generation; request.author = "edge-author";
        request.expectedObjects = {{a, store.currentObjectRevision(txn, a)->reference()}, {b, store.currentObjectRevision(txn, b)->reference()}};
    }
    request.operations.push_back(AddEdgeOp{edgeRecordJson(edge)});
    store.submit(request);
    TransactionContext deletion; deletion.author = "deletion-author";
    int64_t cursor = 0;
    std::string snapshot;
    {
        auto txn = store.read(); const auto record = store.edgeByUuid(txn, edge.uuid);
        require(record && record->createdBy == "edge-author" && record->changedBy == "edge-author",
                "checked request author stamps new relationships");
        deletion.expectedObjects = {{a, store.currentObjectRevision(txn, a)->reference()}, {b, store.currentObjectRevision(txn, b)->reference()}};
        snapshot = store.exportJson(txn);
        if (history == HistoryMode::On) {
            const auto events = store.journalSince(txn, 0);
            require(events.back().recordUuid == edge.uuid && events.back().author == "edge-author",
                    "relationship journal uses request author");
            cursor = store.journalPosition(txn);
            const auto status = store.status(txn);
            const auto version = store.relationshipAtVersion(txn, {status.storeUuid, status.generation, edge.uuid, 1});
            require(version && version->record.createdBy == "edge-author", "relationship history preserves request author");
        }
    }
    {
        auto txn = store.write(deletion);
        store.removeEdge(txn, edge.uuid); store.deleteObject(txn, b);
        txn.commit();
    }
    {
        auto txn = store.read(); auto* sql = ObjectStoreImplAccess::conn(txn);
        for (const auto& [table, uuid] : {std::pair{"objects", b}, std::pair{"edges", edge.uuid}}) {
            detail::Statement row(sql, std::string("SELECT state,changed_by FROM ") + table + " WHERE uuid=?", detail::parseUuid(uuid));
            require(row.row() && row.i64(0) == 2 && row.text(1) == "deletion-author",
                    "object and relationship tombstones use transaction author");
        }
        if (history == HistoryMode::On) {
            const auto events = store.journalSince(txn, cursor);
            require(events.size() == 2, "both deletions are journaled");
            for (const auto& event : events) require(event.author == "deletion-author", "deletion journal uses transaction author");
        }
    }
    require(store.author() == "store-owner", "transaction identity does not change the store default");

    auto imported = ObjectStore::create(dir.path / "imported.vcobj", options);
    TransactionContext context; context.author = "import-author";
    context.expectedObjects = {{a, std::nullopt}, {b, std::nullopt}};
    {
        auto txn = imported.write(context); imported.importJson(txn, snapshot); txn.commit();
    }
    {
        auto txn = imported.read();
        require(imported.edgeByUuid(txn, edge.uuid)->createdBy == "edge-author", "snapshot retains original creator");
        if (history == HistoryMode::On) {
            const auto events = imported.journalSince(txn, 0);
            require(events.size() == 3, "snapshot journals two objects and one relationship");
            for (const auto& event : events) require(event.author == "import-author", "snapshot journal uses transaction author");
        }
    }
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
        EdgeRecord relation; relation.kind = "declared_reference"; relation.aUuid = a; relation.bUuid = util::newUuid();
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
    {auto txn=store.writeNumbered(selected);store.retireObject(txn,uuid);txn.commit();}
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

void metadataImportReservations(HistoryMode sourceHistory, HistoryMode targetHistory, bool referenced)
{
    TemporaryDirectory dir;
    OpenOptions options; options.author = "metadata-reservations"; options.history = sourceHistory;
    const auto sourcePath = dir.path / "source.vcobj", targetPath = dir.path / "target.vcobj";
    const auto a = util::newUuid(), b = util::newUuid(), deletedObject = util::newUuid();
    std::string deletedEdge, liveEdge, sourceUuid;
    nlohmann::json snapshot;
    {
        auto source = ObjectStore::create(sourcePath, options);
        {
            auto txn = source.write();
            for (const auto& id : {a, b, deletedObject}) source.upsertObject(txn, fiber(id));
            EdgeRecord edge; edge.kind = "perpendicular"; edge.aUuid = a; edge.bUuid = b;
            deletedEdge = source.addEdge(txn, edge);
            edge.kind = "parallel";
            liveEdge = source.addEdge(txn, edge);
            txn.commit();
        }
        { auto txn = source.read(); snapshot = nlohmann::json::parse(source.exportJson(txn)); sourceUuid = source.uuid(txn); }
        {
            auto txn = source.write(); source.removeEdge(txn, deletedEdge); source.deleteObject(txn, deletedObject);
            if (referenced) source.seal(txn);
            txn.commit();
        }
    }
    options.history = targetHistory;
    if (referenced) { auto collection = ObjectCollection::create(targetPath, {sourcePath}, options); }
    auto target = ObjectStore::open(referenced ? targetPath : sourcePath, options);
    const auto entry = [&](const char* kind, const std::string& id) {
        for (const auto& record : snapshot.at(kind)) if (record.at("uuid") == id) return record;
        throw std::runtime_error("metadata reservation fixture record missing");
    };
    auto live = snapshot;
    live["objects"] = nlohmann::json::array({entry("objects", a), entry("objects", b)});
    live["edges"] = nlohmann::json::array({entry("edges", liveEdge)});
    const auto before = commitSequence(target);
    { auto txn = target.write(); target.importJson(txn, live.dump()); txn.commit(); }
    require(commitSequence(target) == before, "matching live metadata import preserves publication and inherited ownership");
    {
        auto txn = target.read();
        require(target.objectByUuid(txn, a)->reference().storeUuid == sourceUuid &&
                target.edgeByUuid(txn, liveEdge)->reference().storeUuid == sourceUuid,
                "idempotent import does not materialize inherited live records");
    }
    auto conflict = live;
    conflict["objects"][0]["name"] = "conflicting imported metadata";
    auto objectOnly = snapshot;
    objectOnly["objects"] = nlohmann::json::array({entry("objects", deletedObject)});
    objectOnly["edges"] = nlohmann::json::array();
    auto edgeOnly = snapshot;
    edgeOnly["objects"] = nlohmann::json::array();
    edgeOnly["edges"] = nlohmann::json::array({entry("edges", deletedEdge)});
    auto objectAtEdge = objectOnly;
    objectAtEdge["objects"][0]["uuid"] = deletedEdge;
    auto edgeAtObject = edgeOnly;
    edgeAtObject["edges"][0]["uuid"] = deletedObject;
    for (auto rejected : {conflict, objectOnly, edgeOnly, objectAtEdge, edgeAtObject}) {
        const auto provisional = util::newUuid(), provisionalEdge = util::newUuid(), unrelated = util::newUuid();
        auto prefixObject = entry("objects", a); prefixObject["uuid"] = provisional;
        rejected["objects"].insert(rejected["objects"].begin(), std::move(prefixObject));
        auto prefixEdge = entry("edges", liveEdge); prefixEdge["uuid"] = provisionalEdge;
        rejected["edges"].insert(rejected["edges"].begin(), std::move(prefixEdge));
        const auto prior = commitSequence(target);
        {
            auto txn = target.write();
            mustFail([&] { target.importJson(txn, rejected.dump()); });
            target.upsertObject(txn, fiber(unrelated));
            txn.commit();
        }
        require(commitSequence(target) == prior + 1, "caught import rejection permits only unrelated outer work to publish");
        auto txn = target.read();
        require(target.objectByUuid(txn, unrelated).has_value() && !target.objectByUuid(txn, provisional) &&
                !target.edgeByUuid(txn, provisionalEdge), "failed metadata import rolls back all earlier object and edge inserts");
        require(!target.objectByUuid(txn, deletedObject) && !target.edgeByUuid(txn, deletedEdge),
                "metadata import cannot resurrect deleted object or edge identities");
        require(!target.objectByUuid(txn, deletedEdge) && !target.edgeByUuid(txn, deletedObject),
                "deleted identities remain reserved across object and edge namespaces");
        auto* connection = ObjectStoreImplAccess::conn(txn);
        for (const auto& [table, id] : {std::pair{"objects", deletedObject}, std::pair{"edges", deletedEdge}}) {
            auto* owner = std::string_view(table) == "objects" ? detail::objectSource(connection, id, false, true)
                                                              : detail::edgeSource(connection, id, false, true);
            require(owner && detail::storeUuidText(owner) == sourceUuid, "rejected import leaves tombstone with its original owner");
            detail::Statement row(owner, std::string("SELECT state,version FROM ") + table + " WHERE uuid=?", detail::parseUuid(id));
            require(row.row() && row.i64(0) == 2 && row.i64(1) == 2, "rejected import does not alter tombstone state or version");
        }
    }
}


}

int main()
{
    transactionAuthors(HistoryMode::On);
    transactionAuthors(HistoryMode::Off);
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
    for (const auto source : {HistoryMode::On, HistoryMode::Off}) {
        metadataImportReservations(source, source, false);
        for (const auto target : {HistoryMode::On, HistoryMode::Off}) metadataImportReservations(source, target, true);
    }
}
