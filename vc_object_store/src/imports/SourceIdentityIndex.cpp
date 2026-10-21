#include "imports/SourceIdentityIndex.hpp"
#include "geometry/GeometryReader.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include "geometry/PersistentTree.hpp"

namespace vc::objects::detail {
namespace {
using Json = nlohmann::json;
void clearDirty(sqlite3* db, const std::string& object) {
    Statement s(db,"DELETE FROM source_identity_dirty WHERE object_uuid=?");
    s.bindBlob(1,parseUuid(object)); s.stepDone();
}
void putPoint(sqlite3* db, const std::string& object, const std::string& origin, int64_t id, int64_t offset) {
    Statement s(db,"INSERT INTO source_points VALUES(?,?,?,?) ON CONFLICT(object_uuid,point_id) DO UPDATE SET origin=excluded.origin,point_offset=excluded.point_offset");
    s.bindBlob(1,parseUuid(object)); s.bind(2,id); s.bind(3,origin); s.bind(4,offset); s.stepDone();
}
Json idsDescriptor(GeometryReader& reader) {
    auto descriptor=reader.manifest().at("arrays").at("point_ids");
    descriptor.erase("source_revision"); return descriptor;
}
}
void createSourceIdentityIndex(sqlite3* db) {
    exec(db,R"SQL(
CREATE TABLE source_collections(
 object_uuid BLOB PRIMARY KEY REFERENCES objects(uuid) ON DELETE CASCADE,
 origin TEXT NOT NULL, collection_id TEXT NOT NULL, revision_uuid BLOB NOT NULL
);
CREATE INDEX source_collection_identity ON source_collections(origin,collection_id);
CREATE TABLE source_points(
 object_uuid BLOB NOT NULL REFERENCES source_collections(object_uuid) ON DELETE CASCADE,
 point_id INTEGER NOT NULL, origin TEXT NOT NULL, point_offset INTEGER NOT NULL,
 PRIMARY KEY(object_uuid,point_id)
) WITHOUT ROWID;
CREATE INDEX source_point_identity ON source_points(origin,point_id);
CREATE TABLE source_identity_dirty(
 object_uuid BLOB PRIMARY KEY REFERENCES objects(uuid) ON DELETE CASCADE
) WITHOUT ROWID;
CREATE TRIGGER source_identity_object_insert AFTER INSERT ON objects
 WHEN new.kind='point_collection' BEGIN
 INSERT OR IGNORE INTO source_identity_dirty VALUES(new.uuid); END;
CREATE TRIGGER source_identity_object_update AFTER UPDATE OF meta,storage_ref,kind ON objects
 WHEN (new.kind='point_collection' OR old.kind='point_collection') AND
 (new.kind IS NOT old.kind OR
 json_extract(new.meta,'$.source.origin') IS NOT json_extract(old.meta,'$.source.origin') OR
 json_extract(new.meta,'$.source.source_collection_id') IS NOT json_extract(old.meta,'$.source.source_collection_id') OR
 (json_extract(new.meta,'$.source.origin') IS NULL AND new.storage_ref IS NOT old.storage_ref)) BEGIN
 INSERT OR IGNORE INTO source_identity_dirty VALUES(new.uuid); END;
CREATE TRIGGER source_identity_head_insert AFTER INSERT ON geometry_heads
 WHEN (SELECT kind FROM objects WHERE uuid=new.object_uuid)='point_collection' BEGIN
 INSERT OR IGNORE INTO source_identity_dirty VALUES(new.object_uuid); END;
CREATE TRIGGER source_identity_head_update AFTER UPDATE ON geometry_heads
 WHEN (SELECT kind FROM objects WHERE uuid=new.object_uuid)='point_collection' BEGIN
 INSERT OR IGNORE INTO source_identity_dirty VALUES(new.object_uuid); END;
CREATE TRIGGER source_identity_head_delete AFTER DELETE ON geometry_heads
 WHEN EXISTS(SELECT 1 FROM objects WHERE uuid=old.object_uuid AND kind='point_collection') BEGIN
 INSERT OR IGNORE INTO source_identity_dirty VALUES(old.object_uuid); END;
)SQL");
}
void refreshSourceIdentityIndex(sqlite3* db) {
    GeometryPhase phase("sourceIdentityIndex");
    std::vector<std::string> dirty;
    Statement pending(db,"SELECT object_uuid FROM source_identity_dirty ORDER BY object_uuid");
    while(pending.row()) dirty.push_back(formatUuid(pending.blob16(0)));
    for(const auto& object:dirty) {
        Statement record(db,"SELECT o.kind,o.meta,o.storage_ref,h.revision_uuid FROM objects o LEFT JOIN geometry_heads h ON h.object_uuid=o.uuid WHERE o.uuid=?");
        record.bindBlob(1,parseUuid(object));
        if(!record.row()) continue;
        const auto source=Json::parse(record.text(1)).value("source",Json::object());
        if(record.text(0)!="point_collection" || record.isNull(3) || !source.contains("source_collection_id")) {
            Statement erase(db,"DELETE FROM source_collections WHERE object_uuid=?"); erase.bindBlob(1,parseUuid(object)); erase.stepDone();
            clearDirty(db,object); continue;
        }
        const auto origin=source.value("origin",record.text(2)), collection=source.at("source_collection_id").get<std::string>();
        const auto revision=formatUuid(record.blob16(3));
        Statement old(db,"SELECT origin,revision_uuid FROM source_collections WHERE object_uuid=?"); old.bindBlob(1,parseUuid(object));
        bool rebuild=true; std::string previous;
        if(old.row()) {
            previous=formatUuid(old.blob16(1));
            if(previous==revision) rebuild=false;
            else {
                GeometryReader before(db,previous), after(db,revision);
                rebuild=idsDescriptor(before)!=idsDescriptor(after);
            }
            if(old.text(0)!=origin) {
                Statement move(db,"UPDATE source_points SET origin=? WHERE object_uuid=?"); move.bind(1,origin); move.bindBlob(2,parseUuid(object)); move.stepDone();
            }
        }
        Statement put(db,"INSERT INTO source_collections VALUES(?,?,?,?) ON CONFLICT(object_uuid) DO UPDATE SET origin=excluded.origin,collection_id=excluded.collection_id,revision_uuid=excluded.revision_uuid");
        put.bindBlob(1,parseUuid(object)); put.bind(2,origin); put.bind(3,collection); put.bindBlob(4,parseUuid(revision)); put.stepDone();
        if(rebuild) {
            Statement erase(db,"DELETE FROM source_points WHERE object_uuid=?"); erase.bindBlob(1,parseUuid(object)); erase.stepDone();
            GeometryReader reader(db,revision); GeometrySnapshot snapshot;
            snapshot.arrays["point_ids"]=reader.read("point_ids",0);
            const auto ids=snapshot.values<int64_t>("point_ids");
            for(size_t i=0;i<ids.size();++i) {
                if(geometryWork) ++geometryWork->identityRows;
                putPoint(db,object,origin,ids[i],i);
            }
        }
        clearDirty(db,object);
    }
}
void updateSourcePointIdentities(sqlite3* db, const std::string& object,
    const std::string& previous, const std::string& revision, const Json& manifest,
    const std::vector<int64_t>& changed) {
    Statement state(db,"SELECT s.origin FROM source_collections s JOIN objects o ON o.uuid=s.object_uuid WHERE s.object_uuid=? AND s.revision_uuid=? AND s.origin=coalesce(json_extract(o.meta,'$.source.origin'),o.storage_ref) AND s.collection_id=json_extract(o.meta,'$.source.source_collection_id')");
    state.bindBlob(1,parseUuid(object)); state.bindBlob(2,parseUuid(previous));
    if(!state.row()) return; // New/unindexed source: normal publication backfill.
    const auto origin=state.text(0);
    const auto& identities=manifest.at("sequences").at("points").at("identities");
    for(auto id:std::set<int64_t>(changed.begin(),changed.end())) {
        if(geometryWork) ++geometryWork->identityRows;
        if(identityGet(db,identities,id).is_null()) {
            Statement erase(db,"DELETE FROM source_points WHERE object_uuid=? AND point_id=?");
            erase.bindBlob(1,parseUuid(object)); erase.bind(2,id); erase.stepDone();
        } else putPoint(db,object,origin,id,-1); // Shared sequences resolve through their identity tree, never this offset.
    }
    Statement update(db,"UPDATE source_collections SET revision_uuid=? WHERE object_uuid=?");
    update.bindBlob(1,parseUuid(revision)); update.bindBlob(2,parseUuid(object)); update.stepDone();
    clearDirty(db,object);
}
std::optional<SourcePoint> sourcePoint(sqlite3* db, const std::string& origin, int64_t id) {
    Statement s(db,"SELECT p.object_uuid,p.point_offset FROM source_points p JOIN objects o ON o.uuid=p.object_uuid LEFT JOIN object_lifecycle l ON l.object_id=o.id WHERE p.origin=? AND p.point_id=? AND coalesce(l.retired,0)=0 LIMIT 2");
    s.bind(1,origin); s.bind(2,id);
    if(geometryWork) ++geometryWork->identityRows;
    if(!s.row()) return std::nullopt;
    SourcePoint result{formatUuid(s.blob16(0)),s.i64(1)};
    if(s.row()) throw std::runtime_error("duplicate point ID in source document");
    return result;
}
std::optional<std::string> sourceCollection(sqlite3* db, const std::string& origin, const std::string& id) {
    Statement s(db,"SELECT s.object_uuid FROM source_collections s JOIN objects o ON o.uuid=s.object_uuid LEFT JOIN object_lifecycle l ON l.object_id=o.id WHERE s.origin=? AND s.collection_id=? AND coalesce(l.retired,0)=0 LIMIT 2");
    s.bind(1,origin); s.bind(2,id);
    if(geometryWork) ++geometryWork->identityRows;
    if(!s.row()) return std::nullopt;
    const auto result=formatUuid(s.blob16(0));
    if(s.row()) throw std::runtime_error("duplicate source collection ID");
    return result;
}
bool indexedSourceRevision(sqlite3* db, const std::string& object, const std::string& revision) {
    Statement s(db,"SELECT 1 FROM source_collections WHERE object_uuid=? AND revision_uuid=?");
    s.bindBlob(1,parseUuid(object)); s.bindBlob(2,parseUuid(revision)); return s.row();
}
std::optional<int64_t> indexedSourcePointOffset(sqlite3* db, const std::string& object, int64_t id) {
    Statement s(db,"SELECT point_offset FROM source_points WHERE object_uuid=? AND point_id=?");
    s.bindBlob(1,parseUuid(object)); s.bind(2,id);
    if(geometryWork) ++geometryWork->identityRows;
    return s.row()?std::optional(s.i64(0)):std::nullopt;
}
void validateSourceIdentities(sqlite3* db, const std::string& object) {
    Statement s(db,"SELECT origin,collection_id FROM source_collections WHERE object_uuid=?"); s.bindBlob(1,parseUuid(object));
    if(!s.row()) return;
    (void)sourceCollection(db,s.text(0),s.text(1));
    Statement points(db,"SELECT point_id FROM source_points WHERE object_uuid=?"); points.bindBlob(1,parseUuid(object));
    while(points.row()) (void)sourcePoint(db,s.text(0),points.i64(0));
}
}
