#include "imports/SourceIdentityIndex.hpp"
#include "vc_object_store/objects/GeometryIngest.hpp"
#include "geometry/GeometryReader.hpp"
#include "geometry/GeometrySnapshot.hpp"
#include "geometry/SequenceMetadata.hpp"
#include "geometry/GeometryToken.hpp"
#include "internal/ImmutableImport.hpp"
#include "internal/JsonFields.hpp"

namespace vc::objects::detail {
namespace {
using Json = nlohmann::json;
bool hasSourceIdentityIndex(sqlite3* db) {
    Statement present(db,"SELECT 1 FROM sqlite_schema WHERE type='table' AND name='source_collections'");
    return present.row();
}
void clearDirty(sqlite3* db, const std::string& object) {
    execute(db,"DELETE FROM source_identity_dirty WHERE object_uuid=?", parseUuid(object));
}
void putPoint(sqlite3* db, const std::string& object, const std::string& origin, int64_t id, int64_t offset) {
    execute(db,"INSERT INTO source_points VALUES(?,?,?,?) ON CONFLICT(object_uuid,point_id) DO UPDATE SET origin=excluded.origin,point_offset=excluded.point_offset",
        parseUuid(object), id, origin, offset);
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
)SQL");
    if(immutableInitial(db))return;
    exec(db,R"SQL(
CREATE TRIGGER source_identity_object_insert AFTER INSERT ON objects
 WHEN new.kind='point_collection' BEGIN
 INSERT OR IGNORE INTO source_identity_dirty VALUES(new.uuid); END;
CREATE TRIGGER source_identity_object_update AFTER UPDATE OF meta,storage_ref,kind,geometry_manifest,state ON objects
 WHEN (new.kind='point_collection' OR old.kind='point_collection') AND
 (new.state IS NOT old.state OR new.kind IS NOT old.kind OR new.geometry_manifest IS NOT old.geometry_manifest OR
 json_extract(nullif(new.meta,''),'$.source.origin') IS NOT json_extract(nullif(old.meta,''),'$.source.origin') OR
 json_extract(nullif(new.meta,''),'$.source.source_collection_id') IS NOT json_extract(nullif(old.meta,''),'$.source.source_collection_id') OR
 (json_extract(nullif(new.meta,''),'$.source.origin') IS NULL AND new.storage_ref IS NOT old.storage_ref)) BEGIN
 INSERT OR IGNORE INTO source_identity_dirty VALUES(new.uuid); END;
)SQL");
}
void refreshSourceIdentityIndex(sqlite3* db) {
    GeometryPhase phase("sourceIdentityIndex");
    if(!hasSourceIdentityIndex(db)) {
        // Fiber-only/native object stores need no source-point accelerator.
        // Install it only once there is actual source identity to resolve.
        Statement source(db,"SELECT 1 FROM objects WHERE state<>2 AND kind='point_collection' "
            "AND json_type(nullif(meta,''),'$.source.source_collection_id') IS NOT NULL LIMIT 1");
        if(!source.row())return;
        createSourceIdentityIndex(db);
        exec(db,"INSERT INTO source_identity_dirty SELECT uuid FROM objects WHERE state<>2 AND kind='point_collection';");
        // Publication's initial schema check precedes this lazy creation.
        registerStorageTables(db);
    }
    std::vector<std::string> dirty;
    Statement pending(db,"SELECT object_uuid FROM source_identity_dirty ORDER BY object_uuid");
    while(pending.row()) dirty.push_back(formatUuid(pending.blob16(0)));
    for(const auto& object:dirty) {
        Statement record(db,"SELECT o.kind,o.meta,o.storage_ref,CASE WHEN o.geometry_manifest IS NOT NULL THEN o.storage_ref END FROM objects o WHERE o.uuid=? AND o.state<>2",
            parseUuid(object));
        if(!record.row()) {
            execute(db,"DELETE FROM source_collections WHERE object_uuid=?",parseUuid(object));
            clearDirty(db,object);continue;
        }
        const auto source=parseObjectJson(record.text(1)).value("source",Json::object());
        if(record.text(0)!="point_collection" || record.isNull(3) || !source.contains("source_collection_id")) {
            execute(db,"DELETE FROM source_collections WHERE object_uuid=?", parseUuid(object));
            clearDirty(db,object); continue;
        }
        const auto origin=source.value("origin",record.text(2)), collection=source.at("source_collection_id").get<std::string>();
        const auto revision=readGeometryTokenColumn(record,3);
        Statement old(db,"SELECT origin,revision_uuid FROM source_collections WHERE object_uuid=?", parseUuid(object));
        bool rebuild=true; std::string previous;
        if(old.row()) {
            previous=readGeometryTokenColumn(old,1);
            // Current tokens are object+publication-generation: multiple
            // edits inside one transaction intentionally share a token.
            if(previous==revision && historyEnabled(db)) rebuild=false;
            else if (historyEnabled(db)) {
                GeometryReader before(db,previous), after(db,revision);
                rebuild=idsDescriptor(before)!=idsDescriptor(after);
            }
            if(old.text(0)!=origin) {
                execute(db,"UPDATE source_points SET origin=? WHERE object_uuid=?", origin, parseUuid(object));
            }
        }
        {Statement put(db,"INSERT INTO source_collections VALUES(?,?,?,?) ON CONFLICT(object_uuid) DO UPDATE SET origin=excluded.origin,collection_id=excluded.collection_id,revision_uuid=excluded.revision_uuid");
            put.bind(1,parseUuid(object));put.bind(2,origin);put.bind(3,collection);bindGeometryToken(put,4,revision);put.stepDone();}
        if(rebuild) {
            execute(db,"DELETE FROM source_points WHERE object_uuid=?", parseUuid(object));
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
    if(!hasSourceIdentityIndex(db))return;
    Statement state(db,"SELECT s.origin FROM source_collections s JOIN objects o ON o.uuid=s.object_uuid WHERE o.state<>2 AND s.object_uuid=? AND s.revision_uuid=? AND s.origin=coalesce(json_extract(nullif(o.meta,''),'$.source.origin'),o.storage_ref) AND s.collection_id=json_extract(nullif(o.meta,''),'$.source.source_collection_id')");
    state.bind(1,parseUuid(object));bindGeometryToken(state,2,previous);
    if(!state.row()) return; // The source index is created at publication.
    const auto origin=state.text(0);
    const auto& identities=manifest.at("sequences").at("points").at("identities");
    for(auto id:std::set<int64_t>(changed.begin(),changed.end())) {
        if(geometryWork) ++geometryWork->identityRows;
        if(identityGet(db,identities,id).is_null()) {
            execute(db,"DELETE FROM source_points WHERE object_uuid=? AND point_id=?", parseUuid(object), id);
        } else putPoint(db,object,origin,id,-1); // Managed sequences resolve through their metadata snapshot, never this offset.
    }
    {Statement update(db,"UPDATE source_collections SET revision_uuid=? WHERE object_uuid=?");
        bindGeometryToken(update,1,revision);update.bind(2,parseUuid(object));update.stepDone();}
    clearDirty(db,object);
}
std::optional<SourcePoint> sourcePoint(sqlite3* db, const std::string& origin, int64_t id) {
    if(!hasSourceIdentityIndex(db))return std::nullopt;
    Statement s(db,"SELECT p.object_uuid,p.point_offset FROM source_points p JOIN objects o ON o.uuid=p.object_uuid WHERE o.state=0 AND p.origin=? AND p.point_id=? LIMIT 2",
        origin, id);
    if(geometryWork) ++geometryWork->identityRows;
    if(!s.row()) return std::nullopt;
    SourcePoint result{formatUuid(s.blob16(0)),s.i64(1)};
    if(s.row()) throw GeometrySourceRejected("duplicate point ID in source document");
    return result;
}
std::optional<std::string> sourceCollection(sqlite3* db, const std::string& origin, const std::string& id) {
    if(!hasSourceIdentityIndex(db))return std::nullopt;
    Statement s(db,"SELECT s.object_uuid FROM source_collections s JOIN objects o ON o.uuid=s.object_uuid WHERE o.state=0 AND s.origin=? AND s.collection_id=? LIMIT 2",
        origin, id);
    if(geometryWork) ++geometryWork->identityRows;
    if(!s.row()) return std::nullopt;
    const auto result=formatUuid(s.blob16(0));
    if(s.row()) throw GeometrySourceRejected("duplicate source collection ID");
    return result;
}
void validateSourceIdentities(sqlite3* db, const std::string& object) {
    if(!hasSourceIdentityIndex(db))return;
    Statement s(db,"SELECT origin,collection_id FROM source_collections WHERE object_uuid=?", parseUuid(object));
    if(!s.row()) return;
    (void)sourceCollection(db,s.text(0),s.text(1));
    Statement points(db,"SELECT point_id FROM source_points WHERE object_uuid=?", parseUuid(object));
    while(points.row()) (void)sourcePoint(db,s.text(0),points.i64(0));
}
void markSourceIdentityDirty(sqlite3* db,const std::string& object) {
    // If not yet provisioned, the first refresh discovers all source objects.
    if(hasSourceIdentityIndex(db))
        execute(db,"INSERT OR IGNORE INTO source_identity_dirty VALUES(?)",parseUuid(object));
}
}
