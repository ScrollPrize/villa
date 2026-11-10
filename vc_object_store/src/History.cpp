#include "internal/History.hpp"
#include "internal/Artifacts.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include "internal/Versions.hpp"
#include "internal/RecordJson.hpp"
#include "internal/RecordSql.hpp"
#include "internal/EdgeSql.hpp"
#include "geometry/Dependencies.hpp"
#include "internal/ImmutableImport.hpp"
#include "internal/RecordResolver.hpp"
#include "internal/RecordOverrides.hpp"
#include "internal/ReferenceResolver.hpp"
#include <limits>

namespace vc::objects {
using namespace detail;
nlohmann::json detail::dependencyFields(const nlohmann::json* meta, bool retired,
                                const std::string& epoch, const Anchor& anchor)
{
    auto selected=nlohmann::json::object();
    for(const auto& key:anchor.metadataKeys)
        if(meta->contains(key)) selected[key]=meta->at(key);
    return nlohmann::json{{"metadata",selected},{"retired",retired},{"lifecycle_epoch",epoch}};
}
nlohmann::json detail::dependencyMetadata(const nlohmann::json& record, const Anchor& anchor)
{
    return dependencyFields(anchor.metadataKeys.empty() ? nullptr : &record.at("meta"),
                            record.value("retired",false),record.value("lifecycle_epoch",""),anchor);
}
nlohmann::json detail::currentDependencyMetadata(sqlite3* db,const std::string& object,const Anchor& anchor)
{
    auto* source=objectSource(db,object);
    if(!source)throw std::runtime_error("relationship endpoint missing or deleted");
    Statement row(source,anchor.metadataKeys.empty()
        ? "SELECT (state<>0),lifecycle_epoch,NULL FROM objects WHERE uuid=?"
        : "SELECT (state<>0),lifecycle_epoch,meta FROM objects WHERE uuid=?",parseUuid(object));
    if(!row.row())throw std::runtime_error("relationship endpoint disappeared");
    const auto meta=anchor.metadataKeys.empty()?nlohmann::json{}:nlohmann::json::parse(row.text(2));
    return dependencyFields(&meta,row.i64(0)!=0,row.text(1),anchor);
}
namespace {
using Json = nlohmann::json;

// Archived revision UUIDs and ordinary four-field ObjectVersion references.
// Both forms are binary in SQLite; string transport is internal only.
using SqlReference=std::variant<std::array<unsigned char,16>,std::array<uint8_t,56>,std::string>;
SqlReference sqlReference(const std::string& value)
{
    if(const auto ref=parseVersionReference(value))return versionReferenceBytes(*ref);
    return parseUuid(value);
}
std::optional<SqlReference> sqlReference(const std::optional<std::string>& value)
{return value?std::optional(sqlReference(*value)):std::nullopt;}
std::string referenceColumn(const Statement& row,int column)
{
    if(sqlite3_column_type(row.value,column)!=SQLITE_BLOB)throw std::runtime_error("history: binary reference required");
    if(sqlite3_column_bytes(row.value,column)==16)return row.uuid(column);
    return versionReferenceText(versionReferenceFromBytes(std::span(
        static_cast<const uint8_t*>(sqlite3_column_blob(row.value,column)),size_t(sqlite3_column_bytes(row.value,column)))));
}
std::optional<std::string> optionalReference(const Statement& row,int column)
{return row.isNull(column)?std::nullopt:std::optional(referenceColumn(row,column));}
bool borrowedObject(sqlite3* db,const std::string& uuid)
{
    const auto source=objectSource(db,uuid);
    return source&&source!=db;
}
std::optional<std::string> head(sqlite3* db,const std::string& object)
{
    // Publication must still see the preceding head after deleting its current
    // record. Visible-object resolution intentionally hides tombstones.
    if(historyEnabled(db)) {
        Statement local(db,"SELECT revision_uuid FROM object_heads WHERE object_uuid=?",parseUuid(object));
        if(local.row())return local.uuid(0);
    }
    const auto source=objectSource(db,object);
    if(!source)return {};
    if(historyEnabled(source)){
        Statement row(source,"SELECT revision_uuid FROM object_heads WHERE object_uuid=?",parseUuid(object));
        if(row.row())return row.uuid(0);
        return {};
    }
    const auto resolved=resolveObject(db,object);
    return resolved?std::optional(versionReferenceText(objectVersion(*resolved))):std::nullopt;
}

std::optional<std::string> publicationBase(sqlite3* db,const std::string& object)
{
    Statement local(db,"SELECT revision_uuid FROM object_heads WHERE object_uuid=?",parseUuid(object));
    if(local.row())return local.uuid(0);
    auto* inherited=objectSource(db,object,true);
    return inherited?head(inherited,object):std::nullopt;
}

std::optional<std::string> optionalUuid(Statement& s, int column)
{
    return s.isNull(column) ? std::nullopt : std::optional(formatUuid(s.blob16(column)));
}

Json metadata(sqlite3* db, const std::string& revision, bool legacyTextUuid=false)
{
    if(const auto ref=parseVersionReference(revision)) {
        const auto value=resolveObjectVersion(db,*ref);
        if(!value)throw std::runtime_error("history: unavailable object version");
        return objectToJson(value->revision.record);
    }
    if(!legacyTextUuid){
        db=resolveReference(db,[&](sqlite3* source){
            if(!historyEnabled(source))return false;
            Statement row(source,"SELECT 1 FROM object_revisions WHERE uuid=?",parseUuid(revision));return row.row();
        });
        if(!db)throw std::runtime_error("history: unavailable object revision");
    }
    // Common case: the materialized head, without any history replay.
    const SqlReference key=legacyTextUuid?SqlReference{revision}:SqlReference{parseUuid(revision)};
    Statement current(db, "SELECT record FROM object_head_records WHERE revision_uuid=?", key);
    if (current.row()) return Json::parse(current.text(0));
    Statement chain(db,
        "WITH RECURSIVE chain(uuid,parent_uuid,seq,metadata_delta) AS ("
        "SELECT uuid,parent_uuid,seq,metadata_delta FROM object_revisions WHERE uuid=? UNION ALL "
        "SELECT r.uuid,r.parent_uuid,r.seq,r.metadata_delta FROM object_revisions r "
        "JOIN chain c ON r.uuid=c.parent_uuid) SELECT metadata_delta FROM chain ORDER BY seq", key);
    auto result = Json::object();
    while (chain.row()) {
        auto value = Json::parse(chain.text(0));
        result = value.is_object() ? std::move(value) : result.patch(value);
    }
    return result;
}

std::map<std::string, std::string> members(sqlite3* db, const std::string& revision)
{
    std::map<std::string, std::string> result;
    if(const auto ref=parseVersionReference(revision)) {
        const auto selected=resolveObjectVersion(db,*ref);
        if(!selected)throw std::runtime_error("history: unavailable selected version");
        if(!selected->revision.uuid.empty()&&!parseVersionReference(selected->revision.uuid))
            result=members(selected->source,selected->revision.uuid);
        if(selected->source!=db&&historyEnabled(db)){
            Statement rows(db,"SELECT edge_uuid,relationship_uuid FROM object_relationship_heads WHERE object_uuid=?",parseUuid(ref->objectUuid));
            while(rows.row())result.insert_or_assign(rows.uuid(0),rows.uuid(1));
        }
        return result;
    }
    const auto resolved=resolveObjectRevision(db,revision);
    if(!resolved)throw std::runtime_error("history: unavailable membership revision");
    if(resolved->source!=db)return members(db,versionReferenceText({storeUuidText(resolved->source),
        resolved->revision.generation,resolved->revision.objectUuid,resolved->revision.version}));
    Statement isHead(db, "SELECT object_uuid FROM object_heads WHERE revision_uuid=?", parseUuid(revision));
    if (isHead.row()) {
        Statement s(db, "SELECT edge_uuid,relationship_uuid FROM object_relationship_heads WHERE object_uuid=?",
            isHead.blob16(0));
        while (s.row()) result.emplace(formatUuid(s.blob16(0)), formatUuid(s.blob16(1)));
        return result;
    }
    Statement s(db,
        "WITH RECURSIVE chain(uuid,parent_uuid,seq) AS ("
        "SELECT uuid,parent_uuid,seq FROM object_revisions WHERE uuid=? UNION ALL "
        "SELECT r.uuid,r.parent_uuid,r.seq FROM object_revisions r JOIN chain c ON r.uuid=c.parent_uuid) "
        "SELECT d.edge_uuid,d.relationship_uuid FROM chain c JOIN object_relationship_deltas d "
        "ON d.revision_uuid=c.uuid ORDER BY c.seq", parseUuid(revision));
    while (s.row()) {
        if (s.isNull(1)) result.erase(formatUuid(s.blob16(0)));
        else result[formatUuid(s.blob16(0))] = formatUuid(s.blob16(1));
    }
    return result;
}

RelationshipRevision relationship(sqlite3* db, const std::string& uuid)
{
    db=resolveReference(db,[&](sqlite3* source){
        if(!historyEnabled(source))return false;
        Statement row(source,"SELECT 1 FROM relationship_revisions WHERE uuid=?",parseUuid(uuid));return row.row();
    });
    if(!db)throw std::runtime_error("object history: unavailable relationship revision");
    Statement s(db, "SELECT record,a_revision,b_revision,a_geometry,b_geometry FROM relationship_revisions WHERE uuid=?",
        parseUuid(uuid));
    if (!s.row()) throw std::runtime_error("object history: missing relationship revision");
    RelationshipRevision result;
    result.uuid = uuid;
    const auto record = Json::parse(s.text(0));
    result.record = edgeFromJson(record);
    result.aRevision = referenceColumn(s,1); result.bRevision = referenceColumn(s,2);
    result.aGeometryRevision = optionalReference(s,3); result.bGeometryRevision = optionalReference(s,4);
    Statement version(db, "SELECT version,generation FROM relationship_versions WHERE revision_uuid=?", parseUuid(uuid));
    if (version.row()) { result.version=version.i64(0); result.generation=version.uuid(1); }
    const auto endpointVersion = [&](const std::string& reference) {
        if(const auto ref=parseVersionReference(reference))return ref->version;
        const auto value=resolveObjectRevision(db,reference);
        return value?value->revision.version:0;
    };
    result.aVersion=endpointVersion(result.aRevision);result.bVersion=endpointVersion(result.bRevision);
    return result;
}

// One dependency definition for selected graphs and current indexed queries.
bool endpointApplies(const ObjectStore& store, const Txn& txn, const Anchor& anchor,
                     const std::optional<std::string>& geometry, const std::string& pinned,
                     const ObjectRevision& selected)
{
    if (selected.deleted || selected.record.retired) return false;
    const auto original = ObjectStoreImplAccess::objectRevision(store, txn, pinned);
    if (!original || (!anchor.metadataKeys.empty() && !original->metadataComplete)) return false;
    return dependencyMetadata(objectToJson(original->record), anchor) ==
               dependencyMetadata(objectToJson(selected.record), anchor) &&
           !endpointChanged(store, txn, anchor, geometry, selected.geometryRevision);
}

Json currentRecord(sqlite3* db, const std::string& object)
{
    Statement row(db, std::string("SELECT ") + kObjectCols + " FROM objects WHERE uuid=? AND state<>2", parseUuid(object));
    if (!row.row()) throw std::runtime_error("object history: endpoint missing");
    return objectToJson(readObjectRow(db, row));
}
}

nlohmann::json detail::historicalDependencyMetadata(sqlite3* db,const std::string& revision,const Anchor& anchor,bool legacyTextUuid)
{
    return dependencyMetadata(metadata(db,revision,legacyTextUuid),anchor);
}

std::optional<std::string> detail::currentObjectReference(sqlite3* db,const std::string& object)
{return head(db,object);}

void detail::createHistorySchema(sqlite3* db)
{
    exec(db, R"SQL(
-- metadata_delta: the whole record (a JSON object) for a revision without a
-- parent, else an RFC 6902 patch from the parent's record (an array).
CREATE TABLE object_revisions(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  uuid BLOB NOT NULL UNIQUE,
  object_uuid BLOB NOT NULL,
  parent_uuid BLOB REFERENCES object_revisions(uuid),
  geometry_uuid BLOB,
  metadata_delta TEXT NOT NULL CHECK(json_valid(metadata_delta) AND json_type(metadata_delta)=iif(parent_uuid IS NULL,'object','array')),
  metadata_complete INTEGER NOT NULL DEFAULT 1,
  published_at_us INTEGER,
  deleted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX object_revision_history ON object_revisions(object_uuid,seq);
CREATE INDEX object_revision_time ON object_revisions(object_uuid,published_at_us DESC,seq DESC);
CREATE INDEX object_publication_time ON object_revisions(published_at_us);
CREATE TABLE object_heads(object_uuid BLOB PRIMARY KEY, revision_uuid BLOB NOT NULL UNIQUE REFERENCES object_revisions(uuid));
CREATE TABLE object_head_records(object_uuid BLOB PRIMARY KEY, revision_uuid BLOB NOT NULL UNIQUE, record TEXT NOT NULL);
CREATE TABLE relationship_revisions(
  uuid BLOB PRIMARY KEY,
  edge_uuid BLOB NOT NULL,
  record TEXT NOT NULL CHECK(json_valid(record)),
  a_revision BLOB NOT NULL,
  b_revision BLOB NOT NULL,
  a_geometry BLOB, b_geometry BLOB
);
CREATE INDEX relationship_history ON relationship_revisions(edge_uuid);
CREATE TABLE object_relationship_deltas(
  revision_uuid BLOB NOT NULL REFERENCES object_revisions(uuid),
  edge_uuid BLOB NOT NULL,
  relationship_uuid BLOB REFERENCES relationship_revisions(uuid),
  PRIMARY KEY(revision_uuid,edge_uuid)
) WITHOUT ROWID;
CREATE TABLE object_relationship_heads(
  object_uuid BLOB NOT NULL, edge_uuid BLOB NOT NULL,
  relationship_uuid BLOB NOT NULL REFERENCES relationship_revisions(uuid),
  PRIMARY KEY(object_uuid,edge_uuid)
) WITHOUT ROWID;
CREATE INDEX object_relationship_edges ON object_relationship_heads(edge_uuid);
CREATE TABLE history_dirty_objects(uuid BLOB PRIMARY KEY);
CREATE TABLE history_dirty_edges(uuid BLOB PRIMARY KEY);
CREATE TABLE history_expected(uuid BLOB PRIMARY KEY, revision_uuid BLOB);
CREATE TABLE history_checked(singleton INTEGER PRIMARY KEY CHECK(singleton=1));
CREATE TABLE history_required(uuid BLOB PRIMARY KEY);
CREATE TABLE history_validated_edges(uuid BLOB PRIMARY KEY, a_meta TEXT NOT NULL, b_meta TEXT NOT NULL, recorded_only INTEGER NOT NULL);
)SQL");
    for (const auto* table : {"object_revisions", "relationship_revisions", "object_relationship_deltas"}) {
        for (const auto* operation : {"UPDATE", "DELETE"}) {
            const std::string name = std::string(table) + "_" + operation;
            exec(db, ("CREATE TRIGGER " + name + " BEFORE " + operation + " ON " + table +
                " BEGIN SELECT RAISE(ABORT,'object history is immutable and retained'); END;").c_str());
        }
    }
    // Dirty sets live inside the action's SQLite transaction/savepoints. Failed
    // operations cannot leave false dirty entries or publish partial revisions.
    for (const auto* operation : {"INSERT", "UPDATE", "DELETE"}) {
        const std::string row = std::string(operation) == "DELETE" ? "old" : "new";
        const std::string suffix = std::string("_history_") + operation;
        auto trigger = [&](const std::string& table, const std::string& body) {
            exec(db, ("CREATE TRIGGER " + table + suffix + " AFTER " + operation + " ON " + table +
                " BEGIN " + body + " END;").c_str());
        };
        // Explicit UPSERT is essential here: an outer UPSERT can override a
        // trigger's INSERT OR IGNORE conflict policy.
        auto mark = [](const std::string& table, const std::string& values) {
            return "INSERT INTO " + table + " " + values + " ON CONFLICT(uuid) DO NOTHING;";
        };
        trigger("objects", mark("history_dirty_objects", "VALUES(" + row + ".uuid)"));
        for (const auto* table : {"fibers", "surfaces", "tracks", "object_tags"})
            trigger(table, mark("history_dirty_objects", "SELECT uuid FROM objects WHERE id=" + row + ".object_id"));
        trigger("edges",
            mark("history_dirty_edges", "VALUES(" + row + ".uuid)") +
            mark("history_dirty_objects", "SELECT uuid FROM objects WHERE uuid IN (" + row + ".a_uuid," + row + ".b_uuid)"));
        for (const auto* table : {"edge_tags"})
            trigger(table,
                mark("history_dirty_edges", "SELECT uuid FROM edges WHERE id=" + row + ".edge_id") +
                mark("history_dirty_objects", "SELECT o.uuid FROM objects o JOIN edges e "
                "ON o.uuid=e.a_uuid OR o.uuid=e.b_uuid WHERE e.id=" + row + ".edge_id"));
    }
}

void detail::expectHistory(sqlite3* db, const RevisionSelection& expected)
{
    requireHistory(db);
    activeStatementPools.at(db)->checkedAction = true;
    exec(db, "INSERT INTO history_checked VALUES(1)");
    for (const auto& [object, revision] : expected) {
        const auto uuid = formatUuid(parseUuid(object));
        const auto wanted = revision ? std::optional(parseVersionReference(*revision)?versionReferenceText(*parseVersionReference(*revision)):formatUuid(parseUuid(*revision))) : std::nullopt;
        if (head(db, uuid) != wanted)
            throw std::runtime_error("object history: expected revision conflict for " + uuid);
        execute(db, "INSERT INTO history_expected VALUES(?,?)", parseUuid(uuid), sqlReference(wanted));
    }
}

void detail::historyConflict(sqlite3* db, const std::string& message)
{
    if (const auto pool = activeStatementPools.find(db); pool != activeStatementPools.end() && pool->second->checkedAction)
        pool->second->actionConflict = true;
    throw std::runtime_error(message);
}

std::optional<std::string> ObjectStore::geometryAtVersion(const Txn& txn, const std::string& objectUuid,
                                                         int64_t expectedVersion) const
{
    const auto uuid = formatUuid(parseUuid(objectUuid));
    if (const auto version = currentVersion(txn, uuid); version != expectedVersion)
        detail::historyConflict(static_cast<sqlite3*>(connection(txn)),
            "geometry: object " + uuid + " is at version " + (version ? std::to_string(*version) : std::string("none")) +
            ", not " + std::to_string(expectedVersion) + "; reload before committing");
    const auto geometry = currentGeometry(txn, uuid);
    return geometry ? std::optional(geometry->uuid) : std::nullopt;
}

void detail::recordRelationshipValidation(sqlite3* db, const EdgeRecord& edge, bool recordedOnly)
{
    GeometryPhase phase("detail::recordRelationshipValidation");
    const auto history=historyEnabled(db);
    if (!history && recordedOnly) throw std::runtime_error("object store: keeping a historical relationship requires history");
    if(!lifecycleBindingMatches(db,edge)) throw std::runtime_error("relationship endpoints must be active");
    if(!history && immutableInitial(db)) return;
    if(!history) {
        const auto pool=activeStatementPools.find(db);
        if(pool==activeStatementPools.end())return; // Offline history removal has no endpoint edits.
        if (pool->second->checkedAction) {
            for (const auto& endpoint : {edge.aUuid, edge.bUuid}) {
                if (borrowedObject(db,endpoint)) continue;
                if (!pool->second->checkedObjects.contains(endpoint)) {
                    pool->second->actionConflict = true;
                    throw std::runtime_error("object store: missing expected validation endpoint " + endpoint);
                }
            }
        }
        const auto& state=pool->second->currentWrite;
        if(!state||(!state->before.contains(edge.aUuid)&&!state->before.contains(edge.bUuid)))return;
    }
    const auto a=currentDependencyMetadata(db,edge.aUuid,edge.aAnchor), b=currentDependencyMetadata(db,edge.bUuid,edge.bAnchor);
    if (!history) {
        auto& state=activeStatementPools.at(db)->currentWrite;
        if(state.use_count()!=1)state=std::make_shared<CurrentWriteState>(*state);
        state->validated.insert_or_assign(edge.uuid,std::pair{a,b});
        return;
    }
    execute(db, "INSERT INTO history_validated_edges VALUES(?,?,?,?) ON CONFLICT(uuid) "
        "DO UPDATE SET a_meta=excluded.a_meta,b_meta=excluded.b_meta,recorded_only=excluded.recorded_only",
        parseUuid(edge.uuid), a.dump(), b.dump(), int64_t(recordedOnly));
    for (const auto& object : {edge.aUuid, edge.bUuid}) {
        if(borrowedObject(db,object))continue;
        execute(db, "INSERT OR IGNORE INTO history_required VALUES(?)", parseUuid(object));
        if (!recordedOnly) {
            execute(db, "INSERT OR IGNORE INTO history_dirty_objects VALUES(?)", parseUuid(object));
        }
    }
    if (!recordedOnly) {
        bindLifecycle(db, edge, a.value("lifecycle_epoch", ""), b.value("lifecycle_epoch", ""));
        execute(db, "INSERT OR IGNORE INTO history_dirty_edges VALUES(?)", parseUuid(edge.uuid));
    }
}

namespace {
CurrentWriteState& mutableCurrentWrite(sqlite3* db)
{
    auto& state=activeStatementPools.at(db)->currentWrite;
    if(!state)state=std::make_shared<CurrentWriteState>();
    else if(state.use_count()!=1)state=std::make_shared<CurrentWriteState>(*state);
    return *state;
}
}

void detail::beginCurrentWrite(sqlite3* db)
{
    if(const auto pool=activeStatementPools.find(db);pool!=activeStatementPools.end())
        pool->second->currentWrite.reset();
}

void detail::beforeCurrentObjectEdit(sqlite3* db,const std::string& input)
{
    if(historyEnabled(db)||immutableInitial(db))return;
    const auto uuid=formatUuid(parseUuid(input));
    const auto& state=activeStatementPools.at(db)->currentWrite;
    if(state&&state->before.contains(uuid))return;
    Statement object(db,"SELECT meta,(state<>0),lifecycle_epoch FROM objects WHERE uuid=?",parseUuid(uuid));
    if(!object.row()) {mutableCurrentWrite(db).before.emplace(uuid,Json::object());return;}
    mutableCurrentWrite(db).before.emplace(uuid,Json{{"meta",Json::parse(object.text(0))},
        {"retired",bool(object.i64(1))},{"lifecycle_epoch",object.text(2)}});
}

void detail::validateCurrentRelationships(sqlite3* db)
{
    GeometryPhase phase("detail::validateCurrentRelationships");
    const auto pool=activeStatementPools.find(db);
    // Offline schema conversion has no editable transaction or changed endpoints.
    if(pool==activeStatementPools.end())return;
    auto& state=pool->second->currentWrite;
    if(!state)return;
    std::set<std::string> seen;
    for(const auto& [uuid,before]:state->before) {
        const auto current=resolveObject(db,uuid);
        if(current) {
            const auto meta=Json::parse(current->record.meta);
            if(meta.is_object()&&meta.contains(kArtifactsMetaKey)) {
                const auto old=before.value("meta",Json::object());
                if(!old.contains(kArtifactsMetaKey)||old.at(kArtifactsMetaKey)!=meta.at(kArtifactsMetaKey))
                    checkArtifactReferences(db,uuid,meta.at(kArtifactsMetaKey));
            }
        }
        for(const auto& edge:incidentEdges(db,uuid)) {
            if(!seen.insert(edge.uuid).second)continue;
            const auto validated=state->validated.find(edge.uuid);
            const auto expected=[&](const std::string& object,const Anchor& anchor,bool second) {
                if(validated!=state->validated.end())return second?validated->second.second:validated->second.first;
                const auto old=state->before.find(object);
                return old==state->before.end()?currentDependencyMetadata(db,object,anchor):dependencyMetadata(old->second,anchor);
            };
            if(expected(edge.aUuid,edge.aAnchor,false)!=currentDependencyMetadata(db,edge.aUuid,edge.aAnchor)||
               expected(edge.bUuid,edge.bAnchor,true)!=currentDependencyMetadata(db,edge.bUuid,edge.bAnchor))
                throw std::runtime_error("object store: metadata/lifecycle edit requires explicit relationship revalidation or removal: "+edge.uuid);
        }
    }
}

void detail::publishHistory(sqlite3* db, bool baseline)
{
    GeometryPhase phase("detail::publishHistory");
    if (const auto pool = activeStatementPools.find(db); pool != activeStatementPools.end() && pool->second->actionConflict)
        throw std::runtime_error("object history: action had a revision conflict; all changes rolled back");
    struct Pending {
        std::string uuid;
        std::optional<std::string> parent, geometry;
        Json record, delta;
        bool exists = false, changed = false;
    };
    std::map<std::string, Pending> objects;
    Statement dirty(db, "SELECT uuid FROM history_dirty_objects");
    Statement checked(db, "SELECT 1 FROM history_checked");
    const bool strict = checked.row();
    if (geometryWork) {
        Statement bytes(db,"SELECT coalesce(sum(length(meta)),0) FROM objects WHERE uuid IN (SELECT uuid FROM history_dirty_objects)");
        bytes.row(); geometryWork->metadataBytes+=bytes.i64(0);
    }
    if (strict) {
        Statement required(db, "SELECT uuid FROM history_required");
        while (required.row()) {
            Statement expected(db, "SELECT revision_uuid FROM history_expected WHERE uuid=?", required.blob16(0));
            if (!expected.row() || optionalReference(expected,0) != publicationBase(db, formatUuid(required.blob16(0))))
                throw std::runtime_error("object history: missing expected validation endpoint " + formatUuid(required.blob16(0)));
        }
    }
    while (dirty.row()) {
        const auto object = formatUuid(dirty.blob16(0));
        if(borrowedObject(db,object))continue;
        Pending p;
        p.parent = head(db, object);
        if (strict) {
            Statement expected(db, "SELECT revision_uuid FROM history_expected WHERE uuid=?", parseUuid(object));
            if (!expected.row() || optionalReference(expected,0) != publicationBase(db,object))
                throw std::runtime_error("object history: missing or conflicting expected revision for " + object);
        }
        Statement row(db, std::string("SELECT ") + kObjectCols + " FROM objects WHERE uuid=? AND state<>2", parseUuid(object));
        p.exists = row.row();
        if (p.exists) {
            p.record = objectToJson(readObjectRow(db, row));
            p.geometry = geometryHead(db, object);
            p.delta = Json::diff(p.parent ? metadata(db, *p.parent) : Json::object(), p.record);
            p.changed = !p.parent || !p.delta.empty();
            if (p.parent) {
                Statement previous(db, "SELECT geometry_uuid FROM object_revisions WHERE uuid=?", parseUuid(*p.parent)); previous.row();
                p.changed |= optionalReference(previous, 0) != p.geometry;
            }
        }
        objects.emplace(object, std::move(p));
    }
    // A changed artifact manifest may only name bytes the store holds, whichever
    // API wrote it.
    for (const auto& [object, p] : objects) {
        // Policy maintenance preserves the existing artifact references; its
        // temporary sibling filename is not the published artifact location.
        if (baseline) break;
        if (!p.exists || !p.changed) continue;
        const auto& meta = p.record.at("meta");
        if (!meta.is_object() || !meta.contains(kArtifactsMetaKey)) continue;
        const auto& manifest = meta.at(kArtifactsMetaKey);
        if (p.parent) {
            const auto before = metadata(db, *p.parent).at("meta");
            if (before.is_object() && before.contains(kArtifactsMetaKey) && before.at(kArtifactsMetaKey) == manifest)
                continue;
        }
        checkArtifactReferences(db, object, manifest);
    }
    // Ordinary metadata is independent unless the relationship declares a
    // dependency. Check the final state, not the order of earlier API calls.
    for (const auto& [object, p] : objects) {
        if (!p.exists || p.delta.empty()) continue;
        // A promoted external endpoint has no fabricated local predecessor.
        // Its sealed source still provides the baseline for validating the
        // first genuine override against existing local relationships.
        if(!p.parent && !resolveObject(db,object,true))continue;
        const auto before = p.parent ? metadata(db, *p.parent) : objectToJson(resolveObject(db,object,true)->record);
        const bool lifecycleChanged = before.value("lifecycle_epoch", "") != p.record.value("lifecycle_epoch", "");
        if (!lifecycleChanged && before.at("meta") == p.record.at("meta")) continue;
        for (const auto& edge : incidentEdges(db,object)) {
            bool affected = false;
            for (const auto& [endpoint, anchor] : {std::pair{edge.aUuid, edge.aAnchor}, std::pair{edge.bUuid, edge.bAnchor}})
                affected |= endpoint == object && dependencyMetadata(before, anchor) != dependencyMetadata(p.record, anchor);
            if (!affected) continue;
            Statement validated(db, "SELECT a_meta,b_meta,recorded_only FROM history_validated_edges WHERE uuid=?",
                parseUuid(edge.uuid));
            if (!validated.row() ||
                Json::parse(validated.text(0)) != currentDependencyMetadata(db,edge.aUuid,edge.aAnchor) ||
                Json::parse(validated.text(1)) != currentDependencyMetadata(db,edge.bUuid,edge.bAnchor))
                throw std::runtime_error("object history: metadata/lifecycle edit requires explicit retain/update/remove or keep_recorded for " + edge.uuid);
            if (validated.i64(2)) continue;
            // A validated metadata dependency changes the shared binding even
            // when neither the edge JSON nor its geometry bindings changed.
            execute(db, "INSERT OR IGNORE INTO history_dirty_edges VALUES(?)", parseUuid(edge.uuid));
        }
    }
    struct ChangedEdge {
        std::string uuid, version;
        std::optional<EdgeRecord> record;
        std::optional<std::string> aGeometry, bGeometry;
        std::optional<std::string> aPin, bPin;
        std::set<std::string> owners;
    };
    std::vector<ChangedEdge> edges;
    Statement dirtyEdges(db, "SELECT uuid FROM history_dirty_edges ORDER BY uuid");
    while (dirtyEdges.row()) {
        ChangedEdge e;
        e.uuid = formatUuid(dirtyEdges.blob16(0));
        std::optional<RelationshipRevision> previous;
        Statement old(db, "SELECT object_uuid,relationship_uuid FROM object_relationship_heads WHERE edge_uuid=?",
            parseUuid(e.uuid));
        while (old.row()) {
            e.owners.insert(formatUuid(old.blob16(0)));
            if (!previous) previous = relationship(db, formatUuid(old.blob16(1)));
        }
        if(!previous)previous=recordedRelationship(db,e.uuid);
        Statement row(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE uuid=? AND state=0", parseUuid(e.uuid));
        if (row.row()) {
            e.record = readEdgeRow(db, row);
            // Legacy snapshots can retain a different endpoint side order.
            // Compare their historical pins in the current record's order.
            if(previous && previous->record.aUuid==e.record->bUuid && previous->record.bUuid==e.record->aUuid &&
                previous->record.aUuid!=previous->record.bUuid){
                std::swap(previous->record.aUuid,previous->record.bUuid);
                std::swap(previous->record.aAnchor,previous->record.bAnchor);
                std::swap(previous->aRevision,previous->bRevision);
                std::swap(previous->aGeometryRevision,previous->bGeometryRevision);
            }
            e.owners.insert(e.record->aUuid); e.owners.insert(e.record->bUuid);
            e.aGeometry = geometryHead(db,e.record->aUuid);
            e.bGeometry = geometryHead(db,e.record->bUuid);
            Statement validated(db, "SELECT recorded_only FROM history_validated_edges WHERE uuid=?", parseUuid(e.uuid));
            const bool hasValidation = validated.row();
            const bool recordedOnly = hasValidation && validated.i64(0);
            const bool revalidated = hasValidation && !recordedOnly;
            if (recordedOnly) {
                if (!previous) throw std::runtime_error("object history: no published relationship state to keep");
                e.aGeometry = previous->aGeometryRevision;
                e.bGeometry = previous->bGeometryRevision;
                e.aPin = previous->aRevision;
                e.bPin = previous->bRevision;
            } else if (previous && !revalidated) {
                if (e.aGeometry == previous->aGeometryRevision) e.aPin = previous->aRevision;
                if (e.bGeometry == previous->bGeometryRevision) e.bPin = previous->bRevision;
            }
            if (previous && edgeToJson(previous->record, nullptr, false) == edgeToJson(*e.record, nullptr, false) &&
                e.aGeometry == previous->aGeometryRevision && e.bGeometry == previous->bGeometryRevision &&
                dependencyMetadata(metadata(db, previous->aRevision), e.record->aAnchor) ==
                    currentDependencyMetadata(db,e.record->aUuid,e.record->aAnchor) &&
                dependencyMetadata(metadata(db, previous->bRevision), e.record->bAnchor) ==
                    currentDependencyMetadata(db,e.record->bUuid,e.record->bAnchor)) continue;
            e.version = util::newUuid();
        } else if (!previous) continue;
        for (const auto& owner : e.owners) {
            if(borrowedObject(db,owner))continue; // Sealed endpoints never acquire local object history.
            // An existing endpoint is always dirty through the SQL triggers.
            auto found = objects.find(owner);
            if (found == objects.end()) {
                Pending p;
                p.parent = head(db, owner);
                if (strict) {
                    Statement expected(db, "SELECT revision_uuid FROM history_expected WHERE uuid=?", parseUuid(owner));
                    if (!expected.row() || optionalReference(expected,0) != publicationBase(db,owner))
                        throw std::runtime_error("object history: missing expected relationship endpoint " + owner);
                }
                p.exists = true; p.record = currentRecord(db, owner);
                p.geometry = geometryHead(db, owner); p.delta = Json::array();
                found = objects.emplace(owner, std::move(p)).first;
            }
            found->second.changed = true;
        }
        edges.push_back(std::move(e));
    }
    // A publication time lives on the object revisions themselves. The writer
    // lock serializes allocation; max+1 handles clock ties and clock rollback.
    int64_t publishedAtUs = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    Statement lastTime(db, "SELECT coalesce(max(published_at_us),0) FROM object_revisions");
    lastTime.row();
    if (lastTime.i64(0) == std::numeric_limits<int64_t>::max())
        throw std::overflow_error("object history: publication clock exhausted");
    publishedAtUs = std::max(publishedAtUs, lastTime.i64(0) + 1);
    CommitPublisher commit(db, publishedAtUs, baseline);
    // Allocate all identities first: shared records may pin both revisions
    // published by this action, including cycles and self relationships.
    for (auto& [object, p] : objects) {
        if (!p.exists) {
            if (p.parent) {
                // Record a physical deletion without changing existing deletion
                // eligibility. This prevents time-based reads resurrecting it.
                const auto tombstone = util::newUuid();
                execute(db, "INSERT INTO object_revisions(uuid,object_uuid,parent_uuid,"
                    "metadata_delta,published_at_us,deleted) VALUES(?,?,?,'[]',?,1)", parseUuid(tombstone),
                    parseUuid(object), parseUuid(*p.parent), publishedAtUs);
                commit.object(object,tombstone,true);
            }
            for (const auto* table : {"object_heads", "object_head_records", "object_relationship_heads"}) {
                execute(db, std::string("DELETE FROM ") + table + " WHERE object_uuid=?", parseUuid(object));
            }
            continue;
        }
        if (!p.changed) continue;
        if (!baseline) writeObjectVersion(db, object);
        p.record = currentRecord(db, object);
        p.delta = Json::diff(p.parent ? metadata(db,*p.parent) : Json::object(), p.record);
        p.uuid = util::newUuid();
        // A revision without a parent stores its whole record; it then needs
        // no separate materialized head record.
        Statement insert(db, "INSERT INTO object_revisions(uuid,object_uuid,parent_uuid,geometry_uuid,"
            "metadata_delta,metadata_complete,published_at_us) VALUES(?,?,?,?,?,?,?)", parseUuid(p.uuid), parseUuid(object), sqlReference(p.parent));
        insert.bind(4, sqlReference(p.geometry));
        insert.bind(5, (p.parent ? p.delta : p.record).dump()); insert.bind(6, int64_t(1));
        insert.bind(7, publishedAtUs); insert.stepDone();
        execute(db, "INSERT INTO object_heads VALUES(?,?) ON CONFLICT(object_uuid) DO UPDATE SET revision_uuid=excluded.revision_uuid",
            parseUuid(object), parseUuid(p.uuid));
        if (p.parent) execute(db, "INSERT INTO object_head_records VALUES(?,?,?) ON CONFLICT(object_uuid) "
            "DO UPDATE SET revision_uuid=excluded.revision_uuid,record=excluded.record", parseUuid(object), parseUuid(p.uuid), p.record.dump());
        commit.object(object,p.uuid,false);
    }
    auto pin = [&](const std::string& object,const std::optional<std::string>& geometry) {
        const auto resolved=resolveObject(db,object);
        if(!resolved)throw std::runtime_error("history: endpoint missing");
        if(!historyEnabled(resolved->source)){
            if(geometryHead(resolved->source,object)!=geometry)throw std::runtime_error("history: endpoint geometry unavailable");
            return versionReferenceText(objectVersion(*resolved));
        }
        Statement row(resolved->source,"SELECT r.uuid FROM object_revisions r "
            "WHERE r.object_uuid=? AND r.geometry_uuid IS ? ORDER BY r.seq DESC",parseUuid(object),sqlReference(geometry));
        if(!row.row())throw std::runtime_error("history: endpoint version unavailable");
        return row.uuid(0);
    };
    for (const auto& edge : edges) {
        if (edge.record) {
            const auto aPin = edge.aPin ? *edge.aPin : pin(edge.record->aUuid, edge.aGeometry);
            const auto bPin = edge.bPin ? *edge.bPin : pin(edge.record->bUuid, edge.bGeometry);
            if (!baseline) writeRecordVersion(db,"edges",edge.uuid);
            Statement currentEdge(db, std::string("SELECT ")+kEdgeCols+" FROM edges WHERE uuid=?",parseUuid(edge.uuid)); currentEdge.row();
            execute(db, "INSERT INTO relationship_revisions VALUES(?,?,?,?,?,?,?)", parseUuid(edge.version), parseUuid(edge.uuid),
                edgeToJson(readEdgeRow(db,currentEdge), nullptr, false).dump(), sqlReference(aPin), sqlReference(bPin), sqlReference(edge.aGeometry), sqlReference(edge.bGeometry));
            const auto a = metadata(db, aPin);
            const auto b = metadata(db, bPin);
            bindLifecycle(db, *edge.record, a.value("lifecycle_epoch", ""), b.value("lifecycle_epoch", ""));
            commit.relationship(edge.uuid, edge.version);
        } else {
            commit.relationship(edge.uuid, std::nullopt);
        }
        for (const auto& owner : edge.owners) {
            if(borrowedObject(db,owner)){
                execute(db,"DELETE FROM object_relationship_heads WHERE object_uuid=? AND edge_uuid=?",parseUuid(owner),parseUuid(edge.uuid));
                if(edge.record)execute(db,"INSERT INTO object_relationship_heads VALUES(?,?,?)",parseUuid(owner),parseUuid(edge.uuid),parseUuid(edge.version));
                continue;
            }
            const auto& p = objects.at(owner);
            if (!p.exists) continue;
            Statement delta(db, "INSERT INTO object_relationship_deltas VALUES(?,?,?)", parseUuid(p.uuid), parseUuid(edge.uuid));
            if (edge.record) delta.bind(3, parseUuid(edge.version)); else delta.bindNull(3);
            delta.stepDone();
            execute(db, "DELETE FROM object_relationship_heads WHERE object_uuid=? AND edge_uuid=?", parseUuid(owner), parseUuid(edge.uuid));
            if (edge.record) {
                execute(db, "INSERT INTO object_relationship_heads VALUES(?,?,?)", parseUuid(owner), parseUuid(edge.uuid), parseUuid(edge.version));
            }
        }
    }
    if (commit.sequence())
        if (const auto pool = activeStatementPools.find(db); pool != activeStatementPools.end())
            pool->second->publishedSequence = commit.sequence();
    exec(db, "DELETE FROM history_dirty_objects; DELETE FROM history_dirty_edges;"
             "DELETE FROM history_expected; DELETE FROM history_checked; DELETE FROM history_validated_edges;"
             "DELETE FROM history_required;");
}

namespace {
// Revision columns shared by the single and batched reads, starting at `c`:
// object_uuid, parent_uuid, geometry_uuid, seq, metadata_complete,
// published_at_us, deleted.
ObjectRevision revisionColumns(Statement& s, int c, std::string uuid)
{
    ObjectRevision result;
    result.uuid = std::move(uuid); result.objectUuid = s.uuid(c); result.parentUuid = s.optUuid(c + 1);
    result.geometryRevision = optionalReference(s, c + 2); result.sequence = s.i64(c + 3);
    result.metadataComplete = s.i64(c + 4) != 0;
    if (!s.isNull(c + 5)) result.publishedAtUs = s.i64(c + 5);
    result.deleted = s.i64(c + 6) != 0;
    return result;
}

void setRecord(ObjectRevision& revision, const Json& record)
{
    revision.record = objectFromJson(record); commonFromJson(record, revision.record);
    revision.record.versionStoreUuid=revision.storeUuid;
}
} // namespace

std::optional<ResolvedObjectVersion> detail::resolveObjectRevision(sqlite3* db,const std::string& uuid)
{
    if(const auto ref=parseVersionReference(uuid))return resolveObjectVersion(db,*ref);
    const auto canonical=formatUuid(parseUuid(uuid));
    db=resolveReference(db,[&](sqlite3* source){
        if(!historyEnabled(source))return false;
        Statement row(source,"SELECT 1 FROM object_revisions WHERE uuid=?",parseUuid(canonical));return row.row();
    });
    if(!db)return std::nullopt;
    Statement s(db,"SELECT object_uuid,parent_uuid,geometry_uuid,seq,metadata_complete,published_at_us,deleted FROM object_revisions WHERE uuid=?",parseUuid(canonical));
    if(!s.row())return std::nullopt;
    auto result=revisionColumns(s,0,canonical);
    result.storeUuid=storeUuidText(db);
    Statement version(db,"SELECT version,generation,commit_seq FROM object_versions WHERE revision_uuid=?",parseUuid(canonical));
    if(version.row()){result.version=version.i64(0);result.generation=version.uuid(1);if(!version.isNull(2))result.commitSequence=version.i64(2);}
    setRecord(result,metadata(db,canonical));
    return ResolvedObjectVersion{db,std::move(result)};
}
std::optional<ResolvedObjectVersion> detail::resolveObjectVersion(sqlite3* db,const ObjectVersion& reference)
{
    const auto store=formatUuid(parseUuid(reference.storeUuid)),generation=formatUuid(parseUuid(reference.generation));
    std::optional<ResolvedObjectVersion> result;
    resolveReference(db,[&](sqlite3* source){
        if(storeUuidText(source)!=store)return false;
        if(!historyEnabled(source)){
            Statement row(source,std::string("SELECT ")+kObjectCols+" FROM objects WHERE uuid=? AND state<>2",parseUuid(reference.objectUuid));
            if(!row.row())return false;
            auto record=readObjectRow(source,row);
            if(record.version!=reference.version||record.versionGeneration!=generation)return false;
            ObjectRevision revision;
            revision.storeUuid=store;
            revision.objectUuid=record.uuid;
            revision.version=reference.version;revision.generation=generation;revision.metadataComplete=true;
            revision.commitSequence=record.commitSequence;revision.geometryRevision=geometryHead(source,record.uuid);
            revision.record=std::move(record);
            result=ResolvedObjectVersion{source,std::move(revision)};return true;
        }
        Statement row(source,"SELECT revision_uuid FROM object_versions WHERE object_uuid=? AND version=? AND generation=?",
            parseUuid(reference.objectUuid),reference.version,parseUuid(generation));
        if(!row.row())return false;
        result=resolveObjectRevision(source,row.uuid(0));return bool(result);
    });
    return result;
}
std::optional<ObjectRevision> ObjectStore::objectRevision(const Txn& txn,const std::string& uuid)const
{
    const auto value=resolveObjectRevision(static_cast<sqlite3*>(connection(txn)),uuid);
    return value?std::optional(value->revision):std::nullopt;
}
std::vector<std::optional<ObjectRevision>> ObjectStore::objectsAtVersions(const Txn& txn,const std::vector<ObjectVersion>& references)const
{
    auto* db=static_cast<sqlite3*>(connection(txn));
    std::vector<std::optional<ObjectRevision>> result;result.reserve(references.size());
    for(const auto& reference:references){
        auto value=resolveObjectVersion(db,reference);
        result.push_back(value?std::optional(std::move(value->revision)):std::nullopt);
    }
    return result;
}

std::optional<ObjectRevision> ObjectStore::currentObjectRevision(const Txn& txn, const std::string& uuid) const
{
    auto* db=static_cast<sqlite3*>(connection(txn));
    const auto revision = head(db, formatUuid(parseUuid(uuid)));
    return revision ? objectRevision(txn, *revision) : std::nullopt;
}

std::vector<ObjectRevision> ObjectStore::objectHistory(const Txn& txn, const std::string& object) const
{
    auto* db=static_cast<sqlite3*>(connection(txn));
    // History follows the visible owner, just like current reads. A deleted
    // local object has no visible owner but still retains its local history.
    auto* source=objectSource(db,object);
    if(!source)source=db;
    requireHistory(source);
    Statement s(source, "SELECT uuid FROM object_revisions WHERE object_uuid=? ORDER BY seq",
        parseUuid(object));
    std::vector<ObjectRevision> result;
    while (s.row()) result.push_back(*objectRevision(txn, formatUuid(s.blob16(0))));
    return result;
}

RevisionSelection ObjectStore::revisionSelectionAt(const Txn& txn, int64_t publishedAtUs, bool includeRetired) const
{
    auto* db=static_cast<sqlite3*>(connection(txn));
    return detail::revisionSelectionAt(db,storeUuidText(db),publishedAtUs,includeRetired);
}

RevisionSelection detail::revisionSelectionAt(sqlite3* db,const std::string& owner,int64_t publishedAtUs,bool includeRetired)
{
    if (publishedAtUs < 0) throw std::invalid_argument("object history: publication time must be nonnegative");
    RevisionSelection result;bool found=false;
    for(auto* source:querySources(db)) {
    if(storeUuidText(source)!=owner)continue;
    found=true;requireHistory(source);
    Statement s(source,
        "SELECT object_uuid,uuid FROM (SELECT object_uuid,uuid,deleted,"
        "row_number() OVER (PARTITION BY object_uuid ORDER BY published_at_us DESC,seq DESC) AS position "
        "FROM object_revisions WHERE published_at_us<=?) WHERE position=1 AND deleted=0 ORDER BY object_uuid",
        publishedAtUs);
    while (s.row())
        if (includeRetired || !resolveObjectRevision(source,formatUuid(s.blob16(1)))->revision.record.retired)
            result.emplace(formatUuid(s.blob16(0)), formatUuid(s.blob16(1)));
    }
    if(!found)throw std::invalid_argument("object history: snapshot owner is unavailable");
    return result;
}

std::vector<ObjectVersion> ObjectStore::versionsAt(const Txn& txn, int64_t publishedAtUs, bool includeRetired) const
{
    std::vector<ObjectVersion> result;
    for (const auto& [object, revision] : revisionSelectionAt(txn, publishedAtUs, includeRetired)) {
        const auto value = objectRevision(txn, *revision);
        if (!value->version)
            throw std::runtime_error("object history: revision of " + object + " has no numbered version");
        result.push_back(value->reference());
    }
    return result;
}

std::vector<SelectedRelationship> ObjectStore::relationshipsAt(const Txn& txn, const RevisionSelection& selection) const
{
    DependencyScope dependencies;
    auto* db = static_cast<sqlite3*>(connection(txn));
    std::map<std::string, ObjectRevision> revisions;
    std::map<std::string, std::map<std::string, std::string>> memberships;
    std::set<std::string> versions;
    std::map<std::string,RelationshipRevision> currentOnly;
    for (const auto& [object, revision] : selection) {
        const auto uuid = formatUuid(parseUuid(object));
        const auto value = revision ? objectRevision(txn, *revision) : currentObjectRevision(txn, uuid);
        if (!value || value->objectUuid != uuid || value->deleted)
            throw std::invalid_argument("object history: selection revision does not belong to object");
        auto selected = members(db, revision.value_or(head(db,uuid).value_or(value->uuid)));
        // A current-only state has no archived membership rows. Its pinned
        // source supplies the binding; use the same endpoint evaluator below.
        const auto source=resolveObjectVersion(db,versionReference(txn,*value));
        if(value->uuid.empty()||!historyEnabled(db)){
            std::vector<sqlite3*> scopes{source->source};
            if(source->source!=db)scopes.push_back(db);
            for(auto* scope:scopes)for(const auto& visible:incidentEdges(scope,uuid)){
                const auto binding=recordedRelationship(scope,visible.uuid);
                if(!binding)continue;
                const auto key=binding->uuid.empty()?Json::array({visible.uuid,binding->generation,binding->version}).dump():binding->uuid;
                selected[visible.uuid]=key;
                currentOnly.insert_or_assign(key,*binding);
            }
        }
        for (const auto& [edge, version] : selected) versions.insert(version);
        memberships.emplace(uuid, std::move(selected));
        revisions.emplace(uuid, *value);
    }
    std::vector<SelectedRelationship> result;
    for (const auto& version : versions) {
        auto edge = currentOnly.contains(version)?currentOnly.at(version):relationship(db, version);
        auto endpoint = [&](const std::string& object, const Anchor& anchor, const std::optional<std::string>& geometry,
                            const std::string& pinned) {
            const auto found = revisions.find(object);
            if (found == revisions.end()) return false;
            const auto& membership = memberships.at(object);
            const auto member = membership.find(edge.record.uuid);
            return member != membership.end() && member->second == version &&
                endpointApplies(*this, txn, anchor, geometry, pinned, found->second);
        };
        const bool applicable = endpoint(edge.record.aUuid, edge.record.aAnchor, edge.aGeometryRevision, edge.aRevision) &&
                                endpoint(edge.record.bUuid, edge.record.bAnchor, edge.bGeometryRevision, edge.bRevision);
        result.push_back({std::move(edge), applicable});
    }
    return result;
}

std::optional<RelationshipRevision> ObjectStore::relationshipAtVersion(const Txn& txn, const RelationshipVersion& ref) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto store=formatUuid(parseUuid(ref.storeUuid)),generation=formatUuid(parseUuid(ref.generation));
    std::optional<RelationshipRevision> result;
    resolveReference(db,[&](sqlite3* source) {
        if(storeUuidText(source)!=store)return false;
        if(!historyEnabled(source)) {
            Statement row(source,std::string("SELECT ")+kEdgeCols+
                " FROM edges WHERE uuid=? AND state=0",parseUuid(ref.relationshipUuid));
            if(!row.row())return false;
            const auto edge=readEdgeRow(source,row);
            if(edge.version!=ref.version||edge.versionGeneration!=generation)return false;
            RelationshipRevision value;
            value.record=edge;value.version=ref.version;value.generation=generation;
            const auto a=resolveObject(source,edge.aUuid),b=resolveObject(source,edge.bUuid);
            value.aVersion=a?a->record.version.value_or(0):0;
            value.bVersion=b?b->record.version.value_or(0):0;
            value.aGeometryRevision=geometryHead(source,edge.aUuid);value.bGeometryRevision=geometryHead(source,edge.bUuid);
            result=std::move(value);return true;
        }
        Statement row(source,"SELECT revision_uuid FROM relationship_versions WHERE edge_uuid=? AND version=? AND generation=? "
            "AND revision_uuid IS NOT NULL",parseUuid(ref.relationshipUuid),ref.version,parseUuid(generation));
        if(!row.row())return false;
        result=relationship(source,row.uuid(0));return true;
    });
    return result;
}

void detail::requirePublishedGraphRead(const Txn& txn)
{
    auto* db = ObjectStoreImplAccess::conn(txn);
    const auto pool = activeStatementPools.find(db);
    if (pool == activeStatementPools.end() || !pool->second->readSnapshot)
        throw std::invalid_argument("applicability graph queries require a read transaction");
}

bool detail::currentRelationshipApplicable(const ObjectStore& store, const Txn& txn,
                                          const std::string& edgeUuid)
{
    auto* db = ObjectStoreImplAccess::conn(txn);
    if(!resolveEdge(db,edgeUuid))return false;
    const auto recorded=recordedRelationship(db,edgeUuid);
    if(!recorded)return false;
    const auto& edge=*recorded;
    const auto a=store.currentObjectRevision(txn,edge.record.aUuid);
    const auto b=store.currentObjectRevision(txn,edge.record.bUuid);
    return a && b && endpointApplies(store,txn,edge.record.aAnchor,edge.aGeometryRevision,edge.aRevision,*a) &&
        endpointApplies(store,txn,edge.record.bAnchor,edge.bGeometryRevision,edge.bRevision,*b);
}

std::optional<RelationshipRevision> detail::recordedRelationship(sqlite3* db,const std::string& uuid)
{
    // A local retained binding takes precedence. Otherwise the visible edge's
    // source supplies its original binding, including current-only sources.
    if(historyEnabled(db)){
        Statement head(db,"SELECT relationship_uuid FROM object_relationship_heads WHERE edge_uuid=? LIMIT 1",parseUuid(uuid));
        if(head.row())return relationship(db,head.uuid(0));
    }
    auto* source=edgeSource(db,uuid);
    if(source==db&&historyEnabled(db))source=edgeSource(db,uuid,true);
    if(!source)return {};
    if(historyEnabled(source)){
        Statement head(source,"SELECT relationship_uuid FROM object_relationship_heads WHERE edge_uuid=? LIMIT 1",parseUuid(uuid));
        return head.row()?std::optional(relationship(source,head.uuid(0))):std::nullopt;
    }
    const auto edge=resolveEdge(source,uuid);
    if(!edge)return {};
    const auto a=resolveObject(source,edge->record.aUuid),b=resolveObject(source,edge->record.bUuid);
    if(!a||!b)return {};
    RelationshipRevision result;result.record=edge->record;
    result.version=edge->record.version.value_or(0);result.generation=edge->record.versionGeneration;
    result.aRevision=versionReferenceText(objectVersion(*a));result.bRevision=versionReferenceText(objectVersion(*b));
    result.aVersion=a->record.version.value_or(0);result.bVersion=b->record.version.value_or(0);
    result.aGeometryRevision=geometryHead(source,edge->record.aUuid);result.bGeometryRevision=geometryHead(source,edge->record.bUuid);
    return result;
}

void ObjectStore::retainRelationship(WriteTxn& txn, const std::string& uuid)
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto edge = edgeByUuid(txn, uuid);
    if (!edge) throw std::invalid_argument("object history: unknown relationship");
    Savepoint savepoint(db);
    EdgeGeometryRepair repair;
    repair.edgeUuid = edge->uuid; repair.expectedVersion = edge->version.value_or(0);
    repair.aRevision = geometryHead(db, edge->aUuid); repair.bRevision = geometryHead(db, edge->bUuid);
    repair.aAnchor = edge->aAnchor; repair.bAnchor = edge->bAnchor;
    repairEdgeGeometry(txn, repair);
    recordRelationshipValidation(db, *edge);
    savepoint.release();
}

void ObjectStore::keepRecordedRelationshipsOf(WriteTxn& txn, const std::string& objectUuid)
{
    for (const auto& edge : edgesOf(txn, objectUuid, true)) keepRecordedRelationship(txn, edge.uuid);
}

void ObjectStore::keepRecordedRelationship(WriteTxn& txn, const std::string& uuid)
{
    DependencyScope dependencies;
    auto* db = static_cast<sqlite3*>(connection(txn));
    requireHistory(db);
    const auto edge = edgeByUuid(txn, uuid);
    if (!edge) throw std::invalid_argument("object history: unknown relationship");
    const auto published=recordedRelationship(db,uuid);
    if (!published)
        throw std::invalid_argument("object history: an unpublished relationship has no recorded state to keep; retain, update or remove it instead");
    auto binding=*published;
    if (binding.record.aUuid==edge->bUuid && binding.record.bUuid==edge->aUuid && edge->aUuid!=edge->bUuid) {
        std::swap(binding.record.aUuid,binding.record.bUuid);
        std::swap(binding.aGeometryRevision,binding.bGeometryRevision);
    }
    if (binding.record.aUuid!=edge->aUuid || binding.record.bUuid!=edge->bUuid)
        throw std::invalid_argument("object history: changed relationship endpoints require retain, update or remove");
    for (const auto& [revision,anchor] : {std::pair{binding.aGeometryRevision,edge->aAnchor},
                                         std::pair{binding.bGeometryRevision,edge->bAnchor}}) {
        if (!revision) continue;
        const auto geometry=geometryRevision(txn,*revision);
        if (!geometry) throw std::runtime_error("recorded relationship geometry is missing");
        validateDependencies(anchor,dependencyGeometry(*this,txn,*geometry));
    }
    Savepoint save(db);
    if(createRecordOverride(db,"edges",uuid))
        stampContentChange(db,author(),hostname(),"edges",*edgeIdByUuid(db,parseUuid(uuid)),parseUuid(uuid),"keep_recorded");
    recordRelationshipValidation(db,*edge,true);
    save.release();
}
} // namespace vc::objects
