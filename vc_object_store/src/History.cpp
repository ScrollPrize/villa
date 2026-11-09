#include "internal/History.hpp"
#include "internal/Artifacts.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include "internal/Versions.hpp"
#include "internal/RecordJson.hpp"
#include "internal/RecordSql.hpp"
#include "internal/EdgeSql.hpp"
#include "geometry/Dependencies.hpp"
#include "internal/ImmutableImport.hpp"
#include "internal/ExternalEndpoint.hpp"
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
    // No geometry descriptors, tags, subtype rows, provenance, or full-record
    // serialization are dependencies. Do not even fetch/parse meta when the
    // relationship declares no metadata keys; lifecycle checks remain present.
    Statement row(db,anchor.metadataKeys.empty()
        ? "SELECT (state<>0),lifecycle_epoch,NULL FROM objects WHERE uuid=?"
        : "SELECT (state<>0),lifecycle_epoch,meta FROM objects WHERE uuid=?",
        parseUuid(object));
    if(!row.row()) {
        const auto record=externalEndpointRecord(db,object);
        const auto meta=anchor.metadataKeys.empty()?nlohmann::json{}:nlohmann::json::parse(record.meta);
        return dependencyFields(&meta,record.retired,record.lifecycleEpoch,anchor);
    }
    const auto meta=anchor.metadataKeys.empty() ? nlohmann::json{} : nlohmann::json::parse(row.text(2));
    return dependencyFields(&meta,row.i64(0)!=0,row.text(1),anchor);
}
namespace {
using Json = nlohmann::json;

// UUID references use the same BLOB16 representation as current records.
// Explicit external/current version tokens are not UUIDs and remain tagged text.
using SqlReference=std::variant<std::array<unsigned char,16>,std::string>;
SqlReference sqlReference(const std::string& value)
{
    if(auto external=parseExternalEndpointToken(value))
        return externalEndpointToken(external->ownerUuid,external->objectUuid,external->generation,external->commitSequence);
    if(auto current=parseCurrentGeometryToken(value))
        return currentGeometryToken(current->objectUuid,current->version);
    return parseUuid(value);
}
std::optional<SqlReference> sqlReference(const std::optional<std::string>& value)
{return value?std::optional(sqlReference(*value)):std::nullopt;}
std::string referenceColumn(const Statement& row,int column)
{
    if(sqlite3_column_type(row.value,column)==SQLITE_BLOB)return formatUuid(row.blob16(column));
    const auto text=row.text(column);
    if(parseExternalEndpointToken(text)||parseCurrentGeometryToken(text))return text;
    throw std::runtime_error("object history: UUID reference must be BLOB16");
}
std::optional<std::string> optionalReference(const Statement& row,int column)
{return row.isNull(column)?std::nullopt:std::optional(referenceColumn(row,column));}

std::optional<std::string> head(sqlite3* db, const std::string& object)
{
    Statement s(db, "SELECT revision_uuid FROM object_heads WHERE object_uuid=?", parseUuid(object));
    if(s.row())return formatUuid(s.blob16(0));
    if(externalEndpointKey(db,object))return externalEndpointVersion(db,object);
    return {};
}

std::optional<std::string> optionalUuid(Statement& s, int column)
{
    return s.isNull(column) ? std::nullopt : std::optional(formatUuid(s.blob16(column)));
}

Json metadata(sqlite3* db, const std::string& revision, bool legacyTextUuid=false)
{
    if(auto endpoint=externalEndpointObject(db,revision))return objectToJson(externalEndpointRecord(db,*endpoint));
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
    if(auto endpoint=externalEndpointObject(db,revision)) {
        Statement rows(db,"SELECT edge_uuid,relationship_uuid FROM object_relationship_heads WHERE object_uuid=?",parseUuid(*endpoint));
        while(rows.row())result.emplace(formatUuid(rows.blob16(0)),formatUuid(rows.blob16(1)));return result;
    }
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
        if (const auto external = parseExternalEndpointToken(reference))
            return externalEndpointRecord(db, external->objectUuid).version.value_or(0);
        return revisionVersion(db,reference);
    };
    result.aVersion = endpointVersion(result.aRevision);
    result.bVersion = endpointVersion(result.bRevision);
    return result;
}

// One dependency definition for selected graphs and current indexed queries.
bool endpointApplies(const ObjectStore& store, const Txn& txn, const Anchor& anchor,
                     const std::optional<std::string>& geometry, const std::string& pinned,
                     const ObjectRevision& selected)
{
    if (selected.deleted || selected.record.retired) return false;
    if(parseExternalEndpointToken(pinned))return pinned==selected.uuid;
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
  geometry_uuid BLOB REFERENCES geometry_revisions(uuid),
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
        const auto wanted = revision ? std::optional(parseExternalEndpointToken(*revision)?*revision:formatUuid(parseUuid(*revision))) : std::nullopt;
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
                if (externalEndpointKey(db,endpoint)) continue;
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
        if(externalEndpointKey(db,object))continue;
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
    if(!object.row())return;
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
        Statement incident(db,std::string("SELECT ")+kEdgeCols+" FROM edges WHERE state=0 AND (a_uuid=?1 OR b_uuid=?1)",parseUuid(uuid));
        while(incident.row()) {
            const auto edge=readEdgeRow(db,incident);
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
            if (!expected.row() || optionalReference(expected,0) != head(db, formatUuid(required.blob16(0))))
                throw std::runtime_error("object history: missing expected validation endpoint " + formatUuid(required.blob16(0)));
        }
    }
    while (dirty.row()) {
        const auto object = formatUuid(dirty.blob16(0));
        if(externalEndpointKey(db,object))continue;
        Pending p;
        p.parent = head(db, object);
        if (strict) {
            Statement expected(db, "SELECT revision_uuid FROM history_expected WHERE uuid=?", parseUuid(object));
            if (!expected.row() || optionalReference(expected,0) != p.parent)
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
                p.changed |= optionalUuid(previous, 0) != p.geometry;
            }
        }
        objects.emplace(object, std::move(p));
    }
    // A changed artifact manifest may only name bytes the store holds, whichever
    // API wrote it.
    std::optional<std::filesystem::path> storePath;
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
        if (!storePath) storePath = std::filesystem::path(sqlite3_db_filename(db, "main"));
        checkArtifactReferences(*storePath, object, manifest);
    }
    // Ordinary metadata is independent unless the relationship declares a
    // dependency. Check the final state, not the order of earlier API calls.
    for (const auto& [object, p] : objects) {
        if (!p.exists || p.delta.empty()) continue;
        // A promoted external endpoint has no fabricated local predecessor.
        // Its sealed source still provides the baseline for validating the
        // first genuine override against existing local relationships.
        if(!p.parent && !externalEndpointKey(db,object,true))continue;
        const auto before = p.parent ? metadata(db, *p.parent) : objectToJson(externalEndpointRecord(db,object));
        const bool lifecycleChanged = before.value("lifecycle_epoch", "") != p.record.value("lifecycle_epoch", "");
        if (!lifecycleChanged && before.at("meta") == p.record.at("meta")) continue;
        Statement incident(db, std::string("SELECT ") + kEdgeCols +
            " FROM edges WHERE state=0 AND (a_uuid=? OR b_uuid=?)",
            parseUuid(object), parseUuid(object));
        while (incident.row()) {
            const auto edge = readEdgeRow(db, incident);
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
            if(externalEndpointKey(db,owner))continue; // Sealed endpoints never acquire local object history.
            // An existing endpoint is always dirty through the SQL triggers.
            auto found = objects.find(owner);
            if (found == objects.end()) {
                Pending p;
                p.parent = head(db, owner);
                if (strict) {
                    Statement expected(db, "SELECT revision_uuid FROM history_expected WHERE uuid=?", parseUuid(owner));
                    if (!expected.row() || optionalReference(expected,0) != p.parent)
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
        if (p.geometry) insert.bind(4, parseUuid(*p.geometry)); else insert.bindNull(4);
        insert.bind(5, (p.parent ? p.delta : p.record).dump()); insert.bind(6, int64_t(1));
        insert.bind(7, publishedAtUs); insert.stepDone();
        execute(db, "INSERT INTO object_heads VALUES(?,?) ON CONFLICT(object_uuid) DO UPDATE SET revision_uuid=excluded.revision_uuid",
            parseUuid(object), parseUuid(p.uuid));
        if (p.parent) execute(db, "INSERT INTO object_head_records VALUES(?,?,?) ON CONFLICT(object_uuid) "
            "DO UPDATE SET revision_uuid=excluded.revision_uuid,record=excluded.record", parseUuid(object), parseUuid(p.uuid), p.record.dump());
        commit.object(object,p.uuid,false);
    }
    auto pin = [&](const std::string& object, const std::optional<std::string>& geometry) {
        if(externalEndpointKey(db,object))return externalEndpointVersion(db,object);
        Statement s(db, "SELECT uuid FROM object_revisions WHERE object_uuid=? AND geometry_uuid IS ? ORDER BY seq DESC",
            parseUuid(object));
        if (geometry) s.bind(2, parseUuid(*geometry)); else s.bindNull(2);
        if (s.row()) return formatUuid(s.blob16(0));
        throw std::runtime_error("object history: relationship binding has no object revision");
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
            if(externalEndpointKey(db,owner)){
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
    result.geometryRevision = optionalUuid(s, c + 2); result.sequence = s.i64(c + 3);
    result.metadataComplete = s.i64(c + 4) != 0;
    if (!s.isNull(c + 5)) result.publishedAtUs = s.i64(c + 5);
    result.deleted = s.i64(c + 6) != 0;
    return result;
}

void setRecord(ObjectRevision& revision, const Json& record)
{
    revision.record = objectFromJson(record); commonFromJson(record, revision.record);
}
} // namespace

std::optional<ObjectRevision> ObjectStore::objectRevision(const Txn& txn, const std::string& uuid) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    requireHistory(db);
    if(auto endpoint=externalEndpointObject(db,uuid)) {
        ObjectRevision result;result.uuid=externalEndpointVersion(db,*endpoint);result.objectUuid=*endpoint;
        result.record=externalEndpointRecord(db,*endpoint);result.metadataComplete=true;
        result.generation=parseExternalEndpointToken(result.uuid)->generation;
        result.version=result.record.version.value_or(0);
        result.commitSequence=result.record.commitSequence;
        return result;
    }
    const auto canonical = formatUuid(parseUuid(uuid));
    Statement s(db, "SELECT object_uuid,parent_uuid,geometry_uuid,seq,metadata_complete,published_at_us,deleted "
        "FROM object_revisions WHERE uuid=?", parseUuid(canonical));
    if (!s.row()) return std::nullopt;
    ObjectRevision result;
    result.uuid = canonical; result.objectUuid = formatUuid(s.blob16(0)); result.parentUuid = optionalUuid(s,1);
    result.geometryRevision = optionalUuid(s, 2); result.sequence = s.i64(3);
    result.metadataComplete = s.i64(4) != 0;
    if (!s.isNull(5)) result.publishedAtUs = s.i64(5);
    result.deleted = s.i64(6) != 0;
    Statement version(db, "SELECT version,generation,commit_seq FROM object_versions WHERE revision_uuid=?",parseUuid(canonical));
    if (version.row()) {
        result.version=version.i64(0); result.generation=version.uuid(1);
        if (!version.isNull(2)) result.commitSequence=version.i64(2);
    }
    const auto record = metadata(db, canonical);
    result.record = objectFromJson(record); commonFromJson(record, result.record);
    return result;
}

std::vector<std::optional<ObjectRevision>> ObjectStore::objectsAtVersions(const Txn& txn,
    const std::vector<ObjectVersion>& references) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto store = storeUuidText(db);
    if (!historyEnabled(db)) {
        std::vector<std::optional<ObjectRevision>> result;
        result.reserve(references.size());
        for (const auto& reference : references) {
            auto current = currentObjectRevision(txn, reference.objectUuid);
            if (formatUuid(parseUuid(reference.storeUuid)) != store || !current ||
                current->version != reference.version || current->generation != util::canonicalUuid(reference.generation))
                current.reset();
            result.push_back(std::move(current));
        }
        return result;
    }
    // One prepared lookup per reference. A version that is still the head
    // takes its record from the materialized head in the same row, and a
    // revision without a parent holds its whole record; other versions
    // replay their metadata history.
    Statement s(db, "SELECT v.revision_uuid,r.object_uuid,r.parent_uuid,r.geometry_uuid,r.seq,r.metadata_complete,"
        "r.published_at_us,r.deleted,v.commit_seq,h.record,iif(r.parent_uuid IS NULL,r.metadata_delta,NULL) FROM object_versions v "
        "JOIN object_revisions r ON r.uuid=v.revision_uuid "
        "LEFT JOIN object_head_records h ON h.revision_uuid=v.revision_uuid "
        "WHERE v.object_uuid=? AND v.version=? AND v.generation=?");
    std::vector<std::optional<ObjectRevision>> result;
    result.reserve(references.size());
    for (const auto& reference : references) {
        if (formatUuid(parseUuid(reference.storeUuid)) != store) {
            result.emplace_back();
            continue;
        }
        const auto generation = formatUuid(parseUuid(reference.generation));
        s.reset();
        s.bindAll(parseUuid(reference.objectUuid), reference.version, parseUuid(generation));
        if (!s.row()) {
            result.emplace_back();
            continue;
        }
        auto revision = revisionColumns(s, 1, s.uuid(0));
        revision.version = reference.version;
        revision.generation = generation;
        if (!s.isNull(8)) revision.commitSequence = s.i64(8);
        setRecord(revision, !s.isNull(9) ? Json::parse(s.text(9)) : !s.isNull(10) ? Json::parse(s.text(10)) : metadata(db, revision.uuid));
        result.push_back(std::move(revision));
    }
    return result;
}

std::optional<ObjectRevision> ObjectStore::currentObjectRevision(const Txn& txn, const std::string& uuid) const
{
    auto* db=static_cast<sqlite3*>(connection(txn));
    if (!historyEnabled(db)) {
        const auto record=objectByUuid(txn,uuid);
        if (!record) return std::nullopt;
        ObjectRevision current;
        current.objectUuid=record->uuid; current.record=*record;
        current.version=record->version.value_or(0); current.generation=record->versionGeneration;
        current.commitSequence=record->commitSequence;
        current.geometryRevision=geometryHead(db,record->uuid);
        return current;
    }
    const auto revision = head(static_cast<sqlite3*>(connection(txn)), formatUuid(parseUuid(uuid)));
    return revision ? objectRevision(txn, *revision) : std::nullopt;
}

std::vector<ObjectRevision> ObjectStore::objectHistory(const Txn& txn, const std::string& object) const
{
    requireHistory(static_cast<sqlite3*>(connection(txn)));
    Statement s(static_cast<sqlite3*>(connection(txn)), "SELECT uuid FROM object_revisions WHERE object_uuid=? ORDER BY seq",
        parseUuid(object));
    std::vector<ObjectRevision> result;
    while (s.row()) result.push_back(*objectRevision(txn, formatUuid(s.blob16(0))));
    return result;
}

RevisionSelection ObjectStore::revisionSelectionAt(const Txn& txn, int64_t publishedAtUs, bool includeRetired) const
{
    requireHistory(static_cast<sqlite3*>(connection(txn)));
    if (publishedAtUs < 0) throw std::invalid_argument("object history: publication time must be nonnegative");
    Statement s(static_cast<sqlite3*>(connection(txn)),
        "SELECT object_uuid,uuid FROM (SELECT object_uuid,uuid,deleted,"
        "row_number() OVER (PARTITION BY object_uuid ORDER BY published_at_us DESC,seq DESC) AS position "
        "FROM object_revisions WHERE published_at_us<=?) WHERE position=1 AND deleted=0 ORDER BY object_uuid",
        publishedAtUs);
    RevisionSelection result;
    while (s.row())
        if (includeRetired || !objectRevision(txn, formatUuid(s.blob16(1)))->record.retired)
            result.emplace(formatUuid(s.blob16(0)), formatUuid(s.blob16(1)));
    return result;
}

std::vector<ObjectVersion> ObjectStore::versionsAt(const Txn& txn, int64_t publishedAtUs, bool includeRetired) const
{
    const auto store = status(txn).storeUuid;
    std::vector<ObjectVersion> result;
    for (const auto& [object, revision] : revisionSelectionAt(txn, publishedAtUs, includeRetired)) {
        const auto value = objectRevision(txn, *revision);
        if (!value->version)
            throw std::runtime_error("object history: revision of " + object + " has no numbered version");
        result.push_back({store, value->generation, object, value->version});
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
    for (const auto& [object, revision] : selection) {
        const auto uuid = formatUuid(parseUuid(object));
        const auto value = revision ? objectRevision(txn, *revision) : currentObjectRevision(txn, uuid);
        if (!value || value->objectUuid != uuid || value->deleted)
            throw std::invalid_argument("object history: selection revision does not belong to object");
        auto selected = members(db, value->uuid);
        for (const auto& [edge, version] : selected) versions.insert(version);
        memberships.emplace(uuid, std::move(selected));
        revisions.emplace(uuid, *value);
    }
    std::vector<SelectedRelationship> result;
    for (const auto& version : versions) {
        auto edge = relationship(db, version);
        auto endpoint = [&](const std::string& object, const Anchor& anchor, const std::optional<std::string>& geometry,
                            const std::string& pinned) {
            const auto found = revisions.find(object);
            if (found == revisions.end()) return false;
            if(parseExternalEndpointToken(pinned))return endpointApplies(*this,txn,anchor,geometry,pinned,found->second);
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
    if (formatUuid(parseUuid(ref.storeUuid)) != storeUuidText(db)) return std::nullopt;
    if (!historyEnabled(db)) {
        const auto edge=edgeByUuid(txn,ref.relationshipUuid);
        if (!edge || edge->version!=ref.version || edge->versionGeneration!=util::canonicalUuid(ref.generation)) return std::nullopt;
        RelationshipRevision result;
        result.record=*edge; result.version=ref.version; result.generation=edge->versionGeneration;
        result.aVersion=currentVersion(txn,edge->aUuid).value_or(0);
        result.bVersion=currentVersion(txn,edge->bUuid).value_or(0);
        result.aGeometryRevision=geometryHead(db,edge->aUuid); result.bGeometryRevision=geometryHead(db,edge->bUuid);
        return result;
    }
    Statement s(db, "SELECT revision_uuid FROM relationship_versions WHERE edge_uuid=? AND version=? AND generation=? "
                    "AND revision_uuid IS NOT NULL",
        parseUuid(ref.relationshipUuid), ref.version, parseUuid(ref.generation));
    return s.row() ? std::optional(relationship(db, s.uuid(0))) : std::nullopt;
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
    Statement current(db,"SELECT a_uuid,b_uuid FROM edges WHERE state=0 AND uuid=? AND "+activeEdgeSql(),parseUuid(edgeUuid));
    if (!current.row()) return false;
    if (!historyEnabled(db)) return true;
    // Retained history may contain deliberately kept-as-recorded observations.
    // Resolve their existing archived pins, not a second current binding table.
    Statement membership(db,
        "SELECT a.relationship_uuid FROM object_relationship_heads a "
        "JOIN object_relationship_heads b ON b.edge_uuid=a.edge_uuid AND b.relationship_uuid=a.relationship_uuid "
        "WHERE a.object_uuid=? AND b.object_uuid=? AND a.edge_uuid=?",
        current.blob16(0),current.blob16(1),parseUuid(edgeUuid));
    if (!membership.row()) return false;
    const auto edge=relationship(db,membership.uuid(0));
    const auto a=store.currentObjectRevision(txn,edge.record.aUuid);
    const auto b=store.currentObjectRevision(txn,edge.record.bUuid);
    return a && b && endpointApplies(store,txn,edge.record.aAnchor,edge.aGeometryRevision,edge.aRevision,*a) &&
        endpointApplies(store,txn,edge.record.bAnchor,edge.bGeometryRevision,edge.bRevision,*b);
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
    Statement published(db,"SELECT relationship_uuid FROM object_relationship_heads WHERE edge_uuid=? LIMIT 1",parseUuid(edge->uuid));
    if (!published.row())
        throw std::invalid_argument("object history: an unpublished relationship has no recorded state to keep; retain, update or remove it instead");
    auto binding=relationship(db,published.uuid(0));
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
    recordRelationshipValidation(db,*edge,true);
}
} // namespace vc::objects
