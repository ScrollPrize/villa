#include "internal/History.hpp"
#include "internal/RecordJson.hpp"
#include "internal/RecordSql.hpp"
#include "internal/EdgeSql.hpp"
#include "geometry/Dependencies.hpp"
#include <limits>

namespace vc::objects {
using namespace detail;
namespace {
using Json = nlohmann::json;

std::optional<std::string> head(sqlite3* db, const std::string& object)
{
    Statement s(db, "SELECT revision_uuid FROM object_heads WHERE object_uuid=?");
    s.bind(1, object);
    return s.row() ? std::optional(s.text(0)) : std::nullopt;
}

std::optional<std::string> optionalUuid(Statement& s, int column)
{
    return s.isNull(column) ? std::nullopt : std::optional(formatUuid(s.blob16(column)));
}

Json metadata(sqlite3* db, const std::string& revision)
{
    // Common case: the materialized head, without any history replay.
    Statement current(db, "SELECT record FROM object_head_records WHERE revision_uuid=?");
    current.bind(1, revision);
    if (current.row()) return Json::parse(current.text(0));
    Statement chain(db,
        "WITH RECURSIVE chain(uuid,parent_uuid,seq,metadata_delta) AS ("
        "SELECT uuid,parent_uuid,seq,metadata_delta FROM object_revisions WHERE uuid=? UNION ALL "
        "SELECT r.uuid,r.parent_uuid,r.seq,r.metadata_delta FROM object_revisions r "
        "JOIN chain c ON r.uuid=c.parent_uuid) SELECT metadata_delta FROM chain ORDER BY seq");
    chain.bind(1, revision);
    auto result = Json::object();
    while (chain.row()) result = result.patch(Json::parse(chain.text(0)));
    return result;
}

std::map<std::string, std::string> members(sqlite3* db, const std::string& revision)
{
    std::map<std::string, std::string> result;
    Statement isHead(db, "SELECT object_uuid FROM object_heads WHERE revision_uuid=?");
    isHead.bind(1, revision);
    if (isHead.row()) {
        Statement s(db, "SELECT edge_uuid,relationship_uuid FROM object_relationship_heads WHERE object_uuid=?");
        s.bind(1, isHead.text(0));
        while (s.row()) result.emplace(s.text(0), s.text(1));
        return result;
    }
    Statement s(db,
        "WITH RECURSIVE chain(uuid,parent_uuid,seq) AS ("
        "SELECT uuid,parent_uuid,seq FROM object_revisions WHERE uuid=? UNION ALL "
        "SELECT r.uuid,r.parent_uuid,r.seq FROM object_revisions r JOIN chain c ON r.uuid=c.parent_uuid) "
        "SELECT d.edge_uuid,d.relationship_uuid FROM chain c JOIN object_relationship_deltas d "
        "ON d.revision_uuid=c.uuid ORDER BY c.seq");
    s.bind(1, revision);
    while (s.row()) {
        if (s.isNull(1)) result.erase(s.text(0));
        else result[s.text(0)] = s.text(1);
    }
    return result;
}

RelationshipRevision relationship(sqlite3* db, const std::string& uuid)
{
    Statement s(db, "SELECT record,a_revision,b_revision,a_geometry,b_geometry FROM relationship_revisions WHERE uuid=?");
    s.bind(1, uuid);
    if (!s.row()) throw std::runtime_error("object history: missing relationship revision");
    RelationshipRevision result;
    result.uuid = uuid;
    const auto record = Json::parse(s.text(0));
    result.record = edgeFromJson(record);
    commonFromJson(record, result.record);
    result.aRevision = s.text(1); result.bRevision = s.text(2);
    result.aGeometryRevision = s.optText(3); result.bGeometryRevision = s.optText(4);
    return result;
}

Json dependencyMetadata(const Json& record, const Anchor& anchor)
{
    auto result = Json::object();
    for (const auto& key : anchor.metadataKeys)
        if (record.at("meta").contains(key)) result[key] = record.at("meta").at(key);
    return Json{{"metadata", result}, {"retired", record.value("retired", false)},
                {"lifecycle_epoch", record.value("lifecycle_epoch", "")}};
}

Json currentRecord(sqlite3* db, const std::string& object)
{
    Statement row(db, std::string("SELECT ") + kObjectCols + " FROM objects WHERE uuid=?");
    row.bindBlob(1, parseUuid(object));
    if (!row.row()) throw std::runtime_error("object history: endpoint missing");
    return objectToJson(readObjectRow(db, row));
}
}

void detail::createHistorySchema(sqlite3* db)
{
    exec(db, R"SQL(
CREATE TABLE object_revisions(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  uuid TEXT NOT NULL UNIQUE,
  object_uuid TEXT NOT NULL,
  parent_uuid TEXT REFERENCES object_revisions(uuid),
  geometry_uuid BLOB REFERENCES geometry_revisions(uuid),
  metadata_delta TEXT NOT NULL CHECK(json_valid(metadata_delta)),
  metadata_complete INTEGER NOT NULL DEFAULT 1,
  published_at_us INTEGER,
  deleted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX object_revision_history ON object_revisions(object_uuid,seq);
CREATE INDEX object_revision_time ON object_revisions(object_uuid,published_at_us DESC,seq DESC);
CREATE INDEX object_publication_time ON object_revisions(published_at_us);
CREATE TABLE object_heads(object_uuid TEXT PRIMARY KEY, revision_uuid TEXT NOT NULL UNIQUE REFERENCES object_revisions(uuid));
CREATE TABLE object_head_records(object_uuid TEXT PRIMARY KEY, revision_uuid TEXT NOT NULL UNIQUE, record TEXT NOT NULL);
CREATE TABLE relationship_revisions(
  uuid TEXT PRIMARY KEY,
  edge_uuid TEXT NOT NULL,
  record TEXT NOT NULL CHECK(json_valid(record)),
  a_revision TEXT NOT NULL REFERENCES object_revisions(uuid) DEFERRABLE INITIALLY DEFERRED,
  b_revision TEXT NOT NULL REFERENCES object_revisions(uuid) DEFERRABLE INITIALLY DEFERRED,
  a_geometry TEXT, b_geometry TEXT
);
CREATE INDEX relationship_history ON relationship_revisions(edge_uuid);
CREATE TABLE object_relationship_deltas(
  revision_uuid TEXT NOT NULL REFERENCES object_revisions(uuid),
  edge_uuid TEXT NOT NULL,
  relationship_uuid TEXT REFERENCES relationship_revisions(uuid),
  PRIMARY KEY(revision_uuid,edge_uuid)
);
CREATE TABLE object_relationship_heads(
  object_uuid TEXT NOT NULL, edge_uuid TEXT NOT NULL,
  relationship_uuid TEXT NOT NULL REFERENCES relationship_revisions(uuid),
  PRIMARY KEY(object_uuid,edge_uuid)
);
CREATE INDEX object_relationship_edges ON object_relationship_heads(edge_uuid);
CREATE TABLE history_dirty_objects(uuid BLOB PRIMARY KEY);
CREATE TABLE history_dirty_edges(uuid BLOB PRIMARY KEY);
CREATE TABLE history_expected(uuid TEXT PRIMARY KEY, revision_uuid TEXT);
CREATE TABLE history_checked(singleton INTEGER PRIMARY KEY CHECK(singleton=1));
CREATE TABLE history_required(uuid TEXT PRIMARY KEY);
CREATE TABLE history_validated_edges(uuid TEXT PRIMARY KEY, a_meta TEXT NOT NULL, b_meta TEXT NOT NULL, recorded_only INTEGER NOT NULL);
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
        for (const auto* table : {"fibers", "patches", "tracks", "object_tags"})
            trigger(table, mark("history_dirty_objects", "SELECT uuid FROM objects WHERE id=" + row + ".object_id"));
        trigger("geometry_heads", mark("history_dirty_objects", "VALUES(" + row + ".object_uuid)"));
        trigger("edges",
            mark("history_dirty_edges", "VALUES(" + row + ".uuid)") +
            mark("history_dirty_objects", "SELECT uuid FROM objects WHERE id IN (" + row + ".a_id," + row + ".b_id)"));
        for (const auto* table : {"edge_tags", "edge_geometry_bindings"})
            trigger(table,
                mark("history_dirty_edges", "SELECT uuid FROM edges WHERE id=" + row + ".edge_id") +
                mark("history_dirty_objects", "SELECT o.uuid FROM objects o JOIN edges e "
                "ON o.id=e.a_id OR o.id=e.b_id WHERE e.id=" + row + ".edge_id"));
    }
    exec(db, "INSERT INTO history_dirty_objects SELECT uuid FROM objects;"
             "INSERT INTO history_dirty_edges SELECT uuid FROM edges;");
}

void detail::expectHistory(sqlite3* db, const RevisionSelection& expected)
{
    activeStatementPools.at(db)->checkedAction = true;
    exec(db, "INSERT INTO history_checked VALUES(1)");
    for (const auto& [object, revision] : expected) {
        const auto uuid = formatUuid(parseUuid(object));
        const auto wanted = revision ? std::optional(formatUuid(parseUuid(*revision))) : std::nullopt;
        if (head(db, uuid) != wanted)
            throw std::runtime_error("object history: expected revision conflict for " + uuid);
        Statement s(db, "INSERT INTO history_expected VALUES(?,?)");
        s.bind(1, uuid);
        if (wanted) s.bind(2, *wanted); else s.bindNull(2);
        s.stepDone();
    }
}

void detail::historyConflict(sqlite3* db, const std::string& message)
{
    if (const auto pool = activeStatementPools.find(db); pool != activeStatementPools.end() && pool->second->checkedAction)
        pool->second->actionConflict = true;
    throw std::runtime_error(message);
}

void detail::recordRelationshipValidation(sqlite3* db, const EdgeRecord& edge, bool recordedOnly)
{
    Statement s(db, "INSERT INTO history_validated_edges VALUES(?,?,?,?) ON CONFLICT(uuid) "
        "DO UPDATE SET a_meta=excluded.a_meta,b_meta=excluded.b_meta,recorded_only=excluded.recorded_only");
    s.bind(1, edge.uuid);
    s.bind(2, dependencyMetadata(currentRecord(db, edge.aUuid), edge.aAnchor).dump());
    s.bind(3, dependencyMetadata(currentRecord(db, edge.bUuid), edge.bAnchor).dump());
    s.bind(4, int64_t(recordedOnly));
    s.stepDone();
    for (const auto& object : {edge.aUuid, edge.bUuid}) {
        Statement required(db, "INSERT OR IGNORE INTO history_required VALUES(?)");
        required.bind(1, object); required.stepDone();
        if (!recordedOnly) {
            Statement dirty(db, "INSERT OR IGNORE INTO history_dirty_objects VALUES(?)");
            dirty.bindBlob(1, parseUuid(object)); dirty.stepDone();
        }
    }
    if (!recordedOnly) {
        const auto a = currentRecord(db, edge.aUuid), b = currentRecord(db, edge.bUuid);
        bindLifecycle(db, edge, a.value("lifecycle_epoch", ""), b.value("lifecycle_epoch", ""));
        Statement dirty(db, "INSERT OR IGNORE INTO history_dirty_edges VALUES(?)");
        dirty.bindBlob(1, parseUuid(edge.uuid)); dirty.stepDone();
    }
}

void detail::publishHistory(sqlite3* db, bool migration, bool importedBindings)
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
            Statement expected(db, "SELECT revision_uuid FROM history_expected WHERE uuid=?");
            expected.bind(1, required.text(0));
            if (!expected.row() || expected.optText(0) != head(db, required.text(0)))
                throw std::runtime_error("object history: missing expected validation endpoint " + required.text(0));
        }
    }
    while (dirty.row()) {
        const auto object = formatUuid(dirty.blob16(0));
        Pending p;
        p.parent = head(db, object);
        if (strict) {
            Statement expected(db, "SELECT revision_uuid FROM history_expected WHERE uuid=?");
            expected.bind(1, object);
            if (!expected.row() || expected.optText(0) != p.parent)
                throw std::runtime_error("object history: missing or conflicting expected revision for " + object);
        }
        Statement row(db, std::string("SELECT ") + kObjectCols + " FROM objects WHERE uuid=?");
        row.bindBlob(1, parseUuid(object));
        p.exists = row.row();
        if (p.exists) {
            p.record = objectToJson(readObjectRow(db, row));
            p.geometry = geometryHead(db, object);
            p.delta = Json::diff(p.parent ? metadata(db, *p.parent) : Json::object(), p.record);
            p.changed = !p.parent || !p.delta.empty();
            if (p.parent) {
                Statement previous(db, "SELECT geometry_uuid FROM object_revisions WHERE uuid=?");
                previous.bind(1, *p.parent); previous.row();
                p.changed |= optionalUuid(previous, 0) != p.geometry;
            }
        }
        objects.emplace(object, std::move(p));
    }
    // Ordinary metadata is independent unless the relationship declares a
    // dependency. Check the final state, not the order of earlier API calls.
    for (const auto& [object, p] : objects) {
        if (!p.exists || !p.parent || p.delta.empty()) continue;
        const auto before = metadata(db, *p.parent);
        const bool lifecycleChanged = before.value("lifecycle_epoch", "") != p.record.value("lifecycle_epoch", "");
        if (!lifecycleChanged && before.at("meta") == p.record.at("meta")) continue;
        Statement incident(db, std::string("SELECT ") + kEdgeCols +
            " FROM edges WHERE a_id=(SELECT id FROM objects WHERE uuid=?) OR b_id=(SELECT id FROM objects WHERE uuid=?)");
        incident.bindBlob(1, parseUuid(object)); incident.bindBlob(2, parseUuid(object));
        while (incident.row()) {
            const auto edge = readEdgeRow(db, incident);
            bool affected = false;
            for (const auto& [endpoint, anchor] : {std::pair{edge.aUuid, edge.aAnchor}, std::pair{edge.bUuid, edge.bAnchor}})
                affected |= endpoint == object && dependencyMetadata(before, anchor) != dependencyMetadata(p.record, anchor);
            if (!affected) continue;
            // Synchronization imports a previously published lifecycle decision.
            // Preserve recorded pins; imported edge bindings below carry any
            // separate revalidation. It does not import the peer's timeline.
            if (importedBindings && lifecycleChanged) recordRelationshipValidation(db, edge, true);
            Statement validated(db, "SELECT a_meta,b_meta,recorded_only FROM history_validated_edges WHERE uuid=?");
            validated.bind(1, edge.uuid);
            if (!validated.row() ||
                Json::parse(validated.text(0)) != dependencyMetadata(currentRecord(db, edge.aUuid), edge.aAnchor) ||
                Json::parse(validated.text(1)) != dependencyMetadata(currentRecord(db, edge.bUuid), edge.bAnchor))
                throw std::runtime_error("object history: metadata/lifecycle edit requires explicit retain/update/remove or keep_recorded for " + edge.uuid);
            if (validated.i64(2)) continue;
            // A validated metadata dependency changes the shared binding even
            // when neither the edge JSON nor its geometry bindings changed.
            Statement mark(db, "INSERT OR IGNORE INTO history_dirty_edges VALUES(?)");
            mark.bindBlob(1, parseUuid(edge.uuid)); mark.stepDone();
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
    Statement dirtyEdges(db, "SELECT uuid FROM history_dirty_edges");
    while (dirtyEdges.row()) {
        ChangedEdge e;
        e.uuid = formatUuid(dirtyEdges.blob16(0));
        std::optional<RelationshipRevision> previous;
        Statement old(db, "SELECT object_uuid,relationship_uuid FROM object_relationship_heads WHERE edge_uuid=?");
        old.bind(1, e.uuid);
        while (old.row()) {
            e.owners.insert(old.text(0));
            if (!previous) previous = relationship(db, old.text(1));
        }
        Statement row(db, std::string("SELECT ") + kEdgeCols + " FROM edges WHERE uuid=?");
        row.bindBlob(1, parseUuid(e.uuid));
        if (row.row()) {
            e.record = readEdgeRow(db, row);
            e.owners.insert(e.record->aUuid); e.owners.insert(e.record->bUuid);
            Statement bindings(db, "SELECT a_revision,b_revision FROM edge_geometry_bindings WHERE edge_id=?");
            bindings.bind(1, *edgeIdByUuid(db, parseUuid(e.uuid)));
            if (bindings.row()) {
                e.aGeometry = optionalUuid(bindings, 0); e.bGeometry = optionalUuid(bindings, 1);
            }
            Statement validated(db, "SELECT recorded_only FROM history_validated_edges WHERE uuid=?");
            validated.bind(1, e.uuid);
            const bool revalidated = validated.row() && !validated.i64(0);
            const auto importedEpochs = lifecycleBinding(db, *e.record);
            if (previous && !revalidated) {
                if (e.aGeometry == previous->aGeometryRevision && (!importedBindings ||
                    metadata(db, previous->aRevision).value("lifecycle_epoch", "") == importedEpochs.first))
                    e.aPin = previous->aRevision;
                if (e.bGeometry == previous->bGeometryRevision && (!importedBindings ||
                    metadata(db, previous->bRevision).value("lifecycle_epoch", "") == importedEpochs.second))
                    e.bPin = previous->bRevision;
            }
            if (previous && (!importedBindings || (e.aPin && e.bPin)) && edgeToJson(previous->record, nullptr, false) == edgeToJson(*e.record, nullptr, false) &&
                e.aGeometry == previous->aGeometryRevision && e.bGeometry == previous->bGeometryRevision &&
                dependencyMetadata(metadata(db, previous->aRevision), e.record->aAnchor) ==
                    dependencyMetadata(currentRecord(db, e.record->aUuid), e.record->aAnchor) &&
                dependencyMetadata(metadata(db, previous->bRevision), e.record->bAnchor) ==
                    dependencyMetadata(currentRecord(db, e.record->bUuid), e.record->bAnchor)) continue;
            e.version = util::newUuid();
        } else if (!previous) continue;
        for (const auto& owner : e.owners) {
            // An existing endpoint is always dirty through the SQL triggers.
            auto found = objects.find(owner);
            if (found == objects.end()) {
                Pending p;
                p.parent = head(db, owner);
                if (strict) {
                    Statement expected(db, "SELECT revision_uuid FROM history_expected WHERE uuid=?");
                    expected.bind(1, owner);
                    if (!expected.row() || expected.optText(0) != p.parent)
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
    // Allocate all identities first: shared records may pin both revisions
    // published by this action, including cycles and self relationships.
    for (auto& [object, p] : objects) {
        if (!p.exists) {
            if (p.parent) {
                // Record a physical deletion without changing existing deletion
                // eligibility. This prevents time-based reads resurrecting it.
                Statement tombstone(db, "INSERT INTO object_revisions(uuid,object_uuid,parent_uuid,"
                    "metadata_delta,published_at_us,deleted) VALUES(?,?,?,'[]',?,1)");
                tombstone.bind(1, util::newUuid()); tombstone.bind(2, object);
                tombstone.bind(3, *p.parent); tombstone.bind(4, publishedAtUs); tombstone.stepDone();
            }
            for (const auto* table : {"object_heads", "object_head_records", "object_relationship_heads"}) {
                Statement erase(db, std::string("DELETE FROM ") + table + " WHERE object_uuid=?");
                erase.bind(1, object); erase.stepDone();
            }
            continue;
        }
        if (!p.changed) continue;
        p.uuid = util::newUuid();
        Statement insert(db, "INSERT INTO object_revisions(uuid,object_uuid,parent_uuid,geometry_uuid,"
            "metadata_delta,metadata_complete,published_at_us) VALUES(?,?,?,?,?,?,?)");
        insert.bind(1, p.uuid); insert.bind(2, object);
        if (p.parent) insert.bind(3, *p.parent); else insert.bindNull(3);
        if (p.geometry) insert.bindBlob(4, parseUuid(*p.geometry)); else insert.bindNull(4);
        insert.bind(5, p.delta.dump()); insert.bind(6, int64_t(!migration));
        insert.bind(7, publishedAtUs); insert.stepDone();
        Statement setHead(db, "INSERT INTO object_heads VALUES(?,?) ON CONFLICT(object_uuid) DO UPDATE SET revision_uuid=excluded.revision_uuid");
        setHead.bind(1, object); setHead.bind(2, p.uuid); setHead.stepDone();
        Statement cache(db, "INSERT INTO object_head_records VALUES(?,?,?) ON CONFLICT(object_uuid) "
            "DO UPDATE SET revision_uuid=excluded.revision_uuid,record=excluded.record");
        cache.bind(1, object); cache.bind(2, p.uuid); cache.bind(3, p.record.dump()); cache.stepDone();
    }
    auto pin = [&](const std::string& object, const std::optional<std::string>& geometry,
                   const std::optional<std::string>& epoch = std::nullopt) {
        Statement s(db, "SELECT uuid FROM object_revisions WHERE object_uuid=? AND geometry_uuid IS ? ORDER BY seq DESC");
        s.bind(1, object);
        if (geometry) s.bindBlob(2, parseUuid(*geometry)); else s.bindNull(2);
        while (s.row())
            if (!epoch || metadata(db, s.text(0)).value("lifecycle_epoch", "") == *epoch) return s.text(0);
        if (!migration && !importedBindings)
            throw std::runtime_error("object history: relationship binding has no object revision");
        // Old stores and current-state sync imports can supply older geometry
        // without its object history. Preserve that binding with an explicitly
        // incomplete, untimed baseline, never fabricated historical metadata.
        const auto uuid = util::newUuid();
        Statement baseline(db, "INSERT INTO object_revisions(uuid,object_uuid,geometry_uuid,metadata_delta,metadata_complete) VALUES(?,?,?,?,0)");
        baseline.bind(1, uuid); baseline.bind(2, object);
        if (geometry) baseline.bindBlob(3, parseUuid(*geometry)); else baseline.bindNull(3);
        auto record = currentRecord(db, object);
        if (epoch) { record["lifecycle_epoch"] = *epoch; record["retired"] = false; }
        baseline.bind(4, Json::diff(Json::object(), record).dump()); baseline.stepDone();
        return uuid;
    };
    for (const auto& edge : edges) {
        if (edge.record) {
            const auto epochs = lifecycleBinding(db, *edge.record);
            const auto aPin = edge.aPin ? *edge.aPin : pin(edge.record->aUuid, edge.aGeometry,
                importedBindings ? std::optional(epochs.first) : std::nullopt);
            const auto bPin = edge.bPin ? *edge.bPin : pin(edge.record->bUuid, edge.bGeometry,
                importedBindings ? std::optional(epochs.second) : std::nullopt);
            Statement s(db, "INSERT INTO relationship_revisions VALUES(?,?,?,?,?,?,?)");
            s.bind(1, edge.version); s.bind(2, edge.uuid); s.bind(3, edgeToJson(*edge.record, nullptr, false).dump());
            s.bind(4, aPin); s.bind(5, bPin);
            if (edge.aGeometry) s.bind(6, *edge.aGeometry); else s.bindNull(6);
            if (edge.bGeometry) s.bind(7, *edge.bGeometry); else s.bindNull(7);
            s.stepDone();
            const auto a = metadata(db, aPin);
            const auto b = metadata(db, bPin);
            bindLifecycle(db, *edge.record, a.value("lifecycle_epoch", ""), b.value("lifecycle_epoch", ""));
        }
        for (const auto& owner : edge.owners) {
            const auto& p = objects.at(owner);
            if (!p.exists) continue;
            Statement delta(db, "INSERT INTO object_relationship_deltas VALUES(?,?,?)");
            delta.bind(1, p.uuid); delta.bind(2, edge.uuid);
            if (edge.record) delta.bind(3, edge.version); else delta.bindNull(3);
            delta.stepDone();
            Statement erase(db, "DELETE FROM object_relationship_heads WHERE object_uuid=? AND edge_uuid=?");
            erase.bind(1, owner); erase.bind(2, edge.uuid); erase.stepDone();
            if (edge.record) {
                Statement set(db, "INSERT INTO object_relationship_heads VALUES(?,?,?)");
                set.bind(1, owner); set.bind(2, edge.uuid); set.bind(3, edge.version); set.stepDone();
            }
        }
    }
    exec(db, "DELETE FROM history_dirty_objects; DELETE FROM history_dirty_edges;"
             "DELETE FROM history_expected; DELETE FROM history_checked; DELETE FROM history_validated_edges;"
             "DELETE FROM history_required;");
}

std::optional<ObjectRevision> ObjectStore::objectRevision(const Txn& txn, const std::string& uuid) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto canonical = formatUuid(parseUuid(uuid));
    Statement s(db, "SELECT object_uuid,parent_uuid,geometry_uuid,seq,metadata_complete,published_at_us,deleted "
        "FROM object_revisions WHERE uuid=?");
    s.bind(1, canonical);
    if (!s.row()) return std::nullopt;
    ObjectRevision result;
    result.uuid = canonical; result.objectUuid = s.text(0); result.parentUuid = s.optText(1);
    result.geometryRevision = optionalUuid(s, 2); result.sequence = s.i64(3);
    result.metadataComplete = s.i64(4) != 0;
    if (!s.isNull(5)) result.publishedAtUs = s.i64(5);
    result.deleted = s.i64(6) != 0;
    const auto record = metadata(db, canonical);
    result.record = objectFromJson(record); commonFromJson(record, result.record);
    return result;
}

std::optional<ObjectRevision> ObjectStore::currentObjectRevision(const Txn& txn, const std::string& uuid) const
{
    const auto revision = head(static_cast<sqlite3*>(connection(txn)), formatUuid(parseUuid(uuid)));
    return revision ? objectRevision(txn, *revision) : std::nullopt;
}

std::vector<ObjectRevision> ObjectStore::objectHistory(const Txn& txn, const std::string& object) const
{
    Statement s(static_cast<sqlite3*>(connection(txn)), "SELECT uuid FROM object_revisions WHERE object_uuid=? ORDER BY seq");
    s.bind(1, formatUuid(parseUuid(object)));
    std::vector<ObjectRevision> result;
    while (s.row()) result.push_back(*objectRevision(txn, s.text(0)));
    return result;
}

RevisionSelection ObjectStore::revisionSelectionAt(const Txn& txn, int64_t publishedAtUs, bool includeRetired) const
{
    if (publishedAtUs < 0) throw std::invalid_argument("object history: publication time must be nonnegative");
    Statement s(static_cast<sqlite3*>(connection(txn)),
        "SELECT object_uuid,uuid FROM (SELECT object_uuid,uuid,deleted,"
        "row_number() OVER (PARTITION BY object_uuid ORDER BY published_at_us DESC,seq DESC) AS position "
        "FROM object_revisions WHERE published_at_us<=?) WHERE position=1 AND deleted=0 ORDER BY object_uuid");
    s.bind(1, publishedAtUs);
    RevisionSelection result;
    while (s.row())
        if (includeRetired || !objectRevision(txn, s.text(1))->record.retired)
            result.emplace(s.text(0), s.text(1));
    return result;
}

std::vector<RelationshipRevision> ObjectStore::recordedRelationships(const Txn& txn, const std::string& uuid) const
{
    const auto revision = objectRevision(txn, uuid);
    if (!revision) throw std::invalid_argument("object history: unknown object revision");
    auto* db = static_cast<sqlite3*>(connection(txn));
    std::vector<RelationshipRevision> result;
    if (revision->deleted) return result;
    for (const auto& [edge, version] : members(db, revision->uuid))
        result.push_back(relationship(db, version));
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
            if (found == revisions.end() || found->second.record.retired) return false;
            if (!anchor.metadataKeys.empty() && !objectRevision(txn, pinned)->metadataComplete)
                return false; // imported geometry cannot prove lost metadata dependencies
            const auto& membership = memberships.at(object);
            const auto member = membership.find(edge.record.uuid);
            return member != membership.end() && member->second == version &&
                dependencyMetadata(metadata(db, pinned), anchor) == dependencyMetadata(objectToJson(found->second.record), anchor) &&
                !endpointChanged(*this, txn, anchor, geometry, found->second.geometryRevision);
        };
        const bool applicable = endpoint(edge.record.aUuid, edge.record.aAnchor, edge.aGeometryRevision, edge.aRevision) &&
                                endpoint(edge.record.bUuid, edge.record.bAnchor, edge.bGeometryRevision, edge.bRevision);
        result.push_back({std::move(edge), applicable});
    }
    return result;
}

void ObjectStore::retainRelationship(WriteTxn& txn, const std::string& uuid)
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto edge = edgeByUuid(txn, uuid);
    if (!edge) throw std::invalid_argument("object history: unknown relationship");
    Savepoint savepoint(db);
    EdgeGeometryRepair repair;
    repair.edgeUuid = edge->uuid; repair.expectedGeneration = edge->generation;
    repair.aRevision = geometryHead(db, edge->aUuid); repair.bRevision = geometryHead(db, edge->bUuid);
    repair.aAnchor = edge->aAnchor; repair.bAnchor = edge->bAnchor;
    repairEdgeGeometry(txn, repair);
    recordRelationshipValidation(db, *edge);
    savepoint.release();
}

void ObjectStore::keepRecordedRelationship(WriteTxn& txn, const std::string& uuid)
{
    DependencyScope dependencies;
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto edge = edgeByUuid(txn, uuid);
    if (!edge) throw std::invalid_argument("object history: unknown relationship");
    const auto binding=edgeGeometryState(txn,uuid);
    // Preservation validates the immutable reference, not a substitution of
    // today's endpoint. Structural deletion may remove that point from the
    // current sequence while the recorded historical point remains resolvable.
    for (const auto& [revision,anchor] : {std::pair{binding.aRevision,edge->aAnchor},std::pair{binding.bRevision,edge->bAnchor}}) {
        if (revision) {
            const auto object=revision==binding.aRevision ? edge->aUuid : edge->bUuid;
            const auto published=currentObjectRevision(txn,object);
            if (published && published->geometryRevision==revision && geometryHead(db,object)==revision &&
                published->record.lifecycleEpoch!=currentRecord(db,object).value("lifecycle_epoch","")) continue;
            const auto geometry=geometryRevision(txn,*revision);
            if (!geometry) throw std::runtime_error("recorded relationship geometry is missing");
            validateDependencies(anchor,dependencyGeometry(*this,txn,*geometry));
        }
    }
    recordRelationshipValidation(db, *edge, true);
}
} // namespace vc::objects
