#include "internal/RecordSql.hpp"
#include "internal/History.hpp"
#include "geometry/GeometryDelta.hpp"

namespace vc::objects {
using namespace detail;

void detail::createLifecycleSchema(sqlite3* db)
{
    exec(db, R"SQL(
CREATE TABLE object_lifecycle(
  object_id INTEGER PRIMARY KEY REFERENCES objects(id) ON DELETE CASCADE,
  retired INTEGER NOT NULL CHECK(retired IN (0,1)), epoch TEXT NOT NULL
);
CREATE INDEX retired_objects ON object_lifecycle(retired,object_id);
CREATE TABLE edge_lifecycle_bindings(
  edge_id INTEGER PRIMARY KEY REFERENCES edges(id) ON DELETE CASCADE,
  a_epoch TEXT NOT NULL, b_epoch TEXT NOT NULL
);
)SQL");
}

void detail::readLifecycle(sqlite3* db, int64_t id, ObjectRecord& record)
{
    Statement s(db, "SELECT retired,epoch FROM object_lifecycle WHERE object_id=?");
    s.bind(1, id);
    if (s.row()) { record.retired = s.i64(0); record.lifecycleEpoch = s.text(1); }
}

void detail::writeLifecycle(sqlite3* db, int64_t id, const ObjectRecord& record)
{
    Statement s(db, "INSERT INTO object_lifecycle VALUES(?,?,?) ON CONFLICT(object_id) "
        "DO UPDATE SET retired=excluded.retired,epoch=excluded.epoch");
    s.bind(1, id); s.bind(2, int64_t(record.retired)); s.bind(3, record.lifecycleEpoch); s.stepDone();
}

std::string detail::activeObjectSql(const std::string& id)
{
    return "NOT EXISTS(SELECT 1 FROM object_lifecycle lc WHERE lc.object_id=" + id + " AND lc.retired=1)";
}

std::string detail::activeEdgeSql()
{
    return activeObjectSql("edges.a_id") + " AND " + activeObjectSql("edges.b_id") +
        " AND coalesce((SELECT a_epoch FROM edge_lifecycle_bindings WHERE edge_id=edges.id),'')="
        "coalesce((SELECT epoch FROM object_lifecycle WHERE object_id=edges.a_id),'')"
        " AND coalesce((SELECT b_epoch FROM edge_lifecycle_bindings WHERE edge_id=edges.id),'')="
        "coalesce((SELECT epoch FROM object_lifecycle WHERE object_id=edges.b_id),'')";
}

std::pair<std::string, std::string> detail::lifecycleBinding(sqlite3* db, const EdgeRecord& edge)
{
    Statement s(db, "SELECT a_epoch,b_epoch FROM edge_lifecycle_bindings WHERE edge_id=(SELECT id FROM edges WHERE uuid=?)");
    s.bindBlob(1, parseUuid(edge.uuid));
    return s.row() ? std::pair{s.text(0), s.text(1)} : std::pair<std::string, std::string>{};
}

bool detail::lifecycleBindingMatches(sqlite3* db, const EdgeRecord& edge)
{
    ObjectRecord a, b;
    readLifecycle(db, *objectIdByUuid(db, parseUuid(edge.aUuid)), a);
    readLifecycle(db, *objectIdByUuid(db, parseUuid(edge.bUuid)), b);
    return lifecycleBinding(db, edge) == std::pair{a.lifecycleEpoch, b.lifecycleEpoch};
}

void detail::bindLifecycle(sqlite3* db, const EdgeRecord& edge, const std::string& aEpoch, const std::string& bEpoch)
{
    Statement s(db, "INSERT INTO edge_lifecycle_bindings VALUES(?,?,?) ON CONFLICT(edge_id) "
        "DO UPDATE SET a_epoch=excluded.a_epoch,b_epoch=excluded.b_epoch");
    s.bind(1, *edgeIdByUuid(db, parseUuid(edge.uuid))); s.bind(2, aEpoch); s.bind(3, bEpoch); s.stepDone();
}

namespace {
void requireCheckedLifecycle(sqlite3* db)
{
    Statement checked(db, "SELECT 1 FROM history_checked");
    if (!checked.row()) throw std::runtime_error("object lifecycle requires write(expected_revisions)");
}
}

void ObjectStore::retireObject(WriteTxn& txn, const std::string& uuid)
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    requireCheckedLifecycle(db);
    auto record = objectByUuid(txn, uuid);
    if (!record) throw std::invalid_argument("object lifecycle: unknown object");
    if (record->retired) return;
    Savepoint savepoint(db);
    record->retired = true; record->lifecycleEpoch = util::newUuid();
    const auto id = *objectIdByUuid(db, parseUuid(uuid));
    writeLifecycle(db, id, *record);
    stampContentChange(db, author(), hostname(), "objects", id, parseUuid(uuid), "retire");
    savepoint.release();
}

void ObjectStore::restoreObject(WriteTxn& txn, const std::string& uuid,
                               const std::optional<std::string>& sourceRevision)
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    requireCheckedLifecycle(db);
    const auto current = objectByUuid(txn, uuid);
    if (!current) throw std::invalid_argument("object lifecycle: unknown object");
    if (!current->retired && !sourceRevision) return;
    std::optional<ObjectRevision> source;
    if (sourceRevision) source = objectRevision(txn, *sourceRevision);
    else {
        auto history = objectHistory(txn, uuid);
        for (auto it = history.rbegin(); it != history.rend(); ++it)
            if (it->publishedAtUs && !it->deleted && !it->record.retired) { source = *it; break; }
    }
    if (!source || source->objectUuid != current->uuid || source->deleted || source->record.retired)
        throw std::invalid_argument("object lifecycle: restoration requires an active revision of this object");
    Savepoint savepoint(db);
    const auto id = *objectIdByUuid(db, parseUuid(uuid));
    const auto oldGeometry = geometryHead(db, uuid);
    if (oldGeometry != source->geometryRevision) {
        // Reuse immutable payloads and manifests. Only a content restoration
        // needs to rebuild the current spatial/materialized views.
        clearGeometryMaterialization(db, uuid);
        if (source->geometryRevision) {
            Statement s(db, "INSERT INTO geometry_heads VALUES(?,?) ON CONFLICT(object_uuid) "
                "DO UPDATE SET revision_uuid=excluded.revision_uuid");
            s.bindBlob(1, parseUuid(uuid)); s.bindBlob(2, parseUuid(*source->geometryRevision)); s.stepDone();
        } else {
            Statement s(db, "DELETE FROM geometry_heads WHERE object_uuid=?");
            s.bindBlob(1, parseUuid(uuid)); s.stepDone();
        }
    }
    auto restored = source->record;
    // upsert preserves identity/creation stamps and resets approval normally.
    upsertObjectImpl(db, author(), hostname(), restored);
    restored.retired = false; restored.lifecycleEpoch = util::newUuid();
    writeLifecycle(db, id, restored);
    if (oldGeometry != source->geometryRevision && source->geometryRevision) {
        const auto geometry = geometryRevision(txn, *source->geometryRevision);
        Statement s(db, "UPDATE objects SET storage_kind='sqlite',storage_ref=?,data_checksum=?,data_generation=? WHERE id=?");
        s.bind(1, source->record.storageRef); s.bind(2, geometry->contentHash);
        s.bind(3, geometry->objectGeneration); s.bind(4, id); s.stepDone();
    }
    if (oldGeometry != source->geometryRevision) refreshGeometryBounds(db, uuid);
    appendJournal(db, author(), hostname(), parseUuid(uuid), objectByUuid(txn, uuid)->generation, "restore");
    savepoint.release();
}
} // namespace vc::objects
