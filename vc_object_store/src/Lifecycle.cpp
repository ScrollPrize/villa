#include "internal/RecordSql.hpp"
#include "internal/History.hpp"
#include "internal/SqlValues.hpp"
#include "geometry/GeometryDelta.hpp"
#include "internal/ImmutableImport.hpp"
#include "internal/ExternalEndpoint.hpp"

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
)SQL");
}

void detail::readLifecycle(sqlite3* db, int64_t id, ObjectRecord& record)
{
    Statement s(db, "SELECT retired,epoch FROM object_lifecycle WHERE object_id=?", id);
    if (s.row()) { record.retired = s.i64(0); record.lifecycleEpoch = s.text(1); }
}

void detail::writeLifecycle(sqlite3* db, int64_t id, const ObjectRecord& record)
{
    beforeCurrentObjectEdit(db,record.uuid);
    execute(db, "INSERT INTO object_lifecycle VALUES(?,?,?) ON CONFLICT(object_id) "
        "DO UPDATE SET retired=excluded.retired,epoch=excluded.epoch", id, int64_t(record.retired),
        record.lifecycleEpoch);
}

std::string detail::activeObjectSql(const std::string& id)
{
    return "NOT EXISTS(SELECT 1 FROM objects owned LEFT JOIN object_lifecycle lc ON lc.object_id=owned.id "
        "WHERE owned.id="+id+" AND (owned.deleted=1 OR lc.retired=1))";
}

std::string detail::activeEdgeSql()
{
    return "edges.deleted=0 AND NOT EXISTS(SELECT 1 FROM objects owned LEFT JOIN object_lifecycle lc ON lc.object_id=owned.id "
        "WHERE (owned.uuid=edges.a_uuid OR owned.uuid=edges.b_uuid) AND (owned.deleted=1 OR lc.retired=1))";
}

namespace {
ObjectRecord endpointLifecycle(sqlite3* db,const std::string& uuid)
{
    Statement row(db,"SELECT id,deleted FROM objects WHERE uuid=?",parseUuid(uuid));
    if(!row.row())return externalEndpointRecord(db,uuid);
    ObjectRecord record;readLifecycle(db,row.i64(0),record);
    record.retired=record.retired||row.i64(1)!=0;return record;
}
}

std::pair<std::string, std::string> detail::lifecycleBinding(sqlite3* db, const EdgeRecord& edge)
{
    const auto a=endpointLifecycle(db,edge.aUuid),b=endpointLifecycle(db,edge.bUuid);
    return {a.lifecycleEpoch,b.lifecycleEpoch};
}

bool detail::lifecycleBindingMatches(sqlite3* db, const EdgeRecord& edge)
{
    const auto a=endpointLifecycle(db,edge.aUuid),b=endpointLifecycle(db,edge.bUuid);
    return !a.retired && !b.retired;
}

void detail::bindLifecycle(sqlite3* db, const EdgeRecord& edge, const std::string& aEpoch, const std::string& bEpoch)
{
    // Wire/import compatibility check, not persistent per-edge state.
    if(lifecycleBinding(db,edge)!=std::pair{aEpoch,bEpoch})
        throw std::runtime_error("relationship lifecycle does not match current endpoints");
}

namespace {
void requireCheckedLifecycle(sqlite3* db)
{
    if (!historyEnabled(db)) return;
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
    // Retiring is a native publication, not an independent version namespace.
    // Removing current relationships also prevents restore from silently
    // reviving observations whose endpoint was deliberately retired.
    for (const auto& edge : edgesOf(txn,uuid,true)) removeEdge(txn,edge.uuid);
    record->retired = true; record->lifecycleEpoch = std::to_string(writeGeneration(db));
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
    if (!historyEnabled(db)) {
        if (sourceRevision) requireHistory(db);
        Savepoint savepoint(db);
        auto restored = *current;
        restored.retired = false; restored.lifecycleEpoch = std::to_string(writeGeneration(db));
        const auto id = *objectIdByUuid(db, parseUuid(uuid));
        writeLifecycle(db, id, restored);
        stampContentChange(db, author(), hostname(), "objects", id, parseUuid(uuid), "restore");
        savepoint.release();
        return;
    }
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
    // An explicit historical restore can replace active geometry/metadata.
    // Its old observations remain in history, not attached to the new current
    // state without an explicit revalidation decision.
    for(const auto& edge:edgesOf(txn,uuid,true))removeEdge(txn,edge.uuid);
    const auto id = *objectIdByUuid(db, parseUuid(uuid));
    const auto oldGeometry = geometryHead(db, uuid);
    if (oldGeometry != source->geometryRevision) {
        // Reuse immutable payloads and manifests. Only a content restoration
        // needs to rebuild the current spatial/materialized views.
        clearGeometryMaterialization(db, uuid);
        if (source->geometryRevision) {
            // Install the selected managed descriptor before restoring its
            // metadata, just as an existing geometry head is preserved by
            // upsert. This also supports a metadata-only current object.
            materializeGeometry(db,uuid,*source->geometryRevision);
        } else {
            execute(db,"UPDATE objects SET geometry_manifest=NULL WHERE uuid=?",parseUuid(uuid));
        }
    }
    auto restored = source->record;
    // upsert preserves identity/creation stamps and resets approval normally.
    upsertObjectImpl(db, author(), hostname(), restored);
    restored.retired = false; restored.lifecycleEpoch = std::to_string(writeGeneration(db));
    writeLifecycle(db, id, restored);
    if (oldGeometry != source->geometryRevision) refreshGeometryBounds(db, uuid);
    appendJournal(db, author(), hostname(), parseUuid(uuid), objectByUuid(txn, uuid)->generation, "restore");
    savepoint.release();
}
} // namespace vc::objects
