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
    if(id<0){
        Statement key(db,"SELECT uuid FROM external_object_keys WHERE id=?",id);
        if(!key.row())throw std::runtime_error("external endpoint key missing");
        const auto resolved=externalEndpointRecord(db,formatUuid(key.blob16(0)));
        record.retired=resolved.retired;record.lifecycleEpoch=resolved.lifecycleEpoch;return;
    }
    Statement s(db, "SELECT retired,epoch FROM object_lifecycle WHERE object_id=?", id);
    if (s.row()) { record.retired = s.i64(0); record.lifecycleEpoch = s.text(1); }
}

void detail::writeLifecycle(sqlite3* db, int64_t id, const ObjectRecord& record)
{
    execute(db, "INSERT INTO object_lifecycle VALUES(?,?,?) ON CONFLICT(object_id) "
        "DO UPDATE SET retired=excluded.retired,epoch=excluded.epoch", id, int64_t(record.retired),
        record.lifecycleEpoch);
}

std::string detail::activeObjectSql(const std::string& id)
{
    return "NOT EXISTS(SELECT 1 FROM object_lifecycle lc WHERE lc.object_id=" + id + " AND lc.retired=1)";
}

std::string detail::activeEdgeSql()
{
    return activeObjectSql("edges.a_id") + " AND " + activeObjectSql("edges.b_id");
}

std::pair<std::string, std::string> detail::lifecycleBinding(sqlite3* db, const EdgeRecord& edge)
{
    ObjectRecord a,b;
    const auto aId=endpointIdByUuid(db,parseUuid(edge.aUuid)),bId=endpointIdByUuid(db,parseUuid(edge.bUuid));
    if(!aId||!bId)throw std::runtime_error("relationship endpoint missing");
    readLifecycle(db,*aId,a);readLifecycle(db,*bId,b);
    return {a.lifecycleEpoch,b.lifecycleEpoch};
}

bool detail::lifecycleBindingMatches(sqlite3* db, const EdgeRecord& edge)
{
    ObjectRecord a, b;
    const auto aId=endpointIdByUuid(db,parseUuid(edge.aUuid)),bId=endpointIdByUuid(db,parseUuid(edge.bUuid));
    if(!aId||!bId)return false;
    readLifecycle(db,*aId,a);readLifecycle(db,*bId,b);
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
            execute(db, "INSERT INTO geometry_heads VALUES(?,?) ON CONFLICT(object_uuid) "
                "DO UPDATE SET revision_uuid=excluded.revision_uuid", parseUuid(uuid),
                parseUuid(*source->geometryRevision));
        } else {
            execute(db, "DELETE FROM geometry_heads WHERE object_uuid=?", parseUuid(uuid));
        }
    }
    auto restored = source->record;
    // upsert preserves identity/creation stamps and resets approval normally.
    upsertObjectImpl(db, author(), hostname(), restored);
    restored.retired = false; restored.lifecycleEpoch = std::to_string(writeGeneration(db));
    writeLifecycle(db, id, restored);
    if (oldGeometry != source->geometryRevision && source->geometryRevision) {
        const auto geometry = geometryRevision(txn, *source->geometryRevision);
        execute(db, "UPDATE objects SET storage_kind='sqlite',storage_ref=?,data_checksum=?,data_generation=? WHERE id=?",
            source->record.storageRef, geometry->contentHash, geometry->objectGeneration, id);
    }
    if (oldGeometry != source->geometryRevision) refreshGeometryBounds(db, uuid);
    appendJournal(db, author(), hostname(), parseUuid(uuid), objectByUuid(txn, uuid)->generation, "restore");
    savepoint.release();
}
} // namespace vc::objects
