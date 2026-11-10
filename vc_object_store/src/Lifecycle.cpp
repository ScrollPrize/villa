#include "internal/RecordSql.hpp"
#include "internal/History.hpp"
#include "internal/ExtensionHistory.hpp"
#include "internal/SqlValues.hpp"
#include "geometry/GeometryDelta.hpp"
#include "internal/ImmutableImport.hpp"
#include "internal/RecordResolver.hpp"
#include "internal/RecordOverrides.hpp"

namespace vc::objects {
using namespace detail;

void detail::readLifecycle(sqlite3* db, int64_t id, ObjectRecord& record)
{
    Statement s(db, "SELECT state,lifecycle_epoch FROM objects WHERE id=?", id);
    if (s.row()) { record.retired = s.i64(0)==1; record.lifecycleEpoch = s.text(1); }
}

void detail::writeLifecycle(sqlite3* db, int64_t id, const ObjectRecord& record)
{
    beforeCurrentObjectEdit(db,record.uuid);
    execute(db, "UPDATE objects SET state=?,lifecycle_epoch=? WHERE id=?",int64_t(record.retired),
        record.lifecycleEpoch,id);
}

std::string detail::activeObjectSql(const std::string& id)
{
    return "NOT EXISTS(SELECT 1 FROM objects owned WHERE owned.id="+id+" AND owned.state<>0)";
}

std::string detail::activeEdgeSql()
{
    return "edges.state=0 AND NOT EXISTS(SELECT 1 FROM objects owned "
        "WHERE (owned.uuid=edges.a_uuid OR owned.uuid=edges.b_uuid) AND owned.state<>0)";
}

namespace {
ObjectRecord endpointLifecycle(sqlite3* db,const std::string& uuid)
{
    const auto object=resolveObject(db,uuid);
    if(!object)throw std::runtime_error("relationship endpoint is missing or deleted");
    return object->record;
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
    if (!checked.row()) throw std::runtime_error("object lifecycle requires a checked write (expected versions)");
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
    createRecordOverride(db,"objects",uuid);
    // Retiring is a native publication, not an independent version namespace.
    // Removing current relationships also prevents restore from silently
    // reviving observations whose endpoint was deliberately retired.
    for (const auto& edge : edgesOf(txn,uuid,true)) removeEdge(txn,edge.uuid);
    record->retired = true; record->lifecycleEpoch = std::to_string(writeCommitSequence(db));
    const auto id = *objectIdByUuid(db, parseUuid(uuid));
    writeLifecycle(db, id, *record);
    stampContentChange(db, author(), hostname(), "objects", id, parseUuid(uuid), "retire");
    savepoint.release();
}

void ObjectStore::restoreObject(WriteTxn& txn, const std::string& uuid,
                               const std::optional<ObjectVersion>& sourceVersion)
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    requireCheckedLifecycle(db);
    const auto current = objectByUuid(txn, uuid);
    if (!current) throw std::invalid_argument("object lifecycle: unknown object");
    if (!current->retired && !sourceVersion) return;
    if (!historyEnabled(db)) {
        if (sourceVersion) requireHistory(db);
        Savepoint savepoint(db);
        createRecordOverride(db,"objects",uuid);
        auto restored = *current;
        restored.retired = false; restored.lifecycleEpoch = std::to_string(writeCommitSequence(db));
        const auto id = *objectIdByUuid(db, parseUuid(uuid));
        writeLifecycle(db, id, restored);
        stampContentChange(db, author(), hostname(), "objects", id, parseUuid(uuid), "restore");
        savepoint.release();
        return;
    }
    std::optional<ObjectRevision> source;
    if (sourceVersion) {
        source = objectAtVersion(txn, *sourceVersion);
        if (!source) throw std::invalid_argument("object lifecycle: restore source version is not recorded in this store");
    } else {
        auto history = objectHistory(txn, uuid);
        for (auto it = history.rbegin(); it != history.rend(); ++it)
            if (it->publishedAtUs && !it->deleted && !it->record.retired) { source = *it; break; }
    }
    if (!source || source->objectUuid != current->uuid || source->deleted || source->record.retired)
        throw std::invalid_argument("object lifecycle: restoration requires an active revision of this object");
    Savepoint savepoint(db);
    createRecordOverride(db,"objects",uuid);
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
    const auto origin=resolveObjectVersion(db,source->reference());
    if(!origin)throw std::runtime_error("object lifecycle: restore source is unavailable");
    prepareOverrideReferences(origin->source,db,restored);
    restoreCurrentExtensionRows(db, "objects", uuid, restored.meta);
    // upsert preserves identity/creation stamps and resets approval normally.
    upsertObjectImpl(db, author(), hostname(), restored, false, "restore", true);
    restored.retired = false; restored.lifecycleEpoch = std::to_string(writeCommitSequence(db));
    writeLifecycle(db, id, restored);
    if (oldGeometry != source->geometryRevision) refreshGeometryBounds(db, uuid);
    appendJournal(db, author(), hostname(), parseUuid(uuid), objectByUuid(txn, uuid)->commitSequence, "restore");
    savepoint.release();
}
} // namespace vc::objects
