#include "internal/RecordSql.hpp"
#include "internal/RecordResolver.hpp"
#include "internal/RecordOverrides.hpp"
#include "internal/ExtensionHistory.hpp"
#include "geometry/GeometryDelta.hpp"

namespace vc::objects {
using namespace detail;
std::string ObjectStore::upsertObject(WriteTxn& txn, const ObjectRecord& record)
{
    return upsertObjectImpl(static_cast<sqlite3*>(connection(txn)), author(), impl_->hostname, record, false);
}

std::string ObjectStore::insertObjectWithId(WriteTxn& txn, const ObjectRecord& record, int64_t id)
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    ExplicitRecordId allocation(db,"objects",record.uuid,id);
    const auto result=upsertObject(txn,record);
    allocation.finish();return result;
}

std::optional<ObjectRecord> ObjectStore::objectByUuid(const Txn& txn, const std::string& uuid) const
{
    const auto object=resolveObject(static_cast<sqlite3*>(connection(txn)),uuid);
    return object?std::optional(object->record):std::nullopt;
}

std::vector<ObjectRecord> ObjectStore::objectsByKind(const Txn& txn, const std::string& kind,
                                                 const std::optional<std::string>& approvalState, bool includeRetired) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    if(querySources(db).size()>1){
        ObjectSelection selection;selection.kind=kind;selection.includeRetired=includeRetired;
        auto records=selectObjects(txn,selection);
        if(approvalState)std::erase_if(records,[&](const auto& r){return r.approvalState!=*approvalState;});
        return records;
    }
    std::string sql = std::string("SELECT ") + kObjectCols + " FROM objects WHERE kind=? AND state<>2";
    if (!includeRetired) sql += " AND " + activeObjectSql("objects.id");
    if (approvalState) sql += " AND approval_state=?";
    sql += " ORDER BY id";
    Statement s(db, sql, kind);
    if (approvalState) s.bind(2, *approvalState);
    return readObjectRows(db, s);
}

int64_t ObjectStore::objectCount(const Txn& txn, const std::string& kind, bool includeRetired) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    if(querySources(db).size()>1){
        ObjectSelection selection;selection.kind=kind;selection.includeRetired=includeRetired;
        return selectObjects(txn,selection,std::vector<std::string>{}).size();
    }
    Statement s(db, "SELECT count(*) FROM objects WHERE kind=? AND state<>2" +
        (includeRetired ? std::string() : " AND " + activeObjectSql("objects.id")), kind);
    (void)s.row();
    return s.i64(0);
}

std::vector<ObjectRecord> ObjectStore::objectsByTag(const Txn& txn, const std::string& tag, bool includeRetired) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    if(querySources(db).size()>1){
        ObjectSelection selection;selection.tag=tag;selection.includeRetired=includeRetired;
        return selectObjects(txn,selection);
    }
    // Column names are ambiguous under the join only for id; qualify by selecting explicit object columns.
    Statement s(db, "SELECT o.id,o.uuid,o.kind,o.name,o.commit_sequence,o.created_by,o.created_at,o.created_host,o.changed_by,o.changed_at,"
                    "o.changed_host,o.approval_state,o.approved_by,o.approved_at,o.approved_host,o.approved_version,o.format_version,"
                    "o.storage_kind,o.storage_ref,o.version,o.data_checksum,o.meta,o.version_generation FROM objects o "
                    "JOIN object_tags t INDEXED BY object_tags_tag ON t.object_id=o.id WHERE t.tag=? AND o.state<>2" +
                    (includeRetired ? std::string() : " AND " + activeObjectSql("o.id")) + " ORDER BY o.id", tag);
    return readObjectRows(db, s);
}

std::vector<std::string> ObjectStore::objectTags(const Txn& txn, const std::optional<std::string>& kind) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    if(querySources(db).size()>1){
        ObjectSelection selection;selection.kind=kind;std::set<std::string> tags;
        for(const auto& record:selectObjects(txn,selection,std::vector<std::string>{"tags"}))
            tags.insert(record.tags.begin(),record.tags.end());
        return {tags.begin(),tags.end()};
    }
    Statement s(db, "SELECT DISTINCT t.tag FROM object_tags t JOIN objects o ON o.id=t.object_id WHERE (?1 IS NULL OR o.kind=?1) AND " +
                    activeObjectSql("o.id") + " ORDER BY t.tag");
    if (kind) s.bind(1, *kind); else s.bindNull(1);
    std::vector<std::string> tags;
    while (s.row()) tags.push_back(s.text(0));
    return tags;
}

void ObjectStore::deleteObject(WriteTxn& txn, const std::string& uuid)
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Savepoint savepoint(db);
    createRecordOverride(db,"objects",uuid);
    const auto bytes = parseUuid(uuid);
    Statement row(db, "SELECT id,kind FROM objects WHERE uuid=? AND state<>2", bytes);
    if (!row.row()) throw std::runtime_error("object store: object not found: " + uuid);
    const auto kind = row.text(1);
    // UUID endpoints may live in referenced stores; RESTRICT is checked by the API.
    if (!incidentEdges(db,uuid).empty()) throw std::runtime_error("object store: remove incident relationships before deletion");
    beforeCurrentObjectEdit(db,uuid);
    writeObjectVersion(db,uuid);
    clearCurrentExtensionRows(db, "objects", uuid);
    // Keep only the same-key deletion marker, not live subtype identities or
    // tags. In particular a deleted track must release its source identity.
    for(const auto* table:{"fibers","surfaces","tracks","object_tags"})
        execute(db,std::string("DELETE FROM ")+table+" WHERE object_id=?",row.i64(0));
    clearGeometryMaterialization(db,uuid);
    execute(db,"UPDATE objects SET state=2,lifecycle_epoch='',geometry_manifest=NULL,storage_kind='',storage_ref='',data_checksum=NULL,"
        "commit_sequence=?,changed_by=?,changed_host=?,changed_at=? WHERE id=?",
        writeCommitSequence(db),author(),impl_->hostname,nowIsoUtc(),row.i64(0));
    appendJournal(db, author(), impl_->hostname, bytes, 0, "delete", kind);
    savepoint.release();
}

} // namespace vc::objects
