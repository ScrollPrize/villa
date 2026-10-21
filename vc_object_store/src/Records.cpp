#include "internal/RecordSql.hpp"

namespace vc::objects {
using namespace detail;
std::string ObjectStore::upsertObject(WriteTxn& txn, const ObjectRecord& record)
{
    return upsertObjectImpl(static_cast<sqlite3*>(connection(txn)), impl_->options.author, impl_->hostname, record, false);
}

std::optional<ObjectRecord> ObjectStore::objectByUuid(const Txn& txn, const std::string& uuid) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, std::string("SELECT ") + kObjectCols + " FROM objects WHERE uuid=?");
    s.bindBlob(1, parseUuid(uuid));
    if (!s.row()) return std::nullopt;
    return readObjectRow(db, s);
}

std::vector<ObjectRecord> ObjectStore::objectsByKind(const Txn& txn, const std::string& kind,
                                                 const std::optional<std::string>& approvalState, bool includeRetired) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    std::string sql = std::string("SELECT ") + kObjectCols + " FROM objects WHERE kind=?";
    if (!includeRetired) sql += " AND " + activeObjectSql("objects.id");
    if (approvalState) sql += " AND approval_state=?";
    sql += " ORDER BY id";
    Statement s(db, sql);
    s.bind(1, kind);
    if (approvalState) s.bind(2, *approvalState);
    return readObjectRows(db, s);
}

int64_t ObjectStore::objectCount(const Txn& txn, const std::string& kind, bool includeRetired) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, "SELECT count(*) FROM objects WHERE kind=?" +
        (includeRetired ? std::string() : " AND " + activeObjectSql("objects.id")));
    s.bind(1, kind);
    (void)s.row();
    return s.i64(0);
}

std::vector<ObjectRecord> ObjectStore::objectsByTag(const Txn& txn, const std::string& tag, bool includeRetired) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    // Column names are ambiguous under the join only for id; qualify by selecting explicit object columns.
    Statement s(db, "SELECT o.id,o.uuid,o.kind,o.name,o.generation,o.created_by,o.created_at,o.created_host,o.changed_by,o.changed_at,"
                    "o.changed_host,o.approval_state,o.approved_by,o.approved_at,o.approved_host,o.approved_generation,o.format_version,"
                    "o.storage_kind,o.storage_ref,o.data_generation,o.data_checksum,o.meta FROM objects o "
                    "JOIN object_tags t INDEXED BY object_tags_tag ON t.object_id=o.id WHERE t.tag=?" +
                    (includeRetired ? std::string() : " AND " + activeObjectSql("o.id")) + " ORDER BY o.id");
    s.bind(1, tag);
    return readObjectRows(db, s);
}

void ObjectStore::deleteObject(WriteTxn& txn, const std::string& uuid)
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Savepoint savepoint(db);
    const auto bytes = parseUuid(uuid);
    auto id = objectIdByUuid(db, bytes);
    if (!id) throw std::runtime_error("object store: object not found: " + uuid);
    Statement s(db, "DELETE FROM objects WHERE id=?");  // FK RESTRICT on edges refuses while referenced
    s.bind(1, *id);
    s.stepDone();
    appendJournal(db, impl_->options.author, impl_->hostname, bytes, 0, "delete");
    savepoint.release();
}

} // namespace vc::objects
