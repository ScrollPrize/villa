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
    Statement s(db, std::string("SELECT ") + kObjectCols + " FROM objects WHERE uuid=?", parseUuid(uuid));
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
    Statement s(db, sql, kind);
    if (approvalState) s.bind(2, *approvalState);
    return readObjectRows(db, s);
}

int64_t ObjectStore::objectCount(const Txn& txn, const std::string& kind, bool includeRetired) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, "SELECT count(*) FROM objects WHERE kind=?" +
        (includeRetired ? std::string() : " AND " + activeObjectSql("objects.id")), kind);
    (void)s.row();
    return s.i64(0);
}

std::vector<std::string> ObjectStore::objectTags(const Txn& txn, const std::optional<std::string>& kind) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
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
    const auto bytes = parseUuid(uuid);
    Statement row(db, "SELECT id,kind FROM objects WHERE uuid=?", bytes);
    if (!row.row()) throw std::runtime_error("object store: object not found: " + uuid);
    const auto kind = row.text(1);
    execute(db, "DELETE FROM objects WHERE id=?", row.i64(0));  // FK RESTRICT on edges refuses while referenced
    appendJournal(db, impl_->options.author, impl_->hostname, bytes, 0, "delete", kind);
    savepoint.release();
}

} // namespace vc::objects
