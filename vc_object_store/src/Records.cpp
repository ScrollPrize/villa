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

std::vector<ObjectCoordinateGroup> ObjectStore::objectCoordinateGroups(const Txn& txn, const std::string& kind,
                                                                     const std::atomic_bool* cancel) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    struct Cancellation {
        sqlite3* db;
        ~Cancellation() { if (db) sqlite3_progress_handler(db, 0, nullptr, nullptr); }
    } cancellation{cancel ? db : nullptr};
    if (cancel) sqlite3_progress_handler(db, 1000, [](void* context) {
        return static_cast<const std::atomic_bool*>(context)->load() ? 1 : 0;
    }, const_cast<std::atomic_bool*>(cancel));
    Statement rows(db,
        "SELECT json_object('volume_uuid',json_extract(o.meta,'$.coordinates.volume_uuid'),"
        "'scale_factor',json_extract(o.meta,'$.coordinates.scale_factor')),count(*) FROM objects o "
        "WHERE o.kind=? AND " + activeObjectSql("o.id") +
        " GROUP BY json_extract(o.meta,'$.coordinates.volume_uuid'),json_extract(o.meta,'$.coordinates.scale_factor')", kind);
    std::vector<ObjectCoordinateGroup> result;
    while (rows.row()) result.push_back({rows.text(0), rows.i64(1)});
    return result;
}

std::vector<ObjectRecord> ObjectStore::objectsByTag(const Txn& txn, const std::string& tag, bool includeRetired) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    // Column names are ambiguous under the join only for id; qualify by selecting explicit object columns.
    Statement s(db, "SELECT o.id,o.uuid,o.kind,o.name,o.generation,o.created_by,o.created_at,o.created_host,o.changed_by,o.changed_at,"
                    "o.changed_host,o.approval_state,o.approved_by,o.approved_at,o.approved_host,o.approved_generation,o.format_version,"
                    "o.storage_kind,o.storage_ref,o.data_generation,o.data_checksum,o.meta FROM objects o "
                    "JOIN object_tags t INDEXED BY object_tags_tag ON t.object_id=o.id WHERE t.tag=?" +
                    (includeRetired ? std::string() : " AND " + activeObjectSql("o.id")) + " ORDER BY o.id", tag);
    return readObjectRows(db, s);
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
    auto id = objectIdByUuid(db, bytes);
    if (!id) throw std::runtime_error("object store: object not found: " + uuid);
    Statement s(db, "DELETE FROM objects WHERE id=?");  // FK RESTRICT on edges refuses while referenced
    s.bind(1, *id);
    s.stepDone();
    appendJournal(db, impl_->options.author, impl_->hostname, bytes, 0, "delete");
    savepoint.release();
}

} // namespace vc::objects
