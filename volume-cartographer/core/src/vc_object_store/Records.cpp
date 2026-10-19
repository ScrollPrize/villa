#include "internal/RecordSql.hpp"

namespace vc::objects {
std::string ObjectStore::upsertObject(WriteTxn& txn, const ObjectRecord& record)
{
    return upsertObjectImpl(static_cast<sqlite3*>(connection(txn)), impl_->options.author, impl_->hostname, record, false);
}

namespace {
void refreshLineMeasurements(sqlite3* db, const std::string& author, const std::string& host,
    const std::string& table, const std::string& uuid, double length,
    const std::optional<std::string>& family, const char* journalKind)
{
    if (!std::isfinite(length) || length < 0)
        throw std::invalid_argument("object store: line length must be finite and nonnegative");
    const auto key = parseUuid(uuid);
    Statement current(db, "SELECT o.id,o.generation,t.length,t.family FROM objects o JOIN " + table +
        " t ON t.object_id=o.id WHERE o.uuid=?");
    current.bindBlob(1, key);
    if (!current.row()) throw std::runtime_error("object store: " + table + " record not found: " + uuid);
    const auto previousFamily = current.text(3);
    if (family && !previousFamily.empty() && previousFamily != *family)
        throw std::runtime_error("object store: registered family differs from source: " + uuid);
    const auto newFamily = family.value_or(previousFamily);
    if (current.f64(2) == length && previousFamily == newFamily) return;
    Statement update(db, "UPDATE " + table + " SET length=?,family=? WHERE object_id=?");
    update.bind(1, length);
    update.bind(2, newFamily);
    update.bind(3, current.i64(0));
    update.stepDone();
    stampContentChange(db, author, host, "objects", current.i64(0), key, journalKind);
}
} // namespace

void ObjectStore::setTrackLength(WriteTxn& txn, const std::string& uuid, double length,
    const std::optional<std::string>& family)
{
    if (family && *family != "" && *family != "horizontal" && *family != "vertical")
        throw std::invalid_argument("object store: invalid packed track family");
    refreshLineMeasurements(static_cast<sqlite3*>(connection(txn)), impl_->options.author, impl_->hostname,
        "tracks", uuid, length, family, family ? "track_measurements" : "track_length");
}

void ObjectStore::setFiberLength(WriteTxn& txn, const std::string& uuid, double length)
{
    refreshLineMeasurements(static_cast<sqlite3*>(connection(txn)), impl_->options.author, impl_->hostname,
        "fibers", uuid, length, std::nullopt, "fiber_length");
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
                                                 const std::optional<std::string>& approvalState) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    std::string sql = std::string("SELECT ") + kObjectCols + " FROM objects WHERE kind=?";
    if (approvalState) sql += " AND approval_state=?";
    sql += " ORDER BY id";
    Statement s(db, sql);
    s.bind(1, kind);
    if (approvalState) s.bind(2, *approvalState);
    std::vector<ObjectRecord> out;
    while (s.row()) out.push_back(readObjectRow(db, s));
    return out;
}

int64_t ObjectStore::objectCount(const Txn& txn, const std::string& kind) const
{
    Statement s(static_cast<sqlite3*>(connection(txn)), "SELECT count(*) FROM objects WHERE kind=?");
    s.bind(1, kind);
    (void)s.row();
    return s.i64(0);
}

std::vector<ObjectRecord> ObjectStore::objectsByTag(const Txn& txn, const std::string& tag) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    std::string cols = kObjectCols;
    // Column names are ambiguous under the join only for id; qualify by selecting explicit object columns.
    Statement s(db, "SELECT o.id,o.uuid,o.kind,o.name,o.generation,o.created_by,o.created_at,o.created_host,o.changed_by,o.changed_at,"
                    "o.changed_host,o.approval_state,o.approved_by,o.approved_at,o.approved_host,o.approved_generation,o.format_version,"
                    "o.storage_kind,o.storage_ref,o.data_generation,o.data_checksum,o.meta FROM objects o "
                    "JOIN object_tags t ON t.object_id=o.id WHERE t.tag=? ORDER BY o.id");
    (void)cols;
    s.bind(1, tag);
    std::vector<ObjectRecord> out;
    while (s.row()) out.push_back(readObjectRow(db, s));
    return out;
}

void ObjectStore::deleteObject(WriteTxn& txn, const std::string& uuid)
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    const auto bytes = parseUuid(uuid);
    auto id = objectIdByUuid(db, bytes);
    if (!id) throw std::runtime_error("object store: object not found: " + uuid);
    Statement s(db, "DELETE FROM objects WHERE id=?");  // FK RESTRICT on edges refuses while referenced
    s.bind(1, *id);
    s.stepDone();
    appendJournal(db, impl_->options.author, impl_->hostname, bytes, 0, "delete");
}

} // namespace vc::objects
