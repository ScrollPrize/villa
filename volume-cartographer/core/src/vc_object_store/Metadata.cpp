#include "internal/RecordSql.hpp"

namespace vc::objects {
using namespace detail;
void ObjectStore::setApproval(WriteTxn& txn, const std::string& recordUuid, const std::string& state)
{
    if (state != "pending" && state != "approved" && state != "rejected")
        throw std::runtime_error("object store: invalid approval state: " + state);
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Savepoint savepoint(db);
    if (state == "approved" && edgeByUuid(txn, recordUuid) && edgeGeometryState(txn, recordUuid).stale)
        throw std::runtime_error("object store: explicitly update the relationship before approving stale geometry");
    const auto bytes = parseUuid(recordUuid);
    const char* table = nullptr;
    int64_t id = 0;
    int64_t generation = 0;
    if (auto oid = objectIdByUuid(db, bytes)) {
        table = "objects";
        id = *oid;
    } else if (auto eid = edgeIdByUuid(db, bytes)) {
        table = "edges";
        id = *eid;
    } else {
        throw std::runtime_error("object store: record not found: " + recordUuid);
    }
    {
        Statement cur(db, std::string("SELECT generation FROM ") + table + " WHERE id=?");
        cur.bind(1, id);
        (void)cur.row();
        generation = cur.i64(0);
    }
    Statement s(db, std::string("UPDATE ") + table +
                        " SET approval_state=?,approved_by=?,approved_at=?,approved_host=?,approved_generation=? WHERE id=?");
    s.bind(1, state);
    if (state == "pending") {
        s.bindNull(2); s.bindNull(3); s.bindNull(4); s.bindNull(5);
    } else {
        s.bind(2, impl_->options.author);
        s.bind(3, nowIsoUtc());
        s.bind(4, impl_->hostname);
        s.bind(5, generation);
    }
    s.bind(6, id);
    s.stepDone();
    appendJournal(db, impl_->options.author, impl_->hostname, bytes, generation, "approval");
    savepoint.release();
}

namespace {

void mutateMeta(sqlite3* db, const std::string& author, const std::string& host, const std::string& recordUuid,
                const std::string& key, const std::optional<std::string>& jsonValue)
{
    Savepoint savepoint(db);
    if (key.empty()) throw std::runtime_error("object store: meta key must be non-empty");
    const auto bytes = parseUuid(recordUuid);
    const char* table = nullptr;
    int64_t id = 0;
    if (auto oid = objectIdByUuid(db, bytes)) {
        table = "objects";
        id = *oid;
    } else if (auto eid = edgeIdByUuid(db, bytes)) {
        table = "edges";
        id = *eid;
    } else {
        throw std::runtime_error("object store: record not found: " + recordUuid);
    }
    if (jsonValue) {
        Statement s(db, std::string("UPDATE ") + table + " SET meta=json_set(meta,?,json(?)) WHERE id=?");
        s.bind(1, jsonPathForKey(key));
        s.bind(2, *jsonValue);
        s.bind(3, id);
        s.stepDone();
    } else {
        Statement s(db, std::string("UPDATE ") + table + " SET meta=json_remove(meta,?) WHERE id=?");
        s.bind(1, jsonPathForKey(key));
        s.bind(2, id);
        s.stepDone();
    }
    stampContentChange(db, author, host, table, id, bytes, "meta");
    savepoint.release();
}

}  // namespace

void ObjectStore::setMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key, const std::string& jsonValue)
{
    mutateMeta(static_cast<sqlite3*>(connection(txn)), impl_->options.author, impl_->hostname, recordUuid, key, jsonValue);
}

void ObjectStore::removeMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key)
{
    mutateMeta(static_cast<sqlite3*>(connection(txn)), impl_->options.author, impl_->hostname, recordUuid, key, std::nullopt);
}

std::vector<JournalEntry> ObjectStore::journalSince(const Txn& txn, int64_t afterSeq) const
{
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, "SELECT seq,record_uuid,generation,author,hostname,timestamp,change_kind,"
                    "approval_state,approved_by,approved_at,approved_host,approved_generation FROM journal WHERE seq>? ORDER BY seq");
    s.bind(1, afterSeq);
    std::vector<JournalEntry> out;
    while (s.row()) {
        JournalEntry e;
        e.seq = s.i64(0);
        e.recordUuid = formatUuid(s.blob16(1));
        e.generation = s.i64(2);
        e.author = s.text(3);
        e.hostname = s.text(4);
        e.timestamp = s.text(5);
        e.changeKind = s.text(6);
        e.approvalState = s.optText(7);
        e.approvedBy = s.optText(8);
        e.approvedAt = s.optText(9);
        e.approvedHost = s.optText(10);
        if (!s.isNull(11)) e.approvedGeneration = s.i64(11);
        out.push_back(std::move(e));
    }
    return out;
}

} // namespace vc::objects
