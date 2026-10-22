#include "internal/RecordSql.hpp"
#include "internal/GeometryCoordinates.hpp"

namespace vc::objects {
using namespace detail;
namespace {
std::pair<const char*, int64_t> recordLocation(sqlite3* db, const std::array<unsigned char, 16>& bytes, const std::string& uuid)
{
    if (auto id = objectIdByUuid(db, bytes)) return {"objects", *id};
    if (auto id = edgeIdByUuid(db, bytes)) return {"edges", *id};
    throw std::runtime_error("object store: record not found: " + uuid);
}
}

void ObjectStore::setApproval(WriteTxn& txn, const std::string& recordUuid, const std::string& state)
{
    if (state != "pending" && state != "approved" && state != "rejected")
        throw std::runtime_error("object store: invalid approval state: " + state);
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Savepoint savepoint(db);
    if (state == "approved" && edgeByUuid(txn, recordUuid) && edgeGeometryState(txn, recordUuid).stale)
        throw std::runtime_error("object store: explicitly update the relationship before approving stale geometry");
    const auto bytes = parseUuid(recordUuid);
    const auto [table, id] = recordLocation(db, bytes, recordUuid);
    int64_t generation;
    {
        Statement cur(db, std::string("SELECT generation FROM ") + table + " WHERE id=?", id);
        (void)cur.row();
        generation = cur.i64(0);
    }
    Statement s(db, std::string("UPDATE ") + table +
                        " SET approval_state=?,approved_by=?,approved_at=?,approved_host=?,approved_generation=? WHERE id=?",
        state);
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
    const auto path = jsonPathForKey(key);
    const auto bytes = parseUuid(recordUuid);
    const auto [table, id] = recordLocation(db, bytes, recordUuid);
    // Compare JSON values, not their spelling. Missing and explicit null are
    // distinct; removing a missing key is also a no-op. Keep approval/history
    // untouched when an autosave submits the same logical value.
    {
        Statement current(db, std::string("SELECT meta FROM ") + table + " WHERE id=?", id);
        (void)current.row();
        const auto meta = nlohmann::json::parse(current.text(0));
        const auto found = meta.find(key);
        if (jsonValue ? found != meta.end() && *found == nlohmann::json::parse(*jsonValue)
                      : found == meta.end()) {
            savepoint.release();
            return;
        }
    }
    if (jsonValue) {
        execute(db, std::string("UPDATE ") + table + " SET meta=json_set(meta,?,json(?)) WHERE id=?", path, *jsonValue,
            id);
    } else {
        execute(db, std::string("UPDATE ") + table + " SET meta=json_remove(meta,?) WHERE id=?", path, id);
    }
    stampContentChange(db, author, host, table, id, bytes, "meta");
    savepoint.release();
}

}  // namespace

void ObjectStore::setMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key, const std::string& jsonValue)
{
    if (key == "coordinates") {
        const auto object = objectByUuid(txn, recordUuid);
        if (object && hasGeometryCoordinates(object->kind)) {
            const auto value = geometryCoordinates(nlohmann::json::parse(jsonValue));
            mutateMeta(static_cast<sqlite3*>(connection(txn)), impl_->options.author, impl_->hostname, recordUuid, key, value.dump());
            return;
        }
    }
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
                    "approval_state,approved_by,approved_at,approved_host,approved_generation FROM journal WHERE seq>? ORDER BY seq",
        afterSeq);
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
