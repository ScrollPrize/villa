#include "vc_object_store/objects/StoreReader.hpp"
#include "internal/RecordSql.hpp"
#include "internal/EdgeLayout.hpp"
#include "internal/JsonFields.hpp"
#include "vc_object_store/objects/EdgeKinds.hpp"
#include "internal/RecordOverrides.hpp"
#include "internal/ExtensionHistory.hpp"
#include "vc_object_store/objects/GeometryCoordinates.hpp"

namespace vc::objects {
using namespace detail;
namespace {
std::pair<const char*, RecordCommon> currentRecord(sqlite3* db, const std::string& uuid)
{
    if (const auto object = resolveObject(db, uuid)) return {"objects", object->record};
    if (const auto edge = resolveEdge(db, uuid)) return {"edges", edge->record};
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
    const auto [table, current] = currentRecord(db, recordUuid);
    const bool edge=std::string_view(table)=="edges";
    if(edge)requireEdgeColumns(db,{"approval_state","approved_by","approved_at","approved_host","approved_version"});
    if (current.approvalState == state) return;
    createRecordOverride(db, table, recordUuid);
    const auto version = writeRecordVersion(db, table, recordUuid);
    const int64_t commit_sequence = writeCommitSequence(db);
    Statement s(db, std::string("UPDATE ") + table +
                        " SET approval_state=?,approved_by=?,approved_at=?,approved_host=?,approved_version=?,commit_sequence=? WHERE uuid=?");
    if(edge)s.bind(1,edgeApprovalCode(state));else s.bind(1,state);
    if (state == "pending") {
        s.bindNull(2); s.bindNull(3); s.bindNull(4); s.bindNull(5);
    } else {
        s.bind(2, author());
        s.bind(3, nowIsoUtc());
        s.bind(4, impl_->hostname);
        s.bind(5, version);
    }
    s.bind(6, commit_sequence); s.bind(7, bytes);
    s.stepDone();
    appendJournal(db, author(), impl_->hostname, bytes, commit_sequence, "approval");
    savepoint.release();
}

namespace {

void mutateMeta(sqlite3* db, const std::string& author, const std::string& host, const std::string& recordUuid,
                const std::string& key, const std::optional<std::string>& jsonValue)
{
    Savepoint savepoint(db);
    if (key.empty()) throw std::runtime_error("object store: meta key must be non-empty");
    if (key == extensionManifestKey)
        throw std::invalid_argument("extension: vcobj_extensions is reserved; use replaceExtensionRows");
    const auto path = jsonPathForKey(key);
    const auto bytes = parseUuid(recordUuid);
    const auto [table, current] = currentRecord(db, recordUuid);
    // Compare JSON values, not their spelling. Missing and explicit null are
    // distinct; removing a missing key is also a no-op. Keep approval/history
    // untouched when an autosave submits the same logical value.
    if(std::string_view(table)=="edges")requireEdgeColumns(db,{"meta"});
    {
        const auto meta = parseObjectJson(current.meta);
        const auto found = meta.find(key);
        if (jsonValue ? found != meta.end() && *found == nlohmann::json::parse(*jsonValue)
                      : found == meta.end()) {
            return;
        }
    }
    createRecordOverride(db, table, recordUuid);
    int64_t id=0;
    if(std::string_view(table)=="objects") {
        Statement location(db,"SELECT id FROM objects WHERE uuid=?",bytes);
        if(!location.row())throw std::runtime_error("object store: record override is missing: "+recordUuid);
        id=location.i64(0);
    }
    if(std::string_view(table)=="objects")beforeCurrentObjectEdit(db,recordUuid);
    if (jsonValue) {
        execute(db, std::string("UPDATE ") + table + " SET meta=json_set(coalesce(nullif(meta,''),'{}'),?,json(?)) WHERE uuid=?", path, *jsonValue,
            bytes);
    } else {
        execute(db, std::string("UPDATE ") + table + " SET meta=nullif(json_remove(coalesce(nullif(meta,''),'{}'),?),'{}') WHERE uuid=?", path, bytes);
    }
    stampContentChange(db, author, host, table, id, bytes, "meta");
    savepoint.release();
}

}  // namespace

void ObjectStore::setMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key, const std::string& jsonValue)
{
    if (key == "source")
        validateGeometrySourceMetadata(static_cast<sqlite3*>(connection(txn)), recordUuid, nlohmann::json::parse(jsonValue));
    if (key == "coordinates") {
        const auto object = objectByUuid(txn, recordUuid);
        if (object && hasGeometryCoordinates(object->kind)) {
            const auto value = geometryCoordinates(nlohmann::json::parse(jsonValue));
            mutateMeta(static_cast<sqlite3*>(connection(txn)), author(), impl_->hostname, recordUuid, key, value.dump());
            return;
        }
    }
    mutateMeta(static_cast<sqlite3*>(connection(txn)), author(), impl_->hostname, recordUuid, key, jsonValue);
}

SetMetaOp bindCoordinatesOperation(const ObjectRecord& object, const std::string& volume, double storedUnitsPerVoxel)
{
    if (!hasGeometryCoordinates(object.kind))
        throw std::invalid_argument("objects of kind " + object.kind + " have no coordinate frame");
    const auto meta = parseObjectJson(object.meta);
    const auto merged = mergeGeometryCoordinates(meta.value("coordinates", nlohmann::json::object()),
                                                 volumeCoordinates(volume, storedUnitsPerVoxel));
    return {object.uuid, "coordinates", merged.dump()};
}

void ObjectStore::bindGeometryCoordinates(WriteTxn& txn, const std::string& objectUuid, const std::string& volume,
                                          double storedUnitsPerVoxel)
{
    const auto object = objectByUuid(txn, objectUuid);
    if (!object) throw std::invalid_argument("object not found: " + objectUuid);
    const auto operation = bindCoordinatesOperation(*object, volume, storedUnitsPerVoxel);
    setMeta(txn, objectUuid, operation.key, operation.jsonValue);
}

void ObjectStore::removeMeta(WriteTxn& txn, const std::string& recordUuid, const std::string& key)
{
    mutateMeta(static_cast<sqlite3*>(connection(txn)), author(), impl_->hostname, recordUuid, key, std::nullopt);
}

int64_t ObjectStore::journalPosition(const Txn& txn) const
{
    requireHistory(static_cast<sqlite3*>(connection(txn)));
    Statement position(static_cast<sqlite3*>(connection(txn)), "SELECT COALESCE(MAX(seq),0) FROM journal");
    position.row();
    return position.i64(0);
}

std::vector<JournalEntry> ObjectStore::journalSince(const Txn& txn, int64_t afterSeq) const
{
    requireHistory(static_cast<sqlite3*>(connection(txn)));
    sqlite3* db = static_cast<sqlite3*>(connection(txn));
    Statement s(db, "SELECT seq,record_uuid,commit_sequence,author,hostname,timestamp,change_kind,"
                    "approval_state,approved_by,approved_at,approved_host,approved_version,record_kind FROM journal WHERE seq>? ORDER BY seq",
        afterSeq);
    std::vector<JournalEntry> out;
    while (s.row()) {
        JournalEntry e;
        e.seq = s.i64(0);
        e.recordUuid = formatUuid(s.blob16(1));
        e.commitSequence = s.i64(2);
        e.author = s.text(3);
        e.hostname = s.text(4);
        e.timestamp = s.text(5);
        e.changeKind = s.text(6);
        e.approvalState = s.optText(7);
        e.approvedBy = s.optText(8);
        e.approvedAt = s.optText(9);
        e.approvedHost = s.optText(10);
        if (!s.isNull(11)) e.approvedVersion = s.i64(11);
        e.kind = s.text(12);
        out.push_back(std::move(e));
    }
    return out;
}

} // namespace vc::objects
