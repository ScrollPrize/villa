#include "vc/core/graph/GraphDb.hpp"
#include "vc/core/objects/ObjectStore.hpp"
#include "RecordDbInternal.hpp"

namespace vc::graph {
namespace {
struct ValidatedObject {
    ObjectRecord record;
    objects::GeometryRevision revision;
    std::string storeUuid;
};

ValidatedObject validate(ObjectStore& store, const std::string& object, const std::string& revision)
{
    // A new read transaction sees only committed, durably published revisions.
    auto read = store.read();
    const auto record = store.objectByUuid(read, object);
    const auto geometry = store.geometryRevision(read, revision);
    if (!record || !geometry || geometry->objectUuid != record->uuid)
        throw std::invalid_argument("graph: revision does not belong to the object in this store");
    return {*record, *geometry, store.uuid(read)};
}

void selectRevision(sqlite3* db, const ValidatedObject& value)
{
    Statement head(db, "INSERT INTO geometry_heads(object_uuid,revision_uuid,store_uuid) VALUES(?,?,?) "
                       "ON CONFLICT(object_uuid) DO UPDATE SET revision_uuid=excluded.revision_uuid");
    head.bindBlob(1, parseUuid(value.record.uuid));
    head.bindBlob(2, parseUuid(value.revision.uuid));
    head.bindBlob(3, parseUuid(value.storeUuid));
    head.stepDone();
    Statement descriptor(db, "UPDATE objects SET storage_kind='object_store',data_checksum=?,data_generation=? WHERE uuid=?");
    descriptor.bind(1, value.revision.contentHash);
    descriptor.bind(2, value.revision.objectGeneration);
    descriptor.bindBlob(3, parseUuid(value.record.uuid));
    descriptor.stepDone();

}
}

std::optional<ObjectSelection> GraphDb::selectedObject(const Txn& txn, const std::string& object) const
{
    Statement s(static_cast<sqlite3*>(connection(txn)),
                "SELECT object_uuid,store_uuid,revision_uuid FROM geometry_heads WHERE object_uuid=?");
    s.bindBlob(1, parseUuid(object));
    if (!s.row()) return std::nullopt;
    return ObjectSelection{formatUuid(s.blob16(0)), formatUuid(s.blob16(1)), formatUuid(s.blob16(2))};
}

void GraphDb::includeObject(WriteTxn& txn, ObjectStore& store, const std::string& object, const std::string& revision)
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto value = validate(store, object, revision);
    if (const auto selected = selectedObject(txn, object)) {
        if (selected->storeUuid == value.storeUuid && selected->revisionUuid == value.revision.uuid) return;
        throw std::runtime_error("graph: object already included; use advanceObject with its expected revision");
    }
    exec(db, "SAVEPOINT include_object;");
    try {
        if (const auto existing = objectByUuid(txn, object)) {
            if (existing->kind != value.record.kind) throw std::invalid_argument("graph: object kind mismatch");
            stampContentChange(db, author(), hostname(), "objects", *objectIdByUuid(db, parseUuid(object)),
                               parseUuid(object), "include_object");
        } else {
            auto membership = value.record;
            // Seed graph-local annotations without copying object approval.
            membership.storageKind.clear();
            membership.dataGeneration = 0;
            membership.dataChecksum.reset();
            // Relative source hints belong to the store's directory.
            if (!membership.storageRef.empty() && membership.storageRef.find("://") == std::string::npos)
                membership.storageRef = std::filesystem::absolute(store.path().parent_path() / membership.storageRef).string();
            upsertObject(txn, membership);
        }
        selectRevision(db, value);
        for (const auto& edge : edgesOf(txn, object))
            stampContentChange(db, author(), hostname(), "edges", *edgeIdByUuid(db, parseUuid(edge.uuid)),
                               parseUuid(edge.uuid), "geometry_invalidated");
        exec(db, "RELEASE include_object;");
    } catch (...) { exec(db, "ROLLBACK TO include_object; RELEASE include_object;"); throw; }
}

void GraphDb::advanceObject(WriteTxn& txn, ObjectStore& store, const std::string& object,
                            const std::string& revision, const std::string& expected)
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    const auto selected = selectedObject(txn, object);
    if (!selected || selected->revisionUuid != formatUuid(parseUuid(expected)))
        throw std::runtime_error("graph: selected revision changed; reload before committing");
    const auto value = validate(store, object, revision);
    if (selected->storeUuid != value.storeUuid) throw std::invalid_argument("graph: object belongs to another store");
    if (selected->revisionUuid == value.revision.uuid) return;
    exec(db, "SAVEPOINT advance_object;");
    try {
        selectRevision(db, value);
        stampContentChange(db, author(), hostname(), "objects", *objectIdByUuid(db, parseUuid(object)),
                           parseUuid(object), "advance_object");
        for (const auto& edge : edgesOf(txn, object))
            stampContentChange(db, author(), hostname(), "edges", *edgeIdByUuid(db, parseUuid(edge.uuid)),
                               parseUuid(edge.uuid), "geometry_invalidated");
        exec(db, "RELEASE advance_object;");
    } catch (...) { exec(db, "ROLLBACK TO advance_object; RELEASE advance_object;"); throw; }
}

std::vector<ObjectUpdate> GraphDb::availableUpdates(const Txn& txn, ObjectStore& store) const
{
    auto read = store.read();
    const auto id = store.uuid(read);
    Statement s(static_cast<sqlite3*>(connection(txn)),
                "SELECT object_uuid,revision_uuid FROM geometry_heads WHERE store_uuid=? ORDER BY object_uuid");
    s.bindBlob(1, parseUuid(id));
    std::vector<ObjectUpdate> result;
    while (s.row()) {
        const auto object = formatUuid(s.blob16(0)), selected = formatUuid(s.blob16(1));
        const auto current = store.currentGeometry(read, object);
        if (!current) throw std::runtime_error("graph: referenced object is missing from the store");
        if (current->uuid != selected) result.push_back({object, selected, current->uuid});
    }
    return result;
}
} // namespace vc::graph
