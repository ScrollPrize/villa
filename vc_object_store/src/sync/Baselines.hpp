#pragma once
#include "internal/RecordSql.hpp"
#include "internal/RecordJson.hpp"
#include "internal/EdgeSql.hpp"
#include "geometry/GeometryInternal.hpp"

namespace vc::objects::detail {
using Json = nlohmann::json;

struct SyncCursor { int64_t journal = 0, geometry = 0; };
inline SyncCursor syncCursor(sqlite3* db) {
    Statement row(db, "SELECT coalesce((SELECT max(seq) FROM journal),0),"
                      "coalesce((SELECT max(rowid) FROM geometry_revisions),0)");
    (void)row.row(); return {row.i64(0), row.i64(1)};
}

inline void changedRecords(sqlite3* db, SyncCursor after, std::set<std::string>& ids) {
    Statement rows(db, "SELECT DISTINCT record_uuid FROM journal WHERE seq>? UNION "
                       "SELECT object_uuid FROM geometry_revisions WHERE rowid>?", after.journal, after.geometry);
    while (rows.row()) ids.insert(formatUuid(rows.blob16(0)));
}

inline Json syncRecord(ObjectStore& store, const Txn& txn, const std::string& id) {
    if (const auto object = store.objectByUuid(txn, id)) {
        auto record = objectToJson(*object); record["record_type"] = "objects";
        if (const auto geometry = store.currentGeometry(txn, id))
            record["geometry"] = {{"revision_uuid", geometry->uuid}, {"content_hash", geometry->contentHash}};
        return record;
    }
    if (const auto edge = store.edgeByUuid(txn, id)) {
        auto record = edgeToJson(*edge); record["record_type"] = "edges";
        Statement binding(ObjectStoreImplAccess::conn(txn), "SELECT a_revision,b_revision FROM edge_geometry_bindings "
                                                          "WHERE edge_id=(SELECT id FROM edges WHERE uuid=?)",
            parseUuid(id));
        if (!binding.row()) throw std::runtime_error("sync: missing edge geometry binding");
        const bool swap = record["a"] != edge->aUuid;
        const auto epochs = lifecycleBinding(ObjectStoreImplAccess::conn(txn), *edge);
        if (!epochs.first.empty() || !epochs.second.empty())
            record["lifecycle_bindings"] = {{"a", swap ? epochs.second : epochs.first},
                                             {"b", swap ? epochs.first : epochs.second}};
        auto revision = [&](int column) -> Json { return binding.isNull(column) ? Json() : Json(formatUuid(binding.blob16(column))); };
        record["geometry_bindings"] = {{"a", revision(swap ? 1 : 0)}, {"b", revision(swap ? 0 : 1)}};
        return record;
    }
    return nullptr;
}

} // namespace vc::objects::detail
