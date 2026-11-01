#include "sync/Apply.hpp"
#include "internal/EdgeSql.hpp"
#include "internal/RecordSql.hpp"
#include "geometry/GeometryDelta.hpp"

namespace vc::objects::detail {
void applyObjects(ObjectStore& store, WriteTxn& txn, const Json& before, const Json& merged) {
    auto* db = ObjectStoreImplAccess::conn(txn);
    // Remove changed edges first so deletions and identity swaps cannot transiently collide.
    for (const auto& [id, record] : before.items())
        if (!isObjectRecord(record) && record != recordAt(merged, id)) store.removeEdge(txn, id);
    for (const auto& [id, record] : before.items())
        if (isObjectRecord(record) && recordAt(merged, id).is_null()) store.deleteObject(txn, id);
    // Release track identity constraints for all changed records before restoring details.
    for (const auto& [id, record] : merged.items()) if (isObjectRecord(record) && record != recordAt(before, id)) {
        execute(db, "DELETE FROM tracks WHERE object_id=(SELECT id FROM objects WHERE uuid=?)", parseUuid(id));
    }
    for (const auto& [id, record] : merged.items()) if (isObjectRecord(record) && record != recordAt(before, id)) {
        auto value = objectFromJson(record); commonFromJson(record, value);
        // The head is copied after its owner exists. Restore managed fields once installed.
        if (!store.currentGeometry(txn, id) && recordHead(record) != nullptr) value.storageKind.clear();
        upsertObjectImpl(db, store.author(), store.hostname(), value, true, "sync");
    }
}

void applyHeadsAndEdges(ObjectStore& store, WriteTxn& txn, const Json& before, const Json& merged) {
    auto* db = ObjectStoreImplAccess::conn(txn);
    for (const auto& [id, record] : merged.items()) if (isObjectRecord(record) && record != recordAt(before, id)) {
        if (recordHead(record) != nullptr) {
            execute(db, "INSERT INTO geometry_heads VALUES(?,?) ON CONFLICT(object_uuid) DO UPDATE SET revision_uuid=excluded.revision_uuid",
                parseUuid(id), parseUuid(recordHead(record).get<std::string>()));
            if (recordHead(record) != recordHead(recordAt(before, id))) {
                materializeGeometry(db, id, recordHead(record).get<std::string>());
                refreshGeometryBounds(db, id);
            }
        }
        Statement storage(db, "UPDATE objects SET storage_kind=?,storage_ref=?,data_generation=?,data_checksum=? WHERE uuid=?",
            record.at("storage_kind").get<std::string>(), record.at("storage_ref").get<std::string>(),
            record.at("data_generation").get<int64_t>());
        if (record.contains("data_checksum")) storage.bind(4, record.at("data_checksum").get<std::string>()); else storage.bindNull(4);
        storage.bind(5, parseUuid(id)); storage.stepDone();
    }
    for (const auto& [id, record] : merged.items()) if (!isObjectRecord(record) && record != recordAt(before, id)) {
        auto edge = edgeFromJson(record);
        addEdgeImpl(db, store.author(), store.hostname(), edge, true, "sync");
        const auto actual = *store.edgeByUuid(txn, id);
        const bool swap = actual.aUuid != edge.aUuid;
        const auto epochs = record.value("lifecycle_bindings", Json::object());
        bindLifecycle(db, actual, epochs.value(swap ? "b" : "a", ""), epochs.value(swap ? "a" : "b", ""));
        const auto bindings = record.value("geometry_bindings", Json::object());
        const auto a = bindings.value(swap ? "b" : "a", Json()), b = bindings.value(swap ? "a" : "b", Json());
        Statement put(db, "UPDATE edge_geometry_bindings SET a_revision=?,b_revision=? WHERE edge_id=(SELECT id FROM edges WHERE uuid=?)");
        if (a.is_null()) put.bindNull(1); else put.bind(1, parseUuid(a.get<std::string>()));
        if (b.is_null()) put.bindNull(2); else put.bind(2, parseUuid(b.get<std::string>()));
        put.bind(3, parseUuid(id)); put.stepDone();
    }
}

std::vector<std::string> undecidedRelationships(const Json& merged, const DependencyLookup& geometryFor)
{
    // Synchronization must not turn unresolved geometry/relationship combinations
    // into stale links implicitly. A source edit must carry its explicit decision.
    std::vector<std::string> result;
    for (const auto& [id, edge] : merged.items()) if (!isObjectRecord(edge)) {
        const auto binding = edge.value("geometry_bindings", Json::object());
        for (const auto* endpoint : {"a", "b"}) {
            const auto bound = binding.value(endpoint, Json());
            const auto current = recordHead(merged.at(edge.at(endpoint).get<std::string>()));
            if (bound == current) continue;
            const auto before = geometryFor(bound), after = geometryFor(current);
            const auto anchor = anchorFromJson(edge.at(std::string(endpoint) + "_anchor"));
            if (!before || !after || dependenciesChanged(anchor, *before, *after)) result.push_back(id);
        }
    }
    return result;
}
}
