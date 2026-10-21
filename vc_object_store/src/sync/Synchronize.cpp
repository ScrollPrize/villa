#include "geometry/Dependencies.hpp"
#include "internal/RecordJson.hpp"
#include "internal/EdgeSql.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/GeometryDelta.hpp"
#include "sync/GeometryTransfer.hpp"
#include "sync/Baselines.hpp"
#include "internal/History.hpp"

namespace vc::objects {
using namespace detail;
namespace {
using Json = nlohmann::json;
std::string hash(const std::string& text) {
    return detail::geometryHash(std::span(reinterpret_cast<const uint8_t*>(text.data()), text.size()));
}
Json snapshot(ObjectStore& store, const Txn& txn) {
    // Synchronization includes retired objects and dormant relationships, and
    // uses the same representation as incremental reads.
    Statement rows(ObjectStoreImplAccess::conn(txn), "SELECT uuid FROM objects UNION SELECT uuid FROM edges ORDER BY uuid");
    auto result = Json::object();
    while (rows.row()) {
        const auto id = formatUuid(rows.blob16(0));
        result[id] = syncRecord(store, txn, id);
    }
    return result;
}
Json at(const Json& state, const std::string& id) { return state.value(id, Json()); }
Json head(const Json& record) { return record.is_object() ? record.value("geometry", Json::object()).value("revision_uuid", Json()) : Json(); }
bool object(const Json& record) { return record.is_object() && record.at("record_type") == "objects"; }
void applyObjects(ObjectStore& store, WriteTxn& txn, const Json& before, const Json& merged) {
    auto* db = ObjectStoreImplAccess::conn(txn);
    // Remove changed edges first so deletions and identity swaps cannot transiently collide.
    for (const auto& [id, record] : before.items())
        if (!object(record) && record != at(merged, id)) store.removeEdge(txn, id);
    for (const auto& [id, record] : before.items())
        if (object(record) && !merged.contains(id)) store.deleteObject(txn, id);
    // Release track identity constraints for all changed records before restoring details.
    for (const auto& [id, record] : merged.items()) if (object(record) && record != at(before, id)) {
        Statement remove(db, "DELETE FROM tracks WHERE object_id=(SELECT id FROM objects WHERE uuid=?)");
        remove.bindBlob(1, parseUuid(id)); remove.stepDone();
    }
    for (const auto& [id, record] : merged.items()) if (object(record) && record != at(before, id)) {
        auto value = objectFromJson(record); commonFromJson(record, value);
        // The head is copied after its owner exists. Restore managed fields once installed.
        if (!store.currentGeometry(txn, id) && head(record) != nullptr) value.storageKind.clear();
        upsertObjectImpl(db, store.author(), store.hostname(), value, true, "sync");
    }
}
void applyHeadsAndEdges(ObjectStore& store, WriteTxn& txn, const Json& before, const Json& merged) {
    auto* db = ObjectStoreImplAccess::conn(txn);
    for (const auto& [id, record] : merged.items()) if (object(record) && record != at(before, id)) {
        if (head(record) != nullptr) {
            Statement put(db, "INSERT INTO geometry_heads VALUES(?,?) ON CONFLICT(object_uuid) DO UPDATE SET revision_uuid=excluded.revision_uuid");
            put.bindBlob(1, parseUuid(id)); put.bindBlob(2, parseUuid(head(record).get<std::string>())); put.stepDone();
            if (head(record) != head(at(before, id))) {
                detail::materializeGeometry(db, id, head(record).get<std::string>());
                detail::refreshGeometryBounds(db, id);
            }
        }
        Statement storage(db, "UPDATE objects SET storage_kind=?,storage_ref=?,data_generation=?,data_checksum=? WHERE uuid=?");
        storage.bind(1, record.at("storage_kind").get<std::string>()); storage.bind(2, record.at("storage_ref").get<std::string>());
        storage.bind(3, record.at("data_generation").get<int64_t>());
        if (record.contains("data_checksum")) storage.bind(4, record.at("data_checksum").get<std::string>()); else storage.bindNull(4);
        storage.bindBlob(5, parseUuid(id)); storage.stepDone();
    }
    for (const auto& [id, record] : merged.items()) if (!object(record) && record != at(before, id)) {
        auto edge = edgeFromJson(record);
        addEdgeImpl(db, store.author(), store.hostname(), edge, true, "sync");
        const auto actual = *store.edgeByUuid(txn, id);
        const bool swap = actual.aUuid != edge.aUuid;
        const auto epochs = record.value("lifecycle_bindings", Json::object());
        bindLifecycle(db, actual, epochs.value(swap ? "b" : "a", ""), epochs.value(swap ? "a" : "b", ""));
        const auto bindings = record.value("geometry_bindings", Json::object());
        const auto a = bindings.value(swap ? "b" : "a", Json()), b = bindings.value(swap ? "a" : "b", Json());
        Statement put(db, "UPDATE edge_geometry_bindings SET a_revision=?,b_revision=? WHERE edge_id=(SELECT id FROM edges WHERE uuid=?)");
        if (a.is_null()) put.bindNull(1); else put.bindBlob(1, parseUuid(a.get<std::string>()));
        if (b.is_null()) put.bindNull(2); else put.bindBlob(2, parseUuid(b.get<std::string>()));
        put.bindBlob(3, parseUuid(id)); put.stepDone();
    }
}
}

std::string ObjectStore::replicaUuid(const Txn& txn) const {
    Statement row(static_cast<sqlite3*>(connection(txn)), "SELECT uuid FROM sync_replica WHERE singleton=1");
    if (!row.row()) throw std::runtime_error("sync: missing replica identity");
    return row.text(0);
}
std::string ObjectStore::forkReplica() {
    auto txn = write(); auto* db = static_cast<sqlite3*>(connection(txn));
    const auto id = util::newUuid();
    Statement set(db, "UPDATE sync_replica SET uuid=? WHERE singleton=1"); set.bind(1, id); set.stepDone();
    exec(db, "DELETE FROM sync_baselines"); txn.commit(); return id;
}
SyncReport ObjectStore::synchronize(ObjectStore& peer, const SyncOptions& options)
{
    detail::DependencyScope dependencyScope;
    if (this == &peer || std::filesystem::equivalent(path(), peer.path()))
        throw std::invalid_argument("sync: source and peer are the same database");
    if (impl_->options.readOnly || peer.impl_->options.readOnly) throw std::invalid_argument("sync: both stores must be writable");
    // Consistent path ordering avoids deadlocks across handles and processes.
    const bool localFirst = std::filesystem::canonical(path()) < std::filesystem::canonical(peer.path());
    auto first = localFirst ? write() : peer.write();
    auto second = localFirst ? peer.write() : write();
    auto& localTxn = localFirst ? first : second;
    auto& remoteTxn = localFirst ? second : first;
    auto* localDb = static_cast<sqlite3*>(connection(localTxn));
    auto* remoteDb = static_cast<sqlite3*>(peer.connection(remoteTxn));
    SyncReport report;
    report.localReplica = replicaUuid(localTxn); report.remoteReplica = peer.replicaUuid(remoteTxn);
    if (report.localReplica == report.remoteReplica)
        throw std::runtime_error("sync: copied stores share a replica UUID; call fork_replica on the copy, then synchronize before editing either replica");
    const auto shared = sharedBaseline(localDb, remoteDb, report.remoteReplica, report.localReplica);
    const auto localCursor = syncCursor(localDb), remoteCursor = syncCursor(remoteDb);
    auto local = Json::object(), remote = Json::object(), base = Json::object();
    std::set<std::string> ids;
    if (shared.incremental) {
        if (shared.localCursor.journal > localCursor.journal || shared.remoteCursor.journal > remoteCursor.journal ||
            shared.localCursor.geometry > localCursor.geometry || shared.remoteCursor.geometry > remoteCursor.geometry)
            throw std::runtime_error("sync: baseline cursor is ahead of retained history");
        changedRecords(localDb, shared.localCursor, ids);
        changedRecords(remoteDb, shared.remoteCursor, ids);
        const auto changed = ids;
        for (const auto& [id, side] : options.resolutions) ids.insert(id);
        // This token covers all writes, including approvals and geometry history
        // received from a third replica, without serializing unchanged records.
        report.planToken = hash(Json{{"local_replica", report.localReplica}, {"remote_replica", report.remoteReplica},
            {"base_token", shared.token}, {"base", shared.state},
            {"local_cursor", {localCursor.journal, localCursor.geometry}},
            {"remote_cursor", {remoteCursor.journal, remoteCursor.geometry}}}.dump());
        std::set<std::string> loaded;
        while (loaded.size() != ids.size()) {
            const auto pending = ids;
            for (const auto& id : pending) {
                if (!loaded.insert(id).second) continue;
                const auto l = syncRecord(*this, localTxn, id), r = syncRecord(peer, remoteTxn, id);
                const auto b = baselineRecord(localDb, report.remoteReplica, shared.localSeq, id);
                if (b != baselineRecord(remoteDb, report.localReplica, shared.remoteSeq, id))
                    throw std::runtime_error("sync: inconsistent shared baseline record");
                if (!l.is_null()) local[id] = l;
                if (!r.is_null()) remote[id] = r;
                if (!b.is_null()) base[id] = b;
                for (const auto* record : {&l, &r, &b}) {
                    identityPeers(localDb, *record, ids); identityPeers(remoteDb, *record, ids);
                    if (!record->is_null() && !object(*record)) {
                        ids.insert(record->at("a").get<std::string>()); ids.insert(record->at("b").get<std::string>());
                    }
                }
                // Metadata-only edits need no graph walk. Endpoint deletions or
                // geometry changes must check even unchanged incident edges.
                if (changed.contains(id) && (object(l) || object(r) || object(b)) &&
                    (l.is_null() || r.is_null() || head(l) != head(b) || head(r) != head(b) ||
                     l.value("lifecycle_epoch", "") != (b.is_object() ? b.value("lifecycle_epoch", "") : "") ||
                     r.value("lifecycle_epoch", "") != (b.is_object() ? b.value("lifecycle_epoch", "") : ""))) {
                    for (auto* db : {localDb, remoteDb}) {
                        Statement edges(db, "SELECT uuid FROM edges WHERE a_id=(SELECT id FROM objects WHERE uuid=?1) "
                                            "OR b_id=(SELECT id FROM objects WHERE uuid=?1)");
                        edges.bindBlob(1, parseUuid(id));
                        while (edges.row()) ids.insert(formatUuid(edges.blob16(0)));
                    }
                }
            }
        }
    } else {
        local = snapshot(*this, localTxn); remote = snapshot(peer, remoteTxn);
        for (const auto* state : {&base, &local, &remote}) for (const auto& item : state->items()) ids.insert(item.key());
        report.planToken = hash(Json{{"local_replica", report.localReplica}, {"remote_replica", report.remoteReplica},
            {"local", local}, {"remote", remote}, {"base", base}}.dump());
    }
    if ((!options.resolutions.empty() && !options.expectedPlan) || (options.expectedPlan && *options.expectedPlan != report.planToken))
        throw std::runtime_error("sync: conflict resolutions require the current preview's expected_plan token");
    for (const auto& [id, side] : options.resolutions)
        if ((!local.contains(id) && !remote.contains(id) && !base.contains(id)) || (side != "local" && side != "remote"))
            throw std::invalid_argument("sync: invalid record resolution");
    auto merged = Json::object();
    auto conflict = [&](const std::string& id, const std::string& reason) {
        report.conflicts.push_back({id, reason, at(base, id).dump(), at(local, id).dump(), at(remote, id).dump()});
    };
    for (const auto& id : ids) {
        const auto b = at(base, id), l = at(local, id), r = at(remote, id);
        Json chosen;
        if (options.resolutions.contains(id)) chosen = options.resolutions.at(id) == "local" ? l : r;
        else if (l == r) chosen = l;
        else if (l == b) chosen = r;
        else if (r == b) chosen = l;
        else { conflict(id, "both replicas changed this record"); continue; }
        if (!chosen.is_null()) merged[id] = chosen;
    }
    if (!report.conflicts.empty()) return report;
    // Validate dependencies before making any writes, including conflicts arising
    // from otherwise independent endpoint deletions and edge additions.
    std::map<std::string, std::string> identities;
    for (const auto& id : ids) {
        const auto chosen = at(merged, id);
        for (const auto* side : {&local, &remote}) {
            const auto old = at(*side, id);
            if (!old.is_null() && !chosen.is_null() && object(old) != object(chosen)) conflict(id, "object/edge UUID collision");
            if (!object(old)) continue;
            if (object(chosen) && old.at("kind") != chosen.at("kind"))
                conflict(id, "object kind cannot change; assign distinct UUIDs before synchronizing");
            if (head(old) != nullptr && (chosen.is_null() || head(chosen) == nullptr))
                conflict(id, "retained geometry cannot be deleted; select the replica containing geometry");

        }
        if (chosen.is_null()) continue;
        if (object(chosen)) {
            if (chosen.contains("track") && chosen["track"]["external_key"] != "") {
                const auto identity = Json::array({"track", chosen["track"]["dataset_ref"], chosen["track"]["external_key"]}).dump();
                auto [it, added] = identities.emplace(identity, id);
                if (!added) conflict(id, "track identity also belongs to " + it->second);
            }
        } else {
            for (const auto* endpoint : {"a", "b"}) {
                const auto target = at(merged, chosen.at(endpoint).get<std::string>());
                if (!object(target)) conflict(id, "edge endpoint was deleted or is not an object");
            }
            const auto identity = edgeIdentity(edgeFromJson(chosen));
            auto [it, added] = identities.emplace(identity, id);
            if (!added) conflict(id, "edge identity also belongs to " + it->second + "; resolve one duplicate to its absent side");
        }
    }
    if (!report.conflicts.empty()) return report;
    // Synchronization must not turn unresolved geometry/relationship combinations
    // into stale links implicitly. A source edit must carry its explicit decision.
    auto geometryFor = [&](const Json& uuid) -> std::optional<detail::DependencyGeometry> {
        if (uuid.is_null()) return std::nullopt;
        const auto id = uuid.get<std::string>();
        if (const auto revision = geometryRevision(localTxn, id))
            return detail::dependencyGeometry(*this, localTxn, *revision);
        if (const auto revision = peer.geometryRevision(remoteTxn, id))
            return detail::dependencyGeometry(peer, remoteTxn, *revision);
        return std::nullopt;
    };
    for (const auto& [id, edge] : merged.items()) if (!object(edge)) {
        const auto binding = edge.value("geometry_bindings", Json::object());
        for (const auto* endpoint : {"a", "b"}) {
            const auto bound = binding.value(endpoint, Json());
            const auto current = head(merged.at(edge.at(endpoint).get<std::string>()));
            if (bound == current) continue;
            const auto before = geometryFor(bound), after = geometryFor(current);
            const auto anchor = anchorFromJson(edge.at(std::string(endpoint) + "_anchor"));
            if (!before || !after || detail::dependenciesChanged(anchor, *before, *after))
                conflict(id, "geometry requires an explicit relationship retain/update/remove decision before synchronization");
        }
    }
    if (!report.conflicts.empty()) return report;
    for (const auto& id : ids) {
        if (at(local, id) != at(merged, id)) ++report.localChanges;
        if (at(remote, id) != at(merged, id)) ++report.remoteChanges;
    }
    if (options.dryRun) return report;
    if (shared.incremental && ids.empty()) {
        first.commit(); second.commit(); report.applied = true; return report;
    }
    applyObjects(*this, localTxn, local, merged); applyObjects(peer, remoteTxn, remote, merged);
    for (const auto& [id, record] : merged.items()) if (object(record)) {
        detail::transferGeometry(localDb, remoteDb, id, shared.incremental ? shared.localCursor.geometry : 0);
        detail::transferGeometry(remoteDb, localDb, id, shared.incremental ? shared.remoteCursor.geometry : 0);
    }
    applyHeadsAndEdges(*this, localTxn, local, merged); applyHeadsAndEdges(peer, remoteTxn, remote, merged);
    // Verify exactly the records in the merge and its dependency closure.
    for (const auto& id : ids)
        if (syncRecord(*this, localTxn, id) != at(merged, id) || syncRecord(peer, remoteTxn, id) != at(merged, id))
            throw std::runtime_error("sync: applied state differs from merge; both transactions rolled back");
    // Validate/materialize both local history publications before either store
    // commits. Sync merges current records, not the peer's historical timeline.
    detail::publishHistory(localDb, true);
    detail::publishHistory(remoteDb, true);
    std::string digest = shared.incremental ? shared.state.at("digest").get<std::string>() : std::string(64, '0');
    if (shared.incremental) {
        for (const auto& [id, record] : base.items()) toggleDigest(digest, record);
    }
    for (const auto& [id, record] : merged.items()) toggleDigest(digest, record);
    const auto token = util::newUuid();
    saveBaseline(localDb, report.remoteReplica, token, digest, merged, ids, !shared.incremental);
    saveBaseline(remoteDb, report.localReplica, token, digest, merged, ids, !shared.incremental);
    // SQLite cannot atomically commit two WAL databases. Keep historical baselines
    // in both files so a retry after the first commit uses their last common state.
    first.commit();
    try { second.commit(); }
    catch (...) { throw std::runtime_error("sync: first replica committed but second commit failed; retry synchronize to recover from the retained shared baseline"); }
    report.applied = true;
    return report;
}
} // namespace vc::objects
