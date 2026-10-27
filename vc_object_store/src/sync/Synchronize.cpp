#include "geometry/Dependencies.hpp"
#include "internal/RecordJson.hpp"
#include "internal/EdgeSql.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/GeometryDelta.hpp"
#include "sync/GeometryTransfer.hpp"
#include "sync/Baselines.hpp"
#include "internal/History.hpp"
#include "sync/Apply.hpp"

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
}

std::string ObjectStore::replicaUuid(const Txn& txn) const {
    Statement row(static_cast<sqlite3*>(connection(txn)), "SELECT uuid FROM sync_replica WHERE singleton=1");
    if (!row.row()) throw std::runtime_error("sync: missing replica identity");
    return row.text(0);
}
std::string ObjectStore::forkReplica() {
    auto txn = write(); auto* db = static_cast<sqlite3*>(connection(txn));
    const auto id = util::newUuid();
    execute(db, "UPDATE sync_replica SET uuid=? WHERE singleton=1", id);
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
                    if (!record->is_null() && !isObjectRecord(*record)) {
                        ids.insert(record->at("a").get<std::string>()); ids.insert(record->at("b").get<std::string>());
                    }
                }
                // Metadata-only edits need no graph walk. Endpoint deletions or
                // geometry changes must check even unchanged incident edges.
                if (changed.contains(id) && (isObjectRecord(l) || isObjectRecord(r) || isObjectRecord(b)) &&
                    (l.is_null() || r.is_null() || recordHead(l) != recordHead(b) || recordHead(r) != recordHead(b) ||
                     l.value("lifecycle_epoch", "") != (b.is_object() ? b.value("lifecycle_epoch", "") : "") ||
                     r.value("lifecycle_epoch", "") != (b.is_object() ? b.value("lifecycle_epoch", "") : ""))) {
                    for (auto* db : {localDb, remoteDb}) incidentEdges(db, id, ids);
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
        report.conflicts.push_back({id, reason, recordAt(base, id).dump(), recordAt(local, id).dump(), recordAt(remote, id).dump()});
    };
    for (const auto& id : ids) {
        const auto b = recordAt(base, id), l = recordAt(local, id), r = recordAt(remote, id);
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
        const auto chosen = recordAt(merged, id);
        for (const auto* side : {&local, &remote}) {
            const auto old = recordAt(*side, id);
            if (!old.is_null() && !chosen.is_null() && isObjectRecord(old) != isObjectRecord(chosen)) conflict(id, "object/edge UUID collision");
            if (!isObjectRecord(old)) continue;
            if (isObjectRecord(chosen) && old.at("kind") != chosen.at("kind"))
                conflict(id, "object kind cannot change; assign distinct UUIDs before synchronizing");
            if (recordHead(old) != nullptr && (chosen.is_null() || recordHead(chosen) == nullptr))
                conflict(id, "retained geometry cannot be deleted; select the replica containing geometry");

        }
        if (chosen.is_null()) continue;
        if (isObjectRecord(chosen)) {
            if (chosen.contains("track") && chosen["track"]["external_key"] != "") {
                const auto identity = Json::array({"track", chosen["track"]["dataset_ref"], chosen["track"]["external_key"]}).dump();
                auto [it, added] = identities.emplace(identity, id);
                if (!added) conflict(id, "track identity also belongs to " + it->second);
            }
        } else {
            for (const auto* endpoint : {"a", "b"}) {
                const auto target = recordAt(merged, chosen.at(endpoint).get<std::string>());
                if (!isObjectRecord(target)) conflict(id, "edge endpoint was deleted or is not an object");
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
    for (const auto& id : undecidedRelationships(merged, geometryFor))
        conflict(id, "geometry requires an explicit relationship retain/update/remove decision before synchronization");
    if (!report.conflicts.empty()) return report;
    for (const auto& id : ids) {
        if (recordAt(local, id) != recordAt(merged, id)) ++report.localChanges;
        if (recordAt(remote, id) != recordAt(merged, id)) ++report.remoteChanges;
    }
    if (options.dryRun) return report;
    if (shared.incremental && ids.empty()) {
        first.commit(); second.commit(); report.applied = true; return report;
    }
    applyObjects(*this, localTxn, local, merged); applyObjects(peer, remoteTxn, remote, merged);
    for (const auto& [id, record] : merged.items()) if (isObjectRecord(record)) {
        detail::transferGeometry(localDb, remoteDb, id, shared.incremental ? shared.localCursor.geometry : 0);
        detail::transferGeometry(remoteDb, localDb, id, shared.incremental ? shared.remoteCursor.geometry : 0);
    }
    applyHeadsAndEdges(*this, localTxn, local, merged); applyHeadsAndEdges(peer, remoteTxn, remote, merged);
    // Verify exactly the records in the merge and its dependency closure.
    for (const auto& id : ids)
        if (syncRecord(*this, localTxn, id) != recordAt(merged, id) || syncRecord(peer, remoteTxn, id) != recordAt(merged, id))
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
