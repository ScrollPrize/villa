#pragma once
#include "internal/RecordJson.hpp"
#include "geometry/GeometryInternal.hpp"

namespace vc::objects::detail {
using Json = nlohmann::json;

struct SyncCursor { int64_t journal = 0, geometry = 0; };
SyncCursor syncCursor(sqlite3* db) {
    Statement row(db, "SELECT coalesce((SELECT max(seq) FROM journal),0),"
                      "coalesce((SELECT max(rowid) FROM geometry_revisions),0)");
    (void)row.row(); return {row.i64(0), row.i64(1)};
}

struct SharedBaseline {
    std::string token;
    Json state = Json::object();
    int64_t localSeq = 0, remoteSeq = 0;
    SyncCursor localCursor, remoteCursor;
    bool incremental = false;
};

SharedBaseline sharedBaseline(sqlite3* local, sqlite3* remote, const std::string& localPeer, const std::string& remotePeer) {
    SharedBaseline result;
    Statement ours(local, "SELECT seq,token FROM sync_baselines WHERE peer=? ORDER BY seq DESC");
    ours.bind(1, localPeer);
    Statement theirs(remote, "SELECT seq,state FROM sync_baselines WHERE peer=? AND token=?");
    while (ours.row()) {
        theirs.reset(); theirs.bind(1, remotePeer); theirs.bind(2, ours.text(1));
        if (!theirs.row()) continue;
        result.localSeq = ours.i64(0); result.remoteSeq = theirs.i64(0); result.token = ours.text(1);
        Statement state(local, "SELECT state FROM sync_baselines WHERE seq=?");
        state.bind(1, result.localSeq); (void)state.row();
        result.state = Json::parse(state.text(0));
        if (result.state != Json::parse(theirs.text(1))) throw std::runtime_error("sync: inconsistent shared baseline");
        // Legacy version-2 snapshots have no cursor. One full merge establishes
        // the new representation; old tokens remain available for recovery.
        if (!result.state.contains("digest")) break;
        if (!result.state.at("digest").is_string()) throw std::runtime_error("sync: invalid baseline digest");
        const auto digest = result.state.at("digest").get<std::string>();
        if (digest.size() != 64 || digest.find_first_not_of("0123456789abcdef") != std::string::npos)
            throw std::runtime_error("sync: invalid baseline digest");
        auto cursor = [](sqlite3* db, int64_t seq) {
            Statement row(db, "SELECT journal_seq,geometry_seq FROM sync_baseline_cursors WHERE baseline_seq=?");
            row.bind(1, seq);
            if (!row.row()) throw std::runtime_error("sync: missing baseline cursor");
            return SyncCursor{row.i64(0), row.i64(1)};
        };
        result.localCursor = cursor(local, result.localSeq); result.remoteCursor = cursor(remote, result.remoteSeq);
        result.incremental = true;
        break;
    }
    return result;
}

Json baselineRecord(sqlite3* db, const std::string& peer, int64_t seq, const std::string& id) {
    Statement row(db, "SELECT state FROM sync_baseline_records WHERE peer=? AND record_uuid=? AND baseline_seq<=? "
                      "ORDER BY baseline_seq DESC LIMIT 1");
    row.bind(1, peer); row.bind(2, id); row.bind(3, seq);
    return row.row() ? Json::parse(row.text(0)) : Json();
}

void changedRecords(sqlite3* db, SyncCursor after, std::set<std::string>& ids) {
    Statement rows(db, "SELECT DISTINCT record_uuid FROM journal WHERE seq>? UNION "
                       "SELECT object_uuid FROM geometry_revisions WHERE rowid>?");
    rows.bind(1, after.journal); rows.bind(2, after.geometry);
    while (rows.row()) ids.insert(formatUuid(rows.blob16(0)));
}

Json syncRecord(ObjectStore& store, const Txn& txn, const std::string& id) {
    if (const auto object = store.objectByUuid(txn, id)) {
        auto record = objectToJson(*object); record["record_type"] = "objects";
        if (const auto geometry = store.currentGeometry(txn, id))
            record["geometry"] = {{"revision_uuid", geometry->uuid}, {"content_hash", geometry->contentHash}};
        return record;
    }
    if (const auto edge = store.edgeByUuid(txn, id)) {
        auto record = edgeToJson(*edge); record["record_type"] = "edges";
        Statement binding(ObjectStoreImplAccess::conn(txn), "SELECT a_revision,b_revision FROM edge_geometry_bindings "
                                                          "WHERE edge_id=(SELECT id FROM edges WHERE uuid=?)");
        binding.bindBlob(1, parseUuid(id));
        if (!binding.row()) throw std::runtime_error("sync: missing edge geometry binding");
        const bool swap = record["a"] != edge->aUuid;
        auto revision = [&](int column) -> Json { return binding.isNull(column) ? Json() : Json(formatUuid(binding.blob16(column))); };
        record["geometry_bindings"] = {{"a", revision(swap ? 1 : 0)}, {"b", revision(swap ? 0 : 1)}};
        return record;
    }
    return nullptr;
}

// Include competing identities even if that competing record did not change.
void identityPeers(sqlite3* db, const Json& record, std::set<std::string>& ids) {
    if (record.is_null()) return;
    if (record.at("record_type") == "objects") {
        if (!record.contains("track") || record["track"]["external_key"] == "") return;
        Statement rows(db, "SELECT o.uuid FROM tracks t JOIN objects o ON o.id=t.object_id WHERE dataset_ref=? AND external_key=?");
        rows.bind(1, record["track"]["dataset_ref"].get<std::string>());
        rows.bind(2, record["track"]["external_key"].get<std::string>());
        while (rows.row()) ids.insert(formatUuid(rows.blob16(0)));
    } else {
        auto a = objectIdByUuid(db, parseUuid(record.at("a"))), b = objectIdByUuid(db, parseUuid(record.at("b")));
        if (!a || !b) return;
        auto aa = dependenciesToJson(anchorFromJson(record.at("a_anchor"))).dump();
        auto ba = dependenciesToJson(anchorFromJson(record.at("b_anchor"))).dump();
        if (*a > *b || (*a == *b && ba < aa)) { std::swap(a, b); std::swap(aa, ba); }
        Statement rows(db, "SELECT uuid FROM edges WHERE kind=? AND a_id=? AND b_id=? AND a_dependencies=? AND b_dependencies=?");
        rows.bind(1, record.at("kind").get<std::string>()); rows.bind(2, *a); rows.bind(3, *b); rows.bind(4, aa); rows.bind(5, ba);
        while (rows.row()) ids.insert(formatUuid(rows.blob16(0)));
    }
}

// An order-independent fingerprint lets both replicas compare a complete
// baseline without serializing it. Each hashed record includes its UUID.
void toggleDigest(std::string& digest, const Json& record) {
    if (record.is_null()) return;
    const auto text = record.dump();
    const auto hash = detail::geometryHash(std::span(reinterpret_cast<const uint8_t*>(text.data()), text.size()));
    constexpr const char* hex = "0123456789abcdef";
    for (size_t i = 0; i < digest.size(); ++i) {
        auto digit = [](char ch) { return ch <= '9' ? ch - '0' : ch - 'a' + 10; };
        digest[i] = hex[digit(digest[i]) ^ digit(hash[i])];
    }
}

void saveBaseline(sqlite3* db, const std::string& peer, const std::string& token, const std::string& digest,
                  const Json& merged, std::set<std::string> ids, bool full) {
    Statement latest(db, "SELECT coalesce(max(seq),0) FROM sync_baselines WHERE peer=?");
    latest.bind(1, peer); (void)latest.row(); const auto previous = latest.i64(0);
    if (full) {
        Statement known(db, "SELECT DISTINCT record_uuid FROM sync_baseline_records WHERE peer=?");
        known.bind(1, peer); while (known.row()) ids.insert(known.text(0));
    }
    Statement row(db, "INSERT INTO sync_baselines(peer,token,state) VALUES(?,?,?)");
    row.bind(1, peer); row.bind(2, token); row.bind(3, Json{{"digest", digest}}.dump()); row.stepDone();
    const auto seq = sqlite3_last_insert_rowid(db);
    Statement change(db, "INSERT INTO sync_baseline_records VALUES(?,?,?,?)");
    for (const auto& id : ids) {
        const auto value = merged.value(id, Json());
        if (value == baselineRecord(db, peer, previous, id)) continue;
        change.reset(); change.bind(1, peer); change.bind(2, id); change.bind(3, int64_t(seq)); change.bind(4, value.dump()); change.stepDone();
    }
    const auto cursor = syncCursor(db);
    Statement mark(db, "INSERT INTO sync_baseline_cursors VALUES(?,?,?)");
    mark.bind(1, int64_t(seq)); mark.bind(2, cursor.journal); mark.bind(3, cursor.geometry); mark.stepDone();
}
} // namespace vc::objects::detail
