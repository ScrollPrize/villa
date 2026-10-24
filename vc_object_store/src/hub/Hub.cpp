#include "vc_object_store/objects/Hub.hpp"
#include "hub/HubRules.hpp"
#include "sync/Apply.hpp"
#include "sync/GeometryTransfer.hpp"
#include "internal/History.hpp"
#include <thread>

namespace vc::objects {
using namespace detail;
namespace {
struct Upstream {
    std::filesystem::path hub;
    std::string hubReplica;
    SyncCursor cursor;
};

sqlite3* conn(const Txn& txn) { return ObjectStoreImplAccess::conn(txn); }
StatementPool& pool(const Txn& txn) { return *activeStatementPools.at(conn(txn)); }

Upstream upstream(sqlite3* db)
{
    if (detectHubRole(db) != HubRole::Replica) throw std::runtime_error("hub: store is not a hosted replica");
    Statement row(db, "SELECT hub_path,hub_replica,journal_cursor,geometry_cursor FROM hub_upstream");
    if (!row.row()) throw std::runtime_error("hub: missing upstream");
    return {row.text(0), row.text(1), {row.i64(2), row.i64(3)}};
}

ObjectStore openHub(const ObjectStore& replica, const Txn& txn, const Upstream& up)
{
    OpenOptions options;
    options.author = replica.author();
    options.sharedFile = true;
    // Other clients hold the dot-file lock for whole syncs, not single statements.
    options.busyTimeoutMs = 60000;
    auto hub = ObjectStore::open(up.hub, options);
    {
        auto check = hub.read();
        if (hub.uuid(check) != replica.uuid(txn) || hub.replicaUuid(check) != up.hubReplica)
            throw std::runtime_error("hub: " + up.hub.string() + " is not this replica's hub");
    }
    return hub;
}

std::string hash(const Json& record)
{
    const auto text = record.dump();
    return geometryHash(std::span(reinterpret_cast<const uint8_t*>(text.data()), text.size()));
}

std::vector<HubLease> readLeases(sqlite3* db, const std::optional<std::string>& holder = std::nullopt)
{
    std::vector<HubLease> result;
    Statement rows(db, "SELECT object_uuid,holder,author,host,acquired_us,expires_us FROM hub_leases ORDER BY object_uuid");
    while (rows.row())
        if (!holder || rows.text(1) == *holder)
            result.push_back({rows.text(0), rows.text(1), rows.text(2), rows.text(3), rows.i64(4), rows.i64(5)});
    return result;
}

void writeLeases(sqlite3* db, const std::vector<HubLease>& leases)
{
    Statement put(db, "INSERT INTO hub_leases VALUES(?,?,?,?,?,?) ON CONFLICT(object_uuid) DO UPDATE SET "
                      "holder=excluded.holder,author=excluded.author,host=excluded.host,"
                      "acquired_us=excluded.acquired_us,expires_us=excluded.expires_us");
    for (const auto& lease : leases)
        put.run(lease.objectUuid, lease.holder, lease.author, lease.host, lease.acquiredUs, lease.expiresUs);
}

std::set<std::string> uuidColumn(sqlite3* db, const char* sql)
{
    std::set<std::string> ids;
    Statement rows(db, sql);
    while (rows.row()) ids.insert(rows.text(0));
    return ids;
}

std::vector<HubConflict> conflicts(ObjectStore& replica, const Txn& txn)
{
    std::vector<HubConflict> result;
    Statement rows(conn(txn), "SELECT record_uuid,version,record FROM hub_conflicts ORDER BY record_uuid");
    while (rows.row())
        result.push_back({rows.text(0), rows.i64(1), rows.text(2), syncRecord(replica, txn, rows.text(0)).dump()});
    return result;
}

// Install hub records for ids in one replica write. hubRecords holds the hub's
// sync JSON (null for absent records); geometry is copied from afterRow.
void installHubRecords(ObjectStore& replica, WriteTxn& txn, ObjectStore& hub, const ReadTxn& hubTxn,
                       const std::map<std::string, Json>& hubRecords, int64_t afterRow)
{
    auto before = Json::object(), merged = Json::object();
    for (const auto& [id, record] : hubRecords) {
        const auto local = syncRecord(replica, txn, id);
        if (!local.is_null()) before[id] = local;
        if (!record.is_null()) merged[id] = record;
    }
    applyObjects(replica, txn, before, merged);
    for (const auto& [id, record] : merged.items())
        if (isObjectRecord(record)) transferGeometry(conn(hubTxn), conn(txn), id, afterRow);
    applyHeadsAndEdges(replica, txn, before, merged);
    for (const auto& [id, record] : hubRecords)
        if (syncRecord(replica, txn, id) != record) throw std::runtime_error("hub: installed state differs from the hub for " + id);
    publishHistory(conn(txn), true);
}

std::vector<HubLease> grantLeases(ObjectStore& replica, const std::vector<std::string>& objects, int64_t seconds, bool fresh)
{
    if (seconds <= 0) throw std::invalid_argument("hub: lease duration must be positive");
    std::string self;
    Upstream up;
    std::map<std::string, int64_t> known;
    std::set<std::string> pending;
    auto local = replica.read();
    self = replicaIdentity(conn(local)); up = upstream(conn(local));
    for (const auto& object : objects) known[formatUuid(parseUuid(object))] = knownHubVersion(conn(local), object);
    pending = uuidColumn(conn(local), "SELECT record_uuid FROM hub_pending");
    auto hub = openHub(replica, local, up);
    std::vector<HubLease> granted;
    {
        auto txn = hub.write();
        const auto now = nowMicros();
        std::string message = "hub: cannot lease";
        std::vector<std::string> held, stale;
        for (const auto& [object, version] : known) {
            Statement row(conn(txn), "SELECT holder,author,host,expires_us FROM hub_leases WHERE object_uuid=?", object);
            if (row.row() && row.text(0) != self && row.i64(3) > now) {
                held.push_back(object);
                message += "\n  " + object + " is leased to " + row.text(1) + "@" + row.text(2) + " until " + isoFromMicros(row.i64(3));
                continue;
            }
            // Editing must start from the hub's current record, unless this
            // replica created it and only waits to push it.
            const auto current = hubVersion(conn(txn), object);
            if (fresh && current != version && !(current == 0 && pending.contains(object))) {
                stale.push_back(object);
                message += "\n  " + object + " changed on the hub; sync before editing it";
            }
        }
        if (!held.empty() || !stale.empty()) {
            HubLeaseRefused refused(message);
            refused.held = std::move(held); refused.stale = std::move(stale);
            throw refused;
        }
        for (const auto& [object, version] : known)
            granted.push_back({object, self, replica.author(), replica.hostname(), now, now + seconds * 1000000});
        writeLeases(conn(txn), granted);
        txn.commit();
    }
    local.commit();
    auto mirror = replica.write();
    pool(mirror).hubApply = true;
    writeLeases(conn(mirror), granted);
    mirror.commit();
    return granted;
}
}

void initHub(const std::filesystem::path& path, const std::string& author)
{
    OpenOptions options; options.author = author;
    {
        // The ordinary open validates the format and schema.
        auto check = options; check.readOnly = true;
        auto store = ObjectStore::open(path, check);
    }
    auto* db = openConnection(path, SQLITE_OPEN_READWRITE, options, false);
    try {
        if (detectHubRole(db) != HubRole::None) throw std::runtime_error("hub: store is already a hub or a replica");
        {
            Statement mode(db, "PRAGMA journal_mode=DELETE");
            if (!mode.row() || mode.text(0) != "delete")
                throw std::runtime_error("hub: cannot leave WAL mode; close every other connection to " + path.string());
        }
        exec(db, "BEGIN IMMEDIATE;");
        try { createHubTables(db); exec(db, "COMMIT;"); }
        catch (...) { sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr); throw; }
    } catch (...) { sqlite3_close(db); throw; }
    sqlite3_close(db);
}

void cloneHub(const std::filesystem::path& hubPath, const std::filesystem::path& replicaPath, const OpenOptions& options,
              const util::ProgressCallback& progress)
{
    if (std::filesystem::exists(replicaPath)) throw std::runtime_error("hub: clone refuses existing " + replicaPath.string());
    const auto hub = std::filesystem::absolute(hubPath);
    auto partial = replicaPath; partial += ".partial";
    std::filesystem::remove(partial);
    auto shared = options; shared.sharedFile = true;
    auto* source = openConnection(hub, SQLITE_OPEN_READONLY, shared, false);
    sqlite3* target = nullptr;
    auto close = [&] { if (target) sqlite3_close(target); sqlite3_close(source); target = nullptr; };
    try {
        if (pragmaInt(source, "application_id") != 0x56434f42 || detectHubRole(source) != HubRole::Hub)
            throw std::runtime_error("hub: not a hosted hub: " + hub.string());
        target = openConnection(partial, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, options, false);
        // Each step holds the hub lock briefly; a concurrent hub write restarts the copy.
        auto* backup = sqlite3_backup_init(target, "main", source, "main");
        if (!backup) fail(target, "hub: clone");
        int rc = SQLITE_OK;
        while ((rc = sqlite3_backup_step(backup, 1024)) == SQLITE_OK || rc == SQLITE_BUSY || rc == SQLITE_LOCKED) {
            const auto total = sqlite3_backup_pagecount(backup);
            util::reportProgress(progress, "copy hub", total - sqlite3_backup_remaining(backup), total);
            if (rc != SQLITE_OK) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        const auto pages = sqlite3_backup_pagecount(backup);
        sqlite3_backup_finish(backup);
        if (rc != SQLITE_DONE) fail(target, "hub: clone");
        util::reportProgress(progress, "copy hub", pages, pages);
        exec(target, "PRAGMA journal_mode=WAL;");
        exec(target, "BEGIN IMMEDIATE;");
        try {
            createReplicaTables(target);
            const auto cursor = syncCursor(target);
            execute(target, "INSERT INTO hub_upstream VALUES(1,?,?,?,?,?)", hub.string(), replicaIdentity(target),
                cursor.journal, cursor.journal, cursor.geometry);
            exec(target, "COMMIT;");
        } catch (...) { sqlite3_exec(target, "ROLLBACK;", nullptr, nullptr, nullptr); throw; }
        close();
    } catch (...) {
        close();
        std::error_code ignored;
        std::filesystem::remove(partial, ignored);
        throw;
    }
    try {
        // The copy needs its own replica identity before anything is edited.
        auto replica = ObjectStore::open(partial, options);
        replica.forkReplica();
        replica.checkpoint();
    } catch (...) {
        std::error_code ignored;
        for (const auto* suffix : {"", "-wal", "-shm"}) std::filesystem::remove(partial.string() + suffix, ignored);
        throw;
    }
    std::filesystem::rename(partial, replicaPath);
}

void detachHub(const std::filesystem::path& path, const std::string& author)
{
    OpenOptions options; options.author = author;
    auto* db = openConnection(path, SQLITE_OPEN_READWRITE, options, false);
    try {
        if (detectHubRole(db) != HubRole::Replica) throw std::runtime_error("hub: store is not a hosted replica");
        exec(db, "BEGIN IMMEDIATE;");
        try {
            for (const auto* table : {"hub_upstream", "hub_versions", "hub_pending", "hub_inflight", "hub_conflicts", "hub_leases"})
                exec(db, (std::string("DROP TABLE ") + table + ";").c_str());
            exec(db, "COMMIT;");
        } catch (...) { sqlite3_exec(db, "ROLLBACK;", nullptr, nullptr, nullptr); throw; }
    } catch (...) { sqlite3_close(db); throw; }
    sqlite3_close(db);
}

HubSyncReport hubPull(ObjectStore& replica)
{
    DependencyScope dependencyScope;
    HubSyncReport report;
    Upstream up;
    std::optional<ObjectStore> hub;
    {
        auto local = replica.read();
        up = upstream(conn(local));
        hub.emplace(openHub(replica, local, up));
    }
    // The hub read lock is held until the replica commits, so the installed
    // state and the advanced cursor describe one hub snapshot.
    auto hubTxn = hub->read();
    const auto cursor = syncCursor(conn(hubTxn));
    std::set<std::string> ids;
    changedRecords(conn(hubTxn), up.cursor, ids);
    std::map<std::string, Json> hubRecords;
    std::map<std::string, int64_t> versions;
    for (const auto& id : ids) {
        hubRecords[id] = syncRecord(*hub, hubTxn, id);
        versions[id] = hubVersion(conn(hubTxn), id);
    }
    const auto leases = readLeases(conn(hubTxn));

    auto txn = replica.write();
    pool(txn).hubApply = true;
    auto* db = conn(txn);
    const auto pending = uuidColumn(db, "SELECT record_uuid FROM hub_pending UNION SELECT record_uuid FROM hub_conflicts");
    std::map<std::string, Json> install;
    Statement conflict(db, "INSERT INTO hub_conflicts VALUES(?,?,?) ON CONFLICT(record_uuid) DO UPDATE SET "
                           "version=excluded.version,record=excluded.record");
    Statement version(db, "INSERT INTO hub_versions VALUES(?,?) ON CONFLICT(record_uuid) DO UPDATE SET version=excluded.version");
    for (const auto& [id, record] : hubRecords) {
        const auto local = syncRecord(replica, txn, id);
        if (local == record) { version.run(id, versions.at(id)); continue; }
        if (pending.contains(id)) {
            // Our own earlier push: the hub has nothing newer than our base.
            if (versions.at(id) == knownHubVersion(db, id)) continue;
            conflict.run(id, versions.at(id), record.dump());
            report.conflicts.push_back({id, versions.at(id), record.dump(), local.dump()});
            continue;
        }
        install[id] = record;
        version.run(id, versions.at(id));
    }
    installHubRecords(replica, txn, *hub, hubTxn, install, up.cursor.geometry);
    execute(db, "UPDATE hub_upstream SET journal_cursor=?,geometry_cursor=?", cursor.journal, cursor.geometry);
    exec(db, "DELETE FROM hub_leases;");
    writeLeases(db, leases);
    txn.commit();
    report.pulled = int64_t(install.size());
    return report;
}

HubSyncReport hubPush(ObjectStore& replica)
{
    DependencyScope dependencyScope;
    HubSyncReport report;
    struct Change { Json record; int64_t counter = 0, geometryAfter = 0, known = 0; std::optional<std::string> inflight; };
    std::map<std::string, Change> changes;
    // One replica snapshot supplies records, geometry and counters.
    auto local = replica.read();
    auto* db = conn(local);
    const auto up = upstream(db);
    const auto self = replicaIdentity(db);
    if (uuidColumn(db, "SELECT record_uuid FROM hub_conflicts").size())
        throw std::runtime_error("hub: resolve conflicts before pushing");
    {
        Statement rows(db, "SELECT p.record_uuid,p.counter,p.geometry_after,i.record_hash FROM hub_pending p "
                           "LEFT JOIN hub_inflight i ON i.record_uuid=p.record_uuid");
        while (rows.row()) {
            Change change{syncRecord(replica, local, rows.text(0)), rows.i64(1), rows.i64(2),
                          knownHubVersion(db, rows.text(0)), rows.optText(3)};
            changes.emplace(rows.text(0), std::move(change));
        }
    }
    if (changes.empty()) return report;
    {
        // Lets a retry recognize this push on the hub if its outcome is lost.
        auto mark = replica.write();
        pool(mark).hubApply = true;
        Statement put(conn(mark), "INSERT INTO hub_inflight VALUES(?,?) ON CONFLICT(record_uuid) DO UPDATE SET record_hash=excluded.record_hash");
        for (const auto& [id, change] : changes) put.run(id, hash(change.record));
        mark.commit();
    }
    auto hub = openHub(replica, local, up);
    std::map<std::string, int64_t> published;
    std::vector<HubLease> created;
    {
        auto txn = hub.write();
        pool(txn).actingHolder = self;
        auto* hubDb = conn(txn);
        const auto now = nowMicros();
        auto before = Json::object(), merged = Json::object();
        std::set<std::string> apply;
        std::vector<std::string> unleased;
        for (const auto& [id, change] : changes) {
            const auto current = syncRecord(hub, txn, id);
            const auto version = hubVersion(hubDb, id);
            if (current == change.record) { published[id] = version; continue; }
            if (version != change.known && change.inflight != hash(current)) {
                report.conflicts.push_back({id, version, current.dump(), change.record.dump()});
                continue;
            }
            if (!current.is_null() && isObjectRecord(current)) {
                Statement lease(hubDb, "SELECT holder,expires_us FROM hub_leases WHERE object_uuid=?", id);
                if (!lease.row() || lease.text(0) != self || lease.i64(1) <= now) unleased.push_back(id);
            }
            if (!current.is_null()) before[id] = current;
            if (!change.record.is_null()) merged[id] = change.record;
            apply.insert(id);
        }
        if (!report.conflicts.empty()) {
            auto mark = replica.write();
            pool(mark).hubApply = true;
            Statement put(conn(mark), "INSERT INTO hub_conflicts VALUES(?,?,?) ON CONFLICT(record_uuid) DO UPDATE SET "
                                      "version=excluded.version,record=excluded.record");
            for (const auto& conflict : report.conflicts) put.run(conflict.uuid, conflict.hubVersion, conflict.hubRecord);
            mark.commit();
            return report;
        }
        if (!unleased.empty()) {
            std::string message = "hub: push needs leases on objects changed here";
            for (const auto& id : unleased) message += "\n  " + id;
            throw std::runtime_error(message);
        }
        // Relationship validation needs every endpoint, and the hub's own
        // relationships of objects whose geometry or lifecycle changes.
        std::set<std::string> context;
        for (const auto& id : apply) {
            const auto mine = recordAt(merged, id), theirs = recordAt(before, id);
            if (isObjectRecord(mine) && (theirs.is_null() || recordHead(mine) != recordHead(theirs) ||
                mine.value("lifecycle_epoch", "") != theirs.value("lifecycle_epoch", "")))
                incidentEdges(hubDb, id, context);
        }
        for (const auto& id : std::set<std::string>(context)) {
            const auto edge = syncRecord(hub, txn, id);
            if (!edge.is_null() && !apply.contains(id)) { before[id] = edge; merged[id] = edge; }
        }
        const auto relationships = merged;
        for (const auto& [id, record] : relationships.items())
            if (!isObjectRecord(record))
                for (const auto* endpoint : {"a", "b"}) {
                    const auto object = record.at(endpoint).get<std::string>();
                    if (merged.contains(object)) continue;
                    const auto current = syncRecord(hub, txn, object);
                    if (current.is_null()) throw std::runtime_error("hub: relationship " + id + " needs missing object " + object);
                    before[object] = current; merged[object] = current;
                }
        applyObjects(hub, txn, before, merged);
        for (const auto& id : apply)
            if (isObjectRecord(recordAt(merged, id))) transferGeometry(db, hubDb, id, changes.at(id).geometryAfter);
        applyHeadsAndEdges(hub, txn, before, merged);
        const auto undecided = undecidedRelationships(merged, [&](const Json& uuid) -> std::optional<DependencyGeometry> {
            if (uuid.is_null()) return std::nullopt;
            const auto revision = hub.geometryRevision(txn, uuid.get<std::string>());
            if (!revision) return std::nullopt;
            return dependencyGeometry(hub, txn, *revision);
        });
        if (!undecided.empty())
            throw std::runtime_error("hub: relationship " + undecided.front() + " needs an explicit retain/update/remove decision");
        for (const auto& id : apply)
            if (syncRecord(hub, txn, id) != recordAt(merged, id)) throw std::runtime_error("hub: published state differs for " + id);
        publishHistory(hubDb, true);
        for (const auto& id : apply) {
            published[id] = hubVersion(hubDb, id);
            // Objects created here stay editable by their creator.
            if (before.contains(id) || !isObjectRecord(recordAt(merged, id))) continue;
            created.push_back({id, self, replica.author(), replica.hostname(), now, now + kDefaultLeaseSeconds * 1000000});
        }
        writeLeases(hubDb, created);
        txn.commit();
        report.pushed = int64_t(apply.size());
    }
    local.commit();
    auto mark = replica.write();
    pool(mark).hubApply = true;
    Statement version(conn(mark), "INSERT INTO hub_versions VALUES(?,?) ON CONFLICT(record_uuid) DO UPDATE SET version=excluded.version");
    Statement done(conn(mark), "DELETE FROM hub_pending WHERE record_uuid=? AND counter=?");
    for (const auto& [id, value] : published) {
        version.run(id, value);
        done.run(id, changes.at(id).counter);
    }
    exec(conn(mark), "DELETE FROM hub_inflight;");
    writeLeases(conn(mark), created);
    mark.commit();
    return report;
}

HubSyncReport hubSync(ObjectStore& replica)
{
    auto report = hubPull(replica);
    if (!report.conflicts.empty()) return report;
    auto pushed = hubPush(replica);
    report.pushed = pushed.pushed;
    report.conflicts = std::move(pushed.conflicts);
    return report;
}

std::vector<HubLease> acquireLeases(ObjectStore& replica, const std::vector<std::string>& objects, int64_t seconds)
{
    return grantLeases(replica, objects, seconds, true);
}

std::vector<HubLease> renewLeases(ObjectStore& replica, const std::vector<std::string>& objects, int64_t seconds)
{
    return grantLeases(replica, objects, seconds, false);
}

void releaseLeases(ObjectStore& replica, const std::vector<std::string>& objects)
{
    auto local = replica.read();
    const auto up = upstream(conn(local));
    const auto self = replicaIdentity(conn(local));
    const auto pending = uuidColumn(conn(local), "SELECT record_uuid FROM hub_pending");
    std::vector<std::string> ids;
    for (const auto& object : objects) {
        ids.push_back(formatUuid(parseUuid(object)));
        if (pending.contains(ids.back())) throw std::runtime_error("hub: push " + ids.back() + " before releasing its lease");
    }
    auto hub = openHub(replica, local, up);
    {
        auto txn = hub.write();
        Statement drop(conn(txn), "DELETE FROM hub_leases WHERE object_uuid=? AND holder=?");
        for (const auto& id : ids) drop.run(id, self);
        txn.commit();
    }
    local.commit();
    auto mirror = replica.write();
    pool(mirror).hubApply = true;
    Statement drop(conn(mirror), "DELETE FROM hub_leases WHERE object_uuid=? AND holder=?");
    for (const auto& id : ids) drop.run(id, self);
    mirror.commit();
}

std::optional<HubStatus> hubStatus(ObjectStore& store)
{
    auto txn = store.read();
    if (detectHubRole(conn(txn)) != HubRole::Replica) return std::nullopt;
    HubStatus status;
    status.hub = upstream(conn(txn)).hub;
    status.replica = replicaIdentity(conn(txn));
    Statement count(conn(txn), "SELECT count(*) FROM hub_pending"); count.row();
    status.pending = count.i64(0);
    status.conflicts = conflicts(store, txn);
    status.leases = readLeases(conn(txn));
    return status;
}

HubSyncReport resolveHubConflict(ObjectStore& replica, const std::string& uuid, const std::string& keep)
{
    DependencyScope dependencyScope;
    const auto id = formatUuid(parseUuid(uuid));
    if (keep != "hub" && keep != "local") throw std::invalid_argument("hub: keep must be \"hub\" or \"local\"");
    Upstream up;
    std::optional<ObjectStore> hub;
    {
        auto local = replica.read();
        up = upstream(conn(local));
        Statement row(conn(local), "SELECT 1 FROM hub_conflicts WHERE record_uuid=?", id);
        if (!row.row()) throw std::runtime_error("hub: no conflict for " + id);
        if (keep == "local") {
            Statement lease(conn(local), "SELECT 1 FROM hub_leases WHERE object_uuid=? AND holder=? AND expires_us>?",
                id, replicaIdentity(conn(local)), nowMicros());
            if (!lease.row() && isObjectRecord(syncRecord(replica, local, id)))
                throw std::runtime_error("hub: lease " + id + " before keeping the local record");
        }
        hub.emplace(openHub(replica, local, up));
    }
    auto hubTxn = hub->read();
    const auto record = syncRecord(*hub, hubTxn, id);
    const auto version = hubVersion(conn(hubTxn), id);
    auto txn = replica.write();
    pool(txn).hubApply = true;
    if (keep == "hub") {
        // Earlier hub geometry may have been skipped while the conflict stood.
        installHubRecords(replica, txn, *hub, hubTxn, {{id, record}}, 0);
        execute(conn(txn), "DELETE FROM hub_pending WHERE record_uuid=?", id);
    }
    // Keeping local: the next push compares against the version seen now.
    execute(conn(txn), "INSERT INTO hub_versions VALUES(?,?) ON CONFLICT(record_uuid) DO UPDATE SET version=excluded.version", id, version);
    execute(conn(txn), "DELETE FROM hub_conflicts WHERE record_uuid=?", id);
    txn.commit();
    hubTxn.commit();
    hub.reset();
    HubSyncReport report;
    if (keep == "hub") report.pulled = 1;
    else report = hubPush(replica);
    return report;
}

} // namespace vc::objects
