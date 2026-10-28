#include "hub/HubRules.hpp"

namespace vc::objects::detail {
namespace {
std::set<std::string> changedRecords(sqlite3* db, const StatementPool& pool)
{
    std::set<std::string> ids;
    Statement rows(db, "SELECT record_uuid FROM journal WHERE seq>? UNION "
                       "SELECT object_uuid FROM geometry_revisions WHERE rowid>?", pool.startJournal, pool.startGeometry);
    while (rows.row()) ids.insert(formatUuid(rows.blob16(0)));
    return ids;
}

// Objects whose own record or geometry this write changed. Relationship
// membership revisions are not lease-protected: relationships are shared
// records, and push compares their versions like any other record.
std::set<std::string> changedObjects(sqlite3* db, const std::set<std::string>& changed)
{
    std::set<std::string> objects;
    Statement known(db, "SELECT 1 FROM object_revisions WHERE object_uuid=? LIMIT 1");
    for (const auto& id : changed) {
        known.reset(); known.bind(1, id);
        if (known.row()) objects.insert(id);
    }
    return objects;
}

struct Lease { std::string holder, author, host; int64_t expiresUs = 0; };
std::optional<Lease> lease(sqlite3* db, const std::string& object)
{
    Statement row(db, "SELECT holder,author,host,expires_us FROM hub_leases WHERE object_uuid=?", object);
    if (!row.row()) return std::nullopt;
    return Lease{row.text(0), row.text(1), row.text(2), row.i64(3)};
}

std::string describe(const Lease& lease)
{
    return lease.author + "@" + lease.host + " until " + isoFromMicros(lease.expiresUs);
}

int64_t latestPublicationUs(sqlite3* db)
{
    Statement row(db, "SELECT coalesce(max(published_at_us),0) FROM object_revisions");
    (void)row.row();
    return row.i64(0);
}

bool hasPublications(sqlite3* db)
{
    Statement table(db, "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hub_publications'");
    return table.row();
}

// Positions name hub states by journal sequence; each hub write that journals
// records the publication time its history reads at. Hubs made before the
// index start it, with the state they had, at their next write.
void createPublications(sqlite3* db, int64_t head, int64_t publishedAtUs)
{
    exec(db, R"SQL(
CREATE TABLE hub_publications(
  position INTEGER PRIMARY KEY,
  previous INTEGER NOT NULL,
  published_at_us INTEGER NOT NULL,
  replica TEXT NOT NULL
);)SQL");
    execute(db, "INSERT INTO hub_publications VALUES(?,0,?,'')", head, publishedAtUs);
    exec(db, ("PRAGMA user_version=" + std::to_string(kIndexedHubSchemaVersion) + ";").c_str());
}

HubPublication publication(Statement& row)
{
    return {row.i64(0), row.i64(1), row.i64(2), row.text(3)};
}
}

HubRole detectHubRole(sqlite3* db)
{
    bool leases = false, upstream = false;
    Statement tables(db, "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('hub_leases','hub_upstream')");
    while (tables.row()) (tables.text(0) == "hub_leases" ? leases : upstream) = true;
    if (upstream) return HubRole::Replica;
    return leases ? HubRole::Hub : HubRole::None;
}

void createHubTables(sqlite3* db)
{
    exec(db, R"SQL(
CREATE TABLE hub_leases(
  object_uuid TEXT PRIMARY KEY,
  holder TEXT NOT NULL, author TEXT NOT NULL, host TEXT NOT NULL,
  acquired_us INTEGER NOT NULL, expires_us INTEGER NOT NULL
);)SQL");
    createPublications(db, hubHead(db), latestPublicationUs(db));
}

int64_t hubHead(sqlite3* hub)
{
    Statement row(hub, "SELECT coalesce(max(seq),0) FROM journal");
    (void)row.row();
    return row.i64(0);
}

HubPublication hubPublicationAt(sqlite3* hub, int64_t position)
{
    const auto head = hubHead(hub);
    if (position < 0 || position > head)
        throw std::runtime_error("hub: position " + std::to_string(position) + " is not a hub state (head " +
                                 std::to_string(head) + ")");
    if (!hasPublications(hub)) {
        if (position != head)
            throw std::runtime_error("hub: position " + std::to_string(position) + " predates the hub's publication index");
        return {head, 0, latestPublicationUs(hub), ""};
    }
    Statement row(hub, "SELECT position,previous,published_at_us,replica FROM hub_publications "
                       "WHERE position<=? ORDER BY position DESC LIMIT 1", position);
    if (!row.row())
        throw std::runtime_error("hub: position " + std::to_string(position) + " predates the hub's publication index");
    auto result = publication(row);
    Statement between(hub, "SELECT 1 FROM journal WHERE seq>? AND seq<=? LIMIT 1", result.position, position);
    if (between.row())
        throw std::runtime_error("hub: position " + std::to_string(position) + " falls inside a hub write");
    return result;
}

std::optional<HubPublication> hubPublicationAfter(sqlite3* hub, int64_t previous)
{
    if (!hasPublications(hub)) return std::nullopt;
    Statement row(hub, "SELECT position,previous,published_at_us,replica FROM hub_publications "
                       "WHERE previous=? AND position>? ORDER BY position LIMIT 1", previous, previous);
    if (!row.row()) return std::nullopt;
    return publication(row);
}

std::optional<HubPublication> hubPublicationOf(sqlite3* hub, int64_t version)
{
    if (!hasPublications(hub)) return std::nullopt;
    Statement row(hub, "SELECT position,previous,published_at_us,replica FROM hub_publications "
                       "WHERE position>=?1 AND previous<?1 ORDER BY position LIMIT 1", version);
    if (!row.row()) return std::nullopt;
    return publication(row);
}

void createReplicaTables(sqlite3* db)
{
    // hub_leases (copied from the hub) is this replica's lease mirror.
    exec(db, R"SQL(
CREATE TABLE hub_upstream(
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  hub_path TEXT NOT NULL, hub_replica TEXT NOT NULL,
  clone_journal INTEGER NOT NULL, journal_cursor INTEGER NOT NULL, geometry_cursor INTEGER NOT NULL
);
CREATE TABLE hub_versions(record_uuid TEXT PRIMARY KEY, version INTEGER NOT NULL);
CREATE TABLE hub_pending(record_uuid TEXT PRIMARY KEY, counter INTEGER NOT NULL, geometry_after INTEGER NOT NULL);
CREATE TABLE hub_inflight(record_uuid TEXT PRIMARY KEY, record_hash TEXT NOT NULL);
CREATE TABLE hub_conflicts(record_uuid TEXT PRIMARY KEY, version INTEGER NOT NULL, record TEXT NOT NULL);
)SQL");
}

int64_t nowMicros()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string isoFromMicros(int64_t micros)
{
    const std::time_t seconds = micros / 1000000;
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &seconds);
#else
    gmtime_r(&seconds, &utc);
#endif
    char text[32];
    std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return text;
}

std::string replicaIdentity(sqlite3* db)
{
    Statement row(db, "SELECT uuid FROM sync_replica WHERE singleton=1");
    if (!row.row()) throw std::runtime_error("object store: missing replica identity");
    return row.text(0);
}

int64_t hubVersion(sqlite3* hub, const std::string& record)
{
    Statement row(hub, "SELECT coalesce(max(seq),0) FROM journal WHERE record_uuid=?", parseUuid(record));
    (void)row.row();
    return row.i64(0);
}

int64_t knownHubVersion(sqlite3* replica, const std::string& record)
{
    Statement known(replica, "SELECT version FROM hub_versions WHERE record_uuid=?", record);
    if (known.row()) return known.i64(0);
    // Untouched since the clone: the copied journal holds the hub's sequences.
    Statement cloned(replica, "SELECT coalesce(max(j.seq),0) FROM journal j, hub_upstream u "
                              "WHERE j.record_uuid=? AND j.seq<=u.clone_journal", parseUuid(record));
    (void)cloned.row();
    return cloned.i64(0);
}

void beginHubWrite(sqlite3* db, StatementPool& pool)
{
    pool.hubApply = false;
    pool.actingHolder.reset();
    Statement cursors(db, "SELECT coalesce((SELECT max(seq) FROM journal),0),"
                          "coalesce((SELECT max(rowid) FROM geometry_revisions),0),"
                          "coalesce((SELECT max(published_at_us) FROM object_revisions),0)");
    (void)cursors.row();
    pool.startJournal = cursors.i64(0); pool.startGeometry = cursors.i64(1); pool.startPublishedUs = cursors.i64(2);
}

void enforceHubRules(sqlite3* db, const StatementPool& pool, HubRole role)
{
    if (role == HubRole::None || (role == HubRole::Replica && pool.hubApply)) return;
    const auto changed = changedRecords(db, pool);
    if (role == HubRole::Hub) {
        const auto acting = pool.actingHolder.value_or(replicaIdentity(db));
        const auto now = nowMicros();
        for (const auto& object : changedObjects(db, changed))
            if (const auto held = lease(db, object); held && held->expiresUs > now && held->holder != acting)
                throw std::runtime_error("object store: " + object + " is leased to " + describe(*held));
        if (const auto head = hubHead(db); head > pool.startJournal) {
            if (!hasPublications(db)) createPublications(db, pool.startJournal, pool.startPublishedUs);
            else if (pragmaInt(db, "user_version") != kIndexedHubSchemaVersion)
                // Indexed before the version gate existed.
                exec(db, ("PRAGMA user_version=" + std::to_string(kIndexedHubSchemaVersion) + ";").c_str());
            execute(db, "INSERT INTO hub_publications VALUES(?,?,?,?)", head, pool.startJournal, latestPublicationUs(db), acting);
        }
        return;
    }
    const auto self = replicaIdentity(db);
    for (const auto& object : changedObjects(db, changed)) {
        const auto held = lease(db, object);
        if (held && held->holder == self) continue;
        // Created here and never pushed: nobody else can hold it yet.
        if (knownHubVersion(db, object) == 0) continue;
        throw std::runtime_error("object store: " + object + (held ? " is leased to " + describe(*held)
            : " is not leased by this replica; acquire a lease before editing"));
    }
    Statement pending(db, "INSERT INTO hub_pending VALUES(?,1,?) ON CONFLICT(record_uuid) DO UPDATE SET counter=counter+1");
    for (const auto& id : changed) pending.run(id, pool.startGeometry);
}
}
