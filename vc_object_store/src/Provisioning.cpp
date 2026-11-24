#include "internal/Database.hpp"
#include "geometry/SharedGeometry.hpp"
#include "internal/ImmutableImport.hpp"
#include "internal/Versions.hpp"
#include "internal/Extensions.hpp"
#include "internal/EdgeLayout.hpp"

namespace vc::objects {
using namespace detail;
namespace {
void protectDatabaseIdentity(sqlite3* db)
{
    exec(db, "CREATE TRIGGER database_identity_uuid_immutable BEFORE UPDATE OF uuid ON database_identity "
             "BEGIN SELECT RAISE(ABORT,'database identity is immutable'); END;");
}
}

int64_t detail::writeCommitSequence(sqlite3* db)
{
    Statement s(db, "SELECT commit_sequence FROM store_manifest WHERE singleton=1");
    if (!s.row() || s.i64(0) == INT64_MAX) throw std::runtime_error("object store: commit sequence exhausted or absent");
    return s.i64(0) + 1;
}

int64_t detail::writeRecordVersion(sqlite3* db, const char* table, const std::string& uuid)
{
    const std::string name(table);
    if (name != "objects" && name != "edges") throw std::invalid_argument("invalid versioned record table");
    if(name=="edges")requireEdgeColumns(db,{"version","version_generation"});
    const auto key = parseUuid(uuid);
    const auto sequence = writeCommitSequence(db);
    Statement current(db, "SELECT version,commit_sequence FROM " + name + " WHERE uuid=?", key);
    if (!current.row()) throw std::runtime_error("cannot version missing record: " + uuid);
    const auto version = current.i64(0);
    if (current.i64(1) == sequence) return version;
    if (version == INT64_MAX) throw std::overflow_error("object store: record version exhausted");
    // A graph-only publication keeps existing approval evidence. Actual
    // content edits clear it in stampContentChange, as do metadata/geometry
    // edits made later in this same transaction.
    execute(db, "UPDATE " + name + " SET approved_version=CASE WHEN approved_version=version THEN version+1 ELSE approved_version END,"
        "version=?,commit_sequence=?,version_generation=? WHERE uuid=?",
        version + 1, sequence, parseUuid(activeGeneration(db)), key);
    return version + 1;
}

int64_t detail::writeObjectVersion(sqlite3* db, const std::string& uuid)
{
    return writeRecordVersion(db, "objects", uuid);
}

bool detail::immutableInitial(sqlite3* db) {
    if(const auto pool=activeStatementPools.find(db);pool!=activeStatementPools.end())return pool->second->immutableInitial;
    Statement feature(db,"SELECT 1 FROM store_required_features WHERE name=?",std::string(immutableImportFeature));
    return feature.row();
}

int64_t detail::allocateRecordId(sqlite3* db, const char* keySpace)
{
    // NULL next_id denotes exhaustion; do not overflow the signed key domain.
    Statement row(db,"UPDATE key_allocators SET next_id=CASE WHEN next_id=upper_bound THEN NULL ELSE next_id+1 END "
                     "WHERE name=? AND next_id IS NOT NULL RETURNING coalesce(next_id-1,upper_bound)",std::string(keySpace));
    if (!row.row()) throw std::runtime_error(std::string("object store: reserved ID range exhausted: ")+keySpace);
    return row.i64(0);
}

bool detail::historyEnabled(sqlite3* db)
{
    const auto pool = activeStatementPools.find(db);
    if (pool != activeStatementPools.end()) return pool->second->historyEnabled;
    Statement s(db, "SELECT history_enabled FROM store_manifest WHERE singleton=1");
    if (!s.row()) throw std::runtime_error("object store: missing store manifest");
    return s.i64(0) != 0;
}

void detail::requireHistory(sqlite3* db)
{
    if (!historyEnabled(db)) throw std::runtime_error("object store: history is disabled for this store");
}

bool detail::spatialEnabled(sqlite3* db)
{
    const auto pool = activeStatementPools.find(db);
    if (pool != activeStatementPools.end()) return pool->second->spatialEnabled;
    Statement s(db, "SELECT spatial_enabled FROM store_manifest WHERE singleton=1");
    if (!s.row()) throw std::runtime_error("object store: missing store manifest");
    return s.i64(0) != 0;
}

void detail::requireSpatial(sqlite3* db)
{
    if (!spatialEnabled(db)) throw std::runtime_error("object store: spatial index is absent from this store");
}

void detail::createProvisioning(sqlite3* db, bool history, bool spatial)
{
    exec(db, R"SQL(
CREATE TABLE store_manifest(
 singleton INTEGER PRIMARY KEY CHECK(singleton=1),
 schema_version INTEGER NOT NULL CHECK(schema_version=22),
 commit_sequence INTEGER NOT NULL CHECK(commit_sequence>=0),
 generation BLOB NOT NULL CHECK(length(generation)=16),
 initial_generation BLOB NOT NULL CHECK(length(initial_generation)=16),
 feed_start INTEGER NOT NULL DEFAULT 0 CHECK(feed_start>=0),
 history_enabled INTEGER NOT NULL CHECK(history_enabled IN (0,1)),
 sealed INTEGER NOT NULL CHECK(sealed IN (0,1)),
 spatial_enabled INTEGER NOT NULL CHECK(spatial_enabled IN (0,1))
);
CREATE TABLE store_required_features(name TEXT PRIMARY KEY);
CREATE TABLE store_write_state(singleton INTEGER PRIMARY KEY CHECK(singleton=1), dirty INTEGER NOT NULL CHECK(dirty IN (0,1)));
INSERT INTO store_write_state VALUES(1,0);
CREATE TABLE key_allocators(
 name TEXT PRIMARY KEY CHECK(name='objects'),
 namespace_uuid BLOB NOT NULL CHECK(length(namespace_uuid)=16),
 lower_bound INTEGER NOT NULL CHECK(lower_bound>0),
 upper_bound INTEGER NOT NULL CHECK(upper_bound>=lower_bound),
 next_id INTEGER CHECK(next_id>=lower_bound AND next_id<=upper_bound)
);
)SQL");
    const auto lineage = parseUuid(util::newUuid());
    execute(db, "INSERT INTO store_manifest VALUES(1,22,0,?,?,0,?,0,?)", lineage, lineage, int64_t(history), int64_t(spatial));
}

void detail::createChangeTracking(sqlite3* db)
{
    // Transactional dirty marker, including savepoint rollback. Do not use
    // sqlite3_total_changes(): that also counts changes rolled back to a savepoint.
    for (const auto* table : {"objects", "fibers", "surfaces", "tracks", "object_tags", "edges", "edge_tags", "table_directory", "table_key_columns", "store_required_features"}) {
        for (const auto* op : {"INSERT", "UPDATE", "DELETE"}) {
            const auto sql = std::string("CREATE TRIGGER ") + table + "_changed_" + op + " AFTER " + op + " ON " + table +
                " BEGIN UPDATE store_write_state SET dirty=1 WHERE singleton=1 AND dirty=0; END;";
            exec(db, sql.c_str());
        }
    }
    exec(db, "CREATE TRIGGER store_policy_changed AFTER UPDATE OF sealed,history_enabled,spatial_enabled ON store_manifest "
             "WHEN old.sealed<>new.sealed OR old.history_enabled<>new.history_enabled OR old.spatial_enabled<>new.spatial_enabled "
             "BEGIN UPDATE store_write_state SET dirty=1 WHERE singleton=1 AND dirty=0; END;");
}

StoreProvisioning ObjectStore::provisioning(const Txn& txn) const
{
    Statement s(static_cast<sqlite3*>(connection(txn)),
        "SELECT schema_version,commit_sequence,generation,history_enabled,sealed,spatial_enabled FROM store_manifest WHERE singleton=1");
    if (!s.row()) throw std::runtime_error("object store: missing store manifest");
    return {int(s.i64(0)), s.i64(1), s.uuid(2), s.i64(3) ? HistoryMode::On : HistoryMode::Off, s.i64(4) != 0, s.i64(5) != 0};
}

void ObjectStore::seal(WriteTxn& txn)
{
    // Include any tables provisioned by optional subsystems before publication.
    registerStorageTables(static_cast<sqlite3*>(connection(txn)));
    execute(static_cast<sqlite3*>(connection(txn)), "UPDATE store_manifest SET sealed=1 WHERE singleton=1 AND sealed=0");
}

void detail::createStorageDirectory(sqlite3* db, const std::string& databaseUuid)
{
    exec(db, R"SQL(
CREATE TABLE database_identity(
 singleton INTEGER PRIMARY KEY CHECK(singleton=1),
 uuid BLOB NOT NULL UNIQUE CHECK(length(uuid)=16),
 schema_version INTEGER NOT NULL
);
CREATE TABLE table_directory(
 uuid BLOB PRIMARY KEY CHECK(length(uuid)=16),
 name TEXT NOT NULL UNIQUE
);
CREATE TABLE table_key_columns(
 table_uuid BLOB NOT NULL REFERENCES table_directory(uuid) ON DELETE CASCADE,
 ordinal INTEGER NOT NULL CHECK(ordinal>0),
 name TEXT NOT NULL,
 PRIMARY KEY(table_uuid,ordinal)
);
)SQL");
    execute(db, "INSERT INTO database_identity VALUES(1,?,0)", parseUuid(databaseUuid.empty() ? util::newUuid() : util::canonicalUuid(databaseUuid)));
    protectDatabaseIdentity(db);
    exec(db, "CREATE TRIGGER database_identity_retained BEFORE DELETE ON database_identity "
             "BEGIN SELECT RAISE(ABORT,'database identity is retained'); END;"
             "CREATE TRIGGER table_identity_immutable BEFORE UPDATE OF uuid ON table_directory "
             "BEGIN SELECT RAISE(ABORT,'table identity is immutable'); END;");
}

void detail::forkStorageIdentity(sqlite3* db)
{
    // This is only for an explicit copied-database fork. Normal edits never
    // change physical identities. Unknown extension references must not be
    // silently invalidated by resetting the table directory.
    Statement tables(db, "SELECT name FROM pragma_table_list WHERE schema='main' AND type='table'");
    while (tables.row()) {
        const auto name = tables.text(0);
        if (name == "table_key_columns" || name == "extension_tables") continue;
        Statement refs(db, "SELECT 1 FROM pragma_foreign_key_list(?) WHERE \"table\" IN "
                           "('database_identity','table_directory','table_key_columns')", name);
        if (refs.row()) throw std::runtime_error("replica fork: extension table references physical identity: " + name);
    }
    exec(db, "DROP TRIGGER database_identity_uuid_immutable;");
    execute(db, "UPDATE database_identity SET uuid=? WHERE singleton=1", parseUuid(util::newUuid()));
    protectDatabaseIdentity(db);
    // A database fork preserves every extension's rows, source pins and schema.
    // Only physical table UUIDs change. Defer the known FK checks while moving
    // the parent and its two core registry references together; commit checks
    // their completed state. Unknown identity-dependent extensions were rejected
    // above, rather than silently rewriting their private interpretation.
    exec(db, "PRAGMA defer_foreign_keys=ON; DROP TRIGGER table_identity_immutable;");
    const bool extensions = hasExtensionTables(db);
    std::vector<std::array<unsigned char,16>> tableIds;
    {
        Statement rows(db,"SELECT uuid FROM table_directory ORDER BY name");
        while (rows.row()) tableIds.push_back(rows.blob16(0));
    }
    for (const auto& oldId : tableIds) {
        const auto newId = parseUuid(util::newUuid());
        execute(db,"UPDATE table_directory SET uuid=? WHERE uuid=?",newId,oldId);
        execute(db,"UPDATE table_key_columns SET table_uuid=? WHERE table_uuid=?",newId,oldId);
        if (extensions) execute(db,"UPDATE extension_tables SET table_uuid=? WHERE table_uuid=?",newId,oldId);
    }
    exec(db,"CREATE TRIGGER table_identity_immutable BEFORE UPDATE OF uuid ON table_directory "
            "BEGIN SELECT RAISE(ABORT,'table identity is immutable'); END;");
    registerStorageTables(db);
}

void detail::registerStorageTables(sqlite3* db)
{
    // SQL table names from the catalog are passed as parameters, never SQL.
    // SQLite-owned tables (including R-tree shadow tables) are implementation
    // details, not independently routable application tables. Views are not
    // separate physical table instances either.
    exec(db, "DELETE FROM table_key_columns WHERE table_uuid IN (SELECT uuid FROM table_directory WHERE name NOT IN "
             "(SELECT name FROM pragma_table_list WHERE schema='main' AND type IN ('table','virtual')));"
             "DELETE FROM table_directory WHERE name NOT IN (SELECT name FROM pragma_table_list WHERE schema='main' AND type IN ('table','virtual'));");
    Statement tables(db, "SELECT name FROM pragma_table_list WHERE schema='main' AND type IN ('table','virtual') "
                         "AND substr(name,1,7)<>'sqlite_' AND name NOT IN (SELECT name FROM table_directory) ORDER BY name");
    std::vector<std::string> names;
    while (tables.row()) names.push_back(tables.text(0));
    Statement keys(db, "SELECT pk,name FROM pragma_table_info(?) WHERE pk>0 ORDER BY pk");
    for (const auto& name : names) {
        const auto id = parseUuid(util::newUuid());
        execute(db, "INSERT INTO table_directory VALUES(?,?)", id, name);
        keys.reset(); keys.bind(1, name);
        while (keys.row())
            execute(db, "INSERT INTO table_key_columns VALUES(?,?,?)", id, keys.i64(0), keys.text(1));
    }
}

DatabaseIdentity ObjectStore::databaseIdentity(const Txn& txn) const
{
    auto* db = static_cast<sqlite3*>(connection(txn));
    Statement identity(db, "SELECT uuid,schema_version FROM database_identity WHERE singleton=1");
    if (!identity.row()) throw std::runtime_error("object store: missing database identity");
    DatabaseIdentity result{formatUuid(identity.blob16(0)), int(identity.i64(1)), {}};
    Statement tables(db, "SELECT t.uuid,t.name,k.name FROM table_directory t "
                         "LEFT JOIN table_key_columns k ON k.table_uuid=t.uuid ORDER BY t.name,k.ordinal");
    while (tables.row()) {
        const auto name = tables.text(1);
        if (result.tables.empty() || result.tables.back().name != name)
            result.tables.push_back({formatUuid(tables.blob16(0)), name, {}});
        if (!tables.isNull(2)) result.tables.back().primaryKey.push_back(tables.text(2));
    }
    return result;
}

} // namespace vc::objects
