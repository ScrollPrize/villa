#include "vc_object_store/migration/ConvertStore.hpp"
#include "internal/Database.hpp"
#include "internal/Sqlite.hpp"
#include "internal/Versions.hpp"
#include "geometry/GeometryDelta.hpp"

#include <functional>
#include <map>
#include <set>

namespace vc::objects::migration {
using namespace vc::objects::detail;
using Json = nlohmann::json;

namespace {
constexpr int kApplicationId = 0x56434f42;
constexpr int kNumberedSchema = 18; // schema 17 plus numbered history, text keys
constexpr size_t kSampleLimit = 50;

struct Connection {
    sqlite3* db = nullptr;
    ~Connection() { if (db) sqlite3_close(db); }
};

sqlite3* openRaw(const std::filesystem::path& path, int flags)
{
    sqlite3* db = nullptr;
    const int rc = sqlite3_open_v2(reinterpret_cast<const char*>(path.u8string().c_str()), &db, flags, nullptr);
    if (rc != SQLITE_OK) {
        const std::string message = db ? sqlite3_errmsg(db) : "out of memory";
        if (db) sqlite3_close(db);
        throw std::runtime_error("convert: cannot open " + path.string() + ": " + message);
    }
    sqlite3_busy_timeout(db, 5000);
    return db;
}

bool tableExists(sqlite3* db, const std::string& name)
{
    Statement s(db, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", name);
    return s.row();
}

int64_t scalar(sqlite3* db, const std::string& sql)
{
    Statement s(db, sql);
    if (!s.row() || s.isNull(0)) return 0;
    return s.i64(0);
}

void addSample(Json& list, const Json& value)
{
    if (list.size() < kSampleLimit) list.push_back(value);
}

// Strings in metadata that equal recorded revision UUIDs: references that
// need an explicit registered rewrite, never an implicit one.
void findReferences(const Json& value, const std::string& path, const std::map<std::string, std::pair<std::string, int64_t>>& objects,
                    const std::map<std::string, std::pair<std::string, int64_t>>& relationships, Json& found, int64_t& count)
{
    if (value.is_object()) {
        for (const auto& [key, child] : value.items()) findReferences(child, path + "/" + key, objects, relationships, found, count);
    } else if (value.is_array()) {
        for (size_t i = 0; i < value.size(); ++i) findReferences(value[i], path + "/" + std::to_string(i), objects, relationships, found, count);
    } else if (value.is_string()) {
        const auto text = value.get<std::string>();
        if (text.size() != 36) return;
        if (const auto o = objects.find(text); o != objects.end()) {
            ++count; addSample(found, {{"path", path}, {"revision", text}, {"object", o->second.first}, {"version", o->second.second}});
        } else if (const auto r = relationships.find(text); r != relationships.end()) {
            ++count; addSample(found, {{"path", path}, {"revision", text}, {"relationship", r->second.first}, {"version", r->second.second}});
        }
    }
}
// The numbered-history tables exactly as schema 18 defined them (text keys).
// Frozen: schema-17 stores pass through schema 18 on the way to the current one.
void createVersionSchema18(sqlite3* db, const std::string& generation, const std::string& reason)
{
    exec(db, R"SQL(
CREATE TABLE store_history(
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  active_generation TEXT NOT NULL,
  commit_sequence INTEGER NOT NULL DEFAULT 0 CHECK(commit_sequence>=0),
  feed_start INTEGER NOT NULL DEFAULT 0 CHECK(feed_start>=0)
);
CREATE TABLE history_generations(
  generation TEXT PRIMARY KEY CHECK(length(generation)=36),
  after_sequence INTEGER NOT NULL,
  started_at_us INTEGER NOT NULL,
  reason TEXT NOT NULL
);
CREATE TABLE store_commits(
  seq INTEGER PRIMARY KEY CHECK(seq>0),
  generation TEXT NOT NULL REFERENCES history_generations(generation),
  request_uuid TEXT,
  author TEXT NOT NULL, host TEXT NOT NULL,
  published_at_us INTEGER NOT NULL
);
CREATE TABLE store_changes(
  seq INTEGER NOT NULL REFERENCES store_commits(seq),
  position INTEGER NOT NULL,
  record_kind TEXT NOT NULL CHECK(record_kind IN ('object','relationship')),
  record_uuid TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version>0),
  deleted INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(seq,position)
);
CREATE TABLE object_versions(
  object_uuid TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version>0),
  generation TEXT NOT NULL REFERENCES history_generations(generation),
  commit_seq INTEGER REFERENCES store_commits(seq),
  revision_uuid TEXT NOT NULL UNIQUE REFERENCES object_revisions(uuid),
  PRIMARY KEY(object_uuid,version)
);
CREATE TABLE relationship_versions(
  edge_uuid TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version>0),
  generation TEXT NOT NULL REFERENCES history_generations(generation),
  commit_seq INTEGER REFERENCES store_commits(seq),
  revision_uuid TEXT UNIQUE REFERENCES relationship_revisions(uuid),
  deleted INTEGER NOT NULL DEFAULT 0 CHECK(deleted=(revision_uuid IS NULL)),
  PRIMARY KEY(edge_uuid,version)
);
CREATE TABLE transaction_receipts(
  request_uuid TEXT PRIMARY KEY CHECK(length(request_uuid)=36),
  generation TEXT NOT NULL REFERENCES history_generations(generation),
  digest TEXT NOT NULL,
  commit_seq INTEGER REFERENCES store_commits(seq),
  receipt TEXT NOT NULL CHECK(json_valid(receipt))
);
)SQL");
    for (const auto* table : {"history_generations", "store_commits", "store_changes", "object_versions",
                              "relationship_versions", "transaction_receipts"}) {
        for (const auto* operation : {"UPDATE", "DELETE"}) {
            const std::string name = std::string(table) + "_" + operation;
            exec(db, ("CREATE TRIGGER " + name + " BEFORE " + operation + " ON " + table +
                " BEGIN SELECT RAISE(ABORT,'numbered history is immutable and retained'); END;").c_str());
        }
    }
    const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    execute(db, "INSERT INTO history_generations VALUES(?,0,?,?)", generation, int64_t(now), reason);
    execute(db, "INSERT INTO store_history(singleton,active_generation) VALUES(1,?)", generation);
}

// Columns that hold a UUID in schema 18 text form and a 16-byte key in the
// current schema, per table; and a row adjustment run after the default
// binding (returns false to drop the row).
using Adjust = std::function<bool(Statement& in, Statement& out)>;
struct TableCopy {
    std::set<std::string> uuidColumns;
    // SELECT of the source rows when not every row is copied.
    std::string sourceFilter;
    Adjust adjust;
    // Columns holding SHA-256 hex text that become 32-byte blobs.
    std::set<std::string> hashColumns = {};
};

std::vector<std::string> copiedColumns(sqlite3* db, const std::string& schema, const std::string& table)
{
    // Generated columns are hidden and recomputed by the destination.
    std::vector<std::string> columns;
    Statement s(db, "SELECT name FROM pragma_table_xinfo(?, ?) WHERE hidden=0 ORDER BY cid", table, schema);
    while (s.row()) columns.push_back(s.text(0));
    return columns;
}

std::string quoted(const std::string& name) { return "\"" + name + "\""; }

int64_t copyTable(sqlite3* db, const std::string& table, const TableCopy& copy)
{
    const auto columns = copiedColumns(db, "main", table);
    const auto source = copiedColumns(db, "src", table);
    const std::set<std::string> available(source.begin(), source.end());
    std::string list, marks;
    for (const auto& column : columns) {
        if (!available.contains(column))
            throw std::runtime_error("convert: source table " + table + " lacks column " + column);
        list += (list.empty() ? "" : ",") + quoted(column);
        marks += marks.empty() ? "?" : ",?";
    }
    if (available.size() != columns.size())
        throw std::runtime_error("convert: source table " + table + " has columns the current schema does not");
    Statement in(db, copy.sourceFilter.empty() ? "SELECT " + list + " FROM src." + quoted(table) : copy.sourceFilter);
    Statement out(db, "INSERT INTO main." + quoted(table) + "(" + list + ") VALUES(" + marks + ")");
    int64_t rows = 0;
    while (in.row()) {
        out.reset();
        for (int c = 0; c < int(columns.size()); ++c) {
            if (copy.uuidColumns.contains(columns[c]) && sqlite3_column_type(in.value, c) == SQLITE_TEXT)
                out.bind(c + 1, parseUuid(in.text(c)));
            else if (copy.hashColumns.contains(columns[c]) && sqlite3_column_type(in.value, c) == SQLITE_TEXT)
                out.bind(c + 1, parseHash(in.text(c)));
            else
                check(sqlite3_bind_value(out.value, c + 1, sqlite3_column_value(in.value, c)), db, "convert: bind");
        }
        if (copy.adjust && !copy.adjust(in, out)) continue;
        out.stepDone();
        ++rows;
    }
    return rows;
}

// Schema 18 -> current, from `stage` (schema 18) into a new `destination`
// built by the runtime's own schema definition. Unchanged tables are copied
// as they are; history and version keys become 16-byte blobs, a revision
// without a parent stores its whole record, materialized head records are
// kept only for heads with a parent, and geometry hashes become 32-byte blobs.
void copyToCurrentSchema(const std::filesystem::path& stage, const std::filesystem::path& destination, Json& report)
{
    Connection out;
    out.db = openRaw(destination, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE);
    auto* db = out.db;
    exec(db, "PRAGMA journal_mode=WAL;");
    exec(db, "PRAGMA synchronous=OFF;");
    exec(db, ("PRAGMA application_id=" + std::to_string(kApplicationId) + ";").c_str());
    execute(db, "ATTACH DATABASE ? AS src", stage.string());
    exec(db, "BEGIN IMMEDIATE;");
    createStoreSchema(db, util::newUuid());
    // Copy without triggers (history tracking, spatial upkeep, immutability),
    // then restore them exactly.
    std::vector<std::pair<std::string, std::string>> triggers;
    {
        Statement s(db, "SELECT name,sql FROM main.sqlite_master WHERE type='trigger' ORDER BY rowid");
        while (s.row()) triggers.emplace_back(s.text(0), s.text(1));
    }
    for (const auto& [name, sql] : triggers) exec(db, ("DROP TRIGGER main." + quoted(name) + ";").c_str());

    auto tables = [&](const char* schema) {
        std::set<std::string> result, shadows;
        Statement virtuals(db, std::string("SELECT name FROM ") + schema + ".sqlite_master WHERE type='table' AND sql LIKE 'CREATE VIRTUAL TABLE%'");
        while (virtuals.row())
            for (const auto* suffix : {"_node", "_parent", "_rowid"}) shadows.insert(virtuals.text(0) + suffix);
        Statement s(db, std::string("SELECT name FROM ") + schema + ".sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'");
        while (s.row()) if (!shadows.contains(s.text(0))) result.insert(s.text(0));
        return result;
    };
    const auto target = tables("main"), source = tables("src");
    for (const auto& table : source)
        if (!target.contains(table)) throw std::runtime_error("convert: source table " + table + " has no successor");

    // Revisions without a parent: their patch from {} becomes the record.
    // Columns in schema order: seq, uuid, object_uuid, parent_uuid, geometry_uuid, metadata_delta, ...
    const Adjust rootRecord = [](Statement& in, Statement& out) {
        if (sqlite3_column_type(in.value, 3) != SQLITE_NULL) return true;
        const auto stored = Json::parse(in.text(5));
        out.bind(6, (stored.is_object() ? stored : Json::object().patch(stored)).dump());
        return true;
    };
    const std::map<std::string, TableCopy> transforms{
        {"object_revisions", {{"uuid", "object_uuid", "parent_uuid"}, "", rootRecord}},
        {"object_heads", {{"object_uuid", "revision_uuid"}}},
        {"object_head_records", {{"object_uuid", "revision_uuid"},
            "SELECT h.object_uuid,h.revision_uuid,h.record FROM src.object_head_records h "
            "JOIN src.object_revisions r ON r.uuid=h.revision_uuid WHERE r.parent_uuid IS NOT NULL"}},
        {"relationship_revisions", {{"uuid", "edge_uuid", "a_revision", "b_revision", "a_geometry", "b_geometry"}}},
        {"object_relationship_deltas", {{"revision_uuid", "edge_uuid", "relationship_uuid"}}},
        {"object_relationship_heads", {{"object_uuid", "edge_uuid", "relationship_uuid"}}},
        {"store_history", {{"active_generation"}}},
        {"history_generations", {{"generation"}}},
        {"store_commits", {{"generation", "request_uuid"}}},
        {"store_changes", {{"record_uuid"}}},
        {"object_versions", {{"object_uuid", "generation", "revision_uuid"}}},
        {"relationship_versions", {{"edge_uuid", "generation", "revision_uuid"}}},
        {"transaction_receipts", {{"request_uuid", "generation"}}},
        // Columns: uuid, object_uuid, parent_uuid, content_hash, manifest, ...
        {"geometry_revisions", {.hashColumns = {"content_hash"}}},
        {"geometry_chunks", {.hashColumns = {"hash"}}},
        {"geometry_revision_chunks", {.hashColumns = {"hash"}}},
    };
    Json counts = Json::object();
    for (const auto& table : target) {
        exec(db, ("DELETE FROM main." + quoted(table) + ";").c_str()); // rows the schema seeded
        if (!source.contains(table)) throw std::runtime_error("convert: source lacks table " + table);
        const auto found = transforms.find(table);
        const auto rows = copyTable(db, table, found == transforms.end() ? TableCopy{} : found->second);
        const auto expected = table == "object_head_records"
            ? scalar(db, "SELECT count(*) FROM src.object_head_records h JOIN src.object_revisions r "
                         "ON r.uuid=h.revision_uuid WHERE r.parent_uuid IS NOT NULL")
            : scalar(db, "SELECT count(*) FROM src." + quoted(table));
        if (rows != expected || scalar(db, "SELECT count(*) FROM main." + quoted(table)) != expected)
            throw std::runtime_error("convert: validation failed: row count of " + table);
        counts[table] = rows;
    }
    if (tableExists(db, "sqlite_sequence")) {
        exec(db, "DELETE FROM main.sqlite_sequence;");
        exec(db, "INSERT INTO main.sqlite_sequence SELECT * FROM src.sqlite_sequence;");
    }
    for (const auto& [name, sql] : triggers) exec(db, (sql + ";").c_str());
    // Every geometry manifest still matches its recorded content hash, now a
    // blob (the runtime's own loader verifies it).
    {
        Statement revisions(db, "SELECT uuid,manifest,content_hash FROM main.geometry_revisions");
        while (revisions.row()) loadGeometryManifest(db, revisions.uuid(0), revisions.text(1), revisions.hash(2));
    }
    exec(db, ("PRAGMA user_version=" + std::to_string(kSchemaVersion) + ";").c_str());
    exec(db, "COMMIT;");
    exec(db, "DETACH DATABASE src;");
    {
        Statement fk(db, "PRAGMA foreign_key_check");
        if (fk.row()) throw std::runtime_error("convert: validation failed: foreign key check on " + fk.text(0));
        Statement quick(db, "PRAGMA quick_check");
        if (!quick.row() || quick.text(0) != "ok") throw std::runtime_error("convert: validation failed: quick_check");
    }
    exec(db, "PRAGMA synchronous=FULL;");
    check(sqlite3_wal_checkpoint_v2(db, nullptr, SQLITE_CHECKPOINT_TRUNCATE, nullptr, nullptr), db, "checkpoint");
    out = {};
    report["copied_rows"] = counts;
}

// Settle-before-convert: refuse replicas with unpublished state and unfinished
// writes, then drop the coordination tables, which have no successor.
Json settle(sqlite3* db, Json& report)
{
    const bool replica = tableExists(db, "hub_upstream"), hub = !replica && tableExists(db, "hub_leases");
    report["source_role"] = replica ? "replica" : hub ? "hub" : "standalone";
    if (replica) {
        for (const auto* table : {"hub_pending", "hub_inflight", "hub_conflicts"})
            if (tableExists(db, table) && scalar(db, std::string("SELECT count(*) FROM ") + table) > 0)
                throw std::runtime_error(std::string("convert: replica has unpublished state (") + table +
                    "); sync or settle it with the old tooling before converting");
    }
    for (const auto* table : {"history_dirty_objects", "history_dirty_edges", "history_expected", "history_checked",
                              "history_required", "history_validated_edges"})
        if (scalar(db, std::string("SELECT count(*) FROM ") + table) > 0)
            throw std::runtime_error(std::string("convert: source has an unfinished write (") + table + ")");
    Json removed = Json::array();
    std::vector<std::string> tables;
    {
        Statement s(db, "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'hub\\_%' ESCAPE '\\' ORDER BY name");
        while (s.row()) tables.push_back(s.text(0));
    }
    for (const auto& table : tables) {
        exec(db, ("DROP TABLE \"" + table + "\";").c_str());
        removed.push_back(table);
    }
    return removed;
}
// Schema 16 -> 17 in place on the staged copy, for stores without surface
// data: the patches detail table becomes surfaces (with the area the runtime
// keeps), and the hub's sync baselines, which have no successor, go. A store
// with patch records needs their record JSON rewritten (detail key "patch")
// and surface areas computed; that is refused, as are unsettled baselines.
void upgradeSchema16(sqlite3* db, Json& report)
{
    const auto patches = scalar(db, "SELECT count(*) FROM patches");
    const auto surfaceObjects = scalar(db, "SELECT count(*) FROM objects WHERE kind='surface'");
    const auto patchJson = scalar(db, "SELECT (SELECT count(*) FROM object_revisions WHERE metadata_delta LIKE '%\"patch\"%' "
                                      "OR metadata_delta LIKE '%\"/patch%') + (SELECT count(*) FROM object_head_records "
                                      "WHERE record LIKE '%\"patch\"%')");
    if (patches || surfaceObjects || patchJson)
        throw std::runtime_error("convert: schema-16 surface (patch) records are not supported (" + std::to_string(surfaceObjects) +
            " surfaces, " + std::to_string(patches) + " patch rows, " + std::to_string(patchJson) + " records naming patch)");
    for (const auto* table : {"sync_baselines", "sync_baseline_records", "sync_baseline_cursors"})
        if (tableExists(db, table) && scalar(db, std::string("SELECT count(*) FROM ") + table) > 0)
            throw std::runtime_error(std::string("convert: schema-16 store has sync baselines (") + table +
                "); sync or settle it with the old tooling before converting");
    exec(db, "DROP TRIGGER IF EXISTS patches_history_INSERT; DROP TRIGGER IF EXISTS patches_history_UPDATE;"
             "DROP TRIGGER IF EXISTS patches_history_DELETE;");
    exec(db, "ALTER TABLE patches RENAME TO surfaces; ALTER TABLE surfaces ADD COLUMN area REAL NOT NULL DEFAULT 0;");
    for (const auto* table : {"sync_baseline_cursors", "sync_baseline_records", "sync_baselines"})
        exec(db, (std::string("DROP TABLE IF EXISTS ") + table + ";").c_str());
    report["upgraded_from_schema_16"] = {{"renamed", {{"patches", "surfaces"}}},
                                         {"removed_tables", {"sync_baselines", "sync_baseline_records", "sync_baseline_cursors"}}};
}

// Schema 17 -> 18 in place on the staged copy: numbered versions and commits
// from recorded publication order (the only reconstruction), plus the gaps
// report. Returns the old revision UUID -> numbered reference mapping.
Json numberSchema17(sqlite3* db, Json& report, const std::filesystem::path& source)
{
    const auto store = storeUuidText(db);
    std::string replicaIdentity;
    { Statement s(db, "SELECT uuid FROM sync_replica WHERE singleton=1"); if (s.row()) replicaIdentity = s.text(0); }
    // Deterministic per source file: re-running yields the same mapping, and
    // a hub and its (settled) replicas never share a generation.
    const auto generation = util::nameUuid(store, "vc_object_store/converted-generation/" + replicaIdentity);
    createVersionSchema18(db, generation, "converted from schema 17");
    report["store_uuid"] = store;
    report["generation"] = generation;

    // Commits: one per recorded publication time (assigned once per atomic
    // publication). Their authors were never recorded.
    std::map<int64_t, int64_t> commitOf;
    {
        Statement times(db, "SELECT DISTINCT published_at_us FROM object_revisions WHERE published_at_us IS NOT NULL ORDER BY 1");
        Statement insert(db, "INSERT INTO store_commits VALUES(?,?,NULL,'unrecorded','',?)");
        while (times.row()) {
            const auto seq = int64_t(commitOf.size()) + 1;
            commitOf.emplace(times.i64(0), seq);
            insert.run(seq, generation, times.i64(0));
        }
    }
    std::map<int64_t, int64_t> positions;
    Statement change(db, "INSERT INTO store_changes VALUES(?,?,?,?,?,?)");

    // Object versions in recorded order (seq is allocation order).
    Json mappingObjects = Json::object(), untimed = Json::array();
    std::map<std::string, std::pair<std::string, int64_t>> objectMap;
    int64_t untimedCount = 0;
    {
        std::map<std::string, int64_t> next;
        Statement revisions(db, "SELECT uuid,object_uuid,published_at_us,deleted,metadata_complete FROM object_revisions ORDER BY seq");
        Statement insert(db, "INSERT INTO object_versions VALUES(?,?,?,?,?)");
        while (revisions.row()) {
            const auto revision = revisions.text(0), object = revisions.text(1);
            const auto version = ++next[object];
            std::optional<int64_t> commit;
            if (!revisions.isNull(2)) commit = commitOf.at(revisions.i64(2));
            insert.reset(); insert.bind(1, object); insert.bind(2, version); insert.bind(3, generation);
            insert.bind(4, commit); insert.bind(5, revision); insert.stepDone();
            if (commit) change.run(*commit, positions[*commit]++, std::string("object"), object, version, revisions.i64(3));
            else { ++untimedCount; addSample(untimed, {{"object", object}, {"version", version}, {"revision", revision},
                                                     {"metadata_complete", revisions.i64(4) != 0}}); }
            mappingObjects[revision] = {{"object", object}, {"version", version}};
            objectMap[revision] = {object, version};
        }
    }

    // Relationship states: published with the earliest object revision
    // whose membership delta records them. Removals are the null deltas.
    struct Event { std::optional<int64_t> commit; int64_t order; std::optional<std::string> revision; };
    std::map<std::string, std::vector<Event>> events;
    Json unpublished = Json::array();
    int64_t unpublishedCount = 0;
    {
        Statement states(db,
            "SELECT rr.rowid,rr.uuid,rr.edge_uuid,(SELECT min(r.published_at_us) FROM object_relationship_deltas d "
            "JOIN object_revisions r ON r.uuid=d.revision_uuid WHERE d.relationship_uuid=rr.uuid) "
            "FROM relationship_revisions rr ORDER BY rr.rowid");
        while (states.row()) {
            std::optional<int64_t> commit;
            if (!states.isNull(3)) commit = commitOf.at(states.i64(3));
            else { ++unpublishedCount; addSample(unpublished, {{"relationship", states.text(2)}, {"revision", states.text(1)}}); }
            events[states.text(2)].push_back({commit, states.i64(0), states.text(1)});
        }
        Statement removals(db,
            "SELECT d.edge_uuid,min(r.published_at_us) FROM object_relationship_deltas d "
            "JOIN object_revisions r ON r.uuid=d.revision_uuid WHERE d.relationship_uuid IS NULL "
            "AND r.published_at_us IS NOT NULL GROUP BY d.edge_uuid,r.published_at_us");
        while (removals.row())
            events[removals.text(0)].push_back({commitOf.at(removals.i64(1)), std::numeric_limits<int64_t>::max(), std::nullopt});
    }
    Json mappingRelationships = Json::object();
    std::map<std::string, std::pair<std::string, int64_t>> relationshipMap;
    int64_t relationshipVersions = 0;
    {
        Statement insert(db, "INSERT INTO relationship_versions VALUES(?,?,?,?,?,?)");
        for (auto& [edge, list] : events) {
            // Recorded publication order; states never published sort last.
            std::stable_sort(list.begin(), list.end(), [](const Event& x, const Event& y) {
                const auto cx = x.commit.value_or(std::numeric_limits<int64_t>::max());
                const auto cy = y.commit.value_or(std::numeric_limits<int64_t>::max());
                return cx != cy ? cx < cy : x.order < y.order;
            });
            int64_t version = 0;
            for (const auto& event : list) {
                ++version; ++relationshipVersions;
                insert.reset(); insert.bind(1, edge); insert.bind(2, version); insert.bind(3, generation);
                insert.bind(4, event.commit); insert.bind(5, event.revision); insert.bind(6, int64_t(!event.revision));
                insert.stepDone();
                if (event.commit)
                    change.run(*event.commit, positions[*event.commit]++, std::string("relationship"), edge, version,
                               int64_t(!event.revision));
                if (event.revision) {
                    mappingRelationships[*event.revision] = {{"relationship", edge}, {"version", version}};
                    relationshipMap[*event.revision] = {edge, version};
                }
            }
        }
    }
    execute(db, "UPDATE store_history SET commit_sequence=?", int64_t(commitOf.size()));

    // Geometry revisions name payloads, not object states: report those
    // that do not identify exactly one object version.
    Json ambiguous = Json::array();
    int64_t ambiguousCount = 0, orphanCount = 0;
    {
        Statement s(db, "SELECT lower(hex(g.uuid)),(SELECT count(*) FROM object_revisions r WHERE r.geometry_uuid=g.uuid) "
                        "FROM geometry_revisions g ORDER BY g.rowid");
        while (s.row()) {
            const auto n = s.i64(1);
            if (n == 1) continue;
            auto hex = s.text(0);
            const auto uuid = hex.substr(0, 8) + "-" + hex.substr(8, 4) + "-" + hex.substr(12, 4) + "-" + hex.substr(16, 4) + "-" + hex.substr(20);
            (n == 0 ? orphanCount : ambiguousCount)++;
            addSample(ambiguous, {{"geometry_revision", uuid}, {"object_versions", n}});
        }
    }

    // Metadata that holds old revision UUIDs (e.g. copy provenance).
    Json references = Json::array();
    int64_t referenceCount = 0;
    {
        Statement s(db, "SELECT uuid,meta FROM objects UNION ALL SELECT uuid,meta FROM edges");
        while (s.row())
            findReferences(Json::parse(s.text(1)), formatUuid(s.blob16(0)) + "/meta", objectMap, relationshipMap,
                           references, referenceCount);
        Statement history(db, "SELECT uuid,metadata_delta FROM object_revisions");
        while (history.row())
            findReferences(Json::parse(history.text(1)), "history/" + history.text(0), objectMap, relationshipMap,
                           references, referenceCount);
    }

    // Artifacts outside the database: an unversioned inventory; today's
    // file is not attached to any historical version.
    Json artifacts = Json::array();
    int64_t missingArtifacts = 0, artifactCount = 0, externalArtifacts = 0;
    {
        const auto root = std::filesystem::absolute(source).parent_path();
        Statement s(db, "SELECT uuid,storage_kind,storage_ref FROM objects WHERE storage_kind NOT IN ('','sqlite') ORDER BY uuid");
        while (s.row()) {
            ++artifactCount;
            const auto kind = s.text(1), ref = s.text(2);
            // Only local files can be checked here; datasets and Open Data
            // entries are external references by URI/catalog identity.
            Json entry{{"object", formatUuid(s.blob16(0))}, {"storage_kind", kind}, {"storage_ref", ref}};
            if (kind == "file") {
                std::error_code error;
                const auto path = std::filesystem::path(ref).is_absolute() ? std::filesystem::path(ref) : root / ref;
                const bool exists = !ref.empty() && std::filesystem::exists(path, error);
                if (!exists) ++missingArtifacts;
                entry["exists"] = exists;
                addSample(artifacts, entry);
            } else {
                ++externalArtifacts;
            }
        }
    }

    // Validation before the schema version changes.
    const auto objectRevisions = scalar(db, "SELECT count(*) FROM object_revisions");
    if (scalar(db, "SELECT count(*) FROM object_versions") != objectRevisions)
        throw std::runtime_error("convert: validation failed: object versions");
    if (scalar(db, "SELECT count(*) FROM object_heads h LEFT JOIN object_versions v ON v.revision_uuid=h.revision_uuid "
                   "WHERE v.version IS NULL") != 0)
        throw std::runtime_error("convert: validation failed: unnumbered head");
    if (scalar(db, "SELECT count(*) FROM object_relationship_heads h LEFT JOIN relationship_versions v "
                   "ON v.revision_uuid=h.relationship_uuid WHERE v.version IS NULL") != 0)
        throw std::runtime_error("convert: validation failed: unnumbered relationship head");
    if (scalar(db, "SELECT count(*) FROM relationship_revisions r LEFT JOIN relationship_versions v ON v.revision_uuid=r.uuid "
                   "WHERE v.version IS NULL") != 0)
        throw std::runtime_error("convert: validation failed: relationship versions");
    {
        Statement fk(db, "PRAGMA foreign_key_check");
        if (fk.row()) throw std::runtime_error("convert: validation failed: foreign key check on " + fk.text(0));
    }
    exec(db, ("PRAGMA user_version=" + std::to_string(kNumberedSchema) + ";").c_str());

    report["counts"] = {{"objects", scalar(db, "SELECT count(*) FROM objects")},
                        {"relationships", scalar(db, "SELECT count(*) FROM edges")},
                        {"object_versions", objectRevisions}, {"relationship_versions", relationshipVersions},
                        {"commits", int64_t(commitOf.size())},
                        {"geometry_revisions", scalar(db, "SELECT count(*) FROM geometry_revisions")}};
    report["gaps"] = {
        {"untimed_object_versions", {{"count", untimedCount}, {"samples", untimed},
            {"note", "baselines installed without their publication or metadata history; numbered in allocation order, no commit"}}},
        {"unpublished_relationship_versions", {{"count", unpublishedCount}, {"samples", unpublished}}},
        {"geometry_revisions_without_unique_object_version", {{"ambiguous", ambiguousCount}, {"unreferenced", orphanCount},
            {"samples", ambiguous},
            {"note", "geometry-only references to these cannot select one object version; resolve them from object pins or regenerate"}}},
        {"metadata_revision_references", {{"count", referenceCount}, {"samples", references},
            {"note", "old revision UUIDs kept verbatim in metadata; rewrite through a registered reference rule or regenerate"}}},
        {"artifacts", {{"count", artifactCount}, {"external", externalArtifacts}, {"missing", missingArtifacts}, {"samples", artifacts},
            {"note", "unversioned imported inventory; not attached to historical versions"}}},
        {"commit_attribution", "commit authors were not recorded before schema 18; records keep their own stamps"}};
    return Json{{"store", store}, {"generation", generation}, {"objects", mappingObjects},
                {"relationships", mappingRelationships}};
}

// A schema-18 source already has numbered history: the mapping is read back.
Json mappingOfSchema18(sqlite3* db, Json& report)
{
    const auto store = storeUuidText(db);
    Statement active(db, "SELECT active_generation FROM store_history");
    if (!active.row()) throw std::runtime_error("convert: missing history state");
    report["store_uuid"] = store;
    report["generation"] = active.text(0);
    Json objects = Json::object(), relationships = Json::object();
    Statement o(db, "SELECT revision_uuid,object_uuid,version,generation FROM object_versions ORDER BY object_uuid,version");
    while (o.row()) objects[o.text(0)] = {{"object", o.text(1)}, {"version", o.i64(2)}, {"generation", o.text(3)}};
    Statement r(db, "SELECT revision_uuid,edge_uuid,version,generation FROM relationship_versions "
                    "WHERE revision_uuid IS NOT NULL ORDER BY edge_uuid,version");
    while (r.row()) relationships[r.text(0)] = {{"relationship", r.text(1)}, {"version", r.i64(2)}, {"generation", r.text(3)}};
    report["counts"] = {{"objects", scalar(db, "SELECT count(*) FROM objects")},
                        {"relationships", scalar(db, "SELECT count(*) FROM edges")},
                        {"object_versions", scalar(db, "SELECT count(*) FROM object_versions")},
                        {"relationship_versions", scalar(db, "SELECT count(*) FROM relationship_versions")},
                        {"commits", scalar(db, "SELECT commit_sequence FROM store_history")},
                        {"geometry_revisions", scalar(db, "SELECT count(*) FROM geometry_revisions")}};
    return Json{{"store", store}, {"generation", active.text(0)}, {"objects", objects}, {"relationships", relationships}};
}

} // namespace

ConversionResult convertStore(const std::filesystem::path& source, const std::filesystem::path& destination)
{
    if (std::filesystem::exists(destination))
        throw std::runtime_error("convert: destination exists: " + destination.string());
    if (std::filesystem::exists(source) && std::filesystem::exists(destination.parent_path().empty() ? "." : destination.parent_path()) &&
        std::filesystem::equivalent(source.parent_path().empty() ? "." : source.parent_path(),
                                    destination.parent_path().empty() ? "." : destination.parent_path()) &&
        source.filename() == destination.filename())
        throw std::runtime_error("convert: source and destination are the same file");
    const auto stage = std::filesystem::path(destination.string() + ".converting");
    if (std::filesystem::exists(stage)) throw std::runtime_error("convert: staging file exists: " + stage.string());
    Connection in;
    in.db = openRaw(source, SQLITE_OPEN_READONLY);
    if (pragmaInt(in.db, "application_id") != kApplicationId)
        throw std::runtime_error("convert: not an object store: " + source.string());
    const int sourceSchema = pragmaInt(in.db, "user_version");
    if (sourceSchema != 16 && sourceSchema != 17 && sourceSchema != kNumberedSchema)
        throw std::runtime_error("convert: expected schema 16, 17 or 18, found " + std::to_string(sourceSchema));

    Json report{{"source", source.string()}, {"destination", destination.string()},
                {"source_schema", sourceSchema}, {"target_schema", kSchemaVersion}};
    auto removeFiles = [](const std::filesystem::path& path) {
        std::error_code error;
        for (const auto* suffix : {"", "-wal", "-shm", "-journal"}) std::filesystem::remove(path.string() + suffix, error);
    };
    bool created = false;
    Connection staged;
    try {
        // One consistent snapshot of the source, copied page for page into a
        // scratch file beside the destination; the source is never written.
        staged.db = openRaw(stage, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE);
        created = true;
        sqlite3_backup* backup = sqlite3_backup_init(staged.db, "main", in.db, "main");
        if (!backup) fail(staged.db, "convert: backup");
        const int rc = sqlite3_backup_step(backup, -1);
        sqlite3_backup_finish(backup);
        if (rc != SQLITE_DONE) fail(staged.db, "convert: snapshot copy");
        in = {};
        report["source_bytes"] = int64_t(std::filesystem::file_size(source));

        auto* db = staged.db;
        exec(db, "PRAGMA journal_mode=MEMORY;");
        exec(db, "PRAGMA synchronous=OFF;");
        exec(db, "PRAGMA foreign_keys=ON;");
        exec(db, "BEGIN IMMEDIATE;");
        report["removed_tables"] = settle(db, report);
        if (sourceSchema == 16) upgradeSchema16(db, report);
        const auto mapping = sourceSchema < kNumberedSchema ? numberSchema17(db, report, source) : mappingOfSchema18(db, report);
        exec(db, "COMMIT;");
        staged = {};

        copyToCurrentSchema(stage, destination, report);
        removeFiles(stage);
        report["destination_bytes"] = int64_t(std::filesystem::file_size(destination));
        report["validation"] = "ok";
        return {report.dump(2), mapping.dump()};
    } catch (...) {
        in = {};
        staged = {};
        if (created) { removeFiles(stage); removeFiles(destination); }
        throw;
    }
}

} // namespace vc::objects::migration
