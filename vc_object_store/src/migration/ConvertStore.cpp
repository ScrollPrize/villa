#include "vc_object_store/migration/ConvertStore.hpp"
#include "migration/ConvertSchema21.hpp"
#include "internal/Database.hpp"
#include "internal/DatabaseCopy.hpp"
#include "internal/Sqlite.hpp"
#include "internal/Versions.hpp"
#include "internal/RecordJson.hpp"
#include "internal/RecordOverrides.hpp"
#include "internal/History.hpp"
#include "internal/Publication.hpp"
#include "internal/EdgeSql.hpp"
#include "geometry/GeometryDelta.hpp"
#include "geometry/GeometryReader.hpp"
#include "vc_object_store/objects/Artifacts.hpp"
#include "vc_object_store/objects/EdgeKinds.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_reader/native/platform_io.hpp"
#include "vc_reader/native/atomic_file.hpp"

#include <functional>
#include <map>
#include <set>
#include <utility>
#include <tuple>

namespace vc::objects::migration {
using namespace vc::objects::detail;
using Json = nlohmann::json;

namespace {
constexpr int kApplicationId = 0x56434f42;
constexpr int kNumberedSchema = 18; // schema 17 plus numbered history, text keys
constexpr int kLastUpstreamSchema = 20;
constexpr size_t kSampleLimit = 50;

struct Connection {
    sqlite3* db = nullptr;
    ~Connection() { if (db) sqlite3_close(db); }
    Connection() = default;
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    Connection(Connection&& other) noexcept : db(std::exchange(other.db, nullptr)) {}
    Connection& operator=(Connection&& other) noexcept {
        if (this != &other) {
            if (db) sqlite3_close(db);
            db = std::exchange(other.db, nullptr);
        }
        return *this;
    }
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
    // Explicit target-column expressions for a changed layout. Every source
    // column not retained must be listed, preventing accidental silent loss.
    std::map<std::string, std::string> expressions = {};
    std::set<std::string> removedColumns = {};
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
    std::string list, marks, projection;
    for (const auto& column : columns) {
        const auto expression = copy.expressions.find(column);
        if (!available.contains(column) && expression == copy.expressions.end())
            throw std::runtime_error("convert: source table " + table + " lacks column " + column);
        list += (list.empty() ? "" : ",") + quoted(column);
        marks += marks.empty() ? "?" : ",?";
        projection += (projection.empty() ? "" : ",") +
            (expression == copy.expressions.end() ? "s." + quoted(column) : expression->second);
    }
    for (const auto& column : source)
        if (std::find(columns.begin(), columns.end(), column) == columns.end() && !copy.removedColumns.contains(column))
            throw std::runtime_error("convert: source table " + table + " has unmapped column " + column);
    Statement in(db, copy.sourceFilter.empty() ? "SELECT " + projection + " FROM src." + quoted(table) + " s" : copy.sourceFilter);
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

// The staged upstream formats use both textual and binary UUIDs. This is
// confined to conversion; normal reads always use indexed binary keys.
std::string keySql(const std::string& value)
{
    return "(CASE typeof(" + value + ") WHEN 'blob' THEN lower(hex(" + value +
        ")) ELSE replace(lower(" + value + "),'-','') END)";
}

std::string latestVersionSql(const char* table, const char* key, const char* column)
{
    return "(SELECT v." + std::string(column) + " FROM src." + table + " v WHERE " +
        keySql("v." + std::string(key)) + "=" + keySql("s.uuid") + " ORDER BY v.version DESC LIMIT 1)";
}

Json numberedRecord(Json record, int64_t version, int64_t sequence, const std::string& generation)
{
    const auto approved = record.find("approved_generation");
    if (approved != record.end() && !approved->is_null() &&
        (!record.contains("generation") || *approved != record.at("generation")))
        throw std::runtime_error("convert: stale legacy approval cannot be promoted to the current numbered version: " +
                                 record.value("uuid", "unknown record"));
    const bool hasApproval = approved != record.end() ? !approved->is_null() :
        (record.contains("approved_version") && !record.at("approved_version").is_null());
    record["approved_version"] = hasApproval ? Json(version) : Json(nullptr);
    record.erase("generation");
    record.erase("approved_generation");
    record.erase("data_generation");
    record["version"] = version;
    record["commit_sequence"] = sequence;
    record["version_generation"] = generation;
    return record;
}

std::map<std::string, Json> numberArchivedRecords(sqlite3* db)
{
    // Rewrite complete states and re-diff against the converted parent. Editing
    // individual old JSON-patch operations would miss inherited fields or
    // confuse a per-setter mutation count with a committed numbered version.
    std::map<std::string, Json> converted;
    Statement version(db,"SELECT version,coalesce(commit_seq,0),generation FROM object_versions WHERE revision_uuid=?");
    transformArchivedObjectRecords(db,[&](const std::string& uuid,Json& value) {
        version.reset();version.bind(1,parseUuid(uuid));
        if(!version.row())throw std::runtime_error("convert: archived object has no numbered version");
        value=numberedRecord(std::move(value),version.i64(0),version.i64(1),version.uuid(2));
        converted.emplace(uuid,value);
    });
    Statement edges(db, "SELECT r.uuid,r.record,v.version,v.commit_seq,v.generation FROM relationship_revisions r "
                       "JOIN relationship_versions v ON v.revision_uuid=r.uuid");
    Statement updateEdge(db, "UPDATE relationship_revisions SET record=? WHERE uuid=?");
    while (edges.row())
        updateEdge.run(numberedRecord(Json::parse(edges.text(1)), edges.i64(2), edges.isNull(3) ? 0 : edges.i64(3),
                                     edges.uuid(4)).dump(), edges.blob16(0));
    return converted;
}

void retainDeletedVersions(sqlite3* db, const std::map<std::string, Json>& records, Json& report)
{
    // Upstream keeps a deleted object's last number only in history. Current
    // tombstones now retain that number even after history is disabled.
    int64_t nextObject = scalar(db, "SELECT coalesce(max(id),0) FROM objects");
    int64_t objectCount = 0, edgeCount = 0;
    Statement objects(db, "SELECT v.object_uuid,v.version,v.generation,v.commit_seq,v.revision_uuid "
        "FROM object_versions v JOIN object_revisions r ON r.uuid=v.revision_uuid WHERE r.deleted=1 "
        "AND v.version=(SELECT max(n.version) FROM object_versions n WHERE n.object_uuid=v.object_uuid) "
        "AND NOT EXISTS(SELECT 1 FROM objects o WHERE o.uuid=v.object_uuid)");
    while (objects.row()) {
        if (nextObject == INT64_MAX) throw std::overflow_error("convert: object ID exhausted by tombstones");
        const auto& value = records.at(objects.uuid(4));
        auto record = objectFromJson(value); commonFromJson(value, record);
        insertObjectRow(db, record, ++nextObject, objects.isNull(3) ? 0 : objects.i64(3),
                        record.createdBy, record.createdHost, record.createdAt, true);
        execute(db, "UPDATE objects SET state=2,version=?,version_generation=?,changed_by=?,changed_at=?,changed_host=? WHERE uuid=?",
            objects.i64(1), objects.blob16(2), record.changedBy, record.changedAt, record.changedHost, objects.blob16(0));
        ++objectCount;
    }
    Statement edges(db, "SELECT v.edge_uuid,v.version,v.generation,v.commit_seq,(SELECT r.record FROM relationship_versions p "
        "JOIN relationship_revisions r ON r.uuid=p.revision_uuid WHERE p.edge_uuid=v.edge_uuid ORDER BY p.version DESC LIMIT 1) "
        "FROM relationship_versions v WHERE v.deleted=1 "
        "AND v.version=(SELECT max(n.version) FROM relationship_versions n WHERE n.edge_uuid=v.edge_uuid) "
        "AND NOT EXISTS(SELECT 1 FROM edges e WHERE e.uuid=v.edge_uuid)");
    while (edges.row()) {
        if (edges.isNull(4)) throw std::runtime_error("convert: deleted relationship has no recorded endpoints");
        auto value = numberedRecord(Json::parse(edges.text(4)), edges.i64(1), edges.isNull(3) ? 0 : edges.i64(3), edges.uuid(2));
        auto record = edgeFromJson(value);
        insertEdgeRow(db, record, edges.isNull(3) ? 0 : edges.i64(3),
                      record.createdBy, record.createdHost, record.createdAt, true);
        execute(db, "UPDATE edges SET state=2,version=?,version_generation=?,changed_by=?,changed_at=?,changed_host=? WHERE uuid=?",
            edges.i64(1), edges.blob16(2), record.changedBy, record.changedAt, record.changedHost, edges.blob16(0));
        ++edgeCount;
    }
    report["current_tombstones"] = {{"objects", objectCount}, {"relationships", edgeCount}};
}

// Prepared upstream schema 18–20 -> current, from `stage` into a new destination built by
// the runtime's own schema definition. Unchanged tables are copied as they
// are; text history and version keys become 16-byte blobs, a revision without
// a parent stores its whole record, materialized head records are kept only
// for heads with a parent, and hex geometry hashes become 32-byte blobs
// (schema-19 sources already have these forms and pass through).
void copyToCurrentSchema(const std::filesystem::path& stage, const std::filesystem::path& sourceLocation,
                         const std::filesystem::path& destination, Json& report)
{
    Connection out;
    out.db = openRaw(destination, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE);
    auto* db = out.db;
    exec(db, "PRAGMA journal_mode=WAL;");
    exec(db, "PRAGMA synchronous=OFF;");
    exec(db, ("PRAGMA application_id=" + std::to_string(kApplicationId) + ";").c_str());
    execute(db, "ATTACH DATABASE ? AS src", stage.string());
    exec(db, "BEGIN IMMEDIATE;");
    std::string lineage;
    { Statement value(db, "SELECT active_generation FROM src.store_history");
      if (!value.row()) throw std::runtime_error("convert: source has no active lineage");
      lineage = sqlite3_column_type(value.value, 0) == SQLITE_BLOB ? value.uuid(0) : value.text(0); }
    createStoreSchema(db, lineage);
    // Copy without triggers (history tracking, spatial upkeep, immutability),
    // then restore them exactly.
    std::vector<std::pair<std::string, std::string>> triggers;
    std::set<std::pair<std::string, std::string>> knownPrograms;
    {
        Statement s(db, "SELECT name,sql FROM main.sqlite_master WHERE type='trigger' ORDER BY rowid");
        while (s.row()) triggers.emplace_back(s.text(0), s.text(1));
        Statement programs(db, "SELECT type,name FROM main.sqlite_master WHERE type IN ('trigger','view','index') AND sql IS NOT NULL");
        while (programs.row()) knownPrograms.emplace(programs.text(0), programs.text(1));
    }
    // These upstream programs belonged to structures replaced by direct keys,
    // inline current geometry and the shared state field. Everything else
    // must have a current-schema successor; do not silently discard extensions.
    for (const auto* name : {"objects_kind", "edges_unique", "edges_kind_id", "retired_objects", "geometry_dependencies_edge", "geometry_revision_generation"})
        knownPrograms.emplace("index", name);
    for (const auto* name : {"geometry_dependencies_insert", "geometry_dependencies_update", "source_identity_head_insert",
                             "source_identity_head_update", "source_identity_head_delete"})
        knownPrograms.emplace("trigger", name);
    for (const auto* table : {"geometry_heads", "edge_geometry_bindings"})
        for (const auto* operation : {"INSERT", "UPDATE", "DELETE"})
            knownPrograms.emplace("trigger", std::string(table) + "_history_" + operation);
    {
        Statement programs(db, "SELECT type,name FROM src.sqlite_master WHERE type IN ('trigger','view','index') AND sql IS NOT NULL");
        while (programs.row())
            if (!knownPrograms.contains({programs.text(0), programs.text(1)}))
                throw std::runtime_error("convert: unsupported extension " + programs.text(0) + " " + programs.text(1) +
                                         "; remove or explicitly port it before converting");
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
    for (const auto* table : {"objects", "edges"})
        if (scalar(db, std::string("SELECT count(*) FROM src.") + table +
                       " WHERE approved_generation IS NOT NULL AND approved_generation<>generation"))
            throw std::runtime_error(std::string("convert: stale legacy approval in ") + table +
                                     " must be resolved with the source tooling before conversion");
    const std::set<std::string> removed{
        "store_history", "object_lifecycle", "geometry_heads", "edge_geometry_bindings",
        "edge_lifecycle_bindings", "geometry_edge_dependencies"};
    const std::set<std::string> provisioned{
        "store_manifest", "store_required_features", "store_write_state", "database_identity",
        "table_directory", "table_key_columns", "key_allocators"};
    for (const auto& table : source)
        if (!target.contains(table) && !removed.contains(table))
            throw std::runtime_error("convert: source table " + table + " has no successor");
        else if (!target.contains(table)) report["removed_tables"].push_back(table);

    // Revisions without a parent: their patch from {} becomes the record.
    // Columns in schema order: seq, uuid, object_uuid, parent_uuid, geometry_uuid, metadata_delta, ...
    const Adjust rootRecord = [](Statement& in, Statement& out) {
        if (sqlite3_column_type(in.value, 3) != SQLITE_NULL) return true;
        const auto stored = Json::parse(in.text(5));
        out.bind(6, (stored.is_object() ? stored : Json::object().patch(stored)).dump());
        return true;
    };
    std::map<std::string, TableCopy> transforms{
        {"object_revisions", {{"uuid", "object_uuid", "parent_uuid"}, "", rootRecord}},
        {"object_heads", {{"object_uuid", "revision_uuid"}}},
        {"object_head_records", {{"object_uuid", "revision_uuid"},
            "SELECT h.object_uuid,h.revision_uuid,h.record FROM src.object_head_records h "
            "JOIN src.object_revisions r ON r.uuid=h.revision_uuid WHERE r.parent_uuid IS NOT NULL"}},
        {"relationship_revisions", {{"uuid", "edge_uuid", "a_revision", "b_revision", "a_geometry", "b_geometry"}}},
        {"object_relationship_deltas", {{"revision_uuid", "edge_uuid", "relationship_uuid"}}},
        {"object_relationship_heads", {{"object_uuid", "edge_uuid", "relationship_uuid"}}},
        {"history_generations", {{"generation"}}},
        {"store_commits", {{"generation", "request_uuid"}}},
        {"store_changes", {{"record_uuid"}}},
        {"object_versions", {{"object_uuid", "generation", "revision_uuid"}}},
        {"relationship_versions", {{"edge_uuid", "generation", "revision_uuid"}}},
        {"transaction_receipts", {{"request_uuid", "generation"}}},
        // Columns: uuid, object_uuid, parent_uuid, content_hash, manifest, ...
        {"geometry_revisions", {.hashColumns = {"content_hash"}}},
        {"geometry_chunks", {.hashColumns = {"hash"}}},
        {"geometry_current_chunks", {.hashColumns = {"hash"}}},
        {"geometry_revision_chunks", {.hashColumns = {"hash"}}},
        {"geometry_deltas", {.hashColumns = {"manifest_hash"}}},
    };
    for (const auto& [table, versions, key] : {
            std::tuple{"objects", "object_versions", "object_uuid"},
            std::tuple{"edges", "relationship_versions", "edge_uuid"}}) {
        auto& copy = transforms[table];
        const auto version = latestVersionSql(versions, key, "version");
        copy.uuidColumns.insert("version_generation");
        copy.expressions["version"] = version;
        copy.expressions["commit_sequence"] = "coalesce(" + latestVersionSql(versions, key, "commit_seq") + ",0)";
        copy.expressions["version_generation"] = latestVersionSql(versions, key, "generation");
        copy.expressions["approved_version"] = "CASE WHEN s.approved_generation IS NULL THEN NULL ELSE " + version + " END";
        copy.removedColumns = {"generation", "approved_generation"};
    }
    auto& objects = transforms.at("objects");
    objects.removedColumns.insert("data_generation");
    objects.expressions["state"] = "coalesce((SELECT CASE WHEN l.retired<>0 THEN 1 ELSE 0 END FROM src.object_lifecycle l WHERE l.object_id=s.id),0)";
    objects.expressions["lifecycle_epoch"] = "coalesce((SELECT l.epoch FROM src.object_lifecycle l WHERE l.object_id=s.id),'')";
    // Filled from the archived reader after every chunk/delta has been copied.
    // A compact delta descriptor is not a valid direct current descriptor.
    objects.expressions["geometry_manifest"] = "NULL";
    auto& edges = transforms.at("edges");
    edges.expressions["state"] = "0";
    edges.expressions["a_uuid"] = "(SELECT o.uuid FROM src.objects o WHERE o.id=s.a_id)";
    edges.expressions["b_uuid"] = "(SELECT o.uuid FROM src.objects o WHERE o.id=s.b_id)";
    edges.removedColumns.insert("a_id"); edges.removedColumns.insert("b_id");
    edges.removedColumns.insert("id");
    {
        const auto fields=copiedColumns(db,"main","edges");
        const auto kind=std::find(fields.begin(),fields.end(),"kind")-fields.begin();
        const auto approval=std::find(fields.begin(),fields.end(),"approval_state")-fields.begin();
        edges.adjust=[kind,approval](Statement& source,Statement& target){
            target.bind(int(kind+1),edgeKindCode(source.text(int(kind))));
            target.bind(int(approval+1),edgeApprovalCode(source.text(int(approval))));return true;
        };
    }
    transforms["edge_tags"].expressions["edge_uuid"]="(SELECT e.uuid FROM src.edges e WHERE e.id=s.edge_id)";
    transforms["edge_tags"].uuidColumns.insert("edge_uuid");
    transforms["edge_tags"].removedColumns.insert("edge_id");
    auto& geometries = transforms.at("geometry_revisions");
    geometries.expressions["object_version"] = "coalesce((SELECT min(v.version) FROM src.object_versions v "
        "JOIN src.object_revisions r ON r.uuid=v.revision_uuid WHERE " + keySql("r.geometry_uuid") + "=" + keySql("s.uuid") + "),0)";
    geometries.removedColumns.insert("object_generation");
    auto& journal = transforms["journal"];
    journal.expressions = {{"commit_sequence", "0"}, {"approved_version", "NULL"}};
    journal.removedColumns = {"generation", "approved_generation"};
    report["gaps"]["legacy_journal_commit_mapping"] = {{"count", scalar(db, "SELECT count(*) FROM src.journal")},
        {"note", "legacy per-mutation generations do not identify commits; commit_sequence is 0 and approval-version pin is absent; attribution and approval provenance are retained"}};
    Json counts = Json::object();
    for (const auto& table : target) {
        if (provisioned.contains(table)) continue;
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
    execute(db, "UPDATE store_manifest SET generation=?,initial_generation=?,commit_sequence="
                "(SELECT commit_sequence FROM src.store_history),feed_start=(SELECT feed_start FROM src.store_history)",
            parseUuid(lineage), parseUuid(lineage));
    execute(db, "UPDATE database_identity SET schema_version=?", int64_t(kSchemaVersion));
    const auto convertedRecords = numberArchivedRecords(db);
    retainDeletedVersions(db, convertedRecords, report);
    {
        Statement heads(db, "SELECT h.object_uuid,h.revision_uuid,g.content_hash FROM src.geometry_heads h "
                            "JOIN main.geometry_revisions g ON g.uuid=h.revision_uuid");
        Statement current(db, "UPDATE objects SET geometry_manifest=?,storage_kind='sqlite',storage_ref=?,data_checksum=? WHERE uuid=?");
        while (heads.row()) {
            GeometryReader geometry(db, heads.uuid(1), false);
            current.run(geometry.manifest().dump(), heads.uuid(1), heads.hash(2), heads.blob16(0));
        }
    }
    for (const auto* table : {"objects"}) {
        const auto last = scalar(db, std::string("SELECT coalesce(max(id),0) FROM ") + table);
        execute(db, "UPDATE key_allocators SET namespace_uuid=(SELECT uuid FROM store_identity),next_id=? WHERE name=?",
            last == INT64_MAX ? std::optional<int64_t>() : std::optional(last + 1), std::string(table));
    }
    registerStorageTables(db);
    if (tableExists(db, "sqlite_sequence")) {
        exec(db, "DELETE FROM main.sqlite_sequence;");
        exec(db, "INSERT INTO main.sqlite_sequence SELECT * FROM src.sqlite_sequence;");
    }
    // Every geometry manifest still matches its recorded content hash, now a
    // blob (the runtime's own loader verifies it).
    {
        Statement revisions(db, "SELECT uuid,manifest,content_hash FROM main.geometry_revisions");
        while (revisions.row()) loadGeometryManifest(db, revisions.uuid(0), revisions.text(1), revisions.hash(2));
    }
    relocateExternalLocators(db,sourceLocation,destination);
    for (const auto& [name, sql] : triggers) exec(db, (sql + ";").c_str());
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

// The converted history is readable by the runtime, so use its manifest
// parser and artifact writer for both current and historical references.
void copyArtifacts(const std::filesystem::path& source, const std::filesystem::path& destination, Json& report)
{
    OpenOptions options;
    options.author = "vc_object_convert";
    auto store = ObjectStore::open(destination, options);
    std::map<std::string, uint64_t> files;
    {
        auto txn = store.read();
        Statement objects(ObjectStoreImplAccess::conn(txn), "SELECT DISTINCT object_uuid FROM object_versions");
        while (objects.row())
            for (const auto& revision : store.objectHistory(txn, objects.uuid(0)))
                for (const auto& file : artifactManifest(revision.record)) {
                    const auto [entry, inserted] = files.emplace(file.sha256, file.length);
                    if (!inserted && entry->second != file.length)
                        throw std::runtime_error("convert: inconsistent artifact lengths for " + file.sha256);
                }
    }
    for (const auto& [hash, length] : files) {
        const auto path = artifactBlobPath(source, hash);
        if (std::filesystem::file_size(path) != length || store.putArtifact(path) != hash)
            throw std::runtime_error("convert: artifact does not match its manifest: " + path.string());
    }
    report["copied_artifacts"] = files.size();
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

// A schema-18 or schema-19 source already has numbered history: the mapping
// is read back (text keys in schema 18, 16-byte keys in schema 19).
Json mappingOfNumbered(sqlite3* db, Json& report)
{
    const auto key = [](const Statement& s, int c) {
        return sqlite3_column_type(s.value, c) == SQLITE_BLOB ? s.uuid(c) : s.text(c);
    };
    const auto store = storeUuidText(db);
    Statement active(db, "SELECT active_generation FROM store_history");
    if (!active.row()) throw std::runtime_error("convert: missing history state");
    report["store_uuid"] = store;
    const auto generation = key(active, 0);
    report["generation"] = generation;
    Json objects = Json::object(), relationships = Json::object();
    Statement o(db, "SELECT revision_uuid,object_uuid,version,generation FROM object_versions ORDER BY object_uuid,version");
    while (o.row()) objects[key(o, 0)] = {{"object", key(o, 1)}, {"version", o.i64(2)}, {"generation", key(o, 3)}};
    Statement r(db, "SELECT revision_uuid,edge_uuid,version,generation FROM relationship_versions "
                    "WHERE revision_uuid IS NOT NULL ORDER BY edge_uuid,version");
    while (r.row()) relationships[key(r, 0)] = {{"relationship", key(r, 1)}, {"version", r.i64(2)}, {"generation", key(r, 3)}};
    report["counts"] = {{"objects", scalar(db, "SELECT count(*) FROM objects")},
                        {"relationships", scalar(db, "SELECT count(*) FROM edges")},
                        {"object_versions", scalar(db, "SELECT count(*) FROM object_versions")},
                        {"relationship_versions", scalar(db, "SELECT count(*) FROM relationship_versions")},
                        {"commits", scalar(db, "SELECT commit_sequence FROM store_history")},
                        {"geometry_revisions", scalar(db, "SELECT count(*) FROM geometry_revisions")}};
    return Json{{"store", store}, {"generation", generation}, {"objects", objects}, {"relationships", relationships}};
}

} // namespace

namespace {
ConversionResult convertStoreImpl(const std::filesystem::path& source, const std::filesystem::path& destination,
                                  sqlite3* lockedSource = nullptr)
{
    if (std::filesystem::exists(destination) || std::filesystem::is_symlink(destination))
        throw std::runtime_error("convert: destination exists: " + destination.string());
    const auto artifacts = artifactBlobDirectory(destination).parent_path();
    if (std::filesystem::exists(artifacts) || std::filesystem::is_symlink(artifacts))
        throw std::runtime_error("convert: destination artifacts exist: " + artifacts.string());
    if (std::filesystem::exists(source) && std::filesystem::exists(destination.parent_path().empty() ? "." : destination.parent_path()) &&
        std::filesystem::equivalent(source.parent_path().empty() ? "." : source.parent_path(),
                                    destination.parent_path().empty() ? "." : destination.parent_path()) &&
        source.filename() == destination.filename())
        throw std::runtime_error("convert: source and destination are the same file");
    const auto stage = std::filesystem::path(destination.string() + ".converting");
    if (std::filesystem::exists(stage) || std::filesystem::is_symlink(stage))
        throw std::runtime_error("convert: staging file exists: " + stage.string());
    Connection in;
    if (!lockedSource) in.db = openRaw(source, SQLITE_OPEN_READONLY);
    auto* input = lockedSource ? lockedSource : in.db;
    if (pragmaInt(input, "application_id") != kApplicationId)
        throw std::runtime_error("convert: not an object store: " + source.string());
    const int sourceSchema = pragmaInt(input, "user_version");
    if (sourceSchema < 16 || sourceSchema > kLastUpstreamSchema)
        throw std::runtime_error("convert: expected upstream schema 16–20, found " + std::to_string(sourceSchema));
    if (tableExists(input, "store_manifest"))
        throw std::runtime_error("convert: unpublished experimental store layouts are not supported");

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
        backupSqliteSnapshot(input,staged.db);
        in = {};
        report["source_bytes"] = int64_t(std::filesystem::file_size(source));

        auto* db = staged.db;
        exec(db, "PRAGMA journal_mode=MEMORY;");
        exec(db, "PRAGMA synchronous=OFF;");
        exec(db, "PRAGMA foreign_keys=ON;");
        exec(db, "BEGIN IMMEDIATE;");
        report["removed_tables"] = settle(db, report);
        if (sourceSchema == 16) upgradeSchema16(db, report);
        const auto mapping = sourceSchema < kNumberedSchema ? numberSchema17(db, report, source) : mappingOfNumbered(db, report);
        // The replica identity (the generation seed above) has no successor.
        if (tableExists(db, "sync_replica")) {
            exec(db, "DROP TABLE sync_replica;");
            report["removed_tables"].push_back("sync_replica");
        }
        exec(db, "COMMIT;");
        staged = {};

        copyToCurrentSchema(stage, source, destination, report);
        copyArtifacts(source, destination, report);
        removeFiles(stage);
        report["destination_bytes"] = int64_t(std::filesystem::file_size(destination));
        report["validation"] = "ok";
        return {report.dump(2), mapping.dump()};
    } catch (...) {
        in = {};
        staged = {};
        if (created) {
            removeFiles(stage);
            removeFiles(destination);
            std::error_code ignored;
            std::filesystem::remove_all(artifacts, ignored);
        }
        throw;
    }
}
} // namespace

ConversionResult convertStore(const std::filesystem::path& source, const std::filesystem::path& destination)
{
    if(isSchema21Store(source))return convertSchema21(source,destination);
    return convertStoreImpl(source, destination);
}

void writeConversionReports(const ConversionResult& result, const std::filesystem::path& mapping,
    const std::filesystem::path& report)
{
    for (const auto& path : {mapping, report})
        if (const auto parent = path.parent_path(); !parent.empty() && !std::filesystem::is_directory(parent))
            throw std::runtime_error("conversion report directory does not exist: " + parent.string());
    StagedDatabase stagedMapping(mapping, false), stagedReport(report, false);
    const auto prepare = [](const auto& path, const std::string& json) {
        vcr::AtomicFile file(path.string());
        const auto text = json + '\n';
        file.write(reinterpret_cast<const uint8_t*>(text.data()), text.size());
        file.publish(true);
    };
    prepare(stagedMapping.path, result.mappingJson);
    prepare(stagedReport.path, result.reportJson);
    stagedMapping.publish();
    stagedReport.publish();
}

ConversionResult convertStoreInPlace(const std::filesystem::path& input,
    const std::function<void(const ConversionResult&)>& beforeReplace)
{
    const auto source = std::filesystem::absolute(input).lexically_normal();
    if (!std::filesystem::is_regular_file(source) || std::filesystem::is_symlink(source))
        throw std::invalid_argument("convert: in-place input must be a regular non-symlink SQLite file");
    OwnershipGuard ownership(source, OwnershipGuard::Mode::Service);
    Connection locked;
    const bool packed=isPackedStore(source);
    if(!packed)locked.db = openRaw(source, SQLITE_OPEN_READWRITE);
    // Switch out of WAL before replacement: no old sidecar may be replayed
    // against the newly published schema. Exclusive locking is retained by
    // this connection while its snapshot is copied and validated.
    if(!packed)exec(locked.db, "PRAGMA locking_mode=EXCLUSIVE; PRAGMA journal_mode=DELETE; BEGIN EXCLUSIVE; COMMIT;");
    const auto sibling = source.parent_path() / (source.filename().string() + ".convert-" + util::newUuid());
    const auto members = source.parent_path() / (source.filename().string() + ".members-" + util::newUuid());
    const auto siblingArtifacts = artifactBlobDirectory(sibling).parent_path();
    if (std::filesystem::exists(sibling) || std::filesystem::is_symlink(sibling) ||
        std::filesystem::exists(siblingArtifacts) || std::filesystem::is_symlink(siblingArtifacts))
        throw std::runtime_error("convert: private sibling path already exists");
    struct Cleanup {
        std::filesystem::path path;
        std::filesystem::path members;
        bool published=false;
        ~Cleanup() {
            std::error_code ignored;
            for (const auto* suffix : {"", "-wal", "-shm", "-journal", "-owner"})
                std::filesystem::remove(path.string() + suffix, ignored);
            std::filesystem::remove_all(artifactBlobDirectory(path).parent_path(), ignored);
            if(!published)std::filesystem::remove_all(members,ignored);
        }
    } cleanup{sibling,members};
    auto result = packed?convertSchema21(source,sibling,nullptr,members):
        pragmaInt(locked.db,"user_version")==21?convertSchema21(source,sibling,locked.db,members):convertStoreImpl(source, sibling, locked.db);
    auto report = Json::parse(result.reportJson);
    report["destination"] = source.string();
    report["in_place"] = true;
    result.reportJson = report.dump(2);
    if (beforeReplace) beforeReplace(result);
    // The converter verified and copied every referenced artifact. In-place
    // replacement keeps the original, identical content-addressed files;
    // only the SQLite file changes atomically. The verified copies are private.
    vcr::platform::replace_file(sibling, source, true);
    cleanup.published=true;
    vcr::platform::sync_directory(source.parent_path(), "vcobj schema conversion");
    return result;
}

} // namespace vc::objects::migration
