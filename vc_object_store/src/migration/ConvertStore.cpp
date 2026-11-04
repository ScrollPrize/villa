#include "vc_object_store/migration/ConvertStore.hpp"
#include "internal/Sqlite.hpp"
#include "internal/Versions.hpp"

#include <map>
#include <set>

namespace vc::objects::migration {
using namespace vc::objects::detail;
using Json = nlohmann::json;

namespace {
constexpr int kApplicationId = 0x56434f42;
constexpr int kSourceSchema = 17, kTargetSchema = 18;
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
    Connection in;
    in.db = openRaw(source, SQLITE_OPEN_READONLY);
    if (pragmaInt(in.db, "application_id") != kApplicationId)
        throw std::runtime_error("convert: not an object store: " + source.string());
    if (const int version = pragmaInt(in.db, "user_version"); version != kSourceSchema)
        throw std::runtime_error("convert: expected schema " + std::to_string(kSourceSchema) + ", found " + std::to_string(version));

    Json report{{"source", source.string()}, {"destination", destination.string()},
                {"source_schema", kSourceSchema}, {"target_schema", kTargetSchema}};
    bool created = false;
    Connection out;
    try {
        out.db = openRaw(destination, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE);
        created = true;
        // One consistent snapshot of the source, copied page for page.
        sqlite3_backup* backup = sqlite3_backup_init(out.db, "main", in.db, "main");
        if (!backup) fail(out.db, "convert: backup");
        const int rc = sqlite3_backup_step(backup, -1);
        sqlite3_backup_finish(backup);
        if (rc != SQLITE_DONE) fail(out.db, "convert: snapshot copy");
        in = {};

        auto* db = out.db;
        exec(db, "PRAGMA journal_mode=WAL;");
        exec(db, "PRAGMA synchronous=FULL;");
        exec(db, "PRAGMA foreign_keys=ON;");
        exec(db, "BEGIN IMMEDIATE;");

        // Settle-before-convert: refuse replicas with unpublished state.
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

        // Coordination tables have no successor.
        Json removed = Json::array();
        {
            std::vector<std::string> tables;
            Statement s(db, "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'hub\\_%' ESCAPE '\\' ORDER BY name");
            while (s.row()) tables.push_back(s.text(0));
            for (const auto& table : tables) {
                exec(db, ("DROP TABLE \"" + table + "\";").c_str());
                removed.push_back(table);
            }
        }
        report["removed_tables"] = removed;

        const auto store = storeUuidText(db);
        std::string replicaIdentity;
        { Statement s(db, "SELECT uuid FROM sync_replica WHERE singleton=1"); if (s.row()) replicaIdentity = s.text(0); }
        // Deterministic per source file: re-running yields the same mapping, and
        // a hub and its (settled) replicas never share a generation.
        const auto generation = util::nameUuid(store, "vc_object_store/converted-generation/" + replicaIdentity);
        createVersionSchema(db, generation, "converted from schema 17");
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
        exec(db, ("PRAGMA user_version=" + std::to_string(kTargetSchema) + ";").c_str());
        exec(db, "COMMIT;");
        {
            Statement check(db, "PRAGMA quick_check");
            if (!check.row() || check.text(0) != "ok") throw std::runtime_error("convert: validation failed: quick_check");
        }
        check(sqlite3_wal_checkpoint_v2(db, nullptr, SQLITE_CHECKPOINT_TRUNCATE, nullptr, nullptr), db, "checkpoint");

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
        report["validation"] = "ok";
        Json mapping{{"store", store}, {"generation", generation}, {"objects", mappingObjects},
                     {"relationships", mappingRelationships}};
        out = {};
        return {report.dump(2), mapping.dump()};
    } catch (...) {
        out = {};
        if (created) {
            std::error_code error;
            for (const auto* suffix : {"", "-wal", "-shm", "-journal"})
                std::filesystem::remove(destination.string() + suffix, error);
        }
        throw;
    }
}

} // namespace vc::objects::migration
