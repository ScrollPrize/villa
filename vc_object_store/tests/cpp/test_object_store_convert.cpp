// Offline conversion of schema-17 and schema-18 stores to the current schema.
// The fixtures (data/convert_schema17.sql, data/convert_schema18.sql) were
// written by the schema-18 runtime from one history: tracks with geometry,
// metadata edits, relationships retained and kept recorded through a geometry
// edit, a removal, retirement and deletion, copy provenance naming an old
// revision UUID, and a tiled surface with a sparse (delta) edit.
// data/convert_expected.json holds what that runtime read back: every
// revision, its numbered version and commit, the feed, relationship heads and
// every geometry descriptor and array. A converted store must read back the
// same, with the source untouched.
#include "TestSupport.hpp"
#include "vc_object_store/migration/ConvertStore.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"

#include <sqlite3.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>

using namespace vc::objects;
using Json = nlohmann::json;

namespace {
const std::filesystem::path kData = VC_OBJECT_STORE_TEST_DATA;

std::string bytesOf(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

void sql(const std::filesystem::path& path, const std::string& text)
{
    sqlite3* db = nullptr;
    require(sqlite3_open(path.string().c_str(), &db) == SQLITE_OK, "open fixture");
    char* error = nullptr;
    const int rc = sqlite3_exec(db, text.c_str(), nullptr, nullptr, &error);
    const std::string message = error ? error : "";
    sqlite3_free(error);
    sqlite3_close(db);
    if (rc != SQLITE_OK) throw std::runtime_error("fixture SQL: " + message);
}

int64_t scalar(const std::filesystem::path& path, const std::string& query)
{
    sqlite3* db = nullptr;
    require(sqlite3_open_v2(path.string().c_str(), &db, SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK, "open store");
    sqlite3_stmt* s = nullptr;
    require(sqlite3_prepare_v2(db, query.c_str(), -1, &s, nullptr) == SQLITE_OK, ("prepare " + query).c_str());
    require(sqlite3_step(s) == SQLITE_ROW, ("row " + query).c_str());
    const auto value = sqlite3_column_int64(s, 0);
    sqlite3_finalize(s);
    sqlite3_close(db);
    return value;
}

std::string hex(const std::vector<uint8_t>& bytes)
{
    std::string out;
    for (auto b : bytes) { char t[3]; std::snprintf(t, 3, "%02x", b); out += t; }
    return out;
}

// Everything the fixture's runtime read back, read again from `path`.
// `generation`: the converted store's generation replaces the recorded one
// when a schema-17 source got a new one.
void requireSameHistory(const std::filesystem::path& path, const Json& expected, const std::optional<std::string>& generation)
{
    OpenOptions options; options.readOnly = true;
    auto db = ObjectStore::open(path, options);
    auto txn = db.read();
    const auto status = db.status(txn);
    require(status.storeUuid == expected.at("store").get<std::string>() && status.commitSequence == expected.at("sequence").get<int64_t>(), "identity and sequence");
    for (const auto& object : expected.at("objects")) {
        const auto history = db.objectHistory(txn, object);
        const auto& recorded = expected.at("history").at(object.get<std::string>());
        require(history.size() == recorded.size(), "revision count");
        for (size_t i = 0; i < history.size(); ++i) {
            auto want = recorded.at(i);
            if (generation) want["generation"] = *generation;
            require(Json::parse(objectRevisionToJson(history[i])) == want, ("revision " + std::to_string(i) + " of " + object.get<std::string>()).c_str());
        }
    }
    for (const auto& [uuid, geometry] : expected.at("geometry").items()) {
        const auto g = db.geometryRevision(txn, uuid);
        require(g && Json::parse(geometryRevisionToJson(*g)) == geometry.at("descriptor"), ("geometry descriptor " + uuid).c_str());
        for (const auto& [name, bytes] : geometry.at("arrays").items())
            require(hex(db.readGeometryArray(txn, uuid, name)) == bytes.get<std::string>(), ("geometry array " + name).c_str());
    }
    for (const auto& [edge, version] : expected.at("relationship_heads").items()) {
        const auto head = db.currentRelationshipVersion(txn, edge);
        require((head ? Json(*head) : Json(nullptr)) == version, "relationship head");
    }
    Json feed = Json::array();
    for (const auto& commit : db.changesSince(txn, {status.storeUuid, status.generation, 0, 0}, 1 << 20).commits)
        for (const auto& change : commit.changes) feed.push_back({commit.sequence, change.uuid, change.version, change.deleted});
    require(feed == expected.at("feed"), "feed");
}

// A converted store keeps working: a checked edit, and the old version still
// reads at its number.
void requireWritable(const std::filesystem::path& path, const std::string& object)
{
    OpenOptions options; options.author = "convert-test";
    auto db = ObjectStore::open(path, options);
    StoreStatus status; std::optional<int64_t> before; Json oldRecord;
    { auto txn = db.read(); status = db.status(txn); before = db.currentVersion(txn, object);
      oldRecord = Json::parse(objectRevisionToJson(*db.currentObjectRevision(txn, object))).at("record"); }
    TransactionRequest r; r.requestUuid = util::newUuid(); r.storeUuid = status.storeUuid; r.generation = status.generation;
    r.author = "convert-test"; r.expectedObjects = {{object, before}};
    r.operations = {SetMetaOp{object, "after_conversion", "true"}};
    const auto receipt = db.submit(r);
    require(receipt.objects.at(object) == *before + 1, "edit after conversion");
    auto txn = db.read();
    const auto old = db.objectAtVersion(txn, {status.storeUuid, status.generation, object, *before});
    require(old && Json::parse(objectRevisionToJson(*old)).at("record") == oldRecord, "old version reads back");
}
}

int main()
try {
    TemporaryDirectory directory;
    const auto expected = Json::parse(bytesOf(kData / "convert_expected.json"));
    OpenOptions options; options.author = "convert-test";
    for (const int schema : {17, 18}) {
        const auto source = directory.path / ("old" + std::to_string(schema) + ".vcobj");
        sql(source, bytesOf(kData / ("convert_schema" + std::to_string(schema) + ".sql")));
        const auto before = bytesOf(source);
        mustFail([&] { (void)ObjectStore::open(source, options); }); // the runtime refuses old schemas

        const auto destination = directory.path / ("new" + std::to_string(schema) + ".vcobj");
        const auto result = migration::convertStore(source, destination);
        require(bytesOf(source) == before, "source untouched");
        require(!std::filesystem::exists(destination.string() + ".converting"), "no staging file left");
        const auto report = Json::parse(result.reportJson);
        const auto mapping = Json::parse(result.mappingJson);
        require(report.at("validation") == "ok" && report.at("source_role") == "standalone", "report");
        require(report.at("source_schema") == schema && report.at("target_schema") == 19, "schemas reported");
        mustFail([&] { (void)migration::convertStore(source, destination); }); // never overwrites
        // Storage form: whole records for revisions without a parent, no
        // materialized head record duplicating one, 16-byte keys.
        require(scalar(destination, "SELECT count(*) FROM object_revisions WHERE parent_uuid IS NULL AND json_type(metadata_delta)<>'object'") == 0,
                "root revisions hold records");
        require(scalar(destination, "SELECT count(*) FROM object_head_records h JOIN object_revisions r ON r.uuid=h.revision_uuid "
                                    "WHERE r.parent_uuid IS NULL") == 0, "no duplicate head records");
        require(scalar(destination, "SELECT count(*) FROM object_versions WHERE length(object_uuid)<>16 OR length(revision_uuid)<>16") == 0,
                "blob keys");

        requireSameHistory(destination, expected,
                           schema == 17 ? std::optional(mapping.at("generation").get<std::string>()) : std::nullopt);
        {
            const auto ref = mapping.at("objects").at(expected.at("first_revision").get<std::string>());
            OpenOptions readOnly; readOnly.readOnly = true;
            auto db = ObjectStore::open(destination, readOnly);
            auto txn = db.read();
            const auto revision = db.objectAtVersion(txn, {expected.at("store"), mapping.at("generation"), ref.at("object"), ref.at("version")});
            require(revision && revision->uuid == expected.at("first_revision").get<std::string>(), "mapping resolves");
        }
        requireWritable(destination, expected.at("objects").at(1));
        if (schema != 17) continue;
        require(report.at("gaps").at("metadata_revision_references").at("count") == 2, "provenance reference reported");
        require(report.at("gaps").at("untimed_object_versions").at("count") == 0, "no untimed history");
        // Deterministic: converting again yields the same mapping.
        require(migration::convertStore(source, directory.path / "again.vcobj").mappingJson == result.mappingJson, "deterministic mapping");
        // A replica with unpublished edits is refused and leaves nothing behind.
        sql(source, "CREATE TABLE hub_upstream(singleton INTEGER PRIMARY KEY); CREATE TABLE hub_pending(record_uuid TEXT PRIMARY KEY, "
                    "counter INTEGER, geometry_after INTEGER); INSERT INTO hub_pending VALUES('x',1,0);");
        const auto refused = directory.path / "refused.vcobj";
        mustFail([&] { (void)migration::convertStore(source, refused); });
        require(!std::filesystem::exists(refused) && !std::filesystem::exists(refused.string() + ".converting"), "no partial destination");
        sql(source, "DELETE FROM hub_pending;");
        const auto settled = Json::parse(migration::convertStore(source, directory.path / "settled.vcobj").reportJson);
        require(settled.at("source_role") == "replica" && settled.at("removed_tables").size() == 2, "coordination removed");
    }
    // Schema 16 (the patches detail table, hub sync baselines): its form of
    // the schema-17 fixture. Surface data is refused; without it the history
    // converts as from schema 17.
    {
        const auto source = directory.path / "old16.vcobj";
        sql(source, bytesOf(kData / "convert_schema17.sql"));
        sql(source, "DROP TRIGGER surfaces_history_INSERT; DROP TRIGGER surfaces_history_UPDATE; DROP TRIGGER surfaces_history_DELETE;"
                    "ALTER TABLE surfaces RENAME TO patches; ALTER TABLE patches DROP COLUMN area;"
                    "CREATE TABLE sync_baselines(seq INTEGER PRIMARY KEY AUTOINCREMENT, peer TEXT NOT NULL, token TEXT NOT NULL,"
                    " state TEXT NOT NULL, UNIQUE(peer,token));"
                    "CREATE TABLE sync_baseline_records(peer TEXT NOT NULL, record_uuid TEXT NOT NULL, baseline_seq INTEGER NOT NULL,"
                    " state TEXT NOT NULL, PRIMARY KEY(peer,record_uuid,baseline_seq));"
                    "CREATE TABLE sync_baseline_cursors(baseline_seq INTEGER PRIMARY KEY, journal_seq INTEGER NOT NULL, geometry_seq INTEGER NOT NULL);"
                    "PRAGMA user_version=16;");
        const auto refused = directory.path / "refused16.vcobj";
        mustFail([&] { (void)migration::convertStore(source, refused); }); // a surface
        require(!std::filesystem::exists(refused) && !std::filesystem::exists(refused.string() + ".converting"), "nothing left");
        sql(source, "DELETE FROM patches; UPDATE objects SET kind='test_surface' WHERE kind='surface';"
                    "DELETE FROM history_dirty_objects; DELETE FROM history_dirty_edges;"
                    "INSERT INTO sync_baselines(peer,token,state) VALUES('p','t','{}');");
        mustFail([&] { (void)migration::convertStore(source, refused); }); // unsettled baselines
        sql(source, "DELETE FROM sync_baselines;");
        const auto destination = directory.path / "new16.vcobj";
        const auto result = migration::convertStore(source, destination);
        const auto report = Json::parse(result.reportJson);
        require(report.at("source_schema") == 16 && report.at("validation") == "ok" && report.contains("upgraded_from_schema_16") &&
                scalar(destination, "SELECT count(*) FROM sqlite_master WHERE name LIKE 'sync_baseline%' OR name='patches'") == 0,
                "schema 16 upgraded");
        // The unedited schema-17 conversion.
        const auto reference = directory.path / "again.vcobj";
        for (const auto* query : {"SELECT count(*) FROM object_versions", "SELECT count(*) FROM relationship_versions",
                                  "SELECT count(*) FROM store_changes", "SELECT count(*) FROM store_commits"})
            require(scalar(destination, query) == scalar(reference, query), "same numbered history as schema 17");
        requireWritable(destination, expected.at("objects").at(1));
    }
    std::cout << "object store convert: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "object store convert failed: " << error.what() << "\n";
    return 1;
}
