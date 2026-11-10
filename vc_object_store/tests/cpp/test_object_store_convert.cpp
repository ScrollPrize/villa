// Offline conversion of upstream schema-17 through schema-20 stores.
// The fixtures (data/convert_schema17.sql, data/convert_schema18.sql) were
// written by the schema-18 runtime from one history (data/convert_schema19.sql
// is the schema-18 fixture converted by the last schema-19 converter): tracks with geometry,
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
#include "vc_object_store/objects/Artifacts.hpp"
#include "vc_object_store/objects/StoreConversion.hpp"

#include <sqlite3.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <utility>

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

// Add artifacts to an authentic frozen upstream fixture, not a current store
// whose user_version was merely changed to impersonate an old schema.
size_t fixtureArtifacts(const std::filesystem::path& path, const std::string& object,
                       const std::string& first, const std::string& later)
{
    sqlite3* db = nullptr;
    require(sqlite3_open(path.string().c_str(), &db) == SQLITE_OK, "open artifact fixture");
    const auto close = std::unique_ptr<sqlite3, decltype(&sqlite3_close)>(db, sqlite3_close);
    sqlite3_stmt* rows = nullptr;
    require(sqlite3_prepare_v2(db, "SELECT name FROM sqlite_master WHERE type='trigger'", -1, &rows, nullptr) == SQLITE_OK, "fixture triggers");
    std::vector<std::string> triggers;
    while (sqlite3_step(rows) == SQLITE_ROW) triggers.emplace_back(reinterpret_cast<const char*>(sqlite3_column_text(rows, 0)));
    sqlite3_finalize(rows);
    for (const auto& trigger : triggers)
        require(sqlite3_exec(db, ("DROP TRIGGER \"" + trigger + "\"").c_str(), nullptr, nullptr, nullptr) == SQLITE_OK, "drop fixture trigger");
    const auto key = util::canonicalUuid(object);
    auto compact = key; compact.erase(std::remove(compact.begin(), compact.end(), '-'), compact.end());
    const auto query = "SELECT uuid,parent_uuid,metadata_delta FROM object_revisions WHERE object_uuid=x'" + compact + "' ORDER BY seq";
    require(sqlite3_prepare_v2(db, query.c_str(), -1, &rows, nullptr) == SQLITE_OK, "fixture revisions");
    std::vector<std::pair<std::vector<uint8_t>, Json>> rewritten;
    Json old = Json::object(), previous = Json::object(), current;
    while (sqlite3_step(rows) == SQLITE_ROW) {
        const auto delta = Json::parse(reinterpret_cast<const char*>(sqlite3_column_text(rows, 2)));
        old = delta.is_object() ? delta : old.patch(delta);
        current = old;
        current["meta"][kArtifactsMetaKey] = Json::parse(rewritten.empty() ? first : later);
        const auto* bytes = static_cast<const uint8_t*>(sqlite3_column_blob(rows, 0));
        const bool root = sqlite3_column_type(rows, 1) == SQLITE_NULL;
        rewritten.emplace_back(std::vector<uint8_t>(bytes, bytes + sqlite3_column_bytes(rows, 0)), root ? current : Json::diff(previous, current));
        previous = current;
    }
    sqlite3_finalize(rows);
    require(rewritten.size() >= 2, "fixture has two artifact versions");
    sqlite3_stmt* update = nullptr;
    require(sqlite3_prepare_v2(db, "UPDATE object_revisions SET metadata_delta=? WHERE uuid=?", -1, &update, nullptr) == SQLITE_OK, "update fixture revisions");
    for (const auto& [uuid, delta] : rewritten) {
        const auto json = delta.dump();
        sqlite3_bind_text(update, 1, json.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_blob(update, 2, uuid.data(), int(uuid.size()), SQLITE_TRANSIENT);
        require(sqlite3_step(update) == SQLITE_DONE, "write fixture revision");
        sqlite3_reset(update);
    }
    sqlite3_finalize(update);
    const auto writeJson = [&](const std::string& statement, const Json& value) {
        sqlite3_stmt* target = nullptr;
        require(sqlite3_prepare_v2(db, statement.c_str(), -1, &target, nullptr) == SQLITE_OK, "fixture current update");
        const auto text = value.dump(); sqlite3_bind_text(target, 1, text.c_str(), -1, SQLITE_TRANSIENT);
        const auto result = sqlite3_step(target); sqlite3_finalize(target);
        require(result == SQLITE_DONE, "write fixture current record");
    };
    writeJson("UPDATE object_head_records SET record=? WHERE object_uuid=x'" + compact + "'", current);
    writeJson("UPDATE objects SET meta=? WHERE uuid=x'" + compact + "'", current.at("meta"));
    return rewritten.size();
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
            want["store"] = expected.at("store");
            if (generation) want["generation"] = *generation;
            auto& record = want.at("record");
            record["approved_version"] = record.at("approved_generation").is_null() ? Json(nullptr) : want.at("version");
            record.erase("approved_generation"); record.erase("generation"); record.erase("data_generation");
            record["version"] = want.at("version");
            record["commit_sequence"] = want.at("commit_sequence").is_null() ? Json(0) : want.at("commit_sequence");
            record["version_generation"] = want.at("generation");
            require(Json::parse(objectRevisionToJson(history[i])) == want, ("revision " + std::to_string(i) + " of " + object.get<std::string>()).c_str());
        }
    }
    for (const auto& [uuid, geometry] : expected.at("geometry").items()) {
        const auto g = db.geometryRevision(txn, uuid);
        auto descriptor = geometry.at("descriptor");
        descriptor.erase("object_generation");
        int64_t version = 0;
        for (const auto& history : expected.at("history"))
            for (const auto& state : history)
                if (state.at("geometry_revision") == uuid && (!version || state.at("version").get<int64_t>() < version))
                    version = state.at("version").get<int64_t>();
        descriptor["object_version"] = version;
        require(g && Json::parse(geometryRevisionToJson(*g)) == descriptor, ("geometry descriptor " + uuid).c_str());
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
    std::string publicationGeneration;
    { auto txn = db.read(); status = db.status(txn); before = db.currentVersion(txn, object);
      const auto revision = *db.currentObjectRevision(txn, object);
      oldRecord = Json::parse(objectRevisionToJson(revision)).at("record"); publicationGeneration = revision.generation; }
    TransactionRequest r; r.requestUuid = util::newUuid(); r.storeUuid = status.storeUuid; r.generation = status.generation;
    r.author = "convert-test"; r.expectedObjects = {{object, before}};
    r.operations = {SetMetaOp{object, "after_conversion", std::to_string(*before + 1)}};
    const auto receipt = db.submit(r);
    require(receipt.objects.at(object) == *before + 1, "edit after conversion");
    auto txn = db.read();
    const auto old = db.objectAtVersion(txn, {status.storeUuid, publicationGeneration, object, *before});
    require(old && Json::parse(objectRevisionToJson(*old)).at("record") == oldRecord, "old version reads back");
}
}

int main()
try {
    TemporaryDirectory directory;
    const auto expected = Json::parse(bytesOf(kData / "convert_expected.json"));
    OpenOptions options; options.author = "convert-test";
    for (const int schema : {17, 18, 19, 20}) {
        const auto source = directory.path / ("old" + std::to_string(schema) + ".vcobj");
        // Upstream schema 20 differs from 19 only by dropping replica identity.
        sql(source, bytesOf(kData / ("convert_schema" + std::to_string(std::min(schema, 19)) + ".sql")));
        if (schema == 20) sql(source, "DROP TABLE sync_replica; PRAGMA user_version=20;");
        const auto before = bytesOf(source);
        mustFail([&] { (void)ObjectStore::open(source, options); }); // the runtime refuses old schemas

        const auto destination = directory.path / ("new" + std::to_string(schema) + ".vcobj");
        const auto result = migration::convertStore(source, destination);
        require(bytesOf(source) == before, "source untouched");
        require(!std::filesystem::exists(destination.string() + ".converting"), "no staging file left");
        const auto report = Json::parse(result.reportJson);
        const auto mapping = Json::parse(result.mappingJson);
        require(report.at("validation") == "ok" && report.at("source_role") == "standalone", "report");
        require(report.at("source_schema") == schema && report.at("target_schema") == 21, "schemas reported");
        mustFail([&] { (void)migration::convertStore(source, destination); }); // never overwrites
        // Storage form: whole records for revisions without a parent, no
        // materialized head record duplicating one, 16-byte keys.
        require(scalar(destination, "SELECT count(*) FROM object_revisions WHERE parent_uuid IS NULL AND json_type(metadata_delta)<>'object'") == 0,
                "root revisions hold records");
        require(scalar(destination, "SELECT count(*) FROM object_head_records h JOIN object_revisions r ON r.uuid=h.revision_uuid "
                                    "WHERE r.parent_uuid IS NULL") == 0, "no duplicate head records");
        require(scalar(destination, "SELECT count(*) FROM object_versions WHERE length(object_uuid)<>16 OR length(revision_uuid)<>16") == 0,
                "blob keys");
        require(scalar(destination, "SELECT count(*) FROM sqlite_master WHERE name='sync_replica'") == 0,
                "no obsolete replica identity");
        if (schema < 20)
            require(std::find(report.at("removed_tables").begin(), report.at("removed_tables").end(), "sync_replica") !=
                    report.at("removed_tables").end(), "replica identity dropped");

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
        require(settled.at("source_role") == "replica", "replica conversion reported");
        for (const auto* table : {"hub_pending", "hub_upstream", "sync_replica"})
            require(std::find(settled.at("removed_tables").begin(), settled.at("removed_tables").end(), table) !=
                    settled.at("removed_tables").end(), "coordination removed");
    }
    // The unpublished experimental branch reused version 18, but it is not
    // upstream schema 18 and must never be treated as a migration input.
    {
        const auto source = directory.path / "unpublished18.vcobj";
        sql(source, bytesOf(kData / "convert_schema18.sql"));
        sql(source, "CREATE TABLE store_manifest(schema_version INTEGER); INSERT INTO store_manifest VALUES(18);");
        const auto before = bytesOf(source);
        const auto destination = directory.path / "unpublished-converted.vcobj";
        mustFail([&] { (void)migration::convertStore(source, destination); });
        require(bytesOf(source) == before && !std::filesystem::exists(destination) &&
                !std::filesystem::exists(destination.string() + ".converting"), "unpublished layout rejected without changes");
    }
    {
        const auto source = directory.path / "in-place20.vcobj";
        sql(source, bytesOf(kData / "convert_schema19.sql"));
        sql(source, "DROP TABLE sync_replica; PRAGMA user_version=20;");
        const auto receipts = scalar(source, "SELECT count(*) FROM transaction_receipts");
        sql(source, "CREATE VIEW extension_current AS SELECT uuid FROM objects;");
        const auto before = bytesOf(source);
        mustFail([&] { (void)migration::convertStoreInPlace(source); });
        require(bytesOf(source) == before && scalar(source, "PRAGMA user_version") == 20,
                "unsupported extension rejects in-place conversion without modifying source");
        sql(source, "DROP VIEW extension_current;");
        const auto result = migration::convertStoreInPlace(source);
        require(Json::parse(result.reportJson).at("in_place") == true && scalar(source, "PRAGMA user_version") == 21,
                "explicit in-place conversion publishes current schema");
        requireSameHistory(source, expected, std::nullopt);
        require(scalar(source, "SELECT count(*) FROM transaction_receipts") == receipts, "in-place receipts retained");
        requireWritable(source, expected.at("objects").at(1));
    }
    // Policy transitions retain numbered current identities, including the
    // old publication lineage after an intervening history-off lineage reset.
    {
        const auto path = directory.path / "policy-converted.vcobj";
        (void)migration::convertStore(directory.path / "old20.vcobj", path);
        const auto object = expected.at("objects").at(1).get<std::string>();
        requireWritable(path, object); // Produces a request receipt to retain.
        const auto receiptCount = scalar(path, "SELECT count(*) FROM transaction_receipts");
        require(receiptCount > 0, "policy fixture contains request receipts");
        ObjectRecord before;
        StoreStatus status;
        {
            auto store = ObjectStore::open(path, options);
            auto txn = store.read(); before = *store.objectByUuid(txn, object); status = store.status(txn);
        }
        changeHistoryMode(path, HistoryMode::Off);
        {
            auto store = ObjectStore::open(path, options);
            {
                auto txn = store.read(); const auto after = *store.objectByUuid(txn, object);
                require(after.version == before.version && after.versionGeneration == before.versionGeneration &&
                        after.commitSequence == before.commitSequence, "history removal retains record identity");
                require(store.status(txn).commitSequence == status.commitSequence + 1, "history removal advances policy clock once");
            }
            store.startHistoryGeneration("conversion policy regression");
        }
        require(scalar(path, "SELECT count(*) FROM sqlite_schema WHERE name IN ('object_versions','relationship_versions','store_commits')") == 0,
                "history-off has no archival numbered tables");
        require(scalar(path, "SELECT count(*) FROM transaction_receipts") == receiptCount, "history removal preserves idempotency receipts");
        changeHistoryMode(path, HistoryMode::On);
        {
            auto store = ObjectStore::open(path, options);
            auto txn = store.read(); const auto after = *store.objectByUuid(txn, object);
            require(after.version == before.version && after.versionGeneration == before.versionGeneration &&
                    after.commitSequence == before.commitSequence, "enabling history retains original publication identity");
            require(store.status(txn).commitSequence == status.commitSequence + 3, "each policy or lineage change advances clock once");
            const auto baseline = store.objectAtVersion(txn, {status.storeUuid, before.versionGeneration, object, *before.version});
            require(baseline && !baseline->commitSequence, "history enablement archives untimed current baseline without inventing a commit");
        }
        require(scalar(path, "SELECT count(*) FROM transaction_receipts") == receiptCount, "history enablement preserves idempotency receipts");
        requireWritable(path, object);
    }
    // Schema 19 differs from 20 only by sync_replica. Preserve artifacts
    // named solely by old versions too, without modifying the source.
    {
        const auto source = directory.path / "artifacts19.vcobj";
        const auto input = directory.path / "model.pt";
        std::string object, oldHash, currentHash;
        const std::string oldBytes = "old model", currentBytes = "new model";
        {
            auto store = ObjectStore::create(directory.path / "artifact-writer.vcobj", options);
            std::ofstream(input) << oldBytes;
            oldHash = store.putArtifact(input);
            std::ofstream(input) << currentBytes;
            currentHash = store.putArtifact(input);
        }
        sql(source, bytesOf(kData / "convert_schema19.sql"));
        object = expected.at("objects").at(1).get<std::string>();
        const auto historyCount = fixtureArtifacts(source, object,
            artifactManifestJson({{"model.pt", oldHash, oldBytes.size()}}),
            artifactManifestJson({{"model.pt", currentHash, currentBytes.size()}}));
        for (const auto& [hash, content] : {std::pair{oldHash, oldBytes}, std::pair{currentHash, currentBytes}}) {
            const auto blob = artifactBlobPath(source, hash);
            std::filesystem::create_directories(blob.parent_path());
            std::ofstream(blob, std::ios::binary) << content;
        }
        const auto before = bytesOf(source);
        const auto destination = directory.path / "converted-artifacts.vcobj";
        const auto result = migration::convertStore(source, destination);
        require(Json::parse(result.reportJson).at("copied_artifacts") == 2, "both artifact versions copied");
        {
            auto store = ObjectStore::open(destination, options);
            auto txn = store.read();
            const auto history = store.objectHistory(txn, object);
            require(history.size() == historyCount, "artifact history preserved");
            for (size_t i = 0; i < history.size(); ++i) {
                const auto manifest = artifactManifest(history[i].record);
                const auto bytes = store.readArtifact(manifest.at(0).sha256);
                require(std::string(bytes.begin(), bytes.end()) == (i == 0 ? oldBytes : currentBytes), "artifact bytes preserved");
            }
        }
        require(bytesOf(source) == before && bytesOf(artifactBlobPath(source, oldHash)) == oldBytes &&
                bytesOf(artifactBlobPath(source, currentHash)) == currentBytes, "artifact source untouched");
        changeHistoryMode(destination, HistoryMode::Off);
        changeHistoryMode(destination, HistoryMode::On);
        {
            auto store = ObjectStore::open(destination, options);
            auto txn = store.read();
            const auto manifest = artifactManifest(*store.objectByUuid(txn, object));
            const auto bytes = store.readArtifact(manifest.at(0).sha256);
            require(std::string(bytes.begin(), bytes.end()) == currentBytes,
                    "history-policy staging retains artifacts at the published store location");
        }

        const auto occupied = directory.path / "occupied.vcobj";
        const auto occupiedArtifacts = artifactBlobDirectory(occupied).parent_path();
        std::filesystem::create_directories(occupiedArtifacts);
        std::ofstream(occupiedArtifacts / "keep") << "keep";
        mustFail([&] { migration::convertStore(source, occupied); });
        require(bytesOf(occupiedArtifacts / "keep") == "keep", "existing artifacts untouched");

        const auto oldBlob = artifactBlobPath(source, oldHash);
        for (const bool missing : {false, true}) {
            std::filesystem::remove(oldBlob);
            if (!missing) std::ofstream(oldBlob) << "bad model"; // same length, different hash
            const auto refused = directory.path / (missing ? "missing.vcobj" : "corrupt.vcobj");
            mustFail([&] { migration::convertStore(source, refused); });
            require(!std::filesystem::exists(refused) && !std::filesystem::exists(refused.string() + ".converting") &&
                    !std::filesystem::exists(artifactBlobDirectory(refused).parent_path()), "failed artifact conversion cleaned up");
            const auto beforeFailure = bytesOf(source);
            mustFail([&] { (void)migration::convertStoreInPlace(source); });
            require(bytesOf(source) == beforeFailure && scalar(source, "PRAGMA user_version") == 19,
                    "failed in-place artifact validation leaves upstream source untouched");
        }
        std::ofstream(oldBlob, std::ios::binary) << oldBytes;
        (void)migration::convertStoreInPlace(source);
        {
            auto store = ObjectStore::open(source, options);
            auto txn = store.read();
            require(store.objectHistory(txn, object).size() == historyCount, "in-place artifact history retained");
            require(bytesOf(artifactBlobPath(source, oldHash)) == oldBytes &&
                    bytesOf(artifactBlobPath(source, currentHash)) == currentBytes, "in-place artifacts retain their published location");
        }
        for (const auto& entry : std::filesystem::directory_iterator(directory.path))
            require(!entry.path().filename().string().starts_with(source.filename().string() + ".convert-"),
                    "in-place private conversion files cleaned on success and failure");
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
