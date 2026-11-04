// Offline schema-17 conversion: numbered history reconstructed from recorded
// publication order equals the numbering the runtime assigns, the source is
// untouched, gaps are reported, and unsettled replicas are refused.
#include "TestSupport.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/migration/ConvertStore.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/TransactionJson.hpp"

#include <sqlite3.h>
#include <nlohmann/json.hpp>

#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>

using namespace vc::objects;
using Json = nlohmann::json;

namespace {
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

// Schema 18 is schema 17 plus the numbered-history tables; removing them
// reproduces the previous format exactly.
void downgrade(const std::filesystem::path& path)
{
    sql(path, "PRAGMA journal_mode=DELETE; DROP TABLE transaction_receipts; DROP TABLE store_changes; "
              "DROP TABLE relationship_versions; DROP TABLE object_versions; DROP TABLE store_commits; "
              "DROP TABLE history_generations; DROP TABLE store_history; PRAGMA user_version=17;");
}

std::string bytesOf(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

struct Numbered {
    std::map<std::string, std::vector<std::tuple<int64_t, std::optional<int64_t>, bool>>> objects;
    std::map<std::string, std::optional<int64_t>> relationshipHeads;
    std::vector<std::tuple<int64_t, std::string, int64_t, bool>> feed;
    int64_t sequence = 0;
};

Numbered numbered(ObjectStore& db, const std::vector<std::string>& objects, const std::vector<std::string>& edges)
{
    Numbered result;
    auto txn = db.read();
    for (const auto& object : objects)
        for (const auto& revision : db.objectHistory(txn, object))
            result.objects[object].emplace_back(revision.version, revision.commitSequence, revision.deleted);
    for (const auto& edge : edges) result.relationshipHeads[edge] = db.currentRelationshipVersion(txn, edge);
    const auto status = db.status(txn);
    result.sequence = status.commitSequence;
    for (const auto& commit : db.changesSince(txn, {status.storeUuid, status.generation, 0, 0}, 1 << 20).commits)
        for (const auto& change : commit.changes)
            result.feed.emplace_back(commit.sequence, change.uuid, change.version, change.deleted);
    return result;
}
}

int main()
try {
    TemporaryDirectory directory;
    const auto source = directory.path / "old.vcobj";
    OpenOptions options; options.author = "convert-test";
    std::vector<std::string> objects, edges;
    Numbered expected;
    std::string storeUuid, firstRevision;
    {
        auto db = ObjectStore::create(source, options);
        std::string a, b, gone;
        {
            auto txn = db.write();
            auto add = [&](int n) {
                ObjectRecord record; record.kind = "track"; record.track.emplace();
                record.track->datasetRef = "convert"; record.track->externalKey = util::newUuid();
                const auto uuid = db.upsertObject(txn, record);
                if (n == 0) return uuid; // deletable: no retained geometry
                vc::geometry::Track track; for (int i = 0; i < n; ++i) track.points.push_back({double(i), 1, 2});
                db.putGeometry(txn, uuid, vc::geometry::encode(track));
                return uuid;
            };
            a = add(3); b = add(2); gone = add(0);
            txn.commit();
        }
        objects = {a, b, gone};
        auto heads = [&](std::vector<std::string> uuids) {
            auto txn = db.read(); VersionSelection selection;
            for (const auto& uuid : uuids) selection[uuid] = db.currentVersion(txn, uuid);
            return selection;
        };
        storeUuid = [&] { auto txn = db.read(); return db.uuid(txn); }();
        auto submit = [&](VersionSelection expected, std::vector<TransactionOperation> operations) {
            TransactionRequest r; r.requestUuid = util::newUuid(); r.storeUuid = storeUuid;
            r.generation = [&] { auto txn = db.read(); return db.status(txn).generation; }();
            r.author = "convert-test"; r.expectedObjects = std::move(expected); r.operations = std::move(operations);
            return db.submit(r);
        };
        submit(heads({a}), {SetMetaOp{a, "note", "\"first\""}});
        const auto edge = util::newUuid(), other = util::newUuid();
        EdgeRecord link; link.uuid = edge; link.kind = "test_link"; link.aUuid = a; link.bUuid = b;
        link.aAnchor.wholeGeometry = link.bAnchor.wholeGeometry = true;
        EdgeRecord second = link; second.uuid = other; second.kind = "other_link";
        submit(heads({a, b}), {AddEdgeOp{edgeRecordJson(link)}, AddEdgeOp{edgeRecordJson(second)}});
        std::vector<uint8_t> nine(8); const double value = 9; std::memcpy(nine.data(), &value, 8);
        submit(heads({a, b}), {EditGeometryOp{a, {{"points", 0, nine}}, {{edge, "retain", "{}", "{}"}, {other, "keep_recorded", "{}", "{}"}}}});
        submit(heads({a, b}), {RemoveEdgeOp{other}});
        submit(heads({gone}), {RetireObjectOp{gone}});
        submit(heads({gone}), {DeleteObjectOp{gone}});
        // Copy provenance that names an old revision UUID in metadata.
        firstRevision = [&] { auto txn = db.read(); return db.objectHistory(txn, a).front().uuid; }();
        submit(heads({b}), {SetMetaOp{b, "copied_from", Json{{"object_revision", firstRevision}}.dump()}});
        edges = {edge, other};
        expected = numbered(db, objects, edges);
    }
    downgrade(source);
    const auto before = bytesOf(source);
    mustFail([&] { (void)ObjectStore::open(source, options); }); // the runtime refuses schema 17

    const auto destination = directory.path / "new.vcobj";
    const auto result = migration::convertStore(source, destination);
    require(bytesOf(source) == before, "source untouched");
    const auto report = Json::parse(result.reportJson);
    const auto mapping = Json::parse(result.mappingJson);
    require(report.at("validation") == "ok" && report.at("source_role") == "standalone", "report");
    require(report.at("gaps").at("metadata_revision_references").at("count") == 2, "provenance reference reported");
    require(report.at("gaps").at("untimed_object_versions").at("count") == 0, "no untimed history");
    mustFail([&] { (void)migration::convertStore(source, destination); }); // never overwrites

    {
        auto db = ObjectStore::open(destination, options);
        const auto converted = numbered(db, objects, edges);
        require(converted.objects == expected.objects, "object versions and commits match runtime numbering");
        require(converted.relationshipHeads == expected.relationshipHeads, "relationship versions match");
        require(converted.feed == expected.feed && converted.sequence == expected.sequence, "feed matches");
        auto txn = db.read();
        const auto status = db.status(txn);
        require(status.storeUuid == storeUuid && status.generation == mapping.at("generation").get<std::string>(), "identity kept");
        const auto ref = mapping.at("objects").at(firstRevision);
        const auto revision = db.objectAtVersion(txn, {storeUuid, status.generation, ref.at("object"), ref.at("version")});
        require(revision && revision->uuid == firstRevision, "mapping resolves");
    }
    // Deterministic: converting again yields the same mapping.
    const auto again = directory.path / "again.vcobj";
    require(migration::convertStore(source, again).mappingJson == result.mappingJson, "deterministic mapping");

    // A replica with unpublished edits is refused and leaves no destination.
    sql(source, "CREATE TABLE hub_upstream(singleton INTEGER PRIMARY KEY); CREATE TABLE hub_pending(record_uuid TEXT PRIMARY KEY, "
                "counter INTEGER, geometry_after INTEGER); INSERT INTO hub_pending VALUES('x',1,0);");
    const auto refused = directory.path / "refused.vcobj";
    mustFail([&] { (void)migration::convertStore(source, refused); });
    require(!std::filesystem::exists(refused), "no partial destination");
    sql(source, "DELETE FROM hub_pending;");
    const auto settled = migration::convertStore(source, directory.path / "settled.vcobj");
    const auto settledReport = Json::parse(settled.reportJson);
    require(settledReport.at("source_role") == "replica" && settledReport.at("removed_tables").size() == 2, "coordination removed");
    std::cout << "object store convert: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "object store convert failed: " << error.what() << "\n";
    return 1;
}
