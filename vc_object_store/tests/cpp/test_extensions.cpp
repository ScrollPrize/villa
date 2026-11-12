#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include "vc_object_store/objects/StoreConversion.hpp"
#include "internal/Database.hpp"
#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>

using namespace vc::objects;
namespace {
using Json = nlohmann::json;
const std::string firstUuid = "00000000-0000-4000-8000-000000000001";
const std::string secondUuid = "00000000-0000-4000-8000-000000000002";

ExtensionTable definition()
{
    ExtensionTable table;
    table.name = "ext_test_measurements";
    table.extension = "test.measurements";
    table.formatVersion = 3;
    table.provisioning = R"({"units":{"residual":"windings"},"source":"fixture"})";
    table.basedOn = {{"10000000-0000-4000-8000-000000000001",
                      "20000000-0000-4000-8000-000000000001", 40}};
    table.columns = {{"fiber_uuid", ExtensionType::Blob}, {"run", ExtensionType::Integer},
                     {"residual", ExtensionType::Real}, {"label", ExtensionType::Text},
                     {"payload", ExtensionType::Blob, true}, {"optional_value", ExtensionType::Real, true}};
    table.primaryKey = {"fiber_uuid", "run"};
    table.indexes = {{{"label", "run"}, false}};
    table.keySpace = "objects";
    return table;
}

ExtensionRow sample(const std::string& uuid, int64_t run)
{
    return {extensionUuid(uuid), run, -1.25, std::string("measured"),
            std::vector<uint8_t>{0, 1, 127, 128, 255}, std::monostate{}};
}

std::vector<ExtensionRow> rows(ObjectStore& store, const Txn& txn, const std::string& table,
                               std::span<const ExtensionValue> key = {})
{
    std::vector<ExtensionRow> result;
    store.scanExtensionRows(txn, table, [&](const auto& row) { result.push_back(row); return true; }, key);
    return result;
}

std::vector<ExtensionRow> rows(ObjectCollection& collection, const CollectionTxn& txn, const std::string& table,
                               std::span<const ExtensionValue> key = {})
{
    std::vector<ExtensionRow> result;
    collection.scanExtensionRows(txn, table, [&](const auto& row) { result.push_back(row); return true; }, key);
    return result;
}

int64_t sequence(ObjectStore& store)
{
    auto txn = store.read();
    return store.status(txn).commitSequence;
}

void sourceReferences()
{
    const auto spec = definition();
    require(extensionSourcesFromJson(extensionSourcesJson(spec.basedOn)) == spec.basedOn,
            "source provenance round trips independently from destination publication");
    require(extensionSourcesFromJson(extensionSourcesJson({})).empty(), "empty source set round trips");
    mustFail([&] { extensionSourcesFromJson("{}"); });
    auto invalid = spec.basedOn;
    invalid[0].commitSequence = -1;
    mustFail([&] { extensionSourcesJson(invalid); });
    invalid = spec.basedOn; invalid[0].storeUuid = "not-a-uuid";
    mustFail([&] { extensionSourcesJson(invalid); });
    require(extensionUuid(extensionUuid(firstUuid)) == firstUuid, "UUIDs round trip through binary extension values");
    require(std::get<std::vector<uint8_t>>(extensionUuid(firstUuid)).size() == 16,
            "extension identity uses binary UUID rather than textual encoding");
    mustFail([&] { extensionUuid(ExtensionValue(std::vector<uint8_t>(15))); });
    mustFail([&] { extensionUuid(ExtensionValue(std::string(firstUuid))); });
}

void storeChanges(HistoryMode history)
{
    TemporaryDirectory dir;
    OpenOptions options; options.author = "extension-test"; options.history = history;
    const auto path = dir.path / "extensions.vcobj";
    auto store = ObjectStore::create(path, options);
    const auto spec = definition();
    {
        auto txn = store.read();
        require(store.extensionTables(txn).empty(), "plain stores need no extension payloads");
        require(store.provisioning(txn).schemaVersion == 22, "extension support retains core schema 22");
    }
    { auto txn = store.write(); store.createExtensionTable(txn, spec); txn.commit(); }
    require(sequence(store) == 1, "extension registration publishes one store commit");
    { auto txn = store.write(); store.createExtensionTable(txn, spec); txn.commit(); }
    require(sequence(store) == 1, "identical registration is an idempotent no-op");
    std::string tableUuid;
    {
        auto txn = store.read();
        const auto tables = store.extensionTables(txn);
        require(tables.size() == 1, "one registered physical extension table");
        const auto& info = tables.front(); tableUuid = info.tableUuid;
        require(info.name == spec.name && info.extension == spec.extension && info.formatVersion == 3,
                "extension identity and format version are not core schema version");
        require(info.columns == spec.columns && info.primaryKey == spec.primaryKey && info.keySpace == spec.keySpace,
                "typed table declaration and key layout remain inspectable");
        require(info.basedOn == spec.basedOn && Json::parse(info.provisioning) == Json::parse(spec.provisioning),
                "extension owns provisioning while input pins retain their common representation");
        require(info.lastModifiedCommit == 1, "new empty table records creation publication");
        require(rows(store, txn, spec.name).empty(), "registered table begins empty");
        mustFail([&] { store.exportJson(txn); });
        require(store.objectCount(txn, "fiber", true) == 0, "extension rows do not create proxy objects");
        const auto identities = store.databaseIdentity(txn);
        require(std::any_of(identities.tables.begin(), identities.tables.end(), [&](const auto& table) {
            return table.name == spec.name && table.uuid == tableUuid;
        }), "extension uses the existing physical table identity");
    }
    const auto first = sample(firstUuid, 1), second = sample(firstUuid, 2), third = sample(secondUuid, 1);
    {
        auto txn = store.write();
        const std::vector<ExtensionRow> batch{second, first};
        store.appendExtensionRows(txn, spec.name, batch);
        store.appendExtensionRows(txn, spec.name, std::span<const ExtensionRow>(&third, 1));
        txn.commit();
    }
    require(sequence(store) == 2, "multiple batches share one destination commit");
    {
        auto txn = store.read();
        require(rows(store, txn, spec.name) == std::vector<ExtensionRow>{first, second, third},
                "rows preserve typed values and scan in primary-key order");
        const std::vector<ExtensionValue> prefix{extensionUuid(firstUuid)};
        require(rows(store, txn, spec.name, prefix) == std::vector<ExtensionRow>{first, second},
                "UUID key-prefix lookup finds multiple runs without duplicating source fibers");
        const std::vector<ExtensionValue> exact{extensionUuid(firstUuid), int64_t(2)};
        require(rows(store, txn, spec.name, exact) == std::vector<ExtensionRow>{second},
                "complete composite key selects an exact result");
        size_t visited = 0;
        store.scanExtensionRows(txn, spec.name, [&](const auto&) { ++visited; return false; });
        require(visited == 1, "streaming scans obey early termination");
    }
    { auto txn = store.write(); store.appendExtensionRows(txn, spec.name, {}); txn.commit(); }
    require(sequence(store) == 2, "empty extension batches are no-ops");
    const auto fourth = sample(secondUuid, 2);
    {
        auto txn = store.write();
        store.appendExtensionRows(txn, spec.name, std::span<const ExtensionRow>(&fourth, 1));
    }
    require(sequence(store) == 2, "rolled-back rows do not publish a destination commit");
    {
        auto txn = store.write();
        const std::vector<ExtensionRow> duplicate{fourth, first};
        mustFail([&] { store.appendExtensionRows(txn, spec.name, duplicate); });
        txn.commit();
    }
    require(sequence(store) == 2, "failed append rolls back its earlier rows and dirty stamp");
    {
        auto txn = store.read();
        require(rows(store, txn, spec.name).size() == 3, "failed batch cannot leave a partial result");
        const auto info = store.extensionTables(txn).front();
        require(info.lastModifiedCommit == 2 && info.basedOn == spec.basedOn && info.tableUuid == tableUuid,
                "rollback preserves publication stamp, source references and table identity");
    }
    {
        auto txn = store.write();
        ObjectRecord object; object.uuid = firstUuid; object.kind = "fiber";
        store.upsertObject(txn, object); txn.commit();
    }
    require(sequence(store) == 3, "ordinary core edits still publish independently");
    {
        auto txn = store.read(); const auto info = store.extensionTables(txn).front();
        require(info.lastModifiedCommit == 2 && info.basedOn.front().commitSequence == 40,
                "unrelated commits never claim results were recomputed against newer inputs");
        require(store.objectByUuid(txn, firstUuid).has_value(), "core consumers ignore an unfamiliar extension");
        require(!Json::parse(store.objectByUuid(txn, firstUuid)->meta).contains("vcobj_extensions"),
                "immutable results do not add owner snapshot metadata");
        auto* db = ObjectStoreImplAccess::conn(txn);
        detail::Statement chunks(db, "SELECT COUNT(*) FROM geometry_chunks");
        require(chunks.row() && chunks.i64(0) == 0,
                "immutable result imports store typed rows without redundant history payloads");
        mustFail([&] { store.exportJson(txn); });
        detail::Statement value(db, "SELECT typeof(fiber_uuid),typeof(run),typeof(residual),typeof(label),typeof(payload),"
                                   "typeof(optional_value) FROM ext_test_measurements LIMIT 1");
        require(value.row() && value.text(0) == "blob" && value.text(1) == "integer" && value.text(2) == "real" &&
                value.text(3) == "text" && value.text(4) == "blob" && value.text(5) == "null",
                "measurements are typed SQLite columns, not per-row JSON");
    }
    {
        auto pinned = store.read();
        { auto txn = store.write(); store.appendExtensionRows(txn, spec.name, std::span<const ExtensionRow>(&fourth, 1)); txn.commit(); }
        require(rows(store, pinned, spec.name).size() == 3, "extension reads keep the caller's pinned snapshot");
    }
    require(sequence(store) == 4, "later result append publishes a separate commit");
    {
        auto txn = store.read(); require(rows(store, txn, spec.name).size() == 4, "new snapshot sees committed extension results");
        require(store.extensionTables(txn).front().lastModifiedCommit == 4, "table stamp follows the writing commit");
    }
    { auto txn = store.write(); store.seal(txn); txn.commit(); }
    mustFail([&] { auto txn = store.write(); });
    auto readOnly = ObjectStore::open(path, OpenOptions{.readOnly = true});
    auto txn = readOnly.read();
    require(rows(readOnly, txn, spec.name).size() == 4, "sealed/read-only stores retain readable extensions");
    require(readOnly.extensionTables(txn).front().tableUuid == tableUuid, "reopening preserves extension identity");
}

void invalidDefinitionsAndRows()
{
    TemporaryDirectory dir;
    auto store = ObjectStore::create(dir.path / "invalid.vcobj", OpenOptions{.author = "extension-test"});
    const auto spec = definition();
    const auto rejected = [&](auto mutate) {
        auto invalid = spec; mutate(invalid);
        auto txn = store.write(); mustFail([&] { store.createExtensionTable(txn, invalid); }); txn.commit();
        require(sequence(store) == 0, "invalid definitions do not leave metadata or dirty state");
    };
    rejected([](auto& table) { table.name = "objects"; });
    rejected([](auto& table) { table.name = "ext_bad; DROP TABLE objects"; });
    rejected([](auto& table) { table.formatVersion = 0; });
    rejected([](auto& table) { table.extension.clear(); });
    rejected([](auto& table) { table.provisioning = "[]"; });
    rejected([](auto& table) { table.columns.push_back(table.columns.front()); });
    rejected([](auto& table) { table.primaryKey = {"absent_column"}; });
    rejected([](auto& table) { table.indexes = {{{"absent_column"}, false}}; });
    rejected([](auto& table) { table.keySpace = "arbitrary"; });
    { auto txn = store.write(); store.createExtensionTable(txn, spec); txn.commit(); }
    {
        auto txn = store.write(); auto changed = spec; ++changed.formatVersion;
        mustFail([&] { store.createExtensionTable(txn, changed); }); txn.commit();
    }
    require(sequence(store) == 1, "registered extension schema cannot silently change interpretation");
    {
        auto txn = store.write(); auto changed = spec; ++changed.basedOn.front().commitSequence;
        mustFail([&] { store.createExtensionTable(txn, changed); }); txn.commit();
    }
    require(sequence(store) == 1, "registration cannot silently relabel the source snapshot of old results");
    const auto invalidRow = [&](auto mutate) {
        auto row = sample(firstUuid, 1); mutate(row);
        auto txn = store.write(); mustFail([&] {
            store.appendExtensionRows(txn, spec.name, std::span<const ExtensionRow>(&row, 1));
        }); txn.commit();
        require(sequence(store) == 1, "invalid typed row does not publish a result");
    };
    invalidRow([](auto& row) { row.pop_back(); });
    invalidRow([](auto& row) { row[1] = std::string("1"); });
    invalidRow([](auto& row) { row[2] = std::numeric_limits<double>::infinity(); });
    invalidRow([](auto& row) { row[3] = std::monostate{}; });
    auto txn = store.read(); require(rows(store, txn, spec.name).empty(), "invalid batches leave the table empty");
}

void emptyBlobRoundTrip()
{
    TemporaryDirectory dir;
    const auto path = dir.path / "empty-blob.vcobj";
    const auto spec = definition();
    auto empty = sample(firstUuid, 1), absent = sample(firstUuid, 2);
    empty[4] = std::vector<uint8_t>{};
    absent[4] = std::monostate{};
    const std::vector<ExtensionRow> expected{empty, absent};
    {
        auto store = ObjectStore::create(path, OpenOptions{.author = "extension-test"});
        auto txn = store.write();
        store.createExtensionTable(txn, spec);
        store.appendExtensionRows(txn, spec.name, expected);
        txn.commit();
    }
    auto store = ObjectStore::open(path, OpenOptions{.readOnly = true});
    auto txn = store.read();
    require(rows(store, txn, spec.name) == expected,
            "zero-byte BLOB and NULL remain distinct through insertion and reopen");
    detail::Statement sql(ObjectStoreImplAccess::conn(txn),
        "SELECT typeof(payload),length(payload),payload IS NULL FROM ext_test_measurements ORDER BY run");
    require(sql.row() && sql.text(0) == "blob" && sql.i64(1) == 0 && sql.i64(2) == 0,
            "empty binary payload is a genuine SQLite BLOB, not a null pointer binding");
    require(sql.row() && sql.text(0) == "null" && sql.i64(2) == 1,
            "absent binary payload remains SQLite NULL");
}

void initialReservedMetadata()
{
    TemporaryDirectory dir;
    InitialImportOptions options;
    options.store.author = "extension-test";
    const auto reserved = R"({"vcobj_extensions":{"ext_forged":{"encoding":"cbor-rows-v1","chunks":[]}}})";
    for (const bool borrowed : {false, true}) {
        const auto path = dir.path / (borrowed ? "borrowed-object.vcobj" : "owned-object.vcobj");
        mustFail([&] {
            importInitialCollection(path, options, [&](size_t, InitialShardImport& shard) {
                InitialObjectRecord record;
                record.record.uuid = firstUuid;
                record.record.kind = "fiber";
                record.record.meta = reserved;
                if (borrowed) {
                    const InitialObjectRecord* pointer = &record;
                    shard.appendObjectRefs(std::span<const InitialObjectRecord* const>(&pointer, 1));
                } else {
                    shard.appendObjects(std::span<const InitialObjectRecord>(&record, 1));
                }
            });
        });
        require(!std::filesystem::exists(path), "initial object import rejects forged manual snapshots before publication");
    }
    const auto path = dir.path / "reserved-edge.vcobj";
    mustFail([&] {
        importInitialCollection(path, options,
            [&](size_t, InitialShardImport& shard) {
                for (const auto& uuid : {firstUuid, secondUuid}) {
                    InitialObjectRecord record;
                    record.record.uuid = uuid; record.record.kind = "fiber";
                    shard.appendObjects(std::span<const InitialObjectRecord>(&record, 1));
                }
            },
            [&](size_t, InitialShardImport& shard) {
                bool sent = false;
                shard.importEdges([&](std::vector<InitialCollectionEdge>& batch) {
                    if (sent) return false;
                    sent = true;
                    InitialCollectionEdge edge;
                    edge.a = {shard.storeUuid(), firstUuid}; edge.b = {shard.storeUuid(), secondUuid};
                    edge.record.uuid = "00000000-0000-4000-8000-000000000003";
                    edge.record.kind = "declared_reference";
                    edge.record.aAnchor.wholeGeometry = edge.record.bAnchor.wholeGeometry = true;
                    edge.record.meta = reserved;
                    batch.push_back(std::move(edge));
                    return true;
                });
            });
    });
    require(!std::filesystem::exists(path), "initial edge import rejects forged manual snapshots before publication");
    require(std::filesystem::is_empty(dir.path), "reserved-metadata failures clean their unpublished construction files");
}

void historyPolicy()
{
    TemporaryDirectory dir; const auto path = dir.path / "history.vcobj";
    OpenOptions options; options.author = "extension-test"; options.history = HistoryMode::Off;
    const auto spec = definition(); const std::vector<ExtensionRow> expected{sample(firstUuid, 1)};
    std::string tableUuid;
    {
        auto store = ObjectStore::create(path, options);
        { auto txn = store.write(); store.createExtensionTable(txn, spec); store.appendExtensionRows(txn, spec.name, expected); txn.commit(); }
        auto txn = store.read(); tableUuid = store.extensionTables(txn).front().tableUuid;
    }
    for (const auto mode : {HistoryMode::On, HistoryMode::Off}) {
        changeHistoryMode(path, mode);
        auto store = ObjectStore::open(path, options); auto txn = store.read();
        require(store.provisioning(txn).history == mode, "history policy conversion succeeds with typed extensions");
        require(rows(store, txn, spec.name) == expected, "policy conversion preserves every extension value");
        const auto info = store.extensionTables(txn).front();
        require(info.tableUuid == tableUuid && info.basedOn == spec.basedOn && info.lastModifiedCommit == 1,
                "physical history conversion does not rewrite extension provenance/publication");
    }
}

ExtensionTable manualDefinition(const std::string& keySpace = "objects")
{
    auto table = definition();
    table.name = "ext_test_manual";
    table.extension = "test.manual";
    table.versioned = true;
    table.keySpace = keySpace;
    table.basedOn.clear();
    return table;
}

ObjectVersion objectHead(ObjectStore& store, const std::string& id)
{
    auto txn = store.read();
    return store.versionReference(txn, *store.currentObjectRevision(txn, id));
}

template<class Version>
std::vector<ExtensionRow> historicalRows(ObjectStore& store, const Txn& txn,
                                         const std::string& table, const Version& version)
{
    std::vector<ExtensionRow> result;
    store.scanExtensionRowsAtVersion(txn, table, version,
        [&](const auto& row) { result.push_back(row); return true; });
    return result;
}

void versionedObject(HistoryMode history)
{
    TemporaryDirectory dir;
    auto store = ObjectStore::create(dir.path / "manual.vcobj",
                                    OpenOptions{.history = history, .author = "extension-test"});
    const auto spec = manualDefinition();
    {
        auto txn = store.write();
        store.createExtensionTable(txn, spec);
        ObjectRecord object; object.uuid = firstUuid; object.kind = "fiber";
        store.upsertObject(txn, object); txn.commit();
    }
    const auto emptyHead = objectHead(store, firstUuid);
    require(emptyHead.version == 1, "manual fields reuse the ordinary object version clock");
    {
        auto txn = store.read();
        require(store.extensionTables(txn).front().versioned, "owner-history policy remains discoverable");
        require(historicalRows(store, txn, spec.name, emptyHead).empty(),
                "a current object version with no manual fields reads an empty extension");
    }
    std::vector<ExtensionRow> first{sample(firstUuid, 1), sample(firstUuid, 2)};
    first.front()[4] = std::vector<uint8_t>{};
    first.back()[4] = std::monostate{};
    {
        auto txn = store.write(VersionSelection{{firstUuid, emptyHead.version}});
        store.replaceExtensionRows(txn, spec.name, firstUuid, first); txn.commit();
    }
    const auto firstHead = objectHead(store, firstUuid);
    require(firstHead.version == 2 && sequence(store) == 2,
            "first manual edit advances owner and store once without a separate version counter");
    {
        auto txn = store.read();
        require(rows(store, txn, spec.name) == first, "manual fields remain typed queryable current rows");
        require(historicalRows(store, txn, spec.name, firstHead) == first,
                "manual current-version snapshot agrees with typed current rows");
        require(store.extensionTables(txn).front().lastModifiedCommit == 2,
                "manual field table records its actual publication commit");
        require(Json::parse(store.objectByUuid(txn, firstUuid)->meta).contains("vcobj_extensions"),
                "manual snapshot participates in the owning object's metadata history");
    }
    {
        auto txn = store.write(VersionSelection{{firstUuid, firstHead.version}});
        const std::vector<ExtensionRow> reversed{first.back(), first.front()};
        store.replaceExtensionRows(txn, spec.name, firstUuid, reversed); txn.commit();
    }
    require(objectHead(store, firstUuid) == firstHead && sequence(store) == 2,
            "identical manual replacement in a different input order neither versions the owner nor publishes a commit");
    {
        auto txn = store.write(VersionSelection{{firstUuid, firstHead.version}});
        const std::vector<ExtensionRow> duplicate{first.front(), first.front()};
        mustFail([&] { store.replaceExtensionRows(txn, spec.name, firstUuid, duplicate); });
        const std::vector<ExtensionRow> wrongOwner{sample(secondUuid, 1)};
        mustFail([&] { store.replaceExtensionRows(txn, spec.name, firstUuid, wrongOwner); });
        mustFail([&] { store.appendExtensionRows(txn, spec.name, first); });
        txn.commit();
    }
    require(objectHead(store, firstUuid) == firstHead && sequence(store) == 2,
            "failed owner replacements roll back their rows, snapshot and version changes");
    std::vector<ExtensionRow> changed{sample(firstUuid, 3)};
    auto& payload = std::get<std::vector<uint8_t>>(changed.front()[4]);
    payload.resize(1024 * 1024 + 17);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = uint8_t(i % 251);
    changed.front()[3] = std::string("large binary manual value");
    {
        auto txn = store.write(VersionSelection{{firstUuid, firstHead.version}});
        store.replaceExtensionRows(txn, spec.name, firstUuid, {});
        store.replaceExtensionRows(txn, spec.name, firstUuid, changed); txn.commit();
    }
    const auto changedHead = objectHead(store, firstUuid);
    require(changedHead.version == 3 && sequence(store) == 3,
            "several edits in one transaction publish only one owner version");
    {
        auto txn = store.read();
        require(rows(store, txn, spec.name) == changed &&
                historicalRows(store, txn, spec.name, changedHead) == changed,
                "large binary extension values survive snapshot chunking without JSON conversion");
        detail::Statement chunks(ObjectStoreImplAccess::conn(txn), "SELECT COUNT(*) FROM geometry_chunks");
        require(chunks.row() && chunks.i64(0) >= 2,
                "manual snapshots reuse existing immutable chunk storage, including large payloads");
        if (history == HistoryMode::On) {
            require(historicalRows(store, txn, spec.name, firstHead) == first,
                    "old manual values follow the historical owner version");
            require(historicalRows(store, txn, spec.name, emptyHead).empty(),
                    "versions before the first manual edit remain empty");
        } else {
            mustFail([&] { historicalRows(store, txn, spec.name, firstHead); });
        }
    }
    mustFail([&] { auto txn = store.write(VersionSelection{{firstUuid, firstHead.version}}); });
    {
        auto txn = store.write(VersionSelection{{firstUuid, changedHead.version}});
        store.replaceExtensionRows(txn, spec.name, firstUuid, first);
    }
    require(objectHead(store, firstUuid) == changedHead && sequence(store) == 3,
            "transaction rollback restores the manual owner head");
    {
        auto txn = store.read();
        require(rows(store, txn, spec.name) == changed, "transaction rollback also restores typed rows");
    }
    {
        auto txn = store.write(VersionSelection{{firstUuid, changedHead.version}});
        mustFail([&] { store.setMeta(txn, firstUuid, "vcobj_extensions", "{}"); });
        mustFail([&] { store.removeMeta(txn, firstUuid, "vcobj_extensions"); });
        auto object = *store.objectByUuid(txn, firstUuid); object.meta = "{}";
        mustFail([&] { store.upsertObject(txn, object); });
        txn.commit();
    }
    require(sequence(store) == 3, "generic metadata edits cannot detach library-managed manual snapshots");
    if (history == HistoryMode::On) {
        {
            auto txn = store.write(VersionSelection{{firstUuid, changedHead.version}});
            store.restoreObject(txn, firstUuid, firstHead); txn.commit();
        }
        {
            auto txn = store.read();
            require(rows(store, txn, spec.name) == first, "restoring an object restores its typed manual fields");
            require(store.currentVersion(txn, firstUuid) == 4, "restore creates a new ordinary owner version");
        }
        {
            auto txn = store.write(VersionSelection{{firstUuid, int64_t(4)}});
            store.restoreObject(txn, firstUuid, emptyHead); txn.commit();
        }
        auto txn = store.read();
        require(rows(store, txn, spec.name).empty(), "restoring a pre-extension version clears its current manual fields");
        require(historicalRows(store, txn, spec.name, changedHead) == changed,
                "restoration does not discard a later historical binary payload");
    }
}

void versionedRelationship(HistoryMode history)
{
    TemporaryDirectory dir;
    auto store = ObjectStore::create(dir.path / "manual-edge.vcobj",
                                    OpenOptions{.history = history, .author = "extension-test"});
    const auto spec = manualDefinition("edges");
    const auto edgeUuid = util::newUuid();
    {
        auto txn = store.write(); store.createExtensionTable(txn, spec);
        for (const auto& id : {firstUuid, secondUuid}) {
            ObjectRecord object; object.uuid = id; object.kind = "fiber"; store.upsertObject(txn, object);
        }
        txn.commit();
    }
    {
        auto txn = store.write(VersionSelection{{firstUuid, int64_t(1)}, {secondUuid, int64_t(1)}});
        EdgeRecord edge; edge.uuid = edgeUuid; edge.kind = "declared_reference"; edge.aUuid = firstUuid; edge.bUuid = secondUuid;
        edge.aAnchor.wholeGeometry = edge.bAnchor.wholeGeometry = true;
        store.addEdge(txn, edge); txn.commit();
    }
    StoreStatus status;
    { auto txn = store.read(); status = store.status(txn); }
    const RelationshipVersion emptyHead{status.storeUuid, status.generation, edgeUuid, 1};
    const std::vector<ExtensionRow> first{sample(edgeUuid, 1)};
    TransactionContext expected;
    expected.expectedObjects = {{firstUuid, int64_t(2)}, {secondUuid, int64_t(2)}};
    expected.expectedRelationships = {{edgeUuid, int64_t(1)}};
    {
        auto txn = store.write(expected); store.replaceExtensionRows(txn, spec.name, edgeUuid, first); txn.commit();
    }
    const RelationshipVersion firstHead{status.storeUuid, status.generation, edgeUuid, 2};
    {
        auto txn = store.read();
        require(store.currentRelationshipVersion(txn, edgeUuid) == 2 && store.currentVersion(txn, firstUuid) == 3 &&
                store.currentVersion(txn, secondUuid) == 3,
                "manual relationship fields reuse normal relationship and endpoint versioning");
        require(historicalRows(store, txn, spec.name, firstHead) == first,
                "relationship-version references retrieve typed manual fields");
    }
    expected.expectedObjects = {{firstUuid, int64_t(3)}, {secondUuid, int64_t(3)}};
    expected.expectedRelationships = {{edgeUuid, int64_t(2)}};
    const auto beforeNoop = sequence(store);
    {
        auto txn = store.write(expected); store.replaceExtensionRows(txn, spec.name, edgeUuid, first); txn.commit();
    }
    require(sequence(store) == beforeNoop, "identical edge extension replacement does not version either endpoint");
    auto changed = first; changed.front()[2] = 2.5;
    {
        auto txn = store.write(expected); store.replaceExtensionRows(txn, spec.name, edgeUuid, changed); txn.commit();
    }
    {
        auto txn = store.read();
        require(store.currentRelationshipVersion(txn, edgeUuid) == 3 && store.currentVersion(txn, firstUuid) == 4 &&
                store.currentVersion(txn, secondUuid) == 4, "second edge extension edit versions all normal participants once");
        require(rows(store, txn, spec.name) == changed, "current edge extension values stay queryable");
        if (history == HistoryMode::On) {
            require(historicalRows(store, txn, spec.name, firstHead) == first, "old edge fields retain their exact values");
            require(historicalRows(store, txn, spec.name, emptyHead).empty(), "initial edge version had no manual rows");
        } else {
            mustFail([&] { historicalRows(store, txn, spec.name, firstHead); });
        }
    }
    expected.expectedObjects = {{firstUuid, int64_t(4)}, {secondUuid, int64_t(4)}};
    mustFail([&] { auto txn = store.write(expected); }); // stale relationship expectation
    expected.expectedRelationships = {{edgeUuid, int64_t(3)}};
    {
        auto txn = store.write(expected); store.replaceExtensionRows(txn, spec.name, edgeUuid, {});
    }
    auto txn = store.read();
    require(rows(store, txn, spec.name) == changed && store.currentRelationshipVersion(txn, edgeUuid) == 3,
            "rolled-back edge edits restore typed data and the relationship version");
}

void versionedHistoryPolicy()
{
    TemporaryDirectory dir; const auto path = dir.path / "manual-policy.vcobj";
    const auto spec = manualDefinition();
    const std::vector<ExtensionRow> first{sample(firstUuid, 1)}, second{sample(firstUuid, 2)};
    ObjectVersion oldHead;
    {
        auto store = ObjectStore::create(path, OpenOptions{.author = "extension-test"});
        {
            auto txn = store.write(); store.createExtensionTable(txn, spec);
            ObjectRecord object; object.uuid = firstUuid; object.kind = "fiber";
            store.upsertObject(txn, object);
            store.replaceExtensionRows(txn, spec.name, firstUuid, first); txn.commit();
        }
        oldHead = objectHead(store, firstUuid);
        {
            auto txn = store.write(VersionSelection{{firstUuid, oldHead.version}});
            store.replaceExtensionRows(txn, spec.name, firstUuid, second); txn.commit();
        }
    }
    changeHistoryMode(path, HistoryMode::Off);
    {
        auto store = ObjectStore::open(path, OpenOptions{.author = "extension-test"});
        const auto current = objectHead(store, firstUuid);
        auto txn = store.read();
        require(rows(store, txn, spec.name) == second && historicalRows(store, txn, spec.name, current) == second,
                "removing history retains current manual snapshot chunks and typed values");
        mustFail([&] { historicalRows(store, txn, spec.name, oldHead); });
    }
    changeHistoryMode(path, HistoryMode::On);
    {
        auto store = ObjectStore::open(path, OpenOptions{.author = "extension-test"});
        const auto baseline = objectHead(store, firstUuid);
        {
            auto txn = store.write(VersionSelection{{firstUuid, baseline.version}});
            store.replaceExtensionRows(txn, spec.name, firstUuid, first); txn.commit();
        }
        auto txn = store.read();
        require(historicalRows(store, txn, spec.name, baseline) == second && rows(store, txn, spec.name) == first,
                "reenabling history preserves the manual-field baseline for subsequent normal edits");
    }
}

void storeReferenceVersionedFields(HistoryMode history)
{
    TemporaryDirectory dir;
    const auto basePath = dir.path / "base.vcobj";
    const auto overlayPath = dir.path / "overlay.vcobj";
    const auto objectTable = manualDefinition();
    auto edgeTable = manualDefinition("edges");
    edgeTable.name = "ext_test_manual_edges";
    const auto immutableTable = definition();
    const std::string edgeUuid = "00000000-0000-4000-8000-000000000003";
    const std::vector<ExtensionRow> first{sample(firstUuid, 1), sample(firstUuid, 2)};
    const std::vector<ExtensionRow> second{sample(secondUuid, 1)};
    const std::vector<ExtensionRow> edges{sample(edgeUuid, 1), sample(edgeUuid, 2)};
    const std::vector<ExtensionRow> immutable{sample(firstUuid, 9)};
    OpenOptions options; options.history = history; options.author = "extension-test";
    std::string baseUuid;
    {
        auto base = ObjectStore::create(basePath, options);
        auto txn = base.write();
        baseUuid = base.uuid(txn);
        for (const auto& table : {objectTable, edgeTable, immutableTable}) base.createExtensionTable(txn, table);
        for (const auto& uuid : {firstUuid, secondUuid}) {
            ObjectRecord object; object.uuid = uuid; object.kind = "fiber";
            base.upsertObject(txn, object);
        }
        base.replaceExtensionRows(txn, objectTable.name, firstUuid, first);
        base.replaceExtensionRows(txn, objectTable.name, secondUuid, second);
        EdgeRecord edge; edge.uuid = edgeUuid; edge.kind = "declared_reference";
        edge.aUuid = firstUuid; edge.bUuid = secondUuid;
        edge.aAnchor.wholeGeometry = edge.bAnchor.wholeGeometry = true;
        base.addEdge(txn, edge);
        base.replaceExtensionRows(txn, edgeTable.name, edgeUuid, edges);
        base.appendExtensionRows(txn, immutableTable.name, immutable);
        base.seal(txn); txn.commit();
    }
    const ObjectKey firstKey{baseUuid, firstUuid}, secondKey{baseUuid, secondUuid}, edgeKey{baseUuid, edgeUuid};
    const std::vector<ExtensionValue> firstPrefix{extensionUuid(firstUuid)};
    const std::vector<ExtensionValue> edgePrefix{extensionUuid(edgeUuid)};
    auto overlay = ObjectCollection::create(overlayPath, {basePath}, options);
    int64_t initialCommit = 0;
    {
        auto txn = overlay.read();
        initialCommit = overlay.provisioning(txn).commitSequence;
        require(rows(overlay, txn, objectTable.name, firstPrefix) == first &&
                rows(overlay, txn, edgeTable.name, edgePrefix) == edges,
                "referenced owner fields are visible before creating a local owner");
    }
    {
        auto txn = overlay.write();
        const std::vector<ExtensionRow> reversed{first.back(), first.front()};
        overlay.replaceExtensionRows(txn, objectTable.name, firstKey, reversed);
        overlay.replaceExtensionRows(txn, edgeTable.name, edgeKey, edges);
        txn.commit();
    }
    {
        auto txn = overlay.read();
        require(overlay.provisioning(txn).commitSequence == initialCommit,
                "inherited object and relationship no-ops do not publish a new commit");
        auto local = ObjectStore::open(overlayPath, OpenOptions{.physicalPartitionOnly = true, .readOnly = true});
        auto localTxn = local.read();
        require(!local.objectByUuid(localTxn, firstUuid) && !local.objectByUuid(localTxn, secondUuid) &&
                !local.edgeByUuid(localTxn, edgeUuid),
                "inherited no-ops roll back temporary localization rather than creating root owner rows");
        require(local.extensionTables(localTxn).empty(),
                "inherited no-ops do not leave unused root extension definitions");
    }
    {
        auto txn = overlay.write();
        overlay.setMeta(txn, firstKey, "note", "\"local metadata change\"");
        txn.commit();
    }
    {
        auto txn = overlay.read();
        require(rows(overlay, txn, objectTable.name, firstPrefix) == first,
                "localizing an owner for ordinary metadata preserves its typed manual fields");
    }
    std::vector<ExtensionRow> changed{first.front()};
    changed.front()[3] = std::string("local replacement");
    {
        auto txn = overlay.write();
        overlay.replaceExtensionRows(txn, objectTable.name, firstKey, changed);
        txn.commit();
    }
    {
        auto txn = overlay.read();
        require(rows(overlay, txn, objectTable.name, firstPrefix) == changed,
                "owner replacement shadows every inherited row, not just duplicate primary keys");
        const std::vector<ExtensionValue> removedKey{extensionUuid(firstUuid), int64_t(2)};
        require(rows(overlay, txn, objectTable.name, removedKey).empty(),
                "exact lookup cannot resurrect a removed inherited owner row");
    }
    {
        auto txn = overlay.write();
        overlay.replaceExtensionRows(txn, objectTable.name, firstKey, {});
    }
    {
        auto txn = overlay.read();
        require(rows(overlay, txn, objectTable.name, firstPrefix) == changed,
                "rolled-back store_reference clear restores owner visibility and values");
    }
    {
        auto txn = overlay.write();
        overlay.replaceExtensionRows(txn, objectTable.name, firstKey, {});
        txn.commit();
    }
    {
        auto txn = overlay.read();
        require(rows(overlay, txn, objectTable.name, firstPrefix).empty(),
                "explicit empty local owner fields hide inherited values");
        require(rows(overlay, txn, objectTable.name) == second,
                "full scans use the same owner precedence as UUID-prefix scans");
        require(rows(overlay, txn, immutableTable.name) == immutable,
                "immutable computed results survive independent owner edits with their original provenance");
    }
    std::vector<ExtensionRow> changedEdge{edges.back()};
    changedEdge.front()[2] = 3.5;
    {
        auto txn = overlay.write();
        overlay.replaceExtensionRows(txn, edgeTable.name, edgeKey, changedEdge);
        txn.commit();
    }
    {
        auto txn = overlay.read();
        require(rows(overlay, txn, edgeTable.name, edgePrefix) == changedEdge,
                "relationship manual fields also replace the complete inherited owner row set");
    }
    {
        auto txn = overlay.write();
        overlay.removeEdge(txn, edgeKey);
        overlay.deleteObject(txn, firstKey);
        overlay.deleteObject(txn, secondKey);
        txn.commit();
    }
    {
        auto txn = overlay.read();
        require(!overlay.object(txn, firstKey) && !overlay.object(txn, secondKey) && !overlay.edge(txn, edgeKey),
                "overlay tombstones hide the original object and relationship keys");
        require(rows(overlay, txn, objectTable.name).empty() && rows(overlay, txn, edgeTable.name).empty(),
                "owner tombstones suppress inherited versioned rows in all-layer scans");
        require(rows(overlay, txn, objectTable.name, firstPrefix).empty() &&
                rows(overlay, txn, edgeTable.name, edgePrefix).empty(),
                "owner tombstones also suppress inherited versioned rows in routed scans");
        require(rows(overlay, txn, immutableTable.name) == immutable,
                "historical immutable computation results are not mistaken for editable current owner fields");
    }
    {
        auto base = ObjectStore::open(basePath, OpenOptions{.readOnly = true});
        auto txn = base.read();
        const std::vector<ExtensionValue> prefix{extensionUuid(firstUuid)};
        require(rows(base, txn, objectTable.name, prefix) == first && rows(base, txn, edgeTable.name) == edges,
                "overlay replacement, clear and deletion leave the sealed base fields unchanged");
    }
    for (const bool relationship : {false, true}) {
        const auto mismatchPath = dir.path / (relationship ? "edge-mismatch.vcobj" : "object-mismatch.vcobj");
        auto mismatch = ObjectCollection::create(mismatchPath, {basePath}, options);
        auto incompatible = relationship ? edgeTable : objectTable;
        incompatible.provisioning = R"({"units":{"residual":"base_voxels"},"source":"fixture"})";
        {
            auto txn = mismatch.write(); mismatch.createExtensionTable(txn, incompatible); txn.commit();
        }
        int64_t before = 0;
        {
            auto txn = mismatch.read(); before = mismatch.provisioning(txn).commitSequence;
            mustFail([&] { rows(mismatch, txn, incompatible.name); });
            mustFail([&] { rows(mismatch, txn, incompatible.name, relationship ? edgePrefix : firstPrefix); });
        }
        {
            auto txn = mismatch.write();
            mustFail([&] {
                mismatch.replaceExtensionRows(txn, incompatible.name, relationship ? edgeKey : firstKey,
                                              relationship ? edges : first);
            });
            txn.commit();
        }
        {
            auto txn = mismatch.read();
            require(mismatch.provisioning(txn).commitSequence == before,
                    "mismatched extension conventions reject writes without publishing changes");
            auto local = ObjectStore::open(mismatchPath, OpenOptions{.physicalPartitionOnly = true, .readOnly = true});
            auto localTxn = local.read();
            require(!local.objectByUuid(localTxn, firstUuid) && !local.objectByUuid(localTxn, secondUuid) &&
                    !local.edgeByUuid(localTxn, edgeUuid),
                    "mismatched extension conventions fail before materializing inherited owners");
        }
    }
}

void shardedOverlayScan(bool packed)
{
    TemporaryDirectory dir;
    const auto basePath = dir.path / "base.vcobj";
    auto uuidInShard = [](const std::string& prefix, size_t shard) {
        for (size_t i = 0;; ++i) {
            auto uuid = util::nameUuid(prefix + std::to_string(i));
            if (uuidShardIndex(uuid, 2) == shard) return uuid;
        }
    };
    // Deterministic UUIDs ensure every replacement is in the later shard.
    // Loading shard 0 visits the shared base before loading shard 1, which
    // must not make the base win a collection-wide scan.
    const auto objectUuid = uuidInShard("extension-object-", 1);
    const auto otherUuid = uuidInShard("extension-other-", 0);
    const auto edgeUuid = uuidInShard("extension-edge-", 1);
    const auto objectTable = definition();
    auto edgeTable = definition();
    edgeTable.name = "ext_test_edge_results"; edgeTable.keySpace = "edges";
    auto unkeyedTable = definition();
    unkeyedTable.name = "ext_test_unrouted_results"; unkeyedTable.keySpace.clear();
    const std::vector<ExtensionTable> tables{objectTable, edgeTable, unkeyedTable};
    auto changedObject = sample(objectUuid, 1); changedObject[2] = 8.5;
    auto changedEdge = sample(edgeUuid, 1); changedEdge[2] = 9.5;
    const std::vector<ExtensionRow> objectReplacement{changedObject}, edgeReplacement{changedEdge};
    OpenOptions storeOptions; storeOptions.history = HistoryMode::Off; storeOptions.author = "extension-test";
    {
        auto base = ObjectStore::create(basePath, storeOptions);
        auto txn = base.write();
        for (const auto& table : tables) base.createExtensionTable(txn, table);
        for (const auto& uuid : {objectUuid, otherUuid}) {
            ObjectRecord object; object.uuid = uuid; object.kind = "fiber";
            base.upsertObject(txn, object);
        }
        EdgeRecord edge; edge.uuid = edgeUuid; edge.kind = "perpendicular";
        edge.aUuid = objectUuid; edge.bUuid = otherUuid;
        edge.aAnchor.wholeGeometry = edge.bAnchor.wholeGeometry = true;
        base.addEdge(txn, edge);
        const std::vector<ExtensionRow> objectRows{sample(objectUuid, 1), sample(otherUuid, 1)};
        const std::vector<ExtensionRow> edgeRows{sample(edgeUuid, 1)};
        base.appendExtensionRows(txn, objectTable.name, objectRows);
        base.appendExtensionRows(txn, edgeTable.name, edgeRows);
        base.appendExtensionRows(txn, unkeyedTable.name, objectRows);
        base.seal(txn); txn.commit();
    }
    InitialImportOptions options;
    options.store.author = "extension-overlay-test";
    options.dependencies = {basePath}; options.extensions = tables;
    options.shards = {{{1, 100}}, {{101, 200}}};
    if (packed) {
        options.packed.emplace(); options.packed->memoryFirst = true;
        options.packed->disposableBulk = true; options.packed->backgroundFlush = false;
        options.packed->residentBytes = 64 * 1024 * 1024;
    }
    const auto result = importInitialCollection(dir.path / "derived", options,
        [&](size_t i, InitialShardImport& shard) {
            if (i != 1) return;
            shard.appendExtensionRows(objectTable.name, objectReplacement);
            shard.appendExtensionRows(edgeTable.name, edgeReplacement);
            shard.appendExtensionRows(unkeyedTable.name, objectReplacement);
        });
    const auto verify = [&](ObjectCollection& collection) {
        auto txn = collection.read();
        for (const auto& table : tables) {
            const bool isEdge = table.keySpace == "edges";
            const auto& uuid = isEdge ? edgeUuid : objectUuid;
            const auto& replacement = isEdge ? edgeReplacement : objectReplacement;
            const std::vector<ExtensionValue> prefix{extensionUuid(uuid)};
            require(rows(collection, txn, table.name, prefix) == replacement,
                    "prefix scans resolve replacement rows from the later sibling shard");
            auto all = rows(collection, txn, table.name);
            auto expected = replacement;
            if (!isEdge) expected.push_back(sample(otherUuid, 1));
            std::sort(all.begin(), all.end()); std::sort(expected.begin(), expected.end());
            require(all == expected,
                    "full extension scans resolve all sibling replacements before shared dependencies");
            size_t count = 0;
            collection.scanExtensionRows(txn, table.name, [&](const auto& row) {
                require(row == replacement.front(), "early scan termination must not return a stale base row");
                ++count; return false;
            });
            require(count == 1, "early termination applies across all partitions and bases");
        }
    };
    auto collection = ObjectCollection::open(result.path, OpenOptions{.readOnly = true});
    verify(collection);
    auto wrapper = ObjectCollection::create(dir.path / "wrapper.vcobj", {result.path}, storeOptions);
    verify(wrapper);
}

void shardedImport(bool packed)
{
    TemporaryDirectory dir;
    constexpr size_t count = 4;
    InitialImportOptions options;
    options.store.author = "extension-import-test";
    options.shards.clear();
    for (size_t i = 0; i < count; ++i)
        options.shards.push_back({{int64_t(i * 100 + 1), int64_t((i + 1) * 100)}});
    if (packed) {
        options.packed.emplace(); options.packed->memoryFirst = true;
        options.packed->disposableBulk = true; options.packed->backgroundFlush = false;
        options.packed->residentBytes = 64 * 1024 * 1024;
    }
    const auto objectTable = definition();
    ExtensionTable edgeTable;
    edgeTable.name = "ext_test_edge_measurements"; edgeTable.extension = "test.measurements";
    edgeTable.columns = {{"edge_uuid", ExtensionType::Blob}, {"las_winding", ExtensionType::Real},
                         {"normal_valid", ExtensionType::Integer}};
    edgeTable.primaryKey = {"edge_uuid"}; edgeTable.keySpace = "edges";
    edgeTable.basedOn = objectTable.basedOn;
    options.extensions = {objectTable, edgeTable};
    std::vector<std::string> objects(count), edges(count);
    for (size_t i = 0; i < count; ++i) {
        do { objects[i] = util::newUuid(); } while (uuidShardIndex(objects[i], count) != i);
        do { edges[i] = util::newUuid(); } while (uuidShardIndex(edges[i], count) != i);
    }
    const auto result = importInitialCollection(dir.path / "import", options,
        [&](size_t i, InitialShardImport& shard) {
            InitialObjectRecord row;
            row.record.uuid = objects[i]; row.record.kind = "fiber"; row.record.fiber.emplace();
            row.extensions.push_back({objectTable.name, sample(objects[i], 1)});
            shard.appendObjects(std::span<const InitialObjectRecord>(&row, 1));
        },
        [&](size_t i, InitialShardImport& shard) {
            bool sent = false;
            shard.importEdges([&](std::vector<InitialCollectionEdge>& batch) {
                if (sent) return false;
                sent = true;
                InitialCollectionEdge row;
                row.a = {shard.storeUuid(), objects[i]}; row.b = {shard.storeUuid(), objects[(i + 1) % count]};
                row.record.uuid = edges[i]; row.record.kind = "perpendicular";
                row.record.aAnchor.wholeGeometry = row.record.bAnchor.wholeGeometry = true;
                row.extensions.push_back({edgeTable.name, {extensionUuid(edges[i]), 1.3, int64_t(0)}});
                batch.push_back(std::move(row)); return true;
            });
        });
    auto collection = ObjectCollection::open(result.path, OpenOptions{.readOnly = true});
    auto txn = collection.read();
    const auto tables = collection.extensionTables(txn);
    require(std::any_of(tables.begin(), tables.end(), [&](const auto& info) { return info.name == objectTable.name; }) &&
            std::any_of(tables.begin(), tables.end(), [&](const auto& info) { return info.name == edgeTable.name; }),
            "collection discovers extension tables stored in physical shards");
    for (size_t i = 0; i < count; ++i) {
        std::vector<ExtensionRow> found;
        const std::vector<ExtensionValue> objectKey{extensionUuid(objects[i])};
        collection.scanExtensionRows(txn, objectTable.name, [&](const auto& row) { found.push_back(row); return true; }, objectKey);
        require(found == std::vector<ExtensionRow>{sample(objects[i], 1)},
                "colocated object extension routes by the same UUID as its core object");
        found.clear();
        const std::vector<ExtensionValue> edgeKey{extensionUuid(edges[i])};
        collection.scanExtensionRows(txn, edgeTable.name, [&](const auto& row) { found.push_back(row); return true; }, edgeKey);
        require(found == std::vector<ExtensionRow>{{extensionUuid(edges[i]), 1.3, int64_t(0)}},
                "constraint extension preserves one fractional measurement per native edge across shards");
        require(collection.edge(txn, {result.storeUuid, edges[i]}).has_value(),
                "extension import retains the ordinary cross-shard relationship");
    }
    size_t all = 0;
    collection.scanExtensionRows(txn, edgeTable.name, [&](const auto&) { ++all; return true; });
    require(all == count, "collection-wide extension scan includes all shards once");
    size_t first = 0;
    collection.scanExtensionRows(txn, objectTable.name, [&](const auto&) { ++first; return false; });
    require(first == 1, "early termination stops the composed scan, not only one shard");
    auto storeReference = ObjectCollection::create(dir.path / "store-reference.vcobj", {result.path},
                                          OpenOptions{.history = HistoryMode::Off, .author = "extension-test"});
    auto referenceTxn = storeReference.read();
    size_t referenceCount = 0;
    storeReference.scanExtensionRows(referenceTxn, edgeTable.name, [&](const auto&) { ++referenceCount; return true; });
    require(referenceCount == count, "store_references expose extension rows without copying the source graph");
    const std::vector<ExtensionValue> referencePrefix{extensionUuid(objects.front())};
    std::vector<ExtensionRow> referenceRows;
    storeReference.scanExtensionRows(referenceTxn, objectTable.name,
        [&](const auto& row) { referenceRows.push_back(row); return true; }, referencePrefix);
    require(referenceRows == std::vector<ExtensionRow>{sample(objects.front(), 1)},
            "store_reference UUID-prefix lookup preserves typed results through plain and packed partitions");
}
} // namespace

int main()
{
    sourceReferences();
    storeChanges(HistoryMode::Off);
    storeChanges(HistoryMode::On);
    invalidDefinitionsAndRows();
    emptyBlobRoundTrip();
    initialReservedMetadata();
    historyPolicy();
    versionedObject(HistoryMode::Off);
    versionedObject(HistoryMode::On);
    versionedRelationship(HistoryMode::Off);
    versionedRelationship(HistoryMode::On);
    versionedHistoryPolicy();
    storeReferenceVersionedFields(HistoryMode::Off);
    storeReferenceVersionedFields(HistoryMode::On);
    shardedOverlayScan(false);
    shardedOverlayScan(true);
    shardedImport(false);
    shardedImport(true);
}
