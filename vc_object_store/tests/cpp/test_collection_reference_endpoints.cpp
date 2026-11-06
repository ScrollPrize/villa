#include "TestSupport.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/RecordSql.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <atomic>

using namespace vc::objects;
using namespace vc::objects::detail;
using Json = nlohmann::json;

namespace {
OpenOptions directOptions()
{
    OpenOptions options;
    options.author = "reference-endpoint-test";
    options.history = HistoryMode::Off;
    options.spatialIndex = false;
    return options;
}

ObjectRecord fiber(const std::string& uuid, int quality)
{
    ObjectRecord record;
    record.uuid = uuid;
    record.kind = "fiber";
    record.fiber.emplace();
    record.meta = Json{{"quality", quality}}.dump();
    return record;
}

GeometryData geometry()
{
    vc::geometry::Fiber shape;
    shape.linePoints = {{0, 1, 2}, {3, 4, 5}};
    return vc::geometry::encode(shape);
}

struct Base {
    std::filesystem::path path;
    ObjectKey active, retired;
    std::vector<uint8_t> bytes;
};

Base makeBase(const std::filesystem::path& path, const std::string& sharedUuid, int quality)
{
    auto db = ObjectStore::create(path, directOptions());
    auto write = db.write();
    const auto owner = db.uuid(write);
    Base base{path, {owner, db.upsertObject(write, fiber(sharedUuid, quality))},
                   {owner, db.upsertObject(write, fiber(util::newUuid(), quality))}, {}};
    const auto revision = db.putGeometry(write, base.active.objectUuid, geometry());
    base.bytes = db.readGeometryArray(write, revision, "line_points");
    db.retireObject(write, base.retired.objectUuid);
    db.seal(write);
    write.commit();
    return base;
}

EdgeRecord observation(const std::string& key)
{
    EdgeRecord edge;
    edge.kind = "perpendicular";
    edge.relationKey = key;
    edge.meta = Json{{"distance", 1.25}, {"enabled", true}}.dump();
    edge.aAnchor.points = edge.bAnchor.points = {{"line_points", {1}}};
    edge.aAnchor.metadataKeys = edge.bAnchor.metadataKeys = {"quality"};
    return edge;
}

int64_t tableRows(sqlite3* db, const std::string& table)
{
    Statement exists(db, "SELECT 1 FROM sqlite_schema WHERE name=? AND type='table'", table);
    if (!exists.row()) return 0;
    // Callers use fixed internal table names, never input text.
    Statement count(db, "SELECT count(*) FROM " + table);
    require(count.row(), "table count returned no row");
    return count.i64(0);
}

void noEndpointCopies(const std::filesystem::path& path)
{
    auto db = ObjectStore::open(path, OpenOptions{.physicalPartitionOnly = true, .readOnly = true});
    auto read = db.read();
    auto* sql = ObjectStoreImplAccess::conn(read);
    for (const auto* table : {"objects", "fibers", "current_geometry", "geometry_chunks", "collection_refs"}) {
        const auto message = "reference-only edge copied endpoint state into " + std::string(table);
        require(tableRows(sql, table) == 0, message.c_str());
    }
    const auto edges=db.edgesByKind(read,"perpendicular",true);
    if(!edges.empty()){
        const auto& first=edges.front();
        require(!db.neighbours(read,first.aUuid,{}, {},true).empty(),"physical neighbour traversal dropped an external endpoint");
        require(db.componentOf(read,first.aUuid,{},nullptr,true).size()==2,"physical component traversal lost negative endpoint handles");
        mustFail([&]{db.neighbours(read,first.aUuid);});
        mustFail([&]{db.componentOf(read,first.aUuid);});
        mustFail([&]{db.neighbours(read,first.aUuid,{},"fiber",true);});
    }
}

void verify(ObjectCollection& collection, const Base& a, const Base& b, size_t count)
{
    auto read = collection.read();
    const auto edges = collection.edgesByKind(read, "perpendicular");
    require(edges.size() == count, "relation count mismatch");
    require(collection.edgesOf(read, a.active).size() == count &&
            collection.edgesOf(read, b.active).size() == count, "qualified endpoint adjacency mismatch");
    require(collection.edgesOf(read, a.retired).empty(), "unrelated retired endpoint gained edges");
    for (const auto& edge : edges) {
        require((edge.a == a.active && edge.b == b.active) ||
                (edge.a == b.active && edge.b == a.active), "same UUID in separate stores lost qualification");
        const auto direct = collection.edge(read, edge.key);
        require(direct && direct->a == edge.a && direct->b == edge.b, "edge key lookup lost endpoint routing");
        require(Json::parse(edge.record.meta).at("distance") == 1.25, "edge evidence changed");
    }
    require(collection.object(read, a.active)->key == a.active &&
            collection.object(read, b.active)->key == b.active, "external object reads do not preserve owner");
    require(collection.readGeometryArray(read, a.active, "line_points") == a.bytes &&
            collection.readGeometryArray(read, b.active, "line_points") == b.bytes, "geometry lookup failed through references");
    size_t objectCount = 0;
    collection.scanObjects(read, [&](const auto&) { ++objectCount; return true; }, "fiber");
    require(objectCount == 2, "endpoint references surfaced as duplicate active objects");
}

void externalReferences()
{
    TemporaryDirectory directory;
    const auto shared = util::newUuid();
    const auto a = makeBase(directory.path / "a", shared, 1);
    const auto b = makeBase(directory.path / "b", shared, 2);
    for (const bool bulk : {false, true}) {
        const auto output = directory.path / (bulk ? "bulk" : "scalar");
        auto collection = ObjectCollection::create(output, {a.path, b.path}, directOptions());
        {
            auto write = collection.write();
            const ObjectKey dangling{a.active.storeUuid, util::newUuid()};
            mustFail([&] { collection.addEdge(write, a.active, dangling, observation("missing")); });
            mustFail([&] { collection.addEdge(write, a.active, b.retired, observation("retired")); });
            require(collection.edgesByKind(write, "perpendicular").empty(), "rejected endpoint left partial edge");
            if (bulk) {
                bool produced = false;
                collection.importInitialEdges(write, [&](auto& batch) {
                    if (produced) return false;
                    produced = true;
                    batch.push_back({a.active, b.active, observation("one")});
                    batch.push_back({b.active, a.active, observation("two")});
                    return true;
                });
            } else {
                collection.addEdge(write, a.active, b.active, observation("one"));
                collection.addEdge(write, b.active, a.active, observation("two"));
            }
            write.commit();
        }
        noEndpointCopies(output);
        verify(collection, a, b, 2);
        {
            auto reopened = ObjectCollection::open(output, OpenOptions{.readOnly = true});
            verify(reopened, a, b, 2);
        }
        // An endpoint mutation is a real override, unlike merely referencing it.
        // Referenced metadata cannot silently invalidate the current relations.
        mustFail([&] {
            auto write = collection.write();
            collection.setMeta(write, a.active, "quality", "9");
            write.commit();
        });
        noEndpointCopies(output);
        {
            auto write = collection.write();
            for (const auto& edge : collection.edgesOf(write, a.active)) collection.removeEdge(write, edge.key);
            collection.setMeta(write, a.active, "quality", "9");
            write.commit();
        }
        {
            auto read = collection.read();
            require(collection.edgesOf(read, a.active).empty(), "explicitly removed external relation survived");
            require(Json::parse(collection.object(read, a.active)->record.meta).at("quality") == 9, "actual override missing");
            require(Json::parse(collection.object(read, b.active)->record.meta).at("quality") == 2, "other same-UUID endpoint changed");
        }
        auto base = ObjectStore::open(a.path, OpenOptions{.readOnly = true});
        auto read = base.read();
        require(Json::parse(base.objectByUuid(read, a.active.objectUuid)->meta).at("quality") == 1,
                "overlay changed its sealed base");
    }
}

void failedPublication()
{
    TemporaryDirectory directory;
    const auto base = makeBase(directory.path / "base", util::newUuid(), 1);
    for (const bool packed : {false, true}) {
        const auto name = std::string(packed ? "failed-packed" : "failed-plain");
        const auto output = directory.path / name;
        InitialImportOptions options;
        options.store = directOptions();
        options.dependencies = {base.path};
        if (packed) {
            PackedStoreOptions settings;
            settings.memoryFirst = true;
            settings.disposableBulk = true;
            settings.backgroundFlush = false;
            settings.residentBytes = 32 * 1024 * 1024;
            options.packed = settings;
        }
        mustFail([&] {
            importInitialCollection(output, options, [&](size_t, InitialShardImport& shard) {
                size_t batchNumber = 0;
                shard.importEdges([&](auto& batch) {
                    if (batchNumber == 2) return false;
                    const auto other = batchNumber++ == 0 ? base.active
                        : ObjectKey{base.active.storeUuid, util::newUuid()};
                    auto edge=observation("source-batch-"+std::to_string(batchNumber));
                    edge.bAnchor.points={{"line_points",{0}}};
                    batch.push_back({base.active,other,std::move(edge)});
                    return true;
                });
            });
        });
        require(!std::filesystem::exists(output), "dangling edge published an incomplete collection");
        for (const auto& entry : std::filesystem::directory_iterator(directory.path)) {
            const auto filename = entry.path().filename().string();
            require(!filename.starts_with("." + name + "-") && !filename.starts_with(name + ".building-"),
                    "failed collection left private construction files");
        }
    }
}

void aliasCollisions()
{
    TemporaryDirectory directory;
    const auto shared=util::newUuid();
    const auto a=makeBase(directory.path/"a",shared,1);
    const auto b=makeBase(directory.path/"b",shared,2);
    auto collection=ObjectCollection::create(directory.path/"alias-existing",{a.path,b.path},directOptions());
    std::string owner;
    {auto read=collection.read();owner=collection.uuid(read);}
    const auto alias=util::nameUuid(owner,"reference:"+a.active.storeUuid+":"+a.active.objectUuid);
    {
        auto write=collection.write();
        collection.addEdge(write,a.active,b.active,observation("first"));
        write.commit();
    }
    {
        auto write=collection.write();
        mustFail([&]{collection.upsertObject(write,fiber(alias,9));});
        auto edge=observation("bad-edge-id");edge.uuid=alias;
        mustFail([&]{collection.addEdge(write,a.active,b.active,edge);});
        write.commit();
    }
    noEndpointCopies(directory.path/"alias-existing");
    verify(collection,a,b,1);

    auto reverse=ObjectCollection::create(directory.path/"edge-first",{a.path,b.path},directOptions());
    {
        auto write=reverse.write();
        const auto futureAlias=util::nameUuid(reverse.uuid(write),"reference:"+a.active.storeUuid+":"+a.active.objectUuid);
        const auto x=reverse.upsertObject(write,fiber(util::newUuid(),1));
        const auto y=reverse.upsertObject(write,fiber(util::newUuid(),1));
        reverse.putGeometry(write,x,geometry());reverse.putGeometry(write,y,geometry());
        auto edge=observation("owned");edge.uuid=futureAlias;
        const auto original=reverse.addEdge(write,x,y,edge);
        mustFail([&]{reverse.addEdge(write,a.active,b.active,observation("external"));});
        require(reverse.edge(write,original).has_value(),"external alias collision changed existing owned edge");
        require(reverse.edgesByKind(write,"perpendicular").size()==1,"alias collision left partial relation");
        write.commit();
    }
}

void historicalWrapper()
{
    TemporaryDirectory directory;
    const auto shared = util::newUuid();
    const auto a = makeBase(directory.path / "a", shared, 1);
    const auto b = makeBase(directory.path / "b", shared, 2);
    const auto output = directory.path / "history-wrapper";
    auto options = directOptions();
    options.history = HistoryMode::On;
    auto collection = ObjectCollection::create(output, {a.path, b.path}, options);
    ObjectKey edge;
    {
        auto write = collection.write();
        edge = collection.addEdge(write, a.active, b.active, observation("historical"));
        write.commit();
    }
    noEndpointCopies(output);
    verify(collection, a, b, 1);
    {
        auto write = collection.write();
        collection.removeEdge(write, edge);
        write.commit();
    }
    noEndpointCopies(output);
    {
        auto raw = ObjectStore::open(output, OpenOptions{.physicalPartitionOnly = true, .readOnly = true});
        auto read = raw.read();
        auto* sql = ObjectStoreImplAccess::conn(read);
        require(tableRows(sql, "object_revisions") == 0, "external edge fabricated local endpoint history");
        require(tableRows(sql, "relationship_revisions") > 0, "removed external edge lost its recorded relationship history");
    }
    auto reopened = ObjectCollection::open(output, OpenOptions{.readOnly = true});
    verify(reopened, a, b, 0);
}

void siblingReferences()
{
    TemporaryDirectory directory;
    const std::array ids{util::newUuid(), util::newUuid()};
    for (const bool packed : {false, true}) {
        InitialImportOptions options;
        options.store = directOptions();
        options.shards = {InitialShardSpec{{1,100},{1,100}}, InitialShardSpec{{101,200},{101,200}}};
        if (packed) {
            PackedStoreOptions settings;
            settings.memoryFirst = true;
            settings.disposableBulk = true;
            settings.backgroundFlush = false;
            settings.residentBytes = 64 * 1024 * 1024;
            options.packed = settings;
        }
        std::atomic_size_t objectCallbacks{0};
        const auto output = directory.path / (packed ? "siblings-packed" : "siblings-plain");
        const auto result = importInitialCollection(output, options,
            [&](size_t index, InitialShardImport& shard) {
                const std::array rows{InitialObjectRecord{fiber(ids[index], int(index)), {}, geometry()}};
                shard.appendObjects(rows);
                ++objectCallbacks;
            },
            [&](size_t index, InitialShardImport& shard) {
                require(objectCallbacks.load() == 2, "edge producer ran before all object producers finished");
                bool produced = false;
                shard.importEdges([&](auto& batch) {
                    if (produced) return false;
                    produced = true;
                    batch.push_back({{shard.storeUuid(), ids[index]}, {shard.storeUuid(), ids[1-index]},
                                     observation("sibling-" + std::to_string(index))});
                    return true;
                });
            });
        auto collection = ObjectCollection::open(result.path, OpenOptions{.readOnly = true});
        auto read = collection.read();
        require(collection.edgesByKind(read, "perpendicular").size() == 2, "cross-shard edges missing after publication");
        for (const auto& id : ids) {
            const ObjectKey key{result.storeUuid, id};
            require(collection.edgesOf(read, key).size() == 2, "sibling adjacency missing");
            require(collection.readGeometryArray(read, key, "line_points").size() == 6*sizeof(double),
                    "sibling geometry does not route to its owner");
        }
        size_t partitions = 0;
        for (const auto& member : collection.members()) {
            if (!member.partition) continue;
            ++partitions;
            auto raw = ObjectStore::open(member.locator, OpenOptions{.physicalPartitionOnly = true, .readOnly = true});
            auto nativeRead = raw.read();
            auto* sql = ObjectStoreImplAccess::conn(nativeRead);
            require(tableRows(sql, "objects") == 1 && tableRows(sql, "fibers") == 1 &&
                    tableRows(sql, "current_geometry") == 1 && tableRows(sql, "collection_refs") == 0,
                    "sibling relation materialized a foreign endpoint");
        }
        require(partitions == 2, "published sibling shard count differs");
    }

    InitialImportOptions failed;
    failed.store = directOptions();
    failed.shards = {InitialShardSpec{{1,100},{1,100}}, InitialShardSpec{{101,200},{101,200}}};
    for (const bool failObjectPhase : {false, true}) {
        const auto output = directory.path / (failObjectPhase ? "object-phase-failure" : "missing-sibling");
        mustFail([&] {
            importInitialCollection(output, failed,
                [&](size_t index, InitialShardImport& shard) {
                    if (failObjectPhase && index == 0) throw std::runtime_error("source failed before barrier");
                    const std::array rows{InitialObjectRecord{fiber(ids[index], int(index)), {}, geometry()}};
                    shard.appendObjects(rows);
                },
                [&](size_t index, InitialShardImport& shard) {
                    if (index) return;
                    bool produced = false;
                    shard.importEdges([&](auto& batch) {
                        if (produced) return false;
                        produced = true;
                        batch.push_back({{shard.storeUuid(),ids[0]}, {shard.storeUuid(),util::newUuid()}, observation("missing")});
                        return true;
                    });
                });
        });
        require(!std::filesystem::exists(output), "failed sibling import published partial data");
        const auto prefix = "." + output.filename().string() + "-";
        for (const auto& entry : std::filesystem::directory_iterator(directory.path))
            require(!entry.path().filename().string().starts_with(prefix), "failed sibling import leaked staging");
    }
}
}

int main()
{
    try {
        externalReferences();
        failedPublication();
        aliasCollisions();
        historicalWrapper();
        siblingReferences();
        std::cout << "reference-only endpoint tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
