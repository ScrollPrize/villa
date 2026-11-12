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

EdgeRecord observation(const std::string&)
{
    EdgeRecord edge;
    edge.kind = "perpendicular";
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
    for (const auto* table : {"objects", "fibers", "geometry_chunks"}) {
        const auto message = "reference-only edge copied endpoint state into " + std::string(table);
        require(tableRows(sql, table) == 0, message.c_str());
    }
    Statement redundant(sql,"SELECT count(*) FROM sqlite_schema WHERE name IN "
        "('external_object_keys','endpoint_keys','collection_refs','collection_edge_tombstones',"
        "'collection_external_chunks','current_geometry','current_geometry_chunks','current_shared_payloads')");
    require(redundant.row()&&redundant.i64(0)==0,"relations must not provision alias, proxy or chunk registration structures");
    const auto edges=db.edgesByKind(read,"perpendicular",true);
    if(!edges.empty()){
        const auto& first=edges.front();
        require(!db.neighbours(read,first.aUuid,{}, {},true).empty(),"physical neighbour traversal dropped an external endpoint");
        require(db.componentOf(read,first.aUuid,{},nullptr,true).size()==2,"physical component traversal lost direct endpoint keys");
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
            collection.edgesOf(read, b.active).size() == count, "direct endpoint adjacency mismatch");
    require(collection.edgesOf(read, a.retired).empty(), "unrelated retired endpoint gained edges");
    for (const auto& edge : edges) {
        require((edge.a == a.active && edge.b == b.active) ||
                (edge.a == b.active && edge.b == a.active), "relation changed original endpoint UUIDs");
        require(edge.record.aUuid==edge.a.objectUuid&&edge.record.bUuid==edge.b.objectUuid,
                "native edge must contain original keys, not aliases");
        const auto direct = collection.edge(read, edge.key);
        require(direct && direct->a == edge.a && direct->b == edge.b, "edge key lookup lost endpoint routing");
        require(Json::parse(edge.record.meta).at("distance") == 1.25, "edge evidence changed");
    }
    require(collection.object(read, a.active)->key == a.active &&
            collection.object(read, b.active)->key == b.active, "external object reads do not preserve logical UUID");
    require(collection.readGeometryArray(read, a.active, "line_points") == a.bytes &&
            collection.readGeometryArray(read, b.active, "line_points") == b.bytes, "geometry lookup failed through references");
    size_t objectCount = 0;
    collection.scanObjects(read, [&](const auto&) { ++objectCount; return true; }, "fiber");
    require(objectCount == 2, "endpoint references surfaced as duplicate active objects");
}

void externalReferences()
{
    TemporaryDirectory directory;
    const auto a = makeBase(directory.path / "a", util::newUuid(), 1);
    const auto b = makeBase(directory.path / "b", util::newUuid(), 2);
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
            require(Json::parse(collection.object(read, b.active)->record.meta).at("quality") == 2, "unrelated endpoint changed");
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
        InitialImportOptions options;options.uuidHashRouting=false;
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
                    if(batchNumber==1)throw std::runtime_error("source failure after first inserted batch");
                    ++batchNumber;const auto other=base.active;
                    auto edge=observation("source-batch-"+std::to_string(batchNumber));
                    edge.bAnchor.points={{"line_points",{0}}};
                    batch.push_back({base.active,other,std::move(edge)});
                    return true;
                });
            });
        });
        require(!std::filesystem::exists(output), "producer failure published an incomplete collection");
        for (const auto& entry : std::filesystem::directory_iterator(directory.path)) {
            const auto filename = entry.path().filename().string();
            require(!filename.starts_with("." + name + "-") && !filename.starts_with(name + ".building-"),
                    "failed collection left private construction files");
        }
    }
}

void uuidPrecedence()
{
    TemporaryDirectory directory;
    const auto shared=util::newUuid();
    const auto a=makeBase(directory.path/"a",shared,1);
    const auto b=makeBase(directory.path/"b",shared,2);
    const auto path=directory.path/"overlay";
    auto collection=ObjectCollection::create(path,{a.path,b.path},directOptions());
    {
        auto read=collection.read();
        require(a.active==b.active,"store hints must not split one logical object UUID");
        require(Json::parse(collection.object(read,a.active)->record.meta).at("quality")==1,
                "first referenced store has explicit precedence");
        require(Json::parse(collection.object(read,b.active)->record.meta).at("quality")==1,
                "store hint must not bypass overlay precedence");
        size_t count=0;
        collection.scanObjects(read,[&](const auto&){++count;return true;},"fiber");
        require(count==1,"same UUID must enumerate once across referenced stores");
    }
    auto reverse=ObjectCollection::create(directory.path/"reversed",{b.path,a.path},directOptions());
    {
        auto read=reverse.read();
        require(Json::parse(reverse.object(read,a.active)->record.meta).at("quality")==2,
                "store_reference ordering must define deterministic precedence");
    }
    {
        auto write=collection.write();
        collection.setMeta(write,b.active,"quality","9");
        write.commit();
    }
    {
        auto read=collection.read();
        for(const auto& key:{a.active,b.active})
            require(collection.object(read,key)->record.uuid==shared&&
                    Json::parse(collection.object(read,key)->record.meta).at("quality")==9,
                    "replacement retains UUID and shadows every occurrence");
        require(collection.readGeometryArray(read,a.active,"line_points")==a.bytes,
                "metadata override preserves inherited geometry");
    }
    {
        auto raw=ObjectStore::open(path,OpenOptions{.physicalPartitionOnly=true,.readOnly=true});
        auto read=raw.read();auto* sql=ObjectStoreImplAccess::conn(read);
        Statement record(sql,"SELECT uuid,state FROM objects");
        require(record.row()&&formatUuid(record.blob16(0))==shared&&record.i64(1)==0&&!record.row(),
                "overlay stores one same-key replacement, not an alias record");
        require(tableRows(sql,"geometry_chunks")==0,"metadata replacement copied geometry");
    }
    {
        auto write=collection.write();collection.deleteObject(write,a.active);collection.seal(write);write.commit();
        auto read=collection.read();
        require(!collection.object(read,a.active)&&!collection.object(read,b.active),
                "deleted key must stop fallback to every referenced store");
    }
    {
        auto raw=ObjectStore::open(path,OpenOptions{.physicalPartitionOnly=true,.readOnly=true});
        auto read=raw.read();auto* sql=ObjectStoreImplAccess::conn(read);
        Statement record(sql,"SELECT uuid,state FROM objects");
        require(record.row()&&formatUuid(record.blob16(0))==shared&&record.i64(1)==2&&!record.row(),
                "deletion belongs to the same-key object row");
        Statement removed(sql,"SELECT count(*) FROM sqlite_schema WHERE name IN "
            "('external_object_keys','endpoint_keys','collection_refs','collection_edge_tombstones')");
        require(removed.row()&&removed.i64(0)==0,"overlay provisioned redundant registration tables");
    }
    auto child=ObjectCollection::create(directory.path/"child",{path,b.path},directOptions());
    {
        auto read=child.read();
        require(!child.object(read,a.active),"deeper store_reference resurrected a shadowed deleted object");
    }
    for(const auto& base:{a,b}) {
        auto source=ObjectStore::open(base.path,OpenOptions{.readOnly=true});auto read=source.read();
        require(source.objectByUuid(read,shared).has_value(),"overlay deletion changed readonly source");
    }
}

void formerAliasNamesRemainOrdinaryKeys()
{
    TemporaryDirectory directory;
    const auto a=makeBase(directory.path/"a",util::newUuid(),1);
    const auto b=makeBase(directory.path/"b",util::newUuid(),2);
    auto collection=ObjectCollection::create(directory.path/"overlay",{a.path,b.path},directOptions());
    auto write=collection.write();
    const auto formerAlias=util::nameUuid(collection.uuid(write),"reference:"+a.active.storeUuid+":"+a.active.objectUuid);
    const auto owned=collection.upsertObject(write,fiber(formerAlias,9));
    const auto edge=collection.addEdge(write,a.active,b.active,observation("external"));
    require(collection.object(write,owned)->record.uuid==formerAlias,"alias-free references reserved an unrelated key");
    require(collection.edge(write,edge).has_value(),"ordinary owned key blocked a direct external endpoint");
    auto duplicate=observation("duplicate");duplicate.uuid=formerAlias;
    mustFail([&]{collection.addEdge(write,a.active,b.active,duplicate);});
    write.commit();
}

void historicalWrapper()
{
    TemporaryDirectory directory;
    const auto a = makeBase(directory.path / "a", util::newUuid(), 1);
    const auto b = makeBase(directory.path / "b", util::newUuid(), 2);
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

void routedPartitionsPrecedeTheirBases()
{
    TemporaryDirectory directory;
    const auto inSecondShard=[] {
        auto uuid=util::newUuid();while(uuidShardIndex(uuid,2)!=1)uuid=util::newUuid();return uuid;
    };
    const auto a=inSecondShard(),b=util::newUuid(),edgeUuid=inSecondShard();
    const auto basePath=directory.path/"base";
    {
        auto base=ObjectStore::create(basePath,directOptions());auto write=base.write();
        base.upsertObject(write,fiber(a,1));base.upsertObject(write,fiber(b,2));
        EdgeRecord e;e.uuid=edgeUuid;e.aUuid=a;e.bUuid=b;e.kind="declared_reference";e.meta=R"({"quality":1})";
        e.aAnchor.wholeGeometry=e.bAnchor.wholeGeometry=true;
        base.addEdge(write,e);base.seal(write);write.commit();
    }
    InitialImportOptions options;options.store=directOptions();options.dependencies={basePath};
    options.shards={InitialShardSpec{{1,100}},InitialShardSpec{{101,200}}};
    const auto result=importInitialCollection(directory.path/"routed",options,
        [&](size_t i,InitialShardImport& shard){
            if(i!=1)return;
            const std::array rows{InitialObjectRecord{fiber(a,9),{},geometry()}};shard.appendObjects(rows);
            bool supplied=false;shard.importEdges([&](auto& batch){
                if(supplied)return false;supplied=true;
                EdgeRecord e;e.uuid=edgeUuid;e.kind="declared_reference";e.meta=R"({"quality":9})";
                e.aAnchor.wholeGeometry=e.bAnchor.wholeGeometry=true;
                batch.push_back({{shard.storeUuid(),a},{{},b},e});
                return true;
            });
        });
    {
        auto collection=ObjectCollection::open(result.path,OpenOptions{.readOnly=true});auto read=collection.read();
        require(Json::parse(collection.object(read,{{},a})->record.meta).at("quality")==9,
            "selected later shard replacement must precede an earlier shard's shared base");
        require(Json::parse(collection.edge(read,{{},edgeUuid})->record.meta).at("quality")==9,
            "selected later shard edge must precede inherited edge with same UUID");
        require(collection.object(read,{{},b}).has_value(),"unselected shard dependencies remain visible");
        size_t objectCount=0;
        collection.scanObjects(read,[&](const CollectionObject& row){
            ++objectCount;
            if(row.key.objectUuid==a)require(Json::parse(row.record.meta).at("quality")==9,
                "object scans resolve later-shard replacements like keyed reads");
            return true;
        });
        require(objectCount==2,"object scans deduplicate base and partition records");
        const auto byKind=collection.edgesByKind(read,"declared_reference");
        const auto adjacent=collection.edgesOf(read,{{},a});
        require(byKind.size()==1&&adjacent.size()==1&&
            Json::parse(byKind.front().record.meta).at("quality")==9&&
            Json::parse(adjacent.front().record.meta).at("quality")==9,
            "edge kind and adjacency scans resolve later-shard replacements like keyed reads");
    }
    auto overlay=ObjectCollection::create(directory.path/"overlay",{result.path},directOptions());
    {auto write=overlay.write();overlay.setMeta(write,{{},a},"quality","11");write.commit();}
    {auto read=overlay.read();require(Json::parse(overlay.object(read,{{},a})->record.meta).at("quality")==11,
        "local overlay replacement precedes UUID-routed shards");}
    {auto write=overlay.write();overlay.removeEdge(write,{{},edgeUuid});overlay.deleteObject(write,{{},a});overlay.seal(write);write.commit();}
    {auto read=overlay.read();require(!overlay.object(read,{{},a})&&!overlay.edge(read,{{},edgeUuid}),
        "local tombstones stop fallback to routed partitions and their bases");}
    auto child=ObjectCollection::create(directory.path/"child",{directory.path/"overlay",basePath},directOptions());
    {auto read=child.read();require(!child.object(read,{{},a})&&!child.edge(read,{{},edgeUuid}),
        "routed-layer tombstones stay authoritative through a deeper store_reference overlay");
        size_t count=0;child.scanObjects(read,[&](const auto& row){
            require(row.key.objectUuid==b,"object scans must not resurrect a shadowed tombstone");++count;return true;
        });
        require(count==1&&child.edgesByKind(read,"declared_reference").empty()&&child.edgesOf(read,{{},b}).empty(),
            "full object and edge scans honor nested overlay tombstones");}
}

void siblingReferences()
{
    TemporaryDirectory directory;
    const std::array ids{util::newUuid(), util::newUuid()};
    for (const bool packed : {false, true}) {
        InitialImportOptions options;options.uuidHashRouting=false;
        options.store = directOptions();
        options.shards = {InitialShardSpec{{1,100}}, InitialShardSpec{{101,200}}};
        if (packed) {
            PackedStoreOptions settings;
            settings.memoryFirst = true;
            settings.disposableBulk = true;
            settings.backgroundFlush = false;
            settings.residentBytes = 64 * 1024 * 1024;
            options.packed = settings;
        }
        const auto output = directory.path / (packed ? "siblings-packed" : "siblings-plain");
        const auto result = importInitialCollection(output, options,
            [&](size_t index, InitialShardImport& shard) {
                const std::array rows{InitialObjectRecord{fiber(ids[index], int(index)), {}, geometry()}};
                shard.appendObjects(rows);
            },
            [&](size_t index, InitialShardImport& shard) {
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
            require(tableRows(sql, "objects") == 1 && tableRows(sql, "fibers") == 1,
                    "sibling relation materialized a foreign endpoint");
            Statement registrations(sql,"SELECT count(*) FROM sqlite_schema WHERE name IN "
                "('external_object_keys','endpoint_keys','collection_refs','collection_edge_tombstones')");
            require(registrations.row()&&registrations.i64(0)==0,"sibling references require no local registration structures");
            Statement endpoints(sql,"SELECT a_uuid,b_uuid FROM edges WHERE state=0");
            while(endpoints.row()) {
                const auto endpointA=formatUuid(endpoints.blob16(0)),endpointB=formatUuid(endpoints.blob16(1));
                require((endpointA==ids[0]&&endpointB==ids[1])||(endpointA==ids[1]&&endpointB==ids[0]),
                    "sharding must not translate endpoint UUIDs into local aliases");
            }
        }
        require(partitions == 2, "published sibling shard count differs");
    }

    InitialImportOptions failed;failed.uuidHashRouting=false;
    failed.store = directOptions();
    failed.shards = {InitialShardSpec{{1,100}}, InitialShardSpec{{101,200}}};
    for (const bool failObjectPhase : {false, true}) {
        const auto output = directory.path / (failObjectPhase ? "object-phase-failure" : "malformed-edge");
        mustFail([&] {
            importInitialCollection(output, failed,
                [&](size_t index, InitialShardImport& shard) {
                    if (failObjectPhase && index == 0) throw std::runtime_error("object source failed");
                    const std::array rows{InitialObjectRecord{fiber(ids[index], int(index)), {}, geometry()}};
                    shard.appendObjects(rows);
                },
                [&](size_t index, InitialShardImport& shard) {
                    if (index) return;
                    bool produced = false;
                    shard.importEdges([&](auto& batch) {
                        if (produced) return false;
                        produced = true;
                        auto invalid=observation("malformed");invalid.uuid="invalid-uuid";
                        batch.push_back({{shard.storeUuid(),ids[0]}, {shard.storeUuid(),ids[1]}, invalid});
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
        uuidPrecedence();
        formerAliasNamesRemainOrdinaryKeys();
        historicalWrapper();
        routedPartitionsPrecedeTheirBases();
        siblingReferences();
        std::cout << "reference-only endpoint tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
