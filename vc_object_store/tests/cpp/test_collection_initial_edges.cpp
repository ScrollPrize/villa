#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/RecordSql.hpp"
#include "internal/Anchors.hpp"
#include <nlohmann/json.hpp>

using namespace vc::objects;
using Json=nlohmann::json;

int main()
{
    TemporaryDirectory dir;
    OpenOptions options;options.author="initial-test";options.history=HistoryMode::Off;options.spatialIndex=false;
    const auto basePath=dir.path/"base.vcobj";
    ObjectKey a,b,c,d;
    {
        auto base=ObjectStore::create(basePath,options);auto w=base.write();
        ObjectRecord fiber;fiber.kind="fiber";fiber.fiber.emplace();fiber.meta=R"({"label":"A"})";
        a={base.uuid(w),base.upsertObject(w,fiber)};
        fiber.uuid.clear();fiber.meta=R"({"label":"B"})";b={a.storeUuid,base.upsertObject(w,fiber)};
        fiber.uuid.clear();c={a.storeUuid,base.upsertObject(w,fiber)};
        fiber.uuid.clear();d={a.storeUuid,base.upsertObject(w,fiber)};
        vc::geometry::Fiber shape;shape.linePoints={{0,1,2},{3,4,5},{6,7,8}};
        base.putGeometry(w,a.objectUuid,vc::geometry::encode(shape));
        base.retireObject(w,c.objectUuid);
        base.seal(w);w.commit();
    }
    std::vector<InitialCollectionEdge> inputs;
    for(int i=0;i<7;++i){
        EdgeRecord edge;edge.uuid=util::nameUuid("initial-edge-"+std::to_string(i));edge.kind="perpendicular";
        edge.meta=Json({{"measurement",i}}).dump();
        edge.aAnchor.wholeGeometry=edge.bAnchor.wholeGeometry=true;
        edge.aAnchor.metadataKeys={"label"};edge.bAnchor.metadataKeys={"label","missing"};
        edge.aAnchor.position=std::array<double,3>{double(i),2,3};edge.bAnchor.position=std::array<double,3>{4,5,double(i)};
        edge.tags={"b","a","b",""};inputs.push_back({i%2?a:b,i%2?b:a,edge});
    }
    auto populate=[&](ObjectCollection& store,CollectionTxn& txn,bool defer){
        size_t next=0;
        const auto count=store.importInitialEdges(txn,[&](auto& batch){
            for(int n=0;n<2 && next<inputs.size();++n)batch.push_back(inputs[next++]);
            return !batch.empty();
        },{defer,3});
        require(count==7,"all streamed relation batches imported");
    };
    std::vector<CollectionEdge> expected;
    {
        auto scalar=ObjectCollection::create(dir.path/"scalar.vcobj",{basePath},options);auto w=scalar.write();
        for(const auto& e:inputs)scalar.addEdge(w,e.a,e.b,e.record);
        w.commit();auto r=scalar.read();expected=scalar.edgesByKind(r,"perpendicular");
    }
    auto verify=[&](ObjectCollection& store){
        auto r=store.read();const auto actual=store.edgesByKind(r,"perpendicular");
        require(actual.size()==expected.size(),"bulk edge count equals scalar");
        for(const auto& e:expected){
            const auto result=store.edge(r,{store.uuid(r),e.record.uuid});
            require(result && result->a==e.a && result->b==e.b && result->record.aAnchor==e.record.aAnchor &&
                result->record.bAnchor==e.record.bAnchor && result->record.tags==e.record.tags &&
                Json::parse(result->record.meta)==Json::parse(e.record.meta),
                "bulk native edge equals scalar, including canonical endpoints and dependencies");
        }
        require(store.edgesOf(r,a).size()==7,"adjacency indexes available after finish");
    };
    for(bool defer:{false,true}){
        auto store=ObjectCollection::create(dir.path/(defer?"bulk-deferred.vcobj":"bulk-indexed.vcobj"),{basePath},options);
        auto w=store.write();
        int batchCount=0;
        mustFail([&]{store.importInitialEdges(w,[&](auto& batch){
            if(batchCount++==2)return false;batch.push_back(inputs.front());return true;
        },{defer,2});});
        require(store.edgesByKind(w,"perpendicular").empty(),"failed late batch rolls back all rows and restores indexes");
        mustFail([&]{store.importInitialEdges(w,[&](auto&){store.setMeta(w,a,"changed","true");return false;},{defer,2});});
        require(!Json::parse(store.object(w,a)->record.meta).contains("changed"),"producer mutation rejected and rolled back");
        mustFail([&]{store.importInitialEdges(w,[&](auto&){w.commit();return false;},{defer,2});});
        require(store.edgesByKind(w,"perpendicular").empty(),"producer cannot commit the unfinished operation");
        populate(store,w,defer);
        mustFail([&]{store.importInitialEdges(w,[](auto&){return false;},{defer,2});});
        w.commit();verify(store);
    }
    {
        // A real edit creates one ordinary same-key record; merely inserting
        // relations above did not create endpoint or alias rows.
        {
            auto collection=ObjectCollection::open(dir.path/"bulk-deferred.vcobj",options);
            auto edit=collection.write();collection.setMeta(edit,d,"unrelated","true");edit.commit();
        }
        auto rawOptions=options;rawOptions.physicalPartitionOnly=true;
        auto store=ObjectStore::open(dir.path/"bulk-deferred.vcobj",rawOptions);
        auto read=store.read();auto* db=ObjectStoreImplAccess::conn(read);
        detail::Statement record(db,"SELECT uuid,state FROM objects");
        require(record.row()&&detail::formatUuid(record.blob16(0))==d.objectUuid&&
            record.i64(1)==0&&!record.row(),"actual override must retain its original UUID");
        detail::Statement redundant(db,"SELECT count(*) FROM sqlite_schema WHERE name IN "
            "('external_object_keys','endpoint_keys','collection_refs','collection_edge_tombstones')");
        require(redundant.row()&&redundant.i64(0)==0,"batch import must not create endpoint registration structures");
    }
    {
        auto store=ObjectCollection::create(dir.path/"late-endpoint.vcobj",{basePath},options);
        auto w=store.write();size_t next=0;
        auto indexed=inputs.front();indexed.a=a;indexed.b=b;
        indexed.record.aAnchor.wholeGeometry=false;
        indexed.record.aAnchor.points.push_back({"line_points",{1}});
        auto rejected=inputs.back();rejected.a=a;rejected.b=c;rejected.record.meta="[]";bool supplied=false;
        mustFail([&]{store.importInitialEdges(w,[&](auto& batch){
            if(supplied)return false;supplied=true;batch.push_back(rejected);return true;
        },{true,1});});
        require(store.edgesByKind(w,"perpendicular").empty(),"invalid record rejects stream without partial rows");
        auto late=inputs.back();late.a=a;late.b=c;
        late.record.aAnchor.points={{"line_points",{999}}};late.record.aAnchor.wholeGeometry=false;
        store.importInitialEdges(w,[&](auto& batch){
            if(next==2)return false;batch.push_back(next++?late:indexed);return true;
        },{true,1});
        require(store.importInitialObjects(w,{}).empty(),"trusted empty object batch remains a no-op after relations");
        w.commit();
        auto r=store.read();require(store.object(r,c)->record.retired,"readonly retired endpoint remains retired");
        require(!store.object(r,d)->record.retired,"late active endpoint remains active");
        require(store.readGeometryArray(r,a,"line_points").size()==9*sizeof(double),"indexed inherited geometry readable");
        auto raw=ObjectStore::open(dir.path/"late-endpoint.vcobj",OpenOptions{.physicalPartitionOnly=true,.readOnly=true});
        auto rt=raw.read();
        const auto actual=store.edge(r,{store.uuid(r),indexed.record.uuid});
        require(actual&&((actual->a==a&&actual->b==b)||(actual->a==b&&actual->b==a)),"initial indexed edge retains direct endpoint keys");
        detail::Statement unchecked(ObjectStoreImplAccess::conn(rt),
            "SELECT a_uuid,b_uuid,a_dependencies,b_dependencies FROM edges WHERE uuid=?",detail::parseUuid(late.record.uuid));
        require(unchecked.row(),"trusted import persists references that ordinary graph readers may hide");
        const auto first=detail::formatUuid(unchecked.blob16(0)),second=detail::formatUuid(unchecked.blob16(1));
        require((first==a.objectUuid&&second==c.objectUuid)||(first==c.objectUuid&&second==a.objectUuid),
            "trusted initial import stores supplied retired endpoint UUIDs");
        require(Json::parse(unchecked.text(first==a.objectUuid?2:3))==detail::dependenciesToJson(late.record.aAnchor),
            "indexed geometry references are persisted without fetching or validating endpoint geometry");
        require(store.currentGeometry(r,a).has_value(),"indexed geometry resolves through the collection");
        detail::Statement counts(ObjectStoreImplAccess::conn(rt),"SELECT (SELECT count(*) FROM objects),"
            "(SELECT count(*) FROM fibers),(SELECT count(*) FROM objects WHERE geometry_manifest IS NOT NULL),(SELECT count(*) FROM geometry_chunks),"
            "(SELECT count(*) FROM sqlite_schema WHERE name IN ('external_object_keys','endpoint_keys','collection_refs','collection_edge_tombstones'))");
        require(counts.row()&&counts.i64(0)==0&&counts.i64(1)==0&&counts.i64(2)==0&&counts.i64(3)==0&&counts.i64(4)==0,
            "indexed initial references must not materialize endpoint records or geometry");
    }
    const auto packedPath=dir.path/"packed.vcobj";
    PackedStoreOptions packedOptions;packedOptions.disposableBulk=true;packedOptions.memoryFirst=true;
    packedOptions.backgroundFlush=false;packedOptions.residentBytes=32*1024*1024;
    PackedStoreBuilder builder(packedPath,packedOptions);const auto database=util::newUuid();
    {
        auto store=ObjectCollection::create(builder.addDatabase(database),{basePath},options);auto w=store.write();
        populate(store,w,true);store.seal(w);w.commit();verify(store);
    }
    builder.finalize(database);
    auto published=ObjectCollection::open(packedPath,OpenOptions{.readOnly=true});verify(published);
}
