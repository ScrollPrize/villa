#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/RecordSql.hpp"
#include <nlohmann/json.hpp>

using namespace vc::objects;
using Json=nlohmann::json;

int main()
{
    TemporaryDirectory dir;
    OpenOptions options;options.author="initial-test";options.history=HistoryMode::Off;options.spatialIndex=false;
    const auto basePath=dir.path/"base.vcobj";
    ObjectKey a,b,c;
    {
        auto base=ObjectStore::create(basePath,options);auto w=base.write();
        ObjectRecord fiber;fiber.kind="fiber";fiber.fiber.emplace();fiber.meta=R"({"label":"A"})";
        a={base.uuid(w),base.upsertObject(w,fiber)};
        fiber.uuid.clear();fiber.meta=R"({"label":"B"})";b={a.storeUuid,base.upsertObject(w,fiber)};
        fiber.uuid.clear();c={a.storeUuid,base.upsertObject(w,fiber)};
        vc::geometry::Fiber shape;shape.linePoints={{0,1,2},{3,4,5},{6,7,8}};
        base.putGeometry(w,a.objectUuid,vc::geometry::encode(shape));
        base.retireObject(w,c.objectUuid);
        base.seal(w);w.commit();
    }
    std::vector<InitialCollectionEdge> inputs;
    for(int i=0;i<7;++i){
        EdgeRecord edge;edge.uuid=util::nameUuid("initial-edge-"+std::to_string(i));edge.kind="perpendicular";
        edge.relationKey="observation-"+std::to_string(i);edge.meta=Json({{"measurement",i}}).dump();
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
        },{defer,2,3});
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
                result->record.relationKey==e.record.relationKey && Json::parse(result->record.meta)==Json::parse(e.record.meta),
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
        // The direct comparison CHECK has the same SQL affinity/NULL domain
        // as IN, without constructing a temporary B-tree for every proxy row.
        auto rawOptions=options;rawOptions.physicalPartitionOnly=true;
        auto store=ObjectStore::open(dir.path/"bulk-deferred.vcobj",rawOptions);
        auto w=store.write();auto* db=ObjectStoreImplAccess::conn(w);
        detail::exec(db,"CREATE TEMP TABLE old_state(value TEXT NOT NULL CHECK(value IN ('reference','override','deleted')))");
        for(const auto* value:{"'reference'","'override'","'deleted'","NULL","0","''","'Reference'","'reference '","X'7265666572656e6365'","'reference'||char(0)"}){
            detail::Savepoint rollback(db);
            auto accepts=[&](const std::string& sql){
                const int code=sqlite3_exec(db,sql.c_str(),nullptr,nullptr,nullptr);
                require(code==SQLITE_OK||(code&255)==SQLITE_CONSTRAINT,"unexpected CHECK failure");return code==SQLITE_OK;
            };
            require(accepts(std::string("INSERT INTO old_state VALUES(")+value+")")==
                accepts(std::string("UPDATE collection_refs SET state=")+value),"reference CHECK domain parity");
        }
    }
    {
        auto store=ObjectCollection::create(dir.path/"late-endpoint.vcobj",{basePath},options);
        auto w=store.write();size_t next=0;
        auto indexed=inputs.front();indexed.a=a;indexed.b=b;
        indexed.record.aAnchor.wholeGeometry=false;
        indexed.record.aAnchor.points.push_back({"line_points",{1}});
        auto late=inputs.back();late.a=a;late.b=c;
        store.importInitialEdges(w,[&](auto& batch){
            if(next==2)return false;batch.push_back(next++?late:indexed);return true;
        },{true,1});
        mustFail([&]{store.importInitialObjects(w,{});});
        w.commit();
        auto r=store.read();require(store.object(r,c)->record.retired,"late endpoint lifecycle preserved");
        require(store.readGeometryArray(r,a,"line_points").size()==9*sizeof(double),"indexed inherited geometry readable");
        auto raw=ObjectStore::open(dir.path/"late-endpoint.vcobj",OpenOptions{.physicalPartitionOnly=true,.readOnly=true});
        auto rt=raw.read();
        detail::Statement bindings(ObjectStoreImplAccess::conn(rt),
            "SELECT count(*) FROM edge_geometry_bindings WHERE a_revision IS NOT NULL OR b_revision IS NOT NULL");
        require(bindings.row()&&bindings.i64(0)>0,"initial indexed anchors retain native bindings");
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
