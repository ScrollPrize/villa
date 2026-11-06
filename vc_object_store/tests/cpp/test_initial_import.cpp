#include "TestSupport.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <atomic>
#include <fstream>
#include <future>
#include <iostream>

using namespace vc::objects;
namespace {
InitialImportOptions options(size_t shards=1,bool packed=false)
{
    InitialImportOptions result;result.store.author="bulk-api-test";result.shards.clear();
    for(size_t i=0;i<shards;++i)result.shards.push_back({{int64_t(i*100+1),int64_t((i+1)*100)},
        {int64_t(i*100+1),int64_t((i+1)*100)}});
    if(packed){result.packed.emplace();result.packed->memoryFirst=true;result.packed->disposableBulk=true;
        result.packed->backgroundFlush=false;result.packed->residentBytes=64*1024*1024;}
    return result;
}
InitialObjectRecord fiber(const std::string& uuid)
{
    InitialObjectRecord row;row.record.uuid=uuid;row.record.kind="fiber";row.record.fiber.emplace();
    row.record.fiber->family="H";row.record.meta=R"({"native":{"value":42}})";
    vc::geometry::Fiber geometry;geometry.linePoints={{1,2,3},{4,5,6}};row.geometry=vc::geometry::encode(geometry);
    return row;
}
void roundTrip(size_t shards,bool packed)
{
    TemporaryDirectory dir;auto opt=options(shards,packed);
    const auto base=dir.path/"base.vcobj";ObjectKey reference;int64_t baseGeneration=0;
    {
        auto store=ObjectStore::create(base,opt.store);auto w=store.write();auto row=fiber(util::newUuid());
        reference={store.uuid(w),store.upsertObject(w,row.record)};store.putGeometry(w,reference.objectUuid,*row.geometry);
        store.seal(w);w.commit();auto r=store.read();baseGeneration=store.provisioning(r).generation;
    }
    opt.dependencies={base};std::vector<std::string> uuids(shards),edges(shards);
    for(size_t i=0;i<shards;++i){uuids[i]=util::newUuid();edges[i]=util::newUuid();}
    const auto output=dir.path/"output";
    auto result=importInitialCollection(output,opt,[&](size_t i,InitialShardImport& part){
        auto row=fiber(uuids[i]);row.id=opt.shards[i].objectIds[0];
        auto ids=part.appendObjects(std::span(&row,1));require(ids.size()==1,"identity result");
        require(ids[0].key==ObjectKey{part.storeUuid(),uuids[i]},"qualified native identity");
        require(ids[0].id==*row.id,"explicit numeric ID preserved");
        bool sent=false;require(part.importEdges([&](auto& batch){
            if(sent)return false;sent=true;EdgeRecord e;e.uuid=edges[i];e.kind="perpendicular";e.relationKey="one";
            e.aAnchor.wholeGeometry=e.bAnchor.wholeGeometry=true;e.aAnchor.position=std::array<double,3>{1,2,3};
            e.meta=R"({"source":{"measurement":1.25}})";batch.push_back({ids[0].key,reference,e});return true;
        })==1,"native edge count");
    });
    require(result.path==(shards>1&&!packed?output/"collection.vcobj":output),"public root locator");
    require(result.shards.size()==shards,"all shards reported");
    require(result.totalSeconds>=result.ingestionSeconds && result.persistenceSeconds>=0,"timings reported");
    auto store=ObjectCollection::open(result.path,OpenOptions{.readOnly=true});auto r=store.read();
    require(store.uuid(r)==result.storeUuid,"published logical owner");
    require(store.provisioning(r).sealed && store.provisioning(r).generation>baseGeneration,
        "sealed generation follows dependencies");
    require(store.provisioning(r).history==HistoryMode::Off && !store.provisioning(r).spatialIndex,
        "initial provisioning omits history and spatial structures");
    for(size_t i=0;i<shards;++i){
        require(result.shards[i].objects==1 && result.shards[i].edges==1,"reported row counts");
        const ObjectKey key{result.storeUuid,uuids[i]};auto obj=store.object(r,key);
        require(obj && obj->record.fiber->family=="H","typed fiber round trip");
        require(store.readGeometryArray(r,key,"line_points").size()==6*sizeof(double),"local geometry readable");
        require(store.readGeometryArray(r,reference,"line_points").size()==6*sizeof(double),"external geometry readable after publication");
        const auto edge=store.edge(r,{result.storeUuid,edges[i]});
        require(edge && (edge->a==reference || edge->b==reference),"dependency endpoint routed");
        require(edge->record.relationKey=="one","independent relation identity retained");
    }
    // A sealed output cannot be reopened writable, independently of callback lifetime.
    mustFail([&]{auto mutableStore=ObjectCollection::open(result.path,opt.store);auto w=mutableStore.write();mutableStore.seal(w);});
}
void failures()
{
    TemporaryDirectory dir;auto opt=options();
    auto failed=[&](const auto& callback){
        const auto output=dir.path/"failure";
        mustFail([&]{importInitialCollection(output,opt,callback);});
        require(!std::filesystem::exists(output),"failed import is unpublished");
        // Readonly SQLite opens can create WAL/SHM sidecars beside a dependency;
        // those are not unpublished output owned by the builder.
        for(const auto& entry:std::filesystem::directory_iterator(dir.path)){
            const auto name=entry.path().filename().string();
            require(!name.starts_with(".failure-")&&!name.starts_with("failure.building-"),"failed staging cleaned");
        }
    };
    failed([](size_t,auto&){throw std::runtime_error("source error");});
    failed([](size_t,auto& part){auto row=fiber(util::newUuid());row.id=101;
        try{part.appendObjects(std::span(&row,1));}catch(const std::exception&){};});
    failed([](size_t,auto& part){part.importEdges([](auto&){return false;});
        try{part.appendObjects({});}catch(const std::exception&){};});
    failed([](size_t,auto& part){part.importEdges([](auto&){return false;});
        try{part.importEdges([](auto&){return false;});}catch(const std::exception&){};});
    failed([](size_t,auto& part){try{part.importEdges([](auto&)->bool{throw std::runtime_error("edge reader failed");});}
        catch(const std::exception&){};});
    failed([](size_t,auto& part){auto job=std::async(std::launch::async,[&]{mustFail([&]{part.storeUuid();});});job.get();});
    opt=options(2);failed([](size_t i,auto& part){if(i==0)throw std::runtime_error("peer failure");
        auto row=fiber(util::newUuid());part.appendObjects(std::span(&row,1));});
    opt=options();opt.store.creationStoreUuid=util::newUuid();failed([](size_t,auto&){});
    opt=options(2);opt.shards[1]=opt.shards[0];failed([](size_t,auto&){});
    opt=options();opt.store.history=HistoryMode::On;failed([](size_t,auto&){});
    opt=options();opt.store.spatialIndex=true;failed([](size_t,auto&){});
    opt=options(2,true);failed([](size_t i,auto&){if(i==0)throw std::runtime_error("packed source error");});
    opt=options();const auto mutableBase=dir.path/"mutable-base";
    {auto base=ObjectStore::create(mutableBase,opt.store);}
    opt.dependencies={mutableBase};failed([](size_t,auto&){});
    opt=options();const auto existing=dir.path/"existing";{std::ofstream file(existing);file<<"keep";}
    mustFail([&]{importInitialCollection(existing,opt,[](size_t,auto&){});});
    require(std::filesystem::file_size(existing)==4,"existing destination preserved");
    // Destination appearing during construction is never overwritten.
    const auto racing=dir.path/"racing";
    mustFail([&]{importInitialCollection(racing,opt,[&](size_t,auto&){std::ofstream file(racing);file<<"winner";});});
    require(std::filesystem::file_size(racing)==6,"publication does not clobber concurrent creator");
    opt=options(2);const auto racingDir=dir.path/"racing-dir";
    mustFail([&]{importInitialCollection(racingDir,opt,[&](size_t i,auto&){if(i==0)std::filesystem::create_directory(racingDir);});});
    require(std::filesystem::is_empty(racingDir),"concurrent directory preserved");
}
void phased(bool packed)
{
    TemporaryDirectory dir;auto opt=options(3,packed);
    std::vector<std::string> ids(3),edges(3);for(auto& id:ids)id=util::newUuid();for(auto& id:edges)id=util::newUuid();
    std::atomic<size_t> objectsFinished=0;
    const auto result=importInitialCollection(dir.path/"phased",opt,[&](size_t i,auto& shard){
        auto row=fiber(ids[i]);row.id=opt.shards[i].objectIds[0];shard.appendObjects(std::span(&row,1));
        ++objectsFinished;
    },[&](size_t i,auto& shard){
        require(objectsFinished==3,"all object producers precede any edge producer");
        bool sent=false;shard.importEdges([&](auto& batch){if(sent)return false;sent=true;
            EdgeRecord edge;edge.uuid=edges[i];edge.kind="sibling";edge.aAnchor.points={{"line_points",{1}}};
            edge.bAnchor.points={{"line_points",{0}}};edge.aAnchor.metadataKeys={"native"};
            batch.push_back({{shard.storeUuid(),ids[i]},{shard.storeUuid(),ids[(i+1)%3]},edge});return true;});
    });
    auto store=ObjectCollection::open(result.path,OpenOptions{.readOnly=true});auto txn=store.read();
    for(size_t i=0;i<3;++i) {
        const auto edge=store.edge(txn,{result.storeUuid,edges[i]});require(bool(edge),"phased sibling edge published");
        const ObjectKey a{result.storeUuid,ids[i]},b{result.storeUuid,ids[(i+1)%3]};
        require((edge->a==a&&edge->b==b)||(edge->a==b&&edge->b==a),"phased endpoint identities retained");
        require(store.readGeometryArray(txn,b,"line_points").size()==6*sizeof(double),"phased sibling geometry routes after publication");
    }
}
void phaseFailures()
{
    TemporaryDirectory dir;auto opt=options(3);
    auto fail=[&](const auto& objects,const auto& edges){
        mustFail([&]{importInitialCollection(dir.path/"failed",opt,objects,edges);});
        require(!std::filesystem::exists(dir.path/"failed"),"failed phase never publishes");
        require(std::filesystem::is_empty(dir.path),"failed phase cleans private shards");
    };
    fail([](size_t i,auto&){if(i==0)throw std::runtime_error("object producer failed");},[](size_t,auto&){});
    fail([](size_t,auto&){},[](size_t i,auto&){if(i==0)throw std::runtime_error("edge producer failed");});
    fail([](size_t,auto& shard){shard.importEdges([](auto&){return false;});},[](size_t,auto&){});
    fail([](size_t,auto&){},[](size_t,auto& shard){shard.appendObjects({});});
    const auto duplicate=util::newUuid();
    fail([&](size_t,auto& shard){auto row=fiber(duplicate);shard.appendObjects(std::span(&row,1));},[](size_t,auto&){});
    // Cross-shard geometry validation still rejects an out-of-range anchor.
    std::vector<std::string> ids(3);for(auto& id:ids)id=util::newUuid();
    fail([&](size_t i,auto& shard){auto row=fiber(ids[i]);shard.appendObjects(std::span(&row,1));},[&](size_t i,auto& shard){
        bool sent=false;shard.importEdges([&](auto& batch){if(sent)return false;sent=true;
            EdgeRecord edge;edge.kind="bad";edge.bAnchor.points={{"line_points",{999}}};
            batch.push_back({{shard.storeUuid(),ids[i]},{shard.storeUuid(),ids[(i+1)%3]},edge});return true;});
    });
    // Uniqueness must reject even cross-batch duplicates before any
    // ordinary or packed shard becomes visible, including when the producer
    // catches the import exception itself.
    for(bool packed:{false,true}){
        opt=options(3,packed);
        fail([&](size_t i,auto& shard){auto row=fiber(ids[i]);shard.appendObjects(std::span(&row,1));},
            [&](size_t i,auto& shard){
                int batchNumber=0;
                try{shard.importEdges([&](auto& batch){
                    if(batchNumber++==2)return false;
                    EdgeRecord edge;edge.uuid=util::newUuid();edge.kind="duplicate";
                    edge.aAnchor.wholeGeometry=edge.bAnchor.wholeGeometry=true;
                    batch.push_back({{shard.storeUuid(),ids[i]},{shard.storeUuid(),ids[(i+1)%3]},edge});
                    return true;
                });}catch(const std::exception&){}
            });
    }
}
}
int main(){try{for(bool packed:{false,true}){for(size_t count:{1u,2u})roundTrip(count,packed);phased(packed);}failures();phaseFailures();
    std::cout<<"initial import passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
