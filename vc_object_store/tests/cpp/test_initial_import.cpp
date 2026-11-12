#include "TestSupport.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include <atomic>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <sqlite3.h>
#include <thread>

using namespace vc::objects;
namespace {
std::atomic<size_t> connectionCount=0;
int countConnection(sqlite3*,char**,const sqlite3_api_routines*) noexcept {
    ++connectionCount;return SQLITE_OK;
}
struct CountConnections {
    CountConnections(){connectionCount=0;require(sqlite3_auto_extension(reinterpret_cast<void(*)()>(countConnection))==SQLITE_OK,"install connection counter");}
    ~CountConnections(){sqlite3_cancel_auto_extension(reinterpret_cast<void(*)()>(countConnection));}
};
InitialImportOptions options(size_t shards=1,bool packed=false)
{
    InitialImportOptions result;result.store.author="bulk-api-test";result.uuidHashRouting=false;result.shards.clear();
    for(size_t i=0;i<shards;++i)result.shards.push_back({{int64_t(i*100+1),int64_t((i+1)*100)}});
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
    const auto base=dir.path/"base.vcobj";ObjectKey reference;int64_t baseSequence=0;
    {
        auto store=ObjectStore::create(base,opt.store);auto w=store.write();auto row=fiber(util::newUuid());
        reference={store.uuid(w),store.upsertObject(w,row.record)};store.putGeometry(w,reference.objectUuid,*row.geometry);
        store.seal(w);w.commit();auto r=store.read();baseSequence=store.provisioning(r).commitSequence;
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
            if(sent)return false;sent=true;EdgeRecord e;e.uuid=edges[i];e.kind="perpendicular";
            e.aAnchor.wholeGeometry=e.bAnchor.wholeGeometry=true;e.aAnchor.position=std::array<double,3>{1,2,3};
            e.meta=R"({"source":{"measurement":1.25}})";batch.push_back({ids[0].key,reference,e});return true;
        })==1,"native edge count");
    });
    require(result.path==(shards>1&&!packed?output/"collection.vcobj":output),"public root locator");
    require(result.shards.size()==shards,"all shards reported");
    require(result.totalSeconds>=result.ingestionSeconds && result.persistenceSeconds>=0,"timings reported");
    auto store=ObjectCollection::open(result.path,OpenOptions{.readOnly=true});auto r=store.read();
    require(store.uuid(r)==result.storeUuid,"published logical owner");
    require(store.provisioning(r).sealed && store.provisioning(r).commitSequence>baseSequence,
        "sealed commit sequence follows dependencies");
    const auto pin=store.provisioning(r);
    require(result.generation==pin.generation && result.commitSequence==pin.commitSequence,
        "import returns the sealed snapshot without a reader reopen");
    for(const auto& member:store.members())if(member.storeUuid==result.storeUuid)
        require(member.generation==pin.generation&&member.commitSequence==pin.commitSequence,
            "parallel shards share the manifest lineage and commit sequence");
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
        require(edge->record.uuid==edges[i],"independent relation identity retained");
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
    opt=options();opt.store.creationGeneration=util::newUuid();failed([](size_t,auto&){});
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
    TemporaryDirectory dir;auto opt=options(2,packed);opt.uuidHashRouting=true;
    std::vector<std::string> ids(2),edges(2);
    for(size_t i=0;i<2;++i){
        do{ids[i]=util::newUuid();}while(uuidShardIndex(ids[i],2)!=i);
        do{edges[i]=util::newUuid();}while(uuidShardIndex(edges[i],2)!=i);
    }
    std::promise<void> firstEdgesImported;auto ready=firstEdgesImported.get_future();
    std::atomic<bool> secondObjectStarted=false;
    const auto result=importInitialCollection(dir.path/"phased",opt,[&](size_t i,auto& shard){
        if(i==1){
            require(ready.wait_for(std::chrono::seconds(5))==std::future_status::ready,
                "one worker's edges must not wait for other object producers");
            secondObjectStarted=true;
        }
        auto row=fiber(ids[i]);row.id=opt.shards[i].objectIds[0];shard.appendObjects(std::span(&row,1));
    },[&](size_t i,auto& shard){
        if(i==0)require(!secondObjectStarted,"referenced sibling object must not exist yet");
        bool sent=false;shard.importEdges([&](auto& batch){if(sent)return false;sent=true;
            EdgeRecord edge;edge.uuid=edges[i];edge.kind="declared_reference";edge.aAnchor.points={{"line_points",{1}}};
            edge.bAnchor.points={{"line_points",{0}}};edge.aAnchor.metadataKeys={"native"};
            batch.push_back({{shard.storeUuid(),ids[i]},{shard.storeUuid(),ids[(i+1)%2]},edge});return true;});
        if(i==0)firstEdgesImported.set_value();
    });
    auto store=ObjectCollection::open(result.path,OpenOptions{.readOnly=true});auto txn=store.read();
    for(size_t i=0;i<2;++i) {
        const auto edge=store.edge(txn,{result.storeUuid,edges[i]});require(bool(edge),"phased sibling edge published");
        const ObjectKey a{result.storeUuid,ids[i]},b{result.storeUuid,ids[(i+1)%2]};
        require((edge->a==a&&edge->b==b)||(edge->a==b&&edge->b==a),"phased endpoint identities retained");
        require(store.readGeometryArray(txn,b,"line_points").size()==6*sizeof(double),"phased sibling geometry routes after publication");
    }
}
void directConnections(bool packed) {
    TemporaryDirectory dir;
    for(size_t count:{1u,4u}) {
        auto opt=options(count,packed);std::vector<std::string> ids(count);
        for(auto& id:ids)id=util::newUuid();
        InitialImportResult result;
        {
            CountConnections observed;
            result=importInitialCollection(dir.path/("direct-"+std::to_string(count)),opt,
                [&](size_t i,auto& shard){auto row=fiber(ids[i]);row.id=opt.shards[i].objectIds[0];shard.appendObjects(std::span(&row,1));},
                [&](size_t i,auto& shard){if(count==1)return;bool sent=false;
                    shard.importEdges([&](auto& batch){if(sent)return false;sent=true;
                        EdgeRecord edge;edge.uuid=util::newUuid();edge.kind="declared_reference";
                        edge.aAnchor.wholeGeometry=edge.bAnchor.wholeGeometry=true;
                        batch.push_back({{shard.storeUuid(),ids[i]},{shard.storeUuid(),ids[(i+1)%count]},edge});return true;});});
            require(connectionCount==count+(count>1?1:0),
                "exactly one writer per new database: no reader pools or reopened shards/collection");
        }
        // Validation is deliberately outside construction, like a consumer.
        auto collection=ObjectCollection::open(result.path,OpenOptions{.readOnly=true});auto read=collection.read();
        const auto pin=collection.provisioning(read);
        require(collection.provisioning(read).sealed,"direct output sealed");
        for(const auto& member:collection.members())if(member.storeUuid==result.storeUuid)
            require(member.generation==pin.generation&&member.commitSequence==pin.commitSequence,
                "direct manifest pins committed shard lineage and sequence");
        for(size_t i=0;i<count;++i) {
            const auto object=collection.objectById(read,result.storeUuid,opt.shards[i].objectIds[0]);
            require(object&&object->key.objectUuid==ids[i],"direct range routing");
        }
    }
}
void dependencyReaderConnections(bool packed)
{
    TemporaryDirectory dir;
    constexpr size_t shards=4;
    auto opt=options(shards,packed);
    const auto base=importInitialCollection(dir.path/"base",opt,[](size_t,auto&){});
    opt.dependencies={base.path};
    // Initial-import workers are thread-confined. An ordinary reader-pool
    // setting must not multiply connections by both input and output shards.
    opt.store.readerPoolSize=16;
    {
        CountConnections observed;
        const auto result=importInitialCollection(dir.path/"derived",opt,[](size_t,auto&){});
        require(result.shards.size()==shards,"derived import keeps all shards");
        const size_t maxConnections=shards*(2*(shards+1)+1)+1;
        require(connectionCount<=maxConnections,"batch dependencies need one reader per database per worker, not a pool");
    }
    auto collection=ObjectCollection::open(packed?dir.path/"derived":dir.path/"derived/collection.vcobj",
        OpenOptions{.readOnly=true});
    auto read=collection.read();
    require(collection.provisioning(read).sealed,"derived import remains readable and sealed");
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
    // Per-shard primary key uniqueness remains enforced by SQLite.
    fail([&](size_t,auto& shard){auto row=fiber(util::newUuid());shard.appendObjects(std::span(&row,1));
        shard.appendObjects(std::span(&row,1));},[](size_t,auto&){});
    // Malformed fields still fail without inspecting referenced geometry.
    std::vector<std::string> ids(3);for(auto& id:ids)id=util::newUuid();
    fail([&](size_t i,auto& shard){auto row=fiber(ids[i]);shard.appendObjects(std::span(&row,1));},[&](size_t i,auto& shard){
        bool sent=false;shard.importEdges([&](auto& batch){if(sent)return false;sent=true;
            EdgeRecord edge;edge.kind="bad";edge.uuid="not-a-uuid";
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
                const auto edgeUuid=util::newUuid();
                try{shard.importEdges([&](auto& batch){
                    if(batchNumber++==2)return false;
                    EdgeRecord edge;edge.uuid=edgeUuid;edge.kind="duplicate";
                    edge.aAnchor.wholeGeometry=edge.bAnchor.wholeGeometry=true;
                    batch.push_back({{shard.storeUuid(),ids[i]},{shard.storeUuid(),ids[(i+1)%3]},edge});
                    return true;
                });}catch(const std::exception&){}
            });
    }
}

void sourceCompletion(size_t shards,bool packed)
{
    TemporaryDirectory dir;auto opt=options(shards,packed);
    const auto output=dir.path/"output";
    const auto coordinator=std::this_thread::get_id();
    std::atomic<size_t> finished=0;
    size_t completions=0;bool reject=true;
    opt.beforePublish=[&]{
        ++completions;
        require(std::this_thread::get_id()==coordinator,"source completion runs on the caller thread");
        require(finished==shards,"source completion follows every shard producer");
        require(!std::filesystem::exists(output),"source completion precedes publication");
        if(reject)throw std::runtime_error("source aggregate mismatch");
    };
    const auto produce=[&](size_t,auto& shard){
        auto row=fiber(util::newUuid());shard.appendObjects(std::span(&row,1));++finished;
    };
    mustFail([&]{importInitialCollection(output,opt,produce);});
    require(completions==1,"source completion runs exactly once per successful ingestion");
    require(std::filesystem::is_empty(dir.path),"source completion rejection removes all private staging");
    reject=false;finished=0;
    const auto result=importInitialCollection(output,opt,produce);
    require(completions==2&&std::filesystem::exists(result.path),"corrected source can retry the same destination");
    // A producer failure must not be followed by a successful completion hook.
    opt.beforePublish=[&]{++completions;};
    mustFail([&]{importInitialCollection(dir.path/"failed",opt,[](size_t,auto&){throw std::runtime_error("producer failed");});});
    require(completions==2&&!std::filesystem::exists(dir.path/"failed"),"producer failure skips completion and publication");
}

ExtensionTable stagedMeasurements()
{
    ExtensionTable table;table.name="ext_test_staged_measurements";table.extension="test_staging";table.keySpace="edges";
    table.columns={{"edge_uuid",ExtensionType::Blob},{"value",ExtensionType::Real},
        {"valid",ExtensionType::Integer},{"label",ExtensionType::Text,true},{"payload",ExtensionType::Blob}};
    table.primaryKey={"edge_uuid"};return table;
}
ExtensionTable immediateMeasurements()
{
    ExtensionTable table;table.name="ext_test_immediate_measurements";table.extension="test_staging";table.keySpace="edges";
    table.columns={{"edge_uuid",ExtensionType::Blob},{"ordinal",ExtensionType::Integer}};
    table.primaryKey={"edge_uuid"};return table;
}
void stagedExtensionTransfer(bool packed)
{
    TemporaryDirectory dir;auto opt=options(2,packed);
    const auto definition=stagedMeasurements(),immediate=immediateMeasurements();
    opt.extensions={definition,immediate};opt.edges.stagedExtensions={definition.name};
    std::vector<std::string> ids(2);for(auto& id:ids)id=util::newUuid();
    std::array<std::array<std::string,2>,2> edges;
    for(auto& shard:edges)for(auto& id:shard)id=util::newUuid();
    std::atomic<size_t> transformed=0;InitialImportResult result;
    {
        CountConnections connections;
        result=importInitialCollection(dir.path/"staged",opt,[&](size_t i,auto& shard){
            auto row=fiber(ids[i]);shard.appendObjects(std::span(&row,1));
        },[&](size_t i,auto& shard){
            size_t next=0;
            const auto count=shard.importEdges([&](auto& batch){
                if(next==2)return false;
                InitialCollectionEdge edge;edge.record.uuid=edges[i][next];edge.record.kind="declared_reference";
                edge.record.aAnchor.wholeGeometry=edge.record.bAnchor.wholeGeometry=true;
                edge.a={shard.storeUuid(),ids[i]};edge.b={shard.storeUuid(),ids[(i+1)%2]};
                const auto ordinal=int64_t(i*2+next);
                ExtensionRow values={extensionUuid(edge.record.uuid),double(ordinal)+.25,int64_t(1),
                    next?ExtensionValue(std::monostate{}):ExtensionValue(std::string("kept")),std::vector<uint8_t>{0,1,2}};
                edge.extensions.push_back({definition.name,std::move(values),std::vector<uint8_t>{3,4,5}});
                edge.extensions.push_back({immediate.name,{extensionUuid(edge.record.uuid),ordinal}});
                batch.push_back(std::move(edge));++next;return true;
            });
            require(count==2,"staging preserves native edge count");
            require(shard.transferStagedExtensionRows(definition.name,[&](ExtensionRow& row){
                require(row.size()==definition.columns.size()+1&&std::get<std::vector<uint8_t>>(row.back())==std::vector<uint8_t>({3,4,5}),
                    "temporary staging payload is available to the transform only");
                row[1]=-std::get<double>(row[1]);++transformed;
            },1)==2,"staged transfer returns rows written across bounded batches");
        });
        require(connectionCount==3,"staging must not reopen shards or create additional SQLite connections");
    }
    require(transformed==4,"every staged row is transformed exactly once");
    auto store=ObjectCollection::open(result.path,OpenOptions{.readOnly=true});auto txn=store.read();
    std::map<std::string,size_t> publishedDefinitions;
    for(const auto& table:store.extensionTables(txn)) {
        require(table.name==definition.name||table.name==immediate.name,
            "temporary staging must not leak into published extension metadata");
        ++publishedDefinitions[table.name];
    }
    require(publishedDefinitions.size()==2,"both public extension definitions are retained across shards");
    std::map<std::string,size_t> expected;
    for(size_t i=0;i<edges.size();++i)for(size_t j=0;j<edges[i].size();++j)expected.emplace(edges[i][j],i*2+j);
    size_t seen=0;
    store.scanExtensionRows(txn,definition.name,[&](const ExtensionRow& row){
        require(row.size()==definition.columns.size(),"staging-only payload is not a published column");
        const auto ordinal=expected.at(extensionUuid(row[0]));
        require(std::get<double>(row[1])==-(double(ordinal)+.25),"published rows contain transformed values, not staged originals");
        require(std::get<int64_t>(row[2])==1&&std::get<std::vector<uint8_t>>(row[4])==std::vector<uint8_t>({0,1,2}),
            "staging retains untouched integer and binary values");
        if(ordinal%2)require(std::holds_alternative<std::monostate>(row[3]),"staging retains NULL values");
        else require(std::get<std::string>(row[3])=="kept","staging retains text values");
        require(bool(store.edge(txn,{result.storeUuid,extensionUuid(row[0])})),"transferred measurement addresses the original native edge");
        ++seen;return true;
    });
    require(seen==4,"published staged row count");
    seen=0;store.scanExtensionRows(txn,immediate.name,[&](const ExtensionRow& row){
        require(std::get<int64_t>(row[1])==int64_t(expected.at(extensionUuid(row[0]))),"nonstaged extension rows are unchanged");
        ++seen;return true;
    });
    require(seen==4,"staged and immediate extensions coexist in the same edge stream");
}
void stagedExtensionFailures(bool packed)
{
    TemporaryDirectory dir;auto opt=options(1,packed);const auto definition=stagedMeasurements();
    opt.extensions={definition};opt.edges.stagedExtensions={definition.name};
    const auto object=util::newUuid(),otherObject=util::newUuid();
    for(int mode:{0,1,2}){
        const auto output=dir.path/("failed-"+std::to_string(mode));
        mustFail([&]{importInitialCollection(output,opt,[&](size_t,auto& shard){
            auto row=fiber(object);shard.appendObjects(std::span(&row,1));
            row=fiber(otherObject);shard.appendObjects(std::span(&row,1));
        },[&](size_t,auto& shard){
            int sent=0;shard.importEdges([&](auto& batch){
                if(sent==2)return false;
                InitialCollectionEdge edge;edge.record.uuid=util::newUuid();edge.record.kind="declared_reference";
                edge.record.aAnchor.wholeGeometry=edge.record.bAnchor.wholeGeometry=true;
                edge.a={shard.storeUuid(),object};edge.b={shard.storeUuid(),otherObject};
                edge.extensions.push_back({definition.name,{extensionUuid(edge.record.uuid),double(sent),int64_t(1),
                    std::monostate{},std::vector<uint8_t>{7}}});
                batch.push_back(std::move(edge));++sent;return true;
            });
            if(mode==0)return; // Untransferred staged rows must prevent sealing.
            size_t transformed=0;
            shard.transferStagedExtensionRows(definition.name,[&](ExtensionRow& row){
                if(mode==1&&++transformed==2)throw std::runtime_error("transform failed after one row");
                if(mode==2)row.pop_back(); // Invalid final row must not be published.
            },1);
        });});
        require(!std::filesystem::exists(output),"staging/transform failure must not publish any partial store");
        require(std::filesystem::is_empty(dir.path),"staging/transform failure cleans all private data");
    }
}
}
int main(){try{for(bool packed:{false,true}){for(size_t count:{1u,2u}){roundTrip(count,packed);sourceCompletion(count,packed);}phased(packed);directConnections(packed);dependencyReaderConnections(packed);stagedExtensionTransfer(packed);stagedExtensionFailures(packed);}failures();phaseFailures();
    std::cout<<"initial import passed\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
