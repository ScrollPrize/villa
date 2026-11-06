#include "vc_object_store/objects/AfvSource.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/Uuid.hpp"
#include "vc_reader/publication.hpp"
#include "../src/internal/Publication.hpp"
#include "geometry/GeometryInternal.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <future>
#include <atomic>
#include <mutex>
#include <csignal>
#include <sqlite3.h>
#ifndef _WIN32
#include <sys/resource.h>
#endif

using namespace vc::objects;
using Json = nlohmann::json;
namespace {
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t cancelled=0;
void cancelImport(int) { cancelled=1; }
void checkCancelled(const std::atomic<bool>* failed=nullptr) { if(cancelled || (failed && failed->load()))throw std::runtime_error("import cancelled"); }
struct CancelPeersOnFailure {
    std::atomic<bool>& failed;
    int exceptions=std::uncaught_exceptions();
    ~CancelPeersOnFailure(){if(std::uncaught_exceptions()>exceptions)failed=true;}
};
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
Json peakRssBytes()
{
#ifndef _WIN32
    rusage usage{};
    if(getrusage(RUSAGE_SELF,&usage)==0){
#ifdef __APPLE__
        return int64_t(usage.ru_maxrss);
#else
        return int64_t(usage.ru_maxrss)*1024;
#endif
    }
#endif
    return nullptr;
}
OpenOptions readOptions(const Json& config)
{
    OpenOptions options;options.readOnly=true;options.readerPoolSize=1;
    options.geometryCacheBytes=0;options.spatialCacheBytes=0;
    options.sqliteCacheBytes=config.value("sqlite_cache_bytes",options.sqliteCacheBytes);
    return options;
}
struct Staging {
    std::filesystem::path path;
    explicit Staging(const std::filesystem::path& output)
        : path(output.parent_path()/("."+output.filename().string()+"-"+util::newUuid()))
    { if (!std::filesystem::create_directory(path)) throw std::runtime_error("cannot create staging directory"); }
    ~Staging() { std::error_code ec; std::filesystem::remove_all(path,ec); }
};
GeometryData geometry(const AfvFiber& fiber)
{
    GeometryData g; g.kind="fiber";
    // Empty native arrays declare that the source has NO controls; no endpoints
    // or other artificial samples are introduced. Only line_points has payload.
    g.arrays={{"control_points","<f8",{0,3},{}}, {"control_point_ids","<i8",{0},{}},
              {"line_points","<f8",{fiber.detail.linePointCount,3},fiber.xyz}};
    return g;
}
ObjectRecord recordFor(const AfvFiber& fiber,const std::string& sourceUuid,const Json& coordinates)
{
    ObjectRecord record;record.uuid=util::nameUuid(sourceUuid,"fiber:"+std::to_string(fiber.id));
    record.kind="fiber";record.name=fiber.name;record.formatVersion=4;record.fiber=fiber.detail;
    record.meta=Json({{"coordinates",coordinates},{"source",{{"afv",{{"collection_uuid",sourceUuid},
        {"fiber_id",fiber.id},{"annotation",Json::parse(fiber.annotation)}}}}}}).dump();
    return record;
}
void verifyFiber(ObjectStore& store,const Txn& txn,const AfvFiber& fiber,const std::string& sourceUuid)
{
    const auto id=util::nameUuid(sourceUuid,"fiber:"+std::to_string(fiber.id));
    const auto record=store.objectByUuid(txn,id);const auto g=store.currentGeometry(txn,id);
    if(!record || !record->fiber || !g || record->name!=fiber.name || record->fiber->family!=fiber.detail.family ||
        record->fiber->length!=fiber.detail.length || record->fiber->bbox!=fiber.detail.bbox ||
        Json::parse(record->meta).at("source").at("afv").at("annotation")!=Json::parse(fiber.annotation) ||
        record->fiber->controlPointCount!=0 || record->fiber->linePointCount!=fiber.detail.linePointCount ||
        store.readGeometryArray(txn,g->uuid,"line_points")!=fiber.xyz ||
        !store.readGeometryArray(txn,g->uuid,"control_points").empty() ||
        !store.readGeometryArray(txn,g->uuid,"control_point_ids").empty())
        throw std::runtime_error("fiber metadata or geometry readback differs from source");
}
Json importRelationShards(const Json& config,const std::filesystem::path& output,AfvSource& fibers)
{
    const auto base=std::filesystem::absolute(config.at("geometry_collection").get<std::string>());
    const auto input=std::filesystem::absolute(config.at("constraints").get<std::string>());
    const auto limit=config.value("constraint_limit",int64_t(0));const auto n=config.value("shards",1);
    if(n<1 || n>256 || limit<0)throw std::invalid_argument("invalid relation partition settings");
    if(config.value("packed",false))throw std::invalid_argument("relation-only packed publication is not yet supported by this benchmark");
    const auto start=Clock::now();AfcSource source(input,fibers.uuid());const auto ranges=source.idRanges(n,limit);
    if(ranges.empty())throw std::invalid_argument("constraint source is empty");
    Staging staging(output);const auto owner=util::newUuid();const auto sourceUuid=fibers.uuid();
    std::string fiberOwner;
    {auto geometryStore=ObjectCollection::open(base,readOptions(config));auto txn=geometryStore.read();fiberOwner=geometryStore.uuid(txn);}
    std::vector<std::filesystem::path> paths;
    for(size_t i=0;i<ranges.size();++i)paths.push_back(staging.path/("relations-"+std::to_string(i)+".vcobj"));
    std::vector<std::future<Json>> workers;std::atomic<int64_t> completed{0};std::atomic<bool> failed{false};std::mutex progressMutex;auto progress=start;
    for(size_t i=0;i<ranges.size();++i)workers.push_back(std::async(std::launch::async,[&,i]{
        CancelPeersOnFailure cancelPeers{failed};
        const auto workerStart=Clock::now();int64_t count=0,metadata=0;double decode=0,insert=0;
        detail::GeometryWork work;
        std::optional<detail::GeometryWorkScope> workScope;
        if(config.value("profile",false))workScope.emplace(work);
        auto workerProgress=workerStart;
        {
            OpenOptions options;options.author=config.value("author",std::string("afv-import"));options.history=HistoryMode::Off;options.spatialIndex=false;options.creationStoreUuid=owner;
            options.sqliteCacheBytes=config.value("sqlite_cache_bytes",options.sqliteCacheBytes);
            options.readerPoolSize=1;options.geometryCacheBytes=0;options.spatialCacheBytes=0;
            // Each relation shard owns its own sparse endpoint references and
            // relation IDs; geometry remains solely in immutable dependencies.
            constexpr int64_t reservation=int64_t(1)<<40;
            options.objectIdRange={int64_t(i)*reservation+1,int64_t(i+1)*reservation};
            options.edgeIdRange=options.objectIdRange;
            auto store=ObjectCollection::create(paths[i],{base},options);auto txn=store.write();
            if(i==0){ObjectRecord metadataObject;metadataObject.uuid=owner;metadataObject.kind="collection";
                metadataObject.name="constraint-source";metadataObject.meta=Json({{"afc",Json::parse(source.metadataJson())}}).dump();
                store.upsertObject(txn,metadataObject);}
            AfcSource part(input,sourceUuid,INT64_MAX,ranges[i][0],ranges[i][1]);EdgeRecord edge;
            while(true){checkCancelled(&failed);auto t=Clock::now();const bool found=part.next(edge);decode+=seconds(t);if(!found)break;
                t=Clock::now();store.addEdge(txn,{fiberOwner,edge.aUuid},{fiberOwner,edge.bUuid},edge);insert+=seconds(t);
                ++count;++completed;metadata+=edge.meta.size();
                if(count%256==0){std::lock_guard lock(progressMutex);if(seconds(workScope?workerProgress:progress)>=5){
                    Json report={{"stage","relation_shard_import"},{"constraints",completed.load()},
                        {"shards",ranges.size()},{"elapsed_seconds",seconds(start)}};
                    if(workScope){report["worker"]=i;report["worker_constraints"]=count;
                        report["source_read_seconds"]=decode;report["native_import_seconds"]=insert;
                        report["phase_nanoseconds"]=work.phaseNanoseconds;workerProgress=Clock::now();}
                    std::cout<<report.dump()<<std::endl;progress=Clock::now();}}
            }
            store.seal(txn);txn.commit();
        }
        const auto imported=seconds(workerStart);const auto validation=Clock::now();
        // Resolve the qualified logical endpoints, not internal proxy IDs.
        {auto store=ObjectCollection::open(paths[i],readOptions(config));auto txn=store.read();
            AfcSource check(input,sourceUuid,INT64_MAX,ranges[i][0],ranges[i][1]);EdgeRecord expected;
            while(check.next(expected)){checkCancelled(&failed);const auto actual=store.edge(txn,{owner,expected.uuid});
                if(!actual)throw std::runtime_error("relation shard edge missing");
                if(actual->a.objectUuid==expected.bUuid){std::swap(expected.aUuid,expected.bUuid);std::swap(expected.aAnchor,expected.bAnchor);}
                if(actual->record.relationKey!=expected.relationKey || actual->record.kind!=expected.kind ||
                    Json::parse(actual->record.meta)!=Json::parse(expected.meta) ||
                    actual->a!=ObjectKey{fiberOwner,expected.aUuid} || actual->b!=ObjectKey{fiberOwner,expected.bUuid} ||
                    actual->record.aAnchor!=expected.aAnchor || actual->record.bAnchor!=expected.bAnchor)
                    throw std::runtime_error("relation shard readback differs from source");}
        }
        return Json({{"shard",i},{"constraints",count},{"metadata_json_bytes",metadata},{"source_read_seconds",decode},
            {"native_import_seconds",insert},{"import_wall_seconds",imported},{"validation_seconds",seconds(validation)}});
    }));
    Json perShard=Json::array();int64_t count=0,metadata=0;for(auto& worker:workers){auto r=worker.get();count+=r.at("constraints").get<int64_t>();metadata+=r.at("metadata_json_bytes").get<int64_t>();perShard.push_back(std::move(r));}
    if(count!=(limit?std::min(limit,source.constraintCount()):source.constraintCount()))throw std::runtime_error("relation aggregate count mismatch");
    const auto workersSeconds=seconds(start);const auto publish=Clock::now();
    {OpenOptions options;options.author=config.value("author",std::string("afv-import"));options.history=HistoryMode::Off;options.spatialIndex=false;
        options.sqliteCacheBytes=config.value("sqlite_cache_bytes",options.sqliteCacheBytes);
        auto collection=ObjectCollection::publishShards(staging.path/"collection.vcobj",paths,options);}
    uintmax_t bytes=0;for(const auto& entry:std::filesystem::directory_iterator(staging.path))if(entry.is_regular_file())bytes+=entry.file_size();
    vcr::files::publishDirectory(staging.path,output,false);
    return Json({{"stage","complete"},{"mode","relations"},{"source",input.string()},{"output",output.string()},
        {"sqlite_memory_statistics",config.value("sqlite_memory_statistics",true)},
        {"sqlite_cache_bytes",readOptions(config).sqliteCacheBytes},
        {"geometry_collection",base.string()},{"geometry_copied",false},{"shards",ranges.size()},{"constraints",count},
        {"constraint_metadata_json_bytes",metadata},{"source_metadata_json_bytes",source.metadataJson().size()},
        {"workers_including_readback_seconds",workersSeconds},{"publication_seconds",seconds(publish)},
        {"total_seconds",seconds(start)},{"output_bytes",bytes},{"peak_rss_bytes",peakRssBytes()},{"per_shard",perShard}});
}
Json importShards(const Json& config,const std::filesystem::path& input,const std::filesystem::path& output,
                  const Json& coordinates,AfvSource& source,int64_t limit)
{
    if(config.contains("constraints"))throw std::invalid_argument("sharded benchmark currently imports geometry only; use plain AFC import for relationship validation");
    const int shards=config.value("shards",1);const bool packed=config.value("packed",false);
    if(shards<1 || shards>256)throw std::invalid_argument("shards must be in [1,256]");
    const auto start=Clock::now();
    const auto ranges=source.idRanges(shards,limit);
    if(ranges.empty())throw std::invalid_argument("sharded AFV source is empty");
    Staging staging(output);
    std::unique_ptr<PackedStoreBuilder> builder;
    if(packed){
        PackedStoreOptions options;options.disposableBulk=true;
        options.residentBytes=config.value("resident_bytes",size_t(256*1024*1024));
        builder=std::make_unique<PackedStoreBuilder>(staging.path/"collection.vcobj",options);
    }
    std::vector<std::filesystem::path> paths;
    for(size_t i=0;i<ranges.size();++i)paths.push_back(packed?builder->addDatabase(util::newUuid()):staging.path/("part-"+std::to_string(i)+".vcobj"));
    const auto manifestId=util::newUuid();
    const auto manifest=packed?builder->addDatabase(manifestId):staging.path/"collection.vcobj";
    const auto namespaceUuid=source.uuid();const auto logicalOwner=util::newUuid();
    std::atomic<int64_t> completed{0};std::atomic<bool> failed{false};std::mutex progressMutex;auto progress=start;
    const auto metadata=source.metadataJson();
    std::vector<std::future<Json>> workers;
    for(size_t i=0;i<ranges.size();++i)workers.push_back(std::async(std::launch::async,[&,i]{
        CancelPeersOnFailure cancelPeers{failed};
        int64_t count=0,points=0,metadataBytes=0,first=-1,last=-1;
        double decode=0,insert=0;
        const auto workerStart=Clock::now();
        {
            OpenOptions options;options.author=config.value("author",std::string("afv-import"));
            options.sqliteCacheBytes=config.value("sqlite_cache_bytes",options.sqliteCacheBytes);
            options.history=HistoryMode::Off;options.spatialIndex=false;options.creationStoreUuid=logicalOwner;
            options.initialGeneration=1;options.objectIdRange={i?ranges[i][0]+2:1,ranges[i][1]+1};
            options.edgeIdRange={int64_t(i+1),int64_t(i+1)};options.readerPoolSize=1;
            options.geometryCacheBytes=0;options.spatialCacheBytes=0;
            auto store=ObjectStore::create(paths[i],options);auto txn=store.write();
            if(i==0){
                ObjectRecord collection;collection.uuid=namespaceUuid;collection.kind="collection";collection.name=input.stem().string();
                collection.meta=Json({{"coordinates",coordinates},{"afv",{{"metadata",Json::parse(metadata)}}}}).dump();
                store.insertObjectWithId(txn,collection,1);
            }
            AfvSource part(input,ranges[i][0],ranges[i][1]);AfvFiber fiber;
            if(part.uuid()!=namespaceUuid)throw std::runtime_error("AFV identity changed between partitions");
            while(true){
                checkCancelled(&failed);
                auto t=Clock::now();const bool found=part.next(fiber);decode+=seconds(t);if(!found)break;
                t=Clock::now();auto record=recordFor(fiber,namespaceUuid,coordinates);metadataBytes+=record.meta.size();
                store.insertObjectWithId(txn,record,fiber.id+2);store.putGeometry(txn,record.uuid,geometry(fiber));
                insert+=seconds(t);++count;points+=fiber.detail.linePointCount;if(first<0)first=fiber.id;last=fiber.id;
                ++completed;
                if(count%1024==0){
                    std::lock_guard lock(progressMutex);
                    if(seconds(progress)>=5){std::cout<<Json({{"stage","shard_import"},{"fibers",completed.load()},
                        {"shards",ranges.size()},{"elapsed_seconds",seconds(start)}}).dump()<<std::endl;progress=Clock::now();}
                }
            }
            store.seal(txn);txn.commit();
        }
        const auto imported=seconds(workerStart);const auto validation=Clock::now();
        {
            auto store=ObjectStore::open(paths[i],readOptions(config));
            auto txn=store.read();
            if(store.provisioning(txn).generation!=2 || !store.provisioning(txn).sealed || store.objectCount(txn,"fiber")!=count)
                throw std::runtime_error("shard count, generation or sealing failed");
            AfvSource part(input,ranges[i][0],ranges[i][1]);AfvFiber fiber;
            while(part.next(fiber)){checkCancelled(&failed);verifyFiber(store,txn,fiber,namespaceUuid);}
        }
        return Json({{"shard",i},{"fibers",count},{"points",points},{"first_id",first},{"last_id",last},
            {"source_read_seconds",decode},{"native_import_seconds",insert},{"import_wall_seconds",imported},
            {"validation_seconds",seconds(validation)},{"metadata_json_bytes",metadataBytes}});
    }));
    Json perShard=Json::array();int64_t fibers=0,points=0,metadataBytes=0;
    for(auto& worker:workers){auto result=worker.get();fibers+=result.at("fibers").get<int64_t>();points+=result.at("points").get<int64_t>();metadataBytes+=result.at("metadata_json_bytes").get<int64_t>();perShard.push_back(std::move(result));}
    if(fibers!=(limit?std::min(limit,source.fiberCount()):source.fiberCount()) || (!limit && points!=source.pointCount()))
        throw std::runtime_error("shard aggregate totals do not match AFV source");
    const auto workersSeconds=seconds(start);const auto publicationStart=Clock::now();
    const auto verifyRoutes=[&](ObjectCollection& collection){
        auto txn=collection.read();
        // Validate range routing at both actual endpoints of every partition.
        for(const auto& shard:perShard)for(const auto* key:{"first_id","last_id"}){
            const auto id=shard.at(key).get<int64_t>();const auto record=collection.objectById(txn,logicalOwner,id+2);
            if(!record || record->record.uuid!=util::nameUuid(namespaceUuid,"fiber:"+std::to_string(id)))
                throw std::runtime_error("collection numeric routing mismatch");
            AfvSource one(input,id,id+1);AfvFiber fiber;
            if(!one.next(fiber) || collection.readGeometryArray(txn,record->key,"line_points")!=fiber.xyz)
                throw std::runtime_error("collection geometry routing mismatch");
        }
    };
    {
        OpenOptions options;options.author=config.value("author",std::string("afv-import"));options.history=HistoryMode::Off;options.spatialIndex=false;
        options.sqliteCacheBytes=config.value("sqlite_cache_bytes",options.sqliteCacheBytes);
        auto collection=ObjectCollection::publishShards(manifest,paths,options);
        verifyRoutes(collection);
    }
    if(packed){
        builder->finalize(manifestId);
        auto published=ObjectCollection::open(staging.path/"collection.vcobj",readOptions(config));
        verifyRoutes(published);
    }
    const auto publicationSeconds=seconds(publicationStart);
    uintmax_t outputBytes=0;
    if(packed){outputBytes=std::filesystem::file_size(staging.path/"collection.vcobj");detail::publishNewFile(staging.path/"collection.vcobj",output);}
    else{
        for(const auto& entry:std::filesystem::directory_iterator(staging.path))if(entry.is_regular_file())outputBytes+=entry.file_size();
        vcr::files::publishDirectory(staging.path,output,false);
    }
    return Json({{"stage","complete"},{"source",input.string()},{"output",output.string()},
        {"sqlite_memory_statistics",config.value("sqlite_memory_statistics",true)},
        {"sqlite_cache_bytes",readOptions(config).sqliteCacheBytes},
        {"shards",ranges.size()},{"packed",packed},{"store_uuid",logicalOwner},{"source_uuid",namespaceUuid},{"fibers",fibers},{"points",points},{"geometry_bytes",points*24},
        {"metadata_json_bytes",metadataBytes},{"workers_including_readback_seconds",workersSeconds},
        {"publication_seconds",publicationSeconds},{"total_seconds",seconds(start)},{"output_bytes",outputBytes},
        {"peak_rss_bytes",peakRssBytes()},{"constraints_imported",false},{"per_shard",perShard}});
}
}

int main(int argc,char** argv)
{
    try {
        std::signal(SIGINT,cancelImport);std::signal(SIGTERM,cancelImport);
        if (argc!=2) throw std::invalid_argument("usage: vc_object_import_afv CONFIG.json");
        std::ifstream file(argv[1]); if (!file) throw std::runtime_error("cannot read config");
        const auto config=Json::parse(file);
        // Benchmark-only opt-in. Do not alter the application's SQLite global
        // allocator configuration inside ObjectStore or a worker thread.
        if(!config.value("sqlite_memory_statistics",true) && sqlite3_config(SQLITE_CONFIG_MEMSTATUS,0)!=SQLITE_OK)
            throw std::runtime_error("SQLite was already initialized; cannot disable memory statistics");
        const auto input=std::filesystem::absolute(config.at("source").get<std::string>());
        const auto output=std::filesystem::absolute(config.at("output").get<std::string>());
        if (std::filesystem::exists(output) || std::filesystem::is_symlink(output)) throw std::runtime_error("output exists");
        if (!std::filesystem::is_directory(output.parent_path())) throw std::runtime_error("output parent must exist");
        const auto limit=config.value("fiber_limit", int64_t(0));
        const auto batch=config.value("batch_size", int64_t(256));
        const auto maxBytes=config.value("batch_bytes", int64_t(64*1024*1024));
        if (limit<0 || batch<1 || maxBytes<1) throw std::invalid_argument("invalid limits");
        const auto volume=config.value("volume_id",std::string{});
        if (volume.empty() && !config.value("unbound",false)) throw std::invalid_argument("volume_id or explicit unbound required");
        const auto scale=config.value("scale_factor",1.);
        if (!std::isfinite(scale) || scale<=0) throw std::invalid_argument("invalid scale_factor");
        const Json coordinates={{"volume_uuid",volume.empty()?Json(nullptr):Json(volume)}, {"scale_factor",scale}};
        AfvSource source(input);
        const auto sourceMetadata=Json::parse(source.metadataJson());
        if(config.contains("geometry_collection")){
            const auto result=importRelationShards(config,output,source);std::cout<<result.dump(2)<<std::endl;
            if(config.contains("report")){std::ofstream report(config.at("report").get<std::string>());report<<result.dump(2)<<'\n';if(!report)throw std::runtime_error("report write failed");}
            return 0;
        }
        if(config.value("shards",1)>1 || config.value("packed",false)){
            const auto result=importShards(config,input,output,coordinates,source,limit);
            std::cout<<result.dump(2)<<std::endl;
            if(config.contains("report")){std::ofstream report(config.at("report").get<std::string>());report<<result.dump(2)<<'\n';if(!report)throw std::runtime_error("report write failed");}
            return 0;
        }
        std::cout<<Json({{"stage","inventory"},{"fibers",source.fiberCount()},{"points",source.pointCount()},
            {"geometry_bytes",source.pointCount()*24},{"fiber_limit",limit},{"constraints_requested",config.contains("constraints")}}).dump()<<std::endl;
        Staging staging(output);
        int64_t fibers=0,points=0,metadataBytes=0,edges=0,edgeMetadataBytes=0,lastFiberId=-1;
        double readSeconds=0,importSeconds=0,finishSeconds=0,edgeReadSeconds=0,edgeImportSeconds=0;
        const bool hasConstraints=config.contains("constraints");
        const auto edgeLimit=config.value("constraint_limit",int64_t(0));
        if(edgeLimit<0)throw std::invalid_argument("invalid constraint limit");
        const auto start=Clock::now(); auto progress=start,closeStarted=start;
        {
            OpenOptions options; options.author=config.value("author",std::string("afv-import"));
            options.sqliteCacheBytes=config.value("sqlite_cache_bytes",options.sqliteCacheBytes);
            options.history=HistoryMode::Off; options.spatialIndex=false;
            options.readerPoolSize=1; options.geometryCacheBytes=0; options.spatialCacheBytes=0;
            auto store=ObjectStore::create(staging.path/"objects.vcobj",options);
            {
                auto txn=store.write(); ObjectRecord collection;
                collection.uuid=source.uuid(); collection.kind="collection"; collection.name=input.stem().string();
                collection.meta=Json({{"coordinates",coordinates},{"afv",{{"metadata",sourceMetadata}}}}).dump();
                store.upsertObject(txn,collection); txn.commit();
            }
            bool ended=false;
            while (!ended && (!limit || fibers<limit)) {
                checkCancelled();
                std::vector<AfvFiber> pending; int64_t bytes=0;
                auto t=Clock::now();
                while (static_cast<int64_t>(pending.size())<batch && bytes<maxBytes && (!limit || fibers+static_cast<int64_t>(pending.size())<limit)) {
                    AfvFiber fiber; if (!source.next(fiber)) { ended=true; break; }
                    bytes+=fiber.xyz.size(); pending.push_back(std::move(fiber));
                }
                readSeconds+=seconds(t); t=Clock::now();
                if (!pending.empty()) {
                    auto txn=store.write();
                    for (const auto& fiber:pending) {
                        auto record=recordFor(fiber,source.uuid(),coordinates);
                        metadataBytes+=record.meta.size();
                        store.upsertObject(txn,record); store.putGeometry(txn,record.uuid,geometry(fiber));
                        ++fibers; points+=fiber.detail.linePointCount;
                        lastFiberId=fiber.id;
                    }
                    txn.commit();
                }
                importSeconds+=seconds(t);
                if (seconds(progress)>=5) {
                    const double rate=fibers/seconds(start);
                    std::cout<<Json({{"stage","import"},{"fibers",fibers},{"points",points},{"elapsed_seconds",seconds(start)},
                        {"fibers_per_second",rate},{"estimated_full_import_seconds",source.fiberCount()/rate}}).dump()<<std::endl;
                    progress=Clock::now();
                }
            }
            if(hasConstraints){
                AfcSource afc(config.at("constraints").get<std::string>(),source.uuid(),limit?lastFiberId+1:INT64_MAX);
                {auto txn=store.write();store.setMeta(txn,source.uuid(),"afc",afc.metadataJson());txn.commit();}
                bool eof=false;
                while(!eof && (!edgeLimit || edges<edgeLimit)){
                    auto t=Clock::now();std::vector<EdgeRecord> pending;
                    while(static_cast<int64_t>(pending.size())<batch && (!edgeLimit || edges+static_cast<int64_t>(pending.size())<edgeLimit)){
                        checkCancelled();
                        EdgeRecord edge;if(!afc.next(edge)){eof=true;break;}pending.push_back(std::move(edge));
                    }
                    edgeReadSeconds+=seconds(t);t=Clock::now();
                    if(!pending.empty()){
                        auto txn=store.write();for(const auto& edge:pending){store.addEdge(txn,edge);++edges;edgeMetadataBytes+=edge.meta.size();}txn.commit();
                    }
                    edgeImportSeconds+=seconds(t);
                    if(seconds(progress)>=5){
                        std::cout<<Json({{"stage","constraints"},{"constraints",edges},{"elapsed_seconds",seconds(start)},
                            {"source_read_seconds",edgeReadSeconds},{"native_import_seconds",edgeImportSeconds}}).dump()<<std::endl;
                        progress=Clock::now();
                    }
                }
            }
            const auto t=Clock::now();
            { auto txn=store.write(); store.seal(txn); txn.commit(); }
            finishSeconds+=seconds(t);
            closeStarted=Clock::now();
        }
        const auto closeSeconds=seconds(closeStarted);
        const auto validateStart=Clock::now();
        {
            auto store=ObjectStore::open(staging.path/"objects.vcobj",readOptions(config));
            auto txn=store.read();
            if (store.objectCount(txn,"fiber")!=fibers || !store.provisioning(txn).sealed)
                throw std::runtime_error("output count or seal validation failed");
            AfvSource check(input); AfvFiber fiber;
            // Validate every imported source sample, not a random spot check.
            for (int64_t i=0;i<fibers;++i) {
                if (!check.next(fiber)) throw std::runtime_error("source ended during readback");
                verifyFiber(store,txn,fiber,check.uuid());
            }
            if(hasConstraints){
                AfcSource checkEdges(config.at("constraints").get<std::string>(),source.uuid(),limit?lastFiberId+1:INT64_MAX);
                for(int64_t i=0;i<edges;++i){
                    EdgeRecord expected;if(!checkEdges.next(expected))throw std::runtime_error("constraint source ended during readback");
                    const auto actual=store.edgeByUuid(txn,expected.uuid);
                    if(!actual || actual->relationKey!=expected.relationKey || Json::parse(actual->meta)!=Json::parse(expected.meta) || actual->kind!=expected.kind)
                        throw std::runtime_error("constraint metadata readback differs from source");
                    if(actual->aUuid==expected.bUuid){std::swap(expected.aUuid,expected.bUuid);std::swap(expected.aAnchor,expected.bAnchor);}
                    if(actual->aUuid!=expected.aUuid || actual->bUuid!=expected.bUuid || actual->aAnchor!=expected.aAnchor || actual->bAnchor!=expected.bAnchor)
                        throw std::runtime_error("constraint endpoint readback differs from source");
                }
            }
        }
        const double validationSeconds=seconds(validateStart);
        // No overwrite race: publication fails if another process created output.
        detail::publishNewFile(staging.path/"objects.vcobj",output);
        const Json result={{"stage","complete"},{"source",input.string()},{"output",output.string()},
            {"sqlite_memory_statistics",config.value("sqlite_memory_statistics",true)},
            {"sqlite_cache_bytes",readOptions(config).sqliteCacheBytes},
            {"fibers",fibers},{"points",points},{"geometry_bytes",points*24},{"metadata_json_bytes",metadataBytes},
            {"source_read_seconds",readSeconds},{"native_import_seconds",importSeconds},
            {"seal_seconds",finishSeconds},{"validation_seconds",validationSeconds},{"total_seconds",seconds(start)},
            {"close_checkpoint_seconds",closeSeconds},{"peak_rss_bytes",peakRssBytes()},
            {"output_bytes",std::filesystem::file_size(output)},{"constraints_imported",hasConstraints},{"constraints",edges},
            {"constraint_metadata_json_bytes",edgeMetadataBytes},{"constraint_source_read_seconds",edgeReadSeconds},
            {"constraint_import_seconds",edgeImportSeconds},{"constraint_limit",edgeLimit},
            {"estimated_full_import_seconds",fibers ? (readSeconds+importSeconds)*source.fiberCount()/fibers : 0.}};
        std::cout<<result.dump(2)<<std::endl;
        if (config.contains("report")) { std::ofstream report(config.at("report").get<std::string>()); report<<result.dump(2)<<'\n'; if (!report) throw std::runtime_error("report write failed"); }
        return 0;
    } catch (const std::exception& error) { std::cerr<<"AFV import failed: "<<error.what()<<'\n'; return 1; }
}
