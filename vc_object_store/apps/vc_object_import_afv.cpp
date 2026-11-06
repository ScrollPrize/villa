#include "vc_object_store/objects/AfvSource.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/Uuid.hpp"
#include "geometry/GeometryInternal.hpp"
#include "support/SqliteMutexProfile.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <climits>
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
Json collectWorkers(std::vector<std::future<Json>>& workers)
{
    Json results=Json::array();std::exception_ptr failure,cancellation;
    for(auto& worker:workers){
        try{results.push_back(worker.get());}
        catch(const std::exception& error){
            if(std::string_view(error.what())=="import cancelled")cancellation=std::current_exception();
            else if(!failure)failure=std::current_exception();
        }
        catch(...){if(!failure)failure=std::current_exception();}
    }
    if(failure)std::rethrow_exception(failure);
    if(cancellation)std::rethrow_exception(cancellation);
    return results;
}
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
template<class Read>
void verifyFiberData(const ObjectRecord* record,bool hasGeometry,const AfvFiber& fiber,Read&& read)
{
    if(!record || !record->fiber || !hasGeometry || record->name!=fiber.name || record->fiber->family!=fiber.detail.family ||
        record->fiber->length!=fiber.detail.length || record->fiber->bbox!=fiber.detail.bbox ||
        Json::parse(record->meta).at("source").at("afv").at("annotation")!=Json::parse(fiber.annotation) ||
        record->fiber->controlPointCount!=0 || record->fiber->linePointCount!=fiber.detail.linePointCount ||
        read("line_points")!=fiber.xyz || !read("control_points").empty() || !read("control_point_ids").empty())
        throw std::runtime_error("fiber metadata or geometry readback differs from source");
}
Json importNative(const Json& config,const std::filesystem::path& input,const std::filesystem::path& output,
                  const Json& coordinates,AfvSource& source,int64_t limit)
{
    const bool linked=config.contains("geometry_collection"),hasEdges=config.contains("constraints");
    const auto edgeLimit=config.value("constraint_limit",int64_t(0));
    const auto workers=config.value("shards",1);
    const auto batchSize=config.value("batch_size",size_t(1024));
    const auto batchBytes=config.value("batch_bytes",size_t(64*1024*1024));
    if(workers<1||workers>256||edgeLimit<0||!batchSize||batchSize>1000000||!batchBytes)
        throw std::invalid_argument("invalid initial import limits");
    if(linked&&!hasEdges)throw std::invalid_argument("geometry_collection requires constraints");
    const auto start=Clock::now();const auto sourceUuid=source.uuid();
    const auto afvMetadata=source.metadataJson();
    std::unique_ptr<AfcSource> edgeSource;
    if(linked)edgeSource=std::make_unique<AfcSource>(config.at("constraints").get<std::string>(),sourceUuid);
    const auto ranges=linked?edgeSource->idRanges(workers,edgeLimit):source.idRanges(workers,limit);
    if(ranges.empty())throw std::invalid_argument("initial import source is empty");
    if(!linked&&hasEdges&&ranges.size()>1)
        throw std::invalid_argument("combined geometry/constraint import requires one shard; use a sealed geometry_collection for relation shards");
    if(!linked&&hasEdges)edgeSource=std::make_unique<AfcSource>(config.at("constraints").get<std::string>(),sourceUuid,
        limit?ranges.back()[1]:INT64_MAX);
    InitialImportOptions options;options.store.author=config.value("author",std::string("afv-import"));
    // This streaming source partitions by AFV numeric ID ranges, not UUID hash.
    // Do not advertise direct UUID routing until the producer is repartitioned.
    options.uuidHashRouting=false;
    options.store.sqliteCacheBytes=config.value("sqlite_cache_bytes",options.store.sqliteCacheBytes);
    options.store.readerPoolSize=1;options.store.geometryCacheBytes=0;options.store.spatialCacheBytes=0;
    options.edges.deferIndexes=config.value("defer_indexes",true);
    options.edges.sqlBatchRows=config.value("sql_batch_rows",size_t(1));
    if(config.value("packed",false)){
        options.packed=PackedStoreOptions{};options.packed->disposableBulk=true;
        options.packed->residentBytes=config.value("resident_bytes",size_t(256*1024*1024));
        options.packed->memoryFirst=config.value("memory_first",false);
        options.packed->backgroundFlush=config.value("background_flush",true);
    }
    std::string fiberOwner;
    if(linked){const auto base=std::filesystem::absolute(config.at("geometry_collection").get<std::string>());
        options.dependencies={base};auto geometryStore=ObjectCollection::open(base,readOptions(config));auto t=geometryStore.read();fiberOwner=geometryStore.uuid(t);}
    options.shards.clear();
    for(size_t i=0;i<ranges.size();++i){
        if(linked){constexpr int64_t reservation=int64_t(1)<<40;
            options.shards.push_back({{int64_t(i)*reservation+1,int64_t(i+1)*reservation},{int64_t(i)*reservation+1,int64_t(i+1)*reservation}});
        }else{
            if(ranges[i][1]>INT64_MAX-2)throw std::invalid_argument("source IDs exceed native ID mapping range");
            options.shards.push_back({{i?ranges[i][0]+2:1,ranges[i][1]+1},
                hasEdges?std::array<int64_t,2>{1,INT64_MAX}:std::array<int64_t,2>{int64_t(i+1),int64_t(i+1)}});
        }
    }
    struct Counts {int64_t fibers=0,points=0,metadata=0,edges=0,edgeMetadata=0,first=-1,last=-1;double read=0,insert=0,edgeRead=0,edgeInsert=0;};
    std::vector<Counts> counts(ranges.size());std::vector<Json> phases(ranges.size());
    std::atomic<int64_t> completedFibers{0},completedEdges{0};std::mutex progressMutex;auto progress=start;
    const auto imported=importInitialCollection(output,options,[&](size_t i,InitialShardImport& shard){
        auto& c=counts[i];detail::GeometryWork work;std::optional<detail::GeometryWorkScope> scope;
        if(config.value("profile",false))scope.emplace(work);
        auto workerProgress=Clock::now();
        auto report=[&]{std::lock_guard lock(progressMutex);if(seconds(scope?workerProgress:progress)<5)return;
            Json line={{"stage",linked?"relation_shard_import":"shard_import"},{"fibers",completedFibers.load()},
                {"constraints",completedEdges.load()},{"shards",ranges.size()},{"elapsed_seconds",seconds(start)}};
            if(scope){line["worker"]=i;line["worker_constraints"]=c.edges;line["source_read_seconds"]=linked?c.edgeRead:c.read;
                line["native_import_seconds"]=linked?c.edgeInsert:c.insert;line["phase_nanoseconds"]=work.phaseNanoseconds;workerProgress=Clock::now();}
            if(config.value("profile_sqlite_mutex",false)){line["worker"]=i;line["sqlite_mutex_profile"]=benchmark::sqlite_profile::snapshot();}
            std::cout<<line.dump()<<std::endl;progress=Clock::now();};
        if(i==0){ObjectRecord metadata;metadata.uuid=linked?shard.storeUuid():sourceUuid;metadata.kind="collection";
            metadata.name=linked?"constraint-source":input.stem().string();
            Json fields=linked?Json::object():Json{{"coordinates",coordinates},{"afv",{{"metadata",Json::parse(afvMetadata)}}}};
            if(edgeSource)fields["afc"]=Json::parse(edgeSource->metadataJson());metadata.meta=fields.dump();
            InitialObjectRecord row{std::move(metadata),linked?std::optional<int64_t>{}:std::optional<int64_t>{1},{}};
            shard.appendObjects(std::span(&row,1));}
        if(!linked){AfvSource part(input,ranges[i][0],ranges[i][1]);AfvFiber fiber;bool eof=false;
            if(part.uuid()!=sourceUuid)throw std::runtime_error("AFV identity changed between partitions");
            while(!eof){checkCancelled();const auto decode=Clock::now();std::vector<InitialObjectRecord> rows;size_t bytes=0;rows.reserve(batchSize);
                while(rows.size()<batchSize&&bytes<batchBytes){if(!part.next(fiber)){eof=true;break;}
                    auto r=recordFor(fiber,sourceUuid,coordinates);c.metadata+=r.meta.size();bytes+=fiber.xyz.size();
                    rows.push_back({std::move(r),fiber.id+2,geometry(fiber)});
                    ++c.fibers;c.points+=fiber.detail.linePointCount;if(c.first<0)c.first=fiber.id;c.last=fiber.id;
                }
                c.read+=seconds(decode);const auto writing=Clock::now();
                if(!rows.empty()){shard.appendObjects(rows);completedFibers+=int64_t(rows.size());}
                c.insert+=seconds(writing);report();
            }
        }
        if(hasEdges){std::unique_ptr<AfcSource> partition;
            if(linked)partition=std::make_unique<AfcSource>(config.at("constraints").get<std::string>(),sourceUuid,INT64_MAX,ranges[i][0],ranges[i][1]);
            auto& part=linked?*partition:*edgeSource;
            size_t pending=0;auto previous=Clock::now();EdgeRecord edge;
            const auto count=shard.importEdges([&](std::vector<InitialCollectionEdge>& rows){
                checkCancelled();if(pending){c.edgeInsert+=seconds(previous);c.edges+=int64_t(pending);completedEdges+=int64_t(pending);report();}
                const auto decode=Clock::now();rows.reserve(batchSize);
                while(rows.size()<batchSize&&(linked||!edgeLimit||c.edges+int64_t(rows.size())<edgeLimit)&&part.next(edge)){
                    c.edgeMetadata+=edge.meta.size();const auto& owner=linked?fiberOwner:shard.storeUuid();
                    ObjectKey a{owner,edge.aUuid},b{owner,edge.bUuid};rows.push_back({std::move(a),std::move(b),std::move(edge)});
                }
                c.edgeRead+=seconds(decode);pending=rows.size();previous=Clock::now();return !rows.empty();
            });
            c.edgeInsert+=seconds(previous);if(count!=c.edges)throw std::runtime_error("initial relation count mismatch");
        }
        phases[i]=work.phaseNanoseconds;
    });
    int64_t fibers=0,points=0,edges=0,metadata=0,edgeMetadata=0;
    Json perShard=Json::array();
    for(size_t i=0;i<ranges.size();++i){
        const auto& c=counts[i];const auto& timing=imported.shards[i];
        if(timing.objects!=c.fibers+(i==0?1:0)||timing.edges!=c.edges)throw std::runtime_error("native initial counts differ from producer");
        perShard.push_back({{"shard",i},{"fibers",c.fibers},{"points",c.points},{"constraints",c.edges},{"first_id",c.first},{"last_id",c.last},
            {"source_read_seconds",linked?c.edgeRead:c.read},{"native_import_seconds",linked?c.edgeInsert:c.insert},
            {"constraint_source_read_seconds",c.edgeRead},{"constraint_import_seconds",c.edgeInsert},{"metadata_json_bytes",linked?c.edgeMetadata:c.metadata},
            {"creation_seconds",timing.creationSeconds},{"seal_commit_seconds",timing.sealCommitSeconds},
            {"import_wall_seconds",timing.producerSeconds+timing.objectSeconds+timing.edgeSeconds+timing.creationSeconds+timing.sealCommitSeconds},
            {"validation_seconds",0.0},{"phase_nanoseconds",phases[i]}});
    }
    const bool verify=config.value("verify_readback",false);
    double validationSeconds=0;
    if(verify) {
    const auto verifyStart=Clock::now();
    auto joined=ObjectCollection::open(imported.path,readOptions(config));const auto members=joined.members();
    struct FiberSample {int64_t id;std::string uuid;std::vector<uint8_t> xyz;};
    std::vector<std::vector<FiberSample>> fiberSamples(ranges.size());
    std::vector<std::optional<EdgeRecord>> edgeSamples(ranges.size());
    std::vector<std::future<Json>> validationJobs;std::atomic<bool> validationFailed{false};
    for(size_t i=0;i<ranges.size();++i)validationJobs.push_back(std::async(std::launch::async,[&,i]{
        CancelPeersOnFailure cancelPeers{validationFailed};
        const auto& c=counts[i];const auto validation=Clock::now();checkCancelled(&validationFailed);
        auto physical=imported.path;
        if(ranges.size()>1){
            const auto found=std::find_if(members.begin(),members.end(),[&](const auto& m){return m.partition&&m.storeUuid==imported.storeUuid&&
                m.objectIds==options.shards[i].objectIds&&m.edgeIds==options.shards[i].edgeIds;});
            if(found==members.end())throw std::runtime_error("published shard routing metadata missing");physical=found->locator;
        }
        auto published=ObjectCollection::open(physical,readOptions(config));auto txn=published.read();
        if(!linked){AfvSource part(input,ranges[i][0],ranges[i][1]);AfvFiber fiber;int64_t verified=0;
            while(part.next(fiber)){checkCancelled(&validationFailed);const ObjectKey key{imported.storeUuid,util::nameUuid(sourceUuid,"fiber:"+std::to_string(fiber.id))};
                const auto value=published.object(txn,key);const auto g=published.currentGeometry(txn,key);
                verifyFiberData(value?&value->record:nullptr,bool(g),fiber,[&](const char* name){return published.readGeometryArray(txn,key,name);});++verified;
                if(fiber.id==c.first||fiber.id==c.last)fiberSamples[i].push_back({fiber.id,key.objectUuid,fiber.xyz});}
            if(verified!=c.fibers)throw std::runtime_error("published fiber count differs from import");
            for(const auto id:{c.first,c.last})if(id>=0){const auto routed=published.objectById(txn,imported.storeUuid,id+2);
                if(!routed||routed->record.uuid!=util::nameUuid(sourceUuid,"fiber:"+std::to_string(id)))throw std::runtime_error("published fiber ID routing mismatch");}
        }
        if(hasEdges){AfcSource part(config.at("constraints").get<std::string>(),sourceUuid,
                !linked&&limit?c.last+1:INT64_MAX,linked?ranges[i][0]:0,linked?ranges[i][1]:INT64_MAX);
            EdgeRecord expected;const auto& owner=linked?fiberOwner:imported.storeUuid;
            for(int64_t n=0;n<c.edges;++n){checkCancelled(&validationFailed);if(!part.next(expected))throw std::runtime_error("constraint source ended during readback");
                const auto actual=published.edge(txn,{imported.storeUuid,expected.uuid});if(!actual)throw std::runtime_error("published relation missing");
                if(actual->a.objectUuid==expected.bUuid){std::swap(expected.aUuid,expected.bUuid);std::swap(expected.aAnchor,expected.bAnchor);}
                if(actual->a!=ObjectKey{owner,expected.aUuid}||actual->b!=ObjectKey{owner,expected.bUuid}||
                    actual->record.aAnchor!=expected.aAnchor||actual->record.bAnchor!=expected.bAnchor||actual->record.kind!=expected.kind||
                    Json::parse(actual->record.meta)!=Json::parse(expected.meta))
                    throw std::runtime_error("published constraint differs from source");
                if(n==0)edgeSamples[i]=expected;
            }
        }
        return Json({{"validation_seconds",seconds(validation)}});
    }));
    const auto validation=collectWorkers(validationJobs);
    for(size_t i=0;i<ranges.size();++i)perShard[i]["validation_seconds"]=validation[i].at("validation_seconds");
    // Full exact readback above stays parallel and shard-local. Test composed
    // numeric/UUID/dependency routing on representative already-read samples.
    {auto txn=joined.read();for(size_t i=0;i<ranges.size();++i){
        for(const auto& sample:fiberSamples[i]){
            const auto value=joined.objectById(txn,imported.storeUuid,sample.id+2);
            if(!value||value->record.uuid!=sample.uuid||joined.readGeometryArray(txn,value->key,"line_points")!=sample.xyz)
                throw std::runtime_error("composed fiber routing differs from source");
        }
        if(edgeSamples[i]){
            auto expected=*edgeSamples[i];const auto actual=joined.edge(txn,{imported.storeUuid,expected.uuid});
            if(!actual)throw std::runtime_error("composed relation routing missing");
            if(actual->a.objectUuid==expected.bUuid){std::swap(expected.aUuid,expected.bUuid);std::swap(expected.aAnchor,expected.bAnchor);}
            const auto& owner=linked?fiberOwner:imported.storeUuid;
            if(actual->a!=ObjectKey{owner,expected.aUuid}||actual->b!=ObjectKey{owner,expected.bUuid}||
                actual->record.aAnchor!=expected.aAnchor||actual->record.bAnchor!=expected.bAnchor||Json::parse(actual->record.meta)!=Json::parse(expected.meta))
                throw std::runtime_error("composed relation routing differs from source");
            if(linked)for(const auto& endpoint:{actual->a,actual->b})
                if(!joined.currentGeometry(txn,endpoint)||joined.readGeometryArray(txn,endpoint,"line_points").empty())
                    throw std::runtime_error("composed linked geometry routing failed");
        }
    }}
    validationSeconds=seconds(verifyStart);
    }
    for(const auto& c:counts){fibers+=c.fibers;points+=c.points;edges+=c.edges;metadata+=c.metadata;edgeMetadata+=c.edgeMetadata;}
    if(!linked&&(fibers!=(limit?std::min(limit,source.fiberCount()):source.fiberCount())||(!limit&&points!=source.pointCount())))
        throw std::runtime_error("initial fiber aggregate totals differ from source");
    if(linked&&edges!=(edgeLimit?std::min(edgeLimit,edgeSource->constraintCount()):edgeSource->constraintCount()))
        throw std::runtime_error("initial constraint aggregate total differs from source");
    double sourceSeconds=0,nativeSeconds=0,edgeReadSeconds=0,edgeWriteSeconds=0,sealSeconds=0;
    for(size_t i=0;i<counts.size();++i){sourceSeconds+=counts[i].read;nativeSeconds+=counts[i].insert;
        edgeReadSeconds+=counts[i].edgeRead;edgeWriteSeconds+=counts[i].edgeInsert;sealSeconds+=imported.shards[i].sealCommitSeconds;}
    uintmax_t bytes=0;if(std::filesystem::is_regular_file(output))bytes=std::filesystem::file_size(output);
    else for(const auto& e:std::filesystem::recursive_directory_iterator(output))if(e.is_regular_file())bytes+=e.file_size();
    Json result={{"stage","complete"},{"mode",linked?"relations":"geometry"},{"source",linked?std::filesystem::absolute(config.at("constraints").get<std::string>()).string():input.string()},{"output",output.string()},
        {"store_uuid",imported.storeUuid},{"source_uuid",sourceUuid},{"initial_batch",true},{"shards",ranges.size()},
        {"batch_size",batchSize},{"sql_batch_rows",options.edges.sqlBatchRows},{"packed",bool(options.packed)},
        {"memory_first",config.value("memory_first",false)},{"background_flush",config.value("background_flush",true)},
        {"sqlite_memory_statistics",config.value("sqlite_memory_statistics",true)},{"sqlite_cache_bytes",options.store.sqliteCacheBytes},
        {"sqlite_pagecache_bulk_bytes",config.value("sqlite_pagecache_bulk_bytes",int64_t(0))},
        {"fibers",fibers},{"points",points},{"geometry_bytes",points*24},{"constraints",edges},{"constraints_imported",hasEdges},
        {"metadata_json_bytes",metadata},{"constraint_metadata_json_bytes",edgeMetadata},{"source_metadata_json_bytes",linked?edgeSource->metadataJson().size():afvMetadata.size()},
        {"source_read_seconds",sourceSeconds},{"native_import_seconds",nativeSeconds},{"seal_seconds",sealSeconds},
        {"constraint_source_read_seconds",edgeReadSeconds},{"constraint_import_seconds",edgeWriteSeconds},{"constraint_limit",edgeLimit},
        {"close_checkpoint_seconds",nullptr},{"close_checkpoint_timing","included in ingestion_seconds"},
        {"estimated_full_import_seconds",linked?(edges?imported.ingestionSeconds*edgeSource->constraintCount()/edges:0.):
            (fibers?imported.ingestionSeconds*source.fiberCount()/fibers:0.)},
        {"worker_timing_aggregation","source/native/seal durations are sums over workers; ingestion/total are wall time"},
        {"ingestion_seconds",imported.ingestionSeconds},{"workers_including_readback_seconds",imported.ingestionSeconds+validationSeconds},
        {"manifest_seconds",imported.manifestSeconds},{"persistence_seconds",imported.persistenceSeconds},
        {"publication_seconds",imported.manifestSeconds+imported.persistenceSeconds},{"validation_seconds",validationSeconds},{"readback_verified",verify},
        {"published_reopen_seconds",validationSeconds},
        {"total_seconds",seconds(start)},{"output_bytes",bytes},{"peak_rss_bytes",peakRssBytes()},{"per_shard",perShard}};
    if(linked){result["geometry_collection"]=options.dependencies.front().string();result["geometry_copied"]=false;}
    return result;
}
}

int main(int argc,char** argv)
{
    try {
        std::signal(SIGINT,cancelImport);std::signal(SIGTERM,cancelImport);
        if (argc!=2) throw std::invalid_argument("usage: vc_object_import_afv CONFIG.json");
        std::ifstream file(argv[1]); if (!file) throw std::runtime_error("cannot read config");
        const auto config=Json::parse(file);
        if(config.value("profile_sqlite_stacks",false) && !config.value("profile_sqlite_mutex",false))
            throw std::invalid_argument("profile_sqlite_stacks requires profile_sqlite_mutex");
        if(config.value("profile_sqlite_mutex",false))
            benchmark::sqlite_profile::install(config.value("profile_sqlite_stacks",false));
        // Benchmark-only opt-in. Do not alter the application's SQLite global
        // allocator configuration inside ObjectStore or a worker thread.
        if(!config.value("sqlite_memory_statistics",true) && sqlite3_config(SQLITE_CONFIG_MEMSTATUS,0)!=SQLITE_OK)
            throw std::runtime_error("SQLite was already initialized; cannot disable memory statistics");
        const auto pageBulk=config.value("sqlite_pagecache_bulk_bytes",int64_t(0));
        if(pageBulk<0 || pageBulk>int64_t(INT_MAX)*1024)
            throw std::invalid_argument("invalid SQLite page-cache allocation size");
        // A null buffer with negative N selects an independent initial bulk
        // allocation per cache, not the shared process-wide page buffer. This
        // avoids allocator-lock traffic without changing SQLite's page format.
        if(pageBulk && sqlite3_config(SQLITE_CONFIG_PAGECACHE,nullptr,0,-int((pageBulk+1023)/1024))!=SQLITE_OK)
            throw std::runtime_error("SQLite was already initialized; cannot configure page-cache allocation");
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
        if(!config.value("initial_batch",true))
            throw std::invalid_argument("initial_batch:false is removed; this command only uses the native batch importer");
        const auto result=importNative(config,input,output,coordinates,source,limit);
        std::cout<<result.dump(2)<<std::endl;
        if(config.contains("report")){std::ofstream report(config.at("report").get<std::string>());report<<result.dump(2)<<'\n';
            if(!report)throw std::runtime_error("report write failed");}
        return 0;
    } catch (const std::exception& error) { std::cerr<<"AFV import failed: "<<error.what()<<'\n'; return 1; }
}
