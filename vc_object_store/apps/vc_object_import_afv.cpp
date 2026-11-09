#include "vc_object_store/objects/AfvSource.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/Uuid.hpp"
#include "geometry/GeometryInternal.hpp"
#include "support/SqliteMutexProfile.hpp"
#include "support/ShardQueue.hpp"
#include "support/ProcessMetrics.hpp"
#include "imports/AfvRecords.hpp"
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

using namespace vc::objects;
using Json = nlohmann::json;
namespace {
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t cancelled=0;
void cancelImport(int) { cancelled=1; }
void checkCancelled(const std::atomic<bool>* failed=nullptr) { if(cancelled || (failed && failed->load()))throw std::runtime_error("import cancelled"); }
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
OpenOptions readOptions(const Json& config)
{
    OpenOptions options;options.readOnly=true;options.readerPoolSize=1;
    options.geometryCacheBytes=0;options.spatialCacheBytes=0;
    options.sqliteCacheBytes=config.value("sqlite_cache_bytes",options.sqliteCacheBytes);
    return options;
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
    using namespace vc::objects::imports;
    const bool linked=config.contains("geometry_collection"),hasEdges=config.contains("constraints");
    const auto integerOption=[&](const char* key,uint64_t fallback,uint64_t maximum,bool allowZero=false){
        const auto it=config.find(key);if(it==config.end())return fallback;
        if(!it->is_number_integer()||(!it->is_number_unsigned()&&it->get<int64_t>()<0))
            throw std::invalid_argument(std::string("invalid nonnegative integer option: ")+key);
        const auto value=it->get<uint64_t>();
        if((!allowZero&&!value)||value>maximum)throw std::invalid_argument(std::string("option outside supported range: ")+key);
        return value;
    };
    const auto sizeLimit=std::min<uint64_t>(std::numeric_limits<size_t>::max(),INT64_MAX);
    const int workers=int(integerOption("shards",1,256)),readers=int(integerOption("readers",std::min(workers,8),256));
    const auto edgeLimit=int64_t(integerOption("constraint_limit",0,INT64_MAX,true));
    const size_t batchSize=integerOption("batch_size",1024,1000000);
    const size_t batchBytes=integerOption("batch_bytes",8*1024*1024,sizeLimit);
    const size_t queueBytes=integerOption("queue_bytes",256*1024*1024,sizeLimit);
    if(linked&&!hasEdges)throw std::invalid_argument("geometry_collection requires constraints");
    const auto start=Clock::now();const auto sourceUuid=source.uuid();
    const auto fiberRanges=linked?std::vector<std::array<int64_t,2>>{}:source.idRanges(readers,limit);
    const int64_t fiberEnd=!linked&&limit&&!fiberRanges.empty()?fiberRanges.back()[1]:INT64_MAX;
    std::unique_ptr<AfcSource> edgeSource;
    std::vector<std::array<int64_t,2>> edgeRanges;
    if(hasEdges){
        edgeSource=std::make_unique<AfcSource>(config.at("constraints").get<std::string>(),sourceUuid,fiberEnd);
        edgeRanges=edgeSource->idRanges(readers,edgeLimit);
    }
    InitialImportOptions options;options.store.author=config.value("author",std::string("afv-import"));
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
    if(linked)options.dependencies={std::filesystem::absolute(config.at("geometry_collection").get<std::string>())};
    options.shards.clear();
    const int64_t reservation=INT64_MAX/workers;
    for(int i=0;i<workers;++i)options.shards.push_back({{int64_t(i)*reservation+1,int64_t(i+1)*reservation},
        {int64_t(i)*reservation+1,int64_t(i+1)*reservation}});
    StreamFailure failure;
    ShardQueues<InitialObjectRecord> objects(workers,std::max(size_t(1),queueBytes/size_t(workers)),failure);
    ShardQueues<InitialCollectionEdge> edges(workers,std::max(size_t(1),queueBytes/size_t(workers)),failure);
    // Readers batch across UUID destinations. Each reader holds at most
    // batchBytes plus one record; queue_bytes is each stream's queued budget
    // (plus one oversize batch per destination). SQLite consumers own no source.
    const auto dispatch=[&]<class T>(auto& queues,auto&& scan){
        std::vector<std::vector<T>> buckets(workers);std::vector<size_t> sizes(workers);size_t bytes=0;
        double waiting=0;
        const auto flush=[&](size_t i){if(buckets[i].empty())return;bytes-=sizes[i];const auto start=Clock::now();
            queues.push(i,std::move(buckets[i]),sizes[i]);waiting+=seconds(start);buckets[i].clear();sizes[i]=0;};
        scan([&](T row,size_t size,const std::string& uuid){
            checkCancelled();failure.check();const size_t i=uuidShardIndex(uuid,workers);
            buckets[i].push_back(std::move(row));sizes[i]+=size;bytes+=size;
            if(buckets[i].size()>=batchSize)flush(i);
            if(bytes>=batchBytes)for(size_t k=0;k<size_t(workers);++k)flush(k);
        });
        for(size_t i=0;i<size_t(workers);++i)flush(i);
        return waiting;
    };
    double sourceSeconds=0;
    std::atomic<double> fiberPreparation{0},edgePreparation{0};
    std::atomic<int64_t> sourceFibers{0},sourcePoints{0},sourceEdges{0};
    auto inputJob=std::async(std::launch::async,[&]{
        const auto inputStart=Clock::now();
        try {
            ObjectRecord metadata;metadata.uuid=linked?util::nameUuid(sourceUuid,"constraint-source"):sourceUuid;
            metadata.kind="collection";metadata.name=linked?"constraint-source":input.stem().string();
            Json fields=linked?Json::object():Json{{"coordinates",coordinates},{"afv",{{"metadata",Json::parse(source.metadataJson())}}}};
            if(edgeSource)fields["afc"]=Json::parse(edgeSource->metadataJson());
            metadata.meta=fields.dump();const auto shard=uuidShardIndex(metadata.uuid,workers);
            objects.push(shard,{{std::move(metadata),{}, {}}},fields.dump().size()+sizeof(InitialObjectRecord));
            const auto parallel=[&](const auto& ranges,auto&& read){
                std::vector<std::future<void>> jobs;
                try {for(const auto& range:ranges)jobs.push_back(std::async(std::launch::async,[&,range]{
                    try {read(range);}catch(...){failure.fail(std::current_exception());throw;}
                }));}catch(...){failure.fail(std::current_exception());}
                for(auto& job:jobs)try{job.get();}catch(...){failure.fail(std::current_exception());}
                failure.check();
            };
            if(!linked)parallel(fiberRanges,[&](const auto& range){
                const auto preparing=Clock::now();
                AfvSource part(input,range[0],range[1]);if(part.uuid()!=sourceUuid)throw std::runtime_error("AFV source identity changed");
                int64_t n=0,points=0;
                const auto waiting=dispatch.template operator()<InitialObjectRecord>(objects,[&](auto&& emit){
                    AfvFiber fiber;while(part.next(fiber)){
                        ++n;points+=fiber.detail.linePointCount;
                        auto r=afvObjectRecord(fiber,sourceUuid,coordinates);auto uuid=r.uuid;
                        const size_t bytes=fiber.xyz.size()+r.meta.size()+r.name.size()+sizeof(InitialObjectRecord);
                        auto g=afvGeometry(fiber);emit(InitialObjectRecord{std::move(r),{},std::move(g)},bytes,uuid);
                    }
                });
                sourceFibers.fetch_add(n);sourcePoints.fetch_add(points);
                fiberPreparation.fetch_add(seconds(preparing)-waiting);
            });
            if(!linked&&(sourceFibers.load()!=(limit?std::min(limit,source.fiberCount()):source.fiberCount())||
                         (!limit&&sourcePoints.load()!=source.pointCount())))
                throw std::runtime_error("initial fiber aggregate totals differ from source");
            objects.close();
            // Start reading edges after sources finished producing objects, not
            // after shard commits: no global database/object barrier is needed.
            if(hasEdges)parallel(edgeRanges,[&](const auto& range){
                const auto preparing=Clock::now();
                AfcSource part(config.at("constraints").get<std::string>(),sourceUuid,fiberEnd,range[0],range[1]);
                int64_t n=0;
                const auto waiting=dispatch.template operator()<InitialCollectionEdge>(edges,[&](auto&& emit){
                    EdgeRecord e;while(part.next(e)){
                        ++n;
                        const auto uuid=e.uuid;const size_t bytes=e.meta.size()+sizeof(InitialCollectionEdge);
                        ObjectKey a{"",e.aUuid},b{"",e.bUuid};
                        emit(InitialCollectionEdge{std::move(a),std::move(b),std::move(e)},bytes,uuid);
                    }
                });
                sourceEdges.fetch_add(n);
                edgePreparation.fetch_add(seconds(preparing)-waiting);
            });
            if(hasEdges&&fiberEnd==INT64_MAX&&sourceEdges.load()!=(edgeLimit?std::min(edgeLimit,edgeSource->constraintCount()):edgeSource->constraintCount()))
                throw std::runtime_error("initial constraint aggregate totals differ from source");
            edges.close();sourceSeconds=seconds(inputStart);
        }catch(...){failure.fail(std::current_exception());throw;}
    });
    struct Counts {int64_t fibers=0,points=0,edges=0,metadata=0,edgeMetadata=0;};
    std::vector<Counts> counts(workers);std::vector<detail::GeometryWork> profiles(workers);
    std::vector<Json> mutexProfiles(workers);
    std::atomic<int64_t> completedFibers{0},completedEdges{0};
    std::mutex progressMutex;auto lastProgress=start;
    const auto progress=[&]{
        std::lock_guard lock(progressMutex);if(seconds(lastProgress)<5)return;
        std::cout<<Json({{"stage","uuid_shard_import"},{"fibers",completedFibers.load()},
            {"constraints",completedEdges.load()},{"shards",workers},{"elapsed_seconds",seconds(start)}}).dump()<<std::endl;
        lastProgress=Clock::now();
    };
    InitialImportResult imported;
    try {
        imported=importInitialCollection(output,options,[&](size_t i,InitialShardImport& shard){
            try {
                std::optional<detail::GeometryWorkScope> profile;
                if(config.value("profile",false))profile.emplace(profiles[i]);
                std::vector<InitialObjectRecord> rows;
                while(objects.pop(i,rows,[&]{checkCancelled();shard.checkCancelled();})){checkCancelled();
                    for(const auto& row:rows)if(row.record.fiber){
                        ++counts[i].fibers;counts[i].points+=row.record.fiber->linePointCount;
                        counts[i].metadata+=row.record.meta.size();
                    }
                    shard.appendObjects(rows);
                    // Counters are per batch; never read another worker's Counts.
                    int64_t n=0;for(const auto& row:rows)n+=bool(row.record.fiber);
                    completedFibers.fetch_add(n);progress();
                }
            }catch(...){failure.fail(std::current_exception());throw;}
        },[&](size_t i,InitialShardImport& shard){
            try {
                std::optional<detail::GeometryWorkScope> profile;
                if(config.value("profile",false))profile.emplace(profiles[i]);
                if(hasEdges)shard.importEdges([&](std::vector<InitialCollectionEdge>& rows){
                    checkCancelled();if(!edges.pop(i,rows,[&]{checkCancelled();shard.checkCancelled();}))return false;
                    for(const auto& row:rows)counts[i].edgeMetadata+=row.record.meta.size();
                    counts[i].edges+=int64_t(rows.size());completedEdges.fetch_add(int64_t(rows.size()));progress();return true;
                });
                if(config.value("profile_sqlite_mutex",false))mutexProfiles[i]=benchmark::sqlite_profile::snapshot();
            }catch(...){failure.fail(std::current_exception());throw;}
        });
        inputJob.get();
    }catch(...){failure.fail(std::current_exception());try{inputJob.get();}catch(...){}throw;}
    int64_t fibers=0,points=0,nedges=0,metadataBytes=0,edgeMetadataBytes=0;Json perShard=Json::array();
    double objectSeconds=0,edgeSeconds=0,sealSeconds=0;
    for(size_t i=0;i<counts.size();++i){const auto& c=counts[i];const auto& t=imported.shards[i];
        fibers+=c.fibers;points+=c.points;nedges+=c.edges;
        metadataBytes+=c.metadata;edgeMetadataBytes+=c.edgeMetadata;
        objectSeconds+=t.objectSeconds;edgeSeconds+=t.edgeSeconds;sealSeconds+=t.sealCommitSeconds;
        perShard.push_back({{"shard",i},{"fibers",c.fibers},{"points",c.points},{"constraints",c.edges},
            {"native_import_seconds",t.objectSeconds+t.edgeSeconds},{"creation_seconds",t.creationSeconds},
            {"seal_commit_seconds",t.sealCommitSeconds},{"phase_nanoseconds",profiles[i].phaseNanoseconds}});
        if(config.value("profile_sqlite_mutex",false))perShard.back()["sqlite_mutex_profile"]=mutexProfiles[i];
    }
    if(!linked&&(fibers!=(limit?std::min(limit,source.fiberCount()):source.fiberCount())||(!limit&&points!=source.pointCount())))
        throw std::runtime_error("initial fiber aggregate totals differ from source");
    if(hasEdges && fiberEnd==INT64_MAX && nedges!=(edgeLimit?std::min(edgeLimit,edgeSource->constraintCount()):edgeSource->constraintCount()))
        throw std::runtime_error("initial constraint aggregate totals differ from source");
    const double pipelineSeconds=seconds(start);
    const bool verify=config.value("verify_readback",false);double validationSeconds=0;
    if(verify){
        const auto vstart=Clock::now();auto joined=ObjectCollection::open(imported.path,readOptions(config));auto txn=joined.read();
        if(!linked)for(const auto& range:fiberRanges){
            AfvSource part(input,range[0],range[1]);AfvFiber fiber;
            while(part.next(fiber)){
                checkCancelled();
                const ObjectKey key{"",util::nameUuid(sourceUuid,"fiber:"+std::to_string(fiber.id))};
                const auto value=joined.object(txn,key);const auto g=joined.currentGeometry(txn,key);
                verifyFiberData(value?&value->record:nullptr,bool(g),fiber,[&](const char* name){return joined.readGeometryArray(txn,key,name);});
            }
        }
        int64_t verifiedEdges=0;
        for(const auto& range:edgeRanges){
            AfcSource part(config.at("constraints").get<std::string>(),sourceUuid,fiberEnd,range[0],range[1]);EdgeRecord e;
            while(part.next(e)){
                checkCancelled();
                const auto actual=joined.edge(txn,{"",e.uuid});if(!actual)throw std::runtime_error("published relation missing");
                if(actual->a.objectUuid==e.bUuid){std::swap(e.aUuid,e.bUuid);std::swap(e.aAnchor,e.bAnchor);}
                if(actual->a.objectUuid!=e.aUuid||actual->b.objectUuid!=e.bUuid||actual->record.aAnchor!=e.aAnchor||
                   actual->record.bAnchor!=e.bAnchor||actual->record.kind!=e.kind||Json::parse(actual->record.meta)!=Json::parse(e.meta))
                    throw std::runtime_error("published constraint differs from source");
                for(const auto& endpoint:{actual->a,actual->b})if(!joined.currentGeometry(txn,endpoint))
                    throw std::runtime_error("constraint endpoint geometry missing");
                ++verifiedEdges;
            }
        }
        if(verifiedEdges!=nedges)throw std::runtime_error("constraint count mismatch");
        validationSeconds=seconds(vstart);
    }
    uintmax_t bytes=0;
    if(std::filesystem::is_directory(output))for(const auto& entry:std::filesystem::recursive_directory_iterator(output))
        {if(entry.is_regular_file())bytes+=entry.file_size();}
    else bytes=std::filesystem::file_size(output);
    Json result={{"source",input.string()},{"output",output.string()},{"store_uuid",imported.storeUuid},
        {"stage","complete"},{"mode",linked?"relations":hasEdges?"combined":"geometry"},
        {"source_uuid",sourceUuid},{"initial_batch",true},{"uuid_hash_routing",true},{"shards",workers},{"readers",readers},
        {"batch_size",batchSize},{"batch_bytes",batchBytes},{"queue_bytes",queueBytes},{"packed",bool(options.packed)},
        {"fibers",fibers},{"points",points},{"constraints",nedges},{"constraints_imported",hasEdges},{"source_pipeline_seconds",sourceSeconds},
        {"geometry_bytes",points*24},{"metadata_json_bytes",metadataBytes},{"constraint_metadata_json_bytes",edgeMetadataBytes},
        {"source_metadata_json_bytes",linked?edgeSource->metadataJson().size():source.metadataJson().size()},
        {"source_read_seconds",fiberPreparation.load()},{"constraint_source_read_seconds",edgePreparation.load()},
        {"source_read_timing","reader-worker preparation sum (decode/UUID/batching), excludes queue push time"},
        {"native_import_seconds",objectSeconds},{"constraint_import_seconds",edgeSeconds},{"seal_seconds",sealSeconds},
        {"constraint_limit",edgeLimit},{"sql_batch_rows",options.edges.sqlBatchRows},{"sqlite_cache_bytes",options.store.sqliteCacheBytes},
        {"memory_first",config.value("memory_first",false)},{"background_flush",config.value("background_flush",true)},
        {"sqlite_memory_statistics",config.value("sqlite_memory_statistics",true)},
        {"sqlite_pagecache_bulk_bytes",config.value("sqlite_pagecache_bulk_bytes",int64_t(0))},
        {"source_pipeline_timing","wall time including queue backpressure; overlaps native ingestion"},
        {"ingestion_seconds",imported.ingestionSeconds},{"manifest_seconds",imported.manifestSeconds},
        {"persistence_seconds",imported.persistenceSeconds},{"validation_seconds",validationSeconds},{"readback_verified",verify},
        {"total_seconds",seconds(start)},{"output_bytes",bytes},{"peak_rss_bytes",benchmark::peakRssBytes()},{"per_shard",perShard}};
    if(linked){result["geometry_collection"]=options.dependencies.front().string();result["geometry_copied"]=false;}
    if(!linked&&hasEdges&&edgeLimit){
        result["estimated_full_import_seconds"]=nullptr;
        result["estimate_basis"]="unavailable for combined import with constraint_limit: fiber and relation sampling fractions differ; do not scale both phases by fiber count";
    }else{
        result["estimated_full_import_seconds"]=linked?(nedges?pipelineSeconds*edgeSource->constraintCount()/nedges:0.):
            (fibers?pipelineSeconds*source.fiberCount()/fibers:0.);
        result["estimate_basis"]="linear record-count projection, source preparation included; not a full-volume measurement";
    }
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
        const auto sourceStart=Clock::now();AfvSource source(input);
        const auto sourceOpenSeconds=seconds(sourceStart);
        if(!config.value("initial_batch",true))
            throw std::invalid_argument("initial_batch:false is removed; this command only uses the native batch importer");
        auto result=importNative(config,input,output,coordinates,source,limit);
        result["source_open_seconds"]=sourceOpenSeconds;
        result["end_to_end_seconds"]=seconds(sourceStart);
        std::cout<<result.dump(2)<<std::endl;
        if(config.contains("report")){std::ofstream report(config.at("report").get<std::string>());report<<result.dump(2)<<'\n';
            if(!report)throw std::runtime_error("report write failed");}
        return 0;
    } catch (const std::exception& error) { std::cerr<<"AFV import failed: "<<error.what()<<'\n'; return 1; }
}
