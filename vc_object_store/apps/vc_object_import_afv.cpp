#include "vc_object_store/objects/AfvSource.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/objects/ObjectStoreSession.hpp"
#include "vc_object_store/objects/StoreReader.hpp"
#include <vc_reader/native/atomic_file.hpp>
#include "vc_object_store/Uuid.hpp"
#include "geometry/GeometryInternal.hpp"
#include "support/SqliteMutexProfile.hpp"
#include "vc_object_store/objects/imports/ParallelImportStream.hpp"
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
void protectReport(const Json& config, const std::filesystem::path& configPath,
                   const std::filesystem::path& output, const std::filesystem::path& report)
{
    namespace fs = std::filesystem;
    const auto target = fs::weakly_canonical(fs::absolute(report));
    const auto destination = fs::weakly_canonical(output);
    for (auto parent = target; !parent.empty(); parent = parent.parent_path()) {
        if (parent == destination) throw std::invalid_argument("report overlaps import output");
        if (parent == parent.root_path()) break;
    }
    std::vector<fs::path> inputs{configPath, config.at("source").get<std::string>()};
    if (config.contains("constraints")) inputs.emplace_back(config.at("constraints").get<std::string>());
    for (const auto& input : inputs)
        if (target == fs::weakly_canonical(fs::absolute(input)) ||
            (fs::exists(target) && fs::exists(input) && fs::equivalent(target, input)))
            throw std::invalid_argument("report overlaps an import input");
}
void protectStoreReport(const std::filesystem::path& store, const std::filesystem::path& report)
{
    ObjectStoreSession session;
    OpenOptions options; options.readOnly = true;
    const auto attachment = session.attach(store, options);
    session.readStore(attachment, [&](const StoreReader& reader, const auto&) {
        reader.validateExportDestination(report);
    });
}
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
        record->fiber->controlPointCount!=0 || record->fiber->linePointCount!=fiber.detail.linePointCount ||
        read("line_points")!=fiber.xyz || !read("control_points").empty() || !read("control_point_ids").empty())
        throw std::runtime_error("fiber metadata or geometry readback differs from source");
}
Json importNative(const Json& config,const std::filesystem::path& input,const std::filesystem::path& output,
                  const Json& coordinates,AfvSource& source,int64_t limit)
{
    using namespace vc::objects::imports;
    const bool storeReference=config.contains("geometry_collection"),hasEdges=config.contains("constraints");
    std::optional<std::array<double,6>> sourceBounds;
    if(config.contains("source_bounds_xyz")) {
        sourceBounds=config.at("source_bounds_xyz").get<std::array<double,6>>();
        if(storeReference||hasEdges)throw std::invalid_argument("Spatial source selection is for fiber imports only");
        for(int k=0;k<3;++k)if(!std::isfinite((*sourceBounds)[k])||!std::isfinite((*sourceBounds)[k+3])||
            !((*sourceBounds)[k]<(*sourceBounds)[k+3]))throw std::invalid_argument("Invalid source selection bounds");
    }
    const auto selected=[&](const AfvFiber& fiber) {
        if(!sourceBounds)return true;
        for(int k=0;k<3;++k)if((*fiber.detail.bbox)[k]>(*sourceBounds)[k+3] ||
            (*fiber.detail.bbox)[k+3]<(*sourceBounds)[k])return false;
        return true;
    };
    const auto integerOption=[&](const char* key,uint64_t fallback,uint64_t maximum,bool allowZero=false){
        const auto it=config.find(key);if(it==config.end())return fallback;
        if(!it->is_number_integer()||(!it->is_number_unsigned()&&it->get<int64_t>()<0))
            throw std::invalid_argument(std::string("invalid nonnegative integer option: ")+key);
        const auto value=it->get<uint64_t>();
        if((!allowZero&&!value)||value>maximum)throw std::invalid_argument(std::string("option outside supported range: ")+key);
        return value;
    };
    const auto sizeLimit=std::min<uint64_t>(std::numeric_limits<size_t>::max(),INT64_MAX);
    const int workers=int(integerOption("shards",defaultParallelImportWorkers,256)),
        readers=int(integerOption("readers",std::min(size_t(workers),defaultParallelImportWorkers),256));
    const auto edgeLimit=int64_t(integerOption("constraint_limit",0,INT64_MAX,true));
    const size_t batchSize=integerOption("batch_size",1024,1000000);
    const size_t batchBytes=integerOption("batch_bytes",8*1024*1024,sizeLimit);
    const size_t queueBytes=integerOption("queue_bytes",256*1024*1024,sizeLimit);
    const size_t sourceCacheBytes=integerOption("source_cache_bytes",128*1024*1024,sizeLimit);
    const size_t sourceMmapBytes=integerOption("source_mmap_bytes",0,sizeLimit,true);
    if(storeReference&&!hasEdges)throw std::invalid_argument("geometry_collection requires constraints");
    const auto start=Clock::now();const auto sourceUuid=source.uuid();
    source.setMmapBytes(sourceMmapBytes);
    const auto fiberRanges=storeReference?std::vector<std::array<int64_t,2>>{}:source.workRanges(
        integerOption("source_chunk_points",65536,sizeLimit),integerOption("source_chunk_fibers",1024,sizeLimit),limit);
    const int64_t fiberEnd=!storeReference&&limit&&!fiberRanges.empty()?fiberRanges.back()[1]:INT64_MAX;
    std::unique_ptr<AfcSource> edgeSource;
    std::vector<std::array<int64_t,2>> edgeRanges;
    if(hasEdges){
        edgeSource=std::make_unique<AfcSource>(config.at("constraints").get<std::string>(),sourceUuid,fiberEnd);
        edgeSource->setMmapBytes(sourceMmapBytes);
        const auto estimated=edgeLimit?edgeLimit:edgeSource->constraintCount();
        const auto chunkRows=integerOption("source_chunk_edges",2048,sizeLimit);
        edgeRanges=edgeSource->idRanges(std::max(size_t(readers),size_t(estimated)/chunkRows+1),edgeLimit);
    }
    InitialImportOptions options;options.store.author=config.value("author",std::string("afv-import"));
    options.store.sqliteCacheBytes=integerOption("sqlite_cache_bytes",512*1024*1024,sizeLimit,true);
    options.store.readerPoolSize=1;options.store.geometryCacheBytes=0;options.store.spatialCacheBytes=0;
    options.edges.deferIndexes=config.value("defer_indexes",true);
    options.edges.sqlBatchRows=config.value("sql_batch_rows",size_t(1));
    if(config.value("packed",false)){
        options.packed=PackedStoreOptions{};options.packed->disposableBulk=true;
        options.packed->residentBytes=config.value("resident_bytes",size_t(256*1024*1024));
        options.packed->memoryFirst=config.value("memory_first",false);
        options.packed->backgroundFlush=config.value("background_flush",true);
    }
    if(storeReference)options.dependencies={std::filesystem::absolute(config.at("geometry_collection").get<std::string>())};
    options.shards.clear();
    const int64_t reservation=INT64_MAX/workers;
    for(int i=0;i<workers;++i)options.shards.push_back({{int64_t(i)*reservation+1,int64_t(i+1)*reservation}});
    std::atomic<int64_t> sourceFibers{0},sourcePoints{0},sourceEdges{0};
    std::vector<std::unique_ptr<AfvSource>> fiberReaders(readers);
    std::vector<std::unique_ptr<AfcSource>> edgeReaders(readers);
    std::vector<int64_t> readerPoints(readers),readerFibers(readers),readerEdges(readers);
    std::vector<Json> readerMutexProfiles(readers);
    std::vector<detail::GeometryWork> sourceProfiles(readers);
    std::vector<size_t> fiberMmap(readers),edgeMmap(readers);
    // Join/cancel readers before destroying the cursor/stat buffers they use.
    ParallelImportStream stream({size_t(readers),size_t(workers),batchSize,batchBytes,queueBytes});
    if(!storeReference){
        FiberSourceInfo info;info.uuid=sourceUuid;info.fiberCount=source.fiberCount();info.pointCount=source.pointCount();
        if(config.contains("source_sha256"))info.sha256=config.at("source_sha256").get<std::string>();
        options.extensions.push_back(fiberSourceExtension(info));
    }
    if(hasEdges){
        std::vector<ExtensionSource> pins;
        if(storeReference){auto base=ObjectCollection::open(config.at("geometry_collection").get<std::string>(),readOptions(config));auto tx=base.read();
            const auto state=base.provisioning(tx);pins.push_back({base.uuid(tx),state.generation,state.commitSequence});}
        FiberConstraintSourceInfo info;info.fiberSourceUuid=sourceUuid;info.constraintCount=edgeSource->constraintCount();
        if(config.contains("constraints_sha256"))info.sha256=config.at("constraints_sha256").get<std::string>();
        options.extensions.push_back(fiberConstraintExtension(info,pins));
    }
    stream.start(fiberRanges.size(),edgeRanges.size(),[&](size_t worker,size_t task,ObjectEmitter& emit){
        checkCancelled();
        const auto range=fiberRanges[task];auto& part=fiberReaders[worker];
        if(!part){part=std::make_unique<AfvSource>(input,0,0);part->setCacheBytes(sourceCacheBytes);
            fiberMmap[worker]=part->setMmapBytes(sourceMmapBytes);
            if(part->uuid()!=sourceUuid)throw std::runtime_error("AFV source identity changed");}
        part->selectRange(range[0],range[1]);AfvFiber fiber;int64_t n=0,points=0;
        while(part->next(fiber)) {
            if(!selected(fiber))continue;
            checkCancelled();++n;points+=fiber.detail.linePointCount;
            const auto xyzBytes=fiber.xyz.size();auto row=afvInitialRecord(fiber,sourceUuid,coordinates);const auto uuid=row.record.uuid;
            const auto bytes=xyzBytes+row.record.meta.size()+row.record.name.size()+sizeof(InitialObjectRecord)+extensionPayloadBytes(row.extensions);
            emit.emit(ImportObject{std::move(row)},bytes,uuid);
        }
        readerFibers[worker]+=n;readerPoints[worker]+=points;sourceFibers.fetch_add(n);sourcePoints.fetch_add(points);
    },[&](size_t worker,size_t task,EdgeEmitter& emit){
        std::optional<detail::GeometryWorkScope> profile;
        if(config.value("profile_source",false))profile.emplace(sourceProfiles[worker]);
        checkCancelled();auto& part=edgeReaders[worker];const auto range=edgeRanges[task];
        if(!part){part=std::make_unique<AfcSource>(config.at("constraints").get<std::string>(),sourceUuid,fiberEnd);part->setCacheBytes(sourceCacheBytes);
            edgeMmap[worker]=part->setMmapBytes(sourceMmapBytes);}
        part->selectRange(range[0],range[1]);AfcConstraint c;int64_t n=0;
        while(part->next(c)){
            checkCancelled();++n;auto row=afcInitialRecord(std::move(c));const auto uuid=row.record.uuid;
            const auto bytes=row.record.meta.size()+sizeof(InitialCollectionEdge)+extensionPayloadBytes(row.extensions);
            emit.emit(std::move(row),bytes,uuid);
        }
        readerEdges[worker]+=n;sourceEdges.fetch_add(n);
    },[&](size_t worker){
        if(config.value("profile_sqlite_mutex",false))readerMutexProfiles[worker]=benchmark::sqlite_profile::snapshot();
    });
    struct Counts {int64_t fibers=0,points=0,edges=0,metadata=0,edgeMetadata=0;};
    std::vector<Counts> counts(workers);std::vector<detail::GeometryWork> profiles(workers);
    std::vector<Json> mutexProfiles(workers);
    std::atomic<int64_t> completedFibers{0},completedEdges{0};
    std::atomic<int64_t> nextProgress{5};
    const auto progress=[&]{
        const auto now=int64_t(seconds(start));auto next=nextProgress.load(std::memory_order_relaxed);
        if(now<next||!nextProgress.compare_exchange_strong(next,now+5))return;
        std::cout<<Json({{"stage","uuid_shard_import"},{"fibers",completedFibers.load()},
            {"constraints",completedEdges.load()},{"shards",workers},{"elapsed_seconds",seconds(start)}}).dump()<<std::endl;
    };
    int64_t fibers=0,points=0,nedges=0;
    options.beforePublish=[&]{
        stream.finish();
        checkCancelled();
        // All writers have joined. Check their existing counters while every
        // destination is still private; no source scan or store readback.
        for(const auto& c:counts){fibers+=c.fibers;points+=c.points;nedges+=c.edges;}
        if(!storeReference && sourceBounds && (fibers!=sourceFibers.load() || points!=sourcePoints.load()))
            throw std::runtime_error("Selected fiber aggregate totals differ from producer");
        if(!storeReference&&!sourceBounds&&(fibers!=(limit?std::min(limit,source.fiberCount()):source.fiberCount())||(!limit&&points!=source.pointCount())))
            throw std::runtime_error("initial fiber aggregate totals differ from source");
        if(hasEdges && fiberEnd==INT64_MAX && nedges!=(edgeLimit?std::min(edgeLimit,edgeSource->constraintCount()):edgeSource->constraintCount()))
            throw std::runtime_error("initial constraint aggregate totals differ from source");
    };
    InitialImportResult imported;
    try {
        imported=importInitialCollection(output,options,[&](size_t i,InitialShardImport& shard){
            try {
                std::optional<detail::GeometryWorkScope> profile;
                if(config.value("profile",false))profile.emplace(profiles[i]);
                std::vector<ImportObject> rows;std::vector<const InitialObjectRecord*> refs;
                while(stream.popObjects(i,rows,[&]{checkCancelled();shard.checkCancelled();})){checkCancelled();refs.clear();
                    int64_t n=0;
                    for(const auto& input:rows){const auto& row=input.get();refs.push_back(&row);if(row.record.fiber){
                        ++counts[i].fibers;counts[i].points+=row.record.fiber->linePointCount;
                        counts[i].metadata+=row.record.meta.size();++n;
                    }}
                    shard.appendObjectRefs(refs);
                    // Counters are per batch; never read another worker's Counts.
                    completedFibers.fetch_add(n);progress();
                }
            }catch(...){stream.fail(std::current_exception());throw;}
        },[&](size_t i,InitialShardImport& shard){
            try {
                std::optional<detail::GeometryWorkScope> profile;
                if(config.value("profile",false))profile.emplace(profiles[i]);
                if(hasEdges)shard.importEdges([&](std::vector<InitialCollectionEdge>& rows){
                    checkCancelled();if(!stream.popEdges(i,rows,[&]{checkCancelled();shard.checkCancelled();}))return false;
                    for(const auto& row:rows)counts[i].edgeMetadata+=row.record.meta.size();
                    counts[i].edges+=int64_t(rows.size());completedEdges.fetch_add(int64_t(rows.size()));progress();return true;
                });
                if(config.value("profile_sqlite_mutex",false))mutexProfiles[i]=benchmark::sqlite_profile::snapshot();
            }catch(...){stream.fail(std::current_exception());throw;}
        });
    }catch(...){stream.fail(std::current_exception());try{stream.finish();}catch(...){}throw;}
    double sourceSeconds=0,fiberPreparation=0,edgePreparation=0;Json perReader=Json::array();
    for(size_t i=0;i<stream.readers.size();++i){const auto& s=stream.readers[i];fiberPreparation+=s.objectSeconds;edgePreparation+=s.edgeSeconds;
        sourceSeconds=std::max(sourceSeconds,s.objectSeconds+s.edgeSeconds+s.queueWaitSeconds);
        perReader.push_back({{"reader",i},{"fibers",readerFibers[i]},{"points",readerPoints[i]},{"constraints",readerEdges[i]},
            {"fiber_mmap_bytes",fiberMmap[i]},{"edge_mmap_bytes",edgeMmap[i]},
            {"phase_nanoseconds",sourceProfiles[i].phaseNanoseconds},
            {"object_tasks",s.objectTasks},{"edge_tasks",s.edgeTasks},{"object_seconds",s.objectSeconds},{"edge_seconds",s.edgeSeconds},{"queue_wait_seconds",s.queueWaitSeconds}});
        if(config.value("profile_sqlite_mutex",false))perReader.back()["sqlite_mutex_profile"]=readerMutexProfiles[i];}
    int64_t metadataBytes=0,edgeMetadataBytes=0;Json perShard=Json::array();
    double objectSeconds=0,edgeSeconds=0,sealSeconds=0;
    for(size_t i=0;i<counts.size();++i){const auto& c=counts[i];const auto& t=imported.shards[i];
        metadataBytes+=c.metadata;edgeMetadataBytes+=c.edgeMetadata;
        objectSeconds+=t.objectSeconds;edgeSeconds+=t.edgeSeconds;sealSeconds+=t.sealCommitSeconds;
        perShard.push_back({{"shard",i},{"fibers",c.fibers},{"points",c.points},{"constraints",c.edges},
            {"native_import_seconds",t.objectSeconds+t.edgeSeconds},{"creation_seconds",t.creationSeconds},
            {"seal_commit_seconds",t.sealCommitSeconds},{"phase_nanoseconds",profiles[i].phaseNanoseconds}});
        if(config.value("profile_sqlite_mutex",false))perShard.back()["sqlite_mutex_profile"]=mutexProfiles[i];
    }
    const double pipelineSeconds=seconds(start);
    const bool verify=config.value("verify_readback",false);double validationSeconds=0;
    if(verify){
        const auto vstart=Clock::now();auto joined=ObjectCollection::open(imported.path,readOptions(config));auto txn=joined.read();
        if(!storeReference)for(const auto& range:fiberRanges){
            AfvSource part(input,range[0],range[1]);AfvFiber fiber;
            while(part.next(fiber)){
                checkCancelled();
                if(!selected(fiber))continue;
                const ObjectKey key{"",util::nameUuid(sourceUuid,"fiber:"+std::to_string(fiber.id))};
                const auto value=joined.object(txn,key);const auto g=joined.currentGeometry(txn,key);
                verifyFiberData(value?&value->record:nullptr,bool(g),fiber,[&](const char* name){return joined.readGeometryArray(txn,key,name);});
                const auto expected=fiberSourceRow(key.objectUuid,fiber.id,fiber.annotation);size_t found=0;
                const ExtensionValue prefix[]={extensionUuid(key.objectUuid)};
                joined.scanExtensionRows(txn,kFiberSourceTable,[&](const auto& row){++found;
                    if(row!=expected.values)throw std::runtime_error("fiber source extension differs");return true;},prefix);
                if(found!=1)throw std::runtime_error("fiber source extension missing");
            }
        }
        int64_t verifiedEdges=0;
        for(const auto& range:edgeRanges){
            AfcSource part(config.at("constraints").get<std::string>(),sourceUuid,fiberEnd,range[0],range[1]);AfcConstraint c;
            while(part.next(c)){
                auto& e=c.edge;
                checkCancelled();
                const auto actual=joined.edge(txn,{"",e.uuid});if(!actual)throw std::runtime_error("published relation missing");
                if(actual->a.objectUuid==e.bUuid){std::swap(e.aUuid,e.bUuid);std::swap(e.aAnchor,e.bAnchor);}
                if(actual->a.objectUuid!=e.aUuid||actual->b.objectUuid!=e.bUuid||actual->record.aAnchor!=e.aAnchor||
                   actual->record.bAnchor!=e.bAnchor||actual->record.kind!=e.kind||Json::parse(actual->record.meta)!=Json::parse(e.meta))
                    throw std::runtime_error("published constraint differs from source");
                const ExtensionValue prefix[]={extensionUuid(e.uuid)};size_t found=0;
                joined.scanExtensionRows(txn,kFiberConstraintTable,[&](const auto& row){++found;
                    if(row!=c.measurement.values)throw std::runtime_error("constraint extension differs");return true;},prefix);
                if(found!=1)throw std::runtime_error("constraint extension missing");
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
        {"stage","complete"},{"mode",storeReference?"relations":hasEdges?"combined":"geometry"},
        {"source_uuid",sourceUuid},{"initial_batch",true},{"uuid_hash_routing",true},{"shards",workers},{"readers",readers},
        {"batch_size",batchSize},{"batch_bytes",batchBytes},{"queue_bytes",queueBytes},{"packed",bool(options.packed)},
        {"fibers",fibers},{"points",points},{"constraints",nedges},{"constraints_imported",hasEdges},{"source_pipeline_seconds",sourceSeconds},
        {"geometry_bytes",points*24},{"metadata_json_bytes",metadataBytes},{"constraint_metadata_json_bytes",edgeMetadataBytes},
        {"fiber_source_extension_rows",fibers},{"constraint_extension_rows",nedges},
        {"source_metadata_json_bytes",storeReference?edgeSource->metadataJson().size():source.metadataJson().size()},
        {"source_read_seconds",fiberPreparation},{"constraint_source_read_seconds",edgePreparation},{"per_reader",perReader},
        {"source_read_timing","reader-worker preparation sum (decode/UUID/batching), excludes queue push time"},
        {"native_import_seconds",objectSeconds},{"constraint_import_seconds",edgeSeconds},{"seal_seconds",sealSeconds},
        {"constraint_limit",edgeLimit},{"sql_batch_rows",options.edges.sqlBatchRows},{"sqlite_cache_bytes",options.store.sqliteCacheBytes},
        {"source_cache_bytes",sourceCacheBytes},
        {"source_mmap_bytes",sourceMmapBytes},
        {"memory_first",config.value("memory_first",false)},{"background_flush",config.value("background_flush",true)},
        {"sqlite_memory_statistics",config.value("sqlite_memory_statistics",true)},
        {"sqlite_pagecache_bulk_bytes",config.value("sqlite_pagecache_bulk_bytes",int64_t(0))},
        {"source_pipeline_timing","maximum reader elapsed time including queue backpressure; overlaps native ingestion"},
        {"ingestion_seconds",imported.ingestionSeconds},{"manifest_seconds",imported.manifestSeconds},
        {"persistence_seconds",imported.persistenceSeconds},{"validation_seconds",validationSeconds},{"readback_verified",verify},
        {"total_seconds",seconds(start)},{"output_bytes",bytes},{"peak_rss_bytes",benchmark::peakRssBytes()},{"per_shard",perShard}};
    if(storeReference){result["geometry_collection"]=options.dependencies.front().string();result["geometry_copied"]=false;}
    if(sourceBounds) {
        result["source_bounds_xyz"]=*sourceBounds;
        result["estimated_full_import_seconds"]=nullptr;
        result["estimate_basis"]="spatial subset; source scan includes omitted fibers, so record-count extrapolation is not valid";
    }else if(!storeReference&&hasEdges&&edgeLimit){
        result["estimated_full_import_seconds"]=nullptr;
        result["estimate_basis"]="unavailable for combined import with constraint_limit: fiber and relation sampling fractions differ; do not scale both phases by fiber count";
    }else{
        result["estimated_full_import_seconds"]=storeReference?(nedges?pipelineSeconds*edgeSource->constraintCount()/nedges:0.):
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
            benchmark::sqlite_profile::install(config.value("profile_sqlite_stacks",false),config.value("profile_stack_interval",uint64_t(65536)));
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
        const auto report = config.contains("report")
            ? std::optional<std::filesystem::path>(config.at("report").get<std::string>()) : std::nullopt;
        if (report) {
            protectReport(config, argv[1], output, *report);
            if (config.contains("geometry_collection"))
                protectStoreReport(config.at("geometry_collection").get<std::string>(), *report);
        }
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
        if (report) {
            protectStoreReport(std::filesystem::is_directory(output) ? output / "collection.vcobj" : output, *report);
            const auto text = result.dump(2) + '\n';
            vcr::AtomicFile file(report->string());
            file.write(reinterpret_cast<const uint8_t*>(text.data()), text.size());
            file.publish(true);
        }
        return 0;
    } catch (const std::exception& error) { std::cerr<<"AFV import failed: "<<error.what()<<'\n'; return 1; }
}
