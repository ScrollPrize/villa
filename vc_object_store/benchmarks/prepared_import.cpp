// Diagnostic benchmark, NOT a second supported object-store implementation.
// Source decoding and UUID generation finish before any import timer starts.
#include "vc_object_store/objects/AfvSource.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include "vc_object_store/objects/imports/ParallelImportStream.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/SharedGeometry.hpp"
#include "internal/RecordJson.hpp"
#include "internal/Database.hpp"
#include "internal/Sqlite.hpp"
#include "imports/AfvRecords.hpp"
#include "../apps/support/ProcessMetrics.hpp"
#include <fstream>
#include <iostream>
#include <unordered_map>

using namespace vc::objects;
namespace d=vc::objects::detail;
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
namespace fs=std::filesystem;
namespace {
double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Db {
    sqlite3* p=nullptr;
    explicit Db(const fs::path& path,bool read=false){
        const bool locator=isPackedDatabasePath(path);
        const bool packed=locator||(fs::exists(path)&&isPackedStore(path));
        const auto name=packed&&!locator?packedDatabasePath(path).string():path.string();
        d::check(sqlite3_open_v2(name.c_str(),&p,
            read?SQLITE_OPEN_READONLY:SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE,packed?packedStoreVfs():nullptr),p);}
    ~Db(){if(p)sqlite3_close_v2(p);}
};
struct Prepared {
    std::vector<InitialObjectRecord> objects;
    std::vector<EdgeRecord> edges;
    std::vector<std::array<unsigned char,16>> objectIds,edgeIds;
    std::vector<std::pair<int64_t,int64_t>> endpoints;
    std::vector<std::string> objectPayload,edgePayload;
    uint64_t points=0,bytes=0;
    int64_t sourceFibers=0,sourcePoints=0,sourceConstraints=0;
    size_t geometryReaders=0,geometryRanges=0;
    double readSeconds=0,sourceOpenSeconds=0,sourceSelectSeconds=0,sourceDecodeSeconds=0,recordSeconds=0,baselineEncodingSeconds=0;
};
InitialObjectRecord geometryRecord(Prepared& data,AfvFiber&& fiber,const std::string& uuid,int64_t id,uint64_t maxBytes)
{
    if(fiber.xyz.size()>maxBytes || data.bytes>maxBytes-fiber.xyz.size())
        throw std::runtime_error("prepared geometry exceeds max_prepared_xyz_bytes; reduce the input count, not its precision");
    InitialObjectRecord o;o.id=id;o.record.uuid=uuid;
    o.record.kind="fiber";o.record.name=std::move(fiber.name);o.record.fiber=std::move(fiber.detail);
    o.record.meta=Json({{"annotation",Json::parse(fiber.annotation)}}).dump();
    GeometryData g;g.kind="fiber";
    g.arrays={{"control_points","<f8",{0,3},{}},{"control_point_ids","<i8",{0},{}},
        {"line_points","<f8",{o.record.fiber->linePointCount,3},std::move(fiber.xyz)}};
    data.points+=o.record.fiber->linePointCount;data.bytes+=g.arrays.back().bytes.size();
    o.geometry=std::move(g);return o;
}
Prepared prepare(const Json& config){
    Prepared data;const auto start=Clock::now();
    const auto sourcePath=config.at("source").get<std::string>();AfvSource source(sourcePath,0,0);
    data.sourceOpenSeconds=elapsed(start);
    data.sourceFibers=source.fiberCount();data.sourcePoints=source.pointCount();
    const size_t limit=config.at("count");require(limit>0,"count must be positive");
    const uint64_t maxBytes=config.value("max_prepared_xyz_bytes",uint64_t(1024)*1024*1024);
    const auto workload=config.at("workload").get<std::string>();
    if(workload=="relations"||workload=="combined"){
        AfcSource constraints(config.at("constraints").get<std::string>(),source.uuid());
        data.sourceConstraints=constraints.constraintCount();
        std::unordered_map<std::string,int64_t> ids;
        std::map<int64_t,int64_t> sourceIds;
        const auto endpoint=[&](const std::string& uuid){
            auto [it,added]=ids.try_emplace(uuid,int64_t(ids.size()+1));
            if(added){InitialObjectRecord o;o.id=it->second;o.record.uuid=uuid;
                o.record.kind="fiber";o.record.fiber.emplace();data.objects.push_back(std::move(o));}
            return it->second;
        };
        EdgeRecord edge;
        while(data.edges.size()<limit&&constraints.next(edge)){
            const auto a=endpoint(edge.aUuid),b=endpoint(edge.bUuid);
            if(workload=="combined") {
                const auto meta=Json::parse(edge.meta).at("afc");
                for(const auto& [key,logical]:std::array<std::pair<const char*,int64_t>,2>{{{"source_fiber_a",a},{"source_fiber_b",b}}}) {
                    const auto original=meta.at(key).get<int64_t>();
                    auto [found,added]=sourceIds.emplace(original,logical);
                    require(added||found->second==logical,"AFC endpoint source identity is inconsistent");
                }
            }
            data.endpoints.emplace_back(a,b);data.edges.push_back(std::move(edge));
        }
        require(data.edges.size()==limit,"insufficient constraints");
        // Seek only required fibers using the native range reader. Consecutive
        // source IDs share one cursor; each requested geometry is decoded once.
        if(workload=="combined")for(auto first=sourceIds.begin();first!=sourceIds.end();) {
            auto last=std::next(first);
            while(last!=sourceIds.end()&&last->first==std::prev(last)->first+1)++last;
            const auto finalId=std::prev(last)->first;
            require(finalId<INT64_MAX,"source fiber ID cannot form a bounded interval");
            source.selectRange(first->first,finalId+1);data.geometryReaders=1;++data.geometryRanges;
            AfvFiber fiber;
            for(auto entry=first;entry!=last;++entry) {
                require(source.next(fiber)&&fiber.id==entry->first,"required AFC endpoint geometry is missing");
                const auto index=size_t(entry->second-1);
                const auto uuid=util::nameUuid(source.uuid(),"fiber:"+std::to_string(fiber.id));
                require(data.objects.at(index).record.uuid==uuid,"AFC and AFV endpoint UUID mismatch");
                data.objects[index]=geometryRecord(data,std::move(fiber),uuid,entry->second,maxBytes);
            }
            require(!source.next(fiber),"bounded AFV reader returned extra fibers");first=last;
        }
    }else{
        require(workload=="geometry","unknown workload");
        const bool production=config.value("afv_records",false);
        const bool includeConstraints=config.value("include_constraints",false);
        require(!includeConstraints||(production&&config.contains("constraints")),"include_constraints requires afv_records and constraints");
        const auto volume=config.value("volume_id",std::string{});
        const Json coordinates={{"volume_uuid",volume.empty()?Json(nullptr):Json(volume)},
                                {"scale_factor",config.value("scale_factor",1.)}};
        if(config.contains("constraints")) {
            AfcSource constraints(config.at("constraints").get<std::string>(),source.uuid());
            data.sourceConstraints=constraints.constraintCount();
        }
        const auto sampling=config.value("geometry_sampling",std::string("prefix"));
        require(sampling=="prefix"||sampling=="uniform","unknown geometry_sampling");
        require(!includeConstraints||sampling=="prefix","matched constraint import requires prefix geometry");
        int64_t lastSourceId=-1;
        const auto accept=[&](AfvFiber&& fiber) {
            const auto recordStart=Clock::now();
            lastSourceId=fiber.id;
            if(production){
                require(fiber.xyz.size()<=maxBytes&&data.bytes<=maxBytes-fiber.xyz.size(),"prepared XYZ exceeds budget");
                InitialObjectRecord row;row.id=int64_t(data.objects.size()+1);row.record=imports::afvObjectRecord(fiber,source.uuid(),coordinates);
                data.points+=fiber.detail.linePointCount;data.bytes+=fiber.xyz.size();
                row.geometry=imports::afvGeometry(fiber);data.objects.push_back(std::move(row));
            }else {
                const auto uuid=util::nameUuid(source.uuid(),"fiber:"+std::to_string(fiber.id));
                data.objects.push_back(geometryRecord(data,std::move(fiber),uuid,int64_t(data.objects.size()+1),maxBytes));
            }
            data.recordSeconds+=elapsed(recordStart);
        };
        AfvFiber fiber;
        if(sampling=="prefix") {
            const auto selection=Clock::now();source.selectRange(0,INT64_MAX);data.sourceSelectSeconds+=elapsed(selection);
            data.geometryReaders=1;++data.geometryRanges;
            while(data.objects.size()<limit){
                const auto read=Clock::now();const bool available=source.next(fiber);data.sourceDecodeSeconds+=elapsed(read);
                if(!available)break;accept(std::move(fiber));
            }
        } else {
            // One catalog-only partition scan, not one scan per sample. The
            // first fiber in each balanced interval is a deterministic sample.
            const auto ranges=source.idRanges(limit);
            require(ranges.size()==limit,"insufficient fibers for uniform sample");
            for(const auto& range:ranges) {
                source.selectRange(range[0],range[1]);data.geometryReaders=1;++data.geometryRanges;
                require(source.next(fiber),"empty uniform geometry interval");accept(std::move(fiber));
            }
        }
        require(data.objects.size()==limit,"insufficient fibers");
        Json constraintMetadata;
        if(includeConstraints){
            require(lastSourceId<INT64_MAX,"last source ID cannot form a bounded interval");
            AfcSource constraints(config.at("constraints").get<std::string>(),source.uuid(),lastSourceId+1);
            constraintMetadata=Json::parse(constraints.metadataJson());
            std::unordered_map<std::string,int64_t> ids;
            for(const auto& row:data.objects)ids.emplace(row.record.uuid,*row.id);
            const auto constraintLimit=config.at("constraint_limit").get<size_t>();
            require(constraintLimit>0,"matched benchmark needs a positive constraint_limit");
            EdgeRecord edge;
            while(data.edges.size()<constraintLimit&&constraints.next(edge)){
                require(ids.contains(edge.aUuid)&&ids.contains(edge.bUuid),"constraint endpoint is absent from the prepared geometry");
                data.endpoints.emplace_back(ids.at(edge.aUuid),ids.at(edge.bUuid));data.edges.push_back(std::move(edge));
            }
            require(data.edges.size()==constraintLimit,"insufficient eligible constraints");
        }
        if(production){
            InitialObjectRecord metadata;metadata.id=int64_t(data.objects.size()+1);metadata.record.uuid=source.uuid();metadata.record.kind="collection";
            metadata.record.name=fs::path(sourcePath).stem().string();
            Json fields={{"coordinates",coordinates},{"afv",{{"metadata",Json::parse(source.metadataJson())}}}};
            if(includeConstraints)fields["afc"]=std::move(constraintMetadata);
            metadata.record.meta=fields.dump();
            data.objects.push_back(std::move(metadata));
        }
    }
    data.readSeconds=elapsed(start);return data;
}
void preparePrototype(Prepared& data){
    const auto encoding=Clock::now();
    // The reduced prototype stores the SAME input attributes in opaque payloads;
    // they are not dropped to make its insertion cheaper. No event bundling.
    for(const auto& o:data.objects){data.objectIds.push_back(d::parseUuid(o.record.uuid));
        data.objectPayload.push_back(d::objectToJson(o.record).dump());}
    for(const auto& e:data.edges){data.edgeIds.push_back(d::parseUuid(e.uuid));
        data.edgePayload.push_back(d::edgeToJson(e,nullptr,false).dump());}
    data.baselineEncodingSeconds=elapsed(encoding);
}
Json inventory(const fs::path& path){
    std::vector<fs::path> members;
    if(fs::is_directory(path)) {
        for(const auto& file:fs::directory_iterator(path))if(file.path().extension()==".vcobj")members.push_back(file.path());
    } else if(!isPackedDatabasePath(path)&&isPackedStore(path)) {
        for(const auto& member:packedDatabases(path))members.push_back(packedDatabasePath(path,member.uuid));
    }
    if(!members.empty()) {
        Json total=Json::object();
        for(const auto& member:members) {
            const auto rows=inventory(member);
            for(const auto& [table,count]:rows.items())total[table]=total.value(table,int64_t(0))+count.get<int64_t>();
        }
        return total;
    }
    Db db(path,true);Json tables=Json::object();
    d::Statement names(db.p,"SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name");
    while(names.row()){
        std::string name=names.text(0),quoted="\"";
        for(char c:name){quoted+=c;if(c=='\"')quoted+='\"';}quoted+='\"';
        d::Statement count(db.p,"SELECT count(*) FROM "+quoted);count.row();
        if(count.i64(0))tables[name]=count.i64(0);
    }
    return tables;
}
Json reduced(const Prepared& data,const fs::path& path,bool verifyReadback){
    auto start=Clock::now();Json result;Db db(path);
    d::exec(db.p,"PRAGMA page_size=65536;PRAGMA journal_mode=OFF;PRAGMA synchronous=OFF;"
        "PRAGMA foreign_keys=ON;PRAGMA cache_size=-524288;"
        "CREATE TABLE objects(id INTEGER PRIMARY KEY,uuid BLOB NOT NULL UNIQUE,kind TEXT NOT NULL,meta TEXT NOT NULL);"
        "CREATE TABLE fibers(object_id INTEGER PRIMARY KEY REFERENCES objects(id),line_point_count INTEGER NOT NULL);"
        "CREATE TABLE edges(id INTEGER PRIMARY KEY,uuid BLOB NOT NULL UNIQUE,a_id INTEGER NOT NULL REFERENCES objects(id),"
        "b_id INTEGER NOT NULL REFERENCES objects(id),kind TEXT NOT NULL,meta TEXT NOT NULL);"
        "CREATE TABLE geometry_chunks(hash TEXT PRIMARY KEY,data BLOB NOT NULL);"
        "CREATE TABLE geometry_revision_chunks(revision_uuid BLOB NOT NULL REFERENCES objects(uuid),array_name TEXT NOT NULL,"
        "chunk_index INTEGER NOT NULL,hash TEXT NOT NULL REFERENCES geometry_chunks(hash),PRIMARY KEY(revision_uuid,array_name,chunk_index));BEGIN;");
    result["creation_seconds"]=elapsed(start);auto objects=Clock::now();double hash=0;
    {d::Statement object(db.p,"INSERT INTO objects VALUES(?,?,?,?)"),fiber(db.p,"INSERT INTO fibers VALUES(?,?)"),
        chunk(db.p,"INSERT INTO geometry_chunks VALUES(?,?) ON CONFLICT(hash) DO NOTHING"),mapping(db.p,"INSERT INTO geometry_revision_chunks VALUES(?,'line_points',?,?)");
    for(size_t i=0;i<data.objects.size();++i){const auto& o=data.objects[i];
        object.run(*o.id,data.objectIds[i],o.record.kind,data.objectPayload[i]);
        if(o.record.fiber)fiber.run(*o.id,o.record.fiber->linePointCount);
        if(o.geometry){
            d::forEachGeometryChunk(o.geometry->arrays.back(),0,[&](int64_t index,std::span<const uint8_t> bytes){
                const auto t=Clock::now();const auto key=d::geometryPayloadHash("line_points",bytes);hash+=elapsed(t);
                chunk.run(key,bytes);mapping.run(data.objectIds[i],int64_t(index),key);
            });}
    }}
    result["objects_seconds"]=elapsed(objects);result["hash_seconds_in_objects"]=hash;
    const auto edges=Clock::now();
    {d::Statement edge(db.p,"INSERT INTO edges VALUES(?,?,?,?,?,?)");
    for(size_t i=0;i<data.edges.size();++i)edge.run(int64_t(i+1),data.edgeIds[i],data.endpoints[i].first,
        data.endpoints[i].second,data.edges[i].kind,data.edgePayload[i]);}
    result["edges_seconds"]=elapsed(edges);const auto finish=Clock::now();
    d::exec(db.p,"CREATE INDEX edges_a ON edges(a_id,kind);CREATE INDEX edges_b ON edges(b_id,kind);"
        "CREATE INDEX objects_kind ON objects(kind);COMMIT;");
    result["finish_seconds"]=elapsed(finish);result["total_seconds"]=elapsed(start);
    result["readback_verified"]=verifyReadback;result["verify_seconds"]=0.0;
    if(!verifyReadback)return result;
    // Full payload readback outside the timed region (no dropped observations).
    const auto verify=Clock::now();
    d::Statement rows(db.p,"SELECT meta FROM edges ORDER BY id");
    for(const auto& payload:data.edgePayload)require(rows.row()&&rows.text(0)==payload,"reduced payload mismatch");
    require(!rows.row(),"extra reduced edge");
    for(size_t i=0;i<data.objects.size();i+=std::max(size_t(1),data.objects.size()/10))if(data.objects[i].geometry) {
        d::Statement chunks(db.p,"SELECT c.data FROM geometry_revision_chunks r JOIN geometry_chunks c ON c.hash=r.hash "
            "WHERE r.revision_uuid=? AND r.array_name='line_points' ORDER BY r.chunk_index",data.objectIds[i]);
        const auto& expected=data.objects[i].geometry->arrays.back().bytes;size_t offset=0;
        while(chunks.row()) {
            const auto size=size_t(sqlite3_column_bytes(chunks.value,0));
            require(offset<=expected.size()&&size<=expected.size()-offset,"extra reduced geometry bytes");
            require(!size||std::memcmp(sqlite3_column_blob(chunks.value,0),expected.data()+offset,size)==0,"reduced geometry mismatch");offset+=size;
        }
        require(offset==expected.size(),"missing reduced geometry bytes");
    }
    result["verify_seconds"]=elapsed(verify);
    return result;
}
Json native(Prepared& data,const fs::path& path,bool linked,bool packed,size_t workers,size_t sqlRows,
    size_t exportWorkers,bool preSizeExport,size_t residentBytes,bool verifyReadback,size_t sqliteCacheBytes){
    using namespace vc::objects::imports;
    InitialImportOptions opt;opt.store.author="prepared-benchmark";opt.store.sqliteCacheBytes=sqliteCacheBytes;
    opt.edges.sqlBatchRows=sqlRows;
    require(workers>0&&workers<=data.objects.size(),"sharded benchmark needs at least one object per worker");
    // Retain benchmark-local prototype IDs, but let native shards allocate their
    // own disjoint numeric ranges. Geometry and metadata are never copied here.
    const auto partitionStart=Clock::now();
    struct RestoreIds {Prepared& data;std::vector<std::optional<int64_t>> ids;
        ~RestoreIds(){for(size_t i=0;i<ids.size();++i)data.objects[i].id=ids[i];}} restore{data,{}};
    restore.ids.reserve(data.objects.size());
    for(auto& row:data.objects){restore.ids.push_back(row.id);row.id.reset();}
    opt.shards.clear();const auto reservation=INT64_MAX/int64_t(workers);
    for(size_t i=0;i<workers;++i){
        const std::array<int64_t,2> range{int64_t(i)*reservation+1,int64_t(i+1)*reservation};opt.shards.push_back({range,range});
    }
    const auto partitionSeconds=elapsed(partitionStart);
    if(packed){opt.packed.emplace();opt.packed->disposableBulk=true;opt.packed->memoryFirst=true;
        opt.packed->backgroundFlush=false;opt.packed->residentBytes=residentBytes;
        opt.packed->exportWorkers=exportWorkers;opt.packed->preSizeExport=preSizeExport;}
    std::string owner;double dependency=0;
    std::vector<d::GeometryWork> work(workers);
    Json perReader=Json::array();
    const auto runImport=[&](const fs::path& target,bool withObjects,bool withEdges,bool profile){
      ParallelImportStream stream({workers,workers,256,8*1024*1024,256*1024*1024});
      constexpr size_t objectChunk=1024,edgeChunk=2048;
      stream.start(withObjects?(data.objects.size()+objectChunk-1)/objectChunk:0,
        withEdges?(data.edges.size()+edgeChunk-1)/edgeChunk:0,
        [&](size_t,size_t task,ObjectEmitter& emit){
            for(size_t i=task*objectChunk;i<std::min((task+1)*objectChunk,data.objects.size());++i)
                emit.emit(ImportObject{&data.objects[i]},sizeof(ImportObject),data.objects[i].record.uuid);
        },[&](size_t,size_t task,EdgeEmitter& emit){
            for(size_t i=task*edgeChunk;i<std::min((task+1)*edgeChunk,data.edges.size());++i){
                const auto& e=data.edges[i];emit.emit({{"",e.aUuid},{"",e.bUuid},e},sizeof(InitialCollectionEdge)+e.meta.size(),e.uuid);
            }
        });
      const auto objectProducer=[&](size_t i,InitialShardImport& part){try{
        // The builder uses fresh std::async worker threads and joins them before
        // returning. Keep profiling until thread exit so seal/commit/checkpoint
        // are covered too; work remains alive until after every thread joins.
        thread_local std::unique_ptr<d::GeometryWorkScope> profileScope;
        if(profile)profileScope=std::make_unique<d::GeometryWorkScope>(work[i]);
        std::vector<ImportObject> rows;std::vector<const InitialObjectRecord*> refs;
        while(stream.popObjects(i,rows,[&]{part.checkCancelled();})){
            refs.clear();refs.reserve(rows.size());for(const auto& row:rows)refs.push_back(&row.get());part.appendObjectRefs(refs);
        }
      }catch(...){stream.fail(std::current_exception());throw;}};
      const auto edgeProducer=[&](size_t i,InitialShardImport& part){try{
        if(withEdges)part.importEdges([&](std::vector<InitialCollectionEdge>& batch){return stream.popEdges(i,batch,[&]{part.checkCancelled();});});
      }catch(...){stream.fail(std::current_exception());throw;}};
      InitialImportResult result;
      try{result=importInitialCollection(target,opt,objectProducer,edgeProducer);stream.finish();}
      catch(...){stream.fail(std::current_exception());try{stream.finish();}catch(...){}throw;}
      if(profile)for(size_t i=0;i<stream.readers.size();++i){const auto& r=stream.readers[i];perReader.push_back({{"reader",i},
        {"object_tasks",r.objectTasks},{"edge_tasks",r.edgeTasks},{"object_seconds",r.objectSeconds},{"edge_seconds",r.edgeSeconds},{"queue_wait_seconds",r.queueWaitSeconds}});}
      return result;
    };
    if(linked){const auto base=path.string()+".base";const auto start=Clock::now();const auto result=runImport(base,true,false,false);
        owner=result.storeUuid;dependency=elapsed(start);opt.dependencies={base};}
    const auto result=runImport(path,!linked,true,true);
    if(!linked)owner=result.storeUuid;
    InitialShardResult summed;
    for(const auto& shard:result.shards){summed.objectSeconds+=shard.objectSeconds;summed.edgeSeconds+=shard.edgeSeconds;
        summed.producerSeconds+=shard.producerSeconds;summed.creationSeconds+=shard.creationSeconds;summed.sealCommitSeconds+=shard.sealCommitSeconds;}
    Json phases=Json::object();for(const auto& worker:work)for(const auto& [name,ns]:worker.phaseNanoseconds)
        phases[name]=phases.value(name,0.0)+double(ns)*1e-9;
    Json perWorker=Json::array();
    for(size_t i=0;i<workers;++i) {
        const auto& shard=result.shards[i];Json seconds=Json::object();
        for(const auto& [name,ns]:work[i].phaseNanoseconds)seconds[name]=double(ns)*1e-9;
        perWorker.push_back({{"worker",i},{"objects",shard.objects},{"edges",shard.edges},
            {"objects_seconds",shard.objectSeconds},{"edges_seconds",shard.edgeSeconds},
            {"creation_seconds",shard.creationSeconds},{"seal_commit_seconds",shard.sealCommitSeconds},
            {"producer_seconds",shard.producerSeconds},{"phase_seconds",seconds}});
    }
    Json out={{"total_seconds",result.totalSeconds},{"objects_seconds",summed.objectSeconds},
        {"edges_seconds",summed.edgeSeconds},{"producer_seconds",summed.producerSeconds},
        {"creation_seconds",summed.creationSeconds},{"seal_commit_seconds",summed.sealCommitSeconds},
        {"manifest_seconds",result.manifestSeconds},{"persistence_seconds",result.persistenceSeconds},
        {"uuid_partition_preparation_seconds_excluded",partitionSeconds},{"uuid_hash_routing",true},{"parallel_stream",true},{"per_reader",perReader},
        {"dependency_preparation_seconds_excluded",dependency},{"phase_seconds",phases},{"per_worker",perWorker},
        {"workers",workers},{"component_times_are_worker_sums",workers>1},{"ingestion_seconds",result.ingestionSeconds}};
    if(workers>1) {
        size_t crossShard=0,remote=0;
        for(size_t n=0;n<data.edges.size();++n){const auto i=uuidShardIndex(data.edges[n].uuid,workers);
            const auto left=uuidShardIndex(data.edges[n].aUuid,workers),right=uuidShardIndex(data.edges[n].bUuid,workers);
            crossShard+=left!=right;remote+=(left!=i)+(right!=i);
        }
        out["cross_shard_constraints"]=crossShard;out["remote_endpoint_references"]=remote;
    }
    out["readback_verified"]=verifyReadback;out["verify_seconds"]=0.0;
    if(!verifyReadback)return out;
    const auto verify=Clock::now();auto collection=ObjectCollection::open(result.path,OpenOptions{.readOnly=true});auto txn=collection.read();
    for(size_t i=0;i<data.edges.size();i+=std::max(size_t(1),data.edges.size()/10)){
        const auto found=collection.edge(txn,{result.storeUuid,data.edges[i].uuid});
        require(found&&found->record.meta==data.edges[i].meta,"native metadata mismatch");
        require((found->a.objectUuid==data.edges[i].aUuid&&found->b.objectUuid==data.edges[i].bUuid)||
            (found->b.objectUuid==data.edges[i].aUuid&&found->a.objectUuid==data.edges[i].bUuid),"native endpoints mismatch");
    }
    for(size_t i=0;i<data.objects.size();i+=std::max(size_t(1),data.objects.size()/10))if(data.objects[i].geometry)
        require(collection.readGeometryArray(txn,{owner,data.objects[i].record.uuid},"line_points")==
            data.objects[i].geometry->arrays.back().bytes,"native geometry mismatch");
    out["verify_seconds"]=elapsed(verify);return out;
}
}
int main(int argc,char** argv){try{
    require(argc==2,"usage: vc_bench_prepared_import CONFIG.json");
    sqlite3_config(SQLITE_CONFIG_MEMSTATUS,0);
    std::ifstream input(argv[1]);const auto config=Json::parse(input);
    Json cases=config.value("cases",Json::array());
    if(!config.contains("cases"))cases.push_back({{"variant",config.at("variant")}});
    require(cases.is_array()&&!cases.empty(),"cases must be a nonempty array");
    bool needsPrototype=false;
    for(const auto& run:cases) {
        require(run.is_object(),"each case must be an object");
        for(const auto& [key,value]:run.items())require(key=="variant"||key=="workers"||key=="sql_batch_rows"||key=="repetitions"||key=="label"||
            key=="export_workers"||key=="pre_size_export"||key=="resident_bytes"||key=="sqlite_cache_bytes",
            "cases may change only variant, workers, sql_batch_rows, repetitions, label, export_workers, pre_size_export, resident_bytes or sqlite_cache_bytes; input is shared");
        const std::string variant=run.at("variant");
        require(variant=="prototype"||variant=="native"||variant=="linked"||variant=="native_packed"||
            variant=="native_sharded"||variant=="native_packed_sharded","unknown variant");
        require(run.value("repetitions",config.value("repetitions",1))>0,"repetitions must be positive");
        needsPrototype|=variant=="prototype";
    }
    auto data=prepare(config);
    if(needsPrototype)preparePrototype(data);
    size_t indexedAnchors=0,positionAnchors=0,wholeAnchors=0;
    for(const auto& edge:data.edges)for(const auto* anchor:{&edge.aAnchor,&edge.bAnchor}) {
        indexedAnchors+=!anchor->points.empty();positionAnchors+=bool(anchor->position);wholeAnchors+=anchor->wholeGeometry;
    }
    std::cout<<Json({{"stage","prepared"},{"objects",data.objects.size()},{"constraints",data.edges.size()},
        {"points",data.points},{"xyz_bytes",data.bytes},{"source_seconds",data.readSeconds},{"peak_rss_bytes",benchmark::peakRssBytes()},
        {"source_open_seconds",data.sourceOpenSeconds},{"source_select_seconds",data.sourceSelectSeconds},
        {"source_decode_seconds",data.sourceDecodeSeconds},{"record_preparation_seconds",data.recordSeconds},
        {"prototype_encoding_seconds",data.baselineEncodingSeconds},{"prototype_encoding_performed",needsPrototype},
        {"geometry_reader_count",data.geometryReaders},{"geometry_range_count",data.geometryRanges},{"indexed_anchors",indexedAnchors},
        {"position_anchors",positionAnchors},{"whole_geometry_anchors",wholeAnchors},
        {"source_totals",{{"fibers",data.sourceFibers},{"points",data.sourcePoints},{"constraints",data.sourceConstraints}}},
        {"build",{{"compiler",__VERSION__},{"compiled",std::string(__DATE__)+" "+__TIME__},
            {"cplusplus",__cplusplus},{"sqlite",sqlite3_libversion()},
            {"source_revision",config.value("source_revision",std::string("unrecorded"))}}},
        {"configuration",config}}).dump()<<std::endl;
    const fs::path root=config.at("output_root").get<std::string>();require(fs::is_directory(root),"output_root must exist");
    for(size_t caseIndex=0;caseIndex<cases.size();++caseIndex) {
      const auto& run=cases[caseIndex];const std::string variant=run.at("variant");
      const bool sharded=variant.ends_with("_sharded");
      const size_t workers=sharded?run.value("workers",config.value("workers",size_t(4))):1;
      const size_t sqlRows=run.value("sql_batch_rows",config.value("sql_batch_rows",size_t(1)));
      const size_t exportWorkers=run.value("export_workers",config.value("export_workers",size_t(1)));
      const bool preSizeExport=run.value("pre_size_export",config.value("pre_size_export",false));
      const size_t residentBytes=run.value("resident_bytes",config.value("resident_bytes",size_t(4)*1024*1024*1024));
      require(exportWorkers>0&&residentBytes>0,"export_workers and resident_bytes must be positive");
      for(int repeat=0;repeat<run.value("repetitions",config.value("repetitions",1));++repeat){
        const auto path=root/("case"+std::to_string(caseIndex)+"-"+variant+std::to_string(repeat)+".sqlite");require(!fs::exists(path),"refusing overwrite");
        std::cout<<Json({{"stage","case_start"},{"case",caseIndex},{"variant",variant},{"workers",workers},{"repetition",repeat}}).dump()<<std::endl;
        const bool verifyReadback=config.value("verify_readback",false);
        Json result=variant=="prototype"?reduced(data,path,verifyReadback):native(data,path,variant=="linked",variant.starts_with("native_packed"),workers,sqlRows,
            exportWorkers,preSizeExport,residentBytes,verifyReadback,run.value("sqlite_cache_bytes",config.value("sqlite_cache_bytes",size_t(512)*1024*1024)));
        result["sqlite_cache_bytes"]=run.value("sqlite_cache_bytes",config.value("sqlite_cache_bytes",size_t(512)*1024*1024));
        result["case"]=caseIndex;result["label"]=run.value("label",variant);result["variant"]=variant;result["workers"]=workers;
        result["sql_batch_rows"]=sqlRows;result["repetition"]=repeat;
        result["packed_export_options_applied"]=variant.starts_with("native_packed");
        result["export_workers"]=exportWorkers;result["pre_size_export"]=preSizeExport;result["resident_bytes"]=residentBytes;
        result["inventory_scanned"]=config.value("inventory",false);
        if(result["inventory_scanned"].get<bool>()) {
            result["tables"]=inventory(path);
            require(result["tables"].value("edges",size_t(0))==data.edges.size(),"edge count mismatch");
            require(result["tables"].value("objects",size_t(0))==(variant=="linked"?0:data.objects.size()),"objects must not be duplicated into relation shards");
        }
        uint64_t bytes=0;
        if(fs::is_directory(path)){for(const auto& member:fs::directory_iterator(path))if(member.path().extension()==".vcobj")bytes+=fs::file_size(member.path());}
        else bytes=fs::file_size(path);
        result["database_bytes"]=bytes;result["peak_rss_bytes"]=benchmark::peakRssBytes();std::cout<<result.dump()<<std::endl;
        // Outputs are private benchmark fixtures, not experiment deliverables.
        // Remove only successfully created paths; the runner handles aborts.
        if(config.value("remove_completed_outputs",true)) {
            std::vector<fs::path> outputs{path};if(variant=="linked")outputs.emplace_back(path.string()+".base");
            for(const auto& output:outputs) {
                fs::remove_all(output);
                for(const auto* suffix:{"-wal","-shm","-journal"})fs::remove(output.string()+suffix);
            }
        }
      }
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
