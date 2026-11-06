// Diagnostic benchmark, NOT a second supported object-store implementation.
// Source decoding and UUID generation finish before any import timer starts.
#include "vc_object_store/objects/AfvSource.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include "geometry/GeometryInternal.hpp"
#include "geometry/SharedGeometry.hpp"
#include "internal/RecordJson.hpp"
#include "internal/Database.hpp"
#include "internal/Sqlite.hpp"
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
    double readSeconds=0,baselineEncodingSeconds=0;
};
Prepared prepare(const Json& config){
    Prepared data;const auto start=Clock::now();
    AfvSource source(config.at("source").get<std::string>());
    const size_t limit=config.at("count");require(limit>0,"count must be positive");
    if(config.at("workload")=="relations"){
        AfcSource constraints(config.at("constraints").get<std::string>(),source.uuid());
        std::unordered_map<std::string,int64_t> ids;
        const auto endpoint=[&](const std::string& uuid){
            auto [it,added]=ids.try_emplace(uuid,int64_t(ids.size()+1));
            if(added){InitialObjectRecord o;o.id=it->second;o.record.uuid=uuid;
                o.record.kind="fiber";o.record.fiber.emplace();data.objects.push_back(std::move(o));}
            return it->second;
        };
        EdgeRecord edge;
        while(data.edges.size()<limit&&constraints.next(edge)){
            const auto a=endpoint(edge.aUuid),b=endpoint(edge.bUuid);
            data.endpoints.emplace_back(a,b);data.edges.push_back(std::move(edge));
        }
        require(data.edges.size()==limit,"insufficient constraints");
    }else{
        require(config.at("workload")=="geometry","unknown workload");
        AfvFiber fiber;
        while(data.objects.size()<limit&&source.next(fiber)){
            InitialObjectRecord o;o.id=int64_t(data.objects.size()+1);
            o.record.uuid=util::nameUuid(source.uuid(),"fiber:"+std::to_string(fiber.id));
            o.record.kind="fiber";o.record.name=fiber.name;o.record.fiber=fiber.detail;
            o.record.meta=Json({{"annotation",Json::parse(fiber.annotation)}}).dump();
            GeometryData g;g.kind="fiber";
            g.arrays={{"control_points","<f8",{0,3},{}},{"control_point_ids","<i8",{0},{}},
                {"line_points","<f8",{fiber.detail.linePointCount,3},std::move(fiber.xyz)}};
            data.points+=fiber.detail.linePointCount;data.bytes+=g.arrays.back().bytes.size();
            o.geometry=std::move(g);data.objects.push_back(std::move(o));
        }
        require(data.objects.size()==limit,"insufficient fibers");
    }
    data.readSeconds=elapsed(start);const auto encoding=Clock::now();
    // The reduced prototype stores the SAME input attributes in opaque payloads;
    // they are not dropped to make its insertion cheaper. No event bundling.
    for(const auto& o:data.objects){data.objectIds.push_back(d::parseUuid(o.record.uuid));
        data.objectPayload.push_back(d::objectToJson(o.record).dump());}
    for(const auto& e:data.edges){data.edgeIds.push_back(d::parseUuid(e.uuid));
        data.edgePayload.push_back(d::edgeToJson(e,nullptr,false).dump());}
    data.baselineEncodingSeconds=elapsed(encoding);return data;
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
Json reduced(const Prepared& data,const fs::path& path){
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
        fiber.run(*o.id,o.record.fiber->linePointCount);
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
    // Full payload readback outside the timed region (no dropped observations).
    const auto verify=Clock::now();
    d::Statement rows(db.p,"SELECT meta FROM edges ORDER BY id");
    for(const auto& payload:data.edgePayload)require(rows.row()&&rows.text(0)==payload,"reduced payload mismatch");
    require(!rows.row(),"extra reduced edge");result["verify_seconds"]=elapsed(verify);
    return result;
}
Json native(const Prepared& data,const fs::path& path,bool linked,bool packed,size_t workers=1,size_t sqlRows=1){
    InitialImportOptions opt;opt.store.author="prepared-benchmark";opt.store.sqliteCacheBytes=512*1024*1024;
    opt.edges.sqlBatchRows=sqlRows;
    require(workers>0&&workers<=data.objects.size(),"sharded benchmark needs at least one object per worker");
    const auto bounds=[&](size_t count,size_t i){return std::pair{count*i/workers,count*(i+1)/workers};};
    if(workers>1) {
        opt.shards.clear();
        const auto edgeRange=std::max(data.edges.size(),workers);
        for(size_t i=0;i<workers;++i) {
            const auto [a,b]=bounds(data.objects.size(),i);const auto [x,y]=bounds(edgeRange,i);
            opt.shards.push_back({{int64_t(a+1),int64_t(b)},{int64_t(x+1),int64_t(y)}});
        }
    }
    if(packed){opt.packed.emplace();opt.packed->disposableBulk=true;opt.packed->memoryFirst=true;
        opt.packed->backgroundFlush=false;opt.packed->residentBytes=size_t(4)*1024*1024*1024;}
    // Same 256-record native batches and statement size as the production pilot.
    std::string owner;double dependency=0;
    const auto appendObjects=[&](size_t worker,InitialShardImport& part){
        const auto [start,end]=bounds(data.objects.size(),worker);
        for(size_t i=start;i<end;i+=256)part.appendObjects(std::span(data.objects).subspan(i,std::min(size_t(256),end-i)));
    };
    if(linked){const auto base=path.string()+".base";const auto start=Clock::now();
        const auto result=importInitialCollection(base,opt,[&](size_t i,InitialShardImport& part){appendObjects(i,part);});
        owner=result.storeUuid;dependency=elapsed(start);opt.dependencies={base};}
    std::vector<d::GeometryWork> work(workers);
    const auto objectProducer=[&](size_t i,InitialShardImport& part){
        d::GeometryWorkScope profile(work[i]);if(!linked)appendObjects(i,part);
    };
    const auto edgeProducer=[&](size_t i,InitialShardImport& part){
        d::GeometryWorkScope profile(work[i]);
        const auto& endpointOwner=linked?owner:part.storeUuid();
        const auto [start,last]=bounds(data.edges.size(),i);size_t next=start;
        if(!data.edges.empty())part.importEdges([&](std::vector<InitialCollectionEdge>& batch){
            batch.clear();const auto end=std::min(next+256,last);batch.reserve(end-next);
            for(;next<end;++next){const auto& e=data.edges[next];batch.push_back({{endpointOwner,e.aUuid},{endpointOwner,e.bUuid},e});}
            return !batch.empty();
        });
    };
    const auto result=workers>1?importInitialCollection(path,opt,objectProducer,edgeProducer):
        importInitialCollection(path,opt,[&](size_t i,auto& part){objectProducer(i,part);edgeProducer(i,part);});
    if(!linked)owner=result.storeUuid;
    InitialShardResult summed;
    for(const auto& shard:result.shards){summed.objectSeconds+=shard.objectSeconds;summed.edgeSeconds+=shard.edgeSeconds;
        summed.producerSeconds+=shard.producerSeconds;summed.creationSeconds+=shard.creationSeconds;summed.sealCommitSeconds+=shard.sealCommitSeconds;}
    Json phases=Json::object();for(const auto& worker:work)for(const auto& [name,ns]:worker.phaseNanoseconds)
        phases[name]=phases.value(name,0.0)+double(ns)*1e-9;
    Json out={{"total_seconds",result.totalSeconds},{"objects_seconds",summed.objectSeconds},
        {"edges_seconds",summed.edgeSeconds},{"producer_seconds",summed.producerSeconds},
        {"creation_seconds",summed.creationSeconds},{"seal_commit_seconds",summed.sealCommitSeconds},
        {"manifest_seconds",result.manifestSeconds},{"persistence_seconds",result.persistenceSeconds},
        {"dependency_preparation_seconds_excluded",dependency},{"phase_seconds",phases},
        {"workers",workers},{"component_times_are_worker_sums",workers>1},{"ingestion_seconds",result.ingestionSeconds}};
    if(workers>1) {
        std::vector<size_t> ownerById(data.objects.size()+1);
        for(size_t i=0;i<workers;++i){const auto [a,b]=bounds(data.objects.size(),i);for(size_t n=a;n<b;++n)ownerById.at(size_t(*data.objects[n].id))=i;}
        size_t crossShard=0,remote=0;
        for(size_t i=0;i<workers;++i){const auto [a,b]=bounds(data.edges.size(),i);for(size_t n=a;n<b;++n){
            const auto left=ownerById.at(size_t(data.endpoints[n].first)),right=ownerById.at(size_t(data.endpoints[n].second));
            crossShard+=left!=right;remote+=(left!=i)+(right!=i);
        }}
        out["cross_shard_constraints"]=crossShard;out["remote_endpoint_references"]=remote;
    }
    const auto verify=Clock::now();auto collection=ObjectCollection::open(result.path,OpenOptions{.readOnly=true});auto txn=collection.read();
    for(size_t i=0;i<data.edges.size();i+=std::max(size_t(1),data.edges.size()/10)){
        const auto found=collection.edge(txn,{result.storeUuid,data.edges[i].uuid});
        require(found&&found->record.meta==data.edges[i].meta,"native metadata mismatch");
        require((found->a.objectUuid==data.edges[i].aUuid&&found->b.objectUuid==data.edges[i].bUuid)||
            (found->b.objectUuid==data.edges[i].aUuid&&found->a.objectUuid==data.edges[i].bUuid),"native endpoints mismatch");
    }
    if(!data.objects.empty()&&data.objects.front().geometry)
        require(collection.readGeometryArray(txn,{owner,data.objects.front().record.uuid},"line_points")==
            data.objects.front().geometry->arrays.back().bytes,"native geometry mismatch");
    out["verify_seconds"]=elapsed(verify);return out;
}
}
int main(int argc,char** argv){try{
    require(argc==2,"usage: vc_bench_prepared_import CONFIG.json");
    sqlite3_config(SQLITE_CONFIG_MEMSTATUS,0);
    std::ifstream input(argv[1]);const auto config=Json::parse(input);auto data=prepare(config);
    std::cout<<Json({{"stage","prepared"},{"objects",data.objects.size()},{"constraints",data.edges.size()},
        {"points",data.points},{"xyz_bytes",data.bytes},{"source_seconds",data.readSeconds},
        {"prototype_encoding_seconds",data.baselineEncodingSeconds},
        {"build",{{"compiler",__VERSION__},{"compiled",std::string(__DATE__)+" "+__TIME__},
            {"cplusplus",__cplusplus},{"sqlite",sqlite3_libversion()},
            {"source_revision",config.value("source_revision",std::string("unrecorded"))}}},
        {"configuration",config}}).dump()<<std::endl;
    const fs::path root=config.at("output_root").get<std::string>();require(fs::is_directory(root),"output_root must exist");
    const std::string variant=config.at("variant");
    require(variant=="prototype"||variant=="native"||variant=="linked"||variant=="native_packed"||
        variant=="native_sharded"||variant=="native_packed_sharded","unknown variant");
    const bool sharded=variant.ends_with("_sharded");
    const size_t workers=sharded?config.value("workers",size_t(4)):1;
    for(int repeat=0;repeat<config.value("repetitions",1);++repeat){
        const auto path=root/(variant+std::to_string(repeat)+".sqlite");require(!fs::exists(path),"refusing overwrite");
        Json result=variant=="prototype"?reduced(data,path):native(data,path,variant=="linked",variant.starts_with("native_packed"),workers,config.value("sql_batch_rows",size_t(1)));
        result["variant"]=variant;result["repetition"]=repeat;result["tables"]=inventory(path);
        require(result["tables"].value("edges",size_t(0))==data.edges.size(),"edge count mismatch");
        require(result["tables"].value("objects",size_t(0))==(variant=="linked"?0:data.objects.size()),"objects must not be duplicated into relation shards");
        uint64_t bytes=0;
        if(fs::is_directory(path)){for(const auto& member:fs::directory_iterator(path))if(member.path().extension()==".vcobj")bytes+=fs::file_size(member.path());}
        else bytes=fs::file_size(path);
        result["database_bytes"]=bytes;std::cout<<result.dump()<<std::endl;
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
