#include "vc_object_store/objects/FiberConstraintGraph.hpp"
#include "vc_object_store/Uuid.hpp"
#include "vc_object_store/ParallelFor.hpp"
#include "internal/CollectionReads.hpp"
#include "internal/Extensions.hpp"
#include "internal/Sqlite.hpp"
#include "internal/EdgeLayout.hpp"
#include "internal/VolumeFrames.hpp"
#include "internal/TransformedLength.hpp"
#include "geometry/GeometryToken.hpp"
#include "vc_object_store/objects/FiberGraphSolveExtension.hpp"
#include <algorithm>
#include <atomic>
#include <climits>
#include <future>
#include <memory_resource>
#include <mutex>
#include <numeric>
#include <unordered_map>

namespace vc::objects {
FiberGraphUuid fiberGraphUuid(const std::string& value){return detail::parseUuid(value);}
std::string fiberGraphUuid(const FiberGraphUuid& value){return detail::formatUuid(value);}
std::unique_ptr<uint8_t[]> fiberGraphUuidAscii(std::span<const FiberGraphUuid> values,size_t workers){
    auto text=std::make_unique_for_overwrite<uint8_t[]>(values.size()*36);
    parallelFor((values.size()+4095)/4096,workers,[&](size_t task){
        for(size_t i=task*4096;i<std::min(values.size(),(task+1)*4096);++i){
            const auto id=fiberGraphUuid(values[i]);std::copy(id.begin(),id.end(),text.get()+i*36);
        }
    });return text;
}
namespace {
struct UuidHash {size_t operator()(const FiberGraphUuid& key)const {
    return util::uuidHash64(key);
}};
std::span<const uint8_t> blob(const detail::Statement& row,int column,size_t size){
    if(sqlite3_column_type(row.value,column)!=SQLITE_BLOB||sqlite3_column_bytes(row.value,column)!=int(size))
        throw std::invalid_argument("invalid compact fiber measurement BLOB size/type");
    return {static_cast<const uint8_t*>(sqlite3_column_blob(row.value,column)),size};
}
void requireFormat(const ExtensionTableInfo& info){
    const auto expected=fiberConstraintExtension({});
    if(info.formatVersion!=kFiberConstraintFormat||info.columns!=expected.columns||info.primaryKey!=expected.primaryKey||info.keySpace!=expected.keySpace||info.versioned)
        throw std::invalid_argument("unsupported fiber measurement format; explicitly convert to compact format 4");
}
}
static std::vector<CompactFiberConstraint> readCompactFiberConstraintGraphImpl(const ObjectCollection& store,const CollectionTxn& txn,size_t workers,
    const std::function<void(const char*,size_t,size_t)>& progress,std::vector<FiberGraphNodeColumn>* nodes=nullptr,
    std::string* targetVolume=nullptr){
    using namespace detail;if(!workers)throw std::invalid_argument("graph read workers must be positive");
    const auto parts=CollectionScanAccess::partitions(store,txn);
    struct Edge {FiberGraphUuid a,b;int state=0,kind=-1,measurementOwner=INT_MAX;};
    using EdgeMap=std::pmr::unordered_map<FiberGraphUuid,Edge,UuidHash>;
    struct Part {
        std::unique_ptr<std::pmr::monotonic_buffer_resource> arena=std::make_unique<std::pmr::monotonic_buffer_resource>();
        std::unique_ptr<EdgeMap> edges=std::make_unique<EdgeMap>(arena.get());
        std::unordered_map<FiberGraphUuid,int,UuidHash> states;
        std::optional<ExtensionTableInfo> schema;std::vector<CompactFiberConstraint> rows;
    };
    std::vector<Part> data(parts.size());std::optional<ExtensionTableInfo> schema;
    for(size_t n=0;n<parts.size();++n)if(auto info=extensionTable(parts[n].db,kFiberConstraintTable)){
        requireFormat(*info);
        if(schema&&(schema->extension!=info->extension||schema->formatVersion!=info->formatVersion||schema->columns!=info->columns||
            schema->primaryKey!=info->primaryKey||schema->keySpace!=info->keySpace||
            nlohmann::json::parse(schema->provisioning)!=nlohmann::json::parse(info->provisioning)))
            throw std::invalid_argument("incompatible fiber measurement definitions");
        schema=info;data[n].schema=std::move(info);
    }
    if(!schema)throw std::invalid_argument("store has no fiber constraint measurements");
    std::string volume;std::optional<std::vector<FiberGraphUuid>> selected;
    if(nodes){
        const auto provisioning=nlohmann::json::parse(schema->provisioning);
        if(!provisioning.contains("coordinate_space")||!provisioning["coordinate_space"].is_string())
            throw std::invalid_argument("compact format-4 graph must name its measurement volume");
        const auto frame=provisioning["coordinate_space"].get<std::string>();
        volume=frame.substr(frame.find_last_of('/')==std::string::npos?0:frame.find_last_of('/')+1);
        if(volume.empty())throw std::invalid_argument("empty fiber measurement volume");
        const auto runTable=fiberGraphSolveTable(FiberGraphTable::Runs);
        const auto fiberTable=fiberGraphSolveTable(FiberGraphTable::FiberResults);
        const auto tables=store.extensionTables(txn);
        if(std::any_of(tables.begin(),tables.end(),[&](const auto& t){return t.name==runTable.name;})){
            std::optional<ExtensionValue> extraction;
            store.scanExtensionRows(txn,runTable.name,[&](const ExtensionRow& row){
                if(std::get<std::string>(row[1])=="constraint_extraction"){
                    if(extraction)throw std::invalid_argument("select one extraction graph, not multiple independent graphs");
                    extraction=row[0];
                }return true;
            });
            if(extraction){
                selected.emplace();
                store.scanExtensionRows(txn,fiberTable.name,[&](const ExtensionRow& row){
                    const auto& key=std::get<std::vector<uint8_t>>(row[0]);
                    if(key.size()!=16)throw std::invalid_argument("invalid extraction fiber UUID");
                    FiberGraphUuid uuid;std::copy(key.begin(),key.end(),uuid.begin());selected->push_back(uuid);return true;
                },{},{{"run_uuid",*extraction}});
                std::sort(selected->begin(),selected->end());
            }
        }
        if(targetVolume)*targetVolume=volume;
    }
    std::mutex progressMutex;std::atomic<size_t> readEdges{0},readRows{0};
    const auto report=[&](const char* phase,size_t count,size_t node){if(progress){std::lock_guard lock(progressMutex);progress(phase,count,node);}};
    parallelFor(parts.size(),workers,[&](size_t n){auto* db=parts[n].db;auto& p=data[n];
        Statement objects(db,"SELECT uuid,state FROM objects");while(objects.row())p.states.emplace(objects.blob16(0),int(objects.i64(1)));
        Statement count(db,"SELECT count(*) FROM edges");count.row();p.edges->reserve(count.i64(0));
        Statement edges(db,"SELECT uuid,a_uuid,b_uuid,"+edgeReadColumn(db,"state")+",kind FROM edges");size_t local=0;
        while(edges.row()){
            p.edges->emplace(edges.blob16(0),Edge{edges.blob16(1),edges.blob16(2),int(edges.i64(3)),int(edges.i64(4))});
            if(++local%1000000==0)report("endpoints",readEdges.fetch_add(1000000)+1000000,n);
        }report("endpoints",readEdges.fetch_add(local%1000000)+local%1000000,n);
    });
    const auto eligible=[&](size_t n,uint64_t hash,const char* space){for(const auto& r:parts[n].uuidRoutes)if(r.keySpace==space)return hash%r.divisor==r.remainder;return true;};
    const auto candidates=[&](const char* space,bool edges){size_t period=1;
        for(const auto& p:parts)for(const auto& r:p.uuidRoutes)if(r.keySpace==space){
            if(r.divisor>4096||period/std::gcd(period,size_t(r.divisor))>4096/r.divisor){period=1;goto arbitrary;}
            period=std::lcm(period,size_t(r.divisor));
        }
        arbitrary:std::vector<std::vector<size_t>> out(period);
        for(size_t h=0;h<period;++h)for(size_t n=0;n<data.size();++n)
            if((edges?!data[n].edges->empty():!data[n].states.empty())&&(period==1||eligible(n,h,space)))out[h].push_back(n);
        return out;
    };
    const auto edgeCandidates=candidates("edges.uuid",true),objectCandidates=candidates("objects.uuid",false);
    const auto edgeOwner=[&](const FiberGraphUuid& uuid)->std::pair<size_t,Edge*>{const auto hash=UuidHash{}(uuid);
        for(auto n:edgeCandidates[hash%edgeCandidates.size()])if(eligible(n,hash,"edges.uuid")){
            auto it=data[n].edges->find(uuid);if(it!=data[n].edges->end())return {n,&it->second};
        }return {data.size(),nullptr};
    };
    const auto objectOwner=[&](const FiberGraphUuid& uuid){const auto hash=UuidHash{}(uuid);
        for(auto n:objectCandidates[hash%objectCandidates.size()])if(eligible(n,hash,"objects.uuid")){
            auto it=data[n].states.find(uuid);if(it!=data[n].states.end())return std::pair{n,it->second};
        }return std::pair{data.size(),-1};
    };
    const auto active=[&](const FiberGraphUuid& uuid){return objectOwner(uuid).second==0;};
    if(nodes){
        const auto transforms=store.volumeTransforms(txn);
        std::vector<std::vector<FiberGraphNodeColumn>> batches(parts.size());
        parallelFor(parts.size(),workers,[&](size_t n){
            auto* db=parts[n].db;auto& output=batches[n];
            Statement rows(db,"SELECT o.uuid,f.length,o.coordinate_volume,o.coordinate_scale,o.storage_ref,o.geometry_manifest IS NOT NULL "
                "FROM objects o LEFT JOIN fibers f ON f.object_id=o.id WHERE o.kind='fiber' AND o.state=0 ORDER BY o.uuid");
            while(rows.row()){
                const auto uuid=rows.blob16(0);
                if(objectOwner(uuid).first!=n||(selected&&!std::binary_search(selected->begin(),selected->end(),uuid)))continue;
                const auto source=rows.text(2);const auto scale=rows.f64(3);
                if(rows.isNull(2)||source.empty()||rows.isNull(3)||!std::isfinite(scale)||scale<=0)
                    throw std::invalid_argument("fiber has no valid coordinate frame");
                double length;
                if(source==volume&&!rows.isNull(1))length=rows.f64(1)*(1./scale);
                else{
                    const auto mapping=source==volume?std::optional(VolumeAffine::scale(1./scale)):
                        frameMapping(nlohmann::json{{"volume_uuid",source},{"scale_factor",scale}},volume,&transforms);
                    if(!mapping)throw std::invalid_argument("fiber is not mapped to the measurement volume");
                    if(rows.isNull(4)||!rows.i64(5))throw std::invalid_argument("mapped fiber has no geometry");
                    length=transformedLength(db,readGeometryTokenColumn(rows,4),*mapping,"fiber");
                }
                if(!std::isfinite(length)||length<0)throw std::invalid_argument("invalid measured fiber length");
                output.push_back({uuid,length});
            }
        });
        for(auto& batch:batches)nodes->insert(nodes->end(),std::make_move_iterator(batch.begin()),std::make_move_iterator(batch.end()));
        std::sort(nodes->begin(),nodes->end(),[](const auto& a,const auto& b){return a.uuid<b.uuid;});
        if(std::adjacent_find(nodes->begin(),nodes->end(),[](const auto& a,const auto& b){return a.uuid==b.uuid;})!=nodes->end())
            throw std::invalid_argument("duplicate selected graph fiber");
        if(selected&&nodes->size()!=selected->size())throw std::invalid_argument("extraction selects missing or inactive fibers");
        report("fibers",nodes->size(),parts.size());
    }
    parallelFor(parts.size(),workers,[&](size_t n){if(!data[n].schema)return;auto* db=parts[n].db;auto& rows=data[n].rows;
        Statement count(db,std::string("SELECT count(*) FROM ")+kFiberConstraintTable);count.row();rows.reserve(count.i64(0));
        Statement scan(db,std::string("SELECT edge_uuid,arc_a,arc_b,normal_a,normal_b,connector,winding_distance FROM ")+kFiberConstraintTable);size_t local=0;
        while(scan.row()){
            CompactFiberConstraint r;r.uuid=scan.blob16(0);r.measurementPartition=n;const auto [owner,edge]=edgeOwner(r.uuid);(void)owner;
            if(!edge||edge->state!=0||(edge->kind!=0&&edge->kind!=1)||!active(edge->a)||!active(edge->b))
                throw std::invalid_argument("measurement references missing/inactive/wrong-kind edge");
            r.a=edge->a;r.b=edge->b;r.parallel=edge->kind==0;
            std::atomic_ref<int> winning(edge->measurementOwner);int old=winning.load();while(old>int(n)&&!winning.compare_exchange_weak(old,int(n))){}
            r.arcA=decodeFiberArc(blob(scan,1,4));r.arcB=decodeFiberArc(blob(scan,2,4));
            const auto a=blob(scan,3,3),b=blob(scan,4,3),c=blob(scan,5,3);
            r.normalA=decodeFiberDirection(a);r.normalB=decodeFiberDirection(b);r.connector=decodeFiberDirection(c);
            r.validA=a[2]!=0;r.validB=b[2]!=0;r.validConnector=c[2]!=0;
            const auto distance=blob(scan,6,1);if(distance[0]){r.windingDistance=decodeFiberWindingDistance(distance);rows.push_back(r);}
            if(++local%1000000==0)report("measurements",readRows.fetch_add(1000000)+1000000,n);
        }report("measurements",readRows.fetch_add(local%1000000)+local%1000000,n);
    });
    parallelFor(parts.size(),workers,[&](size_t n){auto& rows=data[n].rows;std::erase_if(rows,[&](const auto& r){return edgeOwner(r.uuid).second->measurementOwner!=int(n);});});
    report("resolved",readRows.load(),parts.size());
    parallelFor(data.size(),workers,[&](size_t n){auto& p=data[n];p.edges.reset();p.arena.reset();p.states.clear();p.states.rehash(0);});
    report("indexes_released",readRows.load(),parts.size());
    size_t total=0;for(const auto& p:data)total+=p.rows.size();std::vector<CompactFiberConstraint> result;result.reserve(total);
    for(auto& p:data){result.insert(result.end(),std::make_move_iterator(p.rows.begin()),std::make_move_iterator(p.rows.end()));std::vector<CompactFiberConstraint>().swap(p.rows);}
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.uuid<b.uuid;});
    for(size_t i=0;i<result.size();++i){if(i&&result[i-1].uuid==result[i].uuid)throw std::invalid_argument("duplicate compact edge UUID");result[i].runtimeIndex=i;}
    report("complete",result.size(),parts.size());return result;
}
std::vector<CompactFiberConstraint> readCompactFiberConstraintGraph(const ObjectCollection& store,const CollectionTxn& txn,size_t workers,
    const std::function<void(const char*,size_t,size_t)>& progress){
    return readCompactFiberConstraintGraphImpl(store,txn,workers,progress);
}

FiberGraphColumns readFiberGraphColumns(const ObjectCollection& store,const CollectionTxn& txn,size_t workers,
    const std::function<void(const char*,size_t,size_t)>& progress){
    FiberGraphColumns out;std::vector<FiberGraphNodeColumn> nodes;
    auto records=readCompactFiberConstraintGraphImpl(store,txn,workers,progress,&nodes,&out.volume);
    const size_t n=nodes.size(),m=records.size();
    if(progress)progress("column_projection",0,workers);
    out.nodes=n;out.events=m;
    out.fiberUuids=std::make_unique_for_overwrite<FiberGraphUuid[]>(n);out.lengths=std::make_unique_for_overwrite<double[]>(n);
    out.edgeUuids=std::make_unique_for_overwrite<FiberGraphUuid[]>(m);out.eventIds=std::make_unique_for_overwrite<int64_t[]>(m);
    out.endpoints=std::make_unique_for_overwrite<std::array<int64_t,2>[]>(m);out.arcs=std::make_unique_for_overwrite<std::array<double,2>[]>(m);
    out.normalA=std::make_unique_for_overwrite<std::array<double,3>[]>(m);out.normalB=std::make_unique_for_overwrite<std::array<double,3>[]>(m);
    out.connector=std::make_unique_for_overwrite<std::array<double,3>[]>(m);out.distance=std::make_unique_for_overwrite<double[]>(m);
    out.parallel=std::make_unique_for_overwrite<uint8_t[]>(m);out.validA=std::make_unique_for_overwrite<uint8_t[]>(m);
    out.validB=std::make_unique_for_overwrite<uint8_t[]>(m);out.validConnector=std::make_unique_for_overwrite<uint8_t[]>(m);
    parallelFor((n+4095)/4096,workers,[&](size_t task){
        for(size_t i=task*4096;i<std::min(n,(task+1)*4096);++i){out.fiberUuids[i]=nodes[i].uuid;out.lengths[i]=nodes[i].length;}
    });
    const auto index=[&](const FiberGraphUuid& key){
        const auto found=std::lower_bound(nodes.begin(),nodes.end(),key,[](const auto& row,const auto& value){return row.uuid<value;});
        if(found==nodes.end()||found->uuid!=key)throw std::invalid_argument("constraint refers to a fiber outside the extracted graph");
        return int64_t(found-nodes.begin());
    };
    parallelFor((m+65535)/65536,workers,[&](size_t task){
        for(size_t i=task*65536;i<std::min(m,(task+1)*65536);++i){const auto& r=records[i];
            out.edgeUuids[i]=r.uuid;out.eventIds[i]=int64_t(r.runtimeIndex);out.endpoints[i]={index(r.a),index(r.b)};
            out.arcs[i]={r.arcA,r.arcB};out.normalA[i]=r.normalA;out.normalB[i]=r.normalB;out.connector[i]=r.connector;
            out.distance[i]=r.windingDistance;out.parallel[i]=r.parallel;out.validA[i]=r.validA;out.validB[i]=r.validB;out.validConnector[i]=r.validConnector;
        }
    });
    if(progress)progress("columns_complete",m,workers);
    return out;
}
std::function<void()> prepareFiberConstraintMeasurementScan(const ObjectCollection& store,const CollectionTxn& txn,
    std::span<const CompactFiberConstraint> records,size_t workers,const std::function<void(FiberMeasurementBatch&&)>& visit){
    using namespace detail;if(!workers||!visit)throw std::invalid_argument("invalid measurement stream");const auto parts=CollectionScanAccess::partitions(store,txn);
    return [parts,records,workers,visit]{parallelFor(parts.size(),workers,[&](size_t n){auto* db=parts[n].db;
        const auto schema=extensionTable(db,kFiberConstraintTable);if(!schema)return;requireFormat(*schema);FiberMeasurementBatch batch;batch.reserve(4096);
        scanExtensionRows(db,kFiberConstraintTable,[&](const ExtensionRow& row){const auto& bytes=std::get<std::vector<uint8_t>>(row.at(0));
            if(bytes.size()!=16)throw std::invalid_argument("invalid measurement UUID");FiberGraphUuid uuid;std::copy(bytes.begin(),bytes.end(),uuid.begin());
            const auto found=std::lower_bound(records.begin(),records.end(),uuid,[](const auto& r,const auto& v){return r.uuid<v;});
            if(found==records.end()||found->uuid!=uuid||found->measurementPartition!=n)return true;
            batch.emplace_back(size_t(found-records.begin()),row);if(batch.size()==4096){visit(std::move(batch));batch.clear();batch.reserve(4096);}return true;
        },{});if(!batch.empty())visit(std::move(batch));
    });};
}
std::vector<FiberConstraintRecord> readFiberConstraintGraph(const ObjectCollection& store,const CollectionTxn& txn){
    const auto compact=readCompactFiberConstraintGraph(store,txn);std::unordered_map<std::string,EdgeRecord> edges;
    for(const auto* kind:{"parallel","perpendicular"})for(auto& e:store.edgesByKind(txn,kind))edges.emplace(e.record.uuid,std::move(e.record));
    std::vector<FiberConstraintRecord> result(compact.size());
    for(size_t i=0;i<compact.size();++i){const auto& c=compact[i];auto& r=result[i];auto found=edges.find(fiberGraphUuid(c.uuid));
        if(found==edges.end())throw std::invalid_argument("missing compact edge");r.edge=std::move(found->second);r.runtimeIndex=i;
        r.a=fiberGraphUuid(c.a);r.b=fiberGraphUuid(c.b);r.arcA=c.arcA;r.arcB=c.arcB;r.normalA=c.normalA;r.normalB=c.normalB;r.connector=c.connector;
        r.parallel=c.parallel;r.windingDistance=c.windingDistance;r.validA=c.validA;r.validB=c.validB;r.validConnector=c.validConnector;
    }
    std::mutex mutex;auto scan=prepareFiberConstraintMeasurementScan(store,txn,compact,32,[&](FiberMeasurementBatch&& batch){std::lock_guard lock(mutex);
        for(auto& [i,row]:batch)result[i].measurement={kFiberConstraintTable,std::move(row)};});scan();return result;
}
} // namespace vc::objects
