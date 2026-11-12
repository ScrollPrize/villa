#include "TestSupport.hpp"
#include "vc_object_store/objects/FiberConstraintGraph.hpp"
#include "vc_object_store/objects/InitialImport.hpp"
#include <future>
#include <mutex>
#include <iostream>
using namespace vc::objects;

void compare(ObjectCollection& store) {
    auto txn=store.read();const auto rich=readFiberConstraintGraph(store,txn);
    for(size_t workers:{1,4}){
        const auto rows=readCompactFiberConstraintGraph(store,txn,workers);
        require(rows.size()==rich.size(),"compact graph event count");
        for(size_t i=0;i<rows.size();++i){const auto& a=rows[i];const auto& b=rich[i];
            require(fiberGraphUuid(a.uuid)==b.edge.uuid&&fiberGraphUuid(a.a)==b.a&&fiberGraphUuid(a.b)==b.b&&
                a.runtimeIndex==b.runtimeIndex&&a.normalA==b.normalA&&a.normalB==b.normalB&&a.connector==b.connector&&
                a.arcA==b.arcA&&a.arcB==b.arcB&&a.parallel==b.parallel&&a.windingDistance==b.windingDistance&&
                a.validA==b.validA&&a.validB==b.validB&&a.validConnector==b.validConnector,"compact projection exact");
        }
        std::vector<bool> seen(rows.size());std::mutex mutex;
        auto scan=prepareFiberConstraintMeasurementScan(store,txn,rows,workers,[&](FiberMeasurementBatch&& batch){
            std::lock_guard lock(mutex);
            for(const auto& [i,row]:batch){require(!seen[i]&&row==rich[i].measurement.values,"stream preserves complete winning measurement");seen[i]=true;}
        });
        std::async(std::launch::async,scan).get();
        for(bool found:seen)require(found,"stream visited every record");
    }
}
void run(bool packed,bool legacy) {
    TemporaryDirectory dir;InitialImportOptions options;options.store.author="compact-graph-test";options.shards.resize(4);
    const auto stride=INT64_MAX/4;
    for(int i=0;i<4;++i)options.shards[i].objectIds={1+i*stride,(i+1)*stride};
    if(packed){options.packed.emplace();options.packed->memoryFirst=true;options.packed->disposableBulk=true;options.packed->backgroundFlush=false;}
    auto schema=fiberConstraintExtension({util::newUuid(),{},{},"sample/volume"});
    if(legacy)schema.formatVersion=3;
    options.extensions.push_back(schema);
    std::vector<std::string> fibers;for(int i=0;i<8;++i)fibers.push_back(util::newUuid());
    std::vector<InitialCollectionEdge> edges;
    for(int i=0;i<48;++i){InitialCollectionEdge e;e.record.uuid=util::newUuid();e.record.kind=i%2?"parallel":"perpendicular";
        auto a=fibers[i%8],b=fibers[(i+1)%8];const bool reverse=b<a;
        e.record.aUuid=std::min(a,b);e.record.bUuid=std::max(a,b);e.a={"",e.record.aUuid};e.b={"",e.record.bUuid};
        e.record.aAnchor.position={1.,2.,3.};e.record.bAnchor.position={4.,5.,6.};
        FiberConstraintData data;data.values={1.,2.,.3,.7,1.25,.75,.5};
        data.normals[0].valid=true;data.normals[0].xyz={1.,0.,0.};data.normals[1].valid=i%3!=0;data.normals[1].xyz={0.,1.,0.};
        data.measuredPoints=std::array<std::array<double,3>,2>{{{7.,8.,9.},{10.,11.,12.}}};
        auto row=fiberConstraintRow(e.record.uuid,legacy?i*2:i,i%8,(i+1)%8,reverse,std::move(data));
        e.extensions.push_back(std::move(row));edges.push_back(std::move(e));
    }
    const auto result=importInitialCollection(dir.path/"graph",options,[&](size_t shard,auto& writer){
        std::vector<InitialObjectRecord> records;
        for(const auto& id:fibers)if(uuidShardIndex(id,4)==shard){InitialObjectRecord r;r.record.uuid=id;r.record.kind="fiber";r.record.fiber.emplace();records.push_back(std::move(r));}
        writer.appendObjects(records);
    },[&](size_t shard,auto& writer){bool sent=false;writer.importEdges([&](auto& batch){if(sent)return false;sent=true;
        for(const auto& e:edges)if(uuidShardIndex(e.record.uuid,4)==shard)batch.push_back(e);return !batch.empty();});});
    auto store=ObjectCollection::open(result.path,OpenOptions{.readOnly=true});
    if(legacy){auto r=store.read();mustFail([&]{readCompactFiberConstraintGraph(store,r,4);});return;}
    compare(store);
    auto overlay=ObjectCollection::create(dir.path/"overlay",{result.path},OpenOptions{.history=HistoryMode::Off,.author="compact-graph-test"});
    {auto w=overlay.write();overlay.createExtensionTable(w,schema);auto row=edges[0].extensions[0].values;
        size_t column=0;while(schema.columns[column].name!="normal_a")++column;row[column]=encodeFiberDirection({-.25,1.,0.});
        overlay.appendExtensionRows(w,schema.name,std::span(&row,1));w.commit();}
    compare(overlay);
    {auto w=overlay.write();overlay.removeEdge(w,{"",edges[1].record.uuid});w.commit();}
    auto r=overlay.read();mustFail([&]{readCompactFiberConstraintGraph(overlay,r,4);});
    mustFail([&]{readFiberConstraintGraph(overlay,r);});
}
int main()try{for(bool packed:{false,true})for(bool legacy:{false,true})run(packed,legacy);}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
