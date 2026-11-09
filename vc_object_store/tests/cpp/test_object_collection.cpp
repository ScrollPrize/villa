#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/objects/PackedStore.hpp"
#include "vc_object_store/objects/StoreConversion.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/Database.hpp"
#include <iostream>
#include <cstring>
using namespace vc::objects;
using namespace vc::objects::detail;
ObjectRecord fiber(){ObjectRecord r;r.kind="fiber";r.fiber.emplace();return r;}
int main()try{
    TemporaryDirectory dir;
    auto options=OpenOptions{.history=HistoryMode::Off,.spatialIndex=false,.author="test"};
    const auto basePath=dir.path/"base.vcobj",overlayPath=dir.path/"overlay.vcobj";
    ObjectKey a,b;std::vector<uint8_t> geometry;
    {
        auto base=ObjectStore::create(basePath,options);auto w=base.write();
        a={base.uuid(w),base.upsertObject(w,fiber())};b={base.uuid(w),base.upsertObject(w,fiber())};
        vc::geometry::Fiber shape;for(int i=0;i<60000;++i)shape.linePoints.push_back({double(i),2,3});
        auto rev=base.putGeometry(w,a.objectUuid,vc::geometry::encode(shape));
        w.commit();auto read=base.read();geometry=base.readGeometryArray(read,rev,"line_points");read.commit();
        mustFail([&]{ObjectCollection::create(overlayPath,{basePath},options);});
        auto seal=base.write();base.seal(seal);seal.commit();
    }
    auto overlay=ObjectCollection::create(overlayPath,{basePath},options);
    mustFail([&]{ObjectStore::open(overlayPath,OpenOptions{.readOnly=true});});
    {
        auto r=overlay.read();require(overlay.readGeometryArray(r,a,"line_points")==geometry,"inherited geometry exact");
        require(overlay.provisioning(r).commitSequence>overlay.members()[1].commitSequence,"derived commit sequence above parent");
        require(overlay.objectById(r,a.storeUuid,1)->key==a,"numeric base routing");
    }
    ObjectKey newFiber,edge;
    {
        auto w=overlay.write();overlay.setMeta(w,a,"classification","\"H\"");newFiber=overlay.upsertObject(w,fiber());
        EdgeRecord e;e.kind="fiber_constraint";e.aAnchor.wholeGeometry=true;e.bAnchor.wholeGeometry=true;
        edge=overlay.addEdge(w,a,newFiber,e);w.commit();
        auto r=overlay.read();require(overlay.edgesOf(r,a).size()==1,"cross store adjacency");
        require(overlay.edgesOf(r,newFiber)[0].key==edge,"reverse adjacency");
        require(overlay.object(r,a)->record.meta.find("H")!=std::string::npos,"metadata override");
        require(overlay.readGeometryArray(r,a,"line_points")==geometry,"metadata override borrows chunks");
        size_t count=0;overlay.scanObjects(r,[&](auto&){++count;return true;});require(count==3,"references do not duplicate objects");
    }
    {
        auto base=ObjectStore::open(basePath,OpenOptions{.readOnly=true});auto r=base.read();
        require(base.objectByUuid(r,a.objectUuid)->meta.find("classification")==std::string::npos,"base not modified");
        auto w=overlay.write();overlay.setMeta(w,a,"rollback","true");
    }
    {
        auto r=overlay.read();require(overlay.object(r,a)->record.meta.find("rollback")==std::string::npos,"overlay rollback");
        auto w=overlay.write();mustFail([&]{overlay.deleteObject(w,a);});overlay.seal(w);w.commit();
    }
    auto child=ObjectCollection::create(dir.path/"child.vcobj",{overlayPath},options);
    {
        auto w=child.write();child.setMeta(w,a,"classification","\"V\"");child.removeEdge(w,edge);child.deleteObject(w,a);w.commit();
        auto r=child.read();require(!child.object(r,a),"inherited deletion tombstone");require(child.edgesOf(r,newFiber).empty(),"inherited edge tombstone");
        size_t count=0;child.scanObjects(r,[&](auto&){++count;return true;});require(count==2,"deleted inherited geometry not enumerated");
    }
    {
        auto r=overlay.read();require(overlay.object(r,a).has_value()&&overlay.edgesOf(r,a).size()==1,"child changes do not change sealed parent");
    }
    {
        auto raw=ObjectStore::open(dir.path/"child.vcobj",OpenOptions{.physicalPartitionOnly=true,.readOnly=true});
        auto r=raw.read();auto* sql=ObjectStoreImplAccess::conn(r);
        Statement object(sql,"SELECT state FROM objects WHERE uuid=?",parseUuid(a.objectUuid));
        require(object.row()&&object.i64(0)==2,"inherited object deletion must use its original UUID row");
        Statement relation(sql,"SELECT state FROM edges WHERE uuid=?",parseUuid(edge.objectUuid));
        require(relation.row()&&relation.i64(0)==2,"inherited edge deletion must use its original UUID row");
        Statement redundant(sql,"SELECT count(*) FROM sqlite_schema WHERE name IN "
            "('collection_refs','collection_edge_tombstones','external_object_keys','endpoint_keys')");
        require(redundant.row()&&redundant.i64(0)==0,"overlay must not add alias or tombstone tables");
    }
    for(const auto history:{HistoryMode::Off,HistoryMode::On}){
        auto opt=options;opt.history=history;auto file=dir.path/(history==HistoryMode::Off?"geometry-off.vcobj":"geometry-on.vcobj");
        auto c=ObjectCollection::create(file,{basePath},opt);
        std::string initial;{auto r=c.read();initial=c.currentGeometry(r,a)->uuid;}
        GeometryArrayEdit change{"line_points",0,std::vector<uint8_t>(8)};double value=123;std::memcpy(change.bytes.data(),&value,8);
        {auto w=c.write();c.editGeometry(w,a,{change},initial);w.commit();}
        auto expected=geometry;std::memcpy(expected.data(),&value,8);
        {auto r=c.read();require(c.readGeometryArray(r,a,"line_points")==expected,"linked sparse geometry edit exact");}
        {auto raw=ObjectStore::open(file,OpenOptions{.physicalPartitionOnly=true,.readOnly=true});auto r=raw.read();
            Statement bytes(ObjectStoreImplAccess::conn(r),"SELECT count(*) FROM geometry_chunks WHERE data IS NULL");bytes.row();require(bytes.i64(0)==0,"linked chunks create no placeholder rows");
            mustFail([&]{auto records=raw.objectsByKind(r,"fiber");auto rev=raw.currentGeometry(r,records[0].uuid);raw.readGeometryArray(r,rev->uuid,"line_points");});}
        {auto w=c.write();c.seal(w);w.commit();}
        auto deeper=ObjectCollection::create(dir.path/(history==HistoryMode::Off?"deeper-off.vcobj":"deeper-on.vcobj"),{file},opt);
        {auto r=deeper.read();require(deeper.readGeometryArray(r,a,"line_points")==expected,"linked geometry resolves through deeper overlay");}
    }
    {
        const auto file=dir.path/"linked-policy.vcobj";auto expected=geometry;double value=111;std::memcpy(expected.data(),&value,8);
        {auto c=ObjectCollection::create(file,{basePath},options);std::string token;{auto r=c.read();token=c.currentGeometry(r,a)->uuid;}
            GeometryArrayEdit change{"line_points",0,std::vector<uint8_t>(8)};std::memcpy(change.bytes.data(),&value,8);
            auto w=c.write();c.editGeometry(w,a,{change},token);w.commit();}
        for(auto mode:{HistoryMode::On,HistoryMode::Off}){
            changeHistoryMode(file,mode);auto c=ObjectCollection::open(file,OpenOptions{.readOnly=true});auto r=c.read();
            require(c.readGeometryArray(r,a,"line_points")==expected,"history policy preserves linked geometry");
            require(c.provisioning(r).history==mode,"linked history policy persisted");
        }
    }
    // Two independent workers reserve disjoint ranges under one publication.
    const auto logical=util::newUuid(),lineage=util::newUuid();std::vector<std::filesystem::path> shards;std::vector<ObjectKey> keys;
    for(int i=0;i<2;++i){auto path=dir.path/("shard"+std::to_string(i)+".vcobj");shards.push_back(path);
        auto opt=options;opt.creationStoreUuid=logical;opt.creationGeneration=lineage;opt.initialCommitSequence=9;opt.objectIdRange={1+i*100,100+i*100};opt.edgeIdRange=opt.objectIdRange;
        auto db=ObjectStore::create(path,opt);auto w=db.write();keys.push_back({logical,db.upsertObject(w,fiber())});db.seal(w);w.commit();}
    auto collection=ObjectCollection::publishShards(dir.path/"sharded.vcobj",shards,options);
    {
        auto r=collection.read();require(collection.provisioning(r).commitSequence==10&&collection.provisioning(r).generation==lineage,"common reserved publication lineage and commit sequence");
        require(collection.objectById(r,logical,101)->key==keys[1],"numeric range routes to second shard");
        require(!collection.objectById(r,logical,99),"reserved unpopulated ID remains absent");
        size_t count=0;collection.scanObjects(r,[&](auto&){++count;return true;});require(count==2,"all partitions enumerated");
    }
    mustFail([&]{ObjectCollection::publishShards(dir.path/"overlap.vcobj",{shards[0],shards[0]},options);});
    require(!std::filesystem::exists(dir.path/"overlap.vcobj"),"invalid shard publication invisible");
    for(auto mode:{RoutingKind::Modulo,RoutingKind::Bits,RoutingKind::HashV1}){
        const auto owner=util::newUuid(),generation=util::newUuid();std::vector<std::filesystem::path> parts;std::vector<KeyRoute> rules;std::map<int64_t,ObjectKey> ids,edgeIds;
        for(int partition=0;partition<2;++partition){auto p=dir.path/("route-"+std::to_string(int(mode))+"-"+std::to_string(partition));parts.push_back(p);
            auto opt=options;opt.creationStoreUuid=owner;opt.creationGeneration=generation;opt.objectIdRange={1,100};opt.edgeIdRange={1,100};
            auto db=ObjectStore::create(p,opt);auto w=db.write();const auto physical=db.databaseIdentity(w).uuid;
            KeyRoute rule{"objects",mode,1,100,2,uint64_t(partition),mode==RoutingKind::Bits?1u:0u,physical};
            std::vector<std::string> endpoints;
            for(int64_t id=1;id<=12;++id)if(rule.matches(id)){
                const auto object=db.insertObjectWithId(w,fiber(),id);ids.emplace(id,ObjectKey{owner,object});endpoints.push_back(object);}
            require(endpoints.size()>=2,"custom edge route fixture needs two endpoints");
            for(int64_t id=1;id<=12;++id)if(rule.matches(id)){
                EdgeRecord edge;edge.kind="routed";edge.aUuid=endpoints[0];edge.bUuid=endpoints[1];
                edgeIds.emplace(id,ObjectKey{owner,db.insertEdgeWithId(w,edge,id)});
                mustFail([&]{db.insertEdgeWithId(w,edge,id);});}
            rules.push_back(rule);rule.keySpace="edges";rules.push_back(rule);db.seal(w);w.commit();}
        auto c=ObjectCollection::publishShards(dir.path/("routed-"+std::to_string(int(mode))),parts,options,rules);
        auto r=c.read();for(auto& [id,key]:ids)require(c.objectById(r,owner,id)->key==key,"custom routing exact");
        for(auto& [id,key]:edgeIds)require(c.edgeById(r,owner,id)->key==key,"custom edge routing exact");
    }
    // Same manifest/routing API inside the packed VFS; no absolute staging refs.
    const auto packedPath=dir.path/"packed.vcobj";const auto packedOwner=util::newUuid(),packedLineage=util::newUuid();
    {
        PackedStoreBuilder builder(packedPath);std::vector<std::filesystem::path> paths;
        for(int i=0;i<2;++i){auto p=builder.addDatabase(util::newUuid());paths.push_back(p);auto opt=options;opt.creationStoreUuid=packedOwner;opt.creationGeneration=packedLineage;
            opt.objectIdRange={1+i*100,100+i*100};opt.edgeIdRange=opt.objectIdRange;
            auto db=ObjectStore::create(p,opt);auto w=db.write();db.upsertObject(w,fiber());db.seal(w);w.commit();}
        const auto root=util::newUuid();auto rootPath=builder.addDatabase(root);
        {auto c=ObjectCollection::publishShards(rootPath,paths,options);auto r=c.read();size_t n=0;c.scanObjects(r,[&](auto&){++n;return true;});require(n==2,"unpublished packed routing");}
        builder.finalize(root);
    }
    {
        auto c=ObjectCollection::open(packedPath,OpenOptions{.readOnly=true});auto r=c.read();size_t n=0;c.scanObjects(r,[&](auto&){++n;return true;});require(n==2,"published packed routing");
    }
    std::filesystem::rename(packedPath,dir.path/"moved.vcobj");
    {
        auto c=ObjectCollection::open(dir.path/"moved.vcobj",OpenOptions{.readOnly=true});auto r=c.read();require(c.objectById(r,packedOwner,101).has_value(),"packed references survive move");
    }
    // Identity mismatch must never silently resolve to the wrong file.
    const auto brokenPath=dir.path/"broken.vcobj";
    {auto c=ObjectCollection::create(brokenPath,{basePath},options);}
    {auto db=ObjectStore::open(brokenPath,OpenOptions{.physicalPartitionOnly=true,.author="test"});auto w=db.write();
        execute(ObjectStoreImplAccess::conn(w),"UPDATE collection_members SET commit_sequence=commit_sequence+1");w.commit();}
    mustFail([&]{ObjectCollection::open(brokenPath,OpenOptions{.readOnly=true});});
    const auto brokenLineage=dir.path/"broken-lineage.vcobj";
    {auto c=ObjectCollection::create(brokenLineage,{basePath},options);}
    {auto db=ObjectStore::open(brokenLineage,OpenOptions{.physicalPartitionOnly=true,.author="test"});auto w=db.write();
        execute(ObjectStoreImplAccess::conn(w),"UPDATE collection_members SET generation=?",parseUuid(util::newUuid()));w.commit();}
    mustFail([&]{ObjectCollection::open(brokenLineage,OpenOptions{.readOnly=true});});
    std::cout<<"collections passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
