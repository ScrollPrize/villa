#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "internal/Database.hpp"
#include "internal/RecordSql.hpp"
#include <future>
#include <iostream>

using namespace vc::objects;
using namespace vc::objects::detail;
namespace {
ObjectRecord fiber() { ObjectRecord r; r.kind="fiber"; r.fiber.emplace(); r.meta=R"({"foo":1})"; return r; }
int64_t generation(ObjectCollection& collection) { auto r=collection.read(); return collection.provisioning(r).generation; }
}
int main() try {
    TemporaryDirectory dir;
    OpenOptions options;options.author="audit";options.history=HistoryMode::Off;options.spatialIndex=false;options.readerPoolSize=1;
    const auto basePath=dir.path/"base.vcobj", overlayPath=dir.path/"overlay.vcobj";
    ObjectKey a,b,c;std::vector<uint8_t> expected;
    {
        auto base=ObjectStore::create(basePath,options);auto w=base.write();const auto owner=base.uuid(w);
        a={owner,base.upsertObject(w,fiber())};b={owner,base.upsertObject(w,fiber())};c={owner,base.upsertObject(w,fiber())};
        vc::geometry::Fiber shape;for(int i=0;i<60000;++i)shape.linePoints.push_back({double(i),2,3});
        const auto revision=base.putGeometry(w,a.objectUuid,vc::geometry::encode(shape));base.seal(w);w.commit();
        auto r=base.read();expected=base.readGeometryArray(r,revision,"line_points");
    }
    auto overlay=ObjectCollection::create(overlayPath,{basePath},options);
    const auto initial=generation(overlay);
    {
        auto w=overlay.write();overlay.setMeta(w,a,"foo","1.0");
        overlay.replaceObject(w,a,overlay.object(w,a)->record);w.commit();
    }
    require(generation(overlay)==initial,"same inherited content advanced generation");
    {
        auto raw=ObjectStore::open(overlayPath,OpenOptions{.physicalPartitionOnly=true,.readOnly=true});auto r=raw.read();
        Statement count(ObjectStoreImplAccess::conn(r),"SELECT count(*) FROM objects");count.row();require(count.i64(0)==0,"no-op materialized an inherited object");
    }
    {
        auto w=overlay.write();overlay.setMeta(w,a,"foo","2");w.commit();
    }
    const auto changed=generation(overlay);
    {
        auto w=overlay.write();overlay.setMeta(w,a,"foo","2.0");overlay.replaceObject(w,a,overlay.object(w,a)->record);w.commit();
    }
    require(generation(overlay)==changed,"same overlay content advanced generation");
    std::string owner;
    {auto r=overlay.read();owner=overlay.uuid(r);}
    {
        auto raw=ObjectStore::open(overlayPath,OpenOptions{.physicalPartitionOnly=true,.readOnly=true});auto r=raw.read();
        Statement record(ObjectStoreImplAccess::conn(r),"SELECT uuid,state FROM objects");
        require(record.row()&&formatUuid(record.blob16(0))==a.objectUuid&&record.i64(1)==0&&!record.row(),
            "changed inherited record must be stored once at its original UUID");
    }
    {auto r=overlay.read();require(overlay.object(r,{owner,a.objectUuid})->key==a,
        "the same UUID with an overlay routing hint must resolve to the same logical object");}
    const auto collision=util::nameUuid(owner,"reference:"+b.storeUuid+":"+b.objectUuid);
    {
        auto w=overlay.write();auto record=fiber();record.uuid=collision;record.name="keep-owned";overlay.upsertObject(w,record);w.commit();
        auto retry=overlay.write();overlay.setMeta(retry,b,"foo","9");retry.commit();
        auto r=overlay.read();require(overlay.object(r,{owner,collision})->record.name=="keep-owned","reference creation overwrote owned UUID");
        require(nlohmann::json::parse(overlay.object(r,b)->record.meta).at("foo")==9,
            "an unrelated former alias UUID must not prevent an inherited edit");
    }
    ObjectKey edge;
    {
        auto w=overlay.write();EdgeRecord record;record.kind="indexed";record.aAnchor.points.push_back({"line_points",{2}});
        record.aAnchor.metadataKeys={"foo"};record.bAnchor.wholeGeometry=true;
        edge=overlay.insertEdgeWithId(w,a,c,record,17);w.commit();
    }
    {auto r=overlay.read();require(overlay.edgeById(r,owner,17)->key==edge,"qualified explicit edge ID was not preserved");}
    {
        auto read=overlay.read();const auto current=overlay.edge(read,edge);
        require(current&&((current->a==a&&current->b==c)||(current->a==c&&current->b==a)),"indexed edge retains direct endpoint keys");
        require(overlay.currentGeometry(read,a)&&overlay.readGeometryArray(read,a,"line_points")==expected,
                "indexed inherited geometry resolves through its original owner");
        auto raw=ObjectStore::open(overlayPath,OpenOptions{.physicalPartitionOnly=true,.readOnly=true});auto r=raw.read();
        Statement counts(ObjectStoreImplAccess::conn(r),"SELECT (SELECT count(*) FROM objects),"
            "(SELECT count(*) FROM sqlite_schema WHERE name IN ('external_object_keys','endpoint_keys','collection_refs','collection_edge_tombstones')),"
            "(SELECT count(*) FROM objects WHERE geometry_manifest IS NOT NULL),(SELECT count(*) FROM geometry_chunks)");
        // The metadata override for a retains its existing geometry descriptor
        // in the ordinary object row, but all payload bytes remain inherited.
        require(counts.row()&&counts.i64(0)==3&&counts.i64(1)==0&&counts.i64(2)==1&&counts.i64(3)==0,
            "adding indexed edge must not copy endpoints beyond two existing overrides and the owned object");
        Statement descriptor(ObjectStoreImplAccess::conn(r),"SELECT uuid FROM objects WHERE geometry_manifest IS NOT NULL");
        require(descriptor.row()&&formatUuid(descriptor.blob16(0))==a.objectUuid&&!descriptor.row(),
            "only the explicit metadata override retains its inherited geometry descriptor");
    }
    const auto boundGeneration=generation(overlay);
    {
        auto w=overlay.write();auto revision=overlay.currentGeometry(w,a)->uuid;
        GeometryArrayEdit edit{"line_points",6,std::vector<uint8_t>(8)};double value=1000;std::memcpy(edit.bytes.data(),&value,8);
        mustFail([&]{overlay.editGeometry(w,a,{edit},revision);});w.commit();
    }
    require(generation(overlay)==boundGeneration,"rejected geometry edit advanced generation");
    {
        auto w=overlay.write();overlay.setMeta(w,a,"foo","3");mustFail([&]{w.commit();});
    }
    require(generation(overlay)==boundGeneration,"failed validation advanced generation");
    // Repeatedly return and re-lease the same root reader from another thread.
    // Resolver cleanup must happen before checkin, never after lease release.
    auto work=[&]{for(int i=0;i<24;++i){auto r=overlay.read();require(overlay.readGeometryArray(r,a,"line_points")==expected,"pooled resolver lost a pinned payload");r.commit();}};
    auto first=std::async(std::launch::async,work),second=std::async(std::launch::async,work);first.get();second.get();
    {
        auto readonly=ObjectCollection::open(overlayPath,OpenOptions{.readOnly=true});mustFail([&]{readonly.write();});
        auto r=readonly.read();require(readonly.edgesOf(r,a).size()==1,"indexed relation missing after reopen");
    }
    {
        const auto parent=dir.path/"metadata-parent.vcobj";ObjectKey endpoint;std::string inheritedEdge;
        {auto db=ObjectStore::create(parent,options);auto w=db.write();endpoint={db.uuid(w),db.upsertObject(w,fiber())};
         auto restored=*db.objectByUuid(w,endpoint.objectUuid);restored.lifecycleEpoch=util::newUuid();
         writeLifecycle(ObjectStoreImplAccess::conn(w),*objectIdByUuid(ObjectStoreImplAccess::conn(w),parseUuid(endpoint.objectUuid)),restored);
         const auto other=db.upsertObject(w,fiber());EdgeRecord edge;edge.kind="metadata";edge.aUuid=endpoint.objectUuid;edge.bUuid=other;
         edge.aAnchor.metadataKeys={"foo"};inheritedEdge=db.addEdge(w,edge);db.seal(w);w.commit();}
        auto c=ObjectCollection::create(dir.path/"metadata-overlay.vcobj",{parent},options);
        {auto w=c.write();c.setMeta(w,endpoint,"family","\"H\"");w.commit();}
        const auto before=generation(c);
        {auto w=c.write();mustFail([&]{c.setMeta(w,endpoint,"foo","9");});
         auto replacement=c.object(w,endpoint)->record;replacement.meta=R"({"foo":9})";
         mustFail([&]{c.replaceObject(w,endpoint,replacement);});w.commit();}
        require(generation(c)==before,"rejected inherited metadata change advanced generation");
        {auto w=c.write();c.removeEdge(w,{endpoint.storeUuid,inheritedEdge});c.setMeta(w,endpoint,"foo","9");w.commit();}
        auto r=c.read();require(nlohmann::json::parse(c.object(r,endpoint)->record.meta).at("foo")==9,"removed relationship still blocked endpoint change");r.commit();
        const auto dependency=c.members().at(1).databaseUuid;
        {auto raw=ObjectStore::open(dir.path/"metadata-overlay.vcobj",OpenOptions{.physicalPartitionOnly=true,.author="audit"});auto w=raw.write();
         for(const auto* keySpace:{"objects","edges"})execute(ObjectStoreImplAccess::conn(w),
             "INSERT INTO collection_routes VALUES(?,0,1,100,1,0,0,?)",std::string(keySpace),parseUuid(dependency));w.commit();}
        mustFail([&]{ObjectCollection::open(dir.path/"metadata-overlay.vcobj",OpenOptions{.readOnly=true});});
    }
    for(auto mode:{HistoryMode::On,HistoryMode::Off}){
        auto explicitOptions=options;explicitOptions.history=mode;explicitOptions.objectIdRange={10,15};explicitOptions.edgeIdRange={100,103};
        auto db=ObjectStore::create(dir.path/(mode==HistoryMode::On?"explicit-on.vcobj":"explicit-off.vcobj"),explicitOptions);
        EdgeRecord edge;edge.kind="explicit";std::string high,low;
        {
            auto w=db.write();auto* sql=ObjectStoreImplAccess::conn(w);
            edge.aUuid=db.insertObjectWithId(w,fiber(),15);edge.bUuid=db.insertObjectWithId(w,fiber(),10);
            mustFail([&]{db.upsertObject(w,fiber());}); // Out-of-order insertion cannot reopen exhaustion.
            mustFail([&]{db.insertObjectWithId(w,fiber(),15);});
            mustFail([&]{db.insertObjectWithId(w,fiber(),9);});
            auto duplicate=fiber();duplicate.uuid=edge.aUuid;mustFail([&]{db.insertObjectWithId(w,duplicate,11);});
            db.insertObjectWithId(w,fiber(),11);
            auto invalid=edge;invalid.kind.clear();mustFail([&]{db.insertEdgeWithId(w,invalid,102);});
            const auto automatic=db.addEdge(w,edge);
            require(edgeIdByUuid(sql,parseUuid(automatic))==100,"failed explicit edge insertion moved allocator");
            high=db.insertEdgeWithId(w,edge,103);
            low=db.insertEdgeWithId(w,edge,101);
            require(edgeIdByUuid(sql,parseUuid(high))==103&&edgeIdByUuid(sql,parseUuid(low))==101,"explicit edge IDs changed");
            mustFail([&]{db.insertEdgeWithId(w,edge,101);});
            mustFail([&]{db.insertEdgeWithId(w,edge,104);});
            auto sameUuid=edge;sameUuid.uuid=high;mustFail([&]{db.insertEdgeWithId(w,sameUuid,102);});
            mustFail([&]{db.addEdge(w,edge);});w.commit();
        }
        int64_t before;{auto r=db.read();before=db.provisioning(r).generation;}
        {auto rollback=db.write();db.insertEdgeWithId(rollback,edge,102);}
        {auto r=db.read();require(db.provisioning(r).generation==before&&db.edgesByKind(r,"explicit").size()==3,"explicit edge rollback leaked state");}
        {
            auto w=db.write();const auto final=db.insertEdgeWithId(w,edge,102);
            require(edgeIdByUuid(ObjectStoreImplAccess::conn(w),parseUuid(final))==102,"rolled-back edge ID was not reusable");
            mustFail([&]{db.addEdge(w,edge);});w.commit();
        }
        auto r=db.read();require(db.edgesByKind(r,"explicit").size()==4,"explicit edges not published");
        require(db.provisioning(r).generation==before+1,"explicit edge publication generation changed incorrectly");
    }
    std::cout<<"collection audit passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
