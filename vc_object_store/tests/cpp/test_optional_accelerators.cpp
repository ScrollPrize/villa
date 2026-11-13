#include "TestSupport.hpp"
#include "internal/Database.hpp"
#include "imports/SourceIdentityIndex.hpp"
#include "geometry/DependencyIndex.hpp"
#include "vc_object_store/geometry/Geometry.hpp"
#include "vc_object_store/objects/StoreConversion.hpp"
#include "vc_object_store/objects/GeometryEdits.hpp"
#include <iostream>

using namespace vc::objects;
using namespace vc::objects::detail;
namespace {
bool hasTable(const Txn& txn,const std::string& name)
{
    Statement row(ObjectStoreImplAccess::conn(txn),"SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?",name);
    return row.row();
}
ObjectRecord object(const std::string& kind="note")
{
    ObjectRecord record;record.kind=kind;record.uuid=util::newUuid();return record;
}
WriteTxn editObject(ObjectStore& store,const std::string& object)
{
    std::optional<int64_t> version;{auto read=store.read();version=store.currentVersion(read,object);}
    return store.writeNumbered(VersionSelection{{object,version}});
}
void sourceIndex(HistoryMode history)
{
    TemporaryDirectory dir;const auto path=dir.path/"source.vcobj";
    OpenOptions options;options.history=history;options.spatialIndex=false;options.author="test";
    auto store=ObjectStore::create(path,options);
    auto points=object("point_collection");
    const auto geometry=vc::geometry::encode(vc::geometry::PointCollection{{7,9},{{1,2,3},{4,5,6}}});
    {
        auto txn=store.write();store.upsertObject(txn,points);store.putGeometry(txn,points.uuid,geometry);txn.commit();
        auto read=store.read();
        require(!hasTable(read,"source_points"),"plain point geometry must not provision source identity index");
        require(!hasTable(read,"geometry_edge_dependencies"),"no duplicate geometry dependency materialization");
    }
    const std::string source=R"({"origin":"document","source_collection_id":"points"})";
    {
        auto txn=store.write();store.setMeta(txn,points.uuid,"source",source);
        refreshSourceIdentityIndex(ObjectStoreImplAccess::conn(txn));
        require(hasTable(txn,"source_points"),"first actual source identity provisions index");
        require(sourcePoint(ObjectStoreImplAccess::conn(txn),"document",7)->object==points.uuid,"lazy point lookup");
        // Intentionally roll back both the records and schema provisioning.
    }
    {auto read=store.read();require(!hasTable(read,"source_points"),"lazy index rolls back atomically");}
    {
        auto txn=store.write();store.setMeta(txn,points.uuid,"source",source);txn.commit();
        auto read=store.read();auto* db=ObjectStoreImplAccess::conn(read);
        require(sourceCollection(db,"document","points")==points.uuid,"lazy collection identity lookup");
        require(sourcePoint(db,"document",9)->offset==1,"source point offset remains exact");
        const auto tables=store.databaseIdentity(read).tables;
        require(std::any_of(tables.begin(),tables.end(),[](const auto& t){return t.name=="source_points";}),"lazy table has native directory identity");
    }
    {
        auto txn=editObject(store,points.uuid);
        store.setMeta(txn,points.uuid,"source",R"({"origin":"renamed","source_collection_id":"points"})");
        refreshSourceIdentityIndex(ObjectStoreImplAccess::conn(txn));
        updateGeometryArrays(store,txn,points.uuid,vc::geometry::encode(vc::geometry::PointCollection{{9,11},{{4,5,6},{7,8,9}}}),
            store.currentGeometry(txn,points.uuid)->uuid);
        refreshSourceIdentityIndex(ObjectStoreImplAccess::conn(txn));
        require(!sourcePoint(ObjectStoreImplAccess::conn(txn),"renamed",7)&&sourcePoint(ObjectStoreImplAccess::conn(txn),"renamed",11).has_value(),
            "same-publication geometry changes invalidate source identity cache");
        txn.commit();
        auto read=store.read();auto* db=ObjectStoreImplAccess::conn(read);
        require(!sourcePoint(db,"document",9)&&!sourcePoint(db,"renamed",7),"old source identities removed");
        require(sourcePoint(db,"renamed",11)->object==points.uuid,"lazy index tracks later metadata and geometry edits");
    }
}
void sourcePolicyTransitions()
{
    TemporaryDirectory dir;const auto path=dir.path/"policy";const auto empty=dir.path/"empty";
    OpenOptions options;options.history=HistoryMode::Off;options.spatialIndex=false;options.author="test";
    auto points=object("point_collection");points.meta=R"({"source":{"origin":"document","source_collection_id":"points"}})";
    {auto store=ObjectStore::create(path,options);auto txn=store.write();store.upsertObject(txn,points);
        store.putGeometry(txn,points.uuid,vc::geometry::encode(vc::geometry::PointCollection{{7},{{1,2,3}}}));txn.commit();}
    {auto store=ObjectStore::create(empty,options);}
    for(const auto mode:{HistoryMode::On,HistoryMode::Off,HistoryMode::On}) {
        changeHistoryMode(path,mode);changeHistoryMode(empty,mode);
        {auto store=ObjectStore::open(empty,options);auto txn=store.read();
            require(!hasTable(txn,"source_points"),"history policy cannot provision unused source accelerator");}
        auto store=ObjectStore::open(path,options);
        {auto txn=store.read();require(sourcePoint(ObjectStoreImplAccess::conn(txn),"document",7)->object==points.uuid,"history transition retains source identities");}
        auto txn=editObject(store,points.uuid);
        updateGeometryArrays(store,txn,points.uuid,vc::geometry::encode(vc::geometry::PointCollection{{7},{{2,3,4}}}),
            store.currentGeometry(txn,points.uuid)->uuid);
        refreshSourceIdentityIndex(ObjectStoreImplAccess::conn(txn));
        require(sourcePoint(ObjectStoreImplAccess::conn(txn),"document",7).has_value(),"source accelerator refreshes after policy transition");
        txn.commit();
    }
}
void adjacencyCandidates()
{
    TemporaryDirectory dir;OpenOptions options;options.history=HistoryMode::Off;options.spatialIndex=false;options.author="test";
    auto store=ObjectStore::create(dir.path/"adjacency",options);
    auto a=object("fiber"),b=object("fiber");
    const auto geometry=vc::geometry::encode(vc::geometry::Fiber{{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}},{7,9}});
    auto txn=store.write();for(const auto& r:{a,b}){store.upsertObject(txn,r);store.putGeometry(txn,r.uuid,geometry);}
    auto add=[&](const std::string& key,Anchor anchor,bool reverse=false){
        EdgeRecord edge;edge.kind="test";edge.aUuid=a.uuid;edge.bUuid=b.uuid;
        edge.aAnchor=std::move(anchor);edge.bAnchor.wholeGeometry=true;
        if(reverse){std::swap(edge.aUuid,edge.bUuid);std::swap(edge.aAnchor,edge.bAnchor);}
        return store.addEdge(txn,edge);
    };
    Anchor global;global.wholeGeometry=true;
    Anchor positional;positional.points={{"line_points",{0}}};
    Anchor stable7;stable7.points={{"control_point_ids",{7}}};
    Anchor stable9;stable9.points={{"control_point_ids",{9}}};
    const auto full=add("global",global),position=add("position",positional),seven=add("seven",stable7,true),nine=add("nine",stable9);
    const auto candidates=[&](const std::vector<int64_t>& changed){
        std::set<std::string> result;for(const auto& edge:structuralEdgeCandidates(ObjectStoreImplAccess::conn(txn),a.uuid,"control_point_ids",changed))result.insert(edge.uuid);return result;
    };
    require(candidates({7,7})==std::set<std::string>{full,position,seven},"stable ID pruning matches native dependencies");
    require(candidates({})==std::set<std::string>{full,position},"empty change set retains only conservative candidates");
    require(candidates({9})==std::set<std::string>{full,position,nine},"stable ID candidates do not depend on endpoint order");
    require(!hasTable(txn,"geometry_edge_dependencies"),"adjacency query creates no materialized rows");
    txn.commit();
}
void lifecycle(HistoryMode history)
{
    TemporaryDirectory dir;OpenOptions options;options.history=history;options.spatialIndex=false;options.author="test";
    auto store=ObjectStore::create(dir.path/"lifecycle",options);auto a=object(),b=object();std::string relation;
    auto edge=[&]{EdgeRecord e;e.kind="test";e.aUuid=a.uuid;e.bUuid=b.uuid;e.aAnchor.wholeGeometry=true;e.bAnchor.wholeGeometry=true;return e;};
    {auto txn=store.write();store.upsertObject(txn,a);store.upsertObject(txn,b);relation=store.addEdge(txn,edge());txn.commit();}
    std::optional<ObjectVersion> original;
    if(history==HistoryMode::On){auto read=store.read();original=store.currentObjectRevision(read,a.uuid)->reference();}
    auto write=[&]{
        VersionSelection selected;if(history==HistoryMode::On){auto read=store.read();for(const auto& r:{a,b})selected[r.uuid]=store.currentVersion(read,r.uuid);}
        return history==HistoryMode::On?store.writeNumbered(selected):store.write();
    };
    {auto txn=write();store.retireObject(txn,a.uuid);require(!store.edgeByUuid(txn,relation),"retire removes current edge");}
    {auto read=store.read();require(store.edgeByUuid(read,relation).has_value()&&!store.objectByUuid(read,a.uuid)->retired,"retire rollback restores relationships and state");}
    {auto txn=write();store.retireObject(txn,a.uuid);txn.commit();}
    {auto read=store.read();require(store.objectByUuid(read,a.uuid)->lifecycleEpoch==std::to_string(store.provisioning(read).commitSequence),"retire token is publication commit sequence");}
    {auto txn=write();auto retired=*store.objectByUuid(txn,a.uuid);retired.name="still-retired";store.upsertObject(txn,retired);
        require(store.objectByUuid(txn,a.uuid)->retired,"ordinary upsert must not activate a retired object");
        Statement state(ObjectStoreImplAccess::conn(txn),"SELECT state FROM objects WHERE uuid=?",parseUuid(a.uuid));
        require(state.row()&&state.i64(0)==1,"retirement has the single state value one");txn.commit();}
    {auto txn=write();store.restoreObject(txn,a.uuid);txn.commit();}
    {auto read=store.read();require(!store.edgeByUuid(read,relation),"restore never revives retired relationships");
        require(store.objectByUuid(read,a.uuid)->lifecycleEpoch==std::to_string(store.provisioning(read).commitSequence),"restore token is publication commit sequence");}
    {
        auto txn=write();const auto created=store.addEdge(txn,edge());store.retireObject(txn,a.uuid);
        require(!store.edgeByUuid(txn,created),"retire removes edges created in same transaction");
        store.restoreObject(txn,a.uuid);txn.commit();
    }
    {auto read=store.read();require(store.edgesOf(read,a.uuid,true).empty(),"same-transaction retire/restore cannot revive edges");}
    if(original) {
        std::string currentEdge;
        {auto txn=write();store.setMeta(txn,a.uuid,"current_only","true");currentEdge=store.addEdge(txn,edge());txn.commit();}
        {auto txn=write();store.restoreObject(txn,a.uuid,original);require(!store.edgeByUuid(txn,currentEdge),"historical active-object restore removes current observations");}
        {auto read=store.read();require(store.edgeByUuid(read,currentEdge).has_value(),"historical restore rolls back relation removal atomically");}
        {auto txn=write();store.restoreObject(txn,a.uuid,original);txn.commit();}
        {auto read=store.read();require(!store.edgeByUuid(read,currentEdge),"historical restore never reattaches earlier relations");
            require(!nlohmann::json::parse(store.objectByUuid(read,a.uuid)->meta).contains("current_only"),"historical restore still restores object metadata");}
    }
}
}
int main()try{
    for(auto history:{HistoryMode::Off,HistoryMode::On}){sourceIndex(history);lifecycle(history);}
    adjacencyCandidates();sourcePolicyTransitions();std::cout<<"optional accelerators and lifecycle passed\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
