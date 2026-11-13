#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/ObjectCollection.hpp"
#include <nlohmann/json.hpp>

using namespace vc::objects;
void test(bool history)
{
    TemporaryDirectory dir;
    const auto path=dir.path/"objects.vcobj";
    OpenOptions options;options.author="multiedges";options.history=history?HistoryMode::On:HistoryMode::Off;
    const auto a=util::newUuid(),b=util::newUuid();std::string id,exported;
    {
        auto db=ObjectStore::create(path,options);
        {
            auto w=db.write();
            for(const auto& uuid:{a,b}){ObjectRecord r;r.kind="fiber";r.uuid=uuid;db.upsertObject(w,r);}
            EdgeRecord e;e.kind="perpendicular";e.aUuid=a;e.bUuid=b;
            e.aAnchor.position=std::array<double,3>{1,2,3};e.bAnchor.position=std::array<double,3>{4,5,6};
            id=db.addEdge(w,e);
            const auto second=db.addEdge(w,e),third=db.addEdge(w,e);
            require(id!=second&&second!=third&&id!=third,"independent edges have independent UUIDs");
            e.uuid=id;mustFail([&]{db.addEdge(w,e);});
            w.commit();
        }
        {
            auto r=db.read();require(db.edgesOf(r,a).size()==3,"same endpoints retain all observations");
            auto selection=EdgeSelection{};selection.uuids=std::vector<std::string>{id};
            const auto projected=db.selectEdges(r,selection,std::vector<std::string>{"uuid"});
            require(projected.size()==1&&projected[0].uuid==id,"edge identity projection");
            exported=db.exportJson(r);
            require(nlohmann::json::parse(exported).at("edges").size()==3,"JSON preserves all observations");
        }
        {
            auto w=db.write();auto e=*db.edgeByUuid(w,id);e.meta=R"({"updated":true})";db.updateEdge(w,e);w.commit();
        }
        auto copy=ObjectStore::create(dir.path/"copy.vcobj",options);
        {auto w=copy.write();copy.importJson(w,exported);w.commit();}
        {auto r=copy.read();require(copy.edgesOf(r,a).size()==3&&copy.edgeByUuid(r,id).has_value(),"JSON preserves original edge keys");}
    }
}
void traversal(bool history)
{
    TemporaryDirectory dir;
    OpenOptions options;options.author="traversal";options.history=history?HistoryMode::On:HistoryMode::Off;
    const auto base=dir.path/"base.vcobj",overlay=dir.path/"overlay.vcobj";
    const auto a=util::newUuid(),b=util::newUuid(),c=util::newUuid();
    const std::string first="00000000-0000-4000-8000-000000000001",second="00000000-0000-4000-8000-000000000002";
    auto link=[&](auto& db,auto& txn,const auto& id,const auto& x,const auto& y) {
        EdgeRecord edge;edge.uuid=id;edge.kind="fiber_branch";edge.aUuid=x;edge.bUuid=y;db.addEdge(txn,edge);
    };
    {
        auto db=ObjectStore::create(base,options);
        {auto w=db.write();
            for(const auto& id:{a,b,c}){ObjectRecord object;object.uuid=id;object.kind="fiber";db.upsertObject(w,object);}
            link(db,w,second,a,c);link(db,w,first,a,b);
            link(db,w,"00000000-0000-4000-8000-000000000003",a,b);
            link(db,w,"00000000-0000-4000-8000-000000000004",b,c);w.commit();}
        {auto r=db.read();
            require(db.neighbours(r,a)==std::vector{c,b},"local neighbours preserve insertion order and deduplicate parallel edges");
            require(db.reachableWithin(r,a,1)==std::vector{a,c,b},"local indexed BFS preserves neighbour order");
            require(db.componentOf(r,a)==std::vector{a,c,b},"cycles do not duplicate local component nodes");}
        {auto w=db.write();db.seal(w);w.commit();}
    }
    {auto collection=ObjectCollection::create(overlay,{base},options);}
    auto db=ObjectStore::open(overlay,options);
    {auto r=db.read();
        require(db.neighbours(r,a)==std::vector{b,c},"collection neighbours preserve edge-UUID order");
        require(db.reachableWithin(r,a,1)==std::vector{a,b,c},"collections use the same BFS with logical edge order");
        require(db.componentOf(r,a)==std::vector{a,b,c},"collection cycles and parallel edges are deduplicated");
        require(db.neighbours(r,a,std::nullopt,std::string("surface")).empty(),"neighbour kind filters apply to referenced endpoints");}
    {auto w=db.write();db.removeEdge(w,second);w.commit();}
    {auto r=db.read();
        require(db.neighbours(r,a)==std::vector{b},"a local edge tombstone suppresses inherited adjacency");
        require(db.reachableWithin(r,a,1)==std::vector{a,b},"bounded walks do not cross a tombstoned edge");
        require(db.componentOf(r,a)==std::vector{a,b,c},"the remaining cycle path still reaches the third node");}
}
int main(){test(false);test(true);traversal(false);traversal(true);}
