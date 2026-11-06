#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/Hub.hpp"
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
    if(history){
        initHub(path,"multiedges");cloneHub(path,dir.path/"replica.vcobj",options);
        auto replica=ObjectStore::open(dir.path/"replica.vcobj",options);
        acquireLeases(replica,{a,b});std::string extra;
        {auto w=replica.write();auto edge=*replica.edgeByUuid(w,id);edge.uuid.clear();extra=replica.addEdge(w,edge);w.commit();}
        require(hubPush(replica).conflicts.empty(),"push independent observation");
        cloneHub(path,dir.path/"other.vcobj",options);
        auto other=ObjectStore::open(dir.path/"other.vcobj",options);
        {auto r=other.read();require(other.edgeByUuid(r,extra).has_value()&&other.edgesOf(r,a).size()==4,"hub preserves edge identities");}
    }
}
int main(){test(false);test(true);}
