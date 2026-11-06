#include "TestSupport.hpp"
#include "vc_object_store/objects/ObjectStore.hpp"
#include "vc_object_store/objects/Hub.hpp"
#include "internal/Anchors.hpp"
#include <nlohmann/json.hpp>

using namespace vc::objects;
using Json=nlohmann::json;
void test(bool history)
{
    TemporaryDirectory dir; const auto path=dir.path/"objects.vcobj";
    OpenOptions options;options.author="multiedges";options.history=history?HistoryMode::On:HistoryMode::Off;
    const auto a=util::newUuid(),b=util::newUuid();std::string id,exported;
    {
        auto db=ObjectStore::create(path,options);
        {
            auto w=db.write();
            for(const auto& uuid:{a,b}){ObjectRecord r;r.kind="fiber";r.uuid=uuid;db.upsertObject(w,r);}
            EdgeRecord e;e.kind="perpendicular";e.aUuid=a;e.bUuid=b;
            e.aAnchor.position=std::array<double,3>{1,2,3};e.bAnchor.position=std::array<double,3>{4,5,6};
            db.addEdge(w,e);
            mustFail([&]{db.addEdge(w,e);}); // Existing default uniqueness unchanged.
            e.relationKey="measurement:one";id=db.addEdge(w,e);
            mustFail([&]{db.addEdge(w,e);});
            const auto identity=detail::edgeIdentity(e);
            e.relationKey="measurement:two";db.addEdge(w,e);
            require(detail::edgeIdentity(e)!=identity,"candidate identity respects independent observations");
            w.commit();
        }
        {
            auto r=db.read();require(db.edgesOf(r,a).size()==3,"three native edges");
            require(db.edgeByUuid(r,id)->relationKey=="measurement:one","native key readback");
            auto selection=EdgeSelection{};selection.uuids=std::vector<std::string>{id};
            const auto projected=db.selectEdges(r,selection,std::vector<std::string>{"relation_key"});
            require(projected.size()==1&&projected[0].relationKey=="measurement:one","native projection");
            JsonExportOptions projection;projection.fields={"relation_key"};
            const auto json=Json::parse(db.exportJson(r,projection));
            require(json.at("edges").size()==3,"projected edges preserved");
            int keyed=0;for(const auto& e:json.at("edges"))if(!e.at("relation_key").get<std::string>().empty())++keyed;
            require(keyed==2,"JSON projection keys");
            if(history){
                RevisionSelection revisions{{a,std::nullopt},{b,std::nullopt}};
                const auto relations=db.relationshipsAt(r,revisions);int retained=0;
                for(const auto& e:relations)if(!e.relationship.record.relationKey.empty())++retained;
                require(retained==2,"history record keys");
            }
            exported=db.exportJson(r);
        }
        {
            auto w=db.write();auto e=*db.edgeByUuid(w,id);e.meta="{\"updated\":true}";db.updateEdge(w,e);
            e.relationKey="changed";mustFail([&]{db.updateEdge(w,e);});w.commit();
        }
        auto copy=ObjectStore::create(dir.path/"copy.vcobj",options);
        {auto w=copy.write();copy.importJson(w,exported);w.commit();}
        {auto r=copy.read();require(copy.edgesOf(r,a).size()==3&&copy.edgeByUuid(r,id)->relationKey=="measurement:one","metadata transfer preserves independent observations");}
    }
    if(history){
        initHub(path,"multiedges");
        cloneHub(path,dir.path/"replica.vcobj",options);
        auto replica=ObjectStore::open(dir.path/"replica.vcobj",options);
        {auto r=replica.read();require(replica.edgeByUuid(r,id)->relationKey=="measurement:one","hub clone key");}
        acquireLeases(replica,{a,b});
        std::string extra;
        {auto w=replica.write();auto edge=*replica.edgeByUuid(w,id);edge.uuid.clear();edge.relationKey="measurement:three";extra=replica.addEdge(w,edge);w.commit();}
        require(hubPush(replica).conflicts.empty(),"push independent observation");
        cloneHub(path,dir.path/"other.vcobj",options);
        auto other=ObjectStore::open(dir.path/"other.vcobj",options);
        {auto r=other.read();require(other.edgeByUuid(r,extra)->relationKey=="measurement:three"&&other.edgesOf(r,a).size()==4,"hub transfer preserves multiedges");}
    }
}
int main(){test(false);test(true);}
