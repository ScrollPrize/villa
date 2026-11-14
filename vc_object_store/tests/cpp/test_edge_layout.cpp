#include "TestSupport.hpp"
#include "internal/EdgeLayout.hpp"
#include "internal/JsonFields.hpp"
#include "internal/Anchors.hpp"
#include "vc_object_store/objects/EdgeKinds.hpp"
#include <iostream>
using namespace vc::objects;
using namespace vc::objects::detail;
int main(){try{
    sqlite3* db=nullptr;require(sqlite3_open(":memory:",&db)==SQLITE_OK,"memory database");
    struct Close{sqlite3* db;~Close(){sqlite3_close(db);}} close{db};
    OpenOptions options;options.edgeColumns=std::vector<std::string>{};options.edgeAdjacencyIndex=false;
    createEdgeSchema(db,options,true);
    auto names=edgeColumns(db);require(names.size()==5,"exactly five required columns");
    require(names.contains("uuid")&&!names.contains("id")&&!names.contains("meta"),"UUID-only reduced layout");
    require(edgeReadColumn(db,"state")=="0"&&edgeReadColumn(db,"version")=="1","omitted field defaults");
    mustFail([&]{requireEdgeColumns(db,{"meta"});});
    mustFail([&]{requireAdjacencyIndex(db);});
    exec(db,"CREATE INDEX edges_a ON edges(a_uuid,kind);CREATE INDEX edges_b ON edges(b_uuid,kind)");
    requireAdjacencyIndex(db);
    const auto a=util::newUuid(),b=util::newUuid(),e=util::newUuid();
    execute(db,"INSERT INTO edges VALUES(?,?,?,?,?)",parseUuid(e),parseUuid(a),parseUuid(b),int64_t(0),int64_t(2));
    Statement read(db,"SELECT version,state,meta,uuid FROM "+edgeReadSource(db));
    require(read.row()&&read.i64(0)==1&&read.i64(1)==0&&read.isNull(2)&&read.uuid(3)==e,"logical defaults on reduced row");
    require(parseObjectJson("").empty()&&parseObjectJson("{}").empty()&&parseObjectJson("null").empty(),"empty JSON forms");
    mustFail([&]{parseObjectJson("[]");});mustFail([&]{parseObjectJson("broken");});
    Anchor anchor;dependenciesFromJson(anchor,nlohmann::json());
    require(!anchor.wholeGeometry&&anchor.points.empty(),"NULL dependencies unknown, not whole geometry");
    anchor.points={{"line_points",{1}}};dependenciesFromJson(anchor,nlohmann::json::object());
    require(anchor.points.empty(),"decoding replaces previous dependencies");
    require(edgeKindCode("parallel")==0&&edgeKindCode("perpendicular")==1,"stable compact codes");
    for(int64_t i=0;i<int64_t(kCoreEdgeKinds.size());++i)require(edgeKindCode(edgeKindName(i))==i,"kind round trip");
    require(edgeApprovalCode("pending")==0&&edgeApprovalName(2)=="rejected","approval codes");
    {
        TemporaryDirectory dir;OpenOptions opts;opts.author="edge-layout";opts.history=HistoryMode::Off;
        opts.spatialIndex=false;opts.edgeAdjacencyIndex=false;opts.edgeColumns=std::vector<std::string>{};
        auto store=ObjectStore::create(dir.path/"slim.vcobj",opts);std::string left,right,edge;
        {auto w=store.write();ObjectRecord object;object.kind="note";
            left=store.upsertObject(w,object);right=store.upsertObject(w,object);
            EdgeRecord relation;relation.kind="parallel";relation.aUuid=left;relation.bUuid=right;
            edge=store.addEdge(w,relation);w.commit();}
        EdgeRecord copied;
        {auto r=store.read();auto relation=store.edgeByUuid(r,edge);
            require(relation&&relation->kind=="parallel"&&relation->meta=="{}"&&relation->version==1,"store reads omitted defaults");
            copied=*relation;copied.uuid.clear();copied.version=87;
            copied.createdBy=copied.changedBy="old author";copied.createdAt=copied.changedAt="old timestamp";
            copied.createdHost=copied.changedHost="old host";
            mustFail([&]{store.edgesOf(r,left);});}
        {auto w=store.write();edge=store.addEdge(w,copied);w.commit();}
        {auto r=store.read();const auto relation=store.edgeByUuid(r,edge);
            require(relation&&relation->version==1&&relation->createdBy.empty()&&relation->changedAt.empty(),
                    "new reduced-layout edge ignores copied managed version and stamp fields");}
        {auto w=store.write();mustFail([&]{store.setMeta(w,edge,"unexpected","1");});
            mustFail([&]{store.setApproval(w,edge,"approved");});}
    }
    std::cout<<"edge layouts and empty JSON passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
