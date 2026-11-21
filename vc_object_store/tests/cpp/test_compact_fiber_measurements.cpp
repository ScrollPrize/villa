#include "TestSupport.hpp"
#include "vc_object_store/objects/FiberImportExtension.hpp"
#include "vc_object_store/objects/FiberConstraintGraph.hpp"
#include <vc_reader/compact_normal.hpp>
#include <cmath>
#include <iostream>
#include <limits>
using namespace vc::objects;
namespace {
void codec(){
    for(double arc:{0.,.25,12345.125,1e8}){
        const auto bytes=encodeFiberArc(arc);require(bytes.size()==4,"arc BLOB length");
        require(decodeFiberArc(bytes)==double(float(arc)),"exact requested f32 rounding");}
    mustFail([]{encodeFiberArc(-1.);});mustFail([]{decodeFiberArc(std::array<uint8_t,3>{});});
    mustFail([]{encodeFiberArc(std::numeric_limits<double>::infinity());});
    require(*encodeFiberWindingDistance(0.)==1,"zero distance is valid code one");
    require(*encodeFiberWindingDistance(15.875)==255,"maximum valid winding code");
    require(!encodeFiberWindingDistance(std::nextafter(15.875,16.)),"skip before rounding above maximum");
    for(int i=0;i<=254;++i){const auto code=encodeFiberWindingDistance(i/16.);
        require(code&&decodeFiberWindingDistance(std::array<uint8_t,1>{*code})==i/16.,"distance grid round trip");}
    mustFail([]{decodeFiberWindingDistance(std::array<uint8_t,1>{0});});
    for(const std::array<double,3> n:std::array<std::array<double,3>,7>{{{0,0,1},{0,0,-1},{1,0,0},{1,0,1e-10},{1,0,-1e-10},{.3,.4,.5},{-.3,.4,-.5}}}){
        const auto bytes=encodeFiberDirection(n);const auto shared=vcr::normal::encodeCompactAxis(n);
        require(bytes.size()==3&&shared&&bytes[0]==(*shared)[0]&&bytes[1]==(*shared)[1],"shared normal encoder bytes");
        require(bytes[2]==(n[2]<0?255:1),"signed hemisphere including equator");
        const auto decoded=decodeFiberDirection(bytes);double norm=0;for(auto v:decoded)norm+=v*v;
        require(std::abs(norm-1.)<1e-12,"normal decodes to unit vector");
    }
    require(encodeFiberDirection({0,0,0})==std::vector<uint8_t>({0,0,0}),"undefined direction code");
    mustFail([]{decodeFiberDirection(std::array<uint8_t,3>{128,128,2});});
    double angleSum=0.,angleMax=0.;size_t angleCount=0;
    for(int iz=0;iz<=100;++iz)for(int ip=0;ip<128;++ip){
        const double z=-1.+iz/50.,phi=2.*std::acos(-1.)*ip/128.;
        const double radius=std::sqrt(std::max(0.,1.-z*z));
        const std::array<double,3> n{radius*std::cos(phi),radius*std::sin(phi),z};
        const auto decoded=decodeFiberDirection(encodeFiberDirection(n));
        const double dot=std::clamp(n[0]*decoded[0]+n[1]*decoded[1]+n[2]*decoded[2],-1.,1.);
        const double angle=std::acos(dot)*180./std::acos(-1.);
        angleSum+=angle;angleMax=std::max(angleMax,angle);++angleCount;
    }
    std::cout<<"compact_normal_quantization samples="<<angleCount<<" mean_degrees="<<angleSum/angleCount
        <<" max_degrees="<<angleMax<<'\n';
    const auto spec=fiberConstraintExtension({"",{},{},"sample/volume"});
    require(spec.formatVersion==4&&spec.columns.size()==8&&spec.indexes.empty(),"eight compact columns and no indexes");
}
void graph(){
    TemporaryDirectory dir;OpenOptions options;options.author="compact-measurements";options.history=HistoryMode::Off;
    options.spatialIndex=false;options.edgeAdjacencyIndex=false;options.edgeColumns=std::vector<std::string>{};
    auto store=ObjectCollection::create(dir.path/"graph.vcobj",{},options);auto w=store.write();
    ObjectRecord object;object.kind="note";const auto a=store.upsertObject(w,object),b=store.upsertObject(w,object);
    EdgeRecord edge;edge.kind="perpendicular";edge.aUuid=a.objectUuid;edge.bUuid=b.objectUuid;
    const auto key=store.addEdge(w,a,b,edge);const auto spec=fiberConstraintExtension({"",{},{},"sample/volume"});
    store.createExtensionTable(w,spec);
    FiberConstraintData data;data.values={12.25,17.5,0.,1.,.3,9.,1.};
    data.normals[0].valid=1;data.normals[0].xyz={0.,0.,1.};data.normals[1].valid=1;data.normals[1].xyz={0.,0.,-1.};
    data.measuredPoints=std::array<std::array<double,3>,2>{{{0,0,0},{1,2,3}}};
    auto measurement=fiberConstraintRow(key.objectUuid,999,200,300,b.objectUuid<a.objectUuid,std::move(data));
    store.appendExtensionRows(w,spec.name,std::span(&measurement.values,1));w.commit();
    auto r=store.read();const auto records=readCompactFiberConstraintGraph(store,r,2);
    require(records.size()==1&&records[0].runtimeIndex==0,"dense runtime identity");const auto& row=records[0];
    require(row.a<row.b&&!row.parallel&&row.windingDistance==.3125,"canonical endpoints and effective quantized distance");
    require(row.validA&&row.validB&&row.validConnector,"validity from signed BLOBs");
    const auto stored=store.edge(r,key);require(stored&&!stored->record.aAnchor.position&&!stored->record.bAnchor.position,"no endpoint coordinates persisted");
    const auto rich=readFiberConstraintGraph(store,r);require(rich.size()==1&&rich[0].measurement.values==measurement.values,"rich and compact measurement consistency");
}
}
int main(){try{codec();graph();std::cout<<"compact fiber measurement codec and graph passed\n";}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
