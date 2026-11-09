#include "FiberConstraintDatabase.hpp"
#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/objects/InitialImport.hpp>
#include <vc_object_store/objects/FiberConstraintGraph.hpp>
#include <vc_object_store/Uuid.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <sqlite3.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
namespace fs=std::filesystem;
namespace vo=vc::objects;
namespace ft=vc::fiber_tracer;
using Json=nlohmann::json;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> void mustFail(F&& f,const char* message){
    try{f();}catch(const std::exception&){return;}throw std::runtime_error(message);
}
struct TemporaryDirectory {
    fs::path path=fs::temp_directory_path()/("vc-fiber-normal-"+vo::util::newUuid());
    TemporaryDirectory(){fs::create_directories(path);}
    ~TemporaryDirectory(){std::error_code error;fs::remove_all(path,error);}
};

// A sign-ambiguous normal field whose decoded gauge changes along H, plus a
// genuine invalid endpoint. All I/O and interpolation use the production reader.
vc::lasagna::LasagnaDataset normalField(const fs::path& root,bool otherAxis=false,bool allValid=false){
    constexpr size_t volume=3*4*9;
    std::vector<uint8_t> payload(3*volume);
    for(size_t i=0;i<volume;++i){
        payload[i]=(!allValid&&i%9==8)?0:255;
        payload[volume+i]=otherAxis?128:(i%9<4?248:8);
        payload[2*volume+i]=otherAxis?248:128;
    }
    auto writer=vcr::zarr::Writer::create(root/"channels",{
        .shape={3,3,4,9},.chunks={3,3,4,9},.compressor=""});
    writer.writeChunk({0,0,0,0},payload.data(),payload.size());
    Json manifest={{"version",3},{"source_to_base",1.},{"base_shape_zyx",{3,4,9}},
        {"grad_mag_encode_scale",255.},{"grad_mag_factor",1.},
        {"mask_invalid_normal_corners",true},
        {"groups",{{"packed",{{"zarr","channels"},{"scaledown",0},
            {"channels",{"grad_mag","nx","ny"}}}}}}};
    vcr::zarr::atomicWrite(root/"normal.lasagna.json",manifest.dump());
    return vc::lasagna::LasagnaDataset::open(root/"normal.lasagna.json");
}

void roundTrip(size_t shards,bool packed){
    TemporaryDirectory dir;
    const auto dataset=normalField(dir.path/"normals");
    const vc::lasagna::LasagnaNormalSampler normals(dataset);
    const std::vector<std::string> ids={"f0000000-0000-4000-8000-000000000001",
        "10000000-0000-4000-8000-000000000002","90000000-0000-4000-8000-000000000003"};
    std::vector<ft::FiberletCropTraceLine> lines(3);
    lines[0].pointsBaseXYZ={{1,1,1},{8,1,1}};
    lines[1].pointsBaseXYZ={{6,0,1},{6,3,1}};
    lines[2].pointsBaseXYZ={{0,10,1},{1,10,1}};
    std::vector<vcr::catalog::Affine> baseToStored(ids.size());
    if(packed){
        baseToStored[0]=vcr::catalog::Affine::scale(.5);baseToStored[0](0,3)=4.;
        baseToStored[1]=vcr::catalog::Affine::scale(2.);baseToStored[1](1,3)=-3.;
    }
    const auto source=dir.path/"source.vcobj";
    vo::InitialImportOptions sourceOptions;sourceOptions.store.author="native-fiber-regression";
    const auto base=vo::importInitialCollection(source,sourceOptions,[&](size_t,auto& writer){
        std::vector<vo::InitialObjectRecord> objects;
        for(size_t i=0;i<ids.size();++i){
            vo::InitialObjectRecord row;row.record.uuid=ids[i];row.record.kind="fiber";row.record.fiber.emplace();
            vc::geometry::Fiber geometry;
            for(const auto& p:lines[i].pointsBaseXYZ)geometry.linePoints.push_back(baseToStored[i].apply({p[0],p[1],p[2]}));
            row.geometry=vc::geometry::encode(geometry);objects.push_back(std::move(row));
        }
        writer.appendObjects(objects);
    });
    // Deliberately permute piece and source indexes; they are not interchangeable.
    ft::FiberTraceConstraintReport report;report.pieces.resize(3);
    report.pieces[0].traceIndex=2;report.pieces[0].endArcBaseVoxels=1;
    report.pieces[1].traceIndex=0;report.pieces[1].endArcBaseVoxels=7;
    report.pieces[2].traceIndex=1;report.pieces[2].endArcBaseVoxels=3;
    ft::FiberTraceConstraint c;c.pieceA=1;c.pieceB=2;c.arcABaseVoxels=5;c.arcBBaseVoxels=1;
    c.pointABaseXYZ={6,1,1};c.pointBBaseXYZ={6,1,1};c.parallelScore=.25;c.perpendicularScore=.75;
    c.windingDistance=1.25;c.parallelWindingDistance=.625;c.signedWindingDelta=-1.25;
    c.signedParallelWindingDelta=.625;c.perpendicularNormalAlignment=-.75;c.parallelNormalAlignment=.25;
    c.windingNormalComponent=2;c.evidenceWeight=.875;c.localEvent=true;c.localOrientation=-1;
    c.supportABegin=.125;c.supportAEnd=6.75;c.supportBBegin=.25;c.supportBEnd=2.75;
    c.parallelCorrespondenceSamples=3;c.parallelMeanAdvanceResidualFraction=.125;
    c.parallelMeanConnectorTangentAbsDot=.25;c.parallelMeanConnectorLengthChangeFraction=.375;
    c.parallelMeanConnectorDirectionChange=.5;c.parallelLimitHitFraction=.625;
    report.constraints.push_back(c);
    c.pieceA=2;c.pieceB=1;c.arcABaseVoxels=2;c.arcBBaseVoxels=0;
    c.pointABaseXYZ={6,2,1};c.pointBBaseXYZ={1,1,1};c.signedWindingDelta=2.5;
    c.parallelScore=.8;c.perpendicularScore=.2;c.hardContinuity=true;c.allowMeasurementDisable=false;
    report.constraints.push_back(c);
    c.pieceA=1;c.pieceB=0;c.arcABaseVoxels=7;c.arcBBaseVoxels=1;
    c.pointABaseXYZ={8,1,1};c.pointBBaseXYZ={1,10,1};c.parallelScore=.1;c.perpendicularScore=.9;
    c.signedWindingDelta.reset();c.windingNormalComponent.reset();
    report.constraints.push_back(c);
    Json provenance={{"config",{{"fibers",source.string()},{"import_workers",shards},{"packed_store",packed},
        {"coordinate_space","synthetic/volume"}}},
        {"fibers_store_uuid",base.storeUuid},{"normal_manifest",dataset.manifest().raw},
        {"normal_provenance",{{"source_uuid","synthetic-normal-field"}}}};
    FiberConstraintDatabaseOptions options;options.workers=int(shards);options.packed=packed;
    options.baseToStored=baseToStored;
    const auto begin=std::chrono::steady_clock::now();
    const auto imported=writeFiberConstraintDatabase(dir.path/"graph",ids,report,normals,2,provenance,{},options);
    const fs::path graph=imported.at("path").get<std::string>();
    require(imported.at("constraints")==3&&imported.at("parallel")==1,"constraint counters");
    require(imported.at("invalid_endpoint_normals")==2,"invalid normals retained, not invented");
    verifyFiberConstraintDatabaseMeasurements(graph,report);
    auto store=vo::ObjectCollection::open(graph,vo::OpenOptions{.readOnly=true});auto txn=store.read();
    const auto noCopiedObjects=[](vo::ObjectCollection& collection,const std::string& owner){
        for(const auto& member:collection.members())if(member.storeUuid==owner){
            // Physical inventory assertion, not a production validation pass.
            // ObjectCollection intentionally composes the geometry dependency.
            sqlite3* raw=nullptr;
            const int rc=sqlite3_open_v2(member.locator.string().c_str(),&raw,SQLITE_OPEN_READONLY,
                vo::isPackedDatabasePath(member.locator)?vo::packedStoreVfs():nullptr);
            std::unique_ptr<sqlite3,decltype(&sqlite3_close)> db(raw,sqlite3_close);
            require(rc==SQLITE_OK,"open physical inventory");
            sqlite3_stmt* statement=nullptr;
            const int prepared=sqlite3_prepare_v2(raw,"SELECT count(*) FROM objects",-1,&statement,nullptr);
            std::unique_ptr<sqlite3_stmt,decltype(&sqlite3_finalize)> query(statement,sqlite3_finalize);
            require(prepared==SQLITE_OK&&sqlite3_step(statement)==SQLITE_ROW,"read physical inventory");
            require(sqlite3_column_int64(statement,0)==0,"derived store must not copy source objects");
        }
    };
    noCopiedObjects(store,store.uuid(txn));
    const auto definitions=store.extensionTables(txn);
    for(const auto& definition:definitions)if(definition.name==vo::kFiberConstraintTable){
        require(definition.formatVersion==2,"native measurement frame uses extension v2, not a core schema change");
        const auto metadata=Json::parse(definition.provisioning);
        require(metadata.at("coordinate_space")=="synthetic/volume"&&metadata.at("anchor_frame")=="object_geometry","coordinate frame metadata");
        require(metadata.at("normal_source_sha256").get<std::string>().size()==64,"compact normal-source identity");
        require(!metadata.contains("config")&&!metadata.contains("selected_ids"),"no configuration/data dumps in provisioning");
    }
    auto records=vo::readFiberConstraintGraph(store,txn);
    require(records[0].a==ids[0]&&records[0].b==ids[1],"restore source endpoint order");
    require(records[0].edge.aUuid==ids[1]&&records[0].edge.bUuid==ids[0],"canonical UUID order");
    require(records[0].edge.aAnchor.position==baseToStored[1].apply({6,1,1})&&
        records[0].edge.bAnchor.position==baseToStored[0].apply({6,1,1}),"core anchors use source geometry frames");
    require(records[0].pointA==std::array<double,3>{6,1,1}&&
        records[0].pointB==std::array<double,3>{6,1,1},"solver measurements retain exact base XYZ");
    require(std::get<int64_t>(records[0].measurement.values[4])==1,"reversed source flag");
    require(records[0].arcA==5&&records[0].arcB==1,"source arc order");
    require(!records[2].validA&&!records[2].validB,"invalid endpoint flags");
    require(!std::get<std::string>(records[2].measurement.values[16]).empty(),"invalid reason retained");
    require(std::holds_alternative<std::monostate>(records[2].measurement.values[13]),"invalid normal has no invented coordinates");
    for(const auto& id:ids){
        const auto obj=store.object(txn,{base.storeUuid,id});
        require(obj&&obj->key.storeUuid==base.storeUuid,"geometry inherited, not copied");
    }
    size_t objects=0;store.scanObjects(txn,[&](const auto&){++objects;return true;});
    require(objects==3,"object inventory unchanged");
    const auto rawNormal=normals.sampleNormal({6,1,1});require(rawNormal.valid,"fixture valid endpoint");
    const auto oriented=orientFiberConstraintDatabase(graph,dir.path/"oriented",ids,lines,normals,2,1.,provenance);
    require(oriented.at("flipped_endpoints").get<size_t>()>0,"fixture exercises sign flips");
    require(oriented.at("invalid_endpoints")==2&&oriented.at("fibers_without_valid_normals")==1,"invalid sweep invariants");
    const fs::path orientedPath=oriented.at("path").get<std::string>();
    verifyFiberConstraintDatabaseMeasurements(orientedPath,report);
    auto aligned=vo::ObjectCollection::open(orientedPath,vo::OpenOptions{.readOnly=true});auto read=aligned.read();
    noCopiedObjects(aligned,aligned.uuid(read));
    const auto updated=vo::readFiberConstraintGraph(aligned,read);
    for(size_t i=0;i<records.size();++i){
        require(updated[i].edge.uuid==records[i].edge.uuid,"orientation keeps relation identity");
        const std::array<vo::ExtensionValue,1> prefix{vo::extensionUuid(updated[i].edge.uuid)};
        size_t routed=0;
        aligned.scanExtensionRows(read,vo::kFiberConstraintTable,[&](const auto& row){
            ++routed;require(row==updated[i].measurement.values,"sharded full scan disagrees with UUID-routed reads");return true;
        },prefix);
        require(routed==1,"routed measurement row count");
        for(size_t j=0;j<records[i].measurement.values.size();++j)
            if(!(j>=13&&j<=15)&&!(j>=18&&j<=20))
                require(updated[i].measurement.values[j]==records[i].measurement.values[j],"non-normal measurement changed");
    }
    const auto vector=[](const auto& p){return cv::Vec3d(p[0],p[1],p[2]);};
    require(vector(updated[0].normalA).dot(vector(updated[1].normalB))>0.,"H normals retain the minimum-turn gauge");
    require(cv::norm(vector(updated[0].normalB)-vector(updated[1].normalA))<=1e-12,"V normals have a consistent independent gauge");
    for(size_t i=0;i<records.size();++i)for(bool b:{false,true})if(b?records[i].validB:records[i].validA){
        const auto old=vector(b?records[i].normalB:records[i].normalA),now=vector(b?updated[i].normalB:updated[i].normalA);
        require(std::min(cv::norm(old-now),cv::norm(old+now))<=1e-12,"normal axes are unchanged");
    }
    const auto curves=dir.path/"curves";
    orientFiberConstraintDatabase(orientedPath,curves,ids,lines,normals,2,1.,provenance,nullptr,true);
    Json index;std::ifstream(curves/"index.json")>>index;
    require(index.at("index").size()==3,"all curve catalog entries exported");
    require(fs::file_size(curves/"normals.bin")==14*5*sizeof(double),"curve binary sample count");
    const auto subset=dir.path/"selected-curves";
    orientFiberConstraintDatabase(orientedPath,subset,{ids[0]},{lines[0]},normals,2,1.,provenance,nullptr,true,true);
    Json selected;std::ifstream(subset/"index.json")>>selected;
    require(selected.at("index").size()==1&&selected.at("index")[0][0]==ids[0],"selected UUID curve catalog");
    mustFail([&]{orientFiberConstraintDatabase(graph,dir.path/"bad-unoriented",ids,lines,normals,2,1.,provenance,nullptr,true);},"unoriented export accepted");
    mustFail([&]{orientFiberConstraintDatabase(orientedPath,dir.path/"bad-spacing",ids,lines,normals,2,2.,provenance,nullptr,true);},"changed sweep spacing accepted");
    auto changedSource=provenance;changedSource["normal_provenance"]["source_uuid"]="different-source";
    mustFail([&]{orientFiberConstraintDatabase(graph,dir.path/"bad-source",ids,lines,normals,2,1.,changedSource);},"changed source identity accepted");
    const vc::lasagna::LasagnaNormalSampler other(normalField(dir.path/"other-axis",true));
    mustFail([&]{orientFiberConstraintDatabase(graph,dir.path/"bad-axis",ids,lines,other,2,1.,provenance);},"changed normal axes accepted");
    const vc::lasagna::LasagnaNormalSampler valid(normalField(dir.path/"all-valid",false,true));
    mustFail([&]{orientFiberConstraintDatabase(graph,dir.path/"bad-validity",ids,lines,valid,2,1.,provenance);},"changed normal validity accepted");
    for(const auto* name:{"bad-unoriented","bad-spacing","bad-source","bad-axis","bad-validity"}){
        require(!fs::exists(dir.path/name),"failed validation published output");
        require(!fs::exists(dir.path/(std::string(name)+".partial")),"failed validation left temporary data");
    }
    std::cout<<Json{{"shards",shards},{"packed",packed},{"constraints",3},{"exact_round_trip",true},
        {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()}}.dump()<<'\n';
}
}
int main(){
    try{roundTrip(1,false);roundTrip(2,false);roundTrip(2,true);}
    catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
