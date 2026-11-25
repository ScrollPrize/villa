#include "FiberConstraintDatabase.hpp"
#include "FiberConstraintConfig.hpp"
#include "FiberConstraintChunks.hpp"
#include "ChunkSaturationWindow.hpp"
#include "FiberConstraintParallelism.hpp"
#include "FiberConstraintSources.hpp"
#include "vc/core/util/SpatialBatches.hpp"
#include <vc_object_store/geometry/Geometry.hpp>
#include <vc_object_store/objects/InitialImport.hpp>
#include <vc_object_store/objects/FiberConstraintGraph.hpp>
#include <vc_object_store/Uuid.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <sqlite3.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <future>
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
vc::lasagna::LasagnaDataset normalField(const fs::path& root,bool otherAxis=false,bool allValid=false,size_t invalidX=8){
    constexpr size_t volume=3*4*9;
    std::vector<uint8_t> payload(3*volume);
    for(size_t i=0;i<volume;++i){
        payload[i]=(!allValid&&i%9==invalidX)?0:255;
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

void extractionConfigModes(){
    TemporaryDirectory dir;
    Json cfg={{"orient_normals",true},{"repetitions",1},{"database",(dir.path/"graph.vcobj").string()},
        {"normal_prior_output",(dir.path/"prior.json").string()},{"umbilicus","axis.json"},{"umbilicus_sha256","axis-id"}};
    require(validateFiberConstraintJointNormals(cfg),"joint extraction configuration must validate");
    cfg.erase("database");
    mustFail([&]{validateFiberConstraintJointNormals(cfg);},"joint extraction without graph accepted");
    cfg["orient_normals"]=false;cfg["axial_prior_only"]=true;cfg["axial_prior_output"]="axial.json";
    cfg.erase("normal_prior_output");
    require(!validateFiberConstraintJointNormals(cfg),"axial-only features must not require normal outputs");
}

void sourceSelection(){
    TemporaryDirectory dir;
    const auto field=normalField(dir.path/"compound.zarr");
    auto manifest=field.manifest().raw;
    manifest["groups"]["packed"]["channels"]={"sheet_grad_mag","sheet_nx","sheet_ny"};
    vcr::zarr::atomicWrite(dir.path/"compound.zarr/.zattrs",Json{{"lasagna",manifest}}.dump());
    const Json provenance={{"source",{{"sample_id","sample"},{"volume_id","volume"}}},
        {"artifacts",Json::array({{{"kind","manifest"},{"path","compound.zarr/.zattrs"}}})}};
    vcr::zarr::atomicWrite(dir.path/"inference.json",provenance.dump());
    Json config={{"normal_manifest",(dir.path/"compound.zarr").string()},
        {"normal_provenance",(dir.path/"inference.json").string()}};
    auto source=openFiberConstraintSource(config,"normal","sample/volume",{});
    const vc::lasagna::LasagnaNormalSampler sampler(vc::lasagna::fiberSheetNormalDataset(source.dataset));
    const auto channels=sampler.regionChannels();
    require(channels[0].array==channels[1].array&&channels[1].array==channels[2].array,
        "sheet-normal aliases must share one packed array for grouped I/O");
    require(sampler.sampleNormal({1,1,1}).valid,"compound normal source");
    mustFail([&]{openFiberConstraintSource(config,"normal","sample/other",{});},"wrong source volume accepted");
    auto wrong=provenance;wrong["artifacts"][0]["path"]="other.zarr/.zattrs";
    vcr::zarr::atomicWrite(dir.path/"inference.json",wrong.dump());
    mustFail([&]{openFiberConstraintSource(config,"normal","sample/volume",{});},"wrong compound artifact accepted");
    auto winding=normalField(dir.path/"winding",true);
    config["winding_manifest"]=(dir.path/"winding/normal.lasagna.json").string();
    config["winding_provenance"]=(dir.path/"winding.json").string();
    auto wp=provenance;wp["artifacts"][0]["path"]="normal.lasagna.json";
    vcr::zarr::atomicWrite(dir.path/"winding.json",wp.dump());
    auto ws=openFiberConstraintSource(config,"winding","sample/volume",{});
    const vc::lasagna::LasagnaNormalSampler density(ws.dataset);
    require(std::abs(density.sampleNormal({1,1,1}).normal.dot(sampler.sampleNormal({1,1,1}).normal))<.2,
        "separate sources keep independent normal axes");
}

void spatialPlans(){
    const auto position=[](size_t i){return cv::Vec3d{double((i*17)%1031)-100.,double((i*41)%997),double((i*101)%503)};};
    const auto serial=vc::core::util::spatialBatches(300000,128.,128,1,position);
    const auto parallel=vc::core::util::spatialBatches(300000,128.,128,4,position);
    require(serial.order==parallel.order,"spatial plan must not depend on worker count");
    require(serial.regions==parallel.regions,"spatial region accounting");
    for(size_t j=1;j<serial.order.size();++j){
        const auto key=[&](size_t i){const auto p=position(i);return std::array<double,3>{
            std::floor(p[2]/128.),std::floor(p[1]/128.),std::floor(p[0]/128.)};};
        const auto a=serial.order[j-1],b=serial.order[j];
        require(key(a)<key(b)||(key(a)==key(b)&&a<b),"ROI order must match stable lexicographic sorting");
    }
    std::vector<bool> seen(serial.order.size(),false);
    for(const auto& batch:serial.batches){
        size_t sum=0;const auto first=position(serial.order[batch.begin]);
        for(size_t j=batch.begin;j<batch.end;++j){
            const auto i=serial.order[j];require(!seen[i],"spatial item duplicated");seen[i]=true;++sum;
            const auto p=position(i);
            for(int d=0;d<3;++d)require(std::floor(p[d]/128.)==std::floor(first[d]/128.),"batch crosses ROI boundary");
        }
        require(sum==batch.weight&&sum<=128,"spatial batch bounds");
    }
    require(std::all_of(seen.begin(),seen.end(),[](bool x){return x;}),"spatial item omitted");
    require(vc::core::util::spatialBatches(0,128.,128,2,position).batches.empty(),"empty spatial plan");
    const std::array<cv::Vec3d,4> invalidPositions{{{0,0,0},{NAN,0,0},{0,INFINITY,0},{-1,0,0}}};
    const auto invalidPlan=vc::core::util::spatialBatches(4,1.,8,2,[&](size_t i){return invalidPositions[i];});
    require(invalidPlan.order==std::vector<size_t>{3,0,1,2},"invalid samples keep a stable isolated ROI");
    for(size_t count:{size_t{0},size_t{1},size_t{3},size_t{1000}}){
        std::vector<unsigned> visits(count,0);
        vc::core::util::parallelSpatialBatches(count,8,[&](size_t job,size_t worker,int budget){
            require(worker<std::min<size_t>(count,8),"spatial worker index");
            require(budget==8/int(std::min<size_t>(count,8)),"spatial inner CPU budget");
            ++visits[job];
        });
        require(std::all_of(visits.begin(),visits.end(),[](auto n){return n==1;}),"spatial jobs must run exactly once");
    }
    bool threw=false;
    try{vc::core::util::parallelSpatialBatches(100,4,[](auto,auto,int){throw std::runtime_error("read failed");});}
    catch(const std::runtime_error&){threw=true;}
    require(threw,"spatial read errors must propagate after joining workers");
}

void failedSinkCancelsScoring(){
    std::vector<ft::FiberletCropTraceLine> lines(16);
    for(int i=0;i<8;++i){
        lines[i].pointsBaseXYZ={{0.,64.*i,0.},{512.,64.*i,0.}};
        lines[8+i].pointsBaseXYZ={{64.*i,0.,20.},{64.*i,512.,20.}};
    }
    ft::FiberTraceConstraintConfig config;
    config.localEvents=true;config.localKeepWholeFibers=true;config.parallelThreads=1;
    config.certifyWinding=false;config.enforceMaximumWindingDistance=false;
    config.candidateBatchSize=64;
    const auto sampler=[](const auto&,const auto&,double){return .5;};
    const auto baseline=ft::extractFiberTraceConstraints(lines,config,sampler);
    require(baseline.scoringBatches>1,"cancellation fixture requires multiple scoring batches");
    size_t attempted=0;bool notified=false;
    ft::FiberTraceConstraintStream stream;
    stream.prepared=[](auto){};
    stream.batch=[&](auto,auto,auto){++attempted;throw std::runtime_error("sink failed");};
    stream.failed=[&](auto){notified=true;};
    mustFail([&]{(void)ft::extractFiberTraceConstraints(lines,config,sampler,{},{},nullptr,{},stream);},
        "failed sink must propagate its exception");
    require(attempted==1&&notified,"failed sink must cancel remaining scoring jobs immediately");
}

void saturationWindow(){
    require(fiberConstraintWorkersForPhysicalCores(96)==96,"default is all physical cores, not SMT threads");
    require(fiberConstraintWorkersForPhysicalCores(1)==1&&fiberConstraintWorkersForPhysicalCores(3)==3,
        "worker default must preserve the detected physical CPU count");
#if defined(__linux__) || defined(__APPLE__)
    const auto defaultWorkers=defaultFiberConstraintWorkers();
    require(defaultWorkers>0,"OS physical-core topology must give a positive worker default");
    std::cout<<Json{{"default_compute_workers",defaultWorkers}}.dump()<<std::endl;
#endif
    ChunkSaturationWindow window(128,1.,28.);
    window.observe({0.,0.,128,1000,{0,0,0}});
    window.observe({1.,100.,128,900,{10,20,30}});
    window.observe({2.,200.,128,128,{20,40,60}});
    require(!window.done(),"128 remaining jobs still sustain 128 workers");
    window.observe({3.,300.,127,127,{30,60,90}});
    const auto result=window.summary();
    require(window.done()&&result.at("valid")==true,"cutoff must complete a valid saturated window");
    require(result.at("window_seconds")==2.&&result.at("effective_parallelism")==100.,"exclude warmup CPU and time");
    require(result.at("emitted_constraints")==60,"exclude warmup output counts");
    ChunkSaturationWindow capped(128,0.,28.);
    capped.observe({0.,0.,128,1000,{0,0,0}});capped.observe({28.,280.,128,800,{100,200,300}});
    require(capped.done()&&capped.summary().at("reason")=="measurement_cap","cap stops even a still-saturated pool");
    ChunkSaturationWindow completed(128,0.,60.,true);
    completed.observe({0.,0.,128,1000,{0,0,0},0});
    completed.observe({60.,600.,128,800,{100,200,300},127});
    require(!completed.done(),"duration alone must not stop before each worker completes a chunk");
    completed.observe({61.,610.,128,799,{101,201,301},128});
    require(completed.done()&&completed.summary().at("completed_workers")==128,"all workers completed permits duration cutoff");
    ChunkSaturationWindow tail(128,0.,120.,true);
    tail.observe({0.,0.,128,1000,{0,0,0},0});
    tail.observe({40.,400.,127,127,{100,200,300},127});
    require(!tail.done(),"tail freezes statistics but cannot stop an unfinished first worker");
    tail.observe({45.,450.,126,126,{110,220,330},128});
    require(tail.done()&&tail.summary().at("window_seconds")==40.&&tail.summary().at("worker_completion_wait_seconds")==5.,"completion wait must not contaminate saturated rates");
    ChunkSaturationWindow invalid(128,1.,28.);invalid.abortUnstarted({});
    require(invalid.done()&&invalid.summary().at("valid")==false,"never fabricate saturation or a remaining-job cutoff");
}

void standardNormalStorage(const vo::ObjectCollection& store,const vo::CollectionTxn& txn,
    const std::vector<vo::FiberConstraintRecord>& records){
    const auto tables=store.extensionTables(txn);
    const auto schema=std::find_if(tables.begin(),tables.end(),[](const auto& table){
        return table.name==vo::kFiberConstraintTable;
    });
    require(schema!=tables.end()&&schema->formatVersion==4&&schema->columns.size()==8&&schema->indexes.empty(),"compact format 4 has eight columns and no measurement indexes");
    require(std::none_of(tables.begin(),tables.end(),[](const auto& table){
        return table.name=="ext_fiber_graph_solve_normal_frames";
    }),"piece corrections must not be required by a published graph");
    require(!Json::parse(schema->provisioning).contains("normal_gauge"),"published measurements must not declare a piece-local gauge");
    const auto column=[&](const std::string& name){
        const auto found=std::find_if(schema->columns.begin(),schema->columns.end(),[&](const auto& c){return c.name==name;});
        require(found!=schema->columns.end(),"normal column is missing");
        return size_t(found-schema->columns.begin());
    };
    for(const auto& record:records)for(int side=0;side<2;++side){
        const auto name=side?std::string("normal_b"):std::string("normal_a");
        const auto& normal=side?record.normalB:record.normalA;
        const auto valid=side?record.validB:record.validA;
        const auto& raw=record.measurement.values;
        const auto& blob=std::get<std::vector<uint8_t>>(raw.at(column(name)));
        require(blob.size()==3&&(blob[2]!=0)==valid,
            "normal validity must be final in the stored measurement row");
        require(vo::decodeFiberDirection(blob)==normal,"reader must not need a hidden piece-sign correction");
        require(!record.edge.aAnchor.position&&!record.edge.bAnchor.position,"core endpoint XYZ is omitted");
    }
}

void sortMeasurements(std::vector<vo::FiberConstraintRecord>& records){
    std::sort(records.begin(),records.end(),[](const auto& a,const auto& b){
        return std::tie(a.a,a.b,a.arcA,a.arcB,a.parallel,a.windingDistance,a.connector)<std::tie(b.a,b.b,b.arcA,b.arcB,b.parallel,b.windingDistance,b.connector);
    });
}

void chunkPipeline(){
    TemporaryDirectory dir;
    const auto dataset=normalField(dir.path/"normal",false,true);
    const vc::lasagna::LasagnaNormalSampler reader(dataset);
    const std::vector<std::string> ids={"10000000-0000-4000-8000-000000000001",
        "20000000-0000-4000-8000-000000000002"};
    std::vector<ft::FiberletCropTraceLine> lines(2);
    lines[0].pointsBaseXYZ={{1,1,1},{7,1,1}};
    lines[1].pointsBaseXYZ={{4,0,2},{4,3,2}};
    const auto source=dir.path/"source.vcobj";
    vo::InitialImportOptions sourceOptions;sourceOptions.store.author="chunk-regression";
    const auto imported=vo::importInitialCollection(source,sourceOptions,[&](size_t,auto& writer){
        std::vector<vo::InitialObjectRecord> objects;
        for(size_t i=0;i<ids.size();++i){
            vo::InitialObjectRecord object;object.record.uuid=ids[i];object.record.kind="fiber";object.record.fiber.emplace();
            vc::geometry::Fiber fiber;for(const auto& p:lines[i].pointsBaseXYZ)fiber.linePoints.push_back({p[0],p[1],p[2]});
            object.geometry=vc::geometry::encode(fiber);objects.push_back(std::move(object));
        }
        writer.appendObjects(objects);
    });
    auto axis=vc::core::util::Umbilicus::FromPoints({{0,0,0},{0,0,3}},{4,12,12});
    ft::FiberTraceConstraintConfig options;
    options.localEvents=true;options.localKeepWholeFibers=true;options.certifyWinding=false;
    options.enforceMaximumWindingDistance=false;options.closestParallelEstimator=false;
    options.localSpacingBaseVoxels=.5;options.localTangentSpanBaseVoxels=.125;
    options.localParallelSpanBaseVoxels=.5;options.maximumDistanceBaseVoxels=2.;
    options.localReciprocalToleranceBaseVoxels=.1;
    std::vector<vo::FiberConstraintRecord> reference;
    for(size_t workers:{size_t{1},size_t{4}}){
        const auto label=std::to_string(workers);
        Json cfg={{"fibers",source.string()},{"database",(dir.path/("graph-"+label)).string()},
            {"normal_prior_output",(dir.path/("priors-"+label+".json")).string()},
            {"coordinate_space","synthetic/volume"},{"umbilicus_sha256","test-axis"},
            {"chunk_size_base_voxels",2.},{"chunk_geometry_halo",4.},{"normal_sample_spacing",.5}};
        Json metadata={{"config",cfg},{"fibers_store_uuid",imported.storeUuid},
            {"fiber_source",{{"store",imported.storeUuid},{"generation",imported.generation},{"commit_sequence",imported.commitSequence}}},
            {"normal_manifest",dataset.manifest().raw},{"normal_provenance",Json::object()}};
        options.parallelThreads=workers;
        FiberConstraintDatabaseOptions database;database.workers=2;database.preparationWorkers=2;
        const auto result=extractFiberConstraintChunks(cfg,metadata,ids,lines,options,reader,reader,axis,database);
        require(result.at("chunk_size_base_voxels")==2.,"report explicit spatial chunk size");
        require(result.at("chunk_count").get<size_t>()>=5,"fixture must span several spatial jobs");
        require(result.at("unresolved_pieces")==0,"identical valid boundaries must reconcile");
        require(result.at("export").at("sampling_seconds")==0.,"chunk output must never reread attachment normals");
        // Check ownership accounting in addition to worker-independent output.
        size_t encountered=0,emitted=0;
        for(const auto& chunk:result.at("chunks")) {
            encountered+=chunk.at("candidates").get<size_t>();emitted+=chunk.at("events").get<size_t>();
        }
        require(encountered==result.at("geometry_accepted_encounters").get<size_t>(),
            "each finalized geometric encounter must belong to exactly one sampling chunk");
        require(emitted==result.at("measured_encounters").get<size_t>(),
            "streamed measurement counts must match the shared extraction report");
        auto store=vo::ObjectCollection::open(result.at("export").at("path").get<std::string>(),vo::OpenOptions{.readOnly=true});
        auto txn=store.read();auto rows=vo::readFiberConstraintGraph(store,txn);
        standardNormalStorage(store,txn,rows);
        const auto compact=vo::readCompactFiberConstraintGraph(store,txn,2);
        require(compact.size()==rows.size(),"compact and rich chunk read counts");
        for(size_t i=0;i<rows.size();++i)require(compact[i].normalA==rows[i].normalA&&compact[i].normalB==rows[i].normalB,
            "both graph readers must return the same baked endpoint normals");
        require(!rows.empty(),"chunk fixture must emit a perpendicular encounter");
        sortMeasurements(rows);
        const auto anchor=reader.sampleNormal(lines[0].pointsBaseXYZ.front()).normal;
        for(const auto& row:rows)if(row.a==ids[0])
            require(cv::Vec3d(row.normalA[0],row.normalA[1],row.normalA[2]).dot(anchor)>0.,
                "stored endpoint normals must already include their final piece sign");
        if(reference.empty())reference=rows;
        else {
            require(rows.size()==reference.size(),"chunk count must be worker independent");
            for(size_t i=0;i<rows.size();++i){
                require(rows[i].a==reference[i].a&&rows[i].b==reference[i].b,
                    "chunk event ownership must be independent of completion order");
                require(rows[i].arcA==reference[i].arcA&&rows[i].arcB==reference[i].arcB&&
                    rows[i].normalA==reference[i].normalA&&rows[i].normalB==reference[i].normalB,
                    "chunk geometry and reconciled normals must be worker independent");
            }
        }
        if(workers==4) {
            auto benchmarkConfig=cfg;
            benchmarkConfig["database"]=(dir.path/"cancelled-benchmark").string();
            benchmarkConfig["normal_prior_output"]=(dir.path/"cancelled-priors.json").string();
            benchmarkConfig["saturated_chunk_benchmark"]=true;
            benchmarkConfig["benchmark_window_seconds"]=.01;
            benchmarkConfig["benchmark_warmup_seconds"]=0.;
            auto benchmarkMetadata=metadata;benchmarkMetadata["config"]=benchmarkConfig;
            const auto stopped=extractFiberConstraintChunks(benchmarkConfig,benchmarkMetadata,ids,lines,options,reader,reader,axis,database);
            require(stopped.at("benchmark").at("partial_graph_discarded")==true,"bounded benchmark must discard its partial graph");
            require(!std::filesystem::exists(dir.path/"cancelled-benchmark")&&
                    !std::filesystem::exists(dir.path/"cancelled-priors.json"),"never publish incomplete benchmark graph or priors");
        }
        if(workers==1) {
            auto defaultConfig=cfg;defaultConfig.erase("chunk_size_base_voxels");
            defaultConfig["database"]=(dir.path/"default-chunk-size").string();
            defaultConfig["normal_prior_output"]=(dir.path/"default-chunk-priors.json").string();
            auto defaultMetadata=metadata;defaultMetadata["config"]=defaultConfig;
            const auto defaultResult=extractFiberConstraintChunks(defaultConfig,defaultMetadata,ids,lines,options,reader,reader,axis,database);
            require(defaultResult.at("chunk_size_base_voxels")==2048.,"default spatial chunk size must be 2048 base voxels");
            require(defaultResult.at("unresolved_pieces")==0,"default-sized pieces must retain consistent normal frames");
            auto defaultStore=vo::ObjectCollection::open(defaultResult.at("export").at("path").get<std::string>(),vo::OpenOptions{.readOnly=true});
            auto defaultTxn=defaultStore.read();const auto defaultRows=vo::readFiberConstraintGraph(defaultStore,defaultTxn);
            standardNormalStorage(defaultStore,defaultTxn,defaultRows);
            require(defaultRows.size()==rows.size(),"changing chunk size must not change the measurement count");
            for(const auto& row:rows){
                const auto other=std::find_if(defaultRows.begin(),defaultRows.end(),[&](const auto& candidate){
                    return candidate.a==row.a&&candidate.b==row.b&&candidate.arcA==row.arcA&&candidate.arcB==row.arcB;
                });
                require(other!=defaultRows.end()&&other->normalA==row.normalA&&other->normalB==row.normalB,
                    "stored globally aligned normals must be independent of spatial chunk size");
            }
        }
    }
}

void unresolvedChunkBoundary(){
    TemporaryDirectory dir;
    // The invalid x=4 plane is exactly an interior piece boundary. The
    // constraint at x=6 has valid prediction data but cannot inherit a known
    // orientation through that boundary, so its stored H normal must be invalid.
    const auto dataset=normalField(dir.path/"normal",false,false,4);
    const vc::lasagna::LasagnaNormalSampler reader(dataset);
    const std::vector<std::string> ids={"10000000-0000-4000-8000-000000000001",
        "20000000-0000-4000-8000-000000000002"};
    std::vector<ft::FiberletCropTraceLine> lines(2);
    lines[0].pointsBaseXYZ={{1,1,1},{7,1,1}};
    lines[1].pointsBaseXYZ={{6,0,2},{6,3,2}};
    require(reader.sampleNormal({6,1,1}).valid,"unresolved fixture requires a valid prediction at the encounter");
    const auto source=dir.path/"source.vcobj";
    vo::InitialImportOptions sourceOptions;sourceOptions.store.author="unresolved-normal-regression";
    const auto imported=vo::importInitialCollection(source,sourceOptions,[&](size_t,auto& writer){
        std::vector<vo::InitialObjectRecord> objects;
        for(size_t i=0;i<ids.size();++i){
            vo::InitialObjectRecord object;object.record.uuid=ids[i];object.record.kind="fiber";object.record.fiber.emplace();
            vc::geometry::Fiber fiber;for(const auto& p:lines[i].pointsBaseXYZ)fiber.linePoints.push_back({p[0],p[1],p[2]});
            object.geometry=vc::geometry::encode(fiber);objects.push_back(std::move(object));
        }
        writer.appendObjects(objects);
    });
    Json cfg={{"fibers",source.string()},{"database",(dir.path/"graph.vcobj").string()},
        {"normal_prior_output",(dir.path/"priors.json").string()},{"coordinate_space","synthetic/volume"},
        {"umbilicus_sha256","test-axis"},{"chunk_size_base_voxels",2.},{"normal_sample_spacing",.5}};
    Json metadata={{"config",cfg},{"fibers_store_uuid",imported.storeUuid},
        {"fiber_source",{{"store",imported.storeUuid},{"generation",imported.generation},{"commit_sequence",imported.commitSequence}}},
        {"normal_manifest",dataset.manifest().raw},{"normal_provenance",Json::object()}};
    ft::FiberTraceConstraintConfig options;options.localEvents=true;options.localKeepWholeFibers=true;
    options.parallelThreads=4;options.certifyWinding=false;options.enforceMaximumWindingDistance=false;
    options.closestParallelEstimator=false;options.localSpacingBaseVoxels=.5;options.localTangentSpanBaseVoxels=.125;
    options.localParallelSpanBaseVoxels=.5;options.maximumDistanceBaseVoxels=2.;options.localReciprocalToleranceBaseVoxels=.1;
    FiberConstraintDatabaseOptions database;database.workers=2;database.preparationWorkers=2;
    const auto axis=vc::core::util::Umbilicus::FromPoints({{0,0,0},{0,0,3}},{4,12,12});
    const auto result=extractFiberConstraintChunks(cfg,metadata,ids,lines,options,reader,reader,axis,database);
    require(result.at("unresolved_pieces").get<size_t>()>0,"invalid shared boundary must remain unresolved");
    auto store=vo::ObjectCollection::open(result.at("export").at("path").get<std::string>(),vo::OpenOptions{.readOnly=true});
    auto txn=store.read();const auto rows=vo::readFiberConstraintGraph(store,txn);
    require(!rows.empty(),"unresolved fixture must retain the geometric constraint");
    standardNormalStorage(store,txn,rows);
    for(const auto& row:rows){
        require(row.a==ids[0]||row.b==ids[0],"unresolved fixture endpoint identity");
        require(!(row.a==ids[0]?row.validA:row.validB),"unknown piece orientation must be invalid in the final standard row");
    }
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
        {"fibers_store_uuid",base.storeUuid},
        {"fiber_source",{{"store",base.storeUuid},{"generation",base.generation},{"commit_sequence",base.commitSequence}}},
        {"normal_manifest",dataset.manifest().raw},
        {"normal_provenance",{{"source_uuid","synthetic-normal-field"}}}};
    FiberConstraintDatabaseOptions options;options.workers=int(shards);options.packed=packed;
    options.baseToStored=baseToStored;
    const auto begin=std::chrono::steady_clock::now();
    const auto imported=writeFiberConstraintDatabase(dir.path/"graph",ids,report,normals,2,provenance,{},options);
    const fs::path graph=imported.at("path").get<std::string>();
    require(imported.at("constraints")==3&&imported.at("parallel")==1,"constraint counters");
    require(imported.at("invalid_endpoint_normals")==2,"invalid normals retained, not invented");
    verifyFiberConstraintDatabaseMeasurements(graph,report,ids);
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
        require(definition.formatVersion==4,"native compact measurement format4");
        const auto metadata=Json::parse(definition.provisioning);
        require(metadata.at("coordinate_space")=="synthetic/volume"&&metadata.at("endpoint_order")=="canonical","coordinate frame metadata");
        require(metadata.at("normal_source_sha256").get<std::string>().size()==64,"compact normal-source identity");
        require(!metadata.contains("config")&&!metadata.contains("selected_ids"),"no configuration/data dumps in provisioning");
    }
    auto records=vo::readFiberConstraintGraph(store,txn);sortMeasurements(records);
    // Coarse work assignment may complete out of order; scientific rows and
    // source IDs must nevertheless remain identical to the ordered source.
    ft::FiberTraceConstraintReport repeated;repeated.pieces=report.pieces;
    for(size_t i=0;i<96;++i)repeated.constraints.push_back(report.constraints[i%3]);
    auto parallelOptions=options;parallelOptions.preparationWorkers=4;parallelOptions.preparationBatchRows=1;
    size_t cursor=0;
    const auto concurrent=streamFiberConstraintDatabase(dir.path/"concurrent",ids,repeated.pieces,
        [&](std::stop_token){
            if(cursor==repeated.constraints.size())return std::vector<ft::FiberTraceConstraint>{};
            return std::vector<ft::FiberTraceConstraint>{repeated.constraints[cursor++]};
        },normals,2,provenance,{},parallelOptions,repeated.constraints.size());
    verifyFiberConstraintDatabaseMeasurements(concurrent.at("path").get<std::string>(),repeated,ids);
    require(concurrent.at("constraints")==96&&concurrent.at("invalid_endpoint_normals")==64,
        "parallel preparation preserves counters and invalid normals");
    const auto pushed=streamFiberConstraintDatabase(dir.path/"pushed",ids,repeated.pieces,
        {},normals,2,provenance,{},parallelOptions,repeated.constraints.size(),
        [&](const auto& consume){
            std::vector<std::future<void>> jobs;
            for(size_t worker=0;worker<4;++worker)jobs.push_back(std::async(std::launch::async,[&,worker]{
                // Submit source ordinals in reverse order, on independent workers.
                for(size_t j=0;j<24;++j){const auto i=95-worker-4*j;
                    consume(worker,i,std::span(&repeated.constraints[i],1));}
            }));
            for(auto& job:jobs)job.get();
        },4);
    verifyFiberConstraintDatabaseMeasurements(pushed.at("path").get<std::string>(),repeated,ids);
    require(pushed.at("constraints")==96&&pushed.at("invalid_endpoint_normals")==64,
        "unordered push preserves measurements, identities and counters");
    require(pushed.at("shared_extraction_preparation_workers")==true&&pushed.at("source_claim_worker_seconds")==0,
        "push uses existing workers without serialized source claiming");
    mustFail([&]{streamFiberConstraintDatabase(dir.path/"failed-push",ids,report.pieces,
        {},normals,2,provenance,{},parallelOptions,0,
        [](const auto&){throw std::runtime_error("push source failed");},4);},"push failure must propagate");
    require(!fs::exists(dir.path/"failed-push"),"failed push must not publish a partial store");
    auto concurrentStore=vo::ObjectCollection::open(concurrent.at("path").get<std::string>(),vo::OpenOptions{.readOnly=true});
    auto concurrentTxn=concurrentStore.read();auto concurrentRows=vo::readFiberConstraintGraph(concurrentStore,concurrentTxn);
    sortMeasurements(concurrentRows);
    for(size_t i=0;i<concurrentRows.size();++i)
        for(size_t k=3;k<5;++k)require(concurrentRows[i].measurement.values[k]==records[i/32].measurement.values[k],
            "parallel sampling preserves endpoint normals exactly");
    mustFail([&]{streamFiberConstraintDatabase(dir.path/"failed",ids,report.pieces,
        [](std::stop_token)->std::vector<ft::FiberTraceConstraint>{throw std::runtime_error("source failed");},
        normals,2,provenance,{},parallelOptions);},"parallel source failure must propagate");
    require(!fs::exists(dir.path/"failed"),"failed parallel import must not publish");
    require(records[0].a==ids[1]&&records[0].b==ids[0],"measurement endpoint order is canonical");
    require(records[0].edge.aUuid==ids[1]&&records[0].edge.bUuid==ids[0],"canonical UUID order");
    require(!records[0].edge.aAnchor.position&&!records[0].edge.bAnchor.position,"core anchors are omitted, not copied or transformed");
    require(!records[0].validConnector,"coincident attachment connector explicitly undefined");
    require(records[0].arcA==1&&records[0].arcB==5,"canonical arc order");
    require(!records[2].validA&&!records[2].validB,"invalid endpoint flags");
    require(std::get<std::vector<uint8_t>>(records[2].measurement.values[3])==std::vector<uint8_t>({0,0,0}),"invalid normal uses undefined encoding");
    for(const auto& id:ids){
        const auto obj=store.object(txn,{base.storeUuid,id});
        require(obj&&obj->key.storeUuid==base.storeUuid,"geometry inherited, not copied");
    }
    size_t objects=0;store.scanObjects(txn,[&](const auto&){++objects;return true;});
    require(objects==3,"object inventory unchanged");
    const auto rawNormal=normals.sampleNormal({6,1,1});require(rawNormal.valid,"fixture valid endpoint");
    const auto oriented=orientFiberConstraintDatabase(graph,dir.path/"oriented",ids,lines,normals,2,1.,provenance);
    const auto serialOriented=orientFiberConstraintDatabase(graph,dir.path/"oriented-serial",ids,lines,normals,1,1.,provenance);
    require(oriented.at("sweep_workers")==2&&serialOriented.at("sweep_workers")==1,"orientation uses independent fiber workers");
    for(const auto* metric:{"samples","flipped_samples","flipped_endpoints","invalid_samples","invalid_endpoints","fibers_without_valid_normals"})
        require(oriented.at(metric)==serialOriented.at(metric),"parallel orientation counters match serial sweep");
    require(oriented.at("flipped_endpoints").get<size_t>()>0,"fixture exercises sign flips");
    std::cout<<Json{{"normal_resampling",{{"comparisons",oriented.at("compact_resampled_normal_comparisons")},
        {"axis_mean_degrees",oriented.at("compact_resampled_axis_angle_mean_degrees")},
        {"axis_max_degrees",oriented.at("compact_resampled_axis_angle_max_degrees")},
        {"oriented_mean_degrees",oriented.at("compact_resampled_oriented_angle_mean_degrees")},
        {"oriented_max_degrees",oriented.at("compact_resampled_oriented_angle_max_degrees")}}}}.dump()<<'\n';
    require(oriented.at("invalid_endpoints")==2&&oriented.at("fibers_without_valid_normals")==1,"invalid sweep invariants");
    const fs::path orientedPath=oriented.at("path").get<std::string>();
    verifyFiberConstraintDatabaseMeasurements(orientedPath,report,ids);
    auto aligned=vo::ObjectCollection::open(orientedPath,vo::OpenOptions{.readOnly=true});auto read=aligned.read();
    noCopiedObjects(aligned,aligned.uuid(read));
    auto updated=vo::readFiberConstraintGraph(aligned,read);sortMeasurements(updated);
    const auto axis=vc::core::util::Umbilicus::FromPoints({{0,0,0},{0,0,3}},{4,12,12});
    auto jointSamples=sampleFiberConstraintNormals(ids,lines,report,normals,2,1.,&axis);
    for(const auto [region,limit]:{std::pair{2.,size_t{1}},std::pair{4.,size_t{8}},std::pair{4096.,size_t{262144}}}) {
        const auto tiled=sampleFiberConstraintNormals(ids,lines,report,normals,2,1.,&axis,{region,limit});
        require(tiled.endpoints==jointSamples.endpoints&&tiled.valid==jointSamples.valid&&
            tiled.invalidReasons==jointSamples.invalidReasons,"ROI size/batch cap must preserve exact endpoint normals/reasons");
        require(tiled.outwardAgreement==jointSamples.outwardAgreement&&tiled.validArcWeight==jointSamples.validArcWeight,
            "ROI scheduling must preserve exact arc-ordered prior accumulation");
        require(tiled.statistics.at("maximum_batch_points").get<size_t>()<=limit,"normal point cap honored");
    }
    auto jointOptions=options;jointOptions.normalSamples=&jointSamples;
    auto jointProvenance=provenance;jointProvenance["config"]["umbilicus_sha256"]="test-axis";
    const auto joint=writeFiberConstraintDatabase(dir.path/"joint",ids,report,normals,2,jointProvenance,{},jointOptions);
    require(joint.at("normal_samples_reused")==true&&joint.at("sampling_seconds")==0.,
        "joint import must not sample attachment normals again");
    require(jointSamples.statistics.at("samples")==oriented.at("samples"),"one identical fiber sweep");
    auto jointStore=vo::ObjectCollection::open(joint.at("path").get<std::string>(),vo::OpenOptions{.readOnly=true});
    auto jointRead=jointStore.read();auto jointRows=vo::readFiberConstraintGraph(jointStore,jointRead);sortMeasurements(jointRows);
    require(jointRows.size()==updated.size(),"joint measurement count");
    for(size_t i=0;i<updated.size();++i) {
        require(jointRows[i].normalA==updated[i].normalA&&jointRows[i].normalB==updated[i].normalB,
            "joint endpoint normals exactly match staged orientation");
        require(jointRows[i].validA==updated[i].validA&&jointRows[i].validB==updated[i].validB,
            "joint endpoint validity exactly matches staged orientation");
        for(size_t column=3;column<5;++column)
            require(jointRows[i].measurement.values[column]==updated[i].measurement.values[column],
                "joint normal values and reasons preserve staged fields exactly");
    }
    verifyFiberConstraintDatabaseMeasurements(joint.at("path").get<std::string>(),report,ids);
    orientFiberConstraintDatabase(orientedPath,dir.path/"staged-priors.json",ids,lines,normals,2,1.,jointProvenance,&axis);
    writeFiberConstraintOutwardPriors(dir.path/"joint-priors.json",ids,jointSamples,joint.at("source_store"),jointProvenance);
    Json stagedPriors,jointPriors;
    std::ifstream(dir.path/"staged-priors.json")>>stagedPriors;
    std::ifstream(dir.path/"joint-priors.json")>>jointPriors;
    require(jointPriors.at("agreement")==stagedPriors.at("agreement")&&
        jointPriors.at("valid_arc_weight")==stagedPriors.at("valid_arc_weight"),"joint priors match staged sweep exactly");
    const auto jointPin=jointStore.provisioning(jointRead);
    require(jointPriors.at("source_store")==Json{{"store_uuid",jointStore.uuid(jointRead)},
        {"generation",jointPin.generation},{"commit_sequence",jointPin.commitSequence}},"prior pins final imported graph");
    require(!fs::exists(dir.path/"joint"/"normals.bin"),"regular normal samples are not persisted");
    for(size_t i=0;i<records.size();++i){
        require(updated[i].edge.uuid==records[i].edge.uuid,"orientation keeps relation identity");
        const std::array<vo::ExtensionValue,1> prefix{vo::extensionUuid(updated[i].edge.uuid)};
        size_t routed=0;
        aligned.scanExtensionRows(read,vo::kFiberConstraintTable,[&](const auto& row){
            ++routed;require(row==updated[i].measurement.values,"sharded full scan disagrees with UUID-routed reads");return true;
        },prefix);
        require(routed==1,"routed measurement row count");
        for(size_t j=0;j<records[i].measurement.values.size();++j)
            if(j!=3&&j!=4)
                require(updated[i].measurement.values[j]==records[i].measurement.values[j],"non-normal measurement changed");
    }
    const auto vector=[](const auto& p){return cv::Vec3d(p[0],p[1],p[2]);};
    require(vector(updated[0].normalB).dot(vector(updated[1].normalB))>0.,"H normals retain the minimum-turn gauge");
    require(cv::norm(vector(updated[0].normalA)-vector(updated[1].normalA))<=1e-12,"V normals have a consistent independent gauge");
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
    // Source identity is authoritative; axis discrepancies are measurements,
    // not a hidden acceptance tolerance for intentionally quantized storage.
    const auto axisDifference=orientFiberConstraintDatabase(graph,dir.path/"different-axis",ids,lines,other,2,1.,provenance);
    require(axisDifference.at("compact_resampled_axis_angle_max_degrees").get<double>()>20.,"changed axis discrepancy reported");
    const vc::lasagna::LasagnaNormalSampler valid(normalField(dir.path/"all-valid",false,true));
    mustFail([&]{orientFiberConstraintDatabase(graph,dir.path/"bad-validity",ids,lines,valid,2,1.,provenance);},"changed normal validity accepted");
    for(const auto* name:{"bad-unoriented","bad-spacing","bad-source","bad-validity"}){
        require(!fs::exists(dir.path/name),"failed validation published output");
        require(!fs::exists(dir.path/(std::string(name)+".partial")),"failed validation left temporary data");
    }
    std::cout<<Json{{"shards",shards},{"packed",packed},{"constraints",3},{"exact_round_trip",true},
        {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()}}.dump()<<'\n';
}
}
int main(){
    try{extractionConfigModes();spatialPlans();sourceSelection();failedSinkCancelsScoring();saturationWindow();chunkPipeline();unresolvedChunkBoundary();roundTrip(1,false);roundTrip(2,false);roundTrip(2,true);}
    catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
