#include "FiberConstraintExtraction.hpp"
#include "FiberConstraintConfig.hpp"
#include "FiberConstraintGeometry.hpp"
#include "FiberGeometryStageBenchmark.hpp"
#include <vc_object_store/objects/ObjectCollection.hpp>
#include <vc_object_store/geometry/SegmentClip.hpp>
#include <vc_object_store/geometry/ArrayBytes.hpp>
#include "FiberConstraintDatabase.hpp"
#include "FiberConstraintChunks.hpp"
#include "FiberConstraintParallelism.hpp"
#include "FiberNormalSamplingBenchmark.hpp"
#include "FiberConstraintSources.hpp"
#include "vc/fiber_tracer/FiberAxialAlignment.hpp"
#include <vc_reader/zarr.hpp>
#include <vc_reader/array.hpp>
#include "vc/core/util/Umbilicus.hpp"
#include <vc_object_store/objects/UmbilicusFile.hpp>
#include <sqlite3.h>
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstring>
#include <map>
#include <vc_object_store/objects/GeometryCoordinates.hpp>
#include <vc_object_store/objects/GeometryRegion.hpp>
#include <vc_object_store/objects/ObjectStore.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <atomic>
#include <exception>
#include <memory>
#include <future>
#include <map>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

vc::fiber_tracer::FiberTraceConstraintConfig extractionOptions(
    const Json& cfg,int threads,size_t crossQueryPrefix,size_t lineCount)
{
    vc::fiber_tracer::FiberTraceConstraintConfig options;
    options.localEvents=true;options.localKeepWholeFibers=true;options.parallelThreads=threads;
    options.candidateBatchSize=cfg.value("candidate_batch_size",options.candidateBatchSize);
    options.scoringRegionBaseVoxels=cfg.value("scoring_region_base_voxels",options.scoringRegionBaseVoxels);
    options.crossQueryPrefix=crossQueryPrefix;
    if(crossQueryPrefix>=lineCount&&crossQueryPrefix)throw std::invalid_argument("Cross query prefix must leave reference lines");
    options.certifyWinding=cfg.value("certify_winding",false);
    options.enforceMaximumWindingDistance=cfg.value("enforce_maximum_winding",false);
    options.closestParallelEstimator=cfg.value("closest_parallel_estimator",false);
    options.localParallelReciprocal=cfg.value("parallel_reciprocal",false);
    options.localMovingProjectionWindows=cfg.value("moving_projection_windows",true);
    options.localPerpendicularProjectionIterations=cfg.value("perpendicular_iterations",16u);
    options.pairOwnedGeometry=cfg.value("pair_owned_geometry",false);
    return options;
}

Json processResources()
{
    Json result=Json::object();
#if defined(__unix__) || defined(__APPLE__)
    rusage usage{};
    if(getrusage(RUSAGE_SELF,&usage)==0) {
#if defined(__APPLE__)
        result["peak_rss_bytes"]=usage.ru_maxrss;
#else
        result["peak_rss_bytes"]=static_cast<uint64_t>(usage.ru_maxrss)*1024;
#endif
        result["cpu_seconds"]=usage.ru_utime.tv_sec+usage.ru_utime.tv_usec*1e-6+
            usage.ru_stime.tv_sec+usage.ru_stime.tv_usec*1e-6;
    }
#endif
    return result;
}

class ExtractionProgress {
    std::mutex mutex_;
    Clock::time_point started_=Clock::now();
    struct Phase {
        Clock::time_point started,last;
        std::size_t done=0,sequence=0;
        bool terminal=false;
    };
    std::map<std::string,Phase> phases_;
    std::size_t sequence_=0;
    double interval_;
    std::ofstream stream_;
    std::function<Json()> readerStats_;
public:
    ExtractionProgress(double interval,const std::string& file):interval_(interval) {
        if(!std::isfinite(interval_) || interval_<0)throw std::invalid_argument("Invalid progress interval");
        if(!file.empty()) {
            const auto parent=std::filesystem::path(file).parent_path();
            if(!parent.empty())std::filesystem::create_directories(parent);
            stream_.open(file,std::ios::app);
            if(!stream_)throw std::runtime_error("Cannot open progress log");
        }
    }
    bool enabled()const{return interval_>0;}
    void readerStats(std::function<Json()> fn) {
        std::lock_guard lock(mutex_);
        readerStats_=std::move(fn);
    }
    void operator()(const vc::fiber_tracer::FiberTraceConstraintProgress& p) {
        if(!enabled())return;
        std::lock_guard lock(mutex_);
        const auto now=Clock::now();
        const std::string name(p.phase);
        auto [it,inserted]=phases_.try_emplace(name);
        auto& phase=it->second;
        const bool changed=inserted || (p.completed==0 && (phase.done>0 || phase.terminal));
        if(changed)phase={now,now,0,++sequence_,false};
        phase.done=std::max(phase.done,p.completed);
        const bool finished=p.total>0 && phase.done==p.total;
        if(!changed && ((!finished && std::chrono::duration<double>(now-phase.last).count()<interval_) ||
                        (finished && phase.terminal)))return;
        const double elapsed=std::chrono::duration<double>(now-phase.started).count();
        const double rate=elapsed>0?phase.done/elapsed:0;
        Json row={{"event","constraint_progress"},{"phase",name},
            {"stage_index",phase.sequence},
            {"completed",phase.done},{"total",p.total},{"candidates",p.candidates},
            {"phase_seconds",elapsed},{"elapsed_seconds",std::chrono::duration<double>(now-started_).count()},
            {"units_per_second",rate},
            {"eta_seconds",rate>0 && p.total>0 && p.total>=phase.done?Json((p.total-phase.done)/rate):Json(nullptr)},
            {"resources",processResources()}};
        if(readerStats_)row["reader_stats"]=readerStats_();
        if(vcr::diagnostics::Trace::enabled())row["coordination_monotonic_seconds"]=std::chrono::duration<double>(now.time_since_epoch()).count();
        std::cout<<row.dump()<<std::endl;
        if(stream_)stream_<<row.dump()<<std::endl;
        phase.terminal=finished;phase.last=now;
    }
};
}

int runFiberConstraintExtraction(int argc, char** argv)
{
    using namespace vc::fiber_tracer;
    using Clock = std::chrono::steady_clock;
    using Json = nlohmann::json;
    if (argc != 2) throw std::invalid_argument("usage: vc_fiber_constraints CONFIG.json [--cuda] [--cuda-cache-gb N]");
    Json cfg; std::ifstream input(argv[1]); input >> cfg;
    if(!cfg.value("sqlite_memory_statistics",true) && sqlite3_config(SQLITE_CONFIG_MEMSTATUS,0)!=SQLITE_OK)
        throw std::runtime_error("SQLite memory statistics must be configured before opening any store");
    for(const auto* key:{"sqlite_serialize_memory","sqlite_in_memory","sqlite_page_size","sqlite_insert_batch_rows"})
        if(cfg.contains(key))throw std::invalid_argument(std::string("Legacy AFC setting is unsupported: ")+key+"; use import_workers/packed_store");
    const auto seconds = [](auto start) { return std::chrono::duration<double>(Clock::now()-start).count(); };
    const auto xyz = [](const Json& v) {
        if (!v.is_array() || v.size()!=3) throw std::invalid_argument("Expected XYZ triple");
        return cv::Vec3d{v.at(0).get<double>(),v.at(1).get<double>(),v.at(2).get<double>()};
    };
    const int threads=cfg.contains("threads")?cfg.at("threads").get<int>():defaultFiberConstraintWorkers();
    const int repetitions=cfg.value("repetitions",3);
    const bool jointNormals=validateFiberConstraintJointNormals(cfg);
    // SQLite reader fan-out can regress badly through backend contention;
    // keep it independent of the compute pool and opt in only after measuring.
    const int loadThreads=cfg.value("load_threads",1);
    if (threads<1 || repetitions<1) throw std::invalid_argument("Positive threads/repetitions required");
    if(loadThreads<1)throw std::invalid_argument("Positive load_threads required");
    if(cfg.contains("database") && repetitions!=1)throw std::invalid_argument("Database export requires one repetition");
    if(cfg.contains("database") && (std::filesystem::exists(cfg.at("database").get<std::string>()) ||
        std::filesystem::exists(cfg.at("database").get<std::string>()+".partial")))
        throw std::runtime_error("Database output already exists");
    const auto started=Clock::now();
    ExtractionProgress progress(cfg.value("progress_interval_seconds",0.),cfg.value("progress_file",std::string{}));
    // The fibers are every object-store fiber bound to the volume that
    // coordinate_space names (SAMPLE/VOLUME_ID), read in its voxels.
    const std::string scan=cfg.at("coordinate_space");
    const auto slash=scan.find('/');
    if (slash==std::string::npos) throw std::invalid_argument("coordinate_space must be SAMPLE/VOLUME_ID");
    const std::string volume=scan.substr(slash+1);
    vc::objects::OpenOptions storeOptions; storeOptions.readOnly=true;
    auto store=vc::objects::ObjectCollection::open(cfg.at("fibers").get<std::string>(),storeOptions);
    const auto txn=store.read();
    vc::lasagna::LasagnaDatasetOpenOptions open;
    open.workingToBaseScale=1.;
    open.remoteCacheRoot=cfg.at("remote_cache").get<std::string>();
    auto normalInput=openFiberConstraintSource(cfg,"normal",scan,open);
    auto dataset=std::move(normalInput.dataset);
    auto provenance=std::move(normalInput.provenance);
    const auto normalSource=cfg.value("normal_source",std::string("lasagna"));
    if(normalSource=="fiber"){
        // The compound sheet_grad_mag channel is a validity mask, never a
        // Lasagna winding-density measurement.
        if(!cfg.contains("orient_database") && !cfg.value("axial_prior_only",false) && !cfg.contains("winding_manifest"))
            throw std::invalid_argument("Fiber sheet normals require an independent Lasagna winding_manifest for extraction");
        dataset=vc::lasagna::fiberSheetNormalDataset(dataset);
    }else if(normalSource!="lasagna")throw std::invalid_argument("normal_source must be lasagna or fiber");
    if (!dataset.manifest().baseShapeZYX)
        throw std::invalid_argument("Lasagna manifest has no base shape");
    const auto baseShapeZYX=*dataset.manifest().baseShapeZYX;
    std::optional<FiberConstraintSource> windingInput;
    if(cfg.contains("winding_manifest")) {
        windingInput=openFiberConstraintSource(cfg,"winding",scan,open);
        if(*windingInput->dataset.manifest().baseShapeZYX!=baseShapeZYX)
            throw std::invalid_argument("Normal and winding sources use different base shapes");
        if(windingInput->dataset.manifest().raw.value("normal_source",std::string{})=="fiber_sheet")
            throw std::invalid_argument("Fiber sheet validity is not winding density");
    }
    const auto queryStart=Clock::now();
    // Fiber UUID -> stored-to-volume mapping (bound or registered), in UUID
    // order: that order is the trace order and the database's source_index.
    std::map<std::string,vc::objects::VolumeAffine> members;
    const auto transforms=store.volumeTransforms(txn);
    std::set<std::string> requested;
    if(cfg.contains("fiber_ids"))for(const auto& id:cfg.at("fiber_ids"))requested.insert(id.get<std::string>());
    std::optional<std::array<double,6>> selectionBounds;
    if(!cfg.contains("fiber_ids")&&!cfg.value("full_volume",false)){
        const auto lower=xyz(cfg.at("minimum_xyz")),upper=xyz(cfg.at("maximum_xyz"));
        std::array<double,6> box;
        for(int d=0;d<3;++d){
            if(!std::isfinite(lower[d])||!std::isfinite(upper[d])||lower[d]>=upper[d])throw std::invalid_argument("Invalid selection box");
            box[d]=lower[d];box[d+3]=upper[d];
        }
        selectionBounds=box;
    }
    store.scanObjects(txn,[&](const auto& object){
        if(cfg.contains("fiber_ids")&&!requested.contains(object.record.uuid))return true;
        const auto mapping=transforms.toVolume(Json::parse(object.record.meta),volume);
        if(mapping)members.emplace(object.record.uuid,*mapping);
        else if(requested.contains(object.record.uuid))throw std::runtime_error("Requested fiber has no mapping to target volume");
        return true;
    },"fiber");
    if(cfg.contains("fiber_ids")&&members.size()!=requested.size())throw std::invalid_argument("Unknown requested fiber UUID");
    std::vector<std::string> ids;
    std::vector<vc::objects::VolumeAffine> mappings;
    for(const auto& [id,mapping]:members){ids.push_back(id);mappings.push_back(mapping);}
    std::size_t crossQueryPrefix=cfg.value("cross_query_prefix",std::size_t{0});
    if(cfg.contains("cross_query_fiber_ids")){
        if(crossQueryPrefix)throw std::invalid_argument("Use cross_query_fiber_ids, not both cross selection modes");
        const auto requestedCross=cfg.at("cross_query_fiber_ids").get<std::vector<std::string>>();
        const std::set<std::string> queries(requestedCross.begin(),requestedCross.end());
        if(queries.empty())throw std::invalid_argument("Cross-query fiber selection must not be empty");
        if(queries.size()!=requestedCross.size())throw std::invalid_argument("Duplicate cross-query UUID");
        for(const auto& id:queries)if(!members.contains(id))throw std::invalid_argument("Unknown cross-query fiber UUID");
        ids.clear();mappings.clear();
        for(bool query:{true,false})for(const auto& [id,mapping]:members)if(queries.contains(id)==query){
            ids.push_back(id);mappings.push_back(mapping);
        }
        crossQueryPrefix=queries.size();
        if(selectionBounds)throw std::invalid_argument("Cross-query UUID selection requires full_volume or explicit fiber_ids");
    }
    progress({"catalog",ids.size(),ids.size()});
    const double querySeconds=seconds(queryStart);
    const auto loadStart=Clock::now();
    std::vector<FiberletCropTraceLine> lines(ids.size());
    std::atomic<size_t> next{0},loaded{0};
    constexpr size_t kLoadBatch=1024;
    std::vector<std::future<void>> loaders;
    for(int worker=0;worker<std::min<int>(loadThreads,ids.size());++worker)loaders.push_back(std::async(std::launch::async,[&]{
        auto reader=vc::objects::ObjectCollection::open(cfg.at("fibers").get<std::string>(),storeOptions);
        auto read=reader.read();
        for(size_t begin;(begin=next.fetch_add(kLoadBatch))<ids.size();){
            const auto end=std::min(ids.size(),begin+kLoadBatch);
            for(size_t i=begin;i<end;++i){
                const vc::objects::ObjectKey key{"",ids[i]};
                const auto head=reader.currentGeometry(read,key);
                if(!head)throw std::runtime_error("Fiber has no geometry: "+ids[i]);
                const auto array=std::find_if(head->arrays.begin(),head->arrays.end(),[](const auto& a){return a.name=="line_points";});
                if(array==head->arrays.end()||array->shape.at(0)<2)continue;
                if(array->dtype!="<f8")throw std::runtime_error("Fiber line points must be float64");
                const auto bytes=reader.readGeometryArray(read,key,"line_points");
                auto& line=lines[i];
                for(const auto& stored:vc::objects::xyzFromBytes<double>(bytes)){
                    const auto p=mappings[i].apply(stored);line.pointsBaseXYZ.emplace_back(p[0],p[1],p[2]);
                }
                if(selectionBounds){
                    bool intersects=false;
                    for(size_t j=1;j<line.pointsBaseXYZ.size()&&!intersects;++j){
                        const auto& a=line.pointsBaseXYZ[j-1];const auto& b=line.pointsBaseXYZ[j];double first,last;
                        intersects=vc::geometry::clipSegment({a[0],a[1],a[2]},{b[0],b[1],b[2]},*selectionBounds,first,last);
                    }
                    if(!intersects)line.pointsBaseXYZ.clear();
                }
            }
            progress({"load",loaded.fetch_add(end-begin)+(end-begin),ids.size()});
        }
    }));
    for(auto& loader:loaders)loader.get();
    size_t kept=0;cv::Vec3d lo{INFINITY,INFINITY,INFINITY},hi{-INFINITY,-INFINITY,-INFINITY};
    double totalArc=0.;size_t totalPoints=0;
    for(size_t i=0;i<ids.size();++i){
        if(lines[i].pointsBaseXYZ.size()<2){
            if(crossQueryPrefix)throw std::invalid_argument("Cross-query graph contains a fiber without a polyline");
            continue;
        }
        for(const auto& p:lines[i].pointsBaseXYZ)for(int d=0;d<3;++d){lo[d]=std::min(lo[d],p[d]);hi[d]=std::max(hi[d],p[d]);}
        totalArc+=makePolylineArcGeometry(lines[i].pointsBaseXYZ).length();totalPoints+=lines[i].pointsBaseXYZ.size();
        if(i!=kept){ids[kept]=std::move(ids[i]);lines[kept]=std::move(lines[i]);mappings[kept]=mappings[i];}
        ++kept;
    }
    ids.resize(kept);lines.resize(kept);
    if(ids.empty())throw std::runtime_error("No fibers intersect selection");
    mappings.resize(ids.size());
    progress({"load",lines.size(),ids.size()});
    const double loadSeconds=seconds(loadStart);
    auto options=extractionOptions(cfg,threads,crossQueryPrefix,lines.size());
    if(cfg.value("geometry_only",false)) {
        // Use the EXACT production source catalogue, UUID ordering, transforms,
        // geometry loading and options above. No alternative fixture loader.
        // Stop before constructing readers/samplers or exporting graph data.
        Json result={{"config",cfg},{"geometry_only",true},{"selected_fibers",lines.size()},
            {"points",totalPoints},{"fibers_store_uuid",store.uuid(txn)},
            {"query_seconds",querySeconds},{"load_seconds",loadSeconds}};
        result["runs"]=benchmarkFiberGeometryStage(lines,options,cfg);
        result["wall_seconds"]=seconds(started);result["resources"]=processResources();
        const auto output=std::filesystem::path(cfg.at("output").get<std::string>());
        std::filesystem::create_directories(output.parent_path());
        vcr::zarr::atomicWrite(output,result.dump(2)+"\n");
        return 0;
    }
    vc::lasagna::LasagnaNormalSamplerOptions samplerOptions;
    samplerOptions.maxCachedBytes=cfg.value("normal_cache_bytes",samplerOptions.maxCachedBytes);
    const auto cacheMode=cfg.value("normal_cache_mode",std::string("auto"));
    if(cacheMode=="auto") {
        samplerOptions.autoCacheFraction=cfg.value("normal_cache_fraction",0.5);
        // One process budget, not one fraction of available RAM per dataset.
        samplerOptions.memoryCache=vcr::MemoryCache::automatic(*samplerOptions.autoCacheFraction);
    } else if(cacheMode!="fixed") {
        throw std::invalid_argument("normal_cache_mode must be fixed or auto");
    } else if(cfg.value("normal_cache_shared",false)) {
        samplerOptions.memoryCache=std::make_shared<vcr::MemoryCache>(samplerOptions.maxCachedBytes);
    }
    samplerOptions.channelCache=cfg.value("normal_channel_cache",false);
    samplerOptions.readerThreads=cfg.value("reader_threads",size_t{0});
    samplerOptions.pointInterpolation=cfg.value("normal_point_interpolation",true);
    samplerOptions.windingPointInterpolation=cfg.value("winding_point_interpolation",true);
    samplerOptions.windingRegionBaseVoxels=cfg.value("winding_region_base_voxels",
        cfg.value("scoring_region_base_voxels",4096.));
    samplerOptions.windingBatchPoints=cfg.value("winding_batch_points",size_t{262144});
    const vc::lasagna::LasagnaNormalSampler normals(dataset,samplerOptions);
    std::unique_ptr<vc::lasagna::LasagnaNormalSampler> separateWindings;
    if(windingInput)separateWindings=std::make_unique<vc::lasagna::LasagnaNormalSampler>(windingInput->dataset,samplerOptions);
    const auto& windingSampler=separateWindings?*separateWindings:normals;
    const auto maintenanceWorkers=cfg.value("normal_cache_maintenance_workers",std::size_t(0));
    if(maintenanceWorkers) {
        std::set<vcr::MemoryCache*> configured;
        for(const auto* sampler : {&normals,&windingSampler})
            for(const auto& binding:sampler->regionChannels())
                if(const auto& cache=binding.array->memory_cache(); cache && configured.insert(cache.get()).second)
                    cache->enable_maintenance(maintenanceWorkers,cfg.value("normal_cache_candidate_scan",std::size_t(8)));
    }
    if(cfg.value("normal_cache_profile",false))
        for(const auto* sampler : {&normals,&windingSampler})
            for(const auto& binding:sampler->regionChannels())
                if(const auto& cache=binding.array->memory_cache()) cache->enable_profiling();
    const auto readerStats=[](const auto& sampler) {
        Json result={{"arrays",Json::array()}};
        std::set<const vcr::Array*> seen;
        for(const auto& binding:sampler.regionChannels()) {
            if(!seen.insert(binding.array.get()).second)continue;
            const auto& stats=binding.array->stats();
            result["arrays"].push_back({{"memory_hits",stats.memory_hits.load()},
                {"disk_hits",stats.disk_hits.load()},{"remote_fetches",stats.remote_fetches.load()},
                {"remote_bytes",stats.remote_bytes.load()},{"corrupt",stats.corrupt.load()},
                {"channel_read_bytes",stats.channel_read_bytes.load()},
                {"channel_decoded_bytes",stats.channel_decoded_bytes.load()}});
            if(const auto& cache=binding.array->memory_cache()) {
                result["cache"]={{"bytes",cache->nbytes()},{"entries",cache->size()},
                    {"hits",cache->hits()},{"misses",cache->misses()},{"limit_bytes",cache->max_bytes()}};
                const auto maintenance=cache->maintenance_stats();
                result["cache"]["maintenance"]={{"evicted_entries",maintenance.evicted_entries},
                    {"batches",maintenance.batches},{"assist_puts",maintenance.assist_puts},
                    {"headroom_bytes",maintenance.headroom_bytes}};
                const auto profile=cache->profile();
                const auto timing=[](const auto& t){return Json{{"calls",t.calls},
                    {"wait_seconds",t.wait_ns*1e-9},{"hold_seconds",t.hold_ns*1e-9}};};
                result["cache"]["lock_profile"]={{"get",timing(profile.get)},{"put",timing(profile.put)},
                    {"erase",timing(profile.erase)},{"refresh",timing(profile.refresh)},
                    {"refresh_calls",profile.refresh_calls},{"refresh_probe_seconds",profile.refresh_probe_ns*1e-9}};
            }
        }
        return result;
    };
    progress.readerStats([&] {
        return Json{{"normals",readerStats(normals)},{"windings",readerStats(windingSampler)}};
    });
    if(cfg.contains("normal_sampling_benchmark")) {
        const auto timings=benchmarkFiberNormalSampling(cfg.at("normal_sampling_benchmark"),lines,dataset,samplerOptions);
        const Json benchmark={{"normal_sampling_benchmark",timings},{"query_seconds",querySeconds},
            {"load_seconds",loadSeconds},{"wall_seconds",seconds(started)},{"resources",processResources()}};
        vcr::zarr::atomicWrite(std::filesystem::path(cfg.at("output").get<std::string>()),benchmark.dump(2)+"\n");
        std::cout<<benchmark.dump()<<std::endl;return 0;
    }
    vc::lasagna::LasagnaGroupArrays arrays(0);
    const double spacing=vc::lasagna::bindLasagnaChannel(dataset.manifest(),"nx",arrays).spacing;
    std::cout<<"selected fibers="<<lines.size()<<" points="<<totalPoints<<" arc_base="<<totalArc
             <<" unsigned normals; no alignment"<<std::endl;
    Json result={{"config",cfg},{"fibers_store_uuid",store.uuid(txn)},
        {"normal_manifest",dataset.manifest().raw},{"normal_provenance",provenance},
        {"selected_ids",ids},{"selected_fibers",lines.size()},{"points",totalPoints},
        {"fiber_base_shape_zyx",baseShapeZYX},
        {"normal_base_shape_zyx",*dataset.manifest().baseShapeZYX},
        {"arc_base",totalArc},{"query_seconds",querySeconds},{"load_seconds",loadSeconds},
        {"normal_alignment",false},
        {"geometry_minimum_xyz",{lo[0],lo[1],lo[2]}},
        {"geometry_maximum_xyz",{hi[0],hi[1],hi[2]}},
        {"settings",{{"local_events",true},{"keep_whole_fibers",true},
            {"load_threads",loadThreads},{"threads",threads},
            {"certify_winding",*options.certifyWinding},
            {"enforce_maximum_winding",options.enforceMaximumWindingDistance},
            {"closest_parallel_estimator",*options.closestParallelEstimator},
            {"parallel_reciprocal",options.localParallelReciprocal},{"perpendicular_iterations",options.localPerpendicularProjectionIterations},
            {"moving_projection_windows",options.localMovingProjectionWindows},
            {"event_spacing",options.localSpacingBaseVoxels},{"tangent_span",options.localTangentSpanBaseVoxels},
            {"candidate_batch_size",options.candidateBatchSize},
            {"scoring_region_base_voxels",options.scoringRegionBaseVoxels},
            {"parallel_span",options.localParallelSpanBaseVoxels},{"maximum_distance",options.maximumDistanceBaseVoxels},
            {"integration_step",options.windingIntegrationStepBaseVoxels},{"maximum_windings",options.maximumWindingDistance},
            {"reciprocal_tolerance",options.localReciprocalToleranceBaseVoxels},
            {"angular_tolerance",options.localAngularToleranceDegrees},{"distance_tolerance",options.localDistanceTolerance},
            {"refinement_rounds",options.localRefinementRounds},{"normalize_evidence",options.localNormalizeEvidence}}},
        {"normal_spacing_base",spacing},{"runs",Json::array()}};
    if(windingInput) {
        result["winding_manifest"]=windingInput->dataset.manifest().raw;
        result["winding_provenance"]=windingInput->provenance;
    }
    const auto output=std::filesystem::path(cfg.at("output").get<std::string>());
    std::filesystem::create_directories(output.parent_path());
    const auto loadAxis=[&]() {
        // The volume's grid, as the Lasagna manifest states it; the umbilicus
        // file's own voxel size is not cross-checked. Only a frame the file
        // states is used; an inferred or unstated one would need confirming.
        const auto& shape=baseShapeZYX;
        const auto file=vc::objects::readUmbilicusFile(cfg.at("umbilicus").get<std::string>());
        const auto binding=vc::objects::umbilicusBinding(file,
            {{double(shape[2]),double(shape[1]),double(shape[0])},std::nullopt},std::nullopt);
        if(binding.refusal)throw std::runtime_error("Umbilicus refused: "+*binding.refusal);
        if(binding.confirm || !binding.storedUnitsPerVoxel)
            throw std::runtime_error("Umbilicus frame not verified: the file does not state its frame");
        result["umbilicus_frame"]=binding.frame;
        std::vector<cv::Vec3f> points;points.reserve(file.points.size());
        for(const auto& p:file.points)points.emplace_back(cv::Vec3f(float(p[0]),float(p[1]),float(p[2]))/float(*binding.storedUnitsPerVoxel));
        return vc::core::util::Umbilicus::FromPoints(std::move(points),{int(shape[0]),int(shape[1]),int(shape[2])});
    };
    if(cfg.contains("axial_prior_output")) {
        const auto target=std::filesystem::path(cfg.at("axial_prior_output").get<std::string>());
        if(std::filesystem::exists(target))throw std::runtime_error("Axial output already exists");
        const auto axis=loadAxis();const auto& centers=axis.centers();
        const double step=cfg.value("axial_sample_spacing",16.);
        const auto axisAt=[&](const cv::Vec3d& p) {
            const int z=std::clamp(int(std::lround(p[2])),0,int(centers.size())-1);
            const int lo=std::max(0,z-8),hi=std::min(int(centers.size())-1,z+8);
            return cv::Vec3d(centers[hi]-centers[lo]);
        };
        std::vector<double> agreement(lines.size());
        const auto begin=Clock::now();
        // Fixed per-fiber accumulation order; independent fibers run in parallel.
        cv::parallel_for_(cv::Range(0,int(lines.size())),[&](const cv::Range& range) {
            for(int i=range.start;i<range.end;++i)
                agreement[i]=fiberAxialAlignment(lines[i].pointsBaseXYZ,axisAt,step);
        });
        Json features={{"format","fiber-axial-priors"},{"version",1},{"fibers_store_uuid",result["fibers_store_uuid"]},
            {"source_ids",ids},{"agreement",agreement},{"umbilicus",cfg.at("umbilicus")},
            {"fibers_digest",cfg.at("fibers_digest")},{"umbilicus_sha256",cfg.at("umbilicus_sha256")},
            {"sample_spacing_base",step},{"axis_secant_halfspan_z",8},
            {"semantics","arc mean 2*(fiber tangent dot axis tangent)^2-1; V positive, H negative"},
            {"seconds",seconds(begin)},{"umbilicus_frame",result["umbilicus_frame"]}};
        std::filesystem::create_directories(target.parent_path());
        vcr::zarr::atomicWrite(target,features.dump()+"\n");
        result["axial_seconds"]=seconds(begin);
        if(cfg.value("axial_prior_only",false)) {
            result["wall_seconds"]=seconds(started);
            vcr::zarr::atomicWrite(output,result.dump(2)+"\n");
            std::cout<<"axial features complete fibers="<<lines.size()<<" seconds="<<seconds(started)<<std::endl;
            return 0;
        }
    }
    if(cfg.contains("orient_database")) {
        if(repetitions!=1 || cfg.contains("compare_database"))
            throw std::invalid_argument("Orientation is one second pass, not extraction/comparison");
        std::optional<vc::core::util::Umbilicus> umbilicus;
        if(cfg.contains("normal_prior_output")) {
            umbilicus=loadAxis();
        }
        result["orientation"]=orientFiberConstraintDatabase(
            cfg.at("orient_database").get<std::string>(),cfg.at(cfg.contains("normal_curves_output")?"normal_curves_output":(umbilicus?"normal_prior_output":"database")).get<std::string>(),
            ids,lines,normals,threads,
            cfg.value("normal_sample_spacing",spacing),result,umbilicus?&*umbilicus:nullptr,
            cfg.contains("normal_curves_output"),cfg.contains("fiber_ids"));
        result["wall_seconds"]=seconds(started);
        vcr::zarr::atomicWrite(output,result.dump(2)+"\n");
        std::cout<<"orientation complete "<<result["orientation"].dump()<<std::endl;
        return 0;
    }
    if(cfg.value("chunk_pipeline",jointNormals)) {
        if(!jointNormals || repetitions!=1 || cfg.contains("compare_database"))
            throw std::invalid_argument("Chunk pipeline requires one joint-normal extraction, without legacy comparison mode");
        auto databaseOptions=fiberConstraintDatabaseOptions(cfg);
        for(const auto& mapping:mappings)databaseOptions.baseToStored.push_back(mapping.inverse());
        const auto axis=loadAxis();
        result["chunk_pipeline"]=extractFiberConstraintChunks(cfg,result,ids,lines,options,normals,windingSampler,axis,std::move(databaseOptions),[&] {
            // The two samplers share the auto cache: count each cache once.
            Json caches=Json::array();std::set<const vcr::MemoryCache*> seen;size_t bytes=0;
            for(const auto* sampler:{&normals,&windingSampler})for(const auto& binding:sampler->regionChannels())
                if(const auto& cache=binding.array->memory_cache();cache&&seen.insert(cache.get()).second) {
                    const auto resident=cache->nbytes();bytes+=resident;
                    caches.push_back({{"bytes",resident},{"entries",cache->size()},{"limit_bytes",cache->max_bytes()}});
                }
            Json snapshot={{"decoded_cache_bytes",bytes},{"caches",caches},{"resources",processResources()}};
            if(cfg.value("normal_cache_profile",false))
                snapshot["reader_stats"]={{"normals",readerStats(normals)},{"windings",readerStats(windingSampler)}};
            return snapshot;
        });
        if(cfg.value("saturated_chunk_benchmark",false))
            result["saturated_chunk_benchmark"]=result["chunk_pipeline"].at("benchmark");
        else result["export"]=result["chunk_pipeline"]["export"];
        result["wall_seconds"]=seconds(started);
        result["reader_stats"]={{"normals",readerStats(normals)},{"windings",readerStats(windingSampler)}};
        result["resources"]=processResources();
        vcr::zarr::atomicWrite(output,result.dump(2)+"\n");
        std::cout<<"chunk extraction complete "<<result["chunk_pipeline"].dump()<<std::endl;
        return 0;
    }
    for(int i=0;i<repetitions;++i) {
        const auto begin=Clock::now();
        std::cout<<"extraction repetition="<<i<<std::endl;
        const auto extract=[&](const FiberTraceConstraintStream& stream) {
            return extractFiberTraceConstraints(lines,options,
                [&](const auto& a,const auto& b,double step){return windingSampler.normalAlignedWindingDistance(a,b,step);},
                [&](const auto& pairs,double step,int workers){return windingSampler.normalAlignedWindingDistancesBatch(pairs,step,workers);},
                {},nullptr,progress.enabled()?FiberTraceConstraintProgressCallback(
                    [&](const auto& p){progress(p);}):FiberTraceConstraintProgressCallback{},stream);
        };
        auto databaseOptions=fiberConstraintDatabaseOptions(cfg);
        databaseOptions.baseToStored.reserve(mappings.size());
        for(const auto& mapping:mappings)databaseOptions.baseToStored.push_back(mapping.inverse());
        const bool streaming=cfg.contains("database") && cfg.value("stream_export",true) && !jointNormals;
        double extractionSeconds=0.;
        const auto report=[&]() {
            if(!streaming) {
                auto report=extract({});extractionSeconds=seconds(begin);return report;
            }
            std::promise<std::vector<FiberTraceConstraintPiece>> prepared;
            auto preparedFuture=prepared.get_future();
            std::promise<FiberConstraintBatchConsumer> consumerReady;
            auto consumerFuture=consumerReady.get_future().share();
            auto extractionDone=begin;
            std::optional<FiberTraceConstraintReport> extracted;
            // The extraction worker samples normals and prepares its own
            // completed events, then emits directly to independent shard queues.
            // Source job IDs keep identity independent of completion order.
            auto extraction=std::async(std::launch::async,[&] {
                bool delivered=false;
                try {
                    FiberTraceConstraintStream stream;
                    stream.prepared=[&](auto pieces) {
                        std::vector<FiberTraceConstraintPiece> compact;
                        compact.reserve(pieces.size());
                        for(const auto& p:pieces) {
                            FiberTraceConstraintPiece q;
                            q.traceIndex=p.traceIndex;q.pieceIndex=p.pieceIndex;
                            q.beginArcBaseVoxels=p.beginArcBaseVoxels;q.endArcBaseVoxels=p.endArcBaseVoxels;
                            compact.push_back(std::move(q));
                        }
                        prepared.set_value(std::move(compact));delivered=true;
                    };
                    stream.batch=[&](auto worker,auto index,auto events) {
                        if(index>0x7fffffffULL || events.size()>0xffffffffULL)
                            throw std::overflow_error("Constraint source event ID exceeds int64 range");
                        consumerFuture.get()(worker,std::size_t(index)*0x100000000ULL,events);
                    };
                    auto report=extract(stream);
                    if(report.pieces.size()!=lines.size() || report.hardConstraints || report.localMidpointCuts)
                        throw std::runtime_error("Unexpected split/continuation or degenerate source");
                    extractionDone=Clock::now();extractionSeconds=seconds(begin);
                    return report;
                } catch(...) {
                    const auto failure=std::current_exception();
                    if(!delivered)prepared.set_exception(failure);
                    throw;
                }
            });
            bool consumerDelivered=false;
            try {
                const auto pieces=preparedFuture.get();
                // Progress maintains independent timelines for concurrent phases.
                result["export"]=streamFiberConstraintDatabase(
                    cfg.at("database").get<std::string>(),ids,pieces,
                    {},normals,threads,result,
                    progress.enabled()?FiberTraceConstraintProgressCallback([&](const auto& p){progress(p);}):
                        FiberTraceConstraintProgressCallback{},databaseOptions,0,
                    [&](const auto& consumer) {
                        consumerReady.set_value(consumer);consumerDelivered=true;
                        // Propagate failures before the store can be finalized.
                        extracted.emplace(extraction.get());
                    },std::max(1,threads));
                auto report=std::move(extracted.value());
                result["export"]["streamed"]=true;
                result["export"]["tail_after_extraction_seconds"]=seconds(extractionDone);
                return report;
            } catch(...) {
                const auto failure=std::current_exception();
                try {std::rethrow_exception(failure);}
                catch(const std::exception& e){std::cerr<<"constraint stream failed: "<<e.what()<<std::endl;}
                catch(...){std::cerr<<"constraint stream failed: unknown exception"<<std::endl;}
                if(!consumerDelivered)consumerReady.set_exception(failure);
                // Writer failures wake shard-queue producers through the shared
                // import stream failure state; no ordered queue needs draining.
                if(extraction.valid())try{extraction.get();}catch(...){}
                throw;
            }
        }();
        if(report.pieces.size()!=lines.size() || report.hardConstraints || report.localMidpointCuts)
            throw std::runtime_error("Unexpected split/continuation or degenerate source");
        if(jointNormals) {
            const auto axis=loadAxis();
            progress({"normal_sweep",0,lines.size()});
            auto sampled=sampleFiberConstraintNormals(ids,lines,report,normals,threads,
                cfg.value("normal_sample_spacing",spacing),&axis,
                FiberNormalBatchOptions{cfg.value("normal_region_base_voxels",4096.),
                    cfg.value("normal_batch_points",size_t{262144})});
            progress({"normal_sweep",lines.size(),lines.size()});
            result["normal_orientation"]=sampled.statistics;
            databaseOptions.normalSamples=&sampled;
            result["export"]=writeFiberConstraintDatabase(cfg.at("database").get<std::string>(),ids,
                report,normals,threads,result,
                progress.enabled()?FiberTraceConstraintProgressCallback([&](const auto& p){progress(p);}):
                    FiberTraceConstraintProgressCallback{},databaseOptions);
            writeFiberConstraintOutwardPriors(cfg.at("normal_prior_output").get<std::string>(),
                ids,sampled,result.at("export").at("source_store"),result);
            databaseOptions.normalSamples=nullptr;
        }
        std::size_t parallel=0,signedCount=0;
        for(const auto& c:report.constraints){parallel+=c.parallelScore>.5;signedCount+=c.signedWindingDelta.has_value();}
        Json run={{"seconds",extractionSeconds},{"prepare_seconds",report.prepareSeconds},
            {"search_seconds",report.searchSeconds},{"orientation_seconds",report.orientationScoreSeconds},
            {"winding_seconds",report.windingScoreSeconds},{"resampled_points",report.resampledPoints},
            {"candidates",report.measuredCandidates},{"refined_candidates",report.localRefinedCandidates},
            {"events",report.constraints.size()},{"parallel",parallel},
            {"orthogonal",report.constraints.size()-parallel},{"signed",signedCount},
            {"rejected_consistency",report.localRejectedConsistency},{"rejected_winding",report.rejectedWinding},
            {"rejected_support",report.localRejectedSupport},{"unsigned_events",report.constraints.size()-signedCount},
            {"rejected_winding_cutoff",report.rejectedWindingCutoff}};
        run["resources"]=processResources();
        run["scoring_batches"]=report.scoringBatches;
        run["scoring_regions"]=report.scoringRegions;
        run["winding_batches"]=report.windingBatches;
        run["winding_regions"]=report.windingRegions;
        run["maximum_winding_batch_candidates"]=report.maximumWindingBatchCandidates;
        run["maximum_batch_candidates"]=report.maximumBatchCandidates;
        run["first_failure_checks"]=report.localFailureChecks;
        run["outcomes_by_initial_relation"]=report.localOutcomesByRelation;
        run["winding_overlap_checks"]=report.localWindingFailureOverlap;
        run["winding_checks_evaluated"]=report.localWindingChecksEvaluated;
        run["deduplicated_candidates"]=report.localDeduplicatedCandidates;
        run["perpendicular_reciprocal_evaluated"]=report.localPerpendicularReciprocalEvaluated;
        run["perpendicular_reciprocal_window_failures"]=report.localPerpendicularReciprocalWindowFailures;
        run["perpendicular_reciprocal_endpoint_failures"]=report.localPerpendicularReciprocalEndpointFailures;
        result["runs"].push_back(run);result["wall_seconds"]=seconds(started);
        if(cfg.contains("compare_database")) {
            const auto comparisonStarted=Clock::now();
            progress({"verify",0,report.constraints.size(),report.constraints.size()});
            verifyFiberConstraintDatabaseMeasurements(cfg.at("compare_database").get<std::string>(),report,ids);
            result["comparison_seconds"]=result.value("comparison_seconds",0.)+seconds(comparisonStarted);
            progress({"verify",report.constraints.size(),report.constraints.size(),report.constraints.size()});
            result["all_measurements_match_database"]=true;
        }
        if(cfg.contains("database") && !streaming && !jointNormals) {
            result["export"]=writeFiberConstraintDatabase(cfg.at("database").get<std::string>(),
                ids,report,normals,threads,result,
                progress.enabled()?FiberTraceConstraintProgressCallback([&](const auto& p){progress(p);}):
                    FiberTraceConstraintProgressCallback{},databaseOptions);
            result["wall_seconds"]=seconds(started);
            std::cout<<"export complete "<<result["export"].dump()<<std::endl;
        }
        result["wall_seconds"]=seconds(started);
        result["processing_seconds"]=result.at("wall_seconds").get<double>()-result.value("comparison_seconds",0.);
        result["resources"]=processResources();
        result["reader_stats"]={{"normals",readerStats(normals)},{"windings",readerStats(windingSampler)}};
        vcr::zarr::atomicWrite(output,result.dump(2)+"\n");
        std::cout<<run.dump()<<std::endl;
    }
    return 0;
}
