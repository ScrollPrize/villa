#include "FiberConstraintExtraction.hpp"
#include "FiberConstraintConfig.hpp"
#include "FiberConstraintGeometry.hpp"
#include "FiberGeometryStageBenchmark.hpp"
#include <vc_object_store/objects/ObjectCollection.hpp>
#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/geometry/SegmentClip.hpp>
#include <vc_object_store/geometry/ArrayBytes.hpp>
#include "FiberConstraintDatabase.hpp"
#include "FiberConstraintChunks.hpp"
#include "FiberCrossingAxes.hpp"
#include "FiberNeighbourNormals.hpp"
#include "FiberConstraintParallelism.hpp"
#include "FiberNormalSamplingBenchmark.hpp"
#include "FiberConstraintSources.hpp"
#include "vc/fiber_tracer/FiberAxialAlignment.hpp"
#include "vc/fiber_tracer/FiberBendSeeds.hpp"
#include "vc/fiber_tracer/FiberPredictionNormals.hpp"
#include "vc/fiber_tracer/FiberStripNormals.hpp"
#include <vc_object_store/objects/FiberConstraintGraph.hpp>
#include <vc_object_store/objects/FiberGraphSolveExtension.hpp>
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
    options.maximumDistanceBaseVoxels=cfg.value("maximum_distance_base_voxels",options.maximumDistanceBaseVoxels);
    if(!(options.maximumDistanceBaseVoxels>0.))throw std::invalid_argument("maximum_distance_base_voxels must be positive");
    options.crossQueryPrefix=crossQueryPrefix;
    if(crossQueryPrefix>=lineCount&&crossQueryPrefix)throw std::invalid_argument("Cross query prefix must leave reference lines");
    options.certifyWinding=cfg.value("certify_winding",false);
    // winding_distances:false measures geometry and normals only; the
    // stored constraints carry no winding distance (byte 0).
    options.measureWindingDistances=cfg.value("winding_distances",true);
    options.enforceMaximumWindingDistance=cfg.value("enforce_maximum_winding",false);
    options.closestParallelEstimator=cfg.value("closest_parallel_estimator",false);
    options.localParallelReciprocal=cfg.value("parallel_reciprocal",false);
    options.localMovingProjectionWindows=cfg.value("moving_projection_windows",true);
    options.localPerpendicularProjectionIterations=cfg.value("perpendicular_iterations",16u);
    options.pairOwnedGeometry=cfg.value("pair_owned_geometry",false);
    options.localMinimumSeeds=cfg.value("local_minimum_seeds",false);
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
    if(cfg.contains("neighbour_normals_output")) {
        std::cout<<"neighbour normals "<<inferFiberNeighbourNormals(cfg).dump()<<std::endl;
        return 0;
    }
    if(cfg.contains("crossing_filter_output")) {
        std::cout<<"crossing filter "<<writeFiberCrossingFilter(cfg).dump()<<std::endl;
        return 0;
    }
    if(cfg.contains("ribbon_crossings_output")) {
        std::cout<<"ribbon crossings "<<writeFiberRibbonCrossings(cfg).dump()<<std::endl;
        return 0;
    }
    if(cfg.contains("signed_strip_samples_output")) {
        std::cout<<"signed strip samples "<<writeSignedFiberStripSamples(cfg).dump()<<std::endl;
        return 0;
    }
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
    const int loadThreads=cfg.value("load_threads",threads);
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
    // Status, selection, transforms and payloads all use this same read snapshot.
    const auto sourceStatus=store.provisioning(txn);
    if(!sourceStatus.sealed&&sourceStatus.history!=vc::objects::HistoryMode::On)
        throw std::invalid_argument("Fiber source must be sealed or history enabled; input was not modified");
    vc::lasagna::LasagnaDatasetOpenOptions open;
    open.workingToBaseScale=1.;
    open.remoteCacheRoot=cfg.at("remote_cache").get<std::string>();
    auto normalInput=openFiberConstraintSource(cfg,"normal",scan,open);
    auto dataset=std::move(normalInput.dataset);
    auto provenance=std::move(normalInput.provenance);
    const auto normalSource=cfg.value("normal_source",std::string("lasagna"));
    // "fused": the compound sheet axis, weighted by its presence, pulls the geometry
    // strip normals (chunk pipeline). The presence channel is bound from the
    // prediction's own manifest, which the sheet-normal view below renames.
    const auto predictionManifest=dataset.manifest();
    if(normalSource=="fiber"||normalSource=="fused"){
        if(normalSource=="fused"&&!cfg.value("chunk_pipeline",true))
            throw std::invalid_argument("normal_source fused requires the chunk pipeline");
        // Compound sheet normals carry no winding density: their validity is
        // the prediction's (chunk existence or a channel), not a measurement.
        if(!cfg.contains("orient_database") && !cfg.value("axial_prior_only",false) && !cfg.contains("winding_manifest") &&
           cfg.value("winding_distances",true))
            throw std::invalid_argument("Fiber sheet normals require an independent Lasagna winding_manifest for extraction, or winding_distances:false");
        dataset=vc::lasagna::fiberSheetNormalDataset(dataset);
    }else if(normalSource=="geometry"){
        // Strip normals from the fiber geometry (chunk pipeline). The manifest
        // still supplies the base shape and, unless winding_distances is
        // false, the winding density.
        if(!cfg.value("chunk_pipeline",true))throw std::invalid_argument("normal_source geometry requires the chunk pipeline");
    }else if(normalSource!="lasagna")throw std::invalid_argument("normal_source must be lasagna, fiber, geometry or fused");
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
    {
        // One bulk column read of every fiber's line points, in the target volume's
        // coordinates, on load_threads native workers (the per-fiber reads this
        // replaced decoded one object at a time).
        std::vector<std::array<unsigned char,16>> keys(ids.size());
        std::map<std::array<unsigned char,16>,size_t> keyIndex;
        for(size_t i=0;i<ids.size();++i){
            const auto canonical=vc::objects::util::canonicalUuid(ids[i]);
            size_t k=0;
            for(size_t c=0;c<canonical.size();++c){
                if(canonical[c]=='-')continue;
                const auto hex=[](char h){return h<='9'?h-'0':h-'a'+10;};
                keys[i][k/2]=static_cast<unsigned char>((keys[i][k/2]<<4)|hex(canonical[c]));++k;
            }
            keyIndex[keys[i]]=i;
        }
        const vc::objects::StoreStatus pinned{store.uuid(txn),sourceStatus.generation,sourceStatus.commitSequence};
        const auto columns=store.readGeometryArrayColumns(txn,keys,"line_points","fiber",volume,pinned,
            size_t(loadThreads),false,store.supportsTrustedGeometryReads(txn));
        if(columns.geometry.dtype!="<f8"||columns.geometry.shape.size()!=2||columns.geometry.shape[1]!=3)
            throw std::runtime_error("Fiber line points must be float64 xyz");
        const auto* values=reinterpret_cast<const double*>(columns.geometry.data.data());
        for(size_t k=0;k<columns.objectUuids.size();++k){
            const auto found=keyIndex.find(columns.objectUuids[k]);
            if(found==keyIndex.end())throw std::runtime_error("Bulk geometry read returned an unrequested fiber");
            const auto i=found->second;
            const auto begin=columns.geometry.offsets.at(k),end=columns.geometry.offsets.at(k+1);
            if(end<begin+2)continue;
            auto& line=lines[i];line.pointsBaseXYZ.reserve(end-begin);
            for(auto row=begin;row<end;++row)line.pointsBaseXYZ.emplace_back(values[row*3],values[row*3+1],values[row*3+2]);
            if(selectionBounds){
                bool intersects=false;
                for(size_t j=1;j<line.pointsBaseXYZ.size()&&!intersects;++j){
                    const auto& a=line.pointsBaseXYZ[j-1];const auto& b=line.pointsBaseXYZ[j];double first,last;
                    intersects=vc::geometry::clipSegment({a[0],a[1],a[2]},{b[0],b[1],b[2]},*selectionBounds,first,last);
                }
                if(!intersects)line.pointsBaseXYZ.clear();
            }
        }
        progress({"load",ids.size(),ids.size()});
    }
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
    if(cfg.contains("crossing_axes_output")) {
        // Crossing-axis extraction: the geometry of every perpendicular crossing
        // of a crossing search store (crossing_search_store) against the loaded
        // fibers, written once for later normal solves (crossing_axes).
        const auto axesStarted=Clock::now();
        std::vector<std::vector<cv::Vec3d>> polylines(lines.size());
        for(size_t i=0;i<lines.size();++i)polylines[i]=lines[i].pointsBaseXYZ;
        const auto axesPath=std::filesystem::path(cfg.at("crossing_axes_output").get<std::string>());
        std::filesystem::create_directories(axesPath.parent_path());
        writeFiberCrossingAxes(axesPath,cfg.at("crossing_search_store").get<std::string>(),ids,polylines,size_t(threads));
        Json result={{"config",cfg},{"crossing_axes_output",axesPath.string()},{"selected_fibers",lines.size()},
            {"query_seconds",querySeconds},{"load_seconds",loadSeconds},{"axes_seconds",seconds(axesStarted)},
            {"wall_seconds",seconds(started)},{"resources",processResources()}};
        const auto output=std::filesystem::path(cfg.at("output").get<std::string>());
        std::filesystem::create_directories(output.parent_path());
        vcr::zarr::atomicWrite(output,result.dump(2)+"\n");
        std::cout<<"crossing axes "<<result.dump()<<std::endl;
        return 0;
    }
    if(cfg.contains("bend_seeds_output")) {
        // Bend-surface seeds (FiberBendSeeds.hpp) over the smoothed, resampled chains of the
        // loaded fibers, with the families of the classification run in bend_seed_classes.
        // Output: magic VCBSEED1, u64 count, then per seed 16 doubles: midpoint (3), the two
        // directions (3 + 3), angle (degrees), node distance, family, and the two fibers'
        // indices in the order of <output>.json's ids.
        const auto seedsStarted=Clock::now();
        vc::objects::OpenOptions classOptions;classOptions.readOnly=true;
        auto classes=vc::objects::ObjectCollection::open(cfg.at("bend_seed_classes").get<std::string>(),classOptions);
        const auto classTxn=classes.read();
        const auto results=vc::objects::readFiberGraphResultColumns(classes,classTxn,
            vc::objects::selectFiberGraphRun(classes,classTxn,"classification"));
        constexpr size_t familyColumn=0;
        static_assert(std::string_view(vc::objects::FiberGraphResultColumns::names[familyColumn])=="family");
        std::map<vc::objects::FiberGraphUuid,int> familyOf;
        for(size_t r=0;r<results.fiberUuids.size();++r)
            if(results.valid[r][familyColumn]!=0)familyOf[results.fiberUuids[r]]=int(results.integers[r][familyColumn]);
        vc::fiber_tracer::FiberStripNormalOptions chainOptions;
        chainOptions.smoothingSigma=cfg.value("strip_smoothing_sigma",chainOptions.smoothingSigma);
        chainOptions.resampleSpacing=cfg.value("strip_resample_spacing",chainOptions.resampleSpacing);
        std::vector<std::vector<vc::fiber_tracer::FiberBendNode>> perFiber(lines.size());
        cv::parallel_for_(cv::Range(0,int(lines.size())),[&](const cv::Range& range){
            for(int fi=range.start;fi<range.end;++fi){
                const size_t i=size_t(fi);
                const auto family=familyOf.find(vc::objects::fiberGraphUuid(ids[i]));
                if(family==familyOf.end())continue;
                const auto chain=vc::fiber_tracer::fiberStripChain(lines[i].pointsBaseXYZ,chainOptions);
                for(size_t k=0;k<chain.points.size();++k){
                    cv::Vec3d t=chain.points[std::min(k+1,chain.points.size()-1)]-chain.points[k>0?k-1:0];
                    const double length=cv::norm(t);
                    if(!(length>0.))continue;
                    perFiber[i].push_back({chain.points[k],t/length,std::int64_t(i),family->second});
                }
            }
        },threads);
        std::vector<vc::fiber_tracer::FiberBendNode> nodes;
        for(const auto& list:perFiber)nodes.insert(nodes.end(),list.begin(),list.end());
        vc::fiber_tracer::FiberBendSeedOptions seedOptions;
        seedOptions.radius=cfg.value("bend_seed_radius",seedOptions.radius);
        seedOptions.minAngleDegrees=cfg.value("bend_seed_min_angle",seedOptions.minAngleDegrees);
        seedOptions.threads=unsigned(threads);
        const auto seeds=vc::fiber_tracer::fiberBendSeeds(nodes,seedOptions);
        const auto seedsPath=std::filesystem::path(cfg.at("bend_seeds_output").get<std::string>());
        std::filesystem::create_directories(seedsPath.parent_path());
        std::ofstream file(seedsPath,std::ios::binary);
        if(!file)throw std::runtime_error("bend seeds: cannot write "+seedsPath.string());
        const char magic[8]={'V','C','B','S','E','E','D','1'};file.write(magic,sizeof magic);
        const uint64_t count=seeds.size();file.write(reinterpret_cast<const char*>(&count),sizeof count);
        for(const auto& seed:seeds){
            const std::array<double,16> v{seed.point[0],seed.point[1],seed.point[2],seed.directionA[0],seed.directionA[1],seed.directionA[2],
                seed.directionB[0],seed.directionB[1],seed.directionB[2],seed.angleDegrees,seed.distance,double(seed.family),
                double(seed.fiberA),double(seed.fiberB),0.,0.};
            file.write(reinterpret_cast<const char*>(v.data()),sizeof v);
        }
        if(!file)throw std::runtime_error("bend seeds: write failed "+seedsPath.string());
        std::ofstream(seedsPath.string()+".json")<<Json{{"record_doubles",16},{"ids",ids}}.dump();
        Json result={{"config",cfg},{"bend_seeds_output",seedsPath.string()},{"selected_fibers",lines.size()},{"nodes",nodes.size()},
            {"seeds",seeds.size()},{"radius",seedOptions.radius},{"min_angle_degrees",seedOptions.minAngleDegrees},
            {"load_seconds",loadSeconds},{"seed_seconds",seconds(seedsStarted)},{"wall_seconds",seconds(started)},{"resources",processResources()}};
        const auto output=std::filesystem::path(cfg.at("output").get<std::string>());
        std::filesystem::create_directories(output.parent_path());
        vcr::zarr::atomicWrite(output,result.dump(2)+"\n");
        std::cout<<"bend seeds "<<result.dump()<<std::endl;
        return 0;
    }
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
    std::unique_ptr<vc::lasagna::LasagnaGroupArrays> presenceArrays;
    std::optional<vc::lasagna::LasagnaChannelBinding> presence;
    if(normalSource=="fused"){
        presenceArrays=samplerOptions.memoryCache
            ?std::make_unique<vc::lasagna::LasagnaGroupArrays>(samplerOptions.memoryCache,samplerOptions.channelCache)
            :std::make_unique<vc::lasagna::LasagnaGroupArrays>(samplerOptions.maxCachedBytes,samplerOptions.channelCache);
        presenceArrays->readerThreads=samplerOptions.readerThreads;
        presence=vc::lasagna::bindLasagnaChannel(predictionManifest,"presence",*presenceArrays);
    }
    if(cfg.contains("prediction_normals_output")) {
        // Experiment: normals from the fiber prediction close to the fiber
        // (FiberPredictionNormals.hpp). Per fiber the annotated line is sampled every
        // prediction_normal_spacing voxels within prediction_normal_arc_range (widened by the
        // largest `along`); around each sample the prediction is read on a cubic grid of
        // prediction_normal_step voxels within the largest ellipse axis. Each entry of
        // prediction_normal_variants gives one estimate per sample:
        //   height, width  the ellipse
        //   along          the scatters of the samples within this arc distance are summed
        //                  with a Gaussian of sigma along/2 (0: the sample alone)
        //   mode           "sample": ellipse of each prediction sample's own normal;
        //                  "two_pass": a first isotropic estimate, then the ellipse of that estimate;
        //                  "condense": "sample", then `iterations` rounds that weight every
        //                  prediction sample by |normal . previous estimate|^power;
        //                  "track": a running estimate from the best supported sample outwards
        //                  (trackFiberPredictionNormals; `along` is its decay length, `power` its agreement power)
        //   Options of the along-fiber sum (modes sample, two_pass, condense):
        //   min_support    the window grows from 0 only until the summed weight within it reaches this
        //                  (at most `along`); 0: always `along`
        //   turn           each neighbour's scatter is turned with the fiber direction before it is added
        //   slice_power    the sum is repeated with each neighbour weighted by |its own axis . first sum|^slice_power
        //   balance        the samples on either side of a first estimate's plane get equal summed weight
        //   presence_base  share of the sample weight that does not depend on presence (1: ignored)
        //   strip          > 0: the estimates become sign-free targets of a strip solve over the sampled arc
        //                  range, with this axis weight times min(1, support / strip_support)
        // Output doubles per sample: fiber index, arc, point (3), tangent (3), then per variant
        // axis (3), weight, coherence, count.
        if(normalSource!="fused")throw std::invalid_argument("prediction_normals_output requires normal_source fused");
        const vc::lasagna::LasagnaNormalSampler directions(vc::lasagna::LasagnaDataset(predictionManifest),samplerOptions);
        struct Variant{double height,width,along,power;std::string mode;int iterations;
            double minSupport,slicePower,presenceBase,strip,stripSupport;bool turn,balance;};
        std::vector<Variant> variants;
        for(const auto& v:cfg.at("prediction_normal_variants")){
            Variant variant{v.at("height").get<double>(),v.at("width").get<double>(),v.value("along",0.),v.value("power",4.),
                v.value("mode",std::string("sample")),v.value("iterations",2),
                v.value("min_support",0.),v.value("slice_power",0.),v.value("presence_base",cfg.value("prediction_normal_presence_base",1.)),
                v.value("strip",0.),v.value("strip_support",10.),v.value("turn",false),v.value("balance",false)};
            if(variant.mode!="sample"&&variant.mode!="two_pass"&&variant.mode!="condense"&&variant.mode!="track")
                throw std::invalid_argument("prediction_normal_variants mode must be sample, two_pass, condense or track");
            if(!(variant.along>=0.)||variant.iterations<1)throw std::invalid_argument("prediction_normal_variants: invalid along or iterations");
            variants.push_back(std::move(variant));
        }
        if(variants.empty())throw std::invalid_argument("prediction_normal_variants is empty");
        const double sampleSpacing=cfg.value("prediction_normal_spacing",32.),step=cfg.value("prediction_normal_step",4.);
        const double tangentSpan=cfg.value("prediction_normal_tangent_span",64.);
        const auto arcRange=cfg.value("prediction_normal_arc_range",std::array<double,2>{0.,std::numeric_limits<double>::infinity()});
        vc::fiber_tracer::FiberPredictionNormalOptions base;
        base.presenceFloor=cfg.value("fused_presence_floor",base.presenceFloor);
        base.presenceFull=cfg.value("fused_presence_full",base.presenceFull);
        base.minAlignment=cfg.value("prediction_normal_min_alignment",base.minAlignment);
        double radius=0.,reachAlong=0.;
        for(const auto& v:variants){radius=std::max({radius,v.height,v.width});reachAlong=std::max(reachAlong,v.along);}
        std::vector<cv::Vec3d> grid;
        const int reach=int(std::ceil(radius/step));
        for(int z=-reach;z<=reach;++z)for(int y=-reach;y<=reach;++y)for(int x=-reach;x<=reach;++x){
            const cv::Vec3d offset(x*step,y*step,z*step);
            if(cv::norm(offset)<=radius)grid.push_back(offset);
        }
        struct Node{size_t fiber;double arc;cv::Vec3d point,tangent;};
        std::vector<Node> nodes;
        for(size_t i=0;i<lines.size();++i){
            const auto geometry=makePolylineArcGeometry(lines[i].pointsBaseXYZ);
            const double first=std::ceil(std::max(0.,arcRange[0]-reachAlong)/sampleSpacing)*sampleSpacing;
            for(double arc=first;arc<=std::min(geometry.length(),arcRange[1]+reachAlong);arc+=sampleSpacing){
                cv::Vec3d tangent=samplePolylineArc(geometry,std::min(geometry.length(),arc+tangentSpan)).point-samplePolylineArc(geometry,std::max(0.,arc-tangentSpan)).point;
                const double length=cv::norm(tangent);
                if(!(length>0.))continue;
                nodes.push_back({i,arc,samplePolylineArc(geometry,arc).point,tangent/length});
            }
        }
        // The prediction samples of every node.
        using Sample=vc::fiber_tracer::FiberPredictionNormalSample;
        using Scatter=vc::fiber_tracer::FiberPredictionNormalScatter;
        using Frame=vc::fiber_tracer::FiberPredictionNormalFrame;
        std::vector<std::vector<Sample>> samples(nodes.size());
        cv::parallel_for_(cv::Range(0,int(nodes.size())),[&](const cv::Range& range){
            for(int n=range.start;n<range.end;++n){
                const auto& node=nodes[size_t(n)];
                std::vector<cv::Vec3d> points(grid.size());std::vector<cv::Vec3f> pointsF(grid.size());
                for(size_t g=0;g<grid.size();++g){points[g]=node.point+grid[g];pointsF[g]=cv::Vec3f(points[g]);}
                std::vector<vc::lasagna::NormalSampleWithDerivative> sheet,direction;
                (void)normals.sampleNormalBatch(points,false,1,sheet);
                (void)directions.sampleNormalBatch(points,false,1,direction);
                std::vector<std::vector<float>> presenceValues;
                vc::lasagna::sampleLasagnaVectorBatch({*presence},pointsF,presenceValues,false);
                auto& list=samples[size_t(n)];list.reserve(grid.size());
                for(size_t g=0;g<grid.size();++g){
                    if(!sheet[g].sample.valid||!direction[g].sample.valid||presenceValues[g].empty())continue;
                    list.push_back({grid[g],direction[g].sample.normal,sheet[g].sample.normal,double(presenceValues[g][0])});
                }
            }
        },threads);
        const size_t record=8+6*variants.size();
        std::vector<double> rows(nodes.size()*record,0.);
        for(size_t n=0;n<nodes.size();++n){
            double* row=&rows[n*record];
            row[0]=double(nodes[n].fiber);row[1]=nodes[n].arc;
            for(int k=0;k<3;++k){row[2+k]=nodes[n].point[k];row[5+k]=nodes[n].tangent[k];}
        }
        for(size_t s=0;s<variants.size();++s){
            const auto& variant=variants[s];
            auto shape=base;shape.height=variant.height;shape.width=variant.width;shape.presenceBase=variant.presenceBase;
            // One pass: every node's scatter (in `frame`, about the previous estimates), summed along the fiber.
            std::vector<vc::fiber_tracer::FiberPredictionNormalEstimate> estimates(nodes.size());
            const auto pass=[&](Frame frame,double power){
                std::vector<Scatter> own(nodes.size());
                cv::parallel_for_(cv::Range(0,int(nodes.size())),[&](const cv::Range& range){
                    for(int n=range.start;n<range.end;++n){
                        const auto& node=nodes[size_t(n)];const auto& list=samples[size_t(n)];
                        auto& out=own[size_t(n)];
                        out=vc::fiber_tracer::fiberPredictionNormalScatter(node.tangent,list,shape,frame,estimates[size_t(n)].axis,power);
                        if(!variant.balance)continue;
                        // Both sides of the plane of this first estimate, each scaled to the weaker side's weight.
                        const cv::Vec3d plane=vc::fiber_tracer::fiberPredictionNormal(out).axis;
                        if(!(plane.dot(plane)>0.))continue;
                        std::array<std::vector<Sample>,2> sides;
                        for(const auto& sample:list)sides[sample.offset.dot(plane)>0.?1:0].push_back(sample);
                        std::array<Scatter,2> part;
                        for(int k=0;k<2;++k)part[k]=vc::fiber_tracer::fiberPredictionNormalScatter(node.tangent,sides[k],shape,frame,estimates[size_t(n)].axis,power);
                        const double weaker=std::min(part[0].weight,part[1].weight);
                        if(!(weaker>0.))continue;   // one side is empty: nothing to balance against
                        out=part[0].scaled(weaker/part[0].weight);out+=part[1].scaled(weaker/part[1].weight);
                    }
                },threads);
                std::vector<vc::fiber_tracer::FiberPredictionNormalEstimate> next(nodes.size());
                for(size_t n=0;n<nodes.size();++n){
                    // The window: `along`, or the smallest reach whose summed weight is min_support.
                    double window=variant.along;
                    if(variant.minSupport>0.&&variant.along>0.){
                        std::vector<std::pair<double,double>> near;
                        for(size_t m=0;m<nodes.size();++m)if(nodes[m].fiber==nodes[n].fiber&&std::abs(nodes[m].arc-nodes[n].arc)<=variant.along)
                            near.push_back({std::abs(nodes[m].arc-nodes[n].arc),own[m].weight});
                        std::sort(near.begin(),near.end());
                        double sum=0.;window=variant.along;
                        for(const auto& [distance,weight]:near){sum+=weight;if(sum>=variant.minSupport){window=distance;break;}}
                    }
                    const auto gather=[&](const cv::Vec3d* agree){
                        Scatter sum;
                        for(size_t m=0;m<nodes.size();++m){
                            if(nodes[m].fiber!=nodes[n].fiber)continue;
                            const double distance=std::abs(nodes[m].arc-nodes[n].arc);
                            if(distance>window)continue;
                            const double sigma=window/2.;
                            double factor=m==n||!(sigma>0.)?1.:std::exp(-0.5*distance*distance/(sigma*sigma));
                            Scatter part=variant.turn&&m!=n?vc::fiber_tracer::turned(own[m],vc::fiber_tracer::fiberDirectionTurn(nodes[m].tangent,nodes[n].tangent)):own[m];
                            if(agree){
                                const cv::Vec3d axis=vc::fiber_tracer::fiberPredictionNormal(part).axis;
                                factor*=std::pow(std::abs(axis.dot(*agree)),variant.slicePower);
                            }
                            sum+=part.scaled(factor);
                        }
                        return sum;
                    };
                    auto sum=gather(nullptr);
                    if(variant.slicePower>0.){
                        const cv::Vec3d first=vc::fiber_tracer::fiberPredictionNormal(sum).axis;
                        if(first.dot(first)>0.)sum=gather(&first);
                    }
                    next[n]=vc::fiber_tracer::fiberPredictionNormal(sum);
                }
                estimates=std::move(next);
            };
            if(variant.mode=="track"){
                // Per fiber: its nodes are consecutive and ascend in arc.
                for(size_t first=0;first<nodes.size();){
                    size_t last=first;while(last<nodes.size()&&nodes[last].fiber==nodes[first].fiber)++last;
                    std::vector<double> arcs;std::vector<cv::Vec3d> tangents;
                    for(size_t n=first;n<last;++n){arcs.push_back(nodes[n].arc);tangents.push_back(nodes[n].tangent);}
                    vc::fiber_tracer::FiberPredictionNormalTrackOptions trackOptions;
                    trackOptions.length=variant.along;trackOptions.agreementPower=variant.power;
                    size_t seed=0;
                    const auto tracked=vc::fiber_tracer::trackFiberPredictionNormals(arcs,tangents,
                        std::span<const std::vector<Sample>>(samples.data()+first,last-first),shape,trackOptions,&seed);
                    std::cout<<"prediction normals track fiber="<<nodes[first].fiber<<" seed_arc="<<(seed<arcs.size()?arcs[seed]:-1.)<<std::endl;
                    for(size_t n=first;n<last;++n)estimates[n]=tracked[n-first];
                    first=last;
                }
            }
            else if(variant.mode=="two_pass"){pass(Frame::Isotropic,0.);pass(Frame::Given,0.);}
            else{
                if(variant.balance)pass(Frame::Sample,0.);   // the first estimate whose plane splits the sides
                pass(Frame::Sample,0.);
                if(variant.mode=="condense")for(int k=0;k<variant.iterations;++k)pass(Frame::Sample,variant.power);
            }
            if(variant.strip>0.){
                // The estimates as sign-free targets of a strip solve over each fiber's sampled arc range.
                for(size_t first=0;first<nodes.size();){
                    size_t last=first;while(last<nodes.size()&&nodes[last].fiber==nodes[first].fiber)++last;
                    const auto geometry=makePolylineArcGeometry(lines[nodes[first].fiber].pointsBaseXYZ);
                    const double begin=nodes[first].arc,end=nodes[last-1].arc;
                    const auto piece=slicePolylineArc(geometry,begin,end);
                    vc::fiber_tracer::FiberStripNormalOptions stripOptions;
                    stripOptions.twistWeight=cfg.value("strip_twist_weight",300.);
                    stripOptions.smoothingSigma=cfg.value("prediction_normal_strip_sigma",64.);
                    stripOptions.resampleSpacing=cfg.value("prediction_normal_strip_spacing",64.);
                    stripOptions.axisWeight=variant.strip;
                    const auto chain=vc::fiber_tracer::fiberStripChain(piece,stripOptions);
                    std::vector<vc::fiber_tracer::FiberStripAxisTarget> stripTargets(chain.points.size());
                    for(size_t k=0;k<chain.points.size();++k){
                        size_t nearest=first;
                        for(size_t n=first;n<last;++n)if(std::abs(nodes[n].arc-begin-chain.arc[k])<std::abs(nodes[nearest].arc-begin-chain.arc[k]))nearest=n;
                        if(!(estimates[nearest].weight>0.))continue;
                        stripTargets[k].axis=estimates[nearest].axis;
                        stripTargets[k].weight=std::min(1.,estimates[nearest].weight/variant.stripSupport);
                    }
                    const auto solved=vc::fiber_tracer::solveFiberStripNormals(piece,stripOptions,&stripTargets);
                    if(solved.normals.size()!=piece.size())throw std::runtime_error("prediction normals: strip solve gave no normals");
                    for(size_t n=first;n<last;++n){
                        size_t vertex=0;
                        for(size_t j=0;j<chain.vertexArc.size();++j)if(std::abs(chain.vertexArc[j]-(nodes[n].arc-begin))<std::abs(chain.vertexArc[vertex]-(nodes[n].arc-begin)))vertex=j;
                        estimates[n].axis=solved.normals[vertex];
                    }
                    first=last;
                }
            }
            for(size_t n=0;n<nodes.size();++n){
                double* out=&rows[n*record]+8+6*s;
                for(int k=0;k<3;++k)out[k]=estimates[n].axis[k];
                out[3]=estimates[n].weight;out[4]=estimates[n].coherence;out[5]=estimates[n].count;
            }
        }
        const auto outputPath=std::filesystem::path(cfg.at("prediction_normals_output").get<std::string>());
        std::filesystem::create_directories(outputPath.parent_path());
        std::ofstream file(outputPath,std::ios::binary);
        if(!file)throw std::runtime_error("cannot write "+outputPath.string());
        file.write(reinterpret_cast<const char*>(rows.data()),std::streamsize(rows.size()*sizeof(double)));
        if(!file)throw std::runtime_error("write failed "+outputPath.string());
        Json result={{"record_doubles",record},{"ids",ids},{"variants",cfg.at("prediction_normal_variants")},{"samples",nodes.size()},
            {"grid_points",grid.size()},{"spacing",sampleSpacing},{"step",step},{"min_alignment",base.minAlignment},
            {"presence_floor",base.presenceFloor},{"presence_full",base.presenceFull},{"wall_seconds",seconds(started)}};
        vcr::zarr::atomicWrite(outputPath.string()+".json",result.dump(1)+"\n");
        std::cout<<"prediction normals samples="<<nodes.size()<<" grid_points="<<grid.size()<<std::endl;
        return 0;
    }
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
        {"fiber_source",{{"store",store.uuid(txn)},{"generation",sourceStatus.generation},
                         {"commit_sequence",sourceStatus.commitSequence}}},
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
            {"certify_winding",*options.certifyWinding},{"winding_distances",options.measureWindingDistances},{"normal_source",normalSource},
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
            {"fiber_source",result.at("fiber_source")},
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
        },presence?&*presence:nullptr);
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
