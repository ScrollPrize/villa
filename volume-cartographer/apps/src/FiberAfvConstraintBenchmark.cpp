#include "FiberAfvConstraintBenchmark.hpp"
#include "FiberConstraintDatabase.hpp"
#include "vc/fiber_tracer/FiberAxialAlignment.hpp"
#include "vc/core/types/FiberCollection.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include "vc/core/util/OrderedBatchQueue.hpp"
#include "vc/core/util/ScrollUmbilicus.hpp"
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
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
        std::cout<<row.dump()<<std::endl;
        if(stream_)stream_<<row.dump()<<std::endl;
        phase.terminal=finished;phase.last=now;
    }
};
}

int runFiberAfvConstraintBenchmark(int argc, char** argv)
{
    using namespace vc::fiber_tracer;
    using Clock = std::chrono::steady_clock;
    using Json = nlohmann::json;
    if (argc != 2) throw std::invalid_argument("usage: vc_fiber_constraints CONFIG.json");
    Json cfg; std::ifstream input(argv[1]); input >> cfg;
    if(cfg.value("sqlite_serialize_memory",false))
        throw std::invalid_argument("Serialized-image export is not supported; use sqlite_in_memory with backup persistence");
    const auto seconds = [](auto start) { return std::chrono::duration<double>(Clock::now()-start).count(); };
    const auto xyz = [](const Json& v) {
        if (!v.is_array() || v.size()!=3) throw std::invalid_argument("Expected XYZ triple");
        return cv::Vec3d{v.at(0).get<double>(),v.at(1).get<double>(),v.at(2).get<double>()};
    };
    const int threads=cfg.value("threads",32), repetitions=cfg.value("repetitions",3);
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
    vc::fibers::FiberCollection afv(cfg.at("afv").get<std::string>());
    const auto root=Json::parse(afv.metadata("root"));
    const auto frame=Json::parse(afv.metadata("frame"));
    if (frame.at("vc_open_data_coordinate_space")!=cfg.at("coordinate_space") ||
        frame.value("vc_open_data_source_coordinate_level",0)!=0 ||
        frame.value("vc_open_data_source_coordinate_scale_factor",1.)!=1.)
        throw std::invalid_argument("AFV must use expected native coordinate frame");
    vc::lasagna::LasagnaDatasetOpenOptions open;
    open.workingToBaseScale=1.;
    open.remoteCacheRoot=cfg.at("remote_cache").get<std::string>();
    const auto dataset=vc::lasagna::LasagnaDataset::openLocation(cfg.at("normal_manifest"),open);
    if (!dataset.manifest().baseShapeZYX ||
        vc::lasagna::dyadicCoordinateScaleBetweenShapes(
            root.at("coordinate_base_shape_zyx").get<std::array<std::size_t,3>>(),
            *dataset.manifest().baseShapeZYX,0)!=1.0)
        throw std::invalid_argument("AFV and Lasagna base shapes differ");
    // This historical manifest has no coordinate identity. Require its sibling
    // inference provenance to name the expected scan rather than trusting shape.
    Json provenance; std::ifstream prov(cfg.at("normal_provenance").get<std::string>()); prov>>provenance;
    const std::string scan=cfg.at("coordinate_space");
    const auto slash=scan.find('/');
    if (slash==std::string::npos || provenance.at("source").at("sample_id")!=scan.substr(0,slash) ||
        provenance.at("source").at("volume_id")!=scan.substr(slash+1))
        throw std::invalid_argument("Normal provenance does not name expected scan");
    const auto manifestName=std::filesystem::path(cfg.at("normal_manifest").get<std::string>()).filename().string();
    bool artifactFound=false;
    for(const auto& artifact:provenance.at("artifacts"))
        artifactFound=artifactFound || (artifact.at("kind")=="manifest" && artifact.at("path")==manifestName);
    if(!artifactFound)throw std::invalid_argument("Normal manifest is not the recorded provenance artifact");
    const auto queryStart=Clock::now();
    std::set<int64_t> ids;
    progress({"catalog",0,0});
    if(cfg.contains("fiber_ids")) {
        for(const auto& id:cfg.at("fiber_ids"))ids.insert(id.get<int64_t>());
    } else if(cfg.value("full_volume",false)) {
        double before=std::numeric_limits<double>::infinity(); int64_t after=0;
        while(true) {
            const auto page=afv.catalogByLength(before,after,8192);
            if(page.empty())break;
            for(const auto& row:page)ids.insert(row.id);
            before=page.back().length;after=page.back().id;
        }
    } else {
    const auto lower=xyz(cfg.at("minimum_xyz")), upper=xyz(cfg.at("maximum_xyz"));
    vc::fibers::ViewRegion region;
    for(int d=0;d<3;++d) {
        if (!std::isfinite(lower[d]) || !std::isfinite(upper[d]) || lower[d]>=upper[d])
            throw std::invalid_argument("Invalid selection box");
        region.box.lower[d]=region.local.lower[d]=lower[d];
        region.box.upper[d]=region.local.upper[d]=upper[d];
    }
    const auto visible=afv.viewRegion(region,2,0,0,512ULL*1024*1024);
    if (!visible.complete) throw std::runtime_error("AFV spatial query exceeded memory budget; no truncated benchmark allowed");
    for(const auto& block:visible.blocks) ids.insert(block.fiberId);
    }
    if (ids.empty()) throw std::runtime_error("No fibers intersect selection box");
    progress({"catalog",ids.size(),ids.size()});
    const double querySeconds=seconds(queryStart);
    const auto loadStart=Clock::now();
    std::vector<FiberletCropTraceLine> lines(ids.size());
    const std::vector<int64_t> orderedIds(ids.begin(),ids.end());
    std::vector<double> lengths(ids.size());
    std::vector<std::pair<cv::Vec3d,cv::Vec3d>> bounds(ids.size());
    std::vector<std::exception_ptr> loadFailures(ids.size());
    std::atomic<std::size_t> loaded{0};
    cv::Vec3d lo{INFINITY,INFINITY,INFINITY}, hi{-INFINITY,-INFINITY,-INFINITY};
    double totalArc=0.; std::size_t totalPoints=0;
    progress({"load",0,ids.size()});
    // Separate read-only SQLite connections, not concurrent use of one
    // connection. Each worker consumes complete fibers from the same queue.
#ifdef _OPENMP
#pragma omp parallel num_threads(loadThreads)
#endif
    {
    std::unique_ptr<vc::fibers::FiberCollection> reader;
    std::exception_ptr openFailure;
    try {
        reader=std::make_unique<vc::fibers::FiberCollection>(cfg.at("afv").get<std::string>());
        reader->beginReadSnapshot();
    }
    catch(...) {openFailure=std::current_exception();}
#ifdef _OPENMP
#pragma omp for schedule(dynamic,16)
#endif
    for(std::ptrdiff_t index=0;index<static_cast<std::ptrdiff_t>(orderedIds.size());++index) {
        try {
        if(openFailure)std::rethrow_exception(openFailure);
        const auto id=orderedIds[index];
        auto& line=lines[index];
        int64_t first=0;
        while(true) {
            const auto blocks=reader->fiberBlocks(id,first,64);
            if(blocks.empty())break;
            for(const auto& block:blocks) {
                if(block.firstSegment!=first)throw std::runtime_error("Noncontiguous AFV blocks");
                for(std::size_t i=line.pointsBaseXYZ.empty()?0:1;i<block.points.size();++i) {
                    const auto& p=block.points[i];
                    line.pointsBaseXYZ.emplace_back(p[0],p[1],p[2]);
                }
                first+=static_cast<int64_t>(block.points.size())-1;
            }
        }
        if(line.pointsBaseXYZ.size()!=static_cast<std::size_t>(reader->summary(id).pointCount))
            throw std::runtime_error("AFV point-count mismatch");
        lengths[index]=makePolylineArcGeometry(line.pointsBaseXYZ).length();
        auto& [low,high]=bounds[index];
        low={INFINITY,INFINITY,INFINITY};high={-INFINITY,-INFINITY,-INFINITY};
        for(const auto& p:line.pointsBaseXYZ)
            for(int d=0;d<3;++d){low[d]=std::min(low[d],p[d]);high[d]=std::max(high[d],p[d]);}
        const auto done=loaded.fetch_add(1,std::memory_order_relaxed)+1;
        if(done%1024==0)progress({"load",done,ids.size()});
        } catch(...) {loadFailures[index]=std::current_exception();}
    }
    }
    for(const auto& failure:loadFailures)if(failure)std::rethrow_exception(failure);
    // Original sorted-ID accumulation order, independent of worker completion.
    for(size_t index=0;index<lines.size();++index) {
        const auto& line=lines[index];
        totalArc+=lengths[index];
        totalPoints+=line.pointsBaseXYZ.size();
        for(int d=0;d<3;++d){lo[d]=std::min(lo[d],bounds[index].first[d]);hi[d]=std::max(hi[d],bounds[index].second[d]);}
    }
    progress({"load",lines.size(),ids.size()});
    const double loadSeconds=seconds(loadStart);
    const vc::lasagna::LasagnaNormalSampler normals(dataset);
    const double spacing=vc::lasagna::bindLasagnaChannel(dataset.manifest(),"nx").spacing;
    std::cout<<"AFV selected fibers="<<lines.size()<<" points="<<totalPoints<<" arc_base="<<totalArc
             <<" unsigned normals; no alignment"<<std::endl;
    FiberTraceConstraintConfig options;
    options.localEvents=true; options.localKeepWholeFibers=true; options.parallelThreads=threads;
    options.crossQueryPrefix=cfg.value("cross_query_prefix",std::size_t{0});
    if(options.crossQueryPrefix>=lines.size() && options.crossQueryPrefix)
        throw std::invalid_argument("Cross query prefix must leave reference lines");
    // Extraction records continuous evidence; rejection/quantization belongs to
    // downstream processing. Explicit flags retain historical benchmark modes.
    options.certifyWinding=cfg.value("certify_winding",false);
    options.enforceMaximumWindingDistance=cfg.value("enforce_maximum_winding",false);
    options.closestParallelEstimator=cfg.value("closest_parallel_estimator",false);
    options.localParallelReciprocal=cfg.value("parallel_reciprocal",false);
    options.localMovingProjectionWindows=cfg.value("moving_projection_windows",true);
    options.localPerpendicularProjectionIterations=cfg.value("perpendicular_iterations",16u);
    options.candidateBatchSize=cfg.value("candidate_batch_size",std::size_t{262144});
    Json result={{"config",cfg},{"afv_uuid",Json::parse(afv.metadata("uuid"))},
        {"normal_manifest",dataset.manifest().raw},{"normal_provenance",provenance},
        {"selected_ids",ids},{"selected_fibers",lines.size()},{"points",totalPoints},
        {"fiber_base_shape_zyx",root.at("coordinate_base_shape_zyx")},
        {"normal_base_shape_zyx",*dataset.manifest().baseShapeZYX},
        {"arc_base",totalArc},{"query_seconds",querySeconds},{"load_seconds",loadSeconds},
        {"normal_alignment",false},
        {"geometry_minimum_xyz",{lo[0],lo[1],lo[2]}},
        {"geometry_maximum_xyz",{hi[0],hi[1],hi[2]}},
        {"settings",{{"local_events",true},{"keep_whole_fibers",true},
            {"load_threads",loadThreads},
            {"certify_winding",*options.certifyWinding},
            {"enforce_maximum_winding",options.enforceMaximumWindingDistance},
            {"closest_parallel_estimator",*options.closestParallelEstimator},
            {"parallel_reciprocal",options.localParallelReciprocal},{"perpendicular_iterations",options.localPerpendicularProjectionIterations},
            {"moving_projection_windows",options.localMovingProjectionWindows},
            {"event_spacing",options.localSpacingBaseVoxels},{"tangent_span",options.localTangentSpanBaseVoxels},
            {"candidate_batch_size",options.candidateBatchSize},
            {"parallel_span",options.localParallelSpanBaseVoxels},{"maximum_distance",options.maximumDistanceBaseVoxels},
            {"integration_step",options.windingIntegrationStepBaseVoxels},{"maximum_windings",options.maximumWindingDistance},
            {"reciprocal_tolerance",options.localReciprocalToleranceBaseVoxels},
            {"angular_tolerance",options.localAngularToleranceDegrees},{"distance_tolerance",options.localDistanceTolerance},
            {"refinement_rounds",options.localRefinementRounds},{"normalize_evidence",options.localNormalizeEvidence}}},
        {"normal_spacing_base",spacing},{"runs",Json::array()}};
    const auto output=std::filesystem::path(cfg.at("output").get<std::string>());
    std::filesystem::create_directories(output.parent_path());
    const auto loadAxis=[&]() {
        const auto& shape=root.at("coordinate_base_shape_zyx");
        auto loaded=vc::core::util::loadUmbilicusWithFrameCheck(cfg.at("umbilicus").get<std::string>(),
            {shape[2].get<double>(),shape[1].get<double>(),shape[0].get<double>()},
            {shape[0].get<int>(),shape[1].get<int>(),shape[2].get<int>()},
            vc::core::util::UmbilicusTargetGridAuthority::Authoritative,root.at("voxel_size_mm").get<double>()*1000.);
        if(!loaded.umbilicus || !loaded.warning.empty() || loaded.scaleDescription.empty())
            throw std::runtime_error("Umbilicus frame not verified: "+loaded.error+loaded.warning);
        result["umbilicus_frame"]=loaded.scaleDescription;
        return std::move(*loaded.umbilicus);
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
        Json features={{"format","fiber-axial-priors"},{"version",1},{"afv_uuid",result["afv_uuid"]},
            {"source_ids",ids},{"agreement",agreement},{"umbilicus",cfg.at("umbilicus")},
            {"afv_sha256",cfg.at("afv_sha256")},{"umbilicus_sha256",cfg.at("umbilicus_sha256")},
            {"sample_spacing_base",step},{"axis_secant_halfspan_z",8},
            {"semantics","arc mean 2*(fiber tangent dot axis tangent)^2-1; V positive, H negative"},
            {"seconds",seconds(begin)},{"umbilicus_frame",result["umbilicus_frame"]}};
        std::filesystem::create_directories(target.parent_path());
        vc::core::util::ephemeralWriteString(target,features.dump()+"\n");
        result["axial_seconds"]=seconds(begin);
        if(cfg.value("axial_prior_only",false)) {
            result["wall_seconds"]=seconds(started);
            result["resources"]=processResources();
            vc::core::util::ephemeralWriteString(output,result.dump(2)+"\n");
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
            std::vector<int64_t>(ids.begin(),ids.end()),lines,normals,threads,
            cfg.value("normal_sample_spacing",spacing),result,umbilicus?&*umbilicus:nullptr,
            cfg.contains("normal_curves_output"),cfg.contains("fiber_ids"));
        result["wall_seconds"]=seconds(started);
        result["resources"]=processResources();
        vc::core::util::ephemeralWriteString(output,result.dump(2)+"\n");
        std::cout<<"orientation complete "<<result["orientation"].dump()<<std::endl;
        return 0;
    }
    for(int i=0;i<repetitions;++i) {
        const auto begin=Clock::now();
        std::cout<<"extraction repetition="<<i<<std::endl;
        const auto extract=[&](const FiberTraceConstraintStream& stream) {
            return extractFiberTraceConstraints(lines,options,
                [&](const auto& a,const auto& b,double step){return normals.normalAlignedWindingDistance(a,b,step);},
                [&](const auto& pairs,double step,int workers){return normals.normalAlignedWindingDistancesBatch(pairs,step,workers);},
                {},nullptr,progress.enabled()?FiberTraceConstraintProgressCallback(
                    [&](const auto& p){progress(p);}):FiberTraceConstraintProgressCallback{},stream);
        };
        FiberConstraintDatabaseOptions databaseOptions;
        databaseOptions.inMemory=cfg.value("sqlite_in_memory",false);
        databaseOptions.pageSize=cfg.value("sqlite_page_size",0);
        databaseOptions.insertBatchRows=cfg.value("sqlite_insert_batch_rows",32);
        const bool streaming=cfg.contains("database") && cfg.value("stream_export",true);
        double extractionSeconds=0.;
        const auto report=[&]() {
            if(!streaming) {
                auto report=extract({});extractionSeconds=seconds(begin);return report;
            }
            using Batch=std::vector<FiberTraceConstraint>;
            vc::core::util::OrderedBatchQueue<Batch> queue(cfg.value("export_queue_batches",std::size_t{128}));
            std::promise<std::vector<FiberTraceConstraintPiece>> prepared;
            auto preparedFuture=prepared.get_future();
            auto extractionDone=begin;
            // Extraction owns its report; SQLite owns its connection. Only
            // immutable finalized batches cross the bounded queue.
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
                    stream.batch=[&](auto index,auto events) {
                        queue.push(index,Batch(events.begin(),events.end()));
                    };
                    stream.failed=[&](auto failure){queue.finish(failure);};
                    auto report=extract(stream);
                    if(report.pieces.size()!=lines.size() || report.hardConstraints || report.localMidpointCuts)
                        throw std::runtime_error("Unexpected split/continuation or degenerate source");
                    extractionDone=Clock::now();extractionSeconds=seconds(begin);
                    queue.finish();return report;
                } catch(...) {
                    const auto failure=std::current_exception();
                    queue.finish(failure);
                    if(!delivered)prepared.set_exception(failure);
                    throw;
                }
            });
            try {
                const auto pieces=preparedFuture.get();
                // Progress maintains independent timelines for concurrent phases.
                result["export"]=streamFiberConstraintDatabase(
                    cfg.at("database").get<std::string>(),std::vector<int64_t>(ids.begin(),ids.end()),pieces,
                    [&](std::stop_token stop) {
                        while(!stop.stop_requested()) {
                            auto batch=queue.next(stop);
                            if(!batch)return Batch{};
                            if(!batch->empty())return std::move(*batch);
                        }
                        return Batch{};
                    },normals,threads,result,
                    progress.enabled()?FiberTraceConstraintProgressCallback([&](const auto& p){progress(p);}):
                        FiberTraceConstraintProgressCallback{},databaseOptions);
                auto report=extraction.get();
                result["export"]["streamed"]=true;
                result["export"]["queue_peak_batches"]=queue.peakPending();
                result["export"]["tail_after_extraction_seconds"]=seconds(extractionDone);
                return report;
            } catch(...) {
                queue.finish(std::current_exception());
                // Unblock producers before waiting for the extraction future.
                if(extraction.valid())try{extraction.get();}catch(...){}
                throw;
            }
        }();
        if(report.pieces.size()!=lines.size() || report.hardConstraints || report.localMidpointCuts)
            throw std::runtime_error("Unexpected split/continuation or degenerate source");
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
            verifyFiberConstraintDatabaseMeasurements(cfg.at("compare_database").get<std::string>(),report);
            result["comparison_seconds"]=result.value("comparison_seconds",0.)+seconds(comparisonStarted);
            progress({"verify",report.constraints.size(),report.constraints.size(),report.constraints.size()});
            result["all_measurements_match_database"]=true;
        }
        if(cfg.contains("database") && !streaming) {
            result["export"]=writeFiberConstraintDatabase(cfg.at("database").get<std::string>(),
                std::vector<int64_t>(ids.begin(),ids.end()),report,normals,threads,result,
                progress.enabled()?FiberTraceConstraintProgressCallback([&](const auto& p){progress(p);}):
                    FiberTraceConstraintProgressCallback{},databaseOptions);
            result["wall_seconds"]=seconds(started);
            std::cout<<"export complete "<<result["export"].dump()<<std::endl;
        }
        result["wall_seconds"]=seconds(started);
        result["processing_seconds"]=result.at("wall_seconds").get<double>()-result.value("comparison_seconds",0.);
        result["resources"]=processResources();
        vc::core::util::ephemeralWriteString(output,result.dump(2)+"\n");
        std::cout<<run.dump()<<std::endl;
    }
    return 0;
}
