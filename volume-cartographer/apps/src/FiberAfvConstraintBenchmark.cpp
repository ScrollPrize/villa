#include "FiberAfvConstraintBenchmark.hpp"
#include "FiberConstraintDatabase.hpp"
#include "vc/fiber_tracer/FiberAxialAlignment.hpp"
#include "vc/core/types/FiberCollection.hpp"
#include "vc/core/util/AtomicFile.hpp"
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
#include <set>
#include <stdexcept>

int runFiberAfvConstraintBenchmark(int argc, char** argv)
{
    using namespace vc::fiber_tracer;
    using Clock = std::chrono::steady_clock;
    using Json = nlohmann::json;
    if (argc != 3) throw std::invalid_argument("afv-constraints expects one JSON config path");
    Json cfg; std::ifstream input(argv[2]); input >> cfg;
    const auto seconds = [](auto start) { return std::chrono::duration<double>(Clock::now()-start).count(); };
    const auto xyz = [](const Json& v) {
        if (!v.is_array() || v.size()!=3) throw std::invalid_argument("Expected XYZ triple");
        return cv::Vec3d{v.at(0).get<double>(),v.at(1).get<double>(),v.at(2).get<double>()};
    };
    const int threads=cfg.value("threads",32), repetitions=cfg.value("repetitions",3);
    if (threads<1 || repetitions<1) throw std::invalid_argument("Positive threads/repetitions required");
    if(cfg.contains("database") && repetitions!=1)throw std::invalid_argument("Database export requires one repetition");
    if(cfg.contains("database") && (std::filesystem::exists(cfg.at("database").get<std::string>()) ||
        std::filesystem::exists(cfg.at("database").get<std::string>()+".partial")))
        throw std::runtime_error("Database output already exists");
    const auto started=Clock::now();
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
        Json(*dataset.manifest().baseShapeZYX)!=root.at("coordinate_base_shape_zyx"))
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
    if(cfg.value("full_volume",false)) {
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
    const double querySeconds=seconds(queryStart);
    const auto loadStart=Clock::now();
    std::vector<FiberletCropTraceLine> lines;
    cv::Vec3d lo{INFINITY,INFINITY,INFINITY}, hi{-INFINITY,-INFINITY,-INFINITY};
    double totalArc=0.; std::size_t totalPoints=0;
    for(auto id:ids) {
        FiberletCropTraceLine line;
        int64_t first=0;
        while(true) {
            const auto blocks=afv.fiberBlocks(id,first,64);
            if(blocks.empty())break;
            for(const auto& block:blocks) {
                if(block.firstSegment!=first)throw std::runtime_error("Noncontiguous AFV blocks");
                for(std::size_t i=line.pointsBaseXYZ.empty()?0:1;i<block.points.size();++i) {
                    const auto& p=block.points[i];
                    line.pointsBaseXYZ.emplace_back(p[0],p[1],p[2]);
                    for(int d=0;d<3;++d){lo[d]=std::min(lo[d],p[d]);hi[d]=std::max(hi[d],p[d]);}
                }
                first+=static_cast<int64_t>(block.points.size())-1;
            }
        }
        if(line.pointsBaseXYZ.size()!=static_cast<std::size_t>(afv.summary(id).pointCount))
            throw std::runtime_error("AFV point-count mismatch");
        totalArc+=makePolylineArcGeometry(line.pointsBaseXYZ).length();
        totalPoints+=line.pointsBaseXYZ.size();
        lines.push_back(std::move(line));
    }
    const double loadSeconds=seconds(loadStart);
    const vc::lasagna::LasagnaNormalSampler normals(dataset);
    const double spacing=vc::lasagna::bindLasagnaChannel(dataset.manifest(),"nx").spacing;
    std::cout<<"AFV selected fibers="<<lines.size()<<" points="<<totalPoints<<" arc_base="<<totalArc
             <<" unsigned normals; no alignment"<<std::endl;
    FiberTraceConstraintConfig options;
    options.localEvents=true; options.localKeepWholeFibers=true; options.parallelThreads=threads;
    // Extraction records continuous evidence; rejection/quantization belongs to
    // downstream processing. Explicit flags retain historical benchmark modes.
    options.certifyWinding=cfg.value("certify_winding",false);
    options.enforceMaximumWindingDistance=cfg.value("enforce_maximum_winding",false);
    options.closestParallelEstimator=cfg.value("closest_parallel_estimator",false);
    options.localParallelReciprocal=cfg.value("parallel_reciprocal",false);
    options.localMovingProjectionWindows=cfg.value("moving_projection_windows",true);
    options.localPerpendicularProjectionIterations=cfg.value("perpendicular_iterations",16u);
    Json result={{"config",cfg},{"afv_uuid",Json::parse(afv.metadata("uuid"))},
        {"normal_manifest",dataset.manifest().raw},{"normal_provenance",provenance},
        {"selected_ids",ids},{"selected_fibers",lines.size()},{"points",totalPoints},
        {"arc_base",totalArc},{"query_seconds",querySeconds},{"load_seconds",loadSeconds},
        {"normal_alignment",false},
        {"geometry_minimum_xyz",{lo[0],lo[1],lo[2]}},
        {"geometry_maximum_xyz",{hi[0],hi[1],hi[2]}},
        {"settings",{{"local_events",true},{"keep_whole_fibers",true},
            {"certify_winding",*options.certifyWinding},
            {"enforce_maximum_winding",options.enforceMaximumWindingDistance},
            {"closest_parallel_estimator",*options.closestParallelEstimator},
            {"parallel_reciprocal",options.localParallelReciprocal},{"perpendicular_iterations",options.localPerpendicularProjectionIterations},
            {"moving_projection_windows",options.localMovingProjectionWindows},
            {"event_spacing",options.localSpacingBaseVoxels},{"tangent_span",options.localTangentSpanBaseVoxels},
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
            cfg.at("orient_database").get<std::string>(),cfg.at(umbilicus?"normal_prior_output":"database").get<std::string>(),
            std::vector<int64_t>(ids.begin(),ids.end()),lines,normals,threads,
            cfg.value("normal_sample_spacing",spacing),result,umbilicus?&*umbilicus:nullptr);
        result["wall_seconds"]=seconds(started);
        vc::core::util::ephemeralWriteString(output,result.dump(2)+"\n");
        std::cout<<"orientation complete "<<result["orientation"].dump()<<std::endl;
        return 0;
    }
    for(int i=0;i<repetitions;++i) {
        const auto begin=Clock::now();
        std::cout<<"extraction repetition="<<i<<std::endl;
        const auto report=extractFiberTraceConstraints(lines,options,
            [&](const auto& a,const auto& b,double step){return normals.normalAlignedWindingDistance(a,b,step);},
            [&](const auto& pairs,double step,int workers){return normals.normalAlignedWindingDistancesBatch(pairs,step,workers);},
            {},nullptr);
        if(report.pieces.size()!=lines.size() || report.hardConstraints || report.localMidpointCuts)
            throw std::runtime_error("Unexpected split/continuation or degenerate source");
        std::size_t parallel=0,signedCount=0;
        for(const auto& c:report.constraints){parallel+=c.parallelScore>.5;signedCount+=c.signedWindingDelta.has_value();}
        Json run={{"seconds",seconds(begin)},{"prepare_seconds",report.prepareSeconds},
            {"search_seconds",report.searchSeconds},{"orientation_seconds",report.orientationScoreSeconds},
            {"winding_seconds",report.windingScoreSeconds},{"resampled_points",report.resampledPoints},
            {"candidates",report.measuredCandidates},{"refined_candidates",report.localRefinedCandidates},
            {"events",report.constraints.size()},{"parallel",parallel},
            {"orthogonal",report.constraints.size()-parallel},{"signed",signedCount},
            {"rejected_consistency",report.localRejectedConsistency},{"rejected_winding",report.rejectedWinding},
            {"rejected_support",report.localRejectedSupport},{"unsigned_events",report.constraints.size()-signedCount},
            {"rejected_winding_cutoff",report.rejectedWindingCutoff}};
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
            verifyFiberConstraintDatabaseMeasurements(cfg.at("compare_database").get<std::string>(),report);
            result["all_measurements_match_database"]=true;
        }
        if(cfg.contains("database")) {
            result["export"]=writeFiberConstraintDatabase(cfg.at("database").get<std::string>(),
                std::vector<int64_t>(ids.begin(),ids.end()),report,normals,threads,result);
            result["wall_seconds"]=seconds(started);
            std::cout<<"export complete "<<result["export"].dump()<<std::endl;
        }
        vc::core::util::ephemeralWriteString(output,result.dump(2)+"\n");
        std::cout<<run.dump()<<std::endl;
    }
    return 0;
}
