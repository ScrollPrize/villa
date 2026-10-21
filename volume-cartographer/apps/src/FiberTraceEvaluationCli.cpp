#include "FiberTraceEvaluationCli.hpp"
#include "FiberTraceCli.hpp"
#include "vc/fiber_tracer/FiberLiveBridge.hpp"
#include "vc/fiber_tracer/FiberCaseQueue.hpp"
#include "vc/fiber_tracer/FiberEvaluation.hpp"
#include "FiberTrialPruning.hpp"
#include "vc/fiber_tracer/FiberReplayVisualization.hpp"
#include "vc/fiber_tracer/PolylineNormalSampler.hpp"
#include "vc/fiber_tracer/FiberJson.hpp"
#include "vc/fiber_tracer/FiberReferenceReplayBenchmark.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include "vc/core/util/GlobalCacheRoot.hpp"
#include "vc/core/util/RemoteCacheSettings.hpp"
#include "vc/core/util/SurfaceTexture.hpp"
#include "vc/core/render/ChunkCache.hpp"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__unix__)
#include <unistd.h>
#endif
#include <limits>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <syncstream>
#include <thread>
#include <unordered_map>
#include <opencv2/imgcodecs.hpp>

namespace {
using namespace vc::fiber_tracer;
using Json = nlohmann::json;

// One budget for all native sources and sampling paths, configured before binding.
size_t configureEvaluationCache(const Json& settings)
{
    if (settings.contains("cache_bytes_per_channel"))
        throw std::invalid_argument("cache_bytes_per_channel was removed; use one decoded_cache_bytes budget");
    uint64_t bytes = 0;
    if (settings.contains("decoded_cache_bytes")) {
        const auto& value = settings.at("decoded_cache_bytes");
        if (!value.is_number_integer() || value.get<int64_t>() <= 0)
            throw std::invalid_argument("decoded_cache_bytes must be a positive integer");
        bytes = value.get<uint64_t>();
    } else {
#if defined(__APPLE__)
        uint64_t total = 0;
        size_t size = sizeof(total);
        if (sysctlbyname("hw.memsize", &total, &size, nullptr, 0) == 0)
            bytes = total / 10;
#elif defined(__unix__) && defined(_SC_PHYS_PAGES)
        const long pages = sysconf(_SC_PHYS_PAGES);
        const long pageSize = sysconf(_SC_PAGESIZE);
        if (pages > 0 && pageSize > 0 && uint64_t(pages) <= UINT64_MAX / uint64_t(pageSize))
            bytes = uint64_t(pages) * uint64_t(pageSize) / 10;
#endif
    }
    if (!bytes || bytes > std::numeric_limits<size_t>::max())
        throw std::invalid_argument("cannot determine decoded cache budget; set decoded_cache_bytes explicitly");
    vc::render::processChunkCacheService()->configureDecodedByteCapacity(size_t(bytes));
    std::cout << "Shared decoded cache budget: " << bytes << " bytes" << std::endl;
    return size_t(bytes);
}

Json readJson(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    return Json::parse(input);
}

bool completeInspectionGallery(const std::filesystem::path& directory)
{
    const auto metadataPath = directory / "render.json";
    if (!std::filesystem::is_regular_file(metadataPath)) return false;
    try {
        const auto metadata = readJson(metadataPath);
        if (metadata.value("format", std::string{}) != "vc_fiber_inspection" ||
            !metadata.contains("pages") || !metadata.at("pages").is_number_unsigned())
            return false;
        const auto pages = metadata.at("pages").get<size_t>();
        if (!pages) return false;
        for (size_t page = 0; page < pages; ++page)
            if (!std::filesystem::is_regular_file(directory / ("strips_" + std::to_string(page) + ".jpg")))
                return false;
        return true;
    } catch (...) {
        return false;
    }
}

Json expectedReplayArtifact(const std::vector<cv::Vec3d>& points,
                            const FiberReplayTraceResult& replay,
                            double threshold, const Json& provenance)
{
    Json reference = Json::array();
    for (const auto& point : points) reference.push_back({point[0], point[1], point[2]});
    return {{"format", "vc_native_replay_artifact"}, {"version", 1},
        {"reference_points_base_xyz", std::move(reference)},
        {"provenance", provenance}, {"replay", fiberReplayTraceJson(replay, threshold)}};
}

std::filesystem::path localPath(const std::filesystem::path& base, const std::string& value)
{
    std::filesystem::path path(value);
    return path.is_absolute() ? path : base / path;
}

std::string sourceLocation(const std::filesystem::path& base, const std::string& value)
{
    return value.find("://") != std::string::npos ? value
        : std::filesystem::absolute(localPath(base, value)).lexically_normal().string();
}

FiberTraceConfig traceOptions(const Json& values)
{
    FiberTraceConfig config;
    // Parallelize cases by default, not a full nested candidate pool per case.
    config.parallelThreads = 1;
    if (!values.is_array()) throw std::invalid_argument("trace_args must be an array of CLI argument strings");
    std::vector<std::string> strings = values.get<std::vector<std::string>>();
    std::vector<char*> args;
    for (auto& s : strings) args.push_back(s.data());
    for (int i = 0; i < static_cast<int>(args.size()); ++i)
        if (!cli::parseTraceOption(strings[i], i, static_cast<int>(args.size()), args.data(), config))
            throw std::invalid_argument("unknown trace argument " + strings[i]);
    cli::validateTraceOptions(config);
    if (config.lookaheadRetryParentCap != 0)
        throw std::invalid_argument("replay-set does not implement CP-segment retry; use --lookahead-retry-parent-cap 0");
    return config;
}

Json caseProfileJson(size_t index, double elapsed, const FiberReadinessProfile& ready, const FiberTraceProfile& trace)
{
    return {{"case", index}, {"elapsed_seconds", elapsed},
        {"readiness", {{"calls", ready.calls}, {"local_hits", ready.localHits},
            {"bridge_calls", ready.bridgeCalls}, {"submitted_dependencies", ready.submittedDependencies},
            {"cache_clears", ready.cacheClears}, {"local_lookup_seconds", ready.localLookupSeconds},
            {"bridge_and_permit_seconds", ready.bridgeSeconds}, {"bridge_lock_seconds", ready.bridgeLockSeconds},
            {"ready_wait_seconds", ready.readyWaitSeconds}}},
        {"trace", {{"generations", trace.generations}, {"candidates", trace.candidateTasks},
            {"candidate_score_seconds_inclusive", trace.candidateScoreSeconds},
            {"prediction_batch_seconds_inclusive", trace.predictionBatchSeconds},
            {"readiness_plan_seconds", trace.readinessPlanSeconds},
            {"readiness_plan_points", trace.readinessPlanPoints},
            {"readiness_plan_dependencies", trace.readinessPlanDependencies},
            {"corner_prepare_seconds", trace.predictionCornerPrepareSeconds},
            {"corner_layout_seconds", trace.predictionCornerLayoutSeconds},
            {"corner_pin_seconds", trace.predictionCornerPinSeconds},
            {"corner_gather_seconds", trace.predictionCornerGatherSeconds},
            {"corner_points", trace.cornerPointCount},
            {"normal_batch_seconds_inclusive", trace.normalBatchSeconds},
            {"frontier_seconds", trace.frontierSeconds}, {"prune_seconds", trace.pruneSeconds},
            {"lookahead_decision_seconds", trace.lookaheadDecisionSeconds}}}};
}

struct Source {
    std::string id;
    std::unique_ptr<FiberPredictionField> predictions;
    std::unique_ptr<vc::lasagna::LasagnaNormalSampler> normals;
    std::unique_ptr<vc::lasagna::LasagnaNormalSampler> tracingNormals;
    double traceScale = 1;
    double voxelSizeUm = 0;
    Json provenance;
    vc::lasagna::LasagnaDatasetManifest predictionManifest;
    std::shared_ptr<Volume> renderVolume;
    std::string renderCt;
    FiberReplayStripTextureSource renderSource;
};



struct Case {
    size_t source = 0;
    size_t fiber = 0;
    bool reverse = false;
    std::filesystem::path fiberPath;
    std::vector<cv::Vec3d> points;
    double length = 0;
    size_t run = 0;
    Json selection;
    std::vector<std::vector<cv::Vec3d>> contrastChoices;
    std::shared_ptr<std::vector<cv::Vec3d>> trialFull;
    std::shared_ptr<std::vector<double>> trialArc;
    double trialLo = 0, trialHi = 0;
    std::shared_ptr<vc::lasagna::NormalSampler> referenceNormals;
};

// Global failure-search trials are initially only compact interval
// descriptors.  Materialising every trial polyline while parsing the config
// made a large worklist effectively single-threaded before the evaluator's
// worker pool could start.  Keep the shared source geometry here and slice the
// individual intervals in a bounded native preparation pool below.
struct DeferredGlobalTrial {
    size_t source = 0;
    bool reverse = false;
    std::filesystem::path fiberPath;
    std::shared_ptr<std::vector<cv::Vec3d>> full;
    std::shared_ptr<std::vector<double>> arc;
    double lo = 0.0;
    double hi = 0.0;
    Json provenance;
};

std::vector<cv::Vec3d> fiberLinePoints(const Json& annotation,
                                       const std::filesystem::path& path)
{
    if (!annotation.contains("line_points") ||
        !annotation.at("line_points").is_array() ||
        annotation.at("line_points").size() < 2)
        throw std::invalid_argument("fiber has no usable line_points: " + path.string());
    std::vector<cv::Vec3d> points;
    points.reserve(annotation.at("line_points").size());
    for (const auto& p : annotation.at("line_points")) {
        if (!p.is_array() || p.size() != 3)
            throw std::invalid_argument("fiber line point is not xyz: " + path.string());
        cv::Vec3d point{p[0].get<double>(), p[1].get<double>(), p[2].get<double>()};
        if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
            !std::isfinite(point[2]))
            throw std::invalid_argument("fiber line point is nonfinite: " + path.string());
        points.push_back(point);
    }
    (void)makePolylineArcGeometry(points);
    return points;
}

// Return a polyline slice at arc coordinates measured from the first point.
// Trial extraction uses this instead of materializing one annotation per
// distance; all descriptors can therefore be submitted to one replay-set.
std::vector<cv::Vec3d> sliceFiberLine(const std::vector<cv::Vec3d>& input,
                                      double lo, double hi)
{
    if (!(std::isfinite(lo) && std::isfinite(hi) && lo >= 0.0 && hi > lo))
        throw std::invalid_argument("invalid trial arc interval");
    std::vector<double> arc(input.size(), 0.0);
    for (size_t i = 1; i < input.size(); ++i) arc[i] = arc[i-1] + cv::norm(input[i]-input[i-1]);
    if (input.size() < 2 || hi > arc.back() + 1e-6)
        throw std::invalid_argument("trial arc interval outside reference fiber");
    auto at = [&](double s) {
        const auto it = std::lower_bound(arc.begin(), arc.end(), s);
        if (it == arc.begin()) return input.front();
        if (it == arc.end()) return input.back();
        const size_t i = size_t(it - arc.begin());
        const double d = arc[i] - arc[i-1];
        return d > 0.0 ? input[i-1] + (input[i]-input[i-1]) * ((s-arc[i-1])/d) : input[i];
    };
    std::vector<cv::Vec3d> out{at(lo)};
    for (size_t i = 1; i + 1 < input.size(); ++i)
        if (arc[i] > lo && arc[i] < hi) out.push_back(input[i]);
    out.push_back(at(hi));
    return out;
}

std::vector<cv::Vec3d> sliceFiberLine(const std::vector<cv::Vec3d>& input,
                                      const std::vector<double>& arc,
                                      double lo, double hi)
{
    if (input.size() < 2 || arc.size() != input.size() ||
        !(std::isfinite(lo) && std::isfinite(hi) && lo >= 0.0 && hi > lo) ||
        hi > arc.back() + 1e-6)
        throw std::invalid_argument("invalid cached trial arc interval");
    auto at = [&](double s) {
        const auto it = std::lower_bound(arc.begin(), arc.end(), s);
        if (it == arc.begin()) return input.front();
        if (it == arc.end()) return input.back();
        const size_t i = size_t(it - arc.begin());
        const double d = arc[i] - arc[i-1];
        return d > 0.0 ? input[i-1] + (input[i]-input[i-1]) * ((s-arc[i-1])/d) : input[i];
    };
    const auto first = std::upper_bound(arc.begin(), arc.end(), lo);
    const auto last = std::lower_bound(arc.begin(), arc.end(), hi);
    const size_t firstIndex = size_t(first - arc.begin());
    const size_t lastIndex = size_t(last - arc.begin());
    std::vector<cv::Vec3d> out;
    out.reserve((lastIndex > firstIndex ? lastIndex - firstIndex : 0) + 2);
    out.push_back(at(lo));
    for (size_t i = firstIndex; i < lastIndex && i < input.size(); ++i)
        out.push_back(input[i]);
    out.push_back(at(hi));
    return out;
}

Json selfProximity(const Json& plan, const vc::lasagna::NormalSampler& normals,
                   double threshold, double factor, double separation, double spacing)
{
    struct Segment { cv::Vec3d a, b; double arc, length; };
    struct Hash {
        size_t operator()(const std::array<int64_t,3>& k) const {
            size_t h=0;
            for (auto x:k) h ^= std::hash<int64_t>{}(x)+0x9e3779b9+(h<<6)+(h>>2);
            return h;
        }
    };
    if (!(threshold>0 && factor>0 && separation>0 && spacing>0) ||
        !std::isfinite(threshold+factor+separation+spacing))
        throw std::invalid_argument("invalid self-proximity parameters");
    const auto started=std::chrono::steady_clock::now();
    const double radius=threshold*std::max(1.0,factor);
    auto cell=[&](const cv::Vec3d& p) {
        return std::array<int64_t,3>{int64_t(std::floor(p[0]/radius)),int64_t(std::floor(p[1]/radius)),int64_t(std::floor(p[2]/radius))};
    };
    std::vector<Segment> segments;
    std::unordered_map<std::array<int64_t,3>,std::vector<size_t>,Hash> grid;
    for (const auto& run:plan.at("runs")) {
        double arc=run.at("original_arcs_base")[0];
        const auto& points=run.at("points_base_xyz");
        for (size_t i=1;i<points.size();++i) {
            cv::Vec3d a(points[i-1][0],points[i-1][1],points[i-1][2]);
            cv::Vec3d b(points[i][0],points[i][1],points[i][2]);
            const double length=cv::norm(b-a);
            const size_t pieces=std::max(size_t{1},size_t(std::ceil(length/spacing)));
            for (size_t j=0;j<pieces;++j) {
                Segment s{a+(b-a)*(double(j)/pieces), a+(b-a)*(double(j+1)/pieces),arc+length*j/pieces,length/pieces};
                if (!(s.length>0)) continue;
                const auto lo=cell({std::min(s.a[0],s.b[0]),std::min(s.a[1],s.b[1]),std::min(s.a[2],s.b[2])});
                const auto hi=cell({std::max(s.a[0],s.b[0]),std::max(s.a[1],s.b[1]),std::max(s.a[2],s.b[2])});
                for(auto x=lo[0];x<=hi[0];++x) for(auto y=lo[1];y<=hi[1];++y) for(auto z=lo[2];z<=hi[2];++z)
                    grid[{x,y,z}].push_back(segments.size());
                segments.push_back(s);
            }
            arc+=length;
        }
    }
    size_t checks=0;
    Json result{{"excluded",false},{"minimum_arc_separation_base",separation},{"sample_spacing_base",spacing},
                {"normal_radius_base",threshold},{"tangential_radius_base",threshold*factor},{"indexed_segments",segments.size()}};
    for (const auto& query:segments) {
        // Include both endpoints; shared vertices may repeat but never all-pairs.
        for (int endpoint=0;endpoint<2;++endpoint) {
            const auto p=endpoint?query.b:query.a;
            const double arc=query.arc+(endpoint?query.length:0);
            const auto key=cell(p);
            for(int dx=-1;dx<=1;++dx) for(int dy=-1;dy<=1;++dy) for(int dz=-1;dz<=1;++dz) {
                const auto found=grid.find({key[0]+dx,key[1]+dy,key[2]+dz});
                if(found==grid.end()) continue;
                for(auto index:found->second) {
                    const auto& s=segments[index];
                    if (std::max(std::abs(arc-s.arc),std::abs(arc-s.arc-s.length))<separation) continue;
                    const auto v=s.b-s.a;
                    const double euclideanT=std::clamp((p-s.a).dot(v)/v.dot(v),0.0,1.0);
                    if(cv::norm(s.a+v*euclideanT-p)>radius) continue;
                    ++checks;
                    auto measure=[&](double t) {return measureFiberReplayThreshold(s.a+v*t,p,normals,1.0,threshold,factor);};
                    auto square=[&](double t) {auto q=measure(t).thresholdErrorBaseVoxels;return q*q;};
                    // Shared metric is quadratic along a segment with the normal
                    // sampled at fixed p. Minimize it without duplicating its math.
                    const double q0=square(0),q1=square(1),qm=square(.5);
                    const double aa=2*(q0+q1-2*qm),bb=q1-q0-aa;
                    const double optimum=aa>0?-bb/(2*aa):(q0<=q1?0:1);
                    double t=0, minimum=std::numeric_limits<double>::infinity();
                    const std::array<std::pair<double,double>,2> intervals{{
                        {0,std::min(1.0,(arc-separation-s.arc)/s.length)},
                        {std::max(0.0,(arc+separation-s.arc)/s.length),1}}};
                    for(const auto& [lo,hi]:intervals) if(lo<=hi) {
                        const double candidate=std::clamp(optimum,lo,hi);
                        const double q=square(candidate);
                        if(q<minimum) {minimum=q;t=candidate;}
                    }
                    if(!std::isfinite(minimum)) continue;
                    const auto m=measure(t);
                    if(!fiberReplayThresholdExceeded(m,threshold)) {
                        const auto target=s.a+v*t;
                        result.update({{"excluded",true},{"original_arcs_base",{arc,s.arc+s.length*t}},
                            {"points_base_xyz",{{p[0],p[1],p[2]},{target[0],target[1],target[2]}}},
                            {"threshold_ratio",m.thresholdErrorRatio},{"normal_valid",m.localNormalValid}});
                        goto done;
                    }
                }
            }
        }
    }
done:
    result["narrow_phase_checks"]=checks;
    result["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    return result;
}

Json referencePlan(const Json& config, const std::filesystem::path& base)
{
    Json result{{"policy", "cp-bounded-no-lasagna-v1"}, {"sources", Json::array()}};
    for (const auto& source : config.at("sources")) {
        std::unique_ptr<vc::lasagna::LasagnaNormalSampler> proximityNormals;
        if(config.contains("self_proximity")) {
            vc::lasagna::LasagnaDatasetOpenOptions options;
            options.remoteCacheRoot=vc::core::util::globalCacheRoot();
            options.workingToBaseScale=1.0;
            proximityNormals=std::make_unique<vc::lasagna::LasagnaNormalSampler>(
                vc::lasagna::LasagnaDataset::openLocation(sourceLocation(base,source.at("normal_manifest").get<std::string>()),options));
        }
        Json fibers = Json::array();
        for (const auto& value : source.at("fibers")) {
            // The Python coordinator resolves symlinks when binding sources.
            // Use the same existing-file identity for export and replay checks.
            const auto path = std::filesystem::canonical(localPath(base, value.get<std::string>()));
            auto plan = vc3dEvaluationReferencePlan(parseVc3dFiberJson(readJson(path), path.string()), path.string());
            if(proximityNormals) {
                const auto& settings=config.at("self_proximity");
                plan["self_proximity"]=selfProximity(plan,*proximityNormals,
                    config.value("error_threshold_base_voxels",40.0),config.value("tangential_threshold_factor",2.0),
                    settings.value("minimum_arc_separation_base",500.0),settings.value("sample_spacing_base",4.0));
            }
            plan["fiber_path"] = path.string(); fibers.push_back(std::move(plan));
        }
        result["sources"].push_back({{"source_id", source.at("id")}, {"fibers", std::move(fibers)}});
    }
    return result;
}

void writeRenderedOverview(const std::filesystem::path& output,
                           const FiberReplayOverview& overview, Json metadata)
{
    std::filesystem::create_directories(output.parent_path());
    if (!std::filesystem::create_directory(output)) throw std::invalid_argument("render output already exists");
    for (size_t i = 0; i < overview.pages.size(); ++i)
        if (!cv::imwrite((output / ("strips_" + std::to_string(i) + ".jpg")).string(), overview.pages[i].image))
            throw std::runtime_error("failed writing replay strip image");
    metadata["pages"] = overview.pages.size();
    metadata["rows"] = {"reference_top", "reference_side", "traced_top", "traced_side"};
    metadata["reference_color"] = "yellow";
    metadata["trace_and_failure_color"] = "cyan";
    metadata["texture_source"] = {{"locator", overview.textureSource.locator},
        {"shape_zyx", overview.textureSource.shapeZYX},
        {"scale_from_base_xyz", overview.textureSource.scaleFromBaseXYZ},
        {"offset_from_base_xyz", overview.textureSource.offsetFromBaseXYZ}};
    metadata["render_scale"] = overview.renderScale;
    metadata["component_gap_columns"] = overview.fiberletComponentGapColumns;
    metadata["components"] = Json::array();
    for (const auto& c : overview.fiberletComponents)
        metadata["components"].push_back({{"segment", c.sourceSegmentIndex},
            {"reference_arc_begin_base", c.referenceArcBeginBase},
            {"reference_arc_end_base", c.referenceArcEndBase},
            {"top_columns", c.topColumns}, {"side_columns", c.sideColumns},
            {"top_rows", c.topRows}, {"side_rows", c.sideRows}});
    metadata["page_layout"] = Json::array();
    for (const auto& page : overview.pages) {
        Json panels = Json::array();
        for (const auto& p : page.panels)
            panels.push_back({{"progress", {p.progressFractionBegin, p.progressFractionEnd}},
                {"reference_top_columns", p.referenceTopColumns}, {"reference_side_columns", p.referenceSideColumns},
                {"traced_top_columns", p.fiberletTopColumns}, {"traced_side_columns", p.fiberletSideColumns},
                {"reference_top_rows", p.referenceTopRows}, {"reference_side_rows", p.referenceSideRows},
                {"traced_top_rows", p.fiberletTopRows}, {"traced_side_rows", p.fiberletSideRows}});
        metadata["page_layout"].push_back({{"shape_yx", {page.image.rows, page.image.cols}}, {"panels", panels}});
    }
    vc::core::util::atomicWriteString(output / "render.json", metadata.dump(2) + "\n");
}

void renderArtifact(const std::filesystem::path& artifact, const std::filesystem::path& settingsPath)
{
    const auto document = readJson(artifact);
    if (document.at("format") != "vc_native_replay_artifact" || document.at("version") != 1)
        throw std::invalid_argument("unsupported replay artifact");
    const auto settings = readJson(settingsPath);
    configureEvaluationCache(settings);
    const auto replay = fiberReplayTraceFromJson(document.at("replay"));
    std::vector<cv::Vec3d> reference;
    for (const auto& point : document.at("reference_points_base_xyz")) {
        const auto p = point.get<std::array<double, 3>>();
        reference.emplace_back(p[0], p[1], p[2]);
    }
    vc::lasagna::LasagnaDatasetOpenOptions normalOptions;
    normalOptions.workingToBaseScale = settings.at("normal_working_to_base").get<double>();
    normalOptions.remoteCacheRoot = settings.contains("remote_cache_dir")
        ? std::filesystem::path(settings.at("remote_cache_dir").get<std::string>())
        : vc::core::util::globalCacheRoot();
    const auto normalLocation = sourceLocation(settingsPath.parent_path(), settings.at("normal_manifest").get<std::string>());
    if (normalLocation != document.at("provenance").at("normal_manifest").get<std::string>())
        throw std::invalid_argument("render normals differ from the evaluated source binding");
    if (normalOptions.workingToBaseScale != document.at("provenance").at("trace_to_base").get<double>())
        throw std::invalid_argument("render normal scale differs from evaluation");
    const auto dataset = vc::lasagna::LasagnaDataset::openLocation(normalLocation, normalOptions);
    const vc::lasagna::LasagnaNormalSampler normals(dataset);
    const auto ct = sourceLocation(settingsPath.parent_path(), settings.at("ct_volume").get<std::string>());
    if (!document.at("provenance").contains("ct_volume") ||
        ct != document.at("provenance").at("ct_volume").get<std::string>())
        throw std::invalid_argument("render CT differs from or is absent in the evaluated source binding");
    if (ct.find("://") != std::string::npos)
        throw std::invalid_argument("post-hoc strip rendering currently requires local OME-Zarr CT metadata");
    auto volume = Volume::New(ct);
    auto overview = renderFiberReplayTraceOverview(reference, replay, normals,
        normalOptions.workingToBaseScale, settings.value("threads", 1), *volume, ct);
    const auto output = localPath(settingsPath.parent_path(), settings.at("output").get<std::string>());
    writeRenderedOverview(output, overview, Json{
        {"artifact", std::filesystem::absolute(artifact).string()}, {"settings", settings}});
}
} // namespace

int runFiberTraceEvaluationCli(int argc, char** argv)
{
    if (argc == 5 && std::string_view(argv[1]) == "render-replay" && std::string_view(argv[3]) == "--config") {
        renderArtifact(argv[2], argv[4]);
        return 0;
    }
    const auto usage = "Usage: vc_fiber_trace_metric replay-set --config evaluation.json [--render] | render-replay trace.json --config render.json";
    if (argc < 2 || (std::string_view(argv[1]) != "replay-set" && std::string_view(argv[1]) != "replay-references")) throw std::invalid_argument(usage);
    std::filesystem::path configArgument;
    bool renderFlag = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--render" && !renderFlag) renderFlag = true;
        else if (std::string_view(argv[i]) == "--config" && configArgument.empty() && i + 1 < argc)
            configArgument = argv[++i];
        else throw std::invalid_argument(usage);
    }
    if (configArgument.empty()) throw std::invalid_argument(usage);
    const std::filesystem::path configPath = std::filesystem::absolute(configArgument);
    const auto config = readJson(configPath);
    if (config.at("version") != 1) throw std::invalid_argument("unsupported replay-set version");
    const auto references = referencePlan(config, configPath.parent_path());
    if (std::string_view(argv[1]) == "replay-references") {
        std::cout << references.dump() << std::endl;
        return 0;
    }
    if (config.contains("reference_plan") && config.at("reference_plan") != references)
        throw std::invalid_argument("reference selection changed since inference prefetch planning");
    const auto trace = traceOptions(config.value("trace_args", Json::array()));
    const auto backend = config.value("backend", std::string("beam"));
    if (backend != "beam")
        throw std::invalid_argument("This entrypoint supports direct beam replay");
    const bool live = config.contains("live_endpoint");
    const bool profileTracing = config.value("profile_tracing", false);
    const bool resumeVisualization = config.value("resume_visualization", false);
    const auto resumeRenderLimit = config.value("resume_render_limit", size_t{24});
    if (resumeVisualization && (!config.value("render", false) || renderFlag ||
            profileTracing || !resumeRenderLimit))
        throw std::invalid_argument("visualization resume requires configured beam/fiberlet rendering, a positive limit and no profiling/CLI render override");
    const auto readinessEntries = config.value("readiness_cache_entries", int64_t{4096});
    if (readinessEntries < 0) throw std::invalid_argument("readiness_cache_entries must be nonnegative");
    auto persistedConfig = config;
    persistedConfig.erase("reference_plan");
    persistedConfig.erase("resume_visualization");
    persistedConfig.erase("resume_render_limit");
    persistedConfig.erase("resume_render_cases");
    persistedConfig["decoded_cache_bytes"] = configureEvaluationCache(config);
    const bool render = config.value("render", false) || renderFlag;
    persistedConfig["render"] = render;
    if (live) persistedConfig["live_endpoint"].erase("token");
    if (live && trace.parallelThreads != 1)
        throw std::invalid_argument("live replay requires --threads 1; concurrency is across cases");
    const double threshold = config.value("error_threshold_base_voxels", 20.0);
    const double refine = config.value("match_refine_steps", 1.0);
    const double excursionGrace = config.value("excursion_grace_base_voxels", 0.0);
    const double tangentialFactor = config.value("tangential_threshold_factor", 4.0);
    if (!std::isfinite(tangentialFactor) || tangentialFactor <= 0)
        throw std::invalid_argument("invalid tangential_threshold_factor");
    if (!std::isfinite(excursionGrace) || excursionGrace < 0)
        throw std::invalid_argument("invalid excursion_grace_base_voxels");
    if (!std::isfinite(threshold) || threshold < 0 || !std::isfinite(refine) || refine < 0)
        throw std::invalid_argument("invalid reference threshold/refinement");
    const auto output = localPath(configPath.parent_path(), config.at("output").get<std::string>());
    const auto cacheRoot = config.contains("remote_cache_dir")
        ? std::filesystem::path(config.at("remote_cache_dir").get<std::string>())
        : vc::core::util::globalCacheRoot();
    std::vector<Source> sources;
    std::vector<Case> cases;
    std::set<std::string> ids;
    for (const auto& entry : config.at("sources")) {
        Source source;
        source.id = entry.at("id").get<std::string>();
        if (source.id.empty() || !ids.insert(source.id).second) throw std::invalid_argument("empty or duplicate source ID");
        source.voxelSizeUm = entry.at("base_voxel_size_um").get<double>();
        if (!std::isfinite(source.voxelSizeUm) || source.voxelSizeUm <= 0) throw std::invalid_argument("invalid source voxel size");
        vc::lasagna::LasagnaDatasetOpenOptions options;
        options.remoteCacheRoot = cacheRoot;
        const auto fiberLocation = sourceLocation(configPath.parent_path(), entry.at("fiber_manifest").get<std::string>());
        const auto referenceNormalMode = entry.value("reference_normals", std::string("lasagna"));
        if (referenceNormalMode != "lasagna" && referenceNormalMode != "baked")
            throw std::invalid_argument("invalid reference_normals");
        const auto normalLocation = referenceNormalMode == "baked" ? std::string{} :
            sourceLocation(configPath.parent_path(), entry.at("normal_manifest").get<std::string>());
        auto dataset = vc::lasagna::LasagnaDataset::openLocation(fiberLocation, options);
        const auto scales = resolveFiberPredictionTraceScales(dataset.manifest(), entry.value("inference_scaledown_power", 2));
        source.traceScale = scales.traceToBaseScale;
        auto manifest = dataset.manifest();
        manifest.workingToBaseScale = source.traceScale;
        source.predictionManifest = manifest;
        source.predictions = std::make_unique<FiberPredictionField>(vc::lasagna::LasagnaDataset(std::move(manifest)),
            FiberPredictionFieldOptions{.enableEmbeddings = trace.embeddingWeight > 0, .enableWidth = trace.widthRelativeTolerance > 0});
        options.workingToBaseScale = source.traceScale;
        if (referenceNormalMode == "lasagna") {
        const auto normals = vc::lasagna::LasagnaDataset::openLocation(normalLocation, options);
        if (!normals.manifest().fiberPredictionPrefixes().empty() ||
            !normals.manifest().groupForChannel("grad_mag"))
            throw std::invalid_argument("reference ellipsoid requires sheet normals, not fiber directions");
        if (!dataset.manifest().baseShapeZYX || !normals.manifest().baseShapeZYX)
            throw std::invalid_argument("replay-set requires explicit base shapes for prediction/normal compatibility");
        for (size_t axis = 0; axis < 3; ++axis) {
            const auto a = dataset.manifest().baseShapeZYX->at(axis);
            const auto b = normals.manifest().baseShapeZYX->at(axis);
            if ((a > b ? a - b : b - a) > 1)
                throw std::invalid_argument("prediction and normal base shapes disagree");
        }
        source.normals = std::make_unique<vc::lasagna::LasagnaNormalSampler>(normals);
        }
        const auto tracingNormalMode = entry.value("tracing_normals", std::string("lasagna"));
        if (tracingNormalMode != "lasagna" && tracingNormalMode != "fiber")
            throw std::invalid_argument("invalid tracing_normals");
        if (referenceNormalMode == "baked" && tracingNormalMode != "fiber")
            throw std::invalid_argument("baked reference normals require fiber tracing normals");
        if (tracingNormalMode == "fiber") {
            auto normalManifest = source.predictionManifest;
            normalManifest.groups.clear();
            for (const auto* channel : {"nx", "ny", "grad_mag"}) {
                const auto name = std::string("sheet_") + channel;
                const auto* group = source.predictionManifest.groupForChannel(name);
                if (!group) throw std::invalid_argument("missing predicted normal channel " + name);
                auto alias = *group;
                alias.name = channel;
                alias.channels = {channel};
                normalManifest.groups.push_back(std::move(alias));
            }
            normalManifest.raw["grad_mag_encode_scale"] = 1.0;
            normalManifest.raw["grad_mag_factor"] = 1.0;
            vc::lasagna::LasagnaDataset normalDataset(std::move(normalManifest));
            vc::lasagna::validateLasagnaNormalDatasetStructure(normalDataset);
            source.tracingNormals = std::make_unique<vc::lasagna::LasagnaNormalSampler>(normalDataset);
        }
        source.provenance = entry;
        source.provenance["fiber_manifest"] = fiberLocation;
        if (!normalLocation.empty()) source.provenance["normal_manifest"] = normalLocation;
        if (entry.contains("ct_volume")) source.provenance["ct_volume"] = sourceLocation(configPath.parent_path(), entry.at("ct_volume").get<std::string>());
        if (render) {
            // Fail before tracing/output creation for manifests that cannot
            // describe the presence raster lattice.
            (void)source.predictions->storedGridInfo();
            if (!entry.contains("ct_volume")) throw std::invalid_argument("--render requires ct_volume for every source");
            source.renderCt = source.provenance.at("ct_volume").get<std::string>();
            if (entry.value("ct_base_volume", false)) {
                // The input pyramid's level zero defines base coordinates;
                // no inferred power-of-two transform for selected groups.
                const bool remoteCt = source.renderCt.find("://") != std::string::npos;
                if (remoteCt && std::filesystem::weakly_canonical(options.remoteCacheRoot) !=
                    std::filesystem::weakly_canonical(vc::settings::remoteCachePath()))
                    throw std::invalid_argument("render CT cache root must match the process-wide VC3D cache root");
                source.renderVolume = remoteCt
                    ? Volume::NewFromUrl(source.renderCt)
                    : Volume::New(source.renderCt);
                if (source.renderVolume->dtype() != vc::render::ChunkDtype::UInt8 ||
                    !source.renderVolume->hasScaleLevel(0))
                    throw std::invalid_argument("render CT base volume requires uint8 level zero");
                source.renderSource = {source.renderCt, source.renderVolume->shape(0), {1, 1, 1}, {0, 0, 0}};
                for (size_t axis = 0; axis < 3; ++axis)
                    if (source.renderSource.shapeZYX[axis] != dataset.manifest().baseShapeZYX->at(axis))
                        throw std::invalid_argument("render CT base shape differs from predictions");
            } else {
                if (source.renderCt.find("://") != std::string::npos)
                    throw std::invalid_argument("remote render CT must bind the base pyramid with ct_base_volume=true");
                source.renderVolume = Volume::New(source.renderCt);
                source.renderSource = validateFiberReplayStripCtVolume(*source.renderVolume, source.renderCt);
            }
        }
        source.provenance["trace_to_base"] = source.traceScale;
        source.provenance["effective_trace"] = cli::traceConfigJson(trace);
        auto selection = references.at("sources").at(sources.size()).at("fibers");
        for (auto& f : selection) for (auto& r : f["runs"]) r.erase("points_base_xyz");
        source.provenance["reference_selection"] = selection;
        source.provenance["reference_selection_policy"] = references.at("policy");
        size_t fiberIndex = 0;
        for (const auto& path : entry.at("fibers")) {
            const auto fiberPath = localPath(configPath.parent_path(), path.get<std::string>());
            const auto annotation = readJson(fiberPath);
            std::shared_ptr<vc::lasagna::NormalSampler> bakedNormals;
            if (referenceNormalMode == "baked") {
                const auto points = fiberLinePoints(annotation, fiberPath);
                const auto& baked = annotation.at("baked_annotations");
                if (baked.at("version").get<int>() != 1)
                    throw std::invalid_argument("unsupported baked reference normals version");
                std::vector<cv::Vec3d> axes;
                for (const auto& axis : baked.at("normals_xyz"))
                    axes.emplace_back(axis.at(0).get<double>(), axis.at(1).get<double>(), axis.at(2).get<double>());
                if (baked.contains("valid")) {
                    if (baked.at("valid").size() != points.size())
                        throw std::invalid_argument("baked reference validity length mismatch");
                    for (const auto& valid : baked.at("valid"))
                        if (!valid.get<bool>()) throw std::invalid_argument("invalid baked reference normal");
                }
                bakedNormals = std::make_shared<PolylineNormalSampler>(points, std::move(axes), source.traceScale);
            }
            if (annotation.contains("vc_open_data_coordinate_space")) {
                if (!entry.contains("coordinate_space") ||
                    annotation.at("vc_open_data_coordinate_space") != entry.at("coordinate_space"))
                    throw std::invalid_argument("annotation coordinate-space binding mismatch: " + fiberPath.string());
            }
            if (annotation.contains("vc_open_data_source_original_resolution")) {
                const double annotatedVoxelSize = annotation.at("vc_open_data_source_original_resolution").get<double>();
                if (!std::isfinite(annotatedVoxelSize) || std::abs(annotatedVoxelSize - source.voxelSizeUm) > 1e-6)
                    throw std::invalid_argument("annotation base voxel size mismatch: " + fiberPath.string());
            }
            const auto& selected = references.at("sources").at(sources.size()).at("fibers").at(fiberIndex);
            const std::vector<std::vector<cv::Vec3d>> contrastChoices;
            const auto evaluationDirection = annotation.value("evaluation_direction", std::string("both"));
            if (evaluationDirection != "both" && evaluationDirection != "forward" && evaluationDirection != "reverse")
                throw std::invalid_argument("invalid annotation evaluation_direction");
            for (const auto& run : selected.at("runs")) {
                for (bool reverse : {false, true}) {
                    if ((evaluationDirection == "forward" && reverse) || (evaluationDirection == "reverse" && !reverse)) continue;
                    std::vector<cv::Vec3d> points;
                    for (const auto& p : run.at("points_base_xyz")) points.emplace_back(p[0].get<double>(), p[1].get<double>(), p[2].get<double>());
                    if (reverse) std::reverse(points.begin(), points.end());
                    auto provenance = run; provenance.erase("points_base_xyz");
                    provenance["original_length_base_voxels"] = selected.at("original_length_base_voxels");
                    cases.push_back({sources.size(), fiberIndex, reverse, fiberPath, std::move(points),
                        run.at("length_base_voxels").get<double>(), run.at("run_index").get<size_t>(),
                        std::move(provenance), contrastChoices});
                    cases.back().referenceNormals = bakedNormals;
                }
            }
            ++fiberIndex;
        }
        if (!fiberIndex) throw std::invalid_argument("source has no selected fibers: " + source.id);
        sources.push_back(std::move(source));
    }
    const int requestedWorkers = config.value("workers", 0);
    if (requestedWorkers < 0) throw std::invalid_argument("workers must be nonnegative");
    std::vector<DeferredGlobalTrial> deferredTrials;

    // Optional global trial worklist.  The Python extractor can submit every
    // distance for every candidate in one replay invocation.  Native workers
    // then execute the fixed list continuously; ordering/cancellation is
    // decided by the coordinator from the emitted audit fields.
    if (config.contains("failure_search_trials")) {
        const auto& trials = config.at("failure_search_trials");
        if (!trials.is_array()) throw std::invalid_argument("failure_search_trials must be an array");
        std::unordered_map<std::string, size_t> sourceIndex;
        std::unordered_map<std::string, std::shared_ptr<std::vector<cv::Vec3d>>> trialFiberCache;
        std::unordered_map<std::string, std::shared_ptr<std::vector<double>>> trialArcCache;
        for (size_t i = 0; i < sources.size(); ++i) sourceIndex.emplace(sources[i].id, i);
        if (config.value("failure_search_trials_only", false)) cases.clear();
        for (const auto& trial : trials) {
            for (const auto& key : {"candidate_id", "direction", "distance", "fiber_path", "source_id"})
                if (!trial.contains(key)) throw std::invalid_argument(std::string("trial missing ") + key);
            const auto sourceIt = sourceIndex.find(trial.at("source_id").get<std::string>());
            if (sourceIt == sourceIndex.end()) throw std::invalid_argument("trial source_id is unknown");
            const bool reverse = trial.at("direction").get<std::string>() == "reverse";
            if (!reverse && trial.at("direction").get<std::string>() != "forward")
                throw std::invalid_argument("trial direction must be forward or reverse");
            const std::filesystem::path path = sourceLocation(configPath.parent_path(), trial.at("fiber_path").get<std::string>());
            const auto cacheKey = path.string();
            auto cacheIt = trialFiberCache.find(cacheKey);
            if (cacheIt == trialFiberCache.end()) {
                auto full = std::make_shared<std::vector<cv::Vec3d>>(fiberLinePoints(readJson(path), path));
                auto arc = std::make_shared<std::vector<double>>(full->size(), 0.0);
                for (size_t i = 1; i < full->size(); ++i)
                    (*arc)[i] = (*arc)[i-1] + cv::norm((*full)[i]-(*full)[i-1]);
                cacheIt = trialFiberCache.emplace(cacheKey, std::move(full)).first;
                trialArcCache.emplace(cacheKey, std::move(arc));
            }
            const double distance = trial.at("distance").get<double>();
            if (trial.value("coordinate_contract", "") != "original-fiber-arclength-v2")
                throw std::invalid_argument("failure trial requires original-fiber-arclength-v2 coordinates");
            const double lo = trial.at("original_fiber_arc_start_base").get<double>();
            const double hi = trial.at("original_fiber_arc_end_base").get<double>();
            for (const auto* key : {"expected_start_base_xyz", "expected_end_base_xyz"}) {
                const auto& xyz = trial.at(key);
                if (!xyz.is_array() || xyz.size() != 3)
                    throw std::invalid_argument("failure trial requires expected endpoint XYZ");
                for (const auto& v : xyz)
                    if (!std::isfinite(v.get<double>()))
                        throw std::invalid_argument("failure trial expected endpoint must be finite");
            }
            if (!(hi > lo)) throw std::invalid_argument("trial requires lo < hi");
            Json provenance = trial;
            provenance["global_trial"] = true;
            provenance["candidate_id"] = trial.at("candidate_id");
            provenance["target_distance_base"] = distance;
            provenance["fiber_path"] = path;
            deferredTrials.push_back({sourceIt->second, reverse, path,
                cacheIt->second, trialArcCache.at(cacheKey), lo, hi,
                std::move(provenance)});
        }
    }

    // Keep trial polylines shared until a bounded execution context admits them.
    for (auto& trial : deferredTrials) {
        cases.push_back({trial.source, cases.size(), trial.reverse, trial.fiberPath,
            {}, trial.hi-trial.lo, 0, std::move(trial.provenance), {},
            trial.full, trial.arc, trial.lo, trial.hi});
    }
    const bool trialSearch = config.value("failure_search_trials_only", false);
    const size_t workers = std::min(cases.size(), requestedWorkers ? size_t(requestedWorkers) : size_t(std::max(1u, std::thread::hardware_concurrency())));
    const bool queuedCases = live || trialSearch;
    if (live && !queuedCases && cases.size() > config.value("max_live_cases", size_t{256}))
        throw std::invalid_argument("live case count exceeds explicit stack-preserving thread budget");
    std::unique_ptr<FiberLiveBridge> bridge;
    if (live) bridge = std::make_unique<FiberLiveBridge>(config.at("live_endpoint"));
    std::vector<std::atomic<double>> caseRemaining(cases.size());
    for (size_t i = 0; i < cases.size(); ++i) caseRemaining[i].store(cases[i].length);
    const auto cpuBudget = workers;
    std::counting_semaphore<> cpuPermits(static_cast<std::ptrdiff_t>(cpuBudget));
    std::unique_ptr<vc::core::util::ExclusiveDirectoryLock> outputLock;
    if (resumeVisualization) {
        if (!std::filesystem::is_directory(output) || !std::filesystem::is_directory(output / "cases"))
            throw std::invalid_argument("visualization resume requires an existing evaluation output");
        outputLock = std::make_unique<vc::core::util::ExclusiveDirectoryLock>(output);
    } else {
        std::filesystem::create_directories(output.parent_path());
        if (!std::filesystem::create_directory(output)) throw std::invalid_argument("evaluation output already exists");
        std::filesystem::create_directory(output / "cases");
    }
    std::vector<FiberReferenceReplayOutcome> outcomes(cases.size());
    std::vector<std::optional<Json>> infrastructureFailures(cases.size());
    std::vector<std::optional<FiberReplayTraceResult>> savedReplays(cases.size());
    std::vector<std::optional<Json>> savedReplayProvenance(cases.size());
    std::set<size_t> resumeRenderCases;
    if (resumeVisualization) {
        // Validate the complete deterministic case manifest before rendering
        // anything. The saved artifact remains the scientific trace authority;
        // resume only reconstructs its in-memory replay for the inline renderer.
        std::vector<std::vector<size_t>> missingBySource(sources.size());
        for (size_t index = 0; index < cases.size(); ++index) {
            const auto& item = cases[index];
            const auto& source = sources[item.source];
            auto provenance = source.provenance;
            provenance["source_id"] = source.id;
            provenance["fiber_path"] = item.fiberPath.string();
            provenance["reverse"] = item.reverse;
            provenance["case_index"] = index;
            provenance["reference_run"] = item.selection;
            const auto artifactPath = output / "cases" / std::to_string(index) / "trace.json";
            if (!std::filesystem::is_regular_file(artifactPath))
                throw std::invalid_argument("visualization resume is missing trace artifact for case " + std::to_string(index));
            const auto artifact = readJson(artifactPath);
            auto replay = fiberReplayTraceFromJson(artifact.at("replay"));
            const auto& savedProvenance = artifact.at("provenance");
            // A render-only resume may add CT bindings and deliberately skips
            // rebuilding derived fiberlet-cache metadata.  Every scientific
            // binding present in the current evaluation must still match; the
            // saved superset remains the authority for the canonical artifact.
            for (auto it = provenance.begin(); it != provenance.end(); ++it) {
                if (it.key() == "ct_volume" || it.key() == "ct_base_volume")
                    continue;
                if (!savedProvenance.contains(it.key()) ||
                    savedProvenance.at(it.key()) != it.value())
                    throw std::invalid_argument("visualization resume trace/provenance mismatch for case " + std::to_string(index));
            }
            if (artifact != expectedReplayArtifact(item.points, replay, threshold, savedProvenance))
                throw std::invalid_argument("visualization resume trace/provenance mismatch for case " + std::to_string(index));
            savedReplays[index] = std::move(replay);
            savedReplayProvenance[index] = savedProvenance;
            if (!completeInspectionGallery(artifactPath.parent_path() / "rendered"))
                missingBySource.at(item.source).push_back(index);
        }
        if (config.contains("resume_render_cases")) {
            const auto& selected = config.at("resume_render_cases");
            if (!selected.is_array() || selected.size() > resumeRenderLimit)
                throw std::invalid_argument("invalid explicit visualization-resume case selection");
            std::set<size_t> missing;
            for (const auto& sourceCases : missingBySource)
                missing.insert(sourceCases.begin(), sourceCases.end());
            for (const auto& value : selected) {
                if (!value.is_number_unsigned())
                    throw std::invalid_argument("visualization-resume case index must be unsigned");
                const auto index = value.get<size_t>();
                if (!missing.contains(index) || !resumeRenderCases.insert(index).second)
                    throw std::invalid_argument("visualization-resume case selection is stale or duplicated");
            }
        } else {
            // Select across sources deterministically. The shared chunk caches
            // support concurrent readers and every selected direction may
            // issue demand independently.
            for (size_t rank = 0; resumeRenderCases.size() < resumeRenderLimit; ++rank) {
                bool added = false;
                for (const auto& sourceCases : missingBySource) if (rank < sourceCases.size()) {
                    resumeRenderCases.insert(sourceCases[rank]);
                    added = true;
                    if (resumeRenderCases.size() == resumeRenderLimit) break;
                }
                if (!added) break;
            }
        }
        std::cout << "visualization-resume selected=" << resumeRenderCases.size()
                  << " limit=" << resumeRenderLimit << " cases=" << cases.size() << std::endl;
    }
    std::vector<size_t> order(cases.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (trialSearch) return cases[a].selection.at("target_distance_base").get<double>() <
                                cases[b].selection.at("target_distance_base").get<double>();
        return cases[a].length > cases[b].length;
    });
    FiberTrialPruning trialPruning;
    std::atomic<size_t> skippedTrials{0};
    std::atomic<size_t> next{0}, completed{0};
    std::atomic<bool> failed{false};
    std::mutex errorMutex;
    std::exception_ptr error;
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::jthread> pool;
    const auto caseWork = [&] {
        try {
            while (!failed.load()) {
                const size_t rank = next.fetch_add(1);
                if (rank >= order.size()) break;
                const size_t index = order[rank];
                auto& item = cases[index];
                const std::string trialKey = trialSearch ? item.selection.at("candidate_id").get<std::string>() +
                    (item.reverse ? ":reverse" : ":forward") : "";
                const double trialDistance = trialSearch ? item.selection.at("target_distance_base").get<double>() : 0;
                if (trialSearch && trialPruning.skip(trialKey, trialDistance)) {
                    ++skippedTrials;
                    ++completed;
                    caseRemaining[index].store(0);
                    return; // One trial per reusable context invocation.
                }
                if (item.trialFull) {
                    item.points = sliceFiberLine(*item.trialFull, *item.trialArc, item.trialLo, item.trialHi);
                    if (item.reverse) std::reverse(item.points.begin(), item.points.end());
                    const auto& start = item.selection.at("expected_start_base_xyz");
                    const auto& end = item.selection.at("expected_end_base_xyz");
                    const cv::Vec3d expectedStart(start[0].get<double>(), start[1].get<double>(), start[2].get<double>());
                    const cv::Vec3d expectedEnd(end[0].get<double>(), end[1].get<double>(), end[2].get<double>());
                    constexpr double endpointToleranceBase = 0.25;
                    if (cv::norm(item.points.front()-expectedStart) > endpointToleranceBase ||
                        cv::norm(item.points.back()-expectedEnd) > endpointToleranceBase)
                        throw std::invalid_argument("failure trial endpoint mismatch for candidate " +
                            item.selection.at("candidate_id").get<std::string>());
                }
                const auto& source = sources[item.source];
                const auto* referenceNormals = item.referenceNormals ? item.referenceNormals.get() : source.normals.get();
                if (!referenceNormals) throw std::invalid_argument("missing reference normals for evaluation case");
                FiberCaseCpuPermit permit(cpuPermits);
                if (failed.load()) break;

                std::atomic<double> remaining{item.length};
                FiberReadinessProfile readinessProfile;
                FiberTraceProfile traceProfile;
                FiberCaseReadiness readiness(static_cast<size_t>(readinessEntries), profileTracing ? &readinessProfile : nullptr);
                const auto caseStart = std::chrono::steady_clock::now();
                auto nextProfile = caseStart + std::chrono::seconds(10);
                std::unique_ptr<FiberPredictionField> livePredictions;
                const bool renderThisCase = render && (!resumeVisualization || resumeRenderCases.contains(index));
                if (bridge && (!resumeVisualization || renderThisCase)) {
                    livePredictions = std::make_unique<FiberPredictionField>(
                        vc::lasagna::LasagnaDataset(source.predictionManifest),
                        FiberPredictionFieldOptions{.enableEmbeddings = trace.embeddingWeight > 0, .enableWidth = trace.widthRelativeTolerance > 0});
                    livePredictions->setReadinessGate([&](const auto& dependencies) {
                        readiness.await(dependencies, [&](const auto& missing) {
                            permit.suspend([&] { bridge->await(source.id, index, remaining.load(), missing,
                                profileTracing ? &readinessProfile : nullptr); });
                        }, [&] { bridge->checkActive(); });
                    });
                }
                auto effectiveTrace = trace;
                effectiveTrace.profile = profileTracing ? &traceProfile : nullptr;
                effectiveTrace.traceToBaseScale = source.traceScale;
                effectiveTrace.baseVoxelSizeUm = source.voxelSizeUm;
                FiberEvaluationCase request{item.points};
                FiberChunkReplayConfig replayConfig{threshold, excursionGrace, tangentialFactor, refine};
                const auto replay = [&]() -> FiberReplayTraceResult {
                    if (resumeVisualization) return *savedReplays.at(index);
                    return replayFiberTrainingCase(livePredictions ? *livePredictions : *source.predictions,
                    effectiveTrace, source.traceScale,
                    source.tracingNormals ? *source.tracingNormals : *source.normals,
                    *referenceNormals, source.traceScale, request, replayConfig,
                    [&](const FiberTraceProgress& progress) {
                        if (progress.referenceArcFraction)
                            remaining.store(item.length * (1 - std::clamp(*progress.referenceArcFraction, 0.0, 1.0)));
                        if (profileTracing && std::chrono::steady_clock::now() >= nextProfile) {
                            const auto now = std::chrono::steady_clock::now();
                            nextProfile = now + std::chrono::seconds(10);
                            auto profile = caseProfileJson(index, std::chrono::duration<double>(now - caseStart).count(), readinessProfile, traceProfile);
                            profile["remaining_base_voxels"] = remaining.load();
                            std::osyncstream(std::cout) << "trace-profile " << profile.dump() << std::endl;
                        }
                    });
                }();
                auto provenance = source.provenance;
                provenance["source_id"] = source.id;
                provenance["fiber_path"] = item.fiberPath.string();
                provenance["reverse"] = item.reverse;
                provenance["case_index"] = index;
                provenance["reference_run"] = item.selection;
                if (resumeVisualization) {
                    provenance = *savedReplayProvenance.at(index);
                    // Texture provenance is specific to this rendering pass;
                    // it is not part of the immutable saved trace binding.
                    for (const char* key : {"ct_volume", "ct_base_volume"})
                        if (source.provenance.contains(key))
                            provenance[key] = source.provenance.at(key);
                }

                if (!resumeVisualization && !trialSearch)
                    writeFiberReplayTraceArtifacts(output / "cases" / std::to_string(index), item.points, replay, threshold, provenance);
                if (trialSearch) {
                    bool valid = !replay.failures.empty() && !replay.segments.empty() &&
                        replay.failures.front().segmentIndex == 0 &&
                        std::abs(replay.failures.front().referenceArcBase - trialDistance) <= 100;
                    std::optional<double> outside;
                    if (valid) {
                        const auto& segment = replay.segments.front();
                        for (const auto& match : segment.matches) {
                            if (match.thresholdMeasurement.thresholdErrorRatio > 1) {
                                if (!outside) outside = 0.;
                                else if (match.tracePointIndex > 0)
                                    *outside += cv::norm(segment.tracePointsBase.at(match.tracePointIndex) -
                                                       segment.tracePointsBase.at(match.tracePointIndex - 1));
                            } else outside.reset();
                        }
                    }
                    valid = valid && outside && *outside + 1e-7 >= excursionGrace;
                    provenance.update(item.selection);
                    provenance["native_reproduced"] = valid;
                    writeFiberReplayTraceArtifacts(output / "cases" / std::to_string(index), item.points, replay, threshold, provenance);
                    trialPruning.record(trialKey, trialDistance, valid);
                    caseRemaining[index].store(0);
                    std::vector<cv::Vec3d>().swap(item.points);
                    const auto done = ++completed;
                    if (done % 100 == 0)
                        std::osyncstream(std::cout) << "trial progress=" << done << '/' << cases.size()
                            << " skipped=" << skippedTrials.load() << std::endl;
                    return;
                }
                if (profileTracing)
                    vc::core::util::atomicWriteString(output / "cases" / std::to_string(index) / "profile.json",
                        caseProfileJson(index, std::chrono::duration<double>(std::chrono::steady_clock::now() - caseStart).count(),
                            readinessProfile, traceProfile).dump(2) + "\n");
                if (renderThisCase) {
                    // Render directly from this completed replay. Volume,
                    // prediction and normal chunk caches are concurrent; all
                    // selected directions may render and request data together.
                    const auto& predictions = livePredictions ? *livePredictions : *source.predictions;
                    // Coordinates stay in the bound source lattice; Volume applies
                    // the pyramid transform. Do not divide coordinates a second time.
                    const int ctLevel = source.provenance.value("ct_base_volume", false) &&
                        source.renderVolume->hasScaleLevel(1) ? 1 : 0;
                    const std::string ctSampling = "source level " + std::to_string(ctLevel) +
                        "; trilinear; fine-to-coarse fallback";
                    auto inspection = renderFiberReplayInspection(item.points, replay,
                        *referenceNormals, source.traceScale, source.renderSource,
                        [&](const auto& coords, int scale) -> cv::Mat_<uint8_t> {
                            return vc::core::util::renderCoordsTextureFineToCoarse(
                                coords, *source.renderVolume, ctLevel, scale, "Replay CT");
                        },
                        [&](const auto& coords, int scale) {
                            return renderFiberReplayPresenceTexture(coords, scale, source.renderSource,
                                source.traceScale, predictions);
                        }, threshold, item.reverse, source.voxelSizeUm, ctSampling,
                        0);
                    const auto caseDirectory = output / "cases" / std::to_string(index);
                    const auto directory = caseDirectory / "rendered";
                    auto temporary = caseDirectory / ("rendered.resume." + std::to_string(
                        std::chrono::steady_clock::now().time_since_epoch().count()));
                    const auto writeDirectory = resumeVisualization ? temporary : directory;
                    if (!std::filesystem::create_directory(writeDirectory))
                        throw std::runtime_error("render staging output already exists");
                    for (size_t p = 0; p < inspection.pages.size(); ++p)
                        if (!cv::imwrite((writeDirectory / ("strips_" + std::to_string(p) + ".jpg")).string(), inspection.pages[p]))
                            throw std::runtime_error("failed writing inspection image");

                    inspection.metadata.update({{"case_index", index}, {"provenance", provenance},
                        {"texture_source", {{"locator", source.renderSource.locator},
                            {"shape_zyx", source.renderSource.shapeZYX},
                            {"scale_from_base_xyz", source.renderSource.scaleFromBaseXYZ},
                            {"offset_from_base_xyz", source.renderSource.offsetFromBaseXYZ}}}});
                    // Renderer arcs are crop-local; recover positions on the original annotation.
                    const double originalBegin = item.selection.at("original_arcs_base")[0];
                    for (const char* key : {"restarts", "cross_sections"}) for (auto& entry : inspection.metadata[key]) {
                        entry["original_fiber_arc_base"] = entry["original_fiber_arc_base"].get<double>() + originalBegin;
                        entry["original_fiber_arc_mm"] = entry["original_fiber_arc_base"].get<double>() * source.voxelSizeUm / 1000;
                    }
                    vc::core::util::atomicWriteString(writeDirectory / "render.json", inspection.metadata.dump(2) + "\n");
                    if (resumeVisualization) {
                        if (std::filesystem::exists(directory)) {
                            auto quarantine = caseDirectory / ("rendered.incomplete." + std::to_string(
                                std::chrono::steady_clock::now().time_since_epoch().count()));
                            std::filesystem::rename(directory, quarantine);
                        }
                        std::filesystem::rename(writeDirectory, directory);
                    }
                }
                FiberReferenceReplayCase replayCase{item.fiber, item.run, item.reverse, item.points, item.length};
                outcomes[index] = measureFiberReferenceReplayOutcome(replayCase, replay);
                caseRemaining[index].store(0);
                std::osyncstream(std::cout) << "replay progress=" << ++completed << '/' << cases.size()
                    << " case=" << index << " source=" << source.id << " reverse=" << item.reverse
                    << '\n'; // Counts are reported by the shared episode metrics.
            }
        } catch (...) {
            failed.store(true);
            if (bridge) bridge->cancel("native replay case failed");
            std::lock_guard lock(errorMutex);
            if (!error) error = std::current_exception();
        }
    };
    std::unique_ptr<FiberCaseQueue> caseQueue;
    if (queuedCases) {
        std::osyncstream(std::cout) << "live case queue: cases=" << cases.size()
            << " workers=" << workers << std::endl;
        // Each reusable coroutine drains the shared atomic case queue.
        caseQueue = std::make_unique<FiberCaseQueue>(workers,
            trialSearch ? cases.size() : std::min(cases.size(), workers * 8), caseWork);
    } else {
        for (size_t w = 0; w < (live ? cases.size() : workers); ++w) pool.emplace_back(caseWork);
    }
    pool.clear(); // join before consuming results or propagating errors
    if (caseQueue) caseQueue->join();
    if (error) {
        vc::core::util::atomicWriteString(output / "incomplete.json", Json{{"completed_cases", completed.load()}, {"requested_cases", cases.size()}, {"config", persistedConfig}}.dump(2) + "\n");
        std::rethrow_exception(error);
    }
    if (resumeVisualization) {
        Json remaining = Json::array();
        for (size_t index = 0; index < cases.size(); ++index)
            if (!completeInspectionGallery(output / "cases" / std::to_string(index) / "rendered"))
                remaining.push_back(index);
        const Json status{{"format", "vc_visualization_resume"}, {"version", 1},
            {"cases", cases.size()}, {"rendered_cases", cases.size() - remaining.size()},
            {"remaining_cases", remaining}, {"batch_selected_cases", resumeRenderCases}};
        vc::core::util::atomicWriteString(output / "visualization_resume.json", status.dump(2) + "\n");
        // The Python publisher removes interruption markers only after the
        // complete visualization report validates every declared page.  The
        // original scientific summary and its timing remain byte-for-byte
        // untouched by this render-only operation.
        vc::core::util::atomicWriteString(output / "incomplete.json", Json{
            {"visualization_resume", status}, {"config", persistedConfig}}.dump(2) + "\n");
        return 0;
    }
    Json summaries = Json::array();
    if (trialSearch) {
        if (completed.load() != cases.size())
            throw std::runtime_error("incomplete failure-search work queue");
        vc::core::util::atomicWriteString(output / "summary.json", Json{
            {"completed", true}, {"backend", "failure_search"}, {"cases", cases.size()},
            {"evaluated_cases", completed.load() - skippedTrials.load()},
            {"cancelled_cases", skippedTrials.load()}, {"workers", workers},
            {"wall_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}
        }.dump(2) + "\n");
        return 0;
    }

    double directedLengthMm = 0.0, requestedDirectedLengthMm = 0.0;
    size_t totalFailures = 0, selectedFibers = 0, evaluatedCases = 0;
    Json allInfrastructureFailures = Json::array();
    auto commonConfig = persistedConfig;
    commonConfig["error_threshold_base_voxels"] = threshold;
    commonConfig["match_refine_steps"] = refine;
    for (size_t s = 0; s < sources.size(); ++s) {
        selectedFibers += sources[s].provenance.at("fibers").size();
        std::vector<FiberReferenceReplayOutcome> selected;
        Json sourceInfrastructureFailures = Json::array();
        size_t requestedCases = 0;
        double requestedLengthBase = 0.0;
        for (size_t i = 0; i < cases.size(); ++i) if (cases[i].source == s) {
            ++requestedCases;
            requestedLengthBase += cases[i].length;
            if (infrastructureFailures[i]) {
                sourceInfrastructureFailures.push_back(*infrastructureFailures[i]);
                allInfrastructureFailures.push_back(*infrastructureFailures[i]);
            } else {
                selected.push_back(outcomes[i]);
            }
        }
        const auto summary = summarizeFiberReferenceReplay(sources[s].provenance.at("fibers").size(), selected, sources[s].voxelSizeUm);
        auto value = fiberReferenceReplayBenchmarkJson(summary, selected,
            trace.beamWidth == 1 ? "native_greedy" : "native_beam", commonConfig, sources[s].provenance,
            {0, 0, 0}, {0, 0, 0});
        value["coordinates"]["crop_half_open"] = false;
        value["coordinates"].erase("minimum_base_xyz");
        value["coordinates"].erase("maximum_base_xyz");
        value["selection"] = references.at("policy");
        for (auto& c : value["cases"]) {
            const auto& run = sources[s].provenance.at("reference_selection")
                .at(c.at("source_index").get<size_t>()).at("runs").at(c.at("run_index").get<size_t>());
            c["reference_run"] = run;
            const double begin = run.at("original_arcs_base")[0], end = run.at("original_arcs_base")[1];
            for (auto& failure : c["failures"]) {
                const double arc = failure.at("directional_reference_arc_base_voxels");
                failure["original_fiber_reference_arc_base_voxels"] = c["direction"] == "reverse" ? end - arc : begin + arc;
            }
        }
        value["source_id"] = sources[s].id;
        value["threshold"] = fiberReplayThresholdDescriptorJson(threshold, tangentialFactor);
        value["requested_directed_cases"] = requestedCases;
        value["requested_directed_reference_length_base_voxels"] = requestedLengthBase;
        value["infrastructure_failures"] = std::move(sourceInfrastructureFailures);
        value["summary"]["requested_directed_cases"] = requestedCases;
        value["summary"]["requested_directed_reference_length_base_voxels"] = requestedLengthBase;
        value["summary"]["route_state_limit_cases"] = value["infrastructure_failures"].size();
        summaries.push_back(std::move(value));
        directedLengthMm += summary.directedReferenceLengthBaseVoxels * sources[s].voxelSizeUm / 1000.0;
        requestedDirectedLengthMm += requestedLengthBase * sources[s].voxelSizeUm / 1000.0;
        evaluatedCases += selected.size();
        totalFailures += summary.totalFailures;
    }
    vc::core::util::atomicWriteString(output / "summary.json", Json{
        {"version", 3}, {"metric_definition", "corpus-failures-plus-one-v1"},
        {"completed", true}, {"cases", cases.size()}, {"workers", workers},
        {"evaluated_cases", evaluatedCases},
        {"route_state_limit_cases", allInfrastructureFailures.size()},
        {"infrastructure_failures", std::move(allInfrastructureFailures)},
        {"selected_fibers", selectedFibers}, {"eligible_runs", [&] {
            size_t count = 0;
            for (const auto& source : references.at("sources"))
                for (const auto& fiber : source.at("fibers")) count += fiber.at("runs").size();
            return count;
        }()},
        {"directed_reference_length_mm", directedLengthMm},
        {"requested_directed_reference_length_mm", requestedDirectedLengthMm},
        {"failures", totalFailures},
        {"reference_selection_policy", references.at("policy")},
        {"failures_per_directed_mm", directedLengthMm > 0 ? Json(totalFailures / directedLengthMm) : Json(nullptr)},
        {"reliability_segments", evaluatedCases == 0 ? 0 : totalFailures + 1},
        {"mean_segment_length_mm", evaluatedCases == 0 ? Json(nullptr) : Json(directedLengthMm / (totalFailures + 1))},
        {"mean_segment_length_percent", evaluatedCases == 0 ? Json(nullptr) : Json(100.0 / (totalFailures + 1))},
        {"wall_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()},
        {"config", persistedConfig}, {"effective_trace", cli::traceConfigJson(trace)}, {"sources", summaries}}.dump(2) + "\n");
    return 0;
}
