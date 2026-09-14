#include "FiberTraceEvaluationCli.hpp"
#include "FiberTraceCli.hpp"
#include "FiberLiveBridge.hpp"
#include "vc/fiber_tracer/FiberReplay.hpp"
#include "vc/fiber_tracer/FiberJson.hpp"
#include "vc/fiber_tracer/FiberReferenceReplayBenchmark.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include "vc/core/util/GlobalCacheRoot.hpp"
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
#include <set>
#include <syncstream>
#include <thread>
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
    double traceScale = 1;
    double voxelSizeUm = 0;
    Json provenance;
    vc::lasagna::LasagnaDatasetManifest predictionManifest;
    std::shared_ptr<Volume> renderVolume;
    std::unique_ptr<std::mutex> renderMutex;
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
};

Json referencePlan(const Json& config, const std::filesystem::path& base)
{
    Json result{{"policy", "cp-bounded-no-lasagna-v1"}, {"sources", Json::array()}};
    for (const auto& source : config.at("sources")) {
        Json fibers = Json::array();
        for (const auto& value : source.at("fibers")) {
            // The Python coordinator resolves symlinks when binding sources.
            // Use the same existing-file identity for export and replay checks.
            const auto path = std::filesystem::canonical(localPath(base, value.get<std::string>()));
            auto plan = vc3dEvaluationReferencePlan(parseVc3dFiberJson(readJson(path), path.string()), path.string());
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
    const bool live = config.contains("live_endpoint");
    const bool profileTracing = config.value("profile_tracing", false);
    const auto readinessEntries = config.value("readiness_cache_entries", int64_t{4096});
    if (readinessEntries < 0) throw std::invalid_argument("readiness_cache_entries must be nonnegative");
    auto persistedConfig = config;
    persistedConfig.erase("reference_plan");
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
        const auto normalLocation = sourceLocation(configPath.parent_path(), entry.at("normal_manifest").get<std::string>());
        auto dataset = vc::lasagna::LasagnaDataset::openLocation(fiberLocation, options);
        const auto scales = resolveFiberPredictionTraceScales(dataset.manifest(), entry.value("inference_scaledown_power", 2));
        source.traceScale = scales.traceToBaseScale;
        auto manifest = dataset.manifest();
        manifest.workingToBaseScale = source.traceScale;
        source.predictionManifest = manifest;
        source.predictions = std::make_unique<FiberPredictionField>(vc::lasagna::LasagnaDataset(std::move(manifest)),
            FiberPredictionFieldBindingMode::TraceOptions, trace.embeddingWeight > 0);
        options.workingToBaseScale = source.traceScale;
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
        source.provenance = entry;
        source.provenance["fiber_manifest"] = fiberLocation;
        source.provenance["normal_manifest"] = normalLocation;
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
                source.renderVolume = source.renderCt.find("://") != std::string::npos
                    ? Volume::NewFromUrl(source.renderCt, options.remoteCacheRoot)
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
            source.renderMutex = std::make_unique<std::mutex>();
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
            for (const auto& run : selected.at("runs")) {
                for (bool reverse : {false, true}) {
                    std::vector<cv::Vec3d> points;
                    for (const auto& p : run.at("points_base_xyz")) points.emplace_back(p[0].get<double>(), p[1].get<double>(), p[2].get<double>());
                    if (reverse) std::reverse(points.begin(), points.end());
                    auto provenance = run; provenance.erase("points_base_xyz");
                    provenance["original_length_base_voxels"] = selected.at("original_length_base_voxels");
                    cases.push_back({sources.size(), fiberIndex, reverse, fiberPath, std::move(points),
                        run.at("length_base_voxels").get<double>(), run.at("run_index").get<size_t>(), std::move(provenance)});
                }
            }
            ++fiberIndex;
        }
        if (!fiberIndex) throw std::invalid_argument("source has no selected fibers: " + source.id);
        sources.push_back(std::move(source));
    }
    const int requestedWorkers = config.value("workers", 0);
    if (requestedWorkers < 0) throw std::invalid_argument("workers must be nonnegative");
    const size_t workers = std::min(cases.size(), requestedWorkers ? size_t(requestedWorkers) : size_t(std::max(1u, std::thread::hardware_concurrency())));
    if (live && cases.size() > config.value("max_live_cases", size_t{256}))
        throw std::invalid_argument("live case count exceeds explicit stack-preserving thread budget");
    std::unique_ptr<FiberLiveBridge> bridge;
    if (live) bridge = std::make_unique<FiberLiveBridge>(config.at("live_endpoint"));
    std::counting_semaphore<> cpuPermits(static_cast<std::ptrdiff_t>(workers));
    std::filesystem::create_directories(output.parent_path());
    if (!std::filesystem::create_directory(output)) throw std::invalid_argument("evaluation output already exists");
    std::filesystem::create_directory(output / "cases");
    std::vector<FiberReferenceReplayOutcome> outcomes(cases.size());
    std::vector<size_t> order(cases.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return cases[a].length > cases[b].length; });
    std::atomic<size_t> next{0}, completed{0};
    std::atomic<bool> failed{false};
    std::mutex errorMutex;
    std::exception_ptr error;
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::jthread> pool;
    for (size_t w = 0; w < (live ? cases.size() : workers); ++w) pool.emplace_back([&] {
        try {
            while (!failed.load()) {
                const size_t rank = next.fetch_add(1);
                if (rank >= order.size()) break;
                const size_t index = order[rank];
                const auto& item = cases[index];
                const auto& source = sources[item.source];
                FiberCaseCpuPermit permit(cpuPermits);
                if (failed.load()) break;
                std::atomic<double> remaining{item.length};
                FiberReadinessProfile readinessProfile;
                FiberTraceProfile traceProfile;
                FiberCaseReadiness readiness(static_cast<size_t>(readinessEntries), profileTracing ? &readinessProfile : nullptr);
                const auto caseStart = std::chrono::steady_clock::now();
                auto nextProfile = caseStart + std::chrono::seconds(10);
                std::unique_ptr<FiberPredictionField> livePredictions;
                if (bridge) {
                    livePredictions = std::make_unique<FiberPredictionField>(
                        vc::lasagna::LasagnaDataset(source.predictionManifest),
                        FiberPredictionFieldBindingMode::TraceOptions, trace.embeddingWeight > 0);
                    livePredictions->setReadinessGate([&](const auto& dependencies) {
                        readiness.await(dependencies, [&](const auto& missing) {
                            permit.suspend([&] { bridge->await(source.id, index, remaining.load(), missing,
                                profileTracing ? &readinessProfile : nullptr); });
                        }, [&] { bridge->checkActive(); });
                    });
                }
                FiberReplayTraceRequest request;
                request.fiber.linePointsXyzBase = item.points;
                request.fiber.controlPointsXyzBase = {item.points.front(), item.points.back()};
                request.fiber.controlPointLineIndices = {0, item.points.size() - 1};
                request.traceToBaseScale = source.traceScale;
                request.errorThresholdBaseVoxels = threshold;
                request.excursionGraceBaseVoxels = excursionGrace;
                request.tangentialThresholdFactor = tangentialFactor;
                request.matchRefineSteps = refine;
                request.config = trace;
                request.config.profile = profileTracing ? &traceProfile : nullptr;
                request.config.traceToBaseScale = source.traceScale;
                request.config.baseVoxelSizeUm = source.voxelSizeUm;
                const auto replay = traceFiberReplay(livePredictions ? *livePredictions : *source.predictions,
                    request, *source.normals, source.traceScale,
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
                auto provenance = source.provenance;
                provenance["source_id"] = source.id;
                provenance["fiber_path"] = item.fiberPath.string();
                provenance["reverse"] = item.reverse;
                provenance["case_index"] = index;
                provenance["reference_run"] = item.selection;
                writeFiberReplayTraceArtifacts(output / "cases" / std::to_string(index), item.points, replay, threshold, provenance);
                if (profileTracing)
                    vc::core::util::atomicWriteString(output / "cases" / std::to_string(index) / "profile.json",
                        caseProfileJson(index, std::chrono::duration<double>(std::chrono::steady_clock::now() - caseStart).count(),
                            readinessProfile, traceProfile).dump(2) + "\n");
                if (render) {
                    // Render directly from this completed replay, never reload
                    // trace.json. Bound concurrent textures per source and keep
                    // the same normal scale/geometry as the post-hoc API.
                    std::unique_lock renderLock(*source.renderMutex, std::defer_lock);
                    // A renderer may await inference and release its CPU permit.
                    // Mutex waiters must not consume all permits needed to resume it.
                    permit.suspend([&] { renderLock.lock(); });
                    const auto& predictions = livePredictions ? *livePredictions : *source.predictions;
                    // Coordinates stay in the bound source lattice; Volume applies
                    // the pyramid transform. Do not divide coordinates a second time.
                    const int ctLevel = source.provenance.value("ct_base_volume", false) &&
                        source.renderVolume->hasScaleLevel(1) ? 1 : 0;
                    const std::string ctSampling = "source level " + std::to_string(ctLevel) +
                        "; trilinear; fine-to-coarse fallback";
                    auto inspection = renderFiberReplayInspection(item.points, replay,
                        *source.normals, source.traceScale, source.renderSource,
                        [&](const auto& coords, int scale) -> cv::Mat_<uint8_t> {
                            return vc::core::util::renderCoordsTextureFineToCoarse(
                                coords, *source.renderVolume, ctLevel, scale, "Replay CT");
                        },
                        [&](const auto& coords, int scale) {
                            return renderFiberReplayPresenceTexture(coords, scale, source.renderSource,
                                source.traceScale, predictions);
                        }, threshold, item.reverse, source.voxelSizeUm, ctSampling);
                    const auto directory = output / "cases" / std::to_string(index) / "rendered";
                    if (!std::filesystem::create_directory(directory)) throw std::runtime_error("render output already exists");
                    for (size_t p = 0; p < inspection.pages.size(); ++p)
                        if (!cv::imwrite((directory / ("strips_" + std::to_string(p) + ".jpg")).string(), inspection.pages[p]))
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
                    vc::core::util::atomicWriteString(directory / "render.json", inspection.metadata.dump(2) + "\n");
                }
                FiberReferenceReplayCase replayCase{item.fiber, item.run, item.reverse, item.points, item.length};
                outcomes[index] = measureFiberReferenceReplayOutcome(replayCase, replay);
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
    });
    pool.clear(); // join before consuming results or propagating errors
    if (error) {
        vc::core::util::atomicWriteString(output / "incomplete.json", Json{{"completed_cases", completed.load()}, {"requested_cases", cases.size()}, {"config", persistedConfig}}.dump(2) + "\n");
        std::rethrow_exception(error);
    }
    Json summaries = Json::array();
    double directedLengthMm = 0.0;
    size_t totalFailures = 0, selectedFibers = 0;
    auto commonConfig = persistedConfig;
    commonConfig["error_threshold_base_voxels"] = threshold;
    commonConfig["match_refine_steps"] = refine;
    for (size_t s = 0; s < sources.size(); ++s) {
        selectedFibers += sources[s].provenance.at("fibers").size();
        std::vector<FiberReferenceReplayOutcome> selected;
        for (size_t i = 0; i < cases.size(); ++i) if (cases[i].source == s) selected.push_back(outcomes[i]);
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
        summaries.push_back(std::move(value));
        directedLengthMm += summary.directedReferenceLengthBaseVoxels * sources[s].voxelSizeUm / 1000.0;
        totalFailures += summary.totalFailures;
    }
    vc::core::util::atomicWriteString(output / "summary.json", Json{
        {"version", 2}, {"metric_definition", "complete-reference-partition-v1"},
        {"completed", true}, {"cases", cases.size()}, {"workers", workers},
        {"selected_fibers", selectedFibers}, {"eligible_runs", cases.size() / 2},
        {"directed_reference_length_mm", directedLengthMm}, {"failures", totalFailures},
        {"reference_selection_policy", references.at("policy")},
        {"failures_per_directed_mm", directedLengthMm > 0 ? Json(totalFailures / directedLengthMm) : Json(nullptr)},
        {"reliability_segments", totalFailures + cases.size()},
        {"mean_segment_length_mm", cases.empty() ? Json(nullptr) : Json(directedLengthMm / (totalFailures + cases.size()))},
        {"mean_segment_length_percent", cases.empty() ? Json(nullptr) : Json(100.0 / (totalFailures + cases.size()))},
        {"wall_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()},
        {"config", persistedConfig}, {"effective_trace", cli::traceConfigJson(trace)}, {"sources", summaries}}.dump(2) + "\n");
    return 0;
}
