#include "FiberTraceEvaluationCli.hpp"
#include "FiberTraceCli.hpp"
#include "FiberLiveBridge.hpp"
#include "vc/fiber_tracer/FiberReplay.hpp"
#include "vc/fiber_tracer/FiberJson.hpp"
#include "vc/fiber_tracer/FiberReferenceReplayBenchmark.hpp"
#include "vc/core/types/Volume.hpp"
#include "vc/core/util/AtomicFile.hpp"
#include "vc/core/util/GlobalCacheRoot.hpp"

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
};

struct Case {
    size_t source = 0;
    size_t fiber = 0;
    bool reverse = false;
    std::filesystem::path fiberPath;
    std::vector<cv::Vec3d> points;
    double length = 0;
};

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
    vc::core::util::atomicWriteString(output / "render.json", metadata.dump(2) + "\n");
}

void renderArtifact(const std::filesystem::path& artifact, const std::filesystem::path& settingsPath)
{
    const auto document = readJson(artifact);
    if (document.at("format") != "vc_native_replay_artifact" || document.at("version") != 1)
        throw std::invalid_argument("unsupported replay artifact");
    const auto settings = readJson(settingsPath);
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
    const vc::lasagna::LasagnaNormalSampler normals(dataset,
        vc::lasagna::LasagnaNormalSamplerOptions{settings.value("cache_bytes", size_t{256} << 20)});
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
    if (argc < 2 || std::string_view(argv[1]) != "replay-set") throw std::invalid_argument(usage);
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
    const auto trace = traceOptions(config.value("trace_args", Json::array()));
    const bool live = config.contains("live_endpoint");
    auto persistedConfig = config;
    const bool render = config.value("render", false) || renderFlag;
    persistedConfig["render"] = render;
    if (live) persistedConfig["live_endpoint"].erase("token");
    if (live && trace.parallelThreads != 1)
        throw std::invalid_argument("live replay requires --threads 1; concurrency is across cases");
    const double threshold = config.value("error_threshold_base_voxels", 20.0);
    const double refine = config.value("match_refine_steps", 1.0);
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
            config.value("cache_bytes_per_channel", size_t{256} << 20));
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
        source.normals = std::make_unique<vc::lasagna::LasagnaNormalSampler>(normals,
            vc::lasagna::LasagnaNormalSamplerOptions{config.value("cache_bytes_per_channel", size_t{256} << 20)});
        source.provenance = entry;
        source.provenance["fiber_manifest"] = fiberLocation;
        source.provenance["normal_manifest"] = normalLocation;
        if (entry.contains("ct_volume")) source.provenance["ct_volume"] = sourceLocation(configPath.parent_path(), entry.at("ct_volume").get<std::string>());
        if (render) {
            if (!entry.contains("ct_volume")) throw std::invalid_argument("--render requires ct_volume for every source");
            source.renderCt = source.provenance.at("ct_volume").get<std::string>();
            if (source.renderCt.find("://") != std::string::npos)
                throw std::invalid_argument("--render currently requires local OME-Zarr CT metadata");
            source.renderVolume = Volume::New(source.renderCt);
            (void)validateFiberReplayStripCtVolume(*source.renderVolume, source.renderCt);
            source.renderMutex = std::make_unique<std::mutex>();
        }
        source.provenance["trace_to_base"] = source.traceScale;
        source.provenance["effective_trace"] = cli::traceConfigJson(trace);
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
            const auto fiber = parseVc3dFiberJson(annotation, fiberPath.string());
            const double length = makePolylineArcGeometry(fiber.linePoints).length();
            if (!(length > 0)) throw std::invalid_argument("degenerate fiber " + fiberPath.string());
            for (bool reverse : {false, true}) {
                auto points = fiber.linePoints;
                if (reverse) std::reverse(points.begin(), points.end());
                cases.push_back({sources.size(), fiberIndex, reverse, fiberPath, std::move(points), length});
            }
            ++fiberIndex;
        }
        if (!fiberIndex) throw std::invalid_argument("source has no selected fibers: " + source.id);
        sources.push_back(std::move(source));
    }
    if (cases.empty()) throw std::invalid_argument("empty replay set");
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
                std::unique_ptr<FiberPredictionField> livePredictions;
                if (bridge) {
                    livePredictions = std::make_unique<FiberPredictionField>(
                        vc::lasagna::LasagnaDataset(source.predictionManifest),
                        config.value("cache_bytes_per_channel", size_t{256} << 20));
                    livePredictions->setReadinessGate([&](const auto& dependencies) {
                        permit.suspend([&] { bridge->await(source.id, index, remaining.load(), dependencies); });
                    });
                }
                FiberReplayTraceRequest request;
                request.fiber.linePointsXyzBase = item.points;
                request.fiber.controlPointsXyzBase = {item.points.front(), item.points.back()};
                request.fiber.controlPointLineIndices = {0, item.points.size() - 1};
                request.traceToBaseScale = source.traceScale;
                request.errorThresholdBaseVoxels = threshold;
                request.matchRefineSteps = refine;
                request.config = trace;
                request.config.traceToBaseScale = source.traceScale;
                request.config.baseVoxelSizeUm = source.voxelSizeUm;
                const auto replay = traceFiberReplay(livePredictions ? *livePredictions : *source.predictions,
                    request, *source.normals, source.traceScale,
                    [&](const FiberTraceProgress& progress) {
                        if (progress.referenceArcFraction)
                            remaining.store(item.length * (1 - std::clamp(*progress.referenceArcFraction, 0.0, 1.0)));
                    });
                auto provenance = source.provenance;
                provenance["source_id"] = source.id;
                provenance["fiber_path"] = item.fiberPath.string();
                provenance["reverse"] = item.reverse;
                provenance["case_index"] = index;
                writeFiberReplayTraceArtifacts(output / "cases" / std::to_string(index), item.points, replay, threshold, provenance);
                if (render) {
                    // Render directly from this completed replay, never reload
                    // trace.json. Bound concurrent textures per source and keep
                    // the same normal scale/geometry as the post-hoc API.
                    std::lock_guard renderLock(*source.renderMutex);
                    const auto overview = renderFiberReplayTraceOverview(item.points, replay,
                        *source.normals, source.traceScale, 1, *source.renderVolume, source.renderCt);
                    writeRenderedOverview(output / "cases" / std::to_string(index) / "rendered", overview,
                        Json{{"mode", "inline"}, {"case_index", index}, {"provenance", provenance}});
                }
                FiberReferenceReplayCase replayCase{item.fiber, 0, item.reverse, item.points, item.length};
                outcomes[index] = measureFiberReferenceReplayOutcome(replayCase, replay);
                std::osyncstream(std::cout) << "replay progress=" << ++completed << '/' << cases.size()
                    << " case=" << index << " source=" << source.id << " reverse=" << item.reverse
                    << " failures=" << replay.failures.size() << '\n';
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
    size_t totalFailures = 0;
    auto commonConfig = persistedConfig;
    commonConfig["error_threshold_base_voxels"] = threshold;
    commonConfig["match_refine_steps"] = refine;
    for (size_t s = 0; s < sources.size(); ++s) {
        std::vector<FiberReferenceReplayOutcome> selected;
        for (size_t i = 0; i < cases.size(); ++i) if (cases[i].source == s) selected.push_back(outcomes[i]);
        const auto summary = summarizeFiberReferenceReplay(selected.size() / 2, selected, sources[s].voxelSizeUm);
        auto value = fiberReferenceReplayBenchmarkJson(summary, selected,
            trace.beamWidth == 1 ? "native_greedy" : "native_beam", commonConfig, sources[s].provenance,
            {0, 0, 0}, {0, 0, 0});
        value["coordinates"]["crop_half_open"] = false;
        value["coordinates"].erase("minimum_base_xyz");
        value["coordinates"].erase("maximum_base_xyz");
        value["selection"] = "whole_explicit_fibers_no_crop";
        value["source_id"] = sources[s].id;
        summaries.push_back(std::move(value));
        directedLengthMm += summary.directedReferenceLengthBaseVoxels * sources[s].voxelSizeUm / 1000.0;
        totalFailures += summary.totalFailures;
    }
    vc::core::util::atomicWriteString(output / "summary.json", Json{
        {"version", 1}, {"completed", true}, {"cases", cases.size()}, {"workers", workers},
        {"directed_reference_length_mm", directedLengthMm}, {"failures", totalFailures},
        {"failures_per_directed_mm", totalFailures / directedLengthMm},
        {"mean_segment_length_mm", directedLengthMm / (totalFailures + 1)},
        {"wall_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()},
        {"config", persistedConfig}, {"effective_trace", cli::traceConfigJson(trace)}, {"sources", summaries}}.dump(2) + "\n");
    return 0;
}
