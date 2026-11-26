// Scaffold winding solve from the command line.
//
// Reads a store's annotated fibers, their links, approval layers and
// umbilicus, assembles the integer winding arcs, solves them exactly with
// LEMON and writes the residual ranking the approval workflow orders by. The
// same library backs the fiber map's solver dock, so a run here and a run in
// VC3D are the same solve.
//
// With `constraints.enabled` the run also measures the scaffold's own
// crossings: strip normals anchored by the annotation, the whole-fiber normal
// signs those crossings decide, and the perpendicular, parallel and enclosure
// relations they assert. Measuring a winding separation needs a Lasagna
// normal dataset (`winding.manifest`), whose grad_mag winding density is the
// measure; without one the relations keep no separation and witness
// enclosures only.
#include "vc/fiber_graph/ScaffoldCache.hpp"
#include "vc/fiber_graph/ScaffoldConstraints.hpp"
#include <future>
#include <mutex>
#include <vc_object_store/objects/ObjectCollection.hpp>
#include <vc_object_store/objects/FiberGraphSolveExtension.hpp>
#include <vc_object_store/objects/FiberConstraintGraph.hpp>
#include "vc/fiber_graph/ScaffoldFieldComparison.hpp"
#include "vc/fiber_graph/ScaffoldSolveRuns.hpp"
#include "vc/fiber_graph/ScaffoldWindingSolve.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "vc/lasagna/LasagnaNormalSampler.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/ObjectStore.hpp>
#include <vc_object_store/objects/StoreReader.hpp>

#include <nlohmann/json.hpp>

#include <chrono>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace
{

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using namespace vc::fiber_graph;

Json readJson(const std::string& path)
{
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot read " + path);
    Json value;
    stream >> value;
    return value;
}

void writeJson(const std::string& path, const Json& value)
{
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot write " + path);
    stream << value.dump(1) << '\n';
}

double seconds(const Clock::time_point since)
{
    return std::chrono::duration<double>(Clock::now() - since).count();
}

ScaffoldLoadOptions loadOptionsFrom(const Json& cfg)
{
    ScaffoldLoadOptions options;
    options.volume = cfg.value("volume", std::string{});
    options.scale = cfg.value("scale", 1.0);
    if (cfg.contains("handedness")) options.handedness = cfg.at("handedness").get<int>();
    const auto source = cfg.value("family_source", std::string{"any"});
    if (source == "any") options.familySource = ScaffoldFamilySource::Any;
    else if (source == "manual") options.familySource = ScaffoldFamilySource::Manual;
    else if (source == "automatic") options.familySource = ScaffoldFamilySource::Automatic;
    else throw std::invalid_argument("family_source must be any, manual or automatic");
    options.includeUntagged = cfg.value("include_untagged", false);
    // approvals: false solves as if nothing had been approved, marked broken or reviewed.
    options.readApprovals = cfg.value("approvals", true);
    // A restricted run: only these fibers are loaded (the GUI's scope
    // selector states the map's selected network here).
    for (const auto& uuid : cfg.value("fibers", Json::array()))
        options.fiberSubset.push_back(uuid.get<std::string>());
    return options;
}

ScaffoldNormalOptions normalOptionsFrom(const Json& cfg)
{
    const auto section = cfg.value("normals", Json::object());
    ScaffoldNormalOptions options;
    options.strip.twistWeight = section.value("twist_weight", options.strip.twistWeight);
    options.strip.lateralWeight = section.value("lateral_weight", options.strip.lateralWeight);
    options.strip.smoothingSigma = section.value("smoothing_sigma", options.strip.smoothingSigma);
    options.strip.resampleSpacing = section.value("resample_spacing", options.strip.resampleSpacing);
    options.strip.maxIterations = section.value("max_iterations", options.strip.maxIterations);
    options.predictionChunkPoints = section.value("prediction_chunk", options.predictionChunkPoints);
    options.manualDisplayNormals = section.value("manual_display_normals", options.manualDisplayNormals);
    options.approvedNormalPieces = section.value("approved_pieces", options.approvedNormalPieces);
    options.adjacentLinkAnchors = section.value("adjacent_links", options.adjacentLinkAnchors);
    options.anchorWeight = section.value("anchor_weight", options.anchorWeight);
    options.strip.axisWeight = section.value("axis_weight", options.strip.axisWeight);
    options.presenceFloor = section.value("presence_floor", options.presenceFloor);
    options.presenceFull = section.value("presence_full", options.presenceFull);
    // The links that anchor a normal are the links that constrain a winding,
    // unless the caller names its own set.
    const auto links = cfg.value("links", Json::object());
    if (links.contains("approval_states"))
        options.linkApprovalStates = links.at("approval_states").get<std::vector<std::string>>();
    options.includeAutoLinks = links.value("include_auto", options.includeAutoLinks);
    if (section.contains("link_approval_states"))
        options.linkApprovalStates = section.at("link_approval_states").get<std::vector<std::string>>();
    options.includeAutoLinks = section.value("include_auto", options.includeAutoLinks);
    options.workers = cfg.value("workers", std::size_t{1});
    return options;
}

ScaffoldEventOptions eventOptionsFrom(const Json& cfg)
{
    const auto section = cfg.value("constraints", Json::object());
    ScaffoldEventOptions options;
    options.resampleSpacing = section.value("resample_spacing", options.resampleSpacing);
    options.maximumDistance = section.value("maximum_distance", options.maximumDistance);
    options.tangentWindow = section.value("tangent_window", options.tangentWindow);
    options.relaxEndpointCertification =
        section.value("relax_endpoint_certification", options.relaxEndpointCertification);
    options.measureWindingDistances = section.value("winding_distances", options.measureWindingDistances);
    options.windingIntegrationStep = section.value("winding_step", options.windingIntegrationStep);
    options.maximumWindingDistance = section.value("maximum_winding_distance", options.maximumWindingDistance);
    options.enforceMaximumWindingDistance =
        section.value("enforce_maximum_winding", options.enforceMaximumWindingDistance);
    // constraints.crossing_filter (default on): follow every perpendicular crossing through its
    // fibers' normal surfaces and apply the shared crossing rule in the sign solve and the constraints.
    options.crossingFindings = section.value("crossing_filter", options.crossingFindings);
    options.workers = cfg.value("workers", std::size_t{1});
    return options;
}

ScaffoldMeasuredOptions measuredOptionsFrom(const Json& cfg)
{
    const auto weights = cfg.value("weights", Json::object());
    const auto section = cfg.value("constraints", Json::object());
    ScaffoldMeasuredOptions options;
    options.perpendicularWeight = weights.value("perpendicular", options.perpendicularWeight);
    options.parallelWeight = weights.value("parallel", options.parallelWeight);
    options.enclosureWeight = weights.value("enclosure", options.enclosureWeight);
    options.signWeight = weights.value("sign", options.signWeight);
    options.approvedSignFactor = weights.value("sign_approved_factor", options.approvedSignFactor);
    options.reviewedWeight = weights.value("reviewed", options.reviewedWeight);
    options.crossingConfidence = weights.value("crossing_confidence", options.crossingConfidence);
    options.perpendicularOffset = section.value("perpendicular_offset", options.perpendicularOffset);
    options.windingScale = section.value("winding_scale", options.windingScale);
    options.enclosures = section.value("enclosures", options.enclosures);
    options.crossingFilter = section.value("crossing_filter", options.crossingFilter);
    // "split" (the automatic stage's two weighted integers) or "rounded" (one
    // constraint at the nearer integer, what the scaffold dock runs).
    const auto separations = section.value("separations", std::string{"split"});
    if (separations == "rounded") options.separations = ScaffoldSeparationMode::Rounded;
    else if (separations != "split")
        throw std::invalid_argument("constraints.separations must be \"split\" or \"rounded\"");
    return options;
}

ScaffoldSignOptions signOptionsFrom(const Json& cfg)
{
    using namespace vc::fiber_tracer;
    const auto section = cfg.value("sign", Json::object());
    ScaffoldSignOptions options;
    options.solve = section.value("solve", options.solve);
    options.crossingFilter = cfg.value("constraints", Json::object()).value("crossing_filter", options.crossingFilter);
    const auto backend = section.value("backend", std::string{"irls"});
    if (backend == "irls") options.sign.backend = FiberSignBackend::Irls;
    else if (backend == "ceres") options.sign.backend = FiberSignBackend::Ceres;
    else throw std::invalid_argument("sign.backend must be irls or ceres");
    const auto distance = section.value("winding_distance_weight", std::string{"inverse_square"});
    if (distance == "inverse_square") options.sign.distanceWeight = FiberSignDistanceWeight::InverseSquare;
    else if (distance == "none") options.sign.distanceWeight = FiberSignDistanceWeight::None;
    else throw std::invalid_argument("sign.winding_distance_weight must be inverse_square or none");
    options.sign.dotWeightPower = section.value("normal_dot_weight_power", options.sign.dotWeightPower);
    options.sign.priorWeight = section.value("prior_weight", options.sign.priorWeight);
    options.sign.fixedSignWeight = section.value("fixed_sign_weight", options.sign.fixedSignWeight);
    options.sign.brokenPenalty = section.value("broken_penalty", options.sign.brokenPenalty);
    options.sign.epsilon = section.value("epsilon", options.sign.epsilon);
    options.sign.iterations = section.value("iterations", options.sign.iterations);
    options.sign.cgIterations = section.value("cg_iterations", options.sign.cgIterations);
    options.sign.keepRounds = section.value("keep_rounds", options.sign.keepRounds);
    options.sign.fixedKeep = section.value("fixed_keep", options.sign.fixedKeep);
    options.sign.threads = int(cfg.value("workers", std::size_t{1}));
    return options;
}

// The run's time as a person reads it, for the rows it writes.
std::string isoTimestampNow()
{
    const auto now = std::chrono::system_clock::now();
    return std::format("{:%FT%TZ}", std::chrono::time_point_cast<std::chrono::seconds>(now));
}

// The run's time, Unix seconds UTC.
std::int64_t unixSecondsNow()
{
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

ScaffoldSolveOptions solveOptionsFrom(const Json& cfg)
{
    ScaffoldSolveOptions options;
    const auto links = cfg.value("links", Json::object());
    if (links.contains("approval_states"))
        options.links.approvalStates = links.at("approval_states").get<std::vector<std::string>>();
    options.links.useLinks = links.value("use_links", true);
    options.links.includeAuto = links.value("include_auto", false);
    options.links.maxAnchorDistance = links.value("max_anchor_distance", 5.0);
    options.links.refineMaximumDistance = links.value("refine_maximum_distance", options.links.refineMaximumDistance);
    options.links.deduplicateArc = links.value("deduplicate_arc", options.links.deduplicateArc);
    options.links.weight = cfg.value("weights", Json::object()).value("manual", 1.0);
    const auto lemon = cfg.value("lemon", Json::object());
    options.flow.backend = vc::geometry::parseWindingFlowBackend(lemon.value("backend", std::string{"network_simplex"}));
    options.flow.capacityScale = lemon.value("capacity_scale", std::int64_t{1000});
    const auto continuation = cfg.value("continuation", Json::object());
    options.continuation.enabled = continuation.value("enabled", options.continuation.enabled);
    options.continuation.weight = continuation.value("weight", options.continuation.weight);
    options.continuation.intervalVx = continuation.value("interval_vx", options.continuation.intervalVx);
    options.maxAlternativePaths = cfg.value("max_alternative_paths", std::int64_t{4});
    options.workers = cfg.value("workers", std::size_t{1});
    return options;
}

Json linkReportJson(const ScaffoldLinkReport& report)
{
    return Json{{"links", report.links},
                {"used", report.used},
                {"ordinary", report.ordinary},
                {"adjacent", report.adjacent},
                {"skipped_approval_state", report.skippedApprovalState},
                {"skipped_automatic", report.skippedAutomatic},
                {"skipped_endpoint", report.skippedEndpoint},
                {"skipped_self_link", report.skippedSelfLink},
                {"skipped_adjacent_same_family", report.skippedAdjacentSameFamily},
                {"skipped_missing_anchor", report.skippedMissingAnchor},
                {"skipped_untagged", report.skippedUntagged},
                {"max_anchor_distance_vx", report.maxAnchorDistance},
                {"rejected_angle", report.skippedAngle},
                {"rejected_end_support", report.skippedEndSupport},
                {"rejected_distance", report.skippedDistance},
                {"rejected_consistency", report.skippedConsistency},
                {"rejected_tail", report.skippedTail},
                {"rejected_gap", report.skippedGap},
                {"max_refinement_shift_vx", report.maxRefinementShift},
                {"links_with_crossing", report.linksWithCrossing},
                {"crossings_replaced", report.deduplicatedCrossings}};
}

Json pieceJson(const ScaffoldGraph& graph, const ScaffoldPiece& piece)
{
    return Json{{"fiber", graph.fibers[std::size_t(piece.fiber)].uuid},
                {"arc_from_vx", piece.arcFrom},
                {"arc_to_vx", piece.arcTo},
                {"length_vx", piece.length()},
                {"control_from", piece.controlFrom},
                {"control_to", piece.controlTo},
                {"constraint_from", piece.constraintFrom},
                {"constraint_to", piece.constraintTo}};
}

Json constraintJson(const ScaffoldGraph& graph, const ScaffoldConstraint& constraint)
{
    Json value{{"a", graph.fibers[std::size_t(constraint.a)].uuid},
               {"b", graph.fibers[std::size_t(constraint.b)].uuid},
               {"kind", vc::fiber_graph::scaffoldConstraintKindName(constraint.kind)},
               {"winding_distance", std::isfinite(constraint.windingDistance) ? Json(constraint.windingDistance) : Json(nullptr)},
               {"arc_a_vx", constraint.arcA},
               {"arc_b_vx", constraint.arcB},
               {"step", constraint.step},
               {"target", constraint.target},
               {"weight", constraint.weight}};
    if (constraint.link >= 0) value["link"] = graph.links[std::size_t(constraint.link)].uuid;
    return value;
}

int run(const std::string& configPath)
{
    const Json cfg = readJson(configPath);
    const auto readStarted = Clock::now();

    // Persisting a run appends extension rows to the solved store itself, so
    // it is opened writable only when asked for.
    const bool persist = cfg.value("persist_run", false);
    vc::objects::OpenOptions storeOptions;
    storeOptions.readOnly = !persist;
    storeOptions.author = "vc_scaffold_solve";
    auto store = vc::objects::ObjectStore::open(cfg.at("store").get<std::string>(), storeOptions);
    ScaffoldGraph graph;
    // Derived results earlier runs left in the store, and what this run
    // computes for the next one (vc/fiber_graph/ScaffoldCache.hpp).
    ScaffoldCache derivedCache;
    ScaffoldCacheSink cacheSink;
    cacheSink.author = storeOptions.author;
    cacheSink.computedAt = isoTimestampNow();
    std::vector<vc::objects::FiberConstraintApproval> constraintApprovals;
    // When the annotation itself last changed, so a log says whether a later
    // run read the same fibers (the Solves dock's stale verdict).
    std::int64_t annotationCommit = 0;
    double datasetSeconds = 0, openSeconds = 0, graphSeconds = 0, approvalSeconds = 0, cacheSeconds = 0, neighbourSeconds = 0,
           windingSeconds = 0, persistSeconds = 0;
    // The three reads that need nothing of each other run side by side: the
    // fibers (below), the derived rows through a reader of their own, and the
    // full volume's normals.
    auto cacheReading = std::async(std::launch::async, [&] {
        const auto started = Clock::now();
        vc::objects::OpenOptions readOnly;
        readOnly.readOnly = true;
        auto second = vc::objects::ObjectStore::open(cfg.at("store").get<std::string>(), readOnly);
        const auto txn = second.read();
        ScaffoldCache cache{vc::objects::LocalStoreReader(second, txn)};
        cacheSeconds = seconds(started);
        return cache;
    });
    std::future<vc::fiber_graph::ScaffoldNeighbourSource> neighbourOpening;
    if (cfg.value("constraints", Json::object()).value("enabled", false) &&
        cfg.value("normals", Json::object()).value("source", std::string{"geometry"}) == "neighbours" &&
        cfg.value("neighbours", Json::object()).contains("samples"))
        neighbourOpening = std::async(std::launch::async, [&] {
            const auto started = Clock::now();
            auto source = vc::fiber_graph::openScaffoldNeighbourSource(
                cfg.at("neighbours").at("samples").get<std::string>(), ScaffoldNormalOptions{}.neighbours.variant.radius);
            neighbourSeconds = seconds(started);
            return source;
        });
    {
        openSeconds = seconds(readStarted);
        const auto txn = store.read();
        const vc::objects::LocalStoreReader reader(store, txn);
        auto started = Clock::now();
        graph = loadScaffoldGraph(reader, loadOptionsFrom(cfg));
        graphSeconds = seconds(started);
        started = Clock::now();
        if (cfg.value("approvals", true)) constraintApprovals = vc::objects::readFiberConstraintApprovals(reader);
        annotationCommit = scaffoldAnnotationCommit(reader);
        approvalSeconds = seconds(started);
    }
    derivedCache = cacheReading.get();
    const double readSeconds = seconds(readStarted);

    const auto geometryStarted = Clock::now();
    const ScaffoldGeometry geometry(graph, cfg.value("workers", std::size_t{1}));
    const double geometrySeconds = seconds(geometryStarted);

    const auto options = solveOptionsFrom(cfg);
    const bool measure = cfg.value("constraints", Json::object()).value("enabled", false);
    std::optional<ScaffoldNormalOptions> normalOptions;
    std::optional<ScaffoldEventOptions> eventOptions;
    std::optional<ScaffoldMeasuredOptions> measuredOptions;
    std::optional<ScaffoldSignOptions> signOptions;
    std::optional<ScaffoldNormals> normals;
    std::vector<ScaffoldEvent> events;
    ScaffoldEventReport eventReport;
    ScaffoldMeasuredReport measuredReport;
    ScaffoldSignResult signs;
    ScaffoldApprovalReport approvalReport;
    vc::fiber_graph::ScaffoldFacingReport facingReport;
    std::vector<ScaffoldCrossing> crossings;
    std::vector<ScaffoldConstraint> measured;
    double normalSeconds = 0, eventSeconds = 0, signSeconds = 0, measuredSeconds = 0;
    std::unique_ptr<vc::lasagna::LasagnaNormalSampler> sampler;
    std::mutex samplerMutex;
    ScaffoldWindingDistances distances;
    if (measure) {
        normalOptions = normalOptionsFrom(cfg);
        eventOptions = eventOptionsFrom(cfg);
        normalOptions->cache = &derivedCache;
        eventOptions->cache = &derivedCache;
        if (persist) {
            normalOptions->sink = &cacheSink;
            eventOptions->sink = &cacheSink;
        }
        measuredOptions = measuredOptionsFrom(cfg);
        signOptions = signOptionsFrom(cfg);
        measuredOptions->confidence = signOptions->sign;
        if (eventOptions->measureWindingDistances) {
            const auto section = cfg.value("winding", Json::object());
            if (!section.contains("manifest"))
                throw std::invalid_argument("constraints.winding_distances needs winding.manifest, or set it to false");
            // The density is sampled on the Lasagna dataset's base grid, so
            // the annotation frame has to be read at full resolution.
            if (graph.scale != 1)
                throw std::invalid_argument("measured winding separations are sampled in base voxels; run at scale 1");
            vc::lasagna::LasagnaDatasetOpenOptions open;
            open.workingToBaseScale = 1.;
            if (section.contains("remote_cache")) open.remoteCacheRoot = section.at("remote_cache").get<std::string>();
            vc::lasagna::LasagnaNormalSamplerOptions samplerOptions;
            samplerOptions.maxCachedBytes = section.value("cache_bytes", samplerOptions.maxCachedBytes);
            samplerOptions.readerThreads = section.value("reader_threads", std::size_t{0});
            samplerOptions.windingRegionBaseVoxels =
                section.value("region_base_voxels", samplerOptions.windingRegionBaseVoxels);
            samplerOptions.windingBatchPoints = section.value("batch_points", samplerOptions.windingBatchPoints);
            // Opened when the first separation has to be measured: a run whose
            // crossings are all cached reads nothing from the dataset.
            distances = [&sampler, &datasetSeconds, &samplerMutex, open, samplerOptions,
                         manifest = section.at("manifest")](
                            const std::vector<std::pair<cv::Vec3d, cv::Vec3d>>& connectors, double step, int workers) {
                {
                    const std::lock_guard<std::mutex> guard(samplerMutex);
                    if (!sampler) {
                        const auto opened = Clock::now();
                        auto dataset = vc::lasagna::LasagnaDataset::openLocation(manifest, open);
                        auto fresh = std::make_unique<vc::lasagna::LasagnaNormalSampler>(dataset, samplerOptions);
                        if (!fresh->hasWindingDensity())
                            throw std::invalid_argument("winding.manifest states no grad_mag winding density: it "
                                                        "cannot measure a winding separation");
                        sampler = std::move(fresh);
                        datasetSeconds = seconds(opened);
                    }
                }
                return sampler->normalAlignedWindingDistancesBatch(connectors, step, workers);
            };
        }
        // Variant B: the sheet axis the fiber-inference prediction states at
        // every vertex, weighted by its presence, pulls the strip normals.
        // The annotation's anchors still win at the nodes they fix.
        ScaffoldAxisSampler axisSampler;
        vc::fiber_graph::ScaffoldNeighbourSource neighbourSource;
        const auto normalSource = cfg.value("normals", Json::object()).value("source", std::string{"geometry"});
        std::shared_ptr<vc::lasagna::LasagnaDataset> predictionDataset, sheetDataset;
        std::shared_ptr<vc::lasagna::LasagnaNormalSampler> sheetSampler;
        std::shared_ptr<vc::lasagna::LasagnaGroupArrays> predictionArrays;
        if (normalSource == "fused") {
            const auto section = cfg.value("prediction", Json::object());
            if (!section.contains("manifest"))
                throw std::invalid_argument("normals.source fused needs prediction.manifest");
            vc::lasagna::LasagnaDatasetOpenOptions open;
            open.workingToBaseScale = 1.;
            if (section.contains("remote_cache")) open.remoteCacheRoot = section.at("remote_cache").get<std::string>();
            predictionDataset = std::make_shared<vc::lasagna::LasagnaDataset>(
                vc::lasagna::LasagnaDataset::openLocation(section.at("manifest"), open));
            normalOptions->predictionIdentity = section.at("manifest").get<std::string>();
            sheetDataset = std::make_shared<vc::lasagna::LasagnaDataset>(
                vc::lasagna::fiberSheetNormalDataset(*predictionDataset));
            vc::lasagna::LasagnaNormalSamplerOptions samplerOptions;
            samplerOptions.maxCachedBytes = section.value("cache_bytes", samplerOptions.maxCachedBytes);
            samplerOptions.readerThreads = section.value("reader_threads", std::size_t{0});
            sheetSampler = std::make_shared<vc::lasagna::LasagnaNormalSampler>(*sheetDataset, samplerOptions);
            predictionArrays = std::make_shared<vc::lasagna::LasagnaGroupArrays>(samplerOptions.maxCachedBytes, true);
            const auto presence =
                vc::lasagna::bindLasagnaChannel(predictionDataset->manifest(), "presence", *predictionArrays);
            axisSampler = [sheetSampler, presence](std::span<const vc::geometry::Point> points,
                                                   std::size_t workers) {
                std::vector<cv::Vec3d> xyz;
                std::vector<cv::Vec3f> xyzf;
                xyz.reserve(points.size());
                xyzf.reserve(points.size());
                for (const auto& point : points) {
                    xyz.emplace_back(point[0], point[1], point[2]);
                    xyzf.emplace_back(float(point[0]), float(point[1]), float(point[2]));
                }
                std::vector<vc::lasagna::NormalSampleWithDerivative> axes;
                (void)sheetSampler->sampleNormalBatch(xyz, false, int(workers), axes);
                std::vector<std::vector<float>> presenceValues;
                vc::lasagna::sampleLasagnaVectorBatch({presence}, xyzf, presenceValues, false);
                std::vector<ScaffoldPredictedAxis> out(points.size());
                for (std::size_t i = 0; i < points.size(); ++i) {
                    if (i >= axes.size() || !axes[i].sample.valid) continue;
                    if (i >= presenceValues.size() || presenceValues[i].empty()) continue;
                    const auto& normal = axes[i].sample.normal;
                    out[i] = {{normal[0], normal[1], normal[2]}, double(presenceValues[i][0]), true};
                }
                return out;
            };
        } else if (normalSource == "neighbours") {
            // The signed normals of a resolved fiber volume, around every fiber.
            const auto section = cfg.value("neighbours", Json::object());
            if (!section.contains("samples"))
                throw std::invalid_argument("normals.source neighbours needs neighbours.samples");
            auto& neighbour = normalOptions->neighbours;
            neighbour.weight = section.value("weight", neighbour.weight);
            neighbour.sigma = section.value("sigma", neighbour.sigma);
            neighbour.minimumCoverage = section.value("minimum_coverage", neighbour.minimumCoverage);
            neighbour.fieldCrossings = section.value("field_crossings", neighbour.fieldCrossings);
            neighbour.scaffoldCrossings = section.value("scaffold_crossings", neighbour.scaffoldCrossings);
            neighbour.crossing.weight = section.value("crossing_weight", neighbour.crossing.weight);
            neighbourSource = neighbourOpening.get();
            neighbour.field = neighbourSource.field.get();
            neighbour.identity = neighbourSource.identity;
            neighbour.volume = neighbourSource.volume;
        } else if (normalSource != "geometry") {
            throw std::invalid_argument("normals.source must be geometry, fused or neighbours");
        }
        auto started = Clock::now();
        normals.emplace(graph, geometry, *normalOptions, axisSampler);
        normalSeconds = seconds(started);
        started = Clock::now();
        events = scaffoldEvents(graph, geometry, *normals, *eventOptions, distances, eventReport);
        eventSeconds = seconds(started);
        started = Clock::now();
        facingReport = vc::fiber_graph::scaffoldFaceApprovedNormals(graph, geometry, *normals, events, *normalOptions,
                                                                    *eventOptions, *signOptions, eventReport);
        signs = scaffoldNormalSigns(graph, *normals, events, *signOptions);
        signSeconds = seconds(started);
        started = Clock::now();
        const auto stated = scaffoldCrossingApprovals(graph, geometry, events, constraintApprovals, approvalReport);
        measured = scaffoldMeasuredConstraints(graph, geometry, events, signs.sign, *measuredOptions, measuredReport,
                                               stated, &crossings);
        measuredSeconds = seconds(started);
    }
    const auto windingStarted = Clock::now();
    const auto result = solveScaffoldWinding(graph, geometry, options, measured);
    windingSeconds = seconds(windingStarted);
    // field_comparison: this scaffold solve against the winding solve of a resolved fiber volume,
    // through the crossings between the two (vc/fiber_graph/ScaffoldFieldComparison.hpp).
    //   samples       the volume's signed strip samples (its fibers and their signed normals)
    //   winding_store the volume's winding result store (families and solved windings)
    //   winding_run   optional run UUID when the store holds several
    //   weights       optional perpendicular / parallel / sign weights of the cross constraints
    //                 (default 1 / 0 / 3)
    //   output        the constraints as doubles, 11 per row: scaffold fiber, volume fiber, kind,
    //                 target, weight, winding distance, placed, step, residual, arc on the scaffold
    //                 fiber, arc on the volume fiber; `<output>.json` names the fibers
    Json fieldComparison;
    if (cfg.contains("field_comparison")) {
        if (!measure || !normals)
            throw std::invalid_argument("field_comparison needs constraints.enabled: it uses the run's normals and signs");
        const auto section = cfg.at("field_comparison");
        const auto started = Clock::now();
        auto file = vc::fiber_tracer::readFiberStripSampleFile(section.at("samples").get<std::string>());
        vc::objects::OpenOptions storeOptions;
        storeOptions.readOnly = true;
        auto store = vc::objects::ObjectCollection::open(section.at("winding_store").get<std::string>(), storeOptions);
        const auto txn = store.read();
        const std::string run = vc::objects::selectFiberGraphRun(store, txn, "winding",
            section.contains("winding_run") ? std::optional<std::string>(section.at("winding_run").get<std::string>())
                                            : std::nullopt);
        const auto solved = vc::objects::readFiberGraphResultColumns(store, txn, run);
        // Columns: family, enabled, normal_sign, selected_winding, component.
        std::map<vc::objects::FiberGraphUuid, std::size_t> rowOf;
        for (std::size_t r = 0; r < solved.fiberUuids.size(); ++r) rowOf[solved.fiberUuids[r]] = r;
        ScaffoldFieldVolume volume;
        const auto fibers = file.ids.size();
        volume.uuids.resize(fibers);
        volume.family.assign(fibers, -1);
        volume.winding.assign(fibers, 0);
        volume.component.assign(fibers, -1);
        volume.solved.assign(fibers, 0);
        std::int64_t unknown = 0;
        for (std::size_t f = 0; f < fibers; ++f) {
            volume.uuids[f] = vc::objects::fiberGraphUuid(file.ids[f]);
            const auto found = rowOf.find(file.ids[f]);
            if (found == rowOf.end()) { ++unknown; continue; }
            const auto& values = solved.integers[found->second];
            const auto& valid = solved.valid[found->second];
            if (valid[0] && valid[1] && values[1] != 0) volume.family[f] = int(values[0]);
            if (valid[3] && valid[4]) {
                volume.winding[f] = values[3];
                volume.component[f] = values[4];
                volume.solved[f] = 1;
            }
        }
        volume.samples = std::move(file.samples);
        ScaffoldFieldComparisonOptions comparisonOptions;
        comparisonOptions.events = *eventOptions;
        // The cross constraints are weighed on their own, whatever the scaffold solve itself used
        // (a links-only solve has no measured weights): the measured defaults, perpendicular 1 and
        // sign 3, or field_comparison.weights; the crossing evidence and filter are the run's.
        comparisonOptions.measured = *measuredOptions;
        {
            const ScaffoldMeasuredOptions defaults;
            const auto weights = section.value("weights", Json::object());
            comparisonOptions.measured.perpendicularWeight = weights.value("perpendicular", defaults.perpendicularWeight);
            comparisonOptions.measured.parallelWeight = weights.value("parallel", defaults.parallelWeight);
            comparisonOptions.measured.signWeight = weights.value("sign", defaults.signWeight);
            comparisonOptions.measured.enclosureWeight = 0;
        }
        const auto comparison = compareScaffoldWithField(graph, *normals, signs.sign, result.winding, result.component,
                                                         volume, comparisonOptions, distances);
        const std::string output = section.at("output").get<std::string>();
        std::vector<double> rows;
        rows.reserve(comparison.constraints.size() * 11);
        struct Tally { std::int64_t count = 0, placed = 0, met = 0, offOne = 0, offMore = 0; double weight = 0, metWeight = 0; };
        std::map<std::string, Tally> tallies;
        for (const auto& row : comparison.constraints) {
            rows.insert(rows.end(), {double(row.scaffold), double(row.field), double(int(row.kind)), double(row.target),
                                     row.weight, row.windingDistance, double(row.placed), double(row.step), row.residual,
                                     row.arcScaffold, row.arcField});
            const bool sign = row.kind == ScaffoldConstraintKind::SignAtLeast || row.kind == ScaffoldConstraintKind::SignAtMost;
            auto& tally = tallies[sign ? "sign" : row.kind == ScaffoldConstraintKind::Parallel ? "parallel" : "perpendicular"];
            ++tally.count;
            if (!row.placed) continue;
            ++tally.placed;
            tally.weight += row.weight;
            if (row.residual == 0) { ++tally.met; tally.metWeight += row.weight; }
            else if (row.residual <= 1) ++tally.offOne;
            else ++tally.offMore;
        }
        std::ofstream(output, std::ios::binary)
            .write(reinterpret_cast<const char*>(rows.data()), std::streamsize(rows.size() * sizeof(double)));
        Json kinds = Json::object();
        for (const auto& [name, tally] : tallies)
            kinds[name] = Json{{"constraints", tally.count}, {"placed", tally.placed}, {"met", tally.met},
                               {"off_by_one", tally.offOne}, {"off_by_more", tally.offMore},
                               {"weight", tally.weight}, {"met_weight", tally.metWeight}};
        Json offsets = Json::array();
        for (const auto& offset : comparison.offsets)
            offsets.push_back(Json{{"scaffold_component", offset.scaffoldComponent}, {"volume_component", offset.fieldComponent},
                                   {"offset", offset.offset}, {"equations", offset.equations}, {"weight", offset.weight},
                                   {"agreeing_weight", offset.agreeingWeight}});
        fieldComparison = Json{{"winding_run", run}, {"volume_fibers", std::int64_t(fibers)},
                               {"volume_fibers_searched", comparison.fieldFibers}, {"volume_fibers_without_result", unknown},
                               {"crossings", comparison.events.events}, {"perpendicular", comparison.events.perpendicular},
                               {"parallel", comparison.events.parallel}, {"crossing_filtered", comparison.measured.crossingFiltered},
                               {"constraints", std::int64_t(comparison.constraints.size())}, {"kinds", kinds},
                               {"offsets", offsets}, {"output", output}, {"columns", 11},
                               {"seconds", seconds(started)}};
        Json names{{"scaffold", Json::array()}, {"volume", volume.uuids}, {"summary", fieldComparison}};
        for (const auto& fiber : graph.fibers) names["scaffold"].push_back(fiber.uuid);
        std::ofstream(output + ".json") << names.dump() << "\n";
    }
    std::vector<ScaffoldNormalFlip> flips;
    ScaffoldNormalFlipReport flipReport;
    const auto flipsStarted = Clock::now();
    if (normals) flips = scaffoldLinkNormalFlips(graph, geometry, *normals, signs.sign, result.constraints, flipReport);
    const double flipSeconds = seconds(flipsStarted);
    // The stretches to approve, for the report: derived from this solve and the approvals read
    // with the graph. They are not stored with the run.
    ScaffoldStretchReport stretchReport;
    const auto stretchesStarted = Clock::now();
    const auto stretches = scaffoldStretches(graph, geometry, result, {}, stretchReport);
    const double stretchSeconds = seconds(stretchesStarted);
    const auto reportStarted = Clock::now();

    Json summary{
        {"store", cfg.at("store")},
        {"volume", graph.volume},
        {"handedness", graph.handedness},
        {"input", Json{{"fiber_records", graph.report.fiberRecords},
                       {"fibers", graph.report.fibers},
                       {"untagged", graph.report.untagged},
                       {"manual_family", graph.report.manualFamily},
                       {"automatic_family", graph.report.automaticFamily},
                       {"filtered_family", graph.report.filteredFamily},
                       {"filtered_subset", graph.report.filteredSubset},
                       {"short_fibers", graph.report.shortFibers},
                       {"links", graph.report.links},
                       {"approval_layers", graph.report.approvalLayers},
                       {"approved_stretches", graph.report.approvedStretches},
                       {"broken_stretches", graph.report.brokenStretches},
                       {"stale_approval_items", graph.report.staleApprovalItems},
                       {"line_points", std::int64_t(graph.linePoints.size() / 3)},
                       {"umbilicus_points", std::int64_t(graph.umbilicus.size())},
                       {"commit_sequence", graph.source.commitSequence},
                       {"annotation_commit", annotationCommit}}},
        {"map_links", linkReportJson(result.linkReport)},
        {"solve", Json{{"constraints", std::int64_t(result.constraints.size())},
                       {"arcs", std::int64_t(result.arcs.size())},
                       {"components", result.components},
                       {"isolated_fibers", result.isolatedFibers},
                       {"continuation_links", result.continuationLinks},
                       {"continuation_links_broken", result.brokenContinuations},
                       {"weighted_l1_turns", result.objective},
                       {"max_residual_turns", result.maxResidual},
                       {"split_fibers", result.splitFibers},
                       {"split_parts", std::int64_t(result.parts.size())},
                       {"violated_constraints", result.violatedConstraints},
                       {"violated_arcs", result.violatedArcs},
                       {"solve_seconds", result.seconds}}},
        {"quantization", Json{{"scale", result.quantization.scale},
                              {"original_arcs", std::int64_t(result.quantization.originalArcs)},
                              {"retained_arcs", std::int64_t(result.quantization.retainedArcs)},
                              {"dropped_arcs", std::int64_t(result.quantization.droppedArcs)},
                              {"original_weight", result.quantization.originalWeight},
                              {"dropped_weight", result.quantization.droppedWeight},
                              {"capacity_absolute_error", result.quantization.capacityAbsoluteError}}},
        {"constraint_kinds", Json{{"from_map_links", result.linkConstraints},
                                  {"perpendicular",
                                   result.constraintsByKind[std::size_t(ScaffoldConstraintKind::Perpendicular)]},
                                  {"parallel", result.constraintsByKind[std::size_t(ScaffoldConstraintKind::Parallel)]},
                                  {"enclosure",
                                   result.constraintsByKind[std::size_t(ScaffoldConstraintKind::Enclosure)]},
                                  {"sign_at_least",
                                   result.constraintsByKind[std::size_t(ScaffoldConstraintKind::SignAtLeast)]},
                                  {"sign_at_most",
                                   result.constraintsByKind[std::size_t(ScaffoldConstraintKind::SignAtMost)]}}},
        {"timings", Json{{"store_read_seconds", readSeconds},
                         {"store_open_seconds", openSeconds},
                         {"graph_read_seconds", graphSeconds},
                         {"approvals_read_seconds", approvalSeconds},
                         {"cache_read_seconds", cacheSeconds},
                         {"neighbour_open_seconds", neighbourSeconds},
                         {"winding_dataset_open_seconds", datasetSeconds},
                         {"winding_seconds", windingSeconds},
                         {"geometry_seconds", geometrySeconds},
                         {"normals_seconds", normalSeconds},
                         {"events_seconds", eventSeconds},
                         {"sign_seconds", signSeconds},
                         {"measured_seconds", measuredSeconds},
                         {"solve_seconds", result.seconds}}},
        {"settings", Json{{"approval_states", options.links.approvalStates},
                          {"include_auto", options.links.includeAuto},
                          {"max_anchor_distance", options.links.maxAnchorDistance},
                          {"manual_weight", options.links.weight},
                          {"lemon_backend", vc::geometry::windingFlowBackendName(options.flow.backend)},
                          {"lemon_capacity_scale", options.flow.capacityScale},
                          {"continuation_links", options.continuation.enabled},
                          {"continuation_weight", options.continuation.weight},
                          {"continuation_interval_vx", options.continuation.intervalVx},
                          {"max_alternative_paths", options.maxAlternativePaths},
                          {"constraints_enabled", measure}}}};

    if (measure) {
        const auto& normalReport = normals->report();
        summary["normals"] = Json{{"reused", normalReport.reused},
                                  {"cached", normalReport.cached},
                                  {"fibers", normalReport.fibers},
                                  {"solved", normalReport.solved},
                                  {"degenerate", normalReport.degenerate},
                                  {"converged", normalReport.converged},
                                  {"anchored_fibers", normalReport.anchoredFibers},
                                  {"unanchored_fibers", normalReport.unanchoredFibers},
                                  {"manual_anchors", normalReport.manualAnchors},
                                  {"approved_piece_anchors", normalReport.approvedPieceAnchors},
                                  {"approved_normal_points", normalReport.approvedNormalPoints},
                                  {"stale_normal_approvals", normalReport.staleNormalApprovals},
                                  {"adjacent_link_anchors", normalReport.adjacentLinkAnchors},
                                  {"degenerate_anchors", normalReport.degenerateAnchors},
                                  {"skipped_links", normalReport.skippedLinks},
                                  {"source", cfg.value("normals", Json::object()).value("source",
                                                                                        std::string{"geometry"})},
                                  {"predicted_vertices", normalReport.predictedVertices},
                                  {"vertices_with_axis", normalReport.verticesWithAxis},
                                  {"chain_nodes", normalReport.chainNodes},
                                  {"nodes_with_axis", normalReport.nodesWithAxis},
                                  {"mean_axis_weight", normalReport.meanAxisWeight},
                                  {"prediction_seconds", normalReport.predictionSeconds},
                                  {"neighbour_samples", normalReport.neighbourSamples},
                                  {"neighbour_samples_answered", normalReport.neighbourSamplesAnswered},
                                  {"neighbour_nodes", normalReport.neighbourNodes},
                                  {"neighbour_fibers_covered", normalReport.neighbourCoveredFibers},
                                  {"neighbour_fibers_faced", normalReport.neighbourFacedFibers},
                                  {"crossings_field", normalReport.fieldCrossings},
                                  {"crossings_scaffold", normalReport.scaffoldCrossings},
                                  {"crossed_nodes", normalReport.crossedNodes},
                                  {"neighbour_seconds", normalReport.neighbourSeconds},
                                  {"seconds", normalReport.seconds}};
        summary["events"] = Json{{"reused_pairs", eventReport.reusedPairs},
                                 {"reused_events", eventReport.reusedEvents},
                                 {"cached_pairs", eventReport.cachedPairs},
                                 {"lines", eventReport.lines},
                                 {"short_lines", eventReport.shortLines},
                                 {"resampled_points", eventReport.resampledPoints},
                                 {"events", eventReport.events},
                                 {"perpendicular", eventReport.perpendicular},
                                 {"parallel", eventReport.parallel},
                                 {"without_normals", eventReport.withoutNormals},
                                 {"with_winding_distance", eventReport.withDistance},
                                 {"rejected_tangents", eventReport.rejectedTangents},
                                 {"rejected_gap", eventReport.rejectedGap},
                                 {"rejected_winding", eventReport.rejectedWinding},
                                 {"rejected_winding_cutoff", eventReport.rejectedWindingCutoff},
                                 {"seconds", eventReport.seconds}};
        summary["sign"] = Json{{"events", signs.events},
                               {"fixed", signs.fixed},
                               {"decided", signs.decided},
                               {"flipped", signs.flipped},
                               {"undecided", signs.undecided},
                               {"fixed_disagreements", signs.fixedDisagreements},
                               {"solved", signs.solved}};
        if (signs.solved)
            summary["sign"]["solve"] =
                Json{{"objective", signs.solve.objective},
                     {"exact_wrapped_l1_objective", signs.solve.exactObjective},
                     {"iterations", signs.solve.iterations},
                     {"converged", signs.solve.converged},
                     {"linear_solver", signs.solve.linearSolver},
                     {"seconds", signs.solve.seconds},
                     {"weighted_disagreement_all_percent", signs.solve.metrics.weightedDisagreementPercent},
                     {"strict_events", signs.solve.metrics.strictEvents},
                     {"strict_disagreement_percent", signs.solve.metrics.strictDisagreementPercent},
                     {"outward_agreement", signs.solve.metrics.outwardAgreement}};
        summary["measured"] = Json{{"events", measuredReport.events},
                                   {"used", measuredReport.used},
                                   {"expanded_arcs", measuredReport.expanded},
                                   {"fractional_separations", measuredReport.fractional},
                                   {"perpendicular", measuredReport.perpendicular},
                                   {"parallel", measuredReport.parallel},
                                   {"without_distance", measuredReport.withoutDistance},
                                   {"skipped_invalid", measuredReport.skippedInvalid},
                                   {"skipped_untagged", measuredReport.skippedUntagged},
                                   {"skipped_self", measuredReport.skippedSelf},
                                   {"skipped_family_mismatch", measuredReport.skippedFamilyMismatch},
                                   {"sign_constraints", measuredReport.signConstraints},
                                   {"enclosure_pairs", measuredReport.enclosurePairs},
                                   {"enclosure_arcs", measuredReport.enclosureArcs},
                                   {"enclosure_one_sided", measuredReport.enclosureOneSided},
                                   {"enclosure_contradictory", measuredReport.enclosureContradictory},
                                   {"enclosure_nonunique", measuredReport.enclosureNonunique}};
        summary["constraint_approvals"] = Json{{"rows", approvalReport.rows},
                                               {"live", approvalReport.live},
                                               {"approved", approvalReport.approved},
                                               {"unmatched", approvalReport.unmatched},
                                               {"stale", approvalReport.stale},
                                               {"direct", measuredReport.statedDirect},
                                               {"distant_kept", measuredReport.statedDistantKept},
                                               {"distant_dropped", measuredReport.statedDistantDropped},
                                               {"without_sign", measuredReport.statedWithoutSign}};
        summary["approved_normals"] = Json{{"stretches", facingReport.stretches},
                                           {"objects", facingReport.objects},
                                           {"crossings", facingReport.crossings},
                                           {"objects_faced_by_neighbours", facingReport.facedByNeighbours},
                                           {"neighbour_samples", facingReport.samples},
                                           {"weakest_agreement_share", facingReport.weakestAgreement},
                                           {"faced_fibers", normals->report().facedFibers},
                                           {"gaps_unrolled", normals->report().unrolledGaps},
                                           {"sign_constraints_on_approved_normals",
                                            measuredReport.signOnApprovedNormals}};
    }

    if (persist) {
        const auto persistStarted = Clock::now();
        ScaffoldRunRecord record;
        record.runUuid = vc::objects::util::newUuid();
        record.createdAtUnix = unixSecondsNow();
        record.input = {graph.source.storeUuid, graph.source.generation, graph.source.commitSequence};
        ScaffoldRunStages stages;
        if (measure) {
            stages.normalOptions = &*normalOptions;
            stages.normals = &normals->report();
            stages.normalRows = normals->cacheRows();
            stages.eventOptions = &*eventOptions;
            stages.events = &eventReport;
            stages.measuredOptions = &*measuredOptions;
            stages.measured = &measuredReport;
            stages.signOptions = &*signOptions;
            stages.signs = &signs;
            stages.approvals = &approvalReport;
            stages.facing = &facingReport;
            stages.crossings = &crossings;
            stages.flips = &flips;
            stages.flipReport = &flipReport;
        }
        const auto rows = scaffoldSolveRunRows(graph, result, options, record, stages);
        auto txn = store.write();
        for (const auto& table : rows.tables) store.createExtensionTable(txn, table);
        for (const auto& [table, values] : rows.rows) store.appendExtensionRows(txn, table, values);
        if (!cacheSink.normals.empty()) {
            std::vector<vc::objects::ExtensionRow> values;
            for (const auto& row : cacheSink.normals) values.push_back(vc::objects::fiberScaffoldNormalsRow(row));
            store.createExtensionTable(txn, vc::objects::fiberScaffoldNormalsTable());
            store.appendExtensionRows(txn, vc::objects::kFiberScaffoldNormalsTable, values);
        }
        if (!cacheSink.crossings.empty()) {
            std::vector<vc::objects::ExtensionRow> values;
            for (const auto& row : cacheSink.crossings) values.push_back(vc::objects::fiberScaffoldCrossingsRow(row));
            store.createExtensionTable(txn, vc::objects::fiberScaffoldCrossingsTable());
            store.appendExtensionRows(txn, vc::objects::kFiberScaffoldCrossingsTable, values);
        }
        txn.commit();
        summary["run_uuid"] = record.runUuid;
        persistSeconds = seconds(persistStarted);
    }

    Json residuals = Json::array();
    for (const auto& entry : result.residuals) {
        Json value{{"constraint", constraintJson(graph, result.constraints[std::size_t(entry.constraint)])},
                   {"residual_turns", entry.residual},
                   {"cost", entry.cost},
                   {"cycle_hops", entry.cycleHops},
                   {"cycle_piece_length_vx", entry.cyclePieceLength},
                   {"redundancy", entry.redundancy},
                   {"joint", entry.joint},
                   {"pieces", Json::array()}};
        for (const auto& piece : entry.pieces) value["pieces"].push_back(pieceJson(graph, piece));
        residuals.push_back(std::move(value));
    }
    summary["residuals"] = residuals;
    if (!fieldComparison.is_null()) summary["field_comparison"] = fieldComparison;
    Json pieces = Json::array();
    for (const auto& score : result.pieces) {
        Json value = pieceJson(graph, score.piece);
        value["score"] = score.score;
        value["cycle_hops"] = score.cycleHops;
        value["constraints"] = score.constraints;
        pieces.push_back(std::move(value));
    }
    summary["pieces"] = pieces;

    if (normals) {
        Json rows = Json::array();
        for (const auto& flip : flips) {
            const auto& constraint = result.constraints[std::size_t(flip.constraint)];
            rows.push_back({{"fiber", graph.fibers[std::size_t(flip.fiber)].uuid},
                            {"other", graph.fibers[std::size_t(flip.other)].uuid},
                            {"link", graph.links[std::size_t(constraint.link)].uuid},
                            {"arc", flip.arc},
                            {"agreeing_links", flip.agreeing},
                            {"disagreeing_links", flip.disagreeing},
                            {"bracketed", flip.bracketed},
                            {"arc_from", flip.piece.arcFrom},
                            {"arc_to", flip.piece.arcTo},
                            {"control_from", flip.piece.controlFrom},
                            {"control_to", flip.piece.controlTo}});
        }
        Json suspectRows = Json::array();
        for (const auto& suspect : stretches)
            suspectRows.push_back({{"fiber", graph.fibers[std::size_t(suspect.fiber)].uuid},
                                   {"other", graph.fibers[std::size_t(suspect.other)].uuid},
                                   {"arc", suspect.arc},
                                   {"cost", suspect.cost},
                                   {"violations", suspect.violations},
                                   {"crossings", suspect.crossings},
                                   {"arc_from", suspect.piece.arcFrom},
                                   {"arc_to", suspect.piece.arcTo},
                                   {"control_from", suspect.piece.controlFrom},
                                   {"control_to", suspect.piece.controlTo}});
        summary["stretches"] = Json{{"violations", stretchReport.violations},
                                        {"without_crossing", stretchReport.withoutCrossing},
                                        {"suspects", std::move(suspectRows)}};
        summary["normal_flips"] = Json{{"links_checked", flipReport.checked},
                                       {"agreed", flipReport.agreed},
                                       {"disagreed", flipReport.disagreed},
                                       {"undetermined", flipReport.undetermined},
                                       {"opposed_normals", flipReport.opposed},
                                       {"suspects", std::move(rows)}};
    }
    // The approval layers the run read, as arc ranges of their fibers.
    {
        Json rows = Json::array();
        for (std::size_t i = 0; i < graph.fibers.size(); ++i) {
            const auto& approval = graph.fibers[i].approval;
            const auto add = [&](const ScaffoldApproval::Ranges& ranges, const char* aspect, const char* state) {
                for (const auto& [from, to] : ranges)
                    rows.push_back({{"fiber", graph.fibers[i].uuid}, {"aspect", aspect}, {"state", state},
                                    {"arc_from", geometry.arcAtLineIndex(i, from)},
                                    {"arc_to", geometry.arcAtLineIndex(i, to)}});
            };
            add(approval.approvedGeometry, "geometry", "approved");
            add(approval.approvedNormals, "normals", "approved");
            add(approval.brokenGeometry, "geometry", "broken");
            add(approval.brokenNormals, "normals", "broken");
        }
        summary["approval_stretches"] = std::move(rows);
    }
    const double reportSeconds = seconds(reportStarted) - persistSeconds;
    const auto filesStarted = Clock::now();
    if (cfg.contains("windings_output")) {
        Json windings{{"ids", Json::array()}, {"family", Json::array()}, {"selected_winding", Json::array()},
                      {"component", Json::array()}};
        if (measure) windings["normal_sign"] = Json::array();
        for (std::size_t i = 0; i < graph.fibers.size(); ++i) {
            windings["ids"].push_back(graph.fibers[i].uuid);
            windings["family"].push_back(int(graph.fibers[i].family));
            windings["selected_winding"].push_back(result.winding[i]);
            windings["component"].push_back(result.component[i]);
            if (measure) windings["normal_sign"].push_back(signs.sign[i]);
        }
        writeJson(cfg.at("windings_output").get<std::string>(), windings);
    }
    // layout_output: the map of the solve as the fiber map lays it out - per fiber part its line
    // sampled every `layout_spacing_vx` (default 64) as u = winding + turns along the fiber + the
    // fraction of a turn around the umbilicus, v = height.
    if (cfg.contains("layout_output")) {
        const double spacing = cfg.value("layout_spacing_vx", 64.0);
        if (!(spacing > 0)) throw std::invalid_argument("layout_spacing_vx must be positive");
        Json rows = Json::array();
        const auto handedness = double(geometry.seam().handedness());
        for (std::size_t i = 0; i < graph.fibers.size(); ++i) {
            const double length = geometry.arcLength(i);
            // The fiber's parts, or the whole fiber with its one winding.
            std::vector<vc::fiber_graph::ScaffoldSolveResult::Part> parts;
            for (const auto& part : result.parts)
                if (part.fiber == std::int64_t(i)) parts.push_back(part);
            if (parts.empty())
                parts.push_back({std::int64_t(i), 0., length, result.winding[i], result.component[i], true, false});
            for (const auto& part : parts) {
                Json u = Json::array(), v = Json::array();
                for (double arc = part.arcFrom;; arc += spacing) {
                    const auto sample = geometry.packed().sample(std::int64_t(i), std::min(arc, part.arcTo));
                    u.push_back(double(part.winding + sample.turn) + handedness * geometry.seam().phase(sample.point));
                    v.push_back(sample.point[2]);
                    if (arc >= part.arcTo) break;
                }
                rows.push_back(Json{{"fiber", graph.fibers[i].uuid}, {"family", int(graph.fibers[i].family)},
                                    {"component", part.component}, {"winding", part.winding},
                                    {"arc_from", part.arcFrom}, {"arc_to", part.arcTo},
                                    {"u", std::move(u)}, {"v", std::move(v)}});
            }
        }
        writeJson(cfg.at("layout_output").get<std::string>(), Json{{"parts", std::move(rows)}});
    }
    summary["timings"]["normal_flips_seconds"] = flipSeconds;
    summary["timings"]["stretches_seconds"] = stretchSeconds;
    summary["timings"]["report_seconds"] = reportSeconds;
    summary["timings"]["extra_outputs_seconds"] = seconds(filesStarted);
    summary["timings"]["persist_seconds"] = persistSeconds;
    summary["timings"]["total_seconds"] = seconds(readStarted);
    if (cfg.contains("output")) writeJson(cfg.at("output").get<std::string>(), summary);

    Json line = summary;
    line.erase("residuals");
    line.erase("pieces");
    std::cout << line.dump() << '\n';
    return 0;
}

}  // namespace

int main(const int argc, const char* const* argv)
{
    if (argc != 2) {
        std::cerr << "usage: vc_scaffold_solve <config.json>\n";
        return 2;
    }
    try {
        return run(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << "vc_scaffold_solve: " << error.what() << '\n';
        return 1;
    }
}
