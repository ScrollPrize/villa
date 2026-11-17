// Scaffold winding solve from the command line.
//
// Reads a store's annotated fibers, their links, piece approvals and
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
#include "vc/fiber_graph/ScaffoldConstraints.hpp"
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
#include <fstream>
#include <iostream>
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
    options.perpendicularOffset = section.value("perpendicular_offset", options.perpendicularOffset);
    options.windingScale = section.value("winding_scale", options.windingScale);
    options.enclosures = section.value("enclosures", options.enclosures);
    return options;
}

ScaffoldSignOptions signOptionsFrom(const Json& cfg)
{
    using namespace vc::fiber_tracer;
    const auto section = cfg.value("sign", Json::object());
    ScaffoldSignOptions options;
    options.solve = section.value("solve", options.solve);
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

ScaffoldSolveOptions solveOptionsFrom(const Json& cfg)
{
    ScaffoldSolveOptions options;
    const auto links = cfg.value("links", Json::object());
    if (links.contains("approval_states"))
        options.links.approvalStates = links.at("approval_states").get<std::vector<std::string>>();
    options.links.includeAuto = links.value("include_auto", false);
    options.links.maxAnchorDistance = links.value("max_anchor_distance", 5.0);
    options.links.weight = cfg.value("weights", Json::object()).value("manual", 1.0);
    const auto lemon = cfg.value("lemon", Json::object());
    options.flow.backend = vc::geometry::parseWindingFlowBackend(lemon.value("backend", std::string{"network_simplex"}));
    options.flow.capacityScale = lemon.value("capacity_scale", std::int64_t{1000});
    options.fixApprovedSubgraphs = cfg.value("fix_approved_subgraphs", true);
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
                {"max_anchor_distance_vx", report.maxAnchorDistance}};
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
               {"arc_a_vx", constraint.arcA},
               {"arc_b_vx", constraint.arcB},
               {"step", constraint.step},
               {"target", constraint.target},
               {"weight", constraint.weight},
               {"approved", constraint.approved}};
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
    std::vector<vc::objects::FiberConstraintApproval> constraintApprovals;
    {
        const auto txn = store.read();
        const vc::objects::LocalStoreReader reader(store, txn);
        graph = loadScaffoldGraph(reader, loadOptionsFrom(cfg));
        constraintApprovals = vc::objects::readFiberConstraintApprovals(reader);
    }
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
    std::vector<ScaffoldConstraint> measured;
    double normalSeconds = 0, eventSeconds = 0, signSeconds = 0, measuredSeconds = 0;
    if (measure) {
        normalOptions = normalOptionsFrom(cfg);
        eventOptions = eventOptionsFrom(cfg);
        measuredOptions = measuredOptionsFrom(cfg);
        signOptions = signOptionsFrom(cfg);
        std::unique_ptr<vc::lasagna::LasagnaNormalSampler> sampler;
        ScaffoldWindingDistances distances;
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
            auto dataset = vc::lasagna::LasagnaDataset::openLocation(section.at("manifest"), open);
            vc::lasagna::LasagnaNormalSamplerOptions samplerOptions;
            samplerOptions.maxCachedBytes = section.value("cache_bytes", samplerOptions.maxCachedBytes);
            samplerOptions.readerThreads = section.value("reader_threads", std::size_t{0});
            samplerOptions.windingRegionBaseVoxels =
                section.value("region_base_voxels", samplerOptions.windingRegionBaseVoxels);
            samplerOptions.windingBatchPoints = section.value("batch_points", samplerOptions.windingBatchPoints);
            sampler = std::make_unique<vc::lasagna::LasagnaNormalSampler>(dataset, samplerOptions);
            if (!sampler->hasWindingDensity())
                throw std::invalid_argument("winding.manifest states no grad_mag winding density: it cannot measure a "
                                            "winding separation");
            distances = [&sampler](const std::vector<std::pair<cv::Vec3d, cv::Vec3d>>& connectors, double step,
                                   int workers) {
                return sampler->normalAlignedWindingDistancesBatch(connectors, step, workers);
            };
        }
        // Variant B: the sheet axis the fiber-inference prediction states at
        // every vertex, weighted by its presence, pulls the strip normals.
        // The annotation's anchors still win at the nodes they fix.
        ScaffoldAxisSampler axisSampler;
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
        } else if (normalSource != "geometry") {
            throw std::invalid_argument("normals.source must be geometry or fused");
        }
        auto started = Clock::now();
        normals.emplace(graph, geometry, *normalOptions, axisSampler);
        normalSeconds = seconds(started);
        started = Clock::now();
        events = scaffoldEvents(graph, geometry, *normals, *eventOptions, distances, eventReport);
        eventSeconds = seconds(started);
        started = Clock::now();
        signs = scaffoldNormalSigns(graph, *normals, events, *signOptions);
        signSeconds = seconds(started);
        started = Clock::now();
        measured = scaffoldMeasuredConstraints(graph, geometry, events, signs.sign, *measuredOptions, measuredReport);
        applyScaffoldConstraintApprovals(graph, geometry, measured, constraintApprovals, approvalReport);
        measuredSeconds = seconds(started);
    }
    const auto result = solveScaffoldWinding(graph, geometry, options, measured);

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
                       {"short_fibers", graph.report.shortFibers},
                       {"links", graph.report.links},
                       {"piece_approval_rows", graph.report.pieceApprovalRows},
                       {"line_points", std::int64_t(graph.linePoints.size() / 3)},
                       {"umbilicus_points", std::int64_t(graph.umbilicus.size())}}},
        {"manual_links", linkReportJson(result.linkReport)},
        {"solve", Json{{"constraints", std::int64_t(result.constraints.size())},
                       {"arcs", std::int64_t(result.arcs.size())},
                       {"components", result.components},
                       {"isolated_fibers", result.isolatedFibers},
                       {"approved_subgraphs", result.approvedSubgraphs},
                       {"contracted_fibers", result.contractedFibers},
                       {"weighted_l1_turns", result.objective},
                       {"max_residual_turns", result.maxResidual},
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
        {"constraint_kinds", Json{{"manual", result.constraintsByKind[std::size_t(ScaffoldConstraintKind::Manual)]},
                                  {"perpendicular",
                                   result.constraintsByKind[std::size_t(ScaffoldConstraintKind::Perpendicular)]},
                                  {"parallel", result.constraintsByKind[std::size_t(ScaffoldConstraintKind::Parallel)]},
                                  {"enclosure",
                                   result.constraintsByKind[std::size_t(ScaffoldConstraintKind::Enclosure)]}}},
        {"timings", Json{{"store_read_seconds", readSeconds},
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
                          {"fix_approved_subgraphs", options.fixApprovedSubgraphs},
                          {"max_alternative_paths", options.maxAlternativePaths},
                          {"constraints_enabled", measure}}}};

    if (measure) {
        const auto& normalReport = normals->report();
        summary["normals"] = Json{{"fibers", normalReport.fibers},
                                  {"solved", normalReport.solved},
                                  {"degenerate", normalReport.degenerate},
                                  {"converged", normalReport.converged},
                                  {"anchored_fibers", normalReport.anchoredFibers},
                                  {"unanchored_fibers", normalReport.unanchoredFibers},
                                  {"manual_anchors", normalReport.manualAnchors},
                                  {"approved_piece_anchors", normalReport.approvedPieceAnchors},
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
                                  {"seconds", normalReport.seconds}};
        summary["events"] = Json{{"lines", eventReport.lines},
                                 {"short_lines", eventReport.shortLines},
                                 {"resampled_points", eventReport.resampledPoints},
                                 {"events", eventReport.events},
                                 {"perpendicular", eventReport.perpendicular},
                                 {"parallel", eventReport.parallel},
                                 {"without_normals", eventReport.withoutNormals},
                                 {"with_winding_distance", eventReport.withDistance},
                                 {"rejected_tangents", eventReport.rejectedTangents},
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
                                   {"perpendicular", measuredReport.perpendicular},
                                   {"parallel", measuredReport.parallel},
                                   {"without_distance", measuredReport.withoutDistance},
                                   {"skipped_invalid", measuredReport.skippedInvalid},
                                   {"skipped_untagged", measuredReport.skippedUntagged},
                                   {"skipped_self", measuredReport.skippedSelf},
                                   {"skipped_family_mismatch", measuredReport.skippedFamilyMismatch},
                                   {"enclosure_pairs", measuredReport.enclosurePairs},
                                   {"enclosure_arcs", measuredReport.enclosureArcs},
                                   {"enclosure_one_sided", measuredReport.enclosureOneSided},
                                   {"enclosure_contradictory", measuredReport.enclosureContradictory},
                                   {"enclosure_nonunique", measuredReport.enclosureNonunique}};
        summary["constraint_approvals"] = Json{{"rows", approvalReport.rows},
                                               {"live", approvalReport.live},
                                               {"approved", approvalReport.approved},
                                               {"unmatched", approvalReport.unmatched},
                                               {"stale", approvalReport.stale}};
    }

    if (persist) {
        ScaffoldRunRecord record;
        record.runUuid = vc::objects::util::newUuid();
        record.input = {graph.source.storeUuid, graph.source.generation, graph.source.commitSequence};
        ScaffoldRunStages stages;
        if (measure) {
            stages.normalOptions = &*normalOptions;
            stages.normals = &normals->report();
            stages.eventOptions = &*eventOptions;
            stages.events = &eventReport;
            stages.measuredOptions = &*measuredOptions;
            stages.measured = &measuredReport;
            stages.signOptions = &*signOptions;
            stages.signs = &signs;
        }
        const auto rows = scaffoldSolveRunRows(graph, result, options, record, stages);
        auto txn = store.write();
        for (const auto& table : rows.tables) store.createExtensionTable(txn, table);
        for (const auto& [table, values] : rows.rows) store.appendExtensionRows(txn, table, values);
        txn.commit();
        summary["run_uuid"] = record.runUuid;
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
    Json pieces = Json::array();
    for (const auto& score : result.pieces) {
        Json value = pieceJson(graph, score.piece);
        value["score"] = score.score;
        value["cycle_hops"] = score.cycleHops;
        value["constraints"] = score.constraints;
        pieces.push_back(std::move(value));
    }
    summary["pieces"] = pieces;

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
