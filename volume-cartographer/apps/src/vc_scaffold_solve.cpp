// Scaffold winding solve from the command line.
//
// Reads a store's annotated fibers, their links, piece approvals and
// umbilicus, assembles the integer winding arcs, solves them exactly with
// LEMON and writes the residual ranking the approval workflow orders by. The
// same library backs the fiber map's solver dock, so a run here and a run in
// VC3D are the same solve.
#include "vc/fiber_graph/ScaffoldSolveRuns.hpp"
#include "vc/fiber_graph/ScaffoldWindingSolve.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/ObjectStore.hpp>
#include <vc_object_store/objects/StoreReader.hpp>

#include <nlohmann/json.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
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
    {
        const auto txn = store.read();
        const vc::objects::LocalStoreReader reader(store, txn);
        graph = loadScaffoldGraph(reader, loadOptionsFrom(cfg));
    }
    const double readSeconds = seconds(readStarted);

    const auto geometryStarted = Clock::now();
    const ScaffoldGeometry geometry(graph, cfg.value("workers", std::size_t{1}));
    const double geometrySeconds = seconds(geometryStarted);

    const auto options = solveOptionsFrom(cfg);
    const auto result = solveScaffoldWinding(graph, geometry, options);

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
        {"timings", Json{{"store_read_seconds", readSeconds},
                         {"geometry_seconds", geometrySeconds},
                         {"solve_seconds", result.seconds}}},
        {"settings", Json{{"approval_states", options.links.approvalStates},
                          {"include_auto", options.links.includeAuto},
                          {"max_anchor_distance", options.links.maxAnchorDistance},
                          {"manual_weight", options.links.weight},
                          {"lemon_backend", vc::geometry::windingFlowBackendName(options.flow.backend)},
                          {"lemon_capacity_scale", options.flow.capacityScale},
                          {"fix_approved_subgraphs", options.fixApprovedSubgraphs},
                          {"max_alternative_paths", options.maxAlternativePaths}}}};

    if (persist) {
        ScaffoldRunRecord record;
        record.runUuid = vc::objects::util::newUuid();
        record.input = {graph.source.storeUuid, graph.source.generation, graph.source.commitSequence};
        const auto rows = scaffoldSolveRunRows(graph, result, options, record);
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
        for (std::size_t i = 0; i < graph.fibers.size(); ++i) {
            windings["ids"].push_back(graph.fibers[i].uuid);
            windings["family"].push_back(int(graph.fibers[i].family));
            windings["selected_winding"].push_back(result.winding[i]);
            windings["component"].push_back(result.component[i]);
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
