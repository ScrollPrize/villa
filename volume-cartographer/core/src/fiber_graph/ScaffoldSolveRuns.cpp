#include "vc/fiber_graph/ScaffoldSolveRuns.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>

namespace vc::fiber_graph {
namespace {
using objects::ExtensionRow;
using objects::ExtensionValue;
using objects::FiberGraphTable;

ExtensionValue uuidValue(const std::string& text) { return objects::extensionUuid(text); }
ExtensionValue nullValue() { return ExtensionValue{std::monostate{}}; }

// Column lookup by name over a table definition, so a reader never depends on
// the declared column order.
struct Columns {
    std::map<std::string, std::size_t> index;
    explicit Columns(FiberGraphTable kind)
    {
        const auto table = objects::fiberGraphSolveTable(kind);
        for (std::size_t i = 0; i < table.columns.size(); ++i) index.emplace(table.columns[i].name, i);
    }
    const ExtensionValue& at(const ExtensionRow& row, const char* name) const { return row.at(index.at(name)); }
    std::int64_t integer(const ExtensionRow& row, const char* name) const
    {
        return std::get<std::int64_t>(at(row, name));
    }
    std::optional<std::int64_t> optionalInteger(const ExtensionRow& row, const char* name) const
    {
        const auto& value = at(row, name);
        if (std::holds_alternative<std::monostate>(value)) return std::nullopt;
        return std::get<std::int64_t>(value);
    }
    double real(const ExtensionRow& row, const char* name) const
    {
        const auto& value = at(row, name);
        return std::holds_alternative<std::monostate>(value) ? 0. : std::get<double>(value);
    }
    std::string uuid(const ExtensionRow& row, const char* name) const
    {
        const auto& value = at(row, name);
        if (std::holds_alternative<std::monostate>(value)) return {};
        return objects::extensionUuid(value);
    }
    std::string text(const ExtensionRow& row, const char* name) const
    {
        const auto& value = at(row, name);
        return std::holds_alternative<std::monostate>(value) ? std::string{} : std::get<std::string>(value);
    }
};

// Immutable rows of one run, in primary-key order. A store without the table
// simply has no rows of that kind.
void scanRun(const objects::StoreReader& reader, FiberGraphTable kind, const std::string& runUuid,
             const std::function<void(const ExtensionRow&)>& row)
{
    const auto table = objects::fiberGraphSolveTable(kind);
    // An empty run UUID scans every run; only a named one keys the prefix.
    std::vector<ExtensionValue> prefix;
    if (!runUuid.empty()) prefix.push_back(uuidValue(runUuid));
    try {
        reader.scanExtensionRows(table.name, [&](const ExtensionRow& source) {
            row(source);
            return true;
        }, prefix);
    } catch (const std::invalid_argument&) {
        // No such table in this store: no runs of this kind were ever written.
    }
}
} // namespace

ScaffoldRunRows scaffoldSolveRunRows(const ScaffoldGraph& graph, const ScaffoldSolveResult& result,
                                     const ScaffoldSolveOptions& options, const ScaffoldRunRecord& record)
{
    if (record.runUuid.empty()) throw std::invalid_argument("scaffold run: a run UUID is required");
    if (result.winding.size() != graph.fibers.size() || result.component.size() != graph.fibers.size())
        throw std::invalid_argument("scaffold run: the result does not cover the loaded fibers");

    const auto backend = geometry::windingFlowBackendName(options.flow.backend);
    const auto states = options.links.approvalStates;
    const auto usesState = [&](const char* name) {
        return std::find(states.begin(), states.end(), name) != states.end();
    };
    std::map<std::string, objects::FiberGraphSetting> settings{
        {"handedness", std::int64_t(graph.handedness)},
        {"weight_manual", options.links.weight},
        {"link_include_auto", options.links.includeAuto},
        {"link_max_anchor_distance", options.links.maxAnchorDistance},
        {"link_state_approved", usesState("approved")},
        {"link_state_pending", usesState("pending")},
        {"link_state_rejected", usesState("rejected")},
        {"link_state_count", std::int64_t(states.size())},
        {"lemon_capacity_scale", options.flow.capacityScale},
        {"fix_approved_subgraphs", options.fixApprovedSubgraphs},
        {"max_alternative_paths", options.maxAlternativePaths}};

    ScaffoldRunRows out;
    const auto add = [&](FiberGraphTable kind, std::vector<ExtensionRow> rows) {
        const auto table = objects::fiberGraphSolveTable(kind);
        out.tables.push_back(table);
        out.rows.emplace_back(table.name, std::move(rows));
    };
    add(FiberGraphTable::Runs,
        {objects::fiberGraphRunRow(record.runUuid, record.stage, "scaffold_lemon_" + backend, settings).values});
    add(FiberGraphTable::Inputs, {objects::fiberGraphInputRow(record.runUuid, "input_0", record.input).values});

    std::vector<ExtensionRow> fibers;
    fibers.reserve(graph.fibers.size());
    for (std::size_t i = 0; i < graph.fibers.size(); ++i) {
        const auto& fiber = graph.fibers[i];
        fibers.push_back(objects::fiberGraphSolveRow(FiberGraphTable::FiberResults,
            {{"fiber_uuid", uuidValue(fiber.uuid)},
             {"run_uuid", uuidValue(record.runUuid)},
             {"family", std::int64_t(fiber.family)},
             {"enabled", std::int64_t(fiber.family == ScaffoldFamily::Untagged ? 0 : 1)},
             {"selected_winding", result.winding[i]},
             {"component", result.component[i]}}).values);
    }
    add(FiberGraphTable::FiberResults, std::move(fibers));

    std::vector<ExtensionRow> metrics;
    const auto metric = [&](const char* name, double value, const char* units) {
        metrics.push_back(objects::fiberGraphSolveRow(FiberGraphTable::Metrics,
            {{"run_uuid", uuidValue(record.runUuid)}, {"iteration", std::int64_t(0)}, {"scope", std::string("all")},
             {"name", std::string(name)}, {"value", value}, {"units", std::string(units)}}).values);
    };
    const auto count = [&](const char* name, std::int64_t value) {
        metrics.push_back(objects::fiberGraphSolveRow(FiberGraphTable::Metrics,
            {{"run_uuid", uuidValue(record.runUuid)}, {"iteration", std::int64_t(0)}, {"scope", std::string("all")},
             {"name", std::string(name)}, {"count", value}, {"units", std::string("fibers")}}).values);
    };
    metric("weighted_l1", result.objective, "weighted_turns");
    metric("max_residual", result.maxResidual, "turns");
    metric("solve_seconds", result.seconds, "seconds");
    count("fibers", std::int64_t(graph.fibers.size()));
    count("arcs", std::int64_t(result.arcs.size()));
    count("constraints", std::int64_t(result.constraints.size()));
    count("components", result.components);
    count("isolated_fibers", result.isolatedFibers);
    count("violated_arcs", result.violatedArcs);
    count("violated_constraints", result.violatedConstraints);
    count("approved_subgraphs", result.approvedSubgraphs);
    add(FiberGraphTable::Metrics, std::move(metrics));

    std::vector<ExtensionRow> residuals, pieces;
    for (std::size_t rank = 0; rank < result.residuals.size(); ++rank) {
        const auto& entry = result.residuals[rank];
        const auto& constraint = result.constraints[std::size_t(entry.constraint)];
        const bool cycle = entry.cycleHops > 0;
        residuals.push_back(objects::fiberGraphSolveRow(FiberGraphTable::Residuals,
            {{"run_uuid", uuidValue(record.runUuid)},
             {"residual_index", std::int64_t(rank)},
             {"kind", std::int64_t(constraint.kind)},
             {"fiber_a", uuidValue(graph.fibers[std::size_t(constraint.a)].uuid)},
             {"fiber_b", uuidValue(graph.fibers[std::size_t(constraint.b)].uuid)},
             {"target", constraint.target},
             {"weight", constraint.weight},
             {"residual", entry.residual},
             {"edge_uuid", constraint.link >= 0 ? uuidValue(graph.links[std::size_t(constraint.link)].uuid) : nullValue()},
             {"arc_a", constraint.arcA},
             {"arc_b", constraint.arcB},
             {"cycle_hops", cycle ? ExtensionValue{entry.cycleHops} : nullValue()},
             {"cycle_piece_length", cycle ? ExtensionValue{entry.cyclePieceLength} : nullValue()},
             {"redundancy", cycle ? ExtensionValue{entry.redundancy} : nullValue()}}).values);
        for (std::size_t member = 0; member < entry.pieces.size(); ++member) {
            const auto& piece = entry.pieces[member];
            pieces.push_back(objects::fiberGraphSolveRow(FiberGraphTable::ResidualPieces,
                {{"run_uuid", uuidValue(record.runUuid)},
                 {"residual_index", std::int64_t(rank)},
                 {"member_index", std::int64_t(member)},
                 {"fiber_uuid", uuidValue(graph.fibers[std::size_t(piece.fiber)].uuid)},
                 {"arc_from", std::min(piece.arcFrom, piece.arcTo)},
                 {"arc_to", std::max(piece.arcFrom, piece.arcTo)},
                 {"control_from", piece.controlFrom >= 0 ? ExtensionValue{piece.controlFrom} : nullValue()},
                 {"control_to", piece.controlTo >= 0 ? ExtensionValue{piece.controlTo} : nullValue()},
                 {"score", nullValue()}}).values);
        }
    }
    // The piece ranking is the per-piece sum over the violated constraints;
    // carry it on the member rows it belongs to.
    const Columns pieceColumns(FiberGraphTable::ResidualPieces);
    for (auto& row : pieces) {
        const auto fiber = pieceColumns.uuid(row, "fiber_uuid");
        const auto from = pieceColumns.real(row, "arc_from"), to = pieceColumns.real(row, "arc_to");
        for (const auto& score : result.pieces) {
            if (graph.fibers[std::size_t(score.piece.fiber)].uuid != fiber) continue;
            if (std::min(score.piece.arcFrom, score.piece.arcTo) != from ||
                std::max(score.piece.arcFrom, score.piece.arcTo) != to)
                continue;
            row[pieceColumns.index.at("score")] = score.score;
            break;
        }
    }
    add(FiberGraphTable::Residuals, std::move(residuals));
    add(FiberGraphTable::ResidualPieces, std::move(pieces));
    return out;
}

std::vector<ScaffoldStoredRun> readScaffoldRuns(const objects::StoreReader& reader, const std::string& stage)
{
    const Columns runs(FiberGraphTable::Runs);
    std::vector<ScaffoldStoredRun> out;
    scanRun(reader, FiberGraphTable::Runs, {}, [&](const ExtensionRow& row) {
        ScaffoldStoredRun entry;
        entry.runUuid = runs.uuid(row, "run_uuid");
        entry.stage = runs.text(row, "stage");
        entry.method = runs.text(row, "method");
        entry.settingsJson = runs.text(row, "settings");
        if (stage.empty() || entry.stage == stage) out.push_back(std::move(entry));
    });
    const Columns metrics(FiberGraphTable::Metrics);
    for (auto& entry : out) {
        scanRun(reader, FiberGraphTable::Metrics, entry.runUuid, [&](const ExtensionRow& row) {
            if (metrics.text(row, "scope") != "all") return;
            const auto name = metrics.text(row, "name");
            const auto& value = metrics.at(row, "value");
            const auto& number = metrics.at(row, "count");
            if (!std::holds_alternative<std::monostate>(value)) entry.metrics[name] = std::get<double>(value);
            if (!std::holds_alternative<std::monostate>(number)) entry.counts[name] = std::get<std::int64_t>(number);
        });
        const auto found = entry.counts.find("fibers");
        if (found != entry.counts.end()) entry.fibers = found->second;
        const Columns residuals(FiberGraphTable::Residuals);
        scanRun(reader, FiberGraphTable::Residuals, entry.runUuid, [&](const ExtensionRow&) { ++entry.residuals; });
    }
    return out;
}

std::vector<ScaffoldStoredResidual> readScaffoldResiduals(const objects::StoreReader& reader,
                                                          const std::string& runUuid)
{
    if (runUuid.empty()) throw std::invalid_argument("scaffold run: a run UUID is required");
    const Columns columns(FiberGraphTable::Residuals);
    std::vector<ScaffoldStoredResidual> out;
    std::map<std::int64_t, std::size_t> byIndex;
    scanRun(reader, FiberGraphTable::Residuals, runUuid, [&](const ExtensionRow& row) {
        ScaffoldStoredResidual entry;
        entry.index = columns.integer(row, "residual_index");
        entry.kind = columns.integer(row, "kind");
        entry.target = columns.integer(row, "target");
        entry.fiberA = columns.uuid(row, "fiber_a");
        entry.fiberB = columns.uuid(row, "fiber_b");
        entry.edgeUuid = columns.uuid(row, "edge_uuid");
        entry.arcA = columns.real(row, "arc_a");
        entry.arcB = columns.real(row, "arc_b");
        entry.weight = columns.real(row, "weight");
        entry.residual = columns.real(row, "residual");
        entry.cyclePieceLength = columns.real(row, "cycle_piece_length");
        entry.cycleHops = columns.optionalInteger(row, "cycle_hops");
        entry.redundancy = columns.optionalInteger(row, "redundancy");
        byIndex.emplace(entry.index, out.size());
        out.push_back(std::move(entry));
    });
    const Columns pieces(FiberGraphTable::ResidualPieces);
    scanRun(reader, FiberGraphTable::ResidualPieces, runUuid, [&](const ExtensionRow& row) {
        const auto found = byIndex.find(pieces.integer(row, "residual_index"));
        if (found == byIndex.end()) return;
        ScaffoldStoredPiece piece;
        piece.fiberUuid = pieces.uuid(row, "fiber_uuid");
        piece.arcFrom = pieces.real(row, "arc_from");
        piece.arcTo = pieces.real(row, "arc_to");
        piece.score = pieces.real(row, "score");
        piece.controlFrom = pieces.optionalInteger(row, "control_from");
        piece.controlTo = pieces.optionalInteger(row, "control_to");
        out[found->second].pieces.push_back(std::move(piece));
    });
    return out;
}

std::map<std::string, ScaffoldStoredWinding> readScaffoldWindings(const objects::StoreReader& reader,
                                                                 const std::string& runUuid)
{
    if (runUuid.empty()) throw std::invalid_argument("scaffold run: a run UUID is required");
    const auto table = objects::fiberGraphSolveTable(FiberGraphTable::FiberResults);
    const Columns columns(FiberGraphTable::FiberResults);
    std::map<std::string, ScaffoldStoredWinding> out;
    // The fiber result table is keyed by fiber first, so a run prefix does
    // not apply: filter the run out of the scan.
    try {
        reader.scanExtensionRows(table.name, [&](const ExtensionRow& row) {
            if (columns.uuid(row, "run_uuid") != runUuid) return true;
            ScaffoldStoredWinding entry;
            const auto winding = columns.optionalInteger(row, "selected_winding");
            const auto component = columns.optionalInteger(row, "component");
            if (!winding) return true;
            entry.winding = *winding;
            entry.component = component.value_or(0);
            entry.family = columns.optionalInteger(row, "family");
            entry.normalSign = columns.optionalInteger(row, "normal_sign");
            out.emplace(columns.uuid(row, "fiber_uuid"), entry);
            return true;
        });
    } catch (const std::invalid_argument&) {
        // No fiber result table: no runs were written to this store.
    }
    return out;
}

} // namespace vc::fiber_graph
