#include "vc/fiber_graph/ScaffoldSolveRuns.hpp"

#include <cstring>
#include <tuple>
#include <vc_object_store/objects/FiberScaffoldCacheExtension.hpp>
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
                                     const ScaffoldSolveOptions& options, const ScaffoldRunRecord& record,
                                     const ScaffoldRunStages& stages)
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
        {"links_used", options.links.useLinks},
        {"weight_manual", options.links.weight},
        {"link_include_auto", options.links.includeAuto},
        {"link_max_anchor_distance", options.links.maxAnchorDistance},
        {"link_state_approved", usesState("approved")},
        {"link_state_pending", usesState("pending")},
        {"link_state_rejected", usesState("rejected")},
        {"link_state_count", std::int64_t(states.size())},
        {"lemon_capacity_scale", options.flow.capacityScale},
        {"continuation_links", options.continuation.enabled},
        {"weight_continuation", options.continuation.weight},
        {"continuation_interval", options.continuation.intervalVx},
        {"max_alternative_paths", options.maxAlternativePaths},
        // A restricted run says so: the fibers its scope left out.
        {"scope_excluded_fibers", graph.report.filteredSubset},
        {"created_at_unix", record.createdAtUnix}};
    // The measured stages, when the run ran them. A run without them is a
    // links-only run and says so by the absence of these keys.
    if (stages.normalOptions) {
        // Which normals the run used: the prediction states a sheet axis only
        // on a fused run, so its own count is the honest answer.
        settings.emplace("normals_fused", stages.normals && stages.normals->predictedVertices > 0);
        settings.emplace("normals_neighbours", stages.normalOptions->neighbours.field != nullptr);
        settings.emplace("normals_twist_weight", stages.normalOptions->strip.twistWeight);
        settings.emplace("normals_lateral_weight", stages.normalOptions->strip.lateralWeight);
        settings.emplace("normals_smoothing_sigma", stages.normalOptions->strip.smoothingSigma);
        settings.emplace("normals_resample_spacing", stages.normalOptions->strip.resampleSpacing);
        settings.emplace("normals_manual_display", stages.normalOptions->manualDisplayNormals);
        settings.emplace("normals_approved_pieces", stages.normalOptions->approvedNormalPieces);
        settings.emplace("normals_adjacent_links", stages.normalOptions->adjacentLinkAnchors);
        settings.emplace("normals_anchor_weight", stages.normalOptions->anchorWeight);
        settings.emplace("normals_prediction_chunk", std::int64_t(stages.normalOptions->predictionChunkPoints));
    }
    if (stages.eventOptions) {
        settings.emplace("events_resample_spacing", stages.eventOptions->resampleSpacing);
        settings.emplace("events_maximum_distance", stages.eventOptions->maximumDistance);
        settings.emplace("events_tangent_window", stages.eventOptions->tangentWindow);
        settings.emplace("events_relax_endpoint_certification", stages.eventOptions->relaxEndpointCertification);
        settings.emplace("events_winding_distances", stages.eventOptions->measureWindingDistances);
        settings.emplace("events_winding_step", stages.eventOptions->windingIntegrationStep);
        settings.emplace("events_maximum_winding_distance", stages.eventOptions->maximumWindingDistance);
        settings.emplace("events_enforce_maximum_winding", stages.eventOptions->enforceMaximumWindingDistance);
    }
    if (stages.measuredOptions) {
        settings.emplace("weight_perpendicular", stages.measuredOptions->perpendicularWeight);
        settings.emplace("weight_parallel", stages.measuredOptions->parallelWeight);
        settings.emplace("weight_enclosure", stages.measuredOptions->enclosureWeight);
        settings.emplace("weight_sign", stages.measuredOptions->signWeight);
        settings.emplace("weight_sign_approved_factor", stages.measuredOptions->approvedSignFactor);
        settings.emplace("weight_reviewed", stages.measuredOptions->reviewedWeight);
        settings.emplace("weight_crossing_confidence", stages.measuredOptions->crossingConfidence);
        settings.emplace("crossing_filter", stages.measuredOptions->crossingFilter);
        settings.emplace("perpendicular_offset", stages.measuredOptions->perpendicularOffset);
        settings.emplace("winding_scale", stages.measuredOptions->windingScale);
        settings.emplace("enclosures", stages.measuredOptions->enclosures);
        settings.emplace("separations_rounded",
                         stages.measuredOptions->separations == ScaffoldSeparationMode::Rounded);
    }
    if (stages.signOptions) {
        settings.emplace("sign_solve", stages.signOptions->solve);
        settings.emplace("sign_backend_ceres",
                         stages.signOptions->sign.backend == fiber_tracer::FiberSignBackend::Ceres);
        settings.emplace("sign_dot_weight_power", stages.signOptions->sign.dotWeightPower);
        settings.emplace("sign_prior_weight", stages.signOptions->sign.priorWeight);
        settings.emplace("sign_fixed_weight", stages.signOptions->sign.fixedSignWeight);
        settings.emplace("sign_broken_penalty", stages.signOptions->sign.brokenPenalty);
        settings.emplace("sign_epsilon", stages.signOptions->sign.epsilon);
        settings.emplace("sign_iterations", std::int64_t(stages.signOptions->sign.iterations));
        settings.emplace("sign_fixed_keep", stages.signOptions->sign.fixedKeep);
        settings.emplace("sign_inverse_square_distance",
                         stages.signOptions->sign.distanceWeight == fiber_tracer::FiberSignDistanceWeight::InverseSquare);
    }

    ScaffoldRunRows out;
    const auto add = [&](FiberGraphTable kind, std::vector<ExtensionRow> rows) {
        const auto table = objects::fiberGraphSolveTable(kind);
        out.tables.push_back(table);
        out.rows.emplace_back(table.name, std::move(rows));
    };
    add(FiberGraphTable::Runs,
        {objects::fiberGraphRunRow(record.runUuid, record.stage, "scaffold_lemon_" + backend, settings).values});
    add(FiberGraphTable::Inputs, {objects::fiberGraphInputRow(record.runUuid, "input_0", record.input).values});
    // The normals the run used, by reference: one row per fiber.
    if (stages.normals) {
        if (stages.normalRows.size() != graph.fibers.size())
            throw std::invalid_argument("scaffold run: the run solved normals but states no normals row per fiber");
        std::vector<ExtensionRow> references;
        references.reserve(graph.fibers.size());
        for (std::size_t i = 0; i < graph.fibers.size(); ++i) {
            const auto& fiber = graph.fibers[i];
            if (stages.normalRows[i].empty() || !fiber.version)
                throw std::runtime_error("scaffold run: the normals of fiber " + fiber.uuid + " (" + fiber.name +
                                         ") have no stored row, so the run cannot reference them");
            references.push_back(objects::fiberScaffoldRunNormalsRow(
                {record.runUuid, fiber.uuid, *fiber.version, stages.normalRows[i]}));
        }
        const auto table = objects::fiberScaffoldRunNormalsTable();
        out.tables.push_back(table);
        out.rows.emplace_back(table.name, std::move(references));
    }

    std::vector<ExtensionRow> fibers;
    fibers.reserve(graph.fibers.size());
    for (std::size_t i = 0; i < graph.fibers.size(); ++i) {
        const auto& fiber = graph.fibers[i];
        fibers.push_back(objects::fiberGraphSolveRow(FiberGraphTable::FiberResults,
            {{"fiber_uuid", uuidValue(fiber.uuid)},
             {"run_uuid", uuidValue(record.runUuid)},
             {"family", std::int64_t(fiber.family)},
             {"enabled", std::int64_t(fiber.family == ScaffoldFamily::Untagged ? 0 : 1)},
             {"normal_sign", stages.signs && stages.signs->sign.size() == graph.fibers.size()
                                 ? ExtensionValue{stages.signs->sign[i]}
                                 : nullValue()},
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
    // Every count states what it counts: the rows are read by people.
    const auto count = [&](const char* name, std::int64_t value, const char* units) {
        metrics.push_back(objects::fiberGraphSolveRow(FiberGraphTable::Metrics,
            {{"run_uuid", uuidValue(record.runUuid)}, {"iteration", std::int64_t(0)}, {"scope", std::string("all")},
             {"name", std::string(name)}, {"count", value}, {"units", std::string(units)}}).values);
    };
    metric("weighted_l1", result.objective, "weighted_turns");
    metric("max_residual", result.maxResidual, "turns");
    metric("solve_seconds", result.seconds, "seconds");
    count("fibers", std::int64_t(graph.fibers.size()), "fibers");
    count("arcs", std::int64_t(result.arcs.size()), "arcs");
    count("constraints", std::int64_t(result.constraints.size()), "constraints");
    count("components", result.components, "components");
    count("isolated_fibers", result.isolatedFibers, "fibers");
    count("violated_arcs", result.violatedArcs, "arcs");
    count("violated_constraints", result.violatedConstraints, "constraints");
    count("continuation_links", result.continuationLinks, "links");
    count("continuation_links_broken", result.brokenContinuations, "links");
    count("approval_layers", graph.report.approvalLayers, "fibers");
    count("approved_stretches", graph.report.approvedStretches, "stretches");
    count("broken_stretches", graph.report.brokenStretches, "stretches");
    count("stale_approval_items", graph.report.staleApprovalItems, "items");
    count("split_fibers", result.splitFibers, "fibers");
    count("split_parts", std::int64_t(result.parts.size()), "parts");
    // The map links, refined onto their encounters: how many assert a
    // relation, how many the refinement rejected and why, and how many
    // measured crossings they replaced.
    count("link_constraints", result.linkConstraints, "constraints");
    count("links_rejected_angle", result.linkReport.skippedAngle, "links");
    count("links_rejected_end_support", result.linkReport.skippedEndSupport, "links");
    count("links_rejected_distance", result.linkReport.skippedDistance, "links");
    count("links_rejected_consistency", result.linkReport.skippedConsistency, "links");
    count("links_rejected_tail", result.linkReport.skippedTail, "links");
    count("links_rejected_gap", result.linkReport.skippedGap, "links");
    if (stages.events) count("crossings_rejected_gap", stages.events->rejectedGap, "crossings");
    count("links_with_crossing", result.linkReport.linksWithCrossing, "links");
    count("crossings_replaced_by_links", result.linkReport.deduplicatedCrossings, "crossings");
    metric("link_refinement_max_shift", result.linkReport.maxRefinementShift, "voxels");
    if (stages.flipReport) {
        count("normal_order_links_checked", stages.flipReport->checked, "links");
        count("normal_order_links_agreed", stages.flipReport->agreed, "links");
        count("normal_order_links_disagreed", stages.flipReport->disagreed, "links");
        count("normal_order_links_undetermined", stages.flipReport->undetermined, "links");
        count("normal_order_links_opposed", stages.flipReport->opposed, "links");
    }
    count("perpendicular_constraints", result.constraintsByKind[std::size_t(ScaffoldConstraintKind::Perpendicular)], "constraints");
    count("parallel_constraints", result.constraintsByKind[std::size_t(ScaffoldConstraintKind::Parallel)], "constraints");
    count("enclosure_constraints", result.constraintsByKind[std::size_t(ScaffoldConstraintKind::Enclosure)], "constraints");
    count("sign_constraints",
          result.constraintsByKind[std::size_t(ScaffoldConstraintKind::SignAtLeast)] +
              result.constraintsByKind[std::size_t(ScaffoldConstraintKind::SignAtMost)],
          "constraints");
    if (stages.normals) {
        count("normal_fibers_solved", stages.normals->solved, "fibers");
        count("normal_fibers_reused", stages.normals->reused, "fibers");
        count("normal_fibers_cached", stages.normals->cached, "fibers");
        count("normal_fibers_degenerate", stages.normals->degenerate, "fibers");
        count("normal_fibers_anchored", stages.normals->anchoredFibers, "fibers");
        count("normal_anchors_manual", stages.normals->manualAnchors, "anchors");
        count("normal_anchors_approved_piece", stages.normals->approvedPieceAnchors, "anchors");
        count("normal_points_approved", stages.normals->approvedNormalPoints, "points");
        count("normal_approvals_stale", stages.normals->staleNormalApprovals, "approvals");
        count("normal_anchors_adjacent_link", stages.normals->adjacentLinkAnchors, "anchors");
        if (stages.normalOptions && stages.normalOptions->neighbours.field) {
            count("neighbour_samples", stages.normals->neighbourSamples, "samples");
            count("neighbour_samples_answered", stages.normals->neighbourSamplesAnswered, "samples");
            count("neighbour_nodes", stages.normals->neighbourNodes, "nodes");
            count("neighbour_fibers_covered", stages.normals->neighbourCoveredFibers, "fibers");
            count("neighbour_fibers_faced", stages.normals->neighbourFacedFibers, "fibers");
            count("crossings_field", stages.normals->fieldCrossings, "crossings");
            count("crossings_scaffold", stages.normals->scaffoldCrossings, "crossings");
            count("crossed_nodes", stages.normals->crossedNodes, "nodes");
            metric("neighbour_seconds", stages.normals->neighbourSeconds, "seconds");
        }
        metric("normals_seconds", stages.normals->seconds, "seconds");
    }
    if (stages.events) {
        count("events", stages.events->events, "events");
        count("events_reused_pairs", stages.events->reusedPairs, "fiber_pairs");
        count("events_reused", stages.events->reusedEvents, "events");
        count("events_cached_pairs", stages.events->cachedPairs, "fiber_pairs");
        count("events_perpendicular", stages.events->perpendicular, "events");
        count("events_parallel", stages.events->parallel, "events");
        count("events_with_winding_distance", stages.events->withDistance, "events");
        count("events_without_normals", stages.events->withoutNormals, "events");
        metric("events_seconds", stages.events->seconds, "seconds");
    }
    if (stages.measured) {
        count("measured_events_used", stages.measured->used, "events");
        count("measured_without_distance", stages.measured->withoutDistance, "events");
        count("measured_fractional_separations", stages.measured->fractional, "events");
        // Why the rest were dropped, so the gap between the crossings read
        // and the relations used is accounted for rather than implied. A
        // family mismatch means the pair's geometry and their H/V tags
        // disagree, which is a tagging question, not a measurement one.
        count("measured_skipped_invalid", stages.measured->skippedInvalid, "events");
        count("measured_skipped_untagged", stages.measured->skippedUntagged, "events");
        count("measured_skipped_self", stages.measured->skippedSelf, "events");
        count("measured_skipped_family_mismatch", stages.measured->skippedFamilyMismatch, "events");
        count("enclosure_pairs", stages.measured->enclosurePairs, "fiber_pairs");
        count("enclosure_one_sided", stages.measured->enclosureOneSided, "fiber_pairs");
        count("enclosure_contradictory", stages.measured->enclosureContradictory, "fiber_pairs");
        count("enclosure_nonunique", stages.measured->enclosureNonunique, "fiber_pairs");
        count("crossings_stated_direct", stages.measured->statedDirect, "crossings");
        count("crossings_stated_distant_kept", stages.measured->statedDistantKept, "crossings");
        count("crossings_stated_distant_dropped", stages.measured->statedDistantDropped, "crossings");
        count("crossings_stated_without_sign", stages.measured->statedWithoutSign, "crossings");
        count("sign_constraints_on_approved_normals", stages.measured->signOnApprovedNormals, "constraints");
    }
    if (stages.signs) {
        count("sign_events", stages.signs->events, "events");
        count("signs_fixed", stages.signs->fixed, "fibers");
        count("signs_decided", stages.signs->decided, "fibers");
        count("signs_flipped", stages.signs->flipped, "fibers");
        count("signs_undecided", stages.signs->undecided, "fibers");
        count("sign_fixed_disagreements", stages.signs->fixedDisagreements, "fibers");
        if (stages.signs->solved) {
            metric("sign_objective", stages.signs->solve.objective, "weighted_phase");
            metric("sign_exact_objective", stages.signs->solve.exactObjective, "weighted_phase");
            metric("sign_seconds", stages.signs->solve.seconds, "seconds");
            metric("sign_weighted_disagreement_percent", stages.signs->solve.metrics.weightedDisagreementPercent,
                   "percent");
            metric("sign_strict_disagreement_percent", stages.signs->solve.metrics.strictDisagreementPercent,
                   "percent");
        }
    }
    if (stages.facing) {
        count("approved_normal_stretches", stages.facing->stretches, "stretches");
        count("approved_normal_objects", stages.facing->objects, "objects");
        count("approved_normal_crossings", stages.facing->crossings, "crossings");
        count("approved_normal_objects_faced_by_neighbours", stages.facing->facedByNeighbours, "objects");
        metric("approved_normal_weakest_agreement", stages.facing->weakestAgreement, "share");
        if (stages.normals) count("approved_normal_gaps_unrolled", stages.normals->unrolledGaps, "gaps");
    }
    if (stages.approvals) {
        count("constraint_approval_rows", stages.approvals->live, "rows");
        count("constraint_approvals_applied", stages.approvals->approved, "constraints");
        count("constraint_approvals_stale", stages.approvals->stale, "rows");
    }
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
    if (stages.flips) {
        for (std::size_t at = 0; at < stages.flips->size(); ++at) {
            const auto& flip = (*stages.flips)[at];
            const auto& constraint = result.constraints.at(std::size_t(flip.constraint));
            const auto rank = std::int64_t(result.residuals.size() + at);
            residuals.push_back(objects::fiberGraphSolveRow(FiberGraphTable::Residuals,
                {{"run_uuid", uuidValue(record.runUuid)},
                 {"residual_index", rank},
                 {"kind", kScaffoldNormalFlipRowKind},
                 {"fiber_a", uuidValue(graph.fibers[std::size_t(flip.fiber)].uuid)},
                 {"fiber_b", uuidValue(graph.fibers[std::size_t(flip.other)].uuid)},
                 {"target", constraint.target},
                 {"weight", double(flip.agreeing)},
                 {"residual", double(flip.disagreeing)},
                 {"edge_uuid", constraint.link >= 0 ? uuidValue(graph.links[std::size_t(constraint.link)].uuid) : nullValue()},
                 {"arc_a", flip.arc},
                 {"arc_b", flip.otherArc},
                 {"cycle_hops", nullValue()},
                 {"cycle_piece_length", nullValue()},
                 {"redundancy", nullValue()}}).values);
            pieces.push_back(objects::fiberGraphSolveRow(FiberGraphTable::ResidualPieces,
                {{"run_uuid", uuidValue(record.runUuid)},
                 {"residual_index", rank},
                 {"member_index", std::int64_t{0}},
                 {"fiber_uuid", uuidValue(graph.fibers[std::size_t(flip.piece.fiber)].uuid)},
                 {"arc_from", std::min(flip.piece.arcFrom, flip.piece.arcTo)},
                 {"arc_to", std::max(flip.piece.arcFrom, flip.piece.arcTo)},
                 {"control_from", flip.piece.controlFrom >= 0 ? ExtensionValue{flip.piece.controlFrom} : nullValue()},
                 {"control_to", flip.piece.controlTo >= 0 ? ExtensionValue{flip.piece.controlTo} : nullValue()},
                 {"score", nullValue()}}).values);
        }
    }
    if (stages.crossings) {
        const auto first = std::int64_t(result.residuals.size() + (stages.flips ? stages.flips->size() : 0));
        for (std::size_t at = 0; at < stages.crossings->size(); ++at) {
            const auto& crossing = (*stages.crossings)[at];
            residuals.push_back(objects::fiberGraphSolveRow(FiberGraphTable::Residuals,
                {{"run_uuid", uuidValue(record.runUuid)},
                 {"residual_index", first + std::int64_t(at)},
                 {"kind", kScaffoldCrossingRowKind},
                 {"fiber_a", uuidValue(graph.fibers[std::size_t(crossing.a)].uuid)},
                 {"fiber_b", uuidValue(graph.fibers[std::size_t(crossing.b)].uuid)},
                 {"target", crossing.estimateHalves},
                 {"weight", double((crossing.signKnown ? 1 : 0) + (crossing.separation ? 2 : 0))},
                 {"residual", std::isfinite(crossing.distance) ? crossing.distance : -1.},
                 {"edge_uuid", nullValue()},
                 {"arc_a", crossing.arcA},
                 {"arc_b", crossing.arcB},
                 {"cycle_hops", nullValue()},
                 {"cycle_piece_length", nullValue()},
                 {"redundancy", nullValue()}}).values);
        }
    }
    {
        // Every continuation link with what the solve made of it.
        std::map<std::pair<std::int64_t, double>, double> broken;
        for (const auto& entry : result.residuals) {
            const auto& constraint = result.constraints[std::size_t(entry.constraint)];
            if (constraint.kind == ScaffoldConstraintKind::Continuation)
                broken[{constraint.a, constraint.arcB}] = entry.residual;
        }
        auto index = std::int64_t(residuals.size());
        for (const auto& constraint : result.constraints) {
            if (constraint.kind != ScaffoldConstraintKind::Continuation) continue;
            const auto found = broken.find({constraint.a, constraint.arcB});
            residuals.push_back(objects::fiberGraphSolveRow(FiberGraphTable::Residuals,
                {{"run_uuid", uuidValue(record.runUuid)},
                 {"residual_index", index++},
                 {"kind", kScaffoldContinuationRowKind},
                 {"fiber_a", uuidValue(graph.fibers[std::size_t(constraint.a)].uuid)},
                 {"fiber_b", uuidValue(graph.fibers[std::size_t(constraint.b)].uuid)},
                 {"target", std::int64_t{0}},
                 {"weight", constraint.weight},
                 {"residual", found == broken.end() ? 0. : found->second},
                 {"edge_uuid", nullValue()},
                 {"arc_a", constraint.arcB},
                 {"arc_b", constraint.arcB},
                 {"cycle_hops", nullValue()},
                 {"cycle_piece_length", nullValue()},
                 {"redundancy", nullValue()}}).values);
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
        // The columns the docks show, out of the settings the run recorded.
        try {
            const auto settings = nlohmann::json::parse(entry.settingsJson, nullptr, false);
            if (settings.is_object()) {
                if (settings.contains("created_at_unix") && settings["created_at_unix"].is_number_integer())
                    entry.createdAtUnix = settings["created_at_unix"].get<std::int64_t>();
                if (settings.contains("normals_fused") && settings["normals_fused"].is_boolean())
                    entry.normalSource = settings["normals_fused"].get<bool>() ? "fused" : "geometry";
                if (settings.contains("normals_neighbours") && settings["normals_neighbours"].is_boolean() &&
                    settings["normals_neighbours"].get<bool>())
                    entry.normalSource = "neighbours";
                if (settings.contains("sign_backend_ceres") && settings["sign_backend_ceres"].is_boolean())
                    entry.signBackend = settings["sign_backend_ceres"].get<bool>() ? "ceres" : "irls";
            }
        } catch (const std::exception&) {
            // Settings that cannot be parsed leave the columns empty; the run
            // itself is still a run.
        }
        if (stage.empty() || entry.stage == stage) out.push_back(std::move(entry));
    });
    const Columns inputs(FiberGraphTable::Inputs);
    for (auto& entry : out) {
        scanRun(reader, FiberGraphTable::Inputs, entry.runUuid, [&](const ExtensionRow& row) {
            if (inputs.text(row, "role") != "input_0") return;
            entry.input.storeUuid = inputs.uuid(row, "store_uuid");
            entry.input.generation = inputs.uuid(row, "generation_uuid");
            entry.input.commitSequence = inputs.integer(row, "commit_sequence");
        });
    }
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
        scanRun(reader, FiberGraphTable::Residuals, entry.runUuid, [&](const ExtensionRow& row) {
            const auto kind = residuals.integer(row, "kind");
            if (kind < kScaffoldNormalFlipRowKind) ++entry.residuals;
        });
    }
    return out;
}

namespace {
// `suspects`: 0 the violated constraints, otherwise the rows of that suspect kind.
std::vector<ScaffoldStoredResidual> readResidualRows(const objects::StoreReader& reader, const std::string& runUuid,
                                                     std::int64_t suspects);
}

std::vector<ScaffoldStoredResidual> readScaffoldResiduals(const objects::StoreReader& reader,
                                                          const std::string& runUuid)
{
    return readResidualRows(reader, runUuid, 0);
}

std::vector<ScaffoldStoredResidual> readScaffoldNormalFlips(const objects::StoreReader& reader,
                                                            const std::string& runUuid)
{
    return readResidualRows(reader, runUuid, kScaffoldNormalFlipRowKind);
}

std::vector<ScaffoldStoredCrossing> readScaffoldCrossings(const objects::StoreReader& reader,
                                                          const std::string& runUuid)
{
    std::vector<ScaffoldStoredCrossing> out;
    for (const auto& row : readResidualRows(reader, runUuid, kScaffoldCrossingRowKind)) {
        ScaffoldStoredCrossing crossing;
        crossing.fiberA = row.fiberA;
        crossing.fiberB = row.fiberB;
        crossing.arcA = row.arcA;
        crossing.arcB = row.arcB;
        crossing.estimateHalves = row.target;
        const auto flags = std::int64_t(std::llround(row.weight));
        crossing.signKnown = (flags & 1) != 0;
        crossing.separation = (flags & 2) != 0;
        crossing.distance = row.residual < 0 ? std::numeric_limits<double>::quiet_NaN() : row.residual;
        out.push_back(std::move(crossing));
    }
    return out;
}

std::vector<ScaffoldStoredContinuation> readScaffoldContinuations(const objects::StoreReader& reader,
                                                                  const std::string& runUuid)
{
    std::vector<ScaffoldStoredContinuation> out;
    for (const auto& row : readResidualRows(reader, runUuid, kScaffoldContinuationRowKind))
        out.push_back({row.fiberA, row.arcA, row.weight, row.residual});
    return out;
}

namespace {
std::vector<ScaffoldStoredResidual> readResidualRows(const objects::StoreReader& reader, const std::string& runUuid,
                                                     std::int64_t suspects)
{
    if (runUuid.empty()) throw std::invalid_argument("scaffold run: a run UUID is required");
    const Columns columns(FiberGraphTable::Residuals);
    std::vector<ScaffoldStoredResidual> out;
    std::map<std::int64_t, std::size_t> byIndex;
    scanRun(reader, FiberGraphTable::Residuals, runUuid, [&](const ExtensionRow& row) {
        ScaffoldStoredResidual entry;
        entry.index = columns.integer(row, "residual_index");
        entry.kind = columns.integer(row, "kind");
        // Kinds from the flip kind on are not violated constraints.
        const bool suspect = entry.kind >= kScaffoldNormalFlipRowKind;
        if (suspects == 0 ? suspect : entry.kind != suspects) return;
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
} // namespace

std::vector<double> readScaffoldRunFiberNormals(const objects::StoreReader& reader, const std::string& runUuid,
                                                const std::string& fiberUuid, std::size_t linePoints)
{
    if (runUuid.empty() || fiberUuid.empty())
        throw std::invalid_argument("scaffold run: a run and a fiber UUID are required");
    const auto row = objects::readFiberScaffoldRunNormals(reader, runUuid, fiberUuid);
    if (row.normals.size() != linePoints * 3 * sizeof(double))
        throw std::runtime_error("the run's normals cover " + std::to_string(row.normals.size() / (3 * sizeof(double))) +
                                 " line points of fiber " + fiberUuid + " at version " +
                                 std::to_string(row.fiberVersion) + ", the fiber now has " +
                                 std::to_string(linePoints) + ": it was edited since the run; solve again");
    std::vector<double> normals(linePoints * 3);
    std::memcpy(normals.data(), row.normals.data(), row.normals.size());
    return normals;
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
