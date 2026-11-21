#include "vc/fiber_graph/ScaffoldWindingSolve.hpp"

#include "vc/fiber_graph/ScaffoldConstraints.hpp"

#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include "vc_object_store/ParallelFor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>

namespace vc::fiber_graph {
namespace {

struct UnionFind {
    std::vector<std::int64_t> parent;
    explicit UnionFind(std::size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    std::int64_t find(std::int64_t x)
    {
        while (parent[std::size_t(x)] != x) x = parent[std::size_t(x)] = parent[std::size_t(parent[std::size_t(x)])];
        return x;
    }
    void join(std::int64_t x, std::int64_t y)
    {
        x = find(x);
        y = find(y);
        if (x != y) parent[std::size_t(y)] = x;
    }
};

// Component labels in node order, as the reference stage's
// connected_components numbers them.
std::vector<std::int64_t> componentLabels(UnionFind& sets, std::size_t nodes, std::int64_t& count)
{
    std::vector<std::int64_t> label(nodes, -1);
    count = 0;
    for (std::size_t i = 0; i < nodes; ++i) {
        const auto root = sets.find(std::int64_t(i));
        if (label[std::size_t(root)] < 0) label[std::size_t(root)] = count++;
        label[i] = label[std::size_t(root)];
    }
    return label;
}

// Exact minimum-cost winding of one arc list, gauged to minimum 0 per
// component. Nodes without arcs stay at 0.
std::vector<std::int64_t> solveComponents(std::size_t nodes, const std::vector<WindingArc>& arcs,
                                          const geometry::WindingFlowOptions& options,
                                          geometry::WindingFlowQuantization* quantization = nullptr,
                                          ScaffoldRunMonitor* monitor = nullptr)
{
    std::vector<std::int64_t> winding(nodes, 0);
    if (arcs.empty()) return winding;
    UnionFind sets(nodes);
    for (const auto& arc : arcs) sets.join(arc.a, arc.b);
    std::int64_t components = 0;
    const auto label = componentLabels(sets, nodes, components);
    const std::size_t componentCount = std::size_t(components);
    std::vector<std::vector<const WindingArc*>> grouped(componentCount);
    for (const auto& arc : arcs) grouped[std::size_t(label[std::size_t(arc.a)])].push_back(&arc);
    std::vector<std::int64_t> local(nodes, -1);
    if (monitor) monitor->enter("winding", std::int64_t(grouped.size()));
    for (std::size_t component = 0; component < grouped.size(); ++component) {
        if (monitor) {
            monitor->checkpoint();
            monitor->advance();
        }
        if (grouped[component].empty()) continue;
        std::vector<std::int64_t> members;
        for (const auto* arc : grouped[component])
            for (const auto node : {arc->a, arc->b})
                if (local[std::size_t(node)] < 0) {
                    local[std::size_t(node)] = std::int64_t(members.size());
                    members.push_back(node);
                }
        std::vector<std::int64_t> a, b, target;
        std::vector<double> weight;
        a.reserve(grouped[component].size());
        for (const auto* arc : grouped[component]) {
            a.push_back(local[std::size_t(arc->a)]);
            b.push_back(local[std::size_t(arc->b)]);
            target.push_back(arc->target);
            weight.push_back(arc->weight);
        }
        const auto flow = geometry::solveWindingMinCostFlow(std::int64_t(members.size()), a, b, target, weight, options);
        std::int64_t minimum = flow.winding[0];
        for (std::size_t i = 1; i < members.size(); ++i) minimum = std::min(minimum, flow.winding[i]);
        for (std::size_t i = 0; i < members.size(); ++i) winding[std::size_t(members[i])] = flow.winding[i] - minimum;
        if (quantization) {
            quantization->scale = flow.quantization.scale;
            quantization->originalArcs += flow.quantization.originalArcs;
            quantization->retainedArcs += flow.quantization.retainedArcs;
            quantization->droppedArcs += flow.quantization.droppedArcs;
            quantization->originalWeight += flow.quantization.originalWeight;
            quantization->droppedWeight += flow.quantization.droppedWeight;
            quantization->capacityAbsoluteError =
                std::max(quantization->capacityAbsoluteError, flow.quantization.capacityAbsoluteError);
        }
    }
    return winding;
}

double arcOf(const ScaffoldConstraint& constraint, std::int64_t fiber)
{
    if (constraint.a == fiber) return constraint.arcA;
    if (constraint.b == fiber) return constraint.arcB;
    throw std::invalid_argument("scaffold winding: constraint is not incident to the fiber");
}

std::int64_t otherEnd(const ScaffoldConstraint& constraint, std::int64_t fiber)
{
    return constraint.a == fiber ? constraint.b : constraint.a;
}

// The controls bracketing an arc interval, outward: the stretch covers the
// spans [controlFrom, controlTo - 1], and a piece opened between those two
// controls contains it. A fiber without control points keeps -1.
void controlRange(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry, ScaffoldPiece& piece)
{
    scaffoldPieceControls(graph, geometry, piece);
}
} // namespace

void scaffoldPieceControls(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry, ScaffoldPiece& piece)
{
    const auto& fiber = graph.fibers.at(std::size_t(piece.fiber));
    const auto controls = fiber.content.controlIds.size();
    if (controls == 0) return;
    const double low = std::min(piece.arcFrom, piece.arcTo), high = std::max(piece.arcFrom, piece.arcTo);
    piece.controlFrom = 0;
    piece.controlTo = std::int64_t(controls) - 1;
    for (std::size_t control = 0; control < controls; ++control) {
        const double arc = geometry.controlArc(std::size_t(piece.fiber), control);
        if (arc <= low) piece.controlFrom = std::int64_t(control);
        if (arc >= high) {
            piece.controlTo = std::int64_t(control);
            break;
        }
    }
    // A degenerate range is widened by one span, downward at the fiber's last
    // control (where there is nothing above it): cp n-n is no piece at all.
    if (piece.controlTo <= piece.controlFrom) {
        if (piece.controlFrom + 1 <= std::int64_t(controls) - 1)
            piece.controlTo = piece.controlFrom + 1;
        else
            piece.controlFrom = std::max<std::int64_t>(0, piece.controlTo - 1);
    }
    if (piece.controlTo <= piece.controlFrom) { piece.controlFrom = -1; piece.controlTo = -1; }
}

namespace {

struct PathSearch {
    // Satisfied constraints incident to each fiber.
    std::vector<std::vector<std::int64_t>> incident;
    const std::vector<ScaffoldConstraint>* constraints;

    // Shortest alternative path from `from` to `to` over satisfied
    // constraints: fewest hops, ties by the total length of the fiber
    // stretches the cycle runs along. The search state is the constraint a
    // fiber was entered by, because the stretch depends on both incident
    // anchors. The search reads only its arguments, so violated constraints
    // can be analysed in parallel.
    std::vector<std::int64_t> shortest(std::int64_t from, std::int64_t to, const ScaffoldConstraint& closing,
                                       const std::set<std::int64_t>& blocked) const
    {
        struct State {
            std::int64_t hops;
            double length;
            std::int64_t node, constraint;
            bool operator>(const State& other) const
            {
                return std::tie(hops, length) > std::tie(other.hops, other.length);
            }
        };
        std::map<std::pair<std::int64_t, std::int64_t>, std::pair<std::int64_t, double>> best;
        std::map<std::pair<std::int64_t, std::int64_t>, std::pair<std::int64_t, std::int64_t>> parent;
        std::priority_queue<State, std::vector<State>, std::greater<State>> queue;
        const std::pair<std::int64_t, std::int64_t> start{from, -1};
        best[start] = {0, 0.};
        queue.push({0, 0., from, -1});
        while (!queue.empty()) {
            const auto state = queue.top();
            queue.pop();
            const std::pair<std::int64_t, std::int64_t> key{state.node, state.constraint};
            const auto known = best.find(key);
            if (known == best.end() || known->second.first != state.hops || known->second.second != state.length) continue;
            const double entry = state.constraint < 0 ? arcOf(closing, state.node)
                                                      : arcOf((*constraints)[std::size_t(state.constraint)], state.node);
            if (state.node == to && state.constraint >= 0) {
                std::vector<std::int64_t> path;
                auto cursor = key;
                while (cursor.second >= 0) {
                    path.push_back(cursor.second);
                    cursor = parent.at(cursor);
                }
                std::reverse(path.begin(), path.end());
                return path;
            }
            for (const auto next : incident[std::size_t(state.node)]) {
                if (next == state.constraint || blocked.contains(next)) continue;
                const auto& constraint = (*constraints)[std::size_t(next)];
                const double length = std::abs(arcOf(constraint, state.node) - entry);
                const std::pair<std::int64_t, std::int64_t> candidate{otherEnd(constraint, state.node), next};
                // Reaching the far fiber closes the cycle: its stretch between
                // this constraint and the violated one belongs to the length.
                const double closingLength =
                    candidate.first == to ? std::abs(arcOf(constraint, to) - arcOf(closing, to)) : 0.;
                const std::pair<std::int64_t, double> cost{state.hops + 1, state.length + length + closingLength};
                const auto found = best.find(candidate);
                if (found != best.end() && !(cost < found->second)) continue;
                best[candidate] = cost;
                parent[candidate] = key;
                queue.push({cost.first, cost.second, candidate.first, next});
            }
        }
        return {};
    }
};

// The fiber stretches a cycle (the violated constraint plus `path`) runs
// along, in cycle order.
std::vector<ScaffoldPiece> cyclePieces(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry,
                                       const std::vector<ScaffoldConstraint>& constraints, std::int64_t violated,
                                       const std::vector<std::int64_t>& path)
{
    std::vector<ScaffoldPiece> pieces;
    if (path.empty()) return pieces;
    const auto& closing = constraints[std::size_t(violated)];
    std::int64_t node = closing.a, incoming = violated;
    for (const auto next : path) {
        const auto& constraint = constraints[std::size_t(next)];
        ScaffoldPiece piece;
        piece.fiber = node;
        piece.arcFrom = arcOf(constraints[std::size_t(incoming)], node);
        piece.arcTo = arcOf(constraint, node);
        piece.constraintFrom = incoming;
        piece.constraintTo = next;
        controlRange(graph, geometry, piece);
        pieces.push_back(piece);
        node = otherEnd(constraint, node);
        incoming = next;
    }
    ScaffoldPiece last;
    last.fiber = node;
    last.arcFrom = arcOf(constraints[std::size_t(incoming)], node);
    last.arcTo = arcOf(closing, node);
    last.constraintFrom = incoming;
    last.constraintTo = violated;
    controlRange(graph, geometry, last);
    pieces.push_back(last);
    return pieces;
}
} // namespace

const char* scaffoldConstraintKindName(ScaffoldConstraintKind kind)
{
    switch (kind) {
        case ScaffoldConstraintKind::Perpendicular: return "perpendicular";
        case ScaffoldConstraintKind::Parallel: return "parallel";
        case ScaffoldConstraintKind::Enclosure: return "enclosure";
    }
    throw std::invalid_argument("scaffold winding: unknown constraint kind");
}

std::optional<std::int64_t> adjacentStep(ScaffoldFamily a, ScaffoldFamily b)
{
    if (a == ScaffoldFamily::Horizontal && b == ScaffoldFamily::Vertical) return -1;
    if (a == ScaffoldFamily::Vertical && b == ScaffoldFamily::Horizontal) return 1;
    return std::nullopt;
}

std::vector<WindingArc> aggregateWindingArcs(std::vector<WindingArc> arcs)
{
    if (arcs.empty()) return arcs;
    std::stable_sort(arcs.begin(), arcs.end(), [](const WindingArc& x, const WindingArc& y) {
        return std::tie(x.a, x.b, x.target) < std::tie(y.a, y.b, y.target);
    });
    std::size_t out = 0;
    for (std::size_t i = 1; i < arcs.size(); ++i) {
        if (arcs[out].a == arcs[i].a && arcs[out].b == arcs[i].b && arcs[out].target == arcs[i].target)
            arcs[out].weight += arcs[i].weight;
        else
            arcs[++out] = arcs[i];
    }
    arcs.resize(out + 1);
    return arcs;
}

std::vector<ScaffoldConstraint> scaffoldLinkConstraints(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry,
                                                       const ScaffoldLinkArcOptions& options, ScaffoldLinkReport& report)
{
    if (!std::isfinite(options.maxAnchorDistance) || options.maxAnchorDistance < 0)
        throw std::invalid_argument("scaffold winding: max anchor distance must be finite and non-negative");
    if (!std::isfinite(options.weight) || options.weight < 0)
        throw std::invalid_argument("scaffold winding: manual link weight must be finite and non-negative");
    if (options.approvalStates.empty())
        throw std::invalid_argument("scaffold winding: at least one link approval state is required");

    if (!(options.refineMaximumDistance > 0) || !std::isfinite(options.refineMaximumDistance))
        throw std::invalid_argument("scaffold winding: the link refinement distance must be positive and finite");

    report = ScaffoldLinkReport{};
    std::vector<ScaffoldConstraint> constraints;
    // The extraction's own refinement, in the mode the scaffold's crossings
    // are extracted in (whole fibers, endpoint support certified).
    fiber_tracer::FiberTraceConstraintConfig refinement;
    refinement.localEvents = true;
    refinement.localKeepWholeFibers = true;
    refinement.maximumDistanceBaseVoxels = options.refineMaximumDistance;
    std::map<std::int64_t, fiber_tracer::PolylineArcGeometry> lines;
    const auto lineOf = [&](std::int64_t fiber) -> const fiber_tracer::PolylineArcGeometry& {
        auto found = lines.find(fiber);
        if (found == lines.end()) {
            const auto& entry = graph.fibers[std::size_t(fiber)];
            std::vector<cv::Vec3d> points;
            points.reserve(std::size_t(entry.pointCount));
            for (std::int64_t j = 0; j < entry.pointCount; ++j) {
                const auto at = std::size_t(3 * (entry.pointFirst + j));
                points.emplace_back(graph.linePoints[at], graph.linePoints[at + 1], graph.linePoints[at + 2]);
            }
            found = lines.emplace(fiber, fiber_tracer::makePolylineArcGeometry(points)).first;
        }
        return found->second;
    };
    for (std::size_t index = 0; index < graph.links.size(); ++index) {
        const auto& link = graph.links[index];
        ++report.links;
        if (std::find(options.approvalStates.begin(), options.approvalStates.end(), link.approvalState) ==
            options.approvalStates.end()) {
            ++report.skippedApprovalState;
            continue;
        }
        if (link.autoLink && !options.includeAuto) { ++report.skippedAutomatic; continue; }
        if (link.a < 0 || link.b < 0) { ++report.skippedEndpoint; continue; }
        if (link.a == link.b) { ++report.skippedSelfLink; continue; }
        const auto& fiberA = graph.fibers[std::size_t(link.a)];
        const auto& fiberB = graph.fibers[std::size_t(link.b)];
        if (fiberA.family == ScaffoldFamily::Untagged || fiberB.family == ScaffoldFamily::Untagged) {
            ++report.skippedUntagged;
            continue;
        }
        std::int64_t step = 0;
        if (link.adjacent) {
            const auto adjacent = adjacentStep(fiberA.family, fiberB.family);
            if (!adjacent) { ++report.skippedAdjacentSameFamily; continue; }
            step = *adjacent;
            ++report.adjacent;
        } else {
            ++report.ordinary;
        }
        if (!link.anchorA || !link.anchorB) { ++report.skippedMissingAnchor; continue; }
        const auto projectedA = geometry.project(std::size_t(link.a), *link.anchorA);
        const auto projectedB = geometry.project(std::size_t(link.b), *link.anchorB);
        const auto drop = [&]() { if (link.adjacent) --report.adjacent; else --report.ordinary; };
        // An anchor past the first or last control point is on a tail: the
        // solve reads the annotated line only, so such an anchor projects onto
        // the line's end from outside it. A tail is not fiber.
        const auto onTail = [&](std::int64_t fiber, const ScaffoldGeometry::Projection& projected) {
            const double length = geometry.arcLength(std::size_t(fiber));
            return projected.distance > 0.5 && (projected.arc <= 1e-9 || projected.arc >= length - 1e-9);
        };
        if (onTail(link.a, projectedA) || onTail(link.b, projectedB)) { ++report.skippedTail; drop(); continue; }
        report.maxAnchorDistance = std::max({report.maxAnchorDistance, projectedA.distance, projectedB.distance});
        // A frame error is reported by the guard below, not as a rejection.
        if (std::max(projectedA.distance, projectedB.distance) > options.maxAnchorDistance) continue;
        if (fiberA.pointCount < 2 || fiberB.pointCount < 2) {
            ++report.skippedEndSupport;
            if (link.adjacent) --report.adjacent; else --report.ordinary;
            continue;
        }
        const auto refined =
            fiber_tracer::refineFiberCrossing(lineOf(link.a), lineOf(link.b), projectedA.arc, projectedB.arc, refinement);
        const bool sameFamily = fiberA.family == fiberB.family;
        if (!refined.accepted) {
            const auto& why = refined.failure;
            if (why.find("orientation") != std::string::npos) ++report.skippedAngle;
            else if (why.find("support") != std::string::npos) ++report.skippedEndSupport;
            else if (why.find("distance") != std::string::npos) ++report.skippedDistance;
            else ++report.skippedConsistency;
            if (link.adjacent) --report.adjacent; else --report.ordinary;
            continue;
        }
        // The geometry's verdict has to be the families': as for a crossing.
        if (refined.parallel != sameFamily) {
            ++report.skippedAngle;
            if (link.adjacent) --report.adjacent; else --report.ordinary;
            continue;
        }
        // Neither the stated anchors nor the refined encounter may lie on a
        // gap span, or closer to one than the encounter needs fiber.
        {
            const double support = refined.parallel ? kScaffoldParallelSupport : kScaffoldPerpendicularSupport;
            const auto a = std::size_t(link.a), b = std::size_t(link.b);
            if (geometry.nearGap(a, projectedA.arc, 0.) || geometry.nearGap(b, projectedB.arc, 0.) ||
                geometry.nearGap(a, refined.arcA, support) || geometry.nearGap(b, refined.arcB, support)) {
                ++report.skippedGap;
                drop();
                continue;
            }
        }
        report.maxRefinementShift = std::max({report.maxRefinementShift, std::abs(refined.arcA - projectedA.arc),
                                              std::abs(refined.arcB - projectedB.arc)});

        ScaffoldConstraint constraint;
        constraint.kind = refined.parallel ? ScaffoldConstraintKind::Parallel : ScaffoldConstraintKind::Perpendicular;
        constraint.link = std::int64_t(index);
        constraint.a = link.a;
        constraint.b = link.b;
        constraint.arcA = refined.arcA;
        constraint.arcB = refined.arcB;
        constraint.step = step;
        constraint.weight = options.weight;
        // An approved record is not enough: a link is held rigid only when
        // the two pieces its anchors lie on are approved geometry.
        constraint.approved = link.approvalState == "approved";
        if (constraint.approved && !scaffoldConstraintPiecesApproved(graph, geometry, constraint)) {
            constraint.approved = false;
            ++report.approvedWithoutPieces;
        }
        constraints.push_back(constraint);
        ++report.used;
    }
    if (report.maxAnchorDistance > options.maxAnchorDistance)
        throw std::invalid_argument("scaffold winding: link anchors lie " + std::to_string(report.maxAnchorDistance) +
                                    " voxels from their fiber polyline; check the coordinate frame or raise the "
                                    "maximum anchor distance");
    // The whole-fiber step: the anchor assertion plus the transport along both
    // fibers and across the connector.
    for (auto& constraint : constraints) {
        const auto start = geometry.packed().sample(constraint.a, constraint.arcA);
        const auto finish = geometry.packed().sample(constraint.b, constraint.arcB);
        const auto connector = geometry.seam().segment(start.point, finish.point);
        constraint.target = constraint.step + connector - finish.turn + start.turn;
        if (constraint.a > constraint.b) {
            std::swap(constraint.a, constraint.b);
            std::swap(constraint.arcA, constraint.arcB);
            constraint.step = -constraint.step;
            constraint.target = -constraint.target;
        }
    }
    return constraints;
}

ScaffoldSolveResult solveScaffoldWinding(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry,
                                         const ScaffoldSolveOptions& options,
                                         std::span<const ScaffoldConstraint> measured)
{
    if (!geometry::windingMinCostFlowAvailable())
        throw std::runtime_error("scaffold winding: vc_object_store was built without LEMON (VC_OBJECT_STORE_WITH_LEMON)");
    if (options.maxAlternativePaths < 0)
        throw std::invalid_argument("scaffold winding: the alternative path cap must be non-negative");
    const auto started = std::chrono::steady_clock::now();
    const auto nodes = graph.fibers.size();
    if (options.monitor) {
        options.monitor->enter("constraints");
        options.monitor->checkpoint();
    }

    ScaffoldSolveResult result;
    // A run with the links switched off reads none of them: the report stays
    // empty and only the measured constraints below constrain the windings.
    if (options.links.useLinks)
        result.constraints = scaffoldLinkConstraints(graph, geometry, options.links, result.linkReport);
    else
        result.linkReport = ScaffoldLinkReport{};
    result.linkConstraints = std::int64_t(result.constraints.size());
    // One encounter, one relation: a measured crossing at a refined link is
    // that link's encounter, and the link's step stands. An enclosure is a
    // statement about the pair, not about one encounter, and stays.
    {
        if (!(options.links.deduplicateArc >= 0))
            throw std::invalid_argument("scaffold winding: the deduplication arc distance must be non-negative");
        std::multimap<std::pair<std::int64_t, std::int64_t>, std::size_t> linked;
        for (std::size_t i = 0; i < result.constraints.size(); ++i)
            linked.emplace(std::pair{result.constraints[i].a, result.constraints[i].b}, i);
        std::vector<char> hasCrossing(result.constraints.size(), 0);
        std::set<std::int64_t> droppedEvents;
        for (const auto& constraint : measured) {
            bool duplicate = false;
            if (constraint.kind != ScaffoldConstraintKind::Enclosure) {
                const auto [first, last] = linked.equal_range(std::pair{constraint.a, constraint.b});
                for (auto at = first; at != last; ++at) {
                    const auto& link = result.constraints[at->second];
                    if (std::abs(link.arcA - constraint.arcA) > options.links.deduplicateArc ||
                        std::abs(link.arcB - constraint.arcB) > options.links.deduplicateArc)
                        continue;
                    duplicate = true;
                    hasCrossing[at->second] = 1;
                }
            }
            if (duplicate) droppedEvents.insert(constraint.event);
            else result.constraints.push_back(constraint);
        }
        result.linkReport.deduplicatedCrossings = std::int64_t(droppedEvents.size());
        result.linkReport.linksWithCrossing = std::count(hasCrossing.begin(), hasCrossing.end(), char(1));
    }
    for (const auto& constraint : result.constraints) {
        const auto kind = std::size_t(constraint.kind);
        if (kind >= result.constraintsByKind.size())
            throw std::invalid_argument("scaffold winding: unknown constraint kind");
        ++result.constraintsByKind[kind];
        if (constraint.a < 0 || constraint.b < 0 || std::size_t(constraint.a) >= nodes ||
            std::size_t(constraint.b) >= nodes || constraint.a == constraint.b)
            throw std::invalid_argument("scaffold winding: a constraint names an unknown fiber pair");
        if (!std::isfinite(constraint.weight) || constraint.weight < 0)
            throw std::invalid_argument("scaffold winding: a constraint weight is negative or nonfinite");
    }
    std::vector<WindingArc> arcs;
    for (const auto& constraint : result.constraints)
        if (constraint.weight > 0) arcs.push_back({constraint.a, constraint.b, constraint.target, constraint.weight});
    result.arcs = aggregateWindingArcs(arcs);
    if (result.arcs.empty()) throw std::invalid_argument("scaffold winding: no usable winding constraints");

    // Approved subgraphs: solved alone, then held rigid. Each contributes one
    // node to the global problem, with its members' offsets.
    std::vector<std::int64_t> offset(nodes, 0), representative(nodes);
    std::iota(representative.begin(), representative.end(), 0);
    if (options.fixApprovedSubgraphs) {
        UnionFind approved(nodes);
        std::vector<WindingArc> approvedArcs;
        for (const auto& constraint : result.constraints)
            if (constraint.approved && constraint.weight > 0) {
                approved.join(constraint.a, constraint.b);
                approvedArcs.push_back({constraint.a, constraint.b, constraint.target, constraint.weight});
            }
        if (!approvedArcs.empty()) {
            offset = solveComponents(nodes, aggregateWindingArcs(approvedArcs), options.flow);
            std::vector<std::string> violated;
            for (const auto& constraint : result.constraints) {
                if (!constraint.approved || constraint.weight <= 0) continue;
                if (offset[std::size_t(constraint.b)] - offset[std::size_t(constraint.a)] == constraint.target) continue;
                violated.push_back(graph.fibers[std::size_t(constraint.a)].uuid + " - " +
                                   graph.fibers[std::size_t(constraint.b)].uuid);
            }
            if (!violated.empty()) {
                std::string message = "scaffold winding: approved constraints contradict each other and cannot all be "
                                      "satisfied; revoke one of ";
                for (std::size_t i = 0; i < violated.size(); ++i) message += (i ? ", " : "") + violated[i];
                throw std::invalid_argument(message);
            }
            std::set<std::int64_t> subgraphs;
            for (std::size_t i = 0; i < nodes; ++i) {
                representative[i] = approved.find(std::int64_t(i));
                if (representative[i] != std::int64_t(i)) {
                    ++result.contractedFibers;
                    subgraphs.insert(representative[i]);
                }
            }
            result.approvedSubgraphs = std::int64_t(subgraphs.size());
        }
    }

    // Contracted arcs: k_S(b) - k_S(a) = target + offset_a - offset_b. Arcs
    // inside one approved subgraph are already decided and carry no freedom.
    std::vector<WindingArc> contracted;
    for (const auto& arc : result.arcs) {
        const auto a = representative[std::size_t(arc.a)], b = representative[std::size_t(arc.b)];
        if (a == b) continue;
        const auto target = arc.target + offset[std::size_t(arc.a)] - offset[std::size_t(arc.b)];
        contracted.push_back(a < b ? WindingArc{a, b, target, arc.weight} : WindingArc{b, a, -target, arc.weight});
    }
    const auto solved =
        solveComponents(nodes, aggregateWindingArcs(contracted), options.flow, &result.quantization, options.monitor);
    result.winding.resize(nodes);
    for (std::size_t i = 0; i < nodes; ++i)
        result.winding[i] = solved[std::size_t(representative[i])] + offset[i];

    // Components of the solved relation, and the gauge per component.
    UnionFind sets(nodes);
    for (const auto& arc : result.arcs) sets.join(arc.a, arc.b);
    result.component = componentLabels(sets, nodes, result.components);
    std::vector<std::int64_t> minimum(std::size_t(result.components), std::numeric_limits<std::int64_t>::max());
    std::vector<std::int64_t> members(std::size_t(result.components), 0);
    for (std::size_t i = 0; i < nodes; ++i) {
        minimum[std::size_t(result.component[i])] = std::min(minimum[std::size_t(result.component[i])], result.winding[i]);
        ++members[std::size_t(result.component[i])];
    }
    for (std::size_t i = 0; i < nodes; ++i) result.winding[i] -= minimum[std::size_t(result.component[i])];
    for (const auto count : members) if (count == 1) ++result.isolatedFibers;

    for (const auto& arc : result.arcs) {
        const double residual =
            std::abs(double(result.winding[std::size_t(arc.b)] - result.winding[std::size_t(arc.a)] - arc.target));
        result.objective += arc.weight * residual;
        result.maxResidual = std::max(result.maxResidual, residual);
        if (residual > 0) ++result.violatedArcs;
    }

    // Residual analysis: the shortest cycle through every violated constraint
    // over the satisfied ones.
    PathSearch search;
    search.constraints = &result.constraints;
    search.incident.resize(nodes);
    std::vector<double> residual(result.constraints.size(), 0.);
    for (std::size_t i = 0; i < result.constraints.size(); ++i) {
        const auto& constraint = result.constraints[i];
        residual[i] = std::abs(
            double(result.winding[std::size_t(constraint.b)] - result.winding[std::size_t(constraint.a)] - constraint.target));
        if (residual[i] == 0) {
            search.incident[std::size_t(constraint.a)].push_back(std::int64_t(i));
            search.incident[std::size_t(constraint.b)].push_back(std::int64_t(i));
        }
    }
    // The cycle of one violated constraint is found from the satisfied ones
    // alone, so the searches are independent. They run in parallel and are
    // merged in constraint order afterwards: the ranking never depends on
    // how the work was split.
    std::vector<std::int64_t> violated;
    for (std::size_t i = 0; i < result.constraints.size(); ++i)
        if (residual[i] != 0) violated.push_back(std::int64_t(i));
    result.violatedConstraints = std::int64_t(violated.size());
    std::vector<ScaffoldResidual> entries(violated.size());
    if (options.monitor) options.monitor->enter("residual cycles", std::int64_t(violated.size()));
    objects::parallelFor(violated.size(), std::max<std::size_t>(1, options.workers), [&](std::size_t task) {
        // A cancelled run leaves the loop through its first checkpoint:
        // parallelFor stops its other workers and rethrows here.
        if (options.monitor) {
            options.monitor->checkpoint();
            options.monitor->advance();
        }
        const auto i = std::size_t(violated[task]);
        const auto& constraint = result.constraints[i];
        auto& entry = entries[task];
        entry.constraint = std::int64_t(i);
        entry.residual = residual[i];
        entry.cost = constraint.weight * residual[i];
        std::set<std::int64_t> blocked;
        const auto path = search.shortest(constraint.a, constraint.b, constraint, blocked);
        if (path.empty()) return;
        entry.cycleHops = std::int64_t(path.size()) + 1;
        entry.pieces = cyclePieces(graph, geometry, result.constraints, std::int64_t(i), path);
        for (const auto& piece : entry.pieces) entry.cyclePieceLength += piece.length();
        entry.joint = entry.cost / double(entry.cycleHops);
        entry.redundancy = 1;
        auto blocking = path;
        for (std::int64_t round = 1; round < options.maxAlternativePaths; ++round) {
            for (const auto edge : blocking) blocked.insert(edge);
            blocking = search.shortest(constraint.a, constraint.b, constraint, blocked);
            if (blocking.empty()) break;
            ++entry.redundancy;
        }
    });
    std::map<std::tuple<std::int64_t, std::int64_t, std::int64_t>, std::size_t> pieceIndex;
    for (auto& entry : entries) {
        // A violated constraint accuses the stretches its cycle runs along, a
        // short stretch more specifically than a long one: its cost is split
        // in proportion to 1 / max(length, 1 voxel).
        double total = 0;
        for (const auto& piece : entry.pieces) total += 1. / std::max(piece.length(), 1.);
        for (const auto& piece : entry.pieces) {
            const double share = entry.cost * (1. / std::max(piece.length(), 1.)) / std::max(total, 1e-300);
            const std::tuple<std::int64_t, std::int64_t, std::int64_t> key{piece.fiber, piece.controlFrom,
                                                                          piece.controlTo};
            const auto found = pieceIndex.find(key);
            if (found == pieceIndex.end()) {
                pieceIndex.emplace(key, result.pieces.size());
                result.pieces.push_back({piece, share, entry.cycleHops, 1});
            } else {
                auto& score = result.pieces[found->second];
                score.score += share;
                ++score.constraints;
                score.cycleHops = std::min(score.cycleHops, entry.cycleHops);
            }
        }
        result.residuals.push_back(std::move(entry));
    }
    std::stable_sort(result.residuals.begin(), result.residuals.end(),
                     [](const ScaffoldResidual& x, const ScaffoldResidual& y) { return x.cost > y.cost; });
    std::stable_sort(result.pieces.begin(), result.pieces.end(),
                     [](const ScaffoldPieceScore& x, const ScaffoldPieceScore& y) { return x.score > y.score; });
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

} // namespace vc::fiber_graph
