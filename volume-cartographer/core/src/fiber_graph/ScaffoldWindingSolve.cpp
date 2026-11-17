#include "vc/fiber_graph/ScaffoldWindingSolve.hpp"

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
                                          geometry::WindingFlowQuantization* quantization = nullptr)
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
    for (std::size_t component = 0; component < grouped.size(); ++component) {
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

// The control-point index range covering an arc interval.
void controlRange(const ScaffoldGraph& graph, const ScaffoldGeometry& geometry, ScaffoldPiece& piece)
{
    const auto& fiber = graph.fibers.at(std::size_t(piece.fiber));
    const double low = std::min(piece.arcFrom, piece.arcTo), high = std::max(piece.arcFrom, piece.arcTo);
    for (std::size_t control = 0; control < fiber.content.controlIds.size(); ++control) {
        const double arc = geometry.controlArc(std::size_t(piece.fiber), control);
        if (arc < low || arc > high) continue;
        if (piece.controlFrom < 0) piece.controlFrom = std::int64_t(control);
        piece.controlTo = std::int64_t(control);
    }
}

struct PathSearch {
    // Satisfied constraints incident to each fiber.
    std::vector<std::vector<std::int64_t>> incident;
    const std::vector<ScaffoldConstraint>* constraints;
    std::set<std::int64_t> blocked;

    // Shortest alternative path from `from` to `to` over satisfied
    // constraints: fewest hops, ties by the total length of the fiber
    // stretches the cycle runs along. The search state is the constraint a
    // fiber was entered by, because the stretch depends on both incident
    // anchors.
    std::vector<std::int64_t> shortest(std::int64_t from, std::int64_t to, const ScaffoldConstraint& closing) const
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

    report = ScaffoldLinkReport{};
    std::vector<ScaffoldConstraint> constraints;
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
        report.maxAnchorDistance = std::max({report.maxAnchorDistance, projectedA.distance, projectedB.distance});

        ScaffoldConstraint constraint;
        constraint.kind = ScaffoldConstraintKind::Manual;
        constraint.link = std::int64_t(index);
        constraint.a = link.a;
        constraint.b = link.b;
        constraint.arcA = projectedA.arc;
        constraint.arcB = projectedB.arc;
        constraint.step = step;
        constraint.weight = options.weight;
        constraint.approved = link.approvalState == "approved";
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
                                         const ScaffoldSolveOptions& options)
{
    if (!geometry::windingMinCostFlowAvailable())
        throw std::runtime_error("scaffold winding: vc_object_store was built without LEMON (VC_OBJECT_STORE_WITH_LEMON)");
    if (options.maxAlternativePaths < 0)
        throw std::invalid_argument("scaffold winding: the alternative path cap must be non-negative");
    const auto started = std::chrono::steady_clock::now();
    const auto nodes = graph.fibers.size();

    ScaffoldSolveResult result;
    result.constraints = scaffoldLinkConstraints(graph, geometry, options.links, result.linkReport);
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
    const auto solved = solveComponents(nodes, aggregateWindingArcs(contracted), options.flow, &result.quantization);
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
    std::map<std::tuple<std::int64_t, std::int64_t, std::int64_t>, std::size_t> pieceIndex;
    for (std::size_t i = 0; i < result.constraints.size(); ++i) {
        if (residual[i] == 0) continue;
        ++result.violatedConstraints;
        const auto& constraint = result.constraints[i];
        ScaffoldResidual entry;
        entry.constraint = std::int64_t(i);
        entry.residual = residual[i];
        entry.cost = constraint.weight * residual[i];
        search.blocked.clear();
        const auto path = search.shortest(constraint.a, constraint.b, constraint);
        if (!path.empty()) {
            entry.cycleHops = std::int64_t(path.size()) + 1;
            entry.pieces = cyclePieces(graph, geometry, result.constraints, std::int64_t(i), path);
            for (const auto& piece : entry.pieces) entry.cyclePieceLength += piece.length();
            entry.joint = entry.cost / double(entry.cycleHops);
            entry.redundancy = 1;
            auto blocking = path;
            for (std::int64_t round = 1; round < options.maxAlternativePaths; ++round) {
                for (const auto edge : blocking) search.blocked.insert(edge);
                blocking = search.shortest(constraint.a, constraint.b, constraint);
                if (blocking.empty()) break;
                ++entry.redundancy;
            }
            // A violated constraint accuses the stretches its cycle runs
            // along, a short stretch more specifically than a long one: its
            // cost is split in proportion to 1 / max(length, 1 voxel).
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
