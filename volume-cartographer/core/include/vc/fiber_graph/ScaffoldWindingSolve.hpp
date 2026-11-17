#pragma once
// Integer winding assignment for a loaded scaffold, and the residual analysis
// the approval workflow ranks by.
//
// Every constraint asserts an integer winding step between two fibers at two
// anchor positions. The whole-fiber step adds the seam and connector
// transport, exactly as the automatic measured constraints do:
//
//     target = step + connector(x_a, x_b) - turns_b(arc_b) + turns_a(arc_a)
//
// The solve minimizes sum_e weight_e * |k_b - k_a - target_e| exactly, as a
// capacitated minimum-cost circulation (LEMON), one connected component at a
// time with its own gauge (minimum winding 0).
//
// A violated constraint alone does not say which annotation is wrong: the
// winding is overdetermined only along cycles. For each violated constraint
// the analysis therefore finds the shortest cycle through it over satisfied
// constraints, the fiber stretches ("pieces") that cycle runs along, and how
// many edge-disjoint alternatives exist. That is what the approval dock
// orders by: smallest cycle first, then cost.
#include "vc/fiber_graph/ScaffoldGraph.hpp"
#include "vc_object_store/geometry/FiberWindingFlow.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace vc::fiber_graph {

// Aggregated L1 arc as the flow solver takes it (a < b).
struct WindingArc {
    std::int64_t a = 0, b = 0, target = 0;
    double weight = 0;
};

// Manual: an annotation link. Perpendicular and Parallel: a measured
// crossing's rounded winding separation. Enclosure: the exact H->V step the
// signed order of a pair's perpendicular crossings bounds from both sides
// (ScaffoldConstraints.hpp).
enum class ScaffoldConstraintKind : int { Manual = 0, Perpendicular = 1, Parallel = 2, Enclosure = 3 };

const char* scaffoldConstraintKindName(ScaffoldConstraintKind);

// One asserted winding relation, before aggregation, with the anchors it was
// asserted at: the residual analysis needs them to name the pieces a cycle
// runs along.
struct ScaffoldConstraint {
    ScaffoldConstraintKind kind = ScaffoldConstraintKind::Manual;
    // Index into ScaffoldGraph::links for a manual link, -1 otherwise.
    std::int64_t link = -1;
    std::int64_t a = 0, b = 0;      // fiber indices, a < b
    double arcA = 0, arcB = 0;      // anchor arc lengths on a and b
    std::int64_t step = 0;          // the anchor-level assertion (0, or +-1 adjacent)
    std::int64_t target = 0;        // the whole-fiber step
    double weight = 0;
    // Approved: a link whose record approval state is approved, or a
    // measured constraint with a live approval row
    // (vc_object_store FiberConstraintApprovalExtension.hpp).
    bool approved = false;
    // The crossing a measured constraint came from (an index into the event
    // list it was assembled from), -1 for a manual link.
    std::int64_t event = -1;
    // The measured separation in turns; NaN for a manual link and for a
    // relation no sampler measured.
    double windingDistance = std::numeric_limits<double>::quiet_NaN();
};

struct ScaffoldLinkArcOptions {
    // Record approval states to use; the Python stage's default is approved only.
    std::vector<std::string> approvalStates{"approved"};
    bool includeAuto = false;
    // Anchors farther than this from their fiber polyline fail the run: a
    // coordinate-frame guard, not a tolerance to widen silently.
    double maxAnchorDistance = 5.0;
    double weight = 1.0;
};

struct ScaffoldLinkReport {
    std::int64_t links = 0, used = 0, ordinary = 0, adjacent = 0, skippedApprovalState = 0, skippedAutomatic = 0,
                 skippedEndpoint = 0, skippedSelfLink = 0, skippedAdjacentSameFamily = 0, skippedMissingAnchor = 0,
                 skippedUntagged = 0;
    double maxAnchorDistance = 0;
};

// W_b - W_a asserted by an adjacent link, or nullopt when the pair is not one
// H and one V (an annotation error that constrains nothing).
std::optional<std::int64_t> adjacentStep(ScaffoldFamily a, ScaffoldFamily b);

// The manual-link constraints of a scaffold. Throws std::invalid_argument when
// an anchor lies farther than `maxAnchorDistance` from its polyline.
std::vector<ScaffoldConstraint> scaffoldLinkConstraints(const ScaffoldGraph&, const ScaffoldGeometry&,
                                                        const ScaffoldLinkArcOptions&, ScaffoldLinkReport&);

// Sum the weights of identical (a, b, target) relations.
std::vector<WindingArc> aggregateWindingArcs(std::vector<WindingArc>);

// One fiber stretch a cycle runs along: the fiber between the two constraints
// that enter and leave it.
struct ScaffoldPiece {
    std::int64_t fiber = 0;
    double arcFrom = 0, arcTo = 0;
    // The constraints bounding the stretch (indices into
    // ScaffoldSolveResult::constraints).
    std::int64_t constraintFrom = -1, constraintTo = -1;
    // The controls bracketing [arcFrom, arcTo] outward: the stretch covers
    // the spans [controlFrom, controlTo - 1], and a piece opened between
    // these two controls contains it. -1 for a fiber without controls.
    std::int64_t controlFrom = -1, controlTo = -1;
    double length() const { return std::abs(arcTo - arcFrom); }
};

struct ScaffoldResidual {
    std::int64_t constraint = 0;
    double residual = 0; // |k_b - k_a - target| in windings
    double cost = 0;     // weight * residual
    // Hops of the shortest cycle through this constraint over satisfied
    // constraints, -1 when the two fibers are not otherwise connected (the
    // violation is then not localized by any cycle).
    std::int64_t cycleHops = -1;
    double cyclePieceLength = 0;
    // Edge-disjoint alternative paths found, up to the option's cap.
    std::int64_t redundancy = 0;
    // cost / cycleHops: the per-hop suspicion the dock can order by.
    double joint = 0;
    std::vector<ScaffoldPiece> pieces;
};

struct ScaffoldPieceScore {
    ScaffoldPiece piece;
    // Sum over the violated constraints whose shortest cycle runs along this
    // piece of their cost, split by 1/length over equally short cycles.
    double score = 0;
    std::int64_t cycleHops = -1;
    std::int64_t constraints = 0;
};

struct ScaffoldSolveOptions {
    ScaffoldLinkArcOptions links;
    geometry::WindingFlowOptions flow;
    // Solve the approved constraints alone first and keep their relative
    // windings fixed by contracting each approved subgraph into one node.
    bool fixApprovedSubgraphs = true;
    // Cap on the edge-disjoint alternative paths counted per violated
    // constraint.
    std::int64_t maxAlternativePaths = 4;
    std::size_t workers = 1;
};

struct ScaffoldSolveResult {
    std::vector<ScaffoldConstraint> constraints;
    // Constraints per kind, indexed by ScaffoldConstraintKind.
    std::array<std::int64_t, 4> constraintsByKind{};
    std::vector<WindingArc> arcs; // aggregated, as solved
    std::vector<std::int64_t> winding, component; // per fiber; unconstrained fibers get their own component
    std::int64_t components = 0, isolatedFibers = 0;
    std::vector<ScaffoldResidual> residuals; // violated constraints, worst cost first
    std::vector<ScaffoldPieceScore> pieces;  // highest score first
    double objective = 0, maxResidual = 0;
    std::int64_t violatedConstraints = 0, violatedArcs = 0;
    std::int64_t approvedSubgraphs = 0, contractedFibers = 0;
    geometry::WindingFlowQuantization quantization;
    double seconds = 0;
    ScaffoldLinkReport linkReport;
};

// Throws std::runtime_error when the library was built without LEMON, and
// std::invalid_argument when an approved subgraph cannot be satisfied (the
// message names the violated approved constraints so they can be revoked).
// `measured` holds the constraints of the measured crossings
// (ScaffoldConstraints.hpp), which the solve treats exactly like the links:
// same aggregation, same contraction of the approved ones, same residual
// analysis.
ScaffoldSolveResult solveScaffoldWinding(const ScaffoldGraph&, const ScaffoldGeometry&,
                                         const ScaffoldSolveOptions& = {},
                                         std::span<const ScaffoldConstraint> measured = {});

} // namespace vc::fiber_graph
