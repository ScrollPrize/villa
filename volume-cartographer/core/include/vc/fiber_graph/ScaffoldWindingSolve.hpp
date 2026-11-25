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
// time with its own gauge (minimum winding 0). a sign constraint enters the
// same sum with a one-sided term: weight_e * max(0, target_e - (k_b - k_a))
// for "at least", and the mirror for "at most".
//
// A violated constraint alone does not say which annotation is wrong: the
// winding is overdetermined only along cycles. For each violated constraint
// the analysis therefore finds the shortest cycle through it over satisfied
// constraints, the fiber stretches ("pieces") that cycle runs along, and how
// many edge-disjoint alternatives exist. That is what the approval dock
// orders by: smallest cycle first, then cost.
#include "vc/fiber_graph/ScaffoldGraph.hpp"
#include "vc/fiber_graph/ScaffoldRunMonitor.hpp"
#include "vc_object_store/geometry/FiberWindingFlow.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace vc::fiber_graph {

// Aggregated L1 arc as the flow solver takes it (a < b). `side` 0 is an
// equation; +1 a lower bound (k_b - k_a at least the target), -1 an upper
// bound, each costing only when violated (geometry::windingArcResidual).
struct WindingArc {
    std::int64_t a = 0, b = 0, target = 0;
    double weight = 0;
    int side = 0;
};

// Perpendicular and Parallel: one encounter of two fibers, the geometry's own
// verdict - a measured crossing with its rounded winding separation, or an
// annotation link refined onto the same encounter (`ScaffoldConstraint::link`
// names the source; the link's step replaces the measurement). Enclosure: the
// exact H->V step the signed order of a pair's perpendicular crossings bounds
// from both sides (ScaffoldConstraints.hpp). SignAtLeast and SignAtMost:
// the bound one signed perpendicular crossing alone states - the step
// k_b - k_a is at least, or at most, the target. a sign constraint costs
// only when the solve breaks the bound; it is not an equation.
enum class ScaffoldConstraintKind : int {
    Perpendicular = 1, Parallel = 2, Enclosure = 3, SignAtLeast = 4, SignAtMost = 5
};
inline constexpr std::size_t kScaffoldConstraintKinds = 6;

const char* scaffoldConstraintKindName(ScaffoldConstraintKind);
// 0 for an equation, +1 for SignAtLeast, -1 for SignAtMost.
int scaffoldConstraintSide(ScaffoldConstraintKind);

// One asserted winding relation, before aggregation, with the anchors it was
// asserted at: the residual analysis needs them to name the pieces a cycle
// runs along.
struct ScaffoldConstraint {
    ScaffoldConstraintKind kind = ScaffoldConstraintKind::Perpendicular;
    // The source: an index into ScaffoldGraph::links for a map link (manual
    // or automatic), -1 for a relation the extraction found itself.
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
    // Off solves on the measured constraints alone: the annotation's links
    // assert nothing, which is a run against the crossings only. Empty
    // approval states remain an error, so a misconfigured state list cannot
    // silently become a linkless run.
    bool useLinks = true;
    // Record approval states to use; the Python stage's default is approved only.
    std::vector<std::string> approvalStates{"approved"};
    bool includeAuto = false;
    // Anchors farther than this from their fiber polyline fail the run: a
    // coordinate-frame guard, not a tolerance to widen silently.
    double maxAnchorDistance = 5.0;
    double weight = 1.0;
    // A link is refined onto the encounter of its two fibers with the
    // crossing extraction's own refinement and acceptance rules (angle
    // tolerance, fiber support on both sides), seeded at its anchors. This is
    // the farthest the two refined points may lie apart, in voxels.
    double refineMaximumDistance = 512.0;
    // A measured crossing of the same pair within this arc distance of a
    // refined link, on both fibers, is the same encounter: the link is kept
    // and the crossing's own relation is dropped.
    double deduplicateArc = 64.0;
};

struct ScaffoldLinkReport {
    std::int64_t links = 0, used = 0, ordinary = 0, adjacent = 0, skippedApprovalState = 0, skippedAutomatic = 0,
                 skippedEndpoint = 0, skippedSelfLink = 0, skippedAdjacentSameFamily = 0, skippedMissingAnchor = 0,
                 skippedUntagged = 0;
    // Links the record approved whose covering pieces are not both approved
    // for geometry: they are used, but they are not held rigid.
    std::int64_t approvedWithoutPieces = 0;
    double maxAnchorDistance = 0;
    // Links the refinement rejected, by the rule that rejected them: the two
    // fibers do not meet at the angle their families state (or at no clear
    // angle), a fiber ends too close to the encounter to certify it, the
    // refined points lie farther apart than the limit, or another
    // consistency check of the extraction.
    std::int64_t skippedAngle = 0, skippedEndSupport = 0, skippedDistance = 0, skippedConsistency = 0;
    // Links with an anchor on a tail (past the fiber's first or last control
    // point), and links whose anchor or refined point lies on a gap span or
    // too close to one: neither is fiber, so the link asserts nothing.
    std::int64_t skippedTail = 0, skippedGap = 0;
    // The largest arc distance a refinement moved an anchor, in voxels.
    double maxRefinementShift = 0;
    // Measured relations dropped because a refined link states the same
    // encounter, and the links that had one.
    std::int64_t deduplicatedCrossings = 0, linksWithCrossing = 0;
};

// W_b - W_a asserted by an adjacent link, or nullopt when the pair is not one
// H and one V (an annotation error that constrains nothing).
std::optional<std::int64_t> adjacentStep(ScaffoldFamily a, ScaffoldFamily b);

// The map-link constraints of a scaffold, each refined onto the encounter of
// its two fibers; a link the refinement rejects asserts nothing and is
// counted. Throws std::invalid_argument when an anchor lies farther than
// `maxAnchorDistance` from its polyline.
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

// Fill a piece's controlFrom / controlTo from its arcs.
void scaffoldPieceControls(const ScaffoldGraph&, const ScaffoldGeometry&, ScaffoldPiece&);

struct ScaffoldResidual {
    std::int64_t constraint = 0;
    double residual = 0; // |k_b - k_a - target| in windings; for a sign constraint, how far past its bound
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
    // Optional progress and cancellation (ScaffoldRunMonitor.hpp): the solve
    // stops between components and between the cycle searches of the
    // residual analysis.
    ScaffoldRunMonitor* monitor = nullptr;
};

struct ScaffoldSolveResult {
    std::vector<ScaffoldConstraint> constraints;
    // Constraints per kind, indexed by ScaffoldConstraintKind (map links
    // included), and how many of them came from a map link.
    std::array<std::int64_t, kScaffoldConstraintKinds> constraintsByKind{};
    std::int64_t linkConstraints = 0;
    std::vector<WindingArc> arcs; // aggregated, as solved
    std::vector<std::int64_t> winding, component; // per fiber; unconstrained fibers get their own component
    std::int64_t components = 0, isolatedFibers = 0;
    // A fiber with broken geometry is solved as its parts (the stretches of
    // arc between the broken ones), each with a winding of its own. `winding`
    // and `component` hold the longest part; all parts are listed here.
    // Components and isolated fibers count parts.
    struct Part {
        std::int64_t fiber = 0;
        double arcFrom = 0, arcTo = 0;
        std::int64_t winding = 0, component = 0;
        bool main = false;
    };
    std::vector<Part> parts;
    std::int64_t splitFibers = 0;
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
