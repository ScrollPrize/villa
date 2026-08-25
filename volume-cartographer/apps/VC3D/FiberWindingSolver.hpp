#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Global winding assignment for unrolled fibers, from H-vs-V crossing evidence.
//
// Every fiber arrives with its unwrapped angle theta about the umbilicus (own
// arbitrary 2*pi gauge), radius and z per sample. The solver assigns each fiber
// an integer turn offset k so that the winding coordinate
//
//     W = s * theta / (2*pi) + k        (s = global chirality sign)
//
// is consistent across fibers. The evidence, in order of authority:
//
//  - Papyrus structure: horizontal fibers lie on the front of the sheet,
//    verticals behind, so an H fiber on the SAME winding as a V fiber passes
//    between that V fiber and the umbilicus. An H fiber crossing a V fiber's
//    angular position *inside* it (smaller radius) is therefore on the same
//    winding or further inward (W_h <= W_v, weak); crossing *outside* means
//    strictly outward (W_h >= W_v + 1). A crossing whose radial separation is
//    within the sheet-thickness tie band is same-winding evidence (equality):
//    adjacent windings sit a wrap apart, same-winding H/V pairs a sheet
//    thickness apart, and sheet thickness is the one radial scale that stays
//    roughly uniform on a crumpled scroll. No pitch model exists anywhere in
//    this solver: the scroll is too non-uniform for one, so radius is only
//    ever used locally and ordinally.
//  - Links: annotated same-crossing ties between two fibers' control points,
//    an integer equality on k with confidence from the angular residual.
//  - Local radial ordering: along any ray from the umbilicus the windings stay
//    radially ordered even when crumpling destroys their spacing. This never
//    forms a constraint; it breaks the gauge freedom the constraints leave
//    (slack nodes, islands).
//
// Crossings are detected as transversal intersections of the two polylines in
// (theta, z) surface coordinates over every 2*pi translate that can meet the V
// fiber's lift, so the integer turn gap n at a crossing is exact by
// construction; there is no angular residual to round. (Links do carry a real
// residual - their endpoints are separately annotated - which is why they have
// a residual-based confidence and crossings do not.)
//
// All constraints are integer difference constraints k_u - k_w >= c.
// Infeasibility is a positive-weight cycle; repair removes, per detected
// cycle, the constraint with the lowest confidence / (1 + times already seen
// in a cycle) - the discount is what lets a wrong link (whose residual, and
// hence confidence, looks clean) lose to several correct crossings instead of
// eliminating them one by one. Every dropped constraint is reported.
//
// The feasible solution is packed densest from below (longest-path
// tightening), then nodes with slack take the value in their feasible interval
// that best matches local radial ordering. Components unreachable from the
// primary component through constraints are anchored rigidly by the same
// ordinal cost, largest first; components with no anchored neighbours at all
// are reported unresolved rather than guessed.
namespace vc3d::fiber_map::winding
{

struct FiberTrace {
    char hvTag = '?';
    // At least one span of this fiber was traced by the fiber model. A fiber
    // with none is pure control-point interpolation: its line geometry - and
    // with it the unwrapped angle, which accumulates along that geometry -
    // can be off by whole turns between controls. Untrusted fibers still
    // take part in the solve (their crossings are the only evidence placing
    // them), but their evidence is attenuated so it loses conflicts against
    // model-traced geometry, and no winding error is ever declared over it:
    // a contradiction involving an untrusted fiber is expected interpolation
    // noise, not an annotation mistake worth flagging.
    bool trusted = true;
    // Parallel arrays over the fiber's visible (control-point-bounded) domain.
    std::vector<double> theta;
    std::vector<double> radius;
    std::vector<double> z;
};

// One deduped link; point indices index the FiberTrace arrays.
struct LinkInput {
    std::size_t fiberA = 0;
    std::size_t pointA = 0;
    std::size_t fiberB = 0;
    std::size_t pointB = 0;
};

// Lengths in voxels, like the layout's own parameters. Defaults are the
// physical intents at 2.4 um/voxel; callers that know the voxel size convert
// the intents themselves.
struct SolverParams {
    // Same-winding tie band on |r_h - r_v| at a crossing. Intent: ~0.03 cm,
    // a few papyrus sheet thicknesses. Never pitch-relative.
    double tieBandVx = 125.0;
    // Samples closer to the umbilicus than this are angularly ill-conditioned
    // and take part in no crossing. Intent: 0.1 cm.
    double minUmbilicusRadiusVx = 417.0;
    // A single polyline step swinging more than this many turns risks the
    // wrong homotopy class; the segment is gated out of crossing detection.
    double maxStepTurns = 0.25;
    // |sin| of the crossing angle below which a pass is tangential, not
    // transversal, and yields no constraint.
    double minTransversality = 0.05;
    // Crossings of one (H, V, n) triple within this z distance are one
    // physical traversal seen by several segment pairs. Intent: 0.2 cm.
    double zMergeVx = 833.0;
    // Neighbourhood for the local radial-ordering cost. Intent: 0.5 cm each.
    double neighborhoodZVx = 2083.0;
    double neighborhoodArcVx = 2083.0;
    // A crumpled sheet's radius drifts with z and with angle; two samples a
    // |dz| and an arc apart only assert a strict radial order when |dr|
    // clears the tie band plus these slope allowances times the separations.
    // Pairs in between carry no information.
    double radialSlopePerZVx = 1.0;
    double radialSlopePerArcVx = 0.3;
    // Second-best anchoring cost within this of the best marks the island
    // ambiguous (violation-count units).
    double anchorAmbiguityMargin = 2.0;
    // Link residual (turns) at and beyond which a link's confidence is zero;
    // also the layout's suspect threshold.
    double linkSuspectTurns = 0.25;
    // Confidence multiplier for evidence (crossings and links) touching an
    // untrusted fiber, so trusted geometry wins repair conflicts.
    double untrustedConfidenceFactor = 0.5;
    // A dropped crossing is only a declared winding error when the final map
    // violates it by at least this many windings. Measured violations are
    // bimodal at 0 and 1, so anything between the modes works.
    double declarationViolationTurns = 0.5;
    // 0 = infer from the data; +1 / -1 force the winding direction.
    int chiralityOverride = 0;
};

enum class CrossingKind { Inside, Outside, Tie };
enum class CrossingStatus { Used, Dropped };

struct Crossing {
    std::size_t hFiber = 0;
    std::size_t vFiber = 0;
    // Both fibers trusted: only such a crossing may be declared a winding
    // error when dropped (an untrusted fiber's contradictions are expected).
    bool declarable = true;
    // How far the FINAL map sits from what this crossing demanded, in
    // windings (0 when satisfied). Greedy cycle repair routinely drops
    // constraints the eventual placement satisfies anyway - the true culprit
    // falls in a later cycle - and such a drop is repair debris, not evidence
    // of a winding error at this spot. Declarations gate on this.
    double violationTurns = 0.0;
    // Position of the traversal, for markers: z and the H fiber's own-gauge
    // psi (= s * theta) at the intersection.
    double zVx = 0.0;
    double psiH = 0.0;
    // Exact integer turn gap between the two gauges at the intersection.
    long long n = 0;
    // r_h - r_v at the intersection.
    double deltaR = 0.0;
    double confidence = 0.0;
    int mergedCount = 1;
    CrossingKind kind = CrossingKind::Inside;
    CrossingStatus status = CrossingStatus::Used;
};

enum class ComponentAnchor {
    // In the primary constraint component: placed by crossings/links. The
    // primary component is the largest one that actually carries a crossing
    // constraint (a link-only network, however large, proves no winding).
    Primary,
    // Island shifted onto the primary map by local radial ordering. A later
    // island can anchor onto an earlier ambiguous one and still read Radius:
    // ambiguity is per-island evidence, not inherited down the chain.
    Radius,
    // Radius-anchored, but a second shift scored within the ambiguity margin.
    AmbiguousRadius,
    // No anchored neighbours anywhere near: own gauge, not comparable.
    Unresolved,
};

struct Placement {
    // Integer turn offset k (stored as double for the caller's arithmetic).
    double turns = 0.0;
    ComponentAnchor anchor = ComponentAnchor::Unresolved;
    bool linked = false;
    bool sheetDriftSuspect = false;
    // W range over the fiber's samples, after everything.
    double windingLo = 0.0;
    double windingHi = 0.0;
};

struct SolveResult {
    int chirality = 1;
    std::vector<Placement> placements;
    // Every surviving traversal (post-merge representatives) plus every
    // dropped one, in deterministic order.
    std::vector<Crossing> crossings;
    // Indices into the input link list whose constraints were dropped by
    // cycle repair.
    std::vector<std::size_t> droppedLinks;
    // Per input link: residual in turns after placement (|.| of the gauge
    // disagreement), for suspect marking by the caller. Infinity for a link
    // that never took part (an endpoint out of range or on a degenerate
    // trace), so it can never read as a perfect link.
    std::vector<double> linkTurnErrors;
    int islandCount = 0;
    int unresolvedCount = 0;
    int tieCount = 0;
    int droppedCrossingCount = 0;
    // Segment-level gate tallies, for the build summary.
    int gatedSegmentCount = 0;
    int tangentialCount = 0;
};

[[nodiscard]] SolveResult solveWindings(const std::vector<FiberTrace>& fibers,
                                        const std::vector<LinkInput>& links,
                                        const SolverParams& params);

} // namespace vc3d::fiber_map::winding
