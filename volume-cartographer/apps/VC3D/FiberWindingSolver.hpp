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
//    angular position *inside* it (dr <= 0) is therefore on the same winding
//    or further inward (W_h <= W_v, weak); crossing *outside* (dr > 0) means
//    strictly outward (W_h >= W_v + 1). The sign of dr is the whole
//    classification - there is no same-winding tie equality, because on a
//    tightly wound scroll the wrap spacing is only a few sheet thicknesses,
//    and a dr-band equality manufactures constraints out of sub-band radial
//    noise. The weak inside form absorbs same-winding contacts at equality;
//    a contact whose noise flips the sign costs a winding of separation,
//    accepted rather than guessed away.
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

// "No sample": for the optional sample indices below.
inline constexpr std::size_t kNoSample = static_cast<std::size_t>(-1);

struct FiberTrace {
    char hvTag = '?';
    // At least one span of this fiber was traced by the fiber model. A fiber
    // with none is pure control-point interpolation: its line geometry - and
    // with it the unwrapped angle, which accumulates along that geometry -
    // can be off by whole turns between controls. Untrusted fibers take part
    // in the solve like any other, but their evidence is attenuated so it
    // loses conflicts against model-traced geometry. Declarations are not
    // gated on trust: the annotation is taken as accurate, and a winding
    // error over an interpolated span points at the span to trace.
    bool trusted = true;
    // Kollesis: where two sheets are glued the outer sheet lies in front of
    // (toward the core from) the inner sheet over the overlap, so the inner
    // sheet's H fibers sit one thickness BEHIND the outer sheet's V fibers
    // there, which the radial rule would read as a whole winding out. An H
    // fiber's end tagged kollesis_termination marks such a seam; a V fiber is
    // on a kollesis when the annotator has linked it to tagged H ends on
    // both sides (derived by the layout, never guessed from geometry). At
    // the seam encounter of a tagged H end with such a V, the crossing is
    // read as "same winding or inward" whatever its radial sign - when the
    // annotator has linked the H fiber to that V (at the crossing, as links
    // are drawn). The tagged ends are the samples of the tagged first / last
    // control point (the trace may run a sample beyond them); kNoSample when
    // untagged.
    std::size_t kollesisStartSample = kNoSample;
    std::size_t kollesisEndSample = kNoSample;
    bool onKollesis = false;
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
    // Sheet-thickness radial scale. Intent: ~0.03 cm, a few papyrus sheet
    // thicknesses; never pitch-relative. Crossings classify purely by the
    // SIGN of deltaR (<= 0 inside, > 0 outside) - there is no tie
    // classification - and this scale only sets the crossing confidence ramp
    // (full confidence at 3x this), the dedup clustering distance, and the
    // ordinal same-winding band for slack and island placement.
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
    // A traversal group (see CrossingGroup) is only eligible when the H
    // trace's stretch at the V fiber's angle is entered and left with this
    // much angular clearance on opposite sides, and a trace end inside that
    // stretch clears the V fiber's angle at its height by the same amount:
    // an H trace cut at the V fiber's angle may have been cut
    // mid-traversal, and its crossing count is then incomplete. Intent: a
    // few hundred voxels of arc at the scroll's radii.
    double endpointClearanceTurns = 0.01;
    // 0 = infer from the data; +1 / -1 force the winding direction.
    int chiralityOverride = 0;
};

// Tie is retained for the constraint/violation switch exhaustiveness but no
// longer produced: classification is by the sign of deltaR alone.
enum class CrossingKind { Inside, Outside, Tie };
// InGroup: the crossing did not constrain on its own; its traversal group did
// (see CrossingGroup), and the group carries the status that matters.
enum class CrossingStatus { Used, Dropped, InGroup };

struct Crossing {
    std::size_t hFiber = 0;
    std::size_t vFiber = 0;
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
    // |sin| of the crossing angle in arc-scaled (psi, z); below
    // params.minTransversality the event is `tangential`: it is counted as a
    // traversal event (N, T below) and reported, but never constrains on its
    // own, exactly as such passes were gated before they were recorded.
    double transversality = 0.0;
    bool tangential = false;
    // Which way the H polyline crosses the V polyline in (psi, z), +1 or -1,
    // both taken in their own polyline order (a V branch re-sorted to
    // ascending z is walked back in its fiber's order). Two events of one
    // pair with opposite orientation are a pass and return (a wobble or a
    // touch), not a traversal; a traversal sums to an odd count.
    int orientation = 0;
    // Provenance: the H segment and its parameter, and the V vertex identity
    // (Branch::vertexId) when the hit is at a V vertex (kNoSample otherwise).
    // A V vertex shared by two branches is detected once per branch; the two
    // are one event when their orientations agree (a straight pass through
    // the vertex) and a `touch` when they oppose: the V fiber comes up to the
    // H fiber at its apex and retraces, which crosses nothing. Touches are
    // recorded and never counted.
    static constexpr std::size_t kNoSample = static_cast<std::size_t>(-1);
    std::size_t hSegment = 0;
    double hT = 0.0;
    std::size_t vSample = kNoSample;
    // The seam encounter of a kollesis-tagged H end with a V fiber on that
    // kollesis: annotation-classified as Inside (see FiberTrace), kept out
    // of the traversal-group counts like a touch.
    bool kollesis = false;
    // A seam encounter read without a tag on this H fiber: on a V fiber the
    // annotator certified as on a kollesis, an Outside crossing the rest of
    // the evidence contradicted by exactly one turn, of an H fiber that ends
    // within a turn past it (see `terminal`) - the glued inner sheet, one
    // thickness behind the outer sheet's V. Implies `kollesis`.
    bool kollesisInferred = false;
    // The H fiber ends within one turn past this crossing, toward one of its
    // ends, without meeting the V fiber again and without leaving the V
    // branch's height range (so a further crossing could not have gone
    // unseen, and no gated or unresolved segment on the way either): how an
    // inner sheet's H fiber ends in a kollesis overlap. A
    // property of the pair's geometry, no length in it. `terminalSides`
    // says which way those ends lie from the crossing in canonical angle:
    // bit 1 the +psi side, bit 2 the -psi side (a short H fiber may end
    // within a turn both ways).
    bool terminal = false;
    int terminalSides = 0;
    // A pass and return at a vertex of either polyline (both incident
    // segments on one side of the other segment), or the two opposing
    // records of a V apex: crosses nothing, counted by no group.
    bool touch = false;
    // The z-monotone V branch the event was found on.
    std::size_t vBranch = 0;
    // The raw detection this record came from (its position in the pair's
    // detection order); unique per detection.
    std::size_t detection = 0;
    // On an event (PairCrossings::events): the representative in
    // PairCrossings::crossings that stands for it in the legacy constraint
    // path. On a representative: unused.
    std::size_t representative = 0;
    // On a representative: every detection it stands for belongs to a
    // traversal group with a verdict, so the group constrains in its place
    // (status InGroup in the solve). A representative standing for both
    // covered and uncovered detections constrains for the uncovered ones,
    // with its confidence recomputed over them.
    bool coveredByGroups = false;
    // Index into PairCrossings::groups / SolveResult::groups, or -1. On a
    // representative: the group of its own detection, for display.
    long long groupIndex = -1;
};

// The crossings of one (H, V) pair on one 2*pi translate n, read together.
//
// Each crossing compares radii at one point where the two polylines share
// (psi, z). Where the sheet folds, one pair produces several such points on
// one translate with contradictory radial signs, and each sign alone is
// meaningless. What is meaningful is the count: the number of crossings at
// which the H fiber lies radially inside the V fiber is the intersection
// number of the H curve with the "radial curtain" swept from the umbilicus
// out to the V fiber. An odd count means the H fiber passes between the V
// fiber and the umbilicus (same winding or inward, the weak Inside claim);
// an even count means it does not (strictly Outside). Crossings at
// different heights are legitimately one count: they are intersections with
// one surface. This is the user's picture - the V fiber sits inside or
// outside the wiggly H arc - made exact.
//
// The count is taken per z-monotone V branch: a V fiber that folds back in
// height sweeps a curtain that covers the same (theta, z) several times over,
// and events on different limbs are not crossings of one separator. Each
// branch is a graph over z, so its own curtain is single-sheeted; a folded V
// fiber therefore contributes one group per limb, and its limbs' disagreement
// surfaces as it always did.
//
// The verdict is only issued where it can change anything and where the
// count is trustworthy: the signs must be mixed (a uniform group already
// says what its members say, and keeps their individual constraints), the
// signed orientation sum must be odd (a wobble crossing back and forth sums
// to zero and is no traversal), no gated or degenerate geometry may have hid
// an event on that translate, and the H trace must run from one side of the
// V fiber's angular locus to the other with clearance (an H trace cut at the
// V fiber's angle has an incomplete count). Otherwise the members constrain
// individually, as they always did, and the group is reported for
// inspection with the flag that stopped it.
struct CrossingGroup {
    std::size_t hFiber = 0;
    std::size_t vFiber = 0;
    long long n = 0;
    std::size_t vBranch = 0;
    // Indices into the EVENT list this group belongs to (PairCrossings::events
    // in the shard, SolveResult::events in the solve), touches excluded.
    std::vector<std::size_t> members;
    // Counted events: one per resolved crossing, touches excluded.
    int multiplicity = 0;
    // N: events with r_h < r_v. T: sum of orientations. J: T over N's events.
    int insideCount = 0;
    int orientationSum = 0;
    int insideOrientationSum = 0;
    bool mixedSigns = false;
    // Eligibility diagnostics (see above).
    bool coverageGap = false;
    bool unresolved = false;
    bool onCurtain = false;
    // The H trace's stretch at the branch's angle runs from one side of the
    // angular window to the other with clearance and stays within the
    // branch's height range throughout: the count is a complete traversal's.
    bool traversalCovered = false;
    // Smallest |deltaR| over the events: the margin the verdict hangs on.
    double minAbsDeltaR = 0.0;
    double meanTransversality = 0.0;
    // hasVerdict: the group constrains in place of its members.
    bool hasVerdict = false;
    CrossingKind verdict = CrossingKind::Inside;
    double confidence = 0.0;
    CrossingStatus status = CrossingStatus::Used;
    double violationTurns = 0.0;
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
    // Every resolved event (see PairCrossings::events), in shard order, with
    // hFiber/vFiber bound and `representative` indexing `crossings`; its
    // status mirrors its representative's.
    std::vector<Crossing> events;
    // Traversal groups in shard order; members index `events`. A group with
    // hasVerdict constrained in place of its members' representatives (their
    // status is InGroup).
    std::vector<CrossingGroup> groups;
    int droppedGroupCount = 0;
    // Owner-segment pairs that were exactly parallel, summed over pairs.
    int unresolvedIntersectionCount = 0;
    // Events read as kollesis seam encounters.
    int kollesisCrossingCount = 0;
    // Of those, read from the solve contradiction rather than a tag.
    int kollesisInferredCount = 0;
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
    // Phase timings (milliseconds): crossing detection + merge, and
    // everything after (constraints, repair, packing, ascent, islands).
    double detectMs = 0.0;
    double solveMs = 0.0;
};

[[nodiscard]] SolveResult solveWindings(const std::vector<FiberTrace>& fibers,
                                        const std::vector<LinkInput>& links,
                                        const SolverParams& params);

// ---------------------------------------------------------------------------
// The detection stage, exposed piecewise so a caller can memoize it per
// (H, V) fiber pair: detection is pair-local by construction - no global
// state enters a pair's result - which is what makes the shards cacheable
// and the cached build structurally identical to the fresh one.

// The winding-direction vote, from radii one whole turn apart along each
// fiber (crumpling cancels over a full turn), covariance as the fallback
// when nothing wraps. 0 override = infer.
[[nodiscard]] int inferChirality(const std::vector<FiberTrace>& fibers,
                                 int chiralityOverride);

// A fiber's trace in the solve's canonical frame: psi = chirality * theta -
// 2*pi*gauge, the gauge chosen from the fiber's own median angle. V fibers
// carry their z-monotone branches precomputed. Empty psi = unusable trace.
struct CanonicalTrace {
    char hvTag = '?';
    bool trusted = true;
    std::size_t kollesisStartSample = kNoSample;
    std::size_t kollesisEndSample = kNoSample;
    bool onKollesis = false;
    long long gauge = 0;
    std::vector<double> psi;
    std::vector<double> radius;
    std::vector<double> z;
    struct Branch {
        std::vector<double> psi;
        std::vector<double> z;
        std::vector<double> r;
        // Vertex identity of each branch sample: the id of its run of
        // consecutive original samples with identical (psi, z), so a vertex
        // repeated in the projection - at a fold apex, say - is one vertex
        // to both branches.
        std::vector<std::size_t> vertexId;
        // Walking the branch in its stored (ascending z) order follows the
        // fiber's own polyline order.
        bool forwardAscending = true;
        double psiMin = 0.0;
        double psiMax = 0.0;
    };
    std::vector<Branch> branches;
};
[[nodiscard]] CanonicalTrace canonicalizeTrace(const FiberTrace& fiber,
                                               int chirality);

// The geometry half of a pair's detection, and what the layout's per-pair
// cache stores: every raw detection with its provenance, plus the gate
// tallies and the translates on which an event may have gone unseen. A pure
// function of the two canonical traces and the detection parameters - no
// annotation (link, tag) enters, so an annotation edit never invalidates a
// shard. classifyPairCrossings turns it into the constraints, events and
// groups the solve consumes.
struct PairDetections {
    std::vector<Crossing> raw;
    std::vector<Crossing> shallow;
    std::size_t detectionCount = 0;
    // Sorted, unique.
    std::vector<long long> gapTranslates;
    std::vector<long long> unresolvedTranslates;
    // H segments (by index) on which an encounter with this V may have gone
    // unseen: the segment was gated, met a gated V segment in angle, or
    // shared an unresolved collinear stretch. Sorted, unique.
    std::vector<std::size_t> uncoveredSegments;
    int gatedSegmentCount = 0;
    int tangentialCount = 0;
    int unresolvedCount = 0;
};
[[nodiscard]] PairDetections detectPairCrossings(const CanonicalTrace& h,
                                                 const CanonicalTrace& v,
                                                 const SolverParams& params);
// Field-by-field, bit-exact equality of two detection shards: the test of
// the cache's contract that a cached shard IS the fresh one.
[[nodiscard]] bool identicalPairDetections(const PairDetections& a, const PairDetections& b);

struct PairCrossings {
    // Representatives: the proximity-merged crossings the legacy constraint
    // path is built from, exactly as before.
    std::vector<Crossing> crossings;
    // Every resolved event, one per crossing of the two polylines (exact
    // vertex duplicates collapsed, apex touches kept and flagged), each
    // pointing at its representative. The counts and the inspection records
    // are over these.
    std::vector<Crossing> events;
    // Every (translate, V branch) with at least two counted events, verdict
    // or not.
    std::vector<CrossingGroup> groups;
    int gatedSegmentCount = 0;
    int tangentialCount = 0;
    // Owner segments that were exactly parallel: an intersection the
    // detector cannot place. Disables the verdict on the translates it
    // touched (recorded on the groups).
    int unresolvedCount = 0;
};
// One tagged end of an H fiber, to be read against a V fiber on a kollesis,
// through the link the annotator drew between the two: the tagged control's
// sample on the H trace, and the link's samples on the H and V traces (the
// link nearest the tagged end when the pair is linked more than once). The
// link names the encounter: the V limbs holding the linked V sample's
// vertex (two at a fold apex), on the translate that lifts the linked H
// sample onto it. Where none of those limbs has a detection on that
// translate (the annotator linked the V's nearest control, which sat on a
// fold the H never reaches), the encounter is the pair's detection that
// lifts the linked H sample best and sits nearest it along the H fiber,
// whatever height the linked control is at. A detection on the linked limb on that translate, however far
// along the H fiber, keeps the link in charge; a crossing of that limb on
// another translate does not (it is another turn's encounter).
struct SeamAnchor {
    std::size_t hSample = kNoSample;
    std::size_t hLinkSample = kNoSample;
    std::size_t vSample = kNoSample;
};
// The seam anchors of the pair (h, v): one per tagged end of the H trace
// that is linked to this V - empty unless the V is on a kollesis and the
// pair is linked. A tagged H fiber is never read against a V it is not
// linked to. `traces` and `links` index fibers the way hIndex / vIndex do.
[[nodiscard]] std::vector<SeamAnchor> seamAnchors(const std::vector<CanonicalTrace>& traces,
                                                  std::size_t hIndex, std::size_t vIndex,
                                                  const std::vector<LinkInput>& links);
// The classification half, run at solve time (never cached): the proximity
// merge into representatives, the resolved events, the traversal groups -
// and the readings that depend on annotation: the pair's seam anchors (see
// SeamAnchor), which the caller derives from the traces' kollesis fields and
// the links. Deterministic in its inputs, so fresh and cached builds
// classify identically.
// `inferredSeams`: detection ids (Crossing::detection) the caller has found
// to be seam encounters by the solve itself - on a V on a kollesis, an
// Outside crossing contradicted by one turn whose H fiber is `terminal`
// there - read Inside like a tagged encounter and flagged kollesisInferred.
// The layout supplies them from a first solve and solves again; the plain
// solveWindings overload never infers.
[[nodiscard]] PairCrossings classifyPairCrossings(const PairDetections& detections,
                                                  const CanonicalTrace& h,
                                                  const CanonicalTrace& v,
                                                  const std::vector<SeamAnchor>& seams,
                                                  const std::vector<std::size_t>& inferredSeams,
                                                  const SolverParams& params);

// Field-by-field, bit-exact equality of two classified shards.
[[nodiscard]] bool identicalPairCrossings(const PairCrossings& a, const PairCrossings& b);

// A detection shard bound to the current build's fiber indices.
struct PairDetection {
    std::size_t hFiber = 0;
    std::size_t vFiber = 0;
    const PairCrossings* detection = nullptr;
};

// The solve over externally supplied detection shards (cached or fresh).
// Shards may arrive in any order; they are assembled in canonical
// (hFiber, vFiber) order, which reproduces the plain overload's constraint
// order exactly.
[[nodiscard]] SolveResult solveWindings(const std::vector<FiberTrace>& fibers,
                                        const std::vector<LinkInput>& links,
                                        const SolverParams& params,
                                        int chirality,
                                        const std::vector<PairDetection>& detections);

} // namespace vc3d::fiber_map::winding
