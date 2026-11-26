#pragma once
// Measured winding evidence for a scaffold: the crossings of its annotated
// fibers, the whole-fiber normal signs those crossings decide, and the integer
// winding relations they assert.
//
// The manual links of ScaffoldWindingSolve.hpp are one kind of constraint; a
// crossing of two fibers is another. Two H fibers (or two V fibers) that run
// side by side are a *parallel* relation, an H and a V fiber that cross are a
// *perpendicular* one, and the signed order of a perpendicular crossing - which
// of the two lies further out - also bounds the H->V step of the pair exactly,
// the *enclosure* relation. All three are assembled exactly as the automatic
// pipeline assembles them (vc_object_store/geometry/FiberWindingGraph.hpp:
// windingEventRelation), so a scaffold run and a traced run speak one language.
//
// The geometry comes from the scaffold's own polylines
// (extractFiberTraceConstraints with preserveInputLinesAsPieces), the normals
// from ScaffoldNormals, and the winding separation along a connector from a
// sampler the caller supplies - VC3D the package's Lasagna normal dataset, the
// CLI a dataset it opens. A relation without a measured separation is kept: it
// still witnesses an enclosure.
#include "vc/fiber_graph/ScaffoldNormals.hpp"
#include "vc/fiber_graph/ScaffoldRunMonitor.hpp"
#include "vc/fiber_graph/ScaffoldWindingSolve.hpp"
#include "vc/fiber_tracer/FiberNormalSign.hpp"
#include "vc/fiber_tracer/FiberRibbonCrossings.hpp"
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include "vc_object_store/objects/FiberConstraintApprovalExtension.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <limits>
#include <span>
#include <vector>

namespace vc::fiber_graph {

// One measured crossing of two scaffold fibers, in the fibers' own normal
// gauges (the whole-fiber signs are applied later, by the assembly).
struct ScaffoldEvent {
    std::int64_t a = 0, b = 0; // fiber indices in graph order
    double arcA = 0, arcB = 0;
    geometry::Point pointA{}, pointB{};
    geometry::Point normalA{}, normalB{};
    bool validA = false, validB = false, validConnector = false;
    // The geometry's own verdict: the two fibers run side by side.
    bool parallel = false;
    // Separation along the connector in turns; NaN when none was measured.
    double distance = std::numeric_limits<double>::quiet_NaN();
    double closestDistance = 0; // voxels between the two anchors
    double parallelScore = 0, perpendicularScore = 0;
    // What the two fibers' normal surfaces find about a perpendicular
    // crossing (vc/fiber_tracer/FiberRibbonCrossings.hpp), in the form the
    // traced volume's extraction stores with its constraints.
    std::uint8_t crossingFindings = 0;
    double crossingAlternative = 0;
};

struct ScaffoldEventOptions {
    double resampleSpacing = 32.0;
    double maximumDistance = 128.0;
    double tangentWindow = 32.0;
    // Accept crossings near a fiber's ends, which the automatic pipeline
    // certifies against the trace's own endpoint support. A hand-drawn fiber
    // ends where the annotator stopped, so relaxing it keeps the crossings
    // there; off is the automatic extraction's own mode and the one a traced
    // run compares against.
    bool relaxEndpointCertification = false;
    // Measure the winding separation of every accepted crossing. Off keeps
    // the relations with a NaN separation: enclosure evidence only.
    bool measureWindingDistances = true;
    double windingIntegrationStep = 8.0;
    // Reject a crossing whose measured separation exceeds this many turns.
    // Off by default, as in the automatic extraction: a sparse scaffold has
    // real crossings several windings apart, and a rejected one is evidence
    // thrown away rather than a bad measurement caught.
    double maximumWindingDistance = 1.5;
    bool enforceMaximumWindingDistance = false;
    // Follow every perpendicular crossing through its two fibers' normal
    // surfaces and record what is found (ScaffoldEvent::crossingFindings),
    // with the traced volume's own surface and matching parameters.
    bool crossingFindings = true;
    fiber_tracer::FiberRibbonOptions crossingSurface = [] {
        fiber_tracer::FiberRibbonOptions surface;
        surface.halfHeight = 512.0;
        surface.maxTangentCosine = 1.0;
        return surface;
    }();
    fiber_tracer::FiberCrossingFilterOptions crossingFilter;
    double crossingEncounterArcGap = 3000.0;
    // Arc spacing of the strip normals the surfaces are built from, voxels.
    double crossingSampleSpacing = 16.0;
    std::size_t workers = 0; // 0: the extractor's own default
    // Two fiber sets in one graph: the fibers [0, crossQueryPrefix) are
    // searched only against the others, never among themselves or the others
    // among themselves (the extractor's bipartite mode). Zero: every pair.
    std::size_t crossQueryPrefix = 0;
    // The normal of a fiber at an arc, replacing the ScaffoldNormals lookup:
    // for a graph whose fibers take their normals from more than one source.
    // Events extracted this way are not cached.
    std::function<std::optional<geometry::Point>(std::size_t fiber, double arc)> normalAt;
    // Optional progress and cancellation. The extraction itself is one call
    // into the tracer, so a run can be taken back before and after it, not
    // during it.
    ScaffoldRunMonitor* monitor = nullptr;
    // Reuse: a pair whose two fibers and parameters are already cached is not
    // extracted again (the search skips it), and what this run extracted is
    // written into `sink` for the caller to commit.
    const class ScaffoldCache* cache = nullptr;
    struct ScaffoldCacheSink* sink = nullptr;
};

struct ScaffoldEventReport {
    // Pairs whose crossings came from the cache, and the events they
    // contributed; pairs this run extracted and wrote back.
    std::int64_t reusedPairs = 0, reusedEvents = 0, cachedPairs = 0;
    std::int64_t lines = 0, shortLines = 0, resampledPoints = 0;
    std::int64_t events = 0, perpendicular = 0, parallel = 0;
    std::int64_t withoutNormals = 0, withDistance = 0;
    std::int64_t rejectedTangents = 0, rejectedWinding = 0, rejectedWindingCutoff = 0;
    // Crossings dropped because one of their two points lies on a gap span
    // or too close to one to be certified.
    std::int64_t rejectedGap = 0;
    // Perpendicular crossings followed through the normal surfaces, and those
    // with a finding beyond the single matching passage.
    std::int64_t crossingsAsked = 0, crossingsWithFindings = 0;
    double seconds = 0;
};

// Winding separations along connectors, in turns. Required when
// `measureWindingDistances` is set: the library substitutes no guess.
using ScaffoldWindingDistances = fiber_tracer::FiberTraceWindingDistanceBatch;

// The crossings of a scaffold. Throws std::invalid_argument when a separation
// was asked for and no sampler was given.
std::vector<ScaffoldEvent> scaffoldEvents(const ScaffoldGraph&, const ScaffoldGeometry&, const ScaffoldNormals&,
                                          const ScaffoldEventOptions&, const ScaffoldWindingDistances&,
                                          ScaffoldEventReport&);

// The normals of `events` looked up again and their crossing findings
// followed again: after the normals changed (ScaffoldNormals::face), the
// crossings themselves being a matter of the geometry alone. With `changed`
// (one flag per fiber: its normals changed) only the events of those fibers
// are refreshed; the others keep what they carry.
void scaffoldRefreshEventNormals(const ScaffoldGraph&, const ScaffoldGeometry&, const ScaffoldNormals&,
                                 const ScaffoldEventOptions&, std::vector<ScaffoldEvent>&, ScaffoldEventReport&,
                                 const std::vector<char>* changed = nullptr);

struct ScaffoldSignOptions {
    fiber_tracer::FiberSignOptions sign;
    // Decide the gauge of the fibers no anchor fixed, from the perpendicular
    // crossings. Off leaves them at their geometry gauge, which only means
    // something when every fiber carries an anchor.
    bool solve = true;
    // Leave out the crossings the shared rule drops while no signs exist
    // (vc_object_store geometry/FiberCrossingFilter.hpp: nothing left, mixed).
    bool crossingFilter = true;
    // Optional progress and cancellation, checked before the solve starts.
    ScaffoldRunMonitor* monitor = nullptr;
};

struct ScaffoldSignResult {
    // Crossings the shared crossing rule left out of the sign solve.
    std::int64_t crossingFiltered = 0;
    // Per fiber: the multiplier that turns its gauge normal into the outward
    // one, as the winding assembly's `normalSign` column.
    std::vector<std::int64_t> sign;
    // Per fiber: what decided that sign.
    std::vector<ScaffoldSignSource> source;
    fiber_tracer::FiberSignResult solve;
    std::int64_t events = 0, fixed = 0, decided = 0, flipped = 0, undecided = 0;
    // Anchored fibers whose sign the crossings would have flipped: the
    // anchors held, but the evidence disagreed with them.
    std::int64_t fixedDisagreements = 0;
    bool solved = false;
};

// The whole-fiber normal signs of a scaffold: the anchored fibers keep their
// gauge, the rest are decided by the crossings (vc/fiber_tracer/FiberNormalSign.hpp).
ScaffoldSignResult scaffoldNormalSigns(const ScaffoldGraph&, const ScaffoldNormals&, std::span<const ScaffoldEvent>,
                                       const ScaffoldSignOptions& = {});

// The facing of the stretches approved for normals, decided before any
// winding is solved.
//
// A normals approval states an axis along a stretch, not which way it faces.
// Every perpendicular crossing whose two anchors lie on approved stretches
// carries the facing from one to the other (their two normals point the same
// way through the sheets), so the stretches the crossings connect are one
// object with one free sign. Crossings that contradict each other there are
// an annotation error and fail the run: one of the approvals on that loop is
// wrong. The object's sign is the one that agrees best with the signed
// normals of the resolved fiber volume around its stretches (the neighbour
// normals), each sample weighted by the cosine of its angle, so a sample
// counts with cos^2; without neighbour normals it is the radial prior the sign
// solve uses.
struct ScaffoldFacingReport {
    std::int64_t stretches = 0, objects = 0;
    // Crossings that carried a facing, and objects faced by the neighbour
    // normals (the others by the radial prior).
    std::int64_t crossings = 0, facedByNeighbours = 0;
    // Samples of the neighbour normals read along the stretches.
    std::int64_t samples = 0;
    // The weakest object: the share of its weighted samples that agree with
    // the facing it got, 1 without any object.
    double weakestAgreement = 1;
};
// Throws std::runtime_error naming the crossing where two approved normals
// contradict the others. `signs` supplies the crossing rule and `normals`
// the neighbour field (ScaffoldNormalOptions::neighbours).
std::vector<ScaffoldApprovedFacing> scaffoldApprovedFacing(const ScaffoldGraph&, const ScaffoldGeometry&,
                                                           const ScaffoldNormals&, std::span<const ScaffoldEvent>,
                                                           const ScaffoldNormalOptions& normals,
                                                           const ScaffoldSignOptions& signs, ScaffoldFacingReport&);
// The whole step: the facing, applied to the normals, and the events' normals
// refreshed. Nothing happens where no stretch is approved for normals.
ScaffoldFacingReport scaffoldFaceApprovedNormals(const ScaffoldGraph&, const ScaffoldGeometry&, ScaffoldNormals&,
                                                 std::vector<ScaffoldEvent>&, const ScaffoldNormalOptions&,
                                                 const ScaffoldEventOptions&, const ScaffoldSignOptions&,
                                                 ScaffoldEventReport&);

// How a fractional measured separation becomes integer constraints. Both
// modes exist in the Python reference stage, so both are kept here.
enum class ScaffoldSeparationMode {
    // The automatic stage's rule: the two integers around the measurement,
    // weighted 1 - fraction and fraction. Faithful to the measurement, but
    // the two halves assert different windings on one pair, so one of them is
    // always violated.
    Split,
    // One constraint at the nearer integer with the event's whole weight (a
    // separation of exactly one half rounds down, the tie the split's own
    // weights make). What the scaffold approval workflow uses: a fractional
    // winding is not something an annotation can be edited to satisfy or an
    // approval can assert, and a split pair would be reported as a
    // two-constraint cycle over a zero-length stretch.
    Rounded,
};

struct ScaffoldMeasuredOptions {
    ScaffoldSeparationMode separations = ScaffoldSeparationMode::Split;
    // Per-type weights, the names the automatic winding stage uses. A zero
    // weight omits that type's arcs; enclosures are derived from the signed
    // perpendicular events whatever the perpendicular weight is.
    double perpendicularWeight = 1.0, parallelWeight = 0.0, enclosureWeight = 0.0;
    // The local sign every signed perpendicular crossing states on its own: the
    // V fiber lies outside the H fiber there, or inside, which bounds the
    // pair's H->V step from one side (the bound an enclosure needs from both).
    // Each becomes a sign constraint of this weight, costing only when the
    // solve breaks it; zero states none. It needs no winding separation.
    // The defaults - perpendicular 1, sign 3, parallel and enclosure 0 -
    // are the scaffold workflow's.
    double signWeight = 3.0;
    // A sign constraint with an anchor on a stretch approved for normals
    // counts this many times: the normal there was looked at.
    double approvedSignFactor = 10.0;
    // The weight of a separation a reviewer stated (direct, or distant with
    // a measurement that agrees). It is not scaled by the crossing's
    // confidence, and it is not exact: nothing is held rigid.
    double reviewedWeight = 10.0;
    // The confidence of one crossing, exactly as the sign stage weights it
    // (fiber_tracer::fiberSignEventWeight: |dot|^power of its two normal axes
    // times the winding-distance factor, min(1, 1/d^2) by default). It scales
    // the perpendicular, parallel and sign weight of that crossing, so an
    // oblique or far crossing still counts but gives way to the clean near
    // ones. An enclosure is a statement about a pair and keeps its weight.
    bool crossingConfidence = true;
    fiber_tracer::FiberSignOptions confidence{};
    // Pass every perpendicular crossing through the shared rule under the
    // solved signs before it states a constraint (the traced volume's
    // winding stage does the same with `crossing_filter`).
    bool crossingFilter = true;
    // Half a winding between an H and a V fiber of the same sheet pair, and
    // the scale the measured separations are read in.
    double perpendicularOffset = 0.5, windingScale = 1.0;
    bool enclosures = true;
};

// What a reviewer stated about one perpendicular crossing (vc_object_store
// FiberConstraintApprovalExtension.hpp): its two fibers lie on neighbouring
// sheets, or farther apart. It is a statement about the separation only; which
// of the two lies outside stays the normals' statement.
enum class ScaffoldNeighbour : std::uint8_t { Undecided = 0, Direct = 1, Distant = 2 };

// One perpendicular crossing as the approval window offers it: what was
// measured, what a reviewer stated and what the crossing states in this run.
struct ScaffoldCrossing {
    std::int64_t event = 0; // an index into the events given
    std::int64_t a = 0, b = 0;
    double arcA = 0, arcB = 0;
    // The measured separation in turns; NaN when none was measured.
    double distance = std::numeric_limits<double>::quiet_NaN();
    // The separation the measurement rounds to, in half windings: 1 a direct
    // neighbour (0.5), 3 one sheet over (1.5), and so on; 0 without a
    // measurement. Unsigned: it does not say which fiber lies outside.
    std::int64_t estimateHalves = 0;
    // The two normals state which fiber lies outside. Without it a
    // separation cannot be stated as a winding step.
    bool signKnown = false;
    ScaffoldNeighbour stated = ScaffoldNeighbour::Undecided;
    // The crossing states a separation in this run (not only a sign bound).
    bool separation = false;
};

// The separation a measurement rounds to, in half windings (see
// ScaffoldCrossing::estimateHalves); 0 for no measurement.
std::int64_t scaffoldCrossingEstimateHalves(double distance, const ScaffoldMeasuredOptions&);

struct ScaffoldMeasuredReport {
    std::int64_t events = 0, used = 0, expanded = 0, withoutDistance = 0;
    // Crossings the shared crossing rule took out (ScaffoldMeasuredOptions::crossingFilter).
    std::int64_t crossingFiltered = 0;
    // Crossings whose separation was fractional: split into two constraints
    // under Split, rounded to the nearer integer under Rounded.
    std::int64_t fractional = 0;
    std::int64_t perpendicular = 0, parallel = 0;
    // sign constraints stated (ScaffoldMeasuredOptions::signWeight).
    std::int64_t signConstraints = 0;
    std::int64_t skippedInvalid = 0, skippedUntagged = 0, skippedSelf = 0, skippedFamilyMismatch = 0;
    std::int64_t enclosurePairs = 0, enclosureArcs = 0, enclosureOneSided = 0, enclosureContradictory = 0,
                 enclosureNonunique = 0;
    // Reviewed crossings: stated direct, stated distant with a measurement
    // that agrees (kept), and stated distant against a measurement of a
    // direct neighbour or none (the separation is dropped: how far is not
    // known). Every distant one moves its sign bound one winding out.
    std::int64_t statedDirect = 0, statedDistantKept = 0, statedDistantDropped = 0;
    // Stated direct where the normals do not say which fiber lies outside:
    // nothing can be stated there until they do.
    std::int64_t statedWithoutSign = 0;
    // Sign constraints with an anchor on a stretch approved for normals.
    std::int64_t signOnApprovedNormals = 0;
};

// The measured constraints of a scaffold, one per asserted relation (a
// fractional separation asserts the two integers around it, with the
// weight split between them). `normalSign` holds one +-1 per fiber.
// `stated` holds what a reviewer stated per event (scaffoldCrossingApprovals),
// or is empty: a direct neighbour replaces the measured separation, a distant
// one keeps a measurement that agrees and otherwise leaves only the sign
// bound, moved one winding out. `crossings` receives the perpendicular
// crossings that passed the crossing rule, in event order.
std::vector<ScaffoldConstraint> scaffoldMeasuredConstraints(const ScaffoldGraph&, const ScaffoldGeometry&,
                                                           std::span<const ScaffoldEvent>,
                                                           std::span<const std::int64_t> normalSign,
                                                           const ScaffoldMeasuredOptions&, ScaffoldMeasuredReport&,
                                                           std::span<const ScaffoldNeighbour> stated = {},
                                                           std::vector<ScaffoldCrossing>* crossings = nullptr);

// A measured constraint's identity in the store, and the spans an approval of
// it is fingerprinted over. A manual link is not one of these: its approval is
// its edge record's state.
objects::FiberConstraintKind scaffoldConstraintStoreKind(ScaffoldConstraintKind);

// The covering span of each endpoint: the span of that fiber between the two
// control points around the constraint's anchor. Nullopt when a fiber carries
// too few control points to have one.
struct ScaffoldConstraintSpans {
    std::size_t spanA = 0, spanB = 0;
};
std::optional<ScaffoldConstraintSpans> scaffoldConstraintSpans(const ScaffoldGraph&, const ScaffoldGeometry&,
                                                               const ScaffoldConstraint&);

objects::FiberConstraintIdentity scaffoldConstraintIdentity(const ScaffoldGraph&, const ScaffoldConstraint&);

// A fresh approval row for one measured constraint, fingerprinted over the
// two covering spans of the loaded scaffold. Throws std::invalid_argument for
// a manual link or a constraint whose spans cannot be resolved.
objects::FiberConstraintApproval approveScaffoldConstraint(const ScaffoldGraph&, const ScaffoldGeometry&,
                                                          const ScaffoldConstraint&, objects::FiberConstraintDistance,
                                                          const std::string& author, const std::string& approvedAt,
                                                          const std::string& host);

struct ScaffoldApprovalReport {
    std::int64_t rows = 0, live = 0, approved = 0;
    // Live rows whose crossing this run did not measure, and rows whose
    // covering spans have changed since they were approved.
    std::int64_t unmatched = 0, stale = 0;
};

// What the live approval rows state per event; Undecided where none does. A
// row whose spans no longer hash to what it recorded states nothing: the
// annotation moved under it. Only perpendicular crossings are reviewed.
std::vector<ScaffoldNeighbour> scaffoldCrossingApprovals(const ScaffoldGraph&, const ScaffoldGeometry&,
                                                         std::span<const ScaffoldEvent>,
                                                         std::span<const objects::FiberConstraintApproval>,
                                                         ScaffoldApprovalReport&);

// A map link states which of its two fibers lies inside the other (an
// ordinary H-V link: the V fiber outside the H fiber; an adjacent one: inside;
// a same-family link: side by side, normals alike). The solved fiber normals
// state the same order at the link's encounter. Where the two disagree, one of
// the two fibers' normals points the wrong way there: a flip somewhere along
// it. One entry per disagreeing link and fiber, with the stretch the flip has
// to lie on - from the nearest link of that fiber whose order agrees to the
// disagreeing one.
struct ScaffoldNormalFlip {
    std::int64_t constraint = 0; // the disagreeing link, an index into the constraints given
    std::int64_t fiber = 0, other = 0;
    double arc = 0, otherArc = 0;
    // This fiber's other links: how many agree and disagree with its normal
    // (the disagreeing one included).
    std::int64_t agreeing = 0, disagreeing = 0;
    // The stretch to look at; the fiber's covering span around the anchor
    // when none of its links agrees (then nothing brackets the flip).
    ScaffoldPiece piece;
    bool bracketed = false;
};
struct ScaffoldNormalFlipReport {
    // Links compared, and those whose order the normals could not state (no
    // normal, or a connector nearly along the sheet).
    std::int64_t checked = 0, agreed = 0, disagreed = 0, undetermined = 0;
    // Of the disagreeing links: those whose two normals face each other
    // across the connector, which is a flip whatever the link states.
    std::int64_t opposed = 0;
};
// The stretches to approve: the one list the approval workflow works through.
//
// A stretch is not part of a solve. It is derived from two things: what a
// solve left violated, and what is approved right now. Whoever lists stretches
// (the approval dock, the command line's report) computes them from those,
// so an approval moves the list at once and a solve only changes the costs.
//
// An annotation states a normal without a facing and a crossing's distance
// without a side, so approving at one constraint alone says nothing: a
// stretch has to tie constraints together. Per violated constraint (of any
// kind, a broken continuation link included: it counts on both sides of its
// cut) and per fiber of it: the stretch from its anchor to the nearest other
// place of that fiber where a constraint is anchored (always), grown over the
// next places until it is `minStretchVx` long (whatever their distance), then
// over further places no farther than `joinVx` from it while it stays within
// `maxStretchVx`. Stretches of one fiber that overlap are one while that stays
// within `maxStretchVx`. A stretch does not end on its outer places: it
// reaches `marginVx` past them and is padded to `minStretchVx`. The cost of a
// stretch is the summed cost of every violated constraint anchored on it.
//
// A stretch that is already approved is fixed: it is offered exactly as it
// was approved, with the violated constraints anchored on it, and never cut
// up or shifted. New stretches lie in the fiber between the approved ones and
// reach at most `marginVx` into one. A broken range holds no stretch and no
// stretch reaches into it. A fiber part with no other place and no approved
// neighbour has no stretch, which is counted.
struct ScaffoldStretchFiber {
    double length = 0;                                // of the annotated line, voxels
    std::vector<double> cuts;                         // arcs where broken geometry splits the fiber
    std::vector<std::pair<double, double>> approved;  // arc ranges approved for geometry or normals
    // Arc ranges marked broken (geometry or normals). The solve drops them, so
    // the list does too: no stretch lies on a broken range or reaches into one.
    std::vector<std::pair<double, double>> broken;
};
// A place of a fiber where constraints are anchored, and how many.
struct ScaffoldStretchPlace {
    std::int64_t fiber = 0;
    double arc = 0;
    std::int64_t constraints = 1;
};
// One fiber side of a violated constraint.
struct ScaffoldStretchViolation {
    std::int64_t fiber = 0;
    double arc = 0, cost = 0;
    std::int64_t other = 0;
    double otherArc = 0;
};
struct ScaffoldStretch {
    std::int64_t fiber = 0;
    // The violated constraints anchored on the stretch and their summed
    // cost, and all constraints anchored on it.
    std::int64_t violations = 0, constraints = 0;
    double cost = 0;
    // The worst of them: its anchor on this fiber, and the other fiber.
    double arc = 0, otherArc = 0;
    std::int64_t other = 0;
    // The fiber's places on the stretch.
    std::int64_t crossings = 0;
    ScaffoldPiece piece;
};
struct ScaffoldStretchOptions {
    double minStretchVx = 500.0, joinVx = 300.0, maxStretchVx = 2000.0;
    // How far a stretch reaches past its outer places.
    double marginVx = 100.0;
};
struct ScaffoldStretchReport {
    std::int64_t violations = 0, suspects = 0;
    // Fiber sides of a violation without another place to tie it to.
    std::int64_t withoutCrossing = 0;
};
// Highest cost first. Fiber indices are into `fibers`.
std::vector<ScaffoldStretch> scaffoldStretches(std::span<const ScaffoldStretchFiber> fibers,
                                               std::span<const ScaffoldStretchPlace>,
                                               std::span<const ScaffoldStretchViolation>,
                                               const ScaffoldStretchOptions&, ScaffoldStretchReport&);
// The same from a solve in memory and the approvals its graph was read with;
// the pieces also name their control range.
std::vector<ScaffoldStretch> scaffoldStretches(const ScaffoldGraph&, const ScaffoldGeometry&,
                                               const ScaffoldSolveResult&, const ScaffoldStretchOptions&,
                                               ScaffoldStretchReport&);

// `constraints` are a solve's; only its map links are compared. Ordered with
// the bracketed stretches first, the shortest before the longer.
std::vector<ScaffoldNormalFlip> scaffoldLinkNormalFlips(const ScaffoldGraph&, const ScaffoldGeometry&,
                                                        const ScaffoldNormals&, std::span<const std::int64_t> normalSign,
                                                        std::span<const ScaffoldConstraint> constraints,
                                                        ScaffoldNormalFlipReport&);

} // namespace vc::fiber_graph
