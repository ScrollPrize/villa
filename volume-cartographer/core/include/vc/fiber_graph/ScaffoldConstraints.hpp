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
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include "vc_object_store/objects/FiberConstraintApprovalExtension.hpp"

#include <cstdint>
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
    std::size_t workers = 0; // 0: the extractor's own default
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

struct ScaffoldSignOptions {
    fiber_tracer::FiberSignOptions sign;
    // Decide the gauge of the fibers no anchor fixed, from the perpendicular
    // crossings. Off leaves them at their geometry gauge, which only means
    // something when every fiber carries an anchor.
    bool solve = true;
    // Optional progress and cancellation, checked before the solve starts.
    ScaffoldRunMonitor* monitor = nullptr;
};

struct ScaffoldSignResult {
    // Per fiber: the multiplier that turns its gauge normal into the inward
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
    double perpendicularWeight = 1.0, parallelWeight = 1.0, enclosureWeight = 1.0;
    // Half a winding between an H and a V fiber of the same sheet pair, and
    // the scale the measured separations are read in.
    double perpendicularOffset = 0.5, windingScale = 1.0;
    bool enclosures = true;
};

struct ScaffoldMeasuredReport {
    std::int64_t events = 0, used = 0, expanded = 0, withoutDistance = 0;
    // Crossings whose separation was fractional: split into two constraints
    // under Split, rounded to the nearer integer under Rounded.
    std::int64_t fractional = 0;
    std::int64_t perpendicular = 0, parallel = 0;
    std::int64_t skippedInvalid = 0, skippedUntagged = 0, skippedSelf = 0, skippedFamilyMismatch = 0;
    std::int64_t enclosurePairs = 0, enclosureArcs = 0, enclosureOneSided = 0, enclosureContradictory = 0,
                 enclosureNonunique = 0;
};

// The measured constraints of a scaffold, one per asserted relation (a
// fractional separation asserts the two integers around it, with the
// weight split between them). `normalSign` holds one +-1 per fiber.
std::vector<ScaffoldConstraint> scaffoldMeasuredConstraints(const ScaffoldGraph&, const ScaffoldGeometry&,
                                                           std::span<const ScaffoldEvent>,
                                                           std::span<const std::int64_t> normalSign,
                                                           const ScaffoldMeasuredOptions&, ScaffoldMeasuredReport&);

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

// Whether the geometry of both covering pieces is an approved span of its
// fiber right now. A constraint may only count as approved when it is: an
// approval of a relation between unreviewed geometry would pin a solve to
// geometry nobody has looked at. False when the spans cannot be resolved.
bool scaffoldConstraintPiecesApproved(const ScaffoldGraph&, const ScaffoldGeometry&, const ScaffoldConstraint&);
objects::FiberConstraintIdentity scaffoldConstraintIdentity(const ScaffoldGraph&, const ScaffoldConstraint&);

// The span of a fiber covering an arc position, over a content read straight
// from the store: the arc is the cumulative chord length of its line points,
// which is the measure every anchor arc is stated in. Nullopt when the fiber
// has fewer than two control points on its line.
std::optional<std::size_t> scaffoldSpanCoveringArc(const objects::FiberPieceContent&, double arc);

// A fresh approval row for one measured constraint, fingerprinted over the
// two covering spans of the loaded scaffold. Throws std::invalid_argument for
// a manual link or a constraint whose spans cannot be resolved.
objects::FiberConstraintApproval approveScaffoldConstraint(const ScaffoldGraph&, const ScaffoldGeometry&,
                                                          const ScaffoldConstraint&, const std::string& author,
                                                          const std::string& approvedAt, const std::string& host);

struct ScaffoldApprovalReport {
    std::int64_t rows = 0, live = 0, approved = 0;
    // Live rows whose constraint this run did not measure, and rows whose
    // covering spans have changed since they were approved.
    std::int64_t unmatched = 0, stale = 0;
    // Live and current rows whose covering pieces are no longer approved for
    // geometry: the approval is kept, but it fixes nothing in this run.
    std::int64_t withoutPieces = 0;
};

// Mark the measured constraints that a live approval row covers. A row whose
// spans no longer hash to what it recorded approves nothing: the annotation
// moved under it.
void applyScaffoldConstraintApprovals(const ScaffoldGraph&, const ScaffoldGeometry&, std::span<ScaffoldConstraint>,
                                      std::span<const objects::FiberConstraintApproval>, ScaffoldApprovalReport&);

} // namespace vc::fiber_graph
