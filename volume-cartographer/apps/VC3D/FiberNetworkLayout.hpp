#pragma once

#include <QPointF>
#include <QString>

#include <opencv2/core/matx.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "FiberWindingSolver.hpp"

// Extrinsic unroll of manually linked H/V fiber networks about the scroll
// umbilicus, ported from the fiber_network_unroll.py review script.
//
//     x = unwrapped angle about the umbilicus * the network's median radius
//     y = z
//
// Everything here is in voxels of the frame the fibers are annotated in, input
// and output alike: the layout never learns the voxel size and so cannot get it
// wrong. Turning a voxel count into a physical length is the caller's business,
// and only when it actually knows the voxel size. The tuning lengths in
// LayoutParams are voxels too; their defaults are documented below.
//
// Each fiber's unwrapped angle carries its own arbitrary multiple of 2*pi, so
// the offsets are made mutually consistent by walking the link graph and
// snapping each fiber to the nearest whole turn of an already-placed
// neighbour (best-agreeing link first). Crossings then coincide and loops
// close by construction: there is no solver and no accumulated drift. A link
// whose endpoints still disagree by a large fraction of a turn afterwards was
// annotated on the wrong winding and is reported as suspect.
namespace vc3d::fiber_map
{

struct InputLink {
    int controlPointIndex = -1;
    uint64_t branchFiberId = 0;
    int branchControlPointIndex = -1;
    // The annotation keeps this in sync on both reciprocal refs; the layout
    // ORs the two sides on dedup anyway.
    bool pending = false;
    // The endpoints are one winding apart, the V fiber's inside the H
    // fiber's (FiberBranchRef::adjacent). The solver gets the one-turn
    // offset instead of an equality. A pair that is not one H and one V has
    // no inside: such a link is an annotation error (PlacedLink::
    // adjacentUnpaired), reported suspect and given no constraint. ORed on
    // dedup like pending.
    bool adjacent = false;
    // The ref stated its kind explicitly (the containing JSON array).
    // Two explicit refs of one pair disagreeing (true here, false there) is
    // reported as PlacedLink::adjacentDisagrees.
    bool adjacentExplicit = true;
};

struct InputFiber {
    uint64_t id = 0;
    // Stable identity, carried through so callers can act on a placed fiber
    // after the runtime ids have been reassigned.
    std::string fileName;
    QString label;
    char hvTag = '?';
    std::vector<cv::Vec3d> controlPoints;
    std::vector<cv::Vec3d> linePoints;
    // Per control-point-span "was fiber-model traced"; anything else is only
    // an interpolation. Empty or mismatched renders as a single traced run.
    std::vector<bool> tracedSegments;
    // Per control point: tagged kollesis_termination. Read at solve time
    // (a V fiber linked to two tagged H fibers is on a kollesis), never by
    // the cached detection, so it is not part of the cache keys; empty or
    // mismatched means no tags.
    std::vector<bool> kollesisTerminations;
    // Per control point: tagged break. Display only (the gap span between two
    // consecutive flags draws dotted amber, the points get a dotted rim);
    // read when the placed fiber's runs are built from the fresh input, so
    // not part of the cache keys either. Empty or mismatched means no tags.
    std::vector<bool> breaks;
    // Raw directed refs; the layout dedupes reciprocal pairs.
    std::vector<InputLink> links;
};

// Every length here is in voxels. The defaults are what these lengths' physical
// intents come to at 2.4 µm/voxel (the resolution of the open-data scrolls), and
// they exist for the one case in which no better answer exists: a project whose
// voxel size is unknown. A caller that knows the voxel size is expected to
// convert each physical intent itself, which reproduces the geometry these
// numbers were chosen for at any resolution. The intent is named in each
// comment; the number after it is that intent at 2.4 µm.
struct LayoutParams {
    int minFibers = 3;
    int maxNetworks = 3;
    double suspectTurns = 0.25;
    // Gaussian arclength sigma for de-bumping the drawn fibers; 0 disables
    // smoothing. Intent: 1.2 mm.
    double smoothVx = 500.0;
    // Uniform arclength resampling step of the drawn geometry. Intent: 0.025 cm.
    double resampleStepVx = 104.0;
    // Minimum padding around a network: room for a few rows of label chips.
    // Intent: 2.2 cm across, 1.6 cm up.
    double minPadXVx = 9167.0;
    double minPadYVx = 6667.0;
    // Grid the panel starts snap to, which is also the winding-label interval.
    // Intent: 5 cm.
    double panelTickVx = 20833.0;
    // Smallest gap left between two panels. Intent: 1 cm.
    double minGapVx = 4167.0;
};

// One styling run of a fiber, in voxels with +y = +z. Exactly one of the
// three styles applies: gap (both endpoint controls tagged break) outranks
// traced / interpolated. The points carry the layout's own geometry, with the
// one-sample overlap past each bounding control that lets neighbouring runs
// join visually; the gap heat map seeds from these, so they are the same
// whether or not any break is tagged. Drawing a gap exactly is the drawer's
// job: see displayRunPoints.
struct Run {
    bool traced = true;
    bool gap = false;
    // The controls (indices into PlacedFiber::controlPoints) bounding the
    // run's spans; -1 when the run is the whole fiber without span flags.
    int firstControl = -1;
    int lastControl = -1;
    std::vector<QPointF> points;
};

struct PlacedFiber {
    uint64_t id = 0;
    std::string fileName;
    QString label;
    char hvTag = '?';
    std::vector<Run> runs;
    // Control-point positions read off the smoothed geometry, so they land
    // exactly on the drawn curve.
    std::vector<QPointF> controlPoints;
    // Parallel to controlPoints: the point carries the kollesis_termination
    // tag (copied from the input; always sized to controlPoints).
    std::vector<bool> kollesisTerminations;
    // Parallel to controlPoints: the point carries the break tag (same rule).
    std::vector<bool> breaks;
};

// The points to DRAW for fiber.runs[runIndex]: the run's own points, except
// that a gap run, and any run next to one, ends exactly at the shared
// control's position on the curve instead of one sample past it, so the
// dotted amber covers the gap span and nothing else and no solid stroke runs
// on underneath it. Runs away from every gap are returned unchanged.
[[nodiscard]] std::vector<QPointF> displayRunPoints(const PlacedFiber& fiber,
                                                    std::size_t runIndex);

struct PlacedLink {
    uint64_t fiberA = 0;
    int cpA = -1;
    uint64_t fiberB = 0;
    int cpB = -1;
    QPointF a;
    QPointF b;
    // |dTheta| / 2pi after placement.
    double turnErr = 0.0;
    bool suspect = false;
    // True when either input ref of the deduped pair still awaits review.
    bool pending = false;
    // An adjacent-winding link (InputLink::adjacent): turnErr is measured
    // against the one-turn offset, and the map draws the endpoints as
    // triangles.
    bool adjacent = false;
    // An adjacent link between fibers that are not one H and one V: the
    // error the map flags for it (always suspect; it constrained nothing).
    bool adjacentUnpaired = false;
    // The pair's two refs state different kinds (one adjacent, one explicitly
    // ordinary): the annotation is inconsistent between the two files. Always
    // suspect, constrains nothing; the sync merge arbitrates.
    bool adjacentDisagrees = false;
};

struct WindingMark {
    double xVx = 0.0;
    int number = 0;
};

struct PlacedNetwork {
    // 0-based index into the size-sorted list of networks with >= minFibers.
    int networkIndex = 0;
    double rRefVx = 0.0;
    double x0Vx = 0.0;
    double x1Vx = 0.0;
    std::vector<PlacedFiber> fibers;
    std::vector<PlacedLink> links;
    // Continuous numbering across panels.
    std::vector<WindingMark> windings;
};

struct Result {
    // Ordered inner -> outer by median umbilicus radius, panel offsets applied.
    std::vector<PlacedNetwork> networks;
    double widthVx = 0.0;
    double yMinVx = 0.0;
    double yMaxVx = 0.0;
    // Networks with >= minFibers, before the top-N cut.
    int qualifyingNetworkCount = 0;
    int suspectLinkCount = 0;
};

// umbilicusCenters are dense volume-frame centers (x, y, z), one per z slice;
// an empty list means the network cannot be unrolled and yields an empty
// result.
//
// This per-network unroll has no winding dimension - x is angle at one
// reference radius - so every link, adjacent or not, is an angular tie here
// and InputLink::adjacent only rides through to PlacedLink (and marks an
// unpaired one suspect). The winding semantics of adjacent links live in
// buildGlobalLayout, which is what the Fiber Map draws.
[[nodiscard]] Result buildLayout(const std::vector<InputFiber>& fibers,
                                 const std::vector<cv::Vec3f>& umbilicusCenters,
                                 const LayoutParams& params);

// ---------------------------------------------------------------------------
// The global map: every fiber on one unrolled plane, each at the winding the
// solver inferred for it, x = (s*theta + 2*pi*k) * rRef, y = z. Voxels
// throughout, like buildLayout.

struct GlobalLayoutParams {
    // One link-suspicion threshold: this value also overrides
    // solver.linkSuspectTurns inside buildGlobalLayout, so the confidence a
    // link solves with and the suspicion it is reported with can never
    // disagree.
    double suspectTurns = 0.25;
    // Same intents as the LayoutParams entries of the same names.
    double smoothVx = 500.0;
    double resampleStepVx = 104.0;
    double minPadXVx = 9167.0;
    double minPadYVx = 6667.0;
    winding::SolverParams solver;
};

// How a fiber's component got its absolute winding; mirrors the solver's
// ComponentAnchor so the UI never implies winding knowledge the solve does
// not have.
enum class GlobalAnchor { Primary, Radius, AmbiguousRadius, Unresolved };

struct GlobalFiberMeta {
    GlobalAnchor anchor = GlobalAnchor::Unresolved;
    bool linked = false;
    bool sheetDriftSuspect = false;
    // A V fiber the annotator has linked to kollesis-tagged H ends on both
    // sides: its seam encounters read as same winding or inward.
    bool onKollesis = false;
    // W range over the fiber; a multi-turn H fiber has no single winding.
    double windingLo = 0.0;
    double windingHi = 0.0;
    // Linked-network membership: fibers connected through manual links share
    // an id, numbered by network size descending (0 = largest, ties by first
    // fiber label). -1 for fibers with no valid links.
    int networkId = -1;
    // Fibers in this fiber's network (1 for unlinked fibers).
    int networkSize = 1;
};

struct GlobalPlacedFiber {
    PlacedFiber fiber;
    GlobalFiberMeta meta;
};

// Every H-vs-V crossing event the solver resolved (one per crossing of the
// two polylines, apex touches included and flagged), positioned on the drawn
// H polyline (y = +z, like every placed coordinate here), with what it read
// and what became of the representative that stood for it: the inspection
// record behind the map's markers. Ids are the runtime fiber ids of the build.
struct CrossingEvent {
    QPointF posVx;
    uint64_t hFiberId = 0;
    uint64_t vFiberId = 0;
    long long n = 0;
    winding::CrossingKind kind = winding::CrossingKind::Inside;
    winding::CrossingStatus status = winding::CrossingStatus::Used;
    double deltaR = 0.0;
    double transversality = 0.0;
    bool tangential = false;
    bool touch = false;
    // Read as a kollesis seam encounter (same winding or inward).
    bool kollesis = false;
    // ... from the solve contradiction on a certified kollesis V rather than
    // from a tag on this H fiber (see winding::Crossing::kollesisInferred).
    bool kollesisInferred = false;
    int orientation = 0;
    int mergedCount = 1;
    double confidence = 0.0;
    double violationTurns = 0.0;
    // Index into GlobalResult::crossingGroups, or -1.
    long long groupId = -1;
};

// A pair's crossings on one translate and V branch read together (see
// winding::CrossingGroup); members index GlobalResult::crossingEvents.
struct CrossingGroupRecord {
    uint64_t hFiberId = 0;
    uint64_t vFiberId = 0;
    long long n = 0;
    std::size_t vBranch = 0;
    std::vector<std::size_t> members;
    int multiplicity = 0;
    int insideCount = 0;
    int orientationSum = 0;
    int insideOrientationSum = 0;
    bool mixedSigns = false;
    bool coverageGap = false;
    bool unresolved = false;
    bool onCurtain = false;
    bool traversalCovered = false;
    // A seam encounter lies on the translate: no verdict (incomplete count).
    bool seamed = false;
    double minAbsDeltaR = 0.0;
    double meanTransversality = 0.0;
    bool hasVerdict = false;
    winding::CrossingKind verdict = winding::CrossingKind::Inside;
    double confidence = 0.0;
    winding::CrossingStatus status = winding::CrossingStatus::Used;
    double violationTurns = 0.0;
};

// A declared winding error, marked on the map: a dropped crossing the final
// map still violates, or every member of such a traversal group. Group
// members share a groupId; they are one conflict drawn at each place the
// pair met, not independent errors.
struct CrossingMark {
    QPointF posVx;
    uint64_t hFiberId = 0;
    uint64_t vFiberId = 0;
    long long n = 0;
    winding::CrossingKind kind = winding::CrossingKind::Inside;
    double deltaR = 0.0;
    double violationTurns = 0.0;
    // Index into GlobalResult::crossingEvents.
    std::size_t eventIndex = 0;
    // Index into GlobalResult::crossingGroups when the error is a group's, else -1.
    long long groupId = -1;
    bool kollesis = false;
};

// A fiber that could not be placed: no geometry, no umbilicus to unroll
// about, or geometry too degenerate to draw. Listed so "every fiber" stays
// honest.
struct UnplacedFiber {
    uint64_t id = 0;
    std::string fileName;
    QString label;
    char hvTag = '?';
};

struct GlobalResult {
    // Ordered by (label, fileName, id), every placeable fiber of the input;
    // fileName before the runtime id so the order survives id reassignment
    // across package loads.
    std::vector<GlobalPlacedFiber> fibers;
    std::vector<PlacedLink> links;
    // One mark per integer winding across the padded extent.
    std::vector<WindingMark> windings;
    std::vector<CrossingMark> suspectCrossings;
    // Every crossing and every traversal group of the solve, for inspection.
    std::vector<CrossingEvent> crossingEvents;
    std::vector<CrossingGroupRecord> crossingGroups;
    std::vector<UnplacedFiber> unplaced;
    double rRefVx = 0.0;
    double x0Vx = 0.0;
    double x1Vx = 0.0;
    double yMinVx = 0.0;
    double yMaxVx = 0.0;
    int chirality = 1;
    int islandCount = 0;
    int unresolvedCount = 0;
    int tieCount = 0;
    int suspectLinkCount = 0;
    // Declared winding errors: dropped constraints the final map still
    // violates. droppedCrossingCount counts individual crossings,
    // declaredGroupCount traversal groups (each one conflict);
    // traversalGroupCount is the groups that took a verdict.
    int droppedCrossingCount = 0;
    int declaredGroupCount = 0;
    int traversalGroupCount = 0;
    // Owner-segment pairs the detector could not intersect (exactly
    // parallel); their translates take no group verdict.
    int unresolvedIntersectionCount = 0;
    // Events read as kollesis seam encounters.
    int kollesisCrossingCount = 0;
    int kollesisInferredCount = 0;
    // Geometry the solver refused to learn from: angularly ill-conditioned
    // or wild segments, and tangential contacts. Nonzero values say the map
    // may be underconstrained for a reason the fibers themselves can't show.
    int gatedSegmentCount = 0;
    int tangentialCount = 0;
    // The sheet model, for distances along the sheet rather than along the
    // map: the fibers' umbilicus radius fitted as a straight line in the
    // winding coordinate, r(W) = sheetRadius0Vx + sheetPitchVx * W. That is
    // an Archimedean spiral, which a scroll is to first order, and it costs
    // one pass over samples the solve already holds. Fitted over the samples
    // of anchored, drawable fibers; when they span too little winding to fix
    // a slope, or the fit is not a sensible spiral (radius or pitch not
    // positive), the model falls back to r0 = rRefVx, pitch = 0, and sheet
    // distance degrades to the map's own arclength at rRef. See SheetModel.
    double sheetRadius0Vx = 0.0;
    double sheetPitchVx = 0.0;
    // Phase timings (milliseconds), for the rebuild's one-line profile.
    double prepMs = 0.0;
    double detectMs = 0.0;
    double solveMs = 0.0;
    double geometryMs = 0.0;
};

// Distance along the sheet as a function of map position. The map's x is
// theta * rRef, arclength at one reference radius, which understates the
// outer windings and overstates the inner ones; with the radius modelled as
// r(W) = radius0 + pitch * W the distance from winding 0 to winding W is the
// integral of r over the angle, 2*pi*(radius0*W + pitch*W^2/2). All voxels.
struct SheetModel {
    double rRefVx = 0.0;
    double radius0Vx = 0.0;
    double pitchVx = 0.0;
};

// Sheet distance (voxels, signed) from winding 0 to the sheet position drawn at
// scene x. Monotonic wherever the modelled radius is positive.
[[nodiscard]] double sheetDistanceVx(const SheetModel& model, double xVx);
// Inverse of sheetDistanceVx: the scene x at which the sheet distance reads
// distanceVx. NaN when no such position exists (the modelled radius would have
// to be negative there) or the model is degenerate (rRef or radius0 not
// positive).
[[nodiscard]] double sheetXForDistanceVx(const SheetModel& model, double distanceVx);
// The sheet model a result carries.
[[nodiscard]] SheetModel sheetModelOf(const GlobalResult& result);

// 128-bit content digest (two independent FNV-1a lanes over raw bytes).
// Collisions are the design's one stated deviation from literal exactness:
// ~2^-64 per comparison, with Full rebuild as the recovery path.
struct ContentDigest {
    uint64_t a = 0;
    uint64_t b = 0;
    bool operator==(const ContentDigest& other) const
    {
        return a == other.a && b == other.b;
    }
    bool operator!=(const ContentDigest& other) const { return !(*this == other); }
};

// Memoization for buildGlobalLayout, keyed on content digests of exactly what
// each cached artifact consumes - never on anyone's generation counters:
//
//   prep slot  (per fileName):  H(fiber geometry fields, umbilicus)
//   pair slot  (per H,V pair):  H(prepKey_H, prepKey_V, chirality,
//                                 detection params)
//
// Fiber identity across builds is the stored fileName; runtime ids are
// reassigned per load. Links are deliberately NOT part of the fiber digest:
// no cached artifact consumes them (they are re-collected fresh each build),
// so a link edit invalidates nothing here. One replaceable slot per key;
// slots for fileNames absent from the current snapshot are swept after every
// successful build, so memory is bounded by the current fiber set. Entries
// are pure key -> value memoizations, so a build that throws mid-way leaves
// only valid entries behind.
//
// The cached build is identical to an uncached one in every SEMANTIC field
// BY CONSTRUCTION: the fresh path runs the same per-pair detection function
// and the same shard assembly, so a cache hit substitutes an equal value
// into an identical computation. The phase timing fields are telemetry and
// necessarily differ; digestGlobalResult() defines the semantic field set.
class GlobalLayoutCache;

[[nodiscard]] GlobalResult buildGlobalLayout(
    const std::vector<InputFiber>& fibers,
    const std::vector<cv::Vec3f>& umbilicusCenters,
    const GlobalLayoutParams& params,
    GlobalLayoutCache* cache = nullptr);

class GlobalLayoutCache {
public:
    void clear();

    struct Stats {
        bool used = false;
        int fibersReused = 0;
        int fibersRecomputed = 0;
        int pairsReused = 0;
        int pairsRecomputed = 0;
    };
    [[nodiscard]] const Stats& lastStats() const { return _stats; }
    // The cached detection shards in (H file, V file) order, for tests of the
    // contract that a cached shard is the fresh one bit for bit.
    [[nodiscard]] std::vector<const winding::PairDetections*> cachedDetections() const;

private:
    friend GlobalResult buildGlobalLayout(const std::vector<InputFiber>&,
                                          const std::vector<cv::Vec3f>&,
                                          const GlobalLayoutParams&,
                                          GlobalLayoutCache*);
    struct PrepSlot {
        ContentDigest key;
        std::vector<double> thetaLine;
        std::vector<double> radius;
        std::vector<std::size_t> controlLineIndex;
    };
    struct PairSlot {
        ContentDigest key;
        winding::PairDetections detection;
    };
    std::map<std::string, PrepSlot> _prep;
    std::map<std::pair<std::string, std::string>, PairSlot> _pairs;
    Stats _stats;
};

// (buildGlobalLayout is declared above the cache class: cache may be nullptr
// for today's uncached behavior, and duplicate fileNames in the input disable
// the cache for that build, logged - fileName is the slot identity and must
// be unique.)

// Digest of everything a layout was built from (fiber contents, umbilicus,
// params) and of everything it produced. Full rebuild compares the output
// digest against the last Update's when the inputs digest matches - a
// memoization bug cannot hide.
[[nodiscard]] ContentDigest digestGlobalInputs(
    const std::vector<InputFiber>& fibers,
    const std::vector<cv::Vec3f>& umbilicusCenters,
    const GlobalLayoutParams& params);
[[nodiscard]] ContentDigest digestGlobalResult(const GlobalResult& result);

} // namespace vc3d::fiber_map
