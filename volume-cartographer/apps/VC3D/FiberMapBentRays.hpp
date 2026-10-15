#pragma once

#include "FiberMapBentRayParams.hpp"

#include <opencv2/core/matx.hpp>

#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Bent rays for the Fiber Map winding solver.
//
// The solver reads an H/V fiber pair where the two polylines share (theta, z)
// about the umbilicus and compares radii there - exact for a sheet that is a
// graph r(theta, z), meaningless where the sheet contains the ray direction
// (a spoke: the sheet running along the umbilicus rays; a shelf: the sheet
// level, its windings stacked in z). There the ray is bent instead: traced
// from a fiber sample along the local sheet normal, which a sheet-normal
// field supplies, so that every sheet crosses it transversally again, and
// the family of such rays through an ill-conditioned run of a fiber is a
// curved curtain. Where the other fiber's polyline crosses that curtain, the
// side it crosses on - outward or inward along the ray - is the winding
// order, read by sign exactly as the radial rule reads r_h - r_v. No voxel
// distance decides an order here: the ray length, step and spacing bound
// the tracing, the reading is the side.
//
// Orientation (which way is outward) is never read from where a ray ends
// up, nor re-read at every sample: the field's axis is transported once
// (from the stretch's canonical end, so the basis is storage-independent)
// along the whole fiber by continuity, and the fiber is cut into SECTIONS
// over which its well-conditioned samples agree on the sign of the
// transported axis against +e_r: each section gets that unanimous sign as
// its PROVISIONAL sign. A change of sign between two voters means the
// transported basis flipped between them, or the sheet returned on itself;
// the voteless samples between two sections of different signs are a
// contested section of their own, which the vote cannot anchor. The vote
// justifies nothing by itself: every section is traced with its provisional
// sign and its status, and the assembly decides from the annotation (link
// witnesses) whether a section's readings constrain, with which sign, or
// are withheld.
//
// Everything is in voxels of the fibers' frame, like the layout; QtCore-free
// so it compiles into the core tests; OpenCV header-only (cv::Vec3d).
namespace vc3d::fiber_map::bent
{

// The unoriented axis of the local papyrus sheet normal over the volume.
class SheetNormalField {
public:
    virtual ~SheetNormalField() = default;
    // Unit axis (sign arbitrary) at a volume point; nullopt where the field
    // has no value.
    [[nodiscard]] virtual std::optional<cv::Vec3d> axis(const cv::Vec3d& volumePoint) const = 0;
    // Batch form; the default loops over axis(). Implementations with a
    // faster path (chunked volumes) override it. `out` is resized.
    virtual void axes(const std::vector<cv::Vec3d>& points,
                      std::vector<std::optional<cv::Vec3d>>& out) const;
    // Content identity for cache keys and the input digest: two fields with
    // the same identity are the same input. Stable across sessions for the
    // same data.
    [[nodiscard]] virtual std::string identity() const = 0;
};

// The umbilicus frame the rays are read in: the unit radial direction, the
// distance from the axis and the angle about it, at a volume point.
struct UmbilicusFrame {
    std::function<cv::Vec3d(const cv::Vec3d&)> radialUnit;
    std::function<double(const cv::Vec3d&)> radius;
    std::function<double(const cv::Vec3d&)> theta;
    // The smallest radius along the straight segment from a to b (the
    // umbilicus may curve between the two heights). Optional: without it
    // the tracer samples the segment finely, which is exact enough for
    // the cutoff only up to the sampling.
    std::function<double(const cv::Vec3d&, const cv::Vec3d&)> minRadiusAlong;
};

// The field read at every sample of a fiber.
struct ConditioningProfile {
    // |axis . e_r| per sample; NaN where the field has no value.
    std::vector<double> value;
    // The field's axis per sample where it has a value.
    std::vector<std::optional<cv::Vec3d>> axis;
};

[[nodiscard]] ConditioningProfile conditioningProfile(const SheetNormalField& field,
                                                      const std::vector<cv::Vec3d>& points,
                                                      const UmbilicusFrame& frame);

// True where the profile's value is below the gate (NaN is not: the
// straight reading stands where the field is silent).
[[nodiscard]] std::vector<unsigned char> illConditionedSamples(const ConditioningProfile& profile,
                                                               double gate);

// Maximal runs of ill-conditioned samples, as [first, last] index pairs.
[[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> illConditionedRuns(
    const std::vector<unsigned char>& illConditioned);

// Sample indices at which the cumulative arclength, measured along the
// run from its canonical end (the lexicographically smaller end point; equal
// end points are told apart by the first differing pair of samples from
// the two ends inward), crosses a multiple of `strideVx`, plus both ends:
// the same physical points whichever way the samples are stored. Sorted,
// unique. One-ended by design: the canonical traversal makes the rounding
// the same for both storage orders.
// Whether the canonical traversal of samples first..last runs forward in
// storage: from the lexicographically smaller end point (equal end points:
// the first differing pair from the two ends inward decides). Used wherever
// a result must not depend on the storage order: ray starts and the
// transport basis of an orientation.
[[nodiscard]] bool canonicalForward(const std::vector<cv::Vec3d>& points, std::size_t first, std::size_t last);

[[nodiscard]] std::vector<std::size_t> symmetricStarts(const std::vector<cv::Vec3d>& points,
                                                       std::size_t first, std::size_t last,
                                                       double strideVx);

enum class RunStatus {
    // The run's section has a unanimous vote: side +1 of every ray is its
    // provisional outward.
    Oriented,
    // The run lies between two sections of different vote signs: the
    // transported basis flipped (or the sheet returned) across it, so no
    // vote anchors it.
    Unoriented,
    // The run's stretch has no well-conditioned sample to anchor it.
    Unsupported,
};

struct OrientedRun {
    std::size_t firstSample = 0;
    std::size_t lastSample = 0;
    RunStatus status = RunStatus::Unsupported;
};

// One section of a supported stretch (see the file comment): samples over
// which the voters (the well-conditioned samples, conditioning at or above
// conditioningGate) agree
// on the sign of the transported axis against +e_r, with the voteless
// samples that join them; or a contested section between two of different
// signs; or a whole voteless stretch. The weights are the voters'
// conditioning, all on the one side the section has.
struct OrientedStretch {
    std::size_t firstSample = 0;
    std::size_t lastSample = 0;
    double agreeWeight = 0.0;
    double disagreeWeight = 0.0;
    // Oriented (unanimous vote), Unoriented (contested, between sections
    // of different signs) or Unsupported (no voter). The normals are
    // supplied in every case, with `provisionalSign` applied to the
    // transported axis: the assembly anchors or withholds from link
    // witnesses.
    RunStatus status = RunStatus::Unsupported;
    int provisionalSign = 1;
};

struct FiberOrientation {
    // The provisionally oriented normal per sample on every supported
    // stretch (transported axis times the stretch's provisional sign),
    // nullopt where the field has no value.
    std::vector<std::optional<cv::Vec3d>> normal;
    std::vector<OrientedStretch> stretches;
    // The ill-conditioned runs, each with its stretch's status.
    std::vector<OrientedRun> runs;
};

// Transport the axis along the fiber by continuity within each supported
// stretch, cut the stretch into sections by the voters' sign, and derive
// the runs' status from their section.
[[nodiscard]] FiberOrientation orientFiber(const std::vector<cv::Vec3d>& points,
                                           const ConditioningProfile& profile,
                                           const UmbilicusFrame& frame,
                                           const BentRayParams& params);

// One traced ray from a start sample: side +1 outward (seeded by the
// oriented normal), -1 inward. `points[k]` at arclength `s[k]`, points[0]
// the start itself; `theta[k]` the angle about the umbilicus accumulated
// along the ray (continuous, radians, 0 at the start: the caller applies
// the winding sense).
struct BentRay {
    std::size_t startSample = 0;
    int side = 1;
    std::vector<cv::Vec3d> points;
    std::vector<double> s;
    std::vector<double> theta;
    // |axis . e_r| of the field at each point (the radial rule's
    // conditioning there): telemetry for readings taken through a crease.
    std::vector<double> conditioning;
};

// The rays of one oriented run: for each start both sides, in start order
// (outward at even indices, inward at odd).
struct BentStretch {
    std::size_t firstSample = 0;
    std::size_t lastSample = 0;
    // The vote's verdict on the run's stretch and the provisional sign the
    // rays were seeded with (side +1 = transported axis * sign).
    RunStatus status = RunStatus::Unsupported;
    int provisionalSign = 1;
    std::vector<std::size_t> starts;
    std::vector<BentRay> rays;
};

struct BentCurtain {
    std::vector<BentStretch> stretches;
    int unorientedRunCount = 0;
    int unsupportedRunCount = 0;
};

// Trace the curtain over every supported run of the fiber, whatever its
// vote (the status travels with the stretch). Ray starts: the run's ends
// and every sample at which the arclength along the run, measured from its
// canonical end (symmetricStarts), crosses a multiple of params.spacingVx.
// A ray follows the
// field axis in steps of params.stepVx, the sign chosen for continuity with
// the previous step; a step is kept only when the field has a value at its
// end point (no extrapolation: the last point of a ray is supported); a ray
// ends where the next step would come closer to the umbilicus axis than
// minRadiusVx anywhere along it (frame.minRadiusAlong, else a fine
// sampling of the step), or at params.maxLengthVx; a start already closer
// than that gets no ray at all (a dead ray, breaking the curtain there).
[[nodiscard]] BentCurtain traceCurtain(const SheetNormalField& field,
                                       const std::vector<cv::Vec3d>& points,
                                       const FiberOrientation& orientation,
                                       const UmbilicusFrame& frame,
                                       const BentRayParams& params,
                                       double minRadiusVx);

// The other fiber's polyline meeting a curtain strip (the surface between
// two consecutive rays of one side of one run).
struct CurtainHit {
    std::size_t stretch = 0;
    // The two rays bounding the strip (indices into BentStretch::rays) and
    // their start samples.
    std::size_t rayA = 0;
    std::size_t rayB = 0;
    // All contributing strips (unordered ray-index pairs, sorted and
    // unique), including both incident strips at a shared ray. rayA/rayB
    // select only the representative; eligibility must consider them all.
    std::vector<std::pair<std::size_t, std::size_t>> contributingStrips;
    std::size_t seedA = 0;
    std::size_t seedB = 0;
    int side = 1;
    // Position on the strip: `across` from ray A (0) to ray B (1), `quad`
    // the row along the rays and `along` within it (0..1); arclength `s`
    // and accumulated angle `theta` interpolated there.
    double across = 0.0;
    std::size_t quad = 0;
    // Which of the quad's two triangles reported the hit (0 or 1).
    int triangle = 0;
    double along = 0.0;
    double s = 0.0;
    double theta = 0.0;
    cv::Vec3d point{0.0, 0.0, 0.0};
    // The other polyline's segment (index of its first sample) and its
    // parameter at the hit; for a vertex hit the vertex's lower incident
    // segment with t = 1 (t = 0 for the polyline's first vertex).
    std::size_t segment = 0;
    double t = 0.0;
    // |sin| of the angle between the polyline and the strip surface; at a
    // vertex the smaller of the two incident segments'.
    double transversality = 0.0;
    // A vertex of the other polyline on the strip with both incident
    // segments on one side (or the polyline's end on the strip): the
    // polyline came up to the curtain and turned back. Recorded, never a
    // crossing.
    bool touch = false;
    // The segment lies in the strip's surface: no side to read. Recorded,
    // never a crossing. `overlapT0`/`overlapT1` bound the segment's overlap
    // with the face (parameters along the segment); the record sits at
    // their midpoint.
    bool tangential = false;
    double overlapT0 = 0.0;
    double overlapT1 = 0.0;
    // The larger turn of the strip's two bounding rays between their seed
    // steps and the hit row, in degrees: used by the crease rule. A shared
    // boundary takes the largest turn over its incident strips/rows.
    double turnDeg = 0.0;
    // The smallest field conditioning along either bounding ray up to the
    // hit row, including incident strips/rows at a shared boundary (1 when
    // unknown). Both measures are invariant under swapping the rays.
    double minConditioning = 1.0;
};

// Every crossing of every strip by the polyline, with the strip's touches
// and tangential contacts. Nothing is reduced here: which crossings of
// one strip are repeated encounters of one passage is decided where the
// winding translate is known (the solver), and nothing occludes anything.
//
// The strip is a triangle mesh with consistently oriented faces (the quad's
// diagonal is the shorter one, ties by the lower midpoint: the same surface
// whichever ray is A). A passage is classified across the SURFACE, not a
// single plane: points just off the hit along the polyline, into and out
// of the passage (towards the nearest distinct samples at a vertex, along
// the segment otherwise, no further than a thousandth of the shortest
// incident edge), are each given the side of the incident face nearest to
// them; different sides make a crossing, the same side a touch. The
// incident faces are the faces that CONTAIN the point: the face hit, the
// quad's other triangle when the point is on the diagonal, at a row or a
// shared ray the neighbouring quad's triangle(s) containing it, and at a
// mesh vertex the containing triangles of every quad meeting it (the
// diagonal neighbour included); a face that merely shares a vertex or an
// edge with the hit's face elsewhere is not incident. Transversality at a crease is the smallest over the
// incident faces (and over the incident segments at a polyline vertex).
//
// Records at ONE polyline position (same segment and parameter, or the
// same vertex run) reported by two faces are one record only when the
// point lies on the boundary those faces share; a record stays its own
// otherwise, however close another's point. A record on a ray shared by
// two strips is owned by the strip whose OTHER ray (the one not shared,
// by ray provenance) starts at the lexicographically smaller point. A segment lying in a face
// (tangential) makes a crossing at an END of its overlap a contact when
// that end lies on a boundary the crossed face shares with the face lain
// in: it leaves the surface across a crease. The two sides of
// a run share the edge between the seeds (s = 0): a polyline through that
// edge passes through the fiber itself and is one touch owned by the
// outward side, and a segment lying along that edge is one tangential
// record owned the same way. Records of ALL runs and sides are in one canonical order:
// the strip (the starts of its two rays as an unordered pair), then the
// outward side before the inward, then s, then the greater transversality
// first, then the hit point; the same order in either storage order of
// either fiber (the run index `stretch` follows the owner's storage order
// and is a tiebreaker for identical records only). Two crossings of one strip at one s are
// therefore ordered by confidence, and the consumer taking the first
// encounter of a strip takes the more transversal one.
[[nodiscard]] std::vector<CurtainHit> intersectCurtain(const BentCurtain& curtain,
                                                       const std::vector<cv::Vec3d>& polyline);

// First same-winding self-crossing per (stretch, lower ray, upper ray,
// side). A shared-boundary crossing bounds every contributing strip.
using CurtainSelfCrossings = std::map<std::tuple<std::size_t, std::size_t, std::size_t, int>, double>;
[[nodiscard]] CurtainSelfCrossings curtainSelfCrossings(const BentCurtain& curtain,
                                                      const std::vector<cv::Vec3d>& points,
                                                      const std::vector<double>& thetaLine);
// Minimum fold limit over all strips contributing to a hit; infinity if
// none crosses itself. Independent of the representative chosen for display.
[[nodiscard]] double firstSelfCrossing(const CurtainHit& hit, const CurtainSelfCrossings& crossings);

} // namespace vc3d::fiber_map::bent
