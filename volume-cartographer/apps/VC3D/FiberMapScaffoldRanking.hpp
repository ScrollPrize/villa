#pragma once
// What the Fiber Map's approval dock ranks, and in which order.
//
// A winding solve records, for every constraint it could not satisfy, the
// shortest cycle the violation lies on and the fiber stretches that cycle
// runs along (vc/fiber_graph, ext_fiber_graph_solve_residuals). This header
// turns those rows into the dock's two kinds of row and defines the orders it
// offers. It is deliberately free of Qt and of the store: the ordering is
// testable on its own.
#include "vc/fiber_graph/ScaffoldSolveRuns.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace vc3d::fiber_map
{

// One accused stretch of a fiber: the arcs it spans, the controls bracketing
// it, and the share of the violated costs it carries.
struct ScaffoldRankedPiece {
    std::string fiberUuid;
    double arcFrom = 0, arcTo = 0, score = 0;
    std::optional<std::int64_t> controlFrom, controlTo;
    // The shortest cycle any accusing constraint lies on; -1 when none does.
    std::int64_t cycleHops = -1;
    // How many violated constraints accuse this stretch.
    std::int64_t constraints = 0;
    // The numbered cycles this stretch lies on (scaffoldCycleNumbers),
    // ascending; empty when no accusing constraint had one.
    std::vector<std::int64_t> cycles;
};

// One entry per distinct stretch, summing the shares the violated constraints
// gave it, highest score first.
[[nodiscard]] std::vector<ScaffoldRankedPiece> rankScaffoldPieces(
    const std::vector<vc::fiber_graph::ScaffoldStoredResidual>& residuals);

// A number per cycle, so the dock can say which rows are about the same loop.
// A run stores a cycle as the stretches it runs along, so two residuals are
// on the same cycle when they name the same stretches; the cycles are
// numbered from 1 in the order the residuals appear (worst first), and a
// residual that no cycle localizes gets 0. Indexed by residual order.
[[nodiscard]] std::vector<std::int64_t> scaffoldCycleNumbers(
    const std::vector<vc::fiber_graph::ScaffoldStoredResidual>& residuals);

// How the approval table is ordered.
enum class ScaffoldApprovalOrder {
    // The highest cost per voxel of fiber first: where a look at a short
    // piece settles the most. A constraint's cost is spread over its cycle's
    // stretches; a segment states the densest stretch it was built from.
    Density,
    // The shortest loop first, then the shorter stretch: the most specific
    // suspects, which is where a correction is cheapest to judge.
    Cycle,
    // The most expensive violation first, ties by the shorter cycle and
    // stretch.
    Cost,
    // Cost per constraint of the cycle: an expensive violation on a short
    // loop outranks the same cost spread over a long one.
    Joint,
};

// The sort keys of one row, whether it is a violated constraint or a stretch.
struct ScaffoldApprovalRow {
    double cost = 0;
    // Constraints of the shortest cycle the row lies on; <= 0 when no cycle
    // localizes it, which sorts last under every order.
    std::int64_t cycle = -1;
    // The stretch's length, or the cycle's total stretch length for a
    // constraint (voxels).
    double length = 0;
    // Cost per voxel: a segment's densest accusation, a constraint's cost
    // over its cycle's stretch length. 0 when no length says.
    double density = 0;
};
[[nodiscard]] bool scaffoldApprovalRowBefore(const ScaffoldApprovalRow& x, const ScaffoldApprovalRow& y,
                                             ScaffoldApprovalOrder order);

// What two solves disagree about.
//
// A winding is gauged per component and per run (minimum 0), so comparing the
// stored numbers one by one would count a gauge shift - or a different
// component partition - as every fiber having moved. The comparison is
// therefore relative: fibers that are in the same component here *and* in the
// same component there move together, and the ones counted are those whose
// offset between the two runs differs from the offset most of their group
// shares. That is exactly "which fibers moved against their neighbours".
struct ScaffoldRunDiff {
    // Fibers both runs placed, and the ones only one of them did.
    std::int64_t shared = 0, onlyFirst = 0, onlySecond = 0;
    // Shared fibers that moved relative to their group, and shared fibers
    // whose normal sign differs (counted only where both runs state one).
    std::int64_t winding = 0, sign = 0;
    // The (component here, component there) groups the comparison was made
    // in, and the shared fibers whose normal sign both runs state.
    std::int64_t groups = 0, signStated = 0;
};
[[nodiscard]] ScaffoldRunDiff scaffoldRunDiff(
    const std::map<std::string, vc::fiber_graph::ScaffoldStoredWinding>& first,
    const std::map<std::string, vc::fiber_graph::ScaffoldStoredWinding>& second);

// Per fiber, which spans carry a live and current piece approval, by aspect
// (0 geometry, 1 normals). Spans are indexed by their lower control point.
struct ScaffoldApprovalSpans {
    std::vector<std::int64_t> controlIds;
    std::array<std::vector<std::uint8_t>, 2> approved;
    // Arc length of each control point along the fiber's line points
    // (voxels), so an anchor arc can be mapped to the span covering it. Same
    // length as controlIds; empty for a fiber with fewer than two controls.
    std::vector<double> controlArcs;
};

// The span of a fiber covering an anchor, with its approval states. `known`
// is false when the fiber's controls could not place the arc, which is not
// the same as a span that is simply unapproved.
struct ScaffoldAnchorSpan {
    bool known = false;
    std::int64_t span = -1, controlFrom = -1, controlTo = -1;
    bool geometryApproved = false, normalsApproved = false;
};

// The segments of a fiber the dock asks to be looked at.
//
// A run stores, per violated constraint, the stretches its cycle runs along;
// cycles that leave a fiber at different crossings give stretches that
// overlap almost everywhere. Listing each of them asks for the same piece of
// fiber to be checked many times, so the dock lists *segments* instead: per
// fiber, non-overlapping ranges built from the accused stretches, the
// densest accusation first.
//
//   1. Every accused stretch is an interval of arc length carrying its share
//      of its constraint's cost (cost split over the cycle's stretches in
//      proportion to 1 / length, as the solve ranks them) and that share's
//      density (share / length).
//   2. Stretches are taken densest first. What a stretch covers that no
//      earlier segment and no approved piece covers is new: it is absorbed
//      into a neighbouring segment when the grown segment stays within
//      `absorbVx` of that segment's *original* extent, and is a segment of
//      its own otherwise. The original extent is what bounds the growth, so
//      absorptions never chain along the fiber.
//   3. Approved geometry is never offered again: only the unapproved part of
//      a stretch becomes a segment, reaching `approvedOverlapVx` back into
//      the approved piece it continues.
//   4. A segment end is moved outward to the nearest control point within
//      `snapVx`, never inward (that would leave accused fiber outside every
//      segment) and never into a neighbouring segment. The boundary two
//      touching segments share moves to the nearest control point within
//      `snapVx` that lies inside both: together they cover the same fiber.
//      An end no control point is near stays where it is, and is stated as
//      a fractional control position.
//   5. A segment shorter than `minSegmentVx` joins a segment it touches (the
//      denser of two): a few voxels of fiber are not a piece to judge.
struct ScaffoldSegmentOptions {
    double minSegmentVx = 100.0;
    double absorbVx = 500.0;
    double snapVx = 750.0;
    double approvedOverlapVx = 50.0;
};

struct ScaffoldSegment {
    std::string fiberUuid;
    double arcFrom = 0, arcTo = 0;
    // The ends as control positions: 3.0 is control point 3, 3.42 is 42 % of
    // the way from it to control point 4. -1 when the fiber's controls are
    // unknown.
    double controlFrom = -1, controlTo = -1;
    // The summed shares of every constraint accusing the segment, and the
    // highest density among the stretches it was built from.
    double cost = 0, density = 0;
    std::int64_t cycleHops = -1;
    std::int64_t constraints = 0;
    std::vector<std::int64_t> cycles;
    // The segment continues a piece whose geometry is already approved.
    bool extendsApproved = false;
    double length() const { return arcTo - arcFrom; }
};

// Where an arc lies among a fiber's control points, as a control position
// (see ScaffoldSegment); -1 without at least two control arcs. Arcs before
// the first or past the last control clamp to them.
[[nodiscard]] double scaffoldControlPosition(const std::vector<double>& controlArcs, double arc);

// The segments of every fiber the residuals accuse, densest first. `fibers`
// states each fiber's control arcs and approved spans; a fiber it does not
// name has no snapping, no approved parts and no control positions.
[[nodiscard]] std::vector<ScaffoldSegment> scaffoldApprovalSegments(
    const std::vector<vc::fiber_graph::ScaffoldStoredResidual>& residuals,
    const std::map<std::string, ScaffoldApprovalSpans>& fibers, const ScaffoldSegmentOptions& options = {});

// What a selected row asks the map to show: the fibers to select, and the
// control ranges to draw on them - a stretch, or the two pieces a
// constraint's anchors lie on.
struct ScaffoldHighlight {
    struct Span {
        std::string fiberUuid;
        // The control points bracketing the range, and - for a segment whose
        // ends are not on control points - the exact control positions to
        // draw between (-1: draw the bracketing range).
        std::int64_t controlFrom = -1, controlTo = -1;
        double positionFrom = -1, positionTo = -1;
    };
    std::vector<std::string> fiberUuids;
    std::vector<Span> spans;
};

} // namespace vc3d::fiber_map
