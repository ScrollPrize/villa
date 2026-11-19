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

// What a selected row asks the map to show: the fibers to select, and the
// control ranges to draw on them - a stretch, or the two pieces a
// constraint's anchors lie on.
struct ScaffoldHighlight {
    struct Span {
        std::string fiberUuid;
        std::int64_t controlFrom = -1, controlTo = -1;
    };
    std::vector<std::string> fiberUuids;
    std::vector<Span> spans;
};

} // namespace vc3d::fiber_map
