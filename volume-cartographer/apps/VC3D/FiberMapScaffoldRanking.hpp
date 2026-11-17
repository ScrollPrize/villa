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
};

// One entry per distinct stretch, summing the shares the violated constraints
// gave it, highest score first.
[[nodiscard]] std::vector<ScaffoldRankedPiece> rankScaffoldPieces(
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

// Per fiber, which spans carry a live and current piece approval, by aspect
// (0 geometry, 1 normals). Spans are indexed by their lower control point.
struct ScaffoldApprovalSpans {
    std::vector<std::int64_t> controlIds;
    std::array<std::vector<std::uint8_t>, 2> approved;
};

// What a selected row asks the map to show: the fibers to select and, for a
// stretch, the control range on the first of them (-1: the whole fiber).
struct ScaffoldHighlight {
    std::vector<std::string> fiberUuids;
    std::int64_t controlFrom = -1, controlTo = -1;
};

} // namespace vc3d::fiber_map
