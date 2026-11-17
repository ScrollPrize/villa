#include "FiberMapScaffoldRanking.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <tuple>

namespace vc3d::fiber_map
{

std::vector<ScaffoldRankedPiece> rankScaffoldPieces(
    const std::vector<vc::fiber_graph::ScaffoldStoredResidual>& residuals)
{
    std::map<std::tuple<std::string, double, double>, ScaffoldRankedPiece> merged;
    for (const auto& residual : residuals) {
        for (const auto& piece : residual.pieces) {
            const std::tuple<std::string, double, double> key{piece.fiberUuid, piece.arcFrom, piece.arcTo};
            const auto found = merged.find(key);
            if (found == merged.end()) {
                ScaffoldRankedPiece entry;
                entry.fiberUuid = piece.fiberUuid;
                entry.arcFrom = piece.arcFrom;
                entry.arcTo = piece.arcTo;
                entry.controlFrom = piece.controlFrom;
                entry.controlTo = piece.controlTo;
                entry.score = piece.score;
                entry.cycleHops = residual.cycleHops.value_or(-1);
                entry.constraints = 1;
                merged.emplace(key, entry);
                continue;
            }
            found->second.score += piece.score;
            ++found->second.constraints;
            if (residual.cycleHops && (found->second.cycleHops < 0 || *residual.cycleHops < found->second.cycleHops))
                found->second.cycleHops = *residual.cycleHops;
        }
    }
    std::vector<ScaffoldRankedPiece> pieces;
    pieces.reserve(merged.size());
    for (const auto& [key, piece] : merged) pieces.push_back(piece);
    std::stable_sort(pieces.begin(), pieces.end(), [](const auto& x, const auto& y) { return x.score > y.score; });
    return pieces;
}

bool scaffoldApprovalRowBefore(const ScaffoldApprovalRow& x, const ScaffoldApprovalRow& y,
                               ScaffoldApprovalOrder order)
{
    // A row no cycle localizes sorts last: nothing says where its fault is.
    const auto cycleKey = [](const ScaffoldApprovalRow& row) {
        return row.cycle > 0 ? row.cycle : std::numeric_limits<std::int64_t>::max();
    };
    switch (order) {
    case ScaffoldApprovalOrder::Cycle:
        if (cycleKey(x) != cycleKey(y)) return cycleKey(x) < cycleKey(y);
        return x.length < y.length;
    case ScaffoldApprovalOrder::Cost:
        if (x.cost != y.cost) return x.cost > y.cost;
        if (cycleKey(x) != cycleKey(y)) return cycleKey(x) < cycleKey(y);
        return x.length < y.length;
    case ScaffoldApprovalOrder::Joint:
    default: {
        const double a = x.cost / double(std::max<std::int64_t>(1, x.cycle > 0 ? x.cycle : 1));
        const double b = y.cost / double(std::max<std::int64_t>(1, y.cycle > 0 ? y.cycle : 1));
        if (a != b) return a > b;
        return cycleKey(x) < cycleKey(y);
    }
    }
}

} // namespace vc3d::fiber_map
