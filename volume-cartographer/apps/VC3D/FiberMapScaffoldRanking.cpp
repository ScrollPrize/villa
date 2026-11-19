#include "FiberMapScaffoldRanking.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <tuple>

namespace vc3d::fiber_map
{

std::vector<std::int64_t> scaffoldCycleNumbers(
    const std::vector<vc::fiber_graph::ScaffoldStoredResidual>& residuals)
{
    // A cycle's identity is the set of stretches it runs along, which is what
    // the run stored about it; the same loop found from two of its own
    // constraints names the same stretches.
    using Signature = std::vector<std::tuple<std::string, double, double>>;
    std::map<Signature, std::int64_t> numbers;
    std::vector<std::int64_t> out(residuals.size(), 0);
    for (std::size_t index = 0; index < residuals.size(); ++index) {
        const auto& residual = residuals[index];
        if (!residual.cycleHops || *residual.cycleHops <= 0 || residual.pieces.empty()) continue;
        Signature signature;
        signature.reserve(residual.pieces.size());
        for (const auto& piece : residual.pieces)
            signature.emplace_back(piece.fiberUuid, piece.arcFrom, piece.arcTo);
        std::sort(signature.begin(), signature.end());
        signature.erase(std::unique(signature.begin(), signature.end()), signature.end());
        const auto found = numbers.find(signature);
        out[index] = found != numbers.end() ? found->second : std::int64_t(numbers.size()) + 1;
        if (found == numbers.end()) numbers.emplace(std::move(signature), out[index]);
    }
    return out;
}

std::vector<ScaffoldRankedPiece> rankScaffoldPieces(
    const std::vector<vc::fiber_graph::ScaffoldStoredResidual>& residuals)
{
    const auto cycles = scaffoldCycleNumbers(residuals);
    std::map<std::tuple<std::string, double, double>, ScaffoldRankedPiece> merged;
    for (std::size_t index = 0; index < residuals.size(); ++index) {
        const auto& residual = residuals[index];
        const auto cycle = cycles[index];
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
                if (cycle > 0) entry.cycles.push_back(cycle);
                merged.emplace(key, entry);
                continue;
            }
            found->second.score += piece.score;
            ++found->second.constraints;
            if (cycle > 0 && std::find(found->second.cycles.begin(), found->second.cycles.end(), cycle) ==
                                 found->second.cycles.end()) {
                auto& list = found->second.cycles;
                list.insert(std::upper_bound(list.begin(), list.end(), cycle), cycle);
            }
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

ScaffoldRunDiff scaffoldRunDiff(const std::map<std::string, vc::fiber_graph::ScaffoldStoredWinding>& first,
                                const std::map<std::string, vc::fiber_graph::ScaffoldStoredWinding>& second)
{
    ScaffoldRunDiff out;
    // Per (component, component) group: how many shared fibers share each
    // offset. The most frequent offset is that group's gauge difference.
    std::map<std::pair<std::int64_t, std::int64_t>, std::map<std::int64_t, std::int64_t>> offsets;
    for (const auto& [uuid, here] : first) {
        const auto found = second.find(uuid);
        if (found == second.end()) {
            ++out.onlyFirst;
            continue;
        }
        ++out.shared;
        ++offsets[{here.component, found->second.component}][found->second.winding - here.winding];
        if (here.normalSign && found->second.normalSign) {
            ++out.signStated;
            if (*here.normalSign != *found->second.normalSign) ++out.sign;
        }
    }
    for (const auto& entry : second)
        if (first.find(entry.first) == first.end()) ++out.onlySecond;
    out.groups = std::int64_t(offsets.size());
    std::map<std::pair<std::int64_t, std::int64_t>, std::int64_t> gauge;
    for (const auto& [group, counts] : offsets) {
        std::int64_t best = 0, bestCount = -1;
        for (const auto& [offset, count] : counts)
            // Ties go to the smaller offset, so the answer does not depend on
            // the order the fibers came in.
            if (count > bestCount) { best = offset; bestCount = count; }
        gauge[group] = best;
    }
    for (const auto& [uuid, here] : first) {
        const auto found = second.find(uuid);
        if (found == second.end()) continue;
        const std::pair<std::int64_t, std::int64_t> group{here.component, found->second.component};
        if (found->second.winding - here.winding != gauge[group]) ++out.winding;
    }
    return out;
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
