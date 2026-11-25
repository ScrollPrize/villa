#include "FiberMapScaffoldRanking.hpp"

#include <cmath>
#include <set>
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
    case ScaffoldApprovalOrder::Density:
        if (x.density != y.density) return x.density > y.density;
        if (x.cost != y.cost) return x.cost > y.cost;
        return x.length < y.length;
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

double scaffoldControlPosition(const std::vector<double>& controlArcs, double arc)
{
    if (controlArcs.size() < 2) return -1;
    if (arc <= controlArcs.front()) return 0;
    if (arc >= controlArcs.back()) return double(controlArcs.size() - 1);
    const auto upper = std::upper_bound(controlArcs.begin(), controlArcs.end(), arc);
    const auto index = std::size_t(upper - controlArcs.begin()) - 1;
    const double span = controlArcs[index + 1] - controlArcs[index];
    return double(index) + (span > 0 ? (arc - controlArcs[index]) / span : 0.);
}

std::vector<ScaffoldSegment> scaffoldApprovalSegments(
    const std::vector<vc::fiber_graph::ScaffoldStoredResidual>& residuals,
    const std::map<std::string, ScaffoldApprovalSpans>& fibers, const ScaffoldSegmentOptions& options)
{
    struct Accusation {
        double from = 0, to = 0, share = 0;
        std::size_t residual = 0;
        std::int64_t cycle = 0, hops = -1;
    };
    // A cycle stretch whose geometry is approved from end to end cannot be
    // the error: it takes no share, and the others take more.
    const auto approved = [&](const std::string& uuid, double from, double to) {
        const auto found = fibers.find(uuid);
        if (found == fibers.end()) return false;
        for (const auto& [first, last] : found->second.approvedArcs[0])
            if (from >= first - 1e-6 && to <= last + 1e-6) return true;
        return false;
    };
    const auto cycles = scaffoldCycleNumbers(residuals);
    std::map<std::string, std::vector<Accusation>> accused;
    for (std::size_t index = 0; index < residuals.size(); ++index) {
        const auto& residual = residuals[index];
        const double cost = residual.weight * residual.residual;
        std::vector<Accusation> open;
        std::vector<const std::string*> owners;
        double total = 0;
        for (const auto& piece : residual.pieces) {
            Accusation entry;
            entry.from = std::min(piece.arcFrom, piece.arcTo);
            entry.to = std::max(piece.arcFrom, piece.arcTo);
            if (approved(piece.fiberUuid, entry.from, entry.to)) continue;
            entry.share = 1. / std::max(entry.to - entry.from, 1.);
            entry.residual = index;
            entry.cycle = cycles[index];
            entry.hops = residual.cycleHops.value_or(-1);
            total += entry.share;
            open.push_back(entry);
            owners.push_back(&piece.fiberUuid);
        }
        for (std::size_t at = 0; at < open.size(); ++at) {
            open[at].share = cost * open[at].share / std::max(total, 1e-300);
            accused[*owners[at]].push_back(open[at]);
        }
    }
    std::vector<ScaffoldSegment> out;
    for (auto& [uuid, list] : accused) {
        std::stable_sort(list.begin(), list.end(), [](const Accusation& x, const Accusation& y) {
            return x.from != y.from ? x.from < y.from : x.to < y.to;
        });
        static const std::vector<double> noArcs;
        const auto known = fibers.find(uuid);
        const auto& arcs = known != fibers.end() ? known->second.controlArcs : noArcs;
        for (std::size_t first = 0; first < list.size();) {
            ScaffoldSegment entry;
            entry.fiberUuid = uuid;
            entry.arcFrom = list[first].from;
            entry.arcTo = list[first].to;
            std::set<std::size_t> constraints;
            std::set<std::int64_t> loops;
            std::size_t next = first;
            for (; next < list.size(); ++next) {
                const auto& item = list[next];
                // The next stretch joins when it overlaps this one or starts
                // close behind it, while the whole stays within the limit.
                if (next > first && (item.from > entry.arcTo + options.joinVx ||
                                     std::max(entry.arcTo, item.to) - entry.arcFrom > options.maxSegmentVx))
                    break;
                entry.arcTo = std::max(entry.arcTo, item.to);
                entry.cost += item.share;
                constraints.insert(item.residual);
                if (item.cycle > 0) loops.insert(item.cycle);
                if (item.hops > 0 && (entry.cycleHops < 0 || item.hops < entry.cycleHops)) entry.cycleHops = item.hops;
            }
            first = next;
            // A stretch outside the fiber's control points is fiber the
            // annotation no longer has: it stays a row without positions.
            const bool placed = arcs.size() >= 2 && entry.arcFrom >= arcs.front() - 1e-6 && entry.arcTo <= arcs.back() + 1e-6;
            entry.controlFrom = placed ? scaffoldControlPosition(arcs, entry.arcFrom) : -1;
            entry.controlTo = placed ? scaffoldControlPosition(arcs, entry.arcTo) : -1;
            entry.density = entry.cost / std::max(entry.length(), 1.);
            entry.constraints = std::int64_t(constraints.size());
            entry.cycles.assign(loops.begin(), loops.end());
            out.push_back(std::move(entry));
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const ScaffoldSegment& x, const ScaffoldSegment& y) {
        if (x.density != y.density) return x.density > y.density;
        return x.cost > y.cost;
    });
    return out;
}

} // namespace vc3d::fiber_map
