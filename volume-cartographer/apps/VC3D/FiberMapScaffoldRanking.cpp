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
    const auto cycles = scaffoldCycleNumbers(residuals);
    std::map<std::string, std::vector<Accusation>> accused;
    for (std::size_t index = 0; index < residuals.size(); ++index) {
        const auto& residual = residuals[index];
        const double cost = residual.weight * residual.residual;
        double total = 0;
        for (const auto& piece : residual.pieces) total += 1. / std::max(std::abs(piece.arcTo - piece.arcFrom), 1.);
        for (const auto& piece : residual.pieces) {
            const double from = std::min(piece.arcFrom, piece.arcTo), to = std::max(piece.arcFrom, piece.arcTo);
            Accusation entry;
            entry.from = from;
            entry.to = to;
            entry.share = cost * (1. / std::max(to - from, 1.)) / std::max(total, 1e-300);
            entry.residual = index;
            entry.cycle = cycles[index];
            entry.hops = residual.cycleHops.value_or(-1);
            accused[piece.fiberUuid].push_back(entry);
        }
    }
    // The share of an accusation that lies in [from, to]: by length, or the
    // whole share of a point when the range holds it (`closed`: its upper
    // end included).
    const auto within = [](const Accusation& item, double from, double to, bool closed) {
        if (!(item.to > item.from)) return item.from >= from && (item.from < to || (closed && item.from <= to)) ? 1. : 0.;
        return std::max(0., std::min(item.to, to) - std::max(item.from, from)) / (item.to - item.from);
    };
    struct Building {
        double from = 0, to = 0, cost = 0;
        std::int64_t hops = -1;
        std::set<std::size_t> residuals;
        std::set<std::int64_t> cycles;
        bool extendsApproved = false, placed = true;
    };
    const auto credit = [](Building& segment, const Accusation& item, double share) {
        if (!(share > 0)) return;
        segment.cost += item.share * share;
        segment.residuals.insert(item.residual);
        if (item.cycle > 0) segment.cycles.insert(item.cycle);
        if (item.hops > 0 && (segment.hops < 0 || item.hops < segment.hops)) segment.hops = item.hops;
    };
    std::vector<ScaffoldSegment> out;
    const auto add = [&](const std::string& uuid, const std::vector<double>& arcs, const Building& segment) {
        ScaffoldSegment entry;
        entry.fiberUuid = uuid;
        entry.arcFrom = segment.from;
        entry.arcTo = segment.to;
        entry.controlFrom = segment.placed ? scaffoldControlPosition(arcs, segment.from) : -1;
        entry.controlTo = segment.placed ? scaffoldControlPosition(arcs, segment.to) : -1;
        entry.cost = segment.cost;
        entry.density = segment.cost / std::max(segment.to - segment.from, 1.);
        entry.cycleHops = segment.hops;
        entry.constraints = std::int64_t(segment.residuals.size());
        entry.cycles.assign(segment.cycles.begin(), segment.cycles.end());
        entry.extendsApproved = segment.extendsApproved;
        out.push_back(std::move(entry));
    };
    static const std::vector<Accusation> none;
    // Every fiber whose controls are known is cut, accused or not.
    for (const auto& [uuid, spans] : fibers) {
        const auto& arcs = spans.controlArcs;
        if (arcs.size() < 2) continue;
        const auto found = accused.find(uuid);
        const auto& list = found != accused.end() ? found->second : none;
        const auto& flags = spans.approved[0];
        const auto approvedSpan = [&](std::size_t span) { return span < flags.size() && flags[span] != 0; };
        const std::size_t count = arcs.size() - 1;
        // The accused weight a cut at each control point would split.
        std::vector<double> crossing(arcs.size(), 0.);
        for (std::size_t control = 1; control + 1 < arcs.size(); ++control)
            for (const auto& item : list)
                if (item.from < arcs[control] && arcs[control] < item.to) crossing[control] += item.share;
        for (std::size_t first = 0; first < count;) {
            if (approvedSpan(first)) { ++first; continue; }
            std::size_t last = first;
            while (last < count && !approvedSpan(last)) ++last;
            // The unapproved run [first, last] of control points: the fewest
            // stretches no longer than the limit (a single longer span is a
            // stretch of its own), and among those the cuts that split the
            // least accused weight.
            struct Best { std::int64_t stretches = 0; double split = 0; std::size_t from = 0; };
            std::vector<Best> best(last - first + 1);
            for (std::size_t to = first + 1; to <= last; ++to) {
                Best chosen{std::numeric_limits<std::int64_t>::max(), 0., first};
                for (std::size_t from = to; from-- > first;) {
                    if (from + 1 < to && arcs[to] - arcs[from] > options.maxSegmentVx) break;
                    const auto& before = best[from - first];
                    const Best candidate{before.stretches + 1, before.split + (from > first ? crossing[from] : 0.), from};
                    if (candidate.stretches < chosen.stretches ||
                        (candidate.stretches == chosen.stretches && candidate.split < chosen.split))
                        chosen = candidate;
                }
                best[to - first] = chosen;
            }
            std::vector<std::pair<std::size_t, std::size_t>> cut;
            for (std::size_t to = last; to > first; to = best[to - first].from) cut.emplace_back(best[to - first].from, to);
            std::reverse(cut.begin(), cut.end());
            for (const auto [from, to] : cut) {
                Building segment;
                segment.from = arcs[from];
                segment.to = arcs[to];
                for (const auto& item : list) credit(segment, item, within(item, segment.from, segment.to, to == last));
                // A stretch that continues approved geometry reaches a little
                // back into it, so the two are judged as one piece of fiber.
                if (from == first && first > 0) {
                    segment.from = std::max(arcs[first - 1], segment.from - options.approvedOverlapVx);
                    segment.extendsApproved = true;
                }
                if (to == last && last < count) {
                    segment.to = std::min(arcs[last + 1], segment.to + options.approvedOverlapVx);
                    segment.extendsApproved = true;
                }
                add(uuid, arcs, segment);
            }
            first = last;
        }
        // What a run accuses outside the fiber's control points is fiber the
        // annotation no longer has: it stays a row, unplaced, and says so.
        for (const auto& item : list) {
            const auto outside = [&](double from, double to, double share) {
                Building segment;
                segment.from = from;
                segment.to = to;
                segment.placed = false;
                credit(segment, item, share);
                add(uuid, arcs, segment);
            };
            if (!(item.to > item.from)) {
                if (item.from < arcs.front() || item.from > arcs.back()) outside(item.from, item.to, 1.);
                continue;
            }
            if (item.from < arcs.front())
                outside(item.from, std::min(item.to, arcs.front()),
                        (std::min(item.to, arcs.front()) - item.from) / (item.to - item.from));
            if (item.to > arcs.back())
                outside(std::max(item.from, arcs.back()), item.to,
                        (item.to - std::max(item.from, arcs.back())) / (item.to - item.from));
        }
    }
    // A fiber the run accuses whose controls are unknown: its stretches as
    // the run states them, unplaced.
    for (const auto& [uuid, list] : accused) {
        const auto known = fibers.find(uuid);
        if (known != fibers.end() && known->second.controlArcs.size() >= 2) continue;
        static const std::vector<double> noArcs;
        for (const auto& item : list) {
            Building segment;
            segment.from = item.from;
            segment.to = item.to;
            segment.placed = false;
            credit(segment, item, 1.);
            add(uuid, noArcs, segment);
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const ScaffoldSegment& x, const ScaffoldSegment& y) {
        if (x.density != y.density) return x.density > y.density;
        return x.cost > y.cost;
    });
    return out;
}

} // namespace vc3d::fiber_map
