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
    using Range = std::pair<double, double>;
    struct Accusation {
        double from = 0, to = 0, share = 0, density = 0;
        std::size_t residual = 0;
        std::int64_t cycle = 0, hops = -1;
    };
    struct Building {
        double coreFrom = 0, coreTo = 0, from = 0, to = 0;
        double cost = 0, density = 0;
        std::int64_t hops = -1;
        std::set<std::size_t> residuals;
        std::set<std::int64_t> cycles;
        bool extendsApproved = false;
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
            const double length = std::max(to - from, 1.);
            Accusation entry;
            entry.from = from;
            entry.to = to;
            entry.share = cost * (1. / length) / std::max(total, 1e-300);
            entry.density = entry.share / length;
            entry.residual = index;
            entry.cycle = cycles[index];
            entry.hops = residual.cycleHops.value_or(-1);
            accused[piece.fiberUuid].push_back(entry);
        }
    }
    const auto overlap = [](double a, double b, double c, double d) { return std::max(0., std::min(b, d) - std::max(a, c)); };
    std::vector<ScaffoldSegment> out;
    for (auto& [uuid, list] : accused) {
        const auto known = fibers.find(uuid);
        static const std::vector<double> noArcs;
        const auto& arcs = known != fibers.end() ? known->second.controlArcs : noArcs;
        // The approved geometry of this fiber, as merged arc ranges.
        std::vector<Range> approved;
        if (known != fibers.end() && arcs.size() >= 2) {
            const auto& flags = known->second.approved[0];
            for (std::size_t span = 0; span + 1 < arcs.size() && span < flags.size(); ++span) {
                if (!flags[span]) continue;
                if (!approved.empty() && approved.back().second >= arcs[span]) approved.back().second = arcs[span + 1];
                else approved.emplace_back(arcs[span], arcs[span + 1]);
            }
        }
        std::stable_sort(list.begin(), list.end(), [](const Accusation& x, const Accusation& y) {
            if (x.density != y.density) return x.density > y.density;
            return x.to - x.from < y.to - y.from;
        });
        std::vector<Building> segments;
        // Whether anything claimed - a segment or approved geometry - lies in
        // the open range (a, b): a gap an absorption may not bridge.
        const auto blocked = [&](double a, double b, const Building* except) {
            if (!(b > a)) return false;
            for (const auto& range : approved)
                if (overlap(a, b, range.first, range.second) > 0) return true;
            for (const auto& other : segments)
                if (&other != except && overlap(a, b, other.from, other.to) > 0) return true;
            return false;
        };
        for (const auto& item : list) {
            const double length = std::max(item.to - item.from, 1.);
            const auto credit = [&](Building& segment, double covered) {
                segment.cost += item.share * std::min(1., covered / length);
                segment.density = std::max(segment.density, item.density);
                segment.residuals.insert(item.residual);
                if (item.cycle > 0) segment.cycles.insert(item.cycle);
                if (item.hops > 0 && (segment.hops < 0 || item.hops < segment.hops)) segment.hops = item.hops;
            };
            // What earlier segments already cover is credited to them.
            for (auto& segment : segments) {
                const double covered = item.to > item.from ? overlap(item.from, item.to, segment.from, segment.to)
                                                           : (item.from >= segment.from && item.from <= segment.to ? 1. : 0.);
                if (covered > 0) credit(segment, covered);
            }
            // What nothing covers yet.
            std::vector<Range> claimed = approved;
            for (const auto& segment : segments) claimed.emplace_back(segment.from, segment.to);
            std::sort(claimed.begin(), claimed.end());
            std::vector<Range> fresh;
            double cursor = item.from;
            bool pointCovered = false;
            for (const auto& range : claimed) {
                if (range.second <= cursor) continue;
                if (range.first >= item.to && item.to > item.from) break;
                if (item.to == item.from) {
                    if (range.first <= item.from && item.from <= range.second) pointCovered = true;
                    continue;
                }
                if (range.first > cursor) fresh.emplace_back(cursor, std::min(range.first, item.to));
                cursor = std::max(cursor, range.second);
                if (cursor >= item.to) break;
            }
            if (item.to > item.from) {
                if (cursor < item.to) fresh.emplace_back(cursor, item.to);
            } else if (!pointCovered) {
                fresh.emplace_back(item.from, item.to);
            }
            for (const auto& part : fresh) {
                // Absorbed by a neighbour when the grown segment stays within
                // reach of that neighbour's original extent and nothing
                // claimed lies in between; the nearest such neighbour wins.
                Building* host = nullptr;
                double hostGap = std::numeric_limits<double>::infinity();
                for (auto& segment : segments) {
                    if (part.first < segment.coreFrom - options.absorbVx || part.second > segment.coreTo + options.absorbVx)
                        continue;
                    const double gapFrom = std::min(segment.to, part.second), gapTo = std::max(segment.from, part.first);
                    const double gap = part.first >= segment.to ? part.first - segment.to
                                                                : (part.second <= segment.from ? segment.from - part.second : 0.);
                    if (gap > 0 && blocked(gapFrom, gapTo, &segment)) continue;
                    if (gap < hostGap) {
                        host = &segment;
                        hostGap = gap;
                    }
                }
                const double covered = std::max(part.second - part.first, item.to > item.from ? 0. : 1.);
                if (host != nullptr) {
                    host->from = std::min(host->from, part.first);
                    host->to = std::max(host->to, part.second);
                    credit(*host, covered);
                    continue;
                }
                Building segment;
                segment.coreFrom = segment.from = part.first;
                segment.coreTo = segment.to = part.second;
                segments.push_back(segment);
                credit(segments.back(), covered);
            }
        }
        std::sort(segments.begin(), segments.end(), [](const Building& x, const Building& y) { return x.from < y.from; });
        // A segment that continues approved geometry reaches a little back
        // into it, so the two are judged as one piece of fiber.
        for (auto& segment : segments)
            for (const auto& range : approved) {
                if (std::abs(segment.from - range.second) <= 1e-6) {
                    segment.from = std::max(range.first, segment.from - options.approvedOverlapVx);
                    segment.extendsApproved = true;
                }
                if (std::abs(segment.to - range.first) <= 1e-6) {
                    segment.to = std::min(range.second, segment.to + options.approvedOverlapVx);
                    segment.extendsApproved = true;
                }
            }
        // Two segments that touch share a boundary, which is wherever two
        // accusations happened to meet. It moves to the nearest control point
        // within reach when that point lies inside the two of them: together
        // they still cover exactly the same fiber.
        for (std::size_t index = 1; index < segments.size() && !arcs.empty(); ++index) {
            auto& left = segments[index - 1];
            auto& right = segments[index];
            if (std::abs(left.to - right.from) > 1e-6) continue;
            const auto above = std::lower_bound(arcs.begin(), arcs.end(), left.to);
            double best = left.to, distance = std::numeric_limits<double>::infinity();
            for (const auto candidate : {above, above == arcs.begin() ? arcs.end() : above - 1}) {
                if (candidate == arcs.end()) continue;
                const double away = std::abs(*candidate - left.to);
                if (away <= options.snapVx && away < distance && *candidate > left.from && *candidate < right.to) {
                    best = *candidate;
                    distance = away;
                }
            }
            left.to = right.from = best;
        }
        // A sliver - a very dense accusation a few voxels long, or what a
        // moved boundary left of one - is not a piece anybody can judge: it
        // joins the touching neighbour, the denser of two. Once, here, so it
        // cannot chain either.
        for (std::size_t index = 0; index < segments.size() && segments.size() > 1;) {
            auto& sliver = segments[index];
            if (sliver.to - sliver.from >= options.minSegmentVx) { ++index; continue; }
            // "Touches" with the same tolerance: a neighbour a sliver's
            // length away is the piece it belongs to.
            const bool hasLeft = index > 0 && sliver.from - segments[index - 1].to <= options.minSegmentVx;
            const bool hasRight = index + 1 < segments.size() && segments[index + 1].from - sliver.to <= options.minSegmentVx;
            if (!hasLeft && !hasRight) { ++index; continue; }
            const bool intoLeft = hasLeft && (!hasRight || segments[index - 1].density >= segments[index + 1].density);
            auto& host = segments[intoLeft ? index - 1 : index + 1];
            host.from = std::min(host.from, sliver.from);
            host.to = std::max(host.to, sliver.to);
            host.cost += sliver.cost;
            host.density = std::max(host.density, sliver.density);
            host.residuals.insert(sliver.residuals.begin(), sliver.residuals.end());
            host.cycles.insert(sliver.cycles.begin(), sliver.cycles.end());
            if (sliver.hops > 0 && (host.hops < 0 || sliver.hops < host.hops)) host.hops = sliver.hops;
            host.extendsApproved = host.extendsApproved || sliver.extendsApproved;
            segments.erase(segments.begin() + std::ptrdiff_t(index));
            if (intoLeft && index > 0) --index;
        }
        // Ends move outward to a control point within reach, never across a
        // neighbouring segment. Left to right, so a segment's start is held
        // by the end its left neighbour already took.
        for (std::size_t index = 0; index < segments.size(); ++index) {
            auto& segment = segments[index];
            const double floor = index > 0 ? segments[index - 1].to : -std::numeric_limits<double>::infinity();
            const double ceiling = index + 1 < segments.size() ? segments[index + 1].from
                                                               : std::numeric_limits<double>::infinity();
            if (!arcs.empty()) {
                const auto after = std::upper_bound(arcs.begin(), arcs.end(), segment.from);
                if (after != arcs.begin()) {
                    const double at = *(after - 1);
                    if (segment.from - at <= options.snapVx && at >= floor) segment.from = at;
                }
                const auto beyond = std::lower_bound(arcs.begin(), arcs.end(), segment.to);
                if (beyond != arcs.end() && *beyond - segment.to <= options.snapVx && *beyond <= ceiling)
                    segment.to = *beyond;
            }
            ScaffoldSegment entry;
            entry.fiberUuid = uuid;
            entry.arcFrom = segment.from;
            entry.arcTo = segment.to;
            entry.controlFrom = scaffoldControlPosition(arcs, segment.from);
            entry.controlTo = scaffoldControlPosition(arcs, segment.to);
            entry.cost = segment.cost;
            entry.density = segment.density;
            entry.cycleHops = segment.hops;
            entry.constraints = std::int64_t(segment.residuals.size());
            entry.cycles.assign(segment.cycles.begin(), segment.cycles.end());
            entry.extendsApproved = segment.extendsApproved;
            // Snapping into approved geometry is an extension of it too.
            for (const auto& range : approved)
                if (overlap(entry.arcFrom, entry.arcTo, range.first, range.second) > 0) entry.extendsApproved = true;
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
