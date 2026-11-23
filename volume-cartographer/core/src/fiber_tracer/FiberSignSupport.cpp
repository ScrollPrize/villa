#include "vc/fiber_tracer/FiberSignSupport.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace vc::fiber_tracer
{
namespace
{

struct Edge {
    std::int64_t u = 0, w = 0;
    double length = 0.0;
    double cost = 0.0;  // 1 / weight for a crossing, 0 along a chain
};

// Undirected adjacency in compressed rows: (neighbour, edge index) pairs.
struct Adjacency {
    std::vector<std::int64_t> offset, target, edge;
    Adjacency(const std::int64_t nodes, const std::vector<Edge>& edges) : offset(nodes + 1, 0)
    {
        for (const auto& e : edges) {
            ++offset[e.u + 1];
            ++offset[e.w + 1];
        }
        std::partial_sum(offset.begin(), offset.end(), offset.begin());
        target.resize(offset.back());
        edge.resize(offset.back());
        std::vector<std::int64_t> fill(offset.begin(), offset.end() - 1);
        for (std::size_t k = 0; k < edges.size(); ++k) {
            const auto& e = edges[k];
            target[fill[e.u]] = e.w;
            edge[fill[e.u]++] = std::int64_t(k);
            target[fill[e.w]] = e.u;
            edge[fill[e.w]++] = std::int64_t(k);
        }
    }
};

// Tarjan's bridges, iterative. Skips the arriving edge by index, so parallel
// edges between two nodes are never bridges.
std::vector<char> bridgesOf(const std::int64_t nodes, const std::vector<Edge>& edges, const Adjacency& adj)
{
    std::vector<char> bridge(edges.size(), 0);
    std::vector<std::int64_t> disc(nodes, -1), low(nodes, 0);
    struct Frame {
        std::int64_t node, parentEdge, next;
    };
    std::vector<Frame> stack;
    std::int64_t time = 0;
    for (std::int64_t start = 0; start < nodes; ++start) {
        if (disc[start] >= 0) continue;
        disc[start] = low[start] = time++;
        stack.push_back({start, -1, adj.offset[start]});
        while (!stack.empty()) {
            Frame& top = stack.back();
            const std::int64_t v = top.node;
            if (top.next < adj.offset[v + 1]) {
                const std::int64_t w = adj.target[top.next], k = adj.edge[top.next];
                ++top.next;
                if (k == top.parentEdge) continue;
                if (disc[w] < 0) {
                    disc[w] = low[w] = time++;
                    stack.push_back({w, k, adj.offset[w]});
                } else {
                    low[v] = std::min(low[v], disc[w]);
                }
                continue;
            }
            const std::int64_t parentEdge = top.parentEdge;
            stack.pop_back();
            if (stack.empty()) continue;
            const std::int64_t p = stack.back().node;
            low[p] = std::min(low[p], low[v]);
            if (low[v] > disc[p]) bridge[parentEdge] = 1;
        }
    }
    return bridge;
}

// The support graph of the agreeing crossings, queryable at any fiber point.
class SupportGraph
{
public:
    SupportGraph(const std::vector<double>& fiberLengths, const std::vector<FiberSupportCrossing>& crossings,
                 FiberSupportResult& result)
        : fibers_(std::int64_t(fiberLengths.size())), ends_(std::int64_t(2 * crossings.size())),
          endFiber_(ends_), endArc_(ends_), order_(ends_), chainAfter_(ends_, -1), first_(fibers_ + 1, 0)
    {
        for (std::size_t k = 0; k < crossings.size(); ++k) {
            const auto& c = crossings[k];
            if (c.a < 0 || c.a >= fibers_ || c.b < 0 || c.b >= fibers_)
                throw std::invalid_argument("support crossing addresses an unknown fiber");
            if (!(c.weight > 0.0)) throw std::invalid_argument("support crossing weight must be positive");
            endFiber_[2 * k] = c.a;
            endArc_[2 * k] = c.arcA;
            endFiber_[2 * k + 1] = c.b;
            endArc_[2 * k + 1] = c.arcB;
        }
        // Crossing ends ordered along each fiber; chain edges join neighbours.
        std::iota(order_.begin(), order_.end(), 0);
        std::sort(order_.begin(), order_.end(), [&](const std::int64_t x, const std::int64_t y) {
            return endFiber_[x] != endFiber_[y] ? endFiber_[x] < endFiber_[y]
                   : endArc_[x] != endArc_[y]   ? endArc_[x] < endArc_[y]
                                                : x < y;
        });
        for (std::int64_t v = 0; v < ends_; ++v) ++first_[endFiber_[v] + 1];
        std::partial_sum(first_.begin(), first_.end(), first_.begin());
        std::vector<Edge> edges;
        edges.reserve(2 * crossings.size());
        for (std::int64_t i = 0; i + 1 < ends_; ++i) {
            const std::int64_t x = order_[i], y = order_[i + 1];
            if (endFiber_[x] != endFiber_[y]) continue;
            chainAfter_[i] = std::int64_t(edges.size());
            edges.push_back({x, y, endArc_[y] - endArc_[x], 0.0});
        }
        result.chainEdges = std::int64_t(edges.size());
        for (std::int64_t k = 0; k < std::int64_t(crossings.size()); ++k)
            edges.push_back({2 * k, 2 * k + 1, 0.0, 1.0 / crossings[std::size_t(k)].weight});

        const Adjacency adj(ends_, edges);
        bridge_ = bridgesOf(ends_, edges, adj);

        // 2-edge-connected components: connectivity without the bridges.
        std::vector<std::int64_t> comp(ends_, -1);
        std::vector<std::int64_t> size, queue;
        for (std::int64_t s = 0; s < ends_; ++s) {
            if (comp[s] >= 0) continue;
            const auto id = std::int64_t(size.size());
            comp[s] = id;
            queue.assign(1, s);
            for (std::size_t q = 0; q < queue.size(); ++q) {
                const std::int64_t v = queue[q];
                for (std::int64_t i = adj.offset[v]; i < adj.offset[v + 1]; ++i) {
                    const std::int64_t w = adj.target[i];
                    if (bridge_[adj.edge[i]] || comp[w] >= 0) continue;
                    comp[w] = id;
                    queue.push_back(w);
                }
            }
            size.push_back(std::int64_t(queue.size()));
        }
        const auto components = std::int64_t(size.size());
        result.components = components;

        // Bridge tree walked from the core, summing bridge chain length and crossing cost.
        std::vector<double> compLength(components, kInf), compCost(components, kInf);
        if (components > 0) {
            std::vector<Edge> treeEdges;
            for (std::size_t k = 0; k < edges.size(); ++k) {
                if (!bridge_[k]) continue;
                const auto& e = edges[k];
                treeEdges.push_back({comp[e.u], comp[e.w], e.length, e.cost});
                if (k >= std::size_t(result.chainEdges)) ++result.bridgeCrossings;
                else ++result.bridgeChainEdges;
            }
            const Adjacency tree(components, treeEdges);
            const auto core = std::int64_t(std::max_element(size.begin(), size.end()) - size.begin());
            compLength[core] = 0.0;
            compCost[core] = 0.0;
            queue.assign(1, core);
            for (std::size_t q = 0; q < queue.size(); ++q) {
                const std::int64_t c = queue[q];
                for (std::int64_t i = tree.offset[c]; i < tree.offset[c + 1]; ++i) {
                    const std::int64_t d = tree.target[i];
                    if (std::isfinite(compLength[d])) continue;
                    const auto& e = treeEdges[tree.edge[i]];
                    compLength[d] = compLength[c] + e.length;
                    compCost[d] = compCost[c] + e.cost;
                    queue.push_back(d);
                }
            }
            for (std::int64_t v = 0; v < ends_; ++v) {
                if (comp[v] == core) ++result.coreEnds;
                if (!std::isfinite(compLength[comp[v]])) ++result.unlinkedEnds;
            }
        }
        result.endMissedLength.resize(ends_);
        result.endMissedCrossingCost.resize(ends_);
        for (std::int64_t v = 0; v < ends_; ++v) {
            result.endMissedLength[v] = compLength[comp[v]];
            result.endMissedCrossingCost[v] = compCost[comp[v]];
        }
        L_ = &result.endMissedLength;
        X_ = &result.endMissedCrossingCost;
    }

    // The support of fiber `f` at arc `t`. `i` is the first order position on the
    // fiber whose end lies at or after t (see lowerBound).
    FiberSupportPoint at(const std::int64_t f, const double t, const std::int64_t i) const
    {
        const auto& L = *L_;
        const auto& X = *X_;
        FiberSupportPoint p{t, kInf, kInf};
        const std::int64_t lo = first_[f], hi = first_[f + 1];
        if (hi == lo) return p;
        if (i == lo) {
            const std::int64_t e = order_[lo];
            p.missedLength = L[e] + (endArc_[e] - t);
            p.missedCrossingCost = X[e];
        } else if (i == hi) {
            const std::int64_t e = order_[hi - 1];
            p.missedLength = L[e] + (t - endArc_[e]);
            p.missedCrossingCost = X[e];
        } else {
            const std::int64_t x = order_[i - 1], y = order_[i];
            if (bridge_[chainAfter_[i - 1]]) {
                // Walk in from the end nearer the core.
                const bool fromX = L[x] < L[y] || (L[x] == L[y] && X[x] <= X[y]);
                p.missedLength = fromX ? L[x] + (t - endArc_[x]) : L[y] + (endArc_[y] - t);
                p.missedCrossingCost = fromX ? X[x] : X[y];
            } else {
                p.missedLength = L[x];
                p.missedCrossingCost = X[x];
            }
        }
        return p;
    }

    std::int64_t lowerBound(const std::int64_t f, const double t) const
    {
        const auto begin = order_.begin() + first_[f], end = order_.begin() + first_[f + 1];
        return std::int64_t(std::lower_bound(begin, end, t, [&](const std::int64_t e, const double arc) {
                                return endArc_[e] < arc;
                            }) - order_.begin());
    }

    std::int64_t firstOf(const std::int64_t f) const { return first_[f]; }
    std::int64_t lastOf(const std::int64_t f) const { return first_[f + 1]; }
    double arcAt(const std::int64_t i) const { return endArc_[order_[i]]; }
    std::int64_t endAt(const std::int64_t i) const { return order_[i]; }
    std::int64_t ends() const { return ends_; }
    // Whether the chain from order position i to i + 1 is a bridge.
    bool bridgeAfter(const std::int64_t i) const { return chainAfter_[i] >= 0 && bridge_[chainAfter_[i]] != 0; }

private:
    static constexpr double kInf = std::numeric_limits<double>::infinity();
    std::int64_t fibers_, ends_;
    std::vector<std::int64_t> endFiber_;
    std::vector<double> endArc_;
    std::vector<std::int64_t> order_, chainAfter_, first_;
    std::vector<char> bridge_;
    const std::vector<double>* L_ = nullptr;
    const std::vector<double>* X_ = nullptr;
};

}  // namespace

FiberSupportResult fiberSignSupport(const std::vector<double>& fiberLengths,
                                    const std::vector<FiberSupportCrossing>& crossings,
                                    const double spacing)
{
    if (!(spacing > 0.0)) throw std::invalid_argument("support spacing must be positive");
    FiberSupportResult result;
    const SupportGraph graph(fiberLengths, crossings, result);
    result.points.resize(fiberLengths.size());
    for (std::int64_t f = 0; f < std::int64_t(fiberLengths.size()); ++f) {
        const double length = std::max(fiberLengths[f], 0.0);
        auto& out = result.points[f];
        out.reserve(std::size_t(length / spacing) + 1);
        std::int64_t i = graph.firstOf(f);  // first end at or after t
        for (double t = 0.0; t <= length; t += spacing) {
            while (i < graph.lastOf(f) && graph.arcAt(i) < t) ++i;
            out.push_back(graph.at(f, t, i));
        }
    }
    return result;
}

namespace
{

// The bin of `value` among ascending upper `edges`; the last bin is open.
std::size_t binOf(const std::vector<double>& edges, const double value)
{
    return std::size_t(std::lower_bound(edges.begin(), edges.end(), value) - edges.begin());
}

std::vector<FiberCalibrationBin> binsFor(const std::vector<double>& edges)
{
    std::vector<FiberCalibrationBin> bins(edges.size() + 1);
    for (std::size_t b = 0; b < bins.size(); ++b) {
        bins[b].lo = b == 0 ? 0.0 : edges[b - 1];
        bins[b].hi = b < edges.size() ? edges[b] : std::numeric_limits<double>::infinity();
    }
    return bins;
}

}  // namespace

FiberSignCalibration fiberSignCalibration(const std::vector<double>& fiberLengths,
                                          const std::vector<FiberSupportCrossing>& agreeing,
                                          const std::vector<FiberSupportCrossing>& disagreeing,
                                          const std::int64_t minBinFlips)
{
    FiberSupportResult support;
    const SupportGraph graph(fiberLengths, agreeing, support);
    const auto fibers = std::int64_t(fiberLengths.size());

    // Core crossings: both ends where every flip is detectable (no missed length or cost).
    struct End {
        std::int64_t fiber;
        double arc;
        std::int64_t crossing;
    };
    std::vector<FiberSupportCrossing> all(agreeing);
    all.insert(all.end(), disagreeing.begin(), disagreeing.end());
    const auto agreeCount = std::int64_t(agreeing.size());
    const auto inCore = [&](const std::int64_t f, const double t) {
        if (f < 0 || f >= fibers) throw std::invalid_argument("calibration crossing addresses an unknown fiber");
        const auto p = graph.at(f, t, graph.lowerBound(f, t));
        return p.missedLength == 0.0 && p.missedCrossingCost == 0.0;
    };
    FiberSignCalibration out;
    std::vector<char> core(all.size(), 0);
    std::vector<End> ends;
    for (std::int64_t k = 0; k < std::int64_t(all.size()); ++k) {
        const auto& c = all[std::size_t(k)];
        if (!(c.weight > 0.0)) throw std::invalid_argument("calibration crossing weight must be positive");
        if (c.a == c.b || !inCore(c.a, c.arcA) || !inCore(c.b, c.arcB)) continue;
        core[std::size_t(k)] = 1;
        ends.push_back({c.a, c.arcA, k});
        ends.push_back({c.b, c.arcB, k});
        ++out.coreCrossings;
        if (k >= agreeCount) ++out.coreDisagreeing;
    }
    std::sort(ends.begin(), ends.end(), [](const End& x, const End& y) {
        return x.fiber != y.fiber ? x.fiber < y.fiber : x.arc != y.arc ? x.arc < y.arc : x.crossing < y.crossing;
    });

    // Runs of consecutive disagreeing crossings along each fiber: a run of two or
    // more is a flipped stretch (a chain flip at each interior boundary), a crossing
    // that is a run of one on both its fibers is a wrong crossing.
    const std::vector<double> gapEdges{64.0, 256.0, 1024.0, 4096.0};
    const std::vector<double> weightEdges{0.1, 0.15, 0.25, 0.5, 0.9};
    out.gapBins = binsFor(gapEdges);
    out.weightBins = binsFor(weightEdges);
    std::vector<char> inLongRun(all.size(), 0);
    for (std::size_t s = 0; s < ends.size();) {
        std::size_t e = s;
        while (e < ends.size() && ends[e].fiber == ends[s].fiber) ++e;
        for (std::size_t i = s; i + 1 < e; ++i) {
            const double gap = ends[i + 1].arc - ends[i].arc;
            out.coreChainLength += gap;
            auto& bin = out.gapBins[binOf(gapEdges, gap)];
            ++bin.count;
            bin.exposure += gap;
        }
        for (std::size_t i = s; i < e;) {
            if (ends[i].crossing < agreeCount) {
                ++i;
                continue;
            }
            std::size_t j = i;
            while (j < e && ends[j].crossing >= agreeCount) ++j;
            if (j - i >= 2) {
                for (std::size_t r = i; r < j; ++r) inLongRun[std::size_t(ends[r].crossing)] = 1;
                // A switch in the gap before the run and after it, where the fiber goes on.
                for (const std::size_t g : {i, j}) {
                    if (g == s || g == e) continue;
                    const double gap = ends[g].arc - ends[g - 1].arc;
                    ++out.chainSwitches;
                    ++out.gapBins[binOf(gapEdges, gap)].events;
                }
            }
            i = j;
        }
        s = e;
    }
    for (std::size_t k = 0; k < all.size(); ++k) {
        if (!core[k]) continue;
        const double w = all[k].weight;
        out.coreInverseWeight += 1.0 / w;
        auto& bin = out.weightBins[binOf(weightEdges, w)];
        ++bin.count;
        bin.exposure += 1.0 / w;
        if (std::int64_t(k) >= agreeCount && !inLongRun[k]) {
            ++out.wrongCrossings;
            ++bin.events;
        }
    }
    out.lambda = out.coreChainLength > 0.0 ? double(out.chainSwitches) / out.coreChainLength : 0.0;
    out.q = out.coreInverseWeight > 0.0 ? double(out.wrongCrossings) / out.coreInverseWeight : 0.0;
    const auto bound = [&](const std::vector<FiberCalibrationBin>& bins) {
        double rate = 0.0;
        for (const auto& b : bins)
            if (b.events >= minBinFlips && b.exposure > 0.0) rate = std::max(rate, double(b.events) / b.exposure);
        return rate;
    };
    out.lambdaBound = bound(out.gapBins);
    out.qBound = bound(out.weightBins);
    return out;
}

namespace
{

// The cut graph in compressed rows: parallel edges merged, weights summed.
struct CutGraph {
    std::vector<std::int64_t> offset;
    std::vector<std::int64_t> target;
    std::vector<double> capacity;
};

struct CutEdge {
    std::int64_t a = 0, b = 0;
    double capacity = 0.0;
};

CutGraph cutGraphOf(const std::int64_t nodes, std::vector<CutEdge> edges)
{
    for (auto& e : edges)
        if (e.a > e.b) std::swap(e.a, e.b);
    std::sort(edges.begin(), edges.end(),
              [](const CutEdge& x, const CutEdge& y) { return x.a != y.a ? x.a < y.a : x.b < y.b; });
    std::vector<CutEdge> merged;
    for (const auto& e : edges) {
        if (e.a == e.b) continue;
        if (!merged.empty() && merged.back().a == e.a && merged.back().b == e.b) merged.back().capacity += e.capacity;
        else merged.push_back(e);
    }
    CutGraph g;
    g.offset.assign(nodes + 1, 0);
    for (const auto& e : merged) {
        ++g.offset[e.a + 1];
        ++g.offset[e.b + 1];
    }
    std::partial_sum(g.offset.begin(), g.offset.end(), g.offset.begin());
    g.target.resize(g.offset.back());
    g.capacity.resize(g.offset.back());
    std::vector<std::int64_t> fill(g.offset.begin(), g.offset.end() - 1);
    for (const auto& e : merged) {
        g.target[fill[e.a]] = e.b;
        g.capacity[fill[e.a]++] = e.capacity;
        g.target[fill[e.b]] = e.a;
        g.capacity[fill[e.b]++] = e.capacity;
    }
    return g;
}

// Per-thread scratch for the local max flow; `local` maps fibers to region
// indices and is reset after every source.
class LocalCut {
public:
    LocalCut(const CutGraph& graph, const FiberCutOptions& options)
        : g_(graph), o_(options), local_(graph.offset.size() - 1, -1) {}

    double operator()(const std::int64_t source)
    {
        region_.assign(1, source);
        hop_.assign(1, 0);
        local_[source] = 0;
        for (std::size_t i = 0; i < region_.size(); ++i) {
            if (hop_[i] >= o_.maxHops) continue;
            const auto f = region_[i];
            for (auto k = g_.offset[f]; k < g_.offset[f + 1]; ++k) {
                const auto v = g_.target[k];
                if (local_[v] >= 0 || std::int64_t(region_.size()) >= o_.maxFibers) continue;
                local_[v] = std::int64_t(region_.size());
                region_.push_back(v);
                hop_.push_back(hop_[i] + 1);
            }
        }
        const auto n = std::int64_t(region_.size()), sink = n;
        // Local edges: region pairs once, every edge leaving the region to the sink.
        edges_.clear();
        for (std::int64_t i = 0; i < n; ++i) {
            const auto f = region_[i];
            double out = 0.0;
            for (auto k = g_.offset[f]; k < g_.offset[f + 1]; ++k) {
                const auto j = local_[g_.target[k]];
                if (j < 0) out += g_.capacity[k];
                else if (j > i) edges_.push_back({i, j, g_.capacity[k], 0.0});
            }
            if (out > 0.0) edges_.push_back({i, sink, out, 0.0});
        }
        for (const auto f : region_) local_[f] = -1;

        // Arcs in compressed rows over n + 1 nodes.
        arcOffset_.assign(n + 2, 0);
        for (const auto& e : edges_) {
            ++arcOffset_[e.a + 1];
            ++arcOffset_[e.b + 1];
        }
        std::partial_sum(arcOffset_.begin(), arcOffset_.end(), arcOffset_.begin());
        arcEdge_.resize(arcOffset_.back());
        fill_.assign(arcOffset_.begin(), arcOffset_.end() - 1);
        for (std::size_t k = 0; k < edges_.size(); ++k) {
            arcEdge_[fill_[edges_[k].a]++] = std::int64_t(k);
            arcEdge_[fill_[edges_[k].b]++] = std::int64_t(k);
        }

        // Edmonds-Karp: shortest augmenting paths until the cap or no path.
        double flow = 0.0;
        constexpr double eps = 1e-12;
        const auto residual = [&](const Local& e, const std::int64_t from) {
            return from == e.a ? e.capacity - e.flow : e.capacity + e.flow;
        };
        for (int round = 0; round < 100000 && flow < o_.cap; ++round) {
            parentArc_.assign(n + 1, -1);
            parentArc_[0] = -2;
            queue_.assign(1, 0);
            for (std::size_t h = 0; h < queue_.size() && parentArc_[sink] == -1; ++h) {
                const auto u = queue_[h];
                for (auto k = arcOffset_[u]; k < arcOffset_[u + 1]; ++k) {
                    const auto& e = edges_[arcEdge_[k]];
                    const auto v = e.a == u ? e.b : e.a;
                    if (parentArc_[v] != -1 || residual(e, u) <= eps) continue;
                    parentArc_[v] = arcEdge_[k];
                    queue_.push_back(v);
                }
            }
            if (parentArc_[sink] == -1) break;
            double push = o_.cap - flow;
            for (auto v = sink; v != 0;) {
                const auto& e = edges_[parentArc_[v]];
                const auto u = e.a == v ? e.b : e.a;
                push = std::min(push, residual(e, u));
                v = u;
            }
            for (auto v = sink; v != 0;) {
                auto& e = edges_[parentArc_[v]];
                const auto u = e.a == v ? e.b : e.a;
                e.flow += u == e.a ? push : -push;
                v = u;
            }
            flow += push;
        }
        return std::min(flow, o_.cap);
    }

private:
    struct Local {
        std::int64_t a, b;
        double capacity, flow;  // flow > 0 runs from a to b
    };
    const CutGraph& g_;
    const FiberCutOptions& o_;
    std::vector<std::int64_t> local_, region_, hop_, arcOffset_, arcEdge_, fill_, parentArc_, queue_;
    std::vector<Local> edges_;
};

}  // namespace

FiberCutResult fiberSignCut(const std::vector<double>& fiberLengths,
                           const std::vector<FiberSupportCrossing>& crossings,
                           const double spacing,
                           const FiberCutOptions& options)
{
    if (!(options.cap > 0.0) || options.maxHops < 1 || options.maxFibers < 2 || !(options.chainWeightLength > 0.0))
        throw std::invalid_argument("fiber cut needs a positive cap and chain weight length, one hop and two nodes");
    if (!(spacing > 0.0)) throw std::invalid_argument("cut spacing must be positive");
    FiberSupportResult unused;
    const SupportGraph graph(fiberLengths, crossings, unused);
    const auto fibers = std::int64_t(fiberLengths.size());

    // Nodes: the pieces of each fiber between its chain bridges.
    std::vector<std::int64_t> segmentAt(graph.ends()), segmentOfEnd(graph.ends());
    std::vector<CutEdge> edges;
    std::int64_t segments = 0;
    for (std::int64_t f = 0; f < fibers; ++f)
        for (auto i = graph.firstOf(f); i < graph.lastOf(f); ++i) {
            if (i == graph.firstOf(f) || graph.bridgeAfter(i - 1)) ++segments;
            segmentAt[i] = segments - 1;
            segmentOfEnd[graph.endAt(i)] = segments - 1;
            if (i > graph.firstOf(f) && graph.bridgeAfter(i - 1))
                edges.push_back({segments - 2, segments - 1,
                                 options.chainWeightLength / std::max(graph.arcAt(i) - graph.arcAt(i - 1), 1e-9)});
        }
    for (std::size_t k = 0; k < crossings.size(); ++k)
        edges.push_back({segmentOfEnd[2 * k], segmentOfEnd[2 * k + 1], crossings[k].weight});
    const CutGraph cutGraph = cutGraphOf(segments, std::move(edges));

    std::vector<double> segmentCut(segments, 0.0);
    std::atomic<std::int64_t> next{0};
    constexpr std::int64_t chunk = 256;
    const auto work = [&] {
        LocalCut local(cutGraph, options);
        for (std::int64_t begin; (begin = next.fetch_add(chunk)) < segments;)
            for (auto v = begin; v < std::min(begin + chunk, segments); ++v)
                if (cutGraph.offset[v + 1] > cutGraph.offset[v]) segmentCut[v] = local(v);
    };
    const unsigned threads = options.threads > 0 ? options.threads : std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();

    // Samples: a point takes its piece's cut; one on a chain bridge the
    // smaller of its two pieces' and the bridge's own; tails their end piece's.
    FiberCutResult result;
    result.segments = segments;
    result.points.resize(fibers);
    for (std::int64_t f = 0; f < fibers; ++f) {
        const double length = std::max(fiberLengths[f], 0.0);
        auto& out = result.points[f];
        out.reserve(std::size_t(length / spacing) + 1);
        const auto lo = graph.firstOf(f), hi = graph.lastOf(f);
        std::int64_t i = lo;
        for (double t = 0.0; t <= length; t += spacing) {
            while (i < hi && graph.arcAt(i) < t) ++i;
            double c = 0.0;
            if (hi > lo) {
                if (i == lo) c = segmentCut[segmentAt[lo]];
                else if (i == hi) c = segmentCut[segmentAt[hi - 1]];
                else if (!graph.bridgeAfter(i - 1)) c = segmentCut[segmentAt[i]];
                else
                    c = std::min({segmentCut[segmentAt[i - 1]], segmentCut[segmentAt[i]],
                                  options.chainWeightLength / std::max(graph.arcAt(i) - graph.arcAt(i - 1), 1e-9)});
            }
            out.push_back({t, std::min(c, options.cap)});
        }
    }
    return result;
}

}  // namespace vc::fiber_tracer
