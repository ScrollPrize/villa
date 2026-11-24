#include "vc/fiber_tracer/FiberRibbonCrossings.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <thread>

namespace vc::fiber_tracer
{
namespace
{

struct Segment {
    std::int64_t fiber = 0, first = 0;  // samples first, first + 1 of the fiber
};

constexpr std::int64_t kAxisBits = 21, kAxisMask = (std::int64_t(1) << kAxisBits) - 1;

std::int64_t packCell(const std::int64_t x, const std::int64_t y, const std::int64_t z)
{
    return (x << (2 * kAxisBits)) | (y << kAxisBits) | z;
}

// Parameter along q0 -> q1 in [0, 1) where it passes through the triangle.
std::optional<double> hitTriangle(const cv::Vec3d& q0, const cv::Vec3d& q1, const cv::Vec3d& a, const cv::Vec3d& b,
                                  const cv::Vec3d& c)
{
    const cv::Vec3d d = q1 - q0, e1 = b - a, e2 = c - a, p = d.cross(e2);
    const double det = e1.dot(p);
    if (std::abs(det) < 1e-12) return std::nullopt;
    const cv::Vec3d s = q0 - a;
    const double u = s.dot(p) / det;
    if (u < 0.0 || u > 1.0) return std::nullopt;
    const cv::Vec3d q = s.cross(e1);
    const double v = d.dot(q) / det;
    if (v < 0.0 || u + v > 1.0) return std::nullopt;
    const double t = e2.dot(q) / det;
    if (t < 0.0 || t >= 1.0) return std::nullopt;
    return t;
}

cv::Vec3d unit(const cv::Vec3d& v)
{
    const double n = cv::norm(v);
    return n > 0.0 ? v / n : v;
}

}  // namespace

std::vector<FiberRibbonCrossing> fiberRibbonCrossings(const std::span<const std::vector<FiberStripSample>> fibers,
                                                      const FiberRibbonOptions& options)
{
    if (!(options.halfHeight > 0.0) || !(options.maxSegmentLength > 0.0) ||
        !(options.maxTangentCosine > 0.0 && options.maxTangentCosine <= 1.0))
        throw std::invalid_argument("ribbon crossings need a positive height and segment length and a cosine in (0, 1]");

    std::vector<Segment> segments;
    cv::Vec3d lo{0.0, 0.0, 0.0};
    bool any = false;
    for (std::size_t f = 0; f < fibers.size(); ++f)
        for (std::size_t i = 0; i + 1 < fibers[f].size(); ++i) {
            const double length = cv::norm(fibers[f][i + 1].point - fibers[f][i].point);
            if (!(length > 0.0) || length > options.maxSegmentLength) continue;
            segments.push_back({std::int64_t(f), std::int64_t(i)});
            for (const auto& p : {fibers[f][i].point, fibers[f][i + 1].point})
                for (int d = 0; d < 3; ++d) lo[d] = any ? std::min(lo[d], p[d]) : p[d];
            any = true;
        }
    const auto count = std::int64_t(segments.size());
    if (count == 0) return {};

    // Uniform grid over the segment midpoints.
    const double cellSize = 64.0, reach = 0.5 * options.maxSegmentLength;
    const auto cellAxis = [&](const double v, const int d) { return std::int64_t(std::floor((v - lo[d]) / cellSize)) + 8; };
    const auto midpoint = [&](const Segment& s) {
        return 0.5 * (fibers[s.fiber][s.first].point + fibers[s.fiber][s.first + 1].point);
    };
    std::vector<std::int64_t> key(segments.size());
    for (std::int64_t k = 0; k < count; ++k) {
        const cv::Vec3d m = midpoint(segments[k]);
        const std::int64_t x = cellAxis(m[0], 0), y = cellAxis(m[1], 1), z = cellAxis(m[2], 2);
        if (std::max({x, y, z}) + 16 > kAxisMask) throw std::invalid_argument("ribbon crossings: the samples span too large a volume");
        key[k] = packCell(x, y, z);
    }
    std::vector<std::int64_t> order(segments.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](const std::int64_t a, const std::int64_t b) {
        return key[a] != key[b] ? key[a] < key[b] : a < b;
    });
    std::vector<std::int64_t> sorted(segments.size());
    for (std::int64_t k = 0; k < count; ++k) sorted[k] = key[order[k]];

    // Segments are grouped by fiber already, so a chunk of them is a chunk of ribbons.
    constexpr std::int64_t chunk = 2048;
    const std::int64_t chunks = (count + chunk - 1) / chunk;
    std::vector<std::vector<FiberRibbonCrossing>> found(chunks);
    std::atomic<std::int64_t> next{0};
    const auto work = [&] {
        for (std::int64_t c; (c = next.fetch_add(1)) < chunks;) {
            auto& out = found[c];
            for (std::int64_t r = c * chunk; r < std::min((c + 1) * chunk, count); ++r) {
                const auto& s0 = fibers[segments[r].fiber][segments[r].first];
                const auto& s1 = fibers[segments[r].fiber][segments[r].first + 1];
                if (!(cv::norm(s0.normal) > 0.0) || !(cv::norm(s1.normal) > 0.0)) continue;
                const cv::Vec3d n0 = unit(s0.normal), n1 = unit(s1.normal);
                const cv::Vec3d a = s0.point - options.halfHeight * n0, b = s1.point - options.halfHeight * n1,
                                cc = s1.point + options.halfHeight * n1, d = s0.point + options.halfHeight * n0;
                std::int64_t low[3], high[3];
                for (int k = 0; k < 3; ++k) {
                    low[k] = cellAxis(std::min({a[k], b[k], cc[k], d[k]}) - reach, k);
                    high[k] = cellAxis(std::max({a[k], b[k], cc[k], d[k]}) + reach, k);
                }
                const cv::Vec3d along = s1.point - s0.point, tangent = unit(along);
                const double along2 = along.dot(along);
                for (std::int64_t x = low[0]; x <= high[0]; ++x)
                    for (std::int64_t y = low[1]; y <= high[1]; ++y) {
                        const std::int64_t last = packCell(x, y, high[2]);
                        auto k = std::lower_bound(sorted.begin(), sorted.end(), packCell(x, y, low[2])) - sorted.begin();
                        for (; k < count && sorted[k] <= last; ++k) {
                            const Segment& other = segments[order[k]];
                            if (other.fiber == segments[r].fiber) continue;
                            const auto& q0 = fibers[other.fiber][other.first];
                            const auto& q1 = fibers[other.fiber][other.first + 1];
                            // Both triangles are wound the same way, so their normals are
                            // tangent x normal on a regular span and turn over where the
                            // ribbon folds back on itself. A segment may pass through both.
                            for (int half = 0; half < 2; ++half) {
                            const cv::Vec3d& second = half == 0 ? b : cc;
                            const cv::Vec3d& third = half == 0 ? cc : d;
                            const cv::Vec3d facing = (second - a).cross(third - a);
                            const auto t = hitTriangle(q0.point, q1.point, a, second, third);
                            if (!t) continue;
                            const cv::Vec3d direction = unit(q1.point - q0.point);
                            const double cosine = tangent.dot(direction);
                            if (std::abs(cosine) > options.maxTangentCosine) continue;
                            const cv::Vec3d hit = q0.point + *t * (q1.point - q0.point);
                            const double u = std::clamp((hit - s0.point).dot(along) / along2, 0.0, 1.0);
                            const cv::Vec3d normal = unit(n0 + u * (n1 - n0));
                            FiberRibbonCrossing crossing;
                            crossing.ribbon = segments[r].fiber;
                            crossing.fiber = other.fiber;
                            crossing.ribbonArc = s0.arc + u * (s1.arc - s0.arc);
                            crossing.fiberArc = q0.arc + *t * (q1.arc - q0.arc);
                            crossing.height = (hit - (s0.point + u * along)).dot(normal);
                            crossing.direction = direction.dot(facing) >= 0.0 ? 1 : -1;
                            crossing.tangentCosine = cosine;
                            crossing.normalDot = normal.dot(unit(q0.normal + *t * (q1.normal - q0.normal)));
                            crossing.point = hit;
                            out.push_back(crossing);
                            }
                        }
                    }
            }
            std::stable_sort(out.begin(), out.end(), [](const FiberRibbonCrossing& x, const FiberRibbonCrossing& y) {
                if (x.ribbon != y.ribbon) return x.ribbon < y.ribbon;
                if (x.ribbonArc != y.ribbonArc) return x.ribbonArc < y.ribbonArc;
                return x.fiber != y.fiber ? x.fiber < y.fiber : x.fiberArc < y.fiberArc;
            });
        }
    };
    const unsigned threads = options.threads > 0 ? options.threads : std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();

    std::vector<FiberRibbonCrossing> crossings;
    std::size_t total = 0;
    for (const auto& list : found) total += list.size();
    crossings.reserve(total);
    for (const auto& list : found) crossings.insert(crossings.end(), list.begin(), list.end());
    // A ribbon fiber may straddle two chunks; within each the order is already right.
    std::stable_sort(crossings.begin(), crossings.end(), [](const FiberRibbonCrossing& x, const FiberRibbonCrossing& y) {
        return x.ribbon != y.ribbon ? x.ribbon < y.ribbon : x.ribbonArc < y.ribbonArc;
    });
    return crossings;
}

std::vector<FiberRibbonEncounter> fiberRibbonEncounters(std::vector<FiberRibbonCrossing>& crossings, const double maxArcGap)
{
    if (!(maxArcGap > 0.0)) throw std::invalid_argument("ribbon encounters need a positive arc gap");
    std::sort(crossings.begin(), crossings.end(), [](const FiberRibbonCrossing& x, const FiberRibbonCrossing& y) {
        if (x.ribbon != y.ribbon) return x.ribbon < y.ribbon;
        if (x.fiber != y.fiber) return x.fiber < y.fiber;
        return x.fiberArc != y.fiberArc ? x.fiberArc < y.fiberArc : x.ribbonArc < y.ribbonArc;
    });
    std::vector<FiberRibbonEncounter> encounters;
    const auto count = std::int64_t(crossings.size());
    std::vector<std::int64_t> sideList;
    for (std::int64_t first = 0; first < count;) {
        std::int64_t end = first + 1;
        while (end < count && crossings[end].ribbon == crossings[first].ribbon &&
               crossings[end].fiber == crossings[first].fiber &&
               crossings[end].fiberArc - crossings[end - 1].fiberArc < maxArcGap &&
               std::abs(crossings[end].ribbonArc - crossings[end - 1].ribbonArc) < maxArcGap)
            ++end;
        FiberRibbonEncounter encounter;
        encounter.ribbon = crossings[first].ribbon;
        encounter.fiber = crossings[first].fiber;
        encounter.first = first;
        encounter.count = end - first;
        encounter.closest = first;
        for (std::int64_t i = first; i < end; ++i)
            if (std::abs(crossings[i].height) < std::abs(crossings[encounter.closest].height)) encounter.closest = i;
        for (const int side : {-1, 1}) {
            sideList.clear();
            for (std::int64_t i = first; i < end; ++i)
                if ((crossings[i].height >= 0.0 ? 1 : -1) == side) sideList.push_back(i);
            // Farthest first: every passage cancels against a return, the pair
            // furthest from the fiber going first, so what is left are the
            // nearest crossings of the direction in excess.
            std::int64_t net = 0;
            for (const auto i : sideList) net += crossings[i].direction;
            const int excess = net > 0 ? 1 : -1;
            std::erase_if(sideList, [&](const std::int64_t i) { return crossings[i].direction != excess; });
            std::sort(sideList.begin(), sideList.end(), [&](const std::int64_t x, const std::int64_t y) {
                const double hx = std::abs(crossings[x].height), hy = std::abs(crossings[y].height);
                return hx != hy ? hx < hy : x < y;
            });
            sideList.resize(std::size_t(std::abs(net)));
            encounter.left += std::int64_t(sideList.size());
            if (sideList.size() == 1) encounter.remaining = sideList[0];
        }
        if (encounter.left != 1) encounter.remaining = -1;
        encounters.push_back(encounter);
        first = end;
    }
    return encounters;
}

std::vector<FiberCrossingVerdict> fiberCrossingVerdicts(const std::span<const FiberRibbonCrossing> crossings,
                                                        const std::span<const FiberRibbonEncounter> encounters,
                                                        const std::span<const FiberCrossingEventView> views,
                                                        const FiberCrossingFilterOptions& options)
{
    if (!(options.matchTolerance > 0.0) || !(options.minimumDot >= 0.0))
        throw std::invalid_argument("crossing verdicts need a positive match tolerance and a non-negative dot");
    std::vector<FiberCrossingVerdict> verdicts(views.size());
    const auto pairLess = [](const FiberRibbonEncounter& e, const std::pair<std::int64_t, std::int64_t>& key) {
        return std::pair{e.ribbon, e.fiber} < key;
    };
    for (std::size_t v = 0; v < views.size(); ++v) {
        const auto& view = views[v];
        const std::pair key{view.ribbon, view.fiber};
        const FiberRibbonEncounter* best = nullptr;
        std::int64_t bestCrossing = -1;
        double bestGap = options.matchTolerance;
        for (auto e = std::lower_bound(encounters.begin(), encounters.end(), key, pairLess);
             e != encounters.end() && e->ribbon == view.ribbon && e->fiber == view.fiber; ++e)
            for (std::int64_t i = e->first; i < e->first + e->count; ++i) {
                const double gap = std::max(std::abs(crossings[i].ribbonArc - view.ribbonArc),
                                            std::abs(crossings[i].fiberArc - view.fiberArc));
                if (gap <= bestGap) { bestGap = gap; best = &*e; bestCrossing = i; }
            }
        if (!best) continue;
        auto& verdict = verdicts[v];
        verdict.matched = true;
        verdict.multiple = best->count > 1;
        if (!verdict.multiple) continue;
        verdict.nothingLeft = best->left == 0;
        verdict.severalLeft = best->left > 1;
        bool agree = false, oppose = false, agreeNear = false, opposeNear = false;
        for (std::int64_t i = best->first; i < best->first + best->count; ++i) {
            const bool near = std::abs(crossings[i].height) <= options.nearHeight;
            if (crossings[i].normalDot >= options.minimumDot) { agree = true; agreeNear = agreeNear || near; }
            if (crossings[i].normalDot <= -options.minimumDot) { oppose = true; opposeNear = opposeNear || near; }
        }
        verdict.mixed = agree && oppose;
        verdict.mixedNear = agreeNear && opposeNear;
        const double own = crossings[bestCrossing].normalDot;
        for (std::int64_t i = best->first; i < best->first + best->count; ++i) {
            const double other = crossings[i].normalDot, height = std::abs(crossings[i].height);
            if (std::abs(other) < options.minimumDot || other * own >= 0.0) continue;
            if (verdict.alternativeDot == 0.0 || height < verdict.alternativeHeight) {
                verdict.alternativeDot = std::abs(other);
                verdict.alternativeHeight = height;
            }
        }
    }
    return verdicts;
}

}  // namespace vc::fiber_tracer
