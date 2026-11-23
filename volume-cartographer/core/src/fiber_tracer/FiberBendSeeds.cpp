#include "vc/fiber_tracer/FiberBendSeeds.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace vc::fiber_tracer
{

std::vector<FiberBendSeed> fiberBendSeeds(const std::vector<FiberBendNode>& nodes, const FiberBendSeedOptions& options)
{
    if (!(options.radius > 0.0) || !(options.minAngleDegrees >= 0.0 && options.minAngleDegrees <= 90.0))
        throw std::invalid_argument("bend seeds need a positive radius and an angle in [0, 90]");
    const auto count = std::int64_t(nodes.size());
    if (count == 0) return {};

    // Uniform grid of cells one radius wide; nodes sorted by cell.
    cv::Vec3d lo = nodes[0].point;
    for (const auto& node : nodes)
        for (int d = 0; d < 3; ++d) lo[d] = std::min(lo[d], node.point[d]);
    using Cell = std::array<std::int64_t, 3>;
    const auto cellOf = [&](const cv::Vec3d& p) {
        return Cell{std::int64_t((p[0] - lo[0]) / options.radius), std::int64_t((p[1] - lo[1]) / options.radius),
                    std::int64_t((p[2] - lo[2]) / options.radius)};
    };
    std::vector<Cell> cell(nodes.size());
    for (std::int64_t i = 0; i < count; ++i) cell[i] = cellOf(nodes[i].point);
    std::vector<std::int64_t> order(nodes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](const std::int64_t a, const std::int64_t b) {
        return cell[a] != cell[b] ? cell[a] < cell[b] : a < b;
    });
    std::vector<Cell> sorted(nodes.size());
    for (std::int64_t k = 0; k < count; ++k) sorted[k] = cell[order[k]];

    const double maxCos = std::cos(options.minAngleDegrees * CV_PI / 180.0);
    const double radius2 = options.radius * options.radius;
    constexpr std::int64_t chunk = 4096;
    const std::int64_t chunks = (count + chunk - 1) / chunk;
    std::vector<std::vector<FiberBendSeed>> found(chunks);
    std::atomic<std::int64_t> next{0};
    const auto work = [&] {
        for (std::int64_t c; (c = next.fetch_add(1)) < chunks;) {
            auto& out = found[c];
            for (std::int64_t i = c * chunk; i < std::min((c + 1) * chunk, count); ++i) {
                const auto& a = nodes[i];
                if (!(cv::norm(a.tangent) > 0.0)) continue;
                for (std::int64_t dx = -1; dx <= 1; ++dx)
                    for (std::int64_t dy = -1; dy <= 1; ++dy)
                        for (std::int64_t dz = -1; dz <= 1; ++dz) {
                            const Cell key{cell[i][0] + dx, cell[i][1] + dy, cell[i][2] + dz};
                            auto k = std::lower_bound(sorted.begin(), sorted.end(), key) - sorted.begin();
                            for (; k < count && sorted[k] == key; ++k) {
                                const std::int64_t j = order[k];
                                const auto& b = nodes[j];
                                if (j <= i || b.family != a.family || b.fiber == a.fiber) continue;
                                const cv::Vec3d d = b.point - a.point;
                                const double distance2 = d.dot(d);
                                if (distance2 > radius2 || !(cv::norm(b.tangent) > 0.0)) continue;
                                const double cosine = std::min(1.0, std::abs(a.tangent.dot(b.tangent)));
                                if (cosine > maxCos) continue;
                                out.push_back({0.5 * (a.point + b.point), a.tangent, b.tangent,
                                               std::acos(cosine) * 180.0 / CV_PI, std::sqrt(distance2), a.family,
                                               a.fiber, b.fiber});
                            }
                        }
            }
        }
    };
    const unsigned threads = options.threads > 0 ? options.threads : std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();

    std::vector<FiberBendSeed> seeds;
    std::size_t total = 0;
    for (const auto& list : found) total += list.size();
    seeds.reserve(total);
    for (auto& list : found) {
        // Within a chunk the pairs of one node come out by cell, not by partner index.
        std::stable_sort(list.begin(), list.end(), [](const FiberBendSeed& x, const FiberBendSeed& y) {
            return x.fiberA != y.fiberA ? x.fiberA < y.fiberA : x.fiberB < y.fiberB;
        });
        seeds.insert(seeds.end(), list.begin(), list.end());
    }
    return seeds;
}

}  // namespace vc::fiber_tracer
