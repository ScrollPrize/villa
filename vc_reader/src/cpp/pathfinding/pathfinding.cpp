#include "common.hpp"
#include <queue>
#include <utility>
namespace vcr::pathfinding {
std::vector<std::uint32_t> shortestPath(const float* costs, std::size_t rows, std::size_t cols,
                                      std::uint32_t source, std::uint32_t target) {
    const auto n = detail::validate(costs, rows, cols, source, target);
    if (!std::isfinite(costs[source]) || !std::isfinite(costs[target])) return {};
    std::vector<detail::Label> labels(n, detail::infinity);
    using Entry = std::pair<detail::Label, std::uint32_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
    labels[source] = 0;
    queue.emplace(0, source);
    while (!queue.empty()) {
        const auto [value, p] = queue.top(); queue.pop();
        if (value != labels[p]) continue;
        if (p == target) break;
        detail::neighbors(p, rows, cols, [&](auto q) {
            const float distance = detail::distance(value) + costs[q];
            if (!std::isfinite(distance)) return;
            const auto next = detail::label(distance, std::uint32_t(value) + 1);
            if (next < labels[q]) { labels[q] = next; queue.emplace(next, q); }
        });
    }
    return detail::path(costs, rows, cols, source, target, labels);
}
}
