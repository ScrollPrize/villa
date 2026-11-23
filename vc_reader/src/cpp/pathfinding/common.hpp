#pragma once
#include "vc_reader/pathfinding.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vcr::pathfinding::detail {
using Label = unsigned long long;
inline constexpr Label infinity = 0x7f800000ffffffffULL;
inline Label label(float distance, std::uint32_t steps) {
    return (Label(std::bit_cast<std::uint32_t>(distance)) << 32) | steps;
}
inline float distance(Label label) { return std::bit_cast<float>(std::uint32_t(label >> 32)); }
inline std::size_t validate(const float* costs, std::size_t rows, std::size_t cols,
                            std::uint32_t source, std::uint32_t target) {
    if (!costs || !rows || !cols || rows > (std::numeric_limits<std::uint32_t>::max() - 1) / cols)
        throw std::invalid_argument("invalid pathfinding grid");
    const auto n = rows * cols;
    if (source >= n || target >= n) throw std::out_of_range("pathfinding endpoint outside grid");
    for (std::size_t i = 0; i < n; ++i)
        if (std::isnan(costs[i]) || costs[i] < 0) throw std::invalid_argument("pathfinding costs must be nonnegative or +infinity");
    return n;
}
template<class Visit> void neighbors(std::uint32_t p, std::size_t rows, std::size_t cols, Visit visit) {
    const auto row = std::int64_t(p / cols), col = std::int64_t(p % cols);
    for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
        const auto y = row + dy, x = col + dx;
        if ((dy || dx) && y >= 0 && x >= 0 && y < std::int64_t(rows) && x < std::int64_t(cols))
            visit(std::uint32_t(y * cols + x));
    }
}
inline std::vector<std::uint32_t> path(const float* costs, std::size_t rows, std::size_t cols,
    std::uint32_t source, std::uint32_t target, const std::vector<Label>& labels) {
    if (!std::isfinite(distance(labels[target]))) return {};
    std::vector<std::uint32_t> result{target};
    while (target != source) {
        auto predecessor = std::numeric_limits<std::uint32_t>::max();
        neighbors(target, rows, cols, [&](auto p) {
            if (std::uint32_t(labels[p]) + 1 == std::uint32_t(labels[target]) &&
                distance(labels[p]) + costs[target] == distance(labels[target]))
                predecessor = std::min(predecessor, p);
        });
        if (predecessor == std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("pathfinding predecessor is missing");
        result.push_back(predecessor);
        target = predecessor;
    }
    std::reverse(result.begin(), result.end());
    return result;
}
}
