#include <spiral_graph/theta_topology.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>

namespace spiral::winding {
namespace {

std::vector<std::uint8_t> largest_component(
    std::size_t rows,
    std::size_t columns,
    std::span<const std::uint8_t> input)
{
    std::vector<std::uint8_t> output(input.begin(), input.end());
    std::vector<std::uint8_t> visited(input.size(), 0);
    std::vector<std::size_t> best;
    std::deque<std::size_t> queue;
    constexpr int offsets[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
        {0, 1}, {1, -1}, {1, 0}, {1, 1},
    };
    for (std::size_t seed = 0; seed < input.size(); ++seed) {
        if (!input[seed] || visited[seed]) continue;
        std::vector<std::size_t> component;
        visited[seed] = 1;
        queue.push_back(seed);
        while (!queue.empty()) {
            const std::size_t current = queue.front();
            queue.pop_front();
            component.push_back(current);
            const auto row = static_cast<std::int64_t>(current / columns);
            const auto column = static_cast<std::int64_t>(current % columns);
            for (const auto& offset : offsets) {
                const std::int64_t next_row = row + offset[0];
                const std::int64_t next_column = column + offset[1];
                if (next_row < 0 || next_column < 0
                    || static_cast<std::size_t>(next_row) >= rows
                    || static_cast<std::size_t>(next_column) >= columns) {
                    continue;
                }
                const std::size_t next = static_cast<std::size_t>(next_row) * columns
                    + static_cast<std::size_t>(next_column);
                if (input[next] && !visited[next]) {
                    visited[next] = 1;
                    queue.push_back(next);
                }
            }
        }
        // First row-major component wins an exact-size tie.
        if (component.size() > best.size()) best = std::move(component);
    }
    std::fill(output.begin(), output.end(), 0);
    for (const std::size_t cell : best) output[cell] = 1;
    return output;
}

} // namespace

PatchThetaTopology PatchThetaTopology::from_mask(
    std::size_t rows,
    std::size_t columns,
    std::span<const std::uint8_t> valid_quads,
    bool retain_largest_component)
{
    if (rows == 0 || columns == 0 || valid_quads.size() != rows * columns) {
        throw std::invalid_argument("theta topology mask has the wrong shape");
    }
    PatchThetaTopology result;
    result.rows_ = rows;
    result.columns_ = columns;
    result.valid_ = retain_largest_component
        ? largest_component(rows, columns, valid_quads)
        : std::vector<std::uint8_t>(valid_quads.begin(), valid_quads.end());
    if (!topology::build_compact_patch_topology(
            rows, columns,
            [&](std::uint64_t row, std::uint64_t column) {
                return result.valid_[static_cast<std::size_t>(row) * columns
                                     + static_cast<std::size_t>(column)] != 0;
            },
            result.compact_)) {
        throw std::invalid_argument("theta topology contains no valid quads");
    }
    result.node_for_cell_.assign(rows * columns, -1);
    for (std::size_t node = 0; node < topology::index_size(result.compact_.valid_cells); ++node) {
        result.node_for_cell_[topology::index_at(result.compact_.valid_cells, node)]
            = static_cast<std::int32_t>(node);
    }
    return result;
}

std::size_t PatchThetaTopology::node_count() const noexcept
{
    return topology::index_size(compact_.valid_cells);
}

std::pair<std::size_t, std::size_t> PatchThetaTopology::cell(std::size_t node) const
{
    if (node >= node_count()) throw std::out_of_range("theta node is out of range");
    const std::uint64_t linear = topology::index_at(compact_.valid_cells, node);
    return {
        static_cast<std::size_t>(linear / columns_),
        static_cast<std::size_t>(linear % columns_),
    };
}

std::optional<std::size_t> PatchThetaTopology::node_at(
    std::size_t row, std::size_t column) const
{
    if (row >= rows_ || column >= columns_) return std::nullopt;
    const std::int32_t node = node_for_cell_[row * columns_ + column];
    if (node < 0) return std::nullopt;
    return static_cast<std::size_t>(node);
}

std::int32_t PatchThetaTopology::crossing_step(float delta) noexcept
{
    constexpr float pi = 3.14159265358979323846f;
    return static_cast<std::int32_t>(delta > pi)
        - static_cast<std::int32_t>(delta < -pi);
}

std::optional<ThetaTopologyConflict> PatchThetaTopology::assign_theta(
    std::span<const float> theta)
{
    if (theta.size() != node_count()) {
        throw std::invalid_argument("theta count does not match patch topology");
    }
    for (const float value : theta) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("patch theta contains a non-finite value");
        }
    }
    theta_.assign(theta.begin(), theta.end());
    potentials_.assign(node_count(), std::numeric_limits<std::int32_t>::min());
    tree_parent_.assign(node_count(), -1);

    // The compact preorder is the authoritative tree used by fit_spiral's
    // PatchSamplingAtlas. Rectangles use its implicit boustrophedon chain.
    if (compact_.rectangular) {
        for (std::size_t position = 0; position < node_count(); ++position) {
            const std::size_t node = static_cast<std::size_t>(
                topology::rectangle_preorder_ordinal(compact_, position));
            if (position == 0) {
                potentials_[node] = 0;
                continue;
            }
            const std::size_t parent = static_cast<std::size_t>(
                topology::rectangle_preorder_ordinal(compact_, position - 1));
            tree_parent_[node] = static_cast<std::int32_t>(parent);
            potentials_[node] = potentials_[parent]
                + crossing_step(theta_[node] - theta_[parent]);
        }
    } else {
        for (std::size_t position = 0; position < node_count(); ++position) {
            const std::size_t node = static_cast<std::size_t>(
                topology::index_at(compact_.preorder_ordinals, position));
            const std::uint64_t parent_position = topology::index_at(
                compact_.parent_positions, position);
            if (parent_position == topology::missing_index
                || parent_position == std::numeric_limits<std::uint32_t>::max()) {
                potentials_[node] = 0;
                continue;
            }
            const std::size_t parent = static_cast<std::size_t>(
                topology::index_at(compact_.preorder_ordinals,
                                   static_cast<std::size_t>(parent_position)));
            tree_parent_[node] = static_cast<std::int32_t>(parent);
            potentials_[node] = potentials_[parent]
                + crossing_step(theta_[node] - theta_[parent]);
        }
    }

    constexpr int forward_offsets[4][2] = {
        {0, 1}, {1, -1}, {1, 0}, {1, 1},
    };
    for (std::size_t from = 0; from < node_count(); ++from) {
        const auto [row, column] = cell(from);
        for (const auto& offset : forward_offsets) {
            const auto next_row = static_cast<std::int64_t>(row) + offset[0];
            const auto next_column = static_cast<std::int64_t>(column) + offset[1];
            if (next_row < 0 || next_column < 0) continue;
            const auto to = node_at(
                static_cast<std::size_t>(next_row),
                static_cast<std::size_t>(next_column));
            if (!to) continue;
            const std::int32_t expected = crossing_step(theta_[*to] - theta_[from]);
            const std::int32_t actual = potentials_[*to] - potentials_[from];
            if (actual != expected) {
                ThetaTopologyConflict conflict{
                    from, *to, expected, actual, actual - expected, {},
                };
                std::vector<std::uint8_t> from_ancestors(node_count(), 0);
                for (std::int32_t cursor = static_cast<std::int32_t>(from);
                     cursor >= 0; cursor = tree_parent_[static_cast<std::size_t>(cursor)]) {
                    from_ancestors[static_cast<std::size_t>(cursor)] = 1;
                }
                std::int32_t lca = static_cast<std::int32_t>(*to);
                while (lca >= 0 && !from_ancestors[static_cast<std::size_t>(lca)]) {
                    lca = tree_parent_[static_cast<std::size_t>(lca)];
                }
                if (lca < 0) throw std::logic_error("theta tree has disconnected neighbors");
                for (std::int32_t cursor = static_cast<std::int32_t>(from);
                     cursor != lca;) {
                    const std::int32_t parent = tree_parent_[static_cast<std::size_t>(cursor)];
                    conflict.cycle.push_back({
                        static_cast<std::uint64_t>(cursor),
                        static_cast<std::uint64_t>(parent),
                        potentials_[static_cast<std::size_t>(parent)]
                            - potentials_[static_cast<std::size_t>(cursor)],
                        false,
                    });
                    cursor = parent;
                }
                std::vector<std::int32_t> down;
                for (std::int32_t cursor = static_cast<std::int32_t>(*to);
                     cursor != lca;
                     cursor = tree_parent_[static_cast<std::size_t>(cursor)]) {
                    down.push_back(cursor);
                }
                for (auto iterator = down.rbegin(); iterator != down.rend(); ++iterator) {
                    const std::int32_t child = *iterator;
                    const std::int32_t parent = tree_parent_[static_cast<std::size_t>(child)];
                    conflict.cycle.push_back({
                        static_cast<std::uint64_t>(parent),
                        static_cast<std::uint64_t>(child),
                        potentials_[static_cast<std::size_t>(child)]
                            - potentials_[static_cast<std::size_t>(parent)],
                        false,
                    });
                }
                conflict.cycle.push_back({*to, from, -expected, true});
                return conflict;
            }
        }
    }
    return std::nullopt;
}

std::int32_t PatchThetaTopology::potential(std::size_t node) const
{
    if (node >= potentials_.size()) throw std::out_of_range("theta node is out of range");
    if (potentials_[node] == std::numeric_limits<std::int32_t>::min()) {
        throw std::logic_error("theta topology has not been assigned theta");
    }
    return potentials_[node];
}

float PatchThetaTopology::theta(std::size_t node) const
{
    if (node >= theta_.size()) throw std::out_of_range("theta node is out of range");
    return theta_[node];
}

std::int32_t PatchThetaTopology::potential_at(
    double row,
    double column,
    float sampled_theta) const
{
    if (!std::isfinite(row) || !std::isfinite(column)
        || !std::isfinite(sampled_theta)) {
        throw std::invalid_argument("attachment coordinates/theta must be finite");
    }
    const double floor_row = std::floor(row);
    const double floor_column = std::floor(column);
    const auto clamped_row = static_cast<std::size_t>(std::clamp(
        floor_row, 0.0, static_cast<double>(rows_ - 1)));
    const auto clamped_column = static_cast<std::size_t>(std::clamp(
        floor_column, 0.0, static_cast<double>(columns_ - 1)));
    std::array<std::size_t, 2> candidate_rows{clamped_row, clamped_row};
    std::array<std::size_t, 2> candidate_columns{clamped_column, clamped_column};
    constexpr double boundary_epsilon = 1e-5;
    if (clamped_row > 0
        && std::abs(row - std::round(row)) <= boundary_epsilon) {
        candidate_rows[1] = clamped_row - 1;
    }
    if (clamped_column > 0
        && std::abs(column - std::round(column)) <= boundary_epsilon) {
        candidate_columns[1] = clamped_column - 1;
    }
    for (const std::size_t candidate_row : candidate_rows) {
        for (const std::size_t candidate_column : candidate_columns) {
            if (const auto node = node_at(candidate_row, candidate_column)) {
                return potential(*node)
                    + crossing_step(sampled_theta - theta(*node));
            }
        }
    }
    throw std::invalid_argument("attachment is outside valid patch topology");
}

} // namespace spiral::winding
