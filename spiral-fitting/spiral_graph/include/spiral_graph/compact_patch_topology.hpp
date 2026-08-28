#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace spiral::topology {

using IndexStorage = std::variant<std::vector<std::uint32_t>, std::vector<std::uint64_t>>;
inline constexpr std::uint64_t missing_index = std::numeric_limits<std::uint64_t>::max();

inline std::size_t index_size(const IndexStorage& values)
{
    return std::visit([](const auto& held) { return held.size(); }, values);
}

inline std::uint64_t index_at(const IndexStorage& values, std::size_t position)
{
    return std::visit(
        [position](const auto& held) {
            return static_cast<std::uint64_t>(held[position]);
        },
        values);
}

inline std::size_t index_bytes(const IndexStorage& values)
{
    return std::visit(
        [](const auto& held) {
            return held.size()
                * sizeof(typename std::decay_t<decltype(held)>::value_type);
        },
        values);
}

inline IndexStorage compact_indices(
    std::vector<std::uint64_t>&& values, std::uint64_t maximum)
{
    if (maximum <= std::numeric_limits<std::uint32_t>::max()) {
        std::vector<std::uint32_t> compact;
        compact.reserve(values.size());
        for (const std::uint64_t value : values) {
            compact.push_back(static_cast<std::uint32_t>(value));
        }
        return compact;
    }
    return std::move(values);
}

struct CompactPatchTopology {
    std::uint64_t height = 0;
    std::uint64_t width = 0;
    std::uint64_t row_lo = 0;
    std::uint64_t column_lo = 0;
    std::uint64_t rectangle_width = 0;
    bool rectangular = false;
    IndexStorage valid_cells;
    IndexStorage preorder_ordinals;
    IndexStorage parent_positions;
    IndexStorage subtree_ends;
};

inline std::uint64_t valid_ordinal(
    const CompactPatchTopology& patch, std::uint64_t linear)
{
    return std::visit([linear](const auto& held) -> std::uint64_t {
        using Value = typename std::decay_t<decltype(held)>::value_type;
        if (linear > static_cast<std::uint64_t>(std::numeric_limits<Value>::max())) {
            return missing_index;
        }
        const auto value = static_cast<Value>(linear);
        const auto found = std::lower_bound(held.begin(), held.end(), value);
        return found != held.end() && *found == value
            ? static_cast<std::uint64_t>(found - held.begin()) : missing_index;
    }, patch.valid_cells);
}

inline std::uint64_t rectangle_preorder_ordinal(
    const CompactPatchTopology& patch, std::uint64_t position)
{
    const std::uint64_t row = position / patch.rectangle_width;
    const std::uint64_t offset = position % patch.rectangle_width;
    const std::uint64_t column = row % 2 == 0
        ? offset : patch.rectangle_width - 1 - offset;
    return row * patch.rectangle_width + column;
}

inline void build_ragged_tree(CompactPatchTopology& patch)
{
    const std::size_t count = index_size(patch.valid_cells);
    constexpr std::uint32_t no_ordinal = std::numeric_limits<std::uint32_t>::max();
    const std::uint64_t area = patch.height * patch.width;
    std::vector<std::uint32_t> dense_ordinals;
    if (area < no_ordinal && count < no_ordinal) {
        dense_ordinals.assign(static_cast<std::size_t>(area), no_ordinal);
        for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
            dense_ordinals[index_at(patch.valid_cells, ordinal)]
                = static_cast<std::uint32_t>(ordinal);
        }
    }
    const auto resolve = [&patch, &dense_ordinals](std::uint64_t linear) {
        if (dense_ordinals.empty()) return valid_ordinal(patch, linear);
        const std::uint32_t ordinal = dense_ordinals[static_cast<std::size_t>(linear)];
        return ordinal == no_ordinal ? missing_index
                                     : static_cast<std::uint64_t>(ordinal);
    };
    struct Frame {
        std::uint64_t ordinal = 0;
        std::uint64_t preorder_position = 0;
        std::array<std::uint64_t, 8> neighbors{};
        std::uint8_t neighbor_count = 0;
        std::uint8_t next = 0;
    };
    const auto make_frame = [&patch, &resolve](
                                std::uint64_t ordinal,
                                std::uint64_t preorder_position) {
        Frame frame;
        frame.ordinal = ordinal;
        frame.preorder_position = preorder_position;
        const std::uint64_t linear = index_at(patch.valid_cells, ordinal);
        const std::int64_t row = static_cast<std::int64_t>(linear / patch.width);
        const std::int64_t column = static_cast<std::int64_t>(linear % patch.width);
        // This order is the established PatchSamplingAtlas order and therefore
        // fixes roots, witnesses, and warm/cold reproducibility.
        constexpr int offsets[8][2] = {
            {0, 1}, {1, -1}, {1, 0}, {1, 1},
            {-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
        };
        for (const auto& offset : offsets) {
            const std::int64_t next_row = row + offset[0];
            const std::int64_t next_column = column + offset[1];
            if (next_row < 0 || next_column < 0
                || static_cast<std::uint64_t>(next_row) >= patch.height
                || static_cast<std::uint64_t>(next_column) >= patch.width) {
                continue;
            }
            const std::uint64_t next_linear
                = static_cast<std::uint64_t>(next_row) * patch.width
                + static_cast<std::uint64_t>(next_column);
            const std::uint64_t next_ordinal = resolve(next_linear);
            if (next_ordinal != missing_index) {
                frame.neighbors[frame.neighbor_count++] = next_ordinal;
            }
        }
        return frame;
    };

    std::vector<std::uint8_t> visited(count, 0);
    std::vector<std::uint64_t> preorder;
    std::vector<std::uint64_t> parents;
    std::vector<Frame> stack;
    preorder.reserve(count);
    parents.reserve(count);
    stack.reserve(std::min<std::size_t>(count, 1'000'000));
    for (std::uint64_t seed = 0; seed < count; ++seed) {
        if (visited[seed]) continue;
        visited[seed] = 1;
        const std::uint64_t root_position = preorder.size();
        preorder.push_back(seed);
        parents.push_back(missing_index);
        stack.push_back(make_frame(seed, root_position));
        while (!stack.empty()) {
            Frame& frame = stack.back();
            if (frame.next >= frame.neighbor_count) {
                stack.pop_back();
                continue;
            }
            const std::uint64_t child = frame.neighbors[frame.next++];
            if (visited[child]) continue;
            visited[child] = 1;
            const std::uint64_t child_position = preorder.size();
            preorder.push_back(child);
            parents.push_back(frame.preorder_position);
            stack.push_back(make_frame(child, child_position));
        }
    }
    std::vector<std::uint64_t> subtree_sizes(count, 1);
    for (std::size_t child = count; child-- > 0;) {
        if (parents[child] != missing_index) {
            subtree_sizes[parents[child]] += subtree_sizes[child];
        }
    }
    std::vector<std::uint64_t> subtree_ends(count);
    for (std::size_t position = 0; position < count; ++position) {
        subtree_ends[position] = position + subtree_sizes[position] - 1;
    }
    const std::uint64_t maximum = count - 1;
    patch.preorder_ordinals = compact_indices(std::move(preorder), maximum);
    patch.parent_positions = compact_indices(std::move(parents), maximum);
    patch.subtree_ends = compact_indices(std::move(subtree_ends), maximum);
}

template <typename IsValid>
bool build_compact_patch_topology(
    std::uint64_t height,
    std::uint64_t width,
    IsValid&& is_valid,
    CompactPatchTopology& patch)
{
    patch = {};
    patch.height = height;
    patch.width = width;
    std::vector<std::uint64_t> cells;
    std::uint64_t row_hi = 0;
    std::uint64_t column_hi = 0;
    patch.row_lo = height;
    patch.column_lo = width;
    for (std::uint64_t row = 0; row < height; ++row) {
        for (std::uint64_t column = 0; column < width; ++column) {
            if (!is_valid(row, column)) continue;
            cells.push_back(row * width + column);
            patch.row_lo = std::min(patch.row_lo, row);
            patch.column_lo = std::min(patch.column_lo, column);
            row_hi = std::max(row_hi, row + 1);
            column_hi = std::max(column_hi, column + 1);
        }
    }
    if (cells.empty()) return false;
    patch.rectangle_width = column_hi - patch.column_lo;
    patch.rectangular = cells.size()
        == (row_hi - patch.row_lo) * patch.rectangle_width;
    const std::uint64_t maximum = cells.back();
    patch.valid_cells = compact_indices(std::move(cells), maximum);
    if (!patch.rectangular) build_ragged_tree(patch);
    return true;
}

} // namespace spiral::topology
