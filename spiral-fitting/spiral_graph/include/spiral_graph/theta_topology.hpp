#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <spiral_graph/compact_patch_topology.hpp>

namespace spiral::winding {

struct ThetaTopologyEdge {
    std::uint64_t from_node = 0;
    std::uint64_t to_node = 0;
    std::int32_t delta = 0;
    bool closing = false;
};

struct ThetaTopologyConflict {
    std::uint64_t from_node = 0;
    std::uint64_t to_node = 0;
    std::int32_t expected_delta = 0;
    std::int32_t actual_delta = 0;
    std::int32_t residual = 0;
    std::vector<ThetaTopologyEdge> cycle;
};

// Compact, checkpoint-independent quad topology plus checkpoint-dependent
// wrapped theta and integer lift. Node order is row-major valid-cell order.
class PatchThetaTopology {
public:
    static PatchThetaTopology from_mask(
        std::size_t rows,
        std::size_t columns,
        std::span<const std::uint8_t> valid_quads,
        bool retain_largest_component = true);

    std::size_t rows() const noexcept { return rows_; }
    std::size_t columns() const noexcept { return columns_; }
    std::size_t node_count() const noexcept;
    std::pair<std::size_t, std::size_t> cell(std::size_t node) const;
    std::optional<std::size_t> node_at(std::size_t row, std::size_t column) const;
    const std::vector<std::uint8_t>& valid_mask() const noexcept { return valid_; }

    std::optional<ThetaTopologyConflict> assign_theta(std::span<const float> theta);
    std::int32_t potential(std::size_t node) const;
    std::int32_t potential_at(
        double row,
        double column,
        float sampled_theta) const;
    float theta(std::size_t node) const;
    const std::vector<float>& theta_values() const noexcept { return theta_; }

    static std::int32_t crossing_step(float delta) noexcept;

private:
    std::size_t rows_ = 0;
    std::size_t columns_ = 0;
    std::vector<std::uint8_t> valid_;
    topology::CompactPatchTopology compact_;
    std::vector<std::int32_t> node_for_cell_;
    std::vector<float> theta_;
    std::vector<std::int32_t> potentials_;
    std::vector<std::int32_t> tree_parent_;
};

} // namespace spiral::winding
