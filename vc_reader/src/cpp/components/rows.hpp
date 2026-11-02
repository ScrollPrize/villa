#pragma once
// Neighbour rows of run-based labelling, shared by the CPU and CUDA paths.
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace vcr::components::detail {
// A previously scanned neighbour row (z + dz, y + dy) and whether runs that
// touch only diagonally (one voxel apart in x) connect.
struct NeighborRow { int dz, dy, diagonal; };
struct NeighborRows { std::array<NeighborRow, 4> rows; int count; };

inline NeighborRows neighbor_rows(std::size_t dim, int connectivity) {
    if (dim == 2 && connectivity == 4) return {{{{0, -1, 0}}}, 1};
    if (dim == 2 && connectivity == 8) return {{{{0, -1, 1}}}, 1};
    if (dim == 3 && connectivity == 6) return {{{{0, -1, 0}, {-1, 0, 0}}}, 2};
    if (dim == 3 && connectivity == 18) return {{{{0, -1, 1}, {-1, 0, 1}, {-1, -1, 0}, {-1, 1, 0}}}, 4};
    if (dim == 3 && connectivity == 26) return {{{{0, -1, 1}, {-1, -1, 1}, {-1, 0, 1}, {-1, 1, 1}}}, 4};
    throw std::invalid_argument("connectivity must be 4 or 8 in 2D, 6, 18 or 26 in 3D, got " +
                                std::to_string(connectivity) + " for " + std::to_string(dim) + "D");
}
}
