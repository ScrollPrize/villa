#pragma once
// Internal C++17 boundary around LEMON (its 1.3.1 headers use std::allocator
// members that C++20 removed). Plain pointers only; no std::span here.
#include <cstddef>
#include <cstdint>

namespace vc::geometry::lemon_detail {

// backend: 0 network simplex, 1 cost scaling. Arcs must be grouped by source
// node in non-decreasing order (StaticDigraph). Solves the circulation with
// bounds [-capacity, capacity] and cost -target per arc; writes node
// potentials and arc flows. Throws std::runtime_error when not OPTIMAL.
void solveCirculation(int backend, int nodes, std::size_t arcs, const int* a, const int* b,
                      const std::int64_t* capacity, const std::int64_t* target,
                      std::int64_t* potential, std::int64_t* flow);

} // namespace vc::geometry::lemon_detail
