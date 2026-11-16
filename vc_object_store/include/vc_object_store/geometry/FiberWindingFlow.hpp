#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace vc::geometry {

// Exact integer winding assignment for an aggregated L1 arc list
// (minimize sum_e capacity_e * |k_b - k_a - target_e|) as a capacitated
// minimum-cost circulation solved with LEMON. Intended for small graphs such
// as manual scaffolds: node and arc counts are limited to LEMON's int indices
// and the solver is single-threaded.
enum class WindingFlowBackend { NetworkSimplex, CostScaling };

WindingFlowBackend parseWindingFlowBackend(const std::string& name);
std::string windingFlowBackendName(WindingFlowBackend backend);

struct WindingFlowOptions {
    WindingFlowBackend backend = WindingFlowBackend::NetworkSimplex;
    // Capacities are round(weight * capacityScale); arcs rounding to zero are
    // dropped and reported, never clamped to one.
    int64_t capacityScale = 1000;
};

struct WindingFlowQuantization {
    int64_t scale{};
    size_t originalArcs{}, retainedArcs{}, droppedArcs{};
    double originalWeight{}, droppedWeight{}, capacityAbsoluteError{};
};

struct WindingFlowResult {
    size_t nodes{}, arcs{};
    std::unique_ptr<int64_t[]> winding;   // per node; isolated nodes are 0
    std::unique_ptr<int64_t[]> flow;      // per input arc; 0 for dropped arcs
    std::unique_ptr<int64_t[]> capacity;  // per input arc; 0 for dropped arcs
    WindingFlowQuantization quantization;
    int64_t quantizedObjective{};  // sum capacity*|k_b-k_a-target| over retained arcs
    double objective{};            // sum weight*|k_b-k_a-target| over all input arcs
    double graphSeconds{}, solveSeconds{}, totalSeconds{};
};

// True when this library was built with LEMON (VC_OBJECT_STORE_WITH_LEMON).
bool windingMinCostFlowAvailable();

// a/b are node indices in [0,nodes), target integer steps k_b-k_a, weight
// finite and >= 0. Arcs may be in any order; flows are returned in input order.
// Throws std::invalid_argument for invalid input or index/overflow limits and
// std::runtime_error when the backend does not report OPTIMAL or when the
// library was built without LEMON.
WindingFlowResult solveWindingMinCostFlow(int64_t nodes, std::span<const int64_t> a, std::span<const int64_t> b,
                                          std::span<const int64_t> target, std::span<const double> weight,
                                          const WindingFlowOptions& options = {});

} // namespace vc::geometry
