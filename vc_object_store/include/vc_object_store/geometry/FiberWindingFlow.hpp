#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace vc::geometry {

// Exact integer winding assignment for an aggregated L1 arc list
// (minimize sum_e capacity_e * |k_b - k_a - target_e|) as a capacitated
// minimum-cost circulation solved with LEMON. An arc may be one-sided: it
// then costs only on one side of its target (a bound, not an equation). Intended for small graphs such
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
    int64_t quantizedObjective{};  // sum capacity*residual over retained arcs
    double objective{};            // sum weight*residual over all input arcs
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

// The residual of one arc at a step k_b - k_a: |step - target| for a two-sided
// arc (side 0), max(0, target - step) for a lower bound (side +1: the step
// should be at least the target) and max(0, step - target) for an upper bound
// (side -1: at most the target).
inline int64_t windingArcResidual(int64_t step, int64_t target, int side)
{
    const int64_t d = step - target;
    return side > 0 ? (d < 0 ? -d : 0) : side < 0 ? (d > 0 ? d : 0) : (d < 0 ? -d : d);
}

// As above with a side per arc (windingArcResidual): 0, +1 or -1. An empty
// `side` means every arc is two-sided.
WindingFlowResult solveWindingMinCostFlow(int64_t nodes, std::span<const int64_t> a, std::span<const int64_t> b,
                                          std::span<const int64_t> target, std::span<const double> weight,
                                          std::span<const int> side, const WindingFlowOptions& options = {});

} // namespace vc::geometry
