#include "vc_object_store/geometry/FiberWindingFlow.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#ifdef VC_OBJECT_STORE_HAS_LEMON
#include "geometry/FiberWindingFlowLemon.hpp"
#endif

namespace vc::geometry {

WindingFlowBackend parseWindingFlowBackend(const std::string& name)
{
    if (name == "network_simplex") return WindingFlowBackend::NetworkSimplex;
    if (name == "cost_scaling") return WindingFlowBackend::CostScaling;
    throw std::invalid_argument("unknown winding flow backend: " + name + " (network_simplex or cost_scaling)");
}

std::string windingFlowBackendName(WindingFlowBackend backend)
{
    return backend == WindingFlowBackend::NetworkSimplex ? "network_simplex" : "cost_scaling";
}

bool windingMinCostFlowAvailable()
{
#ifdef VC_OBJECT_STORE_HAS_LEMON
    return true;
#else
    return false;
#endif
}

namespace {
using Clock = std::chrono::steady_clock;
constexpr int64_t kSafeSum = int64_t(1) << 60;

WindingFlowResult quantize(int64_t nodes, std::span<const int64_t> a, std::span<const int64_t> b,
                           std::span<const int64_t> target, std::span<const double> weight, int64_t scale)
{
    const size_t m = a.size();
    if (nodes < 1) throw std::invalid_argument("winding flow needs at least one node");
    if (b.size() != m || target.size() != m || weight.size() != m)
        throw std::invalid_argument("winding flow arc columns differ in length");
    if (scale < 1) throw std::invalid_argument("capacity scale must be a positive integer");
    constexpr int64_t indexLimit = std::numeric_limits<int>::max() / 4;
    if (nodes > indexLimit || int64_t(m) > indexLimit)
        throw std::invalid_argument("winding flow graph exceeds LEMON index limits; this backend is for small graphs");
    WindingFlowResult result;
    result.nodes = size_t(nodes); result.arcs = m;
    result.winding = std::make_unique<int64_t[]>(result.nodes);
    result.flow = std::make_unique<int64_t[]>(m);
    result.capacity = std::make_unique<int64_t[]>(m);
    auto& q = result.quantization;
    q.scale = scale; q.originalArcs = m;
    int64_t maxCapacity = 0, maxTarget = 0;
    const double capacityLimit = double(int64_t(1) << 52);
    for (size_t i = 0; i < m; ++i) {
        if (a[i] < 0 || a[i] >= nodes || b[i] < 0 || b[i] >= nodes)
            throw std::invalid_argument("winding flow arc endpoint outside the node range");
        const double w = weight[i];
        if (!std::isfinite(w) || w < 0) throw std::invalid_argument("winding flow weights must be finite and non-negative");
        const double scaled = w * double(scale);
        if (scaled > capacityLimit) throw std::invalid_argument("winding flow weight too large for integer capacities");
        const auto capacity = int64_t(std::floor(scaled + .5));
        q.originalWeight += w;
        q.capacityAbsoluteError += std::abs(double(capacity) / double(scale) - w);
        if (capacity > 0) {
            ++q.retainedArcs; result.capacity[i] = capacity;
            maxCapacity = std::max(maxCapacity, capacity);
            maxTarget = std::max(maxTarget, std::abs(target[i]));
        } else {
            ++q.droppedArcs; q.droppedWeight += w;
        }
    }
    // Conservative guards for integer sums and cost-scaling potential arithmetic.
    if (maxCapacity > 0 && maxCapacity > kSafeSum / int64_t(std::max<size_t>(m, 1)))
        throw std::invalid_argument("winding flow capacities could overflow their integer sum");
    const long double potential = (long double)maxTarget * (long double)(nodes + 1) * (long double)(nodes + 1);
    if (potential > (long double)kSafeSum)
        throw std::invalid_argument("winding flow targets could overflow cost-scaling potentials");
    return result;
}

} // namespace

WindingFlowResult solveWindingMinCostFlow(int64_t nodes, std::span<const int64_t> a, std::span<const int64_t> b,
                                          std::span<const int64_t> target, std::span<const double> weight,
                                          const WindingFlowOptions& options)
{
    const auto started = Clock::now();
    auto result = quantize(nodes, a, b, target, weight, options.capacityScale);
#ifndef VC_OBJECT_STORE_HAS_LEMON
    (void)options;
    throw std::runtime_error("vc_object_store was built without LEMON (VC_OBJECT_STORE_WITH_LEMON=OFF)");
#else
    // StaticDigraph wants arcs grouped by source node: sort an index
    // permutation of the retained arcs and map flows back afterwards.
    std::vector<size_t> order; order.reserve(result.quantization.retainedArcs);
    for (size_t i = 0; i < result.arcs; ++i) if (result.capacity[i] > 0) order.push_back(i);
    std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return a[x] < a[y]; });
    std::vector<int> sortedA(order.size()), sortedB(order.size());
    std::vector<int64_t> capacity(order.size()), cost(order.size());
    for (size_t k = 0; k < order.size(); ++k) {
        const auto i = order[k];
        sortedA[k] = int(a[i]); sortedB[k] = int(b[i]); capacity[k] = result.capacity[i]; cost[k] = target[i];
    }
    const auto built = Clock::now();
    std::vector<int64_t> potential(static_cast<size_t>(nodes));
    std::vector<int64_t> flow(order.size());
    lemon_detail::solveCirculation(options.backend == WindingFlowBackend::NetworkSimplex ? 0 : 1, int(nodes), order.size(),
                                   sortedA.data(), sortedB.data(), capacity.data(), cost.data(), potential.data(), flow.data());
    const auto solved = Clock::now();
    // Potentials of nodes without retained arcs are arbitrary: pin them to 0.
    std::vector<bool> touched(result.nodes, false);
    for (const auto i : order) { touched[size_t(a[i])] = true; touched[size_t(b[i])] = true; }
    for (size_t n = 0; n < result.nodes; ++n) result.winding[n] = touched[n] ? -potential[n] : 0;
    for (size_t k = 0; k < order.size(); ++k) result.flow[order[k]] = flow[k];
    long double quantized = 0, objective = 0;
    for (size_t i = 0; i < result.arcs; ++i) {
        const auto residual = std::abs(result.winding[size_t(b[i])] - result.winding[size_t(a[i])] - target[i]);
        quantized += (long double)result.capacity[i] * (long double)residual;
        objective += (long double)weight[i] * (long double)residual;
    }
    result.quantizedObjective = int64_t(quantized);
    result.objective = double(objective);
    result.graphSeconds = std::chrono::duration<double>(built - started).count();
    result.solveSeconds = std::chrono::duration<double>(solved - built).count();
    result.totalSeconds = std::chrono::duration<double>(Clock::now() - started).count();
    return result;
#endif
}

} // namespace vc::geometry
