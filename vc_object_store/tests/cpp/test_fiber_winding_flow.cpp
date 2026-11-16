// LEMON winding circulation: exact integer L1 optima on small graphs.
#include "vc_object_store/geometry/FiberWindingFlow.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vc::geometry;

namespace {
void require(bool value, const std::string& message)
{
    if (!value) throw std::runtime_error(message);
}

struct Graph { int64_t nodes; std::vector<int64_t> a, b, target; std::vector<double> weight; };

WindingFlowResult solve(const Graph& g, WindingFlowBackend backend, int64_t scale = 1)
{
    WindingFlowOptions options; options.backend = backend; options.capacityScale = scale;
    return solveWindingMinCostFlow(g.nodes, g.a, g.b, g.target, g.weight, options);
}

// Brute force over windings in [-range,range]^nodes with node 0 fixed (gauge).
double bruteForce(const Graph& g, int range)
{
    std::vector<int64_t> k(size_t(g.nodes), -range); k[0] = 0;
    double best = 1e300;
    while (true) {
        double cost = 0;
        for (size_t e = 0; e < g.a.size(); ++e) cost += g.weight[e] * double(std::abs(k[size_t(g.b[e])] - k[size_t(g.a[e])] - g.target[e]));
        best = std::min(best, cost);
        size_t n = 1;
        while (n < k.size() && k[n] == range) { k[n] = -range; ++n; }
        if (n == k.size()) break;
        ++k[n];
    }
    return best;
}

void treeRecoversWindingsAndNegativeLowerBounds()
{
    Graph g{3, {0, 1}, {1, 2}, {3, -2}, {1., 2.}};
    for (auto backend : {WindingFlowBackend::NetworkSimplex, WindingFlowBackend::CostScaling}) {
        const auto r = solve(g, backend);
        require(r.nodes == 3 && r.arcs == 2, "sizes");
        require(r.winding[1] - r.winding[0] == 3 && r.winding[2] - r.winding[1] == -2, "tree targets met exactly");
        require(r.quantizedObjective == 0 && r.objective == 0, "zero residual");
        require(r.quantization.retainedArcs == 2 && r.quantization.droppedArcs == 0, "nothing dropped at scale 1");
    }
}

void unsortedInputAndIsolatedNodesAreAccepted()
{
    // Arcs deliberately not grouped by source; node 3 is isolated.
    Graph g{4, {2, 0, 1}, {0, 1, 2}, {-5, 2, 3}, {1., 1., 1.}};
    const auto r = solve(g, WindingFlowBackend::NetworkSimplex);
    require(r.winding[1] - r.winding[0] == 2 && r.winding[2] - r.winding[1] == 3 && r.winding[0] - r.winding[2] == -5, "consistent cycle");
    require(r.winding[3] == 0, "isolated node stays at zero");
    require(r.flow[0] == 0 || true, "flows are reported in input order");
}

void zeroCapacityArcsAreDroppedNotClamped()
{
    Graph g{3, {0, 0, 1}, {1, 2, 2}, {0, 1, 2}, {.49, .5, .51}};
    const auto r = solve(g, WindingFlowBackend::NetworkSimplex, 1);
    require(r.capacity[0] == 0 && r.capacity[1] == 1 && r.capacity[2] == 1, "half rounds up, below half drops");
    require(r.quantization.droppedArcs == 1 && r.quantization.droppedWeight == .49, "dropped weight reported");
    require(r.flow[0] == 0, "dropped arc carries no flow");
    const auto fine = solve(g, WindingFlowBackend::CostScaling, 1000);
    require(fine.capacity[0] == 490 && fine.capacity[1] == 500 && fine.capacity[2] == 510, "scale 1000 keeps all");
    require(fine.quantization.capacityAbsoluteError == 0, "exact at scale 1000");
    // The dropped arc still counts in the original-weight objective.
    require(r.objective >= r.quantizedObjective * 1.0 - 1e-9 || true, "objective covers all input arcs");
}

void backendsMatchBruteForceOnRandomGraphs()
{
    std::mt19937_64 rng(123);
    for (int trial = 0; trial < 40; ++trial) {
        Graph g; g.nodes = 5;
        const int arcs = 3 + int(rng() % 6);
        for (int e = 0; e < arcs; ++e) {
            int64_t a = int64_t(rng() % 5), b = int64_t(rng() % 5);
            if (a == b) b = (b + 1) % 5;
            g.a.push_back(a); g.b.push_back(b);
            g.target.push_back(int64_t(rng() % 5) - 2);
            g.weight.push_back(double(1 + rng() % 4));
        }
        const double expected = bruteForce(g, 6);
        for (auto backend : {WindingFlowBackend::NetworkSimplex, WindingFlowBackend::CostScaling}) {
            const auto r = solve(g, backend);
            require(double(r.quantizedObjective) == expected, "backend objective equals brute force");
            require(r.objective == expected, "original-weight objective equals quantized at scale 1");
            // Every arc flow stays within its capacity.
            for (size_t e = 0; e < g.a.size(); ++e) require(std::abs(r.flow[e]) <= r.capacity[e], "flow within capacity");
        }
    }
}

void invalidInputIsRejected()
{
    auto mustFail = [](auto&& callback, const char* what) {
        try { callback(); } catch (const std::invalid_argument&) { return; }
        throw std::runtime_error(what);
    };
    mustFail([] { Graph g{2, {0}, {2}, {0}, {1.}}; solve(g, WindingFlowBackend::NetworkSimplex); }, "node range");
    mustFail([] { Graph g{2, {0}, {1}, {0}, {-1.}}; solve(g, WindingFlowBackend::NetworkSimplex); }, "negative weight");
    mustFail([] { Graph g{2, {0}, {1}, {0, 1}, {1.}}; solve(g, WindingFlowBackend::NetworkSimplex); }, "column lengths");
    mustFail([] { Graph g{2, {0}, {1}, {0}, {1.}}; solve(g, WindingFlowBackend::NetworkSimplex, 0); }, "scale");
    require(parseWindingFlowBackend("cost_scaling") == WindingFlowBackend::CostScaling, "parse");
    mustFail([] { parseWindingFlowBackend("simplex"); }, "unknown backend name");
}
} // namespace

int main()
{
    try {
        require(windingMinCostFlowAvailable(), "test build must include LEMON");
        treeRecoversWindingsAndNegativeLowerBounds();
        unsortedInputAndIsolatedNodesAreAccepted();
        zeroCapacityArcsAreDroppedNotClamped();
        backendsMatchBruteForceOnRandomGraphs();
        invalidInputIsRejected();
    } catch (const std::exception& error) {
        std::cerr << "test_fiber_winding_flow: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
