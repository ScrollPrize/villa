#include "geometry/FiberWindingFlowLemon.hpp"
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>
#include <lemon/cost_scaling.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vc::geometry::lemon_detail {
namespace {
using Graph = lemon::StaticDigraph;

struct ArcValues {
    using Key = Graph::Arc;
    using Value = std::int64_t;
    const std::int64_t* values; std::int64_t sign;
    std::int64_t operator[](Key arc) const { return sign * values[Graph::index(arc)]; }
};

template<class Solver>
void execute(const Graph& graph, const std::int64_t* capacity, const std::int64_t* target,
             std::int64_t* potential, std::int64_t* flow)
{
    Solver solver(graph);
    solver.lowerMap(ArcValues{capacity, -1}).upperMap(ArcValues{capacity, 1}).costMap(ArcValues{target, -1});
    if (solver.run() != Solver::OPTIMAL) throw std::runtime_error("LEMON did not return OPTIMAL for the winding circulation");
    for (int n = 0; n < graph.nodeNum(); ++n) potential[n] = solver.potential(Graph::node(n));
    for (int e = 0; e < graph.arcNum(); ++e) flow[e] = solver.flow(Graph::arc(e));
}
} // namespace

void solveCirculation(int backend, int nodes, std::size_t arcs, const int* a, const int* b,
                      const std::int64_t* capacity, const std::int64_t* target,
                      std::int64_t* potential, std::int64_t* flow)
{
    std::vector<std::pair<int, int>> ends; ends.reserve(arcs);
    for (std::size_t i = 0; i < arcs; ++i) ends.emplace_back(a[i], b[i]);
    Graph graph; graph.build(nodes, ends.begin(), ends.end());
    std::vector<std::pair<int, int>>().swap(ends);
    if (backend == 0) execute<lemon::NetworkSimplex<Graph, std::int64_t, std::int64_t>>(graph, capacity, target, potential, flow);
    else if (backend == 1) execute<lemon::CostScaling<Graph, std::int64_t, std::int64_t>>(graph, capacity, target, potential, flow);
    else throw std::invalid_argument("unknown LEMON backend index");
}

} // namespace vc::geometry::lemon_detail
