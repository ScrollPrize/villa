// Optional LEMON backend. Integer costs/capacities, no floating-point solver math.
#include <lemon/static_graph.h>
#include <lemon/network_simplex.h>
template<class S, class C, class E>
void fiber_lemon_progress(const S&, double, C, const E&);
#include <lemon/cost_scaling.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>

using Graph = lemon::StaticDigraph;
using Int = std::int64_t;
using Clock = std::chrono::steady_clock;

// Read-only snapshots from the solver thread, never concurrent access to state.
struct Progress {
    std::string path;
    Int nodes=0, arcs=0;
    const Int *a=nullptr, *b=nullptr, *target=nullptr;
    const double* weight=nullptr;
    double denominator=1, reference=1, interval=20;
    const Graph* graph=nullptr;
    Clock::time_point start, last;
};
static thread_local Progress progress;

extern "C" void fiber_lemon_set_progress(const char* path, Int nodes, Int arcs,
        const Int* a, const Int* b, const Int* target, const double* weight,
        double denominator, double reference, double interval) {
    progress=Progress{};
    progress.path=path;
    progress.nodes=nodes; progress.arcs=arcs;
    progress.a=a; progress.b=b; progress.target=target; progress.weight=weight;
    progress.denominator=denominator; progress.reference=reference; progress.interval=interval;
    progress.start=Clock::now(); progress.last=progress.start-std::chrono::hours(1);
}

template<class S, class C, class E>
void fiber_lemon_progress(const S& solver, double potential_scale, C epsilon, const E& excess) {
    if (progress.path.empty() || !progress.graph) return;
    const auto now=Clock::now();
    if (std::chrono::duration<double>(now-progress.last).count()<progress.interval) return;
    progress.last=now;
    std::vector<Int> winding(progress.nodes);
    const double origin=static_cast<double>(solver.potential(Graph::node(0)))/potential_scale;
    for (Int n=0;n<progress.nodes;++n)
        winding[n]=std::llround(origin-static_cast<double>(solver.potential(Graph::node(n)))/potential_scale);
    long double objective=0, imbalance=0;
    Int infeasible=0;
    for (Int a=0;a<progress.arcs;++a)
        objective+=progress.weight[a]*std::abs(static_cast<double>(winding[progress.b[a]]-winding[progress.a[a]]-progress.target[a]));
    for (auto value:excess) {
        if (value!=0) ++infeasible;
        imbalance+=std::abs(static_cast<long double>(value));
    }
    std::ofstream out(progress.path+".tmp");
    if (!out) throw std::runtime_error("Cannot write LEMON progress snapshot");
    out<<std::setprecision(16)<<"{\"stage\":\"solve\",\"seconds\":"
       <<std::chrono::duration<double>(now-progress.start).count()
       <<",\"epsilon\":"<<epsilon<<",\"objective_original_weights\":"<<objective
       <<",\"normalized_objective\":"<<objective/progress.denominator
       <<",\"factor_vs_small\":"<<objective/progress.denominator/progress.reference
       <<",\"infeasible_entries\":"<<infeasible<<",\"imbalance_l1_quantized_units\":"<<imbalance
       <<",\"optimal\":false,\"objective_kind\":\"rounded_current_potentials_original_weight_L1\"}\n";
    out.close();
    if (std::rename((progress.path+".tmp").c_str(),progress.path.c_str())!=0)
        throw std::runtime_error("Cannot publish LEMON progress snapshot");
}

struct ArcValues {
    using Key = Graph::Arc;
    using Value = Int;
    const Int* values;
    Int sign;
    Int operator[](Key arc) const { return sign * values[Graph::index(arc)]; }
};

template<class Solver>
void execute(const Graph& graph, const Int* capacity, const Int* target,
             Int* winding, Int* flow, double* timing) {
    const auto start = Clock::now();
    Solver solver(graph);
    solver.lowerMap(ArcValues{capacity, -1})
          .upperMap(ArcValues{capacity, 1})
          .costMap(ArcValues{target, -1});
    const auto ready = Clock::now();
    if (solver.run() != Solver::OPTIMAL) {
        throw std::runtime_error("LEMON did not return OPTIMAL");
    }
    const auto solved = Clock::now();
    for (int n=0; n<graph.nodeNum(); ++n) {
        winding[n] = -solver.potential(Graph::node(n));
    }
    for (int a=0; a<graph.arcNum(); ++a) flow[a] = solver.flow(Graph::arc(a));
    timing[1] = std::chrono::duration<double>(ready-start).count();
    timing[2] = std::chrono::duration<double>(solved-ready).count();
    timing[3] = std::chrono::duration<double>(Clock::now()-solved).count();
}

extern "C" int fiber_lemon_solve(int backend, Int nodes, Int arcs,
        const Int* a, const Int* b, const Int* target, const Int* capacity,
        Int* winding, Int* flow, double* timing, char* error, std::size_t error_size) {
    try {
        const auto start = Clock::now();
        if (nodes < 1 || nodes > std::numeric_limits<int>::max()/4 ||
            arcs < 0 || arcs > std::numeric_limits<int>::max()/4) {
            throw std::invalid_argument("Graph exceeds LEMON index limits");
        }
        std::vector<std::pair<int,int>> ends;
        ends.reserve(static_cast<std::size_t>(arcs));
        for (Int i=0; i<arcs; ++i) {
            if (a[i]<0 || a[i]>=nodes || b[i]<0 || b[i]>=nodes || capacity[i]<=0 ||
                (i>0 && a[i]<a[i-1])) {
                throw std::invalid_argument("Invalid or unsorted graph input");
            }
            ends.emplace_back(static_cast<int>(a[i]), static_cast<int>(b[i]));
        }
        Graph graph;
        graph.build(static_cast<int>(nodes), ends.begin(), ends.end());
        progress.graph=&graph;
        std::vector<std::pair<int,int>>().swap(ends);
        timing[0] = std::chrono::duration<double>(Clock::now()-start).count();
        if (backend == 0) {
            execute<lemon::NetworkSimplex<Graph,Int,Int>>(graph,capacity,target,winding,flow,timing);
        } else if (backend == 1) {
            execute<lemon::CostScaling<Graph,Int,Int>>(graph,capacity,target,winding,flow,timing);
        } else throw std::invalid_argument("Unknown backend");
        timing[4] = std::chrono::duration<double>(Clock::now()-start).count();
        progress=Progress{};
        return 0;
    } catch (const std::exception& ex) {
        progress=Progress{};
        if (error_size) std::snprintf(error,error_size,"%s",ex.what());
        return 1;
    }
}
