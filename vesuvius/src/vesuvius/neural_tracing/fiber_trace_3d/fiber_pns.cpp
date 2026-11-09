// Adapter only: PNS remains an external, unmodified MIT-licensed dependency.
// Reuse its parser/initialization and pivot routines; do not use its printer,
// which omits flows on nonpositive-cost arcs.
#include <limits>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#if defined(OMP) || defined(OMP_AVX2)
#include <omp.h>
#endif
#define main pns_upstream_main
#include PNS_SOURCE_FILE
#undef main

extern "C" int fiber_pns_solve(const char* path, int32_t nodes, int32_t arcs,
        int threads, double time_limit, int64_t* winding, int64_t* shifted_flow,
        double* timing, char* error, std::size_t error_size) {
    try {
#if defined(OMP) || defined(OMP_AVX2)
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
#else
        if (threads!=1) throw std::invalid_argument("Sequential PNS requires one thread");
#endif
        find_entering_arc_time = find_join_node_time = find_leaving_arc_time =
            change_flows_time = change_states_time = update_tree_time = update_pots_time =
            std::chrono::duration<double>::zero();
        auto begin=std::chrono::steady_clock::now();
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Cannot open DIMACS input");
        auto graph=read_dimacs(input);
        if (graph.n!=nodes || graph.m!=arcs) throw std::runtime_error("DIMACS graph dimensions differ");
        graph.block_size=static_cast<int32_t>(std::floor(std::sqrt(graph.m)));
        graph.block_size=((graph.block_size+15)/16)*16;
        graph.block_size=std::min(std::max(graph.block_size,16),graph.m);
        auto ready=std::chrono::steady_clock::now();
        int64_t iterations=0;
        bool timeout=false;
        while (find_entering_arc(graph)) {
            ++iterations;
            find_join_node(graph);
            find_leaving_arc(graph);
            change_flows(graph);
            change_states(graph);
            if (graph.change) {
                update_tree(graph);
                update_pots(graph);
            }
            if (iterations%64==0 && std::chrono::duration<double>(std::chrono::steady_clock::now()-ready).count()>time_limit) {
                timeout=true; break;
            }
        }
        auto end=std::chrono::steady_clock::now();
        timing[0]=std::chrono::duration<double>(ready-begin).count();
        timing[1]=std::chrono::duration<double>(end-ready).count();
        timing[2]=std::chrono::duration<double>(end-begin).count();
        timing[3]=static_cast<double>(iterations);
        timing[4]=find_entering_arc_time.count();
        timing[5]=update_tree_time.count();
        timing[6]=update_pots_time.count();
        if (timeout) return 2;
        for (int a=graph.m;a<graph.M;++a) {
            if (graph.flows[a]!=0) throw std::runtime_error("PNS retained artificial flow: infeasible result");
        }
        for (int n=0;n<nodes;++n) winding[n]=-static_cast<int64_t>(graph.pots[n]);
        for (int a=0;a<arcs;++a) shifted_flow[a]=graph.flows[a];
        return 0;
    } catch (const std::exception& ex) {
        if (error_size) std::snprintf(error,error_size,"%s",ex.what());
        return 1;
    }
}
