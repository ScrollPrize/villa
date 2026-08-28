#include <spiral_graph/winding_graph.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    const std::size_t nodes = argc > 1 ? std::stoull(argv[1]) : 80'000;
    const std::size_t edges = argc > 2 ? std::stoull(argv[2]) : 547'115;
    const int repeats = argc > 3 ? std::stoi(argv[3]) : 5;
    if (nodes < 2 || edges < nodes - 1 || repeats < 1) {
        throw std::invalid_argument("usage: benchmark_winding_graph_core [nodes>=2] [edges>=nodes-1] [repeats>=1]");
    }
    std::vector<double> seconds;
    seconds.reserve(static_cast<std::size_t>(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        spiral::winding::WindingGraph graph;
        std::vector<spiral::winding::NodeId> node_ids;
        node_ids.reserve(nodes);
        for (std::size_t node = 0; node < nodes; ++node) {
            node_ids.push_back(graph.ensure_patch("patch-" + std::to_string(node)));
        }
        std::vector<spiral::winding::Constraint> constraints;
        constraints.reserve(edges);
        for (std::size_t node = 1; node < nodes; ++node) {
            constraints.push_back({
                node_ids[0],
                node_ids[node],
                static_cast<std::int64_t>(node),
                static_cast<std::int64_t>(node), {}, false,
            });
        }
        std::uint64_t state = 0x9e3779b97f4a7c15ULL;
        while (constraints.size() < edges) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            const std::size_t a = static_cast<std::size_t>(state % nodes);
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            const std::size_t b = static_cast<std::size_t>(state % nodes);
            constraints.push_back({
                node_ids[a],
                node_ids[b],
                static_cast<std::int64_t>(b) - static_cast<std::int64_t>(a),
                static_cast<std::int64_t>(b) - static_cast<std::int64_t>(a),
                {}, false,
            });
        }
        const auto result = graph.add_constraints(constraints);
        if (!result.committed || graph.stats().constraint_count != edges) {
            throw std::runtime_error("benchmark graph unexpectedly conflicted");
        }
        const auto stop = std::chrono::steady_clock::now();
        seconds.push_back(std::chrono::duration<double>(stop - start).count());
    }
    std::sort(seconds.begin(), seconds.end());
    const double total = std::accumulate(seconds.begin(), seconds.end(), 0.0);
    std::cout << "{\"nodes\":" << nodes
              << ",\"edges\":" << edges
              << ",\"repeats\":" << repeats
              << ",\"mean_seconds\":" << total / repeats
              << ",\"min_seconds\":" << seconds.front()
              << ",\"median_seconds\":" << seconds[seconds.size() / 2]
              << ",\"max_seconds\":" << seconds.back() << "}\n";
}
