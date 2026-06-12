#include "vc/lasagna/EclMaxflow.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "maxflow.cu"

namespace vc::lasagna {
namespace {

void checkCuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

[[nodiscard]] int checkedInt(uint64_t value, const char* name)
{
    if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " exceeds ECL-MaxFlow int32 range");
    }
    return static_cast<int>(value);
}

[[nodiscard]] double median(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        return 0.5 * (values[mid - 1] + values[mid]);
    }
    return values[mid];
}

} // namespace

bool eclMaxflowAvailable() noexcept
{
    int deviceCount = 0;
    return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
}

EclMaxflowResult runEclMaxflow(
    const MaxflowGraph& graph,
    int32_t sourceNode,
    int32_t sinkNode,
    const EclMaxflowOptions& options)
{
    const int runs = options.runs;
    if (runs <= 0) {
        throw std::runtime_error("ECL-MaxFlow runs must be positive");
    }
    const int nodes = checkedInt(graph.stats.graphNodes, "node count");
    const int edges = checkedInt(graph.stats.directedEdges, "edge count");
    if (nodes <= 0) {
        throw std::runtime_error("ECL-MaxFlow graph has no nodes");
    }
    if (sourceNode < 0 || sourceNode >= nodes || sinkNode < 0 || sinkNode >= nodes) {
        throw std::runtime_error("ECL-MaxFlow source/sink node is outside graph");
    }
    if (sourceNode == sinkNode) {
        throw std::runtime_error("ECL-MaxFlow source and sink nodes must differ");
    }
    if (graph.nindex.size() != static_cast<size_t>(nodes) + 1 ||
        graph.nlist.size() != static_cast<size_t>(edges) ||
        graph.capacity.size() != static_cast<size_t>(edges)) {
        throw std::runtime_error("ECL-MaxFlow graph CSR arrays are inconsistent");
    }

    std::vector<int> nindex(graph.nindex.size());
    for (size_t i = 0; i < graph.nindex.size(); ++i) {
        nindex[i] = checkedInt(graph.nindex[i], "nindex entry");
    }
    std::vector<int> nlist(graph.nlist.begin(), graph.nlist.end());
    std::vector<int> capacity(graph.capacity.begin(), graph.capacity.end());
    std::vector<int> flow(static_cast<size_t>(edges), 0);

    ECLgraph g{};
    g.nodes = nodes;
    g.edges = edges;
    g.nindex = nindex.data();
    g.nlist = nlist.data();
    g.eweight = capacity.data();

    ECLgraph d_g = g;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_g.nindex), (static_cast<size_t>(nodes) + 1) * sizeof(int)),
              "cudaMalloc d_g.nindex");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_g.nlist), static_cast<size_t>(edges) * sizeof(int)),
              "cudaMalloc d_g.nlist");
    checkCuda(cudaMemcpy(d_g.nindex, nindex.data(), (static_cast<size_t>(nodes) + 1) * sizeof(int), cudaMemcpyHostToDevice),
              "cudaMemcpy d_g.nindex");
    checkCuda(cudaMemcpy(d_g.nlist, nlist.data(), static_cast<size_t>(edges) * sizeof(int), cudaMemcpyHostToDevice),
              "cudaMemcpy d_g.nlist");

    std::vector<int> rnindex(static_cast<size_t>(nodes) + 1, 0);
    for (int v = 0; v < nodes; ++v) {
        for (int e = nindex[static_cast<size_t>(v)]; e < nindex[static_cast<size_t>(v) + 1]; ++e) {
            const int nbor = nlist[static_cast<size_t>(e)];
            ++rnindex[static_cast<size_t>(nbor) + 1];
        }
    }
    for (size_t i = 1; i < rnindex.size(); ++i) {
        rnindex[i] += rnindex[i - 1];
    }
    std::vector<int> rnlist(static_cast<size_t>(edges), 0);
    std::vector<int> retoe(static_cast<size_t>(edges), 0);
    std::vector<int> cursor = rnindex;
    for (int v = 0; v < nodes; ++v) {
        for (int e = nindex[static_cast<size_t>(v)]; e < nindex[static_cast<size_t>(v) + 1]; ++e) {
            const int nbor = nlist[static_cast<size_t>(e)];
            const int reidx = cursor[static_cast<size_t>(nbor)]++;
            rnlist[static_cast<size_t>(reidx)] = v;
            retoe[static_cast<size_t>(reidx)] = e;
        }
    }

    int* d_rnindex = nullptr;
    int* d_rnlist = nullptr;
    int* d_retoe = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_rnindex), (static_cast<size_t>(nodes) + 1) * sizeof(int)),
              "cudaMalloc d_rnindex");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_rnlist), static_cast<size_t>(edges) * sizeof(int)),
              "cudaMalloc d_rnlist");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_retoe), static_cast<size_t>(edges) * sizeof(int)),
              "cudaMalloc d_retoe");
    checkCuda(cudaMemcpy(d_rnindex, rnindex.data(), (static_cast<size_t>(nodes) + 1) * sizeof(int), cudaMemcpyHostToDevice),
              "cudaMemcpy d_rnindex");
    checkCuda(cudaMemcpy(d_rnlist, rnlist.data(), static_cast<size_t>(edges) * sizeof(int), cudaMemcpyHostToDevice),
              "cudaMemcpy d_rnlist");
    checkCuda(cudaMemcpy(d_retoe, retoe.data(), static_cast<size_t>(edges) * sizeof(int), cudaMemcpyHostToDevice),
              "cudaMemcpy d_retoe");

    std::vector<double> runtimes;
    runtimes.reserve(static_cast<size_t>(runs));
    int64_t maxFlow = 0;
    for (int run = 0; run < runs; ++run) {
        const double runtime = GPUmaxflow(
            g,
            d_g,
            sourceNode,
            sinkNode,
            flow.data(),
            capacity.data(),
            d_rnindex,
            d_rnlist,
            d_retoe);
        runtimes.push_back(runtime);
        maxFlow = static_cast<int64_t>(g_last_maxflow_result);
    }

    cudaFree(d_g.nindex);
    cudaFree(d_g.nlist);
    cudaFree(d_rnindex);
    cudaFree(d_rnlist);
    cudaFree(d_retoe);

    const double med = median(runtimes);
    EclMaxflowResult result{
        maxFlow,
        med,
        med > 0.0 ? 0.000000001 * static_cast<double>(edges) / med : 0.0,
        runs,
    };
    if (options.computeMinCut) {
        result.minCut = computeMinCutFromFinalFlow(
            graph,
            flow,
            sourceNode,
            sinkNode,
            options.storeSourceReachable,
            options.storeCutEdges);
    }
    return result;
}

EclMaxflowResult runEclMaxflow(
    const MaxflowGraph& graph,
    int32_t sourceNode,
    int32_t sinkNode,
    int runs)
{
    EclMaxflowOptions options;
    options.runs = runs;
    return runEclMaxflow(graph, sourceNode, sinkNode, options);
}

} // namespace vc::lasagna
