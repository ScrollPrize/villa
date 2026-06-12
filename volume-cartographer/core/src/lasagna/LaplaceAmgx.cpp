#include "vc/lasagna/LaplaceAmgx.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(VC_ENABLE_AMGX)
#include <amgx_c.h>
#endif

namespace vc::lasagna {
namespace {

void validateNode(const MaxflowGraph& graph, int32_t node, const char* name)
{
    if (node < 0 || static_cast<uint64_t>(node) >= graph.stats.graphNodes) {
        throw std::runtime_error(std::string(name) + " is outside graph");
    }
}

[[nodiscard]] int32_t checkedInt32(uint64_t value, const char* what)
{
    if (value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(std::string(what) + " exceeds int32 range required by AMGX");
    }
    return static_cast<int32_t>(value);
}

#if defined(VC_ENABLE_AMGX)
[[nodiscard]] std::string defaultAmgxConfig()
{
    return R"({
        "config_version": 2,
        "determinism_flag": 1,
        "solver": {
            "preconditioner": {
                "solver": "AMG",
                "algorithm": "AGGREGATION",
                "selector": "SIZE_2",
                "interpolator": "D2",
                "smoother": {
                    "solver": "BLOCK_JACOBI",
                    "scope": "jacobi",
                    "relaxation_factor": 0.8,
                    "monitor_residual": 0,
                    "print_solve_stats": 0
                },
                "presweeps": 0,
                "postsweeps": 3,
                "cycle": "V",
                "coarse_solver": "NOSOLVER",
                "max_levels": 50,
                "max_iters": 1,
                "monitor_residual": 0,
                "print_grid_stats": 0,
                "print_solve_stats": 0,
                "store_res_history": 0,
                "scope": "amg"
            },
            "solver": "FGMRES",
            "tolerance": 1e-8,
            "max_iters": 500,
            "monitor_residual": 1,
            "convergence": "RELATIVE_INI",
            "norm": "L2",
            "scope": "main",
            "print_solve_stats": 0,
            "obtain_timings": 0
        }
    })";
}

void checkAmgx(AMGX_RC rc, const char* call)
{
    if (rc == AMGX_RC_OK) {
        return;
    }
    std::ostringstream msg;
    msg << call << " failed with AMGX_RC=" << static_cast<int>(rc);
    throw std::runtime_error(msg.str());
}

[[nodiscard]] const char* solveStatusName(AMGX_SOLVE_STATUS status)
{
    switch (status) {
    case AMGX_SOLVE_SUCCESS:
        return "success";
    case AMGX_SOLVE_FAILED:
        return "failed";
    case AMGX_SOLVE_DIVERGED:
        return "diverged";
    case AMGX_SOLVE_NOT_CONVERGED:
        return "not_converged";
    }
    return "unknown";
}

struct AmgxRuntime {
    AmgxRuntime()
    {
        checkAmgx(AMGX_initialize(), "AMGX_initialize");
    }

    ~AmgxRuntime()
    {
        AMGX_finalize();
    }
};

AmgxRuntime& amgxRuntime()
{
    static AmgxRuntime runtime;
    return runtime;
}

std::mutex& amgxSolveMutex()
{
    static std::mutex mutex;
    return mutex;
}
#endif

} // namespace

bool laplaceAmgxAvailable() noexcept
{
#if defined(VC_ENABLE_AMGX)
    return true;
#else
    return false;
#endif
}

LaplaceSourceRegion buildLaplaceSourceRegion(
    const MaxflowGraph& graph,
    int32_t sourceNode,
    int sourceDepth)
{
    validateNode(graph, sourceNode, "Laplace source node");
    if (sourceDepth < 0) {
        throw std::runtime_error("Laplace source depth must be non-negative");
    }

    LaplaceSourceRegion region;
    const size_t nodeCount = static_cast<size_t>(graph.stats.graphNodes);
    region.mask.assign(nodeCount, 0);
    region.nodes.reserve(1);

    std::queue<int32_t> frontier;
    std::vector<int32_t> distance(nodeCount, -1);
    frontier.push(sourceNode);
    distance[static_cast<size_t>(sourceNode)] = 0;
    region.mask[static_cast<size_t>(sourceNode)] = 1;
    region.nodes.push_back(sourceNode);

    while (!frontier.empty()) {
        const int32_t node = frontier.front();
        frontier.pop();
        const int32_t nextDistance = distance[static_cast<size_t>(node)] + 1;
        if (nextDistance > sourceDepth) {
            continue;
        }
        for (uint64_t edge = graph.nindex[static_cast<size_t>(node)];
             edge < graph.nindex[static_cast<size_t>(node) + 1];
             ++edge) {
            const int32_t dst = graph.nlist[static_cast<size_t>(edge)];
            const size_t dstIndex = static_cast<size_t>(dst);
            if (distance[dstIndex] >= 0) {
                continue;
            }
            distance[dstIndex] = nextDistance;
            region.mask[dstIndex] = 1;
            region.nodes.push_back(dst);
            frontier.push(dst);
        }
    }

    std::sort(region.nodes.begin(), region.nodes.end());
    return region;
}

LaplaceCsrSystem assembleScreenedLaplaceSystem(
    const MaxflowGraph& graph,
    int32_t sourceNode,
    double lambda,
    int sourceDepth)
{
    validateNode(graph, sourceNode, "Laplace source node");
    if (!std::isfinite(lambda) || lambda < 0.0) {
        throw std::runtime_error("Laplace lambda must be finite and non-negative");
    }
    (void)checkedInt32(graph.stats.graphNodes, "Laplace graph node count");

    LaplaceCsrSystem system;
    system.lambda = lambda;
    system.sourceRegion = buildLaplaceSourceRegion(graph, sourceNode, sourceDepth);
    const size_t nodeCount = static_cast<size_t>(graph.stats.graphNodes);
    system.nodeToUnknown.assign(nodeCount, -1);
    system.unknownToNode.reserve(nodeCount - system.sourceRegion.nodes.size());

    for (uint64_t node = 0; node < graph.stats.graphNodes; ++node) {
        const size_t nodeIndex = static_cast<size_t>(node);
        if (system.sourceRegion.mask[nodeIndex] != 0) {
            continue;
        }
        const int32_t unknown = checkedInt32(
            static_cast<uint64_t>(system.unknownToNode.size()),
            "Laplace unknown count");
        system.nodeToUnknown[nodeIndex] = unknown;
        system.unknownToNode.push_back(static_cast<int32_t>(node));
    }

    const size_t unknownCount = system.unknownToNode.size();
    system.rowOffsets.assign(unknownCount + 1, 0);
    system.rhs.assign(unknownCount, 0.0);

    uint64_t nnz = 0;
    for (int32_t node : system.unknownToNode) {
        ++nnz; // diagonal
        for (uint64_t edge = graph.nindex[static_cast<size_t>(node)];
             edge < graph.nindex[static_cast<size_t>(node) + 1];
             ++edge) {
            const int32_t dst = graph.nlist[static_cast<size_t>(edge)];
            if (system.sourceRegion.mask[static_cast<size_t>(dst)] == 0) {
                ++nnz;
            }
        }
    }
    (void)checkedInt32(nnz, "Laplace matrix nonzero count");
    system.columns.reserve(static_cast<size_t>(nnz));
    system.values.reserve(static_cast<size_t>(nnz));

    for (size_t row = 0; row < unknownCount; ++row) {
        const int32_t node = system.unknownToNode[row];
        system.rowOffsets[row] = checkedInt32(
            static_cast<uint64_t>(system.columns.size()),
            "Laplace row offset");

        double diagonal = lambda;
        std::vector<std::pair<int32_t, double>> offDiagonal;
        for (uint64_t edge = graph.nindex[static_cast<size_t>(node)];
             edge < graph.nindex[static_cast<size_t>(node) + 1];
             ++edge) {
            const int32_t dst = graph.nlist[static_cast<size_t>(edge)];
            const double weight = static_cast<double>(graph.capacity[static_cast<size_t>(edge)]);
            diagonal += weight;
            if (system.sourceRegion.mask[static_cast<size_t>(dst)] != 0) {
                system.rhs[row] += weight;
            } else {
                offDiagonal.push_back({
                    system.nodeToUnknown[static_cast<size_t>(dst)],
                    -weight,
                });
            }
        }
        std::sort(offDiagonal.begin(), offDiagonal.end());

        system.columns.push_back(static_cast<int32_t>(row));
        system.values.push_back(diagonal);
        for (const auto& [column, value] : offDiagonal) {
            system.columns.push_back(column);
            system.values.push_back(value);
        }
    }
    system.rowOffsets[unknownCount] = checkedInt32(
        static_cast<uint64_t>(system.columns.size()),
        "Laplace final row offset");
    return system;
}

double laplaceValueForNode(
    const LaplaceCsrSystem& system,
    const std::vector<double>& unknownValues,
    int32_t node)
{
    if (node < 0 || static_cast<size_t>(node) >= system.nodeToUnknown.size()) {
        throw std::runtime_error("Laplace value node is outside graph");
    }
    if (system.sourceRegion.mask[static_cast<size_t>(node)] != 0) {
        return 1.0;
    }
    const int32_t unknown = system.nodeToUnknown[static_cast<size_t>(node)];
    if (unknown < 0 || static_cast<size_t>(unknown) >= unknownValues.size()) {
        throw std::runtime_error("Laplace value node is not represented in solution");
    }
    return unknownValues[static_cast<size_t>(unknown)];
}

LaplaceAmgxResult solveScreenedLaplaceAmgx(
    const MaxflowGraph& graph,
    int32_t sourceNode,
    const LaplaceAmgxOptions& options)
{
    const auto system = assembleScreenedLaplaceSystem(
        graph,
        sourceNode,
        options.lambda,
        options.sourceDepth);

#if !defined(VC_ENABLE_AMGX)
    (void)system;
    throw std::runtime_error("AMGX support is disabled; configure with -DVC_ENABLE_AMGX=ON");
#else
    amgxRuntime();
    std::lock_guard<std::mutex> amgxLock(amgxSolveMutex());

    LaplaceAmgxResult result;
    result.valuesByNode.assign(static_cast<size_t>(graph.stats.graphNodes), 1.0);
    if (system.unknownToNode.empty()) {
        result.success = true;
        result.status = "success";
        return result;
    }

    AMGX_config_handle config = nullptr;
    AMGX_resources_handle resources = nullptr;
    AMGX_matrix_handle matrix = nullptr;
    AMGX_vector_handle rhs = nullptr;
    AMGX_vector_handle solution = nullptr;
    AMGX_solver_handle solver = nullptr;

    try {
        if (options.configPath.empty()) {
            const std::string configText = defaultAmgxConfig();
            checkAmgx(AMGX_config_create(&config, configText.c_str()), "AMGX_config_create");
        } else {
            checkAmgx(
                AMGX_config_create_from_file(&config, options.configPath.string().c_str()),
                "AMGX_config_create_from_file");
        }
        checkAmgx(AMGX_resources_create_simple(&resources, config), "AMGX_resources_create_simple");

        constexpr AMGX_Mode mode = AMGX_mode_dDDI;
        checkAmgx(AMGX_matrix_create(&matrix, resources, mode), "AMGX_matrix_create");
        checkAmgx(AMGX_vector_create(&rhs, resources, mode), "AMGX_vector_create rhs");
        checkAmgx(AMGX_vector_create(&solution, resources, mode), "AMGX_vector_create solution");
        checkAmgx(AMGX_solver_create(&solver, resources, mode, config), "AMGX_solver_create");

        const int n = checkedInt32(
            static_cast<uint64_t>(system.unknownToNode.size()),
            "Laplace unknown count");
        const int nnz = checkedInt32(
            static_cast<uint64_t>(system.values.size()),
            "Laplace nonzero count");
        std::vector<double> x(static_cast<size_t>(n), 0.0);

        checkAmgx(
            AMGX_matrix_upload_all(
                matrix,
                n,
                nnz,
                1,
                1,
                system.rowOffsets.data(),
                system.columns.data(),
                system.values.data(),
                nullptr),
            "AMGX_matrix_upload_all");
        checkAmgx(AMGX_vector_upload(rhs, n, 1, system.rhs.data()), "AMGX_vector_upload rhs");
        checkAmgx(AMGX_vector_upload(solution, n, 1, x.data()), "AMGX_vector_upload solution");
        checkAmgx(AMGX_solver_setup(solver, matrix), "AMGX_solver_setup");

        const auto solveStart = std::chrono::steady_clock::now();
        checkAmgx(AMGX_solver_solve(solver, rhs, solution), "AMGX_solver_solve");
        result.solveSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - solveStart).count();

        AMGX_SOLVE_STATUS status = AMGX_SOLVE_FAILED;
        checkAmgx(AMGX_solver_get_status(solver, &status), "AMGX_solver_get_status");
        result.status = solveStatusName(status);
        result.success = status == AMGX_SOLVE_SUCCESS;
        checkAmgx(AMGX_vector_download(solution, x.data()), "AMGX_vector_download solution");

        result.valuesByNode.assign(static_cast<size_t>(graph.stats.graphNodes), 1.0);
        for (size_t unknown = 0; unknown < system.unknownToNode.size(); ++unknown) {
            result.valuesByNode[static_cast<size_t>(system.unknownToNode[unknown])] = x[unknown];
        }

        AMGX_solver_destroy(solver);
        AMGX_vector_destroy(solution);
        AMGX_vector_destroy(rhs);
        AMGX_matrix_destroy(matrix);
        AMGX_resources_destroy(resources);
        AMGX_config_destroy(config);
        return result;
    } catch (...) {
        if (solver != nullptr) AMGX_solver_destroy(solver);
        if (solution != nullptr) AMGX_vector_destroy(solution);
        if (rhs != nullptr) AMGX_vector_destroy(rhs);
        if (matrix != nullptr) AMGX_matrix_destroy(matrix);
        if (resources != nullptr) AMGX_resources_destroy(resources);
        if (config != nullptr) AMGX_config_destroy(config);
        throw;
    }
#endif
}

} // namespace vc::lasagna
