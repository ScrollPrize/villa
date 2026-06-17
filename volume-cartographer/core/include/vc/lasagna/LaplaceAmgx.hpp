#pragma once

#include "vc/lasagna/MaxflowGraph.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vc::lasagna {

struct LaplaceSourceRegion {
    std::vector<uint8_t> mask;
    std::vector<int32_t> nodes;
};

struct LaplaceCsrSystem {
    std::vector<int32_t> rowOffsets;
    std::vector<int32_t> columns;
    std::vector<double> values;
    std::vector<double> rhs;
    std::vector<int32_t> nodeToUnknown;
    std::vector<int32_t> unknownToNode;
    LaplaceSourceRegion sourceRegion;
    double lambda = 0.0;
};

struct LaplaceAmgxOptions {
    double lambda = 1.0 / 64.0;
    int sourceDepth = 0;
    std::filesystem::path configPath;
};

struct LaplaceAmgxResult {
    bool success = false;
    std::string status;
    double solveSeconds = 0.0;
    std::vector<double> valuesByNode;
};

[[nodiscard]] bool laplaceAmgxAvailable() noexcept;

[[nodiscard]] LaplaceSourceRegion buildLaplaceSourceRegion(
    const MaxflowGraph& graph,
    int32_t sourceNode,
    int sourceDepth);

[[nodiscard]] LaplaceCsrSystem assembleScreenedLaplaceSystem(
    const MaxflowGraph& graph,
    int32_t sourceNode,
    double lambda,
    int sourceDepth = 0);

[[nodiscard]] LaplaceAmgxResult solveScreenedLaplaceAmgx(
    const MaxflowGraph& graph,
    int32_t sourceNode,
    const LaplaceAmgxOptions& options);

[[nodiscard]] double laplaceValueForNode(
    const LaplaceCsrSystem& system,
    const std::vector<double>& unknownValues,
    int32_t node);

} // namespace vc::lasagna
