#pragma once

#include "vc/fiber_tracer/FiberTraceConstraints.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vc::fiber_tracer
{

enum class FiberTracePieceLabel : unsigned char {
    HEven,
    HOdd,
    VEven,
    VOdd,
    Broken,
};

struct FiberTraceLabelingConfig {
    double brokenCostPerConstraint = 0.5;
    double relativeMipGap = 1.0e-4;
    std::size_t parallelThreads = 0;
};

struct FiberTraceLabelingReport {
    std::vector<FiberTracePieceLabel> labels;
    std::array<std::size_t, 5> labelCounts{};
    std::size_t variables = 0;
    std::size_t integerVariables = 0;
    std::size_t rows = 0;
    std::int64_t mipNodes = 0;
    double objective = 0.0;
    double orientationCost = 0.0;
    double windingCost = 0.0;
    double brokenCost = 0.0;
    double mipGap = 0.0;
    double solveSeconds = 0.0;
    std::string modelStatus;
};

struct FiberTraceLabelObjPaths {
    std::filesystem::path hEven;
    std::filesystem::path hOdd;
    std::filesystem::path vEven;
    std::filesystem::path vOdd;
    std::filesystem::path broken;
};

struct FiberTraceLabelObjReport {
    FiberTraceLabelObjPaths paths;
    std::array<std::size_t, 5> pieceCounts{};
};

[[nodiscard]] FiberTraceLabelingReport solveFiberTraceLabels(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceLabelingConfig& config = {});

[[nodiscard]] FiberTraceLabelObjPaths fiberTraceLabelObjPaths(
    const std::filesystem::path& outputBase);

[[nodiscard]] FiberTraceLabelObjReport writeFiberTraceLabelObjs(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceLabelingReport& labeling,
    const std::filesystem::path& outputBase);

}  // namespace vc::fiber_tracer
