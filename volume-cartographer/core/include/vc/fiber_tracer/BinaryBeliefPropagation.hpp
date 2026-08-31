#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace vc::fiber_tracer
{

enum class BinaryBeliefState : std::int8_t {
    Free = -1,
    Zero = 0,
    One = 1,
};

struct BinaryPairwiseFactor {
    std::size_t a = 0;
    std::size_t b = 0;
    double sameCost = 0.0;
    double differentCost = 0.0;
};

struct BinaryBeliefPropagationConfig {
    double temperature = 0.25;
    double messageDamping = 0.5;
    double messageResidualTolerance = 1.0e-8;
    std::size_t maximumMessageIterations = 500;
    std::size_t parallelWorkers = 1;
};

struct BinaryBeliefPropagationReport {
    std::vector<double> probabilityOne;
    std::vector<double> logOdds;
    std::size_t messageIterations = 0;
    double messageResidual = 0.0;
    bool messageConverged = false;
    std::size_t effectiveWorkers = 1;
    double setupMilliseconds = 0.0;
    double nodeTotalMilliseconds = 0.0;
    double messageUpdateMilliseconds = 0.0;
    double solveMilliseconds = 0.0;
    double elapsedMilliseconds = 0.0;
};

struct BinaryBeliefPropagationProgress {
    std::size_t messageIteration = 0;
    std::size_t maximumMessageIterations = 0;
    double messageResidual = 0.0;
    bool complete = false;
};

using BinaryBeliefPropagationProgressCallback =
    std::function<void(const BinaryBeliefPropagationProgress&)>;

[[nodiscard]] BinaryBeliefPropagationReport solveBinaryPairwiseSumProduct(
    std::size_t nodeCount,
    std::span<const BinaryPairwiseFactor> factors,
    std::span<const BinaryBeliefState> fixedStates,
    const BinaryBeliefPropagationConfig& config = {},
    const BinaryBeliefPropagationProgressCallback& progress = {});

}  // namespace vc::fiber_tracer
