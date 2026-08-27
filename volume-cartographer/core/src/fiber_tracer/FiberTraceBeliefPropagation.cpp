#include "vc/fiber_tracer/FiberTraceBeliefPropagation.hpp"

#include "vc/fiber_tracer/FiberTraceSeed.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>

namespace vc::fiber_tracer
{
namespace
{

struct Factor {
    std::size_t a = 0;
    std::size_t b = 0;
    double sameCost = 0.0;
    double differentCost = 0.0;
    std::size_t measurements = 0;
};

struct Graph {
    std::vector<Factor> factors;
    std::vector<std::vector<std::size_t>> adjacency;
    std::size_t measurements = 0;
    std::size_t neutralFactors = 0;
    std::size_t neutralMeasurements = 0;
    std::size_t components = 0;
    std::size_t isolated = 0;
};

struct FieldSolution {
    std::vector<double> horizontalness;
    std::vector<double> advantage;
    std::size_t iterations = 0;
    double residual = 0.0;
    double fraction = 0.0;
    bool converged = false;
};

struct PreparedProblem {
    Graph graph;
    std::vector<double> normalizedArcWeights;
    std::size_t seed = 0;
};

void validateConfig(const FiberTraceBeliefPropagationConfig& config)
{
    const auto finite = [](double value) { return std::isfinite(value); };
    if (!finite(config.targetHorizontalFraction) ||
        config.targetHorizontalFraction < 0.0 ||
        config.targetHorizontalFraction > 1.0) {
        throw std::invalid_argument(
            "BP target horizontal fraction must be in [0, 1]");
    }
    if (!finite(config.softBalanceStrength) ||
        config.softBalanceStrength < 0.0) {
        throw std::invalid_argument(
            "BP soft balance strength must be finite and nonnegative");
    }
    if (!finite(config.horizontalnessTemperature) ||
        !(config.horizontalnessTemperature > 0.0)) {
        throw std::invalid_argument(
            "BP horizontalness temperature must be finite and positive");
    }
    if (!finite(config.mixedUnaryCost) || config.mixedUnaryCost < 0.0) {
        throw std::invalid_argument(
            "BP Mixed unary cost must be finite and nonnegative");
    }
    if (!finite(config.messageDamping) || !(config.messageDamping > 0.0) ||
        config.messageDamping > 1.0) {
        throw std::invalid_argument(
            "BP message damping must be in (0, 1]");
    }
    if (!finite(config.messageResidualTolerance) ||
        config.messageResidualTolerance < 0.0 ||
        !finite(config.balanceTolerance) || config.balanceTolerance < 0.0) {
        throw std::invalid_argument(
            "BP tolerances must be finite and nonnegative");
    }
    if (config.maximumMessageIterations == 0 ||
        config.maximumBalanceIterations == 0) {
        throw std::invalid_argument(
            "BP iteration limits must be positive");
    }
}

Graph buildGraph(
    std::size_t traceCount,
    const FiberTraceConstraintReport& constraints)
{
    if (constraints.pieces.size() != traceCount) {
        throw std::invalid_argument(
            "BP requires exactly one constraint piece per represented fiber");
    }
    std::vector<std::size_t> traceByPiece(traceCount, traceCount);
    std::vector<unsigned char> represented(traceCount, 0);
    for (std::size_t piece = 0; piece < traceCount; ++piece) {
        const std::size_t trace = constraints.pieces[piece].traceIndex;
        if (trace >= traceCount || represented[trace] != 0) {
            throw std::invalid_argument(
                "BP requires unique contiguous represented fiber indices");
        }
        represented[trace] = 1;
        traceByPiece[piece] = trace;
    }

    std::map<std::pair<std::size_t, std::size_t>, Factor> merged;
    for (const auto& constraint : constraints.constraints) {
        if (constraint.pieceA >= traceCount ||
            constraint.pieceB >= traceCount) {
            throw std::invalid_argument(
                "BP constraint references an invalid piece");
        }
        if (constraint.hardContinuity) {
            throw std::invalid_argument(
                "BP no-split input cannot contain hard continuity constraints");
        }
        if (!std::isfinite(constraint.parallelScore) ||
            !std::isfinite(constraint.perpendicularScore) ||
            constraint.parallelScore < 0.0 ||
            constraint.parallelScore > 1.0 ||
            constraint.perpendicularScore < 0.0 ||
            constraint.perpendicularScore > 1.0 ||
            std::abs(
                constraint.parallelScore + constraint.perpendicularScore -
                1.0) > 1.0e-9) {
            throw std::invalid_argument(
                "BP requires complementary measured orientation scores");
        }
        const std::size_t traceA = traceByPiece[constraint.pieceA];
        const std::size_t traceB = traceByPiece[constraint.pieceB];
        if (traceA == traceB) {
            throw std::invalid_argument(
                "BP no-split input contains a same-fiber constraint");
        }
        const auto key = std::minmax(traceA, traceB);
        auto [found, inserted] = merged.try_emplace(
            std::pair<std::size_t, std::size_t>{key.first, key.second});
        auto& factor = found->second;
        if (inserted) {
            factor.a = key.first;
            factor.b = key.second;
        }
        factor.sameCost += 1.0 - constraint.parallelScore;
        factor.differentCost += constraint.parallelScore;
        ++factor.measurements;
    }

    Graph graph;
    graph.adjacency.resize(traceCount);
    graph.factors.reserve(merged.size());
    for (const auto& [key, factor] : merged) {
        (void)key;
        if (factor.sameCost == factor.differentCost) {
            ++graph.neutralFactors;
            graph.neutralMeasurements += factor.measurements;
            continue;
        }
        auto normalized = factor;
        const double commonCost = std::min(
            normalized.sameCost, normalized.differentCost);
        normalized.sameCost -= commonCost;
        normalized.differentCost -= commonCost;
        const std::size_t index = graph.factors.size();
        graph.factors.push_back(normalized);
        graph.adjacency[normalized.a].push_back(index);
        graph.adjacency[normalized.b].push_back(index);
        graph.measurements += normalized.measurements;
    }

    std::vector<unsigned char> visited(traceCount, 0);
    for (std::size_t start = 0; start < traceCount; ++start) {
        if (visited[start] != 0)
            continue;
        ++graph.components;
        if (graph.adjacency[start].empty())
            ++graph.isolated;
        std::queue<std::size_t> pending;
        pending.push(start);
        visited[start] = 1;
        while (!pending.empty()) {
            const std::size_t node = pending.front();
            pending.pop();
            for (const std::size_t factorIndex : graph.adjacency[node]) {
                const auto& factor = graph.factors[factorIndex];
                const std::size_t neighbor =
                    factor.a == node ? factor.b : factor.a;
                if (visited[neighbor] == 0) {
                    visited[neighbor] = 1;
                    pending.push(neighbor);
                }
            }
        }
    }
    return graph;
}

PreparedProblem prepareProblem(
    const std::vector<FiberletCropTraceLine>& traces,
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefPropagationConfig& config)
{
    if (traces.empty())
        throw std::invalid_argument("BP requires at least one represented fiber");
    const auto geometry = measureFiberTraceSeedGeometry(
        traces, config.cropMinimumBaseXYZ, config.cropMaximumBaseXYZ);
    const auto seed = selectCentralStraightFiberTrace(geometry);
    if (!seed) {
        throw std::invalid_argument(
            "BP has no primary seed longer than half the nominal crop size");
    }
    for (const auto& trace : geometry.traces) {
        if (!trace.valid) {
            throw std::invalid_argument(
                "BP requires finite nondegenerate represented fibers");
        }
    }

    PreparedProblem problem;
    problem.seed = *seed;
    problem.graph = buildGraph(traces.size(), constraints);
    problem.normalizedArcWeights.reserve(traces.size());
    double totalArc = 0.0;
    for (const auto& trace : geometry.traces)
        totalArc += trace.arcLengthBaseVoxels;
    const double meanArc = totalArc / static_cast<double>(traces.size());
    for (const auto& trace : geometry.traces) {
        problem.normalizedArcWeights.push_back(
            trace.arcLengthBaseVoxels / meanArc);
    }
    return problem;
}

FiberTraceBeliefPropagationReport initializeReport(
    const PreparedProblem& problem,
    const FiberTraceBeliefPropagationConfig& config,
    FiberTraceBeliefInference inference)
{
    FiberTraceBeliefPropagationReport report;
    report.normalizedArcWeights = problem.normalizedArcWeights;
    report.seedTraceIndex = problem.seed;
    report.factors = problem.graph.factors.size();
    report.mergedMeasurements = problem.graph.measurements;
    report.neutralFactors = problem.graph.neutralFactors;
    report.neutralMeasurements = problem.graph.neutralMeasurements;
    report.connectedComponents = problem.graph.components;
    report.isolatedTraces = problem.graph.isolated;
    report.targetHorizontalFraction = config.targetHorizontalFraction;
    report.inference = inference;
    report.inferenceTemperature = config.horizontalnessTemperature;
    return report;
}

double updateMessage(
    double cavityGap,
    double sameCost,
    double differentCost)
{
    const double targetH = std::min(
        differentCost, sameCost + cavityGap);
    const double targetV = std::min(
        sameCost, differentCost + cavityGap);
    return targetH - targetV;
}

double horizontalness(double advantage, double temperature)
{
    const double scaled = advantage / temperature;
    if (scaled >= 0.0) {
        const double exponential = std::exp(-scaled);
        return 1.0 / (1.0 + exponential);
    }
    const double exponential = std::exp(scaled);
    return exponential / (1.0 + exponential);
}

double logAddExp(double a, double b)
{
    const double maximum = std::max(a, b);
    return maximum + std::log1p(std::exp(std::min(a, b) - maximum));
}

using TernaryLogMessage = std::array<double, 3>;

double logSumExp(const TernaryLogMessage& values)
{
    const double maximum = *std::max_element(values.begin(), values.end());
    if (!std::isfinite(maximum))
        return maximum;
    double sum = 0.0;
    for (const double value : values)
        sum += std::exp(value - maximum);
    return maximum + std::log(sum);
}

void normalizeLogMessage(TernaryLogMessage& values)
{
    const double normalization = logSumExp(values);
    for (double& value : values)
        value -= normalization;
}

double updateSumProductMessage(
    double cavityLogOdds,
    double logSamePotential,
    double logDifferentPotential)
{
    const double targetH = logAddExp(
        logDifferentPotential,
        cavityLogOdds + logSamePotential);
    const double targetV = logAddExp(
        logSamePotential,
        cavityLogOdds + logDifferentPotential);
    return targetH - targetV;
}

FieldSolution solveField(
    const Graph& graph,
    std::span<const double> weights,
    std::size_t seed,
    double field,
    const FiberTraceBeliefPropagationConfig& config)
{
    const std::size_t nodeCount = graph.adjacency.size();
    std::vector<double> aToB(graph.factors.size(), 0.0);
    std::vector<double> bToA(graph.factors.size(), 0.0);
    std::vector<double> nextAToB(graph.factors.size(), 0.0);
    std::vector<double> nextBToA(graph.factors.size(), 0.0);
    std::vector<double> totalGap(nodeCount, 0.0);

    FieldSolution result;
    for (std::size_t iteration = 0;
         iteration < config.maximumMessageIterations;
         ++iteration) {
        for (std::size_t node = 0; node < nodeCount; ++node)
            totalGap[node] = node == seed ? 0.0 : -field * weights[node];
        for (std::size_t index = 0; index < graph.factors.size(); ++index) {
            totalGap[graph.factors[index].a] += bToA[index];
            totalGap[graph.factors[index].b] += aToB[index];
        }

        double residual = 0.0;
        for (std::size_t index = 0; index < graph.factors.size(); ++index) {
            const auto& factor = graph.factors[index];
            const double rawAToB = factor.a == seed
                ? factor.sameCost - factor.differentCost
                : updateMessage(
                      totalGap[factor.a] - bToA[index],
                      factor.sameCost,
                      factor.differentCost);
            const double rawBToA = factor.b == seed
                ? factor.sameCost - factor.differentCost
                : updateMessage(
                      totalGap[factor.b] - aToB[index],
                      factor.sameCost,
                      factor.differentCost);
            nextAToB[index] = aToB[index] + config.messageDamping *
                (rawAToB - aToB[index]);
            nextBToA[index] = bToA[index] + config.messageDamping *
                (rawBToA - bToA[index]);
            residual = std::max({
                residual,
                std::abs(nextAToB[index] - aToB[index]),
                std::abs(nextBToA[index] - bToA[index]),
            });
        }
        aToB.swap(nextAToB);
        bToA.swap(nextBToA);
        result.iterations = iteration + 1;
        result.residual = residual;
        if (residual <= config.messageResidualTolerance) {
            result.converged = true;
            break;
        }
    }

    for (std::size_t node = 0; node < nodeCount; ++node)
        totalGap[node] = node == seed ? 0.0 : -field * weights[node];
    for (std::size_t index = 0; index < graph.factors.size(); ++index) {
        totalGap[graph.factors[index].a] += bToA[index];
        totalGap[graph.factors[index].b] += aToB[index];
    }
    result.horizontalness.resize(nodeCount);
    result.advantage.resize(nodeCount);
    double weightedHorizontal = 0.0;
    double totalWeight = 0.0;
    for (std::size_t node = 0; node < nodeCount; ++node) {
        result.advantage[node] = node == seed
            ? std::numeric_limits<double>::infinity()
            : -totalGap[node];
        result.horizontalness[node] = node == seed
            ? 1.0
            : horizontalness(
                  result.advantage[node],
                  config.horizontalnessTemperature);
        weightedHorizontal += weights[node] * result.horizontalness[node];
        totalWeight += weights[node];
    }
    result.fraction = weightedHorizontal / totalWeight;
    return result;
}

double fieldBound(
    const Graph& graph,
    std::span<const double> weights,
    double temperature)
{
    std::vector<double> incident(graph.adjacency.size(), 0.0);
    for (const auto& factor : graph.factors) {
        const double range = std::abs(
            factor.sameCost - factor.differentCost);
        incident[factor.a] += range;
        incident[factor.b] += range;
    }
    double bound = 1.0;
    for (std::size_t node = 0; node < incident.size(); ++node) {
        bound = std::max(
            bound, incident[node] / weights[node] + 32.0 * temperature);
    }
    return bound;
}

void assignFieldSolution(
    FiberTraceBeliefPropagationReport& report,
    FieldSolution solution,
    double field)
{
    report.horizontalness = std::move(solution.horizontalness);
    report.minMarginalAdvantage = std::move(solution.advantage);
    report.messageIterations += solution.iterations;
    report.messageResidual = solution.residual;
    report.messageConverged = solution.converged;
    report.achievedHorizontalFraction = solution.fraction;
    report.balanceField = field;
}

}  // namespace

const char* fiberTraceBalanceModeName(FiberTraceBalanceMode mode) noexcept
{
    switch (mode) {
    case FiberTraceBalanceMode::None:
        return "none";
    case FiberTraceBalanceMode::Soft:
        return "soft";
    case FiberTraceBalanceMode::Tight:
        return "tight";
    }
    return "invalid";
}

const char* fiberTraceBeliefInferenceName(
    FiberTraceBeliefInference inference) noexcept
{
    switch (inference) {
    case FiberTraceBeliefInference::MinSum:
        return "min_sum";
    case FiberTraceBeliefInference::SumProduct:
        return "sum_product";
    case FiberTraceBeliefInference::SumProductMixed:
        return "sum_product_mixed";
    }
    return "invalid";
}

FiberTraceBeliefPropagationReport solveFiberTraceBeliefPropagation(
    const std::vector<FiberletCropTraceLine>& traces,
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefPropagationConfig& config)
{
    const auto started = std::chrono::steady_clock::now();
    validateConfig(config);
    const auto problem = prepareProblem(traces, constraints, config);
    const auto& graph = problem.graph;
    const auto& weights = problem.normalizedArcWeights;
    const std::size_t seed = problem.seed;
    auto report = initializeReport(
        problem, config, FiberTraceBeliefInference::MinSum);

    if (config.balanceMode == FiberTraceBalanceMode::None) {
        auto solution = solveField(graph, weights, seed, 0.0, config);
        assignFieldSolution(report, std::move(solution), 0.0);
        report.balanceConverged = true;
        report.status = report.messageConverged
            ? "converged"
            : "message_limit";
    } else if (config.balanceMode == FiberTraceBalanceMode::Soft) {
        double field = 0.0;
        double bestField = 0.0;
        FieldSolution best;
        double bestFieldResidual = std::numeric_limits<double>::infinity();
        bool balanceConverged = false;
        for (std::size_t outer = 0;
             outer < config.maximumBalanceIterations;
             ++outer) {
            auto current = solveField(graph, weights, seed, field, config);
            report.messageIterations += current.iterations;
            report.balanceIterations = outer + 1;
            const double desired = config.softBalanceStrength *
                (config.targetHorizontalFraction - current.fraction);
            const double change = desired - field;
            if (std::abs(change) < bestFieldResidual) {
                bestFieldResidual = std::abs(change);
                best = current;
                bestField = field;
            }
            if (std::abs(change) <= config.balanceTolerance) {
                balanceConverged = true;
                break;
            }
            field += 0.5 * change;
        }
        const std::size_t accumulatedIterations = report.messageIterations;
        assignFieldSolution(report, std::move(best), bestField);
        report.messageIterations = accumulatedIterations;
        report.balanceConverged = balanceConverged;
        report.status = !report.messageConverged
            ? "message_limit"
            : balanceConverged ? "converged" : "balance_limit";
    } else {
        const double bound = fieldBound(
            graph, weights, config.horizontalnessTemperature);
        double lowField = -bound;
        double highField = bound;
        auto low = solveField(graph, weights, seed, lowField, config);
        auto high = solveField(graph, weights, seed, highField, config);
        report.messageIterations = low.iterations + high.iterations;
        report.balanceIterations = 2;
        const auto error = [&](const FieldSolution& solution) {
            return std::abs(
                solution.fraction - config.targetHorizontalFraction);
        };
        FieldSolution best = error(low) <= error(high) ? low : high;
        double bestField = error(low) <= error(high) ? lowField : highField;
        const bool infeasible =
            config.targetHorizontalFraction <
                low.fraction - config.balanceTolerance ||
            config.targetHorizontalFraction >
                high.fraction + config.balanceTolerance;
        bool balanceConverged = !infeasible &&
            error(best) <= config.balanceTolerance;
        if (!infeasible && !balanceConverged) {
            for (std::size_t outer = 2;
                 outer < config.maximumBalanceIterations;
                 ++outer) {
                const double middleField = 0.5 * (lowField + highField);
                auto middle = solveField(
                    graph, weights, seed, middleField, config);
                report.messageIterations += middle.iterations;
                report.balanceIterations = outer + 1;
                if (!middle.converged)
                    break;
                if (error(middle) < error(best)) {
                    best = middle;
                    bestField = middleField;
                }
                if (error(middle) <= config.balanceTolerance) {
                    best = std::move(middle);
                    bestField = middleField;
                    balanceConverged = true;
                    break;
                }
                if (middle.fraction < config.targetHorizontalFraction) {
                    lowField = middleField;
                    low = std::move(middle);
                } else {
                    highField = middleField;
                    high = std::move(middle);
                }
            }
        }
        bool mixedBracket = false;
        if (!infeasible && !balanceConverged &&
            low.fraction < config.targetHorizontalFraction &&
            high.fraction > config.targetHorizontalFraction) {
            const double highWeight =
                (config.targetHorizontalFraction - low.fraction) /
                (high.fraction - low.fraction);
            FieldSolution mixture;
            mixture.horizontalness.resize(traces.size());
            mixture.advantage.resize(traces.size());
            for (std::size_t trace = 0; trace < traces.size(); ++trace) {
                const double value = std::lerp(
                    low.horizontalness[trace],
                    high.horizontalness[trace],
                    highWeight);
                mixture.horizontalness[trace] = value;
                if (trace == seed) {
                    mixture.advantage[trace] =
                        std::numeric_limits<double>::infinity();
                } else if (value <= 0.0) {
                    mixture.advantage[trace] =
                        -std::numeric_limits<double>::infinity();
                } else if (value >= 1.0) {
                    mixture.advantage[trace] =
                        std::numeric_limits<double>::infinity();
                } else {
                    mixture.advantage[trace] =
                        config.horizontalnessTemperature *
                        std::log(value / (1.0 - value));
                }
            }
            mixture.fraction = config.targetHorizontalFraction;
            mixture.residual = std::max(low.residual, high.residual);
            mixture.converged = low.converged && high.converged;
            best = std::move(mixture);
            bestField = std::lerp(lowField, highField, highWeight);
            balanceConverged = true;
            mixedBracket = true;
        }
        const std::size_t accumulatedIterations = report.messageIterations;
        assignFieldSolution(report, std::move(best), bestField);
        report.messageIterations = accumulatedIterations;
        report.balanceConverged = balanceConverged;
        report.status = infeasible
            ? "infeasible"
            : !report.messageConverged
                ? "message_limit"
                : mixedBracket
                    ? "converged_mixture"
                    : balanceConverged ? "converged" : "balance_limit";
    }

    report.solveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return report;
}

FiberTraceBeliefPropagationReport solveFiberTraceSumProduct(
    const std::vector<FiberletCropTraceLine>& traces,
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefPropagationConfig& config)
{
    const auto started = std::chrono::steady_clock::now();
    validateConfig(config);
    if (config.balanceMode != FiberTraceBalanceMode::None) {
        throw std::invalid_argument(
            "Sum-product BP does not support population balance modes");
    }
    const auto problem = prepareProblem(traces, constraints, config);
    const auto& graph = problem.graph;
    const std::size_t nodeCount = graph.adjacency.size();
    const double temperature = config.horizontalnessTemperature;

    std::vector<double> logSame(graph.factors.size());
    std::vector<double> logDifferent(graph.factors.size());
    for (std::size_t index = 0; index < graph.factors.size(); ++index) {
        const double minimumCost = std::min(
            graph.factors[index].sameCost,
            graph.factors[index].differentCost);
        logSame[index] =
            -(graph.factors[index].sameCost - minimumCost) / temperature;
        logDifferent[index] =
            -(graph.factors[index].differentCost - minimumCost) / temperature;
        if (!std::isfinite(logSame[index]) ||
            !std::isfinite(logDifferent[index])) {
            throw std::invalid_argument(
                "Sum-product BP temperature is too small for factor costs");
        }
    }

    std::vector<double> aToB(graph.factors.size(), 0.0);
    std::vector<double> bToA(graph.factors.size(), 0.0);
    std::vector<double> nextAToB(graph.factors.size(), 0.0);
    std::vector<double> nextBToA(graph.factors.size(), 0.0);
    std::vector<double> totalLogOdds(nodeCount, 0.0);

    auto report = initializeReport(
        problem, config, FiberTraceBeliefInference::SumProduct);
    for (std::size_t iteration = 0;
         iteration < config.maximumMessageIterations;
         ++iteration) {
        std::fill(totalLogOdds.begin(), totalLogOdds.end(), 0.0);
        for (std::size_t index = 0; index < graph.factors.size(); ++index) {
            totalLogOdds[graph.factors[index].a] += bToA[index];
            totalLogOdds[graph.factors[index].b] += aToB[index];
        }

        double residual = 0.0;
        for (std::size_t index = 0; index < graph.factors.size(); ++index) {
            const auto& factor = graph.factors[index];
            const double rawAToB = factor.a == problem.seed
                ? logSame[index] - logDifferent[index]
                : updateSumProductMessage(
                      totalLogOdds[factor.a] - bToA[index],
                      logSame[index],
                      logDifferent[index]);
            const double rawBToA = factor.b == problem.seed
                ? logSame[index] - logDifferent[index]
                : updateSumProductMessage(
                      totalLogOdds[factor.b] - aToB[index],
                      logSame[index],
                      logDifferent[index]);
            nextAToB[index] = aToB[index] + config.messageDamping *
                (rawAToB - aToB[index]);
            nextBToA[index] = bToA[index] + config.messageDamping *
                (rawBToA - bToA[index]);
            residual = std::max({
                residual,
                std::abs(nextAToB[index] - aToB[index]),
                std::abs(nextBToA[index] - bToA[index]),
            });
        }
        aToB.swap(nextAToB);
        bToA.swap(nextBToA);
        report.messageIterations = iteration + 1;
        report.messageResidual = residual;
        if (residual <= config.messageResidualTolerance) {
            report.messageConverged = true;
            break;
        }
    }

    std::fill(totalLogOdds.begin(), totalLogOdds.end(), 0.0);
    for (std::size_t index = 0; index < graph.factors.size(); ++index) {
        totalLogOdds[graph.factors[index].a] += bToA[index];
        totalLogOdds[graph.factors[index].b] += aToB[index];
    }
    report.horizontalness.resize(nodeCount);
    report.logOdds.resize(nodeCount);
    double weightedHorizontal = 0.0;
    double totalWeight = 0.0;
    for (std::size_t node = 0; node < nodeCount; ++node) {
        report.logOdds[node] = node == problem.seed
            ? std::numeric_limits<double>::infinity()
            : totalLogOdds[node];
        report.horizontalness[node] = node == problem.seed
            ? 1.0
            : horizontalness(totalLogOdds[node], 1.0);
        weightedHorizontal += problem.normalizedArcWeights[node] *
            report.horizontalness[node];
        totalWeight += problem.normalizedArcWeights[node];
    }
    report.achievedHorizontalFraction = weightedHorizontal / totalWeight;
    report.status = report.messageConverged
        ? "converged"
        : "message_limit";
    report.solveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return report;
}

FiberTraceBeliefPropagationReport solveFiberTraceMixedSumProduct(
    const std::vector<FiberletCropTraceLine>& traces,
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefPropagationConfig& config)
{
    const auto started = std::chrono::steady_clock::now();
    validateConfig(config);
    if (config.balanceMode != FiberTraceBalanceMode::None) {
        throw std::invalid_argument(
            "Mixed-state sum-product BP does not support population balance modes");
    }
    const auto problem = prepareProblem(traces, constraints, config);
    const auto& graph = problem.graph;
    const std::size_t nodeCount = graph.adjacency.size();
    const double temperature = config.horizontalnessTemperature;

    using Potential = std::array<TernaryLogMessage, 3>;
    std::vector<Potential> logPotential(graph.factors.size());
    for (std::size_t index = 0; index < graph.factors.size(); ++index) {
        const auto& factor = graph.factors[index];
        std::array<std::array<double, 3>, 3> energies{};
        double minimum = std::numeric_limits<double>::infinity();
        for (std::size_t source = 0; source < 3; ++source) {
            for (std::size_t target = 0; target < 3; ++target) {
                double energy = 0.0;
                if (source != 1 && target != 1) {
                    energy = source == target
                        ? factor.sameCost
                        : factor.differentCost;
                }
                energies[source][target] = energy;
                minimum = std::min(minimum, energy);
            }
        }
        for (std::size_t source = 0; source < 3; ++source) {
            for (std::size_t target = 0; target < 3; ++target) {
                logPotential[index][source][target] =
                    -(energies[source][target] - minimum) / temperature;
                if (!std::isfinite(logPotential[index][source][target])) {
                    throw std::invalid_argument(
                        "Mixed-state sum-product BP temperature is too small for factor costs");
                }
            }
        }
    }

    const TernaryLogMessage zeroMessage{
        -std::log(3.0), -std::log(3.0), -std::log(3.0)};
    const TernaryLogMessage logUnary{
        0.0, -config.mixedUnaryCost / temperature, 0.0};
    if (!std::isfinite(logUnary[1])) {
        throw std::invalid_argument(
            "Mixed-state sum-product BP temperature is too small for the Mixed unary cost");
    }
    std::vector<TernaryLogMessage> aToB(
        graph.factors.size(), zeroMessage);
    std::vector<TernaryLogMessage> bToA(
        graph.factors.size(), zeroMessage);
    std::vector<TernaryLogMessage> nextAToB(
        graph.factors.size(), zeroMessage);
    std::vector<TernaryLogMessage> nextBToA(
        graph.factors.size(), zeroMessage);
    std::vector<TernaryLogMessage> totals(
        nodeCount, TernaryLogMessage{0.0, 0.0, 0.0});

    const auto rawMessage = [&] (
                                const TernaryLogMessage& cavity,
                                const Potential& potential,
                                bool fixedHorizontal) {
        TernaryLogMessage raw{};
        for (std::size_t target = 0; target < 3; ++target) {
            if (fixedHorizontal) {
                raw[target] = potential[2][target];
            } else {
                TernaryLogMessage terms{};
                for (std::size_t source = 0; source < 3; ++source)
                    terms[source] = cavity[source] + potential[source][target];
                raw[target] = logSumExp(terms);
            }
        }
        normalizeLogMessage(raw);
        return raw;
    };

    auto report = initializeReport(
        problem, config, FiberTraceBeliefInference::SumProductMixed);
    report.mixedUnaryCost = config.mixedUnaryCost;
    for (std::size_t iteration = 0;
         iteration < config.maximumMessageIterations;
         ++iteration) {
        std::fill(
            totals.begin(), totals.end(), TernaryLogMessage{0.0, 0.0, 0.0});
        for (std::size_t index = 0; index < graph.factors.size(); ++index) {
            for (std::size_t state = 0; state < 3; ++state) {
                totals[graph.factors[index].a][state] += bToA[index][state];
                totals[graph.factors[index].b][state] += aToB[index][state];
            }
        }

        double residual = 0.0;
        for (std::size_t index = 0; index < graph.factors.size(); ++index) {
            const auto& factor = graph.factors[index];
            TernaryLogMessage cavityA{};
            TernaryLogMessage cavityB{};
            for (std::size_t state = 0; state < 3; ++state) {
                cavityA[state] = totals[factor.a][state] -
                    bToA[index][state] + logUnary[state];
                cavityB[state] = totals[factor.b][state] -
                    aToB[index][state] + logUnary[state];
            }
            const auto rawAToB = rawMessage(
                cavityA, logPotential[index], factor.a == problem.seed);
            const auto rawBToA = rawMessage(
                cavityB, logPotential[index], factor.b == problem.seed);
            for (std::size_t state = 0; state < 3; ++state) {
                nextAToB[index][state] = aToB[index][state] +
                    config.messageDamping *
                    (rawAToB[state] - aToB[index][state]);
                nextBToA[index][state] = bToA[index][state] +
                    config.messageDamping *
                    (rawBToA[state] - bToA[index][state]);
            }
            normalizeLogMessage(nextAToB[index]);
            normalizeLogMessage(nextBToA[index]);
            for (std::size_t state = 0; state < 3; ++state) {
                residual = std::max({
                    residual,
                    std::abs(nextAToB[index][state] - aToB[index][state]),
                    std::abs(nextBToA[index][state] - bToA[index][state]),
                });
            }
        }
        aToB.swap(nextAToB);
        bToA.swap(nextBToA);
        report.messageIterations = iteration + 1;
        report.messageResidual = residual;
        if (residual <= config.messageResidualTolerance) {
            report.messageConverged = true;
            break;
        }
    }

    std::fill(
        totals.begin(), totals.end(), TernaryLogMessage{0.0, 0.0, 0.0});
    for (std::size_t index = 0; index < graph.factors.size(); ++index) {
        for (std::size_t state = 0; state < 3; ++state) {
            totals[graph.factors[index].a][state] += bToA[index][state];
            totals[graph.factors[index].b][state] += aToB[index][state];
        }
    }
    report.verticalProbability.resize(nodeCount);
    report.mixedProbability.resize(nodeCount);
    report.horizontalProbability.resize(nodeCount);
    report.horizontalness.resize(nodeCount);
    double weightedHorizontalness = 0.0;
    double totalWeight = 0.0;
    for (std::size_t node = 0; node < nodeCount; ++node) {
        TernaryLogMessage marginal{
            totals[node][0] + logUnary[0],
            totals[node][1] + logUnary[1],
            totals[node][2] + logUnary[2]};
        if (node == problem.seed) {
            marginal = {
                -std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(),
                0.0};
        } else {
            normalizeLogMessage(marginal);
        }
        report.verticalProbability[node] = std::exp(marginal[0]);
        report.mixedProbability[node] = std::exp(marginal[1]);
        report.horizontalProbability[node] = std::exp(marginal[2]);
        report.horizontalness[node] = report.horizontalProbability[node] +
            0.5 * report.mixedProbability[node];
        weightedHorizontalness += problem.normalizedArcWeights[node] *
            report.horizontalness[node];
        totalWeight += problem.normalizedArcWeights[node];
    }
    report.achievedHorizontalFraction = weightedHorizontalness / totalWeight;
    report.status = report.messageConverged
        ? "converged"
        : "message_limit";
    report.solveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return report;
}

namespace
{

FiberTraceConstraintConsistencyReport analyzeConstraintConsistency(
    const FiberTraceConstraintReport& constraints,
    std::span<const double> horizontalnessValues,
    std::span<const double> verticalProbabilities,
    std::span<const double> mixedProbabilities,
    std::span<const double> horizontalProbabilities,
    double verticalThreshold,
    double horizontalThreshold)
{
    if (!std::isfinite(verticalThreshold) ||
        !std::isfinite(horizontalThreshold) || verticalThreshold < 0.0 ||
        horizontalThreshold > 1.0 ||
        !(verticalThreshold < horizontalThreshold)) {
        throw std::invalid_argument(
            "BP consistency thresholds must satisfy 0 <= V < H <= 1");
    }
    if (horizontalnessValues.empty()) {
        throw std::invalid_argument(
            "BP consistency requires represented fibers");
    }
    for (const double value : horizontalnessValues) {
        if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
            throw std::invalid_argument(
                "BP consistency horizontalness must be finite in [0, 1]");
        }
    }
    const bool ternary = !verticalProbabilities.empty() ||
        !mixedProbabilities.empty() || !horizontalProbabilities.empty();
    if (ternary &&
        (verticalProbabilities.size() != horizontalnessValues.size() ||
         mixedProbabilities.size() != horizontalnessValues.size() ||
         horizontalProbabilities.size() != horizontalnessValues.size())) {
        throw std::invalid_argument(
            "BP consistency ternary probabilities must match represented fibers");
    }
    if (ternary) {
        for (std::size_t trace = 0;
             trace < horizontalnessValues.size();
             ++trace) {
            const double vertical = verticalProbabilities[trace];
            const double mixed = mixedProbabilities[trace];
            const double horizontal = horizontalProbabilities[trace];
            if (!std::isfinite(vertical) || !std::isfinite(mixed) ||
                !std::isfinite(horizontal) || vertical < 0.0 ||
                mixed < 0.0 || horizontal < 0.0 || vertical > 1.0 ||
                mixed > 1.0 || horizontal > 1.0 ||
                std::abs(vertical + mixed + horizontal - 1.0) > 1.0e-9) {
                throw std::invalid_argument(
                    "BP consistency ternary probabilities must be normalized in [0, 1]");
            }
        }
    }

    const Graph graph = buildGraph(horizontalnessValues.size(), constraints);
    FiberTraceConstraintConsistencyReport report;
    report.verticalThreshold = verticalThreshold;
    report.horizontalThreshold = horizontalThreshold;
    report.traces.resize(horizontalnessValues.size());
    std::vector<double> weightedHardMismatch(horizontalnessValues.size(), 0.0);
    std::vector<double> weightedSoftMismatch(horizontalnessValues.size(), 0.0);
    std::vector<double> horizontalSupport(horizontalnessValues.size(), 0.0);
    std::vector<double> verticalSupport(horizontalnessValues.size(), 0.0);
    std::vector<double> neighborCertainty(horizontalnessValues.size(), 0.0);
    for (std::size_t trace = 0; trace < report.traces.size(); ++trace)
        report.traces[trace].traceIndex = trace;

    const auto resolvedLabel = [&](std::size_t trace) -> int {
        const double value = horizontalnessValues[trace];
        if (value <= verticalThreshold)
            return 0;
        if (value >= horizontalThreshold)
            return 1;
        return -1;
    };
    for (const auto& factor : graph.factors) {
        const double strength = std::abs(
            factor.sameCost - factor.differentCost);
        const double aValue = horizontalnessValues[factor.a];
        const double bValue = horizontalnessValues[factor.b];
        const bool prefersSame = factor.sameCost < factor.differentCost;
        const double aHorizontal = ternary
            ? horizontalProbabilities[factor.a]
            : aValue;
        const double aVertical = ternary
            ? verticalProbabilities[factor.a]
            : 1.0 - aValue;
        const double bHorizontal = ternary
            ? horizontalProbabilities[factor.b]
            : bValue;
        const double bVertical = ternary
            ? verticalProbabilities[factor.b]
            : 1.0 - bValue;
        const double sameProbability =
            aHorizontal * bHorizontal + aVertical * bVertical;
        const double differentProbability =
            aHorizontal * bVertical + aVertical * bHorizontal;
        const double violationProbability = prefersSame
            ? differentProbability
            : sameProbability;
        const int aLabel = resolvedLabel(factor.a);
        const int bLabel = resolvedLabel(factor.b);
        const bool resolved = aLabel >= 0 && bLabel >= 0;
        const bool labelsSame = aLabel == bLabel;
        const bool mismatch = resolved && labelsSame != prefersSame;

        for (const std::size_t trace : {factor.a, factor.b}) {
            auto& current = report.traces[trace];
            ++current.degree;
            current.incidentMeasurements += factor.measurements;
            current.totalStrength += strength;
            weightedSoftMismatch[trace] += strength * violationProbability;
            if (resolved) {
                ++current.resolvedDegree;
                current.resolvedStrength += strength;
                if (mismatch) {
                    ++current.hardMismatches;
                    weightedHardMismatch[trace] += strength;
                }
            } else {
                ++current.unresolvedDegree;
                current.unresolvedStrength += strength;
            }
        }
        horizontalSupport[factor.a] += strength *
            (prefersSame ? bHorizontal : bVertical);
        verticalSupport[factor.a] += strength *
            (prefersSame ? bVertical : bHorizontal);
        horizontalSupport[factor.b] += strength *
            (prefersSame ? aHorizontal : aVertical);
        verticalSupport[factor.b] += strength *
            (prefersSame ? aVertical : aHorizontal);
        neighborCertainty[factor.a] += strength *
            std::abs(bHorizontal - bVertical);
        neighborCertainty[factor.b] += strength *
            std::abs(aHorizontal - aVertical);
    }

    for (std::size_t trace = 0; trace < report.traces.size(); ++trace) {
        auto& current = report.traces[trace];
        if (current.resolvedDegree != 0) {
            current.hardMismatchRate =
                static_cast<double>(current.hardMismatches) /
                static_cast<double>(current.resolvedDegree);
        }
        if (current.resolvedStrength > 0.0) {
            current.weightedHardMismatchRate =
                weightedHardMismatch[trace] / current.resolvedStrength;
        }
        if (current.totalStrength > 0.0) {
            current.softMismatchProxy =
                weightedSoftMismatch[trace] / current.totalStrength;
            current.neighborCertainty =
                neighborCertainty[trace] / current.totalStrength;
        }
        const double support =
            horizontalSupport[trace] + verticalSupport[trace];
        if (support > 0.0) {
            current.neighborSupportBalance = 2.0 * std::min(
                horizontalSupport[trace], verticalSupport[trace]) / support;
        }
    }
    return report;
}

}  // namespace

FiberTraceConstraintConsistencyReport
analyzeFiberTraceConstraintConsistency(
    const FiberTraceConstraintReport& constraints,
    std::span<const double> horizontalnessValues,
    double verticalThreshold,
    double horizontalThreshold)
{
    return analyzeConstraintConsistency(
        constraints,
        horizontalnessValues,
        {},
        {},
        {},
        verticalThreshold,
        horizontalThreshold);
}

FiberTraceConstraintConsistencyReport
analyzeMixedFiberTraceConstraintConsistency(
    const FiberTraceConstraintReport& constraints,
    std::span<const double> verticalProbabilities,
    std::span<const double> mixedProbabilities,
    std::span<const double> horizontalProbabilities,
    double verticalThreshold,
    double horizontalThreshold)
{
    if (verticalProbabilities.size() != mixedProbabilities.size() ||
        verticalProbabilities.size() != horizontalProbabilities.size()) {
        throw std::invalid_argument(
            "BP consistency ternary probabilities must have equal sizes");
    }
    std::vector<double> horizontalnessValues(verticalProbabilities.size());
    for (std::size_t trace = 0; trace < horizontalnessValues.size(); ++trace) {
        horizontalnessValues[trace] = horizontalProbabilities[trace] +
            0.5 * mixedProbabilities[trace];
    }
    return analyzeConstraintConsistency(
        constraints,
        horizontalnessValues,
        verticalProbabilities,
        mixedProbabilities,
        horizontalProbabilities,
        verticalThreshold,
        horizontalThreshold);
}

}  // namespace vc::fiber_tracer
