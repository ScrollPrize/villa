#include "vc/fiber_tracer/FiberTraceBeliefPropagation.hpp"

#include "vc/fiber_tracer/FiberTraceSeed.hpp"

#include <algorithm>
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
            std::abs(
                constraint.parallelScore + constraint.perpendicularScore -
                1.0) > 1.0e-9 ||
            !(constraint.parallelScore < 0.5) ||
            !(constraint.perpendicularScore > 0.5)) {
            throw std::invalid_argument(
                "BP requires complementary measured perpendicular scores");
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
        const std::size_t index = graph.factors.size();
        graph.factors.push_back(factor);
        graph.adjacency[factor.a].push_back(index);
        graph.adjacency[factor.b].push_back(index);
        graph.measurements += factor.measurements;
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

FiberTraceBeliefPropagationReport solveFiberTraceBeliefPropagation(
    const std::vector<FiberletCropTraceLine>& traces,
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefPropagationConfig& config)
{
    const auto started = std::chrono::steady_clock::now();
    validateConfig(config);
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
    const Graph graph = buildGraph(traces.size(), constraints);

    std::vector<double> weights;
    weights.reserve(traces.size());
    double totalArc = 0.0;
    for (const auto& trace : geometry.traces)
        totalArc += trace.arcLengthBaseVoxels;
    const double meanArc = totalArc / static_cast<double>(traces.size());
    for (const auto& trace : geometry.traces)
        weights.push_back(trace.arcLengthBaseVoxels / meanArc);

    FiberTraceBeliefPropagationReport report;
    report.normalizedArcWeights = weights;
    report.seedTraceIndex = *seed;
    report.factors = graph.factors.size();
    report.mergedMeasurements = graph.measurements;
    report.connectedComponents = graph.components;
    report.isolatedTraces = graph.isolated;
    report.targetHorizontalFraction = config.targetHorizontalFraction;

    if (config.balanceMode == FiberTraceBalanceMode::None) {
        auto solution = solveField(graph, weights, *seed, 0.0, config);
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
            auto current = solveField(graph, weights, *seed, field, config);
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
        auto low = solveField(graph, weights, *seed, lowField, config);
        auto high = solveField(graph, weights, *seed, highField, config);
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
                    graph, weights, *seed, middleField, config);
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
                if (trace == *seed) {
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

FiberTraceConstraintConsistencyReport
analyzeFiberTraceConstraintConsistency(
    const FiberTraceConstraintReport& constraints,
    std::span<const double> horizontalnessValues,
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
        const double sameProbability =
            aValue * bValue + (1.0 - aValue) * (1.0 - bValue);
        const int aLabel = resolvedLabel(factor.a);
        const int bLabel = resolvedLabel(factor.b);
        const bool resolved = aLabel >= 0 && bLabel >= 0;
        const bool mismatch = resolved && aLabel == bLabel;

        for (const std::size_t trace : {factor.a, factor.b}) {
            auto& current = report.traces[trace];
            ++current.degree;
            current.incidentMeasurements += factor.measurements;
            current.totalStrength += strength;
            weightedSoftMismatch[trace] += strength * sameProbability;
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
        horizontalSupport[factor.a] += strength * (1.0 - bValue);
        verticalSupport[factor.a] += strength * bValue;
        horizontalSupport[factor.b] += strength * (1.0 - aValue);
        verticalSupport[factor.b] += strength * aValue;
        neighborCertainty[factor.a] += strength * std::abs(2.0 * bValue - 1.0);
        neighborCertainty[factor.b] += strength * std::abs(2.0 * aValue - 1.0);
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
            current.softSameLabelProxy =
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

}  // namespace vc::fiber_tracer
