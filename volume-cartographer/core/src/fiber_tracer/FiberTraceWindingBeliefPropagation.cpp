#include "vc/fiber_tracer/FiberTraceWindingBeliefPropagation.hpp"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

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

#include <omp.h>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-12;

enum class JointClass : unsigned char { A = 0, Mixed = 1, B = 2 };

struct Measurement {
    std::size_t constraintIndex = 0;
    std::size_t a = 0;
    std::size_t b = 0;
    double parallel = 0.0;
    double perpendicular = 0.0;
    double parallelMultiplier = 1.0;
    double perpendicularMultiplier = 1.0;
    double parallelDistance = 0.0;
    std::optional<double> perpendicularSignedDelta;
    std::optional<std::size_t> normalComponent;
};

struct Edge {
    std::size_t a = 0;
    std::size_t b = 0;
    std::vector<Measurement> measurements;
};

struct PreparedWinding {
    std::vector<std::size_t> pieceToNode;
    std::vector<std::vector<std::size_t>> piecesByNode;
    std::vector<Edge> edges;
    std::vector<std::vector<std::size_t>> adjacency;
    std::vector<std::size_t> componentByNode;
    std::vector<std::size_t> integerGaugeByNode;
    std::vector<std::size_t> gaugeNodeByComponent;
    std::vector<std::size_t> gaugePieceByComponent;
    std::vector<std::size_t> integerGaugeNodes;
    std::vector<FiberTraceWindingFactorDiagnostic> diagnostics;
};

void validateConfig(const FiberTraceWindingBeliefPropagationConfig& config)
{
    if (!std::isfinite(config.temperature) || !(config.temperature > 0.0))
        throw std::invalid_argument("Winding BP temperature must be positive and finite");
    if (!std::isfinite(config.messageDamping) ||
        !(config.messageDamping > 0.0) || config.messageDamping > 1.0) {
        throw std::invalid_argument("Winding BP damping must be in (0, 1]");
    }
    if (!std::isfinite(config.messageResidualTolerance) ||
        config.messageResidualTolerance < 0.0 ||
        !std::isfinite(config.boundaryProbabilityThreshold) ||
        !(config.boundaryProbabilityThreshold > 0.0) ||
        config.boundaryProbabilityThreshold >= 1.0) {
        throw std::invalid_argument("Winding BP tolerances are invalid");
    }
    if (config.maximumMessageIterations == 0 ||
        config.maximumTotalCandidateStates == 0 ||
        config.parallelWorkers == 0) {
        throw std::invalid_argument("Winding BP limits and worker count must be positive");
    }
    if (config.parallelWindingDistanceCutoff &&
        (!std::isfinite(*config.parallelWindingDistanceCutoff) ||
         !(*config.parallelWindingDistanceCutoff > 0.0))) {
        throw std::invalid_argument(
            "Winding BP parallel winding cutoff must be finite and positive");
    }
}

double windingWeightMultiplier(double effectiveTarget)
{
    const double exponent = std::floor(std::abs(effectiveTarget));
    if (!(exponent > 0.0))
        return 1.0;
    constexpr int maximumSubnormalExponent =
        std::numeric_limits<double>::digits -
        std::numeric_limits<double>::min_exponent;
    if (exponent > static_cast<double>(maximumSubnormalExponent))
        return 0.0;
    return std::ldexp(1.0, -static_cast<int>(exponent));
}

double parallelWindingWeight(const Measurement& measurement)
{
    return measurement.parallelMultiplier * measurement.parallel;
}

double perpendicularWindingWeight(const Measurement& measurement)
{
    return measurement.perpendicularSignedDelta
        ? measurement.perpendicularMultiplier * measurement.perpendicular
        : 0.0;
}

PreparedWinding prepareWinding(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceWindingBeliefPropagationConfig& config,
    std::span<const JointClass> fixedOrientations = {},
    bool quantizeComponentTargets = false)
{
    if (topology.pieceLines.size() != constraints.pieces.size() ||
        topology.pieceCenterDistanceBaseVoxels.size() != constraints.pieces.size()) {
        throw std::invalid_argument("Winding BP topology does not match constraints");
    }
    PreparedWinding result;
    result.pieceToNode.resize(constraints.pieces.size());
    result.piecesByNode.resize(constraints.pieces.size());
    for (std::size_t piece = 0; piece < constraints.pieces.size(); ++piece) {
        result.pieceToNode[piece] = piece;
        result.piecesByNode[piece].push_back(piece);
    }

    std::map<std::pair<std::size_t, std::size_t>, std::size_t> edgeByPair;
    result.diagnostics.reserve(
        topology.hardConstraintIndices.size() +
        topology.softConstraintIndices.size());
    const auto addConstraint = [&](std::size_t index, bool continuity) {
        const auto& constraint = constraints.constraints.at(index);
        if (!fixedOrientations.empty() &&
            (fixedOrientations[constraint.pieceA] == JointClass::Mixed ||
             fixedOrientations[constraint.pieceB] == JointClass::Mixed)) {
            return;
        }
        const std::size_t originalA = result.pieceToNode[constraint.pieceA];
        const std::size_t originalB = result.pieceToNode[constraint.pieceB];
        const std::size_t a = std::min(originalA, originalB);
        const std::size_t b = std::max(originalA, originalB);
        std::optional<double> canonicalSignedDelta = continuity
            ? std::optional<double>{0.0}
            : constraint.signedWindingDelta;
        if (canonicalSignedDelta && originalA > originalB)
            *canonicalSignedDelta = -*canonicalSignedDelta;
        double effectiveParallelDistance = 0.0;
        std::optional<double> effectivePerpendicularSignedDelta =
            canonicalSignedDelta;
        if (quantizeComponentTargets && !continuity) {
            effectiveParallelDistance = quantizedIntegerWindingTarget(
                constraint.windingDistance);
            if (canonicalSignedDelta) {
                effectivePerpendicularSignedDelta = quantizedHalfWindingTarget(
                    *canonicalSignedDelta);
            }
        }
        const bool parallelRetained =
            (!config.parallelWindingDistanceCutoff || continuity ||
             effectiveParallelDistance <
                 *config.parallelWindingDistanceCutoff);
        const double parallelMultiplier =
            quantizeComponentTargets && !continuity
            ? windingWeightMultiplier(effectiveParallelDistance)
            : 1.0;
        const double perpendicularMultiplier =
            effectivePerpendicularSignedDelta && quantizeComponentTargets &&
                !continuity
            ? windingWeightMultiplier(*effectivePerpendicularSignedDelta)
            : 1.0;
        Measurement measurement{
            index,
            a,
            b,
            constraint.parallelScore,
            constraint.perpendicularScore,
            parallelRetained ? parallelMultiplier : 0.0,
            perpendicularMultiplier,
            effectiveParallelDistance,
            effectivePerpendicularSignedDelta,
            continuity ? std::nullopt : constraint.windingNormalComponent,
        };
        result.diagnostics.push_back({
            index,
            constraint.pieceA,
            constraint.pieceB,
            a,
            b,
            constraint.parallelScore,
            constraint.perpendicularScore,
            parallelMultiplier,
            perpendicularMultiplier,
            parallelWindingWeight(measurement),
            perpendicularWindingWeight(measurement),
            constraint.signedWindingDelta,
            canonicalSignedDelta,
            effectiveParallelDistance,
            effectivePerpendicularSignedDelta,
            continuity ? std::nullopt : constraint.windingNormalComponent,
            parallelRetained,
            false,
        });
        const std::pair<std::size_t, std::size_t> key{a, b};
        const auto [found, inserted] = edgeByPair.try_emplace(
            key, result.edges.size());
        if (inserted)
            result.edges.push_back({a, b, {}});
        result.edges[found->second].measurements.push_back(std::move(measurement));
    };
    for (const std::size_t index : topology.hardConstraintIndices)
        addConstraint(index, true);
    for (const std::size_t index : topology.softConstraintIndices)
        addConstraint(index, false);

    result.adjacency.resize(result.piecesByNode.size());
    std::vector<std::vector<std::size_t>> windingAdjacency(
        result.piecesByNode.size());
    for (std::size_t edge = 0; edge < result.edges.size(); ++edge) {
        bool factorPositive = false;
        bool windingPositive = false;
        for (const auto& measurement : result.edges[edge].measurements) {
            factorPositive = factorPositive || measurement.parallel > 0.0 ||
                measurement.perpendicular > 0.0;
            windingPositive = windingPositive ||
                parallelWindingWeight(measurement) > 0.0 ||
                perpendicularWindingWeight(measurement) > 0.0;
        }
        if (factorPositive) {
            result.adjacency[result.edges[edge].a].push_back(edge);
            result.adjacency[result.edges[edge].b].push_back(edge);
        }
        if (windingPositive) {
            windingAdjacency[result.edges[edge].a].push_back(edge);
            windingAdjacency[result.edges[edge].b].push_back(edge);
        }
    }

    const auto centralGauge = [&](std::span<const std::size_t> nodes) {
        std::size_t gaugeNode = nodes.front();
        std::size_t gaugePiece = result.piecesByNode[gaugeNode].front();
        for (const std::size_t node : nodes) {
            for (const std::size_t piece : result.piecesByNode[node]) {
                if (topology.pieceCenterDistanceBaseVoxels[piece] <
                        topology.pieceCenterDistanceBaseVoxels[gaugePiece] ||
                    (topology.pieceCenterDistanceBaseVoxels[piece] ==
                         topology.pieceCenterDistanceBaseVoxels[gaugePiece] &&
                     piece < gaugePiece)) {
                    gaugeNode = node;
                    gaugePiece = piece;
                }
            }
        }
        return std::pair{gaugeNode, gaugePiece};
    };

    // H/V and normal-sign variables remain connected by every active factor,
    // even when one component of that factor's winding loss is filtered.
    result.componentByNode.assign(result.piecesByNode.size(), 0);
    std::vector<unsigned char> visited(result.piecesByNode.size(), 0);
    for (std::size_t start = 0; start < result.piecesByNode.size(); ++start) {
        if (visited[start] != 0)
            continue;
        const std::size_t component = result.gaugeNodeByComponent.size();
        std::vector<std::size_t> nodes;
        std::queue<std::size_t> pending;
        pending.push(start);
        visited[start] = 1;
        while (!pending.empty()) {
            const std::size_t node = pending.front();
            pending.pop();
            nodes.push_back(node);
            result.componentByNode[node] = component;
            for (const std::size_t edgeIndex : result.adjacency[node]) {
                const auto& edge = result.edges[edgeIndex];
                const std::size_t neighbor = edge.a == node ? edge.b : edge.a;
                if (visited[neighbor] == 0) {
                    visited[neighbor] = 1;
                    pending.push(neighbor);
                }
            }
        }
        const auto [gaugeNode, gaugePiece] = centralGauge(nodes);
        result.gaugeNodeByComponent.push_back(gaugeNode);
        result.gaugePieceByComponent.push_back(gaugePiece);
    }

    // Winding offsets have a separate gauge for every component that retains
    // an effective winding term. Additional gauges fix only integer zero; they
    // must not pin H/V and thereby sever orientation-only factors.
    std::fill(visited.begin(), visited.end(), 0);
    result.integerGaugeByNode.assign(result.piecesByNode.size(), 0);
    std::vector<unsigned char> classGauge(result.piecesByNode.size(), 0);
    for (const std::size_t node : result.gaugeNodeByComponent)
        classGauge[node] = 1;
    for (std::size_t start = 0; start < result.piecesByNode.size(); ++start) {
        if (visited[start] != 0)
            continue;
        const std::size_t integerGauge = result.integerGaugeNodes.size();
        std::vector<std::size_t> nodes;
        std::queue<std::size_t> pending;
        pending.push(start);
        visited[start] = 1;
        while (!pending.empty()) {
            const std::size_t node = pending.front();
            pending.pop();
            nodes.push_back(node);
            result.integerGaugeByNode[node] = integerGauge;
            for (const std::size_t edgeIndex : windingAdjacency[node]) {
                const auto& edge = result.edges[edgeIndex];
                const std::size_t neighbor = edge.a == node ? edge.b : edge.a;
                if (visited[neighbor] == 0) {
                    visited[neighbor] = 1;
                    pending.push(neighbor);
                }
            }
        }
        std::size_t gaugeNode = centralGauge(nodes).first;
        const auto classGaugeInComponent = std::find_if(
            nodes.begin(), nodes.end(), [&](std::size_t node) {
                return classGauge[node] != 0;
            });
        if (classGaugeInComponent != nodes.end())
            gaugeNode = *classGaugeInComponent;
        result.integerGaugeNodes.push_back(gaugeNode);

        std::optional<std::size_t> normalComponent;
        for (const std::size_t node : nodes) {
            for (const std::size_t edgeIndex : windingAdjacency[node]) {
                const auto& edge = result.edges[edgeIndex];
                if (edge.a != node)
                    continue;
                for (const auto& measurement : edge.measurements) {
                    if (!measurement.perpendicularSignedDelta ||
                        !(measurement.perpendicular > 0.0) ||
                        !measurement.normalComponent) {
                        continue;
                    }
                    if (normalComponent &&
                        *normalComponent != *measurement.normalComponent) {
                        throw std::invalid_argument(
                            "Winding component combines independent normal alignment gauges");
                    }
                    normalComponent = measurement.normalComponent;
                }
            }
        }
    }
    return result;
}

double measurementSquaredWeight(const Measurement& measurement)
{
    return parallelWindingWeight(measurement) +
        perpendicularWindingWeight(measurement);
}

double measurementSquaredTarget(const Measurement& measurement)
{
    const double weight = measurementSquaredWeight(measurement);
    if (!(weight > 0.0))
        return 0.0;
    const double parallelTarget = measurement.parallelDistance == 0.0
        ? 0.0
        : measurement.perpendicularSignedDelta
            ? std::copysign(
                  measurement.parallelDistance,
                  *measurement.perpendicularSignedDelta)
            : measurement.parallelDistance;
    return (parallelWindingWeight(measurement) * parallelTarget +
            perpendicularWindingWeight(measurement) *
                measurement.perpendicularSignedDelta.value_or(0.0)) /
        weight;
}

std::vector<double> solveContinuous(
    const PreparedWinding& problem,
    double& rootMeanSquareResidual)
{
    const std::size_t nodeCount = problem.piecesByNode.size();
    std::vector<unsigned char> gauge(nodeCount, 0);
    for (const std::size_t node : problem.integerGaugeNodes)
        gauge[node] = 1;
    std::vector<std::size_t> variable(nodeCount, std::numeric_limits<std::size_t>::max());
    std::size_t variables = 0;
    for (std::size_t node = 0; node < nodeCount; ++node) {
        if (gauge[node] == 0)
            variable[node] = variables++;
    }
    Eigen::SparseMatrix<double> matrix(
        static_cast<Eigen::Index>(variables),
        static_cast<Eigen::Index>(variables));
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(variables));
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(problem.edges.size() * 4);
    for (const auto& edge : problem.edges) {
        double weight = 0.0;
        double weightedTarget = 0.0;
        for (const auto& measurement : edge.measurements) {
            const double currentWeight = measurementSquaredWeight(measurement);
            weight += currentWeight;
            weightedTarget += currentWeight * measurementSquaredTarget(measurement);
        }
        if (!(weight > 0.0))
            continue;
        const double target = weightedTarget / weight;
        const bool variableA = gauge[edge.a] == 0;
        const bool variableB = gauge[edge.b] == 0;
        if (variableA) {
            triplets.emplace_back(variable[edge.a], variable[edge.a], weight);
            rhs[static_cast<Eigen::Index>(variable[edge.a])] -= weight * target;
        }
        if (variableB) {
            triplets.emplace_back(variable[edge.b], variable[edge.b], weight);
            rhs[static_cast<Eigen::Index>(variable[edge.b])] += weight * target;
        }
        if (variableA && variableB) {
            triplets.emplace_back(variable[edge.a], variable[edge.b], -weight);
            triplets.emplace_back(variable[edge.b], variable[edge.a], -weight);
        }
    }
    std::vector<double> values(nodeCount, 0.0);
    if (variables > 0) {
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(matrix);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Winding continuous factorization failed");
        const Eigen::VectorXd solution = solver.solve(rhs);
        if (solver.info() != Eigen::Success || !solution.allFinite())
            throw std::runtime_error("Winding continuous solve failed");
        for (std::size_t node = 0; node < nodeCount; ++node) {
            if (gauge[node] == 0)
                values[node] = solution[static_cast<Eigen::Index>(variable[node])];
        }
    }

    double squared = 0.0;
    std::size_t terms = 0;
    const auto accumulate = [&](const Measurement& measurement) {
        const double delta = values[measurement.b] - values[measurement.a];
        const double parallelWeight = parallelWindingWeight(measurement);
        if (parallelWeight > 0.0) {
            const double residual =
                std::abs(delta) - measurement.parallelDistance;
            squared += parallelWeight * residual * residual;
            ++terms;
        }
        const double perpendicularWeight =
            perpendicularWindingWeight(measurement);
        if (measurement.perpendicularSignedDelta && perpendicularWeight > 0.0) {
            const double residual =
                delta - *measurement.perpendicularSignedDelta;
            squared += perpendicularWeight * residual * residual;
            ++terms;
        }
    };
    for (const auto& edge : problem.edges)
        for (const auto& measurement : edge.measurements)
            accumulate(measurement);
    rootMeanSquareResidual = terms == 0 ? 0.0 :
        std::sqrt(squared / static_cast<double>(terms));
    return values;
}

double robustCost(const Edge& edge, int labelA, int labelB)
{
    const double delta = static_cast<double>(labelB - labelA);
    double cost = 0.0;
    for (const auto& measurement : edge.measurements) {
        cost += parallelWindingWeight(measurement) *
            std::abs(std::abs(delta) - measurement.parallelDistance);
        if (measurement.perpendicularSignedDelta) {
            cost += perpendicularWindingWeight(measurement) *
                std::abs(delta - *measurement.perpendicularSignedDelta);
        }
    }
    return cost;
}

double logSumExp(const std::vector<double>& values)
{
    const double maximum = *std::max_element(values.begin(), values.end());
    if (!std::isfinite(maximum))
        return maximum;
    double sum = 0.0;
    for (const double value : values)
        sum += std::exp(value - maximum);
    return maximum + std::log(sum);
}

struct LogProductAccumulator {
    double finiteSum = 0.0;
    std::size_t negativeInfinityCount = 0;
};

bool isNegativeInfinity(double value)
{
    return std::isinf(value) && value < 0.0;
}

void addLogFactor(LogProductAccumulator& accumulator, double value)
{
    if (isNegativeInfinity(value)) {
        ++accumulator.negativeInfinityCount;
    } else if (std::isfinite(value)) {
        accumulator.finiteSum += value;
    } else {
        throw std::runtime_error("Winding BP encountered an invalid log factor");
    }
}

double logProductValue(const LogProductAccumulator& accumulator)
{
    return accumulator.negativeInfinityCount == 0
        ? accumulator.finiteSum
        : -std::numeric_limits<double>::infinity();
}

double logCavityValue(
    const LogProductAccumulator& accumulator,
    double removed)
{
    LogProductAccumulator cavity = accumulator;
    if (isNegativeInfinity(removed)) {
        if (cavity.negativeInfinityCount == 0)
            throw std::logic_error("Winding BP cavity infinity count underflow");
        --cavity.negativeInfinityCount;
    } else if (std::isfinite(removed)) {
        cavity.finiteSum -= removed;
    } else {
        throw std::runtime_error("Winding BP encountered an invalid cavity factor");
    }
    return logProductValue(cavity);
}

void normalizeLogVector(std::vector<double>& values)
{
    const double normalization = logSumExp(values);
    if (!std::isfinite(normalization))
        throw std::runtime_error("Winding BP message has no finite state");
    for (double& value : values)
        value -= normalization;
}

double dampMessage(
    std::vector<double>& target,
    const std::vector<double>& current,
    double damping)
{
    normalizeLogVector(target);
    double residual = 0.0;
    for (std::size_t state = 0; state < target.size(); ++state) {
        const bool targetImpossible = isNegativeInfinity(target[state]);
        const bool currentImpossible = isNegativeInfinity(current[state]);
        if (targetImpossible || currentImpossible) {
            if (targetImpossible != currentImpossible)
                residual = std::numeric_limits<double>::infinity();
            if (currentImpossible && !targetImpossible) {
                // A remapped support state has no meaningful old probability.
                // Adopt its finite update immediately instead of interpolating
                // through negative infinity.
            } else if (targetImpossible) {
                target[state] = -std::numeric_limits<double>::infinity();
            }
            continue;
        }
        const double damped = current[state] +
            damping * (target[state] - current[state]);
        residual = std::max(residual, std::abs(damped - current[state]));
        target[state] = damped;
    }
    normalizeLogVector(target);
    return residual;
}

template <std::size_t Size>
double logSumExp(const std::array<double, Size>& values)
{
    const double maximum = *std::max_element(values.begin(), values.end());
    if (!std::isfinite(maximum))
        return maximum;
    double sum = 0.0;
    for (const double value : values)
        sum += std::exp(value - maximum);
    return maximum + std::log(sum);
}

struct DiscreteRound {
    std::vector<std::vector<double>> probabilities;
    std::vector<std::vector<double>> totals;
    std::vector<std::vector<double>> aToB;
    std::vector<std::vector<double>> bToA;
    std::size_t iterations = 0;
    double residual = 0.0;
    bool converged = false;
    std::size_t effectiveWorkers = 1;
};

template <typename LogPotential>
DiscreteRound solvePairwiseRound(
    const PreparedWinding& problem,
    const std::vector<std::vector<double>>& logUnary,
    const FiberTraceWindingBeliefPropagationConfig& config,
    const LogPotential& logPotential,
    const std::function<void(std::size_t, double)>& progress = {})
{
    const std::size_t nodeCount = problem.piecesByNode.size();
    if (logUnary.size() != nodeCount)
        throw std::invalid_argument("Pairwise BP unary size does not match nodes");
    DiscreteRound result;
    result.aToB.resize(problem.edges.size());
    result.bToA.resize(problem.edges.size());
    std::vector<std::vector<double>> nextAToB(problem.edges.size());
    std::vector<std::vector<double>> nextBToA(problem.edges.size());
    std::vector<std::vector<LogProductAccumulator>> accumulators(nodeCount);
    for (std::size_t edge = 0; edge < problem.edges.size(); ++edge) {
        const auto& current = problem.edges[edge];
        result.aToB[edge].assign(logUnary[current.b].size(), 0.0);
        result.bToA[edge].assign(logUnary[current.a].size(), 0.0);
        nextAToB[edge] = result.aToB[edge];
        nextBToA[edge] = result.bToA[edge];
    }
    result.totals.resize(nodeCount);
    const std::size_t runtimeWorkers = static_cast<std::size_t>(
        std::max(1, omp_get_max_threads()));
    const int workers = static_cast<int>(std::min({
        config.parallelWorkers,
        std::max<std::size_t>(1, problem.edges.size()),
        runtimeWorkers,
        static_cast<std::size_t>(std::numeric_limits<int>::max()),
    }));
    const bool useParallel = workers > 1 && problem.edges.size() >= 256;
    result.effectiveWorkers = useParallel
        ? static_cast<std::size_t>(workers)
        : 1;
    for (std::size_t iteration = 0; iteration < config.maximumMessageIterations; ++iteration) {
        #pragma omp parallel for schedule(static) num_threads(workers) if(useParallel)
        for (std::size_t node = 0; node < nodeCount; ++node) {
            accumulators[node].assign(logUnary[node].size(), {});
            result.totals[node].resize(logUnary[node].size());
            for (std::size_t state = 0; state < logUnary[node].size(); ++state)
                addLogFactor(accumulators[node][state], logUnary[node][state]);
            for (const std::size_t edge : problem.adjacency[node]) {
                const auto& current = problem.edges[edge];
                const auto& incoming = current.a == node
                    ? result.bToA[edge]
                    : result.aToB[edge];
                for (std::size_t state = 0; state < incoming.size(); ++state)
                    addLogFactor(accumulators[node][state], incoming[state]);
            }
            for (std::size_t state = 0; state < logUnary[node].size(); ++state)
                result.totals[node][state] =
                    logProductValue(accumulators[node][state]);
        }
        double residual = 0.0;
        #pragma omp parallel for schedule(static) num_threads(workers) if(useParallel) reduction(max : residual)
        for (std::size_t edge = 0; edge < problem.edges.size(); ++edge) {
            const auto& current = problem.edges[edge];
            std::vector<double> candidates;
            candidates.reserve(result.totals[current.a].size());
            for (std::size_t stateB = 0; stateB < result.aToB[edge].size(); ++stateB) {
                candidates.clear();
                for (std::size_t stateA = 0; stateA < result.bToA[edge].size(); ++stateA) {
                    candidates.push_back(
                        logCavityValue(
                            accumulators[current.a][stateA],
                            result.bToA[edge][stateA]) +
                        logPotential(edge, stateA, stateB));
                }
                nextAToB[edge][stateB] = logSumExp(candidates);
            }
            residual = std::max(
                residual,
                dampMessage(
                    nextAToB[edge],
                    result.aToB[edge],
                    config.messageDamping));

            candidates.reserve(result.totals[current.b].size());
            for (std::size_t stateA = 0; stateA < result.bToA[edge].size(); ++stateA) {
                candidates.clear();
                for (std::size_t stateB = 0; stateB < result.aToB[edge].size(); ++stateB) {
                    candidates.push_back(
                        logCavityValue(
                            accumulators[current.b][stateB],
                            result.aToB[edge][stateB]) +
                        logPotential(edge, stateA, stateB));
                }
                nextBToA[edge][stateA] = logSumExp(candidates);
            }
            residual = std::max(
                residual,
                dampMessage(
                    nextBToA[edge],
                    result.bToA[edge],
                    config.messageDamping));
        }
        result.aToB.swap(nextAToB);
        result.bToA.swap(nextBToA);
        result.iterations = iteration + 1;
        result.residual = residual;
        if (progress)
            progress(result.iterations, residual);
        if (residual <= config.messageResidualTolerance) {
            result.converged = true;
            break;
        }
    }

    #pragma omp parallel for schedule(static) num_threads(workers) if(useParallel)
    for (std::size_t node = 0; node < nodeCount; ++node) {
        accumulators[node].assign(logUnary[node].size(), {});
        result.totals[node].resize(logUnary[node].size());
        for (std::size_t state = 0; state < logUnary[node].size(); ++state)
            addLogFactor(accumulators[node][state], logUnary[node][state]);
        for (const std::size_t edge : problem.adjacency[node]) {
            const auto& current = problem.edges[edge];
            const auto& incoming = current.a == node
                ? result.bToA[edge]
                : result.aToB[edge];
            for (std::size_t state = 0; state < incoming.size(); ++state)
                addLogFactor(accumulators[node][state], incoming[state]);
        }
        for (std::size_t state = 0; state < logUnary[node].size(); ++state)
            result.totals[node][state] =
                logProductValue(accumulators[node][state]);
    }
    result.probabilities.resize(nodeCount);
    for (std::size_t node = 0; node < nodeCount; ++node) {
        const double normalization = logSumExp(result.totals[node]);
        result.probabilities[node].resize(result.totals[node].size());
        for (std::size_t state = 0; state < result.totals[node].size(); ++state) {
            result.probabilities[node][state] =
                std::exp(result.totals[node][state] - normalization);
        }
    }
    return result;
}

DiscreteRound solveDiscreteRound(
    const PreparedWinding& problem,
    const std::vector<int>& lower,
    const std::vector<int>& upper,
    const FiberTraceWindingBeliefPropagationConfig& config)
{
    std::vector<std::vector<double>> logUnary(lower.size());
    for (std::size_t node = 0; node < lower.size(); ++node) {
        logUnary[node].assign(
            static_cast<std::size_t>(upper[node] - lower[node] + 1), 0.0);
    }
    return solvePairwiseRound(
        problem,
        logUnary,
        config,
        [&](std::size_t edge, std::size_t stateA, std::size_t stateB) {
            const auto& current = problem.edges[edge];
            const int labelA = lower[current.a] + static_cast<int>(stateA);
            const int labelB = lower[current.b] + static_cast<int>(stateB);
            return -robustCost(current, labelA, labelB) / config.temperature;
        });
}

struct JointState {
    JointClass orientation = JointClass::A;
    int winding = 0;
};

JointClass jointClass(FiberTraceFixedOrientation orientation)
{
    switch (orientation) {
    case FiberTraceFixedOrientation::Horizontal:
        return JointClass::A;
    case FiberTraceFixedOrientation::Mixed:
        return JointClass::Mixed;
    case FiberTraceFixedOrientation::Vertical:
        return JointClass::B;
    }
    throw std::invalid_argument("Fixed winding orientation is invalid");
}

std::vector<JointClass> fixedJointClasses(
    std::span<const FiberTraceFixedOrientation> orientations,
    std::size_t pieceCount)
{
    if (orientations.empty())
        return {};
    if (orientations.size() != pieceCount) {
        throw std::invalid_argument(
            "Fixed winding orientations do not match constraint pieces");
    }
    std::vector<JointClass> result;
    result.reserve(orientations.size());
    for (const auto orientation : orientations)
        result.push_back(jointClass(orientation));
    return result;
}

std::size_t jointActiveOrientationCount(
    std::size_t node,
    std::span<const JointClass> fixedOrientations)
{
    if (fixedOrientations.empty())
        return 2;
    return fixedOrientations[node] == JointClass::Mixed ? 0 : 1;
}

std::size_t jointPieceStateCount(
    std::size_t node,
    std::size_t integerCount,
    std::span<const JointClass> fixedOrientations)
{
    return 1 +
        jointActiveOrientationCount(node, fixedOrientations) * integerCount;
}

JointState jointState(
    std::size_t node,
    std::size_t state,
    int lower,
    std::span<const JointClass> fixedOrientations)
{
    if (state == 0)
        return {JointClass::Mixed, 0};
    --state;
    const std::size_t orientationCount = jointActiveOrientationCount(
        node, fixedOrientations);
    if (orientationCount == 0)
        throw std::out_of_range("Defect-only winding state is invalid");
    if (!fixedOrientations.empty())
        return {fixedOrientations[node],
                lower + static_cast<int>(state / orientationCount)};
    return {
        state % 2 == 0 ? JointClass::A : JointClass::B,
        lower + static_cast<int>(state / orientationCount),
    };
}

double classOffset(JointClass orientation, int sign, double phase)
{
    return orientation == JointClass::B
        ? static_cast<double>(sign) * phase
        : 0.0;
}

FiberTraceFixedOrientation publicOrientation(JointClass orientation)
{
    switch (orientation) {
        case JointClass::A:
            return FiberTraceFixedOrientation::Horizontal;
        case JointClass::Mixed:
            return FiberTraceFixedOrientation::Mixed;
        case JointClass::B:
            return FiberTraceFixedOrientation::Vertical;
    }
    throw std::logic_error("Invalid decoded winding orientation");
}

bool requiresHardWindingSign(const Measurement& measurement)
{
    return measurement.perpendicular > 0.0 &&
        measurement.perpendicularSignedDelta &&
        *measurement.perpendicularSignedDelta != 0.0;
}

bool hardWindingSignCompatible(
    const Measurement& measurement,
    double predictedDelta)
{
    return !requiresHardWindingSign(measurement) ||
        *measurement.perpendicularSignedDelta * predictedDelta > 0.0;
}

bool hardWindingSignCompatible(const Edge& edge, double predictedDelta)
{
    return std::all_of(
        edge.measurements.begin(),
        edge.measurements.end(),
        [&](const Measurement& measurement) {
            return hardWindingSignCompatible(measurement, predictedDelta);
        });
}

double windingEnergy(
    const Edge& edge,
    const JointState& a,
    const JointState& b,
    int sign,
    double phase,
    double scale)
{
    const double delta = static_cast<double>(b.winding - a.winding) +
        classOffset(b.orientation, sign, phase) -
        classOffset(a.orientation, sign, phase);
    const double predictedDelta = delta / scale;
    if (!hardWindingSignCompatible(edge, predictedDelta))
        return std::numeric_limits<double>::infinity();
    double energy = 0.0;
    for (const auto& measurement : edge.measurements) {
        energy += parallelWindingWeight(measurement) *
            std::abs(std::abs(delta) - measurement.parallelDistance);
        if (measurement.perpendicularSignedDelta) {
            energy += perpendicularWindingWeight(measurement) *
                std::abs(
                    predictedDelta - *measurement.perpendicularSignedDelta);
        }
    }
    return energy;
}

double jointLogPotential(
    const Edge& edge,
    JointState a,
    JointState b,
    int sign,
    double phase,
    double scale,
    double temperature)
{
    if (a.orientation == JointClass::Mixed ||
        b.orientation == JointClass::Mixed)
        return 0.0;
    return -windingEnergy(edge, a, b, sign, phase, scale) / temperature;
}

template <typename PredictedDelta>
std::size_t projectDecodedHardSigns(
    const PreparedWinding& problem,
    std::vector<JointState>& decoded,
    std::span<const double> activeConfidence,
    const PredictedDelta& predictedDelta)
{
    if (decoded.size() != problem.piecesByNode.size() ||
        activeConfidence.size() != decoded.size()) {
        throw std::logic_error("Winding hard-sign projection size mismatch");
    }
    std::vector<unsigned char> gauge(decoded.size(), 0);
    for (const std::size_t node : problem.gaugeNodeByComponent)
        gauge[node] = 1;
    std::size_t projected = 0;
    for (std::size_t edgeIndex = 0;
         edgeIndex < problem.edges.size();
         ++edgeIndex) {
        const auto& edge = problem.edges[edgeIndex];
        if (decoded[edge.a].orientation == JointClass::Mixed ||
            decoded[edge.b].orientation == JointClass::Mixed ||
            hardWindingSignCompatible(
                edge,
                predictedDelta(edgeIndex, decoded[edge.a], decoded[edge.b]))) {
            continue;
        }
        std::size_t disable = edge.b;
        if (gauge[edge.a] != 0 && gauge[edge.b] == 0) {
            disable = edge.b;
        } else if (gauge[edge.b] != 0 && gauge[edge.a] == 0) {
            disable = edge.a;
        } else if (activeConfidence[edge.a] < activeConfidence[edge.b]) {
            disable = edge.a;
        } else if (activeConfidence[edge.b] < activeConfidence[edge.a]) {
            disable = edge.b;
        } else {
            disable = std::max(edge.a, edge.b);
        }
        decoded[disable] = {JointClass::Mixed, 0};
        ++projected;
    }
    return projected;
}

struct JointParameters {
    double phase = 0.25;
    double scale = 1.0;
    std::vector<int> componentSign;
};

struct JointAdaptiveRound {
    DiscreteRound discrete;
    std::vector<int> lower;
    std::vector<int> upper;
    std::size_t expansionRounds = 0;
    std::size_t messageIterations = 0;
    std::size_t totalStates = 0;
};

JointAdaptiveRound solveJointAdaptive(
    const PreparedWinding& problem,
    std::vector<int> lower,
    std::vector<int> upper,
    const FiberTraceBeliefPropagationReport& orientationBeliefs,
    std::span<const JointClass> fixedOrientations,
    const JointParameters& parameters,
    const FiberTraceInterleavedWindingConfig& config,
    std::size_t initialization,
    std::size_t initializationCount,
    std::size_t calibrationIteration,
    std::size_t accumulatedMessageIterations,
    const std::chrono::steady_clock::time_point& started,
    const FiberTraceInterleavedWindingProgressCallback& progress)
{
    std::vector<unsigned char> gauge(problem.piecesByNode.size(), 0);
    for (const std::size_t node : problem.integerGaugeNodes)
        gauge[node] = 1;
    for (const std::size_t node : problem.gaugeNodeByComponent)
        gauge[node] = 2;
    JointAdaptiveRound result;
    for (;;) {
        result.totalStates = 0;
        std::vector<std::vector<double>> logUnary(lower.size());
        for (std::size_t node = 0; node < lower.size(); ++node) {
            const std::size_t integers = static_cast<std::size_t>(
                upper[node] - lower[node] + 1);
            const std::size_t stateCount = jointPieceStateCount(
                node, integers, fixedOrientations);
            result.totalStates += stateCount;
            logUnary[node].resize(stateCount, 0.0);
            if (fixedOrientations.empty()) {
                const std::size_t piece = problem.piecesByNode[node].front();
                const std::array orientationPrior{
                    orientationBeliefs.horizontalProbability[piece],
                    orientationBeliefs.mixedProbability[piece],
                    orientationBeliefs.verticalProbability[piece],
                };
                logUnary[node][0] = std::log(
                    std::max(orientationPrior[1], kEpsilon));
                for (std::size_t integer = 0; integer < integers; ++integer) {
                    for (std::size_t orientation = 0; orientation < 2; ++orientation) {
                        const std::size_t prior = orientation == 0 ? 0 : 2;
                        logUnary[node][1 + 2 * integer + orientation] = std::log(
                            std::max(orientationPrior[prior], kEpsilon));
                    }
                }
            } else if (fixedOrientations[node] != JointClass::Mixed) {
                logUnary[node][0] = -config.mixedUnaryCost /
                    config.orientationTemperature;
            }
            if (gauge[node] == 2 && fixedOrientations.empty()) {
                std::fill(
                    logUnary[node].begin(),
                    logUnary[node].end(),
                    -std::numeric_limits<double>::infinity());
                logUnary[node][1] = 0.0;
            }
        }
        if (result.totalStates > config.maximumTotalCandidateStates) {
            throw std::runtime_error(
                "Interleaved winding BP adaptive support exceeded its resource guard");
        }
        result.discrete = solvePairwiseRound(
            problem,
            logUnary,
            config,
            [&](std::size_t edge, std::size_t stateA, std::size_t stateB) {
                const auto& current = problem.edges[edge];
                const auto a = jointState(
                    current.a, stateA, lower[current.a], fixedOrientations);
                const auto b = jointState(
                    current.b, stateB, lower[current.b], fixedOrientations);
                const int sign = parameters.componentSign.at(
                    problem.componentByNode[current.a]);
                return jointLogPotential(
                    current,
                    a,
                    b,
                    sign,
                    parameters.phase,
                    parameters.scale,
                    config.temperature);
            },
            [&](std::size_t messageIteration, double residual) {
                if (!progress)
                    return;
                progress({
                    FiberTraceInterleavedWindingProgressPhase::MessagePassing,
                    initialization,
                    initializationCount,
                    calibrationIteration,
                    config.maximumCalibrationIterations,
                    result.expansionRounds + 1,
                    messageIteration,
                    config.maximumMessageIterations,
                    accumulatedMessageIterations + result.messageIterations +
                        messageIteration,
                    result.totalStates,
                    residual,
                    parameters.phase,
                    parameters.scale,
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started).count(),
                });
            });
        result.messageIterations += result.discrete.iterations;
        ++result.expansionRounds;
        bool expand = false;
        for (std::size_t node = 0; node < lower.size(); ++node) {
            if (gauge[node] != 0)
                continue;
            const auto& probabilities = result.discrete.probabilities[node];
            const std::size_t orientationCount = jointActiveOrientationCount(
                node, fixedOrientations);
            if (orientationCount == 0)
                continue;
            const std::size_t integers =
                (probabilities.size() - 1) / orientationCount;
            std::vector<double> windingProbability(integers, 0.0);
            for (std::size_t integer = 0; integer < integers; ++integer) {
                for (std::size_t orientation = 0;
                     orientation < orientationCount;
                     ++orientation) {
                    windingProbability[integer] += probabilities[
                        1 + orientationCount * integer + orientation];
                }
            }
            const double activeProbability = std::accumulate(
                windingProbability.begin(), windingProbability.end(), 0.0);
            if (!(activeProbability > 0.0))
                continue;
            for (double& probability : windingProbability)
                probability /= activeProbability;
            const std::size_t map = static_cast<std::size_t>(std::distance(
                windingProbability.begin(),
                std::max_element(
                    windingProbability.begin(), windingProbability.end())));
            bool expandLower = map == 0;
            bool expandUpper = map + 1 == integers;
            if (windingProbability.front() + windingProbability.back() >
                config.boundaryProbabilityThreshold) {
                expandLower = expandLower ||
                    windingProbability.front() >= windingProbability.back();
                expandUpper = expandUpper ||
                    windingProbability.back() >= windingProbability.front();
            }
            if (expandLower) {
                --lower[node];
                expand = true;
            }
            if (expandUpper) {
                ++upper[node];
                expand = true;
            }
        }
        if (!expand)
            break;
    }
    result.lower = std::move(lower);
    result.upper = std::move(upper);
    return result;
}

struct PairBelief {
    std::size_t edge = 0;
    JointState a;
    JointState b;
    double probability = 0.0;
};

std::vector<PairBelief> jointPairBeliefs(
    const PreparedWinding& problem,
    const JointAdaptiveRound& round,
    std::span<const JointClass> fixedOrientations,
    const JointParameters& parameters,
    const FiberTraceInterleavedWindingConfig& config)
{
    std::vector<PairBelief> result;
    for (std::size_t edge = 0; edge < problem.edges.size(); ++edge) {
        const auto& current = problem.edges[edge];
        std::vector<double> values;
        values.reserve(
            round.discrete.bToA[edge].size() *
            round.discrete.aToB[edge].size());
        const int sign = parameters.componentSign.at(
            problem.componentByNode[current.a]);
        for (std::size_t stateA = 0;
             stateA < round.discrete.bToA[edge].size();
             ++stateA) {
            const auto a = jointState(
                current.a,
                stateA,
                round.lower[current.a],
                fixedOrientations);
            for (std::size_t stateB = 0;
                 stateB < round.discrete.aToB[edge].size();
                 ++stateB) {
                const auto b = jointState(
                    current.b,
                    stateB,
                    round.lower[current.b],
                    fixedOrientations);
                values.push_back(
                    round.discrete.totals[current.a][stateA] -
                    round.discrete.bToA[edge][stateA] +
                    round.discrete.totals[current.b][stateB] -
                    round.discrete.aToB[edge][stateB] +
                    jointLogPotential(
                        current,
                        a,
                        b,
                        sign,
                        parameters.phase,
                        parameters.scale,
                        config.temperature));
            }
        }
        const double normalization = logSumExp(values);
        std::size_t index = 0;
        for (std::size_t stateA = 0;
             stateA < round.discrete.bToA[edge].size();
             ++stateA) {
            const auto a = jointState(
                current.a,
                stateA,
                round.lower[current.a],
                fixedOrientations);
            for (std::size_t stateB = 0;
                 stateB < round.discrete.aToB[edge].size();
                 ++stateB) {
                const auto b = jointState(
                    current.b,
                    stateB,
                    round.lower[current.b],
                    fixedOrientations);
                const double probability = std::exp(values[index++] - normalization);
                if (probability > 0.0)
                    result.push_back({edge, a, b, probability});
            }
        }
    }
    return result;
}

struct CalibrationTerm {
    double weight = 0.0;
    double integer = 0.0;
    double phaseCoefficient = 0.0;
    std::optional<double> parallelDistance;
    std::optional<double> signedDelta;
};

double calibrationL1(
    const std::vector<CalibrationTerm>& terms,
    double phase,
    double scale)
{
    double result = 0.0;
    for (const auto& term : terms) {
        const double latent =
            term.integer + term.phaseCoefficient * phase;
        if (!term.parallelDistance && term.signedDelta &&
            *term.signedDelta != 0.0 &&
            *term.signedDelta * (latent / scale) <= 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        const double residual = term.parallelDistance
            ? std::abs(latent) - *term.parallelDistance
            : latent / scale - *term.signedDelta;
        result += term.weight * std::abs(residual);
    }
    return result;
}

struct BoundedGainPhaseFit {
    double gain = 1.0;
    double gainPhase = 0.25;
    bool rankDeficient = false;
};

BoundedGainPhaseFit fitBoundedGainPhase(
    double h00,
    double h01,
    double h11,
    double rhs0,
    double rhs1,
    double minimumGain,
    double maximumGain,
    double previousGain,
    double previousGainPhase)
{
    const double determinant = h00 * h11 - h01 * h01;
    const double matrixScale = std::max(1.0, h00 * h11);
    if (!(determinant > 1.0e-10 * matrixScale)) {
        return {previousGain, previousGainPhase, true};
    }

    const auto error = [&](double gain, double gainPhase) {
        return h00 * gain * gain + 2.0 * h01 * gain * gainPhase +
            h11 * gainPhase * gainPhase - 2.0 * rhs0 * gain -
            2.0 * rhs1 * gainPhase;
    };
    BoundedGainPhaseFit best{
        previousGain, previousGainPhase, false};
    double bestError = error(best.gain, best.gainPhase);
    auto consider = [&](double gain, double gainPhase) {
        gain = std::clamp(gain, minimumGain, maximumGain);
        gainPhase = std::clamp(gainPhase, 0.0, 0.5 * gain);
        const double candidateError = error(gain, gainPhase);
        if (candidateError < bestError) {
            best = {gain, gainPhase, false};
            bestError = candidateError;
        }
    };

    const double unconstrainedGain =
        (rhs0 * h11 - rhs1 * h01) / determinant;
    const double unconstrainedGainPhase =
        (h00 * rhs1 - h01 * rhs0) / determinant;
    if (unconstrainedGain >= minimumGain &&
        unconstrainedGain <= maximumGain &&
        unconstrainedGainPhase >= 0.0 &&
        unconstrainedGainPhase <= 0.5 * unconstrainedGain) {
        consider(unconstrainedGain, unconstrainedGainPhase);
    }

    for (const double gain : {minimumGain, maximumGain}) {
        const double gainPhase = h11 > kEpsilon
            ? (rhs1 - h01 * gain) / h11
            : 0.0;
        consider(gain, gainPhase);
    }
    if (h00 > kEpsilon)
        consider(rhs0 / h00, 0.0);
    const double halfHessian = h00 + h01 + 0.25 * h11;
    if (halfHessian > kEpsilon) {
        const double gain = (rhs0 + 0.5 * rhs1) / halfHessian;
        consider(gain, 0.5 * gain);
    }
    for (const double gain : {minimumGain, maximumGain}) {
        consider(gain, 0.0);
        consider(gain, 0.5 * gain);
    }
    return best;
}

struct CalibrationUpdate {
    double phase = 0.0;
    double scale = 1.0;
    std::vector<int> componentSign;
    bool rankDeficient = false;
};

CalibrationUpdate updateCalibration(
    const PreparedWinding& problem,
    const std::vector<PairBelief>& beliefs,
    const JointParameters& current,
    const FiberTraceInterleavedWindingConfig& config)
{
    CalibrationUpdate result{
        current.phase, current.scale, current.componentSign, false};
    std::vector<std::array<double, 2>> signLoss(
        problem.gaugeNodeByComponent.size(), {0.0, 0.0});
    for (const auto& belief : beliefs) {
        if (belief.a.orientation == JointClass::Mixed ||
            belief.b.orientation == JointClass::Mixed) {
            continue;
        }
        const auto& edge = problem.edges[belief.edge];
        const std::size_t component = problem.componentByNode[edge.a];
        for (int signIndex = 0; signIndex < 2; ++signIndex) {
            const int sign = signIndex == 0 ? 1 : -1;
            signLoss[component][signIndex] += belief.probability * windingEnergy(
                edge,
                belief.a,
                belief.b,
                sign,
                current.phase,
                current.scale);
        }
    }
    for (std::size_t component = 0; component < signLoss.size(); ++component) {
        result.componentSign[component] =
            signLoss[component][1] < signLoss[component][0] ? -1 : 1;
    }

    std::vector<CalibrationTerm> terms;
    double h00 = 0.0;
    double h01 = 0.0;
    double h11 = 0.0;
    double rhs0 = 0.0;
    double rhs1 = 0.0;
    for (const auto& belief : beliefs) {
        if (belief.a.orientation == JointClass::Mixed ||
            belief.b.orientation == JointClass::Mixed) {
            continue;
        }
        const auto& edge = problem.edges[belief.edge];
        const int sign = result.componentSign.at(problem.componentByNode[edge.a]);
        const double integer = static_cast<double>(
            belief.b.winding - belief.a.winding);
        const double phaseCoefficient = static_cast<double>(sign) *
            ((belief.b.orientation == JointClass::B ? 1.0 : 0.0) -
             (belief.a.orientation == JointClass::B ? 1.0 : 0.0));
        const auto addParallel = [&](double weight, double distance) {
            if (!(weight > 0.0))
                return;
            terms.push_back({
                belief.probability * weight,
                integer,
                phaseCoefficient,
                distance,
                std::nullopt,
            });
        };
        const auto addSigned = [&](double weight, double signedDelta) {
            if (!(weight > 0.0))
                return;
            terms.push_back({
                belief.probability * weight,
                integer,
                phaseCoefficient,
                std::nullopt,
                signedDelta,
            });
            const double w = belief.probability * weight;
            h00 += w * integer * integer;
            h01 += w * integer * phaseCoefficient;
            h11 += w * phaseCoefficient * phaseCoefficient;
            rhs0 += w * integer * signedDelta;
            rhs1 += w * phaseCoefficient * signedDelta;
        };
        for (const auto& measurement : edge.measurements) {
            addParallel(
                parallelWindingWeight(measurement),
                measurement.parallelDistance);
            if (measurement.perpendicularSignedDelta)
                addSigned(
                    perpendicularWindingWeight(measurement),
                    *measurement.perpendicularSignedDelta);
        }
    }
    const double previousGain = 1.0 / current.scale;
    const auto proposed = fitBoundedGainPhase(
        h00,
        h01,
        h11,
        rhs0,
        rhs1,
        1.0 / config.maximumMeasurementScale,
        1.0 / config.minimumMeasurementScale,
        previousGain,
        previousGain * current.phase);
    if (proposed.rankDeficient) {
        result.rankDeficient = true;
        return result;
    }
    const double proposedPhase = proposed.gainPhase / proposed.gain;
    const double proposedScale = 1.0 / proposed.gain;
    const double oldLoss = calibrationL1(terms, current.phase, current.scale);
    for (double step = 1.0; step >= 1.0 / 1024.0; step *= 0.5) {
        const double phase = current.phase + step * (proposedPhase - current.phase);
        const double measurementScale =
            current.scale + step * (proposedScale - current.scale);
        if (calibrationL1(terms, phase, measurementScale) <=
            oldLoss + 1.0e-12) {
            result.phase = phase;
            result.scale = measurementScale;
            break;
        }
    }
    return result;
}

double decodedJointEnergy(
    const PreparedWinding& problem,
    const JointAdaptiveRound& round,
    const FiberTraceBeliefPropagationReport& orientationBeliefs,
    std::span<const JointClass> fixedOrientations,
    const JointParameters& parameters,
    const FiberTraceInterleavedWindingConfig& config)
{
    std::vector<JointState> decoded(problem.piecesByNode.size());
    std::vector<double> activeConfidence(decoded.size(), 0.0);
    for (std::size_t node = 0; node < decoded.size(); ++node) {
        const auto& probabilities = round.discrete.probabilities[node];
        std::array<double, 3> classProbability{};
        for (std::size_t state = 0; state < probabilities.size(); ++state) {
            const auto current = jointState(
                node, state, round.lower[node], fixedOrientations);
            classProbability[static_cast<std::size_t>(current.orientation)] +=
                probabilities[state];
        }
        const JointClass finalClass =
            classProbability[1] >= classProbability[0] &&
                classProbability[1] >= classProbability[2]
            ? JointClass::Mixed
            : classProbability[0] > classProbability[2]
                ? JointClass::A
                : JointClass::B;
        activeConfidence[node] =
            std::max(classProbability[0], classProbability[2]) -
            classProbability[1];
        double maximum = -1.0;
        for (std::size_t state = 0; state < probabilities.size(); ++state) {
            const auto current = jointState(
                node, state, round.lower[node], fixedOrientations);
            if (current.orientation == finalClass &&
                probabilities[state] > maximum) {
                maximum = probabilities[state];
                decoded[node] = current;
            }
        }
    }
    projectDecodedHardSigns(
        problem,
        decoded,
        activeConfidence,
        [&](std::size_t edgeIndex, const JointState& a, const JointState& b) {
            const auto& edge = problem.edges[edgeIndex];
            const int sign = parameters.componentSign.at(
                problem.componentByNode[edge.a]);
            const double latent = static_cast<double>(b.winding - a.winding) +
                classOffset(b.orientation, sign, parameters.phase) -
                classOffset(a.orientation, sign, parameters.phase);
            return latent / parameters.scale;
        });
    double energy = 0.0;
    for (std::size_t node = 0; node < decoded.size(); ++node) {
        if (fixedOrientations.empty()) {
            const std::size_t piece = problem.piecesByNode[node].front();
            const double prior = decoded[node].orientation == JointClass::A
                ? orientationBeliefs.horizontalProbability[piece]
                : decoded[node].orientation == JointClass::Mixed
                    ? orientationBeliefs.mixedProbability[piece]
                    : orientationBeliefs.verticalProbability[piece];
            energy -= std::log(std::max(prior, kEpsilon));
        } else if (decoded[node].orientation == JointClass::Mixed &&
                   fixedOrientations[node] != JointClass::Mixed) {
            energy += config.mixedUnaryCost / config.orientationTemperature;
        }
    }
    for (std::size_t edge = 0; edge < problem.edges.size(); ++edge) {
        const auto& current = problem.edges[edge];
        const int sign = parameters.componentSign.at(
            problem.componentByNode[current.a]);
        const double logPotential = jointLogPotential(
            current,
            decoded[current.a],
            decoded[current.b],
            sign,
            parameters.phase,
            parameters.scale,
            config.temperature);
        energy -= fixedOrientations.empty()
            ? config.temperature * logPotential
            : logPotential;
    }
    return energy;
}

struct JointCandidate {
    JointAdaptiveRound round;
    JointParameters parameters;
    std::size_t calibrationIterations = 0;
    std::size_t rankDeficientUpdates = 0;
    std::size_t initialization = 0;
    std::size_t totalMessageIterations = 0;
    std::size_t totalExpansionRounds = 0;
    bool calibrationConverged = false;
    double decodedEnergy = std::numeric_limits<double>::infinity();
};

struct GridCalibrationCell {
    int gainIndex = 0;
    std::size_t phaseIndex = 0;
    double gain = 1.0;
    double phase = 0.0;
};

struct GridMessages {
    std::vector<double> toA;
    std::vector<double> toB;
    std::vector<double> toCalibration;
    std::array<double, 2> toSign{0.0, 0.0};
};

struct GridRound {
    std::vector<GridMessages> messages;
    std::vector<std::vector<double>> pieceProbabilities;
    std::vector<double> calibrationProbabilities;
    std::vector<std::array<double, 2>> signProbabilities;
    std::vector<int> lower;
    std::vector<int> upper;
    std::vector<unsigned char> gauge;
    std::vector<JointClass> fixedOrientations;
    std::vector<GridCalibrationCell> calibrationCells;
    GridCalibrationCell fixedCalibrationParameters;
    std::size_t iterations = 0;
    std::size_t gridShifts = 0;
    std::size_t supportChanges = 0;
    std::size_t totalStates = 0;
    std::size_t effectiveWorkers = 1;
    double residual = 0.0;
    double calibrationResidual = 0.0;
    double lowerBoundaryProbability = 0.0;
    double upperBoundaryProbability = 0.0;
    double fixedMeasurementScale = 1.0;
    bool fixedCalibration = false;
    bool converged = false;
};

double logAddExp(double a, double b)
{
    if (!std::isfinite(a))
        return b;
    if (!std::isfinite(b))
        return a;
    const double maximum = std::max(a, b);
    return maximum + std::log(std::exp(a - maximum) + std::exp(b - maximum));
}

std::vector<double> normalizedProbabilities(const std::vector<double>& values)
{
    const double normalization = logSumExp(values);
    if (!std::isfinite(normalization))
        throw std::runtime_error("Joint-grid BP produced an invalid marginal");
    std::vector<double> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index)
        result[index] = std::exp(values[index] - normalization);
    return result;
}

std::vector<GridCalibrationCell> makeCalibrationCells(
    int minimumGainIndex,
    int maximumGainIndex,
    const FiberTraceJointGridWindingConfig& config)
{
    std::vector<GridCalibrationCell> result;
    result.reserve(
        static_cast<std::size_t>(maximumGainIndex - minimumGainIndex + 1) *
        config.phaseCells);
    for (int gainIndex = minimumGainIndex;
         gainIndex <= maximumGainIndex;
         ++gainIndex) {
        const double gain = std::exp(
            static_cast<double>(gainIndex) * config.logGainStep);
        for (std::size_t phaseIndex = 0;
             phaseIndex < config.phaseCells;
             ++phaseIndex) {
            const double phase = 0.5 * static_cast<double>(phaseIndex) /
                static_cast<double>(config.phaseCells - 1);
            result.push_back({gainIndex, phaseIndex, gain, phase});
        }
    }
    return result;
}

bool hasFixedCalibration(const FiberTraceJointGridWindingConfig& config)
{
    return config.fixedPhaseMagnitude.has_value() &&
        config.fixedMeasurementScale.has_value();
}

GridCalibrationCell fixedCalibrationCell(
    const FiberTraceJointGridWindingConfig& config)
{
    return {
        0,
        0,
        1.0 / *config.fixedMeasurementScale,
        *config.fixedPhaseMagnitude,
    };
}

const GridCalibrationCell& activeCalibrationCell(
    const GridRound& round,
    std::size_t index)
{
    if (round.fixedCalibration) {
        if (index != 0)
            throw std::out_of_range("Fixed winding calibration state is invalid");
        return round.fixedCalibrationParameters;
    }
    return round.calibrationCells.at(index);
}

std::size_t gridPieceStateCount(
    std::size_t node,
    const std::vector<int>& lower,
    const std::vector<int>& upper,
    const std::vector<unsigned char>& gauge,
    std::span<const JointClass> fixedOrientations)
{
    if (gauge[node] == 2 && fixedOrientations.empty())
        return 1;
    if (gauge[node] != 0)
        return jointPieceStateCount(node, 1, fixedOrientations);
    return jointPieceStateCount(
        node,
        static_cast<std::size_t>(upper[node] - lower[node] + 1),
        fixedOrientations);
}

JointState gridPieceState(
    std::size_t node,
    std::size_t state,
    const std::vector<int>& lower,
    const std::vector<unsigned char>& gauge,
    std::span<const JointClass> fixedOrientations)
{
    if (gauge[node] != 0) {
        const std::size_t stateCount =
            gauge[node] == 2 && fixedOrientations.empty()
            ? 1
            : jointPieceStateCount(node, 1, fixedOrientations);
        if (state >= stateCount)
            throw std::out_of_range("Joint-grid gauge state is invalid");
        if (gauge[node] == 2 && fixedOrientations.empty())
            return {JointClass::A, 0};
        return jointState(node, state, 0, fixedOrientations);
    }
    return jointState(node, state, lower[node], fixedOrientations);
}

double gridWindingEnergy(
    const Edge& edge,
    const JointState& a,
    const JointState& b,
    int sign,
    double gain,
    double phase)
{
    const double delta = static_cast<double>(b.winding - a.winding) +
        classOffset(b.orientation, sign, phase) -
        classOffset(a.orientation, sign, phase);
    const double predictedDelta = gain * delta;
    if (!hardWindingSignCompatible(edge, predictedDelta))
        return std::numeric_limits<double>::infinity();
    double result = 0.0;
    for (const auto& measurement : edge.measurements) {
        result += parallelWindingWeight(measurement) *
            std::abs(std::abs(delta) - measurement.parallelDistance);
        if (measurement.perpendicularSignedDelta) {
            result += perpendicularWindingWeight(measurement) *
                std::abs(
                    predictedDelta - *measurement.perpendicularSignedDelta);
        }
    }
    return result;
}

double gridOrientationEnergy(
    const Edge& edge,
    JointClass a,
    JointClass b)
{
    const bool same = a == b;
    double result = 0.0;
    for (const auto& measurement : edge.measurements) {
        result += same ? measurement.perpendicular : measurement.parallel;
    }
    return result;
}

double gridLogPotential(
    const Edge& edge,
    JointState a,
    JointState b,
    int sign,
    const GridCalibrationCell& calibration,
    const FiberTraceJointGridWindingConfig& config,
    bool includeOrientationEnergy)
{
    if (a.orientation == JointClass::Mixed ||
        b.orientation == JointClass::Mixed)
        return 0.0;
    const double windingEnergy = gridWindingEnergy(
        edge,
        a,
        b,
        sign,
        calibration.gain,
        calibration.phase);
    double result = -windingEnergy / config.temperature;
    if (includeOrientationEnergy) {
        result -= gridOrientationEnergy(
            edge, a.orientation, b.orientation) /
            config.orientationTemperature;
    }
    return result;
}

double dampMessage(
    std::array<double, 2>& target,
    const std::array<double, 2>& current,
    double damping)
{
    const double normalization = logAddExp(target[0], target[1]);
    if (!std::isfinite(normalization))
        throw std::runtime_error("Joint-grid BP sign message has no finite state");
    double residual = 0.0;
    for (std::size_t state = 0; state < 2; ++state) {
        target[state] -= normalization;
        const bool targetImpossible = isNegativeInfinity(target[state]);
        const bool currentImpossible = isNegativeInfinity(current[state]);
        if (targetImpossible || currentImpossible) {
            if (targetImpossible != currentImpossible)
                residual = std::numeric_limits<double>::infinity();
            if (targetImpossible)
                target[state] = -std::numeric_limits<double>::infinity();
            continue;
        }
        const double damped = current[state] +
            damping * (target[state] - current[state]);
        residual = std::max(residual, std::abs(damped - current[state]));
        target[state] = damped;
    }
    const double dampedNormalization = logAddExp(target[0], target[1]);
    target[0] -= dampedNormalization;
    target[1] -= dampedNormalization;
    return residual;
}

struct GridLogTotals {
    std::vector<std::vector<LogProductAccumulator>> pieceAccumulators;
    std::vector<LogProductAccumulator> calibrationAccumulators;
    std::vector<std::array<LogProductAccumulator, 2>> signAccumulators;
    std::vector<std::vector<double>> pieceValues;
    std::vector<double> calibrationValues;
    std::vector<std::array<double, 2>> signValues;
};

void validateJointGridConfig(const FiberTraceJointGridWindingConfig& config)
{
    validateConfig(config);
    const bool hasFixedPhase = config.fixedPhaseMagnitude.has_value();
    const bool hasFixedScale = config.fixedMeasurementScale.has_value();
    const bool fixedInvalid = hasFixedPhase != hasFixedScale ||
        (hasFixedPhase &&
         (!std::isfinite(*config.fixedPhaseMagnitude) ||
          *config.fixedPhaseMagnitude < 0.0 ||
          *config.fixedPhaseMagnitude > 0.5)) ||
        (hasFixedScale &&
         (!std::isfinite(*config.fixedMeasurementScale) ||
          !(*config.fixedMeasurementScale > 0.0)));
    const bool adaptiveInvalid = !hasFixedPhase &&
        (!std::isfinite(config.logGainStep) || !(config.logGainStep > 0.0) ||
         !std::isfinite(config.calibrationBoundaryProbabilityThreshold) ||
         !(config.calibrationBoundaryProbabilityThreshold > 0.0) ||
         config.calibrationBoundaryProbabilityThreshold >= 1.0 ||
         !std::isfinite(config.calibrationDiscardProbabilityThreshold) ||
         config.calibrationDiscardProbabilityThreshold < 0.0 ||
         config.calibrationDiscardProbabilityThreshold >=
             config.calibrationBoundaryProbabilityThreshold ||
         !std::isfinite(config.calibrationPosteriorTolerance) ||
         config.calibrationPosteriorTolerance < 0.0 ||
         config.initialGainCells == 0 || config.initialGainCells % 2 == 0 ||
         config.phaseCells < 2 ||
         config.maximumGainCells < config.initialGainCells);
    if (!std::isfinite(config.mixedUnaryCost) || config.mixedUnaryCost < 0.0 ||
        !std::isfinite(config.orientationTemperature) ||
        !(config.orientationTemperature > 0.0) ||
        fixedInvalid || adaptiveInvalid ||
        config.stableIterations == 0) {
        throw std::invalid_argument("Joint-grid winding BP config is invalid");
    }
}

std::size_t jointGridStateCount(
    const PreparedWinding& problem,
    const GridRound& round)
{
    std::size_t result = (round.fixedCalibration ? 0 : round.calibrationCells.size()) +
        2 * problem.gaugeNodeByComponent.size();
    for (std::size_t node = 0; node < problem.piecesByNode.size(); ++node)
        result += gridPieceStateCount(
            node,
            round.lower,
            round.upper,
            round.gauge,
            round.fixedOrientations);
    for (const auto& edge : problem.edges) {
        result += gridPieceStateCount(
            edge.a,
            round.lower,
            round.upper,
            round.gauge,
            round.fixedOrientations);
        result += gridPieceStateCount(
            edge.b,
            round.lower,
            round.upper,
            round.gauge,
            round.fixedOrientations);
        result += (round.fixedCalibration ? 0 : round.calibrationCells.size()) + 2;
    }
    return result;
}

void initializeGridMessages(
    const PreparedWinding& problem,
    GridRound& round)
{
    round.messages.resize(problem.edges.size());
    for (std::size_t edgeIndex = 0;
         edgeIndex < problem.edges.size();
         ++edgeIndex) {
        const auto& edge = problem.edges[edgeIndex];
        auto& message = round.messages[edgeIndex];
        message.toA.assign(
            gridPieceStateCount(
                edge.a,
                round.lower,
                round.upper,
                round.gauge,
                round.fixedOrientations),
            0.0);
        message.toB.assign(
            gridPieceStateCount(
                edge.b,
                round.lower,
                round.upper,
                round.gauge,
                round.fixedOrientations),
            0.0);
        if (!round.fixedCalibration)
            message.toCalibration.assign(round.calibrationCells.size(), 0.0);
    }
}

GridLogTotals buildGridTotals(
    const PreparedWinding& problem,
    const GridRound& round,
    const FiberTraceJointGridWindingConfig& config)
{
    GridLogTotals totals;
    totals.pieceAccumulators.resize(problem.piecesByNode.size());
    totals.pieceValues.resize(problem.piecesByNode.size());
    for (std::size_t node = 0; node < problem.piecesByNode.size(); ++node) {
        const std::size_t stateCount = gridPieceStateCount(
            node,
            round.lower,
            round.upper,
            round.gauge,
            round.fixedOrientations);
        totals.pieceAccumulators[node].resize(stateCount);
        totals.pieceValues[node].resize(stateCount);
        for (std::size_t state = 0; state < stateCount; ++state) {
            const auto decoded = gridPieceState(
                node,
                state,
                round.lower,
                round.gauge,
                round.fixedOrientations);
            const bool chargeDefect =
                decoded.orientation == JointClass::Mixed &&
                (round.fixedOrientations.empty() ||
                 round.fixedOrientations[node] != JointClass::Mixed);
            addLogFactor(totals.pieceAccumulators[node][state], chargeDefect
                ? -config.mixedUnaryCost / config.orientationTemperature
                : 0.0);
        }
    }
    if (!round.fixedCalibration) {
        totals.calibrationAccumulators.resize(round.calibrationCells.size());
        totals.calibrationValues.resize(round.calibrationCells.size());
    }
    totals.signAccumulators.resize(problem.gaugeNodeByComponent.size());
    totals.signValues.resize(problem.gaugeNodeByComponent.size());
    for (std::size_t edgeIndex = 0;
         edgeIndex < problem.edges.size();
         ++edgeIndex) {
        const auto& edge = problem.edges[edgeIndex];
        const auto& message = round.messages[edgeIndex];
        for (std::size_t state = 0; state < message.toA.size(); ++state)
            addLogFactor(
                totals.pieceAccumulators[edge.a][state], message.toA[state]);
        for (std::size_t state = 0; state < message.toB.size(); ++state)
            addLogFactor(
                totals.pieceAccumulators[edge.b][state], message.toB[state]);
        if (!round.fixedCalibration) {
            for (std::size_t state = 0;
                 state < message.toCalibration.size();
                 ++state) {
                addLogFactor(
                    totals.calibrationAccumulators[state],
                    message.toCalibration[state]);
            }
        }
        const std::size_t component = problem.componentByNode[edge.a];
        addLogFactor(totals.signAccumulators[component][0], message.toSign[0]);
        addLogFactor(totals.signAccumulators[component][1], message.toSign[1]);
    }
    for (std::size_t node = 0; node < totals.pieceValues.size(); ++node) {
        for (std::size_t state = 0;
             state < totals.pieceValues[node].size();
             ++state) {
            totals.pieceValues[node][state] =
                logProductValue(totals.pieceAccumulators[node][state]);
        }
    }
    for (std::size_t state = 0;
         state < totals.calibrationValues.size();
         ++state) {
        totals.calibrationValues[state] =
            logProductValue(totals.calibrationAccumulators[state]);
    }
    for (std::size_t component = 0;
         component < totals.signValues.size();
         ++component) {
        for (std::size_t state = 0; state < 2; ++state) {
            totals.signValues[component][state] =
                logProductValue(totals.signAccumulators[component][state]);
        }
    }
    return totals;
}

double updateGridFactor(
    const Edge& edge,
    const GridMessages& current,
    GridMessages& next,
    const std::vector<LogProductAccumulator>& totalA,
    const std::vector<LogProductAccumulator>& totalB,
    const std::vector<LogProductAccumulator>& calibrationTotal,
    const std::array<LogProductAccumulator, 2>& signTotal,
    const GridRound& round,
    const FiberTraceJointGridWindingConfig& config)
{
    std::vector<double> cavityA(totalA.size());
    std::vector<double> cavityB(totalB.size());
    std::vector<double> cavityCalibration;
    if (!round.fixedCalibration)
        cavityCalibration.resize(calibrationTotal.size());
    std::array<double, 2> cavitySign{};
    for (std::size_t state = 0; state < cavityA.size(); ++state)
        cavityA[state] = logCavityValue(totalA[state], current.toA[state]);
    for (std::size_t state = 0; state < cavityB.size(); ++state)
        cavityB[state] = logCavityValue(totalB[state], current.toB[state]);
    for (std::size_t state = 0; state < cavityCalibration.size(); ++state) {
        cavityCalibration[state] =
            logCavityValue(
                calibrationTotal[state], current.toCalibration[state]);
    }
    cavitySign[0] = logCavityValue(signTotal[0], current.toSign[0]);
    cavitySign[1] = logCavityValue(signTotal[1], current.toSign[1]);

    next.toA.assign(cavityA.size(), -std::numeric_limits<double>::infinity());
    next.toB.assign(cavityB.size(), -std::numeric_limits<double>::infinity());
    if (round.fixedCalibration) {
        next.toCalibration.clear();
    } else {
        next.toCalibration.assign(
            cavityCalibration.size(), -std::numeric_limits<double>::infinity());
    }
    next.toSign = {
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    for (std::size_t stateA = 0; stateA < cavityA.size(); ++stateA) {
        const auto a = gridPieceState(
            edge.a,
            stateA,
            round.lower,
            round.gauge,
            round.fixedOrientations);
        for (std::size_t stateB = 0; stateB < cavityB.size(); ++stateB) {
            const auto b = gridPieceState(
                edge.b,
                stateB,
                round.lower,
                round.gauge,
                round.fixedOrientations);
            const std::size_t calibrationCount = round.fixedCalibration
                ? 1
                : cavityCalibration.size();
            for (std::size_t calibration = 0;
                 calibration < calibrationCount;
                 ++calibration) {
                for (std::size_t signState = 0; signState < 2; ++signState) {
                    const int sign = signState == 0 ? 1 : -1;
                    const double calibrationCavity = round.fixedCalibration
                        ? 0.0
                        : cavityCalibration[calibration];
                    const double logPotential = gridLogPotential(
                        edge,
                        a,
                        b,
                        sign,
                        activeCalibrationCell(round, calibration),
                        config,
                        round.fixedOrientations.empty());
                    next.toA[stateA] = logAddExp(
                        next.toA[stateA],
                        cavityB[stateB] + calibrationCavity +
                            cavitySign[signState] + logPotential);
                    next.toB[stateB] = logAddExp(
                        next.toB[stateB],
                        cavityA[stateA] + calibrationCavity +
                            cavitySign[signState] + logPotential);
                    if (!round.fixedCalibration) {
                        next.toCalibration[calibration] = logAddExp(
                            next.toCalibration[calibration],
                            cavityA[stateA] + cavityB[stateB] +
                                cavitySign[signState] + logPotential);
                    }
                    next.toSign[signState] = logAddExp(
                        next.toSign[signState],
                        cavityA[stateA] + cavityB[stateB] +
                            calibrationCavity + logPotential);
                }
            }
        }
    }
    double residual = 0.0;
    residual = std::max(
        residual,
        dampMessage(next.toA, current.toA, config.messageDamping));
    residual = std::max(
        residual,
        dampMessage(next.toB, current.toB, config.messageDamping));
    if (!round.fixedCalibration) {
        residual = std::max(
            residual,
            dampMessage(
                next.toCalibration,
                current.toCalibration,
                config.messageDamping));
    }
    residual = std::max(
        residual,
        dampMessage(next.toSign, current.toSign, config.messageDamping));
    return residual;
}

void remapCalibrationMessages(
    GridRound& round,
    const std::vector<GridCalibrationCell>& cells)
{
    std::map<std::pair<int, std::size_t>, std::size_t> oldIndex;
    for (std::size_t index = 0; index < round.calibrationCells.size(); ++index) {
        oldIndex.emplace(
            std::pair{
                round.calibrationCells[index].gainIndex,
                round.calibrationCells[index].phaseIndex},
            index);
    }
    for (auto& message : round.messages) {
        std::vector<double> remapped(cells.size(), 0.0);
        for (std::size_t index = 0; index < cells.size(); ++index) {
            const auto found = oldIndex.find(
                {cells[index].gainIndex, cells[index].phaseIndex});
            if (found != oldIndex.end())
                remapped[index] = message.toCalibration[found->second];
        }
        normalizeLogVector(remapped);
        message.toCalibration = std::move(remapped);
    }
    round.calibrationCells = cells;
}

void remapPieceMessages(
    const PreparedWinding& problem,
    GridRound& round,
    std::size_t node,
    int oldLower,
    int oldUpper)
{
    const int newLower = round.lower[node];
    const int newUpper = round.upper[node];
    if (oldLower == newLower && oldUpper == newUpper)
        return;
    for (const std::size_t edgeIndex : problem.adjacency[node]) {
        const auto& edge = problem.edges[edgeIndex];
        auto& message = edge.a == node
            ? round.messages[edgeIndex].toA
            : round.messages[edgeIndex].toB;
        const std::size_t orientationCount = jointActiveOrientationCount(
            node, round.fixedOrientations);
        std::vector<double> remapped(
            jointPieceStateCount(
                node,
                static_cast<std::size_t>(newUpper - newLower + 1),
                round.fixedOrientations),
            0.0);
        remapped[0] = message[0];
        for (int winding = oldLower; winding <= oldUpper; ++winding) {
            for (std::size_t orientation = 0;
                 orientation < orientationCount;
                 ++orientation) {
                const std::size_t oldState =
                    1 + orientationCount *
                        static_cast<std::size_t>(winding - oldLower) +
                    orientation;
                const std::size_t newState =
                    1 + orientationCount *
                        static_cast<std::size_t>(winding - newLower) +
                    orientation;
                remapped[newState] = message[oldState];
            }
        }
        normalizeLogVector(remapped);
        message = std::move(remapped);
    }
}

bool ensureIntegerSupport(
    const PreparedWinding& problem,
    GridRound& round,
    std::span<const double> continuousNodes)
{
    double minimumGain = std::numeric_limits<double>::infinity();
    double maximumGain = 0.0;
    if (round.fixedCalibration) {
        minimumGain = round.fixedCalibrationParameters.gain;
        maximumGain = round.fixedCalibrationParameters.gain;
    } else {
        for (const auto& cell : round.calibrationCells) {
            minimumGain = std::min(minimumGain, cell.gain);
            maximumGain = std::max(maximumGain, cell.gain);
        }
    }
    bool changed = false;
    for (std::size_t node = 0; node < problem.piecesByNode.size(); ++node) {
        if (round.gauge[node] != 0)
            continue;
        if (jointActiveOrientationCount(node, round.fixedOrientations) == 0)
            continue;
        const int oldLower = round.lower[node];
        const int oldUpper = round.upper[node];
        const double low = std::min(
            continuousNodes[node] / minimumGain,
            continuousNodes[node] / maximumGain);
        const double high = std::max(
            continuousNodes[node] / minimumGain,
            continuousNodes[node] / maximumGain);
        round.lower[node] = std::min(
            round.lower[node], static_cast<int>(std::floor(low)) - 1);
        round.upper[node] = std::max(
            round.upper[node], static_cast<int>(std::ceil(high)) + 1);
        if (oldLower != round.lower[node] || oldUpper != round.upper[node]) {
            remapPieceMessages(problem, round, node, oldLower, oldUpper);
            changed = true;
        }
    }
    return changed;
}

bool adjustIntegerSupport(
    const PreparedWinding& problem,
    GridRound& round,
    const FiberTraceJointGridWindingConfig& config)
{
    bool changed = false;
    for (std::size_t node = 0; node < problem.piecesByNode.size(); ++node) {
        if (round.gauge[node] != 0)
            continue;
        const auto& probabilities = round.pieceProbabilities[node];
        const std::size_t orientationCount = jointActiveOrientationCount(
            node, round.fixedOrientations);
        if (orientationCount == 0)
            continue;
        const std::size_t integers =
            (probabilities.size() - 1) / orientationCount;
        const double lowerProbability = std::accumulate(
            probabilities.begin() + 1,
            probabilities.begin() + 1 +
                static_cast<std::ptrdiff_t>(orientationCount),
            0.0);
        const double upperProbability = std::accumulate(
            probabilities.begin() + 1 +
                static_cast<std::ptrdiff_t>(orientationCount * (integers - 1)),
            probabilities.end(),
            0.0);
        double meanWinding = 0.0;
        double activeProbability = 0.0;
        for (std::size_t state = 1; state < probabilities.size(); ++state) {
            activeProbability += probabilities[state];
            meanWinding += probabilities[state] * static_cast<double>(
                gridPieceState(
                    node,
                    state,
                    round.lower,
                    round.gauge,
                    round.fixedOrientations).winding);
        }
        if (!(activeProbability > 0.0))
            continue;
        meanWinding /= activeProbability;
        const double normalizedLowerProbability =
            lowerProbability / activeProbability;
        const double normalizedUpperProbability =
            upperProbability / activeProbability;
        const int oldLower = round.lower[node];
        const int oldUpper = round.upper[node];
        if (normalizedLowerProbability > config.boundaryProbabilityThreshold &&
            meanWinding - static_cast<double>(oldLower) <= 0.75) {
            --round.lower[node];
        }
        if (normalizedUpperProbability > config.boundaryProbabilityThreshold &&
            static_cast<double>(oldUpper) - meanWinding <= 0.75) {
            ++round.upper[node];
        }
        if (oldLower != round.lower[node] || oldUpper != round.upper[node]) {
            remapPieceMessages(problem, round, node, oldLower, oldUpper);
            changed = true;
        }
    }
    if (changed)
        ++round.supportChanges;
    return changed;
}

void updateGridMarginals(
    const PreparedWinding& problem,
    GridRound& round,
    const FiberTraceJointGridWindingConfig& config)
{
    const auto totals = buildGridTotals(problem, round, config);
    round.pieceProbabilities.resize(totals.pieceValues.size());
    for (std::size_t node = 0; node < totals.pieceValues.size(); ++node) {
        round.pieceProbabilities[node] =
            normalizedProbabilities(totals.pieceValues[node]);
    }
    if (round.fixedCalibration)
        round.calibrationProbabilities.clear();
    else
        round.calibrationProbabilities =
            normalizedProbabilities(totals.calibrationValues);
    round.signProbabilities.resize(totals.signValues.size());
    for (std::size_t component = 0;
         component < totals.signValues.size();
         ++component) {
        const double normalization = logAddExp(
            totals.signValues[component][0], totals.signValues[component][1]);
        round.signProbabilities[component] = {
            std::exp(totals.signValues[component][0] - normalization),
            std::exp(totals.signValues[component][1] - normalization),
        };
    }
    round.lowerBoundaryProbability = 0.0;
    round.upperBoundaryProbability = 0.0;
    if (!round.fixedCalibration) {
        const int minimumIndex = round.calibrationCells.front().gainIndex;
        const int maximumIndex = round.calibrationCells.back().gainIndex;
        for (std::size_t index = 0;
             index < round.calibrationCells.size();
             ++index) {
            if (round.calibrationCells[index].gainIndex == minimumIndex) {
                round.lowerBoundaryProbability +=
                    round.calibrationProbabilities[index];
            }
            if (round.calibrationCells[index].gainIndex == maximumIndex) {
                round.upperBoundaryProbability +=
                    round.calibrationProbabilities[index];
            }
        }
    }
}

bool adjustCalibrationSupport(
    GridRound& round,
    const FiberTraceJointGridWindingConfig& config)
{
    const bool lowerPressure = round.lowerBoundaryProbability >
        config.calibrationBoundaryProbabilityThreshold;
    const bool upperPressure = round.upperBoundaryProbability >
        config.calibrationBoundaryProbabilityThreshold;
    if (!lowerPressure && !upperPressure)
        return false;
    int minimumGainIndex = round.calibrationCells.front().gainIndex;
    int maximumGainIndex = round.calibrationCells.back().gainIndex;
    const std::size_t gainCells = static_cast<std::size_t>(
        maximumGainIndex - minimumGainIndex + 1);
    bool shifted = false;
    if (lowerPressure && !upperPressure &&
        round.upperBoundaryProbability <=
            config.calibrationDiscardProbabilityThreshold) {
        if (round.gridShifts >= config.maximumGridShifts)
            throw std::runtime_error("Joint-grid winding BP exceeded its shift guard");
        --minimumGainIndex;
        --maximumGainIndex;
        shifted = true;
    } else if (upperPressure && !lowerPressure &&
               round.lowerBoundaryProbability <=
                   config.calibrationDiscardProbabilityThreshold) {
        if (round.gridShifts >= config.maximumGridShifts)
            throw std::runtime_error("Joint-grid winding BP exceeded its shift guard");
        ++minimumGainIndex;
        ++maximumGainIndex;
        shifted = true;
    } else {
        std::size_t required = gainCells;
        if (lowerPressure)
            ++required;
        if (upperPressure)
            ++required;
        if (required > config.maximumGainCells) {
            throw std::runtime_error(
                "Joint-grid winding BP calibration support exceeded its resource guard");
        }
        if (lowerPressure)
            --minimumGainIndex;
        if (upperPressure)
            ++maximumGainIndex;
    }
    remapCalibrationMessages(
        round,
        makeCalibrationCells(minimumGainIndex, maximumGainIndex, config));
    if (shifted)
        ++round.gridShifts;
    ++round.supportChanges;
    return true;
}

FiberTraceJointGridProgress gridProgress(
    FiberTraceJointGridProgressPhase phase,
    const GridRound& round,
    const FiberTraceJointGridWindingConfig& config,
    const std::chrono::steady_clock::time_point& started)
{
    FiberTraceJointGridProgress result;
    result.phase = phase;
    result.calibrationMode = round.fixedCalibration
        ? FiberTraceWindingCalibrationMode::Fixed
        : FiberTraceWindingCalibrationMode::Adaptive;
    result.messageIteration = round.iterations;
    result.maximumMessageIterations = config.maximumMessageIterations;
    result.candidateStates = round.totalStates;
    result.gainCells = round.fixedCalibration
        ? 1
        : (round.calibrationCells.empty()
               ? 0
               : static_cast<std::size_t>(
                     round.calibrationCells.back().gainIndex -
                     round.calibrationCells.front().gainIndex + 1));
    result.phaseCells = round.fixedCalibration ? 1 : config.phaseCells;
    result.gridShifts = round.gridShifts;
    result.messageResidual = round.residual;
    result.calibrationPosteriorResidual = round.calibrationResidual;
    result.lowerGainBoundaryProbability = round.lowerBoundaryProbability;
    result.upperGainBoundaryProbability = round.upperBoundaryProbability;
    if (round.fixedCalibration) {
        result.minimumGain = round.fixedCalibrationParameters.gain;
        result.maximumGain = round.fixedCalibrationParameters.gain;
        result.phaseMap = round.fixedCalibrationParameters.phase;
        result.phaseMean = round.fixedCalibrationParameters.phase;
        result.scaleMap = round.fixedMeasurementScale;
        result.scaleMean = round.fixedMeasurementScale;
    } else if (!round.calibrationCells.empty()) {
        result.minimumGain = round.calibrationCells.front().gain;
        result.maximumGain = round.calibrationCells.back().gain;
        if (!round.calibrationProbabilities.empty()) {
            const std::size_t map = static_cast<std::size_t>(std::distance(
                round.calibrationProbabilities.begin(),
                std::max_element(
                    round.calibrationProbabilities.begin(),
                    round.calibrationProbabilities.end())));
            result.phaseMap = round.calibrationCells[map].phase;
            result.scaleMap = 1.0 / round.calibrationCells[map].gain;
            result.phaseMean = 0.0;
            result.scaleMean = 0.0;
            for (std::size_t cell = 0;
                 cell < round.calibrationCells.size();
                 ++cell) {
                result.phaseMean += round.calibrationProbabilities[cell] *
                    round.calibrationCells[cell].phase;
                result.scaleMean += round.calibrationProbabilities[cell] /
                    round.calibrationCells[cell].gain;
            }
        }
    }
    result.elapsedSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    return result;
}

GridRound solveJointGridRound(
    const PreparedWinding& problem,
    std::span<const double> continuousNodes,
    std::span<const JointClass> fixedOrientations,
    const FiberTraceJointGridWindingConfig& config,
    const FiberTraceJointGridProgressCallback& progress,
    const std::chrono::steady_clock::time_point& started)
{
    GridRound round;
    round.fixedOrientations.assign(
        fixedOrientations.begin(), fixedOrientations.end());
    round.fixedCalibration = hasFixedCalibration(config);
    if (round.fixedCalibration)
        round.fixedMeasurementScale = *config.fixedMeasurementScale;
    round.gauge.assign(problem.piecesByNode.size(), 0);
    round.lower.assign(problem.piecesByNode.size(), 0);
    round.upper.assign(problem.piecesByNode.size(), 0);
    for (const std::size_t node : problem.integerGaugeNodes)
        round.gauge[node] = 1;
    for (const std::size_t node : problem.gaugeNodeByComponent)
        round.gauge[node] = 2;
    if (round.fixedCalibration) {
        round.fixedCalibrationParameters = fixedCalibrationCell(config);
    } else {
        const int halfGain = static_cast<int>(config.initialGainCells / 2);
        round.calibrationCells = makeCalibrationCells(-halfGain, halfGain, config);
    }
    for (std::size_t node = 0; node < problem.piecesByNode.size(); ++node) {
        if (round.gauge[node] != 0)
            continue;
        double low = std::numeric_limits<double>::infinity();
        double high = -std::numeric_limits<double>::infinity();
        if (round.fixedCalibration) {
            low = continuousNodes[node] /
                round.fixedCalibrationParameters.gain;
            high = low;
        } else {
            for (const auto& cell : round.calibrationCells) {
                low = std::min(low, continuousNodes[node] / cell.gain);
                high = std::max(high, continuousNodes[node] / cell.gain);
            }
        }
        round.lower[node] = static_cast<int>(std::floor(low)) - 1;
        round.upper[node] = static_cast<int>(std::ceil(high)) + 1;
    }
    initializeGridMessages(problem, round);
    round.totalStates = jointGridStateCount(problem, round);
    if (round.totalStates > config.maximumTotalCandidateStates) {
        throw std::runtime_error(
            "Joint-grid winding BP initial support exceeded its resource guard");
    }
    if (progress)
        progress(gridProgress(
            FiberTraceJointGridProgressPhase::Preparing,
            round,
            config,
            started));

    std::vector<double> previousCalibration;
    std::size_t stableIterations = 0;
    const std::size_t runtimeWorkers = static_cast<std::size_t>(
        std::max(1, omp_get_max_threads()));
    const int workers = static_cast<int>(std::min({
        config.parallelWorkers,
        std::max<std::size_t>(1, problem.edges.size()),
        runtimeWorkers,
        static_cast<std::size_t>(std::numeric_limits<int>::max()),
    }));
    const bool useParallel = workers > 1 && problem.edges.size() >= 64;
    round.effectiveWorkers = useParallel ? static_cast<std::size_t>(workers) : 1;
    for (std::size_t iteration = 0;
         iteration < config.maximumMessageIterations;
         ++iteration) {
        const auto totals = buildGridTotals(problem, round, config);
        std::vector<GridMessages> next(problem.edges.size());
        double residual = 0.0;
        #pragma omp parallel for schedule(dynamic, 4) num_threads(workers) if(useParallel) reduction(max : residual)
        for (std::size_t edgeIndex = 0;
             edgeIndex < problem.edges.size();
             ++edgeIndex) {
            const auto& edge = problem.edges[edgeIndex];
            const std::size_t component = problem.componentByNode[edge.a];
            residual = std::max(
                residual,
                updateGridFactor(
                    edge,
                    round.messages[edgeIndex],
                    next[edgeIndex],
                    totals.pieceAccumulators[edge.a],
                    totals.pieceAccumulators[edge.b],
                    totals.calibrationAccumulators,
                    totals.signAccumulators[component],
                    round,
                    config));
        }
        round.messages = std::move(next);
        round.iterations = iteration + 1;
        round.residual = residual;
        updateGridMarginals(problem, round, config);
        round.calibrationResidual = round.fixedCalibration
            ? 0.0
            : (previousCalibration.empty()
                   ? std::numeric_limits<double>::infinity()
                   : 0.0);
        if (!round.fixedCalibration && !previousCalibration.empty()) {
            if (previousCalibration.size() == round.calibrationProbabilities.size()) {
                for (std::size_t cell = 0;
                     cell < previousCalibration.size();
                     ++cell) {
                    round.calibrationResidual = std::max(
                        round.calibrationResidual,
                        std::abs(
                            previousCalibration[cell] -
                            round.calibrationProbabilities[cell]));
                }
            } else {
                round.calibrationResidual =
                    std::numeric_limits<double>::infinity();
            }
        }
        const bool posteriorSettled =
            (round.fixedCalibration || !previousCalibration.empty()) &&
            round.residual <= std::max(
                1.0e-6, 10.0 * config.messageResidualTolerance) &&
            (round.fixedCalibration ||
             round.calibrationResidual <= std::max(
                 1.0e-5, 10.0 * config.calibrationPosteriorTolerance));
        const bool posteriorSupportChanged = posteriorSettled &&
            adjustIntegerSupport(problem, round, config);
        const bool calibrationSupportChanged = !round.fixedCalibration &&
            posteriorSettled &&
            adjustCalibrationSupport(round, config);
        const bool calibrationIntegerSupportChanged =
            calibrationSupportChanged &&
            ensureIntegerSupport(problem, round, continuousNodes);
        if (posteriorSupportChanged || calibrationSupportChanged ||
            calibrationIntegerSupportChanged) {
            stableIterations = 0;
            previousCalibration.clear();
            round.totalStates = jointGridStateCount(problem, round);
            if (round.totalStates > config.maximumTotalCandidateStates) {
                throw std::runtime_error(
                    "Joint-grid winding BP adaptive support exceeded its resource guard");
            }
            if (progress)
                progress(gridProgress(
                    FiberTraceJointGridProgressPhase::SupportChanged,
                    round,
                    config,
                    started));
            continue;
        }
        if (!round.fixedCalibration)
            previousCalibration = round.calibrationProbabilities;
        if (round.residual <= config.messageResidualTolerance &&
            (round.fixedCalibration ||
             round.calibrationResidual <= config.calibrationPosteriorTolerance)) {
            ++stableIterations;
        } else {
            stableIterations = 0;
        }
        if (progress)
            progress(gridProgress(
                FiberTraceJointGridProgressPhase::MessagePassing,
                round,
                config,
                started));
        if (stableIterations >= config.stableIterations) {
            round.converged = true;
            break;
        }
    }
    updateGridMarginals(problem, round, config);
    round.totalStates = jointGridStateCount(problem, round);
    if (progress)
        progress(gridProgress(
            FiberTraceJointGridProgressPhase::Complete,
            round,
            config,
            started));
    return round;
}

FiberTraceInterleavedWindingReport makeJointGridReport(
    const PreparedWinding& prepared,
    const FiberTraceConstraintReport& constraints,
    std::span<const double> continuousNodes,
    double continuousResidual,
    double continuousSeconds,
    double discreteSeconds,
    const GridRound& round,
    const FiberTraceJointGridWindingConfig& config)
{
    const std::size_t pieceCount = constraints.pieces.size();
    FiberTraceInterleavedWindingReport report;
    report.solver = FiberTraceWindingSolver::JointGrid;
    report.defectUnaryCost = config.mixedUnaryCost;
    report.calibrationMode = round.fixedCalibration
        ? FiberTraceWindingCalibrationMode::Fixed
        : FiberTraceWindingCalibrationMode::Adaptive;
    report.variables = prepared.piecesByNode.size();
    report.factors = prepared.edges.size();
    report.connectedComponents = prepared.gaugeNodeByComponent.size();
    report.gaugePieces = prepared.gaugePieceByComponent;
    report.factorDiagnostics = prepared.diagnostics;
    report.continuousRootMeanSquareResidual = continuousResidual;
    report.temperature = config.temperature;
    report.continuousSolveSeconds = continuousSeconds;
    report.discreteSolveSeconds = discreteSeconds;
    report.expansionRounds = round.supportChanges + 1;
    report.messageIterations = round.iterations;
    report.messageResidual = round.residual;
    report.messageConverged = round.converged;
    report.effectiveWorkers = round.effectiveWorkers;
    report.totalCandidateStates = round.totalStates;
    report.calibrationGridCells = round.fixedCalibration
        ? 1
        : round.calibrationCells.size();
    report.calibrationGridShifts = round.gridShifts;
    report.lowerGainBoundaryProbability = round.lowerBoundaryProbability;
    report.upperGainBoundaryProbability = round.upperBoundaryProbability;
    std::size_t calibrationMap = 0;
    if (round.fixedCalibration) {
        report.minimumCalibrationGain = round.fixedCalibrationParameters.gain;
        report.maximumCalibrationGain = round.fixedCalibrationParameters.gain;
        report.phaseMagnitude = round.fixedCalibrationParameters.phase;
        report.measurementScale = round.fixedMeasurementScale;
        report.calibrationPhaseMean = round.fixedCalibrationParameters.phase;
        report.calibrationScaleMean = round.fixedMeasurementScale;
    } else {
        report.minimumCalibrationGain = round.calibrationCells.front().gain;
        report.maximumCalibrationGain = round.calibrationCells.back().gain;
        calibrationMap = static_cast<std::size_t>(std::distance(
            round.calibrationProbabilities.begin(),
            std::max_element(
                round.calibrationProbabilities.begin(),
                round.calibrationProbabilities.end())));
        report.phaseMagnitude = round.calibrationCells[calibrationMap].phase;
        report.measurementScale = 1.0 /
            round.calibrationCells[calibrationMap].gain;
        report.calibrationPhaseMean = 0.0;
        report.calibrationScaleMean = 0.0;
        for (std::size_t cell = 0;
             cell < round.calibrationCells.size();
             ++cell) {
            const double probability = round.calibrationProbabilities[cell];
            report.calibrationPhaseMean +=
                probability * round.calibrationCells[cell].phase;
            report.calibrationScaleMean += probability /
                round.calibrationCells[cell].gain;
            if (probability > 0.0) {
                report.calibrationEntropy -=
                    probability * std::log(probability);
            }
        }
    }
    report.calibrationConverged = round.converged;
    report.status = round.converged ? "converged" : "message_limit";
    report.componentPhaseSign.resize(round.signProbabilities.size());
    report.componentPositivePhaseSignProbability.resize(
        round.signProbabilities.size());
    for (std::size_t component = 0;
         component < round.signProbabilities.size();
         ++component) {
        report.componentPositivePhaseSignProbability[component] =
            round.signProbabilities[component][0];
        report.componentPhaseSign[component] =
            round.signProbabilities[component][0] >=
                    round.signProbabilities[component][1]
                ? 1
                : -1;
    }

    report.windingValid.resize(pieceCount);
    report.continuousWinding.resize(pieceCount);
    report.mapWinding.resize(pieceCount);
    report.posteriorMeanWinding.resize(pieceCount);
    report.mapProbability.resize(pieceCount);
    report.entropy.resize(pieceCount);
    report.candidateMinimum.resize(pieceCount);
    report.candidateMaximum.resize(pieceCount);
    report.componentByPiece.resize(pieceCount);
    report.integerGaugeByPiece.resize(pieceCount);
    report.classAProbability.resize(pieceCount);
    report.mixedProbability.resize(pieceCount);
    report.classBProbability.resize(pieceCount);
    report.posteriorMeanLatentCoordinate.resize(pieceCount);
    report.mapLatentCoordinate.resize(pieceCount);
    report.mapOrientationByPiece.resize(pieceCount);
    report.incidentSignedConstraints.assign(pieceCount, 0);
    report.incidentSkippedConstraints.assign(pieceCount, 0);
    for (const auto& diagnostic : report.factorDiagnostics) {
        const bool signedEvidence = diagnostic.canonicalSignedDelta.has_value();
        const bool expected = diagnostic.perpendicularScore > 0.0;
        for (const std::size_t piece : {diagnostic.pieceA, diagnostic.pieceB}) {
            if (signedEvidence)
                ++report.incidentSignedConstraints[piece];
            else if (expected)
                ++report.incidentSkippedConstraints[piece];
        }
    }
    std::vector<JointState> decoded(prepared.piecesByNode.size());
    std::vector<double> activeConfidence(decoded.size(), 0.0);
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        const std::size_t node = prepared.pieceToNode[piece];
        const auto& probabilities = round.pieceProbabilities[node];
        const std::size_t map = static_cast<std::size_t>(std::distance(
            probabilities.begin(),
            std::max_element(probabilities.begin(), probabilities.end())));
        auto mapState = gridPieceState(
            node, map, round.lower, round.gauge, round.fixedOrientations);
        double meanWinding = 0.0;
        double activeProbability = 0.0;
        double entropy = 0.0;
        double classA = 0.0;
        double mixed = 0.0;
        double classB = 0.0;
        for (std::size_t state = 0; state < probabilities.size(); ++state) {
            const auto current = gridPieceState(
                node,
                state,
                round.lower,
                round.gauge,
                round.fixedOrientations);
            const double probability = probabilities[state];
            if (current.orientation != JointClass::Mixed) {
                activeProbability += probability;
                meanWinding += probability * static_cast<double>(current.winding);
            }
            if (current.orientation == JointClass::A)
                classA += probability;
            else if (current.orientation == JointClass::Mixed)
                mixed += probability;
            else
                classB += probability;
            if (probability > 0.0)
                entropy -= probability * std::log(probability);
        }
        const std::size_t component = prepared.componentByNode[node];
        const double signMean = 2.0 *
                report.componentPositivePhaseSignProbability[component] -
            1.0;
        double meanLatent = 0.0;
        if (activeProbability > 0.0) {
            meanLatent = (meanWinding +
                signMean * report.calibrationPhaseMean * classB) /
                activeProbability;
            meanWinding /= activeProbability;
        }
        const JointClass finalClass = mixed >= classA && mixed >= classB
            ? JointClass::Mixed
            : classA > classB ? JointClass::A : JointClass::B;
        activeConfidence[node] = std::max(classA, classB) - mixed;
        double finalStateProbability = -1.0;
        for (std::size_t state = 0; state < probabilities.size(); ++state) {
            const auto current = gridPieceState(
                node,
                state,
                round.lower,
                round.gauge,
                round.fixedOrientations);
            if (current.orientation == finalClass &&
                probabilities[state] > finalStateProbability) {
                finalStateProbability = probabilities[state];
                mapState = current;
            }
        }
        decoded[node] = mapState;
        const bool activeMap = finalClass != JointClass::Mixed;
        report.windingValid[piece] = activeMap ? 1 : 0;
        report.continuousWinding[piece] =
            activeMap ? continuousNodes[node] : 0.0;
        report.mapWinding[piece] = activeMap ? mapState.winding : 0;
        report.posteriorMeanWinding[piece] = meanWinding;
        report.posteriorMeanLatentCoordinate[piece] = meanLatent;
        report.mapProbability[piece] = finalStateProbability;
        report.entropy[piece] = entropy;
        report.candidateMinimum[piece] = round.lower[node];
        report.candidateMaximum[piece] = round.upper[node];
        report.componentByPiece[piece] = component;
        report.integerGaugeByPiece[piece] = prepared.integerGaugeByNode[node];
        const double classTotal = classA + mixed + classB;
        if (!std::isfinite(classTotal) || !(classTotal > 0.0)) {
            throw std::runtime_error(
                "Joint-grid winding class marginal is invalid");
        }
        report.classAProbability[piece] = classA / classTotal;
        report.mixedProbability[piece] = mixed / classTotal;
        report.classBProbability[piece] = classB / classTotal;
    }
    const auto& selectedCalibration = round.fixedCalibration
        ? round.fixedCalibrationParameters
        : round.calibrationCells[calibrationMap];
    report.hardSignProjectedDefects = projectDecodedHardSigns(
        prepared,
        decoded,
        activeConfidence,
        [&](std::size_t edgeIndex, const JointState& a, const JointState& b) {
            const auto& edge = prepared.edges[edgeIndex];
            const std::size_t component = prepared.componentByNode[edge.a];
            const int sign = report.componentPhaseSign[component];
            const double latent = static_cast<double>(b.winding - a.winding) +
                classOffset(b.orientation, sign, selectedCalibration.phase) -
                classOffset(a.orientation, sign, selectedCalibration.phase);
            return selectedCalibration.gain * latent;
        });
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        const std::size_t node = prepared.pieceToNode[piece];
        const auto& state = decoded[node];
        const bool active = state.orientation != JointClass::Mixed;
        report.windingValid[piece] = active ? 1 : 0;
        report.continuousWinding[piece] =
            active ? continuousNodes[node] : 0.0;
        report.mapWinding[piece] = active ? state.winding : 0;
        report.mapOrientationByPiece[piece] = publicOrientation(state.orientation);
        report.mapLatentCoordinate[piece] =
            active ? static_cast<double>(state.winding) +
                         classOffset(state.orientation, report.componentPhaseSign.at(prepared.componentByNode[node]), selectedCalibration.phase)
                   : std::numeric_limits<double>::quiet_NaN();
        if (!active)
            report.mapProbability[piece] = report.mixedProbability[piece];
    }
    report.decodedEnergy = 0.0;
    report.orientationMode = round.fixedOrientations.empty()
        ? FiberTraceWindingOrientationMode::Joint
        : FiberTraceWindingOrientationMode::FixedPrepass;
    if (!round.fixedOrientations.empty()) {
        report.fixedOrientationByPiece.resize(pieceCount);
        for (std::size_t piece = 0; piece < pieceCount; ++piece) {
            switch (round.fixedOrientations[prepared.pieceToNode[piece]]) {
            case JointClass::A:
                report.fixedOrientationByPiece[piece] =
                    FiberTraceFixedOrientation::Horizontal;
                break;
            case JointClass::Mixed:
                report.fixedOrientationByPiece[piece] =
                    FiberTraceFixedOrientation::Mixed;
                break;
            case JointClass::B:
                report.fixedOrientationByPiece[piece] =
                    FiberTraceFixedOrientation::Vertical;
                break;
            }
        }
    }
    for (std::size_t node = 0; node < decoded.size(); ++node) {
        if (decoded[node].orientation == JointClass::Mixed &&
            (round.fixedOrientations.empty() ||
             round.fixedOrientations[node] != JointClass::Mixed)) {
            report.decodedEnergy += config.mixedUnaryCost;
        }
    }
    for (const auto& edge : prepared.edges) {
        const std::size_t component = prepared.componentByNode[edge.a];
        const auto a = decoded[edge.a];
        const auto b = decoded[edge.b];
        if (a.orientation == JointClass::Mixed ||
            b.orientation == JointClass::Mixed) {
            report.decodedEnergy -= config.temperature * gridLogPotential(
                edge,
                a,
                b,
                report.componentPhaseSign[component],
                selectedCalibration,
                config,
                round.fixedOrientations.empty());
        } else {
            if (round.fixedOrientations.empty()) {
                report.decodedEnergy += gridOrientationEnergy(
                    edge, a.orientation, b.orientation);
            }
            report.decodedEnergy += gridWindingEnergy(
                edge,
                a,
                b,
                report.componentPhaseSign[component],
                selectedCalibration.gain,
                selectedCalibration.phase);
        }
    }
    return report;
}

}  // namespace

FiberTraceWindingComponentSelection selectLargestFiberTraceWindingComponent(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceWindingBeliefPropagationConfig& config,
    std::span<const FiberTraceFixedOrientation> fixedOrientations,
    bool quantizeComponentTargets,
    std::optional<std::size_t> preferredPiece)
{
    validateConfig(config);
    if (preferredPiece && *preferredPiece >= constraints.pieces.size()) {
        throw std::invalid_argument("Preferred winding component piece is out of range");
    }
    const auto fixedClasses = fixedJointClasses(fixedOrientations, constraints.pieces.size());
    const auto prepared = prepareWinding(constraints, topology, config, fixedClasses, quantizeComponentTargets);

    FiberTraceWindingComponentSelection result;
    result.components = prepared.integerGaugeNodes.size();
    if (constraints.pieces.empty())
        return result;
    if (result.components == 0) {
        throw std::logic_error("Nonempty winding graph has no integer gauge component");
    }

    std::vector<std::size_t> counts(result.components, 0);
    std::vector<std::size_t> minimumPiece(result.components, constraints.pieces.size());
    std::vector<unsigned char> preferred(result.components, 0);
    for (std::size_t piece = 0; piece < constraints.pieces.size(); ++piece) {
        const std::size_t node = prepared.pieceToNode[piece];
        const std::size_t component = prepared.integerGaugeByNode[node];
        ++counts.at(component);
        minimumPiece[component] = std::min(minimumPiece[component], piece);
        if (preferredPiece && piece == *preferredPiece)
            preferred[component] = 1;
    }
    std::size_t selected = 0;
    for (std::size_t component = 1; component < result.components; ++component) {
        if (counts[component] > counts[selected] || (counts[component] == counts[selected] && preferred[component] > preferred[selected]) ||
            (counts[component] == counts[selected] && preferred[component] == preferred[selected] && minimumPiece[component] < minimumPiece[selected])) {
            selected = component;
        }
    }
    result.retainedPieceIndices.reserve(counts[selected]);
    for (std::size_t piece = 0; piece < constraints.pieces.size(); ++piece) {
        const std::size_t node = prepared.pieceToNode[piece];
        if (prepared.integerGaugeByNode[node] == selected)
            result.retainedPieceIndices.push_back(piece);
    }
    result.retainedPieces = result.retainedPieceIndices.size();
    result.removedPieces = constraints.pieces.size() - result.retainedPieces;
    return result;
}

double quantizedHalfWindingTarget(double value)
{
    if (value == 0.0)
        return 0.0;
    const double magnitude = std::ceil(std::abs(value)) - 0.5;
    return std::copysign(magnitude, value);
}

double quantizedIntegerWindingTarget(double value)
{
    if (value == 0.0)
        return 0.0;
    const double magnitude = std::abs(value);
    const double lower = std::floor(magnitude);
    const double rounded = magnitude - lower >= 0.5 ? lower + 1.0 : lower;
    return std::copysign(rounded, value);
}

void FiberTraceCanonicalConstraintCounts::add(
    double canonicalStep,
    double groundTruthStep) noexcept
{
    ++total;
    if (canonicalStep == groundTruthStep)
        ++correct;
    else
        ++falseCount;
}

FiberTraceReferenceWindingBenchmark calibrateFiberTraceReferenceWindings(std::span<const FiberTraceReferenceWindingObservation> observations, double tolerance)
{
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw std::invalid_argument("Reference winding benchmark tolerance must be finite and nonnegative");
    }

    std::map<std::size_t, std::vector<std::size_t>> observationsByGauge;
    for (std::size_t index = 0; index < observations.size(); ++index) {
        const auto& observation = observations[index];
        if (!std::isfinite(observation.virtualReferenceWinding) ||
            observation.inferredReferenceWindingCount > observation.inferredReferenceWindings.size()) {
            throw std::invalid_argument("Reference winding benchmark observation is invalid");
        }
        for (std::size_t candidate = 0; candidate < observation.inferredReferenceWindingCount; ++candidate) {
            if (!std::isfinite(observation.inferredReferenceWindings[candidate])) {
                throw std::invalid_argument("Reference winding benchmark candidate is invalid");
            }
        }
        const auto classIndex = static_cast<std::size_t>(observation.constraintClass);
        if (classIndex >= 3) {
            throw std::invalid_argument("Reference winding benchmark class is invalid");
        }
        if (observation.inferredReferenceWindingCount == 0)
            continue;
        observationsByGauge[observation.integerGauge].push_back(index);
    }

    FiberTraceReferenceWindingBenchmark result;
    result.tolerance = tolerance;
    std::map<std::size_t, double> offsetByGauge;
    const auto isRight = [&](const FiberTraceReferenceWindingObservation& o, double offset) {
        const double expected = o.virtualReferenceWinding + offset;
        for (std::size_t candidate = 0; candidate < o.inferredReferenceWindingCount; ++candidate) {
            if (std::abs(o.inferredReferenceWindings[candidate] - expected) <= tolerance + kEpsilon) {
                return true;
            }
        }
        return false;
    };
    const auto betterOffset = [](std::size_t right, double offset, std::size_t bestRight, double bestOffset) {
        if (right != bestRight)
            return right > bestRight;
        if (std::abs(offset) != std::abs(bestOffset))
            return std::abs(offset) < std::abs(bestOffset);
        return offset < bestOffset;
    };

    struct OffsetEvents {
        std::size_t starts = 0;
        std::size_t ends = 0;
    };
    for (const auto& [gauge, indices] : observationsByGauge) {
        std::map<double, OffsetEvents> events;
        for (const std::size_t index : indices) {
            const auto& observation = observations[index];
            std::array<std::pair<double, double>, 2> intervals{};
            std::size_t intervalCount = 0;
            for (std::size_t candidate = 0; candidate < observation.inferredReferenceWindingCount; ++candidate) {
                const double center = observation.inferredReferenceWindings[candidate] - observation.virtualReferenceWinding;
                intervals[intervalCount++] = {center - tolerance, center + tolerance};
            }
            if (intervalCount == 2 && intervals[1].first < intervals[0].first) {
                std::swap(intervals[0], intervals[1]);
            }
            if (intervalCount == 2 && intervals[1].first <= intervals[0].second + kEpsilon) {
                intervals[0].second = std::max(intervals[0].second, intervals[1].second);
                intervalCount = 1;
            }
            for (std::size_t interval = 0; interval < intervalCount; ++interval) {
                ++events[intervals[interval].first].starts;
                ++events[intervals[interval].second].ends;
            }
        }

        double bestOffset = 0.0;
        std::size_t bestRight = 0;
        for (const std::size_t index : indices)
            bestRight += isRight(observations[index], bestOffset) ? 1 : 0;
        std::size_t active = 0;
        for (const auto& [offset, event] : events) {
            active += event.starts;
            if (betterOffset(active, offset, bestRight, bestOffset)) {
                bestRight = active;
                bestOffset = offset;
            }
            active -= event.ends;
        }
        result.gauges.push_back({gauge, bestOffset, indices.size(), bestRight});
        offsetByGauge.emplace(gauge, bestOffset);
    }

    for (const auto& observation : observations) {
        if (observation.inferredReferenceWindingCount == 0)
            continue;
        auto& counts = result.classes[static_cast<std::size_t>(observation.constraintClass)];
        ++counts.total;
        ++result.sum.total;
        const bool right = isRight(observation, offsetByGauge.at(observation.integerGauge));
        if (right) {
            ++counts.right;
            ++result.sum.right;
        } else {
            ++counts.wrong;
            ++result.sum.wrong;
        }
    }
    return result;
}

FiberTraceReferenceWindingObservation makeFiberTraceReferenceWindingObservation(
    const FiberTraceConstraint& constraint, bool referenceIsEndpointA, double virtualReferenceWinding, std::size_t bpPiece, const FiberTraceInterleavedWindingReport& winding)
{
    if (!std::isfinite(virtualReferenceWinding) || bpPiece >= winding.windingValid.size() ||
        winding.mapLatentCoordinate.size() != winding.windingValid.size() || winding.mapOrientationByPiece.size() != winding.windingValid.size() ||
        winding.integerGaugeByPiece.size() != winding.windingValid.size() || !std::isfinite(winding.measurementScale) ||
        !(winding.measurementScale > 0.0)) {
        throw std::invalid_argument("Reference winding observation inputs are invalid");
    }

    FiberTraceReferenceWindingObservation observation;
    observation.integerGauge = winding.integerGaugeByPiece[bpPiece];
    observation.virtualReferenceWinding = virtualReferenceWinding;
    const bool active = winding.windingValid[bpPiece] != 0 && winding.mapOrientationByPiece[bpPiece] != FiberTraceFixedOrientation::Mixed &&
                        std::isfinite(winding.mapLatentCoordinate[bpPiece]);
    const bool perpendicular = constraint.perpendicularScore >= constraint.parallelScore;
    if (perpendicular) {
        observation.constraintClass = FiberTraceReferenceConstraintClass::Perpendicular;
        if (active && constraint.signedWindingDelta) {
            const double target = quantizedHalfWindingTarget(*constraint.signedWindingDelta);
            const double direction = referenceIsEndpointA ? -1.0 : 1.0;
            observation.inferredReferenceWindings[0] = winding.mapLatentCoordinate[bpPiece] + direction * winding.measurementScale * target;
            observation.inferredReferenceWindingCount = 1;
        }
        return observation;
    }

    const double target = std::abs(quantizedIntegerWindingTarget(constraint.windingDistance));
    observation.constraintClass =
        target == 0.0 ? FiberTraceReferenceConstraintClass::ParallelSameWinding : FiberTraceReferenceConstraintClass::ParallelOtherWinding;
    if (active) {
        observation.inferredReferenceWindings[0] = winding.mapLatentCoordinate[bpPiece] - target;
        observation.inferredReferenceWindings[1] = winding.mapLatentCoordinate[bpPiece] + target;
        observation.inferredReferenceWindingCount = target == 0.0 ? 1 : 2;
    }
    return observation;
}

const char* fiberTraceWindingSolverName(FiberTraceWindingSolver solver) noexcept
{
    switch (solver) {
    case FiberTraceWindingSolver::JointGrid:
        return "joint_grid";
    case FiberTraceWindingSolver::Alternating:
        return "alternating";
    }
    return "invalid";
}

const char* fiberTraceWindingOrientationModeName(
    FiberTraceWindingOrientationMode mode) noexcept
{
    switch (mode) {
    case FiberTraceWindingOrientationMode::Joint:
        return "joint";
    case FiberTraceWindingOrientationMode::FixedPrepass:
        return "fixed-prepass";
    }
    return "invalid";
}

const char* fiberTraceFixedOrientationName(
    FiberTraceFixedOrientation orientation) noexcept
{
    switch (orientation) {
    case FiberTraceFixedOrientation::Horizontal:
        return "h";
    case FiberTraceFixedOrientation::Mixed:
        return "mixed";
    case FiberTraceFixedOrientation::Vertical:
        return "v";
    }
    return "invalid";
}

std::vector<FiberTraceFixedOrientation> fixedFiberTraceOrientations(
    const FiberTraceBeliefPropagationReport& orientationBeliefs)
{
    const std::size_t count = orientationBeliefs.horizontalProbability.size();
    if (orientationBeliefs.mixedProbability.size() != count ||
        orientationBeliefs.verticalProbability.size() != count || count == 0) {
        throw std::invalid_argument(
            "Fixed winding orientation beliefs must have equal nonempty H/Mixed/V marginals");
    }
    std::vector<FiberTraceFixedOrientation> result(count);
    for (std::size_t piece = 0; piece < count; ++piece) {
        const std::array probabilities{
            orientationBeliefs.horizontalProbability[piece],
            orientationBeliefs.mixedProbability[piece],
            orientationBeliefs.verticalProbability[piece],
        };
        if (std::any_of(
                probabilities.begin(), probabilities.end(),
                [](double probability) {
                    return !std::isfinite(probability) || probability < 0.0;
                })) {
            throw std::invalid_argument(
                "Fixed winding orientation belief is invalid");
        }
        const double total = std::accumulate(
            probabilities.begin(), probabilities.end(), 0.0);
        if (!std::isfinite(total) || !(total > 0.0)) {
            throw std::invalid_argument(
                "Fixed winding orientation belief is invalid");
        }
        const double maximum = *std::max_element(
            probabilities.begin(), probabilities.end());
        if (std::count(
                probabilities.begin(), probabilities.end(), maximum) != 1) {
            result[piece] = FiberTraceFixedOrientation::Mixed;
        } else {
            const std::size_t state = static_cast<std::size_t>(std::distance(
                probabilities.begin(),
                std::find(probabilities.begin(), probabilities.end(), maximum)));
            result[piece] = state == 0
                ? FiberTraceFixedOrientation::Horizontal
                : state == 1
                    ? FiberTraceFixedOrientation::Mixed
                    : FiberTraceFixedOrientation::Vertical;
        }
    }
    return result;
}

const char* fiberTraceWindingCalibrationModeName(
    FiberTraceWindingCalibrationMode mode) noexcept
{
    switch (mode) {
    case FiberTraceWindingCalibrationMode::Adaptive:
        return "adaptive";
    case FiberTraceWindingCalibrationMode::Fixed:
        return "fixed";
    }
    return "invalid";
}

FiberTraceInterleavedWindingReport
solveFiberTraceJointGridWindingBeliefPropagation(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceJointGridWindingConfig& config,
    const FiberTraceJointGridProgressCallback& progress,
    std::span<const FiberTraceFixedOrientation> fixedOrientations)
{
    const auto started = std::chrono::steady_clock::now();
    validateJointGridConfig(config);
    const auto fixedClasses = fixedJointClasses(
        fixedOrientations, constraints.pieces.size());
    const auto prepared = prepareWinding(
        constraints, topology, config, fixedClasses, true);
    const auto continuousStarted = std::chrono::steady_clock::now();
    double continuousResidual = 0.0;
    const auto continuousNodes = solveContinuous(prepared, continuousResidual);
    const double continuousSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - continuousStarted).count();
    const auto discreteStarted = std::chrono::steady_clock::now();
    const auto round = solveJointGridRound(
        prepared,
        continuousNodes,
        fixedClasses,
        config,
        progress,
        started);
    const double discreteSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - discreteStarted).count();
    return makeJointGridReport(
        prepared,
        constraints,
        continuousNodes,
        continuousResidual,
        continuousSeconds,
        discreteSeconds,
        round,
        config);
}

FiberTraceWindingBeliefPropagationReport solveFiberTraceWindingBeliefPropagation(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceWindingBeliefPropagationConfig& config)
{
    validateConfig(config);
    const auto prepared = prepareWinding(constraints, topology, config);
    FiberTraceWindingBeliefPropagationReport report;
    report.variables = prepared.piecesByNode.size();
    report.factors = prepared.edges.size();
    report.connectedComponents = prepared.gaugeNodeByComponent.size();
    report.gaugePieces = prepared.gaugePieceByComponent;
    report.factorDiagnostics = prepared.diagnostics;

    const auto continuousStarted = std::chrono::steady_clock::now();
    double continuousResidual = 0.0;
    const auto continuousNodes = solveContinuous(prepared, continuousResidual);
    report.continuousRootMeanSquareResidual = continuousResidual;
    report.temperature = config.temperature;
    report.continuousSolveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - continuousStarted).count();

    std::vector<unsigned char> gauge(prepared.piecesByNode.size(), 0);
    for (const std::size_t node : prepared.integerGaugeNodes)
        gauge[node] = 1;
    std::vector<int> lower(prepared.piecesByNode.size());
    std::vector<int> upper(prepared.piecesByNode.size());
    for (std::size_t node = 0; node < prepared.piecesByNode.size(); ++node) {
        if (gauge[node] != 0) {
            lower[node] = 0;
            upper[node] = 0;
        } else {
            const int center = static_cast<int>(std::round(continuousNodes[node]));
            lower[node] = center - 1;
            upper[node] = center + 1;
        }
    }

    const auto discreteStarted = std::chrono::steady_clock::now();
    DiscreteRound discrete;
    for (;;) {
        std::size_t totalStates = 0;
        for (std::size_t node = 0; node < lower.size(); ++node)
            totalStates += static_cast<std::size_t>(upper[node] - lower[node] + 1);
        if (totalStates > config.maximumTotalCandidateStates) {
            throw std::runtime_error(
                "Winding BP adaptive candidate support exceeded its resource guard");
        }
        discrete = solveDiscreteRound(prepared, lower, upper, config);
        report.effectiveWorkers = discrete.effectiveWorkers;
        report.messageIterations += discrete.iterations;
        ++report.expansionRounds;
        std::vector<unsigned char> expandLower(lower.size(), 0);
        std::vector<unsigned char> expandUpper(upper.size(), 0);
        bool expand = false;
        for (std::size_t node = 0; node < lower.size(); ++node) {
            if (gauge[node] != 0)
                continue;
            const auto& probabilities = discrete.probabilities[node];
            const auto maximum = std::max_element(
                probabilities.begin(), probabilities.end());
            const std::size_t map = static_cast<std::size_t>(
                std::distance(probabilities.begin(), maximum));
            if (map == 0)
                expandLower[node] = 1;
            if (map + 1 == probabilities.size())
                expandUpper[node] = 1;
            if (probabilities.front() + probabilities.back() >
                config.boundaryProbabilityThreshold) {
                if (probabilities.front() >= probabilities.back())
                    expandLower[node] = 1;
                if (probabilities.back() >= probabilities.front())
                    expandUpper[node] = 1;
            }
            if (expandLower[node] != 0) {
                --lower[node];
                expand = true;
            }
            if (expandUpper[node] != 0) {
                ++upper[node];
                expand = true;
            }
        }
        if (!expand)
            break;
    }
    report.discreteSolveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - discreteStarted).count();
    report.messageResidual = discrete.residual;
    report.messageConverged = discrete.converged;
    report.status = discrete.converged ? "converged" : "message_limit";
    report.totalCandidateStates = 0;
    for (std::size_t node = 0; node < lower.size(); ++node)
        report.totalCandidateStates += static_cast<std::size_t>(upper[node] - lower[node] + 1);

    const std::size_t pieceCount = constraints.pieces.size();
    report.windingValid.assign(pieceCount, 1);
    report.continuousWinding.resize(pieceCount);
    report.mapWinding.resize(pieceCount);
    report.posteriorMeanWinding.resize(pieceCount);
    report.mapProbability.resize(pieceCount);
    report.entropy.resize(pieceCount);
    report.candidateMinimum.resize(pieceCount);
    report.candidateMaximum.resize(pieceCount);
    report.componentByPiece.resize(pieceCount);
    report.integerGaugeByPiece.resize(pieceCount);
    report.incidentSignedConstraints.assign(pieceCount, 0);
    report.incidentSkippedConstraints.assign(pieceCount, 0);
    for (const auto& diagnostic : report.factorDiagnostics) {
        const bool signedEvidence = diagnostic.canonicalSignedDelta.has_value();
        const bool expected = diagnostic.perpendicularScore > 0.0;
        for (const std::size_t piece : {diagnostic.pieceA, diagnostic.pieceB}) {
            if (signedEvidence)
                ++report.incidentSignedConstraints[piece];
            else if (expected)
                ++report.incidentSkippedConstraints[piece];
        }
    }
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        const std::size_t node = prepared.pieceToNode[piece];
        const auto& probabilities = discrete.probabilities[node];
        const auto maximum = std::max_element(probabilities.begin(), probabilities.end());
        const std::size_t map = static_cast<std::size_t>(
            std::distance(probabilities.begin(), maximum));
        double mean = 0.0;
        double entropy = 0.0;
        for (std::size_t state = 0; state < probabilities.size(); ++state) {
            const double probability = probabilities[state];
            mean += probability * static_cast<double>(lower[node] + static_cast<int>(state));
            if (probability > 0.0)
                entropy -= probability * std::log(probability);
        }
        report.continuousWinding[piece] = continuousNodes[node];
        report.mapWinding[piece] = lower[node] + static_cast<int>(map);
        report.posteriorMeanWinding[piece] = mean;
        report.mapProbability[piece] = *maximum;
        report.entropy[piece] = entropy;
        report.candidateMinimum[piece] = lower[node];
        report.candidateMaximum[piece] = upper[node];
        report.componentByPiece[piece] = prepared.componentByNode[node];
        report.integerGaugeByPiece[piece] = prepared.integerGaugeByNode[node];
    }
    return report;
}

FiberTraceInterleavedWindingReport
solveFiberTraceInterleavedWindingBeliefPropagation(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceBeliefPropagationReport& orientationBeliefs,
    const FiberTraceInterleavedWindingConfig& config,
    const FiberTraceInterleavedWindingProgressCallback& progress,
    std::span<const FiberTraceFixedOrientation> fixedOrientations)
{
    const auto progressStarted = std::chrono::steady_clock::now();
    validateConfig(config);
    const std::size_t pieceCount = constraints.pieces.size();
    const auto fixedClasses = fixedJointClasses(
        fixedOrientations, pieceCount);
    if (orientationBeliefs.horizontalProbability.size() != pieceCount ||
        orientationBeliefs.mixedProbability.size() != pieceCount ||
        orientationBeliefs.verticalProbability.size() != pieceCount) {
        throw std::invalid_argument(
            "Interleaved winding orientation beliefs do not match constraints");
    }
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        const std::array probabilities{
            orientationBeliefs.horizontalProbability[piece],
            orientationBeliefs.mixedProbability[piece],
            orientationBeliefs.verticalProbability[piece],
        };
        if (std::any_of(
                probabilities.begin(), probabilities.end(),
                [](double probability) {
                    return !std::isfinite(probability) || probability < 0.0;
                })) {
            throw std::invalid_argument(
                "Interleaved winding orientation beliefs are invalid");
        }
        const double total = std::accumulate(
            probabilities.begin(), probabilities.end(), 0.0);
        if (!std::isfinite(total) || !(total > 0.0)) {
            throw std::invalid_argument(
                "Interleaved winding orientation beliefs are invalid");
        }
    }
    if (!std::isfinite(config.mixedUnaryCost) ||
        config.mixedUnaryCost < 0.0 ||
        !std::isfinite(config.orientationTemperature) ||
        !(config.orientationTemperature > 0.0) ||
        !std::isfinite(config.minimumMeasurementScale) ||
        !std::isfinite(config.maximumMeasurementScale) ||
        !(config.minimumMeasurementScale > 0.0) ||
        config.maximumMeasurementScale < config.minimumMeasurementScale ||
        !std::isfinite(config.calibrationTolerance) ||
        config.calibrationTolerance < 0.0 ||
        config.maximumCalibrationIterations == 0) {
        throw std::invalid_argument("Interleaved winding calibration config is invalid");
    }
    constexpr std::array initialPhases{0.2, 0.4};
    constexpr std::array initialScales{1.0, 1.25};
    constexpr std::size_t initializationCount =
        initialPhases.size() * initialScales.size();
    if (progress) {
        progress({
            FiberTraceInterleavedWindingProgressPhase::Preparing,
            0,
            initializationCount,
            0,
            config.maximumCalibrationIterations,
            0,
            0,
            config.maximumMessageIterations,
            0,
            0,
            0.0,
            0.0,
            1.0,
            0.0,
        });
    }
    const auto prepared = prepareWinding(
        constraints, topology, config, fixedClasses, true);
    const auto continuousStarted = std::chrono::steady_clock::now();
    double continuousResidual = 0.0;
    const auto continuousNodes = solveContinuous(prepared, continuousResidual);
    const double continuousSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - continuousStarted).count();

    const auto discreteStarted = std::chrono::steady_clock::now();
    std::optional<JointCandidate> best;
    std::size_t initialization = 0;
    std::size_t accumulatedMessageIterations = 0;
    for (const double initialPhase : initialPhases) {
        for (const double initialScale : initialScales) {
            JointCandidate candidate;
            candidate.initialization = initialization++;
            candidate.parameters.phase = initialPhase;
            candidate.parameters.scale = initialScale;
            candidate.parameters.componentSign.assign(
                prepared.gaugeNodeByComponent.size(), 1);
            std::vector<unsigned char> gauge(prepared.piecesByNode.size(), 0);
            for (const std::size_t node : prepared.integerGaugeNodes)
                gauge[node] = 1;
            std::vector<int> lower(prepared.piecesByNode.size());
            std::vector<int> upper(prepared.piecesByNode.size());
            for (std::size_t node = 0; node < prepared.piecesByNode.size(); ++node) {
                if (gauge[node] != 0) {
                    lower[node] = 0;
                    upper[node] = 0;
                } else {
                    const int center = static_cast<int>(std::round(
                        continuousNodes[node] * initialScale));
                    lower[node] = center - 1;
                    upper[node] = center + 1;
                }
            }
            for (std::size_t iteration = 0;
                 iteration < config.maximumCalibrationIterations;
                 ++iteration) {
                candidate.round = solveJointAdaptive(
                    prepared,
                    std::move(lower),
                    std::move(upper),
                    orientationBeliefs,
                    fixedClasses,
                    candidate.parameters,
                    config,
                    candidate.initialization + 1,
                    initializationCount,
                    iteration + 1,
                    accumulatedMessageIterations,
                    progressStarted,
                    progress);
                lower = candidate.round.lower;
                upper = candidate.round.upper;
                candidate.totalMessageIterations +=
                    candidate.round.messageIterations;
                accumulatedMessageIterations +=
                    candidate.round.messageIterations;
                candidate.totalExpansionRounds += candidate.round.expansionRounds;
                candidate.calibrationIterations = iteration + 1;
                if (!candidate.round.discrete.converged)
                    break;
                const auto beliefs = jointPairBeliefs(
                    prepared,
                    candidate.round,
                    fixedClasses,
                    candidate.parameters,
                    config);
                const auto update = updateCalibration(
                    prepared, beliefs, candidate.parameters, config);
                if (update.rankDeficient)
                    ++candidate.rankDeficientUpdates;
                const bool signsChanged =
                    update.componentSign != candidate.parameters.componentSign;
                const bool stable = !signsChanged &&
                    std::abs(update.phase - candidate.parameters.phase) <=
                        config.calibrationTolerance &&
                    std::abs(update.scale - candidate.parameters.scale) <=
                        config.calibrationTolerance;
                if (progress) {
                    progress({
                        FiberTraceInterleavedWindingProgressPhase::Calibration,
                        candidate.initialization + 1,
                        initializationCount,
                        iteration + 1,
                        config.maximumCalibrationIterations,
                        candidate.round.expansionRounds,
                        candidate.round.discrete.iterations,
                        config.maximumMessageIterations,
                        accumulatedMessageIterations,
                        candidate.round.totalStates,
                        candidate.round.discrete.residual,
                        update.phase,
                        update.scale,
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - progressStarted)
                            .count(),
                    });
                }
                if (stable) {
                    candidate.calibrationConverged = true;
                    break;
                }
                if (iteration + 1 == config.maximumCalibrationIterations)
                    break;
                candidate.parameters.phase = update.phase;
                candidate.parameters.scale = update.scale;
                candidate.parameters.componentSign = update.componentSign;
            }
            candidate.decodedEnergy = decodedJointEnergy(
                prepared,
                candidate.round,
                orientationBeliefs,
                fixedClasses,
                candidate.parameters,
                config);
            const bool candidateConverged =
                candidate.round.discrete.converged &&
                candidate.calibrationConverged;
            const bool bestConverged = best &&
                best->round.discrete.converged &&
                best->calibrationConverged;
            if (progress) {
                progress({
                    FiberTraceInterleavedWindingProgressPhase::InitializationComplete,
                    initialization,
                    initializationCount,
                    candidate.calibrationIterations,
                    config.maximumCalibrationIterations,
                    candidate.round.expansionRounds,
                    candidate.round.discrete.iterations,
                    config.maximumMessageIterations,
                    accumulatedMessageIterations,
                    candidate.round.totalStates,
                    candidate.round.discrete.residual,
                    candidate.parameters.phase,
                    candidate.parameters.scale,
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - progressStarted).count(),
                });
            }
            if (!best || (candidateConverged && !bestConverged) ||
                (candidateConverged == bestConverged &&
                 candidate.decodedEnergy < best->decodedEnergy)) {
                best = std::move(candidate);
            }
        }
    }
    if (!best)
        throw std::logic_error("Interleaved winding produced no initialization");

    FiberTraceInterleavedWindingReport report;
    report.defectUnaryCost = config.mixedUnaryCost;
    report.variables = prepared.piecesByNode.size();
    report.factors = prepared.edges.size();
    report.connectedComponents = prepared.gaugeNodeByComponent.size();
    report.gaugePieces = prepared.gaugePieceByComponent;
    report.factorDiagnostics = prepared.diagnostics;
    report.continuousRootMeanSquareResidual = continuousResidual;
    report.temperature = config.temperature;
    report.orientationMode = fixedClasses.empty()
        ? FiberTraceWindingOrientationMode::Joint
        : FiberTraceWindingOrientationMode::FixedPrepass;
    if (!fixedClasses.empty()) {
        report.fixedOrientationByPiece.assign(
            fixedOrientations.begin(), fixedOrientations.end());
    }
    report.continuousSolveSeconds = continuousSeconds;
    report.discreteSolveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - discreteStarted).count();
    report.expansionRounds = best->totalExpansionRounds;
    report.messageIterations = best->totalMessageIterations;
    report.messageResidual = best->round.discrete.residual;
    report.messageConverged = best->round.discrete.converged;
    report.effectiveWorkers = best->round.discrete.effectiveWorkers;
    report.totalCandidateStates = best->round.totalStates;
    report.phaseMagnitude = best->parameters.phase;
    report.measurementScale = best->parameters.scale;
    report.componentPhaseSign = best->parameters.componentSign;
    report.decodedEnergy = best->decodedEnergy;
    report.calibrationIterations = best->calibrationIterations;
    report.selectedInitialization = best->initialization;
    report.rankDeficientUpdates = best->rankDeficientUpdates;
    report.calibrationConverged = best->calibrationConverged;
    if (!report.messageConverged)
        report.status = "message_limit";
    else if (!report.calibrationConverged)
        report.status = "calibration_limit";
    else
        report.status = "converged";

    report.windingValid.resize(pieceCount);
    report.continuousWinding.resize(pieceCount);
    report.mapWinding.resize(pieceCount);
    report.posteriorMeanWinding.resize(pieceCount);
    report.mapProbability.resize(pieceCount);
    report.entropy.resize(pieceCount);
    report.candidateMinimum.resize(pieceCount);
    report.candidateMaximum.resize(pieceCount);
    report.componentByPiece.resize(pieceCount);
    report.integerGaugeByPiece.resize(pieceCount);
    report.classAProbability.resize(pieceCount);
    report.mixedProbability.resize(pieceCount);
    report.classBProbability.resize(pieceCount);
    report.posteriorMeanLatentCoordinate.resize(pieceCount);
    report.mapLatentCoordinate.resize(pieceCount);
    report.mapOrientationByPiece.resize(pieceCount);
    report.incidentSignedConstraints.assign(pieceCount, 0);
    report.incidentSkippedConstraints.assign(pieceCount, 0);
    for (const auto& diagnostic : report.factorDiagnostics) {
        const bool signedEvidence = diagnostic.canonicalSignedDelta.has_value();
        const bool expected = diagnostic.perpendicularScore > 0.0;
        for (const std::size_t piece : {diagnostic.pieceA, diagnostic.pieceB}) {
            if (signedEvidence)
                ++report.incidentSignedConstraints[piece];
            else if (expected)
                ++report.incidentSkippedConstraints[piece];
        }
    }
    std::vector<JointState> decoded(prepared.piecesByNode.size());
    std::vector<double> activeConfidence(decoded.size(), 0.0);
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        const std::size_t node = prepared.pieceToNode[piece];
        const auto& probabilities = best->round.discrete.probabilities[node];
        const std::size_t map = static_cast<std::size_t>(std::distance(
            probabilities.begin(),
            std::max_element(probabilities.begin(), probabilities.end())));
        auto mapState = jointState(
            node, map, best->round.lower[node], fixedClasses);
        double meanWinding = 0.0;
        double meanLatent = 0.0;
        double activeProbability = 0.0;
        double entropy = 0.0;
        double classA = 0.0;
        double mixed = 0.0;
        double classB = 0.0;
        const int sign = best->parameters.componentSign.at(
            prepared.componentByNode[node]);
        for (std::size_t state = 0; state < probabilities.size(); ++state) {
            const double probability = probabilities[state];
            const auto current = jointState(
                node, state, best->round.lower[node], fixedClasses);
            if (current.orientation != JointClass::Mixed) {
                activeProbability += probability;
                meanWinding += probability * static_cast<double>(current.winding);
                meanLatent += probability *
                    (static_cast<double>(current.winding) +
                     classOffset(
                         current.orientation, sign, best->parameters.phase));
            }
            if (current.orientation == JointClass::A)
                classA += probability;
            else if (current.orientation == JointClass::Mixed)
                mixed += probability;
            else
                classB += probability;
            if (probability > 0.0)
                entropy -= probability * std::log(probability);
        }
        if (activeProbability > 0.0) {
            meanWinding /= activeProbability;
            meanLatent /= activeProbability;
        }
        const JointClass finalClass = mixed >= classA && mixed >= classB
            ? JointClass::Mixed
            : classA > classB ? JointClass::A : JointClass::B;
        activeConfidence[node] = std::max(classA, classB) - mixed;
        double finalStateProbability = -1.0;
        for (std::size_t state = 0; state < probabilities.size(); ++state) {
            const auto current = jointState(
                node, state, best->round.lower[node], fixedClasses);
            if (current.orientation == finalClass &&
                probabilities[state] > finalStateProbability) {
                finalStateProbability = probabilities[state];
                mapState = current;
            }
        }
        decoded[node] = mapState;
        const bool activeMap = finalClass != JointClass::Mixed;
        report.windingValid[piece] = activeMap ? 1 : 0;
        report.continuousWinding[piece] =
            activeMap ? continuousNodes[node] : 0.0;
        report.mapWinding[piece] = activeMap ? mapState.winding : 0;
        report.posteriorMeanWinding[piece] = meanWinding;
        report.posteriorMeanLatentCoordinate[piece] = meanLatent;
        report.mapProbability[piece] = finalStateProbability;
        report.entropy[piece] = entropy;
        report.candidateMinimum[piece] = best->round.lower[node];
        report.candidateMaximum[piece] = best->round.upper[node];
        report.componentByPiece[piece] = prepared.componentByNode[node];
        report.integerGaugeByPiece[piece] = prepared.integerGaugeByNode[node];
        const double classTotal = classA + mixed + classB;
        if (!std::isfinite(classTotal) || !(classTotal > 0.0))
            throw std::runtime_error("Interleaved winding class marginal is invalid");
        report.classAProbability[piece] = classA / classTotal;
        report.mixedProbability[piece] = mixed / classTotal;
        report.classBProbability[piece] = classB / classTotal;
    }
    report.hardSignProjectedDefects = projectDecodedHardSigns(
        prepared,
        decoded,
        activeConfidence,
        [&](std::size_t edgeIndex, const JointState& a, const JointState& b) {
            const auto& edge = prepared.edges[edgeIndex];
            const int sign = best->parameters.componentSign.at(
                prepared.componentByNode[edge.a]);
            const double latent = static_cast<double>(b.winding - a.winding) +
                classOffset(b.orientation, sign, best->parameters.phase) -
                classOffset(a.orientation, sign, best->parameters.phase);
            return latent / best->parameters.scale;
        });
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        const std::size_t node = prepared.pieceToNode[piece];
        const auto& state = decoded[node];
        const bool active = state.orientation != JointClass::Mixed;
        report.windingValid[piece] = active ? 1 : 0;
        report.continuousWinding[piece] =
            active ? continuousNodes[node] : 0.0;
        report.mapWinding[piece] = active ? state.winding : 0;
        report.mapOrientationByPiece[piece] = publicOrientation(state.orientation);
        report.mapLatentCoordinate[piece] =
            active ? static_cast<double>(state.winding) +
                         classOffset(state.orientation, best->parameters.componentSign.at(prepared.componentByNode[node]), best->parameters.phase)
                   : std::numeric_limits<double>::quiet_NaN();
        if (!active)
            report.mapProbability[piece] = report.mixedProbability[piece];
    }
    if (progress) {
        progress({
            FiberTraceInterleavedWindingProgressPhase::Complete,
            initializationCount,
            initializationCount,
            report.calibrationIterations,
            config.maximumCalibrationIterations,
            best->round.expansionRounds,
            best->round.discrete.iterations,
            config.maximumMessageIterations,
            accumulatedMessageIterations,
            report.totalCandidateStates,
            report.messageResidual,
            report.phaseMagnitude,
            report.measurementScale,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - progressStarted).count(),
        });
    }
    return report;
}

}  // namespace vc::fiber_tracer
