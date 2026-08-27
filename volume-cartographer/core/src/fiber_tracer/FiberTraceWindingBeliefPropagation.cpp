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

struct Measurement {
    std::size_t constraintIndex = 0;
    std::size_t a = 0;
    std::size_t b = 0;
    double parallel = 0.0;
    double perpendicular = 0.0;
    std::optional<double> signedDelta;
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
    std::vector<std::size_t> gaugeNodeByComponent;
    std::vector<std::size_t> gaugePieceByComponent;
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
}

PreparedWinding prepareWinding(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology)
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
        const std::size_t originalA = result.pieceToNode[constraint.pieceA];
        const std::size_t originalB = result.pieceToNode[constraint.pieceB];
        const std::size_t a = std::min(originalA, originalB);
        const std::size_t b = std::max(originalA, originalB);
        std::optional<double> signedDelta = continuity
            ? std::optional<double>{0.0}
            : constraint.signedWindingDelta;
        if (signedDelta && originalA > originalB)
            *signedDelta = -*signedDelta;
        Measurement measurement{
            index,
            a,
            b,
            constraint.parallelScore,
            constraint.perpendicularScore,
            signedDelta,
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
            constraint.signedWindingDelta,
            signedDelta,
            continuity ? std::nullopt : constraint.windingNormalComponent,
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
    for (std::size_t edge = 0; edge < result.edges.size(); ++edge) {
        bool positive = false;
        for (const auto& measurement : result.edges[edge].measurements) {
            positive = positive || measurement.parallel > 0.0 ||
                (measurement.signedDelta && measurement.perpendicular > 0.0);
        }
        if (!positive)
            continue;
        result.adjacency[result.edges[edge].a].push_back(edge);
        result.adjacency[result.edges[edge].b].push_back(edge);
    }

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
        std::optional<std::size_t> normalComponent;
        for (const std::size_t node : nodes) {
            for (const std::size_t edgeIndex : result.adjacency[node]) {
                const auto& edge = result.edges[edgeIndex];
                if (edge.a != node)
                    continue;
                for (const auto& measurement : edge.measurements) {
                    if (!measurement.signedDelta ||
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
        result.gaugeNodeByComponent.push_back(gaugeNode);
        result.gaugePieceByComponent.push_back(gaugePiece);
    }
    return result;
}

double measurementSquaredWeight(const Measurement& measurement)
{
    return measurement.parallel +
        (measurement.signedDelta ? measurement.perpendicular : 0.0);
}

double measurementSquaredTarget(const Measurement& measurement)
{
    const double weight = measurementSquaredWeight(measurement);
    if (!(weight > 0.0) || !measurement.signedDelta)
        return 0.0;
    return measurement.perpendicular * *measurement.signedDelta / weight;
}

std::vector<double> solveContinuous(
    const PreparedWinding& problem,
    double& rootMeanSquareResidual)
{
    const std::size_t nodeCount = problem.piecesByNode.size();
    std::vector<unsigned char> gauge(nodeCount, 0);
    for (const std::size_t node : problem.gaugeNodeByComponent)
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
        if (measurement.parallel > 0.0) {
            squared += measurement.parallel * delta * delta;
            ++terms;
        }
        if (measurement.signedDelta && measurement.perpendicular > 0.0) {
            const double residual = delta - *measurement.signedDelta;
            squared += measurement.perpendicular * residual * residual;
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
        cost += measurement.parallel * std::abs(delta);
        if (measurement.signedDelta) {
            cost += measurement.perpendicular *
                std::abs(delta - *measurement.signedDelta);
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
            result.totals[node] = logUnary[node];
            for (const std::size_t edge : problem.adjacency[node]) {
                const auto& current = problem.edges[edge];
                const auto& incoming = current.a == node
                    ? result.bToA[edge]
                    : result.aToB[edge];
                for (std::size_t state = 0; state < incoming.size(); ++state)
                    result.totals[node][state] += incoming[state];
            }
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
                        result.totals[current.a][stateA] -
                        result.bToA[edge][stateA] +
                        logPotential(edge, stateA, stateB));
                }
                nextAToB[edge][stateB] = logSumExp(candidates);
            }
            const double normAToB = logSumExp(nextAToB[edge]);
            for (std::size_t state = 0; state < nextAToB[edge].size(); ++state) {
                const double raw = nextAToB[edge][state] - normAToB;
                const double damped = result.aToB[edge][state] +
                    config.messageDamping * (raw - result.aToB[edge][state]);
                residual = std::max(
                    residual, std::abs(damped - result.aToB[edge][state]));
                nextAToB[edge][state] = damped;
            }

            candidates.reserve(result.totals[current.b].size());
            for (std::size_t stateA = 0; stateA < result.bToA[edge].size(); ++stateA) {
                candidates.clear();
                for (std::size_t stateB = 0; stateB < result.aToB[edge].size(); ++stateB) {
                    candidates.push_back(
                        result.totals[current.b][stateB] -
                        result.aToB[edge][stateB] +
                        logPotential(edge, stateA, stateB));
                }
                nextBToA[edge][stateA] = logSumExp(candidates);
            }
            const double normBToA = logSumExp(nextBToA[edge]);
            for (std::size_t state = 0; state < nextBToA[edge].size(); ++state) {
                const double raw = nextBToA[edge][state] - normBToA;
                const double damped = result.bToA[edge][state] +
                    config.messageDamping * (raw - result.bToA[edge][state]);
                residual = std::max(
                    residual, std::abs(damped - result.bToA[edge][state]));
                nextBToA[edge][state] = damped;
            }
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
        result.totals[node] = logUnary[node];
        for (const std::size_t edge : problem.adjacency[node]) {
            const auto& current = problem.edges[edge];
            const auto& incoming = current.a == node
                ? result.bToA[edge]
                : result.aToB[edge];
            for (std::size_t state = 0; state < incoming.size(); ++state)
                result.totals[node][state] += incoming[state];
        }
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

enum class JointClass : unsigned char { A = 0, Mixed = 1, B = 2 };

struct JointState {
    JointClass orientation = JointClass::A;
    int winding = 0;
};

JointState jointState(std::size_t state, int lower)
{
    return {
        static_cast<JointClass>(state % 3),
        lower + static_cast<int>(state / 3),
    };
}

double classOffset(JointClass orientation, int sign, double phase)
{
    return orientation == JointClass::B
        ? static_cast<double>(sign) * phase
        : 0.0;
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
    double energy = 0.0;
    for (const auto& measurement : edge.measurements) {
        energy += measurement.parallel * std::abs(delta);
        if (measurement.signedDelta) {
            energy += measurement.perpendicular *
                std::abs(delta / scale - *measurement.signedDelta);
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
    if (a.orientation != JointClass::Mixed &&
        b.orientation != JointClass::Mixed) {
        return -windingEnergy(edge, a, b, sign, phase, scale) / temperature;
    }
    std::array<double, 4> latent{};
    std::size_t index = 0;
    for (const JointClass classA : {JointClass::A, JointClass::B}) {
        for (const JointClass classB : {JointClass::A, JointClass::B}) {
            a.orientation = classA;
            b.orientation = classB;
            latent[index++] =
                -windingEnergy(edge, a, b, sign, phase, scale) / temperature;
        }
    }
    return logSumExp(latent) - std::log(4.0);
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
    for (const std::size_t node : problem.gaugeNodeByComponent)
        gauge[node] = 1;
    JointAdaptiveRound result;
    for (;;) {
        result.totalStates = 0;
        std::vector<std::vector<double>> logUnary(lower.size());
        for (std::size_t node = 0; node < lower.size(); ++node) {
            const std::size_t integers = static_cast<std::size_t>(
                upper[node] - lower[node] + 1);
            result.totalStates += 3 * integers;
            logUnary[node].resize(3 * integers);
            const std::size_t piece = problem.piecesByNode[node].front();
            const std::array orientationPrior{
                orientationBeliefs.horizontalProbability[piece],
                orientationBeliefs.mixedProbability[piece],
                orientationBeliefs.verticalProbability[piece],
            };
            for (std::size_t integer = 0; integer < integers; ++integer) {
                for (std::size_t orientation = 0; orientation < 3; ++orientation) {
                    logUnary[node][3 * integer + orientation] = std::log(
                        std::max(orientationPrior[orientation], kEpsilon));
                }
            }
            if (gauge[node] != 0) {
                std::fill(
                    logUnary[node].begin(),
                    logUnary[node].end(),
                    -std::numeric_limits<double>::infinity());
                logUnary[node][0] = 0.0;
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
                const auto a = jointState(stateA, lower[current.a]);
                const auto b = jointState(stateB, lower[current.b]);
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
            const std::size_t integers = probabilities.size() / 3;
            std::vector<double> windingProbability(integers, 0.0);
            for (std::size_t integer = 0; integer < integers; ++integer) {
                windingProbability[integer] = probabilities[3 * integer] +
                    probabilities[3 * integer + 1] +
                    probabilities[3 * integer + 2];
            }
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
            const auto a = jointState(stateA, round.lower[current.a]);
            for (std::size_t stateB = 0;
                 stateB < round.discrete.aToB[edge].size();
                 ++stateB) {
                const auto b = jointState(stateB, round.lower[current.b]);
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
            const auto a = jointState(stateA, round.lower[current.a]);
            for (std::size_t stateB = 0;
                 stateB < round.discrete.aToB[edge].size();
                 ++stateB) {
                const auto b = jointState(stateB, round.lower[current.b]);
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
        const double residual = term.signedDelta
            ? latent / scale - *term.signedDelta
            : latent;
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
        const auto addParallel = [&](double weight) {
            if (!(weight > 0.0))
                return;
            terms.push_back({
                belief.probability * weight,
                integer,
                phaseCoefficient,
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
            addParallel(measurement.parallel);
            if (measurement.signedDelta)
                addSigned(measurement.perpendicular, *measurement.signedDelta);
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
    const JointParameters& parameters,
    const FiberTraceInterleavedWindingConfig& config)
{
    std::vector<JointState> decoded(problem.piecesByNode.size());
    double energy = 0.0;
    for (std::size_t node = 0; node < decoded.size(); ++node) {
        const auto& probabilities = round.discrete.probabilities[node];
        const std::size_t state = static_cast<std::size_t>(std::distance(
            probabilities.begin(),
            std::max_element(probabilities.begin(), probabilities.end())));
        decoded[node] = jointState(state, round.lower[node]);
        const std::size_t piece = problem.piecesByNode[node].front();
        const double prior = decoded[node].orientation == JointClass::A
            ? orientationBeliefs.horizontalProbability[piece]
            : decoded[node].orientation == JointClass::Mixed
                ? orientationBeliefs.mixedProbability[piece]
                : orientationBeliefs.verticalProbability[piece];
        energy -= std::log(std::max(prior, kEpsilon));
    }
    for (std::size_t edge = 0; edge < problem.edges.size(); ++edge) {
        const auto& current = problem.edges[edge];
        const int sign = parameters.componentSign.at(
            problem.componentByNode[current.a]);
        energy -= config.temperature * jointLogPotential(
            current,
            decoded[current.a],
            decoded[current.b],
            sign,
            parameters.phase,
            parameters.scale,
            config.temperature);
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

}  // namespace

FiberTraceWindingBeliefPropagationReport solveFiberTraceWindingBeliefPropagation(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceWindingBeliefPropagationConfig& config)
{
    validateConfig(config);
    const auto prepared = prepareWinding(constraints, topology);
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
    for (const std::size_t node : prepared.gaugeNodeByComponent)
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
    report.continuousWinding.resize(pieceCount);
    report.mapWinding.resize(pieceCount);
    report.posteriorMeanWinding.resize(pieceCount);
    report.mapProbability.resize(pieceCount);
    report.entropy.resize(pieceCount);
    report.candidateMinimum.resize(pieceCount);
    report.candidateMaximum.resize(pieceCount);
    report.componentByPiece.resize(pieceCount);
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
    }
    return report;
}

FiberTraceInterleavedWindingReport
solveFiberTraceInterleavedWindingBeliefPropagation(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceBeliefPropagationReport& orientationBeliefs,
    const FiberTraceInterleavedWindingConfig& config,
    const FiberTraceInterleavedWindingProgressCallback& progress)
{
    const auto progressStarted = std::chrono::steady_clock::now();
    validateConfig(config);
    const std::size_t pieceCount = constraints.pieces.size();
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
    if (!std::isfinite(config.minimumMeasurementScale) ||
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
    const auto prepared = prepareWinding(constraints, topology);
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
            for (const std::size_t node : prepared.gaugeNodeByComponent)
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
                    prepared, candidate.round, candidate.parameters, config);
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
    report.variables = prepared.piecesByNode.size();
    report.factors = prepared.edges.size();
    report.connectedComponents = prepared.gaugeNodeByComponent.size();
    report.gaugePieces = prepared.gaugePieceByComponent;
    report.factorDiagnostics = prepared.diagnostics;
    report.continuousRootMeanSquareResidual = continuousResidual;
    report.temperature = config.temperature;
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

    report.continuousWinding.resize(pieceCount);
    report.mapWinding.resize(pieceCount);
    report.posteriorMeanWinding.resize(pieceCount);
    report.mapProbability.resize(pieceCount);
    report.entropy.resize(pieceCount);
    report.candidateMinimum.resize(pieceCount);
    report.candidateMaximum.resize(pieceCount);
    report.componentByPiece.resize(pieceCount);
    report.classAProbability.resize(pieceCount);
    report.mixedProbability.resize(pieceCount);
    report.classBProbability.resize(pieceCount);
    report.posteriorMeanLatentCoordinate.resize(pieceCount);
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
        const auto& probabilities = best->round.discrete.probabilities[node];
        const std::size_t map = static_cast<std::size_t>(std::distance(
            probabilities.begin(),
            std::max_element(probabilities.begin(), probabilities.end())));
        const JointState mapState = jointState(map, best->round.lower[node]);
        double meanWinding = 0.0;
        double meanLatent = 0.0;
        double entropy = 0.0;
        double classA = 0.0;
        double mixed = 0.0;
        double classB = 0.0;
        const int sign = best->parameters.componentSign.at(
            prepared.componentByNode[node]);
        for (std::size_t state = 0; state < probabilities.size(); ++state) {
            const double probability = probabilities[state];
            const auto current = jointState(state, best->round.lower[node]);
            meanWinding += probability * static_cast<double>(current.winding);
            double offset = classOffset(
                current.orientation, sign, best->parameters.phase);
            if (current.orientation == JointClass::Mixed)
                offset = 0.5 * static_cast<double>(sign) * best->parameters.phase;
            meanLatent += probability *
                (static_cast<double>(current.winding) + offset);
            if (current.orientation == JointClass::A)
                classA += probability;
            else if (current.orientation == JointClass::Mixed)
                mixed += probability;
            else
                classB += probability;
            if (probability > 0.0)
                entropy -= probability * std::log(probability);
        }
        report.continuousWinding[piece] = continuousNodes[node];
        report.mapWinding[piece] = mapState.winding;
        report.posteriorMeanWinding[piece] = meanWinding;
        report.posteriorMeanLatentCoordinate[piece] = meanLatent;
        report.mapProbability[piece] = probabilities[map];
        report.entropy[piece] = entropy;
        report.candidateMinimum[piece] = best->round.lower[node];
        report.candidateMaximum[piece] = best->round.upper[node];
        report.componentByPiece[piece] = prepared.componentByNode[node];
        const double classTotal = classA + mixed + classB;
        if (!std::isfinite(classTotal) || !(classTotal > 0.0))
            throw std::runtime_error("Interleaved winding class marginal is invalid");
        report.classAProbability[piece] = classA / classTotal;
        report.mixedProbability[piece] = mixed / classTotal;
        report.classBProbability[piece] = classB / classTotal;
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
