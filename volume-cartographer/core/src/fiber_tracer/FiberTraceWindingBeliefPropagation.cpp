#include "vc/fiber_tracer/FiberTraceWindingBeliefPropagation.hpp"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
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

struct DiscreteRound {
    std::vector<std::vector<double>> probabilities;
    std::size_t iterations = 0;
    double residual = 0.0;
    bool converged = false;
    std::size_t effectiveWorkers = 1;
};

DiscreteRound solveDiscreteRound(
    const PreparedWinding& problem,
    const std::vector<int>& lower,
    const std::vector<int>& upper,
    const FiberTraceWindingBeliefPropagationConfig& config)
{
    const std::size_t nodeCount = problem.piecesByNode.size();
    std::vector<std::vector<double>> aToB(problem.edges.size());
    std::vector<std::vector<double>> bToA(problem.edges.size());
    std::vector<std::vector<double>> nextAToB(problem.edges.size());
    std::vector<std::vector<double>> nextBToA(problem.edges.size());
    for (std::size_t edge = 0; edge < problem.edges.size(); ++edge) {
        const auto& current = problem.edges[edge];
        aToB[edge].assign(static_cast<std::size_t>(upper[current.b] - lower[current.b] + 1), 0.0);
        bToA[edge].assign(static_cast<std::size_t>(upper[current.a] - lower[current.a] + 1), 0.0);
        nextAToB[edge] = aToB[edge];
        nextBToA[edge] = bToA[edge];
    }
    std::vector<std::vector<double>> totals(nodeCount);
    DiscreteRound result;
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
            totals[node].assign(static_cast<std::size_t>(upper[node] - lower[node] + 1), 0.0);
            for (const std::size_t edge : problem.adjacency[node]) {
                const auto& current = problem.edges[edge];
                const auto& incoming = current.a == node
                    ? bToA[edge]
                    : aToB[edge];
                for (std::size_t state = 0; state < incoming.size(); ++state)
                    totals[node][state] += incoming[state];
            }
        }
        double residual = 0.0;
        #pragma omp parallel for schedule(static) num_threads(workers) if(useParallel) reduction(max : residual)
        for (std::size_t edge = 0; edge < problem.edges.size(); ++edge) {
            const auto& current = problem.edges[edge];
            std::vector<double> candidates;
            candidates.reserve(totals[current.a].size());
            for (std::size_t stateB = 0; stateB < aToB[edge].size(); ++stateB) {
                candidates.clear();
                const int labelB = lower[current.b] + static_cast<int>(stateB);
                for (std::size_t stateA = 0; stateA < bToA[edge].size(); ++stateA) {
                    const int labelA = lower[current.a] + static_cast<int>(stateA);
                    candidates.push_back(
                        totals[current.a][stateA] - bToA[edge][stateA] -
                        robustCost(current, labelA, labelB) / config.temperature);
                }
                nextAToB[edge][stateB] = logSumExp(candidates);
            }
            const double normAToB = logSumExp(nextAToB[edge]);
            for (std::size_t state = 0; state < nextAToB[edge].size(); ++state) {
                const double raw = nextAToB[edge][state] - normAToB;
                const double damped = aToB[edge][state] +
                    config.messageDamping * (raw - aToB[edge][state]);
                residual = std::max(residual, std::abs(damped - aToB[edge][state]));
                nextAToB[edge][state] = damped;
            }

            candidates.reserve(totals[current.b].size());
            for (std::size_t stateA = 0; stateA < bToA[edge].size(); ++stateA) {
                candidates.clear();
                const int labelA = lower[current.a] + static_cast<int>(stateA);
                for (std::size_t stateB = 0; stateB < aToB[edge].size(); ++stateB) {
                    const int labelB = lower[current.b] + static_cast<int>(stateB);
                    candidates.push_back(
                        totals[current.b][stateB] - aToB[edge][stateB] -
                        robustCost(current, labelA, labelB) / config.temperature);
                }
                nextBToA[edge][stateA] = logSumExp(candidates);
            }
            const double normBToA = logSumExp(nextBToA[edge]);
            for (std::size_t state = 0; state < nextBToA[edge].size(); ++state) {
                const double raw = nextBToA[edge][state] - normBToA;
                const double damped = bToA[edge][state] +
                    config.messageDamping * (raw - bToA[edge][state]);
                residual = std::max(residual, std::abs(damped - bToA[edge][state]));
                nextBToA[edge][state] = damped;
            }
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

    #pragma omp parallel for schedule(static) num_threads(workers) if(useParallel)
    for (std::size_t node = 0; node < nodeCount; ++node) {
        totals[node].assign(static_cast<std::size_t>(upper[node] - lower[node] + 1), 0.0);
        for (const std::size_t edge : problem.adjacency[node]) {
            const auto& current = problem.edges[edge];
            const auto& incoming = current.a == node
                ? bToA[edge]
                : aToB[edge];
            for (std::size_t state = 0; state < incoming.size(); ++state)
                totals[node][state] += incoming[state];
        }
    }
    result.probabilities.resize(nodeCount);
    for (std::size_t node = 0; node < nodeCount; ++node) {
        const double normalization = logSumExp(totals[node]);
        result.probabilities[node].resize(totals[node].size());
        for (std::size_t state = 0; state < totals[node].size(); ++state)
            result.probabilities[node][state] = std::exp(totals[node][state] - normalization);
    }
    return result;
}

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

}  // namespace vc::fiber_tracer
