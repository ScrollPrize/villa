#include "vc/fiber_tracer/FiberTraceWindingBeliefPropagation.hpp"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

#include <omp.h>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-12;

enum class JointClass : unsigned char { A = 0, Mixed = 1, B = 2 };

using Measurement = FiberTracePreparedWindingMeasurement;
using Edge = FiberTracePreparedWindingEdge;
using PreparedWinding = FiberTracePreparedWindingModel;

void validateConfig(const FiberTraceWindingBeliefPropagationConfig& config)
{
    const bool validDecisionConfidence =
        config.decisionConfidence ==
            FiberTraceWindingDecisionConfidence::Legacy ||
        config.decisionConfidence ==
            FiberTraceWindingDecisionConfidence::Linear ||
        config.decisionConfidence ==
            FiberTraceWindingDecisionConfidence::Cosine;
    const bool validNormalConfidence =
        config.normalConfidence == FiberTraceWindingNormalConfidence::None ||
        config.normalConfidence == FiberTraceWindingNormalConfidence::Linear ||
        config.normalConfidence == FiberTraceWindingNormalConfidence::Cosine;
    if (!validDecisionConfidence || !validNormalConfidence)
        throw std::invalid_argument("Winding BP confidence mode is invalid");
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
    if (config.finiteSignInfringementCost &&
        (!std::isfinite(*config.finiteSignInfringementCost) ||
         *config.finiteSignInfringementCost < 0.0)) {
        throw std::invalid_argument(
            "Winding BP finite sign cost must be finite and nonnegative");
    }
    if (config.hardSignMinimumNormalAlignment &&
        (!std::isfinite(*config.hardSignMinimumNormalAlignment) ||
         *config.hardSignMinimumNormalAlignment < 0.0 ||
         *config.hardSignMinimumNormalAlignment > 1.0)) {
        throw std::invalid_argument(
            "Winding BP hard-sign normal alignment must be in [0, 1]");
    }
    const std::array classWeights{
        config.perpendicularNextWeight,
        config.perpendicularFarWeight,
        config.parallelSameWeight,
        config.parallelOneWeight,
        config.parallelFarWeight,
        config.perpendicularSignWeight,
        config.parallelSignWeight,
    };
    if (std::any_of(classWeights.begin(), classWeights.end(), [](double weight) {
            return !std::isfinite(weight) || weight < 0.0;
        })) {
        throw std::invalid_argument(
            "Winding BP value and sign-hardness weights must be finite and nonnegative");
    }
}

bool hardSignPromotedByAlignment(
    const FiberTraceWindingBeliefPropagationConfig& config,
    const std::optional<double>& alignment)
{
    return config.hardSignMinimumNormalAlignment && alignment &&
        std::isfinite(*alignment) &&
        *alignment + kEpsilon >= *config.hardSignMinimumNormalAlignment;
}

double decisionConfidenceMultiplier(
    FiberTraceWindingDecisionConfidence mode,
    double selectedScore)
{
    const double score = std::clamp(selectedScore, 0.5, 1.0);
    if (mode == FiberTraceWindingDecisionConfidence::Legacy)
        return score;
    const double margin = std::clamp(2.0 * score - 1.0, 0.0, 1.0);
    if (mode == FiberTraceWindingDecisionConfidence::Linear)
        return margin;
    if (mode == FiberTraceWindingDecisionConfidence::Cosine)
        return 0.5 - 0.5 * std::cos(std::numbers::pi * margin);
    throw std::invalid_argument("Invalid winding decision-confidence mode");
}

double normalConfidenceMultiplier(
    FiberTraceWindingNormalConfidence mode,
    std::optional<double> absoluteAlignment)
{
    if (mode == FiberTraceWindingNormalConfidence::None)
        return 1.0;
    if (!absoluteAlignment || !std::isfinite(*absoluteAlignment))
        return 0.0;
    const double dot = std::clamp(*absoluteAlignment, 0.0, 1.0);
    if (mode == FiberTraceWindingNormalConfidence::Cosine)
        return dot;
    if (mode == FiberTraceWindingNormalConfidence::Linear) {
        return std::clamp(
            1.0 - 2.0 * std::acos(dot) / std::numbers::pi,
            0.0,
            1.0);
    }
    throw std::invalid_argument("Invalid winding normal-confidence mode");
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

double windingClassWeight(
    const FiberTraceWindingBeliefPropagationConfig& config,
    bool perpendicular,
    double effectiveTarget)
{
    constexpr double epsilon = 1.0e-12;
    const double distance = std::abs(effectiveTarget);
    if (perpendicular) {
        return distance <= 0.5 + epsilon
            ? config.perpendicularNextWeight
            : config.perpendicularFarWeight;
    }
    if (distance <= epsilon)
        return config.parallelSameWeight;
    return distance <= 1.0 + epsilon
        ? config.parallelOneWeight
        : config.parallelFarWeight;
}

Measurement materializeMeasurement(
    const Measurement& source,
    const FiberTraceWindingBeliefPropagationConfig& config,
    double measurementScale)
{
    if (!std::isfinite(measurementScale) || !(measurementScale > 0.0)) {
        throw std::invalid_argument(
            "Winding measurement scale must be finite and positive");
    }
    Measurement result = source;
    std::optional<double> signedParallel = source.rawParallelSignedDelta;
    std::optional<double> signedPerpendicular =
        source.rawPerpendicularSignedDelta;
    double parallelDistance = source.rawParallelDistance;
    if (!source.continuity) {
        parallelDistance *= measurementScale;
        if (signedParallel)
            *signedParallel *= measurementScale;
        if (signedPerpendicular)
            *signedPerpendicular *= measurementScale;
    }
    if (source.quantizeTargets && !source.continuity) {
        if (signedParallel) {
            signedParallel = quantizedIntegerWindingTarget(*signedParallel);
            parallelDistance = std::abs(*signedParallel);
        } else {
            parallelDistance = std::abs(
                quantizedIntegerWindingTarget(parallelDistance));
        }
        if (signedPerpendicular) {
            signedPerpendicular = quantizedHalfWindingTarget(
                *signedPerpendicular);
        }
    } else if (signedParallel) {
        parallelDistance = std::abs(*signedParallel);
    }

    const bool parallelRetained = source.parallelDominant &&
        (!config.parallelWindingDistanceCutoff || source.continuity ||
         parallelDistance < *config.parallelWindingDistanceCutoff);
    const bool perpendicularMagnitudePresent = source.perpendicularDominant &&
        signedPerpendicular.has_value();
    const double parallelMultiplier = source.quantizeTargets &&
            !source.continuity
        ? windingWeightMultiplier(parallelDistance) *
            windingClassWeight(config, false, parallelDistance)
        : 1.0;
    const double perpendicularMultiplier = signedPerpendicular &&
            source.quantizeTargets && !source.continuity
        ? windingWeightMultiplier(*signedPerpendicular) *
            windingClassWeight(config, true, *signedPerpendicular)
        : 1.0;
    const bool parallelSignPresent = !source.continuity &&
        source.parallelDominant && parallelRetained &&
        config.enforceParallelWindingSign && signedParallel &&
        *signedParallel != 0.0;
    const bool perpendicularSignPresent = !source.continuity &&
        source.perpendicularDominant &&
        config.enforcePerpendicularWindingSign && signedPerpendicular &&
        *signedPerpendicular != 0.0;
    const bool parallelSignEnabled = parallelSignPresent &&
        config.parallelSignWeight > 0.0;
    const bool perpendicularSignEnabled = perpendicularSignPresent &&
        config.perpendicularSignWeight > 0.0;
    const bool finiteSigns = config.finiteSignInfringementCost.has_value();
    const bool alignmentPromotesHardSign = hardSignPromotedByAlignment(
        config, source.selectedNormalAlignment);
    const bool strictSign = !finiteSigns || alignmentPromotesHardSign;

    result.parallelDistance = parallelDistance;
    result.parallelSignedDelta = source.parallelDominant && parallelRetained
        ? signedParallel
        : std::nullopt;
    result.perpendicularSignedDelta = source.perpendicularDominant
        ? signedPerpendicular
        : std::nullopt;
    result.parallelConfidence = source.continuity ||
            (source.parallelDominant && parallelRetained)
        ? source.selectedConfidence
        : 0.0;
    result.perpendicularConfidence = source.perpendicularDominant
        ? source.selectedConfidence
        : 0.0;
    result.parallelMultiplier = parallelMultiplier;
    result.perpendicularMultiplier = perpendicularMultiplier;
    result.hardParallelSign = parallelSignEnabled && strictSign;
    result.hardPerpendicularSign = perpendicularSignEnabled && strictSign;
    result.parallelSignPenalty = parallelSignEnabled && finiteSigns &&
            !alignmentPromotesHardSign
        ? *config.finiteSignInfringementCost * config.parallelSignWeight *
            source.selectedConfidence
        : 0.0;
    result.perpendicularSignPenalty = perpendicularSignEnabled && finiteSigns &&
            !alignmentPromotesHardSign
        ? *config.finiteSignInfringementCost * config.perpendicularSignWeight *
            source.selectedConfidence
        : 0.0;
    result.parallelMagnitudePresent = parallelRetained;
    result.perpendicularMagnitudePresent = perpendicularMagnitudePresent;
    result.parallelSignPresent = parallelSignPresent;
    result.perpendicularSignPresent = perpendicularSignPresent;
    return result;
}

FiberTraceReferenceConstraintGroup referenceConstraintGroup(
    bool perpendicular,
    double canonicalTarget)
{
    constexpr double epsilon = 1.0e-12;
    const double distance = std::abs(canonicalTarget);
    if (perpendicular) {
        return distance <= 0.5 + epsilon
            ? FiberTraceReferenceConstraintGroup::PerpendicularNext
            : FiberTraceReferenceConstraintGroup::PerpendicularFar;
    }
    if (distance <= epsilon)
        return FiberTraceReferenceConstraintGroup::ParallelSame;
    return distance <= 1.0 + epsilon
        ? FiberTraceReferenceConstraintGroup::ParallelOne
        : FiberTraceReferenceConstraintGroup::ParallelTwoPlus;
}

FiberTraceReferenceFactorClass referenceMagnitudeFactorClass(
    bool perpendicular,
    double canonicalTarget)
{
    switch (referenceConstraintGroup(perpendicular, canonicalTarget)) {
    case FiberTraceReferenceConstraintGroup::PerpendicularNext:
        return FiberTraceReferenceFactorClass::PerpendicularMagnitudeNext;
    case FiberTraceReferenceConstraintGroup::PerpendicularFar:
        return FiberTraceReferenceFactorClass::PerpendicularMagnitudeFar;
    case FiberTraceReferenceConstraintGroup::ParallelSame:
        return FiberTraceReferenceFactorClass::ParallelMagnitudeSame;
    case FiberTraceReferenceConstraintGroup::ParallelOne:
        return FiberTraceReferenceFactorClass::ParallelMagnitudeOne;
    case FiberTraceReferenceConstraintGroup::ParallelTwoPlus:
        return FiberTraceReferenceFactorClass::ParallelMagnitudeFar;
    case FiberTraceReferenceConstraintGroup::Count:
        break;
    }
    throw std::logic_error("Invalid reference constraint group");
}

double parallelWindingWeight(const Measurement& measurement)
{
    if (!measurement.parallelMagnitudePresent)
        return 0.0;
    return measurement.parallelMultiplier * measurement.parallelConfidence;
}

double parallelWindingResidual(
    const Measurement& measurement,
    double predictedDelta)
{
    return measurement.parallelSignedDelta
        ? predictedDelta - *measurement.parallelSignedDelta
        : std::abs(predictedDelta) - measurement.parallelDistance;
}

double perpendicularWindingWeight(const Measurement& measurement)
{
    return measurement.perpendicularMagnitudePresent &&
            measurement.perpendicularSignedDelta
        ? measurement.perpendicularMultiplier *
            measurement.perpendicularConfidence
        : 0.0;
}

double perpendicularWindingResidual(
    const Measurement& measurement,
    double predictedDelta)
{
    return predictedDelta - *measurement.perpendicularSignedDelta;
}

double finiteSignPenalty(
    const Measurement& measurement,
    double predictedDelta)
{
    double result = 0.0;
    if (measurement.parallelSignPenalty > 0.0 &&
        measurement.parallelSignedDelta &&
        *measurement.parallelSignedDelta * predictedDelta <= 0.0) {
        result += measurement.parallelSignPenalty;
    }
    if (measurement.perpendicularSignPenalty > 0.0 &&
        measurement.perpendicularSignedDelta &&
        *measurement.perpendicularSignedDelta * predictedDelta <= 0.0) {
        result += measurement.perpendicularSignPenalty;
    }
    return result;
}

PreparedWinding prepareWinding(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceWindingBeliefPropagationConfig& config,
    std::span<const JointClass> fixedOrientations = {},
    bool quantizeComponentTargets = false,
    double measurementScale = 1.0)
{
    if (topology.pieceLines.size() != constraints.pieces.size() ||
        topology.pieceCenterDistanceBaseVoxels.size() != constraints.pieces.size()) {
        throw std::invalid_argument("Winding BP topology does not match constraints");
    }
    PreparedWinding result;
    result.config = config;
    result.preparedMeasurementScale = measurementScale;
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
        if (!continuity && !fixedOrientations.empty() &&
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
        std::optional<double> canonicalSignedParallelDelta = continuity
            ? std::optional<double>{0.0}
            : constraint.signedParallelWindingDelta;
        if (canonicalSignedDelta && originalA > originalB)
            *canonicalSignedDelta = -*canonicalSignedDelta;
        if (canonicalSignedParallelDelta && originalA > originalB)
            *canonicalSignedParallelDelta = -*canonicalSignedParallelDelta;
        const bool perpendicularDominant = !continuity &&
            constraint.perpendicularScore >= constraint.parallelScore;
        const bool parallelDominant = !perpendicularDominant;
        const double selectedDecisionConfidence = continuity
            ? 1.0
            : decisionConfidenceMultiplier(
                  config.decisionConfidence,
                  perpendicularDominant
                      ? constraint.perpendicularScore
                      : constraint.parallelScore);
        const auto selectedAlignment = continuity
            ? std::optional<double>{1.0}
            : perpendicularDominant
                ? constraint.perpendicularNormalAlignment
                : constraint.parallelNormalAlignment;
        const double selectedNormalConfidence = continuity
            ? 1.0
            : normalConfidenceMultiplier(
                  config.normalConfidence, selectedAlignment);
        const double selectedConfidence =
            selectedDecisionConfidence * selectedNormalConfidence;
        const bool finiteSigns = config.finiteSignInfringementCost.has_value();
        const bool alignmentPromotesHardSign =
            hardSignPromotedByAlignment(config, selectedAlignment);

        Measurement rawMeasurement;
        rawMeasurement.constraintIndex = index;
        rawMeasurement.a = a;
        rawMeasurement.b = b;
        rawMeasurement.parallel = constraint.parallelScore;
        rawMeasurement.perpendicular = constraint.perpendicularScore;
        rawMeasurement.normalComponent = continuity
            ? std::nullopt
            : constraint.windingNormalComponent;
        rawMeasurement.rawParallelDistance = continuity
            ? 0.0
            : constraint.parallelWindingDistance;
        rawMeasurement.rawParallelSignedDelta =
            canonicalSignedParallelDelta;
        rawMeasurement.rawPerpendicularSignedDelta = canonicalSignedDelta;
        rawMeasurement.selectedNormalAlignment = selectedAlignment;
        rawMeasurement.selectedConfidence = selectedConfidence;
        rawMeasurement.continuity = continuity;
        rawMeasurement.quantizeTargets = quantizeComponentTargets;
        rawMeasurement.parallelDominant = parallelDominant;
        rawMeasurement.perpendicularDominant = perpendicularDominant;
        Measurement measurement = materializeMeasurement(
            rawMeasurement, config, measurementScale);
        const bool parallelSignEnabled = measurement.hardParallelSign ||
            measurement.parallelSignPenalty > 0.0;
        const bool perpendicularSignEnabled =
            measurement.hardPerpendicularSign ||
            measurement.perpendicularSignPenalty > 0.0;

        FiberTraceWindingFactorDiagnostic diagnostic;
        diagnostic.constraintIndex = index;
        diagnostic.pieceA = constraint.pieceA;
        diagnostic.pieceB = constraint.pieceB;
        diagnostic.canonicalNodeA = a;
        diagnostic.canonicalNodeB = b;
        diagnostic.parallelScore = constraint.parallelScore;
        diagnostic.perpendicularScore = constraint.perpendicularScore;
        diagnostic.parallelWindingWeightMultiplier =
            measurement.parallelMultiplier;
        diagnostic.perpendicularWindingWeightMultiplier =
            measurement.perpendicularMultiplier;
        diagnostic.effectiveParallelWindingWeight =
            parallelWindingWeight(measurement);
        diagnostic.effectivePerpendicularWindingWeight =
            perpendicularWindingWeight(measurement);
        diagnostic.decisionConfidenceMultiplier = selectedDecisionConfidence;
        diagnostic.normalConfidenceMultiplier = selectedNormalConfidence;
        diagnostic.effectiveParallelSignPenalty =
            measurement.parallelSignPenalty;
        diagnostic.effectivePerpendicularSignPenalty =
            measurement.perpendicularSignPenalty;
        diagnostic.parallelSignWeightMultiplier = config.parallelSignWeight;
        diagnostic.perpendicularSignWeightMultiplier =
            config.perpendicularSignWeight;
        diagnostic.perpendicularNormalAlignment =
            constraint.perpendicularNormalAlignment;
        diagnostic.parallelNormalAlignment = constraint.parallelNormalAlignment;
        diagnostic.originalSignedDelta = constraint.signedWindingDelta;
        diagnostic.canonicalSignedDelta = canonicalSignedDelta;
        diagnostic.effectiveParallelWindingDistance =
            measurement.parallelDistance;
        diagnostic.effectivePerpendicularSignedDelta = perpendicularDominant
            ? measurement.perpendicularSignedDelta
            : std::nullopt;
        diagnostic.normalComponent = continuity
            ? std::nullopt
            : constraint.windingNormalComponent;
        diagnostic.parallelWindingRetained =
            measurement.parallelMagnitudePresent;
        diagnostic.parallelMagnitudePresent =
            measurement.parallelMagnitudePresent;
        diagnostic.perpendicularMagnitudePresent =
            measurement.perpendicularMagnitudePresent;
        diagnostic.parallelSignPresent = measurement.parallelSignPresent;
        diagnostic.perpendicularSignPresent =
            measurement.perpendicularSignPresent;
        diagnostic.selfEdge = false;
        diagnostic.originalSignedParallelDelta =
            constraint.signedParallelWindingDelta;
        diagnostic.canonicalSignedParallelDelta =
            canonicalSignedParallelDelta;
        diagnostic.effectiveSignedParallelDelta = parallelDominant
            ? measurement.parallelSignedDelta
            : std::nullopt;
        diagnostic.hardParallelSign = measurement.hardParallelSign;
        diagnostic.hardPerpendicularSign =
            measurement.hardPerpendicularSign;
        diagnostic.parallelSignPromotedByAlignment = parallelSignEnabled &&
            finiteSigns && alignmentPromotesHardSign;
        diagnostic.perpendicularSignPromotedByAlignment =
            perpendicularSignEnabled && finiteSigns &&
            alignmentPromotesHardSign;
        result.diagnostics.push_back(std::move(diagnostic));
        const std::pair<std::size_t, std::size_t> key{a, b};
        const auto [found, inserted] = edgeByPair.try_emplace(
            key, result.edges.size());
        if (inserted)
            result.edges.push_back({a, b, {}});
        auto& edge = result.edges[found->second];
        edge.containsHardContinuity =
            edge.containsHardContinuity || continuity;
        edge.measurements.push_back(std::move(measurement));
    };
    for (const std::size_t index : topology.hardConstraintIndices)
        addConstraint(index, true);
    for (const std::size_t index : topology.softConstraintIndices)
        addConstraint(index, false);

    result.adjacency.resize(result.piecesByNode.size());
    result.incidentMeasurements.assign(result.piecesByNode.size(), 0);
    result.incidentWindingMeasurements.assign(result.piecesByNode.size(), 0);
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
                perpendicularWindingWeight(measurement) > 0.0 ||
                measurement.parallelSignPenalty > 0.0 ||
                measurement.perpendicularSignPenalty > 0.0 ||
                measurement.hardParallelSign ||
                measurement.hardPerpendicularSign;
            if (measurement.parallel > 0.0 ||
                measurement.perpendicular > 0.0) {
                ++result.incidentMeasurements[result.edges[edge].a];
                ++result.incidentMeasurements[result.edges[edge].b];
            }
            if (parallelWindingWeight(measurement) > 0.0 ||
                perpendicularWindingWeight(measurement) > 0.0 ||
                measurement.parallelSignPenalty > 0.0 ||
                measurement.perpendicularSignPenalty > 0.0 ||
                measurement.hardParallelSign ||
                measurement.hardPerpendicularSign) {
                ++result.incidentWindingMeasurements[result.edges[edge].a];
                ++result.incidentWindingMeasurements[result.edges[edge].b];
            }
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
                    const bool signedPerpendicular =
                        measurement.hardPerpendicularSign ||
                        measurement.perpendicularSignPenalty > 0.0 ||
                        (perpendicularWindingWeight(measurement) > 0.0 &&
                         measurement.perpendicularSignedDelta);
                    const bool signedParallel =
                        measurement.hardParallelSign ||
                        measurement.parallelSignPenalty > 0.0 ||
                        (parallelWindingWeight(measurement) > 0.0 &&
                         measurement.parallelSignedDelta);
                    if ((!signedPerpendicular && !signedParallel) ||
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
    const double parallelTarget =
        measurement.parallelSignedDelta.value_or(measurement.parallelDistance);
    return (parallelWindingWeight(measurement) * parallelTarget +
            perpendicularWindingWeight(measurement) *
                measurement.perpendicularSignedDelta.value_or(0.0)) /
        weight;
}

std::vector<double> solveContinuous(
    const PreparedWinding& problem,
    double& rootMeanSquareResidual,
    std::span<const std::optional<double>> fixedValues = {})
{
    const std::size_t nodeCount = problem.piecesByNode.size();
    if (!fixedValues.empty() && fixedValues.size() != nodeCount) {
        throw std::invalid_argument(
            "Fixed continuous winding values do not match nodes");
    }
    std::vector<unsigned char> gauge(nodeCount, 0);
    std::vector<unsigned char> preferredGauge(nodeCount, 0);
    for (const std::size_t node : problem.integerGaugeNodes)
        preferredGauge[node] = 1;
    std::vector<std::vector<std::size_t>> magnitudeAdjacency(nodeCount);
    for (const auto& edge : problem.edges) {
        const bool hasMagnitude = std::any_of(
            edge.measurements.begin(),
            edge.measurements.end(),
            [](const Measurement& measurement) {
                return measurementSquaredWeight(measurement) > 0.0;
            });
        if (hasMagnitude) {
            magnitudeAdjacency[edge.a].push_back(edge.b);
            magnitudeAdjacency[edge.b].push_back(edge.a);
        }
    }
    std::vector<unsigned char> visited(nodeCount, 0);
    for (std::size_t start = 0; start < nodeCount; ++start) {
        if (visited[start] != 0)
            continue;
        std::vector<std::size_t> nodes;
        std::queue<std::size_t> pending;
        pending.push(start);
        visited[start] = 1;
        while (!pending.empty()) {
            const std::size_t node = pending.front();
            pending.pop();
            nodes.push_back(node);
            for (const std::size_t neighbor : magnitudeAdjacency[node]) {
                if (visited[neighbor] == 0) {
                    visited[neighbor] = 1;
                    pending.push(neighbor);
                }
            }
        }
        bool anchored = false;
        if (!fixedValues.empty()) {
            for (const std::size_t node : nodes) {
                if (fixedValues[node]) {
                    gauge[node] = 1;
                    anchored = true;
                }
            }
        }
        if (!anchored) {
            const auto preferred = std::find_if(
                nodes.begin(), nodes.end(), [&](std::size_t node) {
                    return preferredGauge[node] != 0;
                });
            gauge[preferred == nodes.end() ? nodes.front() : *preferred] = 1;
        }
    }
    std::vector<double> values(nodeCount, 0.0);
    if (!fixedValues.empty()) {
        for (std::size_t node = 0; node < nodeCount; ++node) {
            if (fixedValues[node])
                values[node] = *fixedValues[node];
        }
    }
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
            rhs[static_cast<Eigen::Index>(variable[edge.a])] +=
                weight * (values[edge.b] - target);
        }
        if (variableB) {
            triplets.emplace_back(variable[edge.b], variable[edge.b], weight);
            rhs[static_cast<Eigen::Index>(variable[edge.b])] +=
                weight * (values[edge.a] + target);
        }
        if (variableA && variableB) {
            triplets.emplace_back(variable[edge.a], variable[edge.b], -weight);
            triplets.emplace_back(variable[edge.b], variable[edge.a], -weight);
        }
    }
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
            const double residual = parallelWindingResidual(
                measurement, delta);
            squared += parallelWeight * residual * residual;
            ++terms;
        }
        const double perpendicularWeight =
            perpendicularWindingWeight(measurement);
        if (measurement.perpendicularSignedDelta && perpendicularWeight > 0.0) {
            const double residual =
                perpendicularWindingResidual(measurement, delta);
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
            std::abs(parallelWindingResidual(measurement, delta));
        if (measurement.perpendicularSignedDelta) {
            cost += perpendicularWindingWeight(measurement) *
                std::abs(perpendicularWindingResidual(measurement, delta));
        }
        cost += finiteSignPenalty(measurement, delta);
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

struct FixedJointStates {
    std::vector<std::optional<JointState>> byNode;
    std::vector<std::optional<int>> phaseSignByComponent;
    std::vector<unsigned char> anchoredClassComponents;
    std::vector<unsigned char> anchoredIntegerComponents;
};

struct InitialJointStates {
    std::vector<JointState> byNode;
    std::vector<int> phaseSignByComponent;
    std::size_t defectGaugeComponents = 0;
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

double classOffset(JointClass orientation, int sign, double phase);

bool hardWindingSignCompatible(
    const Edge& edge,
    double predictedDelta,
    const FiberTraceWindingBeliefPropagationConfig& config,
    double measurementScale);

bool hasFixedCalibration(const FiberTraceJointGridWindingConfig& config);

FixedJointStates fixedJointStates(
    const PreparedWinding& problem,
    std::span<const FiberTraceFixedWindingState> fixedStates,
    std::span<const JointClass> fixedOrientations)
{
    FixedJointStates result;
    result.phaseSignByComponent.resize(problem.gaugeNodeByComponent.size());
    result.anchoredClassComponents.assign(
        problem.gaugeNodeByComponent.size(), 0);
    result.anchoredIntegerComponents.assign(
        problem.integerGaugeNodes.size(), 0);
    if (fixedStates.empty())
        return result;
    result.byNode.resize(problem.piecesByNode.size());
    if (fixedStates.size() != problem.pieceToNode.size()) {
        throw std::invalid_argument(
            "Fixed winding states do not match constraint pieces");
    }
    for (std::size_t piece = 0; piece < fixedStates.size(); ++piece) {
        const auto& fixed = fixedStates[piece];
        if (!fixed.fixed)
            continue;
        if (fixed.orientation == FiberTraceFixedOrientation::Mixed ||
            (fixed.componentPhaseSign != -1 &&
             fixed.componentPhaseSign != 1)) {
            throw std::invalid_argument("Fixed winding state is invalid");
        }
        const JointState state{jointClass(fixed.orientation), fixed.winding};
        const std::size_t node = problem.pieceToNode[piece];
        if (!fixedOrientations.empty() &&
            fixedOrientations[node] != state.orientation) {
            throw std::invalid_argument(
                "Fixed winding state disagrees with fixed orientation");
        }
        if (result.byNode[node] &&
            (result.byNode[node]->orientation != state.orientation ||
             result.byNode[node]->winding != state.winding)) {
            throw std::invalid_argument(
                "Constraint node has contradictory fixed winding states");
        }
        result.byNode[node] = state;
        const std::size_t component = problem.componentByNode[node];
        auto& sign = result.phaseSignByComponent[component];
        if (sign && *sign != fixed.componentPhaseSign) {
            throw std::invalid_argument(
                "Constraint component has contradictory fixed phase signs");
        }
        sign = fixed.componentPhaseSign;
        result.anchoredClassComponents[component] = 1;
        result.anchoredIntegerComponents[
            problem.integerGaugeByNode[node]] = 1;
    }
    return result;
}

InitialJointStates initialJointStates(
    const PreparedWinding& problem,
    std::span<const FiberTraceJointGridInitialState> initialStates,
    std::span<const JointClass> fixedOrientations,
    const FiberTraceJointGridWindingConfig& config)
{
    InitialJointStates result;
    if (initialStates.empty())
        return result;
    if (initialStates.size() != problem.pieceToNode.size()) {
        throw std::invalid_argument(
            "Initial winding states do not match constraint pieces");
    }
    if (fixedOrientations.empty()) {
        throw std::invalid_argument(
            "Joint-grid MAP initialization currently requires fixed orientations");
    }
    if (!hasFixedCalibration(config) ||
        std::abs(*config.fixedPhaseMagnitude - 0.5) > 1.0e-12) {
        throw std::invalid_argument(
            "Joint-grid MAP initialization requires fixed half-step phase");
    }
    result.byNode.assign(
        problem.piecesByNode.size(), {JointClass::Mixed, 0});
    std::vector<unsigned char> assigned(problem.piecesByNode.size(), 0);
    result.phaseSignByComponent.assign(
        problem.gaugeNodeByComponent.size(), 0);
    for (std::size_t piece = 0; piece < initialStates.size(); ++piece) {
        const auto& initial = initialStates[piece];
        if ((initial.componentPhaseSign != -1 &&
             initial.componentPhaseSign != 1) ||
            (initial.active &&
             initial.orientation == FiberTraceFixedOrientation::Mixed)) {
            throw std::invalid_argument("Initial winding state is invalid");
        }
        const std::size_t node = problem.pieceToNode[piece];
        const JointState state = initial.active
            ? JointState{jointClass(initial.orientation), initial.winding}
            : JointState{JointClass::Mixed, 0};
        if (initial.active && fixedOrientations[node] != state.orientation) {
            throw std::invalid_argument(
                "Initial winding state disagrees with fixed orientation");
        }
        if (assigned[node] != 0 &&
            (result.byNode[node].orientation != state.orientation ||
             result.byNode[node].winding != state.winding)) {
            throw std::invalid_argument(
                "Constraint node has contradictory initial winding states");
        }
        result.byNode[node] = state;
        assigned[node] = 1;
        const std::size_t component = problem.componentByNode[node];
        int& sign = result.phaseSignByComponent[component];
        if (sign != 0 && sign != initial.componentPhaseSign) {
            throw std::invalid_argument(
                "Constraint component has contradictory initial phase signs");
        }
        sign = initial.componentPhaseSign;
    }
    if (std::find(assigned.begin(), assigned.end(), 0) != assigned.end() ||
        std::find(
            result.phaseSignByComponent.begin(),
            result.phaseSignByComponent.end(), 0) !=
            result.phaseSignByComponent.end()) {
        throw std::invalid_argument("Initial winding state is incomplete");
    }

    for (std::size_t integerComponent = 0;
         integerComponent < problem.integerGaugeNodes.size();
         ++integerComponent) {
        const std::size_t gaugeNode =
            problem.integerGaugeNodes[integerComponent];
        std::optional<int> offset;
        if (result.byNode[gaugeNode].orientation != JointClass::Mixed)
            offset = result.byNode[gaugeNode].winding;
        if (!offset) {
            ++result.defectGaugeComponents;
            for (std::size_t node = 0; node < result.byNode.size(); ++node) {
                if (problem.integerGaugeByNode[node] == integerComponent &&
                    result.byNode[node].orientation != JointClass::Mixed) {
                    offset = result.byNode[node].winding;
                    break;
                }
            }
        }
        if (!offset)
            offset = 0;
        if (config.minimumActiveWinding) {
            int minimum = std::numeric_limits<int>::max();
            int maximum = std::numeric_limits<int>::lowest();
            for (std::size_t node = 0; node < result.byNode.size(); ++node) {
                if (problem.integerGaugeByNode[node] != integerComponent ||
                    result.byNode[node].orientation == JointClass::Mixed) {
                    continue;
                }
                minimum = std::min(minimum, result.byNode[node].winding);
                maximum = std::max(maximum, result.byNode[node].winding);
            }
            if (minimum <= maximum) {
                const int minimumOffset =
                    maximum - *config.maximumActiveWinding;
                const int maximumOffset =
                    minimum - *config.minimumActiveWinding;
                if (minimumOffset > maximumOffset) {
                    throw std::invalid_argument(
                        "Initial winding component does not fit the active range");
                }
                offset = std::clamp(*offset, minimumOffset, maximumOffset);
            }
        }
        for (std::size_t node = 0; node < result.byNode.size(); ++node) {
            if (problem.integerGaugeByNode[node] == integerComponent &&
                result.byNode[node].orientation != JointClass::Mixed) {
                result.byNode[node].winding -= *offset;
            }
        }
    }
    for (const auto& edge : problem.edges) {
        const auto& a = result.byNode[edge.a];
        const auto& b = result.byNode[edge.b];
        const bool active = a.orientation != JointClass::Mixed &&
            b.orientation != JointClass::Mixed;
        if (config.enforceHardSplitContinuity &&
            edge.containsHardContinuity && active &&
            (a.orientation != b.orientation || a.winding != b.winding)) {
            throw std::invalid_argument(
                "Initial winding states violate hard continuity");
        }
        if (active) {
            const std::size_t component = problem.componentByNode[edge.a];
            const int sign = result.phaseSignByComponent[component];
            const double predicted = static_cast<double>(
                    b.winding - a.winding) +
                classOffset(
                    b.orientation, sign, *config.fixedPhaseMagnitude) -
                classOffset(
                    a.orientation, sign, *config.fixedPhaseMagnitude);
            if (!hardWindingSignCompatible(
                    edge,
                    predicted,
                    config,
                    *config.fixedMeasurementScale)) {
                throw std::invalid_argument(
                    "Initial winding states violate a hard sign constraint");
            }
        }
    }
    return result;
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

bool hardPerpendicularSignCompatible(
    const Measurement& measurement,
    double predictedDelta)
{
    return !measurement.hardPerpendicularSign ||
        *measurement.perpendicularSignedDelta * predictedDelta > 0.0;
}

bool hardParallelSignCompatible(
    const Measurement& measurement,
    double predictedDelta)
{
    return !measurement.hardParallelSign ||
        *measurement.parallelSignedDelta * predictedDelta > 0.0;
}

bool requiresHardWindingSign(const Measurement& measurement)
{
    return measurement.hardPerpendicularSign ||
        measurement.hardParallelSign;
}

bool hardWindingSignCompatible(
    const Measurement& measurement,
    double predictedDelta)
{
    return hardPerpendicularSignCompatible(measurement, predictedDelta) &&
        hardParallelSignCompatible(measurement, predictedDelta);
}

bool hardWindingSignCompatible(
    const Edge& edge,
    double predictedDelta,
    const FiberTraceWindingBeliefPropagationConfig& config,
    double measurementScale)
{
    return std::all_of(
        edge.measurements.begin(),
        edge.measurements.end(),
        [&](const Measurement& measurement) {
            return hardWindingSignCompatible(
                materializeMeasurement(measurement, config, measurementScale),
                predictedDelta);
        });
}

double windingEnergy(
    const Edge& edge,
    const JointState& a,
    const JointState& b,
    int sign,
    double phase,
    double scale,
    const FiberTraceWindingBeliefPropagationConfig& config)
{
    const double delta = static_cast<double>(b.winding - a.winding) +
        classOffset(b.orientation, sign, phase) -
        classOffset(a.orientation, sign, phase);
    const double predictedDelta = delta;
    if (!hardWindingSignCompatible(edge, predictedDelta, config, scale))
        return std::numeric_limits<double>::infinity();
    double energy = 0.0;
    for (const auto& raw : edge.measurements) {
        const auto measurement = materializeMeasurement(raw, config, scale);
        energy += parallelWindingWeight(measurement) *
            std::abs(parallelWindingResidual(measurement, delta));
        if (measurement.perpendicularSignedDelta) {
            energy += perpendicularWindingWeight(measurement) *
                std::abs(perpendicularWindingResidual(
                    measurement, predictedDelta));
        }
        energy += finiteSignPenalty(measurement, predictedDelta);
    }
    return energy;
}

double pieceBreakLogPotential(
    const Edge& edge,
    const JointState& a,
    const JointState& b,
    double pieceBreakCost,
    double orientationTemperature)
{
    const bool defectA = a.orientation == JointClass::Mixed;
    const bool defectB = b.orientation == JointClass::Mixed;
    return edge.containsHardContinuity && defectA != defectB
        ? -pieceBreakCost / orientationTemperature
        : 0.0;
}

bool hardContinuityCompatible(
    const Edge& edge,
    const JointState& a,
    const JointState& b,
    bool enforceHardSplitContinuity)
{
    if (!enforceHardSplitContinuity || !edge.containsHardContinuity)
        return true;
    const bool defectA = a.orientation == JointClass::Mixed;
    const bool defectB = b.orientation == JointClass::Mixed;
    if (defectA || defectB)
        return true;
    return a.orientation == b.orientation && a.winding == b.winding;
}

double jointLogPotential(
    const Edge& edge,
    JointState a,
    JointState b,
    int sign,
    double phase,
    double scale,
    double temperature,
    double pieceBreakCost,
    double orientationTemperature,
    bool enforceHardSplitContinuity,
    const FiberTraceWindingBeliefPropagationConfig& config)
{
    if (!hardContinuityCompatible(
            edge, a, b, enforceHardSplitContinuity)) {
        return -std::numeric_limits<double>::infinity();
    }
    if (a.orientation == JointClass::Mixed ||
        b.orientation == JointClass::Mixed) {
        if (enforceHardSplitContinuity && edge.containsHardContinuity)
            return 0.0;
        return pieceBreakLogPotential(
            edge,
            a,
            b,
            pieceBreakCost,
            orientationTemperature);
    }
    return -windingEnergy(edge, a, b, sign, phase, scale, config) /
        temperature;
}

template <typename PredictedDelta>
std::size_t projectDecodedHardSigns(
    const PreparedWinding& problem,
    std::vector<JointState>& decoded,
    std::span<const double> activeConfidence,
    double measurementScale,
    const PredictedDelta& predictedDelta,
    std::span<const std::optional<JointState>> fixedStates = {})
{
    if (decoded.size() != problem.piecesByNode.size() ||
        activeConfidence.size() != decoded.size() ||
        (!fixedStates.empty() && fixedStates.size() != decoded.size())) {
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
                predictedDelta(edgeIndex, decoded[edge.a], decoded[edge.b]),
                problem.config,
                measurementScale)) {
            continue;
        }
        std::size_t disable = edge.b;
        const bool fixedA = !fixedStates.empty() && fixedStates[edge.a];
        const bool fixedB = !fixedStates.empty() && fixedStates[edge.b];
        if (fixedA && fixedB) {
            throw std::runtime_error(
                "Fixed winding states violate a hard sign constraint");
        } else if (fixedA) {
            disable = edge.b;
        } else if (fixedB) {
            disable = edge.a;
        } else if (gauge[edge.a] != 0 && gauge[edge.b] == 0) {
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

std::size_t projectDecodedHardContinuity(
    const PreparedWinding& problem,
    std::vector<JointState>& decoded,
    std::span<const double> activeConfidence,
    std::span<const std::optional<JointState>> fixedStates = {})
{
    if (decoded.size() != problem.piecesByNode.size() ||
        activeConfidence.size() != decoded.size() ||
        (!fixedStates.empty() && fixedStates.size() != decoded.size())) {
        throw std::logic_error(
            "Winding hard-continuity projection size mismatch");
    }
    std::vector<unsigned char> gauge(decoded.size(), 0);
    for (const std::size_t node : problem.gaugeNodeByComponent)
        gauge[node] = 1;
    std::size_t changed = 0;
    for (const auto& edge : problem.edges) {
        if (!edge.containsHardContinuity ||
            decoded[edge.a].orientation == JointClass::Mixed ||
            decoded[edge.b].orientation == JointClass::Mixed) {
            continue;
        }
        if (decoded[edge.a].orientation == decoded[edge.b].orientation &&
            decoded[edge.a].winding == decoded[edge.b].winding) {
            continue;
        }
        std::size_t disable = edge.b;
        const bool fixedA = !fixedStates.empty() && fixedStates[edge.a];
        const bool fixedB = !fixedStates.empty() && fixedStates[edge.b];
        if (fixedA && fixedB) {
            throw std::runtime_error(
                "Fixed winding states violate hard continuity");
        } else if (fixedA) {
            disable = edge.b;
        } else if (fixedB) {
            disable = edge.a;
        } else if (gauge[edge.a] != 0 && gauge[edge.b] == 0) {
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
        ++changed;
    }
    return changed;
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
                logUnary[node][0] = -config.mixedUnaryCost *
                    static_cast<double>(
                        problem.incidentWindingMeasurements[node]) /
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
                    config.temperature,
                    config.pieceBreakCost,
                    config.orientationTemperature,
                    config.enforceHardSplitContinuity,
                    config);
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
                        config.temperature,
                        config.pieceBreakCost,
                        config.orientationTemperature,
                        config.enforceHardSplitContinuity,
                        config));
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
    double probability = 0.0;
    double integer = 0.0;
    double phaseCoefficient = 0.0;
    Measurement measurement;
    bool perpendicular = false;
};

double calibrationL1(
    const std::vector<CalibrationTerm>& terms,
    double phase,
    double scale,
    const FiberTraceWindingBeliefPropagationConfig& config)
{
    double result = 0.0;
    for (const auto& term : terms) {
        const double latent =
            term.integer + term.phaseCoefficient * phase;
        const auto measurement = materializeMeasurement(
            term.measurement, config, scale);
        if (!hardWindingSignCompatible(measurement, latent)) {
            return std::numeric_limits<double>::infinity();
        }
        if (term.perpendicular) {
            if (measurement.perpendicularSignedDelta) {
                result += term.probability *
                    perpendicularWindingWeight(measurement) *
                    std::abs(perpendicularWindingResidual(
                        measurement, latent));
            }
        } else {
            result += term.probability * parallelWindingWeight(measurement) *
                std::abs(parallelWindingResidual(measurement, latent));
        }
        result += term.probability * finiteSignPenalty(measurement, latent);
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
                current.scale,
                config);
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
        const auto addParallel = [&](
            double weight,
            const Measurement& rawMeasurement) {
            terms.push_back({
                belief.probability,
                integer,
                phaseCoefficient,
                rawMeasurement,
                false,
            });
        };
        const auto addSigned = [&](
            double weight,
            double rawSignedDelta,
            const Measurement& rawMeasurement) {
            terms.push_back({
                belief.probability,
                integer,
                phaseCoefficient,
                rawMeasurement,
                true,
            });
            if (!(weight > 0.0))
                return;
            const double w = belief.probability * weight;
            h00 += w * integer * integer;
            h01 += w * integer * phaseCoefficient;
            h11 += w * phaseCoefficient * phaseCoefficient;
            rhs0 += w * integer * rawSignedDelta;
            rhs1 += w * phaseCoefficient * rawSignedDelta;
        };
        for (const auto& rawMeasurement : edge.measurements) {
            const auto measurement = materializeMeasurement(
                rawMeasurement, config, current.scale);
            if (rawMeasurement.parallelDominant ||
                rawMeasurement.continuity) {
                addParallel(
                    parallelWindingWeight(measurement),
                    rawMeasurement);
            }
            if (rawMeasurement.perpendicularDominant &&
                rawMeasurement.rawPerpendicularSignedDelta) {
                addSigned(
                    perpendicularWindingWeight(measurement),
                    *rawMeasurement.rawPerpendicularSignedDelta,
                    rawMeasurement);
            }
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
    const double oldLoss = calibrationL1(
        terms, current.phase, current.scale, config);
    for (double step = 1.0; step >= 1.0 / 1024.0; step *= 0.5) {
        const double phase = current.phase + step * (proposedPhase - current.phase);
        const double measurementScale =
            current.scale + step * (proposedScale - current.scale);
        if (calibrationL1(terms, phase, measurementScale, config) <=
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
        parameters.scale,
        [&](std::size_t edgeIndex, const JointState& a, const JointState& b) {
            const auto& edge = problem.edges[edgeIndex];
            const int sign = parameters.componentSign.at(
                problem.componentByNode[edge.a]);
            const double latent = static_cast<double>(b.winding - a.winding) +
                classOffset(b.orientation, sign, parameters.phase) -
                classOffset(a.orientation, sign, parameters.phase);
            return latent;
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
            energy += config.mixedUnaryCost * static_cast<double>(
                problem.incidentWindingMeasurements[node]) /
                config.orientationTemperature;
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
            config.temperature,
            config.pieceBreakCost,
            config.orientationTemperature,
            config.enforceHardSplitContinuity,
            config);
        const bool hasDefectEndpoint =
            decoded[current.a].orientation == JointClass::Mixed ||
            decoded[current.b].orientation == JointClass::Mixed;
        energy -= fixedOrientations.empty() && !hasDefectEndpoint
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
    std::vector<std::optional<JointState>> fixedStates;
    std::vector<std::optional<int>> fixedPhaseSigns;
    std::vector<JointState> initialStates;
    std::vector<int> initialPhaseSigns;
    std::vector<double> continuousNodes;
    FiberTraceJointGridInitializationMode initializationMode =
        FiberTraceJointGridInitializationMode::ConditionedMessages;
    std::vector<GridCalibrationCell> calibrationCells;
    GridCalibrationCell fixedCalibrationParameters;
    std::size_t iterations = 0;
    std::size_t gridShifts = 0;
    std::size_t supportChanges = 0;
    std::size_t totalStates = 0;
    std::size_t effectiveWorkers = 1;
    std::size_t initialDefectGaugeComponents = 0;
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
    std::span<const JointClass> fixedOrientations,
    std::span<const std::optional<JointState>> fixedStates)
{
    if (!fixedStates.empty() && fixedStates[node])
        return 1;
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
    std::span<const JointClass> fixedOrientations,
    std::span<const std::optional<JointState>> fixedStates)
{
    if (!fixedStates.empty() && fixedStates[node]) {
        if (state != 0)
            throw std::out_of_range("Fixed joint-grid state is invalid");
        return *fixedStates[node];
    }
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
    double phase,
    const FiberTraceJointGridWindingConfig& config)
{
    const double delta = static_cast<double>(b.winding - a.winding) +
        classOffset(b.orientation, sign, phase) -
        classOffset(a.orientation, sign, phase);
    const double measurementScale = 1.0 / gain;
    const double predictedDelta = delta;
    if (!hardWindingSignCompatible(
            edge, predictedDelta, config, measurementScale))
        return std::numeric_limits<double>::infinity();
    const auto minimumPositiveDelta = [&] (const double targetSign) {
        const double offset = targetSign * (
            classOffset(b.orientation, sign, phase) -
            classOffset(a.orientation, sign, phase));
        double fractional = offset - std::floor(offset);
        if (fractional <= 1.0e-12)
            fractional = 1.0;
        return fractional;
    };
    double result = 0.0;
    for (const auto& raw : edge.measurements) {
        const auto measurement = materializeMeasurement(
            raw, config, measurementScale);
        if (!raw.continuity) {
            result += config.localSpanCost *
                measurement.selectedConfidence * std::abs(predictedDelta);
        }
        if (requiresHardWindingSign(measurement)) {
            const double target = measurement.hardPerpendicularSign
                ? *measurement.perpendicularSignedDelta
                : *measurement.parallelSignedDelta;
            const double targetSign = std::copysign(1.0, target);
            const double signedDelta = targetSign * predictedDelta;
            result += config.hardSignSlackCost * std::max(
                0.0, signedDelta - minimumPositiveDelta(targetSign));
        }
        result += parallelWindingWeight(measurement) *
            std::abs(parallelWindingResidual(measurement, delta));
        if (measurement.perpendicularSignedDelta) {
            result += perpendicularWindingWeight(measurement) *
                std::abs(perpendicularWindingResidual(
                    measurement, predictedDelta));
        }
        result += finiteSignPenalty(measurement, predictedDelta);
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

double gridDecodedEnergy(
    const PreparedWinding& problem,
    std::span<const JointState> decoded,
    std::span<const int> phaseSigns,
    std::span<const JointClass> fixedOrientations,
    const GridCalibrationCell& calibration,
    const FiberTraceJointGridWindingConfig& config,
    std::span<const double> continuousNodes = {})
{
    if (!continuousNodes.empty() && continuousNodes.size() != decoded.size()) {
        throw std::logic_error(
            "Joint-grid continuous coordinate size mismatch");
    }
    double result = 0.0;
    const auto& incident = fixedOrientations.empty()
        ? problem.incidentMeasurements
        : problem.incidentWindingMeasurements;
    for (std::size_t node = 0; node < decoded.size(); ++node) {
        if (decoded[node].orientation == JointClass::Mixed &&
            (fixedOrientations.empty() ||
             fixedOrientations[node] != JointClass::Mixed)) {
            result += config.mixedUnaryCost *
                static_cast<double>(incident[node]);
        } else if (decoded[node].orientation != JointClass::Mixed &&
                   !continuousNodes.empty()) {
            result += config.continuousCoordinateCost *
                std::abs(
                    static_cast<double>(decoded[node].winding) -
                    std::round(continuousNodes[node]));
        }
    }
    for (const auto& edge : problem.edges) {
        const auto& a = decoded[edge.a];
        const auto& b = decoded[edge.b];
        if (a.orientation == JointClass::Mixed ||
            b.orientation == JointClass::Mixed) {
            const bool exactlyOneDefect =
                (a.orientation == JointClass::Mixed) !=
                (b.orientation == JointClass::Mixed);
            if (edge.containsHardContinuity && exactlyOneDefect)
                result += config.pieceBreakCost;
            continue;
        }
        if (fixedOrientations.empty()) {
            result += gridOrientationEnergy(
                edge, a.orientation, b.orientation);
        }
        result += gridWindingEnergy(
            edge,
            a,
            b,
            phaseSigns[problem.componentByNode[edge.a]],
            calibration.gain,
            calibration.phase,
            config);
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
    if (!hardContinuityCompatible(
            edge, a, b, config.enforceHardSplitContinuity)) {
        return -std::numeric_limits<double>::infinity();
    }
    if (a.orientation == JointClass::Mixed ||
        b.orientation == JointClass::Mixed) {
        if (config.enforceHardSplitContinuity && edge.containsHardContinuity)
            return 0.0;
        return pieceBreakLogPotential(
            edge,
            a,
            b,
            config.pieceBreakCost,
            config.orientationTemperature);
    }
    const double windingEnergy = gridWindingEnergy(
        edge,
        a,
        b,
        sign,
        calibration.gain,
        calibration.phase,
        config);
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
    const bool hasMinimumWinding = config.minimumActiveWinding.has_value();
    const bool hasMaximumWinding = config.maximumActiveWinding.has_value();
    const bool activeRangeInvalid = hasMinimumWinding != hasMaximumWinding ||
        (hasMinimumWinding &&
         (*config.minimumActiveWinding > *config.maximumActiveWinding ||
          *config.minimumActiveWinding > 0 ||
          *config.maximumActiveWinding < 0));
    if (!std::isfinite(config.mixedUnaryCost) || config.mixedUnaryCost < 0.0 ||
        !std::isfinite(config.pieceBreakCost) || config.pieceBreakCost < 0.0 ||
        !std::isfinite(config.continuousCoordinateCost) ||
        config.continuousCoordinateCost < 0.0 ||
        !std::isfinite(config.localSpanCost) || config.localSpanCost < 0.0 ||
        !std::isfinite(config.hardSignSlackCost) ||
        config.hardSignSlackCost < 0.0 ||
        !std::isfinite(config.orientationTemperature) ||
        !(config.orientationTemperature > 0.0) ||
        fixedInvalid || adaptiveInvalid || activeRangeInvalid ||
        config.stableIterations == 0) {
        throw std::invalid_argument("Joint-grid winding BP config is invalid");
    }
}

std::size_t jointGridStateCount(
    const PreparedWinding& problem,
    const GridRound& round)
{
    std::size_t result =
        round.fixedCalibration ? 0 : round.calibrationCells.size();
    for (const auto& sign : round.fixedPhaseSigns)
        result += sign ? 1 : 2;
    for (std::size_t node = 0; node < problem.piecesByNode.size(); ++node)
        result += gridPieceStateCount(
            node,
            round.lower,
            round.upper,
            round.gauge,
            round.fixedOrientations,
            round.fixedStates);
    for (const auto& edge : problem.edges) {
        result += gridPieceStateCount(
            edge.a,
            round.lower,
            round.upper,
            round.gauge,
            round.fixedOrientations,
            round.fixedStates);
        result += gridPieceStateCount(
            edge.b,
            round.lower,
            round.upper,
            round.gauge,
            round.fixedOrientations,
            round.fixedStates);
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
                round.fixedOrientations,
                round.fixedStates),
            0.0);
        message.toB.assign(
            gridPieceStateCount(
                edge.b,
                round.lower,
                round.upper,
                round.gauge,
                round.fixedOrientations,
                round.fixedStates),
            0.0);
        if (!round.fixedCalibration)
            message.toCalibration.assign(round.calibrationCells.size(), 0.0);
    }
}

void initializeGridMessagesFromState(
    const PreparedWinding& problem,
    GridRound& round,
    const FiberTraceJointGridWindingConfig& config)
{
    initializeGridMessages(problem, round);
    if (round.initialStates.empty() ||
        round.initializationMode ==
            FiberTraceJointGridInitializationMode::SupportOnly)
        return;
    if (!round.fixedCalibration ||
        round.initialStates.size() != problem.piecesByNode.size() ||
        round.initialPhaseSigns.size() !=
            problem.gaugeNodeByComponent.size()) {
        throw std::logic_error("Joint-grid initialization is inconsistent");
    }
    const auto normalizeSign = [](std::array<double, 2>& values) {
        const double normalization = logAddExp(values[0], values[1]);
        if (!std::isfinite(normalization)) {
            throw std::runtime_error(
                "Joint-grid initial sign message has no finite state");
        }
        values[0] -= normalization;
        values[1] -= normalization;
    };
    for (std::size_t edgeIndex = 0;
         edgeIndex < problem.edges.size(); ++edgeIndex) {
        const auto& edge = problem.edges[edgeIndex];
        auto& message = round.messages[edgeIndex];
        const JointState initialA = round.initialStates[edge.a];
        const JointState initialB = round.initialStates[edge.b];
        const int initialSign = round.initialPhaseSigns[
            problem.componentByNode[edge.a]];
        const auto& calibration = round.fixedCalibrationParameters;
        for (std::size_t state = 0; state < message.toA.size(); ++state) {
            message.toA[state] = gridLogPotential(
                edge,
                gridPieceState(
                    edge.a, state, round.lower, round.gauge,
                    round.fixedOrientations, round.fixedStates),
                initialB,
                initialSign,
                calibration,
                config,
                round.fixedOrientations.empty());
        }
        for (std::size_t state = 0; state < message.toB.size(); ++state) {
            message.toB[state] = gridLogPotential(
                edge,
                initialA,
                gridPieceState(
                    edge.b, state, round.lower, round.gauge,
                    round.fixedOrientations, round.fixedStates),
                initialSign,
                calibration,
                config,
                round.fixedOrientations.empty());
        }
        for (std::size_t signState = 0; signState < 2; ++signState) {
            message.toSign[signState] = gridLogPotential(
                edge,
                initialA,
                initialB,
                signState == 0 ? 1 : -1,
                calibration,
                config,
                round.fixedOrientations.empty());
        }
        normalizeLogVector(message.toA);
        normalizeLogVector(message.toB);
        normalizeSign(message.toSign);
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
    const auto& incident = round.fixedOrientations.empty()
        ? problem.incidentMeasurements
        : problem.incidentWindingMeasurements;
    for (std::size_t node = 0; node < problem.piecesByNode.size(); ++node) {
        const std::size_t stateCount = gridPieceStateCount(
            node,
            round.lower,
            round.upper,
            round.gauge,
            round.fixedOrientations,
            round.fixedStates);
        totals.pieceAccumulators[node].resize(stateCount);
        totals.pieceValues[node].resize(stateCount);
        for (std::size_t state = 0; state < stateCount; ++state) {
            const auto decoded = gridPieceState(
                node,
                state,
                round.lower,
                round.gauge,
                round.fixedOrientations,
                round.fixedStates);
            const bool chargeDefect =
                decoded.orientation == JointClass::Mixed &&
                (round.fixedOrientations.empty() ||
                 round.fixedOrientations[node] != JointClass::Mixed);
            double logUnary = chargeDefect
                ? -config.mixedUnaryCost *
                      static_cast<double>(incident[node]) /
                      config.orientationTemperature
                : 0.0;
            if (!chargeDefect &&
                decoded.orientation != JointClass::Mixed &&
                config.continuousCoordinateCost > 0.0) {
                logUnary -= config.continuousCoordinateCost *
                    std::abs(
                        static_cast<double>(decoded.winding) -
                        std::round(round.continuousNodes.at(node))) /
                    config.temperature;
            }
            addLogFactor(totals.pieceAccumulators[node][state], logUnary);
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
    for (std::size_t component = 0;
         component < round.fixedPhaseSigns.size();
         ++component) {
        if (!round.fixedPhaseSigns[component])
            continue;
        const std::size_t rejected = *round.fixedPhaseSigns[component] == 1
            ? 1
            : 0;
        addLogFactor(
            totals.signAccumulators[component][rejected],
            -std::numeric_limits<double>::infinity());
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
            round.fixedOrientations,
            round.fixedStates);
        for (std::size_t stateB = 0; stateB < cavityB.size(); ++stateB) {
            const auto b = gridPieceState(
                edge.b,
                stateB,
                round.lower,
                round.gauge,
                round.fixedOrientations,
                round.fixedStates);
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
    if (!round.fixedStates.empty() && round.fixedStates[node])
        return;
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
    std::span<const double> continuousNodes,
    const FiberTraceJointGridWindingConfig& config)
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
        if (round.gauge[node] != 0 ||
            (!round.fixedStates.empty() && round.fixedStates[node]))
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
        if (config.minimumActiveWinding) {
            round.lower[node] = std::max(
                round.lower[node], *config.minimumActiveWinding);
            round.upper[node] = std::min(
                round.upper[node], *config.maximumActiveWinding);
        }
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
        if (round.gauge[node] != 0 ||
            (!round.fixedStates.empty() && round.fixedStates[node]))
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
                    round.fixedOrientations,
                    round.fixedStates).winding);
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
            meanWinding - static_cast<double>(oldLower) <= 0.75 &&
            (!config.minimumActiveWinding ||
             oldLower > *config.minimumActiveWinding)) {
            --round.lower[node];
        }
        if (normalizedUpperProbability > config.boundaryProbabilityThreshold &&
            static_cast<double>(oldUpper) - meanWinding <= 0.75 &&
            (!config.maximumActiveWinding ||
             oldUpper < *config.maximumActiveWinding)) {
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
    const FixedJointStates& fixedStates,
    const InitialJointStates& initialStates,
    FiberTraceJointGridInitializationMode initializationMode,
    const FiberTraceJointGridWindingConfig& config,
    const FiberTraceJointGridProgressCallback& progress,
    const std::chrono::steady_clock::time_point& started)
{
    GridRound round;
    round.fixedOrientations.assign(
        fixedOrientations.begin(), fixedOrientations.end());
    round.fixedStates = fixedStates.byNode;
    round.fixedPhaseSigns = fixedStates.phaseSignByComponent;
    round.initialStates = initialStates.byNode;
    round.initialPhaseSigns = initialStates.phaseSignByComponent;
    round.continuousNodes.assign(continuousNodes.begin(), continuousNodes.end());
    round.initializationMode = initializationMode;
    round.initialDefectGaugeComponents =
        initialStates.defectGaugeComponents;
    round.fixedCalibration = hasFixedCalibration(config);
    if (round.fixedCalibration)
        round.fixedMeasurementScale = *config.fixedMeasurementScale;
    round.gauge.assign(problem.piecesByNode.size(), 0);
    round.lower.assign(problem.piecesByNode.size(), 0);
    round.upper.assign(problem.piecesByNode.size(), 0);
    for (const std::size_t node : problem.integerGaugeNodes) {
        if (fixedStates.anchoredIntegerComponents[
                problem.integerGaugeByNode[node]] == 0) {
            round.gauge[node] = 1;
        }
    }
    for (const std::size_t node : problem.gaugeNodeByComponent) {
        if (fixedStates.anchoredClassComponents[
                problem.componentByNode[node]] == 0) {
            round.gauge[node] = 2;
        }
    }
    if (round.fixedCalibration) {
        round.fixedCalibrationParameters = fixedCalibrationCell(config);
    } else {
        const int halfGain = static_cast<int>(config.initialGainCells / 2);
        round.calibrationCells = makeCalibrationCells(-halfGain, halfGain, config);
    }
    for (std::size_t node = 0; node < problem.piecesByNode.size(); ++node) {
        if (!round.fixedStates.empty() && round.fixedStates[node]) {
            if (config.minimumActiveWinding &&
                round.fixedStates[node]->orientation != JointClass::Mixed &&
                (round.fixedStates[node]->winding <
                     *config.minimumActiveWinding ||
                 round.fixedStates[node]->winding >
                     *config.maximumActiveWinding)) {
                throw std::invalid_argument(
                    "Fixed winding state is outside the active component range");
            }
            round.lower[node] = round.fixedStates[node]->winding;
            round.upper[node] = round.fixedStates[node]->winding;
            continue;
        }
        if (round.gauge[node] != 0)
            continue;
        double low = std::numeric_limits<double>::infinity();
        double high = -std::numeric_limits<double>::infinity();
        if (round.fixedCalibration) {
            low = continuousNodes[node];
            high = low;
        } else {
            low = continuousNodes[node];
            high = continuousNodes[node];
        }
        round.lower[node] = static_cast<int>(std::floor(low)) - 1;
        round.upper[node] = static_cast<int>(std::ceil(high)) + 1;
        if (config.minimumActiveWinding) {
            round.lower[node] = std::max(
                round.lower[node], *config.minimumActiveWinding);
            round.upper[node] = std::min(
                round.upper[node], *config.maximumActiveWinding);
            if (round.lower[node] > round.upper[node]) {
                const int nearest = std::clamp(
                    static_cast<int>(std::round(continuousNodes[node])),
                    *config.minimumActiveWinding,
                    *config.maximumActiveWinding);
                round.lower[node] = nearest;
                round.upper[node] = nearest;
            }
        }
        if (!round.initialStates.empty() &&
            round.initialStates[node].orientation != JointClass::Mixed) {
            if (config.minimumActiveWinding &&
                (round.initialStates[node].winding <
                     *config.minimumActiveWinding ||
                 round.initialStates[node].winding >
                     *config.maximumActiveWinding)) {
                throw std::invalid_argument(
                    "Initial winding state is outside the active component range");
            }
            round.lower[node] = std::min(
                round.lower[node], round.initialStates[node].winding);
            round.upper[node] = std::max(
                round.upper[node], round.initialStates[node].winding);
        }
    }
    initializeGridMessagesFromState(problem, round, config);
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
            ensureIntegerSupport(problem, round, continuousNodes, config);
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
    report.pieceBreakCost = config.pieceBreakCost;
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
    report.initializedFromState = !round.initialStates.empty();
    report.conditionedMessageInitialization =
        report.initializedFromState &&
        round.initializationMode ==
            FiberTraceJointGridInitializationMode::ConditionedMessages;
    report.initialDefectGaugeComponents =
        round.initialDefectGaugeComponents;
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
        const bool signedEvidence =
            diagnostic.hardPerpendicularSign || diagnostic.hardParallelSign ||
            diagnostic.effectivePerpendicularSignPenalty > 0.0 ||
            diagnostic.effectiveParallelSignPenalty > 0.0 ||
            (diagnostic.effectivePerpendicularWindingWeight > 0.0 &&
             diagnostic.effectivePerpendicularSignedDelta.has_value()) ||
            (diagnostic.effectiveParallelWindingWeight > 0.0 &&
             diagnostic.effectiveSignedParallelDelta.has_value());
        const bool expected =
            diagnostic.effectivePerpendicularWindingWeight > 0.0 ||
            diagnostic.effectiveParallelWindingWeight > 0.0 ||
            diagnostic.effectivePerpendicularSignPenalty > 0.0 ||
            diagnostic.effectiveParallelSignPenalty > 0.0 ||
            diagnostic.hardPerpendicularSign || diagnostic.hardParallelSign;
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
            node,
            map,
            round.lower,
            round.gauge,
            round.fixedOrientations,
            round.fixedStates);
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
                round.fixedOrientations,
                round.fixedStates);
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
                round.fixedOrientations,
                round.fixedStates);
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
    const auto predictedDelta = [&] (
        std::size_t edgeIndex, const JointState& a, const JointState& b) {
            const auto& edge = prepared.edges[edgeIndex];
            const std::size_t component = prepared.componentByNode[edge.a];
            const int sign = report.componentPhaseSign[component];
            const double latent = static_cast<double>(b.winding - a.winding) +
                classOffset(b.orientation, sign, selectedCalibration.phase) -
                classOffset(a.orientation, sign, selectedCalibration.phase);
            return latent;
        };
    report.hardSignProjectedDefects = projectDecodedHardSigns(
        prepared,
        decoded,
        activeConfidence,
        1.0 / selectedCalibration.gain,
        predictedDelta,
        round.fixedStates);
    if (config.enforceHardSplitContinuity) {
        projectDecodedHardContinuity(
            prepared, decoded, activeConfidence, round.fixedStates);
        report.hardSignProjectedDefects += projectDecodedHardSigns(
            prepared,
            decoded,
            activeConfidence,
            1.0 / selectedCalibration.gain,
            predictedDelta,
            round.fixedStates);
        projectDecodedHardContinuity(
            prepared, decoded, activeConfidence, round.fixedStates);
    }
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
            const auto& incident = round.fixedOrientations.empty()
                ? prepared.incidentMeasurements
                : prepared.incidentWindingMeasurements;
            report.decodedEnergy += config.mixedUnaryCost *
                static_cast<double>(incident[node]);
        }
    }
    for (const auto& edge : prepared.edges) {
        const std::size_t component = prepared.componentByNode[edge.a];
        const auto a = decoded[edge.a];
        const auto b = decoded[edge.b];
        if (a.orientation == JointClass::Mixed ||
            b.orientation == JointClass::Mixed) {
            const bool exactlyOneDefect =
                (a.orientation == JointClass::Mixed) !=
                (b.orientation == JointClass::Mixed);
            if (edge.containsHardContinuity && exactlyOneDefect)
                report.decodedEnergy += config.pieceBreakCost;
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
                selectedCalibration.phase,
                config);
            const double latentDelta = static_cast<double>(b.winding - a.winding) +
                classOffset(
                    b.orientation,
                    report.componentPhaseSign[component],
                    selectedCalibration.phase) -
                classOffset(
                    a.orientation,
                    report.componentPhaseSign[component],
                    selectedCalibration.phase);
            for (const auto& raw : edge.measurements) {
                if (raw.continuity)
                    continue;
                const auto measurement = materializeMeasurement(
                    raw, config, 1.0 / selectedCalibration.gain);
                report.localSpanEnergy += config.localSpanCost *
                    measurement.selectedConfidence * std::abs(latentDelta);
                if (requiresHardWindingSign(measurement)) {
                    const double target = measurement.hardPerpendicularSign
                        ? *measurement.perpendicularSignedDelta
                        : *measurement.parallelSignedDelta;
                    const double targetSign = std::copysign(1.0, target);
                    const double offset = targetSign * (
                        classOffset(
                            b.orientation,
                            report.componentPhaseSign[component],
                            selectedCalibration.phase) -
                        classOffset(
                            a.orientation,
                            report.componentPhaseSign[component],
                            selectedCalibration.phase));
                    double minimum = offset - std::floor(offset);
                    if (minimum <= 1.0e-12)
                        minimum = 1.0;
                    report.hardSignSlackEnergy += config.hardSignSlackCost *
                        std::max(0.0, targetSign * latentDelta - minimum);
                }
            }
        }
    }
    report.decodedDataEnergy = report.decodedEnergy -
        report.localSpanEnergy - report.hardSignSlackEnergy;
    for (std::size_t node = 0; node < decoded.size(); ++node) {
        if (decoded[node].orientation == JointClass::Mixed)
            continue;
        report.continuousCoordinateEnergy +=
            config.continuousCoordinateCost *
            std::abs(
                static_cast<double>(decoded[node].winding) -
                std::round(continuousNodes[node]));
    }
    report.decodedEnergy += report.continuousCoordinateEnergy;
    return report;
}

}  // namespace

FiberTracePreparedWindingModel prepareFiberTraceWindingModel(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceWindingBeliefPropagationConfig& config,
    std::span<const FiberTraceFixedOrientation> fixedOrientations,
    bool quantizeComponentTargets,
    double measurementScale)
{
    validateConfig(config);
    const auto fixedClasses = fixedJointClasses(
        fixedOrientations, constraints.pieces.size());
    return prepareWinding(
        constraints,
        topology,
        config,
        fixedClasses,
        quantizeComponentTargets,
        measurementScale);
}

std::vector<FiberTraceWindingFactorDiagnostic>
diagnoseFiberTraceWindingFactors(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceWindingBeliefPropagationConfig& config,
    std::span<const FiberTraceFixedOrientation> fixedOrientations,
    bool quantizeComponentTargets,
    double measurementScale)
{
    return prepareFiberTraceWindingModel(
               constraints,
               topology,
               config,
               fixedOrientations,
               quantizeComponentTargets,
               measurementScale)
        .diagnostics;
}

FiberTraceWindingComponentSelection selectLargestFiberTraceWindingComponent(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceWindingBeliefPropagationConfig& config,
    std::span<const FiberTraceFixedOrientation> fixedOrientations,
    bool quantizeComponentTargets,
    std::optional<std::size_t> preferredPiece,
    double measurementScale)
{
    validateConfig(config);
    if (preferredPiece && *preferredPiece >= constraints.pieces.size()) {
        throw std::invalid_argument("Preferred winding component piece is out of range");
    }
    const auto fixedClasses = fixedJointClasses(fixedOrientations, constraints.pieces.size());
    const auto prepared = prepareWinding(
        constraints,
        topology,
        config,
        fixedClasses,
        quantizeComponentTargets,
        measurementScale);

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

const char* fiberTraceConstraintAgreementClassName(
    FiberTraceConstraintAgreementClass agreementClass) noexcept
{
    switch (agreementClass) {
    case FiberTraceConstraintAgreementClass::Continuity:
        return "continuity";
    case FiberTraceConstraintAgreementClass::PerpendicularOrientation:
        return "perp_hv";
    case FiberTraceConstraintAgreementClass::PerpendicularMagnitudeNext:
        return "perp_winding_0.5";
    case FiberTraceConstraintAgreementClass::PerpendicularMagnitudeFar:
        return "perp_winding_1.5+";
    case FiberTraceConstraintAgreementClass::PerpendicularSign:
        return "perp_sign";
    case FiberTraceConstraintAgreementClass::ParallelOrientation:
        return "parallel_hv";
    case FiberTraceConstraintAgreementClass::ParallelMagnitudeSame:
        return "parallel_winding_0";
    case FiberTraceConstraintAgreementClass::ParallelMagnitudeOne:
        return "parallel_winding_1";
    case FiberTraceConstraintAgreementClass::ParallelMagnitudeFar:
        return "parallel_winding_2+";
    case FiberTraceConstraintAgreementClass::ParallelSign:
        return "parallel_sign";
    case FiberTraceConstraintAgreementClass::Count:
        break;
    }
    return "invalid";
}

FiberTraceConstraintAgreementSummary summarizeFiberTraceConstraintAgreement(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceInterleavedWindingReport& winding)
{
    const std::size_t pieceCount = constraints.pieces.size();
    if (winding.windingValid.size() != pieceCount ||
        winding.mapWinding.size() != pieceCount ||
        winding.mapLatentCoordinate.size() != pieceCount ||
        winding.mapOrientationByPiece.size() != pieceCount ||
        !std::isfinite(winding.measurementScale) ||
        !(winding.measurementScale > 0.0)) {
        throw std::invalid_argument(
            "Constraint agreement inputs do not match winding output");
    }
    FiberTraceConstraintAgreementSummary result;
    const auto add = [](FiberTraceConstraintAgreementCounts& target,
                        const FiberTraceConstraintAgreementCounts& value) {
        target.prepared += value.prepared;
        target.evaluated += value.evaluated;
        target.defectNeutralized += value.defectNeutralized;
        target.infringed += value.infringed;
    };
    const auto activeOrientation = [](FiberTraceFixedOrientation value) {
        return value == FiberTraceFixedOrientation::Horizontal ||
            value == FiberTraceFixedOrientation::Vertical;
    };
    for (const auto& diagnostic : winding.factorDiagnostics) {
        if (diagnostic.constraintIndex >= constraints.constraints.size() ||
            diagnostic.pieceA >= pieceCount ||
            diagnostic.pieceB >= pieceCount) {
            throw std::invalid_argument(
                "Constraint agreement diagnostic is invalid");
        }
        const auto& constraint =
            constraints.constraints[diagnostic.constraintIndex];
        const bool perpendicular = !constraint.hardContinuity &&
            diagnostic.perpendicularScore >= diagnostic.parallelScore;
        const std::size_t a = std::min(
            diagnostic.pieceA, diagnostic.pieceB);
        const std::size_t b = std::max(
            diagnostic.pieceA, diagnostic.pieceB);
        const bool activeA = winding.windingValid[a] != 0 &&
            activeOrientation(winding.mapOrientationByPiece[a]);
        const bool activeB = winding.windingValid[b] != 0 &&
            activeOrientation(winding.mapOrientationByPiece[b]);
        const bool active = activeA && activeB;
        const bool sameOrientation = active &&
            winding.mapOrientationByPiece[a] ==
                winding.mapOrientationByPiece[b];
        const double latentDelta = active
            ? winding.mapLatentCoordinate[b] -
                winding.mapLatentCoordinate[a]
            : 0.0;
        const auto emit = [&](FiberTraceConstraintAgreementClass itemClass,
                              bool infringed) {
            FiberTraceConstraintAgreementCounts counts;
            counts.prepared = 1;
            if (active) {
                counts.evaluated = 1;
                counts.infringed = infringed ? 1 : 0;
            } else {
                counts.defectNeutralized = 1;
            }
            add(result.classes[static_cast<std::size_t>(itemClass)], counts);
            add(result.total, counts);
        };
        if (constraint.hardContinuity) {
            emit(
                FiberTraceConstraintAgreementClass::Continuity,
                active && (!sameOrientation ||
                    winding.mapWinding[a] != winding.mapWinding[b]));
            continue;
        }
        emit(
            perpendicular
                ? FiberTraceConstraintAgreementClass::PerpendicularOrientation
                : FiberTraceConstraintAgreementClass::ParallelOrientation,
            active && (perpendicular ? sameOrientation : !sameOrientation));
        if (perpendicular && diagnostic.perpendicularMagnitudePresent) {
            const double target =
                *diagnostic.effectivePerpendicularSignedDelta;
            emit(
                std::abs(target) <= 0.5 + kEpsilon
                    ? FiberTraceConstraintAgreementClass::
                          PerpendicularMagnitudeNext
                    : FiberTraceConstraintAgreementClass::
                          PerpendicularMagnitudeFar,
                active && quantizedHalfWindingTarget(latentDelta) != target);
        }
        if (perpendicular && diagnostic.perpendicularSignPresent) {
            emit(
                FiberTraceConstraintAgreementClass::PerpendicularSign,
                active && *diagnostic.effectivePerpendicularSignedDelta *
                        latentDelta <= 0.0);
        }
        if (!perpendicular && diagnostic.parallelMagnitudePresent) {
            const double target =
                diagnostic.effectiveSignedParallelDelta.value_or(
                    diagnostic.effectiveParallelWindingDistance);
            const auto magnitudeClass = std::abs(target) <= kEpsilon
                ? FiberTraceConstraintAgreementClass::ParallelMagnitudeSame
                : std::abs(target) <= 1.0 + kEpsilon
                    ? FiberTraceConstraintAgreementClass::ParallelMagnitudeOne
                    : FiberTraceConstraintAgreementClass::ParallelMagnitudeFar;
            emit(
                magnitudeClass,
                active && (diagnostic.effectiveSignedParallelDelta
                    ? quantizedIntegerWindingTarget(latentDelta) != target
                    : std::abs(quantizedIntegerWindingTarget(latentDelta)) !=
                        target));
        }
        if (!perpendicular && diagnostic.parallelSignPresent) {
            emit(
                FiberTraceConstraintAgreementClass::ParallelSign,
                active && *diagnostic.effectiveSignedParallelDelta *
                        latentDelta <= 0.0);
        }
    }
    return result;
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

FiberTraceReferenceConstraintDiagnosticReport
makeFiberTraceReferenceConstraintDiagnosticReport(
    const FiberTraceConstraintReport& constraints,
    const std::vector<std::size_t>& sourceIdsByTrace,
    const FiberTraceReferenceWindingBenchmark& calibration)
{
    if (calibration.globalSign != -1 && calibration.globalSign != 1) {
        throw std::invalid_argument(
            "Reference winding calibration sign must be -1 or 1");
    }

    const auto ordered = orderMeasuredCrossSourceFiberTraceConstraints(
        constraints, sourceIdsByTrace);
    FiberTraceReferenceConstraintDiagnosticReport result;
    result.rows.reserve(ordered.size());
    for (const auto& link : ordered) {
        const auto& constraint =
            constraints.constraints.at(link.constraintIndex);
        const auto signedRaw = link.perpendicularDominant
            ? constraint.signedWindingDelta
            : constraint.signedParallelWindingDelta;
        double rawStep = link.perpendicularDominant
            ? constraint.windingDistance
            : constraint.parallelWindingDistance;
        if (signedRaw) {
            if (!std::isfinite(*signedRaw)) {
                throw std::invalid_argument(
                    "Reference constraint signed winding step is not finite");
            }
            const std::size_t traceA =
                constraints.pieces.at(constraint.pieceA).traceIndex;
            if (traceA >= sourceIdsByTrace.size()) {
                throw std::invalid_argument(
                    "Reference constraint piece source is out of range");
            }
            rawStep = sourceIdsByTrace[traceA] == link.ownerSource
                ? *signedRaw
                : -*signedRaw;
        } else if (!std::isfinite(rawStep) || rawStep < 0.0) {
            throw std::invalid_argument(
                "Reference constraint unsigned winding step is invalid");
        }

        const double calibratedStep = signedRaw
            ? static_cast<double>(calibration.globalSign) * rawStep
            : rawStep;
        const double canonicalStep = link.perpendicularDominant
            ? quantizedHalfWindingTarget(calibratedStep)
            : quantizedIntegerWindingTarget(calibratedStep);
        const double groundTruthStep = 0.5 * static_cast<double>(
            link.targetSource - link.ownerSource);
        result.counts.add(canonicalStep, groundTruthStep);
        result.rows.push_back({
            link.constraintIndex,
            link.ownerSource,
            link.targetSource,
            link.perpendicularDominant,
            signedRaw.has_value(),
            rawStep,
            calibratedStep,
            canonicalStep,
            groundTruthStep,
        });
    }
    return result;
}

static std::map<std::size_t, const FiberTraceWindingFactorDiagnostic*>
indexReferenceWindingFactors(
    std::span<const FiberTraceWindingFactorDiagnostic> factors)
{
    std::map<std::size_t, const FiberTraceWindingFactorDiagnostic*> result;
    for (const auto& factor : factors) {
        if (!result.emplace(factor.constraintIndex, &factor).second) {
            throw std::invalid_argument(
                "Reference winding factor diagnostics contain a duplicate constraint");
        }
    }
    return result;
}

FiberTraceReferenceScaleCalibrationReport
calibrateFiberTraceReferenceConstraintScales(
    const FiberTraceReferenceConstraintDiagnosticReport& reference,
    std::span<const FiberTraceWindingFactorDiagnostic> factors,
    double minimumScale,
    double maximumScale)
{
    if (!std::isfinite(minimumScale) ||
        !std::isfinite(maximumScale) ||
        !(minimumScale > 0.0) ||
        !(maximumScale >= minimumScale)) {
        throw std::invalid_argument(
            "Reference winding scale bounds must be finite and positive");
    }

    const auto factorByConstraint = indexReferenceWindingFactors(factors);

    struct ScaleObservation {
        double weight = 0.0;
        double truth = 0.0;
        double rawTarget = 0.0;
        double canonicalTarget = 0.0;
    };
    constexpr std::size_t groupCount = static_cast<std::size_t>(
        FiberTraceReferenceConstraintGroup::Count);
    std::vector<ScaleObservation> perpendicular;
    std::vector<ScaleObservation> parallel;
    std::vector<ScaleObservation> all;
    std::array<std::vector<ScaleObservation>, groupCount> grouped;
    perpendicular.reserve(reference.rows.size());
    parallel.reserve(reference.rows.size());
    all.reserve(reference.rows.size());
    for (const auto& row : reference.rows) {
        const auto found = factorByConstraint.find(row.constraintIndex);
        if (found == factorByConstraint.end()) {
            throw std::invalid_argument(
                "Reference winding row has no prepared factor diagnostic");
        }
        const auto& factor = *found->second;
        const double weight = row.perpendicularDominant
            ? factor.effectivePerpendicularWindingWeight
            : factor.effectiveParallelWindingWeight;
        if (!std::isfinite(weight) || weight < 0.0 ||
            !std::isfinite(row.groundTruthStep) ||
            !std::isfinite(row.calibratedStep) ||
            !std::isfinite(row.canonicalStep)) {
            throw std::invalid_argument(
                "Reference winding scale observation is invalid");
        }
        const ScaleObservation observation{
            weight,
            row.groundTruthStep,
            row.calibratedStep,
            row.canonicalStep,
        };
        const auto group = referenceConstraintGroup(
            row.perpendicularDominant, row.canonicalStep);
        grouped[static_cast<std::size_t>(group)].push_back(observation);
        all.push_back(observation);
        if (row.perpendicularDominant)
            perpendicular.push_back(observation);
        else
            parallel.push_back(observation);
    }

    const auto fit = [&](std::span<const ScaleObservation> observations,
                         bool rawTargets) {
        FiberTraceReferenceScaleFit result;
        result.observations = observations.size();
        struct MedianPoint {
            double reciprocalScale = 0.0;
            double weight = 0.0;
        };
        std::vector<MedianPoint> points;
        points.reserve(observations.size());
        const auto loss = [&](double scale) {
            double total = 0.0;
            for (const auto& observation : observations) {
                const double target = rawTargets
                    ? observation.rawTarget
                    : observation.canonicalTarget;
                total += observation.weight * std::abs(
                    observation.truth / scale - target);
            }
            return total;
        };
        for (const auto& observation : observations) {
            result.effectiveWeight += observation.weight;
            if (!(observation.weight > 0.0))
                continue;
            ++result.admittedObservations;
            const double medianWeight =
                observation.weight * std::abs(observation.truth);
            result.reciprocalScaleWeight += medianWeight;
            if (!(medianWeight > 0.0))
                continue;
            ++result.informativeObservations;
            const double target = rawTargets
                ? observation.rawTarget
                : observation.canonicalTarget;
            points.push_back({
                target / observation.truth,
                medianWeight,
            });
        }
        result.unitScaleLoss = loss(1.0);
        result.fittedLoss = result.unitScaleLoss;
        if (points.empty())
            return result;

        std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) {
            return a.reciprocalScale < b.reciprocalScale;
        });
        std::vector<MedianPoint> merged;
        merged.reserve(points.size());
        for (const auto& point : points) {
            if (!merged.empty() &&
                point.reciprocalScale == merged.back().reciprocalScale) {
                merged.back().weight += point.weight;
            } else {
                merged.push_back(point);
            }
        }

        const double halfWeight = 0.5 * result.reciprocalScaleWeight;
        double cumulative = 0.0;
        double medianMinimum = merged.back().reciprocalScale;
        double medianMaximum = medianMinimum;
        for (std::size_t index = 0; index < merged.size(); ++index) {
            cumulative += merged[index].weight;
            if (cumulative < halfWeight)
                continue;
            medianMinimum = merged[index].reciprocalScale;
            const double tolerance = 16.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, result.reciprocalScaleWeight);
            if (std::abs(cumulative - halfWeight) <= tolerance &&
                index + 1 < merged.size()) {
                medianMaximum = merged[index + 1].reciprocalScale;
            } else {
                medianMaximum = medianMinimum;
            }
            break;
        }

        const double minimumReciprocal = 1.0 / maximumScale;
        const double maximumReciprocal = 1.0 / minimumScale;
        const double preferredReciprocal = std::clamp(
            1.0, medianMinimum, medianMaximum);
        const double reciprocalScale = std::clamp(
            preferredReciprocal, minimumReciprocal, maximumReciprocal);
        const double scale = 1.0 / reciprocalScale;
        result.fittedScale = scale;
        result.fittedLoss = loss(scale);
        const double boundTolerance = 16.0 *
            std::numeric_limits<double>::epsilon() *
            std::max(1.0, maximumScale);
        result.atLowerBound =
            std::abs(scale - minimumScale) <= boundTolerance;
        result.atUpperBound =
            std::abs(scale - maximumScale) <= boundTolerance;
        return result;
    };

    FiberTraceReferenceScaleCalibrationReport result;
    result.minimumScale = minimumScale;
    result.maximumScale = maximumScale;
    result.rawPerpendicular = fit(perpendicular, true);
    result.canonicalPerpendicular = fit(perpendicular, false);
    result.rawParallel = fit(parallel, true);
    result.canonicalParallel = fit(parallel, false);
    result.rawAll = fit(all, true);
    result.canonicalAll = fit(all, false);
    for (std::size_t group = 0; group < groupCount; ++group) {
        result.rawGroups[group] = fit(grouped[group], true);
        result.canonicalGroups[group] = fit(grouped[group], false);
    }
    return result;
}

FiberTraceReferencePhaseCalibrationReport
calibrateFiberTraceReferenceConstraintPhase(
    const FiberTraceReferenceConstraintDiagnosticReport& reference,
    double measurementScale)
{
    if (!std::isfinite(measurementScale) || !(measurementScale > 0.0)) {
        throw std::invalid_argument(
            "Reference winding phase scale must be finite and positive");
    }

    struct Observation {
        std::size_t ownerSource = 0;
        std::size_t targetSource = 0;
        double target = 0.0;
    };
    std::vector<Observation> observations;
    observations.reserve(reference.rows.size());
    std::size_t identifyingRows = 0;
    std::size_t perpendicularSameParityRows = 0;
    std::size_t parallelSameParityRows = 0;
    std::size_t parallelOppositeParityRows = 0;
    for (const auto& row : reference.rows) {
        if (row.ownerSource >= row.targetSource ||
            !std::isfinite(row.rawStep)) {
            throw std::invalid_argument(
                "Reference winding phase observation is invalid");
        }
        const bool oppositeParity =
            (row.ownerSource & 1U) != (row.targetSource & 1U);
        if (!row.perpendicularDominant) {
            if (oppositeParity)
                ++parallelOppositeParityRows;
            else
                ++parallelSameParityRows;
            continue;
        }
        if (!oppositeParity) {
            ++perpendicularSameParityRows;
            continue;
        }
        ++identifyingRows;
        if (!row.signedMeasurement)
            continue;
        observations.push_back({
            row.ownerSource,
            row.targetSource,
            row.rawStep,
        });
    }

    const auto coordinate = [](
                                std::size_t source,
                                bool evenReferenceIsHorizontal,
                                int direction,
                                double phase) {
        const double half = 0.5 * static_cast<double>(source);
        const bool odd = (source & 1U) != 0;
        const double value = evenReferenceIsHorizontal
            ? std::floor(half) + (odd ? phase : 0.0)
            : std::ceil(half) - (odd ? phase : 0.0);
        return static_cast<double>(direction) * value;
    };
    const auto predicted = [&](const Observation& observation,
                               bool evenReferenceIsHorizontal,
                               int direction,
                               double phase) {
        return (
                   coordinate(
                       observation.targetSource,
                       evenReferenceIsHorizontal,
                       direction,
                       phase) -
                   coordinate(
                       observation.ownerSource,
                       evenReferenceIsHorizontal,
                       direction,
                       phase)) /
            measurementScale;
    };

    FiberTraceReferencePhaseCalibrationReport result;
    result.measurementScale = measurementScale;
    std::size_t gaugeIndex = 0;
    for (const int direction : {1, -1}) {
        for (const bool evenReferenceIsHorizontal : {true, false}) {
            auto& fit = result.gauges[gaugeIndex++];
            fit.windingDirection = direction;
            fit.evenReferenceIsHorizontal = evenReferenceIsHorizontal;
            fit.totalRows = reference.rows.size();
            fit.identifyingRows = identifyingRows;
            fit.usedRows = observations.size();
            fit.perpendicularSameParityRows =
                perpendicularSameParityRows;
            fit.parallelSameParityRows = parallelSameParityRows;
            fit.parallelOppositeParityRows = parallelOppositeParityRows;
            fit.effectiveWeight = static_cast<double>(observations.size());
            const auto loss = [&](double phase) {
                double total = 0.0;
                for (const auto& observation : observations) {
                    total += std::abs(
                        predicted(
                            observation,
                            evenReferenceIsHorizontal,
                            direction,
                            phase) -
                        observation.target);
                }
                return total;
            };
            fit.lossAtZero = loss(0.0);
            fit.lossAtHalf = loss(0.5);
            fit.fittedLoss = fit.lossAtZero;
            if (observations.empty() || !(fit.effectiveWeight > 0.0))
                continue;

            std::vector<double> candidates{0.0, 0.5};
            candidates.reserve(observations.size() + 2);
            for (const auto& observation : observations) {
                const double atZero = predicted(
                    observation,
                    evenReferenceIsHorizontal,
                    direction,
                    0.0);
                const double slope = 2.0 * (
                    predicted(
                        observation,
                        evenReferenceIsHorizontal,
                        direction,
                        0.5) -
                    atZero);
                if (slope == 0.0)
                    continue;
                const double breakpoint =
                    (observation.target - atZero) / slope;
                if (breakpoint >= 0.0 && breakpoint <= 0.5)
                    candidates.push_back(breakpoint);
            }
            std::sort(candidates.begin(), candidates.end());
            candidates.erase(
                std::unique(candidates.begin(), candidates.end()),
                candidates.end());
            double bestPhase = candidates.front();
            double bestLoss = loss(bestPhase);
            for (const double phase : candidates) {
                const double currentLoss = loss(phase);
                const double tolerance = 32.0 *
                    std::numeric_limits<double>::epsilon() *
                    std::max({1.0, std::abs(bestLoss), std::abs(currentLoss)});
                if (currentLoss < bestLoss - tolerance ||
                    (std::abs(currentLoss - bestLoss) <= tolerance &&
                     phase < bestPhase)) {
                    bestPhase = phase;
                    bestLoss = currentLoss;
                }
            }
            fit.fittedPhase = bestPhase;
            fit.fittedLoss = bestLoss;
            for (const auto& observation : observations) {
                const double delta = predicted(
                    observation,
                    evenReferenceIsHorizontal,
                    direction,
                    bestPhase);
                fit.fittedSignDisagreements +=
                    observation.target * delta <= 0.0 ? 1 : 0;
            }
        }
    }

    const auto better = [](const FiberTraceReferencePhaseFit& candidate,
                           const FiberTraceReferencePhaseFit& current) {
        const double tolerance = 32.0 *
            std::numeric_limits<double>::epsilon() *
            std::max({
                1.0,
                std::abs(candidate.fittedLoss),
                std::abs(current.fittedLoss),
            });
        if (candidate.fittedLoss < current.fittedLoss - tolerance)
            return true;
        if (current.fittedLoss < candidate.fittedLoss - tolerance)
            return false;
        if (*candidate.fittedPhase != *current.fittedPhase)
            return *candidate.fittedPhase < *current.fittedPhase;
        if (candidate.windingDirection != current.windingDirection)
            return candidate.windingDirection > current.windingDirection;
        return candidate.evenReferenceIsHorizontal &&
            !current.evenReferenceIsHorizontal;
    };
    for (std::size_t index = 0; index < result.gauges.size(); ++index) {
        if (!result.gauges[index].fittedPhase)
            continue;
        if (!result.selectedGauge ||
            better(result.gauges[index], result.gauges[*result.selectedGauge])) {
            result.selectedGauge = index;
        }
    }
    return result;
}

FiberTraceReferenceStepStatisticsReport
summarizeFiberTraceReferenceConstraintSteps(
    const FiberTraceReferenceConstraintDiagnosticReport& reference)
{
    using Values = std::array<std::array<std::array<std::array<
        std::vector<double>, 3>, 2>, 2>, 2>;
    Values values;
    for (const auto& row : reference.rows) {
        if (row.ownerSource >= row.targetSource ||
            !std::isfinite(row.rawStep) ||
            !std::isfinite(row.groundTruthStep) ||
            !(row.groundTruthStep > 0.0)) {
            throw std::invalid_argument(
                "Reference winding step statistic row is invalid");
        }
        if (!row.signedMeasurement)
            continue;
        const std::size_t relation = row.perpendicularDominant ? 0 : 1;
        const std::size_t ownerParity = row.ownerSource & 1U;
        const std::size_t targetParity = row.targetSource & 1U;
        const bool oppositeParity = ownerParity != targetParity;
        std::size_t band = 2;
        if (oppositeParity) {
            if (row.groundTruthStep < 1.0)
                band = 0;
            else if (row.groundTruthStep < 2.0)
                band = 1;
        } else {
            if (row.groundTruthStep < 1.5)
                band = 0;
            else if (row.groundTruthStep < 2.5)
                band = 1;
        }
        values[relation][ownerParity][targetParity][band].push_back(
            row.rawStep);
    }

    FiberTraceReferenceStepStatisticsReport result;
    for (std::size_t relation = 0; relation < 2; ++relation) {
        for (std::size_t owner = 0; owner < 2; ++owner) {
            for (std::size_t target = 0; target < 2; ++target) {
                for (std::size_t band = 0; band < 3; ++band) {
                    auto& source = values[relation][owner][target][band];
                    auto& destination =
                        result.groups[relation][owner][target][band];
                    if (source.empty())
                        continue;
                    std::sort(source.begin(), source.end());
                    destination.observations = source.size();
                    destination.minimum = source.front();
                    destination.maximum = source.back();
                    destination.mean = std::accumulate(
                        source.begin(), source.end(), 0.0) /
                        static_cast<double>(source.size());
                    const std::size_t middle = source.size() / 2;
                    destination.median = source.size() % 2 != 0
                        ? source[middle]
                        : 0.5 * (source[middle - 1] + source[middle]);
                }
            }
        }
    }
    return result;
}

namespace
{

struct ReferenceScoredObservation {
    const FiberTraceReferenceWindingObservation* source = nullptr;
    int rawFromWindingSign = 1;
    double rawFromWindingOffset = 0.0;
    std::array<double, 2> candidates{0.0, 0.0};
    std::size_t candidateCount = 0;
};

struct ReferenceScore {
    std::size_t hardViolations = 0;
    double loss = 0.0;
};

void validateReferenceObservation(
    const FiberTraceReferenceWindingObservation& observation)
{
    constexpr double epsilon = 1.0e-12;
    const auto classIndex =
        static_cast<std::size_t>(observation.constraintClass);
    const auto orientationIndex =
        static_cast<std::size_t>(observation.bpEndpointOrientation);
    if (classIndex >= 3 || orientationIndex >= 3 ||
        (observation.bpEndpointActive &&
         observation.bpEndpointOrientation ==
             FiberTraceFixedOrientation::Mixed) ||
        observation.referenceSource ==
            std::numeric_limits<std::size_t>::max() ||
        observation.inferredReferenceWindingCount >
            observation.inferredReferenceWindings.size() ||
        !std::isfinite(observation.canonicalWindingDistance) ||
        observation.canonicalWindingDistance < 0.0 ||
        !std::isfinite(observation.rawCoefficient) ||
        observation.rawCoefficient < 0.0 ||
        !std::isfinite(observation.admittedCoefficient) ||
        observation.admittedCoefficient < 0.0 ||
        observation.admittedCoefficient >
            observation.rawCoefficient + epsilon ||
        !std::isfinite(observation.coordinateResidualScale) ||
        !(observation.coordinateResidualScale > 0.0) ||
        (observation.exactWindingFactor &&
         (!std::isfinite(observation.bpLatentCoordinate) ||
          !std::isfinite(observation.referenceDeltaSign) ||
          std::abs(observation.referenceDeltaSign) != 1.0 ||
          !std::isfinite(observation.rawParallelCoefficient) ||
          observation.rawParallelCoefficient < 0.0 ||
          !std::isfinite(observation.admittedParallelCoefficient) ||
          observation.admittedParallelCoefficient < 0.0 ||
          !std::isfinite(observation.perpendicularCoefficient) ||
          observation.perpendicularCoefficient < 0.0 ||
          !std::isfinite(observation.decisionConfidenceMultiplier) ||
          observation.decisionConfidenceMultiplier < 0.0 ||
          observation.decisionConfidenceMultiplier > 1.0 ||
          !std::isfinite(observation.normalConfidenceMultiplier) ||
          observation.normalConfidenceMultiplier < 0.0 ||
          observation.normalConfidenceMultiplier > 1.0 ||
          !std::isfinite(observation.parallelSignPenalty) ||
          observation.parallelSignPenalty < 0.0 ||
          !std::isfinite(observation.perpendicularSignPenalty) ||
          observation.perpendicularSignPenalty < 0.0 ||
          !std::isfinite(observation.parallelDistance) ||
          observation.parallelDistance < 0.0 ||
          (observation.hardParallelSign &&
           (!observation.signedParallelTarget ||
            *observation.signedParallelTarget == 0.0)) ||
          (observation.hardPerpendicularSign &&
           (!observation.signedPerpendicularTarget ||
            *observation.signedPerpendicularTarget == 0.0)) ||
          (observation.parallelSignPenalty > 0.0 &&
           (!observation.signedParallelTarget ||
            *observation.signedParallelTarget == 0.0)) ||
          (observation.perpendicularSignPenalty > 0.0 &&
           (!observation.signedPerpendicularTarget ||
            *observation.signedPerpendicularTarget == 0.0))))) {
        throw std::invalid_argument(
            "Reference winding benchmark observation is invalid");
    }
    for (std::size_t candidate = 0;
         candidate < observation.inferredReferenceWindingCount;
         ++candidate) {
        if (!std::isfinite(observation.inferredReferenceWindings[candidate])) {
            throw std::invalid_argument(
                "Reference winding benchmark candidate is invalid");
        }
    }
}

ReferenceScoredObservation makeReferenceScoredObservation(
    const FiberTraceReferenceWindingObservation& observation,
    int rawFromWindingSign,
    double rawFromWindingOffset)
{
    validateReferenceObservation(observation);
    if ((rawFromWindingSign != -1 && rawFromWindingSign != 1) ||
        !std::isfinite(rawFromWindingOffset)) {
        throw std::invalid_argument(
            "Reference winding observation mapping is invalid");
    }
    ReferenceScoredObservation scored;
    scored.source = &observation;
    scored.rawFromWindingSign = rawFromWindingSign;
    scored.rawFromWindingOffset = rawFromWindingOffset;
    scored.candidateCount = observation.inferredReferenceWindingCount;
    for (std::size_t candidate = 0; candidate < scored.candidateCount;
         ++candidate) {
        scored.candidates[candidate] =
            static_cast<double>(rawFromWindingSign) *
            (observation.inferredReferenceWindings[candidate] -
             rawFromWindingOffset);
    }
    return scored;
}

bool hasAdmittedReferenceEvidence(
    const FiberTraceReferenceWindingObservation& observation)
{
    return observation.admittedCoefficient > 0.0 ||
        observation.hardParallelSign ||
        observation.hardPerpendicularSign;
}

ReferenceScore scoreReferenceObservation(
    const ReferenceScoredObservation& scored,
    double winding)
{
    const auto& observation = *scored.source;
    const double rawWinding =
        static_cast<double>(scored.rawFromWindingSign) * winding +
        scored.rawFromWindingOffset;
    if (!observation.exactWindingFactor) {
        double distance = std::numeric_limits<double>::infinity();
        for (std::size_t candidate = 0;
             candidate < observation.inferredReferenceWindingCount;
             ++candidate) {
            distance = std::min(
                distance,
                std::abs(
                    rawWinding -
                    observation.inferredReferenceWindings[candidate]));
        }
        return {
            0,
            observation.admittedCoefficient * distance /
                observation.coordinateResidualScale};
    }

    const double delta = observation.referenceDeltaSign *
        (rawWinding - observation.bpLatentCoordinate);
    const double predictedPerpendicular =
        delta / observation.coordinateResidualScale;
    ReferenceScore score;
    if (observation.hardPerpendicularSign &&
        *observation.signedPerpendicularTarget * predictedPerpendicular <=
            0.0) {
        ++score.hardViolations;
    }
    if (observation.hardParallelSign &&
        *observation.signedParallelTarget * delta <= 0.0) {
        ++score.hardViolations;
    }
    if (observation.perpendicularSignPenalty > 0.0 &&
        *observation.signedPerpendicularTarget * predictedPerpendicular <=
            0.0) {
        score.loss += observation.perpendicularSignPenalty;
    }
    if (observation.parallelSignPenalty > 0.0 &&
        *observation.signedParallelTarget * delta <= 0.0) {
        score.loss += observation.parallelSignPenalty;
    }
    if (observation.admittedParallelCoefficient > 0.0) {
        const double residual = observation.signedParallelTarget
            ? delta - *observation.signedParallelTarget
            : std::abs(delta) - observation.parallelDistance;
        score.loss += observation.admittedParallelCoefficient *
            std::abs(residual);
    }
    if (observation.signedPerpendicularTarget) {
        score.loss += observation.perpendicularCoefficient *
            std::abs(
                predictedPerpendicular -
                *observation.signedPerpendicularTarget);
    }
    return score;
}

FiberTraceReferenceConstraintGroupDiagnostic summarizeReferenceObservations(
    std::span<const ReferenceScoredObservation> observations,
    std::optional<double> truth)
{
    constexpr double epsilon = 1.0e-12;
    FiberTraceReferenceConstraintGroupDiagnostic summary;
    summary.observations = observations.size();
    for (const auto& observation : observations) {
        summary.rawCoefficient += observation.source->rawCoefficient;
        summary.admittedCoefficient +=
            observation.source->admittedCoefficient;
    }
    const bool hasHardSign = std::any_of(
        observations.begin(), observations.end(),
        [](const ReferenceScoredObservation& observation) {
            return observation.source->hardParallelSign ||
                observation.source->hardPerpendicularSign;
        });
    if (observations.empty() ||
        (!(summary.admittedCoefficient > 0.0) && !hasHardSign))
        return summary;

    const auto scoreAt = [&](double winding) {
        ReferenceScore score;
        for (const auto& observation : observations) {
            const auto current =
                scoreReferenceObservation(observation, winding);
            score.hardViolations += current.hardViolations;
            score.loss += current.loss;
        }
        return score;
    };
    if (truth) {
        const auto score = scoreAt(*truth);
        summary.truthHardViolations = score.hardViolations;
        summary.truthLoss = score.loss;
    }

    std::set<long long> candidateTicks{0};
    const auto addBreakpoint = [&](double breakpoint) {
        const double tick = 2.0 * breakpoint;
        if (tick < static_cast<double>(
                       std::numeric_limits<long long>::min() + 4) ||
            tick > static_cast<double>(
                       std::numeric_limits<long long>::max() - 4)) {
            throw std::invalid_argument(
                "Reference winding candidate is out of range");
        }
        const auto lower = static_cast<long long>(std::floor(tick));
        for (long long offset = -1; offset <= 2; ++offset)
            candidateTicks.insert(lower + offset);
    };
    for (const auto& observation : observations) {
        for (std::size_t candidate = 0;
             candidate < observation.candidateCount;
             ++candidate) {
            addBreakpoint(observation.candidates[candidate]);
        }
        if (observation.source->exactWindingFactor &&
            observation.source->hardPerpendicularSign) {
            addBreakpoint(
                static_cast<double>(observation.rawFromWindingSign) *
                (observation.source->bpLatentCoordinate -
                 observation.rawFromWindingOffset));
        }
        if (observation.source->exactWindingFactor &&
            observation.source->hardParallelSign) {
            addBreakpoint(
                static_cast<double>(observation.rawFromWindingSign) *
                (observation.source->bpLatentCoordinate -
                 observation.rawFromWindingOffset));
        }
        for (std::size_t first = 0;
             first < observation.candidateCount;
             ++first) {
            for (std::size_t second = first + 1;
                 second < observation.candidateCount;
                 ++second) {
                addBreakpoint(0.5 *
                    (observation.candidates[first] +
                     observation.candidates[second]));
            }
        }
    }

    double bestWinding = 0.0;
    ReferenceScore best{std::numeric_limits<std::size_t>::max(),
                        std::numeric_limits<double>::infinity()};
    for (const long long tick : candidateTicks) {
        const double winding = 0.5 * static_cast<double>(tick);
        const auto score = scoreAt(winding);
        const bool better =
            score.hardViolations < best.hardViolations ||
            (score.hardViolations == best.hardViolations &&
             (score.loss < best.loss - epsilon ||
              (std::abs(score.loss - best.loss) <= epsilon &&
               winding < bestWinding)));
        if (better) {
            bestWinding = winding;
            best = score;
        }
    }
    summary.preferredWinding = bestWinding;
    summary.preferredHardViolations = best.hardViolations;
    summary.preferredLoss = best.loss;
    return summary;
}

FiberTraceReferenceConstraintGroup groupForReferenceObservation(
    const FiberTraceReferenceWindingObservation& observation)
{
    return referenceConstraintGroup(
        observation.constraintClass ==
            FiberTraceReferenceConstraintClass::Perpendicular,
        observation.canonicalWindingDistance);
}

long long referenceHalfStepIndex(double winding)
{
    if (!std::isfinite(winding)) {
        throw std::invalid_argument(
            "Reference winding parity input is not finite");
    }
    const double scaled = 2.0 * winding;
    if (scaled < static_cast<double>(
                     std::numeric_limits<long long>::min() + 1) ||
        scaled > static_cast<double>(
                     std::numeric_limits<long long>::max() - 1)) {
        throw std::invalid_argument(
            "Reference winding parity input is out of range");
    }
    const auto index = static_cast<long long>(std::llround(scaled));
    const double tolerance = 32.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::abs(scaled));
    if (std::abs(scaled - static_cast<double>(index)) > tolerance) {
        throw std::invalid_argument(
            "Reference winding estimate is not on the half-step lattice");
    }
    return index;
}

int normalizedHalfStepParity(long long index) noexcept
{
    const int remainder = static_cast<int>(index % 2);
    return remainder < 0 ? remainder + 2 : remainder;
}

}  // namespace

std::optional<bool> fiberTraceReferenceEstimateParityMatches(
    std::size_t referenceSource,
    std::optional<double> estimatedWinding)
{
    if (!estimatedWinding)
        return std::nullopt;
    if (referenceSource > static_cast<std::size_t>(
                              std::numeric_limits<long long>::max())) {
        throw std::invalid_argument(
            "Reference source is out of half-step parity range");
    }
    return normalizedHalfStepParity(
               referenceHalfStepIndex(*estimatedWinding)) ==
        static_cast<int>(referenceSource % 2);
}

std::optional<int> fiberTraceReferenceOutputWinding(
    std::size_t referenceSource,
    const FiberTraceReferenceSourceBenchmark& reference,
    const FiberTraceReferenceOrientationBenchmark& orientation,
    std::span<const int> componentPhaseSigns,
    double phaseMagnitude,
    int outputOffset)
{
    if (!reference.rawEstimatedWinding ||
        !reference.estimatedIntegerGauge ||
        !reference.estimatedOrientationComponent) {
        return std::nullopt;
    }
    const std::size_t component =
        *reference.estimatedOrientationComponent;
    const auto calibration = std::find_if(
        orientation.components.begin(), orientation.components.end(),
        [component](const auto& current) {
            return current.component == component;
        });
    if (calibration == orientation.components.end() ||
        component >= componentPhaseSigns.size() ||
        !std::isfinite(phaseMagnitude)) {
        return std::nullopt;
    }
    const int phaseSign = componentPhaseSigns[component];
    if (phaseSign != -1 && phaseSign != 1) {
        throw std::invalid_argument(
            "Reference winding component phase sign is invalid");
    }
    const bool evenSource = referenceSource % 2 == 0;
    const bool horizontal = evenSource
        ? calibration->evenReferenceIsHorizontal
        : !calibration->evenReferenceIsHorizontal;
    const double classOffset = horizontal
        ? 0.0
        : static_cast<double>(phaseSign) * phaseMagnitude;
    (void)referenceHalfStepIndex(*reference.rawEstimatedWinding);
    const double relativeWinding =
        *reference.rawEstimatedWinding - classOffset;
    const double rounded = std::round(relativeWinding);
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::abs(relativeWinding));
    if (std::abs(relativeWinding - rounded) > tolerance) {
        return std::nullopt;
    }
    if (rounded < static_cast<double>(std::numeric_limits<int>::min()) ||
        rounded > static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "Reference winding output layer is out of range");
    }
    return static_cast<int>(rounded) + outputOffset;
}

FiberTraceReferenceOrientationBenchmark
benchmarkFiberTraceReferenceOrientations(
    std::span<const FiberTraceReferenceWindingObservation> observations)
{
    using Relation = FiberTraceReferenceOrientationRelation;
    struct ComponentVotes {
        std::size_t evenHorizontal = 0;
        std::size_t evenVertical = 0;
    };
    std::size_t sourceCount = 0;
    std::map<std::size_t, ComponentVotes> votesByComponent;
    for (const auto& observation : observations) {
        validateReferenceObservation(observation);
        sourceCount = std::max(
            sourceCount, observation.referenceSource + 1);
        const auto truthIndex = referenceHalfStepIndex(
            observation.virtualReferenceWinding);
        if (truthIndex < 0 ||
            static_cast<unsigned long long>(truthIndex) !=
                static_cast<unsigned long long>(
                    observation.referenceSource)) {
            throw std::invalid_argument(
                "Reference orientation truth does not match source order");
        }
        if (!observation.bpEndpointActive)
            continue;
        const bool perpendicular =
            observation.constraintClass ==
            FiberTraceReferenceConstraintClass::Perpendicular;
        const int expectedWithEvenHorizontal =
            static_cast<int>(observation.referenceSource % 2) ^
            static_cast<int>(perpendicular);
        const int actual = observation.bpEndpointOrientation ==
                FiberTraceFixedOrientation::Vertical
            ? 1
            : 0;
        auto& votes = votesByComponent[observation.bpOrientationComponent];
        if (actual == expectedWithEvenHorizontal)
            ++votes.evenHorizontal;
        else
            ++votes.evenVertical;
    }

    FiberTraceReferenceOrientationBenchmark result;
    result.references.resize(sourceCount);
    std::map<std::size_t, bool> evenHorizontalByComponent;
    for (const auto& [component, votes] : votesByComponent) {
        const bool evenHorizontal =
            votes.evenHorizontal >= votes.evenVertical;
        evenHorizontalByComponent.emplace(component, evenHorizontal);
        result.components.push_back({
            component,
            evenHorizontal,
            votes.evenHorizontal,
            votes.evenVertical,
        });
    }

    const auto add = [](FiberTraceReferenceOrientationCounts& counts,
                        bool right) {
        if (right)
            ++counts.right;
        else
            ++counts.wrong;
    };
    for (const auto& observation : observations) {
        if (!observation.bpEndpointActive) {
            ++result.excludedInactive;
            continue;
        }
        const auto mapping = evenHorizontalByComponent.find(
            observation.bpOrientationComponent);
        if (mapping == evenHorizontalByComponent.end()) {
            throw std::logic_error(
                "Reference orientation component has no calibration");
        }
        const bool perpendicular =
            observation.constraintClass ==
            FiberTraceReferenceConstraintClass::Perpendicular;
        const auto relation = perpendicular
            ? Relation::Perpendicular
            : Relation::Parallel;
        int expected =
            static_cast<int>(observation.referenceSource % 2) ^
            static_cast<int>(perpendicular);
        if (!mapping->second)
            expected ^= 1;
        const int actual = observation.bpEndpointOrientation ==
                FiberTraceFixedOrientation::Vertical
            ? 1
            : 0;
        const bool right = actual == expected;
        const auto relationIndex = static_cast<std::size_t>(relation);
        auto& reference = result.references.at(observation.referenceSource);
        add(reference.relations[relationIndex], right);
        add(reference.sum, right);
        add(result.relations[relationIndex], right);
        add(result.sum, right);
    }
    return result;
}

std::vector<FiberTraceReferenceRawWindingEstimate>
inferFiberTraceReferenceRawWindings(
    std::span<const FiberTraceReferenceWindingObservation> observations)
{
    using Key = std::pair<std::size_t, std::size_t>;
    std::map<Key, std::vector<ReferenceScoredObservation>> grouped;
    for (const auto& observation : observations) {
        validateReferenceObservation(observation);
        if (observation.inferredReferenceWindingCount == 0 ||
            !hasAdmittedReferenceEvidence(observation)) {
            continue;
        }
        grouped[{observation.referenceSource, observation.integerGauge}]
            .push_back(makeReferenceScoredObservation(observation, 1, 0.0));
    }

    std::vector<FiberTraceReferenceRawWindingEstimate> result;
    result.reserve(grouped.size());
    for (const auto& [key, scored] : grouped) {
        const auto summary =
            summarizeReferenceObservations(scored, std::nullopt);
        if (!summary.preferredWinding)
            continue;
        result.push_back({key.first,
                          key.second,
                          *summary.preferredWinding,
                          summary.observations,
                          summary.admittedCoefficient});
    }
    return result;
}

FiberTraceReferenceWindingBenchmark calibrateFiberTraceReferenceWindings(std::span<const FiberTraceReferenceWindingObservation> observations, double tolerance)
{
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw std::invalid_argument("Reference winding benchmark tolerance must be finite and nonnegative");
    }

    std::size_t referenceSources = 0;
    std::vector<std::optional<double>> truthBySource;
    for (const auto& observation : observations) {
        validateReferenceObservation(observation);
        if (!std::isfinite(observation.virtualReferenceWinding))
            throw std::invalid_argument("Reference winding truth is invalid");
        referenceSources = std::max(
            referenceSources, observation.referenceSource + 1);
        if (truthBySource.size() < referenceSources)
            truthBySource.resize(referenceSources);
        auto& truth = truthBySource[observation.referenceSource];
        if (truth && *truth != observation.virtualReferenceWinding) {
            throw std::invalid_argument(
                "Reference winding truth is inconsistent for one source");
        }
        truth = observation.virtualReferenceWinding;
    }

    FiberTraceReferenceWindingBenchmark result;
    result.tolerance = tolerance;
    result.references.resize(referenceSources);
    std::vector<FiberTraceReferenceWindingObservation> benchmarkObservations(
        observations.begin(), observations.end());
    for (auto& observation : benchmarkObservations) {
        const bool legacyCandidateOnly =
            !observation.parallelMagnitudePresent &&
            !observation.perpendicularMagnitudePresent &&
            !observation.parallelSignPresent &&
            !observation.perpendicularSignPresent &&
            observation.inferredReferenceWindingCount > 0 &&
            hasAdmittedReferenceEvidence(observation);
        observation.admittedParallelCoefficient =
            observation.parallelMagnitudePresent ? 1.0 : 0.0;
        observation.perpendicularCoefficient =
            observation.perpendicularMagnitudePresent ? 1.0 : 0.0;
        observation.parallelSignPenalty =
            observation.parallelSignPresent ? 1.0 : 0.0;
        observation.perpendicularSignPenalty =
            observation.perpendicularSignPresent ? 1.0 : 0.0;
        observation.hardParallelSign = false;
        observation.hardPerpendicularSign = false;
        observation.rawCoefficient =
            observation.admittedParallelCoefficient +
            observation.perpendicularCoefficient +
            observation.parallelSignPenalty +
            observation.perpendicularSignPenalty;
        observation.admittedCoefficient = observation.rawCoefficient;
        if (legacyCandidateOnly) {
            observation.rawCoefficient = 1.0;
            observation.admittedCoefficient = 1.0;
        }
    }
    const auto rawEstimates = inferFiberTraceReferenceRawWindings(
        benchmarkObservations);
    std::map<
        std::size_t,
        std::vector<const FiberTraceReferenceRawWindingEstimate*>>
        estimatesByGauge;
    for (const auto& estimate : rawEstimates)
        estimatesByGauge[estimate.integerGauge].push_back(&estimate);
    std::vector<std::set<std::size_t>> estimateGaugesBySource(
        referenceSources);
    std::vector<std::set<std::size_t>> estimateComponentsBySource(
        referenceSources);
    for (const auto& observation : benchmarkObservations) {
        if (observation.inferredReferenceWindingCount == 0 ||
            !hasAdmittedReferenceEvidence(observation)) {
            continue;
        }
        estimateGaugesBySource.at(observation.referenceSource).insert(
            observation.integerGauge);
        estimateComponentsBySource.at(observation.referenceSource).insert(
            observation.bpOrientationComponent);
    }

    struct SignCalibration {
        int sign = 1;
        std::vector<FiberTraceReferenceGaugeCalibration> gauges;
        std::size_t exactMatches = 0;
        double residual = 0.0;
    };
    const auto calibrateSign = [&](int sign) {
        SignCalibration calibration;
        calibration.sign = sign;
        for (const auto& [gauge, estimates] : estimatesByGauge) {
            std::set<double> candidateOffsets;
            for (const auto* estimate : estimates) {
                candidateOffsets.insert(
                    estimate->winding - static_cast<double>(sign) *
                        *truthBySource[estimate->referenceSource]);
            }
            double bestOffset = 0.0;
            std::size_t bestExact = 0;
            double bestResidual = std::numeric_limits<double>::infinity();
            bool haveBest = false;
            for (const double offset : candidateOffsets) {
                std::size_t exact = 0;
                double residual = 0.0;
                for (const auto* estimate : estimates) {
                    const double expected =
                        static_cast<double>(sign) *
                            *truthBySource[estimate->referenceSource] +
                        offset;
                    exact += estimate->winding == expected ? 1 : 0;
                    residual += std::abs(estimate->winding - expected);
                }
                const bool better =
                    !haveBest || exact > bestExact ||
                    (exact == bestExact &&
                     (residual < bestResidual - kEpsilon ||
                      (std::abs(residual - bestResidual) <= kEpsilon &&
                       (std::abs(offset) < std::abs(bestOffset) ||
                        (std::abs(offset) == std::abs(bestOffset) &&
                         offset < bestOffset)))));
                if (better) {
                    bestOffset = offset;
                    bestExact = exact;
                    bestResidual = residual;
                    haveBest = true;
                }
            }
            calibration.gauges.push_back({
                gauge, bestOffset, estimates.size(), bestExact});
            calibration.exactMatches += bestExact;
            calibration.residual += bestResidual;
        }
        return calibration;
    };

    SignCalibration calibration = calibrateSign(1);
    const SignCalibration reversed = calibrateSign(-1);
    if (reversed.exactMatches > calibration.exactMatches ||
        (reversed.exactMatches == calibration.exactMatches &&
         reversed.residual < calibration.residual - kEpsilon)) {
        calibration = reversed;
    }
    result.globalSign = calibration.sign;
    result.gauges = std::move(calibration.gauges);

    std::map<std::size_t, double> offsetByGauge;
    for (const auto& gauge : result.gauges)
        offsetByGauge.emplace(gauge.integerGauge, gauge.offset);
    const auto addResult = [&](std::size_t source,
                               FiberTraceReferenceBenchmarkClass itemClass,
                               bool right) {
        auto& counts = result.classes[static_cast<std::size_t>(itemClass)];
        auto& reference = result.references[source];
        auto& referenceCounts =
            reference.classes[static_cast<std::size_t>(itemClass)];
        ++counts.total;
        ++referenceCounts.total;
        ++reference.sum.total;
        ++result.sum.total;
        if (right) {
            ++counts.right;
            ++referenceCounts.right;
            ++reference.sum.right;
            ++result.sum.right;
        } else {
            ++counts.wrong;
            ++referenceCounts.wrong;
            ++reference.sum.wrong;
            ++result.sum.wrong;
        }
    };
    for (const auto& observation : observations) {
        const bool legacyCandidateOnly =
            !observation.parallelMagnitudePresent &&
            !observation.perpendicularMagnitudePresent &&
            !observation.parallelSignPresent &&
            !observation.perpendicularSignPresent &&
            observation.inferredReferenceWindingCount > 0 &&
            hasAdmittedReferenceEvidence(observation);
        if (!observation.bpEndpointActive && !legacyCandidateOnly)
            continue;
        const auto offset = offsetByGauge.find(observation.integerGauge);
        const bool calibrated = offset != offsetByGauge.end();
        const double expected = calibrated
            ? static_cast<double>(result.globalSign) *
                    observation.virtualReferenceWinding + offset->second
            : 0.0;
        const double delta = calibrated
            ? observation.referenceDeltaSign *
                (expected - observation.bpLatentCoordinate)
            : 0.0;
        if (legacyCandidateOnly) {
            const double expectedWinding = calibrated
                ? static_cast<double>(result.globalSign) *
                        observation.virtualReferenceWinding + offset->second
                : 0.0;
            bool right = false;
            if (calibrated) {
                for (std::size_t candidate = 0;
                     candidate < observation.inferredReferenceWindingCount;
                     ++candidate) {
                    right = right || std::abs(
                        observation.inferredReferenceWindings[candidate] -
                        expectedWinding) <= tolerance + kEpsilon;
                }
            }
            const auto itemClass = observation.constraintClass ==
                    FiberTraceReferenceConstraintClass::Perpendicular
                ? FiberTraceReferenceBenchmarkClass::PerpendicularMagnitude
                : observation.constraintClass ==
                        FiberTraceReferenceConstraintClass::ParallelSameWinding
                    ? FiberTraceReferenceBenchmarkClass::ParallelSameMagnitude
                    : FiberTraceReferenceBenchmarkClass::ParallelOtherMagnitude;
            addResult(observation.referenceSource, itemClass, right);
            continue;
        }
        if (observation.perpendicularMagnitudePresent) {
            const bool right = calibrated &&
                quantizedHalfWindingTarget(delta) ==
                    *observation.signedPerpendicularTarget;
            addResult(
                observation.referenceSource,
                FiberTraceReferenceBenchmarkClass::PerpendicularMagnitude,
                right);
        }
        if (observation.perpendicularSignPresent) {
            const bool right = calibrated &&
                *observation.signedPerpendicularTarget * delta > 0.0;
            addResult(
                observation.referenceSource,
                FiberTraceReferenceBenchmarkClass::PerpendicularSign,
                right);
        }
        if (observation.parallelMagnitudePresent) {
            const bool right = calibrated &&
                (observation.signedParallelTarget
                    ? quantizedIntegerWindingTarget(delta) ==
                        *observation.signedParallelTarget
                    : std::abs(quantizedIntegerWindingTarget(delta)) ==
                        observation.parallelDistance);
            addResult(
                observation.referenceSource,
                observation.parallelDistance == 0.0
                    ? FiberTraceReferenceBenchmarkClass::
                          ParallelSameMagnitude
                    : FiberTraceReferenceBenchmarkClass::
                          ParallelOtherMagnitude,
                right);
        }
        if (observation.parallelSignPresent) {
            const bool right = calibrated &&
                *observation.signedParallelTarget * delta > 0.0;
            addResult(
                observation.referenceSource,
                FiberTraceReferenceBenchmarkClass::ParallelSign,
                right);
        }
    }
    const auto weighted = summarizeFiberTraceReferenceConstraintGroups(
        benchmarkObservations, result);
    for (std::size_t source = 0;
         source < result.references.size() && source < weighted.size();
         ++source) {
        auto& reference = result.references[source];
        reference.estimatedWinding = weighted[source].all.preferredWinding;
        if (reference.estimatedWinding &&
            estimateGaugesBySource[source].size() == 1) {
            const std::size_t gauge = *estimateGaugesBySource[source].begin();
            const auto offset = offsetByGauge.find(gauge);
            if (offset != offsetByGauge.end()) {
                reference.rawEstimatedWinding =
                    static_cast<double>(result.globalSign) *
                        *reference.estimatedWinding +
                    offset->second;
                reference.estimatedIntegerGauge = gauge;
            }
        }
        if (estimateComponentsBySource[source].size() == 1) {
            reference.estimatedOrientationComponent =
                *estimateComponentsBySource[source].begin();
        }
        reference.estimatedWindingObservations =
            weighted[source].all.observations;
        reference.estimatedWindingSupport = 0;
        reference.estimatedParityMatches =
            fiberTraceReferenceEstimateParityMatches(
                source, reference.estimatedWinding);
        if (!reference.estimatedWinding)
            continue;
        for (const auto& observation : benchmarkObservations) {
            if (observation.referenceSource != source ||
                observation.inferredReferenceWindingCount == 0 ||
                !hasAdmittedReferenceEvidence(observation)) {
                continue;
            }
            const auto offset = offsetByGauge.find(observation.integerGauge);
            if (offset == offsetByGauge.end())
                continue;
            const double rawWinding =
                static_cast<double>(result.globalSign) *
                    *reference.estimatedWinding + offset->second;
            if (!(observation.admittedCoefficient > 0.0) &&
                (observation.hardParallelSign ||
                 observation.hardPerpendicularSign)) {
                reference.estimatedWindingSupport +=
                    scoreReferenceObservation(
                        makeReferenceScoredObservation(
                            observation, 1, 0.0),
                        rawWinding)
                            .hardViolations == 0
                    ? 1
                    : 0;
            } else {
                double distance = std::numeric_limits<double>::infinity();
                for (std::size_t candidate = 0;
                     candidate < observation.inferredReferenceWindingCount;
                     ++candidate) {
                    distance = std::min(
                        distance,
                        std::abs(
                            rawWinding -
                            observation.inferredReferenceWindings[candidate]));
                }
                reference.estimatedWindingSupport +=
                    distance <= tolerance + kEpsilon ? 1 : 0;
            }
        }
    }
    return result;
}

namespace
{

template <typename Append>
void appendMaterializedReferenceFactorConflicts(
    bool perpendicular,
    double predictedDelta,
    bool magnitudePresent,
    double magnitudeWeight,
    std::optional<double> signedMagnitudeTarget,
    double unsignedMagnitudeDistance,
    bool signPresent,
    bool hardSign,
    double signPenalty,
    Append&& append)
{
    if (!std::isfinite(predictedDelta) ||
        !std::isfinite(magnitudeWeight) || magnitudeWeight < 0.0 ||
        !std::isfinite(unsignedMagnitudeDistance) ||
        unsignedMagnitudeDistance < 0.0 ||
        !std::isfinite(signPenalty) || signPenalty < 0.0 ||
        (signedMagnitudeTarget && !std::isfinite(*signedMagnitudeTarget))) {
        throw std::invalid_argument(
            "Reference factor conflict input is invalid");
    }
    if (perpendicular && magnitudePresent && !signedMagnitudeTarget) {
        throw std::invalid_argument(
            "Perpendicular reference magnitude factor has no signed target");
    }

    if (magnitudePresent && magnitudeWeight > 0.0) {
        const double target = signedMagnitudeTarget
            ? *signedMagnitudeTarget
            : unsignedMagnitudeDistance;
        const double residual = signedMagnitudeTarget
            ? std::abs(predictedDelta - target)
            : std::abs(std::abs(predictedDelta) - unsignedMagnitudeDistance);
        const double displayedPrediction = signedMagnitudeTarget
            ? predictedDelta
            : std::abs(predictedDelta);
        append(
            referenceMagnitudeFactorClass(perpendicular, target),
            false,
            displayedPrediction,
            target,
            residual,
            magnitudeWeight,
            magnitudeWeight * residual);
    }

    if (signPresent && (hardSign || signPenalty > 0.0)) {
        if (!signedMagnitudeTarget || *signedMagnitudeTarget == 0.0) {
            throw std::invalid_argument(
                "Reference sign factor has no nonzero signed target");
        }
        const bool violation =
            *signedMagnitudeTarget * predictedDelta <= 0.0;
        append(
            perpendicular
                ? FiberTraceReferenceFactorClass::PerpendicularSign
                : FiberTraceReferenceFactorClass::ParallelSign,
            hardSign && violation,
            predictedDelta,
            *signedMagnitudeTarget,
            violation ? 1.0 : 0.0,
            signPenalty,
            violation ? signPenalty : 0.0);
    }
}

} // namespace

std::vector<FiberTraceReferenceClampedFactorConflict>
diagnoseFiberTraceReferenceClampedConflicts(
    std::span<const FiberTraceReferenceWindingObservation> observations,
    const FiberTraceReferenceWindingBenchmark& calibration)
{
    if (calibration.globalSign != -1 && calibration.globalSign != 1) {
        throw std::invalid_argument(
            "Reference conflict calibration sign is invalid");
    }
    std::map<std::size_t, double> offsetByGauge;
    for (const auto& gauge : calibration.gauges) {
        if (!std::isfinite(gauge.offset) ||
            !offsetByGauge.emplace(gauge.integerGauge, gauge.offset).second) {
            throw std::invalid_argument(
                "Reference conflict gauge calibration is invalid");
        }
    }

    std::vector<FiberTraceReferenceClampedFactorConflict> result;
    result.reserve(2 * observations.size());
    const auto append = [&result](
                            const FiberTraceReferenceWindingObservation& observation,
                            FiberTraceReferenceFactorClass factorClass,
                            bool hardViolation,
                            double predictedDelta,
                            double targetDelta,
                            double residual,
                            double effectiveWeight,
                            double weightedLoss) {
        if (!std::isfinite(predictedDelta) ||
            !std::isfinite(targetDelta) ||
            !std::isfinite(residual) || residual < 0.0 ||
            !std::isfinite(effectiveWeight) || effectiveWeight < 0.0 ||
            !std::isfinite(weightedLoss) || weightedLoss < 0.0) {
            throw std::logic_error(
                "Reference conflict factor diagnostic is invalid");
        }
        result.push_back({
            observation.referenceSource,
            observation.bpPiece,
            observation.constraintIndex,
            factorClass,
            hardViolation,
            predictedDelta,
            targetDelta,
            residual,
            effectiveWeight,
            weightedLoss,
        });
    };

    for (const auto& observation : observations) {
        validateReferenceObservation(observation);
        if (!observation.bpEndpointActive)
            continue;
        const auto offset = offsetByGauge.find(observation.integerGauge);
        if (offset == offsetByGauge.end())
            continue;
        const double fixedReferenceLatent =
            static_cast<double>(calibration.globalSign) *
                observation.virtualReferenceWinding +
            offset->second;
        const double delta = observation.referenceDeltaSign *
            (fixedReferenceLatent - observation.bpLatentCoordinate);
        const double perpendicularDelta =
            delta / observation.coordinateResidualScale;
        const bool perpendicular =
            observation.constraintClass ==
            FiberTraceReferenceConstraintClass::Perpendicular;
        appendMaterializedReferenceFactorConflicts(
            perpendicular,
            perpendicular ? perpendicularDelta : delta,
            perpendicular
                ? observation.perpendicularMagnitudePresent
                : observation.parallelMagnitudePresent,
            perpendicular
                ? observation.perpendicularCoefficient
                : observation.admittedParallelCoefficient,
            perpendicular
                ? observation.signedPerpendicularTarget
                : observation.signedParallelTarget,
            observation.parallelDistance,
            perpendicular
                ? observation.perpendicularSignPresent
                : observation.parallelSignPresent,
            perpendicular
                ? observation.hardPerpendicularSign
                : observation.hardParallelSign,
            perpendicular
                ? observation.perpendicularSignPenalty
                : observation.parallelSignPenalty,
            [&](FiberTraceReferenceFactorClass factorClass,
                bool hardViolation,
                double predictedDelta,
                double targetDelta,
                double residual,
                double effectiveWeight,
                double weightedLoss) {
                append(
                    observation,
                    factorClass,
                    hardViolation,
                    predictedDelta,
                    targetDelta,
                    residual,
                    effectiveWeight,
                    weightedLoss);
            });
    }
    return result;
}

std::vector<FiberTraceReferenceConstraintFactorConflict>
diagnoseFiberTraceReferenceConstraintConflicts(
    const FiberTraceConstraintReport& constraints,
    std::span<const std::size_t> sourceIdsByTrace,
    std::span<const FiberTraceWindingFactorDiagnostic> factors,
    int globalSign)
{
    if (sourceIdsByTrace.size() != constraints.inputTraces ||
        (globalSign != -1 && globalSign != 1)) {
        throw std::invalid_argument(
            "Reference constraint conflict inputs are invalid");
    }

    std::vector<FiberTraceReferenceConstraintFactorConflict> result;
    result.reserve(2 * factors.size());
    for (const auto& factor : factors) {
        if (factor.constraintIndex >= constraints.constraints.size() ||
            factor.canonicalNodeA >= constraints.pieces.size() ||
            factor.canonicalNodeB >= constraints.pieces.size()) {
            throw std::invalid_argument(
                "Reference factor diagnostic identifies invalid geometry");
        }
        const auto& constraint =
            constraints.constraints[factor.constraintIndex];
        if (constraint.hardContinuity)
            continue;

        const std::size_t traceA =
            constraints.pieces[factor.canonicalNodeA].traceIndex;
        const std::size_t traceB =
            constraints.pieces[factor.canonicalNodeB].traceIndex;
        if (traceA >= sourceIdsByTrace.size() ||
            traceB >= sourceIdsByTrace.size()) {
            throw std::invalid_argument(
                "Reference factor diagnostic identifies an invalid trace");
        }
        const std::size_t sourceA = sourceIdsByTrace[traceA];
        const std::size_t sourceB = sourceIdsByTrace[traceB];
        if (sourceA == sourceB)
            continue;
        const double predictedDelta = static_cast<double>(globalSign) * 0.5 *
            (static_cast<double>(sourceB) - static_cast<double>(sourceA));
        const bool perpendicular =
            factor.perpendicularMagnitudePresent ||
            factor.perpendicularSignPresent;
        const bool parallel =
            factor.parallelMagnitudePresent || factor.parallelSignPresent;
        if (!perpendicular && !parallel)
            continue;
        if (perpendicular && parallel) {
            throw std::invalid_argument(
                "Reference factor diagnostic has ambiguous dominant relation");
        }

        appendMaterializedReferenceFactorConflicts(
            perpendicular,
            predictedDelta,
            perpendicular
                ? factor.perpendicularMagnitudePresent
                : factor.parallelMagnitudePresent,
            perpendicular
                ? factor.effectivePerpendicularWindingWeight
                : factor.effectiveParallelWindingWeight,
            perpendicular
                ? factor.effectivePerpendicularSignedDelta
                : factor.effectiveSignedParallelDelta,
            factor.effectiveParallelWindingDistance,
            perpendicular
                ? factor.perpendicularSignPresent
                : factor.parallelSignPresent,
            perpendicular
                ? factor.hardPerpendicularSign
                : factor.hardParallelSign,
            perpendicular
                ? factor.effectivePerpendicularSignPenalty
                : factor.effectiveParallelSignPenalty,
            [&](FiberTraceReferenceFactorClass factorClass,
                bool hardViolation,
                double fixedDelta,
                double targetDelta,
                double residual,
                double effectiveWeight,
                double weightedLoss) {
                result.push_back({
                    factor.constraintIndex,
                    sourceA,
                    sourceB,
                    factorClass,
                    hardViolation,
                    fixedDelta,
                    targetDelta,
                    residual,
                    effectiveWeight,
                    weightedLoss,
                });
            });
    }
    return result;
}

FiberTraceReferenceWindingObservation makeFiberTraceReferenceWindingObservation(
    const FiberTraceConstraint& constraint, bool referenceIsEndpointA, double virtualReferenceWinding, std::size_t bpPiece, const FiberTraceInterleavedWindingReport& winding,
    const FiberTraceWindingBeliefPropagationConfig& config)
{
    if (!std::isfinite(virtualReferenceWinding) || bpPiece >= winding.windingValid.size() ||
        winding.mapLatentCoordinate.size() != winding.windingValid.size() || winding.mapOrientationByPiece.size() != winding.windingValid.size() ||
        winding.integerGaugeByPiece.size() != winding.windingValid.size() ||
        winding.componentByPiece.size() != winding.windingValid.size() ||
        !std::isfinite(winding.measurementScale) ||
        !(winding.measurementScale > 0.0) ||
        (config.parallelWindingDistanceCutoff &&
         (!std::isfinite(*config.parallelWindingDistanceCutoff) ||
          !(*config.parallelWindingDistanceCutoff > 0.0)))) {
        throw std::invalid_argument("Reference winding observation inputs are invalid");
    }
    validateConfig(config);

    FiberTraceReferenceWindingObservation observation;
    observation.bpPiece = bpPiece;
    observation.integerGauge = winding.integerGaugeByPiece[bpPiece];
    observation.bpEndpointOrientation =
        winding.mapOrientationByPiece[bpPiece];
    observation.bpOrientationComponent =
        winding.componentByPiece[bpPiece];
    observation.virtualReferenceWinding = virtualReferenceWinding;
    const bool active = winding.windingValid[bpPiece] != 0 && winding.mapOrientationByPiece[bpPiece] != FiberTraceFixedOrientation::Mixed &&
                        std::isfinite(winding.mapLatentCoordinate[bpPiece]);
    observation.bpEndpointActive = active;
    const bool perpendicular = constraint.perpendicularScore >= constraint.parallelScore;
    observation.decisionConfidenceMultiplier =
        decisionConfidenceMultiplier(
            config.decisionConfidence,
            perpendicular
                ? constraint.perpendicularScore
                : constraint.parallelScore);
    observation.normalConfidenceMultiplier = normalConfidenceMultiplier(
        config.normalConfidence,
        perpendicular
            ? constraint.perpendicularNormalAlignment
            : constraint.parallelNormalAlignment);
    const double selectedConfidence =
        observation.decisionConfidenceMultiplier *
        observation.normalConfidenceMultiplier;
    const bool finiteSigns = config.finiteSignInfringementCost.has_value();
    observation.exactWindingFactor = true;
    observation.bpLatentCoordinate = active
        ? winding.mapLatentCoordinate[bpPiece]
        : 0.0;
    observation.referenceDeltaSign = referenceIsEndpointA ? -1.0 : 1.0;
    // Scale-first targets and latent winding coordinates now share the same
    // coordinate system. No residual coordinate conversion remains.
    observation.coordinateResidualScale = 1.0;
    const auto addCandidate = [&](double candidate) {
        if (!active || !std::isfinite(candidate))
            return;
        for (std::size_t index = 0;
             index < observation.inferredReferenceWindingCount;
             ++index) {
            if (observation.inferredReferenceWindings[index] == candidate)
                return;
        }
        if (observation.inferredReferenceWindingCount >=
            observation.inferredReferenceWindings.size()) {
            throw std::logic_error(
                "Reference winding observation has too many factor targets");
        }
        observation.inferredReferenceWindings
            [observation.inferredReferenceWindingCount++] = candidate;
    };

    if (perpendicular && constraint.signedWindingDelta) {
        const double target = quantizedHalfWindingTarget(
            *constraint.signedWindingDelta * winding.measurementScale);
        observation.signedPerpendicularTarget = target;
        observation.perpendicularMagnitudePresent = true;
        observation.perpendicularSignPresent =
            config.enforcePerpendicularWindingSign && target != 0.0;
        const bool perpendicularSignEnabled =
            observation.perpendicularSignPresent &&
            config.perpendicularSignWeight > 0.0;
        observation.hardPerpendicularSign =
            perpendicularSignEnabled &&
            (!finiteSigns || hardSignPromotedByAlignment(
                 config, constraint.perpendicularNormalAlignment));
        observation.perpendicularSignPenalty =
            perpendicularSignEnabled && finiteSigns &&
                !hardSignPromotedByAlignment(
                    config, constraint.perpendicularNormalAlignment)
            ? *config.finiteSignInfringementCost *
                config.perpendicularSignWeight * selectedConfidence
            : 0.0;
        observation.perpendicularSignWeightMultiplier =
            config.perpendicularSignWeight;
        observation.perpendicularCoefficient =
            selectedConfidence * windingWeightMultiplier(target) *
            windingClassWeight(config, true, target);
        addCandidate(
            observation.bpLatentCoordinate +
            observation.referenceDeltaSign * target);
    }

    const double parallelTarget = perpendicular
        ? 0.0
        : constraint.signedParallelWindingDelta
        ? std::abs(quantizedIntegerWindingTarget(
              *constraint.signedParallelWindingDelta *
              winding.measurementScale))
        : std::abs(quantizedIntegerWindingTarget(
              constraint.parallelWindingDistance *
              winding.measurementScale));
    observation.parallelDistance = parallelTarget;
    observation.rawParallelCoefficient = perpendicular
        ? 0.0
        : selectedConfidence * windingWeightMultiplier(parallelTarget) *
            windingClassWeight(config, false, parallelTarget);
    const bool parallelAdmitted = !perpendicular &&
        (!config.parallelWindingDistanceCutoff ||
         parallelTarget < *config.parallelWindingDistanceCutoff);
    observation.parallelMagnitudePresent = parallelAdmitted;
    observation.admittedParallelCoefficient = parallelAdmitted
        ? observation.rawParallelCoefficient
        : 0.0;
    if (!perpendicular && parallelTarget == 0.0) {
        addCandidate(observation.bpLatentCoordinate);
    } else if (!perpendicular && constraint.signedParallelWindingDelta) {
        observation.signedParallelTarget = quantizedIntegerWindingTarget(
            *constraint.signedParallelWindingDelta *
            winding.measurementScale);
        observation.parallelSignPresent = parallelAdmitted &&
            config.enforceParallelWindingSign &&
            *observation.signedParallelTarget != 0.0;
        const bool parallelSignEnabled = observation.parallelSignPresent &&
            config.parallelSignWeight > 0.0;
        observation.hardParallelSign = parallelAdmitted &&
            parallelSignEnabled &&
            (!finiteSigns || hardSignPromotedByAlignment(
                 config, constraint.parallelNormalAlignment));
        observation.parallelSignPenalty = parallelAdmitted &&
            parallelSignEnabled && finiteSigns &&
                !hardSignPromotedByAlignment(
                    config, constraint.parallelNormalAlignment)
            ? *config.finiteSignInfringementCost * config.parallelSignWeight *
                selectedConfidence
            : 0.0;
        observation.parallelSignWeightMultiplier = config.parallelSignWeight;
        addCandidate(
            observation.bpLatentCoordinate +
            observation.referenceDeltaSign *
                *observation.signedParallelTarget);
    } else if (!perpendicular) {
        addCandidate(
            observation.bpLatentCoordinate + parallelTarget);
        addCandidate(
            observation.bpLatentCoordinate - parallelTarget);
    }

    observation.rawCoefficient = observation.rawParallelCoefficient +
        observation.perpendicularCoefficient +
        observation.parallelSignPenalty +
        observation.perpendicularSignPenalty;
    observation.admittedCoefficient =
        observation.admittedParallelCoefficient +
        observation.perpendicularCoefficient +
        observation.parallelSignPenalty +
        observation.perpendicularSignPenalty;
    if (perpendicular) {
        observation.constraintClass = FiberTraceReferenceConstraintClass::Perpendicular;
        if (observation.signedPerpendicularTarget)
            observation.canonicalWindingDistance = std::abs(
                *observation.signedPerpendicularTarget);
        return observation;
    }

    observation.constraintClass =
        parallelTarget == 0.0
        ? FiberTraceReferenceConstraintClass::ParallelSameWinding
        : FiberTraceReferenceConstraintClass::ParallelOtherWinding;
    observation.canonicalWindingDistance = parallelTarget;
    return observation;
}

const char* fiberTraceReferenceConstraintGroupName(
    FiberTraceReferenceConstraintGroup group) noexcept
{
    switch (group) {
    case FiberTraceReferenceConstraintGroup::PerpendicularNext:
        return "perp_0.5";
    case FiberTraceReferenceConstraintGroup::PerpendicularFar:
        return "perp_1.5+";
    case FiberTraceReferenceConstraintGroup::ParallelSame:
        return "parallel_0";
    case FiberTraceReferenceConstraintGroup::ParallelOne:
        return "parallel_1";
    case FiberTraceReferenceConstraintGroup::ParallelTwoPlus:
        return "parallel_2+";
    case FiberTraceReferenceConstraintGroup::Count:
        break;
    }
    return "invalid";
}

const char* fiberTraceReferenceBenchmarkClassName(
    FiberTraceReferenceBenchmarkClass benchmarkClass) noexcept
{
    switch (benchmarkClass) {
    case FiberTraceReferenceBenchmarkClass::PerpendicularMagnitude:
        return "perp_winding";
    case FiberTraceReferenceBenchmarkClass::PerpendicularSign:
        return "perp_sign";
    case FiberTraceReferenceBenchmarkClass::ParallelSameMagnitude:
        return "parallel_same_winding";
    case FiberTraceReferenceBenchmarkClass::ParallelOtherMagnitude:
        return "parallel_other_winding";
    case FiberTraceReferenceBenchmarkClass::ParallelSign:
        return "parallel_sign";
    case FiberTraceReferenceBenchmarkClass::Count:
        break;
    }
    return "invalid";
}

const char* fiberTraceReferenceFactorClassName(
    FiberTraceReferenceFactorClass factorClass) noexcept
{
    switch (factorClass) {
    case FiberTraceReferenceFactorClass::PerpendicularMagnitudeNext:
        return "perp_winding_0.5";
    case FiberTraceReferenceFactorClass::PerpendicularMagnitudeFar:
        return "perp_winding_1.5+";
    case FiberTraceReferenceFactorClass::PerpendicularSign:
        return "perp_sign";
    case FiberTraceReferenceFactorClass::ParallelMagnitudeSame:
        return "parallel_winding_0";
    case FiberTraceReferenceFactorClass::ParallelMagnitudeOne:
        return "parallel_winding_1";
    case FiberTraceReferenceFactorClass::ParallelMagnitudeFar:
        return "parallel_winding_2+";
    case FiberTraceReferenceFactorClass::ParallelSign:
        return "parallel_sign";
    case FiberTraceReferenceFactorClass::Count:
        break;
    }
    return "invalid";
}

std::vector<FiberTraceReferenceSourceConstraintGroups>
summarizeFiberTraceReferenceConstraintGroups(
    std::span<const FiberTraceReferenceWindingObservation> observations,
    const FiberTraceReferenceWindingBenchmark& calibration)
{
    const std::size_t sourceCount = calibration.references.size();
    std::map<std::size_t, double> offsetByGauge;
    for (const auto& gauge : calibration.gauges) {
        if (!std::isfinite(gauge.offset) ||
            !offsetByGauge.emplace(gauge.integerGauge, gauge.offset).second) {
            throw std::invalid_argument(
                "Reference constraint-group calibration is invalid");
        }
    }

    struct WeightedObservation {
        const FiberTraceReferenceWindingObservation* source = nullptr;
        double gaugeOffset = 0.0;
        double truth = 0.0;
    };
    using GroupObservations = std::array<
        std::vector<WeightedObservation>,
        static_cast<std::size_t>(FiberTraceReferenceConstraintGroup::Count)>;
    std::vector<GroupObservations> grouped(sourceCount);

    for (const auto& observation : observations) {
        validateReferenceObservation(observation);
        if ((calibration.globalSign != -1 && calibration.globalSign != 1) ||
            observation.referenceSource >= sourceCount ||
            !std::isfinite(observation.virtualReferenceWinding)) {
            throw std::invalid_argument(
                "Reference constraint-group observation is invalid");
        }
        if (observation.inferredReferenceWindingCount == 0)
            continue;
        const auto offset = offsetByGauge.find(observation.integerGauge);
        if (offset == offsetByGauge.end())
            continue;
        WeightedObservation weighted;
        weighted.source = &observation;
        weighted.gaugeOffset = offset->second;
        weighted.truth = observation.virtualReferenceWinding;
        grouped[observation.referenceSource]
            [static_cast<std::size_t>(
                groupForReferenceObservation(observation))]
                .push_back(weighted);
    }

    const auto summarize = [&](std::span<const WeightedObservation> group) {
        std::vector<ReferenceScoredObservation> scored;
        scored.reserve(group.size());
        for (const auto& observation : group) {
            scored.push_back(makeReferenceScoredObservation(
                *observation.source,
                calibration.globalSign,
                observation.gaugeOffset));
        }
        return summarizeReferenceObservations(
            scored,
            group.empty()
                ? std::nullopt
                : std::optional<double>{group.front().truth});
    };

    std::vector<FiberTraceReferenceSourceConstraintGroups> result(sourceCount);
    for (std::size_t source = 0; source < sourceCount; ++source) {
        for (std::size_t groupIndex = 0;
             groupIndex < grouped[source].size();
             ++groupIndex) {
            result[source].groups[groupIndex] = summarize(
                grouped[source][groupIndex]);
        }
        std::vector<WeightedObservation> all;
        for (const auto& group : grouped[source])
            all.insert(all.end(), group.begin(), group.end());
        result[source].all = summarize(all);
    }
    return result;
}

const char* fiberTraceWindingSolverName(FiberTraceWindingSolver solver) noexcept
{
    switch (solver) {
    case FiberTraceWindingSolver::JointGrid:
        return "joint_grid";
    case FiberTraceWindingSolver::Alternating:
        return "alternating";
    case FiberTraceWindingSolver::Ceres:
        return "ceres";
    case FiberTraceWindingSolver::OrderedCuts:
        return "ordered-cuts";
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

FiberTraceFinalStateCohortSummary summarizeFiberTraceFinalStates(
    std::span<const FiberTraceFixedOrientation> orientations,
    std::span<const unsigned char> windingValid,
    std::span<const unsigned char> selectedCohort)
{
    if (windingValid.size() != orientations.size() ||
        selectedCohort.size() != orientations.size()) {
        throw std::invalid_argument(
            "Final fiber state summary inputs must have equal sizes");
    }

    FiberTraceFinalStateCohortSummary result;
    const auto add = [](FiberTraceFinalStateCounts& counts,
                         FiberTraceFixedOrientation orientation,
                         bool valid) {
        ++counts.pieces;
        if (valid && orientation == FiberTraceFixedOrientation::Horizontal)
            ++counts.horizontal;
        else if (valid && orientation == FiberTraceFixedOrientation::Vertical)
            ++counts.vertical;
        else
            ++counts.defect;
    };
    for (std::size_t piece = 0; piece < orientations.size(); ++piece) {
        auto& cohort = selectedCohort[piece] != 0
            ? result.selected
            : result.other;
        add(cohort, orientations[piece], windingValid[piece] != 0);
        add(result.total, orientations[piece], windingValid[piece] != 0);
    }
    return result;
}

const char* fiberTraceConstraintEvidenceClassName(
    FiberTraceConstraintEvidenceClass evidenceClass) noexcept
{
    switch (evidenceClass) {
    case FiberTraceConstraintEvidenceClass::Continuity:
        return "continuity";
    case FiberTraceConstraintEvidenceClass::PerpendicularMagnitude:
        return "perp_winding";
    case FiberTraceConstraintEvidenceClass::PerpendicularSign:
        return "perp_sign";
    case FiberTraceConstraintEvidenceClass::ParallelSameMagnitude:
        return "parallel_same_winding";
    case FiberTraceConstraintEvidenceClass::ParallelOtherMagnitude:
        return "parallel_other_winding";
    case FiberTraceConstraintEvidenceClass::ParallelSign:
        return "parallel_sign";
    case FiberTraceConstraintEvidenceClass::Count:
        break;
    }
    return "invalid";
}

FiberTraceConstraintEvidenceSummary summarizeFiberTraceConstraintEvidence(
    const FiberTraceConstraintReport& constraints,
    std::span<const FiberTraceWindingFactorDiagnostic> diagnostics,
    std::span<const FiberTraceFixedOrientation> orientations,
    std::span<const unsigned char> windingValid,
    std::span<const unsigned char> selectedCohort)
{
    if (constraints.pieces.size() != orientations.size() ||
        windingValid.size() != orientations.size() ||
        selectedCohort.size() != orientations.size()) {
        throw std::invalid_argument(
            "Constraint evidence summary piece inputs must have equal sizes");
    }

    FiberTraceConstraintEvidenceSummary result;
    const auto states = summarizeFiberTraceFinalStates(
        orientations, windingValid, selectedCohort);
    result.selected.states = states.selected;
    result.other.states = states.other;
    result.total.states = states.total;

    const auto addFinite = [](
        FiberTraceConstraintEvidenceCounts& counts,
        double weight,
        bool active) {
        ++counts.incidences;
        counts.effectiveWeight += weight;
        if (active) {
            ++counts.activeIncidences;
            counts.activeEffectiveWeight += weight;
        } else {
            ++counts.defectIncidences;
            counts.defectEffectiveWeight += weight;
        }
    };
    const auto addHardSign = [](
        FiberTraceConstraintEvidenceCounts& counts,
        bool active) {
        ++counts.hardSignIncidences;
        if (active)
            ++counts.activeHardSignIncidences;
        else
            ++counts.defectHardSignIncidences;
    };
    const auto addEndpoint = [&] (
        std::size_t piece,
        FiberTraceConstraintEvidenceClass evidenceClass,
        double weight,
        bool hardSign) {
        if (!(weight > 0.0) && !hardSign)
            return;
        const bool active = windingValid[piece] != 0 &&
            orientations[piece] != FiberTraceFixedOrientation::Mixed;
        const std::size_t classIndex = static_cast<std::size_t>(evidenceClass);
        auto& cohort = selectedCohort[piece] != 0
            ? result.selected
            : result.other;
        if (weight > 0.0) {
            addFinite(cohort.classes[classIndex], weight, active);
            addFinite(result.total.classes[classIndex], weight, active);
        }
        if (hardSign) {
            addHardSign(cohort.classes[classIndex], active);
            addHardSign(result.total.classes[classIndex], active);
        }
    };
    const auto addTerm = [&] (
        const FiberTraceWindingFactorDiagnostic& diagnostic,
        FiberTraceConstraintEvidenceClass evidenceClass,
        double weight,
        bool hardSign = false) {
        addEndpoint(diagnostic.pieceA, evidenceClass, weight, hardSign);
        addEndpoint(diagnostic.pieceB, evidenceClass, weight, hardSign);
    };
    const auto addMeasurementEndpoint = [&] (
        std::size_t piece,
        double weight,
        bool hardSign) {
        if (!(weight > 0.0) && !hardSign)
            return;
        const bool active = windingValid[piece] != 0 &&
            orientations[piece] != FiberTraceFixedOrientation::Mixed;
        auto& cohort = selectedCohort[piece] != 0
            ? result.selected
            : result.other;
        if (weight > 0.0) {
            addFinite(cohort.total, weight, active);
            addFinite(result.total.total, weight, active);
        }
        if (hardSign) {
            addHardSign(cohort.total, active);
            addHardSign(result.total.total, active);
        }
    };

    for (const auto& diagnostic : diagnostics) {
        if (diagnostic.constraintIndex >= constraints.constraints.size() ||
            diagnostic.pieceA >= orientations.size() ||
            diagnostic.pieceB >= orientations.size() ||
            diagnostic.pieceA == diagnostic.pieceB) {
            throw std::invalid_argument(
                "Constraint evidence diagnostic references invalid input");
        }
        const auto& constraint =
            constraints.constraints[diagnostic.constraintIndex];
        if ((constraint.pieceA != diagnostic.pieceA ||
             constraint.pieceB != diagnostic.pieceB) &&
            (constraint.pieceA != diagnostic.pieceB ||
             constraint.pieceB != diagnostic.pieceA)) {
            throw std::invalid_argument(
                "Constraint evidence diagnostic does not match its constraint");
        }
        const std::array weights{
            diagnostic.effectiveParallelWindingWeight,
            diagnostic.effectivePerpendicularWindingWeight,
            diagnostic.effectiveParallelSignPenalty,
            diagnostic.effectivePerpendicularSignPenalty,
        };
        if (std::any_of(weights.begin(), weights.end(), [](double weight) {
                return !std::isfinite(weight) || weight < 0.0;
            }) ||
            !std::isfinite(diagnostic.effectiveParallelWindingDistance)) {
            throw std::invalid_argument(
                "Constraint evidence diagnostic has invalid effective values");
        }

        const bool perpendicularHardSign =
            diagnostic.hardPerpendicularSign;
        const bool parallelHardSign = diagnostic.hardParallelSign;
        const bool hardSign = perpendicularHardSign || parallelHardSign;
        const double parallelMagnitudeWeight =
            diagnostic.parallelMagnitudePresent
            ? diagnostic.effectiveParallelWindingWeight
            : 0.0;
        const double perpendicularMagnitudeWeight =
            diagnostic.perpendicularMagnitudePresent
            ? diagnostic.effectivePerpendicularWindingWeight
            : 0.0;
        const double parallelSignWeight =
            diagnostic.effectiveParallelSignPenalty;
        const double perpendicularSignWeight =
            diagnostic.effectivePerpendicularSignPenalty;
        const double measurementWeight = constraint.hardContinuity
            ? parallelMagnitudeWeight
            : parallelMagnitudeWeight + perpendicularMagnitudeWeight +
                parallelSignWeight + perpendicularSignWeight;
        addMeasurementEndpoint(
            diagnostic.pieceA, measurementWeight, hardSign);
        addMeasurementEndpoint(
            diagnostic.pieceB, measurementWeight, hardSign);

        if (constraint.hardContinuity) {
            addTerm(
                diagnostic,
                FiberTraceConstraintEvidenceClass::Continuity,
                parallelMagnitudeWeight);
            continue;
        }
        addTerm(
            diagnostic,
            FiberTraceConstraintEvidenceClass::PerpendicularMagnitude,
            perpendicularMagnitudeWeight);
        addTerm(
            diagnostic,
            FiberTraceConstraintEvidenceClass::PerpendicularSign,
            perpendicularSignWeight,
            perpendicularHardSign);
        if (!diagnostic.parallelMagnitudePresent)
            continue;
        const auto parallelClass =
            std::abs(diagnostic.effectiveParallelWindingDistance) <= kEpsilon
            ? FiberTraceConstraintEvidenceClass::ParallelSameMagnitude
            : FiberTraceConstraintEvidenceClass::ParallelOtherMagnitude;
        addTerm(
            diagnostic,
            parallelClass,
            parallelMagnitudeWeight);
        addTerm(
            diagnostic,
            FiberTraceConstraintEvidenceClass::ParallelSign,
            parallelSignWeight,
            parallelHardSign);
    }
    return result;
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

const char* fiberTraceWindingDecisionConfidenceName(
    FiberTraceWindingDecisionConfidence mode) noexcept
{
    switch (mode) {
    case FiberTraceWindingDecisionConfidence::Legacy:
        return "legacy";
    case FiberTraceWindingDecisionConfidence::Linear:
        return "linear";
    case FiberTraceWindingDecisionConfidence::Cosine:
        return "cosine";
    }
    return "invalid";
}

const char* fiberTraceWindingNormalConfidenceName(
    FiberTraceWindingNormalConfidence mode) noexcept
{
    switch (mode) {
    case FiberTraceWindingNormalConfidence::None:
        return "none";
    case FiberTraceWindingNormalConfidence::Linear:
        return "linear";
    case FiberTraceWindingNormalConfidence::Cosine:
        return "cosine";
    }
    return "invalid";
}

FiberTraceInterleavedWindingReport
solveFiberTraceJointGridWindingBeliefPropagation(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceJointGridWindingConfig& config,
    const FiberTraceJointGridProgressCallback& progress,
    std::span<const FiberTraceFixedOrientation> fixedOrientations,
    std::span<const FiberTraceFixedWindingState> fixedStates,
    std::span<const FiberTraceJointGridInitialState> initialStates,
    FiberTraceJointGridInitializationMode initializationMode)
{
    const auto started = std::chrono::steady_clock::now();
    validateJointGridConfig(config);
    const auto fixedClasses = fixedJointClasses(
        fixedOrientations, constraints.pieces.size());
    const double preparationScale = config.fixedMeasurementScale.value_or(1.0);
    const auto prepared = prepareWinding(
        constraints,
        topology,
        config,
        fixedClasses,
        true,
        preparationScale);
    if (!fixedStates.empty() && !hasFixedCalibration(config)) {
        throw std::invalid_argument(
            "Fixed winding states require fixed phase and scale");
    }
    const auto exactStates = fixedJointStates(
        prepared, fixedStates, fixedClasses);
    if (!initialStates.empty() && !exactStates.byNode.empty()) {
        throw std::invalid_argument(
            "Fixed and initial winding states cannot be combined");
    }
    if (!initialStates.empty() && !hasFixedCalibration(config)) {
        throw std::invalid_argument(
            "Initial winding states require fixed phase and scale");
    }
    const auto initial = initialJointStates(
        prepared, initialStates, fixedClasses, config);
    if (!exactStates.byNode.empty()) {
        for (const auto& edge : prepared.edges) {
            if (!exactStates.byNode[edge.a] ||
                !exactStates.byNode[edge.b]) {
                continue;
            }
            const auto& a = *exactStates.byNode[edge.a];
            const auto& b = *exactStates.byNode[edge.b];
            if (!hardContinuityCompatible(
                    edge, a, b, config.enforceHardSplitContinuity)) {
                throw std::runtime_error(
                    "Fixed winding states violate hard continuity");
            }
            const std::size_t component = prepared.componentByNode[edge.a];
            const int sign = *exactStates.phaseSignByComponent[component];
            const double predicted = static_cast<double>(
                    b.winding - a.winding) +
                classOffset(
                    b.orientation,
                    sign,
                    *config.fixedPhaseMagnitude) -
                classOffset(
                    a.orientation,
                    sign,
                    *config.fixedPhaseMagnitude);
            if (!hardWindingSignCompatible(
                    edge,
                    predicted,
                    config,
                    *config.fixedMeasurementScale)) {
                throw std::runtime_error(
                    "Fixed winding states violate a hard sign constraint");
            }
        }
    }
    std::vector<std::optional<double>> fixedContinuous;
    if (!exactStates.byNode.empty()) {
        fixedContinuous.resize(prepared.piecesByNode.size());
        for (std::size_t node = 0; node < exactStates.byNode.size(); ++node) {
            if (!exactStates.byNode[node])
                continue;
            const auto& state = *exactStates.byNode[node];
            const int sign = *exactStates.phaseSignByComponent.at(
                prepared.componentByNode[node]);
            fixedContinuous[node] = static_cast<double>(state.winding) +
                classOffset(
                    state.orientation,
                    sign,
                    *config.fixedPhaseMagnitude);
        }
    }
    const auto continuousStarted = std::chrono::steady_clock::now();
    double continuousResidual = 0.0;
    const auto continuousNodes = solveContinuous(
        prepared, continuousResidual, fixedContinuous);
    const double continuousSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - continuousStarted).count();
    const auto discreteStarted = std::chrono::steady_clock::now();
    const auto round = solveJointGridRound(
        prepared,
        continuousNodes,
        fixedClasses,
        exactStates,
        initial,
        initializationMode,
        config,
        progress,
        started);
    const double discreteSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - discreteStarted).count();
    auto report = makeJointGridReport(
        prepared,
        constraints,
        continuousNodes,
        continuousResidual,
        continuousSeconds,
        discreteSeconds,
        round,
        config);
    if (!initial.byNode.empty()) {
        report.initialStateDecodedEnergy = gridDecodedEnergy(
            prepared,
            initial.byNode,
            initial.phaseSignByComponent,
            fixedClasses,
            fixedCalibrationCell(config),
            config,
            continuousNodes);
    }
    return report;
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
        const bool signedEvidence =
            diagnostic.hardPerpendicularSign || diagnostic.hardParallelSign ||
            diagnostic.effectivePerpendicularSignPenalty > 0.0 ||
            diagnostic.effectiveParallelSignPenalty > 0.0 ||
            (diagnostic.effectivePerpendicularWindingWeight > 0.0 &&
             diagnostic.effectivePerpendicularSignedDelta.has_value()) ||
            (diagnostic.effectiveParallelWindingWeight > 0.0 &&
             diagnostic.effectiveSignedParallelDelta.has_value());
        const bool expected =
            diagnostic.effectivePerpendicularWindingWeight > 0.0 ||
            diagnostic.effectiveParallelWindingWeight > 0.0 ||
            diagnostic.effectivePerpendicularSignPenalty > 0.0 ||
            diagnostic.effectiveParallelSignPenalty > 0.0 ||
            diagnostic.hardPerpendicularSign || diagnostic.hardParallelSign;
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
        !std::isfinite(config.pieceBreakCost) ||
        config.pieceBreakCost < 0.0 ||
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
                        continuousNodes[node]));
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
    report.pieceBreakCost = config.pieceBreakCost;
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
        const bool signedEvidence =
            diagnostic.hardPerpendicularSign || diagnostic.hardParallelSign ||
            diagnostic.effectivePerpendicularSignPenalty > 0.0 ||
            diagnostic.effectiveParallelSignPenalty > 0.0 ||
            (diagnostic.effectivePerpendicularWindingWeight > 0.0 &&
             diagnostic.effectivePerpendicularSignedDelta.has_value()) ||
            (diagnostic.effectiveParallelWindingWeight > 0.0 &&
             diagnostic.effectiveSignedParallelDelta.has_value());
        const bool expected =
            diagnostic.effectivePerpendicularWindingWeight > 0.0 ||
            diagnostic.effectiveParallelWindingWeight > 0.0 ||
            diagnostic.effectivePerpendicularSignPenalty > 0.0 ||
            diagnostic.effectiveParallelSignPenalty > 0.0 ||
            diagnostic.hardPerpendicularSign || diagnostic.hardParallelSign;
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
    const auto predictedDelta = [&] (
        std::size_t edgeIndex, const JointState& a, const JointState& b) {
            const auto& edge = prepared.edges[edgeIndex];
            const int sign = best->parameters.componentSign.at(
                prepared.componentByNode[edge.a]);
            const double latent = static_cast<double>(b.winding - a.winding) +
                classOffset(b.orientation, sign, best->parameters.phase) -
                classOffset(a.orientation, sign, best->parameters.phase);
            return latent;
        };
    report.hardSignProjectedDefects = projectDecodedHardSigns(
        prepared,
        decoded,
        activeConfidence,
        best->parameters.scale,
        predictedDelta);
    if (config.enforceHardSplitContinuity) {
        projectDecodedHardContinuity(prepared, decoded, activeConfidence);
        report.hardSignProjectedDefects += projectDecodedHardSigns(
            prepared,
            decoded,
            activeConfidence,
            best->parameters.scale,
            predictedDelta);
        projectDecodedHardContinuity(prepared, decoded, activeConfidence);
    }
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
