#include "vc/fiber_tracer/FiberTraceWindingLeastSquares.hpp"

#include <ceres/ceres.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace vc::fiber_tracer
{
namespace
{

constexpr double kEpsilon = 1.0e-12;

struct PairCoefficients {
    double parallelOrientation = 0.0;
    double perpendicularOrientation = 0.0;
    double windingMagnitude = 0.0;
    double windingTarget = 0.0;
    bool unsignedWinding = false;
    double sign = 0.0;
    double signCost = 0.0;
    double signMargin = 0.0;
    double continuation = 0.0;
    double pieceBreak = 0.0;
};

struct PairResidual {
    PairCoefficients coefficients;

    template <typename T>
    bool operator()(
        const T* const horizontalA,
        const T* const activityA,
        const T* const windingA,
        const T* const horizontalB,
        const T* const activityB,
        const T* const windingB,
        T* residuals) const
    {
        const T gate = activityA[0] * activityB[0];
        const T delta = windingB[0] - windingA[0];
        residuals[0] = T(coefficients.parallelOrientation) * gate *
            (horizontalA[0] - horizontalB[0]);
        residuals[1] = T(coefficients.perpendicularOrientation) * gate *
            (horizontalA[0] + horizontalB[0] - T(1.0));
        const T windingResidual = coefficients.unsignedWinding
            ? ceres::abs(delta) - T(coefficients.windingTarget)
            : delta - T(coefficients.windingTarget);
        residuals[2] = T(coefficients.windingMagnitude) * gate *
            windingResidual;
        const T signSlack = T(coefficients.signMargin) -
            T(coefficients.sign) * delta;
        residuals[3] = signSlack > T(0.0)
            ? T(coefficients.signCost) * gate * signSlack
            : T(0.0);
        residuals[4] = T(coefficients.continuation) * gate *
            (horizontalA[0] - horizontalB[0]);
        residuals[5] = T(coefficients.continuation) * gate * delta;
        residuals[6] = T(coefficients.pieceBreak) *
            (activityA[0] - activityB[0]);
        return true;
    }
};

struct NodeResidual {
    double defect = 0.0;
    double extremeness = 0.0;

    template <typename T>
    bool operator()(
        const T* const horizontalness,
        const T* const activity,
        T* residuals) const
    {
        residuals[0] = T(defect) * (T(1.0) - activity[0]);
        residuals[1] = T(extremeness) * activity[0] *
            horizontalness[0] * (T(1.0) - horizontalness[0]);
        return true;
    }
};

double checkedSqrt(double value, const char* name)
{
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument(std::string{name} + " must be finite and nonnegative");
    return std::sqrt(value);
}

PairCoefficients pairCoefficients(
    const FiberTracePreparedWindingMeasurement& measurement,
    const FiberTraceWindingLeastSquaresConfig& config)
{
    PairCoefficients result;
    if (measurement.continuity)
        return result;

    result.parallelOrientation = checkedSqrt(
        measurement.parallel, "Ceres parallel orientation coefficient");
    result.perpendicularOrientation = checkedSqrt(
        measurement.perpendicular,
        "Ceres perpendicular orientation coefficient");
    const double parallelWeight = measurement.parallelMagnitudePresent
        ? measurement.parallelMultiplier * measurement.parallelConfidence
        : 0.0;
    const double perpendicularWeight =
        measurement.perpendicularMagnitudePresent &&
            measurement.perpendicularSignedDelta
        ? measurement.perpendicularMultiplier *
              measurement.perpendicularConfidence
        : 0.0;
    if (parallelWeight > 0.0) {
        result.windingMagnitude = checkedSqrt(
            parallelWeight,
            "Ceres parallel winding coefficient");
        if (measurement.parallelSignedDelta) {
            result.windingTarget = *measurement.parallelSignedDelta;
        } else {
            result.windingTarget = measurement.parallelDistance;
            result.unsignedWinding = true;
        }
    } else if (perpendicularWeight > 0.0) {
        result.windingMagnitude = checkedSqrt(
            perpendicularWeight,
            "Ceres perpendicular winding coefficient");
        result.windingTarget = *measurement.perpendicularSignedDelta;
    }

    const bool parallelSign = measurement.parallelSignPresent &&
        measurement.parallelSignedDelta &&
        *measurement.parallelSignedDelta != 0.0;
    const bool perpendicularSign = measurement.perpendicularSignPresent &&
        measurement.perpendicularSignedDelta &&
        *measurement.perpendicularSignedDelta != 0.0;
    if (parallelSign || perpendicularSign) {
        const bool useParallel = parallelSign;
        const double target = useParallel
            ? *measurement.parallelSignedDelta
            : *measurement.perpendicularSignedDelta;
        const bool hard = useParallel
            ? measurement.hardParallelSign
            : measurement.hardPerpendicularSign;
        const double finiteCost = useParallel
            ? measurement.parallelSignPenalty
            : measurement.perpendicularSignPenalty;
        const double signWeight = useParallel
            ? config.parallelSignWeight
            : config.perpendicularSignWeight;
        const double hardCost = config.hardConstraintCost * signWeight *
            measurement.selectedConfidence;
        result.sign = std::copysign(1.0, target);
        result.signCost = checkedSqrt(
            hard ? hardCost : finiteCost, "Ceres winding sign coefficient");
        result.signMargin = config.signMargin;
    }
    return result;
}

FiberTraceWindingLeastSquaresCostSummary evaluateCosts(
    const FiberTracePreparedWindingModel& model,
    std::span<const FiberTraceWindingLeastSquaresState> states,
    std::span<const std::size_t> incident,
    const FiberTraceWindingLeastSquaresConfig& config)
{
    FiberTraceWindingLeastSquaresCostSummary result;
    for (std::size_t piece = 0; piece < states.size(); ++piece) {
        const auto& state = states[piece];
        const double defect = checkedSqrt(
            config.defectCost * static_cast<double>(incident[piece]),
            "Ceres defect cost") * (1.0 - state.activity);
        const double extremeness = checkedSqrt(
            config.orientationExtremenessCost *
                static_cast<double>(incident[piece]),
            "Ceres orientation-extremeness cost") * state.activity *
            state.horizontalness * (1.0 - state.horizontalness);
        result.defect += defect * defect;
        result.orientationExtremeness += extremeness * extremeness;
    }
    for (const auto& edge : model.edges) {
        const auto& a = states[edge.a];
        const auto& b = states[edge.b];
        const double gate = a.activity * b.activity;
        const double delta = b.winding - a.winding;
        for (const auto& measurement : edge.measurements) {
            const auto coefficients = pairCoefficients(measurement, config);
            const double parallelOrientation =
                coefficients.parallelOrientation * gate *
                (a.horizontalness - b.horizontalness);
            const double perpendicularOrientation =
                coefficients.perpendicularOrientation * gate *
                (a.horizontalness + b.horizontalness - 1.0);
            result.orientation += parallelOrientation * parallelOrientation +
                perpendicularOrientation * perpendicularOrientation;
            const double magnitudeError = coefficients.unsignedWinding
                ? std::abs(delta) - coefficients.windingTarget
                : delta - coefficients.windingTarget;
            const double magnitude = coefficients.windingMagnitude * gate *
                magnitudeError;
            result.windingMagnitude += magnitude * magnitude;
            const double signSlack = coefficients.signMargin -
                coefficients.sign * delta;
            const double sign = signSlack > 0.0
                ? coefficients.signCost * gate * signSlack
                : 0.0;
            result.windingSign += sign * sign;
        }
        const double continuation = edge.containsHardContinuity &&
                config.enforceHardSplitContinuity
            ? checkedSqrt(
                  config.hardConstraintCost, "Ceres hard constraint cost")
            : 0.0;
        const double continuationH = continuation * gate *
            (a.horizontalness - b.horizontalness);
        const double continuationW = continuation * gate * delta;
        result.continuation += continuationH * continuationH +
            continuationW * continuationW;
        const double pieceBreak = edge.containsHardContinuity
            ? checkedSqrt(config.pieceBreakCost, "Ceres piece-break cost") *
                  (a.activity - b.activity)
            : 0.0;
        result.pieceBreak += pieceBreak * pieceBreak;
    }
    return result;
}

class ProgressCallback final : public ceres::IterationCallback {
public:
    ProgressCallback(
        std::size_t maximumIterations,
        FiberTraceWindingLeastSquaresProgressCallback callback,
        std::chrono::steady_clock::time_point started)
        : maximumIterations_{maximumIterations},
          callback_{std::move(callback)},
          started_{started}
    {
    }

    ceres::CallbackReturnType operator()(
        const ceres::IterationSummary& summary) override
    {
        if (callback_) {
            callback_({
                FiberTraceWindingLeastSquaresProgressPhase::Iterating,
                static_cast<std::size_t>(summary.iteration + 1),
                maximumIterations_,
                summary.cost,
                summary.gradient_max_norm,
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started_).count(),
            });
        }
        return ceres::SOLVER_CONTINUE;
    }

private:
    std::size_t maximumIterations_;
    FiberTraceWindingLeastSquaresProgressCallback callback_;
    std::chrono::steady_clock::time_point started_;
};

}  // namespace

double FiberTraceWindingLeastSquaresCostSummary::total() const noexcept
{
    return orientation + windingMagnitude + windingSign + defect +
        orientationExtremeness + continuation + pieceBreak;
}

FiberTraceWindingLeastSquaresReport solveFiberTraceWindingLeastSquares(
    const FiberTraceConstraintReport& constraints,
    const FiberTraceBeliefTopology& topology,
    const FiberTraceWindingLeastSquaresConfig& config,
    std::span<const FiberTraceWindingLeastSquaresState> initialStates,
    std::span<const FiberTraceWindingLeastSquaresFixedState> fixedStates,
    const FiberTraceWindingLeastSquaresProgressCallback& progress)
{
    const auto started = std::chrono::steady_clock::now();
    const std::size_t pieceCount = constraints.pieces.size();
    if (pieceCount == 0 || topology.pieceLines.size() != pieceCount ||
        topology.pieceCenterDistanceBaseVoxels.size() != pieceCount ||
        (!initialStates.empty() && initialStates.size() != pieceCount) ||
        (!fixedStates.empty() && fixedStates.size() != pieceCount)) {
        throw std::invalid_argument(
            "Ceres winding inputs do not match represented pieces");
    }
    if (!std::isfinite(config.defectCost) || config.defectCost < 0.0 ||
        !std::isfinite(config.pieceBreakCost) || config.pieceBreakCost < 0.0 ||
        !std::isfinite(config.orientationExtremenessCost) ||
        config.orientationExtremenessCost < 0.0 ||
        !std::isfinite(config.hardConstraintCost) ||
        !(config.hardConstraintCost > 0.0) ||
        !std::isfinite(config.measurementScale) ||
        !(config.measurementScale > 0.0) || config.maximumIterations == 0 ||
        config.parallelWorkers == 0) {
        throw std::invalid_argument("Ceres winding config is invalid");
    }
    if (progress) {
        progress({
            FiberTraceWindingLeastSquaresProgressPhase::Preparing,
            0,
            config.maximumIterations,
            0.0,
            0.0,
            0.0,
        });
    }

    FiberTraceWindingLeastSquaresReport report;
    report.states.assign(pieceCount, {});
    if (!initialStates.empty())
        std::copy(initialStates.begin(), initialStates.end(), report.states.begin());
    for (auto& state : report.states) {
        state.horizontalness = std::clamp(state.horizontalness, 0.0, 1.0);
        state.activity = std::clamp(state.activity, 0.0, 1.0);
        if (!std::isfinite(state.winding))
            state.winding = 0.0;
    }
    const auto model = prepareFiberTraceWindingModel(
        constraints, topology, config, {}, true, config.measurementScale);
    if (model.piecesByNode.size() != pieceCount) {
        throw std::logic_error(
            "Ceres winding currently requires one prepared node per piece");
    }
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        if (model.pieceToNode[piece] != piece ||
            model.piecesByNode[piece] != std::vector<std::size_t>{piece}) {
            throw std::logic_error(
                "Ceres winding prepared-node mapping is not one-to-one");
        }
    }
    report.factorDiagnostics = model.diagnostics;
    report.factors = std::accumulate(
        model.edges.begin(), model.edges.end(), std::size_t{0},
        [](std::size_t total, const auto& edge) {
            return total + edge.measurements.size();
        });
    report.incidentEffectiveConstraints = model.incidentMeasurements;
    report.componentByPiece = model.integerGaugeByNode;
    std::vector<unsigned char> fixedAll(pieceCount, 0);
    std::vector<unsigned char> fixedHorizontalness(pieceCount, 0);
    std::vector<unsigned char> fixedWinding(pieceCount, 0);
    if (!fixedStates.empty()) {
        for (std::size_t piece = 0; piece < pieceCount; ++piece) {
            if (!fixedStates[piece].fixed)
                continue;
            const auto& state = fixedStates[piece].state;
            if (!std::isfinite(state.horizontalness) ||
                state.horizontalness < 0.0 || state.horizontalness > 1.0 ||
                !std::isfinite(state.activity) || state.activity < 0.0 ||
                state.activity > 1.0 || !std::isfinite(state.winding)) {
                throw std::invalid_argument(
                    "Ceres fixed winding state is invalid");
            }
            report.states[piece] = fixedStates[piece].state;
            fixedAll[piece] = 1;
            fixedHorizontalness[piece] = 1;
            fixedWinding[piece] = 1;
        }
    }
    for (std::size_t component = 0;
         component < model.gaugeNodeByComponent.size(); ++component) {
        bool hasFixed = false;
        for (std::size_t piece = 0; piece < pieceCount; ++piece) {
            hasFixed = hasFixed ||
                (model.componentByNode[piece] == component &&
                 fixedHorizontalness[piece] != 0);
        }
        if (!hasFixed) {
            const std::size_t gauge = model.gaugeNodeByComponent[component];
            report.states[gauge].horizontalness = 1.0;
            fixedHorizontalness[gauge] = 1;
        }
    }
    for (std::size_t component = 0;
         component < model.integerGaugeNodes.size(); ++component) {
        bool hasFixed = false;
        for (std::size_t piece = 0; piece < pieceCount; ++piece) {
            hasFixed = hasFixed ||
                (model.integerGaugeByNode[piece] == component &&
                 fixedWinding[piece] != 0);
        }
        const std::size_t gauge = model.integerGaugeNodes[component];
        if (!hasFixed && report.incidentEffectiveConstraints[gauge] != 0) {
            report.states[gauge].winding = 0.0;
            fixedWinding[gauge] = 1;
            report.gaugePieces.push_back(gauge);
        }
    }

    report.initialCosts = evaluateCosts(
        model,
        report.states,
        report.incidentEffectiveConstraints,
        config);

    ceres::Problem problem;
    for (std::size_t piece = 0; piece < pieceCount; ++piece) {
        auto& state = report.states[piece];
        problem.AddParameterBlock(&state.horizontalness, 1);
        problem.AddParameterBlock(&state.activity, 1);
        problem.AddParameterBlock(&state.winding, 1);
        problem.SetParameterLowerBound(&state.horizontalness, 0, 0.0);
        problem.SetParameterUpperBound(&state.horizontalness, 0, 1.0);
        problem.SetParameterLowerBound(&state.activity, 0, 0.0);
        problem.SetParameterUpperBound(&state.activity, 0, 1.0);
        if (fixedAll[piece] != 0) {
            problem.SetParameterBlockConstant(&state.horizontalness);
            problem.SetParameterBlockConstant(&state.activity);
            problem.SetParameterBlockConstant(&state.winding);
        } else {
            const auto residual = NodeResidual{
                checkedSqrt(
                    config.defectCost * static_cast<double>(
                        report.incidentEffectiveConstraints[piece]),
                    "Ceres defect cost"),
                checkedSqrt(
                    config.orientationExtremenessCost *
                        static_cast<double>(
                            report.incidentEffectiveConstraints[piece]),
                    "Ceres orientation-extremeness cost"),
            };
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<NodeResidual, 2, 1, 1>(
                    new NodeResidual(residual)),
                nullptr,
                &state.horizontalness,
                &state.activity);
            if (fixedHorizontalness[piece] != 0)
                problem.SetParameterBlockConstant(&state.horizontalness);
            if (fixedWinding[piece] != 0)
                problem.SetParameterBlockConstant(&state.winding);
        }
    }
    for (const auto& edge : model.edges) {
        auto& a = report.states.at(edge.a);
        auto& b = report.states.at(edge.b);
        for (const auto& measurement : edge.measurements) {
            if (measurement.continuity)
                continue;
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<
                    PairResidual, 7, 1, 1, 1, 1, 1, 1>(
                        new PairResidual{
                            pairCoefficients(measurement, config)}),
                nullptr,
                &a.horizontalness,
                &a.activity,
                &a.winding,
                &b.horizontalness,
                &b.activity,
                &b.winding);
        }
        if (edge.containsHardContinuity) {
            PairCoefficients continuity;
            if (config.enforceHardSplitContinuity) {
                continuity.continuation = checkedSqrt(
                    config.hardConstraintCost,
                    "Ceres hard constraint cost");
            }
            continuity.pieceBreak = checkedSqrt(
                config.pieceBreakCost, "Ceres piece-break cost");
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<
                    PairResidual, 7, 1, 1, 1, 1, 1, 1>(
                        new PairResidual{continuity}),
                nullptr,
                &a.horizontalness,
                &a.activity,
                &a.winding,
                &b.horizontalness,
                &b.activity,
                &b.winding);
        }
    }

    ceres::Solver::Options options;
    options.max_num_iterations = static_cast<int>(std::min<std::size_t>(
        config.maximumIterations,
        static_cast<std::size_t>(std::numeric_limits<int>::max())));
    options.num_threads = static_cast<int>(std::min<std::size_t>(
        config.parallelWorkers,
        static_cast<std::size_t>(std::numeric_limits<int>::max())));
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.logging_type = ceres::SILENT;
    options.update_state_every_iteration = true;
    ProgressCallback callback{config.maximumIterations, progress, started};
    if (progress)
        options.callbacks.push_back(&callback);

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    report.iterations = summary.iterations.size();
    report.effectiveWorkers = config.parallelWorkers;
    report.initialCost = summary.initial_cost;
    report.finalCost = summary.final_cost;
    report.solutionUsable = summary.IsSolutionUsable();
    report.status = ceres::TerminationTypeToString(summary.termination_type);
    report.briefReport = summary.BriefReport();
    report.solveSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    report.finalCosts = evaluateCosts(
        model,
        report.states,
        report.incidentEffectiveConstraints,
        config);
    if (progress) {
        progress({
            FiberTraceWindingLeastSquaresProgressPhase::Complete,
            report.iterations,
            config.maximumIterations,
            report.finalCost,
            0.0,
            report.solveSeconds,
        });
    }
    return report;
}

FiberTraceInterleavedWindingReport makeFiberTraceInterleavedWindingReport(
    const FiberTraceWindingLeastSquaresReport& leastSquares,
    const FiberTraceWindingLeastSquaresConfig& config)
{
    FiberTraceInterleavedWindingReport result;
    const std::size_t pieces = leastSquares.states.size();
    result.solver = FiberTraceWindingSolver::Ceres;
    result.orientationMode = FiberTraceWindingOrientationMode::Joint;
    result.calibrationMode = FiberTraceWindingCalibrationMode::Fixed;
    result.phaseMagnitude = 0.5;
    result.measurementScale = config.measurementScale;
    result.defectUnaryCost = config.defectCost;
    result.pieceBreakCost = config.pieceBreakCost;
    result.variables = pieces;
    result.factors = leastSquares.factors;
    result.connectedComponents = leastSquares.gaugePieces.size();
    result.gaugePieces = leastSquares.gaugePieces;
    result.factorDiagnostics = leastSquares.factorDiagnostics;
    result.discreteSolveSeconds = leastSquares.solveSeconds;
    result.messageIterations = leastSquares.iterations;
    result.messageConverged = leastSquares.solutionUsable;
    result.effectiveWorkers = leastSquares.effectiveWorkers;
    result.decodedEnergy = leastSquares.finalCosts.total();
    result.decodedDataEnergy = result.decodedEnergy;
    result.status = leastSquares.status;
    result.calibrationConverged = true;
    result.windingValid.resize(pieces);
    result.continuousWinding.resize(pieces);
    result.mapWinding.resize(pieces);
    result.posteriorMeanWinding.resize(pieces);
    result.mapProbability.resize(pieces);
    result.entropy.assign(pieces, 0.0);
    result.candidateMinimum.resize(pieces);
    result.candidateMaximum.resize(pieces);
    result.componentByPiece = leastSquares.componentByPiece;
    result.integerGaugeByPiece = leastSquares.componentByPiece;
    result.classAProbability.resize(pieces);
    result.mixedProbability.resize(pieces);
    result.classBProbability.resize(pieces);
    result.posteriorMeanLatentCoordinate.resize(pieces);
    result.mapLatentCoordinate.resize(pieces);
    result.mapOrientationByPiece.resize(pieces);
    result.incidentSignedConstraints.assign(pieces, 0);
    result.incidentSkippedConstraints.assign(pieces, 0);
    const std::size_t componentCount = leastSquares.componentByPiece.empty()
        ? 0
        : 1 + *std::max_element(
              leastSquares.componentByPiece.begin(),
              leastSquares.componentByPiece.end());
    result.componentPhaseSign.assign(componentCount, 1);
    result.componentPositivePhaseSignProbability.assign(componentCount, 1.0);
    for (std::size_t piece = 0; piece < pieces; ++piece) {
        const auto& state = leastSquares.states[piece];
        const double horizontal = state.activity * state.horizontalness;
        const double mixed = 1.0 - state.activity;
        const double vertical = state.activity * (1.0 - state.horizontalness);
        result.classAProbability[piece] = horizontal;
        result.mixedProbability[piece] = mixed;
        result.classBProbability[piece] = vertical;
        const std::array probabilities{horizontal, mixed, vertical};
        const std::size_t map = static_cast<std::size_t>(std::distance(
            probabilities.begin(),
            std::max_element(probabilities.begin(), probabilities.end())));
        result.mapOrientationByPiece[piece] = map == 0
            ? FiberTraceFixedOrientation::Horizontal
            : map == 1 ? FiberTraceFixedOrientation::Mixed
                       : FiberTraceFixedOrientation::Vertical;
        result.windingValid[piece] = map == 1 ? 0 : 1;
        result.continuousWinding[piece] = state.winding;
        result.mapWinding[piece] = static_cast<int>(std::llround(state.winding));
        result.posteriorMeanWinding[piece] = state.winding;
        result.mapProbability[piece] = probabilities[map];
        result.candidateMinimum[piece] = result.mapWinding[piece];
        result.candidateMaximum[piece] = result.mapWinding[piece];
        result.posteriorMeanLatentCoordinate[piece] = state.winding;
        result.mapLatentCoordinate[piece] = state.winding;
    }
    for (const auto& diagnostic : result.factorDiagnostics) {
        const bool signedEvidence = diagnostic.parallelSignPresent ||
            diagnostic.perpendicularSignPresent;
        const bool expected = signedEvidence ||
            diagnostic.effectiveParallelWindingWeight > 0.0 ||
            diagnostic.effectivePerpendicularWindingWeight > 0.0;
        for (const std::size_t piece : {diagnostic.pieceA, diagnostic.pieceB}) {
            if (signedEvidence)
                ++result.incidentSignedConstraints[piece];
            else if (expected)
                ++result.incidentSkippedConstraints[piece];
        }
    }
    return result;
}

}  // namespace vc::fiber_tracer
