#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberTraceBeliefPropagation.hpp"
#include "vc/fiber_tracer/FiberTraceWindingBeliefPropagation.hpp"
#include "vc/fiber_tracer/FiberTraceWindingLeastSquares.hpp"
#include "vc/fiber_tracer/FiberTraceWindingOrderedCuts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <string>
#include <vector>

namespace
{

using namespace vc::fiber_tracer;

std::vector<FiberletCropTraceLine> lines(std::size_t count)
{
    std::vector<FiberletCropTraceLine> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        result[index].pointsBaseXYZ = {
            {-3.0, static_cast<double>(index), 0.0},
            {3.0, static_cast<double>(index), 0.0},
        };
    }
    return result;
}

FiberTraceConstraintReport pieces(std::size_t count)
{
    FiberTraceConstraintReport report;
    report.inputTraces = count;
    report.pieces.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        report.pieces[index].traceIndex = index;
        report.pieces[index].pieceIndex = 0;
        report.pieces[index].beginArcBaseVoxels = 0.0;
        report.pieces[index].endArcBaseVoxels = 6.0;
    }
    return report;
}

void addMeasured(
    FiberTraceConstraintReport& report,
    std::size_t a,
    std::size_t b,
    double parallel,
    double signedDelta,
    std::size_t normalComponent = 0)
{
    FiberTraceConstraint constraint;
    constraint.pieceA = a;
    constraint.pieceB = b;
    constraint.parallelScore = parallel;
    constraint.perpendicularScore = 1.0 - parallel;
    constraint.windingDistance = std::abs(signedDelta);
    constraint.signedWindingDelta = signedDelta;
    constraint.parallelWindingDistance = std::abs(signedDelta);
    constraint.signedParallelWindingDelta = signedDelta;
    constraint.windingNormalComponent = normalComponent;
    report.constraints.push_back(constraint);
}

FiberTraceBeliefTopology topology(
    const std::vector<FiberletCropTraceLine>& source,
    const FiberTraceConstraintReport& report)
{
    return prepareFiberTraceBeliefTopology(
        source, report, {-5.0, -5.0, -5.0}, {5.0, 5.0, 5.0});
}

void useUnitClassWeights(FiberTraceWindingBeliefPropagationConfig& config)
{
    config.perpendicularNextWeight = 1.0;
    config.perpendicularFarWeight = 1.0;
    config.parallelSameWeight = 1.0;
    config.parallelOneWeight = 1.0;
    config.parallelFarWeight = 1.0;
}

void useZeroClassWeights(FiberTraceWindingBeliefPropagationConfig& config)
{
    config.perpendicularNextWeight = 0.0;
    config.perpendicularFarWeight = 0.0;
    config.parallelSameWeight = 0.0;
    config.parallelOneWeight = 0.0;
    config.parallelFarWeight = 0.0;
}

FiberTraceWindingBeliefPropagationConfig config()
{
    FiberTraceWindingBeliefPropagationConfig result;
    useUnitClassWeights(result);
    result.enforcePerpendicularWindingSign = true;
    result.enforceParallelWindingSign = false;
    result.perpendicularSignWeight = 1.0;
    result.parallelSignWeight = 1.0;
    result.decisionConfidence =
        FiberTraceWindingDecisionConfidence::Legacy;
    result.normalConfidence = FiberTraceWindingNormalConfidence::None;
    result.finiteSignInfringementCost.reset();
    result.temperature = 0.25;
    result.messageDamping = 1.0;
    result.messageResidualTolerance = 1.0e-12;
    result.maximumMessageIterations = 1000;
    return result;
}

TEST_CASE("Winding class production defaults use the selected reference tuple")
{
    const FiberTraceWindingBeliefPropagationConfig defaults;
    CHECK(defaults.perpendicularNextWeight == 0.0);
    CHECK(defaults.perpendicularFarWeight == 0.0);
    CHECK(defaults.parallelSameWeight == 0.5);
    CHECK(defaults.parallelOneWeight == 4.0);
    CHECK(defaults.parallelFarWeight == 1.0);
    CHECK(defaults.perpendicularSignWeight == 1.0);
    CHECK(defaults.parallelSignWeight == 0.5);
    CHECK(defaults.enforcePerpendicularWindingSign);
    CHECK(defaults.enforceParallelWindingSign);
    CHECK(defaults.decisionConfidence ==
          FiberTraceWindingDecisionConfidence::Cosine);
    CHECK(defaults.normalConfidence ==
          FiberTraceWindingNormalConfidence::Linear);
    REQUIRE(defaults.finiteSignInfringementCost.has_value());
    CHECK(*defaults.finiteSignInfringementCost == 44.0);
    CHECK(defaults.enforceHardSplitContinuity);
    REQUIRE(defaults.hardSignMinimumNormalAlignment.has_value());
    CHECK(*defaults.hardSignMinimumNormalAlignment ==
          doctest::Approx(std::cos(std::numbers::pi / 6.0)));
    CHECK(std::string(fiberTraceWindingDecisionConfidenceName(
              defaults.decisionConfidence)) == "cosine");
    CHECK(std::string(fiberTraceWindingNormalConfidenceName(
              defaults.normalConfidence)) == "linear");
    const FiberTraceJointGridWindingConfig jointDefaults;
    CHECK(jointDefaults.mixedUnaryCost == 100.0);
    const FiberTraceInterleavedWindingConfig interleavedDefaults;
    CHECK(interleavedDefaults.mixedUnaryCost == 100.0);
    const FiberTraceBeliefPropagationConfig orientationDefaults;
    CHECK(orientationDefaults.horizontalnessTemperature == 1.25);
}

TEST_CASE("Ordered winding offsets use phase-aware sign margins")
{
    const auto source = lines(4);
    auto report = pieces(4);
    addMeasured(report, 0, 1, 0.0, 0.5);
    addMeasured(report, 1, 2, 0.0, 0.5);
    addMeasured(report, 2, 3, 0.0, 0.5);
    addMeasured(report, 0, 2, 1.0, 1.0);
    addMeasured(report, 1, 3, 1.0, 1.0);
    const std::vector orientations{
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Vertical,
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Vertical,
    };
    FiberTraceWindingOrderedCutsConfig settings;
    settings.decisionConfidence =
        FiberTraceWindingDecisionConfidence::Legacy;
    settings.normalConfidence = FiberTraceWindingNormalConfidence::None;
    settings.parallelWorkers = 1;
    settings.measurementScale = 1.0;
    settings.maximumIterations = 100;
    const auto solved = solveFiberTraceWindingOrderedCuts(
        report, topology(source, report), orientations, settings);

    REQUIRE(solved.ordering.solutionUsable);
    CHECK(solved.ordering.signFactors.size() == 5);
    CHECK(solved.ordering.signFactors[0].margin == 0.5);
    CHECK(solved.ordering.signFactors[3].margin == 1.0);
    REQUIRE(solved.steps.size() >= 2);
    CHECK(solved.steps.front().signInfringements == 3);
    CHECK(solved.steps[1].signInfringements == 0);
    CHECK(solved.steps[1].windings == 2);
    CHECK(solved.steps[1].windingByPiece == std::vector<int>{0, 0, 1, 1});
    for (std::size_t index = 1; index < solved.steps.size(); ++index) {
        CHECK(solved.steps[index].signInfringements <
              solved.steps[index - 1].signInfringements);
    }
    const auto published = makeFiberTraceOrderedCutsWindingReport(
        solved, settings, 1);
    CHECK(published.mapLatentCoordinate ==
          std::vector<double>{0.0, 0.5, 1.0, 1.5});
    CHECK(published.mapOrientationByPiece == orientations);
}

TEST_CASE("Ordered cuts exclude Mixed pieces and preserve active continuations")
{
    auto source = lines(3);
    source[0].pointsBaseXYZ = {
        {-5.0, 0.0, 0.0},
        {5.0, 0.0, 0.0},
    };
    auto report = pieces(3);
    report.inputTraces = 2;
    report.pieces[0].traceIndex = 0;
    report.pieces[0].pieceIndex = 0;
    report.pieces[1].traceIndex = 0;
    report.pieces[1].pieceIndex = 1;
    report.pieces[1].beginArcBaseVoxels = 4.0;
    report.pieces[1].endArcBaseVoxels = 10.0;
    report.pieces[2].traceIndex = 1;
    FiberTraceConstraint continuation;
    continuation.pieceA = 0;
    continuation.pieceB = 1;
    continuation.arcABaseVoxels = 5.0;
    continuation.arcBBaseVoxels = 5.0;
    continuation.pointABaseXYZ = {0.0, 0.0, 0.0};
    continuation.pointBBaseXYZ = continuation.pointABaseXYZ;
    continuation.perpendicularScore = 0.0;
    continuation.parallelScore = 1.0;
    continuation.hardContinuity = true;
    report.constraints.push_back(continuation);
    addMeasured(report, 0, 2, 1.0, 1.0);
    addMeasured(report, 1, 2, 1.0, 1.0);
    source.resize(2);
    const std::vector orientations{
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Vertical,
    };
    FiberTraceWindingOrderedCutsConfig settings;
    settings.decisionConfidence =
        FiberTraceWindingDecisionConfidence::Legacy;
    settings.normalConfidence = FiberTraceWindingNormalConfidence::None;
    settings.parallelWorkers = 1;
    settings.measurementScale = 1.0;
    const auto solved = solveFiberTraceWindingOrderedCuts(
        report, topology(source, report), orientations, settings);
    REQUIRE_FALSE(solved.steps.empty());
    for (const auto& step : solved.steps) {
        CHECK(step.continuationCuts == 0);
        CHECK(step.windingByPiece[0] == step.windingByPiece[1]);
    }

    auto mixed = orientations;
    mixed[2] = FiberTraceFixedOrientation::Mixed;
    const auto excluded = solveFiberTraceWindingOrderedCuts(
        report, topology(source, report), mixed, settings);
    CHECK(excluded.ordering.activeByPiece ==
          std::vector<unsigned char>{1, 1, 0});
    CHECK(excluded.ordering.signFactors.empty());
    CHECK(excluded.steps.size() == 1);
}

TEST_CASE("Ordered winding config and fixed-offset inputs are validated")
{
    const auto source = lines(2);
    auto report = pieces(2);
    addMeasured(report, 0, 1, 1.0, 1.0);
    const std::vector orientations(
        2, FiberTraceFixedOrientation::Horizontal);
    auto settings = FiberTraceWindingOrderedCutsConfig{};
    settings.signMarginWeight = -1.0;
    CHECK_THROWS_AS(
        solveFiberTraceWindingOrderedCuts(
            report, topology(source, report), orientations, settings),
        std::invalid_argument);
    settings = {};
    std::vector<FiberTraceWindingOrderedCutsFixedOffset> fixed(2);
    fixed[0] = {true, 2.25};
    const auto fitted = fitFiberTraceWindingOrderedOffsets(
        report,
        topology(source, report),
        orientations,
        settings,
        fixed);
    CHECK(fitted.offsetByPiece[0] == 2.25);
    const std::vector<unsigned char> noActive(2, 0);
    const auto empty = fitFiberTraceWindingOrderedOffsets(
        report,
        topology(source, report),
        orientations,
        settings,
        {},
        {},
        noActive);
    CHECK(empty.solutionUsable);
    CHECK(empty.status == "EMPTY");
    CHECK(empty.activeByPiece == noActive);
    CHECK(empty.orientationByPiece == orientations);
    const std::vector<unsigned char> wrongActive(1, 0);
    CHECK_THROWS_AS(
        fitFiberTraceWindingOrderedOffsets(
            report,
            topology(source, report),
            orientations,
            settings,
            {},
            {},
            wrongActive),
        std::invalid_argument);
    const std::vector<FiberTraceWindingOrderedCutsFixedOffset> wrong(1);
    CHECK_THROWS_AS(
        fitFiberTraceWindingOrderedOffsets(
            report,
            topology(source, report),
            orientations,
            settings,
            wrong),
        std::invalid_argument);
    settings.continuationWeight =
        std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(
        solveFiberTraceWindingOrderedCuts(
            report, topology(source, report), orientations, settings),
        std::invalid_argument);
}

TEST_CASE("Ordered violation ranking uses exact percentages and trace incidence")
{
    auto report = pieces(5);
    report.inputTraces = 3;
    report.pieces[0].traceIndex = 0;
    report.pieces[1].traceIndex = 0;
    report.pieces[0].pieceIndex = 0;
    report.pieces[0].beginArcBaseVoxels = 0.0;
    report.pieces[0].endArcBaseVoxels = 6.0;
    report.pieces[1].pieceIndex = 1;
    report.pieces[1].beginArcBaseVoxels = 6.0;
    report.pieces[1].endArcBaseVoxels = 12.0;
    report.pieces[2].traceIndex = 1;
    report.pieces[3].traceIndex = 1;
    report.pieces[4].traceIndex = 2;
    FiberTraceWindingOrderedOffsetReport ordering;
    ordering.offsetByPiece = {0.0, 1.0, 0.0, 1.0, 0.0};
    ordering.orientationByPiece.assign(
        5, FiberTraceFixedOrientation::Horizontal);
    ordering.activeByPiece.assign(5, 1);
    const auto factor = [](std::size_t a, std::size_t b, double sign) {
        FiberTraceWindingOrderedSignFactor result;
        result.pieceA = a;
        result.pieceB = b;
        result.sign = sign;
        return result;
    };
    ordering.signFactors = {factor(0, 1, -1.0)};
    for (std::size_t index = 0; index < 9; ++index)
        ordering.signFactors.push_back(factor(2, 3, -1.0));
    ordering.signFactors.push_back(factor(2, 3, 1.0));
    const auto unequal = summarizeFiberTraceWindingOrderedViolations(
        ordering, report);
    CHECK(unequal.traces[0].violatedFactors == 1);
    CHECK(unequal.traces[0].incidentFactors == 1);
    CHECK(unequal.traces[1].violatedFactors == 9);
    CHECK(unequal.traces[1].incidentFactors == 10);
    REQUIRE(unequal.worstTrace);
    CHECK(*unequal.worstTrace == 0);

    ordering.signFactors = {
        factor(0, 1, 1.0),
        factor(0, 1, -1.0),
        factor(2, 3, 1.0),
        factor(2, 3, 1.0),
        factor(2, 3, -1.0),
        factor(2, 3, -1.0),
    };
    const auto tied = summarizeFiberTraceWindingOrderedViolations(
        ordering, report);
    CHECK(tied.factors == 6);
    CHECK(tied.infringements == 3);
    CHECK(tied.traces[0].incidentFactors == 2);
    CHECK(tied.traces[0].violatedFactors == 1);
    CHECK(tied.traces[1].incidentFactors == 4);
    CHECK(tied.traces[1].violatedFactors == 2);
    CHECK(tied.traces[2].incidentFactors == 0);
    REQUIRE(tied.worstTrace);
    CHECK(*tied.worstTrace == 1);

    ordering.signFactors = {factor(1, 4, 1.0)};
    const auto cross = summarizeFiberTraceWindingOrderedViolations(
        ordering, report);
    CHECK(cross.factors == 1);
    CHECK(cross.infringements == 1);
    CHECK(cross.traces[0].incidentFactors == 1);
    CHECK(cross.traces[2].incidentFactors == 1);
    REQUIRE(cross.worstTrace);
    CHECK(*cross.worstTrace == 0);

    const std::vector<unsigned char> onlyTraceTwo{0, 0, 1};
    const auto filtered = summarizeFiberTraceWindingOrderedViolations(
        ordering, report, onlyTraceTwo);
    CHECK(filtered.factors == 0);
    CHECK(filtered.infringements == 0);
    CHECK_FALSE(filtered.worstTrace);
}

TEST_CASE("Ordered offender removal excludes a whole trace and re-solves survivors")
{
    auto source = lines(3);
    source[0].pointsBaseXYZ = {
        {-6.0, 0.0, 0.0},
        {6.0, 0.0, 0.0},
    };
    auto report = pieces(4);
    report.inputTraces = 3;
    report.pieces[0].traceIndex = 0;
    report.pieces[1].traceIndex = 0;
    report.pieces[0].pieceIndex = 0;
    report.pieces[0].beginArcBaseVoxels = 0.0;
    report.pieces[0].endArcBaseVoxels = 6.0;
    report.pieces[1].pieceIndex = 1;
    report.pieces[1].beginArcBaseVoxels = 6.0;
    report.pieces[1].endArcBaseVoxels = 12.0;
    report.pieces[2].traceIndex = 1;
    report.pieces[3].traceIndex = 2;
    FiberTraceConstraint continuation;
    continuation.pieceA = 0;
    continuation.pieceB = 1;
    continuation.arcABaseVoxels = 6.0;
    continuation.arcBBaseVoxels = 6.0;
    continuation.pointABaseXYZ = {0.0, 0.0, 0.0};
    continuation.pointBBaseXYZ = continuation.pointABaseXYZ;
    continuation.parallelScore = 1.0;
    continuation.hardContinuity = true;
    report.constraints.push_back(continuation);
    addMeasured(report, 0, 2, 0.0, 0.5);
    addMeasured(report, 2, 1, 0.0, 0.5);
    addMeasured(report, 1, 3, 0.0, 0.5);
    addMeasured(report, 3, 0, 0.0, 0.5);
    const std::vector orientations(
        4, FiberTraceFixedOrientation::Horizontal);
    FiberTraceWindingOrderedCutsConfig settings;
    settings.decisionConfidence =
        FiberTraceWindingDecisionConfidence::Legacy;
    settings.normalConfidence = FiberTraceWindingNormalConfidence::None;
    settings.parallelWorkers = 1;
    settings.measurementScale = 1.0;
    settings.maximumIterations = 100;
    settings.signMarginWeight = 0.0;

    const auto baseline = solveFiberTraceWindingOrderedCuts(
        report, topology(source, report), orientations, settings);
    CHECK(baseline.removals.empty());
    CHECK(std::count(
              baseline.ordering.activeByPiece.begin(),
              baseline.ordering.activeByPiece.end(),
              static_cast<unsigned char>(1)) == 4);

    settings.removeOffendingFibers = true;
    const auto pruned = solveFiberTraceWindingOrderedCuts(
        report, topology(source, report), orientations, settings);
    REQUIRE(pruned.removals.size() == 1);
    const auto& removal = pruned.removals.front();
    CHECK(removal.removedTrace == 0);
    CHECK(removal.removedPieces == 2);
    CHECK(removal.incidentFactors == 4);
    CHECK(removal.violatedFactors == 4);
    CHECK(removal.oldInfringements == 4);
    CHECK(removal.oldFactors == 4);
    CHECK(removal.survivingBeforeInfringements == 0);
    CHECK(removal.survivingFactors == 0);
    CHECK(removal.survivingAfterInfringements == 0);
    CHECK(removal.remainingTraces == 2);
    CHECK(pruned.ordering.activeByPiece ==
          std::vector<unsigned char>{0, 0, 1, 1});
    CHECK(pruned.ordering.orientationByPiece == orientations);
    const auto final = summarizeFiberTraceWindingOrderedViolations(
        pruned.ordering, report);
    CHECK(final.infringements == 0);
}

TEST_CASE("Ordered offender removal re-solves surviving sign factors")
{
    const auto source = lines(3);
    auto report = pieces(3);
    addMeasured(report, 0, 1, 0.0, 0.5);
    addMeasured(report, 1, 2, 0.0, 0.5);
    addMeasured(report, 2, 0, 0.0, 0.5);
    const std::vector orientations(
        3, FiberTraceFixedOrientation::Horizontal);
    FiberTraceWindingOrderedCutsConfig settings;
    settings.decisionConfidence =
        FiberTraceWindingDecisionConfidence::Legacy;
    settings.normalConfidence = FiberTraceWindingNormalConfidence::None;
    settings.parallelWorkers = 1;
    settings.measurementScale = 1.0;
    settings.maximumIterations = 100;
    settings.signMarginWeight = 0.0;
    settings.removeOffendingFibers = true;
    const auto pruned = solveFiberTraceWindingOrderedCuts(
        report, topology(source, report), orientations, settings);
    REQUIRE(pruned.removals.size() == 1);
    const auto& removal = pruned.removals.front();
    CHECK(removal.removedTrace == 0);
    CHECK(removal.oldInfringements == 3);
    CHECK(removal.survivingFactors == 1);
    CHECK(removal.survivingBeforeInfringements == 1);
    CHECK(removal.survivingAfterInfringements == 0);
}

TEST_CASE("Winding confidence transforms affect only dominant winding weight")
{
    const auto source = lines(2);
    const std::vector fixed(
        source.size(), FiberTraceFixedOrientation::Horizontal);
    auto report = pieces(2);
    addMeasured(report, 0, 1, 0.625, 1.0);
    report.constraints.front().parallelNormalAlignment = 0.5;

    const auto solve = [&](FiberTraceWindingDecisionConfidence decision,
                           FiberTraceWindingNormalConfidence normal) {
        FiberTraceJointGridWindingConfig joint;
        static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) =
            config();
        joint.decisionConfidence = decision;
        joint.normalConfidence = normal;
        joint.enforcePerpendicularWindingSign = false;
        joint.fixedPhaseMagnitude = 0.5;
        joint.fixedMeasurementScale = 1.0;
        joint.stableIterations = 1;
        return solveFiberTraceJointGridWindingBeliefPropagation(
            report, topology(source, report), joint, {}, fixed);
    };

    const auto legacy = solve(
        FiberTraceWindingDecisionConfidence::Legacy,
        FiberTraceWindingNormalConfidence::None);
    const auto linear = solve(
        FiberTraceWindingDecisionConfidence::Linear,
        FiberTraceWindingNormalConfidence::None);
    const auto cosine = solve(
        FiberTraceWindingDecisionConfidence::Cosine,
        FiberTraceWindingNormalConfidence::None);
    const auto normalLinear = solve(
        FiberTraceWindingDecisionConfidence::Legacy,
        FiberTraceWindingNormalConfidence::Linear);
    const auto normalCosine = solve(
        FiberTraceWindingDecisionConfidence::Legacy,
        FiberTraceWindingNormalConfidence::Cosine);
    REQUIRE(legacy.factorDiagnostics.size() == 1);
    CHECK(legacy.factorDiagnostics[0].decisionConfidenceMultiplier ==
          doctest::Approx(0.625));
    CHECK(linear.factorDiagnostics[0].decisionConfidenceMultiplier ==
          doctest::Approx(0.25));
    CHECK(cosine.factorDiagnostics[0].decisionConfidenceMultiplier ==
          doctest::Approx(0.5 - 0.5 * std::cos(0.25 * std::numbers::pi)));
    CHECK(normalLinear.factorDiagnostics[0].normalConfidenceMultiplier ==
          doctest::Approx(1.0 / 3.0));
    CHECK(normalCosine.factorDiagnostics[0].normalConfidenceMultiplier ==
          doctest::Approx(0.5));
    CHECK(legacy.factorDiagnostics[0].parallelScore ==
          linear.factorDiagnostics[0].parallelScore);
    CHECK(legacy.factorDiagnostics[0].perpendicularScore ==
          linear.factorDiagnostics[0].perpendicularScore);

    report.constraints.front().parallelNormalAlignment.reset();
    const auto missing = solve(
        FiberTraceWindingDecisionConfidence::Legacy,
        FiberTraceWindingNormalConfidence::Cosine);
    CHECK(missing.factorDiagnostics[0].normalConfidenceMultiplier == 0.0);
    CHECK(missing.factorDiagnostics[0].effectiveParallelWindingWeight == 0.0);
    CHECK(missing.integerGaugeByPiece[0] != missing.integerGaugeByPiece[1]);
    CHECK_THROWS_AS(
        solve(
            static_cast<FiberTraceWindingDecisionConfidence>(255),
            FiberTraceWindingNormalConfidence::None),
        std::invalid_argument);
    CHECK_THROWS_AS(
        solve(
            FiberTraceWindingDecisionConfidence::Legacy,
            static_cast<FiberTraceWindingNormalConfidence>(255)),
        std::invalid_argument);
}

TEST_CASE("Finite winding sign costs replace hard rejection and preserve zero-confidence removal")
{
    const auto source = lines(2);
    const std::vector fixed(
        source.size(), FiberTraceFixedOrientation::Horizontal);
    auto report = pieces(2);
    addMeasured(report, 0, 1, 1.0, 1.0);
    report.constraints.front().parallelNormalAlignment = 1.0;

    const auto solve = [&](std::optional<double> signCost,
                           FiberTraceWindingDecisionConfidence decision =
                               FiberTraceWindingDecisionConfidence::Legacy) {
        FiberTraceJointGridWindingConfig joint;
        static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) =
            config();
        useZeroClassWeights(joint);
        joint.enforcePerpendicularWindingSign = false;
        joint.enforceParallelWindingSign = true;
        joint.finiteSignInfringementCost = signCost;
        joint.hardSignMinimumNormalAlignment.reset();
        joint.decisionConfidence = decision;
        joint.fixedPhaseMagnitude = 0.5;
        joint.fixedMeasurementScale = 1.0;
        joint.mixedUnaryCost = 100.0;
        joint.stableIterations = 1;
        return solveFiberTraceJointGridWindingBeliefPropagation(
            report, topology(source, report), joint, {}, fixed);
    };

    const auto hard = solve(std::nullopt);
    CHECK(hard.factorDiagnostics[0].hardParallelSign);
    CHECK(hard.factorDiagnostics[0].effectiveParallelSignPenalty == 0.0);

    const auto finite = solve(16.0);
    CHECK_FALSE(finite.factorDiagnostics[0].hardParallelSign);
    CHECK(finite.factorDiagnostics[0].effectiveParallelSignPenalty ==
          doctest::Approx(16.0));
    CHECK(finite.integerGaugeByPiece[0] == finite.integerGaugeByPiece[1]);
    CHECK(finite.mapLatentCoordinate[1] > finite.mapLatentCoordinate[0]);

    const auto zero = solve(0.0);
    CHECK_FALSE(zero.factorDiagnostics[0].hardParallelSign);
    CHECK(zero.factorDiagnostics[0].effectiveParallelSignPenalty == 0.0);
    CHECK(zero.integerGaugeByPiece[0] != zero.integerGaugeByPiece[1]);

    report.constraints.front().parallelScore = 0.5;
    report.constraints.front().perpendicularScore = 0.5;
    report.constraints.front().signedWindingDelta = 0.5;
    report.constraints.front().perpendicularNormalAlignment = 1.0;
    FiberTraceJointGridWindingConfig tiedConfig;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(tiedConfig) =
        config();
    useZeroClassWeights(tiedConfig);
    tiedConfig.enforcePerpendicularWindingSign = true;
    tiedConfig.enforceParallelWindingSign = false;
    tiedConfig.decisionConfidence =
        FiberTraceWindingDecisionConfidence::Linear;
    tiedConfig.fixedPhaseMagnitude = 0.5;
    tiedConfig.fixedMeasurementScale = 1.0;
    tiedConfig.stableIterations = 1;
    const auto tiedHard = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), tiedConfig, {}, fixed);
    CHECK(tiedHard.factorDiagnostics[0].decisionConfidenceMultiplier == 0.0);
    CHECK(tiedHard.factorDiagnostics[0].hardPerpendicularSign);
}

TEST_CASE("Signed winding value and extra sign hardness are independently weighted")
{
    const auto source = lines(2);
    const std::vector fixed(
        source.size(), FiberTraceFixedOrientation::Horizontal);
    auto report = pieces(2);
    addMeasured(report, 0, 1, 1.0, 1.0);

    const auto solve = [&](double valueWeight, double signWeight) {
        FiberTraceJointGridWindingConfig joint;
        static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) =
            config();
        useZeroClassWeights(joint);
        joint.parallelOneWeight = valueWeight;
        joint.enforcePerpendicularWindingSign = false;
        joint.enforceParallelWindingSign = true;
        joint.parallelSignWeight = signWeight;
        joint.finiteSignInfringementCost = 8.0;
        joint.hardSignMinimumNormalAlignment.reset();
        joint.fixedPhaseMagnitude = 0.5;
        joint.fixedMeasurementScale = 1.0;
        joint.mixedUnaryCost = 100.0;
        joint.stableIterations = 1;
        return solveFiberTraceJointGridWindingBeliefPropagation(
            report, topology(source, report), joint, {}, fixed);
    };

    const auto valueOnly = solve(1.0, 0.0);
    REQUIRE(valueOnly.factorDiagnostics.size() == 1);
    CHECK(valueOnly.factorDiagnostics[0].parallelMagnitudePresent);
    CHECK(valueOnly.factorDiagnostics[0].parallelSignPresent);
    CHECK(valueOnly.factorDiagnostics[0].effectiveParallelWindingWeight > 0.0);
    CHECK(valueOnly.factorDiagnostics[0].effectiveParallelSignPenalty == 0.0);
    CHECK_FALSE(valueOnly.factorDiagnostics[0].hardParallelSign);
    CHECK(valueOnly.mapLatentCoordinate[1] >
          valueOnly.mapLatentCoordinate[0]);

    const auto weightedSign = solve(1.0, 3.0);
    CHECK(weightedSign.factorDiagnostics[0]
              .effectiveParallelSignPenalty == doctest::Approx(24.0));
    CHECK(weightedSign.factorDiagnostics[0]
              .parallelSignWeightMultiplier == doctest::Approx(3.0));

    report.constraints.front().signedParallelWindingDelta = -1.0;
    const auto reversedValueOnly = solve(1.0, 0.0);
    CHECK(reversedValueOnly.mapLatentCoordinate[1] <
          reversedValueOnly.mapLatentCoordinate[0]);

    auto invalid = config();
    invalid.perpendicularSignWeight = -1.0;
    CHECK_THROWS_AS(
        solveFiberTraceWindingBeliefPropagation(
            report, topology(source, report), invalid),
        std::invalid_argument);
    invalid = config();
    invalid.parallelSignWeight =
        std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(
        solveFiberTraceWindingBeliefPropagation(
            report, topology(source, report), invalid),
        std::invalid_argument);
}

TEST_CASE("Reference benchmark lists signed winding value and sign hardness separately")
{
    FiberTraceInterleavedWindingReport winding;
    winding.windingValid = {1};
    winding.mapLatentCoordinate = {0.0};
    winding.mapOrientationByPiece = {
        FiberTraceFixedOrientation::Horizontal};
    winding.integerGaugeByPiece = {0};
    winding.componentByPiece = {0};
    winding.measurementScale = 1.0;

    FiberTraceConstraint constraint;
    constraint.pieceA = 0;
    constraint.pieceB = 1;
    constraint.parallelScore = 1.0;
    constraint.signedParallelWindingDelta = 1.0;
    constraint.parallelWindingDistance = 1.0;
    auto settings = config();
    settings.enforceParallelWindingSign = true;
    settings.parallelSignWeight = 0.0;
    const auto observation = makeFiberTraceReferenceWindingObservation(
        constraint, false, 1.0, 0, winding, settings);
    CHECK(observation.parallelMagnitudePresent);
    CHECK(observation.parallelSignPresent);
    CHECK(observation.parallelSignPenalty == 0.0);
    CHECK_FALSE(observation.hardParallelSign);

    const std::array observations{observation};
    const auto benchmark = calibrateFiberTraceReferenceWindings(observations);
    const auto& windingCounts = benchmark.classes[static_cast<std::size_t>(
        FiberTraceReferenceBenchmarkClass::ParallelOtherMagnitude)];
    const auto& signCounts = benchmark.classes[static_cast<std::size_t>(
        FiberTraceReferenceBenchmarkClass::ParallelSign)];
    CHECK(windingCounts.total == 1);
    CHECK(signCounts.total == 1);
    CHECK(benchmark.sum.total == 2);
}

TEST_CASE("Raw normal alignment promotes enabled dominant signs to hard")
{
    const auto source = lines(2);
    const std::vector fixed(
        source.size(), FiberTraceFixedOrientation::Horizontal);
    auto report = pieces(2);
    addMeasured(report, 0, 1, 1.0, 1.0);
    auto& constraint = report.constraints.front();
    FiberTraceJointGridWindingConfig config;
    useZeroClassWeights(config);
    config.perpendicularSignWeight = 1.0;
    config.parallelSignWeight = 1.0;
    config.enforcePerpendicularWindingSign = false;
    config.enforceParallelWindingSign = true;
    config.finiteSignInfringementCost = 0.0;
    config.hardSignMinimumNormalAlignment =
        std::cos(std::numbers::pi / 6.0);
    config.decisionConfidence =
        FiberTraceWindingDecisionConfidence::Linear;
    config.fixedPhaseMagnitude = 0.5;
    config.fixedMeasurementScale = 1.0;
    config.stableIterations = 1;

    const auto diagnose = [&] {
        return diagnoseFiberTraceWindingFactors(
            report, topology(source, report), config, fixed, true).front();
    };
    constraint.parallelNormalAlignment =
        *config.hardSignMinimumNormalAlignment;
    auto equal = diagnose();
    CHECK(equal.decisionConfidenceMultiplier == 1.0);
    CHECK(equal.hardParallelSign);
    CHECK(equal.parallelSignPromotedByAlignment);

    constraint.parallelNormalAlignment =
        *config.hardSignMinimumNormalAlignment - 1.0e-6;
    auto below = diagnose();
    CHECK_FALSE(below.hardParallelSign);
    CHECK_FALSE(below.parallelSignPromotedByAlignment);

    constraint.parallelNormalAlignment.reset();
    auto missing = diagnose();
    CHECK_FALSE(missing.hardParallelSign);

    config.hardSignMinimumNormalAlignment.reset();
    constraint.parallelNormalAlignment = 1.0;
    auto disabled = diagnose();
    CHECK_FALSE(disabled.hardParallelSign);

    config.finiteSignInfringementCost.reset();
    constraint.parallelNormalAlignment.reset();
    auto global = diagnose();
    CHECK(global.hardParallelSign);
    CHECK_FALSE(global.parallelSignPromotedByAlignment);

    config.finiteSignInfringementCost = 0.0;
    config.hardSignMinimumNormalAlignment =
        std::cos(std::numbers::pi / 6.0);
    config.enforceParallelWindingSign = false;
    config.enforcePerpendicularWindingSign = true;
    constraint.parallelScore = 0.5;
    constraint.perpendicularScore = 0.5;
    constraint.signedWindingDelta = 0.5;
    constraint.perpendicularNormalAlignment =
        *config.hardSignMinimumNormalAlignment;
    const auto zeroConfidence = diagnose();
    CHECK(zeroConfidence.decisionConfidenceMultiplier == 0.0);
    CHECK(zeroConfidence.hardPerpendicularSign);
    CHECK(zeroConfidence.perpendicularSignPromotedByAlignment);
}

FiberTraceBeliefPropagationReport orientationBeliefs(
    std::initializer_list<std::array<double, 3>> probabilities)
{
    FiberTraceBeliefPropagationReport result;
    for (const auto& probability : probabilities) {
        result.horizontalProbability.push_back(probability[0]);
        result.mixedProbability.push_back(probability[1]);
        result.verticalProbability.push_back(probability[2]);
    }
    return result;
}

TEST_CASE("Fixed winding orientations use MAP and convert ties to Mixed")
{
    const auto fixed = fixedFiberTraceOrientations(orientationBeliefs({
        {0.8, 0.1, 0.1},
        {0.1, 0.8, 0.1},
        {0.1, 0.2, 0.7},
        {0.5, 0.0, 0.5},
        {0.3, 0.3, 0.3},
    }));
    CHECK(fixed == std::vector<FiberTraceFixedOrientation>{
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Mixed,
        FiberTraceFixedOrientation::Vertical,
        FiberTraceFixedOrientation::Mixed,
        FiberTraceFixedOrientation::Mixed,
    });
    CHECK_THROWS_AS(
        fixedFiberTraceOrientations(orientationBeliefs({})),
        std::invalid_argument);
}

TEST_CASE("Final winding state summary separates a stable selected cohort")
{
    const std::vector orientations{
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Vertical,
        FiberTraceFixedOrientation::Mixed,
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Vertical,
    };
    const std::vector<unsigned char> valid{1, 1, 1, 0, 0};
    const std::vector<unsigned char> selected{1, 0, 1, 0, 1};
    const auto summary = summarizeFiberTraceFinalStates(
        orientations, valid, selected);

    CHECK(summary.selected.pieces == 3);
    CHECK(summary.selected.horizontal == 1);
    CHECK(summary.selected.vertical == 0);
    CHECK(summary.selected.active() == 1);
    CHECK(summary.selected.defect == 2);
    CHECK(summary.other.pieces == 2);
    CHECK(summary.other.horizontal == 0);
    CHECK(summary.other.vertical == 1);
    CHECK(summary.other.active() == 1);
    CHECK(summary.other.defect == 1);
    CHECK(summary.total.pieces == 5);
    CHECK(summary.total.horizontal == 1);
    CHECK(summary.total.vertical == 1);
    CHECK(summary.total.active() == 2);
    CHECK(summary.total.defect == 3);
    CHECK(summary.selected.pieces + summary.other.pieces ==
          summary.total.pieces);
    CHECK(summary.total.active() + summary.total.defect ==
          summary.total.pieces);

    const std::vector<unsigned char> none(orientations.size(), 0);
    const auto emptySelected = summarizeFiberTraceFinalStates(
        orientations, valid, none);
    CHECK(emptySelected.selected.pieces == 0);
    CHECK(emptySelected.other.pieces == orientations.size());

    CHECK_THROWS_AS(
        summarizeFiberTraceFinalStates(
            orientations,
            std::span<const unsigned char>(valid).first(4),
            selected),
        std::invalid_argument);
}

TEST_CASE("Constraint evidence summary separates cohorts classes and final states")
{
    auto report = pieces(3);
    FiberTraceConstraint continuity;
    continuity.pieceA = 0;
    continuity.pieceB = 1;
    continuity.parallelScore = 1.0;
    continuity.perpendicularScore = 0.0;
    continuity.hardContinuity = true;
    report.constraints.push_back(continuity);
    addMeasured(report, 0, 2, 0.25, 0.5);
    addMeasured(report, 1, 2, 0.4, 1.5);
    addMeasured(report, 0, 1, 1.0, 0.0);

    const auto diagnostic = [](
        std::size_t constraint,
        std::size_t a,
        std::size_t b) {
        FiberTraceWindingFactorDiagnostic result;
        result.constraintIndex = constraint;
        result.pieceA = a;
        result.pieceB = b;
        return result;
    };
    std::vector<FiberTraceWindingFactorDiagnostic> diagnostics;
    auto hard = diagnostic(0, 0, 1);
    hard.parallelScore = 1.0;
    hard.parallelWindingRetained = true;
    hard.parallelMagnitudePresent = true;
    hard.effectiveParallelWindingWeight = 1.0;
    diagnostics.push_back(hard);

    auto both = diagnostic(1, 0, 2);
    both.parallelScore = 0.25;
    both.perpendicularScore = 0.75;
    both.parallelWindingRetained = true;
    both.parallelMagnitudePresent = true;
    both.perpendicularMagnitudePresent = true;
    both.perpendicularSignPresent = true;
    both.effectiveParallelWindingWeight = 0.25;
    both.effectivePerpendicularWindingWeight = 0.75;
    both.effectivePerpendicularSignedDelta = 0.5;
    both.hardPerpendicularSign = true;
    diagnostics.push_back(both);

    auto other = diagnostic(2, 1, 2);
    other.parallelScore = 0.4;
    other.perpendicularScore = 0.6;
    other.parallelWindingRetained = true;
    other.parallelMagnitudePresent = true;
    other.perpendicularMagnitudePresent = true;
    other.perpendicularSignPresent = true;
    other.effectiveParallelWindingWeight = 0.4;
    other.effectivePerpendicularWindingWeight = 0.2;
    other.effectiveParallelWindingDistance = 2.0;
    other.effectivePerpendicularSignedDelta = 1.5;
    other.hardPerpendicularSign = true;
    diagnostics.push_back(other);

    auto suppressed = diagnostic(3, 0, 1);
    suppressed.parallelScore = 1.0;
    suppressed.parallelWindingRetained = false;
    diagnostics.push_back(suppressed);

    const std::vector orientations{
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Vertical,
        FiberTraceFixedOrientation::Mixed,
    };
    const std::vector<unsigned char> valid{1, 1, 1};
    const std::vector<unsigned char> selected{1, 0, 1};
    const auto summary = summarizeFiberTraceConstraintEvidence(
        report, diagnostics, orientations, valid, selected);
    using Class = FiberTraceConstraintEvidenceClass;
    const auto& selectedContinuity =
        summary.selected.classes[static_cast<std::size_t>(Class::Continuity)];
    CHECK(selectedContinuity.incidences == 1);
    CHECK(selectedContinuity.activeIncidences == 1);
    CHECK(selectedContinuity.effectiveWeight == doctest::Approx(1.0));
    const auto& otherContinuity =
        summary.other.classes[static_cast<std::size_t>(Class::Continuity)];
    CHECK(otherContinuity.incidences == 1);
    CHECK(otherContinuity.activeIncidences == 1);

    const auto& selectedPerpendicularMagnitude = summary.selected.classes[
        static_cast<std::size_t>(Class::PerpendicularMagnitude)];
    CHECK(selectedPerpendicularMagnitude.incidences == 3);
    CHECK(selectedPerpendicularMagnitude.activeIncidences == 1);
    CHECK(selectedPerpendicularMagnitude.defectIncidences == 2);
    CHECK(selectedPerpendicularMagnitude.effectiveWeight ==
          doctest::Approx(1.7));
    CHECK(selectedPerpendicularMagnitude.activeEffectiveWeight ==
          doctest::Approx(0.75));
    CHECK(selectedPerpendicularMagnitude.defectEffectiveWeight ==
          doctest::Approx(0.95));
    const auto& selectedPerpendicularSign = summary.selected.classes[
        static_cast<std::size_t>(Class::PerpendicularSign)];
    CHECK(selectedPerpendicularSign.incidences == 0);
    CHECK(selectedPerpendicularSign.hardSignIncidences == 3);
    CHECK(selectedPerpendicularSign.activeHardSignIncidences == 1);
    CHECK(selectedPerpendicularSign.defectHardSignIncidences == 2);
    const auto& otherPerpendicularMagnitude = summary.other.classes[
        static_cast<std::size_t>(Class::PerpendicularMagnitude)];
    CHECK(otherPerpendicularMagnitude.incidences == 1);
    CHECK(otherPerpendicularMagnitude.effectiveWeight == doctest::Approx(0.2));
    const auto& otherPerpendicularSign = summary.other.classes[
        static_cast<std::size_t>(Class::PerpendicularSign)];
    CHECK(otherPerpendicularSign.hardSignIncidences == 1);

    const auto& selectedSame = summary.selected.classes[
        static_cast<std::size_t>(Class::ParallelSameMagnitude)];
    CHECK(selectedSame.incidences == 2);
    CHECK(selectedSame.effectiveWeight == doctest::Approx(0.5));
    CHECK(selectedSame.activeEffectiveWeight == doctest::Approx(0.25));
    CHECK(selectedSame.defectEffectiveWeight == doctest::Approx(0.25));
    const auto& selectedOther = summary.selected.classes[
        static_cast<std::size_t>(Class::ParallelOtherMagnitude)];
    CHECK(selectedOther.incidences == 1);
    CHECK(selectedOther.defectIncidences == 1);
    CHECK(selectedOther.effectiveWeight == doctest::Approx(0.4));
    const auto& otherOther = summary.other.classes[
        static_cast<std::size_t>(Class::ParallelOtherMagnitude)];
    CHECK(otherOther.incidences == 1);
    CHECK(otherOther.activeIncidences == 1);
    CHECK(otherOther.effectiveWeight == doctest::Approx(0.4));

    CHECK(summary.selected.states.pieces == 2);
    CHECK(summary.selected.states.active() == 1);
    CHECK(summary.selected.states.defect == 1);
    CHECK(summary.other.states.pieces == 1);
    CHECK(summary.other.states.active() == 1);
    CHECK(summary.selected.total.incidences == 4);
    CHECK(summary.other.total.incidences == 2);
    CHECK(summary.total.total.incidences ==
          summary.selected.total.incidences + summary.other.total.incidences);
    CHECK(summary.total.total.effectiveWeight == doctest::Approx(
          summary.selected.total.effectiveWeight +
          summary.other.total.effectiveWeight));

    auto mismatched = diagnostics;
    mismatched.front().pieceB = 2;
    CHECK_THROWS_AS(
        summarizeFiberTraceConstraintEvidence(
            report, mismatched, orientations, valid, selected),
        std::invalid_argument);
}

TEST_CASE("Constraint agreement separates infringed and Defect-neutralized factors")
{
    auto report = pieces(4);
    FiberTraceConstraint continuity;
    continuity.pieceA = 0;
    continuity.pieceB = 1;
    continuity.parallelScore = 1.0;
    continuity.hardContinuity = true;
    report.constraints.push_back(continuity);
    addMeasured(report, 0, 2, 0.0, 0.5);
    addMeasured(report, 0, 3, 1.0, 0.0);
    FiberTraceConstraint invalid = continuity;
    invalid.pieceA = 1;
    invalid.pieceB = 3;
    report.constraints.push_back(invalid);

    FiberTraceInterleavedWindingReport winding;
    winding.measurementScale = 1.0;
    winding.windingValid = {1, 1, 1, 0};
    winding.mapWinding = {0, 0, 0, 0};
    winding.mapLatentCoordinate = {
        0.0,
        0.0,
        0.5,
        std::numeric_limits<double>::quiet_NaN(),
    };
    winding.mapOrientationByPiece = {
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Mixed,
    };
    const auto diagnostic = [](std::size_t constraint,
                               std::size_t a,
                               std::size_t b) {
        FiberTraceWindingFactorDiagnostic value;
        value.constraintIndex = constraint;
        value.pieceA = a;
        value.pieceB = b;
        value.parallelScore = 1.0;
        return value;
    };
    winding.factorDiagnostics.push_back(diagnostic(0, 0, 1));
    auto perpendicular = diagnostic(1, 0, 2);
    perpendicular.parallelScore = 0.0;
    perpendicular.perpendicularScore = 1.0;
    perpendicular.perpendicularMagnitudePresent = true;
    perpendicular.perpendicularSignPresent = true;
    perpendicular.effectivePerpendicularSignedDelta = 0.5;
    perpendicular.effectivePerpendicularWindingWeight = 1.0;
    winding.factorDiagnostics.push_back(perpendicular);
    auto parallel = diagnostic(2, 0, 3);
    parallel.parallelWindingRetained = true;
    parallel.parallelMagnitudePresent = true;
    parallel.effectiveParallelWindingDistance = 0.0;
    parallel.effectiveParallelWindingWeight = 1.0;
    winding.factorDiagnostics.push_back(parallel);
    winding.factorDiagnostics.push_back(diagnostic(3, 1, 3));

    const auto summary = summarizeFiberTraceConstraintAgreement(
        report, winding);
    const auto& continuityCounts = summary.classes[static_cast<std::size_t>(
        FiberTraceConstraintAgreementClass::Continuity)];
    CHECK(continuityCounts.prepared == 2);
    CHECK(continuityCounts.evaluated == 1);
    CHECK(continuityCounts.defectNeutralized == 1);
    CHECK(continuityCounts.infringed == 0);
    const auto& perpendicularOrientation = summary.classes[
        static_cast<std::size_t>(
            FiberTraceConstraintAgreementClass::PerpendicularOrientation)];
    CHECK(perpendicularOrientation.evaluated == 1);
    CHECK(perpendicularOrientation.infringed == 1);
    const auto& perpendicularMagnitude = summary.classes[
        static_cast<std::size_t>(FiberTraceConstraintAgreementClass::
                                     PerpendicularMagnitudeNext)];
    CHECK(perpendicularMagnitude.evaluated == 1);
    CHECK(perpendicularMagnitude.infringed == 0);
    const auto& perpendicularSign = summary.classes[
        static_cast<std::size_t>(
            FiberTraceConstraintAgreementClass::PerpendicularSign)];
    CHECK(perpendicularSign.evaluated == 1);
    CHECK(perpendicularSign.infringed == 0);
    const auto& parallelOrientation = summary.classes[
        static_cast<std::size_t>(
            FiberTraceConstraintAgreementClass::ParallelOrientation)];
    CHECK(parallelOrientation.defectNeutralized == 1);
    const auto& parallelCounts = summary.classes[static_cast<std::size_t>(
        FiberTraceConstraintAgreementClass::ParallelMagnitudeSame)];
    CHECK(parallelCounts.defectNeutralized == 1);
    CHECK(summary.total.prepared == 7);
    CHECK(summary.total.evaluated == 4);
    CHECK(summary.total.defectNeutralized == 3);
    CHECK(summary.total.infringed == 1);
}

TEST_CASE("Largest winding component uses effective factors and deterministic ties")
{
    const auto source = lines(5);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 0.5);
    addMeasured(report, 1, 2, 0.0, 0.5);
    addMeasured(report, 3, 4, 0.0, 0.5);
    const auto prepared = topology(source, report);

    const auto largest = selectLargestFiberTraceWindingComponent(report, prepared, config());
    CHECK(largest.components == 2);
    CHECK(largest.retainedPieceIndices == std::vector<std::size_t>{0, 1, 2});
    CHECK(largest.retainedPieces == 3);
    CHECK(largest.removedPieces == 2);

    auto tieReport = pieces(4);
    addMeasured(tieReport, 0, 1, 0.0, 0.5);
    addMeasured(tieReport, 2, 3, 0.0, 0.5);
    const auto tieTopology = topology(lines(4), tieReport);
    CHECK(selectLargestFiberTraceWindingComponent(tieReport, tieTopology, config()).retainedPieceIndices == std::vector<std::size_t>{0, 1});
    CHECK(selectLargestFiberTraceWindingComponent(tieReport, tieTopology, config(), {}, true, 3).retainedPieceIndices == std::vector<std::size_t>{2, 3});

    const std::vector<FiberTraceFixedOrientation> fixed{
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Mixed,
        FiberTraceFixedOrientation::Vertical,
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Vertical,
    };
    const auto withoutMixed = selectLargestFiberTraceWindingComponent(report, prepared, config(), fixed);
    CHECK(withoutMixed.components == 4);
    CHECK(withoutMixed.retainedPieceIndices == std::vector<std::size_t>{3, 4});
}

TEST_CASE("Largest winding component honors cutoff and signed evidence")
{
    const auto source = lines(4);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 1.0, 2.0);
    report.constraints.back().signedWindingDelta.reset();
    addMeasured(report, 2, 3, 0.0, 0.5);
    auto cutoff = config();
    cutoff.parallelWindingDistanceCutoff = 0.5;
    const auto selected = selectLargestFiberTraceWindingComponent(report, topology(source, report), cutoff, {}, true, 3);
    CHECK(selected.components == 3);
    CHECK(selected.retainedPieceIndices == std::vector<std::size_t>{2, 3});
}

TEST_CASE("Two-stage winding BP recovers signed chains and canonical reversal")
{
    const auto source = lines(3);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 1.0);
    addMeasured(report, 2, 1, 0.0, -1.0);

    const auto solved = solveFiberTraceWindingBeliefPropagation(
        report, topology(source, report), config());
    CHECK(solved.status == "converged");
    CHECK(solved.mapWinding == std::vector<int>{0, 1, 2});
    CHECK(solved.continuousWinding[0] == doctest::Approx(0.0));
    CHECK(solved.continuousWinding[1] == doctest::Approx(1.0));
    CHECK(solved.continuousWinding[2] == doctest::Approx(2.0));
    CHECK(solved.expansionRounds > 1);
    REQUIRE(solved.factorDiagnostics.size() == 2);
    CHECK(solved.factorDiagnostics[1].originalSignedDelta == -1.0);
    CHECK(solved.factorDiagnostics[1].canonicalSignedDelta == 1.0);

    std::reverse(report.constraints.begin(), report.constraints.end());
    const auto reordered = solveFiberTraceWindingBeliefPropagation(
        report, topology(source, report), config());
    CHECK(reordered.mapWinding == solved.mapWinding);
    CHECK(reordered.posteriorMeanWinding == solved.posteriorMeanWinding);
}

TEST_CASE("Interleaved winding uses signed half-integer observation bins")
{
    const auto source = lines(12);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 0.0);
    addMeasured(report, 2, 3, 0.0, std::nextafter(1.0, 0.0));
    addMeasured(report, 4, 5, 0.0, 1.0);
    addMeasured(report, 6, 7, 0.0,
        std::nextafter(1.0, std::numeric_limits<double>::infinity()));
    addMeasured(report, 8, 9, 0.0, 1.49);
    addMeasured(report, 11, 10, 0.0, -1.01);

    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    joint.mixedUnaryCost = 5.0;
    joint.stableIterations = 1;
    const auto solved = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), joint);

    const std::vector<double> expected{
        0.0, 0.5, 0.5, 1.5, 1.5, 1.5,
    };
    REQUIRE(solved.factorDiagnostics.size() == expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        REQUIRE(solved.factorDiagnostics[index]
            .effectivePerpendicularSignedDelta);
        CHECK(*solved.factorDiagnostics[index]
                   .effectivePerpendicularSignedDelta ==
            expected[index]);
    }
    CHECK(solved.factorDiagnostics[5].originalSignedDelta == -1.01);
    CHECK(solved.factorDiagnostics[5].canonicalSignedDelta == 1.01);
    CHECK(solved.continuousWinding[1] == 0.0);
    CHECK(solved.continuousWinding[3] == doctest::Approx(0.5));
    CHECK(solved.continuousWinding[5] == doctest::Approx(0.5));
    CHECK(solved.continuousWinding[7] == doctest::Approx(1.5));
    CHECK(solved.continuousWinding[9] == doctest::Approx(1.5));
    CHECK(solved.continuousWinding[11] == doctest::Approx(1.5));

    const auto raw = solveFiberTraceWindingBeliefPropagation(
        report, topology(source, report), config());
    REQUIRE(raw.factorDiagnostics.size() == expected.size());
    for (const auto& diagnostic : raw.factorDiagnostics) {
        CHECK_FALSE(diagnostic.effectiveSignedParallelDelta.has_value());
        CHECK(diagnostic.effectivePerpendicularSignedDelta ==
            diagnostic.canonicalSignedDelta);
    }
}

TEST_CASE("Half-integer winding quantization preserves boundary behavior")
{
    CHECK(quantizedHalfWindingTarget(0.0) == 0.0);
    CHECK(quantizedHalfWindingTarget(std::numeric_limits<double>::denorm_min()) ==
        0.5);
    CHECK(quantizedHalfWindingTarget(1.0) == 0.5);
    CHECK(quantizedHalfWindingTarget(
              std::nextafter(1.0, std::numeric_limits<double>::infinity())) ==
        1.5);
    CHECK(quantizedHalfWindingTarget(2.0) == 1.5);
    CHECK(quantizedHalfWindingTarget(-2.0) == -1.5);
}

TEST_CASE("Integer winding quantization uses nearest integer boundaries")
{
    CHECK(quantizedIntegerWindingTarget(0.0) == 0.0);
    CHECK(quantizedIntegerWindingTarget(
              std::nextafter(0.5, 0.0)) == 0.0);
    CHECK(quantizedIntegerWindingTarget(0.5) == 1.0);
    CHECK(quantizedIntegerWindingTarget(
              std::nextafter(1.5, 0.0)) == 1.0);
    CHECK(quantizedIntegerWindingTarget(1.5) == 2.0);
    CHECK(quantizedIntegerWindingTarget(-1.5) == -2.0);
}

TEST_CASE("Canonical constraint counts include every emitted row")
{
    FiberTraceCanonicalConstraintCounts counts;
    CHECK(counts.correct == 0);
    CHECK(counts.falseCount == 0);
    CHECK(counts.total == 0);

    counts.add(quantizedHalfWindingTarget(0.51), 0.5);
    counts.add(quantizedHalfWindingTarget(1.51), 0.5);
    counts.add(quantizedIntegerWindingTarget(0.49), 0.0);
    counts.add(quantizedIntegerWindingTarget(0.51), 0.0);
    counts.add(quantizedIntegerWindingTarget(0.49), 0.0);

    CHECK(counts.correct == 3);
    CHECK(counts.falseCount == 2);
    CHECK(counts.total == 5);
    CHECK(counts.correct + counts.falseCount == counts.total);
}

TEST_CASE("Reference constraint diagnostics use benchmark sign before quantization")
{
    using Class = FiberTraceReferenceConstraintClass;
    const std::vector<FiberTraceReferenceWindingObservation> observations{
        {Class::Perpendicular, 0, 0.0, {4.0, 0.0}, 1, 0},
        {Class::Perpendicular, 0, 0.5, {3.5, 0.0}, 1, 1},
        {Class::Perpendicular, 0, 1.0, {3.0, 0.0}, 1, 2},
    };
    const auto calibration = calibrateFiberTraceReferenceWindings(
        observations, 0.1);
    REQUIRE(calibration.globalSign == -1);

    auto report = pieces(3);
    FiberTraceConstraint perpendicular;
    perpendicular.pieceA = 1;
    perpendicular.pieceB = 0;
    perpendicular.parallelScore = 0.1;
    perpendicular.perpendicularScore = 0.9;
    perpendicular.windingDistance = 0.51;
    perpendicular.signedWindingDelta = 0.51;
    report.constraints.push_back(perpendicular);

    FiberTraceConstraint parallel;
    parallel.pieceA = 0;
    parallel.pieceB = 2;
    parallel.parallelScore = 0.9;
    parallel.perpendicularScore = 0.1;
    parallel.parallelWindingDistance = 1.1;
    parallel.signedParallelWindingDelta = -1.1;
    report.constraints.push_back(parallel);

    FiberTraceConstraint unsignedPerpendicular;
    unsignedPerpendicular.pieceA = 0;
    unsignedPerpendicular.pieceB = 1;
    unsignedPerpendicular.parallelScore = 0.1;
    unsignedPerpendicular.perpendicularScore = 0.9;
    unsignedPerpendicular.windingDistance = 0.51;
    report.constraints.push_back(unsignedPerpendicular);

    const auto diagnostics =
        makeFiberTraceReferenceConstraintDiagnosticReport(
            report, {0, 1, 2}, calibration);
    REQUIRE(diagnostics.rows.size() == 3);
    CHECK(diagnostics.rows[0].signedMeasurement);
    CHECK(diagnostics.rows[0].rawStep == doctest::Approx(-0.51));
    CHECK(diagnostics.rows[0].calibratedStep == doctest::Approx(0.51));
    CHECK(diagnostics.rows[0].canonicalStep == doctest::Approx(0.5));
    CHECK(diagnostics.rows[0].groundTruthStep == doctest::Approx(0.5));
    CHECK(diagnostics.rows[1].signedMeasurement);
    CHECK(diagnostics.rows[1].rawStep == doctest::Approx(-1.1));
    CHECK(diagnostics.rows[1].calibratedStep == doctest::Approx(1.1));
    CHECK(diagnostics.rows[1].canonicalStep == doctest::Approx(1.0));
    CHECK(diagnostics.rows[1].groundTruthStep == doctest::Approx(1.0));
    CHECK_FALSE(diagnostics.rows[2].signedMeasurement);
    CHECK(diagnostics.rows[2].rawStep == doctest::Approx(0.51));
    CHECK(diagnostics.rows[2].calibratedStep == doctest::Approx(0.51));
    CHECK(diagnostics.rows[2].canonicalStep == doctest::Approx(0.5));
    CHECK(diagnostics.counts.correct == 3);
    CHECK(diagnostics.counts.falseCount == 0);
    CHECK(diagnostics.counts.total == 3);
    std::vector<FiberTraceWindingFactorDiagnostic> calibratedFactors(3);
    for (std::size_t index = 0; index < calibratedFactors.size(); ++index) {
        calibratedFactors[index].constraintIndex = index;
        if (diagnostics.rows[index].perpendicularDominant) {
            calibratedFactors[index].effectivePerpendicularWindingWeight = 1.0;
        } else {
            calibratedFactors[index].effectiveParallelWindingWeight = 1.0;
        }
    }
    const auto calibratedScale = calibrateFiberTraceReferenceConstraintScales(
        diagnostics, calibratedFactors);
    REQUIRE(calibratedScale.canonicalPerpendicular.fittedScale.has_value());
    CHECK(*calibratedScale.canonicalPerpendicular.fittedScale ==
          doctest::Approx(1.0));
    REQUIRE(calibratedScale.rawPerpendicular.fittedScale.has_value());
    CHECK(*calibratedScale.rawPerpendicular.fittedScale ==
          doctest::Approx(0.5 / 0.51));

    auto invalidCalibration = calibration;
    invalidCalibration.globalSign = 0;
    CHECK_THROWS_AS(
        makeFiberTraceReferenceConstraintDiagnosticReport(
            report, {0, 1, 2}, invalidCalibration),
        std::invalid_argument);

    report.constraints[0].signedWindingDelta =
        std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(
        makeFiberTraceReferenceConstraintDiagnosticReport(
            report, {0, 1, 2}, calibration),
        std::invalid_argument);
}

TEST_CASE("Reference constraint scale calibration minimizes weighted solver L1")
{
    const auto empty = calibrateFiberTraceReferenceConstraintScales(
        FiberTraceReferenceConstraintDiagnosticReport{}, {});
    CHECK(empty.rawPerpendicular.observations == 0);
    CHECK_FALSE(empty.rawPerpendicular.fittedScale.has_value());
    CHECK_FALSE(empty.canonicalPerpendicular.fittedScale.has_value());
    CHECK_FALSE(empty.rawParallel.fittedScale.has_value());
    CHECK_FALSE(empty.canonicalParallel.fittedScale.has_value());
    CHECK_FALSE(empty.rawAll.fittedScale.has_value());
    CHECK_FALSE(empty.canonicalAll.fittedScale.has_value());
    for (const auto& group : empty.rawGroups) {
        CHECK(group.observations == 0);
        CHECK_FALSE(group.fittedScale.has_value());
    }
    for (const auto& group : empty.canonicalGroups) {
        CHECK(group.observations == 0);
        CHECK_FALSE(group.fittedScale.has_value());
    }

    FiberTraceReferenceConstraintDiagnosticReport reference;
    reference.rows = {
        {0, 0, 2, true, true, 1.6, 1.6, 1.5, 1.0},
        {1, 0, 4, true, true, 2.4, 2.4, 2.5, 2.0},
        {2, 0, 2, false, true, 0.1, 0.1, 0.0, 1.0},
        {3, 0, 2, false, true, 1.1, 1.1, 1.0, 1.0},
        {4, 0, 6, false, true, 2.6, 2.6, 3.0, 3.0},
        {5, 0, 2, true, true, 0.6, 0.6, 0.5, 1.0},
    };
    std::vector<FiberTraceWindingFactorDiagnostic> factors(6);
    for (std::size_t index = 0; index < factors.size(); ++index)
        factors[index].constraintIndex = index;
    factors[0].effectivePerpendicularWindingWeight = 3.0;
    factors[1].effectivePerpendicularWindingWeight = 0.5;
    factors[2].effectiveParallelWindingWeight = 1.0;
    factors[3].effectiveParallelWindingWeight = 2.0;
    factors[4].effectiveParallelWindingWeight = 1.0;
    factors[5].effectivePerpendicularWindingWeight = 0.0;

    const auto fitted = calibrateFiberTraceReferenceConstraintScales(
        reference, factors);
    REQUIRE(fitted.rawPerpendicular.fittedScale.has_value());
    CHECK(*fitted.rawPerpendicular.fittedScale == doctest::Approx(0.625));
    REQUIRE(fitted.canonicalPerpendicular.fittedScale.has_value());
    CHECK(*fitted.canonicalPerpendicular.fittedScale ==
          doctest::Approx(2.0 / 3.0));
    CHECK(fitted.canonicalPerpendicular.observations == 3);
    CHECK(fitted.canonicalPerpendicular.admittedObservations == 2);
    CHECK(fitted.canonicalPerpendicular.informativeObservations == 2);
    CHECK(fitted.canonicalPerpendicular.effectiveWeight ==
          doctest::Approx(3.5));
    CHECK(fitted.canonicalPerpendicular.reciprocalScaleWeight ==
          doctest::Approx(4.0));
    CHECK(fitted.canonicalPerpendicular.fittedLoss <
          fitted.canonicalPerpendicular.unitScaleLoss);
    CHECK(fitted.rawParallel.observations == 3);
    CHECK(fitted.canonicalParallel.observations == 3);
    CHECK(fitted.rawAll.observations == 6);
    CHECK(fitted.canonicalAll.observations == 6);
    CHECK(fitted.rawAll.admittedObservations == 5);
    CHECK(fitted.canonicalAll.admittedObservations == 5);
    REQUIRE(fitted.rawParallel.fittedScale.has_value());
    REQUIRE(fitted.canonicalParallel.fittedScale.has_value());
    REQUIRE(fitted.rawAll.fittedScale.has_value());
    REQUIRE(fitted.canonicalAll.fittedScale.has_value());
    CHECK(*fitted.rawParallel.fittedScale ==
          doctest::Approx(3.0 / 2.6));
    CHECK(*fitted.canonicalParallel.fittedScale == doctest::Approx(1.0));
    CHECK(*fitted.rawAll.fittedScale == doctest::Approx(1.0 / 1.1));
    CHECK(*fitted.canonicalAll.fittedScale == doctest::Approx(1.0));

    using Group = FiberTraceReferenceConstraintGroup;
    const auto& next = fitted.canonicalGroups[static_cast<std::size_t>(
        Group::PerpendicularNext)];
    CHECK(next.observations == 1);
    CHECK(next.admittedObservations == 0);
    CHECK_FALSE(next.fittedScale.has_value());
    const auto& far = fitted.canonicalGroups[static_cast<std::size_t>(
        Group::PerpendicularFar)];
    REQUIRE(far.fittedScale.has_value());
    CHECK(*far.fittedScale == doctest::Approx(2.0 / 3.0));
    const auto& parallelSame = fitted.canonicalGroups[static_cast<std::size_t>(
        Group::ParallelSame)];
    REQUIRE(parallelSame.fittedScale.has_value());
    CHECK(*parallelSame.fittedScale == doctest::Approx(2.0));
    CHECK(parallelSame.atUpperBound);
    const auto& parallelOne = fitted.canonicalGroups[static_cast<std::size_t>(
        Group::ParallelOne)];
    REQUIRE(parallelOne.fittedScale.has_value());
    CHECK(*parallelOne.fittedScale == doctest::Approx(1.0));
    const auto& parallelFar = fitted.canonicalGroups[static_cast<std::size_t>(
        Group::ParallelTwoPlus)];
    REQUIRE(parallelFar.fittedScale.has_value());
    CHECK(*parallelFar.fittedScale == doctest::Approx(1.0));

    reference.rows = {
        {0, 0, 2, true, true, 10.0, 10.0, 10.0, 1.0},
    };
    factors.resize(1);
    factors[0].constraintIndex = 0;
    factors[0].effectivePerpendicularWindingWeight = 1.0;
    const auto lower = calibrateFiberTraceReferenceConstraintScales(
        reference, factors);
    REQUIRE(lower.canonicalPerpendicular.fittedScale.has_value());
    CHECK(*lower.canonicalPerpendicular.fittedScale == doctest::Approx(0.5));
    CHECK(lower.canonicalPerpendicular.atLowerBound);

    reference.rows = {
        {0, 0, 2, true, true, 0.5, 0.5, 0.5, 1.0},
        {1, 0, 2, true, true, 1.5, 1.5, 1.5, 1.0},
    };
    factors.resize(2);
    factors[0].constraintIndex = 0;
    factors[0].effectivePerpendicularWindingWeight = 1.0;
    factors[1].constraintIndex = 1;
    factors[1].effectivePerpendicularWindingWeight = 1.0;
    const auto tied = calibrateFiberTraceReferenceConstraintScales(
        reference, factors);
    REQUIRE(tied.canonicalPerpendicular.fittedScale.has_value());
    CHECK(*tied.canonicalPerpendicular.fittedScale == doctest::Approx(1.0));

    reference.rows.resize(1);
    reference.rows[0].canonicalStep = 1.0;
    reference.rows[0].groundTruthStep = 0.0;
    factors.resize(1);
    const auto unidentifiable = calibrateFiberTraceReferenceConstraintScales(
        reference, factors);
    CHECK(unidentifiable.canonicalPerpendicular.admittedObservations == 1);
    CHECK(unidentifiable.canonicalPerpendicular.informativeObservations == 0);
    CHECK_FALSE(unidentifiable.canonicalPerpendicular.fittedScale.has_value());
    CHECK(unidentifiable.canonicalPerpendicular.unitScaleLoss ==
          doctest::Approx(1.0));

    CHECK_THROWS_AS(
        calibrateFiberTraceReferenceConstraintScales(
            reference, factors, 0.0, 2.0),
        std::invalid_argument);
    factors.clear();
    CHECK_THROWS_AS(
        calibrateFiberTraceReferenceConstraintScales(reference, factors),
        std::invalid_argument);
}

TEST_CASE("Reference phase calibration fits the alternating physical ladder")
{
    FiberTraceReferenceConstraintDiagnosticReport reference;
    reference.rows = {
        {0, 0, 1, true, true, 0.2, -9.0, 0.5, 0.5},
        {1, 1, 2, true, true, 0.8, -9.0, 0.5, 0.5},
        {2, 0, 3, true, true, 1.2, -9.0, 1.5, 1.5},
        {3, 0, 2, true, true, 1.0, -9.0, 1.0, 1.0},
        {4, 0, 2, false, true, 1.0, -9.0, 1.0, 1.0},
        {5, 0, 1, false, true, 0.2, -9.0, 0.0, 0.5},
    };
    const auto calibrated = calibrateFiberTraceReferenceConstraintPhase(
        reference, 1.0);
    REQUIRE(calibrated.selectedGauge.has_value());
    const auto& selected = calibrated.gauges[*calibrated.selectedGauge];
    CHECK(selected.windingDirection == 1);
    CHECK(selected.evenReferenceIsHorizontal);
    REQUIRE(selected.fittedPhase.has_value());
    CHECK(*selected.fittedPhase == doctest::Approx(0.2));
    CHECK(selected.fittedLoss == doctest::Approx(0.0));
    CHECK(selected.lossAtZero == doctest::Approx(0.6));
    CHECK(selected.lossAtHalf == doctest::Approx(0.9));
    CHECK(selected.totalRows == 6);
    CHECK(selected.identifyingRows == 3);
    CHECK(selected.usedRows == 3);
    CHECK(selected.perpendicularSameParityRows == 1);
    CHECK(selected.parallelSameParityRows == 1);
    CHECK(selected.parallelOppositeParityRows == 1);
    CHECK(selected.effectiveWeight == doctest::Approx(3.0));
    CHECK(selected.fittedSignDisagreements == 0);

    const auto& reversed = calibrated.gauges[2];
    REQUIRE(reversed.fittedPhase.has_value());
    CHECK(reversed.windingDirection == -1);
    CHECK(reversed.fittedLoss > selected.fittedLoss);
}

TEST_CASE("Reference phase calibration keeps phase zero and scale explicit")
{
    FiberTraceReferenceConstraintDiagnosticReport reference;
    reference.rows = {
        {0, 0, 1, true, true, 0.0, 7.0, 0.5, 0.5},
        {1, 1, 2, true, true, 1.0, 7.0, 0.5, 0.5},
    };
    const auto zero = calibrateFiberTraceReferenceConstraintPhase(
        reference, 1.0);
    REQUIRE(zero.selectedGauge.has_value());
    const auto& selected = zero.gauges[*zero.selectedGauge];
    REQUIRE(selected.fittedPhase.has_value());
    CHECK(*selected.fittedPhase == 0.0);
    CHECK(selected.fittedLoss == 0.0);
    CHECK(selected.fittedSignDisagreements == 1);

    reference.rows = {
        {0, 0, 1, true, true, 0.1, -3.0, 0.5, 0.5},
    };
    const auto scaleOne = calibrateFiberTraceReferenceConstraintPhase(
        reference, 1.0);
    const auto scaleTwo = calibrateFiberTraceReferenceConstraintPhase(
        reference, 2.0);
    REQUIRE(scaleOne.selectedGauge.has_value());
    REQUIRE(scaleTwo.selectedGauge.has_value());
    CHECK(*scaleOne.gauges[*scaleOne.selectedGauge].fittedPhase ==
          doctest::Approx(0.1));
    CHECK(*scaleTwo.gauges[*scaleTwo.selectedGauge].fittedPhase ==
          doctest::Approx(0.2));
    CHECK_THROWS_AS(
        calibrateFiberTraceReferenceConstraintPhase(reference, 0.0),
        std::invalid_argument);
}

TEST_CASE("Reference phase calibration reports unidentifiable evidence")
{
    FiberTraceReferenceConstraintDiagnosticReport reference;
    reference.rows = {
        {0, 0, 1, false, true, 0.5, 0.5, 0.0, 0.5},
        {1, 0, 2, true, true, 1.0, 1.0, 1.0, 1.0},
        {2, 0, 3, true, false, 1.5, 1.5, 1.5, 1.5},
    };
    const auto unidentifiable = calibrateFiberTraceReferenceConstraintPhase(
        reference, 1.0);
    CHECK_FALSE(unidentifiable.selectedGauge.has_value());
    for (const auto& gauge : unidentifiable.gauges) {
        CHECK(gauge.identifyingRows == 1);
        CHECK(gauge.usedRows == 0);
        CHECK_FALSE(gauge.fittedPhase.has_value());
    }
}

TEST_CASE("Reference step statistics retain direction and distance bands")
{
    FiberTraceReferenceConstraintDiagnosticReport reference;
    reference.rows = {
        {0, 0, 1, true, true, 0.1, 0.1, 0.5, 0.5},
        {1, 0, 1, true, true, 0.3, 0.3, 0.5, 0.5},
        {2, 1, 2, true, true, 0.9, 0.9, 0.5, 0.5},
        {3, 0, 3, true, true, 1.2, 1.2, 1.5, 1.5},
        {4, 1, 6, true, true, 2.8, 2.8, 2.5, 2.5},
        {5, 0, 2, false, true, 0.8, 0.8, 1.0, 1.0},
        {6, 1, 5, false, true, 2.2, 2.2, 2.0, 2.0},
        {7, 0, 6, false, true, 3.1, 3.1, 3.0, 3.0},
        {8, 0, 1, false, false, 99.0, 99.0, 0.5, 0.5},
    };
    const auto stats = summarizeFiberTraceReferenceConstraintSteps(reference);
    const auto& hToVHalf = stats.groups[0][0][1][0];
    CHECK(hToVHalf.observations == 2);
    CHECK(hToVHalf.minimum == doctest::Approx(0.1));
    CHECK(hToVHalf.mean == doctest::Approx(0.2));
    CHECK(hToVHalf.median == doctest::Approx(0.2));
    CHECK(hToVHalf.maximum == doctest::Approx(0.3));
    CHECK(stats.groups[0][1][0][0].observations == 1);
    CHECK(stats.groups[0][0][1][1].observations == 1);
    CHECK(stats.groups[0][1][0][2].observations == 1);
    CHECK(stats.groups[1][0][0][0].observations == 1);
    CHECK(stats.groups[1][1][1][1].observations == 1);
    CHECK(stats.groups[1][0][0][2].observations == 1);
    CHECK(stats.groups[1][0][1][0].observations == 0);
}

TEST_CASE("Public winding factor diagnostics match solver preparation")
{
    const auto source = lines(2);
    auto report = pieces(2);
    addMeasured(report, 0, 1, 0.2, 0.6);
    report.constraints.front().perpendicularNormalAlignment = 0.75;
    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.decisionConfidence =
        FiberTraceWindingDecisionConfidence::Linear;
    joint.normalConfidence = FiberTraceWindingNormalConfidence::Cosine;
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    joint.stableIterations = 1;
    const auto preparedTopology = topology(source, report);
    const auto diagnostic = diagnoseFiberTraceWindingFactors(
        report, preparedTopology, joint);
    const auto solved = solveFiberTraceJointGridWindingBeliefPropagation(
        report, preparedTopology, joint);
    REQUIRE(diagnostic.size() == 1);
    REQUIRE(solved.factorDiagnostics.size() == 1);
    CHECK(diagnostic[0].constraintIndex ==
          solved.factorDiagnostics[0].constraintIndex);
    CHECK(diagnostic[0].effectiveParallelWindingWeight ==
          solved.factorDiagnostics[0].effectiveParallelWindingWeight);
    CHECK(diagnostic[0].effectivePerpendicularWindingWeight ==
          solved.factorDiagnostics[0].effectivePerpendicularWindingWeight);
    CHECK(diagnostic[0].decisionConfidenceMultiplier ==
          solved.factorDiagnostics[0].decisionConfidenceMultiplier);
    CHECK(diagnostic[0].normalConfidenceMultiplier ==
          solved.factorDiagnostics[0].normalConfidenceMultiplier);
    CHECK(diagnostic[0].effectivePerpendicularSignedDelta ==
          solved.factorDiagnostics[0].effectivePerpendicularSignedDelta);
}

TEST_CASE("Measurement scale precedes winding quantization and class filtering")
{
    const auto source = lines(4);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.2, 0.6);
    addMeasured(report, 2, 3, 0.8, 0.4);
    FiberTraceWindingBeliefPropagationConfig winding = config();
    winding.enforcePerpendicularWindingSign = false;
    winding.enforceParallelWindingSign = false;
    winding.parallelWindingDistanceCutoff = 0.5;
    const auto preparedTopology = topology(source, report);

    const auto unit = diagnoseFiberTraceWindingFactors(
        report, preparedTopology, winding, {}, true, 1.0);
    const auto doubled = diagnoseFiberTraceWindingFactors(
        report, preparedTopology, winding, {}, true, 2.0);
    REQUIRE(unit.size() == 2);
    REQUIRE(doubled.size() == 2);

    CHECK(unit[0].effectivePerpendicularSignedDelta == 0.5);
    CHECK(doubled[0].effectivePerpendicularSignedDelta == 1.5);
    CHECK(unit[0].perpendicularWindingWeightMultiplier == 1.0);
    CHECK(doubled[0].perpendicularWindingWeightMultiplier == 0.5);

    CHECK(unit[1].effectiveSignedParallelDelta == 0.0);
    CHECK(unit[1].parallelWindingRetained);
    CHECK(doubled[1].effectiveParallelWindingDistance == 1.0);
    CHECK_FALSE(doubled[1].effectiveSignedParallelDelta.has_value());
    CHECK_FALSE(doubled[1].parallelWindingRetained);
}

TEST_CASE("Reference winding benchmark calibrates each integer gauge")
{
    using Class = FiberTraceReferenceConstraintClass;
    std::vector<FiberTraceReferenceWindingObservation> observations{
        {Class::Perpendicular, 0, 0.0, {2.0, 0.0}, 1, 0},
        {Class::ParallelSameWinding, 0, 0.0, {2.0, 0.0}, 1, 0},
        {Class::ParallelOtherWinding, 0, 0.5, {1.5, 0.0}, 1, 1},
        {Class::Perpendicular, 0, 1.0, {1.0, 0.0}, 1, 2},
        {Class::Perpendicular, 1, 0.0, {-3.0, 0.0}, 1, 0},
        {Class::ParallelOtherWinding, 1, 0.5, {-3.5, 0.0}, 1, 1},
        {Class::Perpendicular, 2, 1.5, {7.0, 0.0}, 1, 3,
         0.5, 1.0, 0.0},
    };
    const auto benchmark = calibrateFiberTraceReferenceWindings(observations);
    CHECK(benchmark.globalSign == -1);
    REQUIRE(benchmark.gauges.size() == 2);
    CHECK(benchmark.gauges[0].integerGauge == 0);
    CHECK(benchmark.gauges[0].offset == doctest::Approx(2.0));
    CHECK(benchmark.gauges[0].exactMatches == 3);
    CHECK(benchmark.gauges[0].estimateVotes == 3);
    CHECK(benchmark.gauges[1].integerGauge == 1);
    CHECK(benchmark.gauges[1].offset == doctest::Approx(-3.0));
    CHECK(benchmark.gauges[1].exactMatches == 2);
    CHECK(benchmark.gauges[1].estimateVotes == 2);
    CHECK(benchmark.sum.right == 6);
    CHECK(benchmark.sum.wrong == 0);
    CHECK(benchmark.sum.total == 6);
    REQUIRE(benchmark.references.size() == 4);
    CHECK(benchmark.references[0].classes[0].right == 2);
    CHECK(benchmark.references[0].classes[2].right == 1);
    CHECK(benchmark.references[0].sum.right == 3);
    CHECK(benchmark.references[0].sum.wrong == 0);
    CHECK(benchmark.references[0].sum.total == 3);
    CHECK(benchmark.references[1].classes[3].right == 2);
    CHECK(benchmark.references[1].sum.right == 2);
    CHECK(benchmark.references[1].sum.wrong == 0);
    CHECK(benchmark.references[1].sum.total == 2);
    REQUIRE(benchmark.references[0].estimatedWinding.has_value());
    CHECK_FALSE(benchmark.references[0].rawEstimatedWinding.has_value());
    CHECK(*benchmark.references[0].estimatedWinding == doctest::Approx(0.0));
    REQUIRE(benchmark.references[1].estimatedWinding.has_value());
    CHECK_FALSE(benchmark.references[1].rawEstimatedWinding.has_value());
    CHECK(*benchmark.references[1].estimatedWinding == doctest::Approx(0.5));
    CHECK(benchmark.references[1].estimatedWindingSupport == 2);
    CHECK(benchmark.references[1].estimatedWindingObservations == 2);
    REQUIRE(benchmark.references[2].rawEstimatedWinding.has_value());
    REQUIRE(benchmark.references[2].estimatedWinding.has_value());
    CHECK(*benchmark.references[2].estimatedWinding == doctest::Approx(
        static_cast<double>(benchmark.globalSign) *
        (*benchmark.references[2].rawEstimatedWinding -
         benchmark.gauges[0].offset)));
    CHECK(*benchmark.references[2].estimatedWinding == 1.0);
    CHECK_FALSE(benchmark.references[3].estimatedWinding.has_value());
    CHECK_FALSE(benchmark.references[3].rawEstimatedWinding.has_value());
    CHECK(std::accumulate(
              benchmark.references.begin(),
              benchmark.references.end(),
              std::size_t{0},
              [](std::size_t total, const auto& counts) {
                  return total + counts.sum.right;
              }) == benchmark.sum.right);
    CHECK(std::accumulate(
              benchmark.references.begin(),
              benchmark.references.end(),
              std::size_t{0},
              [](std::size_t total, const auto& counts) {
                  return total + counts.sum.wrong;
              }) == benchmark.sum.wrong);
    CHECK(std::accumulate(
              benchmark.references.begin(),
              benchmark.references.end(),
              std::size_t{0},
              [](std::size_t total, const auto& counts) {
                  return total + counts.sum.total;
              }) == benchmark.sum.total);
    for (std::size_t classIndex = 0; classIndex < benchmark.classes.size(); ++classIndex) {
        CHECK(std::accumulate(
                  benchmark.references.begin(),
                  benchmark.references.end(),
                  std::size_t{0},
                  [classIndex](std::size_t total, const auto& counts) {
                      return total + counts.classes[classIndex].total;
                  }) == benchmark.classes[classIndex].total);
    }
}

TEST_CASE("Reference winding calibration includes boundaries and stable ties")
{
    using Class = FiberTraceReferenceConstraintClass;
    const std::vector<FiberTraceReferenceWindingObservation> boundary{
        {Class::Perpendicular, 4, 0.0, {0.0, 0.0}, 1, 0},
        {Class::Perpendicular, 4, 0.5, {1.0, 0.0}, 1, 1},
    };
    const auto atBoundary = calibrateFiberTraceReferenceWindings(boundary);
    const auto strictBoundary =
        calibrateFiberTraceReferenceWindings(boundary, 0.0);
    REQUIRE(atBoundary.gauges.size() == 1);
    CHECK(atBoundary.gauges[0].offset == 0.0);
    CHECK(atBoundary.gauges[0].exactMatches == 1);
    CHECK(atBoundary.sum.right == 2);
    REQUIRE(strictBoundary.gauges.size() == 1);
    CHECK(strictBoundary.gauges[0].offset == atBoundary.gauges[0].offset);
    CHECK(strictBoundary.gauges[0].exactMatches ==
          atBoundary.gauges[0].exactMatches);
    CHECK(strictBoundary.sum.right == 1);

    const std::vector<FiberTraceReferenceWindingObservation> tied{
        {Class::ParallelOtherWinding, 7, 0.0, {-2.0, 0.0}, 1, 0},
    };
    const auto stable = calibrateFiberTraceReferenceWindings(tied, 0.0);
    REQUIRE(stable.gauges.size() == 1);
    CHECK(stable.gauges[0].offset == -2.0);
    CHECK(stable.sum.right == 1);

    const std::vector<FiberTraceReferenceWindingObservation> invalid{
        {Class::Perpendicular, 9, 0.0, {0.0, 0.0}, 0},
    };
    const auto allInvalid = calibrateFiberTraceReferenceWindings(invalid);
    CHECK(allInvalid.gauges.empty());
    CHECK(allInvalid.sum.right == 0);
    CHECK(allInvalid.sum.wrong == 0);
    CHECK(allInvalid.sum.total == 0);
    REQUIRE(allInvalid.references.size() == 1);
    CHECK_FALSE(allInvalid.references[0].rawEstimatedWinding.has_value());
    CHECK(calibrateFiberTraceReferenceWindings(std::span<const FiberTraceReferenceWindingObservation>{}).gauges.empty());
}

TEST_CASE("Raw reference winding inference ignores truth and report tolerance")
{
    using Class = FiberTraceReferenceConstraintClass;
    std::vector<FiberTraceReferenceWindingObservation> observations{
        {Class::Perpendicular, 3, 0.0, {2.0, 0.0}, 1, 0},
        {Class::ParallelSameWinding, 3, 0.0, {2.5, 0.0}, 1, 0},
        {Class::Perpendicular, 3, 0.5, {3.0, 0.0}, 1, 1},
    };
    const auto raw = inferFiberTraceReferenceRawWindings(observations);
    observations[0].virtualReferenceWinding = 20.0;
    observations[1].virtualReferenceWinding = 20.0;
    observations[2].virtualReferenceWinding = -11.0;
    const auto changedTruth =
        inferFiberTraceReferenceRawWindings(observations);
    REQUIRE(raw.size() == changedTruth.size());
    for (std::size_t index = 0; index < raw.size(); ++index) {
        CHECK(raw[index].referenceSource ==
              changedTruth[index].referenceSource);
        CHECK(raw[index].integerGauge == changedTruth[index].integerGauge);
        CHECK(raw[index].winding == changedTruth[index].winding);
        CHECK(raw[index].observations == changedTruth[index].observations);
    }
}

TEST_CASE("Reported raw winding inverse maps the selected calibrated estimate")
{
    using Class = FiberTraceReferenceConstraintClass;
    const std::vector<FiberTraceReferenceWindingObservation> observations{
        {Class::Perpendicular, 0, 0.0, {4.0, 0.0}, 1, 0},
        {Class::Perpendicular, 0, 0.5, {3.5, 0.0}, 1, 1},
        {Class::Perpendicular, 0, 1.0, {2.5, 3.0}, 2, 2},
    };

    const auto independentlyRaw =
        inferFiberTraceReferenceRawWindings(observations);
    const auto rawSourceTwo = std::find_if(
        independentlyRaw.begin(), independentlyRaw.end(),
        [](const auto& estimate) { return estimate.referenceSource == 2; });
    REQUIRE(rawSourceTwo != independentlyRaw.end());
    CHECK(rawSourceTwo->winding == 2.5);

    const auto benchmark = calibrateFiberTraceReferenceWindings(observations);
    REQUIRE(benchmark.globalSign == -1);
    REQUIRE(benchmark.gauges.size() == 1);
    CHECK(benchmark.gauges[0].offset == 4.0);
    REQUIRE(benchmark.references[2].estimatedWinding.has_value());
    REQUIRE(benchmark.references[2].rawEstimatedWinding.has_value());
    CHECK(*benchmark.references[2].estimatedWinding == 1.0);
    CHECK(*benchmark.references[2].rawEstimatedWinding == 3.0);
    CHECK(*benchmark.references[2].rawEstimatedWinding !=
          rawSourceTwo->winding);
}

TEST_CASE("Reference raw winding maps to the published integer layer")
{
    FiberTraceReferenceSourceBenchmark reference;
    reference.estimatedIntegerGauge = 3;
    reference.estimatedOrientationComponent = 0;
    FiberTraceReferenceOrientationBenchmark orientation;
    orientation.components.push_back({0, true, 1, 0});

    reference.rawEstimatedWinding = -2.0;
    CHECK(fiberTraceReferenceOutputWinding(
              0, reference, orientation, std::array{1}, 0.5, 5) == 3);

    reference.rawEstimatedWinding = -1.5;
    CHECK(fiberTraceReferenceOutputWinding(
              1, reference, orientation, std::array{1}, 0.5, 5) == 3);

    reference.rawEstimatedWinding = -2.5;
    CHECK(fiberTraceReferenceOutputWinding(
              1, reference, orientation, std::array{-1}, 0.5, 5) == 3);

    orientation.components[0].evenReferenceIsHorizontal = false;
    reference.rawEstimatedWinding = -1.5;
    CHECK(fiberTraceReferenceOutputWinding(
              0, reference, orientation, std::array{1}, 0.5, 5) == 3);
    reference.rawEstimatedWinding = -2.0;
    CHECK(fiberTraceReferenceOutputWinding(
              1, reference, orientation, std::array{1}, 0.5, 5) == 3);

    orientation.components[0].evenReferenceIsHorizontal = true;
    reference.rawEstimatedWinding = -1.5;
    CHECK_FALSE(fiberTraceReferenceOutputWinding(
        0, reference, orientation, std::array{1}, 0.5, 5));

    reference.rawEstimatedWinding = -1.75;
    CHECK_THROWS_AS(
        fiberTraceReferenceOutputWinding(
            0, reference, orientation, std::array{1}, 0.5, 5),
        std::logic_error);

    reference.rawEstimatedWinding.reset();
    CHECK_FALSE(fiberTraceReferenceOutputWinding(
        0, reference, orientation, std::array{1}, 0.5, 5));

    reference.rawEstimatedWinding = -2.0;
    reference.estimatedOrientationComponent.reset();
    CHECK_FALSE(fiberTraceReferenceOutputWinding(
        0, reference, orientation, std::array{1}, 0.5, 5));
}

TEST_CASE("Calibrated reference inference combines independent gauges")
{
    using Class = FiberTraceReferenceConstraintClass;
    const std::vector<FiberTraceReferenceWindingObservation> observations{
        {Class::Perpendicular, 0, 0.0, {2.0, 0.0}, 1, 0},
        {Class::Perpendicular, 0, 0.5, {2.5, 0.0}, 1, 1},
        {Class::Perpendicular, 1, 0.0, {9.5, 0.0}, 1, 0},
        {Class::Perpendicular, 1, 0.5, {10.5, 0.0}, 1, 1},
        {Class::Perpendicular, 1, 1.0, {11.0, 0.0}, 1, 2},
    };

    const auto raw = inferFiberTraceReferenceRawWindings(observations);
    REQUIRE(raw.size() == 5);
    const auto benchmark = calibrateFiberTraceReferenceWindings(observations);
    CHECK(benchmark.globalSign == 1);
    REQUIRE(benchmark.gauges.size() == 2);
    CHECK(benchmark.gauges[0].offset == 2.0);
    CHECK(benchmark.gauges[1].offset == 10.0);
    REQUIRE(benchmark.references[0].estimatedWinding.has_value());
    CHECK_FALSE(benchmark.references[0].rawEstimatedWinding.has_value());
    CHECK(*benchmark.references[0].estimatedWinding == -0.5);
    CHECK(benchmark.references[0].estimatedWindingObservations == 2);
}

TEST_CASE("Reference winding benchmark estimates each source independently")
{
    using Class = FiberTraceReferenceConstraintClass;
    const std::vector<FiberTraceReferenceWindingObservation> observations{
        {Class::Perpendicular, 2, 0.0, {2.1, 0.0}, 1, 0},
        {Class::ParallelSameWinding, 2, 0.0, {1.9, 0.0}, 1, 0},
        {Class::ParallelOtherWinding, 2, 0.0, {-4.0, 2.2}, 2, 0},
        {Class::Perpendicular, 2, 0.5, {2.3, 0.0}, 1, 1},
        {Class::ParallelSameWinding, 2, 0.5, {2.7, 0.0}, 1, 1},
    };

    const auto benchmark = calibrateFiberTraceReferenceWindings(
        observations, 0.5);

    CHECK(benchmark.globalSign == 1);
    REQUIRE(benchmark.gauges.size() == 1);
    CHECK(benchmark.gauges[0].offset == doctest::Approx(2.0));
    REQUIRE(benchmark.references.size() == 2);
    REQUIRE(benchmark.references[0].estimatedWinding.has_value());
    REQUIRE(benchmark.references[0].rawEstimatedWinding.has_value());
    CHECK(*benchmark.references[0].estimatedWinding == doctest::Approx(
        static_cast<double>(benchmark.globalSign) *
        (*benchmark.references[0].rawEstimatedWinding -
         benchmark.gauges[0].offset)));
    CHECK(*benchmark.references[0].estimatedWinding == doctest::Approx(0.0));
    CHECK(benchmark.references[0].estimatedWindingSupport == 3);
    CHECK(benchmark.references[0].estimatedWindingObservations == 3);
    REQUIRE(benchmark.references[1].estimatedWinding.has_value());
    REQUIRE(benchmark.references[1].rawEstimatedWinding.has_value());
    CHECK(*benchmark.references[1].estimatedWinding == doctest::Approx(
        static_cast<double>(benchmark.globalSign) *
        (*benchmark.references[1].rawEstimatedWinding -
         benchmark.gauges[0].offset)));
    CHECK(*benchmark.references[1].estimatedWinding == doctest::Approx(0.5));
    CHECK(benchmark.references[1].estimatedWindingSupport == 2);
    CHECK(benchmark.references[1].estimatedWindingObservations == 2);
}

TEST_CASE("Reference winding benchmark corrects a global sign reversal")
{
    using Class = FiberTraceReferenceConstraintClass;
    const std::vector<FiberTraceReferenceWindingObservation> observations{
        {Class::Perpendicular, 0, 0.0, {4.0, 0.0}, 1, 0},
        {Class::Perpendicular, 0, 0.5, {3.5, 0.0}, 1, 1},
        {Class::Perpendicular, 0, 1.0, {3.0, 0.0}, 1, 2},
        {Class::Perpendicular, 0, 1.5, {2.5, 0.0}, 1, 3},
    };

    const auto benchmark = calibrateFiberTraceReferenceWindings(
        observations, 0.1);

    CHECK(benchmark.globalSign == -1);
    REQUIRE(benchmark.gauges.size() == 1);
    CHECK(benchmark.gauges[0].offset == doctest::Approx(4.0));
    CHECK(benchmark.sum.right == observations.size());
    REQUIRE(benchmark.references.size() == observations.size());
    for (std::size_t source = 0; source < benchmark.references.size();
         ++source) {
        REQUIRE(benchmark.references[source].estimatedWinding.has_value());
        CHECK(*benchmark.references[source].estimatedWinding ==
              doctest::Approx(0.5 * static_cast<double>(source)));
    }
}

TEST_CASE("Reference observations use authoritative latent endpoint algebra")
{
    FiberTraceInterleavedWindingReport winding;
    winding.windingValid = {1, 1, 0};
    winding.mapLatentCoordinate = {4.25, -2.0, std::numeric_limits<double>::quiet_NaN()};
    winding.mapOrientationByPiece = {
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Vertical,
        FiberTraceFixedOrientation::Mixed,
    };
    winding.integerGaugeByPiece = {3, 4, 5};
    winding.componentByPiece = {0, 0, 0};
    winding.measurementScale = 2.0;

    FiberTraceConstraint perpendicular;
    perpendicular.parallelScore = 0.5;
    perpendicular.perpendicularScore = 0.5;
    perpendicular.signedWindingDelta = 0.75;
    const auto referenceA = makeFiberTraceReferenceWindingObservation(perpendicular, true, 1.0, 0, winding);
    CHECK(referenceA.constraintClass == FiberTraceReferenceConstraintClass::Perpendicular);
    CHECK(referenceA.integerGauge == 3);
    CHECK(referenceA.bpEndpointActive);
    CHECK(referenceA.bpEndpointOrientation ==
          FiberTraceFixedOrientation::Horizontal);
    CHECK(referenceA.bpOrientationComponent == 0);
    CHECK(referenceA.inferredReferenceWindingCount == 1);
    CHECK(referenceA.inferredReferenceWindings[0] == 2.75);

    const auto referenceB = makeFiberTraceReferenceWindingObservation(perpendicular, false, 1.0, 0, winding);
    CHECK(referenceB.inferredReferenceWindings[0] == 5.75);

    FiberTraceConstraint parallel = perpendicular;
    parallel.parallelScore = 0.75;
    parallel.perpendicularScore = 0.25;
    parallel.parallelWindingDistance = 1.6;
    parallel.signedParallelWindingDelta = 1.6;
    const auto other = makeFiberTraceReferenceWindingObservation(parallel, true, 0.5, 0, winding);
    CHECK(other.constraintClass == FiberTraceReferenceConstraintClass::ParallelOtherWinding);
    CHECK(other.inferredReferenceWindingCount == 1);
    CHECK(other.inferredReferenceWindings[0] == 1.25);

    parallel.parallelWindingDistance = 0.49;
    parallel.signedParallelWindingDelta = 0.49;
    const auto same = makeFiberTraceReferenceWindingObservation(parallel, true, 0.5, 1, winding);
    CHECK(same.constraintClass == FiberTraceReferenceConstraintClass::ParallelOtherWinding);
    CHECK(same.integerGauge == 4);
    CHECK(same.inferredReferenceWindingCount == 1);
    CHECK(same.inferredReferenceWindings[0] == -3.0);

    const auto defect = makeFiberTraceReferenceWindingObservation(perpendicular, true, 1.0, 2, winding);
    CHECK(defect.integerGauge == 5);
    CHECK(defect.inferredReferenceWindingCount == 0);
    CHECK_FALSE(defect.bpEndpointActive);
    CHECK(defect.bpEndpointOrientation == FiberTraceFixedOrientation::Mixed);
}

TEST_CASE("Fixed reference conflicts use authoritative factor losses")
{
    FiberTraceReferenceWindingBenchmark calibration;
    calibration.globalSign = 1;
    calibration.gauges.push_back({0, 0.0, 0, 0});

    FiberTraceReferenceWindingObservation perpendicular;
    perpendicular.integerGauge = 0;
    perpendicular.virtualReferenceWinding = 0.0;
    perpendicular.referenceSource = 0;
    perpendicular.bpPiece = 7;
    perpendicular.constraintIndex = 11;
    perpendicular.exactWindingFactor = true;
    perpendicular.bpLatentCoordinate = 1.0;
    perpendicular.referenceDeltaSign = 1.0;
    perpendicular.bpEndpointOrientation =
        FiberTraceFixedOrientation::Horizontal;
    perpendicular.bpEndpointActive = true;
    perpendicular.signedPerpendicularTarget = 0.5;
    perpendicular.perpendicularMagnitudePresent = true;
    perpendicular.perpendicularSignPresent = true;
    perpendicular.perpendicularCoefficient = 2.0;
    perpendicular.perpendicularSignPenalty = 3.0;
    perpendicular.rawCoefficient = 5.0;
    perpendicular.admittedCoefficient = 5.0;

    FiberTraceReferenceWindingObservation parallel;
    parallel.constraintClass =
        FiberTraceReferenceConstraintClass::ParallelOtherWinding;
    parallel.integerGauge = 0;
    parallel.virtualReferenceWinding = 2.0;
    parallel.referenceSource = 4;
    parallel.bpPiece = 9;
    parallel.constraintIndex = 12;
    parallel.exactWindingFactor = true;
    parallel.bpLatentCoordinate = 0.0;
    parallel.referenceDeltaSign = 1.0;
    parallel.bpEndpointOrientation = FiberTraceFixedOrientation::Vertical;
    parallel.bpEndpointActive = true;
    parallel.parallelDistance = 1.0;
    parallel.signedParallelTarget = -1.0;
    parallel.parallelMagnitudePresent = true;
    parallel.parallelSignPresent = true;
    parallel.hardParallelSign = true;
    parallel.rawParallelCoefficient = 4.0;
    parallel.admittedParallelCoefficient = 4.0;
    parallel.rawCoefficient = 4.0;
    parallel.admittedCoefficient = 4.0;

    const std::array observations{perpendicular, parallel};
    const auto conflicts = diagnoseFiberTraceReferenceClampedConflicts(
        observations, calibration);
    REQUIRE(conflicts.size() == 4);

    CHECK(conflicts[0].bpPiece == 7);
    CHECK(conflicts[0].constraintIndex == 11);
    CHECK(conflicts[0].factorClass ==
          FiberTraceReferenceFactorClass::PerpendicularMagnitudeNext);
    CHECK_FALSE(conflicts[0].hardViolation);
    CHECK(conflicts[0].predictedDelta == doctest::Approx(-1.0));
    CHECK(conflicts[0].targetDelta == doctest::Approx(0.5));
    CHECK(conflicts[0].residual == doctest::Approx(1.5));
    CHECK(conflicts[0].effectiveWeight == doctest::Approx(2.0));
    CHECK(conflicts[0].weightedLoss == doctest::Approx(3.0));

    CHECK(conflicts[1].factorClass ==
          FiberTraceReferenceFactorClass::PerpendicularSign);
    CHECK_FALSE(conflicts[1].hardViolation);
    CHECK(conflicts[1].residual == doctest::Approx(1.0));
    CHECK(conflicts[1].weightedLoss == doctest::Approx(3.0));

    CHECK(conflicts[2].bpPiece == 9);
    CHECK(conflicts[2].constraintIndex == 12);
    CHECK(conflicts[2].factorClass ==
          FiberTraceReferenceFactorClass::ParallelMagnitudeOne);
    CHECK_FALSE(conflicts[2].hardViolation);
    CHECK(conflicts[2].predictedDelta == doctest::Approx(2.0));
    CHECK(conflicts[2].targetDelta == doctest::Approx(-1.0));
    CHECK(conflicts[2].residual == doctest::Approx(3.0));
    CHECK(conflicts[2].weightedLoss == doctest::Approx(12.0));

    CHECK(conflicts[3].factorClass ==
          FiberTraceReferenceFactorClass::ParallelSign);
    CHECK(conflicts[3].hardViolation);
    CHECK(conflicts[3].residual == doctest::Approx(1.0));
    CHECK(conflicts[3].weightedLoss == doctest::Approx(0.0));
}

TEST_CASE("Reference-only conflicts use materialized seven-class factors")
{
    FiberTraceConstraintReport constraints;
    constraints.inputTraces = 6;
    constraints.pieces.resize(6);
    constraints.constraints.resize(8);
    for (std::size_t index = 0; index < constraints.pieces.size(); ++index)
        constraints.pieces[index].traceIndex = index;
    for (std::size_t index = 0; index < constraints.constraints.size(); ++index) {
        constraints.constraints[index].pieceA = 0;
        constraints.constraints[index].pieceB = std::min<std::size_t>(index, 4);
    }
    const std::array<std::size_t, 6> sourceIds{0, 1, 2, 3, 4, 0};

    std::vector<FiberTraceWindingFactorDiagnostic> factors(8);
    for (std::size_t index = 0; index < factors.size(); ++index) {
        factors[index].constraintIndex = index;
        factors[index].canonicalNodeA = 0;
        factors[index].canonicalNodeB = std::min<std::size_t>(index, 4);
    }

    factors[0].canonicalNodeB = 1;
    factors[0].perpendicularMagnitudePresent = true;
    factors[0].perpendicularSignPresent = true;
    factors[0].effectivePerpendicularSignedDelta = 0.5;
    factors[0].effectivePerpendicularWindingWeight = 2.0;
    factors[0].effectivePerpendicularSignPenalty = 3.0;

    factors[1].constraintIndex = 1;
    factors[1].canonicalNodeB = 3;
    factors[1].perpendicularMagnitudePresent = true;
    factors[1].perpendicularSignPresent = true;
    factors[1].effectivePerpendicularSignedDelta = -1.5;
    factors[1].effectivePerpendicularWindingWeight = 2.0;
    factors[1].hardPerpendicularSign = true;

    factors[2].constraintIndex = 2;
    factors[2].canonicalNodeB = 2;
    factors[2].parallelMagnitudePresent = true;
    factors[2].effectiveParallelWindingDistance = 0.0;
    factors[2].effectiveParallelWindingWeight = 0.5;

    factors[3].constraintIndex = 3;
    factors[3].canonicalNodeB = 2;
    factors[3].parallelMagnitudePresent = true;
    factors[3].parallelSignPresent = true;
    factors[3].effectiveSignedParallelDelta = -1.0;
    factors[3].effectiveParallelWindingDistance = 1.0;
    factors[3].effectiveParallelWindingWeight = 4.0;
    factors[3].effectiveParallelSignPenalty = 5.0;

    factors[4].constraintIndex = 4;
    factors[4].canonicalNodeB = 4;
    factors[4].parallelMagnitudePresent = true;
    factors[4].effectiveSignedParallelDelta = 2.0;
    factors[4].effectiveParallelWindingDistance = 2.0;
    factors[4].effectiveParallelWindingWeight = 0.0;

    factors[5].constraintIndex = 5;
    factors[5].canonicalNodeB = 4;
    factors[5].parallelMagnitudePresent = true;
    factors[5].effectiveSignedParallelDelta = 2.0;
    factors[5].effectiveParallelWindingDistance = 2.0;
    factors[5].effectiveParallelWindingWeight = 1.0;

    factors[6].constraintIndex = 6;
    factors[6].canonicalNodeB = 5;
    factors[6].parallelMagnitudePresent = true;
    factors[6].effectiveParallelWindingDistance = 0.0;
    factors[6].effectiveParallelWindingWeight = 10.0;

    constraints.constraints[7].hardContinuity = true;
    factors[7].constraintIndex = 7;
    factors[7].canonicalNodeB = 4;
    factors[7].parallelMagnitudePresent = true;
    factors[7].effectiveParallelWindingDistance = 2.0;
    factors[7].effectiveParallelWindingWeight = 10.0;

    const auto conflicts = diagnoseFiberTraceReferenceConstraintConflicts(
        constraints, sourceIds, factors, 1);
    REQUIRE(conflicts.size() == 8);
    CHECK(conflicts[0].factorClass ==
          FiberTraceReferenceFactorClass::PerpendicularMagnitudeNext);
    CHECK(conflicts[0].residual == doctest::Approx(0.0));
    CHECK(conflicts[1].factorClass ==
          FiberTraceReferenceFactorClass::PerpendicularSign);
    CHECK(conflicts[1].residual == doctest::Approx(0.0));
    CHECK(conflicts[2].factorClass ==
          FiberTraceReferenceFactorClass::PerpendicularMagnitudeFar);
    CHECK(conflicts[2].residual == doctest::Approx(3.0));
    CHECK(conflicts[2].weightedLoss == doctest::Approx(6.0));
    CHECK(conflicts[3].factorClass ==
          FiberTraceReferenceFactorClass::PerpendicularSign);
    CHECK(conflicts[3].hardViolation);
    CHECK(conflicts[3].weightedLoss == doctest::Approx(0.0));
    CHECK(conflicts[4].factorClass ==
          FiberTraceReferenceFactorClass::ParallelMagnitudeSame);
    CHECK(conflicts[4].residual == doctest::Approx(1.0));
    CHECK(conflicts[4].weightedLoss == doctest::Approx(0.5));
    CHECK(conflicts[5].factorClass ==
          FiberTraceReferenceFactorClass::ParallelMagnitudeOne);
    CHECK(conflicts[5].residual == doctest::Approx(2.0));
    CHECK(conflicts[5].weightedLoss == doctest::Approx(8.0));
    CHECK(conflicts[6].factorClass ==
          FiberTraceReferenceFactorClass::ParallelSign);
    CHECK(conflicts[6].residual == doctest::Approx(1.0));
    CHECK(conflicts[6].weightedLoss == doctest::Approx(5.0));
    CHECK(conflicts[7].factorClass ==
          FiberTraceReferenceFactorClass::ParallelMagnitudeFar);
    CHECK(conflicts[7].residual == doctest::Approx(0.0));
}

TEST_CASE("Reference orientation benchmark calibrates component H V gauges")
{
    using Class = FiberTraceReferenceConstraintClass;
    using Orientation = FiberTraceFixedOrientation;
    using Relation = FiberTraceReferenceOrientationRelation;
    const auto observation = [](
                                 std::size_t source,
                                 Class constraintClass,
                                 Orientation orientation,
                                 std::size_t component,
                                 bool active = true) {
        FiberTraceReferenceWindingObservation result;
        result.constraintClass = constraintClass;
        result.virtualReferenceWinding =
            0.5 * static_cast<double>(source);
        result.referenceSource = source;
        result.rawCoefficient = 0.0;
        result.admittedCoefficient = 0.0;
        result.bpEndpointOrientation = orientation;
        result.bpOrientationComponent = component;
        result.bpEndpointActive = active;
        return result;
    };

    std::vector<FiberTraceReferenceWindingObservation> observations{
        observation(0, Class::ParallelSameWinding, Orientation::Horizontal, 4),
        observation(1, Class::ParallelOtherWinding, Orientation::Vertical, 4),
        observation(0, Class::Perpendicular, Orientation::Vertical, 4),
        observation(1, Class::Perpendicular, Orientation::Horizontal, 4),
        observation(0, Class::ParallelSameWinding, Orientation::Vertical, 9),
        observation(1, Class::ParallelOtherWinding, Orientation::Horizontal, 9),
        observation(0, Class::Perpendicular, Orientation::Horizontal, 9),
        observation(1, Class::Perpendicular, Orientation::Vertical, 9),
        observation(0, Class::ParallelSameWinding, Orientation::Horizontal, 4),
        observation(2, Class::Perpendicular, Orientation::Mixed, 4, false),
    };
    const auto benchmark = benchmarkFiberTraceReferenceOrientations(
        observations);

    REQUIRE(benchmark.components.size() == 2);
    CHECK(benchmark.components[0].component == 4);
    CHECK(benchmark.components[0].evenReferenceIsHorizontal);
    CHECK(benchmark.components[0].evenHorizontalRight == 5);
    CHECK(benchmark.components[0].evenVerticalRight == 0);
    CHECK(benchmark.components[1].component == 9);
    CHECK_FALSE(benchmark.components[1].evenReferenceIsHorizontal);
    CHECK(benchmark.components[1].evenHorizontalRight == 0);
    CHECK(benchmark.components[1].evenVerticalRight == 4);
    CHECK(benchmark.excludedInactive == 1);
    REQUIRE(benchmark.references.size() == 3);
    CHECK(benchmark.references[2].sum.total() == 0);
    const auto perpendicular = static_cast<std::size_t>(
        Relation::Perpendicular);
    const auto parallel = static_cast<std::size_t>(Relation::Parallel);
    CHECK(benchmark.relations[perpendicular].right == 4);
    CHECK(benchmark.relations[perpendicular].wrong == 0);
    CHECK(benchmark.relations[parallel].right == 5);
    CHECK(benchmark.relations[parallel].wrong == 0);
    CHECK(benchmark.sum.right == 9);
    CHECK(benchmark.sum.wrong == 0);
    std::size_t sourceTotal = 0;
    for (const auto& source : benchmark.references)
        sourceTotal += source.sum.total();
    CHECK(sourceTotal == benchmark.sum.total());

    observations = {
        observation(0, Class::ParallelSameWinding, Orientation::Horizontal, 2),
        observation(0, Class::ParallelSameWinding, Orientation::Vertical, 2),
    };
    const auto tied = benchmarkFiberTraceReferenceOrientations(observations);
    REQUIRE(tied.components.size() == 1);
    CHECK(tied.components[0].evenReferenceIsHorizontal);
    CHECK(tied.sum.right == 1);
    CHECK(tied.sum.wrong == 1);

    observations[0].virtualReferenceWinding = 0.5;
    CHECK_THROWS_AS(
        benchmarkFiberTraceReferenceOrientations(observations),
        std::invalid_argument);
    observations[0].virtualReferenceWinding = 0.0;
    observations[0].bpEndpointActive = true;
    observations[0].bpEndpointOrientation = Orientation::Mixed;
    CHECK_THROWS_AS(
        benchmarkFiberTraceReferenceOrientations(observations),
        std::invalid_argument);
}

TEST_CASE("Reference estimate parity separates half and whole winding errors")
{
    CHECK_FALSE(fiberTraceReferenceEstimateParityMatches(3, std::nullopt));
    REQUIRE(fiberTraceReferenceEstimateParityMatches(7, 3.5).has_value());
    CHECK(*fiberTraceReferenceEstimateParityMatches(7, 3.5));
    CHECK_FALSE(*fiberTraceReferenceEstimateParityMatches(7, 4.0));
    CHECK(*fiberTraceReferenceEstimateParityMatches(1, -0.5));
    CHECK(*fiberTraceReferenceEstimateParityMatches(1, 1.5));
    CHECK_FALSE(*fiberTraceReferenceEstimateParityMatches(0, -0.5));
    CHECK_THROWS_AS(
        fiberTraceReferenceEstimateParityMatches(0, 0.25),
        std::invalid_argument);
}

TEST_CASE("Reference observations retain canonical diagnostic weights")
{
    FiberTraceInterleavedWindingReport winding;
    winding.windingValid = {1};
    winding.mapLatentCoordinate = {4.0};
    winding.mapOrientationByPiece = {
        FiberTraceFixedOrientation::Horizontal};
    winding.integerGaugeByPiece = {0};
    winding.componentByPiece = {0};
    winding.measurementScale = 2.0;

    FiberTraceConstraint perpendicular;
    perpendicular.parallelScore = 0.2;
    perpendicular.perpendicularScore = 0.8;
    perpendicular.signedWindingDelta = -1.0;
    FiberTraceWindingBeliefPropagationConfig config;
    useUnitClassWeights(config);
    config.decisionConfidence =
        FiberTraceWindingDecisionConfidence::Legacy;
    config.normalConfidence = FiberTraceWindingNormalConfidence::None;
    config.enforcePerpendicularWindingSign = false;
    config.enforceParallelWindingSign = false;
    config.finiteSignInfringementCost.reset();
    config.parallelWindingDistanceCutoff = 0.5;
    const auto next = makeFiberTraceReferenceWindingObservation(
        perpendicular, true, 0.0, 0, winding, config);
    CHECK(next.canonicalWindingDistance == 1.5);
    CHECK(next.rawCoefficient == doctest::Approx(0.4));
    CHECK(next.admittedCoefficient == doctest::Approx(0.4));
    CHECK(next.coordinateResidualScale == 1.0);

    perpendicular.signedWindingDelta = 1.00001;
    const auto far = makeFiberTraceReferenceWindingObservation(
        perpendicular, false, 0.0, 0, winding, config);
    CHECK(far.canonicalWindingDistance == 2.5);
    CHECK(far.rawCoefficient == doctest::Approx(0.2));
    CHECK(far.admittedCoefficient == doctest::Approx(0.2));

    FiberTraceConstraint parallel;
    parallel.parallelScore = 0.75;
    parallel.perpendicularScore = 0.25;
    parallel.parallelWindingDistance = 0.49;
    parallel.signedParallelWindingDelta = -0.49;
    const auto same = makeFiberTraceReferenceWindingObservation(
        parallel, true, 0.0, 0, winding, config);
    CHECK(same.canonicalWindingDistance == 1.0);
    CHECK(same.rawCoefficient == doctest::Approx(0.375));
    CHECK(same.admittedCoefficient == 0.0);
    CHECK(same.coordinateResidualScale == 1.0);

    parallel.parallelWindingDistance = 0.5;
    parallel.signedParallelWindingDelta = -0.5;
    const auto one = makeFiberTraceReferenceWindingObservation(
        parallel, false, 0.0, 0, winding, config);
    CHECK(one.canonicalWindingDistance == 1.0);
    CHECK(one.rawCoefficient == doctest::Approx(0.375));
    CHECK(one.admittedCoefficient == 0.0);

    parallel.parallelWindingDistance = 1.5;
    parallel.signedParallelWindingDelta = 1.5;
    config.parallelWindingDistanceCutoff = 3.0;
    const auto two = makeFiberTraceReferenceWindingObservation(
        parallel, true, 0.0, 0, winding, config);
    CHECK(two.canonicalWindingDistance == 3.0);
    CHECK(two.rawCoefficient == doctest::Approx(0.09375));
    CHECK(two.admittedCoefficient == 0.0);

    config.parallelWindingDistanceCutoff = 0.0;
    CHECK_THROWS_AS(
        makeFiberTraceReferenceWindingObservation(
            parallel, true, 0.0, 0, winding, config),
        std::invalid_argument);
}

TEST_CASE("Reference observations apply all canonical class weights")
{
    FiberTraceInterleavedWindingReport winding;
    winding.windingValid = {1};
    winding.mapLatentCoordinate = {0.0};
    winding.mapOrientationByPiece = {
        FiberTraceFixedOrientation::Horizontal};
    winding.integerGaugeByPiece = {0};
    winding.componentByPiece = {0};
    winding.measurementScale = 1.0;

    FiberTraceWindingBeliefPropagationConfig config;
    config.decisionConfidence =
        FiberTraceWindingDecisionConfidence::Legacy;
    config.normalConfidence = FiberTraceWindingNormalConfidence::None;
    config.enforcePerpendicularWindingSign = false;
    config.enforceParallelWindingSign = false;
    config.finiteSignInfringementCost.reset();
    config.perpendicularNextWeight = 2.0;
    config.perpendicularFarWeight = 3.0;
    config.parallelSameWeight = 4.0;
    config.parallelOneWeight = 5.0;
    config.parallelFarWeight = 6.0;

    FiberTraceConstraint constraint;
    constraint.parallelScore = 0.2;
    constraint.perpendicularScore = 0.8;
    constraint.signedWindingDelta = 0.5;
    CHECK(makeFiberTraceReferenceWindingObservation(
              constraint, true, 0.0, 0, winding, config)
              .admittedCoefficient == doctest::Approx(1.6));

    constraint.signedWindingDelta = 1.5;
    CHECK(makeFiberTraceReferenceWindingObservation(
              constraint, true, 0.0, 0, winding, config)
              .admittedCoefficient == doctest::Approx(1.2));

    constraint.parallelScore = 0.8;
    constraint.perpendicularScore = 0.2;
    constraint.parallelWindingDistance = 0.0;
    constraint.signedParallelWindingDelta = 0.0;
    CHECK(makeFiberTraceReferenceWindingObservation(
              constraint, true, 0.0, 0, winding, config)
              .admittedCoefficient == doctest::Approx(3.2));

    constraint.parallelWindingDistance = 1.0;
    constraint.signedParallelWindingDelta = 1.0;
    CHECK(makeFiberTraceReferenceWindingObservation(
              constraint, true, 0.0, 0, winding, config)
              .admittedCoefficient == doctest::Approx(2.0));

    constraint.parallelWindingDistance = 0.1;
    constraint.signedParallelWindingDelta = 2.0;
    CHECK(makeFiberTraceReferenceWindingObservation(
              constraint, true, 0.0, 0, winding, config)
              .admittedCoefficient == doctest::Approx(1.2));

    config.parallelFarWeight = 0.0;
    const auto disabledMagnitude = makeFiberTraceReferenceWindingObservation(
        constraint, true, 0.0, 0, winding, config);
    CHECK(disabledMagnitude.rawCoefficient == 0.0);
    CHECK(disabledMagnitude.admittedCoefficient == 0.0);
}

TEST_CASE("Reference constraint groups expose weighted calibrated inference")
{
    using Class = FiberTraceReferenceConstraintClass;
    using Group = FiberTraceReferenceConstraintGroup;
    FiberTraceReferenceWindingBenchmark calibration;
    calibration.globalSign = -1;
    calibration.gauges = {
        {3, 4.0, 0, 0},
        {7, -2.0, 0, 0},
    };
    calibration.references.resize(2);

    const std::vector<FiberTraceReferenceWindingObservation> observations{
        {Class::Perpendicular, 3, 1.0, {3.0, 0.0}, 1, 0,
         0.5, 2.0, 2.0, 2.0},
        {Class::Perpendicular, 7, 1.0, {-4.0, 0.0}, 1, 0,
         1.5, 1.0, 1.0, 2.0},
        {Class::ParallelSameWinding, 3, 1.0, {3.0, 0.0}, 1, 0,
         0.0, 3.0, 3.0, 1.0},
        {Class::ParallelOtherWinding, 7, 1.0, {-2.5, 0.0}, 1, 0,
         1.0, 4.0, 4.0, 1.0},
        {Class::ParallelOtherWinding, 3, 1.0, {2.5, 0.0}, 1, 0,
         2.0, 5.0, 0.0, 1.0},
        {Class::ParallelSameWinding, 3, 0.5, {5.0, 0.0}, 1, 1,
         0.0, 1.0, 1.0, 1.0},
        {Class::ParallelSameWinding, 7, 0.5, {-3.0, 0.0}, 1, 1,
         0.0, 1.0, 1.0, 1.0},
    };

    const auto summary = summarizeFiberTraceReferenceConstraintGroups(
        observations, calibration);
    REQUIRE(summary.size() == 2);
    const auto& source = summary[0].groups;
    const auto& next = source[static_cast<std::size_t>(Group::PerpendicularNext)];
    CHECK(next.observations == 1);
    CHECK(next.rawCoefficient == 2.0);
    CHECK(next.truthLoss == 0.0);
    REQUIRE(next.preferredWinding.has_value());
    CHECK(*next.preferredWinding == 1.0);

    const auto& far = source[static_cast<std::size_t>(Group::PerpendicularFar)];
    CHECK(far.observations == 1);
    CHECK(far.rawCoefficient == 1.0);
    CHECK(far.truthLoss == doctest::Approx(0.5));
    REQUIRE(far.preferredWinding.has_value());
    CHECK(*far.preferredWinding == 2.0);
    CHECK(far.preferredLoss == 0.0);

    const auto& same = source[static_cast<std::size_t>(Group::ParallelSame)];
    CHECK(same.rawCoefficient == 3.0);
    REQUIRE(same.preferredWinding.has_value());
    CHECK(*same.preferredWinding == 1.0);

    const auto& one = source[static_cast<std::size_t>(Group::ParallelOne)];
    CHECK(one.rawCoefficient == 4.0);
    CHECK(one.truthLoss == doctest::Approx(2.0));
    REQUIRE(one.preferredWinding.has_value());
    CHECK(*one.preferredWinding == 0.5);

    const auto& twoPlus = source[static_cast<std::size_t>(Group::ParallelTwoPlus)];
    CHECK(twoPlus.rawCoefficient == 5.0);
    CHECK(twoPlus.admittedCoefficient == 0.0);
    CHECK_FALSE(twoPlus.preferredWinding.has_value());

    const auto& flat = summary[1].groups[
        static_cast<std::size_t>(Group::ParallelSame)];
    CHECK(flat.observations == 2);
    CHECK(flat.truthLoss == doctest::Approx(2.0));
    REQUIRE(flat.preferredWinding.has_value());
    CHECK(*flat.preferredWinding == -1.0);
    CHECK(flat.preferredLoss == doctest::Approx(2.0));

    CHECK(std::string(fiberTraceReferenceConstraintGroupName(
              Group::PerpendicularNext)) == "perp_0.5");
    CHECK(std::string(fiberTraceReferenceConstraintGroupName(
              Group::PerpendicularFar)) == "perp_1.5+");
    CHECK(std::string(fiberTraceReferenceConstraintGroupName(
              Group::ParallelSame)) == "parallel_0");
    CHECK(std::string(fiberTraceReferenceConstraintGroupName(
              Group::ParallelOne)) == "parallel_1");
    CHECK(std::string(fiberTraceReferenceConstraintGroupName(
              Group::ParallelTwoPlus)) == "parallel_2+");

    auto invalid = observations;
    invalid.front().rawCoefficient =
        std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(
        summarizeFiberTraceReferenceConstraintGroups(invalid, calibration),
        std::invalid_argument);
}

TEST_CASE("Reference constraint inference minimizes hard violations first")
{
    FiberTraceReferenceWindingBenchmark calibration;
    calibration.globalSign = 1;
    calibration.gauges = {{0, 0.0, 0, 0}};
    calibration.references.resize(1);

    const auto observation = [](double target) {
        FiberTraceReferenceWindingObservation result;
        result.constraintClass =
            FiberTraceReferenceConstraintClass::Perpendicular;
        result.integerGauge = 0;
        result.virtualReferenceWinding = 0.0;
        result.inferredReferenceWindings[0] = -target;
        result.inferredReferenceWindingCount = 1;
        result.referenceSource = 0;
        result.canonicalWindingDistance = 0.5;
        result.rawCoefficient = 1.0;
        result.admittedCoefficient = 1.0;
        result.coordinateResidualScale = 1.0;
        result.exactWindingFactor = true;
        result.bpLatentCoordinate = 0.0;
        result.referenceDeltaSign = -1.0;
        result.perpendicularCoefficient = 1.0;
        result.signedPerpendicularTarget = target;
        result.hardPerpendicularSign = true;
        return result;
    };
    const std::array observations{observation(0.5), observation(-0.5)};
    const auto summary = summarizeFiberTraceReferenceConstraintGroups(
        observations, calibration);
    const auto& group = summary[0].groups[static_cast<std::size_t>(
        FiberTraceReferenceConstraintGroup::PerpendicularNext)];
    CHECK(group.truthHardViolations == 2);
    REQUIRE(group.preferredWinding.has_value());
    CHECK(*group.preferredWinding == -0.5);
    CHECK(group.preferredHardViolations == 1);
    CHECK(group.preferredLoss == doctest::Approx(1.0));
    REQUIRE(summary[0].all.preferredWinding.has_value());
    CHECK(*summary[0].all.preferredWinding ==
          *group.preferredWinding);
}

TEST_CASE("H/V-aware winding evidence decays by half-integer distance bin")
{
    const auto source = lines(8);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.25, 0.5);
    addMeasured(report, 2, 3, 0.25, 1.5);
    addMeasured(report, 4, 5, 0.25, 2.5);
    addMeasured(report, 7, 6, 0.25, -3.5);

    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    joint.mixedUnaryCost = 5.0;
    joint.stableIterations = 1;
    const auto solved = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), joint);

    const std::array parallelMultipliers{0.5, 0.25, 0.125, 0.0625};
    const std::array perpendicularMultipliers{1.0, 0.5, 0.25, 0.125};
    REQUIRE(solved.factorDiagnostics.size() == parallelMultipliers.size());
    for (std::size_t index = 0; index < parallelMultipliers.size(); ++index) {
        const auto& diagnostic = solved.factorDiagnostics[index];
        CHECK(diagnostic.parallelScore == 0.25);
        CHECK(diagnostic.perpendicularScore == 0.75);
        CHECK(diagnostic.parallelWindingWeightMultiplier ==
            parallelMultipliers[index]);
        CHECK(diagnostic.perpendicularWindingWeightMultiplier ==
            perpendicularMultipliers[index]);
        CHECK(diagnostic.effectiveParallelWindingWeight == 0.0);
        CHECK(diagnostic.effectivePerpendicularWindingWeight ==
            doctest::Approx(0.75 * perpendicularMultipliers[index]));
    }

    const auto raw = solveFiberTraceWindingBeliefPropagation(
        report, topology(source, report), config());
    REQUIRE(raw.factorDiagnostics.size() == parallelMultipliers.size());
    for (const auto& diagnostic : raw.factorDiagnostics) {
        CHECK(diagnostic.parallelWindingWeightMultiplier == 1.0);
        CHECK(diagnostic.perpendicularWindingWeightMultiplier == 1.0);
        CHECK(diagnostic.effectiveParallelWindingWeight == 0.0);
        CHECK(diagnostic.effectivePerpendicularWindingWeight == 0.75);
    }
}

TEST_CASE("H/V-aware winding applies canonical class weights")
{
    const auto source = lines(10);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.25, 0.5);
    addMeasured(report, 2, 3, 0.25, 1.5);
    addMeasured(report, 4, 5, 0.75, 0.0);
    addMeasured(report, 6, 7, 0.75, 1.0);
    addMeasured(report, 8, 9, 0.75, 2.0);

    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.perpendicularNextWeight = 2.0;
    joint.perpendicularFarWeight = 3.0;
    joint.parallelSameWeight = 4.0;
    joint.parallelOneWeight = 5.0;
    joint.parallelFarWeight = 6.0;
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    joint.mixedUnaryCost = 5.0;
    joint.stableIterations = 1;
    const auto solved = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), joint);
    REQUIRE(solved.factorDiagnostics.size() == report.constraints.size());
    std::vector<const FiberTraceWindingFactorDiagnostic*> byConstraint(
        report.constraints.size());
    for (const auto& diagnostic : solved.factorDiagnostics)
        byConstraint.at(diagnostic.constraintIndex) = &diagnostic;

    CHECK(byConstraint[0]->perpendicularWindingWeightMultiplier ==
          doctest::Approx(2.0));
    CHECK(byConstraint[0]->effectivePerpendicularWindingWeight ==
          doctest::Approx(1.5));
    CHECK(byConstraint[1]->perpendicularWindingWeightMultiplier ==
          doctest::Approx(1.5));
    CHECK(byConstraint[1]->effectivePerpendicularWindingWeight ==
          doctest::Approx(1.125));
    CHECK(byConstraint[2]->parallelWindingWeightMultiplier ==
          doctest::Approx(4.0));
    CHECK(byConstraint[2]->effectiveParallelWindingWeight ==
          doctest::Approx(3.0));
    CHECK(byConstraint[3]->parallelWindingWeightMultiplier ==
          doctest::Approx(2.5));
    CHECK(byConstraint[3]->effectiveParallelWindingWeight ==
          doctest::Approx(1.875));
    CHECK(byConstraint[4]->parallelWindingWeightMultiplier ==
          doctest::Approx(1.5));
    CHECK(byConstraint[4]->effectiveParallelWindingWeight ==
          doctest::Approx(1.125));
    auto signedTargetWins = report;
    signedTargetWins.constraints[4].parallelWindingDistance = 0.1;
    const auto signedSolved =
        solveFiberTraceJointGridWindingBeliefPropagation(
            signedTargetWins, topology(source, signedTargetWins), joint);
    const auto found = std::find_if(
        signedSolved.factorDiagnostics.begin(),
        signedSolved.factorDiagnostics.end(),
        [](const FiberTraceWindingFactorDiagnostic& diagnostic) {
            return diagnostic.constraintIndex == 4;
        });
    REQUIRE(found != signedSolved.factorDiagnostics.end());
    CHECK(found->effectiveParallelWindingDistance == 2.0);
    CHECK(found->parallelWindingWeightMultiplier == doctest::Approx(1.5));
}

TEST_CASE("H/V-aware winding uses signed parallel targets and cutoff")
{
    const auto source = lines(6);
    auto report = pieces(source.size());
    FiberTraceConstraint same;
    same.pieceA = 0;
    same.pieceB = 1;
    same.parallelScore = 1.0;
    same.perpendicularScore = 0.0;
    same.parallelWindingDistance = 0.2;
    same.signedParallelWindingDelta = 0.2;
    report.constraints.push_back(same);
    FiberTraceConstraint separate = same;
    separate.pieceA = 2;
    separate.pieceB = 3;
    separate.parallelWindingDistance = 0.6;
    separate.signedParallelWindingDelta = 0.6;
    report.constraints.push_back(separate);
    addMeasured(report, 4, 5, 0.25, 0.6);

    FiberTraceJointGridWindingConfig unfiltered;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(unfiltered) =
        config();
    unfiltered.fixedPhaseMagnitude = 0.5;
    unfiltered.fixedMeasurementScale = 1.0;
    unfiltered.mixedUnaryCost = 5.0;
    unfiltered.stableIterations = 1;
    const auto all = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), unfiltered);
    REQUIRE(all.factorDiagnostics.size() == 3);
    CHECK(all.factorDiagnostics[0].effectiveParallelWindingDistance == 0.0);
    CHECK(all.factorDiagnostics[1].effectiveParallelWindingDistance == 1.0);
    CHECK(all.factorDiagnostics[1].parallelWindingRetained);
    CHECK(all.factorDiagnostics[1].effectiveParallelWindingWeight == 0.5);
    CHECK_FALSE(
        all.factorDiagnostics[1].effectivePerpendicularSignedDelta.has_value());
    CHECK(all.factorDiagnostics[2].effectiveParallelWindingDistance == 1.0);
    CHECK(all.factorDiagnostics[2].effectivePerpendicularSignedDelta == 0.5);

    auto filtered = unfiltered;
    filtered.parallelWindingDistanceCutoff = 0.5;
    const auto sameOnly = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), filtered);
    REQUIRE(sameOnly.factorDiagnostics.size() == 3);
    CHECK(sameOnly.factorDiagnostics[0].parallelWindingRetained);
    CHECK(sameOnly.factorDiagnostics[0].effectiveParallelWindingWeight == 1.0);
    CHECK_FALSE(sameOnly.factorDiagnostics[1].parallelWindingRetained);
    CHECK(sameOnly.factorDiagnostics[1].effectiveParallelWindingWeight == 0.0);
    CHECK(sameOnly.factorDiagnostics[1].effectivePerpendicularWindingWeight ==
        0.0);
    CHECK_FALSE(sameOnly.factorDiagnostics[2].parallelWindingRetained);
    CHECK(sameOnly.factorDiagnostics[2].effectiveParallelWindingWeight == 0.0);
    CHECK(sameOnly.factorDiagnostics[2].effectivePerpendicularWindingWeight ==
        doctest::Approx(0.75));
    CHECK(sameOnly.connectedComponents == 3);

    FiberTraceInterleavedWindingConfig alternating;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(alternating) =
        config();
    alternating.parallelWindingDistanceCutoff = 0.5;
    alternating.mixedUnaryCost = 5.0;
    alternating.maximumCalibrationIterations = 2;
    const std::vector fixed(
        source.size(), FiberTraceFixedOrientation::Horizontal);
    const auto alternatingResult =
        solveFiberTraceInterleavedWindingBeliefPropagation(
            report,
            topology(source, report),
            orientationBeliefs({
                {0.99, 0.005, 0.005},
                {0.99, 0.005, 0.005},
                {0.99, 0.005, 0.005},
                {0.99, 0.005, 0.005},
                {0.99, 0.005, 0.005},
                {0.99, 0.005, 0.005},
            }),
            alternating,
            {},
            fixed);
    REQUIRE(alternatingResult.factorDiagnostics.size() == 3);
    CHECK(alternatingResult.factorDiagnostics[0].parallelWindingRetained);
    CHECK_FALSE(
        alternatingResult.factorDiagnostics[1].parallelWindingRetained);
    CHECK(alternatingResult.factorDiagnostics[2]
              .effectivePerpendicularWindingWeight == doctest::Approx(0.75));
    CHECK(alternatingResult.connectedComponents == 3);

    auto orientationOnlyReport = pieces(2);
    FiberTraceConstraint orientationOnly;
    orientationOnly.pieceA = 0;
    orientationOnly.pieceB = 1;
    orientationOnly.parallelScore = 0.0;
    orientationOnly.perpendicularScore = 1.0;
    orientationOnly.windingDistance = 0.6;
    orientationOnlyReport.constraints.push_back(orientationOnly);
    const auto orientationOnlyResult =
        solveFiberTraceJointGridWindingBeliefPropagation(
            orientationOnlyReport,
            topology(lines(2), orientationOnlyReport),
            filtered);
    CHECK(orientationOnlyResult.connectedComponents == 1);
    CHECK(orientationOnlyResult.classAProbability[0] > 0.99);
    CHECK(orientationOnlyResult.classBProbability[1] > 0.95);
    CHECK(orientationOnlyResult.windingValid ==
        std::vector<unsigned char>{1, 1});
    CHECK(orientationOnlyResult.mapWinding == std::vector<int>{0, 0});
    CHECK(orientationOnlyResult.integerGaugeByPiece == std::vector<std::size_t>{0, 1});
    CHECK(orientationOnlyResult.mapOrientationByPiece == std::vector{FiberTraceFixedOrientation::Horizontal, FiberTraceFixedOrientation::Vertical});
    CHECK(orientationOnlyResult.mapLatentCoordinate[0] == 0.0);
    CHECK(orientationOnlyResult.mapLatentCoordinate[1] == doctest::Approx(0.5 * static_cast<double>(orientationOnlyResult.componentPhaseSign.at(0))));

    auto invalid = unfiltered;
    invalid.parallelWindingDistanceCutoff = 0.0;
    CHECK_THROWS_AS(
        solveFiberTraceJointGridWindingBeliefPropagation(
            report, topology(source, report), invalid),
        std::invalid_argument);
}

TEST_CASE("Signed parallel winding distinguishes opposite ladder directions")
{
    const auto source = lines(2);
    const std::vector fixed(
        source.size(), FiberTraceFixedOrientation::Horizontal);
    const auto solveJoint = [&](double signedDelta) {
        auto report = pieces(2);
        addMeasured(report, 0, 1, 1.0, signedDelta);
        FiberTraceJointGridWindingConfig joint;
        static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) =
            config();
        joint.fixedPhaseMagnitude = 0.5;
        joint.fixedMeasurementScale = 1.0;
        joint.mixedUnaryCost = 20.0;
        joint.stableIterations = 1;
        return solveFiberTraceJointGridWindingBeliefPropagation(
            report, topology(source, report), joint, {}, fixed);
    };
    const auto positiveJoint = solveJoint(1.0);
    const auto negativeJoint = solveJoint(-1.0);
    CHECK(positiveJoint.mapLatentCoordinate[1] -
              positiveJoint.mapLatentCoordinate[0] == doctest::Approx(1.0));
    CHECK(negativeJoint.mapLatentCoordinate[1] -
              negativeJoint.mapLatentCoordinate[0] == doctest::Approx(-1.0));

    const auto solveAlternating = [&](double signedDelta) {
        auto report = pieces(2);
        addMeasured(report, 0, 1, 1.0, signedDelta);
        FiberTraceInterleavedWindingConfig alternating;
        static_cast<FiberTraceWindingBeliefPropagationConfig&>(alternating) =
            config();
        alternating.mixedUnaryCost = 20.0;
        alternating.maximumCalibrationIterations = 2;
        return solveFiberTraceInterleavedWindingBeliefPropagation(
            report,
            topology(source, report),
            orientationBeliefs({
                {0.99, 0.005, 0.005},
                {0.99, 0.005, 0.005},
            }),
            alternating,
            {},
            fixed);
    };
    const auto positiveAlternating = solveAlternating(1.0);
    const auto negativeAlternating = solveAlternating(-1.0);
    CHECK(positiveAlternating.mapLatentCoordinate[1] -
              positiveAlternating.mapLatentCoordinate[0] > 0.5);
    CHECK(negativeAlternating.mapLatentCoordinate[1] -
              negativeAlternating.mapLatentCoordinate[0] < -0.5);

    auto unsignedReport = pieces(2);
    FiberTraceConstraint unsignedOther;
    unsignedOther.pieceA = 0;
    unsignedOther.pieceB = 1;
    unsignedOther.parallelScore = 1.0;
    unsignedOther.perpendicularScore = 0.0;
    unsignedOther.parallelWindingDistance = 1.0;
    unsignedReport.constraints.push_back(unsignedOther);
    FiberTraceJointGridWindingConfig unsignedConfig;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(unsignedConfig) =
        config();
    unsignedConfig.fixedPhaseMagnitude = 0.5;
    unsignedConfig.fixedMeasurementScale = 1.0;
    unsignedConfig.stableIterations = 1;
    const auto unsignedSolved =
        solveFiberTraceJointGridWindingBeliefPropagation(
            unsignedReport,
            topology(source, unsignedReport),
            unsignedConfig,
            {},
            fixed);
    REQUIRE(unsignedSolved.factorDiagnostics.size() == 1);
    CHECK(unsignedSolved.factorDiagnostics[0].parallelWindingRetained);
    CHECK(unsignedSolved.factorDiagnostics[0]
              .effectiveParallelWindingWeight > 0.0);
    CHECK_FALSE(unsignedSolved.factorDiagnostics[0].parallelSignPresent);
}

TEST_CASE("Zero magnitude weights retain only enabled dominant hard signs")
{
    const auto source = lines(2);
    const std::vector fixed(
        source.size(), FiberTraceFixedOrientation::Horizontal);
    const auto solve = [&] (
        double parallel,
        double signedDelta,
        bool perpendicularSign,
        bool parallelSign,
        std::optional<double> cutoff = std::nullopt) {
        auto report = pieces(2);
        addMeasured(report, 0, 1, parallel, signedDelta);
        FiberTraceJointGridWindingConfig joint;
        static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) =
            config();
        useZeroClassWeights(joint);
        joint.enforcePerpendicularWindingSign = perpendicularSign;
        joint.enforceParallelWindingSign = parallelSign;
        joint.parallelWindingDistanceCutoff = cutoff;
        joint.fixedPhaseMagnitude = 0.5;
        joint.fixedMeasurementScale = 1.0;
        joint.mixedUnaryCost = 100.0;
        joint.stableIterations = 1;
        return solveFiberTraceJointGridWindingBeliefPropagation(
            report, topology(source, report), joint, {}, fixed);
    };

    const auto disabled = solve(1.0, 1.0, false, false);
    REQUIRE(disabled.factorDiagnostics.size() == 1);
    CHECK_FALSE(disabled.factorDiagnostics[0].hardParallelSign);
    CHECK_FALSE(disabled.factorDiagnostics[0].hardPerpendicularSign);
    CHECK(disabled.factorDiagnostics[0].effectiveParallelWindingWeight == 0.0);
    CHECK(disabled.integerGaugeByPiece[0] !=
          disabled.integerGaugeByPiece[1]);
    CHECK(disabled.incidentSignedConstraints ==
          std::vector<std::size_t>{0, 0});

    const auto parallel = solve(1.0, 1.0, false, true);
    REQUIRE(parallel.factorDiagnostics.size() == 1);
    CHECK(parallel.factorDiagnostics[0].hardParallelSign);
    CHECK_FALSE(parallel.factorDiagnostics[0].hardPerpendicularSign);
    CHECK(parallel.integerGaugeByPiece[0] ==
          parallel.integerGaugeByPiece[1]);
    CHECK(parallel.incidentSignedConstraints ==
          std::vector<std::size_t>{1, 1});
    CHECK(parallel.mapLatentCoordinate[1] >
          parallel.mapLatentCoordinate[0]);

    const auto reversed = solve(1.0, -1.0, false, true);
    CHECK(reversed.mapLatentCoordinate[1] <
          reversed.mapLatentCoordinate[0]);

    const auto perpendicular = solve(0.0, 0.5, true, true);
    REQUIRE(perpendicular.factorDiagnostics.size() == 1);
    CHECK(perpendicular.factorDiagnostics[0].hardPerpendicularSign);
    CHECK_FALSE(perpendicular.factorDiagnostics[0].hardParallelSign);

    const auto sameWinding = solve(1.0, 0.0, true, true);
    REQUIRE(sameWinding.factorDiagnostics.size() == 1);
    CHECK_FALSE(sameWinding.factorDiagnostics[0].hardPerpendicularSign);
    CHECK_FALSE(sameWinding.factorDiagnostics[0].hardParallelSign);

    const auto cutoff = solve(1.0, 2.0, false, true, 1.5);
    REQUIRE(cutoff.factorDiagnostics.size() == 1);
    CHECK_FALSE(cutoff.factorDiagnostics[0].hardParallelSign);
    CHECK(cutoff.integerGaugeByPiece[0] != cutoff.integerGaugeByPiece[1]);
}

TEST_CASE("Contradictory parallel hard signs escape through Defect")
{
    const auto source = lines(3);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 1.0, 1.0);
    addMeasured(report, 1, 2, 1.0, 1.0);
    addMeasured(report, 0, 2, 1.0, -1.0);
    const std::vector fixed(
        source.size(), FiberTraceFixedOrientation::Horizontal);

    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    useZeroClassWeights(joint);
    joint.enforcePerpendicularWindingSign = false;
    joint.enforceParallelWindingSign = true;
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    joint.mixedUnaryCost = 100.0;
    joint.stableIterations = 1;
    const auto solved = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), joint, {}, fixed);
    CHECK(std::count(
        solved.windingValid.begin(),
        solved.windingValid.end(),
        static_cast<unsigned char>(0)) >= 1);
    CHECK(solved.hardSignProjectedDefects >= 1);
}

TEST_CASE("Reference inference admits zero-weight parallel hard signs")
{
    FiberTraceInterleavedWindingReport winding;
    winding.windingValid = {1};
    winding.mapLatentCoordinate = {0.0};
    winding.mapOrientationByPiece = {
        FiberTraceFixedOrientation::Horizontal};
    winding.integerGaugeByPiece = {0};
    winding.componentByPiece = {0};
    winding.measurementScale = 1.0;

    FiberTraceConstraint constraint;
    constraint.pieceA = 0;
    constraint.pieceB = 1;
    constraint.parallelScore = 1.0;
    constraint.signedParallelWindingDelta = 1.0;
    constraint.parallelWindingDistance = 1.0;
    auto zero = config();
    useZeroClassWeights(zero);
    zero.enforcePerpendicularWindingSign = false;
    zero.enforceParallelWindingSign = true;
    const auto observation = makeFiberTraceReferenceWindingObservation(
        constraint, false, 0.0, 0, winding, zero);
    CHECK(observation.rawCoefficient == 0.0);
    CHECK(observation.admittedCoefficient == 0.0);
    CHECK(observation.hardParallelSign);
    CHECK_FALSE(observation.hardPerpendicularSign);
    const auto estimates = inferFiberTraceReferenceRawWindings({&observation, 1});
    REQUIRE(estimates.size() == 1);
    CHECK(estimates[0].observations == 1);
}

TEST_CASE("Reference inference uses the solver confidence and finite sign weight")
{
    FiberTraceInterleavedWindingReport winding;
    winding.windingValid = {1};
    winding.mapLatentCoordinate = {0.0};
    winding.mapOrientationByPiece = {
        FiberTraceFixedOrientation::Horizontal};
    winding.integerGaugeByPiece = {0};
    winding.componentByPiece = {0};
    winding.measurementScale = 1.0;

    FiberTraceConstraint constraint;
    constraint.pieceA = 0;
    constraint.pieceB = 1;
    constraint.parallelScore = 0.75;
    constraint.perpendicularScore = 0.25;
    constraint.signedParallelWindingDelta = 1.0;
    constraint.parallelWindingDistance = 1.0;
    constraint.parallelNormalAlignment = 0.5;
    auto finite = config();
    useZeroClassWeights(finite);
    finite.enforcePerpendicularWindingSign = false;
    finite.enforceParallelWindingSign = true;
    finite.decisionConfidence =
        FiberTraceWindingDecisionConfidence::Linear;
    finite.normalConfidence = FiberTraceWindingNormalConfidence::Cosine;
    finite.finiteSignInfringementCost = 4.0;
    const auto observation = makeFiberTraceReferenceWindingObservation(
        constraint, false, 0.0, 0, winding, finite);
    CHECK(observation.decisionConfidenceMultiplier == doctest::Approx(0.5));
    CHECK(observation.normalConfidenceMultiplier == doctest::Approx(0.5));
    CHECK(observation.parallelSignPenalty == doctest::Approx(1.0));
    CHECK(observation.admittedCoefficient == doctest::Approx(1.0));
    CHECK_FALSE(observation.hardParallelSign);
    const auto estimates = inferFiberTraceReferenceRawWindings({&observation, 1});
    REQUIRE(estimates.size() == 1);
    REQUIRE(observation.inferredReferenceWindingCount == 1);
    CHECK(observation.inferredReferenceWindings[0] == doctest::Approx(1.0));
    CHECK(estimates[0].observations == 1);

    constraint.parallelNormalAlignment = 1.0;
    const auto promoted = makeFiberTraceReferenceWindingObservation(
        constraint, false, 0.0, 0, winding, finite);
    CHECK(promoted.hardParallelSign);
    CHECK(promoted.parallelSignPenalty == 0.0);
}

TEST_CASE("Winding BP fixes an independent crop-central gauge per component")
{
    const auto source = lines(4);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 1.0, 0);
    addMeasured(report, 2, 3, 0.0, -1.0, 1);

    const auto solved = solveFiberTraceWindingBeliefPropagation(
        report, topology(source, report), config());
    CHECK(solved.connectedComponents == 2);
    CHECK(solved.gaugePieces == std::vector<std::size_t>{0, 2});
    CHECK(solved.mapWinding == std::vector<int>{0, 1, 0, -1});
}

TEST_CASE("Winding BP keeps same-trace pieces as linked variables")
{
    auto source = lines(2);
    FiberTraceConstraintReport report;
    report.inputTraces = 2;
    report.pieces.resize(3);
    report.pieces[0] = {0, 0, 0.0, 4.0};
    report.pieces[1] = {0, 1, 2.0, 6.0};
    report.pieces[2] = {1, 0, 0.0, 6.0};

    FiberTraceConstraint continuity;
    continuity.pieceA = 0;
    continuity.pieceB = 1;
    continuity.arcABaseVoxels = 3.0;
    continuity.arcBBaseVoxels = 3.0;
    continuity.pointABaseXYZ = {0.0, 0.0, 0.0};
    continuity.pointBBaseXYZ = continuity.pointABaseXYZ;
    continuity.parallelScore = 1.0;
    continuity.perpendicularScore = 0.0;
    continuity.hardContinuity = true;
    continuity.signedWindingDelta = 0.0;
    report.constraints.push_back(continuity);
    addMeasured(report, 1, 2, 0.0, 1.0);

    const auto solved = solveFiberTraceWindingBeliefPropagation(
        report, topology(source, report), config());
    CHECK(solved.variables == 3);
    CHECK(solved.mapWinding == std::vector<int>{0, 0, 1});
    CHECK(solved.factors == 2);
    CHECK_FALSE(solved.factorDiagnostics.front().selfEdge);
    CHECK(solved.factorDiagnostics.front().effectiveParallelWindingDistance ==
        0.0);
    CHECK_FALSE(solved.factorDiagnostics.front()
                    .effectivePerpendicularSignedDelta.has_value());
}

TEST_CASE("Winding BP rejects incomparable aligned-normal gauges")
{
    const auto source = lines(3);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 1.0, 4);
    addMeasured(report, 1, 2, 0.0, 1.0, 9);
    CHECK_THROWS_WITH_AS(
        solveFiberTraceWindingBeliefPropagation(
            report, topology(source, report), config()),
        doctest::Contains("independent normal alignment gauges"),
        std::invalid_argument);
}

TEST_CASE("Parallel winding BP preserves serial marginals")
{
    constexpr std::size_t edgeCount = 300;
    const auto source = lines(2 * edgeCount);
    auto report = pieces(source.size());
    for (std::size_t edge = 0; edge < edgeCount; ++edge)
        addMeasured(report, 2 * edge, 2 * edge + 1, 0.0, 1.0);
    auto serialConfig = config();
    serialConfig.temperature = 0.1;
    const auto prepared = topology(source, report);
    const auto serial = solveFiberTraceWindingBeliefPropagation(
        report, prepared, serialConfig);
    auto parallelConfig = serialConfig;
    parallelConfig.parallelWorkers = 4;
    const auto parallel = solveFiberTraceWindingBeliefPropagation(
        report, prepared, parallelConfig);
    CHECK(parallel.mapWinding == serial.mapWinding);
    CHECK(parallel.posteriorMeanWinding == serial.posteriorMeanWinding);
    CHECK(parallel.mapProbability == serial.mapProbability);
#ifdef _OPENMP
    CHECK(parallel.effectiveWorkers == 4);
#else
    CHECK(parallel.effectiveWorkers == 1);
#endif
}

TEST_CASE("Interleaved winding calibrates scale-first quantized crossings")
{
    const auto source = lines(3);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 0.32);
    addMeasured(report, 1, 2, 0.0, 0.48);

    FiberTraceInterleavedWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.temperature = 0.05;
    joint.maximumCalibrationIterations = 12;
    std::vector<FiberTraceInterleavedWindingProgress> progress;
    const auto prepared = topology(source, report);
    const auto solved = solveFiberTraceInterleavedWindingBeliefPropagation(
        report,
        prepared,
        orientationBeliefs({
            {0.999, 0.0005, 0.0005},
            {0.0005, 0.0005, 0.999},
            {0.999, 0.0005, 0.0005},
        }),
        joint,
        [&](const FiberTraceInterleavedWindingProgress& event) {
            progress.push_back(event);
        });
    CHECK(solved.status == "converged");
    CHECK(solved.mapWinding == std::vector<int>{0, 0, 1});
    CHECK(solved.classAProbability[0] > 0.99);
    CHECK(solved.classBProbability[1] < 0.1);
    CHECK(solved.classAProbability[2] > 0.9);
    CHECK(solved.phaseMagnitude == doctest::Approx(0.2).epsilon(1.0e-6));
    CHECK(solved.hardSignProjectedDefects == 0);
    CHECK(std::isfinite(solved.measurementScale));
    CHECK(solved.measurementScale >= joint.minimumMeasurementScale);
    CHECK(solved.measurementScale <= joint.maximumMeasurementScale);
    REQUIRE(solved.factorDiagnostics.size() == 2);
    CHECK(solved.factorDiagnostics[0].canonicalSignedDelta == 0.32);
    CHECK(solved.factorDiagnostics[0].effectivePerpendicularSignedDelta == 0.5);
    CHECK(solved.factorDiagnostics[1].canonicalSignedDelta == 0.48);
    CHECK(solved.factorDiagnostics[1].effectivePerpendicularSignedDelta == 0.5);

    REQUIRE_FALSE(progress.empty());
    CHECK(progress.front().phase ==
        FiberTraceInterleavedWindingProgressPhase::Preparing);
    CHECK(progress.back().phase ==
        FiberTraceInterleavedWindingProgressPhase::Complete);
    CHECK(std::count_if(
        progress.begin(),
        progress.end(),
        [](const auto& event) {
            return event.phase ==
                FiberTraceInterleavedWindingProgressPhase::Complete;
        }) == 1);
    CHECK(std::count_if(
        progress.begin(),
        progress.end(),
        [](const auto& event) {
            return event.phase == FiberTraceInterleavedWindingProgressPhase::
                InitializationComplete;
        }) == 4);
    for (const auto& event : progress) {
        CHECK(event.initializationCount == 4);
        CHECK(event.maximumCalibrationIterations ==
            joint.maximumCalibrationIterations);
        CHECK(event.maximumMessageIterations == joint.maximumMessageIterations);
        CHECK(event.elapsedSeconds >= 0.0);
        if (event.phase ==
            FiberTraceInterleavedWindingProgressPhase::MessagePassing) {
            CHECK(event.initialization >= 1);
            CHECK(event.initialization <= event.initializationCount);
            CHECK(event.calibrationIteration >= 1);
            CHECK(event.adaptiveSupportRound >= 1);
            CHECK(event.messageIteration >= 1);
            CHECK(event.messageIteration <= event.maximumMessageIterations);
            CHECK(event.candidateStates > 0);
        }
    }

    const auto withoutProgress =
        solveFiberTraceInterleavedWindingBeliefPropagation(
            report,
            prepared,
            orientationBeliefs({
                {0.999, 0.0005, 0.0005},
                {0.0005, 0.0005, 0.999},
                {0.999, 0.0005, 0.0005},
            }),
            joint);
    CHECK(withoutProgress.mapWinding == solved.mapWinding);
    CHECK(withoutProgress.classAProbability == solved.classAProbability);
    CHECK(withoutProgress.mixedProbability == solved.mixedProbability);
    CHECK(withoutProgress.classBProbability == solved.classBProbability);
    CHECK(withoutProgress.phaseMagnitude == solved.phaseMagnitude);
    CHECK(withoutProgress.measurementScale == solved.measurementScale);
}

TEST_CASE("Alternating fixed-prepass winding has compact direction-defect states")
{
    const auto source = lines(2);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 0.4);
    FiberTraceInterleavedWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.mixedUnaryCost = 5.0;
    joint.maximumCalibrationIterations = 2;
    const std::vector fixed{
        FiberTraceFixedOrientation::Vertical,
        FiberTraceFixedOrientation::Mixed,
    };
    const auto solved = solveFiberTraceInterleavedWindingBeliefPropagation(
        report,
        topology(source, report),
        orientationBeliefs({
            {0.9, 0.05, 0.05},
            {0.9, 0.05, 0.05},
        }),
        joint,
        {},
        fixed);

    CHECK(solved.orientationMode ==
        FiberTraceWindingOrientationMode::FixedPrepass);
    CHECK(solved.defectUnaryCost == doctest::Approx(5.0));
    CHECK(solved.fixedOrientationByPiece == fixed);
    CHECK(solved.classBProbability[0] + solved.mixedProbability[0] ==
        doctest::Approx(1.0));
    CHECK(solved.mixedProbability[1] == doctest::Approx(1.0));
    CHECK(solved.classAProbability[0] == doctest::Approx(0.0));
    CHECK(solved.classAProbability[1] == doctest::Approx(0.0));
    const std::size_t directionalIntegerStates = static_cast<std::size_t>(
        solved.candidateMaximum[0] - solved.candidateMinimum[0] + 1);
    CHECK(solved.totalCandidateStates ==
        directionalIntegerStates + 2);
    CHECK_THROWS_WITH_AS(
        solveFiberTraceInterleavedWindingBeliefPropagation(
            report,
            topology(source, report),
            orientationBeliefs({
                {0.9, 0.05, 0.05},
                {0.9, 0.05, 0.05},
            }),
            joint,
            {},
            std::vector{FiberTraceFixedOrientation::Horizontal}),
        doctest::Contains("do not match"),
        std::invalid_argument);
}

TEST_CASE("Fixed Defect pieces disable incident winding constraints")
{
    const auto source = lines(2);
    auto nearReport = pieces(source.size());
    auto farReport = pieces(source.size());
    addMeasured(nearReport, 0, 1, 0.0, 0.25);
    addMeasured(farReport, 0, 1, 0.0, 8.0);
    const std::vector fixed{
        FiberTraceFixedOrientation::Mixed,
        FiberTraceFixedOrientation::Horizontal,
    };
    const auto beliefs = orientationBeliefs({
        {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0},
    });

    FiberTraceInterleavedWindingConfig alternating;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(alternating) =
        config();
    alternating.mixedUnaryCost = 5.0;
    alternating.maximumCalibrationIterations = 2;
    const auto alternatingNear =
        solveFiberTraceInterleavedWindingBeliefPropagation(
            nearReport,
            topology(source, nearReport),
            beliefs,
            alternating,
            {},
            fixed);
    const auto alternatingFar =
        solveFiberTraceInterleavedWindingBeliefPropagation(
            farReport,
            topology(source, farReport),
            beliefs,
            alternating,
            {},
            fixed);

    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    joint.mixedUnaryCost = 5.0;
    joint.stableIterations = 1;
    const auto jointNear = solveFiberTraceJointGridWindingBeliefPropagation(
        nearReport, topology(source, nearReport), joint, {}, fixed);
    const auto jointFar = solveFiberTraceJointGridWindingBeliefPropagation(
        farReport, topology(source, farReport), joint, {}, fixed);

    for (const auto* solved : {
             &alternatingNear, &alternatingFar, &jointNear, &jointFar}) {
        CHECK(solved->factors == 0);
        CHECK(solved->mixedProbability[0] == doctest::Approx(1.0));
        CHECK(solved->classAProbability[1] + solved->mixedProbability[1] ==
            doctest::Approx(1.0));
        CHECK(solved->mapWinding[0] == 0);
    }
    CHECK(alternatingNear.mapWinding == alternatingFar.mapWinding);
    CHECK(alternatingNear.classAProbability ==
        alternatingFar.classAProbability);
    CHECK(jointNear.mapWinding == jointFar.mapWinding);
    CHECK(jointNear.classAProbability == jointFar.classAProbability);
}

TEST_CASE("Joint-grid winding resolves fixed half-step targets")
{
    const auto source = lines(3);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 0.32);
    addMeasured(report, 1, 2, 0.0, 0.48);

    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.temperature = 0.05;
    joint.mixedUnaryCost = 10.0;
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    joint.stableIterations = 2;
    std::vector<FiberTraceJointGridProgress> progress;
    const auto solved = solveFiberTraceJointGridWindingBeliefPropagation(
        report,
        topology(source, report),
        joint,
        [&](const FiberTraceJointGridProgress& event) {
            progress.push_back(event);
        });

    CHECK(solved.solver == FiberTraceWindingSolver::JointGrid);
    CHECK(solved.status == "converged");
    CHECK(solved.mapWinding == std::vector<int>{0, 1, 0});
    CHECK(solved.windingValid == std::vector<unsigned char>{1, 1, 0});
    CHECK(solved.hardSignProjectedDefects == 1);
    CHECK(solved.mapOrientationByPiece[2] == FiberTraceFixedOrientation::Mixed);
    CHECK(std::isnan(solved.mapLatentCoordinate[2]));
    CHECK(solved.classAProbability[0] > 0.99);
    CHECK(solved.classBProbability[1] > 0.9);
    CHECK(solved.classAProbability[2] > 0.9);
    CHECK(solved.phaseMagnitude == 0.5);
    CHECK(solved.measurementScale == 1.0);
    CHECK(solved.calibrationGridCells == 1);
    REQUIRE(solved.factorDiagnostics.size() == 2);
    CHECK(solved.factorDiagnostics[0].effectivePerpendicularSignedDelta == 0.5);
    CHECK(solved.factorDiagnostics[1].effectivePerpendicularSignedDelta == 0.5);
    REQUIRE_FALSE(progress.empty());
    CHECK(progress.front().phase == FiberTraceJointGridProgressPhase::Preparing);
    CHECK(progress.back().phase == FiberTraceJointGridProgressPhase::Complete);
}

TEST_CASE("Joint-grid exact fixed states preserve signed half-step latents")
{
    const auto source = lines(2);
    auto report = pieces(source.size());

    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    joint.stableIterations = 1;
    const std::vector<FiberTraceFixedWindingState> fixed{
        {true, FiberTraceFixedOrientation::Vertical, 0, 1},
        {true, FiberTraceFixedOrientation::Vertical, 1, -1},
    };
    const auto solved = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), joint, {}, {}, fixed);

    CHECK(solved.windingValid == std::vector<unsigned char>{1, 1});
    CHECK(solved.mapOrientationByPiece ==
        std::vector{
            FiberTraceFixedOrientation::Vertical,
            FiberTraceFixedOrientation::Vertical});
    CHECK(solved.mapWinding == std::vector<int>{0, 1});
    CHECK(solved.mapLatentCoordinate[0] == doctest::Approx(0.5));
    CHECK(solved.mapLatentCoordinate[1] == doctest::Approx(0.5));
    CHECK(solved.mixedProbability == std::vector<double>{0.0, 0.0});
    CHECK(solved.mapProbability == std::vector<double>{1.0, 1.0});
}

TEST_CASE("Joint-grid exact fixed reference drives an ordinary neighbor")
{
    const auto source = lines(2);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 0.5);

    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    joint.mixedUnaryCost = 100.0;
    joint.stableIterations = 1;
    const std::vector<FiberTraceFixedWindingState> fixed{
        {true, FiberTraceFixedOrientation::Horizontal, 0, 1},
        {},
    };
    const auto solved = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), joint, {}, {}, fixed);

    CHECK(solved.windingValid[0] == 1);
    CHECK(solved.mapOrientationByPiece[0] ==
        FiberTraceFixedOrientation::Horizontal);
    CHECK(solved.mapWinding[0] == 0);
    CHECK(solved.mapLatentCoordinate[0] == doctest::Approx(0.0));
    CHECK(solved.mapProbability[0] == doctest::Approx(1.0));
    CHECK(solved.windingValid[1] == 1);
    CHECK(solved.mapLatentCoordinate[1] == doctest::Approx(0.5));
}

TEST_CASE("Joint-grid rejects contradictory exact fixed continuity")
{
    const auto source = lines(1);
    FiberTraceConstraintReport report;
    report.inputTraces = 1;
    report.pieces = {
        {0, 0, 0.0, 4.0},
        {0, 1, 2.0, 6.0},
    };
    FiberTraceConstraint continuity;
    continuity.pieceA = 0;
    continuity.pieceB = 1;
    continuity.parallelScore = 1.0;
    continuity.hardContinuity = true;
    continuity.signedWindingDelta = 0.0;
    continuity.signedParallelWindingDelta = 0.0;
    report.constraints.push_back(continuity);

    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    const std::vector<FiberTraceFixedWindingState> fixed{
        {true, FiberTraceFixedOrientation::Horizontal, 0, 1},
        {true, FiberTraceFixedOrientation::Vertical, 0, 1},
    };
    CHECK_THROWS_AS(
        solveFiberTraceJointGridWindingBeliefPropagation(
            report, topology(source, report), joint, {}, {}, fixed),
        std::invalid_argument);
}

TEST_CASE("Joint-grid empty exact fixed state span preserves baseline")
{
    const auto source = lines(3);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 0.5);
    addMeasured(report, 1, 2, 1.0, 1.0);

    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    joint.stableIterations = 1;
    const auto baseline = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), joint);
    const std::vector<FiberTraceFixedWindingState> empty;
    const auto repeated = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), joint, {}, {}, empty);

    CHECK(repeated.windingValid == baseline.windingValid);
    CHECK(repeated.mapOrientationByPiece == baseline.mapOrientationByPiece);
    CHECK(repeated.mapWinding == baseline.mapWinding);
    CHECK(repeated.mapLatentCoordinate == baseline.mapLatentCoordinate);
    CHECK(repeated.messageResidual == baseline.messageResidual);
}

TEST_CASE("Joint-grid fixed-prepass winding has compact direction-defect states")
{
    const auto source = lines(2);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 0.4);
    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    joint.mixedUnaryCost = 5.0;
    joint.stableIterations = 1;
    const std::vector fixed{
        FiberTraceFixedOrientation::Vertical,
        FiberTraceFixedOrientation::Mixed,
    };
    const auto solved = solveFiberTraceJointGridWindingBeliefPropagation(
        report,
        topology(source, report),
        joint,
        {},
        fixed);

    CHECK(solved.orientationMode ==
        FiberTraceWindingOrientationMode::FixedPrepass);
    CHECK(solved.defectUnaryCost == doctest::Approx(5.0));
    CHECK(solved.fixedOrientationByPiece == fixed);
    CHECK(solved.classBProbability[0] + solved.mixedProbability[0] ==
        doctest::Approx(1.0));
    CHECK(solved.mixedProbability[1] == doctest::Approx(1.0));
    CHECK(solved.classAProbability[0] == doctest::Approx(0.0));
    CHECK(solved.classAProbability[1] == doctest::Approx(0.0));
    const std::size_t directionalIntegerStates = static_cast<std::size_t>(
        solved.candidateMaximum[0] - solved.candidateMinimum[0] + 1);
    const std::size_t compactPieceStates =
        directionalIntegerStates + 2;
    const std::size_t expectedStateAccounting =
        2 * solved.connectedComponents + compactPieceStates;
    CHECK(solved.factors == 0);
    CHECK(solved.totalCandidateStates == expectedStateAccounting);

    const auto jointOrientation =
        solveFiberTraceJointGridWindingBeliefPropagation(
            report, topology(source, report), joint);
    CHECK(jointOrientation.orientationMode ==
        FiberTraceWindingOrientationMode::Joint);
    CHECK(jointOrientation.defectUnaryCost == doctest::Approx(5.0));
    CHECK(jointOrientation.fixedOrientationByPiece.empty());
    CHECK(jointOrientation.totalCandidateStates > solved.totalCandidateStates);
}

TEST_CASE("Fixed-prepass winding can select Defect without flipping direction")
{
    const auto source = lines(2);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 0.5);
    const auto prepared = topology(source, report);

    for (const auto fixedDirection : {
             FiberTraceFixedOrientation::Horizontal,
             FiberTraceFixedOrientation::Vertical}) {
        const std::vector fixed(2, fixedDirection);
        const auto beliefs = fixedDirection ==
                FiberTraceFixedOrientation::Horizontal
            ? orientationBeliefs({{0.9, 0.1, 0.0}, {0.9, 0.1, 0.0}})
            : orientationBeliefs({{0.0, 0.1, 0.9}, {0.0, 0.1, 0.9}});

        FiberTraceInterleavedWindingConfig alternating;
        static_cast<FiberTraceWindingBeliefPropagationConfig&>(alternating) =
            config();
        alternating.temperature = 0.02;
        alternating.mixedUnaryCost = 0.0;
        alternating.maximumCalibrationIterations = 2;
        const auto alternatingSolved =
            solveFiberTraceInterleavedWindingBeliefPropagation(
                report, prepared, beliefs, alternating, {}, fixed);

        FiberTraceJointGridWindingConfig joint;
        static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
        joint.temperature = 0.02;
        joint.mixedUnaryCost = 0.0;
        joint.fixedPhaseMagnitude = 0.5;
        joint.fixedMeasurementScale = 1.0;
        joint.stableIterations = 1;
        const auto jointSolved =
            solveFiberTraceJointGridWindingBeliefPropagation(
                report, prepared, joint, {}, fixed);

        for (const auto* solved : {&alternatingSolved, &jointSolved}) {
            const double maximumDefect = *std::max_element(
                solved->mixedProbability.begin(),
                solved->mixedProbability.end());
            CHECK(maximumDefect > 0.5);
            if (fixedDirection == FiberTraceFixedOrientation::Horizontal) {
                CHECK(solved->classBProbability == std::vector<double>(2, 0.0));
            } else {
                CHECK(solved->classAProbability == std::vector<double>(2, 0.0));
            }
            const std::size_t gauge = solved->gaugePieces.front();
            CHECK(solved->mixedProbability[gauge] > 0.0);
        }
    }
}

TEST_CASE("Joint-grid winding matches exact single-factor marginals")
{
    const auto source = lines(2);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.25, 0.30);

    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.temperature = 0.2;
    joint.messageDamping = 0.5;
    joint.orientationTemperature = 0.35;
    joint.mixedUnaryCost = 0.7;
    joint.logGainStep = std::log(1.25);
    joint.initialGainCells = 3;
    joint.phaseCells = 3;
    joint.maximumGainCells = 3;
    joint.boundaryProbabilityThreshold = 0.99;
    joint.calibrationBoundaryProbabilityThreshold = 0.99;
    joint.calibrationDiscardProbabilityThreshold = 0.0;
    joint.stableIterations = 2;
    const auto solved = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), joint);
    std::array<double, 3> classWeight{};
    std::vector<double> windingWeight(static_cast<std::size_t>(
        solved.candidateMaximum[1] - solved.candidateMinimum[1] + 1));
    std::array<double, 2> signWeight{};
    double totalWeight = 0.0;
    double activeWeight = 0.0;
    double phaseWeight = 0.0;
    double scaleWeight = 0.0;
    const auto windingEnergy = [](double delta, double) {
        return 0.75 * std::abs(delta - 0.50);
    };
    const int minimumGainIndex = static_cast<int>(std::llround(
        std::log(solved.minimumCalibrationGain) / joint.logGainStep));
    const int maximumGainIndex = static_cast<int>(std::llround(
        std::log(solved.maximumCalibrationGain) / joint.logGainStep));
    for (int gainIndex = minimumGainIndex;
         gainIndex <= maximumGainIndex;
         ++gainIndex) {
        const double gain = std::exp(
            static_cast<double>(gainIndex) * joint.logGainStep);
        for (std::size_t phaseIndex = 0;
             phaseIndex < joint.phaseCells;
             ++phaseIndex) {
            const double phase = 0.5 * static_cast<double>(phaseIndex) /
                static_cast<double>(joint.phaseCells - 1);
            for (std::size_t signIndex = 0; signIndex < 2; ++signIndex) {
                const double sign = signIndex == 0 ? 1.0 : -1.0;
                const double defectWeight = std::exp(
                    -joint.mixedUnaryCost / joint.orientationTemperature);
                totalWeight += defectWeight;
                classWeight[1] += defectWeight;
                signWeight[signIndex] += defectWeight;
                phaseWeight += defectWeight * phase;
                scaleWeight += defectWeight / gain;
                for (int winding = solved.candidateMinimum[1];
                     winding <= solved.candidateMaximum[1];
                    ++winding) {
                    for (const std::size_t orientation : {0U, 2U}) {
                        const bool classB = orientation == 2;
                        const double delta = static_cast<double>(winding) +
                            (classB ? sign * phase : 0.0);
                        if (!(delta > 0.0))
                            continue;
                        const double orientationEnergy =
                            classB ? 0.25 : 0.75;
                        const double weight = std::exp(
                            -orientationEnergy /
                                joint.orientationTemperature -
                            windingEnergy(delta, gain) /
                                joint.temperature);
                        totalWeight += weight;
                        activeWeight += weight;
                        classWeight[orientation] += weight;
                        windingWeight[static_cast<std::size_t>(
                            winding - solved.candidateMinimum[1])] += weight;
                        signWeight[signIndex] += weight;
                        phaseWeight += weight * phase;
                        scaleWeight += weight / gain;
                    }
                }
            }
        }
    }
    REQUIRE(totalWeight > 0.0);
    CHECK(solved.classAProbability[1] ==
        doctest::Approx(classWeight[0] / totalWeight).epsilon(1.0e-8));
    CHECK(solved.mixedProbability[1] ==
        doctest::Approx(classWeight[1] / totalWeight).epsilon(1.0e-8));
    CHECK(solved.classBProbability[1] ==
        doctest::Approx(classWeight[2] / totalWeight).epsilon(1.0e-8));
    for (std::size_t piece = 0; piece < solved.classAProbability.size(); ++piece) {
        CHECK(solved.classAProbability[piece] >= 0.0);
        CHECK(solved.classAProbability[piece] <= 1.0);
        CHECK(solved.mixedProbability[piece] >= 0.0);
        CHECK(solved.mixedProbability[piece] <= 1.0);
        CHECK(solved.classBProbability[piece] >= 0.0);
        CHECK(solved.classBProbability[piece] <= 1.0);
        CHECK(solved.classAProbability[piece] +
              solved.mixedProbability[piece] +
              solved.classBProbability[piece] == doctest::Approx(1.0));
    }
    double exactMeanWinding = 0.0;
    for (std::size_t index = 0; index < windingWeight.size(); ++index) {
        exactMeanWinding += windingWeight[index] / activeWeight *
            static_cast<double>(solved.candidateMinimum[1] +
                                static_cast<int>(index));
    }
    CHECK(solved.posteriorMeanWinding[1] ==
        doctest::Approx(exactMeanWinding).epsilon(1.0e-8));
    CHECK(solved.componentPositivePhaseSignProbability[0] ==
        doctest::Approx(signWeight[0] / totalWeight).epsilon(1.0e-8));
    CHECK(solved.calibrationPhaseMean ==
        doctest::Approx(phaseWeight / totalWeight).epsilon(1.0e-8));
    CHECK(solved.calibrationScaleMean ==
        doctest::Approx(scaleWeight / totalWeight).epsilon(1.0e-8));
}

TEST_CASE("Fixed-calibration winding matches exact distance-weighted marginals")
{
    const auto source = lines(2);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 0.0);
    addMeasured(report, 0, 1, 0.0, 0.0);
    addMeasured(report, 0, 1, 0.0, 10.0);

    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.temperature = 0.2;
    joint.messageDamping = 0.5;
    joint.orientationTemperature = 0.35;
    joint.mixedUnaryCost = 0.7;
    joint.fixedPhaseMagnitude = 0.37;
    joint.fixedMeasurementScale = 1.25;
    joint.boundaryProbabilityThreshold = 0.01;
    joint.stableIterations = 2;
    std::vector<FiberTraceJointGridProgress> progress;
    const auto solved = solveFiberTraceJointGridWindingBeliefPropagation(
        report,
        topology(source, report),
        joint,
        [&](const FiberTraceJointGridProgress& event) {
            progress.push_back(event);
        });

    CHECK(solved.calibrationMode == FiberTraceWindingCalibrationMode::Fixed);
    CHECK(solved.phaseMagnitude == 0.37);
    CHECK(solved.calibrationPhaseMean == 0.37);
    CHECK(solved.measurementScale == 1.25);
    CHECK(solved.calibrationScaleMean == 1.25);
    CHECK(solved.calibrationGridCells == 1);
    CHECK(solved.calibrationGridShifts == 0);
    CHECK(solved.calibrationEntropy == 0.0);
    CHECK(solved.lowerGainBoundaryProbability == 0.0);
    CHECK(solved.upperGainBoundaryProbability == 0.0);
    CHECK(solved.expansionRounds >= 1);
    REQUIRE_FALSE(progress.empty());
    for (const auto& event : progress) {
        CHECK(event.calibrationMode ==
            FiberTraceWindingCalibrationMode::Fixed);
        CHECK(event.gainCells == 1);
        CHECK(event.phaseCells == 1);
        CHECK(event.gridShifts == 0);
        CHECK(event.calibrationPosteriorResidual == 0.0);
        CHECK(event.phaseMap == 0.37);
        CHECK(event.phaseMean == 0.37);
        CHECK(event.scaleMap == 1.25);
        CHECK(event.scaleMean == 1.25);
    }

    std::array<double, 3> classWeight{};
    std::vector<double> windingWeight(static_cast<std::size_t>(
        solved.candidateMaximum[1] - solved.candidateMinimum[1] + 1));
    std::array<double, 2> signWeight{};
    double totalWeight = 0.0;
    double activeWeight = 0.0;
    const auto windingEnergy = [](double delta) {
        return 2.0 * std::abs(delta) +
            std::ldexp(1.0, -12) * std::abs(delta - 12.5);
    };
    for (std::size_t signIndex = 0; signIndex < 2; ++signIndex) {
        const double sign = signIndex == 0 ? 1.0 : -1.0;
        const double defectWeight = std::exp(
            -3.0 * joint.mixedUnaryCost / joint.orientationTemperature);
        totalWeight += defectWeight;
        classWeight[1] += defectWeight;
        signWeight[signIndex] += defectWeight;
        for (int winding = solved.candidateMinimum[1];
             winding <= solved.candidateMaximum[1];
            ++winding) {
            for (const std::size_t orientation : {0U, 2U}) {
                const bool classB = orientation == 2;
                const double delta = static_cast<double>(winding) +
                    (classB ? sign * 0.37 : 0.0);
                if (!(delta > 0.0))
                    continue;
                const double orientationEnergy = classB ? 0.0 : 3.0;
                const double weight = std::exp(
                    -orientationEnergy / joint.orientationTemperature -
                    windingEnergy(delta) / joint.temperature);
                totalWeight += weight;
                activeWeight += weight;
                classWeight[orientation] += weight;
                windingWeight[static_cast<std::size_t>(
                    winding - solved.candidateMinimum[1])] += weight;
                signWeight[signIndex] += weight;
            }
        }
    }
    REQUIRE(totalWeight > 0.0);
    CHECK(solved.classAProbability[1] ==
        doctest::Approx(classWeight[0] / totalWeight).epsilon(1.0e-8));
    CHECK(solved.mixedProbability[1] ==
        doctest::Approx(classWeight[1] / totalWeight).epsilon(1.0e-8));
    CHECK(solved.classBProbability[1] ==
        doctest::Approx(classWeight[2] / totalWeight).epsilon(1.0e-8));
    double exactMeanWinding = 0.0;
    for (std::size_t index = 0; index < windingWeight.size(); ++index) {
        exactMeanWinding += windingWeight[index] / activeWeight *
            static_cast<double>(
                solved.candidateMinimum[1] + static_cast<int>(index));
    }
    CHECK(solved.posteriorMeanWinding[1] ==
        doctest::Approx(exactMeanWinding).epsilon(1.0e-8));
    CHECK(solved.componentPositivePhaseSignProbability[0] ==
        doctest::Approx(signWeight[0] / totalWeight).epsilon(1.0e-8));
}

TEST_CASE("Hard winding sign contradictions require a Defect endpoint")
{
    const auto source = lines(3);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 0.4);
    addMeasured(report, 1, 2, 0.0, 0.4);
    addMeasured(report, 0, 2, 0.0, -0.4);
    const std::vector fixed{
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Vertical,
        FiberTraceFixedOrientation::Horizontal,
    };

    FiberTraceInterleavedWindingConfig alternating;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(alternating) =
        config();
    alternating.messageDamping = 0.5;
    alternating.mixedUnaryCost = 100.0;
    alternating.maximumCalibrationIterations = 2;
    const auto alternatingSolved =
        solveFiberTraceInterleavedWindingBeliefPropagation(
            report,
            topology(source, report),
            orientationBeliefs({
                {0.999, 0.0005, 0.0005},
                {0.0005, 0.0005, 0.999},
                {0.999, 0.0005, 0.0005},
            }),
            alternating,
            {},
            fixed);

    FiberTraceJointGridWindingConfig grid;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(grid) = config();
    grid.messageDamping = 0.5;
    grid.mixedUnaryCost = 100.0;
    grid.fixedPhaseMagnitude = 0.5;
    grid.fixedMeasurementScale = 1.0;
    grid.stableIterations = 2;
    const auto gridSolved = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), grid, {}, fixed);

    for (const auto* solved : {&alternatingSolved, &gridSolved}) {
        CHECK(std::count(
            solved->windingValid.begin(),
            solved->windingValid.end(),
            static_cast<unsigned char>(0)) >= 1);
        CHECK(std::isfinite(solved->messageResidual));
    }
}

TEST_CASE("Hard winding sign ignores zero and parallel-only evidence")
{
    const auto source = lines(6);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 0.0);
    addMeasured(report, 2, 3, 1.0, -1.0);
    FiberTraceConstraint unsignedMeasurement;
    unsignedMeasurement.pieceA = 4;
    unsignedMeasurement.pieceB = 5;
    unsignedMeasurement.parallelScore = 0.5;
    unsignedMeasurement.perpendicularScore = 0.5;
    report.constraints.push_back(unsignedMeasurement);
    const std::vector fixed{
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Vertical,
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Vertical,
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Vertical,
    };

    FiberTraceJointGridWindingConfig grid;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(grid) = config();
    grid.fixedPhaseMagnitude = 0.0;
    grid.fixedMeasurementScale = 1.0;
    grid.mixedUnaryCost = 20.0;
    grid.stableIterations = 2;
    const auto solved = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), grid, {}, fixed);

    REQUIRE(solved.windingValid.size() == 6);
    CHECK(static_cast<int>(solved.windingValid[0]) == 1);
    CHECK(static_cast<int>(solved.windingValid[1]) == 1);
    CHECK(static_cast<int>(solved.windingValid[2]) == 1);
    CHECK(static_cast<int>(solved.windingValid[3]) == 1);
    CHECK(static_cast<int>(solved.windingValid[4]) == 0);
    CHECK(static_cast<int>(solved.windingValid[5]) == 0);
    CHECK(solved.hardSignProjectedDefects == 0);
}

TEST_CASE("Joint-grid winding shares fixed half-step targets across components")
{
    const auto source = lines(6);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 0.32);
    addMeasured(report, 1, 2, 0.0, 0.48);
    addMeasured(report, 3, 4, 0.0, 0.32);
    addMeasured(report, 4, 5, 0.0, 0.48);

    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.temperature = 0.05;
    joint.mixedUnaryCost = 100.0;
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    joint.stableIterations = 2;
    const auto solved = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), joint);

    CHECK(solved.connectedComponents == 2);
    CHECK(solved.mapWinding[0] == solved.mapWinding[3]);
    CHECK(solved.mapWinding[1] == solved.mapWinding[4]);
    CHECK(solved.mapWinding[2] == solved.mapWinding[5]);
    CHECK(solved.measurementScale == 1.0);
    REQUIRE(solved.componentPhaseSign.size() == 2);
    REQUIRE(solved.componentPositivePhaseSignProbability.size() == 2);
    for (const double probability :
         solved.componentPositivePhaseSignProbability) {
        CHECK(probability >= 0.0);
        CHECK(probability <= 1.0);
    }
}

TEST_CASE("Joint-grid winding validates adaptive calibration controls")
{
    const auto source = lines(2);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 1.0);
    auto joint = FiberTraceJointGridWindingConfig{};
    joint.initialGainCells = 4;
    CHECK_THROWS_AS(
        solveFiberTraceJointGridWindingBeliefPropagation(
            report, topology(source, report), joint),
        std::invalid_argument);
    joint.initialGainCells = 5;
    joint.phaseCells = 1;
    CHECK_THROWS_AS(
        solveFiberTraceJointGridWindingBeliefPropagation(
            report, topology(source, report), joint),
        std::invalid_argument);
}

TEST_CASE("Joint-grid winding validates fixed calibration controls")
{
    const auto source = lines(2);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 1.0);
    const auto prepared = topology(source, report);

    FiberTraceJointGridWindingConfig joint;
    joint.fixedPhaseMagnitude = 0.5;
    CHECK_THROWS_AS(
        solveFiberTraceJointGridWindingBeliefPropagation(
            report, prepared, joint),
        std::invalid_argument);
    joint.fixedMeasurementScale = 1.0;
    joint.fixedPhaseMagnitude = 0.5001;
    CHECK_THROWS_AS(
        solveFiberTraceJointGridWindingBeliefPropagation(
            report, prepared, joint),
        std::invalid_argument);
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 0.0;
    CHECK_THROWS_AS(
        solveFiberTraceJointGridWindingBeliefPropagation(
            report, prepared, joint),
        std::invalid_argument);
    joint.fixedMeasurementScale = 1.0;
    joint.phaseCells = 0;
    joint.initialGainCells = 0;
    joint.maximumGainCells = 0;
    joint.logGainStep = -1.0;
    CHECK_NOTHROW(solveFiberTraceJointGridWindingBeliefPropagation(
        report, prepared, joint));
}

TEST_CASE("Joint-grid MAP initialization preserves objective and support control")
{
    const auto source = lines(2);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 1.0, 1.0);
    const auto prepared = topology(source, report);
    const std::vector fixed(
        source.size(), FiberTraceFixedOrientation::Horizontal);

    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    joint.enforcePerpendicularWindingSign = false;
    joint.enforceParallelWindingSign = false;
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    joint.stableIterations = 1;

    const auto baseline = solveFiberTraceJointGridWindingBeliefPropagation(
        report, prepared, joint, {}, fixed);
    const auto explicitEmpty =
        solveFiberTraceJointGridWindingBeliefPropagation(
            report, prepared, joint, {}, fixed, {}, {});
    CHECK(explicitEmpty.windingValid == baseline.windingValid);
    CHECK(explicitEmpty.mapWinding == baseline.mapWinding);
    CHECK(explicitEmpty.decodedEnergy == baseline.decodedEnergy);

    const std::vector<FiberTraceJointGridInitialState> farSeed{
        {true, FiberTraceFixedOrientation::Horizontal, -5, 1},
        {true, FiberTraceFixedOrientation::Horizontal, 5, 1},
    };
    auto supportControl = joint;
    supportControl.maximumMessageIterations = 1;
    const auto neutral = solveFiberTraceJointGridWindingBeliefPropagation(
        report,
        prepared,
        supportControl,
        {},
        fixed,
        {},
        farSeed,
        FiberTraceJointGridInitializationMode::SupportOnly);
    const auto conditioned = solveFiberTraceJointGridWindingBeliefPropagation(
        report,
        prepared,
        supportControl,
        {},
        fixed,
        {},
        farSeed,
        FiberTraceJointGridInitializationMode::ConditionedMessages);
    CHECK(neutral.initializedFromState);
    CHECK_FALSE(neutral.conditionedMessageInitialization);
    CHECK(conditioned.initializedFromState);
    CHECK(conditioned.conditionedMessageInitialization);
    CHECK(neutral.totalCandidateStates == conditioned.totalCandidateStates);

    const std::vector<FiberTraceJointGridInitialState> defectSeed{
        {false, FiberTraceFixedOrientation::Mixed, 0, 1},
        {false, FiberTraceFixedOrientation::Mixed, 0, 1},
    };
    const auto escaped = solveFiberTraceJointGridWindingBeliefPropagation(
        report,
        prepared,
        joint,
        {},
        fixed,
        {},
        defectSeed);
    CHECK(escaped.windingValid == std::vector<unsigned char>{1, 1});
}

TEST_CASE("Joint-grid MAP initialization validates seed feasibility")
{
    const auto source = lines(2);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 1.0, 1.0);
    const auto prepared = topology(source, report);
    const std::vector fixed(
        source.size(), FiberTraceFixedOrientation::Horizontal);

    FiberTraceJointGridWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    useZeroClassWeights(joint);
    joint.enforcePerpendicularWindingSign = false;
    joint.enforceParallelWindingSign = true;
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    joint.stableIterations = 1;

    std::vector<FiberTraceJointGridInitialState> invalid{
        {true, FiberTraceFixedOrientation::Horizontal, 0, 1},
        {true, FiberTraceFixedOrientation::Horizontal, -1, 1},
    };
    CHECK_THROWS_WITH_AS(
        solveFiberTraceJointGridWindingBeliefPropagation(
            report, prepared, joint, {}, fixed, {}, invalid),
        doctest::Contains("hard sign"),
        std::invalid_argument);

    invalid[1] = {
        false, FiberTraceFixedOrientation::Mixed, 0, 1};
    CHECK_NOTHROW(solveFiberTraceJointGridWindingBeliefPropagation(
        report, prepared, joint, {}, fixed, {}, invalid));

    invalid.resize(1);
    CHECK_THROWS_WITH_AS(
        solveFiberTraceJointGridWindingBeliefPropagation(
            report, prepared, joint, {}, fixed, {}, invalid),
        doctest::Contains("do not match"),
        std::invalid_argument);
}

TEST_CASE("Interleaved winding retains calibration when signed evidence is absent")
{
    const auto source = lines(1);
    FiberTraceConstraintReport report;
    report.inputTraces = 1;
    report.pieces = {
        {0, 0, 0.0, 4.0},
        {0, 1, 2.0, 6.0},
    };
    FiberTraceConstraint continuity;
    continuity.pieceA = 0;
    continuity.pieceB = 1;
    continuity.arcABaseVoxels = 3.0;
    continuity.arcBBaseVoxels = 3.0;
    continuity.pointABaseXYZ = {0.0, 0.0, 0.0};
    continuity.pointBBaseXYZ = continuity.pointABaseXYZ;
    continuity.parallelScore = 1.0;
    continuity.hardContinuity = true;
    continuity.signedWindingDelta = 0.0;
    report.constraints.push_back(continuity);

    FiberTraceInterleavedWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();
    const auto solved = solveFiberTraceInterleavedWindingBeliefPropagation(
        report,
        topology(source, report),
        orientationBeliefs({
            {0.9, 0.05, 0.05},
            {0.9, 0.05, 0.05},
        }),
        joint);

    CHECK(solved.status == "converged");
    CHECK(solved.mapWinding == std::vector<int>{0, 0});
    CHECK(solved.classAProbability[0] == doctest::Approx(1.0));
    CHECK(solved.classAProbability[1] > 0.8);
    CHECK(solved.measurementScale == doctest::Approx(1.0));
    CHECK(solved.rankDeficientUpdates > 0);
}

TEST_CASE("Joint-grid piece break cost applies once to continuity boundaries")
{
    const auto source = lines(1);
    FiberTraceConstraintReport report;
    report.inputTraces = 1;
    report.pieces = {
        {0, 0, 0.0, 3.0},
        {0, 1, 3.0, 6.0},
    };
    FiberTraceConstraint continuity;
    continuity.pieceA = 0;
    continuity.pieceB = 1;
    continuity.arcABaseVoxels = 3.0;
    continuity.arcBBaseVoxels = 3.0;
    continuity.pointABaseXYZ = {0.0, 0.0, 0.0};
    continuity.pointBBaseXYZ = continuity.pointABaseXYZ;
    continuity.parallelScore = 1.0;
    continuity.hardContinuity = true;
    continuity.signedWindingDelta = 0.0;
    report.constraints.push_back(continuity);

    FiberTraceJointGridWindingConfig config;
    config.fixedPhaseMagnitude = 0.5;
    config.fixedMeasurementScale = 1.0;
    config.mixedUnaryCost = 0.0;
    config.enforceHardSplitContinuity = false;
    config.pieceBreakCost = 0.0;
    config.messageDamping = 1.0;
    config.maximumMessageIterations = 1000;
    const auto baseline = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), config);
    REQUIRE(baseline.windingValid.size() == 2);

    config.pieceBreakCost = 3.0;
    const auto penalized = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), config);
    CHECK(penalized.pieceBreakCost == doctest::Approx(3.0));
    CHECK(penalized.mixedProbability[1] < baseline.mixedProbability[1]);
}

TEST_CASE("Hard split continuity is local and Defect splits source-piece chains")
{
    std::vector<FiberletCropTraceLine> source(1);
    source[0].pointsBaseXYZ = {
        {0.0, 0.0, 0.0},
        {9.0, 0.0, 0.0},
    };
    FiberTraceConstraintReport report;
    report.inputTraces = 1;
    report.pieces = {
        {0, 0, 0.0, 4.0},
        {0, 1, 2.0, 7.0},
        {0, 2, 5.0, 9.0},
    };
    const auto addContinuity = [&](std::size_t a,
                                   std::size_t b,
                                   double arc) {
        FiberTraceConstraint constraint;
        constraint.pieceA = a;
        constraint.pieceB = b;
        constraint.arcABaseVoxels = arc;
        constraint.arcBBaseVoxels = arc;
        constraint.pointABaseXYZ = {arc, 0.0, 0.0};
        constraint.pointBBaseXYZ = constraint.pointABaseXYZ;
        constraint.parallelScore = 1.0;
        constraint.perpendicularScore = 0.0;
        constraint.hardContinuity = true;
        constraint.signedWindingDelta = 0.0;
        constraint.signedParallelWindingDelta = 0.0;
        report.constraints.push_back(constraint);
    };
    addContinuity(0, 1, 3.0);
    addContinuity(1, 2, 6.0);

    FiberTraceBeliefPropagationConfig orientation;
    orientation.cropMinimumBaseXYZ = {-1.0, -1.0, -1.0};
    orientation.cropMaximumBaseXYZ = {10.0, 1.0, 1.0};
    orientation.enforceHardSplitContinuity = true;
    orientation.maximumMessageIterations = 1000;
    const auto orientationSolved = solveFiberTraceMixedSumProduct(
        source, report, orientation);
    const auto fixed = fixedFiberTraceOrientations(orientationSolved);
    REQUIRE(fixed.size() == 3);
    CHECK(fixed[0] == fixed[1]);
    CHECK(fixed[1] == fixed[2]);

    FiberTraceJointGridWindingConfig winding;
    winding.fixedPhaseMagnitude = 0.5;
    winding.fixedMeasurementScale = 1.0;
    winding.stableIterations = 1;
    winding.enforceHardSplitContinuity = true;
    const std::vector split{
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Mixed,
        FiberTraceFixedOrientation::Vertical,
    };
    const auto splitSolved =
        solveFiberTraceJointGridWindingBeliefPropagation(
            report,
            prepareFiberTraceBeliefTopology(
                source,
                report,
                orientation.cropMinimumBaseXYZ,
                orientation.cropMaximumBaseXYZ),
            winding,
            {},
            split);
    CHECK(splitSolved.windingValid ==
          std::vector<unsigned char>{1, 0, 1});
    CHECK(splitSolved.mapOrientationByPiece == split);

    const std::vector conflicting{
        FiberTraceFixedOrientation::Horizontal,
        FiberTraceFixedOrientation::Vertical,
        FiberTraceFixedOrientation::Horizontal,
    };
    const auto projected = solveFiberTraceJointGridWindingBeliefPropagation(
        report,
        prepareFiberTraceBeliefTopology(
            source,
            report,
            orientation.cropMinimumBaseXYZ,
            orientation.cropMaximumBaseXYZ),
        winding,
        {},
        conflicting);
    REQUIRE(projected.windingValid.size() == 3);
    CHECK(std::count(projected.windingValid.begin(),
                     projected.windingValid.end(),
                     static_cast<unsigned char>(0)) >= 1);
    for (const auto& constraint : report.constraints) {
        if (!constraint.hardContinuity ||
            projected.windingValid[constraint.pieceA] == 0 ||
            projected.windingValid[constraint.pieceB] == 0) {
            continue;
        }
        CHECK(projected.mapOrientationByPiece[constraint.pieceA] ==
              projected.mapOrientationByPiece[constraint.pieceB]);
        CHECK(projected.mapWinding[constraint.pieceA] ==
              projected.mapWinding[constraint.pieceB]);
    }

    winding.enforceHardSplitContinuity = false;
    winding.pieceBreakCost = 0.0;
    const auto finite = solveFiberTraceJointGridWindingBeliefPropagation(
        report,
        prepareFiberTraceBeliefTopology(
            source,
            report,
            orientation.cropMinimumBaseXYZ,
            orientation.cropMaximumBaseXYZ),
        winding,
        {},
        conflicting);
    CHECK(std::count(finite.windingValid.begin(),
                     finite.windingValid.end(),
                     static_cast<unsigned char>(1)) > 0);
}

TEST_CASE("Alternating piece break cost uses orientation temperature")
{
    const auto source = lines(1);
    FiberTraceConstraintReport report;
    report.inputTraces = 1;
    report.pieces = {
        {0, 0, 0.0, 3.0},
        {0, 1, 3.0, 6.0},
    };
    FiberTraceConstraint continuity;
    continuity.pieceA = 0;
    continuity.pieceB = 1;
    continuity.arcABaseVoxels = 3.0;
    continuity.arcBBaseVoxels = 3.0;
    continuity.pointABaseXYZ = {0.0, 0.0, 0.0};
    continuity.pointBBaseXYZ = continuity.pointABaseXYZ;
    continuity.parallelScore = 1.0;
    continuity.hardContinuity = true;
    continuity.signedWindingDelta = 0.0;
    report.constraints.push_back(continuity);

    FiberTraceInterleavedWindingConfig config;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(config) =
        ::config();
    config.mixedUnaryCost = 0.0;
    config.enforceHardSplitContinuity = false;
    config.orientationTemperature = 0.5;
    const auto beliefs = orientationBeliefs({
        {0.999, 0.0005, 0.0005},
        {0.0005, 0.999, 0.0005},
    });
    config.pieceBreakCost = 0.0;
    const auto baseline = solveFiberTraceInterleavedWindingBeliefPropagation(
        report,
        topology(source, report),
        beliefs,
        config);
    REQUIRE(baseline.windingValid.size() == 2);
    REQUIRE(baseline.windingValid[0] != baseline.windingValid[1]);

    config.pieceBreakCost = 20.0;
    const auto penalized = solveFiberTraceInterleavedWindingBeliefPropagation(
        report,
        topology(source, report),
        beliefs,
        config);
    CHECK(penalized.pieceBreakCost == doctest::Approx(20.0));
    CHECK(penalized.windingValid[0] == penalized.windingValid[1]);

    config.orientationTemperature = 1.0;
    config.pieceBreakCost = 40.0;
    const auto sameNormalizedPenalty =
        solveFiberTraceInterleavedWindingBeliefPropagation(
            report,
            topology(source, report),
            beliefs,
            config);
    CHECK(sameNormalizedPenalty.windingValid == penalized.windingValid);
    CHECK(sameNormalizedPenalty.classAProbability ==
        penalized.classAProbability);
    CHECK(sameNormalizedPenalty.mixedProbability ==
        penalized.mixedProbability);
    CHECK(sameNormalizedPenalty.classBProbability ==
        penalized.classBProbability);
}

TEST_CASE("Defect-capable winding solvers validate piece break cost")
{
    const auto source = lines(2);
    auto report = pieces(2);
    addMeasured(report, 0, 1, 0.0, 0.5);
    const auto prepared = topology(source, report);

    FiberTraceJointGridWindingConfig joint;
    joint.pieceBreakCost = -1.0;
    CHECK_THROWS_AS(
        solveFiberTraceJointGridWindingBeliefPropagation(
            report, prepared, joint),
        std::invalid_argument);

    FiberTraceInterleavedWindingConfig alternating;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(alternating) =
        config();
    alternating.pieceBreakCost =
        std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS_AS(
        solveFiberTraceInterleavedWindingBeliefPropagation(
            report,
            prepared,
            orientationBeliefs({
                {0.9, 0.05, 0.05},
                {0.05, 0.05, 0.9},
            }),
            alternating),
        std::invalid_argument);
}

TEST_CASE("Interleaved winding rejects malformed orientation beliefs")
{
    const auto source = lines(2);
    auto report = pieces(source.size());
    addMeasured(report, 0, 1, 0.0, 0.5);
    FiberTraceInterleavedWindingConfig joint;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(joint) = config();

    CHECK_THROWS_WITH_AS(
        solveFiberTraceInterleavedWindingBeliefPropagation(
            report,
            topology(source, report),
            orientationBeliefs({
                {1.0, 0.0, 0.0},
                {-0.1, 0.5, 0.6},
            }),
            joint),
        doctest::Contains("orientation beliefs are invalid"),
        std::invalid_argument);
}

TEST_CASE("Interleaved winding preserves serial and parallel marginals")
{
    constexpr std::size_t edgeCount = 260;
    const auto source = lines(2 * edgeCount);
    auto report = pieces(source.size());
    FiberTraceBeliefPropagationReport beliefs;
    for (std::size_t edge = 0; edge < edgeCount; ++edge) {
        addMeasured(report, 2 * edge, 2 * edge + 1, 0.0, 0.4, edge);
        for (const auto& probability : {
                 std::array{0.999, 0.0005, 0.0005},
                 std::array{0.0005, 0.0005, 0.999}}) {
            beliefs.horizontalProbability.push_back(probability[0]);
            beliefs.mixedProbability.push_back(probability[1]);
            beliefs.verticalProbability.push_back(probability[2]);
        }
    }
    const auto prepared = topology(source, report);
    FiberTraceInterleavedWindingConfig serialConfig;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(serialConfig) = config();
    serialConfig.temperature = 0.1;
    const auto serial = solveFiberTraceInterleavedWindingBeliefPropagation(
        report, prepared, beliefs, serialConfig);
    auto parallelConfig = serialConfig;
    parallelConfig.parallelWorkers = 4;
    const auto parallel = solveFiberTraceInterleavedWindingBeliefPropagation(
        report, prepared, beliefs, parallelConfig);

    CHECK(parallel.mapWinding == serial.mapWinding);
    CHECK(parallel.classAProbability == serial.classAProbability);
    CHECK(parallel.mixedProbability == serial.mixedProbability);
    CHECK(parallel.classBProbability == serial.classBProbability);
    CHECK(parallel.phaseMagnitude == serial.phaseMagnitude);
    CHECK(parallel.measurementScale == serial.measurementScale);
#ifdef _OPENMP
    CHECK(parallel.effectiveWorkers == 4);
#else
    CHECK(parallel.effectiveWorkers == 1);
#endif
}

TEST_CASE("Ceres winding recovers fractional state from a fixed source")
{
    const auto source = lines(2);
    auto report = pieces(2);
    addMeasured(report, 0, 1, 0.0, 0.5);
    auto prepared = topology(source, report);
    FiberTraceWindingLeastSquaresConfig leastSquares;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(leastSquares) =
        config();
    leastSquares.defectCost = 100.0;
    leastSquares.orientationExtremenessCost = 10.0;
    leastSquares.maximumIterations = 100;
    const std::array initial{
        FiberTraceWindingLeastSquaresState{1.0, 1.0, 0.0},
        FiberTraceWindingLeastSquaresState{0.0, 1.0, 0.5},
    };
    const std::array fixed{
        FiberTraceWindingLeastSquaresFixedState{true, initial[0]},
        FiberTraceWindingLeastSquaresFixedState{},
    };

    const auto solved = solveFiberTraceWindingLeastSquares(
        report, prepared, leastSquares, initial, fixed);

    REQUIRE(solved.solutionUsable);
    CHECK(solved.factorDiagnostics.size() == 1);
    CHECK(solved.incidentEffectiveConstraints ==
          std::vector<std::size_t>{1, 1});
    CHECK(solved.states[1].horizontalness < 0.05);
    CHECK(solved.states[1].activity > 0.95);
    CHECK(solved.states[1].winding == doctest::Approx(0.5).epsilon(1.0e-4));
}

TEST_CASE("Ceres winding uses fractional activity for contradictory sources")
{
    const auto source = lines(3);
    auto report = pieces(3);
    addMeasured(report, 0, 1, 1.0, 0.0);
    addMeasured(report, 1, 2, 1.0, 0.0);
    auto prepared = topology(source, report);
    FiberTraceWindingLeastSquaresConfig leastSquares;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(leastSquares) =
        config();
    leastSquares.defectCost = 0.01;
    leastSquares.orientationExtremenessCost = 1.0;
    leastSquares.maximumIterations = 100;
    const std::array initial{
        FiberTraceWindingLeastSquaresState{1.0, 1.0, 0.0},
        FiberTraceWindingLeastSquaresState{0.5, 1.0, 0.0},
        FiberTraceWindingLeastSquaresState{0.0, 1.0, 4.0},
    };
    const std::array fixed{
        FiberTraceWindingLeastSquaresFixedState{true, initial[0]},
        FiberTraceWindingLeastSquaresFixedState{},
        FiberTraceWindingLeastSquaresFixedState{true, initial[2]},
    };

    const auto solved = solveFiberTraceWindingLeastSquares(
        report, prepared, leastSquares, initial, fixed);

    REQUIRE(solved.solutionUsable);
    CHECK(solved.states[1].activity < 0.25);
    CHECK(solved.finalCosts.total() < solved.initialCosts.total());
}

TEST_CASE("Ceres winding fixes only the winding gauge deterministically")
{
    const auto source = lines(2);
    auto report = pieces(2);
    addMeasured(report, 0, 1, 0.0, 0.5);
    auto prepared = topology(source, report);
    FiberTraceWindingLeastSquaresConfig leastSquares;
    static_cast<FiberTraceWindingBeliefPropagationConfig&>(leastSquares) =
        config();
    leastSquares.defectCost = 100.0;
    leastSquares.orientationExtremenessCost = 10.0;
    leastSquares.maximumIterations = 100;
    const std::array initial{
        FiberTraceWindingLeastSquaresState{1.0, 1.0, 7.0},
        FiberTraceWindingLeastSquaresState{0.0, 1.0, 7.5},
    };

    const auto solved = solveFiberTraceWindingLeastSquares(
        report, prepared, leastSquares, initial);

    REQUIRE(solved.solutionUsable);
    REQUIRE(solved.gaugePieces == std::vector<std::size_t>{0});
    CHECK(solved.states[0].winding == 0.0);
    CHECK(solved.states[1].winding == doctest::Approx(0.5).epsilon(1.0e-4));
    CHECK(solved.states[1].horizontalness < 0.05);
}

}  // namespace
