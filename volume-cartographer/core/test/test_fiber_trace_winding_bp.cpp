#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberTraceBeliefPropagation.hpp"
#include "vc/fiber_tracer/FiberTraceWindingBeliefPropagation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

FiberTraceWindingBeliefPropagationConfig config()
{
    FiberTraceWindingBeliefPropagationConfig result;
    result.temperature = 0.25;
    result.messageDamping = 1.0;
    result.messageResidualTolerance = 1.0e-12;
    result.maximumMessageIterations = 1000;
    return result;
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
        REQUIRE(solved.factorDiagnostics[index].effectiveSignedDelta);
        CHECK(*solved.factorDiagnostics[index].effectiveSignedDelta ==
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
    for (const auto& diagnostic : raw.factorDiagnostics)
        CHECK(diagnostic.effectiveSignedDelta == diagnostic.canonicalSignedDelta);
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

    const std::array multipliers{1.0, 0.5, 0.25, 0.125};
    REQUIRE(solved.factorDiagnostics.size() == multipliers.size());
    for (std::size_t index = 0; index < multipliers.size(); ++index) {
        const auto& diagnostic = solved.factorDiagnostics[index];
        CHECK(diagnostic.parallelScore == 0.25);
        CHECK(diagnostic.perpendicularScore == 0.75);
        CHECK(diagnostic.windingWeightMultiplier == multipliers[index]);
        CHECK(diagnostic.effectiveParallelWindingWeight ==
            doctest::Approx(0.25 * multipliers[index]));
        CHECK(diagnostic.effectivePerpendicularWindingWeight ==
            doctest::Approx(0.75 * multipliers[index]));
    }

    const auto raw = solveFiberTraceWindingBeliefPropagation(
        report, topology(source, report), config());
    REQUIRE(raw.factorDiagnostics.size() == multipliers.size());
    for (const auto& diagnostic : raw.factorDiagnostics) {
        CHECK(diagnostic.windingWeightMultiplier == 1.0);
        CHECK(diagnostic.effectiveParallelWindingWeight == 0.25);
        CHECK(diagnostic.effectivePerpendicularWindingWeight == 0.75);
    }
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
    CHECK(solved.factorDiagnostics.front().effectiveSignedDelta == 0.0);
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

TEST_CASE("Interleaved winding calibrates complementary fractional crossings")
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
    CHECK(solved.mapWinding == std::vector<int>{0, 1, 2});
    CHECK(solved.classAProbability[0] > 0.99);
    CHECK(solved.classBProbability[1] > 0.9);
    CHECK(solved.classAProbability[2] > 0.9);
    CHECK(solved.phaseMagnitude == doctest::Approx(0.0).epsilon(1.0e-6));
    CHECK(solved.hardSignProjectedDefects == 0);
    CHECK(std::isfinite(solved.measurementScale));
    CHECK(solved.measurementScale >= joint.minimumMeasurementScale);
    CHECK(solved.measurementScale <= joint.maximumMeasurementScale);
    REQUIRE(solved.factorDiagnostics.size() == 2);
    CHECK(solved.factorDiagnostics[0].canonicalSignedDelta == 0.32);
    CHECK(solved.factorDiagnostics[0].effectiveSignedDelta == 0.5);
    CHECK(solved.factorDiagnostics[1].canonicalSignedDelta == 0.48);
    CHECK(solved.factorDiagnostics[1].effectiveSignedDelta == 0.5);

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
    CHECK(solved.fixedOrientationByPiece == fixed);
    CHECK(solved.classBProbability[0] > 0.99);
    CHECK(solved.mixedProbability[0] < 0.01);
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
        CHECK(solved->windingValid == std::vector<unsigned char>{0, 1});
        CHECK(solved->mixedProbability[0] == doctest::Approx(1.0));
        CHECK(solved->classAProbability[1] > 0.99);
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
    CHECK(solved.classAProbability[0] > 0.99);
    CHECK(solved.classBProbability[1] > 0.9);
    CHECK(solved.classAProbability[2] > 0.9);
    CHECK(solved.phaseMagnitude == 0.5);
    CHECK(solved.measurementScale == 1.0);
    CHECK(solved.calibrationGridCells == 1);
    REQUIRE(solved.factorDiagnostics.size() == 2);
    CHECK(solved.factorDiagnostics[0].effectiveSignedDelta == 0.5);
    CHECK(solved.factorDiagnostics[1].effectiveSignedDelta == 0.5);
    REQUIRE_FALSE(progress.empty());
    CHECK(progress.front().phase == FiberTraceJointGridProgressPhase::Preparing);
    CHECK(progress.back().phase == FiberTraceJointGridProgressPhase::Complete);
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
    CHECK(solved.fixedOrientationByPiece == fixed);
    CHECK(solved.classBProbability[0] > 0.99);
    CHECK(solved.mixedProbability[0] < 0.01);
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
    const auto windingEnergy = [](double delta, double gain) {
        return 0.25 * std::abs(delta) +
            0.75 * std::abs(gain * delta - 0.50);
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
                        if (!(gain * delta > 0.0))
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
    constexpr double gain = 0.8;
    const auto windingEnergy = [](double delta) {
        return 2.0 * std::abs(gain * delta) +
            std::ldexp(1.0, -9) * std::abs(gain * delta - 9.5);
    };
    for (std::size_t signIndex = 0; signIndex < 2; ++signIndex) {
        const double sign = signIndex == 0 ? 1.0 : -1.0;
        const double defectWeight = std::exp(
            -joint.mixedUnaryCost / joint.orientationTemperature);
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
                if (!(gain * delta > 0.0))
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

    CHECK(solved.windingValid ==
        std::vector<unsigned char>{1, 1, 1, 1, 1, 1});
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
    joint.mixedUnaryCost = 1.0;
    joint.fixedPhaseMagnitude = 0.5;
    joint.fixedMeasurementScale = 1.0;
    joint.stableIterations = 2;
    const auto solved = solveFiberTraceJointGridWindingBeliefPropagation(
        report, topology(source, report), joint);

    CHECK(solved.connectedComponents == 2);
    CHECK(solved.mapWinding == std::vector<int>{0, 0, 1, 0, 0, 1});
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

}  // namespace
