#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/FiberTraceBeliefPropagation.hpp"
#include "vc/fiber_tracer/FiberTraceWindingBeliefPropagation.hpp"

#include <algorithm>
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

}  // namespace
