#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_tracer/BinaryBeliefPropagation.hpp"
#include "vc/fiber_tracer/FiberTraceConstraints.hpp"
#include "vc/fiber_tracer/LasagnaNormalAlignment.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace
{

using namespace vc::fiber_tracer;

TEST_CASE("Binary pairwise sum-product supports either fixed state")
{
    const std::vector<BinaryPairwiseFactor> factors{{0, 1, 0.0, 1.0}};
    BinaryBeliefPropagationConfig config;
    config.temperature = 0.5;
    config.messageDamping = 1.0;

    auto fixed = std::vector<BinaryBeliefState>{BinaryBeliefState::Zero, BinaryBeliefState::Free};
    const auto zero = solveBinaryPairwiseSumProduct(2, factors, fixed, config);
    CHECK(zero.messageConverged);
    CHECK(zero.probabilityOne[0] == 0.0);
    CHECK(std::isinf(zero.logOdds[0]));
    CHECK(zero.logOdds[0] < 0.0);
    CHECK(zero.probabilityOne[1] == doctest::Approx(1.0 / (1.0 + std::exp(2.0))).epsilon(1.0e-12));

    fixed[0] = BinaryBeliefState::One;
    const auto one = solveBinaryPairwiseSumProduct(2, factors, fixed, config);
    CHECK(one.probabilityOne[0] == 1.0);
    CHECK(std::isinf(one.logOdds[0]));
    CHECK(one.logOdds[0] > 0.0);
    CHECK(one.probabilityOne[1] == doctest::Approx(1.0 / (1.0 + std::exp(-2.0))).epsilon(1.0e-12));
}

TEST_CASE("Binary pairwise sum-product rejects malformed problems")
{
    const std::vector<BinaryBeliefState> fixed{BinaryBeliefState::Free};
    CHECK_THROWS_AS(solveBinaryPairwiseSumProduct(0, {}, {}, {}), std::invalid_argument);
    CHECK_THROWS_AS(solveBinaryPairwiseSumProduct(1, std::vector<BinaryPairwiseFactor>{{0, 1, 0.0, 1.0}}, fixed, {}), std::invalid_argument);
    CHECK_THROWS_AS(solveBinaryPairwiseSumProduct(1, std::vector<BinaryPairwiseFactor>{{0, 0, 0.0, 1.0}}, fixed, {}), std::invalid_argument);
    CHECK_THROWS_AS(solveBinaryPairwiseSumProduct(1, {}, {}, {}), std::invalid_argument);
}

TEST_CASE("Parallel binary BP preserves the serial report")
{
    constexpr std::size_t factorCount = 40'000;
    std::vector<BinaryPairwiseFactor> factors;
    factors.reserve(factorCount);
    std::vector<BinaryBeliefState> fixed(factorCount * 2, BinaryBeliefState::Free);
    for (std::size_t index = 0; index < factorCount; ++index) {
        const double same = static_cast<double>(index % 17) / 17.0;
        const double different = 1.0 - same;
        factors.push_back({2 * index, 2 * index + 1, same, different});
        fixed[2 * index] = (index % 2 == 0) ? BinaryBeliefState::Zero : BinaryBeliefState::One;
    }
    BinaryBeliefPropagationConfig serialConfig;
    serialConfig.messageDamping = 1.0;
    serialConfig.messageResidualTolerance = 0.0;
    serialConfig.maximumMessageIterations = 8;
    const auto serial = solveBinaryPairwiseSumProduct(fixed.size(), factors, fixed, serialConfig);

    auto parallelConfig = serialConfig;
    parallelConfig.parallelWorkers = 4;
    const auto parallel = solveBinaryPairwiseSumProduct(fixed.size(), factors, fixed, parallelConfig);

    CHECK(parallel.messageIterations == serial.messageIterations);
    CHECK(parallel.messageResidual == serial.messageResidual);
    CHECK(parallel.messageConverged == serial.messageConverged);
    CHECK(parallel.logOdds == serial.logOdds);
    CHECK(parallel.probabilityOne == serial.probabilityOne);
#ifdef _OPENMP
    CHECK(parallel.effectiveWorkers > 1);
#else
    CHECK(parallel.effectiveWorkers == 1);
#endif
}

TEST_CASE("Lasagna normal factors preserve signed-dot evidence")
{
    const auto parallel = makeLasagnaNormalAlignmentFactor(0, 1, {1.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F});
    REQUIRE(parallel.has_value());
    CHECK(parallel->sameCost == doctest::Approx(0.0));
    CHECK(parallel->differentCost == doctest::Approx(1.0));

    const auto opposite = makeLasagnaNormalAlignmentFactor(0, 1, {1.0F, 0.0F, 0.0F}, {-1.0F, 0.0F, 0.0F});
    REQUIRE(opposite.has_value());
    CHECK(opposite->sameCost == doctest::Approx(1.0));
    CHECK(opposite->differentCost == doctest::Approx(0.0));

    CHECK_FALSE(makeLasagnaNormalAlignmentFactor(0, 1, {1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}).has_value());
}

TEST_CASE("Lasagna normal lattice is globally anchored and de-duplicates neighbors")
{
    const auto first = makeLasagnaNormalLattice({0.1, 0.1, 0.1}, {4.1, 4.1, 4.1}, 2.0);
    const auto overlapping = makeLasagnaNormalLattice({1.9, 1.9, 1.9}, {6.1, 6.1, 6.1}, 2.0);
    REQUIRE(first.positionsBaseXYZ.size() == 8);
    REQUIRE(overlapping.positionsBaseXYZ.size() == 27);
    CHECK(first.positionsBaseXYZ.front() == (cv::Vec3f{2.0F, 2.0F, 2.0F}));
    CHECK(first.positionsBaseXYZ.back() == (cv::Vec3f{4.0F, 4.0F, 4.0F}));
    CHECK(overlapping.positionsBaseXYZ.front() == first.positionsBaseXYZ.front());

    const std::vector<cv::Vec3f> normals(first.positionsBaseXYZ.size(), cv::Vec3f{1.0F, 0.0F, 0.0F});
    std::vector<std::size_t> nodes(normals.size());
    for (std::size_t index = 0; index < nodes.size(); ++index)
        nodes[index] = index;
    const auto factors = makeLasagnaNormalLatticeFactors(first, nodes, normals, 1);
    CHECK(factors.size() == 28);
    for (const auto& factor : factors)
        CHECK(factor.a < factor.b);

    nodes[3] = std::numeric_limits<std::size_t>::max();
    std::vector<cv::Vec3f> compactNormals(7, {1.0F, 0.0F, 0.0F});
    for (std::size_t index = 4; index < nodes.size(); ++index)
        --nodes[index];
    const auto withHole = makeLasagnaNormalLatticeFactors(first, nodes, compactNormals, 1);
    CHECK(withHole.size() == 21);
}

TEST_CASE("Lasagna normal alignment resolves alternating signs per component")
{
    const std::vector<cv::Vec3f> normals{
        {1.0F, 0.0F, 0.0F},
        {-1.0F, 0.0F, 0.0F},
        {1.0F, 0.0F, 0.0F},
        {0.0F, 1.0F, 0.0F},
    };
    std::vector<BinaryPairwiseFactor> factors;
    factors.push_back(*makeLasagnaNormalAlignmentFactor(0, 1, normals[0], normals[1]));
    factors.push_back(*makeLasagnaNormalAlignmentFactor(1, 2, normals[1], normals[2]));
    LasagnaNormalAlignmentConfig config;
    config.beliefPropagation.temperature = 0.1;
    config.beliefPropagation.messageDamping = 1.0;
    const auto report = alignLasagnaNormalSamples(normals, factors, config);

    CHECK(report.connectedComponents == 2);
    CHECK(report.componentByNode == std::vector<std::size_t>{0, 0, 0, 1});
    CHECK(report.isolatedSamples == 1);
    CHECK(report.fixedStates[0] == BinaryBeliefState::Zero);
    CHECK(report.fixedStates[3] == BinaryBeliefState::Zero);
    CHECK(report.flippedSamples == 1);
    for (std::size_t index = 0; index < 3; ++index) {
        CHECK(report.alignedNormals[index][0] == doctest::Approx(1.0));
        CHECK(report.alignedNormals[index][1] == doctest::Approx(0.0));
    }
    CHECK(report.alignedNormals[3][1] == doctest::Approx(1.0));
}

TEST_CASE("Aligned normal field orients winding without changing its magnitude")
{
    LasagnaNormalAlignmentField field;
    field.lattice = makeLasagnaNormalLattice(
        {-1.0, -1.0, -1.0}, {2.0, 2.0, 2.0}, 1.0);
    field.nodeByLatticeSample.resize(field.lattice.positionsBaseXYZ.size());
    for (std::size_t node = 0; node < field.nodeByLatticeSample.size(); ++node)
        field.nodeByLatticeSample[node] = node;
    field.positionsBaseXYZ = field.lattice.positionsBaseXYZ;
    field.alignment.alignedNormals.assign(
        field.nodeByLatticeSample.size(), cv::Vec3f{1.0F, 0.0F, 0.0F});
    field.alignment.componentByNode.assign(
        field.nodeByLatticeSample.size(), 0);

    FiberTraceConstraintReport report;
    FiberTraceConstraint forward;
    forward.pieceA = 0;
    forward.pieceB = 1;
    forward.pointABaseXYZ = {0.0, 0.0, 0.0};
    forward.pointBBaseXYZ = {1.0, 0.0, 0.0};
    forward.parallelScore = 0.0;
    forward.perpendicularScore = 1.0;
    forward.windingDistance = 0.75;
    report.constraints.push_back(forward);
    auto reverse = forward;
    std::swap(reverse.pointABaseXYZ, reverse.pointBBaseXYZ);
    report.constraints.push_back(reverse);

    orientFiberTraceConstraintWindings(report, field);
    CHECK(report.constraints[0].windingDistance == doctest::Approx(0.75));
    CHECK(report.constraints[0].signedWindingDelta == doctest::Approx(0.75));
    CHECK(report.constraints[1].windingDistance == doctest::Approx(0.75));
    CHECK(report.constraints[1].signedWindingDelta == doctest::Approx(-0.75));
    CHECK(report.signedWindingConstraints == 2);
    CHECK(report.skippedSignedWindingConstraints == 0);

    const auto endpoint = field.nearest({1.0, 0.0, 0.0});
    REQUIRE(endpoint.has_value());
    field.alignment.componentByNode[endpoint->node] = 1;
    orientFiberTraceConstraintWindings(report, field);
    CHECK_FALSE(report.constraints[0].signedWindingDelta.has_value());
    CHECK_FALSE(report.constraints[1].signedWindingDelta.has_value());
    CHECK(report.skippedSignedWindingConstraints == 2);
}

TEST_CASE("Normal glyph OBJ writes crossed bases and directed strokes")
{
    const std::filesystem::path path = "test_lasagna_normal_alignment.obj";
    const std::vector<cv::Vec3f> positions{{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}};
    const std::vector<cv::Vec3f> normals{{1.0F, 0.0F, 0.0F}, {0.0F, -1.0F, 0.0F}};
    writeNormalGlyphObj(path, positions, normals, {0.5, 2.0});

    std::ifstream input(path);
    REQUIRE(input.good());
    std::string line;
    std::size_t vertices = 0;
    std::size_t segments = 0;
    bool bases = false;
    bool directions = false;
    while (std::getline(input, line)) {
        vertices += line.starts_with("v ") ? 1 : 0;
        segments += line.starts_with("l ") ? 1 : 0;
        bases = bases || line == "g normal_bases";
        directions = directions || line == "g normal_directions";
    }
    CHECK(vertices == 12);
    CHECK(segments == 6);
    CHECK(bases);
    CHECK(directions);
    std::filesystem::remove(path);
}

}  // namespace
