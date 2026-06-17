#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/lasagna/EclMaxflow.hpp"
#include "vc/lasagna/LaplaceAmgx.hpp"
#include "vc/lasagna/LaplaceRank.hpp"
#include "vc/lasagna/MaxflowGraph.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

namespace {

vc::lasagna::MaxflowGraph build(
    const std::vector<uint8_t>& passable,
    std::array<size_t, 3> shapeZYX,
    bool storeMetadata = false)
{
    vc::lasagna::MaxflowGraphBuildOptions options;
    options.storeNodeMetadata = storeMetadata;
    return vc::lasagna::buildMaxflowGraphFromPassability(passable, shapeZYX, options);
}

uint64_t edgeCapacity(const vc::lasagna::MaxflowGraph& graph, int32_t a, int32_t b)
{
    uint64_t total = 0;
    for (uint64_t i = graph.nindex[static_cast<size_t>(a)];
         i < graph.nindex[static_cast<size_t>(a) + 1];
         ++i) {
        if (graph.nlist[static_cast<size_t>(i)] == b) {
            total += static_cast<uint64_t>(graph.capacity[static_cast<size_t>(i)]);
        }
    }
    return total;
}

} // namespace

TEST_CASE("all-obstacle volume emits no graph nodes")
{
    const std::vector<uint8_t> passable(8, 0);
    const auto graph = build(passable, {2, 2, 2});

    CHECK(graph.stats.passableVoxels == 0);
    CHECK(graph.stats.graphNodes == 0);
    CHECK(graph.nindex.size() == 1);
    CHECK(graph.nlist.empty());
    CHECK(graph.capacity.empty());
}

TEST_CASE("single passable voxel emits one voxel node")
{
    std::vector<uint8_t> passable(8, 0);
    passable[0] = 1;
    const auto graph = build(passable, {2, 2, 2}, true);

    CHECK(graph.stats.passableVoxels == 1);
    CHECK(graph.stats.leafVoxels == 1);
    CHECK(graph.stats.fullBlocks == 0);
    CHECK(graph.stats.mixedBlocks == 0);
    CHECK(graph.stats.graphNodes == 1);
    CHECK(graph.stats.undirectedEdges == 0);
    REQUIRE(graph.nodeMetadata.size() == 1);
    CHECK(graph.nodeMetadata[0].kind == vc::lasagna::MaxflowNodeKind::Voxel);
}

TEST_CASE("adjacent passable voxels emit two directed unit-capacity arcs")
{
    const std::vector<uint8_t> passable{1, 1};
    const auto graph = build(passable, {1, 1, 2});

    CHECK(graph.stats.passableVoxels == 2);
    CHECK(graph.stats.leafVoxels == 2);
    CHECK(graph.stats.graphNodes == 2);
    CHECK(graph.stats.undirectedEdges == 1);
    CHECK(graph.stats.directedEdges == 2);
    CHECK(edgeCapacity(graph, 0, 1) == 1);
    CHECK(edgeCapacity(graph, 1, 0) == 1);
}

TEST_CASE("full cube remains an uncontracted voxel graph")
{
    const std::vector<uint8_t> passable(8, 1);
    const auto graph = build(passable, {2, 2, 2});

    CHECK(graph.stats.passableVoxels == 8);
    CHECK(graph.stats.graphNodes == 8);
    CHECK(graph.stats.leafVoxels == 8);
    CHECK(graph.stats.fullBlocks == 0);
    CHECK(graph.stats.mixedBlocks == 0);
    CHECK(graph.stats.undirectedEdges == 12);
    CHECK(graph.stats.directedEdges == 24);
    CHECK(graph.stats.contractionRatio == doctest::Approx(1.0));
    CHECK(graph.nodeMetadata.empty());
}

TEST_CASE("mixed cube emits only passable voxel nodes")
{
    std::vector<uint8_t> passable(8, 0);
    passable[0] = 1;
    passable[7] = 1;
    const auto graph = build(passable, {2, 2, 2});

    CHECK(graph.stats.mixedBlocks == 0);
    CHECK(graph.stats.fullBlocks == 0);
    CHECK(graph.stats.leafVoxels == 2);
    CHECK(graph.stats.graphNodes == 2);
    CHECK(graph.stats.undirectedEdges == 0);
}

TEST_CASE("external capacity equals uncontracted face contacts")
{
    const std::vector<uint8_t> passable(4, 1);
    const auto graph = build(passable, {1, 2, 2});

    CHECK(graph.stats.passableVoxels == 4);
    CHECK(graph.stats.graphNodes == 4);
    CHECK(graph.stats.undirectedEdges == 4);
    CHECK(graph.stats.totalExternalCapacity == 4);
    CHECK(graph.stats.uncontractedPassableVoxelGraphUndirectedEdges == 4);
}

TEST_CASE("nearest passable node maps base points to voxel-rank ids")
{
    // Shape z=1, y=2, x=3. Passable nodes in z/y/x order:
    // (0,0,0)->0, (2,0,0)->1, (1,1,0)->2.
    const std::vector<uint8_t> passable{
        1, 0, 1,
        0, 1, 0,
    };
    const vc::lasagna::MaxflowBox3 crop{{10, 20, 30}, {13, 22, 31}};
    const auto exact = vc::lasagna::findNearestPassableNode(
        passable,
        {1, 2, 3},
        crop,
        vc::lasagna::MaxflowDouble3{12.0, 20.0, 30.0},
        1.0);
    CHECK(exact.node == 1);
    CHECK(exact.exact);

    const auto nearest = vc::lasagna::findNearestPassableNode(
        passable,
        {1, 2, 3},
        crop,
        vc::lasagna::MaxflowDouble3{11.0, 20.0, 30.0},
        1.0);
    CHECK(nearest.node == 0);
    CHECK_FALSE(nearest.exact);
}

TEST_CASE("residual min-cut identifies the source-side region and cut frontier")
{
    const auto graph = build(std::vector<uint8_t>{1, 1}, {1, 1, 2});
    REQUIRE(graph.stats.directedEdges == 2);

    std::vector<int> flow(static_cast<size_t>(graph.stats.directedEdges), 0);
    for (uint64_t edge = graph.nindex[0]; edge < graph.nindex[1]; ++edge) {
        if (graph.nlist[static_cast<size_t>(edge)] == 1) {
            flow[static_cast<size_t>(edge)] = 1;
        }
    }

    const auto cut = vc::lasagna::computeMinCutFromFinalFlow(
        graph,
        flow,
        0,
        1,
        true,
        true);
    CHECK(cut.valid);
    CHECK(cut.sourceReachableNodes == 1);
    CHECK(cut.sinkSideNodes == 1);
    CHECK(cut.cutDirectedEdges == 1);
    CHECK(cut.cutCapacity == 1);
    REQUIRE(cut.sourceReachable.size() == 2);
    CHECK(cut.sourceReachable[0] == 1);
    CHECK(cut.sourceReachable[1] == 0);
    REQUIRE(cut.cutEdges.size() == 1);
    CHECK(cut.cutEdges[0].sourceSideNode == 0);
    CHECK(cut.cutEdges[0].sinkSideNode == 1);
    CHECK(cut.cutEdges[0].capacity == 1);
}

TEST_CASE("terminal expansion absorbs adjacent minimum-cut boundaries")
{
    const auto graph = build(std::vector<uint8_t>{1, 1, 1}, {1, 1, 3});

    const auto sourceExpansion = vc::lasagna::expandTerminalRegionAcrossMinCutBoundaries(
        graph,
        0,
        2,
        1,
        vc::lasagna::EclTerminalSide::Source,
        10,
        true);
    CHECK(sourceExpansion.valid);
    CHECK(sourceExpansion.regionNodes == 2);
    CHECK(sourceExpansion.absorbedNodes == 1);
    CHECK(sourceExpansion.finalBoundaryCapacity == 1);
    CHECK(sourceExpansion.finalBoundaryIsMinCut);
    CHECK(sourceExpansion.touchedOppositeTerminal);
    REQUIRE(sourceExpansion.region.size() == 3);
    CHECK(sourceExpansion.region[0] == 1);
    CHECK(sourceExpansion.region[1] == 1);
    CHECK(sourceExpansion.region[2] == 0);

    const auto sinkExpansion = vc::lasagna::expandTerminalRegionAcrossMinCutBoundaries(
        graph,
        2,
        0,
        1,
        vc::lasagna::EclTerminalSide::Sink,
        10,
        true);
    CHECK(sinkExpansion.valid);
    CHECK(sinkExpansion.regionNodes == 2);
    CHECK(sinkExpansion.absorbedNodes == 1);
    CHECK(sinkExpansion.finalBoundaryCapacity == 1);
    CHECK(sinkExpansion.finalBoundaryIsMinCut);
    CHECK(sinkExpansion.touchedOppositeTerminal);
    REQUIRE(sinkExpansion.region.size() == 3);
    CHECK(sinkExpansion.region[0] == 0);
    CHECK(sinkExpansion.region[1] == 1);
    CHECK(sinkExpansion.region[2] == 1);
}

TEST_CASE("ECL maxflow runs on a tiny graph when CUDA is available")
{
    if (!vc::lasagna::eclMaxflowAvailable()) {
        MESSAGE("CUDA device unavailable; skipping ECL smoke path");
        return;
    }

    const auto graph = build(std::vector<uint8_t>{1, 1}, {1, 1, 2});
    const auto result = vc::lasagna::runEclMaxflow(graph, 0, 1, 1);
    CHECK(result.maxFlow == 1);
    CHECK(result.runs == 1);
    CHECK(result.minCut.valid);
    CHECK(result.minCut.cutCapacity == result.maxFlow);
}

TEST_CASE("screened Laplace assembly eliminates a chain endpoint source")
{
    const auto graph = build(std::vector<uint8_t>{1, 1, 1}, {1, 1, 3});
    const auto system = vc::lasagna::assembleScreenedLaplaceSystem(
        graph,
        0,
        0.25);

    CHECK(system.lambda == doctest::Approx(0.25));
    CHECK(system.sourceRegion.nodes == std::vector<int32_t>{0});
    CHECK(system.nodeToUnknown == std::vector<int32_t>{-1, 0, 1});
    CHECK(system.unknownToNode == std::vector<int32_t>{1, 2});
    CHECK(system.rowOffsets == std::vector<int32_t>{0, 2, 4});
    CHECK(system.columns == std::vector<int32_t>{0, 1, 1, 0});
    REQUIRE(system.values.size() == 4);
    CHECK(system.values[0] == doctest::Approx(2.25));
    CHECK(system.values[1] == doctest::Approx(-1.0));
    CHECK(system.values[2] == doctest::Approx(1.25));
    CHECK(system.values[3] == doctest::Approx(-1.0));
    REQUIRE(system.rhs.size() == 2);
    CHECK(system.rhs[0] == doctest::Approx(1.0));
    CHECK(system.rhs[1] == doctest::Approx(0.0));
}

TEST_CASE("screened Laplace reports value one for sink inside source region")
{
    const auto graph = build(std::vector<uint8_t>{1, 1, 1}, {1, 1, 3});
    const auto system = vc::lasagna::assembleScreenedLaplaceSystem(
        graph,
        0,
        0.25,
        1);

    CHECK(system.sourceRegion.nodes == std::vector<int32_t>{0, 1});
    CHECK(vc::lasagna::laplaceValueForNode(system, std::vector<double>{0.4}, 1) ==
          doctest::Approx(1.0));
    CHECK(vc::lasagna::laplaceValueForNode(system, std::vector<double>{0.4}, 2) ==
          doctest::Approx(0.4));
}

TEST_CASE("screened Laplace source depth clamps BFS nodes")
{
    const auto graph = build(std::vector<uint8_t>{1, 1, 1, 1}, {1, 1, 4});
    const auto region = vc::lasagna::buildLaplaceSourceRegion(graph, 1, 1);

    CHECK(region.nodes == std::vector<int32_t>{0, 1, 2});
    REQUIRE(region.mask.size() == 4);
    CHECK(region.mask[0] == 1);
    CHECK(region.mask[1] == 1);
    CHECK(region.mask[2] == 1);
    CHECK(region.mask[3] == 0);
}

TEST_CASE("screened Laplace rejects invalid source and sink nodes")
{
    const auto graph = build(std::vector<uint8_t>{1, 1}, {1, 1, 2});

    CHECK_THROWS_AS(
        vc::lasagna::assembleScreenedLaplaceSystem(graph, -1, 0.25),
        std::runtime_error);
    const auto system = vc::lasagna::assembleScreenedLaplaceSystem(graph, 0, 0.25);
    CHECK_THROWS_AS(
        vc::lasagna::laplaceValueForNode(system, std::vector<double>{0.5}, 3),
        std::runtime_error);
}

TEST_CASE("Laplace snap ranking chooses the smaller side as targets")
{
    const auto roles = vc::lasagna::selectLaplaceRankRoles(1, 2);
    CHECK(roles.solveSide == vc::lasagna::LaplaceRankSide::SideB);
    CHECK(roles.targetSide == vc::lasagna::LaplaceRankSide::SideA);
    CHECK(roles.solveCount == 2);
    CHECK(roles.targetCount == 1);

    const auto tie = vc::lasagna::selectLaplaceRankRoles(2, 2);
    CHECK(tie.solveSide == vc::lasagna::LaplaceRankSide::SideA);
    CHECK(tie.targetSide == vc::lasagna::LaplaceRankSide::SideB);
}

TEST_CASE("Laplace snap ranking accepts a pair-set by max absolute matrix value")
{
    vc::lasagna::LaplaceRankEvaluation weak;
    weak.solved = true;
    weak.values = {1.0e-9, -2.0e-9, 9.0e-7};
    for (double value : weak.values) {
        weak.maxAbsValue = std::max(weak.maxAbsValue, std::abs(value));
    }
    CHECK_FALSE(vc::lasagna::laplaceRankAccepted(weak, 1.0e-6));

    vc::lasagna::LaplaceRankEvaluation accepted = weak;
    accepted.values.push_back(-2.5e-6);
    accepted.maxAbsValue = 2.5e-6;
    CHECK(vc::lasagna::laplaceRankAccepted(accepted, 1.0e-6));
}

TEST_CASE("Laplace snap ranking rejects client-owned lambda search options")
{
    const nlohmann::json request = {
        {"manifest", "missing.lasagna.json"},
        {"jobs", nlohmann::json::array()},
        {"options", {
            {"threshold", 110},
            {"margin_base_voxels", 1000},
            {"source_depth", 0},
            {"adaptive_start_lambda", 1.0 / 2048.0},
        }},
    };
    CHECK_THROWS_WITH_AS(
        vc::lasagna::rankSnapPairsJson(request),
        doctest::Contains("adaptive_start_lambda is server-owned"),
        std::invalid_argument);
}

TEST_CASE("Laplace snap ranking accepts max parallel jobs option")
{
    const nlohmann::json request = {
        {"manifest", "missing.lasagna.json"},
        {"jobs", nlohmann::json::array()},
        {"options", {
            {"threshold", 110},
            {"margin_base_voxels", 1000},
            {"source_depth", 0},
            {"max_parallel_jobs", 1},
        }},
    };
    const auto response = vc::lasagna::rankSnapPairsJson(request);
    CHECK(response["results"].is_array());
    CHECK(response["results"].empty());
    CHECK(response["options"]["max_parallel_jobs"] == 1);
}

TEST_CASE("Laplace snap ranking selects one synchronized lambda per pair-set")
{
    std::vector<double> probes;
    const auto selected = vc::lasagna::selectAdaptiveLaplaceRankLambda(
        1.0 / 2048.0,
        1.0e-6,
        [&](double lambda) {
            probes.push_back(lambda);
            vc::lasagna::LaplaceRankEvaluation evaluation;
            evaluation.lambda = lambda;
            evaluation.solved = true;
            evaluation.status = "success";
            evaluation.values = {5.0e-10 / lambda, 1.0e-9 / lambda};
            evaluation.maxAbsValue = std::max(std::abs(evaluation.values[0]), std::abs(evaluation.values[1]));
            return evaluation;
        });

    REQUIRE(probes.size() > 1);
    CHECK(vc::lasagna::laplaceRankAccepted(selected, 1.0e-6));
    CHECK(selected.values.size() == 2);
    CHECK(selected.maxAbsValue >= 1.0e-6);
}

TEST_CASE("AMGX screened Laplace solves a tiny chain when enabled")
{
    if (!vc::lasagna::laplaceAmgxAvailable()) {
        MESSAGE("AMGX unavailable; skipping screened Laplace smoke path");
        return;
    }

    const auto graph = build(std::vector<uint8_t>{1, 1, 1, 1}, {1, 1, 4});
    vc::lasagna::LaplaceAmgxOptions weakLeak;
    weakLeak.lambda = 1.0 / (64.0 * 64.0);
    vc::lasagna::LaplaceAmgxResult weak;
    try {
        weak = vc::lasagna::solveScreenedLaplaceAmgx(graph, 0, weakLeak);
    } catch (const std::exception& e) {
        const std::string message = e.what();
        if (message.find("AMGX_resources_create_simple") != std::string::npos ||
            message.find("no CUDA-capable device") != std::string::npos) {
            MESSAGE("AMGX runtime unavailable; skipping screened Laplace smoke path: " << e.what());
            return;
        }
        throw;
    }
    REQUIRE(weak.success);
    REQUIRE(weak.valuesByNode.size() == 4);
    CHECK(weak.valuesByNode[0] == doctest::Approx(1.0));
    CHECK(weak.valuesByNode[1] > weak.valuesByNode[2]);
    CHECK(weak.valuesByNode[2] > weak.valuesByNode[3]);

    vc::lasagna::LaplaceAmgxOptions strongLeak;
    strongLeak.lambda = 1.0 / (8.0 * 8.0);
    const auto strong = vc::lasagna::solveScreenedLaplaceAmgx(graph, 0, strongLeak);
    REQUIRE(strong.success);
    CHECK(strong.valuesByNode[3] < weak.valuesByNode[3]);
}
