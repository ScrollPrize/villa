#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/lasagna/EclMaxflow.hpp"
#include "vc/lasagna/MaxflowGraph.hpp"

#include <cstdint>
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
