// Scaffold winding solve: link assembly, exact LEMON solve, approved-subgraph
// contraction and the residual cycle analysis.
//
// The graphs here are built by hand rather than read from a store: the fibers
// sit in the +x/+y quadrant so no polyline crosses the seam ray, every turn
// and connector transport is 0, and a link's whole-fiber target is therefore
// exactly the step it asserts. That keeps the expected windings readable.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_graph/ScaffoldRunMonitor.hpp"
#include "vc/fiber_graph/ScaffoldSolveRuns.hpp"
#include "vc/fiber_graph/ScaffoldWindingSolve.hpp"

#include <vc_object_store/Uuid.hpp>
#include <vc_object_store/objects/ObjectStore.hpp>
#include <vc_object_store/objects/StoreReader.hpp>

#include <algorithm>
#include <filesystem>
#include <string>

using namespace vc::fiber_graph;

namespace
{

struct Builder {
    ScaffoldGraph graph;

    Builder()
    {
        graph.handedness = 1;
        graph.volume = "test-volume";
        graph.umbilicus = {{0., 0., -50.}, {0., 0., 50.}};
        graph.offsets.push_back(0);
    }

    std::size_t fiber(ScaffoldFamily family, double x)
    {
        ScaffoldFiber entry;
        entry.uuid = vc::objects::util::newUuid();
        entry.family = family;
        entry.pointFirst = graph.offsets.back();
        const std::vector<vc::geometry::Point> points{{x, 10., 0.}, {x, 10., 5.}, {x, 10., 10.}};
        entry.pointCount = std::int64_t(points.size());
        for (const auto& point : points) graph.linePoints.insert(graph.linePoints.end(), point.begin(), point.end());
        entry.content.controlIds = {1, 2, 3};
        entry.content.controlPoints = points;
        entry.content.controlLineIndex = {0, 1, 2};
        entry.content.annotations.assign(points.size(), nlohmann::json::object());
        graph.offsets.push_back(entry.pointFirst + entry.pointCount);
        graph.index.emplace(entry.uuid, graph.fibers.size());
        graph.fibers.push_back(std::move(entry));
        return graph.fibers.size() - 1;
    }

    // Anchors on the polylines at height `height`, so their projection
    // distance is zero and the arc equals the height.
    void link(std::size_t a, std::size_t b, bool adjacent = false, const std::string& state = "approved",
              double heightA = 5., double heightB = 5., bool automatic = false)
    {
        ScaffoldLink entry;
        entry.uuid = vc::objects::util::newUuid();
        entry.a = std::int64_t(a);
        entry.b = std::int64_t(b);
        entry.aUuid = graph.fibers[a].uuid;
        entry.bUuid = graph.fibers[b].uuid;
        entry.adjacent = adjacent;
        entry.autoLink = automatic;
        entry.approvalState = state;
        entry.anchorA = vc::geometry::Point{graph.fibers[a].content.controlPoints.front()[0], 10., heightA};
        entry.anchorB = vc::geometry::Point{graph.fibers[b].content.controlPoints.front()[0], 10., heightB};
        graph.links.push_back(std::move(entry));
    }
};

ScaffoldSolveOptions pendingAndApproved()
{
    ScaffoldSolveOptions options;
    options.links.approvalStates = {"approved", "pending"};
    return options;
}

} // namespace

TEST_CASE("an adjacent link asserts one winding between an H and a V fiber")
{
    CHECK(adjacentStep(ScaffoldFamily::Horizontal, ScaffoldFamily::Vertical) == -1);
    CHECK(adjacentStep(ScaffoldFamily::Vertical, ScaffoldFamily::Horizontal) == 1);
    CHECK_FALSE(adjacentStep(ScaffoldFamily::Horizontal, ScaffoldFamily::Horizontal).has_value());
    CHECK_FALSE(adjacentStep(ScaffoldFamily::Vertical, ScaffoldFamily::Vertical).has_value());
}

TEST_CASE("identical relations are aggregated into one arc")
{
    const auto arcs = aggregateWindingArcs({{0, 1, 0, 1.}, {0, 2, 1, 2.}, {0, 1, 0, 0.5}, {0, 1, 1, 3.}});
    REQUIRE(arcs.size() == 3);
    CHECK(arcs[0].a == 0);
    CHECK(arcs[0].b == 1);
    CHECK(arcs[0].target == 0);
    CHECK(arcs[0].weight == doctest::Approx(1.5));
    CHECK(arcs[1].target == 1);
    CHECK(arcs[1].weight == doctest::Approx(3.));
    CHECK(arcs[2].b == 2);
    CHECK(arcs[2].weight == doctest::Approx(2.));
}

TEST_CASE("link anchors project onto their polyline")
{
    Builder builder;
    builder.fiber(ScaffoldFamily::Horizontal, 100.);
    const ScaffoldGeometry geometry(builder.graph);
    const auto on = geometry.project(0, {100., 10., 2.5});
    CHECK(on.arc == doctest::Approx(2.5));
    CHECK(on.distance == doctest::Approx(0.));
    const auto beside = geometry.project(0, {103., 10., 7.});
    CHECK(beside.arc == doctest::Approx(7.));
    CHECK(beside.distance == doctest::Approx(3.));
    CHECK(geometry.controlArc(0, 2) == doctest::Approx(10.));
}

TEST_CASE("a consistent scaffold is solved exactly")
{
    Builder builder;
    const auto h = builder.fiber(ScaffoldFamily::Horizontal, 100.);
    const auto v = builder.fiber(ScaffoldFamily::Vertical, 110.);
    const auto other = builder.fiber(ScaffoldFamily::Horizontal, 120.);
    builder.link(h, other);              // same winding
    builder.link(h, v, /*adjacent=*/true); // the V fiber one winding inside
    const ScaffoldGeometry geometry(builder.graph);
    const auto result = solveScaffoldWinding(builder.graph, geometry);
    CHECK(result.linkReport.used == 2);
    CHECK(result.linkReport.ordinary == 1);
    CHECK(result.linkReport.adjacent == 1);
    CHECK(result.linkReport.maxAnchorDistance == doctest::Approx(0.));
    CHECK(result.arcs.size() == 2);
    CHECK(result.objective == doctest::Approx(0.));
    CHECK(result.maxResidual == doctest::Approx(0.));
    CHECK(result.violatedConstraints == 0);
    CHECK(result.components == 1);
    CHECK(result.residuals.empty());
    // The gauge is the component's minimum: the inner V fiber is winding 0.
    CHECK(result.winding[v] == 0);
    CHECK(result.winding[h] == 1);
    CHECK(result.winding[other] == 1);
}

TEST_CASE("a fiber without links keeps its own component and gauge")
{
    Builder builder;
    const auto a = builder.fiber(ScaffoldFamily::Horizontal, 100.);
    const auto b = builder.fiber(ScaffoldFamily::Horizontal, 110.);
    const auto alone = builder.fiber(ScaffoldFamily::Vertical, 120.);
    builder.link(a, b);
    const ScaffoldGeometry geometry(builder.graph);
    const auto result = solveScaffoldWinding(builder.graph, geometry);
    CHECK(result.components == 2);
    CHECK(result.isolatedFibers == 1);
    CHECK(result.winding[alone] == 0);
    CHECK(result.component[a] == result.component[b]);
    CHECK(result.component[alone] != result.component[a]);
}

TEST_CASE("a frustrated triangle names the cycle its violation lies on")
{
    Builder builder;
    const auto h = builder.fiber(ScaffoldFamily::Horizontal, 100.);
    const auto middle = builder.fiber(ScaffoldFamily::Horizontal, 110.);
    const auto v = builder.fiber(ScaffoldFamily::Vertical, 120.);
    builder.link(h, middle, false, "approved", 2., 2.);
    builder.link(middle, v, false, "approved", 8., 8.);
    builder.link(h, v, true, "approved", 5., 5.); // contradicts the two above
    const ScaffoldGeometry geometry(builder.graph);
    ScaffoldSolveOptions options;
    options.fixApprovedSubgraphs = false;
    const auto result = solveScaffoldWinding(builder.graph, geometry, options);
    CHECK(result.linkReport.used == 3);
    CHECK(result.objective == doctest::Approx(1.));
    CHECK(result.maxResidual == doctest::Approx(1.));
    CHECK(result.violatedConstraints == 1);
    CHECK(result.violatedArcs == 1);
    REQUIRE(result.residuals.size() == 1);
    const auto& residual = result.residuals.front();
    CHECK(residual.residual == doctest::Approx(1.));
    CHECK(residual.cost == doctest::Approx(1.));
    // The cycle is the triangle: three constraints, one stretch per fiber.
    CHECK(residual.cycleHops == 3);
    CHECK(residual.redundancy == 1);
    CHECK(residual.joint == doctest::Approx(1. / 3.));
    REQUIRE(residual.pieces.size() == 3);
    std::vector<std::int64_t> fibers;
    for (const auto& piece : residual.pieces) fibers.push_back(piece.fiber);
    std::sort(fibers.begin(), fibers.end());
    CHECK(fibers == std::vector<std::int64_t>{std::int64_t(h), std::int64_t(middle), std::int64_t(v)});
    // Each stretch is bracketed by the controls a piece would be opened at.
    for (const auto& piece : residual.pieces) {
        REQUIRE(piece.controlFrom >= 0);
        CHECK(piece.controlTo > piece.controlFrom);
        CHECK(geometry.controlArc(std::size_t(piece.fiber), std::size_t(piece.controlFrom)) <=
              std::min(piece.arcFrom, piece.arcTo) + 1e-9);
        CHECK(geometry.controlArc(std::size_t(piece.fiber), std::size_t(piece.controlTo)) >=
              std::max(piece.arcFrom, piece.arcTo) - 1e-9);
    }
    // Every stretch is accused, and the cost is distributed over them once.
    double score = 0;
    for (const auto& piece : result.pieces) score += piece.score;
    CHECK(score == doctest::Approx(residual.cost));
    CHECK(result.pieces.size() == 3);
    for (const auto& piece : result.pieces) CHECK(piece.cycleHops == 3);
}

TEST_CASE("a monitor reports the solve's stages and takes the run back")
{
    Builder builder;
    const auto h = builder.fiber(ScaffoldFamily::Horizontal, 100.);
    const auto middle = builder.fiber(ScaffoldFamily::Horizontal, 110.);
    const auto v = builder.fiber(ScaffoldFamily::Vertical, 120.);
    builder.link(h, middle, false, "approved", 2., 2.);
    builder.link(middle, v, false, "approved", 8., 8.);
    builder.link(h, v, true, "approved", 5., 5.);
    const ScaffoldGeometry geometry(builder.graph);
    ScaffoldRunMonitor monitor;
    ScaffoldSolveOptions options;
    options.fixApprovedSubgraphs = false;
    options.monitor = &monitor;
    const auto result = solveScaffoldWinding(builder.graph, geometry, options);
    CHECK(result.violatedConstraints == 1);
    // The last stage a run passes through is the residual analysis, and it
    // counted off exactly the violated constraints it searched cycles for.
    const auto progress = monitor.progress();
    CHECK(progress.stage == "residual cycles");
    CHECK(progress.total == 1);
    CHECK(progress.done == 1);

    // Cancelled before it starts: nothing is solved, and the stage it died in
    // says so rather than reporting a failure.
    monitor.cancel();
    CHECK(monitor.cancelled());
    CHECK_THROWS_AS(solveScaffoldWinding(builder.graph, geometry, options), ScaffoldCancelled);
}

TEST_CASE("the shortest cycle wins over a longer one")
{
    Builder builder;
    const auto a = builder.fiber(ScaffoldFamily::Horizontal, 100.);
    const auto b = builder.fiber(ScaffoldFamily::Vertical, 110.);
    const auto c = builder.fiber(ScaffoldFamily::Horizontal, 120.);
    const auto d = builder.fiber(ScaffoldFamily::Vertical, 130.);
    const auto e = builder.fiber(ScaffoldFamily::Horizontal, 140.);
    // A->B violated by the adjacent assertion, with two edge-disjoint
    // two-hop alternatives (A-C-B, A-E-B) and a longer one through D.
    builder.link(a, c);
    builder.link(c, b);
    builder.link(a, e);
    builder.link(e, b);
    builder.link(c, d);
    builder.link(d, b);
    builder.link(a, b, true);
    const ScaffoldGeometry geometry(builder.graph);
    ScaffoldSolveOptions options;
    options.fixApprovedSubgraphs = false;
    const auto result = solveScaffoldWinding(builder.graph, geometry, options);
    REQUIRE(result.residuals.size() == 1);
    const auto& residual = result.residuals.front();
    CHECK(result.constraints[std::size_t(residual.constraint)].step == -1);
    CHECK(residual.cycleHops == 3);
    CHECK(residual.pieces.size() == 3);
    // Two edge-disjoint alternatives exist between A and B.
    CHECK(residual.redundancy == 2);
}

TEST_CASE("approved subgraphs are held rigid")
{
    Builder builder;
    const auto a = builder.fiber(ScaffoldFamily::Horizontal, 100.);
    const auto b = builder.fiber(ScaffoldFamily::Horizontal, 110.);
    const auto c = builder.fiber(ScaffoldFamily::Vertical, 120.);
    builder.link(a, b, false, "approved");
    builder.link(b, c, false, "approved");
    builder.link(a, c, true, "pending"); // contradicts the approved pair
    const ScaffoldGeometry geometry(builder.graph);
    const auto result = solveScaffoldWinding(builder.graph, geometry, pendingAndApproved());
    CHECK(result.approvedSubgraphs == 1);
    CHECK(result.contractedFibers == 2);
    CHECK(result.objective == doctest::Approx(1.));
    REQUIRE(result.residuals.size() == 1);
    // The pending link is the one left violated, never an approved one.
    CHECK(result.constraints[std::size_t(result.residuals.front().constraint)].approved == false);
    CHECK(result.winding[a] == result.winding[b]);
    CHECK(result.winding[c] == result.winding[a]);
}

TEST_CASE("contradicting approved constraints stop the run")
{
    Builder builder;
    const auto a = builder.fiber(ScaffoldFamily::Horizontal, 100.);
    const auto b = builder.fiber(ScaffoldFamily::Vertical, 110.);
    builder.link(a, b, false, "approved");
    builder.link(a, b, true, "approved"); // the same pair, one winding apart
    const ScaffoldGeometry geometry(builder.graph);
    CHECK_THROWS_AS(solveScaffoldWinding(builder.graph, geometry), std::invalid_argument);
}

TEST_CASE("links are filtered by approval state, automation and family")
{
    Builder builder;
    const auto a = builder.fiber(ScaffoldFamily::Horizontal, 100.);
    const auto b = builder.fiber(ScaffoldFamily::Horizontal, 110.);
    const auto c = builder.fiber(ScaffoldFamily::Vertical, 120.);
    builder.link(a, b, false, "approved");
    builder.link(a, c, false, "rejected");
    builder.link(b, c, false, "approved", 5., 5., /*automatic=*/true);
    builder.link(a, b, true, "approved"); // adjacent between two H fibers
    const ScaffoldGeometry geometry(builder.graph);
    ScaffoldLinkReport report;
    const auto constraints = scaffoldLinkConstraints(builder.graph, geometry, ScaffoldLinkArcOptions{}, report);
    CHECK(report.links == 4);
    CHECK(report.used == 1);
    CHECK(report.skippedApprovalState == 1);
    CHECK(report.skippedAutomatic == 1);
    CHECK(report.skippedAdjacentSameFamily == 1);
    REQUIRE(constraints.size() == 1);
    CHECK(constraints.front().target == 0);

    ScaffoldLinkArcOptions withAuto;
    withAuto.approvalStates = {"approved", "pending"};
    withAuto.includeAuto = true;
    const auto more = scaffoldLinkConstraints(builder.graph, geometry, withAuto, report);
    CHECK(report.used == 2);
    CHECK(report.skippedAutomatic == 0);
    CHECK(more.size() == 2);
}

TEST_CASE("an anchor off its polyline fails the run")
{
    Builder builder;
    const auto a = builder.fiber(ScaffoldFamily::Horizontal, 100.);
    const auto b = builder.fiber(ScaffoldFamily::Horizontal, 110.);
    builder.link(a, b);
    builder.graph.links.back().anchorB = vc::geometry::Point{200., 10., 5.};
    const ScaffoldGeometry geometry(builder.graph);
    ScaffoldLinkReport report;
    CHECK_THROWS_AS(scaffoldLinkConstraints(builder.graph, geometry, ScaffoldLinkArcOptions{}, report),
                    std::invalid_argument);
}

TEST_CASE("a link without anchors constrains nothing")
{
    Builder builder;
    const auto a = builder.fiber(ScaffoldFamily::Horizontal, 100.);
    const auto b = builder.fiber(ScaffoldFamily::Horizontal, 110.);
    builder.link(a, b);
    builder.graph.links.back().anchorA.reset();
    const ScaffoldGeometry geometry(builder.graph);
    ScaffoldLinkReport report;
    const auto constraints = scaffoldLinkConstraints(builder.graph, geometry, ScaffoldLinkArcOptions{}, report);
    CHECK(constraints.empty());
    CHECK(report.skippedMissingAnchor == 1);
    CHECK_THROWS_AS(solveScaffoldWinding(builder.graph, geometry), std::invalid_argument);
}

TEST_CASE("a solve round-trips through the store's extension tables")
{
    namespace fs = std::filesystem;
    using namespace vc::objects;
    Builder builder;
    const auto h = builder.fiber(ScaffoldFamily::Horizontal, 100.);
    const auto middle = builder.fiber(ScaffoldFamily::Horizontal, 110.);
    const auto v = builder.fiber(ScaffoldFamily::Vertical, 120.);
    builder.link(h, middle, false, "approved", 2., 2.);
    builder.link(middle, v, false, "approved", 8., 8.);
    builder.link(h, v, true, "approved", 5., 5.);
    const ScaffoldGeometry geometry(builder.graph);
    ScaffoldSolveOptions options;
    options.fixApprovedSubgraphs = false;
    const auto result = solveScaffoldWinding(builder.graph, geometry, options);
    REQUIRE(result.residuals.size() == 1);

    ScaffoldRunRecord record;
    record.runUuid = util::newUuid();
    record.input = {util::newUuid(), util::newUuid(), 7};
    const auto rows = scaffoldSolveRunRows(builder.graph, result, options, record);
    CHECK(rows.tables.size() == rows.rows.size());

    const fs::path dir = fs::temp_directory_path() / ("vc-scaffold-run-" + util::newUuid());
    fs::create_directories(dir);
    const auto path = dir / "objects.vcobj";
    OpenOptions write;
    write.author = "scaffold-run-test";
    {
        auto store = ObjectStore::create(path, write);
        auto txn = store.write();
        for (const auto& table : rows.tables) store.createExtensionTable(txn, table);
        for (const auto& [table, values] : rows.rows) store.appendExtensionRows(txn, table, values);
        txn.commit();
    }
    OpenOptions read;
    read.readOnly = true;
    auto store = ObjectStore::open(path, read);
    const auto txn = store.read();
    const LocalStoreReader reader(store, txn);

    const auto runs = readScaffoldRuns(reader);
    REQUIRE(runs.size() == 1);
    CHECK(runs.front().runUuid == record.runUuid);
    CHECK(runs.front().stage == "winding");
    CHECK(runs.front().method == "scaffold_lemon_network_simplex");
    CHECK(runs.front().counts.at("fibers") == 3);
    CHECK(runs.front().counts.at("violated_constraints") == 1);
    CHECK(runs.front().metrics.at("weighted_l1") == doctest::Approx(result.objective));
    CHECK(runs.front().residuals == 1);
    CHECK(runs.front().settingsJson.find("\"fix_approved_subgraphs\":false") != std::string::npos);
    // The store state the run read comes back, so a dock can tell a run that
    // is behind the store from a current one.
    CHECK(runs.front().input == record.input);

    const auto stored = readScaffoldResiduals(reader, record.runUuid);
    REQUIRE(stored.size() == 1);
    CHECK(stored.front().residual == doctest::Approx(1.));
    CHECK(stored.front().cycleHops.value_or(0) == 3);
    CHECK(stored.front().redundancy.value_or(0) == 1);
    REQUIRE(stored.front().pieces.size() == 3);
    double score = 0;
    for (const auto& piece : stored.front().pieces) {
        CHECK(piece.arcFrom <= piece.arcTo);
        CHECK(builder.graph.indexOf(piece.fiberUuid).has_value());
        score += piece.score;
    }
    CHECK(score == doctest::Approx(stored.front().residual * result.constraints.front().weight));

    const auto windings = readScaffoldWindings(reader, record.runUuid);
    REQUIRE(windings.size() == 3);
    for (std::size_t i = 0; i < builder.graph.fibers.size(); ++i) {
        const auto& entry = windings.at(builder.graph.fibers[i].uuid);
        CHECK(entry.winding == result.winding[i]);
        CHECK(entry.component == result.component[i]);
        CHECK(entry.family.value_or(-1) == std::int64_t(builder.graph.fibers[i].family));
    }
    CHECK(readScaffoldResiduals(reader, util::newUuid()).empty());
    std::error_code ignored;
    fs::remove_all(dir, ignored);
}
