// Scaffold normals and measured constraints: the anchors that fix a fiber's
// normal gauge, the crossings of the annotation's own polylines, the sign
// solve over those crossings and the integer relations they assert.
//
// The scaffolds here are built by hand around a straight umbilicus on the z
// axis: H fibers are arcs (or one spiral) in a z plane, V fibers are straight
// lines along z. Their normals are therefore radial, which makes every
// expectation below a statement about a direction one can read off the
// geometry.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/fiber_graph/ScaffoldConstraints.hpp"

#include <vc_object_store/geometry/FiberWindingGraph.hpp>

#include <vc_object_store/Uuid.hpp>

#include <cmath>
#include <memory>
#include <numbers>
#include <string>

using namespace vc::fiber_graph;
using vc::geometry::Point;

namespace
{

constexpr double kPi = std::numbers::pi;

double radians(double degrees) { return degrees * kPi / 180.; }

Point polar(double radius, double degrees, double z)
{
    return {radius * std::cos(radians(degrees)), radius * std::sin(radians(degrees)), z};
}

// The outward radial unit direction at a point, the axis being the z axis.
Point outward(const Point& point)
{
    const double radius = std::hypot(point[0], point[1]);
    return {point[0] / radius, point[1] / radius, 0.};
}

double dot(const Point& a, const Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

std::vector<Point> arcLine(double radius, double fromDegrees, double toDegrees, double z, std::size_t steps)
{
    std::vector<Point> points;
    for (std::size_t i = 0; i <= steps; ++i)
        points.push_back(polar(radius, fromDegrees + (toDegrees - fromDegrees) * double(i) / double(steps), z));
    return points;
}

std::vector<Point> spiralLine(double fromRadius, double toRadius, double fromDegrees, double toDegrees, double z,
                              std::size_t steps)
{
    std::vector<Point> points;
    for (std::size_t i = 0; i <= steps; ++i) {
        const double t = double(i) / double(steps);
        points.push_back(polar(fromRadius + (toRadius - fromRadius) * t,
                               fromDegrees + (toDegrees - fromDegrees) * t, z));
    }
    return points;
}

std::vector<Point> verticalLine(double radius, double degrees, double fromZ, double toZ, std::size_t steps)
{
    std::vector<Point> points;
    for (std::size_t i = 0; i <= steps; ++i)
        points.push_back(polar(radius, degrees, fromZ + (toZ - fromZ) * double(i) / double(steps)));
    return points;
}

struct Scaffold {
    ScaffoldGraph graph;

    Scaffold()
    {
        graph.handedness = 1;
        graph.volume = "test-volume";
        graph.umbilicus = {{0., 0., -400.}, {0., 0., 400.}};
        graph.offsets.push_back(0);
    }

    // Every tenth line point is a control point, plus the last one.
    std::size_t fiber(ScaffoldFamily family, const std::vector<Point>& points)
    {
        ScaffoldFiber entry;
        entry.uuid = vc::objects::util::newUuid();
        entry.name = "fiber " + std::to_string(graph.fibers.size());
        entry.family = family;
        entry.version = 1;
        entry.pointFirst = graph.offsets.back();
        entry.pointCount = std::int64_t(points.size());
        for (const auto& point : points) graph.linePoints.insert(graph.linePoints.end(), point.begin(), point.end());
        for (std::size_t i = 0; i < points.size(); i += 10) {
            entry.content.controlIds.push_back(std::int64_t(i) + 1);
            entry.content.controlPoints.push_back(points[i]);
            entry.content.controlLineIndex.push_back(std::int64_t(i));
        }
        if (entry.content.controlLineIndex.back() != std::int64_t(points.size()) - 1) {
            entry.content.controlIds.push_back(std::int64_t(points.size()));
            entry.content.controlPoints.push_back(points.back());
            entry.content.controlLineIndex.push_back(std::int64_t(points.size()) - 1);
        }
        entry.content.annotations.assign(entry.content.controlPoints.size(), nlohmann::json::object());
        graph.offsets.push_back(entry.pointFirst + entry.pointCount);
        graph.index.emplace(entry.uuid, graph.fibers.size());
        graph.fibers.push_back(std::move(entry));
        return graph.fibers.size() - 1;
    }

    // A manual display normal on one control point. The format states them
    // outward, so this is the outward radial direction there.
    void manualOutwardNormal(std::size_t fiber, std::size_t control)
    {
        auto& entry = graph.fibers[fiber];
        const auto direction = outward(entry.content.controlPoints.at(control));
        entry.content.annotations.at(control) = {{"display_normal", {direction[0], direction[1], direction[2]}},
                                                {"display_normal_source", "manual"}};
    }

    void link(std::size_t a, std::size_t b, const Point& anchorA, const Point& anchorB, bool adjacent)
    {
        ScaffoldLink entry;
        entry.uuid = vc::objects::util::newUuid();
        entry.a = std::int64_t(a);
        entry.b = std::int64_t(b);
        entry.aUuid = graph.fibers[a].uuid;
        entry.bUuid = graph.fibers[b].uuid;
        entry.adjacent = adjacent;
        entry.approvalState = "approved";
        entry.anchorA = anchorA;
        entry.anchorB = anchorB;
        graph.links.push_back(std::move(entry));
    }
};

// A sampler that reports the same separation for every connector: enough to
// check what an asserted integer target is made of.
ScaffoldWindingDistances constantDistance(double turns)
{
    return [turns](const std::vector<std::pair<cv::Vec3d, cv::Vec3d>>& connectors, double, int) {
        return std::vector<double>(connectors.size(), turns);
    };
}

// Two fibers, both in the inward gauge.
const std::vector<std::int64_t> kInwardSigns{1, 1};

ScaffoldEventOptions withoutDistances()
{
    ScaffoldEventOptions options;
    options.measureWindingDistances = false;
    return options;
}

} // namespace

TEST_CASE("an adjacent link fixes the inward normal of both of its fibers")
{
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(250., 45., -150., 150., 30));
    const auto anchorH = polar(300., 45., 0.);
    const auto anchorV = polar(250., 45., 0.);
    scaffold.link(horizontal, vertical, anchorH, anchorV, true);

    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry);
    CHECK(normals.report().adjacentLinkAnchors == 2);
    CHECK(normals.report().manualAnchors == 0);
    CHECK(normals.report().anchoredFibers == 2);
    CHECK(normals.report().unanchoredFibers == 0);
    CHECK(normals.signSources()[horizontal] == ScaffoldSignSource::AdjacentLink);
    CHECK(normals.fixedSigns()[vertical] == 1);

    // The V fiber lies inside the H fiber, so both inward normals point from
    // the H anchor towards the V anchor: here exactly the inward radial.
    const Point inward{-std::cos(radians(45.)), -std::sin(radians(45.)), 0.};
    for (const auto& anchor : normals.anchors()) CHECK(dot(anchor.inward, inward) == doctest::Approx(1.).epsilon(1e-6));
    const auto solvedH = normals.at(horizontal, geometry.project(horizontal, anchorH).arc);
    const auto solvedV = normals.at(vertical, geometry.project(vertical, anchorV).arc);
    REQUIRE(solvedH);
    REQUIRE(solvedV);
    CHECK(dot(*solvedH, inward) > 0.99);
    CHECK(dot(*solvedV, inward) > 0.99);
    // An inward field disagrees with the outward radial prior.
    CHECK(normals.radialAgreement()[horizontal] < -0.9);
}

TEST_CASE("a manual display normal is read outward and fixes the gauge inward")
{
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    scaffold.manualOutwardNormal(horizontal, 2);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry);
    REQUIRE(normals.report().manualAnchors == 1);
    CHECK(normals.signSources()[horizontal] == ScaffoldSignSource::Manual);
    const auto& control = scaffold.graph.fibers[horizontal].content.controlPoints[2];
    const auto arc = geometry.controlArc(horizontal, 2);
    const auto solved = normals.at(horizontal, arc);
    REQUIRE(solved);
    // The stored direction points outward; the gauge is the inward one.
    CHECK(dot(*solved, outward(control)) < -0.99);
    CHECK(normals.radialAgreement()[horizontal] < -0.9);
}

TEST_CASE("an unanchored fiber keeps no sign of its own")
{
    Scaffold scaffold;
    scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry);
    CHECK(normals.report().anchoredFibers == 0);
    CHECK(normals.report().unanchoredFibers == 1);
    CHECK(normals.fixedSigns()[0] == 0);
    CHECK(normals.signSources()[0] == ScaffoldSignSource::None);
    CHECK(normals.solved(0));
}

TEST_CASE("an H and a V fiber that pass each other cross perpendicularly")
{
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(260., 45., -150., 150., 30));
    scaffold.manualOutwardNormal(horizontal, 2);
    scaffold.manualOutwardNormal(vertical, 1);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry);

    ScaffoldEventReport report;
    const auto events = scaffoldEvents(scaffold.graph, geometry, normals, withoutDistances(), {}, report);
    INFO("events=" << report.events << " perpendicular=" << report.perpendicular << " parallel=" << report.parallel
                   << " rejected_tangents=" << report.rejectedTangents);
    REQUIRE(report.events > 0);
    CHECK(report.perpendicular == report.events);
    CHECK(report.withoutNormals == 0);
    CHECK(report.withDistance == 0);
    for (const auto& event : events) {
        CHECK(event.validA);
        CHECK(event.validB);
        CHECK(event.validConnector);
        CHECK_FALSE(event.parallel);
        CHECK(std::isnan(event.distance));
        CHECK(((event.a == std::int64_t(horizontal) && event.b == std::int64_t(vertical)) ||
               (event.a == std::int64_t(vertical) && event.b == std::int64_t(horizontal))));
    }

    // Without a measured separation no arc is asserted, but the signed order
    // still witnesses the pair's enclosure.
    ScaffoldMeasuredReport measuredReport;
    const auto measured = scaffoldMeasuredConstraints(scaffold.graph, geometry, events, kInwardSigns, {}, measuredReport);
    CHECK(measuredReport.used == 0);
    CHECK(measuredReport.withoutDistance == report.events);
    CHECK(measuredReport.enclosurePairs == 1);
    CHECK(measuredReport.enclosureOneSided == 1);
    CHECK(measured.empty());
}

TEST_CASE("a measured separation asserts the integer step of a crossing")
{
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(260., 45., -150., 150., 30));
    scaffold.manualOutwardNormal(horizontal, 2);
    scaffold.manualOutwardNormal(vertical, 1);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry);

    ScaffoldEventOptions options;
    options.windingIntegrationStep = 8.;
    ScaffoldEventReport report;
    // One turn of separation: an H and a V fiber a whole winding apart.
    const auto events = scaffoldEvents(scaffold.graph, geometry, normals, options, constantDistance(1.), report);
    REQUIRE(report.events > 0);
    CHECK(report.withDistance == report.events);

    ScaffoldMeasuredReport measuredReport;
    const auto measured = scaffoldMeasuredConstraints(scaffold.graph, geometry, events, kInwardSigns, {}, measuredReport);
    CHECK(measuredReport.used == report.events);
    CHECK(measuredReport.perpendicular == report.events);
    REQUIRE_FALSE(measured.empty());
    for (const auto& constraint : measured) {
        CHECK(constraint.kind == ScaffoldConstraintKind::Perpendicular);
        CHECK(constraint.event >= 0);
        CHECK(constraint.windingDistance == doctest::Approx(1.));
        CHECK(constraint.weight > 0);
        // No fiber here crosses the seam, so the whole-fiber target is the
        // anchor step itself.
        CHECK(constraint.target == constraint.step);
        // The V fiber lies inside, one winding in: the step is one winding
        // either way, offset by the half winding between the two families.
        CHECK(std::abs(constraint.step) >= 1);
        CHECK(std::abs(constraint.step) <= 2);
    }
}

TEST_CASE("a pair that crosses twice in opposite order fixes its enclosure exactly")
{
    Scaffold scaffold;
    // More than one turn of spiral, from inside to outside the V fiber's
    // radius, so the pair meets once with the V fiber outside and once with
    // it inside.
    const auto spiral = scaffold.fiber(ScaffoldFamily::Horizontal, spiralLine(250., 350., 200., 680., 0., 260));
    const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(300., 250., -150., 150., 30));
    scaffold.manualOutwardNormal(spiral, 1);
    scaffold.manualOutwardNormal(vertical, 1);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry);
    CHECK(normals.report().anchoredFibers == 2);

    ScaffoldEventReport report;
    const auto events = scaffoldEvents(scaffold.graph, geometry, normals, withoutDistances(), {}, report);
    INFO("events=" << report.events);
    REQUIRE(report.events >= 2);

    ScaffoldMeasuredReport measuredReport;
    ScaffoldMeasuredOptions measuredOptions;
    const auto measured =
        scaffoldMeasuredConstraints(scaffold.graph, geometry, events, kInwardSigns, measuredOptions, measuredReport);
    INFO("pairs=" << measuredReport.enclosurePairs << " arcs=" << measuredReport.enclosureArcs << " one_sided="
                  << measuredReport.enclosureOneSided << " contradictory=" << measuredReport.enclosureContradictory
                  << " nonunique=" << measuredReport.enclosureNonunique);
    CHECK(measuredReport.enclosurePairs == 1);
    CHECK(measuredReport.enclosureArcs == 1);
    REQUIRE(measured.size() == 1);
    CHECK(measured.front().kind == ScaffoldConstraintKind::Enclosure);
    CHECK(measured.front().weight == doctest::Approx(measuredOptions.enclosureWeight));
    CHECK(measured.front().a == std::int64_t(std::min(spiral, vertical)));
    CHECK(measured.front().b == std::int64_t(std::max(spiral, vertical)));
}

TEST_CASE("the sign solve decides the gauge of a fiber no anchor fixed")
{
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(260., 45., -150., 150., 30));
    scaffold.manualOutwardNormal(horizontal, 2);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry);
    REQUIRE(normals.report().anchoredFibers == 1);
    REQUIRE(normals.report().unanchoredFibers == 1);

    ScaffoldEventReport report;
    const auto events = scaffoldEvents(scaffold.graph, geometry, normals, withoutDistances(), {}, report);
    REQUIRE(report.events > 0);
    const auto signs = scaffoldNormalSigns(scaffold.graph, normals, events);
    CHECK(signs.solved);
    CHECK(signs.events == report.events);
    CHECK(signs.fixed == 1);
    CHECK(signs.decided == 1);
    CHECK(signs.undecided == 0);
    CHECK(signs.sign[horizontal] == 1);
    CHECK(std::abs(signs.sign[vertical]) == 1);
    CHECK(signs.source[vertical] == ScaffoldSignSource::Crossings);
    // The crossing's evidence is the dot of the two gauge normals, so the
    // decided sign is the one that makes the V fiber agree with the anchored
    // H fiber.
    const auto solved = normals.at(vertical, geometry.project(vertical, polar(260., 45., 0.)).arc);
    REQUIRE(solved);
    const Point inward{-std::cos(radians(45.)), -std::sin(radians(45.)), 0.};
    // A straight V fiber's gauge direction is arbitrary within the plane
    // perpendicular to it - only its sign is decided here, and the decided
    // one points to the inward half space.
    CHECK(dot(*solved, inward) * double(signs.sign[vertical]) > 0.5);
    CHECK(signs.fixedDisagreements == 0);
}

TEST_CASE("the sign stage falls back to the anchors when it is switched off")
{
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(260., 45., -150., 150., 30));
    scaffold.manualOutwardNormal(horizontal, 2);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry);
    ScaffoldSignOptions options;
    options.solve = false;
    const auto signs = scaffoldNormalSigns(scaffold.graph, normals, {}, options);
    CHECK_FALSE(signs.solved);
    CHECK(signs.fixed == 1);
    CHECK(signs.decided == 0);
    CHECK(signs.undecided == 1);
    CHECK(signs.sign[0] == 1);
    CHECK(signs.sign[1] == 1);
}

TEST_CASE("the winding solve takes measured constraints beside the links")
{
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(260., 45., -150., 150., 30));
    scaffold.link(horizontal, vertical, polar(300., 45., 0.), polar(260., 45., 0.), true);
    const ScaffoldGeometry geometry(scaffold.graph);

    // One measured relation asserting the same adjacency the link asserts.
    ScaffoldConstraint measured;
    measured.kind = ScaffoldConstraintKind::Perpendicular;
    measured.a = std::int64_t(horizontal);
    measured.b = std::int64_t(vertical);
    measured.arcA = geometry.project(horizontal, polar(300., 45., 0.)).arc;
    measured.arcB = geometry.project(vertical, polar(260., 45., 0.)).arc;
    measured.step = -1;
    measured.target = -1;
    measured.weight = 2.;
    measured.event = 0;

    const auto result = solveScaffoldWinding(scaffold.graph, geometry, {}, {&measured, 1});
    CHECK(result.constraintsByKind[std::size_t(ScaffoldConstraintKind::Manual)] == 1);
    CHECK(result.constraintsByKind[std::size_t(ScaffoldConstraintKind::Perpendicular)] == 1);
    CHECK(result.objective == doctest::Approx(0.));
    CHECK(result.components == 1);
    CHECK(result.winding[horizontal] == 1);
    CHECK(result.winding[vertical] == 0);
    // Both relations are the same arc, so their weights are summed.
    REQUIRE(result.arcs.size() == 1);
    CHECK(result.arcs.front().weight == doctest::Approx(3.));
}

TEST_CASE("a scaffold's measured relations are the ones the bulk assembly builds")
{
    // The per-event path and the bulk preparation share one relation helper
    // (vc_object_store FiberWindingGraph.hpp). Feeding both the same
    // crossings must give the same arcs, weights and enclosure relations.
    Scaffold scaffold;
    const auto spiral = scaffold.fiber(ScaffoldFamily::Horizontal, spiralLine(250., 350., 200., 680., 0., 260));
    const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(300., 250., -150., 150., 30));
    scaffold.manualOutwardNormal(spiral, 1);
    scaffold.manualOutwardNormal(vertical, 1);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry);

    ScaffoldEventReport report;
    ScaffoldEventOptions eventOptions;
    const auto events = scaffoldEvents(scaffold.graph, geometry, normals, eventOptions, constantDistance(0.75), report);
    REQUIRE(report.events >= 2);

    ScaffoldMeasuredOptions measuredOptions;
    measuredOptions.perpendicularWeight = 2.;
    measuredOptions.parallelWeight = 3.;
    ScaffoldMeasuredReport measuredReport;
    const auto measured =
        scaffoldMeasuredConstraints(scaffold.graph, geometry, events, kInwardSigns, measuredOptions, measuredReport);

    // The same events as the bulk preparation's columns.
    std::vector<std::int64_t> a, b;
    std::vector<double> arcA, arcB, distance, normalA, normalB, connector;
    // std::vector<bool> is no contiguous array, so the flags are their own
    // storage the spans can point at.
    std::unique_ptr<bool[]> parallel(new bool[events.size()]), validA(new bool[events.size()]),
        validB(new bool[events.size()]), validConnector(new bool[events.size()]);
    std::size_t at = 0;
    for (const auto& event : events) {
        a.push_back(event.a);
        b.push_back(event.b);
        arcA.push_back(event.arcA);
        arcB.push_back(event.arcB);
        distance.push_back(event.distance);
        parallel[at] = event.parallel;
        validA[at] = event.validA;
        validB[at] = event.validB;
        validConnector[at] = event.validConnector;
        ++at;
        for (std::size_t k = 0; k < 3; ++k) {
            normalA.push_back(event.normalA[k]);
            normalB.push_back(event.normalB[k]);
            connector.push_back(event.pointB[k] - event.pointA[k]);
        }
    }
    const std::vector<std::int64_t> family{std::int64_t(scaffold.graph.fibers[0].family),
                                           std::int64_t(scaffold.graph.fibers[1].family)};
    vc::geometry::FiberWindingGraphInput input;
    input.a = {a.data(), a.size(), 1};
    input.b = {b.data(), b.size(), 1};
    input.family = family;
    input.normalSign = kInwardSigns;
    input.arcA = {arcA.data(), arcA.size(), 1};
    input.arcB = {arcB.data(), arcB.size(), 1};
    input.distance = {distance.data(), distance.size(), 1};
    input.parallel = {parallel.get(), events.size()};
    input.validA = {validA.get(), events.size()};
    input.validB = {validB.get(), events.size()};
    input.validConnector = {validConnector.get(), events.size()};
    input.normalA = normalA;
    input.normalB = normalB;
    input.connector = connector;
    vc::geometry::FiberWindingGraphOptions bulk;
    bulk.perpendicularWeight = measuredOptions.perpendicularWeight;
    bulk.parallelWeight = measuredOptions.parallelWeight;
    bulk.perpendicularOffset = measuredOptions.perpendicularOffset;
    bulk.windingScale = measuredOptions.windingScale;
    bulk.enclosures = true;
    bulk.workers = 1;
    const auto columns = vc::geometry::prepareFiberWindingGraph(geometry.packed(), input, bulk);

    std::vector<WindingArc> mine;
    for (const auto& constraint : measured)
        if (constraint.kind != ScaffoldConstraintKind::Enclosure)
            mine.push_back({constraint.a, constraint.b, constraint.target, constraint.weight});
    mine = aggregateWindingArcs(std::move(mine));
    REQUIRE(mine.size() == columns.arcs);
    for (std::size_t i = 0; i < mine.size(); ++i) {
        CHECK(mine[i].a == columns.a[i]);
        CHECK(mine[i].b == columns.b[i]);
        CHECK(mine[i].target == columns.target[i]);
        CHECK(mine[i].weight == doctest::Approx(columns.weight[i]));
    }
    std::vector<ScaffoldConstraint> enclosures;
    for (const auto& constraint : measured)
        if (constraint.kind == ScaffoldConstraintKind::Enclosure) enclosures.push_back(constraint);
    REQUIRE(enclosures.size() == columns.enclosureArcs);
    for (std::size_t i = 0; i < enclosures.size(); ++i) {
        // The bulk columns state an H -> V relation; a constraint states it
        // in node order.
        const bool ordered = columns.enclosureA[i] < columns.enclosureB[i];
        CHECK(enclosures[i].a == std::min(columns.enclosureA[i], columns.enclosureB[i]));
        CHECK(enclosures[i].b == std::max(columns.enclosureA[i], columns.enclosureB[i]));
        CHECK(enclosures[i].target == (ordered ? columns.enclosureTarget[i] : -columns.enclosureTarget[i]));
    }
    CHECK(measuredReport.withoutDistance == std::int64_t(columns.withoutDistance));
    CHECK(measuredReport.used == std::int64_t(columns.used));
}

TEST_CASE("an approval row marks the measured constraint it names")
{
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(260., 45., -150., 150., 30));
    scaffold.manualOutwardNormal(horizontal, 2);
    scaffold.manualOutwardNormal(vertical, 1);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry);
    ScaffoldEventReport report;
    const auto events = scaffoldEvents(scaffold.graph, geometry, normals, {}, constantDistance(1.), report);
    ScaffoldMeasuredReport measuredReport;
    auto measured = scaffoldMeasuredConstraints(scaffold.graph, geometry, events, kInwardSigns, {}, measuredReport);
    REQUIRE_FALSE(measured.empty());
    for (const auto& constraint : measured) CHECK_FALSE(constraint.approved);

    const auto row = approveScaffoldConstraint(scaffold.graph, geometry, measured.front(), "alice",
                                               "2026-11-17T10:00:00Z", "host-a");
    const auto identity = scaffoldConstraintIdentity(scaffold.graph, measured.front());
    CHECK(row.identity == identity);
    CHECK(row.identity.fiberA < row.identity.fiberB);

    ScaffoldApprovalReport approvals;
    applyScaffoldConstraintApprovals(scaffold.graph, geometry, measured, {&row, 1}, approvals);
    CHECK(approvals.live == 1);
    CHECK(approvals.approved >= 1);
    CHECK(approvals.unmatched == 0);
    CHECK(measured.front().approved);

    // A revoked row approves nothing, and the one it revoked is not live.
    const auto revoked = vc::objects::revokeFiberConstraintApproval(row, "bob", "2026-11-17T11:00:00Z");
    const std::vector<vc::objects::FiberConstraintApproval> both{row, revoked};
    for (auto& constraint : measured) constraint.approved = false;
    applyScaffoldConstraintApprovals(scaffold.graph, geometry, measured, both, approvals);
    CHECK(approvals.rows == 2);
    CHECK(approvals.live == 0);
    CHECK(approvals.approved == 0);
    CHECK_FALSE(measured.front().approved);

    // Moving a line point of the covering span invalidates the approval.
    Scaffold moved = scaffold;
    const auto spans = scaffoldConstraintSpans(scaffold.graph, geometry, measured.front());
    REQUIRE(spans);
    const auto& fiber = moved.graph.fibers[std::size_t(measured.front().a)];
    const auto line = std::size_t(fiber.content.controlLineIndex.at(spans->spanA));
    moved.graph.linePoints[3 * (std::size_t(fiber.pointFirst) + line) + 2] += 0.5;
    const ScaffoldGeometry movedGeometry(moved.graph);
    for (auto& constraint : measured) constraint.approved = false;
    applyScaffoldConstraintApprovals(moved.graph, movedGeometry, measured, {&row, 1}, approvals);
    CHECK(approvals.live == 1);
    CHECK(approvals.approved == 0);
    CHECK(approvals.stale == 1);
    CHECK_FALSE(measured.front().approved);
}

TEST_CASE("a prediction's sheet axis pulls the strip normals")
{
    // A V fiber running straight along z has no geometry preference for its
    // normal direction. A prediction that states the radial axis everywhere
    // decides it, without fixing a sign.
    Scaffold scaffold;
    const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(300., 45., -150., 150., 30));
    const ScaffoldGeometry geometry(scaffold.graph);
    const Point radial{std::cos(radians(45.)), std::sin(radians(45.)), 0.};

    std::size_t sampled = 0;
    const ScaffoldAxisSampler prediction = [&](std::span<const Point> points, std::size_t) {
        sampled = points.size();
        std::vector<ScaffoldPredictedAxis> out;
        out.reserve(points.size());
        for (std::size_t i = 0; i < points.size(); ++i) out.push_back({radial, 128., true});
        return out;
    };
    ScaffoldNormalOptions options;
    options.strip.axisWeight = 100.;
    const ScaffoldNormals fused(scaffold.graph, geometry, options, prediction);
    CHECK(sampled == std::size_t(scaffold.graph.fibers[vertical].pointCount));
    CHECK(fused.report().predictedVertices == scaffold.graph.fibers[vertical].pointCount);
    CHECK(fused.report().verticesWithAxis == fused.report().predictedVertices);
    CHECK(fused.report().chainNodes > 0);
    CHECK(fused.report().nodesWithAxis == fused.report().chainNodes);
    CHECK(fused.report().meanAxisWeight == doctest::Approx(1.));
    // No anchor fixed a sign, so the gauge is still free - but the axis is
    // the predicted one.
    CHECK(fused.report().anchoredFibers == 0);
    const auto solved = fused.at(vertical, geometry.arcLength(vertical) / 2.);
    REQUIRE(solved);
    CHECK(std::abs(dot(*solved, radial)) > 0.99);

    // A prediction whose presence stays below the floor weighs nothing.
    const ScaffoldAxisSampler weightless = [&](std::span<const Point> points, std::size_t) {
        return std::vector<ScaffoldPredictedAxis>(points.size(), ScaffoldPredictedAxis{radial, 0., true});
    };
    const ScaffoldNormals none(scaffold.graph, geometry, options, weightless);
    CHECK(none.report().verticesWithAxis == 0);
    CHECK(none.report().nodesWithAxis == 0);
}
