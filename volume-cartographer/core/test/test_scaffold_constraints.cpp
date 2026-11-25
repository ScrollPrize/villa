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
#include <functional>
#include <doctest/doctest.h>

#include "vc/fiber_graph/ScaffoldCache.hpp"
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

// Every relation type at weight 1, no sign constraints and no crossing
// confidence: the tests below state what each relation asserts, not how the
// scaffold workflow weights it by default.
ScaffoldMeasuredOptions unitWeights()
{
    ScaffoldMeasuredOptions options;
    options.perpendicularWeight = options.parallelWeight = options.enclosureWeight = 1.;
    options.signWeight = 0.;
    options.crossingConfidence = false;
    return options;
}

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

    // A normals approval of the whole fiber as the loader hands it over: the
    // approved normal at the given line points (zero elsewhere), every span
    // approved by one live row.
    void approveNormals(std::size_t fiber, std::size_t firstPoint, std::size_t lastPoint,
                        const std::function<Point(std::size_t point, const Point& at)>& normal)
    {
        auto& entry = graph.fibers[fiber];
        entry.approvedLineNormals.assign(std::size_t(entry.pointCount), Point{0., 0., 0.});
        for (std::size_t j = firstPoint; j <= lastPoint; ++j) {
            const auto at = std::size_t(3 * (entry.pointFirst + std::int64_t(j)));
            entry.approvedLineNormals[j] = normal(j, Point{graph.linePoints[at], graph.linePoints[at + 1], graph.linePoints[at + 2]});
        }
        vc::objects::FiberPieceApproval row;
        row.rowUuid = vc::objects::util::newUuid();
        row.aspect = vc::objects::FiberPieceAspect::Normals;
        entry.approvals.live = {row};
        entry.approvals.rowCurrent = {true};
        const auto aspect = std::size_t(vc::objects::FiberPieceAspect::Normals);
        entry.approvals.spans[aspect].assign(entry.content.controlIds.size() - 1, {});
        entry.approvals.approvedSpans[aspect].clear();
        for (std::size_t span = 0; span + 1 < entry.content.controlIds.size(); ++span) {
            const auto from = std::size_t(entry.content.controlLineIndex[span]), to = std::size_t(entry.content.controlLineIndex[span + 1]);
            if (from < firstPoint || to > lastPoint) continue;
            entry.approvals.spans[aspect][span] = {std::size_t(0), true};
            entry.approvals.approvedSpans[aspect].push_back(span);
        }
    }

    // The test's way to give fibers their outward facing, as a run gets it
    // from a traced volume: neighbour normals right beside the named fibers'
    // own lines (8 voxels inside and outside, facing outward) that reach no
    // other fiber of the scaffold.
    std::vector<std::size_t> faced;
    std::shared_ptr<const vc::fiber_tracer::FiberNeighbourField> facingField;
    void faceOutward(std::size_t fiber) { faced.push_back(fiber); }
    ScaffoldNormalOptions facing()
    {
        std::vector<std::vector<vc::fiber_tracer::FiberStripSample>> beside;
        for (const auto fiber : faced)
            for (const double offset : {-8., 8.}) {
                const auto& entry = graph.fibers.at(fiber);
                std::vector<vc::fiber_tracer::FiberStripSample> samples;
                double arc = 0;
                cv::Vec3d previous;
                for (std::int64_t j = 0; j < entry.pointCount; ++j) {
                    const auto at = std::size_t(3 * (entry.pointFirst + j));
                    const Point point{graph.linePoints[at], graph.linePoints[at + 1], graph.linePoints[at + 2]};
                    const auto normal = outward(point);
                    const cv::Vec3d moved{point[0] + offset * normal[0], point[1] + offset * normal[1], point[2]};
                    if (j > 0) arc += cv::norm(moved - previous);
                    samples.push_back({arc, moved, {normal[0], normal[1], 0.}, {0., 0., 0.}});
                    previous = moved;
                }
                beside.push_back(std::move(samples));
            }
        const auto count = beside.size();
        facingField = std::make_shared<const vc::fiber_tracer::FiberNeighbourField>(std::move(beside), std::vector<int>(count, 1), 24.);
        ScaffoldNormalOptions options;
        options.neighbours.field = facingField.get();
        options.neighbours.identity = "facing";
        options.neighbours.volume = graph.volume;
        options.neighbours.variant.radius = 24.;
        return options;
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

// Two fibers, both in the outward gauge.
const std::vector<std::int64_t> kOutwardSigns{1, 1};
// The same two as the bulk winding assembly takes them: turned the way the
// winding falls, which in these scrolls (handedness +1) is inward.
const std::vector<std::int64_t> kFallingSigns{-1, -1};

ScaffoldEventOptions withoutDistances()
{
    ScaffoldEventOptions options;
    options.measureWindingDistances = false;
    return options;
}

} // namespace

TEST_CASE("the cache reuses unchanged fibers and pairs, and nothing else")
{
    // The same scaffold solved twice: the second run reads every normal and
    // every pair from what the first wrote, and extracts nothing.
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(260., 45., -150., 150., 30));
    scaffold.faceOutward(horizontal);
    scaffold.faceOutward(vertical);
    // A manual normal on each: what fixes a fiber's normal is part of its key.
    scaffold.manualOutwardNormal(horizontal, 2);
    scaffold.manualOutwardNormal(vertical, 1);
    // Versions: the store states them, and the cache is keyed on them.
    for (auto& fiber : scaffold.graph.fibers) fiber.version = 7;
    const ScaffoldGeometry geometry(scaffold.graph);

    ScaffoldCacheSink sink;
    sink.author = "cache-test";
    sink.computedAt = "2026-11-18T12:00:00Z";
    ScaffoldNormalOptions normalOptions = scaffold.facing();
    normalOptions.sink = &sink;
    const ScaffoldNormals first(scaffold.graph, geometry, normalOptions);
    CHECK(first.report().reused == 0);
    CHECK(first.report().cached == 2);
    REQUIRE(sink.normals.size() == 2);

    ScaffoldEventOptions eventOptions;
    eventOptions.sink = &sink;
    ScaffoldEventReport firstReport;
    const auto firstEvents =
        scaffoldEvents(scaffold.graph, geometry, first, eventOptions, constantDistance(1.), firstReport);
    CHECK(firstReport.reusedPairs == 0);
    CHECK(firstReport.cachedPairs == 1);
    REQUIRE_FALSE(firstEvents.empty());
    REQUIRE_FALSE(sink.crossings.empty());

    // Second run, same inputs: everything comes from the cache, and the
    // events are the ones the first run found.
    const ScaffoldCache cache(sink.normals, sink.crossings);
    ScaffoldNormalOptions reuseNormals = scaffold.facing();
    reuseNormals.cache = &cache;
    const ScaffoldNormals second(scaffold.graph, geometry, reuseNormals);
    CHECK(second.report().reused == 2);
    CHECK(second.report().cached == 0);
    // The reused rows are the ones the first run wrote, by their own identity,
    // and what they were solved from is stated in the clear.
    CHECK(second.cacheRows() == first.cacheRows());
    for (const auto& row : sink.normals) {
        CHECK_FALSE(row.normalsUuid.empty());
        CHECK(row.inputs.find("twist=") != std::string::npos);
        CHECK(row.inputs.find("by=control ") != std::string::npos);
    }
    for (std::size_t i = 0; i < scaffold.graph.fibers.size(); ++i) CHECK(second.solved(i));

    ScaffoldEventOptions reuseEvents;
    reuseEvents.cache = &cache;
    ScaffoldEventReport secondReport;
    // The sampler is still handed over: a reusing run must not need it for
    // the pairs it reuses, but the option still says separations are
    // measured, and a pair that was not cached would need it.
    const auto secondEvents =
        scaffoldEvents(scaffold.graph, geometry, second, reuseEvents, constantDistance(1.), secondReport);
    CHECK(secondReport.reusedPairs == 1);
    CHECK(secondReport.reusedEvents == std::int64_t(firstEvents.size()));
    CHECK(secondEvents.size() == firstEvents.size());
    for (std::size_t i = 0; i < secondEvents.size(); ++i) {
        CHECK(secondEvents[i].a == firstEvents[i].a);
        CHECK(secondEvents[i].b == firstEvents[i].b);
        CHECK(secondEvents[i].arcA == doctest::Approx(firstEvents[i].arcA));
        CHECK(secondEvents[i].arcB == doctest::Approx(firstEvents[i].arcB));
        CHECK(secondEvents[i].parallel == firstEvents[i].parallel);
        CHECK(secondEvents[i].distance == doctest::Approx(firstEvents[i].distance));
    }

    // An edited fiber: its version moved, so its normals and every pair it
    // is in are computed again.
    auto edited = scaffold;
    edited.graph.fibers[horizontal].version = 8;
    const ScaffoldGeometry editedGeometry(edited.graph);
    const ScaffoldNormals third(edited.graph, editedGeometry, reuseNormals);
    CHECK(third.report().reused == 1);
    ScaffoldEventReport thirdReport;
    scaffoldEvents(edited.graph, editedGeometry, third, reuseEvents, constantDistance(1.), thirdReport);
    CHECK(thirdReport.reusedPairs == 0);

    // A changed parameter: a different question, so nothing matches.
    ScaffoldNormalOptions otherParams = reuseNormals;
    otherParams.strip.twistWeight *= 2.;
    const ScaffoldNormals fourth(scaffold.graph, geometry, otherParams);
    CHECK(fourth.report().reused == 0);
}

TEST_CASE("a fractional separation is split or rounded as the mode says")
{
    // Two crossing fibers whose measured separation is fractional: the split
    // asserts the two integers around it with weights 1 - fraction and
    // fraction, the rounded mode asserts the nearer one with the whole
    // weight.
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(260., 45., -150., 150., 30));
    scaffold.faceOutward(horizontal);
    scaffold.faceOutward(vertical);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry, scaffold.facing());
    ScaffoldEventReport events;
    const auto crossings = scaffoldEvents(scaffold.graph, geometry, normals, {}, constantDistance(1.8), events);
    REQUIRE_FALSE(crossings.empty());

    ScaffoldMeasuredReport splitReport;
    ScaffoldMeasuredOptions split = unitWeights();
    split.separations = ScaffoldSeparationMode::Split;
    const auto splitArcs =
        scaffoldMeasuredConstraints(scaffold.graph, geometry, crossings, kOutwardSigns, split, splitReport);
    ScaffoldMeasuredReport roundedReport;
    ScaffoldMeasuredOptions rounded = unitWeights();
    rounded.separations = ScaffoldSeparationMode::Rounded;
    const auto roundedArcs =
        scaffoldMeasuredConstraints(scaffold.graph, geometry, crossings, kOutwardSigns, rounded, roundedReport);

    CHECK(splitReport.used == roundedReport.used);
    CHECK(splitReport.fractional == roundedReport.fractional);
    CHECK(splitReport.fractional > 0);
    // The split emits two arcs per fractional crossing, the rounded one.
    CHECK(splitReport.expanded == 2 * roundedReport.expanded);
    CHECK(splitArcs.size() == 2 * roundedArcs.size());
    // Each rounded constraint carries the whole weight of its event, and the
    // two halves of the split sum to it.
    for (const auto& constraint : roundedArcs) CHECK(constraint.weight == doctest::Approx(1.));
    double splitWeight = 0;
    for (const auto& constraint : splitArcs) splitWeight += constraint.weight;
    double roundedWeight = 0;
    for (const auto& constraint : roundedArcs) roundedWeight += constraint.weight;
    CHECK(splitWeight == doctest::Approx(roundedWeight));
    // No two rounded constraints of one event contradict each other on the
    // same pair, which is what the split does.
    for (std::size_t i = 0; i < roundedArcs.size(); ++i)
        for (std::size_t j = i + 1; j < roundedArcs.size(); ++j)
            CHECK_FALSE(roundedArcs[i].a == roundedArcs[j].a && roundedArcs[i].b == roundedArcs[j].b &&
                        roundedArcs[i].event == roundedArcs[j].event);
}

TEST_CASE("an adjacent link fixes the outward normal of both of its fibers")
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

    // The V fiber lies one winding below the H fiber and, in this scroll
    // (handedness +1, the winding falls inward), inside it: the outward
    // normal of both points from the V anchor towards the H anchor, here
    // exactly the outward radial.
    const Point radial{std::cos(radians(45.)), std::sin(radians(45.)), 0.};
    for (const auto& anchor : normals.anchors()) CHECK(dot(anchor.outward, radial) == doctest::Approx(1.).epsilon(1e-6));
    const auto solvedH = normals.at(horizontal, geometry.project(horizontal, anchorH).arc);
    const auto solvedV = normals.at(vertical, geometry.project(vertical, anchorV).arc);
    REQUIRE(solvedH);
    REQUIRE(solvedV);
    CHECK(dot(*solvedH, radial) > 0.99);
    CHECK(dot(*solvedV, radial) > 0.99);
    // Every signed normal points outward, so it agrees with the outward radial prior.
    CHECK(normals.radialAgreement()[horizontal] > 0.9);
}

TEST_CASE("a manual display normal fixes its axis at the control point, not the fiber's facing")
{
    for (const double facing : {1., -1.}) {
        Scaffold scaffold;
        const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
        scaffold.manualOutwardNormal(horizontal, 2);
        // The stored vector's direction says nothing: the annotator sees a cross-section.
        if (facing < 0) {
            auto& stored = scaffold.graph.fibers[horizontal].content.annotations.at(2)["display_normal"];
            for (auto& value : stored) value = -value.get<double>();
        }
        const ScaffoldGeometry geometry(scaffold.graph);
        const ScaffoldNormals normals(scaffold.graph, geometry);
        REQUIRE(normals.report().manualAnchors == 1);
        CHECK(normals.signSources()[horizontal] == ScaffoldSignSource::None);
        CHECK(normals.fixedSigns()[horizontal] == 0);
        CHECK(normals.report().unanchoredFibers == 1);
        const auto& control = scaffold.graph.fibers[horizontal].content.controlPoints[2];
        const auto solved = normals.at(horizontal, geometry.controlArc(horizontal, 2));
        REQUIRE(solved);
        CHECK(std::abs(dot(*solved, outward(control))) > 0.99);
    }
}

namespace {
// The outward radial tilted by 25 degrees towards z, about the fiber: a normal
// the geometry of a flat arc does not give.
Point tilted(const Point& at)
{
    const auto radial = outward(at);
    const double c = std::cos(radians(25.)), s = std::sin(radians(25.));
    return {c * radial[0], c * radial[1], s};
}
} // namespace

TEST_CASE("an approved stretch comes out with its stored normals, and the fiber's facing stays open")
{
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    // Stored as axes: every other point faces the other way.
    scaffold.approveNormals(horizontal, 10, 30, [](std::size_t point, const Point& at) {
        const auto normal = tilted(at);
        return point % 2 ? Point{-normal[0], -normal[1], -normal[2]} : normal;
    });
    // A manual normal inside the stretch is part of what was approved, not an anchor of its own.
    scaffold.manualOutwardNormal(horizontal, 2);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry);
    CHECK(normals.report().approvedNormalPoints == 21);
    CHECK(normals.report().approvedPieceAnchors > 0);
    CHECK(normals.report().manualAnchors == 0);
    CHECK(normals.report().staleNormalApprovals == 0);
    CHECK(normals.signSources()[horizontal] == ScaffoldSignSource::None);
    CHECK(normals.fixedSigns()[horizontal] == 0);
    const auto& fiber = scaffold.graph.fibers[horizontal];
    const auto solved = normals.normals();
    const auto at = [&](std::size_t j) {
        const auto first = std::size_t(3 * (fiber.pointFirst + std::int64_t(j)));
        return Point{solved[first], solved[first + 1], solved[first + 2]};
    };
    const auto point = [&](std::size_t j) {
        const auto first = std::size_t(3 * (fiber.pointFirst + std::int64_t(j)));
        return Point{scaffold.graph.linePoints[first], scaffold.graph.linePoints[first + 1], scaffold.graph.linePoints[first + 2]};
    };
    // Exactly the approved axis at every approved point ...
    for (std::size_t j = 10; j <= 30; ++j) CHECK(std::abs(dot(at(j), tilted(point(j)))) == doctest::Approx(1.).epsilon(1e-9));
    // ... and one facing along the whole fiber, the unapproved ends included.
    for (std::size_t j = 0; j + 1 < std::size_t(fiber.pointCount); ++j) CHECK(dot(at(j), at(j + 1)) > 0.8);
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

// A neighbour field of two arcs beside an H fiber at radius 300, their
// signed normals radial: `facing` +1 outward, -1 inward.
vc::fiber_tracer::FiberNeighbourField radialField(double facing, double radiusOffset = 0.)
{
    std::vector<std::vector<vc::fiber_tracer::FiberStripSample>> fibers;
    for (const double radius : {280. + radiusOffset, 320. + radiusOffset}) {
        std::vector<vc::fiber_tracer::FiberStripSample> samples;
        for (double degrees = 15.; degrees <= 75.; degrees += 2.) {
            const auto point = polar(radius, degrees, 0.);
            const auto normal = outward(point);
            samples.push_back({radians(degrees - 15.) * radius, {point[0], point[1], point[2]},
                               {facing * normal[0], facing * normal[1], 0.}, {0., 0., 0.}});
        }
        fibers.push_back(std::move(samples));
    }
    return vc::fiber_tracer::FiberNeighbourField(std::move(fibers), {1, 1}, 384.);
}

TEST_CASE("neighbour normals face a fiber no anchor fixed")
{
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    const ScaffoldGeometry geometry(scaffold.graph);
    const auto middle = geometry.controlArc(horizontal, 2);
    for (const double facing : {1., -1.}) {
        const auto field = radialField(facing);
        ScaffoldNormalOptions options;
        options.neighbours.field = &field;
        options.neighbours.identity = "test field";
        options.neighbours.volume = "test-volume";
        const ScaffoldNormals normals(scaffold.graph, geometry, options);
        CHECK(normals.report().neighbourSamples > 0);
        CHECK(normals.report().neighbourSamplesAnswered == normals.report().neighbourSamples);
        CHECK(normals.report().neighbourFacedFibers == 1);
        CHECK(normals.signSources()[horizontal] == ScaffoldSignSource::Neighbours);
        CHECK(normals.fixedSigns()[horizontal] == 1);
        const auto solved = normals.at(horizontal, middle);
        REQUIRE(solved);
        CHECK(facing * dot(*solved, outward(scaffold.graph.fibers[horizontal].content.controlPoints[2])) > 0.99);
    }
}

TEST_CASE("crossings by the field's fibers and by the scaffold's own give the crossing axis")
{
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    // A V fiber of the scaffold crossing the H fiber 10 voxels outside it.
    scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(310., 45., -150., 150., 30));
    const ScaffoldGeometry geometry(scaffold.graph);
    // The field: the two arcs beside the H fiber, and a vertical fiber crossing it at 30 degrees.
    auto fibers = std::vector<std::vector<vc::fiber_tracer::FiberStripSample>>{};
    {
        std::vector<vc::fiber_tracer::FiberStripSample> vertical;
        for (double z = -160.; z <= 160.; z += 16.) {
            const auto point = polar(292., 30., z);
            const auto normal = outward(point);
            vertical.push_back({z + 160., {point[0], point[1], point[2]}, {normal[0], normal[1], 0.}, {0., 0., 0.}});
        }
        fibers.push_back(std::move(vertical));
    }
    const vc::fiber_tracer::FiberNeighbourField field(std::move(fibers), {1}, 384.);
    ScaffoldNormalOptions options;
    options.neighbours.field = &field;
    options.neighbours.identity = "test field";
    options.neighbours.volume = "test-volume";
    {
        const ScaffoldNormals normals(scaffold.graph, geometry, options);
        CHECK(normals.report().fieldCrossings == 1);    // the H fiber by the field's vertical fiber
        CHECK(normals.report().scaffoldCrossings == 2); // the H and the V fiber by each other
        CHECK(normals.report().crossedNodes > 0);
        const auto solved = normals.at(horizontal, geometry.controlArc(horizontal, 2));
        REQUIRE(solved);
        // Own tangent x a vertical tangent is radial: the crossing axis the normal lies along.
        CHECK(std::abs(dot(*solved, outward(scaffold.graph.fibers[horizontal].content.controlPoints[2]))) > 0.99);
    }
    {
        options.neighbours.fieldCrossings = options.neighbours.scaffoldCrossings = false;
        const ScaffoldNormals normals(scaffold.graph, geometry, options);
        CHECK(normals.report().fieldCrossings == 0);
        CHECK(normals.report().scaffoldCrossings == 0);
        CHECK(normals.report().crossedNodes == 0);
    }
}

TEST_CASE("an anchor outranks the neighbour normals, and a field elsewhere is refused")
{
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(250., 45., -150., 150., 30));
    scaffold.link(horizontal, vertical, polar(300., 45., 0.), polar(250., 45., 0.), true);
    const ScaffoldGeometry geometry(scaffold.graph);
    ScaffoldNormalOptions options;
    options.neighbours.identity = "test field";
    options.neighbours.volume = "test-volume";
    {
        // A field facing inward against the adjacent link's outward anchor: the link wins.
        const auto field = radialField(-1.);
        options.neighbours.field = &field;
        const ScaffoldNormals normals(scaffold.graph, geometry, options);
        CHECK(normals.signSources()[horizontal] == ScaffoldSignSource::AdjacentLink);
        CHECK(normals.report().neighbourFacedFibers == 0);
        const auto solved = normals.at(horizontal, geometry.project(horizontal, polar(300., 45., 0.)).arc);
        REQUIRE(solved);
        CHECK(dot(*solved, outward(polar(300., 45., 0.))) > 0.99);
    }
    {
        // No field fiber within reach of any scaffold fiber.
        const auto field = radialField(1., 5000.);
        options.neighbours.field = &field;
        CHECK_THROWS(ScaffoldNormals(scaffold.graph, geometry, options));
    }
    {
        // A field in another volume's voxels.
        const auto field = radialField(1.);
        options.neighbours.field = &field;
        options.neighbours.volume = "another-volume";
        CHECK_THROWS(ScaffoldNormals(scaffold.graph, geometry, options));
    }
}

TEST_CASE("an H and a V fiber that pass each other cross perpendicularly")
{
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(260., 45., -150., 150., 30));
    scaffold.faceOutward(horizontal);
    scaffold.faceOutward(vertical);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry, scaffold.facing());

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
    const auto measured = scaffoldMeasuredConstraints(scaffold.graph, geometry, events, kOutwardSigns, unitWeights(), measuredReport);
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
    scaffold.faceOutward(horizontal);
    scaffold.faceOutward(vertical);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry, scaffold.facing());

    ScaffoldEventOptions options;
    options.windingIntegrationStep = 8.;
    ScaffoldEventReport report;
    // One turn of separation: an H and a V fiber a whole winding apart.
    const auto events = scaffoldEvents(scaffold.graph, geometry, normals, options, constantDistance(1.), report);
    REQUIRE(report.events > 0);
    CHECK(report.withDistance == report.events);

    ScaffoldMeasuredReport measuredReport;
    const auto measured = scaffoldMeasuredConstraints(scaffold.graph, geometry, events, kOutwardSigns, unitWeights(), measuredReport);
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
    scaffold.faceOutward(spiral);
    scaffold.faceOutward(vertical);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry, scaffold.facing());
    CHECK(normals.report().anchoredFibers == 2);

    ScaffoldEventReport report;
    const auto events = scaffoldEvents(scaffold.graph, geometry, normals, withoutDistances(), {}, report);
    INFO("events=" << report.events);
    REQUIRE(report.events >= 2);

    ScaffoldMeasuredReport measuredReport;
    ScaffoldMeasuredOptions measuredOptions = unitWeights();
    const auto measured =
        scaffoldMeasuredConstraints(scaffold.graph, geometry, events, kOutwardSigns, measuredOptions, measuredReport);
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

    // The same crossings as sign constraints: each states its own bound,
    // one from below and one from above, and together they leave the solve
    // exactly the step the enclosure states.
    ScaffoldMeasuredOptions boundOptions = unitWeights();
    boundOptions.enclosureWeight = 0.;
    boundOptions.signWeight = 10.;
    ScaffoldMeasuredReport boundReport;
    const auto bounds =
        scaffoldMeasuredConstraints(scaffold.graph, geometry, events, kOutwardSigns, boundOptions, boundReport);
    CHECK(boundReport.signConstraints == std::int64_t(bounds.size()));
    CHECK(boundReport.enclosureArcs == 0);
    std::int64_t atLeast = 0, atMost = 0;
    for (const auto& bound : bounds) {
        CHECK(bound.weight == doctest::Approx(10.));
        CHECK(bound.a == measured.front().a);
        CHECK(bound.b == measured.front().b);
        if (bound.kind == ScaffoldConstraintKind::SignAtLeast) ++atLeast;
        else if (bound.kind == ScaffoldConstraintKind::SignAtMost) ++atMost;
        else FAIL("a sign weight states only sign constraints here");
    }
    CHECK(atLeast >= 1);
    CHECK(atMost >= 1);
    ScaffoldSolveOptions solveOptions;
    solveOptions.links.useLinks = false;
    solveOptions.links.approvalStates.clear();
    const auto solved = solveScaffoldWinding(scaffold.graph, geometry, solveOptions, bounds);
    CHECK(solved.violatedConstraints == 0);
    CHECK(solved.winding[std::size_t(measured.front().b)] - solved.winding[std::size_t(measured.front().a)] ==
          measured.front().target);
}

TEST_CASE("the sign solve decides the gauge of a fiber no anchor fixed")
{
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(260., 45., -150., 150., 30));
    scaffold.faceOutward(horizontal);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry, scaffold.facing());
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
    const Point radial{std::cos(radians(45.)), std::sin(radians(45.)), 0.};
    // A straight V fiber's gauge direction is arbitrary within the plane
    // perpendicular to it - only its sign is decided here, and the decided
    // one points to the outward half space.
    CHECK(dot(*solved, radial) * double(signs.sign[vertical]) > 0.5);
    CHECK(signs.fixedDisagreements == 0);
}

TEST_CASE("the sign stage falls back to the anchors when it is switched off")
{
    Scaffold scaffold;
    const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
    scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(260., 45., -150., 150., 30));
    scaffold.faceOutward(horizontal);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry, scaffold.facing());
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
    // The measured relation is the link's own encounter: the link stands
    // alone, with its step, and the crossing is dropped.
    CHECK(result.linkConstraints == 1);
    CHECK(result.linkReport.linksWithCrossing == 1);
    CHECK(result.linkReport.deduplicatedCrossings == 1);
    CHECK(result.constraintsByKind[std::size_t(ScaffoldConstraintKind::Perpendicular)] == 1);
    CHECK(result.objective == doctest::Approx(0.));
    CHECK(result.components == 1);
    CHECK(result.winding[horizontal] == 1);
    CHECK(result.winding[vertical] == 0);
    // One encounter, one relation: the arc carries the link's weight alone.
    REQUIRE(result.arcs.size() == 1);
    CHECK(result.arcs.front().weight == doctest::Approx(1.));
}

TEST_CASE("a link whose order the normals contradict accuses both fibers of a flip")
{
    // The V fiber lies inside the H fiber (radius 260 against 300) and both
    // carry an outward display normal. An adjacent link states that order; an
    // ordinary link states the opposite one.
    for (const bool adjacent : {true, false}) {
        Scaffold scaffold;
        const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
        const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(260., 45., -150., 150., 30));
        scaffold.faceOutward(horizontal);
        scaffold.faceOutward(vertical);
        scaffold.link(horizontal, vertical, polar(300., 45., 0.), polar(260., 45., 0.), adjacent);
        const ScaffoldGeometry geometry(scaffold.graph);
        const ScaffoldNormals normals(scaffold.graph, geometry, scaffold.facing());
        ScaffoldSignOptions signOptions;
        signOptions.solve = false;
        const auto signs = scaffoldNormalSigns(scaffold.graph, normals, {}, signOptions);
        ScaffoldLinkArcOptions linkOptions;
        linkOptions.approvalStates = {"approved", "pending"};
        ScaffoldLinkReport linkReport;
        const auto constraints = scaffoldLinkConstraints(scaffold.graph, geometry, linkOptions, linkReport);
        REQUIRE(constraints.size() == 1);
        CHECK(constraints.front().kind == ScaffoldConstraintKind::Perpendicular);
        ScaffoldNormalFlipReport report;
        const auto flips = scaffoldLinkNormalFlips(scaffold.graph, geometry, normals, signs.sign, constraints, report);
        CHECK(report.checked == 1);
        CHECK(report.undetermined == 0);
        if (adjacent) {
            CHECK(report.agreed == 1);
            CHECK(flips.empty());
        } else {
            CHECK(report.disagreed == 1);
            // One suspect per fiber; nothing brackets either, so each names
            // the span around its anchor.
            REQUIRE(flips.size() == 2);
            for (const auto& flip : flips) {
                CHECK(flip.disagreeing == 1);
                CHECK(flip.agreeing == 0);
                CHECK_FALSE(flip.bracketed);
                CHECK(flip.piece.controlTo > flip.piece.controlFrom);
            }
            CHECK(flips[0].fiber != flips[1].fiber);
        }
    }
}

TEST_CASE("a scaffold's measured relations are the ones the bulk assembly builds")
{
    // The per-event path and the bulk preparation share one relation helper
    // (vc_object_store FiberWindingGraph.hpp). Feeding both the same
    // crossings must give the same arcs, weights and enclosure relations.
    Scaffold scaffold;
    const auto spiral = scaffold.fiber(ScaffoldFamily::Horizontal, spiralLine(250., 350., 200., 680., 0., 260));
    const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(300., 250., -150., 150., 30));
    scaffold.faceOutward(spiral);
    scaffold.faceOutward(vertical);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry, scaffold.facing());

    ScaffoldEventReport report;
    ScaffoldEventOptions eventOptions;
    const auto events = scaffoldEvents(scaffold.graph, geometry, normals, eventOptions, constantDistance(0.75), report);
    REQUIRE(report.events >= 2);

    ScaffoldMeasuredOptions measuredOptions = unitWeights();
    measuredOptions.perpendicularWeight = 2.;
    measuredOptions.parallelWeight = 3.;
    ScaffoldMeasuredReport measuredReport;
    const auto measured =
        scaffoldMeasuredConstraints(scaffold.graph, geometry, events, kOutwardSigns, measuredOptions, measuredReport);

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
    input.normalSign = kFallingSigns;
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
    scaffold.faceOutward(horizontal);
    scaffold.faceOutward(vertical);
    const ScaffoldGeometry geometry(scaffold.graph);
    const ScaffoldNormals normals(scaffold.graph, geometry, scaffold.facing());
    ScaffoldEventReport report;
    const auto events = scaffoldEvents(scaffold.graph, geometry, normals, {}, constantDistance(1.), report);
    ScaffoldMeasuredReport measuredReport;
    auto measured = scaffoldMeasuredConstraints(scaffold.graph, geometry, events, kOutwardSigns, unitWeights(), measuredReport);
    REQUIRE_FALSE(measured.empty());
    for (const auto& constraint : measured) CHECK_FALSE(constraint.approved);

    const auto row = approveScaffoldConstraint(scaffold.graph, geometry, measured.front(), "alice",
                                               "2026-11-17T10:00:00Z", "host-a");
    const auto identity = scaffoldConstraintIdentity(scaffold.graph, measured.front());
    CHECK(row.identity == identity);
    CHECK(row.identity.fiberA < row.identity.fiberB);

    ScaffoldApprovalReport approvals;
    // The row alone approves nothing while the pieces it lies on are not
    // approved geometry.
    applyScaffoldConstraintApprovals(scaffold.graph, geometry, measured, {&row, 1}, approvals);
    CHECK(approvals.live == 1);
    CHECK(approvals.approved == 0);
    CHECK(approvals.withoutPieces >= 1);
    CHECK_FALSE(measured.front().approved);

    // With both covering pieces approved it applies.
    const auto covering = scaffoldConstraintSpans(scaffold.graph, geometry, measured.front());
    REQUIRE(covering.has_value());
    const auto approveSpan = [&](std::int64_t fiber, std::size_t span) {
        scaffold.graph.fibers[std::size_t(fiber)]
            .approvals.approvedSpans[std::size_t(vc::objects::FiberPieceAspect::Geometry)]
            .push_back(span);
    };
    approveSpan(measured.front().a, covering->spanA);
    approveSpan(measured.front().b, covering->spanB);
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

TEST_CASE("the neighbour normals face a fiber whose normals are approved, and an adjacent link does")
{
    for (const double facing : {1., -1.}) {
        Scaffold scaffold;
        const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
        scaffold.approveNormals(horizontal, 0, 40, [](std::size_t, const Point& at) { return tilted(at); });
        const ScaffoldGeometry geometry(scaffold.graph);
        ScaffoldNormalOptions options;
        options.neighbours.identity = "test field";
        options.neighbours.volume = "test-volume";
        const auto field = radialField(facing);
        options.neighbours.field = &field;
        const ScaffoldNormals normals(scaffold.graph, geometry, options);
        CHECK(normals.signSources()[horizontal] == ScaffoldSignSource::Neighbours);
        const auto solved = normals.at(horizontal, geometry.controlArc(horizontal, 2));
        REQUIRE(solved);
        // The approved axis, faced the way the neighbours face.
        const auto expected = tilted(scaffold.graph.fibers[horizontal].content.controlPoints[2]);
        CHECK(facing * dot(*solved, expected) > 0.999);
    }
    {
        Scaffold scaffold;
        const auto horizontal = scaffold.fiber(ScaffoldFamily::Horizontal, arcLine(300., 20., 70., 0., 40));
        const auto vertical = scaffold.fiber(ScaffoldFamily::Vertical, verticalLine(250., 45., -150., 150., 30));
        // Stored facing inward; the link says which side is outward.
        scaffold.approveNormals(horizontal, 0, 40, [](std::size_t, const Point& at) {
            const auto normal = tilted(at);
            return Point{-normal[0], -normal[1], -normal[2]};
        });
        scaffold.link(horizontal, vertical, polar(300., 45., 0.), polar(250., 45., 0.), true);
        const ScaffoldGeometry geometry(scaffold.graph);
        const ScaffoldNormals normals(scaffold.graph, geometry);
        CHECK(normals.signSources()[horizontal] == ScaffoldSignSource::AdjacentLink);
        CHECK(normals.fixedSigns()[horizontal] == 1);
        const auto solved = normals.at(horizontal, geometry.project(horizontal, polar(300., 45., 0.)).arc);
        REQUIRE(solved);
        CHECK(dot(*solved, tilted(polar(300., 45., 0.))) > 0.999);
    }
}
