// Coverage for apps/VC3D/FiberMapAutoLink.cpp (the Fiber Map's "Link A and
// B" geometry) and for the layout outputs it rests on: the raw arclength
// correspondence (Run::arclengths, PlacedFiber::controlArclength,
// unrolledLineArclengths), the drawn meeting of two fibers, the exact
// inversion to a source line position, the bounded closest 3D pass and the
// control-point choice.

#include <QtTest/QtTest>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "FiberMapAutoLink.hpp"
#include "FiberNetworkLayout.hpp"

using vc3d::fiber_map::GlobalLayoutParams;
using vc3d::fiber_map::GlobalPlacedFiber;
using vc3d::fiber_map::GlobalResult;
using vc3d::fiber_map::InputFiber;
using vc3d::fiber_map::PlacedFiber;
using vc3d::fiber_map::Run;
namespace autolink = vc3d::fiber_map::autolink;

namespace
{

constexpr double kTwoPi = 2.0 * M_PI;
constexpr double kVxPerCm = 10000.0 / 2.4;

constexpr double vx(double centimetres)
{
    return centimetres * kVxPerCm;
}

std::vector<cv::Vec3f> straightUmbilicus(int zMax)
{
    std::vector<cv::Vec3f> centers;
    for (int z = 0; z <= zMax; ++z) {
        centers.push_back(cv::Vec3f(0.0f, 0.0f, static_cast<float>(z)));
    }
    return centers;
}

InputFiber makeFiber(uint64_t id, const QString& label, char hvTag,
                     std::vector<cv::Vec3d> linePoints, const std::vector<int>& controlIndices)
{
    InputFiber fiber;
    fiber.id = id;
    fiber.fileName = label.toStdString() + ".json";
    fiber.label = label;
    fiber.hvTag = hvTag;
    fiber.linePoints = std::move(linePoints);
    for (int index : controlIndices) {
        if (index < 0 || static_cast<std::size_t>(index) >= fiber.linePoints.size()) {
            throw std::out_of_range("makeFiber: control index out of range");
        }
        fiber.controlPoints.push_back(fiber.linePoints[static_cast<std::size_t>(index)]);
    }
    if (fiber.controlPoints.size() > 1) {
        fiber.tracedSegments.assign(fiber.controlPoints.size() - 1, true);
    }
    return fiber;
}

GlobalLayoutParams defaultParams()
{
    GlobalLayoutParams params;
    params.smoothVx = 0.0;
    params.resampleStepVx = vx(0.025);
    params.minPadXVx = vx(2.2);
    params.minPadYVx = vx(1.6);
    return params;
}

// A two-turn H helix (radius growing by `radiusPerTurn` per turn) sampled
// every `step` radians, and a V fiber at `angle` touching the OUTER turn.
std::vector<cv::Vec3d> helix(double radius0, double radiusPerTurn, double turns, double z,
                             double step)
{
    std::vector<cv::Vec3d> points;
    const int count = static_cast<int>(std::floor(turns * kTwoPi / step)) + 1;
    for (int i = 0; i < count; ++i) {
        const double theta = static_cast<double>(i) * step;
        const double r = radius0 + radiusPerTurn * theta / kTwoPi;
        points.push_back(cv::Vec3d(r * std::cos(theta), r * std::sin(theta), z));
    }
    return points;
}

std::vector<cv::Vec3d> vertical(double theta, double radius, double zBegin, double zEnd,
                                double zStep)
{
    std::vector<cv::Vec3d> points;
    const int count = static_cast<int>(std::floor((zEnd - zBegin) / zStep)) + 1;
    for (int i = 0; i < count; ++i) {
        const double z = zBegin + static_cast<double>(i) * zStep;
        points.push_back(cv::Vec3d(radius * std::cos(theta), radius * std::sin(theta), z));
    }
    return points;
}

const GlobalPlacedFiber* placedById(const GlobalResult& result, uint64_t id)
{
    for (const GlobalPlacedFiber& placed : result.fibers) {
        if (placed.fiber.id == id) {
            return &placed;
        }
    }
    return nullptr;
}

PlacedFiber straightRun(const std::vector<QPointF>& points, const std::vector<double>& arclengths)
{
    PlacedFiber fiber;
    Run run;
    run.firstControl = 0;
    run.lastControl = 1;
    run.points = points;
    run.arclengths = arclengths;
    fiber.runs.push_back(run);
    fiber.controlPoints = {points.front(), points.back()};
    fiber.controlArclength = {arclengths.front(), arclengths.back()};
    return fiber;
}

std::vector<double> arcs(const std::vector<QPointF>& points)
{
    std::vector<double> s(points.size(), 0.0);
    for (std::size_t i = 1; i < points.size(); ++i) {
        const QPointF d = points[i] - points[i - 1];
        s[i] = s[i - 1] + std::sqrt(QPointF::dotProduct(d, d));
    }
    return s;
}

}  // namespace

class TestFiberMapAutoLink : public QObject
{
    Q_OBJECT

private slots:
    // The arclength the layout resampled on is what unrolledLineArclengths
    // recomputes from the stored inputs, bitwise, and every drawn sample's
    // arclength is one of that parameterisation.
    void unrolledArclengthMatchesTheLayout()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers;
        fibers.push_back(makeFiber(1, QStringLiteral("h"), 'H',
                                   helix(4000.0, 300.0, 1.5, 30000.0, kTwoPi / 1256.0),
                                   {0, 600, 1200, 1800}));
        fibers.push_back(makeFiber(2, QStringLiteral("v"), 'V',
                                   vertical(0.3 * kTwoPi, 4000.0 + 300.0 * 0.3, 29000.0,
                                            31000.0, 25.0),
                                   {0, 40, 80}));
        // Smoothing ON (the production default, 1.2 mm) and a styled run
        // (a gap span): the correspondence must hold through the resampling
        // and through displayRunPoints' endpoint replacement.
        GlobalLayoutParams params = defaultParams();
        params.smoothVx = vx(0.12);
        fibers[0].gapSegments = {false, true, false};
        const GlobalResult result =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params);
        const GlobalPlacedFiber* h = placedById(result, 1);
        QVERIFY(h != nullptr);
        const std::vector<double> raw = vc3d::fiber_map::unrolledLineArclengths(
            fibers[0].linePoints, umbilicus, h->fiber.thetaScale, h->fiber.offsetRad,
            result.rRefVx);
        QCOMPARE(raw.size(), fibers[0].linePoints.size());
        QVERIFY(std::is_sorted(raw.begin(), raw.end()));
        // Every control's arclength is BITWISE the raw arclength of its line
        // point (buildFiberGeometry reads controls off at rawArclength[index]).
        const auto sameBits = [](double x, double y) {
            return std::memcmp(&x, &y, sizeof(double)) == 0;
        };
        QCOMPARE(h->fiber.controlArclength.size(), std::size_t{4});
        QVERIFY(sameBits(h->fiber.controlArclength[0], raw[0]));
        QVERIFY(sameBits(h->fiber.controlArclength[1], raw[600]));
        QVERIFY(sameBits(h->fiber.controlArclength[2], raw[1200]));
        QVERIFY(sameBits(h->fiber.controlArclength[3], raw[1800]));
        // Every run carries an arclength per drawn sample, monotone, and a
        // drawn sample's point is where the unrolled source line is at that
        // arclength (within the smoothing's own displacement: the resampled
        // curve is parameterised by the raw arclength).
        QVERIFY(h->fiber.runs.size() >= 3);  // traced, gap, traced
        const std::vector<QPointF> unrolled = [&]() {
            std::vector<QPointF> points;
            for (std::size_t i = 0; i < raw.size(); ++i) {
                const cv::Vec3d& p = fibers[0].linePoints[i];
                const double theta = std::atan2(p[1], p[0]);
                (void)theta;
                points.push_back(QPointF(0.0, p[2]));
            }
            return points;
        }();
        for (std::size_t runIndex = 0; runIndex < h->fiber.runs.size(); ++runIndex) {
            const Run& run = h->fiber.runs[runIndex];
            QCOMPARE(run.arclengths.size(), run.points.size());
            QVERIFY(std::is_sorted(run.arclengths.begin(), run.arclengths.end()));
            for (std::size_t k = 0; k < run.points.size(); k += 7) {
                const double position = autolink::sourcePositionAtArclength(raw, run.arclengths[k]);
                const std::size_t lo = static_cast<std::size_t>(std::floor(position));
                const std::size_t hi = std::min(lo + 1, raw.size() - 1);
                // The inverted position brackets the sample's arclength ...
                QVERIFY(raw[lo] <= run.arclengths[k] + 1e-9);
                QVERIFY(run.arclengths[k] <= raw[hi] + 1e-9);
                // ... and the drawn sample's height is the source line's
                // height there (y = z is not smoothed away: the helix is
                // flat, so every sample sits at the fiber's z).
                QVERIFY(std::abs(run.points[k].y() - unrolled[lo].y()) < 1e-6);
            }
            // displayRunPoints replaces the overlap samples of a styled run
            // (and its neighbours) by the bounding controls' drawn
            // positions; the locator maps those to controlArclength.
            const std::vector<QPointF> display = vc3d::fiber_map::displayRunPoints(h->fiber, runIndex);
            QCOMPARE(display.size(), run.points.size());
            if (run.gap) {
                QCOMPARE(display.front(), h->fiber.controlPoints[static_cast<std::size_t>(run.firstControl)]);
                QCOMPARE(display.back(), h->fiber.controlPoints[static_cast<std::size_t>(run.lastControl)]);
            }
        }
        // The drawn meeting of the V fiber (at 0.3 turn) with the helix
        // inverts to a source position at 0.3 turn on one of the helix's
        // turns (1256 samples per turn): the locator's arclength
        // correspondence closes the loop from the drawn crossing back to the
        // line index.
        const GlobalPlacedFiber* v = placedById(result, 2);
        QVERIFY(v != nullptr);
        const auto meeting = autolink::findMapMeeting(h->fiber, v->fiber, QPointF(0.0, 0.0),
                                                      vx(0.1));
        QVERIFY(meeting.has_value());
        const double position = autolink::sourcePositionAtArclength(raw, meeting->a.arclength);
        const double turnPhase = std::fmod(position, 1256.0);
        QVERIFY(std::abs(turnPhase - 0.3 * 1256.0) < 40.0);
    }

    // Interior correspondence, exactly: with smoothing off the drawn
    // samples are the unrolled source line resampled by raw arclength, so a
    // sample's point is the unrolled line interpolated at its inverted
    // position (varying z, so the height is not a constant).
    void drawnSamplesAreTheUnrolledLineAtTheirInvertedPosition()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const double step = kTwoPi / 1256.0;
        std::vector<cv::Vec3d> points;
        for (int i = 0; i <= 1800; ++i) {
            const double theta = static_cast<double>(i) * step;
            const double r = 4000.0 + 300.0 * theta / kTwoPi;
            points.push_back(cv::Vec3d(r * std::cos(theta), r * std::sin(theta),
                                       30000.0 + 400.0 * std::sin(3.0 * theta)));
        }
        std::vector<InputFiber> fibers;
        fibers.push_back(makeFiber(1, QStringLiteral("h"), 'H', points, {0, 900, 1800}));
        fibers.push_back(makeFiber(2, QStringLiteral("v"), 'V',
                                   vertical(0.3 * kTwoPi, 4000.0 + 300.0 * 0.3, 29000.0,
                                            31000.0, 25.0),
                                   {0, 40, 80}));
        const GlobalResult result =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        const GlobalPlacedFiber* h = placedById(result, 1);
        QVERIFY(h != nullptr);
        const std::vector<double> raw = vc3d::fiber_map::unrolledLineArclengths(
            points, umbilicus, h->fiber.thetaScale, h->fiber.offsetRad, result.rRefVx);
        // The unrolled source polyline, as the layout built it.
        std::vector<QPointF> unrolled;
        for (std::size_t i = 0; i < points.size(); ++i) {
            const double theta = static_cast<double>(i) * step;  // unwrapped atan2 of the helix
            unrolled.push_back(QPointF((h->fiber.thetaScale * theta + h->fiber.offsetRad) * result.rRefVx,
                                       points[i][2]));
        }
        int checked = 0;
        for (const Run& run : h->fiber.runs) {
            for (std::size_t k = 0; k < run.points.size(); k += 5) {
                const double position = autolink::sourcePositionAtArclength(raw, run.arclengths[k]);
                const std::size_t lo = static_cast<std::size_t>(std::floor(position));
                const std::size_t hi = std::min(lo + 1, unrolled.size() - 1);
                const double t = position - static_cast<double>(lo);
                const QPointF expected = unrolled[lo] * (1.0 - t) + unrolled[hi] * t;
                QVERIFY(std::abs(expected.x() - run.points[k].x()) < 1e-6 * std::abs(expected.x()) + 1e-6);
                QVERIFY(std::abs(expected.y() - run.points[k].y()) < 1e-6);
                ++checked;
            }
        }
        QVERIFY(checked > 50);
    }

    // The per-endpoint review state the map draws from: every link of the
    // inputs (a link to an unplaceable one-point seed included), pending
    // over approved, automatic only without a human pending link, the
    // same-H/V warning of the winning state, the adjacent shape.
    void endpointReviewsAggregateEveryLink()
    {
        InputFiber h;
        h.id = 1;
        h.hvTag = 'H';
        h.controlPoints = {{0, 0, 0}, {10, 0, 0}, {20, 0, 0}};
        InputFiber v;
        v.id = 2;
        v.hvTag = 'V';
        InputFiber seed;
        seed.id = 3;
        seed.hvTag = '?';
        InputFiber h2;
        h2.id = 4;
        h2.hvTag = 'H';
        // CP0: automatic to V + human pending to the seed -> pending, not auto.
        h.links.push_back({0, 2, 0, true, false, true, true});
        h.links.push_back({0, 3, 0, true, false, true, false});
        // CP1: automatic to V + approved to H2 -> pending, auto, no warning.
        h.links.push_back({1, 2, 1, true, false, true, true});
        h.links.push_back({1, 4, 0, false, false, true, false});
        // CP2: automatic H-V + pending H-H -> pending, auto, same-H/V warning;
        // plus an approved adjacent link -> triangle.
        h.links.push_back({2, 2, 2, true, false, true, true});
        h.links.push_back({2, 4, 1, true, false, true, true});
        h.links.push_back({2, 4, 2, false, true, true, false});
        const auto reviews = vc3d::fiber_map::aggregateEndpointReviews({h, v, seed, h2});
        const auto at = [&](int cp) { return reviews.at(vc3d::fiber_map::EndpointKey{1, cp}); };
        QVERIFY(at(0).pending);
        QVERIFY(!at(0).autoPending);
        QVERIFY(at(0).anyAuto);
        QVERIFY(at(1).pending);
        QVERIFY(at(1).autoPending);
        QVERIFY(!at(1).sameHv);
        QVERIFY(!at(1).adjacent);
        QVERIFY(at(2).autoPending);
        QVERIFY(at(2).sameHv);
        QVERIFY(at(2).adjacent);
        QVERIFY(reviews.find(vc3d::fiber_map::EndpointKey{1, 5}) == reviews.end());
    }

    void inversionIsExactFirstMatchAndInterpolates()
    {
        // Round-3/4 counterexample: projected arcs with a radial run.
        const std::vector<double> raw{0.0, 0.1, 0.2, 0.3, 100.0, 200.0, 1000.0};
        QVERIFY(std::abs(autolink::sourcePositionAtArclength(raw, 1.0) - 3.00702) < 1e-4);
        // Plateau: first exact match; beyond it, interpolation.
        const std::vector<double> plateau{0.0, 1.0, 1.0, 1.0, 2.0};
        QCOMPARE(autolink::sourcePositionAtArclength(plateau, 1.0), 1.0);
        QCOMPARE(autolink::sourcePositionAtArclength(plateau, 1.5), 3.5);
        // Near-plateau: no tolerance snapping.
        const std::vector<double> near{0.0, 1.0, 1.00000000001, 2.0};
        QVERIFY(std::abs(autolink::sourcePositionAtArclength(near, 1.000000000005) - 1.5) < 1e-3);
        QCOMPARE(autolink::sourcePositionAtArclength(raw, -5.0), 0.0);
        QCOMPARE(autolink::sourcePositionAtArclength(raw, 5000.0), 6.0);
    }

    void meetingPrefersTheCrossingNearestTheClick()
    {
        // A straight horizontal fiber and a zigzag crossing it twice.
        const std::vector<QPointF> horizontal{{0.0, 0.0}, {100.0, 0.0}, {200.0, 0.0},
                                              {300.0, 0.0}, {400.0, 0.0}};
        const std::vector<QPointF> zigzag{{50.0, -50.0}, {150.0, 50.0}, {250.0, -50.0},
                                          {350.0, 50.0}};
        const PlacedFiber a = straightRun(horizontal, arcs(horizontal));
        const PlacedFiber b = straightRun(zigzag, arcs(zigzag));
        const auto nearLeft = autolink::findMapMeeting(a, b, QPointF(90.0, 10.0), 10.0);
        QVERIFY(nearLeft.has_value());
        QVERIFY(nearLeft->crossing);
        QVERIFY(std::abs(nearLeft->point.x() - 100.0) < 1e-9);
        QVERIFY(std::abs(nearLeft->a.arclength - 100.0) < 1e-9);
        const auto nearRight = autolink::findMapMeeting(a, b, QPointF(310.0, 10.0), 10.0);
        QVERIFY(nearRight.has_value());
        QVERIFY(std::abs(nearRight->point.x() - 300.0) < 1e-9);
        QVERIFY(std::abs(nearRight->a.arclength - 300.0) < 1e-9);
        // A crossing exactly at a vertex of both polylines is found.
        const std::vector<QPointF> throughVertex{{200.0, -50.0}, {200.0, 0.0}, {200.0, 50.0}};
        const PlacedFiber c = straightRun(throughVertex, arcs(throughVertex));
        const auto atVertex = autolink::findMapMeeting(a, c, QPointF(0.0, 0.0), 10.0);
        QVERIFY(atVertex.has_value());
        QVERIFY(atVertex->crossing);
        QVERIFY(std::abs(atVertex->point.x() - 200.0) < 1e-9);
        QVERIFY(std::abs(atVertex->b.arclength - 50.0) < 1e-9);
    }

    void meetingFallsBackToTheClosestApproachWithinTheLimit()
    {
        const std::vector<QPointF> lower{{0.0, 0.0}, {100.0, 0.0}, {200.0, 0.0}};
        const std::vector<QPointF> upper{{0.0, 30.0}, {100.0, 8.0}, {200.0, 30.0}};
        const PlacedFiber a = straightRun(lower, arcs(lower));
        const PlacedFiber b = straightRun(upper, arcs(upper));
        const auto within = autolink::findMapMeeting(a, b, QPointF(0.0, 0.0), 10.0);
        QVERIFY(within.has_value());
        QVERIFY(!within->crossing);
        QVERIFY(std::abs(within->distance - 8.0) < 1e-9);
        QVERIFY(std::abs(within->point.x() - 100.0) < 1e-9);
        const auto beyond = autolink::findMapMeeting(a, b, QPointF(0.0, 0.0), 5.0);
        QVERIFY(!beyond.has_value());
        // The limit is inclusive: exactly 8 apart is accepted at 8.
        const auto exact = autolink::findMapMeeting(a, b, QPointF(0.0, 0.0), 8.0);
        QVERIFY(exact.has_value());
        QVERIFY(std::abs(exact->distance - 8.0) < 1e-9);
        // A shallow crossing of two long segments is a crossing (x = 500),
        // not a collinear overlap following the click.
        const std::vector<QPointF> flat{{0.0, 0.0}, {1000.0, 0.0}};
        const std::vector<QPointF> shallow{{0.0, -1e-7}, {1000.0, 1e-7}};
        const PlacedFiber f1 = straightRun(flat, arcs(flat));
        const PlacedFiber f2 = straightRun(shallow, arcs(shallow));
        const auto shallowHit = autolink::findMapMeeting(f1, f2, QPointF(900.0, 0.0), 10.0);
        QVERIFY(shallowHit.has_value());
        QVERIFY(shallowHit->crossing);
        QVERIFY(std::abs(shallowHit->point.x() - 500.0) < 1e-3);
        // ... and so is a 2e-9 tilt over 10000 voxels far from the origin.
        const std::vector<QPointF> far1{{0.0, 30000.0}, {10000.0, 30000.0}};
        const std::vector<QPointF> far2{{0.0, 30000.0 - 2e-9}, {10000.0, 30000.0 + 2e-9}};
        const PlacedFiber g1 = straightRun(far1, arcs(far1));
        const PlacedFiber g2 = straightRun(far2, arcs(far2));
        const auto farHit = autolink::findMapMeeting(g1, g2, QPointF(9000.0, 30000.0), 10.0);
        QVERIFY(farHit.has_value());
        QVERIFY(farHit->crossing);
        QVERIFY(std::abs(farHit->point.x() - 5000.0) < 1.0);
        // A collinear overlap follows the click.
        const std::vector<QPointF> same{{0.0, 0.0}, {200.0, 0.0}};
        const PlacedFiber c = straightRun(same, arcs(same));
        const auto collinear = autolink::findMapMeeting(a, c, QPointF(90.0, 5.0), 10.0);
        QVERIFY(collinear.has_value());
        QVERIFY(collinear->crossing);
        QVERIFY(std::abs(collinear->point.x() - 90.0) < 1e-9);
        QVERIFY(std::abs(collinear->a.arclength - 90.0) < 1e-9);
        QVERIFY(std::abs(collinear->b.arclength - 90.0) < 1e-9);
    }

    // Near-parallel 3D segments: the closest pass is the unique shallow
    // crossing at x = 500, found without the a*e - b*b cancellation.
    void closestPassIsStableForNearParallelSegments()
    {
        const std::vector<cv::Vec3d> a{{0.0, 0.0, 0.0}, {1000.0, 0.0, 0.0}};
        const std::vector<cv::Vec3d> b{{0.0, -5e-6, 0.0}, {1000.0, 5e-6, 0.0}};
        const std::vector<double> ca = autolink::cumulativeArclengths(a);
        const std::vector<double> cb = autolink::cumulativeArclengths(b);
        autolink::FiberSide sideA{&a, &ca, 0.5, 10000.0};
        autolink::FiberSide sideB{&b, &cb, 0.5, 10000.0};
        const auto pass = autolink::findClosestPass(sideA, sideB);
        QVERIFY(pass.has_value());
        QVERIFY(std::abs(pass->pointA[0] - 500.0) < 1e-3);
        QVERIFY(pass->distance < 1e-9);
        // Rotated and offset: A along the diagonal, B nearly parallel to it,
        // crossing near x = 1300 (B's line passes 1e-6 below A's start and
        // 1e-6 above its end).
        const std::vector<cv::Vec3d> ra{{1000.0, 1000.0, 100.0}, {2000.0, 2000.0, 100.0}};
        const std::vector<cv::Vec3d> rb{{800.0, 799.999999, 100.0}, {1800.0, 1800.000001, 100.0}};
        const std::vector<double> cra = autolink::cumulativeArclengths(ra);
        const std::vector<double> crb = autolink::cumulativeArclengths(rb);
        autolink::FiberSide rsA{&ra, &cra, 0.5, 10000.0};
        autolink::FiberSide rsB{&rb, &crb, 0.5, 10000.0};
        const auto rotated = autolink::findClosestPass(rsA, rsB);
        QVERIFY(rotated.has_value());
        QVERIFY(std::abs(rotated->pointA[0] - 1300.0) < 1.0);
        QVERIFY(rotated->distance < 1e-6);
    }

    // Translated coordinates: the focus entry reconstructs more than a few
    // ulps outside; the doubling walk still finds the first position inside.
    void placementFindsTheFocusEntryAtLargeCoordinates()
    {
        const std::vector<cv::Vec3d> segment{{7110.6, 0.0, 0.0}, {7917.320000000001, 0.0, 0.0}};
        const std::vector<double> arcs = autolink::cumulativeArclengths(segment);
        autolink::ControlChoiceInput input;
        input.linePoints = &segment;
        input.cumulative = &arcs;
        input.controlLineIndex = {0, 1};
        input.linked = {true, true};
        input.kollesisTerminations = {false, false};
        input.passPosition = 0.005;
        input.windowLo = 0.0;
        input.windowHi = 1.0;
        input.reuseVx = 0.0;
        input.minSpacingVx = 8.0;
        input.focusBox = std::make_pair(cv::Vec3d(7119.0, -1.0, -1.0), cv::Vec3d(7800.0, 1.0, 1.0));
        input.inFocusBounds = [](const cv::Vec3d& p) { return p[0] >= 7119.0 && p[0] <= 7800.0; };
        const auto choice = autolink::chooseControlPoint(input);
        QCOMPARE(static_cast<int>(choice.kind), static_cast<int>(autolink::ControlChoice::Kind::Place));
        QVERIFY(std::abs(choice.point[0] - 7119.0) < 1e-6);
        QVERIFY(choice.point[0] >= 7119.0);
    }

    // The focus entry a hair past a segment's start: the admissible
    // position (3e-12 along the first segment) is far below the
    // position's own ulp walk but within one coordinate ulp of the point.
    void placementFindsAFocusEntryAtTheSegmentStart()
    {
        const std::vector<cv::Vec3d> line{{7118.999999999, 0.0, 0.0}, {7419.99, 0.0, 0.0},
                                          {7519.99, 0.0, 0.0}};
        const std::vector<double> arcs = autolink::cumulativeArclengths(line);
        autolink::ControlChoiceInput input;
        input.linePoints = &line;
        input.cumulative = &arcs;
        input.controlLineIndex = {1, 2};
        input.linked = {true, true};
        input.kollesisTerminations = {false, false};
        input.passPosition = 0.0;
        input.windowLo = 0.0;
        input.windowHi = 1.0;
        input.reuseVx = 0.0;
        input.minSpacingVx = 8.0;
        input.focusBox = std::make_pair(cv::Vec3d(7119.0, -1.0, -1.0), cv::Vec3d(7400.0, 1.0, 1.0));
        input.inFocusBounds = [](const cv::Vec3d& p) { return p[0] >= 7119.0 && p[0] <= 7400.0; };
        const auto choice = autolink::chooseControlPoint(input);
        QCOMPARE(static_cast<int>(choice.kind), static_cast<int>(autolink::ControlChoice::Kind::Place));
        QVERIFY(std::abs(choice.point[0] - 7119.0) < 1e-6);
        QVERIFY(choice.point[0] >= 7119.0);
    }

    // A line that returns to within the ambiguity tolerance of a distant
    // control: the admissible stretch begins a hair along the line, not at
    // the far side of the window; and equidistant candidates choose the
    // lower line position.
    void placementLeavesTheAmbiguitySphereAndBreaksTiesDeterministically()
    {
        const std::vector<cv::Vec3d> line{{0.0, 0.0, 0.0},    {100.0, 0.0, 0.0}, {100.0, 100.0, 0.0},
                                          {0.0, 100.0, 0.0},  {0.0, 0.0, 5e-7},  {-100.0, 0.0, 5e-7}};
        const std::vector<double> arcs = autolink::cumulativeArclengths(line);
        autolink::ControlChoiceInput input;
        input.linePoints = &line;
        input.cumulative = &arcs;
        input.controlLineIndex = {0, 5};
        input.linked = {true, true};
        input.kollesisTerminations = {false, false};
        input.passPosition = 4.0;
        input.windowLo = 0.0;
        input.windowHi = 5.0;
        input.reuseVx = 0.0;
        input.minSpacingVx = 8.0;
        auto choice = autolink::chooseControlPoint(input);
        QCOMPARE(static_cast<int>(choice.kind), static_cast<int>(autolink::ControlChoice::Kind::Place));
        // The sphere's exits on the segment into the pass and on the one
        // out of it are equidistant: the lower position wins (a hair before
        // index 4), still a millionth of a voxel from the pass.
        QVERIFY(std::abs(choice.linePosition - 4.0) < 0.001);
        QVERIFY(std::abs(choice.point[0]) < 0.1 && std::abs(choice.point[1]) < 0.1);
        // The same with large coordinates: the sphere's chord must not
        // vanish against |m|^2 (the pass at 2.5 sits on a segment the line
        // runs back along, within the tolerance of control 0's position).
        const std::vector<cv::Vec3d> far{{2000.0, 2000.0, 0.0}, {3000.0, 3000.0, 0.0},
                                         {3000.0, 2000.0, 0.0}, {1000.0, 2000.0, 0.0},
                                         {1000.0, 3000.0, 0.0}};
        const std::vector<double> farArcs = autolink::cumulativeArclengths(far);
        autolink::ControlChoiceInput farInput;
        farInput.linePoints = &far;
        farInput.cumulative = &farArcs;
        farInput.controlLineIndex = {0, 4};
        farInput.linked = {true, true};
        farInput.kollesisTerminations = {false, false};
        farInput.passPosition = 2.5;
        farInput.windowLo = 2.0;
        farInput.windowHi = 3.0;
        farInput.reuseVx = 0.0;
        farInput.minSpacingVx = 8.0;
        const auto farChoice = autolink::chooseControlPoint(farInput);
        QCOMPARE(static_cast<int>(farChoice.kind), static_cast<int>(autolink::ControlChoice::Kind::Place));
        QVERIFY(std::abs(farChoice.linePosition - 2.5) < 1e-6);
        // Ties: a linked control at 64 on x = [0, 64, 128], spacing 8, pass
        // at 64 -> 56 and 72 are equidistant; the lower position wins.
        const std::vector<cv::Vec3d> sym{{0.0, 0.0, 0.0}, {64.0, 0.0, 0.0}, {128.0, 0.0, 0.0}};
        const std::vector<double> symArcs = autolink::cumulativeArclengths(sym);
        input.linePoints = &sym;
        input.cumulative = &symArcs;
        input.controlLineIndex = {1};
        input.linked = {true};
        input.kollesisTerminations = {false};
        input.passPosition = 1.0;
        input.windowLo = 0.0;
        input.windowHi = 2.0;
        choice = autolink::chooseControlPoint(input);
        QCOMPARE(static_cast<int>(choice.kind), static_cast<int>(autolink::ControlChoice::Kind::Place));
        QVERIFY(std::abs(choice.point[0] - 56.0) < 1e-9);
    }

    // A styled run's replaced overlap sample maps to its bounding control's
    // arclength: a crossing on the gap run's first display segment reports
    // the control's arclength, not the overlap sample's.
    void styledRunEndpointInvertsThroughTheControl()
    {
        PlacedFiber fiber;
        Run traced;
        traced.firstControl = 0;
        traced.lastControl = 1;
        traced.points = {{0.0, 0.0}, {50.0, 0.0}, {100.0, 0.0}, {110.0, 0.0}};
        traced.arclengths = {0.0, 50.0, 100.0, 110.0};
        Run gap;
        gap.gap = true;
        gap.firstControl = 1;
        gap.lastControl = 2;
        // The overlap sample sits 5e-9 from the control: fuzzy point
        // equality would call them equal, the style rule still replaces.
        gap.points = {{100.0 + 5e-9, 0.0}, {150.0, 0.0}, {200.0, 0.0}};
        gap.arclengths = {90.0, 150.0, 200.0};
        fiber.runs = {traced, gap};
        fiber.controlPoints = {{0.0, 0.0}, {100.0, 0.0}, {200.0, 0.0}};
        fiber.controlArclength = {0.0, 100.0, 200.0};
        const std::vector<QPointF> vertical{{120.0, -10.0}, {120.0, 10.0}};
        const PlacedFiber v = straightRun(vertical, arcs(vertical));
        const auto meeting = autolink::findMapMeeting(fiber, v, QPointF(120.0, 0.0), 1.0);
        QVERIFY(meeting.has_value());
        QVERIFY(meeting->crossing);
        // The hit is on the gap run's display segment [control 1 (100), 150]
        // at x = 120: arclength interpolates from the CONTROL's 100, not 90.
        QCOMPARE(meeting->a.runIndex, std::size_t{1});
        QVERIFY(std::abs(meeting->a.arclength - 120.0) < 1e-9);
    }

    // The window never leaves the drawn turn: a two-turn helix passes the V
    // fiber closer on its INNER turn, but the meeting lies on the outer one.
    void closestPassStaysOnTheMeetingTurn()
    {
        const double step = kTwoPi / 1256.0;
        const std::vector<cv::Vec3d> h = helix(4000.0, 300.0, 2.0, 30000.0, step);
        // V fiber at 0.3 turn, 10 vx inside the inner turn's radius: the inner
        // turn is 10 vx away, the outer turn 310 vx away.
        const double theta = 0.3 * kTwoPi;
        const std::vector<cv::Vec3d> v =
            vertical(theta, 4000.0 + 300.0 * 0.3 - 10.0, 29000.0, 31000.0, 25.0);
        const std::vector<double> hArcs = autolink::cumulativeArclengths(h);
        const std::vector<double> vArcs = autolink::cumulativeArclengths(v);
        // The meeting on the OUTER turn: theta + 2 pi.
        const double outerPosition = (theta + kTwoPi) / step;
        const double vPosition = 40.0;  // z = 30000
        autolink::FiberSide sideH{&h, &hArcs, outerPosition, vx(0.5)};
        autolink::FiberSide sideV{&v, &vArcs, vPosition, vx(0.5)};
        const auto pass = autolink::findClosestPass(sideH, sideV);
        QVERIFY(pass.has_value());
        QVERIFY(std::abs(pass->positionA - outerPosition) < 3.0);
        QVERIFY(std::abs(pass->distance - 310.0) < 5.0);
        QVERIFY(pass->firstA > kTwoPi / step * 0.9);  // never reaches the inner turn
        // With the meeting on the inner turn, the pass is the inner one.
        autolink::FiberSide sideInner{&h, &hArcs, theta / step, vx(0.5)};
        const auto inner = autolink::findClosestPass(sideInner, sideV);
        QVERIFY(inner.has_value());
        QVERIFY(std::abs(inner->distance - 10.0) < 5.0);
    }

    // The window never searches outside +-window: a long segment that
    // starts inside the window but runs far past it is clipped at the
    // exact arclength limit.
    void closestPassClipsBoundarySegments()
    {
        // Fiber A: x from 0 to 1000 in two long segments (0, 500, 1000).
        const std::vector<cv::Vec3d> a{{0.0, 0.0, 0.0}, {500.0, 0.0, 0.0}, {1000.0, 0.0, 0.0}};
        // Fiber B: a short vertical segment that touches A at x = 900.
        const std::vector<cv::Vec3d> b{{900.0, -10.0, 0.0}, {900.0, 10.0, 0.0}};
        const std::vector<double> ca = autolink::cumulativeArclengths(a);
        const std::vector<double> cb = autolink::cumulativeArclengths(b);
        // Meeting at x = 600 (position 1.2) with a +-100 window: [500, 700].
        // The closest point to B inside the window is x = 700, not 900.
        autolink::FiberSide sideA{&a, &ca, 1.2, 100.0};
        autolink::FiberSide sideB{&b, &cb, 0.5, 100.0};
        const auto pass = autolink::findClosestPass(sideA, sideB);
        QVERIFY(pass.has_value());
        QVERIFY(std::abs(pass->pointA[0] - 700.0) < 1e-6);
        QVERIFY(std::abs(pass->positionA - 1.4) < 1e-6);
        QVERIFY(std::abs(pass->distance - 200.0) < 1e-6);
    }

    // A pass inside a linked control's spacing band moves to the band's
    // boundary, a fractional position between two line vertices.
    void controlChoiceFindsPositionsBetweenVertices()
    {
        const std::vector<cv::Vec3d> line{{0.0, 0.0, 0.0}, {100.0, 0.0, 0.0}, {200.0, 0.0, 0.0}};
        const std::vector<double> cumulative = autolink::cumulativeArclengths(line);
        autolink::ControlChoiceInput input;
        input.linePoints = &line;
        input.cumulative = &cumulative;
        input.controlLineIndex = {0, 1};
        input.linked = {true, true};
        input.kollesisTerminations = {false, false};
        input.passPosition = 0.1;  // x = 10
        input.windowLo = 0;
        input.windowHi = 2;
        input.reuseVx = 0.0;
        input.minSpacingVx = 30.0;
        auto choice = autolink::chooseControlPoint(input);
        QCOMPARE(static_cast<int>(choice.kind), static_cast<int>(autolink::ControlChoice::Kind::Place));
        QVERIFY(choice.linePosition > 0.29 && choice.linePosition < 0.31);  // x = 30
        QVERIFY(std::abs(choice.point[0] - 30.0) < 1e-6);
        // The spacing boundary reconstructed through an arclength that comes
        // back a few ulps short: segment [0, 49] with linked endpoint
        // controls, spacing 8, pass at 4.9 -> the position with arclength 8
        // (8/49), not the far side.
        {
            const std::vector<cv::Vec3d> segment{{0.0, 0.0, 0.0}, {49.0, 0.0, 0.0}};
            const std::vector<double> segmentArcs = autolink::cumulativeArclengths(segment);
            autolink::ControlChoiceInput rounding;
            rounding.linePoints = &segment;
            rounding.cumulative = &segmentArcs;
            rounding.controlLineIndex = {0, 1};
            rounding.linked = {true, true};
            rounding.kollesisTerminations = {false, false};
            rounding.passPosition = 0.1;
            rounding.windowLo = 0.0;
            rounding.windowHi = 1.0;
            rounding.reuseVx = 0.0;
            rounding.minSpacingVx = 8.0;
            const auto near = autolink::chooseControlPoint(rounding);
            QCOMPARE(static_cast<int>(near.kind), static_cast<int>(autolink::ControlChoice::Kind::Place));
            QVERIFY(std::abs(near.point[0] - 8.0) < 1e-6);
            // The strict gate: the chosen position is never inside a band.
            QVERIFY(near.point[0] >= 8.0 - 1e-12);
            // The same with the focus box [8, 41]: the box's entry reconstructs
            // a few ulps outside; its representable neighbour at 8 is taken.
            rounding.focusBox = std::make_pair(cv::Vec3d(8.0, -1.0, -1.0), cv::Vec3d(41.0, 1.0, 1.0));
            rounding.inFocusBounds = [](const cv::Vec3d& p) { return p[0] >= 8.0 && p[0] <= 41.0; };
            const auto focused = autolink::chooseControlPoint(rounding);
            QCOMPARE(static_cast<int>(focused.kind), static_cast<int>(autolink::ControlChoice::Kind::Place));
            QVERIFY(std::abs(focused.point[0] - 8.0) < 1e-6);
            QVERIFY(focused.point[0] >= 8.0);
        }
        // A tight fit: linked controls at 0 and 60 with spacing 30 admit
        // exactly x = 30 (the boundary is admissible, no nudge).
        const std::vector<cv::Vec3d> tight{{0.0, 0.0, 0.0}, {60.0, 0.0, 0.0}, {120.0, 0.0, 0.0}};
        const std::vector<double> tightArcs = autolink::cumulativeArclengths(tight);
        input.linePoints = &tight;
        input.cumulative = &tightArcs;
        input.controlLineIndex = {0, 1};
        input.passPosition = 0.2;
        choice = autolink::chooseControlPoint(input);
        QCOMPARE(static_cast<int>(choice.kind), static_cast<int>(autolink::ControlChoice::Kind::Place));
        QVERIFY(std::abs(choice.point[0] - 30.0) < 1e-6);
        // A focus box strictly inside a segment: x in [40, 60] on 0..100 with
        // a linked control at 0 (spacing 30) and the pass at 10: the box's
        // entry at x = 40 is the nearest admissible position.
        input.linePoints = &line;
        input.cumulative = &cumulative;
        input.controlLineIndex = {0};
        input.linked = {true};
        input.kollesisTerminations = {false};
        input.passPosition = 0.1;
        input.windowLo = 0.0;
        input.windowHi = 2.0;
        input.focusBox = std::make_pair(cv::Vec3d(40.0, -1.0, -1.0), cv::Vec3d(60.0, 1.0, 1.0));
        input.inFocusBounds = [](const cv::Vec3d& p) { return p[0] >= 40.0 && p[0] <= 60.0; };
        choice = autolink::chooseControlPoint(input);
        QCOMPARE(static_cast<int>(choice.kind), static_cast<int>(autolink::ControlChoice::Kind::Place));
        QVERIFY(std::abs(choice.point[0] - 40.0) < 1e-6);
    }

    void controlChoiceReusesPlacesAndRefuses()
    {
        // A straight line along x, one point per voxel.
        std::vector<cv::Vec3d> line;
        for (int i = 0; i <= 2000; ++i) {
            line.push_back(cv::Vec3d(static_cast<double>(i), 0.0, 0.0));
        }
        const std::vector<double> cumulative = autolink::cumulativeArclengths(line);
        autolink::ControlChoiceInput input;
        input.linePoints = &line;
        input.cumulative = &cumulative;
        input.controlLineIndex = {100, 1000, 1900};
        input.linked = {false, false, false};
        input.kollesisTerminations = {false, false, false};
        input.passPosition = 1050.0;
        input.windowLo = 600;
        input.windowHi = 1500;
        input.reuseVx = 100.0;
        input.minSpacingVx = 30.0;
        // Reuse: the unlinked control 50 vx away.
        auto choice = autolink::chooseControlPoint(input);
        QCOMPARE(static_cast<int>(choice.kind), static_cast<int>(autolink::ControlChoice::Kind::Reuse));
        QCOMPARE(choice.controlIndex, 1);
        // Linked: place instead, at the pass.
        input.linked = {false, true, false};
        choice = autolink::chooseControlPoint(input);
        QCOMPARE(static_cast<int>(choice.kind), static_cast<int>(autolink::ControlChoice::Kind::Place));
        QCOMPARE(choice.linePosition, 1050.0);
        // Pass within the minimum spacing of the linked control: the new
        // control moves outward to the nearer admissible side.
        input.passPosition = 1010.0;
        choice = autolink::chooseControlPoint(input);
        QCOMPARE(static_cast<int>(choice.kind), static_cast<int>(autolink::ControlChoice::Kind::Place));
        QVERIFY(choice.linePosition >= 1030.0);
        QVERIFY(choice.linePosition < 1031.0);
        // ... unless that side is outside the focus bounds: then the farther side.
        input.inFocusBounds = [](const cv::Vec3d& p) { return p[0] < 1005.0; };
        choice = autolink::chooseControlPoint(input);
        QCOMPARE(static_cast<int>(choice.kind), static_cast<int>(autolink::ControlChoice::Kind::Place));
        QVERIFY(choice.linePosition <= 970.0);
        QVERIFY(choice.linePosition >= 969.0);
        // A saturated window refuses.
        input.inFocusBounds = [](const cv::Vec3d&) { return false; };
        choice = autolink::chooseControlPoint(input);
        QCOMPARE(static_cast<int>(choice.kind), static_cast<int>(autolink::ControlChoice::Kind::Refused));
        QVERIFY(!choice.reason.empty());
        // A gap span and a tagged end are closed to placement.
        input.inFocusBounds = nullptr;
        input.linked = {false, false, false};
        input.reuseVx = 0.0;
        input.gapSpans = {{0, 1}};
        input.passPosition = 500.0;
        input.windowLo = 400;
        input.windowHi = 600;
        choice = autolink::chooseControlPoint(input);
        QCOMPARE(static_cast<int>(choice.kind), static_cast<int>(autolink::ControlChoice::Kind::Refused));
        input.gapSpans.clear();
        input.kollesisTerminations = {false, false, true};
        input.passPosition = 1950.0;
        input.windowLo = 1920;
        input.windowHi = 2000;
        choice = autolink::chooseControlPoint(input);
        QCOMPARE(static_cast<int>(choice.kind), static_cast<int>(autolink::ControlChoice::Kind::Refused));
    }
};

QTEST_APPLESS_MAIN(TestFiberMapAutoLink)
#include "test_fiber_map_auto_link.moc"
