// Coverage for buildGlobalLayout in apps/VC3D/FiberNetworkLayout.cpp: the
// all-fibers map built on the winding solver. The solver's own arithmetic is
// covered by test_fiber_winding_solver; this asserts the layout contract on
// top of it - every fiber accounted for, links landing coincident, winding
// gridlines numbered by the winding coordinate, both chiralities.

#include <QtTest/QtTest>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "FiberNetworkLayout.hpp"

using vc3d::fiber_map::ContentDigest;
using vc3d::fiber_map::GlobalAnchor;
using vc3d::fiber_map::GlobalLayoutParams;
using vc3d::fiber_map::GlobalPlacedFiber;
using vc3d::fiber_map::GlobalResult;
using vc3d::fiber_map::InputFiber;
using vc3d::fiber_map::InputLink;
using vc3d::fiber_map::PlacedLink;

namespace
{

constexpr double kTwoPi = 2.0 * M_PI;
constexpr int kStepsPerTurn = 1256;
constexpr double kStep = kTwoPi / static_cast<double>(kStepsPerTurn);
constexpr double kVxPerCm = 10000.0 / 2.4;

constexpr double vx(double centimetres)
{
    return centimetres * kVxPerCm;
}

std::vector<cv::Vec3f> straightUmbilicus(int zMax)
{
    std::vector<cv::Vec3f> centers;
    centers.reserve(static_cast<std::size_t>(zMax) + 1);
    for (int z = 0; z <= zMax; ++z) {
        centers.push_back(cv::Vec3f(0.0f, 0.0f, static_cast<float>(z)));
    }
    return centers;
}

std::vector<cv::Vec3d> arcPoints(double z, double radius, double radiusPerTurn,
                                 double thetaBegin, double thetaEnd)
{
    std::vector<cv::Vec3d> points;
    const int count = static_cast<int>(std::floor((thetaEnd - thetaBegin) / kStep)) + 1;
    points.reserve(static_cast<std::size_t>(std::max(count, 0)));
    for (int i = 0; i < count; ++i) {
        const double theta = thetaBegin + static_cast<double>(i) * kStep;
        const double r = radius + radiusPerTurn * theta / kTwoPi;
        points.push_back(cv::Vec3d(r * std::cos(theta), r * std::sin(theta), z));
    }
    return points;
}

std::vector<cv::Vec3d> verticalPoints(double theta, double radius, double zBegin,
                                      double zEnd, double zStep)
{
    std::vector<cv::Vec3d> points;
    const int count = static_cast<int>(std::floor((zEnd - zBegin) / zStep)) + 1;
    points.reserve(static_cast<std::size_t>(std::max(count, 0)));
    for (int i = 0; i < count; ++i) {
        const double z = zBegin + static_cast<double>(i) * zStep;
        points.push_back(cv::Vec3d(radius * std::cos(theta), radius * std::sin(theta), z));
    }
    return points;
}

InputFiber makeFiber(uint64_t id, const QString& label, char hvTag,
                     std::vector<cv::Vec3d> linePoints,
                     const std::vector<int>& controlIndices)
{
    InputFiber fiber;
    fiber.id = id;
    fiber.fileName = label.toStdString() + ".json";
    fiber.label = label;
    fiber.hvTag = hvTag;
    fiber.linePoints = std::move(linePoints);
    for (int index : controlIndices) {
        // Loud in every build type: a bad fixture index must abort the test,
        // not read past the vector in release.
        if (index < 0 ||
            static_cast<std::size_t>(index) >= fiber.linePoints.size()) {
            throw std::out_of_range(
                "makeFiber: control index " + std::to_string(index) +
                " out of range for " + std::to_string(fiber.linePoints.size()) +
                " line points");
        }
        fiber.controlPoints.push_back(fiber.linePoints[static_cast<std::size_t>(index)]);
    }
    if (fiber.controlPoints.size() > 1) {
        fiber.tracedSegments.assign(fiber.controlPoints.size() - 1, true);
    }
    return fiber;
}

void addLink(InputFiber& a, int controlA, InputFiber& b, int controlB)
{
    a.links.push_back({controlA, b.id, controlB});
    b.links.push_back({controlB, a.id, controlA});
}

double angleOf(const cv::Vec3d& point)
{
    return std::atan2(point[1], point[0]);
}

// One H fiber winding around the scroll with a V fiber linked at every
// requested crossing (same-winding contacts: the V is drawn through the H
// fiber's own point).
std::vector<InputFiber> makeWeave(uint64_t firstId, const QString& prefix, double z,
                                  double radius, double radiusPerTurn,
                                  double thetaBegin, double thetaEnd,
                                  const std::vector<int>& controlIndices)
{
    std::vector<InputFiber> fibers;
    std::vector<cv::Vec3d> line = arcPoints(z, radius, radiusPerTurn, thetaBegin, thetaEnd);
    fibers.push_back(makeFiber(firstId, prefix + QStringLiteral("h-1"), 'H', line,
                               controlIndices));
    for (std::size_t i = 0; i < controlIndices.size(); ++i) {
        const cv::Vec3d crossing = fibers.front().controlPoints[i];
        std::vector<cv::Vec3d> verticalLine =
            verticalPoints(angleOf(crossing), std::hypot(crossing[0], crossing[1]),
                           z - 400.0, z + 400.0, 4.0);
        const int last = static_cast<int>(verticalLine.size()) - 1;
        InputFiber vertical = makeFiber(firstId + 1 + i,
                                        prefix + QStringLiteral("v-%1").arg(i + 1), 'V',
                                        std::move(verticalLine), {0, last / 2, last});
        addLink(fibers.front(), static_cast<int>(i), vertical, 1);
        fibers.push_back(std::move(vertical));
    }
    return fibers;
}

// Two weaves (an H with linked Vs each) for the cache tests: multiple
// networks, multiple pairs, deterministic.
std::vector<InputFiber> cacheFixture()
{
    std::vector<InputFiber> fibers =
        makeWeave(100, QStringLiteral("a-"), 30000.0, 4000.0, 300.0,
                  0.0, 1.5 * kTwoPi, {200, 900, 1600});
    std::vector<InputFiber> small =
        makeWeave(200, QStringLiteral("b-"), 30000.0, 1500.0, 100.0,
                  0.0, 1.2 * kTwoPi, {150, 1100});
    fibers.insert(fibers.end(), small.begin(), small.end());
    return fibers;
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

std::vector<InputFiber> mirrored(std::vector<InputFiber> fibers)
{
    for (InputFiber& fiber : fibers) {
        for (cv::Vec3d& point : fiber.linePoints) {
            point[1] = -point[1];
        }
        for (cv::Vec3d& point : fiber.controlPoints) {
            point[1] = -point[1];
        }
    }
    return fibers;
}

// The dented sheet of the solver's traversal-group tests, in volume space:
// an H fiber at height z along radius R + b(u+2)^2 and angle
// theta0 + eps(u^3 - 3u), u in [-3, 3] (angle forward, back, forward), and a
// V fiber on the ray theta0 at a fixed radius. With the V fiber a thickness
// inside the dent's outer limb it is on the next sheet inward: the three
// crossings read inside, inside, outside, and only their count says so.
constexpr double kHairpinR = 20000.0;
constexpr double kHairpinB = 100.0;
constexpr double kHairpinEps = 0.03;
constexpr double kHairpinTheta0 = 0.3 * kTwoPi;
constexpr int kHairpinOuterIndex = 473;  // u = sqrt3 at 0.01 steps from -3

std::vector<InputFiber> hairpinPair(bool linked)
{
    std::vector<cv::Vec3d> arc;
    for (int i = 0; i <= 600; ++i) {
        const double u = -3.0 + 0.01 * i;
        const double r = kHairpinR + kHairpinB * (u + 2.0) * (u + 2.0) - 100.0;
        const double theta = kHairpinTheta0 + kHairpinEps * (u * u * u - 3.0 * u);
        arc.push_back(cv::Vec3d(r * std::cos(theta), r * std::sin(theta), 30000.0));
    }
    const double outerLimb = kHairpinR + kHairpinB * (1.7320508 + 2.0) * (1.7320508 + 2.0);
    std::vector<InputFiber> fibers;
    fibers.push_back(makeFiber(700, QStringLiteral("f-h"), 'H', arc,
                               {0, kHairpinOuterIndex, 600}));
    fibers.push_back(makeFiber(701, QStringLiteral("f-v"), 'V',
                               verticalPoints(kHairpinTheta0, outerLimb - 300.0,
                                              29000.0, 31000.0, 25.0),
                               {0, 40, 80}));
    if (linked) {
        addLink(fibers[0], 1, fibers[1], 1);
    }
    return fibers;
}

// A kollesis seam in volume space: the inner sheet's H fiber ends just past
// the seam angle with its last control point tagged, the outer sheet's H
// fiber starts just before it with its first control point tagged, one step
// further in, and the outer sheet's V fiber runs at the seam between the two
// - in front of the inner H fiber by 90 vx, so that crossing reads Outside.
// The H fibers overrun the V fiber a little, as annotated ends do. `sameSide` puts
// the second H fiber's tagged end on the same side of the V fiber (a
// negative control); `linkMask` selects which of the two links exist (bit 0
// inner, bit 1 outer); `tagMask` which ends are tagged; `shortControl` ends
// the inner H fiber's controls three line points before its line does and
// lifts the line beyond them above the V fiber, so the tagged control is not
// the trace's end and the trace's end sits at a height the V never reaches;
// `linkAtCrossing` gives each H fiber a control at the crossing and links
// there instead of at the tagged end, as links are drawn.
constexpr double kSeamAngle = 0.4 * kTwoPi;
constexpr double kSeamOverrun = 0.05;
constexpr double kSeamInnerRadius = 4000.0;
constexpr double kSeamOuterRadius = kSeamInnerRadius - 150.0;
constexpr double kSeamVRadius = kSeamOuterRadius + 60.0;

// `extraInner`: 0 none; 1 an untagged inner H fiber (803) alongside the tagged
// one, ending just past the V and linked to the tagged inner H fiber at a
// control at the same angle (same winding, no tag); 2 the same but unlinked;
// 3 linked and running a full turn on past the V, so its only end within a
// turn of the crossing is its start, on the inner sheet's body side.
std::vector<InputFiber> kollesisSeam(bool sameSide, int linkMask, int tagMask,
                                     bool shortControl = false, bool linkAtCrossing = false,
                                     int extraInner = 0)
{
    std::vector<InputFiber> fibers;
    std::vector<cv::Vec3d> inner =
        arcPoints(30000.0, kSeamInnerRadius, 0.0, kSeamAngle - 0.6, kSeamAngle + kSeamOverrun);
    const int innerLast = static_cast<int>(inner.size()) - 1 - (shortControl ? 3 : 0);
    for (std::size_t i = static_cast<std::size_t>(innerLast) + 1; i < inner.size(); ++i) {
        inner[i][2] += 1000.0;
    }
    // The inner H fiber's seam-end control, and the control its link sits on.
    const int innerSeam = 2;
    const int innerCrossing = static_cast<int>(std::lround(0.6 / kStep));
    const int innerLinked = linkAtCrossing ? 1 : innerSeam;
    fibers.push_back(makeFiber(800, QStringLiteral("k-inner"), 'H', std::move(inner),
                               {0, linkAtCrossing ? innerCrossing : innerLast / 2, innerLast}));
    std::vector<cv::Vec3d> outer = sameSide
        ? arcPoints(30000.0, kSeamOuterRadius, 0.0, kSeamAngle - 0.6, kSeamAngle + kSeamOverrun)
        : arcPoints(30000.0, kSeamOuterRadius, 0.0, kSeamAngle - kSeamOverrun, kSeamAngle + 0.6);
    const int outerLast = static_cast<int>(outer.size()) - 1;
    const int outerCrossing = static_cast<int>(
        std::lround((sameSide ? 0.6 : kSeamOverrun) / kStep));
    const int outerSeam = sameSide ? 2 : 0;
    const int outerLinked = linkAtCrossing ? 1 : outerSeam;
    fibers.push_back(makeFiber(801, QStringLiteral("k-outer"), 'H', std::move(outer),
                               {0, linkAtCrossing ? outerCrossing : outerLast / 2, outerLast}));
    fibers.push_back(makeFiber(802, QStringLiteral("k-v"), 'V',
                               verticalPoints(kSeamAngle, kSeamVRadius, 29600.0, 30400.0, 4.0),
                               {0, 100, 200}));
    fibers[0].kollesisTerminations.assign(3, false);
    fibers[1].kollesisTerminations.assign(3, false);
    if (tagMask & 1) {
        fibers[0].kollesisTerminations[static_cast<std::size_t>(innerSeam)] = true;
    }
    if (tagMask & 2) {
        fibers[1].kollesisTerminations[static_cast<std::size_t>(outerSeam)] = true;
    }
    if (linkMask & 1) {
        addLink(fibers[0], innerLinked, fibers[2], 1);
    }
    if (linkMask & 2) {
        addLink(fibers[1], outerLinked, fibers[2], 1);
    }
    if (extraInner != 0) {
        const double end = extraInner == 3 ? kSeamAngle + 0.2 + kTwoPi : kSeamAngle + kSeamOverrun;
        std::vector<cv::Vec3d> extra =
            arcPoints(30000.0, kSeamInnerRadius, 0.0, kSeamAngle - 0.6, end);
        const int extraLast = static_cast<int>(extra.size()) - 1;
        // Its middle control at the tagged inner H fiber's middle control's
        // angle (same start, same step), so the link joins equal angles.
        fibers.push_back(makeFiber(803, QStringLiteral("k-inner2"), 'H', std::move(extra),
                                   {0, linkAtCrossing ? innerCrossing : innerLast / 2, extraLast}));
        fibers.back().kollesisTerminations.assign(3, false);
        if (extraInner != 2) {
            addLink(fibers.back(), 1, fibers[0], 1);
        }
    }
    return fibers;
}

const GlobalPlacedFiber* findFiber(const GlobalResult& result, uint64_t id)
{
    for (const GlobalPlacedFiber& fiber : result.fibers) {
        if (fiber.fiber.id == id) {
            return &fiber;
        }
    }
    return nullptr;
}

} // namespace

// The positive kollesis seam check, for both control placements.
void checkKollesisSeam(bool shortControl, bool linkAtCrossing)
{
    const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
    const GlobalResult result = vc3d::fiber_map::buildGlobalLayout(
        kollesisSeam(false, 3, 3, shortControl, linkAtCrossing), umbilicus, defaultParams());
    const GlobalPlacedFiber* v = findFiber(result, 802);
    QVERIFY(v != nullptr);
    QVERIFY(v->meta.onKollesis);
    // Both tagged ends meet the V fiber: the inner H fiber's encounter is
    // the false Outside reading, the outer H fiber's already reads Inside;
    // both are seam encounters.
    int innerSeam = 0;
    int outerSeam = 0;
    for (const auto& event : result.crossingEvents) {
        if (event.vFiberId != 802) {
            continue;
        }
        QVERIFY(event.kollesis);
        QCOMPARE(event.kind, vc3d::fiber_map::winding::CrossingKind::Inside);
        if (event.hFiberId == 800) {
            QVERIFY(event.deltaR > 0.0);
            ++innerSeam;
        } else {
            QCOMPARE(event.hFiberId, uint64_t{801});
            QVERIFY(event.deltaR < 0.0);
            ++outerSeam;
        }
    }
    QVERIFY(innerSeam >= 1);
    QVERIFY(outerSeam >= 1);
    QCOMPARE(result.kollesisCrossingCount, innerSeam + outerSeam);
    QCOMPARE(result.droppedCrossingCount, 0);
    QCOMPARE(result.declaredGroupCount, 0);
    QCOMPARE(result.suspectLinkCount, 0);
    QVERIFY(result.suspectCrossings.empty());
    const GlobalPlacedFiber* inner = findFiber(result, 800);
    const GlobalPlacedFiber* outer = findFiber(result, 801);
    QVERIFY(inner != nullptr && outer != nullptr);
    QVERIFY(std::abs(inner->meta.windingHi - v->meta.windingLo) < 0.6);
    QVERIFY(std::abs(outer->meta.windingLo - v->meta.windingLo) < 0.6);
}

class TestFiberGlobalLayout : public QObject
{
    Q_OBJECT

private slots:
    // Every input fiber is either placed or reported unplaceable; no gate on
    // network size, no top-N cut.
    void everyFiberIsAccountedFor()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers =
            makeWeave(100, QStringLiteral("a-"), 30000.0, 4000.0, 300.0,
                      0.0, 1.5 * kTwoPi, {200, 900, 1600});
        // An unlinked fiber on the weave's own sheet, a little above it: no
        // crossings (the V fibers stop below its z), so it is an island whose
        // local radial ordering ties it back to the sheet.
        fibers.push_back(makeFiber(900, QStringLiteral("c-h-9"), 'H',
                                   arcPoints(30500.0, 4000.0, 300.0, 0.0, 0.8 * kTwoPi),
                                   {100, 800}));
        // And one with no geometry at all.
        InputFiber empty;
        empty.id = 901;
        empty.fileName = "broken.json";
        empty.label = QStringLiteral("broken");
        empty.hvTag = 'V';
        fibers.push_back(empty);

        const GlobalResult result =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QCOMPARE(result.fibers.size(), std::size_t{5});
        QCOMPARE(result.unplaced.size(), std::size_t{1});
        QCOMPARE(result.unplaced.front().id, uint64_t{901});
        QCOMPARE(QString::fromStdString(result.unplaced.front().fileName),
                 QStringLiteral("broken.json"));

        const GlobalPlacedFiber* island = findFiber(result, 900);
        QVERIFY(island != nullptr);
        QCOMPARE(island->meta.anchor, GlobalAnchor::Radius);
        QVERIFY(!island->meta.linked);
        QCOMPARE(result.islandCount, 1);
        // Ties to the weave H fiber's own sheet: the island's winding range
        // starts 100 angular samples before the weave H fiber's domain.
        const GlobalPlacedFiber* weaveH = findFiber(result, 100);
        QVERIFY(weaveH != nullptr);
        QVERIFY2(std::abs(island->meta.windingLo -
                          (weaveH->meta.windingLo - 100.0 * kStep / kTwoPi)) < 0.01,
                 qPrintable(QStringLiteral("island %1 weave %2")
                                .arg(island->meta.windingLo)
                                .arg(weaveH->meta.windingLo)));

        for (const GlobalPlacedFiber& fiber : result.fibers) {
            QVERIFY(fiber.meta.windingHi >= fiber.meta.windingLo);
            QVERIFY(!fiber.fiber.runs.empty());
        }
    }

    // Linked crossings coincide on the global map exactly as they do on the
    // per-network panels, and the winding gridlines are numbered by the
    // winding coordinate with the innermost anchored winding at zero.
    void linksCoincideAndWindingsAreNumbered()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const std::vector<InputFiber> fibers =
            makeWeave(100, QStringLiteral("a-"), 30000.0, 4000.0, 300.0,
                      -0.4, kTwoPi + 0.4, {100, 100 + kStepsPerTurn});
        const GlobalResult result =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QCOMPARE(result.chirality, 1);
        QCOMPARE(result.fibers.size(), std::size_t{3});
        QCOMPARE(result.links.size(), std::size_t{2});
        QCOMPARE(result.suspectLinkCount, 0);
        for (const PlacedLink& link : result.links) {
            QVERIFY2(link.turnErr < 1e-9, qPrintable(QString::number(link.turnErr)));
            QVERIFY(std::abs(link.a.x() - link.b.x()) < vx(0.01));
            QVERIFY(std::abs(link.a.y() - link.b.y()) < vx(0.01));
        }
        // The two crossings sit one winding apart.
        QVERIFY(std::abs(std::abs(result.links[1].a.x() - result.links[0].a.x()) -
                         kTwoPi * result.rRefVx) < vx(0.05));

        QVERIFY(!result.windings.empty());
        for (std::size_t i = 0; i < result.windings.size(); ++i) {
            QVERIFY(result.windings[i].xVx >= result.x0Vx);
            QVERIFY(result.windings[i].xVx <= result.x1Vx);
            QVERIFY(std::abs(result.windings[i].xVx -
                             static_cast<double>(result.windings[i].number) *
                                 kTwoPi * result.rRefVx) < 1e-6);
            if (i > 0) {
                QCOMPARE(result.windings[i].number, result.windings[i - 1].number + 1);
            }
        }

        double minW = std::numeric_limits<double>::infinity();
        for (const GlobalPlacedFiber& fiber : result.fibers) {
            QCOMPARE(fiber.meta.anchor, GlobalAnchor::Primary);
            QVERIFY(fiber.meta.linked);
            minW = std::min(minW, fiber.meta.windingLo);
        }
        QVERIFY(minW >= 0.0);
        QVERIFY(minW < 1.0);

        // Determinism: same input, identical map.
        const GlobalResult repeat =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QVERIFY(repeat.x0Vx == result.x0Vx);
        QVERIFY(repeat.x1Vx == result.x1Vx);
        QVERIFY(repeat.rRefVx == result.rRefVx);
        for (std::size_t f = 0; f < result.fibers.size(); ++f) {
            QCOMPARE(repeat.fibers[f].fiber.label, result.fibers[f].fiber.label);
            QCOMPARE(repeat.fibers[f].meta.windingLo, result.fibers[f].meta.windingLo);
        }
    }

    // A mirrored scroll fits the same sheet model: the winding coordinate
    // still grows outward, so the pitch keeps its sign and size.
    void sheetModelSurvivesMirroring()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers =
            makeWeave(100, QStringLiteral("a-"), 30000.0, 4000.0, 300.0,
                      -0.4, kTwoPi + 0.4, {100, 100 + kStepsPerTurn});
        const GlobalResult forward =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        for (InputFiber& fiber : fibers) {
            for (cv::Vec3d& point : fiber.linePoints) {
                point[1] = -point[1];
            }
            for (cv::Vec3d& point : fiber.controlPoints) {
                point[1] = -point[1];
            }
        }
        const GlobalResult mirrored =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QCOMPARE(mirrored.chirality, -forward.chirality);
        QVERIFY(forward.sheetPitchVx > 0.0);
        QVERIFY2(std::abs(mirrored.sheetPitchVx - forward.sheetPitchVx) < 1e-6 * forward.sheetPitchVx,
                 qPrintable(QString::number(mirrored.sheetPitchVx)));
        QVERIFY(std::abs(mirrored.sheetRadius0Vx - forward.sheetRadius0Vx) < 1e-6 * forward.sheetRadius0Vx);
    }

    // A mirrored scroll (opposite chirality) produces the same map: the
    // winding coordinate still grows outward and crossings still coincide.
    void mirroredChiralityLaysOutTheSameMap()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const std::vector<InputFiber> fibers = mirrored(
            makeWeave(100, QStringLiteral("a-"), 30000.0, 4000.0, 300.0,
                      -0.4, kTwoPi + 0.4, {100, 100 + kStepsPerTurn}));
        const GlobalResult result =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QCOMPARE(result.chirality, -1);
        QCOMPARE(result.fibers.size(), std::size_t{3});
        QCOMPARE(result.suspectLinkCount, 0);
        for (const PlacedLink& link : result.links) {
            QVERIFY(link.turnErr < 1e-9);
            QVERIFY(std::abs(link.a.x() - link.b.x()) < vx(0.01));
        }
        double minW = std::numeric_limits<double>::infinity();
        for (const GlobalPlacedFiber& fiber : result.fibers) {
            minW = std::min(minW, fiber.meta.windingLo);
        }
        QVERIFY(minW >= 0.0);
        QVERIFY(minW < 1.0);
    }

    // A fiber with no model-traced span never declares winding errors: the
    // same wrong-winding link that is suspect between two traced fibers is
    // silent when one end is pure control-point interpolation, and any
    // dropped crossings it causes draw no red rings.
    // Declarations are not gated on trust: an interpolated fiber's wrong
    // link and the crossings it contradicts are reported exactly as a traced
    // fiber's would be. Its evidence is attenuated uniformly, so the same
    // constraints fall.
    void interpolatedFibersDeclareLikeAnyOther()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers =
            makeWeave(100, QStringLiteral("a-"), 30000.0, 4000.0, 300.0,
                      -0.4, kTwoPi + 0.4, {100, 100 + kStepsPerTurn});
        // The deliberately wrong link: the H fiber's second crossing also
        // claims the first V fiber, one whole turn away.
        addLink(fibers[0], 1, fibers[1], 1);

        const GlobalResult trusted =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QVERIFY(trusted.suspectLinkCount > 0);

        // Same geometry, same wrong link, the H fiber pure interpolation.
        std::vector<InputFiber> untrusted = fibers;
        untrusted[0].tracedSegments.assign(
            untrusted[0].controlPoints.size() - 1, false);
        const GlobalResult declared = vc3d::fiber_map::buildGlobalLayout(
            untrusted, umbilicus, defaultParams());
        QCOMPARE(declared.suspectLinkCount, trusted.suspectLinkCount);
        QCOMPARE(declared.droppedCrossingCount, trusted.droppedCrossingCount);
        QCOMPARE(declared.suspectCrossings.size(), trusted.suspectCrossings.size());
        QCOMPARE(declared.fibers.size(), trusted.fibers.size());
    }

    // Linked-network ids drive the dock grouping and the selection's network
    // co-highlight: components of the manual link graph, numbered by size
    // descending, -1 for unlinked fibers.
    void networkIdsNumberBySizeLargestFirst()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        // Two networks: 4 fibers (1 H + 3 V) and 3 fibers (1 H + 2 V), plus
        // one unlinked fiber.
        std::vector<InputFiber> fibers =
            makeWeave(100, QStringLiteral("a-"), 30000.0, 4000.0, 300.0,
                      0.0, 1.5 * kTwoPi, {200, 900, 1600});
        std::vector<InputFiber> small =
            makeWeave(200, QStringLiteral("b-"), 30000.0, 1500.0, 100.0,
                      0.0, 1.2 * kTwoPi, {150, 1100});
        fibers.insert(fibers.end(), small.begin(), small.end());
        fibers.push_back(makeFiber(900, QStringLiteral("c-h-9"), 'H',
                                   arcPoints(30500.0, 5000.0, 100.0, 0.0, 0.8 * kTwoPi),
                                   {100, 800}));
        const GlobalResult result =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QCOMPARE(result.fibers.size(), std::size_t{8});
        for (const GlobalPlacedFiber& fiber : result.fibers) {
            const uint64_t id = fiber.fiber.id;
            if (id == 900) {
                QCOMPARE(fiber.meta.networkId, -1);
                QCOMPARE(fiber.meta.networkSize, 1);
            } else if (id >= 200) {
                QCOMPARE(fiber.meta.networkId, 1);
                QCOMPARE(fiber.meta.networkSize, 3);
            } else {
                QCOMPARE(fiber.meta.networkId, 0);
                QCOMPARE(fiber.meta.networkSize, 4);
            }
        }
    }

    // --- Memoization: a cached build is bit-identical to an uncached one,
    // only changed slots recompute, and every declared invalidation trigger
    // fires. The deep comparison is the exactness contract itself.
    void cacheMatchesUncachedBitIdentically()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers = cacheFixture();
        const GlobalLayoutParams params = defaultParams();

        const GlobalResult fresh =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params);
        vc3d::fiber_map::GlobalLayoutCache cache;
        const GlobalResult cold =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        const GlobalResult warm =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(cold) ==
                 vc3d::fiber_map::digestGlobalResult(fresh));
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warm) ==
                 vc3d::fiber_map::digestGlobalResult(fresh));
        QCOMPARE(cache.lastStats().fibersRecomputed, 0);
        QCOMPARE(cache.lastStats().pairsRecomputed, 0);
        QVERIFY(cache.lastStats().pairsReused > 0);

        // Mutate one V fiber: only its slots recompute, and the result still
        // equals a from-scratch build of the mutated input.
        for (cv::Vec3d& point : fibers[2].linePoints) {
            point[2] += 40.0;
        }
        fibers[2].controlPoints[1] = fibers[2].linePoints[100];
        const GlobalResult freshMutated =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params);
        const GlobalResult warmMutated =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warmMutated) ==
                 vc3d::fiber_map::digestGlobalResult(freshMutated));
        QCOMPARE(cache.lastStats().fibersRecomputed, 1);
        // The mutated fiber is a V: exactly its pairs (one per H fiber)
        // recompute.
        QCOMPARE(cache.lastStats().pairsRecomputed, 2);
        QVERIFY(cache.lastStats().pairsReused > 0);
    }

    void cacheInvalidationTriggers()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const std::vector<InputFiber> fibers = cacheFixture();
        const GlobalLayoutParams params = defaultParams();
        vc3d::fiber_map::GlobalLayoutCache cache;
        (void)vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);

        // Umbilicus change invalidates prep and, through the chained keys,
        // every pair.
        std::vector<cv::Vec3f> movedUmbilicus = umbilicus;
        movedUmbilicus[20000][0] += 50.0f;
        const GlobalResult freshMoved =
            vc3d::fiber_map::buildGlobalLayout(fibers, movedUmbilicus, params);
        const GlobalResult warmMoved = vc3d::fiber_map::buildGlobalLayout(
            fibers, movedUmbilicus, params, &cache);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warmMoved) ==
                 vc3d::fiber_map::digestGlobalResult(freshMoved));
        QCOMPARE(cache.lastStats().fibersReused, 0);
        QCOMPARE(cache.lastStats().pairsReused, 0);

        // A detection parameter invalidates pairs but not prep.
        (void)vc3d::fiber_map::buildGlobalLayout(fibers, movedUmbilicus, params,
                                                 &cache);
        GlobalLayoutParams tightened = params;
        tightened.solver.zMergeVx *= 0.5;
        const GlobalResult freshTight = vc3d::fiber_map::buildGlobalLayout(
            fibers, movedUmbilicus, tightened);
        const GlobalResult warmTight = vc3d::fiber_map::buildGlobalLayout(
            fibers, movedUmbilicus, tightened, &cache);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warmTight) ==
                 vc3d::fiber_map::digestGlobalResult(freshTight));
        QCOMPARE(cache.lastStats().fibersRecomputed, 0);
        QCOMPARE(cache.lastStats().pairsReused, 0);

        // A geometry-only parameter touches no cached layer.
        GlobalLayoutParams smoother = tightened;
        smoother.smoothVx = vx(0.05);
        const GlobalResult freshSmooth = vc3d::fiber_map::buildGlobalLayout(
            fibers, movedUmbilicus, smoother);
        const GlobalResult warmSmooth = vc3d::fiber_map::buildGlobalLayout(
            fibers, movedUmbilicus, smoother, &cache);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warmSmooth) ==
                 vc3d::fiber_map::digestGlobalResult(freshSmooth));
        QCOMPARE(cache.lastStats().fibersRecomputed, 0);
        QCOMPARE(cache.lastStats().pairsRecomputed, 0);
    }

    void cacheHandlesAddRemoveRenameAndDuplicates()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers = cacheFixture();
        const GlobalLayoutParams params = defaultParams();
        vc3d::fiber_map::GlobalLayoutCache cache;
        (void)vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);

        // Rename: content unchanged, but the name is part of the identity, so
        // its slots recompute and the old ones are swept - and the result
        // still equals a fresh build.
        std::vector<InputFiber> renamed = fibers;
        renamed[1].fileName = "renamed.json";
        const GlobalResult freshRenamed =
            vc3d::fiber_map::buildGlobalLayout(renamed, umbilicus, params);
        const GlobalResult warmRenamed = vc3d::fiber_map::buildGlobalLayout(
            renamed, umbilicus, params, &cache);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warmRenamed) ==
                 vc3d::fiber_map::digestGlobalResult(freshRenamed));
        QCOMPARE(cache.lastStats().fibersRecomputed, 1);

        // Remove a fiber; then a build with the original set again must
        // recompute the removed fiber's slots (they were swept).
        std::vector<InputFiber> reduced = renamed;
        reduced.erase(reduced.begin());
        const GlobalResult freshReduced =
            vc3d::fiber_map::buildGlobalLayout(reduced, umbilicus, params);
        const GlobalResult warmReduced = vc3d::fiber_map::buildGlobalLayout(
            reduced, umbilicus, params, &cache);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warmReduced) ==
                 vc3d::fiber_map::digestGlobalResult(freshReduced));
        const GlobalResult warmRestored = vc3d::fiber_map::buildGlobalLayout(
            renamed, umbilicus, params, &cache);
        QCOMPARE(cache.lastStats().fibersRecomputed, 1);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warmRestored) ==
                 vc3d::fiber_map::digestGlobalResult(freshRenamed));

        // Duplicate fileNames disable the cache but not the build.
        std::vector<InputFiber> duplicated = fibers;
        duplicated[1].fileName = duplicated[0].fileName;
        const GlobalResult freshDup =
            vc3d::fiber_map::buildGlobalLayout(duplicated, umbilicus, params);
        const GlobalResult warmDup = vc3d::fiber_map::buildGlobalLayout(
            duplicated, umbilicus, params, &cache);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warmDup) ==
                 vc3d::fiber_map::digestGlobalResult(freshDup));
        QVERIFY(!cache.lastStats().used);
    }

    // A genuine multi-pair conflict, where WHICH edge each detected cycle
    // sacrifices depends on constraint order: the H fiber passes both V
    // fibers outside on its first turn, then regresses inward and passes
    // them inside on its second - an inward regression per pair, cycles
    // spanning both pairs. The cached replay must reproduce the same drops.
    void cacheReplayPreservesRepairTieBreaks()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers;
        std::vector<cv::Vec3d> regress;
        const double z = 30000.0;
        for (double theta = 0.05 * kTwoPi; theta <= 2.3 * kTwoPi; theta += kStep) {
            const double r = theta < 1.15 * kTwoPi ? 3400.0 : 2400.0;
            regress.push_back(cv::Vec3d(r * std::cos(theta), r * std::sin(theta), z));
        }
        const int lastIndex = static_cast<int>(regress.size()) - 1;
        fibers.push_back(makeFiber(1, QStringLiteral("h-regress"), 'H',
                                   std::move(regress), {10, lastIndex - 10}));
        // Two growing spirals pin the inferred chirality at +1: the
        // regressing fiber's radius drop otherwise wins the turn-lag vote
        // and mirrors the map, absorbing the very conflict this fixture
        // exists to create.
        fibers.push_back(makeFiber(4, QStringLiteral("a-anchor"), 'H',
                                   arcPoints(z + 300.0, 5000.0, 400.0, 0.0,
                                             3.0 * kTwoPi),
                                   {100, 3000}));
        fibers.push_back(makeFiber(5, QStringLiteral("b-anchor"), 'H',
                                   arcPoints(z - 300.0, 5200.0, 400.0, 0.0,
                                             3.0 * kTwoPi),
                                   {100, 3000}));
        fibers.push_back(makeFiber(
            2, QStringLiteral("v-a"), 'V',
            verticalPoints(0.3 * kTwoPi, 3000.0, z - 500.0, z + 500.0, 4.0),
            {0, 125, 250}));
        fibers.push_back(makeFiber(
            3, QStringLiteral("v-b"), 'V',
            verticalPoints(0.4 * kTwoPi, 3000.0, z - 500.0, z + 500.0, 4.0),
            {0, 125, 250}));
        const GlobalLayoutParams params = defaultParams();
        const GlobalResult fresh =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params);
        QCOMPARE(fresh.chirality, 1);
        // The fixture must actually conflict: the two inward-regression
        // drops are declared on the map.
        QCOMPARE(fresh.droppedCrossingCount, 2);
        vc3d::fiber_map::GlobalLayoutCache cache;
        (void)vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        const GlobalResult warm =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        QCOMPARE(cache.lastStats().pairsReused, 6);
        QCOMPARE(cache.lastStats().pairsRecomputed, 0);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warm) ==
                vc3d::fiber_map::digestGlobalResult(fresh));
    }

    // A chirality flip (every fiber mirrored) invalidates every pair shard,
    // and the cached result still equals a fresh build of the mirrored input.
    void cacheInvalidatesOnChiralityFlip()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers = cacheFixture();
        const GlobalLayoutParams params = defaultParams();
        vc3d::fiber_map::GlobalLayoutCache cache;
        (void)vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        for (InputFiber& fiber : fibers) {
            for (cv::Vec3d& point : fiber.linePoints) {
                point[1] = -point[1];
            }
            for (cv::Vec3d& point : fiber.controlPoints) {
                point[1] = -point[1];
            }
        }
        const GlobalResult fresh =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params);
        QCOMPARE(fresh.chirality, -1);
        const GlobalResult warm =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        QCOMPARE(cache.lastStats().pairsReused, 0);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warm) ==
                vc3d::fiber_map::digestGlobalResult(fresh));
    }

    // Pairs with no crossings at all (disjoint z spans) cache and replay as
    // empty shards.
    void cacheReplaysEmptyShards()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers = cacheFixture();
        fibers.push_back(makeFiber(900, QStringLiteral("z-far"), 'H',
                                   arcPoints(38000.0, 4000.0, 300.0, 0.0, 0.8 * kTwoPi),
                                   {100, 800}));
        const GlobalLayoutParams params = defaultParams();
        const GlobalResult fresh =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params);
        vc3d::fiber_map::GlobalLayoutCache cache;
        (void)vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        const GlobalResult warm =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        // The far fiber's pairs are all empty shards - reused like any other.
        QCOMPARE(cache.lastStats().pairsRecomputed, 0);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warm) ==
                vc3d::fiber_map::digestGlobalResult(fresh));
    }

    // The verification digest is sensitive to every semantic field class it
    // exists to guard - a mutation that dodges it would let a cache bug hide.
    void resultDigestIsSensitive()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const std::vector<InputFiber> fibers = cacheFixture();
        const GlobalResult base =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        const ContentDigest baseline = vc3d::fiber_map::digestGlobalResult(base);
        {
            GlobalResult tweaked = base;
            tweaked.droppedCrossingCount += 1;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = base;
            tweaked.gatedSegmentCount += 1;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = base;
            QVERIFY(!tweaked.links.empty());
            tweaked.links[0].pending = !tweaked.links[0].pending;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = base;
            tweaked.fibers[0].meta.networkSize += 1;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = base;
            tweaked.fibers[0].fiber.label += QStringLiteral("x");
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = base;
            tweaked.fibers[0].fiber.id += 1;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        // Timings are the one deliberate exclusion: telemetry, not semantics.
        {
            GlobalResult tweaked = base;
            tweaked.solveMs += 100.0;
            QVERIFY(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline);
        }
    }

    // The sheet model recovers an Archimedean weave's pitch and inner radius,
    // and the distance functions invert each other and integrate the modelled
    // radius over the angle.
    void sheetModelRecoversArchimedeanPitch()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        constexpr double kRadius = 4000.0;
        constexpr double kPitch = 300.0;
        const std::vector<InputFiber> fibers =
            makeWeave(100, QStringLiteral("a-"), 30000.0, kRadius, kPitch,
                      -0.4, kTwoPi + 0.4, {100, 100 + kStepsPerTurn});
        const GlobalResult result =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QVERIFY(result.sheetPitchVx > 0.0);
        QVERIFY2(std::abs(result.sheetPitchVx - kPitch) < 0.02 * kPitch,
                 qPrintable(QString::number(result.sheetPitchVx)));
        // The fixture's radius at theta = 0 is kRadius; winding 0 sits at the
        // innermost anchored winding, within a turn of that angle, so the
        // fitted radius at winding 0 lands within one pitch of it.
        QVERIFY2(std::abs(result.sheetRadius0Vx - kRadius) < 1.5 * kPitch,
                 qPrintable(QString::number(result.sheetRadius0Vx)));

        const vc3d::fiber_map::SheetModel model = vc3d::fiber_map::sheetModelOf(result);
        QCOMPARE(model.rRefVx, result.rRefVx);
        QCOMPARE(vc3d::fiber_map::sheetDistanceVx(model, 0.0), 0.0);
        // One winding out along the map is the integral of r over one turn.
        const double circumference = kTwoPi * result.rRefVx;
        const double oneTurn = vc3d::fiber_map::sheetDistanceVx(model, circumference);
        QVERIFY(std::abs(oneTurn - kTwoPi * (result.sheetRadius0Vx +
                                            0.5 * result.sheetPitchVx)) < 1e-6);
        // Outer windings are longer than inner ones, and the map's own
        // arclength lies between them.
        const double secondTurn =
            vc3d::fiber_map::sheetDistanceVx(model, 2.0 * circumference) - oneTurn;
        QVERIFY(secondTurn > oneTurn);
        // Round trip through the inverse, on both sides of winding 0.
        for (double x : {-0.5 * circumference, 0.0, 0.3 * circumference,
                         2.7 * circumference}) {
            const double distance = vc3d::fiber_map::sheetDistanceVx(model, x);
            const double back = vc3d::fiber_map::sheetXForDistanceVx(model, distance);
            QVERIFY2(std::abs(back - x) < 1e-6,
                     qPrintable(QStringLiteral("%1 -> %2 -> %3").arg(x).arg(distance).arg(back)));
        }
        // A distance no positive radius can reach has no position.
        QVERIFY(std::isnan(vc3d::fiber_map::sheetXForDistanceVx(model, -1e12)));

        // A vanishingly small positive pitch must not lose the answer to
        // cancellation: the inverse tends smoothly to the linear case.
        {
            const vc3d::fiber_map::SheetModel tiny{4000.0, 4000.0, 1e-14};
            const double x = kTwoPi * 4000.0;
            const double distance = vc3d::fiber_map::sheetDistanceVx(tiny, x);
            const double back = vc3d::fiber_map::sheetXForDistanceVx(tiny, distance);
            QVERIFY2(std::abs(back - x) < 1e-6 * x, qPrintable(QString::number(back)));
        }

        // The model is part of the result's identity.
        const ContentDigest baseline = vc3d::fiber_map::digestGlobalResult(result);
        GlobalResult tweaked = result;
        tweaked.sheetPitchVx += 1.0;
        QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        tweaked = result;
        tweaked.sheetRadius0Vx += 1.0;
        QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
    }

    // Too little winding span to fix a slope: the model falls back to the
    // reference radius with no pitch, and sheet distance is then exactly the
    // map's arclength.
    void sheetModelFallsBackWithoutWindingSpan()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const std::vector<InputFiber> fibers =
            makeWeave(100, QStringLiteral("a-"), 30000.0, 4000.0, 300.0,
                      -0.1, 0.8, {10, 120});
        const GlobalResult result =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QVERIFY(!result.fibers.empty());
        // The fallback under test is the short winding span, not an absence
        // of anchored fibers: the fit saw samples and had too little span.
        double lo = std::numeric_limits<double>::infinity();
        double hi = -std::numeric_limits<double>::infinity();
        int anchored = 0;
        for (const GlobalPlacedFiber& fiber : result.fibers) {
            if (fiber.meta.anchor == GlobalAnchor::Unresolved) {
                continue;
            }
            ++anchored;
            lo = std::min(lo, fiber.meta.windingLo);
            hi = std::max(hi, fiber.meta.windingHi);
        }
        QVERIFY(anchored > 0);
        QVERIFY2(hi - lo < 0.5, qPrintable(QString::number(hi - lo)));
        QCOMPARE(result.sheetPitchVx, 0.0);
        QCOMPARE(result.sheetRadius0Vx, result.rRefVx);
        const vc3d::fiber_map::SheetModel model = vc3d::fiber_map::sheetModelOf(result);
        const double x = 0.37 * kTwoPi * result.rRefVx;
        QVERIFY(std::abs(vc3d::fiber_map::sheetDistanceVx(model, x) - x) < 1e-9);
        QVERIFY(std::abs(vc3d::fiber_map::sheetXForDistanceVx(model, x) - x) < 1e-9);
    }

    // Equal labels tie-break by fileName, never by the runtime id: swapping
    // ids between builds must not move anything.
    void equalLabelOrderSurvivesIdSwap()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers = cacheFixture();
        fibers[0].label = fibers[1].label;
        std::vector<InputFiber> swapped = fibers;
        std::swap(swapped[0].id, swapped[1].id);
        // Links reference ids; the swap must follow them to keep the same
        // physical links.
        for (InputFiber& fiber : swapped) {
            for (InputLink& link : fiber.links) {
                if (link.branchFiberId == fibers[0].id) {
                    link.branchFiberId = fibers[1].id;
                } else if (link.branchFiberId == fibers[1].id) {
                    link.branchFiberId = fibers[0].id;
                }
            }
        }
        const GlobalResult a =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        const GlobalResult b =
            vc3d::fiber_map::buildGlobalLayout(swapped, umbilicus, defaultParams());
        QCOMPARE(a.fibers.size(), b.fibers.size());
        for (std::size_t i = 0; i < a.fibers.size(); ++i) {
            QCOMPARE(b.fibers[i].fiber.fileName, a.fibers[i].fiber.fileName);
            QCOMPARE(b.fibers[i].meta.windingLo, a.fibers[i].meta.windingLo);
            QCOMPARE(b.fibers[i].meta.windingHi, a.fibers[i].meta.windingHi);
        }
    }

    // The fixture helper itself fails loudly on a bad control index, in
    // every build type - a broken fixture must never silently read past its
    // line points (the bug this guards against shipped once).
    void fixtureHelperRejectsBadControlIndices()
    {
        bool threw = false;
        try {
            (void)makeFiber(999, QStringLiteral("bad"), 'H',
                            arcPoints(30000.0, 4000.0, 300.0, 0.0, 2.0),
                            {100, 400});
        } catch (const std::out_of_range&) {
            threw = true;
        }
        QVERIFY(threw);
    }

    // No umbilicus: nothing can be unrolled, and EVERY fiber - not just the
    // geometryless one - is reported unplaceable rather than silently absent.
    void noUmbilicusReportsEveryFiberUnplaceable()
    {
        InputFiber empty;
        empty.id = 901;
        empty.fileName = "broken.json";
        empty.label = QStringLiteral("broken");
        InputFiber whole = makeFiber(902, QStringLiteral("whole"), 'H',
                                     arcPoints(30000.0, 4000.0, 300.0, 0.0, 2.0),
                                     {100, 399});
        const GlobalResult result = vc3d::fiber_map::buildGlobalLayout(
            {empty, whole}, {}, defaultParams());
        QVERIFY(result.fibers.empty());
        QCOMPARE(result.unplaced.size(), std::size_t{2});
    }

    // Geometry too degenerate to draw is unplaceable too: a one-point trace
    // must not become a placed fiber the map never shows.
    void degenerateGeometryIsReportedUnplaceable()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers =
            makeWeave(100, QStringLiteral("a-"), 30000.0, 4000.0, 300.0,
                      0.0, 1.5 * kTwoPi, {200, 900, 1600});
        InputFiber dot;
        dot.id = 903;
        dot.fileName = "dot.json";
        dot.label = QStringLiteral("dot");
        dot.hvTag = 'V';
        dot.linePoints.push_back(cv::Vec3d(4000.0, 0.0, 30000.0));
        dot.controlPoints.push_back(dot.linePoints.front());
        fibers.push_back(dot);
        const GlobalResult result =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QCOMPARE(result.fibers.size(), std::size_t{4});
        QCOMPARE(result.unplaced.size(), std::size_t{1});
        QCOMPARE(result.unplaced.front().id, uint64_t{903});
        QVERIFY(findFiber(result, 903) == nullptr);
    }

    // --- Kollesis terminations: a display-only per-control-point flag that
    // must reach the placed fiber aligned to its control points, must not
    // touch the geometry cache keys, and must move both session digests.
    void kollesisTerminationsReachThePlacedFiberWithoutRecomputingGeometry()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const GlobalLayoutParams params = defaultParams();
        std::vector<InputFiber> fibers = cacheFixture();
        const uint64_t taggedId = fibers.front().id;
        const std::size_t controlCount = fibers.front().controlPoints.size();
        QVERIFY(controlCount >= 2);

        vc3d::fiber_map::GlobalLayoutCache cache;
        const GlobalResult untagged =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        const GlobalPlacedFiber* plain = findFiber(untagged, taggedId);
        QVERIFY(plain != nullptr);
        QCOMPARE(plain->fiber.kollesisTerminations.size(), plain->fiber.controlPoints.size());
        QVERIFY(std::none_of(plain->fiber.kollesisTerminations.begin(),
                             plain->fiber.kollesisTerminations.end(),
                             [](bool tagged) { return tagged; }));

        fibers.front().kollesisTerminations.assign(controlCount, false);
        fibers.front().kollesisTerminations.back() = true;
        const GlobalResult tagged =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        const GlobalPlacedFiber* placed = findFiber(tagged, taggedId);
        QVERIFY(placed != nullptr);
        QCOMPARE(placed->fiber.kollesisTerminations.size(), placed->fiber.controlPoints.size());
        QVERIFY(placed->fiber.kollesisTerminations.back());
        QVERIFY(!placed->fiber.kollesisTerminations.front());
        // Same geometry: every cached slot was reused.
        QCOMPARE(cache.lastStats().fibersRecomputed, 0);
        QCOMPARE(cache.lastStats().pairsRecomputed, 0);
        // Still an input change the memoization check must see, on both sides.
        QVERIFY(!(vc3d::fiber_map::digestGlobalInputs(fibers, umbilicus, params) ==
                  vc3d::fiber_map::digestGlobalInputs(cacheFixture(), umbilicus, params)));
        QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tagged) ==
                  vc3d::fiber_map::digestGlobalResult(untagged)));

        // A flag vector that does not match the control points is ignored
        // rather than read misaligned: it still carries a true flag, so a
        // prefix copy would be caught.
        fibers.front().kollesisTerminations.front() = true;
        fibers.front().kollesisTerminations.pop_back();
        const GlobalResult mismatched =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        const GlobalPlacedFiber* ignored = findFiber(mismatched, taggedId);
        QVERIFY(ignored != nullptr);
        QCOMPARE(ignored->fiber.kollesisTerminations.size(), ignored->fiber.controlPoints.size());
        QVERIFY(std::none_of(ignored->fiber.kollesisTerminations.begin(),
                             ignored->fiber.kollesisTerminations.end(),
                             [](bool flagged) { return flagged; }));
    }

    // A folded pair's crossings are read together: one group with a verdict,
    // every event carried out for inspection, no rings while the map honours
    // the verdict - and the verdict recovers the winding gap of one.
    void foldedPairIsReadAsOneGroup()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        for (const bool mirror : {false, true}) {
            std::vector<InputFiber> fibers = hairpinPair(false);
            if (mirror) {
                fibers = mirrored(fibers);
            }
            const GlobalResult result =
                vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
            QCOMPARE(result.crossingEvents.size(), std::size_t{3});
            QCOMPARE(result.crossingGroups.size(), std::size_t{1});
            const auto& group = result.crossingGroups.front();
            QCOMPARE(group.hFiberId, uint64_t{700});
            QCOMPARE(group.vFiberId, uint64_t{701});
            QCOMPARE(group.multiplicity, 3);
            QCOMPARE(group.insideCount, 2);
            QVERIFY(group.mixedSigns);
            QVERIFY(group.hasVerdict);
            QCOMPARE(group.verdict, vc3d::fiber_map::winding::CrossingKind::Outside);
            QCOMPARE(group.members.size(), std::size_t{3});
            for (const auto& event : result.crossingEvents) {
                QCOMPARE(event.status, vc3d::fiber_map::winding::CrossingStatus::InGroup);
                QCOMPARE(event.groupId, 0LL);
                QCOMPARE(event.hFiberId, uint64_t{700});
            }
            QCOMPARE(result.traversalGroupCount, 1);
            QCOMPARE(result.declaredGroupCount, 0);
            QCOMPARE(result.droppedCrossingCount, 0);
            QVERIFY(result.suspectCrossings.empty());
            const GlobalPlacedFiber* h = findFiber(result, 700);
            const GlobalPlacedFiber* v = findFiber(result, 701);
            QVERIFY(h != nullptr && v != nullptr);
            // H strictly outside V: a whole winding between them.
            QVERIFY(h->meta.windingLo > v->meta.windingHi + 0.5);
        }
    }

    // The same pair with a same-winding link the verdict contradicts: the
    // stronger link holds, the group is dropped as one unit and declared as
    // one conflict, marked at each of its three places with a shared group.
    void droppedGroupIsOneConflictMarkedAtEachMember()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const std::vector<InputFiber> fibers = hairpinPair(true);
        const GlobalResult result =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QCOMPARE(result.crossingGroups.size(), std::size_t{1});
        const auto& group = result.crossingGroups.front();
        QVERIFY(group.hasVerdict);
        QCOMPARE(group.status, vc3d::fiber_map::winding::CrossingStatus::Dropped);
        QCOMPARE(group.violationTurns, 1.0);
        QCOMPARE(result.declaredGroupCount, 1);
        QCOMPARE(result.droppedCrossingCount, 0);
        QCOMPARE(result.suspectLinkCount, 0);
        QCOMPARE(result.suspectCrossings.size(), std::size_t{3});
        for (const auto& mark : result.suspectCrossings) {
            QCOMPARE(mark.groupId, 0LL);
            QCOMPARE(mark.violationTurns, 1.0);
            QCOMPARE(mark.hFiberId, uint64_t{700});
            QCOMPARE(mark.vFiberId, uint64_t{701});
            QVERIFY(mark.eventIndex < result.crossingEvents.size());
            QCOMPARE(mark.posVx, result.crossingEvents[mark.eventIndex].posVx);
        }
        const GlobalPlacedFiber* h = findFiber(result, 700);
        const GlobalPlacedFiber* v = findFiber(result, 701);
        QVERIFY(h != nullptr && v != nullptr);
        QVERIFY(std::abs(h->meta.windingLo - v->meta.windingLo) < 0.6);
    }

    // The cache's contract, shard by shard: two independent cold builds of
    // the same input hold bit-identical detection shards - every raw and
    // shallow detection with its provenance, and the gate tallies - and a
    // moved fiber changes some shard.
    void cachedShardsAreTheFreshOnesBitForBit()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers = hairpinPair(false);
        std::vector<InputFiber> weave = cacheFixture();
        fibers.insert(fibers.end(), weave.begin(), weave.end());
        const GlobalLayoutParams params = defaultParams();
        vc3d::fiber_map::GlobalLayoutCache first;
        vc3d::fiber_map::GlobalLayoutCache second;
        vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &first);
        vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &second);
        const auto shardsA = first.cachedDetections();
        const auto shardsB = second.cachedDetections();
        QCOMPARE(shardsA.size(), shardsB.size());
        QVERIFY(!shardsA.empty());
        bool sawDetections = false;
        for (std::size_t i = 0; i < shardsA.size(); ++i) {
            QVERIFY(vc3d::fiber_map::winding::identicalPairDetections(*shardsA[i], *shardsB[i]));
            sawDetections = sawDetections || !shardsA[i]->raw.empty();
        }
        QVERIFY(sawDetections);
        // Nudge the folded V fiber and rebuild INTO the first cache: exactly
        // the shards it takes part in (one per H fiber) recompute, and every
        // shard the warmed cache then holds - recomputed or reused - is the
        // one an independent cold build produces.
        for (cv::Vec3d& point : fibers[1].linePoints) {
            point[2] += 30.0;
        }
        fibers[1].controlPoints[1] = fibers[1].linePoints[40];
        vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &first);
        int hFibers = 0;
        for (const InputFiber& fiber : fibers) {
            hFibers += fiber.hvTag == 'H' ? 1 : 0;
        }
        QCOMPARE(first.lastStats().pairsRecomputed, hFibers);
        QVERIFY(first.lastStats().pairsReused > 0);
        vc3d::fiber_map::GlobalLayoutCache third;
        vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &third);
        const auto shardsWarm = first.cachedDetections();
        const auto shardsC = third.cachedDetections();
        QCOMPARE(shardsWarm.size(), shardsC.size());
        for (std::size_t i = 0; i < shardsC.size(); ++i) {
            QVERIFY(vc3d::fiber_map::winding::identicalPairDetections(*shardsWarm[i], *shardsC[i]));
        }
        // And the move did change some shard against the original build
        // (recomputation need not change every affected shard's output).
        int differing = 0;
        for (std::size_t i = 0; i < shardsB.size(); ++i) {
            if (!vc3d::fiber_map::winding::identicalPairDetections(*shardsB[i], *shardsC[i])) {
                ++differing;
            }
        }
        QVERIFY(differing >= 1);
        QVERIFY(differing <= hFibers);
    }

    // --- Kollesis.

    // A V fiber linked at the tagged ends of two H fibers departing to
    // opposite sides is on the kollesis: its seam encounter with the inner
    // H fiber, which reads Outside by a thickness, is read as Inside and
    // flagged; nothing is declared, and all three fibers share the winding.
    void kollesisVIsIdentifiedByLinksToTaggedEnds()
    {
        checkKollesisSeam(false, false);
        checkKollesisSeam(true, false);
        checkKollesisSeam(false, true);
    }

    // What does NOT identify a kollesis V: both H fibers ending on the same
    // side (linked at the tags or at the crossings); only one link; a link
    // to an untagged H fiber. In each the seam crossing keeps its Outside
    // reading and, opposed by the link, is declared as before.
    void kollesisIdentificationNeedsTwoSidesTagsAndLinks()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        struct Case {
            bool sameSide;
            int linkMask;
            int tagMask;
            bool linkAtCrossing;
        };
        for (const Case& c : {Case{true, 3, 3, false}, Case{true, 3, 3, true},
                              Case{false, 1, 3, false}, Case{false, 3, 2, false}}) {
            const GlobalResult result = vc3d::fiber_map::buildGlobalLayout(
                kollesisSeam(c.sameSide, c.linkMask, c.tagMask, false, c.linkAtCrossing),
                umbilicus, defaultParams());
            const GlobalPlacedFiber* v = findFiber(result, 802);
            QVERIFY(v != nullptr);
            QVERIFY(!v->meta.onKollesis);
            QCOMPARE(result.kollesisCrossingCount, 0);
            bool sawOutside = false;
            for (const auto& event : result.crossingEvents) {
                if (event.hFiberId == 800 && event.vFiberId == 802) {
                    QVERIFY(!event.kollesis);
                    sawOutside = sawOutside ||
                                 event.kind == vc3d::fiber_map::winding::CrossingKind::Outside;
                }
            }
            QVERIFY(sawOutside);
        }
    }

    // The solve finds the seam encounters no tag names: on the certified
    // kollesis V, a third inner H fiber, untagged, linked to the tagged
    // inner H fiber (so the rest of its evidence puts it on the V's winding)
    // and ending just past the V, has its Outside crossing read as an
    // inferred seam: no ring, flagged. Unlinked, nothing contradicts the
    // crossing and nothing is inferred; running a full turn on past the V,
    // the crossing is not terminal and its ring stays.
    void inferredSeamsClearTheUntaggedInnerFibers()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        for (const int extraInner : {1, 2, 3}) {
            const GlobalResult result = vc3d::fiber_map::buildGlobalLayout(
                kollesisSeam(false, 3, 3, false, false, extraInner), umbilicus, defaultParams());
            const GlobalPlacedFiber* v = findFiber(result, 802);
            QVERIFY(v != nullptr && v->meta.onKollesis);
            int extraEvents = 0;
            int extraInferred = 0;
            int extraDropped = 0;
            for (const auto& event : result.crossingEvents) {
                if (event.hFiberId != 803 || event.vFiberId != 802) {
                    continue;
                }
                ++extraEvents;
                extraInferred += event.kollesisInferred ? 1 : 0;
                extraDropped += event.status == vc3d::fiber_map::winding::CrossingStatus::Dropped ? 1 : 0;
                if (event.kollesisInferred) {
                    QVERIFY(event.kollesis);
                    QCOMPARE(event.kind, vc3d::fiber_map::winding::CrossingKind::Inside);
                    QVERIFY(event.deltaR > 0.0);
                }
            }
            QVERIFY(extraEvents >= 1);
            QCOMPARE(result.kollesisInferredCount, extraInferred);
            if (extraInner == 1) {
                QCOMPARE(extraInferred, extraEvents);
                QCOMPARE(extraDropped, 0);
                QCOMPARE(result.droppedCrossingCount, 0);
                const GlobalPlacedFiber* extra = findFiber(result, 803);
                QVERIFY(extra != nullptr);
                QVERIFY(std::abs(extra->meta.windingLo - v->meta.windingLo) < 0.6);
            } else if (extraInner == 2) {
                QCOMPARE(extraInferred, 0);
                QCOMPARE(extraDropped, 0);
            } else {
                QCOMPARE(extraInferred, 0);
                QVERIFY(extraDropped >= 1);
                QVERIFY(result.droppedCrossingCount >= 1);
            }
        }
    }

    // Tags and links are annotation: adding them recomputes no detection
    // shard, yet changes the classified result, and the memoized build
    // equals the fresh one throughout. Every new field is in the digest.
    void kollesisFlagsInvalidateNoShardsAndAreDigested()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const GlobalLayoutParams params = defaultParams();
        vc3d::fiber_map::GlobalLayoutCache cache;
        const GlobalResult plain = vc3d::fiber_map::buildGlobalLayout(
            kollesisSeam(false, 0, 0), umbilicus, params, &cache);
        QCOMPARE(plain.kollesisCrossingCount, 0);
        const GlobalResult warm = vc3d::fiber_map::buildGlobalLayout(
            kollesisSeam(false, 3, 3), umbilicus, params, &cache);
        QCOMPARE(cache.lastStats().pairsRecomputed, 0);
        QVERIFY(warm.kollesisCrossingCount >= 1);
        const GlobalResult fresh = vc3d::fiber_map::buildGlobalLayout(
            kollesisSeam(false, 3, 3), umbilicus, params);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warm) ==
                vc3d::fiber_map::digestGlobalResult(fresh));
        QVERIFY(!(vc3d::fiber_map::digestGlobalResult(warm) ==
                  vc3d::fiber_map::digestGlobalResult(plain)));

        const ContentDigest baseline = vc3d::fiber_map::digestGlobalResult(fresh);
        {
            GlobalResult tweaked = fresh;
            tweaked.kollesisCrossingCount += 1;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = fresh;
            for (auto& fiber : tweaked.fibers) {
                if (fiber.fiber.id == 802) {
                    fiber.meta.onKollesis = false;
                }
            }
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = fresh;
            bool flipped = false;
            for (auto& event : tweaked.crossingEvents) {
                if (event.kollesis && !flipped) {
                    event.kollesis = false;
                    flipped = true;
                }
            }
            QVERIFY(flipped);
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = fresh;
            tweaked.crossingEvents.front().kollesisInferred =
                !tweaked.crossingEvents.front().kollesisInferred;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = fresh;
            tweaked.kollesisInferredCount += 1;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
    }

    // Groups are classified from the memoized detections: cached and fresh
    // builds of a folded pair are identical, and every exported group and
    // event field is in the result digest.
    void groupsAreCachedAndDigested()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers = hairpinPair(false);
        std::vector<InputFiber> weave = cacheFixture();
        fibers.insert(fibers.end(), weave.begin(), weave.end());
        const GlobalLayoutParams params = defaultParams();
        const GlobalResult fresh =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params);
        vc3d::fiber_map::GlobalLayoutCache cache;
        const GlobalResult cold =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        const GlobalResult warm =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(cold) ==
                vc3d::fiber_map::digestGlobalResult(fresh));
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warm) ==
                vc3d::fiber_map::digestGlobalResult(fresh));
        QCOMPARE(cache.lastStats().pairsRecomputed, 0);
        QCOMPARE(warm.traversalGroupCount, 1);
        QCOMPARE(warm.crossingGroups.size(), fresh.crossingGroups.size());
        // Adding a link changes no shard, only the solve.
        addLink(fibers[0], 1, fibers[1], 1);
        const GlobalResult freshLinked =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params);
        const GlobalResult warmLinked =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warmLinked) ==
                vc3d::fiber_map::digestGlobalResult(freshLinked));
        QCOMPARE(cache.lastStats().pairsRecomputed, 0);
        QCOMPARE(warmLinked.declaredGroupCount, 1);
        // The endpoint clearance is a detection parameter: changing it
        // recomputes every pair.
        GlobalLayoutParams strict = params;
        strict.solver.endpointClearanceTurns = 0.02;
        const GlobalResult freshStrict =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, strict);
        const GlobalResult warmStrict =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, strict, &cache);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warmStrict) ==
                vc3d::fiber_map::digestGlobalResult(freshStrict));
        QVERIFY(cache.lastStats().pairsRecomputed > 0);

        const ContentDigest baseline = vc3d::fiber_map::digestGlobalResult(freshLinked);
        {
            GlobalResult tweaked = freshLinked;
            tweaked.crossingGroups[0].hasVerdict = false;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = freshLinked;
            tweaked.crossingGroups[0].insideCount += 1;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = freshLinked;
            tweaked.crossingGroups[0].orientationSum += 2;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = freshLinked;
            tweaked.crossingEvents[0].orientation = -tweaked.crossingEvents[0].orientation;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = freshLinked;
            // The folded pair sorts after the weave: take one of its events.
            tweaked.crossingEvents[tweaked.crossingGroups[0].members[0]].groupId = -1;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = freshLinked;
            QVERIFY(!tweaked.suspectCrossings.empty());
            tweaked.suspectCrossings[0].groupId = -1;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = freshLinked;
            tweaked.declaredGroupCount += 1;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = freshLinked;
            tweaked.traversalGroupCount += 1;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = freshLinked;
            const std::size_t e = tweaked.crossingGroups[0].members[0];
            tweaked.crossingEvents[e].confidence += 0.25;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = freshLinked;
            const std::size_t e = tweaked.crossingGroups[0].members[0];
            tweaked.crossingEvents[e].touch = !tweaked.crossingEvents[e].touch;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = freshLinked;
            tweaked.unresolvedIntersectionCount += 1;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
    }
};

QTEST_APPLESS_MAIN(TestFiberGlobalLayout)
#include "test_fiber_global_layout.moc"
