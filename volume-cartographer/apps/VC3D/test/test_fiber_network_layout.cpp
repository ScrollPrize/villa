#include <QtTest/QtTest>

#include <cmath>
#include <vector>

#include "FiberNetworkLayout.hpp"

using vc3d::fiber_map::InputFiber;
using vc3d::fiber_map::PlacedFiber;
using vc3d::fiber_map::LayoutParams;
using vc3d::fiber_map::PlacedNetwork;
using vc3d::fiber_map::PlacedLink;
using vc3d::fiber_map::Result;

namespace
{

constexpr double kTwoPi = 2.0 * M_PI;
// Line-point angular sampling; a whole turn is exactly this many steps, so
// control points one turn apart differ by exactly 2*pi.
constexpr int kStepsPerTurn = 1256;
constexpr double kStep = kTwoPi / static_cast<double>(kStepsPerTurn);
constexpr double kVoxelUm = 2.4;
constexpr double kToCm = kVoxelUm / 10000.0;

std::vector<cv::Vec3f> straightUmbilicus(int zMax)
{
    std::vector<cv::Vec3f> centers;
    centers.reserve(static_cast<std::size_t>(zMax) + 1);
    for (int z = 0; z <= zMax; ++z) {
        centers.push_back(cv::Vec3f(0.0f, 0.0f, static_cast<float>(z)));
    }
    return centers;
}

// Analytic arc about the umbilicus: an H fiber winding around the scroll while
// its radius grows with the absolute angle, so points one turn apart stay
// distinct in 3D and extending the angular range only adds points.
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

// A V fiber: a vertical line at a fixed angle about the umbilicus.
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
    // The stable identity the workspace acts through; derived from the label so
    // the assertions can tell placed fibers apart without their runtime ids.
    fiber.fileName = label.toStdString() + ".json";
    fiber.label = label;
    fiber.hvTag = hvTag;
    fiber.linePoints = std::move(linePoints);
    for (int index : controlIndices) {
        fiber.controlPoints.push_back(fiber.linePoints[static_cast<std::size_t>(index)]);
    }
    if (fiber.controlPoints.size() > 1) {
        fiber.tracedSegments.assign(fiber.controlPoints.size() - 1, true);
    }
    return fiber;
}

// Reciprocal refs on both fibers, the way the annotation stores them.
void addLink(InputFiber& a, int controlA, InputFiber& b, int controlB)
{
    a.links.push_back({controlA, b.id, controlB});
    b.links.push_back({controlB, a.id, controlA});
}

double angleOf(const cv::Vec3d& point)
{
    return std::atan2(point[1], point[0]);
}

const PlacedLink* findLink(const PlacedNetwork& network, uint64_t fiberA, int cpA,
                           uint64_t fiberB, int cpB)
{
    for (const PlacedLink& link : network.links) {
        if ((link.fiberA == fiberA && link.cpA == cpA && link.fiberB == fiberB &&
             link.cpB == cpB) ||
            (link.fiberA == fiberB && link.cpA == cpB && link.fiberB == fiberA &&
             link.cpB == cpA)) {
            return &link;
        }
    }
    return nullptr;
}

LayoutParams defaultParams()
{
    LayoutParams params;
    params.voxelSizeUm = kVoxelUm;
    return params;
}

// H fiber plus one V fiber per requested crossing, all linked at the crossings.
struct Weave {
    std::vector<InputFiber> fibers;
    std::vector<int> horizontalControlIndices;
};

Weave makeWeave(uint64_t firstId, const QString& prefix, double z, double radius,
                double radiusPerTurn, double thetaBegin, double thetaEnd,
                const std::vector<int>& controlIndices)
{
    Weave weave;
    weave.horizontalControlIndices = controlIndices;
    std::vector<cv::Vec3d> line =
        arcPoints(z, radius, radiusPerTurn, thetaBegin, thetaEnd);
    InputFiber horizontal = makeFiber(firstId, prefix + QStringLiteral("h-1"), 'H',
                                      line, controlIndices);
    weave.fibers.push_back(std::move(horizontal));

    for (std::size_t i = 0; i < controlIndices.size(); ++i) {
        const cv::Vec3d crossing = weave.fibers.front().controlPoints[i];
        std::vector<cv::Vec3d> verticalLine =
            verticalPoints(angleOf(crossing), std::hypot(crossing[0], crossing[1]),
                           z - 400.0, z + 400.0, 4.0);
        // Control points at both ends and at the crossing.
        const int last = static_cast<int>(verticalLine.size()) - 1;
        InputFiber vertical = makeFiber(firstId + 1 + i,
                                        prefix + QStringLiteral("v-%1").arg(i + 1), 'V',
                                        std::move(verticalLine), {0, last / 2, last});
        addLink(weave.fibers.front(), static_cast<int>(i), vertical, 1);
        weave.fibers.push_back(std::move(vertical));
    }
    return weave;
}

std::vector<InputFiber> collect(const std::vector<Weave*>& weaves,
                                const std::vector<InputFiber>& extra = {})
{
    std::vector<InputFiber> fibers;
    for (Weave* weave : weaves) {
        for (InputFiber& fiber : weave->fibers) {
            fibers.push_back(fiber);
        }
    }
    for (const InputFiber& fiber : extra) {
        fibers.push_back(fiber);
    }
    return fibers;
}

} // namespace

class TestFiberNetworkLayout : public QObject
{
    Q_OBJECT

private slots:
    void componentsAndFiltering()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        // Outer network: 4 fibers. Inner network: 3 fibers. Plus one isolated.
        Weave outer = makeWeave(100, QStringLiteral("a-"), 30000.0, 6000.0, 300.0,
                                0.0, 1.5 * kTwoPi, {200, 900, 1600});
        Weave inner = makeWeave(200, QStringLiteral("b-"), 30000.0, 3000.0, 200.0,
                                0.0, 1.2 * kTwoPi, {150, 1100});
        const InputFiber isolated =
            makeFiber(900, QStringLiteral("c-h-9"), 'H',
                      arcPoints(31000.0, 5000.0, 100.0, 0.0, 0.8 * kTwoPi), {100, 800});
        const std::vector<InputFiber> fibers = collect({&outer, &inner}, {isolated});
        QCOMPARE(fibers.size(), std::size_t{8});

        LayoutParams params = defaultParams();
        params.minFibers = 3;
        params.maxNetworks = 3;
        Result result = vc3d::fiber_map::buildLayout(fibers, umbilicus, params);
        QCOMPARE(result.qualifyingNetworkCount, 2);
        QCOMPARE(result.networks.size(), std::size_t{2});
        // Panels run inner -> outer by median umbilicus radius; the inner
        // network is the smaller one, so it carries network index 1.
        QVERIFY(result.networks[0].rRefCm < result.networks[1].rRefCm);
        QCOMPARE(result.networks[0].networkIndex, 1);
        QCOMPARE(result.networks[1].networkIndex, 0);
        QCOMPARE(result.networks[0].fibers.size(), std::size_t{3});
        QCOMPARE(result.networks[1].fibers.size(), std::size_t{4});

        params.maxNetworks = 1;
        const Result single = vc3d::fiber_map::buildLayout(fibers, umbilicus, params);
        QCOMPARE(single.qualifyingNetworkCount, 2);
        QCOMPARE(single.networks.size(), std::size_t{1});
        QCOMPARE(single.networks[0].networkIndex, 0);
        QCOMPARE(single.networks[0].fibers.size(), std::size_t{4});

        params.maxNetworks = 3;
        params.minFibers = 5;
        const Result none = vc3d::fiber_map::buildLayout(fibers, umbilicus, params);
        QCOMPARE(none.qualifyingNetworkCount, 0);
        QVERIFY(none.networks.empty());

        // Two runs on the same input must be identical.
        params.minFibers = 3;
        const Result repeated = vc3d::fiber_map::buildLayout(fibers, umbilicus, params);
        QCOMPARE(repeated.networks.size(), result.networks.size());
        QVERIFY(repeated.widthCm == result.widthCm);
        for (std::size_t i = 0; i < result.networks.size(); ++i) {
            const PlacedNetwork& a = result.networks[i];
            const PlacedNetwork& b = repeated.networks[i];
            QVERIFY(b.rRefCm == a.rRefCm);
            QVERIFY(b.x0Cm == a.x0Cm);
            QCOMPARE(b.fibers.size(), a.fibers.size());
            for (std::size_t f = 0; f < a.fibers.size(); ++f) {
                QCOMPARE(b.fibers[f].label, a.fibers[f].label);
                QCOMPARE(b.fibers[f].controlPoints.size(), a.fibers[f].controlPoints.size());
                for (std::size_t p = 0; p < a.fibers[f].controlPoints.size(); ++p) {
                    QVERIFY(b.fibers[f].controlPoints[p].x() ==
                            a.fibers[f].controlPoints[p].x());
                    QVERIFY(b.fibers[f].controlPoints[p].y() ==
                            a.fibers[f].controlPoints[p].y());
                }
            }
            QCOMPARE(b.links.size(), a.links.size());
            for (std::size_t l = 0; l < a.links.size(); ++l) {
                QVERIFY(b.links[l].turnErr == a.links[l].turnErr);
                QCOMPARE(b.links[l].fiberA, a.links[l].fiberA);
                QCOMPARE(b.links[l].cpA, a.links[l].cpA);
            }
        }
    }

    void offsetSnappingClosesLoops()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const double z = 30000.0;
        std::vector<cv::Vec3d> horizontalLine =
            arcPoints(z, 4000.0, 300.0, -0.4, kTwoPi + 0.4);
        // Two crossings exactly one turn apart along the H fiber.
        const int firstControl = 100;
        const int secondControl = firstControl + kStepsPerTurn;
        InputFiber horizontal = makeFiber(1, QStringLiteral("h-1"), 'H', horizontalLine,
                                          {firstControl, secondControl});

        // Each V fiber sits at the crossing's own angle, whose atan2 branch
        // differs from the H fiber's unwrapped angle by a whole turn on the
        // second crossing.
        std::vector<InputFiber> verticals;
        for (int index = 0; index < 2; ++index) {
            const cv::Vec3d& crossing = horizontal.controlPoints[static_cast<std::size_t>(index)];
            const double theta = angleOf(crossing);
            const double radius = std::hypot(crossing[0], crossing[1]);
            verticals.push_back(makeFiber(
                static_cast<uint64_t>(2 + index),
                QStringLiteral("v-%1").arg(index + 1), 'V',
                verticalPoints(theta, radius, z - 400.0, z + 400.0, 4.0),
                {0, 100, 200}));
            addLink(horizontal, index, verticals.back(), 1);
        }

        // Pending lives on both reciprocal refs; marking only the H side of the
        // first crossing leaves the dedup to OR the pair back together.
        horizontal.links[0].pending = true;

        std::vector<InputFiber> fibers{horizontal, verticals[0], verticals[1]};
        LayoutParams params = defaultParams();
        params.smoothMm = 0.0;
        const Result result = vc3d::fiber_map::buildLayout(fibers, umbilicus, params);
        QCOMPARE(result.networks.size(), std::size_t{1});
        const PlacedNetwork& network = result.networks.front();
        QCOMPARE(network.links.size(), std::size_t{2});
        QCOMPARE(result.suspectLinkCount, 0);

        for (const PlacedLink& link : network.links) {
            QVERIFY2(link.turnErr < 1e-9, qPrintable(QString::number(link.turnErr)));
            QVERIFY(!link.suspect);
            // The crossing coincides by construction.
            QVERIFY(std::abs(link.a.x() - link.b.x()) < 0.01);
            QVERIFY(std::abs(link.a.y() - link.b.y()) < 0.01);
        }

        const PlacedLink* first = findLink(network, 1, 0, 2, 1);
        const PlacedLink* second = findLink(network, 1, 1, 3, 1);
        QVERIFY(first != nullptr);
        QVERIFY(second != nullptr);
        QVERIFY(first->pending);
        QVERIFY(!second->pending);

        // The stable identity has to reach the placed fibers: the workspace
        // resolves navigation through it once runtime ids have been reassigned.
        for (const PlacedFiber& placed : network.fibers) {
            QCOMPARE(QString::fromStdString(placed.fileName),
                     placed.label + QStringLiteral(".json"));
        }
        const double gap = std::abs(second->a.x() - first->a.x());
        QVERIFY2(std::abs(gap - kTwoPi * network.rRefCm) < 0.02,
                 qPrintable(QStringLiteral("gap %1 vs %2")
                                .arg(gap)
                                .arg(kTwoPi * network.rRefCm)));

        // Whichever ref of the pair the dedup sees first, the merged link is
        // pending: here only the V side of the first crossing carries the flag.
        std::vector<InputFiber> reciprocal = fibers;
        reciprocal[0].links[0].pending = false;
        reciprocal[1].links[0].pending = true;
        const Result mirrored = vc3d::fiber_map::buildLayout(reciprocal, umbilicus, params);
        QCOMPARE(mirrored.networks.size(), std::size_t{1});
        const PlacedLink* mirroredFirst = findLink(mirrored.networks.front(), 1, 0, 2, 1);
        const PlacedLink* mirroredSecond = findLink(mirrored.networks.front(), 1, 1, 3, 1);
        QVERIFY(mirroredFirst != nullptr);
        QVERIFY(mirroredSecond != nullptr);
        QVERIFY(mirroredFirst->pending);
        QVERIFY(!mirroredSecond->pending);
    }

    void suspectLinkDetection()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const double z = 30000.0;
        std::vector<cv::Vec3d> horizontalLine =
            arcPoints(z, 4000.0, 300.0, -0.4, kTwoPi + 0.4);
        const int firstControl = 100;
        const int secondControl = firstControl + kStepsPerTurn;
        InputFiber horizontal = makeFiber(1, QStringLiteral("h-1"), 'H', horizontalLine,
                                          {firstControl, secondControl});
        std::vector<InputFiber> verticals;
        for (int index = 0; index < 2; ++index) {
            const cv::Vec3d& crossing = horizontal.controlPoints[static_cast<std::size_t>(index)];
            const double theta = angleOf(crossing);
            const double radius = std::hypot(crossing[0], crossing[1]);
            verticals.push_back(makeFiber(
                static_cast<uint64_t>(2 + index),
                QStringLiteral("v-%1").arg(index + 1), 'V',
                verticalPoints(theta, radius, z - 400.0, z + 400.0, 4.0),
                {0, 100, 200}));
            addLink(horizontal, index, verticals.back(), 1);
        }
        // Deliberately wrong winding: the second H crossing also claims the
        // first V fiber, one turn away from where they meet.
        addLink(horizontal, 1, verticals[0], 1);

        std::vector<InputFiber> fibers{horizontal, verticals[0], verticals[1]};
        LayoutParams params = defaultParams();
        params.smoothMm = 0.0;
        const Result result = vc3d::fiber_map::buildLayout(fibers, umbilicus, params);
        QCOMPARE(result.networks.size(), std::size_t{1});
        const PlacedNetwork& network = result.networks.front();
        QCOMPARE(network.links.size(), std::size_t{3});
        QCOMPARE(result.suspectLinkCount, 1);

        const PlacedLink* wrong = findLink(network, 1, 1, 2, 1);
        QVERIFY(wrong != nullptr);
        QVERIFY(wrong->suspect);
        QVERIFY2(std::abs(wrong->turnErr - 1.0) < 1e-9,
                 qPrintable(QString::number(wrong->turnErr)));
        for (const PlacedLink& link : network.links) {
            if (&link == wrong) {
                continue;
            }
            // The clean links won placement, so they still close exactly.
            QVERIFY(link.turnErr < 1e-9);
            QVERIFY(!link.suspect);
        }
    }

    void unitsAndClipping()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const double z = 30000.0;
        const double turns = 2.0;
        const int lastControl = static_cast<int>(turns) * kStepsPerTurn;

        const auto build = [&](double tailRadians) {
            Weave wide = makeWeave(100, QStringLiteral("a-"), z, 4000.0, 300.0,
                                   -tailRadians, turns * kTwoPi + tailRadians,
                                   {static_cast<int>(std::lround(tailRadians / kStep)),
                                    static_cast<int>(std::lround(tailRadians / kStep)) +
                                        kStepsPerTurn / 2,
                                    static_cast<int>(std::lround(tailRadians / kStep)) +
                                        lastControl});
            Weave narrow = makeWeave(300, QStringLiteral("b-"), z, 1500.0, 100.0, 0.0,
                                     1.4 * kTwoPi, {100, 900});
            const std::vector<InputFiber> fibers = collect({&wide, &narrow});
            LayoutParams params = defaultParams();
            params.smoothMm = 0.0;
            params.maxNetworks = 2;
            return vc3d::fiber_map::buildLayout(fibers, umbilicus, params);
        };

        const Result tight = build(0.0);
        const Result tailed = build(1.0);
        QCOMPARE(tight.networks.size(), std::size_t{2});
        QCOMPARE(tailed.networks.size(), std::size_t{2});

        // The outer, wider network is the second panel.
        const PlacedNetwork& wide = tight.networks[1];
        QVERIFY(wide.rRefCm > tight.networks[0].rRefCm);
        const double circumference = turns * kTwoPi * wide.rRefCm;
        const double pad = std::max(0.05 * circumference, 2.2);
        const double span = wide.x1Cm - wide.x0Cm;
        QVERIFY2(std::abs(span - (circumference + 2.0 * pad)) < 0.1,
                 qPrintable(QStringLiteral("span %1 vs %2")
                                .arg(span)
                                .arg(circumference + 2.0 * pad)));

        // Line-point tails a radian beyond the outermost control points would
        // add ~2 * 1.0 * rRefCm to the panel; they are clipped out instead.
        const double tailedSpan = tailed.networks[1].x1Cm - tailed.networks[1].x0Cm;
        QVERIFY2(std::abs(tailedSpan - span) < 0.1,
                 qPrintable(QStringLiteral("tailed span %1 vs %2")
                                .arg(tailedSpan)
                                .arg(span)));
        QVERIFY(tailed.networks[1].x0Cm == wide.x0Cm);

        // Unrolled length starts at 0 and every later panel starts on the 5 cm
        // grid; winding numbers run continuously across the panels.
        QVERIFY(std::abs(tight.networks[0].x0Cm) < 1e-12);
        const double gridSteps = tight.networks[1].x0Cm / 5.0;
        QVERIFY2(std::abs(gridSteps - std::round(gridSteps)) < 1e-9,
                 qPrintable(QString::number(tight.networks[1].x0Cm)));
        QVERIFY(tight.networks[1].x0Cm >= tight.networks[0].x1Cm + 1.0);

        int expected = 0;
        for (const PlacedNetwork& network : tight.networks) {
            QVERIFY(!network.windings.empty());
            for (const auto& mark : network.windings) {
                QCOMPARE(mark.number, expected);
                ++expected;
                QVERIFY(mark.xCm >= network.x0Cm - 1e-9);
                QVERIFY(mark.xCm <= network.x1Cm + 1e-9);
            }
        }

        // One turn of the panel is one circumference of the reference radius.
        for (const PlacedNetwork& network : tight.networks) {
            for (std::size_t i = 1; i < network.windings.size(); ++i) {
                const double step = network.windings[i].xCm - network.windings[i - 1].xCm;
                QVERIFY(std::abs(step - kTwoPi * network.rRefCm) < 1e-9);
            }
        }

        // The reference radius is the median crossing radius in cm.
        QVERIFY(std::abs(tight.networks[0].rRefCm - 1500.0 * kToCm) < 0.05);
    }

    void interpolatedRunsSplitTheGeometry()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        Weave weave = makeWeave(100, QStringLiteral("a-"), 30000.0, 4000.0, 300.0, 0.0,
                                1.5 * kTwoPi, {100, 900, 1700});
        // Middle span of the H fiber is only an interpolation.
        weave.fibers.front().tracedSegments = {true, false};
        const std::vector<InputFiber> fibers = collect({&weave});
        LayoutParams params = defaultParams();
        const Result result = vc3d::fiber_map::buildLayout(fibers, umbilicus, params);
        QCOMPARE(result.networks.size(), std::size_t{1});

        bool sawHorizontal = false;
        for (const auto& fiber : result.networks.front().fibers) {
            if (fiber.hvTag != 'H') {
                continue;
            }
            sawHorizontal = true;
            QCOMPARE(fiber.runs.size(), std::size_t{2});
            QVERIFY(fiber.runs[0].traced);
            QVERIFY(!fiber.runs[1].traced);
            QCOMPARE(fiber.controlPoints.size(), std::size_t{3});
        }
        QVERIFY(sawHorizontal);
    }
};

QTEST_APPLESS_MAIN(TestFiberNetworkLayout)
#include "test_fiber_network_layout.moc"
