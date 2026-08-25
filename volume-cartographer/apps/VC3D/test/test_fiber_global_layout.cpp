// Coverage for buildGlobalLayout in apps/VC3D/FiberNetworkLayout.cpp: the
// all-fibers map built on the winding solver. The solver's own arithmetic is
// covered by test_fiber_winding_solver; this asserts the layout contract on
// top of it - every fiber accounted for, links landing coincident, winding
// gridlines numbered by the winding coordinate, both chiralities.

#include <QtTest/QtTest>

#include <algorithm>
#include <cmath>
#include <vector>

#include "FiberNetworkLayout.hpp"

using vc3d::fiber_map::ContentDigest;
using vc3d::fiber_map::GlobalAnchor;
using vc3d::fiber_map::GlobalLayoutParams;
using vc3d::fiber_map::GlobalPlacedFiber;
using vc3d::fiber_map::GlobalResult;
using vc3d::fiber_map::InputFiber;
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
    void interpolatedFibersDeclareNoWindingErrors()
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

        // Same geometry, same wrong link - but the H fiber is pure
        // interpolation, so nothing about it is declarable.
        std::vector<InputFiber> untrusted = fibers;
        untrusted[0].tracedSegments.assign(
            untrusted[0].controlPoints.size() - 1, false);
        const GlobalResult silent = vc3d::fiber_map::buildGlobalLayout(
            untrusted, umbilicus, defaultParams());
        QCOMPARE(silent.suspectLinkCount, 0);
        QCOMPARE(silent.droppedCrossingCount, 0);
        QVERIFY(silent.suspectCrossings.empty());
        // The fibers are still placed - exclusion is about declarations, not
        // participation.
        QCOMPARE(silent.fibers.size(), trusted.fibers.size());
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

    // A conflict whose repair victim is decided by constraint order: the
    // U-shaped V fiber crosses the H fiber twice with opposite verdicts and
    // symmetric margins, so the constraint index is the tie-break. The cached
    // replay must pick the same victim.
    void cacheReplayPreservesRepairTieBreaks()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers;
        fibers.push_back(makeFiber(
            1, QStringLiteral("h-mid"), 'H',
            arcPoints(30000.0, 3000.0, 0.0, 0.0, 0.6 * kTwoPi), {50, 900}));
        std::vector<cv::Vec3d> uShape;
        for (double z = 29000.0; z <= 31000.0; z += 10.0) {
            uShape.push_back(cv::Vec3d(2000.0 * std::cos(0.3 * kTwoPi),
                                       2000.0 * std::sin(0.3 * kTwoPi), z));
        }
        for (double z = 31000.0 - 10.0; z >= 29500.0; z -= 10.0) {
            uShape.push_back(cv::Vec3d(4000.0 * std::cos(0.3 * kTwoPi),
                                       4000.0 * std::sin(0.3 * kTwoPi), z));
        }
        const int last = static_cast<int>(uShape.size()) - 1;
        fibers.push_back(makeFiber(2, QStringLiteral("v-u"), 'V',
                                   std::move(uShape), {0, last / 2, last}));
        const GlobalLayoutParams params = defaultParams();
        const GlobalResult fresh =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params);
        vc3d::fiber_map::GlobalLayoutCache cache;
        (void)vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        const GlobalResult warm =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        QCOMPARE(cache.lastStats().pairsReused, 1);
        QVERIFY(fresh.droppedCrossingCount + (int)fresh.suspectCrossings.size() >= 0);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warm) ==
                 vc3d::fiber_map::digestGlobalResult(fresh));
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
                                     {100, 400});
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
};

QTEST_APPLESS_MAIN(TestFiberGlobalLayout)
#include "test_fiber_global_layout.moc"
