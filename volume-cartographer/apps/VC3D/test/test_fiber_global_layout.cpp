// Coverage for buildGlobalLayout in apps/VC3D/FiberNetworkLayout.cpp: the
// all-fibers map built on the winding solver. The solver's own arithmetic is
// covered by test_fiber_winding_solver; this asserts the layout contract on
// top of it - every fiber accounted for, links landing coincident, winding
// gridlines numbered by the winding coordinate, both chiralities.

#include <QtTest/QtTest>

#include <algorithm>
#include <bit>
#include <cmath>
#include <optional>
#include <set>
#include <tuple>
#include <stdexcept>
#include <string>
#include <vector>

#include "FiberNetworkLayout.hpp"
#include "LineAnnotationCoordinateScale.hpp"

using vc3d::fiber_map::ChiralityBasis;
using vc3d::fiber_map::ContentDigest;
using vc3d::fiber_map::GlobalAnchor;
using vc3d::fiber_map::GlobalLayoutParams;
using vc3d::fiber_map::GlobalPlacedFiber;
using vc3d::fiber_map::GlobalResult;
using vc3d::fiber_map::InputFiber;
using vc3d::fiber_map::InputLink;
using vc3d::fiber_map::PlacedLink;

// A verbatim copy of digestGlobalResult as it stood before bent rays (main
// e0bbb8b40), over the fields that existed then: the guard that a build
// without a sheet-normal field still serializes to the same digest stream,
// field for field. (New fields are hashed only when a field is present.)
namespace legacy_digest
{
using vc3d::fiber_map::ContentDigest;
void legacy_hashBytes(ContentDigest& digest, const void* data, std::size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    constexpr uint64_t kPrimeA = 1099511628211ULL;
    constexpr uint64_t kPrimeB = 0x100000001b3ULL ^ 0x9e3779b97f4a7c15ULL;
    uint64_t a = digest.a;
    uint64_t b = digest.b;
    for (std::size_t i = 0; i < size; ++i) {
        a = (a ^ bytes[i]) * kPrimeA;
        b = (b ^ bytes[i]) * (kPrimeB | 1ULL);
    }
    digest.a = a;
    digest.b = b;
}
void legacy_hashU64(ContentDigest& digest, uint64_t value) { legacy_hashBytes(digest, &value, sizeof(value)); }
void legacy_hashDouble(ContentDigest& digest, double value) { legacy_hashBytes(digest, &value, sizeof(value)); }
void legacy_hashString(ContentDigest& digest, const std::string& value)
{
    legacy_hashU64(digest, value.size());
    legacy_hashBytes(digest, value.data(), value.size());
}
ContentDigest legacy_seededDigest(uint64_t seed)
{
    ContentDigest digest{14695981039346656037ULL, 0xcbf29ce484222325ULL};
    legacy_hashU64(digest, seed);
    return digest;
}
ContentDigest digest(const vc3d::fiber_map::GlobalResult& result)
{
    using namespace vc3d::fiber_map;
    ContentDigest digest = legacy_seededDigest(0x0D16);
    legacy_hashDouble(digest, result.rRefVx);
    legacy_hashDouble(digest, result.x0Vx);
    legacy_hashDouble(digest, result.x1Vx);
    legacy_hashDouble(digest, result.yMinVx);
    legacy_hashDouble(digest, result.yMaxVx);
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.chirality)));
    legacy_hashU64(digest, static_cast<uint64_t>(result.chiralityBasis));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.chiralityVote)));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.chiralityNetVotes)));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.comparedChiralityErrors)));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.rejectedChiralityErrors)));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.islandCount)));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.unresolvedCount)));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.tieCount)));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.suspectLinkCount)));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.droppedCrossingCount)));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.declaredGroupCount)));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.traversalGroupCount)));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.unresolvedIntersectionCount)));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.kollesisCrossingCount)));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.kollesisInferredCount)));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.gatedSegmentCount)));
    legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(result.tangentialCount)));
    legacy_hashDouble(digest, result.sheetRadius0Vx);
    legacy_hashDouble(digest, result.sheetPitchVx);
    legacy_hashU64(digest, result.fibers.size());
    for (const GlobalPlacedFiber& fiber : result.fibers) {
        legacy_hashU64(digest, fiber.fiber.id);
        legacy_hashString(digest, fiber.fiber.fileName);
        legacy_hashU64(digest, static_cast<uint64_t>(fiber.fiber.label.size()));
        legacy_hashBytes(digest, fiber.fiber.label.constData(),
                  static_cast<std::size_t>(fiber.fiber.label.size()) * sizeof(QChar));
        legacy_hashU64(digest, static_cast<uint64_t>(fiber.fiber.hvTag));
        legacy_hashU64(digest, static_cast<uint64_t>(fiber.meta.anchor));
        legacy_hashU64(digest, fiber.meta.linked ? 1 : 0);
        legacy_hashU64(digest, fiber.meta.sheetDriftSuspect ? 1 : 0);
        legacy_hashU64(digest, fiber.meta.onKollesis ? 1 : 0);
        legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(fiber.meta.networkId)));
        legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(fiber.meta.networkSize)));
        legacy_hashDouble(digest, fiber.meta.windingLo);
        legacy_hashDouble(digest, fiber.meta.windingHi);
        legacy_hashU64(digest, fiber.fiber.runs.size());
        for (const Run& run : fiber.fiber.runs) {
            legacy_hashU64(digest, run.traced ? 1 : 0);
            legacy_hashU64(digest, run.gap ? 1 : 0);
            legacy_hashU64(digest, run.damaged ? 1 : 0);
            legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(run.firstControl)));
            legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(run.lastControl)));
            legacy_hashU64(digest, run.points.size());
            for (const QPointF& point : run.points) {
                legacy_hashDouble(digest, point.x());
                legacy_hashDouble(digest, point.y());
            }
        }
        legacy_hashU64(digest, fiber.fiber.controlPoints.size());
        for (const QPointF& point : fiber.fiber.controlPoints) {
            legacy_hashDouble(digest, point.x());
            legacy_hashDouble(digest, point.y());
        }
        legacy_hashU64(digest, fiber.fiber.kollesisTerminations.size());
        for (const bool tagged : fiber.fiber.kollesisTerminations) {
            legacy_hashU64(digest, tagged ? 1 : 0);
        }
        legacy_hashU64(digest, fiber.fiber.breaks.size());
        for (const bool tagged : fiber.fiber.breaks) {
            legacy_hashU64(digest, tagged ? 1 : 0);
        }
    }
    legacy_hashU64(digest, result.links.size());
    for (const PlacedLink& link : result.links) {
        legacy_hashU64(digest, link.fiberA);
        legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(link.cpA)));
        legacy_hashU64(digest, link.fiberB);
        legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(link.cpB)));
        legacy_hashDouble(digest, link.a.x());
        legacy_hashDouble(digest, link.a.y());
        legacy_hashDouble(digest, link.b.x());
        legacy_hashDouble(digest, link.b.y());
        legacy_hashDouble(digest, link.turnErr);
        legacy_hashU64(digest, link.suspect ? 1 : 0);
        legacy_hashU64(digest, link.pending ? 1 : 0);
        legacy_hashU64(digest, link.adjacent ? 1 : 0);
        legacy_hashU64(digest, link.adjacentUnpaired ? 1 : 0);
        legacy_hashU64(digest, link.adjacentDisagrees ? 1 : 0);
    }
    legacy_hashU64(digest, result.windings.size());
    for (const WindingMark& mark : result.windings) {
        legacy_hashDouble(digest, mark.xVx);
        legacy_hashU64(digest, static_cast<uint64_t>(static_cast<int64_t>(mark.number)));
    }
    const auto hashI64 = [&digest](long long value) { legacy_hashU64(digest, static_cast<uint64_t>(value)); };
    legacy_hashU64(digest, result.suspectCrossings.size());
    for (const CrossingMark& mark : result.suspectCrossings) {
        legacy_hashDouble(digest, mark.posVx.x());
        legacy_hashDouble(digest, mark.posVx.y());
        legacy_hashU64(digest, mark.hFiberId);
        legacy_hashU64(digest, mark.vFiberId);
        hashI64(mark.n);
        legacy_hashU64(digest, static_cast<uint64_t>(mark.kind));
        legacy_hashDouble(digest, mark.deltaR);
        legacy_hashDouble(digest, mark.violationTurns);
        legacy_hashU64(digest, mark.eventIndex);
        hashI64(mark.groupId);
        legacy_hashU64(digest, mark.kollesis ? 1 : 0);
    }
    legacy_hashU64(digest, result.crossingEvents.size());
    for (const CrossingEvent& event : result.crossingEvents) {
        legacy_hashDouble(digest, event.posVx.x());
        legacy_hashDouble(digest, event.posVx.y());
        legacy_hashU64(digest, event.hFiberId);
        legacy_hashU64(digest, event.vFiberId);
        hashI64(event.n);
        legacy_hashU64(digest, static_cast<uint64_t>(event.kind));
        legacy_hashU64(digest, static_cast<uint64_t>(event.status));
        legacy_hashDouble(digest, event.deltaR);
        legacy_hashDouble(digest, event.transversality);
        legacy_hashU64(digest, (event.tangential ? 1 : 0) | (event.touch ? 2 : 0) |
                            (event.kollesis ? 4 : 0) | (event.kollesisInferred ? 8 : 0));
        hashI64(event.orientation);
        hashI64(event.mergedCount);
        legacy_hashDouble(digest, event.confidence);
        legacy_hashDouble(digest, event.violationTurns);
        hashI64(event.groupId);
    }
    legacy_hashU64(digest, result.crossingGroups.size());
    for (const CrossingGroupRecord& group : result.crossingGroups) {
        legacy_hashU64(digest, group.hFiberId);
        legacy_hashU64(digest, group.vFiberId);
        hashI64(group.n);
        legacy_hashU64(digest, group.vBranch);
        legacy_hashU64(digest, group.members.size());
        for (const std::size_t member : group.members) {
            legacy_hashU64(digest, member);
        }
        hashI64(group.multiplicity);
        hashI64(group.insideCount);
        hashI64(group.orientationSum);
        hashI64(group.insideOrientationSum);
        legacy_hashU64(digest, (group.mixedSigns ? 1 : 0) | (group.coverageGap ? 2 : 0) |
                            (group.unresolved ? 4 : 0) | (group.onCurtain ? 8 : 0) |
                            (group.traversalCovered ? 16 : 0) | (group.hasVerdict ? 32 : 0) |
                            (group.seamed ? 64 : 0));
        legacy_hashDouble(digest, group.minAbsDeltaR);
        legacy_hashDouble(digest, group.meanTransversality);
        legacy_hashU64(digest, static_cast<uint64_t>(group.verdict));
        legacy_hashDouble(digest, group.confidence);
        legacy_hashU64(digest, static_cast<uint64_t>(group.status));
        legacy_hashDouble(digest, group.violationTurns);
    }
    legacy_hashU64(digest, result.unplaced.size());
    for (const UnplacedFiber& fiber : result.unplaced) {
        legacy_hashU64(digest, fiber.id);
        legacy_hashString(digest, fiber.fileName);
        legacy_hashU64(digest, static_cast<uint64_t>(fiber.label.size()));
        legacy_hashBytes(digest, fiber.label.constData(),
                  static_cast<std::size_t>(fiber.label.size()) * sizeof(QChar));
        legacy_hashU64(digest, static_cast<uint64_t>(fiber.hvTag));
    }
    return digest;
}
} // namespace legacy_digest


namespace
{

constexpr double kTwoPi = 2.0 * M_PI;
// Captured on main e0bbb8b40 with this toolchain (gcc 13, RelWithDebInfo,
// -march=x86-64-v3): the field-less digests of cacheFixture().
constexpr uint64_t kPinnedInputsA = 0x86665aa706c04f5fULL;
constexpr uint64_t kPinnedInputsB = 0x16573b03bfc464d3ULL;
constexpr uint64_t kPinnedResultA = 0xb1cc9b14a72438dfULL;
constexpr uint64_t kPinnedResultB = 0x5f2fa1566ea50e3bULL;
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

void addAdjacentLink(InputFiber& a, int controlA, InputFiber& b, int controlB)
{
    a.links.push_back({controlA, b.id, controlB, false, true});
    b.links.push_back({controlB, a.id, controlA, false, true});
}

// An H arc over half a turn at radius `radiusH`, and a V fiber at `angle`
// on it, `inset` inside it (the back of the next wrap in), both with a
// control at the meeting angle: control 1 of each. With `hvTagV` the V
// fiber's tag - 'V' for the real pair, 'H' for a same-kind pair.
std::vector<InputFiber> adjacentPair(double radiusH, double inset, double angle, char hvTagV)
{
    std::vector<cv::Vec3d> arc;
    const double begin = angle - 0.25 * kTwoPi;
    for (int i = 0; i <= 500; ++i) {
        const double theta = begin + 0.5 * kTwoPi * i / 500.0;
        arc.push_back(cv::Vec3d(radiusH * std::cos(theta), radiusH * std::sin(theta), 30000.0));
    }
    std::vector<InputFiber> fibers;
    fibers.push_back(makeFiber(900, QStringLiteral("g-h"), 'H', arc, {0, 250, 500}));
    fibers.push_back(makeFiber(901, QStringLiteral("g-v"), hvTagV,
                               verticalPoints(angle, radiusH - inset, 29000.0, 31000.0, 25.0),
                               {0, 40, 80}));
    return fibers;
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

GlobalLayoutParams sensedParams(int chirality)
{
    GlobalLayoutParams params = defaultParams();
    params.solver.chiralityOverride = chirality;
    return params;
}

// The one-turn weave (its links and crossings agree only in sense +1; six
// V fibers, so the mirror contradicts it by a decisive margin over the one
// net vote against) beside two
// lone H fibers, each an inward spiral over one and a half turns at its own
// height: a fiber that wraps votes on the sense by its radius one turn on,
// so each decoy votes -1 and the three-fiber vote is wrong, 2 to 1.
std::vector<InputFiber> decoyedWeave()
{
    std::vector<InputFiber> fibers =
        makeWeave(100, QStringLiteral("a-"), 30000.0, 4000.0, 300.0,
                  -0.4, kTwoPi + 0.4,
                  {100, 100 + kStepsPerTurn, 130, 130 + kStepsPerTurn, 159,
                   159 + kStepsPerTurn});
    fibers.push_back(makeFiber(
        300, QStringLiteral("d-h-1"), 'H',
        arcPoints(10000.0, 4000.0, -300.0, 0.0, 1.5 * kTwoPi), {0, 1800}));
    fibers.push_back(makeFiber(
        301, QStringLiteral("d-h-2"), 'H',
        arcPoints(50000.0, 4000.0, -300.0, 0.0, 1.5 * kTwoPi), {0, 1800}));
    return fibers;
}

// Independent contradictions of a map: dropped crossings, group conflicts,
// suspect links.
int contradictions(const GlobalResult& result)
{
    return result.droppedCrossingCount + result.declaredGroupCount + result.suspectLinkCount;
}

// The figure the winding-sense comparison uses: the crossing contradictions
// of a solve with the links left out.
int geometryContradictions(const GlobalResult& result)
{
    return result.droppedCrossingCount + result.declaredGroupCount;
}

std::vector<InputFiber> unlinked(std::vector<InputFiber> fibers)
{
    for (InputFiber& fiber : fibers) {
        fiber.links.clear();
    }
    return fibers;
}

// One growing H spiral and five V fibers on its second pass, every one
// linked to the H fiber's first pass: five contradictions in the true sense,
// none in the mirror, one vote.
std::vector<InputFiber> fiveWrongLinksOnOneFiber()
{
    std::vector<InputFiber> fibers =
        makeWeave(100, QStringLiteral("a-"), 30000.0, 4000.0, 300.0, -0.4, kTwoPi + 0.4,
                  {100, 1260, 1300, 1340, 1380, 1415});
    // The first control's V fiber (on the first pass) goes; the rest are
    // relinked from their own crossings to that first control.
    fibers.erase(fibers.begin() + 1);
    InputFiber& h = fibers[0];
    h.links.clear();
    for (std::size_t i = 1; i < fibers.size(); ++i) {
        fibers[i].links.clear();
        addLink(h, 0, fibers[i], 1);
    }
    return fibers;
}

// The one-turn weave's H fiber and its outer V fiber only (the inner one
// would contradict the mirror on its own), the V linked to the H fiber's
// FIRST pass instead of its second: in sense +1 the link contradicts the
// crossing one turn on (one contradiction); mirrored, the wrong link and
// both crossings agree (none). The H fiber's radius grows with theta, so it
// votes +1.
std::vector<InputFiber> wronglyLinkedPair(uint64_t firstId, const QString& prefix, double z)
{
    std::vector<InputFiber> fibers =
        makeWeave(firstId, prefix, z, 4000.0, 300.0, -0.4, kTwoPi + 0.4,
                  {100, 100 + kStepsPerTurn});
    fibers.erase(fibers.begin() + 1);
    InputFiber& h = fibers[0];
    InputFiber& v = fibers[1];
    h.links.clear();
    v.links.clear();
    addLink(h, 0, v, 1);
    return fibers;
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

// --- Sheet normal fields for the bent-ray tests (FiberMapBentRays.hpp).
using vc3d::fiber_map::bent::AnchorPolicy;
using vc3d::fiber_map::CrossingEvent;
using vc3d::fiber_map::CrossingMark;
using vc3d::fiber_map::StretchDecisionRecord;
using vc3d::fiber_map::GlobalLayoutCache;
using vc3d::fiber_map::PairCoverageRecord;
using vc3d::fiber_map::bent::SheetNormalField;
using vc3d::fiber_map::winding::CrossingKind;
using vc3d::fiber_map::winding::CrossingStatus;
using vc3d::fiber_map::winding::PairDetections;

// The sheet normal is radial (a spiral about the z axis) except in a z band
// where the sheet is level (axis +z): a shelf. `silent` gives no value
// anywhere.
class BandField final : public SheetNormalField {
public:
    BandField(double z0, double z1, bool silent, std::string name)
        : z0(z0), z1(z1), silent(silent), name(std::move(name))
    {
    }
    [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d& p) const override
    {
        if (silent) {
            return std::nullopt;
        }
        if (p[2] >= z0 && p[2] <= z1) {
            return cv::Vec3d(0.0, 0.0, 1.0);
        }
        const double r = std::hypot(p[0], p[1]);
        return r > 0.0 ? cv::Vec3d(p[0] / r, p[1] / r, 0.0) : cv::Vec3d(1.0, 0.0, 0.0);
    }
    [[nodiscard]] std::string identity() const override { return "band|" + name; }
    double z0;
    double z1;
    bool silent;
    std::string name;
};

// Radial everywhere except inside a ball, where the sheet is level.
class PocketField final : public SheetNormalField {
public:
    PocketField(cv::Vec3d center, double radius, std::string name)
        : center(center), radius(radius), name(std::move(name))
    {
    }
    [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d& p) const override
    {
        const cv::Vec3d d = p - center;
        if (std::sqrt(d.dot(d)) <= radius) {
            return cv::Vec3d(0.0, 0.0, 1.0);
        }
        const double r = std::hypot(p[0], p[1]);
        return r > 0.0 ? cv::Vec3d(p[0] / r, p[1] / r, 0.0) : cv::Vec3d(1.0, 0.0, 0.0);
    }
    [[nodiscard]] std::string identity() const override { return "pocket|" + name; }
    cv::Vec3d center;
    double radius;
    std::string name;
};

// The sheet normal is one constant axis everywhere.
class ConstantField final : public SheetNormalField {
public:
    ConstantField(cv::Vec3d axis, std::string name) : axis_(axis), name_(std::move(name)) {}
    [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d&) const override { return axis_; }
    [[nodiscard]] std::string identity() const override { return "constant|" + name_; }
    cv::Vec3d axis_;
    std::string name_;
};

// A fiber with its samples (and controls) stored in the other order.
InputFiber reversed(InputFiber fiber)
{
    std::reverse(fiber.linePoints.begin(), fiber.linePoints.end());
    std::reverse(fiber.controlPoints.begin(), fiber.controlPoints.end());
    std::reverse(fiber.tracedSegments.begin(), fiber.tracedSegments.end());
    const int last = static_cast<int>(fiber.controlPoints.size()) - 1;
    for (InputLink& link : fiber.links) {
        link.controlPointIndex = last - link.controlPointIndex;
    }
    return fiber;
}

// Reverse one fiber of a set, re-indexing the links that point at it.
std::vector<InputFiber> withReversed(std::vector<InputFiber> fibers, uint64_t id)
{
    int last = 0;
    for (InputFiber& fiber : fibers) {
        if (fiber.id == id) {
            last = static_cast<int>(fiber.controlPoints.size()) - 1;
            fiber = reversed(std::move(fiber));
        }
    }
    for (InputFiber& fiber : fibers) {
        for (InputLink& link : fiber.links) {
            if (link.branchFiberId == id) {
                link.branchControlPointIndex = last - link.branchControlPointIndex;
            }
        }
    }
    return fibers;
}

// Points along a radial line at one angle and height.
std::vector<cv::Vec3d> radialPoints(double theta, double r0, double r1, double z, double step)
{
    std::vector<cv::Vec3d> points;
    const int count = static_cast<int>(std::floor((r1 - r0) / step)) + 1;
    for (int i = 0; i < count; ++i) {
        const double r = r0 + static_cast<double>(i) * step;
        points.push_back(cv::Vec3d(r * std::cos(theta), r * std::sin(theta), z));
    }
    return points;
}

// The shelf fixture (see shelfIsReadByTheBentRays). `climbing` extends H0
// past the band at both ends so the vote has well-conditioned voters;
// `witnessOffsetZ` places Va that far from H0 along z (the sheet thickness
// behind the H, +z being outward on the shelf); `secondWitness` adds a
// witness on the other side of H0 (a contested stretch).
struct ShelfFixture {
    BandField field;
    std::vector<InputFiber> fibers;
    std::vector<cv::Vec3f> umbilicus = straightUmbilicus(60000);
    GlobalLayoutParams params = sensedParams(1);
    const double radius = 4000.0;
    const double zRun = 29800.0;
    const double pitch = 400.0;
    const double zTop = 31500.0;
    const double thetaA = 0.5 * M_PI;
    const double thetaB = 0.8 * M_PI;
    const uint64_t h0 = 701;
    const uint64_t va = 702;
    const uint64_t vb = 703;
    const uint64_t h1 = 704;
    const uint64_t va2 = 705;
    const uint64_t h2 = 706;
    const uint64_t island = 707;
    // Set before construction through the static factory below.
    bool kollesisTagH0 = false;
    // A seam fixture: H0 tagged at its start, a second tagged H (H2) below
    // the shelf linked to Va on the other side, so Va is on a kollesis and
    // the H0-Va link is a seam anchor, no witness.
    static ShelfFixture withSeam()
    {
        ShelfFixture fixture;
        InputFiber& fh0 = fixture.fibers[0];
        InputFiber& fva = fixture.fibers[1];
        fh0.kollesisTerminations.assign(fh0.controlPoints.size(), false);
        fh0.kollesisTerminations[0] = true;
        std::vector<cv::Vec3d> arc2 =
            arcPoints(fixture.zRun + fixture.pitch - 200.0, fixture.radius, 0.0, 0.2 * M_PI, 1.2 * M_PI);
        const int ia = static_cast<int>(std::llround((fixture.thetaA - 0.2 * M_PI) / kStep));
        const int last2 = static_cast<int>(arc2.size()) - 1;
        InputFiber fh2 = makeFiber(fixture.h2, QStringLiteral("s-h2"), 'H', arc2, {0, ia, last2});
        fh2.kollesisTerminations.assign(fh2.controlPoints.size(), false);
        fh2.kollesisTerminations[2] = true;
        addLink(fh2, 1, fva, 1);
        fixture.fibers.push_back(std::move(fh2));
        return fixture;
    }

    // The witness link adjacent instead of plain: Va one winding INSIDE H0
    // by the link's word, so the sheet's outward side is the one AWAY from
    // Va (w = P_h - P_v = -z): the shelf's orientation turns over.
    static ShelfFixture withAdjacentWitness()
    {
        ShelfFixture fixture;
        fixture.makeWitnessAdjacent();
        return fixture;
    }
    void makeWitnessAdjacent()
    {
        InputFiber& fh0 = fibers[0];
        InputFiber& fva = fibers[1];
        const auto drop = [](InputFiber& fiber, uint64_t other) {
            fiber.links.erase(std::remove_if(fiber.links.begin(), fiber.links.end(),
                                             [&](const InputLink& link) { return link.branchFiberId == other; }),
                              fiber.links.end());
        };
        drop(fh0, va);
        drop(fva, h0);
        addAdjacentLink(fh0, 1, fva, 1);
    }

    // An island V crossing nothing, anchored by radius against the H
    // samples near it when those count: by default above H0 inside the
    // band at 0.3 pi, outside H0's radius (its curtain rises straight from
    // radius 4000 inside the band).
    static ShelfFixture withIsland()
    {
        ShelfFixture fixture;
        fixture.addIsland(0.3 * M_PI, fixture.zRun + fixture.pitch + 300.0);
        return fixture;
    }
    void addIsland(double theta, double z)
    {
        std::vector<cv::Vec3d> run = radialPoints(theta, radius + 300.0, radius + 700.0, z, 25.0);
        const int mid = static_cast<int>(run.size()) / 2;
        fibers.push_back(makeFiber(island, QStringLiteral("s-island"), 'V', run,
                                   {0, mid, static_cast<int>(run.size()) - 1}));
    }

    // H0 comes back over the arc ABOVE the band (out and up from the arc's
    // end, then from 1.2 pi back to 0.2 pi at radius + 500, 100 vx above
    // the band's top): a well-conditioned straight crossing of Vb's climb
    // (radius + 800) reading Inside on the shelf reading's own translate,
    // the opposite of the shelf's bent Outside reading. The rays of the
    // outgoing limb turn radial at the band's top, 100 vx below the limb.
    void addReturnLimbAboveTheBand()
    {
        InputFiber& fh0 = fibers[0];
        const double zShelf = zRun + pitch;
        const double zBack = field.z1 + 100.0;
        const double rBack = radius + 500.0;
        const int steps = 24;
        for (int i = 1; i <= steps; ++i) {
            const double t = static_cast<double>(i) / steps;
            const double r = radius + (rBack - radius) * t;
            const double z = zShelf + (zBack - zShelf) * t;
            fh0.linePoints.emplace_back(r * std::cos(1.2 * M_PI), r * std::sin(1.2 * M_PI), z);
        }
        std::vector<cv::Vec3d> back = arcPoints(zBack, rBack, 0.0, 0.2 * M_PI, 1.2 * M_PI - kStep);
        std::reverse(back.begin(), back.end());
        fh0.linePoints.insert(fh0.linePoints.end(), back.begin(), back.end());
        fh0.controlPoints.back() = fh0.linePoints.back();
    }

    // `passes`: H0 goes round more than once (its line repeats the arc, at
    // the same radius and height for `passRise` 0, climbing by passRise
    // per turn otherwise). `returnLimb`: H0 comes back along the arc at
    // the given height above its outgoing limb, through the outgoing
    // limb's radius (a fold whose returning limb crosses the curtain).
    explicit ShelfFixture(bool climbing = false, double witnessOffsetZ = 10.0,
                          bool secondWitness = false, int passes = 1, double passRise = 0.0,
                          double returnLimbRise = 0.0)
        : field(29500.0, 30700.0, false, "shelf")
    {
        const double zShelf = zRun + pitch;
        // H0: an arc on the shelf from 0.2 pi to 1.2 pi; climbing: the arc
        // leaves the band at both ends (z ramps up outside the shelf).
        std::vector<cv::Vec3d> arc = arcPoints(zShelf, radius, 0.0, 0.2 * M_PI, 1.2 * M_PI);
        const std::size_t firstPassSize = arc.size();
        for (int pass = 1; pass < passes; ++pass) {
            std::vector<cv::Vec3d> again =
                arcPoints(zShelf + passRise * pass, radius, 0.0, 0.2 * M_PI, 1.2 * M_PI);
            // Join through the rest of the circle so the angle unwraps by a
            // whole turn.
            std::vector<cv::Vec3d> join =
                arcPoints(zShelf + passRise * (pass - 0.5), radius, 0.0, 1.2 * M_PI + kStep,
                          2.2 * M_PI - kStep);
            arc.insert(arc.end(), join.begin(), join.end());
            arc.insert(arc.end(), again.begin(), again.end());
        }
        if (returnLimbRise != 0.0) {
            // Back along the arc, through the outgoing limb's radius at
            // thetaB (where Vb reads it): from radius + 240 at 1.2 pi down
            // to radius - 360 at 0.2 pi.
            const int count = static_cast<int>(firstPassSize);
            for (int i = count - 1; i >= 0; --i) {
                const double theta = 0.2 * M_PI + kStep * i;
                const double r = radius + 600.0 * (theta - thetaB) / M_PI;
                arc.emplace_back(r * std::cos(theta), r * std::sin(theta), zShelf + returnLimbRise);
            }
        }
        if (climbing) {
            const std::size_t n = arc.size();
            for (std::size_t i = 0; i < n; ++i) {
                const double edge = 150.0;
                if (i < static_cast<std::size_t>(edge)) {
                    arc[i][2] = field.z1 + 50.0 + (edge - static_cast<double>(i)) * 4.0;
                } else if (i + static_cast<std::size_t>(edge) >= n) {
                    arc[i][2] = field.z1 + 50.0 + (static_cast<double>(i) - (static_cast<double>(n) - edge)) * 4.0;
                }
            }
        }
        const auto indexAt = [](double begin, double theta) {
            return static_cast<int>(std::llround((theta - begin) / kStep));
        };
        const int ia = indexAt(0.2 * M_PI, thetaA);
        const int ib = indexAt(0.2 * M_PI, thetaB);
        const int last = static_cast<int>(arc.size()) - 1;
        InputFiber fh0 = makeFiber(h0, QStringLiteral("s-h0"), 'H', arc, {0, ia, ib, last});
        if (kollesisTagH0) {
            fh0.kollesisTerminations.assign(fh0.controlPoints.size(), false);
            fh0.kollesisTerminations[0] = true;
        }
        // Va: the witness, a short radial run at H0's angle thetaA, a sheet
        // thickness along z from H0, control 1 at H0's radius.
        std::vector<cv::Vec3d> aRun = radialPoints(thetaA, radius - 200.0, radius + 200.0,
                                                   zShelf + witnessOffsetZ, 25.0);
        const int aMid = static_cast<int>(aRun.size()) / 2;
        InputFiber fva = makeFiber(va, QStringLiteral("s-va"), 'V', aRun, {0, aMid, static_cast<int>(aRun.size()) - 1});
        addLink(fh0, 1, fva, 1);
        // Vb: a radial run under H0 at thetaB, one pitch below the shelf,
        // then straight up out of the band to H1's height.
        std::vector<cv::Vec3d> bRun = radialPoints(thetaB, radius - 800.0, radius + 800.0, zRun, 25.0);
        const int bRunLast = static_cast<int>(bRun.size()) - 1;
        std::vector<cv::Vec3d> climb = verticalPoints(thetaB, radius + 800.0, zRun + 25.0, zTop, 25.0);
        bRun.insert(bRun.end(), climb.begin(), climb.end());
        const int bLast = static_cast<int>(bRun.size()) - 1;
        InputFiber fvb = makeFiber(vb, QStringLiteral("s-vb"), 'V', bRun, {0, bRunLast / 2, bRunLast, bLast});
        // H1: a well-conditioned arc above the band, in front of Vb's climb.
        std::vector<cv::Vec3d> arc1 = arcPoints(zTop, radius + 700.0, 0.0, 0.2 * M_PI, 1.2 * M_PI);
        InputFiber fh1 = makeFiber(h1, QStringLiteral("s-h1"), 'H', arc1,
                                   {0, ib, static_cast<int>(arc1.size()) - 1});
        addLink(fh1, 1, fvb, 3);
        fibers.push_back(std::move(fh0));
        fibers.push_back(std::move(fva));
        fibers.push_back(std::move(fvb));
        fibers.push_back(std::move(fh1));
        if (secondWitness) {
            std::vector<cv::Vec3d> a2 = radialPoints(thetaA + 0.1, radius - 200.0, radius + 200.0,
                                                     zShelf - witnessOffsetZ, 25.0);
            const int mid = static_cast<int>(a2.size()) / 2;
            InputFiber fva2 = makeFiber(va2, QStringLiteral("s-va2"), 'V', a2, {0, mid, static_cast<int>(a2.size()) - 1});
            const int ia2 = indexAt(0.2 * M_PI, thetaA + 0.1);
            // H0 needs a control there: rebuild H0 with it.
            InputFiber& hh = fibers.front();
            hh.controlPoints.insert(hh.controlPoints.begin() + 2, hh.linePoints[static_cast<std::size_t>(ia2)]);
            hh.tracedSegments.assign(hh.controlPoints.size() - 1, true);
            for (InputLink& link : hh.links) {
                if (link.controlPointIndex >= 2) {
                    ++link.controlPointIndex;
                }
            }
            for (InputFiber& fiber : fibers) {
                for (InputLink& link : fiber.links) {
                    if (link.branchFiberId == h0 && link.branchControlPointIndex >= 2) {
                        ++link.branchControlPointIndex;
                    }
                }
            }
            addLink(hh, 2, fva2, 1);
            fibers.push_back(std::move(fva2));
        }
    }
};

// The tied-geometry fixture: H0 visits the same arc on the shelf
// twice, one turn apart, joined through an arc out of the band (so the
// passes are two runs seeded from the same positions and their strips tie
// geometrically around the crossing). Vb crosses the arc a quarter strip
// past thetaB, 4 vx above it, twice: `loop` going round the umbilicus once
// between its passages (one winding on: four readings, translates -1, 0,
// 0, +1 at one position), else coming back obliquely half a strip over (a
// hairpin: the same translate at equal ray length on one strip, two
// points, the radial passage the more transversal). Va witnesses +z
// outward at thetaA; H1 above the band is linked to Vb's climb and to H0
// (same winding).
struct TiedOptions {
    // Vb goes round the umbilicus between its passages (else: an oblique
    // hairpin, or with `singlePassage` no second passage at all).
    bool loop = true;
    bool singlePassage = false;
    // H0's second pass runs the arc the other way (a turn on, descending).
    bool reversedSecondPass = false;
    // Height of Vb's passage above the arc: 0 is the seed edge (through
    // H0's own line), 8 the first row of the rays.
    double crossZ = 4.0;
    // Vb's passage at a seed's angle (thetaB) instead of a quarter strip on.
    bool atSeedRay = false;
    // Vb comes straight back at the same angle 2 vx higher (a hairpin whose
    // two passages both lie on the seed's ray).
    bool hairpinOnRay = false;
    // Vb comes back ON the seed's ray (thetaB) 2 vx higher, having crossed
    // inside the strip past it (an interior hit of strip AB followed by a
    // hit on the ray B shares with the next strip).
    bool returnOnSeedRay = false;
    // H0's first sample stored twice: two seeds at one point (the run's
    // first sample is always a seed, and its twin is one too since the
    // seeds are laid from the run's canonical - here its far - end), a
    // zero-width strip beside the first real one; Vb crosses at that
    // sample's angle, on the ray the real strip and the empty one share.
    // H0 is one pass then (a second pass would start at a corner of the
    // drawn curve, whose smoothed resampling is not the same under
    // reversal; its readings' drawn positions would differ by a few vx).
    bool duplicateSample = false;
};

struct TiedFixture {
    BandField field;
    std::vector<InputFiber> fibers;
    std::vector<cv::Vec3f> umbilicus = straightUmbilicus(60000);
    GlobalLayoutParams params = sensedParams(1);
    const double radius = 4000.0;
    const double zShelf = 30200.0;
    const double zTop = 31500.0;
    const double thetaA = 0.5 * M_PI;
    // thetaB is a sample of the arc (a seed): the crossing a quarter strip
    // past it, the hairpin's return three quarters past.
    const int ib = 377;
    const double thetaB = 0.2 * M_PI + static_cast<double>(ib) * kStep;
    const double thetaCross = thetaB + 0.25 * kStep;
    const double thetaBack = thetaB + 0.75 * kStep;
    const uint64_t h0 = 711;
    const uint64_t va = 712;
    const uint64_t vb = 713;
    const uint64_t h1 = 714;

    const double crossAngle;

    explicit TiedFixture(const TiedOptions& options)
        : field(29500.0, 30700.0, false, "tied"),
          crossAngle(options.duplicateSample ? 0.2 * M_PI : (options.atSeedRay ? thetaB : thetaCross))
    {
        const double passEnd = 1.2 * M_PI;
        std::vector<cv::Vec3d> arc = arcPoints(zShelf, radius, 0.0, 0.2 * M_PI, passEnd);
        const int ia = static_cast<int>(std::llround((thetaA - 0.2 * M_PI) / kStep));
        // The second pass: the same points within three samples of thetaB
        // (the tied strips), one voxel out elsewhere, so every control sits
        // on a point of its own pass whichever way the line is stored (the
        // layout maps a control to its first nearest line point).
        // (With the repeated sample the crossing is at the pass's start,
        // outside that window: the whole second pass is one voxel out
        // then - the case needs no tied strips.)
        std::vector<cv::Vec3d> again = arc;
        for (int i = 0; i < static_cast<int>(again.size()); ++i) {
            if (options.duplicateSample || std::abs(i - ib) > 3) {
                const double theta = 0.2 * M_PI + static_cast<double>(i) * kStep;
                again[static_cast<std::size_t>(i)] =
                    cv::Vec3d((radius + 1.0) * std::cos(theta), (radius + 1.0) * std::sin(theta), zShelf);
            }
        }
        if (options.duplicateSample) {
            arc.insert(arc.begin(), arc.front());
        }
        // The join, out of the band: on to the start of the second pass,
        // a turn on - or a whole turn on to its end when it runs the arc
        // the other way.
        // (Above Vb's climb and H1, so the join crosses nothing straight on:
        // a well-conditioned straight reading of H0 x Vb there would
        // outrank the shelf's bent readings by the straight-disagrees rule.)
        std::vector<cv::Vec3d> join =
            options.reversedSecondPass
                ? arcPoints(zTop + 300.0, radius, 0.0, passEnd + kStep, passEnd + kTwoPi - kStep)
                : arcPoints(zTop + 300.0, radius, 0.0, passEnd + kStep, 2.2 * M_PI - kStep);
        if (options.reversedSecondPass) {
            std::reverse(again.begin(), again.end());
        }
        if (!options.duplicateSample) {
            arc.insert(arc.end(), join.begin(), join.end());
            arc.insert(arc.end(), again.begin(), again.end());
        }
        InputFiber fh0 = makeFiber(h0, QStringLiteral("t-h0"), 'H', arc, {0, ia, static_cast<int>(arc.size()) - 1});
        std::vector<cv::Vec3d> aRun = radialPoints(thetaA, radius - 200.0, radius + 200.0, zShelf + 10.0, 25.0);
        const int aMid = static_cast<int>(aRun.size()) / 2;
        InputFiber fva = makeFiber(va, QStringLiteral("t-va"), 'V', aRun, {0, aMid, static_cast<int>(aRun.size()) - 1});
        addLink(fh0, 1, fva, 1);
        // Vb: inward at the crossing angle, round (or back), outward, then
        // up to H1.
        const double zB = zShelf + options.crossZ;
        // The outward passage stops one sample short of the inward one's
        // start, so its controls are its own under reversal too.
        std::vector<cv::Vec3d> inward = radialPoints(crossAngle, radius - 800.0, radius + 800.0, zB, 25.0);
        std::reverse(inward.begin(), inward.end());
        std::vector<cv::Vec3d> bLine = inward;
        std::vector<cv::Vec3d> outward;
        if (options.singlePassage) {
            // Straight up from the inward end: nothing more on the shelf.
        } else if (options.hairpinOnRay) {
            outward = radialPoints(crossAngle, radius - 800.0, radius + 775.0, zB + 2.0, 25.0);
        } else if (options.returnOnSeedRay) {
            outward = radialPoints(thetaB, radius - 800.0, radius + 775.0, zB + 2.0, 25.0);
        } else if (options.loop) {
            std::vector<cv::Vec3d> circle =
                arcPoints(zB, radius - 800.0, 0.0, crossAngle + kStep, crossAngle + kTwoPi - kStep);
            bLine.insert(bLine.end(), circle.begin(), circle.end());
            outward = radialPoints(crossAngle, radius - 800.0, radius + 775.0, zB, 25.0);
        } else {
            // Oblique: the angle drifts half a radian per 4000 vx of radius,
            // through thetaBack at the arc's radius.
            for (double r = radius - 800.0; r <= radius + 775.0 + 1e-9; r += 25.0) {
                const double theta = thetaBack + 0.5 * (r - radius) / radius;
                outward.push_back(cv::Vec3d(r * std::cos(theta), r * std::sin(theta), zB));
            }
        }
        const int turn = static_cast<int>(bLine.size()) - 1;
        bLine.insert(bLine.end(), outward.begin(), outward.end());
        const cv::Vec3d top = bLine.back();
        std::vector<cv::Vec3d> climb =
            verticalPoints(std::atan2(top[1], top[0]), std::hypot(top[0], top[1]), zB + 25.0, zTop, 25.0);
        bLine.insert(bLine.end(), climb.begin(), climb.end());
        const int bLast = static_cast<int>(bLine.size()) - 1;
        InputFiber fvb = makeFiber(vb, QStringLiteral("t-vb"), 'V', bLine, {0, turn, bLast});
        // H1's link control at the climb's angle (the witness vector of Vb's
        // own curtain must be radial enough to count).
        const int ic = static_cast<int>(std::llround((std::atan2(top[1], top[0]) - 0.2 * M_PI) / kStep));
        // H1 inside Vb's climb (a plain link: V outward of H), which starts
        // at the inward end when Vb crosses once.
        const double radiusH1 = options.singlePassage ? radius - 1500.0 : radius + 700.0;
        std::vector<cv::Vec3d> arc1 = arcPoints(zTop, radiusH1, 0.0, 0.2 * M_PI, 1.2 * M_PI);
        std::vector<int> controls1{0, ia, ic, static_cast<int>(arc1.size()) - 1};
        std::sort(controls1.begin(), controls1.end());
        controls1.erase(std::unique(controls1.begin(), controls1.end()), controls1.end());
        const auto controlOf = [&](int index) {
            return static_cast<int>(std::find(controls1.begin(), controls1.end(), index) - controls1.begin());
        };
        InputFiber fh1 = makeFiber(h1, QStringLiteral("t-h1"), 'H', arc1, controls1);
        addLink(fh1, controlOf(ic), fvb, 2);
        addLink(fh0, 1, fh1, controlOf(ia));
        fibers.push_back(std::move(fh0));
        fibers.push_back(std::move(fva));
        fibers.push_back(std::move(fvb));
        fibers.push_back(std::move(fh1));
    }
};

// What a bent reading exports: compared across storage orders.
struct BentExport {
    uint64_t h = 0;
    uint64_t v = 0;
    bool fromV = false;
    int kind = 0;
    long long n = 0;
    bool withheld = false;
    int reason = 0;
    int status = 0;
    double violation = 0.0;
    double x = 0.0;
    double y = 0.0;
    cv::Vec3d hit{0.0, 0.0, 0.0};
    std::size_t eventIndex = 0;
    bool tangential = false;
    bool touch = false;
    double confidence = 0.0;
    double transversality = 0.0;
    int anchor = 0;
    double rayLength = 0.0;
    double turnDeg = 0.0;
    double minConditioning = 1.0;
};
struct MarkExport {
    std::size_t eventIndex = 0;
    double x = 0.0;
    double y = 0.0;
    cv::Vec3d hit{0.0, 0.0, 0.0};
    double violation = 0.0;
};
struct BentSnapshot {
    std::vector<BentExport> events;
    std::vector<MarkExport> marks;
};

BentSnapshot bentSnapshot(const GlobalResult& result)
{
    BentSnapshot out;
    for (std::size_t e = 0; e < result.crossingEvents.size(); ++e) {
        const CrossingEvent& ev = result.crossingEvents[e];
        if (!ev.bent) {
            continue;
        }
        out.events.push_back({ev.hFiberId, ev.vFiberId, ev.curtainFromV, static_cast<int>(ev.kind), ev.n,
                              ev.withheld, ev.bentReason, static_cast<int>(ev.status), ev.violationTurns,
                              ev.posVx.x(), ev.posVx.y(), ev.hitVx, e, ev.tangential, ev.touch, ev.confidence,
                              ev.transversality, ev.anchor, ev.rayLengthVx, ev.turnDeg, ev.minConditioning});
    }
    for (const CrossingMark& mark : result.suspectCrossings) {
        out.marks.push_back({mark.eventIndex, mark.posVx.x(), mark.posVx.y(), mark.hitVx, mark.violationTurns});
    }
    return out;
}

bool sameExport(const BentExport& a, const BentExport& b)
{
    const auto near = [](double p, double q) { return std::abs(p - q) < 1e-6; };
    return a.h == b.h && a.v == b.v && a.fromV == b.fromV && a.kind == b.kind && a.n == b.n &&
           a.withheld == b.withheld && a.reason == b.reason && a.status == b.status &&
           a.violation == b.violation && near(a.x, b.x) && near(a.y, b.y) && near(a.hit[0], b.hit[0]) &&
           near(a.hit[1], b.hit[1]) && near(a.hit[2], b.hit[2]) && a.eventIndex == b.eventIndex &&
           a.tangential == b.tangential && a.touch == b.touch && near(a.confidence, b.confidence) &&
           near(a.transversality, b.transversality) && a.anchor == b.anchor && near(a.rayLength, b.rayLength) &&
           near(a.turnDeg, b.turnDeg) && near(a.minConditioning, b.minConditioning);
}

// The layout under the four storage orders of one H and one V fiber: every
// bent event (its index, constraint, status, violation, winding position
// and hit) and every mark (index, position, hit, violation) must be the
// same. Returns the forward snapshot.
BentSnapshot bentReadingsUnderReversals(const std::vector<InputFiber>& fibers,
                                        const std::vector<cv::Vec3f>& umbilicus,
                                        const GlobalLayoutParams& params, const SheetNormalField& field,
                                        uint64_t hId, uint64_t vId, bool* ok)
{
    *ok = false;
    std::vector<BentSnapshot> snapshots;
    for (const bool reverseH : {false, true}) {
        for (const bool reverseV : {false, true}) {
            std::vector<InputFiber> stored = fibers;
            if (reverseH) {
                stored = withReversed(stored, hId);
            }
            if (reverseV) {
                stored = withReversed(stored, vId);
            }
            snapshots.push_back(bentSnapshot(
                buildGlobalLayout(stored, umbilicus, params, nullptr, &field, std::string())));
        }
    }
    if (qEnvironmentVariableIsSet("VC_TEST_DUMP_BENT")) {
        for (const BentExport& e : snapshots[0].events) {
            qWarning("  forward: h%llu v%llu fromV=%d kind=%d n=%lld withheld=%d reason=%d status=%d viol=%.2f x=%.1f hit=(%.1f,%.1f,%.1f) idx=%zu tang=%d touch=%d",
                     static_cast<unsigned long long>(e.h), static_cast<unsigned long long>(e.v), int(e.fromV), e.kind,
                     e.n, int(e.withheld), e.reason, e.status, e.violation, e.x, e.hit[0], e.hit[1], e.hit[2],
                     e.eventIndex, int(e.tangential), int(e.touch));
        }
    }
    for (std::size_t i = 1; i < snapshots.size(); ++i) {
        if (snapshots[i].events.size() != snapshots[0].events.size() ||
            snapshots[i].marks.size() != snapshots[0].marks.size()) {
            qWarning("storage order %zu: %zu events / %zu marks vs %zu / %zu", i, snapshots[i].events.size(),
                     snapshots[i].marks.size(), snapshots[0].events.size(), snapshots[0].marks.size());
            for (const std::size_t which : {std::size_t{0}, i}) {
                for (const BentExport& e : snapshots[which].events) {
                    qWarning("  order %zu: h%llu v%llu fromV=%d kind=%d n=%lld withheld=%d reason=%d status=%d viol=%.2f x=%.1f hit=(%.1f,%.1f,%.1f) idx=%zu tang=%d touch=%d",
                             which, static_cast<unsigned long long>(e.h), static_cast<unsigned long long>(e.v),
                             int(e.fromV), e.kind, e.n, int(e.withheld), e.reason, e.status, e.violation, e.x,
                             e.hit[0], e.hit[1], e.hit[2], e.eventIndex, int(e.tangential), int(e.touch));
                }
            }
            return snapshots[0];
        }
        for (std::size_t k = 0; k < snapshots[0].events.size(); ++k) {
            if (!sameExport(snapshots[i].events[k], snapshots[0].events[k])) {
                const BentExport& a = snapshots[0].events[k];
                const BentExport& b = snapshots[i].events[k];
                for (const std::size_t which : {std::size_t{0}, i}) {
                    for (const BentExport& e : snapshots[which].events) {
                        qWarning("  order %zu: h%llu v%llu fromV=%d kind=%d n=%lld withheld=%d reason=%d status=%d viol=%.2f x=%.1f hit=(%.1f,%.1f,%.1f) idx=%zu",
                                 which, static_cast<unsigned long long>(e.h), static_cast<unsigned long long>(e.v),
                                 int(e.fromV), e.kind, e.n, int(e.withheld), e.reason, e.status, e.violation, e.x,
                                 e.hit[0], e.hit[1], e.hit[2], e.eventIndex);
                    }
                }
                qWarning("storage order %zu event %zu differs: n %lld/%lld kind %d/%d withheld %d/%d status %d/%d pos (%.3f,%.3f)/(%.3f,%.3f) hit (%.3f,%.3f,%.3f)/(%.3f,%.3f,%.3f) index %zu/%zu",
                         i, k, a.n, b.n, a.kind, b.kind, int(a.withheld), int(b.withheld), a.status, b.status,
                         a.x, a.y, b.x, b.y, a.hit[0], a.hit[1], a.hit[2], b.hit[0], b.hit[1], b.hit[2],
                         a.eventIndex, b.eventIndex);
                return snapshots[0];
            }
        }
        for (std::size_t k = 0; k < snapshots[0].marks.size(); ++k) {
            const MarkExport& a = snapshots[0].marks[k];
            const MarkExport& b = snapshots[i].marks[k];
            if (a.eventIndex != b.eventIndex || std::abs(a.x - b.x) > 1e-6 || std::abs(a.y - b.y) > 1e-6 ||
                std::abs(a.hit[0] - b.hit[0]) > 1e-6 || std::abs(a.hit[1] - b.hit[1]) > 1e-6 ||
                std::abs(a.hit[2] - b.hit[2]) > 1e-6 || a.violation != b.violation) {
                qWarning("storage order %zu mark %zu differs: index %zu/%zu pos (%.3f,%.3f)/(%.3f,%.3f)", i, k,
                         a.eventIndex, b.eventIndex, a.x, a.y, b.x, b.y);
                return snapshots[0];
            }
        }
    }
    *ok = true;
    return snapshots[0];
}

// A field whose axis turns with height: radial far below, level (+z)
// across shelf A, turning on through -e_r to level again across shelf B
// (the axis is unoriented, so B is level like A, but a basis carried
// along a fiber climbing from A to B arrives turned over): the voters
// between the shelves vote the other way round from those below A.
class TwistBandField final : public SheetNormalField {
public:
    TwistBandField(double a0, double a1, double b0, double b1) : a0(a0), a1(a1), b0(b0), b1(b1) {}
    [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d& p) const override
    {
        const double r = std::hypot(p[0], p[1]);
        const cv::Vec3d er = r > 0.0 ? cv::Vec3d(p[0] / r, p[1] / r, 0.0) : cv::Vec3d(1.0, 0.0, 0.0);
        const cv::Vec3d ez(0.0, 0.0, 1.0);
        double phi = 0.0;
        const double z = p[2];
        if (z < a0 - 100.0) {
            phi = 0.0;
        } else if (z < a0) {
            phi = 0.5 * M_PI * (z - (a0 - 100.0)) / 100.0;
        } else if (z <= a1) {
            phi = 0.5 * M_PI;
        } else if (z < b0) {
            phi = 0.5 * M_PI + M_PI * (z - a1) / (b0 - a1);
        } else {
            phi = 1.5 * M_PI;
        }
        return er * std::cos(phi) + ez * std::sin(phi);
    }
    [[nodiscard]] std::string identity() const override { return "twist"; }
    double a0, a1, b0, b1;
};

// Two shelves on one H: H0 climbs out of the radial field below shelf A
// onto A (0.2 pi .. 0.6 pi), up through the twist to shelf B (0.8 pi ..
// 1.2 pi). Its voters below A say +1, those in the twist's upper half -1:
// two sections. Va witnesses H0 on A (a plain link); Vb crosses H0's
// curtain on B, where nothing witnesses.
struct TwoShelfFixture {
    TwistBandField field{30000.0, 30400.0, 31400.0, 31800.0};
    std::vector<InputFiber> fibers;
    std::vector<cv::Vec3f> umbilicus = straightUmbilicus(60000);
    GlobalLayoutParams params = sensedParams(1);
    const double radius = 4000.0;
    const double zA = 30200.0;
    const double zB = 31600.0;
    const double thetaA = 0.4 * M_PI;
    const double thetaB = M_PI;
    const uint64_t h0 = 721;
    const uint64_t va = 722;
    const uint64_t vb = 723;

    TwoShelfFixture()
    {
        std::vector<cv::Vec3d> line;
        // Up from the radial field below A (z 29800) onto A over 40 samples.
        for (int i = 0; i < 40; ++i) {
            const double theta = 0.2 * M_PI - static_cast<double>(40 - i) * kStep;
            const double z = 29800.0 + 400.0 * static_cast<double>(i) / 40.0;
            line.emplace_back(radius * std::cos(theta), radius * std::sin(theta), z);
        }
        const std::vector<cv::Vec3d> arcA = arcPoints(zA, radius, 0.0, 0.2 * M_PI, 0.6 * M_PI);
        line.insert(line.end(), arcA.begin(), arcA.end());
        const int ia = static_cast<int>(line.size()) - static_cast<int>(arcA.size()) +
                       static_cast<int>(std::llround((thetaA - 0.2 * M_PI) / kStep));
        // Through the twist: A's height to B's over 0.6 pi .. 0.8 pi.
        const std::vector<cv::Vec3d> ramp = arcPoints(0.0, radius, 0.0, 0.6 * M_PI + kStep, 0.8 * M_PI - kStep);
        for (std::size_t i = 0; i < ramp.size(); ++i) {
            const double z = zA + (zB - zA) * static_cast<double>(i + 1) / static_cast<double>(ramp.size() + 1);
            line.emplace_back(ramp[i][0], ramp[i][1], z);
        }
        const std::vector<cv::Vec3d> arcB = arcPoints(zB, radius, 0.0, 0.8 * M_PI, 1.2 * M_PI);
        line.insert(line.end(), arcB.begin(), arcB.end());
        InputFiber fh0 = makeFiber(h0, QStringLiteral("t2-h0"), 'H', line, {0, ia, static_cast<int>(line.size()) - 1});
        std::vector<cv::Vec3d> aRun = radialPoints(thetaA, radius - 200.0, radius + 200.0, zA + 10.0, 25.0);
        const int aMid = static_cast<int>(aRun.size()) / 2;
        InputFiber fva = makeFiber(va, QStringLiteral("t2-va"), 'V', aRun, {0, aMid, static_cast<int>(aRun.size()) - 1});
        addLink(fh0, 1, fva, 1);
        std::vector<cv::Vec3d> bRun = radialPoints(thetaB + 0.25 * kStep, radius - 200.0, radius + 200.0, zB + 4.0, 25.0);
        const int bMid = static_cast<int>(bRun.size()) / 2;
        InputFiber fvb = makeFiber(vb, QStringLiteral("t2-vb"), 'V', bRun, {0, bMid, static_cast<int>(bRun.size()) - 1});
        fibers.push_back(std::move(fh0));
        fibers.push_back(std::move(fva));
        fibers.push_back(std::move(fvb));
    }
};

// H0 just outside the umbilicus cutoff (radius 417): a coarse arc (its
// samples 80 vx apart) at radius 418, its chords sagging to 416.1 between
// seeds, and a fine arc at 417.5. Va witnesses at thetaA; Vb crosses at
// the crossing angle 4 vx up.
struct CutoffFixture {
    BandField field{29500.0, 30700.0, false, "cutoff"};
    std::vector<InputFiber> fibers;
    std::vector<cv::Vec3f> umbilicus = straightUmbilicus(60000);
    GlobalLayoutParams params = sensedParams(1);
    const double zShelf = 30200.0;
    const uint64_t h0 = 731;
    const uint64_t va = 732;
    const uint64_t vb = 733;
    double crossAngle = 0.0;

    // `coarse`: the 80 vx arc at 418 (else the fine arc at 417.5); the
    // crossing a quarter step past the sample nearest 0.8 pi.
    explicit CutoffFixture(bool coarse)
    {
        const double radius = coarse ? 418.0 : 417.5;
        const double step = coarse ? 80.0 / radius : kStep;
        std::vector<cv::Vec3d> arc;
        const int count = static_cast<int>(std::floor(M_PI / step)) + 1;
        for (int i = 0; i < count; ++i) {
            const double theta = 0.2 * M_PI + static_cast<double>(i) * step;
            arc.emplace_back(radius * std::cos(theta), radius * std::sin(theta), zShelf);
        }
        const int ia = static_cast<int>(std::llround(0.3 * M_PI / step));
        const int ib = static_cast<int>(std::llround(0.6 * M_PI / step));
        crossAngle = 0.2 * M_PI + (static_cast<double>(ib) + (coarse ? 0.5 : 0.25)) * step;
        InputFiber fh0 = makeFiber(h0, QStringLiteral("c-h0"), 'H', arc, {0, ia, static_cast<int>(arc.size()) - 1});
        const double thetaA = 0.2 * M_PI + static_cast<double>(ia) * step;
        std::vector<cv::Vec3d> aRun = radialPoints(thetaA, radius - 100.0, radius + 100.0, zShelf + 10.0, 10.0);
        const int aMid = static_cast<int>(aRun.size()) / 2;
        InputFiber fva = makeFiber(va, QStringLiteral("c-va"), 'V', aRun, {0, aMid, static_cast<int>(aRun.size()) - 1});
        addLink(fh0, 1, fva, 1);
        std::vector<cv::Vec3d> bRun = radialPoints(crossAngle, 300.0, 700.0, zShelf + 4.0, 10.0);
        const int bMid = static_cast<int>(bRun.size()) / 2;
        InputFiber fvb = makeFiber(vb, QStringLiteral("c-vb"), 'V', bRun, {0, bMid, static_cast<int>(bRun.size()) - 1});
        fibers.push_back(std::move(fh0));
        fibers.push_back(std::move(fva));
        fibers.push_back(std::move(fvb));
        params.solver.minUmbilicusRadiusVx = 417.0;
    }

    // The raw curtain hits of the pair (h0, vb), from a cold cache.
    [[nodiscard]] std::size_t rawHitsOfVb(const std::vector<cv::Vec3f>& centers) const
    {
        GlobalLayoutCache cache;
        (void)buildGlobalLayout(fibers, centers, params, &cache, &field, std::string());
        std::size_t hits = 0;
        for (const PairDetections* shard : cache.cachedDetections()) {
            for (const auto& hit : shard->bentHits) {
                if (!hit.ownerIsV && std::abs(hit.hitZ - (zShelf + 4.0)) < 1e-6) {
                    ++hits;
                }
            }
        }
        return hits;
    }
};

// A crease of the field: level (+z) over the shelf band, then over a
// thin layer the axis turns from +z (tangential to e_r) to +e_r,
// radial above. A ray up from the shelf bends 90 degrees there
// and runs on outward.
class CreaseField final : public SheetNormalField {
public:
    CreaseField(double z0, double z1, double crease) : z0(z0), z1(z1), crease(crease) {}
    [[nodiscard]] std::optional<cv::Vec3d> axis(const cv::Vec3d& p) const override
    {
        const double r = std::hypot(p[0], p[1]);
        const cv::Vec3d er = r > 0.0 ? cv::Vec3d(p[0] / r, p[1] / r, 0.0) : cv::Vec3d(1.0, 0.0, 0.0);
        const cv::Vec3d ez(0.0, 0.0, 1.0);
        if (p[2] < z0) {
            return er;
        }
        if (p[2] <= z1) {
            return ez;
        }
        if (p[2] < z1 + crease) {
            const double t = (p[2] - z1) / crease;
            // From +z (t 0, tangential to e_r) to +e_r (t 1).
            return ez * std::cos(0.5 * M_PI * t) + er * std::sin(0.5 * M_PI * t);
        }
        return er;
    }
    [[nodiscard]] std::string identity() const override { return "crease"; }
    double z0, z1, crease;
};

// H0 on the shelf, witnessed by Va (plain link); Vb a vertical run 200 vx
// outside H0's radius above the crease, where H0's rays arrive after
// bending through it.
struct CreaseFixture {
    CreaseField field{29500.0, 30300.0, 60.0};
    std::vector<InputFiber> fibers;
    std::vector<cv::Vec3f> umbilicus = straightUmbilicus(60000);
    GlobalLayoutParams params = sensedParams(1);
    const double radius = 4000.0;
    const double zShelf = 30200.0;
    const double thetaA = 0.5 * M_PI;
    const double thetaB = 0.8 * M_PI;
    const uint64_t h0 = 741;
    const uint64_t va = 742;
    const uint64_t vb = 743;

    CreaseFixture()
    {
        std::vector<cv::Vec3d> arc = arcPoints(zShelf, radius, 0.0, 0.2 * M_PI, 1.2 * M_PI);
        const int ia = static_cast<int>(std::llround((thetaA - 0.2 * M_PI) / kStep));
        InputFiber fh0 = makeFiber(h0, QStringLiteral("c-h0"), 'H', arc, {0, ia, static_cast<int>(arc.size()) - 1});
        std::vector<cv::Vec3d> aRun = radialPoints(thetaA, radius - 200.0, radius + 200.0, zShelf + 10.0, 25.0);
        const int aMid = static_cast<int>(aRun.size()) / 2;
        InputFiber fva = makeFiber(va, QStringLiteral("c-va"), 'V', aRun, {0, aMid, static_cast<int>(aRun.size()) - 1});
        addLink(fh0, 1, fva, 1);
        // Through the crease the rays turn outward, running on at about
        // z 30360 (328 vx of ray by radius + 200, where a vertical run
        // meets them having turned nearly 90 degrees). The run starts
        // below the shelf so it crosses H0 straight on too (set aside:
        // the shelf is level there).
        std::vector<cv::Vec3d> bRun = verticalPoints(thetaB, radius + 200.0, 30100.0, 30600.0, 10.0);
        const int bMid = static_cast<int>(bRun.size()) / 2;
        InputFiber fvb = makeFiber(vb, QStringLiteral("c-vb"), 'V', bRun, {0, bMid, static_cast<int>(bRun.size()) - 1});
        fibers.push_back(std::move(fh0));
        fibers.push_back(std::move(fva));
        fibers.push_back(std::move(fvb));
    }
};

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
    // An adjacent link asserts W_V = W_H - 1: the V fiber, 150 vx inside the
    // H fiber, lands one winding in, and the link is not suspect. The same
    // geometry with the V fiber's crossing read alone would only say "H is
    // outward of V" (W_H >= W_V + 1), which the link pins to equality.
    void adjacentLinkPlacesTheVerticalOneWindingInside()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers = adjacentPair(4000.0, 150.0, 0.3 * kTwoPi, 'V');
        addAdjacentLink(fibers[0], 1, fibers[1], 1);
        const GlobalResult result =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        const GlobalPlacedFiber* h = findFiber(result, 900);
        const GlobalPlacedFiber* v = findFiber(result, 901);
        QVERIFY(h && v);
        // The H arc spans half a turn and the link sits at its midpoint, so
        // the H fiber's winding AT the link is its start winding + 0.25.
        const double hAtLink = h->meta.windingLo + 0.25;
        QVERIFY2(std::abs((hAtLink - 1.0) - v->meta.windingLo) < 0.05,
                 qPrintable(QStringLiteral("H at link %1 V %2")
                                .arg(hAtLink)
                                .arg(v->meta.windingLo)));
        QCOMPARE(result.links.size(), std::size_t(1));
        QVERIFY(result.links.front().adjacent);
        QVERIFY(!result.links.front().suspect);
        QVERIFY(result.links.front().turnErr < 0.1);
        QCOMPARE(result.suspectLinkCount, 0);
    }

    // An adjacent link is not seam evidence: with the inner H fiber's link
    // to the V fiber made adjacent, the V fiber is linked to a tagged end on
    // one side only and is no longer certified on a kollesis, so the inner
    // encounter reads as the plain Outside crossing it geometrically is.
    void adjacentLinkDoesNotCertifyAKollesis()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers = kollesisSeam(false, 3, 3, false, true);
        int flipped = 0;
        for (InputFiber& fiber : fibers) {
            for (InputLink& link : fiber.links) {
                const bool innerPair = (fiber.id == 800 && link.branchFiberId == 802) ||
                                       (fiber.id == 802 && link.branchFiberId == 800);
                if (innerPair) {
                    link.adjacent = true;
                    ++flipped;
                }
            }
        }
        QCOMPARE(flipped, 2);
        const GlobalResult result =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        const GlobalPlacedFiber* v = findFiber(result, 802);
        QVERIFY(v != nullptr);
        QVERIFY(!v->meta.onKollesis);
        QCOMPARE(result.kollesisCrossingCount, 0);
        for (const auto& event : result.crossingEvents) {
            if (event.vFiberId == 802 && event.hFiberId == 800) {
                QVERIFY(!event.kollesis);
                QCOMPARE(event.kind, vc3d::fiber_map::winding::CrossingKind::Outside);
            }
        }
        // The adjacent link and the Outside crossing agree (W_H = W_V + 1),
        // so nothing is dropped and the link is not suspect.
        QCOMPARE(result.droppedCrossingCount, 0);
        QCOMPARE(result.suspectLinkCount, 0);
    }

    // The two files state different kinds for one pair (true on the H side,
    // an explicit false on the V side): an error, constraining nothing,
    // for the sync merge to arbitrate.
    void adjacentKindDisagreementIsAnError()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers = adjacentPair(4000.0, 150.0, 0.3 * kTwoPi, 'V');
        fibers[0].links.push_back({1, fibers[1].id, 1, false, true, true});
        fibers[1].links.push_back({1, fibers[0].id, 1, false, false, true});
        const GlobalResult result =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QCOMPARE(result.links.size(), std::size_t(1));
        const vc3d::fiber_map::PlacedLink& link = result.links.front();
        QVERIFY(link.adjacentDisagrees);
        QVERIFY(link.suspect);
        QCOMPARE(result.suspectLinkCount, 1);
        const GlobalPlacedFiber* v = findFiber(result, 901);
        QVERIFY(v != nullptr);
        QVERIFY(!v->meta.linked);
        // At the layout API, an unspecified ordinary kind does not disagree
        // with the adjacent ref. Change only this field to exercise its
        // effect on both the solver output and the verification input digest.
        std::vector<InputFiber> implicitKind = fibers;
        implicitKind[1].links.front().adjacentExplicit = false;
        const GlobalResult fine =
            vc3d::fiber_map::buildGlobalLayout(implicitKind, umbilicus, defaultParams());
        QCOMPARE(fine.links.size(), std::size_t(1));
        QVERIFY(!fine.links.front().adjacentDisagrees);
        QVERIFY(fine.links.front().adjacent);
        QVERIFY(!fine.links.front().suspect);
        QVERIFY(!(vc3d::fiber_map::digestGlobalResult(result) ==
                  vc3d::fiber_map::digestGlobalResult(fine)));
        QVERIFY(!(vc3d::fiber_map::digestGlobalInputs(fibers, umbilicus, defaultParams()) ==
                  vc3d::fiber_map::digestGlobalInputs(implicitKind, umbilicus, defaultParams())));
    }

    // A pair that is not one H and one V has no inside: the link is an
    // error - suspect, counted, flagged - and constrains nothing, so the two
    // fibers are NOT tied to the same winding either (which an ordinary
    // link would have done). Both an H-H pair and an untagged one.
    void adjacentLinkBetweenSameKindIsAnErrorAndConstrainsNothing()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        for (const char otherTag : {'H', '?'}) {
            std::vector<InputFiber> fibers = adjacentPair(4000.0, 150.0, 0.3 * kTwoPi, otherTag);
            addAdjacentLink(fibers[0], 1, fibers[1], 1);
            const GlobalResult result =
                vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
            QCOMPARE(result.links.size(), std::size_t(1));
            const vc3d::fiber_map::PlacedLink& link = result.links.front();
            QVERIFY(link.adjacent);
            QVERIFY(link.adjacentUnpaired);
            QVERIFY(link.suspect);
            QCOMPARE(result.suspectLinkCount, 1);
            // No constraint: the same inputs as an ORDINARY link tie the two
            // fibers to one winding; here nothing does, so the second fiber
            // is placed by radial order alone (an island) rather than pinned.
            const GlobalPlacedFiber* other = findFiber(result, 901);
            QVERIFY(other != nullptr);
            QVERIFY(!other->meta.linked);
        }
    }

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

    // The winding sense is settled by which sense the map contradicts less,
    // not by the data's vote: a weave whose links and crossings only agree
    // in one sense, beside two lone H fibers drawn as inward spirals (each
    // votes the other way, so the vote is wrong 2 to 1), lays out in the
    // weave's sense, reporting the vote it overrode and the other sense's
    // error count.
    void unstatedSenseIsSettledByErrorsNotByVote()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(60000);
        const std::vector<InputFiber> fibers = decoyedWeave();
        // The deciding figures: the mirror's crossing contradictions with
        // the links left out, against the true sense's none.
        const GlobalResult mirror = vc3d::fiber_map::buildGlobalLayout(
            unlinked(fibers), umbilicus, sensedParams(-1));
        const int mirrorErrors = geometryContradictions(mirror);
        QVERIFY2(vc3d::fiber_map::chiralityComparisonDecisive(0, mirrorErrors),
                 qPrintable(QString::number(mirrorErrors)));
        const GlobalResult straight = vc3d::fiber_map::buildGlobalLayout(
            unlinked(fibers), umbilicus, sensedParams(1));
        QCOMPARE(geometryContradictions(straight), 0);
        // One net vote against +1 (two decoys to the weave's one).
        QCOMPARE(mirror.chiralityNetVotes, -1);
        const GlobalResult result =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QCOMPARE(result.chiralityVote, -1);
        QCOMPARE(result.chiralityNetVotes, -1);
        QCOMPARE(result.chirality, 1);
        QCOMPARE(result.chiralityBasis, ChiralityBasis::Comparison);
        QCOMPARE(result.suspectCrossings.size(), std::size_t{0});
        QCOMPARE(result.suspectLinkCount, 0);
        QCOMPARE(result.comparedChiralityErrors, 0);
        QCOMPARE(result.rejectedChiralityErrors, mirrorErrors);
        // The kept map is the forced map of its sense, field for field.
        const GlobalResult same = vc3d::fiber_map::buildGlobalLayout(
            fibers, umbilicus, sensedParams(1));
        QCOMPARE(result.fibers.size(), same.fibers.size());
        for (std::size_t i = 0; i < result.fibers.size(); ++i) {
            QCOMPARE(result.fibers[i].fiber.label, same.fibers[i].fiber.label);
            QCOMPARE(result.fibers[i].meta.windingLo, same.fibers[i].meta.windingLo);
            QCOMPARE(result.fibers[i].meta.windingHi, same.fibers[i].meta.windingHi);
        }
        QCOMPARE(result.x0Vx, same.x0Vx);
        QCOMPARE(result.rRefVx, same.rRefVx);
    }

    // A stated sense is taken as given, right or wrong, and the vote is
    // still reported beside it.
    void statedSenseIsTakenAsGiven()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(60000);
        const std::vector<InputFiber> fibers = decoyedWeave();
        const GlobalResult right = vc3d::fiber_map::buildGlobalLayout(
            fibers, umbilicus, sensedParams(1));
        QCOMPARE(right.chirality, 1);
        QCOMPARE(right.chiralityBasis, ChiralityBasis::Override);
        QCOMPARE(right.chiralityVote, -1);
        QCOMPARE(right.rejectedChiralityErrors, -1);
        QCOMPARE(right.suspectCrossings.size(), std::size_t{0});
        const GlobalResult wrong = vc3d::fiber_map::buildGlobalLayout(
            fibers, umbilicus, sensedParams(-1));
        QCOMPARE(wrong.chirality, -1);
        QCOMPARE(wrong.chiralityBasis, ChiralityBasis::Override);
        QCOMPARE(wrong.chiralityVote, -1);
        QVERIFY(!wrong.suspectCrossings.empty());
        // The two senses are different results.
        QVERIFY(!(vc3d::fiber_map::digestGlobalResult(right) ==
                  vc3d::fiber_map::digestGlobalResult(wrong)));
    }

    // Links do not decide the sense: a V fiber linked to the wrong turn of
    // its H fiber is a contradiction in the true sense and none in the
    // mirror, so on links the mirror would win - on one such link, on three
    // (each with its own H fiber), or on five on one H fiber. The geometry
    // of these fixtures orders nothing between turns, so the comparison
    // is a tie in every case and the vote decides; the map built in that
    // sense then reports the bad links as the errors they are. The margin
    // rule itself first.
    void aFewErrorsDoNotDecideTheSense()
    {
        using vc3d::fiber_map::chiralityComparisonDecisive;
        QVERIFY(!chiralityComparisonDecisive(0, 1));
        QVERIFY(!chiralityComparisonDecisive(0, 2));
        QVERIFY(chiralityComparisonDecisive(0, 3));
        QVERIFY(!chiralityComparisonDecisive(3, 6));
        QVERIFY(chiralityComparisonDecisive(3, 7));
        QVERIFY(!chiralityComparisonDecisive(5, 5));
        QVERIFY(!chiralityComparisonDecisive(5, 7));
        // PHerc0139 after the kb-214 edit: a 2-2 vote, 12 against 407.
        QVERIFY(chiralityComparisonDecisive(12, 407));

        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(60000);
        const auto check = [&](const std::vector<InputFiber>& fibers, int badLinks,
                               int expectedVotes) {
            // With the links in, the true sense pays for every bad link
            // and the mirror for none.
            const GlobalResult right =
                vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, sensedParams(1));
            const GlobalResult mirror =
                vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, sensedParams(-1));
            QCOMPARE(right.chiralityNetVotes, expectedVotes);
            QCOMPARE(contradictions(right), badLinks);
            QCOMPARE(contradictions(mirror), 0);
            // With the links out, neither sense contradicts anything.
            QCOMPARE(geometryContradictions(vc3d::fiber_map::buildGlobalLayout(
                         unlinked(fibers), umbilicus, sensedParams(1))),
                     0);
            QCOMPARE(geometryContradictions(vc3d::fiber_map::buildGlobalLayout(
                         unlinked(fibers), umbilicus, sensedParams(-1))),
                     0);
            const GlobalResult result =
                vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
            QCOMPARE(result.chiralityVote, 1);
            QCOMPARE(result.chirality, 1);
            QCOMPARE(result.chiralityBasis, ChiralityBasis::Vote);
            QCOMPARE(result.comparedChiralityErrors, 0);
            QCOMPARE(result.rejectedChiralityErrors, 0);
            QCOMPARE(contradictions(result), badLinks);
        };
        check(wronglyLinkedPair(100, QStringLiteral("a-"), 30000.0), 1, 1);
        {
            std::vector<InputFiber> fibers = wronglyLinkedPair(100, QStringLiteral("a-"), 20000.0);
            for (const auto& [id, prefix, z] :
                 {std::make_tuple(200, QStringLiteral("b-"), 30000.0),
                  std::make_tuple(300, QStringLiteral("c-"), 40000.0)}) {
                const std::vector<InputFiber> more = wronglyLinkedPair(id, prefix, z);
                fibers.insert(fibers.end(), more.begin(), more.end());
            }
            check(fibers, 3, 3);
        }
        check(fiveWrongLinksOnOneFiber(), 5, 1);
    }

    // Both senses tied on errors: the vote decides, and says so.
    void tiedSensesFallToTheVote()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        // One H spiral and nothing to contradict it in either sense.
        std::vector<InputFiber> fibers;
        fibers.push_back(makeFiber(
            100, QStringLiteral("a-h-1"), 'H',
            arcPoints(30000.0, 4000.0, 300.0, 0.0, 1.5 * kTwoPi), {0, 1500}));
        const GlobalResult result =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QCOMPARE(result.chiralityBasis, ChiralityBasis::Vote);
        QCOMPARE(result.chiralityVote, 1);
        QCOMPARE(result.chirality, 1);
        QCOMPARE(result.comparedChiralityErrors, 0);
        QCOMPARE(result.rejectedChiralityErrors, 0);
    }

    // Solving both senses leaves both senses' pair shards in the cache: a
    // later build of either stated sense finds every pair, and the
    // reported pair counts are the kept sense's own.
    void bothSensesAreMemoized()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(60000);
        const std::vector<InputFiber> fibers = decoyedWeave();
        vc3d::fiber_map::GlobalLayoutCache cache;
        const GlobalResult cold =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams(), &cache);
        QCOMPARE(cold.chirality, 1);
        QVERIFY(cache.lastStats().used);
        QCOMPARE(cache.lastStats().fibersReused, 0);
        QCOMPARE(cache.lastStats().fibersRecomputed, static_cast<int>(fibers.size()));
        const int pairs = cache.lastStats().pairsRecomputed;
        QVERIFY(pairs > 0);
        QCOMPARE(cache.lastStats().pairsReused, 0);
        for (const int sense : {1, -1}) {
            const GlobalResult warm = vc3d::fiber_map::buildGlobalLayout(
                fibers, umbilicus, sensedParams(sense), &cache);
            QCOMPARE(warm.chirality, sense);
            QCOMPARE(cache.lastStats().fibersRecomputed, 0);
            QCOMPARE(cache.lastStats().pairsRecomputed, 0);
            QCOMPARE(cache.lastStats().pairsReused, pairs);
        }
        const GlobalResult warm =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams(), &cache);
        QCOMPARE(cache.lastStats().pairsRecomputed, 0);
        QCOMPARE(cache.lastStats().pairsReused, pairs);
        QVERIFY(vc3d::fiber_map::digestGlobalResult(warm) ==
                vc3d::fiber_map::digestGlobalResult(cold));
    }

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
        // The sense is stated: solved in both, the mirrored map absorbs the
        // very conflict this fixture exists to create and would be kept for
        // its fewer errors. The two growing spirals still pin the data's
        // vote at +1 against the regressing fiber's radius drop.
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
        const GlobalLayoutParams params = sensedParams(1);
        const GlobalResult fresh =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params);
        QCOMPARE(fresh.chirality, 1);
        QCOMPARE(fresh.chiralityVote, 1);
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
    // Without a sheet-normal field the result digest is the legacy stream,
    // byte for byte (the test holds a verbatim copy of the pre-bent-ray
    // serialization): a field-less build is bit-identical to the old one.
    void fieldlessDigestIsTheLegacyDigest()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const std::vector<InputFiber> fibers = cacheFixture();
        const GlobalResult base =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QVERIFY(vc3d::fiber_map::digestGlobalResult(base) == legacy_digest::digest(base));
    }

    // The field-less digests of the cache fixture, pinned to the values
    // computed on main e0bbb8b40 before bent rays: a field-less build must
    // still produce them (inputs and result alike).
    void fieldlessDigestsArePinned()
    {
        // Byte-level pins hold only in the environment they were captured
        // in (gcc 13, x86-64-v3, this machine's glibc libm dispatch): the
        // fixture's coordinates come from sin/cos and the digest hashes
        // their bytes. Opt in on the baseline machine; everywhere else the
        // portable fieldlessSemanticsArePinned stands.
        if (qEnvironmentVariableIsEmpty("VC_TEST_DIGEST_GOLDENS")) {
            QSKIP("set VC_TEST_DIGEST_GOLDENS=1 on the baseline machine to check the byte pins");
        }
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        std::vector<InputFiber> fibers = cacheFixture();
        // The controller's snapshot conversion with no selected sheet field:
        // a stamp in a different base must preserve main's stored coordinates.
        for (InputFiber& fiber : fibers) {
            vc3d::line_annotation::scaleFiberMapGeometry(
                fiber.controlPoints, fiber.linePoints,
                vc3d::line_annotation::fiberMapFrameScale(
                    std::array<std::size_t, 3>{32768, 32768, 32768},
                    std::array<int, 3>{65536, 65536, 65536}, 1.0, false), false);
        }
        const GlobalLayoutParams params = defaultParams();
        const ContentDigest inputs = vc3d::fiber_map::digestGlobalInputs(fibers, umbilicus, params);
        const GlobalResult base = vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params);
        const ContentDigest result = vc3d::fiber_map::digestGlobalResult(base);
        QCOMPARE(inputs.a, kPinnedInputsA);
        QCOMPARE(inputs.b, kPinnedInputsB);
        QCOMPARE(result.a, kPinnedResultA);
        QCOMPARE(result.b, kPinnedResultB);
    }

    // Portable pins of the same fixture's semantics: what the field-less
    // build must still conclude on any toolchain.
    void fieldlessSemanticsArePinned()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const std::vector<InputFiber> fibers = cacheFixture();
        const GlobalResult base = vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, defaultParams());
        QCOMPARE(base.islandCount, 0);
        QCOMPARE(base.suspectLinkCount, 0);
        QCOMPARE(base.droppedCrossingCount, 0);
        QCOMPARE(base.traversalGroupCount, 0);
        QCOMPARE(base.declaredGroupCount, 0);
        QCOMPARE(base.crossingEvents.size(), std::size_t{12});
        QCOMPARE(base.fibers.size(), std::size_t{7});
        // Winding ranges to 1e-5, captured on main e0bbb8b40.
        const struct {
            uint64_t id;
            double lo;
            double hi;
        } pins[] = {{100, 1.158439, 2.274682}, {101, 1.159236, 1.159236}, {102, 1.716561, 1.716561},
                    {103, 2.273885, 2.273885}, {200, 0.118631, 0.876592}, {201, 0.119427, 0.119427},
                    {202, 0.875796, 0.875796}};
        for (const auto& pin : pins) {
            const GlobalPlacedFiber* fiber = findFiber(base, pin.id);
            QVERIFY(fiber != nullptr);
            QVERIFY2(std::abs(fiber->meta.windingLo - pin.lo) < 1e-5,
                     qPrintable(QStringLiteral("fiber %1 lo %2").arg(pin.id).arg(fiber->meta.windingLo)));
            QVERIFY2(std::abs(fiber->meta.windingHi - pin.hi) < 1e-5,
                     qPrintable(QStringLiteral("fiber %1 hi %2").arg(pin.id).arg(fiber->meta.windingHi)));
        }
    }

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
            QVERIFY(!tweaked.links.empty());
            tweaked.links[0].adjacent = !tweaked.links[0].adjacent;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = base;
            QVERIFY(!tweaked.links.empty());
            tweaked.links[0].adjacentUnpaired = !tweaked.links[0].adjacentUnpaired;
            QVERIFY(!(vc3d::fiber_map::digestGlobalResult(tweaked) == baseline));
        }
        {
            GlobalResult tweaked = base;
            QVERIFY(!tweaked.links.empty());
            tweaked.links[0].adjacentDisagrees = !tweaked.links[0].adjacentDisagrees;
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

        // The scaling the scene is drawn with agrees with the sheet distance
        // wherever the modelled radius is positive, continues at the map's
        // own scale below that floor, and inverts everywhere.
        const double xFloor = -(model.radius0Vx / model.pitchVx) * circumference;
        for (double x : {-0.5 * circumference, 0.0, 0.3 * circumference,
                         2.7 * circumference}) {
            QVERIFY(std::abs(vc3d::fiber_map::sheetDistanceMonotoneVx(model, x) -
                             vc3d::fiber_map::sheetDistanceVx(model, x)) < 1e-6);
        }
        const double distanceFloor = vc3d::fiber_map::sheetDistanceVx(model, xFloor);
        for (double below : {1.0, 3.0 * circumference}) {
            const double x = xFloor - below;
            const double distance = vc3d::fiber_map::sheetDistanceMonotoneVx(model, x);
            QVERIFY2(std::abs(distance - (distanceFloor - below)) < 1e-6,
                     qPrintable(QString::number(distance)));
            QVERIFY2(std::abs(vc3d::fiber_map::sheetXForDistanceMonotoneVx(model, distance) -
                              x) < 1e-6,
                     qPrintable(QString::number(distance)));
        }
        QCOMPARE(vc3d::fiber_map::sheetDomainFloorXVx(model), xFloor);
        // At the floor and one ulp either side the round trip lands on the
        // floor to rounding (the quadratic is flat there, so no better).
        for (double distance : {std::nextafter(distanceFloor, -1e300), distanceFloor,
                                std::nextafter(distanceFloor, 1e300)}) {
            const double back = vc3d::fiber_map::sheetXForDistanceMonotoneVx(model, distance);
            QVERIFY2(std::abs(back - xFloor) < 1e-6 * std::abs(xFloor),
                     qPrintable(QStringLiteral("%1 -> %2").arg(distance, 0, 'g', 17).arg(back)));
        }
        // Hand-built degenerate models map as the identity, floorless.
        for (const vc3d::fiber_map::SheetModel degenerate :
             {vc3d::fiber_map::SheetModel{0.0, 4000.0, 300.0},
              vc3d::fiber_map::SheetModel{4000.0, 0.0, 300.0},
              vc3d::fiber_map::SheetModel{4000.0, -1.0, 300.0}}) {
            QVERIFY(std::isinf(vc3d::fiber_map::sheetDomainFloorXVx(degenerate)));
            QCOMPARE(vc3d::fiber_map::sheetDistanceMonotoneVx(degenerate, 123.5), 123.5);
            QCOMPARE(vc3d::fiber_map::sheetXForDistanceMonotoneVx(degenerate, 123.5), 123.5);
        }
        double previous = -std::numeric_limits<double>::infinity();
        for (int step = -40; step <= 60; ++step) {
            const double x = xFloor + 0.1 * static_cast<double>(step) * circumference;
            const double distance = vc3d::fiber_map::sheetDistanceMonotoneVx(model, x);
            QVERIFY(distance > previous);
            QVERIFY(std::abs(vc3d::fiber_map::sheetXForDistanceMonotoneVx(model, distance) -
                             x) < 1e-6 * std::max(1.0, std::abs(x)));
            previous = distance;
        }

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
        // So the scene is drawn at the map's own scale, floorless.
        QVERIFY(std::isinf(vc3d::fiber_map::sheetDomainFloorXVx(model)));
        QVERIFY(std::abs(vc3d::fiber_map::sheetDistanceMonotoneVx(model, x) - x) < 1e-9);
        QVERIFY(std::abs(vc3d::fiber_map::sheetXForDistanceMonotoneVx(model, x) - x) < 1e-9);
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

    // --- Break tags: display-only like the kollesis flag, but they also shape
    // the placed runs: the span between two consecutive tagged points is a
    // gap run, bounded exactly at the controls, and a lone tag is only a rim.
    void breakTagsMakeGapRunsWithoutRecomputingGeometry()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const GlobalLayoutParams params = defaultParams();
        std::vector<InputFiber> fibers = cacheFixture();
        const uint64_t taggedId = fibers.front().id;
        const std::size_t controlCount = fibers.front().controlPoints.size();
        QVERIFY(controlCount >= 3);

        vc3d::fiber_map::GlobalLayoutCache cache;
        const GlobalResult untagged =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        const GlobalPlacedFiber* plain = findFiber(untagged, taggedId);
        QVERIFY(plain != nullptr);
        QCOMPARE(plain->fiber.breaks.size(), plain->fiber.controlPoints.size());
        QVERIFY(std::none_of(plain->fiber.runs.begin(), plain->fiber.runs.end(),
                             [](const vc3d::fiber_map::Run& run) { return run.gap; }));
        const std::size_t plainRunCount = plain->fiber.runs.size();

        // A lone break: the flag reaches the placed fiber, no run is a gap,
        // but both session digests move (the rim is drawn from the flag).
        fibers.front().breaks.assign(controlCount, false);
        fibers.front().breaks[1] = true;
        const GlobalResult lone =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        const GlobalPlacedFiber* lonePlaced = findFiber(lone, taggedId);
        QVERIFY(lonePlaced != nullptr);
        QVERIFY(lonePlaced->fiber.breaks[1]);
        QVERIFY(std::none_of(lonePlaced->fiber.runs.begin(), lonePlaced->fiber.runs.end(),
                             [](const vc3d::fiber_map::Run& run) { return run.gap; }));
        QCOMPARE(lonePlaced->fiber.runs.size(), plainRunCount);
        QCOMPARE(cache.lastStats().fibersRecomputed, 0);
        QCOMPARE(cache.lastStats().pairsRecomputed, 0);
        QVERIFY(!(vc3d::fiber_map::digestGlobalInputs(fibers, umbilicus, params) ==
                  vc3d::fiber_map::digestGlobalInputs(cacheFixture(), umbilicus, params)));
        QVERIFY(!(vc3d::fiber_map::digestGlobalResult(lone) ==
                  vc3d::fiber_map::digestGlobalResult(untagged)));

        // A gap span (the span descriptor carries the gap tag; the map reads
        // the span flag, not the pair of rings): exactly one gap run, bounded
        // by controls 1 and 2. The layout's own geometry is unchanged by the
        // tag: the runs are re-partitioned, but the set of drawn/seeded
        // segments is the same, so the gap heat map (which seeds from
        // run.points) sees no difference. Still no geometry recomputation.
        fibers.front().breaks[2] = true;
        fibers.front().gapSegments.assign(controlCount - 1, false);
        fibers.front().gapSegments[1] = true;
        const GlobalResult gapped =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        const GlobalPlacedFiber* placed = findFiber(gapped, taggedId);
        QVERIFY(placed != nullptr);
        QCOMPARE(cache.lastStats().fibersRecomputed, 0);
        std::vector<std::size_t> gapRuns;
        for (std::size_t i = 0; i < placed->fiber.runs.size(); ++i) {
            if (placed->fiber.runs[i].gap) {
                gapRuns.push_back(i);
            }
        }
        QCOMPARE(gapRuns.size(), std::size_t{1});
        const std::size_t gapIndex = gapRuns.front();
        const vc3d::fiber_map::Run& gapRun = placed->fiber.runs[gapIndex];
        QCOMPARE(gapRun.firstControl, 1);
        QCOMPARE(gapRun.lastControl, 2);
        QVERIFY(gapRun.points.size() >= 2);
        const auto segmentsOf = [](const vc3d::fiber_map::PlacedFiber& fiber) {
            std::set<std::tuple<long long, long long, long long, long long>> segments;
            const auto key = [](const QPointF& a, const QPointF& b) {
                return std::make_tuple(std::llround(a.x() * 1000.0), std::llround(a.y() * 1000.0),
                                       std::llround(b.x() * 1000.0), std::llround(b.y() * 1000.0));
            };
            for (const vc3d::fiber_map::Run& run : fiber.runs) {
                for (std::size_t i = 1; i < run.points.size(); ++i) {
                    segments.insert(key(run.points[i - 1], run.points[i]));
                }
            }
            return segments;
        };
        QVERIFY(segmentsOf(placed->fiber) == segmentsOf(plain->fiber));

        // Drawing trims the gap run and its neighbours to the shared controls
        // exactly, while the raw runs keep their one-sample overlap.
        const auto near = [](const QPointF& a, const QPointF& b) {
            return std::hypot(a.x() - b.x(), a.y() - b.y()) < 1e-6;
        };
        const std::vector<QPointF> gapDisplay =
            vc3d::fiber_map::displayRunPoints(placed->fiber, gapIndex);
        QVERIFY(gapDisplay.size() >= 2);
        QVERIFY(near(gapDisplay.front(), placed->fiber.controlPoints[1]));
        QVERIFY(near(gapDisplay.back(), placed->fiber.controlPoints[2]));
        QVERIFY(gapIndex > 0 || gapIndex + 1 < placed->fiber.runs.size());
        if (gapIndex > 0) {
            const std::vector<QPointF> before =
                vc3d::fiber_map::displayRunPoints(placed->fiber, gapIndex - 1);
            QVERIFY(near(before.back(), placed->fiber.controlPoints[1]));
            QVERIFY(!near(placed->fiber.runs[gapIndex - 1].points.back(),
                          placed->fiber.controlPoints[1]));
        }
        if (gapIndex + 1 < placed->fiber.runs.size()) {
            const std::vector<QPointF> after =
                vc3d::fiber_map::displayRunPoints(placed->fiber, gapIndex + 1);
            QVERIFY(near(after.front(), placed->fiber.controlPoints[2]));
            QVERIFY(!near(placed->fiber.runs[gapIndex + 1].points.front(),
                          placed->fiber.controlPoints[2]));
        }
        // A run away from every gap draws its own points unchanged.
        for (std::size_t i = 0; i < placed->fiber.runs.size(); ++i) {
            const bool touchesGap = placed->fiber.runs[i].gap ||
                                    (i > 0 && placed->fiber.runs[i - 1].gap) ||
                                    (i + 1 < placed->fiber.runs.size() &&
                                     placed->fiber.runs[i + 1].gap);
            if (!touchesGap) {
                QVERIFY(vc3d::fiber_map::displayRunPoints(placed->fiber, i) ==
                        placed->fiber.runs[i].points);
            }
        }
        QVERIFY(!(vc3d::fiber_map::digestGlobalResult(gapped) ==
                  vc3d::fiber_map::digestGlobalResult(lone)));

        // A mismatched flag vector is ignored, not read misaligned.
        fibers.front().breaks.pop_back();
        fibers.front().gapSegments.pop_back();
        const GlobalResult mismatched =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        const GlobalPlacedFiber* ignored = findFiber(mismatched, taggedId);
        QVERIFY(ignored != nullptr);
        QVERIFY(std::none_of(ignored->fiber.breaks.begin(), ignored->fiber.breaks.end(),
                             [](bool flagged) { return flagged; }));
        QVERIFY(std::none_of(ignored->fiber.runs.begin(), ignored->fiber.runs.end(),
                             [](const vc3d::fiber_map::Run& run) { return run.gap; }));
    }

    // --- Damaged spans: a third display-only span style, drawn as its own
    // run, never together with a gap, and hashed into the session digests.
    void damagedSpansMakeTheirOwnRuns()
    {
        const std::vector<cv::Vec3f> umbilicus = straightUmbilicus(40000);
        const GlobalLayoutParams params = defaultParams();
        std::vector<InputFiber> fibers = cacheFixture();
        const uint64_t id = fibers.front().id;
        const std::size_t controlCount = fibers.front().controlPoints.size();
        QVERIFY(controlCount >= 3);

        vc3d::fiber_map::GlobalLayoutCache cache;
        const GlobalResult plain =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        fibers.front().damagedSegments.assign(controlCount - 1, false);
        fibers.front().damagedSegments[0] = true;
        const GlobalResult damaged =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        const GlobalPlacedFiber* placed = findFiber(damaged, id);
        QVERIFY(placed != nullptr);
        QCOMPARE(cache.lastStats().fibersRecomputed, 0);
        std::size_t damagedRuns = 0;
        for (const vc3d::fiber_map::Run& run : placed->fiber.runs) {
            if (run.damaged) {
                ++damagedRuns;
                QVERIFY(!run.gap);
                QCOMPARE(run.firstControl, 0);
                QCOMPARE(run.lastControl, 1);
            }
        }
        QCOMPARE(damagedRuns, std::size_t{1});
        // Drawn exactly to its controls, like a gap run, and its neighbour
        // stops at the shared control instead of overlapping into it.
        {
            const auto near = [](const QPointF& a, const QPointF& b) {
                return std::hypot(a.x() - b.x(), a.y() - b.y()) < 1e-6;
            };
            const std::vector<QPointF> shown = vc3d::fiber_map::displayRunPoints(placed->fiber, 0);
            QVERIFY(shown.size() >= 2);
            QVERIFY(near(shown.front(), placed->fiber.controlPoints[0]));
            QVERIFY(near(shown.back(), placed->fiber.controlPoints[1]));
            QVERIFY(placed->fiber.runs.size() >= 2);
            const std::vector<QPointF> after = vc3d::fiber_map::displayRunPoints(placed->fiber, 1);
            QVERIFY(near(after.front(), placed->fiber.controlPoints[1]));
            QVERIFY(!near(placed->fiber.runs[1].points.front(), placed->fiber.controlPoints[1]));
        }
        QVERIFY(!(vc3d::fiber_map::digestGlobalInputs(fibers, umbilicus, params) ==
                  vc3d::fiber_map::digestGlobalInputs(cacheFixture(), umbilicus, params)));
        QVERIFY(!(vc3d::fiber_map::digestGlobalResult(damaged) ==
                  vc3d::fiber_map::digestGlobalResult(plain)));

        // The gap wins where both flags are set on the same span.
        fibers.front().gapSegments.assign(controlCount - 1, false);
        fibers.front().gapSegments[0] = true;
        const GlobalResult both =
            vc3d::fiber_map::buildGlobalLayout(fibers, umbilicus, params, &cache);
        const GlobalPlacedFiber* bothPlaced = findFiber(both, id);
        QVERIFY(bothPlaced != nullptr);
        for (const vc3d::fiber_map::Run& run : bothPlaced->fiber.runs) {
            QVERIFY(!run.damaged);
        }
        QVERIFY(std::any_of(bothPlaced->fiber.runs.begin(), bothPlaced->fiber.runs.end(),
                            [](const vc3d::fiber_map::Run& run) { return run.gap; }));
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
        // The nudged V fiber has a shard per H fiber in each winding sense.
        QVERIFY(differing <= 2 * hFibers);
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

    // --- The sheet normal field: bent rays where the sheet runs along the
    // umbilicus ray (FiberMapBentRays.hpp), through the whole layout.

    // The shelf fixture: H0 lies level on a shelf (the field's axis is +z in
    // a z band, radial elsewhere), one winding above Vb's radial run, with
    // Va a sheet thickness behind it as the link witness; Vb climbs out of
    // the band to H1, which it is plain-linked to. The only evidence tying
    // H0 to the rest is the bent reading of H0's curtain: Vb on its inward
    // side, a strict Outside, so the map must put H0 exactly one winding
    // out of H1.
    void shelfIsReadByTheBentRays()
    {
        const ShelfFixture fixture;
        GlobalLayoutCache cache;
        const GlobalResult result = buildGlobalLayout(fixture.fibers, fixture.umbilicus,
                                                      fixture.params, &cache, &fixture.field,
                                                      std::string());
        QCOMPARE(result.effectiveField, fixture.field.identity());
        // Two readings of H0's curtain: Vb on its inward side (Outside) and
        // the witness Va a thickness behind it (Inside, weak); Vb's own
        // curtain read the same and was dropped for H0's.
        QCOMPARE(result.bentCrossingCount, 2);
        QCOMPARE(result.withheldCount, 0);
        QCOMPARE(result.setAsideCount, 1);
        QCOMPARE(result.radialInvertedCount, 0);
        QCOMPARE(result.linkAnchoredCount, 2);
        QCOMPARE(result.umbilicusAnchoredCount, 0);
        QCOMPARE(result.anchorCorrectedStretchCount, 0);
        QCOMPARE(result.droppedCrossingCount, 0);
        QVERIFY(result.suspectCrossings.empty());
        QCOMPARE(result.suspectLinkCount, 0);
        const CrossingEvent* bent = nullptr;
        const CrossingEvent* witnessReading = nullptr;
        for (const CrossingEvent& event : result.crossingEvents) {
            if (!event.bent || event.withheld || event.tangential) {
                continue;
            }
            if (event.vFiberId == fixture.vb) {
                QVERIFY(bent == nullptr);
                bent = &event;
            } else {
                witnessReading = &event;
            }
        }
        QVERIFY(bent != nullptr);
        QVERIFY(witnessReading != nullptr);
        QCOMPARE(witnessReading->vFiberId, fixture.va);
        QCOMPARE(witnessReading->kind, CrossingKind::Inside);
        QCOMPARE(witnessReading->rayLengthVx, 10.0);
        QCOMPARE(bent->hFiberId, fixture.h0);
        QCOMPARE(bent->vFiberId, fixture.vb);
        QVERIFY(!bent->curtainFromV);
        QCOMPARE(bent->kind, CrossingKind::Outside);
        QCOMPARE(bent->anchor, 2);
        QCOMPARE(bent->status, CrossingStatus::Used);
        QVERIFY(std::abs(bent->hitVx[2] - fixture.zRun) < 1e-6);
        QVERIFY(std::abs(std::hypot(bent->hitVx[0], bent->hitVx[1]) - fixture.radius) < 1.0);
        QVERIFY(bent->rayLengthVx > 0.0);
        QVERIFY(bent->confidence > 0.99);
        const GlobalPlacedFiber* h0 = findFiber(result, fixture.h0);
        const GlobalPlacedFiber* h1 = findFiber(result, fixture.h1);
        QVERIFY(h0 != nullptr && h1 != nullptr);
        QVERIFY(std::abs((h0->meta.windingLo - h1->meta.windingLo) - 1.0) < 1e-9);
        QCOMPARE(result.pairCoverage.size(), std::size_t{1});
        QCOMPARE(result.pairCoverage.front().hFiberId, fixture.h0);
        QCOMPARE(result.pairCoverage.front().vFiberId, fixture.vb);
        QCOMPARE(result.pairCoverage.front().reason, std::string("replaced"));
        QCOMPARE(result.pairCoverage.front().setAsideCount, 1);
        // The field-less build of the same fibers leaves H0 unplaced
        // relative to H1 (its straight readings are not even wrong: the
        // weak Inside of the set-aside crossing says the opposite).
        const GlobalResult plain = buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params);
        QVERIFY(plain.effectiveField.empty());
        QCOMPARE(plain.bentCrossingCount, 0);
        const GlobalPlacedFiber* plainH0 = findFiber(plain, fixture.h0);
        const GlobalPlacedFiber* plainH1 = findFiber(plain, fixture.h1);
        QVERIFY(plainH0 != nullptr && plainH1 != nullptr);
        QVERIFY(std::abs((plainH0->meta.windingLo - plainH1->meta.windingLo) - 1.0) > 0.5);
    }

    // A field with no value anywhere changes no semantic field of the
    // legacy map; only the provenance records it, and the input digest
    // moves with it.
    void silentFieldLeavesTheLegacyMap()
    {
        const std::vector<InputFiber> fibers = cacheFixture();
        const GlobalLayoutParams params = defaultParams();
        const BandField silent(0.0, 0.0, true, "silent");
        const GlobalResult plain = buildGlobalLayout(fibers, straightUmbilicus(60000), params);
        const GlobalResult withField = buildGlobalLayout(
            fibers, straightUmbilicus(60000), params, nullptr, &silent, std::string());
        QCOMPARE(withField.effectiveField, std::string("band|silent"));
        QCOMPARE(withField.bentCrossingCount, 0);
        QCOMPARE(withField.setAsideCount, 0);
        QVERIFY(withField.pairCoverage.empty());
        GlobalResult stripped = withField;
        stripped.effectiveField.clear();
        stripped.fieldMs = plain.fieldMs;
        stripped.fieldSampleCount = plain.fieldSampleCount;
        QVERIFY(digestGlobalResult(stripped) == digestGlobalResult(plain));
        QVERIFY(!(digestGlobalInputs(fibers, straightUmbilicus(60000), params, "band|silent") ==
                  digestGlobalInputs(fibers, straightUmbilicus(60000), params)));
        // The unavailable statement is provenance too.
        QVERIFY(!(digestGlobalInputs(fibers, straightUmbilicus(60000), params, "unavailable|x") ==
                  digestGlobalInputs(fibers, straightUmbilicus(60000), params, "band|silent")));
    }

    // With a field the cached build is the fresh one in every semantic
    // field, cold, warm and uncached alike, and the warm build reuses
    // every pair.
    void cachedFieldBuildMatchesFresh()
    {
        const ShelfFixture fixture;
        GlobalLayoutCache cache;
        const GlobalResult cold = buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params,
                                                    &cache, &fixture.field, std::string());
        const auto coldStats = cache.lastStats();
        const GlobalResult warm = buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params,
                                                    &cache, &fixture.field, std::string());
        const auto warmStats = cache.lastStats();
        const GlobalResult uncached = buildGlobalLayout(fixture.fibers, fixture.umbilicus,
                                                        fixture.params, nullptr, &fixture.field,
                                                        std::string());
        QVERIFY(digestGlobalResult(cold) == digestGlobalResult(warm));
        QVERIFY(digestGlobalResult(cold) == digestGlobalResult(uncached));
        QVERIFY(coldStats.pairsRecomputed > 0);
        QCOMPARE(warmStats.pairsRecomputed, 0);
        QCOMPARE(warmStats.pairsReused, coldStats.pairsRecomputed);
        QCOMPARE(warmStats.fibersRecomputed, 0);
        // The cached shards carry the bent hits and the set-asides.
        int hits = 0;
        int setAside = 0;
        for (const PairDetections* shard : cache.cachedDetections()) {
            hits += static_cast<int>(shard->bentHits.size());
            setAside += shard->setAsideCount;
        }
        QVERIFY(hits >= 2);
        QCOMPARE(setAside, 1);
    }

    // What invalidates what: a field of another identity, a tracing
    // parameter, the umbilicus cutoff each recompute the field
    // preparations and every pair; a link edit recomputes nothing and is
    // read at assembly (the witness gone, the readings are withheld).
    void fieldChangesInvalidateAndLinksDoNot()
    {
        const ShelfFixture fixture;
        GlobalLayoutCache cache;
        // The baseline, warm: a second build recomputes nothing.
        const auto rewarm = [&]() {
            (void)buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params, &cache,
                                    &fixture.field, std::string());
            (void)buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params, &cache,
                                    &fixture.field, std::string());
            QCOMPARE(cache.lastStats().pairsRecomputed, 0);
        };
        (void)buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params, &cache,
                                &fixture.field, std::string());
        const int pairs = cache.lastStats().pairsRecomputed;
        QVERIFY(pairs > 0);
        // Everything but the timings and the sample count.
        const auto comparable = [](GlobalResult result) {
            result.prepMs = 0.0;
            result.detectMs = 0.0;
            result.solveMs = 0.0;
            result.geometryMs = 0.0;
            result.fieldMs = 0.0;
            result.fieldSampleCount = 0;
            return result;
        };
        // One mutation at a time from the warm baseline: every pair is
        // recomputed, and the cached build is the fresh build of the
        // mutated input (not the baseline's shards under a new key).
        {
            rewarm();
            // Another field: another band (its readings differ), another name.
            const BandField other(fixture.field.z0 - 600.0, fixture.field.z1 - 600.0, false, "other");
            const GlobalResult cached = buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params,
                                                          &cache, &other, std::string());
            QCOMPARE(cache.lastStats().pairsRecomputed, pairs);
            QCOMPARE(cache.lastStats().pairsReused, 0);
            const GlobalResult fresh = buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params,
                                                         nullptr, &other, std::string());
            QVERIFY(digestGlobalResult(comparable(cached)) == digestGlobalResult(comparable(fresh)));
            const GlobalResult baseline = buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params,
                                                            nullptr, &fixture.field, std::string());
            QVERIFY(!(digestGlobalResult(comparable(cached)) == digestGlobalResult(comparable(baseline))));
        }
        {
            rewarm();
            GlobalLayoutParams params = fixture.params;
            params.bentRays.stepVx *= 0.5;
            const GlobalResult cached = buildGlobalLayout(fixture.fibers, fixture.umbilicus, params, &cache,
                                                          &fixture.field, std::string());
            QCOMPARE(cache.lastStats().pairsRecomputed, pairs);
            const GlobalResult fresh = buildGlobalLayout(fixture.fibers, fixture.umbilicus, params, nullptr,
                                                         &fixture.field, std::string());
            QVERIFY(digestGlobalResult(comparable(cached)) == digestGlobalResult(comparable(fresh)));
        }
        {
            rewarm();
            GlobalLayoutParams params = fixture.params;
            params.bentRays.spacingVx *= 0.5;
            (void)buildGlobalLayout(fixture.fibers, fixture.umbilicus, params, &cache, &fixture.field,
                                    std::string());
            QCOMPARE(cache.lastStats().pairsRecomputed, pairs);
        }
        {
            rewarm();
            GlobalLayoutParams params = fixture.params;
            params.bentRays.conditioningGate *= 0.5;
            (void)buildGlobalLayout(fixture.fibers, fixture.umbilicus, params, &cache, &fixture.field,
                                    std::string());
            QCOMPARE(cache.lastStats().pairsRecomputed, pairs);
        }
        {
            rewarm();
            GlobalLayoutParams params = fixture.params;
            params.solver.minUmbilicusRadiusVx += 1.0;
            const GlobalResult cached = buildGlobalLayout(fixture.fibers, fixture.umbilicus, params, &cache,
                                                          &fixture.field, std::string());
            QCOMPARE(cache.lastStats().pairsRecomputed, pairs);
            const GlobalResult fresh = buildGlobalLayout(fixture.fibers, fixture.umbilicus, params, nullptr,
                                                         &fixture.field, std::string());
            QVERIFY(digestGlobalResult(comparable(cached)) == digestGlobalResult(comparable(fresh)));
        }
        // Back to the original: warm again.
        rewarm();
        {
            // Every link removed: nothing recomputed, every reading
            // withheld for want of a witness, the pair's coverage saying so.
            std::vector<InputFiber> unwitnessed = fixture.fibers;
            for (InputFiber& fiber : unwitnessed) {
                fiber.links.clear();
            }
            const GlobalResult result = buildGlobalLayout(unwitnessed, fixture.umbilicus,
                                                          fixture.params, &cache, &fixture.field,
                                                          std::string());
            QCOMPARE(cache.lastStats().pairsRecomputed, 0);
            QCOMPARE(result.bentCrossingCount, 0);
            QVERIFY(result.withheldCount >= 1);
            QCOMPARE(result.pairCoverage.size(), std::size_t{1});
            QCOMPARE(result.pairCoverage.front().reason, std::string("noWitness"));
            bool withheldEvent = false;
            for (const CrossingEvent& event : result.crossingEvents) {
                if (event.bent && event.withheld) {
                    withheldEvent = true;
                    QCOMPARE(event.bentReason,
                             static_cast<int>(vc3d::fiber_map::winding::BentDisposition::NoWitness));
                    QCOMPARE(event.anchor, 0);
                }
            }
            QVERIFY(withheldEvent);
            // And the same without a cache.
            const GlobalResult fresh = buildGlobalLayout(unwitnessed, fixture.umbilicus, fixture.params,
                                                         nullptr, &fixture.field, std::string());
            QVERIFY(digestGlobalResult(comparable(result)) == digestGlobalResult(comparable(fresh)));
        }
    }

    // A level shelf and a smooth ramp, with no fold between H and V.
    // H's link anchors +z, but V's remote radial voters orient its shelf
    // toward -z. Those independent signs cannot establish a fold.
    void unwitnessedRampCannotVetoLinkedShelf()
    {
        class RampField final : public SheetNormalField {
        public:
            std::optional<cv::Vec3d> axis(const cv::Vec3d& p) const override
            {
                const double c = std::clamp(1.0 - (p[2] - 10.0) / 40.0, 0.0, 1.0);
                return cv::Vec3d(-std::sqrt(1.0 - c * c), 0.0, c);
            }
            std::string identity() const override { return "shelf-and-ramp"; }
        } field;
        std::vector<cv::Vec3d> h, v, witness;
        for (int y = -20; y <= 20; y += 2) {
            h.emplace_back(1000.0, y, 0.0);
        }
        for (int x = 990; x <= 1020; x += 2) {
            v.emplace_back(x, 0.0, 10.0);
        }
        for (int i = 1; i <= 30; ++i) {
            const double t = 0.5 * M_PI * i / 30.0;
            v.emplace_back(1020.0 + 40.0 * std::sin(t), 0.0, 10.0 + 40.0 * (1.0 - std::cos(t)));
        }
        for (int z = 52; z <= 90; z += 2) {
            v.emplace_back(1060.0, 0.0, z);
        }
        for (int x = 990; x <= 1010; x += 2) {
            witness.emplace_back(x, -20.0, 2.0);
        }
        std::vector<InputFiber> fibers{
            makeFiber(991, QStringLiteral("shelf-h"), 'H', h, {0, 10, 20}),
            makeFiber(992, QStringLiteral("ramp-v"), 'V', v, {0, 5, static_cast<int>(v.size()) - 1}),
            makeFiber(993, QStringLiteral("witness-v"), 'V', witness, {0, 5, 10})};
        addLink(fibers[0], 0, fibers[2], 1);
        GlobalLayoutParams params = sensedParams(1);
        params.resampleStepVx = 2.0;
        params.bentRays.stepVx = 1.0;
        params.bentRays.spacingVx = 4.0;
        params.bentRays.maxLengthVx = 24.0;
        const auto umbilicus = straightUmbilicus(200);
        using vc3d::fiber_map::winding::BentDisposition;
        const auto comparable = [](GlobalResult result) {
            result.fieldSampleCount = 0;
            return result;
        };
        std::optional<BentSnapshot> baseline;
        for (int order = 0; order < 4; ++order) {
            auto input = fibers;
            if (order & 1) {
                input = withReversed(std::move(input), 991);
            }
            if (order & 2) {
                input = withReversed(std::move(input), 992);
            }
            GlobalLayoutCache cache;
            const GlobalResult fresh = buildGlobalLayout(input, umbilicus, params, &cache, &field, {});
            const GlobalResult warm = buildGlobalLayout(input, umbilicus, params, &cache, &field, {});
            const GlobalResult uncached = buildGlobalLayout(input, umbilicus, params, nullptr, &field, {});
            QVERIFY(digestGlobalResult(comparable(fresh)) == digestGlobalResult(comparable(warm)));
            QVERIFY(digestGlobalResult(comparable(fresh)) == digestGlobalResult(comparable(uncached)));
            // Crease telemetry is deliberately absent from result digests;
            // compare its bits separately for the cache oracle.
            for (const GlobalResult* other : {&warm, &uncached}) {
                QCOMPARE(other->crossingEvents.size(), fresh.crossingEvents.size());
                for (std::size_t i = 0; i < fresh.crossingEvents.size(); ++i) {
                    QCOMPARE(std::bit_cast<uint64_t>(other->crossingEvents[i].turnDeg),
                             std::bit_cast<uint64_t>(fresh.crossingEvents[i].turnDeg));
                    QCOMPARE(std::bit_cast<uint64_t>(other->crossingEvents[i].minConditioning),
                             std::bit_cast<uint64_t>(fresh.crossingEvents[i].minConditioning));
                }
            }
            bool voted = false;
            for (const StretchDecisionRecord& decision : fresh.stretchDecisions) {
                if (decision.fiberId == 992 && decision.voteStatus == 0) {
                    voted = true;
                    QCOMPARE(decision.anchor, 0);
                    QCOMPARE(decision.reason, static_cast<int>(BentDisposition::NoWitness));
                }
            }
            QVERIFY(voted);
            const BentSnapshot exported = bentSnapshot(fresh);
            if (baseline) {
                QCOMPARE(exported.events.size(), baseline->events.size());
                for (std::size_t i = 0; i < exported.events.size(); ++i) {
                    QVERIFY(sameExport(exported.events[i], baseline->events[i]));
                }
            } else {
                baseline = exported;
            }
            bool linked = false, provisional = false;
            for (const auto& e : fresh.crossingEvents) {
                if (!e.bent || e.hFiberId != 991 || e.vFiberId != 992 || e.tangential) {
                    continue;
                }
                QCOMPARE(e.n, 0LL);
                QCOMPARE(e.turnDeg, 0.0);
                QCOMPARE(e.transversality, 1.0);
                if (e.curtainFromV) {
                    provisional = true;
                    QCOMPARE(e.kind, CrossingKind::Outside);
                    QCOMPARE(e.bentReason, static_cast<int>(BentDisposition::NoWitness));
                } else {
                    linked = true;
                    QCOMPARE(e.kind, CrossingKind::Inside);
                    QCOMPARE(e.anchor, 2);
                    QVERIFY(!e.withheld);
                }
            }
            QVERIFY(linked && provisional);
        }
    }

    // The anchor policy: without a witness LinksOnly withholds; with
    // UmbilicusOrLinks a voteless shelf stays withheld (unsupportedRun) and a
    // shelf whose stretch has well-conditioned voters constrains on the
    // umbilicus anchor.
    void anchorPolicyDecidesUnwitnessedStretches()
    {
        ShelfFixture fixture;
        std::vector<InputFiber> unwitnessed = fixture.fibers;
        for (InputFiber& fiber : unwitnessed) {
            fiber.links.clear();
        }
        {
            GlobalLayoutParams params = fixture.params;
            params.bentRays.anchorPolicy = AnchorPolicy::UmbilicusOrLinks;
            const GlobalResult result = buildGlobalLayout(unwitnessed, fixture.umbilicus, params,
                                                          nullptr, &fixture.field, std::string());
            // H0 lies wholly on the shelf: no voter, so no umbilicus anchor
            // and its readings are withheld as unsupported; Vb climbs out of
            // the band, its voters anchor it, and its own curtain's reading
            // of H0 stands in on the umbilicus anchor.
            QCOMPARE(result.bentCrossingCount, 1);
            QCOMPARE(result.umbilicusAnchoredCount, 1);
            QVERIFY(result.withheldCount >= 2);
            QVERIFY(result.unsupportedRunCount >= 1);
            bool unsupported = false;
            for (const CrossingEvent& event : result.crossingEvents) {
                if (event.bent && event.withheld &&
                    event.bentReason ==
                        static_cast<int>(vc3d::fiber_map::winding::BentDisposition::Unsupported)) {
                    unsupported = true;
                }
            }
            QVERIFY(unsupported);
            QCOMPARE(result.pairCoverage.front().reason, std::string("replaced"));
            // Under the default nothing constrains and the coverage says why.
            const GlobalResult strict = buildGlobalLayout(unwitnessed, fixture.umbilicus,
                                                          fixture.params, nullptr, &fixture.field,
                                                          std::string());
            QCOMPARE(strict.bentCrossingCount, 0);
            QCOMPARE(strict.pairCoverage.front().reason, std::string("noWitness"));
        }
        {
            // H0 extended beyond the band at both ends, where the field is
            // radial and the samples vote for the provisional +z... no: there
            // the axis is radial, agreeing with +e_r, so the transported
            // basis is anchored outward by the vote and the shelf run
            // inherits it.
            const ShelfFixture climbing(true);
            std::vector<InputFiber> fibers = climbing.fibers;
            for (InputFiber& fiber : fibers) {
                if (fiber.id == climbing.h0 || fiber.id == climbing.va) {
                    fiber.links.clear();
                }
            }
            GlobalLayoutParams params = climbing.params;
            params.bentRays.anchorPolicy = AnchorPolicy::UmbilicusOrLinks;
            const GlobalResult result = buildGlobalLayout(fibers, climbing.umbilicus, params, nullptr,
                                                          &climbing.field, std::string());
            // H0's curtain reads Vb and Va on the umbilicus anchor.
            QCOMPARE(result.bentCrossingCount, 2);
            QCOMPARE(result.umbilicusAnchoredCount, 2);
            QCOMPARE(result.linkAnchoredCount, 0);
            const GlobalPlacedFiber* h0 = findFiber(result, climbing.h0);
            const GlobalPlacedFiber* h1 = findFiber(result, climbing.h1);
            QVERIFY(h0 != nullptr && h1 != nullptr);
            QVERIFY(std::abs((h0->meta.windingLo - h1->meta.windingLo) - 1.0) < 1e-9);
            // Under the default H0's stretch is withheld; Vb's own curtain
            // (link-anchored through H1) stands in.
            const GlobalResult strict = buildGlobalLayout(fibers, climbing.umbilicus, climbing.params,
                                                          nullptr, &climbing.field, std::string());
            QCOMPARE(strict.bentCrossingCount, 1);
            QCOMPARE(strict.linkAnchoredCount, 1);
        }
    }

    // Witnesses that contradict the provisional anchor correct it: the
    // same shelf with Va a thickness BELOW H0 reads the other way round
    // (Vb on the outward side: Inside), and the stretch is counted
    // corrected; a second witness agreeing with the provisional anchor
    // makes the stretch contested and withheld.
    void witnessesCorrectOrContestTheAnchor()
    {
        {
            const ShelfFixture flipped(false, -10.0);
            const GlobalResult result = buildGlobalLayout(flipped.fibers, flipped.umbilicus,
                                                          flipped.params, nullptr, &flipped.field,
                                                          std::string());
            // H0's stretch and the witness Va's own (the link says the same
            // of both).
            QCOMPARE(result.anchorCorrectedStretchCount, 2);
            // H0's corrected curtain now reads Vb Inside while Vb's own
            // curtain (anchored through its link to H1, uncorrected) still
            // reads Outside: the two curtains disagree on the translate,
            // the fold signature, and both readings are withheld; only the
            // witness reading (H0 x Va) constrains, and nothing is dropped.
            QCOMPARE(result.bentCrossingCount, 1);
            int folded = 0;
            for (const CrossingEvent& event : result.crossingEvents) {
                if (event.bent && !event.tangential && event.vFiberId == flipped.vb) {
                    QVERIFY(event.withheld);
                    QCOMPARE(event.bentReason,
                             static_cast<int>(vc3d::fiber_map::winding::BentDisposition::Folded));
                    ++folded;
                }
            }
            QCOMPARE(folded, 2);
            QCOMPARE(result.droppedCrossingCount, 0);
        }
        {
            ShelfFixture contestedFixture(false, 10.0, true);
            const GlobalResult result = buildGlobalLayout(
                contestedFixture.fibers, contestedFixture.umbilicus, contestedFixture.params, nullptr,
                &contestedFixture.field, std::string());
            // H0's readings withheld as contested; the three V curtains (Vb
            // through its link to H1, Va agreeing, Va2 corrected) stand in.
            QVERIFY(result.withheldCount >= 1);
            QCOMPARE(result.bentCrossingCount, 3);
            bool contested = false;
            for (const CrossingEvent& event : result.crossingEvents) {
                if (!event.bent) {
                    continue;
                }
                if (event.withheld) {
                    QVERIFY(!event.curtainFromV);
                    if (event.bentReason ==
                        static_cast<int>(vc3d::fiber_map::winding::BentDisposition::Contested)) {
                        contested = true;
                    }
                } else if (!event.tangential) {
                    QVERIFY(event.curtainFromV);
                }
            }
            QVERIFY(contested);
        }
    }

    // A straight reading set aside with no curtain reaching the other
    // fiber reports noBentReach: a vertical V through the shelf band at a
    // radius the level H's vertical rays never reach.
    void shelfWithoutReachIsReportedUncovered()
    {
        ShelfFixture fixture;
        // Vc: vertical through the band, 100 vx behind H0's radius, at an
        // angle inside H0's arc; its only crossing with H0 is in the band.
        std::vector<cv::Vec3d> vertical = verticalPoints(fixture.thetaA + 0.3, fixture.radius + 100.0,
                                                         fixture.field.z0 + 50.0,
                                                         fixture.field.z1 - 50.0, 25.0);
        const int last = static_cast<int>(vertical.size()) - 1;
        InputFiber vc = makeFiber(990, QStringLiteral("s-vc"), 'V', std::move(vertical), {0, last / 2, last});
        std::vector<InputFiber> fibers = fixture.fibers;
        fibers.push_back(std::move(vc));
        const GlobalResult result = buildGlobalLayout(fibers, fixture.umbilicus, fixture.params,
                                                      nullptr, &fixture.field, std::string());
        bool found = false;
        for (const PairCoverageRecord& record : result.pairCoverage) {
            if (record.hFiberId == fixture.h0 && record.vFiberId == 990) {
                found = true;
                QVERIFY(record.setAsideCount >= 1);
                QCOMPARE(record.bentCount, 0);
                QCOMPARE(record.reason, std::string("noBentReach"));
            }
        }
        QVERIFY(found);
        QVERIFY(result.setAsideCount >= 2);
    }

    // The conditioning gate reads the field at the detection's own
    // positions: an H and a V whose samples are ALL well conditioned (no
    // ill-conditioned run anywhere) but whose crossing sits between samples
    // in a pocket where the sheet is level have that crossing set aside.
    // The crossing lies between H samples (half a step off the control
    // angle) and between V samples; the pocket is smaller than half a
    // sample step, around the H's position only. Then the same with the
    // H's domain starting past its first line points (a nonzero domain
    // offset) and a V folded in height (the hit on its second branch, read
    // through the branch's sample provenance).
    void gateReadsTheFieldAtTheIntersection()
    {
        const double angle = 0.5 * kTwoPi + 0.5 * kStep;
        const double radius = 4000.0;
        const cv::Vec3d hPosition(radius * std::cos(angle), radius * std::sin(angle), 30000.0);
        const GlobalLayoutParams params = sensedParams(1);
        const PocketField pocket(hPosition, 8.0, "pocket");
        {
            std::vector<cv::Vec3d> arc;
            for (int i = 0; i <= 500; ++i) {
                const double theta = 0.25 * kTwoPi + 0.5 * kTwoPi * i / 500.0;
                arc.emplace_back(radius * std::cos(theta), radius * std::sin(theta), 30000.0);
            }
            std::vector<InputFiber> fibers;
            fibers.push_back(makeFiber(900, QStringLiteral("g-h"), 'H', arc, {0, 250, 500}));
            fibers.push_back(makeFiber(901, QStringLiteral("g-v"), 'V',
                                       verticalPoints(angle, radius - 100.0, 29012.0, 31012.0, 25.0),
                                       {0, 40, 80}));
            const GlobalResult plain = buildGlobalLayout(fibers, straightUmbilicus(60000), params);
            QCOMPARE(plain.crossingEvents.size(), std::size_t{1});
            const GlobalResult gated = buildGlobalLayout(fibers, straightUmbilicus(60000), params,
                                                         nullptr, &pocket, std::string());
            QCOMPARE(gated.unorientedRunCount + gated.unsupportedRunCount, 0);
            QCOMPARE(gated.setAsideCount, 1);
            QVERIFY(gated.crossingEvents.empty());
            QCOMPARE(gated.pairCoverage.size(), std::size_t{1});
            QCOMPARE(gated.pairCoverage.front().reason, std::string("noBentReach"));
            // A pocket a sample step away from the crossing sets nothing aside.
            const PocketField beside(hPosition + cv::Vec3d(0.0, 0.0, 40.0), 8.0, "beside");
            const GlobalResult clear = buildGlobalLayout(fibers, straightUmbilicus(60000), params,
                                                         nullptr, &beside, std::string());
            QCOMPARE(clear.setAsideCount, 0);
            QCOMPARE(clear.crossingEvents.size(), std::size_t{1});
        }
        {
            // Domain offset: the H's controls start 60 samples in; the V
            // climbs through the crossing height, folds above, and comes
            // back down through it (two branches, two crossings at the same
            // H position).
            std::vector<cv::Vec3d> arc;
            for (int i = 0; i <= 500; ++i) {
                const double theta = 0.25 * kTwoPi + 0.5 * kTwoPi * i / 500.0;
                arc.emplace_back(radius * std::cos(theta), radius * std::sin(theta), 30000.0);
            }
            std::vector<cv::Vec3d> fold = verticalPoints(angle, radius - 100.0, 29012.0, 30512.0, 25.0);
            std::vector<cv::Vec3d> down = verticalPoints(angle, radius - 60.0, 29012.0, 30487.0, 25.0);
            std::reverse(down.begin(), down.end());
            fold.insert(fold.end(), down.begin(), down.end());
            const int last = static_cast<int>(fold.size()) - 1;
            std::vector<InputFiber> fibers;
            fibers.push_back(makeFiber(900, QStringLiteral("g-h"), 'H', arc, {60, 250, 440}));
            fibers.push_back(makeFiber(901, QStringLiteral("g-v"), 'V', fold, {0, 60, last}));
            const GlobalResult plain = buildGlobalLayout(fibers, straightUmbilicus(60000), params);
            QCOMPARE(plain.crossingEvents.size(), std::size_t{2});
            const GlobalResult gated = buildGlobalLayout(fibers, straightUmbilicus(60000), params,
                                                         nullptr, &pocket, std::string());
            QCOMPARE(gated.unorientedRunCount + gated.unsupportedRunCount, 0);
            QCOMPARE(gated.setAsideCount, 2);
            QVERIFY(gated.crossingEvents.empty());
            // A pocket at the V's SECOND branch only (radius 3940), with the
            // V's domain starting past its first samples: the one crossing
            // read through that branch's provenance is set aside, the
            // other stands.
            const cv::Vec3d vSecond((radius - 60.0) * std::cos(angle), (radius - 60.0) * std::sin(angle),
                                    30000.0);
            const PocketField vPocket(vSecond, 8.0, "v-pocket");
            std::vector<InputFiber> offsetV = fibers;
            offsetV[1] = makeFiber(901, QStringLiteral("g-v"), 'V', fibers[1].linePoints, {10, 60, last - 10});
            const GlobalResult vGated = buildGlobalLayout(offsetV, straightUmbilicus(60000), params,
                                                          nullptr, &vPocket, std::string());
            QCOMPARE(vGated.unorientedRunCount + vGated.unsupportedRunCount, 0);
            QCOMPARE(vGated.setAsideCount, 1);
            QCOMPARE(vGated.crossingEvents.size(), std::size_t{1});
        }
    }

    // The radial-inversion gate reads the final normal interpolated between
    // the hit's samples against the radial direction at the hit's own
    // position: the reviewer's umbilicus, at x = 0 at the fibers' sample
    // heights but x = 2000 at the crossing height, with a witnessed +x
    // normal. Both samples read outward; the hit itself reads inward and
    // is set aside.
    void inversionIsReadAtTheHitPosition()
    {
        std::vector<cv::Vec3f> umbilicus;
        for (int z = -40; z <= 40; ++z) {
            umbilicus.emplace_back(z == 0 ? 2000.0f : 0.0f, 0.0f, static_cast<float>(z));
        }
        InputFiber h = makeFiber(910, QStringLiteral("i-h"), 'H',
                                 {cv::Vec3d(1000.0, -100.0, -10.0), cv::Vec3d(1000.0, 100.0, 10.0)}, {0, 1});
        InputFiber v = makeFiber(911, QStringLiteral("i-v"), 'V',
                                 {cv::Vec3d(1200.0, 0.0, -20.0), cv::Vec3d(1200.0, 0.0, 20.0)}, {0, 1});
        // The witness: P_v - P_h points along +x, agreeing with the field.
        addLink(h, 0, v, 0);
        std::vector<InputFiber> fibers{h, v};
        GlobalLayoutParams params = sensedParams(1);
        params.solver.minUmbilicusRadiusVx = 10.0;
        params.solver.maxStepTurns = 0.5;
        params.minPadXVx = 10.0;
        params.minPadYVx = 10.0;
        params.resampleStepVx = 2.0;
        const GlobalResult plain = buildGlobalLayout(fibers, umbilicus, params);
        QCOMPARE(plain.crossingEvents.size(), std::size_t{1});
        const ConstantField field(cv::Vec3d(1.0, 0.0, 0.0), "x");
        const GlobalResult gated = buildGlobalLayout(fibers, umbilicus, params, nullptr, &field, std::string());
        QCOMPARE(gated.setAsideCount, 0);
        QCOMPARE(gated.radialInvertedCount, 1);
        QVERIFY(gated.crossingEvents.empty());
    }

    // The shelf reads the same whichever way either fiber is stored: the
    // bent events in the same order with the same hits and verdicts, and
    // under the forced repair the same dropped event and mark.
    void shelfReadsTheSameUnderReversal()
    {
        const ShelfFixture fixture;
        std::vector<InputFiber> conflicted = fixture.fibers;
        for (InputFiber& fiber : conflicted) {
            if (fiber.id == fixture.h0) {
                for (InputFiber& other : conflicted) {
                    if (other.id == fixture.h1) {
                        addLink(fiber, 2, other, 1);
                    }
                }
            }
        }
        struct Reading {
            uint64_t h;
            uint64_t v;
            bool fromV;
            int kind;
            long long n;
            bool withheld;
            int status;
            double hitX;
            double hitY;
            double hitZ;
            bool operator==(const Reading& o) const
            {
                return h == o.h && v == o.v && fromV == o.fromV && kind == o.kind && n == o.n &&
                       withheld == o.withheld && status == o.status && std::abs(hitX - o.hitX) < 1e-6 &&
                       std::abs(hitY - o.hitY) < 1e-6 && std::abs(hitZ - o.hitZ) < 1e-6;
            }
        };
        const auto readings = [](const GlobalResult& result) {
            std::vector<Reading> out;
            for (const CrossingEvent& e : result.crossingEvents) {
                if (e.bent) {
                    out.push_back({e.hFiberId, e.vFiberId, e.curtainFromV, static_cast<int>(e.kind), e.n,
                                   e.withheld, static_cast<int>(e.status), e.hitVx[0], e.hitVx[1],
                                   e.hitVx[2]});
                }
            }
            return out;
        };
        std::vector<std::vector<Reading>> plainReadings;
        std::vector<std::vector<Reading>> conflictReadings;
        std::vector<cv::Vec3d> markHits;
        for (const bool reverseH : {false, true}) {
            for (const bool reverseV : {false, true}) {
                std::vector<InputFiber> fibers = fixture.fibers;
                std::vector<InputFiber> withConflict = conflicted;
                if (reverseH) {
                    fibers = withReversed(fibers, fixture.h0);
                    withConflict = withReversed(withConflict, fixture.h0);
                }
                if (reverseV) {
                    fibers = withReversed(fibers, fixture.vb);
                    withConflict = withReversed(withConflict, fixture.vb);
                }
                const GlobalResult plain = buildGlobalLayout(fibers, fixture.umbilicus, fixture.params,
                                                             nullptr, &fixture.field, std::string());
                const GlobalResult repaired = buildGlobalLayout(withConflict, fixture.umbilicus,
                                                                fixture.params, nullptr, &fixture.field,
                                                                std::string());
                QCOMPARE(plain.bentCrossingCount, 2);
                QCOMPARE(plain.droppedCrossingCount, 0);
                QCOMPARE(repaired.droppedCrossingCount, 1);
                QCOMPARE(repaired.suspectCrossings.size(), std::size_t{1});
                QVERIFY(repaired.suspectCrossings.front().bent);
                plainReadings.push_back(readings(plain));
                conflictReadings.push_back(readings(repaired));
                markHits.push_back(repaired.suspectCrossings.front().hitVx);
            }
        }
        for (std::size_t i = 1; i < plainReadings.size(); ++i) {
            QVERIFY(plainReadings[i] == plainReadings[0]);
            QVERIFY(conflictReadings[i] == conflictReadings[0]);
            const cv::Vec3d d = markHits[i] - markHits[0];
            QVERIFY(std::sqrt(d.dot(d)) < 1e-6);
        }
        QVERIFY(plainReadings[0].size() >= 2);
    }

    // The whole pipeline under reversal, with the exported positions and
    // violations pinned: the shelf, and the shelf with H0 going round twice
    // over the same arc (two passes of one strip geometry, translates one
    // apart, ordered by translate whichever way the samples are stored),
    // each with the forced repair. Every bent event's position, kind,
    // translate, status, violation and hit, the dropped events' places in
    // the bent sequence and the mark's event are identical under both
    // fibers' reversals.
    void readingsAndMarksAreStorageIndependent()
    {
        struct Reading {
            uint64_t h, v;
            bool fromV;
            int kind;
            long long n;
            bool withheld;
            int status;
            double violation;
            double x, y, hx, hy, hz;
        };
        const auto same = [](const Reading& a, const Reading& b) {
            return a.h == b.h && a.v == b.v && a.fromV == b.fromV && a.kind == b.kind && a.n == b.n &&
                   a.withheld == b.withheld && a.status == b.status && a.violation == b.violation &&
                   std::abs(a.x - b.x) < 1e-6 && std::abs(a.y - b.y) < 1e-6 && std::abs(a.hx - b.hx) < 1e-6 &&
                   std::abs(a.hy - b.hy) < 1e-6 && std::abs(a.hz - b.hz) < 1e-6;
        };
        for (const int passes : {1, 2}) {
            // The second pass one voxel above the first: the layout maps a
            // control to its first nearest line point, so an exact repeat
            // would snap the end control back onto the first pass.
            const ShelfFixture fixture(false, 10.0, false, passes, passes == 2 ? 1.0 : 0.0);
            std::vector<InputFiber> conflicted = fixture.fibers;
            for (InputFiber& fiber : conflicted) {
                if (fiber.id == fixture.h0) {
                    for (InputFiber& other : conflicted) {
                        if (other.id == fixture.h1) {
                            addLink(fiber, 2, other, 1);
                        }
                    }
                }
            }
            std::vector<std::vector<Reading>> sequences;
            std::vector<std::vector<std::size_t>> markPlacesPerRun;
            std::vector<std::vector<long long>> translates;
            for (const bool reverseH : {false, true}) {
                for (const bool reverseV : {false, true}) {
                    std::vector<InputFiber> fibers = conflicted;
                    if (reverseH) {
                        fibers = withReversed(fibers, fixture.h0);
                    }
                    if (reverseV) {
                        fibers = withReversed(fibers, fixture.vb);
                    }
                    const GlobalResult result = buildGlobalLayout(fibers, fixture.umbilicus,
                                                                  fixture.params, nullptr, &fixture.field,
                                                                  std::string());
                    std::vector<Reading> sequence;
                    std::vector<long long> ns;
                    std::vector<std::size_t> markPlaces;
                    std::size_t place = 0;
                    // One dropped reading per pass, each ringed.
                    QCOMPARE(result.suspectCrossings.size(), static_cast<std::size_t>(passes));
                    for (std::size_t e = 0; e < result.crossingEvents.size(); ++e) {
                        const CrossingEvent& ev = result.crossingEvents[e];
                        if (!ev.bent) {
                            continue;
                        }
                        if (ev.vFiberId == fixture.vb && !ev.withheld && !ev.tangential) {
                            ns.push_back(ev.n);
                        }
                        for (const CrossingMark& mark : result.suspectCrossings) {
                            if (mark.eventIndex == e) {
                                markPlaces.push_back(place);
                            }
                        }
                        sequence.push_back({ev.hFiberId, ev.vFiberId, ev.curtainFromV, static_cast<int>(ev.kind),
                                            ev.n, ev.withheld, static_cast<int>(ev.status), ev.violationTurns,
                                            ev.posVx.x(), ev.posVx.y(), ev.hitVx[0], ev.hitVx[1], ev.hitVx[2]});
                        ++place;
                    }
                    sequences.push_back(sequence);
                    markPlacesPerRun.push_back(markPlaces);
                    translates.push_back(ns);
                }
            }
            for (std::size_t i = 1; i < sequences.size(); ++i) {
                QCOMPARE(sequences[i].size(), sequences[0].size());
                for (std::size_t k = 0; k < sequences[0].size(); ++k) {
                    QVERIFY(same(sequences[i][k], sequences[0][k]));
                }
                QCOMPARE(markPlacesPerRun[i], markPlacesPerRun[0]);
                QCOMPARE(translates[i], translates[0]);
            }
            if (passes == 2) {
                // Two passes read Vb, one winding apart (the second pass is
                // a turn on, so Vb stands one winding closer to it).
                QCOMPARE(translates[0].size(), std::size_t{2});
                QCOMPARE(std::abs(translates[0][1] - translates[0][0]), 1LL);
            }
            bool dropped = false;
            for (const Reading& r : sequences[0]) {
                if (r.status == static_cast<int>(CrossingStatus::Dropped)) {
                    dropped = true;
                    QCOMPARE(r.violation, 1.0);
                }
            }
            QVERIFY(dropped);
        }
    }

    // The owner's own polyline crossing its curtain: a fold (the crossing
    // on the same winding) bounds the readings beyond it, the fiber's next
    // winding passing through does not. H0 with a returning limb 100 vx
    // above its outgoing limb and Vb's run 400 vx above: BeyondFold. H0
    // going round again 60 vx higher: the second pass crosses the first
    // pass's curtain one turn on, and Vb above is read.
    void foldBoundsReadingsButTheNextWindingDoesNot()
    {
        {
            ShelfFixture folded(false, 10.0, false, 1, 0.0, 100.0);
            // Vb's radial run above the shelf instead of below.
            for (InputFiber& fiber : folded.fibers) {
                if (fiber.id == folded.vb) {
                    for (cv::Vec3d& p : fiber.linePoints) {
                        if (p[2] <= folded.zRun + 1e-9) {
                            p[2] = folded.zRun + 2.0 * folded.pitch;
                        }
                    }
                    for (cv::Vec3d& p : fiber.controlPoints) {
                        if (p[2] <= folded.zRun + 1e-9) {
                            p[2] = folded.zRun + 2.0 * folded.pitch;
                        }
                    }
                }
            }
            const GlobalResult result = buildGlobalLayout(folded.fibers, folded.umbilicus, folded.params,
                                                          nullptr, &folded.field, std::string());
            // The outgoing limb's ray (400 vx up to Vb) passes the returning
            // limb at 100 vx: beyond the fold, withheld. The returning limb's
            // own ray (300 vx) meets nothing of H0 on its way: Vb is read
            // from the limb nearest it, at the same translate.
            int beyond = 0;
            int usable = 0;
            std::optional<long long> translate;
            for (const CrossingEvent& ev : result.crossingEvents) {
                if (ev.bent && !ev.curtainFromV && ev.vFiberId == folded.vb && !ev.tangential) {
                    if (ev.rayLengthVx > 350.0) {
                        QVERIFY(ev.withheld);
                        QCOMPARE(ev.bentReason,
                                 static_cast<int>(vc3d::fiber_map::winding::BentDisposition::BeyondFold));
                        ++beyond;
                    } else {
                        QVERIFY(!ev.withheld);
                        ++usable;
                    }
                    if (translate) {
                        QCOMPARE(ev.n, *translate);
                    }
                    translate = ev.n;
                }
            }
            QCOMPARE(beyond, 1);
            QCOMPARE(usable, 1);
        }
        {
            ShelfFixture spiral(false, 10.0, false, 2, 60.0);
            for (InputFiber& fiber : spiral.fibers) {
                if (fiber.id == spiral.vb) {
                    for (cv::Vec3d& p : fiber.linePoints) {
                        if (p[2] <= spiral.zRun + 1e-9) {
                            p[2] = spiral.zRun + 2.0 * spiral.pitch;
                        }
                    }
                    for (cv::Vec3d& p : fiber.controlPoints) {
                        if (p[2] <= spiral.zRun + 1e-9) {
                            p[2] = spiral.zRun + 2.0 * spiral.pitch;
                        }
                    }
                }
            }
            const GlobalResult result = buildGlobalLayout(spiral.fibers, spiral.umbilicus, spiral.params,
                                                          nullptr, &spiral.field, std::string());
            int usable = 0;
            for (const CrossingEvent& ev : result.crossingEvents) {
                if (ev.bent && !ev.curtainFromV && ev.vFiberId == spiral.vb && !ev.tangential &&
                    !ev.withheld) {
                    ++usable;
                }
                if (ev.bent && ev.withheld) {
                    QVERIFY(ev.bentReason !=
                            static_cast<int>(vc3d::fiber_map::winding::BentDisposition::BeyondFold));
                }
            }
            QVERIFY(usable >= 1);
        }
    }

    // Tied geometry through the layout: H0 twice over the same arc one
    // turn apart, Vb through it twice one winding apart. Four readings at one
    // position with translates -1, 0, 0, +1 in event order (Inside: H on or
    // inside Vb's winding, one-sided); the two at translate 0 ordered by the
    // owner's lifted position (pass 1 first); against the links fixing Vb on
    // H0's winding the translate -1 reading (H0's second pass a winding
    // outside Vb) drops with violation 1 and is marked at its own winding
    // position; all of it the same under the four storage orders.
    void tiedGeometryReadingsOrderByTranslateAndLift()
    {
        const TiedFixture fixture(TiedOptions{});
        bool ok = false;
        const BentSnapshot snapshot = bentReadingsUnderReversals(fixture.fibers, fixture.umbilicus, fixture.params,
                                                                 fixture.field, fixture.h0, fixture.vb, &ok);
        QVERIFY(ok);
        std::vector<const BentExport*> readings;
        for (const BentExport& e : snapshot.events) {
            if (e.h == fixture.h0 && e.v == fixture.vb && !e.fromV && !e.tangential && !e.touch) {
                readings.push_back(&e);
            }
        }
        QCOMPARE(readings.size(), std::size_t{4});
        const std::vector<long long> expectedN{-1, 0, 0, 1};
        const cv::Vec3d at(fixture.radius * std::cos(fixture.crossAngle), fixture.radius * std::sin(fixture.crossAngle),
                           fixture.zShelf + 4.0);
        for (std::size_t k = 0; k < 4; ++k) {
            QCOMPARE(readings[k]->n, expectedN[k]);
            QVERIFY(!readings[k]->withheld);
            QCOMPARE(readings[k]->kind, static_cast<int>(CrossingKind::Inside));
            // One position: the crossing of the arc, 4 vx up.
            // On the strip's face, a chord of the arc: within its sagitta.
            QVERIFY(std::abs(readings[k]->hit[0] - at[0]) < 0.05);
            QVERIFY(std::abs(readings[k]->hit[1] - at[1]) < 0.05);
            QVERIFY(std::abs(readings[k]->hit[2] - at[2]) < 1e-6);
            if (k > 0) {
                QVERIFY(readings[k]->eventIndex > readings[k - 1]->eventIndex);
            }
        }
        // The two translate-0 readings: pass 1 (the smaller lifted H
        // position) before pass 2, a turn apart in the layout; the readings
        // of each pass sit at that pass's winding position.
        QVERIFY(readings[1]->x < readings[2]->x);
        QVERIFY(readings[2]->x - readings[1]->x > 0.5 * kTwoPi * fixture.radius);
        QVERIFY(std::abs(readings[0]->x - readings[2]->x) < 1e-6);
        QVERIFY(std::abs(readings[3]->x - readings[1]->x) < 1e-6);
        QCOMPARE(readings[0]->status, static_cast<int>(CrossingStatus::Dropped));
        QCOMPARE(readings[0]->violation, 1.0);
        for (std::size_t k = 1; k < 4; ++k) {
            QCOMPARE(readings[k]->status, static_cast<int>(CrossingStatus::Used));
            QCOMPARE(readings[k]->violation, 0.0);
        }
        int marked = 0;
        for (const MarkExport& mark : snapshot.marks) {
            for (const BentExport* r : readings) {
                if (mark.eventIndex == r->eventIndex) {
                    QCOMPARE(r, readings[0]);
                    QVERIFY(std::abs(mark.x - r->x) < 1e-6);
                    QVERIFY(std::abs(mark.y - r->y) < 1e-6);
                    QVERIFY(std::abs(mark.hit[0] - at[0]) < 0.05);
                    QCOMPARE(mark.violation, 1.0);
                    ++marked;
                }
            }
        }
        QCOMPARE(marked, 1);
    }

    // The equal-length tie through the layout: Vb's hairpin crosses one
    // strip twice at the same ray length (4 vx) on the same translate, half
    // a strip apart, the radial passage more transversal than the oblique
    // return. One reading per (strip, translate), the more transversal: two
    // readings for the two passes, both at the radial passage's point; the
    // second pass's (H0 a winding outside Vb there) drops with violation 1;
    // the same under the four storage orders.
    void equalLengthReadingsKeepTheMoreTransversal()
    {
        TiedOptions options;
        options.loop = false;
        const TiedFixture fixture(options);
        bool ok = false;
        const BentSnapshot snapshot = bentReadingsUnderReversals(fixture.fibers, fixture.umbilicus, fixture.params,
                                                                 fixture.field, fixture.h0, fixture.vb, &ok);
        QVERIFY(ok);
        std::vector<const BentExport*> readings;
        for (const BentExport& e : snapshot.events) {
            if (e.h == fixture.h0 && e.v == fixture.vb && !e.fromV && !e.tangential && !e.touch) {
                readings.push_back(&e);
            }
        }
        QCOMPARE(readings.size(), std::size_t{2});
        // Translates one apart (H0's second pass a turn on), in order.
        QCOMPARE(readings[1]->n - readings[0]->n, 1LL);
        QVERIFY(readings[1]->eventIndex > readings[0]->eventIndex);
        const cv::Vec3d at(fixture.radius * std::cos(fixture.crossAngle), fixture.radius * std::sin(fixture.crossAngle),
                           fixture.zShelf + 4.0);
        QCOMPARE(readings[0]->status, static_cast<int>(CrossingStatus::Dropped));
        QCOMPARE(readings[0]->violation, 1.0);
        QCOMPARE(readings[1]->status, static_cast<int>(CrossingStatus::Used));
        QCOMPARE(readings[1]->violation, 0.0);
        QVERIFY(readings[0]->x > readings[1]->x + 0.5 * kTwoPi * fixture.radius);
        int marked = 0;
        for (const MarkExport& mark : snapshot.marks) {
            if (mark.eventIndex == readings[0]->eventIndex) {
                QVERIFY(std::abs(mark.x - readings[0]->x) < 1e-6);
                QVERIFY(std::abs(mark.hit[0] - at[0]) < 0.05);
                ++marked;
            }
            QVERIFY(mark.eventIndex != readings[1]->eventIndex);
        }
        QCOMPARE(marked, 1);
        for (const BentExport* r : readings) {
            QCOMPARE(r->kind, static_cast<int>(CrossingKind::Inside));
            QVERIFY(!r->withheld);
            QVERIFY(std::abs(r->hit[0] - at[0]) < 0.05);
            QVERIFY(std::abs(r->hit[1] - at[1]) < 0.05);
            QVERIFY(std::abs(r->hit[2] - at[2]) < 1e-6);
        }
    }

    // A boundary case of the curtain geometry through the layout: Vb
    // crosses H0's two passes once each. Expected: two H-owned readings at
    // the hit (one per pass, translates one apart, the second pass's - H0 a
    // winding outside Vb there - dropped with violation 1 and marked at its
    // own winding position, the first pass's used), no V-owned reading left
    // beside them (H preferred), and all of it the same under the four
    // storage orders.
private:
    void checkBoundaryCase(const TiedOptions& options, double hitTolerance, double expectedZ = -1.0)
    {
        const TiedFixture fixture(options);
        bool ok = false;
        const BentSnapshot snapshot = bentReadingsUnderReversals(fixture.fibers, fixture.umbilicus, fixture.params,
                                                                 fixture.field, fixture.h0, fixture.vb, &ok);
        QVERIFY(ok);
        std::vector<const BentExport*> readings;
        for (const BentExport& e : snapshot.events) {
            if (e.h != fixture.h0 || e.v != fixture.vb) {
                continue;
            }
            QVERIFY(!e.tangential && !e.touch);
            QVERIFY(!e.fromV);
            readings.push_back(&e);
        }
        QCOMPARE(readings.size(), std::size_t{2});
        QCOMPARE(readings[1]->n - readings[0]->n, 1LL);
        QVERIFY(readings[1]->eventIndex > readings[0]->eventIndex);
        QVERIFY(readings[0]->x > readings[1]->x + 0.5 * kTwoPi * fixture.radius);
        const double zHit = expectedZ >= 0.0 ? expectedZ : fixture.zShelf + options.crossZ;
        const cv::Vec3d at(fixture.radius * std::cos(fixture.crossAngle), fixture.radius * std::sin(fixture.crossAngle),
                           zHit);
        for (const BentExport* r : readings) {
            QCOMPARE(r->kind, static_cast<int>(CrossingKind::Inside));
            QVERIFY(!r->withheld);
            QVERIFY(std::abs(r->hit[0] - at[0]) < hitTolerance);
            QVERIFY(std::abs(r->hit[1] - at[1]) < hitTolerance);
            QVERIFY(std::abs(r->hit[2] - at[2]) < 1e-6);
            QVERIFY(std::abs(r->rayLength - (zHit - fixture.zShelf)) < 1e-6);
            QVERIFY(r->transversality > 0.99);
        }
        QCOMPARE(readings[0]->status, static_cast<int>(CrossingStatus::Dropped));
        QCOMPARE(readings[0]->violation, 1.0);
        QCOMPARE(readings[1]->status, static_cast<int>(CrossingStatus::Used));
        QCOMPARE(readings[1]->violation, 0.0);
        int marked = 0;
        for (const MarkExport& mark : snapshot.marks) {
            if (mark.eventIndex == readings[0]->eventIndex) {
                QVERIFY(std::abs(mark.x - readings[0]->x) < 1e-6);
                QVERIFY(std::abs(mark.hit[2] - at[2]) < 1e-6);
                QCOMPARE(mark.violation, 1.0);
                ++marked;
            }
            QVERIFY(mark.eventIndex != readings[1]->eventIndex);
        }
        QCOMPARE(marked, 1);
    }

private slots:
    // H0's second pass runs the arc the other way (the reviewer's A->B then
    // B->A a turn on): the strips of the two passes are the same surface
    // from rays in opposite order, read bit for bit alike, so the two
    // readings order by translate, not by the rounding of either pass.
    void oppositeDirectionPassesReadAlike()
    {
        TiedOptions options;
        options.loop = false;
        options.singlePassage = true;
        options.reversedSecondPass = true;
        options.crossZ = 4.2;
        checkBoundaryCase(options, 0.05);
    }

    // Vb through H0's own line (the seed edge of the curtain): a contact of
    // the outward side of each pass, from either owner's curtain - no
    // reading, nothing withheld, nothing marked - the same under the four
    // storage orders.
    void seedEdgePassageIsAContactThroughTheLayout()
    {
        TiedOptions options;
        options.loop = false;
        options.singlePassage = true;
        options.crossZ = 0.0;
        const TiedFixture fixture(options);
        bool ok = false;
        const BentSnapshot snapshot = bentReadingsUnderReversals(fixture.fibers, fixture.umbilicus, fixture.params,
                                                                 fixture.field, fixture.h0, fixture.vb, &ok);
        QVERIFY(ok);
        int hContacts = 0;
        int vContacts = 0;
        for (const BentExport& e : snapshot.events) {
            if (e.h != fixture.h0 || e.v != fixture.vb) {
                continue;
            }
            QVERIFY(e.tangential && e.touch);
            QVERIFY(!e.withheld);
            QVERIFY(std::abs(e.hit[2] - fixture.zShelf) < 1e-6);
            for (const MarkExport& mark : snapshot.marks) {
                QVERIFY(mark.eventIndex != e.eventIndex);
            }
            (e.fromV ? vContacts : hContacts) += 1;
        }
        QCOMPARE(hContacts, 2);
        QCOMPARE(vContacts, 2);
    }

    // Vb exactly on the first row of the rays (8 vx up, the ray step): a
    // crossing on the row shared by two quads is one reading per pass.
    void rowEdgePassageIsOneReading()
    {
        TiedOptions options;
        options.loop = false;
        options.singlePassage = true;
        options.crossZ = 8.0;
        checkBoundaryCase(options, 0.05);
    }

    // Vb exactly at a seed's angle: a crossing on the ray shared by two
    // strips is one reading per pass, owned by provenance.
    void sharedRayPassageIsOneReading()
    {
        TiedOptions options;
        options.loop = false;
        options.singlePassage = true;
        options.atSeedRay = true;
        checkBoundaryCase(options, 1e-6);
    }

    // H0's first sample stored twice (two seeds at one point, a zero-width
    // strip beside the first real one) and Vb crossing on the ray they
    // share: the empty strip is no surface; one reading, used, unmarked,
    // the same under the four storage orders.
    void repeatedSampleIsOneReading()
    {
        TiedOptions options;
        options.loop = false;
        options.singlePassage = true;
        options.duplicateSample = true;
        const TiedFixture fixture(options);
        bool ok = false;
        const BentSnapshot snapshot = bentReadingsUnderReversals(fixture.fibers, fixture.umbilicus, fixture.params,
                                                                 fixture.field, fixture.h0, fixture.vb, &ok);
        QVERIFY(ok);
        std::vector<const BentExport*> readings;
        for (const BentExport& e : snapshot.events) {
            if (e.h != fixture.h0 || e.v != fixture.vb || e.fromV) {
                continue;
            }
            QVERIFY(!e.tangential && !e.touch);
            readings.push_back(&e);
        }
        QCOMPARE(readings.size(), std::size_t{1});
        const BentExport& r = *readings[0];
        QCOMPARE(r.kind, static_cast<int>(CrossingKind::Inside));
        QVERIFY(!r.withheld);
        QCOMPARE(r.status, static_cast<int>(CrossingStatus::Used));
        QCOMPARE(r.violation, 0.0);
        QVERIFY(std::abs(r.hit[0] - fixture.radius * std::cos(fixture.crossAngle)) < 1e-6);
        QVERIFY(std::abs(r.hit[1] - fixture.radius * std::sin(fixture.crossAngle)) < 1e-6);
        QVERIFY(std::abs(r.hit[2] - (fixture.zShelf + 4.0)) < 1e-6);
        QVERIFY(std::abs(r.rayLength - 4.0) < 1e-6);
        for (const MarkExport& mark : snapshot.marks) {
            QVERIFY(mark.eventIndex != r.eventIndex);
        }
    }

    // Vb crosses strip (ib, ib+1) inside it
    // 10 vx up and comes back on the ray ib (shared with the strip before)
    // 12 vx up; the shared-ray hit is the same strip's by provenance (its
    // other ray's start is the lexicographically smaller), so first per
    // strip and translate keeps the 10 vx reading: one per pass, the second
    // pass's dropped and marked, under the four storage orders.
    void interiorThenSharedRayHitsReduceThroughTheLayout()
    {
        TiedOptions options;
        options.loop = false;
        options.returnOnSeedRay = true;
        options.crossZ = 10.0;
        checkBoundaryCase(options, 0.05);
    }

    // Vb exactly at a seed's angle and exactly on the first row (8 vx up):
    // a crossing at a mesh vertex, where four quads meet, is one reading
    // per pass.
    void meshVertexPassageIsOneReading()
    {
        TiedOptions options;
        options.loop = false;
        options.singlePassage = true;
        options.atSeedRay = true;
        options.crossZ = 8.0;
        checkBoundaryCase(options, 1e-6);
    }

    // Shared-ray ownership through reduction: Vb's hairpin
    // crosses the seed's ray twice, 10 and 12 vx up, both hits owned by the
    // same strip (by provenance); first per strip and translate keeps the
    // 10 vx one. One reading per pass, at 10 vx.
    void sharedRayHitsReduceToTheFirstEncounter()
    {
        TiedOptions options;
        options.loop = false;
        options.atSeedRay = true;
        options.hairpinOnRay = true;
        options.crossZ = 10.0;
        const TiedFixture fixture(options);
        checkBoundaryCase(options, 1e-6, fixture.zShelf + 10.0);
    }

    // With a field, every cached shard of a second cold cache is the first
    // cold cache's bit for bit (field readings, set-asides and self
    // crossings included).
    void fieldBearingShardsAreIdenticalAcrossCaches()
    {
        const ShelfFixture fixture;
        GlobalLayoutCache a;
        GlobalLayoutCache b;
        (void)buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params, &a, &fixture.field,
                                std::string());
        (void)buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params, &b, &fixture.field,
                                std::string());
        const auto shardsA = a.cachedDetections();
        const auto shardsB = b.cachedDetections();
        QCOMPARE(shardsA.size(), shardsB.size());
        QVERIFY(!shardsA.empty());
        int hits = 0;
        for (std::size_t i = 0; i < shardsA.size(); ++i) {
            QVERIFY(vc3d::fiber_map::winding::identicalPairDetections(*shardsA[i], *shardsB[i]));
            hits += static_cast<int>(shardsA[i]->bentHits.size());
        }
        QVERIFY(hits > 0);
    }

    // A link that is a kollesis seam anchor is no witness: with Va on a
    // kollesis (linked to H0's tagged start and to a second tagged H on
    // the other side), H0's stretch has no witness and its readings are
    // withheld, where the plain fixture's are anchored by that link.
    void seamLinksWitnessNothing()
    {
        const ShelfFixture plain;
        const GlobalResult anchored = buildGlobalLayout(plain.fibers, plain.umbilicus, plain.params,
                                                        nullptr, &plain.field, std::string());
        const ShelfFixture seam = ShelfFixture::withSeam();
        const GlobalResult withheld = buildGlobalLayout(seam.fibers, seam.umbilicus, seam.params,
                                                        nullptr, &seam.field, std::string());
        const auto decisionOf = [](const GlobalResult& result, uint64_t id) {
            for (const auto& d : result.stretchDecisions) {
                if (d.fiberId == id) {
                    return d;
                }
            }
            return vc3d::fiber_map::StretchDecisionRecord{};
        };
        QCOMPARE(decisionOf(anchored, plain.h0).witnessesAgree, 1);
        QVERIFY(!decisionOf(anchored, plain.h0).withheld);
        QCOMPARE(decisionOf(withheld, seam.h0).witnessesAgree, 0);
        QVERIFY(decisionOf(withheld, seam.h0).withheld);
        bool vaOnKollesis = false;
        for (const auto& fiber : withheld.fibers) {
            if (fiber.fiber.id == seam.va) {
                vaOnKollesis = fiber.meta.onKollesis;
            }
        }
        QVERIFY(vaOnKollesis);
        for (const CrossingEvent& ev : withheld.crossingEvents) {
            if (ev.bent && !ev.curtainFromV && ev.hFiberId == seam.h0 && !ev.tangential) {
                QVERIFY(ev.withheld);
            }
        }
    }

    // The result digest ties a bent payload to its event and mark: moving
    // the payload to another event, everything else kept, is a different
    // digest.
    void digestTiesBentPayloadToItsEvent()
    {
        const ShelfFixture fixture;
        const GlobalResult result = buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params,
                                                      nullptr, &fixture.field, std::string());
        std::size_t bentIndex = result.crossingEvents.size();
        std::size_t straightIndex = result.crossingEvents.size();
        for (std::size_t i = 0; i < result.crossingEvents.size(); ++i) {
            if (result.crossingEvents[i].bent && bentIndex == result.crossingEvents.size()) {
                bentIndex = i;
            }
            if (!result.crossingEvents[i].bent && straightIndex == result.crossingEvents.size()) {
                straightIndex = i;
            }
        }
        QVERIFY(bentIndex < result.crossingEvents.size());
        QVERIFY(straightIndex < result.crossingEvents.size());
        GlobalResult moved = result;
        CrossingEvent& from = moved.crossingEvents[bentIndex];
        CrossingEvent& to = moved.crossingEvents[straightIndex];
        std::swap(from.bent, to.bent);
        std::swap(from.curtainFromV, to.curtainFromV);
        std::swap(from.withheld, to.withheld);
        std::swap(from.anchor, to.anchor);
        std::swap(from.bentReason, to.bentReason);
        std::swap(from.hitVx, to.hitVx);
        std::swap(from.rayLengthVx, to.rayLengthVx);
        QVERIFY(!(digestGlobalResult(moved) == digestGlobalResult(result)));
    }

    // A bent reading the map violates is a declared error marked with its
    // 3D hit: the shelf fixture with H0 plain-linked to H1 as if on one
    // winding (a clean link, confidence 1.5, and no witness: a link between
    // two H fibers says nothing about a sheet normal) against the strict
    // Outside of the curtain (confidence 1).
    void violatedBentReadingIsMarkedAtItsHit()
    {
        ShelfFixture fixture;
        std::vector<InputFiber> fibers = fixture.fibers;
        InputFiber* h0 = nullptr;
        InputFiber* h1 = nullptr;
        for (InputFiber& fiber : fibers) {
            if (fiber.id == fixture.h0) {
                h0 = &fiber;
            }
            if (fiber.id == fixture.h1) {
                h1 = &fiber;
            }
        }
        QVERIFY(h0 != nullptr && h1 != nullptr);
        // Both controls at Vb's angle.
        addLink(*h0, 2, *h1, 1);
        const GlobalResult result = buildGlobalLayout(fibers, fixture.umbilicus, fixture.params,
                                                      nullptr, &fixture.field, std::string());
        QCOMPARE(result.bentCrossingCount, 2);
        QCOMPARE(result.droppedCrossingCount, 1);
        QCOMPARE(result.suspectCrossings.size(), std::size_t{1});
        const CrossingMark& mark = result.suspectCrossings.front();
        QVERIFY(mark.bent);
        QCOMPARE(mark.violationTurns, 1.0);
        QVERIFY(std::abs(mark.hitVx[2] - fixture.zRun) < 1e-6);
        const CrossingEvent& event = result.crossingEvents[mark.eventIndex];
        QVERIFY(event.bent);
        QCOMPARE(event.status, CrossingStatus::Dropped);
        QCOMPARE(event.hitVx, mark.hitVx);
        QCOMPARE(result.suspectLinkCount, 0);
    }

    // Both senses solved with the field (the link-free deciding builds see
    // no witness and emit no bent constraint under LinksOnly), the kept
    // sense built with the links reads the shelf.
    void automaticSenseBuildsWithTheField()
    {
        ShelfFixture fixture;
        GlobalLayoutParams params = fixture.params;
        params.solver.chiralityOverride = 0;
        GlobalLayoutCache cache;
        const GlobalResult result = buildGlobalLayout(fixture.fibers, fixture.umbilicus, params, &cache,
                                                      &fixture.field, std::string());
        QCOMPARE(result.chirality, 1);
        QCOMPARE(result.bentCrossingCount, 2);
        const GlobalResult stated = buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params,
                                                      nullptr, &fixture.field, std::string());
        GlobalResult comparable = result;
        comparable.chiralityBasis = stated.chiralityBasis;
        comparable.comparedChiralityErrors = stated.comparedChiralityErrors;
        comparable.rejectedChiralityErrors = stated.rejectedChiralityErrors;
        comparable.prepMs = stated.prepMs;
        comparable.detectMs = stated.detectMs;
        comparable.solveMs = stated.solveMs;
        comparable.geometryMs = stated.geometryMs;
        comparable.fieldMs = stated.fieldMs;
        comparable.fieldSampleCount = stated.fieldSampleCount;
        QVERIFY(digestGlobalResult(comparable) == digestGlobalResult(stated));
    }

    // Where the field takes part in the sense comparison: under
    // UmbilicusOrLinks the link-free deciding builds carry umbilicus-anchored
    // bent constraints. The automatic sense agrees with explicit link-free
    // builds of both senses, field included; the comparison counts are
    // those builds' dropped crossings; and a warm replay is the same build.
    void automaticSenseComparesWithUmbilicusAnchoredReadings()
    {
        const ShelfFixture climbing(true);
        GlobalLayoutParams params = climbing.params;
        params.bentRays.anchorPolicy = AnchorPolicy::UmbilicusOrLinks;
        params.solver.chiralityOverride = 0;
        const GlobalResult result = buildGlobalLayout(climbing.fibers, climbing.umbilicus, params, nullptr,
                                                      &climbing.field, std::string());
        QCOMPARE(result.chirality, 1);
        QVERIFY(result.chiralityBasis != ChiralityBasis::Override);
        std::vector<InputFiber> linkFree = climbing.fibers;
        for (InputFiber& fiber : linkFree) {
            fiber.links.clear();
        }
        const auto senseBuild = [&](int sense) {
            GlobalLayoutParams p = params;
            p.solver.chiralityOverride = sense;
            return buildGlobalLayout(linkFree, climbing.umbilicus, p, nullptr, &climbing.field, std::string());
        };
        const GlobalResult plus = senseBuild(1);
        const GlobalResult minus = senseBuild(-1);
        // The deciding builds carried the field (bent constraints in both
        // senses) and the counts the result reports are theirs: dropped
        // crossings plus declared group conflicts, links left out. On this
        // fixture both senses solve without contradiction, so the field's
        // part in the decision shows in the work the result sums over the
        // three builds it made: the field samples of the two link-free
        // deciding builds and of the kept build, exactly - a deciding build
        // made without the field would sample nothing.
        QVERIFY(plus.bentCrossingCount >= 1);
        QVERIFY(minus.bentCrossingCount >= 1);
        QCOMPARE(result.comparedChiralityErrors, plus.droppedCrossingCount + plus.declaredGroupCount);
        QCOMPARE(result.rejectedChiralityErrors, minus.droppedCrossingCount + minus.declaredGroupCount);
        GlobalLayoutParams keptParams = params;
        keptParams.solver.chiralityOverride = 1;
        const GlobalResult keptAlone = buildGlobalLayout(climbing.fibers, climbing.umbilicus, keptParams, nullptr,
                                                         &climbing.field, std::string());
        QVERIFY(plus.fieldSampleCount > 0);
        QCOMPARE(result.fieldSampleCount, plus.fieldSampleCount + minus.fieldSampleCount + keptAlone.fieldSampleCount);
        // Warm replay: the automatic build again from a cache it filled.
        GlobalLayoutCache cache;
        const GlobalResult cold = buildGlobalLayout(climbing.fibers, climbing.umbilicus, params, &cache,
                                                    &climbing.field, std::string());
        const GlobalResult replay = buildGlobalLayout(climbing.fibers, climbing.umbilicus, params, &cache,
                                                      &climbing.field, std::string());
        QCOMPARE(cache.lastStats().pairsRecomputed, 0);
        QCOMPARE(replay.chirality, 1);
        GlobalResult a = cold;
        GlobalResult b = replay;
        for (GlobalResult* r : {&a, &b}) {
            r->prepMs = 0.0;
            r->detectMs = 0.0;
            r->solveMs = 0.0;
            r->geometryMs = 0.0;
            r->fieldMs = 0.0;
            r->fieldSampleCount = 0;
        }
        QVERIFY(digestGlobalResult(a) == digestGlobalResult(b));
        GlobalResult c = result;
        c.prepMs = 0.0;
        c.detectMs = 0.0;
        c.solveMs = 0.0;
        c.geometryMs = 0.0;
        c.fieldMs = 0.0;
        c.fieldSampleCount = 0;
        QVERIFY(digestGlobalResult(a) == digestGlobalResult(c));
    }

    // The witness link adjacent: the orientation turns over (the final
    // sign -1, the stretch corrected against the provisional +z), and the
    // same shelf reads the other way round - Vb below H0, Outside under the
    // plain witness (H outside the V on its inward side), reads Inside -
    // under the four storage orders.
    void adjacentWitnessOrientsTheOtherWay()
    {
        const ShelfFixture plain;
        const ShelfFixture adjacent = ShelfFixture::withAdjacentWitness();
        const auto h0Reading = [&](const GlobalResult& result, uint64_t h0, uint64_t vb) {
            const CrossingEvent* found = nullptr;
            for (const CrossingEvent& ev : result.crossingEvents) {
                if (ev.bent && !ev.curtainFromV && ev.hFiberId == h0 && ev.vFiberId == vb && !ev.tangential) {
                    found = &ev;
                }
            }
            return found;
        };
        const auto h0Decision = [&](const GlobalResult& result, uint64_t h0) {
            const StretchDecisionRecord* found = nullptr;
            for (const StretchDecisionRecord& d : result.stretchDecisions) {
                if (d.fiberId == h0) {
                    found = &d;
                }
            }
            return found;
        };
        const GlobalResult plainResult = buildGlobalLayout(plain.fibers, plain.umbilicus, plain.params, nullptr,
                                                           &plain.field, std::string());
        const StretchDecisionRecord* plainDecision = h0Decision(plainResult, plain.h0);
        QVERIFY(plainDecision != nullptr);
        QCOMPARE(plainDecision->finalSign, 1);
        QVERIFY(!plainDecision->corrected);
        const CrossingEvent* plainReading = h0Reading(plainResult, plain.h0, plain.vb);
        QVERIFY(plainReading != nullptr);
        QCOMPARE(plainReading->kind, CrossingKind::Outside);
        bool ok = false;
        const BentSnapshot snapshot = bentReadingsUnderReversals(adjacent.fibers, adjacent.umbilicus, adjacent.params,
                                                                 adjacent.field, adjacent.h0, adjacent.vb, &ok);
        QVERIFY(ok);
        const GlobalResult adjacentResult = buildGlobalLayout(adjacent.fibers, adjacent.umbilicus, adjacent.params,
                                                              nullptr, &adjacent.field, std::string());
        const StretchDecisionRecord* adjacentDecision = h0Decision(adjacentResult, adjacent.h0);
        QVERIFY(adjacentDecision != nullptr);
        QCOMPARE(adjacentDecision->finalSign, -1);
        QVERIFY(adjacentDecision->corrected);
        const CrossingEvent* adjacentReading = h0Reading(adjacentResult, adjacent.h0, adjacent.vb);
        QVERIFY(adjacentReading != nullptr);
        QCOMPARE(adjacentReading->kind, CrossingKind::Inside);
        bool seen = false;
        for (const BentExport& e : snapshot.events) {
            if (!e.fromV && e.h == adjacent.h0 && e.v == adjacent.vb && !e.tangential && !e.touch) {
                QCOMPARE(e.kind, static_cast<int>(CrossingKind::Inside));
                seen = true;
            }
        }
        QVERIFY(seen);
    }

    // Witnesses are local to their section: H0 with two sections (voters
    // +1 below shelf A, -1 in the twist above it), witnessed on A only.
    // Under LinksOnly A's reading of Va stands and B's reading of Vb is
    // withheld for want of a witness; two decisions, one anchored and one
    // not; the same under the four storage orders.
    void witnessesAreLocalToTheirSection()
    {
        const TwoShelfFixture fixture;
        bool ok = false;
        const BentSnapshot snapshot = bentReadingsUnderReversals(fixture.fibers, fixture.umbilicus, fixture.params,
                                                                 fixture.field, fixture.h0, fixture.vb, &ok);
        QVERIFY(ok);
        int vaUsable = 0;
        int vbWithheld = 0;
        for (const BentExport& e : snapshot.events) {
            if (e.fromV || e.h != fixture.h0 || e.tangential || e.touch) {
                continue;
            }
            if (e.v == fixture.va) {
                QVERIFY(!e.withheld);
                ++vaUsable;
            } else if (e.v == fixture.vb) {
                QVERIFY(e.withheld);
                QCOMPARE(e.reason, static_cast<int>(vc3d::fiber_map::winding::BentDisposition::NoWitness));
                ++vbWithheld;
            }
        }
        QVERIFY(vaUsable >= 1);
        QVERIFY(vbWithheld >= 1);
        const GlobalResult result = buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params, nullptr,
                                                      &fixture.field, std::string());
        int anchored = 0;
        int unanchored = 0;
        for (const StretchDecisionRecord& d : result.stretchDecisions) {
            if (d.fiberId != fixture.h0) {
                continue;
            }
            if (d.withheld) {
                ++unanchored;
            } else {
                ++anchored;
                QCOMPARE(d.anchor, 2);
            }
        }
        QCOMPARE(anchored, 1);
        QVERIFY(unanchored >= 1);
    }

    // The field's ill-conditioned samples leave the ordinal placement: an
    // island V above the shelf, crossing nothing, is anchored by radius
    // against H0's samples without a field and unresolved with it (H0 lies
    // wholly on the shelf, every sample ill-conditioned).
    void illConditionedSamplesLeaveTheOrdinalPlacement()
    {
        const ShelfFixture fixture = ShelfFixture::withIsland();
        const GlobalResult fieldless = buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params);
        const GlobalPlacedFiber* byRadius = findFiber(fieldless, fixture.island);
        QVERIFY(byRadius != nullptr);
        QCOMPARE(byRadius->meta.anchor, GlobalAnchor::Radius);
        const GlobalResult withField = buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params, nullptr,
                                                         &fixture.field, std::string());
        const GlobalPlacedFiber* unresolved = findFiber(withField, fixture.island);
        QVERIFY(unresolved != nullptr);
        QCOMPARE(unresolved->meta.anchor, GlobalAnchor::Unresolved);
    }

    // The other exclusion: well-conditioned samples whose final normal
    // points inward (radially inverted). The climbing H0 with the ADJACENT
    // witness is oriented -z on the shelf, so its climbing ends, where the
    // field is radial, carry a final normal of -e_r: inverted. An island
    // beside one of those ends is anchored by radius without a field and
    // unresolved with it.
    void radiallyInvertedSamplesLeaveTheOrdinalPlacement()
    {
        ShelfFixture fixture(true);
        fixture.makeWitnessAdjacent();
        // Beside H0's climbing start (its first samples, out of the band at
        // the start angle 0.2 pi, 650 vx above the band).
        fixture.addIsland(0.2 * M_PI + 20.0 * kStep, fixture.field.z1 + 50.0 + 600.0);
        const GlobalResult fieldless = buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params);
        const GlobalPlacedFiber* byRadius = findFiber(fieldless, fixture.island);
        QVERIFY(byRadius != nullptr);
        QCOMPARE(byRadius->meta.anchor, GlobalAnchor::Radius);
        const GlobalResult withField = buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params, nullptr,
                                                         &fixture.field, std::string());
        QCOMPARE(withField.anchorCorrectedStretchCount, 2);
        const GlobalPlacedFiber* unresolved = findFiber(withField, fixture.island);
        QVERIFY(unresolved != nullptr);
        QCOMPARE(unresolved->meta.anchor, GlobalAnchor::Unresolved);
        // The plain witness leaves the ends' normals outward: the island
        // is anchored against them, field or not.
        ShelfFixture plain(true);
        plain.addIsland(0.2 * M_PI + 20.0 * kStep, plain.field.z1 + 50.0 + 600.0);
        const GlobalResult outward = buildGlobalLayout(plain.fibers, plain.umbilicus, plain.params, nullptr,
                                                       &plain.field, std::string());
        const GlobalPlacedFiber* anchored = findFiber(outward, plain.island);
        QVERIFY(anchored != nullptr);
        QCOMPARE(anchored->meta.anchor, GlobalAnchor::Radius);
    }

    // The input digest names every bent-ray parameter, the crease
    // thresholds included: each changed alone is a different input.
    void inputDigestNamesEveryBentParameter()
    {
        const ShelfFixture fixture;
        const std::string field = fixture.field.identity();
        const ContentDigest base = digestGlobalInputs(fixture.fibers, fixture.umbilicus, fixture.params, field);
        const auto mutated = [&](auto mutate) {
            GlobalLayoutParams params = fixture.params;
            mutate(params.bentRays);
            return digestGlobalInputs(fixture.fibers, fixture.umbilicus, params, field);
        };
        QVERIFY(!(mutated([](vc3d::fiber_map::bent::BentRayParams& p) { p.creaseTurnDeg = 0.0; }) == base));
        QVERIFY(!(mutated([](vc3d::fiber_map::bent::BentRayParams& p) { p.creaseTurnDeg = 45.0; }) == base));
        QVERIFY(!(mutated([](vc3d::fiber_map::bent::BentRayParams& p) { p.creaseConditioning = 0.1; }) == base));
        QVERIFY(!(mutated([](vc3d::fiber_map::bent::BentRayParams& p) { p.stepVx *= 0.5; }) == base));
        QVERIFY(!(mutated([](vc3d::fiber_map::bent::BentRayParams& p) { p.spacingVx *= 0.5; }) == base));
        QVERIFY(!(mutated([](vc3d::fiber_map::bent::BentRayParams& p) { p.maxLengthVx *= 0.5; }) == base));
        QVERIFY(!(mutated([](vc3d::fiber_map::bent::BentRayParams& p) { p.conditioningGate *= 0.5; }) == base));
        QVERIFY(!(mutated([](vc3d::fiber_map::bent::BentRayParams& p) {
            p.anchorPolicy = vc3d::fiber_map::bent::AnchorPolicy::UmbilicusOrLinks;
        }) == base));
        QVERIFY(mutated([](vc3d::fiber_map::bent::BentRayParams&) {}) == base);
        // Without a field none of them enters the digest.
        const ContentDigest fieldless = digestGlobalInputs(fixture.fibers, fixture.umbilicus, fixture.params);
        GlobalLayoutParams params = fixture.params;
        params.bentRays.creaseTurnDeg = 0.0;
        QVERIFY(digestGlobalInputs(fixture.fibers, fixture.umbilicus, params) == fieldless);
    }

    // Pair coverage names the new withholding reasons: a pair whose only
    // readings are withheld as straight-disagreeing (the shelf's return
    // limb crossing Vb's climb straight on, lower than the bent readings
    // would have it) reports "straightDisagrees"; one whose only
    // readings bent through a crease reports "creaseCrossed".
    void pairCoverageNamesTheNewReasons()
    {
        {
            ShelfFixture fixture;
            fixture.addReturnLimbAboveTheBand();
            const GlobalResult result = buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params, nullptr,
                                                          &fixture.field, std::string());
            bool found = false;
            for (const PairCoverageRecord& record : result.pairCoverage) {
                if (record.hFiberId == fixture.h0 && record.vFiberId == fixture.vb) {
                    QCOMPARE(record.reason, std::string("straightDisagrees"));
                    QCOMPARE(record.bentCount, 0);
                    QVERIFY(record.withheldCount >= 1);
                    found = true;
                }
            }
            QVERIFY(found);
            int disagreeing = 0;
            int straightKept = 0;
            for (const CrossingEvent& ev : result.crossingEvents) {
                if (ev.hFiberId != fixture.h0 || ev.vFiberId != fixture.vb || ev.tangential || ev.touch) {
                    continue;
                }
                if (ev.bent) {
                    QVERIFY(ev.withheld);
                    QCOMPARE(ev.bentReason, static_cast<int>(vc3d::fiber_map::winding::BentDisposition::StraightDisagrees));
                    ++disagreeing;
                } else {
                    QCOMPARE(ev.kind, CrossingKind::Inside);
                    ++straightKept;
                }
            }
            QVERIFY(disagreeing >= 1);
            QCOMPARE(straightKept, 1);
            // Without the return limb the same shelf reading replaces the
            // set-aside straight one (the fixture's base behaviour).
            const ShelfFixture plain;
            const GlobalResult base = buildGlobalLayout(plain.fibers, plain.umbilicus, plain.params, nullptr,
                                                        &plain.field, std::string());
            QCOMPARE(base.pairCoverage.size(), std::size_t{1});
            QCOMPARE(base.pairCoverage.front().reason, std::string("replaced"));
        }
        {
            const CreaseFixture fixture;
            const GlobalResult result = buildGlobalLayout(fixture.fibers, fixture.umbilicus, fixture.params, nullptr,
                                                          &fixture.field, std::string());
            bool found = false;
            for (const PairCoverageRecord& record : result.pairCoverage) {
                if (record.hFiberId == fixture.h0 && record.vFiberId == fixture.vb) {
                    QCOMPARE(record.reason, std::string("creaseCrossed"));
                    found = true;
                }
            }
            QVERIFY(found);
            int creased = 0;
            for (const CrossingEvent& ev : result.crossingEvents) {
                if (ev.bent && !ev.curtainFromV && ev.vFiberId == fixture.vb && !ev.tangential && !ev.touch) {
                    QVERIFY(ev.withheld);
                    QCOMPARE(ev.bentReason, static_cast<int>(vc3d::fiber_map::winding::BentDisposition::CreaseCrossed));
                    QVERIFY(ev.turnDeg > 60.0);
                    QVERIFY(ev.minConditioning < 0.05);
                    ++creased;
                }
            }
            QVERIFY(creased >= 1);
            // Disabled, the same readings constrain.
            GlobalLayoutParams params = fixture.params;
            params.bentRays.creaseTurnDeg = 0.0;
            const GlobalResult open = buildGlobalLayout(fixture.fibers, fixture.umbilicus, params, nullptr,
                                                        &fixture.field, std::string());
            QVERIFY(open.bentCrossingCount >= 1);
        }
    }

    // The umbilicus cutoff on the layout's own frame (specialist S5): a hit
    // inside the cutoff is not recorded although both rays of its strip
    // clear it (the coarse arc's chord sags inside); lowering the cutoff
    // records it. And a ray step through an umbilicus knot where the
    // centre swings out is rejected although both step ends clear the
    // cutoff: the crossing beyond it is not read.
    void hitsInsideTheCutoffAreNotRecorded()
    {
        CutoffFixture coarse(true);
        QCOMPARE(coarse.rawHitsOfVb(coarse.umbilicus), std::size_t{0});
        coarse.params.solver.minUmbilicusRadiusVx = 400.0;
        QVERIFY(coarse.rawHitsOfVb(coarse.umbilicus) >= 1);
    }

    void stepsThroughAnUmbilicusKnotAreRejected()
    {
        const CutoffFixture fine(false);
        QVERIFY(fine.rawHitsOfVb(fine.umbilicus) >= 1);
        // The centre swings 1 vx toward the arc at z 30206 (a tent over
        // 30202..30210): the first step up from the arc passes radius
        // 416.5 at the knot.
        std::vector<cv::Vec3f> tent = fine.umbilicus;
        const double zk = 30206.0;
        for (cv::Vec3f& c : tent) {
            const double d = std::abs(static_cast<double>(c[2]) - zk);
            if (d < 4.0) {
                const double x = 1.0 - d / 4.0;
                c[0] = static_cast<float>(x * std::cos(fine.crossAngle));
                c[1] = static_cast<float>(x * std::sin(fine.crossAngle));
            }
        }
        QCOMPARE(fine.rawHitsOfVb(tent), std::size_t{0});
    }

    // The result digest ties a bent payload to its MARK too: two marks alike
    // in every legacy field, the bent payload moved from one to the other,
    // is a different digest.
    void digestTiesBentPayloadToItsMark()
    {
        ShelfFixture fixture;
        std::vector<InputFiber> conflicted = fixture.fibers;
        for (InputFiber& fiber : conflicted) {
            if (fiber.id == fixture.h0) {
                for (InputFiber& other : conflicted) {
                    if (other.id == fixture.h1) {
                        addLink(fiber, 2, other, 1);
                    }
                }
            }
        }
        GlobalResult base = buildGlobalLayout(conflicted, fixture.umbilicus, fixture.params, nullptr,
                                              &fixture.field, std::string());
        std::size_t bentMark = base.suspectCrossings.size();
        for (std::size_t i = 0; i < base.suspectCrossings.size(); ++i) {
            if (base.suspectCrossings[i].bent) {
                bentMark = i;
                break;
            }
        }
        QVERIFY(bentMark < base.suspectCrossings.size());
        CrossingMark straight = base.suspectCrossings[bentMark];
        straight.bent = false;
        straight.hitVx = cv::Vec3d(0.0, 0.0, 0.0);
        base.suspectCrossings.push_back(straight);
        GlobalResult moved = base;
        CrossingMark& from = moved.suspectCrossings[bentMark];
        CrossingMark& to = moved.suspectCrossings.back();
        std::swap(from.bent, to.bent);
        std::swap(from.hitVx, to.hitVx);
        QVERIFY(!(digestGlobalResult(moved) == digestGlobalResult(base)));
    }

};

QTEST_APPLESS_MAIN(TestFiberGlobalLayout)
#include "test_fiber_global_layout.moc"
