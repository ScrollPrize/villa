// Coverage for apps/VC3D/FiberWindingSolver.{hpp,cpp} - the winding assignment
// behind the all-fibers Fiber Map.
//
// Every fixture lives on a deliberately crumpled spiral: the sheet radius is
// modulated in angle and z, so the winding-to-winding spacing is non-uniform
// everywhere and nothing below can pass by assuming a pitch. What survives the
// crumpling, and what the solver is allowed to use, is that windings stay
// radially ordered along any ray and that same-winding H/V contacts sit a
// sheet thickness apart.
//
// Traces are authored directly in (theta, r, z): theta = 2*pi*(w - m) for a
// point at continuous spiral coordinate w, where m plays the role of the
// arbitrary whole-turn gauge that atan2 unwrapping would leave. The ground
// truth of every fixture is therefore its m values: the solver's turn offsets
// must reproduce their differences exactly.

#include <QtTest/QtTest>

#include <cmath>
#include <limits>
#include <vector>

#include "FiberWindingSolver.hpp"

using vc3d::fiber_map::winding::ComponentAnchor;
using vc3d::fiber_map::winding::Crossing;
using vc3d::fiber_map::winding::CrossingGroup;
using vc3d::fiber_map::winding::CrossingKind;
using vc3d::fiber_map::winding::CrossingStatus;
using vc3d::fiber_map::winding::FiberTrace;
using vc3d::fiber_map::winding::LinkInput;
using vc3d::fiber_map::winding::SolveResult;
using vc3d::fiber_map::winding::SolverParams;
using vc3d::fiber_map::winding::solveWindings;
using vc3d::fiber_map::winding::CanonicalTrace;
using vc3d::fiber_map::winding::PairCrossings;
using vc3d::fiber_map::winding::PairDetections;
using vc3d::fiber_map::winding::canonicalizeTrace;
using vc3d::fiber_map::winding::classifyPairCrossings;
using vc3d::fiber_map::winding::detectPairCrossings;
using vc3d::fiber_map::winding::inferChirality;

namespace
{

constexpr double kTwoPi = 2.0 * M_PI;
// Same-winding H fibers sit a sub-tie-band step inside their sheet's V fibers
// (H on the front of the sheet, V behind).
constexpr double kSheetStep = 100.0;

// The crumpled sheet: radius of the spiral surface at continuous winding
// coordinate w and height z. Monotone in w along any ray (nesting), spacing
// anything but uniform.
double sheetR(double w, double z)
{
    const double phi = kTwoPi * w;
    return (20000.0 + 2000.0 * w) *
           (1.0 + 0.15 * std::sin(phi) + 0.08 * std::sin(z / 2500.0));
}

struct World {
    std::vector<FiberTrace> fibers;
    std::vector<LinkInput> links;
    // The whole-turn gauge of each fiber; the ground truth.
    std::vector<long long> trueM;
};

// A V fiber on winding coordinate w (constant angle), spanning [z0, z1].
// radiusOffset models annotation error. Returns the fiber index.
std::size_t addV(World& world, double w, double z0, double z1,
                 double radiusOffset = 0.0)
{
    const long long m = static_cast<long long>(std::floor(w));
    FiberTrace fiber;
    fiber.hvTag = 'V';
    for (double z = z0; z <= z1 + 1e-9; z += 25.0) {
        fiber.theta.push_back(kTwoPi * (w - static_cast<double>(m)));
        fiber.z.push_back(z);
        fiber.radius.push_back(sheetR(w, z) + radiusOffset);
    }
    world.fibers.push_back(std::move(fiber));
    world.trueM.push_back(m);
    return world.fibers.size() - 1;
}

// An H fiber running along the sheet from w0 to w1 at height z, a sheet step
// inside the surface. radiusAt overrides the sheet-following radius.
std::size_t addH(World& world, double w0, double w1, double z,
                 double (*radiusAt)(double w, double z) = nullptr)
{
    const long long m = static_cast<long long>(std::floor(w0));
    FiberTrace fiber;
    fiber.hvTag = 'H';
    for (double w = w0; w <= w1 + 1e-9; w += 1.0 / 256.0) {
        fiber.theta.push_back(kTwoPi * (w - static_cast<double>(m)));
        fiber.z.push_back(z);
        fiber.radius.push_back(radiusAt != nullptr ? radiusAt(w, z)
                                                   : sheetR(w, z) - kSheetStep);
    }
    world.fibers.push_back(std::move(fiber));
    world.trueM.push_back(m);
    return world.fibers.size() - 1;
}

World mirrored(const World& world)
{
    World out = world;
    for (FiberTrace& fiber : out.fibers) {
        for (double& theta : fiber.theta) {
            theta = -theta;
        }
    }
    return out;
}

// Solved winding coordinate of one sample.
double solvedW(const SolveResult& result, const World& world, std::size_t fiber,
               std::size_t sample)
{
    return result.chirality * world.fibers[fiber].theta[sample] / kTwoPi +
           result.placements[fiber].turns;
}

// The solver's turn offsets must reproduce the gauge differences exactly for
// every pair that shares a component.
void checkRelativeTurns(const SolveResult& result, const World& world,
                        const std::vector<std::size_t>& fibers)
{
    for (std::size_t i = 1; i < fibers.size(); ++i) {
        const std::size_t a = fibers[0];
        const std::size_t b = fibers[i];
        QCOMPARE(result.placements[b].turns - result.placements[a].turns,
                 static_cast<double>(world.trueM[b] - world.trueM[a]));
    }
}

// Sample index of the H fiber nearest a spiral coordinate w (H fixtures step w
// uniformly from their start).
std::size_t hSampleAt(const World& world, std::size_t fiber, double wStart, double w)
{
    const double step = 1.0 / 256.0;
    const auto index = static_cast<long long>(std::llround((w - wStart) / step));
    Q_ASSERT(index >= 0 &&
             index < static_cast<long long>(world.fibers[fiber].theta.size()));
    return static_cast<std::size_t>(index);
}

int countDroppedCrossings(const SolveResult& result)
{
    int dropped = 0;
    for (const Crossing& crossing : result.crossings) {
        if (crossing.status == CrossingStatus::Dropped) {
            ++dropped;
        }
    }
    return dropped;
}


// --- Folded sheets, for the traversal-group tests.
//
// The dented sheet of the geometry review: radius R + b*(u+2)^2 and angle
// theta0 + eps*(u^3 - 3u), u in [-3, 3]. The angle runs forward, turns back
// over |u| < 1 (a dent toward the umbilicus) and runs forward again, so the
// ray at theta0 meets the sheet three times: u = -sqrt3, 0, +sqrt3, at radii
// R + 0.07b, R + 4b, R + 13.9b.
constexpr double kHairpinR = 20000.0;
constexpr double kHairpinTheta0 = kTwoPi * 0.3;
constexpr double kSqrt3 = 1.7320508075688772;

double hairpinRadius(double u, double b)
{
    return kHairpinR + b * (u + 2.0) * (u + 2.0);
}

double hairpinTheta(double u, double eps)
{
    return kHairpinTheta0 + eps * (u * u * u - 3.0 * u);
}

// An H fiber along the dented sheet at height z, from u0 to u1, its radius
// offset from the sheet by faceOffset (negative = inner face, as the sheet
// model puts H fibers).
std::size_t addHairpinH(World& world, double z, double u0, double u1, double b,
                        double eps, double faceOffset)
{
    FiberTrace fiber;
    fiber.hvTag = 'H';
    for (double u = u0; u <= u1 + 1e-9; u += 0.01) {
        fiber.theta.push_back(hairpinTheta(u, eps));
        fiber.z.push_back(z);
        fiber.radius.push_back(hairpinRadius(u, b) + faceOffset);
    }
    world.fibers.push_back(std::move(fiber));
    world.trueM.push_back(0);
    return world.fibers.size() - 1;
}

// A V fiber on the ray theta0 at a fixed radius, spanning [z0, z1].
std::size_t addRayV(World& world, double radius, double z0, double z1, long long trueM)
{
    FiberTrace fiber;
    fiber.hvTag = 'V';
    for (double z = z0; z <= z1 + 1e-9; z += 25.0) {
        fiber.theta.push_back(kHairpinTheta0);
        fiber.z.push_back(z);
        fiber.radius.push_back(radius);
    }
    world.fibers.push_back(std::move(fiber));
    world.trueM.push_back(trueM);
    return world.fibers.size() - 1;
}

// --- Kollesis fixtures. The inner sheet's H fiber ends just past the seam
// angle with its end tagged; the outer sheet's V fiber runs at the seam one
// thickness in FRONT of the H fiber's end (the outer sheet is glued toward
// the core), so the crossing reads Outside by a thickness although the two
// are one winding. The H fiber runs a little past the V fiber, as annotated
// ends do: the encounter is interior to the trace, not a vertex coincidence.
constexpr double kSeamWinding = 0.55;
constexpr double kSeamOvershoot = 0.02;
constexpr double kSeamHeight = 30000.0;

// An H fiber whose earlier turn sits well outside the seam V fiber: the
// seam-encounter reading must not spill over to that crossing.
double outerOnEarlierTurn(double w, double z)
{
    return w < kSeamWinding - 0.5 ? sheetR(w + 1.0, z) + 300.0 : sheetR(w, z) - kSheetStep;
}

// The one group of a solve, or fails the test.
const CrossingGroup& singleGroup(const SolveResult& result)
{
    static const CrossingGroup none;
    if (result.groups.size() != 1) {
        return none;
    }
    return result.groups.front();
}

// The standard three-winding weave: one H fiber spiralling through three
// turns, one V fiber per winding at the same angle, every crossing kind
// represented (tie on the shared winding, inside below, outside above).
World threeWindingWorld()
{
    World world;
    addH(world, 0.05, 2.9, 30000.0);
    addV(world, 0.3, 27000.0, 33000.0);
    addV(world, 1.3, 27000.0, 33000.0);
    addV(world, 2.3, 27000.0, 33000.0);
    return world;
}

} // namespace

class TestFiberWindingSolver : public QObject
{
    Q_OBJECT

private slots:
    void exactRecoveryAcrossThreeWindings()
    {
        const World world = threeWindingWorld();
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.chirality, 1);
        checkRelativeTurns(result, world, {0, 1, 2, 3});
        for (const auto& placement : result.placements) {
            QCOMPARE(placement.anchor, ComponentAnchor::Primary);
            QVERIFY(!placement.sheetDriftSuspect);
        }
        // No tie classification exists: same-winding contacts arrive as weak
        // inside constraints and the densest solve collapses them to
        // equality. No repairs.
        QCOMPARE(result.tieCount, 0);
        QCOMPARE(result.droppedCrossingCount, 0);
        QVERIFY(result.droppedLinks.empty());
        // Gauge: the primary component's innermost winding is zero.
        double minW = std::numeric_limits<double>::infinity();
        for (std::size_t f = 0; f < world.fibers.size(); ++f) {
            minW = std::min(minW, result.placements[f].windingLo);
        }
        QVERIFY(minW >= 0.0);
        QVERIFY(minW < 1.0);
        // The H fiber's winding range spans its three turns.
        QVERIFY(result.placements[0].windingHi -
                    result.placements[0].windingLo > 2.5);
    }

    void mirroredChiralityRecoversTheSameMap()
    {
        const World world = mirrored(threeWindingWorld());
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.chirality, -1);
        checkRelativeTurns(result, world, {0, 1, 2, 3});
        QCOMPARE(result.tieCount, 0);
        QCOMPARE(result.droppedCrossingCount, 0);
    }

    void sameWindingPairCollapsesToEquality()
    {
        World world;
        const std::size_t h = addH(world, 0.05, 0.55, 30000.0);
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.crossings.size(), std::size_t{1});
        // The sheet contact reads as a weak inside; the densest solve
        // collapses the weak constraint to equality.
        QCOMPARE(result.crossings.front().kind, CrossingKind::Inside);
        const double wh =
            solvedW(result, world, h, hSampleAt(world, h, 0.05, 0.3));
        const double wv = solvedW(result, world, v, 0);
        QVERIFY2(std::abs(wh - wv) < 0.01,
                 qPrintable(QStringLiteral("wh %1 wv %2").arg(wh).arg(wv)));
    }

    // The accepted cost of having no tie band: a same-winding contact whose
    // radial noise flips the sign of deltaR reads as a strict outside and
    // separates the pair by one winding. Documented behavior, not a bug.
    void signFlippedContactSeparatesOneWinding()
    {
        World world;
        const std::size_t h = addH(world, 0.05, 0.55, 30000.0);
        // V annotated 180 vx inside the sheet, so the same-winding H fiber
        // (100 vx inside) measures 80 vx OUTSIDE it.
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0, -180.0);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.crossings.size(), std::size_t{1});
        QVERIFY(result.crossings.front().deltaR > 0.0);
        QCOMPARE(result.crossings.front().kind, CrossingKind::Outside);
        QCOMPARE(result.droppedCrossingCount, 0);
        const double wh =
            solvedW(result, world, h, hSampleAt(world, h, 0.05, 0.3));
        QVERIFY(std::abs(wh - (solvedW(result, world, v, 0) + 1.0)) < 0.01);
    }

    // Inside-then-outside on successive turns pins the V fiber to the last
    // inside section with no special-casing: the weak and strict constraints
    // meet at equality. The V fiber is annotated off the sheet by more than
    // the band so no tie takes part.
    void insideThenOutsidePinsTheLink()
    {
        World world;
        const std::size_t h = addH(world, 0.05, 2.9, 30000.0);
        const std::size_t v = addV(world, 1.3, 27000.0, 33000.0, 300.0);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.tieCount, 0);
        QCOMPARE(result.droppedCrossingCount, 0);
        checkRelativeTurns(result, world, {h, v});
        const double wh =
            solvedW(result, world, h, hSampleAt(world, h, 0.05, 1.3));
        const double wv = solvedW(result, world, v, 0);
        QVERIFY2(std::abs(wh - wv) < 0.01,
                 qPrintable(QStringLiteral("wh %1 wv %2").arg(wh).arg(wv)));
    }

    // H fibers linked into a chain act as one long H fiber: one member passes
    // the V fiber inside, the other outside, and the pair pins it.
    void linkedChainPinsTheVFiber()
    {
        World world;
        const std::size_t h1 = addH(world, 0.05, 1.05, 30000.0);
        const std::size_t h2 = addH(world, 1.05, 2.55, 30000.0);
        const std::size_t v = addV(world, 1.5, 27000.0, 33000.0, 300.0);
        world.links.push_back(LinkInput{
            h1, world.fibers[h1].theta.size() - 1, h2, 0});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QVERIFY(result.droppedLinks.empty());
        QCOMPARE(result.droppedCrossingCount, 0);
        checkRelativeTurns(result, world, {h1, h2, v});
        QVERIFY(result.placements[h1].linked);
        QVERIFY(result.placements[h2].linked);
        const double wh =
            solvedW(result, world, h2, hSampleAt(world, h2, 1.05, 1.5));
        QVERIFY(std::abs(wh - solvedW(result, world, v, 0)) < 0.01);
    }

    // The densest collapse and its ordinal correction. A second V fiber far
    // outside is only reachable through a weak inside constraint, which the
    // solve first collapses onto the H fiber; local radial ordering then
    // spreads it to the nearest consistent winding - one out, never the true
    // three, because missing windings are not guessed.
    void missingWindingsCollapseToTheDensestMap()
    {
        World world;
        const std::size_t h = addH(world, 1.05, 1.6, 30000.0);
        const std::size_t v0 = addV(world, 0.3, 27000.0, 33000.0);
        const std::size_t vFar = addV(world, 4.3, 27000.0, 33000.0);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        const double wh =
            solvedW(result, world, h, hSampleAt(world, h, 1.05, 1.3));
        const double w0 = solvedW(result, world, v0, 0);
        const double wFar = solvedW(result, world, vFar, 0);
        // Strict outside: exactly one winding out (densest).
        QVERIFY2(std::abs(wh - w0 - 1.0) < 0.01,
                 qPrintable(QStringLiteral("wh %1 w0 %2").arg(wh).arg(w0)));
        // Weak inside plus radial ordering: one winding out, not three.
        QVERIFY2(std::abs(wFar - wh - 1.0) < 0.01,
                 qPrintable(QStringLiteral("wFar %1 wh %2").arg(wFar).arg(wh)));
    }

    // A link wrong by exactly one turn carries a clean residual, so it cannot
    // be outranked by confidence alone; it loses by sitting in every conflict
    // cycle while fresh correct evidence keeps arriving.
    void aWrongLinkLosesToSeveralCorrectCrossings()
    {
        World world = threeWindingWorld();
        addH(world, 0.05, 2.9, 30500.0);
        // Claims the H fiber's first-winding pass IS the second V fiber: one
        // whole winding wrong, residual zero.
        world.links.push_back(LinkInput{
            0, hSampleAt(world, 0, 0.05, 0.3), 2, 0});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.droppedLinks, std::vector<std::size_t>{0});
        checkRelativeTurns(result, world, {0, 1, 2, 3, 4});
    }

    // An H fiber whose annotation drifts across sheets contradicts itself at
    // successive passes of the same V fibers. The repairs are deterministic
    // and the fiber is reported as the drift suspect.
    void sheetDriftIsDetectedAndReported()
    {
        World world;
        // Passes both V fibers OUTSIDE on its first turn, then regresses two
        // sheets inward and passes them INSIDE on the second: inward
        // regression, the one self-contradiction the one-sided constraint
        // forms cannot absorb. (Outward drift is absorbable by design: the
        // weak inside and at-least-one outside both leave outward room.)
        const auto regressing = [](double w, double z) {
            return w < 1.8 ? sheetR(w, z) + 200.0 : sheetR(w - 2.0, z) - 200.0;
        };
        const std::size_t h = addH(world, 1.2, 2.4, 30000.0, regressing);
        addV(world, 1.3, 27000.0, 33000.0);
        addV(world, 1.35, 27000.0, 33000.0);
        // The regressing fiber defeats data-driven chirality on purpose.
        SolverParams params;
        params.chiralityOverride = 1;
        const SolveResult first =
            solveWindings(world.fibers, world.links, params);
        const SolveResult second =
            solveWindings(world.fibers, world.links, params);
        QCOMPARE(countDroppedCrossings(first), 2);
        QVERIFY(first.placements[h].sheetDriftSuspect);
        // Determinism: same drops, same turns.
        QCOMPARE(countDroppedCrossings(second), 2);
        for (std::size_t f = 0; f < world.fibers.size(); ++f) {
            QCOMPARE(second.placements[f].turns, first.placements[f].turns);
        }
        for (std::size_t c = 0; c < first.crossings.size(); ++c) {
            QVERIFY(second.crossings[c].status == first.crossings[c].status);
        }
    }

    // An untrusted fiber (no model-traced span) must lose a repair conflict,
    // H sits well inside trusted V1 (weak inside, conf 0.9) and well outside
    // untrusted V2 (strict, raw conf 1.0 attenuated to 0.5); the V1=V2 link
    // closes the cycle and the attenuated outside is the deterministic victim.
    void untrustedEvidenceLosesConflicts()
    {
        World world;
        const std::size_t h = addH(world, 0.05, 0.55, 30000.0);
        // Well outside H: a strong (high-margin) inside constraint.
        const std::size_t v1 = addV(world, 0.3, 27000.0, 33000.0, 400.0);
        // Untrusted, drawn far inside the sheet: a strong outside claim that
        // contradicts the link below - and would beat the inside constraint
        // on raw confidence if not for the attenuation.
        const std::size_t v2 = addV(world, 0.3, 27000.0, 33000.0, -600.0);
        world.fibers[v2].trusted = false;
        world.links.push_back(LinkInput{v1, 0, v2, 0});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(countDroppedCrossings(result), 1);
        QVERIFY(result.droppedLinks.empty());
        for (const Crossing& crossing : result.crossings) {
            if (crossing.status == CrossingStatus::Dropped) {
                QCOMPARE(crossing.kind, CrossingKind::Outside);
            }
        }
        // The surviving inside and the link keep all three together.
        QCOMPARE(result.placements[v1].turns - result.placements[h].turns, 0.0);
        QCOMPARE(result.placements[v2].turns - result.placements[v1].turns, 0.0);
        // And no drift declaration: one contested traversal is one piece of
        // evidence, whoever it involves.
        QVERIFY(!result.placements[h].sheetDriftSuspect);
    }

    // Greedy repair can drop an innocent constraint before the real culprit
    // falls in a later cycle, leaving a drop the final map satisfies anyway.
    // Such repair debris carries violationTurns ~0 and is not declared; the
    // culprit's drop is violated by a winding or more. Fixture: H passes A
    // outside on its first turn (thin margin - the innocent) and inside on
    // its second (strong margin - the culprit: inward regression), while a
    // linked helper H2 independently passes A outside with a strong margin.
    // The thin outside falls first as the {out, in} cycle's weakest edge;
    // the culprit falls to the {in, link, out2} cycle; the sacrificed
    // outside ends up satisfied by the very map that dropped it.
    void satisfiedDropsAreRepairDebrisNotErrors()
    {
        World world;
        const auto regressing = [](double w, double z) {
            return w < 1.0 ? sheetR(0.3, z) + 200.0
                           : sheetR(0.3, z) - 300.0;
        };
        const std::size_t h = addH(world, 0.05, 1.7, 30000.0, regressing);
        const auto wellOutside = [](double w, double z) {
            return sheetR(0.3, z) + 600.0;
        };
        const std::size_t h2 = addH(world, 0.05, 0.55, 31000.0, wellOutside);
        const std::size_t a = addV(world, 0.3, 27000.0, 33000.0);
        // Same angle, same start: sample 13 of both is the same ray.
        world.links.push_back(LinkInput{h, 13, h2, 13});
        SolverParams params;
        params.chiralityOverride = 1;  // the flat radii defeat inference
        const SolveResult result = solveWindings(world.fibers, world.links, params);
        // Everything settles on the first-turn relation: A one winding
        // inside the linked H pair.
        QCOMPARE(result.placements[h2].turns, result.placements[h].turns);
        QVERIFY(result.droppedLinks.empty());
        int satisfiedDrops = 0;
        int violatedDrops = 0;
        for (const Crossing& crossing : result.crossings) {
            if (crossing.status != CrossingStatus::Dropped) {
                continue;
            }
            if (crossing.violationTurns >= 0.5) {
                ++violatedDrops;
            } else {
                ++satisfiedDrops;
                QVERIFY(crossing.violationTurns < 0.1);
            }
        }
        QCOMPARE(satisfiedDrops, 1);
        QCOMPARE(violatedDrops, 1);
        // One violated drop is one piece of distinct evidence: no drift tag.
        QVERIFY(!result.placements[h].sheetDriftSuspect);
    }

    // Declarations are not gated on trust: the same contradictions raised by
    // an untrusted (interpolated) H fiber are reported like any other's. Its
    // evidence is attenuated uniformly, so the repair falls the same way.
    void untrustedFibersAreDriftSuspectsLikeAnyOther()
    {
        World world;
        const auto regressing = [](double w, double z) {
            return w < 1.8 ? sheetR(w, z) + 200.0 : sheetR(w - 2.0, z) - 200.0;
        };
        const std::size_t h = addH(world, 1.2, 2.4, 30000.0, regressing);
        world.fibers[h].trusted = false;
        addV(world, 1.3, 27000.0, 33000.0);
        addV(world, 1.35, 27000.0, 33000.0);
        SolverParams params;
        params.chiralityOverride = 1;
        const SolveResult result =
            solveWindings(world.fibers, world.links, params);
        QVERIFY(countDroppedCrossings(result) > 0);
        QVERIFY(result.placements[h].sheetDriftSuspect);
    }

    // The common-lift translate search: gauges five turns apart still meet.
    void seamAndGaugeTranslatesAreSearched()
    {
        World world;
        World gaugeShifted;
        const std::size_t h = addH(world, 5.2, 5.7, 30000.0);
        addV(world, 5.45, 27000.0, 33000.0);
        // Rewrite the H fiber's gauge to m = 0: theta = 2*pi*w, five turns
        // above the V fiber's own lift.
        gaugeShifted = world;
        for (double& theta : gaugeShifted.fibers[h].theta) {
            theta += kTwoPi * 5.0;
        }
        gaugeShifted.trueM[h] = 0;
        // No fiber here wraps a full turn, so the data cannot reveal the
        // chirality; the fixture pins it and tests only the translate search.
        SolverParams params;
        params.chiralityOverride = 1;
        const SolveResult result = solveWindings(gaugeShifted.fibers,
                                                 gaugeShifted.links, params);
        QCOMPARE(result.crossings.size(), std::size_t{1});
        QCOMPARE(result.crossings.front().kind, CrossingKind::Inside);
        // The canonical gauge absorbs the five-turn input gauge before
        // detection, so the recorded gap is small; the output turn offsets
        // compensate, which checkRelativeTurns verifies.
        QCOMPARE(result.crossings.front().n, static_cast<long long>(0));
        checkRelativeTurns(result, gaugeShifted, {0, 1});
    }

    // The whole solve must be invariant to each fiber's arbitrary unwrap
    // branch: re-gauging fibers by whole turns changes nothing but the
    // compensating turn offsets.
    void solveIsGaugeInvariant()
    {
        const World base = threeWindingWorld();
        World shifted = base;
        for (double& theta : shifted.fibers[0].theta) {
            theta += kTwoPi * 7.0;
        }
        shifted.trueM[0] -= 7;
        for (double& theta : shifted.fibers[3].theta) {
            theta -= kTwoPi * 3.0;
        }
        shifted.trueM[3] += 3;
        const SolveResult a = solveWindings(base.fibers, base.links, SolverParams{});
        const SolveResult b =
            solveWindings(shifted.fibers, shifted.links, SolverParams{});
        QCOMPARE(b.tieCount, a.tieCount);
        QCOMPARE(b.droppedCrossingCount, a.droppedCrossingCount);
        checkRelativeTurns(b, shifted, {0, 1, 2, 3});
        // Identical physical windings: the turn offsets differ by exactly the
        // injected gauges.
        QCOMPARE(b.placements[0].turns, a.placements[0].turns - 7.0);
        QCOMPARE(b.placements[3].turns, a.placements[3].turns + 3.0);
        for (std::size_t f = 0; f < base.fibers.size(); ++f) {
            QCOMPARE(b.placements[f].windingLo, a.placements[f].windingLo);
            QCOMPARE(b.placements[f].windingHi, a.placements[f].windingHi);
        }

        // And across the half-turn median boundary, where a rounding that is
        // not translation-equivariant would slip a whole turn: a fiber whose
        // median angle sits exactly at half a turn.
        World boundary;
        addH(boundary, 0.0, 1.0, 30000.0);
        addV(boundary, 0.5, 27000.0, 33000.0);
        World boundaryShifted = boundary;
        for (double& theta : boundaryShifted.fibers[0].theta) {
            theta += kTwoPi;
        }
        boundaryShifted.trueM[0] -= 1;
        const SolveResult c =
            solveWindings(boundary.fibers, boundary.links, SolverParams{});
        const SolveResult d = solveWindings(boundaryShifted.fibers,
                                            boundaryShifted.links, SolverParams{});
        QCOMPARE(d.placements[0].turns, c.placements[0].turns - 1.0);
        QCOMPARE(d.placements[1].turns, c.placements[1].turns);
        for (std::size_t f = 0; f < boundary.fibers.size(); ++f) {
            QCOMPARE(d.placements[f].windingLo, c.placements[f].windingLo);
        }
    }

    // A crossing landing exactly on a V fiber's z apex (the reversed end of
    // both monotone branches) is owned by the branches' final segments and
    // merged into one piece of evidence, not lost twice.
    void apexCrossingsAreOwned()
    {
        World world;
        const std::size_t h = addH(world, 1.2, 1.4, 31000.0);
        FiberTrace u;
        u.hvTag = 'V';
        for (double z = 29000.0; z <= 31000.0 + 1e-9; z += 25.0) {
            u.theta.push_back(kTwoPi * 0.3);
            u.z.push_back(z);
            u.radius.push_back(sheetR(1.3, z));
        }
        for (double z = 31000.0 - 25.0; z >= 29500.0; z -= 25.0) {
            u.theta.push_back(kTwoPi * 0.3);
            u.z.push_back(z);
            u.radius.push_back(sheetR(1.3, z));
        }
        world.fibers.push_back(std::move(u));
        world.trueM.push_back(1);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.crossings.size(), std::size_t{1});
        QCOMPARE(result.crossings.front().kind, CrossingKind::Inside);
        QCOMPARE(result.crossings.front().mergedCount, 2);
        checkRelativeTurns(result, world, {h, world.fibers.size() - 1});
        // The apex retracing is a touch: both detections kept as events,
        // flagged, counted by no group.
        QCOMPARE(result.events.size(), std::size_t{2});
        for (const Crossing& event : result.events) {
            QVERIFY(event.touch);
            QCOMPARE(event.representative, std::size_t{0});
        }
        QVERIFY(result.groups.empty());
    }

    // A link-only network proves no winding: however large, it must not be
    // the primary component while any crossing-connected component exists.
    void linkOnlyNetworksAreNotPrimary()
    {
        World world;
        const std::size_t h0 = addH(world, 0.1, 0.5, 30000.0);
        const std::size_t v0 = addV(world, 0.3, 27000.0, 33000.0);
        // A four-fiber linked chain far away in z, crossing nothing.
        std::vector<std::size_t> chain;
        for (int i = 0; i < 4; ++i) {
            chain.push_back(
                addH(world, 0.05 + 0.5 * i, 0.55 + 0.5 * i, 42000.0));
        }
        for (int i = 0; i + 1 < 4; ++i) {
            world.links.push_back(LinkInput{
                chain[static_cast<std::size_t>(i)],
                world.fibers[chain[static_cast<std::size_t>(i)]].theta.size() - 1,
                chain[static_cast<std::size_t>(i + 1)], 0});
        }
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.placements[h0].anchor, ComponentAnchor::Primary);
        QCOMPARE(result.placements[v0].anchor, ComponentAnchor::Primary);
        QCOMPARE(result.islandCount, 1);
        for (const std::size_t f : chain) {
            QCOMPARE(result.placements[f].anchor, ComponentAnchor::Unresolved);
        }
        checkRelativeTurns(result, world, {chain[0], chain[1], chain[2], chain[3]});
    }

    // With no surviving crossing anywhere, nothing proves a winding: no
    // component may claim Primary (the dock would say "crossings"), and
    // nothing radius-anchors against an invented seed.
    void noCrossingsMeansNoPrimary()
    {
        World world;
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
        std::vector<std::size_t> chain;
        for (int i = 0; i < 3; ++i) {
            chain.push_back(
                addH(world, 0.05 + 0.5 * i, 0.55 + 0.5 * i, 42000.0));
        }
        for (int i = 0; i + 1 < 3; ++i) {
            world.links.push_back(LinkInput{
                chain[static_cast<std::size_t>(i)],
                world.fibers[chain[static_cast<std::size_t>(i)]].theta.size() - 1,
                chain[static_cast<std::size_t>(i + 1)], 0});
        }
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        for (const auto& placement : result.placements) {
            QCOMPARE(placement.anchor, ComponentAnchor::Unresolved);
        }
        QCOMPARE(result.islandCount, 2);
        QCOMPARE(result.unresolvedCount, 2);
        // The linked chain still holds together internally.
        checkRelativeTurns(result, world, {chain[0], chain[1], chain[2]});
        Q_UNUSED(v);
    }

    // Non-finite coordinates never reach the solve's sorts and integer
    // casts: a trace carrying any is unusable, the rest of the map is
    // unaffected, and nothing crashes.
    void nonFiniteTracesAreUnusable()
    {
        World world = threeWindingWorld();
        const std::size_t poisoned = addV(world, 1.7, 27000.0, 33000.0);
        world.fibers[poisoned].radius[3] =
            std::numeric_limits<double>::quiet_NaN();
        world.fibers[poisoned].theta[5] =
            std::numeric_limits<double>::infinity();
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.placements[poisoned].anchor,
                 ComponentAnchor::Unresolved);
        // The clean fibers still recover exactly.
        checkRelativeTurns(result, world, {0, 1, 2, 3});
        QCOMPARE(result.droppedCrossingCount, 0);
    }

    // A link that never took part must not read as a perfect link.
    void invalidLinksReadAsSuspect()
    {
        World world;
        addH(world, 0.05, 0.55, 30000.0);
        addV(world, 0.3, 27000.0, 33000.0);
        world.links.push_back(LinkInput{0, 999999, 1, 0});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QVERIFY(std::isinf(result.linkTurnErrors[0]));
        QVERIFY(result.droppedLinks.empty());
    }

    // The island fixture, mirrored: local radial ordering must anchor the
    // same map in the opposite chirality.
    void mirroredIslandsAnchorTheSameWay()
    {
        World world = threeWindingWorld();
        const std::size_t island = addV(world, 2.31, 30500.0, 33000.0);
        world.fibers[island].theta.assign(world.fibers[island].theta.size(),
                                          kTwoPi * 0.31);
        world.trueM[island] = 2;
        const World flipped = mirrored(world);
        const SolveResult result =
            solveWindings(flipped.fibers, flipped.links, SolverParams{});
        QCOMPARE(result.chirality, -1);
        QCOMPARE(result.placements[island].anchor, ComponentAnchor::Radius);
        checkRelativeTurns(result, flipped, {0, 1, 2, 3, island});
    }

    // A crossing landing exactly on a shared polyline vertex is one crossing,
    // not two: the segment convention is half-open.
    void sharedVertexCrossingsCountOnce()
    {
        World world;
        // 0.3 - 0.05 = 0.25 = 64 steps of 1/256: the crossing lands exactly on
        // an H sample.
        addH(world, 0.05, 0.55, 30000.0);
        addV(world, 0.3, 27000.0, 33000.0);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.crossings.size(), std::size_t{1});
        QCOMPARE(result.crossings.front().mergedCount, 1);
    }

    // A V fiber bending back in z is split into monotone branches; when the
    // branches disagree about one H fiber, the conflict surfaces as a repair
    // instead of poisoning the map.
    void uShapedVFiberSurfacesItsConflict()
    {
        World world;
        // Mid-radius H fiber: outside the U's inner branch, inside its outer,
        // measured on the branches' own ray.
        const auto mid = [](double, double z) {
            return 0.5 * (sheetR(1.3, z) + sheetR(2.3, z));
        };
        addH(world, 1.2, 1.4, 30000.0, mid);
        // The U: up the sheet at winding 1.3, back down at winding 2.3, one
        // (mis)annotated fiber.
        FiberTrace u;
        u.hvTag = 'V';
        for (double z = 29000.0; z <= 31000.0; z += 25.0) {
            u.theta.push_back(kTwoPi * 0.3);
            u.z.push_back(z);
            u.radius.push_back(sheetR(1.3, z));
        }
        for (double z = 31000.0 - 25.0; z >= 29500.0; z -= 25.0) {
            u.theta.push_back(kTwoPi * 0.3);
            u.z.push_back(z);
            u.radius.push_back(sheetR(2.3, z));
        }
        world.fibers.push_back(std::move(u));
        world.trueM.push_back(1);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.crossings.size(), std::size_t{2});
        QCOMPARE(countDroppedCrossings(result), 1);
    }

    // Angularly ill-conditioned geometry near the umbilicus takes part in
    // nothing.
    void umbilicusGrazingSegmentsAreGated()
    {
        World world;
        const auto nearCore = [](double, double) { return 300.0; };
        addH(world, 0.05, 0.55, 30000.0, nearCore);
        addV(world, 0.3, 27000.0, 33000.0);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QVERIFY(result.crossings.empty());
        QVERIFY(result.gatedSegmentCount > 0);
    }

    // An island with anchored neighbours lands on the winding its local radial
    // ordering demands, on a scroll whose spacing would defeat any global
    // radius model.
    void islandsAnchorByLocalRadialOrdering()
    {
        World world = threeWindingWorld();
        // Same angular neighbourhood as the V fibers, above the H fiber's z,
        // so it crosses nothing - but its radius reads as winding 2 against
        // its neighbours.
        const std::size_t island = addV(world, 2.31, 30500.0, 33000.0);
        world.fibers[island].theta.assign(world.fibers[island].theta.size(),
                                          kTwoPi * 0.31);
        world.trueM[island] = 2;
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.islandCount, 1);
        QCOMPARE(result.placements[island].anchor, ComponentAnchor::Radius);
        checkRelativeTurns(result, world, {0, 1, 2, 3, island});
    }

    // An island whose radius sits squarely between two anchored windings is
    // placed, but honestly marked ambiguous.
    void anIslandBetweenWindingsIsAmbiguous()
    {
        World world = threeWindingWorld();
        const std::size_t island = world.fibers.size();
        {
            // Beyond the H fiber's neighbourhood in z, so only the two V
            // fibers weigh in - and they weigh exactly evenly, the island
            // radius being the midpoint on their own ray.
            FiberTrace fiber;
            fiber.hvTag = 'V';
            for (double z = 32200.0; z <= 33000.0; z += 25.0) {
                fiber.theta.push_back(kTwoPi * 0.3);
                fiber.z.push_back(z);
                fiber.radius.push_back(
                    0.5 * (sheetR(1.3, z) + sheetR(2.3, z)));
            }
            world.fibers.push_back(std::move(fiber));
            world.trueM.push_back(1);
        }
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.placements[island].anchor,
                 ComponentAnchor::AmbiguousRadius);
    }

    // No anchored neighbours anywhere near: the island is reported unresolved
    // in its own gauge rather than silently guessed.
    void aFarIslandIsUnresolved()
    {
        World world = threeWindingWorld();
        const std::size_t island = addV(world, 2.31, 42000.0, 46000.0);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.islandCount, 1);
        QCOMPARE(result.unresolvedCount, 1);
        QCOMPARE(result.placements[island].anchor, ComponentAnchor::Unresolved);
        QVERIFY(result.placements[island].windingLo >= 0.0);
        QVERIFY(result.placements[island].windingLo < 1.0);
    }

    // --- Traversal groups: the V fiber sits inside or outside the wiggly H
    // arc, decided by the count of crossings at which H is radially inside V,
    // not by any one crossing's sign.

    // V on the outer face of the dented sheet's outer limb: the ray meets the
    // H fiber three times, every time inside V. A uniform group keeps its
    // members' own constraints (there is nothing to correct) and is reported.
    void uniformHairpinGroupKeepsIndividualConstraints()
    {
        World world;
        const double b = 100.0;
        const std::size_t h = addHairpinH(world, 30000.0, -3.0, 3.0, b, 0.03, -kSheetStep);
        const std::size_t v =
            addRayV(world, hairpinRadius(kSqrt3, b), 29000.0, 31000.0, 0);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.crossings.size(), std::size_t{3});
        const CrossingGroup& group = singleGroup(result);
        QCOMPARE(group.multiplicity, 3);
        QCOMPARE(group.insideCount, 3);
        QVERIFY(group.orientationSum % 2 != 0);
        QVERIFY(!group.mixedSigns);
        QVERIFY(!group.hasVerdict);
        for (const Crossing& crossing : result.crossings) {
            QCOMPARE(crossing.status, CrossingStatus::Used);
            QCOMPARE(crossing.kind, CrossingKind::Inside);
        }
        QCOMPARE(countDroppedCrossings(result), 0);
        // Three weak inside constraints say H is on V's winding or inward;
        // with no equality evidence the solver may pack V further out, so
        // only the inequality is asserted.
        QVERIFY(result.placements[v].turns - result.placements[h].turns >= 0.0);
    }

    // V a thickness inside the outer limb, i.e. on the next sheet inward: the
    // crossings read inside, inside, outside. Each sign alone is wrong twice;
    // the count (two inside, even) says V is on the umbilicus side of the H
    // arc, so H is strictly outside, and one group constraint replaces the
    // three - which recovers the true winding gap of one. Both chiralities.
    void mixedHairpinGroupRecoversTheGap()
    {
        for (const bool mirror : {false, true}) {
            World base;
            const double b = 100.0;
            const std::size_t h = addHairpinH(base, 30000.0, -3.0, 3.0, b, 0.03, -kSheetStep);
            const std::size_t v =
                addRayV(base, hairpinRadius(kSqrt3, b) - 300.0, 29000.0, 31000.0, -1);
            const World world = mirror ? mirrored(base) : base;
            const SolveResult result =
                solveWindings(world.fibers, world.links, SolverParams{});
            QCOMPARE(result.crossings.size(), std::size_t{3});
            const CrossingGroup& group = singleGroup(result);
            QCOMPARE(group.multiplicity, 3);
            QCOMPARE(group.insideCount, 2);
            QVERIFY(group.mixedSigns);
            QVERIFY(group.orientationSum % 2 != 0);
            QVERIFY(group.traversalCovered);
            QVERIFY(!group.coverageGap);
            QVERIFY(!group.unresolved);
            QVERIFY(group.hasVerdict);
            QCOMPARE(group.verdict, CrossingKind::Outside);
            QCOMPARE(group.status, CrossingStatus::Used);
            QCOMPARE(group.violationTurns, 0.0);
            for (const Crossing& crossing : result.crossings) {
                QCOMPARE(crossing.status, CrossingStatus::InGroup);
                QCOMPARE(crossing.groupIndex, 0LL);
            }
            QCOMPARE(countDroppedCrossings(result), 0);
            QCOMPARE(result.droppedGroupCount, 0);
            // The group alone carries the winding: it is crossing evidence,
            // so the pair is a primary component.
            QCOMPARE(result.placements[h].anchor, ComponentAnchor::Primary);
            QCOMPARE(result.placements[v].anchor, ComponentAnchor::Primary);
            checkRelativeTurns(result, world, {h, v});
        }
    }

    // Two of the three crossings sit within the merge band of each other with
    // opposite signs; the display representative is one dot, but the count is
    // over the events: two inside, one outside, verdict Outside.
    void mixedKindMergeDoesNotCorruptTheCount()
    {
        World world;
        const double b = 10.0;
        addHairpinH(world, 30000.0, -3.0, 3.0, b, 0.03, 0.0);
        // Between the middle limb (R + 40) and the outer limb (R + 139): the
        // middle crossing reads inside by 50, the outer outside by 49, the
        // inner inside by 89 - all within one tie band of each other.
        addRayV(world, kHairpinR + 90.0, 29000.0, 31000.0, -1);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QVERIFY(result.crossings.size() < 3);
        int merged = 0;
        for (const Crossing& crossing : result.crossings) {
            merged += crossing.mergedCount;
        }
        QCOMPARE(merged, 3);
        const CrossingGroup& group = singleGroup(result);
        QCOMPARE(group.multiplicity, 3);
        QCOMPARE(group.insideCount, 2);
        QVERIFY(group.hasVerdict);
        QCOMPARE(group.verdict, CrossingKind::Outside);
        // Three events behind two representatives, every event kept.
        QCOMPARE(group.members.size(), std::size_t{3});
        QCOMPARE(result.events.size(), std::size_t{3});
        for (const Crossing& event : result.events) {
            QVERIFY(event.representative < result.crossings.size());
            QCOMPARE(event.status, CrossingStatus::InGroup);
        }
    }

    // The group's verdict is a constraint like any other: opposed by a
    // stronger same-winding link it is dropped as one unit, violated by a
    // whole winding, and its members are reported as one shared conflict
    // rather than three independent errors.
    void groupLosesToAStrongerLinkAsOneUnit()
    {
        World world;
        const double b = 100.0;
        const std::size_t h = addHairpinH(world, 30000.0, -3.0, 3.0, b, 0.03, -kSheetStep);
        const std::size_t v =
            addRayV(world, hairpinRadius(kSqrt3, b) - 300.0, 29000.0, 31000.0, 0);
        // Link H's outer-limb sample (u = sqrt3 -> index round((sqrt3+3)/0.01))
        // to V at the same height: an equality the group's Outside contradicts.
        const std::size_t hSample = static_cast<std::size_t>(std::llround((kSqrt3 + 3.0) / 0.01));
        const std::size_t vSample = 40;  // z = 30000
        world.links.push_back(LinkInput{h, hSample, v, vSample});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        const CrossingGroup& group = singleGroup(result);
        QVERIFY(group.hasVerdict);
        QCOMPARE(group.status, CrossingStatus::Dropped);
        QCOMPARE(group.violationTurns, 1.0);
        QCOMPARE(result.droppedGroupCount, 1);
        QCOMPARE(result.droppedCrossingCount, 0);
        QVERIFY(result.droppedLinks.empty());
        for (const Crossing& crossing : result.crossings) {
            QCOMPARE(crossing.status, CrossingStatus::InGroup);
        }
        // One contested traversal is not drift evidence.
        QVERIFY(!result.placements[h].sheetDriftSuspect);
        checkRelativeTurns(result, world, {h, v});
    }

    // An H trace cut off at the V fiber's angle may have an incomplete count:
    // no verdict, the members constrain individually and their conflict
    // surfaces as it always did.
    void hTraceCutAtTheVFiberGetsNoVerdict()
    {
        World world;
        const double b = 100.0;
        // The mixed case, but the H trace ends at u = sqrt3 + 0.02: past the
        // outer crossing by a hair, so its last sample sits at the V fiber's
        // angle. Everything else about the group is eligible.
        addHairpinH(world, 30000.0, -3.0, kSqrt3 + 0.02, b, 0.03, -kSheetStep);
        addRayV(world, hairpinRadius(kSqrt3, b) - 300.0, 29000.0, 31000.0, -1);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.crossings.size(), std::size_t{3});
        const CrossingGroup& group = singleGroup(result);
        QCOMPARE(group.multiplicity, 3);
        QVERIFY(group.mixedSigns);
        QVERIFY(group.orientationSum % 2 != 0);
        QVERIFY(!group.coverageGap);
        QVERIFY(!group.unresolved);
        QVERIFY(!group.traversalCovered);
        QVERIFY(!group.hasVerdict);
        QCOMPARE(countDroppedCrossings(result), 1);
    }

    // An H trace that rises above the V fiber's height range WHILE at the V
    // fiber's angle could cross the fiber's untraced continuation unseen:
    // no verdict. The same trace whose ends leave the height range far from
    // the V fiber's angle is a complete traversal and keeps its verdict.
    void excursionAtTheVAngleGivesNoVerdict()
    {
        const double b = 100.0;
        for (const bool excursion : {true, false}) {
            World world;
            FiberTrace h;
            h.hvTag = 'H';
            for (double u = -3.0; u <= 3.0 + 1e-9; u += 0.01) {
                h.theta.push_back(hairpinTheta(u, 0.03));
                // Excursion: a 900 vx bulge above the V fiber's top between the
                // middle and outer crossings, within its angular window.
                // Otherwise a gentle slope whose ends fall outside the height
                // range only where the trace is far from the V fiber's angle.
                h.z.push_back(excursion
                                  ? 30000.0 + 900.0 * std::exp(-std::pow((u - 0.8) / 0.3, 2.0))
                                  : 30000.0 + 200.0 * u);
                h.radius.push_back(hairpinRadius(u, b) - kSheetStep);
            }
            world.fibers.push_back(std::move(h));
            world.trueM.push_back(0);
            addRayV(world, hairpinRadius(kSqrt3, b) - 300.0, 29500.0, 30500.0, -1);
            const SolveResult result =
                solveWindings(world.fibers, world.links, SolverParams{});
            QCOMPARE(result.crossings.size(), std::size_t{3});
            const CrossingGroup& group = singleGroup(result);
            QVERIFY(group.mixedSigns);
            QVERIFY(group.orientationSum % 2 != 0);
            QVERIFY(!group.coverageGap);
            QVERIFY(!group.unresolved);
            QVERIFY(!group.onCurtain);
            QCOMPARE(group.traversalCovered, !excursion);
            QCOMPARE(group.hasVerdict, !excursion);
            if (!excursion) {
                QCOMPARE(group.verdict, CrossingKind::Outside);
            }
        }
    }

    // An excursion above the V fiber's top that crosses its angle on ONE long
    // segment, with no sample inside the angular window, is still an
    // excursion: the test clips segments to the window, so subdividing the
    // segment changes nothing, and removing the excursion restores the
    // verdict.
    void excursionOnOneSegmentIsSeen()
    {
        const double b = 100.0;
        for (const int variant : {0, 1, 2}) {  // 0 coarse, 1 subdivided, 2 no excursion
            World world;
            FiberTrace h;
            h.hvTag = 'H';
            for (double u = -3.0; u <= 3.0 + 1e-9; u += 0.01) {
                h.theta.push_back(hairpinTheta(u, 0.03));
                h.z.push_back(30000.0);
                h.radius.push_back(hairpinRadius(u, b) - kSheetStep);
            }
            if (variant != 2) {
                // Up above the V fiber's top, back across its angle and
                // forward again, then down to end well past it on the far side.
                const double far = hairpinTheta(3.0, 0.03);
                const auto add = [&h, b](double theta, double z) {
                    h.theta.push_back(theta);
                    h.z.push_back(z);
                    h.radius.push_back(hairpinRadius(3.0, b) - kSheetStep);
                };
                add(far + 0.02, 30800.0);
                if (variant == 1) {
                    add(kHairpinTheta0, 30800.0);
                }
                add(kHairpinTheta0 - 0.6, 30800.0);
                if (variant == 1) {
                    add(kHairpinTheta0, 30800.0);
                }
                add(far + 0.06, 30800.0);
                add(far + 0.08, 30000.0);
            }
            world.fibers.push_back(std::move(h));
            world.trueM.push_back(0);
            addRayV(world, hairpinRadius(kSqrt3, b) - 300.0, 29500.0, 30500.0, -1);
            const SolveResult result =
                solveWindings(world.fibers, world.links, SolverParams{});
            QCOMPARE(result.crossings.size(), std::size_t{3});
            const CrossingGroup& group = singleGroup(result);
            QVERIFY(group.mixedSigns);
            QVERIFY(group.orientationSum % 2 != 0);
            QVERIFY(!group.coverageGap);
            QVERIFY(!group.unresolved);
            QVERIFY(!group.onCurtain);
            QCOMPARE(group.traversalCovered, variant == 2);
            QCOMPARE(group.hasVerdict, variant == 2);
            if (variant == 2) {
                QCOMPARE(group.verdict, CrossingKind::Outside);
            }
        }
    }

    // A V fiber wobbling back and forth across an H fiber crosses it an even
    // number of times with orientations that cancel: no traversal, no
    // verdict, and the members' own signs stand (here they conflict, and the
    // conflict is repaired as before).
    void wobbleIsNotATraversal()
    {
        World world;
        // A straight H climbing in z as it turns.
        FiberTrace h;
        h.hvTag = 'H';
        for (int i = 0; i <= 400; ++i) {
            const double w = 0.2 + 0.2 * i / 400.0;
            h.theta.push_back(kTwoPi * w);
            h.z.push_back(29000.0 + 2000.0 * i / 400.0);
            h.radius.push_back(sheetR(w, h.z.back()) - kSheetStep);
        }
        world.fibers.push_back(std::move(h));
        world.trueM.push_back(0);
        // A V fiber whose angle weaves across the H fiber's four times, on
        // the H fiber's own sheet for the first half and a thickness further
        // in for the second: two crossings inside, two outside.
        FiberTrace v;
        v.hvTag = 'V';
        for (int i = 0; i <= 400; ++i) {
            const double z = 29000.0 + 2000.0 * i / 400.0;
            const double wH = 0.2 + 0.2 * i / 400.0;
            const double w = wH + 0.03 * std::sin(kTwoPi * 2.0 * i / 400.0 + 0.5);
            v.theta.push_back(kTwoPi * w);
            v.z.push_back(z);
            v.radius.push_back(sheetR(w, z) - (i > 200 ? 2.0 * kSheetStep : 0.0));
        }
        world.fibers.push_back(std::move(v));
        world.trueM.push_back(0);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        const CrossingGroup& group = singleGroup(result);
        QVERIFY(group.multiplicity >= 4);
        QVERIFY(group.mixedSigns);
        QCOMPARE(group.orientationSum % 2, 0);
        QVERIFY(!group.hasVerdict);
        for (const Crossing& crossing : result.crossings) {
            QVERIFY(crossing.status != CrossingStatus::InGroup);
        }
        QVERIFY(countDroppedCrossings(result) >= 1);
    }

    // A V fiber folding back in height sweeps a curtain that covers the same
    // point three times over; its limbs are counted separately, so an H fiber
    // on the middle limb gets no verdict and the limbs' disagreement surfaces
    // as before.
    void heightFoldIsCountedPerLimb()
    {
        World world;
        const double b = 100.0;
        FiberTrace h;
        h.hvTag = 'H';
        for (double theta = kHairpinTheta0 - 0.3; theta <= kHairpinTheta0 + 0.3 + 1e-9;
             theta += 0.002) {
            h.theta.push_back(theta);
            h.z.push_back(30000.0);
            h.radius.push_back(hairpinRadius(0.0, b) - kSheetStep);
        }
        world.fibers.push_back(std::move(h));
        world.trueM.push_back(0);
        FiberTrace v;
        v.hvTag = 'V';
        for (double u = -3.0; u <= 3.0 + 1e-9; u += 0.01) {
            v.theta.push_back(kHairpinTheta0);
            v.z.push_back(30000.0 + 400.0 * (u * u * u - 3.0 * u));
            v.radius.push_back(hairpinRadius(u, b));
        }
        world.fibers.push_back(std::move(v));
        world.trueM.push_back(0);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.crossings.size(), std::size_t{3});
        for (const CrossingGroup& group : result.groups) {
            QVERIFY(!group.hasVerdict);
        }
        QVERIFY(countDroppedCrossings(result) >= 1);
    }

    // An H fiber that comes up to the V fiber's angle at a vertex and turns
    // back touches it without crossing: the record is kept and flagged, and
    // no group counts it. The legacy representative is unchanged.
    void hVertexTouchIsNotACrossing()
    {
        World world;
        FiberTrace h;
        h.hvTag = 'H';
        // Angle climbs to exactly the V fiber's angle at a sample, then falls.
        for (int i = 0; i <= 20; ++i) {
            h.theta.push_back(kHairpinTheta0 - 0.2 + 0.01 * i);
            h.z.push_back(30000.0);
            h.radius.push_back(kHairpinR - kSheetStep);
        }
        for (int i = 1; i <= 20; ++i) {
            h.theta.push_back(kHairpinTheta0 - 0.01 * i);
            h.z.push_back(30000.0);
            h.radius.push_back(kHairpinR - kSheetStep);
        }
        world.fibers.push_back(std::move(h));
        world.trueM.push_back(0);
        addRayV(world, kHairpinR, 29000.0, 31000.0, 0);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.crossings.size(), std::size_t{1});
        QCOMPARE(result.events.size(), std::size_t{1});
        QVERIFY(result.events.front().touch);
        QVERIFY(result.groups.empty());
    }

    // A V fold whose apex is sampled twice at one (angle, height) - with
    // differing radius, so every 3D segment has length - is still one vertex
    // to both branches: an H fiber zigzagging across the apex three times
    // makes three touch pairs, no crossing and no group.
    void repeatedApexSamplesAreOneVertex()
    {
        World world;
        FiberTrace v;
        v.hvTag = 'V';
        v.theta = {0.0, 0.0, 0.0, 0.0};
        v.z = {29900.0, 30000.0, 30000.0, 29900.0};
        v.radius = {20000.0, 20000.0, 20100.0, 20100.0};
        world.fibers.push_back(std::move(v));
        world.trueM.push_back(0);
        FiberTrace h;
        h.hvTag = 'H';
        h.theta = {-0.2, 0.2, -0.2, 0.2};
        h.z = {30000.0, 30000.0, 30000.0, 30000.0};
        h.radius = {18000.0, 20000.0, 21000.0, 23000.0};
        world.fibers.push_back(std::move(h));
        world.trueM.push_back(0);
        SolverParams params;
        params.chiralityOverride = 1;
        const SolveResult result = solveWindings(world.fibers, world.links, params);
        QVERIFY(!result.events.empty());
        for (const Crossing& event : result.events) {
            QVERIFY(event.touch);
        }
        for (const CrossingGroup& group : result.groups) {
            QVERIFY(!group.hasVerdict);
        }
    }

    // An H fiber running up the V fiber's own angle for a stretch overlaps it
    // collinearly in the projection with no angular width at all: the
    // overlap is measured along height, and the translate is unresolved.
    void verticalCollinearOverlapIsUnresolved()
    {
        World world;
        FiberTrace h;
        h.hvTag = 'H';
        h.theta = {-0.2, 0.0, 0.0, -0.2, 0.2, -0.2, 0.2};
        for (int i = 0; i < 7; ++i) {
            h.z.push_back(29900.0 + 20.0 * i);
        }
        h.radius = {19000.0, 19000.0, 21000.0, 21000.0, 21000.0, 21000.0, 17000.0};
        world.fibers.push_back(std::move(h));
        world.trueM.push_back(0);
        FiberTrace v;
        v.hvTag = 'V';
        v.theta = {0.0, 0.0};
        v.z = {29000.0, 31000.0};
        v.radius = {20000.0, 20000.0};
        world.fibers.push_back(std::move(v));
        world.trueM.push_back(0);
        SolverParams params;
        params.chiralityOverride = 1;
        const SolveResult result = solveWindings(world.fibers, world.links, params);
        QVERIFY(result.unresolvedIntersectionCount > 0);
        for (const CrossingGroup& group : result.groups) {
            QVERIFY(group.unresolved);
            QVERIFY(!group.hasVerdict);
        }
    }

    // The solve over externally supplied shards assembles them canonically:
    // shuffled shards give the identical result, groups and events included.
    void shuffledShardsSolveIdentically()
    {
        World world = threeWindingWorld();
        addH(world, 0.05, 2.9, 30500.0);
        // Two folded pairs with verdict groups, at different heights.
        const double dent = 100.0;
        addHairpinH(world, 42000.0, -3.0, 3.0, dent, 0.03, -kSheetStep);
        addRayV(world, hairpinRadius(kSqrt3, dent) - 300.0, 41000.0, 43000.0, -1);
        addHairpinH(world, 46000.0, -3.0, 3.0, dent, 0.03, -kSheetStep);
        addRayV(world, hairpinRadius(kSqrt3, dent) - 300.0, 45000.0, 47000.0, -1);
        SolverParams params;
        const int chirality = vc3d::fiber_map::winding::inferChirality(world.fibers,
                                                                        params.chiralityOverride);
        std::vector<vc3d::fiber_map::winding::CanonicalTrace> canonical;
        for (const FiberTrace& fiber : world.fibers) {
            canonical.push_back(vc3d::fiber_map::winding::canonicalizeTrace(fiber, chirality));
        }
        std::vector<vc3d::fiber_map::winding::PairCrossings> shards;
        std::vector<vc3d::fiber_map::winding::PairDetection> ordered;
        for (std::size_t hIndex = 0; hIndex < canonical.size(); ++hIndex) {
            if (canonical[hIndex].hvTag != 'H') {
                continue;
            }
            for (std::size_t vIndex = 0; vIndex < canonical.size(); ++vIndex) {
                if (canonical[vIndex].hvTag != 'V') {
                    continue;
                }
                shards.push_back(vc3d::fiber_map::winding::classifyPairCrossings(
                    vc3d::fiber_map::winding::detectPairCrossings(canonical[hIndex],
                                                                  canonical[vIndex], params),
                    canonical[hIndex], canonical[vIndex], {}, {}, params));
                ordered.push_back({hIndex, vIndex, nullptr});
            }
        }
        for (std::size_t i = 0; i < ordered.size(); ++i) {
            ordered[i].detection = &shards[i];
        }
        std::vector<vc3d::fiber_map::winding::PairDetection> shuffled(ordered.rbegin(),
                                                                       ordered.rend());
        std::swap(shuffled[0], shuffled[shuffled.size() / 2]);
        const SolveResult a =
            solveWindings(world.fibers, world.links, params, chirality, ordered);
        const SolveResult b =
            solveWindings(world.fibers, world.links, params, chirality, shuffled);
        QCOMPARE(a.crossings.size(), b.crossings.size());
        QCOMPARE(a.events.size(), b.events.size());
        QCOMPARE(a.groups.size(), b.groups.size());
        QVERIFY(a.groups.size() >= 2);
        const auto sameCrossing = [](const Crossing& x, const Crossing& y) {
            return x.hFiber == y.hFiber && x.vFiber == y.vFiber && x.zVx == y.zVx &&
                   x.psiH == y.psiH && x.n == y.n && x.deltaR == y.deltaR &&
                   x.confidence == y.confidence && x.kind == y.kind && x.status == y.status &&
                   x.orientation == y.orientation && x.touch == y.touch &&
                   x.vBranch == y.vBranch && x.representative == y.representative &&
                   x.coveredByGroups == y.coveredByGroups && x.groupIndex == y.groupIndex &&
                   x.violationTurns == y.violationTurns;
        };
        for (std::size_t i = 0; i < a.crossings.size(); ++i) {
            QVERIFY(sameCrossing(a.crossings[i], b.crossings[i]));
        }
        for (std::size_t i = 0; i < a.events.size(); ++i) {
            QVERIFY(sameCrossing(a.events[i], b.events[i]));
        }
        int verdicts = 0;
        for (std::size_t i = 0; i < a.groups.size(); ++i) {
            const CrossingGroup& x = a.groups[i];
            const CrossingGroup& y = b.groups[i];
            QCOMPARE(x.hFiber, y.hFiber);
            QCOMPARE(x.vFiber, y.vFiber);
            QCOMPARE(x.n, y.n);
            QCOMPARE(x.members, y.members);
            QCOMPARE(x.hasVerdict, y.hasVerdict);
            QCOMPARE(x.verdict, y.verdict);
            QCOMPARE(x.status, y.status);
            QCOMPARE(x.confidence, y.confidence);
            QCOMPARE(x.violationTurns, y.violationTurns);
            verdicts += x.hasVerdict ? 1 : 0;
        }
        QCOMPARE(verdicts, 2);
        for (std::size_t f = 0; f < world.fibers.size(); ++f) {
            QCOMPARE(a.placements[f].turns, b.placements[f].turns);
            QCOMPARE(a.placements[f].anchor, b.placements[f].anchor);
        }
        QCOMPARE(a.droppedCrossingCount, b.droppedCrossingCount);
        QCOMPARE(a.droppedGroupCount, b.droppedGroupCount);
    }

    // --- Kollesis seam encounters.

    // Without the flags the seam crossing reads Outside and fights the link;
    // with the H end tagged, the V on the kollesis and the pair linked it
    // reads Inside, is flagged, and nothing is dropped.
    void kollesisSeamEncounterReadsInside()
    {
        for (const bool flagged : {false, true}) {
            World world;
            const std::size_t h = addH(world, 0.05, kSeamWinding + kSeamOvershoot, kSeamHeight);
            // In front of the H fiber's end by a thickness: r_v = sheet - 200,
            // r_h = sheet - 100 -> deltaR = +100.
            const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
            world.links.push_back(LinkInput{
                h, world.fibers[h].theta.size() - 1, v, 40});
            if (flagged) {
                world.fibers[h].kollesisEndSample = world.fibers[h].theta.size() - 1;
                world.fibers[v].onKollesis = true;
            }
            const SolveResult result =
                solveWindings(world.fibers, world.links, SolverParams{});
            QCOMPARE(result.crossings.size(), std::size_t{1});
            QCOMPARE(result.events.size(), std::size_t{1});
            const Crossing& event = result.events.front();
            QVERIFY(event.deltaR > 0.0);
            QCOMPARE(event.kollesis, flagged);
            QCOMPARE(event.kind, flagged ? CrossingKind::Inside : CrossingKind::Outside);
            QCOMPARE(result.crossings.front().kind, event.kind);
            QCOMPARE(result.kollesisCrossingCount, flagged ? 1 : 0);
            // The link holds either way; without the flags the crossing is
            // the casualty.
            QVERIFY(result.droppedLinks.empty());
            QCOMPARE(countDroppedCrossings(result), flagged ? 0 : 1);
            checkRelativeTurns(result, world, {h, v});
            QVERIFY(result.groups.empty());
        }
    }

    // Only the encounter at the tagged end is read as the seam: a crossing of
    // the same pair a turn earlier keeps its radial reading.
    void kollesisReadingStaysAtTheTaggedEnd()
    {
        World world;
        const std::size_t h = addH(world, kSeamWinding - 1.2, kSeamWinding + kSeamOvershoot,
                                   kSeamHeight, outerOnEarlierTurn);
        const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
        world.fibers[h].kollesisEndSample = world.fibers[h].theta.size() - 1;
        world.fibers[v].onKollesis = true;
        world.links.push_back(LinkInput{h, world.fibers[h].theta.size() - 1, v, 40});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.events.size(), std::size_t{2});
        int seam = 0;
        int outside = 0;
        for (const Crossing& event : result.events) {
            if (event.kollesis) {
                ++seam;
                QCOMPARE(event.kind, CrossingKind::Inside);
            } else {
                QCOMPARE(event.kind, CrossingKind::Outside);
                ++outside;
            }
        }
        QCOMPARE(seam, 1);
        QCOMPARE(outside, 1);
        QCOMPARE(result.kollesisCrossingCount, 1);
    }

    // A tag on the H fiber's first control point (the outer sheet's H fiber
    // starting at the seam) mirrors the end tag.
    void kollesisStartTagMirrorsTheEndTag()
    {
        World world;
        const std::size_t h = addH(world, kSeamWinding - kSeamOvershoot, kSeamWinding + 0.5,
                                   kSeamHeight);
        const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
        world.fibers[h].kollesisStartSample = 0;
        world.fibers[v].onKollesis = true;
        world.links.push_back(LinkInput{h, 0, v, 40});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.events.size(), std::size_t{1});
        QVERIFY(result.events.front().kollesis);
        QCOMPARE(result.events.front().kind, CrossingKind::Inside);
        QCOMPARE(countDroppedCrossings(result), 0);
    }

    // Tag, kollesis V and link are all needed: a tagged H end against a V
    // that is not on the kollesis, a kollesis V against an untagged H, or a
    // tagged H and kollesis V that are not linked, read as before.
    void kollesisNeedsBothFlags()
    {
        for (const int which : {0, 1, 2}) {
            World world;
            const std::size_t h = addH(world, 0.05, kSeamWinding + kSeamOvershoot, kSeamHeight);
            const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
            if (which != 1) {
                world.fibers[h].kollesisEndSample = world.fibers[h].theta.size() - 1;
            }
            if (which != 0) {
                world.fibers[v].onKollesis = true;
            }
            if (which != 2) {
                world.links.push_back(LinkInput{h, world.fibers[h].theta.size() - 1, v, 40});
            }
            const SolveResult result =
                solveWindings(world.fibers, world.links, SolverParams{});
            QCOMPARE(result.events.size(), std::size_t{1});
            QVERIFY(!result.events.front().kollesis);
            QCOMPARE(result.events.front().kind, CrossingKind::Outside);
            QCOMPARE(result.kollesisCrossingCount, 0);
        }
    }

    // The seam encounter is read as a whole: when the H fiber's end wobbles
    // across the V fiber's angle so two detections share the encounter's
    // cluster, both are reclassified and the representative is Inside even
    // though the more confident detection read Outside.
    void kollesisReadsTheWholeEncounter()
    {
        World world;
        FiberTrace h;
        h.hvTag = 'H';
        for (double w = 0.05; w <= kSeamWinding + kSeamOvershoot + 1e-9; w += 1.0 / 256.0) {
            h.theta.push_back(kTwoPi * w);
            h.z.push_back(kSeamHeight);
            h.radius.push_back(sheetR(w, kSeamHeight) - kSheetStep);
        }
        // Back across the seam angle: a second pass of the V fiber's angle
        // at nearly the same height and radius.
        h.theta.push_back(kTwoPi * (kSeamWinding - 0.5 * kSeamOvershoot));
        h.z.push_back(kSeamHeight + 10.0);
        h.radius.push_back(sheetR(kSeamWinding, kSeamHeight) - kSheetStep);
        h.kollesisEndSample = h.theta.size() - 1;
        world.fibers.push_back(std::move(h));
        world.trueM.push_back(0);
        const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
        world.fibers[v].onKollesis = true;
        world.links.push_back(LinkInput{0, world.fibers[0].theta.size() - 1, v, 40});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QVERIFY(result.events.size() >= 2);
        for (const Crossing& event : result.events) {
            QVERIFY(event.kollesis);
            QCOMPARE(event.kind, CrossingKind::Inside);
        }
        QCOMPARE(result.crossings.size(), std::size_t{1});
        QCOMPARE(result.crossings.front().kind, CrossingKind::Inside);
        QVERIFY(result.crossings.front().kollesis);
        QVERIFY(result.crossings.front().mergedCount >= 2);
        QVERIFY(result.groups.empty());
    }

    // The tagged control need not be the trace's end: the layout runs a
    // sample beyond the outer controls, and here that padding sample climbs
    // above the V fiber's top. The link names the encounter either way.
    void kollesisTaggedSampleNeedNotBeTheTraceEnd()
    {
        for (const bool tagPadding : {false, true}) {
            World world;
            const std::size_t h = addH(world, 0.05, kSeamWinding + kSeamOvershoot, kSeamHeight);
            const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
            world.fibers[v].onKollesis = true;
            FiberTrace& hFiber = world.fibers[h];
            hFiber.theta.push_back(hFiber.theta.back() + kTwoPi / 256.0);
            hFiber.z.push_back(31500.0);
            hFiber.radius.push_back(hFiber.radius.back());
            hFiber.kollesisEndSample = tagPadding ? hFiber.theta.size() - 1
                                                  : hFiber.theta.size() - 2;
            world.links.push_back(LinkInput{h, hFiber.theta.size() - 2, v, 40});
            const SolveResult result =
                solveWindings(world.fibers, world.links, SolverParams{});
            QCOMPARE(result.events.size(), std::size_t{1});
            QVERIFY(result.events.front().kollesis);
            QCOMPARE(result.events.front().kind, CrossingKind::Inside);
        }
    }

    // A V fiber folded in height meets the H fiber's end on both limbs at
    // the same angle and height: one thickness in front (the seam, +100)
    // and well behind (-400). The link names the limb: linked to the front
    // limb the +100 crossing is the seam; linked to the back limb, that one
    // is, and the front limb's crossing keeps its radial reading. Unlinked,
    // nothing is read; links to both limbs from one control disagree and
    // read nothing either.
    void kollesisLinkNamesTheLimbOfAFoldedV()
    {
        enum Mode { Unlinked, LinkFront, LinkBack, LinkBoth };
        for (const Mode mode : {Unlinked, LinkFront, LinkBack, LinkBoth}) {
            World world;
            const std::size_t h = addH(world, 0.05, kSeamWinding + kSeamOvershoot, kSeamHeight);
            FiberTrace vFiber;
            vFiber.hvTag = 'V';
            vFiber.onKollesis = true;
            for (double z = 29000.0; z <= 31000.0 + 1e-9; z += 25.0) {
                vFiber.theta.push_back(kTwoPi * kSeamWinding);
                vFiber.z.push_back(z);
                vFiber.radius.push_back(sheetR(kSeamWinding, z) - 200.0);
            }
            const std::size_t frontAtSeamHeight = 40;
            QCOMPARE(vFiber.z[frontAtSeamHeight], kSeamHeight);
            for (double z = 30975.0; z >= 29000.0 - 1e-9; z -= 25.0) {
                vFiber.theta.push_back(kTwoPi * kSeamWinding);
                vFiber.z.push_back(z);
                vFiber.radius.push_back(sheetR(kSeamWinding, z) + 300.0);
            }
            const std::size_t backAtSeamHeight = 120;
            QCOMPARE(vFiber.z[backAtSeamHeight], kSeamHeight);
            world.fibers.push_back(std::move(vFiber));
            world.trueM.push_back(0);
            const std::size_t v = world.fibers.size() - 1;
            const std::size_t hEnd = world.fibers[h].theta.size() - 1;
            world.fibers[h].kollesisEndSample = hEnd;
            if (mode == LinkFront || mode == LinkBoth) {
                world.links.push_back(LinkInput{h, hEnd, v, frontAtSeamHeight});
            }
            if (mode == LinkBack || mode == LinkBoth) {
                world.links.push_back(LinkInput{h, hEnd, v, backAtSeamHeight});
            }
            const SolveResult result =
                solveWindings(world.fibers, world.links, SolverParams{});
            QCOMPARE(result.events.size(), std::size_t{2});
            for (const Crossing& event : result.events) {
                if (event.deltaR > 0.0) {
                    QCOMPARE(event.kollesis, mode == LinkFront);
                    QCOMPARE(event.kind,
                             mode == LinkFront ? CrossingKind::Inside : CrossingKind::Outside);
                } else {
                    QCOMPARE(event.kollesis, mode == LinkBack);
                    QCOMPARE(event.kind, CrossingKind::Inside);
                }
            }
            QCOMPARE(result.kollesisCrossingCount,
                     mode == LinkFront || mode == LinkBack ? 1 : 0);
        }
    }

    // An H fiber folded back on itself crosses the V fiber twice at one
    // height, first far outside it (+1000) and then, nearest its tagged end,
    // one thickness behind (+100). Only the encounter at the tag is the
    // seam; the radially distinct crossing at the same height keeps its
    // Outside reading.
    void kollesisReadsOnlyTheEncountersOwnRadius()
    {
        World world;
        FiberTrace hFiber;
        hFiber.hvTag = 'H';
        const double zH = kSeamHeight;
        for (double w = 0.05; w < kSeamWinding - 0.011; w += 1.0 / 256.0) {
            hFiber.theta.push_back(kTwoPi * w);
            hFiber.z.push_back(zH);
            hFiber.radius.push_back(sheetR(w, zH) - kSheetStep);
        }
        const double seamSheet = sheetR(kSeamWinding, zH);
        // Out to +800 over the sheet, across the V fiber (+1000), then back
        // across it a thickness behind (+100) to the tagged end.
        const double outward[][2] = {{kSeamWinding - 0.01, seamSheet + 800.0},
                                     {kSeamWinding + 0.02, seamSheet + 800.0},
                                     {kSeamWinding + 0.02, seamSheet - kSheetStep},
                                     {kSeamWinding - 0.005, seamSheet - kSheetStep}};
        for (const auto& [w, r] : outward) {
            hFiber.theta.push_back(kTwoPi * w);
            hFiber.z.push_back(zH);
            hFiber.radius.push_back(r);
        }
        hFiber.kollesisEndSample = hFiber.theta.size() - 1;
        world.fibers.push_back(std::move(hFiber));
        world.trueM.push_back(0);
        const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
        world.fibers[v].onKollesis = true;
        world.links.push_back(LinkInput{0, world.fibers[0].theta.size() - 1, v, 40});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.events.size(), std::size_t{2});
        int seam = 0;
        for (const Crossing& event : result.events) {
            if (event.deltaR > 500.0) {
                QVERIFY(!event.kollesis);
                QCOMPARE(event.kind, CrossingKind::Outside);
            } else {
                QVERIFY(std::abs(event.deltaR - 100.0) < 30.0);
                QVERIFY(event.kollesis);
                QCOMPARE(event.kind, CrossingKind::Inside);
                ++seam;
            }
        }
        QCOMPARE(seam, 1);
        QCOMPARE(result.kollesisCrossingCount, 1);
    }

    // A link to the apex of a height-folded V fiber names a vertex both
    // limbs share. The H fiber crosses the near limb one thickness behind
    // it (+100) and the far limb, at its tagged end, three thicknesses
    // behind (+300): the seam is the encounter at the tag, not the thinner
    // one further back along the H fiber.
    void kollesisApexLinkPicksTheEncounterAtTheTag()
    {
        World world;
        const std::size_t h = addH(world, 0.05, kSeamWinding + 0.09, kSeamHeight);
        FiberTrace vFiber;
        vFiber.hvTag = 'V';
        vFiber.onKollesis = true;
        // Near limb slanting from w = 0.55 at the bottom to the apex at
        // w = 0.60, 2000 vx up; far limb back down to w = 0.65.
        const auto limbW = [](double z, double wBottom, double wTop) {
            return wBottom + (wTop - wBottom) * (z - 29000.0) / 2000.0;
        };
        for (double z = 29000.0; z <= 31000.0 + 1e-9; z += 25.0) {
            const double w = limbW(z, kSeamWinding, kSeamWinding + 0.05);
            vFiber.theta.push_back(kTwoPi * w);
            vFiber.z.push_back(z);
            vFiber.radius.push_back(sheetR(w, z) - 200.0);
        }
        const std::size_t apex = vFiber.z.size() - 1;
        for (double z = 30975.0; z >= 29000.0 - 1e-9; z -= 25.0) {
            const double w = limbW(z, kSeamWinding + 0.1, kSeamWinding + 0.05);
            vFiber.theta.push_back(kTwoPi * w);
            vFiber.z.push_back(z);
            vFiber.radius.push_back(sheetR(w, z) - 400.0);
        }
        world.fibers.push_back(std::move(vFiber));
        world.trueM.push_back(0);
        const std::size_t v = world.fibers.size() - 1;
        const std::size_t hEnd = world.fibers[h].theta.size() - 1;
        world.fibers[h].kollesisEndSample = hEnd;
        world.links.push_back(LinkInput{h, hEnd, v, apex});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.events.size(), std::size_t{2});
        for (const Crossing& event : result.events) {
            const bool far = event.deltaR > 200.0;
            QVERIFY(far || std::abs(event.deltaR - 100.0) < 30.0);
            QCOMPARE(event.kollesis, far);
            QCOMPARE(event.kind, far ? CrossingKind::Inside : CrossingKind::Outside);
        }
        QCOMPARE(result.kollesisCrossingCount, 1);
    }

    // Three limbs of a zigzag V fiber cross the H fiber's coarse last
    // segment about 0.09, 0.43 and 0.77 samples before its tagged end, at
    // +300, +155 and +10; a fourth limb beyond the end, which the link
    // names, is never crossed, so the limb is chosen geometrically. The two
    // within half a sample of the nearest are one place and the thinner of
    // them (+155) is the seam - whichever order the limbs come in; a chained
    // pairwise tie reached +10 in one order and +300 in the other.
    void kollesisRanksLimbsAgainstTheNearestNotPairwise()
    {
        for (const bool reversedLimbs : {false, true}) {
            World world;
            const std::size_t h = addH(world, 0.05, kSeamWinding - 0.01, kSeamHeight);
            world.fibers[h].theta.push_back(kTwoPi * kSeamWinding);
            world.fibers[h].z.push_back(kSeamHeight);
            world.fibers[h].radius.push_back(sheetR(kSeamWinding, kSeamHeight) - kSheetStep);
            const std::size_t hEnd = world.fibers[h].theta.size() - 1;
            world.fibers[h].kollesisEndSample = hEnd;
            // Limbs (w, radius offset) along the last segment, which runs
            // from the last 1/256 step below kSeamWinding - 0.01 to the end;
            // dR = -offset - 100.
            std::vector<std::pair<double, double>> limbs{
                {kSeamWinding - 0.001, -400.0},
                {kSeamWinding - 0.005, -255.0},
                {kSeamWinding - 0.009, -110.0}};
            if (reversedLimbs) {
                std::reverse(limbs.begin(), limbs.end());
            }
            limbs.push_back({kSeamWinding + 0.03, -200.0});
            FiberTrace vFiber;
            vFiber.hvTag = 'V';
            vFiber.onKollesis = true;
            bool up = true;
            for (const auto& [w, offset] : limbs) {
                const double z0 = up ? 29000.0 : 31000.0;
                const double step = up ? 25.0 : -25.0;
                for (int k = 0; k <= 80; ++k) {
                    if (k == 0 && !vFiber.z.empty()) {
                        continue; // the apex sample is shared
                    }
                    const double z = z0 + step * k;
                    vFiber.theta.push_back(kTwoPi * w);
                    vFiber.z.push_back(z);
                    vFiber.radius.push_back(sheetR(w, z) + offset);
                }
                up = !up;
            }
            // The fourth limb descends from the shared apex: its sample at
            // the seam height.
            const std::size_t uncrossedAtSeamHeight = vFiber.z.size() - 41;
            QCOMPARE(vFiber.z[uncrossedAtSeamHeight], kSeamHeight);
            QCOMPARE(vFiber.theta[uncrossedAtSeamHeight], kTwoPi * (kSeamWinding + 0.03));
            world.fibers.push_back(std::move(vFiber));
            world.trueM.push_back(0);
            world.links.push_back(
                LinkInput{h, hEnd, world.fibers.size() - 1, uncrossedAtSeamHeight});
            const SolveResult result =
                solveWindings(world.fibers, world.links, SolverParams{});
            QCOMPARE(result.events.size(), std::size_t{3});
            int seam = 0;
            for (const Crossing& event : result.events) {
                const bool middle = std::abs(event.deltaR - 155.0) < 20.0;
                QCOMPARE(event.kollesis, middle);
                seam += middle ? 1 : 0;
            }
            QCOMPARE(seam, 1);
            QCOMPARE(result.kollesisCrossingCount, 1);
        }
    }

    // The annotator links the tagged end to the V fiber's nearest control,
    // which on a V folded in height may sit on a limb the H fiber never
    // reaches at the tag: here the far limb, 0.05 turn past the H fiber's
    // end at the same heights, which the H fiber crossed a turn earlier but
    // not on the tagged end's translate. The link still identifies the V;
    // the encounter falls back to the limb the end sits against, and the
    // earlier-turn crossings keep their readings.
    void kollesisLinkOnAnUncrossedLimbFallsBackToGeometry()
    {
        World world;
        const std::size_t h = addH(world, kSeamWinding - 1.2, kSeamWinding + kSeamOvershoot,
                                   kSeamHeight);
        FiberTrace vFiber;
        vFiber.hvTag = 'V';
        vFiber.onKollesis = true;
        for (double z = 29000.0; z <= 31000.0 + 1e-9; z += 25.0) {
            vFiber.theta.push_back(kTwoPi * kSeamWinding);
            vFiber.z.push_back(z);
            vFiber.radius.push_back(sheetR(kSeamWinding, z) - 200.0);
        }
        // Across to the far limb at the top, then back down beyond the H
        // fiber's end.
        const double farW = kSeamWinding + 0.05;
        for (double z = 31000.0; z >= 29000.0 - 1e-9; z -= 25.0) {
            vFiber.theta.push_back(kTwoPi * farW);
            vFiber.z.push_back(z);
            vFiber.radius.push_back(sheetR(farW, z) - 200.0);
        }
        const std::size_t farAtSeamHeight = vFiber.z.size() - 41;
        QCOMPARE(vFiber.z[farAtSeamHeight], kSeamHeight);
        QCOMPARE(vFiber.theta[farAtSeamHeight], kTwoPi * farW);
        world.fibers.push_back(std::move(vFiber));
        world.trueM.push_back(0);
        const std::size_t v = world.fibers.size() - 1;
        const std::size_t hEnd = world.fibers[h].theta.size() - 1;
        world.fibers[h].kollesisEndSample = hEnd;
        world.links.push_back(LinkInput{h, hEnd, v, farAtSeamHeight});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        // Both limbs a turn earlier (Inside by a wrap) and the near limb at
        // the tag.
        QCOMPARE(result.events.size(), std::size_t{3});
        int seam = 0;
        for (const Crossing& event : result.events) {
            if (event.kollesis) {
                ++seam;
                QVERIFY(std::abs(event.deltaR - 100.0) < 30.0);
                QCOMPARE(event.kind, CrossingKind::Inside);
            } else {
                QVERIFY(event.deltaR < -1000.0);
            }
        }
        QCOMPARE(seam, 1);
        QCOMPARE(result.kollesisCrossingCount, 1);
    }

    // Links are drawn at the crossing, not at the tagged end: an H fiber
    // whose tagged end lies 0.2 turn past the V fiber, linked to the V at
    // the crossing control, is read there. The tag qualifies the fiber; the
    // link names the encounter.
    void kollesisLinkAtTheCrossingFarFromTheTag()
    {
        World world;
        const std::size_t h = addH(world, 0.05, kSeamWinding + 0.2, kSeamHeight);
        const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
        world.fibers[v].onKollesis = true;
        world.fibers[h].kollesisEndSample = world.fibers[h].theta.size() - 1;
        // The H sample at the V fiber's angle.
        const std::size_t atCrossing = static_cast<std::size_t>(
            std::llround((kSeamWinding - 0.05) * 256.0));
        QVERIFY(std::abs(world.fibers[h].theta[atCrossing] - kTwoPi * kSeamWinding) < 0.02);
        world.links.push_back(LinkInput{h, atCrossing, v, 40});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.events.size(), std::size_t{1});
        QVERIFY(result.events.front().kollesis);
        QCOMPARE(result.events.front().kind, CrossingKind::Inside);
        QVERIFY(result.droppedLinks.empty());
        QCOMPARE(countDroppedCrossings(result), 0);
    }

    // The link localizes the encounter, not the tag: an H fiber running a
    // full turn past the V after the crossing it is linked at, tagged at its
    // end, meets the V again a turn later, nearer the tag. The linked
    // crossing is the seam; the later one, a genuine winding out, is not.
    void kollesisLinkNotTagLocalizesTheEncounter()
    {
        World world;
        const std::size_t h = addH(world, 0.05, kSeamWinding + 1.02, kSeamHeight);
        const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
        world.fibers[v].onKollesis = true;
        world.fibers[h].kollesisEndSample = world.fibers[h].theta.size() - 1;
        const std::size_t atCrossing = static_cast<std::size_t>(
            std::llround((kSeamWinding - 0.05) * 256.0));
        world.links.push_back(LinkInput{h, atCrossing, v, 40});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.events.size(), std::size_t{2});
        for (const Crossing& event : result.events) {
            const bool linked = std::abs(event.deltaR - 100.0) < 30.0;
            QVERIFY(linked || event.deltaR > 1000.0);
            QCOMPARE(event.kollesis, linked);
            QCOMPARE(event.kind, linked ? CrossingKind::Inside : CrossingKind::Outside);
        }
        QCOMPARE(result.kollesisCrossingCount, 1);
    }

    // The link is drawn on a control that has climbed 700 vx above the V
    // fiber's top, 0.15 turn past the crossing, and names a far limb the H
    // never reaches (joined to the near limb below the H fiber). The
    // fallback must still find the near limb's encounter: the linked
    // control's height does not veto the limb.
    void kollesisFallbackIgnoresTheLinkedControlsHeight()
    {
        World world;
        FiberTrace hFiber;
        hFiber.hvTag = 'H';
        const double linkW = kSeamWinding + 0.15;
        for (double w = 0.05; w <= linkW + 1e-9; w += 1.0 / 256.0) {
            hFiber.theta.push_back(kTwoPi * w);
            hFiber.z.push_back(kSeamHeight);
            hFiber.radius.push_back(sheetR(w, kSeamHeight) - kSheetStep);
        }
        hFiber.theta.push_back(kTwoPi * (linkW + 0.001));
        hFiber.z.push_back(kSeamHeight + 700.0);
        hFiber.radius.push_back(sheetR(linkW, kSeamHeight) - kSheetStep);
        const std::size_t climbed = hFiber.theta.size() - 1;
        hFiber.theta.push_back(kTwoPi * (kSeamWinding + 0.4));
        hFiber.z.push_back(kSeamHeight);
        hFiber.radius.push_back(sheetR(kSeamWinding + 0.4, kSeamHeight) - kSheetStep);
        hFiber.kollesisEndSample = hFiber.theta.size() - 1;
        world.fibers.push_back(std::move(hFiber));
        world.trueM.push_back(0);
        // Far limb at 0.2 turn past the crossing, wholly above the H fiber's
        // descent there and below it where it starts; joined at the bottom
        // to the near limb, which tops out 400 vx below the linked control.
        FiberTrace vFiber;
        vFiber.hvTag = 'V';
        vFiber.onKollesis = true;
        const double farW = kSeamWinding + 0.2;
        for (double z = 30050.0; z >= 28900.0 - 1e-9; z -= 25.0) {
            vFiber.theta.push_back(kTwoPi * farW);
            vFiber.z.push_back(z);
            vFiber.radius.push_back(sheetR(farW, z) - 200.0);
        }
        const std::size_t farAtSeamHeight = 2;
        QCOMPARE(vFiber.z[farAtSeamHeight], kSeamHeight);
        for (double z = 28900.0; z <= 30300.0 + 1e-9; z += 25.0) {
            vFiber.theta.push_back(kTwoPi * kSeamWinding);
            vFiber.z.push_back(z);
            vFiber.radius.push_back(sheetR(kSeamWinding, z) - 200.0);
        }
        world.fibers.push_back(std::move(vFiber));
        world.trueM.push_back(0);
        world.links.push_back(LinkInput{0, climbed, 1, farAtSeamHeight});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.events.size(), std::size_t{1});
        QVERIFY(result.events.front().kollesis);
        QCOMPARE(result.events.front().kind, CrossingKind::Inside);
        QVERIFY(result.events.front().deltaR > 0.0);
    }

    // A link whose limb saw nothing on the link's translate does not reach
    // for another turn: the H fiber is linked to the V at a control where
    // it does not cross it (0.05 turn past the V's angle), and first meets
    // the V a whole turn later. That crossing, a genuine winding out, keeps
    // its reading.
    void kollesisFallbackStaysOnTheLinksTranslate()
    {
        World world;
        const std::size_t h = addH(world, kSeamWinding + 0.05, kSeamWinding + 1.05, kSeamHeight);
        const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
        world.fibers[v].onKollesis = true;
        world.fibers[h].kollesisEndSample = world.fibers[h].theta.size() - 1;
        world.links.push_back(LinkInput{h, 0, v, 40});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.events.size(), std::size_t{1});
        QVERIFY(!result.events.front().kollesis);
        QVERIFY(result.events.front().deltaR > 1000.0);
        QCOMPARE(result.events.front().kind, CrossingKind::Outside);
        QCOMPARE(result.kollesisCrossingCount, 0);
    }

    // Terminal events and the inferred seam reading, at the classification
    // level. An H ending 0.02 turn past the V: its one crossing is terminal,
    // reads Outside, and read as an inferred seam becomes Inside, flagged
    // kollesis and kollesisInferred. An H running on for 2.2 turns: its
    // middle crossing is not terminal (the V is met again a turn later, and
    // was met a turn earlier), its first is toward the H's start and its
    // last toward its end (0.2 turn past, within the V's height).
    void inferredSeamReadsTheTerminalEncounter()
    {
        const SolverParams params;
        {
            World world;
            const std::size_t h = addH(world, 0.05, kSeamWinding + kSeamOvershoot, kSeamHeight);
            const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
            const int chirality = inferChirality(world.fibers, params.chiralityOverride);
            const CanonicalTrace ch = canonicalizeTrace(world.fibers[h], chirality);
            const CanonicalTrace cv = canonicalizeTrace(world.fibers[v], chirality);
            const PairDetections geometry = detectPairCrossings(ch, cv, params);
            QCOMPARE(geometry.raw.size(), std::size_t{1});
            const PairCrossings plain = classifyPairCrossings(geometry, ch, cv, {}, {}, params);
            QCOMPARE(plain.events.size(), std::size_t{1});
            QVERIFY(plain.events.front().terminal);
            // Ends both ways within a turn: 0.5 turn to the start, 0.02 to
            // the end.
            QCOMPARE(plain.events.front().terminalSides, 3);
            QCOMPARE(plain.events.front().kind, CrossingKind::Outside);
            QVERIFY(!plain.events.front().kollesis);
            const PairCrossings inferred = classifyPairCrossings(
                geometry, ch, cv, {}, {geometry.raw.front().detection}, params);
            QCOMPARE(inferred.events.size(), std::size_t{1});
            QCOMPARE(inferred.events.front().kind, CrossingKind::Inside);
            QVERIFY(inferred.events.front().kollesis);
            QVERIFY(inferred.events.front().kollesisInferred);
            QCOMPARE(inferred.crossings.front().kind, CrossingKind::Inside);
            QVERIFY(inferred.crossings.front().kollesisInferred);
            QVERIFY(inferred.groups.empty());
        }
        {
            World world;
            const std::size_t h = addH(world, 0.05, kSeamWinding + 2.2, kSeamHeight);
            const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
            const int chirality = inferChirality(world.fibers, params.chiralityOverride);
            const CanonicalTrace ch = canonicalizeTrace(world.fibers[h], chirality);
            const CanonicalTrace cv = canonicalizeTrace(world.fibers[v], chirality);
            const PairCrossings plain = classifyPairCrossings(
                detectPairCrossings(ch, cv, params), ch, cv, {}, {}, params);
            QCOMPARE(plain.events.size(), std::size_t{3});
            std::vector<const Crossing*> byAlong;
            for (const Crossing& event : plain.events) {
                byAlong.push_back(&event);
            }
            std::sort(byAlong.begin(), byAlong.end(), [](const Crossing* a, const Crossing* b) {
                return a->hSegment < b->hSegment;
            });
            QVERIFY(byAlong[0]->terminal);
            QVERIFY(!byAlong[1]->terminal);
            QCOMPARE(byAlong[1]->terminalSides, 0);
            QVERIFY(byAlong[2]->terminal);
            // The first ends toward the start, the last toward the end: on
            // opposite sides of their crossings.
            QVERIFY(byAlong[0]->terminalSides != 0 && byAlong[2]->terminalSides != 0);
            QCOMPARE(byAlong[0]->terminalSides & byAlong[2]->terminalSides, 0);
        }
        // An H that leaves the V's height range on its way to its end is
        // not terminal toward that end (a further crossing could have gone
        // unseen), only toward its start.
        {
            World world;
            const std::size_t h = addH(world, 0.05, kSeamWinding + kSeamOvershoot, kSeamHeight);
            const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
            FiberTrace& hFiber = world.fibers[h];
            hFiber.theta.push_back(hFiber.theta.back() + kTwoPi * 0.1);
            hFiber.z.push_back(31500.0);
            hFiber.radius.push_back(hFiber.radius.back());
            const int chirality = inferChirality(world.fibers, params.chiralityOverride);
            const CanonicalTrace ch = canonicalizeTrace(world.fibers[h], chirality);
            const CanonicalTrace cv = canonicalizeTrace(world.fibers[v], chirality);
            const PairCrossings plain = classifyPairCrossings(
                detectPairCrossings(ch, cv, params), ch, cv, {}, {}, params);
            QCOMPARE(plain.events.size(), std::size_t{1});
            const Crossing& event = plain.events.front();
            const int endBit = ch.psi.back() > event.psiH ? 1 : 2;
            QVERIFY(event.terminal);
            QCOMPARE(event.terminalSides & endBit, 0);
            QVERIFY(event.terminalSides != 0);
        }
        // A gated segment on the way to the end (a 0.4-turn jump between
        // samples, over the step gate) could hide a further crossing: not
        // terminal toward that end either, though the end is within a turn
        // and at the V's height.
        {
            World world;
            const std::size_t h = addH(world, 0.05, kSeamWinding + kSeamOvershoot, kSeamHeight);
            const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
            FiberTrace& hFiber = world.fibers[h];
            hFiber.theta.push_back(hFiber.theta.back() + kTwoPi * 0.4);
            hFiber.z.push_back(kSeamHeight);
            hFiber.radius.push_back(hFiber.radius.back());
            const int chirality = inferChirality(world.fibers, params.chiralityOverride);
            const CanonicalTrace ch = canonicalizeTrace(world.fibers[h], chirality);
            const CanonicalTrace cv = canonicalizeTrace(world.fibers[v], chirality);
            const PairDetections geometry = detectPairCrossings(ch, cv, params);
            QCOMPARE(geometry.uncoveredSegments.size(), std::size_t{1});
            QCOMPARE(geometry.uncoveredSegments.front(), ch.psi.size() - 2);
            const PairCrossings plain = classifyPairCrossings(geometry, ch, cv, {}, {}, params);
            QCOMPARE(plain.events.size(), std::size_t{1});
            const Crossing& event = plain.events.front();
            const int endBit = ch.psi.back() > event.psiH ? 1 : 2;
            QCOMPARE(event.terminalSides & endBit, 0);
            QVERIFY(event.terminalSides != 0);
        }
        // A gated encounter on the crossing's OWN segment, just past the
        // crossing: a V folded in height whose returning limb sits under
        // the radius gate, 0.002 turn past the limb the H crosses. Both
        // limbs fall within the crossing's own H segment (the H runs on a
        // few more); the event is terminal neither way.
        {
            World world;
            const std::size_t h = addH(world, 0.05, kSeamWinding + kSeamOvershoot, kSeamHeight);
            FiberTrace vFiber;
            vFiber.hvTag = 'V';
            const double gatedW = kSeamWinding + 0.002;
            for (double z = 29000.0; z <= 31000.0 + 1e-9; z += 25.0) {
                vFiber.theta.push_back(kTwoPi * kSeamWinding);
                vFiber.z.push_back(z);
                vFiber.radius.push_back(sheetR(kSeamWinding, z) - 200.0);
            }
            for (double z = 31000.0; z >= 29000.0 - 1e-9; z -= 25.0) {
                vFiber.theta.push_back(kTwoPi * gatedW);
                vFiber.z.push_back(z);
                vFiber.radius.push_back(0.5 * params.minUmbilicusRadiusVx);
            }
            world.fibers.push_back(std::move(vFiber));
            world.trueM.push_back(0);
            const std::size_t v = world.fibers.size() - 1;
            const int chirality = inferChirality(world.fibers, params.chiralityOverride);
            const CanonicalTrace ch = canonicalizeTrace(world.fibers[h], chirality);
            const CanonicalTrace cv = canonicalizeTrace(world.fibers[v], chirality);
            const PairDetections geometry = detectPairCrossings(ch, cv, params);
            QCOMPARE(geometry.raw.size(), std::size_t{1});
            QVERIFY(!geometry.uncoveredSegments.empty());
            const PairCrossings plain = classifyPairCrossings(geometry, ch, cv, {}, {}, params);
            QCOMPARE(plain.events.size(), std::size_t{1});
            QVERIFY(std::find(geometry.uncoveredSegments.begin(), geometry.uncoveredSegments.end(),
                              plain.events.front().hSegment) != geometry.uncoveredSegments.end());
            QVERIFY(!plain.events.front().terminal);
            QCOMPARE(plain.events.front().terminalSides, 0);
        }
    }

    // A tagged H fiber linked to one V fiber is read against that V only: a
    // second V on the kollesis that its end also overruns, a thickness
    // behind it, keeps its radial reading. Unlinked, nothing is read.
    void kollesisLinkedEndReadsOnlyItsLinkedV()
    {
        for (const bool linked : {false, true}) {
            World world;
            const std::size_t h = addH(world, 0.05, kSeamWinding + kSeamOvershoot, kSeamHeight);
            const std::size_t vLinked = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
            const std::size_t vOther =
                addV(world, kSeamWinding + 0.5 * kSeamOvershoot, 29000.0, 31000.0, -200.0);
            world.fibers[vLinked].onKollesis = true;
            world.fibers[vOther].onKollesis = true;
            const std::size_t hEnd = world.fibers[h].theta.size() - 1;
            world.fibers[h].kollesisEndSample = hEnd;
            if (linked) {
                world.links.push_back(LinkInput{h, hEnd, vLinked, 40});
            }
            const SolveResult result =
                solveWindings(world.fibers, world.links, SolverParams{});
            QCOMPARE(result.events.size(), std::size_t{2});
            for (const Crossing& event : result.events) {
                const bool other = event.vFiber == vOther;
                QCOMPARE(event.kollesis, linked && !other);
            }
            QCOMPARE(result.kollesisCrossingCount, linked ? 1 : 0);
        }
    }

    // An H fiber whose end climbs steeply through the V fiber meets it at a
    // shallow angle: the encounter is a tangential detection only. It is
    // still the seam encounter, read and flagged like a transversal one.
    void kollesisShallowEncounterIsRead()
    {
        for (const bool flagged : {false, true}) {
            World world;
            FiberTrace hFiber;
            hFiber.hvTag = 'H';
            for (double w = 0.05; w < kSeamWinding - 0.001; w += 1.0 / 256.0) {
                hFiber.theta.push_back(kTwoPi * w);
                hFiber.z.push_back(kSeamHeight);
                hFiber.radius.push_back(sheetR(w, kSeamHeight) - kSheetStep);
            }
            // 0.0002 turn across the V fiber's angle while climbing 600 vx:
            // a few percent transversality.
            const double climbTop = kSeamHeight + 600.0;
            for (const double dw : {-0.0001, 0.0001}) {
                hFiber.theta.push_back(kTwoPi * (kSeamWinding + dw));
                hFiber.z.push_back(dw < 0.0 ? kSeamHeight : climbTop);
                hFiber.radius.push_back(sheetR(kSeamWinding, kSeamHeight + 300.0) - kSheetStep);
            }
            hFiber.kollesisEndSample = hFiber.theta.size() - 1;
            world.fibers.push_back(std::move(hFiber));
            world.trueM.push_back(0);
            const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
            world.fibers[v].onKollesis = flagged;
            world.links.push_back(LinkInput{0, world.fibers[0].theta.size() - 1, v, 40});
            const SolveResult result =
                solveWindings(world.fibers, world.links, SolverParams{});
            QCOMPARE(result.events.size(), std::size_t{1});
            const Crossing& event = result.events.front();
            QVERIFY(event.tangential);
            QVERIFY(event.deltaR > 0.0);
            QCOMPARE(event.kollesis, flagged);
            QCOMPARE(event.kind, flagged ? CrossingKind::Inside : CrossingKind::Outside);
            QCOMPARE(result.kollesisCrossingCount, flagged ? 1 : 0);
            QCOMPARE(result.crossings.size(), std::size_t{1});
            QCOMPARE(result.crossings.front().kollesis, flagged);
        }
    }

    // Exactly parallel owner segments leave an intersection the detector
    // cannot place; the translate is marked unresolved and gets no verdict.
    void parallelOwnersMarkTheTranslateUnresolved()
    {
        World world;
        const double b = 100.0;
        addHairpinH(world, 30000.0, -3.0, 3.0, b, 0.03, -kSheetStep);
        // The V fiber of the mixed case runs up the ray (its samples offset so
        // none sits at the H fiber's height: the three crossings are clean),
        // then returns down beside it and, at the H fiber's height, steps
        // flat across: two samples at z = 30000, exactly collinear with the H
        // segments they overlap - an intersection the detector cannot place.
        // Only that unresolved step stands between the group and a verdict.
        FiberTrace v;
        v.hvTag = 'V';
        const double radius = hairpinRadius(kSqrt3, b) - 300.0;
        for (double z = 29012.5; z <= 31000.0; z += 25.0) {
            v.theta.push_back(kHairpinTheta0);
            v.z.push_back(z);
            v.radius.push_back(radius);
        }
        for (double z = 31000.0; z >= 30000.0 - 1e-9; z -= 25.0) {
            v.theta.push_back(kHairpinTheta0 + 0.4);
            v.z.push_back(z);
            v.radius.push_back(radius);
        }
        v.theta.push_back(kHairpinTheta0 + 0.3);
        v.z.push_back(30000.0);
        v.radius.push_back(radius);
        world.fibers.push_back(std::move(v));
        world.trueM.push_back(-1);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QVERIFY(result.unresolvedIntersectionCount > 0);
        QVERIFY(!result.groups.empty());
        bool sawEligibleButUnresolved = false;
        for (const CrossingGroup& group : result.groups) {
            QVERIFY(!group.hasVerdict);
            if (group.multiplicity >= 3 && group.mixedSigns && group.orientationSum % 2 != 0 &&
                group.traversalCovered && !group.coverageGap) {
                QVERIFY(group.unresolved);
                sawEligibleButUnresolved = true;
            }
        }
        QVERIFY(sawEligibleButUnresolved);
    }
};

QTEST_APPLESS_MAIN(TestFiberWindingSolver)
#include "test_fiber_winding_solver.moc"
