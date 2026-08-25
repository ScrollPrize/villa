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
#include <vector>

#include "FiberWindingSolver.hpp"

using vc3d::fiber_map::winding::ComponentAnchor;
using vc3d::fiber_map::winding::Crossing;
using vc3d::fiber_map::winding::CrossingKind;
using vc3d::fiber_map::winding::CrossingStatus;
using vc3d::fiber_map::winding::FiberTrace;
using vc3d::fiber_map::winding::LinkInput;
using vc3d::fiber_map::winding::SolveResult;
using vc3d::fiber_map::winding::SolverParams;
using vc3d::fiber_map::winding::solveWindings;

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
        // One tie per V fiber (its own winding's pass), no repairs.
        QCOMPARE(result.tieCount, 3);
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
        QCOMPARE(result.tieCount, 3);
        QCOMPARE(result.droppedCrossingCount, 0);
    }

    void sameWindingPairReadsAsTieNotSeparation()
    {
        World world;
        const std::size_t h = addH(world, 0.05, 0.55, 30000.0);
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(result.crossings.size(), std::size_t{1});
        QCOMPARE(result.crossings.front().kind, CrossingKind::Tie);
        const double wh =
            solvedW(result, world, h, hSampleAt(world, h, 0.05, 0.3));
        const double wv = solvedW(result, world, v, 0);
        QVERIFY2(std::abs(wh - wv) < 0.01,
                 qPrintable(QStringLiteral("wh %1 wv %2").arg(wh).arg(wv)));
    }

    // A measured-outside pass inside the sheet-thickness band is still
    // same-winding evidence: a true adjacent winding sits a wrap away, not a
    // sub-band |dr| away.
    void outsideMeasurementInsideTheBandIsATie()
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
        QCOMPARE(result.crossings.front().kind, CrossingKind::Tie);
        const double wh =
            solvedW(result, world, h, hSampleAt(world, h, 0.05, 0.3));
        QVERIFY(std::abs(wh - solvedW(result, world, v, 0)) < 0.01);
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
        // Follows its sheet for half a turn, then drifts one sheet inward:
        // the second passes of both V fibers contradict the first passes.
        const auto drifting = [](double w, double z) {
            return (w < 1.8 ? sheetR(w, z) : sheetR(w - 1.0, z)) - kSheetStep;
        };
        const std::size_t h = addH(world, 1.2, 2.4, 30000.0, drifting);
        addV(world, 1.3, 27000.0, 33000.0);
        addV(world, 1.35, 27000.0, 33000.0);
        // The drifting fiber defeats data-driven chirality on purpose.
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

    // An untrusted fiber (no model-traced span) must lose a repair conflict
    // it would win on raw confidence, and its dropped crossing is not
    // declarable. Cycle: H ties trusted V1 (conf ~0.9), passes far outside
    // untrusted V2 (raw conf ~1.0, attenuated ~0.5), link V1=V2 (~0.75).
    // Unattenuated, the tie would drop; attenuated, the untrusted crossing
    // does, and V2 lands on the shared winding.
    void untrustedEvidenceLosesConflicts()
    {
        World world;
        const std::size_t h = addH(world, 0.05, 0.55, 30000.0);
        const std::size_t v1 = addV(world, 0.3, 27000.0, 33000.0);
        // Same angle as V1, drawn far inside the sheet: H reads as passing
        // well outside it.
        const std::size_t v2 = addV(world, 0.3, 27000.0, 33000.0, -600.0);
        world.fibers[v2].trusted = false;
        world.links.push_back(LinkInput{v1, 0, v2, 0});
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(countDroppedCrossings(result), 1);
        QVERIFY(result.droppedLinks.empty());
        bool sawDroppedUndeclarable = false;
        for (const Crossing& crossing : result.crossings) {
            if (crossing.status == CrossingStatus::Dropped) {
                QVERIFY(!crossing.declarable);
                sawDroppedUndeclarable = true;
            }
        }
        QVERIFY(sawDroppedUndeclarable);
        // The surviving tie and link put all three on one winding.
        QCOMPARE(result.placements[v1].turns - result.placements[h].turns, 0.0);
        QCOMPARE(result.placements[v2].turns - result.placements[v1].turns, 0.0);
        // And no drift declaration: the only drop involved untrusted geometry.
        QVERIFY(!result.placements[h].sheetDriftSuspect);
    }

    // Greedy repair can drop an innocent constraint before the real culprit
    // falls in a later cycle, leaving a drop the final map satisfies anyway.
    // Such repair debris carries violationTurns ~0 and is not declared; the
    // culprit's drop is violated by a full winding. Fixture: H ties A on two
    // successive turns (mutually contradictory - the second pass rides a
    // flattened tail), while a linked helper H2 independently ties A at the
    // first turn. Whichever innocent first-turn tie the repair sacrifices
    // first, the reinforced first-turn relation survives through the other
    // path, the second-turn culprit falls next, and the sacrificed tie ends
    // up satisfied by the very map that dropped it.
    void satisfiedDropsAreRepairDebrisNotErrors()
    {
        World world;
        const auto flattening = [](double w, double z) {
            return w < 1.0 ? sheetR(w, z) - kSheetStep
                           : sheetR(0.3, z) + 100.0;
        };
        const std::size_t h = addH(world, 0.05, 1.7, 30000.0, flattening);
        const std::size_t h2 = addH(world, 0.05, 0.55, 31000.0);
        const std::size_t a = addV(world, 0.3, 27000.0, 33000.0, 20.0);
        // Same angle, same start: sample 13 of both is the same ray.
        world.links.push_back(LinkInput{h, 13, h2, 13});
        SolverParams params;
        params.chiralityOverride = 1;  // the flattened tail defeats inference
        const SolveResult result = solveWindings(world.fibers, world.links, params);
        // Everything settles on the first-turn relation.
        QCOMPARE(result.placements[h2].turns, result.placements[h].turns);
        QCOMPARE(result.placements[a].turns, result.placements[h].turns);
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

    // The drift declaration itself requires trusted geometry: the same
    // contradictions raised by an untrusted H fiber are expected
    // interpolation noise, not a tag.
    void untrustedFibersAreNeverDriftSuspects()
    {
        World world;
        const auto drifting = [](double w, double z) {
            return (w < 1.8 ? sheetR(w, z) : sheetR(w - 1.0, z)) - kSheetStep;
        };
        const std::size_t h = addH(world, 1.2, 2.4, 30000.0, drifting);
        world.fibers[h].trusted = false;
        addV(world, 1.3, 27000.0, 33000.0);
        addV(world, 1.35, 27000.0, 33000.0);
        SolverParams params;
        params.chiralityOverride = 1;
        const SolveResult result =
            solveWindings(world.fibers, world.links, params);
        QVERIFY(countDroppedCrossings(result) > 0);
        QVERIFY(!result.placements[h].sheetDriftSuspect);
        for (const Crossing& crossing : result.crossings) {
            QVERIFY(!crossing.declarable);
        }
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
        QCOMPARE(result.crossings.front().kind, CrossingKind::Tie);
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
        QCOMPARE(result.crossings.front().kind, CrossingKind::Tie);
        QCOMPARE(result.crossings.front().mergedCount, 2);
        checkRelativeTurns(result, world, {h, world.fibers.size() - 1});
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
};

QTEST_APPLESS_MAIN(TestFiberWindingSolver)
#include "test_fiber_winding_solver.moc"
