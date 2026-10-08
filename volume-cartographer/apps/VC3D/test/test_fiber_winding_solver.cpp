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

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <limits>
#include <vector>

#include "FiberWindingSolver.hpp"
#include "FiberMapBentHits.hpp"
#include "FiberMapBentRays.hpp"

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


// --- Bent-reading fixtures (see winding::BentHit and BentAssembly).
using vc3d::fiber_map::winding::BentAssembly;
using vc3d::fiber_map::winding::BentClassification;
using vc3d::fiber_map::winding::BentDisposition;
using vc3d::fiber_map::winding::BentHit;
using vc3d::fiber_map::winding::BentStretchDecision;
using vc3d::fiber_map::winding::PairDetection;

void reverseFiber(FiberTrace& fiber)
{
    std::reverse(fiber.theta.begin(), fiber.theta.end());
    std::reverse(fiber.radius.begin(), fiber.radius.end());
    std::reverse(fiber.z.begin(), fiber.z.end());
}

struct BentSolve {
    SolveResult result;
    // Indices into result.crossings of the bent records, in order.
    std::vector<std::size_t> bentIndices;
    const Crossing* bentRecord(std::size_t k) const
    {
        return k < bentIndices.size() ? &result.crossings[bentIndices[k]] : nullptr;
    }
    std::size_t bentIndex(std::size_t k) const { return bentIndices[k]; }
};

// A world with its canonical traces and one assembly per fiber: one usable
// stretch over the whole fiber, anchored by links.
struct BentFixture {
    World world;
    int chirality = 1;
    std::vector<CanonicalTrace> canonical;
    std::vector<BentAssembly> assemblies;
    // Per fiber and sample: the inwardness the assembly reports (NaN: no
    // final normal); solve() interpolates it linearly between samples.
    std::vector<std::vector<double>> inwardness;

    explicit BentFixture(World w, int chiralityOverride = 0) : world(std::move(w))
    {
        SolverParams params;
        chirality = inferChirality(world.fibers, chiralityOverride);
        for (const FiberTrace& fiber : world.fibers) {
            canonical.push_back(canonicalizeTrace(fiber, chirality));
            BentAssembly assembly;
            BentStretchDecision decision;
            decision.firstSample = 0;
            decision.lastSample = fiber.theta.size() - 1;
            decision.finalSign = 1;
            decision.anchor = 2;
            decision.withheld = false;
            decision.disposition = BentDisposition::Usable;
            assembly.stretches.push_back(decision);
            assembly.radialInverted.assign(fiber.theta.size(), 0);
            assemblies.push_back(std::move(assembly));
            inwardness.emplace_back(fiber.theta.size(), std::numeric_limits<double>::quiet_NaN());
        }
    }

    // A raw hit of the pair (h, v): the owner's seeds and across, the other
    // fiber's segment and t. The accumulated angle is the physical one: the
    // short way round from the owner's position to the other's, in the
    // world's raw angles.
    BentHit hit(std::size_t h, std::size_t v, bool ownerIsV, std::size_t seedA, std::size_t seedB,
                double across, std::size_t segment, double t, double s, int side,
                double transversality = 1.0, double hitX = 0.0) const
    {
        const FiberTrace& owner = world.fibers[ownerIsV ? v : h];
        const FiberTrace& other = world.fibers[ownerIsV ? h : v];
        const double thetaOwner =
            owner.theta[seedA] + across * (owner.theta[seedB] - owner.theta[seedA]);
        const double thetaOther =
            other.theta[segment] + t * (other.theta[segment + 1] - other.theta[segment]);
        double delta = std::fmod(thetaOther - thetaOwner, kTwoPi);
        if (delta > M_PI) {
            delta -= kTwoPi;
        } else if (delta <= -M_PI) {
            delta += kTwoPi;
        }
        BentHit record;
        record.ownerIsV = ownerIsV;
        record.stretch = 0;
        record.side = side;
        record.seedA = seedA;
        record.seedB = seedB;
        record.across = across;
        record.s = s;
        record.theta = delta;
        record.segment = segment;
        record.t = t;
        record.transversality = transversality;
        record.hitX = hitX;
        record.selfCrossingS = std::numeric_limits<double>::infinity();
        return record;
    }

    // The translate the classification must assign a hit: the whole-turn
    // gap between the canonical gauges at the hit, the lift taken out.
    long long expectedN(std::size_t h, std::size_t v, const BentHit& hit) const
    {
        const CanonicalTrace& owner = canonical[hit.ownerIsV ? v : h];
        const CanonicalTrace& other = canonical[hit.ownerIsV ? h : v];
        const double psiOwner =
            owner.psi[hit.seedA] + hit.across * (owner.psi[hit.seedB] - owner.psi[hit.seedA]);
        const double psiOther = other.psi[hit.segment] +
                                hit.t * (other.psi[hit.segment + 1] - other.psi[hit.segment]);
        const double psiH = hit.ownerIsV ? psiOther : psiOwner;
        const double psiV = hit.ownerIsV ? psiOwner : psiOther;
        const double ownerSign = hit.ownerIsV ? -1.0 : 1.0;
        return std::llround((psiV - psiH - ownerSign * chirality * hit.theta) / kTwoPi);
    }

    // Every H-V pair detected and classified, with the given raw hits
    // appended to the named pairs' shards, then solved.
    BentSolve solve(const std::map<std::pair<std::size_t, std::size_t>, std::vector<BentHit>>& hits,
                    bool seamPair = false) const
    {
        SolverParams params;
        std::vector<PairDetections> shards;
        std::vector<std::pair<std::size_t, std::size_t>> pairs;
        for (std::size_t h = 0; h < canonical.size(); ++h) {
            if (canonical[h].hvTag != 'H') {
                continue;
            }
            for (std::size_t v = 0; v < canonical.size(); ++v) {
                if (canonical[v].hvTag != 'V') {
                    continue;
                }
                shards.push_back(detectPairCrossings(canonical[h], canonical[v], params));
                const auto extra = hits.find({h, v});
                if (extra != hits.end()) {
                    shards.back().bentHits = extra->second;
                }
                pairs.emplace_back(h, v);
            }
        }
        std::vector<BentAssembly> local = assemblies;
        for (std::size_t f = 0; f < local.size(); ++f) {
            local[f].inwardnessBetween = [values = inwardness[f]](std::size_t i0, std::size_t i1,
                                                                   double t) {
                if (i0 >= values.size() || i1 >= values.size() || std::isnan(values[i0]) ||
                    std::isnan(values[i1])) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                return values[i0] + t * (values[i1] - values[i0]);
            };
        }
        std::vector<PairCrossings> classified;
        std::vector<BentClassification> inputs;
        inputs.reserve(pairs.size());
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            inputs.push_back(BentClassification{&local[pairs[i].first], &local[pairs[i].second],
                                                chirality, seamPair});
        }
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            classified.push_back(classifyPairCrossings(shards[i], canonical[pairs[i].first],
                                                       canonical[pairs[i].second], {}, {}, params,
                                                       &inputs[i]));
        }
        std::vector<PairDetection> detections;
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            detections.push_back({pairs[i].first, pairs[i].second, &classified[i]});
        }
        BentSolve out;
        out.result = solveWindings(world.fibers, world.links, params, chirality, detections);
        for (std::size_t i = 0; i < out.result.crossings.size(); ++i) {
            if (out.result.crossings[i].bent) {
                out.bentIndices.push_back(i);
            }
        }
        return out;
    }
};

// A trace authored from 3D points (the frame of the hand-made curtains: the
// umbilicus along z through the origin).
FiberTrace traceOf(const std::vector<cv::Vec3d>& points, char tag)
{
    FiberTrace fiber;
    fiber.hvTag = tag;
    for (const cv::Vec3d& p : points) {
        fiber.theta.push_back(std::atan2(p[1], p[0]));
        fiber.radius.push_back(std::hypot(p[0], p[1]));
        fiber.z.push_back(p[2]);
    }
    // FiberTrace takes unwrapped angles. Start at the canonical endpoint
    // so rotating a fixture across atan2's cut also preserves its gauge
    // under reversal (including a median exactly at pi).
    if (!points.empty()) {
        const bool forward = vc3d::fiber_map::bent::canonicalForward(points, 0, points.size() - 1);
        for (std::size_t k = 1; k < points.size(); ++k) {
            const auto i = forward ? k : points.size() - 1 - k;
            const auto previous = forward ? i - 1 : i + 1;
            fiber.theta[i] = fiber.theta[previous] + std::remainder(fiber.theta[i] - fiber.theta[previous], kTwoPi);
        }
    }
    return fiber;
}

// Everything a bent record exports or constrains with, from the
// representatives then from the events (with their indices), in order.
std::vector<double> bentTerms(const SolveResult& result)
{
    std::vector<double> terms;
    const auto exportOf = [&](const Crossing& c) {
        terms.push_back(static_cast<double>(c.n));
        terms.push_back(c.kind == CrossingKind::Inside ? 0.0 : 1.0);
        terms.push_back(c.withheld ? 1.0 : 0.0);
        terms.push_back(static_cast<double>(c.bentReason));
        terms.push_back(static_cast<double>(c.status));
        terms.push_back(c.violationTurns);
        terms.push_back(c.confidence);
        terms.push_back(c.transversality);
        terms.push_back(c.hitX);
        terms.push_back(c.hitY);
        terms.push_back(c.hitZ);
        terms.push_back(c.rayLengthVx);
        terms.push_back(c.curtainFromV ? 1.0 : 0.0);
        terms.push_back(c.touch ? 1.0 : 0.0);
        terms.push_back(c.tangential ? 1.0 : 0.0);
        terms.push_back(c.psiH);
        terms.push_back(c.zVx);
        terms.push_back(static_cast<double>(c.anchor));
        terms.push_back(c.deltaR);
        terms.push_back(c.turnDeg);
        terms.push_back(c.minConditioning);
    };
    for (const Crossing& c : result.crossings) {
        if (c.bent) {
            exportOf(c);
        }
    }
    for (std::size_t e = 0; e < result.events.size(); ++e) {
        const Crossing& c = result.events[e];
        if (c.bent) {
            terms.push_back(static_cast<double>(e));
            exportOf(c);
        }
    }
    return terms;
}

// A hand-made curtain (one stretch of rays with sample provenance) crossed
// by a V polyline, through the classification and the solve under the four
// storage orders of the owner (its rays and samples reversed) and the
// crosser; every exported term must agree. Returns the forward solve.
struct CurtainCase {
    std::vector<vc3d::fiber_map::bent::BentRay> rays;
    std::vector<cv::Vec3d> hPoints;
    std::vector<cv::Vec3d> vPoints;
    // The curtain is the V's (its rays seeded at V samples, the H the
    // crosser) instead of the H's.
    bool curtainFromV = false;
    // The H is control-point interpolation only (no model-traced span):
    // its evidence is attenuated.
    bool untrustedH = false;
    // Links between the H (fiber 0) and the V (fiber 1), by sample index in
    // the given storage order (re-indexed under reversal).
    std::vector<LinkInput> links;
};

BentSolve solveCurtainUnderReversals(const CurtainCase& given, bool* ok)
{
    using vc3d::fiber_map::bent::BentCurtain;
    using vc3d::fiber_map::bent::BentRay;
    using vc3d::fiber_map::bent::BentStretch;
    using vc3d::fiber_map::bent::CurtainHit;
    using vc3d::fiber_map::bent::intersectCurtain;
    using vc3d::fiber_map::bentHitFromCurtain;
    *ok = false;
    std::vector<std::vector<double>> terms;
    std::optional<BentSolve> forward;
    for (const bool reverseH : {false, true}) {
        for (const bool reverseV : {false, true}) {
            std::vector<BentRay> rays = given.rays;
            std::vector<cv::Vec3d> hPoints = given.hPoints;
            std::vector<cv::Vec3d> vPoints = given.vPoints;
            const bool reverseOwner = given.curtainFromV ? reverseV : reverseH;
            const std::size_t ownerCount = given.curtainFromV ? vPoints.size() : hPoints.size();
            if (reverseOwner) {
                std::reverse(rays.begin(), rays.end());
                for (BentRay& ray : rays) {
                    ray.startSample = ownerCount - 1 - ray.startSample;
                }
            }
            if (reverseH) {
                std::reverse(hPoints.begin(), hPoints.end());
            }
            if (reverseV) {
                std::reverse(vPoints.begin(), vPoints.end());
            }
            BentCurtain curtain;
            BentStretch stretch;
            stretch.rays = rays;
            stretch.firstSample = 0;
            stretch.lastSample = ownerCount - 1;
            curtain.stretches.push_back(stretch);
            World world;
            world.fibers.push_back(traceOf(hPoints, 'H'));
            world.fibers.back().trusted = !given.untrustedH;
            world.fibers.push_back(traceOf(vPoints, 'V'));
            world.trueM = {0, 0};
            for (LinkInput link : given.links) {
                const auto reindex = [&](std::size_t fiber, std::size_t point) {
                    const bool reversed = fiber == 0 ? reverseH : reverseV;
                    const std::size_t count = fiber == 0 ? hPoints.size() : vPoints.size();
                    return reversed ? count - 1 - point : point;
                };
                link.pointA = reindex(link.fiberA, link.pointA);
                link.pointB = reindex(link.fiberB, link.pointB);
                world.links.push_back(link);
            }
            const BentFixture fixture(world, 1);
            const auto selfCrossings = vc3d::fiber_map::bent::curtainSelfCrossings(
                curtain, given.curtainFromV ? vPoints : hPoints, world.fibers[given.curtainFromV ? 1 : 0].theta);
            std::vector<BentHit> hits;
            for (const CurtainHit& hit : intersectCurtain(curtain, given.curtainFromV ? hPoints : vPoints)) {
                hits.push_back(bentHitFromCurtain(curtain, hit, given.curtainFromV, selfCrossings));
            }
            BentSolve solved = fixture.solve({{{0, 1}, hits}});
            std::vector<double> allTerms = bentTerms(solved.result);
            // The straight events too: their constraint, height, status and
            // terminal flags (the sides are about the lifted angle's
            // direction, the same under either storage order).
            for (std::size_t e = 0; e < solved.result.events.size(); ++e) {
                const Crossing& c = solved.result.events[e];
                if (c.bent) {
                    continue;
                }
                allTerms.push_back(static_cast<double>(e));
                allTerms.push_back(static_cast<double>(c.n));
                allTerms.push_back(c.kind == CrossingKind::Inside ? 0.0 : 1.0);
                allTerms.push_back(c.psiH);
                allTerms.push_back(c.zVx);
                allTerms.push_back(static_cast<double>(c.status));
                allTerms.push_back(c.violationTurns);
                allTerms.push_back(c.terminal ? 1.0 : 0.0);
                allTerms.push_back(static_cast<double>(c.terminalSides));
            }
            terms.push_back(allTerms);
            if (!forward) {
                forward = std::move(solved);
            }
        }
    }
    for (std::size_t i = 1; i < terms.size(); ++i) {
        if (terms[i].size() != terms[0].size()) {
            qWarning("storage order %zu: %zu terms vs %zu", i, terms[i].size(), terms[0].size());
            return *forward;
        }
        for (std::size_t k = 0; k < terms[0].size(); ++k) {
            if (std::abs(terms[i][k] - terms[0][k]) > 1e-9) {
                qWarning("storage order %zu term %zu: %.17g vs %.17g", i, k, terms[i][k], terms[0][k]);
                return *forward;
            }
        }
    }
    *ok = true;
    return *forward;
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

    // A traced V polyline steps back in height by fractions of a voxel. Such
    // a back-step is no fold apex: the limbs it cuts count as one traversal,
    // so a verdict group is not broken into a singleton and an even rest (the
    // kb-159 / lt-165 rings of PHerc0139, where a 0.9 vx back-step between
    // the first and second of five crossings left two red rings). A back-step
    // deeper than the prominence is a fold and still cuts.
    void subVoxelBackStepDoesNotCutATraversal()
    {
        for (const double backStep : {0.0, 0.9, 50.0}) {
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
            // A V fiber weaving across the H fiber's angle three times, on
            // the H fiber's own sheet for the first half and a thickness
            // further in for the second: inside, outside, outside - one
            // inside, an odd count, verdict Inside (same winding).
            FiberTrace v;
            v.hvTag = 'V';
            for (int i = 0; i <= 400; ++i) {
                const double z = 29000.0 + 2000.0 * i / 400.0;
                const double wH = 0.2 + 0.2 * i / 400.0;
                const double w = wH + 0.03 * std::sin(kTwoPi * 1.5 * i / 400.0 + 0.5);
                v.theta.push_back(kTwoPi * w);
                v.z.push_back(z);
                v.radius.push_back(sheetR(w, z) - (i > 200 ? 2.0 * kSheetStep : 0.0));
            }
            // The back-step: one sample between the first crossing (i ~ 112)
            // and the second (i ~ 245) dips below its predecessor.
            v.z[150] = v.z[149] - backStep;
            world.fibers.push_back(std::move(v));
            world.trueM.push_back(0);
            const SolveResult result =
                solveWindings(world.fibers, world.links, SolverParams{});
            QCOMPARE(result.crossings.size(), std::size_t{3});
            const std::size_t branches =
                canonicalizeTrace(world.fibers[1], result.chirality).branches.size();
            QCOMPARE(branches, backStep > 0.0 ? std::size_t{3} : std::size_t{1});
            if (backStep < 4.0) {
                // One run, one group, verdict taken; all three crossings
                // stand behind it.
                const CrossingGroup& group = singleGroup(result);
                QCOMPARE(group.multiplicity, 3);
                QCOMPARE(group.insideCount, 1);
                QVERIFY(group.mixedSigns);
                QVERIFY(group.orientationSum % 2 != 0);
                QVERIFY(group.traversalCovered);
                QVERIFY(!group.onCurtain);
                QVERIFY(group.hasVerdict);
                QCOMPARE(group.verdict, CrossingKind::Inside);
                QCOMPARE(group.vBranch, std::size_t{0});
                for (const Crossing& crossing : result.crossings) {
                    QCOMPARE(crossing.status, CrossingStatus::InGroup);
                    QCOMPARE(crossing.groupIndex, 0LL);
                }
                QCOMPARE(countDroppedCrossings(result), 0);
                checkRelativeTurns(result, world, {0, 1});
            } else {
                // A real fold: the limbs are counted apart, the first
                // crossing stands alone, the other two are too few for a
                // verdict, and the signs' conflict surfaces as before.
                for (const CrossingGroup& group : result.groups) {
                    QVERIFY(!group.hasVerdict);
                }
                for (const Crossing& crossing : result.crossings) {
                    QVERIFY(crossing.status != CrossingStatus::InGroup);
                }
                QVERIFY(countDroppedCrossings(result) >= 1);
            }
        }
    }

    // Jitter at the apices of a genuine fold: three 2000 vx limbs with a
    // one-voxel back-and-forth at both extrema, the middle limb a wrap
    // inward. Reading the extrema with hysteresis keeps the three limbs
    // apart (a two-neighbour rule would chain the whole polyline into one
    // run through the short branches and issue a verdict across limbs the
    // solver has never counted together): an H fiber crossing all three
    // gets three singletons, no verdict, and the conflict surfaces as
    // before.
    void jitterAtAFoldApexKeepsTheLimbsApart()
    {
        World world;
        FiberTrace h;
        h.hvTag = 'H';
        for (double theta = kHairpinTheta0 - 0.2; theta <= kHairpinTheta0 + 0.2 + 1e-9;
             theta += 0.002) {
            h.theta.push_back(theta);
            h.z.push_back(30000.0);
            h.radius.push_back(20000.0);
        }
        world.fibers.push_back(std::move(h));
        world.trueM.push_back(0);
        FiberTrace v;
        v.hvTag = 'V';
        const auto sample = [&v](double z, double radius) {
            v.theta.push_back(kHairpinTheta0);
            v.z.push_back(z);
            v.radius.push_back(radius);
        };
        for (double z = 29000.0; z <= 31000.0 + 1e-9; z += 25.0) {
            sample(z, 22000.0);
        }
        sample(30999.0, 22000.0);
        sample(31000.0, 22000.0);
        for (double z = 30975.0; z >= 29000.0 - 1e-9; z -= 25.0) {
            sample(z, 21000.0);
        }
        sample(29001.0, 21000.0);
        sample(29000.0, 21000.0);
        for (double z = 29025.0; z <= 31000.0 + 1e-9; z += 25.0) {
            sample(z, 19000.0);
        }
        world.fibers.push_back(std::move(v));
        world.trueM.push_back(0);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(canonicalizeTrace(world.fibers[1], result.chirality).branches.size(),
                 std::size_t{7});
        QCOMPARE(result.crossings.size(), std::size_t{3});
        for (const CrossingGroup& group : result.groups) {
            QVERIFY(!group.hasVerdict);
        }
        for (const Crossing& crossing : result.crossings) {
            QVERIFY(crossing.status != CrossingStatus::InGroup);
        }
        QVERIFY(countDroppedCrossings(result) >= 1);
    }

    // The same with a sloping H fiber and an angle step in the V fiber: the
    // three crossings of the one zigzag traversal land at different heights
    // (below, inside and above the back-step's height band), so only their
    // kinds' pattern - Outside, Inside, Inside - and the middle event's limb
    // tell them apart from three genuine crossings. The event on the
    // back-step limb withholds the verdict. All four sample orders.
    void slopingTraversalThroughABackStepTakesNoVerdict()
    {
        for (int variant = 0; variant < 4; ++variant) {
            World world;
            FiberTrace h;
            h.hvTag = 'H';
            std::vector<std::array<double, 3>> hSamples = {{-0.1, 29800.0, 20300.0},
                                                          {0.1, 30200.0, 20300.0}};
            std::vector<std::array<double, 3>> vSamples = {{-0.001, 29000.0, 20000.0},
                                                          {-0.001, 30000.5, 20200.0},
                                                          {0.001, 29999.5, 20600.0},
                                                          {0.001, 31000.0, 22000.0}};
            if (variant & 1) {
                std::reverse(hSamples.begin(), hSamples.end());
            }
            if (variant & 2) {
                std::reverse(vSamples.begin(), vSamples.end());
            }
            for (const auto& [theta, z, radius] : hSamples) {
                h.theta.push_back(theta);
                h.z.push_back(z);
                h.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(h));
            world.trueM.push_back(0);
            FiberTrace v;
            v.hvTag = 'V';
            for (const auto& [theta, z, radius] : vSamples) {
                v.theta.push_back(theta);
                v.z.push_back(z);
                v.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(v));
            world.trueM.push_back(0);
            SolverParams params;
            params.chiralityOverride = 1;
            const SolveResult result = solveWindings(world.fibers, world.links, params);
            QCOMPARE(canonicalizeTrace(world.fibers[1], result.chirality).branches.size(),
                     std::size_t{3});
            QCOMPARE(result.crossings.size(), std::size_t{3});
            const CrossingGroup& group = singleGroup(result);
            QCOMPARE(group.multiplicity, 3);
            QCOMPARE(group.insideCount, 2);
            QVERIFY(group.mixedSigns);
            QVERIFY(group.orientationSum % 2 != 0);
            QVERIFY(group.onCurtain);
            QVERIFY(!group.hasVerdict);
            for (const Crossing& crossing : result.crossings) {
                QVERIFY(crossing.status != CrossingStatus::InGroup);
            }
        }
    }

    // The encounter with a back-step limb may be read as a touch rather
    // than a crossing: here both back-step limbs are met within rounding of
    // their upper vertices (the intersections lie 2^-38 and 2^-37 inside),
    // detection snaps the hits to the vertices and the ray test reads
    // touches, which no group counts. The veto must still see them, or the
    // three counted full-limb events (Outside, Outside, Inside... read
    // Inside, Inside, Outside in one order) would take a verdict whose
    // inside parity is not the exact five-crossing count's. (With band
    // membership read from the H segment's height interval the touch is
    // covered twice over: its companions lie within 2^-38 of the band
    // edges, so any segment holding them meets the band as well.) All four
    // sample orders.
    void touchOnABackStepLimbVetoesTheVerdict()
    {
        const double e = std::ldexp(1.0, -38);
        const double f = std::ldexp(1.0, -37);
        for (int variant = 0; variant < 4; ++variant) {
            World world;
            std::vector<std::array<double, 3>> hSamples = {{-0.5, 20000.0, 20300.0},
                                                          {0.5, 40000.0, 20300.0}};
            std::vector<std::array<double, 3>> vSamples = {
                {-0.2, 20000.0, 20000.0},    {-0.05, 29000.0 - e, 20000.0},
                {0.0, 30000.0 + e, 20000.0}, {0.001, 29999.0, 22000.0},
                {0.2, 34000.0 - f, 22000.0}, {0.25, 35000.0 + f, 22000.0},
                {0.251, 34999.0, 22000.0},   {0.251, 40000.0, 22000.0}};
            if (variant & 1) {
                std::reverse(hSamples.begin(), hSamples.end());
            }
            if (variant & 2) {
                std::reverse(vSamples.begin(), vSamples.end());
            }
            FiberTrace h;
            h.hvTag = 'H';
            for (const auto& [theta, z, radius] : hSamples) {
                h.theta.push_back(theta);
                h.z.push_back(z);
                h.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(h));
            world.trueM.push_back(0);
            FiberTrace v;
            v.hvTag = 'V';
            for (const auto& [theta, z, radius] : vSamples) {
                v.theta.push_back(theta);
                v.z.push_back(z);
                v.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(v));
            world.trueM.push_back(0);
            SolverParams params;
            params.chiralityOverride = 1;
            const SolveResult result = solveWindings(world.fibers, world.links, params);
            QCOMPARE(canonicalizeTrace(world.fibers[1], result.chirality).branches.size(),
                     std::size_t{5});
            int touches = 0;
            for (const Crossing& event : result.events) {
                touches += event.touch ? 1 : 0;
            }
            QVERIFY(touches >= 1);
            QCOMPARE(result.groups.size(), std::size_t{1});
            QVERIFY(result.groups.front().onCurtain);
            QVERIFY(!result.groups.front().hasVerdict);
            for (const Crossing& crossing : result.crossings) {
                QVERIFY(crossing.status != CrossingStatus::InGroup);
            }
        }
    }

    // A traversal through a back-step can miss the back-step limb itself
    // and meet the limb after twice instead, where that limb wobbles in
    // angle inside the back-step's height band: here the H fiber slips
    // under the one-voxel back-step, crosses the lower limb at 29999.7
    // (Outside, the radius steps across the back-step) and the upper limb
    // at 29999.5 and 30001.3 (Inside, Inside). Two of the three lie in the
    // band [29999, 30000]; the run takes no verdict (an Outside verdict
    // would stand on a count that is not one smoothed traversal's). Both
    // sample orders of the V fiber.
    void angleWobbleInsideTheBackStepBandTakesNoVerdict()
    {
        for (const bool reversed : {false, true}) {
            World world;
            FiberTrace h;
            h.hvTag = 'H';
            for (const auto& [theta, z, radius] :
                 {std::array{-0.2, 37999.7, 20300.0}, std::array{0.2, 21999.7, 20300.0}}) {
                h.theta.push_back(theta);
                h.z.push_back(z);
                h.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(h));
            world.trueM.push_back(0);
            std::vector<std::array<double, 3>> vSamples = {{0.0, 20000.0, 20000.0},
                                                          {0.0, 30000.0, 20000.0},
                                                          {0.00002, 29999.0, 22000.0},
                                                          {-0.00004, 30001.0, 22000.0},
                                                          {0.00002, 40000.0, 22000.0}};
            if (reversed) {
                std::reverse(vSamples.begin(), vSamples.end());
            }
            FiberTrace v;
            v.hvTag = 'V';
            for (const auto& [theta, z, radius] : vSamples) {
                v.theta.push_back(theta);
                v.z.push_back(z);
                v.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(v));
            world.trueM.push_back(0);
            SolverParams params;
            params.chiralityOverride = 1;
            const SolveResult result = solveWindings(world.fibers, world.links, params);
            QCOMPARE(canonicalizeTrace(world.fibers[1], result.chirality).branches.size(),
                     std::size_t{3});
            // Three events; the two Inside ones, 1.8 vx apart at one radial
            // gap, share a display representative.
            QCOMPARE(result.events.size(), std::size_t{3});
            const CrossingGroup& group = singleGroup(result);
            QCOMPARE(group.multiplicity, 3);
            QCOMPARE(group.insideCount, 2);
            QVERIFY(group.onCurtain);
            QVERIFY(!group.hasVerdict);
            for (const Crossing& crossing : result.crossings) {
                QVERIFY(crossing.status != CrossingStatus::InGroup);
            }
        }
    }

    // Nested jitter: a 3 vx back-step whose two ends each carry a half-voxel
    // wobble. The heights the run visits more than once span the whole
    // back-step [29997, 30000], not only the two half-voxel overlaps of
    // consecutive limbs; the H fiber slips through the middle, missing every
    // short limb, and meets the long limbs at 29998.5 (Outside), 29997.6
    // and 30000.1 (Inside, Inside). The bands are the counter-direction
    // limbs' own ranges - here the three descending limbs, of which the 3 vx
    // one spans the whole back-step - so the middle is covered and the run
    // takes no verdict. The H fiber is sampled every 0.1 vx of height around
    // the encounters, so each event's segment enclosure is tight and meets
    // only the 3 vx limb's range, not the two half-voxel overlaps: the
    // fixture depends on the bands being whole limb ranges. All four sample
    // orders.
    void nestedJitterIsCoveredWhole()
    {
        for (int variant = 0; variant < 4; ++variant) {
            World world;
            // Heights descending from 37998.5 to 21998.5 along theta = -0.2
            // .. 0.2 (z = 37998.5 - 40000 (theta + 0.2)).
            std::vector<double> heights = {37998.5, 34000.0, 31000.0};
            for (double z = 30002.05; z >= 29995.95 - 1e-9; z -= 0.1) {
                heights.push_back(z);
            }
            heights.insert(heights.end(), {29000.0, 26000.0, 21998.5});
            std::vector<std::array<double, 3>> hSamples;
            for (const double z : heights) {
                hSamples.push_back({(37998.5 - z) / 40000.0 - 0.2, z, 20300.0});
            }
            std::vector<std::array<double, 3>> vSamples = {
                {0.0, 20000.0, 20000.0},      {0.0, 30000.0, 20000.0},
                {0.0, 29999.5, 20000.0},      {0.0, 30000.0, 20000.0},
                {0.00005, 29997.0, 22000.0},  {0.00005, 29997.5, 22000.0},
                {0.00005, 29997.0, 22000.0},  {-0.00004, 29999.0, 22000.0},
                {0.00005, 40000.0, 22000.0}};
            if (variant & 1) {
                std::reverse(hSamples.begin(), hSamples.end());
            }
            if (variant & 2) {
                std::reverse(vSamples.begin(), vSamples.end());
            }
            FiberTrace h;
            h.hvTag = 'H';
            for (const auto& [theta, z, radius] : hSamples) {
                h.theta.push_back(theta);
                h.z.push_back(z);
                h.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(h));
            world.trueM.push_back(0);
            FiberTrace v;
            v.hvTag = 'V';
            for (const auto& [theta, z, radius] : vSamples) {
                v.theta.push_back(theta);
                v.z.push_back(z);
                v.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(v));
            world.trueM.push_back(0);
            SolverParams params;
            params.chiralityOverride = 1;
            const SolveResult result = solveWindings(world.fibers, world.links, params);
            QCOMPARE(canonicalizeTrace(world.fibers[1], result.chirality).branches.size(),
                     std::size_t{7});
            QCOMPARE(result.events.size(), std::size_t{3});
            const CrossingGroup& group = singleGroup(result);
            QCOMPARE(group.multiplicity, 3);
            QCOMPARE(group.insideCount, 2);
            QVERIFY(group.onCurtain);
            QVERIFY(!group.hasVerdict);
            for (const Crossing& crossing : result.crossings) {
                QVERIFY(crossing.status != CrossingStatus::InGroup);
            }
        }
    }

    // An event exactly at a band edge: the middle crossing of this
    // traversal lies at height 30000, the top of the band [29999, 30000],
    // halfway along an H segment whose interpolation reconstructs it a few
    // ulps above (30000.000000000007) in one sample order and exactly in
    // the other. Band membership is read from the H segment's height
    // interval, which encloses the crossing whatever the rounding, so the
    // veto holds in both orders.
    void bandEdgeIsReadWithinRounding()
    {
        for (const bool reversed : {false, true}) {
            World world;
            std::vector<std::array<double, 3>> hSamples = {
                {-58618.0 / 65536.0, 88616.0, 20300.0}, {13520.0 / 65536.0, 16478.0, 20300.0}};
            if (reversed) {
                std::reverse(hSamples.begin(), hSamples.end());
            }
            FiberTrace h;
            h.hvTag = 'H';
            for (const auto& [theta, z, radius] : hSamples) {
                h.theta.push_back(theta);
                h.z.push_back(z);
                h.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(h));
            world.trueM.push_back(0);
            FiberTrace v;
            v.hvTag = 'V';
            for (const auto& [theta, z, radius] :
                 {std::array{0.0, 20000.0, 20000.0}, std::array{0.0, 30000.0, 20000.0},
                  std::array{1.0 / 65536.0, 29999.0, 22000.0},
                  std::array{-5.0 / 65536.0, 30001.0, 22000.0},
                  std::array{1.0 / 65536.0, 40000.0, 22000.0}}) {
                v.theta.push_back(theta);
                v.z.push_back(z);
                v.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(v));
            world.trueM.push_back(0);
            SolverParams params;
            params.chiralityOverride = 1;
            const SolveResult result = solveWindings(world.fibers, world.links, params);
            QCOMPARE(result.events.size(), std::size_t{3});
            const CrossingGroup& group = singleGroup(result);
            QCOMPARE(group.multiplicity, 3);
            QCOMPARE(group.insideCount, 2);
            QVERIFY(group.onCurtain);
            QVERIFY(!group.hasVerdict);
        }
    }

    // The same at a nearly parallel crossing: the intersection parameter is
    // a determinant quotient whose cancellation puts the reconstructed
    // height a microvoxel above the band's top edge (exactly 30000 in the
    // reals), far beyond any epsilon envelope. The H segment's height
    // interval encloses it regardless. Both sample orders of the H fiber.
    void nearlyParallelBandEdgeIsEnclosed()
    {
        const double a = 10771629145.0 / 281474976710656.0;
        const double b = 10771629982.0 / 281474976710656.0;
        for (const bool reversed : {false, true}) {
            World world;
            std::vector<std::array<double, 3>> hSamples = {{-4529.0 * a, 25471.0, 20300.0},
                                                          {1832.0 * a, 31832.0, 20300.0}};
            if (reversed) {
                std::reverse(hSamples.begin(), hSamples.end());
            }
            FiberTrace h;
            h.hvTag = 'H';
            for (const auto& [theta, z, radius] : hSamples) {
                h.theta.push_back(theta);
                h.z.push_back(z);
                h.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(h));
            world.trueM.push_back(0);
            FiberTrace v;
            v.hvTag = 'V';
            for (const auto& [theta, z, radius] :
                 {std::array{0.0, 20000.0, 20000.0}, std::array{-0.01, 30000.0, 20000.0},
                  std::array{-b, 29999.0, 22000.0}, std::array{1000.0 * b, 31000.0, 22000.0},
                  std::array{-1.0, 40000.0, 22000.0}}) {
                v.theta.push_back(theta);
                v.z.push_back(z);
                v.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(v));
            world.trueM.push_back(0);
            SolverParams params;
            params.chiralityOverride = 1;
            const SolveResult result = solveWindings(world.fibers, world.links, params);
            QCOMPARE(result.events.size(), std::size_t{3});
            const CrossingGroup& group = singleGroup(result);
            QCOMPARE(group.multiplicity, 3);
            QVERIFY(group.onCurtain);
            QVERIFY(!group.hasVerdict);
        }
    }

    // A short FORWARD limb is no back-step (hendrikschilling, PR #1938): a V
    // fiber that climbs 3 vx, steps back 0.1 vx and climbs on is one run
    // whose repeated heights are the back-step's [29002.9, 29003] only. The
    // three crossings of its 3 vx first limb, between 29001 and 29002, lie
    // outside that band, and the limb's group keeps the verdict and the
    // placement the per-branch solver gave it. Both sample orders of the V
    // fiber (reversed, the short limb ends the run and the run descends).
    void shortForwardLimbKeepsItsVerdict()
    {
        for (const bool reversed : {false, true}) {
            World world;
            // An H fiber weaving across the V fiber's angle three times at
            // heights 29001, 29001.5 and 29002: inside, outside, outside.
            FiberTrace h;
            h.hvTag = 'H';
            for (const auto& [theta, z, radius] :
                 {std::array{-0.2, 29001.0, 19000.0}, std::array{0.2, 29001.0, 19000.0},
                  std::array{0.2, 29001.5, 21000.0}, std::array{-0.2, 29001.5, 21000.0},
                  std::array{-0.2, 29002.0, 21000.0}, std::array{0.2, 29002.0, 21000.0}}) {
                h.theta.push_back(theta);
                h.z.push_back(z);
                h.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(h));
            world.trueM.push_back(0);
            std::vector<std::array<double, 3>> vSamples = {{0.0, 29000.0, 20000.0},
                                                          {0.0, 29003.0, 20000.0},
                                                          {0.0, 29002.9, 20000.0},
                                                          {0.0, 29100.0, 20000.0}};
            if (reversed) {
                std::reverse(vSamples.begin(), vSamples.end());
            }
            FiberTrace v;
            v.hvTag = 'V';
            for (const auto& [theta, z, radius] : vSamples) {
                v.theta.push_back(theta);
                v.z.push_back(z);
                v.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(v));
            world.trueM.push_back(0);
            SolverParams params;
            params.chiralityOverride = 1;
            const SolveResult result = solveWindings(world.fibers, world.links, params);
            QCOMPARE(canonicalizeTrace(world.fibers[1], result.chirality).branches.size(),
                     std::size_t{3});
            QCOMPARE(result.events.size(), std::size_t{3});
            const CrossingGroup& group = singleGroup(result);
            QCOMPARE(group.multiplicity, 3);
            QCOMPARE(group.insideCount, 1);
            QVERIFY(!group.onCurtain);
            QVERIFY(group.hasVerdict);
            QCOMPARE(group.verdict, CrossingKind::Inside);
            QCOMPARE(countDroppedCrossings(result), 0);
            // The same placement as with the first limb alone.
            World limb = world;
            limb.fibers[1].theta.resize(2);
            limb.fibers[1].z.resize(2);
            limb.fibers[1].radius.resize(2);
            if (reversed) {
                limb.fibers[1].theta = {0.0, 0.0};
                limb.fibers[1].z = {29003.0, 29000.0};
                limb.fibers[1].radius = {20000.0, 20000.0};
            }
            const SolveResult alone = solveWindings(limb.fibers, limb.links, params);
            QVERIFY(singleGroup(alone).hasVerdict);
            QCOMPARE(singleGroup(alone).verdict, CrossingKind::Inside);
            QCOMPARE(result.placements[1].turns - result.placements[0].turns,
                     alone.placements[1].turns - alone.placements[0].turns);
        }
    }

    // A run whose direction the hysteresis never fixes (the whole V fiber
    // is below the prominence in height: 29000 -> 29003 -> 29002) reads its
    // limbs against the sign of its end height minus its start height, so
    // the counter limb - and the band [29002, 29003] - is the same limb in
    // either sample order (a +1 default would band the long limb when the
    // samples are reversed). Three crossings of the long limb below the
    // band keep their verdict in both orders.
    void undeterminedRunReferenceIsOrderIndependent()
    {
        for (const bool reversed : {false, true}) {
            World world;
            FiberTrace h;
            h.hvTag = 'H';
            for (const auto& [theta, z, radius] :
                 {std::array{-0.2, 29000.5, 19000.0}, std::array{0.2, 29000.5, 19000.0},
                  std::array{0.2, 29001.0, 21000.0}, std::array{-0.2, 29001.0, 21000.0},
                  std::array{-0.2, 29001.5, 21000.0}, std::array{0.2, 29001.5, 21000.0}}) {
                h.theta.push_back(theta);
                h.z.push_back(z);
                h.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(h));
            world.trueM.push_back(0);
            std::vector<std::array<double, 3>> vSamples = {
                {0.0, 29000.0, 20000.0}, {0.0, 29003.0, 20000.0}, {0.0, 29002.0, 20000.0}};
            if (reversed) {
                std::reverse(vSamples.begin(), vSamples.end());
            }
            FiberTrace v;
            v.hvTag = 'V';
            for (const auto& [theta, z, radius] : vSamples) {
                v.theta.push_back(theta);
                v.z.push_back(z);
                v.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(v));
            world.trueM.push_back(0);
            SolverParams params;
            params.chiralityOverride = 1;
            const SolveResult result = solveWindings(world.fibers, world.links, params);
            QCOMPARE(canonicalizeTrace(world.fibers[1], result.chirality).branches.size(),
                     std::size_t{2});
            int counted = 0;
            for (const Crossing& event : result.events) {
                counted += (!event.touch && !event.tangential) ? 1 : 0;
            }
            QCOMPARE(counted, 3);
            const CrossingGroup& group = singleGroup(result);
            QCOMPARE(group.multiplicity, 3);
            QCOMPARE(group.insideCount, 1);
            QVERIFY(group.traversalCovered);
            QVERIFY(!group.onCurtain);
            QVERIFY(group.hasVerdict);
            QCOMPARE(group.verdict, CrossingKind::Inside);
            QCOMPARE(countDroppedCrossings(result), 0);
            QCOMPARE(result.placements[1].turns - result.placements[0].turns, 0.0);
        }
    }

    // Jitter at an apex that revisits the apex height: the run's extreme is
    // reached twice, and the stretch between the visits is a run of its
    // own, so it attaches to neither limb and the runs read the same in
    // either sample order of the V fiber. The theta = 0 limb, crossed three
    // times (Inside, Outside, Outside), keeps its Inside verdict in both
    // orders (attaching the jitter to it in one order would veto it there).
    void tiedApexHeightsAreReadOrderIndependently()
    {
        for (const bool reversed : {false, true}) {
            World world;
            FiberTrace h;
            h.hvTag = 'H';
            for (const auto& [theta, z, radius] :
                 {std::array{0.4, 29009.5, 19000.0}, std::array{0.1, 29009.5, 19000.0},
                  std::array{0.1, 29005.0, 19000.0}, std::array{-0.1, 29005.0, 19000.0},
                  std::array{-0.1, 29005.0, 21000.0}, std::array{0.1, 29005.0, 21000.0},
                  std::array{-0.1, 29005.0, 21000.0}}) {
                h.theta.push_back(theta);
                h.z.push_back(z);
                h.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(h));
            world.trueM.push_back(0);
            std::vector<std::array<double, 3>> vSamples = {
                {0.0, 29000.0, 20000.0}, {0.0, 29010.0, 20000.0}, {0.1, 29009.0, 20000.0},
                {0.2, 29010.0, 20000.0}, {0.2, 29000.0, 20000.0}};
            if (reversed) {
                std::reverse(vSamples.begin(), vSamples.end());
            }
            FiberTrace v;
            v.hvTag = 'V';
            for (const auto& [theta, z, radius] : vSamples) {
                v.theta.push_back(theta);
                v.z.push_back(z);
                v.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(v));
            world.trueM.push_back(0);
            SolverParams params;
            params.chiralityOverride = 1;
            const SolveResult result = solveWindings(world.fibers, world.links, params);
            QCOMPARE(canonicalizeTrace(world.fibers[1], result.chirality).branches.size(),
                     std::size_t{4});
            int verdicts = 0;
            for (const CrossingGroup& group : result.groups) {
                if (group.hasVerdict) {
                    ++verdicts;
                    QCOMPARE(group.verdict, CrossingKind::Inside);
                    QCOMPARE(group.multiplicity, 3);
                }
            }
            QCOMPARE(verdicts, 1);
        }
    }

    // A V fiber of one monotone branch shorter than the prominence in height
    // (a nearly level trace near a fold) is a run of one limb: nothing
    // overlaps, so it is no back-step and its group's verdict stands as it
    // did per branch. All four sample orders.
    void aShortSingleBranchIsNotABackStep()
    {
        for (int variant = 0; variant < 4; ++variant) {
            World world;
            std::vector<std::array<double, 3>> hSamples = {{-0.2, 29001.0, 19000.0},
                                                          {0.2, 29001.0, 19000.0},
                                                          {0.2, 29001.5, 21000.0},
                                                          {-0.2, 29002.0, 21000.0},
                                                          {0.2, 29002.0, 21000.0}};
            std::vector<std::array<double, 3>> vSamples = {{-0.05, 29000.0, 20000.0},
                                                          {0.05, 29003.0, 20000.0}};
            if (variant & 1) {
                std::reverse(hSamples.begin(), hSamples.end());
            }
            if (variant & 2) {
                std::reverse(vSamples.begin(), vSamples.end());
            }
            FiberTrace h;
            h.hvTag = 'H';
            for (const auto& [theta, z, radius] : hSamples) {
                h.theta.push_back(theta);
                h.z.push_back(z);
                h.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(h));
            world.trueM.push_back(0);
            FiberTrace v;
            v.hvTag = 'V';
            for (const auto& [theta, z, radius] : vSamples) {
                v.theta.push_back(theta);
                v.z.push_back(z);
                v.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(v));
            world.trueM.push_back(0);
            SolverParams params;
            params.chiralityOverride = 1;
            const SolveResult result = solveWindings(world.fibers, world.links, params);
            QCOMPARE(canonicalizeTrace(world.fibers[1], result.chirality).branches.size(),
                     std::size_t{1});
            const CrossingGroup& group = singleGroup(result);
            QCOMPARE(group.multiplicity, 3);
            QVERIFY(!group.onCurtain);
            QVERIFY(group.hasVerdict);
            QCOMPARE(group.verdict, CrossingKind::Inside);
        }
    }

    // A back-step at which the V fiber's angle also steps: the run's two
    // limbs lie at different angles below and above it. An H fiber that
    // ends just short of the lower limb's angle, within the clearance, has
    // an incomplete count; read against the limb that spans its height it
    // gets no coverage. (Joining the limbs' samples into one polyline
    // ordered by height would interpolate a locus between the two angles and
    // certify the clearance falsely.)
    void endpointClearanceIsReadPerLimb()
    {
        World world;
        FiberTrace h;
        h.hvTag = 'H';
        for (const auto& [theta, radius] : {std::pair{0.6, 19000.0}, std::pair{-0.2, 19000.0},
                                            std::pair{0.1, 21000.0}, std::pair{-0.02, 21000.0}}) {
            h.theta.push_back(theta);
            h.z.push_back(29050.0);
            h.radius.push_back(radius);
        }
        world.fibers.push_back(std::move(h));
        world.trueM.push_back(0);
        FiberTrace v;
        v.hvTag = 'V';
        for (const auto& [theta, z] : {std::pair{0.0, 29000.0}, std::pair{0.0, 29100.0},
                                       std::pair{0.4, 29099.0}, std::pair{0.4, 29200.0}}) {
            v.theta.push_back(theta);
            v.z.push_back(z);
            v.radius.push_back(20000.0);
        }
        world.fibers.push_back(std::move(v));
        world.trueM.push_back(0);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(canonicalizeTrace(world.fibers[1], result.chirality).branches.size(),
                 std::size_t{3});
        QCOMPARE(result.crossings.size(), std::size_t{3});
        const CrossingGroup& group = singleGroup(result);
        QCOMPARE(group.multiplicity, 3);
        QVERIFY(group.mixedSigns);
        QVERIFY(group.orientationSum % 2 != 0);
        QVERIFY(!group.traversalCovered);
        QVERIFY(!group.hasVerdict);
        for (const Crossing& crossing : result.crossings) {
            QVERIFY(crossing.status != CrossingStatus::InGroup);
        }
    }

    // Before the polyline has moved a prominence from its start, the
    // direction is open and the running lowest and highest extrema decide:
    // a 3 vx rise followed by a 6 vx fall is a genuine reversal although
    // neither extremum is a prominence from the start. The three limbs are
    // then not one run (an H crossing all three at a radius between theirs
    // would otherwise take an Outside verdict). Both sample orders.
    void startupReversalIsReadFromTheRunningExtrema()
    {
        for (const bool reversed : {false, true}) {
            World world;
            FiberTrace h;
            h.hvTag = 'H';
            for (double theta = -0.2; theta <= 0.2 + 1e-9; theta += 0.002) {
                h.theta.push_back(theta);
                h.z.push_back(30001.0);
                h.radius.push_back(20300.0);
            }
            world.fibers.push_back(std::move(h));
            world.trueM.push_back(0);
            FiberTrace v;
            v.hvTag = 'V';
            std::vector<std::pair<double, double>> samples = {
                {30000.0, 20000.0}, {30003.0, 20000.0}, {29997.0, 21200.0}, {30100.0, 22000.0}};
            if (reversed) {
                std::reverse(samples.begin(), samples.end());
            }
            for (const auto& [z, radius] : samples) {
                v.theta.push_back(0.0);
                v.z.push_back(z);
                v.radius.push_back(radius);
            }
            world.fibers.push_back(std::move(v));
            world.trueM.push_back(0);
            const SolveResult result =
                solveWindings(world.fibers, world.links, SolverParams{});
            QCOMPARE(canonicalizeTrace(world.fibers[1], result.chirality).branches.size(),
                     std::size_t{3});
            QCOMPARE(result.crossings.size(), std::size_t{3});
            for (const CrossingGroup& group : result.groups) {
                QVERIFY(group.multiplicity < 3);
                QVERIFY(!group.hasVerdict);
            }
            for (const Crossing& crossing : result.crossings) {
                QVERIFY(crossing.status != CrossingStatus::InGroup);
            }
        }
    }

    // A traversal through a back-step meets the run's curtain three times.
    // Where the V fiber's radius is continuous across the back-step the two
    // extra crossings are of one kind and the inside count's parity holds;
    // here the radius steps across the one-voxel back-step and the H
    // fiber's radius falls between the limbs', so the three events are
    // Outside, Inside, Inside. The middle one lies on the back-step limb,
    // which marks the count as not one traversal's: no verdict (an Outside
    // verdict would separate the H fiber from the middle limb it sits
    // inside of).
    void backStepBandWithBothKindsTakesNoVerdict()
    {
        World world;
        FiberTrace h;
        h.hvTag = 'H';
        for (double theta = -0.2; theta <= 0.2 + 1e-9; theta += 0.002) {
            h.theta.push_back(theta);
            h.z.push_back(30000.0);
            h.radius.push_back(20300.0);
        }
        world.fibers.push_back(std::move(h));
        world.trueM.push_back(0);
        FiberTrace v;
        v.hvTag = 'V';
        for (const auto& [z, radius] : {std::pair{29900.0, 20000.0}, std::pair{30000.5, 20200.0},
                                        std::pair{29999.5, 20600.0}, std::pair{30100.0, 22000.0}}) {
            v.theta.push_back(0.0);
            v.z.push_back(z);
            v.radius.push_back(radius);
        }
        world.fibers.push_back(std::move(v));
        world.trueM.push_back(0);
        const SolveResult result =
            solveWindings(world.fibers, world.links, SolverParams{});
        QCOMPARE(canonicalizeTrace(world.fibers[1], result.chirality).branches.size(),
                 std::size_t{3});
        QCOMPARE(result.crossings.size(), std::size_t{3});
        const CrossingGroup& group = singleGroup(result);
        QCOMPARE(group.multiplicity, 3);
        QCOMPARE(group.insideCount, 2);
        QVERIFY(group.mixedSigns);
        QVERIFY(group.orientationSum % 2 != 0);
        QVERIFY(group.traversalCovered);
        QVERIFY(group.onCurtain);
        QVERIFY(!group.hasVerdict);
        for (const Crossing& crossing : result.crossings) {
            QVERIFY(crossing.status != CrossingStatus::InGroup);
        }
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

    // A hit at a fold apex is classified by the V fiber's own two limbs, not
    // by either branch's extension: the H fiber comes in from above and
    // leaves straight down between the limbs - a crossing - whichever way
    // the V fiber's samples run. (Dyadic angles: the hit must land exactly
    // on the vertex for the case to arise at all.)
    void apexHitIsClassifiedByTheVFibersOwnRays()
    {
        const SolverParams params;
        for (const bool reversed : {false, true}) {
            FiberTrace h;
            h.hvTag = 'H';
            h.theta = {-0.25, 0.0, 0.0};
            h.z = {30100.0, 30000.0, 29900.0};
            h.radius = {19900.0, 19900.0, 19900.0};
            FiberTrace v;
            v.hvTag = 'V';
            v.theta = {-0.125, 0.0, 0.125};
            v.z = {29900.0, 30000.0, 29900.0};
            v.radius = {20000.0, 20000.0, 20000.0};
            if (reversed) {
                std::reverse(v.theta.begin(), v.theta.end());
                std::reverse(v.z.begin(), v.z.end());
            }
            const CanonicalTrace ch = canonicalizeTrace(h, 1);
            const CanonicalTrace cv = canonicalizeTrace(v, 1);
            QCOMPARE(cv.branches.size(), std::size_t{2});
            const PairDetections geometry = detectPairCrossings(ch, cv, params);
            QCOMPARE(geometry.raw.size(), std::size_t{2});
            for (const Crossing& record : geometry.raw) {
                QVERIFY(!record.touch);
            }
            const PairCrossings classified =
                classifyPairCrossings(geometry, ch, cv, {}, {}, params);
            QCOMPARE(classified.events.size(), std::size_t{1});
            QVERIFY(!classified.events.front().touch);
            QCOMPARE(classified.events.front().mergedCount, 2);
        }
    }

    // At a fold apex the two records' orientations differ by the limbs'
    // opposite directions whatever the H fiber does; the ray test decides.
    // The apex crossing lies on the edge of both limbs' curtains: it counts
    // for neither, and both limbs' groups there take no verdict. An H fiber
    // that enters the apex from above, leaves straight down and then crosses
    // the limbs again below reads the same - one apex crossing, no touches,
    // the same groups, none with a verdict - whichever way its own samples
    // run.
    void apexRecordsCollapseByTheRayTestNotOrientation()
    {
        const SolverParams params;
        std::vector<std::vector<int>> summaries;
        for (const bool reversedH : {false, true}) {
            FiberTrace h;
            h.hvTag = 'H';
            h.theta = {-0.25, 0.0, 0.0, -0.25, 0.25};
            h.z = {25000.0, 30000.0, 25000.0, 25000.0, 25000.0};
            h.radius = {19800.0, 19800.0, 19800.0, 20400.0, 20400.0};
            if (reversedH) {
                std::reverse(h.theta.begin(), h.theta.end());
                std::reverse(h.z.begin(), h.z.end());
                std::reverse(h.radius.begin(), h.radius.end());
            }
            FiberTrace v;
            v.hvTag = 'V';
            v.theta = {-0.125, 0.0, 0.125};
            v.z = {20000.0, 30000.0, 20000.0};
            v.radius = {20000.0, 20000.0, 20000.0};
            const CanonicalTrace ch = canonicalizeTrace(h, 1);
            const CanonicalTrace cv = canonicalizeTrace(v, 1);
            const PairCrossings classified = classifyPairCrossings(
                detectPairCrossings(ch, cv, params), ch, cv, {}, {}, params);
            int apexEvents = 0;
            for (const Crossing& event : classified.events) {
                QVERIFY(!event.touch);
                if (event.vSample != Crossing::kNoSample) {
                    ++apexEvents;
                    QCOMPARE(event.mergedCount, 2);
                }
            }
            QCOMPARE(apexEvents, 1);
            std::vector<int> summary;
            for (const CrossingGroup& group : classified.groups) {
                QVERIFY(group.onCurtain);
                QVERIFY(!group.hasVerdict);
                summary.push_back(static_cast<int>(group.vBranch));
                summary.push_back(group.multiplicity);
                summary.push_back(group.insideCount);
                summary.push_back(group.hasVerdict ? 1 : 0);
                summary.push_back(group.hasVerdict ? static_cast<int>(group.verdict) : -1);
            }
            QVERIFY(!summary.empty());
            summaries.push_back(summary);
        }
        QCOMPARE(summaries.front(), summaries.back());
    }

    // A repeated apex sample at another radius: the apex's radial interval
    // is [20000, 21000]. With the H fiber outside it (19500) the hit reads at
    // the interval's nearest point, -500, whichever sample owns it, so the
    // two records are one representative and one event; with the H fiber
    // inside it (20500) the fibers meet in 3D there: one curtain contact,
    // deltaR 0, no verdict. Both are apex crossings and count for no group,
    // whichever way either fiber's samples run.
    void apexRecordsAtTwoRadiiStayTwoEvents()
    {
        const SolverParams params;
        for (const double hRadius : {19500.0, 20500.0}) {
            for (const bool reversedH : {false, true}) {
                for (const bool reversedV : {false, true}) {
                    FiberTrace h;
                    h.hvTag = 'H';
                    h.theta = {-0.25, 0.0, 0.0};
                    h.z = {30100.0, 30000.0, 29900.0};
                    h.radius = {hRadius, hRadius, hRadius};
                    FiberTrace v;
                    v.hvTag = 'V';
                    v.theta = {-0.125, 0.0, 0.0, 0.125};
                    v.z = {20000.0, 30000.0, 30000.0, 20000.0};
                    v.radius = {20000.0, 20000.0, 21000.0, 21000.0};
                    if (reversedH) {
                        std::reverse(h.theta.begin(), h.theta.end());
                        std::reverse(h.z.begin(), h.z.end());
                        std::reverse(h.radius.begin(), h.radius.end());
                    }
                    if (reversedV) {
                        std::reverse(v.theta.begin(), v.theta.end());
                        std::reverse(v.z.begin(), v.z.end());
                        std::reverse(v.radius.begin(), v.radius.end());
                    }
                    const CanonicalTrace ch = canonicalizeTrace(h, 1);
                    const CanonicalTrace cv = canonicalizeTrace(v, 1);
                    const PairCrossings classified = classifyPairCrossings(
                        detectPairCrossings(ch, cv, params), ch, cv, {}, {}, params);
                    for (const Crossing& event : classified.events) {
                        QVERIFY(!event.touch);
                        QVERIFY(event.apex);
                    }
                    for (const CrossingGroup& group : classified.groups) {
                        QVERIFY(!group.hasVerdict);
                    }
                    if (hRadius < 20000.0) {
                        QCOMPARE(classified.events.size(), std::size_t{1});
                        QCOMPARE(classified.events.front().deltaR, -500.0);
                        QCOMPARE(classified.events.front().kind, CrossingKind::Inside);
                        QCOMPARE(classified.events.front().mergedCount, 2);
                    } else {
                        QCOMPARE(classified.events.size(), std::size_t{1});
                        QCOMPARE(classified.events.front().deltaR, 0.0);
                        QCOMPARE(classified.events.front().mergedCount, 2);
                    }
                }
            }
        }
    }

    // Representation invariants under every sample order, with a crossing
    // before the apex whose radius decides how the proximity merge clusters
    // the apex records: every event's representative index is valid, and a
    // dropped representative always has a dropped, violated event to mark.
    // The apex here steps in radius across the H fiber's, so it reads as a
    // curtain contact; with the earlier crossing far outside (+350) it
    // contradicts that contact's weak Inside and one representative drops.
    void apexRecordsUnderTwoRepresentativesStayTwoEvents()
    {
        SolverParams params;
        params.chiralityOverride = 1;
        for (const double r0 : {19900.0, 20400.0}) {
            std::vector<std::vector<int>> summaries;
            for (const bool reversedH : {false, true}) {
                for (const bool reversedV : {false, true}) {
                    FiberTrace h;
                    h.hvTag = 'H';
                    h.theta = {0.0, -0.25, 0.0, 0.0};
                    h.z = {29900.0, 30100.0, 30000.0, 29950.0};
                    h.radius = {r0, 19900.0, 20000.0, 20000.0};
                    FiberTrace v;
                    v.hvTag = 'V';
                    v.theta = {-0.125, 0.0, 0.0, 0.125};
                    v.z = {20000.0, 30000.0, 30000.0, 20000.0};
                    v.radius = {20050.0, 20050.0, 19950.0, 19950.0};
                    if (reversedH) {
                        std::reverse(h.theta.begin(), h.theta.end());
                        std::reverse(h.z.begin(), h.z.end());
                        std::reverse(h.radius.begin(), h.radius.end());
                    }
                    if (reversedV) {
                        std::reverse(v.theta.begin(), v.theta.end());
                        std::reverse(v.z.begin(), v.z.end());
                        std::reverse(v.radius.begin(), v.radius.end());
                    }
                    const SolveResult result = solveWindings({h, v}, {}, params);
                    QCOMPARE(result.events.size(), std::size_t{2});
                    std::vector<int> summary;
                    int apex = 0;
                    for (const Crossing& event : result.events) {
                        QVERIFY(!event.touch);
                        apex += event.apex ? 1 : 0;
                        QVERIFY(event.representative < result.crossings.size());
                        summary.push_back(static_cast<int>(event.kind));
                        summary.push_back(static_cast<int>(event.status));
                    }
                    std::sort(summary.begin(), summary.end());
                    summary.push_back(countDroppedCrossings(result));
                    summaries.push_back(summary);
                    QCOMPARE(apex, 1);
                    QCOMPARE(countDroppedCrossings(result), r0 > 20000.0 ? 1 : 0);
                    for (std::size_t c = 0; c < result.crossings.size(); ++c) {
                        if (result.crossings[c].status != CrossingStatus::Dropped) {
                            continue;
                        }
                        bool marked = false;
                        for (const Crossing& event : result.events) {
                            marked = marked || (event.representative == c &&
                                                event.status == CrossingStatus::Dropped &&
                                                event.violationTurns >= 0.5);
                        }
                        QVERIFY(marked);
                    }
                }
            }
            for (std::size_t k = 1; k < summaries.size(); ++k) {
                QCOMPARE(summaries[k], summaries[0]);
            }
        }
    }

    // Degenerate hits read the same whichever way either fiber's samples run,
    // and where no intersection can be placed the translate is unresolved
    // (no verdict) rather than counted one way in one order and another in
    // the other. Coordinates (theta, z, r), chirality +1.
    void degenerateHitsReadTheSameInEitherOrder()
    {
        const SolverParams params;
        struct Fixture {
            const char* name;
            std::vector<double> ht, hz, hr, vt, vz, vr;
        };
        // 1: a radial step of the H fiber (zero projected length, radius
        //    19000 -> 21000) through the V fiber at (0, 0, 20000).
        // 2: an H vertex touching the V fiber exactly at its radius.
        // 3: collinear owner segments sharing only the vertex (0, 0): one
        //    crossing, read by the rays at the vertex.
        // 4: repeated start samples of an H fiber that a V fold touches from
        //    below and leaves.
        // 5: an apex hit whose two parameters differ in the last place.
        // 6: a hit at t == 0 of a segment on whose far side a gated limb sits.
        // 8 ("vertex on segment"): a V vertex exactly on an H segment whose radii run 5000 -> 35000
        //    through the V's 20000: a contact, deltaR 0, whichever way (the V
        //    is monotone here, so the vertex is interior, not an apex).
        // 9: radial runs of both fibers at one point, overlapping in radius.
        // 10: an H run [1000, 19000] at a V apex of radius 20000: the read
        //    radius is the run's nearest, 19000, not the owner's.
        // 7: a fold with a level top the H fiber crosses at its corner. The
        //    corner is one record or two collapsed ones depending on which
        //    limb the level run joins, so the apex flag is left out of the
        //    comparison; the translate is unresolved either way.
        const std::vector<Fixture> fixtures{
            {"radial step", {-0.25, 0.0, 0.0, 0.25, -0.25, 0.25}, {0, 0, 0, 100, 200, 300},
             {19000, 19000, 21000, 21000, 21000, 17000}, {0.0, 0.0}, {-1000, 1000}, {20000, 20000}},
            {"curtain touch", {-0.25, 0.0, -0.25, 0.25, -0.25, 0.25}, {0, 100, 200, 300, 400, 500},
             {19000, 20000, 21000, 21000, 21000, 17000}, {0.0, 0.0}, {-1000, 1000}, {20000, 20000}},
            {"collinear contact", {0.5, 0.0, 0.0}, {-500, 0, -1000}, {19000, 19000, 19000},
             {0.5, 0.0, 0.0}, {-1000, 0, 1000}, {20000, 20000, 20000}},
            {"repeated start", {0.0, 0.0, 0.25}, {0, 0, 0}, {19000, 19000, 19000},
             {0.125, 0.0, -0.125}, {-1000, 0, -1000}, {20000, 20000, 20000}},
            {"apex ulp", {-0.001, 0.001}, {-100, 100}, {19000, 19000},
             {-0.3, 0.0, 0.2}, {-1000, 0, -1234}, {20000, 20000, 20000}},
            {"terminal own segment", {-0.25, 0.0, 0.25}, {0, 0, 0}, {21000, 21000, 21000},
             {0.0, 0.0, 0.125, 0.125}, {-1000, 1000, 1000, -1000}, {20000, 20000, 100, 100}},
            {"flat top", {0.0, 0.0}, {1100, 900}, {19000, 19000},
             {-0.25, 0.0, 0.25, 0.5}, {0, 1000, 1000, 0}, {20000, 20000, 20000, 20000}},
            {"vertex on segment", {0.5, -0.001, 0.001, 0.5, -0.5}, {-200, -100, 100, 200, 300},
             {5000, 5000, 35000, 35000, 1000}, {-0.3, 0.0, 0.2}, {-1000, 0, 1234},
             {20000, 20000, 20000}},
            {"overlapping runs", {-0.25, 0.0, 0.0, 0.0}, {100, 0, 0, -100},
             {19000, 19000, 21000, 21000}, {-0.125, 0.0, 0.0, 0.125}, {-1000, 0, 0, -1000},
             {20000, 20000, 22000, 22000}},
            {"run nearest radius", {-0.001, 0.0, 0.0, 0.0}, {1000, 0, 0, -1000},
             {1000, 1000, 19000, 19000}, {-0.002, 0.0, 0.002}, {-1000, 0, -1000},
             {20000, 20000, 20000}},
            // 11: a V vertex on an H segment at parameter 1/3, the H radius
            //     there (15000) equal to the V's only in the reals.
            {"rounded contact", {0.5, -0.25, 0.125, 0.5, -0.5}, {-300, -200, 100, 200, 300},
             {35000, 35000, 5000, 5000, 15000}, {-0.25, 0.0, 0.25}, {-1000, 0, 1000},
             {15000, 15000, 15000}},
            // 12: a radial V step at a point of an H segment (parameter 1/3)
            //     whose radius there is the step's end.
            {"radial step at rounded radius", {-0.25, 0.125}, {-200, 100}, {35000, 5000},
             {0.0, 0.0}, {0, 0}, {14000, 15000}},
            // 13: a contact whose interpolated radius cancels from 33670 and
            //     491 down to 956.375: the envelope scales with the inputs.
            {"cancelling contact",
             {0.5, -32713.625 / 65536.0, 465.375 / 65536.0, 0.5, -0.5},
             {-1500, -32713.625 / 32.0, 465.375 / 32.0, 200, 300},
             {33670, 33670, 491, 491, 1000}, {0.0, 0.0, 0.0}, {-2000, 0, 2000},
             {956.375, 956.375, 956.375}},
            // 14: the same H segment against a radial V step ending at that
            //     radius: unresolved either way.
            {"cancelling radial step", {-32713.625 / 65536.0, 465.375 / 65536.0},
             {-32713.625 / 32.0, 465.375 / 32.0}, {33670, 491}, {0.0, 0.0}, {0, 0},
             {955.375, 956.375}},
            // 15: nearly parallel, disjoint: the lines meet far off both
            //     segments; nothing is detected and nothing is unresolved.
            {"near parallel disjoint", {0.0, 0.5}, {0, 1024}, {19000, 19000},
             {0.125, 0.375}, {256 + std::ldexp(1.0, -42), 768 + 3 * std::ldexp(1.0, -43)},
             {20000, 20000}},
        };
        for (const Fixture& f : fixtures) {
            std::vector<std::vector<int>> summaries;
            std::vector<PairCrossings> results;
            for (int order = 0; order < 4; ++order) {
                FiberTrace h;
                h.hvTag = 'H';
                h.theta = f.ht;
                h.z = f.hz;
                h.radius = f.hr;
                FiberTrace v;
                v.hvTag = 'V';
                v.theta = f.vt;
                v.z = f.vz;
                v.radius = f.vr;
                if (order & 1) {
                    std::reverse(h.theta.begin(), h.theta.end());
                    std::reverse(h.z.begin(), h.z.end());
                    std::reverse(h.radius.begin(), h.radius.end());
                }
                if (order & 2) {
                    std::reverse(v.theta.begin(), v.theta.end());
                    std::reverse(v.z.begin(), v.z.end());
                    std::reverse(v.radius.begin(), v.radius.end());
                }
                const CanonicalTrace ch = canonicalizeTrace(h, 1);
                const CanonicalTrace cv = canonicalizeTrace(v, 1);
                const PairDetections geometry = detectPairCrossings(ch, cv, params);
                const PairCrossings classified =
                    classifyPairCrossings(geometry, ch, cv, {}, {}, params);
                // Order-free summary: the multiset of (kind, touch, apex,
                // tangential) over events, and per group whether it has a
                // verdict and which.
                std::vector<int> summary;
                std::vector<std::vector<int>> eventRows;
                const bool flatTop = std::string(f.name) == "flat top";
                for (const Crossing& event : classified.events) {
                    eventRows.push_back({static_cast<int>(event.kind), event.touch ? 1 : 0,
                                         flatTop ? 0 : (event.apex ? 1 : 0),
                                         event.tangential ? 1 : 0});
                }
                std::sort(eventRows.begin(), eventRows.end());
                for (const auto& row : eventRows) {
                    summary.insert(summary.end(), row.begin(), row.end());
                }
                summary.push_back(-1);
                int verdicts = 0;
                for (const CrossingGroup& group : classified.groups) {
                    verdicts += group.hasVerdict ? 1 : 0;
                    summary.push_back(group.hasVerdict ? static_cast<int>(group.verdict) : -2);
                }
                summary.push_back(geometry.unresolvedCount > 0 ? 1 : 0);
                summaries.push_back(summary);
                results.push_back(classified);
                // No fixture here supports a verdict.
                QVERIFY2(verdicts == 0, f.name);
            }
            for (std::size_t k = 1; k < summaries.size(); ++k) {
                QVERIFY2(summaries[k] == summaries[0], f.name);
            }
            const PairCrossings& first = results.front();
            const std::string name = f.name;
            if (name == "radial step" || name == "flat top") {
                QVERIFY2(first.unresolvedCount > 0, f.name);
            } else if (name == "collinear contact") {
                QCOMPARE(first.events.size(), std::size_t{1});
                QVERIFY2(!first.events.front().touch, f.name);
                QVERIFY2(!first.events.front().tangential, f.name);
            } else if (name == "curtain touch") {
                QVERIFY2(!first.groups.empty() && first.groups.front().onCurtain, f.name);
            } else if (name == "repeated start") {
                for (const Crossing& event : first.events) {
                    QVERIFY2(event.touch, f.name);
                }
            } else if (name == "apex ulp") {
                QCOMPARE(first.events.size(), std::size_t{1});
                QVERIFY2(first.events.front().apex, f.name);
                QCOMPARE(first.events.front().mergedCount, 2);
            } else if (name == "vertex on segment") {
                bool contact = false;
                for (const Crossing& event : first.events) {
                    contact = contact || event.deltaR == 0.0;
                }
                QVERIFY2(contact, f.name);
                QVERIFY2(!first.groups.empty() && first.groups.front().onCurtain, f.name);
            } else if (name == "rounded contact") {
                bool contact = false;
                for (const Crossing& event : first.events) {
                    contact = contact || event.deltaR == 0.0;
                }
                QVERIFY2(contact, f.name);
                QVERIFY2(!first.groups.empty() && first.groups.front().onCurtain, f.name);
            } else if (name == "radial step at rounded radius" ||
                       name == "cancelling radial step") {
                QVERIFY2(first.unresolvedCount > 0, f.name);
            } else if (name == "cancelling contact") {
                bool contact = false;
                for (const Crossing& event : first.events) {
                    contact = contact || event.deltaR == 0.0;
                }
                QVERIFY2(contact, f.name);
                QVERIFY2(!first.groups.empty() && first.groups.front().onCurtain, f.name);
            } else if (name == "near parallel disjoint") {
                QCOMPARE(first.events.size(), std::size_t{0});
                QCOMPARE(first.unresolvedCount, 0);
            } else if (name == "overlapping runs") {
                for (const Crossing& event : first.events) {
                    QCOMPARE(event.deltaR, 0.0);
                }
            } else if (name == "run nearest radius") {
                for (const Crossing& event : first.events) {
                    QVERIFY2(std::abs(event.deltaR + 1000.0) < 1e-9, f.name);
                }
            } else if (name == "terminal own segment") {
                QCOMPARE(first.events.size(), std::size_t{1});
                QVERIFY2(first.events.front().terminal, f.name);
                QVERIFY2(first.events.front().terminalSides != 0, f.name);
                for (const PairCrossings& other : results) {
                    QCOMPARE(other.events.front().terminalSides,
                             first.events.front().terminalSides);
                }
            }
        }
    }

    // Independent annotations can meet exactly in the reals while every
    // difference in doubles rounds: integer voxel samples on a straight
    // umbilicus give an H fiber at angles a and 4a and a V apex at 2a, one
    // third along the H segment in height. The incidence is decided inside
    // the rounding envelope: one transversal apex event in either H order.
    void roundedIncidenceIsOneApexEvent()
    {
        const SolverParams params;
        const double a = std::atan2(2048.0, 10240.0);
        for (const bool reversedH : {false, true}) {
            FiberTrace h;
            h.hvTag = 'H';
            h.theta = {a, 4.0 * a};
            h.z = {19900.0, 20200.0};
            h.radius = {std::hypot(10240.0, 2048.0), std::hypot(15232.0, 15360.0)};
            if (reversedH) {
                std::reverse(h.theta.begin(), h.theta.end());
                std::reverse(h.z.begin(), h.z.end());
                std::reverse(h.radius.begin(), h.radius.end());
            }
            FiberTrace v;
            v.hvTag = 'V';
            v.theta = {2.0 * a, 2.0 * a, a};
            v.z = {19990.0, 20000.0, 19990.0};
            v.radius = {std::hypot(15360.0, 6400.0), std::hypot(15360.0, 6400.0),
                        std::hypot(10240.0, 2048.0)};
            const CanonicalTrace ch = canonicalizeTrace(h, 1);
            const CanonicalTrace cv = canonicalizeTrace(v, 1);
            QCOMPARE(cv.branches.size(), std::size_t{2});
            const PairCrossings classified = classifyPairCrossings(
                detectPairCrossings(ch, cv, params), ch, cv, {}, {}, params);
            QCOMPARE(classified.events.size(), std::size_t{1});
            const Crossing& event = classified.events.front();
            QVERIFY(!event.touch);
            QVERIFY(!event.tangential);
            QVERIFY(event.apex);
            QCOMPARE(event.mergedCount, 2);
        }
    }

    // A vertex hit's orientation follows the directed cyclic order of the
    // incident rays and flips with the H fiber's direction, also where the
    // chords are parallel (an H fiber turning back on itself at the vertex).
    void vertexOrientationFlipsWithTheHFiber()
    {
        const SolverParams params;
        int forward = 0;
        int reversed = 0;
        for (const bool reversedH : {false, true}) {
            FiberTrace h;
            h.hvTag = 'H';
            h.theta = {0.25, 0.0, 0.25};
            h.z = {-2000.0, 0.0, 0.0};
            h.radius = {19000.0, 19000.0, 19000.0};
            FiberTrace v;
            v.hvTag = 'V';
            v.theta = {0.25, 0.0, 0.25};
            v.z = {-1000.0, 0.0, 1000.0};
            v.radius = {20000.0, 20000.0, 20000.0};
            if (reversedH) {
                std::reverse(h.theta.begin(), h.theta.end());
                std::reverse(h.z.begin(), h.z.end());
            }
            const CanonicalTrace ch = canonicalizeTrace(h, 1);
            const CanonicalTrace cv = canonicalizeTrace(v, 1);
            const PairCrossings classified = classifyPairCrossings(
                detectPairCrossings(ch, cv, params), ch, cv, {}, {}, params);
            QCOMPARE(classified.events.size(), std::size_t{1});
            QVERIFY(!classified.events.front().touch);
            (reversedH ? reversed : forward) = classified.events.front().orientation;
        }
        QVERIFY(forward != 0);
        QCOMPARE(reversed, -forward);
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
        // few more), and the hit is at that segment's start, so the whole
        // segment lies ahead: not terminal toward the end, terminal toward
        // the start.
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
            const Crossing& event = plain.events.front();
            QCOMPARE(event.hT, 0.0);
            const int startBit = ch.psi.front() > event.psiH ? 1 : 2;
            QVERIFY(event.terminal);
            QCOMPARE(event.terminalSides, startBit);
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

    // A seam encounter on a traversal's translate takes its events out of
    // the count, and the rest is an incomplete traversal: five crossings
    // (inside, outside, inside, inside, outside; orientations alternating)
    // read together say Inside (three inside passes). With the last two read
    // as the linked seam encounter, the remaining three would say Outside -
    // so a seamed group takes no verdict at all.
    void seamedTraversalGroupTakesNoVerdict()
    {
        for (const bool seam : {false, true}) {
            World world;
            FiberTrace hFiber;
            hFiber.hvTag = 'H';
            const double amplitude = 0.1 * kTwoPi;
            const double dRAt[5] = {-200.0, 300.0, -200.0, -25.0, 25.0};
            for (int k = 0; k <= 500; ++k) {
                const double s = -0.5 * M_PI + 5.0 * M_PI * k / 500.0;
                const double z = 29600.0 + 200.0 * s / M_PI;
                // Radial offset from the V fiber, linear between the
                // crossings' values at s = 0, pi, ..., 4 pi.
                const double q = std::clamp(s / M_PI, 0.0, 4.0);
                const int lo = static_cast<int>(std::floor(q));
                const int hi = std::min(lo + 1, 4);
                const double dR = dRAt[lo] + (q - lo) * (dRAt[hi] - dRAt[lo]);
                hFiber.theta.push_back(kTwoPi * kSeamWinding + amplitude * std::sin(s));
                hFiber.z.push_back(z);
                hFiber.radius.push_back(sheetR(kSeamWinding, z) - 200.0 + dR);
            }
            world.fibers.push_back(std::move(hFiber));
            world.trueM.push_back(0);
            const std::size_t v = addV(world, kSeamWinding, 29000.0, 31000.0, -200.0);
            if (seam) {
                world.fibers[v].onKollesis = true;
                world.fibers[0].kollesisEndSample = world.fibers[0].theta.size() - 1;
                world.links.push_back(LinkInput{0, world.fibers[0].theta.size() - 1, v, 56});
            }
            const SolveResult result =
                solveWindings(world.fibers, world.links, SolverParams{});
            QCOMPARE(result.events.size(), std::size_t{5});
            QCOMPARE(result.groups.size(), std::size_t{1});
            const CrossingGroup& group = result.groups.front();
            if (!seam) {
                QCOMPARE(group.multiplicity, 5);
                QCOMPARE(group.insideCount, 3);
                QVERIFY(!group.seamed);
                QVERIFY(group.hasVerdict);
                QCOMPARE(group.verdict, CrossingKind::Inside);
            } else {
                QCOMPARE(result.kollesisCrossingCount, 2);
                QCOMPARE(group.multiplicity, 3);
                QVERIFY(group.seamed);
                QVERIFY(!group.hasVerdict);
            }
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

    // --- Bent readings (see winding::BentHit, BentAssembly): the shard's
    // raw curtain hits classified against the assembly's decisions.

    // A bent reading of a pair the straight rays never meet constrains like
    // a straight crossing: one representative and event, Outside from the
    // V on the inward side of the H curtain (the H one winding out of the
    // V, which the strict reading pins exactly), confidence the
    // transversality, the 3D hit retained; against a clean link asserting
    // the same winding, it is dropped with violation 1 and its record
    // keeps the hit.
    void bentReadingConstrainsAndExports()
    {
        World world;
        const std::size_t h = addH(world, 1.4, 1.9, 30000.0);
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
        const BentFixture fixture(world, 1);
        const std::size_t seed = hSampleAt(world, h, 1.4, 1.42);
        const std::size_t vMid = world.fibers[v].theta.size() / 2;
        BentHit hit = fixture.hit(h, v, false, seed, seed + 2, 0.5, vMid, 0.25, 4.0, -1);
        hit.hitX = 1234.0;
        hit.hitY = 56.0;
        hit.hitZ = 30000.0;
        BentSolve solved = fixture.solve({{{h, v}, {hit}}});
        QCOMPARE(solved.result.bentCrossingCount, 1);
        QCOMPARE(solved.result.withheldCount, 0);
        const Crossing* record = solved.bentRecord(0);
        QVERIFY(record != nullptr);
        QVERIFY(record->bent);
        QVERIFY(!record->curtainFromV);
        QVERIFY(!record->withheld);
        QCOMPARE(record->anchor, 2);
        QCOMPARE(record->n, fixture.expectedN(h, v, hit));
        QCOMPARE(record->kind, CrossingKind::Outside);
        QCOMPARE(record->deltaR, 4.0);
        QCOMPARE(record->confidence, 1.0);
        QCOMPARE(record->mergedCount, 1);
        QCOMPARE(record->groupIndex, -1LL);
        QCOMPARE(record->status, CrossingStatus::Used);
        QCOMPARE(record->hitX, 1234.0);
        QCOMPARE(record->rayLengthVx, 4.0);
        QCOMPARE(record->orientation, 0);
        QVERIFY(record->vSample == Crossing::kNoSample);
        // The event mirrors it and the straight rays saw nothing.
        QCOMPARE(solved.result.events.size(), std::size_t{1});
        QVERIFY(solved.result.events.front().bent);
        QCOMPARE(solved.result.events.front().representative, solved.bentIndex(0));
        // psiH goes out in the caller's gauge, at the seed position.
        const double expectedPsiH = fixture.chirality *
            (world.fibers[h].theta[seed] + world.fibers[h].theta[seed + 2]) / 2.0;
        QVERIFY(std::abs(record->psiH - expectedPsiH) < 1e-9);
        checkRelativeTurns(solved.result, world, {h, v});

        // Opposing independent evidence: a clean plain link (drawn at the
        // V's own angle, so its residual is zero and its confidence 1.5)
        // puts the two on one winding; the V sits above the H's height so
        // the straight rays see nothing. The weaker bent Outside is
        // dropped, violated by one winding, and the exported record keeps
        // its hit.
        World linked;
        const std::size_t h2 = addH(linked, 1.3, 1.9, 30000.0);
        const std::size_t v2 = addV(linked, 0.3, 31000.0, 34000.0);
        const std::size_t vLink = linked.fibers[v2].theta.size() / 2;
        LinkInput link;
        link.fiberA = h2;
        link.pointA = hSampleAt(linked, h2, 1.3, 1.3);
        link.fiberB = v2;
        link.pointB = vLink;
        link.windingOffset = 0;
        linked.links.push_back(link);
        const BentFixture linkedFixture(linked, 1);
        const std::size_t seed2 = hSampleAt(linked, h2, 1.3, 1.5);
        BentHit hit2 = linkedFixture.hit(h2, v2, false, seed2, seed2 + 2, 0.5, vLink, 0.25, 4.0, -1);
        hit2.hitX = 1234.0;
        BentSolve conflicted = linkedFixture.solve({{{h2, v2}, {hit2}}});
        const Crossing* dropped = conflicted.bentRecord(0);
        QVERIFY(dropped != nullptr);
        QCOMPARE(conflicted.result.crossings.size(), std::size_t{1});
        QCOMPARE(dropped->status, CrossingStatus::Dropped);
        QCOMPARE(dropped->violationTurns, 1.0);
        QCOMPARE(dropped->hitX, 1234.0);
        QCOMPARE(conflicted.result.droppedCrossingCount, 1);
        QVERIFY(conflicted.result.droppedLinks.empty());
        QCOMPARE(conflicted.result.placements[v2].turns - conflicted.result.placements[h2].turns,
                 0.0);
    }

    // The lift: the ray's accumulated angle enters with the winding sense
    // and the owner's direction, so the translate is exact for either owner
    // and either chirality, unchanged by a whole-turn re-gauge of the V,
    // and a reading whose residual exceeds a quarter turn is withheld as
    // ambiguous.
    void liftGivesTheTranslateForEitherOwnerAndSense()
    {
        for (const int sense : {1, -1}) {
            for (const bool ownerIsV : {false, true}) {
                World world;
                const std::size_t h = addH(world, 0.4, 0.9, 30000.0);
                const std::size_t v = addV(world, 1.3, 27000.0, 33000.0);
                if (sense < 0) {
                    world = mirrored(world);
                }
                const BentFixture fixture(world, sense);
                QCOMPARE(fixture.chirality, sense);
                const std::size_t seedH = hSampleAt(world, h, 0.4, 0.42);
                const std::size_t vMid = world.fibers[v].theta.size() / 2;
                const BentHit hit = ownerIsV
                    ? fixture.hit(h, v, true, vMid, vMid + 3, 0.5, seedH, 0.5, 6.0, -1)
                    : fixture.hit(h, v, false, seedH, seedH + 2, 0.5, vMid, 0.5, 6.0, 1);
                const long long n = fixture.expectedN(h, v, hit);
                {
                    const BentSolve solved = fixture.solve({{{h, v}, {hit}}});
                    const Crossing* record = solved.bentRecord(0);
                    QVERIFY(record != nullptr);
                    QCOMPARE(record->n, n);
                    QVERIFY(!record->withheld);
                    QCOMPARE(record->curtainFromV, ownerIsV);
                    // Side +1 of the H curtain (V outward) and side -1 of the
                    // V curtain (H inward) both read Inside.
                    QCOMPARE(record->kind, CrossingKind::Inside);
                    QCOMPARE(record->deltaR, -6.0);
                    // The lifted V sits one winding out of the H, so the V's
                    // lift at the hit is a turn ahead of the H's gauge: the
                    // gap the translate names is the physical one.
                    // In the canonical gauges the translate was lifted in:
                    // the placements restore the input gauges, the canonical
                    // traces' whole-turn gauges put them back.
                    const double gap = static_cast<double>(n) +
                                       solved.result.placements[v].turns +
                                       static_cast<double>(fixture.canonical[v].gauge) -
                                       solved.result.placements[h].turns -
                                       static_cast<double>(fixture.canonical[h].gauge);
                    QVERIFY(gap >= 0.0);
                }
                // A whole-turn re-gauge of the V: the same translate.
                World regauged = world;
                for (double& theta : regauged.fibers[v].theta) {
                    theta += kTwoPi;
                }
                const BentFixture regaugedFixture(regauged, sense);
                const BentHit regaugedHit = ownerIsV
                    ? regaugedFixture.hit(h, v, true, vMid, vMid + 3, 0.5, seedH, 0.5, 6.0, -1)
                    : regaugedFixture.hit(h, v, false, seedH, seedH + 2, 0.5, vMid, 0.5, 6.0, 1);
                const BentSolve solved = regaugedFixture.solve({{{h, v}, {regaugedHit}}});
                QVERIFY(solved.bentRecord(0) != nullptr);
                QCOMPARE(solved.bentRecord(0)->n, n);
                QVERIFY(static_cast<double>(n) + solved.result.placements[v].turns +
                            static_cast<double>(regaugedFixture.canonical[v].gauge) -
                            solved.result.placements[h].turns -
                            static_cast<double>(regaugedFixture.canonical[h].gauge) >=
                        0.0);
                // A residual over a quarter turn: ambiguous, withheld.
                BentHit ambiguous = regaugedHit;
                ambiguous.theta += 0.3 * kTwoPi;
                const BentSolve withheld = regaugedFixture.solve({{{h, v}, {ambiguous}}});
                QVERIFY(withheld.bentRecord(0) != nullptr);
                QVERIFY(withheld.bentRecord(0)->withheld);
                QCOMPARE(withheld.bentRecord(0)->bentReason,
                         static_cast<int>(BentDisposition::LiftAmbiguous));
                QCOMPARE(withheld.result.bentCrossingCount, 0);
                QCOMPARE(withheld.result.withheldCount, 1);
            }
        }
    }

    // Repeated encounters of one strip on one translate are one passage:
    // the smallest s stands, an equal s goes to the greater transversality,
    // then to the smaller hit point; distinct translates are distinct
    // encounters, and nothing unusable competes or suppresses.
    void firstEncounterPerStripAndTranslate()
    {
        World world;
        const std::size_t h = addH(world, 0.4, 0.9, 30000.0);
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
        const BentFixture fixture(world, 1);
        const std::size_t seed = hSampleAt(world, h, 0.4, 0.42);
        const std::size_t vMid = world.fibers[v].theta.size() / 2;
        const BentHit near = fixture.hit(h, v, false, seed, seed + 2, 0.5, vMid, 0.25, 4.0, 1);
        const BentHit far = fixture.hit(h, v, false, seed, seed + 2, 0.5, vMid + 2, 0.25, 12.0, 1);
        // The same strip met a turn later: the ray went once more around.
        BentHit otherTurn = fixture.hit(h, v, false, seed, seed + 2, 0.5, vMid + 4, 0.25, 12.0, 1);
        otherTurn.theta += kTwoPi;
        const long long n0 = fixture.expectedN(h, v, near);
        {
            // The tie band is 1e-6 vx about the smallest length: a more
            // transversal passage inside it wins, one outside it does not,
            // and the band is measured from the minimum, not chained.
            const auto at = [&](double s, double transversality, double x) {
                return fixture.hit(h, v, false, seed, seed + 2, 0.5, vMid, 0.25, s, 1, transversality, x);
            };
            const BentHit weakest = at(4.0, 0.5, 0.0);
            const BentHit inside = at(4.0 + 5e-7, 1.0, 1.0);
            const BentHit outside = at(4.0 + 2e-6, 1.0, 2.0);
            const BentHit chained = at(4.0 + 8e-7, 0.6, 3.0);
            const BentHit beyond = at(4.0 + 1.6e-6, 1.0, 4.0);
            const auto kept = [&](std::vector<BentHit> hits) {
                const BentSolve solved = fixture.solve({{{h, v}, hits}});
                return std::make_pair(solved.result.bentCrossingCount, solved.bentRecord(0)->hitX);
            };
            QCOMPARE(kept({weakest, inside}), std::make_pair(1, 1.0));
            QCOMPARE(kept({inside, weakest}), std::make_pair(1, 1.0));
            QCOMPARE(kept({weakest, outside}), std::make_pair(1, 0.0));
            QCOMPARE(kept({outside, weakest}), std::make_pair(1, 0.0));
            QCOMPARE(kept({weakest, chained, beyond}), std::make_pair(1, 3.0));
            QCOMPARE(kept({beyond, chained, weakest}), std::make_pair(1, 3.0));
        }
        {
            const BentSolve solved = fixture.solve({{{h, v}, {far, near, otherTurn}}});
            QCOMPARE(solved.result.bentCrossingCount, 2);
            std::set<double> lengths;
            std::set<long long> translates;
            for (const Crossing& c : solved.result.crossings) {
                if (c.bent) {
                    lengths.insert(c.rayLengthVx);
                    translates.insert(c.n);
                }
            }
            QCOMPARE(lengths, (std::set<double>{4.0, 12.0}));
            QCOMPARE(translates, (std::set<long long>{n0, n0 - 1}));
        }
        {
            // An equal s: the more transversal record; then the smaller point.
            BentHit shallow = near;
            shallow.transversality = 0.5;
            shallow.hitX = 1.0;
            BentHit steep = near;
            steep.transversality = 1.0;
            steep.hitX = 2.0;
            const BentSolve solved = fixture.solve({{{h, v}, {shallow, steep}}});
            QCOMPARE(solved.result.bentCrossingCount, 1);
            QCOMPARE(solved.bentRecord(0)->hitX, 2.0);
            BentHit left = steep;
            left.hitX = 1.0;
            const BentSolve tied = fixture.solve({{{h, v}, {steep, left}}});
            QCOMPARE(tied.result.bentCrossingCount, 1);
            QCOMPARE(tied.bentRecord(0)->hitX, 1.0);
        }
        {
            // A touch nearer than the crossing does not suppress it.
            BentHit touch = near;
            touch.s = 2.0;
            touch.touch = true;
            const BentSolve solved = fixture.solve({{{h, v}, {touch, far}}});
            QCOMPARE(solved.result.bentCrossingCount, 1);
            int contacts = 0;
            int crossings = 0;
            for (const Crossing& c : solved.result.crossings) {
                if (!c.bent) {
                    continue;
                }
                if (c.touch) {
                    ++contacts;
                    QVERIFY(c.tangential);
                    QCOMPARE(c.confidence, 0.0);
                } else {
                    ++crossings;
                    QCOMPARE(c.rayLengthVx, 12.0);
                }
            }
            QCOMPARE(contacts, 1);
            QCOMPARE(crossings, 1);
        }
    }

    // The H curtain is preferred over the V curtain only for an EQUIVALENT
    // reading: the same kind and translate with both positions in the same
    // ill-conditioned runs. Opposing readings are both kept; a withheld or
    // a contact H record suppresses nothing.
    void hCurtainPreferredOnlyForEquivalentReadings()
    {
        World world;
        const std::size_t h = addH(world, 0.4, 0.9, 30000.0);
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
        const BentFixture fixture(world, 1);
        const std::size_t seedH = hSampleAt(world, h, 0.4, 0.42);
        const std::size_t vMid = world.fibers[v].theta.size() / 2;
        const BentHit fromH = fixture.hit(h, v, false, seedH, seedH + 2, 0.5, vMid, 0.5, 4.0, 1);
        const BentHit fromV = fixture.hit(h, v, true, vMid, vMid + 3, 0.5, seedH, 0.5, 6.0, -1);
        QCOMPARE(fixture.expectedN(h, v, fromH), fixture.expectedN(h, v, fromV));
        {
            const BentSolve solved = fixture.solve({{{h, v}, {fromH, fromV}}});
            QCOMPARE(solved.result.bentCrossingCount, 1);
            QVERIFY(!solved.bentRecord(0)->curtainFromV);
        }
        {
            // The V curtain reads Outside (H on its outward side): the two
            // curtains disagree on the translate, a fold signature, and both
            // readings are withheld as such (nothing is preferred, nothing
            // constrains).
            BentHit opposing = fromV;
            opposing.side = 1;
            const BentSolve solved = fixture.solve({{{h, v}, {fromH, opposing}}});
            QCOMPARE(solved.result.bentCrossingCount, 0);
            QCOMPARE(solved.result.withheldCount, 2);
            for (const Crossing& c : solved.result.crossings) {
                if (c.bent) {
                    QCOMPARE(c.bentReason, static_cast<int>(BentDisposition::Folded));
                }
            }
        }
        {
            // A withheld H record (no witness on its run) suppresses nothing.
            BentFixture unwitnessed = fixture;
            unwitnessed.assemblies[h].stretches[0].withheld = true;
            unwitnessed.assemblies[h].stretches[0].disposition = BentDisposition::NoWitness;
            const BentSolve solved = unwitnessed.solve({{{h, v}, {fromH, fromV}}});
            QCOMPARE(solved.result.bentCrossingCount, 1);
            QVERIFY(solved.bentRecord(0)->curtainFromV ||
                    (solved.bentRecord(1) != nullptr && solved.bentRecord(1)->curtainFromV));
            QCOMPARE(solved.result.withheldCount, 1);
        }
        {
            BentHit touch = fromH;
            touch.touch = true;
            const BentSolve solved = fixture.solve({{{h, v}, {touch, fromV}}});
            QCOMPARE(solved.result.bentCrossingCount, 1);
        }
        {
            // Positions in different runs: the H record's V position lies
            // outside the run holding the V record's seed.
            BentFixture split = fixture;
            split.assemblies[v].stretches[0].lastSample = vMid + 1;
            BentHit elsewhere = fromH;
            elsewhere.segment = vMid + 5;
            const BentSolve solved = split.solve({{{h, v}, {elsewhere, fromV}}});
            QCOMPARE(solved.result.bentCrossingCount, 2);
            // The same H reading from both passages of the V (exact twins:
            // the first encounter keeps one, the storage order says which):
            // the V reading is suppressed whichever twin was kept, because
            // the dropped twin keeps standing in for its passage.
            for (const std::vector<BentHit>& order :
                 {std::vector<BentHit>{fromH, elsewhere, fromV}, std::vector<BentHit>{elsewhere, fromH, fromV}}) {
                const BentSolve twins = split.solve({{{h, v}, order}});
                QCOMPARE(twins.result.bentCrossingCount, 1);
                QVERIFY(!twins.bentRecord(0)->curtainFromV);
            }
        }
        {
            // The mirror: the V reading from both passages of the H (exact
            // twins of the V-owned record), the H run cut so that only one
            // passage's H position lies in it. Suppressed whichever twin the
            // storage order kept: the kept one stands for both passages.
            BentFixture split = fixture;
            split.assemblies[h].stretches[0].lastSample = seedH + 2;
            BentHit elsewhere = fromV;
            elsewhere.segment = seedH + 5;
            for (const std::vector<BentHit>& order :
                 {std::vector<BentHit>{fromH, fromV, elsewhere}, std::vector<BentHit>{fromH, elsewhere, fromV}}) {
                const BentSolve twins = split.solve({{{h, v}, order}});
                QCOMPARE(twins.result.bentCrossingCount, 1);
                QVERIFY(!twins.bentRecord(0)->curtainFromV);
            }
        }
    }

    // A shared hit may win in BC and lose in AB. Neither it nor an exact
    // twin can then suppress a V reading on a passage that the actual AB
    // winner does not cover. Test hit permutations and strip-key order.
    void discardedSharedRayTwinsCannotSuppress()
    {
        World world;
        const std::size_t h = addH(world, 0.4, 0.9, 30000.0);
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
        BentFixture fixture(world, 1);
        const std::size_t seedH = hSampleAt(world, h, 0.4, 0.42);
        const std::size_t vMid = world.fibers[v].theta.size() / 2;
        fixture.assemblies[v].stretches[0].lastSample = vMid + 1;
        BentHit shared = fixture.hit(h, v, false, seedH, seedH + 2, 0.5, vMid, 0.5, 4.0, 1);
        BentHit earlier = shared;
        earlier.s = 2.0;
        earlier.segment = vMid + 5; // outside the V-owned reading's run
        const BentHit fromV = fixture.hit(h, v, true, vMid, vMid + 3, 0.5, seedH, 0.5, 6.0, -1);
        for (const bool reverseStrips : {false, true}) {
            shared.contributingStrips = {{seedH, seedH + 1}, {seedH + 1, seedH + 2}};
            earlier.contributingStrips = {shared.contributingStrips[reverseStrips ? 1 : 0]};
            std::array<int, 4> permutation{0, 1, 2, 3};
            const std::array<BentHit, 4> hits{shared, shared, earlier, fromV};
            do {
                std::vector<BentHit> ordered;
                for (const int index : permutation) ordered.push_back(hits[index]);
                const BentSolve solved = fixture.solve({{{h, v}, ordered}});
                QCOMPARE(solved.result.bentCrossingCount, 2);
                QCOMPARE(solved.bentIndices.size(), std::size_t{2});
                QVERIFY(solved.bentRecord(1)->curtainFromV);
            } while (std::next_permutation(permutation.begin(), permutation.end()));
        }
    }

    // The assembly's decisions: a corrected stretch negates the side (the
    // V on the provisional outward side reads Outside), a withheld stretch
    // records without constraining, a seam pair withholds everything.
    void assemblyDecisionsApplyToTheRecords()
    {
        World world;
        const std::size_t h = addH(world, 0.4, 0.9, 30000.0);
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
        const BentFixture fixture(world, 1);
        const std::size_t seed = hSampleAt(world, h, 0.4, 0.42);
        const std::size_t vMid = world.fibers[v].theta.size() / 2;
        const BentHit hit = fixture.hit(h, v, false, seed, seed + 2, 0.5, vMid, 0.5, 4.0, 1);
        {
            BentFixture corrected = fixture;
            corrected.assemblies[h].stretches[0].finalSign = -1;
            corrected.assemblies[h].stretches[0].corrected = true;
            const BentSolve solved = corrected.solve({{{h, v}, {hit}}});
            QCOMPARE(solved.bentRecord(0)->kind, CrossingKind::Outside);
            QCOMPARE(solved.bentRecord(0)->deltaR, 4.0);
            QCOMPARE(solved.bentRecord(0)->anchor, 2);
            // A strict Outside: the V one winding inside.
            QCOMPARE(solved.result.placements[v].turns - solved.result.placements[h].turns, -1.0);
        }
        for (const BentDisposition reason :
             {BentDisposition::NoWitness, BentDisposition::Contested, BentDisposition::Unoriented,
              BentDisposition::Unsupported}) {
            BentFixture withheld = fixture;
            withheld.assemblies[h].stretches[0].withheld = true;
            withheld.assemblies[h].stretches[0].disposition = reason;
            withheld.assemblies[h].stretches[0].anchor = 0;
            const BentSolve solved = withheld.solve({{{h, v}, {hit}}});
            QCOMPARE(solved.result.bentCrossingCount, 0);
            QCOMPARE(solved.result.withheldCount, 1);
            QVERIFY(solved.bentRecord(0)->withheld);
            QCOMPARE(solved.bentRecord(0)->bentReason, static_cast<int>(reason));
            QCOMPARE(solved.bentRecord(0)->status, CrossingStatus::Used);
            QCOMPARE(solved.result.droppedCrossingCount, 0);
        }
        {
            const BentSolve solved = fixture.solve({{{h, v}, {hit}}}, true);
            QVERIFY(solved.bentRecord(0)->withheld);
            QCOMPARE(solved.bentRecord(0)->bentReason,
                     static_cast<int>(BentDisposition::SeamWithheld));
        }
        {
            // No assembly for the owner: withheld.
            BentFixture none = fixture;
            none.assemblies[h].stretches.clear();
            const BentSolve solved = none.solve({{{h, v}, {hit}}});
            QVERIFY(solved.bentRecord(0)->withheld);
        }
    }

    // Straight readings on samples the assembly marked radially inverted
    // are set aside before merging: the detection goes, its segment and
    // translate become uncovered, and the count says so; an interpolated
    // inwardness below the gate at the hit sets it aside too.
    void radialInversionSetsStraightReadingsAside()
    {
        World world = threeWindingWorld();
        const std::size_t h = 0;
        const std::size_t v = 1;
        const BentFixture plain(world);
        const BentSolve before = plain.solve({});
        QVERIFY(before.result.crossings.size() >= 3);
        QCOMPARE(before.result.radialInvertedCount, 0);
        // The straight crossing of (h, v) sits at w = 0.3 on the H fiber.
        const std::size_t at = hSampleAt(world, h, 0.05, 0.3);
        BentFixture inverted = plain;
        for (std::size_t k = at - 1; k <= at + 2; ++k) {
            inverted.assemblies[h].radialInverted[k] = 1;
        }
        const BentSolve after = inverted.solve({});
        // The H segment at w = 0.3 meets the V at 0.3 and, one and two
        // translates on, the Vs at 1.3 and 2.3: every detection on those
        // samples goes.
        QCOMPARE(after.result.radialInvertedCount, 3);
        QCOMPARE(after.result.crossings.size(), before.result.crossings.size() - 3);
        for (const Crossing& c : after.result.crossings) {
            QVERIFY(c.hSegment + 1 < at - 1 || c.hSegment > at + 2);
        }
        // The interpolated inwardness over the same samples sets the same
        // detections aside.
        BentFixture inward = plain;
        for (std::size_t k = at - 1; k <= at + 2; ++k) {
            inward.inwardness[h][k] = -0.9;
        }
        const BentSolve interpolated = inward.solve({});
        QCOMPARE(interpolated.result.radialInvertedCount, after.result.radialInvertedCount);
    }

    // Samples a trace excludes take no part in the ordinal placement, as
    // query or as point: an island whose every sample is excluded is
    // unresolved instead of anchored by radius.
    void excludedSamplesLeaveTheOrdinalPlacement()
    {
        World world = threeWindingWorld();
        // An island: a V fiber above the H's height, anchored by radius
        // against the H samples just below it.
        const std::size_t island = addV(world, 1.6, 31000.0, 34000.0);
        SolverParams params;
        const SolveResult anchored = solveWindings(world.fibers, world.links, params);
        QCOMPARE(anchored.placements[island].anchor, ComponentAnchor::Radius);
        world.fibers[island].ordinalExcluded.assign(world.fibers[island].theta.size(), 1);
        const SolveResult excluded = solveWindings(world.fibers, world.links, params);
        QCOMPARE(excluded.placements[island].anchor, ComponentAnchor::Unresolved);
        // The converse: the island's samples all count, every sample of the
        // anchored fibers is excluded from the point set - nothing to
        // anchor against, unresolved again.
        world.fibers[island].ordinalExcluded.clear();
        for (std::size_t f = 0; f < world.fibers.size(); ++f) {
            if (f != island) {
                world.fibers[f].ordinalExcluded.assign(world.fibers[f].theta.size(), 1);
            }
        }
        const SolveResult noPoints = solveWindings(world.fibers, world.links, params);
        QCOMPARE(noPoints.placements[island].anchor, ComponentAnchor::Unresolved);
    }

    // Records identical in every geometric term but withheld for different
    // reasons (two passages of the owner through one place, a field hole
    // between, one run unwitnessed and one contested) are ordered by their
    // disposition, not by the run index the storage order assigns.
    void equalGeometryRecordsOrderByDisposition()
    {
        World world;
        const std::size_t h = addH(world, 0.4, 0.9, 30000.0);
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
        BentFixture fixture(world, 1);
        const std::size_t seed = hSampleAt(world, h, 0.4, 0.42);
        const std::size_t vMid = world.fibers[v].theta.size() / 2;
        BentStretchDecision noWitness = fixture.assemblies[h].stretches[0];
        noWitness.withheld = true;
        noWitness.anchor = 0;
        noWitness.disposition = BentDisposition::NoWitness;
        BentStretchDecision contested = noWitness;
        contested.disposition = BentDisposition::Contested;
        BentHit base = fixture.hit(h, v, false, seed, seed + 2, 0.5, vMid, 0.25, 4.0, 1);
        base.startAX = 1.0;
        base.startBX = 2.0;
        std::vector<std::vector<int>> orders;
        for (const bool swapped : {false, true}) {
            fixture.assemblies[h].stretches = swapped ? std::vector<BentStretchDecision>{contested, noWitness}
                                                      : std::vector<BentStretchDecision>{noWitness, contested};
            BentHit first = base;
            first.stretch = 0;
            BentHit second = base;
            second.stretch = 1;
            const BentSolve solved = swapped ? fixture.solve({{{h, v}, {second, first}}})
                                             : fixture.solve({{{h, v}, {first, second}}});
            std::vector<int> reasons;
            for (const Crossing& c : solved.result.crossings) {
                if (c.bent) {
                    QVERIFY(c.withheld);
                    reasons.push_back(c.bentReason);
                }
            }
            orders.push_back(reasons);
        }
        QCOMPARE(orders[0], (std::vector<int>{static_cast<int>(BentDisposition::NoWitness),
                                               static_cast<int>(BentDisposition::Contested)}));
        QCOMPARE(orders[1], orders[0]);
    }

    // A V-curtain record's V provenance is the trace segment its seeds'
    // interpolated position falls in, resolved to the branch holding it:
    // seeds several samples apart on the returning limb of a folded V.
    void vCurtainProvenanceResolvesTheBranch()
    {
        World world;
        const std::size_t h = addH(world, 0.4, 0.9, 30000.0);
        // A V that climbs and comes back down: two branches.
        FiberTrace v;
        v.hvTag = 'V';
        for (double z = 27000.0; z <= 33000.0 + 1e-9; z += 25.0) {
            v.theta.push_back(kTwoPi * 0.3);
            v.z.push_back(z);
            v.radius.push_back(sheetR(0.3, z));
        }
        for (double z = 32975.0; z >= 27000.0 - 1e-9; z -= 25.0) {
            v.theta.push_back(kTwoPi * 0.3);
            v.z.push_back(z);
            v.radius.push_back(sheetR(0.3, z) + 300.0);
        }
        world.fibers.push_back(std::move(v));
        world.trueM.push_back(0);
        const std::size_t vIndex = world.fibers.size() - 1;
        const BentFixture fixture(world, 1);
        QCOMPARE(fixture.canonical[vIndex].branches.size(), std::size_t{2});
        const std::size_t seedH = hSampleAt(world, h, 0.4, 0.42);
        // Seeds 300 and 304 lie on the returning limb; the position between
        // them is sample 302.
        const BentHit hit = fixture.hit(h, vIndex, true, 300, 304, 0.5, seedH, 0.5, 6.0, -1);
        const BentSolve solved = fixture.solve({{{h, vIndex}, {hit}}});
        const Crossing* record = solved.bentRecord(0);
        QVERIFY(record != nullptr);
        QVERIFY(record->vBranchSegment != Crossing::kNoSample);
        const auto& branch = fixture.canonical[vIndex].branches[record->vBranch];
        const std::size_t s0 = branch.sample[record->vBranchSegment];
        const std::size_t s1 = branch.sample[record->vBranchSegment + 1];
        QVERIFY((s0 == 302 && s1 == 303) || (s0 == 303 && s1 == 302));
        QVERIFY(std::abs(record->vU - (s0 == 302 ? 0.0 : 1.0)) < 1e-9);
    }

    // Folds: the sign rule reads the next layer along the normal, which a
    // fold puts out of order. One curtain reading both kinds of the pair on
    // one translate, or the two curtains disagreeing on one translate,
    // withholds every usable reading of that translate as Folded; a reading
    // beyond the owner's own polyline crossing the strip side is withheld
    // as BeyondFold; readings on other translates stand.
    void foldSignaturesWithholdTheTranslate()
    {
        World world;
        const std::size_t h = addH(world, 0.4, 0.9, 30000.0);
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
        const BentFixture fixture(world, 1);
        const std::size_t seedH = hSampleAt(world, h, 0.4, 0.42);
        const std::size_t seedH2 = hSampleAt(world, h, 0.4, 0.6);
        const std::size_t vMid = world.fibers[v].theta.size() / 2;
        const BentHit outward = fixture.hit(h, v, false, seedH, seedH + 2, 0.5, vMid, 0.5, 4.0, 1);
        BentHit inward = fixture.hit(h, v, false, seedH2, seedH2 + 2, 0.5, vMid, 0.5, 6.0, -1);
        inward.startAX = 50.0;
        inward.startBX = 52.0;
        const long long n = fixture.expectedN(h, v, outward);
        QCOMPARE(fixture.expectedN(h, v, inward), n);
        {
            // Both kinds from the H curtain on one translate.
            const BentSolve solved = fixture.solve({{{h, v}, {outward, inward}}});
            QCOMPARE(solved.result.bentCrossingCount, 0);
            QCOMPARE(solved.result.withheldCount, 2);
            for (const Crossing& c : solved.result.crossings) {
                if (c.bent) {
                    QCOMPARE(c.bentReason, static_cast<int>(BentDisposition::Folded));
                }
            }
        }
        {
            // The two curtains disagree: the V curtain reads the H on its
            // outward side (Outside) while the H curtain reads Inside.
            const BentHit fromV = fixture.hit(h, v, true, vMid, vMid + 3, 0.5, seedH, 0.5, 6.0, 1);
            QCOMPARE(fixture.expectedN(h, v, fromV), n);
            const BentSolve solved = fixture.solve({{{h, v}, {outward, fromV}}});
            QCOMPARE(solved.result.bentCrossingCount, 0);
            QCOMPARE(solved.result.withheldCount, 2);
            // An agreeing V reading on another translate stands.
            BentHit otherTurn = fixture.hit(h, v, true, vMid, vMid + 3, 0.5, seedH, 0.5, 6.0, -1);
            otherTurn.theta += kTwoPi;
            QCOMPARE(fixture.expectedN(h, v, otherTurn), n + 1);
            const BentSolve mixed = fixture.solve({{{h, v}, {outward, fromV, otherTurn}}});
            QCOMPARE(mixed.result.bentCrossingCount, 1);
            QCOMPARE(mixed.result.withheldCount, 2);
        }
        {
            // Beyond the owner's own fold.
            BentHit beyond = outward;
            beyond.selfCrossingS = 3.0;
            const BentSolve solved = fixture.solve({{{h, v}, {beyond}}});
            QCOMPARE(solved.result.bentCrossingCount, 0);
            QCOMPARE(solved.bentRecord(0)->bentReason, static_cast<int>(BentDisposition::BeyondFold));
            BentHit before = outward;
            before.selfCrossingS = 5.0;
            const BentSolve kept = fixture.solve({{{h, v}, {before}}});
            QCOMPARE(kept.result.bentCrossingCount, 1);
        }
    }

    // The classified bent records come out in one canonical order whichever
    // way either fiber's samples are stored: two strips of the H crossed by
    // the V, both fibers reversed.
    void bentRecordOrderIsStorageIndependent()
    {
        World world;
        const std::size_t h = addH(world, 0.4, 0.9, 30000.0);
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
        const std::size_t hCount = world.fibers[h].theta.size();
        const std::size_t vCount = world.fibers[v].theta.size();
        const std::size_t seedA = hSampleAt(world, h, 0.4, 0.42);
        const std::size_t seedB = hSampleAt(world, h, 0.4, 0.6);
        const std::size_t vMid = vCount / 2;
        std::vector<std::vector<double>> orders;
        for (const bool reverseH : {false, true}) {
            for (const bool reverseV : {false, true}) {
                World stored = world;
                if (reverseH) {
                    reverseFiber(stored.fibers[h]);
                }
                if (reverseV) {
                    reverseFiber(stored.fibers[v]);
                }
                const auto hIndex = [&](std::size_t i) { return reverseH ? hCount - 1 - i : i; };
                const auto vIndex = [&](std::size_t i) { return reverseV ? vCount - 1 - i : i; };
                const BentFixture fixture(stored, 1);
                // Segment + t of the V hit: the same physical point either way.
                const auto vSeg = [&](std::size_t segment, double t, std::size_t& outSegment,
                                      double& outT) {
                    if (!reverseV) {
                        outSegment = segment;
                        outT = t;
                    } else {
                        outSegment = vCount - 2 - segment;
                        outT = 1.0 - t;
                    }
                };
                std::size_t s1 = 0;
                double t1 = 0.0;
                vSeg(vMid, 0.25, s1, t1);
                std::size_t s2 = 0;
                double t2 = 0.0;
                vSeg(vMid + 6, 0.5, s2, t2);
                BentHit first = fixture.hit(h, v, false, hIndex(seedA), hIndex(seedA + 2),
                                            0.5, s1, t1, 4.0, 1, 1.0, 10.0);
                BentHit second = fixture.hit(h, v, false, hIndex(seedB), hIndex(seedB + 2),
                                             0.5, s2, t2, 4.0, 1, 1.0, 20.0);
                // The rays' start points: the physical seed positions.
                first.startAX = 100.0;
                first.startBX = 102.0;
                second.startAX = 160.0;
                second.startBX = 162.0;
                const BentSolve solved = (reverseH ^ reverseV)
                    ? fixture.solve({{{h, v}, {second, first}}})
                    : fixture.solve({{{h, v}, {first, second}}});
                std::size_t bentCount = 0;
                for (const Crossing& c : solved.result.crossings) {
                    bentCount += c.bent ? 1 : 0;
                }
                QCOMPARE(bentCount, std::size_t{2});
                const std::vector<double> terms = bentTerms(solved.result);
                orders.push_back(terms);
            }
        }
        for (std::size_t i = 1; i < orders.size(); ++i) {
            QCOMPARE(orders[i].size(), orders[0].size());
            for (std::size_t k = 0; k < orders[0].size(); ++k) {
                QVERIFY2(std::abs(orders[i][k] - orders[0][k]) < 1e-9,
                         qPrintable(QStringLiteral("order %1 term %2: %3 vs %4").arg(i).arg(k).arg(orders[i][k]).arg(orders[0][k])));
            }
        }
    }


    // Rays A, B, C in a row: the crosser meets AB at s=10, then shared
    // ray B at s=12. The shared hit competes in both AB and BC, regardless
    // of which strip represents it after rotation. Only the s=10 reading
    // survives, under all four owner/crosser storage orders.
    void sharedRayHitsReduceThroughTheClassification_data()
    {
        QTest::addColumn<int>("quarterTurns");
        for (int rotation = 0; rotation < 4; ++rotation) {
            QTest::newRow(QByteArray::number(rotation * 90).constData()) << rotation;
        }
    }

    void sharedRayHitsReduceThroughTheClassification()
    {
        QFETCH(int, quarterTurns);
        const auto rotate = [quarterTurns](cv::Vec3d p) {
            for (int k = 0; k < quarterTurns; ++k) p = cv::Vec3d(-p[1], p[0], p[2]);
            return p;
        };
        using vc3d::fiber_map::bent::BentRay;
        const auto ray = [](const cv::Vec3d& start, std::size_t sample) {
            BentRay r;
            r.startSample = sample;
            r.side = 1;
            for (int k = 0; k <= 4; ++k) {
                r.points.push_back(start + cv::Vec3d(0.0, 0.0, 4.0 * k));
                r.s.push_back(4.0 * k);
                r.theta.push_back(0.0);
            }
            return r;
        };
        CurtainCase given;
        given.hPoints = {cv::Vec3d(1000.0, -4.0, 0.0), cv::Vec3d(1000.0, 4.0, 0.0), cv::Vec3d(1000.0, 12.0, 0.0)};
        given.rays = {ray(given.hPoints[0], 0), ray(given.hPoints[1], 1), ray(given.hPoints[2], 2)};
        given.vPoints = {cv::Vec3d(999.0, 0.0, 10.0), cv::Vec3d(1001.0, 0.0, 10.0), cv::Vec3d(1001.0, 4.0, 12.0),
                         cv::Vec3d(999.0, 4.0, 12.0)};
        for (auto& p : given.hPoints) p = rotate(p);
        for (auto& p : given.vPoints) p = rotate(p);
        for (auto& r : given.rays) for (auto& p : r.points) p = rotate(p);
        const auto expected = rotate(cv::Vec3d(1000.0, 0.0, 10.0));
        bool ok = false;
        const BentSolve solved = solveCurtainUnderReversals(given, &ok);
        QVERIFY(ok);
        QCOMPARE(solved.bentIndices.size(), std::size_t{1});
        const Crossing& record = *solved.bentRecord(0);
        QVERIFY(!record.withheld);
        QVERIFY(!record.touch && !record.tangential);
        QCOMPARE(record.status, CrossingStatus::Used);
        QVERIFY(std::abs(record.rayLengthVx - 10.0) < 1e-9);
        QVERIFY(std::abs(record.hitX - expected[0]) < 1e-9);
        QVERIFY(std::abs(record.hitY - expected[1]) < 1e-9);
        QVERIFY(std::abs(record.hitZ - expected[2]) < 1e-9);
        QVERIFY(std::abs(record.transversality - 1.0) < 1e-9);
        std::size_t bentEvents = 0;
        for (const Crossing& event : solved.result.events) {
            bentEvents += event.bent ? 1 : 0;
        }
        QCOMPARE(bentEvents, std::size_t{1});
        // Against opposing equality evidence (a link putting the V one
        // winding inside the H at the reading's own translate) the retained
        // reading is the one dropped, with violation 1, and the dropped
        // event is that reading at its hit under the four storage orders.
        given.links.push_back(LinkInput{0, 1, 1, 0, -1 - static_cast<int>(record.n)});
        const BentSolve opposed = solveCurtainUnderReversals(given, &ok);
        QVERIFY(ok);
        QCOMPARE(opposed.bentIndices.size(), std::size_t{1});
        const Crossing& dropped = *opposed.bentRecord(0);
        QCOMPARE(dropped.status, CrossingStatus::Dropped);
        QCOMPARE(dropped.violationTurns, 1.0);
        QVERIFY(std::abs(dropped.rayLengthVx - 10.0) < 1e-9);
        QVERIFY(std::abs(dropped.hitZ - 10.0) < 1e-9);
        QVERIFY(opposed.result.droppedLinks.empty());
        std::size_t droppedEvents = 0;
        for (const Crossing& event : opposed.result.events) {
            if (event.bent) {
                QCOMPARE(event.status, CrossingStatus::Dropped);
                QCOMPARE(event.violationTurns, 1.0);
                QVERIFY(std::abs(event.hitZ - 10.0) < 1e-9);
                ++droppedEvents;
            }
        }
        QCOMPARE(droppedEvents, std::size_t{1});
    }

    // A reading withheld for want of a witness, seeded on voted samples, is
    // fold evidence only within its own stretch: opposite kinds there
    // survive reversing the provisional orientation. A single unwitnessed
    // kind cannot be compared with the H's independently anchored sign.
    void unwitnessedVotedReadingsAreFoldEvidence()
    {
        World world;
        const std::size_t h = addH(world, 0.4, 0.9, 30000.0);
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
        BentFixture fixture(world, 1);
        const std::size_t seedH = hSampleAt(world, h, 0.4, 0.42);
        const std::size_t vMid = world.fibers[v].theta.size() / 2;
        const BentHit fromH = fixture.hit(h, v, false, seedH, seedH + 2, 0.5, vMid, 0.5, 4.0, 1);
        BentHit fromV = fixture.hit(h, v, true, vMid, vMid + 3, 0.5, seedH, 0.5, 6.0, 1);
        QCOMPARE(fixture.expectedN(h, v, fromH), fixture.expectedN(h, v, fromV));
        BentStretchDecision& vStretch = fixture.assemblies[v].stretches[0];
        vStretch.withheld = true;
        vStretch.anchor = 0;
        vStretch.disposition = BentDisposition::NoWitness;
        fixture.assemblies[v].votedOrientation.assign(world.fibers[v].theta.size(), 1);
        {
            const BentSolve solved = fixture.solve({{{h, v}, {fromH, fromV}}});
            QCOMPARE(solved.result.bentCrossingCount, 1);
            for (const Crossing& c : solved.result.crossings) {
                if (c.bent && !c.curtainFromV) {
                    QVERIFY(!c.withheld);
                }
                if (c.bent && c.curtainFromV) {
                    QVERIFY(c.withheld);
                    QCOMPARE(c.bentReason, static_cast<int>(BentDisposition::NoWitness));
                }
            }
        }
        {
            // Agreeing kinds: no signature, the H reading stands.
            BentHit agreeing = fromV;
            agreeing.side = -1;
            const BentSolve solved = fixture.solve({{{h, v}, {fromH, agreeing}}});
            QCOMPARE(solved.result.bentCrossingCount, 1);
            QVERIFY(!solved.bentRecord(0)->curtainFromV);
        }
        BentHit otherSide = fromV;
        otherSide.side = -1;
        for (const int sign : {-1, 1}) {
            vStretch.finalSign = sign;
            const BentSolve solved = fixture.solve({{{h, v}, {fromH, fromV, otherSide}}});
            QCOMPARE(solved.result.bentCrossingCount, 0);
            QCOMPARE(solved.bentRecord(0)->bentReason, static_cast<int>(BentDisposition::Folded));
        }
        {
            // The same owner in independently oriented stretches is not a
            // relative orientation either: this was a sibling of the
            // cross-owner false veto.
            fixture.assemblies[v].stretches.push_back(vStretch);
            otherSide.stretch = 1;
            const BentSolve solved = fixture.solve({{{h, v}, {fromH, fromV, otherSide}}});
            QCOMPARE(solved.result.bentCrossingCount, 1);
            otherSide.stretch = 0;
        }
        {
            // No vote behind the V's seeds: no evidence.
            fixture.assemblies[v].votedOrientation.assign(world.fibers[v].theta.size(), 0);
            const BentSolve solved = fixture.solve({{{h, v}, {fromH, fromV, otherSide}}});
            QCOMPARE(solved.result.bentCrossingCount, 1);
            QVERIFY(!solved.bentRecord(0)->curtainFromV);
        }
    }

    // PHerc0139's lt-200 reciprocal curtains agree on the near passage,
    // then disagree at a long, near-tangential reading on another translate.
    // No flip of the unanchored stretch reconciles both passages. In
    // contrast a shelf with uniformly opposite signs has no contradiction.
    void reciprocalReadingsNeedAConsistentRelativeOrientation_data()
    {
        QTest::addColumn<bool>("unwitnessedH");
        QTest::addColumn<int>("provisionalSign");
        QTest::addColumn<bool>("splitRuns");
        for (const bool split : {false, true}) {
            const QByteArray suffix = split ? "-split-runs" : "-one-run";
            QTest::newRow(("unwitnessed-V" + suffix).constData()) << false << 1 << split;
            QTest::newRow(("flipped-V" + suffix).constData()) << false << -1 << split;
            QTest::newRow(("unwitnessed-H" + suffix).constData()) << true << 1 << split;
            QTest::newRow(("flipped-H" + suffix).constData()) << true << -1 << split;
        }
    }

    void reciprocalReadingsNeedAConsistentRelativeOrientation()
    {
        QFETCH(bool, unwitnessedH);
        QFETCH(int, provisionalSign);
        QFETCH(bool, splitRuns);
        for (int order = 0; order < 4; ++order) {
            World world;
            const std::size_t h = addH(world, 0.4, 1.9, 30000.0);
            const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
            if (order & 1) reverseFiber(world.fibers[h]);
            if (order & 2) reverseFiber(world.fibers[v]);
            BentFixture fixture(world, 1);
            for (std::size_t f : {h, v}) {
                // This test supplies curtain evidence only.
                fixture.assemblies[f].radialInverted.assign(world.fibers[f].theta.size(), 1);
            }
            const auto seed = hSampleAt(world, h, 0.4, 0.42);
            const auto laterSeed = hSampleAt(world, h, 0.4, 1.42);
            const auto vMid = world.fibers[v].theta.size() / 2;
            const auto hit = [&](bool ownerIsV, std::size_t a, std::size_t b, std::size_t segment,
                                 double length, int side) {
                const bool reverseOwner = order & (ownerIsV ? 2 : 1);
                const bool reverseOther = order & (ownerIsV ? 1 : 2);
                const auto ownerSize = world.fibers[ownerIsV ? v : h].theta.size();
                const auto otherSize = world.fibers[ownerIsV ? h : v].theta.size();
                if (reverseOwner) { a = ownerSize - 1 - a; b = ownerSize - 1 - b; }
                if (reverseOther) segment = otherSize - 2 - segment;
                return fixture.hit(h, v, ownerIsV, a, b, 0.5, segment, 0.5, length, side);
            };
            BentHit nearH = hit(false, seed, seed + 2, vMid, 69.0, 1);
            BentHit nearV = hit(true, vMid, vMid + 2, seed, 69.0, -1);
            BentHit farH = hit(false, laterSeed, laterSeed + 2, vMid, 359.0, -1);
            BentHit farV = hit(true, vMid, vMid + 2, laterSeed, 357.0, -1);
            if (splitRuns) {
                // Real fibers can leave and re-enter the ill-conditioned
                // band without changing their continuous orientation.
                auto& runs = fixture.assemblies[h].stretches;
                runs.push_back(runs.front());
                for (auto& run : runs) {
                    run.orientationFirstSample = 0;
                    run.orientationLastSample = world.fibers[h].theta.size() - 1;
                }
                runs[0].firstSample = std::min(nearH.seedA, nearH.seedB);
                runs[0].lastSample = std::max(nearH.seedA, nearH.seedB);
                runs[1].firstSample = std::min(farH.seedA, farH.seedB);
                runs[1].lastSample = std::max(farH.seedA, farH.seedB);
                farH.stretch = 1;
            }
            for (BentHit* hit : {&farH, &farV}) {
                hit->turnDeg = 20.0;
                hit->minConditioning = 0.03;
            }
            QCOMPARE(fixture.expectedN(h, v, nearH), fixture.expectedN(h, v, nearV));
            QCOMPARE(fixture.expectedN(h, v, farH), fixture.expectedN(h, v, farV));
            QVERIFY(fixture.expectedN(h, v, nearH) != fixture.expectedN(h, v, farH));
            const auto provisionalOwner = unwitnessedH ? h : v;
            auto& assembly = fixture.assemblies[provisionalOwner];
            for (auto& stretch : assembly.stretches) {
                stretch.withheld = true;
                stretch.anchor = 0;
                stretch.disposition = BentDisposition::NoWitness;
                stretch.finalSign = provisionalSign;
            }
            assembly.votedOrientation.assign(world.fibers[provisionalOwner].theta.size(), 1);
            const auto solve = [&](std::vector<BentHit> hits) { return fixture.solve({{{h, v}, std::move(hits)}}); };
            const auto checkFolded = [&](const BentSolve& solved) {
                QCOMPARE(solved.result.bentCrossingCount, 0);
                int folded = 0, unwitnessed = 0;
                for (const auto index : solved.bentIndices) {
                    const auto& c = solved.result.crossings[index];
                    QVERIFY(c.withheld);
                    if (c.curtainFromV != unwitnessedH) {
                        QCOMPARE(c.bentReason, static_cast<int>(BentDisposition::NoWitness));
                        ++unwitnessed;
                    } else {
                        QCOMPARE(c.bentReason, static_cast<int>(BentDisposition::Folded));
                        ++folded;
                    }
                }
                QCOMPARE(folded, 2);
                QCOMPARE(unwitnessed, 2);
            };
            checkFolded(solve({nearH, nearV, farH, farV}));
            checkFolded(solve({farV, farH, nearV, nearH}));
            // Length is not an ordering or eligibility threshold: even the
            // very same sign signature on short rays remains inconsistent.
            auto shortH = farH, shortV = farV;
            shortH.s = shortV.s = 4.0;
            checkFolded(solve({nearH, nearV, shortH, shortV}));
            // One reciprocal passage cannot establish a relative orientation.
            QCOMPARE(solve({farH, farV}).result.bentCrossingCount, 1);
            // Nor can two passages with a consistent relative sign.
            auto consistentH = farH;
            consistentH.side = nearH.side;
            QCOMPARE(solve({nearH, nearV, consistentH, farV}).result.bentCrossingCount, 2);
            // A different orientation stretch is independent; neither its
            // sign nor a reading outside the reciprocal run is a bridge.
            assembly.stretches.push_back(assembly.stretches.front());
            auto separate = unwitnessedH ? farH : farV;
            separate.stretch = assembly.stretches.size() - 1;
            assembly.stretches.back().orientationFirstSample = vc3d::fiber_map::winding::kNoSample;
            assembly.stretches.back().orientationLastSample = vc3d::fiber_map::winding::kNoSample;
            QCOMPARE(solve(unwitnessedH ? std::vector<BentHit>{nearH, nearV, separate, farV}
                                       : std::vector<BentHit>{nearH, nearV, farH, separate})
                         .result.bentCrossingCount, 2);
            // Missing votes and unsound geometry cannot create the signature.
            for (const int defect : {0, 1, 2, 3, 4}) {
                auto bad = unwitnessedH ? nearH : nearV;
                if (defect == 0) bad.theta += 0.4 * kTwoPi;
                if (defect == 1) bad.transversality = 0.01;
                if (defect == 2) bad.selfCrossingS = 1.0;
                if (defect == 3) { bad.turnDeg = 79.0; bad.minConditioning = 0.03; }
                if (defect == 4) assembly.votedOrientation.assign(assembly.votedOrientation.size(), 0);
                QCOMPARE(solve(unwitnessedH ? std::vector<BentHit>{bad, nearV, farH, farV}
                                           : std::vector<BentHit>{nearH, bad, farH, farV})
                             .result.bentCrossingCount, 2);
            }
        }
    }

    // The shard oracle must compare the crease inputs bit for bit, even
    // when a one-bit change leaves the classification unchanged.
    void shardOracleComparesCreaseFields_data()
    {
        QTest::addColumn<bool>("conditioning");
        QTest::newRow("turn") << false;
        QTest::newRow("conditioning") << true;
    }

    void shardOracleComparesCreaseFields()
    {
        QFETCH(bool, conditioning);
        PairDetections original;
        original.bentHits.emplace_back();
        original.bentHits.front().turnDeg = 79.0;
        original.bentHits.front().minConditioning = 0.03;
        PairDetections changed = original;
        QVERIFY(identicalPairDetections(original, changed));
        double& value = conditioning ? changed.bentHits.front().minConditioning : changed.bentHits.front().turnDeg;
        value = std::nextafter(value, std::numeric_limits<double>::infinity());
        QVERIFY(!identicalPairDetections(original, changed));
    }

    void shardOracleComparesContributingStrips()
    {
        PairDetections original;
        original.bentHits.emplace_back();
        original.bentHits.front().contributingStrips = {{0, 1}, {1, 2}};
        PairDetections changed = original;
        QVERIFY(identicalPairDetections(original, changed));
        changed.bentHits.front().contributingStrips.pop_back();
        QVERIFY(!identicalPairDetections(original, changed));
    }

    // The crease rule: a reading whose ray turned more than the crease
    // angle while the field's axis went through tangential is withheld as
    // crease-crossed; either alone leaves it standing; a disabled rule
    // (angle 0) leaves it standing.
    void readingsThroughACreaseAreWithheld()
    {
        World world;
        const std::size_t h = addH(world, 0.4, 0.9, 30000.0);
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
        BentFixture fixture(world, 1);
        const std::size_t seed = hSampleAt(world, h, 0.4, 0.42);
        const std::size_t vMid = world.fibers[v].theta.size() / 2;
        BentHit hit = fixture.hit(h, v, false, seed, seed + 2, 0.5, vMid, 0.5, 4.0, 1);
        const auto solveWith = [&](double turn, double cond) {
            BentHit h2 = hit;
            h2.turnDeg = turn;
            h2.minConditioning = cond;
            return fixture.solve({{{h, v}, {h2}}});
        };
        {
            const BentSolve solved = solveWith(79.0, 0.03);
            QVERIFY(solved.bentRecord(0)->withheld);
            QCOMPARE(solved.bentRecord(0)->bentReason, static_cast<int>(BentDisposition::CreaseCrossed));
            QCOMPARE(solved.result.bentCrossingCount, 0);
        }
        QVERIFY(!solveWith(79.0, 0.30).bentRecord(0)->withheld);
        QVERIFY(!solveWith(20.0, 0.03).bentRecord(0)->withheld);
        QVERIFY(!solveWith(60.0, 0.03).bentRecord(0)->withheld);
        fixture.assemblies[h].creaseTurnDeg = 0.0;
        QVERIFY(!solveWith(79.0, 0.03).bentRecord(0)->withheld);
    }

    // The crease rule is decided before any reading competes: an H reading
    // through a crease neither reduces a later clean H reading nor
    // suppresses a clean V reading of the same place and translate.
    void creaseReadingsNeitherReduceNorSuppress()
    {
        World world;
        const std::size_t h = addH(world, 0.4, 0.9, 30000.0);
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
        const BentFixture fixture(world, 1);
        const std::size_t seedH = hSampleAt(world, h, 0.4, 0.42);
        const std::size_t vMid = world.fibers[v].theta.size() / 2;
        BentHit fromH = fixture.hit(h, v, false, seedH, seedH + 2, 0.5, vMid, 0.5, 4.0, 1);
        const BentHit fromV = fixture.hit(h, v, true, vMid, vMid + 3, 0.5, seedH, 0.5, 6.0, -1);
        QCOMPARE(fixture.expectedN(h, v, fromH), fixture.expectedN(h, v, fromV));
        fromH.turnDeg = 79.0;
        fromH.minConditioning = 0.03;
        for (const std::vector<BentHit>& order : {std::vector<BentHit>{fromH, fromV}, std::vector<BentHit>{fromV, fromH}}) {
            const BentSolve solved = fixture.solve({{{h, v}, order}});
            QCOMPARE(solved.result.bentCrossingCount, 1);
            int creased = 0;
            for (const Crossing& c : solved.result.crossings) {
                if (!c.bent) {
                    continue;
                }
                if (c.curtainFromV) {
                    QVERIFY(!c.withheld);
                } else {
                    QVERIFY(c.withheld);
                    QCOMPARE(c.bentReason, static_cast<int>(BentDisposition::CreaseCrossed));
                    ++creased;
                }
            }
            QCOMPARE(creased, 1);
        }
        BentHit laterH = fromH;
        laterH.s = fromH.s + 2.0;
        laterH.turnDeg = 20.0;
        laterH.minConditioning = 0.3;
        for (const std::vector<BentHit>& order : {std::vector<BentHit>{fromH, laterH},
                                                 std::vector<BentHit>{laterH, fromH}}) {
            const BentSolve solved = fixture.solve({{{h, v}, order}});
            QCOMPARE(solved.result.bentCrossingCount, 1);
            QCOMPARE(solved.bentIndices.size(), std::size_t{2});
            QCOMPARE(solved.bentRecord(0)->bentReason, static_cast<int>(BentDisposition::CreaseCrossed));
            QVERIFY(!solved.bentRecord(1)->withheld);
            QCOMPARE(solved.bentRecord(1)->rayLengthVx, laterH.s);
        }
    }

    // Unwitnessed fold evidence needs sound geometry: the same voted,
    // unwitnessed V stretch reading both sides is a fold signature, but
    // its second reading with a lift residual of 0.4 turn (ambiguous)
    // or too shallow a crossing, beyond its own fold, or through a crease
    // is no evidence, and the H reading stands.
    void unsoundUnwitnessedReadingsAreNoFoldEvidence()
    {
        World world;
        const std::size_t h = addH(world, 0.4, 0.9, 30000.0);
        const std::size_t v = addV(world, 0.3, 27000.0, 33000.0);
        BentFixture fixture(world, 1);
        const std::size_t seedH = hSampleAt(world, h, 0.4, 0.42);
        const std::size_t vMid = world.fibers[v].theta.size() / 2;
        const BentHit fromH = fixture.hit(h, v, false, seedH, seedH + 2, 0.5, vMid, 0.5, 4.0, 1);
        const BentHit fromV = fixture.hit(h, v, true, vMid, vMid + 3, 0.5, seedH, 0.5, 6.0, 1);
        BentHit otherSide = fromV;
        otherSide.side = -1;
        BentStretchDecision& vStretch = fixture.assemblies[v].stretches[0];
        vStretch.withheld = true;
        vStretch.anchor = 0;
        vStretch.disposition = BentDisposition::NoWitness;
        fixture.assemblies[v].votedOrientation.assign(world.fibers[v].theta.size(), 1);
        // Sound: evidence, the H reading folded.
        QCOMPARE(fixture.solve({{{h, v}, {fromH, fromV, otherSide}}}).result.bentCrossingCount, 0);
        {
            BentHit ambiguous = fromV;
            ambiguous.theta += 0.4 * kTwoPi;
            const BentSolve solved = fixture.solve({{{h, v}, {fromH, ambiguous, otherSide}}});
            QCOMPARE(solved.result.bentCrossingCount, 1);
            QVERIFY(!solved.bentRecord(0)->curtainFromV);
        }
        {
            BentHit shallow = fromV;
            shallow.transversality = 0.01;
            const BentSolve solved = fixture.solve({{{h, v}, {fromH, shallow, otherSide}}});
            QCOMPARE(solved.result.bentCrossingCount, 1);
            QVERIFY(!solved.bentRecord(0)->curtainFromV);
        }
        {
            BentHit beyond = fromV;
            beyond.selfCrossingS = 3.0;
            const BentSolve solved = fixture.solve({{{h, v}, {fromH, beyond, otherSide}}});
            QCOMPARE(solved.result.bentCrossingCount, 1);
            QVERIFY(!solved.bentRecord(0)->curtainFromV);
        }
        {
            BentHit creased = fromV;
            creased.turnDeg = 79.0;
            creased.minConditioning = 0.03;
            for (const std::vector<BentHit>& order : {std::vector<BentHit>{fromH, creased, otherSide},
                                                     std::vector<BentHit>{otherSide, creased, fromH}}) {
                const BentSolve solved = fixture.solve({{{h, v}, order}});
                QCOMPARE(solved.result.bentCrossingCount, 1);
                QVERIFY(!solved.bentRecord(0)->curtainFromV);
                QVERIFY(!solved.bentRecord(0)->withheld);
                QCOMPARE(solved.bentRecord(1)->bentReason, static_cast<int>(BentDisposition::NoWitness));
            }
            // Disabling the crease rule restores the voted fold evidence.
            fixture.assemblies[v].creaseTurnDeg = 0.0;
            const BentSolve open = fixture.solve({{{h, v}, {fromH, creased, otherSide}}});
            QCOMPARE(open.result.bentCrossingCount, 0);
            QCOMPARE(open.bentRecord(0)->bentReason, static_cast<int>(BentDisposition::Folded));
        }
    }

    // A shared-ray reading sees the folds of BOTH contributing strips.
    // Conversely a shared-ray self-crossing bounds both strips' interiors.
    // Rotation changes the representative strip, never fold eligibility.
    void sharedRayReadingsRespectEveryFoldLimit_data()
    {
        QTest::addColumn<int>("quarterTurns");
        QTest::addColumn<bool>("ownerIsV");
        QTest::addColumn<bool>("sharedFold");
        for (int rotation = 0; rotation < 4; ++rotation) {
            for (const bool fromV : {false, true}) {
                for (const bool shared : {false, true}) {
                    const QByteArray name = QByteArray::number(rotation * 90) +
                        (fromV ? "-V" : "-H") + (shared ? "-shared-fold" : "-interior-fold");
                    QTest::newRow(name.constData()) << rotation << fromV << shared;
                }
            }
        }
    }

    void sharedRayReadingsRespectEveryFoldLimit()
    {
        QFETCH(int, quarterTurns);
        QFETCH(bool, ownerIsV);
        QFETCH(bool, sharedFold);
        using vc3d::fiber_map::bent::BentRay;
        const auto rotate = [quarterTurns](cv::Vec3d p) {
            for (int k = 0; k < quarterTurns; ++k) {
                p = cv::Vec3d(-p[1], p[0], p[2]);
            }
            return p;
        };
        CurtainCase given;
        given.curtainFromV = ownerIsV;
        auto& owner = ownerIsV ? given.vPoints : given.hPoints;
        auto& other = ownerIsV ? given.hPoints : given.vPoints;
        // Three +z rays A, B, C. The return limb crosses AB (or the
        // shared ray B) at s=10 on the same winding. Its connector stays
        // outside the curtain until the returning segment crosses it.
        const double foldY = sharedFold ? 0.0 : -4.0;
        owner = {cv::Vec3d(1000.0, -8.0, 0.0), cv::Vec3d(1000.0, 0.0, 0.0),
                 cv::Vec3d(1000.0, 8.0, 0.0), cv::Vec3d(990.0, 12.0, 10.0),
                 cv::Vec3d(990.0, foldY, 10.0), cv::Vec3d(1010.0, foldY, 10.0)};
        for (auto& p : owner) p = rotate(p);
        for (std::size_t sample = 0; sample < 3; ++sample) {
            BentRay ray;
            ray.startSample = sample;
            ray.side = 1;
            ray.points = {owner[sample], owner[sample] + cv::Vec3d(0.0, 0.0, 30.0)};
            ray.s = {0.0, 30.0};
            ray.theta = {0.0, 0.0};
            given.rays.push_back(ray);
        }
        for (const double y : {-4.0, 0.0, 4.0}) {
            for (const double height : {5.0, 10.0, 20.0}) {
                other = {rotate(cv::Vec3d(980.0, y, height)), rotate(cv::Vec3d(1020.0, y, height))};
                bool ok = false;
                const BentSolve solved = solveCurtainUnderReversals(given, &ok);
                QVERIFY(ok);
                QCOMPARE(solved.bentIndices.size(), std::size_t{1});
                const Crossing& record = *solved.bentRecord(0);
                const bool beyond = height > 10.0 && (sharedFold || y <= 0.0);
                QCOMPARE(record.withheld, beyond);
                QCOMPARE(record.bentReason, static_cast<int>(beyond ? BentDisposition::BeyondFold
                                                                   : BentDisposition::Usable));
                QVERIFY(!record.touch && !record.tangential);
                QCOMPARE(record.curtainFromV, ownerIsV);
                QCOMPARE(record.rayLengthVx, height);
                QCOMPARE(solved.result.bentCrossingCount, beyond ? 0 : 1);
            }
        }
    }

    // The strip's two bounding rays contribute symmetrically, even when a
    // rigid rotation swaps their lexicographic order. The field telemetry
    // rotates with the rays; the classification's link anchors are fixed.
    void creaseFieldsAreCanonicalUnderReversal_data()
    {
        QTest::addColumn<int>("quarterTurns");
        QTest::addColumn<bool>("splitMeasures");
        QTest::addColumn<bool>("sharedRay");
        for (int rotation = 0; rotation < 4; ++rotation) {
            for (const bool split : {false, true}) {
                for (const bool shared : {false, true}) {
                    const QByteArray name = QByteArray::number(rotation * 90) +
                        (split ? "-split-measures" : "-one-ray") + (shared ? "-shared-ray" : "-strip");
                    QTest::newRow(name.constData()) << rotation << split << shared;
                }
            }
        }
    }

    void creaseFieldsAreCanonicalUnderReversal()
    {
        QFETCH(int, quarterTurns);
        QFETCH(bool, splitMeasures);
        QFETCH(bool, sharedRay);
        using vc3d::fiber_map::bent::BentRay;
        // Ray from (1000,-4,0) bends 80 degrees by its hit row through a
        // field that goes tangential; ray from (1000,4,0) runs straight up.
        BentRay bending;
        bending.startSample = 0;
        bending.side = 1;
        bending.points = {cv::Vec3d(1000.0, -4.0, 0.0), cv::Vec3d(1000.0, -4.0, 8.0), cv::Vec3d(1007.9, -4.0, 9.4),
                          cv::Vec3d(1015.8, -4.0, 10.8)};
        bending.s = {0.0, 8.0, 16.0, 24.0};
        bending.theta = {0.0, 0.0, 0.0, 0.0};
        bending.conditioning = {0.3, 0.03, 0.3, 0.3};
        BentRay straight;
        straight.startSample = 1;
        straight.side = 1;
        straight.points = {cv::Vec3d(1000.0, 4.0, 0.0), cv::Vec3d(1000.0, 4.0, 8.0), cv::Vec3d(1000.0, 4.0, 16.0),
                           cv::Vec3d(1000.0, 4.0, 24.0)};
        straight.s = {0.0, 8.0, 16.0, 24.0};
        straight.theta = {0.0, 0.0, 0.0, 0.0};
        straight.conditioning = {0.5, 0.5, 0.5, 0.5};
        CurtainCase given;
        given.hPoints = {cv::Vec3d(1000.0, -4.0, 0.0), cv::Vec3d(1000.0, 4.0, 0.0)};
        given.rays = {bending, straight};
        // The crosser through the strip's third row (s about 20) at y 0.
        given.vPoints = {cv::Vec3d(990.0, 0.0, 15.0), cv::Vec3d(1020.0, 0.0, 15.0)};
        if (splitMeasures) {
            // Sibling: the larger turn and smaller conditioning may come
            // from different bounding rays; both still support this strip.
            given.rays[0].conditioning.assign(4, 0.5);
            given.rays[1].conditioning = {0.3, 0.03, 0.3, 0.3};
        }
        if (sharedRay) {
            // Sibling: a hit on a shared ray belongs to both incident
            // strips; rotating must not select only the clean strip.
            auto third = straight;
            third.startSample = 2;
            for (auto& p : third.points) p[1] += 8.0;
            given.hPoints.push_back(third.points.front());
            given.rays.push_back(third);
            for (auto& p : given.vPoints) p[1] = 4.0;
        }
        const auto rotate = [quarterTurns](cv::Vec3d& p) {
            for (int k = 0; k < quarterTurns; ++k) {
                p = cv::Vec3d(-p[1], p[0], p[2]);
            }
        };
        for (auto* points : {&given.hPoints, &given.vPoints}) {
            for (auto& p : *points) {
                rotate(p);
            }
        }
        for (auto& ray : given.rays) {
            for (auto& p : ray.points) {
                rotate(p);
            }
        }
        bool ok = false;
        const BentSolve solved = solveCurtainUnderReversals(given, &ok);
        QVERIFY(ok);
        QCOMPARE(solved.bentIndices.size(), std::size_t{1});
        const Crossing& record = *solved.bentRecord(0);
        QVERIFY(record.withheld);
        QCOMPARE(record.bentReason, static_cast<int>(BentDisposition::CreaseCrossed));
        QVERIFY(record.turnDeg > 60.0);
        QVERIFY(record.minConditioning < 0.05);
    }

    // A kept straight reading of the pair outranks a bent one of the
    // opposite kind on the same translate: the H and V cross directly (a
    // well-conditioned straight reading at translate n); a bent reading of
    // the other kind at n is withheld as straight-disagreeing, one of the
    // same kind stands, and one on another translate is untouched.
    void keptStraightReadingOutranksABentOneOfTheOppositeKind()
    {
        World world;
        const std::size_t h = addH(world, 0.4, 0.9, 30000.0);
        const std::size_t v = addV(world, 0.6, 27000.0, 33000.0);
        const BentFixture fixture(world, 1);
        const BentSolve plain = fixture.solve({});
        long long straightN = 0;
        CrossingKind straightKind = CrossingKind::Inside;
        int straight = 0;
        for (const Crossing& c : plain.result.crossings) {
            if (!c.bent && c.hFiber == h && c.vFiber == v && !c.tangential && !c.touch) {
                straightN = c.n;
                straightKind = c.kind;
                ++straight;
            }
        }
        QVERIFY(straight >= 1);
        const std::size_t seed = hSampleAt(world, h, 0.4, 0.42);
        const std::size_t vMid = world.fibers[v].theta.size() / 2;
        // Side +1 of the H curtain reads Inside (see bentReadingsLiftAndSign).
        const int inside = 1;
        const int wanted = straightKind == CrossingKind::Inside ? -inside : inside;
        BentHit disagreeing = fixture.hit(h, v, false, seed, seed + 2, 0.5, vMid, 0.5, 4.0, wanted);
        BentHit agreeing = fixture.hit(h, v, false, seed, seed + 2, 0.5, vMid, 0.5, 4.0, -wanted);
        const long long n0 = fixture.expectedN(h, v, disagreeing);
        disagreeing.theta += kTwoPi * static_cast<double>(straightN - n0);
        agreeing.theta += kTwoPi * static_cast<double>(straightN - n0);
        QCOMPARE(fixture.expectedN(h, v, disagreeing), straightN);
        {
            const BentSolve solved = fixture.solve({{{h, v}, {disagreeing}}});
            QVERIFY(solved.bentRecord(0) != nullptr);
            QVERIFY(solved.bentRecord(0)->kind != straightKind);
            QVERIFY(solved.bentRecord(0)->withheld);
            QCOMPARE(solved.bentRecord(0)->bentReason, static_cast<int>(BentDisposition::StraightDisagrees));
            QCOMPARE(solved.result.bentCrossingCount, 0);
        }
        {
            const BentSolve solved = fixture.solve({{{h, v}, {agreeing}}});
            QVERIFY(solved.bentRecord(0) != nullptr);
            QCOMPARE(solved.bentRecord(0)->kind, straightKind);
            QVERIFY(!solved.bentRecord(0)->withheld);
            QCOMPARE(solved.result.bentCrossingCount, 1);
        }
        {
            BentHit elsewhere = disagreeing;
            elsewhere.theta += kTwoPi;
            const BentSolve solved = fixture.solve({{{h, v}, {elsewhere}}});
            QVERIFY(solved.bentRecord(0) != nullptr);
            QVERIFY(solved.bentRecord(0)->n != straightN);
            QVERIFY(!solved.bentRecord(0)->withheld);
        }
    }

    // The complete traversal reads Inside/Inside/Outside, whose effective
    // verdict is Outside. Its superseded detections must not veto a bent
    // Outside reading; the group verdict must still veto bent Inside.
    void straightDisagreementUsesTheTraversalVerdict_data()
    {
        QTest::addColumn<bool>("merged");
        QTest::newRow("separate-detections") << false;
        QTest::newRow("mixed-kind-merge") << true;
    }

    void straightDisagreementUsesTheTraversalVerdict()
    {
        QFETCH(bool, merged);
        for (const int sense : {1, -1}) {
            World world;
            const double b = merged ? 10.0 : 100.0;
            const auto h = addHairpinH(world, 30000.0, -3.0, 3.0, b, 0.03, merged ? 0.0 : -kSheetStep);
            const auto v = addRayV(world, merged ? kHairpinR + 90.0 : hairpinRadius(kSqrt3, b) - 300.0,
                                  29000.0, 31000.0, -1);
            if (sense < 0) {
                world = mirrored(world);
            }
            const BentFixture fixture(world, sense);
            const BentSolve plain = fixture.solve({});
            const CrossingGroup& group = singleGroup(plain.result);
            QVERIFY(group.hasVerdict);
            QCOMPARE(group.insideCount, 2);
            QCOMPARE(group.verdict, CrossingKind::Outside);
            for (const int side : {-1, 1}) {
                BentHit hit = fixture.hit(h, v, false, 0, 1, 0.5, 0, 0.5, 4.0, side);
                hit.theta += sense * kTwoPi * static_cast<double>(fixture.expectedN(h, v, hit) - group.n);
                QCOMPARE(fixture.expectedN(h, v, hit), group.n);
                const BentSolve solved = fixture.solve({{{h, v}, {hit}}});
                QVERIFY(singleGroup(solved.result).hasVerdict);
                const Crossing* record = solved.bentRecord(0);
                QVERIFY(record != nullptr);
                QCOMPARE(record->withheld, side == 1);
                QCOMPARE(record->bentReason, static_cast<int>(side == 1 ? BentDisposition::StraightDisagrees
                                                                        : BentDisposition::Usable));
                for (const Crossing& c : solved.result.crossings) {
                    if (!c.bent) {
                        QCOMPARE(c.status, CrossingStatus::InGroup);
                    }
                }
            }
        }
    }

    // A solve-inferred seam changes a raw Outside detection to Inside.
    // The second classification must use that effective sign as well.
    void straightDisagreementUsesTheInferredSeamReading()
    {
        World world;
        const auto h = addH(world, 0.4, 0.9, 30000.0);
        const auto v = addV(world, 0.6, 27000.0, 33000.0, -1000.0);
        const BentFixture fixture(world, 1);
        const SolverParams params;
        PairDetections shard = detectPairCrossings(fixture.canonical[h], fixture.canonical[v], params);
        QVERIFY(!shard.raw.empty());
        const Crossing straight = shard.raw.front();
        QCOMPARE(straight.kind, CrossingKind::Outside);
        const BentClassification input{&fixture.assemblies[h], &fixture.assemblies[v], 1, false};
        for (const int side : {-1, 1}) {
            BentHit hit = fixture.hit(h, v, false, 0, 1, 0.5, 0, 0.5, 4.0, side);
            hit.theta += kTwoPi * static_cast<double>(fixture.expectedN(h, v, hit) - straight.n);
            QCOMPARE(fixture.expectedN(h, v, hit), straight.n);
            shard.bentHits = {hit};
            const PairCrossings classified = classifyPairCrossings(
                shard, fixture.canonical[h], fixture.canonical[v], {}, {straight.detection}, params, &input);
            bool found = false;
            for (const Crossing& c : classified.crossings) {
                if (c.bent) {
                    found = true;
                    QCOMPARE(c.withheld, side == -1);
                    QCOMPARE(c.bentReason, static_cast<int>(side == -1 ? BentDisposition::StraightDisagrees
                                                                     : BentDisposition::Usable));
                } else {
                    QVERIFY(c.kollesisInferred);
                    QCOMPARE(c.kind, CrossingKind::Inside);
                }
            }
            QVERIFY(found);
        }
    }

    // Specialist S3: exact geometric twins from two passages of the H
    // through the V's curtain (the same strip, translate, length,
    // transversality and point; different lifted H positions): the kept
    // twin is the semantically smaller one, so the exported position does
    // not follow the storage order. One record, the same under the four
    // storage orders (bentTerms compares psiH and the height).
    void exactTwinsKeepTheSemanticallySmaller()
    {
        using vc3d::fiber_map::bent::BentRay;
        const auto ray = [](const cv::Vec3d& start, std::size_t sample) {
            BentRay r;
            r.startSample = sample;
            r.side = 1;
            for (int k = 0; k <= 1; ++k) {
                r.points.push_back(start + cv::Vec3d(0.0, 0.0, 8.0 * k));
                r.s.push_back(8.0 * k);
                r.theta.push_back(0.0);
            }
            return r;
        };
        CurtainCase given;
        given.curtainFromV = true;
        given.vPoints = {cv::Vec3d(1000.0, 96.0, 0.0), cv::Vec3d(1000.0, 100.0, 0.0), cv::Vec3d(1000.0, 104.0, 0.0)};
        given.rays = {ray(given.vPoints[0], 0), ray(given.vPoints[1], 1), ray(given.vPoints[2], 2)};
        given.hPoints = {cv::Vec3d(900.0, 100.0, 4.0),  cv::Vec3d(1100.0, 100.0, 4.0), cv::Vec3d(1200.0, 120.0, 4.0),
                         cv::Vec3d(800.0, 120.0, 4.0),  cv::Vec3d(800.0, 100.0, 4.0),  cv::Vec3d(1200.0, 100.0, 4.0)};
        bool ok = false;
        const BentSolve solved = solveCurtainUnderReversals(given, &ok);
        QVERIFY(ok);
        QCOMPARE(solved.bentIndices.size(), std::size_t{1});
        const Crossing& record = *solved.bentRecord(0);
        QVERIFY(record.curtainFromV);
        QVERIFY(!record.withheld);
        QVERIFY(std::abs(record.rayLengthVx - 4.0) < 1e-9);
        QVERIFY(std::abs(record.hitX - 1000.0) < 1e-9);
        QVERIFY(std::abs(record.hitY - 100.0) < 1e-9);
    }

    // Specialist S3: the terminal-event test counts every bent encounter,
    // the ones the reduction drops included. The V's curtain meets the H
    // twice (lengths 4 and 8, one translate: the second is reduced away);
    // the H's own straight crossings of the V are not its last meetings
    // with the V, so none is terminal forward.
    void reducedEncountersStillBlockTerminalEvents()
    {
        using vc3d::fiber_map::bent::BentRay;
        const auto ray = [](const cv::Vec3d& start, std::size_t sample) {
            BentRay r;
            r.startSample = sample;
            r.side = 1;
            for (int k = 0; k <= 2; ++k) {
                r.points.push_back(start + cv::Vec3d(0.0, 0.0, 8.0 * k));
                r.s.push_back(8.0 * k);
                r.theta.push_back(0.0);
            }
            return r;
        };
        CurtainCase given;
        given.curtainFromV = true;
        given.vPoints = {cv::Vec3d(1000.0, 96.0, 0.0), cv::Vec3d(1000.0, 100.0, 0.0), cv::Vec3d(1000.0, 104.0, 0.0),
                         cv::Vec3d(1050.0, 100.0, -20.0), cv::Vec3d(1050.0, 100.0, 20.0)};
        given.rays = {ray(given.vPoints[0], 0), ray(given.vPoints[1], 1), ray(given.vPoints[2], 2)};
        given.hPoints = {cv::Vec3d(900.0, 100.0, 4.0),  cv::Vec3d(1100.0, 100.0, 4.0), cv::Vec3d(1200.0, 120.0, 4.0),
                         cv::Vec3d(900.0, 120.0, 8.0),  cv::Vec3d(900.0, 100.0, 8.0),  cv::Vec3d(1020.0, 100.0, 8.0)};
        bool ok = false;
        const BentSolve solved = solveCurtainUnderReversals(given, &ok);
        QVERIFY(ok);
        QCOMPARE(solved.bentIndices.size(), std::size_t{1});
        QVERIFY(std::abs(solved.bentRecord(0)->rayLengthVx - 4.0) < 1e-9);
        std::size_t straight = 0;
        for (const Crossing& event : solved.result.events) {
            if (event.bent) {
                continue;
            }
            ++straight;
            QVERIFY(!(event.terminalSides & 1) || !event.terminal);
            QVERIFY(!event.terminal);
        }
        QVERIFY(straight >= 1);
    }

    // A warped mesh through the classification: the reviewer's rays A, B
    // (straight, then bending) and C (lifted and bending), and the V
    // through the mesh vertex (1000, 0, 0) of ray B, where four warped
    // quads meet: one reading, 3 vx up, with the transversality of the
    // geometry test, the same under the four storage orders.
    void warpedMeshVertexClassifies()
    {
        using vc3d::fiber_map::bent::BentRay;
        const double a = 3.0 / std::sqrt(2.0);
        const auto ray = [](const std::vector<cv::Vec3d>& points, std::size_t sample) {
            BentRay r;
            r.startSample = sample;
            r.side = 1;
            r.points = points;
            r.s = {0.0, 3.0, 6.0};
            r.theta = {0.0, 0.0, 0.0};
            return r;
        };
        CurtainCase given;
        given.hPoints = {cv::Vec3d(997.0, -3.0, 0.0), cv::Vec3d(1000.0, -3.0, 0.0), cv::Vec3d(1003.0, -3.0, 3.0)};
        given.rays = {ray({cv::Vec3d(997.0, -3.0, 0.0), cv::Vec3d(997.0, 0.0, 0.0), cv::Vec3d(997.0, a, a)}, 0),
                      ray({cv::Vec3d(1000.0, -3.0, 0.0), cv::Vec3d(1000.0, 0.0, 0.0), cv::Vec3d(1000.0, a, a)}, 1),
                      ray({cv::Vec3d(1003.0, -3.0, 3.0), cv::Vec3d(1003.0, 0.0, 3.0), cv::Vec3d(1003.0, a, 3.0 + a)}, 2)};
        given.vPoints = {cv::Vec3d(999.0, -1.0, -0.5), cv::Vec3d(1000.0, 0.0, 0.0), cv::Vec3d(1001.0, 1.0, 3.0)};
        bool ok = false;
        const BentSolve solved = solveCurtainUnderReversals(given, &ok);
        QVERIFY(ok);
        QCOMPARE(solved.bentIndices.size(), std::size_t{1});
        const Crossing& record = *solved.bentRecord(0);
        QVERIFY(!record.withheld);
        QVERIFY(!record.touch && !record.tangential);
        QCOMPARE(record.status, CrossingStatus::Used);
        QVERIFY(std::abs(record.rayLengthVx - 3.0) < 1e-9);
        QVERIFY(std::abs(record.hitX - 1000.0) < 1e-9);
        QVERIFY(std::abs(record.hitY) < 1e-9);
        QVERIFY(std::abs(record.hitZ) < 1e-9);
        QVERIFY(std::abs(record.transversality - 1.0 / std::sqrt(33.0)) < 1e-9);
        // Confidence is the transversality of a trusted owner, attenuated
        // by the untrusted factor for an owner without a traced span.
        QVERIFY(std::abs(record.confidence - 1.0 / std::sqrt(33.0)) < 1e-9);
        given.untrustedH = true;
        const BentSolve untrusted = solveCurtainUnderReversals(given, &ok);
        QVERIFY(ok);
        QCOMPARE(untrusted.bentIndices.size(), std::size_t{1});
        QVERIFY(std::abs(untrusted.bentRecord(0)->confidence -
                         SolverParams{}.untrustedConfidenceFactor / std::sqrt(33.0)) < 1e-9);
        QVERIFY(std::abs(untrusted.bentRecord(0)->transversality - 1.0 / std::sqrt(33.0)) < 1e-9);
    }

    // The off-plane probe case through the classification: the grazing V
    // of the tilted strip is a contact, no constraint, under the four
    // storage orders.
    void offPlaneProbeContactClassifies()
    {
        using vc3d::fiber_map::bent::BentRay;
        const auto ray = [](const cv::Vec3d& p0, const cv::Vec3d& p1, std::size_t sample) {
            BentRay r;
            r.startSample = sample;
            r.side = 1;
            r.points = {p0, p1};
            r.s = {0.0, std::sqrt(14.0)};
            r.theta = {0.0, 0.0};
            return r;
        };
        CurtainCase given;
        given.hPoints = {cv::Vec3d(0.0, 1000.0, 2.0), cv::Vec3d(-3.0, 1001.0, -4.0)};
        given.rays = {ray(given.hPoints[0], cv::Vec3d(-2.0, 999.0, 5.0), 0),
                      ray(given.hPoints[1], cv::Vec3d(0.0, 1000.0, -2.0), 1)};
        given.vPoints = {cv::Vec3d(0.0, 1000.5, -0.2), cv::Vec3d(0.0, 1000.0, 0.0), cv::Vec3d(-0.5, 999.6, 0.0)};
        bool ok = false;
        const BentSolve solved = solveCurtainUnderReversals(given, &ok);
        QVERIFY(ok);
        for (const Crossing& c : solved.result.crossings) {
            if (c.bent) {
                QVERIFY(c.touch || c.tangential);
            }
        }
        for (const Crossing& event : solved.result.events) {
            QVERIFY(!event.bent || event.touch || event.tangential);
        }
    }

    // A crease contact through the classification: the reviewer's warped
    // strip (ray B bending up) and the V grazing the crease from below -
    // a touch: recorded as a contact, no usable reading; the V crossing
    // through the crease: one reading. Both the same under the four
    // storage orders.
    void creaseContactAndCrossingClassify()
    {
        using vc3d::fiber_map::bent::BentRay;
        const auto ray = [](const std::vector<cv::Vec3d>& points, std::size_t sample) {
            BentRay r;
            r.startSample = sample;
            r.side = 1;
            r.points = points;
            r.s = {0.0, 3.0};
            r.theta = {0.0, 0.0};
            return r;
        };
        CurtainCase given;
        given.hPoints = {cv::Vec3d(1000.0, 0.0, 0.0), cv::Vec3d(1003.0, 0.0, 0.0)};
        given.rays = {ray({cv::Vec3d(1000.0, 0.0, 0.0), cv::Vec3d(1000.0, 3.0, 0.0)}, 0),
                      ray({cv::Vec3d(1003.0, 0.0, 0.0), cv::Vec3d(1004.0, 2.0, 2.0)}, 1)};
        given.vPoints = {cv::Vec3d(1001.0, 1.0, -0.25), cv::Vec3d(1001.5, 1.5, 0.0), cv::Vec3d(1002.0, 2.0, 0.25)};
        bool ok = false;
        const BentSolve grazing = solveCurtainUnderReversals(given, &ok);
        QVERIFY(ok);
        QCOMPARE(grazing.bentIndices.size(), std::size_t{1});
        QVERIFY(grazing.bentRecord(0)->touch);
        QCOMPARE(grazing.bentRecord(0)->transversality, 0.0);
        for (const Crossing& event : grazing.result.events) {
            QVERIFY(!event.bent || event.touch || event.tangential);
        }
        given.vPoints = {cv::Vec3d(1001.0, 1.0, -0.25), cv::Vec3d(1001.5, 1.5, 0.0), cv::Vec3d(1002.0, 2.0, 1.0)};
        const BentSolve through = solveCurtainUnderReversals(given, &ok);
        QVERIFY(ok);
        QCOMPARE(through.bentIndices.size(), std::size_t{1});
        const Crossing& record = *through.bentRecord(0);
        QVERIFY(!record.touch && !record.tangential);
        QVERIFY(!record.withheld);
        QCOMPARE(record.status, CrossingStatus::Used);
        QVERIFY(std::abs(record.hitZ) < 1e-9);
        QVERIFY(record.transversality > 0.1);
    }
};

QTEST_APPLESS_MAIN(TestFiberWindingSolver)
#include "test_fiber_winding_solver.moc"
