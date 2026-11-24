#pragma once

#include "vc/fiber_tracer/FiberNeighbourNormals.hpp"

#include <cstdint>
#include <span>
#include <vector>

// Crossings of fibers with the normal ribbons of other fibers. A fiber's
// ribbon is the surface its normals sweep: point(arc) + h * normal(arc) for h
// in [-halfHeight, halfHeight]. Where the closest point of two fibers states
// one encounter, the ribbon states every passage of the other fiber over or
// under this one, each with the side it passes on (the sign of h) and the way
// it passes (direction). A fiber that wanders across and back - an S curve -
// leaves several crossings whose directions alternate; a passage and its
// return on one side cancel, and what remains is the crossing the pair really
// has.
namespace vc::fiber_tracer
{

struct FiberRibbonCrossing {
    std::int64_t ribbon = 0;  // the fiber whose ribbon is crossed
    std::int64_t fiber = 0;   // the fiber crossing it
    double ribbonArc = 0.0, fiberArc = 0.0;
    // Signed distance of the crossing from the ribbon's fiber, along its normal.
    double height = 0.0;
    // +1 when the crossing fiber, in its own arc order, passes through the
    // ribbon along its facing (tangent x normal where the ribbon is regular;
    // the facing turns over where the ribbon folds back on itself, so such a
    // fold is crossed back and forth); -1 when against it.
    int direction = 0;
    // Dot of the two unit tangents and of the two (signed) normals there.
    double tangentCosine = 0.0, normalDot = 0.0;
    cv::Vec3d point{0.0, 0.0, 0.0};
};

struct FiberRibbonOptions {
    double halfHeight = 128.0;
    // Crossings whose tangents are closer to parallel than this are left out:
    // a fiber running along a ribbon grazes it and states no passage.
    double maxTangentCosine = 0.7;
    // Segments longer than this (a jump in the samples) are neither ribbon
    // nor crossing fiber.
    double maxSegmentLength = 64.0;
    unsigned threads = 0;  // 0: all hardware threads
};

// Every crossing of every fiber with every other fiber's ribbon, ordered by
// ribbon fiber and ribbon arc. Each ribbon span between two samples is the two
// triangles of its four corners. Samples with a zero normal carry no ribbon.
[[nodiscard]] std::vector<FiberRibbonCrossing> fiberRibbonCrossings(
    std::span<const std::vector<FiberStripSample>> fibers, const FiberRibbonOptions& options = {});

// One encounter of a fiber with another fiber's ribbon: the crossings of the
// pair that belong to one turn of the scroll. `first`/`count` index the
// crossing list as fiberRibbonEncounters ordered it.
struct FiberRibbonEncounter {
    std::int64_t ribbon = 0, fiber = 0;
    std::int64_t first = 0, count = 0;
    // The crossing nearest the ribbon's fiber (smallest |height|).
    std::int64_t closest = 0;
    // Crossings left after cancelling, and the one left when it is exactly one (else -1).
    std::int64_t left = 0, remaining = -1;
};

// Orders `crossings` by (ribbon, fiber, fiber arc) and groups them into
// encounters: consecutive crossings of a pair stay together while both arc
// gaps are below `maxArcGap` (crossings further apart are on other turns and
// say nothing about each other). Within an encounter a passage and its return
// cancel, on each side of the ribbon's fiber separately (never across it), and
// farthest from the fiber first: on a side with more passages one way than the
// other, the crossings left are the nearest ones of the direction in excess.
// Where a ribbon crumples far from its fiber - at the focus of a bend - the
// crumple goes and the crossing near the fiber stays.
[[nodiscard]] std::vector<FiberRibbonEncounter> fiberRibbonEncounters(std::vector<FiberRibbonCrossing>& crossings,
                                                                      double maxArcGap = 3000.0);

// What the ribbon of one fiber says about a perpendicular crossing event the
// extraction found on it. `ribbon` is the fiber whose normal surface is asked,
// `fiber` the partner; the arcs are the event's own.
struct FiberCrossingEventView {
    std::int64_t ribbon = 0, fiber = 0;
    double ribbonArc = 0.0, fiberArc = 0.0;
};

// The surface's findings on one view. They are sign-free: none needs the whole-fiber signs.
struct FiberCrossingVerdict {
    // A crossing of the pair lies near the event; without one the surface says nothing.
    bool matched = false;
    // The encounter holds more than one crossing.
    bool multiple = false;
    // Every passage has its return: the partner dips through the surface and
    // comes back, so the pair has no crossing here.
    bool nothingLeft = false;
    // More than one crossing is left after cancelling.
    bool severalLeft = false;
    // The encounter holds strong crossings that agree and strong crossings
    // that oppose the two normals: whichever sign the fibers get, one of them
    // fits. `mixedNear` asks the same of the crossings up to nearHeight only.
    bool mixed = false, mixedNear = false;
    // The nearest strong crossing of the encounter that states the other
    // relation of the two normals than the event's own crossing does: its
    // |normal dot| and its distance from the fiber. Zero when there is none.
    // Once whole-fiber signs say the event disagrees, this is the closest
    // crossing that agrees.
    double alternativeDot = 0.0, alternativeHeight = 0.0;
};

struct FiberCrossingFilterOptions {
    // An event belongs to the crossing nearest in both arcs, within this.
    double matchTolerance = 64.0;
    // |normal dot| from which a crossing counts as strong.
    double minimumDot = 0.316;
    double nearHeight = 128.0;
};

// One verdict per view. `crossings` and `encounters` are as
// fiberRibbonEncounters left them.
[[nodiscard]] std::vector<FiberCrossingVerdict> fiberCrossingVerdicts(
    std::span<const FiberRibbonCrossing> crossings, std::span<const FiberRibbonEncounter> encounters,
    std::span<const FiberCrossingEventView> views, const FiberCrossingFilterOptions& options = {});

}  // namespace vc::fiber_tracer
