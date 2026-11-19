#pragma once
// How much one measured crossing counts as evidence.
//
// A crossing is strong evidence when the two fibers meet squarely (their
// normals are aligned or opposed) and when they are close in winding: a
// crossing several turns apart still says something, but only decides a
// question nothing closer answers. The sign solve has weighted its events
// this way since it was written; the winding graph can be asked for the same
// rule, so that the two stages weigh the same crossing the same way.
#include <cmath>

namespace vc::geometry {

// The damping of a crossing by its measured winding separation: 1 within a
// turn, 1/d^2 beyond it. An unmeasured (NaN) or infinite separation counts as
// within one winding, which is what an unmeasured crossing asserted before
// any separation was measured at all.
inline double fiberCrossingDistanceFactor(const double distanceTurns)
{
    const double turns = std::abs(distanceTurns);
    if (!std::isfinite(turns) || !(turns > 1.0)) return 1.0;
    return 1.0 / std::max(turns * turns, 1e-12);
}

// |dot|^power, the squareness of the crossing. `power` at or below zero is no
// dot weighting at all (every crossing counts the same).
inline double fiberCrossingDotWeight(const double dot, const double power)
{
    if (!(power > 0.0)) return 1.0;
    const double magnitude = std::abs(dot);
    return magnitude * std::pow(std::max(magnitude, 1e-12), power - 1.0);
}

} // namespace vc::geometry
