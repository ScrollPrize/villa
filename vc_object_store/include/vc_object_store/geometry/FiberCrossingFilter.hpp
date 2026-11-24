#pragma once
// What the normal surfaces of its two fibers found about a measured crossing,
// and the one rule every consumer applies to it.
//
// A crossing is measured at the closest point of two fibers. The extraction
// also follows the partner through each fiber's normal surface (the fiber
// swept along its normals) and records what it finds there with the
// constraint: `findings` bits and, where the encounter holds a strong crossing
// that states the other relation of the two normals than the measured one,
// the |dot| of the nearest such crossing (`alternative`, 0 for none). The
// findings are evidence, not a verdict: each stage decides whether to filter,
// and all of them decide with fiberCrossingUse below.
#include <cmath>
#include <cstdint>

namespace vc::geometry {

namespace crossing_finding {
// A surface crossing lies at the measured one (without it the surface says nothing).
inline constexpr std::uint8_t Matched = 1;
// The encounter holds more than one crossing.
inline constexpr std::uint8_t Multiple = 2;
// Every passage has its return: the partner dips through the surface and comes back.
inline constexpr std::uint8_t NothingLeft = 4;
// Strong crossings of both relations (normals agreeing, normals opposed) in the encounter.
inline constexpr std::uint8_t Mixed = 8;
// The same, among the crossings near the fiber only.
inline constexpr std::uint8_t MixedNear = 16;
// More than one crossing is left after cancelling.
inline constexpr std::uint8_t SeveralLeft = 32;
}

// The findings that drop a crossing while no whole-fiber signs exist: with
// nothing left there is no crossing, and a mixed encounter fits either sign.
inline constexpr std::uint8_t kCrossingFirstPassDrop = crossing_finding::NothingLeft | crossing_finding::Mixed;

struct FiberCrossingUse {
    // The crossing takes part.
    bool used = true;
    // The dot of the two normals to use: the measured one, or the closest
    // agreeing crossing's when the measured one disagrees under the signs.
    double dot = 0.0;
    // `dot` comes from another crossing of the encounter than the measured
    // one. That crossing has no measured winding separation, so a consumer
    // that needs the separation leaves the crossing out.
    bool replaced = false;
};

// `signProduct` is the product of the two fibers' whole-fiber signs (+1 or
// -1), or 0 while there are none.
//  - Without signs the crossing is dropped when it has a finding of
//    `unsignedDrop`.
//  - With signs only a nothing-left crossing is dropped; one that disagrees
//    (dot * signProduct < 0) takes its agreeing alternative when it has one.
inline FiberCrossingUse fiberCrossingUse(const std::uint8_t findings, const double alternative, const double dot,
                                         const int signProduct, const std::uint8_t unsignedDrop = kCrossingFirstPassDrop)
{
    FiberCrossingUse use;
    use.dot = dot;
    if (signProduct == 0) {
        use.used = (findings & unsignedDrop) == 0;
        return use;
    }
    if (findings & crossing_finding::NothingLeft) {
        use.used = false;
        return use;
    }
    if (dot * double(signProduct) < 0.0 && alternative > 0.0) {
        use.dot = dot < 0.0 ? alternative : -alternative;
        use.replaced = true;
    }
    return use;
}

} // namespace vc::geometry
