#pragma once

#include <cstdint>
#include <vector>

// How well each point of a fiber has its normal sign tied to the rest of the
// crossing graph: the first-order two-terminal unreliability.
//
// Graph: one node per crossing end (fiber, arc); chain edges join consecutive
// crossing ends along a fiber (length = their arc gap), crossing edges join
// the two ends of a crossing. A sign flip along a chain edge (rate lambda per
// voxel of arc) or a wrong crossing (probability q / w for a crossing of
// sign-solve weight w, fiberSignEventWeight) goes unnoticed for a point
// iff it lies on every path from the point to the core, the largest
// 2-edge-connected component: any other flip closes a loop with a
// disagreement. To first order in lambda and q
//     P(missed flip at the point) = lambda * missedLength + q * missedCrossingCost,
// with missedLength the chain length (voxels) and missedCrossingCost the sum of
// 1 / w over the crossings that are bridges between the point and the core
// (a full-weight crossing counts one); a fiber's tail
// beyond its last crossing counts as chain. Parallel paths only enter at
// second order (products of their lengths), which this reports as zero.
namespace vc::fiber_tracer
{

// A crossing that holds: callers drop crossings that disagree with the signs,
// since those are already detected flips and support nothing.
struct FiberSupportCrossing {
    std::int64_t a = 0;
    std::int64_t b = 0;
    double arcA = 0.0;
    double arcB = 0.0;
    double weight = 1.0;  // the sign solve's weight of this crossing, > 0
};

// One sample along a fiber. Both values are +infinity where the fiber is not
// linked to the core at all.
struct FiberSupportPoint {
    double arc = 0.0;
    double missedLength = 0.0;
    double missedCrossingCost = 0.0;
};

struct FiberSupportResult {
    // Per crossing end: end 2k is crossing k's a side, 2k+1 its b side.
    std::vector<double> endMissedLength;
    std::vector<double> endMissedCrossingCost;
    // Per fiber: samples every `spacing` voxels from arc 0 to the fiber length.
    std::vector<std::vector<FiberSupportPoint>> points;
    std::int64_t chainEdges = 0;
    std::int64_t bridgeChainEdges = 0;
    std::int64_t bridgeCrossings = 0;
    std::int64_t coreEnds = 0;
    std::int64_t unlinkedEnds = 0;
    std::int64_t components = 0;  // 2-edge-connected components
};

// `fiberLengths` holds one arc length per fiber; crossing indices address it.
// Throws std::invalid_argument on out-of-range indices, a non-positive weight or spacing.
[[nodiscard]] FiberSupportResult fiberSignSupport(const std::vector<double>& fiberLengths,
                                                  const std::vector<FiberSupportCrossing>& crossings,
                                                  double spacing);

// One bin of a calibration check: `count` items, `events` of them flips, over
// `exposure` (voxels of chain, or the sum of 1 / weight of crossings).
struct FiberCalibrationBin {
    double lo = 0.0;
    double hi = 0.0;
    std::int64_t count = 0;
    std::int64_t events = 0;
    double exposure = 0.0;
};

// lambda and q from the core, where every flip is detectable: crossings whose
// two ends both have zero missed length and cost in the support graph of the
// agreeing crossings. Along each fiber, a run of two or more consecutive
// disagreeing core crossings is a flipped stretch (one chain flip at each end
// where the fiber goes on); a disagreeing crossing that is a run of one on both
// its fibers is a wrong crossing. lambda = chain flips / core chain length,
// q = wrong crossings / sum of 1 / weight over core crossings. The bins check
// the model: flips per voxel by chain gap length, wrong crossings per unit of
// 1 / weight by weight. The bounds are the highest bin rates among bins with
// at least `minBinFlips` flips: a score built on them overrates no bin's
// reliability, it may only underrate some.
struct FiberSignCalibration {
    double lambda = 0.0;  // per voxel of arc, over all core chain
    double q = 0.0;       // per unit of 1 / weight, over all core crossings
    double lambdaBound = 0.0;
    double qBound = 0.0;
    std::int64_t coreCrossings = 0;
    std::int64_t coreDisagreeing = 0;
    std::int64_t chainSwitches = 0;
    std::int64_t wrongCrossings = 0;
    double coreChainLength = 0.0;
    double coreInverseWeight = 0.0;
    std::vector<FiberCalibrationBin> gapBins;
    std::vector<FiberCalibrationBin> weightBins;
};

// Local weighted edge connectivity: the smallest total weight of links whose
// loss cuts a point, with whatever lies around it, off from the graph beyond
// its neighbourhood. A cut of total weight C lets that piece flip as a whole
// without any crossing disagreeing.
//
// Nodes are the pieces of each fiber between its chain bridges (chains that
// are the only path between their two ends, fiberSignSupport's bridge chain
// edges); inside a piece the chain is unbreakable. Links: crossings with
// their sign-solve weight, and chain bridges of length L with weight
// chainWeightLength / L, the crossing weight with the same chance of an
// unnoticed flip (q / w = lambda * L gives chainWeightLength = q / lambda).
// The neighbourhood holds the nodes within `maxHops` links, at most
// `maxFibers` of them in breadth-first order; everything outside is the far
// graph. The value is the max flow to the far graph, capped at `cap`. Zero
// for unlinked fibers and for components that fit inside the neighbourhood.
// Tails beyond a fiber's outermost crossing take their end piece's value:
// nothing can detect a flip there. Flips of a stretch bounded by two chain
// flips are not covered.
struct FiberCutOptions {
    double cap = 4.0;
    std::int64_t maxHops = 2;
    std::int64_t maxFibers = 200;
    double chainWeightLength = 50.0;  // voxels, q / lambda
    unsigned threads = 0;              // 0: all hardware threads
};

struct FiberCutPoint {
    double arc = 0.0;
    double cut = 0.0;
};

struct FiberCutResult {
    // Per fiber: samples every `spacing` voxels from arc 0 to the fiber length.
    std::vector<std::vector<FiberCutPoint>> points;
    std::int64_t segments = 0;
};

[[nodiscard]] FiberCutResult fiberSignCut(const std::vector<double>& fiberLengths,
                                          const std::vector<FiberSupportCrossing>& crossings,
                                          double spacing,
                                          const FiberCutOptions& options = {});

[[nodiscard]] FiberSignCalibration fiberSignCalibration(const std::vector<double>& fiberLengths,
                                                        const std::vector<FiberSupportCrossing>& agreeing,
                                                        const std::vector<FiberSupportCrossing>& disagreeing,
                                                        std::int64_t minBinFlips = 30);

}  // namespace vc::fiber_tracer
