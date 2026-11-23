#pragma once

#include "vc/fiber_tracer/FiberGapFalloff.hpp"
#include "vc/fiber_tracer/FiberStripNormals.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <vector>

// The crossing term of a strip solve: where another fiber crosses, the normal
// of both lies along own tangent x partner tangent. Both tangents are those of
// the traced fibers themselves, not of a sampled direction field.
namespace vc::fiber_tracer
{

// One crossing of a fiber by another.
struct FiberCrossingAxis {
    double arc = 0.;                // own arc, base voxels
    cv::Vec3d axis{0., 0., 0.};     // own tangent x partner tangent (sign-free), unit or of the crossing angle's sine
    double gap = 0.;                // |offset|
    cv::Vec3d offset{0., 0., 0.};   // partner point - own point
    cv::Vec3d ownTangent{0., 0., 0.};
};

// How crossings become the sign-free crossing axis of a chain's nodes. Each
// crossing spreads to the nodes within 2 sigma of its arc with a Gaussian
// falloff, scaled by a falloff of its 3D offset to the crossing fiber. The
// offset is scaled per axis by `scales` (own tangent, side, normal) in
// `frame`: isotropic (|offset| / first scale), crossing (own tangent and
// crossing axis) or prediction (per node: the prediction axis and the own
// tangent perpendicular to it; the crossing frame where the node has no
// prediction axis), and by |axis . prediction axis|^predictionPower (entries
// below predictionMin are dropped; a node without a prediction axis agrees
// fully). anglePower p scales a crossing by sin(crossing angle)^p, endTaper
// (voxels) by its arc distance to the nearer end of the own fiber over the
// taper, up to one. A node takes the principal axis of the weighted outer
// products and the weight `weight` * min(1, summed weight) *
// coherence^coherencePower, where coherence = (l1 - l2) / (l1 + l2 + l3) of
// the scatter eigenvalues.
enum class FiberCrossingGapFrame { Isotropic, Crossing, Prediction };
struct FiberCrossingTargetOptions {
    double weight = 100., sigma = 64.;
    GapFalloff falloff = GapFalloff::None;
    FiberCrossingGapFrame frame = FiberCrossingGapFrame::Isotropic;
    std::array<double, 3> scales{64., 64., 64.};
    double predictionPower = 0., predictionMin = 0., coherencePower = 0.;
    double anglePower = 0., endTaper = 0.;
};

struct FiberCrossingTargetCounts {
    std::size_t used = 0, crossedNodes = 0, predictionFrames = 0, crossingFrames = 0;
};

// Sets `crossing` and `crossingWeight` of the chain's targets (one per chain
// point) from the fiber's crossings; the nodes' prediction axes are read from
// the same targets. Counts are added to `counts`.
void setFiberCrossingTargets(const FiberStripChain& chain, std::span<const FiberCrossingAxis> crossings,
                             const FiberCrossingTargetOptions& options, std::vector<FiberStripAxisTarget>& targets,
                             FiberCrossingTargetCounts* counts = nullptr);

}  // namespace vc::fiber_tracer
