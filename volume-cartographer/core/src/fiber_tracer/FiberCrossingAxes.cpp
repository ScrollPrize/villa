#include "vc/fiber_tracer/FiberCrossingAxes.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc::fiber_tracer
{

void setFiberCrossingTargets(const FiberStripChain& chain, const std::span<const FiberCrossingAxis> crossings,
                             const FiberCrossingTargetOptions& options, std::vector<FiberStripAxisTarget>& targets,
                             FiberCrossingTargetCounts* counts)
{
    const double sigma = options.sigma;
    if (!(sigma > 0.) || options.weight < 0.) throw std::invalid_argument("crossing_sigma must be positive and crossing_weight >= 0");
    if (!(options.scales[0] > 0. && options.scales[1] > 0. && options.scales[2] > 0.) || options.predictionPower < 0. ||
        options.coherencePower < 0.)
        throw std::invalid_argument("crossing_gap_scales must be 3 positive values (own tangent, side, normal), crossing powers >= 0");
    if (options.anglePower < 0. || options.endTaper < 0.) throw std::invalid_argument("crossing_angle_power and crossing_end_taper must be >= 0");
    if (chain.points.empty() || crossings.empty()) return;
    if (targets.size() != chain.points.size()) throw std::invalid_argument("crossing targets: one target per chain point required");
    const bool weighted = options.falloff != GapFalloff::None;
    const auto gapFactor = [&](double g) { return gapFalloffFactor(options.falloff, g); };
    const auto scaledGap = [&](const cv::Vec3d& d, const cv::Vec3d& t, const cv::Vec3d& n) { return scaledFrameGap(d, t, n, options.scales); };
    std::size_t used = 0, crossedNodes = 0, predictionFrames = 0, crossingFrames = 0;
    std::vector<cv::Matx33d> scatter(chain.points.size(), cv::Matx33d::zeros());
    std::vector<double> sum(chain.points.size(), 0.);
    for (const auto& entry : crossings) {
        const double arc = entry.arc, length = cv::norm(entry.axis);
        const cv::Vec3d e = length > 0. ? entry.axis / length : entry.axis;
        const cv::Vec3d offset = entry.offset;
        cv::Vec3d ownTangent = entry.ownTangent;
        if (cv::norm(ownTangent) > 0.) ownTangent /= cv::norm(ownTangent);
        double w = 1.;
        if (weighted && options.frame == FiberCrossingGapFrame::Isotropic) w *= gapFactor(entry.gap / options.scales[0]);
        if (weighted && options.frame == FiberCrossingGapFrame::Crossing) w *= gapFactor(scaledGap(offset, ownTangent, e));
        if (options.anglePower > 0.) w *= std::pow(std::min(length, 1.), options.anglePower);
        if (options.endTaper > 0.) w *= std::clamp(std::min(arc, chain.arc.back() - arc) / options.endTaper, 0., 1.);
        if (!(length > 0.) || !(w > 0.)) continue;
        ++used;
        const auto lo = std::lower_bound(chain.arc.begin(), chain.arc.end(), arc - 2 * sigma);
        for (auto k = std::size_t(lo - chain.arc.begin()); k < chain.arc.size() && chain.arc[k] <= arc + 2 * sigma; ++k) {
            const auto& node = targets[k];
            double agree = 1.;
            if (node.weight > 0. && cv::norm(node.axis) > 0.) {
                agree = std::abs(e.dot(node.axis)) / cv::norm(node.axis);
                if (agree < options.predictionMin) continue;
            }
            double frameWeight = 1.;
            if (weighted && options.frame == FiberCrossingGapFrame::Prediction) {
                // Joint frame: the prediction axis (no crossing information) as normal, the own
                // tangent made perpendicular to it; the crossing frame where the node has no axis.
                cv::Vec3d n = e, t = ownTangent;
                if (node.weight > 0. && cv::norm(node.axis) > 0.) {
                    const cv::Vec3d m = node.axis / cv::norm(node.axis), u = ownTangent - ownTangent.dot(m) * m;
                    if (cv::norm(u) > 1e-6) { n = m; t = u / cv::norm(u); ++predictionFrames; } else ++crossingFrames;
                } else ++crossingFrames;
                frameWeight = gapFactor(scaledGap(offset, t, n));
            }
            const double d = chain.arc[k] - arc, f = w * frameWeight * std::pow(agree, options.predictionPower) * std::exp(-0.5 * d * d / (sigma * sigma));
            scatter[k] += f * (e * e.t());
            sum[k] += f;
        }
    }
    for (std::size_t k = 0; k < chain.points.size(); ++k) {
        if (!(sum[k] > 0.)) continue;
        cv::Vec3d eigenvalues;
        cv::Matx33d eigenvectors;
        cv::eigen(scatter[k], eigenvalues, eigenvectors);
        auto& target = targets[k];
        target.crossing = cv::Vec3d(eigenvectors(0, 0), eigenvectors(0, 1), eigenvectors(0, 2));
        const double trace = eigenvalues[0] + eigenvalues[1] + eigenvalues[2];
        const double coherence = trace > 0. ? std::max(0., eigenvalues[0] - eigenvalues[1]) / trace : 0.;
        target.crossingWeight = options.weight * std::min(1., sum[k]) * std::pow(coherence, options.coherencePower);
        ++crossedNodes;
    }
    if (counts) {
        counts->used += used;
        counts->crossedNodes += crossedNodes;
        counts->predictionFrames += predictionFrames;
        counts->crossingFrames += crossingFrames;
    }
}

}  // namespace vc::fiber_tracer
