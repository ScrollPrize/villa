#include "vc/fiber_tracer/FiberPredictionNormals.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc::fiber_tracer
{

FiberPredictionNormalScatter& FiberPredictionNormalScatter::operator+=(const FiberPredictionNormalScatter& other)
{
    scatter += other.scatter;
    weight += other.weight;
    count += other.count;
    return *this;
}

FiberPredictionNormalScatter FiberPredictionNormalScatter::scaled(double factor) const
{
    FiberPredictionNormalScatter out;
    out.scatter = scatter * factor;
    out.weight = weight * factor;
    out.count = count;
    return out;
}

FiberPredictionNormalScatter fiberPredictionNormalScatter(const cv::Vec3d& tangent,
                                                          std::span<const FiberPredictionNormalSample> samples,
                                                          const FiberPredictionNormalOptions& options,
                                                          FiberPredictionNormalFrame frame, const cv::Vec3d& axis,
                                                          double agreementPower)
{
    if (!(options.height > 0.) || !(options.width > 0.) || !(options.presenceFull > options.presenceFloor) ||
        !(options.minAlignment >= 0.) || !(options.minAlignment < 1.) || !(agreementPower >= 0.) ||
        !(options.presenceBase >= 0.) || !(options.presenceBase <= 1.))
        throw std::invalid_argument("fiber prediction normal: invalid options");
    const bool hasAxis = axis.dot(axis) > 0.;
    if (frame == FiberPredictionNormalFrame::Given && !hasAxis)
        return {};  // no earlier estimate here: nothing to measure along
    FiberPredictionNormalScatter out;
    for (const auto& s : samples) {
        if (!(s.normal.dot(s.normal) > 0.) || !(s.direction.dot(s.direction) > 0.)) continue;
        const double presence = options.presenceBase + (1. - options.presenceBase) *
            std::clamp((s.presence - options.presenceFloor) / (options.presenceFull - options.presenceFloor), 0., 1.);
        const double alignment =
            std::clamp((std::abs(s.direction.dot(tangent)) - options.minAlignment) / (1. - options.minAlignment), 0., 1.);
        if (!(presence > 0.) || !(alignment > 0.)) continue;
        double r2 = 0.;
        if (frame == FiberPredictionNormalFrame::Isotropic) {
            r2 = s.offset.dot(s.offset) / (options.width * options.width);
        } else {
            const cv::Vec3d& n = frame == FiberPredictionNormalFrame::Given ? axis : s.normal;
            const double along = s.offset.dot(n);
            const double across = cv::norm(s.offset - along * n);
            r2 = (along / options.height) * (along / options.height) +
                 (across / options.width) * (across / options.width);
        }
        if (r2 > 1.) continue;
        double w = presence * alignment * std::exp(-2. * r2);
        if (agreementPower > 0. && hasAxis) w *= std::pow(std::abs(s.normal.dot(axis)), agreementPower);
        if (!(w > 0.)) continue;
        out.scatter += w * (s.normal * s.normal.t());
        out.weight += w;
        out.count += 1.;
    }
    return out;
}

FiberPredictionNormalEstimate fiberPredictionNormal(const FiberPredictionNormalScatter& scatter)
{
    FiberPredictionNormalEstimate out;
    out.weight = scatter.weight;
    out.count = scatter.count;
    if (!(scatter.weight > 0.)) return out;
    cv::Mat values, vectors;
    cv::eigen(cv::Mat(scatter.scatter), values, vectors);
    out.axis = cv::Vec3d(vectors.at<double>(0, 0), vectors.at<double>(0, 1), vectors.at<double>(0, 2));
    const double trace = values.at<double>(0) + values.at<double>(1) + values.at<double>(2);
    out.coherence = trace > 0. ? (values.at<double>(0) - values.at<double>(1)) / trace : 0.;
    return out;
}

cv::Matx33d fiberDirectionTurn(const cv::Vec3d& from, const cv::Vec3d& to)
{
    const cv::Vec3d v = from.cross(to);
    const double c = from.dot(to);
    if (!(c > -1. + 1e-9)) return cv::Matx33d::eye();  // opposite directions: no unique turn
    const cv::Matx33d k(0., -v[2], v[1], v[2], 0., -v[0], -v[1], v[0], 0.);
    return cv::Matx33d::eye() + k + k * k * (1. / (1. + c));
}

FiberPredictionNormalScatter turned(const FiberPredictionNormalScatter& scatter, const cv::Matx33d& turn)
{
    auto out = scatter;
    out.scatter = turn * scatter.scatter * turn.t();
    return out;
}

std::vector<FiberPredictionNormalEstimate> trackFiberPredictionNormals(
    std::span<const double> arcs, std::span<const cv::Vec3d> tangents,
    std::span<const std::vector<FiberPredictionNormalSample>> samples, const FiberPredictionNormalOptions& options,
    const FiberPredictionNormalTrackOptions& track, std::size_t* seedOut)
{
    const std::size_t count = arcs.size();
    if (tangents.size() != count || samples.size() != count)
        throw std::invalid_argument("fiber prediction normal track: one tangent and sample list per point");
    if (!(track.length > 0.) || !(track.agreementPower >= 0.))
        throw std::invalid_argument("fiber prediction normal track: invalid options");
    std::vector<FiberPredictionNormalEstimate> out(count);
    if (count == 0) return out;
    // The seed: most support from samples that carry fiber presence.
    auto seeded = options;
    seeded.presenceBase = 0.;
    std::size_t seed = count;
    double best = 0.;
    for (std::size_t i = 0; i < count; ++i) {
        const double weight = fiberPredictionNormalScatter(tangents[i], samples[i], seeded).weight;
        if (weight > best) {
            best = weight;
            seed = i;
        }
    }
    if (seedOut) *seedOut = seed;
    if (seed == count) return out;  // no presence anywhere along the fiber: no start
    const auto start = fiberPredictionNormalScatter(tangents[seed], samples[seed], options);
    out[seed] = fiberPredictionNormal(start);
    for (const int direction : {1, -1}) {
        auto running = start;
        for (std::size_t previous = seed;;) {
            if (direction > 0 ? previous + 1 >= count : previous == 0) break;
            const std::size_t i = direction > 0 ? previous + 1 : previous - 1;
            const cv::Matx33d r = fiberDirectionTurn(tangents[previous], tangents[i]);
            const double decay = std::exp(-std::abs(arcs[i] - arcs[previous]) / track.length);
            running.scatter = r * running.scatter * r.t() * decay;
            running.weight *= decay;
            const cv::Vec3d carried = fiberPredictionNormal(running).axis;
            running += fiberPredictionNormalScatter(tangents[i], samples[i], options, FiberPredictionNormalFrame::Sample,
                                                    carried, track.agreementPower);
            out[i] = fiberPredictionNormal(running);
            previous = i;
        }
    }
    return out;
}

}  // namespace vc::fiber_tracer
