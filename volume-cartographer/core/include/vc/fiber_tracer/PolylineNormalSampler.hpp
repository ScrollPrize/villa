#pragma once

#include "vc/fiber_tracer/PolylineGeometry.hpp"
#include "vc/lasagna/ChannelSampler.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc::fiber_tracer {

// Immutable annotation axes, extended off the reference by nearest polyline
// projection. This is reference acceptance geometry, never a prediction field.
class PolylineNormalSampler final : public vc::lasagna::NormalSampler {
public:
    PolylineNormalSampler(const std::vector<cv::Vec3d>& pointsBase,
                          std::vector<cv::Vec3d> axes, double workingToBase)
        : geometry_(makePolylineArcGeometry(pointsBase)), axes_(std::move(axes)),
          scale_(workingToBase)
    {
        if (pointsBase.size() < 2 || axes_.size() != pointsBase.size() ||
            !std::isfinite(scale_) || scale_ <= 0 || geometry_.length() <= 0)
            throw std::invalid_argument("invalid baked reference normal geometry");
        for (auto& axis : axes_) {
            const double length = cv::norm(axis);
            if (!std::isfinite(length) || length <= 1e-12)
                throw std::invalid_argument("invalid baked reference normal axis");
            axis /= length;
        }
    }

    bool supportsConcurrentSampling() const noexcept override { return true; }

    vc::lasagna::NormalSample sampleNormal(const cv::Vec3d& point) const override
    {
        const auto match = projectPointToPolylineArc(geometry_, point * scale_, 0, geometry_.length());
        const auto i = match.segmentIndex;
        const double span = geometry_.vertexArcs[i+1] - geometry_.vertexArcs[i];
        const double t = span > 0 ? std::clamp((match.arc - geometry_.vertexArcs[i])/span, 0., 1.) : 0.;
        const auto& a = axes_[i];
        const auto& b = axes_[i+1];
        const cv::Matx33d tensor = a * a.t() * (1-t) + b * b.t() * t;
        return {vc::lasagna::principalCompactTensorAxis(tensor, t <= .5 ? a : b), true, {}};
    }
private:
    PolylineArcGeometry geometry_;
    std::vector<cv::Vec3d> axes_;
    double scale_;
};
} // namespace vc::fiber_tracer
