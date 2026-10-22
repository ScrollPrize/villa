#pragma once
#include <opencv2/core/types.hpp>
#include <optional>

namespace vc::fiber_tracer {
// Call in increasing source arc order. Invalid samples do not reset the gauge.
// Unit normal magnitude is preserved; an exactly orthogonal tie is not flipped.
class FiberNormalOrientationSweep {
public:
    bool orient(cv::Vec3d& normal, bool valid) {
        if (!valid) return false;
        const bool flip = previous_ && previous_->dot(normal) < 0.;
        if (flip) normal = -normal;
        previous_ = normal;
        return flip;
    }
private:
    std::optional<cv::Vec3d> previous_;
};
}
