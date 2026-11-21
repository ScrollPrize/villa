#pragma once
#include <opencv2/core/types.hpp>
#include <optional>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

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

// IDs refer to shared immutable (source fiber, exact arc, coordinate) records.
// Each normal is already oriented in THIS piece's local frame.
struct FiberNormalPieceFrame {
    std::size_t beginBoundary=0,endBoundary=0;
    cv::Vec3d beginNormal{0.,0.,0.},endNormal{0.,0.,0.};
    bool beginValid=false,endValid=false;
};

// One source fiber's contiguous pieces, in arc order. Zero means unresolved,
// never a guessed sign. The first piece defines the original global gauge.
inline std::vector<signed char> reconcileFiberNormalPieceFrames(
    std::span<const FiberNormalPieceFrame> pieces)
{
    std::vector<signed char> signs(pieces.size(),0);
    if(pieces.empty())return signs;
    signs[0]=1;
    for(std::size_t i=1;i<pieces.size();++i) {
        const auto& a=pieces[i-1];const auto& b=pieces[i];
        if(a.endBoundary!=b.beginBoundary)
            throw std::invalid_argument("Normal pieces do not share the identical boundary sample");
        if(a.endValid!=b.beginValid)
            throw std::invalid_argument("Identical boundary samples have different validity");
        if(!a.endValid)continue;
        // An identical decoded sample can differ ONLY by the local sign.
        // Do not normalize or tolerate a nearby sample into agreement.
        const bool same=a.endNormal==b.beginNormal;
        const bool opposite=a.endNormal==-b.beginNormal;
        const double magnitude=a.endNormal.dot(a.endNormal);
        if((!same&&!opposite) || !(magnitude>0.) || !std::isfinite(magnitude))
            throw std::invalid_argument("Boundary normals are not the same sample up to sign");
        signs[i]=static_cast<signed char>(signs[i-1]*(same?1:-1));
    }
    return signs;
}
}
