#pragma once
#include "resize_index.hpp"

namespace vcr::transforms {
template<class Real, class Index = std::size_t> struct ResizePosition {
    Index lo, hi;
    Real fraction;
};

// Axis-sized tables compose nearest reduction with linear reconstruction.
// Addresses refer directly to the source, so no reduced image is needed.
// Coordinate construction stays double to resolve large axes and exact ties;
// stored weights follow the image dtype.
template<class Real, class Index = std::size_t>
std::vector<ResizePosition<Real,Index>> resize_positions(
        std::size_t input, std::size_t output, std::size_t stride,
        bool nearest, bool origin_aligned, std::size_t reduced = 0) {
    const auto extent = reduced ? reduced : input;
    const double ratio = double(extent)/output;
    const double offset = origin_aligned ? 0 : .5*ratio-.5;
    std::vector<ResizePosition<Real,Index>> positions(output);
    for (std::size_t x=0; x<output; ++x) {
        if (nearest) {
            const auto at=resize_nearest_index(x,input,output,origin_aligned);
            positions[x]={Index(at*stride),0,0};
        } else {
            const double q=std::clamp(ratio*double(x)+offset,0.,double(extent-1));
            auto lo=std::size_t(std::floor(q)), hi=std::min(lo+1,extent-1);
            const Real fraction=Real(q-lo);
            if (reduced) {
                lo=resize_nearest_index(lo,input,reduced,false);
                hi=resize_nearest_index(hi,input,reduced,false);
            }
            positions[x]={Index(lo*stride),Index(hi*stride),fraction};
        }
    }
    return positions;
}

template<class Real> VCR_HD Real resize_blend(Real a, Real b, Real t) {
    if (t==0) return a;
    if (t==1) return b;
    return (Real(1)-t)*a+t*b;
}
// Factor the tensor product into three (2D) or seven (3D) interpolations.
// Avoid b-a, which can overflow with finite endpoints of opposite signs.
template<int D, class Acc, class T, class Real, class Index>
VCR_HD Acc resize_sample(const T* src, const ResizePosition<Real,Index>* q) {
    const auto x=q[D-1], y=q[D-2];
    const auto plane = [&](Index z) {
        return resize_blend(
            resize_blend(Acc(src[z+y.lo+x.lo]),Acc(src[z+y.lo+x.hi]),Acc(x.fraction)),
            resize_blend(Acc(src[z+y.hi+x.lo]),Acc(src[z+y.hi+x.hi]),Acc(x.fraction)),Acc(y.fraction));
    };
    if constexpr (D==3) return resize_blend(plane(q[0].lo),plane(q[0].hi),Acc(q[0].fraction));
    else return plane(0);
}
template<int D, class T, class Real, class Index>
VCR_HD Real resize_value(const T* src, const ResizePosition<Real,Index>* q) {
    Real value=resize_sample<D,Real>(src,q);
    if constexpr (std::is_same_v<Real,float>)
        if (!std::isfinite(value)) value=float(resize_sample<D,double>(src,q));
    return value;
}
} // namespace vcr::transforms
