#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct UnsharpMask {Scalar strength=Uniform{.1,.5},sigma=Uniform{.5,1.5};ChannelOptions options;};
inline ChannelPlan channel_parameters(const UnsharpMask& t,const Array& a,Random& r){
    volume_detail::require_image(a);double strength=draw(t.strength,r),sigma=draw(t.sigma,r);require(strength>=0 && strength<=64 && sigma>0 && sigma<=64,"invalid unsharp strength or sigma");ChannelPlan p;p.values={strength,sigma};return p;
}
struct UnsharpMaskKernel {
    double strength;
    template<class T,int D> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T,D>& c) const {return v[i]+T(strength)*(v[i]-c.auxiliary[i]);}
};
template<class T> inline void channel_apply(const UnsharpMask&,std::vector<T>& v,const Grid& g,const ChannelPlan& p){
    auto blurred=v;gaussian_blur(blurred,g,std::vector<double>(g.shape.size(),p.values[1]),4,Boundary::Nearest);
    volume_detail::map_channel(v,g,UnsharpMaskKernel{p.values[0]},nullptr,{},blurred.data());
}
inline void apply(const UnsharpMask& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
