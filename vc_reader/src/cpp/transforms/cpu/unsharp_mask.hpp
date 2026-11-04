#pragma once
#include "../ops/unsharp_mask.hpp"
#include "../shared/voxel_transform.hpp"
namespace vcr::transforms {
inline ChannelPlan channel_parameters(const UnsharpMask& t,const Array& a,Random& r){
    volume_detail::require_image(a);double strength=draw(t.strength,r),sigma=draw(t.sigma,r);require(strength>=0 && strength<=64 && sigma>0 && sigma<=64,"invalid unsharp strength or sigma");ChannelPlan p;p.values={strength,sigma};return p;
}
struct UnsharpMaskKernel {
    double strength;
    template<class T,int D> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T,D>& c) const {return v[i]+T(strength)*(v[i]-c.auxiliary[i]);}
};
template<class T> inline void channel_apply(const UnsharpMask&,std::vector<T>& v,const Grid& g,const ChannelPlan& p){
    auto blurred=v;gaussian_blur(blurred,g,std::vector<double>(g.shape.size(),p.values[1]),4,Boundary::Nearest);
    volume_detail::map_channel<UnsharpMask>(v,g,UnsharpMaskKernel{p.values[0]},nullptr,{},blurred.data());
}
// UnsharpMaskKernel (T = float) on uint8 values v and their blur, held in out.
VCR_TARGET_CLONES inline void narrow_sharpen(const std::uint8_t* __restrict in,float* __restrict out,std::size_t n,float strength) {
    std::uint32_t invalid=0;
    for(std::size_t i=0;i<n;++i){
        const float v=float(in[i]);out[i]=v+strength*(v-out[i]);
        invalid|=std::uint32_t((std::bit_cast<std::uint32_t>(out[i])&0x7f800000u)==0x7f800000u);
    }
    require(!invalid,"transform produced nonfinite values");
}
// channel_apply on a uint8 channel's float32 values, its blur reading the uint8 values directly.
inline void narrow_apply(const UnsharpMask& t,const ChannelPlan& p,const Grid& g,const std::uint8_t* in,float* out) {
    if(channel_wide_parameters(t,p)) {
        // apply_checked_channels' wide path: float64 arithmetic, then float32.
        std::vector<double> values(in,in+g.size);channel_apply(t,values,g,p);
        for(std::size_t i=0;i<g.size;++i){out[i]=float(values[i]);require(std::isfinite(out[i]),"transform output is not finite in the image dtype");}
        return;
    }
    // The blur lands in out; UnsharpMaskKernel's expression then combines it in place.
    gaussian_blur_buffer(in,out,g,std::vector<double>(g.shape.size(),p.values[1]),4,Boundary::Nearest,false);
    narrow_sharpen(in,out,g.size,float(p.values[0]));
}
inline void apply(const UnsharpMask& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::transforms
