#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct PlasmaBrightnessContrast { Scalar brightness=Uniform{-.2,.2}, contrast=Uniform{0,.3}, scale=.025, persistence=.5;int octaves=5; ChannelOptions options; };
inline ChannelPlan channel_parameters(const PlasmaBrightnessContrast& t,const Array& a,Random& r) {
    volume_detail::require_image(a);ChannelPlan p;
    double b=draw(t.brightness,r),contrast=draw(t.contrast,r),scale=draw(t.scale,r),persistence=draw(t.persistence,r);require(fabs(contrast)<=1 && scale>0 && scale<=1 && persistence>0 && persistence<=1 && t.octaves>=1 && t.octaves<=16,"invalid fractal parameters");p.values={b,contrast,scale,persistence};
    p.seed=r.next();return p;
}
struct PlasmaBrightnessContrastKernel {
    double brightness,contrast,scale,persistence;int octaves;std::uint64_t seed;
    template<class T> VCR_HD T apply_field(T value,T f,const volume_detail::Stats& input) const {
        T mean=T(input.mean);return (value-mean)*(T(1)+T(contrast)*f)+mean+T(brightness)*T(input.high-input.low)*f;
    }
    template<class T,int D> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T,D>& c) const {
        using namespace volume_detail;
        return apply_field(v[i],fractal(c.grid,i,seed,T(scale),T(persistence),octaves),c.input);
    }
};
inline PlasmaBrightnessContrastKernel make_kernel(const PlasmaBrightnessContrast& t,const ChannelPlan& p) { return {p.values[0],p.values[1],p.values[2],p.values[3],t.octaves,p.seed}; }
template<class T> inline void channel_apply(const PlasmaBrightnessContrast& t,std::vector<T>& v,const Grid& g,const ChannelPlan& p) {
    const auto kernel=make_kernel(t,p);
    const auto input=volume_detail::stats(v.data(),v.size());
    const auto field=volume_detail::fractal_field(g,kernel.seed,kernel.scale,kernel.persistence,kernel.octaves);
    for(std::size_t i=0;i<v.size();++i)v[i]=kernel.apply_field(v[i],T(field[i]),input);
}
inline void apply(const PlasmaBrightnessContrast& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
