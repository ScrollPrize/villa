#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct PlasmaBrightnessContrast { Scalar brightness=Uniform{-.2,.2}, contrast=Uniform{0,.3}, scale=.025, persistence=.5;int octaves=5; ChannelOptions options; };
inline ChannelPlan channel_parameters(const PlasmaBrightnessContrast& t,const Array& a,Random& r) {
    volume_detail::require_volume(a);ChannelPlan p;
    double b=draw(t.brightness,r),contrast=draw(t.contrast,r),scale=draw(t.scale,r),persistence=draw(t.persistence,r);require(fabs(contrast)<=1 && scale>0 && scale<=1 && persistence>0 && persistence<=1 && t.octaves>=1 && t.octaves<=16,"invalid fractal parameters");p.values={b,contrast,scale,persistence};
    p.seed=r.next();return p;
}
struct PlasmaBrightnessContrastKernel {
    double brightness,contrast,scale,persistence;int octaves;std::uint64_t seed;
    VCR_HD double operator()(const double* v,std::size_t i,const volume_detail::ChannelContext& c) const {
        using namespace volume_detail;
        double f=fractal(c.grid,i,seed,scale,persistence,octaves);return (v[i]-c.input.mean)*(1+contrast*f)+c.input.mean+brightness*(c.input.high-c.input.low)*f;
    }
};
inline PlasmaBrightnessContrastKernel make_kernel(const PlasmaBrightnessContrast& t,const ChannelPlan& p) { return {p.values[0],p.values[1],p.values[2],p.values[3],t.octaves,p.seed}; }
inline void channel_apply(const PlasmaBrightnessContrast& t,std::vector<double>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel(v,g,make_kernel(t,p));
}
inline void apply(const PlasmaBrightnessContrast& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
