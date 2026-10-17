#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct PlasmaShadow { Scalar strength=Uniform{.1,.5}, scale=.025, persistence=.5;int octaves=5; ChannelOptions options; };
inline ChannelPlan channel_parameters(const PlasmaShadow& t,const Array& a,Random& r) {
    volume_detail::require_volume(a);ChannelPlan p;
    double strength=draw(t.strength,r),scale=draw(t.scale,r),persistence=draw(t.persistence,r);probability(strength);require(scale>0 && scale<=1 && persistence>0 && persistence<=1 && t.octaves>=1 && t.octaves<=16,"invalid fractal parameters");p.values={strength,scale,persistence};
    p.seed=r.next();return p;
}
struct PlasmaShadowKernel {
    double strength,scale,persistence;int octaves;std::uint64_t seed;
    VCR_HD double operator()(const double* v,std::size_t i,const volume_detail::ChannelContext& c) const {
        using namespace volume_detail;
        return v[i]*(1-strength*(fractal(c.grid,i,seed,scale,persistence,octaves)+1)*.5);
    }
};
inline PlasmaShadowKernel make_kernel(const PlasmaShadow& t,const ChannelPlan& p) { return {p.values[0],p.values[1],p.values[2],t.octaves,p.seed}; }
inline void channel_apply(const PlasmaShadow& t,std::vector<double>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel(v,g,make_kernel(t,p));
}
inline void apply(const PlasmaShadow& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
