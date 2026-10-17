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
    template<class T> VCR_HD T apply_field(T value,T field) const {
        return value*(T(1)-T(strength)*(field+T(1))*T(.5));
    }
    template<class T> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T>& c) const {
        using namespace volume_detail;
        return apply_field(v[i],fractal(c.grid,i,seed,T(scale),T(persistence),octaves));
    }
};
inline PlasmaShadowKernel make_kernel(const PlasmaShadow& t,const ChannelPlan& p) { return {p.values[0],p.values[1],p.values[2],t.octaves,p.seed}; }
inline void channel_apply(const PlasmaShadow& t,std::vector<double>& v,const Grid& g,const ChannelPlan& p) {
    const auto kernel=make_kernel(t,p);
    const auto field=volume_detail::fractal_field(g,kernel.seed,kernel.scale,kernel.persistence,kernel.octaves);
    for(std::size_t i=0;i<v.size();++i)v[i]=kernel.apply_field(v[i],field[i]);
}
inline void apply(const PlasmaShadow& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
