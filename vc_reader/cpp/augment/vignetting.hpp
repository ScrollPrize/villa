#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct Vignetting { Scalar strength=Uniform{.1,.5}, radius=.8;std::vector<double> center; ChannelOptions options; };
inline ChannelPlan channel_parameters(const Vignetting& t,const Array& a,Random& r) {
    volume_detail::require_volume(a);ChannelPlan p;
    double strength=draw(t.strength,r),radius=draw(t.radius,r);probability(strength);require(radius>0,"radius must be positive");auto center=volume_detail::relative_center(t.center);p.values={strength,radius};p.values.insert(p.values.end(),center.begin(),center.end());
    p.seed=r.next();return p;
}
struct VignettingKernel {
    double strength,radius,center[3];
    VCR_HD double operator()(const double* v,std::size_t i,const volume_detail::ChannelContext& c) const {
        using namespace volume_detail;
        double distance=0;for(int a=0;a<3;++a){double q=(c.grid.n[a]>1?double(c.grid.coord(i,a))/double(c.grid.n[a]-1):.5)-center[a];distance+=q*q;}return v[i]*(1-strength*clamp(distance/(radius*radius),0,1));
    }
};
inline VignettingKernel make_kernel(const Vignetting& t,const ChannelPlan& p) { return {p.values[0],p.values[1],{p.values[2],p.values[3],p.values[4]}}; }
inline void channel_apply(const Vignetting& t,std::vector<double>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel(v,g,make_kernel(t,p));
}
inline void apply(const Vignetting& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
