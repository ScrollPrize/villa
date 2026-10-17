#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct BiasField { Scalar std=Uniform{0,.3}, scale=.025; ChannelOptions options; };
inline ChannelPlan channel_parameters(const BiasField& t,const Array& a,Random& r) {
    volume_detail::require_volume(a);ChannelPlan p;
    double std=draw(t.std,r),scale=draw(t.scale,r);require(std>=0 && std<=3 && scale>0 && scale<=1,"bias std must be [0,3], scale (0,1]");p.values={std,scale};
    p.seed=r.next();return p;
}
struct BiasFieldKernel {
    double std,scale;std::uint64_t seed;
    VCR_HD double operator()(const double* v,std::size_t i,const volume_detail::ChannelContext& c) const {
        using namespace volume_detail;
        double q[3];for(int a=0;a<3;++a)q[a]=double(c.grid.coord(i,a))*scale;return v[i]*exp(std*lattice(q,seed,true,false));
    }
};
inline BiasFieldKernel make_kernel(const BiasField& t,const ChannelPlan& p) { return {p.values[0],p.values[1],p.seed}; }
inline void channel_apply(const BiasField& t,std::vector<double>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel(v,g,make_kernel(t,p));
}
inline void apply(const BiasField& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
