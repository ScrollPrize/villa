#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct Defocus { Scalar radius=Uniform{1,3}; ChannelOptions options; };
inline ChannelPlan channel_parameters(const Defocus& t,const Array& a,Random& r) {
    volume_detail::require_volume(a);ChannelPlan p;
    double radius=draw(t.radius,r);require(radius>=0 && radius<=32,"defocus radius must be [0,32]");p.values={radius};
    p.seed=r.next();return p;
}
struct DefocusKernel {
    double radius;
    VCR_HD double operator()(const double* v,std::size_t i,const volume_detail::ChannelContext& c) const {
        using namespace volume_detail;
        if(radius==0)return v[i];int r=int(ceil(radius));double sum=0,count=0;for(int z=-r;z<=r;++z)for(int y=-r;y<=r;++y)for(int x=-r;x<=r;++x)if(double(z*z+y*y+x*x)<=radius*radius){double q[3]{double(c.grid.coord(i,0))+z,double(c.grid.coord(i,1))+y,double(c.grid.coord(i,2))+x};sum+=interpolate(v,c.grid,q);++count;}return sum/count;
    }
};
inline DefocusKernel make_kernel(const Defocus& t,const ChannelPlan& p) { return {p.values[0]}; }
inline void channel_apply(const Defocus& t,std::vector<double>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel(v,g,make_kernel(t,p));
}
inline void apply(const Defocus& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
