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
    template<class T> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T>& c) const {
        using namespace volume_detail;
        if(radius==0)return v[i];int r=int(ceil(radius));T sum=0;int count=0;for(int z=-r;z<=r;++z)for(int y=-r;y<=r;++y)for(int x=-r;x<=r;++x)if(double(z*z+y*y+x*x)<=radius*radius){std::size_t index=0;const int offset[3]{z,y,x};for(int a=0;a<3;++a){auto q=(long long)c.grid.coord(i,a)+offset[a];q=q<0?0:q>=(long long)c.grid.n[a]?(long long)c.grid.n[a]-1:q;index+=std::size_t(q)*c.grid.stride[a];}sum+=v[index];++count;}return sum/count;
    }
};
inline DefocusKernel make_kernel(const Defocus& t,const ChannelPlan& p) { return {p.values[0]}; }
inline void channel_apply(const Defocus& t,std::vector<double>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel(v,g,make_kernel(t,p));
}
inline void apply(const Defocus& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
