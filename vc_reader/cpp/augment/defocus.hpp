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
    const double radius=p.values[0];if(radius==0)return;
    const int r=int(ceil(radius));
    std::vector<std::array<int,3>> taps;
    for(int z=-r;z<=r;++z)for(int y=-r;y<=r;++y)for(int x=-r;x<=r;++x)
        if(double(z*z+y*y+x*x)<=radius*radius)taps.push_back({z,y,x});
    std::vector<double> output(v.size());std::vector<std::size_t> rows(taps.size());
    const auto nz=g.shape[0],ny=g.shape[1],nx=g.shape[2];
    auto clamp_index=[](long long x,std::size_t n){return std::size_t(std::clamp(x,0LL,(long long)n-1));};
    for(std::size_t z=0;z<nz;++z)for(std::size_t y=0;y<ny;++y){
        const auto base=(z*ny+y)*nx;
        for(std::size_t k=0;k<taps.size();++k)
            rows[k]=(clamp_index((long long)z+taps[k][0],nz)*ny+clamp_index((long long)y+taps[k][1],ny))*nx;
        std::size_t x=0;
        while(x<nx){
            // Interior runs have contiguous loads and no per-voxel bounds or
            // coordinate divisions. Each lane retains the original tap order.
            constexpr std::size_t lanes=16;
            if(x>=std::size_t(r) && x+lanes+std::size_t(r)<=nx){
                std::array<double,lanes> sums{};
                for(std::size_t k=0;k<taps.size();++k){
                    const auto* source=v.data()+rows[k]+std::size_t((long long)x+taps[k][2]);
                    for(std::size_t lane=0;lane<lanes;++lane)sums[lane]+=source[lane];
                }
                for(std::size_t lane=0;lane<lanes;++lane)output[base+x+lane]=sums[lane]/int(taps.size());
                x+=lanes;
            }else{
                double sum=0;
                for(std::size_t k=0;k<taps.size();++k)sum+=v[rows[k]+clamp_index((long long)x+taps[k][2],nx)];
                output[base+x++]=sum/int(taps.size());
            }
        }
    }
    v.swap(output);
}
inline void apply(const Defocus& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
