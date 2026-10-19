#pragma once
#include "../ops/foreground_dropout.hpp"
#include "../shared/dropout_mask.hpp"
namespace vcr::transforms {
struct ForegroundDropoutKernel {
    std::size_t size[3],count;double threshold;std::uint64_t seed;
    template<class T,int D> VCR_HD void operator()(const T* mask,const volume_detail::GridND<D>& g,float* field)const{
        using namespace volume_detail;for(std::size_t i=0;i<g.size;++i)field[i]=0;
        for(std::size_t h=0;h<count;++h){auto i=foreground_index(mask,g.size,threshold,mix(seed+h));if(i==g.size)continue;std::size_t lo[3],hi[3];
            for(int a=0;a<D;++a){auto x=g.coord(i,a);lo[a]=x>size[a]/2?x-size[a]/2:0;hi[a]=lo[a]+size[a]<g.n[a]?lo[a]+size[a]:g.n[a];}
            for(std::size_t z=D==3?lo[0]:0;z<(D==3?hi[0]:1);++z)for(auto y=lo[D-2];y<hi[D-2];++y)for(auto x=lo[D-1];x<hi[D-1];++x){auto j=(D==3?z*g.stride[0]:0)+y*g.stride[D-2]+x;if(double(mask[j])>threshold)field[j]=1;}}
    }
};
inline ForegroundDropoutKernel sample_holes(const ForegroundDropout& t,Random& r,std::size_t dim){
    const auto size=volume_detail::spatial_sizes(t.size,dim,8);
    require(std::all_of(size.begin(),size.end(),[](auto x){return x>0 && x<=65536;}),"sizes must be positive and <= 65536");
    auto count=draw_integer(t.count,r);return {{size[0],size[1],dim==3?size[2]:0},std::size_t(count),t.threshold,r.next()};
}
inline Parameters sample(const ForegroundDropout& t,const Sample& s,Random& r){
    auto kernel=sample_holes(t,r,sample_shape(s).size());const auto& f=volume_detail::mask_field(s,t.mask,t.mask_channel);auto p=volume_detail::select_dropout_channels(t.options,s,r);volume_detail::with_grid(p.shape,[&](const auto& g){p.field.resize(g.size);
    f.array.typed([&](const auto* v){kernel(v+t.mask_channel*g.size,g,p.field.data());});});return p;
}
inline void apply(const ForegroundDropout& t,Sample& s,const Parameters& p){volume_detail::erase_mask(s,p,t.fill);}
}
