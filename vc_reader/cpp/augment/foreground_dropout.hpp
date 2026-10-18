#pragma once
#include "dropout_mask.hpp"
namespace vcr::augment {
struct ForegroundDropout {std::string mask="mask";std::size_t mask_channel=0;double threshold=0;Shape size{8,8,8};Scalar count=Uniform{1,3};volume_detail::EraseValues fill;ChannelOptions options;};
struct ForegroundDropoutKernel {
    std::size_t size[3],count;double threshold;std::uint64_t seed;
    template<class T> VCR_HD void operator()(const T* mask,const volume_detail::Grid3& g,float* field)const{
        using namespace volume_detail;for(std::size_t i=0;i<g.size;++i)field[i]=0;
        for(std::size_t h=0;h<count;++h){auto i=foreground_index(mask,g.size,threshold,mix(seed+h));if(i==g.size)continue;std::size_t lo[3],hi[3];
            for(int a=0;a<3;++a){auto x=g.coord(i,a);lo[a]=x>size[a]/2?x-size[a]/2:0;hi[a]=lo[a]+size[a]<g.n[a]?lo[a]+size[a]:g.n[a];}
            for(auto z=lo[0];z<hi[0];++z)for(auto y=lo[1];y<hi[1];++y)for(auto x=lo[2];x<hi[2];++x){auto j=z*g.stride[0]+y*g.stride[1]+x;if(double(mask[j])>threshold)field[j]=1;}}
    }
};
inline ForegroundDropoutKernel sample_holes(const ForegroundDropout& t,Random& r){
    require(t.size.size()==3 && std::all_of(t.size.begin(),t.size.end(),[](auto x){return x>0 && x<=65536;}),"size needs three positive sizes <= 65536");
    auto count=draw_integer(t.count,r);return {{t.size[0],t.size[1],t.size[2]},std::size_t(count),t.threshold,r.next()};
}
inline Parameters sample(const ForegroundDropout& t,const Sample& s,Random& r){
    auto kernel=sample_holes(t,r);const auto& f=volume_detail::mask_field(s,t.mask,t.mask_channel);auto p=volume_detail::select_dropout_channels(t.options,s,r);volume_detail::Grid3 g(p.shape);p.field.resize(g.size);
    f.array.typed([&](const auto* v){kernel(v+t.mask_channel*g.size,g,p.field.data());});return p;
}
inline void apply(const ForegroundDropout& t,Sample& s,const Parameters& p){volume_detail::erase_mask(s,p,t.fill);}
}
