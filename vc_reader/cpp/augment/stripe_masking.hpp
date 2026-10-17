#pragma once
#include "dropout_mask.hpp"
namespace vcr::augment {
struct StripeMasking {Shape unit_size{8,8,8};Scalar ratio=.25;std::string axes="zyx";volume_detail::EraseValues fill;ChannelOptions options;};
struct StripeMaskingKernel {
    volume_detail::PeriodicHoles holes;
    VCR_HD bool operator()(const volume_detail::Grid3& g,std::size_t i) const {bool drop=false;for(int a=0;a<3;++a)if(holes.enabled[a])drop |= holes.hole(g,i,a);return drop;}
};
inline StripeMaskingKernel sample_holes(const StripeMasking& t,Random& r) {double ratio=draw(t.ratio,r);auto seed=r.next();return {volume_detail::PeriodicHoles(t.unit_size,t.axes,ratio,seed)};}
inline Parameters sample(const StripeMasking& t,const Sample& s,Random& r) {
    auto kernel=sample_holes(t,r);auto p=volume_detail::select_dropout_channels(t.options,s,r);volume_detail::Grid3 g(p.shape);p.field.resize(g.size);
    for(std::size_t i=0;i<g.size;++i)p.field[i]=kernel(g,i);return p;
}
inline void apply(const StripeMasking& t,Sample& s,const Parameters& p) {volume_detail::erase_mask(s,p,t.fill);}
}
