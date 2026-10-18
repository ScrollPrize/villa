#pragma once
#include "dropout_mask.hpp"
namespace vcr::augment {
struct GridDropout {Shape unit_size;Scalar ratio=.25;std::string axes;volume_detail::EraseValues fill;ChannelOptions options;};
struct GridDropoutKernel {
    volume_detail::PeriodicHoles holes;
    template<int D> VCR_HD bool operator()(const volume_detail::GridND<D>& g,std::size_t i) const {bool drop=true;for(int a=0;a<D;++a)if(holes.enabled[a])drop &= holes.hole(g,i,a);return drop;}
};
inline GridDropoutKernel sample_holes(const GridDropout& t,Random& r,std::size_t dim) {double ratio=draw(t.ratio,r);auto seed=r.next();return {volume_detail::PeriodicHoles(t.unit_size,t.axes,ratio,seed,dim)};}
inline Parameters sample(const GridDropout& t,const Sample& s,Random& r) {
    auto kernel=sample_holes(t,r,sample_shape(s).size());auto p=volume_detail::select_dropout_channels(t.options,s,r);volume_detail::with_grid(p.shape,[&](const auto& g){p.field.resize(g.size);
    for(std::size_t i=0;i<g.size;++i)p.field[i]=kernel(g,i);});return p;
}
inline void apply(const GridDropout& t,Sample& s,const Parameters& p) {volume_detail::erase_mask(s,p,t.fill);}
}
