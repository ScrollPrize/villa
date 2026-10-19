#pragma once
#include "../ops/stripe_masking.hpp"
#include "../shared/dropout_mask.hpp"
namespace vcr::transforms {
struct StripeMaskingKernel {
    volume_detail::PeriodicHoles holes;
    template<int D> VCR_HD bool operator()(const volume_detail::GridND<D>& g,std::size_t i) const {bool drop=false;for(int a=0;a<D;++a)if(holes.enabled[a])drop |= holes.hole(g,i,a);return drop;}
};
inline StripeMaskingKernel sample_holes(const StripeMasking& t,Random& r,std::size_t dim) {double ratio=draw(t.ratio,r);auto seed=r.next();return {volume_detail::PeriodicHoles(t.unit_size,t.axes,ratio,seed,dim)};}
inline Parameters sample(const StripeMasking& t,const Sample& s,Random& r) {
    auto kernel=sample_holes(t,r,sample_shape(s).size());auto p=volume_detail::select_dropout_channels(t.options,s,r);volume_detail::with_grid(p.shape,[&](const auto& g){p.field.resize(g.size);
    for(std::size_t i=0;i<g.size;++i)p.field[i]=kernel(g,i);});return p;
}
inline void apply(const StripeMasking& t,Sample& s,const Parameters& p) {volume_detail::erase_mask(s,p,t.fill);}
}
