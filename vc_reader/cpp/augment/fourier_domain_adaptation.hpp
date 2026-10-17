#pragma once
#include "voxel_transform.hpp"
#include "fourier.hpp"
namespace vcr::augment {
struct FDA {volume_detail::ReferenceVolume reference;Scalar blend=Uniform{.5,1},fraction=.1;ChannelOptions options;};
inline ChannelPlan channel_parameters(const FDA& t,const Array& a,Random& r) {
    volume_detail::require_volume(a);double blend=draw(t.blend,r),fraction=draw(t.fraction,r);probability(blend);probability(fraction);
    require(t.reference.shape.size()==4 && Shape(t.reference.shape.begin()+1,t.reference.shape.end())==a.spatial(),"FDA reference must match spatial shape");ChannelPlan p;p.values={blend,fraction};return p;
}
VCR_HD inline volume_detail::Complex adapt_amplitude(volume_detail::Complex input,volume_detail::Complex reference,double blend) {
    using namespace volume_detail;double amplitude=hypot(input.re,input.im),target=hypot(reference.re,reference.im);
    auto unit=amplitude?input.scale(1/amplitude):Complex{1,0};return unit.scale((1-blend)*amplitude+blend*target);
}
inline void channel_apply(const FDA& t,std::vector<double>& v,const Grid& grid,const ChannelPlan& p){
    using namespace volume_detail;if(p.values[0]==0)return;Grid3 g(grid.shape);auto ref=reference_channel(t.reference,p.channel);std::vector<Complex> source(g.size),reference(g.size);
    for(std::size_t i=0;i<g.size;++i){source[i].re=v[i];reference[i].re=ref[i];}fft_volume(source,g,false);fft_volume(reference,g,false);
    for(std::size_t i=0;i<g.size;++i)if(low_frequency(g,i,p.values[1]))source[i]=adapt_amplitude(source[i],reference[i],p.values[0]);
    fft_volume(source,g,true);for(std::size_t i=0;i<g.size;++i)v[i]=source[i].re;
}
inline void apply(const FDA& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
