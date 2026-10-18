#pragma once
#include "voxel_transform.hpp"
#include "fourier.hpp"
namespace vcr::augment {
struct FDA {volume_detail::ReferenceVolume reference;Scalar blend=Uniform{.5,1},fraction=.1;ChannelOptions options;};
inline ChannelPlan channel_parameters(const FDA& t,const Array& a,Random& r) {
    volume_detail::require_image(a);double blend=draw(t.blend,r),fraction=draw(t.fraction,r);probability(blend);probability(fraction);
    require(t.reference.shape.size()==a.shape.size() && Shape(t.reference.shape.begin()+1,t.reference.shape.end())==a.spatial(),"FDA reference must match spatial shape");ChannelPlan p;p.values={blend,fraction};return p;
}
template<class T> VCR_HD inline volume_detail::ComplexT<T> adapt_amplitude(volume_detail::ComplexT<T> input,volume_detail::ComplexT<T> reference,T blend) {
    using namespace volume_detail;T amplitude=hypot(input.re,input.im),target=hypot(reference.re,reference.im);
    auto unit=amplitude?input.scale(T(1)/amplitude):ComplexT<T>{1,0};return unit.scale((T(1)-blend)*amplitude+blend*target);
}
template<class T> inline void channel_apply(const FDA& t,std::vector<T>& v,const Grid& grid,const ChannelPlan& p){
    using namespace volume_detail;if(p.values[0]==0)return;with_grid(grid.shape,[&](const auto& g){auto ref=reference_channel<T>(t.reference,p.channel);std::vector<ComplexT<T>> source(g.size),reference(g.size);
    for(std::size_t i=0;i<g.size;++i){source[i].re=v[i];reference[i].re=ref[i];}fft_volume(source,g,false);fft_volume(reference,g,false);
    for(std::size_t i=0;i<g.size;++i)if(low_frequency(g,i,p.values[1]))source[i]=adapt_amplitude(source[i],reference[i],T(p.values[0]));
    fft_volume(source,g,true);for(std::size_t i=0;i<g.size;++i)v[i]=source[i].re;
    });
}
inline void apply(const FDA& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
