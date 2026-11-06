#pragma once
#include "../ops/fourier_domain_adaptation.hpp"
#include "../shared/reference_target.hpp"
#include "../shared/fourier.hpp"
namespace vcr::transforms {
inline ChannelPlan channel_parameters(const FDA& t,const Array& a,Random& r) {
    volume_detail::require_image(a);double blend=draw(t.blend,r),fraction=draw(t.fraction,r);probability(blend);probability(fraction);
    require(!t.reference_key.empty() || (t.reference.shape.size()==a.shape.size() && Shape(t.reference.shape.begin()+1,t.reference.shape.end())==a.spatial()),"FDA reference must match spatial shape");ChannelPlan p;p.values={blend,fraction};return p;
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
inline Parameters sample(const FDA& t,const Sample& s,Random& r) {
    auto p=sample_channels(s,t.options,r,[&](const Array& a,Random& rng){return channel_parameters(t,a,rng);});
    if(!t.reference_key.empty())volume_detail::reference_target(t.reference_key,s,p);
    return p;
}
inline void apply(const FDA& t,Sample& s,const Parameters& p) { volume_detail::apply_reference_channels(t,s,p); }
} // namespace vcr::transforms
