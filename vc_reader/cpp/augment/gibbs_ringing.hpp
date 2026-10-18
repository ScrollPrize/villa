#pragma once
#include "voxel_transform.hpp"
#include "fourier.hpp"
namespace vcr::augment {
struct GibbsRinging {Scalar cutoff=Uniform{.5,.9};ChannelOptions options;};
inline ChannelPlan channel_parameters(const GibbsRinging& t,const Array& a,Random& r){
    volume_detail::require_image(a);double cutoff=draw(t.cutoff,r);require(cutoff>0 && cutoff<=1,"cutoff must be (0,1]");ChannelPlan p;p.values={cutoff};return p;
}
template<class T> inline void channel_apply(const GibbsRinging&,std::vector<T>& v,const Grid& grid,const ChannelPlan& p){
    using namespace volume_detail;if(p.values[0]==1)return;with_grid(grid.shape,[&](const auto& g){std::vector<ComplexT<T>> spectrum(g.size);
    for(std::size_t i=0;i<g.size;++i)spectrum[i].re=v[i];fft_volume(spectrum,g,false);
    for(std::size_t i=0;i<g.size;++i)if(!low_frequency(g,i,p.values[0]))spectrum[i]={};
    fft_volume(spectrum,g,true);for(std::size_t i=0;i<g.size;++i)v[i]=spectrum[i].re;
    });
}
inline void apply(const GibbsRinging& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
