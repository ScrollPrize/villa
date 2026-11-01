#pragma once
#include "../ops/equalize.hpp"
#include "../shared/voxel_transform.hpp"
namespace vcr::transforms {
inline ChannelPlan channel_parameters(const Equalize& t,const Array& a,Random&) {
    volume_detail::require_image(a);require(t.bins>=2 && t.bins<=4096,"bins must be [2,4096]");return {};
}
struct EqualizeKernel {
    int bins;
    template<class T,int D> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T,D>& c) const {
        using namespace volume_detail;double span=c.input.high-c.input.low;if(span==0)return v[i];double first=0;
        for(int b=0;b<bins;++b)if(c.table[b]>0){first=c.table[b];break;}
        double rank=c.table[bin(v[i],c.input.low,c.input.high,bins)];
        return first<1?c.input.low+span*clamp((rank-first)/(1-first),0,1):v[i];
    }
};
template<class T> inline void channel_apply(const Equalize& t,std::vector<T>& v,const Grid& g,const ChannelPlan&) {
    auto table=volume_detail::with_grid(g.shape,[&](const auto& grid){return volume_detail::channel_histograms(v,grid,volume_detail::HistogramGrid(grid,g.shape,t.bins,0));});
    volume_detail::map_channel<Equalize>(v,g,EqualizeKernel{t.bins},table.data());
}
inline void apply(const Equalize& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::transforms
