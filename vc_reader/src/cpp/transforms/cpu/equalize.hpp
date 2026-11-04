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
// EqualizeKernel's output depends only on the voxel's bin: evaluate it once per bin.
template<class T> inline void channel_apply(const Equalize& t,std::vector<T>& v,const Grid& g,const ChannelPlan&) {
    using namespace volume_detail;
    const auto input=stats<false>(v.data(),v.size());
    const double span=input.high-input.low;if(span==0)return;
    std::vector<std::uint16_t> index(v.size());
    bin_indices(v.data(),v.size(),input.low,input.high,t.bins,index.data());
    std::vector<double> table(std::size_t(t.bins),0.);
    count_bins(index.data(),index.size(),t.bins,table.data());
    std::vector<bool> used(table.size());for(std::size_t b=0;b<table.size();++b)used[b]=table[b]>0;
    cdf(table.data(),t.bins);
    double first=0;for(int b=0;b<t.bins;++b)if(table[b]>0){first=table[b];break;}
    if(!(first<1))return;
    std::vector<T> lut(table.size());
    for(std::size_t b=0;b<table.size();++b) {
        lut[b]=T(input.low+span*clamp((table[b]-first)/(1-first),0,1));
        require(!used[b] || std::isfinite(lut[b]),"transform produced nonfinite values");
    }
    for(std::size_t i=0;i<v.size();++i)v[i]=lut[index[i]];
}
// channel_apply on a uint8 channel's float32 values: its bins and table per value, not per voxel.
inline void narrow_apply(const Equalize& t,const ChannelPlan&,const Grid& g,const std::uint8_t* in,float* out) {
    using namespace volume_detail;
    const NarrowValues values(in,g.size);
    const auto input=values.bounds();
    const double span=input.high-input.low;if(span==0){widen_values(in,out,g.size);return;}
    const auto bin_of=values.bins(t.bins);
    std::vector<double> table(std::size_t(t.bins),0.);
    values.histogram(bin_of,table.data());
    std::vector<bool> used(table.size());for(std::size_t b=0;b<table.size();++b)used[b]=table[b]>0;
    cdf(table.data(),t.bins);
    double first=0;for(int b=0;b<t.bins;++b)if(table[b]>0){first=table[b];break;}
    if(!(first<1)){widen_values(in,out,g.size);return;}
    std::array<float,256> map{};
    for(int x=values.low;x<=values.high;++x) {
        const auto b=bin_of[std::size_t(x)];
        map[std::size_t(x)]=float(input.low+span*clamp((table[b]-first)/(1-first),0,1));
        require(!used[b] || std::isfinite(map[std::size_t(x)]),"transform produced nonfinite values");
    }
    lookup(in,out,g.size,map.data());
}
inline void apply(const Equalize& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::transforms
