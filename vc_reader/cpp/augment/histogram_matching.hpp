#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct HistogramMatching {volume_detail::ReferenceVolume reference;Scalar blend=Uniform{.5,1};int bins=256;ChannelOptions options;};
inline ChannelPlan channel_parameters(const HistogramMatching& t,const Array& a,Random& r) {
    volume_detail::require_image(a);double blend=draw(t.blend,r);probability(blend);require(t.bins>=2 && t.bins<=4096,"bins must be [2,4096]");ChannelPlan p;p.values={blend};return p;
}
struct HistogramMatchingKernel {
    double blend;int bins;
    template<class T,int D> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T,D>& c) const {
        using namespace volume_detail;if(blend==0)return v[i];double rank=c.table[bin(v[i],c.input.low,c.input.high,bins)];const double* ref=c.table+bins;
        int j=0;while(j+1<bins && ref[j]<rank)++j;
        double mapped=c.reference.low+(c.reference.high-c.reference.low)*double(j)/(bins-1);return (1-blend)*v[i]+blend*mapped;
    }
};
template<class T> inline void channel_apply(const HistogramMatching& t,std::vector<T>& v,const Grid& g,const ChannelPlan& p) {
    using namespace volume_detail;Stats summary{};auto input=stats(v.data(),v.size());std::vector<double> table(2*t.bins);
    histogram(v.data(),v.size(),table.data(),t.bins,input.low,input.high);cdf(table.data(),t.bins);
    visit_reference_channel(t.reference,p.channel,[&](const auto& ref){
        summary=stats(ref.data(),ref.size());histogram(ref.data(),ref.size(),table.data()+t.bins,t.bins,summary.low,summary.high);
    });cdf(table.data()+t.bins,t.bins);
    map_channel(v,g,HistogramMatchingKernel{p.values[0],t.bins},table.data(),summary);
}
inline void apply(const HistogramMatching& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
