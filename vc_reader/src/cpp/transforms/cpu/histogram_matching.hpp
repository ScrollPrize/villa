#pragma once
#include "../ops/histogram_matching.hpp"
#include "../shared/voxel_transform.hpp"
namespace vcr::transforms {
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
// HistogramMatchingKernel's bin-only part: the reference level of each input bin's rank.
inline std::vector<double> matched_levels(const std::vector<double>& table,int bins,const volume_detail::Stats& reference) {
    std::vector<double> mapped(static_cast<std::size_t>(bins));
    for(int b=0;b<bins;++b) {
        const double rank=table[std::size_t(b)];int j=0;while(j+1<bins && table[std::size_t(bins+j)]<rank)++j;
        mapped[std::size_t(b)]=reference.low+(reference.high-reference.low)*double(j)/(bins-1);
    }
    return mapped;
}
// The reference channel's bounds; its cumulative histogram goes to table[0, bins).
inline volume_detail::Stats reference_histogram(const HistogramMatching& t,std::size_t channel,double* table) {
    using namespace volume_detail;Stats summary{};
    visit_reference_channel(t.reference,channel,[&](const auto& ref){
        summary=stats<false>(ref.data(),ref.size());
        std::vector<std::uint16_t> reference(ref.size());bin_indices(ref.data(),ref.size(),summary.low,summary.high,t.bins,reference.data());
        count_bins(reference.data(),reference.size(),t.bins,table);
    });cdf(table,t.bins);
    return summary;
}
// HistogramMatchingKernel with its bin-only part (rank search and mapped value) evaluated once per bin.
template<class T> inline void channel_apply(const HistogramMatching& t,std::vector<T>& v,const Grid&,const ChannelPlan& p) {
    using namespace volume_detail;Stats summary{};const auto input=stats<false>(v.data(),v.size());std::vector<double> table(2*t.bins);
    std::vector<std::uint16_t> index(v.size());bin_indices(v.data(),v.size(),input.low,input.high,t.bins,index.data());
    count_bins(index.data(),index.size(),t.bins,table.data());cdf(table.data(),t.bins);
    summary=reference_histogram(t,p.channel,table.data()+t.bins);
    const double blend=p.values[0];if(blend==0)return;
    const auto mapped=matched_levels(table,t.bins,summary);
    for(std::size_t i=0;i<v.size();++i) {
        const T value=T((1-blend)*v[i]+blend*mapped[index[i]]);
        require(std::isfinite(value),"transform produced nonfinite values");v[i]=value;
    }
}
// channel_apply on a uint8 channel's float32 values, evaluated once per value.
inline void narrow_apply(const HistogramMatching& t,const ChannelPlan& p,const Grid& g,const std::uint8_t* in,float* out) {
    using namespace volume_detail;
    const NarrowValues values(in,g.size);
    const auto bin_of=values.bins(t.bins);
    std::vector<double> table(2*std::size_t(t.bins),0.);
    values.histogram(bin_of,table.data());cdf(table.data(),t.bins);
    const auto summary=reference_histogram(t,p.channel,table.data()+t.bins);
    const double blend=p.values[0];if(blend==0){widen_values(in,out,g.size);return;}
    const auto mapped=matched_levels(table,t.bins,summary);
    std::array<float,256> map{};
    for(int x=values.low;x<=values.high;++x) if(values.counts[std::size_t(x)]) {
        const float value=float((1-blend)*double(float(x))+blend*mapped[bin_of[std::size_t(x)]]);
        require(std::isfinite(value),"transform produced nonfinite values");map[std::size_t(x)]=value;
    }
    lookup(in,out,g.size,map.data());
}
inline void apply(const HistogramMatching& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::transforms
