#pragma once
#include "../ops/pixel_distribution_adaptation.hpp"
#include "../shared/reference_target.hpp"
namespace vcr::transforms {
inline ChannelPlan channel_parameters(const PixelDistributionAdaptation& t,const Array& a,Random& r) {
    volume_detail::require_image(a);double blend=draw(t.blend,r);probability(blend);ChannelPlan p;p.values={blend};return p;
}
struct PixelDistributionAdaptationKernel {
    double blend;int mode;
    template<class T,int D> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T,D>& c) const {
        if(blend==0)return v[i];double span=c.input.high-c.input.low;
        double mapped=mode==1?(span>0?(v[i]-c.input.low)/span:0)*(c.reference.high-c.reference.low)+c.reference.low:
            (c.input.stddev>0?(v[i]-c.input.mean)/c.input.stddev:0)*c.reference.stddev+c.reference.mean;
        return (1-blend)*v[i]+blend*mapped;
    }
};
template<class T> inline void channel_apply(const PixelDistributionAdaptation& t,std::vector<T>& v,const Grid& g,const ChannelPlan& p) {
    auto summary=volume_detail::visit_reference_channel(t.reference,p.channel,[](const auto& ref){return volume_detail::stats(ref.data(),ref.size());});
    volume_detail::map_channel<PixelDistributionAdaptation>(v,g,PixelDistributionAdaptationKernel{p.values[0],t.mode},nullptr,summary);
}
inline Parameters sample(const PixelDistributionAdaptation& t,const Sample& s,Random& r) {
    auto p=sample_channels(s,t.options,r,[&](const Array& a,Random& rng){return channel_parameters(t,a,rng);});
    if(!t.reference_key.empty())volume_detail::reference_target(t.reference_key,s,p);
    return p;
}
inline void apply(const PixelDistributionAdaptation& t,Sample& s,const Parameters& p) { volume_detail::apply_reference_channels(t,s,p); }
} // namespace vcr::transforms
