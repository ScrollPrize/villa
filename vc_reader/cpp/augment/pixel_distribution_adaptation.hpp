#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct PixelDistributionAdaptation {volume_detail::ReferenceVolume reference;Scalar blend=Uniform{.5,1};int mode=0;ChannelOptions options;};
inline ChannelPlan channel_parameters(const PixelDistributionAdaptation& t,const Array& a,Random& r) {
    volume_detail::require_volume(a);double blend=draw(t.blend,r);probability(blend);ChannelPlan p;p.values={blend};return p;
}
struct PixelDistributionAdaptationKernel {
    double blend;int mode;
    template<class T> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T>& c) const {
        if(blend==0)return v[i];double span=c.input.high-c.input.low;
        double mapped=mode==1?(span>0?(v[i]-c.input.low)/span:0)*(c.reference.high-c.reference.low)+c.reference.low:
            (c.input.stddev>0?(v[i]-c.input.mean)/c.input.stddev:0)*c.reference.stddev+c.reference.mean;
        return (1-blend)*v[i]+blend*mapped;
    }
};
template<class T> inline void channel_apply(const PixelDistributionAdaptation& t,std::vector<T>& v,const Grid& g,const ChannelPlan& p) {
    auto summary=volume_detail::visit_reference_channel(t.reference,p.channel,[](const auto& ref){return volume_detail::stats(ref.data(),ref.size());});
    volume_detail::map_channel(v,g,PixelDistributionAdaptationKernel{p.values[0],t.mode},nullptr,summary);
}
inline void apply(const PixelDistributionAdaptation& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
