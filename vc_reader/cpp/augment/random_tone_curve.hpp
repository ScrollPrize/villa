#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct RandomToneCurve { Scalar scale=.1; ChannelOptions options; };
inline ChannelPlan channel_parameters(const RandomToneCurve& t,const Array& a,Random& r) {
    volume_detail::require_volume(a);ChannelPlan p;
    double scale=draw(t.scale,r);probability(scale);double lower=volume_detail::clamp(1./3+scale*r.normal(),0,1),upper=volume_detail::clamp(2./3+scale*r.normal(),0,1);if(lower>upper)std::swap(lower,upper);p.values={lower,upper};
    p.seed=r.next();return p;
}
struct RandomToneCurveKernel {
    double lower,upper;
    template<class T> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T>& c) const {
        using namespace volume_detail;
        T span=T(c.input.high-c.input.low);if(span==0)return v[i];T t=clamp((v[i]-T(c.input.low))/span,T(0),T(1)),u=T(1)-t;return T(c.input.low)+span*(3*u*u*t*T(lower)+3*u*t*t*T(upper)+t*t*t);
    }
};
inline RandomToneCurveKernel make_kernel(const RandomToneCurve& t,const ChannelPlan& p) { return {p.values[0],p.values[1]}; }
template<class T> inline void channel_apply(const RandomToneCurve& t,std::vector<T>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel(v,g,make_kernel(t,p));
}
inline void apply(const RandomToneCurve& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
