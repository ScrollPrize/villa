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
    VCR_HD double operator()(const double* v,std::size_t i,const volume_detail::ChannelContext& c) const {
        using namespace volume_detail;
        double span=c.input.high-c.input.low;if(span==0)return v[i];double t=clamp((v[i]-c.input.low)/span,0,1),u=1-t;return c.input.low+span*(3*u*u*t*lower+3*u*t*t*upper+t*t*t);
    }
};
inline RandomToneCurveKernel make_kernel(const RandomToneCurve& t,const ChannelPlan& p) { return {p.values[0],p.values[1]}; }
inline void channel_apply(const RandomToneCurve& t,std::vector<double>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel(v,g,make_kernel(t,p));
}
inline void apply(const RandomToneCurve& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
