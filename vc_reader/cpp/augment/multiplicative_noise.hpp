#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct MultiplicativeNoise { Scalar low=.9,high=1.1;bool shared_noise=false; ChannelOptions options; };
inline ChannelPlan channel_parameters(const MultiplicativeNoise& t,const Array& a,Random& r) {
    volume_detail::require_volume(a);ChannelPlan p;
    double low=draw(t.low,r),high=draw(t.high,r);require(low>=0 && high>=low,"multipliers need 0 <= low <= high");p.values={low,high};
    p.seed=r.next();return p;
}
struct MultiplicativeNoiseKernel {
    double low,high;std::uint64_t seed;
    template<class T> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T>& c) const {
        using namespace volume_detail;
        return v[i]*(T(low)+T(high-low)*T(uniform(seed,i)));
    }
};
inline MultiplicativeNoiseKernel make_kernel(const MultiplicativeNoise& t,const ChannelPlan& p) { return {p.values[0],p.values[1],p.seed}; }
inline void channel_apply(const MultiplicativeNoise& t,std::vector<double>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel(v,g,make_kernel(t,p));
}
inline Parameters sample(const MultiplicativeNoise& t,const Sample& s,Random& r) { return volume_detail::sample_noise(t,s,r); }
inline void apply(const MultiplicativeNoise& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
