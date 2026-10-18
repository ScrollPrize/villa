#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct ShotNoise { Scalar photons=Uniform{50,200};bool shared_noise=false; ChannelOptions options; };
inline ChannelPlan channel_parameters(const ShotNoise& t,const Array& a,Random& r) {
    volume_detail::require_image(a);ChannelPlan p;
    double photons=draw(t.photons,r);require(photons>0 && photons<=1e9,"photons must be (0,1e9]");p.values={photons};
    p.seed=r.next();return p;
}
struct ShotNoiseKernel {
    double photons;std::uint64_t seed;
    template<class T,int D> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T,D>& c) const {
        using namespace volume_detail;
        if(v[i]<0 || v[i]*photons>1e12)return NAN;Rng r{mix(seed+i)};return poisson(v[i]*photons,r)/photons;
    }
};
inline ShotNoiseKernel make_kernel(const ShotNoise& t,const ChannelPlan& p) { return {p.values[0],p.seed}; }
template<class T> inline void channel_apply(const ShotNoise& t,std::vector<T>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel(v,g,make_kernel(t,p));
}
inline Parameters sample(const ShotNoise& t,const Sample& s,Random& r) { return volume_detail::sample_noise(t,s,r); }
inline void apply(const ShotNoise& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
