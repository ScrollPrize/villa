#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct SaltAndPepper { Scalar amount=.01,salt_fraction=.5;bool shared_noise=true; ChannelOptions options; };
inline ChannelPlan channel_parameters(const SaltAndPepper& t,const Array& a,Random& r) {
    volume_detail::require_image(a);ChannelPlan p;
    double amount=draw(t.amount,r),salt=draw(t.salt_fraction,r);probability(amount);probability(salt);p.values={amount,salt};
    p.seed=r.next();return p;
}
struct SaltAndPepperKernel {
    double amount,salt_fraction;std::uint64_t seed;
    template<class T,int D> VCR_HD T operator()(const T* v,std::size_t i,const volume_detail::ChannelContextT<T,D>& c) const {
        using namespace volume_detail;
        return uniform(seed,i)<amount?(uniform(seed,i+c.grid.size)<salt_fraction?c.input.high:c.input.low):v[i];
    }
};
inline SaltAndPepperKernel make_kernel(const SaltAndPepper& t,const ChannelPlan& p) { return {p.values[0],p.values[1],p.seed}; }
template<class T> inline void channel_apply(const SaltAndPepper& t,std::vector<T>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel(v,g,make_kernel(t,p));
}
inline Parameters sample(const SaltAndPepper& t,const Sample& s,Random& r) { return volume_detail::sample_noise(t,s,r); }
inline void apply(const SaltAndPepper& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
