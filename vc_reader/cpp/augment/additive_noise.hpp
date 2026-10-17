#pragma once
#include "voxel_transform.hpp"
namespace vcr::augment {
struct AdditiveNoise { Scalar scale=.05,mean=0.,alpha=2.,beta=2.;int distribution=0;bool shared_noise=false; ChannelOptions options; };
inline ChannelPlan channel_parameters(const AdditiveNoise& t,const Array& a,Random& r) {
    volume_detail::require_volume(a);ChannelPlan p;
    double scale=draw(t.scale,r),mean=draw(t.mean,r),alpha=draw(t.alpha,r),beta=draw(t.beta,r);require(scale>=0 && alpha>0 && beta>0,"noise scale must be nonnegative, beta shapes positive");p.values={scale,mean,alpha,beta};
    p.seed=r.next();return p;
}
struct AdditiveNoiseKernel {
    double scale,mean,alpha,beta;int distribution;std::uint64_t seed;
    VCR_HD double operator()(const double* v,std::size_t i,const volume_detail::ChannelContext& c) const {
        using namespace volume_detail;
        Rng r{mix(seed+i)};double u=r.u(),noise=0;
        if(distribution==0)noise=(2*u-1)*scale;
        else if(distribution==1)noise=scale*r.gaussian();
        else if(distribution==2)noise=-scale*(u<.5?-1:1)*log(1-2*fabs(u-.5));
        else {double a=gamma_sample(alpha,r),b=gamma_sample(beta,r);noise=scale*(a/(a+b)-alpha/(alpha+beta));}
        return v[i]+mean+noise;
    }
};
inline AdditiveNoiseKernel make_kernel(const AdditiveNoise& t,const ChannelPlan& p) { return {p.values[0],p.values[1],p.values[2],p.values[3],t.distribution,p.seed}; }
inline void channel_apply(const AdditiveNoise& t,std::vector<double>& v,const Grid& g,const ChannelPlan& p) {
    volume_detail::map_channel(v,g,make_kernel(t,p));
}
inline Parameters sample(const AdditiveNoise& t,const Sample& s,Random& r) { return volume_detail::sample_noise(t,s,r); }
inline void apply(const AdditiveNoise& t,Sample& s,const Parameters& p) { volume_detail::apply_checked_channels(t,s,p); }
} // namespace vcr::augment
