#pragma once
#include "volume_grid.hpp"
namespace vcr::augment::volume_detail {
VCR_HD inline std::uint64_t mix(std::uint64_t z) {
    z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb); return z^(z>>31);
}
VCR_HD inline double uniform(std::uint64_t seed,std::uint64_t index) {
    return (double(mix(seed+UINT64_C(0x9e3779b97f4a7c15)*(index+1))>>11)+.5)*0x1.0p-53;
}
VCR_HD inline double normal(std::uint64_t seed,std::uint64_t index) {
    return sqrt(-2*log(uniform(seed,2*index)))*cos(2*pi*uniform(seed,2*index+1));
}
struct Rng {
    std::uint64_t seed,index=0;
    VCR_HD double u() { return uniform(seed,index++); }
    VCR_HD double gaussian() { double a=u(),b=u(); return sqrt(-2*log(a))*cos(2*pi*b); }
};
// Exact Poisson sampling: inversion for small rates, PTRS transformed rejection
// for large rates (Hoermann 1993). There is no Gaussian approximation.
VCR_HD inline double poisson(double rate,Rng& r) {
    if (rate<=0) return 0;
    if (rate<10) { double limit=exp(-rate),p=1; int k=0; do { ++k;p*=r.u(); } while(p>limit);return k-1; }
    const double b=.931+2.53*sqrt(rate),a=-.059+.02483*b, inv=1.1239+1.1328/(b-3.4),vr=.9277-3.6224/(b-2);
    for (;;) {
        double u=r.u()-.5,v=r.u(),us=.5-fabs(u),k=floor((2*a/us+b)*u+rate+.43);
        if(us>=.07 && v<=vr) return k;
        if(k<0 || (us<.013 && v>us)) continue;
        if(log(v*inv/(a/(us*us)+b))<=-rate+k*log(rate)-lgamma(k+1)) return k;
    }
}
VCR_HD inline double gamma_sample(double shape,Rng& r) {
    if(shape<1) { double u=r.u();return gamma_sample(shape+1,r)*pow(u,1/shape); }
    double d=shape-1./3,c=1/sqrt(9*d);
    for(;;) { double x=r.gaussian(),v=1+c*x; if(v<=0)continue;v=v*v*v;double u=r.u();
        if(u<1-.0331*x*x*x*x || log(u)<.5*x*x+d*(1-v+log(v))) return d*v; }
}

} // namespace vcr::augment::volume_detail
