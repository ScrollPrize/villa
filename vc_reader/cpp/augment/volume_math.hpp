#pragma once
#include "core.hpp"

namespace vcr::augment::volume_detail {
constexpr double pi = 3.14159265358979323846;
// Small POD descriptors and numerical routines shared by CPU and CUDA. No
// allocation, Python objects, global state or host-only algorithms occur here.
struct Grid3 {
    std::size_t n[3], stride[3], size;
    explicit Grid3(const Shape& s) {
        require(s.size() == 3, "volumetric augmentation requires (C,Z,Y,X)");
        size = product(s);
        for (int a=2;a>=0;--a) { n[a]=s[a]; stride[a]=a==2 ? 1 : stride[a+1]*n[a+1]; }
    }
    VCR_HD std::size_t coord(std::size_t i,int a) const { return i/stride[a]%n[a]; }
};
VCR_HD inline double clamp(double x,double a,double b) { return x<a?a:x>b?b:x; }
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
// Random lattice evaluated in 3D. Bias uses trilinear Gaussian log-gains;
// fractal fields use smooth value noise at successively finer octave scales.
VCR_HD inline double lattice(const double* q,std::uint64_t seed,bool gaussian,bool smooth) {
    long long lo[3];double f[3];
    for(int a=0;a<3;++a) { lo[a]=(long long)floor(q[a]);f[a]=q[a]-lo[a];if(smooth) f[a]=f[a]*f[a]*(3-2*f[a]); }
    double value=0;
    for(int c=0;c<8;++c) {
        double w=1;std::uint64_t h=seed;
        for(int a=0;a<3;++a) { int bit=(c>>a)&1; w*=bit?f[a]:1-f[a];h=mix(h^mix(std::uint64_t(lo[a]+bit)+UINT64_C(0x9e3779b97f4a7c15)*(a+1))); }
        value+=w*(gaussian?normal(h,0):2*uniform(h,0)-1);
    }
    return value;
}
template<class T> VCR_HD double interpolate(const T* v,const Grid3& g,const double* q) {
    std::size_t lo[3],hi[3];double f[3];
    for(int a=0;a<3;++a) { double x=clamp(q[a],0,double(g.n[a]-1));lo[a]=std::size_t(floor(x));hi[a]=lo[a]+1<g.n[a]?lo[a]+1:lo[a];f[a]=x-lo[a]; }
    double result=0;
    for(int c=0;c<8;++c) { std::size_t j=0;double w=1;for(int a=0;a<3;++a){int b=(c>>a)&1;j+=(b?hi[a]:lo[a])*g.stride[a];w*=b?f[a]:1-f[a];} result+=w*double(v[j]); }
    return result;
}
struct Stats { double low,high,mean,stddev; };
template<class T> VCR_HD Stats stats(const T* v,std::size_t n) {
    Stats s{double(v[0]),double(v[0]),0,0};double m2=0;
    for(std::size_t i=0;i<n;++i) { double x=v[i];s.low=x<s.low?x:s.low;s.high=x>s.high?x:s.high;double d=x-s.mean;s.mean+=d/double(i+1);m2+=d*(x-s.mean); }
    s.stddev=sqrt(m2>0?m2/double(n):0);return s;
}
VCR_HD inline int bin(double x,double lo,double hi,int bins) {
    return hi>lo?int(clamp((x-lo)/(hi-lo)*bins,0,bins-1)):0;
}
template<class T> VCR_HD void histogram(const T* data,std::size_t n,double* h,int bins,double lo,double hi) {
    for(int b=0;b<bins;++b)h[b]=0;
    for(std::size_t i=0;i<n;++i)h[bin(data[i],lo,hi,bins)]+=1;
}
VCR_HD inline void cdf(double* h,int bins,double clip_limit=0) {
    double total=0;for(int b=0;b<bins;++b)total+=h[b];
    if(clip_limit>0) {
        double limit=fmax(1,clip_limit*total/bins),excess=0;
        for(int b=0;b<bins;++b)if(h[b]>limit){excess+=h[b]-limit;h[b]=limit;}
        for(int b=0;b<bins;++b)h[b]+=excess/bins;
    }
    double sum=0;for(int b=0;b<bins;++b){sum+=h[b];h[b]=total?sum/total:0;}
}
// FFT uses Bluestein for arbitrary line lengths, backed by a radix-2 FFT.
// Identical transform convention and arithmetic are used on both backends.
struct Complex {
    double re=0,im=0;
    VCR_HD Complex operator+(Complex b) const{return {re+b.re,im+b.im};}
    VCR_HD Complex operator-(Complex b) const{return {re-b.re,im-b.im};}
    VCR_HD Complex operator*(Complex b) const{return {re*b.re-im*b.im,re*b.im+im*b.re};}
    VCR_HD Complex scale(double x) const{return {re*x,im*x};}
};
VCR_HD inline Complex phase(double x){return {cos(x),sin(x)};}
VCR_HD inline void radix2(Complex* v,std::size_t n,bool inverse) {
    for(std::size_t i=1,j=0;i<n;++i){std::size_t bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){auto t=v[i];v[i]=v[j];v[j]=t;}}
    for(std::size_t len=2;len<=n;len*=2){auto step=phase((inverse?2:-2)*pi/len);
        for(std::size_t i=0;i<n;i+=len){Complex w{1,0};for(std::size_t j=0;j<len/2;++j){auto a=v[i+j],b=v[i+j+len/2]*w;v[i+j]=a+b;v[i+j+len/2]=a-b;w=w*step;}}}
    if(inverse)for(std::size_t i=0;i<n;++i)v[i]=v[i].scale(1./n);
}
inline std::size_t fft_scratch(std::size_t n){std::size_t m=1;while(m<2*n-1)m*=2;return 2*m;}
VCR_HD inline void fft_line(Complex* v,std::size_t n,std::size_t stride,Complex* scratch,std::size_t m,bool inverse) {
    auto* a=scratch;auto* b=scratch+m;
    for(std::size_t i=0;i<m;++i){a[i]={};b[i]={};}
    double sign=inverse?1:-1;
    for(std::size_t i=0;i<n;++i){double angle=sign*pi*(double(i)*double(i)/n);a[i]=v[i*stride]*phase(angle);b[i]=phase(-angle);if(i)b[m-i]=b[i];}
    radix2(a,m,false);radix2(b,m,false);for(std::size_t i=0;i<m;++i)a[i]=a[i]*b[i];radix2(a,m,true);
    for(std::size_t i=0;i<n;++i)v[i*stride]=(a[i]*phase(sign*pi*(double(i)*double(i)/n))).scale(inverse?1./n:1);
}
} // namespace vcr::augment::volume_detail
