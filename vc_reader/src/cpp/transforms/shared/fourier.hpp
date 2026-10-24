#pragma once
#include "volume_grid.hpp"
namespace vcr::transforms::volume_detail {
// FFT uses Bluestein for arbitrary line lengths, backed by a radix-2 FFT.
// CUDA uses cuFFT with the same transform convention.
template<class T> struct ComplexT {
    T re=0,im=0;
    VCR_HD ComplexT operator+(ComplexT b) const{return {re+b.re,im+b.im};}
    VCR_HD ComplexT operator-(ComplexT b) const{return {re-b.re,im-b.im};}
    VCR_HD ComplexT operator*(ComplexT b) const{return {re*b.re-im*b.im,re*b.im+im*b.re};}
    VCR_HD ComplexT scale(T x) const{return {re*x,im*x};}
};
using Complex=ComplexT<double>;
VCR_HD inline Complex phase(double x){return {cos(x),sin(x)};}
template<class T> VCR_HD inline void radix2(ComplexT<T>* v,std::size_t n,bool inverse) {
    for(std::size_t i=1,j=0;i<n;++i){std::size_t bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j){auto t=v[i];v[i]=v[j];v[j]=t;}}
    for(std::size_t len=2;len<=n;len*=2){auto step=phase((inverse?2:-2)*pi/len);
        for(std::size_t i=0;i<n;i+=len){Complex w{1,0};for(std::size_t j=0;j<len/2;++j){auto a=v[i+j],b=v[i+j+len/2]*ComplexT<T>{T(w.re),T(w.im)};v[i+j]=a+b;v[i+j+len/2]=a-b;w=w*step;}}}
    if(inverse)for(std::size_t i=0;i<n;++i)v[i]=v[i].scale(1./n);
}
inline std::size_t fft_scratch(std::size_t n){std::size_t m=1;while(m<2*n-1)m*=2;return 2*m;}
// Bluestein's chirp and its transformed convolution kernel depend only on the
// axis length/direction, not on the line. Keep the original FFT and arithmetic
// order, but prepare these once per axis instead of once per line.
template<class T> struct FourierLinePlan {
    std::size_t n,m;
    bool inverse;
    std::vector<ComplexT<T>> chirp,kernel;
    FourierLinePlan(std::size_t length,bool inv):n(length),m(fft_scratch(n)/2),inverse(inv),chirp(n),kernel(m) {
        double sign=inverse?1:-1;
        for(std::size_t i=0;i<n;++i){
            double angle=sign*pi*(double(i)*double(i)/n);
            auto c=phase(angle), k=phase(-angle);chirp[i]={T(c.re),T(c.im)};kernel[i]={T(k.re),T(k.im)};
            if(i)kernel[m-i]=kernel[i];
        }
        radix2(kernel.data(),m,false);
    }
    void apply(ComplexT<T>* v,std::size_t stride,ComplexT<T>* scratch) const {
        std::fill(scratch,scratch+m,ComplexT<T>{});
        for(std::size_t i=0;i<n;++i)scratch[i]=v[i*stride]*chirp[i];
        radix2(scratch,m,false);
        for(std::size_t i=0;i<m;++i)scratch[i]=scratch[i]*kernel[i];
        radix2(scratch,m,true);
        for(std::size_t i=0;i<n;++i)v[i*stride]=(scratch[i]*chirp[i]).scale(inverse?1./n:1);
    }
};

} // namespace vcr::transforms::volume_detail

namespace vcr::transforms::volume_detail {
template<class T,int D> inline void fft_volume(std::vector<ComplexT<T>>& data,const GridND<D>& g,bool inverse) {
    for(int a=0;a<D;++a){
        auto len=g.n[a],stride=g.stride[a],count=g.size/len;
        FourierLinePlan<T> plan(len,inverse);std::vector<ComplexT<T>> scratch(plan.m);
        for(std::size_t line=0;line<count;++line){
            auto base=(line/stride)*len*stride+line%stride;
            plan.apply(data.data()+base,stride,scratch.data());
        }
    }
}
template<int D> VCR_HD inline bool low_frequency(const GridND<D>& g,std::size_t i,double fraction) {
    for(int a=0;a<D;++a){auto k=g.coord(i,a);k=k<g.n[a]-k?k:g.n[a]-k;if(double(k)>fraction*double(g.n[a])/2)return false;}return true;
}
} // namespace vcr::transforms::volume_detail
