#pragma once
#include "volume_grid.hpp"
namespace vcr::augment::volume_detail {
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

namespace vcr::augment::volume_detail {
inline void fft_volume(std::vector<Complex>& data,const Grid3& g,bool inverse) {
    for(int a=0;a<3;++a){auto len=g.n[a],stride=g.stride[a],count=g.size/len,size=fft_scratch(len);std::vector<Complex> scratch(size);
        for(std::size_t line=0;line<count;++line){auto base=(line/stride)*len*stride+line%stride;fft_line(data.data()+base,len,stride,scratch.data(),size/2,inverse);}}
}
VCR_HD inline bool low_frequency(const Grid3& g,std::size_t i,double fraction) {
    for(int a=0;a<3;++a){auto k=g.coord(i,a);k=k<g.n[a]-k?k:g.n[a]-k;if(double(k)>fraction*double(g.n[a])/2)return false;}return true;
}
} // namespace vcr::augment::volume_detail
