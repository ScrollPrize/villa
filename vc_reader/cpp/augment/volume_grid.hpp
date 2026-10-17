#pragma once
#include "core.hpp"

namespace vcr::augment::volume_detail {
constexpr double pi = 3.14159265358979323846;
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
template<class T> VCR_HD double interpolate(const T* v,const Grid3& g,const double* q) {
    std::size_t lo[3],hi[3];double f[3];
    for(int a=0;a<3;++a) { double x=clamp(q[a],0,double(g.n[a]-1));lo[a]=std::size_t(floor(x));hi[a]=lo[a]+1<g.n[a]?lo[a]+1:lo[a];f[a]=x-lo[a]; }
    double result=0;
    for(int c=0;c<8;++c) { std::size_t j=0;double w=1;for(int a=0;a<3;++a){int b=(c>>a)&1;j+=(b?hi[a]:lo[a])*g.stride[a];w*=b?f[a]:1-f[a];} result+=w*double(v[j]); }
    return result;
}

} // namespace vcr::augment::volume_detail
