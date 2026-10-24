#pragma once
#include "../core.hpp"

namespace vcr::transforms::volume_detail {
constexpr double pi = 3.14159265358979323846;
template<int D> struct GridND {
    static constexpr int dim = D;
    std::size_t n[D], stride[D], size;
    explicit GridND(const Shape& s) {
        require(s.size() == D, "expected (C,Y,X) or (C,Z,Y,X) matching the grid dimension");
        size = product(s);
        for (int a=D-1;a>=0;--a) { n[a]=s[a]; stride[a]=a==D-1 ? 1 : stride[a+1]*n[a+1]; }
    }
    VCR_HD std::size_t coord(std::size_t i,int a) const { return i/stride[a]%n[a]; }
};
template<class T> VCR_HD inline T clamp(T x,T a,T b) { return x<a?a:x>b?b:x; }
VCR_HD inline double clamp(double x,double a,double b) { return clamp<double>(x,a,b); }
using Grid3 = GridND<3>;
// Resolve dimension once on the host; pixel loops/kernels specialize on D.
template<class F> decltype(auto) with_grid(const Shape& shape,F&& fn) {
    if (shape.size()==3) return fn(Grid3(shape));
    return fn(GridND<2>(shape));
}
inline Shape spatial_sizes(const Shape& sizes,std::size_t dim,std::size_t fallback) {
    require(sizes.empty() || sizes.size()==dim,"size tuple must match spatial dimensions");
    return sizes.empty() ? Shape(dim,fallback) : sizes;
}
template<class T,int D> VCR_HD T interpolate(const T* v,const GridND<D>& g,const double* q) {
    std::size_t lo[D],hi[D];double f[D];
    for(int a=0;a<D;++a) { double x=clamp(q[a],0,double(g.n[a]-1));lo[a]=std::size_t(floor(x));hi[a]=lo[a]+1<g.n[a]?lo[a]+1:lo[a];f[a]=x-lo[a]; }
    T result=0;
    for(int c=0;c<(1<<D);++c) { std::size_t j=0;double w=1;for(int a=0;a<D;++a){int b=(c>>a)&1;j+=(b?hi[a]:lo[a])*g.stride[a];w*=b?f[a]:1-f[a];} result+=T(w)*v[j]; }
    return result;
}

} // namespace vcr::transforms::volume_detail
