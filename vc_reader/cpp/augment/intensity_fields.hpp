#pragma once
#include "random_sampling.hpp"
namespace vcr::augment::volume_detail {
// Random lattice evaluated in 3D. Bias uses trilinear Gaussian log-gains;
// fractal fields use smooth value noise at successively finer octave scales.
template<class T> VCR_HD inline T lattice(const T* q,std::uint64_t seed,bool gaussian,bool smooth) {
    long long lo[3];T f[3];
    for(int a=0;a<3;++a) { lo[a]=(long long)floor(q[a]);f[a]=q[a]-lo[a];if(smooth) f[a]=f[a]*f[a]*(3-2*f[a]); }
    T value=0;
    for(int c=0;c<8;++c) {
        T w=1;std::uint64_t h=seed;
        for(int a=0;a<3;++a) { int bit=(c>>a)&1; w*=bit?f[a]:1-f[a];h=mix(h^mix(std::uint64_t(lo[a]+bit)+UINT64_C(0x9e3779b97f4a7c15)*(a+1))); }
        value+=w*T(gaussian?normal(h,0):2*uniform(h,0)-1);
    }
    return value;
}
template<class T> VCR_HD inline T fractal(const Grid3& g, std::size_t i, std::uint64_t seed,
                              T scale, T persistence, int octaves) {
    T sum=0, weight=1, total=0, frequency=scale;
    for(int octave=0;octave<octaves;++octave) {
        T q[3];for(int axis=0;axis<3;++axis)q[axis]=T(g.coord(i,axis))*frequency;
        sum+=weight*lattice(q,mix(seed+octave),false,true);total+=weight;
        weight*=persistence;frequency*=2;
    }
    return total?sum/total:0;
}

} // namespace vcr::augment::volume_detail
