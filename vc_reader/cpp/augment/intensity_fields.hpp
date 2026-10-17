#pragma once
#include "random_sampling.hpp"
namespace vcr::augment::volume_detail {
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
VCR_HD inline double fractal(const Grid3& g, std::size_t i, std::uint64_t seed,
                              double scale, double persistence, int octaves) {
    double sum=0, weight=1, total=0, frequency=scale;
    for(int octave=0;octave<octaves;++octave) {
        double q[3];for(int axis=0;axis<3;++axis)q[axis]=double(g.coord(i,axis))*frequency;
        sum+=weight*lattice(q,mix(seed+octave),false,true);total+=weight;
        weight*=persistence;frequency*=2;
    }
    return total?sum/total:0;
}

} // namespace vcr::augment::volume_detail
