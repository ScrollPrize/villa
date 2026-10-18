#pragma once
#include "random_sampling.hpp"
namespace vcr::augment::volume_detail {
// Random lattice evaluated in 3D. Bias uses trilinear Gaussian log-gains;
// fractal fields use smooth value noise at successively finer octave scales.
template<class T> struct RandomLattice {
    std::uint64_t seed;bool gaussian;
    VCR_HD T operator()(const long long* lo,int corner) const {
        std::uint64_t h=seed;
        for(int a=0;a<3;++a){int bit=(corner>>a)&1;h=mix(h^mix(std::uint64_t(lo[a]+bit)+UINT64_C(0x9e3779b97f4a7c15)*(a+1)));}
        return gaussian?normal<T>(h,0):T(2)*T(uniform(h,0))-T(1);
    }
};
template<class T,class Nodes> VCR_HD inline T lattice_interpolate(const long long* lo,const T* f,const Nodes& nodes) {
    T value=0;
    for(int c=0;c<8;++c){
        T w=1;for(int a=0;a<3;++a)w*=((c>>a)&1)?f[a]:1-f[a];
        value+=w*T(nodes(lo,c));
    }
    return value;
}
template<class T> VCR_HD inline T lattice(const T* q,std::uint64_t seed,bool gaussian,bool smooth) {
    long long lo[3];T f[3];
    for(int a=0;a<3;++a){lo[a]=(long long)floor(q[a]);f[a]=q[a]-lo[a];if(smooth)f[a]=f[a]*f[a]*(3-2*f[a]);}
    return lattice_interpolate(lo,f,RandomLattice<T>{seed,gaussian});
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

namespace vcr::augment::volume_detail {
struct CachedLatticeNodes {
    const float* values;std::size_t ny,nx;
    VCR_HD float operator()(const long long* lo,int corner) const {
        return values[((std::size_t(lo[0])+(corner&1))*ny+std::size_t(lo[1])+((corner>>1)&1))*nx+std::size_t(lo[2])+((corner>>2)&1)];
    }
};
// Cache coarse lattice vertices, not a random sample per output voxel. The
// bounded cache falls back to the same stateless hash for fine/high octaves.
inline void accumulate_lattice(std::vector<float>& output,const Grid& g,float frequency,
                               std::uint64_t seed,bool gaussian,bool smooth,float weight) {
    std::array<std::vector<long long>,3> lower;
    std::array<std::vector<float>,3> fraction;
    std::size_t extent[3],count=1;
    const std::size_t budget=std::min<std::size_t>(g.size,8*1024*1024);
    bool cached=true;
    for(int a=0;a<3;++a){
        lower[a].resize(g.shape[a]);fraction[a].resize(g.shape[a]);
        for(std::size_t i=0;i<g.shape[a];++i){
            float q=float(i)*frequency;lower[a][i]=(long long)floor(q);
            float f=q-lower[a][i];fraction[a][i]=smooth?f*f*(3-2*f):f;
        }
        extent[a]=std::size_t(lower[a].back())+2;
        if(extent[a]>budget/count)cached=false;
        if(cached)count*=extent[a];
    }
    const RandomLattice<float> random{seed,gaussian};std::vector<float> nodes;
    if(cached){
        nodes.resize(count);
        for(std::size_t z=0;z<extent[0];++z)for(std::size_t y=0;y<extent[1];++y)for(std::size_t x=0;x<extent[2];++x){
            const long long lo[3]{(long long)z,(long long)y,(long long)x};
            nodes[(z*extent[1]+y)*extent[2]+x]=random(lo,0);
        }
    }
    auto accumulate=[&](const auto& source){
        std::size_t i=0;
        for(std::size_t z=0;z<g.shape[0];++z)for(std::size_t y=0;y<g.shape[1];++y)for(std::size_t x=0;x<g.shape[2];++x,++i){
            const long long lo[3]{lower[0][z],lower[1][y],lower[2][x]};
            const float f[3]{fraction[0][z],fraction[1][y],fraction[2][x]};
            output[i]+=weight*lattice_interpolate(lo,f,source);
        }
    };
    if(cached)accumulate(CachedLatticeNodes{nodes.data(),extent[1],extent[2]});else accumulate(random);
}
inline std::vector<float> fractal_field(const Grid& g,std::uint64_t seed,float scale,float persistence,int octaves) {
    std::vector<float> field(g.size);float weight=1,total=0,frequency=float(scale);
    for(int octave=0;octave<octaves;++octave){
        accumulate_lattice(field,g,frequency,mix(seed+octave),false,true,weight);
        total+=weight;weight*=persistence;frequency*=2;
    }
    for(auto& value:field)value=total?value/total:0;
    return field;
}
} // namespace vcr::augment::volume_detail
