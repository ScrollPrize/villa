#pragma once
#include "random_sampling.hpp"
namespace vcr::transforms::volume_detail {
// Random lattice in the image dimension. Bias uses multilinear Gaussian log-gains;
// fractal fields use smooth value noise at successively finer octave scales.
template<class T,int D=3> struct RandomLattice {
    std::uint64_t seed;bool gaussian;
    VCR_HD T operator()(const long long* lo,int corner) const {
        std::uint64_t h=seed;
        for(int a=0;a<D;++a){int bit=(corner>>a)&1;h=mix(h^mix(std::uint64_t(lo[a]+bit)+UINT64_C(0x9e3779b97f4a7c15)*(a+1)));}
        return gaussian?normal<T>(h,0):T(2)*T(uniform(h,0))-T(1);
    }
};
template<int D=3,class T,class Nodes> VCR_HD inline T lattice_interpolate(const long long* lo,const T* f,const Nodes& nodes) {
    T value=0;
    for(int c=0;c<(1<<D);++c){
        T w=1;for(int a=0;a<D;++a)w*=((c>>a)&1)?f[a]:1-f[a];
        value+=w*T(nodes(lo,c));
    }
    return value;
}
template<class T,int D=3> VCR_HD inline T lattice(const T* q,std::uint64_t seed,bool gaussian,bool smooth) {
    long long lo[3];T f[3];
    for(int a=0;a<D;++a){lo[a]=(long long)floor(q[a]);f[a]=q[a]-lo[a];if(smooth)f[a]=f[a]*f[a]*(3-2*f[a]);}
    return lattice_interpolate<D>(lo,f,RandomLattice<T,D>{seed,gaussian});
}
template<class T,int D> VCR_HD inline T fractal(const GridND<D>& g, std::size_t i, std::uint64_t seed,
                              T scale, T persistence, int octaves) {
    T sum=0, weight=1, total=0, frequency=scale;
    for(int octave=0;octave<octaves;++octave) {
        T q[3];for(int axis=0;axis<D;++axis)q[axis]=T(g.coord(i,axis))*frequency;
        sum+=weight*lattice<T,D>(q,mix(seed+octave),false,true);total+=weight;
        weight*=persistence;frequency*=2;
    }
    return total?sum/total:0;
}

} // namespace vcr::transforms::volume_detail

namespace vcr::transforms::volume_detail {
template<int D> struct CachedLatticeNodes {
    const float* values;std::size_t ny,nx;
    VCR_HD float operator()(const long long* lo,int corner) const {
        if constexpr(D==2) return values[(std::size_t(lo[0])+(corner&1))*nx+std::size_t(lo[1])+((corner>>1)&1)];
        return values[((std::size_t(lo[0])+(corner&1))*ny+std::size_t(lo[1])+((corner>>1)&1))*nx+std::size_t(lo[2])+((corner>>2)&1)];
    }
};
// Cache coarse lattice vertices, not a random sample per output voxel. The
// bounded cache falls back to the same stateless hash for fine/high octaves.
template<int D> inline void accumulate_lattice_impl(std::vector<float>& output,const Grid& g,float frequency,
                               std::uint64_t seed,bool gaussian,bool smooth,float weight) {
    std::array<std::vector<long long>,3> lower;
    std::array<std::vector<float>,3> fraction;
    std::size_t extent[3],count=1;
    const std::size_t budget=std::min<std::size_t>(g.size,8*1024*1024);
    bool cached=true;
    for(int a=0;a<D;++a){
        lower[a].resize(g.shape[a]);fraction[a].resize(g.shape[a]);
        for(std::size_t i=0;i<g.shape[a];++i){
            float q=float(i)*frequency;lower[a][i]=(long long)floor(q);
            float f=q-lower[a][i];fraction[a][i]=smooth?f*f*(3-2*f):f;
        }
        extent[a]=std::size_t(lower[a].back())+2;
        if(extent[a]>budget/count)cached=false;
        if(cached)count*=extent[a];
    }
    const RandomLattice<float,D> random{seed,gaussian};std::vector<float> nodes;
    if(cached){
        nodes.resize(count);
        const auto nz=D==3?extent[0]:1,ny=extent[D-2],nx=extent[D-1];
        for(std::size_t z=0;z<nz;++z)for(std::size_t y=0;y<ny;++y)for(std::size_t x=0;x<nx;++x){
            long long lo[3]{(long long)y,(long long)x,0};
            if constexpr(D==3){lo[0]=z;lo[1]=y;lo[2]=x;}
            nodes[(z*ny+y)*nx+x]=random(lo,0);
        }
    }
    auto accumulate=[&](const auto& source){
        std::size_t i=0;
        const auto nz=D==3?g.shape[0]:1,ny=g.shape[D-2],nx=g.shape[D-1];
        for(std::size_t z=0;z<nz;++z)for(std::size_t y=0;y<ny;++y)for(std::size_t x=0;x<nx;++x,++i){
            long long lo[3];float f[3];
            if constexpr(D==3){lo[0]=lower[0][z];lo[1]=lower[1][y];lo[2]=lower[2][x];f[0]=fraction[0][z];f[1]=fraction[1][y];f[2]=fraction[2][x];}
            else {lo[0]=lower[0][y];lo[1]=lower[1][x];f[0]=fraction[0][y];f[1]=fraction[1][x];}
            output[i]+=weight*lattice_interpolate<D>(lo,f,source);
        }
    };
    if(cached)accumulate(CachedLatticeNodes<D>{nodes.data(),extent[D-2],extent[D-1]});else accumulate(random);
}
inline void accumulate_lattice(std::vector<float>& output,const Grid& g,float frequency,
                               std::uint64_t seed,bool gaussian,bool smooth,float weight) {
    if(g.shape.size()==3) accumulate_lattice_impl<3>(output,g,frequency,seed,gaussian,smooth,weight);
    else accumulate_lattice_impl<2>(output,g,frequency,seed,gaussian,smooth,weight);
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
} // namespace vcr::transforms::volume_detail
