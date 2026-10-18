#pragma once
#include "../sampling_math.hpp"
#include "../directional_math.hpp"
#include <cstdint>
#include <cstddef>
#include <cmath>
#ifdef __CUDACC__
#define VCR_ALIGNED_HD __host__ __device__
#else
#define VCR_ALIGNED_HD
#endif
namespace vcr::aligned {
struct Field {
    float matrix[9]{}, translation[3]{};
    int component[3]{0,1,2};
    bool normalize=false, normal=false, enabled=false;
};
VCR_ALIGNED_HD inline void field(float* data,bool* valid,std::size_t n,std::size_t i,Field f) {
    float v[3]{};
    for(int j=0;j<3;++j) v[f.component[j]]=data[j*n+i];
    direction::apply(f.matrix,v,3);
    for(int j=0;j<3;++j) v[j]+=f.translation[j];
    if(f.normalize) {
        // Scale first to avoid overflow/underflow of the squared float32 norm.
        float scale=0;
        for(int j=0;j<3;++j) scale=fmaxf(scale,fabsf(v[j]));
        bool defined=scale>0;
        for(int j=0;j<3;++j) defined &= std::isfinite(v[j]);
        if(defined) { for(auto& x:v) x/=scale; direction::unit(v,3); }
        else if(f.normal) { for(auto& x:v) x=0; valid[i]=false; }
    }
    for(int j=0;j<3;++j) data[j*n+i]=v[f.component[j]];
}
struct Grid {
    std::int64_t source[3]{}, shape[3]{}, output[3]{}, offset[3]{}, base[3]{}, tile[3]{}, tiles[3]{};
    float origin[3]{}, axes[9]{};
    std::size_t count=0, output_count=0, tile_count=0;
    int channels=1, order=1;
    bool plane=false;
    Field field{};
};
struct Cell {
    std::int64_t at[8][3]{};
    float weight[8]{};
    int count=0;
    bool support[8]{}, valid=false;
    std::size_t output=0;
};
VCR_ALIGNED_HD inline bool supported(const Grid& g,std::size_t i) {
    const std::int64_t p[3]{std::int64_t(i/g.shape[2]/g.shape[1])+g.offset[0],
        std::int64_t(i/g.shape[2]%g.shape[1])+g.offset[1],std::int64_t(i%g.shape[2])+g.offset[2]};
    for(int a=0;a<3;++a) {
        const float q=sampling::affine(g.origin[a],g.axes+3*a,float(p[0]),float(p[1]),float(p[2]));
        if(!std::isfinite(q) || q<=-0x1p60f || q>=0x1p60f) return false;
        if(g.plane && a==0) { if(q<-.5f || q>.5f) return false; continue; }
        const float f=floorf(q), fraction=q-f;
        auto lo=g.base[a]+std::int64_t(f);
        if(!g.order) lo+=fraction>.5f || (fraction==.5f && lo%2!=0);
        const auto hi=lo+(g.order && fraction!=0);
        if(lo<0 || hi>=g.source[a]) return false;
    }
    return true;
}
VCR_ALIGNED_HD inline std::size_t output_index(const Grid& g,std::size_t i) {
    return ((i/g.shape[2]/g.shape[1]+g.offset[0])*g.output[1]+i/g.shape[2]%g.shape[1]+g.offset[1])*g.output[2]+i%g.shape[2]+g.offset[2];
}
VCR_ALIGNED_HD inline Cell cell(const Grid& g, std::size_t i) {
    Cell c;
    const auto x=i%g.shape[2], y=i/g.shape[2]%g.shape[1], z=i/g.shape[2]/g.shape[1];
    c.output=((z+g.offset[0])*g.output[1]+y+g.offset[1])*g.output[2]+x+g.offset[2];
    float fraction[3]{}; std::int64_t lo[3]{};
    for (int a=0;a<3;++a) {
        const float q=sampling::affine(g.origin[a],g.axes+3*a,float(z+g.offset[0]),float(y+g.offset[1]),float(x+g.offset[2]));
        if (!std::isfinite(q) || q<=-0x1p60f || q>=0x1p60f) return c;
        if (g.plane && a==0) {
            // The normal coordinate has no integer global origin.
            if (q<-.5f || q>.5f) return c;
            continue;
        }
        const float f=floorf(q);
        lo[a]=g.base[a]+std::int64_t(f); fraction[a]=q-f;
        if (!g.order) lo[a]+=fraction[a]>.5f || (fraction[a]==.5f && lo[a]%2!=0);
    }
    const int first=g.plane ? 1 : 0, corners=g.order ? (1<<(3-first)) : 1;
    c.valid=true;
    for (int k=0;k<corners;++k) {
        float w=1; std::int64_t at[3]{0,0,0}; bool inside=true;
        for (int a=first;a<3;++a) {
            const int bit=g.order ? ((k>>(a-first))&1) : 0;
            if (g.order) w*=bit ? fraction[a] : 1-fraction[a];
            at[a]=lo[a]+bit; inside &= at[a]>=0 && at[a]<g.source[a];
        }
        if (w==0) continue;
        for (int a=0;a<3;++a) c.at[c.count][a]=at[a];
        c.weight[c.count]=w; c.support[c.count]=inside; c.valid &= inside; ++c.count;
    }
    return c;
}
VCR_ALIGNED_HD inline std::int64_t key(const Grid& g,const std::int64_t* p) {
    return (p[0]/g.tile[0]*g.tiles[1]+p[1]/g.tile[1])*g.tiles[2]+p[2]/g.tile[2];
}
VCR_ALIGNED_HD inline std::size_t local(const Grid& g,const std::int64_t* p) {
    return (p[0]%g.tile[0]*g.tile[1]+p[1]%g.tile[1])*g.tile[2]+p[2]%g.tile[2];
}
VCR_ALIGNED_HD inline std::size_t find(const std::int64_t* keys,std::size_t n,std::int64_t k) {
    std::size_t lo=0,hi=n;
    while (lo<hi) { const auto m=lo+(hi-lo)/2; if (keys[m]<k) lo=m+1; else hi=m; }
    return lo;
}
template<class T> VCR_ALIGNED_HD void sample(const Grid& g,const std::int64_t* keys,
        const T* const* tiles,std::size_t n,T fill,void* output,bool* valid,std::size_t i) {
    const auto c=cell(g,i); valid[c.output]=c.valid;
    bool any_support=false;
    const T* neighbours[8]{};
    std::int64_t previous=-1; const T* pointer=nullptr;
    for (int k=0;k<c.count;++k) if(c.support[k]) {
        any_support=true;
        const auto id=key(g,c.at[k]);
        if(id!=previous) { const auto slot=find(keys,n,id); pointer=slot<n ? tiles[slot] : nullptr; previous=id; }
        if(pointer) neighbours[k]=pointer+local(g,c.at[k]);
    }
    for(int ch=0;ch<g.channels;++ch) {
        const auto out=std::size_t(ch)*g.output_count+c.output;
        if(!g.order) static_cast<T*>(output)[out]=neighbours[0] ? neighbours[0][ch*g.tile_count] : fill;
        else {
            float value=0;
            if(!c.count) value=float(fill);
            for(int k=0;k<c.count;++k) value+=c.weight[k]*float(neighbours[k] ? neighbours[k][ch*g.tile_count] : fill);
            static_cast<float*>(output)[out]=value;
        }
    }
    if(g.field.enabled && any_support) field(static_cast<float*>(output),valid,g.output_count,c.output,g.field);
}

}
#undef VCR_ALIGNED_HD
