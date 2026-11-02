#pragma once
#include "../sampling_math.hpp"
#include "../geometry/math.hpp"
#include <cstdint>
#ifdef __CUDACC__
#define VCR_MASK_HD __host__ __device__
#else
#define VCR_MASK_HD
#endif
namespace vcr::projection {
// Binary support swept through a band around already sampled positions/normals.
// Coordinates are ZYX; the supplied normals may be bicubic surface normals.
// Linear interpolation inside a swept cell fills its volume without overshoot.
struct MaskBand {
    const float* positions=nullptr;
    const float* normals=nullptr;
    const bool* valid=nullptr;
    const bool* mask=nullptr;
    int rows=0,cols=0;
    std::int64_t origin[3]{},shape[3]{};
    float halfThickness=0;
};
using geometry::V;
VCR_MASK_HD inline float extent(V a,V b){return fmaxf(fabsf(a.x-b.x),fmaxf(fabsf(a.y-b.y),fabsf(a.z-b.z)));}
VCR_MASK_HD inline bool offsetPoint(const MaskBand& g,int r,int c,float t,V& out){
    if(r<0||c<0||r>=g.rows||c>=g.cols)return false;auto i=r*g.cols+c;
    if(!g.mask[i]||!g.valid[i])return false;
    auto p=geometry::vector(g.positions+i*3),n=geometry::vector(g.normals+i*3);
    float norm=sqrtf(geometry::dot(n,n));if(!geometry::finite(p)||!geometry::finite(n)||norm<=1e-6f)return false;
    out=p+n*(t/norm);return true;
}
template<class Mark>VCR_MASK_HD inline void line(V a,V b,Mark mark){
    int steps=int(ceilf(extent(a,b)));if(steps<1){mark(a);return;}
    for(int i=0;i<=steps;++i)mark(a+(b-a)*(float(i)/steps));
}
template<class Mark>VCR_MASK_HD inline void fillCell(const V* lo,const V* hi,bool slab,Mark mark){
    float dr=fmaxf(extent(lo[0],lo[2]),extent(lo[1],lo[3]));
    float dc=fmaxf(extent(lo[0],lo[1]),extent(lo[2],lo[3])),dt=0;
    if(slab){dr=fmaxf(dr,fmaxf(extent(hi[0],hi[2]),extent(hi[1],hi[3])));dc=fmaxf(dc,fmaxf(extent(hi[0],hi[1]),extent(hi[2],hi[3])));for(int k=0;k<4;++k)dt=fmaxf(dt,extent(lo[k],hi[k]));}
    int nr=int(fmaxf(1,ceilf(dr*2))),nc=int(fmaxf(1,ceilf(dc*2))),nt=slab?int(fmaxf(1,ceilf(dt*2))):0;
    for(int t=0;t<=nt;++t){float ft=nt?float(t)/nt:0;V p[4]{};for(int k=0;k<4;++k)p[k]=slab?lo[k]*(1-ft)+hi[k]*ft:lo[k];
        for(int r=0;r<=nr;++r){float fr=float(r)/nr;V a=p[0]*(1-fr)+p[2]*fr,b=p[1]*(1-fr)+p[3]*fr;
            for(int c=0;c<=nc;++c){float fc=float(c)/nc;mark(a*(1-fc)+b*fc);}}
    }
}
template<class Mark>VCR_MASK_HD inline void maskCell(const MaskBand& g,int at,Mark mark){
    int r=at/g.cols,c=at%g.cols;V previous[4];bool have=false,haveCell=false;
    int radius=int(ceilf(g.halfThickness));
    for(int step=-radius;step<=radius;++step){if(fabsf(float(step))>g.halfThickness+1e-6f)continue;
        V p[4]{};if(!offsetPoint(g,r,c,float(step),p[0]))break;
        mark(p[0]);if(have)line(previous[0],p[0],mark);
        bool right=offsetPoint(g,r,c+1,float(step),p[1]),down=offsetPoint(g,r+1,c,float(step),p[2]),diag=offsetPoint(g,r+1,c+1,float(step),p[3]);
        if(right)line(p[0],p[1],mark);if(down)line(p[0],p[2],mark);
        bool cell=right&&down&&diag;
        if(cell){fillCell(p,p,false,mark);if(haveCell)fillCell(previous,p,true,mark);}
        for(int k=0;k<4;++k)previous[k]=p[k];have=true;haveCell=cell;
    }
}
VCR_MASK_HD inline std::int64_t maskIndex(const MaskBand& g,V p){
    if(!geometry::finite(p)||fabsf(p.x)>0x1p30f||fabsf(p.y)>0x1p30f||fabsf(p.z)>0x1p30f)return -1;
    // Truncation is the voxel convention of this binary band projection.
    std::int64_t q[3]{std::int64_t(p.x)-g.origin[0],std::int64_t(p.y)-g.origin[1],std::int64_t(p.z)-g.origin[2]};
    for(int a=0;a<3;++a)if(q[a]<0||q[a]>=g.shape[a])return -1;
    return (q[0]*g.shape[1]+q[1])*g.shape[2]+q[2];
}
void projectMask(const MaskBand&,std::uint32_t* output,int device=-1);
#ifdef VC_READER_CUDA
void projectMaskCuda(const MaskBand&,std::uint32_t*,int);
#endif
}
#undef VCR_MASK_HD
