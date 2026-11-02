#pragma once
#include "common.hpp"
#include "../grid/points.hpp"
#ifdef __CUDACC__
#define VCR_MASK_HD __host__ __device__
#else
#define VCR_MASK_HD
#endif
namespace vcr::projection {
// In-memory input adapter for the surface-to-volume projector. Coordinates and
// supplied normals are ZYX; positive categorical samples accumulate by union.
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
// A 2x2 patch, repeated row/column for supplied-normal points and edges. All
// interpolation and voxel placement use the same primitives as label projection.
template<class Mark>VCR_MASK_HD inline void maskPatch(const MaskBand& g,const int* indices,Mark mark){
    float xyz[12],normals[12];bool valid[4];
    bool any=false;
    for(int k=0;k<4;++k){
        valid[k]=g.valid[indices[k]];any|=g.mask[indices[k]];
        for(int a=0;a<3;++a){xyz[3*k+a]=g.positions[3*indices[k]+a];normals[3*k+a]=g.normals[3*indices[k]+a];}
    }
    if(!any)return;
    float du=0,dv=0;
    for(int k=0;k<2;++k){
        auto u=geometry::vector(xyz+6*k+3)-geometry::vector(xyz+6*k);
        auto v=geometry::vector(xyz+3*k+6)-geometry::vector(xyz+3*k);
        du=fmaxf(du,sqrtf(geometry::dot(u,u)));dv=fmaxf(dv,sqrtf(geometry::dot(v,v)));
    }
    const int nu=subdivisions(du),nv=subdivisions(dv);
    const auto half=bandHalf(g.halfThickness,1.f);
    grid::GridView view;view.xyz=xyz;view.normals=normals;view.valid=valid;
    grid::PointMap map;map.height=map.width=2;
    const float origin[3]{float(g.origin[0]),float(g.origin[1]),float(g.origin[2])};
    for(int j=0;j<=nv;++j)for(int i=0;i<=nu;++i){
        const float u=float(i)/nu,v=float(j)/nv;
        if(!g.mask[indices[nearest_index(v)*2+nearest_index(u)]])continue;
        float p[3],step[3]{};if(!grid::point(map,view,u,v,p,half?step:nullptr))continue;
        bandSamples(p,step,origin,g.shape,2*half+1,
            [&](std::int64_t k){return bandOffset(g.halfThickness,k-half,half);},
            [&](const std::int64_t* index,std::int64_t){mark((index[0]*g.shape[1]+index[1])*g.shape[2]+index[2]);});
    }
}
template<class Mark>VCR_MASK_HD inline void maskCell(const MaskBand& g,int at,Mark mark){
    const int row=at/g.cols,col=at%g.cols;
    const auto usable=[&](int i){
        const auto p=geometry::vector(g.positions+3*i),n=geometry::vector(g.normals+3*i);
        return g.valid[i]&&geometry::finite(p)&&(g.halfThickness==0||(geometry::finite(n)&&geometry::dot(n,n)>0));
    };
    if(!usable(at))return;
    const bool right=col+1<g.cols&&usable(at+1),down=row+1<g.rows&&usable(at+g.cols);
    const bool cell=right&&down&&usable(at+g.cols+1);
    if(cell){const int ids[4]{at,at+1,at+g.cols,at+g.cols+1};maskPatch(g,ids,mark);return;}
    // With supplied normals, incomplete geometry still has well-defined bands.
    // Repeated edges use grid::point as well; no second line interpolator.
    if(right){const int ids[4]{at,at+1,at,at+1};maskPatch(g,ids,mark);}
    if(down){const int ids[4]{at,at,at+g.cols,at+g.cols};maskPatch(g,ids,mark);}
    if(!right&&!down){const int ids[4]{at,at,at,at};maskPatch(g,ids,mark);}
}

void projectMask(const MaskBand&,std::uint32_t* output,int device=-1);
#ifdef VC_READER_CUDA
void projectMaskCuda(const MaskBand&,std::uint32_t*,int);
#endif
}
#undef VCR_MASK_HD
