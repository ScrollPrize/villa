#pragma once
#include "common.hpp"
#include "../aligned/plan.hpp"
namespace vcr::surface {
struct Plan {
    Raster grid;
    std::vector<float> triangles;
    std::vector<std::int64_t> ids,offsets;
    std::vector<int> candidates;
    Plan(const float* t,const std::int64_t* identifiers,std::size_t n,aligned::Triple shape,
         std::array<float,12> metric,float radius,std::array<float,3> expansion,std::array<float,3> origin,std::size_t budget){
        if(n>INT32_MAX||!std::isfinite(radius)||radius<0)throw std::invalid_argument("invalid triangle count/thickness");
        grid.count=1;grid.radius=radius;
        for(int a=0;a<3;++a){
            if(shape[a]<1||shape[a]>256||!std::isfinite(expansion[a])||expansion[a]<0)throw std::invalid_argument("invalid raster block");
            grid.shape[a]=int(shape[a]);grid.count*=shape[a];
        }
        if(grid.count>32768)throw std::invalid_argument("raster blocks are limited to 32768 voxels");
        std::copy(metric.begin(),metric.end(),grid.metric);
        std::copy(origin.begin(),origin.end(),grid.origin);
        for(float x:metric)if(!std::isfinite(x))throw std::invalid_argument("nonfinite physical metric");
        for(float x:origin)if(!std::isfinite(x))throw std::invalid_argument("nonfinite raster origin");
        const auto overhead=(grid.count+1)*16+n*(15*4+8);
        if(overhead>budget)throw std::length_error("raster candidate batch exceeds sampling_work_bytes");
        triangles.assign(t,t+n*15);ids.assign(identifiers,identifiers+n);
        offsets.resize(grid.count+1);
        auto visit=[&](auto action){
            for(std::size_t j=0;j<n;++j){
                int lo[3],hi[3];
                for(int a=0;a<3;++a){
                    float low=INFINITY,high=-INFINITY;
                    for(int k=0;k<3;++k){float v=t[j*15+k*5+2-a];if(!std::isfinite(v))throw std::invalid_argument("nonfinite triangle");low=std::min(low,v);high=std::max(high,v);}
                    low=std::nextafter(low-origin[2-a]-.5f-expansion[2-a],-INFINITY);
                    high=std::nextafter(high-origin[2-a]+.5f+expansion[2-a],INFINITY);
                    lo[a]=int(std::max(0.f,std::min(float(shape[a]),ceilf(low))));
                    hi[a]=int(std::max(-1.f,std::min(float(shape[a]-1),floorf(high))));
                }
                for(int z=lo[0];z<=hi[0];++z)for(int y=lo[1];y<=hi[1];++y)for(int x=lo[2];x<=hi[2];++x)
                    action((std::size_t(z)*shape[1]+y)*shape[2]+x,j);
            }
        };
        visit([&](std::size_t i,std::size_t){++offsets[i+1];});
        for(std::size_t i=0;i<grid.count;++i)offsets[i+1]+=offsets[i];
        if(std::size_t(offsets.back())>(budget-overhead)/4)throw std::length_error("raster candidate batch exceeds sampling_work_bytes");
        candidates.resize(offsets.back());auto next=offsets;
        visit([&](std::size_t i,std::size_t j){candidates[next[i]++]=int(j);});
    }
};
#ifdef VC_READER_CUDA
void launch(const Plan&,Raster,const void*,ElementType,void*,bool*,bool*,float*,std::int64_t*,float*,bool*,int);
void interpolate_cuda(Interpolation,const float*,const bool*,const float*,const float*,float*,bool*,std::size_t,int);
void copy_cuda(const void*,void*,std::size_t,std::size_t,std::size_t,std::size_t,std::size_t,int);
void copy_region_cuda(const void*,void*,aligned::Triple,aligned::Triple,aligned::Triple,int,std::size_t);
#endif
}
