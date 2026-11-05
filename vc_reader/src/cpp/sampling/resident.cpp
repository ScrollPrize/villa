#include "resident.hpp"
#include "../thread_pool.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace vcr::sampling {
namespace {
void check(const Dense& s) {
    if(s.dimensions<2||s.dimensions>3||s.channels<1||s.order<0||s.order>1)throw std::invalid_argument("dense sampling requires 2D/3D arrays and order 0 or 1");
    for(int a=0;a<s.dimensions;++a)if(s.shape[a]<1||s.shape[a]>(1<<24))throw std::invalid_argument("invalid dense array extent");
}
void checkSurface(int b,int h,int w){if(b<1||h<2||w<2||std::int64_t(h)*w>INT32_MAX/3)throw std::invalid_argument("surface grids require B>=1,H,W>=2 and bounded vertex counts");}
template<class F>void parallel(std::size_t n,F f){
    if(n<4096){for(std::size_t i=0;i<n;++i)f(i);return;}
    samplingPool().parallel_for((n+4095)/4096,[&](std::size_t block){for(auto i=block*4096;i<std::min(n,(block+1)*4096);++i)f(i);});
}
void cpuOnly(int id){if(id>=0)throw std::runtime_error("vc_reader built without CUDA sampling");}
}
void sampleDense(const Dense& s,float* out,int id){check(s);
#ifdef VC_READER_CUDA
    if(id>=0){denseCuda(s,out,id);return;}
#endif
    cpuOnly(id);parallel(s.batches*s.channels*s.queries,[&](auto i){denseForward(s,out,i);});
}
void backwardDense(const Dense& s,const float* cot,float* vg,float* pg,int id){check(s);
#ifdef VC_READER_CUDA
    if(id>=0){denseBackwardCuda(s,cot,vg,pg,id);return;}
#endif
    cpuOnly(id);if(vg)std::fill_n(vg,s.batches*s.channels*volumeSize(s),0.f);
    auto run=[&](std::size_t i){denseBackward(s,cot,vg,pg,i,[](float* p,float value){*p+=value;});};
    if(vg){for(std::size_t i=0;i<s.batches*s.queries;++i)run(i);}else parallel(s.batches*s.queries,run);
}
void prepareSurfaces(const float* xyz,const bool* valid,int b,int h,int w,float* out,int id){checkSurface(b,h,w);
#ifdef VC_READER_CUDA
    if(id>=0){prepareSurfacesCuda(xyz,valid,b,h,w,out,id);return;}
#endif
    cpuOnly(id);std::size_t n=std::size_t(h)*w;
    parallel(b*n,[&](auto i){auto d=surfaceView(xyz,valid,out,int(i/n),h,w);d.firstAt(int(i%n),const_cast<float*>(d.v.du),const_cast<float*>(d.v.dv));});
    parallel(b*n,[&](auto i){auto d=surfaceView(xyz,valid,out,int(i/n),h,w);d.mixedAt(int(i%n),const_cast<float*>(d.v.duv));});
}
void sampleSurfaces(const float* xyz,const bool* valid,const float* derivatives,int b,int h,int w,const float* rc,std::size_t q,float* p,float* row,float* col,bool* ok,int mode,bool border,int id){checkSurface(b,h,w);
    if(mode<1||mode>3||(mode==2&&!derivatives))throw std::invalid_argument("invalid surface interpolation mode or derivatives");
#ifdef VC_READER_CUDA
    if(id>=0){sampleSurfacesCuda(xyz,valid,derivatives,b,h,w,rc,q,p,row,col,ok,mode,border,id);return;}
#endif
    cpuOnly(id);parallel(b*q,[&](auto i){surfaceForward(xyz,valid,derivatives,h,w,rc,q,p,row,col,ok,mode,border,i);});
}

namespace {
std::size_t tileSize(const Tile& t) {
    if(t.rows<1||t.cols<1||t.outputRows<1||t.outputCols<1||t.rowStart<0||t.colStart<0||t.rowEnd>t.outputRows||t.colEnd>t.outputCols||t.rowEnd<=t.rowStart||t.colEnd<=t.colStart)
        throw std::invalid_argument("invalid field tile bounds");
    return std::size_t(t.rowEnd-t.rowStart)*(t.colEnd-t.colStart);
}
}
void sampleFieldTile(const float* field,const bool* valid,const Tile& t,float* out,bool* ok,int id){auto n=tileSize(t);
#ifdef VC_READER_CUDA
    if(id>=0){fieldTileCuda(field,valid,t,out,ok,id);return;}
#endif
    cpuOnly(id);parallel(n,[&](auto i){fieldTileForward(field,valid,t,out,ok,i);});
}
void sampleSurfaceTile(const float* xyz,const bool* valid,const float* derivatives,const Tile& t,int mode,float* out,bool* ok,int id){auto n=tileSize(t);checkSurface(1,t.rows,t.cols);
#ifdef VC_READER_CUDA
    if(id>=0){surfaceTileCuda(xyz,valid,derivatives,t,mode,out,ok,id);return;}
#endif
    cpuOnly(id);parallel(n,[&](auto i){surfaceTileForward(xyz,valid,derivatives,t,mode,out,ok,i);});
}
}
