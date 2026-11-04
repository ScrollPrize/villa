#include "resident.hpp"
#include "../cuda_buffer.hpp"
namespace vcr::sampling {
__global__ void forwardKernel(Dense s,float* out){auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<s.batches*s.channels*s.queries)denseForward(s,out,i);}
struct Add {__device__ void operator()(float* p,float value) const {atomicAdd(p,value);}};
__global__ void backwardKernel(Dense s,const float* cot,float* vg,float* pg){auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<s.batches*s.queries)denseBackward(s,cot,vg,pg,i,Add{});}
__global__ void derivativesKernel(const float* xyz,const bool* valid,int b,int h,int w,float* out,bool mixed){
 auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x,n=std::size_t(h)*w;if(i>=b*n)return;
 auto d=surfaceView(xyz,valid,out,int(i/n),h,w);
 if(mixed)d.mixedAt(int(i%n),const_cast<float*>(d.v.duv));else d.firstAt(int(i%n),const_cast<float*>(d.v.du),const_cast<float*>(d.v.dv));
}
__global__ void surfaceKernel(const float* xyz,const bool* valid,const float* derivatives,int b,int h,int w,const float* rc,std::size_t q,float* p,float* row,float* col,bool* ok,int mode,bool border){auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<b*q)surfaceForward(xyz,valid,derivatives,h,w,rc,q,p,row,col,ok,mode,border,i);}
void denseCuda(const Dense& s,float* out,int id){device::Guard guard(id);auto n=s.batches*s.channels*s.queries;if(n)forwardKernel<<<(n+255)/256,256,0,device::stream()>>>(s,out);device::check(cudaGetLastError());device::complete();}
void denseBackwardCuda(const Dense& s,const float* cot,float* vg,float* pg,int id){device::Guard guard(id);if(vg)device::check(cudaMemsetAsync(vg,0,s.batches*s.channels*volumeSize(s)*4,device::stream()));auto n=s.batches*s.queries;if(n)backwardKernel<<<(n+255)/256,256,0,device::stream()>>>(s,cot,vg,pg);device::check(cudaGetLastError());device::complete();}
void prepareSurfacesCuda(const float* xyz,const bool* valid,int b,int h,int w,float* out,int id){device::Guard guard(id);auto n=std::size_t(b)*h*w;
 derivativesKernel<<<(n+255)/256,256,0,device::stream()>>>(xyz,valid,b,h,w,out,false);
 derivativesKernel<<<(n+255)/256,256,0,device::stream()>>>(xyz,valid,b,h,w,out,true);device::check(cudaGetLastError());device::complete();}
void sampleSurfacesCuda(const float* xyz,const bool* valid,const float* derivatives,int b,int h,int w,const float* rc,std::size_t q,float* p,float* row,float* col,bool* ok,int mode,bool border,int id){device::Guard guard(id);auto n=b*q;if(n)surfaceKernel<<<(n+255)/256,256,0,device::stream()>>>(xyz,valid,derivatives,b,h,w,rc,q,p,row,col,ok,mode,border);device::check(cudaGetLastError());device::complete();}

__global__ void fieldTileKernel(const float* field,const bool* valid,Tile t,float* out,bool* ok){auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<std::size_t(t.rowEnd-t.rowStart)*(t.colEnd-t.colStart))fieldTileForward(field,valid,t,out,ok,i);}
__global__ void surfaceTileKernel(const float* xyz,const bool* valid,const float* derivatives,Tile t,int mode,float* out,bool* ok){auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<std::size_t(t.rowEnd-t.rowStart)*(t.colEnd-t.colStart))surfaceTileForward(xyz,valid,derivatives,t,mode,out,ok,i);}
void fieldTileCuda(const float* field,const bool* valid,const Tile& t,float* out,bool* ok,int id){device::Guard guard(id);auto n=std::size_t(t.rowEnd-t.rowStart)*(t.colEnd-t.colStart);fieldTileKernel<<<(n+255)/256,256,0,device::stream()>>>(field,valid,t,out,ok);device::check(cudaGetLastError());device::complete();}
void surfaceTileCuda(const float* xyz,const bool* valid,const float* derivatives,const Tile& t,int mode,float* out,bool* ok,int id){device::Guard guard(id);auto n=std::size_t(t.rowEnd-t.rowStart)*(t.colEnd-t.colStart);surfaceTileKernel<<<(n+255)/256,256,0,device::stream()>>>(xyz,valid,derivatives,t,mode,out,ok);device::check(cudaGetLastError());device::complete();}
}
