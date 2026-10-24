#include "plan.hpp"
#include "../cuda_buffer.hpp"
namespace vcr::surface {
template<class T> __global__ void project(Raster g,const float* triangles,const std::int64_t* ids,const std::int64_t* offsets,
    const int* candidates,const T* label,T* out,bool* coverage,bool* conflict,float* distance,std::int64_t* winner,float* uv,bool* geometry){
    for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<g.count;i+=std::size_t(blockDim.x)*gridDim.x)
        raster(g,triangles,ids,offsets,candidates,label,out,coverage,conflict,distance,winner,uv,geometry,i);
}
void launch(const Plan& p,Raster g,const void* label,ElementType type,void* out,bool* coverage,bool* conflict,
            float* distance,std::int64_t* winner,float* uv,bool* geometry,int id){
    auto upload=[&](const auto& values){auto a=device::allocate(values.size()*sizeof(values[0]),id);
        if(a->bytes)device::check(cudaMemcpyAsync(a->ptr,values.data(),a->bytes,cudaMemcpyHostToDevice,device::stream()));return a;};
    auto triangles=upload(p.triangles),ids=upload(p.ids),offsets=upload(p.offsets),candidates=upload(p.candidates);
    type.visit_load([&](auto load){using T=decltype(load(static_cast<const std::uint8_t*>(nullptr)));
        project<<<unsigned((g.count+127)/128),128,0,device::stream()>>>(g,static_cast<float*>(triangles->ptr),static_cast<std::int64_t*>(ids->ptr),
            static_cast<std::int64_t*>(offsets->ptr),static_cast<int*>(candidates->ptr),static_cast<const T*>(label),static_cast<T*>(out),coverage,conflict,distance,winner,uv,geometry);
    });device::check(cudaGetLastError());device::complete();
}
__global__ void interpolations(Interpolation g,const float* xyz,const bool* mask,const float* queries,const float* normals,
                              float* out,bool* valid,std::size_t n){
    for(std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=std::size_t(blockDim.x)*gridDim.x)interpolate(g,xyz,mask,queries,normals,out,valid,i);
}
void interpolate_cuda(Interpolation g,const float* xyz,const bool* mask,const float* queries,const float* normals,
                      float* out,bool* valid,std::size_t n,int id){
    if(!n)return;device::Guard guard(id);
    auto upload=[&](const void* data,std::size_t size){auto a=device::allocate(size,id);if(size)device::check(cudaMemcpyAsync(a->ptr,data,size,cudaMemcpyHostToDevice,device::stream()));return a;};
    auto x=upload(xyz,std::size_t(g.height)*g.width*12),m=upload(mask,std::size_t(g.height)*g.width),q=upload(queries,n*8);
    auto norm=upload(normals,normals?x->bytes:0),o=device::allocate(n*12,id),v=device::allocate(n,id);
    interpolations<<<unsigned((n+255)/256),256,0,device::stream()>>>(g,static_cast<float*>(x->ptr),static_cast<bool*>(m->ptr),static_cast<float*>(q->ptr),
        normals?static_cast<float*>(norm->ptr):nullptr,static_cast<float*>(o->ptr),static_cast<bool*>(v->ptr),n);
    device::check(cudaGetLastError());device::check(cudaMemcpyAsync(out,o->ptr,o->bytes,cudaMemcpyDeviceToHost,device::stream()));
    device::check(cudaMemcpyAsync(valid,v->ptr,v->bytes,cudaMemcpyDeviceToHost,device::stream()));device::complete();
}
void copy_cuda(const void* src,void* dst,std::size_t rows,std::size_t sw,std::size_t dw,std::size_t offset,std::size_t item,int){
    if(rows&&sw)device::check(cudaMemcpy2DAsync(static_cast<char*>(dst)+offset*item,dw*item,src,sw*item,sw*item,rows,cudaMemcpyDeviceToDevice,device::stream()));device::complete();
}
void copy_region_cuda(const void* src,void* dst,aligned::Triple shape,aligned::Triple output,aligned::Triple offset,int channels,std::size_t item){
    for(int c=0;c<channels;++c)for(std::int64_t z=0;z<shape[0];++z){
        auto from=(std::size_t(c)*shape[0]+z)*shape[1]*shape[2]*item;
        auto to=(((std::size_t(c)*output[0]+z+offset[0])*output[1]+offset[1])*output[2]+offset[2])*item;
        if(shape[1]&&shape[2])device::check(cudaMemcpy2DAsync(static_cast<char*>(dst)+to,output[2]*item,
            static_cast<const char*>(src)+from,shape[2]*item,shape[2]*item,shape[1],cudaMemcpyDeviceToDevice,device::stream()));
    }
    device::complete();
}
}
