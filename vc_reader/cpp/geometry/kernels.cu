#include "plan.hpp"
#include <cub/device/device_radix_sort.cuh>
namespace vcr::geometry {
__global__ void gather_kernel(CurvesView c,const std::int64_t* ids,bool segments,float* out,float* arcs,std::int64_t* owners,bool* valid,std::size_t n){
    auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;int width=segments?2:1;
    if(i<n)gather(c,ids[i],segments,out+3*width*i,arcs+width*i,owners+i,valid+i);
}
__global__ void interpolation_kernel(CurvesView c,const std::int64_t* ids,const float* s,float* out,bool* valid,std::size_t n){
    auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)interpolate(c,ids[i],s[i],out+3*i,valid+i);
}
__global__ void intersection_kernel(CurvesView c,const std::int64_t* ids,const float* planes,const float* intervals,int direction,
    float* out,float* arcs,std::int64_t* indices,bool* valid,std::size_t n){
    auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)intersect(c,ids[i],planes+4*i,intervals?intervals+2*i:nullptr,direction,out+3*i,arcs+i,indices+i,valid+i);
}
__global__ void closest_kernel(CurvesView c,const float* p,const std::int64_t* ids,float limit,float* out,float* distance,float* arcs,
    std::int64_t* indices,std::int64_t* owners,bool* valid,std::size_t n){
    auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)closest(c,vector(p+3*i),ids?ids[i]:-1,limit,out+3*i,distance+i,arcs+i,indices+i,owners+i,valid+i);
}
__global__ void transform_kernel(const float* p,const float* matrix,std::size_t matrices,bool vectors,float* out,std::size_t n){
    auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n){auto m=matrix+(matrices==1?0:12*i);auto v=vector(p+3*i);store(out+3*i,vectors?linear(m,v):transform(m,v));}
}
__global__ void keys_kernel(const float* p,const std::int64_t* batch,std::int64_t* keys,VoxelGrid grid,std::size_t n){
    auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)keys[i]=voxel_key(vector(p+3*i),batch?batch[i]:0,grid);
}
__global__ void boxes_kernel(CurvesView c,const float* lower,const float* upper,const std::int64_t* ids,int capacity,std::int64_t* out,std::int64_t* counts,bool* valid,std::size_t n){
    auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n)query_box(c,vector(lower+3*i),vector(upper+3*i),ids?ids[i]:-1,capacity,out+i*capacity,counts+i,valid+i);
}
__global__ void contains_kernel(const std::int64_t* keys,std::size_t count,VoxelGrid grid,const float* p,const std::int64_t* batch,bool* out,std::size_t n){
    auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)out[i]=contains(keys,count,voxel_key(vector(p+3*i),batch?batch[i]:0,grid));
}
__global__ void dense_kernel(const std::int64_t* keys,std::size_t n,std::uint8_t* out){
    auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    // Exactly one writer per occupied cell, even when construction retained duplicates.
    if(i<n&&keys[i]!=missing_key&&(i==0||keys[i]!=keys[i-1]))out[keys[i]]=1;
}
inline void finish(){device::check(cudaGetLastError());device::complete();}
void gather_cuda(CurvesView c,const std::int64_t* ids,bool segments,float* out,float* arcs,std::int64_t* owners,bool* valid,std::size_t n){
    if(n)gather_kernel<<<(n+255)/256,256,0,device::stream()>>>(c,ids,segments,out,arcs,owners,valid,n);finish();
}
void interpolate_cuda(CurvesView c,const std::int64_t* ids,const float* s,float* out,bool* valid,std::size_t n){
    if(n)interpolation_kernel<<<(n+127)/128,128,0,device::stream()>>>(c,ids,s,out,valid,n);finish();
}
void intersect_cuda(CurvesView c,const std::int64_t* ids,const float* p,const float* intervals,int direction,float* out,float* arcs,std::int64_t* indices,bool* valid,std::size_t n){
    if(n)intersection_kernel<<<(n+127)/128,128,0,device::stream()>>>(c,ids,p,intervals,direction,out,arcs,indices,valid,n);finish();
}
void closest_cuda(CurvesView c,const float* p,const std::int64_t* ids,float limit,float* out,float* d,float* arcs,std::int64_t* indices,std::int64_t* owners,bool* valid,std::size_t n){
    if(n)closest_kernel<<<(n+127)/128,128,0,device::stream()>>>(c,p,ids,limit,out,d,arcs,indices,owners,valid,n);finish();
}
void transform_cuda(const float* p,const float* m,std::size_t matrices,bool vectors,float* out,std::size_t n){
    if(n)transform_kernel<<<(n+255)/256,256,0,device::stream()>>>(p,m,matrices,vectors,out,n);finish();
}
void boxes_cuda(CurvesView c,const float* lo,const float* hi,const std::int64_t* ids,int capacity,std::int64_t* out,std::int64_t* counts,bool* valid,std::size_t n){
    if(n)boxes_kernel<<<(n+127)/128,128,0,device::stream()>>>(c,lo,hi,ids,capacity,out,counts,valid,n);finish();
}
CudaVoxels voxelize_cuda(const float* p,const std::int64_t* batch,std::size_t n,VoxelGrid grid,int id){
    CudaVoxels result{grid,device::allocate(n*8,id),n,id};if(!n)return result;
    auto input=device::allocate(n*8,id);keys_kernel<<<(n+255)/256,256,0,device::stream()>>>(p,batch,static_cast<std::int64_t*>(input->ptr),grid,n);
    std::size_t bytes=0;
    device::check(cub::DeviceRadixSort::SortKeys(nullptr,bytes,static_cast<const std::int64_t*>(input->ptr),static_cast<std::int64_t*>(result.keys->ptr),n,0,64,device::stream()));
    auto temp=device::allocate(bytes,id);
    device::check(cub::DeviceRadixSort::SortKeys(temp->ptr,bytes,static_cast<const std::int64_t*>(input->ptr),static_cast<std::int64_t*>(result.keys->ptr),n,0,64,device::stream()));
    finish();return result;
}
void contains_cuda(const CudaVoxels& v,const float* p,const std::int64_t* batch,bool* out,std::size_t n){
    if(n)contains_kernel<<<(n+255)/256,256,0,device::stream()>>>(static_cast<const std::int64_t*>(v.keys->ptr),v.count,v.grid,p,batch,out,n);finish();
}
__global__ void copy_keys_kernel(const std::int64_t* source,std::size_t count,std::int64_t* out,std::size_t n){
    auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n)out[i]=i<count?source[i]:missing_key;
}
void keys_cuda(const CudaVoxels& v,std::int64_t* out,std::size_t n){
    if(n)copy_keys_kernel<<<(n+255)/256,256,0,device::stream()>>>(static_cast<const std::int64_t*>(v.keys->ptr),v.count,out,n);
    finish();
}
void dense_cuda(const CudaVoxels& v,std::uint8_t* out){
    device::check(cudaMemsetAsync(out,0,std::size_t(v.grid.batches)*v.grid.volume,device::stream()));
    if(v.count)dense_kernel<<<(v.count+255)/256,256,0,device::stream()>>>(static_cast<const std::int64_t*>(v.keys->ptr),v.count,out);finish();
}
}
