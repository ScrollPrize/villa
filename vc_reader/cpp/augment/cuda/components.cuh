#pragma once
#include "resident.cuh"
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>
namespace vcr::augment::cuda {
using ComponentIndex=unsigned long long;
// Each union links a higher root to a lower root. The canonical root is the
// first scan-order voxel, independent of thread scheduling (also the CPU seed).
__device__ inline ComponentIndex component_root(ComponentIndex* parents,ComponentIndex i){
    for(;;){const auto next=atomicAdd(parents+i,0ULL);if(next==i)return i;i=next;}
}
__device__ inline void join_components(ComponentIndex* parents,ComponentIndex a,ComponentIndex b){
    for(;;){a=component_root(parents,a);b=component_root(parents,b);if(a==b)return;
        const auto high=a>b?a:b,low=a<b?a:b;
        if(atomicCAS(parents+high,high,low)==high)return;
    }
}
template<class T,bool Classes> __global__ void initialize_components(const T* mask,std::size_t n,double threshold,ComponentIndex* parents){
    EACH_INDEX(n)parents[i]=(Classes?double(mask[i])>threshold:mask[i]!=0)?ComponentIndex(i):ComponentIndex(n);
}
template<class T,bool Classes> __global__ void connect_components(const T* mask,Layout g,bool full,ComponentIndex* parents){
    EACH_INDEX(g.size){
        if(atomicAdd(parents+i,0ULL)==g.size)continue;
        if(!full){for(int a=0;a<g.dim;++a)if(g.coord(i,a)){
            const auto j=i-g.stride[a];if(atomicAdd(parents+j,0ULL)!=g.size && (!Classes || mask[j]==mask[i]))join_components(parents,i,j);
        }}else{
            long long point[3]{};for(int a=0;a<g.dim;++a)point[a]=g.coord(i,a);
            for(int code=0;code<(g.dim==3?13:4);++code){
                int value=code;long long offset[3]{};for(int a=g.dim-1;a>=0;--a){offset[a]=value%3-1;value/=3;}
                bool valid=true;std::size_t j=0;
                for(int a=0;a<g.dim;++a){const auto p=point[a]+offset[a];if(p<0 || p>=(long long)g.shape[a]){valid=false;break;}j+=std::size_t(p)*g.stride[a];}
                if(valid && atomicAdd(parents+j,0ULL)!=g.size && (!Classes || mask[j]==mask[i]))join_components(parents,i,j);
            }
        }
    }
}
__global__ inline void flatten_components(ComponentIndex* parents,std::size_t n){
    EACH_INDEX(n)if(atomicAdd(parents+i,0ULL)!=n){const auto root=component_root(parents,i);atomicMin(parents+i,root);}
}
template<bool Classes,class T> void label_components(const T* mask,const Grid& grid,double threshold,bool full,Buffer<ComponentIndex>& parents){
    initialize_components<T,Classes><<<blocks(grid.size),256,0,vcr::device::stream()>>>(mask,grid.size,threshold,parents.ptr);
    connect_components<T,Classes><<<blocks(grid.size),256,0,vcr::device::stream()>>>(mask,Layout(grid),full,parents.ptr);
    flatten_components<<<blocks(grid.size),256,0,vcr::device::stream()>>>(parents.ptr,grid.size);check(cudaGetLastError());
}
inline void select_component_indices(const Buffer<std::uint8_t>& flags,Buffer<ComponentIndex>& selected,Buffer<ComponentIndex>& count){
    thrust::counting_iterator<ComponentIndex> indices(0);std::size_t bytes=0;
    check(cub::DeviceSelect::Flagged(nullptr,bytes,indices,flags.ptr,selected.ptr,count.ptr,flags.size,vcr::device::stream()));Buffer<std::byte> scratch(bytes);
    check(cub::DeviceSelect::Flagged(scratch.ptr,bytes,indices,flags.ptr,selected.ptr,count.ptr,flags.size,vcr::device::stream()));
}
}
