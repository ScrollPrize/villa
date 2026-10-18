#pragma once
#include "voxel_transform.cuh"
namespace vcr::augment::cuda {
// Integer counts permit parallel accumulation without changing histogram/CDF
// values. The short CDF/clip pass retains the CPU's bin order and arithmetic.
template<class T> __global__ void global_histogram_counts(const T* data,std::size_t n,const vd::Stats* summary,int bins,unsigned long long* partial) {
    extern __shared__ unsigned long long counts[];
    for(int b=threadIdx.x;b<bins;b+=blockDim.x)counts[b]=0;__syncthreads();
    const auto s=*summary;
    EACH_INDEX(n) atomicAdd(counts+vd::bin(data[i],s.low,s.high,bins),1ULL);
    __syncthreads();
    for(int b=threadIdx.x;b<bins;b+=blockDim.x)partial[std::size_t(blockIdx.x)*bins+b]=counts[b];
}
__global__ inline void finish_histogram_counts(const unsigned long long* partial,unsigned groups,int bins,double* table) {
    EACH_INDEX(std::size_t(bins)){unsigned long long total=0;for(unsigned j=0;j<groups;++j)total+=partial[std::size_t(j)*bins+i];table[i]=double(total);}
}
__global__ inline void histogram_cdf(double* table,int bins,double clip) {vd::cdf(table,bins,clip);}
template<class T> void build_global_histogram(View<T> data,const vd::Stats* summary,int bins,double* table,double clip=0) {
    const auto groups=std::min(blocks(data.size),1024u);Buffer<unsigned long long> partial(std::size_t(groups)*bins);
    global_histogram_counts<<<groups,256,std::size_t(bins)*sizeof(unsigned long long),vcr::device::stream()>>>(data.ptr,data.size,summary,bins,partial.ptr);
    finish_histogram_counts<<<blocks(bins),256,0,vcr::device::stream()>>>(partial.ptr,groups,bins,table);
    histogram_cdf<<<1,1,0,vcr::device::stream()>>>(table,bins,clip);check(cudaGetLastError());
}
template<class T,int D> __global__ void histogram_tiles(const T* data,vd::GridND<D> grid,const vd::Stats* summary,vd::HistogramGrid layout,double* table) {
    extern __shared__ unsigned long long counts[];
    for(std::size_t tile=blockIdx.x;tile<layout.count;tile+=gridDim.x) {
        for(int b=threadIdx.x;b<layout.bins;b+=blockDim.x)counts[b]=0;__syncthreads();
        std::size_t pos[3];
        if constexpr(D==3){pos[0]=tile/(layout.tiles[1]*layout.tiles[2]);pos[1]=tile/layout.tiles[2]%layout.tiles[1];pos[2]=tile%layout.tiles[2];}
        else {pos[0]=tile/layout.tiles[1];pos[1]=tile%layout.tiles[1];}
        std::size_t lo[3],extent[3],n=1;
        for(int a=0;a<D;++a){lo[a]=pos[a]*grid.n[a]/layout.tiles[a];extent[a]=(pos[a]+1)*grid.n[a]/layout.tiles[a]-lo[a];n*=extent[a];}
        const auto s=*summary;
        for(std::size_t i=threadIdx.x;i<n;i+=blockDim.x){
            const auto x=lo[D-1]+i%extent[D-1],y=lo[D-2]+i/extent[D-1]%extent[D-2];
            const auto z=D==3?lo[0]+i/(extent[1]*extent[2]):0;
            atomicAdd(counts+vd::bin(data[(D==3?z*grid.stride[0]:0)+y*grid.stride[D-2]+x],s.low,s.high,layout.bins),1ULL);
        }
        __syncthreads();double* histogram=table+tile*layout.bins;
        for(int b=threadIdx.x;b<layout.bins;b+=blockDim.x)histogram[b]=double(counts[b]);__syncthreads();
        if(!threadIdx.x)vd::cdf(histogram,layout.bins,layout.clip_limit);__syncthreads();
    }
}
template<class T> void build_histograms(View<T> data,const Grid& grid,const vd::Stats* summary,const vd::HistogramGrid& layout,double* table) {
    if(layout.count==1){build_global_histogram(data,summary,layout.bins,table,layout.clip_limit);return;}
    vd::with_grid(grid.shape,[&](const auto& g){
    histogram_tiles<<<unsigned(std::min<std::size_t>(layout.count,65535)),256,std::size_t(layout.bins)*sizeof(unsigned long long),vcr::device::stream()>>>(data.ptr,g,summary,layout,table);check(cudaGetLastError());
    });
}
}
