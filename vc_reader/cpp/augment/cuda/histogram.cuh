#pragma once
#include "voxel_transform.cuh"
namespace vcr::augment::cuda {
__global__ inline void histogram_tiles(const double* data,vd::Grid3 grid,const vd::Stats* summary,vd::HistogramGrid layout,double* table){
    EACH_INDEX(layout.count)vd::histogram_tile(data,grid,*summary,layout,i,table);
}
inline void build_histograms(View<double> data,const Grid& grid,const vd::Stats* summary,const vd::HistogramGrid& layout,double* table){
    histogram_tiles<<<blocks(layout.count),256,0,vcr::device::stream()>>>(data.ptr,vd::Grid3(grid.shape),summary,layout,table);check(cudaGetLastError());
}
__global__ inline void reference_histogram(const double* ref,std::size_t size,vd::Stats* summary,double* table,int bins){
    if(threadIdx.x || blockIdx.x)return;*summary=vd::stats(ref,size);
    if(table){vd::histogram(ref,size,table,bins,summary->low,summary->high);vd::cdf(table,bins);}
}
} // namespace vcr::augment::cuda
