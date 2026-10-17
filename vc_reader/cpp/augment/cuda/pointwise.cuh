#pragma once
#include "channel.cuh"
namespace vcr::augment::cuda {
template<class T, class Function> __global__ void pointwise_kernel(T* data, Layout g, const double* field, Function fn) {
    EACH_INDEX(g.size) data[i] = T(fn(data[i],i,g,field));
}
template<class T, class Function> void pointwise(std::vector<T>& v, const Grid& grid, const ChannelPlan& c, Function fn) {
    Buffer<T> data(v); Buffer<double> field(c.field); Layout g(grid);
    pointwise_kernel<<<blocks(g.size),256>>>(data.ptr,g,field.ptr,fn);
    check(cudaGetLastError()); data.download(v.data());
}

} // namespace vcr::augment::cuda
