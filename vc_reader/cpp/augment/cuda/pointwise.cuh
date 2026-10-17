#pragma once
#include "channel.cuh"
namespace vcr::augment::cuda {
template<class T, class Function> __global__ void pointwise_kernel(View<T> data, Layout g, const double* field, const Statistics* stats, Function fn) {
    EACH_INDEX(data.size) data.ptr[i] = T(fn(data.ptr[i],i,g,field,stats));
}
template<class T, class Function> void pointwise(View<T> data, const Grid& grid, const ChannelPlan& c, Function fn, bool needs_stats = false, bool weighted = false) {
    auto field = field_buffer(c.field,c.device_field);
    if (needs_stats) {
        auto stats = statistics(data,false,weighted ? field->ptr : nullptr);
        pointwise_kernel<<<blocks(data.size),256>>>(data,Layout(grid),field->ptr,stats.ptr,fn);
    } else pointwise_kernel<<<blocks(data.size),256>>>(data,Layout(grid),field->ptr,nullptr,fn);
    check(cudaGetLastError());
}
} // namespace vcr::augment::cuda
