#pragma once
#include "channel.cuh"
namespace vcr::augment::cuda {
template<class T, class Function> __global__ void pointwise_kernel(View<T> data, Layout g, const float* field, const Statistics* stats, Function fn) {
    EACH_INDEX(data.size) {
        auto y = fn(data.ptr[i],i,g,field,stats);
        // Wider arithmetic is only needed when float32 intermediates overflow.
        if constexpr (std::is_same_v<T,float>) {
            if (!isfinite(y)) y = T(fn(widen(data.ptr[i]),i,g,field,stats));
        }
        data.ptr[i] = T(y);
    }
}
template<class T, class Function> void pointwise(View<T> data, const Grid& grid, const ChannelPlan& c, Function fn, bool needs_stats = false, bool weighted = false) {
    auto field = field_buffer(c.field,c.device_field);
    if (needs_stats) {
        auto stats = statistics(data,false,weighted ? field->ptr : nullptr);
        pointwise_kernel<<<blocks(data.size),256,0,vcr::device::stream()>>>(data,Layout(grid),field->ptr,stats.ptr,fn);
    } else pointwise_kernel<<<blocks(data.size),256,0,vcr::device::stream()>>>(data,Layout(grid),field->ptr,nullptr,fn);
    check(cudaGetLastError());
}
} // namespace vcr::augment::cuda
