#pragma once
#include "channel.cuh"
namespace vcr::augment::cuda {
template<class T, class Function, class Field> __device__ void pointwise_kernel(KernelIndex vcr_launch, View<T> data, Layout g, Field field, const Statistics* stats, Function fn) {
    EACH_INDEX(data.size) {
        auto y = fn(data.ptr[i],i,g,field,stats);
        // Wider arithmetic is only needed when float32 intermediates overflow.
        if constexpr (std::is_same_v<T,float>) {
            if (!isfinite(y)) y = T(fn(widen(data.ptr[i]),i,g,field,stats));
        }
        data.ptr[i] = T(y);
    }
}
struct pointwise_kernel_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { pointwise_kernel(index,args...); }
};
template<class T, class Function> void pointwise(View<T> data, const Grid& grid, const ChannelPlan& c, Function fn, bool needs_stats = false, bool weighted = false) {
    auto field = field_buffer(c.field,c.device_field);
    if (needs_stats) {
        auto stats = statistics(data,false,weighted ? field->ptr : nullptr);
        launch(blocks(data.size),256,0,vcr::device::stream(), pointwise_kernel_launch{}, data,Layout(grid),field->ptr,stats.ptr,fn);
    } else launch(blocks(data.size),256,0,vcr::device::stream(), pointwise_kernel_launch{}, data,Layout(grid),field->ptr,nullptr,fn);
    check(cudaGetLastError());
}
template<class T, class Function> void local_pointwise(View<T> data, const Grid& grid, const ChannelPlan& c, Function fn, bool needs_stats = false, bool weighted = false) {
    auto buffer=field_buffer(c.field,c.device_field);
    const LocalFieldView field(c.field,grid,buffer->ptr);
    if (needs_stats) {
        auto stats=weighted ? local_statistics(data,field) : bounds(data);
        launch(blocks(data.size),256,0,vcr::device::stream(), pointwise_kernel_launch{}, data,Layout(grid),field,stats.ptr,fn);
    } else launch(blocks(data.size),256,0,vcr::device::stream(), pointwise_kernel_launch{}, data,Layout(grid),field,nullptr,fn);
    check(cudaGetLastError());
}
} // namespace vcr::augment::cuda
