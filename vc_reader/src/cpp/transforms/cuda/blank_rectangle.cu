#include "channel.cuh"
#include "sort_keys.cuh"
namespace vcr::transforms::cuda {
__device__ std::size_t box_index(std::size_t i, Layout image, Layout box, std::size_t origin) {
    for (int a = 0; a < image.dim; ++a) origin += box.coord(i,a)*image.stride[a];
    return origin;
}
template<class T> __device__ void gather_box(KernelIndex vcr_launch, View<T> input, T* values, Layout image, Layout box, std::size_t origin) {
    EACH_INDEX(box.size) values[i] = input.ptr[box_index(i,image,box,origin)];
}
struct gather_box_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { gather_box(index,args...); }
};
template<class T> __device__ void fill_box(KernelIndex vcr_launch, View<T> data, Layout image, Layout box, std::size_t origin, double value, const Statistics* stats, const T* median) {
    if (stats) value = stats->mean;
    if (median) value = *median;
    EACH_INDEX(box.size) data.ptr[box_index(i,image,box,origin)] = T(value);
}
template<class T> struct fill_box_launch1 {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { fill_box<T>(index,args...); }
};
void apply_cuda(const BlankRectangle& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto& t, auto v, const Grid& grid, const ChannelPlan& c) {
        using V = typename decltype(v)::value_type;
        Layout g(grid);
        for (std::size_t b = 0; b < c.values.size(); b += 2*g.dim+1) {
            RectangleRegion region(grid,c,b); Layout box{Grid(region.extent)};
            if (t.statistic.empty()) {
                launch(blocks(box.size),256,0,vcr::device::stream(), fill_box_launch1<V>{}, v,g,box,region.origin,c.values[b+2*g.dim],nullptr,nullptr);
            } else {
                Buffer<V> values(box.size);
                launch(blocks(box.size),256,0,vcr::device::stream(), gather_box_launch{}, v,values.ptr,g,box,region.origin);
                if (t.statistic == "mean") {
                    auto stats = statistics(View<V>{values.ptr,box.size});
                    launch(blocks(box.size),256,0,vcr::device::stream(), fill_box_launch1<V>{}, v,g,box,region.origin,0,stats.ptr,nullptr);
                } else {
                    sort_keys(values,box.size);
                    launch(blocks(box.size),256,0,vcr::device::stream(), fill_box_launch1<V>{}, v,g,box,region.origin,0,nullptr,values.ptr+(box.size-1)/2);
                }
            }
            check(cudaGetLastError());
        }
    });
}
} // namespace vcr::transforms::cuda
