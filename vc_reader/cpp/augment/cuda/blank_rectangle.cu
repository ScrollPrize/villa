#include "channel.cuh"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
namespace vcr::augment::cuda {
__device__ std::size_t box_index(std::size_t i, Layout image, Layout box, std::size_t origin) {
    for (int a = 0; a < image.dim; ++a) origin += box.coord(i,a)*image.stride[a];
    return origin;
}
template<class T> __global__ void gather_box(View<T> input, double* values, Layout image, Layout box, std::size_t origin) {
    EACH_INDEX(box.size) values[i] = input.ptr[box_index(i,image,box,origin)];
}
template<class T> __global__ void fill_box(View<T> data, Layout image, Layout box, std::size_t origin, double value, const Statistics* stats, const double* median) {
    if (stats) value = stats->mean;
    if (median) value = *median;
    EACH_INDEX(box.size) data.ptr[box_index(i,image,box,origin)] = T(value);
}
void apply_cuda(const BlankRectangle& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto& t, auto v, const Grid& grid, const ChannelPlan& c) {
        Layout g(grid);
        for (std::size_t b = 0; b < c.values.size(); b += 2*g.dim+1) {
            RectangleRegion region(grid,c,b); Layout box{Grid(region.extent)};
            if (t.statistic.empty()) {
                fill_box<<<blocks(box.size),256>>>(v,g,box,region.origin,c.values[b+2*g.dim],nullptr,nullptr);
            } else {
                Buffer<double> values(box.size);
                gather_box<<<blocks(box.size),256>>>(v,values.ptr,g,box,region.origin);
                if (t.statistic == "mean") {
                    auto stats = statistics(View<double>{values.ptr,box.size});
                    fill_box<<<blocks(box.size),256>>>(v,g,box,region.origin,0,stats.ptr,nullptr);
                } else {
                    thrust::sort(thrust::cuda::par.on(0),values.ptr,values.ptr+box.size);
                    fill_box<<<blocks(box.size),256>>>(v,g,box,region.origin,0,nullptr,values.ptr+(box.size-1)/2);
                }
            }
            check(cudaGetLastError());
        }
    });
}
} // namespace vcr::augment::cuda
