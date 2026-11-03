#include "geometry.cuh"
namespace vcr::transforms::cuda {
template<class T, class Index, int D>
__device__ void resolution_kernel(KernelIndex vcr_launch, const T* src, T* dst, Layout g,
                                  const ResizePosition<T,Index>* positions) {
    EACH_INDEX(g.size) {
        ResizePosition<T,Index> q[D];
        Index remaining=Index(i), offset=0;
        for (int a=0; a<D; ++a) {
            const Index x=remaining/Index(g.stride[a]);
            remaining%=Index(g.stride[a]);
            q[a]=positions[offset+x]; offset+=Index(g.shape[a]);
        }
        dst[i]=resize_value<D>(src,q);
    }
}
template<class T, class Index, int D> struct resolution_kernel_launch3 {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { resolution_kernel<T,Index,D>(index,args...); }
};
template<class T, class Index>
DeviceArray resolution_channel(View<T> data, const Grid& grid, const Shape& reduced) {
    std::vector<ResizePosition<T,Index>> positions;
    for (std::size_t a=0; a<reduced.size(); ++a) {
        auto axis=resize_positions<T,Index>(grid.shape[a],grid.shape[a],grid.stride[a],false,false,reduced[a]);
        positions.insert(positions.end(),axis.begin(),axis.end());
    }
    Buffer<ResizePosition<T,Index>> table(positions);
    Shape shape{1}; shape.insert(shape.end(),grid.shape.begin(),grid.shape.end());
    DeviceArray output(shape,'f',sizeof(T));
    if (reduced.size()==3)
        launch(blocks(data.size),256,0,vcr::device::stream(), resolution_kernel_launch3<T,Index,3>{}, data.ptr,output.view<T>().ptr,Layout(grid),table.ptr);
    else
        launch(blocks(data.size),256,0,vcr::device::stream(), resolution_kernel_launch3<T,Index,2>{}, data.ptr,output.view<T>().ptr,Layout(grid),table.ptr);
    check(cudaGetLastError());
    return output;
}
void apply_cuda(const LowResolution&, Sample& s, const Parameters& p, Session& session) {
    for (const auto& c : p.channels) {
        auto& a = session.arrays.at(c.key); Shape shape = a.spatial(); Grid grid(shape);
        for (std::size_t j = 0; j < shape.size(); ++j) shape[j] = std::max<std::size_t>(1,std::size_t(round_even(shape[j]*c.values[j])));
        if (shape==grid.shape) continue;
        const auto run = [&]<class T>() {
            // The table offset is a sum of extents, distinct from voxel count.
            const auto table_size=std::accumulate(grid.shape.begin(),grid.shape.end(),std::size_t(0));
            auto output=grid.size<=UINT32_MAX && table_size<=UINT32_MAX
                ? resolution_channel<T,std::uint32_t>(a.channel<T>(c.channel),grid,shape)
                : resolution_channel<T,std::size_t>(a.channel<T>(c.channel),grid,shape);
            if (a.shape[0]==1) a=std::move(output);
            else copy_device(a.channel<T>(c.channel),output.template view<T>());
        };
        if (a.bytes==4) run.template operator()<float>(); else run.template operator()<double>();
    }
    session.restore_valid(s);
}
} // namespace vcr::transforms::cuda
