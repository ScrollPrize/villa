#include "channel.cuh"
namespace vcr::transforms::cuda {
namespace {
using namespace oriented_detail;
template<int D> struct OrientedGrid { std::size_t shape[D]; };
// Frame rows from the first row of a resident Vectors target (zyx order).
template<class T,int D> __device__ void frame_from_vectors(KernelIndex vcr_launch, const T* row, int c0, int c1, int c2, double* frame) {
    double axis[3]; axis[c0] = double(row[0]); axis[c1] = double(row[1]); if constexpr(D==3)axis[c2] = double(row[2]);
    basis<D>(axis,frame);
}
template<class T,int D> struct frame_from_vectors_launch2 {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { frame_from_vectors<T,D>(index,args...); }
};
template<class T,int D> __device__ void oriented_pass(KernelIndex vcr_launch, const T* src, T* dst, OrientedGrid<D> g, const double* frame, int row, const T* weights, int radius) {
    const std::size_t plane = g.shape[D-2]*g.shape[D-1];
    EACH_INDEX((D==3?g.shape[0]:1)*plane) {
        const std::size_t z = i/plane, y = i%plane/g.shape[D-1], x = i%g.shape[D-1];
        dst[i] = sizeof(T)==8 || wide_coordinates<D>(g.shape,radius)
            ? pass<T,double,D>(src,g.shape,z,y,x,frame+D*row,weights,radius)
            : pass<T,float,D>(src,g.shape,z,y,x,frame+D*row,weights,radius);
    }
}
template<class T,int D> struct oriented_pass_launch2 {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { oriented_pass<T,D>(index,args...); }
};
} // namespace
void apply_cuda(const OrientedBlur& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[&](const auto& t, auto data, const Grid& grid, const ChannelPlan& c) {
        using T = typename decltype(data)::value_type;
        const auto run=[&]<int D>() {
        Buffer<double> frame(D*D);
        if (!t.axis_from.empty() && session.arrays.count(t.axis_from)) {
            const auto& f = s.at(t.axis_from); const auto& v = session.arrays.at(t.axis_from);
            require(f.target.role == Role::Vectors && v.shape.size() == 2 && v.shape[0] >= 1 && v.shape[1] == D,
                    "OrientedBlur axis_from must name a matching Vectors target with at least one row");
            int component[3]{}; for (int j = 0; j < D; ++j) component[j] = axis(f.target.components[j],D);
            v.typed([&](auto row) { launch(1,1,0,vcr::device::stream(), frame_from_vectors_launch2<typename decltype(row)::value_type,D>{}, row.ptr,component[0],component[1],component[2],frame.ptr); });
        } else {
            double host[D*D]; const auto direction = oriented_detail::axis(t,s,c,t.axis_from.empty() ? nullptr : &s.at(t.axis_from).array);
            basis<D>(direction.data(),host); frame.upload(host);
        }
        OrientedGrid<D> g;std::copy_n(grid.shape.begin(),D,g.shape);
        Buffer<T> scratch(data.size); T* source = data.ptr; T* dest = scratch.ptr;
        for (int pass = 0; pass < D; ++pass) {
            const double sigma = c.values[pass ? 1 : 0];
            if (sigma <= 0) continue;
            Buffer<T> weights(kernel<T>(sigma));
            launch(blocks(data.size),256,0,vcr::device::stream(), oriented_pass_launch2<T,D>{}, source,dest,g,frame.ptr,pass,weights.ptr,int(weights.size/2));
            check(cudaGetLastError()); std::swap(source,dest);
        }
        if (source != data.ptr) copy_device(data,View<T>{source,data.size});
        };
        if(grid.shape.size()==3)run.template operator()<3>();else run.template operator()<2>();
    });
}
} // namespace vcr::transforms::cuda
