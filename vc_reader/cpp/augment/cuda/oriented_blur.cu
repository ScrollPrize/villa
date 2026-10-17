#include "channel.cuh"
namespace vcr::augment::cuda {
namespace {
using namespace oriented_detail;
struct Grid3 { std::size_t shape[3]; };
// Frame rows from the first row of a resident Vectors target (zyx order).
template<class T> __global__ void frame_from_vectors(const T* row, int c0, int c1, int c2, double* frame) {
    double axis[3]; axis[c0] = double(row[0]); axis[c1] = double(row[1]); axis[c2] = double(row[2]);
    basis(axis,frame);
}
template<class T> __global__ void oriented_pass(const T* src, T* dst, Grid3 g, const double* frame, int row, const double* weights, int radius) {
    const std::size_t plane = g.shape[1]*g.shape[2];
    EACH_INDEX(g.shape[0]*plane) {
        const std::size_t z = i/plane, y = i%plane/g.shape[2], x = i%g.shape[2];
        dst[i] = T(pass(src,g.shape,z,y,x,frame+3*row,weights,radius));
    }
}
} // namespace
void apply_cuda(const OrientedBlur& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[&](const auto& t, auto data, const Grid& grid, const ChannelPlan& c) {
        using T = typename decltype(data)::value_type;
        Buffer<double> frame(9);
        if (!t.axis_from.empty() && session.arrays.count(t.axis_from)) {
            const auto& f = s.at(t.axis_from); const auto& v = session.arrays.at(t.axis_from);
            require(f.target.role == Role::Vectors && v.shape.size() == 2 && v.shape[0] >= 1 && v.shape[1] == 3,
                    "OrientedBlur axis_from must name a 3D Vectors target with at least one row");
            int component[3]; for (int j = 0; j < 3; ++j) component[j] = axis(f.target.components[j],3);
            v.typed([&](auto row) { frame_from_vectors<<<1,1,0,vcr::device::stream()>>>(row.ptr,component[0],component[1],component[2],frame.ptr); });
        } else {
            double host[9]; const auto direction = oriented_detail::axis(t,s,c,t.axis_from.empty() ? nullptr : &s.at(t.axis_from).array);
            basis(direction.data(),host); frame.upload(host);
        }
        Grid3 g{{grid.shape[0],grid.shape[1],grid.shape[2]}};
        Buffer<T> scratch(data.size); T* source = data.ptr; T* dest = scratch.ptr;
        for (int pass = 0; pass < 3; ++pass) {
            const double sigma = c.values[pass ? 1 : 0];
            if (sigma <= 0) continue;
            Buffer<double> weights(kernel(sigma));
            oriented_pass<<<blocks(data.size),256,0,vcr::device::stream()>>>(source,dest,g,frame.ptr,pass,weights.ptr,int(weights.size/2));
            check(cudaGetLastError()); std::swap(source,dest);
        }
        if (source != data.ptr) copy_device(data,View<T>{source,data.size});
    });
}
} // namespace vcr::augment::cuda
