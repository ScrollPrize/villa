#include "channel.cuh"
namespace vcr::transforms::cuda {
struct SmearArgs { int axis, previous, shift[3]{}; double alpha; };
template<class T> __device__ void smear_slice(KernelIndex vcr_launch, T* data, Layout g, SmearArgs p, std::size_t slice) {
    EACH_INDEX(g.size/g.shape[p.axis]) {
        // Remove the smear axis from the flattened plane index.
        const auto step = g.stride[p.axis];
        const auto at = (i/step)*step*g.shape[p.axis] + i%step;
        std::size_t from = 0; int pa = 0;
        for (int a = 0; a < g.dim; ++a) if (a != p.axis) {
            auto n = static_cast<long long>(g.shape[a]);
            auto x = (static_cast<long long>(g.coord(at,a))-p.shift[pa++]) % n;
            from += std::size_t((x+n)%n)*g.stride[a];
        }
        T sum = 0;
        for (int j = 1; j <= p.previous; ++j) sum += data[(slice-j)*step+from];
        data[slice*step+at] = (T(1)-T(p.alpha))*data[slice*step+at] + T(p.alpha)*sum/T(p.previous);
    }
}
struct smear_slice_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { smear_slice(index,args...); }
};

void apply_cuda(const Smear& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto& t, auto v, const Grid& grid, const ChannelPlan& c) {
        Layout g(grid); SmearArgs args{int(c.values[0]),t.previous,{},t.alpha};
        std::copy_n(t.shift.begin(),g.dim-1,args.shift);
        for (std::size_t slice = t.previous; slice < grid.shape[args.axis]; ++slice) {
            launch(blocks(g.size/g.shape[args.axis]),256,0,vcr::device::stream(), smear_slice_launch{}, v.ptr,g,args,slice);
            check(cudaGetLastError());
        }
    });
}
} // namespace vcr::transforms::cuda
