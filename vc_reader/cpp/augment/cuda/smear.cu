#include "channel.cuh"
namespace vcr::augment::cuda {
struct SmearArgs { int axis, previous, shift[3]{}; double alpha; };
__global__ void smear_slice(double* data, Layout g, SmearArgs p, std::size_t slice) {
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
        double sum = 0;
        for (int j = 1; j <= p.previous; ++j) sum += data[(slice-j)*step+from];
        data[slice*step+at] = (1-p.alpha)*data[slice*step+at] + p.alpha*sum/p.previous;
    }
}

void apply_cuda(const Smear& t, Sample& s, const Parameters& p) {
    apply_channels(t,s,p,[](const auto& t, std::vector<double>& v, const Grid& grid, const ChannelPlan& c) {
        Buffer<double> data(v); Layout g(grid);
        SmearArgs args{int(c.values[0]),t.previous,{},t.alpha};
        std::copy_n(t.shift.begin(),g.dim-1,args.shift);
        for (std::size_t slice = t.previous; slice < grid.shape[args.axis]; ++slice) {
            smear_slice<<<blocks(g.size/g.shape[args.axis]),256>>>(data.ptr,g,args,slice);
            check(cudaGetLastError());
        }
        data.download(v.data());
    });
}
} // namespace vcr::augment::cuda
