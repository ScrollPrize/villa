#include "channel.cuh"
namespace vcr::augment::cuda {
struct Box { std::size_t low[3]{}, high[3]{}; double value; };
template<class T> __global__ void rectangle(T* data, Layout g, Box box) {
    EACH_INDEX(g.size) {
        bool inside = true;
        for (int a = 0; a < g.dim; ++a) inside &= g.coord(i,a) >= box.low[a] && g.coord(i,a) < box.high[a];
        if (inside) data[i] = T(box.value);
    }
}
void apply_cuda(const BlankRectangle& t, Sample& s, const Parameters& p) {
    apply_channels(t,s,p,[](const auto& t, auto& v, const Grid& grid, const ChannelPlan& c) {
        using V = typename std::decay_t<decltype(v)>::value_type;
        Buffer<V> data(v); Layout g(grid);
        for (std::size_t b = 0; b < c.values.size(); b += 2*g.dim+1) {
            Box box;
            for (int a = 0; a < g.dim; ++a) { box.low[a] = std::size_t(c.values[b+2*a]); box.high[a] = std::size_t(c.values[b+2*a+1]); }
            if (!t.statistic.empty()) data.download(v.data());
            box.value = rectangle_value(v,grid,c,b,t.statistic);
            rectangle<<<blocks(g.size),256>>>(data.ptr,g,box);
            check(cudaGetLastError());
        }
        data.download(v.data());
    });
}
} // namespace vcr::augment::cuda
