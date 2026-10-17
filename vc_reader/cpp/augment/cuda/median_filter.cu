#include "channel.cuh"
#include "median.cuh"
namespace vcr::augment::cuda {
template<class T> __global__ void median3_kernel(const T* src, T* dst, Layout g) {
    EACH_INDEX(g.size) {
        T v[27];
        const auto z = g.coord(i,0), y = g.coord(i,1), x = g.coord(i,2);
        #pragma unroll
        for (int dz = 0; dz < 3; ++dz) {
            #pragma unroll
            for (int dy = 0; dy < 3; ++dy) {
                #pragma unroll
                for (int dx = 0; dx < 3; ++dx)
                    v[(dz*3+dy)*3+dx] = src[boundary(static_cast<long long>(z)+dz-1,g.shape[0],Boundary::Reflect)*g.stride[0] + boundary(static_cast<long long>(y)+dy-1,g.shape[1],Boundary::Reflect)*g.stride[1] + boundary(static_cast<long long>(x)+dx-1,g.shape[2],Boundary::Reflect)];
            }
        }
        select27(v); dst[i] = v[median_detail::network.output];
    }
}
// General exact rank filter. Binary search in the monotone IEEE key space uses
// bounded storage for all supported sizes (1..63), including even windows.
// Size-3 3D uses the much faster register selector above.
template<class T> __device__ unsigned long long ordered_key(T x) {
    if constexpr (std::is_same_v<T,float>) {
        unsigned u = __float_as_uint(x); return (u & 0x80000000u) ? (~u) : (u ^ 0x80000000u);
    } else {
        auto u = static_cast<unsigned long long>(__double_as_longlong(x));
        return (u & (1ull<<63)) ? ~u : (u ^ (1ull<<63));
    }
}
template<class T> __device__ T from_key(unsigned long long key) {
    if constexpr (std::is_same_v<T,float>) {
        unsigned u = unsigned(key); return __uint_as_float((u & 0x80000000u) ? (u ^ 0x80000000u) : ~u);
    } else return __longlong_as_double((key & (1ull<<63)) ? (key ^ (1ull<<63)) : ~key);
}
template<class T> __global__ void median_general(const T* src, T* dst, Layout g, int k) {
    EACH_INDEX(g.size) {
        const int kz = g.dim == 3 ? k : 1;
        const auto nz = g.dim == 3 ? g.shape[0] : 1, ny = g.shape[g.dim-2], nx = g.shape[g.dim-1];
        const auto z = i/(ny*nx), y = i/nx%ny, x = i%nx;
        unsigned long long low = 0, high = sizeof(T)==4 ? 0xffffffffull : ~0ull;
        const int rank = kz*k*k/2;
        while (low < high) {
            auto mid = low + (high-low)/2; int count = 0;
            for (int dz = 0; dz < kz; ++dz) for (int dy = 0; dy < k; ++dy) for (int dx = 0; dx < k; ++dx) {
                auto at = (boundary(static_cast<long long>(z)+dz-kz/2,nz,Boundary::Reflect)*ny + boundary(static_cast<long long>(y)+dy-k/2,ny,Boundary::Reflect))*nx + boundary(static_cast<long long>(x)+dx-k/2,nx,Boundary::Reflect);
                count += ordered_key(src[at]) <= mid;
            }
            if (count > rank) high = mid; else low = mid+1;
        }
        dst[i] = from_key<T>(low);
    }
}
template<class T> __global__ void palette_probe(View<T> input, unsigned long long* table, int* counts) {
    EACH_INDEX(input.size) {
        T value = input.ptr[i];
        if (value == 0) atomicOr(counts+1,signbit(value) ? 2 : 1);
        if (atomicAdd(counts,0) > 256) continue;
        unsigned long long key;
        if constexpr (std::is_same_v<T,float>) key = __float_as_uint(value);
        else key = __double_as_longlong(value);
        auto hash = (key ^ (key>>32))*0x9e3779b97f4a7c15ull;
        for (unsigned slot = unsigned(hash>>54), j = 0; j < 1024; ++j, slot = (slot+1)&1023) {
            auto old = atomicCAS(table+slot,~0ull,key);
            if (old == key) break;
            if (old == ~0ull) { atomicAdd(counts,1); break; }
        }
    }
}
template<class T> __global__ void quantize_kernel(View<T> data, const T* palette, double low, double range) {
    EACH_INDEX(data.size) {
        int code = int(clamp_value(floor((double(data.ptr[i])-low)/range*255+.5),0,255));
        data.ptr[i] = palette[code];
    }
}
template<class T> void quantize_median(View<T> v) {
    Buffer<unsigned long long> table(1024); Buffer<int> counts(2);
    check(cudaMemsetAsync(table.ptr,255,1024*sizeof(unsigned long long),vcr::device::stream()));
    check(cudaMemsetAsync(counts.ptr,0,2*sizeof(int),vcr::device::stream()));
    palette_probe<<<blocks(v.size),256,0,vcr::device::stream()>>>(v,table.ptr,counts.ptr); check(cudaGetLastError());
    int result[2]; counts.download(result);
    if (result[0] <= 256 && result[1] != 3) return;
    auto stats = statistics(v); Statistics bounds; stats.download(&bounds);
    double range = bounds.high-bounds.low;
    if (!(range > 0) || !std::isfinite(range)) return;
    std::vector<T> palette(256);
    for (std::size_t i = 0; i < 256; ++i) palette[i] = T(std::lerp(bounds.low,bounds.high,double(i)/255));
    Buffer<T> values(palette);
    quantize_kernel<<<blocks(v.size),256,0,vcr::device::stream()>>>(v,values.ptr,bounds.low,range); check(cudaGetLastError());
}
void apply_cuda(const MedianFilter& t, Sample& s, const Parameters& p, Session& session) {
    apply_channels(t,s,p,session,[](const auto& t, auto v, const Grid& grid, const ChannelPlan& c) {
        using V = typename decltype(v)::value_type;
        if (c.values[0] == 1) return;
        if (t.quantize) quantize_median(v);
        Buffer<V> out(v.size); Layout g(grid);
        if (g.dim == 3 && c.values[0] == 3) median3_kernel<<<blocks(g.size),256,0,vcr::device::stream()>>>(v.ptr,out.ptr,g);
        else median_general<<<blocks(g.size),256,0,vcr::device::stream()>>>(v.ptr,out.ptr,g,int(c.values[0]));
        check(cudaGetLastError()); copy_device(v,View<V>{out.ptr,v.size});
    });
}
} // namespace vcr::augment::cuda
