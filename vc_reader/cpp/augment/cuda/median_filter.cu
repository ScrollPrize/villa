#include "channel.cuh"
#include "median.cuh"
namespace vcr::augment::cuda {
template<class T> __device__ __forceinline__ void sort_column(T (&v)[3]) {
    const T low = min(v[0],v[1]), high = max(v[0],v[1]);
    v[0] = min(low,v[2]); v[1] = max(low,min(high,v[2])); v[2] = max(high,v[2]);
}
// Each thread sorts one column per y row. Adjacent x outputs share those
// columns through warp shuffles: nine loads and three sorts per 3D output,
// instead of loading/sorting all 27 neighbors independently. Warp endpoints
// load their missing column directly; reflection of a size-3 window clamps.
template<class T, class Index, int D>
__device__ void median3_kernel(KernelIndex vcr_launch, const T* src, T* dst, Index nz, Index ny, Index nx, std::size_t n) {
    EACH_INDEX(n) {
        const Index z = Index(i)/(ny*nx), y = Index(i)/nx%ny, x = Index(i)%nx;
        const Index axis = D == 3 ? z : y, extent = D == 3 ? nz : ny;
        const Index before = axis ? axis-1 : 0, after = axis+1 < extent ? axis+1 : axis;
        const unsigned active = __activemask(), lane = threadIdx.x % 32;
        constexpr int rows = D == 3 ? 3 : 1;
        T v[rows*9];
        #pragma unroll
        for (int row = 0; row < rows; ++row) {
            const Index yy = row == 0 ? (y ? y-1 : 0) : row == 2 ? (y+1 < ny ? y+1 : y) : y;
            const Index b0 = D == 3 ? (before*ny+yy)*nx : before*nx;
            const Index b1 = D == 3 ? (z*ny+yy)*nx : y*nx;
            const Index b2 = D == 3 ? (after*ny+yy)*nx : after*nx;
            T column[3] = {src[b0+x],src[b1+x],src[b2+x]}; sort_column(column);
            #pragma unroll
            for (int dx = 0; dx < 3; ++dx) {
                const Index xx = dx == 0 ? (x ? x-1 : 0) : dx == 2 ? (x+1 < nx ? x+1 : x) : x;
                const int other = int(lane)+(xx > x ? 1 : xx < x ? -1 : 0);
                const bool shared = other >= 0 && other < 32 && (active & (1u << other));
                T neighbor[3];
                #pragma unroll
                for (int j = 0; j < 3; ++j)
                    neighbor[j] = __shfl_sync(active,column[j],shared ? other : int(lane));
                if (!shared) {
                    neighbor[0]=src[b0+xx]; neighbor[1]=src[b1+xx]; neighbor[2]=src[b2+xx];
                    sort_column(neighbor);
                }
                #pragma unroll
                for (int j = 0; j < 3; ++j) v[(row*3+dx)*3+j] = neighbor[j];
            }
        }
        if constexpr (D == 3) {
            select_median<median_detail::column_network>(v); dst[i] = v[median_detail::column_network.output];
        } else {
            select_median<median_detail::plane_network>(v); dst[i] = v[median_detail::plane_network.output];
        }
    }
}
template<class T, class Index, int D> struct median3_kernel_launch3 {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { median3_kernel<T,Index,D>(index,args...); }
};
template<class T, class Index>
void launch_median3(const T* src, T* dst, const Grid& g) {
    const auto& s = g.shape;
    if (s.size() == 3)
        launch(blocks(g.size),256,0,vcr::device::stream(), median3_kernel_launch3<T,Index,3>{}, src,dst,Index(s[0]),Index(s[1]),Index(s[2]),g.size);
    else
        launch(blocks(g.size),256,0,vcr::device::stream(), median3_kernel_launch3<T,Index,2>{}, src,dst,1,Index(s[0]),Index(s[1]),g.size);
}
// General exact rank filter. Binary search in the monotone IEEE key space uses
// bounded storage for all supported sizes (1..63), including even windows.
// Size-3 windows use the register selector above.
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
template<class T> __device__ void median_general(KernelIndex vcr_launch, const T* src, T* dst, Layout g, int k) {
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
struct median_general_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { median_general(index,args...); }
};
template<class T> __device__ void palette_probe(KernelIndex vcr_launch, View<T> input, unsigned long long* table, int* counts) {
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
struct palette_probe_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { palette_probe(index,args...); }
};
struct MedianPalette {
    template<class T> __device__ void operator()(KernelIndex, T* palette,const Statistics* stats) const {
        const double low=stats->low,high=stats->high,t=double(threadIdx.x)/255.;
        const double value=(low<=0 && high>=0)?t*high+(1-t)*low:low+t*(high-low);
        palette[threadIdx.x]=T(fmin(high,fmax(low,value)));
    }
};
struct QuantizeMedian {
    template<class T> __device__ void operator()(KernelIndex vcr_launch,View<T> data,const T* palette,const Statistics* stats,const int* counts) const {
        if (counts[0]<=256 && counts[1]!=3) return;
        const double low=stats->low,range=stats->high-low;
        if (!(range>0) || !isfinite(range)) return;
        EACH_INDEX(data.size) {
            const int code=int(clamp_value(floor((double(data.ptr[i])-low)/range*255+.5),0,255));
            data.ptr[i]=palette[code];
        }
    }
};
template<class T> void quantize_median(View<T> v) {
    Buffer<unsigned long long> table(1024); Buffer<int> counts(2);
    check(memset_async(table.ptr,255,1024*sizeof(unsigned long long),vcr::device::stream()));
    check(memset_async(counts.ptr,0,2*sizeof(int),vcr::device::stream()));
    launch(blocks(v.size),256,0,vcr::device::stream(),palette_probe_launch{},v,table.ptr,counts.ptr);
    auto stats=bounds(v);Buffer<T> palette(256);
    launch(1,256,0,vcr::device::stream(),MedianPalette{},palette.ptr,stats.ptr);
    launch(blocks(v.size),256,0,vcr::device::stream(),QuantizeMedian{},v,palette.ptr,stats.ptr,counts.ptr);
    check(cudaGetLastError());
}
void apply_cuda(const MedianFilter& t, Sample& s, const Parameters& p, Session& session) {
    for (const auto& c : p.channels) {
        if (c.values[0] == 1) continue;
        auto& a = session.arrays.at(c.key); Grid grid(a.spatial());
        const auto run = [&]<class T>() {
            auto input = a.channel<T>(c.channel);
            if (t.quantize) quantize_median(input);
            Shape shape{1}; shape.insert(shape.end(),grid.shape.begin(),grid.shape.end());
            DeviceArray output(shape,'f',sizeof(T)); auto out = output.view<T>();
            if (c.values[0] == 3) {
                if (grid.size <= UINT32_MAX) launch_median3<T,std::uint32_t>(input.ptr,out.ptr,grid);
                else launch_median3<T,std::size_t>(input.ptr,out.ptr,grid);
            } else launch(blocks(grid.size),256,0,vcr::device::stream(), median_general_launch{}, input.ptr,out.ptr,Layout(grid),int(c.values[0]));
            check(cudaGetLastError());
            if (a.shape[0] == 1) a = std::move(output);
            else copy_device(input,out);
        };
        if (a.bytes == 4) run.template operator()<float>(); else run.template operator()<double>();
    }
    session.restore_valid(s);
}
} // namespace vcr::augment::cuda
