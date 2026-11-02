#include "separable_filter.cuh"
#include "../cpu/distance_transform.hpp"
namespace vcr::transforms::cuda {
namespace {
using namespace distance_detail;
using filter_detail::separable_cuda;
// Field values: float64 with 1e300 for the envelope, exact uint32 with UINT32_MAX
// for tiled passes; indices int64 or uint32 likewise.
__device__ constexpr double unreached(double) { return 1e300; }
__device__ constexpr unsigned unreached(unsigned) { return 0xffffffffu; }
template<class S, class F, class I> __device__ void edt_features(KernelIndex vcr_launch, const S* labels, F* f, std::size_t n, double threshold, bool inside, int* any, I* index) {
    EACH_INDEX(n) {
        const bool feature = (double(labels[i]) > threshold) != inside;
        f[i] = feature ? F(0) : unreached(F());
        if (index) index[i] = I(i);
        if (feature) *any = 1;
    }
}
struct edt_features_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { edt_features(index,args...); }
};
// One thread per line along `axis`; lines start where that coordinate is zero.
__device__ void edt_lines(KernelIndex vcr_launch, double* f, Layout g, int axis, int* v, double* z, double* scratch, std::size_t lines,
                          long long* index, long long* index_scratch) {
    const std::size_t n = g.shape[axis], stride = g.stride[axis];
    EACH_INDEX(lines) {
        // Enumerate line starts: all indices with coord(axis) == 0.
        const std::size_t outer = i/stride, inner = i%stride, start = outer*stride*n + inner;
        // Scratch of neighbouring lines is interleaved, so a warp's accesses coalesce.
        envelope(f+start,n,stride,v+i,z+i,scratch+i,index ? index+start : nullptr,index_scratch ? index_scratch+i : nullptr,lines);
    }
}
struct edt_lines_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { edt_lines(index,args...); }
};
// One squared-distance pass along an axis. A block loads a tile of `tile`
// neighbouring lines (so loads coalesce) into shared memory as exact uint32
// (UINT32_MAX: no feature yet). The cost (q - p)^2 + f[p] is Monge, so the
// smallest minimizing p never decreases with q: every `edt_sample`-th voxel (and
// the last) finds its minimizer by an outward search, and each other voxel scans
// only between its neighbouring samples' minimizers. A sample's search is split
// across a group of the line's threads, interleaving distances d, and stops once
// d^2 + min(f) exceeds the best cost found by the group: those candidates cannot
// tie. Keys pack (cost, p), so ties keep the smaller p, as the lower envelope
// does; distances and indices equal envelope's.
constexpr std::size_t edt_sample = 32;
__host__ __device__ constexpr std::size_t edt_samples(std::size_t n) { return (n - 1) / edt_sample + 2; }
__host__ __device__ constexpr std::size_t edt_tile_bytes(std::size_t n, std::size_t tile) {
    return edt_samples(n) * tile * sizeof(unsigned long long) + (2 * n + edt_samples(n) + 1) * tile * sizeof(unsigned);
}
struct edt_tile_pass {
    __device__ void operator()(KernelIndex vcr_launch, unsigned* f, unsigned* index, std::size_t n, std::size_t stride,
                               std::size_t lines, unsigned tile) const {
        extern __shared__ unsigned long long shared_keys[];
        const std::size_t samples = edt_samples(n);
        unsigned long long* keys = shared_keys;  // samples * tile
        unsigned* value = reinterpret_cast<unsigned*>(keys + samples * tile);
        unsigned* nearest = value + n * tile; unsigned* argmin = nearest + n * tile; unsigned* lowest = argmin + samples * tile;
        constexpr unsigned none = 0xffffffffu;
        const unsigned tx = threadIdx.x % tile, ty = threadIdx.x / tile, step = blockDim.x / tile;
        // Threads of a line split into groups, one sample per group at a time.
        const unsigned group_size = unsigned(std::max<std::size_t>(1, step / samples)), groups_per_line = step / group_size;
        const unsigned member = ty % group_size, team = ty / group_size;
        const std::size_t tiles = (lines + tile - 1) / tile;
        for (std::size_t group = vcr_launch.block; group < tiles; group += vcr_launch.blocks) {
            const std::size_t line = group * tile + tx;
            const bool valid = line < lines;
            const std::size_t start = valid ? line / stride * stride * n + line % stride : 0;
            // Element e of the tile: x lines are contiguous, so q runs fastest there;
            // strided lines are neighbours, so the line runs fastest.
            const auto element = [&](std::size_t e, std::size_t& l, std::size_t& q) {
                if (stride == 1) { l = e / n; q = e % n; } else { l = e % tile; q = e / tile; }
            };
            if (threadIdx.x < tile) lowest[threadIdx.x] = none;
            for (std::size_t k = threadIdx.x; k < samples * tile; k += blockDim.x) keys[k] = ~0ull;
            __syncthreads();
            // Each thread's elements lie on one line (blockDim is a multiple of tile).
            unsigned low = none; std::size_t mine = 0;
            for (std::size_t e = threadIdx.x; e < n * tile; e += blockDim.x) {
                std::size_t l, q; element(e, l, q); mine = l;
                const std::size_t at = group * tile + l;
                if (at >= lines) continue;
                const std::size_t base = at / stride * stride * n + at % stride;
                const unsigned u = f[base + q * stride];
                value[q * tile + l] = u; low = min(low, u);
                if (index) nearest[q * tile + l] = index[base + q * stride];
            }
            if (low != none) atomicMin(lowest + mine, low);
            __syncthreads();
            const unsigned floor_ = lowest[tx];
            const bool work = valid && floor_ != none;
            if (work && team < groups_per_line) for (std::size_t k = team; k < samples; k += groups_per_line) {
                const std::size_t q = std::min(k * edt_sample, n - 1);
                unsigned long long* best = keys + k * tile + tx;
                if (member == 0 && value[q * tile + tx] != none) atomicMin(best, (unsigned long long)value[q * tile + tx] << 32 | q);
                for (unsigned long long d = 1 + member; d <= q || q + d < n; d += group_size) {
                    if (d * d + floor_ > (*(volatile unsigned long long*)best >> 32)) break;
                    if (d <= q) { const unsigned w = value[(q - d) * tile + tx];
                        if (w != none) atomicMin(best, (d * d + w) << 32 | (q - d)); }
                    if (q + d < n) { const unsigned w = value[(q + d) * tile + tx];
                        if (w != none) atomicMin(best, (d * d + w) << 32 | (q + d)); }
                }
            }
            __syncthreads();
            if (work) for (std::size_t k = ty; k < samples; k += step) argmin[k * tile + tx] = unsigned(keys[k * tile + tx]);
            __syncthreads();
            if (work) for (std::size_t q = ty; q < n; q += step) {
                // Nearest first within [low, high], stopping as the samples do: outward
                // from q inside the bracket, else from the bracket's near end.
                const std::size_t k = q / edt_sample, low = argmin[k * tile + tx], high = argmin[(k + 1) * tile + tx];
                unsigned long long best = ~0ull; std::size_t from = low;
                const auto cost = [&](std::size_t p) -> unsigned long long {
                    const unsigned w = value[p * tile + tx];
                    const unsigned long long d = q > p ? q - p : p - q;
                    return w == none ? ~0ull : d * d + w;
                };
                const auto beyond = [&](std::size_t p) {  // no candidate this far can tie
                    const unsigned long long d = q > p ? q - p : p - q;
                    return best != ~0ull && d * d + floor_ > best;
                };
                if (q < low) {
                    for (std::size_t p = low; p <= high && !beyond(p); ++p) { const auto c = cost(p); if (c < best) { best = c; from = p; } }
                } else if (q > high) {
                    for (std::size_t p = high + 1; p-- > low && !beyond(p);) { const auto c = cost(p); if (c <= best) { best = c; from = p; } }
                } else {
                    best = cost(q); from = q;
                    for (std::size_t d = 1; q - low >= d || high - q >= d; ++d) {
                        if (best != ~0ull && d * d + floor_ > best) break;
                        if (q - low >= d) { const auto c = cost(q - d); if (c <= best && c != ~0ull) { best = c; from = q - d; } }
                        if (high - q >= d) { const auto c = cost(q + d); if (c < best) { best = c; from = q + d; } }
                    }
                }
                f[start + q * stride] = unsigned(best);
                if (index) index[start + q * stride] = nearest[from * tile + tx];
            }
            __syncthreads();
        }
    }
};
// float(sqrt(squared)), as the CPU path rounds it. Below 2^24 the integer is an
// exact float and the correctly rounded sqrtf gives the same float, without FP64.
__device__ float root(double v) { return float(sqrt(v)); }
// Kept out of line so the compiler branches instead of evaluating FP64 for every voxel.
__device__ __noinline__ float wide_root(unsigned v) { return float(sqrt(double(v))); }
__device__ float root(unsigned v) {
    if (v < (1u << 24)) return sqrtf(float(v));
    return wide_root(v);
}
// Distance to (-1, 0, ...) for a class absent from the grid. The grid arrives as
// scalars: taking the address of a Layout argument copies it to local memory in
// every thread, which made this kernel several times slower.
__device__ float absent_distance(std::size_t i, std::size_t ny, std::size_t nx, int dim) {
    const double x = double(i % nx), y = double(i / nx % ny), z = double(i / (nx * ny));
    const double first = dim == 3 ? z + 1 : y + 1;
    return float(sqrt(first * first + (dim == 3 ? y * y : 0.) + x * x));
}
// In signed mode one distance of every voxel is zero, so the float difference is exact.
template<class F> __device__ void edt_finish(KernelIndex vcr_launch, const F* inside, const F* outside, float* out, std::size_t n, int mode,
                                             const int* any, std::size_t ny, std::size_t nx, int dim, bool implicit_background) {
    const bool use_inside = mode != 2, use_outside = mode != 1;
    const bool has_inside = any[0], has_outside = any[1];
    EACH_INDEX(n) {
        const float in = !use_inside ? 0.f : has_inside ? root(inside[i]) : implicit_background ? absent_distance(i, ny, nx, dim) : 0.f;
        const float out_ = !use_outside ? 0.f : has_outside ? root(outside[i]) : implicit_background ? absent_distance(i, ny, nx, dim) : 0.f;
        out[i] = mode == 1 ? in : mode == 2 ? out_ : out_ - in;
    }
}
struct edt_finish_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { edt_finish(index,args...); }
};
// Nearest feature coordinates, (D, ...) int32, as the CPU path writes them.
template<class S, class I> __device__ void edt_indices(KernelIndex vcr_launch, const S* labels, double threshold, const I* inside, const I* outside,
                                              std::int32_t* out, std::size_t n, int mode, const int* any, std::size_t ny, std::size_t nx, int dim,
                                              bool implicit_background) {
    EACH_INDEX(n) {
        const bool from_inside = mode == 1 || (mode == 0 && double(labels[i]) > threshold);
        const long long j = any[from_inside ? 0 : 1] ? (long long)(from_inside ? inside : outside)[i] : (implicit_background ? -1 : (long long)i);
        const std::int32_t x = j < 0 ? 0 : std::int32_t(std::size_t(j) % nx), y = j < 0 ? (dim == 2 ? -1 : 0) : std::int32_t(std::size_t(j) / nx % ny);
        if (dim == 3) { out[i] = j < 0 ? -1 : std::int32_t(std::size_t(j) / (nx * ny)); out[n + i] = y; out[2 * n + i] = x; }
        else { out[i] = y; out[n + i] = x; }
    }
}
struct edt_indices_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { edt_indices(index,args...); }
};
template<class T> __device__ void normalize_normals(KernelIndex vcr_launch, T* const* gradient, T* out, std::size_t n, int d, int c0, int c1, int c2) {
    const int order[3] = {c0,c1,c2};
    EACH_INDEX(n) {
        double norm = 0; for (int a = 0; a < d; ++a) norm += double(gradient[a][i])*double(gradient[a][i]);
        norm = fmax(sqrt(norm),1e-6);
        for (int j = 0; j < d; ++j) out[j*n+i] = T(double(gradient[order[j]][i])/norm);
    }
}
struct normalize_normals_launch {
    template<class... Args> __device__ void operator()(KernelIndex index, Args... args) const { normalize_normals(index,args...); }
};
} // namespace

void apply_cuda(const DistanceTransform& t, Sample& s, const Parameters&, Session& session) {
    std::vector<std::pair<std::string,Field>> outputs;
    for (const auto& [name,f] : s) if (selected_label(name,f,t.on)) {
        const auto& a = session.arrays.at(name); Grid grid(a.spatial()); Layout g(grid); const int d = int(grid.shape.size());
        require(t.channel < a.shape[0],"DistanceTransform channel out of bounds");
        const std::size_t n = grid.size, extent = *std::max_element(grid.shape.begin(),grid.shape.end());
        Buffer<int> any(2); check(memset_async(any.ptr,0,2*sizeof(int),vcr::device::stream()));
        const bool indices = !t.indices.empty();
        // Tiled passes keep exact uint32 squared distances (at most the sum of squared
        // extents) and flat indices, and a line's tile in 48 KiB of shared memory;
        // larger grids use the float64 envelope.
        double reach = 0; for (auto e : grid.shape) reach += double(e) * double(e);
        const bool tiled = reach < 4294967295.0 && n < 4294967295ull && edt_tile_bytes(extent, 1) <= 48 * 1024;
        Shape shape{1}; shape.insert(shape.end(),grid.shape.begin(),grid.shape.end());
        DeviceArray result(shape,'f',4);
        Shape ishape{std::size_t(d)}; ishape.insert(ishape.end(),grid.shape.begin(),grid.shape.end());
        DeviceArray nearest(indices ? ishape : Shape{0},'i',4);
        const auto run = [&](auto field_type, auto index_type) {
            using F = decltype(field_type); using I = decltype(index_type);
            Buffer<F> inside(n), outside(n);
            Buffer<I> inside_index(indices ? n : 0), outside_index(indices ? n : 0);
            const std::size_t lines = tiled ? 0 : n/std::min_element(grid.shape.begin(),grid.shape.end())[0];
            Buffer<int> v(lines*extent); Buffer<double> z(tiled ? 0 : lines*(extent+1)), scratch(lines*extent);
            Buffer<long long> index_scratch(indices ? lines*extent : 0);
            for (int pass = 0; pass < 2; ++pass) {
                const bool in = pass == 0; auto& field = in ? inside : outside;
                if (in && t.mode == DistanceMode::Outside) continue;
                if (!in && t.mode == DistanceMode::Inside) continue;
                I* index = indices ? (in ? inside_index : outside_index).ptr : nullptr;
                a.typed([&](auto labels) { launch(blocks(n),256,0,vcr::device::stream(), edt_features_launch{}, labels.ptr+t.channel*n,field.ptr,n,t.threshold,in,any.ptr+(in ? 0 : 1),index); });
                for (int axis_ = 0; axis_ < d; ++axis_) {
                    const std::size_t count = n/grid.shape[axis_], extent_ = grid.shape[axis_], stride = grid.stride[axis_];
                    if constexpr (std::is_same_v<F, unsigned>) {
                        // Lines of a tile are neighbours along x for strided axes; x lines load contiguously.
                        // Eight 4-byte lines fill a 32-byte sector, with more blocks than wider
                        // tiles; contiguous x lines take one per block.
                        unsigned tile = stride == 1 ? 1u : 8u;
                        while (tile > 1 && edt_tile_bytes(extent_, tile) > 48 * 1024) tile /= 2;
                        // About one thread per voxel of the tile, in whole warps.
                        const unsigned threads = unsigned(std::clamp<std::size_t>((extent_ * tile + 31) / 32 * 32, 64, 1024));
                        launch(unsigned(std::min<std::size_t>((count + tile - 1) / tile, 65535)),threads,edt_tile_bytes(extent_, tile),vcr::device::stream(),
                               edt_tile_pass{}, field.ptr,index,extent_,stride,count,tile);
                    } else {
                        launch(blocks(count),256,0,vcr::device::stream(), edt_lines_launch{}, field.ptr,g,axis_,v.ptr,z.ptr,scratch.ptr,count,
                               index,indices ? index_scratch.ptr : nullptr);
                    }
                }
                check(cudaGetLastError());
            }
            launch(blocks(n),256,0,vcr::device::stream(), edt_finish_launch{}, inside.ptr,outside.ptr,result.view<float>().ptr,n,int(t.mode),any.ptr,
                   grid.shape[d - 2],grid.shape[d - 1],d,t.implicit_background);
            if (indices) a.typed([&](auto labels) {
                launch(blocks(n),256,0,vcr::device::stream(), edt_indices_launch{}, labels.ptr+t.channel*n,t.threshold,inside_index.ptr,outside_index.ptr,
                       nearest.view<std::int32_t>().ptr,n,int(t.mode),any.ptr,grid.shape[d - 2],grid.shape[d - 1],d,t.implicit_background);
            });
            check(cudaGetLastError());
        };
        if (tiled) run(0u, 0u); else run(0., 0LL);
        if (!t.normals.empty()) {
            std::vector<Buffer<float>> gradient; std::vector<float*> pointers; Buffer<float> tmp(n);
            for (int axis_ = 0; axis_ < d; ++axis_) {
                gradient.emplace_back(n);
                check(copy_async(gradient.back().ptr,result.view<float>().ptr,n*sizeof(float),cudaMemcpyDeviceToDevice,vcr::device::stream()));
                std::vector<Taps> taps(d,scharr_smooth()); taps[axis_] = scharr_derivative();
                float* data = gradient.back().ptr; float* scratch_ptr = tmp.ptr;
                separable_cuda(data,scratch_ptr,grid,taps,Boundary::Reflect);
                if (data != gradient.back().ptr) check(copy_async(gradient.back().ptr,data,n*sizeof(float),cudaMemcpyDeviceToDevice,vcr::device::stream()));
            }
            for (auto& b : gradient) pointers.push_back(b.ptr);
            Buffer<float*> table(pointers); int order[3]{};
            for (int j = 0; j < d; ++j) order[j] = axis(t.components[j],d);
            Shape vshape{std::size_t(d)}; vshape.insert(vshape.end(),grid.shape.begin(),grid.shape.end());
            DeviceArray normals(vshape,'f',4);
            launch(blocks(n),256,0,vcr::device::stream(), normalize_normals_launch{}, table.ptr,normals.view<float>().ptr,n,d,order[0],order[1],order[2]);
            check(cudaGetLastError());
            outputs.emplace_back(name+t.normals,Field{metadata(vshape,4),derived(Role::NormalField,t.components),name});
            session.arrays.emplace(name+t.normals,std::move(normals));
        }
        if (indices) {
            Array meta = metadata(ishape,4); meta.kind = 'i';
            outputs.emplace_back(name+t.indices,Field{std::move(meta),derived(Role::Label),name});
            session.arrays.emplace(name+t.indices,std::move(nearest));
        }
        outputs.emplace_back(name+t.suffix,Field{metadata(shape,4),derived(Role::Scalar),name});
        session.arrays.emplace(name+t.suffix,std::move(result));
    }
    for (auto& [name,f] : outputs) { require(!s.count(name),"derived field already exists: "+name); s.emplace(name,std::move(f)); }
}

} // namespace vcr::transforms::cuda
