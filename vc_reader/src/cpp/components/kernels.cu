#include <vc_reader/components.hpp>
#include "../cuda_buffer.hpp"
#include "rows.hpp"

#include <cub/device/device_scan.cuh>
#include <climits>
#include <stdexcept>
#include <string>

// Union-find over voxels. Each union links the larger root under the smaller
// one, so a component's root is its first voxel in scan order for any thread
// schedule; an inclusive scan of root flags then numbers components in
// that order, as the CPU labelling does.
namespace vcr::components::cuda {
namespace {
struct Grid { long long nz, ny, nx; int reach; };

// Parents only move towards the root, so a stale read still leads to an ancestor;
// the compare-and-swap below retries from the parent it actually finds.
__device__ int root_of(const int* parent, int i) {
    for (int up; (up = parent[i]) != i;) i = up;
    return i;
}
__device__ void unite(int* parent, int a, int b) {
    for (;;) {
        a = root_of(parent, a); b = root_of(parent, b);
        if (a == b) return;
        if (a > b) { const int t = a; a = b; b = t; }
        const int up = atomicCAS(parent + b, b, a);  // link the larger root under the smaller
        if (up == b) return;
        b = up;
    }
}

// Points each foreground voxel at the start of its run within its 32-voxel
// segment (a warp, aligned to the index), so runs are one-hop trees. Each segment's
// first voxel heads the run there even when the run started in an earlier segment.
__global__ void initialize(const std::uint8_t* mask, int* parent, Grid g) {
    const long long n = g.nz * g.ny * g.nx;
    const int lane = threadIdx.x & 31;
    for (long long i = blockIdx.x * 256LL + threadIdx.x; i - lane < n; i += 256LL * gridDim.x) {
        const bool set = i < n && mask[i];
        const unsigned on = __ballot_sync(0xffffffffu, set);
        const bool start = set && !(lane && i % g.nx && (on >> (lane - 1) & 1));
        const unsigned starts = __ballot_sync(0xffffffffu, start) & (lane == 31 ? 0xffffffffu : (2u << lane) - 1);
        if (i < n) parent[i] = set ? int(i - lane + (31 - __clz(starts))) : -1;
    }
}
// Joins each foreground voxel with its foreground neighbours earlier in scan order
// whose offsets have at most `reach` nonzero axes (1: faces, 2: and edges, 3: all).
// With full connectivity a set neighbour makes the unions with its own earlier
// neighbours redundant (they join it themselves), so as on the CPU only the
// neighbours it does not touch are joined: in 3D, behind (z - 1) touches all
// others; left (x - 1) all but the x + 1 column; above (y - 1) all but the y + 1
// row behind. In 2D, above touches all others and left all but above-right.
__global__ void join(const std::uint8_t* mask, int* parent, Grid g) {
    const long long n = g.nz * g.ny * g.nx, plane = g.ny * g.nx;
    const bool full = g.reach == (g.nz > 1 ? 3 : 2);
    const auto with_left = [](long long i) { return (i & 31) == 0; };  // within a segment, left is already joined
    for (long long i = blockIdx.x * 256LL + threadIdx.x; i < n; i += 256LL * gridDim.x) {
        if (!mask[i]) continue;
        const long long x = i % g.nx, y = i / g.nx % g.ny, z = i / plane;
        // A neighbour at (dz, dy, dx), or -1 when outside or background.
        const auto at = [&](int dz, int dy, int dx) -> long long {
            const long long zz = z + dz, yy = y + dy, xx = x + dx;
            if (zz < 0 || yy < 0 || yy >= g.ny || xx < 0 || xx >= g.nx) return -1;
            const long long j = zz * plane + yy * g.nx + xx;
            return mask[j] ? j : -1;
        };
        const auto with = [&](long long j) { if (j >= 0) unite(parent, int(i), int(j)); };
        if (full && g.nz > 1) {
            if (const auto e = at(-1, 0, 0); e >= 0) { with(e); continue; }
            if (const auto m = at(0, 0, -1); m >= 0) {
                if (with_left(i)) with(m);
                if (const auto f = at(-1, 0, 1); f >= 0) with(f);
                else if (const auto l = at(0, -1, 1); l >= 0) { with(l); with(at(-1, 1, 1)); }
                else { with(at(-1, -1, 1)); with(at(-1, 1, 1)); }
                continue;
            }
            if (const auto k = at(0, -1, 0); k >= 0) {
                with(k);
                if (const auto h = at(-1, 1, 0); h >= 0) with(h);
                else { with(at(-1, 1, -1)); with(at(-1, 1, 1)); }
                continue;
            }
        } else if (full) {
            if (const auto k = at(0, -1, 0); k >= 0) { with(k); continue; }
            if (const auto m = at(0, 0, -1); m >= 0) { if (with_left(i)) with(m); with(at(0, -1, 1)); continue; }
        }
        for (int dz = -1; dz <= 0; ++dz) for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
            if (dz == 0 && (dy > 0 || (dy == 0 && dx >= 0))) continue;  // later in scan order, or itself
            if ((dz != 0) + (dy != 0) + (dx != 0) > g.reach) continue;
            if (g.nz == 1 && dz) continue;
            with(at(dz, dy, dx));
        }
    }
}
__global__ void flatten(int* parent, std::uint8_t* roots, long long n) {
    for (long long i = blockIdx.x * 256LL + threadIdx.x; i < n; i += 256LL * gridDim.x) {
        const int up = parent[i];
        if (up < 0) { roots[i] = 0; continue; }
        parent[i] = root_of(parent, int(i));
        roots[i] = up == i;
    }
}
__global__ void number(int* labels, const int* ranks, long long n) {
    for (long long i = blockIdx.x * 256LL + threadIdx.x; i < n; i += 256LL * gridDim.x)
        labels[i] = labels[i] < 0 ? 0 : ranks[labels[i]];
}
unsigned blocks(long long n) { return unsigned(std::min<long long>((n + 255) / 256, 65535)); }
}

std::size_t label(const std::uint8_t* mask, std::span<const std::size_t> shape, int connectivity,
                  std::int32_t* labels) {
    const auto dim = shape.size();
    if (dim != 2 && dim != 3) throw std::invalid_argument("label expects a 2D or 3D mask");
    int reach = 0;
    if (dim == 2 && (connectivity == 4 || connectivity == 8)) reach = connectivity == 4 ? 1 : 2;
    if (dim == 3 && (connectivity == 6 || connectivity == 18 || connectivity == 26)) reach = connectivity == 6 ? 1 : connectivity == 18 ? 2 : 3;
    if (!reach) detail::neighbor_rows(dim, connectivity);  // throws the shared message
    const Grid g{dim == 3 ? (long long)shape[0] : 1, (long long)shape[dim - 2], (long long)shape[dim - 1], reach};
    const long long n = g.nz * g.ny * g.nx;
    if (n > INT_MAX) throw std::invalid_argument("CUDA label supports at most INT32_MAX voxels");
    if (!n) return 0;
    int gpu = 0; device::check(cudaGetDevice(&gpu));
    const auto stream = device::stream();
    // Labels hold the union-find parents until numbered; byte root flags scan into
    // ranks. One allocation holds ranks, flags and the scan's scratch.
    std::size_t scratch_bytes = 0;
    device::check(cub::DeviceScan::InclusiveScan(nullptr, scratch_bytes, static_cast<const std::uint8_t*>(nullptr),
                                                 static_cast<int*>(nullptr), ::cuda::std::plus<int>{}, int(n), stream));
    const std::size_t rank_bytes = std::size_t(n) * sizeof(int), flag_bytes = (std::size_t(n) + 255) / 256 * 256;
    device::Buffer work(rank_bytes + flag_bytes + scratch_bytes, gpu);
    auto* rank = static_cast<int*>(work.ptr);
    auto* root = static_cast<std::uint8_t*>(work.ptr) + rank_bytes;
    void* scratch = root + flag_bytes;
    initialize<<<blocks(n), 256, 0, stream>>>(mask, labels, g);
    join<<<blocks(n), 256, 0, stream>>>(mask, labels, g);
    flatten<<<blocks(n), 256, 0, stream>>>(labels, root, n);
    device::check(cub::DeviceScan::InclusiveScan(scratch, scratch_bytes, root, rank, ::cuda::std::plus<int>{}, int(n), stream));
    number<<<blocks(n), 256, 0, stream>>>(labels, rank, n);
    device::check(cudaGetLastError());
    int count = 0;
    device::check(cudaMemcpyAsync(&count, rank + (n - 1), sizeof(int), cudaMemcpyDeviceToHost, stream));
    device::complete();
    return std::size_t(count);
}
}
