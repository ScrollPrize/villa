#pragma once
#include <cstdint>
#include <cstddef>
namespace vcr::reader_cuda {
struct Grid {
    std::int64_t shape[3]{}, base[3]{};
    float origin[3]{}, axes[9]{};
    std::size_t offset = 0, count = 0;
};
// Open-addressing lookup entry; a key and its pointer share one 16-byte line.
struct alignas(16) Slot { std::int64_t key; const unsigned char* pointer; };
// Fibonacci hashing: the high bits of the product. Low bits would depend only
// on the key's low bits, so bricks one z-slice of bricks apart (often a power
// of two keys) would share a home slot and lengthen linear probes.
#ifdef __CUDACC__
__host__ __device__
#endif
inline std::size_t home_slot(std::int64_t key, std::size_t mask) {
    const auto product=std::uint64_t(key)*UINT64_C(0x9e3779b97f4a7c15);
    return mask ? std::size_t(product>>(64-__builtin_popcountll(mask)))&mask : 0;
}
// Resident source blocks ("bricks") subdivide storage chunks; grid counts bricks.
struct Source {
    std::int64_t shape[3]{}, brick[3]{}, grid[3]{};
    const Slot* table = nullptr;
    std::size_t mask = 0;
    int shift[3]{-1,-1,-1};  // log2(brick) when every brick extent is a power of two
    unsigned char fill = 0;
};
void launch(Source, const Grid*, std::size_t jobs, std::size_t largest,
            const void* points, int point_bits, std::size_t count,
            const std::int64_t* base, int kind, int order, bool floating, void* output, void* valid = nullptr);
// Copies `count` blocks of `bytes` from contiguous `staging` to targets[i] (device pointers).
void scatter(const void* staging, void* const* targets, std::size_t count, std::size_t bytes);
void cast(const void* input, void* output, std::size_t count, int input_bytes, int output_bytes);
// Validity of the samples launch() would produce: 1 when every neighbour the
// sample reads (nonzero linear weights, or the nearest voxel) lies inside the
// volume and, with `resident`, in a brick of the Source table.
void validity(Source, const Grid*, std::size_t jobs, std::size_t largest,
              const void* points, int point_bits, std::size_t count,
              const std::int64_t* base, int kind, int order, bool resident, void* output);
// Grid descriptors from device origins (jobs x 3, float32/float64) and axes
// (jobs x 9 float32) for crops of one shape; base is coordinate_origin or the
// clamped floor of each origin, as affine32().
void device_grids(const void* origins, int origin_bits, const float* axes, std::size_t jobs,
                  const std::int64_t* shape, const std::int64_t* coordinate_origin, Grid* grids);
}
