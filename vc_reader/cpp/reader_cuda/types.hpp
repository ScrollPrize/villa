#pragma once
#include <cstdint>
#include <cstddef>
namespace vcr::reader_cuda {
struct Grid {
    std::int64_t shape[3]{}, base[3]{};
    float origin[3]{}, axes[9]{};
    std::size_t offset = 0, count = 0;
};
// Resident source blocks ("bricks") subdivide storage chunks; grid counts bricks.
struct Source {
    std::int64_t shape[3]{}, brick[3]{}, grid[3]{};
    const std::int64_t* keys = nullptr;
    const unsigned char* const* pointers = nullptr;
    std::size_t mask = 0;
    int shift[3]{-1,-1,-1};  // log2(brick) when every brick extent is a power of two
    unsigned char fill = 0;
};
void launch(Source, const Grid*, std::size_t jobs, std::size_t largest,
            const void* points, int point_bits, std::size_t count,
            const std::int64_t* base, int kind, int order, bool floating, void* output);
// Copies `count` blocks of `bytes` from contiguous `staging` to targets[i] (device pointers).
void scatter(const void* staging, void* const* targets, std::size_t count, std::size_t bytes);
void cast(const void* input, void* output, std::size_t count, int input_bytes, int output_bytes);
}
