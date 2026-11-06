#include "common.hpp"
#include "../cuda_buffer.hpp"
#include <algorithm>

namespace vcr::views::cuda {
using namespace detail;

namespace {
unsigned blocksFor(std::size_t n) { return unsigned(std::min<std::size_t>((n + 255) / 256, 65535)); }

__global__ void supportKernel(const std::uint8_t* valid, std::size_t cells, Index3 grid, Index3 shape, Index3 patch,
                              std::uint8_t* support)
{
    const auto perImage = std::size_t(grid[0] * grid[1] * grid[2]);
    const auto voxels = std::size_t(shape[0] * shape[1] * shape[2]);
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < cells; i += std::size_t(gridDim.x) * blockDim.x) {
        const std::size_t b = i / perImage;
        const auto c = std::int64_t(i % perImage);
        const auto x = c % grid[2], y = c / grid[2] % grid[1], z = c / (grid[1] * grid[2]);
        support[i] = cellSupported(valid + b * voxels, shape.data(), patch.data(), z, y, x);
    }
}

__global__ void correspondenceKernel(PairGeometry g, const PairPlan* plans, std::size_t pairs, const std::uint8_t* support,
                                     std::int32_t* teacherIndex, std::uint8_t* studentValid, std::int32_t* lattice)
{
    const auto n = pairs * std::size_t(g.studentTokens);
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x) {
        const auto p = i / std::size_t(g.studentTokens);
        const auto offset = std::int64_t(p) * g.studentTokens;
        correspond(g, plans[p], support, std::int64_t(i) - offset, teacherIndex + offset, studentValid + offset,
                   lattice ? lattice + 3 * offset : nullptr);
    }
}

template <class T>
__global__ void maskKernel(const T* in, T* out, std::size_t n, std::size_t channels, Index3 shape, Index3 patch,
                           Index3 grid, const std::uint8_t* masks, T fill, bool inPlace)
{
    const auto voxels = std::size_t(shape[0] * shape[1] * shape[2]);
    const auto tokens = std::size_t(grid[0] * grid[1] * grid[2]);
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x) {
        const auto v = std::int64_t(i % voxels);
        const std::size_t b = i / voxels / channels;
        const auto x = v % shape[2], y = v / shape[2] % shape[1], z = v / (shape[1] * shape[2]);
        const auto cell = (z / patch[0] * grid[1] + y / patch[1]) * grid[2] + x / patch[2];
        if (masks[b * tokens + std::size_t(cell)]) out[i] = fill;
        else if (!inPlace) out[i] = in[i];
    }
}
}  // namespace

void tokenSupport(const std::uint8_t* valid, std::size_t batch, const Index3& shape, const Index3& patch,
                  std::uint8_t* support)
{
    const auto grid = cells(shape, patch);
    require(batch == 0 || (valid && support), "valid and support arrays are required");
    const auto n = batch * std::size_t(product(grid));
    if (!n) return;
    supportKernel<<<blocksFor(n), 256, 0, device::stream()>>>(valid, n, grid, shape, patch, support);
    device::check(cudaGetLastError());
}

void correspondence(const Supports& support, const Pairs& pairs, std::int32_t* teacherIndex,
                    std::uint8_t* studentValid, std::int32_t* lattice)
{
    const auto g = pairGeometry(support, pairs);
    const auto plans = pairPlans(support, pairs, g);
    if (!pairs.count) return;
    require(support.data && teacherIndex && studentValid, "support and output arrays are required");
    int id = 0;
    device::check(cudaGetDevice(&id));
    const auto bytes = plans.size() * sizeof(PairPlan);
    // Stream-ordered: freed after the kernel on the same stream. Pageable copies
    // are staged before cudaMemcpyAsync returns, so `plans` may go out of scope.
    device::Buffer staged(bytes, id);
    device::check(cudaMemcpyAsync(staged.ptr, plans.data(), bytes, cudaMemcpyHostToDevice, device::stream()));
    const auto n = pairs.count * std::size_t(g.studentTokens);
    correspondenceKernel<<<blocksFor(n), 256, 0, device::stream()>>>(g, static_cast<const PairPlan*>(staged.ptr), pairs.count,
                                                                      support.data, teacherIndex, studentValid, lattice);
    device::check(cudaGetLastError());
}

void maskPatchCells(const CellImages& images, const std::uint8_t* masks, float fill)
{
    const auto grid = validateCells(images);
    const auto n = images.batch * images.channels * std::size_t(product(images.shape));
    if (!n) return;
    require(masks != nullptr, "masks are required");
    const auto bits = fillBits(images.type, fill);
    const bool inPlace = images.images == images.out;
    const auto stream = device::stream();
    if (images.type == Real::Float32)
        maskKernel<<<blocksFor(n), 256, 0, stream>>>(static_cast<const std::uint32_t*>(images.images), static_cast<std::uint32_t*>(images.out),
                                                     n, images.channels, images.shape, images.patch, grid, masks, bits, inPlace);
    else
        maskKernel<<<blocksFor(n), 256, 0, stream>>>(static_cast<const std::uint16_t*>(images.images), static_cast<std::uint16_t*>(images.out),
                                                     n, images.channels, images.shape, images.patch, grid, masks, std::uint16_t(bits), inPlace);
    device::check(cudaGetLastError());
}
}  // namespace vcr::views::cuda
