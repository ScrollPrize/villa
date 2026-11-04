#include "common.hpp"
#include "../cuda_buffer.hpp"
namespace vcr::pathfinding::cuda {
namespace {
__global__ void relax(const float* costs, std::size_t rows, std::size_t cols,
                      const detail::Label* input, detail::Label* output, int* changed) {
    const auto n = rows * cols;
    for (std::size_t p = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; p < n;
         p += std::size_t(gridDim.x) * blockDim.x) {
        auto best = input[p];
        const auto row = std::int64_t(p / cols), col = std::int64_t(p % cols);
        if (isfinite(costs[p])) {
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                const auto y = row + dy, x = col + dx;
                if (!(dy || dx) || y < 0 || x < 0 || y >= rows || x >= cols) continue;
                const auto previous = input[y * cols + x];
                const float distance = __uint_as_float(unsigned(previous >> 32)) + costs[p];
                if (!isfinite(distance)) continue;
                const auto candidate = (detail::Label(__float_as_uint(distance)) << 32) | (unsigned(previous) + 1);
                if (candidate < best) best = candidate;
            }
        }
        output[p] = best;
        if (best != input[p]) atomicExch(changed, 1);
    }
}
}
std::vector<std::uint32_t> shortestPath(const float* costs, std::size_t rows, std::size_t cols,
                                      std::uint32_t source, std::uint32_t target) {
    const auto n = detail::validate(costs, rows, cols, source, target);
    if (!std::isfinite(costs[source]) || !std::isfinite(costs[target])) return {};
    if (source == target) return {source};
    std::vector<detail::Label> labels(n, detail::infinity); labels[source] = 0;
    int gpu = 0; device::check(cudaGetDevice(&gpu));
    device::Buffer weights(n * sizeof(float), gpu);
    device::Buffer first(n * sizeof(detail::Label), gpu), second(n * sizeof(detail::Label), gpu);
    device::Buffer changed(sizeof(int), gpu);
    const auto stream = device::stream();
    device::check(cudaMemcpyAsync(static_cast<float*>(weights.ptr), costs, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    device::check(cudaMemcpyAsync(static_cast<detail::Label*>(first.ptr), labels.data(), n * sizeof(detail::Label), cudaMemcpyHostToDevice, stream));
    auto* input = static_cast<detail::Label*>(first.ptr); auto* output = static_cast<detail::Label*>(second.ptr);
    const unsigned blocks = unsigned(std::min<std::size_t>((n + 255) / 256, 65535));
    int change = 1;
    while (change) {
        device::check(cudaMemsetAsync(static_cast<int*>(changed.ptr), 0, sizeof(int), stream));
        // Batch wavefront rounds to amortize host synchronization. The change flag
        // accumulates across the batch, so termination still requires a stable round.
        for (int round = 0; round < 8; ++round) {
            relax<<<blocks, 256, 0, stream>>>(static_cast<float*>(weights.ptr), rows, cols, input, output, static_cast<int*>(changed.ptr));
            device::check(cudaGetLastError());
            std::swap(input, output);
        }
        device::check(cudaMemcpyAsync(&change, static_cast<int*>(changed.ptr), sizeof(int), cudaMemcpyDeviceToHost, stream));
        device::complete();
    }
    device::check(cudaMemcpyAsync(labels.data(), input, n * sizeof(detail::Label), cudaMemcpyDeviceToHost, stream));
    device::complete();
    return detail::path(costs, rows, cols, source, target, labels);
}
}
