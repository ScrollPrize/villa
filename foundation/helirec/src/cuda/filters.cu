#include "cuda_utils.cuh"
#include "helirec/filters.hpp"

namespace helirec::cuda {

namespace {
template <int R>
__global__ void median_kernel(const float* __restrict__ src, float* __restrict__ dst, int n_frames, int H, int W) {
    constexpr int K = (2 * R + 1) * (2 * R + 1);
    const size_t total = static_cast<size_t>(n_frames) * H * W;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int x = static_cast<int>(i % W), y = static_cast<int>((i / W) % H);
        const float* fr = src + (i / (static_cast<size_t>(H) * W)) * H * W;
        float v[K];
        int k = 0;
        for (int dy = -R; dy <= R; ++dy) {
            const int yy = min(max(y + dy, 0), H - 1);
            for (int dx = -R; dx <= R; ++dx) v[k++] = fr[static_cast<size_t>(yy) * W + min(max(x + dx, 0), W - 1)];
        }
        // partial selection sort up to the median
        for (int a = 0; a <= K / 2; ++a) {
            int m = a;
            for (int b = a + 1; b < K; ++b)
                if (v[b] < v[m]) m = b;
            const float t = v[a]; v[a] = v[m]; v[m] = t;
        }
        dst[i] = v[K / 2];
    }
}
}  // namespace

void median_frames(float* frames, int n_frames, int H, int W, int radius, void* stream) {
    cudaStream_t s = as_stream(stream);
    const size_t n = static_cast<size_t>(n_frames) * H * W;
    thread_local DeviceBuffer<float> tmp;
    tmp.ensure(n);
    HELIREC_CUDA_CHECK(cudaMemcpyAsync(tmp.data(), frames, n * sizeof(float), cudaMemcpyDeviceToDevice, s));
    const unsigned grid = static_cast<unsigned>(std::min<size_t>((n + 255) / 256, 65535u * 16u));
    switch (radius) {
        case 1: median_kernel<1><<<grid, 256, 0, s>>>(tmp.data(), frames, n_frames, H, W); break;
        case 2: median_kernel<2><<<grid, 256, 0, s>>>(tmp.data(), frames, n_frames, H, W); break;
        case 3: median_kernel<3><<<grid, 256, 0, s>>>(tmp.data(), frames, n_frames, H, W); break;
        default: throw Error("median: radius must be 1..3");
    }
    HELIREC_CUDA_CHECK(cudaGetLastError());
}

}  // namespace helirec::cuda
