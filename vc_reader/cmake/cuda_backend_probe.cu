// Compile-only probe: what the CUDA backend needs from the CUDA toolkit's CUB and libcu++.
// Kept in step with src/cpp/components/kernels.cu (cuda::std::plus in a scan) and
// src/cpp/transforms/cuda/sort_keys.cuh (SortKeys with a decomposer over a tuple of references).
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cstddef>
#include <cstdint>

struct ProbeKey { unsigned long long value; unsigned sample; };
struct ProbeDecomposer {
    __host__ __device__ auto operator()(ProbeKey& key) const { return ::cuda::std::tie(key.sample, key.value); }
};

cudaError_t vc_reader_cuda_probe(void* scratch, std::size_t& bytes, ProbeKey* in, ProbeKey* out,
                                 const std::uint8_t* flags, int* ranks, std::size_t n, cudaStream_t stream) {
    cudaError_t e = cub::DeviceRadixSort::SortKeys(scratch, bytes, in, out, n, ProbeDecomposer{}, 0, 96, stream);
    if (e != cudaSuccess) return e;
    return cub::DeviceScan::InclusiveScan(scratch, bytes, flags, ranks, ::cuda::std::plus<int>{}, int(n), stream);
}

int main() { return 0; }  // try_compile builds an executable; nothing here runs
