#include <vc_reader/array.hpp>
#include <vc_reader/reader.hpp>
#include <vc_reader/sampling.hpp>
#include <vc_reader/zarr.hpp>
#ifdef VC_READER_CUDA
#include <vc_reader/cuda_reader.hpp>
#endif
#include <cmath>
#include <iostream>

// Coordinates are XYZ voxels in the selected level. No Python runtime is used.
int main(int argc, char** argv)
{
    if (argc != 6) {
        std::cerr << "Usage: vc_reader_ct_normals VOLUME LEVEL X Y Z\n";
        return 2;
    }
    try {
        vcr::zarr::OpenOptions options;
        options.memoryCache = std::make_shared<vcr::MemoryCache>(64 * 1024 * 1024);
        auto volume = vcr::zarr::openPyramidLevel(argv[1], std::stoi(argv[2]), options);
        const std::array<double, 3> point{std::stod(argv[3]), std::stod(argv[4]), std::stod(argv[5])};
        std::array<float, 4> cpu{};
        vcr::sheetNormals(volume, point.data(), 1, cpu.data());
        std::cout << "CPU XYZ/gap: " << cpu[0] << ' ' << cpu[1] << ' ' << cpu[2] << ' ' << cpu[3] << '\n';
#ifdef VC_READER_CUDA
        int devices = 0;
        vcr::device::check(cudaGetDeviceCount(&devices));
        if (!devices) throw std::runtime_error("no CUDA device available");
        // The CUDA reader borrows the Array; volume must outlive it.
        vcr::reader_cuda::Reader gpu(*volume.array, 64 * 1024 * 1024, 0);
        const std::int64_t shape[3]{volume.meta.shape[0], volume.meta.shape[1], volume.meta.shape[2]};
        auto result = vcr::sheet_normal::sample_cuda(gpu, shape, point.data(), 1, 0);
        std::array<float, 4> cuda{};
        vcr::device::check(cudaMemcpyAsync(cuda.data(), result->ptr, sizeof(cuda),
                                          cudaMemcpyDeviceToHost, vcr::device::stream()));
        vcr::device::complete();
        std::cout << "CUDA XYZ/gap: " << cuda[0] << ' ' << cuda[1] << ' ' << cuda[2] << ' ' << cuda[3] << '\n';
        const float alignment = std::abs(cpu[0]*cuda[0] + cpu[1]*cuda[1] + cpu[2]*cuda[2]);
        if ((cpu[3] > 0) != (cuda[3] > 0) || std::abs(cpu[3] - cuda[3]) > 2e-5f ||
            (cpu[3] > 0 && alignment < .99998f)) {
            throw std::runtime_error("CPU/CUDA normal comparison failed");
        }
#endif
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
