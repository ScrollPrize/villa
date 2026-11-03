#include <vc_reader/reader.hpp>
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
        vcr::CtReader volume(argv[1], std::stoi(argv[2]));
        const std::array<double, 3> point{std::stod(argv[3]), std::stod(argv[4]), std::stod(argv[5])};
        const auto cpu = volume.sheetNormals(std::span(&point, 1)).front();
        std::cout << "CPU XYZ/gap: " << cpu[0] << ' ' << cpu[1] << ' ' << cpu[2] << ' ' << cpu[3] << '\n';
#ifdef VC_READER_CUDA
        int devices = 0;
        vcr::device::check(cudaGetDeviceCount(&devices));
        if (!devices) throw std::runtime_error("no CUDA device available");
        std::array<int64_t, 3> start{};
        float center[4]{0, 0, 0, 1};
        for (int axis = 0; axis < 3; ++axis) {
            const double lattice = std::floor(point[axis] / 3) * 3;
            start[2 - axis] = int64_t(lattice) - 54;
            center[axis] = std::min(float(point[axis] - lattice) + 48.f, std::nextafter(51.f, 48.f));
            center[3] *= start[2 - axis] >= 0 && start[2 - axis] + 109 <= volume.shape()[2 - axis];
        }
        if (!center[3]) throw std::invalid_argument("point requires complete 109-cubed CT support");
        // The CUDA reader borrows the Array; volume must outlive it.
        vcr::reader_cuda::Reader gpu(volume.array(), 64 * 1024 * 1024, 0);
        vcr::reader_cuda::Grid request;
        request.count = 109 * 109 * 109;
        for (int axis = 0; axis < 3; ++axis) {
            request.base[axis] = start[axis];
            request.shape[axis] = 109;
        }
        // kind=0 denotes an integer region, order=0 a byte-exact read.
        auto patch = gpu.grids({request}, 0, 0, false, "chunks");
        auto result = vcr::sheet_normal::cuda(patch.storage->ptr, false, center, 1, 0);
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
