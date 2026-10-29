// vcr::grid::cuda samples the same requests as the CPU tile loop on a GPU: from
// resident bricks alone (cache-only), loading the bricks the first level lacks
// (blocking), with layers, composites and level fallback.
#include <vc_reader/array.hpp>
#include <vc_reader/native/reader_cuda/reader.hpp>
#include <vc_reader/grid.hpp>
#include <vc_reader/native/cuda_available.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

void require(bool value, const std::string& message)
{
    if (!value)
        throw std::runtime_error(message);
}

std::uint8_t value(int z, int y, int x) { return std::uint8_t((3 * x + 5 * y + 7 * z) % 251); }

// A local store of n^3 voxels in chunks of `chunk`, every voxel `fill` when fill >= 0.
std::unique_ptr<vcr::Array> store(const std::filesystem::path& root, int n, int chunk, int fill = -1)
{
    namespace fs = std::filesystem;
    fs::create_directories(root);
    const int g = n / chunk;
    for (int cz = 0; cz < g; ++cz)
        for (int cy = 0; cy < g; ++cy)
            for (int cx = 0; cx < g; ++cx) {
                std::vector<std::uint8_t> data(std::size_t(chunk) * chunk * chunk);
                for (int z = 0; z < chunk; ++z)
                    for (int y = 0; y < chunk; ++y)
                        for (int x = 0; x < chunk; ++x)
                            data[std::size_t((z * chunk + y) * chunk + x)] =
                                fill >= 0 ? std::uint8_t(fill) : value(cz * chunk + z, cy * chunk + y, cx * chunk + x);
                std::ofstream o(root / (std::to_string(cz) + "." + std::to_string(cy) + "." + std::to_string(cx)),
                                std::ios::binary);
                o.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
            }
    vcr::ArraySpec spec;
    spec.shape = {n, n, n};
    spec.chunks = {chunk, chunk, chunk};
    spec.fill = {0};
    spec.identity = root.string();
    spec.layers.push_back({"array", root.string(), "", "", false});
    return std::make_unique<vcr::Array>(spec, std::make_shared<vcr::MemoryCache>(64 << 20));
}

void compare(const vcr::grid::Result& cpu, const vcr::grid::Result& gpu, float tolerance, const std::string& what)
{
    require(cpu.values.size() == gpu.values.size() && cpu.level == gpu.level, what + ": shape or levels differ");
    for (std::size_t i = 0; i < cpu.values.size(); ++i) {
        const float a = cpu.values[i], b = gpu.values[i];
        require((std::isnan(a) && std::isnan(b)) || std::abs(a - b) <= tolerance,
                what + ": value " + std::to_string(i) + " " + std::to_string(a) + " vs " + std::to_string(b));
    }
}

}  // namespace

int main()
{
    namespace fs = std::filesystem;
    if (!vcr::cuda::available())
        return 77;
    const auto path = fs::temp_directory_path() /
                      ("vcr-native-grid-cuda-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { fs::path path; ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); } } cleanup{path};
    try {
        cudaStream_t stream;
        vcr::device::check(cudaStreamCreate(&stream));
        vcr::device::StreamScope scope(stream);
        auto fine = store(path / "0", 128, 64), coarse = store(path / "1", 64, 64, 200);
        vcr::reader_cuda::Reader fineGpu(*fine, 64 << 20, 0), coarseGpu(*coarse, 64 << 20, 0);
        const vcr::ElementType u8{'u', 1, false};
        const vcr::grid::Level L0{fine.get(), u8, {1, 1, 1}, {0, 0, 0}};
        const vcr::grid::Level L1{coarse.get(), u8, {0.5, 0.5, 0.5}, {-0.25, -0.25, -0.25}};

        // A tilted plane through the volume, partly outside it; nearest and linear.
        vcr::grid::Plane plane{{10.3f, 20.6f, 30.2f}, {0.9f, 0.1f, 0.05f}, {-0.1f, 0.8f, 0.3f}, {0, 0.35f, -0.94f}, 150, 90};
        for (const int order : {0, 1}) {
            vcr::grid::Request request;
            request.levels = {L0};
            request.order = order;
            request.cachedOnly = true;
            auto gpu = vcr::grid::cuda::sample(plane, request, {&fineGpu});
            bool anyMissing = false;
            for (auto l : gpu.level)
                anyMissing |= l == vcr::grid::Result::kMissing;
            require(order == 1 || anyMissing, "nothing resident: samples inside the volume are missing");
            request.cachedOnly = false;
            gpu = vcr::grid::cuda::sample(plane, request, {&fineGpu});
            const auto cpu = vcr::grid::sample(plane, request);
            compare(cpu, gpu, 0.0f, order ? "plane linear" : "plane nearest");
            request.cachedOnly = true;  // now resident
            compare(cpu, vcr::grid::cuda::sample(plane, request, {&fineGpu}), 0.0f, "resident plane");
        }

        // A wavy surface with layers along its normal, reduced to their maximum.
        const int rows = 20, cols = 30;
        std::vector<float> grid(std::size_t(rows) * cols * 3);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                float* p = &grid[(std::size_t(r) * cols + c) * 3];
                p[0] = 4.0f * c;
                p[1] = 5.0f * r;
                p[2] = 60.0f + 8.0f * std::sin(0.3f * c) * std::cos(0.2f * r);
            }
        for (const auto mode : {vcr::grid::Interpolation::Linear, vcr::grid::Interpolation::Smooth}) {
            vcr::grid::Surface surface({grid.data(), rows, cols, std::size_t(cols) * 3}, {mode});
            const vcr::grid::Raster raster{1.3, 0.7, 0.11, 0.13, 220, 140};
            vcr::grid::Request request;
            request.levels = {L0};
            request.order = 0;
            request.offsets = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
            compare(vcr::grid::sample(surface, raster, request), vcr::grid::cuda::sample(surface, raster, request, {&fineGpu}),
                    0.0f, "surface layers");
            request.composite = vcr::composite::Options{vcr::composite::Method::Max};
            compare(vcr::grid::sample(surface, raster, request), vcr::grid::cuda::sample(surface, raster, request, {&fineGpu}),
                    0.0f, "surface max composite");
            request.order = 1;
            request.composite = vcr::composite::Options{vcr::composite::Method::Mean};
            compare(vcr::grid::sample(surface, raster, request), vcr::grid::cuda::sample(surface, raster, request, {&fineGpu}),
                    0.0f, "surface linear mean composite");
        }

        // Level fallback: a fresh fine reader has nothing resident, the coarse level does.
        auto fresh = store(path / "fresh", 128, 64);
        vcr::reader_cuda::Reader freshGpu(*fresh, 64 << 20, 0);
        vcr::grid::Request load;
        load.levels = {L1};
        load.order = 0;
        vcr::grid::cuda::sample(plane, load, {&coarseGpu});  // the coarse bricks become resident
        vcr::grid::Request cached;
        cached.levels = {{fresh.get(), u8, L0.scale, L0.offset}, L1};
        cached.order = 0;
        cached.cachedOnly = true;
        const auto r = vcr::grid::cuda::sample(plane, cached, {&freshGpu, &coarseGpu});
        std::size_t fromCoarse = 0;
        for (std::size_t q = 0; q < r.level.size(); ++q)
            if (r.level[q] == 1) {
                ++fromCoarse;
                require(r.values[q] == 200.0f, "coarse level value");
            }
        require(fromCoarse > 1000, "pixels inside the volume come from the coarse level");
        std::cout << "Native CUDA grid sampling passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
