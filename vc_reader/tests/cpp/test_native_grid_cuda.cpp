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

        // Prepared pixels have no quad-validity requirement, even in a singleton row.
        {
            const float xyz[]{10.25f, 20.5f, 30.75f, 200, 200, 200, 1, 2, 3};
            const float normals[]{0, 0, 1, 0, 0, 1, 0, 0, 1};
            const std::uint8_t valid[]{1, 1, 0};
            vcr::grid::Points points{xyz, normals, 3, 1};
            vcr::grid::Request request;
            request.levels = {L0};
            request.offsets = {-1, 0, 1};
            request.pixels = valid;
            request.composite = vcr::composite::Options{};
            request.composite->method = vcr::composite::Method::Max;
            compare(vcr::grid::sample(points, request), vcr::grid::cuda::sample(points, request, {&fineGpu}),
                    0.0f, "prepared point layers");
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
        for (const auto mode : {vcr::grid::Interpolation::Linear, vcr::grid::Interpolation::Bicubic}) {
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

        // A sheared raster over the surface with vertex normals, sampled through a
        // general (rotated) level map.
        {
            std::vector<float> normals(grid.size());
            for (std::size_t k = 0; k < normals.size(); k += 3) {
                normals[k] = 0.28f;
                normals[k + 1] = -0.96f;
                normals[k + 2] = 0.0f;
            }
            vcr::grid::Surface surface({grid.data(), rows, cols, std::size_t(cols) * 3, normals.data()});
            const vcr::grid::Raster raster{2.1, 0.4, 0.11, 0.1, 200, 150, 0.03, -0.02};
            vcr::grid::Request request;
            request.levels = {{fine.get(), u8, {}, {3, -2, 1.5}, std::array<double, 9>{0.1, 0.2, 0.9, 0, 0.98, -0.2, 0.95, 0, -0.1}}};
            request.offsets = {-1.5f, 0.0f, 2.0f};
            for (const int order : {0, 1}) {
                request.order = order;
                compare(vcr::grid::sample(surface, raster, request), vcr::grid::cuda::sample(surface, raster, request, {&fineGpu}),
                        0.0f, order ? "general map linear" : "general map nearest");
            }
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

        // grid::sample dispatches: levels with readers sample on the GPU; a GPU pass that
        // fails (here: no brick fits the reader's budget) samples on the CPU, names the
        // failure and gives the bricks back.
        {
            vcr::grid::Request request;
            request.levels = {L0};
            request.order = 1;
            const auto cpu = vcr::grid::sample(plane, request);
            require(cpu.cudaFailure.empty(), "a CPU request reports no GPU failure");
            auto dispatched = store(path / "dispatched", 128, 64);
            vcr::reader_cuda::Reader dispatchedGpu(*dispatched, 64 << 20, 0);
            request.levels = {{dispatched.get(), u8, L0.scale, L0.offset, std::nullopt, &dispatchedGpu}};
            const auto gpu = vcr::grid::sample(plane, request);
            require(gpu.cudaFailure.empty(), "a GPU request that succeeds reports no failure");
            compare(cpu, gpu, 0.0f, "dispatched plane");
            request.cachedOnly = true;
            compare(cpu, vcr::grid::cuda::sample(plane, request, {&dispatchedGpu}), 0.0f, "the dispatched pass made bricks resident");
            request.cachedOnly = false;
            // Readers on only some levels: the CPU.
            request.levels = {{dispatched.get(), u8, L0.scale, L0.offset, std::nullopt, &dispatchedGpu}, L1};
            compare(vcr::grid::sample(plane, request), vcr::grid::cuda::sample(plane, request, {&dispatchedGpu, &coarseGpu}), 0.0f,
                    "mixed levels");
            require(vcr::grid::sample(plane, request).cudaFailure.empty(), "mixed levels sample on the CPU without a failure");

            vcr::reader_cuda::Reader tooSmall(*dispatched, 1, 0);
            request.levels = {{dispatched.get(), u8, L0.scale, L0.offset, std::nullopt, &tooSmall}};
            const auto fallen = vcr::grid::sample(plane, request);
            require(!fallen.cudaFailure.empty(), "a failed GPU pass is named");
            compare(cpu, fallen, 0.0f, "fallen back to the CPU");
            std::cout << "GPU failure fell back to the CPU: " << fallen.cudaFailure << '\n';
        }
        {
            // A blocking request whose load evicts bricks it found resident still has no holes.
            const auto root = path / "eviction";
            auto evicting = store(root, 128, 32);
            vcr::reader_cuda::Reader small(*evicting, 8 * 32 * 32 * 32, 0);
            small.load_bricks({0, 1, 2, 3});      // these bricks, the oldest: the plane's first row
            small.load_bricks({16, 17, 18, 19});  // and these, younger, which it does not touch
            const vcr::grid::Plane row{{0, 0, 16}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 128, 64};
            vcr::grid::Request request;
            vcr::grid::Level whole;
            whole.array = evicting.get();
            request.levels = {whole};
            const auto cpu = vcr::grid::sample(row, request);
            const auto gpu = vcr::grid::cuda::sample(row, request, {&small});
            compare(cpu, gpu, 0.0f, "a blocking request after an eviction");
            // Sampling a surface twice reuses its device copy and gives the same values.
            std::vector<float> xyz;
            for (int r = 0; r < 8; ++r)
                for (int c = 0; c < 8; ++c)
                    xyz.insert(xyz.end(), {float(c * 8), float(r * 8), 20.0f});
            vcr::grid::Surface sheet({xyz.data(), 8, 8, 24, nullptr, nullptr});
            const vcr::grid::Raster raster{0, 0, 0.5, 0.5, 15, 15};
            const auto cpuSheet = vcr::grid::sample(sheet, raster, request);
            for (int pass = 0; pass < 2; ++pass)
                compare(cpuSheet, vcr::grid::cuda::sample(sheet, raster, request, {&small}), 1e-3f, "a kept surface");
            vcr::grid::cuda::forgetKeptData();
            compare(cpuSheet, vcr::grid::cuda::sample(sheet, raster, request, {&small}), 1e-3f, "a surface uploaded again");
        }
        std::cout << "Native CUDA grid sampling passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
