// vcr::grid::cuda samples the same requests as the CPU tile loop on a GPU: from
// resident bricks alone (cache-only), loading the bricks the first level lacks
// (blocking), with layers, composites and level fallback.
#include <vc_reader/array.hpp>
#include <vc_reader/native/reader_cuda/reader.hpp>
#include <vc_reader/grid.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/native/cuda_available.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
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

void packedChannels(const std::filesystem::path& root)
{
    namespace fs = std::filesystem;
    fs::create_directories(root);
    // Two channels per compressed chunk, including a nonzero channel-chunk
    // index, an array edge and a missing chunk with nonzero fill.
    vcr::ArraySpec spec;
    spec.shape = {4, 29, 30, 31}; spec.chunks = {2, 16, 16, 16};
    spec.fill = {17}; spec.identity = root.string();
    spec.layers.push_back({"array", root.string(), "zstd", "", false});
    for (int cc=0; cc<2; ++cc) for (int z=0; z<2; ++z)
        for (int y=0; y<2; ++y) for (int x=0; x<2; ++x) {
            if (z==1 && y==1 && x==1) continue;
            std::vector<std::uint8_t> raw(2*16*16*16);
            for (int c=0; c<2; ++c) for (int k=0; k<16*16*16; ++k)
                raw[c*16*16*16+k] = std::uint8_t((19*(2*cc+c)+7*z+11*y+13*x+k)%251);
            const auto compressed = vcr::encode(vcr::Codec::Zstd, {}, raw.data(), raw.size(), 1, spec.chunks);
            std::ofstream file(root/(std::to_string(cc)+"."+std::to_string(z)+"."+std::to_string(y)+"."+std::to_string(x)),std::ios::binary);
            file.write(reinterpret_cast<const char*>(compressed.data()),compressed.size());
        }
    auto cache = std::make_shared<vcr::MemoryCache>(8<<20);
    vcr::Array array(spec,cache);
    auto pool = std::make_shared<vcr::reader_cuda::Pool>(8<<20,0);
    vcr::grid::Plane plane{{-1.25f, -0.5f, 15.4f},{1,0,0},{0,1,0},{0,0,1},34,34};
    std::uint64_t firstChannelReads=0;
    for (int c=0; c<4; ++c) {
        vcr::reader_cuda::Reader reader(array,pool,c);
        require(vcr::reader_cuda::Reader::brick_bytes(array,c)==16*16*16,"scalar brick size excludes channels");
        vcr::grid::Level level{&array,{'u',1,false}}; level.channel=c;
        vcr::grid::Request request; request.levels={level}; request.offsets={-1,0,1};
        for (int order : {0,1}) {
            request.order=order;
            auto cpu=vcr::grid::sample(plane,request);
            request.levels[0].cuda=&reader;
            const auto gpu=vcr::grid::sample(plane,request);
            require(gpu.cudaFailure.empty() && reader.stats()[3]>0,"packed dispatch must use CUDA");
            compare(cpu,gpu,1e-3f,"packed plane channel "+std::to_string(c));
            request.levels[0].cuda=nullptr;
        }
        // Channel 1 reuses channel 0's *whole* decoded storage chunks.
        if (c==0) firstChannelReads=array.stats().disk_hits.load();
        if (c==1) require(array.stats().disk_hits==firstChannelReads,"channels share decoded packed cache entries");
        // Both GPU and CPU routes of the generic reader consume XYZ-independent
        // ZYX triples. Include a chunk boundary, volume edge and missing chunk.
        const double points[]{15.2,15.3,15.4, 16.7,17.1,18.8, -0.4,2,3, 28.5,29.5,30.5};
        for (int order : {0,1}) {
            auto cpu=reader.points(points,4,64,{0,0,0},2,order,"cpu");
            auto gpu=reader.points(points,4,64,{0,0,0},2,order,"chunks");
            const auto bytes=4*(order ? sizeof(float) : 1);
            std::vector<std::uint8_t> a(bytes),b(bytes);
            vcr::device::check(cudaMemcpy(a.data(),cpu.storage->ptr,bytes,cudaMemcpyDeviceToHost));
            vcr::device::check(cudaMemcpy(b.data(),gpu.storage->ptr,bytes,cudaMemcpyDeviceToHost));
            if (!order) require(a==b,"packed nearest CPU route");
            else for (int i=0;i<4;++i) {
                float x,y; std::memcpy(&x,a.data()+4*i,4); std::memcpy(&y,b.data()+4*i,4);
                require(std::abs(x-y)<1e-3f,"packed linear CPU route");
            }
        }
        // Region and affine CPU routes use the same channel selection as brick
        // uploads, including the one-channel leading extent of read_many.
        for (int kind : {0,1}) for (int order : {0,1}) {
            vcr::reader_cuda::Grid g;
            for (int a=0;a<3;++a) { g.shape[a]=3; g.base[a]=14; g.axes[a*3+a]=1; g.origin[a]=0.25f; }
            auto cpu=reader.grids({g},kind,order,true,"cpu");
            auto gpu=reader.grids({g},kind,order,true,"chunks");
            std::vector<float> a(27),b(27);
            vcr::device::check(cudaMemcpy(a.data(),cpu.storage->ptr,108,cudaMemcpyDeviceToHost));
            vcr::device::check(cudaMemcpy(b.data(),gpu.storage->ptr,108,cudaMemcpyDeviceToHost));
            for (int i=0;i<27;++i) require(std::abs(a[i]-b[i])<1e-3f,"packed region/affine CPU route");
        }
        const std::int64_t voxels[]{0,1,2,16,17,18};
        auto cpuVoxels=reader.points(voxels,2,64,{0,0,0},3,0,"cpu");
        auto gpuVoxels=reader.points(voxels,2,64,{0,0,0},3,0,"chunks");
        std::uint8_t a[2],b[2];
        vcr::device::check(cudaMemcpy(a,cpuVoxels.storage->ptr,2,cudaMemcpyDeviceToHost));
        vcr::device::check(cudaMemcpy(b,gpuVoxels.storage->ptr,2,cudaMemcpyDeviceToHost));
        require(std::equal(a,a+2,b),"packed integer point CPU route");
        vcr::reader_cuda::Reader small(array,1,0,c);
        request.levels[0].cuda=&small;
        const auto fallback=vcr::grid::sample(plane,request);
        require(!fallback.cudaFailure.empty(),"packed budget failure is reported");
        request.levels[0].cuda=nullptr;
        compare(vcr::grid::sample(plane,request),fallback,0,"packed budget fallback");
        request.levels[0].cuda=&reader;
        request.levels[0].channel=(c+1)%4;
        bool rejected=false;
        try { vcr::grid::cuda::sample(plane,request,{&reader}); }
        catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"mismatched GPU channel rejected");
    }
    for (int channel : {-1,4}) {
        bool rejected=false;
        try { vcr::reader_cuda::Reader invalid(array,pool,channel); }
        catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"invalid GPU channel rejected");
    }
    std::cout << "Packed CUDA channels: shared decoded cache, channel chunks, fill, edges, CPU fallback passed\n";
}

void realPackedChannels(const char* location)
{
    vcr::zarr::OpenOptions options;
    options.memoryCache=std::make_shared<vcr::MemoryCache>(64<<20);
    auto opened=vcr::zarr::openArray(location,options);
    require(opened.meta.dtype=="|u1" && opened.meta.shape.size()==4,"real test needs uint8 CZYX");
    auto pool=std::make_shared<vcr::reader_cuda::Pool>(64<<20,0);
    // PHerc0332 real annotated fiber seed, base coordinates / 4 at level 2.
    const vcr::grid::Plane plane{{6160.293945f/4-32,8019.695801f/4-32,26137.808594f/4},
        {1,0,0},{0,1,0},{0,0,1},64,64};
    for (std::int64_t c=0;c<opened.meta.shape[0];++c) {
        vcr::reader_cuda::Reader reader(*opened.array,pool,c);
        vcr::grid::Request request;
        vcr::grid::Level level{opened.array.get(),opened.meta.type}; level.channel=c;
        request.levels={level}; request.offsets={-0.5f,0,0.5f};
        for (int order : {0,1}) {
            request.order=order;
            const auto cpu=vcr::grid::sample(plane,request);
            request.levels[0].cuda=&reader;
            vcr::grid::Frame frame; frame.plane=plane;
            vcr::grid::Pending pending;
            vcr::grid::sampleCached(frame,request,&pending);
            require(vcr::grid::load(pending).empty(),"real progressive CUDA load");
            const auto gpu=vcr::grid::sampleCached(frame,request,nullptr);
            require(gpu.cudaFailure.empty() && reader.stats()[3]>0,"real channel must sample on GPU");
            compare(cpu,gpu,1e-3f,"real compound channel "+std::to_string(c));
            request.levels[0].cuda=nullptr;
        }
    }
    std::cout << "Real PHerc0332 CUDA parity/progressive loading passed for " << opened.meta.shape[0] << " channels: " << location << '\n';
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
        packedChannels(path / "packed");
        if (const char* real=std::getenv("VC_READER_COMPOUND_CUDA_TEST_ARRAY")) realPackedChannels(real);
        auto fine = store(path / "0", 128, 64), coarse = store(path / "1", 64, 64, 200);
        {
            const std::int64_t p[]{1,2,3}; std::uint8_t output;
            const vcr::PointCoordinates spatial{p,std::size_t(3)};
            bool rejected=false;
            try { fine->sample_nearest_coordinates(vcr::ChannelCoordinates{spatial,0},1,&output); }
            catch (const std::invalid_argument&) { rejected=true; }
            require(rejected,"channel coordinates cannot enter a three-dimensional sampler");
        }
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
            {
                // Cancelled: the GPU pass reads no brick and does not fall back to the CPU.
                const std::atomic<bool> cancelled{true};
                auto stopped = request;
                stopped.cancel = &cancelled;
                bool threw = false;
                try { vcr::grid::sample(plane, stopped); } catch (const vcr::Cancelled&) { threw = true; }
                stopped.cancel = nullptr;
                stopped.cachedOnly = true;
                const auto after = vcr::grid::cuda::sample(plane, stopped, {&dispatchedGpu});
                require(threw && std::count(after.level.begin(), after.level.end(), vcr::grid::Result::kMissing) > 0,
                        "a cancelled GPU blocking sample throws Cancelled and makes nothing resident");
            }
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
