// Computed arrays and region reads through the chunk service: a derived field (a
// max filter with a border) computed per chunk from a stored array read through the
// service; one local worker cannot deadlock on nested reads; chunks persist on disk
// and are reused by a later open of the same identity; mismatched persisted arrays
// are refused; fill chunks; errors; point sampling of the computed array.
#include "vc_reader/computed.hpp"
#include "vc_reader/chunks.hpp"
#include "vc_reader/zarr.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace vcr;

namespace {
void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

constexpr int Z = 30, Y = 26, X = 22, C = 8, B = 2;  // array, chunk and border sizes

std::uint8_t field(std::int64_t z, std::int64_t y, std::int64_t x)
{
    if (z < 0 || y < 0 || x < 0 || z >= Z || y >= Y || x >= X || z >= 16)
        return 0;  // fill: outside, and the stored array's top slabs are never written
    return std::uint8_t((z * 37 + y * 11 + x * 5) % 200 + 1);
}

// The maximum over the (2B+1)^3 neighbourhood, fill outside the array.
std::uint8_t maxFiltered(std::int64_t z, std::int64_t y, std::int64_t x)
{
    std::uint8_t m = 0;
    for (int dz = -B; dz <= B; ++dz)
        for (int dy = -B; dy <= B; ++dy)
            for (int dx = -B; dx <= B; ++dx)
                m = std::max(m, field(z + dz, y + dy, x + dx));
    return m;
}

struct Fixture {
    fs::path root;
    std::shared_ptr<Array> stored;
    Fixture()
        : root(fs::temp_directory_path() /
               ("vc-reader-computed-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
    {
        zarr::CreateOptions create;
        create.shape = {Z, Y, X};
        create.chunks = {C, C, C};
        create.compressor = "zstd";
        auto writer = zarr::Writer::create(root / "input.zarr", create);
        std::vector<std::uint8_t> data(std::size_t(16) * Y * X);
        for (int z = 0; z < 16; ++z)
            for (int y = 0; y < Y; ++y)
                for (int x = 0; x < X; ++x)
                    data[(std::size_t(z) * Y + y) * X + x] = field(z, y, x);
        writer.writeRegion({0, 0, 0}, {16, Y, X}, data.data());
        stored = std::shared_ptr<Array>(zarr::openArray((root / "input.zarr").string(), {}).array.release());
    }
    ~Fixture()
    {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};

// A max-filter computed array reading its input through `service`.
struct Filter {
    chunks::Service& service;
    chunks::SourceId input;
    std::atomic<int> computed{0};
    bool operator()(const Index& chunk, std::uint8_t* out)
    {
        ++computed;
        const std::int64_t S = C + 2 * B;
        std::vector<std::uint8_t> region(std::size_t(S * S * S));
        service.read(input, 0, {chunk[0] * C - B, chunk[1] * C - B, chunk[2] * C - B}, {S, S, S}, region.data());
        bool any = false;
        for (int z = 0; z < C; ++z)
            for (int y = 0; y < C; ++y)
                for (int x = 0; x < C; ++x) {
                    std::uint8_t m = 0;
                    for (int dz = 0; dz <= 2 * B; ++dz)
                        for (int dy = 0; dy <= 2 * B; ++dy)
                            for (int dx = 0; dx <= 2 * B; ++dx)
                                m = std::max(m, region[((z + dz) * S + (y + dy)) * S + (x + dx)]);
                    out[(z * C + y) * C + x] = m;
                    any |= m != 0;
                }
        return any;
    }
};

computed::Options options(const fs::path& cache)
{
    computed::Options o;
    o.shape = {Z, Y, X};
    o.chunks = {C, C, C};
    o.identity = "max-filter radius 2 of input.zarr";
    o.cacheDir = cache;
    return o;
}

void regionReads(const Fixture& f)
{
    chunks::Service service({.decodedBytes = 1 << 20, .remoteWorkers = 1, .localWorkers = 1});
    const auto source = service.addSource({f.stored});
    const Index start{-3, 5, 15}, size{30, 9, 10};  // crosses chunks, the fill slabs and the edges
    std::vector<std::uint8_t> got(std::size_t(size[0] * size[1] * size[2]));
    service.read(source, 0, start, size, got.data());
    for (int z = 0; z < size[0]; ++z)
        for (int y = 0; y < size[1]; ++y)
            for (int x = 0; x < size[2]; ++x)
                require(got[(std::size_t(z) * size[1] + y) * size[2] + x] == field(start[0] + z, start[1] + y, start[2] + x),
                        "region read");
    std::vector<std::uint8_t> outside(8, 9);
    service.read(source, 0, {100, 0, 0}, {2, 2, 2}, outside.data());
    require(outside == std::vector<std::uint8_t>(8, 0), "a region outside the array reads as fill");
    auto broken = computed::open({.shape = {8, 8, 8}, .chunks = {8, 8, 8}},
                                 [](const Index&, std::uint8_t*) -> bool { throw std::runtime_error("no input"); });
    const auto failing = service.addSource({std::shared_ptr<Array>(std::move(broken))});
    bool threw = false;
    try {
        service.read(failing, 0, {0, 0, 0}, {1, 1, 1}, outside.data());
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("no input") != std::string::npos;
    }
    require(threw, "region reads propagate chunk errors");
}

void computedArrays(const Fixture& f)
{
    const fs::path cache = f.root / "cache";
    // One local worker: computing a chunk reads its inputs through the same service.
    chunks::Service service({.decodedBytes = 1 << 20, .remoteWorkers = 1, .localWorkers = 1});
    const auto input = service.addSource({f.stored});
    Filter filter{service, input};
    auto array = std::shared_ptr<Array>(
        computed::open(options(cache), [&](const Index& c, std::uint8_t* out) { return filter(c, out); }));
    const auto derived = service.addSource({array});
    std::vector<std::uint8_t> all(std::size_t(Z) * Y * X);
    service.read(derived, 0, {0, 0, 0}, {Z, Y, X}, all.data());
    for (int z = 0; z < Z; ++z)
        for (int y = 0; y < Y; ++y)
            for (int x = 0; x < X; ++x)
                require(all[(std::size_t(z) * Y + y) * X + x] == maxFiltered(z, y, x), "computed chunk values");
    const int grid = ((Z + C - 1) / C) * ((Y + C - 1) / C) * ((X + C - 1) / C);
    require(filter.computed == grid, "each chunk computed once");
    require(service.get({derived, 0, 3, 0, 0}).status == chunks::Status::Fill, "an all-fill chunk is Fill");

    // Direct region reads of the array (no service) give the same values.
    std::vector<std::uint8_t> direct(all.size());
    array->read_many({{0, 0, 0}}, {{Z, Y, X}}, {direct.data()});
    require(direct == all, "array reads of computed chunks");
    array->flush();
    require(array->stats().write_errors == 0, "persisted without errors");

    // Point sampling of the computed array.
    const double zyx[] = {5.0, 4.25, 3.5, 10, 10, 10};
    float values[2];
    array->sample_linear(zyx, 2, ElementType{'u', 1, false}, false, values);
    const float y0 = maxFiltered(5, 4, 3) + 0.25f * (maxFiltered(5, 5, 3) - maxFiltered(5, 4, 3));
    const float y1 = maxFiltered(5, 4, 4) + 0.25f * (maxFiltered(5, 5, 4) - maxFiltered(5, 4, 4));
    require(std::abs(values[0] - (y0 + 0.5f * (y1 - y0))) < 1e-4f, "a sample interpolates computed voxels");
    require(values[1] == float(maxFiltered(10, 10, 10)), "a voxel sample is the voxel");

    // A later open of the same identity reads every chunk from disk.
    auto reopened = computed::open(options(cache), [](const Index&, std::uint8_t*) -> bool {
        throw std::runtime_error("persisted chunks are not recomputed");
    });
    std::vector<std::uint8_t> again(all.size());
    reopened->read_many({{0, 0, 0}}, {{Z, Y, X}}, {again.data()});
    require(again == all, "persisted chunks reused");
    require(fs::exists(computed::cachePath(options(cache)) / "3.0.0.missing"), "fill chunks persist as markers");

    auto other = options(cache);
    other.dtype = "<u2";
    bool refused = false;
    try {
        computed::open(other, [](const Index&, std::uint8_t*) { return false; });
    } catch (const std::runtime_error&) {
        refused = true;
    }
    require(refused, "a persisted identity with another dtype is refused");
    other = options(cache);
    other.identity = "max-filter radius 3 of input.zarr";
    require(computed::cachePath(other) != computed::cachePath(options(cache)), "identities persist apart");

    // Lossless compressed persistence.
    auto packed = options(f.root / "packed");
    packed.compressCache = true;
    Filter again2{service, input};
    {
        auto a = computed::open(packed, [&](const Index& c, std::uint8_t* out) { return again2(c, out); });
        a->read_many({{0, 0, 0}}, {{Z, Y, X}}, {again.data()});
        require(again == all, "compressed persistence values");
    }
    auto b = computed::open(packed, [](const Index&, std::uint8_t*) -> bool { throw std::runtime_error("recomputed"); });
    b->read_many({{0, 0, 0}}, {{Z, Y, X}}, {again.data()});
    require(again == all, "compressed persisted chunks reused");
}
}  // namespace

int main()
{
    try {
        Fixture f;
        regionReads(f);
        computedArrays(f);
        std::cout << "computed arrays and service region reads passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
