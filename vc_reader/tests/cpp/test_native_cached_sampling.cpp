// Sampling from the caches alone: Array::sample_nearest_cached / sample_linear_cached
// sample what memory and the disk caches hold, flag the points only the source (a
// remote, the array's own store) could supply, and never read the source.
#include <vc_reader/array.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

void require(bool value, const std::string& message)
{
    if (!value) throw std::runtime_error(message);
}

}  // namespace

int main()
{
    namespace fs = std::filesystem;
    const auto path = fs::temp_directory_path() / ("vcr-native-cached-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { fs::path path; ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); } } cleanup{path};
    try {
        // Shape (4, 4, 12), chunks 4^3. The whole volume as a local store (the blocking
        // reference), and a network volume whose decoded disk cache starts with chunk 0.
        const auto store = path / "store", cache = path / "cache";
        fs::create_directories(store);
        fs::create_directories(cache);
        const auto value_at = [](int z, int y, int x) { return std::uint8_t(z * 48 + y * 12 + x); };
        const auto write_chunk = [&](const fs::path& root, int cx) {
            std::vector<std::uint8_t> chunk(64);
            for (int z = 0; z < 4; ++z)
                for (int y = 0; y < 4; ++y)
                    for (int x = 0; x < 4; ++x)
                        chunk[(z * 4 + y) * 4 + x] = value_at(z, y, cx * 4 + x);
            std::ofstream output(root / ("0.0." + std::to_string(cx)), std::ios::binary);
            output.write(reinterpret_cast<const char*>(chunk.data()), std::streamsize(chunk.size()));
        };
        write_chunk(store, 0);
        write_chunk(store, 1);  // chunk 0.0.2 is not stored
        write_chunk(cache, 0);

        vcr::ArraySpec spec;
        spec.shape = {4, 4, 12};
        spec.chunks = {4, 4, 4};
        spec.fill = {7};
        spec.identity = store.string();
        spec.layers.push_back({"array", store.string(), "", "", false});
        vcr::Array reference(spec, std::make_shared<vcr::MemoryCache>(1 << 20));

        vcr::ArraySpec remote = spec;
        remote.identity = "remote";
        remote.layers = {{"decoded", cache.string(), "", "", false}};
        remote.remote_url = "http://127.0.0.1:9/never-fetched";  // nothing listens: a fetch fails
        remote.http.retries = 0;
        vcr::Array array(remote, std::make_shared<vcr::MemoryCache>(1 << 20));
        require(array.remote(), "the network volume has a remote");
        const vcr::ElementType u8{'u', 1, false};

        // Points (z, y, x): in chunk 0; on the chunk 0/1 border (linear needs both); in
        // chunk 1; in the unstored chunk 2; outside the array; a run of 64 points in chunk 1.
        std::vector<double> coords{1.25, 2.5, 1.5,  1.0, 1.0, 3.5,  2.0, 2.25, 5.75,
                                   0.5, 0.5, 9.0,   -5.0, 0.0, 0.0};
        const std::size_t singles = coords.size() / 3;
        for (int i = 0; i < 64; ++i) {
            coords.push_back(i % 4 * 0.75);
            coords.push_back(i / 16 * 0.75);
            coords.push_back(4.0 + (i / 4 % 4) * 0.75);
        }
        const std::size_t n = coords.size() / 3;

        std::vector<float> linear(n), cachedLinear(n);
        std::vector<std::uint8_t> nearest(n), cachedNearest(n), missing(n);
        reference.sample_linear(coords.data(), n, u8, false, linear.data());
        reference.sample_nearest(coords.data(), n, nearest.data());
        const float sentinel = -12345.f;
        const auto no_fetch = [&] {
            require(array.stats().remote_fetches == 0, "cache-only reads fetched");
        };

        // Chunk 0 comes from the disk cache; chunks 1 and 2 need the remote and are missing,
        // their outputs untouched; the outside point needs no chunk.
        std::fill(cachedLinear.begin(), cachedLinear.end(), sentinel);
        array.sample_linear_cached(coords.data(), n, u8, false, cachedLinear.data(), missing.data());
        no_fetch();
        require(array.stats().disk_hits > 0, "chunk 0 read from the disk cache");
        require(!missing[0] && cachedLinear[0] == linear[0], "disk-cached point");
        require(missing[1] && cachedLinear[1] == sentinel, "border point with a neighbour only the remote has");
        require(missing[2] && missing[3] && cachedLinear[2] == sentinel, "remote-only points");
        require(!missing[4] && cachedLinear[4] == 7.f, "outside point reads as fill");
        for (std::size_t p = singles; p < n; ++p)
            require(missing[p] && cachedLinear[p] == sentinel, "run in a remote-only chunk");
        std::fill(cachedNearest.begin(), cachedNearest.end(), std::uint8_t{200});
        array.sample_nearest_cached(coords.data(), n, cachedNearest.data(), missing.data());
        no_fetch();
        require(!missing[0] && cachedNearest[0] == nearest[0], "disk-cached nearest point");
        require(missing[2] && missing[3] && cachedNearest[2] == 200, "remote-only nearest points");
        require(!missing[4] && cachedNearest[4] == 7, "outside nearest point");
        for (std::size_t p = singles; p < n; ++p)
            require(missing[p] && cachedNearest[p] == 200, "nearest run in a remote-only chunk");

        // cached_chunk: the disk cache holds chunk 0, only the remote has chunk 1; a local
        // store's chunks are its source.
        require(array.cached_chunk({0, 0, 0}).first && array.cached_chunk({0, 0, 0}).second, "disk-cached chunk");
        require(!array.cached_chunk({0, 0, 1}).first, "remote-only chunk is not cached");
        {
            vcr::Array store(spec, std::make_shared<vcr::MemoryCache>(1 << 20));
            require(!store.cached_chunk({0, 0, 0}).first, "a local store's chunk is its source");
            store.chunk({0, 0, 0});
            require(store.cached_chunk({0, 0, 0}).first, "after a read it is in memory");
        }
        no_fetch();

        // The other chunks arrive in the disk cache (chunk 2 as known-missing): every point
        // equals the blocking reference and nothing is fetched.
        write_chunk(cache, 1);
        std::ofstream(cache / "0.0.2.missing").put('\n');
        array.sample_linear_cached(coords.data(), n, u8, false, cachedLinear.data(), missing.data());
        for (std::size_t p = 0; p < n; ++p)
            require(!missing[p] && cachedLinear[p] == linear[p], "linear point " + std::to_string(p));
        array.sample_nearest_cached(coords.data(), n, cachedNearest.data(), missing.data());
        for (std::size_t p = 0; p < n; ++p)
            require(!missing[p] && cachedNearest[p] == nearest[p], "nearest point " + std::to_string(p));
        no_fetch();

        // A local store is the array's source: its chunks are missing until a blocking read
        // brings them into memory, and the cache-only read does not touch the store.
        vcr::Array local(spec, std::make_shared<vcr::MemoryCache>(1 << 20));
        local.sample_linear_cached(coords.data(), n, u8, false, cachedLinear.data(), missing.data());
        require(local.stats().disk_hits == 0, "cache-only read of a local store read the store");
        for (std::size_t p = 0; p < n; ++p)
            require(missing[p] == (p != 4), "local store point " + std::to_string(p) + " before loading");
        local.sample_linear(coords.data(), n, u8, false, linear.data());
        local.sample_linear_cached(coords.data(), n, u8, false, cachedLinear.data(), missing.data());
        for (std::size_t p = 0; p < n; ++p)
            require(!missing[p] && cachedLinear[p] == linear[p], "local store point " + std::to_string(p) + " loaded");

        // An array without any source reads as fill, never missing.
        vcr::ArraySpec sourceless = spec;
        sourceless.identity = "sourceless";
        sourceless.layers.clear();
        vcr::Array empty(sourceless, std::make_shared<vcr::MemoryCache>(1 << 20));
        empty.sample_nearest_cached(coords.data(), n, cachedNearest.data(), missing.data());
        for (std::size_t p = 0; p < n; ++p)
            require(!missing[p] && cachedNearest[p] == 7, "sourceless point " + std::to_string(p));

        // Non-finite points read as NaN without a chunk, as in the blocking call.
        const double nan_point[3]{std::nan(""), 0.0, 0.0};
        float nan_value = 0;
        std::uint8_t nan_missing = 1;
        array.sample_linear_cached(nan_point, 1, u8, false, &nan_value, &nan_missing);
        require(std::isnan(nan_value) && !nan_missing, "non-finite point");
        // A corrupt chunk (here a truncated raw file) is noted and skipped: it reads as
        // not stored, through the memory-cached path and both streaming read modes.
        {
            const auto broken = path / "broken";
            fs::create_directories(broken);
            write_chunk(broken, 0);
            std::ofstream(broken / "0.0.1", std::ios::binary).write("short", 5);
            vcr::ArraySpec b = spec;
            b.identity = broken.string();
            b.layers = {{"array", broken.string(), "", "", false}};
            for (const char* mode : {"cache", "mmap", "slab"}) {
                b.read_mode = std::string(mode) == "slab" ? "slab" : "mmap";
                vcr::Array a(b, std::string(mode) == "cache" ? std::make_shared<vcr::MemoryCache>(1 << 20) : nullptr);
                const double inChunk1[3]{1.0, 1.0, 5.0}, inChunk0[3]{1.0, 1.0, 1.0};
                std::uint8_t v1 = 99, v0 = 99;
                a.sample_nearest(inChunk1, 1, &v1);
                a.sample_nearest(inChunk0, 1, &v0);
                require(v1 == 7 && v0 == value_at(1, 1, 1), std::string(mode) + ": corrupt chunk reads as fill");
                require(a.stats().corrupt >= 1 && a.last_corrupt_chunk().find("0.0.1") != std::string::npos,
                        std::string(mode) + ": corrupt chunk is noted");
            }
        }
        std::cout << "Native cache-only sampling passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
