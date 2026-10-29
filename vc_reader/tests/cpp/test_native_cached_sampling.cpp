// Sampling from memory alone: Array::sample_nearest_cached / sample_linear_cached
// sample what the memory cache holds, flag the rest and do no I/O.
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
        // Shape (4, 4, 12), chunks 4^3: chunks 0.0.0 and 0.0.1 are stored, 0.0.2 is not.
        fs::create_directories(path);
        const auto value_at = [](int z, int y, int x) { return std::uint8_t(z * 48 + y * 12 + x); };
        for (int cx = 0; cx < 2; ++cx) {
            std::vector<std::uint8_t> chunk(64);
            for (int z = 0; z < 4; ++z)
                for (int y = 0; y < 4; ++y)
                    for (int x = 0; x < 4; ++x)
                        chunk[(z * 4 + y) * 4 + x] = value_at(z, y, cx * 4 + x);
            std::ofstream output(path / ("0.0." + std::to_string(cx)), std::ios::binary);
            output.write(reinterpret_cast<const char*>(chunk.data()), std::streamsize(chunk.size()));
        }
        vcr::ArraySpec spec;
        spec.shape = {4, 4, 12};
        spec.chunks = {4, 4, 4};
        spec.fill = {7};
        spec.identity = path.string();
        spec.layers.push_back({"array", path.string(), "", "", false});
        vcr::Array array(spec, std::make_shared<vcr::MemoryCache>(1 << 20));
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
        const auto sentinel = -12345.f;
        const auto disk_hits = [&] { return array.stats().disk_hits.load(); };

        // Nothing in memory: every point that needs a chunk is missing, untouched, and no I/O ran.
        std::fill(cachedLinear.begin(), cachedLinear.end(), sentinel);
        array.sample_linear_cached(coords.data(), n, u8, false, cachedLinear.data(), missing.data());
        for (std::size_t p = 0; p < n; ++p) {
            const bool outside = p == 4;
            require(missing[p] == !outside, "empty cache: point " + std::to_string(p) + " missing flag");
            require(outside ? cachedLinear[p] == 7.f : cachedLinear[p] == sentinel,
                    "empty cache: point " + std::to_string(p) + " output");
        }
        std::fill(cachedNearest.begin(), cachedNearest.end(), std::uint8_t{200});
        array.sample_nearest_cached(coords.data(), n, cachedNearest.data(), missing.data());
        for (std::size_t p = 0; p < n; ++p) {
            require(missing[p] == (p != 4), "empty cache: nearest missing flag");
            require(cachedNearest[p] == (p == 4 ? 7 : 200), "empty cache: nearest output");
        }
        require(disk_hits() == 0 && array.stats().remote_fetches == 0, "memory-only reads did I/O");

        // Chunk 0 in memory: its points sample; the border point and chunk 1/2 points stay missing.
        vcr::Array reference(spec, std::make_shared<vcr::MemoryCache>(1 << 20));  // blocking results
        reference.sample_linear(coords.data(), n, u8, false, linear.data());
        reference.sample_nearest(coords.data(), n, nearest.data());
        require(array.chunk({0, 0, 0}) != nullptr, "chunk 0 is stored");
        const auto loaded = disk_hits();
        std::fill(cachedLinear.begin(), cachedLinear.end(), sentinel);
        array.sample_linear_cached(coords.data(), n, u8, false, cachedLinear.data(), missing.data());
        array.sample_nearest_cached(coords.data(), n, cachedNearest.data(), missing.data());
        require(disk_hits() == loaded, "memory-only reads did I/O");
        require(!missing[0] && cachedLinear[0] == linear[0] && cachedNearest[0] == nearest[0],
                "chunk 0 point samples from memory");
        require(missing[2] && missing[3], "chunk 1 and 2 points are missing");
        require(!missing[4], "outside point needs no chunk");
        for (std::size_t p = singles; p < n; ++p)
            require(missing[p], "chunk 1 run is missing");

        // Linear border point: its neighbours span chunks 0 and 1, so it is missing.
        array.sample_linear_cached(coords.data(), n, u8, false, cachedLinear.data(), missing.data());
        require(missing[1] && cachedLinear[1] == sentinel, "split point with a neighbour not in memory");

        // Every chunk in memory, the unstored one as known-missing: equal to the blocking calls.
        require(array.chunk({0, 0, 1}) != nullptr, "chunk 1 is stored");
        require(array.chunk({0, 0, 2}) == nullptr, "chunk 2 is not stored");
        const auto all = disk_hits();
        array.sample_linear_cached(coords.data(), n, u8, false, cachedLinear.data(), missing.data());
        for (std::size_t p = 0; p < n; ++p)
            require(!missing[p] && cachedLinear[p] == linear[p], "linear point " + std::to_string(p) + " from memory");
        array.sample_nearest_cached(coords.data(), n, cachedNearest.data(), missing.data());
        for (std::size_t p = 0; p < n; ++p)
            require(!missing[p] && cachedNearest[p] == nearest[p], "nearest point " + std::to_string(p) + " from memory");
        require(disk_hits() == all, "memory-only reads did I/O");
        require(cachedNearest[3] == 7 && std::abs(cachedLinear[3] - 7.f) < 1e-6f, "unstored chunk reads as fill");

        // Non-finite points read as NaN without a chunk, as in the blocking call.
        const double nan_point[3]{std::nan(""), 0.0, 0.0};
        float nan_value = 0;
        std::uint8_t nan_missing = 1;
        array.sample_linear_cached(nan_point, 1, u8, false, &nan_value, &nan_missing);
        require(std::isnan(nan_value) && !nan_missing, "non-finite point");
        std::cout << "Native memory-only sampling passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
