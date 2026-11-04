// Offline reads that complete: an offline_fill array reads a chunk no cache holds as
// fill instead of raising CacheMiss, remembers it as absent, and records it
// (Array::offline_misses); known-missing chunks (.missing markers) are not misses.
// Also prefill of a chunk list (cache::prefill with chunks).
#include <vc_reader/array.hpp>
#include <vc_reader/cache.hpp>

#include <chrono>
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
    const auto path = fs::temp_directory_path() / ("vcr-native-offline-fill-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { fs::path path; ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); } } cleanup{path};
    try {
        // Shape (4, 4, 12), chunks 4^3, fill 7. The decoded cache holds chunk 0.0.0 and a
        // known-missing marker for 0.0.2; chunk 0.0.1 is in no cache.
        const auto cache = path / "cache", store = path / "store";
        fs::create_directories(cache);
        fs::create_directories(store);
        const auto write_chunk = [](const fs::path& root, const std::string& key, std::uint8_t value) {
            std::vector<std::uint8_t> chunk(64, value);
            std::ofstream output(root / key, std::ios::binary);
            output.write(reinterpret_cast<const char*>(chunk.data()), std::streamsize(chunk.size()));
        };
        write_chunk(cache, "0.0.0", 42);
        std::ofstream(cache / "0.0.2.missing").put('\n');

        vcr::ArraySpec spec;
        spec.shape = {4, 4, 12};
        spec.chunks = {4, 4, 4};
        spec.fill = {7};
        spec.identity = "remote";
        spec.layers = {{"decoded", cache.string(), "", "", false}};
        spec.remote_url = "http://127.0.0.1:9/never-fetched";  // nothing listens: a fetch fails
        spec.http.retries = 0;
        spec.offline = true;

        // Plain offline: the uncached chunk raises CacheMiss.
        {
            vcr::Array array(spec, std::make_shared<vcr::MemoryCache>(1 << 20));
            bool missed = false;
            try { (void)array.chunk({0, 0, 1}); } catch (const vcr::CacheMiss&) { missed = true; }
            require(missed, "offline without fill raises CacheMiss");
            require(array.offline_misses().empty(), "offline without fill records nothing");
        }

        spec.offline_fill = true;
        vcr::Array array(spec, std::make_shared<vcr::MemoryCache>(1 << 20));
        // One point per chunk, twice: the read completes with fill for the uncached chunk.
        const std::vector<double> coords{1, 1, 1, 1, 1, 5, 1, 1, 9, 2, 2, 2, 2, 2, 6, 2, 2, 10};
        std::vector<std::uint8_t> out(coords.size() / 3, 0);
        array.sample_nearest(coords.data(), out.size(), out.data());
        require(out == std::vector<std::uint8_t>{42, 7, 7, 42, 7, 7}, "cached data, then fill");
        require(array.stats().remote_fetches == 0, "offline never fetches");
        require(array.offline_misses() == std::vector<vcr::Index>{{0, 0, 1}},
                "only the chunk in no cache is a miss (not the known-missing one)");
        require(array.stats().offline_misses == 1, "one distinct miss");

        // The miss is remembered as absent: cache-only samples take it as fill, and reads
        // again do not count it twice.
        std::vector<std::uint8_t> missing(out.size(), 1);
        array.sample_nearest_cached(coords.data(), out.size(), out.data(), missing.data());
        for (std::size_t p = 0; p < out.size(); ++p)
            require(!missing[p], "cache-only point " + std::to_string(p) + " after the miss");
        require(array.chunk({0, 0, 1}) == nullptr, "a missed chunk reads as fill");
        require(array.stats().offline_misses == 1, "a remembered miss is not counted again");

        // prefill of a chunk list: only those chunks, stored or not.
        vcr::ArraySpec local;
        local.shape = spec.shape;
        local.chunks = spec.chunks;
        local.fill = spec.fill;
        local.identity = store.string();
        local.layers = {{"array", store.string(), "", "", false}};
        write_chunk(store, "0.0.0", 1);
        vcr::Array localArray(local, std::make_shared<vcr::MemoryCache>(1 << 20));
        const auto result = vcr::cache::prefill(localArray, std::vector<vcr::Index>{{0, 0, 0}, {0, 0, 2}});
        require(result.status == vcr::cache::PrefillResult::Status::Completed, "chunk-list prefill completes");
        require(result.totalChunks == 2 && result.resolvedChunks == 2, "chunk-list prefill counts its chunks");
        require(result.dataChunks == 1 && result.emptyChunks == 1, "one stored, one unstored chunk");
        const auto all = vcr::cache::prefill(localArray);
        require(all.totalChunks == 3 && all.dataChunks == 1, "whole-array prefill is unchanged");

        std::cout << "offline fill: ok\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_native_offline_fill: " << e.what() << "\n";
        return 1;
    }
}
