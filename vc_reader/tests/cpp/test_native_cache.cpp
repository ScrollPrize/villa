#include <vc_reader/cache.hpp>
#include <vc_reader/zarr_write.hpp>
#include <vc_reader/array.hpp>
#include <vc_reader/native/write_behind.hpp>
#include <future>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
int main()
{
    const auto root = fs::temp_directory_path() / ("vcr-prefill-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { fs::path root; ~Cleanup() { std::error_code ec; fs::remove_all(root, ec); } } cleanup{root};
    try {
        fs::create_directories(root);
        std::ofstream(root / ".zarray") << R"({"zarr_format":2,"shape":[4,4,4],"chunks":[1,1,1],"dtype":"|u1","compressor":null,"filters":null,"fill_value":7,"order":"C"})";
        std::ofstream(root / "0.0.0", std::ios::binary).put(42);
        vcr::zarr::OpenOptions options;
        options.threads = 1;
        auto opened = vcr::zarr::openArray(root.string(), options);
        std::atomic<bool> cancel{true};
        auto result = vcr::cache::prefill(*opened.array, &cancel);
        require(result.status == vcr::cache::PrefillResult::Status::Cancelled && result.resolvedChunks == 0,
                "cancel before reads");
        cancel = false;
        result = vcr::cache::prefill(*opened.array, &cancel, [&](auto done, auto total) {
            require(total == 64, "chunk grid");
            if (done >= 32) cancel = true;
        });
        require(result.status == vcr::cache::PrefillResult::Status::Cancelled && result.resolvedChunks < 64,
                "cancel partway through grid");
        cancel = false;
        std::size_t last = 0;
        result = vcr::cache::prefill(*opened.array, &cancel, [&](auto done, auto total) {
            require(done > last && done <= total, "serialized monotonic progress");
            last = done;
        });
        require(result.status == vcr::cache::PrefillResult::Status::Completed && last == 64, "resume completes");
        require(result.dataChunks == 1 && result.emptyChunks == 63 && result.errorChunks == 0,
                "stored and missing chunks counted separately");
        require(result.fetchedChunks == 0, "local reads do not count as remote fetches");
        std::uint8_t value = 0;
        opened.array->read_many({{0,0,0}}, {{1,1,1}}, {&value});
        require(value == 42, "prefill preserves values");
        result = vcr::cache::prefill(*opened.array, nullptr, [](auto, auto) { throw std::runtime_error("progress failed"); });
        require(result.status == vcr::cache::PrefillResult::Status::Failed && result.message == "progress failed",
                "progress exceptions are reported after worker completion");
        bool rejected = false;
        try { (void)vcr::cache::downloadPrefix("s3://bucket/prefix", {}, 1); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "download must have persistent cache");
        {   // over the byte budget, a non-waiting submit drops its write instead of blocking
            vcr::WriteBehind writes(1, 100);
            std::promise<void> release;
            auto released = release.get_future().share();
            require(writes.submit("a", nullptr, 100, [released] { released.wait(); }, false), "first write queued");
            require(!writes.submit("b", nullptr, 1, [] {}, false), "over budget: dropped");
            require(!writes.pending("b").first, "a dropped write is not pending");
            release.set_value();
            writes.flush();
            require(writes.submit("b", nullptr, 1, [] {}, false), "room again after the queue drains");
        }
        std::cout << "Cache prefill cancellation, resume, counts and values passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
