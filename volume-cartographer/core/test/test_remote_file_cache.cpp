#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/RemoteFileCache.hpp"
#include "vc/core/render/PersistentZarrCacheBudget.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace
{

fs::path temporaryDirectory(std::string_view tag)
{
    std::mt19937_64 rng(std::random_device{}());
    const auto path = fs::temp_directory_path() / ("vc_remote_file_cache_" + std::string(tag) + "_" + std::to_string(rng()));
    fs::create_directories(path);
    return path;
}

std::string readBytes(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

}  // namespace

TEST_CASE("remote file cache stores arbitrary bytes and reuses a validated hit")
{
    const auto root = temporaryDirectory("hit");
    std::atomic<int> fetches{0};
    vc::core::util::RemoteFileCacheOptions options;
    options.cacheRoot = root;
    options.destination = "objects/data.bin";
    options.fetcher = [&](const std::string&, const fs::path& tmp) {
        ++fetches;
        const std::string bytes("a\0b", 3);
        std::ofstream(tmp, std::ios::binary).write(bytes.data(), bytes.size());
    };

    const auto first = vc::core::util::cacheRemoteFile("s3://bucket/path/data.bin", options);
    const auto second = vc::core::util::cacheRemoteFile("s3://bucket/path/data.bin", options);

    CHECK_FALSE(first.cacheHit);
    CHECK(second.cacheHit);
    CHECK(fetches == 1);
    CHECK(readBytes(first.path) == std::string("a\0b", 3));
    CHECK(first.normalizedEndpoint == "https://bucket.s3.us-east-1.amazonaws.com/path/data.bin");
    fs::remove_all(root);
}

TEST_CASE("remote file cache refreshes and invalidates")
{
    const auto root = temporaryDirectory("refresh");
    int fetches = 0;
    vc::core::util::RemoteFileCacheOptions options;
    options.cacheRoot = root;
    options.destination = "payload.txt";
    options.fetcher = [&](const std::string&, const fs::path& tmp) { std::ofstream(tmp) << ++fetches; };

    const auto first = vc::core::util::cacheRemoteFile("https://example.test/file", options);
    options.policy = vc::core::util::RemoteFileCachePolicy::Refresh;
    const auto refreshed = vc::core::util::cacheRemoteFile("https://example.test/file", options);
    CHECK_FALSE(refreshed.cacheHit);
    CHECK(readBytes(first.path) == "2");

    vc::core::util::invalidateRemoteFileCacheEntry(options);
    options.policy = vc::core::util::RemoteFileCachePolicy::CacheFirst;
    (void)vc::core::util::cacheRemoteFile("https://example.test/file", options);
    CHECK(fetches == 3);
    fs::remove_all(root);
}

TEST_CASE("remote file cache retains a valid entry when refresh fails")
{
    const auto root = temporaryDirectory("failed_refresh");
    vc::core::util::RemoteFileCacheOptions options;
    options.cacheRoot = root;
    options.destination = "payload.txt";
    options.fetcher = [](const std::string&, const fs::path& tmp) { std::ofstream(tmp) << "original"; };
    (void)vc::core::util::cacheRemoteFile("https://example.test/file", options);

    options.policy = vc::core::util::RemoteFileCachePolicy::Refresh;
    options.fetcher = [](const std::string&, const fs::path&) { throw std::runtime_error("injected fetch failure"); };
    CHECK_THROWS(vc::core::util::cacheRemoteFile("https://example.test/file", options));
    CHECK(readBytes(root / "payload.txt") == "original");

    options.policy = vc::core::util::RemoteFileCachePolicy::CacheFirst;
    const auto hit = vc::core::util::cacheRemoteFile("https://example.test/file", options);
    CHECK(hit.cacheHit);
    fs::remove_all(root);
}

TEST_CASE("managed arbitrary files participate in persistent cache accounting")
{
    namespace render = vc::render;
    const auto root = temporaryDirectory("managed");
    render::PersistentZarrCacheBudget::resetRegistryForTesting();
    auto budget = render::PersistentZarrCacheBudget::configure(root, {});
    budget->waitForIdle();

    vc::core::util::RemoteFileCacheOptions options;
    options.cacheRoot = root;
    options.destination = "objects/model.weights";
    options.accounting = vc::core::util::RemoteFileCacheAccounting::Managed;
    options.fetcher = [](const std::string&, const fs::path& tmp) { std::ofstream(tmp, std::ios::binary) << "1234567"; };
    {
        auto result = vc::core::util::cacheRemoteFile("https://example.test/model.weights", options);
        REQUIRE(result.readPin.has_value());
        CHECK(budget->stats().managedBytes == 7);
        result.readPin->complete();
    }

    render::PersistentZarrCacheBudget::resetRegistryForTesting();
    budget = render::PersistentZarrCacheBudget::configure(root, {});
    budget->waitForIdle();
    CHECK(budget->stats().managedBytes == 7);
    vc::core::util::invalidateRemoteFileCacheEntry(options);
    CHECK(budget->stats().managedBytes == 0);

    render::PersistentZarrCacheBudget::resetRegistryForTesting();
    fs::remove_all(root);
}

TEST_CASE("remote file cache rejects escaping destinations")
{
    vc::core::util::RemoteFileCacheOptions options;
    options.cacheRoot = temporaryDirectory("escape");
    options.destination = "../outside";
    options.fetcher = [](const std::string&, const fs::path&) {};
    CHECK_THROWS_AS(vc::core::util::cacheRemoteFile("https://example.test/file", options), std::invalid_argument);
    fs::remove_all(options.cacheRoot);
}

TEST_CASE("remote file cache coalesces concurrent requests")
{
    const auto root = temporaryDirectory("coalesce");
    std::atomic<int> fetches{0};
    vc::core::util::RemoteFileCacheOptions options;
    options.cacheRoot = root;
    options.destination = "payload";
    options.fetcher = [&](const std::string&, const fs::path& tmp) {
        ++fetches;
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        std::ofstream(tmp) << "complete";
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&] { (void)vc::core::util::cacheRemoteFile("https://example.test/shared", options); });
    }
    for (auto& thread : threads)
        thread.join();

    CHECK(fetches == 1);
    CHECK(readBytes(root / "payload") == "complete");
    fs::remove_all(root);
}

TEST_CASE("remote file cache sidecar never stores a raw signed URL")
{
    const auto root = temporaryDirectory("secret");
    vc::core::util::RemoteFileCacheOptions options;
    options.cacheRoot = root;
    options.destination = "payload";
    options.fetcher = [](const std::string&, const fs::path& tmp) { std::ofstream(tmp) << "ok"; };
    (void)vc::core::util::cacheRemoteFile("https://example.test/file?X-Amz-Signature=secret", options);
    const auto sidecar = readBytes(root / "payload.vc-remote-file.json");
    CHECK(sidecar.find("secret") == std::string::npos);
    CHECK(sidecar.find("https://example.test/file") != std::string::npos);
    fs::remove_all(root);
}

TEST_CASE("remote file cache mirrors readable source paths")
{
    CHECK(vc::core::util::remoteFileCachePath(
              "https://example.test/bucket/run/file.json?token=secret") ==
          fs::path("remote_sources/https/example.test/bucket/run/file.json"));
    CHECK(vc::core::util::remoteFileCachePath(
              "s3://bucket/run/file.json") ==
          fs::path("remote_sources/s3/bucket/run/file.json"));
    CHECK(vc::core::util::remoteFileCacheSource(
              "s3://bucket/run/file.json?token=secret#fragment") ==
          "s3://bucket/run/file.json");
    CHECK_THROWS_AS(
        vc::core::util::remoteFileCachePath(
            "https://example.test/run/../file.json"),
        std::invalid_argument);
    CHECK_THROWS_AS(
        vc::core::util::remoteFileCachePath(
            "https://example.test/run/bad:name.json"),
        std::invalid_argument);
}
