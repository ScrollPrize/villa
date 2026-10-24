// Exercise the production classifying HTTP store, chunk fetcher and disk cache.
#include <doctest/doctest.h>
#include "LoopbackHttpServer.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/util/HttpFetch.hpp"
#include <QCoreApplication>
#include <QTemporaryDir>
#include <atomic>

namespace {
using namespace vc::render;
using Server = LoopbackHttpServer;
const std::string missing = "<Error><Code>NoSuchKey</Code></Error>";
const std::string bucket = "<Error><Code>NoSuchBucket</Code></Error>";
const std::string metadata = R"({"zarr_format":2,"shape":[4,4,4],"chunks":[4,4,4],"dtype":"|u1","compressor":null,"fill_value":0,"order":"C","filters":null})";

std::unique_ptr<ChunkCache> openCache(const std::string& url, const std::filesystem::path& path) {
    auto opened = openHttpZarrPyramid(url);
    REQUIRE(opened.fetchers.size() == 1);
    ChunkCache::Options options;
    options.persistentCachePath = path;
    ChunkCacheService::Options service;
    service.fetchConcurrency.workerCapacity = 1;
    service.fetchConcurrency.maxConcurrentReads = 1;
    return std::make_unique<ChunkCache>(
        std::vector<ChunkCache::LevelInfo>{{{4, 4, 4}, {4, 4, 4}, {}}},
        std::move(opened.fetchers), 0.0, ChunkDtype::UInt8, options, service);
}
bool hasEmptyMarker(const std::filesystem::path& root) {
    for (const auto& e : std::filesystem::recursive_directory_iterator(root))
        if (e.path().extension() == ".empty") return true;
    return false;
}
}

TEST_CASE("failed S3 lookup never persists an empty chunk and a later read recovers") {
    std::atomic<bool> recovered{false};
    Server server([&](const auto& req) {
        if (req.path == "/0/.zarray") return Server::Reply{200, metadata};
        if (req.path == "/0/0.0.0") return recovered.load()
            ? Server::Reply{200, std::string(64, 'x')} : Server::Reply{404, bucket};
        return Server::Reply{404, missing};
    });
    QTemporaryDir temp;
    REQUIRE(temp.isValid());
    const std::filesystem::path path = temp.path().toStdString();
    {
        auto cache = openCache(server.url(""), path);
        const auto result = cache->getChunkBlocking(0, 0, 0, 0);
        CHECK(result.status == ChunkStatus::Error);
        cache->waitForPendingChunks();
        cache->waitForPersistentWrites();
    }
    CHECK_FALSE(hasEmptyMarker(path));
    recovered = true;
    {
        auto cache = openCache(server.url(""), path);
        const auto result = cache->getChunkBlocking(0, 0, 0, 0);
        REQUIRE(result.status == ChunkStatus::Data);
        REQUIRE(result.bytes);
        CHECK(*result.bytes == std::vector<std::byte>(64, std::byte{'x'}));
        cache->waitForPersistentWrites();
    }
    CHECK_FALSE(hasEmptyMarker(path));
}

TEST_CASE("confirmed NoSuchKey still persists and reuses an empty chunk") {
    std::atomic<int> chunkRequests{0};
    Server server([&](const auto& req) {
        if (req.path == "/0/.zarray") return Server::Reply{200, metadata};
        if (req.path == "/0/0.0.0") ++chunkRequests;
        return Server::Reply{404, missing};
    });
    QTemporaryDir temp;
    REQUIRE(temp.isValid());
    const std::filesystem::path path = temp.path().toStdString();
    {
        auto cache = openCache(server.url(""), path);
        CHECK(cache->getChunkBlocking(0, 0, 0, 0).status == ChunkStatus::Missing);
        cache->waitForPendingChunks();
        cache->waitForPersistentWrites();
    }
    CHECK(hasEmptyMarker(path));
    const int fetched = chunkRequests;
    REQUIRE(fetched == 1);
    {
        auto cache = openCache(server.url(""), path);
        CHECK(cache->getChunkBlocking(0, 0, 0, 0).status == ChunkStatus::Missing);
    }
    CHECK(chunkRequests == fetched);
}

TEST_CASE("text and binary HTTP wrappers report unconfirmed S3 misses") {
    Server server([](auto&) { return Server::Reply{404, bucket}; });
    CHECK_THROWS_AS(vc::httpGetString(server.url()), std::runtime_error);
    CHECK_THROWS_AS(vc::httpGetBytes(server.url()), std::runtime_error);
}

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    qputenv("NO_PROXY", "127.0.0.1");
    qputenv("no_proxy", "127.0.0.1");
    return doctest::detail::runAll();
}
