#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/render/ChunkCache.hpp"
#include "vc/core/render/ZarrChunkFetcher.hpp"
#include "vc/core/util/RemoteFileCache.hpp"
#include "vc/lasagna/Dataset.hpp"
#include "utils/zarr.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

namespace fs = std::filesystem;
using vc::render::ChunkCache;
using vc::render::ChunkStatus;

class TemporaryCacheDirectory {
public:
    TemporaryCacheDirectory()
    {
        auto pattern = (fs::temp_directory_path() / "vc_lasagna_prefetch_XXXXXX").string();
        const auto created = ::mkdtemp(pattern.data());
        if (!created)
            throw std::runtime_error("could not create remote-prefetch test cache");
        path = created;
    }
    ~TemporaryCacheDirectory()
    {
        std::error_code error;
        fs::remove_all(path, error);
    }
    fs::path path;
};

// Gate the first GET while allowing metadata HEAD probes on other connections.
// Responses close the connection for portability. The test waits for the
// production cache's follower counter before releasing a failing owner.
class GatedChunkServer {
public:
    explicit GatedChunkServer(bool failFirst,
                              std::string requestPath = "/0.0.0",
                              std::string successBody = std::string(64, char{37}),
                              int failures = 1)
        : failFirst_(failFirst), requestPath_(std::move(requestPath)),
          successBody_(std::move(successBody)), failures_(failures)
    {
        listener_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listener_ < 0)
            throw std::runtime_error("could not create loopback test socket");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = 0;
        socklen_t size = sizeof(address);
        if (::bind(listener_, reinterpret_cast<sockaddr*>(&address), size) != 0 ||
            ::getsockname(listener_, reinterpret_cast<sockaddr*>(&address), &size) != 0 ||
            ::listen(listener_, 8) != 0) {
            ::close(listener_);
            throw std::runtime_error("could not bind loopback test server");
        }
        url = "http://127.0.0.1:" + std::to_string(ntohs(address.sin_port));
        worker_ = std::thread([this] { run(); });
    }

    ~GatedChunkServer()
    {
        stopped_.store(true);
        release();
        ::shutdown(listener_, SHUT_RDWR);
        ::close(listener_);
        if (worker_.joinable())
            worker_.join();
        for (auto& client : clients_)
            client.join();
    }

    bool waitForChunkRequest()
    {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds{3}, [&] { return chunkRequests_ != 0; });
    }

    void release()
    {
        std::lock_guard lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

    int chunkRequests() const
    {
        std::lock_guard lock(mutex_);
        return chunkRequests_;
    }

    std::string url;

private:
    static void sendResponse(int client, const std::string& response)
    {
        std::size_t sent = 0;
        while (sent < response.size()) {
#if defined(MSG_NOSIGNAL)
            constexpr int flags = MSG_NOSIGNAL;
#else
            constexpr int flags = 0;
#endif
            const auto bytes = ::send(client, response.data() + sent, response.size() - sent, flags);
            if (bytes < 0 && errno == EINTR)
                continue;
            if (bytes <= 0)
                return;
            sent += static_cast<std::size_t>(bytes);
        }
    }

    void run()
    {
        while (!stopped_.load()) {
            const int client = ::accept(listener_, nullptr, nullptr);
            if (client < 0) {
                if (errno == EINTR)
                    continue;
                return;
            }
            clients_.emplace_back([this, client] { serve(client); });
        }
    }

    void serve(int client)
    {
#if defined(SO_NOSIGPIPE)
        const int noSigpipe = 1;
        ::setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &noSigpipe, sizeof(noSigpipe));
#endif
        const timeval timeout{3, 0};
        ::setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        std::string request;
        std::array<char, 1024> buffer{};
        while (request.find("\r\n\r\n") == std::string::npos && request.size() < 8192) {
            const auto count = ::recv(client, buffer.data(), buffer.size(), 0);
            if (count <= 0)
                break;
            request.append(buffer.data(), static_cast<std::size_t>(count));
        }
        const bool chunkGet = request.starts_with("GET " + requestPath_ + " ");
        const bool metadataHead = request.starts_with("HEAD " + requestPath_ + " ");
        bool fail = false;
        if (chunkGet) {
            std::unique_lock lock(mutex_);
            const int requestNumber = ++chunkRequests_;
            cv_.notify_all();
            if (requestNumber == 1)
                cv_.wait(lock, [&] { return released_; });
            fail = requestNumber <= failures_ && failFirst_;
        }
        const std::string status = chunkGet || metadataHead
            ? (fail ? "400 Bad Request" : "200 OK") : "404 Not Found";
        const std::string body = chunkGet && !fail ? successBody_ : std::string{};
        sendResponse(client, "HTTP/1.1 " + status + "\r\nContent-Length: " +
            std::to_string(metadataHead ? successBody_.size() : body.size()) +
            "\r\nConnection: close\r\n\r\n" + body);
        ::close(client);
    }

    int listener_ = -1;
    bool failFirst_;
    std::string requestPath_;
    std::string successBody_;
    int failures_;
    std::atomic<bool> stopped_{false};
    std::thread worker_;
    std::vector<std::thread> clients_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool released_ = false;
    int chunkRequests_ = 0;
};

struct ResponseReleaseGuard {
    GatedChunkServer& server;
    ~ResponseReleaseGuard() { server.release(); }
};

template <typename Predicate>
bool waitFor(Predicate predicate)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline)
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    return true;
}

} // namespace

TEST_CASE("Lasagna scalar readers share speculative source bytes without inheriting optional errors")
{
    bool speculativeOwner = true;
    bool failFirst = true;
    int failures = 1;
    SUBCASE("failed speculative owner gives scalar demand one ordinary attempt") {}
    SUBCASE("successful speculative owner shares exact bytes") { failFirst = false; }
    SUBCASE("original required-owner errors retain their existing behavior") { speculativeOwner = false; }
    SUBCASE("a failed ordinary scalar retry stays terminal") { failures = 2; }

    TemporaryCacheDirectory directory;
    GatedChunkServer server(failFirst, "/0.0.0", std::string(64, char{37}), failures);
    // Metadata is already cached, matching a model whose manifest was opened
    // before corridor warmup. Only its first data chunk requires HTTP.
    {
        std::ofstream metadata(directory.path / ".zarray");
        metadata << R"({"zarr_format":2,"shape":[4,4,4],"chunks":[4,4,4],"dtype":"|u1","compressor":null,"fill_value":0,"order":"C","filters":null,"dimension_separator":"."})";
        REQUIRE(metadata.good());
    }
    vc::lasagna::LasagnaChannelGroup group;
    group.remoteZarrBaseUrl = server.url;
    group.remoteCacheRoot = directory.path;
    group.discoverAwsCredentials = false;
    auto array = std::make_shared<utils::ZarrArray>(
        vc::lasagna::openLasagnaChannelArray({}, group));
    ChunkCache::Options cacheOptions;
    cacheOptions.detectAllFillChunks = false;
    vc::render::ChunkCacheService::Options serviceOptions;
    serviceOptions.fetchConcurrency.workerCapacity = 2;
    serviceOptions.fetchConcurrency.maxConcurrentReads = 2;
    auto cache = vc::render::createChunkCache(array, cacheOptions, serviceOptions);
    const auto before = vc::lasagna::remoteStoreStats();
    std::future<vc::render::ChunkResult> owner;
    std::future<std::optional<std::vector<std::byte>>> follower;
    // Release before future destruction on any fatal assertion.
    ResponseReleaseGuard release{server};
    if (speculativeOwner) {
        REQUIRE(cache->prefetchSpeculativeChunk({0, 0, 0, 0}) ==
                ChunkCache::SpeculativePrefetchStatus::Submitted);
    } else {
        owner = std::async(std::launch::async, [&] { return cache->getChunkBlocking(0, 0, 0, 0); });
    }
    REQUIRE(server.waitForChunkRequest());
    follower = std::async(std::launch::async, [array] {
        const std::array<std::size_t, 3> index{0, 0, 0};
        return array->read_chunk(index);
    });
    REQUIRE(waitFor([&] { return vc::lasagna::remoteStoreStats().objectsJoined > before.objectsJoined; }));
    CHECK(server.chunkRequests() == 1);
    server.release();
    REQUIRE(follower.wait_for(std::chrono::seconds{3}) == std::future_status::ready);
    if (!speculativeOwner) {
        CHECK_THROWS_AS(follower.get(), std::runtime_error);
        REQUIRE(owner.wait_for(std::chrono::seconds{3}) == std::future_status::ready);
        CHECK(owner.get().status == ChunkStatus::Error);
        CHECK(server.chunkRequests() == 1);
    } else if (failures == 2) {
        CHECK_THROWS_AS(follower.get(), std::runtime_error);
        REQUIRE(waitFor([] { return ChunkCache::speculativePrefetchStats().pendingRequests == 0; }));
        CHECK(server.chunkRequests() == 2);
    } else {
        const auto result = follower.get();
        REQUIRE(result.has_value());
        CHECK(*result == std::vector<std::byte>(64, std::byte{37}));
        REQUIRE(waitFor([] { return ChunkCache::speculativePrefetchStats().pendingRequests == 0; }));
        const auto required = cache->getChunkBlocking(0, 0, 0, 0);
        REQUIRE(required.status == ChunkStatus::Data);
        REQUIRE(required.bytes);
        CHECK(*required.bytes == *result);
        CHECK(server.chunkRequests() == (failFirst ? 2 : 1));
    }
}

TEST_CASE("Lasagna embedded HTTP cache reports network bytes but not warm disk bytes")
{
    TemporaryCacheDirectory directory;
    GatedChunkServer server(false);
    {
        std::ofstream metadata(directory.path / ".zarray");
        metadata << R"({"zarr_format":2,"shape":[4,4,4],"chunks":[4,4,4],"dtype":"|u1","compressor":null,"fill_value":0,"order":"C","filters":null,"dimension_separator":"."})";
        REQUIRE(metadata.good());
    }
    vc::lasagna::LasagnaChannelGroup group;
    group.remoteZarrBaseUrl = server.url;
    group.remoteCacheRoot = directory.path;
    group.discoverAwsCredentials = false;
    auto array = std::make_shared<utils::ZarrArray>(
        vc::lasagna::openLasagnaChannelArray({}, group));
    ChunkCache::Options cacheOptions;
    cacheOptions.detectAllFillChunks = false;
    vc::render::ChunkCacheService::Options serviceOptions;
    serviceOptions.fetchConcurrency.workerCapacity = 2;
    serviceOptions.fetchConcurrency.maxConcurrentReads = 2;
    auto cold = vc::render::createChunkCache(array, cacheOptions, serviceOptions, true);
    ResponseReleaseGuard release{server};
    REQUIRE(cold->prefetchSpeculativeChunk({0, 0, 0, 0}) ==
            ChunkCache::SpeculativePrefetchStatus::Submitted);
    REQUIRE(server.waitForChunkRequest());
    server.release();
    REQUIRE(waitFor([] { return ChunkCache::speculativePrefetchStats().pendingRequests == 0; }));
    const auto required = cold->getChunkBlocking(0, 0, 0, 0);
    REQUIRE(required.status == ChunkStatus::Data);
    REQUIRE(required.bytes);
    CHECK(*required.bytes == std::vector<std::byte>(64, std::byte{37}));
    CHECK(cold->stats().remoteDownloadBytesPerSecond > 0.0);
    CHECK(server.chunkRequests() == 1);

    // A separate decoded cache and scheduler cannot inherit the first cache's
    // transfer estimates. It reaches the shared store's published disk object.
    auto warm = vc::render::createChunkCache(array, cacheOptions, serviceOptions, true);
    const auto fromDisk = warm->getChunkBlocking(0, 0, 0, 0);
    REQUIRE(fromDisk.status == ChunkStatus::Data);
    REQUIRE(fromDisk.bytes);
    CHECK(*fromDisk.bytes == *required.bytes);
    CHECK(warm->stats().remoteDownloadBytesPerSecond == 0.0);
    CHECK(server.chunkRequests() == 1);
}

TEST_CASE("Lasagna array metadata isolates optional owner errors from required openers")
{
    bool speculativeOwner = true;
    bool failFirst = true;
    int failures = 1;
    SUBCASE("required opener retries optional metadata failure") {}
    SUBCASE("successful optional metadata is shared") { failFirst = false; }
    SUBCASE("original required metadata failure is unchanged") { speculativeOwner = false; }
    SUBCASE("a failed ordinary retry stays terminal") { failures = 2; }

    TemporaryCacheDirectory directory;
    const std::string metadata = R"({"zarr_format":2,"shape":[4,4,4],"chunks":[4,4,4],"dtype":"|u1","compressor":null,"fill_value":0,"order":"C","filters":null})";
    GatedChunkServer server(failFirst, "/.zarray", metadata, failures);
    vc::lasagna::LasagnaChannelGroup group;
    group.remoteZarrBaseUrl = server.url;
    group.remoteCacheRoot = directory.path;
    group.discoverAwsCredentials = false;
    const auto before = vc::lasagna::remoteStoreStats();
    std::future<utils::ZarrArray> owner;
    std::future<utils::ZarrArray> follower;
    ResponseReleaseGuard release{server};
    owner = std::async(std::launch::async, [group, speculativeOwner] {
        ChunkCache::SpeculativeSourceReadScope metadataScope(speculativeOwner);
        return vc::lasagna::openLasagnaChannelArray({}, group);
    });
    REQUIRE(server.waitForChunkRequest());
    follower = std::async(std::launch::async, [group] {
        return vc::lasagna::openLasagnaChannelArray({}, group);
    });
    REQUIRE(waitFor([&] { return vc::lasagna::remoteStoreStats().objectsJoined > before.objectsJoined; }));
    CHECK(server.chunkRequests() == 1);
    server.release();
    REQUIRE(owner.wait_for(std::chrono::seconds{3}) == std::future_status::ready);
    REQUIRE(follower.wait_for(std::chrono::seconds{3}) == std::future_status::ready);
    if (failFirst)
        CHECK_THROWS_AS(owner.get(), std::runtime_error);
    else
        CHECK(owner.get().metadata().shape == std::vector<std::size_t>{4, 4, 4});
    if ((!speculativeOwner && failFirst) || failures > 1) {
        CHECK_THROWS_AS(follower.get(), std::runtime_error);
    } else {
        const auto result = follower.get();
        CHECK(result.metadata().shape == std::vector<std::size_t>{4, 4, 4});
        CHECK(result.metadata().chunks == std::vector<std::size_t>{4, 4, 4});
        CHECK(result.metadata().dtype == utils::ZarrDtype::uint8);
    }
    CHECK(server.chunkRequests() == (speculativeOwner && failFirst ? 2 : 1));
}

TEST_CASE("Remote manifest cache isolates optional owner errors from required followers")
{
    namespace remote = vc::core::util;
    bool speculativeOwner = true;
    bool failFirst = true;
    int failures = 1;
    SUBCASE("required manifest opener retries optional fetch failure") {}
    SUBCASE("successful optional manifest bytes are shared") { failFirst = false; }
    SUBCASE("original required manifest failure is unchanged") { speculativeOwner = false; }
    SUBCASE("a failed ordinary manifest retry stays terminal") { failures = 2; }

    TemporaryCacheDirectory directory;
    const std::string manifest = R"({"version":2,"groups":{"prediction":{"zarr":"prediction.zarr","scaledown":0,"channels":["presence"]}}})";
    GatedChunkServer server(failFirst, "/model.lasagna.json", manifest, failures);
    remote::RemoteFileCacheOptions options;
    options.cacheRoot = directory.path;
    options.destination = "model.lasagna.json";
    options.discoverAwsCredentials = false;
    const auto location = server.url + "/model.lasagna.json";
    const auto before = remote::remoteFileCacheCurrentFollowers();
    std::future<remote::RemoteFileCacheResult> owner;
    std::future<remote::RemoteFileCacheResult> follower;
    ResponseReleaseGuard release{server};
    owner = std::async(std::launch::async, [location, options, speculativeOwner] {
        ChunkCache::SpeculativeSourceReadScope metadataScope(speculativeOwner);
        return remote::cacheRemoteFile(location, options);
    });
    REQUIRE(server.waitForChunkRequest());
    follower = std::async(std::launch::async, [location, options] {
        return remote::cacheRemoteFile(location, options);
    });
    REQUIRE(waitFor([&] { return remote::remoteFileCacheCurrentFollowers() > before; }));
    CHECK(server.chunkRequests() == 1);
    server.release();
    REQUIRE(owner.wait_for(std::chrono::seconds{3}) == std::future_status::ready);
    REQUIRE(follower.wait_for(std::chrono::seconds{3}) == std::future_status::ready);
    if (failFirst)
        CHECK_THROWS_AS(owner.get(), std::runtime_error);
    else
        CHECK_FALSE(owner.get().cacheHit);
    if ((!speculativeOwner && failFirst) || failures > 1) {
        CHECK_THROWS_AS(follower.get(), std::runtime_error);
    } else {
        const auto result = follower.get();
        std::ifstream input(result.path, std::ios::binary);
        const std::string actual{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        CHECK(actual == manifest);
    }
    CHECK(server.chunkRequests() == (speculativeOwner && failFirst ? 2 : 1));
    CHECK(remote::remoteFileCacheCurrentFollowers() == before);
}

#else
TEST_CASE("Lasagna remote prefetch loopback fixture requires POSIX sockets")
{
    MESSAGE("Loopback integration is exercised on Ubuntu and macOS.");
}
#endif
