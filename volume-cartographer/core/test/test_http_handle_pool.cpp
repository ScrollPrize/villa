// vc_reader's HTTP clients share one pool of libcurl handles per process: many
// clients, or one burst of requests, leave at most CurlHandles::kMaxIdle handles
// (each a connection and an eventfd) open; each client still runs at most its
// `connections` requests at once; a forked child starts with its own pool.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "support/LoopbackHttpServer.hpp"

#include <vc_reader/native/http.hpp>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

using vc::test::LoopbackHttpServer;

namespace {

constexpr auto kMaxIdle = vcr::CurlHandles::kMaxIdle;
// Descriptors one idle handle may keep: its connection, this server's end of
// it, and libcurl's wakeup (an eventfd on Linux, a socket pair elsewhere).
constexpr std::size_t kPerIdleHandle = 4;

// The burst below needs a few hundred descriptors at once, more than macOS's
// default soft limit of 256.
const bool raisedLimit = [] {
    rlimit limit{};
    if (::getrlimit(RLIMIT_NOFILE, &limit) == 0 && limit.rlim_cur != RLIM_INFINITY && limit.rlim_cur < 4096) {
        limit.rlim_cur = limit.rlim_max == RLIM_INFINITY ? 4096 : std::min<rlim_t>(4096, limit.rlim_max);
        ::setrlimit(RLIMIT_NOFILE, &limit);
    }
    return true;
}();

// Answers /slow after `delay`, counting the requests in flight at once.
struct SlowServer {
    explicit SlowServer(std::chrono::milliseconds delay)
    {
        server.setHandler([this, delay](const LoopbackHttpServer::Request& request) -> std::optional<LoopbackHttpServer::Response> {
            if (request.path != "/slow")
                return std::nullopt;
            const int now = ++inFlight;
            for (int seen = maxInFlight; now > seen && !maxInFlight.compare_exchange_weak(seen, now);) {}
            std::this_thread::sleep_for(delay);
            --inFlight;
            return LoopbackHttpServer::Response{200, "hello", {}, false};
        });
    }
    std::string url() const { return server.url() + "/slow"; }

    std::atomic<int> inFlight{0}, maxInFlight{0};
    LoopbackHttpServer server;  // last: stopped before the counters go
};

vcr::HttpOptions localOptions(int connections)
{
    vcr::HttpOptions options;
    options.connections = connections;
    options.bypass_proxy = true;  // never send 127.0.0.1 through an http_proxy
    options.retries = 0;
    return options;
}

std::size_t openDescriptors()
{
    std::error_code ec;
    std::size_t n = 0;
    for (auto it = std::filesystem::directory_iterator("/dev/fd", ec); !ec && it != std::filesystem::directory_iterator(); it.increment(ec))
        ++n;
    return n;
}

// `count` requests through `client` at once; all must answer "hello".
void fetchConcurrently(vcr::HttpClient& client, const std::string& url, int count)
{
    std::vector<std::thread> threads;
    std::atomic<int> ok{0};
    for (int i = 0; i < count; ++i)
        threads.emplace_back([&] {
            try {
                std::vector<std::uint8_t> body;
                if (client.get(url, body) && std::string(body.begin(), body.end()) == "hello")
                    ++ok;
            } catch (const std::exception& e) {
                std::cerr << "request failed: " << e.what() << "\n";
            }
        });
    for (auto& t : threads)
        t.join();
    CHECK(ok.load() == count);
}

// Wait for the server's connection threads to close the ends of connections
// whose client handles were cleaned up.
std::size_t settledDescriptors(std::size_t bound)
{
    auto n = openDescriptors();
    for (int i = 0; i < 50 && n > bound; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        n = openDescriptors();
    }
    return n;
}

}  // namespace

TEST_CASE("Many clients keep at most kMaxIdle handles open")
{
    SlowServer slow(std::chrono::milliseconds(10));
    const auto before = openDescriptors();
    // Before the shared pool, 100 clients x 6 requests kept 600 handles open: a
    // client socket, libcurl's wakeup and this server's socket each, ~1800
    // descriptors on Linux.
    std::vector<std::unique_ptr<vcr::HttpClient>> clients;
    for (int i = 0; i < 100; ++i) {
        clients.push_back(std::make_unique<vcr::HttpClient>(localOptions(16)));
        fetchConcurrently(*clients.back(), slow.url(), 6);
    }
    CHECK(vcr::CurlHandles::instance().idle() <= kMaxIdle);
    const auto bound = before + kPerIdleHandle * kMaxIdle + 16;
    CHECK(settledDescriptors(bound) <= bound);

    // Handles outlive the clients that used them and serve new clients.
    clients.clear();
    vcr::HttpClient again(localOptions(4));
    fetchConcurrently(again, slow.url(), 4);
}

TEST_CASE("A burst beyond kMaxIdle is cut back to kMaxIdle when it ends")
{
    SlowServer slow(std::chrono::milliseconds(400));
    const auto before = openDescriptors();
    constexpr int kBurst = static_cast<int>(kMaxIdle) + 36;
    vcr::HttpClient wide(localOptions(kBurst));
    fetchConcurrently(wide, slow.url(), kBurst);
    // The burst really held more than kMaxIdle handles at once.
    CHECK(slow.maxInFlight.load() > static_cast<int>(kMaxIdle));
    CHECK(vcr::CurlHandles::instance().idle() == kMaxIdle);
    const auto bound = before + kPerIdleHandle * kMaxIdle + 16;
    CHECK(settledDescriptors(bound) <= bound);
}

TEST_CASE("Each client keeps its own limit of requests in flight")
{
    SlowServer slow(std::chrono::milliseconds(50));
    vcr::HttpClient narrow(localOptions(2));
    fetchConcurrently(narrow, slow.url(), 8);
    CHECK(slow.maxInFlight.load() <= 2);

    slow.maxInFlight = 0;
    vcr::HttpClient other(localOptions(2));
    std::thread a([&] { fetchConcurrently(narrow, slow.url(), 6); });
    std::thread b([&] { fetchConcurrently(other, slow.url(), 6); });
    a.join();
    b.join();
    CHECK(slow.maxInFlight.load() <= 4);
    CHECK(slow.maxInFlight.load() >= 3);  // the two limits are separate
}

TEST_CASE("A forked child gets its own pool")
{
    SlowServer slow(std::chrono::milliseconds(1));
    vcr::HttpClient client(localOptions(4));
    fetchConcurrently(client, slow.url(), 4);
    auto* parentPool = &vcr::CurlHandles::instance();
    REQUIRE(parentPool->idle() > 0);

    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        // No doctest in the child: report through the exit status.
        auto& pool = vcr::CurlHandles::instance();
        int status = &pool == parentPool ? 10 : pool.idle() != 0 ? 11 : 0;
        if (status == 0) {
            try {
                vcr::HttpClient childClient(localOptions(1));
                std::vector<std::uint8_t> body;
                if (!childClient.get(slow.url(), body) || std::string(body.begin(), body.end()) != "hello")
                    status = 12;
            } catch (...) {
                status = 13;
            }
        }
        ::_exit(status);
    }
    int status = 0;
    REQUIRE(::waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 0);
    // The parent's pool is untouched and still serves.
    CHECK(&vcr::CurlHandles::instance() == parentPool);
    fetchConcurrently(client, slow.url(), 2);
}
