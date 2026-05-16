// Coverage for core/src/NeuralTracerConnection.cpp.
//
// Connects to a synthetic Unix-socket "server" that replies with a canned
// JSON payload. Also exercises the constructor error paths (missing socket,
// path too long).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/tracer/NeuralTracerConnection.h"

#include <opencv2/core.hpp>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

namespace fs = std::filesystem;

namespace {

std::string tmpSocketPath()
{
    std::mt19937_64 rng(std::random_device{}());
    return (fs::temp_directory_path() /
            ("vc_ntc_sock_" + std::to_string(rng())))
        .string();
}

// Reads one newline-terminated line from `fd`. Returns the body without the
// newline. Returns empty on EOF.
std::string readLine(int fd)
{
    std::string out;
    char ch;
    while (true) {
        ssize_t n = ::recv(fd, &ch, 1, 0);
        if (n <= 0) break;
        if (ch == '\n') break;
        out.push_back(ch);
    }
    return out;
}

void writeAll(int fd, const std::string& s)
{
    size_t off = 0;
    while (off < s.size()) {
        ssize_t n = ::send(fd, s.data() + off, s.size() - off, 0);
        if (n <= 0) break;
        off += static_cast<size_t>(n);
    }
}

// Spin up a Unix-domain socket server on `path` that:
//   accepts one client, reads one newline-terminated request,
//   writes `response + "\n"`, then exits.
std::thread spawnServer(const std::string& path,
                        const std::string& response,
                        std::atomic<bool>* ready)
{
    return std::thread([path, response, ready]() {
        int sfd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        REQUIRE(sfd >= 0);
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
        ::unlink(path.c_str());
        if (::bind(sfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            ::close(sfd);
            return;
        }
        if (::listen(sfd, 1) < 0) {
            ::close(sfd);
            return;
        }
        if (ready) ready->store(true, std::memory_order_release);
        int cfd = ::accept(sfd, nullptr, nullptr);
        if (cfd < 0) {
            ::close(sfd);
            return;
        }
        (void)readLine(cfd);
        writeAll(cfd, response + "\n");
        ::close(cfd);
        ::close(sfd);
        ::unlink(path.c_str());
    });
}

} // namespace

TEST_CASE("Constructor throws when the socket path doesn't exist")
{
    auto make = []() { NeuralTracerConnection c("/__no__/__where__"); (void)c; };
    CHECK_THROWS_AS(make(), std::runtime_error);
}

TEST_CASE("Constructor throws when the socket path is too long")
{
    // sizeof(sun_path) is typically 108; 200 is well past it.
    std::string huge(200, 'a');
    auto make = [&]() { NeuralTracerConnection c(huge); (void)c; };
    CHECK_THROWS_AS(make(), std::runtime_error);
}

TEST_CASE("get_next_points returns the candidates the server emits")
{
    auto path = tmpSocketPath();
    // Two batches; each batch has 1 u-candidate and 1 v-candidate.
    const std::string response =
        R"({"u_candidates":[[[1,2,3]],[[4,5,6]]],)"
        R"("v_candidates":[[[7,8,9]],[[10,11,12]]]})";
    std::atomic<bool> ready{false};
    auto srv = spawnServer(path, response, &ready);
    // Wait for the server to be listening.
    for (int i = 0; i < 100 && !ready.load(std::memory_order_acquire); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    REQUIRE(ready.load());

    {
        NeuralTracerConnection c(path);
        std::vector<cv::Vec3f> centers = { cv::Vec3f(0, 0, 0), cv::Vec3f(1, 0, 0) };
        std::vector<std::optional<cv::Vec3f>> empty(2);
        auto results = c.get_next_points(centers, empty, empty, empty);
        REQUIRE(results.size() == 2);
        CHECK(results[0].next_u_xyzs.size() == 1);
        CHECK(results[0].next_v_xyzs.size() == 1);
        CHECK(results[0].next_u_xyzs[0] == cv::Vec3f(1, 2, 3));
        CHECK(results[0].next_v_xyzs[0] == cv::Vec3f(7, 8, 9));
        CHECK(results[1].next_u_xyzs[0] == cv::Vec3f(4, 5, 6));
    }
    srv.join();
}

TEST_CASE("get_next_points: error response throws runtime_error")
{
    auto path = tmpSocketPath();
    std::atomic<bool> ready{false};
    auto srv = spawnServer(path, R"({"error":"bad request"})", &ready);
    for (int i = 0; i < 100 && !ready.load(std::memory_order_acquire); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    REQUIRE(ready.load());
    {
        NeuralTracerConnection c(path);
        std::vector<cv::Vec3f> centers = { cv::Vec3f(0, 0, 0) };
        std::vector<std::optional<cv::Vec3f>> empty(1);
        CHECK_THROWS_AS(c.get_next_points(centers, empty, empty, empty),
                        std::runtime_error);
    }
    srv.join();
}

TEST_CASE("get_next_points: NaN sanitization in the response")
{
    auto path = tmpSocketPath();
    // Server emits NaN literals; the client should rewrite to null then parse.
    const std::string response =
        R"({"u_candidates":[[[NaN, 2.0, 3.0]]],"v_candidates":[[[4.0, NaN, 6.0]]]})";
    std::atomic<bool> ready{false};
    auto srv = spawnServer(path, response, &ready);
    for (int i = 0; i < 100 && !ready.load(std::memory_order_acquire); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    REQUIRE(ready.load());
    {
        NeuralTracerConnection c(path);
        std::vector<cv::Vec3f> centers = { cv::Vec3f(0, 0, 0) };
        std::vector<std::optional<cv::Vec3f>> prev_u = { cv::Vec3f(1, 1, 1) };
        std::vector<std::optional<cv::Vec3f>> empty(1);
        auto results = c.get_next_points(centers, prev_u, empty, empty);
        REQUIRE(results.size() == 1);
        REQUIRE(results[0].next_u_xyzs.size() == 1);
        // Values that were NaN in the response come back as NaN floats.
        CHECK(std::isnan(results[0].next_u_xyzs[0][0]));
        CHECK(results[0].next_u_xyzs[0][1] == doctest::Approx(2.0f));
        CHECK(std::isnan(results[0].next_v_xyzs[0][1]));
    }
    srv.join();
}
