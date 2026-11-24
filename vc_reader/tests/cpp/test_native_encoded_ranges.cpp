// vcr::sources::EncodedFiles::readRanges: parts of local, cached and remote files, served
// remotely without caching the whole file; missing files, changed objects, servers that
// ignore Range, and many concurrent ranges.
#include "vc_reader/source_reader.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace vcr;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition); \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// Serves one object at /bucket/data.bin, one request per connection.
class Server {
public:
    explicit Server(std::string data) : data_(std::move(data))
    {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        CHECK(fd_ >= 0);
        int one = 1;
        ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        CHECK(::bind(fd_, reinterpret_cast<sockaddr*>(&address), sizeof address) == 0);
        CHECK(::listen(fd_, 256) == 0);
        socklen_t length = sizeof address;
        CHECK(::getsockname(fd_, reinterpret_cast<sockaddr*>(&address), &length) == 0);
        port_ = ntohs(address.sin_port);
        thread_ = std::thread([this] { serve(); });
    }
    ~Server()
    {
        stop_ = true;
        ::shutdown(fd_, SHUT_RDWR);
        ::close(fd_);
        thread_.join();
    }
    std::string url(const std::string& name = "data.bin") const
    {
        return "http://127.0.0.1:" + std::to_string(port_) + "/bucket/" + name;
    }
    std::atomic<int> requests{0};
    std::atomic<bool> ignoreRange{false};
    void setEtag(std::string etag)
    {
        std::lock_guard lock(mutex_);
        etag_ = std::move(etag);
    }

private:
    void serve()
    {
        while (!stop_) {
            const int client = ::accept(fd_, nullptr, nullptr);
            if (client < 0) return;
            std::thread([this, client] { answer(client); }).detach();
        }
    }
    void answer(int client)
    {
        std::string request;
        char buffer[4096];
        while (request.find("\r\n\r\n") == std::string::npos) {
            const auto n = ::recv(client, buffer, sizeof buffer, 0);
            if (n <= 0) {
                ::close(client);
                return;
            }
            request.append(buffer, std::size_t(n));
        }
        ++requests;
        const auto path = request.substr(request.find(' ') + 1, request.find(' ', request.find(' ') + 1) - request.find(' ') - 1);
        std::string etag;
        {
            std::lock_guard lock(mutex_);
            etag = etag_;
        }
        std::string head, body;
        if (path != "/bucket/data.bin") {
            head = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n";
        } else if (const auto at = request.find("Range: bytes="); at != std::string::npos && !ignoreRange) {
            const auto spec = request.substr(at + 13, request.find("\r\n", at) - at - 13);
            const auto first = std::stoull(spec.substr(0, spec.find('-')));
            const auto last = std::stoull(spec.substr(spec.find('-') + 1));
            if (first >= data_.size()) {
                head = "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\n";
            } else {
                const auto end = std::min<std::uint64_t>(last, data_.size() - 1);
                body = data_.substr(first, end - first + 1);
                head = "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes " + std::to_string(first) + "-" +
                       std::to_string(end) + "/" + std::to_string(data_.size()) + "\r\nETag: " + etag +
                       "\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
            }
        } else {
            body = data_;
            head = "HTTP/1.1 200 OK\r\nETag: " + etag + "\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
        }
        const auto answer = head + "Connection: close\r\n\r\n" + body;
        for (std::size_t sent = 0; sent < answer.size();) {
            const auto n = ::send(client, answer.data() + sent, answer.size() - sent, MSG_NOSIGNAL);
            if (n <= 0) break;
            sent += std::size_t(n);
        }
        ::close(client);
    }
    std::string data_, etag_ = "\"v1\"";
    std::mutex mutex_;
    int fd_ = -1, port_ = 0;
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

static sources::OptionsFor cachedIn(const fs::path& root, bool offline = false)
{
    return [root, offline](const std::string& location) {
        zarr::OpenOptions options;
        options.access = zarr::resolveAccess(location, zarr::AuthMode::Anonymous);
        options.cacheDirs = {root.string()};
        options.offline = offline;
        options.connections = 8;
        return options;
    };
}

template<class E, class F> static bool raises(F f)
{
    try {
        f();
    } catch (const E&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

int main()
{
    const auto root = fs::temp_directory_path() / ("vc_reader_ranges_" + std::to_string(::getpid()));
    fs::remove_all(root);
    fs::create_directories(root / "cache");
    std::string data(1 << 20, '\0');
    std::mt19937 random(7);
    for (auto& byte : data) byte = char(random());
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges{{0, 52}, {1000, 1}, {data.size() - 10, 10}, {4096, 0}, {500000, 70000}};
    const auto expect = [&](const sources::EncodedFiles::Ranges& got) {
        CHECK(got.parts.size() == ranges.size());
        for (std::size_t i = 0; i < ranges.size(); ++i) CHECK(got.parts[i] == data.substr(ranges[i].first, ranges[i].second));
        CHECK(got.size && *got.size == data.size());
    };

    // A local file is read in place.
    {
        const auto path = root / "local.bin";
        std::ofstream(path, std::ios::binary) << data;
        sources::EncodedFiles files(sources::defaultOptions());
        const auto got = files.readRanges(path.string(), ranges);
        expect(got);
        CHECK(got.version.empty());
        CHECK(raises<std::out_of_range>([&] { (void)files.readRanges(path.string(), {{data.size() - 1, 2}}); }));
        CHECK(raises<zarr::NotFound>([&] { (void)files.readRanges((root / "absent.bin").string(), {{0, 1}}); }));
    }

    Server server(data);
    // Remote: one request per nonempty range, the ETag and size reported, nothing cached.
    {
        sources::EncodedFiles files(cachedIn(root / "cache"));
        const auto got = files.readRanges(server.url(), ranges);
        expect(got);
        CHECK(got.version == "\"v1\"");
        CHECK(server.requests == 4);
        CHECK(files.bytesRead() == 52 + 1 + 10 + 70000);
        CHECK(!fs::exists(zarr::decodedCachePath(root / "cache", server.url())));
        CHECK(fs::is_empty(root / "cache"));

        // The same version reads on; a replaced object raises instead of mixing bytes.
        (void)files.readRanges(server.url(), {{10, 10}}, got.version);
        server.setEtag("\"v2\"");
        CHECK(raises<sources::EncodedFiles::Changed>([&] { (void)files.readRanges(server.url(), {{10, 10}}, got.version); }));
        server.setEtag("\"v1\"");

        CHECK(raises<zarr::NotFound>([&] { (void)files.readRanges(server.url("absent.bin"), {{0, 1}}); }));
        CHECK(raises<std::exception>([&] { (void)files.readRanges(server.url(), {{data.size(), 4}}); }));

        // A server that ignores Range would send the whole file: refused, not buffered.
        server.ignoreRange = true;
        CHECK(raises<std::exception>([&] { (void)files.readRanges(server.url(), {{0, 52}}); }));
        server.ignoreRange = false;

        // Many ranges at once, in order.
        std::vector<std::pair<std::uint64_t, std::uint64_t>> many;
        for (std::uint64_t i = 0; i < 300; ++i) many.emplace_back(i * 3301 % (data.size() - 900), 1 + i * 7 % 900);
        const auto all = files.readRanges(server.url(), many);
        for (std::size_t i = 0; i < many.size(); ++i) CHECK(all.parts[i] == data.substr(many[i].first, many[i].second));
    }

    // Offline with no cached copy: a cache miss, and no request.
    {
        sources::EncodedFiles files(cachedIn(root / "cache", true));
        const int before = server.requests;
        CHECK(raises<std::runtime_error>([&] { (void)files.readRanges(server.url(), {{0, 1}}); }));
        CHECK(server.requests == before);
    }

    // A cached whole copy is read in place, online or offline, with no request.
    {
        const auto cached = zarr::decodedCachePath(root / "cache", server.url());
        fs::create_directories(cached.parent_path());
        std::ofstream(cached, std::ios::binary) << data;
        for (const bool offline : {false, true}) {
            sources::EncodedFiles files(cachedIn(root / "cache", offline));
            const int before = server.requests;
            const auto got = files.readRanges(server.url(), ranges);
            expect(got);
            CHECK(got.version.empty());
            CHECK(server.requests == before);
            CHECK(files.cacheHits() == 1);
        }
    }

    // A file the caller writes itself is charged to the root's budget like a
    // download: it evicts older cached files, and one larger than the budget is
    // refused.
    {
        const auto budgetRoot = root / "budget";
        fs::create_directories(budgetRoot);
        auto limited = [budgetRoot](const std::string& location) {
            zarr::OpenOptions options;
            options.access = zarr::resolveAccess(location, zarr::AuthMode::Anonymous);
            options.cacheDirs = {budgetRoot.string()};
            options.cacheMaxBytes = 3 << 20;  // the downloaded file (1 MiB) plus 2 MiB
            return options;
        };
        sources::EncodedFiles files(limited);
        const auto downloaded = zarr::decodedCachePath(budgetRoot, server.url());
        { auto opened = files.open(server.url()); }
        CHECK(fs::exists(downloaded));
        const auto mirror = downloaded.string() + ".partial";
        {
            std::ofstream sparse(mirror, std::ios::binary);
            sparse.seekp((64 << 20) - 1);
            sparse.put('\0');  // 64 MiB apparent, a few KiB occupied
        }
        CHECK(files.chargeFile(server.url(), mirror, 1 << 20));
        CHECK(fs::exists(downloaded));                    // 2 MiB in all: nothing evicted
        CHECK(files.chargeFile(server.url(), mirror, (5 << 19)));  // 2.5 MiB: the download goes
        CHECK(!(fs::exists(downloaded)));
        CHECK(fs::exists(mirror));
        CHECK(!files.chargeFile(server.url(), mirror, 4 << 20));  // over the whole budget
        CHECK(fs::exists(mirror));
        sources::EncodedFiles again(limited);
        // Local locations have no budget.
        CHECK(again.chargeFile((root / "local.bin").string(), (root / "local.bin").string(), 1LL << 40));
    }

    fs::remove_all(root);
    std::puts("ok");
    return 0;
}
