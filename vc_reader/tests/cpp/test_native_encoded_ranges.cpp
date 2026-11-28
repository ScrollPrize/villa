// vcr::sources::EncodedFiles::readRanges: parts of local, cached and remote files, served
// remotely without caching the whole file; missing files, changed objects, servers that
// ignore Range, and many concurrent ranges.
#include "vc_reader/file_ownership.hpp"
#include "vc_reader/source_reader.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <sys/file.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <chrono>
#include <optional>
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

    // writeFile holds the budget across a write and its accounting: a writer
    // paused between being admitted and publishing its file is not evicted
    // from under it (the file it then publishes would be on disk unaccounted),
    // and the files on disk stay within the limit.
    {
        const auto publishRoot = root / "publish";
        fs::create_directories(publishRoot);
        auto small = [publishRoot](const std::string& location) {
            zarr::OpenOptions options;
            options.access = zarr::resolveAccess(location, zarr::AuthMode::Anonymous);
            options.cacheDirs = {publishRoot.string()};
            options.cacheMaxBytes = 120;
            return options;
        };
        sources::EncodedFiles files(small);
        const auto base = zarr::decodedCachePath(publishRoot, server.url()).string();
        const std::string a = base + ".a", b = base + ".b";
        fs::create_directories(fs::path(a).parent_path());
        const auto publish = [](const std::string& path, std::size_t n) {
            const auto temporary = path + ".tmp";
            { std::ofstream(temporary, std::ios::binary) << std::string(n, 'x'); }
            fs::rename(temporary, path);
            return static_cast<std::int64_t>(fs::file_size(path));
        };
        std::atomic<bool> admitted{false};
        std::thread first([&] {
            files.writeFile(server.url(), a, [&](const sources::EncodedFiles::Fits& fits) -> std::optional<std::int64_t> {
                if (!fits(105)) return std::nullopt;
                admitted = true;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));  // paused before publishing
                return publish(a, 105);
            });
        });
        while (!admitted) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        bool secondFit = false;
        files.writeFile(server.url(), b, [&](const sources::EncodedFiles::Fits& fits) -> std::optional<std::int64_t> {
            CHECK(fs::exists(a));  // waited for the first to publish
            secondFit = fits(105);
            if (!secondFit) return std::nullopt;
            return publish(b, 105);
        });
        first.join();
        CHECK(secondFit);
        std::uintmax_t onDisk = 0;
        for (const auto& path : {a, b})
            if (fs::exists(path)) onDisk += fs::file_size(path);
        CHECK(onDisk <= 120);
        CHECK(fs::exists(b));    // the newest stays
        CHECK(!fs::exists(a));   // the older was evicted for it, after it was published
        // What does not fit is not written; a refused write leaves the file charged as it is.
        files.writeFile(server.url(), b, [&](const sources::EncodedFiles::Fits& fits) -> std::optional<std::int64_t> {
            CHECK(!fits(121));
            return static_cast<std::int64_t>(fs::file_size(b));
        });
        CHECK(fs::exists(b));
        // Local locations have no budget: anything fits.
        bool ran = false;
        files.writeFile((root / "local.bin").string(), (root / "local.bin").string(),
                        [&](const sources::EncodedFiles::Fits& fits) -> std::optional<std::int64_t> {
                            ran = fits(1LL << 40);
                            return std::nullopt;
                        });
        CHECK(ran);
    }

    // publishFile reserves before it writes, and an interrupted write is
    // reclaimed: two writers that die just before the rename (as a killed
    // process does) leave nothing a third cannot reclaim, and the disk stays
    // within the limit. Each runs in its own process.
    {
        const auto crashRoot = root / "crash";
        fs::create_directories(crashRoot);
        const auto make = [crashRoot](const std::string& location) {
            zarr::OpenOptions options;
            options.access = zarr::resolveAccess(location, zarr::AuthMode::Anonymous);
            options.cacheDirs = {crashRoot.string()};
            options.cacheMaxBytes = 200;
            return options;
        };
        const auto pack = zarr::decodedCachePath(crashRoot, server.url()).string() + ".ranges";
        fs::create_directories(fs::path(pack).parent_path());
        const auto write103 = [](int fd) {
            const std::string bytes(103, 'p');
            return ::write(fd, bytes.data(), bytes.size()) == 103;
        };
        for (int writer = 0; writer < 2; ++writer) {
            const pid_t child = ::fork();
            if (child == 0) {
                sources::EncodedFiles files(make);
                (void)files.publishFile(server.url(), pack, 103, [&](int fd) {
                    write103(fd);
                    ::_exit(0);  // interrupted after writing, before the rename
                    return true;
                });
                ::_exit(1);
            }
            int status = 0;
            ::waitpid(child, &status, 0);
            CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
        }
        std::uintmax_t left = 0;
        for (const auto& entry : fs::recursive_directory_iterator(fs::path(pack).parent_path()))
            if (entry.is_regular_file()) left += entry.file_size();
        CHECK(left > 0);  // the interrupted temporaries are still there
        {
            // A new process's budget reclaims them. On a thread of its own, so
            // its temporary is not named like the forked writers' (they share
            // this process's naming), which would replace theirs by accident.
            bool published = false;
            std::thread third([&] {
                sources::EncodedFiles files(make);
                published = files.publishFile(server.url(), pack, 103, write103);
            });
            third.join();
            CHECK(published);
        }
        std::uintmax_t onDisk = 0;
        int temporaries = 0;
        for (const auto& entry : fs::recursive_directory_iterator(fs::path(pack).parent_path())) {
            if (!entry.is_regular_file()) continue;
            onDisk += entry.file_size();
            if (entry.path().filename().string().find(".tmp.") != std::string::npos) ++temporaries;
        }
        CHECK(temporaries == 0);
        CHECK(onDisk <= 200);
        CHECK(fs::file_size(pack) == 103);
    }

    // An append interrupted after its bytes reach the disk, before its writer
    // finishes: the space it took is already charged, so another writer is not
    // admitted into it.
    {
        const auto appendRoot = root / "append";
        fs::create_directories(appendRoot);
        const auto make = [appendRoot](const std::string& location) {
            zarr::OpenOptions options;
            options.access = zarr::resolveAccess(location, zarr::AuthMode::Anonymous);
            options.cacheDirs = {appendRoot.string()};
            options.cacheMaxBytes = 150;
            return options;
        };
        const auto base = zarr::decodedCachePath(appendRoot, server.url()).string();
        const std::string a = base + ".a.ranges", b = base + ".b.ranges";
        fs::create_directories(fs::path(a).parent_path());
        const pid_t child = ::fork();
        if (child == 0) {
            sources::EncodedFiles files(make);
            files.writeFile(server.url(), a, [&](const sources::EncodedFiles::Fits& fits) -> std::optional<std::int64_t> {
                if (!fits(140)) ::_exit(2);
                { std::ofstream(a, std::ios::binary) << std::string(140, 'a'); }
                ::_exit(0);  // interrupted before the write is accounted
            });
            ::_exit(1);
        }
        int status = 0;
        ::waitpid(child, &status, 0);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
        sources::EncodedFiles files(make);
        bool fit = true;
        files.writeFile(server.url(), b, [&](const sources::EncodedFiles::Fits& fits) -> std::optional<std::int64_t> {
            fit = fits(40);
            if (fit) { std::ofstream(b, std::ios::binary) << std::string(40, 'b'); }
            return fit ? std::optional<std::int64_t>(40) : std::nullopt;
        });
        std::uintmax_t onDisk = 0;
        for (const auto& path : {a, b})
            if (fs::exists(path)) onDisk += fs::file_size(path);
        CHECK(onDisk <= 150);
    }

    // A publication pending while another process reconciles the index, then
    // interrupted just after the rename: the published file is still counted
    // when the cache is next opened.
    {
        const auto pendRoot = root / "pending";
        fs::create_directories(pendRoot);
        const auto make = [pendRoot](const std::string& location) {
            zarr::OpenOptions options;
            options.access = zarr::resolveAccess(location, zarr::AuthMode::Anonymous);
            options.cacheDirs = {pendRoot.string()};
            options.cacheMaxBytes = 200;
            return options;
        };
        const auto base = zarr::decodedCachePath(pendRoot, server.url()).string();
        const std::string pack = base + ".ranges", other = base + ".other.ranges", third = base + ".third.ranges";
        const auto dir = fs::path(pack).parent_path();
        fs::create_directories(dir);
        int filled[2], reconciled[2];
        CHECK(::pipe(filled) == 0);
        CHECK(::pipe(reconciled) == 0);
        const pid_t writer = ::fork();
        if (writer == 0) {
            sources::EncodedFiles files(make);
            (void)files.publishFile(server.url(), pack, 103, [&](int fd) {
                const std::string bytes(103, 'p');
                (void)!::write(fd, bytes.data(), bytes.size());
                char c = 1;
                (void)!::write(filled[1], &c, 1);
                (void)!::read(reconciled[0], &c, 1);  // another process reconciles meanwhile
                for (const auto& entry : fs::directory_iterator(dir))  // the rename, then the interruption
                    if (entry.path().filename().string().rfind(fs::path(pack).filename().string() + ".tmp.", 0) == 0)
                        fs::rename(entry.path(), pack);
                ::_exit(0);
                return true;
            });
            ::_exit(1);
        }
        char c = 0;
        CHECK(::read(filled[0], &c, 1) == 1);
        const pid_t reconciler = ::fork();
        if (reconciler == 0) {
            sources::EncodedFiles files(make);  // a new process: reconciles while `pack` is not there yet
            const bool ok = files.publishFile(server.url(), other, 10, [](int fd) {
                const std::string bytes(10, 'o');
                return ::write(fd, bytes.data(), bytes.size()) == 10;
            });
            ::_exit(ok ? 0 : 1);
        }
        int status = 0, wrote = 0;
        ::waitpid(reconciler, &status, 0);
        c = 1;
        (void)!::write(reconciled[1], &c, 1);  // let the writer go on before any check
        ::waitpid(writer, &wrote, 0);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
        CHECK(WIFEXITED(wrote) && WEXITSTATUS(wrote) == 0);
        CHECK(fs::file_size(pack) == 103);  // published, never accounted by its writer
        bool published = false;
        std::thread next([&] {  // its own thread: a temporary not named like the forked writer's
            sources::EncodedFiles files(make);
            published = files.publishFile(server.url(), third, 103, [](int fd) {
                const std::string bytes(103, 't');
                return ::write(fd, bytes.data(), bytes.size()) == 103;
            });
        });
        next.join();
        CHECK(published);
        std::uintmax_t onDisk = 0;
        for (const auto& entry : fs::directory_iterator(dir))
            if (entry.is_regular_file()) onDisk += entry.file_size();
        CHECK(onDisk <= 200);
        for (int fd : {filled[0], filled[1], reconciled[0], reconciled[1]}) ::close(fd);
    }

    // An append in progress while another process opens the cache and
    // rebuilds its index from the files (where the pack is still short): the
    // growth the append was admitted for stays reserved, so the other process
    // is not let into that space.
    {
        const auto liveRoot = root / "live";
        fs::create_directories(liveRoot);
        const auto make = [liveRoot](const std::string& location) {
            zarr::OpenOptions options;
            options.access = zarr::resolveAccess(location, zarr::AuthMode::Anonymous);
            options.cacheDirs = {liveRoot.string()};
            options.cacheMaxBytes = 300;
            return options;
        };
        const auto base = zarr::decodedCachePath(liveRoot, server.url()).string();
        const std::string pack = base + ".ranges", other = base + ".other.ranges";
        fs::create_directories(fs::path(pack).parent_path());
        { std::ofstream(pack, std::ios::binary) << std::string(10, 'p'); }
        int admitted[2], reconciled[2];
        CHECK(::pipe(admitted) == 0);
        CHECK(::pipe(reconciled) == 0);
        const pid_t appender = ::fork();
        if (appender == 0) {
            sources::EncodedFiles files(make);
            files.writeFile(server.url(), pack, [&](const sources::EncodedFiles::Fits& fits) -> std::optional<std::int64_t> {
                if (!fits(290)) ::_exit(2);
                char c = 1;
                (void)!::write(admitted[1], &c, 1);
                (void)!::read(reconciled[0], &c, 1);  // another process opens the cache meanwhile
                { std::ofstream(pack, std::ios::binary | std::ios::app) << std::string(280, 'p'); }
                return static_cast<std::int64_t>(fs::file_size(pack));
            });
            ::_exit(0);
        }
        char c = 0;
        CHECK(::read(admitted[0], &c, 1) == 1);
        const pid_t other_process = ::fork();
        if (other_process == 0) {
            sources::EncodedFiles files(make);  // a new process: rebuilds the index from the disk
            bool fit = false;
            files.writeFile(server.url(), other, [&](const sources::EncodedFiles::Fits& fits) -> std::optional<std::int64_t> {
                fit = fits(100);
                if (fit) { std::ofstream(other, std::ios::binary) << std::string(100, 'o'); }
                return fit ? std::optional<std::int64_t>(100) : std::nullopt;
            });
            ::_exit(fit ? 3 : 0);
        }
        int status = 0, appended = 0;
        ::waitpid(other_process, &status, 0);
        c = 1;
        (void)!::write(reconciled[1], &c, 1);  // let the append finish before any check
        ::waitpid(appender, &appended, 0);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);  // refused: 280 of the 300 are the append's
        CHECK(WIFEXITED(appended) && WEXITSTATUS(appended) == 0);
        std::uintmax_t onDisk = 0;
        for (const auto& entry : fs::directory_iterator(fs::path(pack).parent_path()))
            if (entry.is_regular_file()) onDisk += entry.file_size();
        CHECK(onDisk <= 300);
        CHECK(fs::file_size(pack) == 290);
        for (int fd : {admitted[0], admitted[1], reconciled[0], reconciled[1]}) ::close(fd);
    }

    // A pack its reader owns (vcr::ownFile) is not evicted (its owner may be
    // writing it, and unlinking it would leave its bytes on disk uncounted): a
    // write that would need its room is refused instead. Released, it goes. An
    // owner in this process or in another; and owning costs no descriptor per file.
    {
        const auto ownRoot = root / "owned";
        fs::create_directories(ownRoot);
        const auto make = [ownRoot](const std::string& location) {
            zarr::OpenOptions options;
            options.access = zarr::resolveAccess(location, zarr::AuthMode::Anonymous);
            options.cacheDirs = {ownRoot.string()};
            options.cacheMaxBytes = 200;
            return options;
        };
        const auto base = zarr::decodedCachePath(ownRoot, server.url()).string();
        const std::string held = base + ".held.ranges", next = base + ".next.ranges";
        fs::create_directories(fs::path(held).parent_path());
        const auto fill = [](std::size_t n) {
            return [n](int fd) {
                const std::string bytes(n, 'x');
                return ::write(fd, bytes.data(), bytes.size()) == static_cast<ssize_t>(n);
            };
        };
        sources::EncodedFiles files(make);
        CHECK(files.publishFile(server.url(), held, 150, fill(150)));
        {
            // Spelled through a symlink, as a reader may (macOS's /var is /private/var).
            const auto link = root / "owned-link";
            fs::create_directory_symlink(fs::path(held).parent_path(), link);
            CHECK(vcr::fileOwned(held) == false);
            auto token = vcr::ownFile((link / fs::path(held).filename()).string());  // its reader, here
            CHECK(vcr::fileOwned(held));
            CHECK(vcr::ownFile(held) == nullptr);
            CHECK(token != nullptr);
            CHECK(vcr::ownFile(held) == nullptr);  // one owner at a time
            CHECK(!files.publishFile(server.url(), next, 100, fill(100)));
            CHECK(fs::exists(held));
        }
        // Its reader in another process.
        int owned[2], release[2];
        CHECK(::pipe(owned) == 0);
        CHECK(::pipe(release) == 0);
        const pid_t owner = ::fork();
        if (owner == 0) {
            auto token = vcr::ownFile(held);
            char c = token ? 1 : 0;
            (void)!::write(owned[1], &c, 1);
            (void)!::read(release[0], &c, 1);
            ::_exit(0);
        }
        char c = 0;
        CHECK(::read(owned[0], &c, 1) == 1);
        CHECK(c == 1);
        const bool admitted = files.publishFile(server.url(), next, 100, fill(100));
        const bool ownedThere = vcr::fileOwned(held);
        c = 1;
        (void)!::write(release[1], &c, 1);
        int status = 0;
        ::waitpid(owner, &status, 0);
        CHECK(!admitted);
        CHECK(ownedThere);
        CHECK(fs::exists(held));
        CHECK(!vcr::fileOwned(held));  // released with its process
        CHECK(files.publishFile(server.url(), next, 100, fill(100)));
        CHECK(!fs::exists(held));  // evicted now, for the newer file
        for (int fd : {owned[0], owned[1], release[0], release[1]}) ::close(fd);

        // Owning many files holds one descriptor for their directory, not one each.
        const auto descriptors = [] {
            std::size_t n = 0;
            for (int fd = 0; fd < 65536; ++fd)
                if (::fcntl(fd, F_GETFD) != -1) ++n;
            return n;
        };
        const auto before = descriptors();
        std::vector<std::shared_ptr<void>> tokens;
        for (int i = 0; i < 2000; ++i) tokens.push_back(vcr::ownFile(base + ".many." + std::to_string(i) + ".ranges"));
        std::size_t got = 0;
        for (const auto& token : tokens) got += token != nullptr;
        CHECK(got >= 1990);  // a few may share a byte
        CHECK(descriptors() <= before + 1);
    }

    fs::remove_all(root);
    std::puts("ok");
    return 0;
}
