// loadCatalog reports whether the catalog it returns is the one in its cache file,
// which is what readers of the cache file (cachedManifest) see.
#include "vc_reader/catalog.hpp"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
using namespace vcr::catalog;
namespace fs = std::filesystem;
void require(bool v, const char* message) { if (!v) throw std::runtime_error(message); }

std::string manifestNaming(const std::string& sample)
{
    return R"({"samples": {")" + sample + R"(": {"sample": {"id": ")" + sample + R"(", "properties": {}}}}})";
}

// Serves one body to every GET, one connection per request.
class Server {
public:
    explicit Server(std::string body) : body_(std::move(body))
    {
        fd_ = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        socklen_t length = sizeof(address);
        require(fd_ >= 0 && bind(fd_, reinterpret_cast<sockaddr*>(&address), length) == 0 && listen(fd_, 8) == 0 &&
                getsockname(fd_, reinterpret_cast<sockaddr*>(&address), &length) == 0, "test server");
        port_ = ntohs(address.sin_port);
        thread_ = std::thread([this] {
            while (true) {
                const int client = accept(fd_, nullptr, nullptr);
                if (stop_) { if (client >= 0) close(client); return; }
                if (client < 0) continue;
                std::string request;
                char block[4096];
                while (request.find("\r\n\r\n") == std::string::npos) {
                    const auto n = recv(client, block, sizeof(block), 0);
                    if (n <= 0) break;
                    request.append(block, n);
                }
                ++requests;
                const auto response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
                                      std::to_string(body_.size()) + "\r\nConnection: close\r\n\r\n" + body_;
                for (std::size_t sent = 0; sent < response.size();) {
                    const auto n = send(client, response.data() + sent, response.size() - sent, 0);
                    if (n <= 0) break;
                    sent += n;
                }
                close(client);
            }
        });
    }
    ~Server()
    {
        stop_ = true;
        // Wake the accept.
        const int wake = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(port_);
        (void)connect(wake, reinterpret_cast<sockaddr*>(&address), sizeof(address));
        thread_.join();
        close(wake);
        close(fd_);
    }
    std::string url() const { return "http://127.0.0.1:" + std::to_string(port_) + "/catalog.json"; }
    std::atomic<int> requests{0};

private:
    std::string body_;
    int fd_ = -1;
    int port_ = 0;
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

bool names(const OpenDataManifest* manifest, const char* sample)
{
    return manifest && manifest->samples.size() == 1 && manifest->samples.front().id == sample;
}

int main()
{
    const auto root = fs::temp_directory_path() / ("vcr_catalog_cache_" + std::to_string(getpid()));
    fs::remove_all(root);
    fs::create_directories(root);
    const auto readOnly = root / "read-only";
    int status = 0;
    try {
        std::optional<Server> server(std::in_place, manifestNaming("Live"));
        const auto url = server->url();

        // A written cache holds the catalog returned.
        const auto written = root / "written" / "metadata.json";
        auto loaded = loadCatalog(url, {written, true});
        require(loaded.onDisk && !loaded.fromCache && loaded.warning.empty(), "a written download is on disk");
        require(names(&loaded.manifest, "Live") && names(cachedManifest(written).get(), "Live"), "the cache file holds it");
        const int asked = server->requests;
        loaded = loadCatalog(url, {written});
        require(loaded.onDisk && loaded.fromCache && server->requests == asked, "a fresh cache is on disk");

        // A cache that cannot be written: the download is returned, but it is not on disk.
        const auto notADirectory = root / "file";
        std::ofstream(notADirectory) << "";
        const auto unwritable = notADirectory / "cache" / "metadata.json";
        loaded = loadCatalog(url, {unwritable, true});
        require(names(&loaded.manifest, "Live") && !loaded.onDisk && !loaded.warning.empty(), "an unwritten download is not on disk");
        require(!cachedManifest(unwritable), "and the cache file has none");

        // An older body without its sidecar is ignored by the loader, but cache-file readers still see it.
        // The download that cannot replace it is not on disk.
        if (geteuid() != 0) {
            fs::create_directories(readOnly);
            const auto older = readOnly / "metadata.json";
            std::ofstream(older) << manifestNaming("Older");
            fs::permissions(readOnly, fs::perms::owner_read | fs::perms::owner_exec);
            loaded = loadCatalog(url, {older, true});
            require(names(&loaded.manifest, "Live") && !loaded.onDisk && !loaded.warning.empty(),
                    "a download over an older unsidecared body is not on disk");
            require(names(cachedManifest(older).get(), "Older"), "readers of the cache file see the older catalog");
        } else {
            std::cout << "skipping the read-only cache directory as root\n";
        }

        // When the source cannot be reached, the cached catalog it falls back to is on disk.
        server.reset();
        loaded = loadCatalog(url, {written, true, 3600, 5});
        require(loaded.onDisk && loaded.fromCache && !loaded.warning.empty() && names(&loaded.manifest, "Live"),
                "a fallback to the cache is on disk");

        // A local file has no cache file.
        loaded = loadCatalog(written.string());
        require(!loaded.onDisk && names(&loaded.manifest, "Live"), "a local file is not a cache");
        std::cout << "ok\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        status = 1;
    }
    std::error_code ignored;
    fs::permissions(readOnly, fs::perms::owner_all, ignored);
    fs::remove_all(root, ignored);
    return status;
}
