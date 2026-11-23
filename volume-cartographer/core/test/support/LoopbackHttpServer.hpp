#pragma once
// A loopback HTTP/1.1 server for boundary tests, on 127.0.0.1 at a free port.
// It serves the files of a directory, an S3 bucket (objects and ListObjectsV2
// listings, path-style /<bucket>/<key>) backed by a directory, and whatever a
// handler answers first; it logs every request. Connections are served on their
// own threads with keep-alive, as libcurl expects. Linux and macOS.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace vc::test {

class LoopbackHttpServer {
public:
    struct Request {
        std::string method;
        std::string target;                          // path and query as sent
        std::string path;                            // percent-decoded path
        std::map<std::string, std::string> query;    // decoded parameters
        std::map<std::string, std::string> headers;  // lower-case names
        std::string body;
        [[nodiscard]] std::string header(const std::string& name) const
        {
            const auto it = headers.find(name);
            return it == headers.end() ? std::string() : it->second;
        }
        // Whether the request carries an AWS Signature Version 4.
        [[nodiscard]] bool signedV4() const { return header("authorization").starts_with("AWS4-HMAC-SHA256"); }
    };
    struct Response {
        int status = 200;
        std::string body;
        std::vector<std::pair<std::string, std::string>> headers;
        bool chunked = false;  // Transfer-Encoding: chunked instead of Content-Length
    };
    // Answers a request itself, or nullopt to fall through to the directories and buckets.
    using Handler = std::function<std::optional<Response>(const Request&)>;

    LoopbackHttpServer()
    {
        listenFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listenFd_ < 0) throw std::system_error(errno, std::generic_category(), "socket");
        const int one = 1;
        ::setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = 0;
        if (::bind(listenFd_, reinterpret_cast<sockaddr*>(&address), sizeof address) != 0 ||
            ::listen(listenFd_, 64) != 0)
            throw std::system_error(errno, std::generic_category(), "bind/listen on 127.0.0.1");
        socklen_t length = sizeof address;
        if (::getsockname(listenFd_, reinterpret_cast<sockaddr*>(&address), &length) != 0)
            throw std::system_error(errno, std::generic_category(), "getsockname");
        port_ = ntohs(address.sin_port);
        acceptThread_ = std::thread([this] { acceptLoop(); });
    }

    ~LoopbackHttpServer()
    {
        stop_.store(true);
        ::shutdown(listenFd_, SHUT_RDWR);
        ::close(listenFd_);
        if (acceptThread_.joinable()) acceptThread_.join();
        std::vector<std::thread> workers;
        {
            std::lock_guard lock(mutex_);
            for (int fd : connectionFds_) ::shutdown(fd, SHUT_RDWR);
            workers.swap(connectionThreads_);
        }
        for (auto& worker : workers)
            if (worker.joinable()) worker.join();
    }
    LoopbackHttpServer(const LoopbackHttpServer&) = delete;
    LoopbackHttpServer& operator=(const LoopbackHttpServer&) = delete;

    [[nodiscard]] int port() const { return port_; }
    [[nodiscard]] std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }

    // Files below `root` at paths below `prefix`: GET and HEAD with Content-Length, byte
    // ranges, 404 when missing. "/" serves the root at the server's root.
    void serveDirectory(std::filesystem::path root, std::string prefix = "/")
    {
        std::lock_guard lock(mutex_);
        directories_.push_back({normalizedPrefix(prefix), std::move(root)});
    }
    // An S3 bucket at /<bucket>/<key> backed by the files below `root`: GET, HEAD and
    // ListObjectsV2 (prefix, delimiter, max-keys, continuation-token). With
    // `requireSignature`, requests without a SigV4 Authorization header get 403.
    void serveBucket(std::string bucket, std::filesystem::path root, bool requireSignature = false)
    {
        std::lock_guard lock(mutex_);
        buckets_.push_back({std::move(bucket), std::move(root), requireSignature});
    }
    void setHandler(Handler handler)
    {
        std::lock_guard lock(mutex_);
        handler_ = std::move(handler);
    }

    [[nodiscard]] std::vector<Request> requests() const
    {
        std::lock_guard lock(mutex_);
        return requests_;
    }
    // Requests whose decoded path is `path` (any method).
    [[nodiscard]] std::size_t requestCount(std::string_view path) const
    {
        std::lock_guard lock(mutex_);
        return std::size_t(std::count_if(requests_.begin(), requests_.end(),
                                         [&](const Request& r) { return r.path == path; }));
    }
    void clearRequests()
    {
        std::lock_guard lock(mutex_);
        requests_.clear();
    }

    // Percent-decoding as the server applies it to paths and query values.
    static std::string percentDecode(std::string_view text)
    {
        std::string out;
        out.reserve(text.size());
        for (std::size_t i = 0; i < text.size(); ++i) {
            if (text[i] == '%' && i + 2 < text.size() && std::isxdigit(static_cast<unsigned char>(text[i + 1])) &&
                std::isxdigit(static_cast<unsigned char>(text[i + 2]))) {
                out.push_back(static_cast<char>(std::stoi(std::string(text.substr(i + 1, 2)), nullptr, 16)));
                i += 2;
            } else {
                out.push_back(text[i]);
            }
        }
        return out;
    }
    static std::string xmlEscape(std::string_view text)
    {
        std::string out;
        for (char c : text) {
            switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            default: out.push_back(c);
            }
        }
        return out;
    }

private:
    struct Directory {
        std::string prefix;  // starts and ends with '/'
        std::filesystem::path root;
    };
    struct Bucket {
        std::string name;
        std::filesystem::path root;
        bool requireSignature = false;
    };

    static std::string normalizedPrefix(std::string prefix)
    {
        if (prefix.empty() || prefix.front() != '/') prefix.insert(prefix.begin(), '/');
        if (prefix.back() != '/') prefix.push_back('/');
        return prefix;
    }

    void acceptLoop()
    {
        while (!stop_.load()) {
            pollfd waiting{listenFd_, POLLIN, 0};
            if (::poll(&waiting, 1, 100) <= 0) continue;
            const int fd = ::accept(listenFd_, nullptr, nullptr);
            if (fd < 0) {
                if (stop_.load()) return;
                continue;
            }
            std::lock_guard lock(mutex_);
            connectionFds_.push_back(fd);
            connectionThreads_.emplace_back([this, fd] { serveConnection(fd); });
        }
    }

    void serveConnection(int fd)
    {
        std::string buffer;
        char block[1 << 14];
        bool keepAlive = true;
        while (keepAlive && !stop_.load()) {
            // One request: head up to the blank line, then Content-Length bytes of body.
            std::size_t headEnd;
            while ((headEnd = buffer.find("\r\n\r\n")) == std::string::npos) {
                const ssize_t n = ::recv(fd, block, sizeof block, 0);
                if (n <= 0) { keepAlive = false; break; }
                buffer.append(block, std::size_t(n));
            }
            if (!keepAlive) break;
            Request request;
            std::string version;
            if (!parseHead(std::string_view(buffer).substr(0, headEnd), request, version)) {
                sendResponse(fd, {400, "bad request", {}, false}, false);
                break;
            }
            buffer.erase(0, headEnd + 4);
            std::size_t bodyLength = 0;
            if (const auto length = request.header("content-length"); !length.empty())
                bodyLength = std::stoull(length);
            while (buffer.size() < bodyLength) {
                const ssize_t n = ::recv(fd, block, sizeof block, 0);
                if (n <= 0) { keepAlive = false; break; }
                buffer.append(block, std::size_t(n));
            }
            if (!keepAlive) break;
            request.body = buffer.substr(0, bodyLength);
            buffer.erase(0, bodyLength);
            const auto connection = request.header("connection");
            keepAlive = version == "HTTP/1.1" ? connection != "close" : connection == "keep-alive";
            const Response response = respond(request);
            if (!sendResponse(fd, response, request.method == "HEAD")) break;
        }
        ::close(fd);
        std::lock_guard lock(mutex_);
        connectionFds_.erase(std::remove(connectionFds_.begin(), connectionFds_.end(), fd), connectionFds_.end());
    }

    static bool parseHead(std::string_view head, Request& request, std::string& version)
    {
        const auto lineEnd = head.find("\r\n");
        const auto line = head.substr(0, lineEnd);
        const auto firstSpace = line.find(' '), lastSpace = line.rfind(' ');
        if (firstSpace == std::string_view::npos || lastSpace == firstSpace) return false;
        request.method = std::string(line.substr(0, firstSpace));
        request.target = std::string(line.substr(firstSpace + 1, lastSpace - firstSpace - 1));
        version = std::string(line.substr(lastSpace + 1));
        const auto questionMark = request.target.find('?');
        request.path = percentDecode(request.target.substr(0, questionMark));
        if (questionMark != std::string::npos) {
            std::string_view query = std::string_view(request.target).substr(questionMark + 1);
            while (!query.empty()) {
                const auto amp = query.find('&');
                const auto pair = query.substr(0, amp);
                const auto eq = pair.find('=');
                request.query[percentDecode(pair.substr(0, eq))] =
                    eq == std::string_view::npos ? std::string() : percentDecode(pair.substr(eq + 1));
                if (amp == std::string_view::npos) break;
                query.remove_prefix(amp + 1);
            }
        }
        std::string_view rest = lineEnd == std::string_view::npos ? std::string_view() : head.substr(lineEnd + 2);
        while (!rest.empty()) {
            const auto end = rest.find("\r\n");
            const auto header = rest.substr(0, end);
            const auto colon = header.find(':');
            if (colon != std::string_view::npos) {
                std::string name(header.substr(0, colon));
                std::transform(name.begin(), name.end(), name.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                auto value = header.substr(colon + 1);
                while (!value.empty() && value.front() == ' ') value.remove_prefix(1);
                while (!value.empty() && value.back() == ' ') value.remove_suffix(1);
                request.headers[name] = std::string(value);
            }
            if (end == std::string_view::npos) break;
            rest.remove_prefix(end + 2);
        }
        return true;
    }

    static const char* statusText(int status)
    {
        switch (status) {
        case 200: return "OK";
        case 204: return "No Content";
        case 206: return "Partial Content";
        case 304: return "Not Modified";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 412: return "Precondition Failed";
        case 416: return "Range Not Satisfiable";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default: return "Unknown";
        }
    }

    static bool sendAll(int fd, std::string_view data)
    {
        while (!data.empty()) {
            const ssize_t n = ::send(fd, data.data(), data.size(), MSG_NOSIGNAL);
            if (n <= 0) return false;
            data.remove_prefix(std::size_t(n));
        }
        return true;
    }

    static bool sendResponse(int fd, const Response& response, bool headOnly)
    {
        std::string head = "HTTP/1.1 " + std::to_string(response.status) + " " + statusText(response.status) + "\r\n";
        for (const auto& [name, value] : response.headers) head += name + ": " + value + "\r\n";
        if (response.chunked) {
            head += "Transfer-Encoding: chunked\r\n\r\n";
            if (!sendAll(fd, head)) return false;
            if (headOnly) return true;
            // Two chunks, so a reader sees growth, then the terminator.
            const auto half = response.body.size() / 2;
            for (const auto part : {std::string_view(response.body).substr(0, half), std::string_view(response.body).substr(half)}) {
                if (part.empty()) continue;
                char size[32];
                std::snprintf(size, sizeof size, "%zx\r\n", part.size());
                if (!sendAll(fd, size) || !sendAll(fd, part) || !sendAll(fd, "\r\n")) return false;
            }
            return sendAll(fd, "0\r\n\r\n");
        }
        head += "Content-Length: " + std::to_string(response.body.size()) + "\r\n\r\n";
        if (!sendAll(fd, head)) return false;
        return headOnly || sendAll(fd, response.body);
    }

    Response respond(const Request& request)
    {
        Handler handler;
        std::vector<Directory> directories;
        std::vector<Bucket> buckets;
        {
            std::lock_guard lock(mutex_);
            requests_.push_back(request);
            handler = handler_;
            directories = directories_;
            buckets = buckets_;
        }
        if (handler)
            if (auto answer = handler(request)) return *answer;
        for (const auto& bucket : buckets) {
            const std::string base = "/" + bucket.name;
            if (request.path != base && !request.path.starts_with(base + "/")) continue;
            if (bucket.requireSignature && !request.signedV4())
                return {403, "<Error><Code>AccessDenied</Code><Message>unsigned request</Message></Error>", {{"Content-Type", "application/xml"}}, false};
            if (request.query.count("list-type")) return listing(bucket, request);
            if (request.path.size() <= base.size() + 1) return {404, "", {}, false};
            return file(bucket.root / request.path.substr(base.size() + 1), request);
        }
        for (const auto& directory : directories) {
            if (!request.path.starts_with(directory.prefix)) continue;
            return file(directory.root / request.path.substr(directory.prefix.size()), request);
        }
        return {404, "not found: " + request.path, {}, false};
    }

    static Response file(const std::filesystem::path& path, const Request& request)
    {
        std::error_code ec;
        if (!std::filesystem::is_regular_file(path, ec)) return {404, "", {}, false};
        if (request.method != "GET" && request.method != "HEAD") return {405, "", {}, false};
        std::ifstream in(path, std::ios::binary);
        std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        Response response;
        response.headers.emplace_back("Content-Type", "application/octet-stream");
        response.headers.emplace_back("Accept-Ranges", "bytes");
        if (const auto range = request.header("range"); range.starts_with("bytes=")) {
            const auto dash = range.find('-');
            const std::string firstText = range.substr(6, dash - 6), lastText = range.substr(dash + 1);
            std::size_t first = 0, last = data.empty() ? 0 : data.size() - 1;
            if (firstText.empty()) first = data.size() - std::min(data.size(), std::size_t(std::stoull(lastText)));
            else {
                first = std::stoull(firstText);
                if (!lastText.empty()) last = std::min(last, std::size_t(std::stoull(lastText)));
            }
            if (first > last || first >= data.size())
                return {416, "", {{"Content-Range", "bytes */" + std::to_string(data.size())}}, false};
            response.status = 206;
            response.headers.emplace_back("Content-Range", "bytes " + std::to_string(first) + "-" + std::to_string(last) + "/" +
                                                               std::to_string(data.size()));
            response.body = data.substr(first, last - first + 1);
            return response;
        }
        response.body = std::move(data);
        return response;
    }

    // ListObjectsV2 over the bucket's files: keys in order, common prefixes rolled up at
    // the delimiter, pages of max-keys with a continuation token (an item index).
    static Response listing(const Bucket& bucket, const Request& request)
    {
        const auto value = [&](const char* name) {
            const auto it = request.query.find(name);
            return it == request.query.end() ? std::string() : it->second;
        };
        const std::string prefix = value("prefix"), delimiter = value("delimiter");
        int maxKeys = 1000;
        if (const auto text = value("max-keys"); !text.empty()) maxKeys = std::stoi(text);
        std::size_t start = 0;
        if (const auto token = value("continuation-token"); !token.empty()) start = std::stoull(token);

        std::vector<std::pair<std::string, std::uintmax_t>> objects;  // key, size
        std::error_code ec;
        if (std::filesystem::is_directory(bucket.root, ec))
            for (auto it = std::filesystem::recursive_directory_iterator(bucket.root, ec);
                 it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
                if (ec || !it->is_regular_file(ec)) continue;
                objects.emplace_back(it->path().lexically_relative(bucket.root).generic_string(), it->file_size(ec));
            }
        std::sort(objects.begin(), objects.end());
        struct Item { std::string key; std::uintmax_t size; bool common; };
        std::vector<Item> items;
        for (const auto& [key, size] : objects) {
            if (!key.starts_with(prefix)) continue;
            if (!delimiter.empty()) {
                const auto at = key.find(delimiter, prefix.size());
                if (at != std::string::npos) {
                    const auto common = key.substr(0, at + delimiter.size());
                    if (items.empty() || !items.back().common || items.back().key != common) items.push_back({common, 0, true});
                    continue;
                }
            }
            items.push_back({key, size, false});
        }
        std::string xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                          "<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
                          "<Name>" + xmlEscape(bucket.name) + "</Name><Prefix>" + xmlEscape(prefix) + "</Prefix>";
        const std::size_t end = std::min(items.size(), start + std::size_t(std::max(maxKeys, 1)));
        xml += "<KeyCount>" + std::to_string(end - start) + "</KeyCount><MaxKeys>" + std::to_string(maxKeys) + "</MaxKeys>";
        if (!delimiter.empty()) xml += "<Delimiter>" + xmlEscape(delimiter) + "</Delimiter>";
        xml += std::string("<IsTruncated>") + (end < items.size() ? "true" : "false") + "</IsTruncated>";
        if (end < items.size()) xml += "<NextContinuationToken>" + std::to_string(end) + "</NextContinuationToken>";
        for (std::size_t i = start; i < end; ++i) {
            if (items[i].common) xml += "<CommonPrefixes><Prefix>" + xmlEscape(items[i].key) + "</Prefix></CommonPrefixes>";
            else xml += "<Contents><Key>" + xmlEscape(items[i].key) + "</Key><Size>" + std::to_string(items[i].size) +
                        "</Size><ETag>\"loopback\"</ETag><StorageClass>STANDARD</StorageClass></Contents>";
        }
        xml += "</ListBucketResult>";
        return {200, std::move(xml), {{"Content-Type", "application/xml"}}, false};
    }

    int listenFd_ = -1;
    int port_ = 0;
    std::atomic<bool> stop_{false};
    std::thread acceptThread_;
    mutable std::mutex mutex_;
    std::vector<int> connectionFds_;
    std::vector<std::thread> connectionThreads_;
    std::vector<Directory> directories_;
    std::vector<Bucket> buckets_;
    Handler handler_;
    std::vector<Request> requests_;
};

}  // namespace vc::test
