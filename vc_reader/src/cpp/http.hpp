// HTTP(S) requests over a bounded pool of libcurl handles: anonymous, HTTP basic
// auth, or signed for S3 (AWS Signature Version 4). sftp:// and ssh:// URLs are
// read through pooled OpenSSH sessions and answered with the same responses.
#pragma once

#include "reader_types.hpp"
#include "sigv4.hpp"
#include "vc_reader/sftp.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <functional>
#include <exception>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace vcr {

struct HttpOptions {
    int connections = 16;          // concurrent requests (and kept-alive connections)
    long connect_timeout_ms = 10000;
    long stall_timeout_s = 10;     // abort an attempt after this long without receiving data
    int retries = 5;
    Auth auth;
    long request_timeout_ms = 0;
};

struct HttpResponse {
    long status = 0;
    std::string etag, modified, content_range;
    std::size_t bytes = 0;
    std::optional<std::size_t> content_length;
    std::string content_encoding;
};

class HttpClient {
public:
    explicit HttpClient(HttpOptions options) : options_(options)
    {
        static std::once_flag init;
        std::call_once(init, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
        if (options_.connections < 1)
            options_.connections = 1;
        handles_.reserve(options_.connections);
        for (int i = 0; i < options_.connections; ++i) {
            CURL* h = curl_easy_init();
            if (!h)
                throw std::runtime_error("curl_easy_init failed");
            handles_.push_back(h);
        }
        free_ = handles_;
        auth_ = std::make_shared<const Auth>(options_.auth);
        if (!auth_->kind.empty() && auth_->kind != "basic" && auth_->kind != "s3")
            throw std::invalid_argument("unknown auth kind '" + auth_->kind + "'");
    }

    // Replaces the credentials of later requests (e.g. refreshed temporary S3 keys).
    void set_auth(Auth auth)
    {
        auto next = std::make_shared<const Auth>(std::move(auth));
        std::lock_guard lock(mutex_);
        auth_ = std::move(next);
    }

    ~HttpClient()
    {
        for (CURL* h : handles_)
            curl_easy_cleanup(h);
    }

    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;

    // GET `url` into `body`. Returns false for 404 (not stored). Retries
    // connection errors, stalls, 429 and 5xx with exponential backoff; 401/403
    // throw HttpDenied and any other status throws.
    bool get(const std::string& url, std::vector<std::uint8_t>& body, std::size_t size_hint = 0)
    {
        return get_stream(url, [&] {
            body.clear();
            if (size_hint) body.reserve(size_hint);
        }, [&](const char* data, std::size_t n) {
            body.insert(body.end(), data, data + n);
        }).status == 200;
    }

    HttpResponse get_range(const std::string& url, std::vector<std::uint8_t>& body,
                           const std::string& range, std::size_t length, const std::string& etag = "")
    {
        std::vector<std::pair<std::string, std::string>> headers{{"Range", "bytes=" + range},
                                                               {"Accept-Encoding", "identity"}};
        if (!etag.empty() && etag.rfind("W/", 0) != 0) headers.emplace_back("If-Match", etag);
        auto response = get_stream(url, [&] { body.clear(); }, [&](const char* p, std::size_t n) {
            if (n > length - body.size()) throw std::runtime_error("HTTP range exceeds requested length");
            body.insert(body.end(), p, p+n);
        }, headers, 206);
        if (response.status != 404 && (response.status != 206 || body.size() != length))
            throw std::runtime_error("server did not return the requested HTTP byte range: " + url);
        return response;
    }

    // The same transport/auth/retry machinery as chunk reads, with a bounded
    // caller-owned sink. Exceptions from callbacks never cross the C ABI.
    HttpResponse get_stream(const std::string& url, std::function<void()> reset,
                            std::function<void(const char*, std::size_t)> write,
                            const std::vector<std::pair<std::string, std::string>>& extra = {}, long body_status = 200,
                            std::function<void(const HttpResponse&)> begin_body = {})
    {
        if (sftp::isSftpUrl(url))
            return get_sftp(url, reset, write, extra, body_status, begin_body);
        return perform("GET", url, reset, std::move(write), extra, body_status, std::move(begin_body), nullptr, 0);
    }

    // PUT `size` bytes to `url`, with the retries of a read: replacing an object is
    // idempotent. Throws unless the server accepted it.
    void put(const std::string& url, const void* data, std::size_t size)
    {
        const auto response = perform("PUT", url, [] {}, [](const char*, std::size_t) {}, {}, 200, {}, data, size);
        if (response.status / 100 != 2)
            throw std::runtime_error("PUT " + url + ": HTTP " + std::to_string(response.status));
    }

    // Whether `url` exists (HEAD): false for 404.
    bool head(const std::string& url)
    {
        return perform("HEAD", url, [] {}, [](const char*, std::size_t) {}, {}, 200, {}, nullptr, 0).status / 100 == 2;
    }

    // DELETE `url`. S3 answers the same whether or not the object was there.
    void remove(const std::string& url)
    {
        perform("DELETE", url, [] {}, [](const char*, std::size_t) {}, {}, 204, {}, nullptr, 0);
    }

private:
    HttpResponse perform(const char* method, const std::string& url, const std::function<void()>& reset,
                         std::function<void(const char*, std::size_t)> write,
                         const std::vector<std::pair<std::string, std::string>>& extra, long body_status,
                         std::function<void(const HttpResponse&)> begin_body, const void* upload, std::size_t upload_size)
    {
        Lease lease(*this);
        CURL* h = lease.handle;
        std::shared_ptr<const Auth> auth;
        { std::lock_guard lock(mutex_); auth = auth_; }
        float delay = 0.25f;
        std::string last_error;
        const bool get = std::strcmp(method, "GET") == 0, put = std::strcmp(method, "PUT") == 0;
        const std::string payload = put && auth->kind == "s3" ? payload_hash(upload, upload_size) : sigv4_empty_payload();
        for (int attempt = 0; attempt <= options_.retries; ++attempt) {
            reset();
            Transfer transfer{write, body_status, begin_body};
            curl_easy_reset(h);
            curl_easy_setopt(h, CURLOPT_URL, url.c_str());
            curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
            curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, get ? 1L : 0L);  // a redirected write is an error
            curl_easy_setopt(h, CURLOPT_USERAGENT, "vc_reader");
            curl_easy_setopt(h, CURLOPT_ACCEPT_ENCODING, "");
            curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT_MS, options_.connect_timeout_ms);
            curl_easy_setopt(h, CURLOPT_TIMEOUT_MS, options_.request_timeout_ms);
            curl_easy_setopt(h, CURLOPT_LOW_SPEED_LIMIT, 1L);
            curl_easy_setopt(h, CURLOPT_LOW_SPEED_TIME, options_.stall_timeout_s);
            curl_easy_setopt(h, CURLOPT_TCP_KEEPALIVE, 1L);
            curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, &HttpClient::receive);
            curl_easy_setopt(h, CURLOPT_WRITEDATA, &transfer);
            curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, &HttpClient::header);
            curl_easy_setopt(h, CURLOPT_HEADERDATA, &transfer);
            Upload body{static_cast<const char*>(upload), upload_size};
            HeaderList headers;
            if (put) {
                curl_easy_setopt(h, CURLOPT_UPLOAD, 1L);
                curl_easy_setopt(h, CURLOPT_READFUNCTION, &HttpClient::send);
                curl_easy_setopt(h, CURLOPT_READDATA, &body);
                curl_easy_setopt(h, CURLOPT_INFILESIZE_LARGE, static_cast<curl_off_t>(upload_size));
                headers.append("Expect:");  // no extra round trip per chunk
            } else if (std::strcmp(method, "HEAD") == 0) {
                curl_easy_setopt(h, CURLOPT_NOBODY, 1L);
            } else if (!get) {
                curl_easy_setopt(h, CURLOPT_CUSTOMREQUEST, method);
            }
            if (auth->kind == "basic") {
                curl_easy_setopt(h, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
                curl_easy_setopt(h, CURLOPT_USERNAME, auth->user.c_str());
                curl_easy_setopt(h, CURLOPT_PASSWORD, auth->secret.c_str());
            }
            if (auth->kind == "s3") {
                const S3Key key{auth->user, auth->secret, auth->token, auth->region};
                for (const auto& [name, value] : sigv4_headers(method, url, key, amz_date(std::time(nullptr)), extra, payload))
                    headers.append(name + ": " + value);
            } else {
                for (const auto& [name, value] : extra) headers.append(name + ": " + value);
            }
            curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers.list);
            const CURLcode rc = curl_easy_perform(h);
            if (transfer.error) std::rethrow_exception(transfer.error);
            if (rc == CURLE_OK) {
                long status = 0;
                curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
                transfer.response.status = status;
                if (status == body_status || status == 304 || status == 404 || (!get && status / 100 == 2))
                    return transfer.response;
                if (status != 429 && status < 500) {
                    const std::string what = std::string(method) + " " + url + ": HTTP " + std::to_string(status) + ": " + transfer.snippet;
                    if (status == 401 || status == 403) throw HttpDenied(what);
                    throw std::runtime_error(what);
                }
                last_error = "HTTP " + std::to_string(status);
            } else last_error = curl_easy_strerror(rc);
            if (attempt < options_.retries) {
                std::this_thread::sleep_for(std::chrono::duration<float>(delay));
                delay = std::min(delay * 2, 8.0f);
            }
        }
        throw std::runtime_error(std::string(method) + " " + url + " failed after " + std::to_string(options_.retries + 1) +
                                 " attempts: " + last_error);
    }

    static std::string payload_hash(const void* data, std::size_t size)
    {
        Sha256 hash;
        hash.update(data, size);
        return to_hex(hash.finish());
    }

    // sftp:// and ssh:// objects through pooled OpenSSH sessions, answered like HTTP:
    // 200/206 with the body, 404 when absent, HttpDenied when not permitted. Range
    // and If-Match headers are honoured; the etag is the file's mtime and size.
    HttpResponse get_sftp(const std::string& url, const std::function<void()>& reset,
                          const std::function<void(const char*, std::size_t)>& write,
                          const std::vector<std::pair<std::string, std::string>>& extra, long body_status,
                          const std::function<void(const HttpResponse&)>& begin_body)
    {
        sftp::ByteRange range;
        bool ranged = false;
        std::string if_match;
        for (const auto& [name, value] : extra) {
            std::string lower = name;
            for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (lower == "range") {
                ranged = true;
                range = parse_byte_range(value);
            } else if (lower == "if-match") {
                if_match = value;
            }
        }
        sftp::Fetcher* fetcher;
        {
            std::lock_guard lock(mutex_);
            if (!sftp_) {
                sftp::FetchOptions options;
                options.session.connectTimeout =
                    std::chrono::seconds(std::max<long>(1, (options_.connect_timeout_ms + 999) / 1000));
                options.session.idleTimeout = std::chrono::milliseconds(std::max<long>(100, options_.stall_timeout_s * 1000));
                options.sessionsPerHost = static_cast<std::size_t>(std::clamp(options_.connections, 1, 8));
                options.retries = options_.retries;
                sftp_ = std::make_unique<sftp::Fetcher>(std::move(options));
            }
            fetcher = sftp_.get();
        }
        HttpResponse response;
        std::optional<sftp::Attributes> found;
        try {
            found = fetcher->read(url, range, [&] { reset(); response = {}; }, [&](const sftp::Attributes& a, uint64_t first, uint64_t count) {
                if (ranged && !range.suffix && count == 0 && range.length.value_or(1) != 0)
                    throw std::runtime_error("sftp range starts past the end of " + url);
                response.status = ranged ? 206 : 200;
                response.etag = "\"" + std::to_string(a.modified.value_or(0)) + "-" + std::to_string(*a.size) + "\"";
                if (!if_match.empty() && if_match != response.etag)
                    throw std::runtime_error("sftp object changed between reads: " + url);
                if (ranged)
                    response.content_range = "bytes " + std::to_string(first) + "-" +
                                             std::to_string(first + count - 1) + "/" + std::to_string(*a.size);
                response.content_length = count;
                if (response.status != body_status)
                    throw std::runtime_error("unexpected sftp response for " + url);
                if (begin_body) begin_body(response);
            }, [&](std::span<const std::byte> data) {
                write(reinterpret_cast<const char*>(data.data()), data.size());
                response.bytes += data.size();
            });
        } catch (const sftp::StatusError& e) {
            if (e.status == sftp::PermissionDenied) throw HttpDenied("GET " + url + ": " + e.what());
            throw std::runtime_error("GET " + url + ": " + e.what());
        }
        if (!found) {
            response = {};
            response.status = 404;
        }
        return response;
    }

    static sftp::ByteRange parse_byte_range(const std::string& value)
    {
        unsigned long long first = 0, last = 0;
        sftp::ByteRange range;
        if (std::sscanf(value.c_str(), "bytes=-%llu", &last) == 1) {
            range.suffix = true;
            range.length = last;
        } else if (std::sscanf(value.c_str(), "bytes=%llu-%llu", &first, &last) == 2 && last >= first) {
            range.offset = first;
            range.length = last - first + 1;
        } else {
            throw std::invalid_argument("unsupported byte range for sftp: " + value);
        }
        return range;
    }

    struct HeaderList {
        curl_slist* list = nullptr;
        void append(const std::string& line)
        {
            curl_slist* next = curl_slist_append(list, line.c_str());
            if (!next)
                throw std::runtime_error("curl_slist_append failed");
            list = next;
        }
        ~HeaderList() { curl_slist_free_all(list); }
    };

    struct Lease {
        HttpClient& client;
        CURL* handle;
        explicit Lease(HttpClient& c) : client(c)
        {
            std::unique_lock lock(client.mutex_);
            client.cv_.wait(lock, [&] { return !client.free_.empty(); });
            handle = client.free_.back();
            client.free_.pop_back();
        }
        ~Lease()
        {
            {
                std::lock_guard lock(client.mutex_);
                client.free_.push_back(handle);
            }
            client.cv_.notify_one();
        }
    };

    struct Transfer {
        std::function<void(const char*, std::size_t)>& sink;
        long body_status;
        std::function<void(const HttpResponse&)>& begin_body;
        HttpResponse response;
        std::string snippet;
        std::exception_ptr error;
    };

    struct Upload {
        const char* data;
        std::size_t left;
    };

    static std::size_t send(char* out, std::size_t size, std::size_t count, void* user) noexcept
    {
        auto& upload = *static_cast<Upload*>(user);
        const auto n = std::min(size * count, upload.left);
        std::memcpy(out, upload.data, n);
        upload.data += n;
        upload.left -= n;
        return n;
    }

    static std::size_t receive(char* data, std::size_t size, std::size_t count, void* user) noexcept
    {
        auto& t = *static_cast<Transfer*>(user);
        const auto n = size * count;
        try {
            if (t.body_status == 206 && t.response.status == 200)
                throw std::runtime_error("server ignored HTTP Range; refusing a whole-shard download");
            if (t.response.status == t.body_status) {
                if (!t.response.bytes && t.begin_body) t.begin_body(t.response);
                t.sink(data, n);
                t.response.bytes += n;
            } else if (t.snippet.size() < 300) t.snippet.append(data, std::min(n, 300-t.snippet.size()));
            return n;
        } catch (...) { t.error = std::current_exception(); return 0; }
    }

    static std::size_t header(char* data, std::size_t size, std::size_t count, void* user) noexcept
    {
        auto& t = *static_cast<Transfer*>(user);
        const auto n = size * count;
        try {
            std::string line(data, n);
            if (line.rfind("HTTP/", 0) == 0) {
                t.response = {};
                t.snippet.clear();
                std::sscanf(line.c_str(), "%*s %ld", &t.response.status);
            } else if (auto colon = line.find(':'); colon != std::string::npos) {
                std::string name = line.substr(0, colon), value = line.substr(colon+1);
                for (auto& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                auto begin = value.find_first_not_of(" \t\r\n"), end = value.find_last_not_of(" \t\r\n");
                value = begin == std::string::npos ? "" : value.substr(begin, end-begin+1);
                if (name == "etag") t.response.etag = value;
                if (name == "last-modified") t.response.modified = value;
                if (name == "content-range") t.response.content_range = value;
                if (name == "content-length") t.response.content_length = std::stoull(value);
                if (name == "content-encoding") t.response.content_encoding = value;
            }
            return n;
        } catch (...) { t.error = std::current_exception(); return 0; }
    }

    HttpOptions options_;
    std::vector<CURL*> handles_;
    std::vector<CURL*> free_;
    std::shared_ptr<const Auth> auth_;  // guarded by mutex_
    std::unique_ptr<sftp::Fetcher> sftp_;  // created on the first sftp:// request; guarded by mutex_
    std::mutex mutex_;
    std::condition_variable cv_;
};

}  // namespace vcr
