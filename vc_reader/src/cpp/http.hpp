// HTTP(S) GETs over a bounded pool of libcurl handles: anonymous, HTTP basic
// auth, or signed for S3 (AWS Signature Version 4).
#pragma once

#include "sigv4.hpp"

#include <curl/curl.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <functional>
#include <exception>
#include <cctype>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace vcr {

// How requests authenticate.
struct Auth {
    std::string kind;  // "" (anonymous), "basic" or "s3"
    std::string user;  // basic: user name; s3: access key id
    std::string secret;  // basic: password; s3: secret access key
    std::string token;   // s3: session token (may be empty)
    std::string region;  // s3: the bucket's region
};

// 401/403: the request was not allowed (missing, wrong or expired credentials).
struct HttpDenied : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct HttpOptions {
    int connections = 16;          // concurrent requests (and kept-alive connections)
    long connect_timeout_ms = 10000;
    long stall_timeout_s = 10;     // abort an attempt after this long without receiving data
    int retries = 5;
    Auth auth;
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
        Lease lease(*this);
        CURL* h = lease.handle;
        std::shared_ptr<const Auth> auth;
        { std::lock_guard lock(mutex_); auth = auth_; }
        float delay = 0.25f;
        std::string last_error;
        for (int attempt = 0; attempt <= options_.retries; ++attempt) {
            reset();
            Transfer transfer{write, body_status, begin_body};
            curl_easy_reset(h);
            curl_easy_setopt(h, CURLOPT_URL, url.c_str());
            curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
            curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(h, CURLOPT_USERAGENT, "vc_reader");
            curl_easy_setopt(h, CURLOPT_ACCEPT_ENCODING, "");
            curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT_MS, options_.connect_timeout_ms);
            curl_easy_setopt(h, CURLOPT_LOW_SPEED_LIMIT, 1L);
            curl_easy_setopt(h, CURLOPT_LOW_SPEED_TIME, options_.stall_timeout_s);
            curl_easy_setopt(h, CURLOPT_TCP_KEEPALIVE, 1L);
            curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, &HttpClient::receive);
            curl_easy_setopt(h, CURLOPT_WRITEDATA, &transfer);
            curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, &HttpClient::header);
            curl_easy_setopt(h, CURLOPT_HEADERDATA, &transfer);
            HeaderList headers;
            if (auth->kind == "basic") {
                curl_easy_setopt(h, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
                curl_easy_setopt(h, CURLOPT_USERNAME, auth->user.c_str());
                curl_easy_setopt(h, CURLOPT_PASSWORD, auth->secret.c_str());
            }
            if (auth->kind == "s3") {
                const S3Key key{auth->user, auth->secret, auth->token, auth->region};
                for (const auto& [name, value] : sigv4_get_headers(url, key, amz_date(std::time(nullptr)), extra))
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
                if (status == body_status || status == 304 || status == 404)
                    return transfer.response;
                if (status != 429 && status < 500) {
                    const std::string what = "GET " + url + ": HTTP " + std::to_string(status) + ": " + transfer.snippet;
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
        throw std::runtime_error("GET " + url + " failed after " + std::to_string(options_.retries + 1) +
                                 " attempts: " + last_error);
    }

private:
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
    std::mutex mutex_;
    std::condition_variable cv_;
};

}  // namespace vcr
