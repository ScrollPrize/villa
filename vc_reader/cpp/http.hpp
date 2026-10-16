// HTTP(S) GETs over a bounded pool of libcurl handles: anonymous, HTTP basic
// auth, or signed for S3 (AWS Signature Version 4).
#pragma once

#include "sigv4.hpp"

#include <curl/curl.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <memory>
#include <mutex>
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
        Lease lease(*this);
        CURL* h = lease.handle;
        std::shared_ptr<const Auth> auth;
        {
            std::lock_guard lock(mutex_);
            auth = auth_;
        }
        double delay = 0.25;
        std::string last_error;
        for (int attempt = 0; attempt <= options_.retries; ++attempt) {
            body.clear();
            if (size_hint)
                body.reserve(size_hint);
            curl_easy_reset(h);
            curl_easy_setopt(h, CURLOPT_URL, url.c_str());
            curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
            curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
            curl_easy_setopt(h, CURLOPT_USERAGENT, "vc_reader");
            curl_easy_setopt(h, CURLOPT_ACCEPT_ENCODING, "");  // .zarray etc. may be served gzip-encoded
            curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT_MS, options_.connect_timeout_ms);
            curl_easy_setopt(h, CURLOPT_LOW_SPEED_LIMIT, 1L);
            curl_easy_setopt(h, CURLOPT_LOW_SPEED_TIME, options_.stall_timeout_s);
            curl_easy_setopt(h, CURLOPT_TCP_KEEPALIVE, 1L);
            curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, &HttpClient::write);
            curl_easy_setopt(h, CURLOPT_WRITEDATA, &body);
            HeaderList headers;
            if (auth->kind == "basic") {
                curl_easy_setopt(h, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
                curl_easy_setopt(h, CURLOPT_USERNAME, auth->user.c_str());
                curl_easy_setopt(h, CURLOPT_PASSWORD, auth->secret.c_str());
            } else if (auth->kind == "s3") {
                // Signed per attempt: the signature covers the request time.
                const S3Key key{auth->user, auth->secret, auth->token, auth->region};
                for (const auto& [name, value] : sigv4_get_headers(url, key, amz_date(std::time(nullptr))))
                    headers.append(name + ": " + value);
                curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers.list);
            }
            const CURLcode rc = curl_easy_perform(h);
            long status = 0;
            if (rc == CURLE_OK) {
                curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
                if (status == 200)
                    return true;
                if (status == 404)
                    return false;
                if (status != 429 && status < 500) {
                    std::string snippet(body.begin(), body.begin() + std::min<std::size_t>(body.size(), 300));
                    const std::string what = "GET " + url + ": HTTP " + std::to_string(status) + ": " + snippet;
                    if (status == 401 || status == 403)
                        throw HttpDenied(what);
                    throw std::runtime_error(what);
                }
                last_error = "HTTP " + std::to_string(status);
            } else {
                last_error = curl_easy_strerror(rc);
            }
            if (attempt < options_.retries) {
                std::this_thread::sleep_for(std::chrono::duration<double>(delay));
                delay = std::min(delay * 2, 8.0);
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

    static std::size_t write(char* data, std::size_t size, std::size_t n, void* user)
    {
        auto* body = static_cast<std::vector<std::uint8_t>*>(user);
        body->insert(body->end(), data, data + size * n);
        return size * n;
    }

    HttpOptions options_;
    std::vector<CURL*> handles_;
    std::vector<CURL*> free_;
    std::shared_ptr<const Auth> auth_;  // guarded by mutex_
    std::mutex mutex_;
    std::condition_variable cv_;
};

}  // namespace vcr
