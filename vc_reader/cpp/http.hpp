// Anonymous HTTP(S) GETs over a bounded pool of libcurl handles.
#pragma once

#include <curl/curl.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
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
    }

    ~HttpClient()
    {
        for (CURL* h : handles_)
            curl_easy_cleanup(h);
    }

    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;

    // GET `url` into `body`. Returns false for 404 (not stored). Retries
    // connection errors, stalls, 429 and 5xx with exponential backoff; any other
    // status throws.
    bool get(const std::string& url, std::vector<std::uint8_t>& body, std::size_t size_hint = 0)
    {
        Lease lease(*this);
        CURL* h = lease.handle;
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
            const CURLcode rc = curl_easy_perform(h);
            long status = 0;
            if (rc == CURLE_OK) {
                curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
                if (status == 200)
                    return true;
                if (status == 404)
                    return false;
                if (status != 429 && status < 500) {
                    std::string snippet(body.begin(), body.begin() + std::min<std::size_t>(body.size(), 200));
                    throw std::runtime_error("GET " + url + ": HTTP " + std::to_string(status) + ": " + snippet);
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
    std::mutex mutex_;
    std::condition_variable cv_;
};

}  // namespace vcr
