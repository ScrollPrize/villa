#include "remote.hpp"

#include <curl/curl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <thread>

#include "helirec/common.hpp"

namespace helirec::remote {

namespace fs = std::filesystem;

namespace {

constexpr std::uintmax_t kMinFreeBytes = std::uintmax_t(2) << 30;   // of a cache's filesystem, to write to it

std::string url_of(const std::string& location) {
    if (location.rfind("s3://", 0) != 0) return location;
    const std::size_t slash = location.find('/', 5);
    const std::string bucket = location.substr(5, slash == std::string::npos ? std::string::npos : slash - 5);
    return "https://" + bucket + ".s3.amazonaws.com" + (slash == std::string::npos ? "/" : location.substr(slash));
}

// <cache_dir>/helirec_zarr/<host>/<path>: an object as it was served; an empty file: the object does not exist
std::string cache_path(const std::string& cache_dir, const std::string& url) {
    std::string rest = url.substr(url.find("://") + 3);
    for (char& c : rest)
        if (c == '?' || c == '#' || c == ':') c = '_';
    return cache_dir + "/helirec_zarr/" + rest;
}

bool read_file(const std::string& path, std::vector<std::uint8_t>& out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(n > 0 ? static_cast<std::size_t>(n) : 0);
    const bool ok = out.empty() || std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    return ok;
}

void write_cache(const std::string& cache_dir, const std::string& path, const std::vector<std::uint8_t>& data) {
    static std::atomic<bool> warned{false};
    std::error_code ec;
    fs::create_directories(fs::path(path).parent_path(), ec);
    const auto space = fs::space(cache_dir, ec);
    if (ec || space.available < kMinFreeBytes + data.size()) {
        if (!warned.exchange(true))
            std::fprintf(stderr, "helirec: little free space in %s: remote data is no longer cached there\n", cache_dir.c_str());
        return;
    }
    // whole or not at all: a temporary beside it, renamed into place
    static std::atomic<unsigned> serial{0};
    const std::string tmp = path + ".tmp" + std::to_string(::getpid()) + "_" + std::to_string(serial++);
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return;
    const bool ok = data.empty() || std::fwrite(data.data(), 1, data.size(), f) == data.size();
    if (std::fclose(f) != 0 || !ok) {
        fs::remove(tmp, ec);
        return;
    }
    fs::rename(tmp, path, ec);
    if (ec) fs::remove(tmp, ec);
}

std::size_t append(char* data, std::size_t size, std::size_t n, void* user) {
    auto* out = static_cast<std::vector<std::uint8_t>*>(user);
    out->insert(out->end(), data, data + size * n);
    return size * n;
}

struct Handle {   // one connection per thread, kept alive between requests
    CURL* curl = nullptr;
    ~Handle() {
        if (curl) curl_easy_cleanup(curl);
    }
};

enum class Result { Found, Absent, Unreachable };

// GET with a few retries of broken transfers and 5xx / 429 answers (a server that cannot be found or connected to is
// not tried again); other answers but 200 and 404 are errors
Result http_get(const std::string& url, std::vector<std::uint8_t>& out, std::string& why) {
    static std::once_flag init;
    std::call_once(init, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    thread_local Handle h;
    if (!h.curl) h.curl = curl_easy_init();
    if (!h.curl) throw Error("cannot start an HTTP client");
    for (int attempt = 0;; ++attempt) {
        out.clear();
        curl_easy_reset(h.curl);
        curl_easy_setopt(h.curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(h.curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(h.curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(h.curl, CURLOPT_CONNECTTIMEOUT, 20L);
        curl_easy_setopt(h.curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);   // stalled for a minute: give up
        curl_easy_setopt(h.curl, CURLOPT_LOW_SPEED_TIME, 60L);
        curl_easy_setopt(h.curl, CURLOPT_USERAGENT, "helirec");
        curl_easy_setopt(h.curl, CURLOPT_WRITEFUNCTION, append);
        curl_easy_setopt(h.curl, CURLOPT_WRITEDATA, &out);
        const CURLcode code = curl_easy_perform(h.curl);
        long status = 0;
        curl_easy_getinfo(h.curl, CURLINFO_RESPONSE_CODE, &status);
        if (code == CURLE_OK && status == 200) return Result::Found;
        if (code == CURLE_OK && status == 404) return Result::Absent;
        const bool again = code != CURLE_OK || status >= 500 || status == 429;
        if (!again) throw Error("HTTP " + std::to_string(status) + " for " + url);
        why = code != CURLE_OK ? curl_easy_strerror(code) : "HTTP " + std::to_string(status);
        if (attempt == 3 || code == CURLE_COULDNT_CONNECT || code == CURLE_COULDNT_RESOLVE_HOST) return Result::Unreachable;
        std::this_thread::sleep_for(std::chrono::milliseconds(250 << attempt));
    }
}

}  // namespace

bool is_remote(const std::string& location) {
    return location.rfind("http://", 0) == 0 || location.rfind("https://", 0) == 0 || location.rfind("s3://", 0) == 0;
}

bool fetch(const std::string& location, std::vector<std::uint8_t>& out, const std::string& cache_dir, bool refresh) {
    const std::string url = url_of(location);
    const std::string cached = cache_dir.empty() ? std::string() : cache_path(cache_dir, url);
    if (!cached.empty() && !refresh && read_file(cached, out)) return !out.empty();
    std::string why;
    const Result r = http_get(url, out, why);
    if (r == Result::Unreachable) {
        if (!cached.empty() && read_file(cached, out)) return !out.empty();
        throw Error("cannot read " + url + ": " + why);
    }
    if (r == Result::Absent) out.clear();
    if (!cached.empty()) write_cache(cache_dir, cached, out);
    return r == Result::Found;
}

}  // namespace helirec::remote
