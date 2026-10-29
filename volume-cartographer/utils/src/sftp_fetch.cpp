#include "utils/sftp_fetch.hpp"

#include "vc_reader/sftp.hpp"

#include <algorithm>
#include <cctype>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>

namespace utils
{
namespace
{
using vcr::sftp::Session;
using vcr::sftp::StatusError;
using vcr::sftp::Url;

vcr::sftp::Options options(const HttpClient::Config& config)
{
    vcr::sftp::Options out;
    out.connectTimeout = std::chrono::seconds(config.connect_timeout.count());
    const int64_t seconds = config.low_speed_time.count() > 0 ? config.low_speed_time.count()
                            : config.transfer_timeout.count() > 0 ? config.transfer_timeout.count()
                                                                  : 30;
    out.idleTimeout = std::chrono::milliseconds(std::max<int64_t>(seconds * 1000, 100));
    out.cancelled = [] { return HttpClient::isAborted(); };
    return out;
}

HttpResponse fetch(Session& session, const Url& url, bool head, std::optional<std::pair<std::size_t, std::size_t>> range,
                   const HttpClient::DownloadObserver& observer)
{
    vcr::sftp::Attributes attributes;
    try {
        attributes = session.stat(url.path);
    } catch (const StatusError& e) {
        // A missing sparse chunk is a normal response, not a broken
        // connection. Keep the session for subsequent chunk requests.
        if (e.status != vcr::sftp::NoSuchFile && e.status != vcr::sftp::PermissionDenied)
            throw;
        HttpResponse out;
        out.status_code = e.status == vcr::sftp::NoSuchFile ? 404 : 403;
        out.error_message = e.what();
        return out;
    }
    if (!attributes.size)
        throw std::runtime_error("SFTP file has no size");
    if (attributes.directory())
        throw std::runtime_error("SFTP path is a directory, not a file");
    const uint64_t fileSize = *attributes.size;
    if (fileSize > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("SFTP file too large");
    HttpResponse out;
    out.status_code = range ? 206 : 200;
    out.content_length = std::size_t(fileSize);
    if (head)
        return out;
    const uint64_t start = range ? range->first : 0;
    if (start > fileSize) {
        out.status_code = 416;
        return out;
    }
    const uint64_t size = range ? std::min<uint64_t>(range->second, fileSize - start) : fileSize;
    out.content_length = size;
    out.body.reserve(size);
    const auto handle = session.open(url.path, vcr::sftp::Read);
    session.read(handle, start, size, [&](std::span<const std::byte> data) {
        out.body.insert(out.body.end(), data.begin(), data.end());
        if (observer)
            observer(data.size());
    });
    session.close(handle);
    if (out.body.size() != size)
        throw std::runtime_error("SFTP file ended early");
    return out;
}

std::vector<SftpDirectoryEntry> list(Session& session, const Url& url)
{
    std::vector<SftpDirectoryEntry> result;
    for (auto& entry : session.list(url.path)) {
        auto child = url.withPath(url.path + (url.path.ends_with('/') ? "" : "/") + entry.name);
        auto a = entry.attributes;
        const bool unknownType = !a.permissions || (*a.permissions & 0170000) == 0;
        if (unknownType || a.symlink()) {
            try {
                a = session.stat(child.path);
            } catch (const StatusError& e) {
                if (unknownType || (e.status != vcr::sftp::NoSuchFile && e.status != vcr::sftp::PermissionDenied))
                    throw;
            }
        }
        if (!a.permissions || (*a.permissions & 0170000) == 0)
            throw std::runtime_error("SFTP server did not supply an entry type for " + child.path);
        const bool directory = a.directory();
        if (directory)
            child.path += '/';
        result.push_back({entry.name, child.str(), directory});
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.directory != b.directory ? a.directory > b.directory : a.name < b.name;
    });
    return result;
}

// A bounded per-worker connection cache.
struct CachedSession {
    std::string authority;
    std::unique_ptr<Session> session;
};
thread_local std::deque<CachedSession> sessions;
Session& session(const Url& url, const HttpClient::Config& config)
{
    const auto key = url.authority();
    auto it = std::find_if(sessions.begin(), sessions.end(), [&](const auto& s) { return s.authority == key; });
    if (it == sessions.end()) {
        if (sessions.size() == 4)
            sessions.pop_front();
        sessions.push_back({key, std::make_unique<Session>(url, options(config))});
        it = std::prev(sessions.end());
    }
    it->session->setOptions(options(config));
    return *it->session;
}
void discard(const Url& url)
{
    std::erase_if(sessions, [&](const auto& s) { return s.authority == url.authority(); });
}
}  // namespace

bool is_sftp_url(std::string_view url) noexcept { return vcr::sftp::isSftpUrl(url); }
std::string canonical_sftp_url(std::string_view url) { return vcr::sftp::parseUrl(url).str(); }
std::string sftp_cache_authority(std::string_view url)
{
    static constexpr char digits[] = "0123456789ABCDEF";
    std::string out;
    for (const char c : vcr::sftp::parseUrl(url).authority()) {
        const auto u = static_cast<unsigned char>(c);
        if (std::isalnum(u) || std::string_view("@.-_~").find(c) != std::string_view::npos)
            out += c;
        else {
            out += '%';
            out += digits[u >> 4];
            out += digits[u & 15];
        }
    }
    return out;
}
HttpResponse fetch_sftp(
    const HttpClient::Config& config, std::string_view input, bool head, std::optional<std::pair<std::size_t, std::size_t>> range, const HttpClient::DownloadObserver& observer)
{
    const auto url = vcr::sftp::parseUrl(input);
    for (size_t attempt = 0;; ++attempt) {
        try {
            if (HttpClient::isAborted())
                throw std::runtime_error("SFTP transfer cancelled");
            return fetch(session(url, config), url, head, range, observer);
        } catch (const StatusError& e) {
            discard(url);
            HttpResponse out;
            out.status_code = e.status == vcr::sftp::NoSuchFile ? 404 : e.status == vcr::sftp::PermissionDenied ? 403 : 500;
            out.error_message = e.what();
            return out;
        } catch (const std::exception& e) {
            discard(url);
            if (attempt < config.max_retries && !HttpClient::isAborted())
                continue;
            HttpResponse out;
            out.error_message = e.what();
            return out;
        }
    }
}
std::vector<SftpDirectoryEntry> list_sftp_directory(std::string_view input)
{
    const auto url = vcr::sftp::parseUrl(input);
    try {
        return list(session(url, {}), url);
    } catch (...) {
        discard(url);
        throw;
    }
}
}  // namespace utils
