#include "vc_reader/sftp.hpp"

#include <algorithm>
#include <cctype>
#include <limits>

namespace vcr::sftp {

struct Fetcher::Pool {
    std::mutex mutex;
    std::condition_variable available;
    std::vector<std::unique_ptr<Session>> idle;
    std::size_t open = 0;
};

Fetcher::Fetcher(FetchOptions options) : options_(std::move(options))
{
    if (options_.sessionsPerHost < 1)
        options_.sessionsPerHost = 1;
}

Fetcher::~Fetcher() = default;

Fetcher::Pool& Fetcher::pool(const Url& url)
{
    std::lock_guard lock(mutex_);
    auto& entry = pools_[url.authority()];
    if (!entry)
        entry = std::make_unique<Pool>();
    return *entry;
}

// Borrows an idle session for the destination, or opens one while fewer than
// sessionsPerHost exist. A session that failed in transport is dropped.
class Fetcher::Lease {
public:
    Lease(Fetcher& fetcher, const Url& url) : pool_(fetcher.pool(url))
    {
        std::unique_lock lock(pool_.mutex);
        pool_.available.wait(lock, [&] {
            return !pool_.idle.empty() || pool_.open < fetcher.options_.sessionsPerHost;
        });
        if (!pool_.idle.empty()) {
            session_ = std::move(pool_.idle.back());
            pool_.idle.pop_back();
            return;
        }
        ++pool_.open;
        lock.unlock();
        try {
            session_ = std::make_unique<Session>(url, fetcher.options_.session);
        } catch (...) {
            lock.lock();
            --pool_.open;
            pool_.available.notify_one();
            throw;
        }
    }
    ~Lease()
    {
        std::lock_guard lock(pool_.mutex);
        if (session_ && healthy_)
            pool_.idle.push_back(std::move(session_));
        else
            --pool_.open;
        pool_.available.notify_one();
    }
    Session& operator*() { return *session_; }
    void broken() { healthy_ = false; }

private:
    Pool& pool_;
    std::unique_ptr<Session> session_;
    bool healthy_ = true;
};

template <class Work>
auto Fetcher::withSession(const Url& url, Work&& work)
{
    for (int attempt = 0;; ++attempt) {
        if (options_.session.cancelled && options_.session.cancelled())
            throw std::runtime_error("SFTP transfer cancelled");
        Lease lease(*this, url);
        try {
            return work(*lease);
        } catch (const StatusError&) {
            throw;  // the server answered: the session stays usable
        } catch (const std::exception&) {
            lease.broken();
            if (attempt >= options_.retries)
                throw;
        }
    }
}

std::optional<Attributes> Fetcher::read(std::string_view location, const ByteRange& range,
                                        const std::function<void()>& reset,
                                        const std::function<void(const Attributes&, uint64_t, uint64_t)>& begin,
                                        const std::function<void(std::span<const std::byte>)>& sink)
{
    const auto url = parseUrl(location);
    return withSession(url, [&](Session& session) -> std::optional<Attributes> {
        reset();
        Attributes attributes;
        try {
            attributes = session.stat(url.path);
        } catch (const StatusError& e) {
            // A missing sparse chunk is a normal answer, not a broken connection.
            if (e.status == NoSuchFile)
                return std::nullopt;
            throw;
        }
        if (attributes.directory())
            throw std::runtime_error("SFTP path is a directory, not a file: " + url.str());
        if (!attributes.size)
            throw std::runtime_error("SFTP server did not report the size of " + url.str());
        const uint64_t size = *attributes.size;
        uint64_t first = range.offset, count = 0;
        if (range.suffix) {
            count = std::min(range.length.value_or(size), size);
            first = size - count;
        } else {
            if (first > size || (range.length && first == size && *range.length))
                throw std::runtime_error("SFTP read starts past the end of " + url.str());
            count = range.length ? std::min(*range.length, size - first) : size - first;
        }
        begin(attributes, first, count);
        if (!count)
            return attributes;
        const auto handle = session.open(url.path, Read);
        uint64_t got = 0;
        try {
            got = session.read(handle, first, count, sink);
        } catch (...) {
            try {
                session.close(handle);
            } catch (...) {
            }
            throw;
        }
        session.close(handle);
        if (got != count)
            throw std::runtime_error("SFTP file ended early: " + url.str());
        return attributes;
    });
}

std::vector<ListedEntry> Fetcher::list(std::string_view location)
{
    const auto url = parseUrl(location);
    return withSession(url, [&](Session& session) {
        std::vector<ListedEntry> result;
        for (auto& entry : session.list(url.path)) {
            auto child = url.withPath(url.path + (url.path.ends_with('/') ? "" : "/") + entry.name);
            auto a = entry.attributes;
            const bool unknownType = !a.permissions || (*a.permissions & 0170000) == 0;
            if (unknownType || a.symlink()) {
                try {
                    a = session.stat(child.path);
                } catch (const StatusError& e) {
                    // A dangling symlink is listed as itself; an untyped entry must resolve.
                    if (unknownType || (e.status != NoSuchFile && e.status != PermissionDenied))
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
    });
}

std::string cacheAuthority(std::string_view location)
{
    static constexpr char digits[] = "0123456789ABCDEF";
    std::string out;
    for (const char c : parseUrl(location).authority()) {
        const auto u = static_cast<unsigned char>(c);
        if (std::isalnum(u) || std::string_view("@.-_~").find(c) != std::string_view::npos) {
            out += c;
        } else {
            out += '%';
            out += digits[u >> 4];
            out += digits[u & 15];
        }
    }
    return out;
}

}  // namespace vcr::sftp
