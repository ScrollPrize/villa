#include "utils/sftp_fetch.hpp"

#include "vc_reader/sftp.hpp"

#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace utils
{
namespace
{
// One process-wide session pool. Its timeouts come from the first request's
// configuration; cancellation follows the HTTP client's global abort.
vcr::sftp::Fetcher& fetcher(const HttpClient::Config& config)
{
    static std::once_flag once;
    static std::unique_ptr<vcr::sftp::Fetcher> shared;
    std::call_once(once, [&] {
        vcr::sftp::FetchOptions options;
        options.session.connectTimeout = std::chrono::seconds(config.connect_timeout.count());
        const int64_t seconds = config.low_speed_time.count() > 0 ? config.low_speed_time.count()
                                : config.transfer_timeout.count() > 0 ? config.transfer_timeout.count()
                                                                      : 30;
        options.session.idleTimeout = std::chrono::milliseconds(std::max<int64_t>(seconds * 1000, 100));
        options.session.cancelled = [] { return HttpClient::isAborted(); };
        options.retries = static_cast<int>(config.max_retries);
        shared = std::make_unique<vcr::sftp::Fetcher>(std::move(options));
    });
    return *shared;
}
}  // namespace

bool is_sftp_url(std::string_view url) noexcept { return vcr::sftp::isSftpUrl(url); }
std::string canonical_sftp_url(std::string_view url) { return vcr::sftp::parseUrl(url).str(); }
std::string sftp_cache_authority(std::string_view url) { return vcr::sftp::cacheAuthority(url); }

HttpResponse fetch_sftp(
    const HttpClient::Config& config, std::string_view url, bool head, std::optional<std::pair<std::size_t, std::size_t>> range, const HttpClient::DownloadObserver& observer)
{
    HttpResponse out;
    vcr::sftp::ByteRange bytes;
    if (head)
        bytes.length = 0;
    else if (range) {
        bytes.offset = range->first;
        bytes.length = range->second;
    }
    try {
        const auto found = fetcher(config).read(
            url, bytes,
            [&] { out = {}; },
            [&](const vcr::sftp::Attributes& attributes, uint64_t first, uint64_t count) {
                out.status_code = range && first > *attributes.size ? 416 : range ? 206 : 200;
                out.content_length = head ? std::size_t(*attributes.size) : std::size_t(count);
                if (!head)
                    out.body.reserve(count);
            },
            [&](std::span<const std::byte> data) {
                out.body.insert(out.body.end(), data.begin(), data.end());
                if (observer)
                    observer(data.size());
            });
        if (!found) {
            out = {};
            out.status_code = 404;
        }
    } catch (const vcr::sftp::StatusError& e) {
        out = {};
        out.status_code = e.status == vcr::sftp::NoSuchFile ? 404 : e.status == vcr::sftp::PermissionDenied ? 403 : 500;
        out.error_message = e.what();
    } catch (const std::exception& e) {
        out = {};
        out.error_message = e.what();
    }
    return out;
}

std::vector<SftpDirectoryEntry> list_sftp_directory(std::string_view url)
{
    std::vector<SftpDirectoryEntry> out;
    for (auto& entry : fetcher({}).list(url))
        out.push_back({std::move(entry.name), std::move(entry.url), entry.directory});
    return out;
}

}  // namespace utils
