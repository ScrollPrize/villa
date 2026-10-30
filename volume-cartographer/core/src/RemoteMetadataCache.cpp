#include "vc/core/util/RemoteMetadataCache.hpp"

#include "vc/core/util/RemoteCacheSettings.hpp"
#include "vc/core/util/RemoteFileCache.hpp"

#include <vc_reader/catalog.hpp>

#include <atomic>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <system_error>

namespace vc::core::util
{
namespace
{

constexpr std::string_view kPresentSuffix = ".vc-meta";
constexpr std::string_view kAbsentSuffix = ".vc-absent";

std::atomic<std::uint64_t> gTemporarySerial{0};
// Distinguishes this process's temporary files from another VC3D's.
const std::uint64_t gProcessToken = std::random_device{}();

// The https form the cache is keyed by; empty when the URL is not cacheable.
std::string cacheKey(std::string_view url)
{
    if (url.find_first_of("?#") != std::string_view::npos)
        return {};
    if (!url.starts_with("https://") && !url.starts_with("s3://") && !url.starts_with("s3+"))
        return {};
    auto https = vcr::catalog::resolveOpenDataUrl(std::string(url));
    if (!vcr::catalog::openDataS3Url(https))
        return {};
    while (!https.empty() && https.back() == '/')
        https.pop_back();
    return https;
}

std::filesystem::path withSuffix(const std::filesystem::path& path, std::string_view suffix)
{
    return std::filesystem::path(path.string() + std::string(suffix));
}

std::optional<std::vector<std::byte>> readFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return std::nullopt;
    std::vector<char> chars{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (input.bad())
        return std::nullopt;
    std::vector<std::byte> bytes(chars.size());
    std::memcpy(bytes.data(), chars.data(), chars.size());
    return bytes;
}

// Publish atomically so a concurrent reader (or another VC3D process) sees
// either no answer or the whole one.
void writeFile(const std::filesystem::path& path, const std::vector<std::byte>& bytes)
{
    std::filesystem::create_directories(path.parent_path());
    const auto tmp = std::filesystem::path(path.string() + ".tmp-" + std::to_string(gProcessToken) + "-" +
                                           std::to_string(gTemporarySerial.fetch_add(1)));
    {
        std::ofstream output(tmp, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        output.close();
        if (!output) {
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            return;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec)
        std::filesystem::remove(tmp, ec);
}

}  // namespace

RemoteMetadataCache::RemoteMetadataCache(std::filesystem::path root)
    : root_(std::move(root))
{
}

bool RemoteMetadataCache::cacheable(std::string_view url)
{
    return !cacheKey(url).empty();
}

std::filesystem::path RemoteMetadataCache::diskPath(const std::string& url) const
{
    if (root_.empty())
        return {};
    try {
        return root_ / remoteFileCachePath(url).lexically_relative("remote_sources");
    } catch (const std::invalid_argument&) {
        return {};
    }
}

std::optional<RemoteMetadataCache::Answer> RemoteMetadataCache::lookup(std::string_view url)
{
    const auto key = cacheKey(url);
    if (key.empty())
        return std::nullopt;
    {
        std::lock_guard lock(mutex_);
        if (const auto it = answers_.find(key); it != answers_.end())
            return it->second;
    }
    const auto path = diskPath(key);
    if (path.empty())
        return std::nullopt;
    std::optional<Answer> answer;
    std::error_code ec;
    if (auto bytes = readFile(withSuffix(path, kPresentSuffix)))
        answer = Answer{std::move(*bytes)};
    else if (std::filesystem::is_regular_file(withSuffix(path, kAbsentSuffix), ec))
        answer = Answer{};
    if (answer) {
        std::lock_guard lock(mutex_);
        answers_.insert_or_assign(key, *answer);
    }
    return answer;
}

void RemoteMetadataCache::record(std::string_view url, const Answer& answer)
{
    const auto key = cacheKey(url);
    if (key.empty())
        return;
    {
        std::lock_guard lock(mutex_);
        answers_.insert_or_assign(key, answer);
    }
    const auto path = diskPath(key);
    if (path.empty())
        return;
    // The disk copy only saves later sessions a request; a failure to write
    // it leaves the network as the source of truth.
    try {
        std::error_code ec;
        if (answer) {
            writeFile(withSuffix(path, kPresentSuffix), *answer);
            std::filesystem::remove(withSuffix(path, kAbsentSuffix), ec);
        } else {
            writeFile(withSuffix(path, kAbsentSuffix), {});
            std::filesystem::remove(withSuffix(path, kPresentSuffix), ec);
        }
    } catch (const std::filesystem::filesystem_error&) {
    }
}

void RemoteMetadataCache::invalidate(std::string_view urlPrefix)
{
    const auto prefix = cacheKey(urlPrefix);
    if (prefix.empty())
        return;
    {
        std::lock_guard lock(mutex_);
        std::erase_if(answers_, [&](const auto& entry) {
            const auto& key = entry.first;
            return key == prefix || (key.starts_with(prefix) && key[prefix.size()] == '/');
        });
    }
    const auto path = diskPath(prefix);
    if (path.empty())
        return;
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    std::filesystem::remove(withSuffix(path, kPresentSuffix), ec);
    std::filesystem::remove(withSuffix(path, kAbsentSuffix), ec);
}

RemoteMetadataCache& remoteMetadataCache()
{
    static RemoteMetadataCache cache(vc::settings::remoteCachePath() / "remote_metadata");
    return cache;
}

}  // namespace vc::core::util
